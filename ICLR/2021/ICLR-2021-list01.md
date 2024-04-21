## [0] What Matters for On-Policy Deep Actor-Critic Methods? A Large-Scale Study

**Authors**: *Marcin Andrychowicz, Anton Raichuk, Piotr Stanczyk, Manu Orsini, Sertan Girgin, Raphaël Marinier, Léonard Hussenot, Matthieu Geist, Olivier Pietquin, Marcin Michalski, Sylvain Gelly, Olivier Bachem*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=nIAxjsniDzg](https://openreview.net/forum?id=nIAxjsniDzg)

**Abstract**:

In recent years, reinforcement learning (RL) has been successfully applied to many different continuous control tasks. While RL algorithms are often conceptually simple, their state-of-the-art implementations take numerous low- and high-level design decisions that strongly affect the performance of the resulting agents. Those choices are usually not extensively discussed in the literature, leading to discrepancy between published descriptions of algorithms and their implementations. This makes it hard to attribute progress in RL and slows down overall progress [Engstrom'20]. As a step towards filling that gap, we implement >50 such ``"choices" in a unified on-policy deep actor-critic framework, allowing us to investigate their impact in a large-scale empirical study. We train over 250'000 agents in five continuous control environments of different complexity and provide insights and practical recommendations for the training of on-policy deep actor-critic RL agents.

----

## [1] Theoretical Analysis of Self-Training with Deep Networks on Unlabeled Data

**Authors**: *Colin Wei, Kendrick Shen, Yining Chen, Tengyu Ma*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=rC8sJ4i6kaH](https://openreview.net/forum?id=rC8sJ4i6kaH)

**Abstract**:

Self-training algorithms, which train a model to fit pseudolabels predicted by another previously-learned model, have been very successful for learning with unlabeled data using neural networks. However, the current theoretical understanding of self-training only applies to linear models. This work provides a unified theoretical analysis of self-training with deep networks for semi-supervised learning, unsupervised domain adaptation, and unsupervised learning. At the core of our analysis is a simple but realistic “expansion” assumption, which states that a low-probability subset of the data must expand to a neighborhood with large probability relative to the subset. We also assume that neighborhoods of examples in different classes have minimal overlap. We prove that under these assumptions, the minimizers of population objectives based on self-training and input-consistency regularization will achieve high accuracy with respect to ground-truth labels. By using off-the-shelf generalization bounds, we immediately convert this result to sample complexity guarantees for neural nets that are polynomial in the margin and Lipschitzness. Our results help explain the empirical successes of recently proposed self-training algorithms which use input consistency regularization.

----

## [2] Learning to Reach Goals via Iterated Supervised Learning

**Authors**: *Dibya Ghosh, Abhishek Gupta, Ashwin Reddy, Justin Fu, Coline Manon Devin, Benjamin Eysenbach, Sergey Levine*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=rALA0Xo6yNJ](https://openreview.net/forum?id=rALA0Xo6yNJ)

**Abstract**:

Current reinforcement learning (RL) algorithms can be brittle and difficult to use, especially when learning goal-reaching behaviors from sparse rewards. Although supervised imitation learning provides a simple and stable alternative, it requires access to demonstrations from a human supervisor. In this paper, we study RL algorithms that use imitation learning to acquire goal reaching policies from scratch, without the need for expert demonstrations or a value function. In lieu of demonstrations, we leverage the property that any trajectory is a successful demonstration for reaching the final state in that same trajectory. We propose a simple algorithm in which an agent continually relabels and imitates the trajectories it generates to progressively learn goal-reaching behaviors from scratch. Each iteration, the agent collects new trajectories using the latest policy, and maximizes the likelihood of the actions along these trajectories under the goal that was actually reached, so as to improve the policy. We formally show that this iterated supervised learning procedure optimizes a bound on the RL objective, derive performance bounds of the learned policy, and empirically demonstrate improved goal-reaching performance and robustness over current RL algorithms in several benchmark tasks.

----

## [3] Deep symbolic regression: Recovering mathematical expressions from data via risk-seeking policy gradients

**Authors**: *Brenden K. Petersen, Mikel Landajuela, T. Nathan Mundhenk, Cláudio Prata Santiago, Sookyung Kim, Joanne Taery Kim*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=m5Qsh0kBQG](https://openreview.net/forum?id=m5Qsh0kBQG)

**Abstract**:

Discovering the underlying mathematical expressions describing a dataset is a core challenge for artificial intelligence. This is the problem of $\textit{symbolic regression}$. Despite recent advances in training neural networks to solve complex tasks, deep learning approaches to symbolic regression are underexplored. We propose a framework that leverages deep learning for symbolic regression via a simple idea: use a large model to search the space of small models. Specifically, we use a recurrent neural network to emit a distribution over tractable mathematical expressions and employ a novel risk-seeking policy gradient to train the network to generate better-fitting expressions. Our algorithm outperforms several baseline methods (including Eureqa, the gold standard for symbolic regression) in its ability to exactly recover symbolic expressions on a series of benchmark problems, both with and without added noise. More broadly, our contributions include a framework that can be applied to optimize hierarchical, variable-length objects under a black-box performance metric, with the ability to incorporate constraints in situ, and a risk-seeking policy gradient formulation that optimizes for best-case performance instead of expected performance.

----

## [4] Optimal Rates for Averaged Stochastic Gradient Descent under Neural Tangent Kernel Regime

**Authors**: *Atsushi Nitanda, Taiji Suzuki*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=PULSD5qI2N1](https://openreview.net/forum?id=PULSD5qI2N1)

**Abstract**:

We analyze the convergence of the averaged stochastic gradient descent for overparameterized two-layer neural networks for regression problems. It was recently found that a neural tangent kernel (NTK) plays an important role in showing the global convergence of gradient-based methods under the NTK regime, where the learning dynamics for overparameterized neural networks can be almost characterized by that for the associated reproducing kernel Hilbert space (RKHS). However, there is still room for a convergence rate analysis in the NTK regime. In this study, we show that the averaged stochastic gradient descent can achieve the minimax optimal convergence rate, with the global convergence guarantee, by exploiting the complexities of the target function and the RKHS associated with the NTK. Moreover, we show that the target function specified by the NTK of a ReLU network can be learned at the optimal convergence rate through a smooth approximation of a ReLU network under certain conditions.

----

## [5] Free Lunch for Few-shot Learning: Distribution Calibration

**Authors**: *Shuo Yang, Lu Liu, Min Xu*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=JWOiYxMG92s](https://openreview.net/forum?id=JWOiYxMG92s)

**Abstract**:

Learning from a limited number of samples is challenging since the learned model can easily become overfitted based on the biased distribution formed by only a few training examples. In this paper, we calibrate the distribution of these few-sample classes by transferring statistics from the classes with sufficient examples. Then an adequate number of examples can be sampled from the calibrated distribution to expand the inputs to the classifier. We assume every dimension in the feature representation follows a Gaussian distribution so that the mean and the variance of the distribution can borrow from that of similar classes whose statistics are better estimated with an adequate number of samples. Our method can be built on top of off-the-shelf pretrained feature extractors and classification models without extra parameters. We show that a simple logistic regression classifier trained using the features sampled from our calibrated distribution can outperform the state-of-the-art accuracy on three datasets (~5% improvement on miniImageNet compared to the next best). The visualization of these generated features demonstrates that our calibrated distribution is an accurate estimation.

----

## [6] Scalable Learning and MAP Inference for Nonsymmetric Determinantal Point Processes

**Authors**: *Mike Gartrell, Insu Han, Elvis Dohmatob, Jennifer Gillenwater, Victor-Emmanuel Brunel*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=HajQFbx_yB](https://openreview.net/forum?id=HajQFbx_yB)

**Abstract**:

Determinantal point processes (DPPs) have attracted significant attention in machine learning for their ability to model subsets drawn from a large item collection. Recent work shows that nonsymmetric DPP (NDPP) kernels have significant advantages over symmetric kernels in terms of modeling power and predictive performance. However, for an item collection of size $M$, existing NDPP learning and inference algorithms require memory quadratic in $M$ and runtime cubic (for learning) or quadratic (for inference) in $M$, making them impractical for many typical subset selection tasks. In this work, we develop a learning algorithm with space and time requirements linear in $M$ by introducing a new NDPP kernel decomposition. We also derive a linear-complexity NDPP maximum a posteriori (MAP) inference algorithm that applies not only to our new kernel but also to that of prior work. Through evaluation on real-world datasets, we show that our algorithms scale significantly better, and can match the predictive performance of prior work.

----

## [7] Randomized Automatic Differentiation

**Authors**: *Deniz Oktay, Nick McGreivy, Joshua Aduol, Alex Beatson, Ryan P. Adams*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=xpx9zj7CUlY](https://openreview.net/forum?id=xpx9zj7CUlY)

**Abstract**:

The successes of deep learning, variational inference, and many other fields have been aided by specialized implementations of reverse-mode automatic differentiation (AD) to compute gradients of mega-dimensional objectives. The AD techniques underlying these tools were designed to compute exact gradients to numerical precision, but modern machine learning models are almost always trained with stochastic gradient descent. Why spend computation and memory on exact (minibatch) gradients only to use them for stochastic optimization? We develop a general framework and approach for randomized automatic differentiation (RAD), which can allow unbiased gradient estimates to be computed with reduced memory in return for variance. We examine limitations of the general approach, and argue that we must leverage problem specific structure to realize benefits. We develop RAD techniques for a variety of simple neural network architectures, and show that for a fixed memory budget, RAD converges in fewer iterations than using a small batch size for feedforward networks, and in a similar number for recurrent networks. We also show that RAD can be applied to scientific computing, and use it to develop a low-memory stochastic gradient method for optimizing the control parameters of a linear reaction-diffusion PDE representing a fission reactor.

----

## [8] Learning Generalizable Visual Representations via Interactive Gameplay

**Authors**: *Luca Weihs, Aniruddha Kembhavi, Kiana Ehsani, Sarah M. Pratt, Winson Han, Alvaro Herrasti, Eric Kolve, Dustin Schwenk, Roozbeh Mottaghi, Ali Farhadi*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=UuchYL8wSZo](https://openreview.net/forum?id=UuchYL8wSZo)

**Abstract**:

A growing body of research suggests that embodied gameplay, prevalent not just in human cultures but across a variety of animal species including turtles and ravens, is critical in developing the neural flexibility for creative problem solving, decision making, and socialization. Comparatively little is known regarding the impact of embodied gameplay upon artificial agents. While recent work has produced agents proficient in abstract games, these environments are far removed the real world and thus these agents can provide little insight into the advantages of embodied play. Hiding games, such as hide-and-seek, played universally, provide a rich ground for studying the impact of embodied gameplay on representation learning in the context of perspective taking, secret keeping, and false belief understanding. Here we are the first to show that embodied adversarial reinforcement learning agents playing Cache, a variant of hide-and-seek, in a high fidelity, interactive, environment, learn generalizable representations of their observations encoding information such as object permanence, free space, and containment. Moving closer to biologically motivated learning strategies, our agents' representations, enhanced by intentionality and memory, are developed through interaction and play. These results serve as a model for studying how facets of vision develop through interaction, provide an experimental framework for assessing what is learned by artificial agents, and demonstrates the value of moving from large, static, datasets towards experiential, interactive, representation learning.

----

## [9] Global Convergence of Three-layer Neural Networks in the Mean Field Regime

**Authors**: *Huy Tuan Pham, Phan-Minh Nguyen*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=KvyxFqZS_D](https://openreview.net/forum?id=KvyxFqZS_D)

**Abstract**:

In the mean field regime, neural networks are appropriately scaled so that as the width tends to infinity, the learning dynamics tends to a nonlinear and nontrivial dynamical limit, known as the mean field limit. This lends a way to study large-width neural networks via analyzing the mean field limit. Recent works have successfully applied such analysis to two-layer networks and provided global convergence guarantees. The extension to multilayer ones however has been a highly challenging puzzle, and little is known about the optimization efficiency in the mean field regime when there are more than two layers.

In this work, we prove a global convergence result for unregularized feedforward three-layer networks in the mean field regime. We first develop a rigorous framework to establish the mean field limit of three-layer networks under stochastic gradient descent training. To that end, we propose the idea of a neuronal embedding, which comprises of a fixed probability space that encapsulates neural networks of arbitrary sizes. The identified mean field limit is then used to prove a global convergence guarantee under suitable regularity and convergence mode assumptions, which – unlike previous works on two-layer networks – does not rely critically on convexity. Underlying the result is a universal approximation property, natural of neural networks, which importantly is shown to hold at any finite training time (not necessarily at convergence) via an algebraic topology argument.

----

## [10] Rao-Blackwellizing the Straight-Through Gumbel-Softmax Gradient Estimator

**Authors**: *Max B. Paulus, Chris J. Maddison, Andreas Krause*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=Mk6PZtgAgfq](https://openreview.net/forum?id=Mk6PZtgAgfq)

**Abstract**:

Gradient estimation in models with discrete latent variables is a challenging problem, because the simplest unbiased estimators tend to have high variance. To counteract this, modern estimators either introduce bias, rely on multiple function evaluations, or use learned, input-dependent baselines. Thus, there is a need for estimators that require minimal tuning, are computationally cheap, and have low mean squared error. In this paper, we show that the variance of the straight-through variant of the popular Gumbel-Softmax estimator can be reduced through Rao-Blackwellization without increasing the number of function evaluations. This provably reduces the mean squared error. We empirically demonstrate that this leads to variance reduction, faster convergence, and generally improved performance in two unsupervised latent variable models.

----

## [11] Rethinking Attention with Performers

**Authors**: *Krzysztof Marcin Choromanski, Valerii Likhosherstov, David Dohan, Xingyou Song, Andreea Gane, Tamás Sarlós, Peter Hawkins, Jared Quincy Davis, Afroz Mohiuddin, Lukasz Kaiser, David Benjamin Belanger, Lucy J. Colwell, Adrian Weller*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=Ua6zuk0WRH](https://openreview.net/forum?id=Ua6zuk0WRH)

**Abstract**:

We introduce Performers, Transformer architectures which can estimate regular (softmax) full-rank-attention Transformers with provable accuracy, but using only linear (as opposed to quadratic) space and time complexity, without relying on any priors such as sparsity or low-rankness. To approximate softmax attention-kernels, Performers use a novel Fast Attention Via positive Orthogonal Random features approach (FAVOR+), which may be of independent interest for scalable kernel methods. FAVOR+ can also be used to efficiently model kernelizable attention mechanisms beyond softmax. This representational power is crucial to accurately compare softmax with other kernels for the first time on large-scale tasks, beyond the reach of regular Transformers, and investigate optimal attention-kernels. Performers are linear architectures fully compatible with regular Transformers and with strong theoretical guarantees: unbiased or nearly-unbiased estimation of the attention matrix, uniform convergence and low  estimation variance. We tested Performers on a rich set of tasks stretching from pixel-prediction through text models to protein sequence modeling. We demonstrate competitive results with other examined efficient sparse and dense attention methods, showcasing effectiveness of the novel attention-learning paradigm leveraged by Performers.

----

## [12] Getting a CLUE: A Method for Explaining Uncertainty Estimates

**Authors**: *Javier Antorán, Umang Bhatt, Tameem Adel, Adrian Weller, José Miguel Hernández-Lobato*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=XSLF1XFq5h](https://openreview.net/forum?id=XSLF1XFq5h)

**Abstract**:

Both uncertainty estimation and interpretability are important factors for trustworthy machine learning systems. However, there is little work at the intersection of these two areas. We address this gap by proposing a novel method for interpreting uncertainty estimates from differentiable probabilistic models, like Bayesian Neural Networks (BNNs). Our method, Counterfactual Latent Uncertainty Explanations (CLUE), indicates how to change an input, while keeping it on the data manifold, such that a BNN becomes more confident about the input's prediction. We validate CLUE through 1) a novel framework for evaluating counterfactual explanations of uncertainty, 2) a series of ablation experiments, and 3) a user study. Our experiments show that CLUE outperforms baselines and enables practitioners to better understand which input patterns are responsible for predictive uncertainty.

----

## [13] When Do Curricula Work?

**Authors**: *Xiaoxia Wu, Ethan Dyer, Behnam Neyshabur*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=tW4QEInpni](https://openreview.net/forum?id=tW4QEInpni)

**Abstract**:

Inspired by human learning, researchers have proposed ordering examples during training based on their difficulty. Both curriculum learning, exposing a network to easier examples early in training, and anti-curriculum learning, showing the most difficult examples first, have been suggested as improvements to the standard i.i.d. training. In this work, we set out to investigate the relative benefits of ordered learning. We first investigate the implicit curricula resulting from architectural and optimization bias and find that samples are learned in a highly consistent order. Next, to quantify the benefit of explicit curricula, we conduct extensive experiments over thousands of orderings spanning three kinds of learning: curriculum, anti-curriculum, and random-curriculum -- in which the size of the training dataset is dynamically increased over time, but the examples are randomly ordered. We find that for standard benchmark datasets, curricula have only marginal benefits, and that randomly ordered samples perform as well or better than curricula and anti-curricula, suggesting that any benefit is entirely due to the dynamic training set size. Inspired by common use cases of curriculum learning in practice, we investigate the role of limited training time budget and noisy data in the success of curriculum learning. Our experiments demonstrate that curriculum, but not anti-curriculum or random ordering can indeed improve the performance either with limited training time budget or in the existence of noisy data.

----

## [14] Federated Learning Based on Dynamic Regularization

**Authors**: *Durmus Alp Emre Acar, Yue Zhao, Ramon Matas Navarro, Matthew Mattina, Paul N. Whatmough, Venkatesh Saligrama*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=B7v4QMR6Z9w](https://openreview.net/forum?id=B7v4QMR6Z9w)

**Abstract**:

We propose a novel federated learning method for distributively training neural network models, where the server orchestrates cooperation between a subset of randomly chosen devices in each round. We view Federated Learning problem primarily from a communication perspective and allow more device level computations to save transmission costs. We point out a fundamental dilemma, in that the minima of the local-device level empirical loss are inconsistent with those of the global empirical loss. Different from recent prior works, that either attempt inexact minimization or utilize devices for parallelizing gradient computation, we propose a dynamic regularizer for each device at each round, so that in the limit the global and device solutions are aligned. We demonstrate both through empirical results on real and synthetic data as well as analytical results that our scheme leads to efficient training, in both convex and non-convex settings, while being fully agnostic to device heterogeneity and robust to large number of devices, partial participation and unbalanced data.

----

## [15] Geometry-aware Instance-reweighted Adversarial Training

**Authors**: *Jingfeng Zhang, Jianing Zhu, Gang Niu, Bo Han, Masashi Sugiyama, Mohan S. Kankanhalli*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=iAX0l6Cz8ub](https://openreview.net/forum?id=iAX0l6Cz8ub)

**Abstract**:

In adversarial machine learning, there was a common belief that robustness and accuracy hurt each other. The belief was challenged by recent studies where we can maintain the robustness and improve the accuracy. However, the other direction, whether we can keep the accuracy and improve the robustness, is conceptually and practically more interesting, since robust accuracy should be lower than standard accuracy for any model. In this paper, we show this direction is also promising. Firstly, we find even over-parameterized deep networks may still have insufficient model capacity, because adversarial training has an overwhelming smoothing effect. Secondly, given limited model capacity, we argue adversarial data should have unequal importance: geometrically speaking, a natural data point closer to/farther from the class boundary is less/more robust, and the corresponding adversarial data point should be assigned with larger/smaller weight. Finally, to implement the idea, we propose geometry-aware instance-reweighted adversarial training, where the weights are based on how difficult it is to attack a natural data point. Experiments show that our proposal boosts the robustness of standard adversarial training; combining two directions, we improve both robustness and accuracy of standard adversarial training.

----

## [16] Co-Mixup: Saliency Guided Joint Mixup with Supermodular Diversity

**Authors**: *Jang-Hyun Kim, Wonho Choo, Hosan Jeong, Hyun Oh Song*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=gvxJzw8kW4b](https://openreview.net/forum?id=gvxJzw8kW4b)

**Abstract**:

While deep neural networks show great performance on fitting to the training distribution, improving the networks' generalization performance to the test distribution and robustness to the sensitivity to input perturbations still remain as a challenge. Although a number of mixup based augmentation strategies have been proposed to partially address them, it remains unclear as to how to best utilize the supervisory signal within each input data for mixup from the optimization perspective. We propose a new perspective on batch mixup and formulate the optimal construction of a batch of mixup data maximizing the data saliency measure of each individual mixup data and encouraging the supermodular diversity among the constructed mixup data. This leads to a novel discrete optimization problem minimizing the difference between submodular functions. We also propose an efficient modular approximation based iterative submodular minimization algorithm for efficient mixup computation per each minibatch suitable for minibatch based neural network training. Our experiments show the proposed method achieves the state of the art generalization, calibration, and weakly supervised localization results compared to other mixup methods. The source code is available at https://github.com/snu-mllab/Co-Mixup.

----

## [17] SenSeI: Sensitive Set Invariance for Enforcing Individual Fairness

**Authors**: *Mikhail Yurochkin, Yuekai Sun*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=DktZb97_Fx](https://openreview.net/forum?id=DktZb97_Fx)

**Abstract**:

In this paper, we cast fair machine learning as invariant machine learning. We first formulate a version of individual fairness that enforces invariance on certain sensitive sets. We then design a transport-based regularizer that enforces this version of individual fairness and develop an algorithm to minimize the regularizer efficiently. Our theoretical results guarantee the proposed approach trains certifiably fair ML models. Finally, in the experimental studies we demonstrate improved fairness metrics in comparison to several recent fair training procedures on three ML tasks that are susceptible to algorithmic bias.

----

## [18] End-to-end Adversarial Text-to-Speech

**Authors**: *Jeff Donahue, Sander Dieleman, Mikolaj Binkowski, Erich Elsen, Karen Simonyan*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=rsf1z-JSj87](https://openreview.net/forum?id=rsf1z-JSj87)

**Abstract**:

Modern text-to-speech synthesis pipelines typically involve multiple processing stages, each of which is designed or learnt independently from the rest. In this work, we take on the challenging task of learning to synthesise speech from normalised text or phonemes in an end-to-end manner, resulting in models which operate directly on character or phoneme input sequences and produce raw speech audio outputs. Our proposed generator is feed-forward and thus efficient for both training and inference, using a differentiable alignment scheme based on token length prediction. It learns to produce high fidelity audio through a combination of adversarial feedback and prediction losses constraining the generated audio to roughly match the ground truth in terms of its total duration and mel-spectrogram. To allow the model to capture temporal variation in the generated audio, we employ soft dynamic time warping in the spectrogram-based prediction loss. The resulting model achieves a mean opinion score exceeding 4 on a 5 point scale, which is comparable to the state-of-the-art models relying on multi-stage training and additional supervision.

----

## [19] Dataset Condensation with Gradient Matching

**Authors**: *Bo Zhao, Konda Reddy Mopuri, Hakan Bilen*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=mSAKhLYLSsl](https://openreview.net/forum?id=mSAKhLYLSsl)

**Abstract**:

As the state-of-the-art machine learning methods in many fields rely on larger datasets, storing datasets and training models on them become significantly more expensive. This paper proposes a training set synthesis technique for data-efficient learning, called Dataset Condensation, that learns to condense large dataset into a small set of informative synthetic samples for training deep neural networks from scratch. We formulate this goal as a gradient matching problem between the gradients of deep neural network weights that are trained on the original and our synthetic data. We rigorously evaluate its performance in several computer vision benchmarks and demonstrate that it significantly outperforms the state-of-the-art methods. Finally we explore the use of our method in continual learning and neural architecture search and report promising gains when limited memory and computations are available.

----

## [20] Rethinking Architecture Selection in Differentiable NAS

**Authors**: *Ruochen Wang, Minhao Cheng, Xiangning Chen, Xiaocheng Tang, Cho-Jui Hsieh*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=PKubaeJkw3](https://openreview.net/forum?id=PKubaeJkw3)

**Abstract**:

Differentiable Neural Architecture Search is one of the most popular Neural Architecture Search (NAS) methods for its search efficiency and simplicity, accomplished by jointly optimizing the model weight and architecture parameters in a weight-sharing supernet via gradient-based algorithms. At the end of the search phase, the operations with the largest architecture parameters will be selected to form the final architecture, with the implicit assumption that the values of architecture parameters reflect the operation strength. While much has been discussed about the supernet's optimization, the architecture selection process has received little attention. We provide empirical and theoretical analysis to show that the magnitude of architecture parameters does not necessarily indicate how much the operation contributes to the supernet's performance. We propose an alternative perturbation-based architecture selection that directly measures each operation's influence on the supernet. We re-evaluate several differentiable NAS methods with the proposed architecture selection and find that it is able to extract significantly improved architectures from the underlying supernets consistently. Furthermore, we find that several failure modes of DARTS can be greatly alleviated with the proposed selection method, indicating that much of the poor generalization observed in DARTS can be attributed to the failure of magnitude-based architecture selection rather than entirely the optimization of its supernet.

----

## [21] A Distributional Approach to Controlled Text Generation

**Authors**: *Muhammad Khalifa, Hady Elsahar, Marc Dymetman*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=jWkw45-9AbL](https://openreview.net/forum?id=jWkw45-9AbL)

**Abstract**:

We propose a  Distributional  Approach for addressing  Controlled  Text  Generation from pre-trained Language Models (LM). This approach permits to specify, in a single formal framework, both “pointwise’” and “distributional” constraints over the target LM — to our knowledge, the first model with such generality —while minimizing KL divergence from the initial LM distribution.  The optimal target distribution is then uniquely determined as an explicit EBM (Energy-BasedModel) representation. From that optimal representation, we then train a target controlled Autoregressive LM through an adaptive distributional variant of PolicyGradient.  We conduct a first set of experiments over pointwise constraints showing the advantages of our approach over a set of baselines, in terms of obtaining a controlled LM balancing constraint satisfaction with divergence from the pretrained LM.  We then perform experiments over distributional constraints, a unique feature of our approach, demonstrating its potential as a remedy to the problem of Bias in Language Models.  Through an ablation study, we show the effectiveness of our adaptive technique for obtaining faster convergence.
Code available at https://github.com/naver/gdc

----

## [22] Learning Cross-Domain Correspondence for Control with Dynamics Cycle-Consistency

**Authors**: *Qiang Zhang, Tete Xiao, Alexei A. Efros, Lerrel Pinto, Xiaolong Wang*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=QIRlze3I6hX](https://openreview.net/forum?id=QIRlze3I6hX)

**Abstract**:

At the heart of many robotics problems is the challenge of learning correspondences across domains. For instance, imitation learning requires obtaining correspondence between humans and robots; sim-to-real requires correspondence between physics simulators and real hardware; transfer learning requires correspondences between different robot environments. In this paper, we propose to learn correspondence across such domains emphasizing on differing modalities (vision and internal state), physics parameters (mass and friction), and morphologies (number of limbs). Importantly, correspondences are learned using unpaired and randomly collected data from the two domains. We propose dynamics cycles that align dynamic robotic behavior across two domains using a cycle consistency constraint. Once this correspondence is found, we can directly transfer the policy trained on one domain to the other, without needing any additional fine-tuning on the second domain. We perform experiments across a variety of problem domains, both in simulation and on real robots. Our framework is able to align uncalibrated monocular video of a real robot arm to dynamic state-action trajectories of a simulated arm without paired data. Video demonstrations of our results are available at: https://sites.google.com/view/cycledynamics .

----

## [23] Human-Level Performance in No-Press Diplomacy via Equilibrium Search

**Authors**: *Jonathan Gray, Adam Lerer, Anton Bakhtin, Noam Brown*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=0-uUGPbIjD](https://openreview.net/forum?id=0-uUGPbIjD)

**Abstract**:

Prior AI breakthroughs in complex games have focused on either the purely adversarial or purely cooperative settings. In contrast, Diplomacy is a game of shifting alliances that involves both cooperation and competition. For this reason, Diplomacy has proven to be a formidable research challenge. In this paper we describe an agent for the no-press variant of Diplomacy that combines supervised learning on human data with one-step lookahead search via regret minimization. Regret minimization techniques have been behind previous AI successes in adversarial games, most notably poker, but have not previously been shown to be successful in large-scale games involving cooperation. We show that our agent greatly exceeds the performance of past no-press Diplomacy bots, is unexploitable by expert humans, and ranks in the top 2% of human players when playing anonymous games on a popular Diplomacy website.

----

## [24] Parrot: Data-Driven Behavioral Priors for Reinforcement Learning

**Authors**: *Avi Singh, Huihan Liu, Gaoyue Zhou, Albert Yu, Nicholas Rhinehart, Sergey Levine*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=Ysuv-WOFeKR](https://openreview.net/forum?id=Ysuv-WOFeKR)

**Abstract**:

Reinforcement learning provides a general framework for flexible decision making and control, but requires extensive data collection for each new task that an agent needs to learn. In other machine learning fields, such as natural language processing or computer vision, pre-training on large, previously collected datasets to bootstrap learning for new tasks has emerged as a powerful paradigm to reduce data requirements when learning a new task. In this paper, we ask the following question: how can we enable similarly useful pre-training for RL agents? We propose a method for pre-training behavioral priors that can capture complex input-output relationships observed in successful trials from a wide range of previously seen tasks, and we show how this learned prior can be used for rapidly learning new tasks without impeding the RL agent's ability to try out novel behaviors. We demonstrate the effectiveness of our approach in challenging robotic manipulation domains involving image observations and sparse reward functions, where our method outperforms prior works by a substantial margin. Additional materials can be found on our project website: https://sites.google.com/view/parrot-rl

----

## [25] Learning Invariant Representations for Reinforcement Learning without Reconstruction

**Authors**: *Amy Zhang, Rowan Thomas McAllister, Roberto Calandra, Yarin Gal, Sergey Levine*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=-2FCwDKRREu](https://openreview.net/forum?id=-2FCwDKRREu)

**Abstract**:

We study how representation learning can accelerate reinforcement learning from rich observations, such as images, without relying either on domain knowledge or pixel-reconstruction. Our goal is to learn representations that provide for effective downstream control and invariance to task-irrelevant details. Bisimulation metrics quantify behavioral similarity between states in continuous MDPs, which we propose using to learn robust latent representations which encode only the task-relevant information from observations. Our method trains encoders such that distances in latent space equal bisimulation distances in state space. We demonstrate the effectiveness of our method at disregarding task-irrelevant information using modified visual MuJoCo tasks, where the background is replaced with moving distractors and natural videos, while achieving SOTA performance. We also test a first-person highway driving task where our method learns invariance to clouds, weather, and time of day. Finally, we provide generalization results drawn from properties of bisimulation metrics, and links to causal inference.

----

## [26] Do 2D GANs Know 3D Shape? Unsupervised 3D Shape Reconstruction from 2D Image GANs

**Authors**: *Xingang Pan, Bo Dai, Ziwei Liu, Chen Change Loy, Ping Luo*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=FGqiDsBUKL0](https://openreview.net/forum?id=FGqiDsBUKL0)

**Abstract**:

Natural images are projections of 3D objects on a 2D image plane. While state-of-the-art 2D generative models like GANs show unprecedented quality in modeling the natural image manifold, it is unclear whether they implicitly capture the underlying 3D object structures. And if so, how could we exploit such knowledge to recover the 3D shapes of objects in the images? To answer these questions, in this work, we present the first attempt to directly mine 3D geometric cues from an off-the-shelf 2D GAN that is trained on RGB images only. Through our investigation, we found that such a pre-trained GAN indeed contains rich 3D knowledge and thus can be used to recover 3D shape from a single 2D image in an unsupervised manner. The core of our framework is an iterative strategy that explores and exploits diverse viewpoint and lighting variations in the GAN image manifold. The framework does not require 2D keypoint or 3D annotations, or strong assumptions on object shapes (e.g. shapes are symmetric), yet it successfully recovers 3D shapes with high precision for human faces, cats, cars, and buildings. The recovered 3D shapes immediately allow high-quality image editing like relighting and object rotation. We quantitatively demonstrate the effectiveness of our approach compared to previous methods in both 3D shape reconstruction and face rotation. Our code is available at https://github.com/XingangPan/GAN2Shape.

----

## [27] VCNet and Functional Targeted Regularization For Learning Causal Effects of Continuous Treatments

**Authors**: *Lizhen Nie, Mao Ye, Qiang Liu, Dan Nicolae*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=RmB-88r9dL](https://openreview.net/forum?id=RmB-88r9dL)

**Abstract**:

Motivated by the rising abundance of observational data with continuous treatments, we investigate the problem of estimating the average dose-response curve (ADRF). Available parametric methods are limited in their model space, and previous attempts in leveraging neural network to enhance model expressiveness relied on partitioning continuous treatment into blocks and using separate heads for each block; this however produces in practice discontinuous ADRFs. Therefore, the question of how to adapt the structure and training of neural network to estimate ADRFs remains open. This paper makes two important contributions. First, we propose a novel varying coefficient neural network (VCNet) that improves model expressiveness while preserving continuity of the estimated ADRF. Second, to improve finite sample performance, we generalize targeted regularization to obtain a doubly robust estimator of the whole ADRF curve.

----

## [28] Rethinking the Role of Gradient-based Attribution Methods for Model Interpretability

**Authors**: *Suraj Srinivas, François Fleuret*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=dYeAHXnpWJ4](https://openreview.net/forum?id=dYeAHXnpWJ4)

**Abstract**:

Current methods for the interpretability of discriminative deep neural networks commonly rely on the model's input-gradients, i.e., the gradients of the output logits w.r.t. the inputs. The common assumption is that these input-gradients contain information regarding $p_{\theta} ( y\mid \mathbf{x} )$, the model's discriminative capabilities, thus justifying their use for interpretability. However, in this work, we show that these input-gradients can be arbitrarily manipulated as a consequence of the shift-invariance of softmax without changing the discriminative function. This leaves an open question: given that input-gradients can be arbitrary, why are they highly structured and explanatory in standard models?

In this work, we re-interpret the logits of standard softmax-based classifiers as unnormalized log-densities of the data distribution and show that input-gradients can be viewed as gradients of a class-conditional generative model $p_{\theta}(\mathbf{x} \mid y)$ implicit in the discriminative model. This leads us to hypothesize that the highly structured and explanatory nature of input-gradients may be due to the alignment of this class-conditional model $p_{\theta}(\mathbf{x} \mid y)$ with that of the ground truth data distribution $p_{\text{data}} (\mathbf{x} \mid y)$. We test this hypothesis by studying the effect of density alignment on gradient explanations. To achieve this density alignment, we use an algorithm called score-matching, and propose novel approximations to this algorithm to enable training large-scale models.

Our experiments show that improving the alignment of the implicit density model with the data distribution enhances gradient structure and explanatory power while reducing this alignment has the opposite effect. This also leads us to conjecture that unintended density alignment in standard neural network training may explain the highly structured nature of input-gradients observed in practice. Overall, our finding that input-gradients capture information regarding an implicit generative model implies that we need to re-think their use for interpreting discriminative models.

----

## [29] Neural Synthesis of Binaural Speech From Mono Audio

**Authors**: *Alexander Richard, Dejan Markovic, Israel D. Gebru, Steven Krenn, Gladstone Alexander Butler, Fernando De la Torre, Yaser Sheikh*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=uAX8q61EVRu](https://openreview.net/forum?id=uAX8q61EVRu)

**Abstract**:

We present a neural rendering approach for binaural sound synthesis that can produce realistic and spatially accurate binaural sound in realtime. The network takes, as input, a single-channel audio source and synthesizes, as output, two-channel binaural sound, conditioned on the relative position and orientation of the listener with respect to the source. We investigate deficiencies of the l2-loss on raw waveforms in a theoretical analysis and introduce an improved loss that overcomes these limitations. In an empirical evaluation, we establish that our approach is the first to generate spatially accurate waveform outputs (as measured by real recordings) and outperforms existing approaches by a considerable margin, both quantitatively and in a perceptual study. Dataset and code are available online.

----

## [30] DiffWave: A Versatile Diffusion Model for Audio Synthesis

**Authors**: *Zhifeng Kong, Wei Ping, Jiaji Huang, Kexin Zhao, Bryan Catanzaro*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=a-xFK8Ymz5J](https://openreview.net/forum?id=a-xFK8Ymz5J)

**Abstract**:

In this work, we propose DiffWave, a versatile diffusion probabilistic model for conditional and unconditional waveform generation. The model is non-autoregressive, and converts the white noise signal into structured waveform through a Markov chain with a constant number of steps at synthesis. It is efficiently trained by optimizing a variant of variational bound on the data likelihood. DiffWave produces high-fidelity audios in different waveform generation tasks, including neural vocoding conditioned on mel spectrogram, class-conditional generation, and unconditional generation. We demonstrate that DiffWave matches a strong WaveNet vocoder in terms of speech quality (MOS: 4.44 versus 4.43), while synthesizing orders of magnitude faster. In particular, it significantly outperforms autoregressive and GAN-based waveform models in the challenging unconditional generation task in terms of audio quality and sample diversity from various automatic and human evaluations.

----

## [31] An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale

**Authors**: *Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=YicbFdNTTy](https://openreview.net/forum?id=YicbFdNTTy)

**Abstract**:

While the Transformer architecture has become the de-facto standard for natural language processing tasks, its applications to computer vision remain limited. In vision, attention is either applied in conjunction with convolutional networks, or used to replace certain components of convolutional networks while keeping their overall structure in place. We show that this reliance on CNNs is not necessary and a pure transformer applied directly to sequences of image patches can perform very well on image classification tasks. When pre-trained on large amounts of data and transferred to multiple mid-sized or small image recognition benchmarks (ImageNet, CIFAR-100, VTAB, etc.), Vision Transformer (ViT) attains excellent results compared to state-of-the-art convolutional networks while requiring substantially fewer computational resources to train.

----

## [32] On the mapping between Hopfield networks and Restricted Boltzmann Machines

**Authors**: *Matthew Smart, Anton Zilman*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=RGJbergVIoO](https://openreview.net/forum?id=RGJbergVIoO)

**Abstract**:

Hopfield networks (HNs) and Restricted Boltzmann Machines (RBMs) are two important models at the interface of statistical physics, machine learning, and neuroscience. Recently, there has been interest in the relationship between HNs and RBMs, due to their similarity under the statistical mechanics formalism. An exact mapping between HNs and RBMs has been previously noted for the special case of orthogonal (“uncorrelated”) encoded patterns. We present here an exact mapping in the case of correlated pattern HNs, which are more broadly applicable to existing datasets. Specifically, we show that any HN with $N$ binary variables and $p<N$ potentially correlated binary patterns can be transformed into an RBM with $N$ binary visible variables and $p$ gaussian hidden variables. We outline the conditions under which the reverse mapping exists, and conduct experiments on the MNIST dataset which suggest the mapping provides a useful initialization to the RBM weights. We discuss extensions, the potential importance of this correspondence for the training of RBMs, and for understanding the performance of feature extraction methods which utilize RBMs.

----

## [33] SMiRL: Surprise Minimizing Reinforcement Learning in Unstable Environments

**Authors**: *Glen Berseth, Daniel Geng, Coline Manon Devin, Nicholas Rhinehart, Chelsea Finn, Dinesh Jayaraman, Sergey Levine*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=cPZOyoDloxl](https://openreview.net/forum?id=cPZOyoDloxl)

**Abstract**:

Every living organism struggles against disruptive environmental forces to carve out and maintain an orderly niche. We propose that such a struggle to achieve and preserve order might offer a principle for the emergence of useful behaviors in artificial agents. We formalize this idea into an unsupervised reinforcement learning method called surprise minimizing reinforcement learning (SMiRL). SMiRL alternates between learning a density model to evaluate the surprise of a stimulus, and improving the policy to seek more predictable stimuli. The policy seeks out stable and repeatable situations that counteract the environment's prevailing sources of entropy. This might include avoiding other hostile agents, or finding a stable, balanced pose for a bipedal robot in the face of disturbance forces. We demonstrate that our surprise minimizing agents can successfully play Tetris, Doom, control a humanoid to avoid falls, and navigate to escape enemies in a maze without any task-specific reward supervision. We further show that SMiRL can be used together with standard task rewards to accelerate reward-driven learning.

----

## [34] Evolving Reinforcement Learning Algorithms

**Authors**: *John D. Co-Reyes, Yingjie Miao, Daiyi Peng, Esteban Real, Quoc V. Le, Sergey Levine, Honglak Lee, Aleksandra Faust*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=0XXpJ4OtjW](https://openreview.net/forum?id=0XXpJ4OtjW)

**Abstract**:

We propose a method for meta-learning reinforcement learning algorithms by searching over the space of computational graphs which compute the loss function for a value-based model-free RL agent to optimize. The learned algorithms are domain-agnostic and can generalize to new environments not seen during training. Our method can both learn from scratch and bootstrap off known existing algorithms, like DQN, enabling interpretable modifications which improve performance. Learning from scratch on simple classical control and gridworld tasks, our method rediscovers the temporal-difference (TD) algorithm. Bootstrapped from DQN, we highlight two learned algorithms which obtain good generalization performance over other classical control tasks, gridworld type tasks, and Atari games. The analysis of the learned algorithm behavior shows resemblance to recently proposed RL algorithms that address overestimation in value-based methods.

----

## [35] Growing Efficient Deep Networks by Structured Continuous Sparsification

**Authors**: *Xin Yuan, Pedro Henrique Pamplona Savarese, Michael Maire*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=wb3wxCObbRT](https://openreview.net/forum?id=wb3wxCObbRT)

**Abstract**:

We develop an approach to growing deep network architectures over the course of training, driven by a principled combination of accuracy and sparsity objectives.  Unlike existing pruning or architecture search techniques that operate on full-sized models or supernet architectures, our method can start from a small, simple seed architecture and dynamically grow and prune both layers and filters.  By combining a continuous relaxation of discrete network structure optimization with a scheme for sampling sparse subnetworks, we produce compact, pruned networks, while also drastically reducing the computational expense of training.  For example, we achieve $49.7\%$ inference FLOPs and $47.4\%$ training FLOPs savings compared to a baseline ResNet-50 on ImageNet, while maintaining $75.2\%$ top-1 validation accuracy --- all without any dedicated fine-tuning stage.  Experiments across CIFAR, ImageNet, PASCAL VOC, and Penn Treebank, with convolutional networks for image classification and semantic segmentation, and recurrent networks for language modeling, demonstrate that we both train faster and produce more efficient networks than competing architecture pruning or search methods.

----

## [36] Deformable DETR: Deformable Transformers for End-to-End Object Detection

**Authors**: *Xizhou Zhu, Weijie Su, Lewei Lu, Bin Li, Xiaogang Wang, Jifeng Dai*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=gZ9hCDWe6ke](https://openreview.net/forum?id=gZ9hCDWe6ke)

**Abstract**:

DETR has been recently proposed to eliminate the need for many hand-designed components in object detection while demonstrating good performance. However, it suffers from slow convergence and limited feature spatial resolution, due to the limitation of Transformer attention modules in processing image feature maps. To mitigate these issues, we proposed Deformable DETR, whose attention modules only attend to a small set of key sampling points around a reference. Deformable DETR can achieve better performance than DETR (especially on small objects) with 10$\times$ less training epochs. Extensive experiments on the COCO benchmark demonstrate the effectiveness of our approach. Code is released at https://github.com/fundamentalvision/Deformable-DETR.

----

## [37] EigenGame: PCA as a Nash Equilibrium

**Authors**: *Ian Gemp, Brian McWilliams, Claire Vernade, Thore Graepel*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=NzTU59SYbNq](https://openreview.net/forum?id=NzTU59SYbNq)

**Abstract**:

We present a novel view on principal components analysis as a competitive game in which each approximate eigenvector is controlled by a player whose goal is to maximize their own utility function. We analyze the properties of this PCA game and the behavior of its gradient based updates. The resulting algorithm---which combines elements from Oja's rule with a  generalized Gram-Schmidt orthogonalization---is naturally decentralized and hence parallelizable through message passing. We demonstrate the scalability of the algorithm with experiments on large image datasets and neural network activations. We discuss how this new view of PCA as a differentiable game can lead to further algorithmic developments and insights.

----

## [38] Augmenting Physical Models with Deep Networks for Complex Dynamics Forecasting

**Authors**: *Yuan Yin, Vincent Le Guen, Jérémie Donà, Emmanuel de Bézenac, Ibrahim Ayed, Nicolas Thome, Patrick Gallinari*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=kmG8vRXTFv](https://openreview.net/forum?id=kmG8vRXTFv)

**Abstract**:

Forecasting complex dynamical phenomena in settings where only partial knowledge of their dynamics is available is a prevalent problem across various scientific fields. While purely data-driven approaches are arguably insufficient in this context, standard physical modeling based approaches tend to be over-simplistic, inducing non-negligible errors. In this work, we introduce the APHYNITY framework, a principled approach for augmenting incomplete physical dynamics described by differential equations with deep data-driven models. It consists in decomposing the dynamics into two components: a physical component accounting for the dynamics for which we have some prior knowledge, and a data-driven component accounting for errors of the physical model. The learning problem is carefully formulated such that the physical model explains as much of the data as possible, while the data-driven component only describes information that cannot be captured by the physical model, no more, no less. This not only provides the existence and uniqueness for this decomposition, but also ensures interpretability and benefits generalization. Experiments made on three important use cases, each representative of a different family of phenomena, i.e. reaction-diffusion equations, wave equations and the non-linear damped pendulum, show that APHYNITY can efficiently leverage approximate physical models to accurately forecast the evolution of the system and correctly identify relevant physical parameters.

----

## [39] Complex Query Answering with Neural Link Predictors

**Authors**: *Erik Arakelyan, Daniel Daza, Pasquale Minervini, Michael Cochez*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=Mos9F9kDwkz](https://openreview.net/forum?id=Mos9F9kDwkz)

**Abstract**:

Neural link predictors are immensely useful for identifying missing edges in large scale Knowledge Graphs. However, it is still not clear how to use these models for answering more complex queries that arise in a number of domains, such as queries using logical conjunctions ($\land$), disjunctions ($\lor$) and existential quantifiers ($\exists$), while accounting for missing edges. In this work, we propose a framework for efficiently answering complex queries on incomplete Knowledge Graphs. We translate each query into an end-to-end differentiable objective, where the truth value of each atom is computed by a pre-trained neural link predictor. We then analyse two solutions to the optimisation problem, including gradient-based and combinatorial search. In our experiments, the proposed approach produces more accurate results than state-of-the-art methods --- black-box neural models trained on millions of generated queries --- without the need of training on a large and diverse set of complex queries. Using orders of magnitude less training data, we obtain relative improvements ranging from 8% up to 40% in Hits@3 across different knowledge graphs containing factual information. Finally, we demonstrate that it is possible to explain the outcome of our model in terms of the intermediate solutions identified for each of the complex query atoms. All our source code and datasets are available online, at https://github.com/uclnlp/cqd.

----

## [40] Towards Nonlinear Disentanglement in Natural Data with Temporal Sparse Coding

**Authors**: *David A. Klindt, Lukas Schott, Yash Sharma, Ivan Ustyuzhaninov, Wieland Brendel, Matthias Bethge, Dylan M. Paiton*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=EbIDjBynYJ8](https://openreview.net/forum?id=EbIDjBynYJ8)

**Abstract**:

Disentangling the underlying generative factors from complex data has so far been limited to carefully constructed scenarios. We propose a path towards natural data by first showing that the statistics of natural data provide enough structure to enable disentanglement, both theoretically and empirically. Specifically, we provide evidence that objects in natural movies undergo transitions that are typically small in magnitude with occasional large jumps, which is characteristic of a temporally sparse distribution. To address this finding we provide a novel proof that relies on a sparse prior on temporally adjacent observations to recover the true latent variables up to permutations and sign flips, directly providing a stronger result than previous work. We show that equipping practical estimation methods with our prior often surpasses the current state-of-the-art on several established benchmark datasets without any impractical assumptions, such as knowledge of the number of changing generative factors. Furthermore, we contribute two new benchmarks, Natural Sprites and KITTI Masks, which integrate the measured natural dynamics to enable disentanglement evaluation with more realistic datasets. We leverage these benchmarks to test our theory, demonstrating improved performance. We also identify non-obvious challenges for current methods in scaling to more natural domains. Taken together our work addresses key issues in disentanglement research for moving towards more natural settings.

----

## [41] Self-training For Few-shot Transfer Across Extreme Task Differences

**Authors**: *Cheng Perng Phoo, Bharath Hariharan*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=O3Y56aqpChA](https://openreview.net/forum?id=O3Y56aqpChA)

**Abstract**:

Most few-shot learning techniques are pre-trained on a large, labeled “base dataset”. In problem domains where such large labeled datasets are not available for pre-training (e.g., X-ray, satellite images), one must resort to pre-training in a different “source” problem domain (e.g., ImageNet), which can be very different from the desired target task. Traditional few-shot and transfer learning techniques fail in the presence of such extreme differences between the source and target tasks. In this paper, we present a simple and effective solution to tackle this extreme domain gap: self-training a source domain representation on unlabeled data from the target domain. We show that this improves one-shot performance on the target domain by 2.9 points on average on the challenging BSCD-FSL benchmark consisting of datasets from multiple domains.

----

## [42] Score-Based Generative Modeling through Stochastic Differential Equations

**Authors**: *Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, Ben Poole*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=PxTIG12RRHS](https://openreview.net/forum?id=PxTIG12RRHS)

**Abstract**:

Creating noise from data is easy; creating data from noise is generative modeling. We present a stochastic differential equation (SDE) that smoothly transforms a complex data distribution to a known prior distribution by slowly injecting noise, and a corresponding reverse-time SDE that transforms the prior distribution back into the data distribution by slowly removing the noise. 
Crucially, the reverse-time SDE depends only on the time-dependent gradient field (a.k.a., score) of the perturbed data distribution. By leveraging advances in score-based generative modeling, we can accurately estimate these scores with neural networks, and use numerical SDE solvers to generate samples. We show that this framework encapsulates previous approaches in score-based generative modeling and diffusion probabilistic modeling, allowing for new sampling procedures and new modeling capabilities. In particular, we introduce a predictor-corrector framework to correct errors in the evolution of the discretized reverse-time SDE. We also derive an equivalent neural ODE that samples from the same distribution as the SDE, but additionally enables exact likelihood computation, and improved sampling efficiency. In addition, we provide a new way to solve inverse problems with score-based models, as demonstrated with experiments on class-conditional generation, image inpainting, and colorization. Combined with multiple architectural improvements, we achieve record-breaking performance for unconditional image generation on CIFAR-10 with an Inception score of 9.89 and FID of 2.20, a competitive likelihood of 2.99 bits/dim, and demonstrate high fidelity generation of $1024\times 1024$ images for the first time from a score-based generative model.

----

## [43] Share or Not? Learning to Schedule Language-Specific Capacity for Multilingual Translation

**Authors**: *Biao Zhang, Ankur Bapna, Rico Sennrich, Orhan Firat*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=Wj4ODo0uyCF](https://openreview.net/forum?id=Wj4ODo0uyCF)

**Abstract**:

Using a mix of shared and language-specific (LS) parameters has shown promise in multilingual neural machine  translation (MNMT), but the question of when and where LS capacity matters most is still under-studied. We offer such a study by proposing conditional language-specific routing (CLSR). CLSR employs hard binary gates conditioned on token representations to dynamically select LS or shared paths. By manipulating these gates, it can schedule LS capacity across sub-layers in MNMT subject to the guidance of translation signals and budget constraints. Moreover, CLSR can easily scale up to massively multilingual settings. Experiments with Transformer on OPUS-100 and WMT datasets show that: 1) MNMT is sensitive to both the amount and the position of LS modeling: distributing 10%-30% LS computation to the top and/or bottom encoder/decoder layers delivers the best performance; and 2) one-to-many translation benefits more from CLSR compared to many-to-one translation, particularly with unbalanced training data. Our study further verifies the trade-off between the shared capacity and LS capacity for multilingual translation. We corroborate our analysis by confirming the soundness of our findings as foundation of our improved multilingual Transformers. Source code and models are available at https://github.com/bzhangGo/zero/tree/iclr2021_clsr.

----

## [44] Image GANs meet Differentiable Rendering for Inverse Graphics and Interpretable 3D Neural Rendering

**Authors**: *Yuxuan Zhang, Wenzheng Chen, Huan Ling, Jun Gao, Yinan Zhang, Antonio Torralba, Sanja Fidler*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=yWkP7JuHX1](https://openreview.net/forum?id=yWkP7JuHX1)

**Abstract**:

Differentiable rendering has paved the way to training neural networks to perform “inverse graphics” tasks such as predicting 3D geometry from monocular photographs. To train high performing models, most of the current approaches rely on multi-view imagery which are not readily available in practice.  Recent Generative Adversarial Networks (GANs) that synthesize images, in contrast, seem to acquire 3D knowledge implicitly during training: object viewpoints can be manipulated by simply manipulating the latent codes. However, these latent codes often lack further physical interpretation and thus GANs cannot easily be inverted to perform explicit 3D reasoning. In this paper, we aim to extract and disentangle 3D knowledge learned by generative models by utilizing differentiable renderers. Key to our approach is to exploit GANs as a multi-view data generator to train an inverse graphics network using an off-the-shelf differentiable renderer, and the trained inverse graphics network as a teacher to disentangle the GAN's latent code into interpretable 3D properties. The entire architecture is trained iteratively using cycle consistency losses. We show that our approach significantly outperforms state-of-the-art inverse graphics networks trained on existing datasets, both quantitatively and via user studies. We further showcase the disentangled GAN as a controllable 3D “neural renderer", complementing traditional graphics renderers.

----

## [45] How Neural Networks Extrapolate: From Feedforward to Graph Neural Networks

**Authors**: *Keyulu Xu, Mozhi Zhang, Jingling Li, Simon Shaolei Du, Ken-ichi Kawarabayashi, Stefanie Jegelka*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=UH-cmocLJC](https://openreview.net/forum?id=UH-cmocLJC)

**Abstract**:

We study how neural networks trained by gradient descent  extrapolate, i.e., what they learn outside the support of the training distribution. Previous works report mixed empirical results when extrapolating with neural networks: while feedforward neural networks, a.k.a. multilayer perceptrons (MLPs), do not extrapolate well in certain simple tasks, Graph Neural Networks (GNNs) -- structured networks with MLP modules -- have shown some success in more complex tasks.  Working towards a theoretical explanation, we identify conditions under which MLPs and GNNs extrapolate well. First, we quantify the observation that ReLU MLPs quickly converge to linear functions along any direction from the origin, which implies that ReLU MLPs do not extrapolate most nonlinear functions. But, they can provably learn a linear target function when the training distribution is sufficiently diverse. Second, in connection to analyzing the successes and limitations of GNNs, these results suggest a hypothesis for which we provide theoretical and empirical evidence: the success of GNNs in extrapolating algorithmic tasks to new data (e.g., larger graphs or edge weights) relies on encoding task-specific non-linearities in the architecture or features. Our theoretical analysis builds on a connection of over-parameterized networks to the neural tangent kernel.  Empirically, our theory holds across different training settings.

----

## [46] Contrastive Explanations for Reinforcement Learning via Embedded Self Predictions

**Authors**: *Zhengxian Lin, Kin-Ho Lam, Alan Fern*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=Ud3DSz72nYR](https://openreview.net/forum?id=Ud3DSz72nYR)

**Abstract**:

We investigate a deep reinforcement learning (RL) architecture that supports explaining why a learned agent prefers one action over another. The key idea is to learn action-values that are directly represented via human-understandable properties of expected futures. This is realized via the embedded self-prediction (ESP) model, which learns said properties in terms of human provided features. Action preferences can then be explained by contrasting the future properties predicted for each action. To address cases where there are a large number of features, we develop a novel method for computing minimal sufficient explanations from an ESP. Our case studies in three domains, including a complex strategy game, show that ESP models can be effectively learned and support insightful explanations.

----

## [47] Improved Autoregressive Modeling with Distribution Smoothing

**Authors**: *Chenlin Meng, Jiaming Song, Yang Song, Shengjia Zhao, Stefano Ermon*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=rJA5Pz7lHKb](https://openreview.net/forum?id=rJA5Pz7lHKb)

**Abstract**:

While autoregressive models excel at image compression, their sample quality is often lacking. Although not realistic, generated images often have high likelihood according to the model, resembling the case of adversarial examples. Inspired by a successful adversarial defense method, we incorporate randomized smoothing into autoregressive generative modeling. We first model a smoothed version of the data distribution, and then reverse the smoothing process to recover the original data distribution. This procedure drastically improves the sample quality of existing autoregressive models on several synthetic and real-world image datasets while obtaining competitive likelihoods on synthetic datasets.

----

## [48] MONGOOSE: A Learnable LSH Framework for Efficient Neural Network Training

**Authors**: *Beidi Chen, Zichang Liu, Binghui Peng, Zhaozhuo Xu, Jonathan Lingjie Li, Tri Dao, Zhao Song, Anshumali Shrivastava, Christopher Ré*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=wWK7yXkULyh](https://openreview.net/forum?id=wWK7yXkULyh)

**Abstract**:

Recent advances by practitioners in the deep learning community have breathed new life into Locality Sensitive Hashing (LSH), using it to reduce memory and time bottlenecks in neural network (NN) training. However, while LSH has sub-linear guarantees for approximate near-neighbor search in theory, it is known to have inefficient query time in practice due to its use of random hash functions. Moreover, when model parameters are changing, LSH suffers from update overhead. This work is motivated by an observation that model parameters evolve slowly, such that the changes do not always require an LSH update to maintain performance. This phenomenon points to the potential for a reduction in update time and allows for a modified learnable version of data-dependent LSH to improve query time at a low cost. We use the above insights to build MONGOOSE, an end-to-end LSH framework for efficient NN training. In particular, MONGOOSE is equipped with a scheduling algorithm to adaptively perform LSH updates with provable guarantees and learnable hash functions to improve query efficiency. Empirically, we validate MONGOOSE on large-scale deep learning models for recommendation systems and language modeling. We find that it achieves up to 8% better accuracy compared to previous LSH approaches, with $6.5 \times$ speed-up and $6\times$ reduction in memory usage.

----

## [49] Gradient Projection Memory for Continual Learning

**Authors**: *Gobinda Saha, Isha Garg, Kaushik Roy*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=3AOj0RCNC2](https://openreview.net/forum?id=3AOj0RCNC2)

**Abstract**:

The ability to learn continually without forgetting the past tasks is a desired attribute for artificial learning systems. Existing approaches to enable such learning in artificial neural networks usually rely on network growth, importance based weight update or replay of old data from the memory. In contrast, we propose a novel approach where a neural network learns new tasks by taking gradient steps in the orthogonal direction to the gradient subspaces deemed important for the past tasks. We find the bases of these subspaces by analyzing network representations (activations) after learning each task with Singular Value Decomposition (SVD) in a single shot manner and store them in the memory as Gradient Projection Memory (GPM). With qualitative and quantitative analyses, we show that such orthogonal gradient descent induces minimum to no interference with the past tasks, thereby mitigates forgetting. We evaluate our algorithm on diverse image classification datasets with short and long sequences of tasks and report better or on-par performance compared to the state-of-the-art approaches.

----

## [50] Why Are Convolutional Nets More Sample-Efficient than Fully-Connected Nets?

**Authors**: *Zhiyuan Li, Yi Zhang, Sanjeev Arora*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=uCY5MuAxcxU](https://openreview.net/forum?id=uCY5MuAxcxU)

**Abstract**:

Convolutional neural networks often dominate fully-connected counterparts in generalization performance, especially on image classification tasks. This is often explained in terms of \textquotedblleft better inductive bias.\textquotedblright\  However, this has not been made mathematically rigorous, and the hurdle is that the sufficiently wide fully-connected net can always simulate the convolutional net. Thus the training algorithm plays a role. The current work describes a natural task on which a provable sample complexity gap can be shown, for standard training algorithms. We construct a single natural distribution on $\mathbb{R}^d\times\{\pm 1\}$ on which any orthogonal-invariant algorithm (i.e. fully-connected networks trained with most gradient-based methods from gaussian initialization) requires $\Omega(d^2)$ samples to generalize while $O(1)$ samples suffice for convolutional architectures. Furthermore, we demonstrate a single target function, learning which on all possible distributions leads to an $O(1)$ vs $\Omega(d^2/\varepsilon)$ gap. The proof relies on the fact that SGD on fully-connected network is orthogonal equivariant. Similar results are achieved for $\ell_2$ regression and adaptive training algorithms, e.g. Adam and AdaGrad, which are only permutation equivariant.

----

## [51] Iterated learning for emergent systematicity in VQA

**Authors**: *Ankit Vani, Max Schwarzer, Yuchen Lu, Eeshan Dhekane, Aaron C. Courville*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=Pd_oMxH8IlF](https://openreview.net/forum?id=Pd_oMxH8IlF)

**Abstract**:

Although neural module networks have an architectural bias towards compositionality, they require gold standard layouts to generalize systematically in practice. When instead learning layouts and modules jointly, compositionality does not arise automatically and an explicit pressure is necessary for the emergence of layouts exhibiting the right structure. We propose to address this problem using iterated learning, a cognitive science theory of the emergence of compositional languages in nature that has primarily been applied to simple referential games in machine learning. Considering the layouts of module networks as samples from an emergent language, we use iterated learning to encourage the development of structure within this language. We show that the resulting layouts support systematic generalization in neural agents solving the more complex task of visual question-answering. Our regularized iterated learning method can outperform baselines without iterated learning on SHAPES-SyGeT (SHAPES Systematic Generalization Test), a new split of the SHAPES dataset we introduce to evaluate systematic generalization, and on CLOSURE, an extension of CLEVR also designed to test systematic generalization. We demonstrate superior performance in recovering ground-truth compositional program structure with limited supervision on both SHAPES-SyGeT and CLEVR.

----

## [52] Coupled Oscillatory Recurrent Neural Network (coRNN): An accurate and (gradient) stable architecture for learning long time dependencies

**Authors**: *T. Konstantin Rusch, Siddhartha Mishra*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=F3s69XzWOia](https://openreview.net/forum?id=F3s69XzWOia)

**Abstract**:

Circuits of biological neurons, such as in the functional parts of the brain can be modeled as networks of coupled oscillators. Inspired by the ability of these systems to express a rich set of outputs while keeping (gradients of) state variables bounded, we propose a novel architecture for recurrent neural networks. Our proposed RNN is based on a time-discretization of a system of second-order ordinary differential equations, modeling networks of controlled nonlinear oscillators. We prove precise bounds on the gradients of the hidden states, leading to the mitigation of the exploding and vanishing gradient problem for this RNN. Experiments show that the proposed RNN is comparable in performance to the state of the art on a variety of benchmarks, demonstrating the potential of this architecture to provide stable and accurate RNNs for processing complex sequential data.

----

## [53] Sparse Quantized Spectral Clustering

**Authors**: *Zhenyu Liao, Romain Couillet, Michael W. Mahoney*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=pBqLS-7KYAF](https://openreview.net/forum?id=pBqLS-7KYAF)

**Abstract**:

Given a large data matrix, sparsifying, quantizing, and/or performing other entry-wise nonlinear operations can have numerous benefits, ranging from speeding up iterative algorithms for core numerical linear algebra problems to providing nonlinear filters to design state-of-the-art neural network models. Here, we exploit tools from random matrix theory to make precise statements about how the eigenspectrum of a matrix changes under such nonlinear transformations. In particular, we show that very little change occurs in the informative eigenstructure, even under drastic sparsification/quantization, and consequently that very little downstream performance loss occurs when working with very aggressively sparsified or quantized spectral clustering problems.
We illustrate how these results depend on the nonlinearity, we characterize a phase transition beyond which spectral clustering becomes possible, and we show when such nonlinear transformations can introduce spurious non-informative eigenvectors.

----

## [54] Graph-Based Continual Learning

**Authors**: *Binh Tang, David S. Matteson*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=HHSEKOnPvaO](https://openreview.net/forum?id=HHSEKOnPvaO)

**Abstract**:

Despite significant advances, continual learning models still suffer from catastrophic forgetting when exposed to incrementally available data from non-stationary distributions. Rehearsal approaches alleviate the problem by maintaining and replaying a small episodic memory of previous samples, often implemented as an array of independent memory slots. In this work, we propose to augment such an array with a learnable random graph that captures pairwise similarities between its samples, and use it not only to learn new tasks but also to guard against forgetting. Empirical results on several benchmark datasets show that our model consistently outperforms recently proposed baselines for task-free continual learning.

----

## [55] Dynamic Tensor Rematerialization

**Authors**: *Marisa Kirisame, Steven Lyubomirsky, Altan Haan, Jennifer Brennan, Mike He, Jared Roesch, Tianqi Chen, Zachary Tatlock*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=Vfs_2RnOD0H](https://openreview.net/forum?id=Vfs_2RnOD0H)

**Abstract**:

Checkpointing enables the training of deep learning models under restricted memory budgets by freeing intermediate activations from memory and recomputing them on demand. Current checkpointing techniques statically plan these recomputations offline and assume static computation graphs. We demonstrate that a simple online algorithm can achieve comparable performance by introducing Dynamic Tensor Rematerialization (DTR), a greedy online algorithm for checkpointing that is extensible and general, is parameterized by eviction policy, and supports dynamic models. We prove that DTR can train an $N$-layer linear feedforward network on an  $\Omega(\sqrt{N})$ memory budget with only $\mathcal{O}(N)$ tensor operations. DTR closely matches the performance of optimal static checkpointing in simulated experiments. We incorporate a DTR prototype into PyTorch merely by interposing on tensor allocations and operator calls and collecting lightweight metadata on tensors.

----

## [56] Gradient Vaccine: Investigating and Improving Multi-task Optimization in Massively Multilingual Models

**Authors**: *Zirui Wang, Yulia Tsvetkov, Orhan Firat, Yuan Cao*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=F1vEjWK-lH_](https://openreview.net/forum?id=F1vEjWK-lH_)

**Abstract**:

Massively multilingual models subsuming tens or even hundreds of languages pose great challenges to multi-task optimization. While it is a common practice to apply a language-agnostic procedure optimizing a joint multilingual task objective, how to properly characterize and take advantage of its underlying problem structure for improving optimization efficiency remains under-explored. In this paper, we attempt to peek into the black-box of multilingual optimization through the lens of loss function geometry. We find that gradient similarity measured along the optimization trajectory is an important signal, which correlates well with not only language proximity but also the overall model performance. Such observation helps us to identify a critical limitation of existing gradient-based multi-task learning methods, and thus we derive a simple and scalable optimization procedure, named Gradient Vaccine, which encourages more geometrically aligned parameter updates for close tasks. Empirically, our method obtains significant model performance gains on multilingual machine translation and XTREME benchmark tasks for multilingual language models. Our work reveals the importance of properly measuring and utilizing language proximity in multilingual optimization, and has broader implications for multi-task learning beyond multilingual modeling.

----

## [57] CPT: Efficient Deep Neural Network Training via Cyclic Precision

**Authors**: *Yonggan Fu, Han Guo, Meng Li, Xin Yang, Yining Ding, Vikas Chandra, Yingyan Lin*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=87ZwsaQNHPZ](https://openreview.net/forum?id=87ZwsaQNHPZ)

**Abstract**:

Low-precision deep neural network (DNN) training has gained tremendous attention as reducing precision is one of the most effective knobs for boosting DNNs' training time/energy efficiency. In this paper, we attempt to explore low-precision training from a new perspective as inspired by recent findings in understanding DNN training: we conjecture that DNNs' precision might have a similar effect as the learning rate during DNN training, and advocate dynamic precision along the training trajectory for further boosting the time/energy efficiency of DNN training. Specifically, we propose Cyclic Precision Training (CPT) to cyclically vary the precision between two boundary values which can be identified using a simple precision range test within the first few training epochs. Extensive simulations and ablation studies on five datasets and eleven models demonstrate that CPT's effectiveness is consistent across various models/tasks (including classification and language modeling). Furthermore, through experiments and visualization we show that CPT helps to (1) converge to a wider minima with a lower generalization error and (2) reduce training variance which we believe opens up a new design knob for simultaneously improving the optimization and efficiency of DNN training.

----

## [58] Learning a Latent Simplex in Input Sparsity Time

**Authors**: *Ainesh Bakshi, Chiranjib Bhattacharyya, Ravi Kannan, David P. Woodruff, Samson Zhou*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=04LZCAxMSco](https://openreview.net/forum?id=04LZCAxMSco)

**Abstract**:

We consider the problem of learning a latent $k$-vertex simplex $K\in\mathbb{R}^d$, given $\mathbf{A}\in\mathbb{R}^{d\times n}$, which can be viewed as $n$ data points that are formed by randomly perturbing some latent points in $K$, possibly beyond $K$. A large class of latent variable models, such as adversarial clustering, mixed membership stochastic block models, and topic models can be cast in this view of learning a latent simplex. Bhattacharyya and Kannan (SODA 2020) give an algorithm for learning such a $k$-vertex latent simplex in time roughly $O(k\cdot\text{nnz}(\mathbf{A}))$, where $\text{nnz}(\mathbf{A})$ is the number of non-zeros in $\mathbf{A}$. We show that the dependence on $k$ in the running time is unnecessary given a natural assumption about the mass of the top $k$ singular values of $\mathbf{A}$, which holds in many of these applications. Further, we show this assumption is necessary, as otherwise an algorithm for learning a latent simplex would imply a better low rank approximation algorithm than what is known. 

We obtain a spectral low-rank approximation to $\mathbf{A}$ in input-sparsity time and show that the column space thus obtained has small $\sin\Theta$ (angular) distance to the right top-$k$ singular space of $\mathbf{A}$. Our algorithm then selects $k$ points in the low-rank  subspace with the largest inner product (in absolute value) with $k$ carefully chosen random vectors. By working in the low-rank subspace, we avoid reading the entire matrix in each iteration and thus circumvent the $\Theta(k\cdot\text{nnz}(\mathbf{A}))$ running time.

----

## [59] Expressive Power of Invariant and Equivariant Graph Neural Networks

**Authors**: *Waïss Azizian, Marc Lelarge*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=lxHgXYN4bwl](https://openreview.net/forum?id=lxHgXYN4bwl)

**Abstract**:

Various classes of Graph Neural Networks (GNN) have been proposed and shown to be successful in a wide range of applications with graph structured data. In this paper, we propose a theoretical framework able to compare the expressive power of these GNN architectures. The current universality theorems only apply to intractable classes of GNNs. Here, we prove the first approximation guarantees for practical GNNs, paving the way for a better understanding of their generalization. Our theoretical results are proved for invariant GNNs computing a graph embedding (permutation of the nodes of the input graph does not affect the output) and equivariant GNNs computing an embedding of the nodes (permutation of the input permutes the output). We show that Folklore Graph Neural Networks (FGNN), which are tensor based GNNs augmented with matrix multiplication are the most expressive architectures proposed so far for a given tensor order. We illustrate our results on the Quadratic Assignment Problem (a NP-Hard combinatorial problem) by showing that FGNNs are able to learn how to solve the problem, leading to much better average performances than existing algorithms (based on spectral, SDP or other GNNs architectures). On a practical side, we also implement masked tensors to handle batches of graphs of varying sizes.

----

## [60] Discovering a set of policies for the worst case reward

**Authors**: *Tom Zahavy, André Barreto, Daniel J. Mankowitz, Shaobo Hou, Brendan O'Donoghue, Iurii Kemaev, Satinder Singh*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=PUkhWz65dy5](https://openreview.net/forum?id=PUkhWz65dy5)

**Abstract**:

We study the problem of how to construct a set of policies that can be composed together to solve a collection of reinforcement learning tasks. Each task is a different reward function defined as a linear combination of  known features. We consider a specific class of policy compositions which we call set improving policies (SIPs): given a set of policies and a set of tasks, a SIP is any composition of the former whose performance is at least as good as that of its constituents across all the tasks. We focus on the most conservative instantiation of SIPs, set-max policies (SMPs), so our analysis extends to any SIP. This includes known policy-composition operators like generalized policy improvement. Our main contribution is an algorithm that builds a set of policies in order to maximize the worst-case performance of the resulting SMP on the set of tasks. The algorithm works by successively adding new policies to the set. We show that the worst-case performance of the resulting SMP strictly improves at each iteration, and the algorithm only stops when there does not exist a policy that leads to improved performance. We empirically evaluate our algorithm on a grid world and also on a set of domains from the DeepMind control suite. We confirm our theoretical results regarding the monotonically improving performance of our algorithm. Interestingly, we also show empirically that the sets of policies computed by the algorithm are diverse, leading to different trajectories in the grid world and very distinct locomotion skills in the control suite.

----

## [61] Model-Based Visual Planning with Self-Supervised Functional Distances

**Authors**: *Stephen Tian, Suraj Nair, Frederik Ebert, Sudeep Dasari, Benjamin Eysenbach, Chelsea Finn, Sergey Levine*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=UcoXdfrORC](https://openreview.net/forum?id=UcoXdfrORC)

**Abstract**:

A generalist robot must be able to complete a variety of tasks in its environment. One appealing way to specify each task is in terms of a goal observation. However, learning goal-reaching policies with reinforcement learning remains a challenging problem, particularly when hand-engineered reward functions are not available. Learned dynamics models are a promising approach for learning about the environment without rewards or task-directed data, but planning to reach goals with such a model requires a notion of functional similarity between observations and goal states. We present a self-supervised method for model-based visual goal reaching, which uses both a visual dynamics model as well as a dynamical distance function learned using model-free reinforcement learning. Our approach learns entirely using offline, unlabeled data, making it practical to scale to large and diverse datasets. In our experiments, we find that our method can successfully learn models that perform a variety of tasks at test-time, moving objects amid distractors with a simulated robotic arm and even learning to open and close a drawer using a real-world robot. In comparisons, we find that this approach substantially outperforms both model-free and model-based prior methods.

----

## [62] Noise against noise: stochastic label noise helps combat inherent label noise

**Authors**: *Pengfei Chen, Guangyong Chen, Junjie Ye, Jingwei Zhao, Pheng-Ann Heng*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=80FMcTSZ6J0](https://openreview.net/forum?id=80FMcTSZ6J0)

**Abstract**:

The noise in stochastic gradient descent (SGD) provides a crucial implicit regularization effect, previously studied in optimization by analyzing the dynamics of parameter updates. In this paper, we are interested in learning with noisy labels, where we have a collection of samples with potential mislabeling. We show that a previously rarely discussed SGD noise, induced by stochastic label noise (SLN), mitigates the effects of inherent label noise. In contrast, the common SGD noise directly applied to model parameters does not. We formalize the differences and connections of SGD noise variants, showing that SLN induces SGD noise dependent on the sharpness of output landscape and the confidence of output probability, which may help escape from sharp minima and prevent overconfidence. SLN not only improves generalization in its simplest form but also boosts popular robust training methods, including sample selection and label correction. Specifically, we present an enhanced algorithm by applying SLN to label correction. Our code is released.

----

## [63] Contrastive Behavioral Similarity Embeddings for Generalization in Reinforcement Learning

**Authors**: *Rishabh Agarwal, Marlos C. Machado, Pablo Samuel Castro, Marc G. Bellemare*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=qda7-sVg84](https://openreview.net/forum?id=qda7-sVg84)

**Abstract**:

Reinforcement learning methods trained on few environments rarely learn policies that generalize to unseen environments. To improve generalization, we incorporate the inherent sequential structure in reinforcement learning into the representation learning process. This approach is orthogonal to recent approaches, which rarely exploit this structure explicitly. Specifically, we introduce a theoretically motivated policy similarity metric (PSM) for measuring behavioral similarity between states. PSM assigns high similarity to states for which the optimal policies in those states as well as in future states are similar. We also present a contrastive representation learning procedure to embed any state similarity metric, which we instantiate with PSM to obtain policy similarity embeddings (PSEs). We demonstrate that PSEs improve generalization on diverse benchmarks, including LQR with spurious correlations, a jumping task from pixels, and Distracting DM Control Suite.

----

## [64] VAEBM: A Symbiosis between Variational Autoencoders and Energy-based Models

**Authors**: *Zhisheng Xiao, Karsten Kreis, Jan Kautz, Arash Vahdat*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=5m3SEczOV8L](https://openreview.net/forum?id=5m3SEczOV8L)

**Abstract**:

Energy-based models (EBMs) have recently been successful in representing complex distributions of small images. However, sampling from them requires expensive Markov chain Monte Carlo (MCMC) iterations that mix slowly in high dimensional pixel space. Unlike EBMs, variational autoencoders (VAEs) generate samples quickly and are equipped with a latent space that enables fast traversal of the data manifold. However, VAEs tend to assign high probability density to regions in data space outside the actual data distribution and often fail at generating sharp images. In this paper, we propose VAEBM, a symbiotic composition of a VAE and an EBM that offers the best of both worlds. VAEBM captures the overall mode structure of the data distribution using a state-of-the-art VAE and it relies on its EBM component to explicitly exclude non-data-like regions from the model and refine the image samples. Moreover, the VAE component in VAEBM allows us to speed up MCMC updates by reparameterizing them in the VAE's latent space. Our experimental results show that VAEBM outperforms state-of-the-art VAEs and EBMs in generative quality on several benchmark image datasets by a large margin. It can generate high-quality images as large as 256$\times$256 pixels with short MCMC chains. We also demonstrate that VAEBM provides complete mode coverage and performs well in out-of-distribution detection.

----

## [65] Geometry-Aware Gradient Algorithms for Neural Architecture Search

**Authors**: *Liam Li, Mikhail Khodak, Nina Balcan, Ameet Talwalkar*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=MuSYkd1hxRP](https://openreview.net/forum?id=MuSYkd1hxRP)

**Abstract**:

Recent state-of-the-art methods for neural architecture search (NAS) exploit gradient-based optimization by relaxing the problem into continuous optimization over architectures and shared-weights, a noisy process that remains poorly understood. We argue for the study of single-level empirical risk minimization to understand NAS with weight-sharing, reducing the design of NAS methods to devising optimizers and regularizers that can quickly obtain high-quality solutions to this problem. Invoking the theory of mirror descent, we present a geometry-aware framework that exploits the underlying structure of this optimization to return sparse architectural parameters, leading to simple yet novel algorithms that enjoy fast convergence guarantees and achieve state-of-the-art accuracy on the latest NAS benchmarks in computer vision. Notably, we exceed the best published results for both CIFAR and ImageNet on both the DARTS search space and NAS-Bench-201; on the latter we achieve near-oracle-optimal performance on CIFAR-10 and CIFAR-100. Together, our theory and experiments demonstrate a principled way to co-design optimizers and continuous relaxations of discrete NAS search spaces.

----

## [66] Learning-based Support Estimation in Sublinear Time

**Authors**: *Talya Eden, Piotr Indyk, Shyam Narayanan, Ronitt Rubinfeld, Sandeep Silwal, Tal Wagner*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=tilovEHA3YS](https://openreview.net/forum?id=tilovEHA3YS)

**Abstract**:

We consider the  problem of estimating the number of distinct elements in a large data set (or, equivalently, the support size of the distribution induced by the data set) from a random sample of its elements. The problem occurs in many applications, including biology, genomics, computer systems and linguistics. A line of research spanning the last decade resulted in algorithms that estimate the support up to $ \pm \varepsilon n$ from a sample of size $O(\log^2(1/\varepsilon) \cdot n/\log n)$, where $n$ is the data set size.  Unfortunately, this bound is known to be tight, limiting further improvements to the complexity of this problem. In this paper we consider estimation algorithms augmented with a machine-learning-based predictor that, given any element, returns an estimation of  its frequency.  We show that if the predictor is correct up to a constant approximation factor, then the sample complexity can be reduced significantly,  to
$$ \ \log (1/\varepsilon) \cdot n^{1-\Theta(1/\log(1/\varepsilon))}. $$
We evaluate the proposed algorithms on a collection of data sets, using the neural-network based estimators from {Hsu et al, ICLR'19} as predictors. Our experiments  demonstrate substantial (up to 3x) improvements in the estimation accuracy compared to the state of the art algorithm.

----

## [67] Deciphering and Optimizing Multi-Task Learning: a Random Matrix Approach

**Authors**: *Malik Tiomoko, Hafiz Tiomoko Ali, Romain Couillet*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=Cri3xz59ga](https://openreview.net/forum?id=Cri3xz59ga)

**Abstract**:

This article provides theoretical insights into the inner workings of multi-task and transfer learning methods, by studying the tractable least-square support vector machine multi-task learning (LS-SVM MTL) method, in the limit of large ($p$) and numerous ($n$) data. By a random matrix analysis applied to a Gaussian mixture data model, the performance of MTL LS-SVM is shown to converge, as $n,p\to\infty$, to a deterministic limit involving simple (small-dimensional) statistics of the data.

We prove (i) that the standard MTL LS-SVM algorithm is in general strongly biased and may dramatically fail (to the point that individual single-task LS-SVMs may outperform the MTL approach, even for quite resembling tasks): our analysis provides a simple method to correct these biases, and that we reveal (ii) the sufficient statistics at play in the method, which can be efficiently estimated, even for quite small datasets. The latter result is exploited to automatically optimize the hyperparameters without resorting to any cross-validation procedure. 

Experiments on popular datasets demonstrate that our improved MTL LS-SVM method is computationally-efficient and outperforms sometimes much more elaborate state-of-the-art multi-task and transfer learning techniques.

----

## [68] Autoregressive Entity Retrieval

**Authors**: *Nicola De Cao, Gautier Izacard, Sebastian Riedel, Fabio Petroni*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=5k8F6UU39V](https://openreview.net/forum?id=5k8F6UU39V)

**Abstract**:

Entities are at the center of how we represent and aggregate knowledge. For instance, Encyclopedias such as Wikipedia are structured by entities (e.g., one per Wikipedia article). The ability to retrieve such entities given a query is fundamental for knowledge-intensive tasks such as entity linking and open-domain question answering. One way to understand current approaches is as classifiers among atomic labels, one for each entity. Their weight vectors are dense entity representations produced by encoding entity meta information such as their descriptions. This approach leads to several shortcomings: (i) context and entity affinity is mainly captured through a vector dot product, potentially missing fine-grained interactions between the two; (ii) a large memory footprint is needed to store dense representations when considering large entity sets; (iii) an appropriately hard set of negative data has to be subsampled at training time. In this work, we propose GENRE, the first system that retrieves entities by generating their unique names, left to right, token-by-token in an autoregressive fashion and conditioned on the context. This enables us to mitigate the aforementioned technical issues since: (i) the autoregressive formulation allows us to directly capture relations between context and entity name, effectively cross encoding both; (ii) the memory footprint is greatly reduced because the parameters of our encoder-decoder architecture scale with vocabulary size, not entity count; (iii) the exact softmax loss can be efficiently computed without the need to subsample negative data. We show the efficacy of the approach, experimenting with more than 20 datasets on entity disambiguation, end-to-end entity linking and document retrieval tasks, achieving new state-of-the-art or very competitive results while using a tiny fraction of the memory footprint of competing systems. Finally, we demonstrate that new entities can be added by simply specifying their unambiguous name. Code and pre-trained models at https://github.com/facebookresearch/GENRE.

----

## [69] Systematic generalisation with group invariant predictions

**Authors**: *Faruk Ahmed, Yoshua Bengio, Harm van Seijen, Aaron C. Courville*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=b9PoimzZFJ](https://openreview.net/forum?id=b9PoimzZFJ)

**Abstract**:

We consider situations where the presence of dominant simpler correlations with the target variable in a training set can cause an SGD-trained neural network to be less reliant on more persistently correlating complex features. When the non-persistent, simpler correlations correspond to non-semantic background factors, a neural network trained on this data can exhibit dramatic failure upon encountering systematic distributional shift, where the correlating background features are recombined with different objects. We perform an empirical study on three synthetic datasets, showing that group invariance methods across inferred partitionings of the training set can lead to significant improvements at such test-time situations. We also suggest a simple invariance penalty, showing with experiments on our setups that it can perform better than alternatives. We find that even without assuming access to any systematically shifted validation sets, one can still find improvements over an ERM-trained reference model.

----

## [70] Iterative Empirical Game Solving via Single Policy Best Response

**Authors**: *Max Olan Smith, Thomas Anthony, Michael P. Wellman*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=R4aWTjmrEKM](https://openreview.net/forum?id=R4aWTjmrEKM)

**Abstract**:

Policy-Space Response Oracles (PSRO) is a general algorithmic framework for learning policies in multiagent systems by interleaving empirical game analysis with deep reinforcement learning (DRL).
At each iteration, DRL is invoked to train a best response to a mixture of opponent policies.
The repeated application of DRL poses an expensive computational burden as we look to apply this algorithm to more complex domains.
We introduce two variations of PSRO designed to reduce the amount of simulation required during DRL training.
Both algorithms modify how PSRO adds new policies to the empirical game, based on learned responses to a single opponent policy.
The first, Mixed-Oracles, transfers knowledge from previous iterations of DRL, requiring training only against the opponent's newest policy.
The second, Mixed-Opponents, constructs a pure-strategy opponent by mixing existing strategy's action-value estimates, instead of their policies.
Learning against a single policy mitigates conflicting experiences on behalf of a learner facing an unobserved distribution of opponents.
We empirically demonstrate that these algorithms substantially reduce the amount of simulation during training required by PSRO, while producing equivalent or better solutions to the game.

----

## [71] Understanding the role of importance weighting for deep learning

**Authors**: *Da Xu, Yuting Ye, Chuanwei Ruan*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=_WnwtieRHxM](https://openreview.net/forum?id=_WnwtieRHxM)

**Abstract**:

The recent paper by Byrd & Lipton (2019), based on empirical observations, raises a major concern on the impact of importance weighting for the over-parameterized deep learning models. They observe that as long as the model can separate the training data, the impact of importance weighting diminishes as the training proceeds. Nevertheless, there lacks a rigorous characterization of this phenomenon. In this paper, we provide formal characterizations and theoretical justifications on the role of importance weighting with respect to the implicit bias of gradient descent and margin-based learning theory. We reveal both the optimization dynamics and generalization performance under deep learning models. Our work not only explains the various novel phenomenons observed for importance weighting in deep learning, but also extends to the studies where the weights are being optimized as part of the model, which applies to a number of topics under active research.

----

## [72] Long-tail learning via logit adjustment

**Authors**: *Aditya Krishna Menon, Sadeep Jayasumana, Ankit Singh Rawat, Himanshu Jain, Andreas Veit, Sanjiv Kumar*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=37nvvqkCo5](https://openreview.net/forum?id=37nvvqkCo5)

**Abstract**:

Real-world classification problems typically exhibit an imbalanced or long-tailed label distribution, wherein many labels have only a few associated samples. This poses a challenge for generalisation on such labels, and also  makes naive learning biased towards dominant labels. In this paper,  we present a statistical framework that unifies and generalises several recent proposals to cope with these challenges. Our framework revisits the classic idea of logit adjustment based on the label frequencies, which encourages a large relative margin between logits of rare positive versus dominant negative labels. This yields two techniques  for long-tail learning, where such adjustment is either applied post-hoc to a trained model, or enforced in the loss during training. These techniques are statistically grounded, and practically effective on four real-world datasets with long-tailed label distributions.

----

## [73] DDPNOpt: Differential Dynamic Programming Neural Optimizer

**Authors**: *Guan-Horng Liu, Tianrong Chen, Evangelos A. Theodorou*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=6s7ME_X5_Un](https://openreview.net/forum?id=6s7ME_X5_Un)

**Abstract**:

Interpretation of Deep Neural Networks (DNNs) training as an optimal control problem with nonlinear dynamical systems has received considerable attention recently, yet the algorithmic development remains relatively limited. In this work, we make an attempt along this line by reformulating the training procedure from the trajectory optimization perspective. We first show that most widely-used algorithms for training DNNs can be linked to the Differential Dynamic Programming (DDP), a celebrated second-order method rooted in the Approximate Dynamic Programming. In this vein, we propose a new class of optimizer, DDP Neural Optimizer (DDPNOpt), for training feedforward and convolution networks. DDPNOpt features layer-wise feedback policies which improve convergence and reduce sensitivity to hyper-parameter over existing methods. It outperforms other optimal-control inspired training methods in both convergence and complexity, and is competitive against state-of-the-art first and second order methods. We also observe DDPNOpt has surprising benefit in preventing gradient vanishing. Our work opens up new avenues for principled algorithmic design built upon the optimal control theory.

----

## [74] Learning with Feature-Dependent Label Noise: A Progressive Approach

**Authors**: *Yikai Zhang, Songzhu Zheng, Pengxiang Wu, Mayank Goswami, Chao Chen*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=ZPa2SyGcbwh](https://openreview.net/forum?id=ZPa2SyGcbwh)

**Abstract**:

Label noise is frequently observed in real-world large-scale datasets. The noise is introduced due to a variety of reasons; it is heterogeneous and feature-dependent. Most existing approaches to handling noisy labels fall into two categories: they either assume an ideal feature-independent noise, or remain heuristic without theoretical guarantees. In this paper, we propose to target a new family of feature-dependent label noise, which is much more general than commonly used i.i.d. label noise and encompasses a broad spectrum of noise patterns. Focusing on this general noise family, we propose a progressive label correction algorithm that iteratively corrects labels and refines the model. We provide theoretical guarantees showing that for a wide variety of (unknown) noise patterns, a classifier trained with this strategy converges to be consistent with the Bayes classifier. In experiments, our method outperforms SOTA baselines and is robust to various noise types and levels.

----

## [75] Information Laundering for Model Privacy

**Authors**: *Xinran Wang, Yu Xiang, Jun Gao, Jie Ding*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=dyaIRud1zXg](https://openreview.net/forum?id=dyaIRud1zXg)

**Abstract**:

In this work, we propose information laundering, a novel framework for enhancing model privacy. Unlike data privacy that concerns the protection of raw data information, model privacy aims to protect an already-learned model that is to be deployed for public use. The private model can be obtained from general learning methods, and its deployment means that it will return a deterministic or random response for a given input query. An information-laundered model consists of probabilistic components that deliberately maneuver the intended input and output for queries of the model, so the model's adversarial acquisition is less likely. Under the proposed framework, we develop an information-theoretic principle to quantify the fundamental tradeoffs between model utility and privacy leakage and derive the optimal design.

----

## [76] Mutual Information State Intrinsic Control

**Authors**: *Rui Zhao, Yang Gao, Pieter Abbeel, Volker Tresp, Wei Xu*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=OthEq8I5v1](https://openreview.net/forum?id=OthEq8I5v1)

**Abstract**:

Reinforcement learning has been shown to be highly successful at many challenging tasks. However, success heavily relies on well-shaped rewards. Intrinsically motivated RL attempts to remove this constraint by defining an intrinsic reward function. Motivated by the self-consciousness concept in psychology, we make a natural assumption that the agent knows what constitutes itself, and propose a new intrinsic objective that encourages the agent to have maximum control on the environment. We mathematically formalize this reward as the mutual information between the agent state and the surrounding state under the current agent policy. With this new intrinsic motivation, we are able to outperform previous methods, including being able to complete the pick-and-place task for the first time without using any task reward. A video showing experimental results is available at https://youtu.be/AUCwc9RThpk.

----

## [77] Benefit of deep learning with non-convex noisy gradient descent: Provable excess risk bound and superiority to kernel methods

**Authors**: *Taiji Suzuki, Shunta Akiyama*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=2m0g1wEafh](https://openreview.net/forum?id=2m0g1wEafh)

**Abstract**:

Establishing a theoretical analysis that explains why deep learning can outperform shallow learning such as kernel methods is one of the biggest issues in the deep learning literature. Towards answering this question, we evaluate excess risk of a deep learning estimator trained by a noisy gradient descent with ridge regularization on a mildly overparameterized neural network, 
and discuss its superiority to a class of linear estimators that includes neural tangent kernel approach, random feature model, other kernel methods, $k$-NN estimator and so on. We consider a teacher-student regression model, and eventually show that {\it any} linear estimator can be outperformed by deep learning in a sense of the minimax optimal rate especially for a high dimension setting. The obtained excess bounds are so-called fast learning rate which is faster than $O(1/\sqrt{n})$ that is obtained by usual Rademacher complexity analysis. This discrepancy is induced by the non-convex geometry of the model and the noisy gradient descent used for neural network training provably reaches a near global optimal solution even though the loss landscape is highly non-convex. Although the noisy gradient descent does not employ any explicit or implicit sparsity inducing regularization, it shows a preferable generalization performance that dominates linear estimators.

----

## [78] How Does Mixup Help With Robustness and Generalization?

**Authors**: *Linjun Zhang, Zhun Deng, Kenji Kawaguchi, Amirata Ghorbani, James Zou*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=8yKEo06dKNo](https://openreview.net/forum?id=8yKEo06dKNo)

**Abstract**:

Mixup is a popular data augmentation technique based on on convex combinations of pairs of examples and their labels. This simple technique has shown to substantially improve both the model's robustness as well as the generalization of the trained model. However,  it is not well-understood why such improvement occurs. In this paper, we provide theoretical analysis to demonstrate how using Mixup in training helps model robustness and generalization. For robustness, we show that minimizing the Mixup loss corresponds to approximately minimizing an upper bound of the adversarial loss. This explains why models obtained by Mixup training exhibits robustness to several kinds of adversarial attacks such as Fast Gradient Sign Method (FGSM). For generalization, we prove that Mixup augmentation corresponds to a specific type of data-adaptive regularization which reduces overfitting. Our analysis provides new insights and a framework to understand Mixup.

----

## [79] Dataset Inference: Ownership Resolution in Machine Learning

**Authors**: *Pratyush Maini, Mohammad Yaghini, Nicolas Papernot*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=hvdKKV2yt7T](https://openreview.net/forum?id=hvdKKV2yt7T)

**Abstract**:

With increasingly more data and computation involved in their training,  machine learning models constitute valuable intellectual property. This has spurred interest in model stealing, which is made more practical by advances in learning with partial, little, or no supervision. Existing defenses focus on inserting unique watermarks in a model's decision surface, but this is insufficient:  the watermarks are not sampled from the training distribution and thus are not always preserved during model stealing. In this paper, we make the key observation that knowledge contained in the stolen model's training set is what is common to all stolen copies. The adversary's goal, irrespective of the attack employed, is always to extract this knowledge or its by-products. This gives the original model's owner a strong advantage over the adversary: model owners have access to the original training data. We thus introduce $\textit{dataset inference}$, the process of identifying whether a suspected model copy has private knowledge from the original model's dataset, as a defense against model stealing. We develop an approach for dataset inference that combines statistical testing with the ability to estimate the distance of multiple data points to the decision boundary. Our experiments on CIFAR10, SVHN, CIFAR100 and ImageNet show that model owners can claim with confidence greater than 99% that their model (or dataset as a matter of fact) was stolen, despite only exposing 50 of the stolen model's training points. Dataset inference defends against state-of-the-art attacks even when the adversary is adaptive. Unlike prior work, it does not require retraining or overfitting the defended model.

----

## [80] Individually Fair Gradient Boosting

**Authors**: *Alexander Vargo, Fan Zhang, Mikhail Yurochkin, Yuekai Sun*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=JBAa9we1AL](https://openreview.net/forum?id=JBAa9we1AL)

**Abstract**:

We consider the task of enforcing individual fairness in gradient boosting. Gradient boosting is a popular method for machine learning from tabular data, which arise often in applications where algorithmic fairness is a concern. At a high level, our approach is a functional gradient descent on a (distributionally) robust loss function that encodes our intuition of algorithmic fairness for the ML task at hand. Unlike prior approaches to individual fairness that only work with smooth ML models, our approach also works with non-smooth models such as decision trees. We show that our algorithm converges globally and generalizes. We also demonstrate the efficacy of our algorithm on three ML problems susceptible to algorithmic bias.

----

## [81] Large Scale Image Completion via Co-Modulated Generative Adversarial Networks

**Authors**: *Shengyu Zhao, Jonathan Cui, Yilun Sheng, Yue Dong, Xiao Liang, Eric I-Chao Chang, Yan Xu*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=sSjqmfsk95O](https://openreview.net/forum?id=sSjqmfsk95O)

**Abstract**:

Numerous task-specific variants of conditional generative adversarial networks have been developed for image completion. Yet, a serious limitation remains that all existing algorithms tend to fail when handling large-scale missing regions. To overcome this challenge, we propose a generic new approach that bridges the gap between image-conditional and recent modulated unconditional generative architectures via co-modulation of both conditional and stochastic style representations. Also, due to the lack of good quantitative metrics for image completion, we propose the new Paired/Unpaired Inception Discriminative Score (P-IDS/U-IDS), which robustly measures the perceptual fidelity of inpainted images compared to real images via linear separability in a feature space. Experiments demonstrate superior performance in terms of both quality and diversity over state-of-the-art methods in free-form image completion and easy generalization to image-to-image translation. Code is available at https://github.com/zsyzzsoft/co-mod-gan.

----

## [82] Self-Supervised Policy Adaptation during Deployment

**Authors**: *Nicklas Hansen, Rishabh Jangir, Yu Sun, Guillem Alenyà, Pieter Abbeel, Alexei A. Efros, Lerrel Pinto, Xiaolong Wang*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=o_V-MjyyGV_](https://openreview.net/forum?id=o_V-MjyyGV_)

**Abstract**:

In most real world scenarios, a policy trained by reinforcement learning in one environment needs to be deployed in another, potentially quite different environment. However, generalization across different environments is known to be hard. A natural solution would be to keep training after deployment in the new environment, but this cannot be done if the new environment offers no reward signal. Our work explores the use of self-supervision to allow the policy to continue training after deployment without using any rewards. While previous methods explicitly anticipate changes in the new environment, we assume no prior knowledge of those changes yet still obtain significant improvements. Empirical evaluations are performed on diverse simulation environments from DeepMind Control suite and ViZDoom, as well as real robotic manipulation tasks in  continuously changing environments, taking observations from an uncalibrated camera. Our method improves generalization in 31 out of 36 environments across various tasks and outperforms domain randomization on a majority of environments. Webpage and implementation: https://nicklashansen.github.io/PAD/.

----

## [83] Sharpness-aware Minimization for Efficiently Improving Generalization

**Authors**: *Pierre Foret, Ariel Kleiner, Hossein Mobahi, Behnam Neyshabur*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=6Tm1mposlrM](https://openreview.net/forum?id=6Tm1mposlrM)

**Abstract**:

In today's heavily overparameterized models, the value of the training loss provides few guarantees on model generalization ability. Indeed, optimizing only the training loss value, as is commonly done, can easily lead to suboptimal model quality. Motivated by the connection between geometry of the loss landscape and generalization---including a generalization bound that we prove here---we introduce a novel, effective procedure for instead simultaneously minimizing loss value and loss sharpness.  In particular, our procedure, Sharpness-Aware Minimization (SAM), seeks parameters that lie in neighborhoods having uniformly low loss; this formulation results in a min-max optimization problem on which gradient descent can be performed efficiently. We present empirical results showing that SAM improves model generalization across a variety of benchmark datasets (e.g., CIFAR-{10, 100}, ImageNet, finetuning tasks) and models, yielding novel state-of-the-art performance for several.  Additionally, we find that SAM natively provides robustness to label noise on par with that provided by state-of-the-art procedures that specifically target learning with noisy labels.

----

## [84] PMI-Masking: Principled masking of correlated spans

**Authors**: *Yoav Levine, Barak Lenz, Opher Lieber, Omri Abend, Kevin Leyton-Brown, Moshe Tennenholtz, Yoav Shoham*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=3Aoft6NWFej](https://openreview.net/forum?id=3Aoft6NWFej)

**Abstract**:

Masking tokens uniformly at random constitutes a common flaw in the pretraining of Masked Language Models (MLMs) such as BERT. We show that such uniform masking allows an MLM to minimize its training objective by latching onto shallow local signals, leading to pretraining inefficiency and suboptimal downstream performance. To address this flaw, we propose PMI-Masking, a principled masking strategy based on the concept of Pointwise Mutual Information (PMI), which jointly masks a token n-gram if it exhibits high collocation over the corpus. PMI-Masking motivates, unifies, and improves upon prior more heuristic approaches that attempt to address the drawback of random uniform token masking, such as whole-word masking, entity/phrase masking, and random-span masking. Specifically, we show experimentally that PMI-Masking reaches the performance of prior masking approaches in half the training time, and consistently improves performance at the end of pretraining.

----

## [85] Very Deep VAEs Generalize Autoregressive Models and Can Outperform Them on Images

**Authors**: *Rewon Child*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=RLRXCV6DbEJ](https://openreview.net/forum?id=RLRXCV6DbEJ)

**Abstract**:

We present a hierarchical VAE that, for the first time, generates samples quickly $\textit{and}$ outperforms the PixelCNN in log-likelihood on all natural image benchmarks. We begin by observing that, in theory, VAEs can actually represent autoregressive models, as well as faster, better models if they exist, when made sufficiently deep. Despite this, autoregressive models have historically outperformed VAEs in log-likelihood. We test if insufficient depth explains why by scaling a VAE to greater stochastic depth than previously explored and evaluating it CIFAR-10, ImageNet, and FFHQ. In comparison to the PixelCNN, these very deep VAEs achieve higher likelihoods, use fewer parameters, generate samples thousands of times faster, and are more easily applied to high-resolution images. Qualitative studies suggest this is because the VAE learns efficient hierarchical visual representations. We release our source code and models at https://github.com/openai/vdvae.

----

## [86] Data-Efficient Reinforcement Learning with Self-Predictive Representations

**Authors**: *Max Schwarzer, Ankesh Anand, Rishab Goel, R. Devon Hjelm, Aaron C. Courville, Philip Bachman*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=uCQfPZwRaUu](https://openreview.net/forum?id=uCQfPZwRaUu)

**Abstract**:

While deep reinforcement learning excels at solving tasks where large amounts of data can be collected through virtually unlimited interaction with the environment, learning from limited interaction remains a key challenge. We posit that an agent can learn more efficiently if we augment reward maximization with self-supervised objectives based on structure in its visual input and sequential interaction with the environment.  Our method, Self-Predictive Representations (SPR), trains an agent to predict its own latent state representations multiple steps into the future. We compute target representations for future states using an encoder which is an exponential moving average of the agent’s parameters and we make predictions using a learned transition model.  On its own,  this future prediction objective outperforms prior methods for sample-efficient deep RL from pixels. We further improve performance by adding data augmentation to the future prediction loss, which forces the agent’s representations to be consistent across multiple views of an observation.  Our full self-supervised objective, which combines future prediction and data augmentation, achieves a median human-normalized score of 0.415 on Atari in a setting limited to 100k steps of environment interaction, which represents a 55% relative improvement over the previous state-of-the-art. Notably, even in this limited data regime, SPR exceeds expert human scores on 7 out of 26 games. We’ve made the code associated with this work available at https://github.com/mila-iqia/spr.

----

## [87] Watch-And-Help: A Challenge for Social Perception and Human-AI Collaboration

**Authors**: *Xavier Puig, Tianmin Shu, Shuang Li, Zilin Wang, Yuan-Hong Liao, Joshua B. Tenenbaum, Sanja Fidler, Antonio Torralba*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=w_7JMpGZRh0](https://openreview.net/forum?id=w_7JMpGZRh0)

**Abstract**:

In this paper, we introduce Watch-And-Help (WAH), a challenge for testing social intelligence in agents. In WAH, an AI agent needs to help a human-like agent perform a complex household task efficiently. To succeed, the AI agent needs to i) understand the underlying goal of the task by watching a single demonstration of the human-like agent performing the same task (social perception), and ii) coordinate with the human-like agent to solve the task in an unseen environment as fast as possible (human-AI collaboration). For this challenge, we build VirtualHome-Social, a multi-agent household environment, and provide a benchmark including both planning and learning based baselines. We evaluate the performance of AI agents with the human-like agent as well as and with real humans using objective metrics and subjective user ratings. Experimental results demonstrate that our challenge and virtual environment enable a systematic evaluation on the important aspects of machine social intelligence at scale.

----

## [88] A Good Image Generator Is What You Need for High-Resolution Video Synthesis

**Authors**: *Yu Tian, Jian Ren, Menglei Chai, Kyle Olszewski, Xi Peng, Dimitris N. Metaxas, Sergey Tulyakov*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=6puCSjH3hwA](https://openreview.net/forum?id=6puCSjH3hwA)

**Abstract**:

Image and video synthesis are closely related areas aiming at generating content from noise. While rapid progress has been demonstrated in improving image-based models to handle large resolutions, high-quality renderings, and wide variations in image content, achieving comparable video generation results remains problematic. We present a framework that leverages contemporary image generators to render high-resolution videos. We frame the video synthesis problem as discovering a trajectory in the latent space of a pre-trained and fixed image generator. Not only does such a framework render high-resolution videos, but it also is an order of magnitude more computationally efficient. We introduce a motion generator that discovers the desired trajectory, in which content and motion are disentangled. With such a representation, our framework allows for a broad range of applications, including content and motion manipulation. Furthermore, we introduce a new task, which we call cross-domain video synthesis, in which the image and motion generators are trained on disjoint datasets belonging to different domains. This allows for generating moving objects for which the desired video data is not available. Extensive experiments on various datasets demonstrate the advantages of our methods over existing video generation techniques. Code will be released at https://github.com/snap-research/MoCoGAN-HD.

----

## [89] UPDeT: Universal Multi-agent RL via Policy Decoupling with Transformers

**Authors**: *Siyi Hu, Fengda Zhu, Xiaojun Chang, Xiaodan Liang*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=v9c7hr9ADKx](https://openreview.net/forum?id=v9c7hr9ADKx)

**Abstract**:

Recent advances in multi-agent reinforcement learning have been largely limited in training one model from scratch for every new task. The limitation is due to the restricted model architecture related to fixed input and output dimensions. This hinders the experience accumulation and transfer of the learned agent over tasks with diverse levels of difficulty (e.g. 3 vs 3 or 5 vs 6 multi-agent games).  In this paper, we make the first attempt to explore a universal multi-agent reinforcement learning pipeline, designing one single architecture to fit tasks with the requirement of different observation and action configurations. Unlike previous RNN-based models, we utilize a transformer-based model to generate a flexible policy by decoupling the policy distribution from the intertwined input observation with an importance weight measured by the merits of the self-attention mechanism. Compared to a standard transformer block, the proposed model, named as Universal Policy Decoupling Transformer (UPDeT), further relaxes the action restriction and makes the multi-agent task's decision process more explainable. UPDeT is general enough to be plugged into any multi-agent reinforcement learning pipeline and equip them with strong generalization abilities that enables the handling of multiple tasks at a time. Extensive experiments on large-scale SMAC multi-agent competitive games demonstrate that the proposed UPDeT-based multi-agent reinforcement learning achieves significant results relative to state-of-the-art approaches, demonstrating advantageous transfer capability in terms of both performance and training speed (10 times faster).

----

## [90] BUSTLE: Bottom-Up Program Synthesis Through Learning-Guided Exploration

**Authors**: *Augustus Odena, Kensen Shi, David Bieber, Rishabh Singh, Charles Sutton, Hanjun Dai*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=yHeg4PbFHh](https://openreview.net/forum?id=yHeg4PbFHh)

**Abstract**:

Program synthesis is challenging largely because of the difficulty of search in a large space of programs. Human programmers routinely tackle the task of writing complex programs by writing sub-programs and then analyzing their intermediate results to compose them in appropriate ways. Motivated by this intuition, we present a new synthesis approach that leverages learning to guide a bottom-up search over programs. In particular, we train a model to prioritize compositions of intermediate values during search conditioned on a given set of input-output examples. This is a powerful combination because of several emergent properties. First, in bottom-up search, intermediate programs can be executed, providing semantic information to the neural network. Second, given the concrete values from those executions, we can exploit rich features based on recent work on property signatures. Finally, bottom-up search allows the system substantial flexibility in what order to generate the solution, allowing the synthesizer to build up a program from multiple smaller sub-programs. Overall, our empirical evaluation finds that the combination of learning and bottom-up search is remarkably effective, even with simple supervised learning approaches. We demonstrate the effectiveness of our technique on two datasets, one from the SyGuS competition and one of our own creation.

----

## [91] Improving Adversarial Robustness via Channel-wise Activation Suppressing

**Authors**: *Yang Bai, Yuyuan Zeng, Yong Jiang, Shu-Tao Xia, Xingjun Ma, Yisen Wang*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=zQTezqCCtNx](https://openreview.net/forum?id=zQTezqCCtNx)

**Abstract**:

The study of adversarial examples and their activations have attracted significant attention for secure and robust learning with deep neural networks (DNNs).  Different from existing works, in this paper, we highlight two new characteristics of adversarial examples from the channel-wise activation perspective:  1) the activation magnitudes of adversarial examples are higher than that of natural examples; and 2) the channels are activated more uniformly by adversarial examples than natural examples. We find that, while the state-of-the-art defense adversarial training has addressed the first issue of high activation magnitude via training on adversarial examples, the second issue of uniform activation remains.  This motivates us to suppress redundant activations from being activated by adversarial perturbations during the adversarial training process, via a Channel-wise Activation Suppressing (CAS) training strategy.  We show that CAS can train a model that inherently suppresses adversarial activations, and can be easily applied to existing defense methods to further improve their robustness. Our work provides a simplebut generic training strategy for robustifying the intermediate layer activations of DNNs.

----

## [92] What are the Statistical Limits of Offline RL with Linear Function Approximation?

**Authors**: *Ruosong Wang, Dean P. Foster, Sham M. Kakade*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=30EvkP2aQLD](https://openreview.net/forum?id=30EvkP2aQLD)

**Abstract**:

Offline reinforcement learning seeks to utilize offline (observational) data to guide the learning of (causal) sequential decision making strategies. The hope is that offline reinforcement learning coupled with function approximation methods (to deal with the curse of dimensionality) can provide a means to help alleviate the excessive sample complexity burden in modern sequential decision making problems. However, the extent to which this broader approach can be effective is not well understood, where the literature largely consists of sufficient conditions.

This work focuses on the basic question of what are necessary representational and distributional conditions that permit provable sample-efficient offline reinforcement learning. Perhaps surprisingly, our main result shows that even if: i) we have realizability in that the true value function of \emph{every} policy is linear in a given set of features and 2) our off-policy data has good  coverage over all features (under a strong spectral condition), any algorithm still (information-theoretically) requires a number of offline samples that is exponential in the problem horizon to non-trivially estimate the value of \emph{any} given policy. Our results highlight that sample-efficient offline policy evaluation is not possible unless significantly stronger conditions hold; such conditions include either having low distribution shift (where the offline data distribution is close to the distribution of the policy to be evaluated) or significantly stronger representational conditions (beyond realizability).

----

## [93] Unlearnable Examples: Making Personal Data Unexploitable

**Authors**: *Hanxun Huang, Xingjun Ma, Sarah Monazam Erfani, James Bailey, Yisen Wang*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=iAmZUo0DxC0](https://openreview.net/forum?id=iAmZUo0DxC0)

**Abstract**:

The volume of "free" data on the internet has been key to the current success of deep learning. However, it also raises privacy concerns about the unauthorized exploitation of personal data for training commercial models. It is thus crucial to develop methods to prevent unauthorized data exploitation. This paper raises the question: can data be made unlearnable for deep learning models? We present a type of error-minimizing noise that can indeed make training examples unlearnable. Error-minimizing noise is intentionally generated to reduce the error of one or more of the training example(s) close to zero, which can trick the model into believing there is "nothing" to learn from these example(s). The noise is restricted to be imperceptible to human eyes, and thus does not affect normal data utility. We empirically verify the effectiveness of error-minimizing noise in both sample-wise and class-wise forms. We also demonstrate its flexibility under extensive experimental settings and practicability in a case study of face recognition. Our work establishes an important ﬁrst step towards making personal data unexploitable to deep learning models.

----

## [94] Learning Mesh-Based Simulation with Graph Networks

**Authors**: *Tobias Pfaff, Meire Fortunato, Alvaro Sanchez-Gonzalez, Peter W. Battaglia*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=roNqYL0_XP](https://openreview.net/forum?id=roNqYL0_XP)

**Abstract**:

Mesh-based simulations are central to modeling complex physical systems in many disciplines across science and engineering. Mesh representations support powerful numerical integration methods and their resolution can be adapted to strike favorable trade-offs between accuracy and efficiency. However, high-dimensional scientific simulations are very expensive to run, and solvers and parameters must often be tuned individually to each system studied.
Here we introduce MeshGraphNets, a framework for learning mesh-based simulations using graph neural networks. Our model can be trained to pass messages on a mesh graph and to adapt the mesh discretization during forward simulation. Our results show it can accurately predict the dynamics of a wide range of physical systems, including aerodynamics, structural mechanics, and cloth. The model's adaptivity supports learning resolution-independent dynamics and can scale to more complex state spaces at test time. Our method is also highly efficient, running 1-2 orders of magnitude faster than the simulation on which it is trained. Our approach broadens the range of problems on which neural network simulators can operate and promises to improve the efficiency of complex, scientific modeling tasks.

----

## [95] Locally Free Weight Sharing for Network Width Search

**Authors**: *Xiu Su, Shan You, Tao Huang, Fei Wang, Chen Qian, Changshui Zhang, Chang Xu*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=S0UdquAnr9k](https://openreview.net/forum?id=S0UdquAnr9k)

**Abstract**:

Searching for network width is an effective way to slim deep neural networks with hardware budgets. With this aim, a one-shot supernet is usually leveraged as a performance evaluator to rank the performance \wrt~different width. Nevertheless, current methods mainly follow a manually fixed weight sharing pattern, which is limited to distinguish the performance gap of different width. In this paper, to better evaluate each width, we propose a locally free weight sharing strategy (CafeNet) accordingly. In CafeNet, weights are more freely shared, and each width is jointly indicated by its base channels and free channels, where free channels are supposed to locate freely in a local zone to better represent each width. Besides, we propose to further reduce the search space by leveraging our introduced FLOPs-sensitive bins. As a result, our CafeNet can be trained stochastically and get optimized within a min-min strategy. Extensive experiments on ImageNet, CIFAR-10, CelebA and MS COCO dataset have verified our superiority comparing to other state-of-the-art baselines. For example, our method can further boost the benchmark NAS network EfficientNet-B0 by 0.41\% via searching its width more delicately.

----

## [96] Graph Convolution with Low-rank Learnable Local Filters

**Authors**: *Xiuyuan Cheng, Zichen Miao, Qiang Qiu*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=9OHFhefeB86](https://openreview.net/forum?id=9OHFhefeB86)

**Abstract**:

Geometric variations like rotation, scaling, and viewpoint changes pose a significant challenge to visual understanding. One common solution is to directly model certain intrinsic structures, e.g., using landmarks. However, it then becomes non-trivial to build effective deep models, especially when the underlying non-Euclidean grid is irregular and coarse. Recent deep models using graph convolutions provide an appropriate framework to handle such non-Euclidean data, but many of them, particularly those based on global graph Laplacians, lack expressiveness to capture local features required for representation of signals lying on the non-Euclidean grid. The current paper introduces a new type of graph convolution with learnable low-rank local filters, which is provably more expressive than previous spectral graph convolution methods. The model also provides a unified framework for both spectral and spatial graph convolutions. To improve model robustness, regularization by local graph Laplacians is introduced. The representation stability against input graph data perturbation is theoretically proved, making use of the graph filter locality and the local graph regularization. Experiments on spherical mesh data, real-world facial expression recognition/skeleton-based action recognition data, and data with simulated graph noise show the empirical advantage of the proposed model.

----

## [97] Regularized Inverse Reinforcement Learning

**Authors**: *Wonseok Jeon, Chen-Yang Su, Paul Barde, Thang Doan, Derek Nowrouzezahrai, Joelle Pineau*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=HgLO8yalfwc](https://openreview.net/forum?id=HgLO8yalfwc)

**Abstract**:

Inverse Reinforcement Learning (IRL) aims to facilitate a learner’s ability to imitate expert behavior by acquiring reward functions that explain the expert’s decisions. Regularized IRLapplies strongly convex regularizers to the learner’s policy in order to avoid the expert’s behavior being rationalized by arbitrary constant rewards, also known as degenerate solutions. We propose tractable solutions, and practical methods to obtain them, for regularized IRL. Current methods are restricted to the maximum-entropy IRL framework, limiting them to Shannon-entropy regularizers, as well as proposing solutions that are intractable in practice.  We present theoretical backing for our proposed IRL method’s applicability to both discrete and continuous controls, empirically validating our performance on a variety of tasks.

----

## [98] Interpreting Graph Neural Networks for NLP With Differentiable Edge Masking

**Authors**: *Michael Sejr Schlichtkrull, Nicola De Cao, Ivan Titov*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=WznmQa42ZAx](https://openreview.net/forum?id=WznmQa42ZAx)

**Abstract**:

Graph neural networks (GNNs) have become a popular approach to integrating structural inductive biases into NLP models. However, there has been little work on interpreting them, and specifically on understanding which parts of the graphs (e.g. syntactic trees or co-reference structures) contribute to a prediction. In this work, we introduce a post-hoc method for interpreting the predictions of GNNs which identifies unnecessary edges. Given a trained GNN model, we learn a simple classifier that, for every edge in every layer, predicts if that edge can be dropped. We demonstrate that such a classifier can be trained in a fully differentiable fashion, employing stochastic gates and encouraging sparsity through the expected $L_0$ norm. We use our technique as an attribution method to analyze GNN models for two tasks -- question answering and semantic role labeling -- providing insights into the information flow in these models. We show that we can drop a large proportion of edges without deteriorating the performance of the model, while we can analyse the remaining edges for interpreting model predictions.

----

## [99] Deep Neural Network Fingerprinting by Conferrable Adversarial Examples

**Authors**: *Nils Lukas, Yuxuan Zhang, Florian Kerschbaum*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=VqzVhqxkjH1](https://openreview.net/forum?id=VqzVhqxkjH1)

**Abstract**:

In Machine Learning as a Service, a provider trains a deep neural network and gives many users access. The hosted (source) model is susceptible to model stealing attacks, where an adversary derives a surrogate model from API access to the source model. For post hoc detection of such attacks, the provider needs a robust method to determine whether a suspect model is a surrogate of their model. We propose a fingerprinting method for deep neural network classifiers that extracts a set of inputs from the source model so that only surrogates agree with the source model on the classification of such inputs. These inputs are a subclass of transferable adversarial examples which we call conferrable adversarial examples that exclusively transfer with a target label from a source model to its surrogates. We propose a new method to generate these conferrable adversarial examples. We present an extensive study on the irremovability of our fingerprint against fine-tuning, weight pruning, retraining, retraining with different architectures, three model extraction attacks from related work, transfer learning, adversarial training, and two new adaptive attacks. Our fingerprint is robust against distillation, related model extraction attacks, and even transfer learning when the attacker has no access to the model provider's dataset. Our fingerprint is the first method that reaches a ROC AUC of 1.0 in verifying surrogates, compared to a ROC AUC of 0.63 by previous fingerprints.

----

## [100] Tent: Fully Test-Time Adaptation by Entropy Minimization

**Authors**: *Dequan Wang, Evan Shelhamer, Shaoteng Liu, Bruno A. Olshausen, Trevor Darrell*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=uXl3bZLkr3c](https://openreview.net/forum?id=uXl3bZLkr3c)

**Abstract**:

A model must adapt itself to generalize to new and different data during testing. In this setting of fully test-time adaptation the model has only the test data and its own parameters. We propose to adapt by test entropy minimization (tent): we optimize the model for confidence as measured by the entropy of its predictions. Our method estimates normalization statistics and optimizes channel-wise affine transformations to update online on each batch. Tent reduces generalization error for image classification on corrupted ImageNet and CIFAR-10/100 and reaches a new state-of-the-art error on ImageNet-C. Tent handles source-free domain adaptation on digit recognition from SVHN to MNIST/MNIST-M/USPS, on semantic segmentation from GTA to Cityscapes, and on the VisDA-C benchmark. These results are achieved in one epoch of test-time optimization without altering training.

----

## [101] GAN "Steerability" without optimization

**Authors**: *Nurit Spingarn, Ron Banner, Tomer Michaeli*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=zDy_nQCXiIj](https://openreview.net/forum?id=zDy_nQCXiIj)

**Abstract**:

Recent research has shown remarkable success in revealing "steering" directions in the latent spaces of pre-trained GANs. These directions correspond to semantically meaningful image transformations (e.g., shift, zoom, color manipulations), and have the same interpretable effect across all categories that the GAN can generate. Some methods focus on user-specified transformations, while others discover transformations in an unsupervised manner. However, all existing techniques rely on an optimization procedure to expose those directions, and offer no control over the degree of allowed interaction between different transformations. In this paper, we show that "steering" trajectories can be computed in closed form directly from the generator's weights without any form of training or optimization. This applies to user-prescribed geometric transformations, as well as to unsupervised discovery of more complex effects. Our approach allows determining both linear and nonlinear trajectories, and has many advantages over previous methods. In particular, we can control whether one transformation is allowed to come on the expense of another (e.g., zoom-in with or without allowing translation to keep the object centered). Moreover, we can determine the natural end-point of the trajectory, which corresponds to the largest extent to which a transformation can be applied without incurring degradation. Finally, we show how transferring attributes between images can be achieved without optimization, even across different categories.

----

## [102] Contrastive Divergence Learning is a Time Reversal Adversarial Game

**Authors**: *Omer Yair, Tomer Michaeli*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=MLSvqIHRidA](https://openreview.net/forum?id=MLSvqIHRidA)

**Abstract**:

Contrastive divergence (CD) learning is a classical method for fitting unnormalized statistical models to data samples. Despite its wide-spread use, the convergence properties of this algorithm are still not well understood. The main source of difficulty is an unjustified approximation which has been used to derive the gradient of the loss. In this paper, we present an alternative derivation of CD that does not require any approximation and sheds new light on the objective that is actually being optimized by the algorithm. Specifically, we show that CD is an adversarial learning procedure, where a discriminator attempts to classify whether a Markov chain generated from the model has been time-reversed. Thus, although predating generative adversarial networks (GANs) by more than a decade, CD is, in fact, closely related to these techniques. Our derivation settles well with previous observations, which have concluded that CD's update steps cannot be expressed as the gradients of any fixed objective function. In addition, as a byproduct, our derivation reveals a simple correction that can be used as an alternative to Metropolis-Hastings rejection, which is required when the underlying Markov chain is inexact (e.g., when using Langevin dynamics with a large step).

----

## [103] Topology-Aware Segmentation Using Discrete Morse Theory

**Authors**: *Xiaoling Hu, Yusu Wang, Fuxin Li, Dimitris Samaras, Chao Chen*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=LGgdb4TS4Z](https://openreview.net/forum?id=LGgdb4TS4Z)

**Abstract**:

In the segmentation of fine-scale structures from natural and biomedical images, per-pixel accuracy is not the only metric of concern. Topological correctness, such as vessel connectivity and membrane closure, is crucial for downstream analysis tasks. In this paper, we propose a new approach to train deep image segmentation networks for better topological accuracy. In particular, leveraging the power of discrete Morse theory (DMT), we identify global structures, including 1D skeletons and 2D patches, which are important for topological accuracy. Trained with a novel loss based on these global structures, the network performance is significantly improved especially near topologically challenging locations (such as weak spots of connections and membranes). On diverse datasets, our method achieves superior performance on both the DICE score and topological metrics.

----

## [104] Are Neural Rankers still Outperformed by Gradient Boosted Decision Trees?

**Authors**: *Zhen Qin, Le Yan, Honglei Zhuang, Yi Tay, Rama Kumar Pasumarthi, Xuanhui Wang, Michael Bendersky, Marc Najork*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=Ut1vF_q_vC](https://openreview.net/forum?id=Ut1vF_q_vC)

**Abstract**:

Despite the success of neural models on many major machine learning problems, their effectiveness on traditional Learning-to-Rank (LTR) problems is still not widely acknowledged. We first validate this concern by showing that most recent neural LTR models are, by a large margin, inferior to the best publicly available Gradient Boosted Decision Trees (GBDT) in terms of their reported ranking accuracy on benchmark datasets. This unfortunately was somehow overlooked in recent neural LTR papers. We then investigate why existing neural LTR models under-perform and identify several of their weaknesses. Furthermore, we propose a unified framework comprising of counter strategies to ameliorate the existing weaknesses of neural models. Our models are the first to be able to perform equally well, comparing with the best tree-based baseline, while outperforming recently published neural LTR models by a large margin. Our results can also serve as a benchmark to facilitate future improvement of neural LTR models.

----

## [105] Predicting Infectiousness for Proactive Contact Tracing

**Authors**: *Yoshua Bengio, Prateek Gupta, Tegan Maharaj, Nasim Rahaman, Martin Weiss, Tristan Deleu, Eilif Benjamin Müller, Meng Qu, Victor Schmidt, Pierre-Luc St-Charles, Hannah Alsdurf, Olexa Bilaniuk, David L. Buckeridge, Gaétan Marceau-Caron, Pierre Luc Carrier, Joumana Ghosn, Satya Ortiz-Gagne, Christopher J. Pal, Irina Rish, Bernhard Schölkopf, Abhinav Sharma, Jian Tang, Andrew Williams*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=lVgB2FUbzuQ](https://openreview.net/forum?id=lVgB2FUbzuQ)

**Abstract**:

The COVID-19 pandemic has spread rapidly worldwide, overwhelming manual contact tracing in many countries and resulting in widespread lockdowns for emergency containment. Large-scale digital contact tracing (DCT) has emerged as a potential solution to resume economic and social activity while minimizing spread of the virus. Various DCT methods have been proposed, each making trade-offs be-tween privacy, mobility restrictions, and public health. The most common approach, binary contact tracing (BCT), models infection as a binary event, informed only by an individual’s test results, with corresponding binary recommendations that either all or none of the individual’s contacts quarantine. BCT ignores the inherent uncertainty in contacts and the infection process, which could be used to tailor messaging to high-risk individuals, and prompt proactive testing or earlier warnings. It also does not make use of observations such as symptoms or pre-existing medical conditions, which could be used to make more accurate infectiousness predictions. In this paper, we use a recently-proposed COVID-19 epidemiological simulator to develop and test methods that can be deployed to a smartphone to locally and proactively predict an individual’s infectiousness (risk of infecting others) based on their contact history and other information, while respecting strong privacy constraints. Predictions are used to provide personalized recommendations to the individual via an app, as well as to send anonymized messages to the individual’s contacts, who use this information to better predict their own infectiousness, an approach we call proactive contact tracing (PCT). Similarly to other works, we find that compared to no tracing, all DCT methods tested are able to reduce spread of the disease and thus save lives, even at low adoption rates, strongly supporting a role for DCT methods in managing the pandemic. Further, we find a deep-learning based PCT method which improves over BCT for equivalent average mobility, suggesting PCT could help in safe re-opening and second-wave prevention.

----

## [106] Regularization Matters in Policy Optimization - An Empirical Study on Continuous Control

**Authors**: *Zhuang Liu, Xuanlin Li, Bingyi Kang, Trevor Darrell*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=yr1mzrH3IC](https://openreview.net/forum?id=yr1mzrH3IC)

**Abstract**:

Deep Reinforcement Learning (Deep RL) has been receiving increasingly more attention  thanks to its encouraging performance on a variety of control tasks. Yet, conventional regularization techniques in training neural networks (e.g., $L_2$ regularization, dropout) have been largely ignored in RL methods, possibly because agents are typically trained and evaluated in the same environment, and because the deep RL community focuses more on high-level algorithm designs. In this work, we present the first comprehensive study of regularization techniques with multiple policy optimization algorithms on continuous control tasks. Interestingly, we find conventional regularization techniques on the policy networks can often bring large improvement, especially on harder tasks. Our findings are shown to be robust against training hyperparameter variations. We also compare these techniques with the more widely used entropy regularization. In addition, we study regularizing different components and find that only regularizing the policy network is typically the best. We further analyze why regularization may help generalization in RL from four perspectives - sample complexity, reward distribution, weight norm, and noise robustness. We hope our study provides guidance for future practices in regularizing policy optimization algorithms. Our code is available at https://github.com/xuanlinli17/iclr2021_rlreg .

----

## [107] Minimum Width for Universal Approximation

**Authors**: *Sejun Park, Chulhee Yun, Jaeho Lee, Jinwoo Shin*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=O-XJwyoIF-k](https://openreview.net/forum?id=O-XJwyoIF-k)

**Abstract**:

The universal approximation property of width-bounded networks has been studied as a dual of classical universal approximation results on depth-bounded networks. However, the critical width enabling the universal approximation has not been exactly characterized in terms of the input dimension $d_x$ and the output dimension $d_y$. In this work, we provide the first definitive result in this direction for networks using the ReLU activation functions: The minimum width required for the universal approximation of the $L^p$ functions is exactly $\max\{d_x+1,d_y\}$. We also prove that the same conclusion does not hold for the uniform approximation with ReLU, but does hold with an additional threshold activation function. Our proof technique can be also used to derive a tighter upper bound on the minimum width required for the universal approximation using networks with general activation functions.

----

## [108] Towards Robustness Against Natural Language Word Substitutions

**Authors**: *Xinshuai Dong, Anh Tuan Luu, Rongrong Ji, Hong Liu*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=ks5nebunVn_](https://openreview.net/forum?id=ks5nebunVn_)

**Abstract**:

Robustness against word substitutions has a well-defined and widely acceptable form, i.e., using semantically similar words as substitutions, and thus it is considered as a fundamental stepping-stone towards broader robustness in natural language processing. Previous defense methods capture word substitutions in vector space by using either l_2-ball or hyper-rectangle, which results in perturbation sets that are not inclusive enough or unnecessarily large, and thus impedes mimicry of worst cases for robust training. In this paper, we introduce a novel Adversarial Sparse Convex Combination (ASCC) method. We model the word substitution attack space as a convex hull and leverages a regularization term to enforce perturbation towards an actual substitution, thus aligning our modeling better with the discrete textual space. Based on  ASCC method, we further propose ASCC-defense, which leverages ASCC to generate worst-case perturbations and incorporates adversarial training towards robustness. Experiments show that ASCC-defense outperforms the current state-of-the-arts in terms of robustness on two prevailing NLP tasks, i.e., sentiment analysis and natural language inference, concerning several attacks across multiple model architectures. Besides, we also envision a new class of defense towards robustness in NLP, where our robustly trained word vectors can be plugged into a normally trained model and enforce its robustness without applying any other defense techniques.

----

## [109] On the Theory of Implicit Deep Learning: Global Convergence with Implicit Layers

**Authors**: *Kenji Kawaguchi*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=p-NZIuwqhI4](https://openreview.net/forum?id=p-NZIuwqhI4)

**Abstract**:

A deep equilibrium model uses implicit layers, which are implicitly defined through an equilibrium point of an infinite sequence of computation. It avoids any explicit computation of the infinite sequence by finding an equilibrium point directly via root-finding and by computing gradients via implicit differentiation. In this paper, we analyze the gradient dynamics of deep equilibrium models with nonlinearity only on weight matrices and non-convex objective functions of weights for regression and classification. Despite non-convexity, convergence to global optimum at a linear rate is guaranteed without any assumption on the width of the models, allowing the width to be smaller than the output dimension and the number of data points. Moreover, we prove a relation between the gradient dynamics of the deep implicit layer and the dynamics of trust region Newton method of a shallow explicit layer. This mathematically proven relation along with our numerical observation suggests the importance of understanding implicit bias of implicit layers and an open problem on the topic. Our proofs deal with implicit layers, weight tying and nonlinearity on weights, and differ from those in the related literature.

----

## [110] Structured Prediction as Translation between Augmented Natural Languages

**Authors**: *Giovanni Paolini, Ben Athiwaratkun, Jason Krone, Jie Ma, Alessandro Achille, Rishita Anubhai, Cícero Nogueira dos Santos, Bing Xiang, Stefano Soatto*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=US-TP-xnXI](https://openreview.net/forum?id=US-TP-xnXI)

**Abstract**:

We propose a new framework, Translation between Augmented Natural Languages (TANL), to solve many structured prediction language tasks including joint entity and relation extraction, nested named entity recognition, relation classification, semantic role labeling, event extraction, coreference resolution, and dialogue state tracking. Instead of tackling the problem by training task-specific discriminative classifiers, we frame it as a translation task between augmented natural languages, from which the task-relevant information can be easily extracted. Our approach can match or outperform task-specific models on all tasks, and in particular achieves new state-of-the-art results on joint entity and relation extraction (CoNLL04, ADE, NYT, and ACE2005 datasets), relation classification (FewRel and TACRED), and semantic role labeling (CoNLL-2005 and CoNLL-2012). We accomplish this while using the same architecture and hyperparameters for all tasks, and even when training a single model to solve all tasks at the same time (multi-task learning). Finally, we show that our framework can also significantly improve the performance in a low-resource regime, thanks to better use of label semantics.

----

## [111] How Benign is Benign Overfitting ?

**Authors**: *Amartya Sanyal, Puneet K. Dokania, Varun Kanade, Philip H. S. Torr*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=g-wu9TMPODo](https://openreview.net/forum?id=g-wu9TMPODo)

**Abstract**:

We investigate two causes for adversarial vulnerability in deep neural networks: bad data and (poorly) trained models. When trained with SGD, deep neural networks essentially achieve zero training error, even in the presence of label noise, while also exhibiting good generalization on natural test data, something referred to as benign overfitting (Bartlett et al., 2020; Chatterji & Long, 2020).  However, these models are vulnerable to adversarial attacks. We identify label noise as one of the causes for adversarial vulnerability, and provide theoretical and empirical evidence in support of this. Surprisingly, we find several instances of label noise in datasets such as MNIST and CIFAR, and that robustly trained models incur training error on some of these, i.e. they don’t fit the noise. However, removing noisy labels alone does not suffice to achieve adversarial robustness. We conjecture that in part sub-optimal representation learning is also responsible for adversarial vulnerability. By means of simple theoretical setups, we show how the choice of representation can drastically affect adversarial robustness.

----

## [112] Correcting experience replay for multi-agent communication

**Authors**: *Sanjeevan Ahilan, Peter Dayan*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=xvxPuCkCNPO](https://openreview.net/forum?id=xvxPuCkCNPO)

**Abstract**:

We consider the problem of learning to communicate using multi-agent reinforcement learning (MARL). A common approach is to learn off-policy, using data sampled from a replay buffer. However, messages received in the past may not accurately reflect the current communication policy of each agent, and this complicates learning. We therefore introduce a 'communication correction' which accounts for the non-stationarity of observed communication induced by multi-agent learning. It works by relabelling the received message to make it likely under the communicator's current policy, and thus be a better reflection of the receiver's current environment. To account for cases in which agents are both senders and receivers, we introduce an ordered relabelling scheme. Our correction is computationally efficient and can be integrated with a range of off-policy algorithms. We find in our experiments that it substantially improves the ability of communicating MARL systems to learn across a variety of cooperative and competitive tasks.

----

## [113] Emergent Symbols through Binding in External Memory

**Authors**: *Taylor Whittington Webb, Ishan Sinha, Jonathan D. Cohen*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=LSFCEb3GYU7](https://openreview.net/forum?id=LSFCEb3GYU7)

**Abstract**:

A key aspect of human intelligence is the ability to infer abstract rules directly from high-dimensional sensory data, and to do so given only a limited amount of training experience. Deep neural network algorithms have proven to be a powerful tool for learning directly from high-dimensional data, but currently lack this capacity for data-efficient induction of abstract rules, leading some to argue that symbol-processing mechanisms will be necessary to account for this capacity. In this work, we take a step toward bridging this gap by introducing the Emergent Symbol Binding Network (ESBN), a recurrent network augmented with an external memory that enables a form of variable-binding and indirection. This binding mechanism allows symbol-like representations to emerge through the learning process without the need to explicitly incorporate symbol-processing machinery, enabling the ESBN to learn rules in a manner that is abstracted away from the particular entities to which those rules apply. Across a series of tasks, we show that this architecture displays nearly perfect generalization of learned rules to novel entities given only a limited number of training examples, and outperforms a number of other competitive neural network architectures.

----

## [114] Influence Estimation for Generative Adversarial Networks

**Authors**: *Naoyuki Terashita, Hiroki Ohashi, Yuichi Nonaka, Takashi Kanemaru*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=opHLcXxYTC_](https://openreview.net/forum?id=opHLcXxYTC_)

**Abstract**:

Identifying harmful instances, whose absence in a training dataset improves model performance, is important for building better machine learning models. 
Although previous studies have succeeded in estimating harmful instances under supervised settings, they cannot be trivially extended to generative adversarial networks (GANs).
This is because previous approaches require that (i) the absence of a training instance directly affects the loss value and that (ii) the change in the loss directly measures the harmfulness of the instance for the performance of a model. 
In GAN training, however, neither of the requirements is satisfied. 
This is because, (i) the generator’s loss is not directly affected by the training instances as they are not part of the generator's training steps, and (ii) the values of GAN's losses normally do not capture the generative performance of a model.
To this end, (i) we propose an influence estimation method that uses the Jacobian of the gradient of the generator's loss with respect to the discriminator’s parameters (and vice versa) to trace how the absence of an instance in the discriminator’s training affects the generator’s parameters, and (ii) we propose a novel evaluation scheme, in which we assess harmfulness of each training instance on the basis of how GAN evaluation metric (e.g., inception score) is expected to change due to the removal of the instance.
We experimentally verified that our influence estimation method correctly inferred the changes in GAN evaluation metrics.
We also demonstrated that the removal of the identified harmful instances effectively improved the model’s generative performance with respect to various GAN evaluation metrics.

----

## [115] PlasticineLab: A Soft-Body Manipulation Benchmark with Differentiable Physics

**Authors**: *Zhiao Huang, Yuanming Hu, Tao Du, Siyuan Zhou, Hao Su, Joshua B. Tenenbaum, Chuang Gan*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=xCcdBRQEDW](https://openreview.net/forum?id=xCcdBRQEDW)

**Abstract**:

Simulated virtual environments serve as one of the main driving forces behind developing and evaluating skill learning algorithms. However, existing environments typically only simulate rigid body physics. Additionally, the simulation process usually does not provide gradients that might be useful for planning and control optimizations. We introduce a new differentiable physics benchmark called PasticineLab, which includes a diverse collection of soft body manipulation tasks. In each task, the agent uses manipulators to deform the plasticine into a desired configuration. The underlying physics engine supports differentiable elastic and plastic deformation using the DiffTaichi system, posing many under-explored challenges to robotic agents. We evaluate several existing reinforcement learning (RL) methods and gradient-based methods on this benchmark. Experimental results suggest that 1) RL-based approaches struggle to solve most of the tasks efficiently;  2) gradient-based approaches, by optimizing open-loop control sequences with the built-in differentiable physics engine, can rapidly find a solution within tens of iterations, but still fall short on multi-stage tasks that require long-term planning. We expect that PlasticineLab will encourage the development of novel algorithms that combine differentiable physics and RL for more complex physics-based skill learning tasks. PlasticineLab will be made publicly available.

----

## [116] Implicit Normalizing Flows

**Authors**: *Cheng Lu, Jianfei Chen, Chongxuan Li, Qiuhao Wang, Jun Zhu*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=8PS8m9oYtNy](https://openreview.net/forum?id=8PS8m9oYtNy)

**Abstract**:

Normalizing flows define a probability distribution by an explicit invertible transformation $\boldsymbol{\mathbf{z}}=f(\boldsymbol{\mathbf{x}})$. In this work, we present implicit normalizing flows (ImpFlows), which generalize normalizing flows by allowing the mapping to be implicitly defined by the roots of an equation $F(\boldsymbol{\mathbf{z}}, \boldsymbol{\mathbf{x}})= \boldsymbol{\mathbf{0}}$. ImpFlows build on residual flows (ResFlows) with a proper balance between expressiveness and tractability. Through theoretical analysis, we show that the function space of ImpFlow is strictly richer than that of ResFlows. Furthermore, for any ResFlow with a fixed number of blocks, there exists some function that ResFlow has a non-negligible approximation error. However, the function is exactly representable by a single-block ImpFlow. We propose a scalable algorithm to train and draw samples from ImpFlows. Empirically, we evaluate ImpFlow on several classification and density modeling tasks, and ImpFlow outperforms ResFlow with a comparable amount of parameters on all the benchmarks.

----

## [117] Support-set bottlenecks for video-text representation learning

**Authors**: *Mandela Patrick, Po-Yao Huang, Yuki Markus Asano, Florian Metze, Alexander G. Hauptmann, João F. Henriques, Andrea Vedaldi*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=EqoXe2zmhrh](https://openreview.net/forum?id=EqoXe2zmhrh)

**Abstract**:

The dominant paradigm for learning video-text representations – noise contrastive learning – increases the similarity of the representations of pairs of samples that are known to be related, such as text and video from the same sample, and pushes away the representations of all other pairs. We posit that this last behaviour is too strict, enforcing dissimilar representations even for samples that are semantically-related – for example, visually similar videos or ones that share the same depicted action. In this paper, we propose a novel method that alleviates this by leveraging a generative model to naturally push these related samples together: each sample’s caption must be reconstructed as a weighted combination of a support set of visual representations. This simple idea ensures that representations are not overly-specialized to individual samples, are reusable across the dataset, and results in representations that explicitly encode semantics shared between samples, unlike noise contrastive learning. Our proposed method outperforms others by a large margin on MSR-VTT, VATEX, ActivityNet, and MSVD for video-to-text and text-to-video retrieval.

----

## [118] Winning the L2RPN Challenge: Power Grid Management via Semi-Markov Afterstate Actor-Critic

**Authors**: *Deunsol Yoon, Sunghoon Hong, Byung-Jun Lee, Kee-Eung Kim*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=LmUJqB1Cz8](https://openreview.net/forum?id=LmUJqB1Cz8)

**Abstract**:

Safe and reliable electricity transmission in power grids is crucial for modern society. It is thus quite natural that there has been a growing interest in the automatic management of power grids, exempliﬁed by the Learning to Run a Power Network Challenge (L2RPN), modeling the problem as a reinforcement learning (RL) task. However, it is highly challenging to manage a real-world scale power grid, mostly due to the massive scale of its state and action space. In this paper, we present an off-policy actor-critic approach that effectively tackles the unique challenges in power grid management by RL, adopting the hierarchical policy together with the afterstate representation. Our agent ranked ﬁrst in the latest challenge (L2RPN WCCI 2020), being able to avoid disastrous situations while maintaining the highest level of operational efﬁciency in every test scenarios. This paper provides a formal description of the algorithmic aspect of our approach, as well as further experimental studies on diverse power grids.

----

## [119] Learning Incompressible Fluid Dynamics from Scratch - Towards Fast, Differentiable Fluid Models that Generalize

**Authors**: *Nils Wandel, Michael Weinmann, Reinhard Klein*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=KUDUoRsEphu](https://openreview.net/forum?id=KUDUoRsEphu)

**Abstract**:

Fast and stable fluid simulations are an essential prerequisite for applications ranging from computer-generated imagery to computer-aided design in research and development. However, solving the partial differential equations of incompressible fluids is a challenging task and traditional numerical approximation schemes come at high computational costs. Recent deep learning based approaches promise vast speed-ups but do not generalize to new fluid domains, require fluid simulation data for training, or rely on complex pipelines that outsource major parts of the fluid simulation to traditional methods.

In this work, we propose a novel physics-constrained training approach that generalizes to new fluid domains, requires no fluid simulation data, and allows convolutional neural networks to map a fluid state from time-point t to a subsequent state at time t+dt in a single forward pass. This simplifies the pipeline to train and evaluate neural fluid models. After training, the framework yields models that are capable of fast fluid simulations and can handle various fluid phenomena including the Magnus effect and Kármán vortex streets. We present an interactive real-time demo to show the speed and generalization capabilities of our trained models. Moreover, the trained neural networks are efficient differentiable fluid solvers as they offer a differentiable update step to advance the fluid simulation in time. We exploit this fact in a proof-of-concept optimal control experiment. Our models significantly outperform a recent differentiable fluid solver in terms of computational speed and accuracy.

----

## [120] The Traveling Observer Model: Multi-task Learning Through Spatial Variable Embeddings

**Authors**: *Elliot Meyerson, Risto Miikkulainen*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=qYda4oLEc1](https://openreview.net/forum?id=qYda4oLEc1)

**Abstract**:

This paper frames a general prediction system as an observer traveling around a continuous space, measuring values at some locations, and predicting them at others. The observer is completely agnostic about any particular task being solved; it cares only about measurement locations and their values. This perspective leads to a machine learning framework in which seemingly unrelated tasks can be solved by a single model, by embedding their input and output variables into a shared space. An implementation of the framework is developed in which these variable embeddings are learned jointly with internal model parameters. In experiments, the approach is shown to (1) recover intuitive locations of variables in space and time, (2) exploit regularities across related datasets with completely disjoint input and output spaces, and (3) exploit regularities across seemingly unrelated tasks, outperforming task-specific single-task models and multi-task learning alternatives. The results suggest that even seemingly unrelated tasks may originate from similar underlying processes, a fact that the traveling observer model can use to make better predictions.

----

## [121] Grounded Language Learning Fast and Slow

**Authors**: *Felix Hill, Olivier Tieleman, Tamara von Glehn, Nathaniel Wong, Hamza Merzic, Stephen Clark*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=wpSWuz_hyqA](https://openreview.net/forum?id=wpSWuz_hyqA)

**Abstract**:

Recent work has shown that large text-based neural language models acquire a surprising propensity for one-shot learning. Here, we show that an agent situated in a simulated 3D world, and endowed with a novel dual-coding external memory, can exhibit similar one-shot word learning when trained with conventional RL algorithms. After a single introduction to a novel object via visual perception and language ("This is a dax"), the agent can manipulate the object as instructed ("Put the dax on the bed"), combining short-term, within-episode knowledge of the nonsense word with long-term lexical and motor knowledge. We find that, under certain training conditions and with a particular memory writing mechanism, the agent's one-shot word-object binding generalizes to novel exemplars within the same ShapeNet category, and is effective in settings with unfamiliar numbers of objects. We further show how dual-coding memory can be exploited as a signal for intrinsic motivation, stimulating the agent to seek names for objects that may be useful later. Together, the results demonstrate that deep neural networks can exploit meta-learning, episodic memory and an explicitly multi-modal environment to account for 'fast-mapping', a fundamental pillar of human cognitive development and a potentially transformative capacity for artificial agents.

----

## [122] Long-tailed Recognition by Routing Diverse Distribution-Aware Experts

**Authors**: *Xudong Wang, Long Lian, Zhongqi Miao, Ziwei Liu, Stella X. Yu*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=D9I3drBz4UC](https://openreview.net/forum?id=D9I3drBz4UC)

**Abstract**:

Natural data are often long-tail distributed over semantic classes.  Existing recognition methods tackle this imbalanced classification by placing more emphasis on the tail data, through class re-balancing/re-weighting or ensembling over different data groups, resulting in increased tail accuracies but reduced head accuracies.
We take a dynamic view of the training data and provide a principled model bias and variance analysis as the training data fluctuates: Existing long-tail classifiers invariably increase the model variance and the head-tail model bias gap remains large, due to more and larger confusion with hard negatives for the tail.
We propose a new long-tailed classifier called RoutIng Diverse Experts (RIDE).  It reduces the model variance with multiple experts, reduces the model bias with a distribution-aware diversity loss, reduces the computational cost with a dynamic expert routing module.  RIDE outperforms the state-of-the-art by 5% to 7% on CIFAR100-LT, ImageNet-LT and iNaturalist 2018 benchmarks.  It is also a universal framework that is applicable to various backbone networks, long-tailed algorithms and training mechanisms for consistent performance gains. Our code is available at: https://github.com/frank-xwang/RIDE-LongTailRecognition.

----

## [123] Differentially Private Learning Needs Better Features (or Much More Data)

**Authors**: *Florian Tramèr, Dan Boneh*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=YTWGvpFOQD-](https://openreview.net/forum?id=YTWGvpFOQD-)

**Abstract**:

We demonstrate that differentially private machine learning has not yet reached its ''AlexNet moment'' on many canonical vision tasks: linear models trained on handcrafted features significantly outperform end-to-end deep neural networks for moderate privacy budgets.
To exceed the performance of handcrafted features, we show that private learning requires either much more private data, or access to features learned on public data from a similar domain.
Our work introduces simple yet strong baselines for differentially private learning that can inform the evaluation of future progress in this area.

----

## [124] Unsupervised Object Keypoint Learning using Local Spatial Predictability

**Authors**: *Anand Gopalakrishnan, Sjoerd van Steenkiste, Jürgen Schmidhuber*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=GJwMHetHc73](https://openreview.net/forum?id=GJwMHetHc73)

**Abstract**:

We propose PermaKey, a novel approach to representation learning based on object keypoints. It leverages the predictability of local image regions from spatial neighborhoods to identify salient regions that correspond to object parts, which are then converted to keypoints. Unlike prior approaches, it utilizes predictability to discover object keypoints, an intrinsic property of objects. This ensures that it does not overly bias keypoints to focus on characteristics that are not unique to objects, such as movement, shape, colour etc.  We demonstrate the efficacy of PermaKey on Atari where it learns keypoints corresponding to the most salient object parts and is robust to certain visual distractors. Further, on downstream RL tasks in the Atari domain we demonstrate how agents equipped with our keypoints outperform those using competing alternatives, even on challenging environments with moving backgrounds or distractor objects.

----

## [125] On Statistical Bias In Active Learning: How and When to Fix It

**Authors**: *Sebastian Farquhar, Yarin Gal, Tom Rainforth*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=JiYq3eqTKY](https://openreview.net/forum?id=JiYq3eqTKY)

**Abstract**:

Active learning is a powerful tool when labelling data is expensive, but it introduces a bias because the training data no longer follows the population distribution. We formalize this bias and investigate the situations in which it can be harmful and sometimes even helpful. We further introduce novel corrective weights to remove bias when doing so is beneficial. Through this, our work not only provides a useful mechanism that can improve the active learning approach, but also an explanation for the empirical successes of various existing approaches which ignore this bias. In particular, we show that this bias can be actively helpful when training overparameterized models---like neural networks---with relatively modest dataset sizes.

----

## [126] Implicit Convex Regularizers of CNN Architectures: Convex Optimization of Two- and Three-Layer Networks in Polynomial Time

**Authors**: *Tolga Ergen, Mert Pilanci*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=0N8jUH4JMv6](https://openreview.net/forum?id=0N8jUH4JMv6)

**Abstract**:

We study training of Convolutional Neural Networks (CNNs) with ReLU activations and introduce exact convex optimization formulations with a polynomial complexity with respect to the number of data samples, the number of neurons, and data dimension. More specifically, we develop a convex analytic framework utilizing semi-infinite duality to obtain equivalent convex optimization problems for several two- and three-layer CNN architectures. We first prove that two-layer CNNs can be globally optimized via an $\ell_2$ norm regularized convex program. We then show that multi-layer circular CNN training problems with a single ReLU layer are equivalent to an $\ell_1$ regularized convex program that encourages sparsity in the spectral domain. We also extend these results to three-layer CNNs with two ReLU layers. Furthermore, we present extensions of our approach to different pooling methods, which elucidates the implicit architectural bias as convex regularizers.

----

## [127] Generalization in data-driven models of primary visual cortex

**Authors**: *Konstantin-Klemens Lurz, Mohammad Bashiri, Konstantin Willeke, Akshay Jagadish, Eric Wang, Edgar Y. Walker, Santiago A. Cadena, Taliah Muhammad, Erick Cobos, Andreas S. Tolias, Alexander S. Ecker, Fabian H. Sinz*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=Tp7kI90Htd](https://openreview.net/forum?id=Tp7kI90Htd)

**Abstract**:

Deep neural networks (DNN) have set new standards at predicting responses of neural populations to visual input.  Most such DNNs consist of a convolutional network (core) shared across all neurons which learns a representation of neural computation in visual cortex and a neuron-specific readout that linearly combines the relevant features in this representation. The goal of this paper is to test whether such a representation is indeed generally characteristic for visual cortex, i.e. generalizes between animals of a species, and what factors contribute to obtaining such a generalizing core. To push all non-linear computations into the core where the generalizing cortical features should be learned, we devise a novel readout that reduces the number of parameters per neuron in the readout by up to two orders of magnitude compared to the previous state-of-the-art. It does so by taking advantage of retinotopy and learns a Gaussian distribution over the neuron’s receptive field position.  With this new readout we train our network on neural responses from mouse primary visual cortex (V1) and obtain a gain in performance of 7% compared to the previous state-of-the-art network.  We then investigate whether the convolutional core indeed captures general cortical features by using the core in transfer learning to a different animal.  When transferring a core trained on thousands of neurons from various animals and scans we exceed the performance of training directly on that animal by 12%, and outperform a commonly used VGG16 core pre-trained on imagenet by 33%. In addition, transfer learning with our data-driven core is more data-efficient than direct training, achieving the same performance with only 40% of the data. Our model with its novel readout thus sets a new state-of-the-art for neural response prediction in mouse visual cortex from natural images, generalizes between animals, and captures better characteristic cortical features than current task-driven pre-training approaches such as VGG16.

----

## [128] Mathematical Reasoning via Self-supervised Skip-tree Training

**Authors**: *Markus Norman Rabe, Dennis Lee, Kshitij Bansal, Christian Szegedy*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=YmqAnY0CMEy](https://openreview.net/forum?id=YmqAnY0CMEy)

**Abstract**:

We demonstrate that self-supervised language modeling applied to mathematical formulas enables logical reasoning. To measure the logical reasoning abilities of language models, we formulate several evaluation (downstream) tasks, such as inferring types, suggesting missing assumptions and completing equalities. For training language models for formal mathematics, we propose a novel skip-tree task. We find that models trained on the skip-tree task show surprisingly strong mathematical reasoning abilities, and outperform models trained on standard skip-sequence tasks. We also analyze the models' ability to formulate new conjectures by measuring how often the predictions are provable and useful in other proofs.

----

## [129] Beyond Fully-Connected Layers with Quaternions: Parameterization of Hypercomplex Multiplications with 1/n Parameters

**Authors**: *Aston Zhang, Yi Tay, Shuai Zhang, Alvin Chan, Anh Tuan Luu, Siu Cheung Hui, Jie Fu*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=rcQdycl0zyk](https://openreview.net/forum?id=rcQdycl0zyk)

**Abstract**:

Recent works have demonstrated reasonable success of representation learning in hypercomplex space. Specifically, “fully-connected layers with quaternions” (quaternions are 4D hypercomplex numbers), which replace real-valued matrix multiplications in fully-connected layers with Hamilton products of quaternions, both enjoy parameter savings with only 1/4 learnable parameters and achieve comparable performance in various applications. However, one key caveat is that hypercomplex space only exists at very few predefined dimensions (4D, 8D, and 16D). This restricts the flexibility of models that leverage hypercomplex multiplications. To this end, we propose parameterizing hypercomplex multiplications, allowing models to learn multiplication rules from data regardless of whether such rules are predefined. As a result, our method not only subsumes the Hamilton product, but also learns to operate on any arbitrary $n$D hypercomplex space, providing more architectural flexibility using arbitrarily $1/n$ learnable parameters compared with the fully-connected layer counterpart. Experiments of applications to the LSTM and transformer models on natural language inference, machine translation, text style transfer, and subject verb agreement demonstrate architectural flexibility and effectiveness of the proposed approach.

----

## [130] Distributional Sliced-Wasserstein and Applications to Generative Modeling

**Authors**: *Khai Nguyen, Nhat Ho, Tung Pham, Hung Bui*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=QYjO70ACDK](https://openreview.net/forum?id=QYjO70ACDK)

**Abstract**:

Sliced-Wasserstein distance (SW) and its variant, Max Sliced-Wasserstein distance (Max-SW), have been used widely in the recent years due to their fast computation and scalability even when the probability measures lie in a very high dimensional space. However, SW requires many unnecessary projection samples to approximate its value while Max-SW only uses the most important projection, which ignores the information of other useful directions. In order to account for these weaknesses, we propose a novel distance, named Distributional Sliced-Wasserstein distance (DSW), that finds an optimal distribution over projections that can balance between exploring distinctive projecting directions and the informativeness of projections themselves. We show that the DSW is a generalization of Max-SW, and it can be computed efficiently by searching for the optimal push-forward measure over a set of probability measures over the unit sphere satisfying certain regularizing constraints that favor distinct directions. Finally, we conduct extensive experiments with large-scale datasets to demonstrate the favorable performances of the proposed distances over the previous sliced-based distances in generative modeling applications.

----

## [131] Async-RED: A Provably Convergent Asynchronous Block Parallel Stochastic Method using Deep Denoising Priors

**Authors**: *Yu Sun, Jiaming Liu, Yiran Sun, Brendt Wohlberg, Ulugbek Kamilov*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=9EsrXMzlFQY](https://openreview.net/forum?id=9EsrXMzlFQY)

**Abstract**:

Regularization by denoising (RED) is a recently developed framework for solving inverse problems by integrating advanced denoisers as image priors. Recent work has shown its state-of-the-art performance when combined with pre-trained deep denoisers. However, current RED algorithms are inadequate for parallel processing on multicore systems. We address this issue by proposing a new{asynchronous RED (Async-RED) algorithm that enables asynchronous parallel processing of data, making it significantly faster than its serial counterparts for large-scale inverse problems. The computational complexity of Async-RED is further reduced by using a random subset of measurements at every iteration. We present a complete theoretical analysis of the algorithm by establishing its convergence under explicit assumptions on the data-fidelity and the denoiser. We validate Async-RED on image recovery using pre-trained deep denoisers as priors.

----

## [132] DeepAveragers: Offline Reinforcement Learning By Solving Derived Non-Parametric MDPs

**Authors**: *Aayam Kumar Shrestha, Stefan Lee, Prasad Tadepalli, Alan Fern*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=eMP1j9efXtX](https://openreview.net/forum?id=eMP1j9efXtX)

**Abstract**:

We study an approach to offline reinforcement learning (RL) based on optimally solving  finitely-represented  MDPs  derived  from  a  static  dataset  of  experience. This approach can be applied on top of any learned representation and has the potential to easily support multiple solution objectives as well as zero-shot adjustment to changing environments and goals.  Our main contribution is to introduce the Deep Averagers with Costs MDP (DAC-MDP) and to investigate its solutions for offline RL.  DAC-MDPs are a non-parametric model that can leverage deep representations and account for limited data by introducing costs for exploiting under-represented parts of the model.  In theory, we show conditions that allow for lower-bounding the performance of DAC-MDP solutions. We also investigate the empirical behavior in a number of environments, including those with image-based observations. Overall, the experiments demonstrate that the framework can work in practice and scale to large complex offline RL problems.

----

## [133] Learning from Protein Structure with Geometric Vector Perceptrons

**Authors**: *Bowen Jing, Stephan Eismann, Patricia Suriana, Raphael John Lamarre Townshend, Ron O. Dror*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=1YLJDvSx6J4](https://openreview.net/forum?id=1YLJDvSx6J4)

**Abstract**:

Learning on 3D structures of large biomolecules is emerging as a distinct area in machine learning, but there has yet to emerge a unifying network architecture that simultaneously leverages the geometric and relational aspects of the problem domain. To address this gap, we introduce geometric vector perceptrons, which extend standard dense layers to operate on collections of Euclidean vectors. Graph neural networks equipped with such layers are able to perform both geometric and relational reasoning on efficient representations of macromolecules. We demonstrate our approach on two important problems in learning from protein structure: model quality assessment and computational protein design. Our approach improves over existing classes of architectures on both problems, including state-of-the-art convolutional neural networks and graph neural networks. We release our code at https://github.com/drorlab/gvp.

----

## [134] Behavioral Cloning from Noisy Demonstrations

**Authors**: *Fumihiro Sasaki, Ryota Yamashina*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=zrT3HcsWSAt](https://openreview.net/forum?id=zrT3HcsWSAt)

**Abstract**:

We consider the problem of learning an optimal expert behavior policy given noisy demonstrations that contain observations from both optimal and non-optimal expert behaviors. Popular imitation learning algorithms, such as generative adversarial imitation learning, assume that (clear) demonstrations are given from optimal expert policies but not the non-optimal ones, and thus often fail to imitate the optimal expert behaviors given the noisy demonstrations. Prior works that address the problem require (1) learning policies through environment interactions in the same fashion as reinforcement learning, and (2) annotating each demonstration with confidence scores or rankings. However, such environment interactions and annotations in real-world settings take impractically long training time and a significant human effort. In this paper, we propose an imitation learning algorithm to address the problem without any environment interactions and annotations associated with the non-optimal demonstrations. The proposed algorithm learns ensemble policies with a generalized behavioral cloning (BC) objective function where we exploit another policy already learned by BC. Experimental results show that the proposed algorithm can learn behavior policies that are much closer to the optimal policies than ones learned by BC.

----

## [135] Undistillable: Making A Nasty Teacher That CANNOT teach students

**Authors**: *Haoyu Ma, Tianlong Chen, Ting-Kuei Hu, Chenyu You, Xiaohui Xie, Zhangyang Wang*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=0zvfm-nZqQs](https://openreview.net/forum?id=0zvfm-nZqQs)

**Abstract**:

Knowledge Distillation (KD) is a widely used technique to transfer knowledge from pre-trained teacher models to  (usually more lightweight) student models. However, in certain situations, this technique is more of a curse than a blessing. For instance, KD poses a potential risk of exposing intellectual properties (IPs): even if a trained machine learning model is released in ``black boxes'' (e.g., as executable software or APIs without open-sourcing code), it can still be replicated by KD through imitating input-output behaviors. To prevent this unwanted effect of KD, this paper introduces and investigates a concept called $\textit{Nasty Teacher}$: a specially trained teacher network that yields nearly the same performance as a normal one, but would significantly degrade the performance of student models learned by imitating it. We propose a simple yet effective algorithm to build the nasty teacher, called $\textit{self-undermining knowledge distillation}$. Specifically, we aim to maximize the difference between the output of the nasty teacher and a normal pre-trained network. Extensive experiments on several datasets demonstrate that our method is effective on both standard KD and data-free KD, providing the desirable KD-immunity to model owners for the first time. We hope our preliminary study can draw more awareness and interest in this new practical problem of both social and legal importance. Our codes and pre-trained models can be found at: $\url{https://github.com/VITA-Group/Nasty-Teacher}$.

----

## [136] Multivariate Probabilistic Time Series Forecasting via Conditioned Normalizing Flows

**Authors**: *Kashif Rasul, Abdul-Saboor Sheikh, Ingmar Schuster, Urs M. Bergmann, Roland Vollgraf*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=WiGQBFuVRv](https://openreview.net/forum?id=WiGQBFuVRv)

**Abstract**:

Time series forecasting is often fundamental to scientific and engineering problems and enables decision making. With ever increasing data set sizes, a trivial solution to scale up predictions is to assume independence between interacting time series. However, modeling statistical dependencies can improve accuracy and enable analysis of interaction effects. Deep learning methods are well suited for this problem, but multi-variate models often assume a simple parametric distribution and do not scale to high dimensions. In this work we model the multi-variate temporal dynamics of time series via an autoregressive deep learning model, where the data distribution  is represented by a conditioned normalizing flow. This combination retains the power of autoregressive models, such as good performance in extrapolation into the future, with the flexibility of flows as a general purpose high-dimensional distribution model, while remaining computationally tractable. We show that it improves over the state-of-the-art for standard metrics on many real-world data sets with several thousand interacting time-series.

----

## [137] Image Augmentation Is All You Need: Regularizing Deep Reinforcement Learning from Pixels

**Authors**: *Denis Yarats, Ilya Kostrikov, Rob Fergus*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=GY6-6sTvGaf](https://openreview.net/forum?id=GY6-6sTvGaf)

**Abstract**:

We propose a simple data augmentation technique that can be applied to standard model-free reinforcement learning algorithms, enabling robust learning directly from pixels without the need for auxiliary losses or pre-training.  The approach leverages input perturbations commonly used in computer vision tasks to transform input examples, as well as regularizing the value function and policy.  Existing model-free approaches, such as Soft Actor-Critic (SAC), are not able to train deep networks effectively from image pixels. However, the addition of our augmentation method dramatically improves SAC’s performance, enabling it to reach state-of-the-art performance on the DeepMind control suite, surpassing model-based (Hafner et al., 2019; Lee et al., 2019; Hafner et al., 2018) methods and recently proposed contrastive learning (Srinivas et al., 2020).  Our approach, which we dub DrQ: Data-regularized Q, can be combined with any model-free reinforcement learning algorithm. We further demonstrate this by applying it to DQN and significantly improve its data-efficiency on the Atari 100k benchmark.

----

## [138] HW-NAS-Bench: Hardware-Aware Neural Architecture Search Benchmark

**Authors**: *Chaojian Li, Zhongzhi Yu, Yonggan Fu, Yongan Zhang, Yang Zhao, Haoran You, Qixuan Yu, Yue Wang, Cong Hao, Yingyan Lin*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=_0kaDkv3dVf](https://openreview.net/forum?id=_0kaDkv3dVf)

**Abstract**:

HardWare-aware Neural Architecture Search (HW-NAS) has recently gained tremendous attention by automating the design of deep neural networks deployed in more resource-constrained daily life devices. Despite its promising performance, developing optimal HW-NAS solutions can be prohibitively challenging as it requires cross-disciplinary knowledge in the algorithm, micro-architecture, and device-specific compilation. First, to determine the hardware-cost to be incorporated into the NAS process, existing works mostly adopt either pre-collected hardware-cost look-up tables or device-specific hardware-cost models. The former can be time-consuming due to the required knowledge of the device’s compilation method and how to set up the measurement pipeline, while building the latter is often a barrier for non-hardware experts like NAS researchers. Both of them limit the development of HW-NAS innovations and impose a barrier-to-entry to non-hardware experts. Second, similar to generic NAS, it can be notoriously difficult to benchmark HW-NAS algorithms due to their significant required computational resources and the differences in adopted search spaces, hyperparameters, and hardware devices. To this end, we develop HW-NAS-Bench, the first public dataset for HW-NAS research which aims to democratize HW-NAS research to non-hardware experts and make HW-NAS research more reproducible and accessible. To design HW-NAS-Bench, we carefully collected the measured/estimated hardware performance (e.g., energy cost and latency) of all the networks in the search spaces of both NAS-Bench-201 and FBNet, on six hardware devices that fall into three categories (i.e., commercial edge devices, FPGA, and ASIC). Furthermore, we provide a comprehensive analysis of the collected measurements in HW-NAS-Bench to provide insights for HW-NAS research. Finally, we demonstrate exemplary user cases to (1) show that HW-NAS-Bench allows non-hardware experts to perform HW-NAS by simply querying our pre-measured dataset and (2) verify that dedicated device-specific HW-NAS can indeed lead to optimal accuracy-cost trade-offs. The codes and all collected data are available at https://github.com/RICE-EIC/HW-NAS-Bench.

----

## [139] Practical Real Time Recurrent Learning with a Sparse Approximation

**Authors**: *Jacob Menick, Erich Elsen, Utku Evci, Simon Osindero, Karen Simonyan, Alex Graves*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=q3KSThy2GwB](https://openreview.net/forum?id=q3KSThy2GwB)

**Abstract**:

Recurrent neural networks are usually trained with backpropagation through time, which requires storing a complete history of network states, and prohibits updating the weights "online" (after every timestep). Real Time Recurrent Learning (RTRL) eliminates the need for history storage and allows for online weight updates, but does so at the expense of computational costs that are quartic in the state size. This renders RTRL training intractable for all but the smallest networks, even ones that are made highly sparse.
We introduce the Sparse n-step Approximation (SnAp) to the RTRL influence matrix. SnAp only tracks the influence of a parameter on hidden units that are reached by the computation graph within $n$ timesteps of the recurrent core. SnAp with $n=1$ is no more expensive than backpropagation but allows training on arbitrarily long sequences. We find that it substantially outperforms other RTRL approximations with comparable costs such as Unbiased Online Recurrent Optimization. For highly sparse networks, SnAp with $n=2$ remains tractable and can outperform backpropagation through time in terms of learning speed when updates are done online.

----

## [140] Random Feature Attention

**Authors**: *Hao Peng, Nikolaos Pappas, Dani Yogatama, Roy Schwartz, Noah A. Smith, Lingpeng Kong*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=QtTKTdVrFBB](https://openreview.net/forum?id=QtTKTdVrFBB)

**Abstract**:

Transformers are state-of-the-art models for a variety of sequence modeling tasks. At their core is an attention function which models pairwise interactions between the inputs at every timestep. While attention is powerful, it does not scale efficiently to long sequences due to its quadratic time and space complexity in the sequence length. We propose RFA, a linear time and space attention that uses random feature methods to approximate the softmax function, and explore its application in transformers. RFA can be used as a drop-in replacement for conventional softmax attention and offers a straightforward way of learning with recency bias through an optional gating mechanism. Experiments on language modeling and machine translation demonstrate that RFA achieves similar or better performance compared to strong transformer baselines. In the machine translation experiment, RFA decodes twice as fast as a vanilla transformer. Compared to existing efficient transformer variants, RFA is competitive in terms of both accuracy and efficiency on three long text classification datasets. Our analysis shows that RFA’s efficiency gains are especially notable on long sequences, suggesting that RFA will be particularly useful in tasks that require working with large inputs, fast decoding speed, or low memory footprints.

----

## [141] A Gradient Flow Framework For Analyzing Network Pruning

**Authors**: *Ekdeep Singh Lubana, Robert P. Dick*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=rumv7QmLUue](https://openreview.net/forum?id=rumv7QmLUue)

**Abstract**:

Recent network pruning methods focus on pruning models early-on in training. To estimate the impact of removing a parameter, these methods use importance measures that were originally designed to prune trained models. Despite lacking justification for their use early-on in training, such measures result in surprisingly low accuracy loss. To better explain this behavior, we develop a general framework that uses gradient flow to unify state-of-the-art importance measures through the norm of model parameters. We use this framework to determine the relationship between pruning measures and evolution of model parameters, establishing several results related to pruning models early-on in training: (i) magnitude-based pruning removes parameters that contribute least to reduction in loss, resulting in models that converge faster than magnitude-agnostic methods; (ii) loss-preservation based pruning preserves first-order model evolution dynamics and its use is therefore justified for pruning minimally trained models; and (iii) gradient-norm based pruning affects second-order model evolution dynamics, such that increasing gradient norm via pruning can produce poorly performing models. We validate our claims on several VGG-13, MobileNet-V1, and ResNet-56 models trained on CIFAR-10/CIFAR-100.

----

## [142] Recurrent Independent Mechanisms

**Authors**: *Anirudh Goyal, Alex Lamb, Jordan Hoffmann, Shagun Sodhani, Sergey Levine, Yoshua Bengio, Bernhard Schölkopf*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=mLcmdlEUxy-](https://openreview.net/forum?id=mLcmdlEUxy-)

**Abstract**:

We explore the hypothesis that learning modular structures which reflect the dynamics of the environment can lead to better generalization and robustness to changes that only affect a few of the underlying causes. We propose Recurrent Independent Mechanisms (RIMs), a new recurrent architecture in which multiple groups of recurrent cells operate with nearly independent transition dynamics, communicate only sparingly through the bottleneck of attention, and compete with each other so they are updated only at time steps where they are most relevant.  We show that this leads to specialization amongst the RIMs, which in turn allows for remarkably improved generalization on tasks where some factors of variation differ systematically between training and evaluation.

----

## [143] The Intrinsic Dimension of Images and Its Impact on Learning

**Authors**: *Phillip Pope, Chen Zhu, Ahmed Abdelkader, Micah Goldblum, Tom Goldstein*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=XJk19XzGq2J](https://openreview.net/forum?id=XJk19XzGq2J)

**Abstract**:

It is widely believed that natural image data exhibits low-dimensional structure despite the high dimensionality of conventional pixel representations.  This idea underlies a common intuition for the remarkable success of deep learning in computer vision. In this work, we apply dimension estimation tools to popular datasets and investigate the role of low-dimensional structure in deep learning.  We find that common natural image datasets indeed have very low intrinsic dimension relative to the high number of pixels in the images.  Additionally, we find that low dimensional datasets are easier for neural networks to learn, and models solving these tasks generalize better from training to test data.   Along the way,  we develop a technique for validating our dimension estimation tools on synthetic data generated by GANs allowing us to actively manipulate the intrinsic dimension by controlling the image generation process. Code for our experiments may be found  \href{https://github.com/ppope/dimensions}{here}.

----

## [144] Uncertainty Sets for Image Classifiers using Conformal Prediction

**Authors**: *Anastasios Nikolas Angelopoulos, Stephen Bates, Michael I. Jordan, Jitendra Malik*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=eNdiU_DbM9](https://openreview.net/forum?id=eNdiU_DbM9)

**Abstract**:

Convolutional image classifiers can achieve high predictive accuracy, but quantifying their uncertainty remains an unresolved challenge, hindering their deployment in consequential settings. Existing uncertainty quantification techniques, such as Platt scaling, attempt to calibrate the network’s probability estimates, but they do not have formal guarantees. We present an algorithm that modifies any classifier to output a predictive set containing the true label with a user-specified probability, such as 90%. The algorithm is simple and fast like Platt scaling, but provides a formal finite-sample coverage guarantee for every model and dataset. Our method modifies an existing conformal prediction algorithm to give more stable predictive sets by regularizing the small scores of unlikely classes after Platt scaling. In experiments on both Imagenet and Imagenet-V2 with ResNet-152 and other classifiers, our scheme outperforms existing approaches, achieving coverage with sets that are often factors of 5 to 10 smaller than a stand-alone Platt scaling baseline.

----

## [145] Sequential Density Ratio Estimation for Simultaneous Optimization of Speed and Accuracy

**Authors**: *Akinori F. Ebihara, Taiki Miyagawa, Kazuyuki Sakurai, Hitoshi Imaoka*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=Rhsu5qD36cL](https://openreview.net/forum?id=Rhsu5qD36cL)

**Abstract**:

Classifying sequential data as early and as accurately as possible is a challenging yet critical problem, especially when a sampling cost is high. One algorithm that achieves this goal is the sequential probability ratio test (SPRT), which is known as Bayes-optimal: it can keep the expected number of data samples as small as possible, given the desired error upper-bound. However, the original SPRT makes two critical assumptions that limit its application in real-world scenarios: (i) samples are independently and identically distributed, and (ii) the likelihood of the data being derived from each class can be calculated precisely. Here, we propose the SPRT-TANDEM, a deep neural network-based SPRT algorithm that overcomes the above two obstacles. The SPRT-TANDEM sequentially estimates the log-likelihood ratio of two alternative hypotheses by leveraging a novel Loss function for Log-Likelihood Ratio estimation (LLLR) while allowing correlations up to $N (\in \mathbb{N})$  preceding samples. In tests on one original and two public video databases, Nosaic MNIST, UCF101, and SiW, the SPRT-TANDEM achieves statistically significantly better classification accuracy than other baseline classifiers, with a smaller number of data samples. The code and Nosaic MNIST are publicly available at https://github.com/TaikiMiyagawa/SPRT-TANDEM.

----

## [146] Disentangled Recurrent Wasserstein Autoencoder

**Authors**: *Jun Han, Martin Renqiang Min, Ligong Han, Li Erran Li, Xuan Zhang*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=O7ms4LFdsX](https://openreview.net/forum?id=O7ms4LFdsX)

**Abstract**:

Learning disentangled representations leads to interpretable models and facilitates data generation with style transfer, which has been extensively studied on static data such as images in an unsupervised learning framework. However, only a few works have explored unsupervised disentangled sequential representation learning due to challenges of generating sequential data. In this paper, we propose recurrent Wasserstein Autoencoder (R-WAE), a new framework for generative modeling of sequential data. R-WAE disentangles the representation of an input sequence into static and dynamic factors (i.e., time-invariant and time-varying parts). Our theoretical analysis shows that, R-WAE minimizes an upper bound of a penalized form of the Wasserstein distance between model distribution and sequential data distribution, and simultaneously maximizes the mutual information between input data and different disentangled latent factors, respectively. This is superior to (recurrent) VAE which does not explicitly enforce mutual information maximization between input data and disentangled latent representations. When the number of actions in sequential data is available as weak supervision information, R-WAE is extended to learn a categorical latent representation of actions to improve its disentanglement. Experiments on a variety of datasets show that our models outperform other baselines with the same settings in terms of disentanglement and unconditional video generation both quantitatively and qualitatively.

----

## [147] Generalization bounds via distillation

**Authors**: *Daniel Hsu, Ziwei Ji, Matus Telgarsky, Lan Wang*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=EGdFhBzmAwB](https://openreview.net/forum?id=EGdFhBzmAwB)

**Abstract**:

This paper theoretically investigates the following empirical phenomenon: given a high-complexity network with poor generalization bounds, one can distill it into a network with nearly identical predictions but low complexity and vastly smaller generalization bounds.  The main contribution is an analysis showing that the original network inherits this good generalization bound from its distillation, assuming the use of well-behaved data augmentation.  This bound is presented both in an abstract and in a concrete form, the latter complemented by a reduction technique to handle modern computation graphs featuring convolutional layers, fully-connected layers, and skip connections, to name a few.  To round out the story, a (looser) classical uniform convergence analysis of compression is also presented, as well as a variety of experiments on cifar and mnist demonstrating similar generalization performance between the original network and its distillation.

----

## [148] Neural Approximate Sufficient Statistics for Implicit Models

**Authors**: *Yanzhi Chen, Dinghuai Zhang, Michael U. Gutmann, Aaron C. Courville, Zhanxing Zhu*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=SRDuJssQud](https://openreview.net/forum?id=SRDuJssQud)

**Abstract**:

We consider the fundamental problem of how to automatically construct summary statistics for implicit generative models where the evaluation of the likelihood function is intractable but sampling data from the model is possible. The idea is to frame the task of constructing sufficient statistics as learning mutual information maximizing representations of the data with the help of deep neural networks. The infomax learning procedure does not need to estimate any density or density ratio. We apply our approach to both traditional approximate Bayesian computation and recent neural likelihood methods, boosting their performance on a range of tasks.

----

## [149] A Panda? No, It's a Sloth: Slowdown Attacks on Adaptive Multi-Exit Neural Network Inference

**Authors**: *Sanghyun Hong, Yigitcan Kaya, Ionut-Vlad Modoranu, Tudor Dumitras*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=9xC2tWEwBD](https://openreview.net/forum?id=9xC2tWEwBD)

**Abstract**:

Recent increases in the computational demands of deep neural networks (DNNs), combined with the observation that most input samples require only simple models, have sparked interest in input-adaptive multi-exit architectures, such as MSDNets or Shallow-Deep Networks. These architectures enable faster inferences and could bring DNNs to low-power devices, e.g., in the Internet of Things (IoT). However, it is unknown if the computational savings provided by this approach are robust against adversarial pressure. In particular, an adversary may aim to slowdown adaptive DNNs by increasing their average inference time—a threat analogous to the denial-of-service attacks from the Internet. In this paper, we conduct a systematic evaluation of this threat by experimenting with three generic multi-exit DNNs (based on VGG16, MobileNet, and ResNet56) and a custom multi-exit architecture, on two popular image classification benchmarks (CIFAR-10 and Tiny ImageNet). To this end, we show that adversarial example-crafting techniques can be modified to cause slowdown, and we propose a metric for comparing their impact on different architectures. We show that a slowdown attack reduces the efficacy of multi-exit DNNs by 90–100%, and it amplifies the latency by 1.5–5× in a typical IoT deployment. We also show that it is possible to craft universal, reusable perturbations and that the attack can be effective in realistic black-box scenarios, where the attacker has limited knowledge about the victim. Finally, we show that adversarial training provides limited protection against slowdowns. These results suggest that further research is needed for defending multi-exit architectures against this emerging threat. Our code is available at https://github.com/sanghyun-hong/deepsloth.

----

## [150] Orthogonalizing Convolutional Layers with the Cayley Transform

**Authors**: *Asher Trockman, J. Zico Kolter*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=Pbj8H_jEHYv](https://openreview.net/forum?id=Pbj8H_jEHYv)

**Abstract**:

Recent work has highlighted several advantages of enforcing orthogonality in the weight layers of deep networks, such as maintaining the stability of activations, preserving gradient norms, and enhancing adversarial robustness by enforcing low Lipschitz constants. Although numerous methods exist for enforcing the orthogonality of fully-connected layers, those for convolutional layers are more heuristic in nature, often focusing on penalty methods or limited classes of convolutions. In this work, we propose and evaluate an alternative approach to directly parameterize convolutional layers that are constrained to be orthogonal. Specifically, we propose to apply the Cayley transform to a skew-symmetric convolution in the Fourier domain, so that the inverse convolution needed by the Cayley transform can be computed efficiently. We compare our method to previous Lipschitz-constrained and orthogonal convolutional layers and show that it indeed preserves orthogonality to a high degree even for large convolutions. Applied to the problem of certified adversarial robustness, we show that networks incorporating the layer outperform existing deterministic methods for certified defense against $\ell_2$-norm-bounded adversaries, while scaling to larger architectures than previously investigated. Code is available at https://github.com/locuslab/orthogonal-convolutions.

----

## [151] LambdaNetworks: Modeling long-range Interactions without Attention

**Authors**: *Irwan Bello*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=xTJEN-ggl1b](https://openreview.net/forum?id=xTJEN-ggl1b)

**Abstract**:

We present lambda layers -- an alternative framework to self-attention -- for capturing long-range interactions between an input and structured contextual information (e.g. a pixel surrounded by other pixels). Lambda layers capture such interactions by transforming available contexts into linear functions, termed lambdas, and applying these linear functions to each input separately. Similar to linear attention, lambda layers bypass expensive attention maps, but in contrast, they model both content and position-based interactions which enables their application to large structured inputs such as images. The resulting neural network architectures, LambdaNetworks, significantly outperform their convolutional and attentional counterparts on ImageNet classification, COCO object detection and instance segmentation, while being more computationally efficient. Additionally, we design LambdaResNets, a family of hybrid architectures across different scales, that considerably improves the speed-accuracy tradeoff of image classification models. LambdaResNets reach excellent accuracies on ImageNet while being 3.2 - 4.4x faster than the popular EfficientNets on modern machine learning accelerators. In large-scale semi-supervised training with an additional 130M pseudo-labeled images, LambdaResNets achieve up to 86.7% ImageNet accuracy while being 9.5x faster than EfficientNet NoisyStudent and 9x faster than a Vision Transformer with comparable accuracies.

----

## [152] Mind the Pad - CNNs Can Develop Blind Spots

**Authors**: *Bilal Alsallakh, Narine Kokhlikyan, Vivek Miglani, Jun Yuan, Orion Reblitz-Richardson*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=m1CD7tPubNy](https://openreview.net/forum?id=m1CD7tPubNy)

**Abstract**:

We show how feature maps in convolutional networks are susceptible to spatial bias. Due to a combination of architectural choices, the activation at certain locations is systematically elevated or weakened. The major source of this bias is the padding mechanism. Depending on several aspects of convolution arithmetic, this mechanism can apply the padding unevenly, leading to asymmetries in the learned weights. We demonstrate how such bias can be detrimental to certain tasks such as small object detection: the activation is suppressed if the stimulus lies in the impacted area, leading to blind spots and misdetection. We explore alternative padding methods and propose solutions for analyzing and mitigating spatial bias.

----

## [153] Meta-GMVAE: Mixture of Gaussian VAE for Unsupervised Meta-Learning

**Authors**: *Dong Bok Lee, Dongchan Min, Seanie Lee, Sung Ju Hwang*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=wS0UFjsNYjn](https://openreview.net/forum?id=wS0UFjsNYjn)

**Abstract**:

Unsupervised learning aims to learn meaningful representations from unlabeled data which can captures its intrinsic structure, that can be transferred to downstream tasks. Meta-learning, whose objective is to learn to generalize across tasks such that the learned model can rapidly adapt to a novel task, shares the spirit of unsupervised learning in that the both seek to learn more effective and efficient learning procedure than learning from scratch. The fundamental difference of the two is that the most meta-learning approaches are supervised, assuming full access to the labels. However, acquiring labeled dataset for meta-training not only is costly as it requires human efforts in labeling but also limits its applications to pre-defined task distributions. In this paper, we propose a principled unsupervised meta-learning model, namely Meta-GMVAE, based on Variational Autoencoder (VAE) and set-level variational inference. Moreover, we introduce a mixture of Gaussian (GMM) prior, assuming that each modality represents each class-concept in a randomly sampled episode, which we optimize with Expectation-Maximization (EM). Then, the learned model can be used for downstream few-shot classification tasks, where we obtain task-specific parameters by performing semi-supervised EM on the latent representations of the support and query set, and predict labels of the query set by computing aggregated posteriors. We validate our model on Omniglot and Mini-ImageNet datasets by evaluating its performance on downstream few-shot classification tasks. The results show that our model obtain impressive performance gains over existing unsupervised meta-learning baselines, even outperforming supervised MAML on a certain setting.

----

## [154] Fast Geometric Projections for Local Robustness Certification

**Authors**: *Aymeric Fromherz, Klas Leino, Matt Fredrikson, Bryan Parno, Corina S. Pasareanu*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=zWy1uxjDdZJ](https://openreview.net/forum?id=zWy1uxjDdZJ)

**Abstract**:

Local robustness ensures that a model classifies all inputs within an $\ell_p$-ball consistently, which precludes various forms of adversarial inputs.
In this paper, we present a fast procedure for checking local robustness in feed-forward neural networks with piecewise-linear activation functions.
Such networks partition the input space into a set of convex polyhedral regions in which the network’s behavior is linear; 
hence, a systematic search for decision boundaries within the regions around a given input is sufficient for assessing robustness.
Crucially, we show how the regions around a point can be analyzed using simple geometric projections, thus admitting an efficient, highly-parallel GPU implementation that excels particularly for the $\ell_2$ norm, where previous work has been less effective.
Empirically we find this approach to be far more precise than many approximate verification approaches, while at the same time performing multiple orders of magnitude faster than complete verifiers, and scaling to much deeper networks.

----

## [155] Fidelity-based Deep Adiabatic Scheduling

**Authors**: *Eli Ovits, Lior Wolf*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=NECTfffOvn1](https://openreview.net/forum?id=NECTfffOvn1)

**Abstract**:

Adiabatic quantum computation is a form of computation that acts by slowly interpolating a quantum system between an easy to prepare initial state and a final state that represents a solution to a given computational problem. The choice of the interpolation schedule is critical to the performance: if at a certain time point, the evolution is too rapid, the system has a high probability to transfer to a higher energy state, which does not represent a solution to the problem. On the other hand, an evolution that is too slow leads to a loss of computation time and increases the probability of failure due to decoherence. In this work, we train deep neural models to produce optimal schedules that are conditioned on the problem at hand.  We consider two types of problem representation: the Hamiltonian form, and the Quadratic Unconstrained Binary Optimization (QUBO) form. A novel loss function that scores schedules according to their approximated success probability is introduced. We benchmark our approach on random QUBO problems, Grover search, 3-SAT, and MAX-CUT problems and show that our approach outperforms, by a sizable margin, the linear schedules as well as alternative approaches that were very recently proposed.

----

## [156] On Self-Supervised Image Representations for GAN Evaluation

**Authors**: *Stanislav Morozov, Andrey Voynov, Artem Babenko*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=NeRdBeTionN](https://openreview.net/forum?id=NeRdBeTionN)

**Abstract**:

The embeddings from CNNs pretrained on Imagenet classification are de-facto standard image representations for assessing GANs via FID, Precision and Recall measures. Despite broad previous criticism of their usage for non-Imagenet domains, these embeddings are still the top choice in most of the GAN literature.

In this paper, we advocate the usage of the state-of-the-art self-supervised representations to evaluate GANs on the established non-Imagenet benchmarks. These representations, typically obtained via contrastive learning, are shown to provide better transfer to new tasks and domains, therefore, can serve as more universal embeddings of natural images. With extensive comparison of the recent GANs on the common datasets, we show that self-supervised representations produce a more reasonable ranking of models in terms of FID/Precision/Recall, while the ranking with classification-pretrained embeddings often can be misleading.

----

## [157] Retrieval-Augmented Generation for Code Summarization via Hybrid GNN

**Authors**: *Shangqing Liu, Yu Chen, Xiaofei Xie, Jing Kai Siow, Yang Liu*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=zv-typ1gPxA](https://openreview.net/forum?id=zv-typ1gPxA)

**Abstract**:

Source code summarization aims to generate natural language summaries from structured code snippets for better understanding code functionalities. However, automatic code summarization is challenging due to the complexity of the source code and the language gap between the source code and natural language summaries. Most previous approaches either rely on retrieval-based (which can take advantage of similar examples seen from the retrieval database, but have low generalization performance) or generation-based methods (which have better generalization performance, but cannot take advantage of similar examples).
This paper proposes a novel retrieval-augmented mechanism to combine the benefits of both worlds.
Furthermore, to mitigate the limitation of Graph Neural Networks (GNNs) on capturing global graph structure information of source code, we propose a novel attention-based dynamic graph to complement the static graph representation of the source code, and design a hybrid message passing GNN for capturing both the local and global structural information. To evaluate the proposed approach, we release a new challenging benchmark, crawled from diversified large-scale open-source C projects (total 95k+ unique functions in the dataset). Our method achieves the state-of-the-art performance, improving existing methods by 1.42, 2.44 and 1.29 in terms of BLEU-4, ROUGE-L and METEOR.

----

## [158] Self-supervised Visual Reinforcement Learning with Object-centric Representations

**Authors**: *Andrii Zadaianchuk, Maximilian Seitzer, Georg Martius*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=xppLmXCbOw1](https://openreview.net/forum?id=xppLmXCbOw1)

**Abstract**:

Autonomous agents need large repertoires of skills to act reasonably on new tasks that they have not seen before. However, acquiring these skills using only a stream of high-dimensional, unstructured, and unlabeled observations is a tricky challenge for any autonomous agent. Previous methods have used variational autoencoders to encode a scene into a low-dimensional vector that can be used as a goal for an agent to discover new skills. Nevertheless, in compositional/multi-object environments it is difficult to disentangle all the factors of variation into such a fixed-length representation of the whole scene. We propose to use object-centric representations as a modular and structured observation space, which is learned with a compositional generative world model.
We show that the structure in the representations in combination with goal-conditioned attention policies helps the autonomous agent to discover and learn useful skills. These skills can be further combined to address compositional tasks like the manipulation of several different objects.

----

## [159] Identifying nonlinear dynamical systems with multiple time scales and long-range dependencies

**Authors**: *Dominik Schmidt, Georgia Koppe, Zahra Monfared, Max Beutelspacher, Daniel Durstewitz*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=_XYzwxPIQu6](https://openreview.net/forum?id=_XYzwxPIQu6)

**Abstract**:

A main theoretical interest in biology and physics is to identify the nonlinear dynamical system (DS) that generated observed time series. Recurrent Neural Networks (RNN) are, in principle, powerful enough to approximate any underlying DS, but in their vanilla form suffer from the exploding vs. vanishing gradients problem. Previous attempts to alleviate this problem resulted either in more complicated, mathematically less tractable RNN architectures, or strongly limited the dynamical expressiveness of the RNN. 
Here we address this issue by suggesting a simple regularization scheme for vanilla RNN with ReLU activation which enables them to solve long-range dependency problems and express slow time scales, while retaining a simple mathematical structure which makes their DS properties partly analytically accessible. We prove two theorems that establish a tight connection between the regularized RNN dynamics and their gradients, illustrate on DS benchmarks that our regularization approach strongly eases the reconstruction of DS which harbor widely differing time scales, and show that our method is also en par with other long-range architectures like LSTMs on several tasks.

----

## [160] Neural Topic Model via Optimal Transport

**Authors**: *He Zhao, Dinh Phung, Viet Huynh, Trung Le, Wray L. Buntine*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=Oos98K9Lv-k](https://openreview.net/forum?id=Oos98K9Lv-k)

**Abstract**:

Recently, Neural Topic Models (NTMs) inspired by variational autoencoders have obtained increasingly research interest due to their promising results on text analysis. However, it is usually hard for existing NTMs to achieve good document representation and coherent/diverse topics at the same time. Moreover, they often degrade their performance severely on short documents. The requirement of reparameterisation could also comprise their training quality and model flexibility. To address these shortcomings, we present a new neural topic model via the theory of optimal transport (OT). Specifically, we propose to learn the topic distribution of a document by directly minimising its OT distance to the document's word distributions. Importantly, the cost matrix of the OT distance models the weights between topics and words, which is constructed by the distances between topics and words in an embedding space. Our proposed model can be trained efficiently with a differentiable loss. Extensive experiments show that our framework significantly outperforms the state-of-the-art NTMs on discovering more coherent and diverse topics and deriving better document representations for both regular and short texts.

----

## [161] Memory Optimization for Deep Networks

**Authors**: *Aashaka Shah, Chao-Yuan Wu, Jayashree Mohan, Vijay Chidambaram, Philipp Krähenbühl*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=bnY0jm4l59](https://openreview.net/forum?id=bnY0jm4l59)

**Abstract**:

Deep learning is slowly, but steadily, hitting a memory bottleneck. While the tensor computation in top-of-the-line GPUs increased by $32\times$ over the last five years, the total available memory only grew by $2.5\times$. This prevents researchers from exploring larger architectures, as training large networks requires more memory for storing intermediate outputs. In this paper, we present MONeT, an automatic framework that minimizes both the memory footprint and computational overhead of deep networks. MONeT jointly optimizes the checkpointing schedule and the implementation of various operators. MONeT is able to outperform all prior hand-tuned operations as well as automated checkpointing. MONeT reduces the overall memory requirement by $3\times$ for various PyTorch models, with a 9-16$\%$ overhead in computation. For the same computation cost, MONeT requires 1.2-1.8$\times$ less memory than current state-of-the-art automated checkpointing frameworks. Our code will be made publicly available upon acceptance.

----

## [162] Stabilized Medical Image Attacks

**Authors**: *Gege Qi, Lijun Gong, Yibing Song, Kai Ma, Yefeng Zheng*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=QfTXQiGYudJ](https://openreview.net/forum?id=QfTXQiGYudJ)

**Abstract**:

Convolutional Neural Networks (CNNs) have advanced existing medical systems for automatic disease diagnosis. However, a threat to these systems arises that adversarial attacks make CNNs vulnerable. Inaccurate diagnosis results make a negative influence on human healthcare. There is a need to investigate potential adversarial attacks to robustify deep medical diagnosis systems. On the other side, there are several modalities of medical images (e.g., CT, fundus, and endoscopic image) of which each type is significantly different from others. It is more challenging to generate adversarial perturbations for different types of medical images. In this paper, we propose an image-based medical adversarial attack method to consistently produce adversarial perturbations on medical images. The objective function of our method consists of a loss deviation term and a loss stabilization term. The loss deviation term increases the divergence between the CNN prediction of an adversarial example and its ground truth label. Meanwhile, the loss stabilization term ensures similar CNN predictions of this example and its smoothed input. From the perspective of the whole iterations for perturbation generation, the proposed loss stabilization term exhaustively searches the perturbation space to smooth the single spot for local optimum escape. We further analyze the KL-divergence of the proposed loss function and find that the loss stabilization term makes the perturbations updated towards a fixed objective spot while deviating from the ground truth. This stabilization ensures the proposed medical attack effective for different types of medical images while producing perturbations in small variance. Experiments on several medical image analysis benchmarks including the recent COVID-19 dataset show the stability of the proposed method.

----

## [163] Quantifying Differences in Reward Functions

**Authors**: *Adam Gleave, Michael Dennis, Shane Legg, Stuart Russell, Jan Leike*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=LwEQnp6CYev](https://openreview.net/forum?id=LwEQnp6CYev)

**Abstract**:

For many tasks, the reward function is inaccessible to introspection or too complex to be specified procedurally, and must instead be learned from user data. Prior work has evaluated learned reward functions by evaluating policies optimized for the learned reward. However, this method cannot distinguish between the learned reward function failing to reflect user preferences and the policy optimization process failing to optimize the learned reward. Moreover, this method can only tell us about behavior in the evaluation environment, but the reward may incentivize very different behavior in even a slightly different deployment environment. To address these problems, we introduce the Equivalent-Policy Invariant Comparison (EPIC) distance to quantify the difference between two reward functions directly, without a policy optimization step. We prove EPIC is invariant on an equivalence class of reward functions that always induce the same optimal policy. Furthermore, we find EPIC can be efficiently approximated and is more robust than baselines to the choice of coverage distribution. Finally, we show that EPIC distance bounds the regret of optimal policies even under different transition dynamics, and we confirm empirically that it predicts policy training success. Our source code is available at https://github.com/HumanCompatibleAI/evaluating-rewards.

----

## [164] MARS: Markov Molecular Sampling for Multi-objective Drug Discovery

**Authors**: *Yutong Xie, Chence Shi, Hao Zhou, Yuwei Yang, Weinan Zhang, Yong Yu, Lei Li*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=kHSu4ebxFXY](https://openreview.net/forum?id=kHSu4ebxFXY)

**Abstract**:

Searching for novel molecules with desired chemical properties is crucial in drug discovery. Existing work focuses on developing neural models to generate either molecular sequences or chemical graphs. However, it remains a big challenge to find novel and diverse compounds satisfying several properties. In this paper, we propose MARS, a method for multi-objective drug molecule discovery. MARS is based on the idea of generating the chemical candidates by iteratively editing fragments of molecular graphs. To search for high-quality candidates, it employs Markov chain Monte Carlo sampling (MCMC) on molecules with an annealing scheme and an adaptive proposal. To further improve sample efficiency, MARS uses a graph neural network (GNN) to represent and select candidate edits, where the GNN is trained on-the-fly with samples from MCMC. Experiments show that MARS achieves state-of-the-art performance in various multi-objective settings where molecular bio-activity, drug-likeness, and synthesizability are considered. Remarkably, in the most challenging setting where all four objectives are simultaneously optimized, our approach outperforms previous methods significantly in comprehensive evaluations. The code is available at https://github.com/yutxie/mars.

----

## [165] Gauge Equivariant Mesh CNNs: Anisotropic convolutions on geometric graphs

**Authors**: *Pim de Haan, Maurice Weiler, Taco Cohen, Max Welling*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=Jnspzp-oIZE](https://openreview.net/forum?id=Jnspzp-oIZE)

**Abstract**:

A common approach to define convolutions on meshes is to interpret them as a graph and apply graph convolutional networks (GCNs).  Such GCNs utilize isotropic kernels and are therefore insensitive to the relative orientation of vertices and thus to the geometry of the mesh as a whole. We propose Gauge Equivariant Mesh CNNs which generalize GCNs to apply anisotropic gauge equivariant kernels. Since the resulting features carry orientation information, we introduce a geometric message passing scheme defined by parallel transporting features over mesh edges. Our experiments validate the significantly improved expressivity of the proposed model over conventional GCNs and other methods.

----

## [166] RMSprop converges with proper hyper-parameter

**Authors**: *Naichen Shi, Dawei Li, Mingyi Hong, Ruoyu Sun*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=3UDSdyIcBDA](https://openreview.net/forum?id=3UDSdyIcBDA)

**Abstract**:

Despite the existence of divergence examples, RMSprop remains 
one of the most popular algorithms in machine learning. Towards closing the gap between theory and practice, we prove that RMSprop converges with proper choice of hyper-parameters under certain conditions. More specifically, we prove that when the hyper-parameter $\beta_2$ is close enough to $1$, RMSprop and its random shuffling version converge to a bounded region in general, and to critical points in the interpolation regime. It is worth mentioning that our results do not depend on  ``bounded gradient"  assumption, which is often the key assumption utilized by existing theoretical work for Adam-type adaptive gradient method. Removing this assumption allows us to establish a phase transition from divergence to non-divergence for RMSprop. 

Finally, based on our theory, we conjecture that in practice there is a critical threshold $\sf{\beta_2^*}$, such that RMSprop generates reasonably good results only if $1>\beta_2\ge \sf{\beta_2^*}$. We provide empirical evidence for such a phase transition in our numerical experiments.

----

## [167] Revisiting Dynamic Convolution via Matrix Decomposition

**Authors**: *Yunsheng Li, Yinpeng Chen, Xiyang Dai, Mengchen Liu, Dongdong Chen, Ye Yu, Lu Yuan, Zicheng Liu, Mei Chen, Nuno Vasconcelos*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=YwpZmcAehZ](https://openreview.net/forum?id=YwpZmcAehZ)

**Abstract**:

Recent research in dynamic convolution shows substantial performance boost for efficient CNNs, due to the adaptive aggregation of K static convolution kernels. It has two limitations: (a) it increases the number of convolutional weights by K-times, and (b) the joint optimization of dynamic attention and static convolution kernels is challenging. In this paper, we revisit it from a new perspective of matrix decomposition and reveal the key issue is that dynamic convolution applies dynamic attention over channel groups after projecting into a higher dimensional latent space. To address this issue, we propose dynamic channel fusion to replace dynamic attention over channel groups. Dynamic channel fusion not only enables significant dimension reduction of the latent space, but also mitigates the joint optimization difficulty. As a result, our method is easier to train and requires significantly fewer parameters without sacrificing accuracy. Source code is at https://github.com/liyunsheng13/dcd.

----

## [168] Explainable Deep One-Class Classification

**Authors**: *Philipp Liznerski, Lukas Ruff, Robert A. Vandermeulen, Billy Joe Franks, Marius Kloft, Klaus-Robert Müller*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=A5VV3UyIQz](https://openreview.net/forum?id=A5VV3UyIQz)

**Abstract**:

Deep one-class classification variants for anomaly detection learn a mapping that concentrates nominal samples in feature space causing anomalies to be mapped away. Because this transformation is highly non-linear, finding interpretations poses a significant challenge. In this paper we present an explainable deep one-class classification method, Fully Convolutional Data Description (FCDD), where the mapped samples are themselves also an explanation heatmap. FCDD yields competitive detection performance and provides reasonable explanations on common anomaly detection benchmarks with CIFAR-10 and ImageNet. On MVTec-AD, a recent manufacturing dataset offering ground-truth anomaly maps, FCDD sets a new state of the art in the unsupervised setting. Our method can incorporate ground-truth anomaly maps during training and using even a few of these (~5) improves performance significantly. Finally, using FCDD's explanations we demonstrate the vulnerability of deep one-class classification models to spurious image features such as image watermarks.

----

## [169] Taking Notes on the Fly Helps Language Pre-Training

**Authors**: *Qiyu Wu, Chen Xing, Yatao Li, Guolin Ke, Di He, Tie-Yan Liu*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=lU5Rs_wCweN](https://openreview.net/forum?id=lU5Rs_wCweN)

**Abstract**:

How to make unsupervised language pre-training more efficient and less resource-intensive is an important research direction in NLP. In this paper, we focus on improving the efficiency of language pre-training methods through providing better data utilization. It is well-known that in language data corpus, words follow a heavy-tail distribution. A large proportion of words appear only very few times and the embeddings of rare words are usually poorly optimized. We argue that such embeddings carry inadequate semantic signals, which could make the data utilization inefficient and slow down the pre-training of the entire model. To mitigate this problem, we propose Taking Notes on the Fly (TNF), which takes notes for rare words on the fly during pre-training to help the model understand them when they occur next time. Specifically, TNF maintains a note dictionary and saves a rare word's contextual information in it as notes when the rare word occurs in a sentence. When the same rare word occurs again during training, the note information saved beforehand can be employed to enhance the semantics of the current sentence. By doing so, TNF provides a better data utilization since cross-sentence information is employed to cover the inadequate semantics caused by rare words in the sentences. We implement TNF on both BERT and ELECTRA to check its efficiency and effectiveness.  Experimental results show that TNF's training time is 60% less than its backbone pre-training models when reaching the same performance.  When trained with same number of iterations, TNF outperforms its backbone methods on most of downstream tasks and the average GLUE score. Code is attached in the supplementary material.

----

## [170] Mixed-Features Vectors and Subspace Splitting

**Authors**: *Alejandro Pimentel-Alarcón, Daniel L. Pimentel-Alarcón*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=l-LGlk4Yl6G](https://openreview.net/forum?id=l-LGlk4Yl6G)

**Abstract**:

Motivated by metagenomics, recommender systems, dictionary learning, and related problems, this paper introduces subspace splitting(SS): the task of clustering the entries of what we call amixed-features vector, that is, a vector whose subsets of coordinates agree with a collection of subspaces. We derive precise identifiability conditions under which SS is well-posed, thus providing the first fundamental theory for this problem. We also propose the first three practical SS algorithms, each with advantages and disadvantages: a random sampling method , a projection-based greedy heuristic , and an alternating Lloyd-type algorithm ; all allow noise, outliers, and missing data. Our extensive experiments outline the performance of our algorithms, and in lack of other SS algorithms, for reference we compare against methods for tightly related problems, like robust matched subspace detection and maximum feasible subsystem, which are special simpler cases of SS.

----

## [171] Neural Pruning via Growing Regularization

**Authors**: *Huan Wang, Can Qin, Yulun Zhang, Yun Fu*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=o966_Is_nPA](https://openreview.net/forum?id=o966_Is_nPA)

**Abstract**:

Regularization has long been utilized to learn sparsity in deep neural network pruning. However, its role is mainly explored in the small penalty strength regime. In this work, we extend its application to a new scenario where the regularization grows large gradually to tackle two central problems of pruning: pruning schedule and weight importance scoring. (1) The former topic is newly brought up in this work, which we find critical to the pruning performance while receives little research attention. Specifically, we propose an L2 regularization variant with rising penalty factors and show it can bring significant accuracy gains compared with its one-shot counterpart, even when the same weights are removed. (2) The growing penalty scheme also brings us an approach to exploit the Hessian information for more accurate pruning without knowing their specific values, thus not bothered by the common Hessian approximation problems. Empirically, the proposed algorithms are easy to implement and scalable to large datasets and networks in both structured and unstructured pruning. Their effectiveness is demonstrated with modern deep neural networks on the CIFAR and ImageNet datasets, achieving competitive results compared to many state-of-the-art algorithms. Our code and trained models are publicly available at https://github.com/mingsun-tse/regularization-pruning.

----

## [172] Practical Massively Parallel Monte-Carlo Tree Search Applied to Molecular Design

**Authors**: *Xiufeng Yang, Tanuj Kr Aasawat, Kazuki Yoshizoe*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=6k7VdojAIK](https://openreview.net/forum?id=6k7VdojAIK)

**Abstract**:

It is common practice to use large computational resources to train neural networks, known from many examples, such as reinforcement learning applications. However, while massively parallel computing is often used for training models, it is rarely used to search solutions for combinatorial optimization problems. This paper proposes a novel massively parallel Monte-Carlo Tree Search (MP-MCTS) algorithm that works efficiently for a 1,000 worker scale on a distributed memory environment using multiple compute nodes and applies it to molecular design. This paper is the first work that applies distributed MCTS to a real-world and non-game problem. Existing works on large-scale parallel MCTS show efficient scalability in terms of the number of rollouts up to 100 workers. Still, they suffer from the degradation in the quality of the solutions. MP-MCTS maintains the search quality at a larger scale. By running MP-MCTS on 256 CPU cores for only 10 minutes, we obtained candidate molecules with similar scores to non-parallel MCTS running for 42 hours. Moreover, our results based on parallel MCTS (combined with a simple RNN model) significantly outperform existing state-of-the-art work. Our method is generic and is expected to speed up other applications of MCTS.

----

## [173] Empirical Analysis of Unlabeled Entity Problem in Named Entity Recognition

**Authors**: *Yangming Li, Lemao Liu, Shuming Shi*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=5jRVa89sZk](https://openreview.net/forum?id=5jRVa89sZk)

**Abstract**:

In many scenarios, named entity recognition (NER) models severely suffer from unlabeled entity problem, where the entities of a sentence may not be fully annotated. Through empirical studies performed on synthetic datasets, we find two causes of performance degradation. One is the reduction of annotated entities and the other is treating unlabeled entities as negative instances. The first cause has less impact than the second one and can be mitigated by adopting pretraining language models. The second cause seriously misguides a model in training and greatly affects its performances. Based on the above observations, we propose a general approach, which can almost eliminate the misguidance brought by unlabeled entities. The key idea is to use negative sampling that, to a large extent, avoids training NER models with unlabeled entities. Experiments on synthetic datasets and real-world datasets show that our model is robust to unlabeled entity problem and surpasses prior baselines. On well-annotated datasets, our model is competitive with the state-of-the-art method.

----

## [174] Deep Networks and the Multiple Manifold Problem

**Authors**: *Sam Buchanan, Dar Gilboa, John Wright*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=O-6Pm_d_Q-](https://openreview.net/forum?id=O-6Pm_d_Q-)

**Abstract**:

We study the multiple manifold problem, a binary classification task modeled on applications in machine vision, in which a deep fully-connected neural network is trained to separate two low-dimensional submanifolds of the unit sphere. We provide an analysis of the one-dimensional case, proving for a simple manifold configuration that when the network depth $L$ is large relative to certain geometric and statistical properties of the data, the network width $n$ grows as a sufficiently large polynomial in $L$, and the number of i.i.d. samples from the manifolds is polynomial in $L$, randomly-initialized gradient descent rapidly learns to classify the two manifolds perfectly with high probability. Our analysis demonstrates concrete benefits of depth and width in the context of a practically-motivated model problem: the depth acts as a fitting resource, with larger depths corresponding to smoother networks that can more readily separate the class manifolds, and the width acts as a statistical resource, enabling concentration of the randomly-initialized network and its gradients. The argument centers around the "neural tangent kernel" of Jacot et al. and its role in the nonasymptotic analysis of training overparameterized neural networks; to this literature, we contribute essentially optimal rates of concentration for the neural tangent kernel of deep fully-connected ReLU networks, requiring width $n \geq L\,\mathrm{poly}(d_0)$ to achieve uniform concentration of the initial kernel over a $d_0$-dimensional submanifold of the unit sphere $\mathbb{S}^{n_0-1}$, and a nonasymptotic framework for establishing generalization of networks trained in the "NTK regime" with structured data. The proof makes heavy use of martingale concentration to optimally treat statistical dependencies across layers of the initial random network. This approach should be of use in establishing similar results for other network architectures.

----

## [175] Knowledge distillation via softmax regression representation learning

**Authors**: *Jing Yang, Brais Martínez, Adrian Bulat, Georgios Tzimiropoulos*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=ZzwDy_wiWv](https://openreview.net/forum?id=ZzwDy_wiWv)

**Abstract**:

This paper addresses the problem of model compression via knowledge distillation. We advocate for a method that optimizes the output feature of the penultimate layer of the student network and hence is directly related to representation learning. Previous distillation methods which typically impose direct feature matching between the student and the teacher do not take into account the classification problem at hand. On the contrary, our distillation method decouples representation learning and classification and utilizes the teacher's pre-trained classifier to train the student's penultimate layer feature. In particular, for the same input image, we wish the teacher's and student's feature to produce the same output when passed through the teacher's classifier which is achieved with a simple $L_2$ loss. Our method is extremely simple to implement and straightforward to train and is shown to consistently outperform previous state-of-the-art methods over a large set of experimental settings including different (a) network architectures, (b) teacher-student capacities, (c) datasets, and (d) domains. The code will be available at \url{https://github.com/jingyang2017/KD_SRRL}.

----

## [176] Nearest Neighbor Machine Translation

**Authors**: *Urvashi Khandelwal, Angela Fan, Dan Jurafsky, Luke Zettlemoyer, Mike Lewis*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=7wCBOfJ8hJM](https://openreview.net/forum?id=7wCBOfJ8hJM)

**Abstract**:

We introduce $k$-nearest-neighbor machine translation ($k$NN-MT), which predicts tokens with a nearest-neighbor classifier over a large datastore of cached examples, using representations from a neural translation model for similarity search. This approach requires no additional training and scales to give the decoder direct access to billions of examples at test time, resulting in a highly expressive model that consistently improves performance across many settings. Simply adding nearest-neighbor search improves a state-of-the-art German-English translation model by 1.5 BLEU. $k$NN-MT allows a single model to be adapted to diverse domains by using a domain-specific datastore, improving results by an average of 9.2 BLEU over zero-shot transfer, and achieving new state-of-the-art results---without training on these domains. A massively multilingual model can also be specialized for particular language pairs, with improvements of 3 BLEU for translating from English into German and Chinese. Qualitatively, $k$NN-MT is easily interpretable; it combines source and target context to retrieve highly relevant examples.

----

## [177] WrapNet: Neural Net Inference with Ultra-Low-Precision Arithmetic

**Authors**: *Renkun Ni, Hong-Min Chu, Oscar Castañeda, Ping-yeh Chiang, Christoph Studer, Tom Goldstein*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=3SqrRe8FWQ-](https://openreview.net/forum?id=3SqrRe8FWQ-)

**Abstract**:

Low-precision neural networks represent both weights and activations with few bits, drastically reducing the cost of multiplications. Meanwhile, these products are accumulated using high-precision (typically 32-bit) additions.  Additions dominate the arithmetic complexity of inference in quantized (e.g., binary) nets, and high precision is needed to avoid overflow. To further optimize inference, we propose WrapNet, an architecture that adapts neural networks to use low-precision (8-bit) additions while achieving classification accuracy comparable to their 32-bit counterparts. We achieve resilience to low-precision accumulation by inserting a cyclic activation layer that makes results invariant to overflow. We demonstrate the efficacy of our approach using both software and hardware platforms.

----

## [178] Wandering within a world: Online contextualized few-shot learning

**Authors**: *Mengye Ren, Michael Louis Iuzzolino, Michael Curtis Mozer, Richard S. Zemel*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=oZIvHV04XgC](https://openreview.net/forum?id=oZIvHV04XgC)

**Abstract**:

We aim to bridge the gap between typical human and machine-learning environments by extending the standard framework of few-shot learning to an online, continual setting. In this setting, episodes do not have separate training and testing phases, and instead models are evaluated online while learning novel classes. As in the real world, where the presence of spatiotemporal context helps us retrieve learned skills in the past, our online few-shot learning setting also features an underlying context that changes throughout time. Object classes are correlated within a context and inferring the correct context can lead to better performance. Building upon this setting, we propose a new few-shot learning dataset based on large scale indoor imagery that mimics the visual
experience of an agent wandering within a world. Furthermore, we convert popular few-shot learning approaches into online versions and we also propose a new model that can make use of spatiotemporal contextual information from the recent past.

----

## [179] Few-Shot Learning via Learning the Representation, Provably

**Authors**: *Simon Shaolei Du, Wei Hu, Sham M. Kakade, Jason D. Lee, Qi Lei*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=pW2Q2xLwIMD](https://openreview.net/forum?id=pW2Q2xLwIMD)

**Abstract**:

This paper studies few-shot learning via representation learning, where one uses $T$ source tasks with $n_1$ data per task to learn a representation in order to reduce the sample complexity of a target task for which there is only $n_2 (\ll n_1)$ data. Specifically, we focus on the setting where there exists a good common representation between source and target, and our goal is to understand how much a sample size reduction is possible. First, we study the setting where this common representation is low-dimensional and provide a risk bound of $\tilde{O}(\frac{dk}{n_1T} + \frac{k}{n_2})$ on the target task for the linear representation class; here $d$ is the ambient input dimension and $k (\ll d)$ is the dimension of the representation. This result bypasses the $\Omega(\frac{1}{T})$ barrier under the i.i.d. task assumption, and can capture the desired property that all $n_1T$ samples from source tasks can be \emph{pooled} together for representation learning. We further extend this result to handle a general representation function class and obtain a similar result. Next, we consider the setting where the common representation may be high-dimensional but is capacity-constrained (say in norm); here, we again demonstrate the advantage of representation learning in both high-dimensional linear regression and neural networks, and show that representation learning can fully utilize all $n_1T$ samples from source tasks.

----

## [180] AdaGCN: Adaboosting Graph Convolutional Networks into Deep Models

**Authors**: *Ke Sun, Zhanxing Zhu, Zhouchen Lin*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=QkRbdiiEjM](https://openreview.net/forum?id=QkRbdiiEjM)

**Abstract**:

The design of deep graph models still remains to be investigated and the crucial part is how to explore and exploit the knowledge from different hops of neighbors in an efficient way. In this paper, we propose a novel RNN-like deep graph neural network architecture by incorporating AdaBoost into the computation of network; and the proposed graph convolutional network called AdaGCN~(Adaboosting Graph Convolutional Network) has the ability to efficiently extract knowledge from high-order neighbors of current nodes and then integrates knowledge from different hops of neighbors into the network in an Adaboost way. Different from other graph neural networks that directly stack many graph convolution layers, AdaGCN shares the same base neural network architecture among all ``layers'' and is recursively optimized, which is similar to an RNN. Besides, We also theoretically established the connection between AdaGCN and existing graph convolutional methods, presenting the benefits of our proposal. Finally, extensive experiments demonstrate the consistent state-of-the-art prediction performance on graphs across different label rates and the computational advantage of our approach AdaGCN~\footnote{Code is available at \url{https://github.com/datake/AdaGCN}.}.

----

## [181] MultiModalQA: complex question answering over text, tables and images

**Authors**: *Alon Talmor, Ori Yoran, Amnon Catav, Dan Lahav, Yizhong Wang, Akari Asai, Gabriel Ilharco, Hannaneh Hajishirzi, Jonathan Berant*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=ee6W5UgQLa](https://openreview.net/forum?id=ee6W5UgQLa)

**Abstract**:

When answering complex questions, people can seamlessly combine information from visual, textual and tabular sources. 
While interest in models that reason over multiple pieces of evidence has surged in recent years, there has been relatively little work on question answering models that reason across multiple modalities.
In this paper, we present MultiModalQA (MMQA): a challenging question answering dataset that requires joint reasoning over text, tables and images. 
We create MMQA using a new framework for generating complex multi-modal questions at scale, harvesting tables from Wikipedia, and attaching images and text paragraphs using entities that appear in each table. We then define a formal language that allows us to take questions that can be answered from a single modality, and combine them to generate cross-modal questions. Last, crowdsourcing workers take these automatically generated questions and rephrase them into more fluent language.
We create 29,918 questions through this procedure, and empirically demonstrate the necessity of a multi-modal multi-hop approach to solve our task: our multi-hop model, ImplicitDecomp, achieves an average F1 of 51.7  over cross-modal questions, substantially outperforming a strong baseline that achieves 38.2 F1, but still lags significantly behind human performance, which is at 90.1 F1.

----

## [182] Net-DNF: Effective Deep Modeling of Tabular Data

**Authors**: *Liran Katzir, Gal Elidan, Ran El-Yaniv*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=73WTGs96kho](https://openreview.net/forum?id=73WTGs96kho)

**Abstract**:

A challenging open question in deep learning is how to handle tabular data. Unlike domains such as image and natural language processing, where deep architectures prevail, there is still no widely accepted neural architecture that dominates tabular data. As a step toward bridging this gap, we present Net-DNF a novel generic architecture whose inductive bias elicits models whose structure corresponds to logical Boolean formulas in disjunctive normal form (DNF) over affine soft-threshold decision terms. Net-DNFs also promote localized decisions that are taken over small subsets of the features. We present an extensive experiments showing that Net-DNFs significantly and consistently outperform fully connected networks over tabular data. With relatively few hyperparameters, Net-DNFs open the door to practical end-to-end handling of tabular data using neural networks. We present ablation studies, which justify the design choices of Net-DNF including the inductive bias elements, namely, Boolean formulation, locality, and feature selection.

----

## [183] Optimal Regularization can Mitigate Double Descent

**Authors**: *Preetum Nakkiran, Prayaag Venkat, Sham M. Kakade, Tengyu Ma*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=7R7fAoUygoa](https://openreview.net/forum?id=7R7fAoUygoa)

**Abstract**:

Recent empirical and theoretical studies have shown that many learning algorithms -- from linear regression to neural networks -- can have test performance that is non-monotonic in quantities such the sample size and model size. This striking phenomenon, often referred to as "double descent", has raised questions of if we need to re-think our current understanding of generalization. In this work, we study whether the double-descent phenomenon can be avoided by using optimal regularization. Theoretically, we prove that for certain linear regression models with isotropic data distribution, optimally-tuned $\ell_2$ regularization achieves monotonic test performance as we grow either the sample size or the model size.
We also demonstrate empirically that optimally-tuned $\ell_2$ regularization can mitigate double descent for more general models, including neural networks.
Our results suggest that it may also be informative to study the test risk scalings of various algorithms in the context of appropriately tuned regularization.

----

## [184] Meta Back-Translation

**Authors**: *Hieu Pham, Xinyi Wang, Yiming Yang, Graham Neubig*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=3jjmdp7Hha](https://openreview.net/forum?id=3jjmdp7Hha)

**Abstract**:

Back-translation is an effective strategy to improve the performance of Neural Machine Translation~(NMT) by generating pseudo-parallel data. However, several recent works have found that better translation quality in the pseudo-parallel data does not necessarily lead to a better final translation model, while lower-quality but diverse data often yields stronger results instead.
In this paper we propose a new way to generate pseudo-parallel data for back-translation that directly optimizes the final model performance.  Specifically, we propose a meta-learning framework where the back-translation model learns to match the forward-translation model's gradients on the development data with those on the pseudo-parallel data. In our evaluations in both the standard datasets WMT En-De'14 and WMT En-Fr'14, as well as a multilingual translation setting, our method leads to significant improvements over strong baselines.

----

## [185] Learning A Minimax Optimizer: A Pilot Study

**Authors**: *Jiayi Shen, Xiaohan Chen, Howard Heaton, Tianlong Chen, Jialin Liu, Wotao Yin, Zhangyang Wang*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=nkIDwI6oO4_](https://openreview.net/forum?id=nkIDwI6oO4_)

**Abstract**:

Solving continuous minimax optimization is of extensive practical interest, yet notoriously unstable and difficult. This paper introduces the learning to optimize(L2O) methodology to the minimax problems for the first time and addresses its accompanying unique challenges. We first present Twin-L2O, the first dedicated minimax L2O method consisting of two LSTMs for updating min and max variables separately. The decoupled design is found to facilitate learning, particularly when the min and max variables are highly asymmetric. Empirical experiments on a variety of minimax problems corroborate the effectiveness of Twin-L2O. We then discuss a crucial concern of Twin-L2O, i.e., its inevitably limited generalizability to unseen optimizees. To address this issue, we present two complementary strategies. Our first solution, Enhanced Twin-L2O, is empirically applicable for general minimax problems, by improving L2O training via leveraging curriculum learning. Our second alternative, called Safeguarded Twin-L2O, is a preliminary theoretical exploration stating that under some strong assumptions, it is possible to theoretically establish the convergence of Twin-L2O. We benchmark our algorithms on several testbed problems and compare against state-of-the-art minimax solvers. The code is available at:  https://github.com/VITA-Group/L2O-Minimax.

----

## [186] A Wigner-Eckart Theorem for Group Equivariant Convolution Kernels

**Authors**: *Leon Lang, Maurice Weiler*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=ajOrOhQOsYx](https://openreview.net/forum?id=ajOrOhQOsYx)

**Abstract**:

Group equivariant convolutional networks (GCNNs) endow classical convolutional networks with additional symmetry priors, which can lead to a considerably improved performance. Recent advances in the theoretical description of GCNNs revealed that such models can generally be understood as performing convolutions with $G$-steerable kernels, that is, kernels that satisfy an equivariance constraint themselves. While the $G$-steerability constraint has been derived, it has to date only been solved for specific use cases - a general characterization of $G$-steerable kernel spaces is still missing. This work provides such a characterization for the practically relevant case of $G$ being any compact group. Our investigation is motivated by a striking analogy between the constraints underlying steerable kernels on the one hand and spherical tensor operators from quantum mechanics on the other hand. By generalizing the famous Wigner-Eckart theorem for spherical tensor operators, we prove that steerable kernel spaces are fully understood and parameterized in terms of 1) generalized reduced matrix elements, 2) Clebsch-Gordan coefficients, and 3) harmonic basis functions on homogeneous spaces.

----

## [187] Viewmaker Networks: Learning Views for Unsupervised Representation Learning

**Authors**: *Alex Tamkin, Mike Wu, Noah D. Goodman*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=enoVQWLsfyL](https://openreview.net/forum?id=enoVQWLsfyL)

**Abstract**:

Many recent methods for unsupervised representation learning train models to be invariant to different "views," or distorted versions of an input. However, designing these views requires considerable trial and error by human experts, hindering widespread adoption of unsupervised representation learning methods across domains and modalities. To address this, we propose viewmaker networks: generative models that learn to produce useful views from a given input. Viewmakers are stochastic bounded adversaries: they produce views by generating and then adding an $\ell_p$-bounded perturbation to the input, and are trained adversarially with respect to the main encoder network. Remarkably, when pretraining on CIFAR-10, our learned views enable comparable transfer accuracy to the well-tuned SimCLR augmentations---despite not including transformations like cropping or color jitter. Furthermore, our learned views significantly outperform baseline augmentations on speech recordings (+9 points on average) and wearable sensor data (+17 points on average). Viewmaker views can also be combined with handcrafted views: they improve robustness to common image corruptions and can increase transfer performance in cases where handcrafted views are less explored. These results suggest that viewmakers may provide a path towards more general representation learning algorithms---reducing the domain expertise and effort needed to pretrain on a much wider set of domains. Code is available at https://github.com/alextamkin/viewmaker.

----

## [188] Scalable Transfer Learning with Expert Models

**Authors**: *Joan Puigcerver, Carlos Riquelme Ruiz, Basil Mustafa, Cédric Renggli, André Susano Pinto, Sylvain Gelly, Daniel Keysers, Neil Houlsby*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=23ZjUGpjcc](https://openreview.net/forum?id=23ZjUGpjcc)

**Abstract**:

Transfer of pre-trained representations can improve sample efficiency and reduce computational requirements for new tasks. However, representations used for transfer are usually generic, and are not tailored to a particular distribution of downstream tasks. We explore the use of expert representations for transfer with a simple, yet effective, strategy. We train a diverse set of experts by exploiting existing label structures, and use cheap-to-compute performance proxies to select the relevant expert for each target task. This strategy scales the process of transferring to new tasks, since it does not revisit the pre-training data during transfer. Accordingly, it requires little extra compute per target task, and results in a speed-up of 2-3 orders of magnitude compared to competing approaches. Further, we provide an adapter-based architecture able to compress many experts into a single model. We evaluate our approach on two different data sources and demonstrate that it outperforms baselines on over 20 diverse vision tasks in both cases.

----

## [189] Negative Data Augmentation

**Authors**: *Abhishek Sinha, Kumar Ayush, Jiaming Song, Burak Uzkent, Hongxia Jin, Stefano Ermon*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=Ovp8dvB8IBH](https://openreview.net/forum?id=Ovp8dvB8IBH)

**Abstract**:

Data augmentation is often used to enlarge datasets with synthetic samples generated in accordance with the underlying data distribution. To enable a wider range of augmentations, we explore negative data augmentation strategies (NDA) that intentionally create out-of-distribution samples. We show that such negative out-of-distribution samples provide information on the support of the data distribution,  and can be leveraged for generative modeling and representation learning. We introduce a new GAN training objective where we use NDA as an additional source of synthetic data for the discriminator. We prove that under suitable conditions, optimizing the resulting objective still recovers the true data distribution but can directly bias the generator towards avoiding samples that lack the desired structure. Empirically, models trained with our method achieve improved conditional/unconditional image generation along with improved anomaly detection capabilities. Further, we incorporate the same negative data augmentation strategy in a contrastive learning framework for self-supervised representation learning on images and videos, achieving improved performance on downstream image classification, object detection, and action recognition tasks. These results suggest that prior knowledge on what does not constitute valid data is an effective form of weak supervision across a range of unsupervised learning tasks.

----

## [190] Fantastic Four: Differentiable and Efficient Bounds on Singular Values of Convolution Layers

**Authors**: *Sahil Singla, Soheil Feizi*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=JCRblSgs34Z](https://openreview.net/forum?id=JCRblSgs34Z)

**Abstract**:

In deep neural networks, the spectral norm of the Jacobian of a layer bounds the factor by which the norm of a signal changes during forward/backward propagation. Spectral norm regularizations have been shown to improve generalization, robustness and optimization of deep learning methods. Existing methods to compute the spectral norm of convolution layers either rely on heuristics that are efficient in computation but lack guarantees or are theoretically-sound but computationally expensive. In this work, we obtain the best of both worlds by deriving {\it four} provable upper bounds on the spectral norm of a standard 2D multi-channel convolution layer. These bounds are differentiable and can be computed efficiently during training with negligible overhead. One of these bounds is in fact the popular heuristic method of Miyato et al. (multiplied by a constant factor depending on filter sizes). Each of these four bounds can achieve the tightest gap depending on convolution filters. Thus, we propose to use the minimum of these four bounds as a tight, differentiable and efficient upper bound on the spectral norm of convolution layers. Moreover, our spectral bound is an effective regularizer and can be used to bound either the lipschitz constant or curvature values (eigenvalues of the Hessian) of neural networks. Through experiments on MNIST and CIFAR-10, we demonstrate the effectiveness of our spectral bound in improving generalization and robustness of deep networks.

----

## [191] CoDA: Contrast-enhanced and Diversity-promoting Data Augmentation for Natural Language Understanding

**Authors**: *Yanru Qu, Dinghan Shen, Yelong Shen, Sandra Sajeev, Weizhu Chen, Jiawei Han*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=Ozk9MrX1hvA](https://openreview.net/forum?id=Ozk9MrX1hvA)

**Abstract**:

Data augmentation has been demonstrated as an effective strategy for improving model generalization and data efficiency.  However, due to the discrete nature of natural language, designing label-preserving transformations for text data tends to be more challenging. In this paper, we propose a novel data augmentation frame-work dubbed CoDA, which synthesizes diverse and informative augmented examples by integrating multiple transformations organically.  Moreover, a contrastive regularization is introduced to capture the global relationship among all the data samples.  A momentum encoder along with a memory bank is further leveraged to better estimate the contrastive loss. To verify the effectiveness of the proposed framework, we apply CoDA to Transformer-based models on a wide range of natural language understanding tasks. On the GLUE benchmark, CoDA gives rise to an average improvement of 2.2%while applied to the Roberta-large model. More importantly, it consistently exhibits stronger results relative to several competitive data augmentation and adversarial training baselines (including the low-resource settings). Extensive experiments show that the proposed contrastive objective can be flexibly combined with various data augmentation approaches to further boost their performance, highlighting the wide applicability of the CoDA framework.

----

## [192] Teaching with Commentaries

**Authors**: *Aniruddh Raghu, Maithra Raghu, Simon Kornblith, David Duvenaud, Geoffrey E. Hinton*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=4RbdgBh9gE](https://openreview.net/forum?id=4RbdgBh9gE)

**Abstract**:

Effective training of deep neural networks can be challenging, and there remain many open questions on how to best learn these models. Recently developed methods to improve neural network training examine teaching: providing learned information during the training process to improve downstream model performance. In this paper, we take steps towards extending the scope of teaching. We propose a flexible teaching framework using commentaries,  learned meta-information helpful for training on a particular task. We present gradient-based methods to learn commentaries, leveraging recent work on implicit differentiation for scalability. We explore diverse applications of commentaries, from weighting training examples, to parameterising label-dependent data augmentation policies, to representing attention masks that highlight salient image regions. We find that commentaries can improve training speed and/or performance, and provide insights about the dataset and training process. We also observe that commentaries generalise: they can be reused when training new models to obtain performance benefits, suggesting a use-case where commentaries are stored with a dataset and leveraged in future for improved model training.

----

## [193] MixKD: Towards Efficient Distillation of Large-scale Language Models

**Authors**: *Kevin J. Liang, Weituo Hao, Dinghan Shen, Yufan Zhou, Weizhu Chen, Changyou Chen, Lawrence Carin*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=UFGEelJkLu5](https://openreview.net/forum?id=UFGEelJkLu5)

**Abstract**:

Large-scale language models have recently demonstrated impressive empirical performance. Nevertheless, the improved results are attained at the price of bigger models, more power consumption, and slower inference, which hinder their applicability to low-resource (both memory and computation) platforms. Knowledge distillation (KD) has been demonstrated as an effective framework for compressing such big models. However, large-scale neural network systems are prone to memorize training instances, and thus tend to make inconsistent predictions when the data distribution is altered slightly. Moreover, the student model has few opportunities to request useful information from the teacher model when there is limited task-specific data available. To address these issues, we propose MixKD, a data-agnostic distillation framework that leverages mixup, a simple yet efficient data augmentation approach, to endow the resulting model with stronger generalization ability. Concretely, in addition to the original training examples, the student model is encouraged to mimic the teacher's behavior on the linear interpolation of example pairs as well. We prove from a theoretical perspective that under reasonable conditions MixKD gives rise to a smaller gap between the generalization error and the empirical error. To verify its effectiveness, we conduct experiments on the GLUE benchmark, where MixKD consistently leads to significant gains over the standard KD training, and outperforms several competitive baselines. Experiments under a limited-data setting and ablation studies further demonstrate the advantages of the proposed approach.

----

## [194] FairFil: Contrastive Neural Debiasing Method for Pretrained Text Encoders

**Authors**: *Pengyu Cheng, Weituo Hao, Siyang Yuan, Shijing Si, Lawrence Carin*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=N6JECD-PI5w](https://openreview.net/forum?id=N6JECD-PI5w)

**Abstract**:

Pretrained text encoders, such as BERT, have been applied increasingly in various natural language processing (NLP) tasks, and have recently demonstrated significant performance gains. However, recent studies have demonstrated the existence of social bias in these pretrained NLP models. Although prior works have made progress on word-level debiasing, improved sentence-level fairness of pretrained encoders still lacks exploration. In this paper, we proposed the first neural debiasing method for a pretrained sentence encoder, which transforms the pretrained encoder outputs into debiased representations via a fair filter (FairFil) network. To learn the FairFil, we introduce a contrastive learning framework that not only minimizes the correlation between filtered embeddings and bias words but also preserves rich semantic information of the original sentences. On real-world datasets, our FairFil effectively reduces the bias degree of pretrained text encoders, while continuously showing desirable performance on downstream tasks. Moreover, our post hoc method does not require any retraining of the text encoders, further enlarging FairFil's application space.

----

## [195] Probabilistic Numeric Convolutional Neural Networks

**Authors**: *Marc Anton Finzi, Roberto Bondesan, Max Welling*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=T1XmO8ScKim](https://openreview.net/forum?id=T1XmO8ScKim)

**Abstract**:

Continuous input signals like images and time series that are irregularly sampled or have missing values are challenging for existing deep learning methods. Coherently defined feature representations must depend on the values in unobserved regions of the input. Drawing from the work in probabilistic numerics, we propose Probabilistic Numeric Convolutional Neural Networks which represent features as Gaussian processes, providing a probabilistic description of discretization error. We then define a convolutional layer as the evolution of a PDE defined on this GP, followed by a nonlinearity. This approach also naturally admits steerable equivariant convolutions under e.g. the rotation group. In experiments we show that our approach yields a $3\times$ reduction of error from the previous state of the art on the SuperPixel-MNIST dataset and competitive performance on the medical time series dataset PhysioNet2012.

----

## [196] Computational Separation Between Convolutional and Fully-Connected Networks

**Authors**: *Eran Malach, Shai Shalev-Shwartz*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=hkMoYYEkBoI](https://openreview.net/forum?id=hkMoYYEkBoI)

**Abstract**:

Convolutional neural networks (CNN) exhibit unmatched performance in a multitude of computer vision tasks. However, the advantage of using convolutional networks over fully-connected networks is not understood from a theoretical perspective. In this work, we show how convolutional networks can leverage locality in the data, and thus achieve a computational advantage over fully-connected networks. Specifically, we show a class of problems that can be efficiently solved using convolutional networks trained with gradient-descent, but at the same time is hard to learn using a polynomial-size fully-connected network.

----

## [197] On the Stability of Fine-tuning BERT: Misconceptions, Explanations, and Strong Baselines

**Authors**: *Marius Mosbach, Maksym Andriushchenko, Dietrich Klakow*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=nzpLWnVAyah](https://openreview.net/forum?id=nzpLWnVAyah)

**Abstract**:

Fine-tuning pre-trained transformer-based language models such as BERT has become a common practice dominating leaderboards across various NLP benchmarks. Despite the strong empirical performance of fine-tuned models, fine-tuning is an unstable process: training the same model with multiple random seeds can result in a large variance of the task performance. Previous literature (Devlin et al., 2019; Lee et al., 2020; Dodge et al., 2020) identified two potential reasons for the observed instability: catastrophic forgetting and small size of the fine-tuning datasets. In this paper, we show that both hypotheses fail to explain the fine-tuning instability. We analyze BERT, RoBERTa, and ALBERT, fine-tuned on commonly used datasets from the GLUE benchmark, and show that the observed instability is caused by optimization difficulties that lead to vanishing gradients. Additionally, we show that the remaining variance of the downstream task performance can be attributed to differences in generalization where fine-tuned models with the same training loss exhibit noticeably different test performance. Based on our analysis, we present a simple but strong baseline that makes fine-tuning BERT-based models significantly more stable than the previously proposed approaches. Code to reproduce our results is available online: https://github.com/uds-lsv/bert-stable-fine-tuning.

----

## [198] Variational Information Bottleneck for Effective Low-Resource Fine-Tuning

**Authors**: *Rabeeh Karimi Mahabadi, Yonatan Belinkov, James Henderson*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=kvhzKz-_DMF](https://openreview.net/forum?id=kvhzKz-_DMF)

**Abstract**:

While large-scale pretrained language models have obtained impressive results when fine-tuned on a wide variety of tasks, they still often suffer from overfitting in low-resource scenarios. Since such models are general-purpose feature extractors, many of these features are inevitably irrelevant for a given target task.  We propose to use Variational Information Bottleneck (VIB) to suppress irrelevant features when fine-tuning on low-resource target tasks, and show that our method successfully reduces overfitting.  Moreover, we show that our VIB model finds sentence representations that are more robust to biases in natural language inference datasets, and thereby obtains better generalization to out-of-domain datasets. Evaluation on seven low-resource datasets in different tasks shows that our method significantly improves transfer learning in low-resource scenarios, surpassing prior work. Moreover, it improves generalization on 13 out of 15 out-of-domain natural language inference benchmarks.  Our code is publicly available in https://github.com/rabeehk/vibert.

----

## [199] Witches' Brew: Industrial Scale Data Poisoning via Gradient Matching

**Authors**: *Jonas Geiping, Liam H. Fowl, W. Ronny Huang, Wojciech Czaja, Gavin Taylor, Michael Moeller, Tom Goldstein*

**Conference**: *iclr 2021*

**URL**: [https://openreview.net/forum?id=01olnfLIbD](https://openreview.net/forum?id=01olnfLIbD)

**Abstract**:

Data Poisoning attacks modify training data to maliciously control a model trained on such data.
In this work, we focus on targeted poisoning attacks which cause a reclassification of an unmodified test image and as such breach model integrity. We consider a
particularly malicious poisoning attack that is both ``from scratch" and ``clean label", meaning we analyze an attack that successfully works against new, randomly initialized models, and is nearly imperceptible to humans, all while perturbing only a small fraction of the training data. 
Previous poisoning attacks against deep neural networks in this setting have been limited in scope and success, working only in simplified settings or being prohibitively expensive for large datasets.
The central mechanism of the new attack is matching the gradient direction of malicious examples. We analyze why this works, supplement with practical considerations. and show its threat to real-world practitioners, finding that it is the first poisoning method to cause targeted misclassification in modern deep networks trained from scratch on a full-sized, poisoned ImageNet dataset.
Finally we demonstrate the limitations of existing defensive strategies against such an attack, concluding that data poisoning is a credible threat, even for large-scale deep learning systems.

----



[Go to the next page](ICLR-2021-list02.md)

[Go to the catalog section](README.md)