## [1000] Quantized Distributed Training of Large Models with Convergence Guarantees

**Authors**: *Ilia Markov, Adrian Vladu, Qi Guo, Dan Alistarh*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/markov23a.html](https://proceedings.mlr.press/v202/markov23a.html)

**Abstract**:

Communication-reduction techniques are a popular way to improve scalability in data-parallel training of deep neural networks (DNNs). The recent emergence of large language models such as GPT has created the need for new approaches to exploit data-parallelism. Among these, fully-sharded data parallel (FSDP) training is highly popular, yet it still encounters scalability bottlenecks. One reason is that applying compression techniques to FSDP is challenging: as the vast majority of the communication involves the model’s weights, direct compression alters convergence and leads to accuracy loss. We present QSDP, a variant of FSDP which supports both gradient and weight quantization with theoretical guarantees, is simple to implement and has essentially no overheads. To derive QSDP we prove that a natural modification of SGD achieves convergence even when we only maintain quantized weights, and thus the domain over which we train consists of quantized points and is, therefore, highly non-convex. We validate this approach by training GPT-family models with up to 1.3 billion parameters on a multi-node cluster. Experiments show that QSDP preserves model accuracy, while completely removing the communication bottlenecks of FSDP, providing end-to-end speedups of up to 2.2x.

----

## [1001] Efficient Transformed Gaussian Processes for Non-Stationary Dependent Multi-class Classification

**Authors**: *Juan Maroñas, Daniel Hernández-Lobato*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/maronas23a.html](https://proceedings.mlr.press/v202/maronas23a.html)

**Abstract**:

This work introduces the Efficient Transformed Gaussian Process (ETGP), a new way of creating $C$ stochastic processes characterized by: 1) the $C$ processes are non-stationary, 2) the $C$ processes are dependent by construction without needing a mixing matrix, 3) training and making predictions is very efficient since the number of Gaussian Processes (GP) operations (e.g. inverting the inducing point’s covariance matrix) do not depend on the number of processes. This makes the ETGP particularly suited for multi-class problems with a very large number of classes, which are the problems studied in this work. ETGP exploits the recently proposed Transformed Gaussian Process (TGP), a stochastic process specified by transforming a Gaussian Process using an invertible transformation. However, unlike TGP, ETGP is constructed by transforming a single sample from a GP using $C$ invertible transformations. We derive an efficient sparse variational inference algorithm for the proposed model and demonstrate its utility in 5 classification tasks which include low/medium/large datasets and a different number of classes, ranging from just a few to hundreds. Our results show that ETGP, in general, outperforms state-of-the-art methods for multi-class classification based on GPs, and has a lower computational cost (around one order of magnitude smaller).

----

## [1002] Computational Asymmetries in Robust Classification

**Authors**: *Samuele Marro, Michele Lombardi*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/marro23a.html](https://proceedings.mlr.press/v202/marro23a.html)

**Abstract**:

In the context of adversarial robustness, we make three strongly related contributions. First, we prove that while attacking ReLU classifiers is $\mathit{NP}$-hard, ensuring their robustness at training time is $\Sigma^2_P$-hard (even on a single example). This asymmetry provides a rationale for the fact that robust classifications approaches are frequently fooled in the literature. Second, we show that inference-time robustness certificates are not affected by this asymmetry, by introducing a proof-of-concept approach named Counter-Attack (CA). Indeed, CA displays a reversed asymmetry: running the defense is $\mathit{NP}$-hard, while attacking it is $\Sigma_2^P$-hard. Finally, motivated by our previous result, we argue that adversarial attacks can be used in the context of robustness certification, and provide an empirical evaluation of their effectiveness. As a byproduct of this process, we also release UG100, a benchmark dataset for adversarial attacks.

----

## [1003] Neural Network Approximations of PDEs Beyond Linearity: A Representational Perspective

**Authors**: *Tanya Marwah, Zachary Chase Lipton, Jianfeng Lu, Andrej Risteski*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/marwah23a.html](https://proceedings.mlr.press/v202/marwah23a.html)

**Abstract**:

A burgeoning line of research has developed deep neural networks capable of approximating the solutions to high dimensional PDEs, opening related lines of theoretical inquiry focused on explaining how it is that these models appear to evade the curse of dimensionality. However, most theoretical analyses thus far have been limited to linear PDEs. In this work, we take a step towards studying the representational power of neural networks for approximating solutions to nonlinear PDEs. We focus on a class of PDEs known as nonlinear elliptic variational PDEs, whose solutions minimize an Euler-Lagrange energy functional $\mathcal{E}(u) = \int_\Omega L(x, u(x), \nabla u(x)) - f(x) u(x)dx$. We show that if composing a function with Barron norm $b$ with partial derivatives of $L$ produces a function of Barron norm at most $B_L b^p$, the solution to the PDE can be $\epsilon$-approximated in the $L^2$ sense by a function with Barron norm $O\left(\left(dB_L\right)^{\max\{p \log(1/ \epsilon), p^{\log(1/\epsilon)}\}}\right)$. By a classical result due to Barron (1993), this correspondingly bounds the size of a 2-layer neural network needed to approximate the solution. Treating $p, \epsilon, B_L$ as constants, this quantity is polynomial in dimension, thus showing neural networks can evade the curse of dimensionality. Our proof technique involves neurally simulating (preconditioned) gradient in an appropriate Hilbert space, which converges exponentially fast to the solution of the PDE, and such that we can bound the increase of the Barron norm at each iterate. Our results subsume and substantially generalize analogous prior results for linear elliptic PDEs over a unit hypercube.

----

## [1004] Generative Pretraining for Black-Box Optimization

**Authors**: *Satvik Mehul Mashkaria, Siddarth Krishnamoorthy, Aditya Grover*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/mashkaria23a.html](https://proceedings.mlr.press/v202/mashkaria23a.html)

**Abstract**:

Many problems in science and engineering involve optimizing an expensive black-box function over a high-dimensional space. In the offline model-based optimization (MBO) setting, we assume access to a fixed, offline dataset for pretraining and a small budget for online function evaluations. Prior approaches seek to utilize the offline data to approximate the function or its inverse but are not sufficiently accurate far from the data distribution. We propose BONET, a generative framework for pretraining a novel model-based optimizer using offline datasets. In BONET, we train an autoregressive model on fixed-length trajectories derived from an offline dataset. We design a sampling strategy to synthesize trajectories from offline data using a simple heuristic of rolling out monotonic transitions from low-fidelity to high-fidelity samples. Empirically, we instantiate BONET using a causally masked Transformer (Radford et al., 2019) and evaluate it on Design-Bench (Trabucco et al., 2022), where we rank the best on average, outperforming state-of-the-art baselines.

----

## [1005] Improved Policy Evaluation for Randomized Trials of Algorithmic Resource Allocation

**Authors**: *Aditya Mate, Bryan Wilder, Aparna Taneja, Milind Tambe*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/mate23a.html](https://proceedings.mlr.press/v202/mate23a.html)

**Abstract**:

We consider the task of evaluating policies of algorithmic resource allocation through randomized controlled trials (RCTs). Such policies are tasked with optimizing the utilization of limited intervention resources, with the goal of maximizing the benefits derived. Evaluation of such allocation policies through RCTs proves difficult, notwithstanding the scale of the trial, because the individuals’ outcomes are inextricably interlinked through resource constraints controlling the policy decisions. Our key contribution is to present a new estimator leveraging our proposed novel concept, that involves retrospective reshuffling of participants across experimental arms at the end of an RCT. We identify conditions under which such reassignments are permissible and can be leveraged to construct counterfactual trials, whose outcomes can be accurately ascertained, for free. We prove theoretically that such an estimator is more accurate than common estimators based on sample means – we show that it returns an unbiased estimate and simultaneously reduces variance. We demonstrate the value of our approach through empirical experiments on synthetic, semisynthetic as well as real case study data and show improved estimation accuracy across the board.

----

## [1006] Multi-Fidelity Covariance Estimation in the Log-Euclidean Geometry

**Authors**: *Aimee Maurais, Terrence Alsup, Benjamin Peherstorfer, Youssef M. Marzouk*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/maurais23a.html](https://proceedings.mlr.press/v202/maurais23a.html)

**Abstract**:

We introduce a multi-fidelity estimator of covariance matrices that employs the log-Euclidean geometry of the symmetric positive-definite manifold. The estimator fuses samples from a hierarchy of data sources of differing fidelities and costs for variance reduction while guaranteeing definiteness, in contrast with previous approaches. The new estimator makes covariance estimation tractable in applications where simulation or data collection is expensive; to that end, we develop an optimal sample allocation scheme that minimizes the mean-squared error of the estimator given a fixed budget. Guaranteed definiteness is crucial to metric learning, data assimilation, and other downstream tasks. Evaluations of our approach using data from physical applications (heat conduction, fluid dynamics) demonstrate more accurate metric learning and speedups of more than one order of magnitude compared to benchmarks.

----

## [1007] Communication-Constrained Bandits under Additive Gaussian Noise

**Authors**: *Prathamesh Mayekar, Jonathan Scarlett, Vincent Y. F. Tan*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/mayekar23a.html](https://proceedings.mlr.press/v202/mayekar23a.html)

**Abstract**:

We study a distributed stochastic multi-armed bandit where a client supplies the learner with communication-constrained feedback based on the rewards for the corresponding arm pulls. In our setup, the client must encode the rewards such that the second moment of the encoded rewards is no more than $P$, and this encoded reward is further corrupted by additive Gaussian noise of variance $\sigma^2$; the learner only has access to this corrupted reward. For this setting, we derive an information-theoretic lower bound of $\Omega\left(\sqrt{\frac{KT}{\mathtt{SNR} \wedge1}} \right)$ on the minimax regret of any scheme, where $\mathtt{SNR}\coloneqq \frac{P}{\sigma^2}$, and $K$ and $T$ are the number of arms and time horizon, respectively. Furthermore, we propose a multi-phase bandit algorithm, $\mathtt{UE}\text{-}\mathtt{UCB}\text{++}$, which matches this lower bound to a minor additive factor. $\mathtt{UE}\text{-}\mathtt{UCB}\text{++}$ performs uniform exploration in its initial phases and then utilizes the upper confidence bound (UCB) bandit algorithm in its final phase. An interesting feature of $\mathtt{UE}\text{-}\mathtt{UCB}\text{++}$ is that the coarser estimates of the mean rewards formed during a uniform exploration phase help to refine the encoding protocol in the next phase, leading to more accurate mean estimates of the rewards in the subsequent phase. This positive reinforcement cycle is critical to reducing the number of uniform exploration rounds and closely matching our lower bound.

----

## [1008] Nonparametric Density Estimation under Distribution Drift

**Authors**: *Alessio Mazzetto, Eli Upfal*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/mazzetto23a.html](https://proceedings.mlr.press/v202/mazzetto23a.html)

**Abstract**:

We study nonparametric density estimation in non-stationary drift settings. Given a sequence of independent samples taken from a distribution that gradually changes in time, the goal is to compute the best estimate for the current distribution. We prove tight minimax risk bounds for both discrete and continuous smooth densities, where the minimum is over all possible estimates and the maximum is over all possible distributions that satisfy the drift constraints. Our technique handles a broad class of drift models and generalizes previous results on agnostic learning under drift.

----

## [1009] PAC-Bayesian Generalization Bounds for Adversarial Generative Models

**Authors**: *Sokhna Diarra Mbacke, Florence Clerc, Pascal Germain*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/mbacke23a.html](https://proceedings.mlr.press/v202/mbacke23a.html)

**Abstract**:

We extend PAC-Bayesian theory to generative models and develop generalization bounds for models based on the Wasserstein distance and the total variation distance. Our first result on the Wasserstein distance assumes the instance space is bounded, while our second result takes advantage of dimensionality reduction. Our results naturally apply to Wasserstein GANs and Energy-Based GANs, and our bounds provide new training objectives for these two. Although our work is mainly theoretical, we perform numerical experiments showing non-vacuous generalization bounds for Wasserstein GANs on synthetic datasets.

----

## [1010] Robustness in Multimodal Learning under Train-Test Modality Mismatch

**Authors**: *Brandon McKinzie, Vaishaal Shankar, Joseph Yitan Cheng, Yinfei Yang, Jonathon Shlens, Alexander T. Toshev*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/mckinzie23a.html](https://proceedings.mlr.press/v202/mckinzie23a.html)

**Abstract**:

Multimodal learning is defined as learning over multiple heterogeneous input modalities such as video, audio, and text. In this work, we are concerned with understanding how models behave as the type of modalities differ between training and deployment, a situation that naturally arises in many applications of multimodal learning to hardware platforms. We present a multimodal robustness framework to provide a systematic analysis of common multimodal representation learning methods. Further, we identify robustness short-comings of these approaches and propose two intervention techniques leading to $1.5\times$-$4\times$ robustness improvements on three datasets, AudioSet, Kinetics-400 and ImageNet-Captions. Finally, we demonstrate that these interventions better utilize additional modalities, if present, to achieve competitive results of $44.2$ mAP on AudioSet 20K.

----

## [1011] A Model-free Closeness-of-influence Test for Features in Supervised Learning

**Authors**: *Mohammad Mehrabi, Ryan A. Rossi*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/mehrabi23a.html](https://proceedings.mlr.press/v202/mehrabi23a.html)

**Abstract**:

Understanding the effect of a feature vector $x\in \mathbb{R}^d$ on the response value (label) $y\in \mathbb{R}$ is the cornerstone of many statistical learning problems. Ideally, it is desired to understand how a set of collected features combine together and influence the response value, but this problem is notoriously difficult, due to the high-dimensionality of data and limited number of labeled data points, among many others. In this work, we take a new perspective on this problem, and we study the question of assessing the difference of influence that the two given features have on the response value. We first propose a notion of closeness for the influence of features, and show that our definition recovers the familiar notion of the magnitude of coefficients in the parametric model. We then propose a novel method to test for the closeness of influence in general model-free supervised learning problems. Our proposed test can be used with finite number of samples with control on type I error rate, no matter the ground truth conditional law $\mathcal{L}(Y|X)$. We analyze the power of our test for two general learning problems i) linear regression, and ii) binary classification under mixture of Gaussian models, and show that under the proper choice of score function, an internal component of our test, with sufficient number of samples will achieve full statistical power. We evaluate our findings through extensive numerical simulations, specifically we adopt the datamodel framework (Ilyas, et al., 2022) for CIFAR-10 dataset to identify pairs of training samples with different influence on the trained model via optional black box training mechanisms.

----

## [1012] Stochastic Gradient Succeeds for Bandits

**Authors**: *Jincheng Mei, Zixin Zhong, Bo Dai, Alekh Agarwal, Csaba Szepesvári, Dale Schuurmans*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/mei23a.html](https://proceedings.mlr.press/v202/mei23a.html)

**Abstract**:

We show that the stochastic gradient bandit algorithm converges to a globally optimal policy at an $O(1/t)$ rate, even with a constant step size. Remarkably, global convergence of the stochastic gradient bandit algorithm has not been previously established, even though it is an old algorithm known to be applicable to bandits. The new result is achieved by establishing two novel technical findings: first, the noise of the stochastic updates in the gradient bandit algorithm satisfies a strong “growth condition” property, where the variance diminishes whenever progress becomes small, implying that additional noise control via diminishing step sizes is unnecessary; second, a form of “weak exploration” is automatically achieved through the stochastic gradient updates, since they prevent the action probabilities from decaying faster than $O(1/t)$, thus ensuring that every action is sampled infinitely often with probability $1$. These two findings can be used to show that the stochastic gradient update is already “sufficient” for bandits in the sense that exploration versus exploitation is automatically balanced in a manner that ensures almost sure convergence to a global optimum. These novel theoretical findings are further verified by experimental results.

----

## [1013] Normalizing Flows for Interventional Density Estimation

**Authors**: *Valentyn Melnychuk, Dennis Frauen, Stefan Feuerriegel*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/melnychuk23a.html](https://proceedings.mlr.press/v202/melnychuk23a.html)

**Abstract**:

Existing machine learning methods for causal inference usually estimate quantities expressed via the mean of potential outcomes (e.g., average treatment effect). However, such quantities do not capture the full information about the distribution of potential outcomes. In this work, we estimate the density of potential outcomes after interventions from observational data. For this, we propose a novel, fully-parametric deep learning method called Interventional Normalizing Flows. Specifically, we combine two normalizing flows, namely (i) a nuisance flow for estimating nuisance parameters and (ii) a target flow for parametric estimation of the density of potential outcomes. We further develop a tractable optimization objective based on a one-step bias correction for efficient and doubly robust estimation of the target flow parameters. As a result, our Interventional Normalizing Flows offer a properly normalized density estimator. Across various experiments, we demonstrate that our Interventional Normalizing Flows are expressive and highly effective, and scale well with both sample size and high-dimensional confounding. To the best of our knowledge, our Interventional Normalizing Flows are the first proper fully-parametric, deep learning method for density estimation of potential outcomes.

----

## [1014] Reprogramming Pretrained Language Models for Antibody Sequence Infilling

**Authors**: *Igor Melnyk, Vijil Chenthamarakshan, Pin-Yu Chen, Payel Das, Amit Dhurandhar, Inkit Padhi, Devleena Das*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/melnyk23a.html](https://proceedings.mlr.press/v202/melnyk23a.html)

**Abstract**:

Antibodies comprise the most versatile class of binding molecules, with numerous applications in biomedicine. Computational design of antibodies involves generating novel and diverse sequences, while maintaining structural consistency. Unique to antibodies, designing the complementarity-determining region (CDR), which determines the antigen binding affinity and specificity, creates its own unique challenges. Recent deep learning models have shown impressive results, however the limited number of known antibody sequence/structure pairs frequently leads to degraded performance, particularly lacking diversity in the generated sequences. In our work we address this challenge by leveraging Model Reprogramming (MR), which repurposes pretrained models on a source language to adapt to the tasks that are in a different language and have scarce data - where it may be difficult to train a high-performing model from scratch or effectively fine-tune an existing pre-trained model on the specific task. Specifically, we introduce ReprogBert in which a pretrained English language model is repurposed for protein sequence infilling - thus considers cross-language adaptation using less data. Results on antibody design benchmarks show that our model on low-resourced antibody sequence dataset provides highly diverse CDR sequences, up to more than a two-fold increase of diversity over the baselines, without losing structural integrity and naturalness. The generated sequences also demonstrate enhanced antigen binding specificity and virus neutralization ability. Code is available at https://github.com/IBM/ReprogBERT

----

## [1015] Superhuman Fairness

**Authors**: *Omid Memarrast, Linh Vu, Brian D. Ziebart*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/memarrast23a.html](https://proceedings.mlr.press/v202/memarrast23a.html)

**Abstract**:

The fairness of machine learning-based decisions has become an increasingly important focus in the design of supervised machine learning methods. Most fairness approaches optimize a specified trade-off between performance measure(s) (e.g., accuracy, log loss, or AUC) and fairness metric(s) (e.g., demographic parity, equalized odds). This begs the question: are the right performance-fairness trade-offs being specified? We instead re-cast fair machine learning as an imitation learning task by introducing superhuman fairness, which seeks to simultaneously outperform human decisions on multiple predictive performance and fairness measures. We demonstrate the benefits of this approach given suboptimal decisions.

----

## [1016] A Model-Based Method for Minimizing CVaR and Beyond

**Authors**: *Si Yi Meng, Robert M. Gower*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/meng23a.html](https://proceedings.mlr.press/v202/meng23a.html)

**Abstract**:

We develop a variant of the stochastic prox-linear method for minimizing the Conditional Value-at-Risk (CVaR) objective. CVaR is a risk measure focused on minimizing worst-case performance, defined as the average of the top quantile of the losses. In machine learning, such a risk measure is useful to train more robust models. Although the stochastic subgradient method (SGM) is a natural choice for minimizing the CVaR objective, we show that our stochastic prox-linear (SPL+) algorithm can better exploit the structure of the objective, while still providing a convenient closed form update. Our SPL+ method also adapts to the scaling of the loss function, which allows for easier tuning. We then specialize a general convergence theorem for SPL+ to our setting, and show that it allows for a wider selection of step sizes compared to SGM. We support this theoretical finding experimentally.

----

## [1017] Tuning Language Models as Training Data Generators for Augmentation-Enhanced Few-Shot Learning

**Authors**: *Yu Meng, Martin Michalski, Jiaxin Huang, Yu Zhang, Tarek F. Abdelzaher, Jiawei Han*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/meng23b.html](https://proceedings.mlr.press/v202/meng23b.html)

**Abstract**:

Recent studies have revealed the intriguing few-shot learning ability of pretrained language models (PLMs): They can quickly adapt to a new task when fine-tuned on a small amount of labeled data formulated as prompts, without requiring abundant task-specific annotations. Despite their promising performance, most existing few-shot approaches that only learn from the small training set still underperform fully supervised training by nontrivial margins. In this work, we study few-shot learning with PLMs from a different perspective: We first tune an autoregressive PLM on the few-shot samples and then use it as a generator to synthesize a large amount of novel training samples which augment the original training set. To encourage the generator to produce label-discriminative samples, we train it via weighted maximum likelihood where the weight of each token is automatically adjusted based on a discriminative meta-learning objective. A classification PLM can then be fine-tuned on both the few-shot and the synthetic samples with regularization for better generalization and stability. Our approach FewGen achieves an overall better result across seven classification tasks of the GLUE benchmark than existing few-shot learning methods, improving no-augmentation methods by 5+ average points, and outperforming augmentation methods by 3+ average points.

----

## [1018] On Preemption and Learning in Stochastic Scheduling

**Authors**: *Nadav Merlis, Hugo Richard, Flore Sentenac, Corentin Odic, Mathieu Molina, Vianney Perchet*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/merlis23a.html](https://proceedings.mlr.press/v202/merlis23a.html)

**Abstract**:

We study single-machine scheduling of jobs, each belonging to a job type that determines its duration distribution. We start by analyzing the scenario where the type characteristics are known and then move to two learning scenarios where the types are unknown: non-preemptive problems, where each started job must be completed before moving to another job; and preemptive problems, where job execution can be paused in the favor of moving to a different job. In both cases, we design algorithms that achieve sublinear excess cost, compared to the performance with known types, and prove lower bounds for the non-preemptive case. Notably, we demonstrate, both theoretically and through simulations, how preemptive algorithms can greatly outperform non-preemptive ones when the durations of different job types are far from one another, a phenomenon that does not occur when the type durations are known.

----

## [1019] Quantile Credit Assignment

**Authors**: *Thomas Mesnard, Wenqi Chen, Alaa Saade, Yunhao Tang, Mark Rowland, Theophane Weber, Clare Lyle, Audrunas Gruslys, Michal Valko, Will Dabney, Georg Ostrovski, Eric Moulines, Rémi Munos*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/mesnard23a.html](https://proceedings.mlr.press/v202/mesnard23a.html)

**Abstract**:

In reinforcement learning, the credit assignment problem is to distinguish luck from skill, that is, separate the inherent randomness in the environment from the controllable effects of the agent’s actions. This paper proposes two novel algorithms, Quantile Credit Assignment (QCA) and Hindsight QCA (HQCA), which incorporate distributional value estimation to perform credit assignment. QCA uses a network that predicts the quantiles of the return distribution, whereas HQCA additionally incorporates information about the future. Both QCA and HQCA have the appealing interpretation of leveraging an estimate of the quantile level of the return (interpreted as the level of "luck") in order to derive a "luck-dependent" baseline for policy gradient methods. We show theoretically that this approach gives an unbiased policy gradient estimate that can yield significant variance reductions over a standard value estimate baseline. QCA and HQCA significantly outperform prior state-of-the-art methods on a range of extremely difficult credit assignment problems.

----

## [1020] Is Consensus Acceleration Possible in Decentralized Optimization over Slowly Time-Varying Networks?

**Authors**: *Dmitry Metelev, Alexander Rogozin, Dmitry Kovalev, Alexander V. Gasnikov*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/metelev23a.html](https://proceedings.mlr.press/v202/metelev23a.html)

**Abstract**:

We consider decentralized optimization problems where one aims to minimize a sum of convex smooth objective functions distributed between nodes in the network. The links in the network can change from time to time. For the setting when the amount of changes is arbitrary, lower complexity bounds and corresponding optimal algorithms are known, and the consensus acceleration is not possible. However, in practice the magnitude of network changes may be limited. We derive lower complexity bounds for several regimes of velocity of networks changes. Moreover, we show how to obtain accelerated communication rates for a certain class of time-varying graphs using a specific consensus algorithm.

----

## [1021] Towards Theoretical Understanding of Inverse Reinforcement Learning

**Authors**: *Alberto Maria Metelli, Filippo Lazzati, Marcello Restelli*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/metelli23a.html](https://proceedings.mlr.press/v202/metelli23a.html)

**Abstract**:

Inverse reinforcement learning (IRL) denotes a powerful family of algorithms for recovering a reward function justifying the behavior demonstrated by an expert agent. A well-known limitation of IRL is the ambiguity in the choice of the reward function, due to the existence of multiple rewards that explain the observed behavior. This limitation has been recently circumvented by formulating IRL as the problem of estimating the feasible reward set, i.e., the region of the rewards compatible with the expert’s behavior. In this paper, we make a step towards closing the theory gap of IRL in the case of finite-horizon problems with a generative model. We start by formally introducing the problem of estimating the feasible reward set, the corresponding PAC requirement, and discussing the properties of particular classes of rewards. Then, we provide the first minimax lower bound on the sample complexity for the problem of estimating the feasible reward set of order ${\Omega}\left( \frac{H^3SA}{\epsilon^2} \left( \log \left(\frac{1}{\delta}\right) + S \right)\right)$, being $S$ and $A$ the number of states and actions respectively, $H$ the horizon, $\epsilon$ the desired accuracy, and $\delta$ the confidence. We analyze the sample complexity of a uniform sampling strategy (US-IRL), proving a matching upper bound up to logarithmic factors. Finally, we outline several open questions in IRL and propose future research directions.

----

## [1022] Quantum Policy Gradient Algorithm with Optimized Action Decoding

**Authors**: *Nico Meyer, Daniel D. Scherer, Axel Plinge, Christopher Mutschler, Michael J. Hartmann*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/meyer23a.html](https://proceedings.mlr.press/v202/meyer23a.html)

**Abstract**:

Quantum machine learning implemented by variational quantum circuits (VQCs) is considered a promising concept for the noisy intermediate-scale quantum computing era. Focusing on applications in quantum reinforcement learning, we propose an action decoding procedure for a quantum policy gradient approach. We introduce a quality measure that enables us to optimize the classical post-processing required for action selection, inspired by local and global quantum measurements. The resulting algorithm demonstrates a significant performance improvement in several benchmark environments. With this technique, we successfully execute a full training routine on a 5-qubit hardware device. Our method introduces only negligible classical overhead and has the potential to improve VQC-based algorithms beyond the field of quantum reinforcement learning.

----

## [1023] Training Deep Surrogate Models with Large Scale Online Learning

**Authors**: *Lucas Thibaut Meyer, Marc Schouler, Robert Alexander Caulk, Alejandro Ribés, Bruno Raffin*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/meyer23b.html](https://proceedings.mlr.press/v202/meyer23b.html)

**Abstract**:

The spatiotemporal resolution of Partial Differential Equations (PDEs) plays important roles in the mathematical description of the world’s physical phenomena. In general, scientists and engineers solve PDEs numerically by the use of computationally demanding solvers. Recently, deep learning algorithms have emerged as a viable alternative for obtaining fast solutions for PDEs. Models are usually trained on synthetic data generated by solvers, stored on disk and read back for training. This paper advocates that relying on a traditional static dataset to train these models does not allow the full benefit of the solver to be used as a data generator. It proposes an open source online training framework for deep surrogate models. The framework implements several levels of parallelism focused on simultaneously generating numerical simulations and training deep neural networks. This approach suppresses the I/O and storage bottleneck associated with disk-loaded datasets, and opens the way to training on significantly larger datasets. Experiments compare the offline and online training of four surrogate models, including state-of-the-art architectures. Results indicate that exposing deep surrogate models to more dataset diversity, up to hundreds of GB, can increase model generalization capabilities. Fully connected neural networks, Fourier Neural Operator (FNO), and Message Passing PDE Solver prediction accuracy is improved by 68%, 16% and 7%, respectively.

----

## [1024] MANSA: Learning Fast and Slow in Multi-Agent Systems

**Authors**: *David Henry Mguni, Haojun Chen, Taher Jafferjee, Jianhong Wang, Longfei Yue, Xidong Feng, Stephen Marcus McAleer, Feifei Tong, Jun Wang, Yaodong Yang*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/mguni23a.html](https://proceedings.mlr.press/v202/mguni23a.html)

**Abstract**:

In multi-agent reinforcement learning (MARL), independent learning (IL) often shows remarkable performance and easily scales with the number of agents. Yet, using IL can be inefficient and runs the risk of failing to successfully train, particularly in scenarios that require agents to coordinate their actions. Using centralised learning (CL) enables MARL agents to quickly learn how to coordinate their behaviour but employing CL everywhere is often prohibitively expensive in real-world applications. Besides, using CL in value-based methods often needs strong representational constraints (e.g. individual-global-max condition) that can lead to poor performance if violated. In this paper, we introduce a novel plug & play IL framework named Multi-Agent Network Selection Algorithm (MANSA) which selectively employs CL only at states that require coordination. At its core, MANSA has an additional agent that uses switching controls to quickly learn the best states to activate CL during training, using CL only where necessary and vastly reducing the computational burden of CL. Our theory proves MANSA preserves cooperative MARL convergence properties, boosts IL performance and can optimally make use of a fixed budget on the number CL calls. We show empirically in Level-based Foraging (LBF) and StarCraft Multi-agent Challenge (SMAC) that MANSA achieves fast, superior and more reliable performance while making 40% fewer CL calls in SMAC and using CL at only 1% CL calls in LBF.

----

## [1025] Representation Learning with Multi-Step Inverse Kinematics: An Efficient and Optimal Approach to Rich-Observation RL

**Authors**: *Zakaria Mhammedi, Dylan J. Foster, Alexander Rakhlin*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/mhammedi23a.html](https://proceedings.mlr.press/v202/mhammedi23a.html)

**Abstract**:

We study the design of sample-efficient algorithms for reinforcement learning in the presence of rich, high-dimensional observations, formalized via the Block MDP problem. Existing algorithms suffer from either 1) computational intractability, 2) strong statistical assumptions that are not necessarily satisfied in practice, or 3) suboptimal sample complexity. We address these issues by providing the first computationally efficient algorithm that attains rate-optimal sample complexity with respect to the desired accuracy level, with minimal statistical assumptions. Our algorithm, MusIK, combines exploration with representation learning based on multi-step inverse kinematics, a learning objective in which the aim is to predict the current action from the current observation and observations in the (potentially distant) future. MusIK is simple and flexible, and can efficiently take advantage of general-purpose function approximation. Our analysis of MusIK leverages several new techniques tailored to non-optimistic algorithms for reward-free exploration, which we anticipate will find broader use.

----

## [1026] Single Point-Based Distributed Zeroth-Order Optimization with a Non-Convex Stochastic Objective Function

**Authors**: *Elissa Mhanna, Mohamad Assaad*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/mhanna23a.html](https://proceedings.mlr.press/v202/mhanna23a.html)

**Abstract**:

Zero-order (ZO) optimization is a powerful tool for dealing with realistic constraints. On the other hand, the gradient-tracking (GT) technique proved to be an efficient method for distributed optimization aiming to achieve consensus. However, it is a first-order (FO) method that requires knowledge of the gradient, which is not always possible in practice. In this work, we introduce a zero-order distributed optimization method based on a one-point estimate of the gradient tracking technique. We prove that this new technique converges with a single noisy function query at a time in the non-convex setting. We then establish a convergence rate of $O(\frac{1}{\sqrt[3]{K}})$ after a number of iterations K, which competes with that of $O(\frac{1}{\sqrt[4]{K}})$ of its centralized counterparts. Finally, a numerical example validates our theoretical results.

----

## [1027] Learning Instance-Specific Augmentations by Capturing Local Invariances

**Authors**: *Ning Miao, Tom Rainforth, Emile Mathieu, Yann Dubois, Yee Whye Teh, Adam Foster, Hyunjik Kim*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/miao23a.html](https://proceedings.mlr.press/v202/miao23a.html)

**Abstract**:

We introduce InstaAug, a method for automatically learning input-specific augmentations from data. Previous methods for learning augmentations have typically assumed independence between the original input and the transformation applied to that input. This can be highly restrictive, as the invariances we hope our augmentation will capture are themselves often highly input dependent. InstaAug instead introduces a learnable invariance module that maps from inputs to tailored transformation parameters, allowing local invariances to be captured. This can be simultaneously trained alongside the downstream model in a fully end-to-end manner, or separately learned for a pre-trained model. We empirically demonstrate that InstaAug learns meaningful input-dependent augmentations for a wide range of transformation classes, which in turn provides better performance on both supervised and self-supervised tasks.

----

## [1028] Path Neural Networks: Expressive and Accurate Graph Neural Networks

**Authors**: *Gaspard Michel, Giannis Nikolentzos, Johannes F. Lutzeyer, Michalis Vazirgiannis*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/michel23a.html](https://proceedings.mlr.press/v202/michel23a.html)

**Abstract**:

Graph neural networks (GNNs) have recently become the standard approach for learning with graph-structured data. Prior work has shed light into their potential, but also their limitations. Unfortunately, it was shown that standard GNNs are limited in their expressive power. These models are no more powerful than the 1-dimensional Weisfeiler-Leman (1-WL) algorithm in terms of distinguishing non-isomorphic graphs. In this paper, we propose Path Neural Networks (PathNNs), a model that updates node representations by aggregating paths emanating from nodes. We derive three different variants of the PathNN model that aggregate single shortest paths, all shortest paths and all simple paths of length up to K. We prove that two of these variants are strictly more powerful than the 1-WL algorithm, and we experimentally validate our theoretical results. We find that PathNNs can distinguish pairs of non-isomorphic graphs that are indistinguishable by 1-WL, while our most expressive PathNN variant can even distinguish between 3-WL indistinguishable graphs. The different PathNN variants are also evaluated on graph classification and graph regression datasets, where in most cases, they outperform the baseline methods.

----

## [1029] Learning to acquire novel cognitive tasks with evolution, plasticity and meta-meta-learning

**Authors**: *Thomas Miconi*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/miconi23a.html](https://proceedings.mlr.press/v202/miconi23a.html)

**Abstract**:

A hallmark of intelligence is the ability to autonomously learn new flexible, cognitive behaviors - that is, behaviors where the appropriate action depends not just on immediate stimuli (as in simple reflexive stimulus-response associations), but on contextual information that must be adequately acquired, stored and processed. While many meta-learning algorithms can design agents that autonomously learn new tasks, cognitive tasks adds another level of learning and memory to typical “learning-to-learn” problems. Here we evolve neural networks, endowed with plastic connections and neuromodulation, over a sizable set of simple cognitive tasks adapted from a computational neuroscience framework. The resulting evolved networks can automatically modify their own connectivity to acquire a novel simple cognitive task, never seen during evolution, from stimuli and rewards alone, through the spontaneous operation of their evolved neural organization and plasticity system. Our results emphasize the importance of carefully considering the multiple learning loops involved in the emergence of intelligent behavior.

----

## [1030] Generative Decoding of Visual Stimuli

**Authors**: *Eleni Miliotou, Panagiotis Kyriakis, Jason D. Hinman, Andrei Irimia, Paul Bogdan*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/miliotou23a.html](https://proceedings.mlr.press/v202/miliotou23a.html)

**Abstract**:

Reconstructing natural images from fMRI recordings is a challenging task of great importance in neuroscience. The current architectures are bottlenecked because they fail to effectively capture the hierarchical processing of visual stimuli that takes place in the human brain. Motivated by that fact, we introduce a novel neural network architecture for the problem of neural decoding. Our architecture uses Hierarchical Variational Autoencoders (HVAEs) to learn meaningful representations of natural images and leverages their latent space hierarchy to learn voxel-to-image mappings. By mapping the early stages of the visual pathway to the first set of latent variables and the higher visual cortex areas to the deeper layers in the latent hierarchy, we are able to construct a latent variable neural decoding model that replicates the hierarchical visual information processing. Our model achieves better reconstructions compared to the state of the art and our ablation study indicates that the hierarchical structure of the latent space is responsible for that performance.

----

## [1031] Cooperative Multi-Agent Reinforcement Learning: Asynchronous Communication and Linear Function Approximation

**Authors**: *Yifei Min, Jiafan He, Tianhao Wang, Quanquan Gu*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/min23a.html](https://proceedings.mlr.press/v202/min23a.html)

**Abstract**:

We study multi-agent reinforcement learning in the setting of episodic Markov decision processes, where many agents cooperate via communication through a central server. We propose a provably efficient algorithm based on value iteration that can simultaneously allow asynchronous communication and guarantee the benefit of cooperation with low communication complexity. Under linear function approximation, we prove that our algorithm enjoys a $\tilde{\mathcal{O}}(d^{3/2}H^2\sqrt{K})$ regret upper bound with $\tilde{\mathcal{O}}(dHM^2)$ communication complexity, where $d$ is the feature dimension, $H$ is the horizon length, $M$ is the total number of agents, and $K$ is the total number of episodes. We also provide a lower bound showing that an $\Omega(dM)$ communication complexity is necessary to improve the performance through collaboration.

----

## [1032] Directed Chain Generative Adversarial Networks

**Authors**: *Ming Min, Ruimeng Hu, Tomoyuki Ichiba*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/min23b.html](https://proceedings.mlr.press/v202/min23b.html)

**Abstract**:

Real-world data can be multimodal distributed, e.g., data describing the opinion divergence in a community, the interspike interval distribution of neurons, and the oscillators natural frequencies. Generating multimodal distributed real-world data has become a challenge to existing generative adversarial networks (GANs). For example, it is often observed that Neural SDEs have only demonstrated successfully performance mainly in generating unimodal time series datasets. In this paper, we propose a novel time series generator, named directed chain GANs (DC-GANs), which inserts a time series dataset (called a neighborhood process of the directed chain or input) into the drift and diffusion coefficients of the directed chain SDEs with distributional constraints. DC-GANs can generate new time series of the same distribution as the neighborhood process, and the neighborhood process will provide the key step in learning and generating multimodal distributed time series. The proposed DC-GANs are examined on four datasets, including two stochastic models from social sciences and computational neuroscience, and two real-world datasets on stock prices and energy consumption. To our best knowledge, DC-GANs are the first work that can generate multimodal time series data and consistently outperforms state-of-the-art benchmarks with respect to measures of distribution, data similarity, and predictive ability.

----

## [1033] An Information-Theoretic Analysis of Nonstationary Bandit Learning

**Authors**: *Seungki Min, Daniel Russo*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/min23c.html](https://proceedings.mlr.press/v202/min23c.html)

**Abstract**:

In nonstationary bandit learning problems, the decision-maker must continually gather information and adapt their action selection as the latent state of the environment evolves. In each time period, some latent optimal action maximizes expected reward under the environment state. We view the optimal action sequence as a stochastic process, and take an information-theoretic approach to analyze attainable performance. We bound per-period regret in terms of the entropy rate of the optimal action process. The bound applies to a wide array of problems studied in the literature and reflects the problem’s information structure through its information-ratio.

----

## [1034] On the Convergence of Gradient Flow on Multi-layer Linear Models

**Authors**: *Hancheng Min, René Vidal, Enrique Mallada*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/min23d.html](https://proceedings.mlr.press/v202/min23d.html)

**Abstract**:

In this paper, we analyze the convergence of gradient flow on a multi-layer linear model with a loss function of the form $f(W_1W_2\cdots W_L)$. We show that when $f$ satisfies the gradient dominance property, proper weight initialization leads to exponential convergence of the gradient flow to a global minimum of the loss. Moreover, the convergence rate depends on two trajectory-specific quantities that are controlled by the weight initialization: the imbalance matrices, which measure the difference between the weights of adjacent layers, and the least singular value of the weight product $W=W_1W_2\cdots W_L$. Our analysis exploits the fact that the gradient of the overparameterized loss can be written as the composition of the non-overparametrized gradient with a time-varying (weight-dependent) linear operator whose smallest eigenvalue controls the convergence rate. The key challenge we address is to derive a uniform lower bound for this time-varying eigenvalue that lead to improved rates for several multi-layer network models studied in the literature.

----

## [1035] Optimal Sets and Solution Paths of ReLU Networks

**Authors**: *Aaron Mishkin, Mert Pilanci*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/mishkin23a.html](https://proceedings.mlr.press/v202/mishkin23a.html)

**Abstract**:

We develop an analytical framework to characterize the set of optimal ReLU neural networks by reformulating the non-convex training problem as a convex program. We show that the global optima of the convex parameterization are given by a polyhedral set and then extend this characterization to the optimal set of the non-convex training objective. Since all stationary points of the ReLU training problem can be represented as optima of sub-sampled convex programs, our work provide a general expression for all critical points of the non-convex objective. We then leverage our results to provide an optimal pruning algorithm for computing minimal networks, establish conditions for the regularization path of ReLU networks to be continuous, and develop sensitivity results for minimal ReLU networks.

----

## [1036] The Numerical Stability of Hyperbolic Representation Learning

**Authors**: *Gal Mishne, Zhengchao Wan, Yusu Wang, Sheng Yang*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/mishne23a.html](https://proceedings.mlr.press/v202/mishne23a.html)

**Abstract**:

The hyperbolic space is widely used for representing hierarchical datasets due to its ability to embed trees with small distortion. However, this property comes at a price of numerical instability such that training hyperbolic learning models will sometimes lead to catastrophic NaN problems, encountering unrepresentable values in floating point arithmetic. In this work, we analyze the limitations of two popular models for the hyperbolic space, namely, the Poincaré ball and the Lorentz model. We find that, under the 64-bit arithmetic system, the Poincaré ball has a relatively larger capacity than the Lorentz model for correctly representing points. However, the Lorentz model is superior to the Poincaré ball from the perspective of optimization, which we theoretically validate. To address these limitations, we identify one Euclidean parametrization of the hyperbolic space which can alleviate these issues. We further extend this Euclidean parametrization to hyperbolic hyperplanes and demonstrate its effectiveness in improving the performance of hyperbolic SVM.

----

## [1037] DetectGPT: Zero-Shot Machine-Generated Text Detection using Probability Curvature

**Authors**: *Eric Mitchell, Yoonho Lee, Alexander Khazatsky, Christopher D. Manning, Chelsea Finn*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/mitchell23a.html](https://proceedings.mlr.press/v202/mitchell23a.html)

**Abstract**:

The increasing fluency and widespread usage of large language models (LLMs) highlight the desirability of corresponding tools aiding detection of LLM-generated text. In this paper, we identify a property of the structure of an LLM’s probability function that is useful for such detection. Specifically, we demonstrate that text sampled from an LLM tends to occupy negative curvature regions of the model’s log probability function. Leveraging this observation, we then define a new curvature-based criterion for judging if a passage is generated from a given LLM. This approach, which we call DetectGPT, does not require training a separate classifier, collecting a dataset of real or generated passages, or explicitly watermarking generated text. It uses only log probabilities computed by the model of interest and random perturbations of the passage from another generic pre-trained language model (e.g., T5). We find DetectGPT is more discriminative than existing zero-shot methods for model sample detection, notably improving detection of fake news articles generated by 20B parameter GPT-NeoX from 0.81 AUROC for the strongest zero-shot baseline to 0.95 AUROC for DetectGPT.

----

## [1038] Diffusion Based Representation Learning

**Authors**: *Sarthak Mittal, Korbinian Abstreiter, Stefan Bauer, Bernhard Schölkopf, Arash Mehrjou*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/mittal23a.html](https://proceedings.mlr.press/v202/mittal23a.html)

**Abstract**:

Diffusion-based methods, represented as stochastic differential equations on a continuous-time domain, have recently proven successful as non-adversarial generative models. Training such models relies on denoising score matching, which can be seen as multi-scale denoising autoencoders. Here, we augment the denoising score matching framework to enable representation learning without any supervised signal. GANs and VAEs learn representations by directly transforming latent codes to data samples. In contrast, the introduced diffusion-based representation learning relies on a new formulation of the denoising score matching objective and thus encodes the information needed for denoising. We illustrate how this difference allows for manual control of the level of details encoded in the representation. Using the same approach, we propose to learn an infinite-dimensional latent code that achieves improvements on state-of-the-art models on semi-supervised image classification. We also compare the quality of learned representations of diffusion score matching with other methods like autoencoder and contrastively trained systems through their performances on downstream tasks. Finally, we also ablate with a different SDE formulation for diffusion models and show that the benefits on downstream tasks are still present on changing the underlying differential equation.

----

## [1039] Disentangled Multiplex Graph Representation Learning

**Authors**: *Yujie Mo, Yajie Lei, Jialie Shen, Xiaoshuang Shi, Heng Tao Shen, Xiaofeng Zhu*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/mo23a.html](https://proceedings.mlr.press/v202/mo23a.html)

**Abstract**:

Unsupervised multiplex graph representation learning (UMGRL) has received increasing interest, but few works simultaneously focused on the common and private information extraction. In this paper, we argue that it is essential for conducting effective and robust UMGRL to extract complete and clean common information, as well as more-complementarity and less-noise private information. To achieve this, we first investigate disentangled representation learning for the multiplex graph to capture complete and clean common information, as well as design a contrastive constraint to preserve the complementarity and remove the noise in the private information. Moreover, we theoretically analyze that the common and private representations learned by our method are provably disentangled and contain more task-relevant and less task-irrelevant information to benefit downstream tasks. Extensive experiments verify the superiority of the proposed method in terms of different downstream tasks.

----

## [1040] A Unified Audio-Visual Learning Framework for Localization, Separation, and Recognition

**Authors**: *Shentong Mo, Pedro Morgado*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/mo23b.html](https://proceedings.mlr.press/v202/mo23b.html)

**Abstract**:

The ability to accurately recognize, localize and separate sound sources is fundamental to any audio-visual perception task. Historically, these abilities were tackled separately, with several methods developed independently for each task. However, given the interconnected nature of source localization, separation, and recognition, independent models are likely to yield suboptimal performance as they fail to capture the interdependence between these tasks. To address this problem, we propose a unified audio-visual learning framework (dubbed OneAVM) that integrates audio and visual cues for joint localization, separation, and recognition. OneAVM comprises a shared audio-visual encoder and task-specific decoders trained with three objectives. The first objective aligns audio and visual representations through a localized audio-visual correspondence loss. The second tackles visual source separation using a traditional mix-and-separate framework. Finally, the third objective reinforces visual feature separation and localization by mixing images in pixel space and aligning their representations with those of all corresponding sound sources. Extensive experiments on MUSIC, VGG-Instruments, VGG-Music, and VGGSound datasets demonstrate the effectiveness of OneAVM for all three tasks, audio-visual source localization, separation, and nearest neighbor recognition, and empirically demonstrate a strong positive transfer between them.

----

## [1041] Pruning via Sparsity-indexed ODE: a Continuous Sparsity Viewpoint

**Authors**: *Zhanfeng Mo, Haosen Shi, Sinno Jialin Pan*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/mo23c.html](https://proceedings.mlr.press/v202/mo23c.html)

**Abstract**:

Neural pruning, which involves identifying the optimal sparse subnetwork, is a key technique for reducing the complexity and improving the efficiency of deep neural networks. To address the challenge of solving neural pruning at a specific sparsity level directly, we investigate the evolution of optimal subnetworks with continuously increasing sparsity, which can provide insight into how to transform an unpruned dense model into an optimal subnetwork with any desired level of sparsity. In this paper, we proposed a novel pruning framework, coined Sparsity-indexed ODE (SpODE) that provides explicit guidance on how to best preserve model performance while ensuring an infinitesimal increase in model sparsity. On top of this, we develop a pruning algorithm, termed Pruning via Sparsity-indexed ODE (PSO), that enables effective pruning via traveling along the SpODE path. Empirical experiments show that PSO achieves either better or comparable performance compared to state-of-the-art baselines across various pruning settings.

----

## [1042] Text-To-Concept (and Back) via Cross-Model Alignment

**Authors**: *Mazda Moayeri, Keivan Rezaei, Maziar Sanjabi, Soheil Feizi*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/moayeri23a.html](https://proceedings.mlr.press/v202/moayeri23a.html)

**Abstract**:

We observe that the mapping between an image’s representation in one model to its representation in another can be learned surprisingly well with just a linear layer, even across diverse models. Building on this observation, we propose text-to-concept, where features from a fixed pretrained model are aligned linearly to the CLIP space, so that text embeddings from CLIP’s text encoder become directly comparable to the aligned features. With text-to-concept, we convert fixed off-the-shelf vision encoders to surprisingly strong zero-shot classifiers for free, with accuracy at times even surpassing that of CLIP, despite being much smaller models and trained on a small fraction of the data compared to CLIP. We show other immediate use-cases of text-to-concept, like building concept bottleneck models with no concept supervision, diagnosing distribution shifts in terms of human concepts, and retrieving images satisfying a set of text-based constraints. Lastly, we demonstrate the feasibility of concept-to-text, where vectors in a model’s feature space are decoded by first aligning to the CLIP before being fed to a GPT-based generative model. Our work suggests existing deep models, with presumably diverse architectures and training, represent input samples relatively similarly, and a two-way communication across model representation spaces and to humans (through language) is viable.

----

## [1043] A Fast, Well-Founded Approximation to the Empirical Neural Tangent Kernel

**Authors**: *Mohamad Amin Mohamadi, Wonho Bae, Danica J. Sutherland*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/mohamadi23a.html](https://proceedings.mlr.press/v202/mohamadi23a.html)

**Abstract**:

Empirical neural tangent kernels (eNTKs) can provide a good understanding of a given network’s representation: they are often far less expensive to compute and applicable more broadly than infinite-width NTKs. For networks with $O$ output units (e.g. an $O$-class classifier), however, the eNTK on $N$ inputs is of size $NO \times NO$, taking $\mathcal O\big( (N O)^2\big)$ memory and up to $\mathcal O\big( (N O)^3 \big)$ computation to use. Most existing applications have therefore used one of a handful of approximations yielding $N \times N$ kernel matrices, saving orders of magnitude of computation, but with limited to no justification. We prove that one such approximation, which we call "sum of logits," converges to the true eNTK at initialization. Our experiments demonstrate the quality of this approximation for various uses across a range of settings.

----

## [1044] Special Properties of Gradient Descent with Large Learning Rates

**Authors**: *Amirkeivan Mohtashami, Martin Jaggi, Sebastian U. Stich*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/mohtashami23a.html](https://proceedings.mlr.press/v202/mohtashami23a.html)

**Abstract**:

When training neural networks, it has been widely observed that a large step size is essential in stochastic gradient descent (SGD) for obtaining superior models. However, the effect of large step sizes on the success of SGD is not well understood theoretically. Several previous works have attributed this success to the stochastic noise present in SGD. However, we show through a novel set of experiments that the stochastic noise is not sufficient to explain good non-convex training, and that instead the effect of a large learning rate itself is essential for obtaining best performance.We demonstrate the same effects also in the noise-less case, i.e. for full-batch GD. We formally prove that GD with large step size —on certain non-convex function classes — follows a different trajectory than GD with a small step size, which can lead to convergence to a global minimum instead of a local one. Our settings provide a framework for future analysis which allows comparing algorithms based on behaviors that can not be observed in the traditional settings.

----

## [1045] Neural Inverse Operators for Solving PDE Inverse Problems

**Authors**: *Roberto Molinaro, Yunan Yang, Björn Engquist, Siddhartha Mishra*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/molinaro23a.html](https://proceedings.mlr.press/v202/molinaro23a.html)

**Abstract**:

A large class of inverse problems for PDEs are only well-defined as mappings from operators to functions. Existing operator learning frameworks map functions to functions and need to be modified to learn inverse maps from data. We propose a novel architecture termed Neural Inverse Operators (NIOs) to solve these PDE inverse problems. Motivated by the underlying mathematical structure, NIO is based on a suitable composition of DeepONets and FNOs to approximate mappings from operators to functions. A variety of experiments are presented to demonstrate that NIOs significantly outperform baselines and solve PDE inverse problems robustly, accurately and are several orders of magnitude faster than existing direct and PDE-constrained optimization methods.

----

## [1046] Input uncertainty propagation through trained neural networks

**Authors**: *Paul Monchot, Loic Coquelin, Sébastien Julien Petit, Sébastien Marmin, Erwan Le Pennec, Nicolas Fischer*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/monchot23a.html](https://proceedings.mlr.press/v202/monchot23a.html)

**Abstract**:

When physical sensors are involved, such as image sensors, the uncertainty over the input data is often a major component of the output uncertainty of machine learning models. In this work, we address the problem of input uncertainty propagation through trained neural networks. We do not rely on a Gaussian distribution assumption of the output or of any intermediate layer. We propagate instead a Gaussian Mixture Model (GMM) that offers much more flexibility, using the Split&Merge algorithm. This paper’s main contribution is the computation of a Wasserstein criterion to control the Gaussian splitting procedure for which theoretical guarantees of convergence on the output distribution estimates are derived. The methodology is tested against a wide range of datasets and networks. It shows robustness, and genericity and offers highly accurate output probability density function estimation while maintaining a reasonable computational cost compared with the standard Monte Carlo (MC) approach.

----

## [1047] Compressing Tabular Data via Latent Variable Estimation

**Authors**: *Andrea Montanari, Eric Weiner*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/montanari23a.html](https://proceedings.mlr.press/v202/montanari23a.html)

**Abstract**:

Data used for analytics and machine learning often take the form of tables with categorical entries. We introduce a family of lossless compression algorithms for such data that proceed in four steps: (i) Estimate latent variables associated to rows and columns; (ii) Partition the table in blocks according to the row/column latents; (iii) Apply a sequential (e.g. Lempel-Ziv) coder to each of the blocks; (iv) Append a compressed encoding of the latents. We evaluate this approach on several benchmark datasets, and study optimal compression in a probabilistic model for tabular data, whereby latent values are independent and table entries are conditionally independent given the latent values. We prove that the model has a well defined entropy rate and satisfies an asymptotic equipartition property. We also prove that classical compression schemes such as Lempel-Ziv and finite-state encoders do not achieve this rate. On the other hand, the latent estimation strategy outlined above achieves the optimal rate.

----

## [1048] An SDE for Modeling SAM: Theory and Insights

**Authors**: *Enea Monzio Compagnoni, Luca Biggio, Antonio Orvieto, Frank Norbert Proske, Hans Kersting, Aurélien Lucchi*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/monzio-compagnoni23a.html](https://proceedings.mlr.press/v202/monzio-compagnoni23a.html)

**Abstract**:

We study the SAM (Sharpness-Aware Minimization) optimizer which has recently attracted a lot of interest due to its increased performance over more classical variants of stochastic gradient descent. Our main contribution is the derivation of continuous-time models (in the form of SDEs) for SAM and two of its variants, both for the full-batch and mini-batch settings. We demonstrate that these SDEs are rigorous approximations of the real discrete-time algorithms (in a weak sense, scaling linearly with the learning rate). Using these models, we then offer an explanation of why SAM prefers flat minima over sharp ones – by showing that it minimizes an implicitly regularized loss with a Hessian-dependent noise structure. Finally, we prove that SAM is attracted to saddle points under some realistic conditions. Our theoretical results are supported by detailed experiments.

----

## [1049] Learning Deductive Reasoning from Synthetic Corpus based on Formal Logic

**Authors**: *Terufumi Morishita, Gaku Morio, Atsuki Yamaguchi, Yasuhiro Sogawa*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/morishita23a.html](https://proceedings.mlr.press/v202/morishita23a.html)

**Abstract**:

We study a synthetic corpus based approach for language models (LMs) to acquire logical deductive reasoning ability. The previous studies generated deduction examples using specific sets of deduction rules. However, these rules were limited or otherwise arbitrary. This can limit the generalizability of acquired deductive reasoning ability. We rethink this and adopt a well-grounded set of deduction rules based on formal logic theory, which can derive any other deduction rules when combined in a multistep way. We empirically verify that LMs trained on the proposed corpora, which we name $\textbf{FLD}$ ($\textbf{F}$ormal $\textbf{L}$ogic $\textbf{D}$eduction), acquire more generalizable deductive reasoning ability. Furthermore, we identify the aspects of deductive reasoning ability on which deduction corpora can enhance LMs and those on which they cannot. Finally, on the basis of these results, we discuss the future directions for applying deduction corpora or other approaches for each aspect. We release the code, data, and models.

----

## [1050] WL meet VC

**Authors**: *Christopher Morris, Floris Geerts, Jan Tönshoff, Martin Grohe*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/morris23a.html](https://proceedings.mlr.press/v202/morris23a.html)

**Abstract**:

Recently, many works studied the expressive power of graph neural networks (GNNs) by linking it to the $1$-dimensional Weisfeiler-Leman algorithm ($1\text{-}\mathsf{WL}$). Here, the $1\text{-}\mathsf{WL}$ is a well-studied heuristic for the graph isomorphism problem, which iteratively colors or partitions a graph’s vertex set. While this connection has led to significant advances in understanding and enhancing GNNs’ expressive power, it does not provide insights into their generalization performance, i.e., their ability to make meaningful predictions beyond the training set. In this paper, we study GNNs’ generalization ability through the lens of Vapnik-Chervonenkis (VC) dimension theory in two settings, focusing on graph-level predictions. First, when no upper bound on the graphs’ order is known, we show that the bitlength of GNNs’ weights tightly bounds their VC dimension. Further, we derive an upper bound for GNNs’ VC dimension using the number of colors produced by the $1\text{-}\mathsf{WL}$. Secondly, when an upper bound on the graphs’ order is known, we show a tight connection between the number of graphs distinguishable by the $1\text{-}\mathsf{WL}$ and GNNs’ VC dimension. Our empirical study confirms the validity of our theoretical findings.

----

## [1051] ReLOAD: Reinforcement Learning with Optimistic Ascent-Descent for Last-Iterate Convergence in Constrained MDPs

**Authors**: *Ted Moskovitz, Brendan O'Donoghue, Vivek Veeriah, Sebastian Flennerhag, Satinder Singh, Tom Zahavy*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/moskovitz23a.html](https://proceedings.mlr.press/v202/moskovitz23a.html)

**Abstract**:

In recent years, reinforcement learning (RL) has been applied to real-world problems with increasing success. Such applications often require to put constraints on the agent’s behavior. Existing algorithms for constrained RL (CRL) rely on gradient descent-ascent, but this approach comes with a caveat. While these algorithms are guaranteed to converge on average, they do not guarantee last-iterate convergence, i.e., the current policy of the agent may never converge to the optimal solution. In practice, it is often observed that the policy alternates between satisfying the constraints and maximizing the reward, rarely accomplishing both objectives simultaneously. Here, we address this problem by introducing Reinforcement Learning with Optimistic Ascent-Descent (ReLOAD), a principled CRL method with guaranteed last-iterate convergence. We demonstrate its empirical effectiveness on a wide variety of CRL problems including discrete MDPs and continuous control. In the process we establish a benchmark of challenging CRL problems.

----

## [1052] Optimistic Planning by Regularized Dynamic Programming

**Authors**: *Antoine Moulin, Gergely Neu*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/moulin23a.html](https://proceedings.mlr.press/v202/moulin23a.html)

**Abstract**:

We propose a new method for optimistic planning in infinite-horizon discounted Markov decision processes based on the idea of adding regularization to the updates of an otherwise standard approximate value iteration procedure. This technique allows us to avoid contraction and monotonicity arguments typically required by existing analyses of approximate dynamic programming methods, and in particular to use approximate transition functions estimated via least-squares procedures in MDPs with linear function approximation. We use our method to recover known guarantees in tabular MDPs and to provide a computationally efficient algorithm for learning near-optimal policies in discounted linear mixture MDPs from a single stream of experience, and show it achieves near-optimal statistical guarantees.

----

## [1053] Neural signature kernels as infinite-width-depth-limits of controlled ResNets

**Authors**: *Nicola Muca Cirone, Maud Lemercier, Cristopher Salvi*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/muca-cirone23a.html](https://proceedings.mlr.press/v202/muca-cirone23a.html)

**Abstract**:

Motivated by the paradigm of reservoir computing, we consider randomly initialized controlled ResNets defined as Euler-discretizations of neural controlled differential equations (Neural CDEs), a unified architecture which enconpasses both RNNs and ResNets. We show that in the infinite-width-depth limit and under proper scaling, these architectures converge weakly to Gaussian processes indexed on some spaces of continuous paths and with kernels satisfying certain partial differential equations (PDEs) varying according to the choice of activation function $\varphi$, extending the results of Hayou (2022); Hayou & Yang (2023) to the controlled and homogeneous case. In the special, homogeneous, case where $\varphi$ is the identity, we show that the equation reduces to a linear PDE and the limiting kernel agrees with the signature kernel of Salvi et al. (2021a). We name this new family of limiting kernels neural signature kernels. Finally, we show that in the infinite-depth regime, finite-width controlled ResNets converge in distribution to Neural CDEs with random vector fields which, depending on whether the weights are shared across layers, are either time-independent and Gaussian or behave like a matrix-valued Brownian motion.

----

## [1054] Improving Statistical Fidelity for Neural Image Compression with Implicit Local Likelihood Models

**Authors**: *Matthew J. Muckley, Alaaeldin El-Nouby, Karen Ullrich, Hervé Jégou, Jakob Verbeek*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/muckley23a.html](https://proceedings.mlr.press/v202/muckley23a.html)

**Abstract**:

Lossy image compression aims to represent images in as few bits as possible while maintaining fidelity to the original. Theoretical results indicate that optimizing distortion metrics such as PSNR or MS-SSIM necessarily leads to a discrepancy in the statistics of original images from those of reconstructions, in particular at low bitrates, often manifested by the blurring of the compressed images. Previous work has leveraged adversarial discriminators to improve statistical fidelity. Yet these binary discriminators adopted from generative modeling tasks may not be ideal for image compression. In this paper, we introduce a non-binary discriminator that is conditioned on quantized local image representations obtained via VQ-VAE autoencoders. Our evaluations on the CLIC2020, DIV2K and Kodak datasets show that our discriminator is more effective for jointly optimizing distortion (e.g., PSNR) and statistical fidelity (e.g., FID) than the PatchGAN of the state-of-the-art HiFiC model. On CLIC2020, we obtain the same FID as HiFiC with 30-40% fewer bits.

----

## [1055] PFNs4BO: In-Context Learning for Bayesian Optimization

**Authors**: *Samuel Müller, Matthias Feurer, Noah Hollmann, Frank Hutter*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/muller23a.html](https://proceedings.mlr.press/v202/muller23a.html)

**Abstract**:

In this paper, we use Prior-data Fitted Networks (PFNs) as a flexible surrogate for Bayesian Optimization (BO). PFNs are neural processes that are trained to approximate the posterior predictive distribution (PPD) through in-context learning on any prior distribution that can be efficiently sampled from. We describe how this flexibility can be exploited for surrogate modeling in BO. We use PFNs to mimic a naive Gaussian process (GP), an advanced GP, and a Bayesian Neural Network (BNN). In addition, we show how to incorporate further information into the prior, such as allowing hints about the position of optima (user priors), ignoring irrelevant dimensions, and performing non-myopic BO by learning the acquisition function. The flexibility underlying these extensions opens up vast possibilities for using PFNs for BO. We demonstrate the usefulness of PFNs for BO in a large-scale evaluation on artificial GP samples and three different hyperparameter optimization testbeds: HPO-B, Bayesmark, and PD1. We publish code alongside trained models at https://github.com/automl/PFNs4BO.

----

## [1056] Achieving High Accuracy with PINNs via Energy Natural Gradient Descent

**Authors**: *Johannes Müller, Marius Zeinhofer*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/muller23b.html](https://proceedings.mlr.press/v202/muller23b.html)

**Abstract**:

We propose energy natural gradient descent, a natural gradient method with respect to a Hessian-induced Riemannian metric as an optimization algorithm for physics-informed neural networks (PINNs) and the deep Ritz method. As a main motivation we show that the update direction in function space resulting from the energy natural gradient corresponds to the Newton direction modulo an orthogonal projection on the model’s tangent space. We demonstrate experimentally that energy natural gradient descent yields highly accurate solutions with errors several orders of magnitude smaller than what is obtained when training PINNs with standard optimizers like gradient descent or Adam, even when those are allowed significantly more computation time.

----

## [1057] Uncertain Evidence in Probabilistic Models and Stochastic Simulators

**Authors**: *Andreas Munk, Alexander Mead, Frank Wood*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/munk23a.html](https://proceedings.mlr.press/v202/munk23a.html)

**Abstract**:

We consider the problem of performing Bayesian inference in probabilistic models where observations are accompanied by uncertainty, referred to as "uncertain evidence.” We explore how to interpret uncertain evidence, and by extension the importance of proper interpretation as it pertains to inference about latent variables. We consider a recently-proposed method "distributional evidence” as well as revisit two older methods: Jeffrey’s rule and virtual evidence. We devise guidelines on how to account for uncertain evidence and we provide new insights, particularly regarding consistency. To showcase the impact of different interpretations of the same uncertain evidence, we carry out experiments in which one interpretation is defined as "correct.” We then compare inference results from each different interpretation illustrating the importance of careful consideration of uncertain evidence.

----

## [1058] GibbsDDRM: A Partially Collapsed Gibbs Sampler for Solving Blind Inverse Problems with Denoising Diffusion Restoration

**Authors**: *Naoki Murata, Koichi Saito, Chieh-Hsin Lai, Yuhta Takida, Toshimitsu Uesaka, Yuki Mitsufuji, Stefano Ermon*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/murata23a.html](https://proceedings.mlr.press/v202/murata23a.html)

**Abstract**:

Pre-trained diffusion models have been successfully used as priors in a variety of linear inverse problems, where the goal is to reconstruct a signal from noisy linear measurements. However, existing approaches require knowledge of the linear operator. In this paper, we propose GibbsDDRM, an extension of Denoising Diffusion Restoration Models (DDRM) to a blind setting in which the linear measurement operator is unknown. GibbsDDRM constructs a joint distribution of the data, measurements, and linear operator by using a pre-trained diffusion model for the data prior, and it solves the problem by posterior sampling with an efficient variant of a Gibbs sampler. The proposed method is problem-agnostic, meaning that a pre-trained diffusion model can be applied to various inverse problems without fine-tuning. In experiments, it achieved high performance on both blind image deblurring and vocal dereverberation tasks, despite the use of simple generic priors for the underlying linear operators.

----

## [1059] DIFF2: Differential Private Optimization via Gradient Differences for Nonconvex Distributed Learning

**Authors**: *Tomoya Murata, Taiji Suzuki*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/murata23b.html](https://proceedings.mlr.press/v202/murata23b.html)

**Abstract**:

Differential private optimization for nonconvex smooth objective is considered. In the previous work, the best known utility bound is $\widetilde O(\sqrt{d}/(n\varepsilon_\mathrm{DP}))$ in terms of the squared full gradient norm, which is achieved by Differential Private Gradient Descent (DP-GD) as an instance, where $n$ is the sample size, $d$ is the problem dimensionality and $\varepsilon_\mathrm{DP}$ is the differential privacy parameter. To improve the best known utility bound, we propose a new differential private optimization framework called DIFF2 (DIFFerential private optimization via gradient DIFFerences) that constructs a differential private global gradient estimator with possibly quite small variance based on communicated gradient differences rather than gradients themselves. It is shown that DIFF2 with a gradient descent subroutine achieves the utility of $\widetilde O(d^{2/3}/(n\varepsilon_\mathrm{DP})^{4/3})$, which can be significantly better than the previous one in terms of the dependence on the sample size $n$. To the best of our knowledge, this is the first fundamental result to improve the standard utility $\widetilde O(\sqrt{d}/(n\varepsilon_\mathrm{DP}))$ for nonconvex objectives. Additionally, a more computational and communication efficient subroutine is combined with DIFF2 and its theoretical analysis is also given. Numerical experiments are conducted to validate the superiority of DIFF2 framework.

----

## [1060] Efficiently predicting high resolution mass spectra with graph neural networks

**Authors**: *Michael Murphy, Stefanie Jegelka, Ernest Fraenkel, Tobias Kind, David Healey, Thomas Butler*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/murphy23a.html](https://proceedings.mlr.press/v202/murphy23a.html)

**Abstract**:

Identifying a small molecule from its mass spectrum is the primary open problem in computational metabolomics. This is typically cast as information retrieval: an unknown spectrum is matched against spectra predicted computationally from a large database of chemical structures. However, current approaches to spectrum prediction model the output space in ways that force a tradeoff between capturing high resolution mass information and tractable learning. We resolve this tradeoff by casting spectrum prediction as a mapping from an input molecular graph to a probability distribution over chemical formulas. We further discover that a large corpus of mass spectra can be closely approximated using a fixed vocabulary constituting only 2% of all observed formulas. This enables efficient spectrum prediction using an architecture similar to graph classification - GrAFF-MS - achieving significantly lower prediction error and greater retrieval accuracy than previous approaches.

----

## [1061] Dynamical Linear Bandits

**Authors**: *Marco Mussi, Alberto Maria Metelli, Marcello Restelli*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/mussi23a.html](https://proceedings.mlr.press/v202/mussi23a.html)

**Abstract**:

In many real-world sequential decision-making problems, an action does not immediately reflect on the feedback and spreads its effects over a long time frame. For instance, in online advertising, investing in a platform produces an instantaneous increase of awareness, but the actual reward, i.e., a conversion, might occur far in the future. Furthermore, whether a conversion takes place depends on: how fast the awareness grows, its vanishing effects, and the synergy or interference with other advertising platforms. Previous work has investigated the Multi-Armed Bandit framework with the possibility of delayed and aggregated feedback, without a particular structure on how an action propagates in the future, disregarding possible dynamical effects. In this paper, we introduce a novel setting, the Dynamical Linear Bandits (DLB), an extension of the linear bandits characterized by a hidden state. When an action is performed, the learner observes a noisy reward whose mean is a linear function of the hidden state and of the action. Then, the hidden state evolves according to linear dynamics, affected by the performed action too. We start by introducing the setting, discussing the notion of optimal policy, and deriving an expected regret lower bound. Then, we provide an optimistic regret minimization algorithm, Dynamical Linear Upper Confidence Bound (DynLin-UCB), that suffers an expected regret of order $\widetilde{\mathcal{O}} \Big( \frac{d \sqrt{T}}{(1-\overline{\rho})^{3/2}} \Big)$, where $\overline{\rho}$ is a measure of the stability of the system, and $d$ is the dimension of the action vector. Finally, we conduct a numerical validation on a synthetic environment and on real-world data to show the effectiveness of DynLin-UCB in comparison with several baselines.

----

## [1062] Representation-Driven Reinforcement Learning

**Authors**: *Ofir Nabati, Guy Tennenholtz, Shie Mannor*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/nabati23a.html](https://proceedings.mlr.press/v202/nabati23a.html)

**Abstract**:

We present a representation-driven framework for reinforcement learning. By representing policies as estimates of their expected values, we leverage techniques from contextual bandits to guide exploration and exploitation. Particularly, embedding a policy network into a linear feature space allows us to reframe the exploration-exploitation problem as a representation-exploitation problem, where good policy representations enable optimal exploration. We demonstrate the effectiveness of this framework through its application to evolutionary and policy gradient-based approaches, leading to significantly improved performance compared to traditional methods. Our framework provides a new perspective on reinforcement learning, highlighting the importance of policy representation in determining optimal exploration-exploitation strategies.

----

## [1063] DADAO: Decoupled Accelerated Decentralized Asynchronous Optimization

**Authors**: *Adel Nabli, Edouard Oyallon*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/nabli23a.html](https://proceedings.mlr.press/v202/nabli23a.html)

**Abstract**:

This work introduces DADAO: the first decentralized, accelerated, asynchronous, primal, first-order algorithm to minimize a sum of $L$-smooth and $\mu$-strongly convex functions distributed over a given network of size $n$. Our key insight is based on modeling the local gradient updates and gossip communication procedures with separate independent Poisson Point Processes. This allows us to decouple the computation and communication steps, which can be run in parallel, while making the whole approach completely asynchronous. This leads to communication acceleration compared to synchronous approaches. Our new method employs primal gradients and does not use a multi-consensus inner loop nor other ad-hoc mechanisms such as Error Feedback, Gradient Tracking, or a Proximal operator. By relating the inverse of the smallest positive eigenvalue of the Laplacian matrix $\chi_1$ and the maximal resistance $\chi_2\leq \chi_1$ of the graph to a sufficient minimal communication rate between the nodes of the network, we show that our algorithm requires $\mathcal{O}(n\sqrt{\frac{L}{\mu}}\log(\frac{1}{\epsilon}))$ local gradients and only $\mathcal{O}(n\sqrt{\chi_1\chi_2}\sqrt{\frac{L}{\mu}}\log(\frac{1}{\epsilon}))$ communications to reach a precision $\epsilon$, up to logarithmic terms. Thus, we simultaneously obtain an accelerated rate for both computations and communications, leading to an improvement over state-of-the-art works, our simulations further validating the strength of our relatively unconstrained method.

----

## [1064] Multi-User Reinforcement Learning with Low Rank Rewards

**Authors**: *Dheeraj Mysore Nagaraj, Suhas S. Kowshik, Naman Agarwal, Praneeth Netrapalli, Prateek Jain*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/nagaraj23a.html](https://proceedings.mlr.press/v202/nagaraj23a.html)

**Abstract**:

We consider collaborative multi-user reinforcement learning, where multiple users have the same state-action space and transition probabilities but different rewards. Under the assumption that the reward matrix of the $N$ users has a low-rank structure – a standard and practically successful assumption in the collaborative filtering setting – we design algorithms with significantly lower sample complexity compared to the ones that learn the MDP individually for each user. Our main contribution is an algorithm which explores rewards collaboratively with $N$ user-specific MDPs and can learn rewards efficiently in two key settings: tabular MDPs and linear MDPs. When $N$ is large and the rank is constant, the sample complexity per MDP depends logarithmically over the size of the state-space, which represents an exponential reduction (in the state-space size) when compared to the standard “non-collaborative” algorithms. Our main technical contribution is a method to construct policies which obtain data such that low rank matrix completion is possible (without a generative model). This goes beyond the regular RL framework and is closely related to mean field limits of multi-agent RL.

----

## [1065] Statistical Foundations of Prior-Data Fitted Networks

**Authors**: *Thomas Nagler*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/nagler23a.html](https://proceedings.mlr.press/v202/nagler23a.html)

**Abstract**:

Prior-data fitted networks (PFNs) were recently proposed as a new paradigm for machine learning. Instead of training the network to an observed training set, a fixed model is pre-trained offline on small, simulated training sets from a variety of tasks. The pre-trained model is then used to infer class probabilities in-context on fresh training sets with arbitrary size and distribution. Empirically, PFNs achieve state-of-the-art performance on tasks with similar size to the ones used in pre-training. Surprisingly, their accuracy further improves when passed larger data sets during inference. This article establishes a theoretical foundation for PFNs and illuminates the statistical mechanisms governing their behavior. While PFNs are motivated by Bayesian ideas, a purely frequentistic interpretation of PFNs as pre-tuned, but untrained predictors explains their behavior. A predictor’s variance vanishes if its sensitivity to individual training samples does and the bias vanishes only if it is appropriately localized around the test feature. The transformer architecture used in current PFN implementations ensures only the former. These findings shall prove useful for designing architectures with favorable empirical behavior.

----

## [1066] Do Machine Learning Models Learn Statistical Rules Inferred from Data?

**Authors**: *Aaditya Naik, Yinjun Wu, Mayur Naik, Eric Wong*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/naik23a.html](https://proceedings.mlr.press/v202/naik23a.html)

**Abstract**:

Machine learning models can make critical errors that are easily hidden within vast amounts of data. Such errors often run counter to rules based on human intuition. However, rules based on human knowledge are challenging to scale or to even formalize. We thereby seek to infer statistical rules from the data and quantify the extent to which a model has learned them. We propose a framework SQRL that integrates logic-based methods with statistical inference to derive these rules from a model’s training data without supervision. We further show how to adapt models at test time to reduce rule violations and produce more coherent predictions. SQRL generates up to 300K rules over datasets from vision, tabular, and language settings. We uncover up to 158K violations of those rules by state-of-the-art models for classification, object detection, and data imputation. Test-time adaptation reduces these violations by up to 68.7% with relative performance improvement up to 32%. SQRL is available at https://github.com/DebugML/sqrl.

----

## [1067] Sample and Predict Your Latent: Modality-free Sequential Disentanglement via Contrastive Estimation

**Authors**: *Ilan Naiman, Nimrod Berman, Omri Azencot*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/naiman23a.html](https://proceedings.mlr.press/v202/naiman23a.html)

**Abstract**:

Unsupervised disentanglement is a long-standing challenge in representation learning. Recently, self-supervised techniques achieved impressive results in the sequential setting, where data is time-dependent. However, the latter methods employ modality-based data augmentations and random sampling or solve auxiliary tasks. In this work, we propose to avoid that by generating, sampling, and comparing empirical distributions from the underlying variational model. Unlike existing work, we introduce a self-supervised sequential disentanglement framework based on contrastive estimation with no external signals, while using common batch sizes and samples from the latent space itself. In practice, we propose a unified, efficient, and easy-to-code sampling strategy for semantically similar and dissimilar views of the data. We evaluate our approach on video, audio, and time series benchmarks. Our method presents state-of-the-art results in comparison to existing techniques. The code is available at https://github.com/azencot-group/SPYL.

----

## [1068] Effectively Using Public Data in Privacy Preserving Machine Learning

**Authors**: *Milad Nasr, Saeed Mahloujifar, Xinyu Tang, Prateek Mittal, Amir Houmansadr*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/nasr23a.html](https://proceedings.mlr.press/v202/nasr23a.html)

**Abstract**:

Differentially private (DP) machine learning techniques are notorious for their degradation of model utility (e.g., they degrade classification accuracy). A recent line of work has demonstrated that leveraging public data can improve the trade-off between privacy and utility when training models with DP guaranteed. In this work, we further explore the potential of using public data in DP models, showing that utility gains can in fact be significantly higher than what shown in prior works. Specifically, we introduce DOPE-SGD, a modified DP-SGD algorithm that leverages public data during its training. DOPE-SGD uses public data in two complementary ways: (1) it uses advance augmentation techniques that leverages public data to generate synthetic data that is effectively embedded in multiple steps of the training pipeline; (2) it uses a modified gradient clipping mechanism (which is a standard technique in DP training) to change the origin of gradient vectors using the information inferred from available public and synthetic data, therefore boosting utility. We also introduce a technique to ensemble intermediate DP models by leveraging the post processing property of differential privacy to further improve the accuracy of the predictions. Our experimental results demonstrate the effectiveness of our approach in improving the state-of-the-art in DP machine learning across multiple datasets, network architectures, and application domains. For instance, assuming access to $2,000$ public images, and for a privacy budget of $\varepsilon=2,\delta=10^{-5}$, our technique achieves an accuracy of $75.1%$ on CIFAR10, significantly higher than $68.1%$ achieved by the state of the art.

----

## [1069] Counterfactual Identifiability of Bijective Causal Models

**Authors**: *Arash Nasr-Esfahany, Mohammad Alizadeh, Devavrat Shah*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/nasr-esfahany23a.html](https://proceedings.mlr.press/v202/nasr-esfahany23a.html)

**Abstract**:

We study counterfactual identifiability in causal models with bijective generation mechanisms (BGM), a class that generalizes several widely-used causal models in the literature. We establish their counterfactual identifiability for three common causal structures with unobserved confounding, and propose a practical learning method that casts learning a BGM as structured generative modeling. Learned BGMs enable efficient counterfactual estimation and can be obtained using a variety of deep conditional generative models. We evaluate our techniques in a visual task and demonstrate its application in a real-world video streaming simulation task.

----

## [1070] Discovering Object-Centric Generalized Value Functions From Pixels

**Authors**: *Somjit Nath, Gopeshh Raaj Subbaraj, Khimya Khetarpal, Samira Ebrahimi Kahou*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/nath23a.html](https://proceedings.mlr.press/v202/nath23a.html)

**Abstract**:

Deep Reinforcement Learning has shown significant progress in extracting useful representations from high-dimensional inputs albeit using hand-crafted auxiliary tasks and pseudo rewards. Automatically learning such representations in an object-centric manner geared towards control and fast adaptation remains an open research problem. In this paper, we introduce a method that tries to discover meaningful features from objects, translating them to temporally coherent ‘question’ functions and leveraging the subsequent learned general value functions for control. We compare our approach with state-of-the-art techniques alongside other ablations and show competitive performance in both stationary and non-stationary settings. Finally, we also investigate the discovered general value functions and through qualitative analysis show that the learned representations are not only interpretable but also, centered around objects that are invariant to changes across tasks facilitating fast adaptation.

----

## [1071] On Many-Actions Policy Gradient

**Authors**: *Michal Nauman, Marek Cygan*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/nauman23a.html](https://proceedings.mlr.press/v202/nauman23a.html)

**Abstract**:

We study the variance of stochastic policy gradients (SPGs) with many action samples per state. We derive a many-actions optimality condition, which determines when many-actions SPG yields lower variance as compared to a single-action agent with proportionally extended trajectory. We propose Model-Based Many-Actions (MBMA), an approach leveraging dynamics models for many-actions sampling in the context of SPG. MBMA addresses issues associated with existing implementations of many-actions SPG and yields lower bias and comparable variance to SPG estimated from states in model-simulated rollouts. We find that MBMA bias and variance structure matches that predicted by theory. As a result, MBMA achieves improved sample efficiency and higher returns on a range of continuous action environments as compared to model-free, many-actions, and model-based on-policy SPG baselines.

----

## [1072] Equivariant Architectures for Learning in Deep Weight Spaces

**Authors**: *Aviv Navon, Aviv Shamsian, Idan Achituve, Ethan Fetaya, Gal Chechik, Haggai Maron*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/navon23a.html](https://proceedings.mlr.press/v202/navon23a.html)

**Abstract**:

Designing machine learning architectures for processing neural networks in their raw weight matrix form is a newly introduced research direction. Unfortunately, the unique symmetry structure of deep weight spaces makes this design very challenging. If successful, such architectures would be capable of performing a wide range of intriguing tasks, from adapting a pre-trained network to a new domain to editing objects represented as functions (INRs or NeRFs). As a first step towards this goal, we present here a novel network architecture for learning in deep weight spaces. It takes as input a concatenation of weights and biases of a pre-trained MLP and processes it using a composition of layers that are equivariant to the natural permutation symmetry of the MLP’s weights: Changing the order of neurons in intermediate layers of the MLP does not affect the function it represents. We provide a full characterization of all affine equivariant and invariant layers for these symmetries and show how these layers can be implemented using three basic operations: pooling, broadcasting, and fully connected layers applied to the input in an appropriate manner. We demonstrate the effectiveness of our architecture and its advantages over natural baselines in a variety of learning tasks.

----

## [1073] Scalable Multi-Agent Reinforcement Learning through Intelligent Information Aggregation

**Authors**: *Siddharth Nayak, Kenneth Choi, Wenqi Ding, Sydney Dolan, Karthik Gopalakrishnan, Hamsa Balakrishnan*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/nayak23a.html](https://proceedings.mlr.press/v202/nayak23a.html)

**Abstract**:

We consider the problem of multi-agent navigation and collision avoidance when observations are limited to the local neighborhood of each agent. We propose InforMARL, a novel architecture for multi-agent reinforcement learning (MARL) which uses local information intelligently to compute paths for all the agents in a decentralized manner. Specifically, InforMARL aggregates information about the local neighborhood of agents for both the actor and the critic using a graph neural network and can be used in conjunction with any standard MARL algorithm. We show that (1) in training, InforMARL has better sample efficiency and performance than baseline approaches, despite using less information, and (2) in testing, it scales well to environments with arbitrary numbers of agents and obstacles. We illustrate these results using four task environments, including one with predetermined goals for each agent, and one in which the agents collectively try to cover all goals.

----

## [1074] Geometric Autoencoders - What You See is What You Decode

**Authors**: *Philipp Nazari, Sebastian Damrich, Fred A. Hamprecht*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/nazari23a.html](https://proceedings.mlr.press/v202/nazari23a.html)

**Abstract**:

Visualization is a crucial step in exploratory data analysis. One possible approach is to train an autoencoder with low-dimensional latent space. Large network depth and width can help unfolding the data. However, such expressive networks can achieve low reconstruction error even when the latent representation is distorted. To avoid such misleading visualizations, we propose first a differential geometric perspective on the decoder, leading to insightful diagnostics for an embedding’s distortion, and second a new regularizer mitigating such distortion. Our “Geometric Autoencoder” avoids stretching the embedding spuriously, so that the visualization captures the data structure more faithfully. It also flags areas where little distortion could not be achieved, thus guarding against misinterpretation.

----

## [1075] Action Matching: Learning Stochastic Dynamics from Samples

**Authors**: *Kirill Neklyudov, Rob Brekelmans, Daniel Severo, Alireza Makhzani*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/neklyudov23a.html](https://proceedings.mlr.press/v202/neklyudov23a.html)

**Abstract**:

Learning the continuous dynamics of a system from snapshots of its temporal marginals is a problem which appears throughout natural sciences and machine learning, including in quantum systems, single-cell biological data, and generative modeling. In these settings, we assume access to cross-sectional samples that are uncorrelated over time, rather than full trajectories of samples. In order to better understand the systems under observation, we would like to learn a model of the underlying process that allows us to propagate samples in time and thereby simulate entire individual trajectories. In this work, we propose Action Matching, a method for learning a rich family of dynamics using only independent samples from its time evolution. We derive a tractable training objective, which does not rely on explicit assumptions about the underlying dynamics and does not require back-propagation through differential equations or optimal transport solvers. Inspired by connections with optimal transport, we derive extensions of Action Matching to learn stochastic differential equations and dynamics involving creation and destruction of probability mass. Finally, we showcase applications of Action Matching by achieving competitive performance in a diverse set of experiments from biology, physics, and generative modeling.

----

## [1076] Extending Conformal Prediction to Hidden Markov Models with Exact Validity via de Finetti's Theorem for Markov Chains

**Authors**: *Buddhika Nettasinghe, Samrat Chatterjee, Ramakrishna Tipireddy, Mahantesh M. Halappanavar*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/nettasinghe23a.html](https://proceedings.mlr.press/v202/nettasinghe23a.html)

**Abstract**:

Conformal prediction is a widely used method to quantify the uncertainty of a classifier under the assumption of exchangeability (e.g., IID data). We generalize conformal prediction to the Hidden Markov Model (HMM) framework where the assumption of exchangeability is not valid. The key idea of the proposed method is to partition the non-exchangeable Markovian data from the HMM into exchangeable blocks by exploiting the de Finetti’s Theorem for Markov Chains discovered by Diaconis and Freedman (1980). The permutations of the exchangeable blocks are viewed as randomizations of the observed Markovian data from the HMM. The proposed method provably retains all desirable theoretical guarantees offered by the classical conformal prediction framework in both exchangeable and Markovian settings. In particular, while the lack of exchangeability introduced by Markovian samples constitutes a violation of a crucial assumption for classical conformal prediction, the proposed method views it as an advantage that can be exploited to improve the performance further. Detailed numerical and empirical results that complement the theoretical conclusions are provided to illustrate the practical feasibility of the proposed method.

----

## [1077] ClimaX: A foundation model for weather and climate

**Authors**: *Tung Nguyen, Johannes Brandstetter, Ashish Kapoor, Jayesh K. Gupta, Aditya Grover*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/nguyen23a.html](https://proceedings.mlr.press/v202/nguyen23a.html)

**Abstract**:

Recent data-driven approaches based on machine learning aim to directly solve a downstream forecasting or projection task by learning a data-driven functional mapping using deep neural networks. However, these networks are trained using curated and homogeneous climate datasets for specific spatiotemporal tasks, and thus lack the generality of currently used computationally intensive physics-informed numerical models for weather and climate modeling. We develop and demonstrate ClimaX, a flexible and generalizable deep learning model for weather and climate science that can be trained using heterogeneous datasets spanning different variables, spatio-temporal coverage, and physical groundings. ClimaX extends the Transformer architecture with novel encoding and aggregation blocks that allow effective use of available compute and data while maintaining general utility. ClimaX is pretrained with a self-supervised learning objective on climate datasets derived from CMIP6. The pretrained ClimaX can then be fine-tuned to address a breadth of climate and weather tasks, including those that involve atmospheric variables and spatio-temporal scales unseen during pretraining. Compared to existing data-driven baselines, we show that this generality in ClimaX results in superior performance on benchmarks for weather forecasting and climate projections, even when pretrained at lower resolutions and compute budgets. Our source code is available at https://github.com/microsoft/ClimaX.

----

## [1078] Provable Reset-free Reinforcement Learning by No-Regret Reduction

**Authors**: *Hoai-An Nguyen, Ching-An Cheng*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/nguyen23b.html](https://proceedings.mlr.press/v202/nguyen23b.html)

**Abstract**:

Reinforcement learning (RL) so far has limited real-world applications. One key challenge is that typical RL algorithms heavily rely on a reset mechanism to sample proper initial states; these reset mechanisms, in practice, are expensive to implement due to the need for human intervention or heavily engineered environments. To make learning more practical, we propose a generic no-regret reduction to systematically design reset-free RL algorithms. Our reduction turns the reset-free RL problem into a two-player game. We show that achieving sublinear regret in this two-player game would imply learning a policy that has both sublinear performance regret and sublinear total number of resets in the original RL problem. This means that the agent eventually learns to perform optimally and avoid resets. To demonstrate the effectiveness of this reduction, we design an instantiation for linear Markov decision processes, which is the first provably correct reset-free RL algorithm.

----

## [1079] Revisiting Over-smoothing and Over-squashing Using Ollivier-Ricci Curvature

**Authors**: *Khang Nguyen, Nong Minh Hieu, Vinh Duc Nguyen, Nhat Ho, Stanley J. Osher, Tan Minh Nguyen*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/nguyen23c.html](https://proceedings.mlr.press/v202/nguyen23c.html)

**Abstract**:

Graph Neural Networks (GNNs) had been demonstrated to be inherently susceptible to the problems of over-smoothing and over-squashing. These issues prohibit the ability of GNNs to model complex graph interactions by limiting their effectiveness in taking into account distant information. Our study reveals the key connection between the local graph geometry and the occurrence of both of these issues, thereby providing a unified framework for studying them at a local scale using the Ollivier-Ricci curvature. Specifically, we demonstrate that over-smoothing is linked to positive graph curvature while over-squashing is linked to negative graph curvature. Based on our theory, we propose the Batch Ollivier-Ricci Flow, a novel rewiring algorithm capable of simultaneously addressing both over-smoothing and over-squashing.

----

## [1080] Deep Clustering with Incomplete Noisy Pairwise Annotations: A Geometric Regularization Approach

**Authors**: *Tri Nguyen, Shahana Ibrahim, Xiao Fu*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/nguyen23d.html](https://proceedings.mlr.press/v202/nguyen23d.html)

**Abstract**:

The recent integration of deep learning and pairwise similarity annotation-based constrained clustering—i.e., deep constrained clustering (DCC)—has proven effective for incorporating weak supervision into massive data clustering: Less than 1% of pair similarity annotations can often substantially enhance the clustering accuracy. However, beyond empirical successes, there is a lack of understanding of DCC. In addition, many DCC paradigms are sensitive to annotation noise, but performance-guaranteed noisy DCC methods have been largely elusive. This work first takes a deep look into a recently emerged logistic loss function of DCC, and characterizes its theoretical properties. Our result shows that the logistic DCC loss ensures the identifiability of data membership under reasonable conditions, which may shed light on its effectiveness in practice. Building upon this understanding, a new loss function based on geometric factor analysis is proposed to fend against noisy annotations. It is shown that even under unknown annotation confusions, the data membership can still be provably identified under our proposed learning criterion. The proposed approach is tested over multiple datasets to validate our claims.

----

## [1081] Self-Attention Amortized Distributional Projection Optimization for Sliced Wasserstein Point-Cloud Reconstruction

**Authors**: *Khai Nguyen, Dang Nguyen, Nhat Ho*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/nguyen23e.html](https://proceedings.mlr.press/v202/nguyen23e.html)

**Abstract**:

Max sliced Wasserstein (Max-SW) distance has been widely known as a solution for less discriminative projections of sliced Wasserstein (SW) distance. In applications that have various independent pairs of probability measures, amortized projection optimization is utilized to predict the “max" projecting directions given two input measures instead of using projected gradient ascent multiple times. Despite being efficient, Max-SW and its amortized version cannot guarantee metricity property due to the sub-optimality of the projected gradient ascent and the amortization gap. Therefore, we propose to replace Max-SW with distributional sliced Wasserstein distance with von Mises-Fisher (vMF) projecting distribution (v-DSW). Since v-DSW is a metric with any non-degenerate vMF distribution, its amortized version can guarantee the metricity when performing amortization. Furthermore, current amortized models are not permutation invariant and symmetric. To address the issue, we design amortized models based on self-attention architecture. In particular, we adopt efficient self-attention architectures to make the computation linear in the number of supports. With the two improvements, we derive self-attention amortized distributional projection optimization and show its appealing performance in point-cloud reconstruction and its downstream applications

----

## [1082] Building Neural Networks on Matrix Manifolds: A Gyrovector Space Approach

**Authors**: *Xuan Son Nguyen, Shuo Yang*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/nguyen23f.html](https://proceedings.mlr.press/v202/nguyen23f.html)

**Abstract**:

Matrix manifolds, such as manifolds of Symmetric Positive Definite (SPD) matrices and Grassmann manifolds, appear in many applications. Recently, by applying the theory of gyrogroups and gyrovector spaces that is a powerful framework for studying hyperbolic geometry, some works have attempted to build principled generalizations of Euclidean neural networks on matrix manifolds. However, due to the lack of many concepts in gyrovector spaces for the considered manifolds, e.g., the inner product and gyroangles, techniques and mathematical tools provided by these works are still limited compared to those developed for studying hyperbolic geometry. In this paper, we generalize some notions in gyrovector spaces for SPD and Grassmann manifolds, and propose new models and layers for building neural networks on these manifolds. We show the effectiveness of our approach in two applications, i.e., human action recognition and knowledge graph completion.

----

## [1083] Simple Disentanglement of Style and Content in Visual Representations

**Authors**: *Lilian Ngweta, Subha Maity, Alex Gittens, Yuekai Sun, Mikhail Yurochkin*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/ngweta23a.html](https://proceedings.mlr.press/v202/ngweta23a.html)

**Abstract**:

Learning visual representations with interpretable features, i.e., disentangled representations, remains a challenging problem. Existing methods demonstrate some success but are hard to apply to large-scale vision datasets like ImageNet. In this work, we propose a simple post-processing framework to disentangle content and style in learned representations from pre-trained vision models. We model the pre-trained features probabilistically as linearly entangled combinations of the latent content and style factors and develop a simple disentanglement algorithm based on the probabilistic model. We show that the method provably disentangles content and style features and verify its efficacy empirically. Our post-processed features yield significant domain generalization performance improvements when the distribution shift occurs due to style changes or style-related spurious correlations.

----

## [1084] MetaDiffuser: Diffusion Model as Conditional Planner for Offline Meta-RL

**Authors**: *Fei Ni, Jianye Hao, Yao Mu, Yifu Yuan, Yan Zheng, Bin Wang, Zhixuan Liang*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/ni23a.html](https://proceedings.mlr.press/v202/ni23a.html)

**Abstract**:

Recently, diffusion model shines as a promising backbone for the sequence modeling paradigm in offline reinforcement learning(RL). However, these works mostly lack the generalization ability across tasks with reward or dynamics change. To tackle this challenge, in this paper we propose a task-oriented conditioned diffusion planner for offline meta-RL(MetaDiffuser), which considers the generalization problem as conditional trajectory generation task with contextual representation. The key is to learn a context conditioned diffusion model which can generate task-oriented trajectories for planning across diverse tasks. To enhance the dynamics consistency of the generated trajectories while encouraging trajectories to achieve high returns, we further design a dual-guided module in the sampling process of the diffusion model. The proposed framework enjoys the robustness to the quality of collected warm-start data from the testing task and the flexibility to incorporate with different task representation method. The experiment results on MuJoCo benchmarks show that MetaDiffuser outperforms other strong offline meta-RL baselines, demonstrating the outstanding conditional generation ability of diffusion architecture.

----

## [1085] LEVER: Learning to Verify Language-to-Code Generation with Execution

**Authors**: *Ansong Ni, Srini Iyer, Dragomir Radev, Veselin Stoyanov, Wen-Tau Yih, Sida I. Wang, Xi Victoria Lin*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/ni23b.html](https://proceedings.mlr.press/v202/ni23b.html)

**Abstract**:

The advent of large language models trained on code (code LLMs) has led to significant progress in language-to-code generation. State-of-the-art approaches in this area combine LLM decoding with sample pruning and reranking using test cases or heuristics based on the execution results. However, it is challenging to obtain test cases for many real-world language-to-code applications, and heuristics cannot well capture the semantic features of the execution results, such as data type and value range, which often indicates the correctness of the program. In this work, we propose LEVER, a simple approach to improve language-to-code generation by learning to verify the generated programs with their execution results. Specifically, we train verifiers to determine whether a program sampled from the LLMs is correct or not based on the natural language input, the program itself and its execution results. The sampled programs are reranked by combining the verification score with the LLM generation probability, and marginalizing over programs with the same execution results. On four datasets across the domains of table QA, math QA and basic Python programming, LEVER consistently improves over the base code LLMs (4.6% to 10.9% with code-davinci-002) and achieves new state-of-the-art results on all of them.

----

## [1086] Continual Vision-Language Representation Learning with Off-Diagonal Information

**Authors**: *Zixuan Ni, Longhui Wei, Siliang Tang, Yueting Zhuang, Qi Tian*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/ni23c.html](https://proceedings.mlr.press/v202/ni23c.html)

**Abstract**:

Large-scale multi-modal contrastive learning frameworks like CLIP typically require a large amount of image-text samples for training. However, these samples are always collected continuously in real scenarios. This paper discusses the feasibility of continual CLIP training using streaming data. Unlike continual learning based on self-supervised learning methods for pure images, which is empirically robust against catastrophic forgetting, CLIP’s performance degeneration in the continual setting is significant and non-neglectable. By analyzing the changes in the model’s representation space during continual CLIP training from a spatial geometry perspective, we explore and summarize these spatial variations as Spatial Disorder (SD), which can be divided into Intra-modal Rotation and Inter-modal Deviation. Moreover, we empirically and theoretically demonstrate how SD leads to a performance decline for CLIP on cross-modal retrieval tasks. To alleviate SD, we propose a new continual vision-language representation learning framework Mod-X: Maintain off-diagonal information-matriX. By selectively aligning the off-diagonal information distribution of contrastive matrices, the Mod-X improves the capability of the multi-modal model by maintaining the multi-modal representation space alignment on the old data domain during continuously fitting the new training data domain. Experiments on commonly used datasets with different scales and scopes have demonstrated the effectiveness of our method.

----

## [1087] Attributing Image Generative Models using Latent Fingerprints

**Authors**: *Guangyu Nie, Changhoon Kim, Yezhou Yang, Yi Ren*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/nie23a.html](https://proceedings.mlr.press/v202/nie23a.html)

**Abstract**:

Generative models have enabled the creation of contents that are indistinguishable from those taken from nature. Open-source development of such models raised concerns about the risks of their misuse for malicious purposes. One potential risk mitigation strategy is to attribute generative models via fingerprinting. Current fingerprinting methods exhibit a significant tradeoff between robust attribution accuracy and generation quality while lacking design principles to improve this tradeoff. This paper investigates the use of latent semantic dimensions as fingerprints, from where we can analyze the effects of design variables, including the choice of fingerprinting dimensions, strength, and capacity, on the accuracy-quality tradeoff. Compared with previous SOTA, our method requires minimum computation and is more applicable to large-scale models. We use StyleGAN2 and the latent diffusion model to demonstrate the efficacy of our method.

----

## [1088] A Framework for Adapting Offline Algorithms to Solve Combinatorial Multi-Armed Bandit Problems with Bandit Feedback

**Authors**: *Guanyu Nie, Yididiya Y. Nadew, Yanhui Zhu, Vaneet Aggarwal, Christopher John Quinn*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/nie23b.html](https://proceedings.mlr.press/v202/nie23b.html)

**Abstract**:

We investigate the problem of stochastic, combinatorial multi-armed bandits where the learner only has access to bandit feedback and the reward function can be non-linear. We provide a general framework for adapting discrete offline approximation algorithms into sublinear $\alpha$-regret methods that only require bandit feedback, achieving $\mathcal{O}\left(T^\frac{2}{3}\log(T)^\frac{1}{3}\right)$ expected cumulative $\alpha$-regret dependence on the horizon $T$. The framework only requires the offline algorithms to be robust to small errors in function evaluation. The adaptation procedure does not even require explicit knowledge of the offline approximation algorithm — the offline algorithm can be used as black box subroutine. To demonstrate the utility of the proposed framework, the proposed framework is applied to multiple problems in submodular maximization, adapting approximation algorithms for cardinality and for knapsack constraints. The new CMAB algorithms for knapsack constraints outperform a full-bandit method developed for the adversarial setting in experiments with real-world data.

----

## [1089] SinFusion: Training Diffusion Models on a Single Image or Video

**Authors**: *Yaniv Nikankin, Niv Haim, Michal Irani*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/nikankin23a.html](https://proceedings.mlr.press/v202/nikankin23a.html)

**Abstract**:

Diffusion models exhibited tremendous progress in image and video generation, exceeding GANs in quality and diversity. However, they are usually trained on very large datasets and are not naturally adapted to manipulate a given input image or video. In this paper we show how this can be resolved by training a diffusion model on a single input image or video. Our image/video-specific diffusion model (SinFusion) learns the appearance and dynamics of the single image or video, while utilizing the conditioning capabilities of diffusion models. It can solve a wide array of image/video-specific manipulation tasks. In particular, our model can learn from few frames the motion and dynamics of a single input video. It can then generate diverse new video samples of the same dynamic scene, extrapolate short videos into long ones (both forward and backward in time) and perform video upsampling. Most of these tasks are not realizable by current video-specific generation methods.

----

## [1090] SparseProp: Efficient Sparse Backpropagation for Faster Training of Neural Networks at the Edge

**Authors**: *Mahdi Nikdan, Tommaso Pegolotti, Eugenia Iofinova, Eldar Kurtic, Dan Alistarh*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/nikdan23a.html](https://proceedings.mlr.press/v202/nikdan23a.html)

**Abstract**:

We provide an efficient implementation of the backpropagation algorithm, specialized to the case where the weights of the neural network being trained are sparse. Our algorithm is general, as it applies to arbitrary (unstructured) sparsity and common layer types (e.g., convolutional or linear). We provide a fast vectorized implementation on commodity CPUs, and show that it can yield speedups in end-to-end runtime experiments, both in transfer learning using already-sparsified networks, and in training sparse networks from scratch. Thus, our results provide the first support for sparse training on commodity hardware.

----

## [1091] Anti-Exploration by Random Network Distillation

**Authors**: *Alexander Nikulin, Vladislav Kurenkov, Denis Tarasov, Sergey Kolesnikov*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/nikulin23a.html](https://proceedings.mlr.press/v202/nikulin23a.html)

**Abstract**:

Despite the success of Random Network Distillation (RND) in various domains, it was shown as not discriminative enough to be used as an uncertainty estimator for penalizing out-of-distribution actions in offline reinforcement learning. In this paper, we revisit these results and show that, with a naive choice of conditioning for the RND prior, it becomes infeasible for the actor to effectively minimize the anti-exploration bonus and discriminativity is not an issue. We show that this limitation can be avoided with conditioning based on Feature-wise Linear Modulation (FiLM), resulting in a simple and efficient ensemble-free algorithm based on Soft Actor-Critic. We evaluate it on the D4RL benchmark, showing that it is capable of achieving performance comparable to ensemble-based methods and outperforming ensemble-free approaches by a wide margin.

----

## [1092] Input Perturbation Reduces Exposure Bias in Diffusion Models

**Authors**: *Mang Ning, Enver Sangineto, Angelo Porrello, Simone Calderara, Rita Cucchiara*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/ning23a.html](https://proceedings.mlr.press/v202/ning23a.html)

**Abstract**:

Denoising Diffusion Probabilistic Models have shown an impressive generation quality although their long sampling chain leads to high computational costs. In this paper, we observe that a long sampling chain also leads to an error accumulation phenomenon, which is similar to the exposure bias problem in autoregressive text generation. Specifically, we note that there is a discrepancy between training and testing, since the former is conditioned on the ground truth samples, while the latter is conditioned on the previously generated results. To alleviate this problem, we propose a very simple but effective training regularization, consisting in perturbing the ground truth samples to simulate the inference time prediction errors. We empirically show that, without affecting the recall and precision, the proposed input perturbation leads to a significant improvement in the sample quality while reducing both the training and the inference times. For instance, on CelebA 64x64, we achieve a new state-of-the-art FID score of 1.27, while saving 37.5% of the training time. The code is available at https://github.com/forever208/DDPM-IP

----

## [1093] Primal and Dual Analysis of Entropic Fictitious Play for Finite-sum Problems

**Authors**: *Atsushi Nitanda, Kazusato Oko, Denny Wu, Nobuhito Takenouchi, Taiji Suzuki*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/nitanda23a.html](https://proceedings.mlr.press/v202/nitanda23a.html)

**Abstract**:

The entropic fictitious play (EFP) is a recently proposed algorithm that minimizes the sum of a convex functional and entropy in the space of measures — such an objective naturally arises in the optimization of a two-layer neural network in the mean-field regime. In this work, we provide a concise primal-dual analysis of EFP in the setting where the learning problem exhibits a finite-sum structure. We establish quantitative global convergence guarantees for both the continuous-time and discrete-time dynamics based on properties of a proximal Gibbs measure introduced in Nitanda et al. (2022). Furthermore, our primal-dual framework entails a memory-efficient particle-based implementation of the EFP update, and also suggests a connection to gradient boosting methods. We illustrate the efficiency of our novel implementation in experiments including neural network optimization and image synthesis.

----

## [1094] The Statistical Scope of Multicalibration

**Authors**: *Georgy Noarov, Aaron Roth*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/noarov23a.html](https://proceedings.mlr.press/v202/noarov23a.html)

**Abstract**:

We make a connection between multicalibration and property elicitation and show that (under mild technical conditions) it is possible to produce a multicalibrated predictor for a continuous scalar property $\Gamma$ if and only if $\Gamma$ is elicitable. On the negative side, we show that for non-elicitable continuous properties there exist simple data distributions on which even the true distributional predictor is not calibrated. On the positive side, for elicitable $\Gamma$, we give simple canonical algorithms for the batch and the online adversarial setting, that learn a $\Gamma$-multicalibrated predictor. This generalizes past work on multicalibrated means and quantiles, and in fact strengthens existing online quantile multicalibration results. To further counter-weigh our negative result, we show that if a property $\Gamma^1$ is not elicitable by itself, but is elicitable conditionally on another elicitable property $\Gamma^0$, then there is a canonical algorithm that jointly multicalibrates $\Gamma^1$ and $\Gamma^0$; this generalizes past work on mean-moment multicalibration. Finally, as applications of our theory, we provide novel algorithmic and impossibility results for fair (multicalibrated) risk assessment.

----

## [1095] Do Embodied Agents Dream of Pixelated Sheep: Embodied Decision Making using Language Guided World Modelling

**Authors**: *Kolby Nottingham, Prithviraj Ammanabrolu, Alane Suhr, Yejin Choi, Hannaneh Hajishirzi, Sameer Singh, Roy Fox*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/nottingham23a.html](https://proceedings.mlr.press/v202/nottingham23a.html)

**Abstract**:

Reinforcement learning (RL) agents typically learn tabula rasa, without prior knowledge of the world. However, if initialized with knowledge of high-level subgoals and transitions between subgoals, RL agents could utilize this Abstract World Model (AWM) for planning and exploration. We propose using few-shot large language models (LLMs) to hypothesize an AWM, that will be verified through world experience, to improve sample efficiency of RL agents. Our DECKARD agent applies LLM-guided exploration to item crafting in Minecraft in two phases: (1) the Dream phase where the agent uses an LLM to decompose a task into a sequence of subgoals, the hypothesized AWM; and (2) the Wake phase where the agent learns a modular policy for each subgoal and verifies or corrects the hypothesized AWM. Our method of hypothesizing an AWM with LLMs and then verifying the AWM based on agent experience not only increases sample efficiency over contemporary methods by an order of magnitude but is also robust to and corrects errors in the LLM, successfully blending noisy internet-scale information from LLMs with knowledge grounded in environment dynamics.

----

## [1096] Gradient-Free Structured Pruning with Unlabeled Data

**Authors**: *Azade Nova, Hanjun Dai, Dale Schuurmans*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/nova23a.html](https://proceedings.mlr.press/v202/nova23a.html)

**Abstract**:

Large Language Models (LLMs) have achieved great success in solving difficult tasks across many domains, but such success comes with a high computation cost, and inference latency. As developers and third parties customize these models, the need to provide efficient inference has increased. Many efforts have attempted to reduce inference cost through model compression techniques such as pruning and distillation. However, these techniques either require labeled data, or are time-consuming as they require the compressed model to be retrained to regain accuracy. In this paper, we propose a gradient-free structured pruning framework that uses only unlabeled data. An evaluation on the GLUE and SQuAD benchmarks using BERT$_{BASE}$ and DistilBERT illustrates the effectiveness of the proposed approach. By only using the weights of the pre-trained model and unlabeled data, in a matter of a few minutes on a single GPU, up to 40% of the original FLOP count can be reduced with less than a $4%$ accuracy loss across all tasks considered.

----

## [1097] CHiLS: Zero-Shot Image Classification with Hierarchical Label Sets

**Authors**: *Zachary Novack, Julian J. McAuley, Zachary Chase Lipton, Saurabh Garg*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/novack23a.html](https://proceedings.mlr.press/v202/novack23a.html)

**Abstract**:

Open vocabulary models (e.g. CLIP) have shown strong performance on zero-shot classification through their ability generate embeddings for each class based on their (natural language) names. Prior work has focused on improving the accuracy of these models through prompt engineering or by incorporating a small amount of labeled downstream data (via finetuning). However, there has been little focus on improving the richness of the class names themselves, which can pose issues when class labels are coarsely-defined and are uninformative. We propose Classification with Hierarchical Label Sets (or CHiLS), an alternative strategy for zero-shot classification specifically designed for datasets with implicit semantic hierarchies. CHiLS proceeds in three steps: (i) for each class, produce a set of subclasses, using either existing label hierarchies or by querying GPT-3; (ii) perform the standard zero-shot CLIP procedure as though these subclasses were the labels of interest; (iii) map the predicted subclass back to its parent to produce the final prediction. Across numerous datasets with underlying hierarchical structure, CHiLS leads to improved accuracy in situations both with and without ground-truth hierarchical information. CHiLS is simple to implement within existing zero-shot pipelines and requires no additional training cost. Code is available at: https://github.com/acmi-lab/CHILS.

----

## [1098] Few-bit Backward: Quantized Gradients of Activation Functions for Memory Footprint Reduction

**Authors**: *Georgii Sergeevich Novikov, Daniel Bershatsky, Julia Gusak, Alex Shonenkov, Denis Valerievich Dimitrov, Ivan V. Oseledets*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/novikov23a.html](https://proceedings.mlr.press/v202/novikov23a.html)

**Abstract**:

Memory footprint is one of the main limiting factors for large neural network training. In backpropagation, one needs to store the input to each operation in the computational graph. Every modern neural network model has quite a few pointwise nonlinearities in its architecture, and such operations induce additional memory costs that, as we show, can be significantly reduced by quantization of the gradients. We propose a systematic approach to compute optimal quantization of the retained gradients of the pointwise nonlinear functions with only a few bits per each element. We show that such approximation can be achieved by computing an optimal piecewise-constant approximation of the derivative of the activation function, which can be done by dynamic programming. The drop-in replacements are implemented for all popular nonlinearities and can be used in any existing pipeline. We confirm the memory reduction and the same convergence on several open benchmarks.

----

## [1099] Efficient Exploration via Epistemic-Risk-Seeking Policy Optimization

**Authors**: *Brendan O'Donoghue*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/o-donoghue23a.html](https://proceedings.mlr.press/v202/o-donoghue23a.html)

**Abstract**:

Exploration remains a key challenge in deep reinforcement learning (RL). Optimism in the face of uncertainty is a well-known heuristic with theoretical guarantees in the tabular setting, but how best to translate the principle to deep reinforcement learning, which involves online stochastic gradients and deep network function approximators, is not fully understood. In this paper we propose a new, differentiable optimistic objective that when optimized yields a policy that provably explores efficiently, with guarantees even under function approximation. Our new objective is a zero-sum two-player game derived from endowing the agent with an epistemic-risk-seeking utility function, which converts uncertainty into value and encourages the agent to explore uncertain states. We show that the solution to this game minimizes an upper bound on the regret, with the ’players’ each attempting to minimize one component of a particular regret decomposition. We derive a new model-free algorithm which we call ’epistemic-risk-seeking actor-critic’ (ERSAC), which is simply an application of simultaneous stochastic gradient ascent-descent to the game. Finally, we discuss a recipe for incorporating off-policy data and show that combining the risk-seeking objective with replay data yields a double benefit in terms of statistical efficiency. We conclude with some results showing good performance of a deep RL agent using the technique on the challenging ’DeepSea’ environment, showing significant performance improvements even over other efficient exploration techniques, as well as improved performance on the Atari benchmark.

----

## [1100] Provable Benefit of Mixup for Finding Optimal Decision Boundaries

**Authors**: *Junsoo Oh, Chulhee Yun*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/oh23a.html](https://proceedings.mlr.press/v202/oh23a.html)

**Abstract**:

We investigate how pair-wise data augmentation techniques like Mixup affect the sample complexity of finding optimal decision boundaries in a binary linear classification problem. For a family of data distributions with a separability constant $\kappa$, we analyze how well the optimal classifier in terms of training loss aligns with the optimal one in test accuracy (i.e., Bayes optimal classifier). For vanilla training without augmentation, we uncover an interesting phenomenon named the curse of separability. As we increase $\kappa$ to make the data distribution more separable, the sample complexity of vanilla training increases exponentially in $\kappa$; perhaps surprisingly, the task of finding optimal decision boundaries becomes harder for more separable distributions. For Mixup training, we show that Mixup mitigates this problem by significantly reducing the sample complexity. To this end, we develop new concentration results applicable to $n^2$ pair-wise augmented data points constructed from $n$ independent data, by carefully dealing with dependencies between overlapping pairs. Lastly, we study other masking-based Mixup-style techniques and show that they can distort the training loss and make its minimizer converge to a suboptimal classifier in terms of test accuracy.

----

## [1101] Shedding a PAC-Bayesian Light on Adaptive Sliced-Wasserstein Distances

**Authors**: *Ruben Ohana, Kimia Nadjahi, Alain Rakotomamonjy, Liva Ralaivola*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/ohana23a.html](https://proceedings.mlr.press/v202/ohana23a.html)

**Abstract**:

The Sliced-Wasserstein distance (SW) is a computationally efficient and theoretically grounded alternative to the Wasserstein distance. Yet, the literature on its statistical properties – or, more accurately, its generalization properties – with respect to the distribution of slices, beyond the uniform measure, is scarce. To bring new contributions to this line of research, we leverage the PAC-Bayesian theory and a central observation that SW may be interpreted as an average risk, the quantity PAC-Bayesian bounds have been designed to characterize. We provide three types of results: i) PAC-Bayesian generalization bounds that hold on what we refer as adaptive Sliced-Wasserstein distances, i.e. SW defined with respect to arbitrary distributions of slices (among which data-dependent distributions), ii) a principled procedure to learn the distribution of slices that yields maximally discriminative SW, by optimizing our theoretical bounds, and iii) empirical illustrations of our theoretical findings.

----

## [1102] Reasons for the Superiority of Stochastic Estimators over Deterministic Ones: Robustness, Consistency and Perceptual Quality

**Authors**: *Guy Ohayon, Theo Joseph Adrai, Michael Elad, Tomer Michaeli*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/ohayon23a.html](https://proceedings.mlr.press/v202/ohayon23a.html)

**Abstract**:

Stochastic restoration algorithms allow to explore the space of solutions that correspond to the degraded input. In this paper we reveal additional fundamental advantages of stochastic methods over deterministic ones, which further motivate their use. First, we prove that any restoration algorithm that attains perfect perceptual quality and whose outputs are consistent with the input must be a posterior sampler, and is thus required to be stochastic. Second, we illustrate that while deterministic restoration algorithms may attain high perceptual quality, this can be achieved only by filling up the space of all possible source images using an extremely sensitive mapping, which makes them highly vulnerable to adversarial attacks. Indeed, we show that enforcing deterministic models to be robust to such attacks profoundly hinders their perceptual quality, while robustifying stochastic models hardly influences their perceptual quality, and improves their output variability. These findings provide a motivation to foster progress in stochastic restoration methods, paving the way to better recovery algorithms.

----

## [1103] On the Within-Group Fairness of Screening Classifiers

**Authors**: *Nastaran Okati, Stratis Tsirtsis, Manuel Gomez Rodriguez*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/okati23a.html](https://proceedings.mlr.press/v202/okati23a.html)

**Abstract**:

Screening classifiers are increasingly used to identify qualified candidates in a variety of selection processes. In this context, it has been recently shown that if a classifier is calibrated, one can identify the smallest set of candidates which contains, in expectation, a desired number of qualified candidates using a threshold decision rule. This lends support to focusing on calibration as the only requirement for screening classifiers. In this paper, we argue that screening policies that use calibrated classifiers may suffer from an understudied type of within-group unfairness—they may unfairly treat qualified members within demographic groups of interest. Further, we argue that this type of unfairness can be avoided if classifiers satisfy within-group monotonicity, a natural monotonicity property within each group. Then, we introduce an efficient post-processing algorithm based on dynamic programming to minimally modify a given calibrated classifier so that its probability estimates satisfy within-group monotonicity. We validate our algorithm using US Census survey data and show that within-group monotonicity can often be achieved at a small cost in terms of prediction granularity and shortlist size.

----

## [1104] Diffusion Models are Minimax Optimal Distribution Estimators

**Authors**: *Kazusato Oko, Shunta Akiyama, Taiji Suzuki*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/oko23a.html](https://proceedings.mlr.press/v202/oko23a.html)

**Abstract**:

While efficient distribution learning is no doubt behind the groundbreaking success of diffusion modeling, its theoretical guarantees are quite limited. In this paper, we provide the first rigorous analysis on approximation and generalization abilities of diffusion modeling for well-known function spaces. The highlight of this paper is that when the true density function belongs to the Besov space and the empirical score matching loss is properly minimized, the generated data distribution achieves the nearly minimax optimal estimation rates in the total variation distance and in the Wasserstein distance of order one. Furthermore, we extend our theory to demonstrate how diffusion models adapt to low-dimensional data distributions. We expect these results advance theoretical understandings of diffusion modeling and its ability to generate verisimilar outputs.

----

## [1105] How Many Perturbations Break This Model? Evaluating Robustness Beyond Adversarial Accuracy

**Authors**: *Raphaël Olivier, Bhiksha Raj*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/olivier23a.html](https://proceedings.mlr.press/v202/olivier23a.html)

**Abstract**:

Robustness to adversarial attacks is typically evaluated with adversarial accuracy. While essential, this metric does not capture all aspects of robustness and in particular leaves out the question of how many perturbations can be found for each point. In this work, we introduce an alternative approach, adversarial sparsity, which quantifies how difficult it is to find a successful perturbation given both an input point and a constraint on the direction of the perturbation. We show that sparsity provides valuable insight into neural networks in multiple ways: for instance, it illustrates important differences between current state-of-the-art robust models them that accuracy analysis does not, and suggests approaches for improving their robustness. When applying broken defenses effective against weak attacks but not strong ones, sparsity can discriminate between the totally ineffective and the partially effective defenses. Finally, with sparsity we can measure increases in robustness that do not affect accuracy: we show for example that data augmentation can by itself increase adversarial robustness, without using adversarial training.

----

## [1106] B-Learner: Quasi-Oracle Bounds on Heterogeneous Causal Effects Under Hidden Confounding

**Authors**: *Miruna Oprescu, Jacob Dorn, Marah Ghoummaid, Andrew Jesson, Nathan Kallus, Uri Shalit*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/oprescu23a.html](https://proceedings.mlr.press/v202/oprescu23a.html)

**Abstract**:

Estimating heterogeneous treatment effects from observational data is a crucial task across many fields, helping policy and decision-makers take better actions. There has been recent progress on robust and efficient methods for estimating the conditional average treatment effect (CATE) function, but these methods often do not take into account the risk of hidden confounding, which could arbitrarily and unknowingly bias any causal estimate based on observational data. We propose a meta-learner called the B-Learner, which can efficiently learn sharp bounds on the CATE function under limits on the level of hidden confounding. We derive the B-Learner by adapting recent results for sharp and valid bounds of the average treatment effect (Dorn et al., 2021) into the framework given by Kallus & Oprescu (2023) for robust and model-agnostic learning of conditional distributional treatment effects. The B-Learner can use any function estimator such as random forests and deep neural networks, and we prove its estimates are valid, sharp, efficient, and have a quasi-oracle property with respect to the constituent estimators under more general conditions than existing methods. Semi-synthetic experimental comparisons validate the theoretical findings, and we use real-world data demonstrate how the method might be used in practice.

----

## [1107] Measuring the Impact of Programming Language Distribution

**Authors**: *Gabriel Orlanski, Kefan Xiao, Xavier Garcia, Jeffrey Hui, Joshua Howland, Jonathan Malmaud, Jacob Austin, Rishabh Singh, Michele Catasta*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/orlanski23a.html](https://proceedings.mlr.press/v202/orlanski23a.html)

**Abstract**:

Current benchmarks for evaluating neural code models focus on only a small subset of programming languages, excluding many popular languages such as Go or Rust. To ameliorate this issue, we present the BabelCode framework for execution-based evaluation of any benchmark in any language. BabelCode enables new investigations into the qualitative performance of models’ memory, runtime, and individual test case results. Additionally, we present a new code translation dataset called Translating Python Programming Puzzles (TP3) from the Python Programming Puzzles (Schuster et al., 2021) benchmark that involves translating expert-level python functions to any language. With both BabelCode and the TP3 benchmark, we investigate if balancing the distributions of 14 languages in a training dataset improves a large language model’s performance on low-resource languages. Training a model on a balanced corpus results in, on average, 12.34% higher $pass@k$ across all tasks and languages compared to the baseline. We find that this strategy achieves 66.48% better $pass@k$ on low-resource languages at the cost of only a 12.94% decrease to high-resource languages. In our three translation tasks, this strategy yields, on average, 30.77% better low-resource $pass@k$ while having 19.58% worse high-resource $pass@k$.

----

## [1108] When does Privileged information Explain Away Label Noise?

**Authors**: *Guillermo Ortiz-Jiménez, Mark Collier, Anant Nawalgaria, Alexander Nicholas D'Amour, Jesse Berent, Rodolphe Jenatton, Efi Kokiopoulou*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/ortiz-jimenez23a.html](https://proceedings.mlr.press/v202/ortiz-jimenez23a.html)

**Abstract**:

Leveraging privileged information (PI), or features available during training but not at test time, has recently been shown to be an effective method for addressing label noise. However, the reasons for its effectiveness are not well understood. In this study, we investigate the role played by different properties of the PI in explaining away label noise. Through experiments on multiple datasets with real PI (CIFAR-N/H) and a new large-scale benchmark ImageNet-PI, we find that PI is most helpful when it allows networks to easily distinguish clean from noisy data, while enabling a learning shortcut to memorize the noisy examples. Interestingly, when PI becomes too predictive of the target label, PI methods often perform worse than their no-PI baselines. Based on these findings, we propose several enhancements to the state-of-the-art PI methods and demonstrate the potential of PI as a means of tackling label noise. Finally, we show how we can easily combine the resulting PI approaches with existing no-PI techniques designed to deal with label noise.

----

## [1109] Resurrecting Recurrent Neural Networks for Long Sequences

**Authors**: *Antonio Orvieto, Samuel L. Smith, Albert Gu, Anushan Fernando, Çaglar Gülçehre, Razvan Pascanu, Soham De*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/orvieto23a.html](https://proceedings.mlr.press/v202/orvieto23a.html)

**Abstract**:

Recurrent Neural Networks (RNNs) offer fast inference on long sequences but are hard to optimize and slow to train. Deep state-space models (SSMs) have recently been shown to perform remarkably well on long sequence modeling tasks, and have the added benefits of fast parallelizable training and RNN-like fast inference. However, while SSMs are superficially similar to RNNs, there are important differences that make it unclear where their performance boost over RNNs comes from. We show that careful design of deep RNNs using standard signal propagation arguments can recover the impressive performance of deep SSMs on long-range reasoning tasks, while matching their training speed. To achieve this, we analyze and ablate a series of changes to standard RNNs including linearizing and diagonalizing the recurrence, using better parameterizations and initializations, and ensuring careful normalization of the forward pass. Our results provide new insights on the origins of the impressive performance of deep SSMs, and introduce an RNN block called the Linear Recurrent Unit (or LRU) that matches both their performance on the Long Range Arena benchmark and their computational efficiency.

----

## [1110] Improving Adversarial Robustness Through the Contrastive-Guided Diffusion Process

**Authors**: *Yidong Ouyang, Liyan Xie, Guang Cheng*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/ouyang23a.html](https://proceedings.mlr.press/v202/ouyang23a.html)

**Abstract**:

Synthetic data generation has become an emerging tool to help improve the adversarial robustness in classification tasks, since robust learning requires a significantly larger amount of training samples compared with standard classification. Among various deep generative models, the diffusion model has been shown to produce high-quality synthetic images and has achieved good performance in improving the adversarial robustness. However, diffusion-type methods are generally slower in data generation as compared with other generative models. Although different acceleration techniques have been proposed recently, it is also of great importance to study how to improve the sample efficiency of synthetic data for the downstream task. In this paper, we first analyze the optimality condition of synthetic distribution for achieving improved robust accuracy. We show that enhancing the distinguishability among the generated data is critical for improving adversarial robustness. Thus, we propose the Contrastive-Guided Diffusion Process (Contrastive-DP), which incorporates the contrastive loss to guide the diffusion model in data generation. We validate our theoretical results using simulations and demonstrate the good performance of Contrastive-DP on image datasets.

----

## [1111] On the Role of Attention in Prompt-tuning

**Authors**: *Samet Oymak, Ankit Singh Rawat, Mahdi Soltanolkotabi, Christos Thrampoulidis*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/oymak23a.html](https://proceedings.mlr.press/v202/oymak23a.html)

**Abstract**:

Prompt-tuning is an emerging strategy to adapt large language models (LLM) to downstream tasks by learning a (soft-)prompt parameter from data. Despite its success in LLMs, there is limited theoretical understanding of the power of prompt-tuning and the role of the attention mechanism in prompting. In this work, we explore prompt-tuning for one-layer attention architectures and study contextual mixture-models where each input token belongs to a context-relevant or -irrelevant set. We isolate the role of prompt-tuning through a self-contained prompt-attention model. Our contributions are as follows: (1) We show that softmax-prompt-attention is provably more expressive than softmax-self-attention and linear-prompt-attention under our contextual data model. (2) We analyze the initial trajectory of gradient descent and show that it learns the prompt and prediction head with near-optimal sample complexity and demonstrate how the prompt can provably attend to sparse context-relevant tokens. (3) Assuming a known prompt but an unknown prediction head, we characterize the exact finite sample performance of prompt-attention which reveals the fundamental performance limits and the precise benefit of the context information. We also provide experiments that verify our theoretical insights on real datasets and demonstrate how prompt-tuning enables the model to attend to context-relevant information.

----

## [1112] Revisiting the Linear-Programming Framework for Offline RL with General Function Approximation

**Authors**: *Asuman E. Ozdaglar, Sarath Pattathil, Jiawei Zhang, Kaiqing Zhang*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/ozdaglar23a.html](https://proceedings.mlr.press/v202/ozdaglar23a.html)

**Abstract**:

Offline reinforcement learning (RL) aims to find an optimal policy for sequential decision-making using a pre-collected dataset, without further interaction with the environment. Recent theoretical progress has focused on developing sample-efficient offline RL algorithms with various relaxed assumptions on data coverage and function approximators, especially to handle the case with excessively large state-action spaces. Among them, the framework based on the linear-programming (LP) reformulation of Markov decision processes has shown promise: it enables sample-efficient offline RL with function approximation, under only partial data coverage and realizability assumptions on the function classes, with favorable computational tractability. In this work, we revisit the LP framework for offline RL, and provide a new reformulation that advances the existing results in several aspects, relaxing certain assumptions and achieving optimal statistical rates in terms of sample size. Our key enabler is to introduce proper constraints in the reformulation, instead of using any regularization as in the literature, also with careful choices of the function classes and initial state distributions. We hope our insights bring into light the use of LP formulations and the induced primal-dual minimax optimization, in offline RL.

----

## [1113] Extrapolative Controlled Sequence Generation via Iterative Refinement

**Authors**: *Vishakh Padmakumar, Richard Yuanzhe Pang, He He, Ankur P. Parikh*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/padmakumar23a.html](https://proceedings.mlr.press/v202/padmakumar23a.html)

**Abstract**:

We study the problem of extrapolative controlled generation, i.e., generating sequences with attribute values beyond the range seen in training. This task is of significant importance in automated design, especially drug discovery, where the goal is to design novel proteins that are better (e.g., more stable) than existing sequences. Thus, by definition the target sequences and their attribute values are out of the training distribution, posing challenges to existing methods that aim to directly generate the target sequence. Instead, in this work, we propose Iterative Controlled Extrapolation (ICE) which iteratively makes local edits to a sequence to enable extrapolation. We train the model on synthetically generated sequence pairs that demonstrate small improvement in the attribute value. Results on one natural language task (sentiment analysis) and two protein engineering tasks (ACE2 stability and AAV fitness) show that ICE outperforms state-of-the-art approaches despite its simplicity.

----

## [1114] Locally Regularized Neural Differential Equations: Some Black Boxes were meant to remain closed!

**Authors**: *Avik Pal, Alan Edelman, Christopher Vincent Rackauckas*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/pal23a.html](https://proceedings.mlr.press/v202/pal23a.html)

**Abstract**:

Neural Differential Equations have become an important modeling framework due to their ability to adapt to new problems automatically. Training a neural differential equation is effectively a search over a space of plausible dynamical systems. Controlling the computational cost for these models is difficult since it relies on the number of steps the adaptive solver takes. Most prior works have used higher-order methods to reduce prediction timings while greatly increasing training time or reducing both training and prediction timings by relying on specific training algorithms, which are harder to use as a drop-in replacement. In this manuscript, we use internal cost heuristics of adaptive differential equation solvers at stochastic time-points to guide the training towards learning a dynamical system that is easier to integrate. We “close the blackbox” and allow the use of our method with any sensitivity method. We perform experimental studies to compare our method to global regularization to show that we attain similar performance numbers without compromising on the flexibility of implementation. We develop two sampling strategies to trade-off between performance and training time. Our method reduces the number of function evaluations to 0.556x - 0.733x and accelerates predictions by 1.3x - 2x.

----

## [1115] Controlled Differential Equations on Long Sequences via Non-standard Wavelets

**Authors**: *Sourav Pal, Zhanpeng Zeng, Sathya N. Ravi, Vikas Singh*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/pal23b.html](https://proceedings.mlr.press/v202/pal23b.html)

**Abstract**:

Neural Controlled Differential equations (NCDE) are a powerful mechanism to model the dynamics in temporal sequences, e.g., applications involving physiological measures, where apart from the initial condition, the dynamics also depend on subsequent measures or even a different "control" sequence. But NCDEs do not scale well to longer sequences. Existing strategies adapt rough path theory, and instead model the dynamics over summaries known as log signatures. While rigorous and elegant, invertibility of these summaries is difficult, and limits the scope of problems where these ideas can offer strong benefits (reconstruction, generative modeling). For tasks where it is sensible to assume that the (long) sequences in the training data are a fixed length of temporal measurements – this assumption holds in most experiments tackled in the literature – we describe an efficient simplification. First, we recast the regression/classification task as an integral transform. We then show how restricting the class of operators (permissible in the integral transform), allows the use of a known algorithm that leverages non-standard Wavelets to decompose the operator. Thereby, our task (learning the operator) radically simplifies. A neural variant of this idea yields consistent improvements across a wide gamut of use cases tackled in existing works. We also describe a novel application on modeling tasks involving coupled differential equations.

----

## [1116] Do the Rewards Justify the Means? Measuring Trade-Offs Between Rewards and Ethical Behavior in the Machiavelli Benchmark

**Authors**: *Alexander Pan, Jun Shern Chan, Andy Zou, Nathaniel Li, Steven Basart, Thomas Woodside, Hanlin Zhang, Scott Emmons, Dan Hendrycks*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/pan23a.html](https://proceedings.mlr.press/v202/pan23a.html)

**Abstract**:

Artificial agents have traditionally been trained to maximize reward, which may incentivize power-seeking and deception, analogous to how next-token prediction in language models (LMs) may incentivize toxicity. So do agents naturally learn to be Machiavellian? And how do we measure these behaviors in general-purpose models such as GPT-4? Towards answering these questions, we introduce Machiavelli, a benchmark of 134 Choose-Your-Own-Adventure games containing over half a million rich, diverse scenarios that center on social decision-making. Scenario labeling is automated with LMs, which are more performant than human annotators. We mathematize dozens of harmful behaviors and use our annotations to evaluate agents’ tendencies to be power-seeking, cause disutility, and commit ethical violations. We observe some tension between maximizing reward and behaving ethically. To improve this trade-off, we investigate LM-based methods to steer agents towards less harmful behaviors. Our results show that agents can both act competently and morally, so concrete progress can currently be made in machine ethics–designing agents that are Pareto improvements in both safety and capabilities.

----

## [1117] Beyond Homophily: Reconstructing Structure for Graph-agnostic Clustering

**Authors**: *Erlin Pan, Zhao Kang*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/pan23b.html](https://proceedings.mlr.press/v202/pan23b.html)

**Abstract**:

Graph neural networks (GNNs) based methods have achieved impressive performance on node clustering task. However, they are designed on the homophilic assumption of graph and clustering on heterophilic graph is overlooked. Due to the lack of labels, it is impossible to first identify a graph as homophilic or heterophilic before a suitable GNN model can be found. Hence, clustering on real-world graph with various levels of homophily poses a new challenge to the graph research community. To fill this gap, we propose a novel graph clustering method, which contains three key components: graph reconstruction, a mixed filter, and dual graph clustering network. To be graph-agnostic, we empirically construct two graphs which are high homophily and heterophily from each data. The mixed filter based on the new graphs extracts both low-frequency and high-frequency information. To reduce the adverse coupling between node attribute and topological structure, we separately map them into two subspaces in dual graph clustering network. Extensive experiments on 11 benchmark graphs demonstrate our promising performance. In particular, our method dominates others on heterophilic graphs.

----

## [1118] Better Training of GFlowNets with Local Credit and Incomplete Trajectories

**Authors**: *Ling Pan, Nikolay Malkin, Dinghuai Zhang, Yoshua Bengio*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/pan23c.html](https://proceedings.mlr.press/v202/pan23c.html)

**Abstract**:

Generative Flow Networks or GFlowNets are related to Monte-Carlo Markov chain methods (as they sample from a distribution specified by an energy function), reinforcement learning (as they learn a policy to sample composed objects through a sequence of steps), generative models (as they learn to represent and sample from a distribution) and amortized variational methods (as they can be used to learn to approximate and sample from an otherwise intractable posterior, given a prior and a likelihood). They are trained to generate an object $x$ through a sequence of steps with probability proportional to some reward function $R(x)$ (or $\exp(-\mathcal{E}(x))$ with $\mathcal{E}(x)$ denoting the energy function), given at the end of the generative trajectory. Like for other RL settings where the reward is only given at the end, the efficiency of training and credit assignment may suffer when those trajectories are longer. With previous GFlowNet work, no learning was possible from incomplete trajectories (lacking a terminal state and the computation of the associated reward). In this paper, we consider the case where the energy function can be applied not just to terminal states but also to intermediate states. This is for example achieved when the energy function is additive, with terms available along the trajectory. We show how to reparameterize the GFlowNet state flow function to take advantage of the partial reward already accrued at each state. This enables a training objective that can be applied to update parameters even with incomplete trajectories. Even when complete trajectories are available, being able to obtain more localized credit and gradients is found to speed up training convergence, as demonstrated across many simulations.

----

## [1119] A Hybrid Quantum-Classical Approach based on the Hadamard Transform for the Convolutional Layer

**Authors**: *Hongyi Pan, Xin Zhu, Salih Furkan Atici, Ahmet Enis Çetin*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/pan23d.html](https://proceedings.mlr.press/v202/pan23d.html)

**Abstract**:

In this paper, we propose a novel Hadamard Transform (HT)-based neural network layer for hybrid quantum-classical computing. It implements the regular convolutional layers in the Hadamard transform domain. The idea is based on the HT convolution theorem which states that the dyadic convolution between two vectors is equivalent to the element-wise multiplication of their HT representation. Computing the HT is simply the application of a Hadamard gate to each qubit individually, so the HT computations of our proposed layer can be implemented on a quantum computer. Compared to the regular Conv2D layer, the proposed HT-perceptron layer is computationally more efficient. Compared to a CNN with the same number of trainable parameters and 99.26% test accuracy, our HT network reaches 99.31% test accuracy with 57.1% MACs reduced in the MNIST dataset; and in our ImageNet-1K experiments, our HT-based ResNet-50 exceeds the accuracy of the baseline ResNet-50 by 0.59% center-crop top-1 accuracy using 11.5% fewer parameters with 12.6% fewer MACs.

----

## [1120] Semi Bandit dynamics in Congestion Games: Convergence to Nash Equilibrium and No-Regret Guarantees

**Authors**: *Ioannis Panageas, Stratis Skoulakis, Luca Viano, Xiao Wang, Volkan Cevher*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/panageas23a.html](https://proceedings.mlr.press/v202/panageas23a.html)

**Abstract**:

In this work, we propose introduce a variant of online stochastic gradient descent and prove it converges to Nash equilibria and simultaneously it has sublinear regret for the class of congestion games in the semi-bandit feedback setting. Our proposed method admits convergence rates depending only polynomially on the number of players and the number of facilities, but not on the size of the action set, which can be exponentially large in terms of the number of facilities. Moreover, the running time of our method has polynomial-time dependence on the implicit description of the game. Our analysis exploits techniques from convex geometry, in particular Caratheodory’s theorem and recent advances in non-convex stochastic optimization. This work improves upon and answers an open question from (Cui et al 2022).

----

## [1121] Flash: Concept Drift Adaptation in Federated Learning

**Authors**: *Kunjal Panchal, Sunav Choudhary, Subrata Mitra, Koyel Mukherjee, Somdeb Sarkhel, Saayan Mitra, Hui Guan*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/panchal23a.html](https://proceedings.mlr.press/v202/panchal23a.html)

**Abstract**:

In Federated Learning (FL), adaptive optimization is an effective approach to addressing the statistical heterogeneity issue but cannot adapt quickly to concept drifts. In this work, we propose a novel adaptive optimizer called Flash that simultaneously addresses both statistical heterogeneity and the concept drift issues. The fundamental insight is that a concept drift can be detected based on the magnitude of parameter updates that are required to fit the global model to each participating client’s local data distribution. Flash uses a two-pronged approach that synergizes client-side early-stopping training to facilitate detection of concept drifts and the server-side drift-aware adaptive optimization to effectively adjust effective learning rate. We theoretically prove that Flash matches the convergence rate of state-of-the-art adaptive optimizers and further empirically evaluate the efficacy of Flash on a variety of FL benchmarks using different concept drift settings.

----

## [1122] Learn to Accumulate Evidence from All Training Samples: Theory and Practice

**Authors**: *Deep Shankar Pandey, Qi Yu*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/pandey23a.html](https://proceedings.mlr.press/v202/pandey23a.html)

**Abstract**:

Evidential deep learning, built upon belief theory and subjective logic, offers a principled and computationally efficient way to turn a deterministic neural network uncertainty-aware. The resultant evidential models can quantify fine-grained uncertainty using the learned evidence. To ensure theoretically sound evidential models, the evidence needs to be non-negative, which requires special activation functions for model training and inference. This constraint often leads to inferior predictive performance compared to standard softmax models, making it challenging to extend them to many large-scale datasets. To unveil the real cause of this undesired behavior, we theoretically investigate evidential models and identify a fundamental limitation that explains the inferior performance: existing evidential activation functions create zero evidence regions, which prevent the model to learn from training samples falling into such regions. A deeper analysis of evidential activation functions based on our theoretical underpinning inspires the design of a novel regularizer that effectively alleviates this fundamental limitation. Extensive experiments over many challenging real-world datasets and settings confirm our theoretical findings and demonstrate the effectiveness of our proposed approach.

----

## [1123] Secure Federated Correlation Test and Entropy Estimation

**Authors**: *Qi Pang, Lun Wang, Shuai Wang, Wenting Zheng, Dawn Song*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/pang23a.html](https://proceedings.mlr.press/v202/pang23a.html)

**Abstract**:

We propose the first federated correlation test framework compatible with secure aggregation, namely FED-$\chi^2$. In our protocol, the statistical computations are recast as frequency moment estimation problems, where the clients collaboratively generate a shared projection matrix and then use stable projection to encode the local information in a compact vector. As such encodings can be linearly aggregated, secure aggregation can be applied to conceal the individual updates. We formally establish the security guarantee of FED-$\chi^2$ by proving that only the minimum necessary information (i.e., the correlation statistics) is revealed to the server. We show that our protocol can be naturally extended to estimate other statistics that can be recast as frequency moment estimations. By accommodating Shannon’e Entropy in FED-$\chi^2$, we further propose the first secure federated entropy estimation protocol, FED-$H$. The evaluation results demonstrate that FED-$\chi^2$ and FED-$H$ achieve good performance with small client-side computation overhead in several real-world case studies.

----

## [1124] Task-Specific Skill Localization in Fine-tuned Language Models

**Authors**: *Abhishek Panigrahi, Nikunj Saunshi, Haoyu Zhao, Sanjeev Arora*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/panigrahi23a.html](https://proceedings.mlr.press/v202/panigrahi23a.html)

**Abstract**:

Pre-trained language models can be fine-tuned to solve diverse NLP tasks, including in few-shot settings. Thus fine-tuning allows the model to quickly pick up task-specific "skills," but there has been limited study of where these newly-learnt skills reside inside the massive model. This paper introduces the term skill localization for this problem and proposes a solution. Given the downstream task and a model fine-tuned on that task, a simple optimization is used to identify a very small subset of parameters ($\sim$0.01% of model parameters) responsible for ($>$95%) of the model’s performance, in the sense that grafting the fine-tuned values for just this tiny subset onto the pre-trained model gives performance almost as well as the fine-tuned model. While reminiscent of recent works on parameter-efficient fine-tuning, the novel aspects here are that: (i) No further retraining is needed on the subset (unlike, say, with lottery tickets). (ii) Notable improvements are seen over vanilla fine-tuning with respect to calibration of predictions in-distribution (40-90% error reduction) as well as quality of predictions out-of-distribution (OOD). In models trained on multiple tasks, a stronger notion of skill localization is observed, where the sparse regions corresponding to different tasks are almost disjoint, and their overlap (when it happens) is a proxy for task similarity. Experiments suggest that localization via grafting can assist certain forms continual learning.

----

## [1125] Kernel Sufficient Dimension Reduction and Variable Selection for Compositional Data via Amalgamation

**Authors**: *Junyoung Park, Jeongyoun Ahn, Cheolwoo Park*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/park23a.html](https://proceedings.mlr.press/v202/park23a.html)

**Abstract**:

Compositional data with a large number of components and an abundance of zeros are frequently observed in many fields recently. Analyzing such sparse high-dimensional compositional data naturally calls for dimension reduction or, more preferably, variable selection. Most existing approaches lack interpretability or cannot handle zeros properly, as they rely on a log-ratio transformation. We approach this problem with sufficient dimension reduction (SDR), one of the most studied dimension reduction frameworks in statistics. Characterized by the conditional independence of the data to the response on the found subspace, the SDR framework has been effective for both linear and nonlinear dimension reduction problems. This work proposes a compositional SDR that can handle zeros naturally while incorporating the nonlinear nature and spurious negative correlations among components rigorously. A critical consideration of sub-composition versus amalgamation for compositional variable selection is discussed. The proposed compositional SDR is shown to be statistically consistent in constructing a sub-simplex consisting of true signal variables. Simulation and real microbiome data are used to demonstrate the performance of the proposed SDR compared to existing state-of-art approaches.

----

## [1126] Learning Affinity with Hyperbolic Representation for Spatial Propagation

**Authors**: *Jin-Hwi Park, Jaesung Choe, Inhwan Bae, Hae-Gon Jeon*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/park23b.html](https://proceedings.mlr.press/v202/park23b.html)

**Abstract**:

Recent approaches to representation learning have successfully demonstrated the benefits in hyperbolic space, driven by an excellent ability to make hierarchical relationships. In this work, we demonstrate that the properties of hyperbolic geometry serve as a valuable alternative to learning hierarchical affinity for spatial propagation tasks. We propose a Hyperbolic Affinity learning Module (HAM) to learn spatial affinity by considering geodesic distance on the hyperbolic space. By simply incorporating our HAM into conventional spatial propagation tasks, we validate its effectiveness, capturing the pixel hierarchy of affinity maps in hyperbolic space. The proposed methodology can lead to performance improvements in explicit propagation processes such as depth completion and semantic segmentation.

----

## [1127] TRAK: Attributing Model Behavior at Scale

**Authors**: *Sung Min Park, Kristian Georgiev, Andrew Ilyas, Guillaume Leclerc, Aleksander Madry*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/park23c.html](https://proceedings.mlr.press/v202/park23c.html)

**Abstract**:

The goal of data attribution is to trace model predictions back to training data. Despite a long line of work towards this goal, existing approaches to data attribution tend to force users to choose between computational tractability and efficacy. That is, computationally tractable methods can struggle with accurately attributing model predictions in non-convex settings (e.g., in the context of deep neural networks), while methods that are effective in such regimes require training thousands of models, which makes them impractical for large models or datasets. In this work, we introduce TRAK (Tracing with the Randomly-projected After Kernel), a data attribution method that is both effective and computationally tractable for large-scale, differentiable models. In particular, by leveraging only a handful of trained models, TRAK can match the performance of attribution methods that require training thousands of models. We demonstrate the utility of TRAK across various modalities and scales: image classifiers trained on ImageNet, vision-language models (CLIP), and language models (BERT and mT5). We provide code for using TRAK (and reproducing our work) at https://github.com/MadryLab/trak .

----

## [1128] Test-Time Style Shifting: Handling Arbitrary Styles in Domain Generalization

**Authors**: *Jungwuk Park, Dong-Jun Han, Soyeong Kim, Jaekyun Moon*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/park23d.html](https://proceedings.mlr.press/v202/park23d.html)

**Abstract**:

In domain generalization (DG), the target domain is unknown when the model is being trained, and the trained model should successfully work on an arbitrary (and possibly unseen) target domain during inference. This is a difficult problem, and despite active studies in recent years, it remains a great challenge. In this paper, we take a simple yet effective approach to tackle this issue. We propose test-time style shifting, which shifts the style of the test sample (that has a large style gap with the source domains) to the nearest source domain that the model is already familiar with, before making the prediction. This strategy enables the model to handle any target domains with arbitrary style statistics, without additional model update at test-time. Additionally, we propose style balancing, which provides a great platform for maximizing the advantage of test-time style shifting by handling the DG-specific imbalance issues. The proposed ideas are easy to implement and successfully work in conjunction with various other DG schemes. Experimental results on different datasets show the effectiveness of our methods.

----

## [1129] Towards Understanding Ensemble Distillation in Federated Learning

**Authors**: *Sejun Park, Kihun Hong, Ganguk Hwang*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/park23e.html](https://proceedings.mlr.press/v202/park23e.html)

**Abstract**:

Federated Learning (FL) is a collaborative machine learning paradigm for data privacy preservation. Recently, a knowledge distillation (KD) based information sharing approach in FL, which conducts ensemble distillation on an unlabeled public dataset, has been proposed. However, despite its experimental success and usefulness, the theoretical analysis of the KD based approach has not been satisfactorily conducted. In this work, we build a theoretical foundation of the ensemble distillation framework in federated learning from the perspective of kernel ridge regression (KRR). In this end, we propose a KD based FL algorithm for KRR models which is related with some existing KD based FL algorithms, and analyze our algorithm theoretically. We show that our algorithm makes local prediction models as much powerful as the centralized KRR model (which is a KRR model trained by all of local datasets) in terms of the convergence rate of the generalization error if the unlabeled public dataset is sufficiently large. We also provide experimental results to verify our theoretical results on ensemble distillation in federated learning.

----

## [1130] Learning Controllable Degradation for Real-World Super-Resolution via Constrained Flows

**Authors**: *Seobin Park, Dongjin Kim, Sungyong Baik, Tae Hyun Kim*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/park23f.html](https://proceedings.mlr.press/v202/park23f.html)

**Abstract**:

Recent deep-learning-based super-resolution (SR) methods have been successful in recovering high-resolution (HR) images from their low-resolution (LR) counterparts, albeit on the synthetic and simple degradation setting: bicubic downscaling. On the other hand, super-resolution on real-world images demands the capability to handle complex downscaling mechanism which produces different artifacts (e.g., noise, blur, color distortion) upon downscaling factors. To account for complex downscaling mechanism in real-world LR images, there have been a few efforts in constructing datasets consisting of LR images with real-world downsampling degradation. However, making such datasets entails a tremendous amount of time and effort, thereby resorting to very few number of downscaling factors (e.g., $\times$2, $\times$3, $\times$4). To remedy the issue, we propose to generate realistic SR datasets for unseen degradation levels by exploring the latent space of real LR images and thereby producing more diverse yet realistic LR images with complex real-world artifacts. Our quantitative and qualitative experiments demonstrate the accuracy of the generated LR images, and we show that the various conventional SR networks trained with our newly generated SR datasets can produce much better HR images.

----

## [1131] Differentially Private Sharpness-Aware Training

**Authors**: *Jinseong Park, Hoki Kim, Yujin Choi, Jaewook Lee*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/park23g.html](https://proceedings.mlr.press/v202/park23g.html)

**Abstract**:

Training deep learning models with differential privacy (DP) results in a degradation of performance. The training dynamics of models with DP show a significant difference from standard training, whereas understanding the geometric properties of private learning remains largely unexplored. In this paper, we investigate sharpness, a key factor in achieving better generalization, in private learning. We show that flat minima can help reduce the negative effects of per-example gradient clipping and the addition of Gaussian noise. We then verify the effectiveness of Sharpness-Aware Minimization (SAM) for seeking flat minima in private learning. However, we also discover that SAM is detrimental to the privacy budget and computational time due to its two-step optimization. Thus, we propose a new sharpness-aware training method that mitigates the privacy-optimization trade-off. Our experimental results demonstrate that the proposed method improves the performance of deep learning models with DP from both scratch and fine-tuning. Code is available at https://github.com/jinseongP/DPSAT.

----

## [1132] Controllability-Aware Unsupervised Skill Discovery

**Authors**: *Seohong Park, Kimin Lee, Youngwoon Lee, Pieter Abbeel*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/park23h.html](https://proceedings.mlr.press/v202/park23h.html)

**Abstract**:

One of the key capabilities of intelligent agents is the ability to discover useful skills without external supervision. However, the current unsupervised skill discovery methods are often limited to acquiring simple, easy-to-learn skills due to the lack of incentives to discover more complex, challenging behaviors. We introduce a novel unsupervised skill discovery method, Controllability-aware Skill Discovery (CSD), which actively seeks complex, hard-to-control skills without supervision. The key component of CSD is a controllability-aware distance function, which assigns larger values to state transitions that are harder to achieve with the current skills. Combined with distance-maximizing skill discovery, CSD progressively learns more challenging skills over the course of training as our jointly trained distance function reduces rewards for easy-to-achieve skills. Our experimental results in six robotic manipulation and locomotion environments demonstrate that CSD can discover diverse complex skills including object manipulation and locomotion skills with no supervision, significantly outperforming prior unsupervised skill discovery methods. Videos and code are available at https://seohong.me/projects/csd/

----

## [1133] Predictable MDP Abstraction for Unsupervised Model-Based RL

**Authors**: *Seohong Park, Sergey Levine*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/park23i.html](https://proceedings.mlr.press/v202/park23i.html)

**Abstract**:

A key component of model-based reinforcement learning (RL) is a dynamics model that predicts the outcomes of actions. Errors in this predictive model can degrade the performance of model-based controllers, and complex Markov decision processes (MDPs) can present exceptionally difficult prediction problems. To mitigate this issue, we propose predictable MDP abstraction (PMA): instead of training a predictive model on the original MDP, we train a model on a transformed MDP with a learned action space that only permits predictable, easy-to-model actions, while covering the original state-action space as much as possible. As a result, model learning becomes easier and more accurate, which allows robust, stable model-based planning or model-based RL. This transformation is learned in an unsupervised manner, before any task is specified by the user. Downstream tasks can then be solved with model-based control in a zero-shot fashion, without additional environment interactions. We theoretically analyze PMA and empirically demonstrate that PMA leads to significant improvements over prior unsupervised model-based RL approaches in a range of benchmark environments. Our code and videos are available at https://seohong.me/projects/pma/

----

## [1134] Neural Stochastic Differential Games for Time-series Analysis

**Authors**: *Sungwoo Park, Byoungwoo Park, Moontae Lee, Changhee Lee*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/park23j.html](https://proceedings.mlr.press/v202/park23j.html)

**Abstract**:

Modeling spatiotemporal dynamics with neural differential equations has become a major line of research that opens new ways to handle various real-world scenarios (e.g., missing observations, irregular times, etc.). Despite such progress, most existing methods still face challenges in providing a general framework for analyzing time series. To tackle this, we adopt stochastic differential games to suggest a new philosophy of utilizing interacting collective intelligence in time series analysis. For the implementation, we develop the novel gradient descent-based algorithm called deep neural fictitious play to approximate the Nash equilibrium. We theoretically analyze the convergence result of the proposed algorithm and discuss the advantage of cooperative games in handling noninformative observation. Throughout the experiments on various datasets, we demonstrate the superiority of our framework over all the tested benchmarks in modeling time-series prediction by capitalizing on the advantages of applying cooperative games. An ablation study shows that neural agents of the proposed framework learn intrinsic temporal relevance to make accurate time-series predictions.

----

## [1135] Accelerated Infeasibility Detection of Constrained Optimization and Fixed-Point Iterations

**Authors**: *Jisun Park, Ernest K. Ryu*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/park23k.html](https://proceedings.mlr.press/v202/park23k.html)

**Abstract**:

As first-order optimization methods become the method of choice for solving large-scale optimization problems, optimization solvers based on first-order algorithms are being built. Such general-purpose solvers must robustly detect infeasible or misspecified problem instances, but the computational complexity of first-order methods for doing so has yet to be formally studied. In this work, we characterize the optimal accelerated rate of infeasibility detection. We show that the standard fixed-point iteration achieves a $\mathcal{O}(1/k^2)$ and $\mathcal{O}(1/k)$ rates, respectively, on the normalized iterates and the fixed-point residual converging to the infimal displacement vector, while the accelerated fixed-point iteration achieves $\mathcal{O}(1/k^2)$ and $\tilde{\mathcal{O}}(1/k^2)$ rates. We then provide a matching complexity lower bound to establish that $\Theta(1/k^2)$ is indeed the optimal accelerated rate.

----

## [1136] Model-based Reinforcement Learning with Scalable Composite Policy Gradient Estimators

**Authors**: *Paavo Parmas, Takuma Seno, Yuma Aoki*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/parmas23a.html](https://proceedings.mlr.press/v202/parmas23a.html)

**Abstract**:

In model-based reinforcement learning (MBRL), policy gradients can be estimated either by derivative-free RL methods, such as likelihood ratio gradients (LR), or by backpropagating through a differentiable model via reparameterization gradients (RP). Instead of using one or the other, the Total Propagation (TP) algorithm in prior work showed that a combination of LR and RP estimators averaged using inverse variance weighting (IVW) can achieve orders of magnitude improvement over either method. However, IVW-based composite estimators have not yet been applied in modern RL tasks, as it is unclear if they can be implemented scalably. We propose a scalable method, Total Propagation X (TPX) that improves over TP by changing the node used for IVW, and employing coordinate wise weighting. We demonstrate the scalability of TPX by applying it to the state of the art visual MBRL algorithm Dreamer. The experiments showed that Dreamer fails with long simulation horizons, while our TPX works reliably for only a fraction of additional computation. One key advantage of TPX is its ease of implementation, which will enable experimenting with IVW on many tasks beyond MBRL.

----

## [1137] PAC Generalization via Invariant Representations

**Authors**: *Advait U. Parulekar, Karthikeyan Shanmugam, Sanjay Shakkottai*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/parulekar23a.html](https://proceedings.mlr.press/v202/parulekar23a.html)

**Abstract**:

Invariant representations are transformations of the covariates such that the best model on top of the representation is invariant across training environments. In the context of linear Structural Equation Models (SEMs), invariant representations might allow us to learn models with out-of-distribution guarantees, i.e., models that are robust to interventions in the SEM. To address the invariant representation problem in a finite sample setting, we consider the notion of $\epsilon$-approximate invariance. We study the following question: If a representation is approximately invariant with respect to a given number of training interventions, will it continue to be approximately invariant on a larger collection of unseen intervened SEMs? Inspired by PAC learning, we obtain finite-sample out-of-distribution generalization guarantees for approximate invariance that holds probabilistically over a family of linear SEMs without faithfulness assumptions.

----

## [1138] Stochastic Gradient Descent-Induced Drift of Representation in a Two-Layer Neural Network

**Authors**: *Farhad Pashakhanloo, Alexei Koulakov*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/pashakhanloo23a.html](https://proceedings.mlr.press/v202/pashakhanloo23a.html)

**Abstract**:

Representational drift refers to over-time changes in neural activation accompanied by a stable task performance. Despite being observed in the brain and in artificial networks, the mechanisms of drift and its implications are not fully understood. Motivated by recent experimental findings of stimulus-dependent drift in the piriform cortex, we use theory and simulations to study this phenomenon in a two-layer linear feedforward network. Specifically, in a continual online learning scenario, we study the drift induced by the noise inherent in the Stochastic Gradient Descent (SGD). By decomposing the learning dynamics into the normal and tangent spaces of the minimum-loss manifold, we show the former corresponds to a finite variance fluctuation, while the latter could be considered as an effective diffusion process on the manifold. We analytically compute the fluctuation and the diffusion coefficients for the stimuli representations in the hidden layer as functions of network parameters and input distribution. Further, consistent with experiments, we show that the drift rate is slower for a more frequently presented stimulus. Overall, our analysis yields a theoretical framework for better understanding of the drift phenomenon in biological and artificial neural networks.

----

## [1139] Reducing SO(3) Convolutions to SO(2) for Efficient Equivariant GNNs

**Authors**: *Saro Passaro, C. Lawrence Zitnick*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/passaro23a.html](https://proceedings.mlr.press/v202/passaro23a.html)

**Abstract**:

Graph neural networks that model 3D data, such as point clouds or atoms, are typically desired to be $SO(3)$ equivariant, i.e., equivariant to 3D rotations. Unfortunately equivariant convolutions, which are a fundamental operation for equivariant networks, increase significantly in computational complexity as higher-order tensors are used. In this paper, we address this issue by reducing the $SO(3)$ convolutions or tensor products to mathematically equivalent convolutions in $SO(2)$ . This is accomplished by aligning the node embeddings’ primary axis with the edge vectors, which sparsifies the tensor product and reduces the computational complexity from $O(L^6)$ to $O(L^3)$, where $L$ is the degree of the representation. We demonstrate the potential implications of this improvement by proposing the Equivariant Spherical Channel Network (eSCN), a graph neural network utilizing our novel approach to equivariant convolutions, which achieves state-of-the-art results on the large-scale OC-20 and OC-22 datasets.

----

## [1140] Federated Online and Bandit Convex Optimization

**Authors**: *Kumar Kshitij Patel, Lingxiao Wang, Aadirupa Saha, Nathan Srebro*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/patel23a.html](https://proceedings.mlr.press/v202/patel23a.html)

**Abstract**:

We study the problems of distributed online and bandit convex optimization against an adaptive adversary. We aim to minimize the average regret on $M$ machines working in parallel over $T$ rounds with $R$ intermittent communications. Assuming the underlying cost functions are convex and can be generated adaptively, our results show that collaboration is not beneficial when the machines have access to the first-order gradient information at the queried points. This is in contrast to the case for stochastic functions, where each machine samples the cost functions from a fixed distribution. Furthermore, we delve into the more challenging setting of federated online optimization with bandit (zeroth-order) feedback, where the machines can only access values of the cost functions at the queried points. The key finding here is identifying the high-dimensional regime where collaboration is beneficial and may even lead to a linear speedup in the number of machines. We further illustrate our findings through federated adversarial linear bandits by developing novel distributed single and two-point feedback algorithms. Our work is the first attempt towards a systematic understanding of federated online optimization with limited feedback, and it attains tight regret bounds in the intermittent communication setting for both first and zeroth-order feedback. Our results thus bridge the gap between stochastic and adaptive settings in federated online optimization.

----

## [1141] Brauer's Group Equivariant Neural Networks

**Authors**: *Edward Pearce-Crump*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/pearce-crump23a.html](https://proceedings.mlr.press/v202/pearce-crump23a.html)

**Abstract**:

We provide a full characterisation of all of the possible group equivariant neural networks whose layers are some tensor power of $\mathbb{R}^{n}$ for three symmetry groups that are missing from the machine learning literature: $O(n)$, the orthogonal group; $SO(n)$, the special orthogonal group; and $Sp(n)$, the symplectic group. In particular, we find a spanning set of matrices for the learnable, linear, equivariant layer functions between such tensor power spaces in the standard basis of $\mathbb{R}^{n}$ when the group is $O(n)$ or $SO(n)$, and in the symplectic basis of $\mathbb{R}^{n}$ when the group is $Sp(n)$.

----

## [1142] How Jellyfish Characterise Alternating Group Equivariant Neural Networks

**Authors**: *Edward Pearce-Crump*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/pearce-crump23b.html](https://proceedings.mlr.press/v202/pearce-crump23b.html)

**Abstract**:

We provide a full characterisation of all of the possible alternating group ($A_n$) equivariant neural networks whose layers are some tensor power of $\mathbb{R}^{n}$. In particular, we find a basis of matrices for the learnable, linear, $A_n$–equivariant layer functions between such tensor power spaces in the standard basis of $\mathbb{R}^{n}$. We also describe how our approach generalises to the construction of neural networks that are equivariant to local symmetries.

----

## [1143] Can Large Language Models Reason about Program Invariants?

**Authors**: *Kexin Pei, David Bieber, Kensen Shi, Charles Sutton, Pengcheng Yin*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/pei23a.html](https://proceedings.mlr.press/v202/pei23a.html)

**Abstract**:

Identifying invariants is an important program analysis task with applications towards program understanding, bug finding, vulnerability analysis, and formal verification. Existing tools for identifying program invariants rely on dynamic analysis, requiring traces collected from multiple executions in order to produce reliable invariants. We study the application of large language models to invariant prediction, finding that models trained on source code and fine-tuned for invariant generation can perform invariant prediction as static rather than dynamic analysis. Using a scratchpad approach where invariants are predicted sequentially through a program gives the best performance, finding invariants statically of quality comparable to those obtained by a dynamic analysis tool with access to five program traces.

----

## [1144] Dynamics-inspired Neuromorphic Visual Representation Learning

**Authors**: *Zhengqi Pei, Shuhui Wang*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/pei23b.html](https://proceedings.mlr.press/v202/pei23b.html)

**Abstract**:

This paper investigates the dynamics-inspired neuromorphic architecture for visual representation learning following Hamilton’s principle. Our method converts weight-based neural structure to its dynamics-based form that consists of finite sub-models, whose mutual relations measured by computing path integrals amongst their dynamical states are equivalent to the typical neural weights. Based on the entropy reduction process derived from the Euler-Lagrange equations, the feedback signals interpreted as stress forces amongst sub-models push them to move. We first train a dynamics-based neural model from scratch and observe that this model outperforms traditional neural models on MNIST. We then convert several pre-trained neural structures into dynamics-based forms, followed by fine-tuning via entropy reduction to obtain the stabilized dynamical states. We observe consistent improvements in these transformed models over their weight-based counterparts on ImageNet and WebVision in terms of computational complexity, parameter size, testing accuracy, and robustness. Besides, we show the correlation between model performance and structural entropy, providing deeper insight into weight-free neuromorphic learning.

----

## [1145] Feature Directions Matter: Long-Tailed Learning via Rotated Balanced Representation

**Authors**: *Peifeng Gao, Qianqian Xu, Peisong Wen, Zhiyong Yang, Huiyang Shao, Qingming Huang*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/peifeng23a.html](https://proceedings.mlr.press/v202/peifeng23a.html)

**Abstract**:

Long-tailed learning is one of the most challenging problems in visual recognition. There are some studies aiming to solve long-tailed classification from the perspective of feature learning. Recent work proposes to learn the balanced representation by fixing the linear classifier as Equiangular Tight Frame (ETF), since they argue what matters in classification is the structure of the feature, instead of their directions. Holding a different view, in this paper, we show that features with fixed directions may be harmful to the generalization of models, even if it is completely symmetric. To avoid this issue, we propose Representation-Balanced Learning Framework (RBL), which introduces orthogonal matrices to learn directions while maintaining the geometric structure of ETF. Theoretically, our contributions are two-fold: 1). we point out that the feature learning of RBL is insensitive toward training set label distribution, it always learns a balanced representation space. 2). we provide a generalization analysis of proposed RBL through training stability. To analyze the stability of the parameter with orthogonal constraint, we propose a novel training stability analysis paradigm, Two-Parameter Model Stability. Practically, our method is extremely simple in implementation but shows great superiority on several benchmark datasets.

----

## [1146] Fair Neighbor Embedding

**Authors**: *Jaakko Peltonen, Wen Xu, Timo Nummenmaa, Jyrki Nummenmaa*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/peltonen23a.html](https://proceedings.mlr.press/v202/peltonen23a.html)

**Abstract**:

We consider fairness in dimensionality reduction. Nonlinear dimensionality reduction yields low dimensional representations that let users visualize and explore high-dimensional data. However, traditional dimensionality reduction may yield biased visualizations overemphasizing relationships of societal phenomena to sensitive attributes or protected groups. We introduce a framework of fair neighbor embedding, the Fair Neighbor Retrieval Visualizer, which formulates fair nonlinear dimensionality reduction as an information retrieval task whose performance and fairness are quantified by information retrieval criteria. The method optimizes low-dimensional embeddings that preserve high-dimensional data neighborhoods without yielding biased association of such neighborhoods to protected groups. In experiments the method yields fair visualizations outperforming previous methods.

----

## [1147] The Ideal Continual Learner: An Agent That Never Forgets

**Authors**: *Liangzu Peng, Paris Giampouras, René Vidal*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/peng23a.html](https://proceedings.mlr.press/v202/peng23a.html)

**Abstract**:

The goal of continual learning is to find a model that solves multiple learning tasks which are presented sequentially to the learner. A key challenge in this setting is that the learner may "forget" how to solve a previous task when learning a new task, a phenomenon known as catastrophic forgetting. To address this challenge, many practical methods have been proposed, including memory-based, regularization-based and expansion-based methods. However, a rigorous theoretical understanding of these methods remains elusive. This paper aims to bridge this gap between theory and practice by proposing a new continual learning framework called "Ideal Continual Learner" (ICL), which is guaranteed to avoid catastrophic forgetting by construction. We show that ICL unifies multiple well-established continual learning methods and gives new theoretical insights into the strengths and weaknesses of these methods. We also derive generalization bounds for ICL which allow us to theoretically quantify "how rehearsal affects generalization". Finally, we connect ICL to several classic subjects and research topics of modern interest, which allows us to make historical remarks and inspire future directions.

----

## [1148] MolDiff: Addressing the Atom-Bond Inconsistency Problem in 3D Molecule Diffusion Generation

**Authors**: *Xingang Peng, Jiaqi Guan, Qiang Liu, Jianzhu Ma*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/peng23b.html](https://proceedings.mlr.press/v202/peng23b.html)

**Abstract**:

Deep generative models have recently achieved superior performance in 3D molecule generation. Most of them first generate atoms and then add chemical bonds based on the generated atoms in a post-processing manner. However, there might be no corresponding bond solution for the temporally generated atoms as their locations are generated without considering potential bonds. We define this problem as the atom-bond inconsistency problem and claim it is the main reason for current approaches to generating unrealistic 3D molecules. To overcome this problem, we propose a new diffusion model called MolDiff which can generate atoms and bonds simultaneously while still maintaining their consistency by explicitly modeling the dependence between their relationships. We evaluated the generation ability of our proposed model and the quality of the generated molecules using criteria related to both geometry and chemical properties. The empirical studies showed that our model outperforms previous approaches, achieving a three-fold improvement in success rate and generating molecules with significantly better quality.

----

## [1149] Diagnosis, Feedback, Adaptation: A Human-in-the-Loop Framework for Test-Time Policy Adaptation

**Authors**: *Andi Peng, Aviv Netanyahu, Mark K. Ho, Tianmin Shu, Andreea Bobu, Julie Shah, Pulkit Agrawal*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/peng23c.html](https://proceedings.mlr.press/v202/peng23c.html)

**Abstract**:

Policies often fail at test-time due to distribution shifts—changes in the state and reward that occur when an end user deploys the policy in environments different from those seen in training. Data augmentation can help models be more robust to such shifts by varying specific concepts in the state, e.g. object color, that are task-irrelevant and should not impact desired actions. However, designers training the agent don’t often know which concepts are irrelevant a priori. We propose a human-in-the-loop framework to leverage feedback from the end user to quickly identify and augment task-irrelevant visual state concepts. Our framework generates counterfactual demonstrations that allow users to quickly isolate shifted state concepts and identify if they should not impact the desired task, and can therefore be augmented using existing actions. We present experiments validating our full pipeline on discrete and continuous control tasks with real human users. Our method better enables users to (1) understand agent failure, (2) improve sample efficiency of demonstrations required for finetuning, and (3) adapt the agent to their desired reward.

----

## [1150] Learning Hidden Markov Models When the Locations of Missing Observations are Unknown

**Authors**: *Binyamin Perets, Mark Kozdoba, Shie Mannor*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/perets23a.html](https://proceedings.mlr.press/v202/perets23a.html)

**Abstract**:

The Hidden Markov Model (HMM) is one of the most widely used statistical models for sequential data analysis. One of the key reasons for this versatility is the ability of HMM to deal with missing data. However, standard HMM learning algorithms rely crucially on the assumption that the positions of the missing observations within the observation sequence are known. In the natural sciences, where this assumption is often violated, special variants of HMM, commonly known as Silent-state HMMs (SHMMs), are used. Despite their widespread use, these algorithms strongly rely on specific structural assumptions of the underlying chain, such as acyclicity, thus limiting the applicability of these methods. Moreover, even in the acyclic case, it has been shown that these methods can lead to poor reconstruction. In this paper we consider the general problem of learning an HMM from data with unknown missing observation locations. We provide reconstruction algorithms that do not require any assumptions about the structure of the underlying chain, and can also be used with limited prior knowledge, unlike SHMM. We evaluate and compare the algorithms in a variety of scenarios, measuring their reconstruction precision, and robustness under model miss-specification. Notably, we show that under proper specifications one can reconstruct the process dynamics as well as if the missing observations positions were known.

----

## [1151] Estimating the Contamination Factor's Distribution in Unsupervised Anomaly Detection

**Authors**: *Lorenzo Perini, Paul-Christian Bürkner, Arto Klami*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/perini23a.html](https://proceedings.mlr.press/v202/perini23a.html)

**Abstract**:

Anomaly detection methods identify examples that do not follow the expected behaviour, typically in an unsupervised fashion, by assigning real-valued anomaly scores to the examples based on various heuristics. These scores need to be transformed into actual predictions by thresholding so that the proportion of examples marked as anomalies equals the expected proportion of anomalies, called contamination factor. Unfortunately, there are no good methods for estimating the contamination factor itself. We address this need from a Bayesian perspective, introducing a method for estimating the posterior distribution of the contamination factor for a given unlabeled dataset. We leverage several anomaly detectors to capture the basic notion of anomalousness and estimate the contamination using a specific mixture formulation. Empirically on 22 datasets, we show that the estimated distribution is well-calibrated and that setting the threshold using the posterior mean improves the detectors’ performance over several alternative methods.

----

## [1152] Are Gaussian Data All You Need? The Extents and Limits of Universality in High-Dimensional Generalized Linear Estimation

**Authors**: *Luca Pesce, Florent Krzakala, Bruno Loureiro, Ludovic Stephan*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/pesce23a.html](https://proceedings.mlr.press/v202/pesce23a.html)

**Abstract**:

In this manuscript we consider the problem of generalized linear estimation on Gaussian mixture data with labels given by a single-index model. Our first result is a sharp asymptotic expression for the test and training errors in the high-dimensional regime. Motivated by the recent stream of results on the Gaussian universality of the test and training errors in generalized linear estimation, we ask ourselves the question: "when is a single Gaussian enough to characterize the error?". Our formula allows us to give sharp answers to this question, both in the positive and negative directions. More precisely, we show that the sufficient conditions for Gaussian universality (or lack thereof) crucially depend on the alignment between the target weights and the means and covariances of the mixture clusters, which we precisely quantify. In the particular case of least-squares interpolation, we prove a strong universality property of the training error and show it follows a simple, closed-form expression. Finally, we apply our results to real datasets, clarifying some recent discussions in the literature about Gaussian universality of the errors in this context.

----

## [1153] Certifying Ensembles: A General Certification Theory with S-Lipschitzness

**Authors**: *Aleksandar Petrov, Francisco Eiras, Amartya Sanyal, Philip H. S. Torr, Adel Bibi*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/petrov23a.html](https://proceedings.mlr.press/v202/petrov23a.html)

**Abstract**:

Improving and guaranteeing the robustness of deep learning models has been a topic of intense research. Ensembling, which combines several classifiers to provide a better model, has been shown to be beneficial for generalisation, uncertainty estimation, calibration, and mitigating the effects of concept drift. However, the impact of ensembling on certified robustness is less well understood. In this work, we generalise Lipschitz continuity by introducing S-Lipschitz classifiers, which we use to analyse the theoretical robustness of ensembles. Our results are precise conditions when ensembles of robust classifiers are more robust than any constituent classifier, as well as conditions when they are less robust.

----

## [1154] The Power of Learned Locally Linear Models for Nonlinear Policy Optimization

**Authors**: *Daniel Pfrommer, Max Simchowitz, Tyler Westenbroek, Nikolai Matni, Stephen Tu*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/pfrommer23a.html](https://proceedings.mlr.press/v202/pfrommer23a.html)

**Abstract**:

A common pipeline in learning-based control is to iteratively estimate a model of system dynamics, and apply a trajectory optimization algorithm - e.g. $\mathtt{iLQR}$ - on the learned model to minimize a target cost. This paper conducts a rigorous analysis of a simplified variant of this strategy for general nonlinear systems. We analyze an algorithm which iterates between estimating local linear models of nonlinear system dynamics and performing $\mathtt{iLQR}$-like policy updates. We demonstrate that this algorithm attains sample complexity polynomial in relevant problem parameters, and, by synthesizing locally stabilizing gains, overcomes exponential dependence in problem horizon. Experimental results validate the performance of our algorithm, and compare to natural deep-learning baselines.

----

## [1155] A Scalable Frank-Wolfe-Based Algorithm for the Max-Cut SDP

**Authors**: *Chi Bach Pham, Wynita Griggs, James Saunderson*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/pham23a.html](https://proceedings.mlr.press/v202/pham23a.html)

**Abstract**:

We consider the problem of solving large-scale instances of the Max-Cut semidefinite program (SDP), i.e., optimizing a linear function over $n\times n$ positive semidefinite (PSD) matrices with unit diagonal. When the cost matrix is PSD, we show how to exactly reformulate the problem as maximizing a smooth concave function over PSD matrices with unit trace. By applying the Frank-Wolfe method, we obtain a simple algorithm that is compatible with recent sampling-based techniques to solve SDPs using low memory. We demonstrate the practical performance of our method on $10^6\times 10^6$ instances of the max-cut SDP with costs having up to $5 \times 10^6$ non-zero entries. Theoretically, we show that our method solves problems with diagonally dominant costs to relative error $\epsilon$ in $O(n\epsilon^{-1})$ calls to a randomized approximate largest eigenvalue subroutine, each of which succeeds with high probability after $O(\log(n)\epsilon^{-1/2})$ matrix-vector multiplications with the cost matrix.

----

## [1156] Attention-Based Recurrence for Multi-Agent Reinforcement Learning under Stochastic Partial Observability

**Authors**: *Thomy Phan, Fabian Ritz, Philipp Altmann, Maximilian Zorn, Jonas Nüßlein, Michael Kölle, Thomas Gabor, Claudia Linnhoff-Popien*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/phan23a.html](https://proceedings.mlr.press/v202/phan23a.html)

**Abstract**:

Stochastic partial observability poses a major challenge for decentralized coordination in multi-agent reinforcement learning but is largely neglected in state-of-the-art research due to a strong focus on state-based centralized training for decentralized execution (CTDE) and benchmarks that lack sufficient stochasticity like StarCraft Multi-Agent Challenge (SMAC). In this paper, we propose Attention-based Embeddings of Recurrence In multi-Agent Learning (AERIAL) to approximate value functions under stochastic partial observability. AERIAL replaces the true state with a learned representation of multi-agent recurrence, considering more accurate information about decentralized agent decisions than state-based CTDE. We then introduce MessySMAC, a modified version of SMAC with stochastic observations and higher variance in initial states, to provide a more general and configurable benchmark regarding stochastic partial observability. We evaluate AERIAL in Dec-Tiger as well as in a variety of SMAC and MessySMAC maps, and compare the results with state-based CTDE. Furthermore, we evaluate the robustness of AERIAL and state-based CTDE against various stochasticity configurations in MessySMAC.

----

## [1157] HyperTuning: Toward Adapting Large Language Models without Back-propagation

**Authors**: *Jason Phang, Yi Mao, Pengcheng He, Weizhu Chen*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/phang23a.html](https://proceedings.mlr.press/v202/phang23a.html)

**Abstract**:

Fine-tuning large language models for different tasks can be costly and inefficient, and even methods that reduce the number of tuned parameters still require full gradient-based optimization. We propose HyperTuning, a novel approach to model adaptation that uses a hypermodel to generate task-specific parameters for a fixed downstream model. We demonstrate a simple setup for hypertuning with HyperT5, a T5-based hypermodel that produces soft prefixes or LoRA parameters for a frozen T5 model from few-shot examples. We train HyperT5 in two stages: first, hyperpretraining with a modified conditional language modeling objective that trains a hypermodel to generate parameters; second, multi-task fine-tuning (MTF) on a large number of diverse language tasks. We evaluate HyperT5 on P3, MetaICL and Super-NaturalInstructions datasets, and show that it can effectively generate parameters for unseen tasks. Moreover, we show that using hypermodel-generated parameters as initializations for further parameter-efficient fine-tuning improves performance. HyperTuning can thus be a flexible and efficient way to leverage large language models for diverse downstream applications.

----

## [1158] Linear CNNs Discover the Statistical Structure of the Dataset Using Only the Most Dominant Frequencies

**Authors**: *Hannah Pinson, Joeri Lenaerts, Vincent Ginis*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/pinson23a.html](https://proceedings.mlr.press/v202/pinson23a.html)

**Abstract**:

We here present a stepping stone towards a deeper understanding of convolutional neural networks (CNNs) in the form of a theory of learning in linear CNNs. Through analyzing the gradient descent equations, we discover that the evolution of the network during training is determined by the interplay between the dataset structure and the convolutional network structure. We show that linear CNNs discover the statistical structure of the dataset with non-linear, ordered, stage-like transitions, and that the speed of discovery changes depending on the relationship between the dataset and the convolutional network structure. Moreover, we find that this interplay lies at the heart of what we call the "dominant frequency bias", where linear CNNs arrive at these discoveries using only the dominant frequencies of the different structural parts present in the dataset. We furthermore provide experiments that show how our theory relates to deep, non-linear CNNs used in practice. Our findings shed new light on the inner working of CNNs, and can help explain their shortcut learning and their tendency to rely on texture instead of shape.

----

## [1159] Conformal Prediction for Federated Uncertainty Quantification Under Label Shift

**Authors**: *Vincent Plassier, Mehdi Makni, Aleksandr Rubashevskii, Eric Moulines, Maxim Panov*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/plassier23a.html](https://proceedings.mlr.press/v202/plassier23a.html)

**Abstract**:

Federated Learning (FL) is a machine learning framework where many clients collaboratively train models while keeping the training data decentralized. Despite recent advances in FL, the uncertainty quantification topic (UQ) remains partially addressed. Among UQ methods, conformal prediction (CP) approaches provides distribution-free guarantees under minimal assumptions. We develop a new federated conformal prediction method based on quantile regression and take into account privacy constraints. This method takes advantage of importance weighting to effectively address the label shift between agents and provides theoretical guarantees for both valid coverage of the prediction sets and differential privacy. Extensive experimental studies demonstrate that this method outperforms current competitors.

----

## [1160] Universal Physics-Informed Neural Networks: Symbolic Differential Operator Discovery with Sparse Data

**Authors**: *Lena Podina, Brydon Eastman, Mohammad Kohandel*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/podina23a.html](https://proceedings.mlr.press/v202/podina23a.html)

**Abstract**:

In this work we perform symbolic discovery of differential operators in a situation where there is sparse experimental data. This small data regime in machine learning can be made tractable by providing our algorithms with prior information about the underlying dynamics. Physics Informed Neural Networks (PINNs) have been very successful in this regime (reconstructing entire ODE solutions using only a single point or entire PDE solutions with very few measurements of the initial condition). The Universal PINN approach (UPINN) adds a neural network that learns a representation of unknown hidden terms in the differential equation. The algorithm yields both a surrogate solution to the differential equation and a black-box representation of the hidden terms. These hidden term neural networks can then be converted into symbolic equations using symbolic regression techniques like AI Feynman. In order to achieve convergence of the neural networks, we provide our algorithms with (noisy) measurements of both the initial condition as well as (synthetic) experimental data obtained at later times. We demonstrate strong performance of UPINNs even when provided with very few measurements of noisy data in both the ODE and PDE regime.

----

## [1161] Sequential Kernelized Independence Testing

**Authors**: *Aleksandr Podkopaev, Patrick Blöbaum, Shiva Prasad Kasiviswanathan, Aaditya Ramdas*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/podkopaev23a.html](https://proceedings.mlr.press/v202/podkopaev23a.html)

**Abstract**:

Independence testing is a classical statistical problem that has been extensively studied in the batch setting when one fixes the sample size before collecting data. However, practitioners often prefer procedures that adapt to the complexity of a problem at hand instead of setting sample size in advance. Ideally, such procedures should (a) stop earlier on easy tasks (and later on harder tasks), hence making better use of available resources, and (b) continuously monitor the data and efficiently incorporate statistical evidence after collecting new data, while controlling the false alarm rate. Classical batch tests are not tailored for streaming data: valid inference after data peeking requires correcting for multiple testing which results in low power. Following the principle of testing by betting, we design sequential kernelized independence tests that overcome such shortcomings. We exemplify our broad framework using bets inspired by kernelized dependence measures, e.g., the Hilbert-Schmidt independence criterion. Our test is also valid under non-i.i.d. time-varying settings. We demonstrate the power of our approaches on both simulated and real data.

----

## [1162] Truncating Trajectories in Monte Carlo Reinforcement Learning

**Authors**: *Riccardo Poiani, Alberto Maria Metelli, Marcello Restelli*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/poiani23a.html](https://proceedings.mlr.press/v202/poiani23a.html)

**Abstract**:

In Reinforcement Learning (RL), an agent acts in an unknown environment to maximize the expected cumulative discounted sum of an external reward signal, i.e., the expected return. In practice, in many tasks of interest, such as policy optimization, the agent usually spends its interaction budget by collecting episodes of fixed length within a simulator (i.e., Monte Carlo simulation). However, given the discounted nature of the RL objective, this data collection strategy might not be the best option. Indeed, the rewards taken in early simulation steps weigh exponentially more than future rewards. Taking a cue from this intuition, in this paper, we design an a-priori budget allocation strategy that leads to the collection of trajectories of different lengths, i.e., truncated. The proposed approach provably minimizes the width of the confidence intervals around the empirical estimates of the expected return of a policy. After discussing the theoretical properties of our method, we make use of our trajectory truncation mechanism to extend Policy Optimization via Importance Sampling (POIS, Metelli et al., 2018) algorithm. Finally, we conduct a numerical comparison between our algorithm and POIS: the results are consistent with our theory and show that an appropriate truncation of the trajectories can succeed in improving performance.

----

## [1163] Hyena Hierarchy: Towards Larger Convolutional Language Models

**Authors**: *Michael Poli, Stefano Massaroli, Eric Nguyen, Daniel Y. Fu, Tri Dao, Stephen Baccus, Yoshua Bengio, Stefano Ermon, Christopher Ré*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/poli23a.html](https://proceedings.mlr.press/v202/poli23a.html)

**Abstract**:

Recent advances in deep learning have relied heavily on the use of large Transformers due to their ability to learn at scale. However, the core building block of Transformers, the attention operator, exhibits quadratic cost in sequence length, limiting the amount of context accessible. Existing subquadratic methods based on low-rank and sparse approximations need to be combined with dense attention layers to match Transformers at scale, indicating a gap in capability. In this work, we propose Hyena, a subquadratic drop-in replacement for attention constructed by interleaving implicitly parametrized long convolutions and data-controlled gating. In challenging reasoning tasks on sequences of thousands to hundreds of thousands of tokens, Hyena improves accuracy by more than 50 points over operators relying on state-space models, transfer functions, and other implicit and explicit methods, matching attention-based models. We set a new state-of-the-art for dense-attention-free architectures on language modeling in standard datasets WikiText103 and The Pile, reaching Transformer quality with a 20% reduction in training compute required at sequence length 2k. Hyena operators are 2x faster than highly optimized attention at sequence length 8k, with speedups of 100x at 64k.

----

## [1164] Spurious Valleys and Clustering Behavior of Neural Networks

**Authors**: *Samuele Pollaci*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/pollaci23a.html](https://proceedings.mlr.press/v202/pollaci23a.html)

**Abstract**:

Neural networks constitute a class of functions that are typically non-surjective, with high-dimensional fibers and complicated image. We prove two main results concerning the geometry of the loss landscape of a neural network. First, we provide an explicit effective bound on the sizes of the hidden layers so that the loss landscape has no spurious valleys, which guarantees the success of gradient descent methods. Second, we present a novel method for analyzing whether a given neural network architecture with monomial activation function can represent a target function of interest. The core of our analysis method is the study of a specific set of error values, and its behavior depending on different training datasets.

----

## [1165] Multisample Flow Matching: Straightening Flows with Minibatch Couplings

**Authors**: *Aram-Alexandre Pooladian, Heli Ben-Hamu, Carles Domingo-Enrich, Brandon Amos, Yaron Lipman, Ricky T. Q. Chen*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/pooladian23a.html](https://proceedings.mlr.press/v202/pooladian23a.html)

**Abstract**:

Simulation-free methods for training continuous-time generative models construct probability paths that go between noise distributions and individual data samples. Recent works, such as Flow Matching, derived paths that are optimal for each data sample. However, these algorithms rely on independent data and noise samples, and do not exploit underlying structure in the data distribution for constructing probability paths. We propose Multisample Flow Matching, a more general framework that uses non-trivial couplings between data and noise samples while satisfying the correct marginal constraints. At small overhead costs, this generalization allows us to (i) reduce gradient variance during training, (ii) obtain straighter flows for the learned vector field, which allows us to generate high-quality samples using fewer function evaluations, and (iii) obtain transport maps with low cost in high dimensions, which has applications beyond generative modeling. Importantly, we do so in a completely simulation-free manner with a simple minimization objective. We show that our proposed methods improve sample consistency on downsampled ImageNet data sets, and lead to better low-cost sample generation.

----

## [1166] Minimax estimation of discontinuous optimal transport maps: The semi-discrete case

**Authors**: *Aram-Alexandre Pooladian, Vincent Divol, Jonathan Niles-Weed*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/pooladian23b.html](https://proceedings.mlr.press/v202/pooladian23b.html)

**Abstract**:

We consider the problem of estimating the optimal transport map between two probability distributions, $P$ and $Q$ in $\mathbb{R}^d$, on the basis of i.i.d. samples. All existing statistical analyses of this problem require the assumption that the transport map is Lipschitz, a strong requirement that, in particular, excludes any examples where the transport map is discontinuous. As a first step towards developing estimation procedures for discontinuous maps, we consider the important special case where the data distribution $Q$ is a discrete measure supported on a finite number of points in $\mathbb{R}^d$. We study a computationally efficient estimator initially proposed by (Pooladian & Niles-Weed, 2021), based on entropic optimal transport, and show in the semi-discrete setting that it converges at the minimax-optimal rate $n^{-1/2}$, independent of dimension. Other standard map estimation techniques both lack finite-sample guarantees in this setting and provably suffer from the curse of dimensionality. We confirm these results in numerical experiments, and provide experiments for other settings, not covered by our theory, which indicate that the entropic estimator is a promising methodology for other discontinuous transport map estimation problems.

----

## [1167] Test-time Adaptation with Slot-Centric Models

**Authors**: *Mihir Prabhudesai, Anirudh Goyal, Sujoy Paul, Sjoerd van Steenkiste, Mehdi S. M. Sajjadi, Gaurav Aggarwal, Thomas Kipf, Deepak Pathak, Katerina Fragkiadaki*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/prabhudesai23a.html](https://proceedings.mlr.press/v202/prabhudesai23a.html)

**Abstract**:

Current visual detectors, though impressive within their training distribution, often fail to parse out-of-distribution scenes into their constituent entities. Recent test-time adaptation methods use auxiliary self-supervised losses to adapt the network parameters to each test example independently and have shown promising results towards generalization outside the training distribution for the task of image classification. In our work, we find evidence that these losses are insufficient for the task of scene decomposition, without also considering architectural inductive biases. Recent slot-centric generative models attempt to decompose scenes into entities in a self-supervised manner by reconstructing pixels. Drawing upon these two lines of work, we propose Slot-TTA, a semi-supervised slot-centric scene decomposition model that at test time is adapted per scene through gradient descent on reconstruction or cross-view synthesis objectives. We evaluate Slot-TTA across multiple input modalities, images or 3D point clouds, and show substantial out-of-distribution performance improvements against state-of-the-art supervised feed-forward detectors, and alternative test-time adaptation methods. Project Webpage: http://slot-tta.github.io/

----

## [1168] JAWS-X: Addressing Efficiency Bottlenecks of Conformal Prediction Under Standard and Feedback Covariate Shift

**Authors**: *Drew Prinster, Suchi Saria, Anqi Liu*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/prinster23a.html](https://proceedings.mlr.press/v202/prinster23a.html)

**Abstract**:

We study the efficient estimation of predictive confidence intervals for black-box predictors when the common data exchangeability (e.g., i.i.d.) assumption is violated due to potentially feedback-induced shifts in the input data distribution. That is, we focus on standard and feedback covariate shift (FCS), where the latter allows for feedback dependencies between train and test data that occur in many decision-making scenarios like experimental design. Whereas prior conformal prediction methods for this problem are in general either extremely computationally demanding or make inefficient use of labeled data, we propose a collection of methods based on the jackknife+ that achieve a practical balance of computational and statistical efficiency. Theoretically, our proposed JAW-FCS method extends the rigorous, finite-sample coverage guarantee of the jackknife+ to FCS. We moreover propose two tunable relaxations to JAW-FCS’s computation that maintain finite-sample guarantees: one using only $K$ leave-one-out models (JAW-$K$LOO) and a second building on $K$-fold cross validation+ (WCV+). Practically, we demonstrate that JAW-FCS and its computational relaxations outperform state-of-the-art baselines on a variety of real-world datasets under standard and feedback covariate shift, including for biomolecular design and active learning tasks.

----

## [1169] Equivariant Polynomials for Graph Neural Networks

**Authors**: *Omri Puny, Derek Lim, Bobak Toussi Kiani, Haggai Maron, Yaron Lipman*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/puny23a.html](https://proceedings.mlr.press/v202/puny23a.html)

**Abstract**:

Graph Neural Networks (GNN) are inherently limited in their expressive power. Recent seminal works (Xu et al., 2019; Morris et al., 2019b) introduced the Weisfeiler-Lehman (WL) hierarchy as a measure of expressive power. Although this hierarchy has propelled significant advances in GNN analysis and architecture developments, it suffers from several significant limitations. These include a complex definition that lacks direct guidance for model improvement and a WL hierarchy that is too coarse to study current GNNs. This paper introduces an alternative expressive power hierarchy based on the ability of GNNs to calculate equivariant polynomials of a certain degree. As a first step, we provide a full characterization of all equivariant graph polynomials by introducing a concrete basis, significantly generalizing previous results. Each basis element corresponds to a specific multi-graph, and its computation over some graph data input corresponds to a tensor contraction problem. Second, we propose algorithmic tools for evaluating the expressiveness of GNNs using tensor contraction sequences, and calculate the expressive power of popular GNNs. Finally, we enhance the expressivity of common GNN architectures by adding polynomial features or additional operations / aggregations inspired by our theory. These enhanced GNNs demonstrate state-of-the-art results in experiments across multiple graph learning benchmarks.

----

## [1170] Contrast with Reconstruct: Contrastive 3D Representation Learning Guided by Generative Pretraining

**Authors**: *Zekun Qi, Runpei Dong, Guofan Fan, Zheng Ge, Xiangyu Zhang, Kaisheng Ma, Li Yi*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/qi23a.html](https://proceedings.mlr.press/v202/qi23a.html)

**Abstract**:

Mainstream 3D representation learning approaches are built upon contrastive or generative modeling pretext tasks, where great improvements in performance on various downstream tasks have been achieved. However, we find these two paradigms have different characteristics: (i) contrastive models are data-hungry that suffer from a representation over-fitting issue; (ii) generative models have a data filling issue that shows inferior data scaling capacity compared to contrastive models. This motivates us to learn 3D representations by sharing the merits of both paradigms, which is non-trivial due to the pattern difference between the two paradigms. In this paper, we propose contrast with reconstruct (ReCon) that unifies these two paradigms. ReCon is trained to learn from both generative modeling teachers and cross-modal contrastive teachers through ensemble distillation, where the generative student is used to guide the contrastive student. An encoder-decoder style ReCon-block is proposed that transfers knowledge through cross attention with stop-gradient, which avoids pretraining over-fitting and pattern difference issues. ReCon achieves a new state-of-the-art in 3D representation learning, e.g., 91.26% accuracy on ScanObjectNN. Codes have been released at https://github.com/qizekun/ReCon.

----

## [1171] An Effective Meaningful Way to Evaluate Survival Models

**Authors**: *Shiang Qi, Neeraj Kumar, Mahtab Farrokh, Weijie Sun, Li-Hao Kuan, Rajesh Ranganath, Ricardo Henao, Russell Greiner*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/qi23b.html](https://proceedings.mlr.press/v202/qi23b.html)

**Abstract**:

One straightforward metric to evaluate a survival prediction model is based on the Mean Absolute Error (MAE) – the average of the absolute difference between the time predicted by the model and the true event time, over all subjects. Unfortunately, this is challenging because, in practice, the test set includes (right) censored individuals, meaning we do not know when a censored individual actually experienced the event. In this paper, we explore various metrics to estimate MAE for survival datasets that include (many) censored individuals. Moreover, we introduce a novel and effective approach for generating realistic semi-synthetic survival datasets to facilitate the evaluation of metrics. Our findings, based on the analysis of the semi-synthetic datasets, reveal that our proposed metric (MAE using pseudo-observations) is able to rank models accurately based on their performance, and often closely matches the true MAE – in particular, is better than several alternative methods.

----

## [1172] Coarse-to-Fine: a Hierarchical Diffusion Model for Molecule Generation in 3D

**Authors**: *Bo Qiang, Yuxuan Song, Minkai Xu, Jingjing Gong, Bowen Gao, Hao Zhou, Wei-Ying Ma, Yanyan Lan*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/qiang23a.html](https://proceedings.mlr.press/v202/qiang23a.html)

**Abstract**:

Generating desirable molecular structures in 3D is a fundamental problem for drug discovery. Despite the considerable progress we have achieved, existing methods usually generate molecules in atom resolution and ignore intrinsic local structures such as rings, which leads to poor quality in generated structures, especially when generating large molecules. Fragment-based molecule generation is a promising strategy, however, it is nontrivial to be adapted for 3D non-autoregressive generations because of the combinational optimization problems. In this paper, we utilize a coarse-to-fine strategy to tackle this problem, in which a Hierarchical Diffusion-based model (i.e. HierDiff) is proposed to preserve the validity of local segments without relying on autoregressive modeling. Specifically, HierDiff first generates coarse-grained molecule geometries via an equivariant diffusion process, where each coarse-grained node reflects a fragment in a molecule. Then the coarse-grained nodes are decoded into fine-grained fragments by a message-passing process and a newly designed iterative refined sampling module. Lastly, the fine-grained fragments are then assembled to derive a complete atomic molecular structure. Extensive experiments demonstrate that HierDiff consistently improves the quality of molecule generation over existing methods.

----

## [1173] Collaborative Causal Inference with Fair Incentives

**Authors**: *Rui Qiao, Xinyi Xu, Bryan Kian Hsiang Low*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/qiao23a.html](https://proceedings.mlr.press/v202/qiao23a.html)

**Abstract**:

Collaborative causal inference (CCI) aims to improve the estimation of the causal effect of treatment variables by utilizing data aggregated from multiple self-interested parties. Since their source data are valuable proprietary assets that can be costly or tedious to obtain, every party has to be incentivized to be willing to contribute to the collaboration, such as with a guaranteed fair and sufficiently valuable reward (than performing causal inference on its own). This paper presents a reward scheme designed using the unique statistical properties that are required by causal inference to guarantee certain desirable incentive criteria (e.g., fairness, benefit) for the parties based on their contributions. To achieve this, we propose a data valuation function to value parties’ data for CCI based on the distributional closeness of its resulting treatment effect estimate to that utilizing the aggregated data from all parties. Then, we show how to value the parties’ rewards fairly based on a modified variant of the Shapley value arising from our proposed data valuation for CCI. Finally, the Shapley fair rewards to the parties are realized in the form of improved, stochastically perturbed treatment effect estimates. We empirically demonstrate the effectiveness of our reward scheme using simulated and real-world datasets.

----

## [1174] FREDIS: A Fusion Framework of Refinement and Disambiguation for Unreliable Partial Label Learning

**Authors**: *Congyu Qiao, Ning Xu, Jiaqi Lv, Yi Ren, Xin Geng*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/qiao23b.html](https://proceedings.mlr.press/v202/qiao23b.html)

**Abstract**:

To reduce the difficulty of annotation, partial label learning (PLL) has been widely studied, where each example is ambiguously annotated with a set of candidate labels instead of the exact correct label. PLL assumes that the candidate label set contains the correct label, which induces disambiguation, i.e., identification of the correct label in the candidate label set, adopted in most PLL methods. However, this assumption is impractical as no one could guarantee the existence of the correct label in the candidate label set under real-world scenarios. Therefore, Unreliable Partial Label Learning (UPLL) is investigated where the correct label of each example may not exist in the candidate label set. In this paper, we propose a fusion framework of refinement and disambiguation named FREDIS to handle the UPLL problem. Specifically, with theoretical guarantees, not only does disambiguation move incorrect labels from candidate labels to non-candidate labels but also refinement, an opposite procedure, moves correct labels from non-candidate labels to candidate labels. Besides, we prove that the classifier trained by our framework could eventually approximate the Bayes optimal classifier. Extensive experiments on widely used benchmark datasets validate the effectiveness of our proposed framework.

----

## [1175] Nugget: Neural Agglomerative Embeddings of Text

**Authors**: *Guanghui Qin, Benjamin Van Durme*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/qin23a.html](https://proceedings.mlr.press/v202/qin23a.html)

**Abstract**:

Embedding text sequences is a widespread requirement in modern language understanding. Existing approaches focus largely on constant-size representations. This is problematic, as the amount of information contained in text often varies with the length of the input. We propose a solution called Nugget, which encodes language into a representation based on a dynamically selected subset of input tokens. These nuggets are learned through tasks like autoencoding and machine translation, and intuitively segment language into meaningful units. We demonstrate Nugget outperforms related approaches in tasks involving semantic comparison. Finally, we illustrate these compact units allow for expanding the contextual window of a language model (LM), suggesting new future LMs that can condition on significantly larger amounts of content.

----

## [1176] BiBench: Benchmarking and Analyzing Network Binarization

**Authors**: *Haotong Qin, Mingyuan Zhang, Yifu Ding, Aoyu Li, Zhongang Cai, Ziwei Liu, Fisher Yu, Xianglong Liu*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/qin23b.html](https://proceedings.mlr.press/v202/qin23b.html)

**Abstract**:

Network binarization emerges as one of the most promising compression approaches offering extraordinary computation and memory savings by minimizing the bit-width. However, recent research has shown that applying existing binarization algorithms to diverse tasks, architectures, and hardware in realistic scenarios is still not straightforward. Common challenges of binarization, such as accuracy degradation and efficiency limitation, suggest that its attributes are not fully understood. To close this gap, we present BiBench, a rigorously designed benchmark with in-depth analysis for network binarization. We first carefully scrutinize the requirements of binarization in the actual production and define evaluation tracks and metrics for a comprehensive and fair investigation. Then, we evaluate and analyze a series of milestone binarization algorithms that function at the operator level and with extensive influence. Our benchmark reveals that 1) the binarized operator has a crucial impact on the performance and deployability of binarized networks; 2) the accuracy of binarization varies significantly across different learning tasks and neural architectures; 3) binarization has demonstrated promising efficiency potential on edge devices despite the limited hardware support. The results and analysis also lead to a promising paradigm for accurate and efficient binarization. We believe that BiBench will contribute to the broader adoption of binarization and serve as a foundation for future research. The code for our BiBench is released https://github.com/htqin/BiBench .

----

## [1177] Not All Semantics are Created Equal: Contrastive Self-supervised Learning with Automatic Temperature Individualization

**Authors**: *Zi-Hao Qiu, Quanqi Hu, Zhuoning Yuan, Denny Zhou, Lijun Zhang, Tianbao Yang*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/qiu23a.html](https://proceedings.mlr.press/v202/qiu23a.html)

**Abstract**:

In this paper, we aim to optimize a contrastive loss with individualized temperatures in a principled manner. The common practice of using a global temperature parameter $\tau$ ignores the fact that “not all semantics are created equal", meaning that different anchor data may have different numbers of samples with similar semantics, especially when data exhibits long-tails. First, we propose a new robust contrastive loss inspired by distributionally robust optimization (DRO), providing us an intuition about the effect of $\tau$ and a mechanism for automatic temperature individualization. Then, we propose an efficient stochastic algorithm for optimizing the robust contrastive loss with a provable convergence guarantee without using large mini-batch sizes. Theoretical and experimental results show that our algorithm automatically learns a suitable $\tau$ for each sample. Specifically, samples with frequent semantics use large temperatures to keep local semantic structures, while samples with rare semantics use small temperatures to induce more separable features. Our method not only outperforms prior strong baselines (e.g., SimCLR, CLIP) on unimodal and bimodal tasks with larger improvements on imbalanced data but also is less sensitive to hyper-parameters. To our best knowledge, this is the first methodical approach to optimizing a contrastive loss with individualized temperatures. Our proposed algorithms are implemented in the LibAUC library at https://libauc.org.

----

## [1178] Shortest Edit Path Crossover: A Theory-driven Solution to the Permutation Problem in Evolutionary Neural Architecture Search

**Authors**: *Xin Qiu, Risto Miikkulainen*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/qiu23b.html](https://proceedings.mlr.press/v202/qiu23b.html)

**Abstract**:

Population-based search has recently emerged as a possible alternative to Reinforcement Learning (RL) for black-box neural architecture search (NAS). It performs well in practice even though it is not theoretically well understood. In particular, whereas traditional population-based search methods such as evolutionary algorithms (EAs) draw much power from crossover operations, it is difficult to take advantage of them in NAS. The main obstacle is believed to be the permutation problem: The mapping between genotype and phenotype in traditional graph representations is many-to-one, leading to a disruptive effect of standard crossover. This paper presents the first theoretical analysis of the behaviors of mutation, crossover and RL in black-box NAS, and proposes a new crossover operator based on the shortest edit path (SEP) in graph space. The SEP crossover is shown theoretically to overcome the permutation problem, and as a result, have a better expected improvement compared to mutation, standard crossover and RL. Further, it empirically outperform these other methods on state-of-the-art NAS benchmarks. The SEP crossover therefore allows taking full advantage of population-based search in NAS, and the underlying theory can serve as a foundation for deeper understanding of black-box NAS methods in general.

----

## [1179] Simple and Fast Group Robustness by Automatic Feature Reweighting

**Authors**: *Shikai Qiu, Andres Potapczynski, Pavel Izmailov, Andrew Gordon Wilson*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/qiu23c.html](https://proceedings.mlr.press/v202/qiu23c.html)

**Abstract**:

A major challenge to out-of-distribution generalization is reliance on spurious features — patterns that are predictive of the class label in the training data distribution, but not causally related to the target. Standard methods for reducing the reliance on spurious features typically assume that we know what the spurious feature is, which is rarely true in the real world. Methods that attempt to alleviate this limitation are complex, hard to tune, and lead to a significant computational overhead compared to standard training. In this paper, we propose Automatic Feature Reweighting (AFR), an extremely simple and fast method for updating the model to reduce the reliance on spurious features. AFR retrains the last layer of a standard ERM-trained base model with a weighted loss that emphasizes the examples where the ERM model predicts poorly, automatically upweighting the minority group without group labels. With this simple procedure, we improve upon the best reported results among competing methods trained without spurious attributes on several vision and natural language classification benchmarks, using only a fraction of their compute.

----

## [1180] DRCFS: Doubly Robust Causal Feature Selection

**Authors**: *Francesco Quinzan, Ashkan Soleymani, Patrick Jaillet, Cristian R. Rojas, Stefan Bauer*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/quinzan23a.html](https://proceedings.mlr.press/v202/quinzan23a.html)

**Abstract**:

Knowing the features of a complex system that are highly relevant to a particular target variable is of fundamental interest in many areas of science. Existing approaches are often limited to linear settings, sometimes lack guarantees, and in most cases, do not scale to the problem at hand, in particular to images. We propose DRCFS, a doubly robust feature selection method for identifying the causal features even in nonlinear and high dimensional settings. We provide theoretical guarantees, illustrate necessary conditions for our assumptions, and perform extensive experiments across a wide range of simulated and semi-synthetic datasets. DRCFS significantly outperforms existing state-of-the-art methods, selecting robust features even in challenging highly non-linear and high-dimensional problems.

----

## [1181] Robust Speech Recognition via Large-Scale Weak Supervision

**Authors**: *Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey, Ilya Sutskever*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/radford23a.html](https://proceedings.mlr.press/v202/radford23a.html)

**Abstract**:

We study the capabilities of speech processing systems trained simply to predict large amounts of transcripts of audio on the internet. When scaled to 680,000 hours of multilingual and multitask supervision, the resulting models generalize well to standard benchmarks and are often competitive with prior fully supervised results without the need for any dataset specific fine-tuning. When compared to humans, the models approach their accuracy and robustness. We are releasing models and inference code to serve as a foundation for further work on robust speech processing.

----

## [1182] Shiftable Context: Addressing Training-Inference Context Mismatch in Simultaneous Speech Translation

**Authors**: *Matthew Raffel, Drew Penney, Lizhong Chen*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/raffel23a.html](https://proceedings.mlr.press/v202/raffel23a.html)

**Abstract**:

Transformer models using segment-based processing have been an effective architecture for simultaneous speech translation. However, such models create a context mismatch between training and inference environments, hindering potential translation accuracy. We solve this issue by proposing Shiftable Context, a simple yet effective scheme to ensure that consistent segment and context sizes are maintained throughout training and inference, even with the presence of partially filled segments due to the streaming nature of simultaneous translation. Shiftable Context is also broadly applicable to segment-based transformers for streaming tasks. Our experiments on the English-German, English-French, and English-Spanish language pairs from the MUST-C dataset demonstrate that when applied to the Augmented Memory Transformer, a state-of-the-art model for simultaneous speech translation, the proposed scheme achieves an average increase of 2.09, 1.83, and 1.95 BLEU scores across each wait-k value for the three language pairs, respectively, with a minimal impact on computation-aware Average Lagging.

----

## [1183] Sequential Multi-Dimensional Self-Supervised Learning for Clinical Time Series

**Authors**: *Aniruddh Raghu, Payal Chandak, Ridwan Alam, John V. Guttag, Collin M. Stultz*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/raghu23a.html](https://proceedings.mlr.press/v202/raghu23a.html)

**Abstract**:

Self-supervised learning (SSL) for clinical time series data has received significant attention in recent literature, since these data are highly rich and provide important information about a patient’s physiological state. However, most existing SSL methods for clinical time series are limited in that they are designed for unimodal time series, such as a sequence of structured features (e.g., lab values and vitals signs) or an individual high-dimensional physiological signal (e.g., an electrocardiogram). These existing methods cannot be readily extended to model time series that exhibit multimodality, with structured features and high-dimensional data being recorded at each timestep in the sequence. In this work, we address this gap and propose a new SSL method — Sequential Multi-Dimensional SSL — where a SSL loss is applied both at the level of the entire sequence and at the level of the individual high-dimensional data points in the sequence in order to better capture information at both scales. Our strategy is agnostic to the specific form of loss function used at each level – it can be contrastive, as in SimCLR, or non-contrastive, as in VICReg. We evaluate our method on two real-world clinical datasets, where the time series contains sequences of (1) high-frequency electrocardiograms and (2) structured data from lab values and vitals signs. Our experimental results indicate that pre-training with our method and then fine-tuning on downstream tasks improves performance over baselines on both datasets, and in several settings, can lead to improvements across different self-supervised loss functions.

----

## [1184] Recovery Bounds on Class-Based Optimal Transport: A Sum-of-Norms Regularization Framework

**Authors**: *Arman Rahbar, Ashkan Panahi, Morteza Haghir Chehreghani, Devdatt P. Dubhashi, Hamid Krim*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/rahbar23a.html](https://proceedings.mlr.press/v202/rahbar23a.html)

**Abstract**:

We develop a novel theoretical framework for understating Optimal Transport (OT) schemes respecting a class structure. For this purpose, we propose a convex OT program with a sum-of-norms regularization term, which provably recovers the underlying class structure under geometric assumptions. Furthermore, we derive an accelerated proximal algorithm with a closed-form projection and proximal operator scheme, thereby affording a more scalable algorithm for computing optimal transport plans. We provide a novel argument for the uniqueness of the optimum even in the absence of strong convexity. Our experiments show that the new regularizer not only results in a better preservation of the class structure in the data but also yields additional robustness to the data geometry, compared to previous regularizers.

----

## [1185] Algorithmic Stability of Heavy-Tailed SGD with General Loss Functions

**Authors**: *Anant Raj, Lingjiong Zhu, Mert Gürbüzbalaban, Umut Simsekli*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/raj23a.html](https://proceedings.mlr.press/v202/raj23a.html)

**Abstract**:

Heavy-tail phenomena in stochastic gradient descent (SGD) have been reported in several empirical studies. Experimental evidence in previous works suggests a strong interplay between the heaviness of the tails and generalization behavior of SGD. To address this empirical phenomena theoretically, several works have made strong topological and statistical assumptions to link the generalization error to heavy tails. Very recently, new generalization bounds have been proven, indicating a non-monotonic relationship between the generalization error and heavy tails, which is more pertinent to the reported empirical observations. While these bounds do not require additional topological assumptions given that SGD can be modeled using a heavy-tailed stochastic differential equation (SDE), they can only apply to simple quadratic problems. In this paper, we build on this line of research and develop generalization bounds for a more general class of objective functions, which includes non-convex functions as well. Our approach is based on developing Wasserstein stability bounds for heavy-tailed SDEs and their discretizations, which we then convert to generalization bounds. Our results do not require any nontrivial assumptions; yet, they shed more light to the empirical observations, thanks to the generality of the loss functions.

----

## [1186] Mastering the Unsupervised Reinforcement Learning Benchmark from Pixels

**Authors**: *Sai Rajeswar, Pietro Mazzaglia, Tim Verbelen, Alexandre Piché, Bart Dhoedt, Aaron C. Courville, Alexandre Lacoste*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/rajeswar23a.html](https://proceedings.mlr.press/v202/rajeswar23a.html)

**Abstract**:

Controlling artificial agents from visual sensory data is an arduous task. Reinforcement learning (RL) algorithms can succeed but require large amounts of interactions between the agent and the environment. To alleviate the issue, unsupervised RL proposes to employ self-supervised interaction and learning, for adapting faster to future tasks. Yet, as shown in the Unsupervised RL Benchmark (URLB; Laskin et al. 2021), whether current unsupervised strategies can improve generalization capabilities is still unclear, especially in visual control settings. In this work, we study the URLB and propose a new method to solve it, using unsupervised model-based RL, for pre-training the agent, and a task-aware fine-tuning strategy combined with a new proposed hybrid planner, Dyna-MPC, to adapt the agent for downstream tasks. On URLB, our method obtains 93.59% overall normalized performance, surpassing previous baselines by a staggering margin. The approach is empirically evaluated through a large-scale empirical study, which we use to validate our design choices and analyze our models. We also show robust performance on the Real-Word RL benchmark, hinting at resiliency to environment perturbations during adaptation. Project website: https://masteringurlb.github.io/

----

## [1187] SpotEM: Efficient Video Search for Episodic Memory

**Authors**: *Santhosh Kumar Ramakrishnan, Ziad Al-Halah, Kristen Grauman*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/ramakrishnan23a.html](https://proceedings.mlr.press/v202/ramakrishnan23a.html)

**Abstract**:

The goal in episodic memory (EM) is to search a long egocentric video to answer a natural language query (e.g., “where did I leave my purse?”). Existing EM methods exhaustively extract expensive fixed-length clip features to look everywhere in the video for the answer, which is infeasible for long wearable-camera videos that span hours or even days. We propose SpotEM, an approach to achieve efficiency for a given EM method while maintaining good accuracy. SpotEM consists of three key ideas: 1) a novel clip selector that learns to identify promising video regions to search conditioned on the language query; 2) a set of low-cost semantic indexing features that capture the context of rooms, objects, and interactions that suggest where to look; and 3) distillation losses that address the optimization issues arising from end-to-end joint training of the clip selector and EM model. Our experiments on 200+ hours of video from the Ego4D EM Natural Language Queries benchmark and three different EM models demonstrate the effectiveness of our approach: computing only 10% – 25% of the clip features, we preserve 84% – 97% of the original EM model’s accuracy. Project page: https://vision.cs.utexas.edu/projects/spotem

----

## [1188] How much does Initialization Affect Generalization?

**Authors**: *Sameera Ramasinghe, Lachlan Ewen MacDonald, Moshiur R. Farazi, Hemanth Saratchandran, Simon Lucey*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/ramasinghe23a.html](https://proceedings.mlr.press/v202/ramasinghe23a.html)

**Abstract**:

Characterizing the remarkable generalization properties of over-parameterized neural networks remains an open problem. A growing body of recent literature shows that the bias of stochastic gradient descent (SGD) and architecture choice implicitly leads to better generalization. In this paper, we show on the contrary that, independently of architecture, SGD can itself be the cause of poor generalization if one does not ensure good initialization. Specifically, we prove that any differentiably parameterized model, trained under gradient flow, obeys a weak spectral bias law which states that sufficiently high frequencies train arbitrarily slowly. This implies that very high frequencies present at initialization will remain after training, and hamper generalization. Further, we empirically test the developed theoretical insights using practical, deep networks. Finally, we contrast our framework with that supplied by the flat-minima conjecture and show that Fourier analysis grants a more reliable framework for understanding the generalization of neural networks.

----

## [1189] Model Ratatouille: Recycling Diverse Models for Out-of-Distribution Generalization

**Authors**: *Alexandre Ramé, Kartik Ahuja, Jianyu Zhang, Matthieu Cord, Léon Bottou, David Lopez-Paz*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/rame23a.html](https://proceedings.mlr.press/v202/rame23a.html)

**Abstract**:

Foundation models are redefining how AI systems are built. Practitioners now follow a standard procedure to build their machine learning solutions: from a pre-trained foundation model, they fine-tune the weights on the target task of interest. So, the Internet is swarmed by a handful of foundation models fine-tuned on many diverse tasks: these individual fine-tunings exist in isolation without benefiting from each other. In our opinion, this is a missed opportunity, as these specialized models contain rich and diverse features. In this paper, we thus propose model ratatouille, a new strategy to recycle the multiple fine-tunings of the same foundation model on diverse auxiliary tasks. Specifically, we repurpose these auxiliary weights as initializations for multiple parallel fine-tunings on the target task; then, we average all fine-tuned weights to obtain the final model. This recycling strategy aims at maximizing the diversity in weights by leveraging the diversity in auxiliary tasks. Empirically, it improves the state of the art on the reference DomainBed benchmark for out-of-distribution generalization. Looking forward, this work contributes to the emerging paradigm of updatable machine learning where, akin to open-source software development, the community collaborates to reliably update machine learning models.

----

## [1190] A Picture of the Space of Typical Learnable Tasks

**Authors**: *Rahul Ramesh, Jialin Mao, Itay Griniasty, Rubing Yang, Han Kheng Teoh, Mark K. Transtrum, James P. Sethna, Pratik Chaudhari*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/ramesh23a.html](https://proceedings.mlr.press/v202/ramesh23a.html)

**Abstract**:

We develop information geometric techniques to understand the representations learned by deep networks when they are trained on different tasks using supervised, meta-, semi-supervised and contrastive learning. We shed light on the following phenomena that relate to the structure of the space of tasks: (1) the manifold of probabilistic models trained on different tasks using different representation learning methods is effectively low-dimensional; (2) supervised learning on one task results in a surprising amount of progress even on seemingly dissimilar tasks; progress on other tasks is larger if the training task has diverse classes; (3) the structure of the space of tasks indicated by our analysis is consistent with parts of the Wordnet phylogenetic tree; (4) episodic meta-learning algorithms and supervised learning traverse different trajectories during training but they fit similar models eventually; (5) contrastive and semi-supervised learning methods traverse trajectories similar to those of supervised learning. We use classification tasks constructed from the CIFAR-10 and Imagenet datasets to study these phenomena. Code is available at https://github.com/grasp-lyrl/picture_of_space_of_tasks.

----

## [1191] Policy Regularization with Dataset Constraint for Offline Reinforcement Learning

**Authors**: *Yuhang Ran, Yi-Chen Li, Fuxiang Zhang, Zongzhang Zhang, Yang Yu*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/ran23a.html](https://proceedings.mlr.press/v202/ran23a.html)

**Abstract**:

We consider the problem of learning the best possible policy from a fixed dataset, known as offline Reinforcement Learning (RL). A common taxonomy of existing offline RL works is policy regularization, which typically constrains the learned policy by distribution or support of the behavior policy. However, distribution and support constraints are overly conservative since they both force the policy to choose similar actions as the behavior policy when considering particular states. It will limit the learned policy’s performance, especially when the behavior policy is sub-optimal. In this paper, we find that regularizing the policy towards the nearest state-action pair can be more effective and thus propose Policy Regularization with Dataset Constraint (PRDC). When updating the policy in a given state, PRDC searches the entire dataset for the nearest state-action sample and then restricts the policy with the action of this sample. Unlike previous works, PRDC can guide the policy with proper behaviors from the dataset, allowing it to choose actions that do not appear in the dataset along with the given state. It is a softer constraint but still keeps enough conservatism from out-of-distribution actions. Empirical evidence and theoretical analysis show that PRDC can alleviate offline RL’s fundamentally challenging value overestimation issue with a bounded performance gap. Moreover, on a set of locomotion and navigation tasks, PRDC achieves state-of-the-art performance compared with existing methods. Code is available at https://github.com/LAMDA-RL/PRDC

----

## [1192] SpENCNN: Orchestrating Encoding and Sparsity for Fast Homomorphically Encrypted Neural Network Inference

**Authors**: *Ran Ran, Xinwei Luo, Wei Wang, Tao Liu, Gang Quan, Xiaolin Xu, Caiwen Ding, Wujie Wen*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/ran23b.html](https://proceedings.mlr.press/v202/ran23b.html)

**Abstract**:

Homomorphic Encryption (HE) is a promising technology to protect clients’ data privacy for Machine Learning as a Service (MLaaS) on public clouds. However, HE operations can be orders of magnitude slower than their counterparts for plaintexts and thus result in prohibitively high inference latency, seriously hindering the practicality of HE. In this paper, we propose a HE-based fast neural network (NN) inference framework–SpENCNN built upon the co-design of HE operation-aware model sparsity and the single-instruction-multiple-data (SIMD)-friendly data packing, to improve NN inference latency. In particular, we first develop an encryption-aware HE-group convolution technique that can partition channels among different groups based on the data size and ciphertext size, and then encode them into the same ciphertext by novel group-interleaved encoding, so as to dramatically reduce the number of bottlenecked operations in HE convolution. We further tailor a HE-friendly sub-block weight pruning to reduce the costly HE-based convolution operation. Our experiments show that SpENCNN can achieve overall speedups of 8.37$\times$, 12.11$\times$, 19.26$\times$, and 1.87$\times$ for LeNet, VGG-5, HEFNet, and ResNet-20 respectively, with negligible accuracy loss. Our code is publicly available at https://github.com/ranran0523/SPECNN.

----

## [1193] Feature learning in deep classifiers through Intermediate Neural Collapse

**Authors**: *Akshay Rangamani, Marius Lindegaard, Tomer Galanti, Tomaso A. Poggio*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/rangamani23a.html](https://proceedings.mlr.press/v202/rangamani23a.html)

**Abstract**:

In this paper, we conduct an empirical study of the feature learning process in deep classifiers. Recent research has identified a training phenomenon called Neural Collapse (NC), in which the top-layer feature embeddings of samples from the same class tend to concentrate around their means, and the top layer’s weights align with those features. Our study aims to investigate if these properties extend to intermediate layers. We empirically study the evolution of the covariance and mean of representations across different layers and show that as we move deeper into a trained neural network, the within-class covariance decreases relative to the between-class covariance. Additionally, we find that in the top layers, where the between-class covariance is dominant, the subspace spanned by the class means aligns with the subspace spanned by the most significant singular vector components of the weight matrix in the corresponding layer. Finally, we discuss the relationship between NC and Associative Memories (Willshaw et. al. 1969).

----

## [1194] The Unintended Consequences of Discount Regularization: Improving Regularization in Certainty Equivalence Reinforcement Learning

**Authors**: *Sarah Rathnam, Sonali Parbhoo, Weiwei Pan, Susan A. Murphy, Finale Doshi-Velez*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/rathnam23a.html](https://proceedings.mlr.press/v202/rathnam23a.html)

**Abstract**:

Discount regularization, using a shorter planning horizon when calculating the optimal policy, is a popular choice to restrict planning to a less complex set of policies when estimating an MDP from sparse or noisy data (Jiang et al., 2015). It is commonly understood that discount regularization functions by de-emphasizing or ignoring delayed effects. In this paper, we reveal an alternate view of discount regularization that exposes unintended consequences. We demonstrate that planning under a lower discount factor produces an identical optimal policy to planning using any prior on the transition matrix that has the same distribution for all states and actions. In fact, it functions like a prior with stronger regularization on state-action pairs with more transition data. This leads to poor performance when the transition matrix is estimated from data sets with uneven amounts of data across state-action pairs. Our equivalence theorem leads to an explicit formula to set regularization parameters locally for individual state-action pairs rather than globally. We demonstrate the failures of discount regularization and how we remedy them using our state-action-specific method across simple empirical examples as well as a medical cancer simulator.

----

## [1195] Beam Tree Recursive Cells

**Authors**: *Jishnu Ray Chowdhury, Cornelia Caragea*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/ray-chowdhury23a.html](https://proceedings.mlr.press/v202/ray-chowdhury23a.html)

**Abstract**:

We propose Beam Tree Recursive Cell (BT-Cell) - a backpropagation-friendly framework to extend Recursive Neural Networks (RvNNs) with beam search for latent structure induction. We further extend this framework by proposing a relaxation of the hard top-$k$ operators in beam search for better propagation of gradient signals. We evaluate our proposed models in different out-of-distribution splits in both synthetic and realistic data. Our experiments show that BT-Cell achieves near-perfect performance on several challenging structure-sensitive synthetic tasks like ListOps and logical inference while maintaining comparable performance in realistic data against other RvNN-based models. Additionally, we identify a previously unknown failure case for neural models in generalization to unseen number of arguments in ListOps. The code is available at: https://github.com/JRC1995/BeamTreeRecursiveCells.

----

## [1196] Monotonic Location Attention for Length Generalization

**Authors**: *Jishnu Ray Chowdhury, Cornelia Caragea*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/ray-chowdhury23b.html](https://proceedings.mlr.press/v202/ray-chowdhury23b.html)

**Abstract**:

We explore different ways to utilize position-based cross-attention in seq2seq networks to enable length generalization in algorithmic tasks. We show that a simple approach of interpolating the original and reversed encoded representations combined with relative attention allows near-perfect length generalization for both forward and reverse lookup tasks or copy tasks that had been generally hard to tackle. We also devise harder diagnostic tasks where the relative distance of the ideal attention position varies with timestep. In such settings, the simple interpolation trick with relative attention is not sufficient. We introduce novel variants of location attention building on top of Dubois et al. (2020) to address the new diagnostic tasks. We also show the benefits of our approaches for length generalization in SCAN (Lake & Baroni, 2018) and CFQ (Keysers et al.,2020). Our code is available on GitHub.

----

## [1197] Automated Search for Conjectures on Mathematical Constants using Analysis of Integer Sequences

**Authors**: *Ofir Razon, Yoav Harris, Shahar Gottlieb, Dan Carmon, Ofir David, Ido Kaminer*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/razon23a.html](https://proceedings.mlr.press/v202/razon23a.html)

**Abstract**:

The discovery of formulas involving mathematical constants such as $\pi$ and $e$ had a great impact on various fields of science and mathematics. However, such discoveries have remained scarce, relying on the intuition of mathematicians such as Ramanujan and Gauss. Recent efforts to automate such discoveries, such as the Ramanujan Machine project, relied solely on exhaustive search and remain limited by the space of options that can be covered. Here we propose a fundamentally different method to search for conjectures on mathematical constants: through analysis of integer sequences. We introduce the Enumerated Signed-continued-fraction Massey Approve (ESMA) algorithm, which builds on the Berlekamp-Massey algorithm to identify patterns in integer sequences that represent mathematical constants. ESMA has found various known formulas and new conjectures for $e, e^2, \tan(1)$, and ratios of values of Bessel functions, many of which provide faster numerical convergence than their corresponding simple continued fractions forms. We also characterize the space of constants that ESMA can catch and quantify its algorithmic advantage in certain scenarios. Altogether, this work continues the development toward algorithm-augmented mathematical intuition, to help accelerate mathematical research.

----

## [1198] Neural networks trained with SGD learn distributions of increasing complexity

**Authors**: *Maria Refinetti, Alessandro Ingrosso, Sebastian Goldt*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/refinetti23a.html](https://proceedings.mlr.press/v202/refinetti23a.html)

**Abstract**:

The uncanny ability of over-parameterised neural networks to generalise well has been explained using various "simplicity biases". These theories postulate that neural networks avoid overfitting by first fitting simple, linear classifiers before learning more complex, non-linear functions. Meanwhile, data structure is also recognised as a key ingredient for good generalisation, yet its role in simplicity biases is not yet understood. Here, we show that neural networks trained using stochastic gradient descent initially classify their inputs using lower-order input statistics, like mean and covariance, and exploit higher-order statistics only later during training. We first demonstrate this distributional simplicity bias (DSB) in a solvable model of a single neuron trained on synthetic data. We then demonstrate DSB empirically in a range of deep convolutional networks and visual transformers trained on CIFAR10, and show that it even holds in networks pre-trained on ImageNet. We discuss the relation of DSB to other simplicity biases and consider its implications for the principle of Gaussian universality in learning.

----

## [1199] Simplex Random Features

**Authors**: *Isaac Reid, Krzysztof Marcin Choromanski, Valerii Likhosherstov, Adrian Weller*

**Conference**: *icml 2023*

**URL**: [https://proceedings.mlr.press/v202/reid23a.html](https://proceedings.mlr.press/v202/reid23a.html)

**Abstract**:

We present Simplex Random Features (SimRFs), a new random feature (RF) mechanism for unbiased approximation of the softmax and Gaussian kernels by geometrical correlation of random projection vectors. We prove that SimRFs provide the smallest possible mean square error (MSE) on unbiased estimates of these kernels among the class of weight-independent geometrically-coupled positive random feature (PRF) mechanisms, substantially outperforming the previously most accurate Orthogonal Random Features (ORFs) at no observable extra cost. We present a more computationally expensive SimRFs+ variant, which we prove is asymptotically optimal in the broader family of weight-dependent geometrical coupling schemes (which permit correlations between random vector directions and norms). In extensive empirical studies, we show consistent gains provided by SimRFs in settings including pointwise kernel estimation, nonparametric classification and scalable Transformers.

----



[Go to the previous page](ICML-2023-list05.md)

[Go to the next page](ICML-2023-list07.md)

[Go to the catalog section](README.md)