## [0] Individual Arbitrariness and Group Fairness

**Authors**: *Carol Xuan Long, Hsiang Hsu, Wael Alghamdi, Flávio P. Calmon*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d891d240b5784656a0356bf4b00f5cdd-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d891d240b5784656a0356bf4b00f5cdd-Abstract-Conference.html)

**Abstract**:

Machine learning tasks may admit multiple competing models that achieve similar performance yet produce conflicting outputs for individual samples---a phenomenon known as predictive multiplicity. We demonstrate that fairness interventions in machine learning optimized solely for group fairness and accuracy can exacerbate predictive multiplicity. Consequently, state-of-the-art fairness interventions can mask high predictive multiplicity behind favorable group fairness and accuracy metrics. We argue that a third axis of ``arbitrariness'' should be considered  when deploying models to aid decision-making in applications of individual-level impact.To address this challenge, we propose an ensemble  algorithm applicable to any fairness intervention that provably ensures  more consistent predictions.

----

## [0] ASPEN: Breaking Operator Barriers for Efficient Parallelization of Deep Neural Networks

**Authors**: *Jongseok Park, Kyungmin Bin, Gibum Park, Sangtae Ha, Kyunghan Lee*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d899a31938c7838965b589d9b14a5ca6-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d899a31938c7838965b589d9b14a5ca6-Abstract-Conference.html)

**Abstract**:

Modern Deep Neural Network (DNN) frameworks use tensor operators as the main building blocks of DNNs. However, we observe that operator-based construction of DNNs incurs significant drawbacks in parallelism in the form of synchronization barriers. Synchronization barriers of operators confine the scope of parallel computation to each operator and obscure the rich parallel computation opportunities that exist across operators. To this end, we present ASPEN, a novel parallel computation solution for DNNs that achieves fine-grained dynamic execution of DNNs, which (1) removes the operator barriers and expresses DNNs in dataflow graphs of fine-grained tiles to expose the parallel computation opportunities across operators, and (2) exploits these opportunities by dynamically locating and scheduling them in runtime. This novel approach of ASPEN enables opportunistic parallelism, a new class of parallelism for DNNs that is unavailable in the existing operator-based approaches. ASPEN also achieves high resource utilization and memory reuse by letting each resource asynchronously traverse depthwise in the DNN graph to its full computing potential. We provide challenges and solutions to our approach and show that our proof-of-concept implementation of ASPEN on CPU shows exceptional performance, outperforming state-of-the-art inference systems of TorchScript and TVM by up to 3.2$\times$ and 4.3$\times$, respectively.

----

## [0] Parallel Submodular Function Minimization

**Authors**: *Deeparnab Chakrabarty, Andrei Graur, Haotian Jiang, Aaron Sidford*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d8a7f2f7e346410e8ac7b39d9ff28c4a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d8a7f2f7e346410e8ac7b39d9ff28c4a-Abstract-Conference.html)

**Abstract**:

We consider the parallel complexity of submodular function minimization (SFM).     We provide a pair of methods which obtain two new query versus depth trade-offs a submodular function defined on subsets of $n$ elements that has integer values between $-M$ and $M$. The first method has depth $2$ and query complexity $n^{O(M)}$ and the second method has depth $\widetilde{O}(n^{1/3} M^{2/3})$ and query complexity $O(\mathrm{poly}(n, M))$. Despite a line of work on improved parallel lower bounds for SFM, prior to our work the only known algorithms for parallel SFM either followed from more general methods for sequential SFM or highly-parallel minimization of convex $\ell_2$-Lipschitz functions. Interestingly, to obtain our second result we provide the first highly-parallel algorithm for minimizing $\ell_\infty$-Lipschitz function over the hypercube which obtains near-optimal depth for obtaining constant accuracy.

----

## [0] Emergent Communication for Rules Reasoning

**Authors**: *Yuxuan Guo, Yifan Hao, Rui Zhang, Enshuai Zhou, Zidong Du, Xishan Zhang, Xinkai Song, Yuanbo Wen, Yongwei Zhao, Xuehai Zhou, Jiaming Guo, Qi Yi, Shaohui Peng, Di Huang, Ruizhi Chen, Qi Guo, Yunji Chen*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d8ace30c68b085556ccce04ed4ae4ebb-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d8ace30c68b085556ccce04ed4ae4ebb-Abstract-Conference.html)

**Abstract**:

Research on emergent communication between deep-learning-based agents has received extensive attention due to its inspiration for linguistics and artificial intelligence.   However, previous attempts have hovered around emerging communication under perception-oriented environmental settings,  that forces agents to describe low-level perceptual features intra image or symbol contexts.  In this work, inspired by the classic human reasoning test (namely Raven's Progressive Matrix), we propose the Reasoning Game, a cognition-oriented environment that encourages agents to reason and communicate high-level rules, rather than perceived low-level contexts.  Moreover, we propose 1) an unbiased dataset (namely rule-RAVEN) as a benchmark to avoid overfitting, 2) and a two-stage curriculum agent training method as a baseline for more stable convergence in the Reasoning Game,  where contexts and semantics are bilaterally drifting.  Experimental results show that, in the Reasoning Game, a semantically stable and compositional language emerges to solve reasoning problems.  The emerged language helps agents apply the extracted rules to the generalization of unseen context attributes, and to the transfer between different context attributes or even tasks.

----

## [0] A Regularized Conditional GAN for Posterior Sampling in Image Recovery Problems

**Authors**: *Matthew Bendel, Rizwan Ahmad, Philip Schniter*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d8b29f07599fecdba93d87ed27a65524-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d8b29f07599fecdba93d87ed27a65524-Abstract-Conference.html)

**Abstract**:

In image recovery problems, one seeks to infer an image from distorted, incomplete, and/or noise-corrupted measurements.Such problems arise in magnetic resonance imaging (MRI), computed tomography, deblurring, super-resolution, inpainting, phase retrieval, image-to-image translation, and other applications. Given a training set of signal/measurement pairs, we seek to do more than just produce one good image estimate. Rather, we aim to rapidly and accurately sample from the posterior distribution. To do this,we propose a regularized conditional Wasserstein GAN that generates dozens of high-quality posterior samples per second. Our regularization comprises an $\ell_1$ penalty and an adaptively weighted standard-deviation reward. Using quantitative evaluation metrics like conditional Fr√©chet inception distance, we demonstrate that our method produces state-of-the-art posterior samples in both multicoil MRI and large-scale inpainting applications. The code for our model can be found here: https://github.com/matt-bendel/rcGAN.

----

## [0] Would I have gotten that reward? Long-term credit assignment by counterfactual contribution analysis

**Authors**: *Alexander Meulemans, Simon Schug, Seijin Kobayashi, Nathaniel Daw, Gregory Wayne*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d8bd445c2abe1343cce0e14b361b2fb3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d8bd445c2abe1343cce0e14b361b2fb3-Abstract-Conference.html)

**Abstract**:

To make reinforcement learning more sample efficient, we need better credit assignment methods that measure an action’s influence on future rewards. Building upon Hindsight Credit Assignment (HCA), we introduce Counterfactual Contribution Analysis (COCOA), a new family of model-based credit assignment algorithms. Our algorithms achieve precise credit assignment by measuring the contribution of actions upon obtaining subsequent rewards, by quantifying a counterfactual query: ‘Would the agent still have reached this reward if it had taken another action?’. We show that measuring contributions w.r.t. rewarding states, as is done in HCA, results in spurious estimates of contributions, causing HCA to degrade towards the high-variance REINFORCE estimator in many relevant environments. Instead, we measure contributions w.r.t. rewards or learned representations of the rewarding objects, resulting in gradient estimates with lower variance. We run experiments on a suite of problems specifically designed to evaluate long-term credit assignment capabilities. By using dynamic programming, we measure ground-truth policy gradients and show that the improved performance of our new model-based credit assignment methods is due to lower bias and variance compared to HCA and common baselines. Our results demonstrate how modeling action contributions towards rewarding outcomes can be leveraged for credit assignment, opening a new path towards sample-efficient reinforcement learning.

----

## [0] HOH: Markerless Multimodal Human-Object-Human Handover Dataset with Large Object Count

**Authors**: *Noah Wiederhold, Ava Megyeri, DiMaggio Paris, Sean Banerjee, Natasha Banerjee*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d8c6a37c4c94e9a63e53d296f1f668ae-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/d8c6a37c4c94e9a63e53d296f1f668ae-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

We present the HOH (Human-Object-Human) Handover Dataset, a large object count dataset with 136 objects, to accelerate data-driven research on handover studies, human-robot handover implementation, and artificial intelligence (AI) on handover parameter estimation from 2D and 3D data of two-person interactions. HOH contains multi-view RGB and depth data, skeletons, fused point clouds, grasp type and handedness labels, object, giver hand, and receiver hand 2D and 3D segmentations, giver and receiver comfort ratings, and paired object metadata and aligned 3D models for 2,720 handover interactions spanning 136 objects and 20 giver-receiver pairs—40 with role-reversal—organized from 40 participants. We also show experimental results of neural networks trained using HOH to perform grasp, orientation, and trajectory prediction. As the only fully markerless handover capture dataset, HOH represents natural human-human handover interactions, overcoming challenges with markered datasets that require specific suiting for body tracking, and lack high-resolution hand tracking. To date, HOH is the largest handover dataset in terms of object count, participant count, pairs with role reversal accounted for, and total interactions captured.

----

## [0] Tight Risk Bounds for Gradient Descent on Separable Data

**Authors**: *Matan Schliserman, Tomer Koren*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d8ca28a32c05cd3b9b0940e43720f31b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d8ca28a32c05cd3b9b0940e43720f31b-Abstract-Conference.html)

**Abstract**:

We study the generalization properties of unregularized gradient methods applied to separable linear classification---a setting that has received considerable attention since the pioneering work of Soudry et al. (2018).We establish tight upper and lower (population) risk bounds for gradient descent in this setting, for any smooth loss function, expressed in terms of its tail decay rate.Our bounds take the form $\Theta(r_{\ell,T}^2 / \gamma^2 T + r_{\ell,T}^2 / \gamma^2 n)$, where $T$ is the number of gradient steps, $n$ is size of the training set, $\gamma$ is the data margin, and $r_{\ell,T}$ is a complexity term that depends on the tail decay rate of the loss function (and on $T$).Our upper bound greatly improves the existing risk bounds due to Shamir (2021) and Schliserman and Koren (2022), that either applied to specific loss functions or imposed extraneous technical assumptions, and applies to virtually any convex and smooth loss function.Our risk lower bound is the first in this context and establish the tightness of our general upper bound for any given tail decay rate and in all parameter regimes.The proof technique used to show these results is also markedly simpler compared to previous work, and is straightforward to extend to other gradient methods; we illustrate this by providing analogous results for Stochastic Gradient Descent.

----

## [0] Video Prediction Models as Rewards for Reinforcement Learning

**Authors**: *Alejandro Escontrela, Ademi Adeniji, Wilson Yan, Ajay Jain, Xue Bin Peng, Ken Goldberg, Youngwoon Lee, Danijar Hafner, Pieter Abbeel*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d9042abf40782fbce28901c1c9c0e8d8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d9042abf40782fbce28901c1c9c0e8d8-Abstract-Conference.html)

**Abstract**:

Specifying reward signals that allow agents to learn complex behaviors is a long-standing challenge in reinforcement learning.A promising approach is to extract preferences for behaviors from unlabeled videos, which are widely available on the internet. We present Video Prediction Rewards (VIPER), an algorithm that leverages pretrained video prediction models as action-free reward signals for reinforcement learning. Specifically, we first train an autoregressive transformer on expert videos and then use the video prediction likelihoods as reward signals for a reinforcement learning agent. VIPER enables expert-level control without programmatic task rewards across a wide range of DMC, Atari, and RLBench tasks. Moreover, generalization of the video prediction model allows us to derive rewards for an out-of-distribution environment where no expert data is available, enabling cross-embodiment generalization for tabletop manipulation. We see our work as starting point for scalable reward specification from unlabeled videos that will benefit from the rapid advances in generative modeling. Source code and datasets are available on the project website: https://ViperRL.com

----

## [0] Provably (More) Sample-Efficient Offline RL with Options

**Authors**: *Xiaoyan Hu, Ho-fung Leung*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d91b532a76ea98ac1ef5226b862bfc49-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d91b532a76ea98ac1ef5226b862bfc49-Abstract-Conference.html)

**Abstract**:

The options framework yields empirical success in long-horizon planning problems of reinforcement learning (RL). Recent works show that options help improve the sample efficiency in online RL. However, these results are no longer applicable to scenarios where exploring the environment online is risky, e.g., automated driving and healthcare. In this paper, we provide the first analysis of the sample complexity for offline RL with options, where the agent learns from a dataset without further interaction with the environment. We derive a novel information-theoretic lower bound, which generalizes the one for offline learning with actions. We propose the PEssimistic Value Iteration for Learning with Options (PEVIO) algorithm and establish near-optimal suboptimality bounds for two popular data-collection procedures, where the first one collects state-option transitions and the second one collects state-action transitions. We show that compared to offline RL with actions, using options not only enjoys a faster finite-time convergence rate (to the optimal value) but also attains a better performance when either the options are carefully designed or the offline data is limited. Based on these results, we analyze the pros and cons of the data-collection procedures.

----

## [0] Rewrite Caption Semantics: Bridging Semantic Gaps for Language-Supervised Semantic Segmentation

**Authors**: *Yun Xing, Jian Kang, Aoran Xiao, Jiahao Nie, Ling Shao, Shijian Lu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d937cb3fe2851ed0ab9af5e38f885077-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d937cb3fe2851ed0ab9af5e38f885077-Abstract-Conference.html)

**Abstract**:

Vision-Language Pre-training has demonstrated its remarkable zero-shot recognition ability and potential to learn generalizable visual representations from languagesupervision. Taking a step ahead, language-supervised semantic segmentation enables spatial localization of textual inputs by learning pixel grouping solely from image-text pairs. Nevertheless, the state-of-the-art suffers from a clear semantic gap between visual and textual modalities: plenty of visual concepts appeared in images are missing in their paired captions. Such semantic misalignment circulates in pre-training, leading to inferior zero-shot performance in dense predictions due to insufficient visual concepts captured in textual representations. To close such semantic gap, we propose Concept Curation (CoCu), a pipeline that leverages CLIP to compensate for the missing semantics. For each image-text pair, we establish a concept archive that maintains potential visually-matched concepts with our proposed vision-driven expansion and text-to-vision-guided ranking. Relevant concepts can thus be identified via cluster-guided sampling and fed into pre-training, thereby bridging the gap between visual and textual semantics. Extensive experiments over a broad suite of 8 segmentation benchmarks show that CoCu achieves superb zero-shot transfer performance and greatly boosts language-supervised segmentation baseline by a large margin, suggesting the value of closing semantic gap in pre-training data.

----

## [0] Approximate Allocation Matching for Structural Causal Bandits with Unobserved Confounders

**Authors**: *Lai Wei, Muhammad Qasim Elahi, Mahsa Ghasemi, Murat Kocaoglu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d938b739ac250e22729cc26e6176f65e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d938b739ac250e22729cc26e6176f65e-Abstract-Conference.html)

**Abstract**:

Structural causal bandit provides a framework for online decision-making problems when causal information is available. It models the stochastic environment with a structural causal model (SCM) that governs the causal relations between random variables. In each round, an agent applies an intervention (or no intervention) by setting certain variables to some constants and receives a stochastic reward from a non-manipulable variable. Though the causal structure is given, the observational and interventional distributions of these random variables are unknown beforehand, and they can only be learned through interactions with the environment. Therefore, to maximize the expected cumulative reward, it is critical to balance the explore-versus-exploit tradeoff. We assume each random variable takes a finite number of distinct values, and consider a semi-Markovian setting, where random variables are affected by unobserved confounders. Using the canonical SCM formulation to discretize the domains of unobserved variables, we efficiently integrate samples to reduce model uncertainty. This gives the decision maker a natural advantage over those in a classical multi-armed bandit setup. We provide a logarithmic asymptotic regret lower bound for the structural causal bandit problem. Inspired by the lower bound, we design an algorithm that can utilize the causal structure to accelerate the learning process and take informative and rewarding interventions. We establish that our algorithm achieves a logarithmic regret and demonstrate that it outperforms the existing methods via simulations.

----

## [0] Human-Guided Complexity-Controlled Abstractions

**Authors**: *Andi Peng, Mycal Tucker, Eoin M. Kenny, Noga Zaslavsky, Pulkit Agrawal, Julie A. Shah*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d94b46ec30adee2bbb134f813fc9dde0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d94b46ec30adee2bbb134f813fc9dde0-Abstract-Conference.html)

**Abstract**:

Neural networks often learn task-specific latent representations that fail to generalize to novel settings or tasks. Conversely, humans learn discrete representations (i.e., concepts or words) at a variety of abstraction levels (e.g., "bird" vs. "sparrow'") and use the appropriate abstraction based on tasks. Inspired by this, we train neural models to generate a spectrum of discrete representations, and control the complexity of the representations (roughly, how many bits are allocated for encoding inputs) by tuning the entropy of the distribution over representations. In finetuning experiments, using only a small number of labeled examples for a new task, we show that (1) tuning the representation to a task-appropriate complexity level supports the greatest finetuning performance, and (2) in a human-participant study, users were able to identify the appropriate complexity level for a downstream task via visualizations of discrete representations. Our results indicate a promising direction for rapid model finetuning by leveraging human insight.

----

## [0] Scenario Diffusion: Controllable Driving Scenario Generation With Diffusion

**Authors**: *Ethan Pronovost, Meghana Reddy Ganesina, Noureldin Hendy, Zeyu Wang, Andres Morales, Kai Wang, Nick Roy*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d95cb79a3421e6d9b6c9a9008c4d07c5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d95cb79a3421e6d9b6c9a9008c4d07c5-Abstract-Conference.html)

**Abstract**:

Automated creation of synthetic traffic scenarios is a key part of scaling the safety validation of autonomous vehicles (AVs). In this paper, we propose Scenario Diffusion, a novel diffusion-based architecture for generating traffic scenarios that enables controllable scenario generation. We combine latent diffusion, object detection and trajectory regression to generate distributions of synthetic agent poses, orientations and trajectories simultaneously. This distribution is conditioned on the map and sets of tokens describing the desired scenario to provide additional control over the generated scenario. We show that our approach has sufficient expressive capacity to model diverse traffic patterns and generalizes to different geographical regions.

----

## [0] Label-Only Model Inversion Attacks via Knowledge Transfer

**Authors**: *Ngoc-Bao Nguyen, Keshigeyan Chandrasegaran, Milad Abdollahzadeh, Ngai-Man Cheung*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d9827e811c5a205c1313fb950c072c7d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d9827e811c5a205c1313fb950c072c7d-Abstract-Conference.html)

**Abstract**:

In a model inversion (MI) attack, an adversary abuses access to a machine learning (ML) model to infer and reconstruct private training data. Remarkable progress has been made in the white-box and black-box setups, where the adversary has access to the complete model or the model's soft output respectively. However, there is very limited study in the most challenging but practically important setup: Label-only MI attacks, where the adversary only has access to the model's predicted  label (hard label) without confidence scores nor any other model information.  In this work, we propose LOKT, a novel approach for label-only MI attacks. Our idea is based on transfer of knowledge from the opaque target model to  surrogate models. Subsequently, using these surrogate models, our approach can harness advanced white-box attacks. We propose knowledge transfer based on generative modelling, and introduce a new model, Target model-assisted ACGAN (T-ACGAN), for effective knowledge transfer. Our method casts the challenging label-only MI into the more tractable white-box setup. We provide analysis to support that surrogate models based on our approach serve as effective proxies for the target model for MI. Our experiments show that our method significantly outperforms existing SOTA Label-only MI attack by more than 15% across all MI benchmarks. Furthermore, our method compares favorably in terms of query budget. Our study highlights rising privacy threats for  ML models even when minimal information (i.e.,  hard labels) is exposed. Our study highlights rising privacy threats for  ML models even when minimal information (i.e.,  hard labels) is exposed. Our code, demo, models and reconstructed data are available at our project page:https://ngoc-nguyen-0.github.io/lokt/

----

## [0] On the Adversarial Robustness of Out-of-distribution Generalization Models

**Authors**: *Xin Zou, Weiwei Liu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d9888cc7baa04c2e44e8115588133515-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d9888cc7baa04c2e44e8115588133515-Abstract-Conference.html)

**Abstract**:

Out-of-distribution (OOD) generalization has attracted increasing research attention in recent years, due to its promising experimental results in real-world applications. Interestingly, we find that existing OOD generalization methods are vulnerable to adversarial attacks. This motivates us to study OOD adversarial robustness. We first present theoretical analyses of OOD adversarial robustness in two different complementary settings. Motivated by the theoretical results, we design two algorithms to improve the OOD adversarial robustness. Finally, we conduct experiments to validate the effectiveness of our proposed algorithms.

----

## [0] Utilitarian Algorithm Configuration

**Authors**: *Devon R. Graham, Kevin Leyton-Brown, Tim Roughgarden*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d98d9cef0c189f1db95f1d94652f7051-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d98d9cef0c189f1db95f1d94652f7051-Abstract-Conference.html)

**Abstract**:

We present the first nontrivial procedure for configuring heuristic algorithms to maximize the utility provided to their end users while also offering theoretical guarantees about performance. Existing procedures seek configurations that minimize expected runtime. However, very recent theoretical work argues that expected runtime minimization fails to capture algorithm designers' preferences. Here we show that the utilitarian objective also confers significant algorithmic benefits. Intuitively, this is because mean runtime is dominated by extremely long runs even when they are incredibly rare; indeed, even when an algorithm never gives rise to such long runs, configuration procedures that provably minimize mean runtime must perform a huge number of experiments to demonstrate this fact. In contrast, utility is bounded and monotonically decreasing in runtime, allowing for meaningful empirical bounds on a configuration's performance. This paper builds on this idea to describe effective and theoretically sound configuration procedures. We prove upper bounds on the runtime of these procedures that are similar to theoretical lower bounds, while also demonstrating their performance empirically.

----

## [0] Double Randomized Underdamped Langevin with Dimension-Independent Convergence Guarantee

**Authors**: *Yuanshi Liu, Cong Fang, Tong Zhang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d9af4d6ac714626b652da5616ca71f99-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d9af4d6ac714626b652da5616ca71f99-Abstract-Conference.html)

**Abstract**:

This paper focuses on the high-dimensional sampling of log-concave distributions with composite structures: $p^*(\mathrm{d}x)\propto \exp(-g(x)-f(x))\mathrm{d}x$. We develop a double randomization technique, which leads to a fast underdamped Langevin algorithm with a dimension-independent convergence guarantee. We prove that the algorithm enjoys an overall $\tilde{\mathcal{O}}\left(\frac{\left(\mathrm{tr}(H)\right)^{1/3}}{\epsilon^{2/3}}\right)$ iteration complexity to reach an $\epsilon$-tolerated sample whose distribution $p$ admits $W_2(p,p^*)\leq \epsilon$.  Here,  $H$ is an upper bound of the Hessian matrices for $f$ and does not explicitly depend on dimension $d$. For the posterior sampling over linear models with normalized data, we show a clear superiority of convergence rate which is dimension-free and outperforms the previous best-known results by a $d^{1/3}$ factor. The analysis to achieve a faster convergence rate brings new insights into high-dimensional sampling.

----

## [0] Non-Asymptotic Analysis of a UCB-based Top Two Algorithm

**Authors**: *Marc Jourdan, Rémy Degenne*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d9b564716709357b4bccec9fc9ad04d2-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d9b564716709357b4bccec9fc9ad04d2-Abstract-Conference.html)

**Abstract**:

A Top Two sampling rule for bandit identification is a method which selects the next arm to sample from among two candidate arms, a leader and a challenger. Due to their simplicity and good empirical performance, they have received increased attention in recent years. However, for fixed-confidence best arm identification, theoretical guarantees for Top Two methods have only been obtained in the asymptotic regime, when the error level vanishes. In this paper, we derive the first non-asymptotic upper bound on the expected sample complexity of a Top Two algorithm, which holds for any error level. Our analysis highlights sufficient properties for a regret minimization algorithm to be used as leader. These properties are satisfied by the UCB algorithm, and our proposed UCB-based Top Two algorithm simultaneously enjoys non-asymptotic guarantees and competitive empirical performance.

----

## [0] Statistical and Computational Trade-off in Multi-Agent Multi-Armed Bandits

**Authors**: *Filippo Vannella, Alexandre Proutière, Jaeseong Jeong*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d9c7c8bd6ad4cebb7d006e5109e0b682-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d9c7c8bd6ad4cebb7d006e5109e0b682-Abstract-Conference.html)

**Abstract**:

We study the problem of regret minimization in Multi-Agent Multi-Armed Bandits (MAMABs) where the rewards are defined through a factor graph. We derive an instance-specific regret lower bound and characterize the minimal expected number of times each global action should be explored. Unfortunately, this bound and the corresponding optimal exploration process are obtained by solving a combinatorial optimization problem with a set of variables and constraints exponentially growing with the number of agents. We approximate the regret lower bound problem via Mean Field techniques to reduce the number of variables and constraints. By tuning the latter, we explore the trade-off between achievable regret and complexity. We devise Efficient Sampling for MAMAB (ESM), an algorithm whose regret asymptotically matches the corresponding approximated lower bound. We assess the regret and computational complexity of ESM numerically, using both synthetic and real-world experiments in radio communications networks.

----

## [0] On permutation symmetries in Bayesian neural network posteriors: a variational perspective

**Authors**: *Simone Rossi, Ankit Singh, Thomas Hannagan*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d9dc5573f7368201d6409e07e882aa77-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d9dc5573f7368201d6409e07e882aa77-Abstract-Conference.html)

**Abstract**:

The elusive nature of gradient-based optimization in neural networks is tied to their loss landscape geometry, which is poorly understood. However recent work has brought solid evidence that there is essentially no loss barrier between the local solutions of gradient descent, once accounting for weight-permutations that leave the network's computation unchanged. This raises questions for approximate inference in Bayesian neural networks (BNNs), where we are interested in marginalizing over multiple points in the loss landscape.In this work, we first extend the formalism of marginalized loss barrier and solution interpolation to BNNs, before proposing a matching algorithm to search for linearly connected solutions. This is achieved by aligning the distributions of two independent approximate Bayesian solutions with respect to permutation matrices. Building on the work of Ainsworth et al. (2023), we frame the problem as a combinatorial optimization one, using an approximation to the sum of bilinear assignment problem. We then experiment on a variety of architectures and datasets, finding nearly zero marginalized loss barriers for linearly connected solutions.

----

## [0] Hierarchical Decomposition of Prompt-Based Continual Learning: Rethinking Obscured Sub-optimality

**Authors**: *Liyuan Wang, Jingyi Xie, Xingxing Zhang, Mingyi Huang, Hang Su, Jun Zhu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d9f8b5abc8e0926539ecbb492af7b2f1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d9f8b5abc8e0926539ecbb492af7b2f1-Abstract-Conference.html)

**Abstract**:

Prompt-based continual learning is an emerging direction in leveraging pre-trained knowledge for downstream continual learning, and has almost reached the performance pinnacle under supervised pre-training. However, our empirical research reveals that the current strategies fall short of their full potential under the more realistic self-supervised pre-training, which is essential for handling vast quantities of unlabeled data in practice. This is largely due to the difficulty of task-specific knowledge being incorporated into instructed representations via prompt parameters and predicted by uninstructed representations at test time. To overcome the exposed sub-optimality, we conduct a theoretical analysis of the continual learning objective in the context of pre-training, and decompose it into hierarchical components: within-task prediction, task-identity inference, and task-adaptive prediction. Following these empirical and theoretical insights, we propose Hierarchical Decomposition (HiDe-)Prompt, an innovative approach that explicitly optimizes the hierarchical components with an ensemble of task-specific prompts and statistics of both uninstructed and instructed representations, further with the coordination of a contrastive regularization strategy. Our extensive experiments demonstrate the superior performance of HiDe-Prompt and its robustness to pre-training paradigms in continual learning (e.g., up to 15.01% and 9.61% lead on Split CIFAR-100 and Split ImageNet-R, respectively).

----

## [0] 3D molecule generation by denoising voxel grids

**Authors**: *Pedro O. Pinheiro, Joshua Rackers, Joseph Kleinhenz, Michael Maser, Omar Mahmood, Andrew M. Watkins, Stephen Ra, Vishnu Sresht, Saeed Saremi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/da1131a86ac3c70e0b7cae89c3d4df22-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/da1131a86ac3c70e0b7cae89c3d4df22-Abstract-Conference.html)

**Abstract**:

We propose a new score-based approach to generate 3D molecules represented as atomic densities on regular grids.First, we train a denoising neural network that learns to map from a smooth distribution of noisy molecules to the distribution of real molecules.Then, we follow the neural empirical Bayes framework [Saremi and Hyvarinen, 2019] and generate molecules in two steps: (i) sample noisy density grids from a smooth distribution via underdamped Langevin Markov chain Monte Carlo, and (ii) recover the "clean" molecule by denoising the noisy grid with a single step.Our method, VoxMol, generates molecules in a fundamentally different way than the current state of the art (ie, diffusion models applied to atom point clouds). It differs in terms of the data representation, the noise model, the network architecture and the generative modeling algorithm.Our experiments show that VoxMol captures the distribution of drug-like molecules better than state of the art, while being faster to generate samples.

----

## [0] Accessing Higher Dimensions for Unsupervised Word Translation

**Authors**: *Sida Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/da31f4275972a58406b95c277ce7bc8d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/da31f4275972a58406b95c277ce7bc8d-Abstract-Conference.html)

**Abstract**:

The striking ability of unsupervised word translation has been demonstrated recently with the help of low-dimensional word vectors / pretraining, which is used by all successful methods and assumed to be necessary. We test and challenge this assumption by developing a method that can also make use of high dimensional signal. Freed from the limits of low dimensions, we show that relying on low-dimensional vectors and their incidental properties miss out on better denoising methods and signals in high dimensions, thus stunting the potential of the data. Our results show that unsupervised translation can be achieved more easily and robustly than previously thought -- less than 80MB and minutes of CPU time is required to achieve over 50\% accuracy for English to Finnish, Hungarian, and Chinese translations when trained in the same domain; even under domain mismatch, the method still works fully unsupervised on English NewsCrawl to Chinese Wikipedia and English Europarl to Spanish Wikipedia, among others. These results challenge prevailing assumptions on the necessity and superiority of low-dimensional vectors and show that the higher dimension signal can be used rather than thrown away.

----

## [0] Inverse Reinforcement Learning with the Average Reward Criterion

**Authors**: *Feiyang Wu, Jingyang Ke, Anqi Wu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/da409884a933ecbc4af03338111bf6aa-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/da409884a933ecbc4af03338111bf6aa-Abstract-Conference.html)

**Abstract**:

We study the problem of Inverse Reinforcement Learning (IRL) with an average-reward criterion. The goal is to recover an unknown policy and a reward function when the agent only has samples of states and actions from an experienced agent. Previous IRL methods assume that the expert is trained in a discounted environment, and the discount factor is known. This work alleviates this assumption by proposing an average-reward framework with efficient learning algorithms. We develop novel stochastic first-order methods to solve the IRL problem under the average-reward setting, which requires solving an Average-reward Markov Decision Process (AMDP) as a subproblem. To solve the subproblem, we develop a Stochastic Policy Mirror Descent (SPMD) method under general state and action spaces that needs $\mathcal{O}(1/\varepsilon)$ steps of gradient computation. Equipped with SPMD, we propose the Inverse Policy Mirror Descent (IPMD) method for solving the IRL problem with a $\mathcal{O}(1/\varepsilon^2)$ complexity. To the best of our knowledge, the aforementioned complexity results are new in IRL with the average reward criterion. Finally, we corroborate our analysis with numerical experiments using the MuJoCo benchmark and additional control tasks.

----

## [0] DisDiff: Unsupervised Disentanglement of Diffusion Probabilistic Models

**Authors**: *Tao Yang, Yuwang Wang, Yan Lu, Nanning Zheng*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/da47bfaf3f3a8d5bbab0d60c5195dc18-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/da47bfaf3f3a8d5bbab0d60c5195dc18-Abstract-Conference.html)

**Abstract**:

Targeting to understand the underlying explainable factors behind observations and modeling the conditional generation process on these factors, we connect disentangled representation learning to diffusion probabilistic models (DPMs) to take advantage of the remarkable modeling ability of DPMs. We propose a new task, disentanglement of (DPMs): given a pre-trained DPM, without any annotations of the factors, the task is to automatically discover the inherent factors behind the observations and disentangle the gradient fields of DPM into sub-gradient fields, each conditioned on the representation of each discovered factor. With disentangled DPMs, those inherent factors can be automatically discovered, explicitly represented and clearly injected into the diffusion process via the sub-gradient fields. To tackle this task, we devise an unsupervised approach, named DisDiff, and for the first time achieving disentangled representation learning in the framework of DPMs. Extensive experiments on synthetic and real-world datasets demonstrate the effectiveness of DisDiff.

----

## [0] Information-guided Planning: An Online Approach for Partially Observable Problems

**Authors**: *Matheus Aparecido do Carmo Alves, Amokh Varma, Yehia Elkhatib, Leandro Soriano Marcolino*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/da5498f88193ff61f0daea1940b819da-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/da5498f88193ff61f0daea1940b819da-Abstract-Conference.html)

**Abstract**:

This paper presents IB-POMCP, a novel algorithm for online planning under partial observability. Our approach enhances the decision-making process by using estimations of the world belief's entropy to guide a tree search process and surpass the limitations of planning in scenarios with sparse reward configurations. By performing what we denominate as an information-guided planning process, the algorithm, which incorporates a novel I-UCB function, shows significant improvements in reward and reasoning time compared to state-of-the-art baselines in several benchmark scenarios, along with theoretical convergence guarantees.

----

## [0] Bayesian Metric Learning for Uncertainty Quantification in Image Retrieval

**Authors**: *Frederik Warburg, Marco Miani, Silas Brack, Søren Hauberg*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/da7ce04b3683b173691ecbb801f2690f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/da7ce04b3683b173691ecbb801f2690f-Abstract-Conference.html)

**Abstract**:

We propose a Bayesian encoder for metric learning. Rather than relying on neural amortization as done in prior works, we learn a distribution over the network weights with the Laplace Approximation. We first prove that the contrastive loss is a negative log-likelihood on the spherical space. We propose three methods that ensure a positive definite covariance matrix. Lastly, we present a novel decomposition of the Generalized Gauss-Newton approximation. Empirically, we show that our Laplacian Metric Learner (LAM) yields well-calibrated uncertainties, reliably detects out-of-distribution examples, and has state-of-the-art predictive performance.

----

## [0] Neural Modulation for Flash Memory: An Unsupervised Learning Framework for Improved Reliability

**Authors**: *Jonathan Zedaka, Elisha Halperin, Evgeny Blaichman, Amit Berman*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/da7e0d7210b99ebc91c4a5f911962d6c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/da7e0d7210b99ebc91c4a5f911962d6c-Abstract-Conference.html)

**Abstract**:

Recent years have witnessed a significant increase in the storage density of NAND flash memory, making it a critical component in modern electronic devices. However, with the rise in storage capacity comes an increased likelihood of errors in data storage and retrieval. The growing number of errors poses ongoing challenges for system designers and engineers, in terms of the characterization, modeling, and optimization of NAND-based systems. We present a novel approach for modeling and preventing errors by utilizing the capabilities of generative and unsupervised machine learning methods. As part of our research, we constructed and trained a neural modulator that translates information bits into programming operations on each memory cell in NAND devices. Our modulator, tailored explicitly for flash memory channels, provides a smart writing scheme that reduces programming errors as well as compensates for data degradation over time. Specifically, the modulator is based on an auto-encoder architecture with an additional channel model embedded between the encoder and the decoder. A conditional generative adversarial network (cGAN) was used to construct the channel model. Optimized for the end-of-life work-point, the learned memory system outperforms the prior art by up to 56\% in raw bit error rate (RBER) and extends the lifetime of the flash memory block by up to 25\%.

----

## [0] Privacy Amplification via Compression: Achieving the Optimal Privacy-Accuracy-Communication Trade-off in Distributed Mean Estimation

**Authors**: *Wei-Ning Chen, Dan Song, Ayfer Özgür, Peter Kairouz*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/da8860a2fe8ddb7589136853bcc313fc-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/da8860a2fe8ddb7589136853bcc313fc-Abstract-Conference.html)

**Abstract**:

Privacy and communication constraints are two major bottlenecks in federated learning (FL) and analytics (FA). We study the optimal accuracy of mean and frequency estimation (canonical models for FL and FA respectively) under joint communication and $(\varepsilon, \delta)$-differential privacy (DP) constraints. We consider both the central and the multi-message shuffled DP models. We show that in order to achieve the optimal $\ell_2$ error under $(\varepsilon, \delta)$-DP, it is sufficient for each client to send $\Theta\left( n \min\left(\varepsilon, \varepsilon^2\right)\right)$ bits for FL %{\color{blue}(assuming the dimension $d \gg n \min\left(\varepsilon, \varepsilon^2\right)$)} and $\Theta\left(\log\left( n\min\left(\varepsilon, \varepsilon^2\right) \right)\right)$ bits for FA to the server, where $n$ is the number of participating clients.  Without compression, each client needs $O(d)$ bits and $O\left(\log d\right)$ bits for the mean and frequency estimation problems respectively (where $d$ corresponds to the number of trainable parameters in FL or the domain size in FA), meaning that we can get significant savings in the regime $ n \min\left(\varepsilon, \varepsilon^2\right)  = o(d)$, which is often the relevant regime in practice. We propose two different ways to leverage compression for privacy amplification and achieve the optimal privacy-communication-accuracy trade-offs. In both cases, each client communicates only partial information about its sample and we show that privacy is amplified by randomly selecting the part contributed by each client. In the first method, the random selection is revealed to the server, which results in a central DP guarantee with optimal privacy-communication-accuracy trade-offs.  In the second method, the random data parts from the clients are  shuffled by a secure shuffler resulting in a multi-message shuffling scheme with the same optimal trade-offs. As a result, we establish the optimal three-way trade-offs between privacy, communication, and accuracy for both the central DP and multi-message shuffling frameworks.

----

## [0] Normalization Layers Are All That Sharpness-Aware Minimization Needs

**Authors**: *Maximilian Müller, Tiffany Vlaar, David Rolnick, Matthias Hein*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/da909fc3893d272f26fd9db82e09d954-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/da909fc3893d272f26fd9db82e09d954-Abstract-Conference.html)

**Abstract**:

Sharpness-aware minimization (SAM) was proposed to reduce sharpness of minima and has been shown to enhance generalization performance in various settings. In this work we show that perturbing only the affine normalization parameters (typically comprising 0.1% of the total parameters) in the adversarial step of SAM can outperform perturbing all of the parameters. This finding generalizesto different SAM variants and both ResNet (Batch Normalization) and Vision Transformer (Layer Normalization) architectures. We consider alternative sparse perturbation approaches and find that these do not achieve similar performance enhancement at such extreme sparsity levels, showing that this behaviour is unique to the normalization layers. Although our findings reaffirm the effectivenessof SAM in improving generalization performance, they cast doubt on whether this is solely caused by reduced sharpness.

----

## [0] Robust Bayesian Satisficing

**Authors**: *Artun Saday, Yasar Cahit Yildirim, Cem Tekin*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/daa098aa8e1fc718943ff1ab7b5b30c9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/daa098aa8e1fc718943ff1ab7b5b30c9-Abstract-Conference.html)

**Abstract**:

Distributional shifts pose a significant challenge to achieving robustness in contemporary machine learning. To overcome this challenge, robust satisficing (RS) seeks a robust solution to an unspecified distributional shift while achieving a utility above a desired threshold. This paper focuses on the problem of RS in contextual Bayesian optimization when there is a discrepancy between the true and reference distributions of the context. We propose a novel robust Bayesian satisficing algorithm called RoBOS for noisy black-box optimization. Our algorithm guarantees sublinear lenient regret under certain assumptions on the amount of distribution shift. In addition, we define a weaker notion of regret called robust satisficing regret, in which our algorithm achieves a sublinear upper bound independent of the amount of distribution shift. To demonstrate the effectiveness of our method, we apply it to various learning problems and compare it to other approaches, such as distributionally robust optimization.

----

## [0] Neural Ideal Large Eddy Simulation: Modeling Turbulence with Neural Stochastic Differential Equations

**Authors**: *Anudhyan Boral, Zhong Yi Wan, Leonardo Zepeda-Núñez, James Lottes, Qing Wang, Yi-Fan Chen, John Anderson, Fei Sha*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/dabaded617b3be96c3ed161498a7d71c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/dabaded617b3be96c3ed161498a7d71c-Abstract-Conference.html)

**Abstract**:

We introduce a data-driven learning framework that assimilates two powerful ideas: ideal large eddy simulation (LES) from turbulence closure modeling and neural stochastic differential equations (SDE) for stochastic modeling. The ideal LES models the LES flow by treating each full-order trajectory as a random realization of the underlying dynamics, as such, the effect of small-scales is marginalized to obtain the deterministic evolution of the LES state. However, ideal LES is analytically intractable. In our work, we use a latent neural SDE to model the evolution of the stochastic process and an encoder-decoder pair for transforming between the latent space and the desired ideal flow field. This stands in sharp contrast to other types of neural parameterization of closure models where each trajectory is treated as a deterministic realization of the dynamics. We show the effectiveness of our approach (niLES – neural ideal LES) on two challenging chaotic dynamical systems: Kolmogorov flow at a Reynolds number of 20,000 and flow past a cylinder at Reynolds number 500. Compared to competing methods, our method can handle non-uniform geometries using unstructured meshes seamlessly. In particular, niLES leads to trajectories with more accurate statistics and enhances stability, particularly for long-horizon rollouts. (Source codes and datasets will be made publicly available.)

----

## [0] On the Generalization Error of Stochastic Mirror Descent for Quadratically-Bounded Losses: an Improved Analysis

**Authors**: *Ta Duy Nguyen, Alina Ene, Huy Nguyen*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/daca83eba0a30a5ff2a3b9c53ff5a976-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/daca83eba0a30a5ff2a3b9c53ff5a976-Abstract-Conference.html)

**Abstract**:

In this work, we revisit the generalization error of stochastic mirror descent for quadratically bounded losses studied in Telgarsky (2022). Quadratically bounded losses is a broad class of loss functions, capturing both Lipschitz and smooth functions, for both regression and classification problems. We study the high probability generalization for this class of losses on linear predictors in both realizable and non-realizable cases when the data are sampled IID or from a Markov chain. The prior work relies on an intricate coupling argument between the iterates of the original problem and those projected onto a bounded domain. This approach enables blackbox application of concentration inequalities, but also leads to suboptimal guarantees due in part to the use of a union bound across all iterations. In this work, we depart significantly from the prior work of Telgarsky (2022), and introduce a novel approach for establishing high probability generalization guarantees. In contrast to the prior work, our work directly analyzes the moment generating function of a novel supermartingale sequence and leverages the structure of stochastic mirror descent. As a result, we obtain improved bounds in all aforementioned settings. Specifically, in the realizable case and non-realizable case with light-tailed sub-Gaussian data, we improve the bounds by a $\log T$ factor, matching the correct rates of $1/T$ and $1/\sqrt{T}$, respectively. In the more challenging case of heavy-tailed polynomial data, we improve the existing bound by a $\mathrm{poly}\ T$ factor.

----

## [0] StableFDG: Style and Attention Based Learning for Federated Domain Generalization

**Authors**: *Jungwuk Park, Dong-Jun Han, Jinho Kim, Shiqiang Wang, Christopher G. Brinton, Jaekyun Moon*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/dae8bdacd265399b193e6b43d44a80f0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/dae8bdacd265399b193e6b43d44a80f0-Abstract-Conference.html)

**Abstract**:

Traditional federated learning (FL) algorithms operate under the assumption that the data distributions at training (source domains) and testing (target domain) are the same. The fact that domain shifts often occur in practice necessitates equipping FL methods with a domain generalization (DG) capability. However, existing DG algorithms face fundamental challenges in FL setups due to the lack of samples/domains in each clientâ€™s local dataset. In this paper, we propose StableFDG, a style and attention based learning strategy for accomplishing federated domain generalization, introducing two key contributions. The first is style-based learning, which enables each client to explore novel styles beyond the original source domains in its local dataset, improving domain diversity based on the proposed style sharing, shifting, and exploration strategies. Our second contribution is an attention-based feature highlighter, which captures the similarities between the features of data samples in the same class, and emphasizes the important/common characteristics to better learn the domain-invariant characteristics of each class in data-poor FL scenarios. Experimental results show that StableFDG outperforms existing baselines on various DG benchmark datasets, demonstrating its efficacy.

----

## [0] MG-ViT: A Multi-Granularity Method for Compact and Efficient Vision Transformers

**Authors**: *Yu Zhang, Yepeng Liu, Duoqian Miao, Qi Zhang, Yiwei Shi, Liang Hu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/daeef96627a461ec43b7567b2930cfde-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/daeef96627a461ec43b7567b2930cfde-Abstract-Conference.html)

**Abstract**:

Vision Transformer (ViT) faces obstacles in wide application due to its huge computational cost. Almost all existing studies on compressing ViT adopt the manner of splitting an image with a single granularity, with very few exploration of splitting an image with multi-granularity. As we know, important information often randomly concentrate in few regions of an image, necessitating multi-granularity attention allocation to an image. Enlightened by this, we introduce the multi-granularity strategy to compress ViT, which is simple but effective. We propose a two-stage multi-granularity framework, MG-ViT, to balance ViTâ€™s performance and computational cost. In single-granularity inference stage, an input image is split into a small number of patches for simple inference. If necessary, multi-granularity inference stage will be instigated, where the important patches are further subsplit into multi-finer-grained patches for subsequent inference. Moreover, prior studies on compression only for classification, while we extend the multi-granularity strategy to hierarchical ViT for downstream tasks such as detection and segmentation. Extensive experiments Prove the effectiveness of the multi-granularity strategy. For instance, on ImageNet, without any loss of performance, MG-ViT reduces 47\% FLOPs of LV-ViT-S and 56\% FLOPs of DeiT-S.

----

## [0] Recurrent Temporal Revision Graph Networks

**Authors**: *Yizhou Chen, Anxiang Zeng, Qingtao Yu, Kerui Zhang, Yuanpeng Cao, Kangle Wu, Guangda Huzhang, Han Yu, Zhiming Zhou*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/dafd116ac8c735f149558b79fd48e090-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/dafd116ac8c735f149558b79fd48e090-Abstract-Conference.html)

**Abstract**:

Temporal graphs offer more accurate modeling of many real-world scenarios than static graphs. However, neighbor aggregation, a critical building block of graph networks, for temporal graphs, is currently straightforwardly extended from that of static graphs. It can be computationally expensive when involving all historical neighbors during such aggregation. In practice, typically only a subset of the most recent neighbors are involved. However, such subsampling leads to incomplete and biased neighbor information. To address this limitation, we propose a novel framework for temporal neighbor aggregation that uses the recurrent neural network with node-wise hidden states to integrate information from all historical neighbors for each node to acquire the complete neighbor information. We demonstrate the superior theoretical expressiveness of the proposed framework as well as its state-of-the-art performance in real-world applications. Notably, it achieves a significant +9.4% improvement on averaged precision in a real-world Ecommerce dataset over existing methods on 2-layer models.

----

## [0] Recursion in Recursion: Two-Level Nested Recursion for Length Generalization with Scalability

**Authors**: *Jishnu Ray Chowdhury, Cornelia Caragea*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/db178cd03313e23cffb8937e93f0d464-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/db178cd03313e23cffb8937e93f0d464-Abstract-Conference.html)

**Abstract**:

Binary Balanced Tree Recursive Neural Networks (BBT-RvNNs) enforce sequence composition according to a preset balanced binary tree structure. Thus, their non-linear recursion depth (which is the tree depth) is just $\log_2 n$ ($n$ being the sequence length). Such logarithmic scaling makes BBT-RvNNs efficient and scalable on long sequence tasks such as Long Range Arena (LRA). However, such computational efficiency comes at a cost because BBT-RvNNs cannot solve simple arithmetic tasks like ListOps. On the flip side, RvNN models (e.g., Beam Tree RvNN) that do succeed on ListOps (and other structure-sensitive tasks like formal logical inference) are generally several times more expensive (in time and space) than even Recurrent Neural Networks. In this paper, we introduce a novel framework --- Recursion in Recursion (RIR) to strike a balance between the two sides - getting some of the benefits from both worlds. In RIR, we use a form of two-level nested recursion - where the outer recursion is a $k$-ary balanced tree model with another recursive model (inner recursion) implementing its cell function. For the inner recursion, we choose Beam Tree RvNNs. To adjust Beam Tree RvNNs within RIR we also propose a novel strategy of beam alignment. Overall, this entails that the total recursive depth in RIR is upper-bounded by $k \log_k n$. Our best RIR-based model is the first model that demonstrates high ($\geq 90\%$) length-generalization performance on ListOps while at the same time being scalable enough to be trainable on long sequence inputs from LRA (it can reduce the memory usage of the original Beam Tree RvNN by hundreds of times). Moreover, in terms of accuracy in the LRA language tasks, it performs competitively with Structured State Space Models (SSMs) without any special initialization - outperforming Transformers by a large margin. On the other hand, while SSMs can marginally outperform RIR on LRA, they (SSMs) fail to length-generalize on ListOps. Our code is available at: https://github.com/JRC1995/BeamRecursionFamily/

----

## [0] xTrimoGene: An Efficient and Scalable Representation Learner for Single-Cell RNA-Seq Data

**Authors**: *Jing Gong, Minsheng Hao, Xingyi Cheng, Xin Zeng, Chiming Liu, Jianzhu Ma, Xuegong Zhang, Taifeng Wang, Le Song*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/db68f1c25678f72561ab7c97ce15d912-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/db68f1c25678f72561ab7c97ce15d912-Abstract-Conference.html)

**Abstract**:

Advances in high-throughput sequencing technology have led to significant progress in measuring gene expressions at the single-cell level. The amount of publicly available single-cell RNA-seq (scRNA-seq) data is already surpassing 50M records for humans with each record measuring 20,000 genes. This highlights the need for unsupervised representation learning to fully ingest these data, yet classical transformer architectures are prohibitive to train on such data in terms of both computation and memory. To address this challenge, we propose a novel asymmetric encoder-decoder transformer for scRNA-seq data, called xTrimoGene$^\alpha$ (or xTrimoGene for short), which leverages the sparse characteristic of the data to scale up the pre-training. This scalable design of xTrimoGene reduces FLOPs by one to two orders of magnitude compared to classical transformers while maintaining high accuracy, enabling us to train the largest transformer models over the largest scRNA-seq dataset today. Our experiments also show that the performance of xTrimoGene improves as we scale up the model sizes, and it also leads to SOTA performance over various downstream tasks, such as cell type annotation, perturb-seq effect prediction, and drug combination prediction. xTrimoGene model is now available for use as a service via the following link: https://api.biomap.com/xTrimoGene/apply.

----

## [0] ANPL: Towards Natural Programming with Interactive Decomposition

**Authors**: *Di Huang, Ziyuan Nan, Xing Hu, Pengwei Jin, Shaohui Peng, Yuanbo Wen, Rui Zhang, Zidong Du, Qi Guo, Yewen Pu, Yunji Chen*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/dba8fa689ede9e56cbcd4f719def38fb-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/dba8fa689ede9e56cbcd4f719def38fb-Abstract-Conference.html)

**Abstract**:

Though LLMs are capable of generating plausible programs, it’s challenging to interact with the LLMs further to revise the program, especially if the user’s specific requirements are different from the initial proposal. In this paper, we introduce ANPL, an interactive programming system that ensures users can always refine the generated code towards their specific programmatic intents via structureddecompositions. Borrowing the paradigm of sketching from program synthesis, an ANPL program consists of a set of input-outputs that it must satisfy, a “sketch” — control/data flow expressed in precise code (e.g. Python), and “holes” — sub-modules to be implemented by the LLM specified with natural language. The user revises an ANPL program by either modifying the sketch, changing the language used to describe the holes, or providing additional input-outputs to a particular hole, turning it into a sub-ANPL program that can be solved recursively. This workflow allows the users to offload programming burdens to the LLM as much as possible while retaining the ability to pinpoint and resolve bugs locally, without exposing the rest of the program to the LLM. We deploy ANPL on the Abstraction and Reasoning Corpus (ARC), a set of unique tasks that are challenging for state-of-the-art AI systems, showing it outperforms baseline programming systems that (a) without the ability to decompose tasks interactively and (b) without the guarantee that the modules can be correctly composed together. Additional evaluations on APPS, HumanEval, and real-world programming tasks have validated that the ANPL framework is applicable to multiple programming domains. We release the ANPL solutions to the ARC tasks as a dataset, providing insights into how humans decompose novel tasks programmatically.

----

## [0] Anonymous and Copy-Robust Delegations for Liquid Democracy

**Authors**: *Markus Utke, Ulrike Schmidt-Kraepelin*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/dbb5180957513805ebeea787b8c66ac9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/dbb5180957513805ebeea787b8c66ac9-Abstract-Conference.html)

**Abstract**:

Liquid democracy with ranked delegations is a novel voting scheme that unites the practicability of representative democracy with the idealistic appeal of direct democracy: Every voter decides between casting their vote on a question at hand or delegating their voting weight to some other, trusted agent. Delegations are transitive, and since voters may end up in a delegation cycle, they are encouraged to indicate not only a single delegate, but a set of potential delegates and a ranking among them. Based on the delegation preferences of all voters, a delegation rule selects one representative per voter. Previous work has revealed a trade-off between two properties of delegation rules called anonymity and copy-robustness. To overcome this issue we study two fractional delegation rules: Mixed Borda branching, which generalizes a rule satisfying copy-robustness, and the random walk rule, which satisfies anonymity. Using the Markov chain tree theorem, we show that the two rules are in fact equivalent, and simultaneously satisfy generalized versions of the two properties. Combining the same theorem with Fulkerson's algorithm, we develop  a polynomial-time algorithm for computing the outcome of the studied delegation rule. This algorithm is of independent interest, having applications in semi-supervised learning and graph theory.

----

## [0] Framework and Benchmarks for Combinatorial and Mixed-variable Bayesian Optimization

**Authors**: *Kamil Dreczkowski, Antoine Grosnit, Haitham Bou-Ammar*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/dbc4b67c6430c22460623186c3d3fdc2-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/dbc4b67c6430c22460623186c3d3fdc2-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

This paper introduces a modular framework for Mixed-variable and Combinatorial Bayesian Optimization (MCBO) to address the lack of systematic benchmarking and standardized evaluation in the field. Current MCBO papers often introduce non-diverse or non-standard benchmarks to evaluate their methods, impeding the proper assessment of different MCBO primitives and their combinations. Additionally,  papers introducing a solution for a single MCBO primitive often omit benchmarking against baselines that utilize the same methods for the remaining primitives. This omission is primarily due to the significant implementation overhead involved, resulting in a lack of controlled assessments and an inability to showcase the merits of a contribution effectively.To overcome these challenges, our proposed framework enables an effortless combination of Bayesian Optimization components, and provides a diverse set of synthetic and real-world benchmarking tasks. Leveraging this flexibility, we implement 47 novel MCBO algorithms and benchmark them against seven existing MCBO solvers and five standard black-box optimization algorithms on ten tasks, conducting over 4000 experiments. Our findings reveal a superior combination of MCBO primitives outperforming existing approaches and illustrate the significance of model fit and the use of a trust region. We make our MCBO library available under the MIT license at \url{https://github.com/huawei-noah/HEBO/tree/master/MCBO}.

----

## [0] KD-Zero: Evolving Knowledge Distiller for Any Teacher-Student Pairs

**Authors**: *Lujun Li, Peijie Dong, Anggeng Li, Zimian Wei, Ya Yang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/dbc8ce0fdfcd55172d73fb05dbae07fc-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/dbc8ce0fdfcd55172d73fb05dbae07fc-Abstract-Conference.html)

**Abstract**:

Knowledge distillation (KD) has emerged as an effective technique for compressing models that can enhance the lightweight model.  Conventional KD methods propose various designs to allow student model to imitate the teacher better.  However, these handcrafted KD designs heavily rely on expert knowledge and may be sub-optimal for various teacher-student pairs.  In this paper, we present a novel framework, KD-Zero, which utilizes evolutionary search to automatically discover promising distiller  from scratch for any teacher-student architectures.  Specifically, we first decompose the generalized distiller into knowledge transformations, distance functions, and loss weights.  Then,  we construct our distiller search space by selecting advanced operations for these three components.  With sharpness and represent gap as fitting objectives, we evolve candidate populations and generate better distillers by crossover and mutation.  To ensure efficient searching, we employ the loss-rejection protocol, search space shrinkage, and proxy settings during the search process.  In this manner, the discovered distiller can address the capacity gap and cross-architecture challenges for any teacher-student pairs in the final distillation stage.  Comprehensive experiments reveal that KD-Zero consistently outperforms other state-of-the-art methods across diverse architectures on classification, detection, and segmentation tasks.  Noticeably, we provide some practical insights in designing the distiller by analyzing the distiller discovered.  Codes are available in supplementary materials.

----

## [0] Minimum-Risk Recalibration of Classifiers

**Authors**: *Zeyu Sun, Dogyoon Song, Alfred O. Hero III*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/dbd6b295535e44f2b8ec0c3f1da7c509-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/dbd6b295535e44f2b8ec0c3f1da7c509-Abstract-Conference.html)

**Abstract**:

Recalibrating probabilistic classifiers is vital for enhancing the reliability and accuracy of predictive models. Despite the development of numerous recalibration algorithms, there is still a lack of a comprehensive theory that integrates calibration and sharpness (which is essential for maintaining predictive power). In this paper, we introduce the concept of minimum-risk recalibration within the framework of mean-squared-error (MSE) decomposition, offering a principled approach for evaluating and recalibrating probabilistic classifiers. Using this framework, we analyze the uniform-mass binning (UMB) recalibration method and establish a finite-sample risk upper bound of order $\tilde{O}(B/n + 1/B^2)$ where $B$ is the number of bins and $n$ is the sample size. By balancing calibration and sharpness, we further determine that the optimal number of bins for UMB scales with $n^{1/3}$, resulting in a risk bound of approximately $O(n^{-2/3})$. Additionally, we tackle the challenge of label shift by proposing a two-stage approach that adjusts the recalibration function using limited labeled data from the target domain. Our results show that transferring a calibrated classifier requires significantly fewer target samples compared to recalibrating from scratch. We validate our theoretical findings through numerical simulations, which confirm the tightness of the proposed bounds, the optimal number of bins, and the effectiveness of label shift adaptation.

----

## [0] DynPoint: Dynamic Neural Point For View Synthesis

**Authors**: *Kaichen Zhou, Jia-Xing Zhong, Sangyun Shin, Kai Lu, Yiyuan Yang, Andrew Markham, Niki Trigoni*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/dbdc7a9779ce0278c6e43b62c7e97759-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/dbdc7a9779ce0278c6e43b62c7e97759-Abstract-Conference.html)

**Abstract**:

The introduction of neural radiance fields has greatly improved the effectiveness of view synthesis for monocular videos. However, existing algorithms face difficulties when dealing with uncontrolled or lengthy scenarios, and require extensive training time specific to each new scenario.To tackle these limitations, we propose DynPoint, an algorithm designed to facilitate the rapid synthesis of novel views for unconstrained monocular videos. Rather than encoding the entirety of the scenario information into a latent representation, DynPoint concentrates on predicting the explicit 3D correspondence between neighboring frames to realize information aggregation.Specifically, this correspondence prediction is achieved through the estimation of consistent depth and scene flow information across frames.Subsequently, the acquired correspondence is utilized to aggregate information from multiple reference frames to a target frame, by constructing hierarchical neural point clouds. The resulting framework enables swift and accurate view synthesis for desired views of target frames. The experimental results obtained demonstrate the considerable acceleration of training time achieved - typically an order of magnitude - by our proposed method while yielding comparable outcomes compared to prior approaches. Furthermore, our method exhibits strong robustness in handling long-duration videos without learning a canonical representation of video content.

----

## [0] Data-driven Optimal Filtering for Linear Systems with Unknown Noise Covariances

**Authors**: *Shahriar Talebi, Amirhossein Taghvaei, Mehran Mesbahi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/dbe8185809cb7032ec7ec6e365e3ed3b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/dbe8185809cb7032ec7ec6e365e3ed3b-Abstract-Conference.html)

**Abstract**:

This paper examines learning the optimal filtering policy, known as the Kalman gain, for a linear system with unknown noise covariance matrices using noisy output data. The learning problem is formulated as a stochastic policy optimiza- tion problem, aiming to minimize the output prediction error. This formulation provides a direct bridge between data-driven optimal control and, its dual, op- timal filtering. Our contributions are twofold. Firstly, we conduct a thorough convergence analysis of the stochastic gradient descent algorithm, adopted for the filtering problem, accounting for biased gradients and stability constraints. Secondly, we carefully leverage a combination of tools from linear system theory and high-dimensional statistics to derive bias-variance error bounds that scale logarithmically with problem dimension, and, in contrast to subspace methods, the length of output trajectories only affects the bias term.

----

## [0] PPi: Pretraining Brain Signal Model for Patient-independent Seizure Detection

**Authors**: *Zhizhang Yuan, Daoze Zhang, Yang Yang, Junru Chen, Yafeng Li*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/dbeb7e621d4a554069a6a775da0f7273-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/dbeb7e621d4a554069a6a775da0f7273-Abstract-Conference.html)

**Abstract**:

Automated seizure detection is of great importance to epilepsy diagnosis and treatment. An emerging method used in seizure detection, stereoelectroencephalography (SEEG), can provide detailed and stereoscopic brainwave information. However, modeling SEEG in clinical scenarios will face challenges like huge domain shift between different patients and dramatic pattern evolution among different brain areas. In this study, we propose a Pretraining-based model for Patient-independent seizure detection (PPi) to address these challenges. Firstly, we design two novel self-supervised tasks which can extract rich information from abundant SEEG data while preserving the unique characteristics between brain signals recorded from different brain areas. Then two techniques channel background subtraction and brain region enhancement are proposed to effectively tackle the domain shift problem. Extensive experiments show that PPi outperforms the SOTA baselines on two public datasets and a real-world clinical dataset collected by ourselves, which demonstrates the effectiveness and practicability of PPi. Finally, visualization analysis illustrates the rationality of the two domain generalization techniques.

----

## [0] Unsupervised Polychromatic Neural Representation for CT Metal Artifact Reduction

**Authors**: *Qing Wu, Lixuan Chen, Ce Wang, Hongjiang Wei, S. Kevin Zhou, Jingyi Yu, Yuyao Zhang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/dbf02b21d77409a2db30e56866a8ab3a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/dbf02b21d77409a2db30e56866a8ab3a-Abstract-Conference.html)

**Abstract**:

Emerging neural reconstruction techniques based on tomography (e.g., NeRF, NeAT, and NeRP) have started showing unique capabilities in medical imaging. In this work, we present a novel Polychromatic neural representation (Polyner) to tackle the challenging problem of CT imaging when metallic implants exist within the human body. CT metal artifacts arise from the drastic variation of metal's attenuation coefficients at various energy levels of the X-ray spectrum, leading to a nonlinear metal effect in CT measurements. Recovering CT images from metal-affected measurements hence poses a complicated nonlinear inverse problem where empirical models adopted in previous metal artifact reduction (MAR) approaches lead to signal loss and strongly aliased reconstructions. Polyner instead models the MAR problem from a nonlinear inverse problem perspective. Specifically, we first derive a polychromatic forward model to accurately simulate the nonlinear CT acquisition process. Then, we incorporate our forward model into the implicit neural representation to accomplish reconstruction. Lastly, we adopt a regularizer to preserve the physical properties of the CT images across different energy levels while effectively constraining the solution space. Our Polyner is an unsupervised method and does not require any external training data. Experimenting with multiple datasets shows that our Polyner achieves comparable or better performance than supervised methods on in-domain datasets while demonstrating significant performance improvements on out-of-domain datasets. To the best of our knowledge, our Polyner is the first unsupervised MAR method that outperforms its supervised counterparts. The code for this work is available at: https://github.com/iwuqing/Polyner.

----

## [0] DAMEX: Dataset-aware Mixture-of-Experts for visual understanding of mixture-of-datasets

**Authors**: *Yash Jain, Harkirat S. Behl, Zsolt Kira, Vibhav Vineet*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/dc192b3eeffebba21bd1d82f6752b84b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/dc192b3eeffebba21bd1d82f6752b84b-Abstract-Conference.html)

**Abstract**:

Construction of a universal detector poses a crucial question: How can we most effectively train a model on a large mixture of datasets?     The answer lies in learning dataset-specific features and ensembling their knowledge but do all this in a single model.    Previous methods achieve this by having separate detection heads on a common backbone but that results in a significant increase in parameters.    In this work, we present Mixture-of-Experts as a solution, highlighting that MoE are much more than a scalability tool.     We propose Dataset-Aware Mixture-of-Experts, DAMEX where we train the experts to become an `expert' of a dataset by learning to route each dataset tokens to its mapped expert.    Experiments on Universal Object-Detection Benchmark show that we outperform the existing state-of-the-art by average +10.2 AP score and improve over our non-MoE baseline by average +2.0 AP score. We also observe consistent gains while mixing datasets with (1) limited availability, (2) disparate domains and (3) divergent label sets.    Further, we qualitatively show that DAMEX is robust against expert representation collapse. Code is available at https://github.com/jinga-lala/DAMEX

----

## [0] FourierGNN: Rethinking Multivariate Time Series Forecasting from a Pure Graph Perspective

**Authors**: *Kun Yi, Qi Zhang, Wei Fan, Hui He, Liang Hu, Pengyang Wang, Ning An, Longbing Cao, Zhendong Niu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/dc1e32dd3eb381dbc71482f6a96cbf86-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/dc1e32dd3eb381dbc71482f6a96cbf86-Abstract-Conference.html)

**Abstract**:

Multivariate time series (MTS) forecasting has shown great importance in numerous industries. Current state-of-the-art graph neural network (GNN)-based forecasting methods usually require both graph networks (e.g., GCN) and temporal networks (e.g., LSTM) to capture inter-series (spatial) dynamics and intra-series (temporal) dependencies, respectively. However, the uncertain compatibility of the two networks puts an extra burden on handcrafted model designs. Moreover, the separate spatial and temporal modeling naturally violates the unified spatiotemporal inter-dependencies in real world, which largely hinders the forecasting performance. To overcome these problems, we explore an interesting direction of directly applying graph networks and rethink MTS forecasting from a pure graph perspective. We first define a novel data structure, hypervariate graph, which regards each series value (regardless of variates or timestamps) as a graph node, and represents sliding windows as space-time fully-connected graphs. This perspective considers spatiotemporal dynamics unitedly and reformulates classic MTS forecasting into the predictions on hypervariate graphs. Then, we propose a novel architecture Fourier Graph Neural Network (FourierGNN) by stacking our proposed Fourier Graph Operator (FGO) to perform matrix multiplications in Fourier space. FourierGNN accommodates adequate expressiveness and achieves much lower complexity, which can effectively and efficiently accomplish {the forecasting}. Besides, our theoretical analysis reveals FGO's equivalence to graph convolutions in the time domain, which further verifies the validity of FourierGNN. Extensive experiments on seven datasets have demonstrated our superior performance with higher efficiency and fewer parameters compared with state-of-the-art methods. Code is available at this repository: https://github.com/aikunyi/FourierGNN.

----

## [0] Representation Equivalent Neural Operators: a Framework for Alias-free Operator Learning

**Authors**: *Francesca Bartolucci, Emmanuel de Bézenac, Bogdan Raonic, Roberto Molinaro, Siddhartha Mishra, Rima Alaifari*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/dc35c593e61f6df62db541b976d09dcf-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/dc35c593e61f6df62db541b976d09dcf-Abstract-Conference.html)

**Abstract**:

Recently, operator learning, or learning mappings between infinite-dimensional function spaces, has garnered significant attention, notably in relation to learning partial differential equations from data. Conceptually clear when outlined on paper, neural operators necessitate discretization in the transition to computer implementations. This step can compromise their integrity, often causing them to deviate from the underlying operators. This research offers a fresh take on neural operators with a framework Representation equivalent Neural Operators (ReNO) designed to address these issues. At its core is the concept of operator aliasing, which measures inconsistency between neural operators and their discrete representations. We explore this for widely-used operator learning techniques. Our findings detail how aliasing introduces errors when handling different discretizations and grids and loss of crucial continuous structures. More generally, this framework not only sheds light on existing challenges but, given its constructive and broad nature, also potentially offers tools for developing new neural operators.

----

## [0] Unsupervised Anomaly Detection with Rejection

**Authors**: *Lorenzo Perini, Jesse Davis*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/dc48c738d3ef8c81b6e968453a84a819-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/dc48c738d3ef8c81b6e968453a84a819-Abstract-Conference.html)

**Abstract**:

Anomaly detection aims at detecting unexpected behaviours in the data. Because anomaly detection is usually an unsupervised task, traditional anomaly detectors learn a decision boundary by employing heuristics based on intuitions, which are hard to verify in practice. This introduces some uncertainty, especially close to the decision boundary, that may reduce the user trust in the detector's predictions. A way to combat this is by allowing the detector to reject predictions with high uncertainty (Learning to Reject). This requires employing a confidence metric that captures the distance to the decision boundary and setting a rejection threshold to reject low-confidence predictions. However, selecting a proper metric and setting the rejection threshold without labels are challenging tasks. In this paper, we solve these challenges by setting a constant rejection threshold on the stability metric computed by ExCeeD. Our insight relies on a theoretical analysis of such a metric. Moreover, setting a constant threshold results in strong guarantees: we estimate the test rejection rate, and derive a theoretical upper bound for both the rejection rate and the expected prediction cost. Experimentally, we show that our method outperforms some metric-based methods.

----

## [0] 4D Panoptic Scene Graph Generation

**Authors**: *Jingkang Yang, Jun Cen, Wenxuan Peng, Shuai Liu, Fangzhou Hong, Xiangtai Li, Kaiyang Zhou, Qifeng Chen, Ziwei Liu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/dc6319dde4fb182b22fb902da9418566-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/dc6319dde4fb182b22fb902da9418566-Abstract-Conference.html)

**Abstract**:

We are living in a three-dimensional space while moving forward through a fourth dimension: time. To allow artificial intelligence to develop a comprehensive understanding of such a 4D environment, we introduce 4D Panoptic Scene Graph (PSG-4D), a new representation that bridges the raw visual data perceived in a dynamic 4D world and high-level visual understanding. Specifically, PSG-4D abstracts rich 4D sensory data into nodes, which represent entities with precise location and status information, and edges, which capture the temporal relations. To facilitate research in this new area, we build a richly annotated PSG-4D dataset consisting of 3K RGB-D videos with a total of 1M frames, each of which is labeled with 4D panoptic segmentation masks as well as fine-grained, dynamic scene graphs. To solve PSG-4D, we propose PSG4DFormer, a Transformer-based model that can predict panoptic segmentation masks, track masks along the time axis, and generate the corresponding scene graphs via a relation component. Extensive experiments on the new dataset show that our method can serve as a strong baseline for future research on PSG-4D. In the end, we provide a real-world application example to demonstrate how we can achieve dynamic scene understanding by integrating a large language model into our PSG-4D system.

----

## [0] ChatGPT-Powered Hierarchical Comparisons for Image Classification

**Authors**: *Zhiyuan Ren, Yiyang Su, Xiaoming Liu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/dc81297c791bb989deade65c6bd8c1d8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/dc81297c791bb989deade65c6bd8c1d8-Abstract-Conference.html)

**Abstract**:

The zero-shot open-vocabulary setting poses challenges for image classification.Fortunately, utilizing a vision-language model like CLIP, pre-trained on image-textpairs, allows for classifying images by comparing embeddings. Leveraging largelanguage models (LLMs) such as ChatGPT can further enhance CLIPâ€™s accuracyby incorporating class-specific knowledge in descriptions. However, CLIP stillexhibits a bias towards certain classes and generates similar descriptions for similarclasses, disregarding their differences. To address this problem, we present anovel image classification framework via hierarchical comparisons. By recursivelycomparing and grouping classes with LLMs, we construct a class hierarchy. Withsuch a hierarchy, we can classify an image by descending from the top to the bottomof the hierarchy, comparing image and text embeddings at each level. Throughextensive experiments and analyses, we demonstrate that our proposed approach isintuitive, effective, and explainable. Code will be released upon publication.

----

## [0] Module-wise Adaptive Distillation for Multimodality Foundation Models

**Authors**: *Chen Liang, Jiahui Yu, Ming-Hsuan Yang, Matthew Brown, Yin Cui, Tuo Zhao, Boqing Gong, Tianyi Zhou*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/dc9544b26ad3579477e567588db18cfc-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/dc9544b26ad3579477e567588db18cfc-Abstract-Conference.html)

**Abstract**:

Pre-trained multimodal foundation models have demonstrated remarkable generalizability but pose challenges for deployment due to their large sizes. One effective approach to reducing their sizes is layerwise distillation, wherein small student models are trained to match the hidden representations of large teacher models at each layer. Motivated by our observation that certain architecture components, referred to as modules, contribute more significantly to the student's performance than others, we propose to track the contributions of individual modules by recording the loss decrement after distillation each module and choose the module with a greater contribution to distill more frequently. Such an approach can be naturally formulated as a multi-armed bandit (MAB) problem, where modules and loss decrements are considered as arms and rewards, respectively. We then develop a modified-Thompson sampling algorithm named OPTIMA to address the nonstationarity of module contributions resulting from model updating. Specifically, we leverage the observed contributions in recent history to estimate the changing contribution of each module and select modules based on these estimations to maximize the cumulative contribution. We evaluate the effectiveness of OPTIMA through distillation experiments on various multimodal understanding and image captioning tasks, using the CoCa-Large model \citep{yu2022coca} as the teacher model.

----

## [0] Evaluating Cognitive Maps and Planning in Large Language Models with CogEval

**Authors**: *Ida Momennejad, Hosein Hasanbeig, Felipe Vieira Frujeri, Hiteshi Sharma, Nebojsa Jojic, Hamid Palangi, Robert Osazuwa Ness, Jonathan Larson*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/dc9d5dcf3e86b83e137bad367227c8ca-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/dc9d5dcf3e86b83e137bad367227c8ca-Abstract-Conference.html)

**Abstract**:

Recently an influx of studies claims emergent cognitive abilities in large language models (LLMs). Yet, most rely on anecdotes, overlook contamination of training sets, or lack systematic Evaluation involving multiple tasks, control conditions, multiple iterations, and statistical robustness tests. Here we make two major contributions. First, we propose CogEval, a cognitive science-inspired protocol for the systematic evaluation of cognitive capacities in LLMs. The CogEval protocol can be followed for the evaluation of various abilities. Second, here we follow CogEval to systematically evaluate cognitive maps and planning ability across eight LLMs (OpenAI GPT-4, GPT-3.5-turbo-175B, davinci-003-175B, Google Bard, Cohere-xlarge-52.4B, Anthropic Claude-1-52B, LLaMA-13B, and Alpaca-7B). We base our task prompts on human experiments, which offer both established construct validity for evaluating planning, and are absent from LLM training sets. We find that, while LLMs show apparent competence in a few planning tasks with simpler structures, systematic evaluation reveals striking failure modes in planning tasks, including hallucinations of invalid trajectories and falling in loops. These findings do not support the idea of emergent out-of-the-box planning ability in LLMs. This could be because LLMs do not understand the latent relational structures underlying planning problems, known as cognitive maps, and fail at unrolling goal-directed trajectories based on the underlying structure. Implications for application and future directions are discussed.

----

## [0] Unsupervised Image Denoising with Score Function

**Authors**: *Yutong Xie, Mingze Yuan, Bin Dong, Quanzheng Li*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/dc9e095f668044e7a0909a4ea3926beb-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/dc9e095f668044e7a0909a4ea3926beb-Abstract-Conference.html)

**Abstract**:

Though achieving excellent performance in some cases, current unsupervised learning methods for single image denoising usually have constraints in applications. In this paper, we propose a new approach which is more general and applicable to complicated noise models. Utilizing the property of score function, the gradient of logarithmic probability, we define a solving system for denoising. Once the score function of noisy images has been estimated, the denoised result can be obtained through the solving system. Our approach can be applied to multiple noise models, such as the mixture of multiplicative and additive noise combined with structured correlation. Experimental results show that our method is comparable when the noise model is simple, and has good performance in complicated cases where other methods are not applicable or perform poorly.

----

## [0] Iterative Reachability Estimation for Safe Reinforcement Learning

**Authors**: *Milan Ganai, Zheng Gong, Chenning Yu, Sylvia L. Herbert, Sicun Gao*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/dca63f2650fe9e88956c1b68440b8ee9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/dca63f2650fe9e88956c1b68440b8ee9-Abstract-Conference.html)

**Abstract**:

Ensuring safety is important for the practical deployment of reinforcement learning (RL). Various challenges must be addressed, such as handling stochasticity in the environments, providing rigorous guarantees of persistent state-wise safety satisfaction, and avoiding overly conservative behaviors that sacrifice performance. We propose a new framework, Reachability Estimation for Safe Policy Optimization (RESPO), for safety-constrained RL in general stochastic settings. In the feasible set where there exist violation-free policies, we optimize for rewards while maintaining persistent safety. Outside this feasible set, our optimization produces the safest behavior by guaranteeing entrance into the feasible set whenever possible with the least cumulative discounted violations. We introduce a class of algorithms using our novel reachability estimation function to optimize in our proposed framework and in similar frameworks such as those concurrently handling multiple hard and soft constraints. We theoretically establish that our algorithms almost surely converge to locally optimal policies of our safe optimization framework. We evaluate the proposed methods on a diverse suite of safe RL environments from Safety Gym, PyBullet, and MuJoCo, and show the benefits in improving both reward performance and safety compared with state-of-the-art baselines.

----

## [0] DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining

**Authors**: *Sang Michael Xie, Hieu Pham, Xuanyi Dong, Nan Du, Hanxiao Liu, Yifeng Lu, Percy Liang, Quoc V. Le, Tengyu Ma, Adams Wei Yu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/dcba6be91359358c2355cd920da3fcbd-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/dcba6be91359358c2355cd920da3fcbd-Abstract-Conference.html)

**Abstract**:

The mixture proportions of pretraining data domains (e.g., Wikipedia, books, web text) greatly affect language model (LM) performance. In this paper, we propose Domain Reweighting with Minimax Optimization (DoReMi), which first trains a small proxy model using group distributionally robust optimization (Group DRO) over domains to produce domain weights (mixture proportions) without knowledge of downstream tasks. We then resample a dataset with these domain weights and train a larger, full-sized model. In our experiments, we use DoReMi on a 280M-parameter proxy model to set the domain weights for training an 8B-parameter model (30x larger) more efficiently. On The Pile, DoReMi improves perplexity across all domains, even when it downweights a domain. DoReMi improves average few-shot downstream accuracy by 6.5% points over a baseline model trained using The Pile's default domain weights and reaches the baseline accuracy with 2.6x fewer training steps. On the GLaM dataset, DoReMi, which has no knowledge of downstream tasks, even matches the performance of using domain weights tuned on downstream tasks.

----

## [0] OpenSTL: A Comprehensive Benchmark of Spatio-Temporal Predictive Learning

**Authors**: *Cheng Tan, Siyuan Li, Zhangyang Gao, Wenfei Guan, Zedong Wang, Zicheng Liu, Lirong Wu, Stan Z. Li*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/dcbff44d11130e75d09d3930411c23e1-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/dcbff44d11130e75d09d3930411c23e1-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Spatio-temporal predictive learning is a learning paradigm that enables models to learn spatial and temporal patterns by predicting future frames from given past frames in an unsupervised manner. Despite remarkable progress in recent years, a lack of systematic understanding persists due to the diverse settings, complex implementation, and difficult reproducibility. Without standardization, comparisons can be unfair and insights inconclusive. To address this dilemma, we propose OpenSTL, a comprehensive benchmark for spatio-temporal predictive learning that categorizes prevalent approaches into recurrent-based and recurrent-free models. OpenSTL provides a modular and extensible framework implementing various state-of-the-art methods. We conduct standard evaluations on datasets across various domains, including synthetic moving object trajectory, human motion, driving scenes, traffic flow, and weather forecasting. Based on our observations, we provide a detailed analysis of how model architecture and dataset properties affect spatio-temporal predictive learning performance. Surprisingly, we find that recurrent-free models achieve a good balance between efficiency and performance than recurrent models. Thus, we further extend the common MetaFormers to boost recurrent-free spatial-temporal predictive learning. We open-source the code and models at https://github.com/chengtan9907/OpenSTL.

----

## [0] Guiding The Last Layer in Federated Learning with Pre-Trained Models

**Authors**: *Gwen Legate, Nicolas Bernier, Lucas Page-Caccia, Edouard Oyallon, Eugene Belilovsky*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/dcc0ac74ac8b95dc1939804acce0317d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/dcc0ac74ac8b95dc1939804acce0317d-Abstract-Conference.html)

**Abstract**:

Federated Learning (FL) is an emerging paradigm that allows a model to be trained across a number of participants without sharing data. Recent works have begun to consider the effects of using pre-trained models as an initialization point for existing FL algorithms; however, these approaches ignore the vast body of efficient transfer learning literature from the centralized learning setting. Here we revisit the problem of FL from a pre-trained model considered in prior work and expand it to a set of computer vision transfer learning problems. We first observe that simply fitting a linear classification head can be efficient in many cases. We then show that in the FL setting, fitting a classifier using the Nearest Class Means (NCM) can be done exactly and  orders of magnitude more efficiently than existing proposals, while obtaining strong performance. Finally, we demonstrate that using a two-stage approach of obtaining the classifier and then fine-tuning the model can yield rapid convergence and improved generalization in the federated setting. We demonstrate the potential our method has to reduce communication and compute costs while achieving better model performance.

----

## [0] Unbiased learning of deep generative models with structured discrete representations

**Authors**: *Henry C. Bendekgey, Gabe Hope, Erik Sudderth*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/dcc337bb2a4d25afefd9ab800721debb-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/dcc337bb2a4d25afefd9ab800721debb-Abstract-Conference.html)

**Abstract**:

By composing graphical models with deep learning architectures, we learn generative models with the strengths of both frameworks. The structured variational autoencoder (SVAE) inherits structure and interpretability from graphical models, and flexible likelihoods for high-dimensional data from deep learning, but poses substantial optimization challenges.  We propose novel algorithms for learning SVAEs, and are the first to demonstrate the SVAE's ability to handle multimodal uncertainty when data is missing by incorporating discrete latent variables.  Our memory-efficient implicit differentiation scheme makes the SVAE tractable to learn via gradient descent, while demonstrating robustness to incomplete optimization. To more rapidly learn accurate graphical model parameters, we derive a method for computing natural gradients without manual derivations, which avoids biases found in prior work.  These optimization innovations enable the first comparisons of the SVAE to state-of-the-art time series models, where the SVAE performs competitively while learning interpretable and structured discrete data representations.

----

## [0] GLEMOS: Benchmark for Instantaneous Graph Learning Model Selection

**Authors**: *Namyong Park, Ryan A. Rossi, Xing Wang, Antoine Simoulin, Nesreen K. Ahmed, Christos Faloutsos*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/dcd18e50ebca0af89187c6e35dabb584-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/dcd18e50ebca0af89187c6e35dabb584-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

The choice of a graph learning (GL) model (i.e., a GL algorithm and its hyperparameter settings) has a significant impact on the performance of downstream tasks. However, selecting the right GL model becomes increasingly difficult and time consuming as more and more GL models are developed. Accordingly, it is of great significance and practical value to equip users of GL with the ability to perform a near-instantaneous selection of an effective GL model without manual intervention. Despite the recent attempts to tackle this important problem, there has been no comprehensive benchmark environment to evaluate the performance of GL model selection methods. To bridge this gap, we present GLEMOS in this work, a comprehensive benchmark for instantaneous GL model selection that makes the following contributions. (i) GLEMOS provides extensive benchmark data for fundamental GL tasks, i.e., link prediction and node classification, including the performances of 366 models on 457 graphs on these tasks. (ii) GLEMOS designs multiple evaluation settings, and assesses how effectively representative model selection techniques perform in these different settings. (iii) GLEMOS is designed to be easily extended with new models, new graphs, and new performance records. (iv) Based on the experimental results, we discuss the limitations of existing approaches and highlight future research directions. To promote research on this significant problem, we make the benchmark data and code publicly available at https://namyongpark.github.io/glemos.

----

## [0] STEVE-1: A Generative Model for Text-to-Behavior in Minecraft

**Authors**: *Shalev Lifshitz, Keiran Paster, Harris Chan, Jimmy Ba, Sheila A. McIlraith*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/dd03f856fc7f2efeec8b1c796284561d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/dd03f856fc7f2efeec8b1c796284561d-Abstract-Conference.html)

**Abstract**:

Constructing AI models that respond to text instructions is challenging, especially for sequential decision-making tasks. This work introduces a methodology, inspired by unCLIP, for instruction-tuning generative models of behavior without relying on a large dataset of instruction-labeled trajectories. Using this methodology, we create an instruction-tuned Video Pretraining (VPT) model called STEVE-1, which can follow short-horizon open-ended text and visual instructions in Minecraft. STEVE-1 is trained in two steps: adapting the pretrained VPT model to follow commands in MineCLIP's latent space, then training a prior to predict latent codes from text. This allows us to finetune VPT through self-supervised behavioral cloning and hindsight relabeling, reducing the need for costly human text annotations, and all for only $60 of compute. By leveraging pretrained models like VPT and MineCLIP and employing best practices from text-conditioned image generation, STEVE-1 sets a new bar for open-ended instruction following in Minecraft with low-level controls (mouse and keyboard) and raw pixel inputs, far outperforming previous baselines and robustly completing 12 of 13 tasks in our early-game evaluation suite. We provide experimental evidence highlighting key factors for downstream performance, including pretraining, classifier-free guidance, and data scaling. All resources, including our model weights, training scripts, and evaluation tools are made available for further research.

----

## [0] Thrust: Adaptively Propels Large Language Models with External Knowledge

**Authors**: *Xinran Zhao, Hongming Zhang, Xiaoman Pan, Wenlin Yao, Dong Yu, Jianshu Chen*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/dd058e9ec9dc012a273594d717c46ef3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/dd058e9ec9dc012a273594d717c46ef3-Abstract-Conference.html)

**Abstract**:

Although large-scale pre-trained language models (PTLMs) are shown to encode rich knowledge in their model parameters, the inherent knowledge in PTLMs can be opaque or static, making external knowledge necessary. However, the existing information retrieval techniques could be costly and may even introduce noisy and sometimes misleading knowledge. To address these challenges, we propose the instance-level adaptive propulsion of external knowledge (IAPEK), where we only conduct the retrieval when necessary. To achieve this goal, we propose to model whether a PTLM contains enough knowledge to solve an instance with a novel metric, Thrust, which leverages the representation distribution of a small amount of seen instances. Extensive experiments demonstrate that Thrust is a good measurement of models' instance-level knowledgeability. Moreover, we can achieve higher cost-efficiency with the Thrust score as the retrieval indicator than the naive usage of external knowledge on 88% of the evaluated tasks with 26% average performance improvement. Such findings shed light on the real-world practice of knowledge-enhanced LMs with a limited budget for knowledge seeking due to computation latency or costs.

----

## [0] OneNet: Enhancing Time Series Forecasting Models under Concept Drift by Online Ensembling

**Authors**: *Yifan Zhang, Qingsong Wen, Xue Wang, Weiqi Chen, Liang Sun, Zhang Zhang, Liang Wang, Rong Jin, Tieniu Tan*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/dd6a47bc0aad6f34aa5e77706d90cdc4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/dd6a47bc0aad6f34aa5e77706d90cdc4-Abstract-Conference.html)

**Abstract**:

Online updating of time series forecasting models aims to address the concept drifting problem by efficiently updating forecasting models based on streaming data. Many algorithms are designed for online time series forecasting, with some exploiting cross-variable dependency while others assume independence among variables. Given every data assumption has its own pros and cons in online time series modeling, we propose **On**line **e**nsembling **Net**work (**OneNet**). It dynamically updates and combines two models, with one focusing on modeling the dependency across the time dimension and the other on cross-variate dependency. Our method incorporates a reinforcement learning-based approach into the traditional online convex programming framework, allowing for the linear combination of the two models with dynamically adjusted weights. OneNet addresses the main shortcoming of classical online learning methods that tend to be slow in adapting to the concept drift. Empirical results show that OneNet reduces online forecasting error by more than $\mathbf{50}\\%$ compared to the State-Of-The-Art (SOTA) method.

----

## [0] Holistic Evaluation of Text-to-Image Models

**Authors**: *Tony Lee, Michihiro Yasunaga, Chenlin Meng, Yifan Mai, Joon Sung Park, Agrim Gupta, Yunzhi Zhang, Deepak Narayanan, Hannah Teufel, Marco Bellagente, Minguk Kang, Taesung Park, Jure Leskovec, Jun-Yan Zhu, Fei-Fei Li, Jiajun Wu, Stefano Ermon, Percy Liang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/dd83eada2c3c74db3c7fe1c087513756-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/dd83eada2c3c74db3c7fe1c087513756-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

The stunning qualitative improvement of text-to-image models has led to their widespread attention and adoption. However, we lack a comprehensive quantitative understanding of their capabilities and risks. To fill this gap, we introduce a new benchmark, Holistic Evaluation of Text-to-Image Models (HEIM). Whereas previous evaluations focus mostly on image-text alignment and image quality, we identify 12 aspects, including text-image alignment, image quality, aesthetics, originality, reasoning, knowledge, bias, toxicity, fairness, robustness, multilinguality, and efficiency. We curate 62 scenarios encompassing these aspects and evaluate 26 state-of-the-art text-to-image models on this benchmark. Our results reveal that no single model excels in all aspects, with different models demonstrating different strengths. We release the generated images and human evaluation results for full transparency at https://crfm.stanford.edu/heim/latest and the code at https://github.com/stanford-crfm/helm, which is integrated with the HELM codebase

----

## [0] Shape Non-rigid Kinematics (SNK): A Zero-Shot Method for Non-Rigid Shape Matching via Unsupervised Functional Map Regularized Reconstruction

**Authors**: *Souhaib Attaiki, Maks Ovsjanikov*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/dd9b76f050a86a3ded6135ad3556e786-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/dd9b76f050a86a3ded6135ad3556e786-Abstract-Conference.html)

**Abstract**:

We present Shape Non-rigid Kinematics (SNK), a novel zero-shot method for non-rigid shape matching that eliminates the need for extensive training or ground truth data.SNK operates on a single pair of shapes, and employs a reconstruction-based strategy using an encoder-decoder architecture, which deforms the source shape to closely match the target shape. During the process, an unsupervised functional map is predicted and converted into a point-to-point map, serving as a supervisory mechanism for the reconstruction. To aid in training, we have designed a new decoder architecture that generates smooth, realistic deformations. SNK demonstrates competitive results on traditional benchmarks, simplifying the shape-matching process without compromising accuracy. Our code can be found online: https://github.com/pvnieo/SNK

----

## [0] Frequency Domain-Based Dataset Distillation

**Authors**: *DongHyeok Shin, Seungjae Shin, Il-Chul Moon*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ddbbcd937d63d5c6b935c07b1a8222ec-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ddbbcd937d63d5c6b935c07b1a8222ec-Abstract-Conference.html)

**Abstract**:

This paper presents FreD, a novel parameterization method for dataset distillation, which utilizes the frequency domain to distill a small-sized synthetic dataset from a large-sized original dataset. Unlike conventional approaches that focus on the spatial domain, FreD employs frequency-based transforms to optimize the frequency representations of each data instance. By leveraging the concentration of spatial domain information on specific frequency components, FreD intelligently selects a subset of frequency dimensions for optimization, leading to a significant reduction in the required budget for synthesizing an instance. Through the selection of frequency dimensions based on the explained variance, FreD demonstrates both theoretical and empirical evidence of its ability to operate efficiently within a limited budget, while better preserving the information of the original dataset compared to conventional parameterization methods. Furthermore, Based on the orthogonal compatibility of FreD with existing methods, we confirm that FreD consistently improves the performances of existing distillation methods over the evaluation scenarios with different benchmark datasets. We release the code at https://github.com/sdh0818/FreD.

----

## [0] Three-Way Trade-Off in Multi-Objective Learning: Optimization, Generalization and Conflict-Avoidance

**Authors**: *Lisha Chen, Heshan Devaka Fernando, Yiming Ying, Tianyi Chen*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ddcf34623ca2d63823b6d40e4d980580-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ddcf34623ca2d63823b6d40e4d980580-Abstract-Conference.html)

**Abstract**:

Multi-objective learning (MOL) often arises in emerging machine learning problems when multiple learning criteria or tasks need to be addressed.  Recent works have developed various _dynamic weighting_ algorithms for MOL, including MGDA and its variants, whose central idea is to find an update direction that _avoids conflicts_ among objectives. Albeit its appealing intuition, empirical studies show that dynamic weighting methods may not always outperform static alternatives. To bridge this gap between theory and practice, we focus on a new variant of stochastic MGDA - the Multi-objective gradient with Double sampling (MoDo) algorithm and study its generalization performance and the interplay with optimization through the lens of algorithm stability. We find that the rationale behind MGDA -- updating along conflict-avoidant direction - may \emph{impede} dynamic weighting algorithms from achieving the optimal ${\cal O}(1/\sqrt{n})$ population risk, where $n$ is the number of training samples. We further highlight the variability of dynamic weights and their impact on the three-way trade-off among optimization, generalization, and conflict avoidance that is unique in MOL. Code is available at https://github.com/heshandevaka/Trade-Off-MOL.

----

## [0] OV-PARTS: Towards Open-Vocabulary Part Segmentation

**Authors**: *Meng Wei, Xiaoyu Yue, Wenwei Zhang, Shu Kong, Xihui Liu, Jiangmiao Pang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/dde53059fdb0f45e1e9ad9c66997d662-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/dde53059fdb0f45e1e9ad9c66997d662-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Segmenting and recognizing diverse object parts is a crucial ability in applications spanning various computer vision and robotic tasks. While significant progress has been made in object-level Open-Vocabulary Semantic Segmentation (OVSS), i.e., segmenting objects with arbitrary text, the corresponding part-level research poses additional challenges. Firstly, part segmentation inherently involves intricate boundaries, while limited annotated data compounds the challenge. Secondly, part segmentation introduces an open granularity challenge due to the diverse and often ambiguous definitions of parts in the open world. Furthermore, the large-scale vision and language models, which play a key role in the open vocabulary setting, struggle to recognize parts as effectively as objects. To comprehensively investigate and tackle these challenges, we propose an Open-Vocabulary Part Segmentation (OV-PARTS) benchmark. OV-PARTS includes refined versions of two publicly available datasets: Pascal-Part-116 and ADE20K-Part-234. And it covers three specific tasks: Generalized Zero-Shot Part Segmentation, Cross-Dataset Part Segmentation, and Few-Shot Part Segmentation, providing insights into analogical reasoning, open granularity and few-shot adapting abilities of models. Moreover, we analyze and adapt two prevailing paradigms of existing object-level OVSS methods for OV-PARTS. Extensive experimental analysis is conducted to inspire future research in leveraging foundational models for OV-PARTS. The code and dataset are available at https://github.com/kellyiss/OV_PARTS.

----

## [0] Large Language Models are Visual Reasoning Coordinators

**Authors**: *Liangyu Chen, Bo Li, Sheng Shen, Jingkang Yang, Chunyuan Li, Kurt Keutzer, Trevor Darrell, Ziwei Liu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ddfe6bae7b869e819f842753009b94ad-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ddfe6bae7b869e819f842753009b94ad-Abstract-Conference.html)

**Abstract**:

Visual reasoning requires multimodal perception and commonsense cognition of the world. Recently, multiple vision-language models (VLMs) have been proposed with excellent commonsense reasoning ability in various domains. However, how to harness the collective power of these complementary VLMs is rarely explored. Existing methods like ensemble still struggle to aggregate these models with the desired higher-order communications. In this work, we propose Cola, a novel paradigm that coordinates multiple VLMs for visual reasoning. Our key insight is that a large language model (LLM) can efficiently coordinate multiple VLMs by facilitating natural language communication that leverages their distinct and complementary capabilities. Extensive experiments demonstrate that our instruction tuning variant, Cola-FT, achieves state-of-the-art performance on visual question answering (VQA), outside knowledge VQA, visual entailment, and visual spatial reasoning tasks. Moreover, we show that our in-context learning variant, Cola-Zero, exhibits competitive performance in zero and few-shot settings, without finetuning. Through systematic ablation studies and visualizations, we validate that a coordinator LLM indeed comprehends the instruction prompts as well as the separate functionalities of VLMs; it then coordinates them to enable impressive visual reasoning capabilities.

----

## [0] Boosting Adversarial Transferability by Achieving Flat Local Maxima

**Authors**: *Zhijin Ge, Xiaosen Wang, Hongying Liu, Fanhua Shang, Yuanyuan Liu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/de1739eba209c682a90ec3669229ab2d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/de1739eba209c682a90ec3669229ab2d-Abstract-Conference.html)

**Abstract**:

Transfer-based attack adopts the adversarial examples generated on the surrogate model to attack various models, making it applicable in the physical world and attracting increasing interest. Recently, various adversarial attacks have emerged to boost adversarial transferability from different perspectives. In this work, inspired by the observation that flat local minima are correlated with good generalization, we assume and empirically validate that adversarial examples at a flat local region tend to have good transferability by introducing a penalized gradient norm to the original loss function. Since directly optimizing the gradient regularization norm is computationally expensive and intractable for generating adversarial examples, we propose an approximation optimization method to simplify the gradient update of the objective function. Specifically, we randomly sample an example and adopt a first-order procedure to approximate the curvature of the second-order Hessian matrix, which makes computing more efficient by interpolating two Jacobian matrices. Meanwhile, in order to obtain a more stable gradient direction, we randomly sample multiple examples and average the gradients of these examples to reduce the variance due to random sampling during the iterative process. Extensive experimental results on the ImageNet-compatible dataset show that the proposed method can generate adversarial examples at flat local regions, and significantly improve the adversarial transferability on either normally trained models or adversarially trained models than the state-of-the-art attacks. Our codes are available at: https://github.com/Trustworthy-AI-Group/PGN.

----

## [0] From Trainable Negative Depth to Edge Heterophily in Graphs

**Authors**: *Yuchen Yan, Yuzhong Chen, Huiyuan Chen, Minghua Xu, Mahashweta Das, Hao Yang, Hanghang Tong*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/de2d52c5cf2bea853ef39bb2e1535dde-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/de2d52c5cf2bea853ef39bb2e1535dde-Abstract-Conference.html)

**Abstract**:

Finding the proper depth $d$ of a graph convolutional network (GCN) that provides strong representation ability has drawn significant attention, yet nonetheless  largely  remains an open problem for the graph learning community. Although noteworthy progress has been made, the depth or the number of layers of a corresponding GCN is realized by a series of graph convolution operations, which naturally makes $d$ a positive integer ($d \in \mathbb{N}+$). An interesting question is whether breaking the constraint of $\mathbb{N}+$ by making $d$ a real number ($d \in \mathbb{R}$) can bring new insights into graph learning mechanisms. In this work, by redefining GCN's depth $d$ as a trainable parameter continuously adjustable within $(-\infty,+\infty)$, we open a new door of controlling its signal processing capability to model graph homophily/heterophily (nodes with similar/dissimilar labels/attributes tend to be inter-connected). A simple and powerful GCN model TEDGCN, is proposed to retain the simplicity of GCN and meanwhile automatically search for the optimal $d$ without the prior knowledge regarding whether the input graph is homophilic or heterophilic. Negative-valued $d$ intrinsically enables high-pass frequency filtering functionality via augmented topology for graph heterophily. Extensive experiments demonstrate the superiority of TEDGCN on node classification tasks for a variety of homophilic and heterophilic graphs.

----

## [0] ADGym: Design Choices for Deep Anomaly Detection

**Authors**: *Minqi Jiang, Chaochuan Hou, Ao Zheng, Songqiao Han, Hailiang Huang, Qingsong Wen, Xiyang Hu, Yue Zhao*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/de670b9d118229d09d9a9bd9dec2598b-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/de670b9d118229d09d9a9bd9dec2598b-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Deep learning (DL) techniques have recently found success in anomaly detection (AD) across various fields such as finance, medical services, and cloud computing. However, most of the current research tends to view deep AD algorithms as a whole, without dissecting the contributions of individual design choices like loss functions and network architectures. This view tends to diminish the value of preliminary steps like data preprocessing, as more attention is given to newly designed loss functions, network architectures, and learning paradigms. In this paper, we aim to bridge this gap by asking two key questions: (i) Which design choices in deep AD methods are crucial for detecting anomalies? (ii) How can we automatically select the optimal design choices for a given AD dataset, instead of relying on generic, pre-existing solutions? To address these questions, we introduce ADGym, a platform specifically crafted for comprehensive evaluation and automatic selection of AD design elements in deep methods. Our extensive experiments reveal that relying solely on existing leading methods is not sufficient. In contrast, models developed using ADGym significantly surpass current state-of-the-art techniques.

----

## [0] Low-shot Object Learning with Mutual Exclusivity Bias

**Authors**: *Anh Thai, Ahmad Humayun, Stefan Stojanov, Zixuan Huang, Bikram Boote, James M. Rehg*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/de6ff07cbd222c10d694c2b2f732aceb-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/de6ff07cbd222c10d694c2b2f732aceb-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

This paper introduces Low-shot Object Learning with Mutual Exclusivity Bias (LSME), the first computational framing of mutual exclusivity bias, a phenomenon commonly observed in infants during word learning. We provide a novel dataset, comprehensive baselines, and a SOTA method to enable the ML community to tackle this challenging learning task. The goal of LSME is to analyze an RGB image of a scene containing multiple objects and correctly associate a previously-unknown object instance with a provided category label. This association is then used to perform low-shot learning to test category generalization. We provide a data generation pipeline for the LSME problem and conduct a thorough analysis of the factors that contribute to its difficulty. Additionally, we evaluate the performance of multiple baselines, including state-of-the-art foundation models. Finally, we present a baseline approach that outperforms state-of-the-art models in terms of low-shot accuracy. Code and data are available at https://github.com/rehg-lab/LSME.

----

## [0] GPT-ST: Generative Pre-Training of Spatio-Temporal Graph Neural Networks

**Authors**: *Zhonghang Li, Lianghao Xia, Yong Xu, Chao Huang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/de7858e3e7f9f0f7b2c7bfdc86f6d928-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/de7858e3e7f9f0f7b2c7bfdc86f6d928-Abstract-Conference.html)

**Abstract**:

In recent years, there has been a rapid development of spatio-temporal prediction techniques in response to the increasing demands of traffic management and travel planning. While advanced end-to-end models have achieved notable success in improving predictive performance, their integration and expansion pose significant challenges. This work aims to address these challenges by introducing a spatio-temporal pre-training framework that seamlessly integrates with downstream baselines and enhances their performance. The framework is built upon two key designs: (i) We propose a spatio-temporal mask autoencoder as a pre-training model for learning spatio-temporal dependencies. The model incorporates customized parameter learners and hierarchical spatial pattern encoding networks. These modules are specifically designed to capture spatio-temporal customized representations and intra- and inter-cluster region semantic relationships, which have often been neglected in existing approaches. (ii) We introduce an adaptive mask strategy as part of the pre-training mechanism. This strategy guides the mask autoencoder in learning robust spatio-temporal representations and facilitates the modeling of different relationships, ranging from intra-cluster to inter-cluster, in an easy-to-hard training manner. Extensive experiments conducted on representative benchmarks demonstrate the effectiveness of our proposed method. We have made our model implementation publicly available at https://github.com/HKUDS/GPT-ST.

----

## [0] Direct Preference-based Policy Optimization without Reward Modeling

**Authors**: *Gaon An, Junhyeok Lee, Xingdong Zuo, Norio Kosaka, Kyung-Min Kim, Hyun Oh Song*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/de8bd6b2b01cfa788e63f62e5b9a99b9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/de8bd6b2b01cfa788e63f62e5b9a99b9-Abstract-Conference.html)

**Abstract**:

Preference-based reinforcement learning (PbRL) is an approach that enables RL agents to learn from preference, which is particularly useful when formulating a reward function is challenging. Existing PbRL methods generally involve a two-step procedure: they first learn a reward model based on given preference data and then employ off-the-shelf reinforcement learning algorithms using the learned reward model. However, obtaining an accurate reward model solely from preference information, especially when the preference is from human teachers, can be difficult. Instead, we propose a PbRL algorithm that directly learns from preference without requiring any reward modeling. To achieve this, we adopt a contrastive learning framework to design a novel policy scoring metric that assigns a high score to policies that align with the given preferences. We apply our algorithm to offline RL tasks with actual human preference labels and show that our algorithm outperforms or is on par with the existing PbRL methods. Notably, on high-dimensional control tasks, our algorithm surpasses offline RL methods that learn with ground-truth reward information. Finally, we show that our algorithm can be successfully applied to fine-tune large language models.

----

## [0] On the Identifiability and Interpretability of Gaussian Process Models

**Authors**: *Jiawen Chen, Wancen Mu, Yun Li, Didong Li*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/dea2b4f9012686bcc1f59a62bcd28158-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/dea2b4f9012686bcc1f59a62bcd28158-Abstract-Conference.html)

**Abstract**:

In this paper, we critically examine the prevalent practice of using additive mixtures of Mat\'ern kernels in single-output Gaussian process (GP) models and explore the properties of multiplicative mixtures of Mat\'ern kernels for multi-output GP models. For the single-output case, we derive a series of theoretical results showing that the smoothness of a mixture of Mat\'ern kernels is determined by the least smooth component and that a GP with such a kernel is effectively equivalent to the least smooth kernel component. Furthermore, we demonstrate that none of the mixing weights or parameters within individual kernel components are identifiable. We then turn our attention to multi-output GP models and analyze the identifiability of the covariance matrix $A$ in the multiplicative kernel $K(x,y) = AK_0(x,y)$, where $K_0$ is a standard single output kernel such as Mat\'ern. We show that $A$ is identifiable up to a multiplicative constant, suggesting that multiplicative mixtures are well suited for multi-output tasks. Our findings are supported by extensive simulations and real applications for both single- and multi-output settings. This work provides insight into kernel selection and interpretation for GP models, emphasizing the importance of choosing appropriate kernel structures for different tasks.

----

## [0] Enhancing Motion Deblurring in High-Speed Scenes with Spike Streams

**Authors**: *Shiyan Chen, Jiyuan Zhang, Yajing Zheng, Tiejun Huang, Zhaofei Yu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/dead3d8ff3f9198e38a36a950ebbcafd-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/dead3d8ff3f9198e38a36a950ebbcafd-Abstract-Conference.html)

**Abstract**:

Traditional cameras produce desirable vision results but struggle with motion blur in high-speed scenes due to long exposure windows. Existing frame-based deblurring algorithms face challenges in extracting useful motion cues from severely blurred images. Recently, an emerging bio-inspired vision sensor known as the spike camera has achieved an extremely high frame rate while preserving rich spatial details, owing to its novel sampling mechanism. However, typical binary spike streams are relatively low-resolution, degraded image signals devoid of color information, making them unfriendly to human vision. In this paper, we propose a novel approach that integrates the two modalities from two branches, leveraging spike streams as auxiliary visual cues for guiding deblurring in high-speed motion scenes. We propose the first spike-based motion deblurring model with bidirectional information complementarity. We introduce a content-aware motion magnitude attention module that utilizes learnable mask to extract relevant information from blurry images effectively, and we incorporate a transposed cross-attention fusion module to efficiently combine features from both spike data and blurry RGB images.Furthermore, we build two extensive synthesized datasets for training and validation purposes, encompassing high-temporal-resolution spikes, blurry images, and corresponding sharp images. The experimental results demonstrate that our method effectively recovers clear RGB images from highly blurry scenes and outperforms state-of-the-art deblurring algorithms in multiple settings.

----

## [0] Faith and Fate: Limits of Transformers on Compositionality

**Authors**: *Nouha Dziri, Ximing Lu, Melanie Sclar, Xiang Lorraine Li, Liwei Jiang, Bill Yuchen Lin, Sean Welleck, Peter West, Chandra Bhagavatula, Ronan Le Bras, Jena D. Hwang, Soumya Sanyal, Xiang Ren, Allyson Ettinger, Zaïd Harchaoui, Yejin Choi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/deb3c28192f979302c157cb653c15e90-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/deb3c28192f979302c157cb653c15e90-Abstract-Conference.html)

**Abstract**:

Transformer large language models (LLMs) have sparked admiration for their exceptional performance on tasks that demand intricate multi-step reasoning. Yet, these models simultaneously show failures on surprisingly trivial problems. This begs the question: Are these errors incidental, or do they signal more substantial limitations?In an attempt to demystify transformer LLMs, we investigate the limits of these models across three representative compositional tasks---multi-digit multiplication, logic grid puzzles, and a classic dynamic programming problem. These tasks require breaking problems down into sub-steps and synthesizing these steps into a precise answer.  We formulate compositional tasks as computation graphs to systematically quantify the level of complexity, and break down reasoning steps into intermediate sub-procedures. Our empirical findings suggest that transformer LLMs solve compositional tasks by reducing multi-step compositional reasoning into linearized subgraph matching, without necessarily developing systematic problem-solving skills.  To round off our empirical study, we provide theoretical arguments on abstract multi-step reasoning problems that highlight how autoregressive generations' performance can rapidly decay with increased task complexity.

----

## [0] Towards a fuller understanding of neurons with Clustered Compositional Explanations

**Authors**: *Biagio La Rosa, Leilani Gilpin, Roberto Capobianco*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/debd0ae2083160397a22a4a8831c7230-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/debd0ae2083160397a22a4a8831c7230-Abstract-Conference.html)

**Abstract**:

Compositional Explanations is a method for identifying logical formulas of concepts that approximate the neurons' behavior. However, these explanations are linked to the small spectrum of neuron activations (i.e., the highest ones) used to check the alignment, thus lacking completeness. In this paper, we propose a generalization, called Clustered Compositional Explanations, that combines  Compositional Explanations with clustering and a novel search heuristic to approximate a broader spectrum of the neuron behavior. We define and address the problems connected to the application of these methods to multiple ranges of activations, analyze the insights retrievable by using our algorithm, and propose desiderata qualities that can be used to study the explanations returned by different algorithms.

----

## [0] TpuGraphs: A Performance Prediction Dataset on Large Tensor Computational Graphs

**Authors**: *Mangpo Mangpo Phothilimthana, Sami Abu-El-Haija, Kaidi Cao, Bahare Fatemi, Michael Burrows, Charith Mendis, Bryan Perozzi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ded1a89e2b3b925444ada973af66336e-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/ded1a89e2b3b925444ada973af66336e-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Precise hardware performance models play a crucial role in code optimizations. They can assist compilers in making heuristic decisions or aid autotuners in identifying the optimal configuration for a given program. For example, the autotuner for XLA, a machine learning compiler, discovered 10â€“20\% speedup on state-of-the-art models serving substantial production traffic at Google. Although there exist a few datasets for program performance prediction, they target small sub-programs such as basic blocks or kernels. This paper introduces TpuGraphs, a performance prediction dataset on full tensor programs, represented as computational graphs, running on Tensor Processing Units (TPUs). Each graph in the dataset represents the main computation of a machine learning workload, e.g., a training epoch or an inference step. Each data sample contains a computational graph, a compilation configuration, and the execution time of the graph when compiled with the configuration. The graphs in the dataset are collected from open-source machine learning programs, featuring popular model architectures (e.g., ResNet, EfficientNet, Mask R-CNN, and Transformer). TpuGraphs provides 25x more graphs than the largest graph property prediction dataset (with comparable graph sizes), and 770x larger graphs on average compared to existing performance prediction datasets on machine learning programs. This graph-level prediction task on large graphs introduces new challenges in learning, ranging from scalability, training efficiency, to model quality.

----

## [0] ScaleLong: Towards More Stable Training of Diffusion Model via Scaling Network Long Skip Connection

**Authors**: *Zhongzhan Huang, Pan Zhou, Shuicheng Yan, Liang Lin*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ded98d28f82342a39f371c013dfb3058-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ded98d28f82342a39f371c013dfb3058-Abstract-Conference.html)

**Abstract**:

In diffusion models, UNet is the most popular network backbone, since its long skip connects (LSCs) to connect distant network blocks can aggregate long-distant information and alleviate vanishing gradient. Unfortunately, UNet often suffers from unstable training in diffusion models which can be alleviated by scaling its LSC coefficients smaller. However, theoretical understandings of the instability of UNet in diffusion models and also the performance improvement of LSC scaling remain absent yet. To solve this issue, we theoretically show that the coefficients of LSCs in UNet have big effects on the stableness of the forward and backward propagation and robustness of UNet. Specifically, the hidden feature and gradient of UNet at any layer can oscillate and their oscillation ranges are actually large which explains the instability of UNet training. Moreover, UNet is also provably sensitive to perturbed input, and predicts an output distant from the desired output, yielding oscillatory loss and thus oscillatory gradient. Besides, we also observe the theoretical benefits of the LSC coefficient scaling of UNet in the stableness of hidden features and gradient and also robustness. Finally,   inspired by our theory, we propose an effective coefficient scaling framework ScaleLong  that scales the coefficients of LSC  in UNet and better improve the  training stability of UNet. Experimental results on CIFAR10, CelebA, ImageNet and COCO show that our methods are superior to stabilize training, and yield about 1.5x training acceleration on different diffusion models with UNet or UViT backbones.

----

## [0] Faster approximate subgraph counts with privacy

**Authors**: *Dung Nguyen, Mahantesh Halappanavar, Venkatesh Srinivasan, Anil Vullikanti*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/deddcfbf08f57489b0088b71a00db640-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/deddcfbf08f57489b0088b71a00db640-Abstract-Conference.html)

**Abstract**:

One of the most common problems studied in the context of differential privacy for graph data is counting the number of non-induced embeddings of a subgraph in a given graph. These counts have very high global sensitivity. Therefore, adding noise based on powerful alternative techniques, such as smooth sensitivity and higher-order local sensitivity have been shown to give significantly better accuracy. However, all these alternatives to global sensitivity become computationally very expensive, and to date efficient polynomial time algorithms are known only for few selected subgraphs, such as triangles, $k$-triangles, and $k$-stars.In this paper, we show that good approximations to these sensitivity metrics can be still used to get private algorithms.Using this approach, we much faster algorithms for privately counting the number of triangles in real-world social networks, which can be easily parallelized.We also give a private polynomial time algorithm for counting any constant size subgraph using less noise than the global sensitivity; we show this can be improved significantly for counting paths in special classes of graphs.

----

## [0] Recasting Continual Learning as Sequence Modeling

**Authors**: *Soochan Lee, Jaehyeon Son, Gunhee Kim*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/dee254cdacbab59f17dc6a8fbdffa59f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/dee254cdacbab59f17dc6a8fbdffa59f-Abstract-Conference.html)

**Abstract**:

In this work, we aim to establish a strong connection between two significant bodies of machine learning research: continual learning and sequence modeling.That is, we propose to formulate continual learning as a sequence modeling problem, allowing advanced sequence models to be utilized for continual learning.Under this formulation, the continual learning process becomes the forward pass of a sequence model.By adopting the meta-continual learning (MCL) framework, we can train the sequence model at the meta-level, on multiple continual learning episodes.As a specific example of our new formulation, we demonstrate the application of Transformers and their efficient variants as MCL methods.Our experiments on seven benchmarks, covering both classification and regression, show that sequence models can be an attractive solution for general MCL.

----

## [0] Multiply Robust Federated Estimation of Targeted Average Treatment Effects

**Authors**: *Larry Han, Zhu Shen, José R. Zubizarreta*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/def4492b32f0248a0e4d92cc46bbdaad-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/def4492b32f0248a0e4d92cc46bbdaad-Abstract-Conference.html)

**Abstract**:

Federated or multi-site studies have distinct advantages over single-site studies, including increased generalizability, the ability to study underrepresented populations, and the opportunity to study rare exposures and outcomes. However, these studies are complicated by the need to preserve the privacy of each individual's data, heterogeneity in their covariate distributions, and different data structures between sites. We propose a novel federated approach to derive valid causal inferences for a target population using multi-site data. We adjust for covariate shift and accommodate covariate mismatch between sites by developing a multiply-robust and privacy-preserving nuisance function estimation approach. Our methodology incorporates transfer learning to estimate ensemble weights to combine information from source sites. We show that these learned weights are efficient and optimal under different scenarios. We showcase the finite sample advantages of our approach in terms of efficiency and robustness compared to existing state-of-the-art approaches. We apply our approach to study the treatment effect of percutaneous coronary intervention (PCI) on the duration of hospitalization for patients experiencing acute myocardial infarction (AMI) with data from the Centers for Medicare \& Medicaid Services (CMS).

----

## [0] Learning Motion Refinement for Unsupervised Face Animation

**Authors**: *Jiale Tao, Shuhang Gu, Wen Li, Lixin Duan*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/df2df463f98abc4de7734dbd0b0dc49d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/df2df463f98abc4de7734dbd0b0dc49d-Abstract-Conference.html)

**Abstract**:

Unsupervised face animation aims to generate a human face video based on theappearance of a source image, mimicking the motion from a driving video. Existingmethods typically adopted a prior-based motion model (e.g., the local affine motionmodel or the local thin-plate-spline motion model). While it is able to capturethe coarse facial motion, artifacts can often be observed around the tiny motionin local areas (e.g., lips and eyes), due to the limited ability of these methodsto model the finer facial motions. In this work, we design a new unsupervisedface animation approach to learn simultaneously the coarse and finer motions. Inparticular, while exploiting the local affine motion model to learn the global coarsefacial motion, we design a novel motion refinement module to compensate forthe local affine motion model for modeling finer face motions in local areas. Themotion refinement is learned from the dense correlation between the source anddriving images. Specifically, we first construct a structure correlation volume basedon the keypoint features of the source and driving images. Then, we train a modelto generate the tiny facial motions iteratively from low to high resolution. Thelearned motion refinements are combined with the coarse motion to generate thenew image. Extensive experiments on widely used benchmarks demonstrate thatour method achieves the best results among state-of-the-art baselines.

----

## [0] Masked Space-Time Hash Encoding for Efficient Dynamic Scene Reconstruction

**Authors**: *Feng Wang, Zilong Chen, Guokang Wang, Yafei Song, Huaping Liu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/df31126302921ca9351fab73923a172f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/df31126302921ca9351fab73923a172f-Abstract-Conference.html)

**Abstract**:

In this paper, we propose the Masked Space-Time Hash encoding (MSTH), a novel method for efficiently reconstructing dynamic 3D scenes from multi-view or monocular videos. Based on the observation that dynamic scenes often contain substantial static areas that result in redundancy in storage and computations, MSTH represents a dynamic scene as a weighted combination of a 3D hash encoding and a 4D hash encoding. The weights for the two components are represented by a learnable mask which is guided by an uncertainty-based objective to reflect the spatial and temporal importance of each 3D position. With this design, our method can reduce the hash collision rate by avoiding redundant queries and modifications on static areas, making it feasible to represent a large number of space-time voxels by hash tables with small size.Besides, without the requirements to fit the large numbers of temporally redundant features independently, our method is easier to optimize and converge rapidly with only twenty minutes of training for a 300-frame dynamic scene. We evaluate our method on extensive dynamic scenes. As a result, MSTH obtains consistently better results than previous state-of-the-art methods with only 20 minutes of training time and 130 MB of memory storage.

----

## [0] Bifurcations and loss jumps in RNN training

**Authors**: *Lukas Eisenmann, Zahra Monfared, Niclas Alexander Göring, Daniel Durstewitz*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/df334022279996b07e0870a629c18857-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/df334022279996b07e0870a629c18857-Abstract-Conference.html)

**Abstract**:

Recurrent neural networks (RNNs) are popular machine learning tools for modeling and forecasting sequential data and for inferring dynamical systems (DS) from observed time series. Concepts from DS theory (DST) have variously been used to further our understanding of both, how trained RNNs solve complex tasks, and the training process itself. Bifurcations are particularly important phenomena in DS, including RNNs, that refer to topological (qualitative) changes in a system's dynamical behavior as one or more of its parameters are varied. Knowing the bifurcation structure of an RNN will thus allow to deduce many of its computational and dynamical properties, like its sensitivity to parameter variations or its behavior during training. In particular, bifurcations may account for sudden loss jumps observed in RNN training that could severely impede the training process. Here we first mathematically prove for a particular class of ReLU-based RNNs that certain bifurcations are indeed associated with loss gradients tending toward infinity or zero. We then introduce a novel heuristic algorithm for detecting all fixed points and $k$-cycles in ReLU-based RNNs and their existence and stability regions, hence bifurcation manifolds in parameter space. In contrast to previous numerical algorithms for finding fixed points and common continuation methods, our algorithm provides $\textit{exact}$ results and returns fixed points and cycles up to high orders with surprisingly good scaling behavior. We exemplify the algorithm on the analysis of the training process of RNNs, and find that the recently introduced technique of generalized teacher forcing completely avoids certain types of bifurcations in training. Thus, besides facilitating the DST analysis of trained RNNs, our algorithm provides a powerful instrument for analyzing the training process itself.

----

## [0] Neural Foundations of Mental Simulation: Future Prediction of Latent Representations on Dynamic Scenes

**Authors**: *Aran Nayebi, Rishi Rajalingham, Mehrdad Jazayeri, Guangyu Robert Yang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/df438caa36714f69277daa92d608dd63-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/df438caa36714f69277daa92d608dd63-Abstract-Conference.html)

**Abstract**:

Humans and animals have a rich and flexible understanding of the physical world, which enables them to infer the underlying dynamical trajectories of objects and events, plausible future states, and use that to plan and anticipate the consequences of actions.However, the neural mechanisms underlying these computations are unclear.We combine a goal-driven modeling approach with dense neurophysiological data and high-throughput human behavioral readouts that contain thousands of comparisons to directly impinge on this question.Specifically, we construct and evaluate several classes of sensory-cognitive networks to predict the future state of rich, ethologically-relevant environments, ranging from self-supervised end-to-end models with pixel-wise or object-slot objectives, to models that future predict in the latent space of purely static image-pretrained or dynamic video-pretrained foundation models.We find that ``scale is \emph{not} all you need'', and that many state-of-the-art machine learning models fail to perform well on our neural and behavioral benchmarks for future prediction.In fact, only one class of models matches these data well overall.We find that neural responses are currently best predicted by models trained to predict the future state of their environment in the \emph{latent} space of pretrained foundation models optimized for \emph{dynamic} scenes in a self-supervised manner.These models also approach the neurons' ability to predict the environmental state variables that are visually hidden from view, despite not being explicitly trained to do so.Finally, we find that not all foundation model latents are equal.Notably, models that future predict in the latent space of video foundation models that are optimized to support a \emph{diverse} range of egocentric sensorimotor tasks, reasonably match \emph{both} human behavioral error patterns and neural dynamics across all environmental scenarios that we were able to test.Overall, these findings suggest that the neural mechanisms and behaviors of primate mental simulation have strong inductive biases associated with them, and are thus far most consistent with being optimized to future predict on \emph{reusable} visual representations that are useful for Embodied AI more generally.

----

## [0] Learning Visual Prior via Generative Pre-Training

**Authors**: *Jinheng Xie, Kai Ye, Yudong Li, Yuexiang Li, Kevin Qinghong Lin, Yefeng Zheng, Linlin Shen, Mike Zheng Shou*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/df4f6e43446b1ee29c5a33d32c279f83-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/df4f6e43446b1ee29c5a33d32c279f83-Abstract-Conference.html)

**Abstract**:

Various stuff and things in visual data possess specific traits, which can be learned by deep neural networks and are implicitly represented as the visual prior, e.g., object location and shape, in the model. Such prior potentially impacts many vision tasks. For example, in conditional image synthesis, spatial conditions failing to adhere to the prior can result in visually inaccurate synthetic results. This work aims to explicitly learn the visual prior and enable the customization of sampling. Inspired by advances in language modeling, we propose to learn Visual prior via Generative Pre-Training, dubbed VisorGPT. By discretizing visual locations, e.g., bounding boxes, human pose, and instance masks, into sequences, VisorGPT can model visual prior through likelihood maximization. Besides, prompt engineering is investigated to unify various visual locations and enable customized sampling of sequential outputs from the learned prior. Experimental results demonstrate the effectiveness of VisorGPT in modeling visual prior and extrapolating to novel scenes, potentially motivating that discrete visual locations can be integrated into the learning paradigm of current language models to further perceive visual world. Code is available at https://sierkinhane.github.io/visor-gpt.

----

## [0] Operator Learning with Neural Fields: Tackling PDEs on General Geometries

**Authors**: *Louis Serrano, Lise Le Boudec, Armand Kassaï Koupaï, Thomas X. Wang, Yuan Yin, Jean-Noël Vittaut, Patrick Gallinari*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/df54302388bbc145aacaa1a54a4a5933-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/df54302388bbc145aacaa1a54a4a5933-Abstract-Conference.html)

**Abstract**:

Machine learning approaches for solving partial differential equations require learning mappings between function spaces. While convolutional or graph neural networks are constrained to discretized functions, neural operators present a promising milestone toward mapping functions directly. Despite impressive results they still face challenges with respect to the domain geometry and typically rely on some form of discretization. In order to alleviate such limitations, we present CORAL, a new method that leverages coordinate-based networks for solving PDEs on general geometries. CORAL is designed to remove constraints on the input mesh, making it applicable to any spatial sampling and geometry. Its ability extends to diverse problem domains, including PDE solving, spatio-temporal forecasting, and inverse problems like geometric design. CORAL demonstrates robust performance across multiple resolutions and performs well in both convex and non-convex domains, surpassing or performing on par with state-of-the-art models.

----

## [0] Learning Dense Flow Field for Highly-accurate Cross-view Camera Localization

**Authors**: *Zhenbo Song, Xianghui Ze, Jianfeng Lu, Yujiao Shi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/df5f94d6ac6e13d830d70536cde9f0d2-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/df5f94d6ac6e13d830d70536cde9f0d2-Abstract-Conference.html)

**Abstract**:

This paper addresses the problem of estimating the 3-DoF camera pose for a ground-level image with respect to a satellite image that encompasses the local surroundings. We propose a novel end-to-end approach that leverages the learning of dense pixel-wise flow fields in pairs of ground and satellite images to calculate the camera pose. Our approach differs from existing methods by constructing the feature metric at the pixel level, enabling full-image supervision for learning distinctive geometric configurations and visual appearances across views. Specifically, our method employs two distinct convolution networks for ground and satellite feature extraction. Then, we project the ground feature map to the bird's eye view (BEV) using a fixed camera height assumption to achieve preliminary geometric alignment. To further establish the content association between the BEV and satellite features, we introduce a residual convolution block to refine the projected BEV feature. Optical flow estimation is performed on the refined BEV feature map and the satellite feature map using flow decoder networks based on RAFT. After obtaining dense flow correspondences, we apply the least square method to filter matching inliers and regress the ground camera pose. Extensive experiments demonstrate significant improvements compared to state-of-the-art methods. Notably, our approach reduces the median localization error by 89\%, 19\%, 80\%, and 35\% on the KITTI, Ford multi-AV, VIGOR, and Oxford RobotCar datasets, respectively.

----

## [0] Spatially Resolved Gene Expression Prediction from Histology Images via Bi-modal Contrastive Learning

**Authors**: *Ronald Xie, Kuan Pang, Sai Chung, Catia Perciani, Sonya MacParland, Bo Wang, Gary D. Bader*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/df656d6ed77b565e8dcdfbf568aead0a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/df656d6ed77b565e8dcdfbf568aead0a-Abstract-Conference.html)

**Abstract**:

Histology imaging is an important tool in medical diagnosis and research, enabling the examination of tissue structure and composition at the microscopic level. Understanding the underlying molecular mechanisms of tissue architecture is critical in uncovering disease mechanisms and developing effective treatments.Gene expression profiling provides insight into the molecular processes underlying tissue architecture, but the process can be time-consuming and expensive. We present BLEEP (Bi-modaL Embedding for Expression Prediction), a bi-modal embedding framework capable of generating spatially resolved gene expression profiles of whole-slide Hematoxylin and eosin (H&E) stained histology images. BLEEP uses contrastive learning to construct a low-dimensional joint embedding space from a reference dataset using paired image and expression profiles at micrometer resolution. With this approach, the gene expression of any query image patch can be imputed using the expression profiles from the reference dataset. We demonstrate BLEEPâ€™s effectiveness in gene expression prediction by benchmarking its performance on a human liver tissue dataset captured using the 10x Visium platform, where it achieves significant improvements over existing methods. Our results demonstrate the potential of BLEEP to provide insights into the molecular mechanisms underlying tissue architecture, with important implications in diagnosis and research of various diseases. The proposed approach can significantly reduce the time and cost associated with gene expression profiling, opening up new avenues for high-throughput analysis of histology images for both research and clinical applications.

----

## [0] Causal-structure Driven Augmentations for Text OOD Generalization

**Authors**: *Amir Feder, Yoav Wald, Claudia Shi, Suchi Saria, David M. Blei*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/df88b275bef31ac96c85f0c4013734fc-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/df88b275bef31ac96c85f0c4013734fc-Abstract-Conference.html)

**Abstract**:

The reliance of text classifiers on spurious correlations can lead to poor generalization at deployment, raising concerns about their use in safety-critical domains such as healthcare. In this work, we propose to use counterfactual data augmentation, guided by knowledge of the causal structure of the data, to simulate interventions on spurious features and to learn more robust text classifiers. We show that this strategy is appropriate in prediction problems where the label is spuriously correlated with an attribute. Under the assumptions of such problems, we discuss the favorable sample complexity of counterfactual data augmentation, compared to importance re-weighting. Pragmatically, we match examples using auxiliary data, based on diff-in-diff methodology, and use a large language model (LLM) to represent a conditional probability of text. Through extensive experimentation on learning caregiver-invariant predictors of clinical diagnoses from medical narratives and on semi-synthetic data, we demonstrate that our method for simulating interventions improves out-of-distribution (OOD) accuracy compared to baseline invariant learning algorithms.

----

## [0] Adversarial Counterfactual Environment Model Learning

**Authors**: *Xiong-Hui Chen, Yang Yu, Zhengmao Zhu, Zhihua Yu, Zhenjun Chen, Chenghe Wang, Yinan Wu, Rong-Jun Qin, Hongqiu Wu, Ruijin Ding, Fangsheng Huang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/df927a06a0d9f5f06d9cd4a91ce58e56-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/df927a06a0d9f5f06d9cd4a91ce58e56-Abstract-Conference.html)

**Abstract**:

An accurate environment dynamics model is crucial for various downstream tasks in sequential decision-making, such as counterfactual prediction, off-policy evaluation, and offline reinforcement learning. Currently, these models were learned through empirical risk minimization (ERM) by step-wise fitting of historical transition data. This way was previously believed unreliable over long-horizon rollouts because of the compounding errors, which can lead to uncontrollable inaccuracies in predictions. In this paper, we find that the challenge extends beyond just long-term prediction errors: we reveal that even when planning with one step, learned dynamics models can also perform poorly due to the selection bias of behavior policies during data collection. This issue will significantly mislead the policy optimization process even in identifying single-step optimal actions, further leading to a greater risk in sequential decision-making scenarios.To tackle this problem, we introduce a novel model-learning objective called adversarial weighted empirical risk minimization (AWRM).  AWRM incorporates an adversarial policy that exploits the model to generate a data distribution that weakens the model's prediction accuracy, and subsequently, the model is learned under this adversarial data distribution.We implement a practical algorithm, GALILEO, for AWRM and evaluate it on two synthetic tasks, three continuous-control tasks, and  \textit{a real-world application}. The experiments demonstrate that GALILEO can accurately predict counterfactual actions and improve various downstream tasks, including offline policy evaluation and improvement, as well as online decision-making.

----

## [0] Finding Safe Zones of Markov Decision Processes Policies

**Authors**: *Lee Cohen, Yishay Mansour, Michal Moshkovitz*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/dfaa29ed28dfa175bcc5e2a54aa199f8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/dfaa29ed28dfa175bcc5e2a54aa199f8-Abstract-Conference.html)

**Abstract**:

Given a policy of a Markov Decision Process, we define a SafeZone as a subset of states, such that most of the policy's trajectories are confined to this subset. The quality of a SafeZone is parameterized by the number of states and the escape probability, i.e., the probability that a random trajectory will leave the subset. SafeZones are especially interesting when they have a small number of states and low escape probability. We study the complexity of finding optimal SafeZones, and show that in general, the problem is computationally hard. For this reason, we concentrate on finding approximate SafeZones. Our main result is a bi-criteria approximation learning algorithm with a factor of almost $2$  approximation for both the escape probability and \newprob size, using a polynomial size sample complexity.

----

## [0] Zero-One Laws of Graph Neural Networks

**Authors**: *Sam Adam-Day, Theodor-Mihai Iliant, Ismail Ilkan Ceylan*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/dfba85bc32a3cb63a96d1412062b4d8e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/dfba85bc32a3cb63a96d1412062b4d8e-Abstract-Conference.html)

**Abstract**:

Graph neural networks (GNNs) are the de facto standard deep learning architectures for machine learning on graphs.  This has led to a large body of work analyzing the capabilities and limitations of these models,  particularly pertaining to their representation and extrapolation capacity.  We offer a novel theoretical perspective on the representation and extrapolation capacity of GNNs, by answering the question: how do GNNs behave as the number of graph nodes become very large? Under mild assumptions, we show that when we draw graphs of increasing size from the Erdős–Rényi model, the probability that such graphs are mapped to a particular output by a class of GNN classifiers tends to either zero or one. This class includes the popular graph convolutional network architecture. The result establishes `zero-one laws' for these GNNs, and analogously to other convergence laws,  entails theoretical limitations on their capacity.  We empirically verify our results, observing that the theoretical asymptotic limits are evident already on relatively small graphs.

----

## [0] Towards Revealing the Mystery behind Chain of Thought: A Theoretical Perspective

**Authors**: *Guhao Feng, Bohang Zhang, Yuntian Gu, Haotian Ye, Di He, Liwei Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/dfc310e81992d2e4cedc09ac47eff13e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/dfc310e81992d2e4cedc09ac47eff13e-Abstract-Conference.html)

**Abstract**:

Recent studies have discovered that Chain-of-Thought prompting (CoT) can dramatically improve the performance of Large Language Models (LLMs), particularly when dealing with complex tasks involving mathematics or reasoning. Despite the enormous empirical success, the underlying mechanisms behind CoT and how it unlocks the potential of LLMs remain elusive. In this paper, we take a first step towards theoretically answering these questions. Specifically, we examine the expressivity of LLMs with CoT in solving fundamental mathematical and decision-making problems. By using circuit complexity theory, we first give impossibility results showing that bounded-depth Transformers are unable to directly produce correct answers for basic arithmetic/equation tasks unless the model size grows super-polynomially with respect to the input length. In contrast, we then prove by construction that autoregressive Transformers of constant size suffice to solve both tasks by generating CoT derivations using a commonly used math language format. Moreover, we show LLMs with CoT can handle a general class of decision-making problems known as Dynamic Programming, thus justifying their power in tackling complex real-world tasks. Finally, an extensive set of experiments show that, while Transformers always fail to directly predict the answers, they can consistently learn to generate correct solutions step-by-step given sufficient CoT demonstrations.

----



[Go to the previous page](NIPS-2023-list3000.md)

[Go to the next page](NIPS-2023-list3002.md)

[Go to the catalog section](README.md)