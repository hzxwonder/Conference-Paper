## [3000] Individual Arbitrariness and Group Fairness

**Authors**: *Carol Xuan Long, Hsiang Hsu, Wael Alghamdi, Flávio P. Calmon*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d891d240b5784656a0356bf4b00f5cdd-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d891d240b5784656a0356bf4b00f5cdd-Abstract-Conference.html)

**Abstract**:

Machine learning tasks may admit multiple competing models that achieve similar performance yet produce conflicting outputs for individual samples---a phenomenon known as predictive multiplicity. We demonstrate that fairness interventions in machine learning optimized solely for group fairness and accuracy can exacerbate predictive multiplicity. Consequently, state-of-the-art fairness interventions can mask high predictive multiplicity behind favorable group fairness and accuracy metrics. We argue that a third axis of ``arbitrariness'' should be considered  when deploying models to aid decision-making in applications of individual-level impact.To address this challenge, we propose an ensemble  algorithm applicable to any fairness intervention that provably ensures  more consistent predictions.

----

## [3001] ASPEN: Breaking Operator Barriers for Efficient Parallelization of Deep Neural Networks

**Authors**: *Jongseok Park, Kyungmin Bin, Gibum Park, Sangtae Ha, Kyunghan Lee*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d899a31938c7838965b589d9b14a5ca6-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d899a31938c7838965b589d9b14a5ca6-Abstract-Conference.html)

**Abstract**:

Modern Deep Neural Network (DNN) frameworks use tensor operators as the main building blocks of DNNs. However, we observe that operator-based construction of DNNs incurs significant drawbacks in parallelism in the form of synchronization barriers. Synchronization barriers of operators confine the scope of parallel computation to each operator and obscure the rich parallel computation opportunities that exist across operators. To this end, we present ASPEN, a novel parallel computation solution for DNNs that achieves fine-grained dynamic execution of DNNs, which (1) removes the operator barriers and expresses DNNs in dataflow graphs of fine-grained tiles to expose the parallel computation opportunities across operators, and (2) exploits these opportunities by dynamically locating and scheduling them in runtime. This novel approach of ASPEN enables opportunistic parallelism, a new class of parallelism for DNNs that is unavailable in the existing operator-based approaches. ASPEN also achieves high resource utilization and memory reuse by letting each resource asynchronously traverse depthwise in the DNN graph to its full computing potential. We provide challenges and solutions to our approach and show that our proof-of-concept implementation of ASPEN on CPU shows exceptional performance, outperforming state-of-the-art inference systems of TorchScript and TVM by up to 3.2$\times$ and 4.3$\times$, respectively.

----

## [3002] Parallel Submodular Function Minimization

**Authors**: *Deeparnab Chakrabarty, Andrei Graur, Haotian Jiang, Aaron Sidford*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d8a7f2f7e346410e8ac7b39d9ff28c4a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d8a7f2f7e346410e8ac7b39d9ff28c4a-Abstract-Conference.html)

**Abstract**:

We consider the parallel complexity of submodular function minimization (SFM).     We provide a pair of methods which obtain two new query versus depth trade-offs a submodular function defined on subsets of $n$ elements that has integer values between $-M$ and $M$. The first method has depth $2$ and query complexity $n^{O(M)}$ and the second method has depth $\widetilde{O}(n^{1/3} M^{2/3})$ and query complexity $O(\mathrm{poly}(n, M))$. Despite a line of work on improved parallel lower bounds for SFM, prior to our work the only known algorithms for parallel SFM either followed from more general methods for sequential SFM or highly-parallel minimization of convex $\ell_2$-Lipschitz functions. Interestingly, to obtain our second result we provide the first highly-parallel algorithm for minimizing $\ell_\infty$-Lipschitz function over the hypercube which obtains near-optimal depth for obtaining constant accuracy.

----

## [3003] Emergent Communication for Rules Reasoning

**Authors**: *Yuxuan Guo, Yifan Hao, Rui Zhang, Enshuai Zhou, Zidong Du, Xishan Zhang, Xinkai Song, Yuanbo Wen, Yongwei Zhao, Xuehai Zhou, Jiaming Guo, Qi Yi, Shaohui Peng, Di Huang, Ruizhi Chen, Qi Guo, Yunji Chen*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d8ace30c68b085556ccce04ed4ae4ebb-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d8ace30c68b085556ccce04ed4ae4ebb-Abstract-Conference.html)

**Abstract**:

Research on emergent communication between deep-learning-based agents has received extensive attention due to its inspiration for linguistics and artificial intelligence.   However, previous attempts have hovered around emerging communication under perception-oriented environmental settings,  that forces agents to describe low-level perceptual features intra image or symbol contexts.  In this work, inspired by the classic human reasoning test (namely Raven's Progressive Matrix), we propose the Reasoning Game, a cognition-oriented environment that encourages agents to reason and communicate high-level rules, rather than perceived low-level contexts.  Moreover, we propose 1) an unbiased dataset (namely rule-RAVEN) as a benchmark to avoid overfitting, 2) and a two-stage curriculum agent training method as a baseline for more stable convergence in the Reasoning Game,  where contexts and semantics are bilaterally drifting.  Experimental results show that, in the Reasoning Game, a semantically stable and compositional language emerges to solve reasoning problems.  The emerged language helps agents apply the extracted rules to the generalization of unseen context attributes, and to the transfer between different context attributes or even tasks.

----

## [3004] A Regularized Conditional GAN for Posterior Sampling in Image Recovery Problems

**Authors**: *Matthew Bendel, Rizwan Ahmad, Philip Schniter*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d8b29f07599fecdba93d87ed27a65524-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d8b29f07599fecdba93d87ed27a65524-Abstract-Conference.html)

**Abstract**:

In image recovery problems, one seeks to infer an image from distorted, incomplete, and/or noise-corrupted measurements.Such problems arise in magnetic resonance imaging (MRI), computed tomography, deblurring, super-resolution, inpainting, phase retrieval, image-to-image translation, and other applications. Given a training set of signal/measurement pairs, we seek to do more than just produce one good image estimate. Rather, we aim to rapidly and accurately sample from the posterior distribution. To do this,we propose a regularized conditional Wasserstein GAN that generates dozens of high-quality posterior samples per second. Our regularization comprises an $\ell_1$ penalty and an adaptively weighted standard-deviation reward. Using quantitative evaluation metrics like conditional Fr√©chet inception distance, we demonstrate that our method produces state-of-the-art posterior samples in both multicoil MRI and large-scale inpainting applications. The code for our model can be found here: https://github.com/matt-bendel/rcGAN.

----

## [3005] Would I have gotten that reward? Long-term credit assignment by counterfactual contribution analysis

**Authors**: *Alexander Meulemans, Simon Schug, Seijin Kobayashi, Nathaniel Daw, Gregory Wayne*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d8bd445c2abe1343cce0e14b361b2fb3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d8bd445c2abe1343cce0e14b361b2fb3-Abstract-Conference.html)

**Abstract**:

To make reinforcement learning more sample efficient, we need better credit assignment methods that measure an action’s influence on future rewards. Building upon Hindsight Credit Assignment (HCA), we introduce Counterfactual Contribution Analysis (COCOA), a new family of model-based credit assignment algorithms. Our algorithms achieve precise credit assignment by measuring the contribution of actions upon obtaining subsequent rewards, by quantifying a counterfactual query: ‘Would the agent still have reached this reward if it had taken another action?’. We show that measuring contributions w.r.t. rewarding states, as is done in HCA, results in spurious estimates of contributions, causing HCA to degrade towards the high-variance REINFORCE estimator in many relevant environments. Instead, we measure contributions w.r.t. rewards or learned representations of the rewarding objects, resulting in gradient estimates with lower variance. We run experiments on a suite of problems specifically designed to evaluate long-term credit assignment capabilities. By using dynamic programming, we measure ground-truth policy gradients and show that the improved performance of our new model-based credit assignment methods is due to lower bias and variance compared to HCA and common baselines. Our results demonstrate how modeling action contributions towards rewarding outcomes can be leveraged for credit assignment, opening a new path towards sample-efficient reinforcement learning.

----

## [3006] HOH: Markerless Multimodal Human-Object-Human Handover Dataset with Large Object Count

**Authors**: *Noah Wiederhold, Ava Megyeri, DiMaggio Paris, Sean Banerjee, Natasha Banerjee*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d8c6a37c4c94e9a63e53d296f1f668ae-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/d8c6a37c4c94e9a63e53d296f1f668ae-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

We present the HOH (Human-Object-Human) Handover Dataset, a large object count dataset with 136 objects, to accelerate data-driven research on handover studies, human-robot handover implementation, and artificial intelligence (AI) on handover parameter estimation from 2D and 3D data of two-person interactions. HOH contains multi-view RGB and depth data, skeletons, fused point clouds, grasp type and handedness labels, object, giver hand, and receiver hand 2D and 3D segmentations, giver and receiver comfort ratings, and paired object metadata and aligned 3D models for 2,720 handover interactions spanning 136 objects and 20 giver-receiver pairs—40 with role-reversal—organized from 40 participants. We also show experimental results of neural networks trained using HOH to perform grasp, orientation, and trajectory prediction. As the only fully markerless handover capture dataset, HOH represents natural human-human handover interactions, overcoming challenges with markered datasets that require specific suiting for body tracking, and lack high-resolution hand tracking. To date, HOH is the largest handover dataset in terms of object count, participant count, pairs with role reversal accounted for, and total interactions captured.

----

## [3007] Tight Risk Bounds for Gradient Descent on Separable Data

**Authors**: *Matan Schliserman, Tomer Koren*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d8ca28a32c05cd3b9b0940e43720f31b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d8ca28a32c05cd3b9b0940e43720f31b-Abstract-Conference.html)

**Abstract**:

We study the generalization properties of unregularized gradient methods applied to separable linear classification---a setting that has received considerable attention since the pioneering work of Soudry et al. (2018).We establish tight upper and lower (population) risk bounds for gradient descent in this setting, for any smooth loss function, expressed in terms of its tail decay rate.Our bounds take the form $\Theta(r_{\ell,T}^2 / \gamma^2 T + r_{\ell,T}^2 / \gamma^2 n)$, where $T$ is the number of gradient steps, $n$ is size of the training set, $\gamma$ is the data margin, and $r_{\ell,T}$ is a complexity term that depends on the tail decay rate of the loss function (and on $T$).Our upper bound greatly improves the existing risk bounds due to Shamir (2021) and Schliserman and Koren (2022), that either applied to specific loss functions or imposed extraneous technical assumptions, and applies to virtually any convex and smooth loss function.Our risk lower bound is the first in this context and establish the tightness of our general upper bound for any given tail decay rate and in all parameter regimes.The proof technique used to show these results is also markedly simpler compared to previous work, and is straightforward to extend to other gradient methods; we illustrate this by providing analogous results for Stochastic Gradient Descent.

----

## [3008] Video Prediction Models as Rewards for Reinforcement Learning

**Authors**: *Alejandro Escontrela, Ademi Adeniji, Wilson Yan, Ajay Jain, Xue Bin Peng, Ken Goldberg, Youngwoon Lee, Danijar Hafner, Pieter Abbeel*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d9042abf40782fbce28901c1c9c0e8d8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d9042abf40782fbce28901c1c9c0e8d8-Abstract-Conference.html)

**Abstract**:

Specifying reward signals that allow agents to learn complex behaviors is a long-standing challenge in reinforcement learning.A promising approach is to extract preferences for behaviors from unlabeled videos, which are widely available on the internet. We present Video Prediction Rewards (VIPER), an algorithm that leverages pretrained video prediction models as action-free reward signals for reinforcement learning. Specifically, we first train an autoregressive transformer on expert videos and then use the video prediction likelihoods as reward signals for a reinforcement learning agent. VIPER enables expert-level control without programmatic task rewards across a wide range of DMC, Atari, and RLBench tasks. Moreover, generalization of the video prediction model allows us to derive rewards for an out-of-distribution environment where no expert data is available, enabling cross-embodiment generalization for tabletop manipulation. We see our work as starting point for scalable reward specification from unlabeled videos that will benefit from the rapid advances in generative modeling. Source code and datasets are available on the project website: https://ViperRL.com

----

## [3009] Provably (More) Sample-Efficient Offline RL with Options

**Authors**: *Xiaoyan Hu, Ho-fung Leung*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d91b532a76ea98ac1ef5226b862bfc49-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d91b532a76ea98ac1ef5226b862bfc49-Abstract-Conference.html)

**Abstract**:

The options framework yields empirical success in long-horizon planning problems of reinforcement learning (RL). Recent works show that options help improve the sample efficiency in online RL. However, these results are no longer applicable to scenarios where exploring the environment online is risky, e.g., automated driving and healthcare. In this paper, we provide the first analysis of the sample complexity for offline RL with options, where the agent learns from a dataset without further interaction with the environment. We derive a novel information-theoretic lower bound, which generalizes the one for offline learning with actions. We propose the PEssimistic Value Iteration for Learning with Options (PEVIO) algorithm and establish near-optimal suboptimality bounds for two popular data-collection procedures, where the first one collects state-option transitions and the second one collects state-action transitions. We show that compared to offline RL with actions, using options not only enjoys a faster finite-time convergence rate (to the optimal value) but also attains a better performance when either the options are carefully designed or the offline data is limited. Based on these results, we analyze the pros and cons of the data-collection procedures.

----

## [3010] Rewrite Caption Semantics: Bridging Semantic Gaps for Language-Supervised Semantic Segmentation

**Authors**: *Yun Xing, Jian Kang, Aoran Xiao, Jiahao Nie, Ling Shao, Shijian Lu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d937cb3fe2851ed0ab9af5e38f885077-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d937cb3fe2851ed0ab9af5e38f885077-Abstract-Conference.html)

**Abstract**:

Vision-Language Pre-training has demonstrated its remarkable zero-shot recognition ability and potential to learn generalizable visual representations from languagesupervision. Taking a step ahead, language-supervised semantic segmentation enables spatial localization of textual inputs by learning pixel grouping solely from image-text pairs. Nevertheless, the state-of-the-art suffers from a clear semantic gap between visual and textual modalities: plenty of visual concepts appeared in images are missing in their paired captions. Such semantic misalignment circulates in pre-training, leading to inferior zero-shot performance in dense predictions due to insufficient visual concepts captured in textual representations. To close such semantic gap, we propose Concept Curation (CoCu), a pipeline that leverages CLIP to compensate for the missing semantics. For each image-text pair, we establish a concept archive that maintains potential visually-matched concepts with our proposed vision-driven expansion and text-to-vision-guided ranking. Relevant concepts can thus be identified via cluster-guided sampling and fed into pre-training, thereby bridging the gap between visual and textual semantics. Extensive experiments over a broad suite of 8 segmentation benchmarks show that CoCu achieves superb zero-shot transfer performance and greatly boosts language-supervised segmentation baseline by a large margin, suggesting the value of closing semantic gap in pre-training data.

----

## [3011] Approximate Allocation Matching for Structural Causal Bandits with Unobserved Confounders

**Authors**: *Lai Wei, Muhammad Qasim Elahi, Mahsa Ghasemi, Murat Kocaoglu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d938b739ac250e22729cc26e6176f65e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d938b739ac250e22729cc26e6176f65e-Abstract-Conference.html)

**Abstract**:

Structural causal bandit provides a framework for online decision-making problems when causal information is available. It models the stochastic environment with a structural causal model (SCM) that governs the causal relations between random variables. In each round, an agent applies an intervention (or no intervention) by setting certain variables to some constants and receives a stochastic reward from a non-manipulable variable. Though the causal structure is given, the observational and interventional distributions of these random variables are unknown beforehand, and they can only be learned through interactions with the environment. Therefore, to maximize the expected cumulative reward, it is critical to balance the explore-versus-exploit tradeoff. We assume each random variable takes a finite number of distinct values, and consider a semi-Markovian setting, where random variables are affected by unobserved confounders. Using the canonical SCM formulation to discretize the domains of unobserved variables, we efficiently integrate samples to reduce model uncertainty. This gives the decision maker a natural advantage over those in a classical multi-armed bandit setup. We provide a logarithmic asymptotic regret lower bound for the structural causal bandit problem. Inspired by the lower bound, we design an algorithm that can utilize the causal structure to accelerate the learning process and take informative and rewarding interventions. We establish that our algorithm achieves a logarithmic regret and demonstrate that it outperforms the existing methods via simulations.

----

## [3012] Human-Guided Complexity-Controlled Abstractions

**Authors**: *Andi Peng, Mycal Tucker, Eoin M. Kenny, Noga Zaslavsky, Pulkit Agrawal, Julie A. Shah*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d94b46ec30adee2bbb134f813fc9dde0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d94b46ec30adee2bbb134f813fc9dde0-Abstract-Conference.html)

**Abstract**:

Neural networks often learn task-specific latent representations that fail to generalize to novel settings or tasks. Conversely, humans learn discrete representations (i.e., concepts or words) at a variety of abstraction levels (e.g., "bird" vs. "sparrow'") and use the appropriate abstraction based on tasks. Inspired by this, we train neural models to generate a spectrum of discrete representations, and control the complexity of the representations (roughly, how many bits are allocated for encoding inputs) by tuning the entropy of the distribution over representations. In finetuning experiments, using only a small number of labeled examples for a new task, we show that (1) tuning the representation to a task-appropriate complexity level supports the greatest finetuning performance, and (2) in a human-participant study, users were able to identify the appropriate complexity level for a downstream task via visualizations of discrete representations. Our results indicate a promising direction for rapid model finetuning by leveraging human insight.

----

## [3013] Scenario Diffusion: Controllable Driving Scenario Generation With Diffusion

**Authors**: *Ethan Pronovost, Meghana Reddy Ganesina, Noureldin Hendy, Zeyu Wang, Andres Morales, Kai Wang, Nick Roy*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d95cb79a3421e6d9b6c9a9008c4d07c5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d95cb79a3421e6d9b6c9a9008c4d07c5-Abstract-Conference.html)

**Abstract**:

Automated creation of synthetic traffic scenarios is a key part of scaling the safety validation of autonomous vehicles (AVs). In this paper, we propose Scenario Diffusion, a novel diffusion-based architecture for generating traffic scenarios that enables controllable scenario generation. We combine latent diffusion, object detection and trajectory regression to generate distributions of synthetic agent poses, orientations and trajectories simultaneously. This distribution is conditioned on the map and sets of tokens describing the desired scenario to provide additional control over the generated scenario. We show that our approach has sufficient expressive capacity to model diverse traffic patterns and generalizes to different geographical regions.

----

## [3014] Label-Only Model Inversion Attacks via Knowledge Transfer

**Authors**: *Ngoc-Bao Nguyen, Keshigeyan Chandrasegaran, Milad Abdollahzadeh, Ngai-Man Cheung*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d9827e811c5a205c1313fb950c072c7d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d9827e811c5a205c1313fb950c072c7d-Abstract-Conference.html)

**Abstract**:

In a model inversion (MI) attack, an adversary abuses access to a machine learning (ML) model to infer and reconstruct private training data. Remarkable progress has been made in the white-box and black-box setups, where the adversary has access to the complete model or the model's soft output respectively. However, there is very limited study in the most challenging but practically important setup: Label-only MI attacks, where the adversary only has access to the model's predicted  label (hard label) without confidence scores nor any other model information.  In this work, we propose LOKT, a novel approach for label-only MI attacks. Our idea is based on transfer of knowledge from the opaque target model to  surrogate models. Subsequently, using these surrogate models, our approach can harness advanced white-box attacks. We propose knowledge transfer based on generative modelling, and introduce a new model, Target model-assisted ACGAN (T-ACGAN), for effective knowledge transfer. Our method casts the challenging label-only MI into the more tractable white-box setup. We provide analysis to support that surrogate models based on our approach serve as effective proxies for the target model for MI. Our experiments show that our method significantly outperforms existing SOTA Label-only MI attack by more than 15% across all MI benchmarks. Furthermore, our method compares favorably in terms of query budget. Our study highlights rising privacy threats for  ML models even when minimal information (i.e.,  hard labels) is exposed. Our study highlights rising privacy threats for  ML models even when minimal information (i.e.,  hard labels) is exposed. Our code, demo, models and reconstructed data are available at our project page:https://ngoc-nguyen-0.github.io/lokt/

----

## [3015] On the Adversarial Robustness of Out-of-distribution Generalization Models

**Authors**: *Xin Zou, Weiwei Liu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d9888cc7baa04c2e44e8115588133515-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d9888cc7baa04c2e44e8115588133515-Abstract-Conference.html)

**Abstract**:

Out-of-distribution (OOD) generalization has attracted increasing research attention in recent years, due to its promising experimental results in real-world applications. Interestingly, we find that existing OOD generalization methods are vulnerable to adversarial attacks. This motivates us to study OOD adversarial robustness. We first present theoretical analyses of OOD adversarial robustness in two different complementary settings. Motivated by the theoretical results, we design two algorithms to improve the OOD adversarial robustness. Finally, we conduct experiments to validate the effectiveness of our proposed algorithms.

----

## [3016] Utilitarian Algorithm Configuration

**Authors**: *Devon R. Graham, Kevin Leyton-Brown, Tim Roughgarden*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d98d9cef0c189f1db95f1d94652f7051-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d98d9cef0c189f1db95f1d94652f7051-Abstract-Conference.html)

**Abstract**:

We present the first nontrivial procedure for configuring heuristic algorithms to maximize the utility provided to their end users while also offering theoretical guarantees about performance. Existing procedures seek configurations that minimize expected runtime. However, very recent theoretical work argues that expected runtime minimization fails to capture algorithm designers' preferences. Here we show that the utilitarian objective also confers significant algorithmic benefits. Intuitively, this is because mean runtime is dominated by extremely long runs even when they are incredibly rare; indeed, even when an algorithm never gives rise to such long runs, configuration procedures that provably minimize mean runtime must perform a huge number of experiments to demonstrate this fact. In contrast, utility is bounded and monotonically decreasing in runtime, allowing for meaningful empirical bounds on a configuration's performance. This paper builds on this idea to describe effective and theoretically sound configuration procedures. We prove upper bounds on the runtime of these procedures that are similar to theoretical lower bounds, while also demonstrating their performance empirically.

----

## [3017] Double Randomized Underdamped Langevin with Dimension-Independent Convergence Guarantee

**Authors**: *Yuanshi Liu, Cong Fang, Tong Zhang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d9af4d6ac714626b652da5616ca71f99-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d9af4d6ac714626b652da5616ca71f99-Abstract-Conference.html)

**Abstract**:

This paper focuses on the high-dimensional sampling of log-concave distributions with composite structures: $p^*(\mathrm{d}x)\propto \exp(-g(x)-f(x))\mathrm{d}x$. We develop a double randomization technique, which leads to a fast underdamped Langevin algorithm with a dimension-independent convergence guarantee. We prove that the algorithm enjoys an overall $\tilde{\mathcal{O}}\left(\frac{\left(\mathrm{tr}(H)\right)^{1/3}}{\epsilon^{2/3}}\right)$ iteration complexity to reach an $\epsilon$-tolerated sample whose distribution $p$ admits $W_2(p,p^*)\leq \epsilon$.  Here,  $H$ is an upper bound of the Hessian matrices for $f$ and does not explicitly depend on dimension $d$. For the posterior sampling over linear models with normalized data, we show a clear superiority of convergence rate which is dimension-free and outperforms the previous best-known results by a $d^{1/3}$ factor. The analysis to achieve a faster convergence rate brings new insights into high-dimensional sampling.

----

## [3018] Non-Asymptotic Analysis of a UCB-based Top Two Algorithm

**Authors**: *Marc Jourdan, Rémy Degenne*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d9b564716709357b4bccec9fc9ad04d2-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d9b564716709357b4bccec9fc9ad04d2-Abstract-Conference.html)

**Abstract**:

A Top Two sampling rule for bandit identification is a method which selects the next arm to sample from among two candidate arms, a leader and a challenger. Due to their simplicity and good empirical performance, they have received increased attention in recent years. However, for fixed-confidence best arm identification, theoretical guarantees for Top Two methods have only been obtained in the asymptotic regime, when the error level vanishes. In this paper, we derive the first non-asymptotic upper bound on the expected sample complexity of a Top Two algorithm, which holds for any error level. Our analysis highlights sufficient properties for a regret minimization algorithm to be used as leader. These properties are satisfied by the UCB algorithm, and our proposed UCB-based Top Two algorithm simultaneously enjoys non-asymptotic guarantees and competitive empirical performance.

----

## [3019] Statistical and Computational Trade-off in Multi-Agent Multi-Armed Bandits

**Authors**: *Filippo Vannella, Alexandre Proutière, Jaeseong Jeong*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d9c7c8bd6ad4cebb7d006e5109e0b682-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d9c7c8bd6ad4cebb7d006e5109e0b682-Abstract-Conference.html)

**Abstract**:

We study the problem of regret minimization in Multi-Agent Multi-Armed Bandits (MAMABs) where the rewards are defined through a factor graph. We derive an instance-specific regret lower bound and characterize the minimal expected number of times each global action should be explored. Unfortunately, this bound and the corresponding optimal exploration process are obtained by solving a combinatorial optimization problem with a set of variables and constraints exponentially growing with the number of agents. We approximate the regret lower bound problem via Mean Field techniques to reduce the number of variables and constraints. By tuning the latter, we explore the trade-off between achievable regret and complexity. We devise Efficient Sampling for MAMAB (ESM), an algorithm whose regret asymptotically matches the corresponding approximated lower bound. We assess the regret and computational complexity of ESM numerically, using both synthetic and real-world experiments in radio communications networks.

----

## [3020] On permutation symmetries in Bayesian neural network posteriors: a variational perspective

**Authors**: *Simone Rossi, Ankit Singh, Thomas Hannagan*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d9dc5573f7368201d6409e07e882aa77-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d9dc5573f7368201d6409e07e882aa77-Abstract-Conference.html)

**Abstract**:

The elusive nature of gradient-based optimization in neural networks is tied to their loss landscape geometry, which is poorly understood. However recent work has brought solid evidence that there is essentially no loss barrier between the local solutions of gradient descent, once accounting for weight-permutations that leave the network's computation unchanged. This raises questions for approximate inference in Bayesian neural networks (BNNs), where we are interested in marginalizing over multiple points in the loss landscape.In this work, we first extend the formalism of marginalized loss barrier and solution interpolation to BNNs, before proposing a matching algorithm to search for linearly connected solutions. This is achieved by aligning the distributions of two independent approximate Bayesian solutions with respect to permutation matrices. Building on the work of Ainsworth et al. (2023), we frame the problem as a combinatorial optimization one, using an approximation to the sum of bilinear assignment problem. We then experiment on a variety of architectures and datasets, finding nearly zero marginalized loss barriers for linearly connected solutions.

----

## [3021] Hierarchical Decomposition of Prompt-Based Continual Learning: Rethinking Obscured Sub-optimality

**Authors**: *Liyuan Wang, Jingyi Xie, Xingxing Zhang, Mingyi Huang, Hang Su, Jun Zhu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d9f8b5abc8e0926539ecbb492af7b2f1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d9f8b5abc8e0926539ecbb492af7b2f1-Abstract-Conference.html)

**Abstract**:

Prompt-based continual learning is an emerging direction in leveraging pre-trained knowledge for downstream continual learning, and has almost reached the performance pinnacle under supervised pre-training. However, our empirical research reveals that the current strategies fall short of their full potential under the more realistic self-supervised pre-training, which is essential for handling vast quantities of unlabeled data in practice. This is largely due to the difficulty of task-specific knowledge being incorporated into instructed representations via prompt parameters and predicted by uninstructed representations at test time. To overcome the exposed sub-optimality, we conduct a theoretical analysis of the continual learning objective in the context of pre-training, and decompose it into hierarchical components: within-task prediction, task-identity inference, and task-adaptive prediction. Following these empirical and theoretical insights, we propose Hierarchical Decomposition (HiDe-)Prompt, an innovative approach that explicitly optimizes the hierarchical components with an ensemble of task-specific prompts and statistics of both uninstructed and instructed representations, further with the coordination of a contrastive regularization strategy. Our extensive experiments demonstrate the superior performance of HiDe-Prompt and its robustness to pre-training paradigms in continual learning (e.g., up to 15.01% and 9.61% lead on Split CIFAR-100 and Split ImageNet-R, respectively).

----

## [3022] 3D molecule generation by denoising voxel grids

**Authors**: *Pedro O. Pinheiro, Joshua Rackers, Joseph Kleinhenz, Michael Maser, Omar Mahmood, Andrew M. Watkins, Stephen Ra, Vishnu Sresht, Saeed Saremi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/da1131a86ac3c70e0b7cae89c3d4df22-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/da1131a86ac3c70e0b7cae89c3d4df22-Abstract-Conference.html)

**Abstract**:

We propose a new score-based approach to generate 3D molecules represented as atomic densities on regular grids.First, we train a denoising neural network that learns to map from a smooth distribution of noisy molecules to the distribution of real molecules.Then, we follow the neural empirical Bayes framework [Saremi and Hyvarinen, 2019] and generate molecules in two steps: (i) sample noisy density grids from a smooth distribution via underdamped Langevin Markov chain Monte Carlo, and (ii) recover the "clean" molecule by denoising the noisy grid with a single step.Our method, VoxMol, generates molecules in a fundamentally different way than the current state of the art (ie, diffusion models applied to atom point clouds). It differs in terms of the data representation, the noise model, the network architecture and the generative modeling algorithm.Our experiments show that VoxMol captures the distribution of drug-like molecules better than state of the art, while being faster to generate samples.

----

## [3023] Accessing Higher Dimensions for Unsupervised Word Translation

**Authors**: *Sida Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/da31f4275972a58406b95c277ce7bc8d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/da31f4275972a58406b95c277ce7bc8d-Abstract-Conference.html)

**Abstract**:

The striking ability of unsupervised word translation has been demonstrated recently with the help of low-dimensional word vectors / pretraining, which is used by all successful methods and assumed to be necessary. We test and challenge this assumption by developing a method that can also make use of high dimensional signal. Freed from the limits of low dimensions, we show that relying on low-dimensional vectors and their incidental properties miss out on better denoising methods and signals in high dimensions, thus stunting the potential of the data. Our results show that unsupervised translation can be achieved more easily and robustly than previously thought -- less than 80MB and minutes of CPU time is required to achieve over 50\% accuracy for English to Finnish, Hungarian, and Chinese translations when trained in the same domain; even under domain mismatch, the method still works fully unsupervised on English NewsCrawl to Chinese Wikipedia and English Europarl to Spanish Wikipedia, among others. These results challenge prevailing assumptions on the necessity and superiority of low-dimensional vectors and show that the higher dimension signal can be used rather than thrown away.

----

## [3024] Inverse Reinforcement Learning with the Average Reward Criterion

**Authors**: *Feiyang Wu, Jingyang Ke, Anqi Wu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/da409884a933ecbc4af03338111bf6aa-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/da409884a933ecbc4af03338111bf6aa-Abstract-Conference.html)

**Abstract**:

We study the problem of Inverse Reinforcement Learning (IRL) with an average-reward criterion. The goal is to recover an unknown policy and a reward function when the agent only has samples of states and actions from an experienced agent. Previous IRL methods assume that the expert is trained in a discounted environment, and the discount factor is known. This work alleviates this assumption by proposing an average-reward framework with efficient learning algorithms. We develop novel stochastic first-order methods to solve the IRL problem under the average-reward setting, which requires solving an Average-reward Markov Decision Process (AMDP) as a subproblem. To solve the subproblem, we develop a Stochastic Policy Mirror Descent (SPMD) method under general state and action spaces that needs $\mathcal{O}(1/\varepsilon)$ steps of gradient computation. Equipped with SPMD, we propose the Inverse Policy Mirror Descent (IPMD) method for solving the IRL problem with a $\mathcal{O}(1/\varepsilon^2)$ complexity. To the best of our knowledge, the aforementioned complexity results are new in IRL with the average reward criterion. Finally, we corroborate our analysis with numerical experiments using the MuJoCo benchmark and additional control tasks.

----

## [3025] DisDiff: Unsupervised Disentanglement of Diffusion Probabilistic Models

**Authors**: *Tao Yang, Yuwang Wang, Yan Lu, Nanning Zheng*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/da47bfaf3f3a8d5bbab0d60c5195dc18-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/da47bfaf3f3a8d5bbab0d60c5195dc18-Abstract-Conference.html)

**Abstract**:

Targeting to understand the underlying explainable factors behind observations and modeling the conditional generation process on these factors, we connect disentangled representation learning to diffusion probabilistic models (DPMs) to take advantage of the remarkable modeling ability of DPMs. We propose a new task, disentanglement of (DPMs): given a pre-trained DPM, without any annotations of the factors, the task is to automatically discover the inherent factors behind the observations and disentangle the gradient fields of DPM into sub-gradient fields, each conditioned on the representation of each discovered factor. With disentangled DPMs, those inherent factors can be automatically discovered, explicitly represented and clearly injected into the diffusion process via the sub-gradient fields. To tackle this task, we devise an unsupervised approach, named DisDiff, and for the first time achieving disentangled representation learning in the framework of DPMs. Extensive experiments on synthetic and real-world datasets demonstrate the effectiveness of DisDiff.

----

## [3026] Information-guided Planning: An Online Approach for Partially Observable Problems

**Authors**: *Matheus Aparecido do Carmo Alves, Amokh Varma, Yehia Elkhatib, Leandro Soriano Marcolino*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/da5498f88193ff61f0daea1940b819da-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/da5498f88193ff61f0daea1940b819da-Abstract-Conference.html)

**Abstract**:

This paper presents IB-POMCP, a novel algorithm for online planning under partial observability. Our approach enhances the decision-making process by using estimations of the world belief's entropy to guide a tree search process and surpass the limitations of planning in scenarios with sparse reward configurations. By performing what we denominate as an information-guided planning process, the algorithm, which incorporates a novel I-UCB function, shows significant improvements in reward and reasoning time compared to state-of-the-art baselines in several benchmark scenarios, along with theoretical convergence guarantees.

----

## [3027] Bayesian Metric Learning for Uncertainty Quantification in Image Retrieval

**Authors**: *Frederik Warburg, Marco Miani, Silas Brack, Søren Hauberg*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/da7ce04b3683b173691ecbb801f2690f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/da7ce04b3683b173691ecbb801f2690f-Abstract-Conference.html)

**Abstract**:

We propose a Bayesian encoder for metric learning. Rather than relying on neural amortization as done in prior works, we learn a distribution over the network weights with the Laplace Approximation. We first prove that the contrastive loss is a negative log-likelihood on the spherical space. We propose three methods that ensure a positive definite covariance matrix. Lastly, we present a novel decomposition of the Generalized Gauss-Newton approximation. Empirically, we show that our Laplacian Metric Learner (LAM) yields well-calibrated uncertainties, reliably detects out-of-distribution examples, and has state-of-the-art predictive performance.

----

## [3028] Neural Modulation for Flash Memory: An Unsupervised Learning Framework for Improved Reliability

**Authors**: *Jonathan Zedaka, Elisha Halperin, Evgeny Blaichman, Amit Berman*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/da7e0d7210b99ebc91c4a5f911962d6c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/da7e0d7210b99ebc91c4a5f911962d6c-Abstract-Conference.html)

**Abstract**:

Recent years have witnessed a significant increase in the storage density of NAND flash memory, making it a critical component in modern electronic devices. However, with the rise in storage capacity comes an increased likelihood of errors in data storage and retrieval. The growing number of errors poses ongoing challenges for system designers and engineers, in terms of the characterization, modeling, and optimization of NAND-based systems. We present a novel approach for modeling and preventing errors by utilizing the capabilities of generative and unsupervised machine learning methods. As part of our research, we constructed and trained a neural modulator that translates information bits into programming operations on each memory cell in NAND devices. Our modulator, tailored explicitly for flash memory channels, provides a smart writing scheme that reduces programming errors as well as compensates for data degradation over time. Specifically, the modulator is based on an auto-encoder architecture with an additional channel model embedded between the encoder and the decoder. A conditional generative adversarial network (cGAN) was used to construct the channel model. Optimized for the end-of-life work-point, the learned memory system outperforms the prior art by up to 56\% in raw bit error rate (RBER) and extends the lifetime of the flash memory block by up to 25\%.

----

## [3029] Privacy Amplification via Compression: Achieving the Optimal Privacy-Accuracy-Communication Trade-off in Distributed Mean Estimation

**Authors**: *Wei-Ning Chen, Dan Song, Ayfer Özgür, Peter Kairouz*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/da8860a2fe8ddb7589136853bcc313fc-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/da8860a2fe8ddb7589136853bcc313fc-Abstract-Conference.html)

**Abstract**:

Privacy and communication constraints are two major bottlenecks in federated learning (FL) and analytics (FA). We study the optimal accuracy of mean and frequency estimation (canonical models for FL and FA respectively) under joint communication and $(\varepsilon, \delta)$-differential privacy (DP) constraints. We consider both the central and the multi-message shuffled DP models. We show that in order to achieve the optimal $\ell_2$ error under $(\varepsilon, \delta)$-DP, it is sufficient for each client to send $\Theta\left( n \min\left(\varepsilon, \varepsilon^2\right)\right)$ bits for FL %{\color{blue}(assuming the dimension $d \gg n \min\left(\varepsilon, \varepsilon^2\right)$)} and $\Theta\left(\log\left( n\min\left(\varepsilon, \varepsilon^2\right) \right)\right)$ bits for FA to the server, where $n$ is the number of participating clients.  Without compression, each client needs $O(d)$ bits and $O\left(\log d\right)$ bits for the mean and frequency estimation problems respectively (where $d$ corresponds to the number of trainable parameters in FL or the domain size in FA), meaning that we can get significant savings in the regime $ n \min\left(\varepsilon, \varepsilon^2\right)  = o(d)$, which is often the relevant regime in practice. We propose two different ways to leverage compression for privacy amplification and achieve the optimal privacy-communication-accuracy trade-offs. In both cases, each client communicates only partial information about its sample and we show that privacy is amplified by randomly selecting the part contributed by each client. In the first method, the random selection is revealed to the server, which results in a central DP guarantee with optimal privacy-communication-accuracy trade-offs.  In the second method, the random data parts from the clients are  shuffled by a secure shuffler resulting in a multi-message shuffling scheme with the same optimal trade-offs. As a result, we establish the optimal three-way trade-offs between privacy, communication, and accuracy for both the central DP and multi-message shuffling frameworks.

----

## [3030] Normalization Layers Are All That Sharpness-Aware Minimization Needs

**Authors**: *Maximilian Müller, Tiffany Vlaar, David Rolnick, Matthias Hein*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/da909fc3893d272f26fd9db82e09d954-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/da909fc3893d272f26fd9db82e09d954-Abstract-Conference.html)

**Abstract**:

Sharpness-aware minimization (SAM) was proposed to reduce sharpness of minima and has been shown to enhance generalization performance in various settings. In this work we show that perturbing only the affine normalization parameters (typically comprising 0.1% of the total parameters) in the adversarial step of SAM can outperform perturbing all of the parameters. This finding generalizesto different SAM variants and both ResNet (Batch Normalization) and Vision Transformer (Layer Normalization) architectures. We consider alternative sparse perturbation approaches and find that these do not achieve similar performance enhancement at such extreme sparsity levels, showing that this behaviour is unique to the normalization layers. Although our findings reaffirm the effectivenessof SAM in improving generalization performance, they cast doubt on whether this is solely caused by reduced sharpness.

----

## [3031] Robust Bayesian Satisficing

**Authors**: *Artun Saday, Yasar Cahit Yildirim, Cem Tekin*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/daa098aa8e1fc718943ff1ab7b5b30c9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/daa098aa8e1fc718943ff1ab7b5b30c9-Abstract-Conference.html)

**Abstract**:

Distributional shifts pose a significant challenge to achieving robustness in contemporary machine learning. To overcome this challenge, robust satisficing (RS) seeks a robust solution to an unspecified distributional shift while achieving a utility above a desired threshold. This paper focuses on the problem of RS in contextual Bayesian optimization when there is a discrepancy between the true and reference distributions of the context. We propose a novel robust Bayesian satisficing algorithm called RoBOS for noisy black-box optimization. Our algorithm guarantees sublinear lenient regret under certain assumptions on the amount of distribution shift. In addition, we define a weaker notion of regret called robust satisficing regret, in which our algorithm achieves a sublinear upper bound independent of the amount of distribution shift. To demonstrate the effectiveness of our method, we apply it to various learning problems and compare it to other approaches, such as distributionally robust optimization.

----

## [3032] Neural Ideal Large Eddy Simulation: Modeling Turbulence with Neural Stochastic Differential Equations

**Authors**: *Anudhyan Boral, Zhong Yi Wan, Leonardo Zepeda-Núñez, James Lottes, Qing Wang, Yi-Fan Chen, John Anderson, Fei Sha*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/dabaded617b3be96c3ed161498a7d71c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/dabaded617b3be96c3ed161498a7d71c-Abstract-Conference.html)

**Abstract**:

We introduce a data-driven learning framework that assimilates two powerful ideas: ideal large eddy simulation (LES) from turbulence closure modeling and neural stochastic differential equations (SDE) for stochastic modeling. The ideal LES models the LES flow by treating each full-order trajectory as a random realization of the underlying dynamics, as such, the effect of small-scales is marginalized to obtain the deterministic evolution of the LES state. However, ideal LES is analytically intractable. In our work, we use a latent neural SDE to model the evolution of the stochastic process and an encoder-decoder pair for transforming between the latent space and the desired ideal flow field. This stands in sharp contrast to other types of neural parameterization of closure models where each trajectory is treated as a deterministic realization of the dynamics. We show the effectiveness of our approach (niLES – neural ideal LES) on two challenging chaotic dynamical systems: Kolmogorov flow at a Reynolds number of 20,000 and flow past a cylinder at Reynolds number 500. Compared to competing methods, our method can handle non-uniform geometries using unstructured meshes seamlessly. In particular, niLES leads to trajectories with more accurate statistics and enhances stability, particularly for long-horizon rollouts. (Source codes and datasets will be made publicly available.)

----

## [3033] On the Generalization Error of Stochastic Mirror Descent for Quadratically-Bounded Losses: an Improved Analysis

**Authors**: *Ta Duy Nguyen, Alina Ene, Huy Nguyen*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/daca83eba0a30a5ff2a3b9c53ff5a976-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/daca83eba0a30a5ff2a3b9c53ff5a976-Abstract-Conference.html)

**Abstract**:

In this work, we revisit the generalization error of stochastic mirror descent for quadratically bounded losses studied in Telgarsky (2022). Quadratically bounded losses is a broad class of loss functions, capturing both Lipschitz and smooth functions, for both regression and classification problems. We study the high probability generalization for this class of losses on linear predictors in both realizable and non-realizable cases when the data are sampled IID or from a Markov chain. The prior work relies on an intricate coupling argument between the iterates of the original problem and those projected onto a bounded domain. This approach enables blackbox application of concentration inequalities, but also leads to suboptimal guarantees due in part to the use of a union bound across all iterations. In this work, we depart significantly from the prior work of Telgarsky (2022), and introduce a novel approach for establishing high probability generalization guarantees. In contrast to the prior work, our work directly analyzes the moment generating function of a novel supermartingale sequence and leverages the structure of stochastic mirror descent. As a result, we obtain improved bounds in all aforementioned settings. Specifically, in the realizable case and non-realizable case with light-tailed sub-Gaussian data, we improve the bounds by a $\log T$ factor, matching the correct rates of $1/T$ and $1/\sqrt{T}$, respectively. In the more challenging case of heavy-tailed polynomial data, we improve the existing bound by a $\mathrm{poly}\ T$ factor.

----

## [3034] StableFDG: Style and Attention Based Learning for Federated Domain Generalization

**Authors**: *Jungwuk Park, Dong-Jun Han, Jinho Kim, Shiqiang Wang, Christopher G. Brinton, Jaekyun Moon*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/dae8bdacd265399b193e6b43d44a80f0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/dae8bdacd265399b193e6b43d44a80f0-Abstract-Conference.html)

**Abstract**:

Traditional federated learning (FL) algorithms operate under the assumption that the data distributions at training (source domains) and testing (target domain) are the same. The fact that domain shifts often occur in practice necessitates equipping FL methods with a domain generalization (DG) capability. However, existing DG algorithms face fundamental challenges in FL setups due to the lack of samples/domains in each clientâ€™s local dataset. In this paper, we propose StableFDG, a style and attention based learning strategy for accomplishing federated domain generalization, introducing two key contributions. The first is style-based learning, which enables each client to explore novel styles beyond the original source domains in its local dataset, improving domain diversity based on the proposed style sharing, shifting, and exploration strategies. Our second contribution is an attention-based feature highlighter, which captures the similarities between the features of data samples in the same class, and emphasizes the important/common characteristics to better learn the domain-invariant characteristics of each class in data-poor FL scenarios. Experimental results show that StableFDG outperforms existing baselines on various DG benchmark datasets, demonstrating its efficacy.

----

## [3035] MG-ViT: A Multi-Granularity Method for Compact and Efficient Vision Transformers

**Authors**: *Yu Zhang, Yepeng Liu, Duoqian Miao, Qi Zhang, Yiwei Shi, Liang Hu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/daeef96627a461ec43b7567b2930cfde-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/daeef96627a461ec43b7567b2930cfde-Abstract-Conference.html)

**Abstract**:

Vision Transformer (ViT) faces obstacles in wide application due to its huge computational cost. Almost all existing studies on compressing ViT adopt the manner of splitting an image with a single granularity, with very few exploration of splitting an image with multi-granularity. As we know, important information often randomly concentrate in few regions of an image, necessitating multi-granularity attention allocation to an image. Enlightened by this, we introduce the multi-granularity strategy to compress ViT, which is simple but effective. We propose a two-stage multi-granularity framework, MG-ViT, to balance ViTâ€™s performance and computational cost. In single-granularity inference stage, an input image is split into a small number of patches for simple inference. If necessary, multi-granularity inference stage will be instigated, where the important patches are further subsplit into multi-finer-grained patches for subsequent inference. Moreover, prior studies on compression only for classification, while we extend the multi-granularity strategy to hierarchical ViT for downstream tasks such as detection and segmentation. Extensive experiments Prove the effectiveness of the multi-granularity strategy. For instance, on ImageNet, without any loss of performance, MG-ViT reduces 47\% FLOPs of LV-ViT-S and 56\% FLOPs of DeiT-S.

----

## [3036] Recurrent Temporal Revision Graph Networks

**Authors**: *Yizhou Chen, Anxiang Zeng, Qingtao Yu, Kerui Zhang, Yuanpeng Cao, Kangle Wu, Guangda Huzhang, Han Yu, Zhiming Zhou*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/dafd116ac8c735f149558b79fd48e090-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/dafd116ac8c735f149558b79fd48e090-Abstract-Conference.html)

**Abstract**:

Temporal graphs offer more accurate modeling of many real-world scenarios than static graphs. However, neighbor aggregation, a critical building block of graph networks, for temporal graphs, is currently straightforwardly extended from that of static graphs. It can be computationally expensive when involving all historical neighbors during such aggregation. In practice, typically only a subset of the most recent neighbors are involved. However, such subsampling leads to incomplete and biased neighbor information. To address this limitation, we propose a novel framework for temporal neighbor aggregation that uses the recurrent neural network with node-wise hidden states to integrate information from all historical neighbors for each node to acquire the complete neighbor information. We demonstrate the superior theoretical expressiveness of the proposed framework as well as its state-of-the-art performance in real-world applications. Notably, it achieves a significant +9.4% improvement on averaged precision in a real-world Ecommerce dataset over existing methods on 2-layer models.

----

## [3037] Recursion in Recursion: Two-Level Nested Recursion for Length Generalization with Scalability

**Authors**: *Jishnu Ray Chowdhury, Cornelia Caragea*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/db178cd03313e23cffb8937e93f0d464-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/db178cd03313e23cffb8937e93f0d464-Abstract-Conference.html)

**Abstract**:

Binary Balanced Tree Recursive Neural Networks (BBT-RvNNs) enforce sequence composition according to a preset balanced binary tree structure. Thus, their non-linear recursion depth (which is the tree depth) is just $\log_2 n$ ($n$ being the sequence length). Such logarithmic scaling makes BBT-RvNNs efficient and scalable on long sequence tasks such as Long Range Arena (LRA). However, such computational efficiency comes at a cost because BBT-RvNNs cannot solve simple arithmetic tasks like ListOps. On the flip side, RvNN models (e.g., Beam Tree RvNN) that do succeed on ListOps (and other structure-sensitive tasks like formal logical inference) are generally several times more expensive (in time and space) than even Recurrent Neural Networks. In this paper, we introduce a novel framework --- Recursion in Recursion (RIR) to strike a balance between the two sides - getting some of the benefits from both worlds. In RIR, we use a form of two-level nested recursion - where the outer recursion is a $k$-ary balanced tree model with another recursive model (inner recursion) implementing its cell function. For the inner recursion, we choose Beam Tree RvNNs. To adjust Beam Tree RvNNs within RIR we also propose a novel strategy of beam alignment. Overall, this entails that the total recursive depth in RIR is upper-bounded by $k \log_k n$. Our best RIR-based model is the first model that demonstrates high ($\geq 90\%$) length-generalization performance on ListOps while at the same time being scalable enough to be trainable on long sequence inputs from LRA (it can reduce the memory usage of the original Beam Tree RvNN by hundreds of times). Moreover, in terms of accuracy in the LRA language tasks, it performs competitively with Structured State Space Models (SSMs) without any special initialization - outperforming Transformers by a large margin. On the other hand, while SSMs can marginally outperform RIR on LRA, they (SSMs) fail to length-generalize on ListOps. Our code is available at: https://github.com/JRC1995/BeamRecursionFamily/

----

## [3038] xTrimoGene: An Efficient and Scalable Representation Learner for Single-Cell RNA-Seq Data

**Authors**: *Jing Gong, Minsheng Hao, Xingyi Cheng, Xin Zeng, Chiming Liu, Jianzhu Ma, Xuegong Zhang, Taifeng Wang, Le Song*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/db68f1c25678f72561ab7c97ce15d912-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/db68f1c25678f72561ab7c97ce15d912-Abstract-Conference.html)

**Abstract**:

Advances in high-throughput sequencing technology have led to significant progress in measuring gene expressions at the single-cell level. The amount of publicly available single-cell RNA-seq (scRNA-seq) data is already surpassing 50M records for humans with each record measuring 20,000 genes. This highlights the need for unsupervised representation learning to fully ingest these data, yet classical transformer architectures are prohibitive to train on such data in terms of both computation and memory. To address this challenge, we propose a novel asymmetric encoder-decoder transformer for scRNA-seq data, called xTrimoGene$^\alpha$ (or xTrimoGene for short), which leverages the sparse characteristic of the data to scale up the pre-training. This scalable design of xTrimoGene reduces FLOPs by one to two orders of magnitude compared to classical transformers while maintaining high accuracy, enabling us to train the largest transformer models over the largest scRNA-seq dataset today. Our experiments also show that the performance of xTrimoGene improves as we scale up the model sizes, and it also leads to SOTA performance over various downstream tasks, such as cell type annotation, perturb-seq effect prediction, and drug combination prediction. xTrimoGene model is now available for use as a service via the following link: https://api.biomap.com/xTrimoGene/apply.

----

## [3039] ANPL: Towards Natural Programming with Interactive Decomposition

**Authors**: *Di Huang, Ziyuan Nan, Xing Hu, Pengwei Jin, Shaohui Peng, Yuanbo Wen, Rui Zhang, Zidong Du, Qi Guo, Yewen Pu, Yunji Chen*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/dba8fa689ede9e56cbcd4f719def38fb-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/dba8fa689ede9e56cbcd4f719def38fb-Abstract-Conference.html)

**Abstract**:

Though LLMs are capable of generating plausible programs, it’s challenging to interact with the LLMs further to revise the program, especially if the user’s specific requirements are different from the initial proposal. In this paper, we introduce ANPL, an interactive programming system that ensures users can always refine the generated code towards their specific programmatic intents via structureddecompositions. Borrowing the paradigm of sketching from program synthesis, an ANPL program consists of a set of input-outputs that it must satisfy, a “sketch” — control/data flow expressed in precise code (e.g. Python), and “holes” — sub-modules to be implemented by the LLM specified with natural language. The user revises an ANPL program by either modifying the sketch, changing the language used to describe the holes, or providing additional input-outputs to a particular hole, turning it into a sub-ANPL program that can be solved recursively. This workflow allows the users to offload programming burdens to the LLM as much as possible while retaining the ability to pinpoint and resolve bugs locally, without exposing the rest of the program to the LLM. We deploy ANPL on the Abstraction and Reasoning Corpus (ARC), a set of unique tasks that are challenging for state-of-the-art AI systems, showing it outperforms baseline programming systems that (a) without the ability to decompose tasks interactively and (b) without the guarantee that the modules can be correctly composed together. Additional evaluations on APPS, HumanEval, and real-world programming tasks have validated that the ANPL framework is applicable to multiple programming domains. We release the ANPL solutions to the ARC tasks as a dataset, providing insights into how humans decompose novel tasks programmatically.

----

## [3040] Anonymous and Copy-Robust Delegations for Liquid Democracy

**Authors**: *Markus Utke, Ulrike Schmidt-Kraepelin*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/dbb5180957513805ebeea787b8c66ac9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/dbb5180957513805ebeea787b8c66ac9-Abstract-Conference.html)

**Abstract**:

Liquid democracy with ranked delegations is a novel voting scheme that unites the practicability of representative democracy with the idealistic appeal of direct democracy: Every voter decides between casting their vote on a question at hand or delegating their voting weight to some other, trusted agent. Delegations are transitive, and since voters may end up in a delegation cycle, they are encouraged to indicate not only a single delegate, but a set of potential delegates and a ranking among them. Based on the delegation preferences of all voters, a delegation rule selects one representative per voter. Previous work has revealed a trade-off between two properties of delegation rules called anonymity and copy-robustness. To overcome this issue we study two fractional delegation rules: Mixed Borda branching, which generalizes a rule satisfying copy-robustness, and the random walk rule, which satisfies anonymity. Using the Markov chain tree theorem, we show that the two rules are in fact equivalent, and simultaneously satisfy generalized versions of the two properties. Combining the same theorem with Fulkerson's algorithm, we develop  a polynomial-time algorithm for computing the outcome of the studied delegation rule. This algorithm is of independent interest, having applications in semi-supervised learning and graph theory.

----

## [3041] Framework and Benchmarks for Combinatorial and Mixed-variable Bayesian Optimization

**Authors**: *Kamil Dreczkowski, Antoine Grosnit, Haitham Bou-Ammar*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/dbc4b67c6430c22460623186c3d3fdc2-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/dbc4b67c6430c22460623186c3d3fdc2-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

This paper introduces a modular framework for Mixed-variable and Combinatorial Bayesian Optimization (MCBO) to address the lack of systematic benchmarking and standardized evaluation in the field. Current MCBO papers often introduce non-diverse or non-standard benchmarks to evaluate their methods, impeding the proper assessment of different MCBO primitives and their combinations. Additionally,  papers introducing a solution for a single MCBO primitive often omit benchmarking against baselines that utilize the same methods for the remaining primitives. This omission is primarily due to the significant implementation overhead involved, resulting in a lack of controlled assessments and an inability to showcase the merits of a contribution effectively.To overcome these challenges, our proposed framework enables an effortless combination of Bayesian Optimization components, and provides a diverse set of synthetic and real-world benchmarking tasks. Leveraging this flexibility, we implement 47 novel MCBO algorithms and benchmark them against seven existing MCBO solvers and five standard black-box optimization algorithms on ten tasks, conducting over 4000 experiments. Our findings reveal a superior combination of MCBO primitives outperforming existing approaches and illustrate the significance of model fit and the use of a trust region. We make our MCBO library available under the MIT license at \url{https://github.com/huawei-noah/HEBO/tree/master/MCBO}.

----

## [3042] KD-Zero: Evolving Knowledge Distiller for Any Teacher-Student Pairs

**Authors**: *Lujun Li, Peijie Dong, Anggeng Li, Zimian Wei, Ya Yang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/dbc8ce0fdfcd55172d73fb05dbae07fc-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/dbc8ce0fdfcd55172d73fb05dbae07fc-Abstract-Conference.html)

**Abstract**:

Knowledge distillation (KD) has emerged as an effective technique for compressing models that can enhance the lightweight model.  Conventional KD methods propose various designs to allow student model to imitate the teacher better.  However, these handcrafted KD designs heavily rely on expert knowledge and may be sub-optimal for various teacher-student pairs.  In this paper, we present a novel framework, KD-Zero, which utilizes evolutionary search to automatically discover promising distiller  from scratch for any teacher-student architectures.  Specifically, we first decompose the generalized distiller into knowledge transformations, distance functions, and loss weights.  Then,  we construct our distiller search space by selecting advanced operations for these three components.  With sharpness and represent gap as fitting objectives, we evolve candidate populations and generate better distillers by crossover and mutation.  To ensure efficient searching, we employ the loss-rejection protocol, search space shrinkage, and proxy settings during the search process.  In this manner, the discovered distiller can address the capacity gap and cross-architecture challenges for any teacher-student pairs in the final distillation stage.  Comprehensive experiments reveal that KD-Zero consistently outperforms other state-of-the-art methods across diverse architectures on classification, detection, and segmentation tasks.  Noticeably, we provide some practical insights in designing the distiller by analyzing the distiller discovered.  Codes are available in supplementary materials.

----

## [3043] Minimum-Risk Recalibration of Classifiers

**Authors**: *Zeyu Sun, Dogyoon Song, Alfred O. Hero III*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/dbd6b295535e44f2b8ec0c3f1da7c509-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/dbd6b295535e44f2b8ec0c3f1da7c509-Abstract-Conference.html)

**Abstract**:

Recalibrating probabilistic classifiers is vital for enhancing the reliability and accuracy of predictive models. Despite the development of numerous recalibration algorithms, there is still a lack of a comprehensive theory that integrates calibration and sharpness (which is essential for maintaining predictive power). In this paper, we introduce the concept of minimum-risk recalibration within the framework of mean-squared-error (MSE) decomposition, offering a principled approach for evaluating and recalibrating probabilistic classifiers. Using this framework, we analyze the uniform-mass binning (UMB) recalibration method and establish a finite-sample risk upper bound of order $\tilde{O}(B/n + 1/B^2)$ where $B$ is the number of bins and $n$ is the sample size. By balancing calibration and sharpness, we further determine that the optimal number of bins for UMB scales with $n^{1/3}$, resulting in a risk bound of approximately $O(n^{-2/3})$. Additionally, we tackle the challenge of label shift by proposing a two-stage approach that adjusts the recalibration function using limited labeled data from the target domain. Our results show that transferring a calibrated classifier requires significantly fewer target samples compared to recalibrating from scratch. We validate our theoretical findings through numerical simulations, which confirm the tightness of the proposed bounds, the optimal number of bins, and the effectiveness of label shift adaptation.

----

## [3044] DynPoint: Dynamic Neural Point For View Synthesis

**Authors**: *Kaichen Zhou, Jia-Xing Zhong, Sangyun Shin, Kai Lu, Yiyuan Yang, Andrew Markham, Niki Trigoni*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/dbdc7a9779ce0278c6e43b62c7e97759-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/dbdc7a9779ce0278c6e43b62c7e97759-Abstract-Conference.html)

**Abstract**:

The introduction of neural radiance fields has greatly improved the effectiveness of view synthesis for monocular videos. However, existing algorithms face difficulties when dealing with uncontrolled or lengthy scenarios, and require extensive training time specific to each new scenario.To tackle these limitations, we propose DynPoint, an algorithm designed to facilitate the rapid synthesis of novel views for unconstrained monocular videos. Rather than encoding the entirety of the scenario information into a latent representation, DynPoint concentrates on predicting the explicit 3D correspondence between neighboring frames to realize information aggregation.Specifically, this correspondence prediction is achieved through the estimation of consistent depth and scene flow information across frames.Subsequently, the acquired correspondence is utilized to aggregate information from multiple reference frames to a target frame, by constructing hierarchical neural point clouds. The resulting framework enables swift and accurate view synthesis for desired views of target frames. The experimental results obtained demonstrate the considerable acceleration of training time achieved - typically an order of magnitude - by our proposed method while yielding comparable outcomes compared to prior approaches. Furthermore, our method exhibits strong robustness in handling long-duration videos without learning a canonical representation of video content.

----

## [3045] Data-driven Optimal Filtering for Linear Systems with Unknown Noise Covariances

**Authors**: *Shahriar Talebi, Amirhossein Taghvaei, Mehran Mesbahi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/dbe8185809cb7032ec7ec6e365e3ed3b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/dbe8185809cb7032ec7ec6e365e3ed3b-Abstract-Conference.html)

**Abstract**:

This paper examines learning the optimal filtering policy, known as the Kalman gain, for a linear system with unknown noise covariance matrices using noisy output data. The learning problem is formulated as a stochastic policy optimiza- tion problem, aiming to minimize the output prediction error. This formulation provides a direct bridge between data-driven optimal control and, its dual, op- timal filtering. Our contributions are twofold. Firstly, we conduct a thorough convergence analysis of the stochastic gradient descent algorithm, adopted for the filtering problem, accounting for biased gradients and stability constraints. Secondly, we carefully leverage a combination of tools from linear system theory and high-dimensional statistics to derive bias-variance error bounds that scale logarithmically with problem dimension, and, in contrast to subspace methods, the length of output trajectories only affects the bias term.

----

## [3046] PPi: Pretraining Brain Signal Model for Patient-independent Seizure Detection

**Authors**: *Zhizhang Yuan, Daoze Zhang, Yang Yang, Junru Chen, Yafeng Li*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/dbeb7e621d4a554069a6a775da0f7273-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/dbeb7e621d4a554069a6a775da0f7273-Abstract-Conference.html)

**Abstract**:

Automated seizure detection is of great importance to epilepsy diagnosis and treatment. An emerging method used in seizure detection, stereoelectroencephalography (SEEG), can provide detailed and stereoscopic brainwave information. However, modeling SEEG in clinical scenarios will face challenges like huge domain shift between different patients and dramatic pattern evolution among different brain areas. In this study, we propose a Pretraining-based model for Patient-independent seizure detection (PPi) to address these challenges. Firstly, we design two novel self-supervised tasks which can extract rich information from abundant SEEG data while preserving the unique characteristics between brain signals recorded from different brain areas. Then two techniques channel background subtraction and brain region enhancement are proposed to effectively tackle the domain shift problem. Extensive experiments show that PPi outperforms the SOTA baselines on two public datasets and a real-world clinical dataset collected by ourselves, which demonstrates the effectiveness and practicability of PPi. Finally, visualization analysis illustrates the rationality of the two domain generalization techniques.

----

## [3047] Unsupervised Polychromatic Neural Representation for CT Metal Artifact Reduction

**Authors**: *Qing Wu, Lixuan Chen, Ce Wang, Hongjiang Wei, S. Kevin Zhou, Jingyi Yu, Yuyao Zhang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/dbf02b21d77409a2db30e56866a8ab3a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/dbf02b21d77409a2db30e56866a8ab3a-Abstract-Conference.html)

**Abstract**:

Emerging neural reconstruction techniques based on tomography (e.g., NeRF, NeAT, and NeRP) have started showing unique capabilities in medical imaging. In this work, we present a novel Polychromatic neural representation (Polyner) to tackle the challenging problem of CT imaging when metallic implants exist within the human body. CT metal artifacts arise from the drastic variation of metal's attenuation coefficients at various energy levels of the X-ray spectrum, leading to a nonlinear metal effect in CT measurements. Recovering CT images from metal-affected measurements hence poses a complicated nonlinear inverse problem where empirical models adopted in previous metal artifact reduction (MAR) approaches lead to signal loss and strongly aliased reconstructions. Polyner instead models the MAR problem from a nonlinear inverse problem perspective. Specifically, we first derive a polychromatic forward model to accurately simulate the nonlinear CT acquisition process. Then, we incorporate our forward model into the implicit neural representation to accomplish reconstruction. Lastly, we adopt a regularizer to preserve the physical properties of the CT images across different energy levels while effectively constraining the solution space. Our Polyner is an unsupervised method and does not require any external training data. Experimenting with multiple datasets shows that our Polyner achieves comparable or better performance than supervised methods on in-domain datasets while demonstrating significant performance improvements on out-of-domain datasets. To the best of our knowledge, our Polyner is the first unsupervised MAR method that outperforms its supervised counterparts. The code for this work is available at: https://github.com/iwuqing/Polyner.

----

## [3048] DAMEX: Dataset-aware Mixture-of-Experts for visual understanding of mixture-of-datasets

**Authors**: *Yash Jain, Harkirat S. Behl, Zsolt Kira, Vibhav Vineet*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/dc192b3eeffebba21bd1d82f6752b84b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/dc192b3eeffebba21bd1d82f6752b84b-Abstract-Conference.html)

**Abstract**:

Construction of a universal detector poses a crucial question: How can we most effectively train a model on a large mixture of datasets?     The answer lies in learning dataset-specific features and ensembling their knowledge but do all this in a single model.    Previous methods achieve this by having separate detection heads on a common backbone but that results in a significant increase in parameters.    In this work, we present Mixture-of-Experts as a solution, highlighting that MoE are much more than a scalability tool.     We propose Dataset-Aware Mixture-of-Experts, DAMEX where we train the experts to become an `expert' of a dataset by learning to route each dataset tokens to its mapped expert.    Experiments on Universal Object-Detection Benchmark show that we outperform the existing state-of-the-art by average +10.2 AP score and improve over our non-MoE baseline by average +2.0 AP score. We also observe consistent gains while mixing datasets with (1) limited availability, (2) disparate domains and (3) divergent label sets.    Further, we qualitatively show that DAMEX is robust against expert representation collapse. Code is available at https://github.com/jinga-lala/DAMEX

----

## [3049] FourierGNN: Rethinking Multivariate Time Series Forecasting from a Pure Graph Perspective

**Authors**: *Kun Yi, Qi Zhang, Wei Fan, Hui He, Liang Hu, Pengyang Wang, Ning An, Longbing Cao, Zhendong Niu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/dc1e32dd3eb381dbc71482f6a96cbf86-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/dc1e32dd3eb381dbc71482f6a96cbf86-Abstract-Conference.html)

**Abstract**:

Multivariate time series (MTS) forecasting has shown great importance in numerous industries. Current state-of-the-art graph neural network (GNN)-based forecasting methods usually require both graph networks (e.g., GCN) and temporal networks (e.g., LSTM) to capture inter-series (spatial) dynamics and intra-series (temporal) dependencies, respectively. However, the uncertain compatibility of the two networks puts an extra burden on handcrafted model designs. Moreover, the separate spatial and temporal modeling naturally violates the unified spatiotemporal inter-dependencies in real world, which largely hinders the forecasting performance. To overcome these problems, we explore an interesting direction of directly applying graph networks and rethink MTS forecasting from a pure graph perspective. We first define a novel data structure, hypervariate graph, which regards each series value (regardless of variates or timestamps) as a graph node, and represents sliding windows as space-time fully-connected graphs. This perspective considers spatiotemporal dynamics unitedly and reformulates classic MTS forecasting into the predictions on hypervariate graphs. Then, we propose a novel architecture Fourier Graph Neural Network (FourierGNN) by stacking our proposed Fourier Graph Operator (FGO) to perform matrix multiplications in Fourier space. FourierGNN accommodates adequate expressiveness and achieves much lower complexity, which can effectively and efficiently accomplish {the forecasting}. Besides, our theoretical analysis reveals FGO's equivalence to graph convolutions in the time domain, which further verifies the validity of FourierGNN. Extensive experiments on seven datasets have demonstrated our superior performance with higher efficiency and fewer parameters compared with state-of-the-art methods. Code is available at this repository: https://github.com/aikunyi/FourierGNN.

----

## [3050] Representation Equivalent Neural Operators: a Framework for Alias-free Operator Learning

**Authors**: *Francesca Bartolucci, Emmanuel de Bézenac, Bogdan Raonic, Roberto Molinaro, Siddhartha Mishra, Rima Alaifari*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/dc35c593e61f6df62db541b976d09dcf-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/dc35c593e61f6df62db541b976d09dcf-Abstract-Conference.html)

**Abstract**:

Recently, operator learning, or learning mappings between infinite-dimensional function spaces, has garnered significant attention, notably in relation to learning partial differential equations from data. Conceptually clear when outlined on paper, neural operators necessitate discretization in the transition to computer implementations. This step can compromise their integrity, often causing them to deviate from the underlying operators. This research offers a fresh take on neural operators with a framework Representation equivalent Neural Operators (ReNO) designed to address these issues. At its core is the concept of operator aliasing, which measures inconsistency between neural operators and their discrete representations. We explore this for widely-used operator learning techniques. Our findings detail how aliasing introduces errors when handling different discretizations and grids and loss of crucial continuous structures. More generally, this framework not only sheds light on existing challenges but, given its constructive and broad nature, also potentially offers tools for developing new neural operators.

----

## [3051] Unsupervised Anomaly Detection with Rejection

**Authors**: *Lorenzo Perini, Jesse Davis*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/dc48c738d3ef8c81b6e968453a84a819-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/dc48c738d3ef8c81b6e968453a84a819-Abstract-Conference.html)

**Abstract**:

Anomaly detection aims at detecting unexpected behaviours in the data. Because anomaly detection is usually an unsupervised task, traditional anomaly detectors learn a decision boundary by employing heuristics based on intuitions, which are hard to verify in practice. This introduces some uncertainty, especially close to the decision boundary, that may reduce the user trust in the detector's predictions. A way to combat this is by allowing the detector to reject predictions with high uncertainty (Learning to Reject). This requires employing a confidence metric that captures the distance to the decision boundary and setting a rejection threshold to reject low-confidence predictions. However, selecting a proper metric and setting the rejection threshold without labels are challenging tasks. In this paper, we solve these challenges by setting a constant rejection threshold on the stability metric computed by ExCeeD. Our insight relies on a theoretical analysis of such a metric. Moreover, setting a constant threshold results in strong guarantees: we estimate the test rejection rate, and derive a theoretical upper bound for both the rejection rate and the expected prediction cost. Experimentally, we show that our method outperforms some metric-based methods.

----

## [3052] 4D Panoptic Scene Graph Generation

**Authors**: *Jingkang Yang, Jun Cen, Wenxuan Peng, Shuai Liu, Fangzhou Hong, Xiangtai Li, Kaiyang Zhou, Qifeng Chen, Ziwei Liu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/dc6319dde4fb182b22fb902da9418566-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/dc6319dde4fb182b22fb902da9418566-Abstract-Conference.html)

**Abstract**:

We are living in a three-dimensional space while moving forward through a fourth dimension: time. To allow artificial intelligence to develop a comprehensive understanding of such a 4D environment, we introduce 4D Panoptic Scene Graph (PSG-4D), a new representation that bridges the raw visual data perceived in a dynamic 4D world and high-level visual understanding. Specifically, PSG-4D abstracts rich 4D sensory data into nodes, which represent entities with precise location and status information, and edges, which capture the temporal relations. To facilitate research in this new area, we build a richly annotated PSG-4D dataset consisting of 3K RGB-D videos with a total of 1M frames, each of which is labeled with 4D panoptic segmentation masks as well as fine-grained, dynamic scene graphs. To solve PSG-4D, we propose PSG4DFormer, a Transformer-based model that can predict panoptic segmentation masks, track masks along the time axis, and generate the corresponding scene graphs via a relation component. Extensive experiments on the new dataset show that our method can serve as a strong baseline for future research on PSG-4D. In the end, we provide a real-world application example to demonstrate how we can achieve dynamic scene understanding by integrating a large language model into our PSG-4D system.

----

## [3053] ChatGPT-Powered Hierarchical Comparisons for Image Classification

**Authors**: *Zhiyuan Ren, Yiyang Su, Xiaoming Liu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/dc81297c791bb989deade65c6bd8c1d8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/dc81297c791bb989deade65c6bd8c1d8-Abstract-Conference.html)

**Abstract**:

The zero-shot open-vocabulary setting poses challenges for image classification.Fortunately, utilizing a vision-language model like CLIP, pre-trained on image-textpairs, allows for classifying images by comparing embeddings. Leveraging largelanguage models (LLMs) such as ChatGPT can further enhance CLIPâ€™s accuracyby incorporating class-specific knowledge in descriptions. However, CLIP stillexhibits a bias towards certain classes and generates similar descriptions for similarclasses, disregarding their differences. To address this problem, we present anovel image classification framework via hierarchical comparisons. By recursivelycomparing and grouping classes with LLMs, we construct a class hierarchy. Withsuch a hierarchy, we can classify an image by descending from the top to the bottomof the hierarchy, comparing image and text embeddings at each level. Throughextensive experiments and analyses, we demonstrate that our proposed approach isintuitive, effective, and explainable. Code will be released upon publication.

----

## [3054] Module-wise Adaptive Distillation for Multimodality Foundation Models

**Authors**: *Chen Liang, Jiahui Yu, Ming-Hsuan Yang, Matthew Brown, Yin Cui, Tuo Zhao, Boqing Gong, Tianyi Zhou*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/dc9544b26ad3579477e567588db18cfc-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/dc9544b26ad3579477e567588db18cfc-Abstract-Conference.html)

**Abstract**:

Pre-trained multimodal foundation models have demonstrated remarkable generalizability but pose challenges for deployment due to their large sizes. One effective approach to reducing their sizes is layerwise distillation, wherein small student models are trained to match the hidden representations of large teacher models at each layer. Motivated by our observation that certain architecture components, referred to as modules, contribute more significantly to the student's performance than others, we propose to track the contributions of individual modules by recording the loss decrement after distillation each module and choose the module with a greater contribution to distill more frequently. Such an approach can be naturally formulated as a multi-armed bandit (MAB) problem, where modules and loss decrements are considered as arms and rewards, respectively. We then develop a modified-Thompson sampling algorithm named OPTIMA to address the nonstationarity of module contributions resulting from model updating. Specifically, we leverage the observed contributions in recent history to estimate the changing contribution of each module and select modules based on these estimations to maximize the cumulative contribution. We evaluate the effectiveness of OPTIMA through distillation experiments on various multimodal understanding and image captioning tasks, using the CoCa-Large model \citep{yu2022coca} as the teacher model.

----

## [3055] Evaluating Cognitive Maps and Planning in Large Language Models with CogEval

**Authors**: *Ida Momennejad, Hosein Hasanbeig, Felipe Vieira Frujeri, Hiteshi Sharma, Nebojsa Jojic, Hamid Palangi, Robert Osazuwa Ness, Jonathan Larson*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/dc9d5dcf3e86b83e137bad367227c8ca-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/dc9d5dcf3e86b83e137bad367227c8ca-Abstract-Conference.html)

**Abstract**:

Recently an influx of studies claims emergent cognitive abilities in large language models (LLMs). Yet, most rely on anecdotes, overlook contamination of training sets, or lack systematic Evaluation involving multiple tasks, control conditions, multiple iterations, and statistical robustness tests. Here we make two major contributions. First, we propose CogEval, a cognitive science-inspired protocol for the systematic evaluation of cognitive capacities in LLMs. The CogEval protocol can be followed for the evaluation of various abilities. Second, here we follow CogEval to systematically evaluate cognitive maps and planning ability across eight LLMs (OpenAI GPT-4, GPT-3.5-turbo-175B, davinci-003-175B, Google Bard, Cohere-xlarge-52.4B, Anthropic Claude-1-52B, LLaMA-13B, and Alpaca-7B). We base our task prompts on human experiments, which offer both established construct validity for evaluating planning, and are absent from LLM training sets. We find that, while LLMs show apparent competence in a few planning tasks with simpler structures, systematic evaluation reveals striking failure modes in planning tasks, including hallucinations of invalid trajectories and falling in loops. These findings do not support the idea of emergent out-of-the-box planning ability in LLMs. This could be because LLMs do not understand the latent relational structures underlying planning problems, known as cognitive maps, and fail at unrolling goal-directed trajectories based on the underlying structure. Implications for application and future directions are discussed.

----

## [3056] Unsupervised Image Denoising with Score Function

**Authors**: *Yutong Xie, Mingze Yuan, Bin Dong, Quanzheng Li*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/dc9e095f668044e7a0909a4ea3926beb-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/dc9e095f668044e7a0909a4ea3926beb-Abstract-Conference.html)

**Abstract**:

Though achieving excellent performance in some cases, current unsupervised learning methods for single image denoising usually have constraints in applications. In this paper, we propose a new approach which is more general and applicable to complicated noise models. Utilizing the property of score function, the gradient of logarithmic probability, we define a solving system for denoising. Once the score function of noisy images has been estimated, the denoised result can be obtained through the solving system. Our approach can be applied to multiple noise models, such as the mixture of multiplicative and additive noise combined with structured correlation. Experimental results show that our method is comparable when the noise model is simple, and has good performance in complicated cases where other methods are not applicable or perform poorly.

----

## [3057] Iterative Reachability Estimation for Safe Reinforcement Learning

**Authors**: *Milan Ganai, Zheng Gong, Chenning Yu, Sylvia L. Herbert, Sicun Gao*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/dca63f2650fe9e88956c1b68440b8ee9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/dca63f2650fe9e88956c1b68440b8ee9-Abstract-Conference.html)

**Abstract**:

Ensuring safety is important for the practical deployment of reinforcement learning (RL). Various challenges must be addressed, such as handling stochasticity in the environments, providing rigorous guarantees of persistent state-wise safety satisfaction, and avoiding overly conservative behaviors that sacrifice performance. We propose a new framework, Reachability Estimation for Safe Policy Optimization (RESPO), for safety-constrained RL in general stochastic settings. In the feasible set where there exist violation-free policies, we optimize for rewards while maintaining persistent safety. Outside this feasible set, our optimization produces the safest behavior by guaranteeing entrance into the feasible set whenever possible with the least cumulative discounted violations. We introduce a class of algorithms using our novel reachability estimation function to optimize in our proposed framework and in similar frameworks such as those concurrently handling multiple hard and soft constraints. We theoretically establish that our algorithms almost surely converge to locally optimal policies of our safe optimization framework. We evaluate the proposed methods on a diverse suite of safe RL environments from Safety Gym, PyBullet, and MuJoCo, and show the benefits in improving both reward performance and safety compared with state-of-the-art baselines.

----

## [3058] DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining

**Authors**: *Sang Michael Xie, Hieu Pham, Xuanyi Dong, Nan Du, Hanxiao Liu, Yifeng Lu, Percy Liang, Quoc V. Le, Tengyu Ma, Adams Wei Yu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/dcba6be91359358c2355cd920da3fcbd-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/dcba6be91359358c2355cd920da3fcbd-Abstract-Conference.html)

**Abstract**:

The mixture proportions of pretraining data domains (e.g., Wikipedia, books, web text) greatly affect language model (LM) performance. In this paper, we propose Domain Reweighting with Minimax Optimization (DoReMi), which first trains a small proxy model using group distributionally robust optimization (Group DRO) over domains to produce domain weights (mixture proportions) without knowledge of downstream tasks. We then resample a dataset with these domain weights and train a larger, full-sized model. In our experiments, we use DoReMi on a 280M-parameter proxy model to set the domain weights for training an 8B-parameter model (30x larger) more efficiently. On The Pile, DoReMi improves perplexity across all domains, even when it downweights a domain. DoReMi improves average few-shot downstream accuracy by 6.5% points over a baseline model trained using The Pile's default domain weights and reaches the baseline accuracy with 2.6x fewer training steps. On the GLaM dataset, DoReMi, which has no knowledge of downstream tasks, even matches the performance of using domain weights tuned on downstream tasks.

----

## [3059] OpenSTL: A Comprehensive Benchmark of Spatio-Temporal Predictive Learning

**Authors**: *Cheng Tan, Siyuan Li, Zhangyang Gao, Wenfei Guan, Zedong Wang, Zicheng Liu, Lirong Wu, Stan Z. Li*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/dcbff44d11130e75d09d3930411c23e1-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/dcbff44d11130e75d09d3930411c23e1-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Spatio-temporal predictive learning is a learning paradigm that enables models to learn spatial and temporal patterns by predicting future frames from given past frames in an unsupervised manner. Despite remarkable progress in recent years, a lack of systematic understanding persists due to the diverse settings, complex implementation, and difficult reproducibility. Without standardization, comparisons can be unfair and insights inconclusive. To address this dilemma, we propose OpenSTL, a comprehensive benchmark for spatio-temporal predictive learning that categorizes prevalent approaches into recurrent-based and recurrent-free models. OpenSTL provides a modular and extensible framework implementing various state-of-the-art methods. We conduct standard evaluations on datasets across various domains, including synthetic moving object trajectory, human motion, driving scenes, traffic flow, and weather forecasting. Based on our observations, we provide a detailed analysis of how model architecture and dataset properties affect spatio-temporal predictive learning performance. Surprisingly, we find that recurrent-free models achieve a good balance between efficiency and performance than recurrent models. Thus, we further extend the common MetaFormers to boost recurrent-free spatial-temporal predictive learning. We open-source the code and models at https://github.com/chengtan9907/OpenSTL.

----

## [3060] Guiding The Last Layer in Federated Learning with Pre-Trained Models

**Authors**: *Gwen Legate, Nicolas Bernier, Lucas Page-Caccia, Edouard Oyallon, Eugene Belilovsky*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/dcc0ac74ac8b95dc1939804acce0317d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/dcc0ac74ac8b95dc1939804acce0317d-Abstract-Conference.html)

**Abstract**:

Federated Learning (FL) is an emerging paradigm that allows a model to be trained across a number of participants without sharing data. Recent works have begun to consider the effects of using pre-trained models as an initialization point for existing FL algorithms; however, these approaches ignore the vast body of efficient transfer learning literature from the centralized learning setting. Here we revisit the problem of FL from a pre-trained model considered in prior work and expand it to a set of computer vision transfer learning problems. We first observe that simply fitting a linear classification head can be efficient in many cases. We then show that in the FL setting, fitting a classifier using the Nearest Class Means (NCM) can be done exactly and  orders of magnitude more efficiently than existing proposals, while obtaining strong performance. Finally, we demonstrate that using a two-stage approach of obtaining the classifier and then fine-tuning the model can yield rapid convergence and improved generalization in the federated setting. We demonstrate the potential our method has to reduce communication and compute costs while achieving better model performance.

----

## [3061] Unbiased learning of deep generative models with structured discrete representations

**Authors**: *Henry C. Bendekgey, Gabe Hope, Erik Sudderth*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/dcc337bb2a4d25afefd9ab800721debb-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/dcc337bb2a4d25afefd9ab800721debb-Abstract-Conference.html)

**Abstract**:

By composing graphical models with deep learning architectures, we learn generative models with the strengths of both frameworks. The structured variational autoencoder (SVAE) inherits structure and interpretability from graphical models, and flexible likelihoods for high-dimensional data from deep learning, but poses substantial optimization challenges.  We propose novel algorithms for learning SVAEs, and are the first to demonstrate the SVAE's ability to handle multimodal uncertainty when data is missing by incorporating discrete latent variables.  Our memory-efficient implicit differentiation scheme makes the SVAE tractable to learn via gradient descent, while demonstrating robustness to incomplete optimization. To more rapidly learn accurate graphical model parameters, we derive a method for computing natural gradients without manual derivations, which avoids biases found in prior work.  These optimization innovations enable the first comparisons of the SVAE to state-of-the-art time series models, where the SVAE performs competitively while learning interpretable and structured discrete data representations.

----

## [3062] GLEMOS: Benchmark for Instantaneous Graph Learning Model Selection

**Authors**: *Namyong Park, Ryan A. Rossi, Xing Wang, Antoine Simoulin, Nesreen K. Ahmed, Christos Faloutsos*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/dcd18e50ebca0af89187c6e35dabb584-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/dcd18e50ebca0af89187c6e35dabb584-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

The choice of a graph learning (GL) model (i.e., a GL algorithm and its hyperparameter settings) has a significant impact on the performance of downstream tasks. However, selecting the right GL model becomes increasingly difficult and time consuming as more and more GL models are developed. Accordingly, it is of great significance and practical value to equip users of GL with the ability to perform a near-instantaneous selection of an effective GL model without manual intervention. Despite the recent attempts to tackle this important problem, there has been no comprehensive benchmark environment to evaluate the performance of GL model selection methods. To bridge this gap, we present GLEMOS in this work, a comprehensive benchmark for instantaneous GL model selection that makes the following contributions. (i) GLEMOS provides extensive benchmark data for fundamental GL tasks, i.e., link prediction and node classification, including the performances of 366 models on 457 graphs on these tasks. (ii) GLEMOS designs multiple evaluation settings, and assesses how effectively representative model selection techniques perform in these different settings. (iii) GLEMOS is designed to be easily extended with new models, new graphs, and new performance records. (iv) Based on the experimental results, we discuss the limitations of existing approaches and highlight future research directions. To promote research on this significant problem, we make the benchmark data and code publicly available at https://namyongpark.github.io/glemos.

----

## [3063] STEVE-1: A Generative Model for Text-to-Behavior in Minecraft

**Authors**: *Shalev Lifshitz, Keiran Paster, Harris Chan, Jimmy Ba, Sheila A. McIlraith*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/dd03f856fc7f2efeec8b1c796284561d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/dd03f856fc7f2efeec8b1c796284561d-Abstract-Conference.html)

**Abstract**:

Constructing AI models that respond to text instructions is challenging, especially for sequential decision-making tasks. This work introduces a methodology, inspired by unCLIP, for instruction-tuning generative models of behavior without relying on a large dataset of instruction-labeled trajectories. Using this methodology, we create an instruction-tuned Video Pretraining (VPT) model called STEVE-1, which can follow short-horizon open-ended text and visual instructions in Minecraft. STEVE-1 is trained in two steps: adapting the pretrained VPT model to follow commands in MineCLIP's latent space, then training a prior to predict latent codes from text. This allows us to finetune VPT through self-supervised behavioral cloning and hindsight relabeling, reducing the need for costly human text annotations, and all for only $60 of compute. By leveraging pretrained models like VPT and MineCLIP and employing best practices from text-conditioned image generation, STEVE-1 sets a new bar for open-ended instruction following in Minecraft with low-level controls (mouse and keyboard) and raw pixel inputs, far outperforming previous baselines and robustly completing 12 of 13 tasks in our early-game evaluation suite. We provide experimental evidence highlighting key factors for downstream performance, including pretraining, classifier-free guidance, and data scaling. All resources, including our model weights, training scripts, and evaluation tools are made available for further research.

----

## [3064] Thrust: Adaptively Propels Large Language Models with External Knowledge

**Authors**: *Xinran Zhao, Hongming Zhang, Xiaoman Pan, Wenlin Yao, Dong Yu, Jianshu Chen*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/dd058e9ec9dc012a273594d717c46ef3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/dd058e9ec9dc012a273594d717c46ef3-Abstract-Conference.html)

**Abstract**:

Although large-scale pre-trained language models (PTLMs) are shown to encode rich knowledge in their model parameters, the inherent knowledge in PTLMs can be opaque or static, making external knowledge necessary. However, the existing information retrieval techniques could be costly and may even introduce noisy and sometimes misleading knowledge. To address these challenges, we propose the instance-level adaptive propulsion of external knowledge (IAPEK), where we only conduct the retrieval when necessary. To achieve this goal, we propose to model whether a PTLM contains enough knowledge to solve an instance with a novel metric, Thrust, which leverages the representation distribution of a small amount of seen instances. Extensive experiments demonstrate that Thrust is a good measurement of models' instance-level knowledgeability. Moreover, we can achieve higher cost-efficiency with the Thrust score as the retrieval indicator than the naive usage of external knowledge on 88% of the evaluated tasks with 26% average performance improvement. Such findings shed light on the real-world practice of knowledge-enhanced LMs with a limited budget for knowledge seeking due to computation latency or costs.

----

## [3065] OneNet: Enhancing Time Series Forecasting Models under Concept Drift by Online Ensembling

**Authors**: *Yifan Zhang, Qingsong Wen, Xue Wang, Weiqi Chen, Liang Sun, Zhang Zhang, Liang Wang, Rong Jin, Tieniu Tan*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/dd6a47bc0aad6f34aa5e77706d90cdc4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/dd6a47bc0aad6f34aa5e77706d90cdc4-Abstract-Conference.html)

**Abstract**:

Online updating of time series forecasting models aims to address the concept drifting problem by efficiently updating forecasting models based on streaming data. Many algorithms are designed for online time series forecasting, with some exploiting cross-variable dependency while others assume independence among variables. Given every data assumption has its own pros and cons in online time series modeling, we propose **On**line **e**nsembling **Net**work (**OneNet**). It dynamically updates and combines two models, with one focusing on modeling the dependency across the time dimension and the other on cross-variate dependency. Our method incorporates a reinforcement learning-based approach into the traditional online convex programming framework, allowing for the linear combination of the two models with dynamically adjusted weights. OneNet addresses the main shortcoming of classical online learning methods that tend to be slow in adapting to the concept drift. Empirical results show that OneNet reduces online forecasting error by more than $\mathbf{50}\\%$ compared to the State-Of-The-Art (SOTA) method.

----

## [3066] Holistic Evaluation of Text-to-Image Models

**Authors**: *Tony Lee, Michihiro Yasunaga, Chenlin Meng, Yifan Mai, Joon Sung Park, Agrim Gupta, Yunzhi Zhang, Deepak Narayanan, Hannah Teufel, Marco Bellagente, Minguk Kang, Taesung Park, Jure Leskovec, Jun-Yan Zhu, Fei-Fei Li, Jiajun Wu, Stefano Ermon, Percy Liang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/dd83eada2c3c74db3c7fe1c087513756-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/dd83eada2c3c74db3c7fe1c087513756-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

The stunning qualitative improvement of text-to-image models has led to their widespread attention and adoption. However, we lack a comprehensive quantitative understanding of their capabilities and risks. To fill this gap, we introduce a new benchmark, Holistic Evaluation of Text-to-Image Models (HEIM). Whereas previous evaluations focus mostly on image-text alignment and image quality, we identify 12 aspects, including text-image alignment, image quality, aesthetics, originality, reasoning, knowledge, bias, toxicity, fairness, robustness, multilinguality, and efficiency. We curate 62 scenarios encompassing these aspects and evaluate 26 state-of-the-art text-to-image models on this benchmark. Our results reveal that no single model excels in all aspects, with different models demonstrating different strengths. We release the generated images and human evaluation results for full transparency at https://crfm.stanford.edu/heim/latest and the code at https://github.com/stanford-crfm/helm, which is integrated with the HELM codebase

----

## [3067] Shape Non-rigid Kinematics (SNK): A Zero-Shot Method for Non-Rigid Shape Matching via Unsupervised Functional Map Regularized Reconstruction

**Authors**: *Souhaib Attaiki, Maks Ovsjanikov*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/dd9b76f050a86a3ded6135ad3556e786-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/dd9b76f050a86a3ded6135ad3556e786-Abstract-Conference.html)

**Abstract**:

We present Shape Non-rigid Kinematics (SNK), a novel zero-shot method for non-rigid shape matching that eliminates the need for extensive training or ground truth data.SNK operates on a single pair of shapes, and employs a reconstruction-based strategy using an encoder-decoder architecture, which deforms the source shape to closely match the target shape. During the process, an unsupervised functional map is predicted and converted into a point-to-point map, serving as a supervisory mechanism for the reconstruction. To aid in training, we have designed a new decoder architecture that generates smooth, realistic deformations. SNK demonstrates competitive results on traditional benchmarks, simplifying the shape-matching process without compromising accuracy. Our code can be found online: https://github.com/pvnieo/SNK

----

## [3068] Frequency Domain-Based Dataset Distillation

**Authors**: *DongHyeok Shin, Seungjae Shin, Il-Chul Moon*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ddbbcd937d63d5c6b935c07b1a8222ec-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ddbbcd937d63d5c6b935c07b1a8222ec-Abstract-Conference.html)

**Abstract**:

This paper presents FreD, a novel parameterization method for dataset distillation, which utilizes the frequency domain to distill a small-sized synthetic dataset from a large-sized original dataset. Unlike conventional approaches that focus on the spatial domain, FreD employs frequency-based transforms to optimize the frequency representations of each data instance. By leveraging the concentration of spatial domain information on specific frequency components, FreD intelligently selects a subset of frequency dimensions for optimization, leading to a significant reduction in the required budget for synthesizing an instance. Through the selection of frequency dimensions based on the explained variance, FreD demonstrates both theoretical and empirical evidence of its ability to operate efficiently within a limited budget, while better preserving the information of the original dataset compared to conventional parameterization methods. Furthermore, Based on the orthogonal compatibility of FreD with existing methods, we confirm that FreD consistently improves the performances of existing distillation methods over the evaluation scenarios with different benchmark datasets. We release the code at https://github.com/sdh0818/FreD.

----

## [3069] Three-Way Trade-Off in Multi-Objective Learning: Optimization, Generalization and Conflict-Avoidance

**Authors**: *Lisha Chen, Heshan Devaka Fernando, Yiming Ying, Tianyi Chen*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ddcf34623ca2d63823b6d40e4d980580-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ddcf34623ca2d63823b6d40e4d980580-Abstract-Conference.html)

**Abstract**:

Multi-objective learning (MOL) often arises in emerging machine learning problems when multiple learning criteria or tasks need to be addressed.  Recent works have developed various _dynamic weighting_ algorithms for MOL, including MGDA and its variants, whose central idea is to find an update direction that _avoids conflicts_ among objectives. Albeit its appealing intuition, empirical studies show that dynamic weighting methods may not always outperform static alternatives. To bridge this gap between theory and practice, we focus on a new variant of stochastic MGDA - the Multi-objective gradient with Double sampling (MoDo) algorithm and study its generalization performance and the interplay with optimization through the lens of algorithm stability. We find that the rationale behind MGDA -- updating along conflict-avoidant direction - may \emph{impede} dynamic weighting algorithms from achieving the optimal ${\cal O}(1/\sqrt{n})$ population risk, where $n$ is the number of training samples. We further highlight the variability of dynamic weights and their impact on the three-way trade-off among optimization, generalization, and conflict avoidance that is unique in MOL. Code is available at https://github.com/heshandevaka/Trade-Off-MOL.

----

## [3070] OV-PARTS: Towards Open-Vocabulary Part Segmentation

**Authors**: *Meng Wei, Xiaoyu Yue, Wenwei Zhang, Shu Kong, Xihui Liu, Jiangmiao Pang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/dde53059fdb0f45e1e9ad9c66997d662-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/dde53059fdb0f45e1e9ad9c66997d662-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Segmenting and recognizing diverse object parts is a crucial ability in applications spanning various computer vision and robotic tasks. While significant progress has been made in object-level Open-Vocabulary Semantic Segmentation (OVSS), i.e., segmenting objects with arbitrary text, the corresponding part-level research poses additional challenges. Firstly, part segmentation inherently involves intricate boundaries, while limited annotated data compounds the challenge. Secondly, part segmentation introduces an open granularity challenge due to the diverse and often ambiguous definitions of parts in the open world. Furthermore, the large-scale vision and language models, which play a key role in the open vocabulary setting, struggle to recognize parts as effectively as objects. To comprehensively investigate and tackle these challenges, we propose an Open-Vocabulary Part Segmentation (OV-PARTS) benchmark. OV-PARTS includes refined versions of two publicly available datasets: Pascal-Part-116 and ADE20K-Part-234. And it covers three specific tasks: Generalized Zero-Shot Part Segmentation, Cross-Dataset Part Segmentation, and Few-Shot Part Segmentation, providing insights into analogical reasoning, open granularity and few-shot adapting abilities of models. Moreover, we analyze and adapt two prevailing paradigms of existing object-level OVSS methods for OV-PARTS. Extensive experimental analysis is conducted to inspire future research in leveraging foundational models for OV-PARTS. The code and dataset are available at https://github.com/kellyiss/OV_PARTS.

----

## [3071] Large Language Models are Visual Reasoning Coordinators

**Authors**: *Liangyu Chen, Bo Li, Sheng Shen, Jingkang Yang, Chunyuan Li, Kurt Keutzer, Trevor Darrell, Ziwei Liu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ddfe6bae7b869e819f842753009b94ad-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ddfe6bae7b869e819f842753009b94ad-Abstract-Conference.html)

**Abstract**:

Visual reasoning requires multimodal perception and commonsense cognition of the world. Recently, multiple vision-language models (VLMs) have been proposed with excellent commonsense reasoning ability in various domains. However, how to harness the collective power of these complementary VLMs is rarely explored. Existing methods like ensemble still struggle to aggregate these models with the desired higher-order communications. In this work, we propose Cola, a novel paradigm that coordinates multiple VLMs for visual reasoning. Our key insight is that a large language model (LLM) can efficiently coordinate multiple VLMs by facilitating natural language communication that leverages their distinct and complementary capabilities. Extensive experiments demonstrate that our instruction tuning variant, Cola-FT, achieves state-of-the-art performance on visual question answering (VQA), outside knowledge VQA, visual entailment, and visual spatial reasoning tasks. Moreover, we show that our in-context learning variant, Cola-Zero, exhibits competitive performance in zero and few-shot settings, without finetuning. Through systematic ablation studies and visualizations, we validate that a coordinator LLM indeed comprehends the instruction prompts as well as the separate functionalities of VLMs; it then coordinates them to enable impressive visual reasoning capabilities.

----

## [3072] Boosting Adversarial Transferability by Achieving Flat Local Maxima

**Authors**: *Zhijin Ge, Xiaosen Wang, Hongying Liu, Fanhua Shang, Yuanyuan Liu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/de1739eba209c682a90ec3669229ab2d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/de1739eba209c682a90ec3669229ab2d-Abstract-Conference.html)

**Abstract**:

Transfer-based attack adopts the adversarial examples generated on the surrogate model to attack various models, making it applicable in the physical world and attracting increasing interest. Recently, various adversarial attacks have emerged to boost adversarial transferability from different perspectives. In this work, inspired by the observation that flat local minima are correlated with good generalization, we assume and empirically validate that adversarial examples at a flat local region tend to have good transferability by introducing a penalized gradient norm to the original loss function. Since directly optimizing the gradient regularization norm is computationally expensive and intractable for generating adversarial examples, we propose an approximation optimization method to simplify the gradient update of the objective function. Specifically, we randomly sample an example and adopt a first-order procedure to approximate the curvature of the second-order Hessian matrix, which makes computing more efficient by interpolating two Jacobian matrices. Meanwhile, in order to obtain a more stable gradient direction, we randomly sample multiple examples and average the gradients of these examples to reduce the variance due to random sampling during the iterative process. Extensive experimental results on the ImageNet-compatible dataset show that the proposed method can generate adversarial examples at flat local regions, and significantly improve the adversarial transferability on either normally trained models or adversarially trained models than the state-of-the-art attacks. Our codes are available at: https://github.com/Trustworthy-AI-Group/PGN.

----

## [3073] From Trainable Negative Depth to Edge Heterophily in Graphs

**Authors**: *Yuchen Yan, Yuzhong Chen, Huiyuan Chen, Minghua Xu, Mahashweta Das, Hao Yang, Hanghang Tong*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/de2d52c5cf2bea853ef39bb2e1535dde-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/de2d52c5cf2bea853ef39bb2e1535dde-Abstract-Conference.html)

**Abstract**:

Finding the proper depth $d$ of a graph convolutional network (GCN) that provides strong representation ability has drawn significant attention, yet nonetheless  largely  remains an open problem for the graph learning community. Although noteworthy progress has been made, the depth or the number of layers of a corresponding GCN is realized by a series of graph convolution operations, which naturally makes $d$ a positive integer ($d \in \mathbb{N}+$). An interesting question is whether breaking the constraint of $\mathbb{N}+$ by making $d$ a real number ($d \in \mathbb{R}$) can bring new insights into graph learning mechanisms. In this work, by redefining GCN's depth $d$ as a trainable parameter continuously adjustable within $(-\infty,+\infty)$, we open a new door of controlling its signal processing capability to model graph homophily/heterophily (nodes with similar/dissimilar labels/attributes tend to be inter-connected). A simple and powerful GCN model TEDGCN, is proposed to retain the simplicity of GCN and meanwhile automatically search for the optimal $d$ without the prior knowledge regarding whether the input graph is homophilic or heterophilic. Negative-valued $d$ intrinsically enables high-pass frequency filtering functionality via augmented topology for graph heterophily. Extensive experiments demonstrate the superiority of TEDGCN on node classification tasks for a variety of homophilic and heterophilic graphs.

----

## [3074] ADGym: Design Choices for Deep Anomaly Detection

**Authors**: *Minqi Jiang, Chaochuan Hou, Ao Zheng, Songqiao Han, Hailiang Huang, Qingsong Wen, Xiyang Hu, Yue Zhao*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/de670b9d118229d09d9a9bd9dec2598b-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/de670b9d118229d09d9a9bd9dec2598b-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Deep learning (DL) techniques have recently found success in anomaly detection (AD) across various fields such as finance, medical services, and cloud computing. However, most of the current research tends to view deep AD algorithms as a whole, without dissecting the contributions of individual design choices like loss functions and network architectures. This view tends to diminish the value of preliminary steps like data preprocessing, as more attention is given to newly designed loss functions, network architectures, and learning paradigms. In this paper, we aim to bridge this gap by asking two key questions: (i) Which design choices in deep AD methods are crucial for detecting anomalies? (ii) How can we automatically select the optimal design choices for a given AD dataset, instead of relying on generic, pre-existing solutions? To address these questions, we introduce ADGym, a platform specifically crafted for comprehensive evaluation and automatic selection of AD design elements in deep methods. Our extensive experiments reveal that relying solely on existing leading methods is not sufficient. In contrast, models developed using ADGym significantly surpass current state-of-the-art techniques.

----

## [3075] Low-shot Object Learning with Mutual Exclusivity Bias

**Authors**: *Anh Thai, Ahmad Humayun, Stefan Stojanov, Zixuan Huang, Bikram Boote, James M. Rehg*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/de6ff07cbd222c10d694c2b2f732aceb-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/de6ff07cbd222c10d694c2b2f732aceb-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

This paper introduces Low-shot Object Learning with Mutual Exclusivity Bias (LSME), the first computational framing of mutual exclusivity bias, a phenomenon commonly observed in infants during word learning. We provide a novel dataset, comprehensive baselines, and a SOTA method to enable the ML community to tackle this challenging learning task. The goal of LSME is to analyze an RGB image of a scene containing multiple objects and correctly associate a previously-unknown object instance with a provided category label. This association is then used to perform low-shot learning to test category generalization. We provide a data generation pipeline for the LSME problem and conduct a thorough analysis of the factors that contribute to its difficulty. Additionally, we evaluate the performance of multiple baselines, including state-of-the-art foundation models. Finally, we present a baseline approach that outperforms state-of-the-art models in terms of low-shot accuracy. Code and data are available at https://github.com/rehg-lab/LSME.

----

## [3076] GPT-ST: Generative Pre-Training of Spatio-Temporal Graph Neural Networks

**Authors**: *Zhonghang Li, Lianghao Xia, Yong Xu, Chao Huang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/de7858e3e7f9f0f7b2c7bfdc86f6d928-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/de7858e3e7f9f0f7b2c7bfdc86f6d928-Abstract-Conference.html)

**Abstract**:

In recent years, there has been a rapid development of spatio-temporal prediction techniques in response to the increasing demands of traffic management and travel planning. While advanced end-to-end models have achieved notable success in improving predictive performance, their integration and expansion pose significant challenges. This work aims to address these challenges by introducing a spatio-temporal pre-training framework that seamlessly integrates with downstream baselines and enhances their performance. The framework is built upon two key designs: (i) We propose a spatio-temporal mask autoencoder as a pre-training model for learning spatio-temporal dependencies. The model incorporates customized parameter learners and hierarchical spatial pattern encoding networks. These modules are specifically designed to capture spatio-temporal customized representations and intra- and inter-cluster region semantic relationships, which have often been neglected in existing approaches. (ii) We introduce an adaptive mask strategy as part of the pre-training mechanism. This strategy guides the mask autoencoder in learning robust spatio-temporal representations and facilitates the modeling of different relationships, ranging from intra-cluster to inter-cluster, in an easy-to-hard training manner. Extensive experiments conducted on representative benchmarks demonstrate the effectiveness of our proposed method. We have made our model implementation publicly available at https://github.com/HKUDS/GPT-ST.

----

## [3077] Direct Preference-based Policy Optimization without Reward Modeling

**Authors**: *Gaon An, Junhyeok Lee, Xingdong Zuo, Norio Kosaka, Kyung-Min Kim, Hyun Oh Song*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/de8bd6b2b01cfa788e63f62e5b9a99b9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/de8bd6b2b01cfa788e63f62e5b9a99b9-Abstract-Conference.html)

**Abstract**:

Preference-based reinforcement learning (PbRL) is an approach that enables RL agents to learn from preference, which is particularly useful when formulating a reward function is challenging. Existing PbRL methods generally involve a two-step procedure: they first learn a reward model based on given preference data and then employ off-the-shelf reinforcement learning algorithms using the learned reward model. However, obtaining an accurate reward model solely from preference information, especially when the preference is from human teachers, can be difficult. Instead, we propose a PbRL algorithm that directly learns from preference without requiring any reward modeling. To achieve this, we adopt a contrastive learning framework to design a novel policy scoring metric that assigns a high score to policies that align with the given preferences. We apply our algorithm to offline RL tasks with actual human preference labels and show that our algorithm outperforms or is on par with the existing PbRL methods. Notably, on high-dimensional control tasks, our algorithm surpasses offline RL methods that learn with ground-truth reward information. Finally, we show that our algorithm can be successfully applied to fine-tune large language models.

----

## [3078] On the Identifiability and Interpretability of Gaussian Process Models

**Authors**: *Jiawen Chen, Wancen Mu, Yun Li, Didong Li*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/dea2b4f9012686bcc1f59a62bcd28158-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/dea2b4f9012686bcc1f59a62bcd28158-Abstract-Conference.html)

**Abstract**:

In this paper, we critically examine the prevalent practice of using additive mixtures of Mat\'ern kernels in single-output Gaussian process (GP) models and explore the properties of multiplicative mixtures of Mat\'ern kernels for multi-output GP models. For the single-output case, we derive a series of theoretical results showing that the smoothness of a mixture of Mat\'ern kernels is determined by the least smooth component and that a GP with such a kernel is effectively equivalent to the least smooth kernel component. Furthermore, we demonstrate that none of the mixing weights or parameters within individual kernel components are identifiable. We then turn our attention to multi-output GP models and analyze the identifiability of the covariance matrix $A$ in the multiplicative kernel $K(x,y) = AK_0(x,y)$, where $K_0$ is a standard single output kernel such as Mat\'ern. We show that $A$ is identifiable up to a multiplicative constant, suggesting that multiplicative mixtures are well suited for multi-output tasks. Our findings are supported by extensive simulations and real applications for both single- and multi-output settings. This work provides insight into kernel selection and interpretation for GP models, emphasizing the importance of choosing appropriate kernel structures for different tasks.

----

## [3079] Enhancing Motion Deblurring in High-Speed Scenes with Spike Streams

**Authors**: *Shiyan Chen, Jiyuan Zhang, Yajing Zheng, Tiejun Huang, Zhaofei Yu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/dead3d8ff3f9198e38a36a950ebbcafd-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/dead3d8ff3f9198e38a36a950ebbcafd-Abstract-Conference.html)

**Abstract**:

Traditional cameras produce desirable vision results but struggle with motion blur in high-speed scenes due to long exposure windows. Existing frame-based deblurring algorithms face challenges in extracting useful motion cues from severely blurred images. Recently, an emerging bio-inspired vision sensor known as the spike camera has achieved an extremely high frame rate while preserving rich spatial details, owing to its novel sampling mechanism. However, typical binary spike streams are relatively low-resolution, degraded image signals devoid of color information, making them unfriendly to human vision. In this paper, we propose a novel approach that integrates the two modalities from two branches, leveraging spike streams as auxiliary visual cues for guiding deblurring in high-speed motion scenes. We propose the first spike-based motion deblurring model with bidirectional information complementarity. We introduce a content-aware motion magnitude attention module that utilizes learnable mask to extract relevant information from blurry images effectively, and we incorporate a transposed cross-attention fusion module to efficiently combine features from both spike data and blurry RGB images.Furthermore, we build two extensive synthesized datasets for training and validation purposes, encompassing high-temporal-resolution spikes, blurry images, and corresponding sharp images. The experimental results demonstrate that our method effectively recovers clear RGB images from highly blurry scenes and outperforms state-of-the-art deblurring algorithms in multiple settings.

----

## [3080] Faith and Fate: Limits of Transformers on Compositionality

**Authors**: *Nouha Dziri, Ximing Lu, Melanie Sclar, Xiang Lorraine Li, Liwei Jiang, Bill Yuchen Lin, Sean Welleck, Peter West, Chandra Bhagavatula, Ronan Le Bras, Jena D. Hwang, Soumya Sanyal, Xiang Ren, Allyson Ettinger, Zaïd Harchaoui, Yejin Choi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/deb3c28192f979302c157cb653c15e90-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/deb3c28192f979302c157cb653c15e90-Abstract-Conference.html)

**Abstract**:

Transformer large language models (LLMs) have sparked admiration for their exceptional performance on tasks that demand intricate multi-step reasoning. Yet, these models simultaneously show failures on surprisingly trivial problems. This begs the question: Are these errors incidental, or do they signal more substantial limitations?In an attempt to demystify transformer LLMs, we investigate the limits of these models across three representative compositional tasks---multi-digit multiplication, logic grid puzzles, and a classic dynamic programming problem. These tasks require breaking problems down into sub-steps and synthesizing these steps into a precise answer.  We formulate compositional tasks as computation graphs to systematically quantify the level of complexity, and break down reasoning steps into intermediate sub-procedures. Our empirical findings suggest that transformer LLMs solve compositional tasks by reducing multi-step compositional reasoning into linearized subgraph matching, without necessarily developing systematic problem-solving skills.  To round off our empirical study, we provide theoretical arguments on abstract multi-step reasoning problems that highlight how autoregressive generations' performance can rapidly decay with increased task complexity.

----

## [3081] Towards a fuller understanding of neurons with Clustered Compositional Explanations

**Authors**: *Biagio La Rosa, Leilani Gilpin, Roberto Capobianco*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/debd0ae2083160397a22a4a8831c7230-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/debd0ae2083160397a22a4a8831c7230-Abstract-Conference.html)

**Abstract**:

Compositional Explanations is a method for identifying logical formulas of concepts that approximate the neurons' behavior. However, these explanations are linked to the small spectrum of neuron activations (i.e., the highest ones) used to check the alignment, thus lacking completeness. In this paper, we propose a generalization, called Clustered Compositional Explanations, that combines  Compositional Explanations with clustering and a novel search heuristic to approximate a broader spectrum of the neuron behavior. We define and address the problems connected to the application of these methods to multiple ranges of activations, analyze the insights retrievable by using our algorithm, and propose desiderata qualities that can be used to study the explanations returned by different algorithms.

----

## [3082] TpuGraphs: A Performance Prediction Dataset on Large Tensor Computational Graphs

**Authors**: *Mangpo Mangpo Phothilimthana, Sami Abu-El-Haija, Kaidi Cao, Bahare Fatemi, Michael Burrows, Charith Mendis, Bryan Perozzi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ded1a89e2b3b925444ada973af66336e-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/ded1a89e2b3b925444ada973af66336e-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Precise hardware performance models play a crucial role in code optimizations. They can assist compilers in making heuristic decisions or aid autotuners in identifying the optimal configuration for a given program. For example, the autotuner for XLA, a machine learning compiler, discovered 10â€“20\% speedup on state-of-the-art models serving substantial production traffic at Google. Although there exist a few datasets for program performance prediction, they target small sub-programs such as basic blocks or kernels. This paper introduces TpuGraphs, a performance prediction dataset on full tensor programs, represented as computational graphs, running on Tensor Processing Units (TPUs). Each graph in the dataset represents the main computation of a machine learning workload, e.g., a training epoch or an inference step. Each data sample contains a computational graph, a compilation configuration, and the execution time of the graph when compiled with the configuration. The graphs in the dataset are collected from open-source machine learning programs, featuring popular model architectures (e.g., ResNet, EfficientNet, Mask R-CNN, and Transformer). TpuGraphs provides 25x more graphs than the largest graph property prediction dataset (with comparable graph sizes), and 770x larger graphs on average compared to existing performance prediction datasets on machine learning programs. This graph-level prediction task on large graphs introduces new challenges in learning, ranging from scalability, training efficiency, to model quality.

----

## [3083] ScaleLong: Towards More Stable Training of Diffusion Model via Scaling Network Long Skip Connection

**Authors**: *Zhongzhan Huang, Pan Zhou, Shuicheng Yan, Liang Lin*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ded98d28f82342a39f371c013dfb3058-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ded98d28f82342a39f371c013dfb3058-Abstract-Conference.html)

**Abstract**:

In diffusion models, UNet is the most popular network backbone, since its long skip connects (LSCs) to connect distant network blocks can aggregate long-distant information and alleviate vanishing gradient. Unfortunately, UNet often suffers from unstable training in diffusion models which can be alleviated by scaling its LSC coefficients smaller. However, theoretical understandings of the instability of UNet in diffusion models and also the performance improvement of LSC scaling remain absent yet. To solve this issue, we theoretically show that the coefficients of LSCs in UNet have big effects on the stableness of the forward and backward propagation and robustness of UNet. Specifically, the hidden feature and gradient of UNet at any layer can oscillate and their oscillation ranges are actually large which explains the instability of UNet training. Moreover, UNet is also provably sensitive to perturbed input, and predicts an output distant from the desired output, yielding oscillatory loss and thus oscillatory gradient. Besides, we also observe the theoretical benefits of the LSC coefficient scaling of UNet in the stableness of hidden features and gradient and also robustness. Finally,   inspired by our theory, we propose an effective coefficient scaling framework ScaleLong  that scales the coefficients of LSC  in UNet and better improve the  training stability of UNet. Experimental results on CIFAR10, CelebA, ImageNet and COCO show that our methods are superior to stabilize training, and yield about 1.5x training acceleration on different diffusion models with UNet or UViT backbones.

----

## [3084] Faster approximate subgraph counts with privacy

**Authors**: *Dung Nguyen, Mahantesh Halappanavar, Venkatesh Srinivasan, Anil Vullikanti*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/deddcfbf08f57489b0088b71a00db640-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/deddcfbf08f57489b0088b71a00db640-Abstract-Conference.html)

**Abstract**:

One of the most common problems studied in the context of differential privacy for graph data is counting the number of non-induced embeddings of a subgraph in a given graph. These counts have very high global sensitivity. Therefore, adding noise based on powerful alternative techniques, such as smooth sensitivity and higher-order local sensitivity have been shown to give significantly better accuracy. However, all these alternatives to global sensitivity become computationally very expensive, and to date efficient polynomial time algorithms are known only for few selected subgraphs, such as triangles, $k$-triangles, and $k$-stars.In this paper, we show that good approximations to these sensitivity metrics can be still used to get private algorithms.Using this approach, we much faster algorithms for privately counting the number of triangles in real-world social networks, which can be easily parallelized.We also give a private polynomial time algorithm for counting any constant size subgraph using less noise than the global sensitivity; we show this can be improved significantly for counting paths in special classes of graphs.

----

## [3085] Recasting Continual Learning as Sequence Modeling

**Authors**: *Soochan Lee, Jaehyeon Son, Gunhee Kim*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/dee254cdacbab59f17dc6a8fbdffa59f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/dee254cdacbab59f17dc6a8fbdffa59f-Abstract-Conference.html)

**Abstract**:

In this work, we aim to establish a strong connection between two significant bodies of machine learning research: continual learning and sequence modeling.That is, we propose to formulate continual learning as a sequence modeling problem, allowing advanced sequence models to be utilized for continual learning.Under this formulation, the continual learning process becomes the forward pass of a sequence model.By adopting the meta-continual learning (MCL) framework, we can train the sequence model at the meta-level, on multiple continual learning episodes.As a specific example of our new formulation, we demonstrate the application of Transformers and their efficient variants as MCL methods.Our experiments on seven benchmarks, covering both classification and regression, show that sequence models can be an attractive solution for general MCL.

----

## [3086] Multiply Robust Federated Estimation of Targeted Average Treatment Effects

**Authors**: *Larry Han, Zhu Shen, José R. Zubizarreta*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/def4492b32f0248a0e4d92cc46bbdaad-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/def4492b32f0248a0e4d92cc46bbdaad-Abstract-Conference.html)

**Abstract**:

Federated or multi-site studies have distinct advantages over single-site studies, including increased generalizability, the ability to study underrepresented populations, and the opportunity to study rare exposures and outcomes. However, these studies are complicated by the need to preserve the privacy of each individual's data, heterogeneity in their covariate distributions, and different data structures between sites. We propose a novel federated approach to derive valid causal inferences for a target population using multi-site data. We adjust for covariate shift and accommodate covariate mismatch between sites by developing a multiply-robust and privacy-preserving nuisance function estimation approach. Our methodology incorporates transfer learning to estimate ensemble weights to combine information from source sites. We show that these learned weights are efficient and optimal under different scenarios. We showcase the finite sample advantages of our approach in terms of efficiency and robustness compared to existing state-of-the-art approaches. We apply our approach to study the treatment effect of percutaneous coronary intervention (PCI) on the duration of hospitalization for patients experiencing acute myocardial infarction (AMI) with data from the Centers for Medicare \& Medicaid Services (CMS).

----

## [3087] Learning Motion Refinement for Unsupervised Face Animation

**Authors**: *Jiale Tao, Shuhang Gu, Wen Li, Lixin Duan*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/df2df463f98abc4de7734dbd0b0dc49d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/df2df463f98abc4de7734dbd0b0dc49d-Abstract-Conference.html)

**Abstract**:

Unsupervised face animation aims to generate a human face video based on theappearance of a source image, mimicking the motion from a driving video. Existingmethods typically adopted a prior-based motion model (e.g., the local affine motionmodel or the local thin-plate-spline motion model). While it is able to capturethe coarse facial motion, artifacts can often be observed around the tiny motionin local areas (e.g., lips and eyes), due to the limited ability of these methodsto model the finer facial motions. In this work, we design a new unsupervisedface animation approach to learn simultaneously the coarse and finer motions. Inparticular, while exploiting the local affine motion model to learn the global coarsefacial motion, we design a novel motion refinement module to compensate forthe local affine motion model for modeling finer face motions in local areas. Themotion refinement is learned from the dense correlation between the source anddriving images. Specifically, we first construct a structure correlation volume basedon the keypoint features of the source and driving images. Then, we train a modelto generate the tiny facial motions iteratively from low to high resolution. Thelearned motion refinements are combined with the coarse motion to generate thenew image. Extensive experiments on widely used benchmarks demonstrate thatour method achieves the best results among state-of-the-art baselines.

----

## [3088] Masked Space-Time Hash Encoding for Efficient Dynamic Scene Reconstruction

**Authors**: *Feng Wang, Zilong Chen, Guokang Wang, Yafei Song, Huaping Liu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/df31126302921ca9351fab73923a172f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/df31126302921ca9351fab73923a172f-Abstract-Conference.html)

**Abstract**:

In this paper, we propose the Masked Space-Time Hash encoding (MSTH), a novel method for efficiently reconstructing dynamic 3D scenes from multi-view or monocular videos. Based on the observation that dynamic scenes often contain substantial static areas that result in redundancy in storage and computations, MSTH represents a dynamic scene as a weighted combination of a 3D hash encoding and a 4D hash encoding. The weights for the two components are represented by a learnable mask which is guided by an uncertainty-based objective to reflect the spatial and temporal importance of each 3D position. With this design, our method can reduce the hash collision rate by avoiding redundant queries and modifications on static areas, making it feasible to represent a large number of space-time voxels by hash tables with small size.Besides, without the requirements to fit the large numbers of temporally redundant features independently, our method is easier to optimize and converge rapidly with only twenty minutes of training for a 300-frame dynamic scene. We evaluate our method on extensive dynamic scenes. As a result, MSTH obtains consistently better results than previous state-of-the-art methods with only 20 minutes of training time and 130 MB of memory storage.

----

## [3089] Bifurcations and loss jumps in RNN training

**Authors**: *Lukas Eisenmann, Zahra Monfared, Niclas Alexander Göring, Daniel Durstewitz*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/df334022279996b07e0870a629c18857-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/df334022279996b07e0870a629c18857-Abstract-Conference.html)

**Abstract**:

Recurrent neural networks (RNNs) are popular machine learning tools for modeling and forecasting sequential data and for inferring dynamical systems (DS) from observed time series. Concepts from DS theory (DST) have variously been used to further our understanding of both, how trained RNNs solve complex tasks, and the training process itself. Bifurcations are particularly important phenomena in DS, including RNNs, that refer to topological (qualitative) changes in a system's dynamical behavior as one or more of its parameters are varied. Knowing the bifurcation structure of an RNN will thus allow to deduce many of its computational and dynamical properties, like its sensitivity to parameter variations or its behavior during training. In particular, bifurcations may account for sudden loss jumps observed in RNN training that could severely impede the training process. Here we first mathematically prove for a particular class of ReLU-based RNNs that certain bifurcations are indeed associated with loss gradients tending toward infinity or zero. We then introduce a novel heuristic algorithm for detecting all fixed points and $k$-cycles in ReLU-based RNNs and their existence and stability regions, hence bifurcation manifolds in parameter space. In contrast to previous numerical algorithms for finding fixed points and common continuation methods, our algorithm provides $\textit{exact}$ results and returns fixed points and cycles up to high orders with surprisingly good scaling behavior. We exemplify the algorithm on the analysis of the training process of RNNs, and find that the recently introduced technique of generalized teacher forcing completely avoids certain types of bifurcations in training. Thus, besides facilitating the DST analysis of trained RNNs, our algorithm provides a powerful instrument for analyzing the training process itself.

----

## [3090] Neural Foundations of Mental Simulation: Future Prediction of Latent Representations on Dynamic Scenes

**Authors**: *Aran Nayebi, Rishi Rajalingham, Mehrdad Jazayeri, Guangyu Robert Yang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/df438caa36714f69277daa92d608dd63-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/df438caa36714f69277daa92d608dd63-Abstract-Conference.html)

**Abstract**:

Humans and animals have a rich and flexible understanding of the physical world, which enables them to infer the underlying dynamical trajectories of objects and events, plausible future states, and use that to plan and anticipate the consequences of actions.However, the neural mechanisms underlying these computations are unclear.We combine a goal-driven modeling approach with dense neurophysiological data and high-throughput human behavioral readouts that contain thousands of comparisons to directly impinge on this question.Specifically, we construct and evaluate several classes of sensory-cognitive networks to predict the future state of rich, ethologically-relevant environments, ranging from self-supervised end-to-end models with pixel-wise or object-slot objectives, to models that future predict in the latent space of purely static image-pretrained or dynamic video-pretrained foundation models.We find that ``scale is \emph{not} all you need'', and that many state-of-the-art machine learning models fail to perform well on our neural and behavioral benchmarks for future prediction.In fact, only one class of models matches these data well overall.We find that neural responses are currently best predicted by models trained to predict the future state of their environment in the \emph{latent} space of pretrained foundation models optimized for \emph{dynamic} scenes in a self-supervised manner.These models also approach the neurons' ability to predict the environmental state variables that are visually hidden from view, despite not being explicitly trained to do so.Finally, we find that not all foundation model latents are equal.Notably, models that future predict in the latent space of video foundation models that are optimized to support a \emph{diverse} range of egocentric sensorimotor tasks, reasonably match \emph{both} human behavioral error patterns and neural dynamics across all environmental scenarios that we were able to test.Overall, these findings suggest that the neural mechanisms and behaviors of primate mental simulation have strong inductive biases associated with them, and are thus far most consistent with being optimized to future predict on \emph{reusable} visual representations that are useful for Embodied AI more generally.

----

## [3091] Learning Visual Prior via Generative Pre-Training

**Authors**: *Jinheng Xie, Kai Ye, Yudong Li, Yuexiang Li, Kevin Qinghong Lin, Yefeng Zheng, Linlin Shen, Mike Zheng Shou*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/df4f6e43446b1ee29c5a33d32c279f83-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/df4f6e43446b1ee29c5a33d32c279f83-Abstract-Conference.html)

**Abstract**:

Various stuff and things in visual data possess specific traits, which can be learned by deep neural networks and are implicitly represented as the visual prior, e.g., object location and shape, in the model. Such prior potentially impacts many vision tasks. For example, in conditional image synthesis, spatial conditions failing to adhere to the prior can result in visually inaccurate synthetic results. This work aims to explicitly learn the visual prior and enable the customization of sampling. Inspired by advances in language modeling, we propose to learn Visual prior via Generative Pre-Training, dubbed VisorGPT. By discretizing visual locations, e.g., bounding boxes, human pose, and instance masks, into sequences, VisorGPT can model visual prior through likelihood maximization. Besides, prompt engineering is investigated to unify various visual locations and enable customized sampling of sequential outputs from the learned prior. Experimental results demonstrate the effectiveness of VisorGPT in modeling visual prior and extrapolating to novel scenes, potentially motivating that discrete visual locations can be integrated into the learning paradigm of current language models to further perceive visual world. Code is available at https://sierkinhane.github.io/visor-gpt.

----

## [3092] Operator Learning with Neural Fields: Tackling PDEs on General Geometries

**Authors**: *Louis Serrano, Lise Le Boudec, Armand Kassaï Koupaï, Thomas X. Wang, Yuan Yin, Jean-Noël Vittaut, Patrick Gallinari*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/df54302388bbc145aacaa1a54a4a5933-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/df54302388bbc145aacaa1a54a4a5933-Abstract-Conference.html)

**Abstract**:

Machine learning approaches for solving partial differential equations require learning mappings between function spaces. While convolutional or graph neural networks are constrained to discretized functions, neural operators present a promising milestone toward mapping functions directly. Despite impressive results they still face challenges with respect to the domain geometry and typically rely on some form of discretization. In order to alleviate such limitations, we present CORAL, a new method that leverages coordinate-based networks for solving PDEs on general geometries. CORAL is designed to remove constraints on the input mesh, making it applicable to any spatial sampling and geometry. Its ability extends to diverse problem domains, including PDE solving, spatio-temporal forecasting, and inverse problems like geometric design. CORAL demonstrates robust performance across multiple resolutions and performs well in both convex and non-convex domains, surpassing or performing on par with state-of-the-art models.

----

## [3093] Learning Dense Flow Field for Highly-accurate Cross-view Camera Localization

**Authors**: *Zhenbo Song, Xianghui Ze, Jianfeng Lu, Yujiao Shi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/df5f94d6ac6e13d830d70536cde9f0d2-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/df5f94d6ac6e13d830d70536cde9f0d2-Abstract-Conference.html)

**Abstract**:

This paper addresses the problem of estimating the 3-DoF camera pose for a ground-level image with respect to a satellite image that encompasses the local surroundings. We propose a novel end-to-end approach that leverages the learning of dense pixel-wise flow fields in pairs of ground and satellite images to calculate the camera pose. Our approach differs from existing methods by constructing the feature metric at the pixel level, enabling full-image supervision for learning distinctive geometric configurations and visual appearances across views. Specifically, our method employs two distinct convolution networks for ground and satellite feature extraction. Then, we project the ground feature map to the bird's eye view (BEV) using a fixed camera height assumption to achieve preliminary geometric alignment. To further establish the content association between the BEV and satellite features, we introduce a residual convolution block to refine the projected BEV feature. Optical flow estimation is performed on the refined BEV feature map and the satellite feature map using flow decoder networks based on RAFT. After obtaining dense flow correspondences, we apply the least square method to filter matching inliers and regress the ground camera pose. Extensive experiments demonstrate significant improvements compared to state-of-the-art methods. Notably, our approach reduces the median localization error by 89\%, 19\%, 80\%, and 35\% on the KITTI, Ford multi-AV, VIGOR, and Oxford RobotCar datasets, respectively.

----

## [3094] Spatially Resolved Gene Expression Prediction from Histology Images via Bi-modal Contrastive Learning

**Authors**: *Ronald Xie, Kuan Pang, Sai Chung, Catia Perciani, Sonya MacParland, Bo Wang, Gary D. Bader*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/df656d6ed77b565e8dcdfbf568aead0a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/df656d6ed77b565e8dcdfbf568aead0a-Abstract-Conference.html)

**Abstract**:

Histology imaging is an important tool in medical diagnosis and research, enabling the examination of tissue structure and composition at the microscopic level. Understanding the underlying molecular mechanisms of tissue architecture is critical in uncovering disease mechanisms and developing effective treatments.Gene expression profiling provides insight into the molecular processes underlying tissue architecture, but the process can be time-consuming and expensive. We present BLEEP (Bi-modaL Embedding for Expression Prediction), a bi-modal embedding framework capable of generating spatially resolved gene expression profiles of whole-slide Hematoxylin and eosin (H&E) stained histology images. BLEEP uses contrastive learning to construct a low-dimensional joint embedding space from a reference dataset using paired image and expression profiles at micrometer resolution. With this approach, the gene expression of any query image patch can be imputed using the expression profiles from the reference dataset. We demonstrate BLEEPâ€™s effectiveness in gene expression prediction by benchmarking its performance on a human liver tissue dataset captured using the 10x Visium platform, where it achieves significant improvements over existing methods. Our results demonstrate the potential of BLEEP to provide insights into the molecular mechanisms underlying tissue architecture, with important implications in diagnosis and research of various diseases. The proposed approach can significantly reduce the time and cost associated with gene expression profiling, opening up new avenues for high-throughput analysis of histology images for both research and clinical applications.

----

## [3095] Causal-structure Driven Augmentations for Text OOD Generalization

**Authors**: *Amir Feder, Yoav Wald, Claudia Shi, Suchi Saria, David M. Blei*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/df88b275bef31ac96c85f0c4013734fc-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/df88b275bef31ac96c85f0c4013734fc-Abstract-Conference.html)

**Abstract**:

The reliance of text classifiers on spurious correlations can lead to poor generalization at deployment, raising concerns about their use in safety-critical domains such as healthcare. In this work, we propose to use counterfactual data augmentation, guided by knowledge of the causal structure of the data, to simulate interventions on spurious features and to learn more robust text classifiers. We show that this strategy is appropriate in prediction problems where the label is spuriously correlated with an attribute. Under the assumptions of such problems, we discuss the favorable sample complexity of counterfactual data augmentation, compared to importance re-weighting. Pragmatically, we match examples using auxiliary data, based on diff-in-diff methodology, and use a large language model (LLM) to represent a conditional probability of text. Through extensive experimentation on learning caregiver-invariant predictors of clinical diagnoses from medical narratives and on semi-synthetic data, we demonstrate that our method for simulating interventions improves out-of-distribution (OOD) accuracy compared to baseline invariant learning algorithms.

----

## [3096] Adversarial Counterfactual Environment Model Learning

**Authors**: *Xiong-Hui Chen, Yang Yu, Zhengmao Zhu, Zhihua Yu, Zhenjun Chen, Chenghe Wang, Yinan Wu, Rong-Jun Qin, Hongqiu Wu, Ruijin Ding, Fangsheng Huang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/df927a06a0d9f5f06d9cd4a91ce58e56-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/df927a06a0d9f5f06d9cd4a91ce58e56-Abstract-Conference.html)

**Abstract**:

An accurate environment dynamics model is crucial for various downstream tasks in sequential decision-making, such as counterfactual prediction, off-policy evaluation, and offline reinforcement learning. Currently, these models were learned through empirical risk minimization (ERM) by step-wise fitting of historical transition data. This way was previously believed unreliable over long-horizon rollouts because of the compounding errors, which can lead to uncontrollable inaccuracies in predictions. In this paper, we find that the challenge extends beyond just long-term prediction errors: we reveal that even when planning with one step, learned dynamics models can also perform poorly due to the selection bias of behavior policies during data collection. This issue will significantly mislead the policy optimization process even in identifying single-step optimal actions, further leading to a greater risk in sequential decision-making scenarios.To tackle this problem, we introduce a novel model-learning objective called adversarial weighted empirical risk minimization (AWRM).  AWRM incorporates an adversarial policy that exploits the model to generate a data distribution that weakens the model's prediction accuracy, and subsequently, the model is learned under this adversarial data distribution.We implement a practical algorithm, GALILEO, for AWRM and evaluate it on two synthetic tasks, three continuous-control tasks, and  \textit{a real-world application}. The experiments demonstrate that GALILEO can accurately predict counterfactual actions and improve various downstream tasks, including offline policy evaluation and improvement, as well as online decision-making.

----

## [3097] Finding Safe Zones of Markov Decision Processes Policies

**Authors**: *Lee Cohen, Yishay Mansour, Michal Moshkovitz*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/dfaa29ed28dfa175bcc5e2a54aa199f8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/dfaa29ed28dfa175bcc5e2a54aa199f8-Abstract-Conference.html)

**Abstract**:

Given a policy of a Markov Decision Process, we define a SafeZone as a subset of states, such that most of the policy's trajectories are confined to this subset. The quality of a SafeZone is parameterized by the number of states and the escape probability, i.e., the probability that a random trajectory will leave the subset. SafeZones are especially interesting when they have a small number of states and low escape probability. We study the complexity of finding optimal SafeZones, and show that in general, the problem is computationally hard. For this reason, we concentrate on finding approximate SafeZones. Our main result is a bi-criteria approximation learning algorithm with a factor of almost $2$  approximation for both the escape probability and \newprob size, using a polynomial size sample complexity.

----

## [3098] Zero-One Laws of Graph Neural Networks

**Authors**: *Sam Adam-Day, Theodor-Mihai Iliant, Ismail Ilkan Ceylan*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/dfba85bc32a3cb63a96d1412062b4d8e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/dfba85bc32a3cb63a96d1412062b4d8e-Abstract-Conference.html)

**Abstract**:

Graph neural networks (GNNs) are the de facto standard deep learning architectures for machine learning on graphs.  This has led to a large body of work analyzing the capabilities and limitations of these models,  particularly pertaining to their representation and extrapolation capacity.  We offer a novel theoretical perspective on the representation and extrapolation capacity of GNNs, by answering the question: how do GNNs behave as the number of graph nodes become very large? Under mild assumptions, we show that when we draw graphs of increasing size from the Erdős–Rényi model, the probability that such graphs are mapped to a particular output by a class of GNN classifiers tends to either zero or one. This class includes the popular graph convolutional network architecture. The result establishes `zero-one laws' for these GNNs, and analogously to other convergence laws,  entails theoretical limitations on their capacity.  We empirically verify our results, observing that the theoretical asymptotic limits are evident already on relatively small graphs.

----

## [3099] Towards Revealing the Mystery behind Chain of Thought: A Theoretical Perspective

**Authors**: *Guhao Feng, Bohang Zhang, Yuntian Gu, Haotian Ye, Di He, Liwei Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/dfc310e81992d2e4cedc09ac47eff13e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/dfc310e81992d2e4cedc09ac47eff13e-Abstract-Conference.html)

**Abstract**:

Recent studies have discovered that Chain-of-Thought prompting (CoT) can dramatically improve the performance of Large Language Models (LLMs), particularly when dealing with complex tasks involving mathematics or reasoning. Despite the enormous empirical success, the underlying mechanisms behind CoT and how it unlocks the potential of LLMs remain elusive. In this paper, we take a first step towards theoretically answering these questions. Specifically, we examine the expressivity of LLMs with CoT in solving fundamental mathematical and decision-making problems. By using circuit complexity theory, we first give impossibility results showing that bounded-depth Transformers are unable to directly produce correct answers for basic arithmetic/equation tasks unless the model size grows super-polynomially with respect to the input length. In contrast, we then prove by construction that autoregressive Transformers of constant size suffice to solve both tasks by generating CoT derivations using a commonly used math language format. Moreover, we show LLMs with CoT can handle a general class of decision-making problems known as Dynamic Programming, thus justifying their power in tackling complex real-world tasks. Finally, an extensive set of experiments show that, while Transformers always fail to directly predict the answers, they can consistently learn to generate correct solutions step-by-step given sufficient CoT demonstrations.

----

## [3100] Divide, Evaluate, and Refine: Evaluating and Improving Text-to-Image Alignment with Iterative VQA Feedback

**Authors**: *Jaskirat Singh, Liang Zheng*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/dfd0bd56e8a6f82d1619f5d093d5f9ca-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/dfd0bd56e8a6f82d1619f5d093d5f9ca-Abstract-Conference.html)

**Abstract**:

The field of text-conditioned image generation has made unparalleled progress with the recent advent of latent diffusion models. While revolutionary, as the complexity of given text input increases, the current state of art diffusion models may still fail in generating images that accurately convey the semantics of the given prompt. Furthermore, such misalignments are often left undetected by pretrained multi-modal models such as CLIP.  To address these problems, in this paper, we explore a simple yet effective decompositional approach towards both evaluation and improvement of text-to-image alignment. In particular,  we first introduce a Decompositional-Alignment-Score which given a complex caption decomposes it into a set of disjoint assertions. The alignment of each assertion with generated images is then measured using a VQA model. Finally, alignment scores for different assertions are combined aposteriori to give the final text-to-image alignment score. Experimental analysis reveals that the proposed alignment metric shows a significantly higher correlation with human ratings as opposed to traditional CLIP, BLIP scores. Furthermore, we also find that the assertion level alignment scores also provide useful feedback which can then be used in a simple iterative procedure to gradually increase the expressivity of different assertions in the final image outputs. Human user studies indicate that the proposed approach surpasses previous state-of-the-art by 8.7% in overall text-to-image alignment accuracy.

----

## [3101] Distributed Personalized Empirical Risk Minimization

**Authors**: *Yuyang Deng, Mohammad Mahdi Kamani, Pouria Mahdavinia, Mehrdad Mahdavi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/dfee09496a5a8b0b01d9d4c589758832-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/dfee09496a5a8b0b01d9d4c589758832-Abstract-Conference.html)

**Abstract**:

This paper advocates a new paradigm  Personalized Empirical Risk Minimization (PERM) to facilitate learning from heterogeneous  data sources without imposing stringent constraints on computational resources shared by participating devices. In PERM, we aim at  learning  a distinct model for each client by personalizing the aggregation of local empirical losses by effectively estimating the  statistical discrepancy among data distributions, which entails  optimal statistical accuracy for all local distributions and overcomes the data heterogeneity issue.  To learn personalized models at scale,  we propose a distributed algorithm that replaces the standard model averaging with model shuffling to simultaneously optimize PERM objectives for all devices. This also allows to learn distinct model architectures (e.g., neural networks with different number of parameters) for  different clients, thus confining to  underlying  memory and compute resources of individual clients. We rigorously analyze the convergence of proposed algorithm and  conduct experiments  that corroborates the effectiveness of proposed paradigm.

----

## [3102] Training-free Diffusion Model Adaptation for Variable-Sized Text-to-Image Synthesis

**Authors**: *Zhiyu Jin, Xuli Shen, Bin Li, Xiangyang Xue*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e0378e0c642b1d292fcb224e8d5a39b3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e0378e0c642b1d292fcb224e8d5a39b3-Abstract-Conference.html)

**Abstract**:

Diffusion models (DMs) have recently gained attention with state-of-the-art performance in text-to-image synthesis. Abiding by the tradition in deep learning, DMs are trained and evaluated on the images with fixed sizes. However, users are demanding for various images with specific sizes and various aspect ratio. This paper focuses on adapting text-to-image diffusion models to handle such variety while maintaining visual fidelity. First we observe that, during the synthesis, lower resolution images suffer from incomplete object portrayal, while higher resolution images exhibit repetitively disordered presentation. Next, we establish a statistical relationship indicating that attention entropy changes with token quantity, suggesting that models aggregate spatial information in proportion to image resolution. The subsequent interpretation on our observations is that objects are incompletely depicted due to limited spatial information for low resolutions, while repetitively disorganized presentation arises from redundant spatial information for high resolutions. From this perspective, we propose a scaling factor to alleviate the change of attention entropy and mitigate the defective pattern observed. Extensive experimental results validate the efficacy of the proposed scaling factor, enabling models to achieve better visual effects, image quality, and text alignment. Notably, these improvements are achieved without additional training or fine-tuning techniques.

----

## [3103] Enhancing Sharpness-Aware Optimization Through Variance Suppression

**Authors**: *Bingcong Li, Georgios B. Giannakis*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e095c0a3717629aa5497601985bfcf0e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e095c0a3717629aa5497601985bfcf0e-Abstract-Conference.html)

**Abstract**:

Sharpness-aware minimization (SAM) has well documented merits in enhancing generalization of deep neural networks, even without sizable data augmentation. Embracing the geometry of the loss function, where neighborhoods of 'flat minima' heighten generalization ability, SAM seeks 'flat valleys' by minimizing the maximum loss caused by an adversary perturbing parameters within the neighborhood.Although critical to account for sharpness of the loss function, such an 'over-friendly adversary' can curtail the outmost level of generalization. The novel approach of this contribution fosters stabilization of adversaries through variance suppression (VaSSO) to avoid such friendliness. VaSSO's provable stability safeguards its numerical improvement over SAM in model-agnostic tasks, including image classification and machine translation. In addition, experiments confirm that VaSSO endows SAM with robustness against high levels of label noise. Code is available at https://github.com/BingcongLi/VaSSO.

----

## [3104] Efficient Algorithms for Generalized Linear Bandits with Heavy-tailed Rewards

**Authors**: *Bo Xue, Yimu Wang, Yuanyu Wan, Jinfeng Yi, Lijun Zhang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e0982cbc81401df3430ee1ff780dc7a2-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e0982cbc81401df3430ee1ff780dc7a2-Abstract-Conference.html)

**Abstract**:

This paper investigates the problem of generalized linear bandits with heavy-tailed rewards, whose $(1+\epsilon)$-th moment is bounded for some $\epsilon\in (0,1]$. Although there exist methods for generalized linear bandits, most of them focus on bounded or sub-Gaussian rewards and are not well-suited for many real-world scenarios, such as financial markets and web-advertising. To address this issue, we propose two novel algorithms based on truncation and mean of medians. These algorithms achieve an almost optimal regret bound of $\widetilde{O}(dT^{\frac{1}{1+\epsilon}})$, where $d$ is the dimension of contextual information and $T$ is the time horizon. Our truncation-based algorithm supports online learning, distinguishing it from existing truncation-based approaches. Additionally, our mean-of-medians-based algorithm requires only $O(\log T)$ rewards and one estimator per epoch, making it more practical. Moreover, our algorithms improve the regret bounds by a logarithmic factor compared to existing algorithms when $\epsilon=1$. Numerical experimental results confirm the merits of our algorithms.

----

## [3105] Multinomial Logistic Regression: Asymptotic Normality on Null Covariates in High-Dimensions

**Authors**: *Kai Tan, Pierre C. Bellec*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e0ac27bf3327c9cb99cc5f548db4f73a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e0ac27bf3327c9cb99cc5f548db4f73a-Abstract-Conference.html)

**Abstract**:

This paper investigates the asymptotic distribution of the maximum-likelihood estimate (MLE) in multinomial logistic models in the high-dimensional regime where dimension and sample size are of the same order. While classical large-sample theory provides asymptotic normality of the MLE under certain conditions, such classical results are expected to fail in high-dimensions as documented for the binary logistic case in the seminal work of Sur and Cand√®s [2019]. We address this issue in classification problems with 3 or more classes, by developing asymptotic normality and asymptotic chi-square results for the multinomial logistic MLE (also known as cross-entropy minimizer) on null covariates. Our theory leads to a new methodology to test the significance of a given feature. Extensive simulation studies on synthetic data corroborate these asymptotic results and confirm the validity of proposed p-values for testing the significance of a given feature.

----

## [3106] Why think step by step? Reasoning emerges from the locality of experience

**Authors**: *Ben Prystawski, Michael Li, Noah D. Goodman*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e0af79ad53a336b4c4b4f7e2a68eb609-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e0af79ad53a336b4c4b4f7e2a68eb609-Abstract-Conference.html)

**Abstract**:

Humans have a powerful and mysterious capacity to reason. Working through a set of mental steps enables us to make inferences we would not be capable of making directly even though we get no additional data from the world. Similarly, when large language models generate intermediate steps (a chain of thought) before answering a question, they often produce better answers than they would directly. We investigate why and how chain-of-thought reasoning is useful in language models, testing the hypothesis that reasoning is effective when training data consists of overlapping local clusters of variables that influence each other strongly. These training conditions enable the chaining of accurate local inferences to estimate relationships between variables that were not seen together in training. We prove that there will exist a "reasoning gap", where reasoning through intermediate variables reduces bias, for the simple case of an autoregressive density estimator trained on local samples from a chain-structured probabilistic model. We then test our hypothesis experimentally in more complex models, training an autoregressive language model on samples from Bayes nets but only including a subset of variables in each sample. We test language modelsâ€™ ability to match conditional probabilities with and without intermediate reasoning steps, finding that intermediate steps are only helpful when the training data is locally structured with respect to dependencies between variables. The combination of locally structured observations and reasoning is much more data-efficient than training on all variables. Our results illustrate how the effectiveness of reasoning step by step is rooted in the local statistical structure of the training data.

----

## [3107] Analyzing Generalization of Neural Networks through Loss Path Kernels

**Authors**: *Yilan Chen, Wei Huang, Hao Wang, Charlotte Loh, Akash Srivastava, Lam M. Nguyen, Lily Weng*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e0b6f389739496e363a89155c9448a8a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e0b6f389739496e363a89155c9448a8a-Abstract-Conference.html)

**Abstract**:

Deep neural networks have been increasingly used in real-world applications, making it critical to ensure their ability to adapt to new, unseen data. In this paper, we study the generalization capability of neural networks trained with (stochastic) gradient flow. We establish a new connection between the loss dynamics of gradient flow and general kernel machines by proposing a new kernel, called loss path kernel. This kernel measures the similarity between two data points by evaluating the agreement between loss gradients along the path determined by the gradient flow. Based on this connection, we derive a new generalization upper bound that applies to general neural network architectures. This new bound is tight and strongly correlated with the true generalization error. We apply our results to guide the design of neural architecture search (NAS) and demonstrate favorable performance compared with state-of-the-art NAS algorithms through numerical experiments.

----

## [3108] Operation-Level Early Stopping for Robustifying Differentiable NAS

**Authors**: *Shen Jiang, Zipeng Ji, Guanghui Zhu, Chunfeng Yuan, Yihua Huang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e0bc6dbcbcc957b2aeadb20c39ba7f05-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e0bc6dbcbcc957b2aeadb20c39ba7f05-Abstract-Conference.html)

**Abstract**:

Differentiable NAS (DARTS) is a simple and efficient neural architecture search method that has been extensively adopted in various machine learning tasks.% Nevertheless, DARTS still encounters several robustness issues, mainly the domination of skip connections.% The resulting architectures are full of parametric-free operations, leading to performance collapse.% Existing methods suggest that the skip connection has additional advantages in optimization compared to other parametric operations and propose to alleviate the domination of skip connections by eliminating these additional advantages.% In this paper, we analyze this issue from a simple and straightforward perspective and propose that the domination of skip connections results from parametric operations overfitting the training data while architecture parameters are trained on the validation data, leading to undesired behaviors.% Based on this observation, we propose the operation-level early stopping (OLES) method to overcome this issue and robustify DARTS without introducing any computation overhead.% Extensive experimental results can verify our hypothesis and the effectiveness of OLES.

----

## [3109] Training on Foveated Images Improves Robustness to Adversarial Attacks

**Authors**: *Muhammad Shah, Aqsa Kashaf, Bhiksha Raj*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e0c256700465c158de71081b4cf5e8c3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e0c256700465c158de71081b4cf5e8c3-Abstract-Conference.html)

**Abstract**:

Deep neural networks (DNNs) have been shown to be vulnerable to adversarial attacks-- subtle,  perceptually indistinguishable perturbations of inputs that change the response of the model. In the context of vision, we hypothesize that an important contributor to the robustness of human visual perception is constant exposure to low-fidelity visual stimuli in our peripheral vision. To investigate this hypothesis, we develop RBlur, an image transform that simulates the loss in fidelity of peripheral vision by blurring the image and reducing its color saturation based on the distance from a given fixation point. We show that compared to DNNs trained on the original images, DNNs trained on images transformed by RBlur are substantially more robust to adversarial attacks, as well as other, non-adversarial, corruptions, achieving up to 25% higher accuracy on perturbed data.

----

## [3110] Label Poisoning is All You Need

**Authors**: *Rishi D. Jha, Jonathan Hayase, Sewoong Oh*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e0c9b65fb3e41aaa86576df3ec33ad2e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e0c9b65fb3e41aaa86576df3ec33ad2e-Abstract-Conference.html)

**Abstract**:

In a backdoor attack, an adversary injects corrupted data into a model's training dataset in order to gain control over its predictions on images with a specific attacker-defined trigger. A typical corrupted training example requires altering both the image, by applying the trigger, and the label. Models trained on clean images, therefore, were considered safe from backdoor attacks. However, in some common machine learning scenarios, the training labels are provided by potentially malicious third-parties. This includes crowd-sourced annotation and knowledge distillation. We, hence, investigate a fundamental question: can we launch a successful backdoor attack by only corrupting labels? We introduce a novel approach to design label-only backdoor attacks, which we call FLIP, and demonstrate its strengths on three datasets (CIFAR-10, CIFAR-100, and Tiny-ImageNet) and four architectures (ResNet-32, ResNet-18, VGG-19, and Vision Transformer). With only 2% of CIFAR-10 labels corrupted, FLIP achieves a near-perfect attack success rate of 99.4% while suffering only a 1.8% drop in the clean test accuracy. Our approach builds upon the recent advances in trajectory matching, originally introduced for dataset distillation.

----

## [3111] Learning Trajectories are Generalization Indicators

**Authors**: *Jingwen Fu, Zhizheng Zhang, Dacheng Yin, Yan Lu, Nanning Zheng*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e0da54d3dbc0107692da952358965f5f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e0da54d3dbc0107692da952358965f5f-Abstract-Conference.html)

**Abstract**:

This paper explores the connection between learning trajectories of Deep Neural Networks (DNNs) and their generalization capabilities when optimized using (stochastic) gradient descent algorithms. Instead of concentrating solely on the generalization error of the DNN post-training, we present a novel perspective for analyzing generalization error by investigating the contribution of each update step to the change in generalization error. This perspective enable a more direct comprehension of how the learning trajectory influences generalization error. Building upon this analysis, we propose a new generalization bound that incorporates more extensive trajectory information.Our proposed generalization bound depends on the complexity of learning trajectory and the ratio between the bias and diversity of training set. Experimental observations reveal that our method effectively captures the generalization error throughout the training process. Furthermore, our approach can also track changes in generalization error when adjustments are made to learning rates and label noise levels. These results demonstrate that learning trajectory information is a valuable indicator of a model's generalization capabilities.

----

## [3112] CoDet: Co-occurrence Guided Region-Word Alignment for Open-Vocabulary Object Detection

**Authors**: *Chuofan Ma, Yi Jiang, Xin Wen, Zehuan Yuan, Xiaojuan Qi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e10a6a906ef323efaf708f76cf3c1d1e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e10a6a906ef323efaf708f76cf3c1d1e-Abstract-Conference.html)

**Abstract**:

Deriving reliable region-word alignment from image-text pairs is critical to learnobject-level vision-language representations for open-vocabulary object detection.Existing methods typically rely on pre-trained or self-trained vision-languagemodels for alignment, which are prone to limitations in localization accuracy orgeneralization capabilities. In this paper, we propose CoDet, a novel approachthat overcomes the reliance on pre-aligned vision-language space by reformulatingregion-word alignment as a co-occurring object discovery problem. Intuitively, bygrouping images that mention a shared concept in their captions, objects corresponding to the shared concept shall exhibit high co-occurrence among the group.CoDet then leverages visual similarities to discover the co-occurring objects andalign them with the shared concept. Extensive experiments demonstrate that CoDethas superior performances and compelling scalability in open-vocabulary detection,e.g., by scaling up the visual backbone, CoDet achieves 37.0 $AP^m_{novel}$ and 44.7 $AP^m_{all}$ on OV-LVIS, surpassing the previous SoTA by 4.2 $AP^m_{novel}$ and 9.8 $AP^m_{all}$. Code is available at https://github.com/CVMI-Lab/CoDet.

----

## [3113] Rewarded soups: towards Pareto-optimal alignment by interpolating weights fine-tuned on diverse rewards

**Authors**: *Alexandre Ramé, Guillaume Couairon, Corentin Dancette, Jean-Baptiste Gaya, Mustafa Shukor, Laure Soulier, Matthieu Cord*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e12a3b98b67e8395f639fde4c2b03168-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e12a3b98b67e8395f639fde4c2b03168-Abstract-Conference.html)

**Abstract**:

Foundation models are first pre-trained on vast unsupervised datasets and then fine-tuned on labeled data. Reinforcement learning, notably from human feedback (RLHF), can further align the network with the intended usage. Yet the imperfections in the proxy reward may hinder the training and lead to suboptimal results; the diversity of objectives in real-world tasks and human opinions exacerbate the issue. This paper proposes embracing the heterogeneity of diverse rewards by following a multi-policy strategy. Rather than focusing on a single a priori reward, we aim for Pareto-optimal generalization across the entire space of preferences. To this end, we propose rewarded soup, first specializing multiple networks independently (one for each proxy reward) and then interpolating their weights linearly. This succeeds empirically because we show that the weights remain linearly connected when fine-tuned on diverse rewards from a shared pre-trained initialization. We demonstrate the effectiveness of our approach for text-to-text (summarization, Q&A, helpful assistant, review), text-image (image captioning, text-to-image generation, visual grounding), and control (locomotion) tasks. We hope to enhance the alignment of deep models, and how they interact with the world in all its diversity.

----

## [3114] Optimal Block-wise Asymmetric Graph Construction for Graph-based Semi-supervised Learning

**Authors**: *Zixing Song, Yifei Zhang, Irwin King*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e142fd2b70f10db2543c64bca1417de8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e142fd2b70f10db2543c64bca1417de8-Abstract-Conference.html)

**Abstract**:

Graph-based semi-supervised learning (GSSL) serves as a powerful tool to model the underlying manifold structures of samples in high-dimensional spaces. It involves two phases: constructing an affinity graph from available data and inferring labels for unlabeled nodes on this graph. While numerous algorithms have been developed for label inference, the crucial graph construction phase has received comparatively less attention, despite its significant influence on the subsequent phase. In this paper, we present an optimal asymmetric graph structure for the label inference phase with theoretical motivations. Unlike existing graph construction methods, we differentiate the distinct roles that labeled nodes and unlabeled nodes could play. Accordingly, we design an efficient block-wise graph learning algorithm with a global convergence guarantee. Other benefits induced by our method, such as enhanced robustness to noisy node features, are explored as well. Finally, we perform extensive experiments on synthetic and real-world datasets to demonstrate its superiority to the state-of-the-art graph construction methods in GSSL.

----

## [3115] On the Complexity of Differentially Private Best-Arm Identification with Fixed Confidence

**Authors**: *Achraf Azize, Marc Jourdan, Aymen Al Marjani, Debabrota Basu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e14de1a0ebc31d9b989f5f5528c125bb-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e14de1a0ebc31d9b989f5f5528c125bb-Abstract-Conference.html)

**Abstract**:

Best Arm Identification (BAI) problems are progressively used for data-sensitive applications, such as designing adaptive clinical trials, tuning hyper-parameters, and conducting user studies to name a few. Motivated by the data privacy concerns invoked by these applications, we study the problem of BAI with fixed confidence under $\epsilon$-global Differential Privacy (DP). First, to quantify the cost of privacy, we derive a lower bound on the sample complexity of any $\delta$-correct BAI algorithm satisfying $\epsilon$-global DP. Our lower bound suggests the existence of two privacy regimes depending on the privacy budget $\epsilon$. In the high-privacy regime (small $\epsilon$), the hardness depends on a coupled effect of privacy and a novel information-theoretic quantity, called the Total Variation Characteristic Time. In the low-privacy regime (large $\epsilon$), the sample complexity lower bound reduces to the classical non-private lower bound. Second, we propose AdaP-TT, an $\epsilon$-global DP variant of the Top Two algorithm. AdaP-TT runs in *arm-dependent adaptive episodes* and adds *Laplace noise* to ensure a good privacy-utility trade-off. We derive an asymptotic upper bound on the sample complexity of AdaP-TT that matches with the lower bound up to multiplicative constants in the high-privacy regime. Finally, we provide an experimental analysis of AdaP-TT that validates our theoretical results.

----

## [3116] COCO-Counterfactuals: Automatically Constructed Counterfactual Examples for Image-Text Pairs

**Authors**: *Tiep Le, Vasudev Lal, Phillip Howard*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e14e4cb8266184ceb234973dfe07faed-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/e14e4cb8266184ceb234973dfe07faed-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Counterfactual examples have proven to be valuable in the field of natural language processing (NLP) for both evaluating and improving the robustness of language models to spurious correlations in datasets. Despite their demonstrated utility for NLP, multimodal counterfactual examples have been relatively unexplored due to the difficulty of creating paired image-text data with minimal counterfactual changes. To address this challenge, we introduce a scalable framework for automatic generation of counterfactual examples using text-to-image diffusion models. We use our framework to create COCO-Counterfactuals, a multimodal counterfactual dataset of paired image and text captions based on the MS-COCO dataset. We validate the quality of COCO-Counterfactuals through human evaluations and show that existing multimodal models are challenged by our counterfactual image-text pairs. Additionally, we demonstrate the usefulness of COCO-Counterfactuals for improving out-of-domain generalization of multimodal vision-language models via training data augmentation. We make our code and the COCO-Counterfactuals dataset publicly available.

----

## [3117] BasisFormer: Attention-based Time Series Forecasting with Learnable and Interpretable Basis

**Authors**: *Zelin Ni, Hang Yu, Shizhan Liu, Jianguo Li, Weiyao Lin*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e150e6d0a1e5214740c39c6e4503ba7a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e150e6d0a1e5214740c39c6e4503ba7a-Abstract-Conference.html)

**Abstract**:

Bases have become an integral part of modern deep learning-based models for time series forecasting due to their ability to act as feature extractors or future references. To be effective, a basis must be tailored to the specific set of time series data and exhibit distinct correlation with each time series within the set. However, current state-of-the-art methods are limited in their ability to satisfy both of these requirements simultaneously. To address this challenge, we propose BasisFormer, an end-to-end time series forecasting architecture that leverages learnable and interpretable bases. This architecture comprises three components: First, we acquire bases through adaptive self-supervised learning, which treats the historical and future sections of the time series as two distinct views and employs contrastive learning. Next, we design a Coef module that calculates the similarity coefficients between the time series and bases in the historical view via bidirectional cross-attention. Finally, we present a Forecast module that selects and consolidates the bases in the future view based on the similarity coefficients, resulting in accurate future predictions. Through extensive experiments on six datasets, we demonstrate that BasisFormer outperforms previous state-of-the-art methods by 11.04% and 15.78% respectively for univariate and multivariate forecasting tasks. Code isavailable at: https://github.com/nzl5116190/Basisformer.

----

## [3118] Towards Foundation Models for Scientific Machine Learning: Characterizing Scaling and Transfer Behavior

**Authors**: *Shashank Subramanian, Peter Harrington, Kurt Keutzer, Wahid Bhimji, Dmitriy Morozov, Michael W. Mahoney, Amir Gholami*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e15790966a4a9d85d688635c88ee6d8a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e15790966a4a9d85d688635c88ee6d8a-Abstract-Conference.html)

**Abstract**:

Pre-trained machine learning (ML) models have shown great performance for awide range of applications, in particular in natural language processing (NLP)and computer vision (CV). Here, we study how pre-training could be used forscientific machine learning (SciML) applications, specifically in the context oftransfer learning. We study the transfer behavior of these models as (i) the pretrainedmodel size is scaled, (ii) the downstream training dataset size is scaled,(iii) the physics parameters are systematically pushed out of distribution, and (iv)how a single model pre-trained on a mixture of different physics problems canbe adapted to various downstream applications. We find that—when fine-tunedappropriately—transfer learning can help reach desired accuracy levels with ordersof magnitude fewer downstream examples (across different tasks that can even beout-of-distribution) than training from scratch, with consistent behaviour across awide range of downstream examples. We also find that fine-tuning these modelsyields more performance gains as model size increases, compared to training fromscratch on new downstream tasks. These results hold for a broad range of PDElearning tasks. All in all, our results demonstrate the potential of the “pre-train andfine-tune” paradigm for SciML problems, demonstrating a path towards buildingSciML foundation models. Our code is available as open-source.

----

## [3119] Hyperbolic Space with Hierarchical Margin Boosts Fine-Grained Learning from Coarse Labels

**Authors**: *Shu-Lin Xu, Yifan Sun, Faen Zhang, Anqi Xu, Xiu-Shen Wei, Yi Yang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e17e11960843febbc2dd22d3c7d79144-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e17e11960843febbc2dd22d3c7d79144-Abstract-Conference.html)

**Abstract**:

Learning fine-grained embeddings from coarse labels is a challenging task due to limited label granularity supervision, i.e., lacking the detailed distinctions required for fine-grained tasks. The task becomes even more demanding when attempting few-shot fine-grained recognition, which holds practical significance in various applications. To address these challenges, we propose a novel method that embeds visual embeddings into a hyperbolic space and enhances their discriminative ability with a hierarchical cosine margins manner. Specifically, the hyperbolic space offers distinct advantages, including the ability to capture hierarchical relationships and increased expressive power, which favors modeling fine-grained objects. Based on the hyperbolic space, we further enforce relatively large/small similarity margins between coarse/fine classes, respectively, yielding the so-called hierarchical cosine margins manner. While enforcing similarity margins in the regular Euclidean space has become popular for deep embedding learning, applying it to the hyperbolic space is non-trivial and validating the benefit for coarse-to-fine generalization is valuable. Extensive experiments conducted on five benchmark datasets showcase the effectiveness of our proposed method, yielding state-of-the-art results surpassing competing methods.

----

## [3120] PromptIR: Prompting for All-in-One Image Restoration

**Authors**: *Vaishnav Potlapalli, Syed Waqas Zamir, Salman H. Khan, Fahad Shahbaz Khan*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e187897ed7780a579a0d76fd4a35d107-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e187897ed7780a579a0d76fd4a35d107-Abstract-Conference.html)

**Abstract**:

Image restoration involves recovering a high-quality clean image from its degraded version. Deep learning-based methods have significantly improved image restoration performance, however, they have limited generalization ability to different degradation types and levels. This restricts their real-world application since it requires training individual models for each specific degradation and knowing the input degradation type to apply the relevant model. We present a prompt-based learning approach, PromptIR, for All-In-One image restoration that can effectively restore images from various types and levels of degradation. In particular, our method uses prompts to encode degradation-specific information, which is then used to dynamically guide the restoration network. This allows our method to generalize to different degradation types and levels, while still achieving state-of-the-art results on image denoising, deraining, and dehazing.  Overall, PromptIR offers a generic and efficient plugin module with few lightweight prompts that can be used to restore images of various types and levels of degradation with no prior information on the corruptions present in the image. Our code and pre-trained models are available here: https://github.com/va1shn9v/PromptIR

----

## [3121] Creating a Public Repository for Joining Private Data

**Authors**: *James Cook, Milind Shyani, Nina Mishra*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e19560e93418dd0d6498bd3b2de856cd-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e19560e93418dd0d6498bd3b2de856cd-Abstract-Conference.html)

**Abstract**:

How can one publish a dataset with sensitive attributes in a way that both preserves privacy and enables joins with other datasets on those same sensitive attributes? This problem arises in many contexts, e.g., a hospital and an airline may want to jointly determine whether people who take long-haul flights are more likely to catch respiratory infections. If they join their data by a common keyed user identifier such as email address, they can determine the answer, though it breaks privacy.  This paper shows how the hospital can generate a private sketch and how the airline can privately join with the hospital's sketch by email address. The proposed solution satisfies pure differential privacy and gives approximate answers to linear queries and optimization problems over those joins. Whereas prior work such as secure function evaluation requires sender/receiver interaction, a distinguishing characteristic of the proposed approach is that it is non-interactive. Consequently, the sketch can be published to a repository for any organization to join with, facilitating data discovery. The accuracy of the method is demonstrated through both theoretical analysis and extensive empirical evidence.

----

## [3122] What Truly Matters in Trajectory Prediction for Autonomous Driving?

**Authors**: *Tran Phong, Haoran Wu, Cunjun Yu, Panpan Cai, Sifa Zheng, David Hsu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e197fe307eb3467035f892dc100d570a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e197fe307eb3467035f892dc100d570a-Abstract-Conference.html)

**Abstract**:

Trajectory prediction plays a vital role in the performance of autonomous driving systems, and prediction accuracy, such as average displacement error (ADE) or final displacement error (FDE), is widely used as a performance metric. However, a significant disparity exists between the accuracy of predictors on fixed datasets and driving performance when the predictors are used downstream for vehicle control, because of a dynamics gap. In the real world, the prediction algorithm influences the behavior of the ego vehicle, which, in turn, influences the behaviors of other vehicles nearby. This interaction results in predictor-specific dynamics that directly impacts prediction results. In fixed datasets, since other vehicles' responses are predetermined, this interaction effect is lost, leading to a significant dynamics gap. This paper studies the overlooked significance of this dynamics gap. We also examine several other factors contributing to the disparity between prediction performance and driving performance. The findings highlight the trade-off between the predictor's computational efficiency and prediction accuracy in determining real-world driving performance. In summary,  an interactive, task-driven evaluation protocol for trajectory prediction is crucial to capture its effectiveness for autonomous driving. Source code along with experimental settings is available online (https://whatmatters23.github.io/).

----

## [3123] AUDIT: Audio Editing by Following Instructions with Latent Diffusion Models

**Authors**: *Yuancheng Wang, Zeqian Ju, Xu Tan, Lei He, Zhizheng Wu, Jiang Bian, Sheng Zhao*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e1b619a9e241606a23eb21767f16cf81-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e1b619a9e241606a23eb21767f16cf81-Abstract-Conference.html)

**Abstract**:

Audio editing is applicable for various purposes, such as adding background sound effects, replacing a musical instrument, and repairing damaged audio. Recently, some diffusion-based methods achieved zero-shot audio editing by using a diffusion and denoising process conditioned on the text description of the output audio. However, these methods still have some problems: 1) they have not been trained on editing tasks and cannot ensure good editing effects; 2) they can erroneously modify audio segments that do not require editing; 3) they need a complete description of the output audio, which is not always available or necessary in practical scenarios. In this work, we propose AUDIT, an instruction-guided audio editing model based on latent diffusion models. Specifically, \textbf{AUDIT} has three main design features: 1) we construct triplet training data (instruction, input audio, output audio) for different audio editing tasks and train a diffusion model using instruction and input (to be edited) audio as conditions and generating output (edited) audio; 2) it can automatically learn to only modify segments that need to be edited by comparing the difference between the input and output audio; 3) it only needs edit instructions instead of full target audio descriptions as text input. AUDIT achieves state-of-the-art results in both objective and subjective metrics for several audio editing tasks (e.g., adding, dropping, replacement, inpainting, super-resolution). Demo samples are available at https://audit-demopage.github.io/.

----

## [3124] An Optimization-based Approach To Node Role Discovery in Networks: Approximating Equitable Partitions

**Authors**: *Michael Scholkemper, Michael T. Schaub*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e1c73e9595126794186536cfbbed012f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e1c73e9595126794186536cfbbed012f-Abstract-Conference.html)

**Abstract**:

Similar to community detection, partitioning the nodes of a complex network according to their structural roles aims to identify fundamental building blocks of a network, which can be used, e.g., to find simplified descriptions of the network connectivity, to derive reduced order models for dynamical processes unfolding on processes, or as ingredients for various network analysis and graph mining tasks. In this work, we offer a fresh look on the problem of role extraction and its differences to community detection and present a definition of node roles and two associated optimization problems (cost functions) grounded in ideas related to graph-isomorphism tests, the Weisfeiler-Leman algorithm and equitable partitions. We present theoretical guarantees and validate our approach via a novel “role-infused partition benchmark”, a network model from which we can sample networks in which nodes are endowed with different roles in a stochastic way.

----

## [3125] Robust Model Reasoning and Fitting via Dual Sparsity Pursuit

**Authors**: *Xingyu Jiang, Jiayi Ma*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e1de63ec74f40d3234c4e053f3528e18-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e1de63ec74f40d3234c4e053f3528e18-Abstract-Conference.html)

**Abstract**:

In this paper, we contribute to solving a threefold problem: outlier rejection, true model reasoning and parameter estimation with a unified optimization modeling. To this end, we first pose this task as a sparse subspace recovering problem, to search a maximum of independent bases under an over-embedded data space. Then we convert the objective into a continuous optimization paradigm that estimates sparse solutions for both bases and errors. Wherein a fast and robust solver is proposed to accurately estimate the sparse subspace parameters and error entries, which is implemented by a proximal approximation method under the alternating optimization framework with the ``optimal'' sub-gradient descent. Extensive experiments regarding known and unknown model fitting on synthetic and challenging real datasets have demonstrated the superiority of our method against the state-of-the-art. We also apply our method to multi-class multi-model fitting and loop closure detection, and achieve promising results both in accuracy and efficiency. Code is released at: https://github.com/StaRainJ/DSP.

----

## [3126] Evaluating the Robustness of Interpretability Methods through Explanation Invariance and Equivariance

**Authors**: *Jonathan Crabbé, Mihaela van der Schaar*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e1f418450107c4a0ddc16d008d131573-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e1f418450107c4a0ddc16d008d131573-Abstract-Conference.html)

**Abstract**:

Interpretability methods are valuable only if their explanations faithfully describe the explained model. In this work, we consider neural networks whose predictions are invariant under a specific symmetry group. This includes popular architectures, ranging from convolutional to graph neural networks. Any explanation that faithfully explains this type of model needs to be in agreement with this invariance property. We formalize this intuition through the notion of explanation invariance and equivariance by leveraging the formalism from geometric deep learning. Through this rigorous formalism, we derive (1) two metrics to measure the robustness of any interpretability method with respect to the model symmetry group; (2) theoretical robustness guarantees for some popular interpretability methods and (3) a systematic approach to increase the invariance of any interpretability method with respect to a symmetry group. By empirically measuring our metrics for explanations of models associated with various modalities and symmetry groups, we derive a set of 5 guidelines to allow users and developers of interpretability methods to produce robust explanations.

----

## [3127] Back-Modality: Leveraging Modal Transformation for Data Augmentation

**Authors**: *Zhi Li, Yifan Liu, Yin Zhang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e20a65c7308b7b94ed1178eebc45bf76-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e20a65c7308b7b94ed1178eebc45bf76-Abstract-Conference.html)

**Abstract**:

We introduce Back-Modality, a novel data augmentation schema predicated on modal transformation. Data from an initial modality undergoes transformation to an intermediate modality, followed by a reverse transformation. This framework serves dual roles. On one hand, it operates as a general data augmentation strategy. On the other hand, it allows for other augmentation techniques, suitable for the intermediate modality, to enhance the initial modality. For instance, data augmentation methods applicable to pure text can be employed to augment images, thereby facilitating the cross-modality of data augmentation techniques. To validate the viability and efficacy of our framework, we proffer three instantiations of Back-Modality: back-captioning, back-imagination, and back-speech. Comprehensive evaluations across tasks such as image classification, sentiment classification, and textual entailment demonstrate that our methods consistently enhance performance under data-scarce circumstances.

----

## [3128] Gradient-Based Feature Learning under Structured Data

**Authors**: *Alireza Mousavi Hosseini, Denny Wu, Taiji Suzuki, Murat A. Erdogdu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e21955c93dede886af1d0d362c756757-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e21955c93dede886af1d0d362c756757-Abstract-Conference.html)

**Abstract**:

Recent works have demonstrated that the sample complexity of gradient-based learning of single index models, i.e. functions that depend on a 1-dimensional projection of the input data, is governed by their information exponent. However, these results are only concerned with isotropic data, while in practice the input often contains additional structure which can implicitly guide the algorithm. In this work, we investigate the effect of a spiked covariance structure and reveal several interesting phenomena. First, we show that in the anisotropic setting, the commonly used spherical gradient dynamics may fail to recover the true direction, even when the spike is perfectly aligned with the target direction. Next, we show that appropriate weight normalization that is reminiscent of batch normalization can alleviate this issue. Further, by exploiting the alignment between the (spiked) input covariance and the target, we obtain improved sample complexity compared to the isotropic case. In particular, under the spiked model with a suitably large spike, the sample complexity of gradient-based training can be made independent of the information exponent while also outperforming lower bounds for rotationally invariant kernel methods.

----

## [3129] Does Invariant Graph Learning via Environment Augmentation Learn Invariance?

**Authors**: *Yongqiang Chen, Yatao Bian, Kaiwen Zhou, Binghui Xie, Bo Han, James Cheng*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e21a7b668ce3ea2c9c964c52d1c9f161-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e21a7b668ce3ea2c9c964c52d1c9f161-Abstract-Conference.html)

**Abstract**:

Invariant graph representation learning aims to learn the invariance among data from different environments for out-of-distribution generalization on graphs. As the graph environment partitions are usually expensive to obtain, augmenting the environment information has become the de facto approach. However, the usefulness of the augmented environment information has never been verified. In this work, we find that it is fundamentally impossible to learn invariant graph representations via environment augmentation without additional assumptions. Therefore, we develop a set of minimal assumptions, including variation sufficiency and variation consistency, for feasible invariant graph learning. We then propose a new framework Graph invAriant Learning Assistant (GALA). GALA incorporates an assistant model that needs to be sensitive to graph environment changes or distribution shifts. The correctness of the proxy predictions by the assistant model hence can differentiate the variations in spurious subgraphs. We show that extracting the maximally invariant subgraph to the proxy predictions provably identifies the underlying invariant subgraph for successful OOD generalization under the established minimal assumptions. Extensive experiments on datasets including DrugOOD with various graph distribution shifts confirm the effectiveness of GALA.

----

## [3130] Mitigating Test-Time Bias for Fair Image Retrieval

**Authors**: *Fanjie Kong, Shuai Yuan, Weituo Hao, Ricardo Henao*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e24570da4fa1c005b189104250993aee-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e24570da4fa1c005b189104250993aee-Abstract-Conference.html)

**Abstract**:

We address the challenge of generating fair and unbiased image retrieval results given neutral textual queries (with no explicit gender or race connotations), while maintaining the utility (performance) of the underlying vision-language (VL) model. Previous methods aim to disentangle learned representations of images and text queries from gender and racial characteristics. However, we show these are inadequate at alleviating bias for the desired equal representation result, as there usually exists test-time bias in the target retrieval set. So motivated, we introduce a straightforward technique, Post-hoc Bias Mitigation (PBM), that post-processes the outputs from the pre-trained vision-language model. We evaluate our algorithm on real-world image search datasets, Occupation 1 and 2, as well as two large-scale image-text datasets, MS-COCO and Flickr30k. Our approach achieves the lowest bias, compared with various existing bias-mitigation methods, in text-based image retrieval result while maintaining satisfactory retrieval performance. The source code is publicly available at \url{https://github.com/timqqt/FairTextbasedImageRetrieval}.

----

## [3131] Lower Bounds on Adaptive Sensing for Matrix Recovery

**Authors**: *Praneeth Kacham, David P. Woodruff*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e258bb98cc032ab6ae9053db453431f7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e258bb98cc032ab6ae9053db453431f7-Abstract-Conference.html)

**Abstract**:

We study lower bounds on adaptive sensing algorithms for recovering low rank matrices using linear measurements. Given an $n \times n$ matrix $A$, a general linear measurement $S(A)$, for an $n \times n$ matrix $S$, is just the inner product of $S$ and $A$, each treated as $n^2$-dimensional vectors. By performing as few linear measurements as possible on a rank-$r$ matrix $A$, we hope to construct a matrix $\hat{A}$ that satisfies $|A - \hat{A}|\_F^2 \le c |A|\_F^2$, for a small constant $c$. Here $|A|\_F$ denotes the Frobenius norm $(\sum_{i,j} A_{i,j}^2)^{1/2}$. It is commonly assumed that when measuring $A$ with $S$, the response is corrupted with an independent Gaussian random variable of mean $0$ and variance $\sigma^2$. CandÃ¨s and Plan (IEEE Trans. Inform. Theory 2011) study non-adaptive algorithms for low rank matrix recovery using random linear measurements. They use the restricted isometry property (RIP) of Random Gaussian Matrices to give tractable algorithms to estimate $A$ from the measurements.At the edge of the noise level where recovery is information-theoretically feasible, it is known that their non-adaptive algorithms need to perform $\Omega(n^2)$ measurements, which amounts to reading the entire matrix. An important question is whether adaptivity helps in decreasing the overall number of measurements. While for the related problem of sparse recovery, adaptive algorithms have been extensively studied, as far as we are aware adaptive algorithms and lower bounds on them seem largely unexplored for matrix recovery. We show that any adaptive algorithm that uses $k$ linear measurements in each round and outputs an approximation as in (1) with probability $\ge 9/10$ must run for $t = \Omega(\log(n^2/k)/\log\log n)$ rounds. Our lower bound shows that any adaptive algorithm which uses $n^{2-\beta}$ ($\beta > 0$ is arbitrary constant) linear measurements in each round must run for $\Omega(\log n/\log\log n)$ rounds. Our techniques also readily extend to obtain lower bounds on adaptive algorithms for tensor recovery. Our hard distribution also allows us to give a measurement-vs-rounds trade-off for many sensing problems in numerical linear algebra, such as spectral norm low rank approximation, Frobenius norm low rank approximation, singular vector approximation, and more.

----

## [3132] Transient Neural Radiance Fields for Lidar View Synthesis and 3D Reconstruction

**Authors**: *Anagh Malik, Parsa Mirdehghan, Sotiris Nousias, Kyros Kutulakos, David B. Lindell*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e261e92e1cfb820da930ad8c38d0aead-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e261e92e1cfb820da930ad8c38d0aead-Abstract-Conference.html)

**Abstract**:

Neural radiance fields (NeRFs) have become a ubiquitous tool for modeling scene appearance and geometry from multiview imagery. Recent work has also begun to explore how to use additional supervision from lidar or depth sensor measurements in the NeRF framework. However, previous lidar-supervised NeRFs focus on rendering conventional camera imagery and use lidar-derived point cloud data as auxiliary supervision; thus, they fail to incorporate the underlying image formation model of the lidar. Here, we propose a novel method for rendering transient NeRFs that take as input the raw, time-resolved photon count histograms measured by a single-photon lidar system, and we seek to render such histograms from novel views. Different from conventional NeRFs, the approach relies on a time-resolved version of the volume rendering equation to render the lidar measurements and capture transient light transport phenomena at picosecond timescales. We evaluate our method on a first-of-its-kind dataset of simulated and captured transient multiview scans from a prototype single-photon lidar. Overall, our work brings NeRFs to a new dimension of imaging at transient timescales, newly enabling rendering of transient imagery from novel views. Additionally, we show that our approach recovers improved geometry and conventional appearance compared to point cloud-based supervision when training on few input viewpoints. Transient NeRFs may be especially useful for applications which seek to simulate raw lidar measurements for downstream tasks in autonomous driving, robotics, and remote sensing.

----

## [3133] An Exploration-by-Optimization Approach to Best of Both Worlds in Linear Bandits

**Authors**: *Shinji Ito, Kei Takemura*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e262fc23ec7275230ee77c55d0cc9555-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e262fc23ec7275230ee77c55d0cc9555-Abstract-Conference.html)

**Abstract**:

In this paper, we consider how to construct best-of-both-worlds linear bandit algorithms that achieve nearly optimal performance for both stochastic and adversarial environments.  For this purpose, we show that a natural approach referred to as exploration by optimization [Lattimore and Szepesv√°ri, 2020] works well. Specifically, an algorithm constructed using this approach achieves $O(d \sqrt{ T \log{T}})$-regret in adversarial environments and $O(\frac{d^2 \log T}{\Delta_{\min}} )$-regret in stochastic environments.  Symbols $d$, $T$ and $\Delta_{\min}$ here represent the dimensionality of the action set, the time horizon, and the minimum sub-optimality gap, respectively.  We also show that this algorithm has even better theoretical guarantees for important special cases including the multi-armed bandit problem and multitask bandits.

----

## [3134] The expressive power of pooling in Graph Neural Networks

**Authors**: *Filippo Maria Bianchi, Veronica Lachi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e26f31de8b13ec569bf507e6ae2cd952-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e26f31de8b13ec569bf507e6ae2cd952-Abstract-Conference.html)

**Abstract**:

In Graph Neural Networks (GNNs), hierarchical pooling operators generate local summaries of the data by coarsening the graph structure and the vertex features. Considerable attention has been devoted to analyzing the expressive power of message-passing (MP) layers in GNNs, while a study on how graph pooling affects the expressiveness of a GNN is still lacking. Additionally, despite the recent advances in the design of pooling operators, there is not a principled criterion to compare them. In this work, we derive sufficient conditions for a pooling operator to fully preserve the expressive power of the MP layers before it. These conditions serve as a universal and theoretically-grounded criterion for choosing among existing pooling operators or designing new ones. Based on our theoretical findings, we analyze several existing pooling operators and identify those that fail to satisfy the expressiveness conditions. Finally, we introduce an experimental setup to verify empirically the expressive power of a GNN equipped with pooling layers, in terms of its capability to perform a graph isomorphism test.

----

## [3135] Cal-DETR: Calibrated Detection Transformer

**Authors**: *Muhammad Akhtar Munir, Salman H. Khan, Muhammad Haris Khan, Mohsen Ali, Fahad Shahbaz Khan*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e271e30de7a2e462ca1f85cefa816380-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e271e30de7a2e462ca1f85cefa816380-Abstract-Conference.html)

**Abstract**:

Albeit revealing impressive predictive performance for several computer vision tasks, deep neural networks (DNNs) are prone to making overconfident predictions. This limits the adoption and wider utilization of DNNs in many safety-critical applications. There have been recent efforts toward calibrating DNNs, however, almost all of them focus on the classification task. Surprisingly, very little attention has been devoted to calibrating modern DNN-based object detectors, especially detection transformers, which have recently demonstrated promising detection performance and are influential in many decision-making systems. In this work, we address the problem by proposing a mechanism for calibrated detection transformers (Cal-DETR), particularly for Deformable-DETR, UP-DETR, and DINO. We pursue the train-time calibration route and make the following contributions. First, we propose a simple yet effective approach for quantifying uncertainty in transformer-based object detectors. Second, we develop an uncertainty-guided logit modulation mechanism that leverages the uncertainty to modulate the class logits. Third, we develop a logit mixing approach that acts as a regularizer with detection-specific losses and is also complementary to the uncertainty-guided logit modulation technique to further improve the calibration performance. Lastly, we conduct extensive experiments across three in-domain and four out-domain scenarios. Results corroborate the effectiveness of Cal-DETR against the competing train-time methods in calibrating both in-domain and out-domain detections while maintaining or even improving the detection performance. Our codebase and pre-trained models can be accessed at \url{https://github.com/akhtarvision/cal-detr}.

----

## [3136] Trajectory Alignment: Understanding the Edge of Stability Phenomenon via Bifurcation Theory

**Authors**: *Minhak Song, Chulhee Yun*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e2a9256bd816ab9e082dfaa22f1f62a2-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e2a9256bd816ab9e082dfaa22f1f62a2-Abstract-Conference.html)

**Abstract**:

Cohen et al. (2021) empirically study the evolution of the largest eigenvalue of the loss Hessian, also known as sharpness, along the gradient descent (GD) trajectory and observe the Edge of Stability (EoS) phenomenon. The sharpness increases at the early phase of training (referred to as progressive sharpening), and eventually saturates close to the threshold of $2 / \text{(step size)}$. In this paper, we start by demonstrating through empirical studies that when the EoS phenomenon occurs, different GD trajectories (after a proper reparameterization) align on a specific bifurcation diagram independent of initialization. We then rigorously prove this trajectory alignment phenomenon for a two-layer fully-connected linear network and a single-neuron nonlinear network trained with a single data point. Our trajectory alignment analysis establishes both progressive sharpening and EoS phenomena, encompassing and extending recent findings in the literature.

----

## [3137] OBELICS: An Open Web-Scale Filtered Dataset of Interleaved Image-Text Documents

**Authors**: *Hugo Laurençon, Lucile Saulnier, Léo Tronchon, Stas Bekman, Amanpreet Singh, Anton Lozhkov, Thomas Wang, Siddharth Karamcheti, Alexander M. Rush, Douwe Kiela, Matthieu Cord, Victor Sanh*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e2cfb719f58585f779d0a4f9f07bd618-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/e2cfb719f58585f779d0a4f9f07bd618-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Large multimodal models trained on natural documents, which interleave images and text, outperform models trained on image-text pairs on various multimodal benchmarks. However, the datasets used to train these models have not been released, and the collection process has not been fully specified.  We introduce the OBELICS dataset, an open web-scale filtered dataset of interleaved image-text documents comprising 141 million web pages extracted from Common Crawl, 353 million associated images, and 115 billion text tokens. We describe the dataset creation process, present comprehensive filtering rules, and provide an analysis of the dataset's content. To show the viability of OBELICS, we train on the dataset vision and language models of 9 and 80 billion parameters, IDEFICS-9B and IDEFICS, and obtain competitive performance on different multimodal benchmarks. We release our dataset, models and code.

----

## [3138] ID and OOD Performance Are Sometimes Inversely Correlated on Real-world Datasets

**Authors**: *Damien Teney, Yong Lin, Seong Joon Oh, Ehsan Abbasnejad*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e304d374c85e385eb217ed4a025b6b63-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e304d374c85e385eb217ed4a025b6b63-Abstract-Conference.html)

**Abstract**:

Several studies have compared the in-distribution (ID) and out-of-distribution (OOD) performance of models in computer vision and NLP. They report a frequent positive correlation and some surprisingly never even observe an inverse correlation indicative of a necessary trade-off. The possibility of inverse patterns is important to determine whether ID performance can serve as a proxy for OOD generalization capabilities.This paper shows that inverse correlations between ID and OOD performance do happen with multiple real-world datasets, not only in artificial worst-case settings. We explain theoretically how these cases arise and how past studies missed them because of improper methodologies that examined a biased selection of models.Our observations lead to recommendations that contradict those found in much of the current literature.- High OOD performance sometimes requires trading off ID performance.- Focusing on ID performance alone may not lead to optimal OOD performance. It may produce diminishing (eventually negative) returns in OOD performance.- In these cases, studies on OOD generalization that use ID performance for model selection (a common recommended practice) will necessarily miss the best-performing models, making these studies blind to a whole range of phenomena.

----

## [3139] On Generalization Bounds for Projective Clustering

**Authors**: *Maria Sofia Bucarelli, Matilde Fjeldsø Larsen, Chris Schwiegelshohn, Mads Toftrup*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e30bf4765ae6b16a87fb4d7b0b3b3dec-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e30bf4765ae6b16a87fb4d7b0b3b3dec-Abstract-Conference.html)

**Abstract**:

Given a set of points, clustering consists of finding a partition of a point set into $k$ clusters such that the center to which a point is assigned is as close as possible. Most commonly, centers are points themselves, which leads to the famous $k$-median and $k$-means objectives. One may also choose centers to be $j$ dimensional subspaces, which gives rise to subspace clustering. In this paper, we consider learning bounds for these problems. That is, given a set of $n$ samples $P$ drawn independently from some unknown, but fixed distribution $\mathcal{D}$, how quickly does a solution computed on $P$ converge to the optimal clustering of $\mathcal{D}$?We give several near optimal results. In particular, 1. For center-based objectives, we show a convergence rate of $\tilde{O}\left(\sqrt{{k}/{n}}\right)$. This matches the known optimal bounds of [Fefferman, Mitter, and Narayanan, Journal of the Mathematical Society 2016] and [Bartlett, Linder, and Lugosi, IEEE Trans. Inf. Theory 1998] for $k$-means and extends it to other important objectives such as $k$-median. 2. For subspace clustering with $j$-dimensional subspaces, we show a convergence rate of $\tilde{O}\left(\sqrt{{(kj^2)}/{n}}\right)$. These are the first provable bounds for most of these problems. For the specific case of projective clustering, which generalizes $k$-means, we show a converge rate of $\Omega\left(\sqrt{{(kj)}/{n}}\right)$ is necessary, thereby proving that the bounds from [Fefferman, Mitter, and Narayanan, Journal of the Mathematical Society 2016] are essentially optimal.

----

## [3140] Emergence of Shape Bias in Convolutional Neural Networks through Activation Sparsity

**Authors**: *Tianqin Li, Ziqi Wen, Yangfan Li, Tai Sing Lee*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e31c16c7b3e0ccee5159ae5443154fac-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e31c16c7b3e0ccee5159ae5443154fac-Abstract-Conference.html)

**Abstract**:

Current deep-learning models for object recognition are known to be heavily biased toward texture. In contrast, human visual systems are known to be biased toward shape and structure. What could be the design principles in human visual systems that led to this difference? How could we introduce more shape bias into the deep learning models? In this paper, we report that sparse coding, a ubiquitous principle in the brain,  can in itself introduce shape bias into the network. We found that enforcing the sparse coding constraint using a non-differential Top-K operation  can lead to the emergence of structural encoding in neurons in convolutional neural networks,  resulting in a smooth decomposition of objects into parts and subparts and endowing the networks with shape bias.  We demonstrated this emergence of shape bias and its functional benefits for different network structures with various datasets. For object recognition convolutional neural networks, the shape bias leads to greater robustness against style and pattern change distraction. For the image synthesis generative adversary networks,  the emerged shape bias leads to more coherent and decomposable structures in the synthesized images. Ablation studies suggest that sparse codes tend to encode structures, whereas the more distributed codes tend to favor texture. Our code is host at the github repository: https://topk-shape-bias.github.io/

----

## [3141] Resilient Constrained Learning

**Authors**: *Ignacio Hounie, Alejandro Ribeiro, Luiz F. O. Chamon*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e32349fe7e3cd4f9ef598c2b7b7a31f4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e32349fe7e3cd4f9ef598c2b7b7a31f4-Abstract-Conference.html)

**Abstract**:

When deploying machine learning solutions, they must satisfy multiple requirements beyond accuracy, such as fairness, robustness, or safety. These requirements are imposed during training either implicitly, using penalties, or explicitly, using constrained optimization methods based on Lagrangian duality. Either way, specifying requirements is hindered by the presence of compromises and limited prior knowledge about the data. Furthermore, their impact on performance can often only be evaluated by actually solving the learning problem. This paper presents a constrained learning approach that adapts the requirements while simultaneously solving the learning task. To do so, it relaxes the learning constraints in a way that contemplates how much they affect the task at hand by balancing the performance gains obtained from the relaxation against a user-defined cost of that relaxation. We call this approach resilient constrained learning after the term used to describe ecological systems that adapt to disruptions by modifying their operation. We show conditions under which this balance can be achieved and introduce a practical algorithm to compute it, for which we derive approximation and generalization guarantees. We showcase the advantages of this resilient learning method in image classification tasks involving multiple potential invariances and in federated learning under distribution shift.

----

## [3142] Recovering Simultaneously Structured Data via Non-Convex Iteratively Reweighted Least Squares

**Authors**: *Christian Kümmerle, Johannes Maly*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e33a4d41305fb34316df6f3fa8a0e58c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e33a4d41305fb34316df6f3fa8a0e58c-Abstract-Conference.html)

**Abstract**:

We propose a new algorithm for the problem of recovering data that adheres to multiple, heterogenous low-dimensional structures from linear observations. Focussing on data matrices that are simultaneously row-sparse and low-rank, we propose and analyze an iteratively reweighted least squares (IRLS) algorithm that is able to leverage both structures. In particular, it optimizes a combination of non-convex surrogates for row-sparsity and rank, a balancing of which is built into the algorithm. We prove locally quadratic convergence of the iterates to a simultaneously structured data matrix in a regime of minimal sample complexity (up to constants and a logarithmic factor), which is known to be impossible for a combination of convex surrogates. In experiments, we show that the IRLS method exhibits favorable empirical convergence, identifying simultaneously row-sparse and low-rank matrices from fewer measurements than state-of-the-art methods.

----

## [3143] Error Bounds for Learning with Vector-Valued Random Features

**Authors**: *Samuel Lanthaler, Nicholas H. Nelsen*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e34d908241aef40440e61d2a27715424-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e34d908241aef40440e61d2a27715424-Abstract-Conference.html)

**Abstract**:

This paper provides a comprehensive error analysis of learning with vector-valued random features (RF). The theory is developed for RF ridge regression in a fully general infinite-dimensional input-output setting, but nonetheless applies to and improves existing finite-dimensional analyses. In contrast to comparable work in the literature, the approach proposed here relies on a direct analysis of the underlying risk functional and completely avoids the explicit RF ridge regression solution formula in terms of random matrices. This removes the need for concentration results in random matrix theory or their generalizations to random operators. The main results established in this paper include strong consistency of vector-valued RF estimators under model misspecification and minimax optimal convergence rates in the well-specified setting. The parameter complexity (number of random features) and sample complexity (number of labeled data) required to achieve such rates are comparable with Monte Carlo intuition and free from logarithmic factors.

----

## [3144] CoDA: Collaborative Novel Box Discovery and Cross-modal Alignment for Open-vocabulary 3D Object Detection

**Authors**: *Yang Cao, Yihan Zeng, Hang Xu, Dan Xu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e352b765e625934ce86919995e2371aa-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e352b765e625934ce86919995e2371aa-Abstract-Conference.html)

**Abstract**:

Open-vocabulary 3D Object Detection (OV-3DDet) aims to detect objects from an arbitrary list of categories within a 3D scene, which remains seldom explored in the literature. There are primarily two fundamental problems in OV-3DDet, i.e., localizing and classifying novel objects. This paper aims at addressing the two problems simultaneously via a unified framework, under the condition of limited base categories. To localize novel 3D objects, we propose an effective 3D Novel Object Discovery strategy, which utilizes both the 3D box geometry priors and 2D semantic open-vocabulary priors to generate pseudo box labels of the novel objects. To classify novel object boxes, we further develop a cross-modal alignment module based on discovered novel boxes, to align feature spaces between 3D pointcloud and image/text modalities. Specifically, the alignment process contains a class-agnostic and a class-discriminative alignment, incorporating not only the base objects with annotations but also the increasingly discovered novel objects, resulting in an iteratively enhanced alignment. The novel box discovery and crossmodal alignment are jointly learned to collaboratively benefit each other. Thenovel object discovery can directly impact the cross-modal alignment, while a better feature alignment can, in turn, boost the localization capability, leading to a unified OV-3DDet framework, named CoDA, for simultaneous novel object localization and classification. Extensive experiments on two challenging datasets (i.e., SUN-RGBD and ScanNet) demonstrate the effectiveness of our method and also show a significant mAP improvement upon the best-performing alternative method by 80%. Codes and pre-trained models are released on the project page.

----

## [3145] Don't blame Dataset Shift! Shortcut Learning due to Gradients and Cross Entropy

**Authors**: *Aahlad Manas Puli, Lily H. Zhang, Yoav Wald, Rajesh Ranganath*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e35460304fdf6df523f068a59aaf8829-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e35460304fdf6df523f068a59aaf8829-Abstract-Conference.html)

**Abstract**:

Common explanations for shortcut learning assume that the shortcut improves prediction only under the training distribution. Thus, models trained in the typical way by minimizing log-loss using gradient descent, which we call default-ERM, should utilize the shortcut. However, even when the stable feature determines the label in the training distribution and the shortcut does not provide any additional information, like in perception tasks, default-ERM exhibits shortcut learning. Why are such solutions preferred when the loss can be driven to zero when using the stable feature alone? By studying a linear perception task, we show that default-ERM’s preference for maximizing the margin, even without overparameterization, leads to models that depend more on the shortcut than the stable feature. This insight suggests that default-ERM’s implicit inductive bias towards max-margin may be unsuitable for perception tasks. Instead, we consider inductive biases toward uniform margins. We show that uniform margins guarantee sole dependence on the perfect stable feature in the linear perception task and suggest alternative loss functions, termed margin control (MARG-CTRL), that encourage uniform-margin solutions. MARG-CTRL techniques mitigate shortcut learning on a variety of vision and language tasks, showing that changing inductive biases can remove the need for complicated shortcut-mitigating methods in perception tasks.

----

## [3146] Scan and Snap: Understanding Training Dynamics and Token Composition in 1-layer Transformer

**Authors**: *Yuandong Tian, Yiping Wang, Beidi Chen, Simon S. Du*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e359ebe56ba306b674e8952349c6049e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e359ebe56ba306b674e8952349c6049e-Abstract-Conference.html)

**Abstract**:

Transformer architecture has shown impressive performance in multiple research domains and has become the backbone of many neural network models. However, there is limited understanding on how it works. In particular, with a simple predictive loss,  how the representation emerges from the gradient \emph{training dynamics} remains a mystery. In this paper, for 1-layer transformer with one self-attention layer plus one decoder layer, we analyze its SGD training dynamics for the task of next token prediction in a mathematically rigorous manner. We open the black box of the dynamic process of how the self-attention layer combines input tokens, and reveal the nature of underlying inductive bias. More specifically, with the assumption (a) no positional encoding, (b) long input sequence, and (c) the decoder layer learns faster than the self-attention layer, we prove that self-attention acts as a \emph{discriminative scanning algorithm}:  starting from uniform attention, it gradually attends more to distinct key tokens for a specific next token to be predicted, and pays less attention to common key tokens that occur across different next tokens. Among distinct tokens, it progressively drops attention weights, following the order of low to high co-occurrence between the key and the query token in the training set. Interestingly, this procedure does not lead to winner-takes-all, but stops due to a \emph{phase transition} that is controllable by the learning rate of the decoder layer, leaving (almost) fixed token combination. We verify this \textbf{\emph{scan and snap}} dynamics on synthetic and real-world data (WikiText-103).

----

## [3147] Stein Π-Importance Sampling

**Authors**: *Congye Wang, Ye Chen, Heishiro Kanagawa, Chris J. Oates*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e389b15166cf98966ba058965a8c17e3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e389b15166cf98966ba058965a8c17e3-Abstract-Conference.html)

**Abstract**:

Stein discrepancies have emerged as a powerful tool for retrospective improvement of Markov chain Monte Carlo output.  However, the question of how to design Markov chains that are well-suited to such post-processing has yet to be addressed.  This paper studies Stein importance sampling, in which weights are assigned to the states visited by a $\Pi$-invariant Markov chain to obtain a consistent approximation of $P$, the intended target.  Surprisingly, the optimal choice of $\Pi$ is not identical to the target $P$; we therefore propose an explicit construction for $\Pi$ based on a novel variational argument.  Explicit conditions for convergence of Stein $\Pi$-Importance Sampling are established.  For $\approx 70$% of tasks in the PosteriorDB benchmark, a significant improvement over the analogous post-processing of $P$-invariant Markov chains is reported.

----

## [3148] GPT4Tools: Teaching Large Language Model to Use Tools via Self-instruction

**Authors**: *Rui Yang, Lin Song, Yanwei Li, Sijie Zhao, Yixiao Ge, Xiu Li, Ying Shan*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e393677793767624f2821cec8bdd02f1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e393677793767624f2821cec8bdd02f1-Abstract-Conference.html)

**Abstract**:

This paper aims to efficiently enable Large Language Models (LLMs) to use multi-modal tools.The advanced proprietary LLMs, such as ChatGPT and GPT-4, have shown great potential for tool usage through sophisticated prompt engineering.Nevertheless, these models typically rely on prohibitive computational costs and publicly inaccessible data.To address these challenges, we propose the GPT4Tools based on self-instruct to enable open-source LLMs, such as LLaMA and OPT, to use tools.It generates an instruction-following dataset by prompting an advanced teacher with various multi-modal contexts.By using the Low-Rank Adaptation (LoRA) optimization, our approach facilitates the open-source LLMs to solve a range of visual problems, including visual comprehension and image generation.Moreover, we provide a benchmark to evaluate the ability of LLMs to use tools, which is performed in both zero-shot and fine-tuning ways.Extensive experiments demonstrate the effectiveness of our method on various language models, which not only significantly improves the accuracy of invoking seen tools, but also enables the zero-shot capacity for unseen tools.

----

## [3149] Reinforcement Learning with Fast and Forgetful Memory

**Authors**: *Steven D. Morad, Ryan Kortvelesy, Stephan Liwicki, Amanda Prorok*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e3bf2f0f10774c474de22a12cb060e2c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e3bf2f0f10774c474de22a12cb060e2c-Abstract-Conference.html)

**Abstract**:

Nearly all real world tasks are inherently partially observable, necessitating the use of memory in Reinforcement Learning (RL). Most model-free approaches summarize the trajectory into a latent Markov state using memory models borrowed from Supervised Learning (SL), even though RL tends to exhibit different training and efficiency characteristics. Addressing this discrepancy, we introduce Fast and Forgetful Memory, an algorithm-agnostic memory model designed specifically for RL. Our approach constrains the model search space via strong structural priors inspired by computational psychology. It is a drop-in replacement for recurrent neural networks (RNNs) in recurrent RL algorithms, achieving greater reward than RNNs across various recurrent benchmarks and algorithms without changing any hyperparameters. Moreover, Fast and Forgetful Memory exhibits training speeds two orders of magnitude faster than RNNs, attributed to its logarithmic time and linear space complexity. Our implementation is available at https://github.com/proroklab/ffm.

----

## [3150] Systematic Visual Reasoning through Object-Centric Relational Abstraction

**Authors**: *Taylor Webb, Shanka Subhra Mondal, Jonathan D. Cohen*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e3cdc587873dd1d00ac78f0c1f9aa60c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e3cdc587873dd1d00ac78f0c1f9aa60c-Abstract-Conference.html)

**Abstract**:

Human visual reasoning is characterized by an ability to identify abstract patterns from only a small number of examples, and to systematically generalize those patterns to novel inputs. This capacity depends in large part on our ability to represent complex visual inputs in terms of both objects and relations. Recent work in computer vision has introduced models with the capacity to extract object-centric representations, leading to the ability to process multi-object visual inputs, but falling short of the systematic generalization displayed by human reasoning. Other recent models have employed inductive biases for relational abstraction to achieve systematic generalization of learned abstract rules, but have generally assumed the presence of object-focused inputs. Here, we combine these two approaches, introducing Object-Centric Relational Abstraction (OCRA), a model that extracts explicit representations of both objects and abstract relations, and achieves strong systematic generalization in tasks (including a novel dataset, CLEVR-ART, with greater visual complexity) involving complex visual displays.

----

## [3151] In-Context Impersonation Reveals Large Language Models' Strengths and Biases

**Authors**: *Leonard Salewski, Stephan Alaniz, Isabel Rio-Torto, Eric Schulz, Zeynep Akata*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e3fe7b34ba4f378df39cb12a97193f41-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e3fe7b34ba4f378df39cb12a97193f41-Abstract-Conference.html)

**Abstract**:

In everyday conversations, humans can take on different roles and adapt their vocabulary to their chosen roles. We explore whether LLMs can take on, that is impersonate, different roles when they generate text in-context. We ask LLMs to assume different personas before solving vision and language tasks. We do this by prefixing the prompt with a persona that is associated either with a social identity or domain expertise. In a multi-armed bandit task, we find that LLMs pretending to be children of different ages recover human-like developmental stages of exploration. In a language-based reasoning task, we find that LLMs impersonating domain experts perform better than LLMs impersonating non-domain experts. Finally, we test whether LLMs' impersonations are complementary to visual information when describing different categories. We find that impersonation can improve performance: an LLM prompted to be a bird expert describes birds better than one prompted to be a car expert. However, impersonation can also uncover LLMs' biases: an LLM prompted to be a man describes cars better than one prompted to be a woman. These findings demonstrate that LLMs are capable of taking on diverse roles and that this in-context impersonation can be used to uncover their strengths and hidden biases. Our code is available at https://github.com/ExplainableML/in-context-impersonation.

----

## [3152] The s-value: evaluating stability with respect to distributional shifts

**Authors**: *Suyash Gupta, Dominik Rothenhäusler*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e3fea99df80195b316cefa7aa6099cd5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e3fea99df80195b316cefa7aa6099cd5-Abstract-Conference.html)

**Abstract**:

Common statistical measures of uncertainty such as $p$-values and confidence intervals quantify the uncertainty due to sampling, that is, the uncertainty due to not observing the full population. However, sampling is not the only source of uncertainty. In practice, distributions change between locations and across time. This makes it difficult to gather knowledge that transfers across data sets. We propose a measure of instability that quantifies the distributional instability of a statistical parameter with respect to Kullback-Leibler divergence, that is, the sensitivity of the parameter under general distributional perturbations within a Kullback-Leibler divergence ball. In addition, we quantify the instability of parameters with respect to directional or variable-specific shifts. Measuring instability with respect to directional shifts can be used to detect under which kind of distribution shifts a statistical conclusion might be reversed. We discuss how such knowledge can inform data collection for transfer learning of statistical parameters under shifted distributions. We evaluate the performance of the proposed measure on real data and show that it can elucidate the distributional instability of a parameter with respect to certain shifts and can be used to improve estimation accuracy under shifted distributions.

----

## [3153] When Does Optimizing a Proper Loss Yield Calibration?

**Authors**: *Jaroslaw Blasiok, Parikshit Gopalan, Lunjia Hu, Preetum Nakkiran*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e4165c96702bac5f4962b70f3cf2f136-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e4165c96702bac5f4962b70f3cf2f136-Abstract-Conference.html)

**Abstract**:

Optimizing proper loss functions is popularly believed to yield predictors with good calibration properties; the intuition being that for such losses, the global optimum is to predict the ground-truth probabilities, which is indeed calibrated. However, typical machine learning models are trained to approximately minimize loss over restricted families of predictors, that are unlikely to contain the ground truth. Under what circumstances does optimizing proper loss  over a restricted family yield calibrated models? What precise calibration guarantees does it give? In this work, we provide a rigorous answer to these questions. We replace the global optimality with a local optimality condition stipulating that the (proper) loss of the predictor cannot be reduced much by post-processing its predictions with a certain family of Lipschitz functions. We show that any predictor with this local optimality satisfies smooth calibration as defined in [Kakade and Foster, 2008, BÅ‚asiok et al., 2023]. Local optimality is plausibly satisfied by well-trained DNNs, which suggests an explanation for why they are calibrated from proper loss minimization alone. Finally, we show that the connection between local optimality and calibration error goes both ways: nearly calibrated predictors are also nearly locally optimal.

----

## [3154] Language Is Not All You Need: Aligning Perception with Language Models

**Authors**: *Shaohan Huang, Li Dong, Wenhui Wang, Yaru Hao, Saksham Singhal, Shuming Ma, Tengchao Lv, Lei Cui, Owais Khan Mohammed, Barun Patra, Qiang Liu, Kriti Aggarwal, Zewen Chi, Nils Johan Bertil Bjorck, Vishrav Chaudhary, Subhojit Som, Xia Song, Furu Wei*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e425b75bac5742a008d643826428787c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e425b75bac5742a008d643826428787c-Abstract-Conference.html)

**Abstract**:

A big convergence of language, multimodal perception, action, and world modeling is a key step toward artificial general intelligence. In this work, we introduce KOSMOS-1, a Multimodal Large Language Model (MLLM) that can perceive general modalities, learn in context (i.e., few-shot), and follow instructions (i.e., zero-shot). Specifically, we train KOSMOS-1 from scratch on web-scale multi-modal corpora, including arbitrarily interleaved text and images, image-caption pairs, and text data. We evaluate various settings, including zero-shot, few-shot, and multimodal chain-of-thought prompting, on a wide range of tasks without any gradient updates or finetuning. Experimental results show that KOSMOS-1 achieves impressive performance on (i) language understanding, generation, and even OCR-free NLP (directly fed with document images), (ii) perception-language tasks, including multimodal dialogue, image captioning, visual question answering, and (iii) vision tasks, such as image recognition with descriptions (specifying classification via text instructions). We also show that MLLMs can benefit from cross-modal transfer, i.e., transfer knowledge from language to multimodal, and from multimodal to language. In addition, we introduce a dataset of Raven IQ test, which diagnoses the nonverbal reasoning capability of MLLMs.

----

## [3155] Out-of-distribution Detection Learning with Unreliable Out-of-distribution Sources

**Authors**: *Haotian Zheng, Qizhou Wang, Zhen Fang, Xiaobo Xia, Feng Liu, Tongliang Liu, Bo Han*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e43f900f571de6c96a70d5724a0fb565-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e43f900f571de6c96a70d5724a0fb565-Abstract-Conference.html)

**Abstract**:

Out-of-distribution (OOD) detection discerns OOD data where the predictor cannot make valid predictions as in-distribution (ID) data, thereby increasing the reliability of open-world classification. However, it is typically hard to collect real out-of-distribution (OOD) data for training a predictor capable of discerning ID and OOD patterns. This obstacle gives rise to data generation-based learning methods, synthesizing OOD data via data generators for predictor training without requiring any real OOD data. Related methods typically pre-train a generator on ID data and adopt various selection procedures to find those data likely to be the OOD cases. However, generated data may still coincide with ID semantics, i.e., mistaken OOD generation remains, confusing the predictor between ID and OOD data. To this end, we suggest that generated data (with mistaken OOD generation) can be used to devise an auxiliary OOD detection task to facilitate real OOD detection. Specifically, we can ensure that learning from such an auxiliary task is beneficial if the ID and the OOD parts have disjoint supports, with the help of a well-designed training procedure for the predictor. Accordingly, we propose a powerful data generation-based learning method named Auxiliary Task-based OOD Learning (ATOL) that can relieve the mistaken OOD generation. We conduct extensive experiments under various OOD detection setups, demonstrating the effectiveness of our method against its advanced counterparts.

----

## [3156] Robust covariance estimation with missing values and cell-wise contamination

**Authors**: *Grégoire Pacreau, Karim Lounici*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e444859b2a22df6b56af9381ad1e9480-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e444859b2a22df6b56af9381ad1e9480-Abstract-Conference.html)

**Abstract**:

Large datasets are often affected by cell-wise outliers in the form of missing or erroneous data. However, discarding any samples containing outliers may result in a dataset that is too small to accurately estimate the covariance matrix. Moreover, the robust procedures designed to address this problem require the invertibility of the covariance operator and thus are not effective on high-dimensional data. In this paper, we propose an unbiased estimator for the covariance in the presence of missing values that does not require any imputation step and still achieves near minimax statistical accuracy with the operator norm. We also advocate for its use in combination with cell-wise outlier detection methods to tackle cell-wise contamination in a high-dimensional and low-rank setting, where state-of-the-art methods may suffer from numerical instability and long computation times. To complement our theoretical findings, we conducted an experimental study which demonstrates the superiority of our approach over the state of the art both in low and high dimension settings.

----

## [3157] Patch Diffusion: Faster and More Data-Efficient Training of Diffusion Models

**Authors**: *Zhendong Wang, Yifan Jiang, Huangjie Zheng, Peihao Wang, Pengcheng He, Zhangyang Wang, Weizhu Chen, Mingyuan Zhou*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e4667dd0a5a54b74019b72b677ed8ec1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e4667dd0a5a54b74019b72b677ed8ec1-Abstract-Conference.html)

**Abstract**:

Diffusion models are powerful, but they require a lot of time and data to train. We propose Patch Diffusion, a generic patch-wise training framework, to significantly reduce the training time costs while improving data efficiency, which thus helps democratize diffusion model training to broader users. At the core of our innovations is a new conditional score function at the patch level, where the patch location in the original image is included as additional coordinate channels, while the patch size is randomized and diversified throughout training to encode the cross-region dependency at multiple scales. Sampling with our method is as easy as in the original diffusion model. Through Patch Diffusion, we could achieve $\mathbf{\ge 2\times}$ faster training, while maintaining comparable or better generation quality. Patch Diffusion meanwhile improves the performance of diffusion models trained on relatively small datasets, $e.g.$, as few as 5,000 images to train from scratch. We achieve outstanding FID scores in line with state-of-the-art benchmarks: 1.77 on CelebA-64$\times$64, 1.93 on AFHQv2-Wild-64$\times$64, and 2.72 on ImageNet-256$\times$256. We share our code and pre-trained models at https://github.com/Zhendong-Wang/Patch-Diffusion.

----

## [3158] Clustering the Sketch: Dynamic Compression for Embedding Tables

**Authors**: *Henry Ling-Hei Tsang, Thomas D. Ahle*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e468a76212a58c1af94a3d235151944a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e468a76212a58c1af94a3d235151944a-Abstract-Conference.html)

**Abstract**:

Embedding tables are used by machine learning systems  to work with categorical features.In modern Recommendation Systems, these tables can be very large, necessitating the development of new methods for fitting them in memory, even during training.We suggest Clustered Compositional Embeddings (CCE) which combines clustering-based compression like quantization to codebooks with dynamic methods like The Hashing Trick and Compositional Embeddings [Shi et al., 2020].Experimentally CCE achieves the best of both worlds: The high compression rate of codebook-based quantization, but \emph{dynamically} like hashing-based methods, so it can be used during training.Theoretically, we prove that CCE is guaranteed to converge to the optimal codebook and give a tight bound for the number of iterations required.

----

## [3159] Dynamic Personalized Federated Learning with Adaptive Differential Privacy

**Authors**: *Xiyuan Yang, Wenke Huang, Mang Ye*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e4724af0e2a0d52ce5a0a4e084b87f59-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e4724af0e2a0d52ce5a0a4e084b87f59-Abstract-Conference.html)

**Abstract**:

Personalized federated learning with differential privacy has been considered a feasible solution to address non-IID distribution of data and privacy leakage risks. However, current personalized federated learning methods suffer from inflexible personalization and convergence difficulties due to two main factors: 1) Firstly, we observe that the prevailing personalization methods mainly achieve this by personalizing a fixed portion of the model, which lacks flexibility. 2) Moreover, we further demonstrate that the default gradient calculation is sensitive to the widely-used clipping operations in differential privacy, resulting in difficulties in convergence. Considering that Fisher information values can serve as an effective measure for estimating the information content of parameters by reflecting the model sensitivity to parameters, we aim to leverage this property to address the aforementioned challenges. In this paper, we propose a novel federated learning method with Dynamic Fisher Personalization and Adaptive Constraint (FedDPA) to handle these challenges. Firstly, by using layer-wise Fisher information to measure the information content of local parameters, we retain local parameters with high Fisher values during the personalization process, which are considered informative, simultaneously prevent these parameters from noise perturbation. Secondly, we introduce an adaptive approach by applying differential constraint strategies to personalized parameters and shared parameters identified in the previous for better convergence.  Our method boosts performance through flexible personalization while mitigating the slow convergence caused by clipping operations. Experimental results on CIFAR-10, FEMNIST and SVHN dataset demonstrate the effectiveness of our approach in achieving better performance and robustness against clipping, under personalized federated learning with differential privacy.

----

## [3160] Bias in Evaluation Processes: An Optimization-Based Model

**Authors**: *L. Elisa Celis, Amit Kumar, Anay Mehrotra, Nisheeth K. Vishnoi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e4748b6b6ca49f04b6a8cfce1d5f9a70-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e4748b6b6ca49f04b6a8cfce1d5f9a70-Abstract-Conference.html)

**Abstract**:

Biases with respect to socially-salient attributes of individuals have been well documented in evaluation processes used in settings such as admissions and hiring. We view such an evaluation process as a transformation of a  distribution of the true utility of an individual for a task to an observed distribution and model it as a solution to a loss minimization problem subject to an information constraint. Our model has two parameters that have been identified as factors leading to biases: the resource-information trade-off parameter in the information constraint and the risk-averseness parameter in the loss function.  We characterize the distributions that arise from our model and study the effect of the parameters on the observed distribution. The outputs of our model enrich the class of distributions that can be used to capture variation across groups in the observed evaluations. We empirically validate our model by fitting real-world datasets and use it to study the effect of interventions in a downstream selection task. These results contribute to an understanding of the emergence of bias in evaluation processes and provide tools to guide the deployment of interventions to mitigate biases.

----

## [3161] Data Minimization at Inference Time

**Authors**: *Cuong Tran, Nando Fioretto*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e48880ea81caa7836e6a0694049093ae-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e48880ea81caa7836e6a0694049093ae-Abstract-Conference.html)

**Abstract**:

In high-stakes domains such as legal, banking, hiring, and healthcare, learning models frequently rely on sensitive user information for inference, necessitating the complete set of features. This not only poses significant privacy risks for individuals but also demands substantial human effort from organizations to verify information accuracy. This study asks whether it is necessary to use all input features for accurate predictions at inference time. The paper demonstrates that, in a personalized setting, individuals may only need to disclose a small subset of features without compromising decision-making accuracy. The paper also provides an efficient sequential algorithm to determine the appropriate attributes for each individual to provide. Evaluations across various learning tasks show that individuals can potentially report as little as 10\% of their information while maintaining the same accuracy level as a model that employs the full set of user information.

----

## [3162] Learning Adaptive Tensorial Density Fields for Clean Cryo-ET Reconstruction

**Authors**: *Yuanhao Wang, Ramzi Idoughi, Wolfgang Heidrich*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e4be7e9867ef163563f4a5e90cec478f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e4be7e9867ef163563f4a5e90cec478f-Abstract-Conference.html)

**Abstract**:

We present a novel learning-based framework for reconstructing 3D structures from tilt-series cryo-Electron Tomography (cryo-ET) data. Cryo-ET is a powerful imaging technique that can achieve near-atomic resolutions. Still, it suffers from challenges such as missing-wedge acquisition, large data size, and high noise levels. Our framework addresses these challenges by using an adaptive tensorial-based representation for the 3D density field of the scanned sample. First, we optimize a quadtree structure to partition the volume of interest. Then, we learn a vector-matrix factorization of the tensor representing the density field in each node. Moreover, we use a loss function that combines a differentiable tomographic formation model with three regularization terms: total variation, boundary consistency constraint, and an isotropic Fourier prior. Our framework allows us to query the density at any location using the learned representation and obtain a high-quality 3D tomogram. We demonstrate the superiority of our framework over existing methods using synthetic and real data. Thus, our framework boosts the quality of the reconstruction while reducing the computation time and the memory footprint. The code is available at https://github.com/yuanhaowang1213/adaptivetensordf.

----

## [3163] Resetting the Optimizer in Deep RL: An Empirical Study

**Authors**: *Kavosh Asadi, Rasool Fakoor, Shoham Sabach*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e4bf5c3245fd92a4554a16af9803b757-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e4bf5c3245fd92a4554a16af9803b757-Abstract-Conference.html)

**Abstract**:

We focus on the task of approximating the optimal value function in deep reinforcement learning. This iterative process is comprised of solving a sequence of optimization problems where the loss function changes per iteration. The common approach to solving this sequence of problems is to employ modern variants of the stochastic gradient descent algorithm such as Adam. These optimizers maintain their own internal parameters such as estimates of the first-order and the second-order moments of the gradient, and update them over time. Therefore, information obtained in previous iterations is used to solve the optimization problem in the current iteration. We demonstrate that this can contaminate the moment estimates because the optimization landscape can change arbitrarily from one iteration to the next one. To hedge against this negative effect, a simple idea is to reset the internal parameters of the optimizer when starting a new iteration. We empirically investigate this resetting idea by employing various optimizers in conjunction with the Rainbow algorithm. We demonstrate that this simple modification significantly improves the performance of deep RL on the Atari benchmark.

----

## [3164] Why Does Sharpness-Aware Minimization Generalize Better Than SGD?

**Authors**: *Zixiang Chen, Junkai Zhang, Yiwen Kou, Xiangning Chen, Cho-Jui Hsieh, Quanquan Gu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e4d3fe32495088805bbbb4f1de63e947-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e4d3fe32495088805bbbb4f1de63e947-Abstract-Conference.html)

**Abstract**:

The challenge of overfitting, in which the model memorizes the training data and fails to generalize to test data, has become increasingly significant in the training of large neural networks. To tackle this challenge, Sharpness-Aware Minimization (SAM) has emerged as a promising training method, which can improve the generalization of neural networks even in the presence of label noise. However, a deep understanding of how SAM works, especially in the setting of nonlinear neural networks and classification tasks, remains largely missing. This paper fills this gap by demonstrating why SAM generalizes better than Stochastic Gradient Descent (SGD) for a certain data model and two-layer convolutional ReLU networks. The loss landscape of our studied problem is nonsmooth, thus current explanations for the success of SAM based on the Hessian information are insufficient. Our result explains the benefits of SAM, particularly its ability to prevent noise learning in the early stages, thereby facilitating more effective learning of features. Experiments on both synthetic and real data corroborate our theory.

----

## [3165] Grassmann Manifold Flows for Stable Shape Generation

**Authors**: *Ryoma Yataka, Kazuki Hirashima, Masashi Shiraishi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e50e253e21cbcdcd200394f61d73acc8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e50e253e21cbcdcd200394f61d73acc8-Abstract-Conference.html)

**Abstract**:

Recently, studies on machine learning have focused on methods that use symmetry implicit in a specific manifold as an inductive bias.Grassmann manifolds provide the ability to handle fundamental shapes represented as shape spaces, enabling stable shape analysis. In this paper, we present a novel approach in which we establish the theoretical foundations for learning distributions on the Grassmann manifold via continuous normalization flows, with the explicit goal of generating stable shapes.Our approach facilitates more robust generation by effectively eliminating the influence of extraneous transformations, such as rotations and inversions, through learning and generating within a Grassmann manifold designed to accommodate the essential shape information of the object.The experimental results indicated that the proposed method could generate high-quality samples by capturing the data structure.Furthermore, the proposed method significantly outperformed state-of-the-art methods in terms of the log-likelihood or evidence lower bound.The results obtained are expected to stimulate further research in this field, leading to advances for stable shape generation and analysis.

----

## [3166] Marich: A Query-efficient Distributionally Equivalent Model Extraction Attack

**Authors**: *Pratik Karmakar, Debabrota Basu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e5440ffceaf4831b5f98652b8a27ffde-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e5440ffceaf4831b5f98652b8a27ffde-Abstract-Conference.html)

**Abstract**:

We study design of black-box model extraction attacks that can *send minimal number of queries from* a *publicly available dataset* to a target ML model through a predictive API with an aim *to create an informative and distributionally equivalent replica* of the target.First, we define *distributionally equivalent* and *Max-Information model extraction* attacks, and reduce them into a variational optimisation problem. The attacker sequentially solves this optimisation problem to select the most informative queries that simultaneously maximise the entropy and reduce the mismatch between the target and the stolen models. This leads to *an active sampling-based query selection algorithm*, Marich, which is *model-oblivious*. Then, we evaluate Marich on different text and image data sets, and different models, including CNNs and BERT. Marich extracts models that achieve $\sim 60-95\%$ of true model's accuracy and uses $\sim 1,000 - 8,500$ queries from the publicly available datasets, which are different from the private training datasets. Models extracted by Marich yield prediction distributions, which are $\sim2-4\times$ closer to the target's distribution in comparison to the existing active sampling-based attacks. The extracted models also lead to 84-96$\%$ accuracy under membership inference attacks. Experimental results validate that Marich is *query-efficient*, and capable of performing task-accurate, high-fidelity, and informative model extraction.

----

## [3167] Evaluating Post-hoc Explanations for Graph Neural Networks via Robustness Analysis

**Authors**: *Junfeng Fang, Wei Liu, Yuan Gao, Zemin Liu, An Zhang, Xiang Wang, Xiangnan He*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e55c2f3fdde519014c879aa3554414c0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e55c2f3fdde519014c879aa3554414c0-Abstract-Conference.html)

**Abstract**:

This work studies the evaluation of explaining graph neural networks (GNNs), which is crucial to the credibility of post-hoc explainability in practical usage. Conventional evaluation metrics, and even explanation methods -- which mainly follow the paradigm of feeding the explanatory subgraph and measuring output difference -- always suffer from the notorious out-of-distribution (OOD) issue. In this work, we endeavor to confront the issue by introducing a novel evaluation metric, termed OOD-resistant Adversarial Robustness (OAR). Specifically, we draw inspiration from the notion of adversarial robustness and evaluate post-hoc explanation subgraphs by calculating their robustness under attack. On top of that, an elaborate OOD reweighting block is inserted into the pipeline to confine the evaluation process to the original data distribution. For applications involving large datasets, we further devise a Simplified version of OAR (SimOAR), which achieves a significant improvement in computational efficiency at the cost of a small amount of performance. Extensive empirical studies validate the effectiveness of our OAR and SimOAR.

----

## [3168] A Unifying Perspective on Multi-Calibration: Game Dynamics for Multi-Objective Learning

**Authors**: *Nika Haghtalab, Michael I. Jordan, Eric Zhao*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e55edcdb01ac45c839a602f96e09fbcb-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e55edcdb01ac45c839a602f96e09fbcb-Abstract-Conference.html)

**Abstract**:

We provide a unifying framework for the design and analysis of multi-calibrated predictors. By placing the multi-calibration problem in the general setting of multi-objective learning---where learning guarantees must hold simultaneously over a set of distributions and loss functions---we exploit connections to game dynamics to achieve state-of-the-art guarantees for a diverse set of multi-calibration learning problems. In addition to shedding light on existing multi-calibration guarantees and greatly simplifying their analysis, our approach also yields improved guarantees, such as error tolerances that scale with the square-root of group size versus the constant tolerances guaranteed by prior works, and improving the complexity of $k$-class multi-calibration by an exponential factor of $k$ versus Gopalan et al.. Beyond multi-calibration, we use these game dynamics to address emerging considerations in the study of group fairness and multi-distribution learning.

----

## [3169] Not All Neuro-Symbolic Concepts Are Created Equal: Analysis and Mitigation of Reasoning Shortcuts

**Authors**: *Emanuele Marconato, Stefano Teso, Antonio Vergari, Andrea Passerini*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e560202b6e779a82478edb46c6f8f4dd-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e560202b6e779a82478edb46c6f8f4dd-Abstract-Conference.html)

**Abstract**:

Neuro-Symbolic (NeSy) predictive models hold the promise of improved compliance with given constraints, systematic generalization, and interpretability, as they allow to infer labels that are consistent with some prior knowledge by reasoning over high-level concepts extracted from sub-symbolic inputs. It was recently shown that NeSy predictors are affected by reasoning shortcuts: they can attain high accuracy but by leveraging concepts with \textit{unintended semantics}, thus coming short of their promised advantages. Yet, a systematic characterization of reasoning shortcuts and of potential mitigation strategies is missing. This work fills this gap by characterizing them as unintended optima of the learning objective and identifying four key conditions behind their occurrence. Based on this, we derive several natural mitigation strategies, and analyze their efficacy both theoretically and empirically. Our analysis shows reasoning shortcuts are difficult to deal with, casting doubts on the trustworthiness and interpretability of existing NeSy solutions.

----

## [3170] Contrastive Moments: Unsupervised Halfspace Learning in Polynomial Time

**Authors**: *Xinyuan Cao, Santosh S. Vempala*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e5a71ba556c84fef542aaace56b6cfe9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e5a71ba556c84fef542aaace56b6cfe9-Abstract-Conference.html)

**Abstract**:

We give a polynomial-time algorithm for learning high-dimensional halfspaces with margins in $d$-dimensional space to within desired Total Variation (TV) distance when the ambient distribution is an unknown affine transformation of the $d$-fold product of an (unknown) symmetric one-dimensional logconcave distribution, and the halfspace is introduced by deleting at least an $\epsilon$ fraction of the data in one of the component distributions. Notably, our algorithm does not need labels and establishes the unique (and efficient) identifiability of the hidden halfspace under this distributional assumption.  The sample and time complexity of the algorithm are polynomial in the dimension and $1/\epsilon$. The algorithm uses only the first two moments of *suitable re-weightings* of the empirical distribution, which we call *contrastive moments*; its analysis uses classical facts about generalized Dirichlet polynomials and relies crucially on a new monotonicity property of the moment ratio of truncations of logconcave distributions. Such algorithms, based only on first and second moments were suggested in earlier work, but hitherto eluded rigorous guarantees.Prior work addressed the special case when the underlying distribution is Gaussian via Non-Gaussian Component Analysis. We improve on this by providing polytime guarantees based on TV distance, in place of existing moment-bound guarantees that can be super-polynomial. Our work is also the first to go beyond Gaussians in this setting.

----

## [3171] Boosting Spectral Clustering on Incomplete Data via Kernel Correction and Affinity Learning

**Authors**: *Fangchen Yu, Runze Zhao, Zhan Shi, Yiwen Lu, Jicong Fan, Yicheng Zeng, Jianfeng Mao, Wenye Li*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e5aa7171449b83f8b4eec1623eac9906-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e5aa7171449b83f8b4eec1623eac9906-Abstract-Conference.html)

**Abstract**:

Spectral clustering has gained popularity for clustering non-convex data due to its simplicity and effectiveness. It is essential to construct a similarity graph using a high-quality affinity measure that models the local neighborhood relations among the data samples. However, incomplete data can lead to inaccurate affinity measures, resulting in degraded clustering performance. To address these issues, we propose an imputation-free framework with two novel approaches to improve spectral clustering on incomplete data. Firstly, we introduce a new kernel correction method that enhances the quality of the kernel matrix estimated on incomplete data with a theoretical guarantee, benefiting classical spectral clustering on pre-defined kernels. Secondly, we develop a series of affinity learning methods that equip the self-expressive framework with $\ell_p$-norm to construct an intrinsic affinity matrix with an adaptive extension. Our methods outperform existing data imputation and distance calibration techniques on benchmark datasets, offering a promising solution to spectral clustering on incomplete data in various real-world applications.

----

## [3172] DASpeech: Directed Acyclic Transformer for Fast and High-quality Speech-to-Speech Translation

**Authors**: *Qingkai Fang, Yan Zhou, Yang Feng*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e5b1c0d4866f72393c522c8a00eed4eb-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e5b1c0d4866f72393c522c8a00eed4eb-Abstract-Conference.html)

**Abstract**:

Direct speech-to-speech translation (S2ST) translates speech from one language into another using a single model. However, due to the presence of linguistic and acoustic diversity, the target speech follows a complex multimodal distribution, posing challenges to achieving both high-quality translations and fast decoding speeds for S2ST models. In this paper, we propose DASpeech, a non-autoregressive direct S2ST model which realizes both fast and high-quality S2ST. To better capture the complex distribution of the target speech, DASpeech adopts the two-pass architecture to decompose the generation process into two steps, where a linguistic decoder first generates the target text, and an acoustic decoder then generates the target speech based on the hidden states of the linguistic decoder. Specifically, we use the decoder of DA-Transformer as the linguistic decoder, and use FastSpeech 2 as the acoustic decoder. DA-Transformer models translations with a directed acyclic graph (DAG). To consider all potential paths in the DAG during training, we calculate the expected hidden states for each target token via dynamic programming, and feed them into the acoustic decoder to predict the target mel-spectrogram. During inference, we select the most probable path and take hidden states on that path as input to the acoustic decoder. Experiments on the CVSS Fr$\rightarrow$En benchmark demonstrate that DASpeech can achieve comparable or even better performance than the state-of-the-art S2ST model Translatotron 2, while preserving up to 18.53$\times$ speedup compared to the autoregressive baseline. Compared with the previous non-autoregressive S2ST model, DASpeech does not rely on knowledge distillation and iterative decoding, achieving significant improvements in both translation quality and decoding speed. Furthermore, DASpeech shows the ability to preserve the speaker's voice of the source speech during translation.

----

## [3173] Learning Large-scale Neural Fields via Context Pruned Meta-Learning

**Authors**: *Jihoon Tack, Subin Kim, Sihyun Yu, Jaeho Lee, Jinwoo Shin, Jonathan Richard Schwarz*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e5b5c402bb7bd5e60bede6961d6fe39e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e5b5c402bb7bd5e60bede6961d6fe39e-Abstract-Conference.html)

**Abstract**:

We introduce an efficient optimization-based meta-learning technique for large-scale neural field training by realizing significant memory savings through automated online context point selection. This is achieved by focusing each learning step on the subset of data with the highest expected immediate improvement in model quality, resulting in the almost instantaneous modeling of global structure and subsequent refinement of high-frequency details. We further improve the quality of our meta-learned initialization by introducing a bootstrap correction resulting in the minimization of any error introduced by reduced context sets while simultaneously mitigating the well-known myopia of optimization-based meta-learning. Finally, we show how gradient re-scaling at meta-test time allows the learning of extremely high-quality neural fields in significantly shortened optimization procedures. Our framework is model-agnostic, intuitive, straightforward to implement, and shows significant reconstruction improvements for a wide range of signals. We provide an extensive empirical evaluation on nine datasets across multiple multiple modalities, demonstrating state-of-the-art results while providing additional insight through careful analysis of the algorithmic components constituting our method. Code is available at https://github.com/jihoontack/GradNCP

----

## [3174] AlberDICE: Addressing Out-Of-Distribution Joint Actions in Offline Multi-Agent RL via Alternating Stationary Distribution Correction Estimation

**Authors**: *Daiki E. Matsunaga, Jongmin Lee, Jaeseok Yoon, Stefanos Leonardos, Pieter Abbeel, Kee-Eung Kim*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e5b6eb1dbabff82838d5e99f62de37c8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e5b6eb1dbabff82838d5e99f62de37c8-Abstract-Conference.html)

**Abstract**:

One of the main challenges in offline Reinforcement Learning (RL) is the distribution shift that arises from the learned policy deviating from the data collection policy. This is often addressed by avoiding out-of-distribution (OOD) actions during policy improvement as their presence can lead to substantial performance degradation. This challenge is amplified in the offline Multi-Agent RL (MARL) setting since the joint action space grows exponentially with the number of agents.To avoid this curse of dimensionality, existing MARL methods adopt either value decomposition methods or fully decentralized training of individual agents. However, even when combined with standard conservatism principles, these methods can still result in the selection of OOD joint actions in offline MARL. To this end, we introduce AlberDICE,an offline MARL algorithm that alternatively performs centralized training of individual agents based on stationary distribution optimization. AlberDICE circumvents the exponential complexity of MARL by computing the best response of one agent at a time while effectively avoiding OOD joint action selection. Theoretically, we show that the alternating optimization procedure converges to Nash policies. In the experiments, we demonstrate that AlberDICE significantly outperforms baseline algorithms on a standard suite of MARL benchmarks.

----

## [3175] Approximate inference of marginals using the IBIA framework

**Authors**: *Shivani Bathla, Vinita Vasudevan*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e5beb17e56bbb8fd562efeefab79425f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e5beb17e56bbb8fd562efeefab79425f-Abstract-Conference.html)

**Abstract**:

Exact inference of marginals in probabilistic graphical models (PGM) is known to be intractable, necessitating the use of approximate methods. Most of the existing variational techniques perform iterative message passing in loopy graphs which is slow to converge for many benchmarks. In this paper, we propose a new algorithm for marginal inference that is based on the incremental build-infer-approximate (IBIA) paradigm. Our algorithm converts the PGM into a sequence of linked clique tree forests (SLCTF) with bounded clique sizes, and then uses a heuristic belief update algorithm to infer the marginals. For the special case of Bayesian networks, we show that if the incremental build step in IBIA uses the topological order of variables then (a) the prior marginals are consistent in all CTFs in the SLCTF  and (b) the posterior marginals are consistent once all evidence variables are added to the SLCTF. In our approach, the belief propagation step is non-iterative and the accuracy-complexity trade-off is controlled using user-defined clique size bounds. Results for several benchmark sets from recent UAI competitions show that our method gives either better or comparable accuracy than existing variational and sampling based methods, with smaller runtimes.

----

## [3176] HiNeRV: Video Compression with Hierarchical Encoding-based Neural Representation

**Authors**: *Ho Man Kwan, Ge Gao, Fan Zhang, Andrew Gower, David Bull*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e5dc475c370ff42f2f96dddf8191a40c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e5dc475c370ff42f2f96dddf8191a40c-Abstract-Conference.html)

**Abstract**:

Learning-based video compression is currently a popular research topic, offering the potential to compete with conventional standard video codecs. In this context, Implicit Neural Representations (INRs) have previously been used to represent and compress image and video content, demonstrating relatively high decoding speed compared to other methods. However, existing INR-based methods have failed to deliver rate quality performance comparable with the state of the art in video compression. This is mainly due to the simplicity of the employed network architectures, which limit their representation capability. In this paper, we propose HiNeRV, an INR that combines light weight layers with novel hierarchical positional encodings. We employs depth-wise convolutional, MLP and interpolation layers to build the deep and wide network architecture with high capacity. HiNeRV is also a unified representation encoding videos in both frames and patches at the same time, which offers higher performance and flexibility than existing methods. We further build a video codec based on HiNeRV and a refined pipeline for training, pruning and quantization that can better preserve HiNeRV's performance during lossy model compression. The proposed method has been evaluated on both UVG and MCL-JCV datasets for video compression, demonstrating significant improvement over all existing INRs baselines and competitive performance when compared to learning-based codecs (72.3\% overall bit rate saving over HNeRV and 43.4\% over DCVC on the UVG dataset, measured in PSNR).

----

## [3177] Bicriteria Approximation Algorithms for the Submodular Cover Problem

**Authors**: *Wenjing Chen, Victoria G. Crawford*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e5eaf67f3405be58cd12848a89cd8ace-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e5eaf67f3405be58cd12848a89cd8ace-Abstract-Conference.html)

**Abstract**:

In this paper, we consider the optimization problem Submodular Cover (SCP), which is to find a minimum cardinality subset of a finite universe $U$ such that the value of a submodular function $f$ is above an input threshold $\tau$. In particular, we consider several variants of SCP including the general case, the case where $f$ is additionally assumed to be monotone, and finally the case where $f$ is a regularized monotone submodular function. Our most significant contributions are that: (i) We propose a scalable algorithm for monotone SCP that achieves nearly the same approximation guarantees as the standard greedy algorithm in significantly faster time; (ii) We are the first to develop an algorithm for general SCP that achieves a solution arbitrarily close to being feasible; and finally (iii) we are the first to develop algorithms for regularized SCP. Our algorithms are then demonstrated to be effective in an extensive experimental section on data summarization and graph cut, two applications of SCP.

----

## [3178] Responsible AI (RAI) Games and Ensembles

**Authors**: *Yash Gupta, Runtian Zhai, Arun Sai Suggala, Pradeep Ravikumar*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e6057bf047bcc5f86ebf4e8db6e24a1f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e6057bf047bcc5f86ebf4e8db6e24a1f-Abstract-Conference.html)

**Abstract**:

Several recent works have studied the societal effects of AI; these include issues such as fairness, robustness, and safety.  In many of these objectives, a learner seeks to minimize its worst-case loss over a set of predefined distributions (known as uncertainty sets), with usual examples being perturbed versions of the empirical distribution. In other words, the aforementioned problems can be written as min-max problems over these uncertainty sets. In this work, we provide a general framework for studying these problems, which we refer to as Responsible AI (RAI) games. We provide two classes of algorithms for solving these games:  (a) game-play based algorithms, and (b) greedy stagewise estimation algorithms. The former class is motivated by online learning and game theory, whereas the latter class is motivated by the classical statistical literature on boosting, and regression. We empirically demonstrate the applicability and competitive performance of our techniques for solving several RAI problems, particularly around subpopulation shift.

----

## [3179] May the Force be with You: Unified Force-Centric Pre-Training for 3D Molecular Conformations

**Authors**: *Rui Feng, Qi Zhu, Huan Tran, Binghong Chen, Aubrey Toland, Rampi Ramprasad, Chao Zhang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e637029c42aa593850eeebf46616444d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e637029c42aa593850eeebf46616444d-Abstract-Conference.html)

**Abstract**:

Recent works have shown the promise of learning pre-trained models for 3D molecular representation.However, existing pre-training models focus predominantly on equilibrium data and largely overlook off-equilibrium conformations.It is challenging to extend these methods to off-equilibrium data because their training objective relies on assumptions ofconformations being the local energy minima. We address this gap by proposing a force-centric pretraining model for 3D molecular conformations covering both equilibrium and off-equilibrium data.For off-equilibrium data, our model learns directly from their atomic forces. For equilibrium data, we introduce zero-force regularization and forced-based denoising techniques to approximate near-equilibrium forces.We obtain a unified pre-trained model for 3D molecular representation with over 15 million diverse conformations. Experiments show that, with our pre-training objective, we increase forces accuracy by around 3 times compared to the un-pre-trained Equivariant Transformer model. By incorporating regularizations on equilibrium data, we solved the problem of unstable MD simulations in vanilla Equivariant Transformers, achieving state-of-the-art simulation performance with 2.45 times faster inference time than NequIP.  As a powerful molecular encoder, our pre-trained model achieves on-par performance with state-of-the-art property prediction tasks.

----

## [3180] Deep Fractional Fourier Transform

**Authors**: *Hu Yu, Jie Huang, Lingzhi Li, Man Zhou, Feng Zhao*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e66309ead63bc1410d2df261a28f602d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e66309ead63bc1410d2df261a28f602d-Abstract-Conference.html)

**Abstract**:

Existing deep learning-based computer vision methods usually operate in the spatial and frequency domains, which are two orthogonal \textbf{individual} perspectives for image processing.In this paper, we introduce a new spatial-frequency analysis tool, Fractional Fourier Transform (FRFT), to provide comprehensive \textbf{unified} spatial-frequency perspectives.The FRFT is a unified continuous spatial-frequency transform that simultaneously reflects an image's spatial and frequency representations, making it optimal for processing non-stationary image signals.We explore the properties of the FRFT for image processing and present a fast implementation of the 2D FRFT, which facilitates its widespread use.Based on these explorations, we introduce a simple yet effective operator, Multi-order FRactional Fourier Convolution (MFRFC), which exhibits the remarkable merits of processing images from more perspectives in the spatial-frequency plane. Our proposed MFRFC is a general and basic operator that can be easily integrated into various tasks for performance improvement.We experimentally evaluate the MFRFC on various computer vision tasks, including object detection, image classification, guided super-resolution, denoising, dehazing, deraining, and low-light enhancement. Our proposed MFRFC consistently outperforms baseline methods by significant margins across all tasks.

----

## [3181] Survival Permanental Processes for Survival Analysis with Time-Varying Covariates

**Authors**: *Hideaki Kim*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e664650506f1cf2b4696df892147c06e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e664650506f1cf2b4696df892147c06e-Abstract-Conference.html)

**Abstract**:

Survival or time-to-event data with time-varying covariates are common in practice, and exploring the non-stationarity in covariates is essential to accurately analyzing the nonlinear dependence of time-to-event outcomes on covariates. Traditional survival analysis methods such as Cox proportional hazards model have been extended to address the time-varying covariates through a counting process formulation, although sophisticated machine learning methods that can accommodate time-varying covariates have been limited. In this paper, we propose a non-parametric Bayesian survival model to analyze the nonlinear dependence of time-to-event outcomes on time-varying covariates. We focus on a computationally feasible Cox process called permanental process, which assumes the square root of hazard function to be generated from a Gaussian process, and tailor it for survival data with time-varying covariates. We verify that the proposed model holds with the representer theorem, a beneficial property for functional analysis, which offers us a fast Bayesian estimation algorithm that scales linearly with the number of observed events without relying on Markov Chain Monte Carlo computation. We evaluate our algorithm on synthetic and real-world data, and show that it achieves comparable predictive accuracy while being tens to hundreds of times faster than state-of-the-art methods.

----

## [3182] Learn to Categorize or Categorize to Learn? Self-Coding for Generalized Category Discovery

**Authors**: *Sarah Rastegar, Hazel Doughty, Cees Snoek*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e6789e468c65a7816760a00a487d3c4e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e6789e468c65a7816760a00a487d3c4e-Abstract-Conference.html)

**Abstract**:

In the quest for unveiling novel categories at test time, we confront the inherent limitations of traditional supervised recognition models that are restricted by a predefined category set. While strides have been made in the realms of self-supervised and open-world learning towards test-time category discovery, a crucial yet often overlooked question persists: what exactly delineates a category? In this paper, we conceptualize a category through the lens of optimization, viewing it as an optimal solution to a well-defined problem. Harnessing this unique conceptualization, we propose a novel, efficient and self-supervised method capable of discovering previously unknown categories at test time. A salient feature of our approach is the assignment of minimum length category codes to individual data instances, which encapsulates the implicit category hierarchy prevalent in real-world datasets. This mechanism affords us enhanced control over category granularity, thereby equipping our model to handle fine-grained categories adeptly. Experimental evaluations, bolstered by state-of-the-art benchmark comparisons, testify to the efficacy of our solution in managing unknown categories at test time. Furthermore, we fortify our proposition with a theoretical foundation, providing proof of its optimality. Our code is available at: https://github.com/SarahRastegar/InfoSieve.

----

## [3183] Training Chain-of-Thought via Latent-Variable Inference

**Authors**: *Matthew Douglas Hoffman, Du Phan, David Dohan, Sholto Douglas, Tuan Anh Le, Aaron Parisi, Pavel Sountsov, Charles Sutton, Sharad Vikram, Rif A. Saurous*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e69a9560c450ca76584d9eb37e7f5ae8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e69a9560c450ca76584d9eb37e7f5ae8-Abstract-Conference.html)

**Abstract**:

Large language models (LLMs) solve problems more accurately and interpretably when instructed to work out the answer step by step using a "chain-of-thought" (CoT) prompt. One can also improve LLMs' performance on a specific task by supervised fine-tuning, i.e., by using gradient ascent on some tunable parameters to maximize the average log-likelihood of correct answers from a labeled training set. Naively combining CoT with supervised tuning requires supervision not just of the correct answers, but also of detailed rationales that lead to those answers; these rationales are expensive to produce by hand. Instead, we propose a fine-tuning strategy that tries to maximize the \emph{marginal} log-likelihood of generating a correct answer using CoT prompting, approximately averaging over all possible rationales. The core challenge is sampling from the posterior over rationales conditioned on the correct answer; we address it using a simple Markov-chain Monte Carlo (MCMC) expectation-maximization (EM) algorithm inspired by the self-taught reasoner (STaR), memoized wake-sleep, Markovian score climbing, and persistent contrastive divergence. This algorithm also admits a novel control-variate technique that drives the variance of our gradient estimates to zero as the model improves. Applying our technique to GSM8K and the tasks in BIG-Bench Hard, we find that this MCMC-EM fine-tuning technique typically improves the model's accuracy on held-out examples more than STaR or prompt-tuning with or without CoT.

----

## [3184] VAST: A Vision-Audio-Subtitle-Text Omni-Modality Foundation Model and Dataset

**Authors**: *Sihan Chen, Handong Li, Qunbo Wang, Zijia Zhao, Mingzhen Sun, Xinxin Zhu, Jing Liu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e6b2b48b5ed90d07c305932729927781-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e6b2b48b5ed90d07c305932729927781-Abstract-Conference.html)

**Abstract**:

Vision and text have been fully explored  in contemporary video-text foundational models, while other modalities such as audio and subtitles in videos have not received sufficient attention. In this paper, we resort to establish connections between multi-modality video tracks, including Vision, Audio, and Subtitle, and Text by exploring an automatically generated large-scale omni-modality video caption dataset called VAST-27M. Specifically, we first collect 27 million open-domain video clips and separately train a vision and an audio captioner to generate vision and audio captions. Then, we employ an off-the-shelf Large Language Model (LLM) to integrate the generated captions, together with subtitles and instructional prompts into omni-modality captions. Based on the proposed VAST-27M dataset, we train an omni-modality video-text foundational model named VAST, which can perceive and process vision, audio, and subtitle modalities from video, and better support various tasks including  vision-text, audio-text, and multi-modal video-text tasks (retrieval, captioning and QA). Extensive experiments have been conducted to demonstrate the effectiveness of our proposed VAST-27M corpus and VAST foundation model. VAST achieves 22 new state-of-the-art results on various cross-modality benchmarks.

----

## [3185] Bayesian Learning via Q-Exponential Process

**Authors**: *Shuyi Li, Michael O'Connor, Shiwei Lan*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e6bfdd58f1326ff821a1b92743963bdf-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e6bfdd58f1326ff821a1b92743963bdf-Abstract-Conference.html)

**Abstract**:

Regularization is one of the most fundamental topics in optimization, statistics and machine learning. To get sparsity in estimating a parameter $u\in\mathbb{R}^d$, an $\ell_q$ penalty term, $\Vert u\Vert_q$, is usually added to the objective function. What is the probabilistic distribution corresponding to such $\ell_q$ penalty? What is the \emph{correct} stochastic process corresponding to $\Vert u\Vert_q$ when we model functions $u\in L^q$? This is important for statistically modeling high-dimensional objects such as images, with penalty to preserve certainty properties, e.g. edges in the image.In this work, we generalize the $q$-exponential distribution (with density proportional to) $\exp{(- \frac{1}{2}|u|^q)}$ to a stochastic process named \emph{$Q$-exponential (Q-EP) process} that corresponds to the $L_q$ regularization of functions. The key step is to specify consistent multivariate $q$-exponential distributions by choosing from a large family of elliptic contour distributions. The work is closely related to Besov process which is usually defined in terms of series. Q-EP can be regarded as a definition of Besov process with explicit probabilistic formulation, direct control on the correlation strength, and tractable prediction formula. From the Bayesian perspective, Q-EP provides a flexible prior on functions with sharper penalty ($q<2$) than the commonly used Gaussian process (GP, $q=2$).We compare GP, Besov and Q-EP in modeling functional data, reconstructing images and solving inverse problems and demonstrate the advantage of our proposed methodology.

----

## [3186] Repetition In Repetition Out: Towards Understanding Neural Text Degeneration from the Data Perspective

**Authors**: *Huayang Li, Tian Lan, Zihao Fu, Deng Cai, Lemao Liu, Nigel Collier, Taro Watanabe, Yixuan Su*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e6c2e85db1f1039177c4495ccd399ac4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e6c2e85db1f1039177c4495ccd399ac4-Abstract-Conference.html)

**Abstract**:

There are a number of diverging hypotheses about the neural text degeneration problem, i.e., generating repetitive and dull loops, which makes this problem both interesting and confusing. In this work, we aim to advance our understanding by presenting a straightforward and fundamental explanation from the data perspective. Our preliminary investigation reveals a strong correlation between the degeneration issue and the presence of repetitions in training data. Subsequent experiments also demonstrate that by selectively dropping out the attention to repetitive words in training data, degeneration can be significantly minimized. Furthermore, our empirical analysis illustrates that prior works addressing the degeneration issue from various standpoints, such as the high-inflow words, the likelihood objective, and the self-reinforcement phenomenon, can be interpreted by one simple explanation. That is, penalizing the repetitions in training data is a common and fundamental factor for their effectiveness. Moreover, our experiments reveal that penalizing the repetitions in training data remains critical even when considering larger model sizes and instruction tuning.

----

## [3187] Facing Off World Model Backbones: RNNs, Transformers, and S4

**Authors**: *Fei Deng, Junyeong Park, Sungjin Ahn*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e6c65eb9b56719c1aa45ff73874de317-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e6c65eb9b56719c1aa45ff73874de317-Abstract-Conference.html)

**Abstract**:

World models are a fundamental component in model-based reinforcement learning (MBRL). To perform temporally extended and consistent simulations of the future in partially observable environments, world models need to possess long-term memory. However, state-of-the-art MBRL agents, such as Dreamer, predominantly employ recurrent neural networks (RNNs) as their world model backbone, which have limited memory capacity. In this paper, we seek to explore alternative world model backbones for improving long-term memory. In particular, we investigate the effectiveness of Transformers and Structured State Space Sequence (S4) models, motivated by their remarkable ability to capture long-range dependencies in low-dimensional sequences and their complementary strengths. We propose S4WM, the first world model compatible with parallelizable SSMs including S4 and its variants. By incorporating latent variable modeling, S4WM can efficiently generate high-dimensional image sequences through latent imagination. Furthermore, we extensively compare RNN-, Transformer-, and S4-based world models across four sets of environments, which we have tailored to assess crucial memory capabilities of world models, including long-term imagination, context-dependent recall, reward prediction, and memory-based reasoning. Our findings demonstrate that S4WM outperforms Transformer-based world models in terms of long-term memory, while exhibiting greater efficiency during training and imagination. These results pave the way for the development of stronger MBRL agents.

----

## [3188] STARSS23: An Audio-Visual Dataset of Spatial Recordings of Real Scenes with Spatiotemporal Annotations of Sound Events

**Authors**: *Kazuki Shimada, Archontis Politis, Parthasaarathy Sudarsanam, Daniel Aleksander Krause, Kengo Uchida, Sharath Adavanne, Aapo Hakala, Yuichiro Koyama, Naoya Takahashi, Shusuke Takahashi, Tuomas Virtanen, Yuki Mitsufuji*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e6c9671ed3b3106b71cafda3ba225c1a-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/e6c9671ed3b3106b71cafda3ba225c1a-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

While direction of arrival (DOA) of sound events is generally estimated from multichannel audio data recorded in a microphone array, sound events usually derive from visually perceptible source objects, e.g., sounds of footsteps come from the feet of a walker. This paper proposes an audio-visual sound event localization and detection (SELD) task, which uses multichannel audio and video information to estimate the temporal activation and DOA of target sound events. Audio-visual SELD systems can detect and localize sound events using signals from a microphone array and audio-visual correspondence. We also introduce an audio-visual dataset, Sony-TAu Realistic Spatial Soundscapes 2023 (STARSS23), which consists of multichannel audio data recorded with a microphone array, video data, and spatiotemporal annotation of sound events. Sound scenes in STARSS23 are recorded with instructions, which guide recording participants to ensure adequate activity and occurrences of sound events. STARSS23 also serves human-annotated temporal activation labels and human-confirmed DOA labels, which are based on tracking results of a motion capture system. Our benchmark results demonstrate the benefits of using visual object positions in audio-visual SELD tasks. The data is available at https://zenodo.org/record/7880637.

----

## [3189] Inserting Anybody in Diffusion Models via Celeb Basis

**Authors**: *Ge Yuan, Xiaodong Cun, Yong Zhang, Maomao Li, Chenyang Qi, Xintao Wang, Ying Shan, Huicheng Zheng*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e6d37cc5723e810b793c834bcb6647cf-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e6d37cc5723e810b793c834bcb6647cf-Abstract-Conference.html)

**Abstract**:

Exquisite demand exists for customizing the pretrained large text-to-image model, $e.g.$ Stable Diffusion, to generate innovative concepts, such as the users themselves. However, the newly-added concept from previous customization methods often shows weaker combination abilities than the original ones even given several images during training. We thus propose a new personalization method that allows for the seamless integration of a unique individual into the pre-trained diffusion model using just $one\ facial\ photograph$ and only $1024\ learnable\ parameters$ under $3\ minutes$. So we can effortlessly generate stunning images of this person in any pose or position, interacting with anyone and doing anything imaginable from text prompts. To achieve this, we first analyze and build a well-defined celeb basis from the embedding space of the pre-trained large text encoder. Then, given one facial photo as the target identity, we generate its own embedding by optimizing the weight of this basis and locking all other parameters. Empowered by the proposed celeb basis, the new identity in our customized model showcases a better concept combination ability than previous personalization methods. Besides, our model can also learn several new identities at once and interact with each other where the previous customization model fails to. Project page is at: http://celeb-basis.github.io. Code is at: https://github.com/ygtxr1997/CelebBasis.

----

## [3190] Scaling Open-Vocabulary Object Detection

**Authors**: *Matthias Minderer, Alexey A. Gritsenko, Neil Houlsby*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e6d58fc68c0f3c36ae6e0e64478a69c0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e6d58fc68c0f3c36ae6e0e64478a69c0-Abstract-Conference.html)

**Abstract**:

Open-vocabulary object detection has benefited greatly from pretrained vision-language models, but is still limited by the amount of available detection training data. While detection training data can be expanded by using Web image-text pairs as weak supervision, this has not been done at scales comparable to image-level pretraining. Here, we scale up detection data with self-training, which uses an existing detector to generate pseudo-box annotations on image-text pairs. Major challenges in scaling self-training are the choice of label space, pseudo-annotation filtering, and training efficiency. We present the OWLv2 model and OWL-ST self-training recipe, which address these challenges. OWLv2 surpasses the performance of previous state-of-the-art open-vocabulary detectors already at comparable training scales (~10M examples). However, with OWL-ST, we can scale to over 1B examples, yielding further large improvement: With an L/14 architecture, OWL-ST improves AP on LVIS rare classes, for which the model has seen no human box annotations, from 31.2% to 44.6% (43% relative improvement). OWL-ST unlocks Web-scale training for open-world localization, similar to what has been seen for image classification and language modelling. Code and checkpoints are available on GitHub.

----

## [3191] Formulating Discrete Probability Flow Through Optimal Transport

**Authors**: *Pengze Zhang, Hubery Yin, Chen Li, Xiaohua Xie*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e6e706454d72c18582b9c1ff70b11f7d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e6e706454d72c18582b9c1ff70b11f7d-Abstract-Conference.html)

**Abstract**:

Continuous diffusion models are commonly acknowledged to display a deterministic probability flow, whereas discrete diffusion models do not. In this paper, we aim to establish the fundamental theory for the probability flow of discrete diffusion models. Specifically, we first prove that the continuous probability flow is the Monge optimal transport map under certain conditions, and also present an equivalent evidence for discrete cases.  In view of these findings, we are then able to define the discrete probability flow in line with the principles of optimal transport. Finally, drawing upon our newly established definitions, we propose a novel sampling method that surpasses previous discrete diffusion models in its ability to generate more certain outcomes. Extensive experiments on the synthetic toy dataset and the CIFAR-10 dataset have validated the effectiveness of our proposed discrete probability flow. Code is released at: https://github.com/PangzeCheung/Discrete-Probability-Flow.

----

## [3192] Successor-Predecessor Intrinsic Exploration

**Authors**: *Changmin Yu, Neil Burgess, Maneesh Sahani, Samuel J. Gershman*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e6f2b968c4ee8ba260cd7077e39590dd-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e6f2b968c4ee8ba260cd7077e39590dd-Abstract-Conference.html)

**Abstract**:

Exploration is essential in reinforcement learning, particularly in environments where external rewards are sparse. Here we focus on exploration with intrinsic rewards, where the agent transiently augments the external rewards with self-generated intrinsic rewards. Although the study of intrinsic rewards has a long history, existing methods focus on composing the intrinsic reward based on measures of future prospects of states, ignoring the information contained in the retrospective structure of transition sequences. Here we argue that the agent can utilise retrospective information to generate explorative behaviour with structure-awareness, facilitating efficient exploration based on global instead of local information. We propose Successor-Predecessor Intrinsic Exploration (SPIE), an exploration algorithm based on a novel intrinsic reward combining prospective and retrospective information. We show that SPIE yields more efficient and ethologically plausible exploratory behaviour in environments with sparse rewards and bottleneck states  than competing methods. We also implement SPIE in deep reinforcement learning agents, and show that the resulting agent achieves stronger empirical performance than existing methods on sparse-reward Atari games.

----

## [3193] TFLEX: Temporal Feature-Logic Embedding Framework for Complex Reasoning over Temporal Knowledge Graph

**Authors**: *Xueyuan Lin, Haihong E, Chengjin Xu, Gengxian Zhou, Haoran Luo, Tianyi Hu, Fenglong Su, Ningyuan Li, Mingzhi Sun*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e71a42c64851834013e2658b69d7fe93-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e71a42c64851834013e2658b69d7fe93-Abstract-Conference.html)

**Abstract**:

Multi-hop logical reasoning over knowledge graph plays a fundamental role in many artificial intelligence tasks.  Recent complex query embedding methods for reasoning focus on static KGs, while temporal knowledge graphs have not been fully explored.  Reasoning over TKGs has two challenges: 1. The query should answer entities or timestamps; 2. The operators should consider both set logic on entity set and temporal logic on timestamp set.To bridge this gap, we introduce the multi-hop logical reasoning problem on TKGs and then propose the first temporal complex query embedding named Temporal Feature-Logic Embedding framework (TFLEX) to answer the temporal complex queries.  Specifically, we utilize fuzzy logic to compute the logic part of the Temporal Feature-Logic embedding, thus naturally modeling all first-order logic operations on the entity set.  In addition, we further extend fuzzy logic on timestamp set to cope with three extra temporal operators (After, Before and Between).Experiments on numerous query patterns demonstrate the effectiveness of our method.

----

## [3194] StyleGAN knows Normal, Depth, Albedo, and More

**Authors**: *Anand Bhattad, Daniel McKee, Derek Hoiem, David A. Forsyth*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e7407ab5e89c405d28ff6807ffec594a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e7407ab5e89c405d28ff6807ffec594a-Abstract-Conference.html)

**Abstract**:

Intrinsic images, in the original sense, are image-like maps of scene properties like depth, normal, albedo, or shading. This paper demonstrates that StyleGAN can easily be induced to produce intrinsic images.  The procedure is straightforward. We show that if StyleGAN produces $G({\bf w})$ from latent ${\bf w}$, then for each type of intrinsic image, there is a fixed offset ${\bf d}_c$ so that $G({\bf w}+{\bf d}_c)$ is that type of intrinsic image for $G({\bf w})$. Here ${\bf d}_c$ is {\em independent of ${\bf w}$}.  The StyleGAN we used was pretrained by others, so this property is not some accident of our training regime.  We show that there are image transformations StyleGAN will {\em not} produce in this fashion, so StyleGAN is not a generic image regression engine.  It is conceptually exciting that an image generator should ``know'' and represent intrinsic images. There may also be practical advantages to using a generative model to produce intrinsic images. The intrinsic images obtained from StyleGAN compare well both qualitatively and quantitatively with those obtained by using SOTA image regression techniques; but StyleGAN's intrinsic images are robust to relighting effects, unlike SOTA methods.

----

## [3195] Are Vision Transformers More Data Hungry Than Newborn Visual Systems?

**Authors**: *Lalit Pandey, Samantha M. W. Wood, Justin N. Wood*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e75dce944052276caf89c17aca8963d3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e75dce944052276caf89c17aca8963d3-Abstract-Conference.html)

**Abstract**:

Vision transformers (ViTs) are top-performing models on many computer vision benchmarks and can accurately predict human behavior on object recognition tasks. However, researchers question the value of using ViTs as models of biological learning because ViTs are thought to be more “data hungry” than brains, with ViTs requiring more training data than brains to reach similar levels of performance. To test this assumption, we directly compared the learning abilities of ViTs and animals, by performing parallel controlled-rearing experiments on ViTs and newborn chicks. We first raised chicks in impoverished visual environments containing a single object, then simulated the training data available in those environments by building virtual animal chambers in a video game engine. We recorded the first-person images acquired by agents moving through the virtual chambers and used those images to train self-supervised ViTs that leverage time as a teaching signal, akin to biological visual systems. When ViTs were trained “through the eyes” of newborn chicks, the ViTs solved the same view-invariant object recognition tasks as the chicks. Thus, ViTs were not more data hungry than newborn chicks: both learned view-invariant object representations in impoverished visual environments. The flexible and generic attention-based learning mechanism in ViTs—combined with the embodied data streams available to newborn animals—appears sufficient to drive the development of animal-like object recognition.

----

## [3196] How to Scale Your EMA

**Authors**: *Dan Busbridge, Jason Ramapuram, Pierre Ablin, Tatiana Likhomanenko, Eeshan Gunesh Dhekane, Xavier Suau Cuadros, Russell Webb*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e7681dd6fe16052433ab68cd1555bdc9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e7681dd6fe16052433ab68cd1555bdc9-Abstract-Conference.html)

**Abstract**:

Preserving training dynamics across batch sizes is an important tool for practical machine learning as it enables the trade-off between batch size and wall-clock time. This trade-off is typically enabled by a scaling rule, for example, in stochastic gradient descent, one should scale the learning rate linearly with the batch size. Another important machine learning tool is the model EMA, a functional copy of a target model, whose parameters move towards those of its target model according to an Exponential Moving Average (EMA) at a rate parameterized by a momentum hyperparameter. This model EMA can improve the robustness and generalization of supervised learning, stabilize pseudo-labeling, and provide a learning signal for Self-Supervised Learning (SSL). Prior works have not considered the optimization of the model EMA when performing scaling, leading to different training dynamics across batch sizes and lower model performance. In this work, we provide a scaling rule for optimization in the presence of a model EMA and demonstrate the rule's validity across a range of architectures, optimizers, and data modalities.  We also show the rule's validity where the model EMA contributes to the optimization of the target model, enabling us to train EMA-based pseudo-labeling and SSL methods at small and large batch sizes. For SSL, we enable training of BYOL up to batch size 24,576 without sacrificing performance, a 6$\times$ wall-clock time reduction under idealized hardware settings.

----

## [3197] Unsupervised Graph Neural Architecture Search with Disentangled Self-Supervision

**Authors**: *Zeyang Zhang, Xin Wang, Ziwei Zhang, Guangyao Shen, Shiqi Shen, Wenwu Zhu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e78399fc43dbb2d87b7e1e6906ce5baf-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e78399fc43dbb2d87b7e1e6906ce5baf-Abstract-Conference.html)

**Abstract**:

The existing graph neural architecture search (GNAS) methods heavily rely on supervised labels during the search process, failing to handle ubiquitous scenarios where supervisions are not available. In this paper, we study the problem of unsupervised graph neural architecture search, which remains unexplored in the literature. The key problem is to discover the latent graph factors that drive the formation of graph data as well as the underlying relations between the factors and the optimal neural architectures. Handling this problem is challenging given that the latent graph factors together with architectures are highly entangled due to the nature of the graph and the complexity of the neural architecture search process. To address the challenge, we propose a novel Disentangled Self-supervised Graph Neural Architecture Search (DSGAS) model, which is able to discover the optimal architectures capturing various latent graph factors in a self-supervised fashion based on unlabeled graph data. Specifically, we first design a disentangled graph super-network capable of incorporating multiple architectures with factor-wise disentanglement, which are optimized simultaneously. Then, we estimate the performance of architectures under different factors by our proposed self-supervised training with joint architecture-graph disentanglement. Finally, we propose a contrastive search with architecture augmentations to discover architectures with factor-specific expertise. Extensive experiments on 11 real-world datasets demonstrate that the proposed model is able to achieve state-of-the-art performance against several baseline methods in an unsupervised manner.

----

## [3198] Setting the Trap: Capturing and Defeating Backdoors in Pretrained Language Models through Honeypots

**Authors**: *Ruixiang (Ryan) Tang, Jiayi Yuan, Yiming Li, Zirui Liu, Rui Chen, Xia Hu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e7938ede51225b490bb69f7b361a9259-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e7938ede51225b490bb69f7b361a9259-Abstract-Conference.html)

**Abstract**:

In the field of natural language processing, the prevalent approach involves fine-tuning pretrained language models (PLMs) using local samples. Recent research has exposed the susceptibility of PLMs to backdoor attacks, wherein the adversaries can embed malicious prediction behaviors by manipulating a few training samples. In this study, our objective is to develop a backdoor-resistant tuning procedure that yields a backdoor-free model, no matter whether the fine-tuning dataset contains poisoned samples. To this end, we propose and integrate an \emph{honeypot module} into the original PLM, specifically designed to absorb backdoor information exclusively. Our design is motivated by the observation that lower-layer representations in PLMs carry sufficient backdoor features while carrying minimal information about the original tasks. Consequently, we can impose penalties on the information acquired by the honeypot module to inhibit backdoor creation during the fine-tuning process of the stem network. Comprehensive experiments conducted on benchmark datasets substantiate the effectiveness and robustness of our defensive strategy. Notably, these results indicate a substantial reduction in the attack success rate ranging from 10\% to 40\% when compared to prior state-of-the-art methods.

----

## [3199] Learning Large-Scale MTP2 Gaussian Graphical Models via Bridge-Block Decomposition

**Authors**: *Xiwen Wang, Jiaxi Ying, Daniel P. Palomar*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/e7e506bc5a94768243083216fe51d98b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/e7e506bc5a94768243083216fe51d98b-Abstract-Conference.html)

**Abstract**:

This paper studies the problem of learning the large-scale Gaussian graphical models that are multivariate totally positive of order two ($\text{MTP}_2$). By introducing the concept of bridge, which commonly exists in large-scale sparse graphs, we show that the entire problem can be equivalently optimized through (1) several smaller-scaled sub-problems induced by a \emph{bridge-block decomposition} on the thresholded sample covariance graph and (2) a set of explicit solutions on entries corresponding to  \emph{bridges}. From practical aspect, this simple and provable discipline can be applied to break down a large problem into small tractable ones, leading to enormous reduction on the computational complexity and substantial improvements for all existing algorithms.  The synthetic and real-world experiments demonstrate that our proposed method presents a significant speed-up compared to the state-of-the-art benchmarks.

----



[Go to the previous page](NIPS-2023-list15.md)

[Go to the next page](NIPS-2023-list17.md)

[Go to the catalog section](README.md)