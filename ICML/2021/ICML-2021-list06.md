## [1000] Bridging Multi-Task Learning and Meta-Learning: Towards Efficient Training and Effective Adaptation

**Authors**: *Haoxiang Wang, Han Zhao, Bo Li*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/wang21ad.html](http://proceedings.mlr.press/v139/wang21ad.html)

**Abstract**:

Multi-task learning (MTL) aims to improve the generalization of several related tasks by learning them jointly. As a comparison, in addition to the joint training scheme, modern meta-learning allows unseen tasks with limited labels during the test phase, in the hope of fast adaptation over them. Despite the subtle difference between MTL and meta-learning in the problem formulation, both learning paradigms share the same insight that the shared structure between existing training tasks could lead to better generalization and adaptation. In this paper, we take one important step further to understand the close connection between these two learning paradigms, through both theoretical analysis and empirical investigation. Theoretically, we first demonstrate that MTL shares the same optimization formulation with a class of gradient-based meta-learning (GBML) algorithms. We then prove that for over-parameterized neural networks with sufficient depth, the learned predictive functions of MTL and GBML are close. In particular, this result implies that the predictions given by these two models are similar over the same unseen task. Empirically, we corroborate our theoretical findings by showing that, with proper implementation, MTL is competitive against state-of-the-art GBML algorithms on a set of few-shot image classification benchmarks. Since existing GBML algorithms often involve costly second-order bi-level optimization, our first-order MTL method is an order of magnitude faster on large-scale datasets such as mini-ImageNet. We believe this work could help bridge the gap between these two learning paradigms, and provide a computationally efficient alternative to GBML that also supports fast task adaptation.

----

## [1001] Towards Better Laplacian Representation in Reinforcement Learning with Generalized Graph Drawing

**Authors**: *Kaixin Wang, Kuangqi Zhou, Qixin Zhang, Jie Shao, Bryan Hooi, Jiashi Feng*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/wang21ae.html](http://proceedings.mlr.press/v139/wang21ae.html)

**Abstract**:

The Laplacian representation recently gains increasing attention for reinforcement learning as it provides succinct and informative representation for states, by taking the eigenvectors of the Laplacian matrix of the state-transition graph as state embeddings. Such representation captures the geometry of the underlying state space and is beneficial to RL tasks such as option discovery and reward shaping. To approximate the Laplacian representation in large (or even continuous) state spaces, recent works propose to minimize a spectral graph drawing objective, which however has infinitely many global minimizers other than the eigenvectors. As a result, their learned Laplacian representation may differ from the ground truth. To solve this problem, we reformulate the graph drawing objective into a generalized form and derive a new learning objective, which is proved to have eigenvectors as its unique global minimizer. It enables learning high-quality Laplacian representations that faithfully approximate the ground truth. We validate this via comprehensive experiments on a set of gridworld and continuous control environments. Moreover, we show that our learned Laplacian representations lead to more exploratory options and better reward shaping.

----

## [1002] Robust Asymmetric Learning in POMDPs

**Authors**: *Andrew Warrington, Jonathan Wilder Lavington, Adam Scibior, Mark Schmidt, Frank Wood*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/warrington21a.html](http://proceedings.mlr.press/v139/warrington21a.html)

**Abstract**:

Policies for partially observed Markov decision processes can be efficiently learned by imitating expert policies generated using asymmetric information. Unfortunately, existing approaches for this kind of imitation learning have a serious flaw: the expert does not know what the trainee cannot see, and as a result may encourage actions that are sub-optimal or unsafe under partial information. To address this issue, we derive an update which, when applied iteratively to an expert, maximizes the expected reward of the trainee’s policy. Using this update, we construct a computationally efficient algorithm, adaptive asymmetric DAgger (A2D), that jointly trains the expert and trainee policies. We then show that A2D allows the trainee to safely imitate the modified expert, and outperforms policies learned either by imitating a fixed expert or through direct reinforcement learning.

----

## [1003] A Unified Generative Adversarial Network Training via Self-Labeling and Self-Attention

**Authors**: *Tomoki Watanabe, Paolo Favaro*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/watanabe21a.html](http://proceedings.mlr.press/v139/watanabe21a.html)

**Abstract**:

We propose a novel GAN training scheme that can handle any level of labeling in a unified manner. Our scheme introduces a form of artificial labeling that can incorporate manually defined labels, when available, and induce an alignment between them. To define the artificial labels, we exploit the assumption that neural network generators can be trained more easily to map nearby latent vectors to data with semantic similarities, than across separate categories. We use generated data samples and their corresponding artificial conditioning labels to train a classifier. The classifier is then used to self-label real data. To boost the accuracy of the self-labeling, we also use the exponential moving average of the classifier. However, because the classifier might still make mistakes, especially at the beginning of the training, we also refine the labels through self-attention, by using the labeling of real data samples only when the classifier outputs a high classification probability score. We evaluate our approach on CIFAR-10, STL-10 and SVHN, and show that both self-labeling and self-attention consistently improve the quality of generated data. More surprisingly, we find that the proposed scheme can even outperform class-conditional GANs.

----

## [1004] Decision-Making Under Selective Labels: Optimal Finite-Domain Policies and Beyond

**Authors**: *Dennis Wei*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/wei21a.html](http://proceedings.mlr.press/v139/wei21a.html)

**Abstract**:

Selective labels are a common feature of high-stakes decision-making applications, referring to the lack of observed outcomes under one of the possible decisions. This paper studies the learning of decision policies in the face of selective labels, in an online setting that balances learning costs against future utility. In the homogeneous case in which individuals’ features are disregarded, the optimal decision policy is shown to be a threshold policy. The threshold becomes more stringent as more labels are collected; the rate at which this occurs is characterized. In the case of features drawn from a finite domain, the optimal policy consists of multiple homogeneous policies in parallel. For the general infinite-domain case, the homogeneous policy is extended by using a probabilistic classifier and bootstrapping to provide its inputs. In experiments on synthetic and real data, the proposed policies achieve consistently superior utility with no parameter tuning in the finite-domain case and lower parameter sensitivity in the general case.

----

## [1005] Inferring serial correlation with dynamic backgrounds

**Authors**: *Song Wei, Yao Xie, Dobromir Rahnev*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/wei21b.html](http://proceedings.mlr.press/v139/wei21b.html)

**Abstract**:

Sequential data with serial correlation and an unknown, unstructured, and dynamic background is ubiquitous in neuroscience, psychology, and econometrics. Inferring serial correlation for such data is a fundamental challenge in statistics. We propose a Total Variation (TV) constrained least square estimator coupled with hypothesis tests to infer the serial correlation in the presence of unknown and unstructured dynamic background. The TV constraint on the dynamic background encourages a piecewise constant structure, which can approximate a wide range of dynamic backgrounds. The tuning parameter is selected via the Ljung-Box test to control the bias-variance trade-off. We establish a non-asymptotic upper bound for the estimation error through variational inequalities. We also derive a lower error bound via Fano’s method and show the proposed method is near-optimal. Numerical simulation and a real study in psychology demonstrate the excellent performance of our proposed method compared with the state-of-the-art.

----

## [1006] Meta-learning Hyperparameter Performance Prediction with Neural Processes

**Authors**: *Ying Wei, Peilin Zhao, Junzhou Huang*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/wei21c.html](http://proceedings.mlr.press/v139/wei21c.html)

**Abstract**:

The surrogate that predicts the performance of hyperparameters has been a key component for sequential model-based hyperparameter optimization. In practical applications, a trial of a hyper-parameter configuration may be so costly that a surrogate is expected to return an optimal configuration with as few trials as possible. Observing that human experts draw on their expertise in a machine learning model by trying configurations that once performed well on other datasets, we are inspired to build a trial-efficient surrogate by transferring the meta-knowledge learned from historical trials on other datasets. We propose an end-to-end surrogate named as Transfer NeuralProcesses (TNP) that learns a comprehensive set of meta-knowledge, including the parameters of historical surrogates, historical trials, and initial configurations for other datasets. Experiments on extensive OpenML datasets and three computer vision datasets demonstrate that the proposed algorithm achieves state-of-the-art performance in at least one order of magnitude less trials.

----

## [1007] A Structured Observation Distribution for Generative Biological Sequence Prediction and Forecasting

**Authors**: *Eli N. Weinstein, Debora S. Marks*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/weinstein21a.html](http://proceedings.mlr.press/v139/weinstein21a.html)

**Abstract**:

Generative probabilistic modeling of biological sequences has widespread existing and potential application across biology and biomedicine, from evolutionary biology to epidemiology to protein design. Many standard sequence analysis methods preprocess data using a multiple sequence alignment (MSA) algorithm, one of the most widely used computational methods in all of science. However, as we show in this article, training generative probabilistic models with MSA preprocessing leads to statistical pathologies in the context of sequence prediction and forecasting. To address these problems, we propose a principled drop-in alternative to MSA preprocessing in the form of a structured observation distribution (the "MuE" distribution). We prove theoretically that the MuE distribution comprehensively generalizes popular methods for inferring biological sequence alignments, and provide a precise characterization of how such biological models have differed from natural language latent alignment models. We show empirically that models that use the MuE as an observation distribution outperform comparable methods across a variety of datasets, and apply MuE models to a novel problem for generative probabilistic sequence models: forecasting pathogen evolution.

----

## [1008] Thinking Like Transformers

**Authors**: *Gail Weiss, Yoav Goldberg, Eran Yahav*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/weiss21a.html](http://proceedings.mlr.press/v139/weiss21a.html)

**Abstract**:

What is the computational model behind a Transformer? Where recurrent neural networks have direct parallels in finite state machines, allowing clear discussion and thought around architecture variants or trained models, Transformers have no such familiar parallel. In this paper we aim to change that, proposing a computational model for the transformer-encoder in the form of a programming language. We map the basic components of a transformer-encoder—attention and feed-forward computation—into simple primitives, around which we form a programming language: the Restricted Access Sequence Processing Language (RASP). We show how RASP can be used to program solutions to tasks that could conceivably be learned by a Transformer, and how a Transformer can be trained to mimic a RASP solution. In particular, we provide RASP programs for histograms, sorting, and Dyck-languages. We further use our model to relate their difficulty in terms of the number of required layers and attention heads: analyzing a RASP program implies a maximum number of heads and layers necessary to encode a task in a transformer. Finally, we see how insights gained from our abstraction might be used to explain phenomena seen in recent works.

----

## [1009] Leveraged Weighted Loss for Partial Label Learning

**Authors**: *Hongwei Wen, Jingyi Cui, Hanyuan Hang, Jiabin Liu, Yisen Wang, Zhouchen Lin*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/wen21a.html](http://proceedings.mlr.press/v139/wen21a.html)

**Abstract**:

As an important branch of weakly supervised learning, partial label learning deals with data where each instance is assigned with a set of candidate labels, whereas only one of them is true. Despite many methodology studies on learning from partial labels, there still lacks theoretical understandings of their risk consistent properties under relatively weak assumptions, especially on the link between theoretical results and the empirical choice of parameters. In this paper, we propose a family of loss functions named \textit{Leveraged Weighted} (LW) loss, which for the first time introduces the leverage parameter $\beta$ to consider the trade-off between losses on partial labels and non-partial ones. From the theoretical side, we derive a generalized result of risk consistency for the LW loss in learning from partial labels, based on which we provide guidance to the choice of the leverage parameter $\beta$. In experiments, we verify the theoretical guidance, and show the high effectiveness of our proposed LW loss on both benchmark and real datasets compared with other state-of-the-art partial label learning algorithms.

----

## [1010] Characterizing the Gap Between Actor-Critic and Policy Gradient

**Authors**: *Junfeng Wen, Saurabh Kumar, Ramki Gummadi, Dale Schuurmans*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/wen21b.html](http://proceedings.mlr.press/v139/wen21b.html)

**Abstract**:

Actor-critic (AC) methods are ubiquitous in reinforcement learning. Although it is understood that AC methods are closely related to policy gradient (PG), their precise connection has not been fully characterized previously. In this paper, we explain the gap between AC and PG methods by identifying the exact adjustment to the AC objective/gradient that recovers the true policy gradient of the cumulative reward objective (PG). Furthermore, by viewing the AC method as a two-player Stackelberg game between the actor and critic, we show that the Stackelberg policy gradient can be recovered as a special case of our more general analysis. Based on these results, we develop practical algorithms, Residual Actor-Critic and Stackelberg Actor-Critic, for estimating the correction between AC and PG and use these to modify the standard AC algorithm. Experiments on popular tabular and continuous environments show the proposed corrections can improve both the sample efficiency and final performance of existing AC methods.

----

## [1011] Toward Understanding the Feature Learning Process of Self-supervised Contrastive Learning

**Authors**: *Zixin Wen, Yuanzhi Li*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/wen21c.html](http://proceedings.mlr.press/v139/wen21c.html)

**Abstract**:

We formally study how contrastive learning learns the feature representations for neural networks by investigating its feature learning process. We consider the case where our data are comprised of two types of features: the sparse features which we want to learn from, and the dense features we want to get rid of. Theoretically, we prove that contrastive learning using ReLU networks provably learns the desired features if proper augmentations are adopted. We present an underlying principle called feature decoupling to explain the effects of augmentations, where we theoretically characterize how augmentations can reduce the correlations of dense features between positive samples while keeping the correlations of sparse features intact, thereby forcing the neural networks to learn from the self-supervision of sparse features. Empirically, we verified that the feature decoupling principle matches the underlying mechanism of contrastive learning in practice.

----

## [1012] Keyframe-Focused Visual Imitation Learning

**Authors**: *Chuan Wen, Jierui Lin, Jianing Qian, Yang Gao, Dinesh Jayaraman*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/wen21d.html](http://proceedings.mlr.press/v139/wen21d.html)

**Abstract**:

Imitation learning trains control policies by mimicking pre-recorded expert demonstrations. In partially observable settings, imitation policies must rely on observation histories, but many seemingly paradoxical results show better performance for policies that only access the most recent observation. Recent solutions ranging from causal graph learning to deep information bottlenecks have shown promising results, but failed to scale to realistic settings such as visual imitation. We propose a solution that outperforms these prior approaches by upweighting demonstration keyframes corresponding to expert action changepoints. This simple approach easily scales to complex visual imitation settings. Our experimental results demonstrate consistent performance improvements over all baselines on image-based Gym MuJoCo continuous control tasks. Finally, on the CARLA photorealistic vision-based urban driving simulator, we resolve a long-standing issue in behavioral cloning for driving by demonstrating effective imitation from observation histories. Supplementary materials and code at: \url{https://tinyurl.com/imitation-keyframes}.

----

## [1013] Learning de-identified representations of prosody from raw audio

**Authors**: *Jack Weston, Raphael Lenain, Udeepa Meepegama, Emil Fristed*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/weston21a.html](http://proceedings.mlr.press/v139/weston21a.html)

**Abstract**:

We propose a method for learning de-identified prosody representations from raw audio using a contrastive self-supervised signal. Whereas prior work has relied on conditioning models with bottlenecks, we introduce a set of inductive biases that exploit the natural structure of prosody to minimize timbral information and decouple prosody from speaker representations. Despite aggressive downsampling of the input and having no access to linguistic information, our model performs comparably to state-of-the-art speech representations on DAMMP, a new benchmark we introduce for spoken language understanding. We use minimum description length probing to show that our representations have selectively learned the subcomponents of non-timbral prosody, and that the product quantizer naturally disentangles them without using bottlenecks. We derive an information-theoretic definition of speech de-identifiability and use it to demonstrate that our prosody representations are less identifiable than the other speech representations.

----

## [1014] Solving Inverse Problems with a Flow-based Noise Model

**Authors**: *Jay Whang, Qi Lei, Alex Dimakis*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/whang21a.html](http://proceedings.mlr.press/v139/whang21a.html)

**Abstract**:

We study image inverse problems with a normalizing flow prior. Our formulation views the solution as the maximum a posteriori estimate of the image conditioned on the measurements. This formulation allows us to use noise models with arbitrary dependencies as well as non-linear forward operators. We empirically validate the efficacy of our method on various inverse problems, including compressed sensing with quantized measurements and denoising with highly structured noise patterns. We also present initial theoretical recovery guarantees for solving inverse problems with a flow prior.

----

## [1015] Composing Normalizing Flows for Inverse Problems

**Authors**: *Jay Whang, Erik M. Lindgren, Alex Dimakis*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/whang21b.html](http://proceedings.mlr.press/v139/whang21b.html)

**Abstract**:

Given an inverse problem with a normalizing flow prior, we wish to estimate the distribution of the underlying signal conditioned on the observations. We approach this problem as a task of conditional inference on the pre-trained unconditional flow model. We first establish that this is computationally hard for a large class of flow models. Motivated by this, we propose a framework for approximate inference that estimates the target conditional as a composition of two flow models. This formulation leads to a stable variational inference training procedure that avoids adversarial training. Our method is evaluated on a variety of inverse problems and is shown to produce high-quality samples with uncertainty quantification. We further demonstrate that our approach can be amortized for zero-shot inference.

----

## [1016] Which transformer architecture fits my data? A vocabulary bottleneck in self-attention

**Authors**: *Noam Wies, Yoav Levine, Daniel Jannai, Amnon Shashua*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/wies21a.html](http://proceedings.mlr.press/v139/wies21a.html)

**Abstract**:

After their successful debut in natural language processing, Transformer architectures are now becoming the de-facto standard in many domains. An obstacle for their deployment over new modalities is the architectural configuration: the optimal depth-to-width ratio has been shown to dramatically vary across data types (i.e., 10x larger over images than over language). We theoretically predict the existence of an embedding rank bottleneck that limits the contribution of self-attention width to the Transformer expressivity. We thus directly tie the input vocabulary size and rank to the optimal depth-to-width ratio, since a small vocabulary size or rank dictates an added advantage of depth over width. We empirically demonstrate the existence of this bottleneck and its implications on the depth-to-width interplay of Transformer architectures, linking the architecture variability across domains to the often glossed-over usage of different vocabulary sizes or embedding ranks in different domains. As an additional benefit, our rank bottlenecking framework allows us to identify size redundancies of 25%-50% in leading NLP models such as ALBERT and T5.

----

## [1017] Prediction-Centric Learning of Independent Cascade Dynamics from Partial Observations

**Authors**: *Mateusz Wilinski, Andrey Y. Lokhov*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/wilinski21a.html](http://proceedings.mlr.press/v139/wilinski21a.html)

**Abstract**:

Spreading processes play an increasingly important role in modeling for diffusion networks, information propagation, marketing and opinion setting. We address the problem of learning of a spreading model such that the predictions generated from this model are accurate and could be subsequently used for the optimization, and control of diffusion dynamics. We focus on a challenging setting where full observations of the dynamics are not available, and standard approaches such as maximum likelihood quickly become intractable for large network instances. We introduce a computationally efficient algorithm, based on a scalable dynamic message-passing approach, which is able to learn parameters of the effective spreading model given only limited information on the activation times of nodes in the network. The popular Independent Cascade model is used to illustrate our approach. We show that tractable inference from the learned model generates a better prediction of marginal probabilities compared to the original model. We develop a systematic procedure for learning a mixture of models which further improves the prediction quality.

----

## [1018] Leveraging Language to Learn Program Abstractions and Search Heuristics

**Authors**: *Catherine Wong, Kevin Ellis, Joshua B. Tenenbaum, Jacob Andreas*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/wong21a.html](http://proceedings.mlr.press/v139/wong21a.html)

**Abstract**:

Inductive program synthesis, or inferring programs from examples of desired behavior, offers a general paradigm for building interpretable, robust, andgeneralizable machine learning systems. Effective program synthesis depends on two key ingredients: a strong library of functions from which to build programs, and an efficient search strategy for finding programs that solve a given task. We introduce LAPS (Language for Abstraction and Program Search), a technique for using natural language annotations to guide joint learning of libraries and neurally-guided search models for synthesis. When integrated into a state-of-the-art library learning system (DreamCoder), LAPS produces higher-quality libraries and improves search efficiency and generalization on three domains {–} string editing, image composition, and abstract reasoning about scenes {–} even when no natural language hints are available at test time.

----

## [1019] Leveraging Sparse Linear Layers for Debuggable Deep Networks

**Authors**: *Eric Wong, Shibani Santurkar, Aleksander Madry*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/wong21b.html](http://proceedings.mlr.press/v139/wong21b.html)

**Abstract**:

We show how fitting sparse linear models over learned deep feature representations can lead to more debuggable neural networks. These networks remain highly accurate while also being more amenable to human interpretation, as we demonstrate quantitatively and via human experiments. We further illustrate how the resulting sparse explanations can help to identify spurious correlations, explain misclassifications, and diagnose model biases in vision and language tasks.

----

## [1020] Learning Neural Network Subspaces

**Authors**: *Mitchell Wortsman, Maxwell Horton, Carlos Guestrin, Ali Farhadi, Mohammad Rastegari*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/wortsman21a.html](http://proceedings.mlr.press/v139/wortsman21a.html)

**Abstract**:

Recent observations have advanced our understanding of the neural network optimization landscape, revealing the existence of (1) paths of high accuracy containing diverse solutions and (2) wider minima offering improved performance. Previous methods observing diverse paths require multiple training runs. In contrast we aim to leverage both property (1) and (2) with a single method and in a single training run. With a similar computational cost as training one model, we learn lines, curves, and simplexes of high-accuracy neural networks. These neural network subspaces contain diverse solutions that can be ensembled, approaching the ensemble performance of independently trained networks without the training cost. Moreover, using the subspace midpoint boosts accuracy, calibration, and robustness to label noise, outperforming Stochastic Weight Averaging.

----

## [1021] Conjugate Energy-Based Models

**Authors**: *Hao Wu, Babak Esmaeili, Michael L. Wick, Jean-Baptiste Tristan, Jan-Willem van de Meent*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/wu21a.html](http://proceedings.mlr.press/v139/wu21a.html)

**Abstract**:

In this paper, we propose conjugate energy-based models (CEBMs), a new class of energy-based models that define a joint density over data and latent variables. The joint density of a CEBM decomposes into an intractable distribution over data and a tractable posterior over latent variables. CEBMs have similar use cases as variational autoencoders, in the sense that they learn an unsupervised mapping from data to latent variables. However, these models omit a generator network, which allows them to learn more flexible notions of similarity between data points. Our experiments demonstrate that conjugate EBMs achieve competitive results in terms of image modelling, predictive power of latent space, and out-of-domain detection on a variety of datasets.

----

## [1022] Making Paper Reviewing Robust to Bid Manipulation Attacks

**Authors**: *Ruihan Wu, Chuan Guo, Felix Wu, Rahul Kidambi, Laurens van der Maaten, Kilian Q. Weinberger*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/wu21b.html](http://proceedings.mlr.press/v139/wu21b.html)

**Abstract**:

Most computer science conferences rely on paper bidding to assign reviewers to papers. Although paper bidding enables high-quality assignments in days of unprecedented submission numbers, it also opens the door for dishonest reviewers to adversarially influence paper reviewing assignments. Anecdotal evidence suggests that some reviewers bid on papers by "friends" or colluding authors, even though these papers are outside their area of expertise, and recommend them for acceptance without considering the merit of the work. In this paper, we study the efficacy of such bid manipulation attacks and find that, indeed, they can jeopardize the integrity of the review process. We develop a novel approach for paper bidding and assignment that is much more robust against such attacks. We show empirically that our approach provides robustness even when dishonest reviewers collude, have full knowledge of the assignment system’s internal workings, and have access to the system’s inputs. In addition to being more robust, the quality of our paper review assignments is comparable to that of current, non-robust assignment approaches.

----

## [1023] LIME: Learning Inductive Bias for Primitives of Mathematical Reasoning

**Authors**: *Yuhuai Wu, Markus N. Rabe, Wenda Li, Jimmy Ba, Roger B. Grosse, Christian Szegedy*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/wu21c.html](http://proceedings.mlr.press/v139/wu21c.html)

**Abstract**:

While designing inductive bias in neural architectures has been widely studied, we hypothesize that transformer networks are flexible enough to learn inductive bias from suitable generic tasks. Here, we replace architecture engineering by encoding inductive bias in the form of datasets. Inspired by Peirce’s view that deduction, induction, and abduction are the primitives of reasoning, we design three synthetic tasks that are intended to require the model to have these three abilities. We specifically design these tasks to be synthetic and devoid of mathematical knowledge to ensure that only the fundamental reasoning biases can be learned from these tasks. This defines a new pre-training methodology called "LIME" (Learning Inductive bias for Mathematical rEasoning). Models trained with LIME significantly outperform vanilla transformers on four very different large mathematical reasoning benchmarks. Unlike dominating the computation cost as traditional pre-training approaches, LIME requires only a small fraction of the computation cost of the typical downstream task. The code for generating LIME tasks is available at https://github.com/tonywu95/LIME.

----

## [1024] ChaCha for Online AutoML

**Authors**: *Qingyun Wu, Chi Wang, John Langford, Paul Mineiro, Marco Rossi*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/wu21d.html](http://proceedings.mlr.press/v139/wu21d.html)

**Abstract**:

We propose the ChaCha (Champion-Challengers) algorithm for making an online choice of hyperparameters in online learning settings. ChaCha handles the process of determining a champion and scheduling a set of ‘live’ challengers over time based on sample complexity bounds. It is guaranteed to have sublinear regret after the optimal configuration is added into consideration by an application-dependent oracle based on the champions. Empirically, we show that ChaCha provides good performance across a wide array of datasets when optimizing over featurization and hyperparameter decisions.

----

## [1025] Temporally Correlated Task Scheduling for Sequence Learning

**Authors**: *Xueqing Wu, Lewen Wang, Yingce Xia, Weiqing Liu, Lijun Wu, Shufang Xie, Tao Qin, Tie-Yan Liu*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/wu21e.html](http://proceedings.mlr.press/v139/wu21e.html)

**Abstract**:

Sequence learning has attracted much research attention from the machine learning community in recent years. In many applications, a sequence learning task is usually associated with multiple temporally correlated auxiliary tasks, which are different in terms of how much input information to use or which future step to predict. For example, (i) in simultaneous machine translation, one can conduct translation under different latency (i.e., how many input words to read/wait before translation); (ii) in stock trend forecasting, one can predict the price of a stock in different future days (e.g., tomorrow, the day after tomorrow). While it is clear that those temporally correlated tasks can help each other, there is a very limited exploration on how to better leverage multiple auxiliary tasks to boost the performance of the main task. In this work, we introduce a learnable scheduler to sequence learning, which can adaptively select auxiliary tasks for training depending on the model status and the current training data. The scheduler and the model for the main task are jointly trained through bi-level optimization. Experiments show that our method significantly improves the performance of simultaneous machine translation and stock trend forecasting.

----

## [1026] Class2Simi: A Noise Reduction Perspective on Learning with Noisy Labels

**Authors**: *Songhua Wu, Xiaobo Xia, Tongliang Liu, Bo Han, Mingming Gong, Nannan Wang, Haifeng Liu, Gang Niu*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/wu21f.html](http://proceedings.mlr.press/v139/wu21f.html)

**Abstract**:

Learning with noisy labels has attracted a lot of attention in recent years, where the mainstream approaches are in \emph{pointwise} manners. Meanwhile, \emph{pairwise} manners have shown great potential in supervised metric learning and unsupervised contrastive learning. Thus, a natural question is raised: does learning in a pairwise manner \emph{mitigate} label noise? To give an affirmative answer, in this paper, we propose a framework called \emph{Class2Simi}: it transforms data points with noisy \emph{class labels} to data pairs with noisy \emph{similarity labels}, where a similarity label denotes whether a pair shares the class label or not. Through this transformation, the \emph{reduction of the noise rate} is theoretically guaranteed, and hence it is in principle easier to handle noisy similarity labels. Amazingly, DNNs that predict the \emph{clean} class labels can be trained from noisy data pairs if they are first pretrained from noisy data points. Class2Simi is \emph{computationally efficient} because not only this transformation is on-the-fly in mini-batches, but also it just changes loss computation on top of model prediction into a pairwise manner. Its effectiveness is verified by extensive experiments.

----

## [1027] On Reinforcement Learning with Adversarial Corruption and Its Application to Block MDP

**Authors**: *Tianhao Wu, Yunchang Yang, Simon S. Du, Liwei Wang*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/wu21g.html](http://proceedings.mlr.press/v139/wu21g.html)

**Abstract**:

We study reinforcement learning (RL) in episodic tabular MDPs with adversarial corruptions, where some episodes can be adversarially corrupted. When the total number of corrupted episodes is known, we propose an algorithm, Corruption Robust Monotonic Value Propagation (\textsf{CR-MVP}), which achieves a regret bound of $\tilde{O}\left(\left(\sqrt{SAK}+S^2A+CSA)\right)\polylog(H)\right)$, where $S$ is the number of states, $A$ is the number of actions, $H$ is the planning horizon, $K$ is the number of episodes, and $C$ is the corruption level. We also provide a corresponding lower bound, which indicates that our upper bound is tight. Finally, as an application, we study RL with rich observations in the block MDP model. We provide the first algorithm that achieves a $\sqrt{K}$-type regret in this setting and is computationally efficient.

----

## [1028] Generative Video Transformer: Can Objects be the Words?

**Authors**: *Yi-Fu Wu, Jaesik Yoon, Sungjin Ahn*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/wu21h.html](http://proceedings.mlr.press/v139/wu21h.html)

**Abstract**:

Transformers have been successful for many natural language processing tasks. However, applying transformers to the video domain for tasks such as long-term video generation and scene understanding has remained elusive due to the high computational complexity and the lack of natural tokenization. In this paper, we propose the ObjectCentric Video Transformer (OCVT) which utilizes an object-centric approach for decomposing scenes into tokens suitable for use in a generative video transformer. By factoring the video into objects, our fully unsupervised model is able to learn complex spatio-temporal dynamics of multiple interacting objects in a scene and generate future frames of the video. Our model is also significantly more memory-efficient than pixel-based models and thus able to train on videos of length up to 70 frames with a single 48GB GPU. We compare our model with previous RNN-based approaches as well as other possible video transformer baselines. We demonstrate OCVT performs well when compared to baselines in generating future frames. OCVT also develops useful representations for video reasoning, achieving start-of-the-art performance on the CATER task.

----

## [1029] Uncertainty Weighted Actor-Critic for Offline Reinforcement Learning

**Authors**: *Yue Wu, Shuangfei Zhai, Nitish Srivastava, Joshua M. Susskind, Jian Zhang, Ruslan Salakhutdinov, Hanlin Goh*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/wu21i.html](http://proceedings.mlr.press/v139/wu21i.html)

**Abstract**:

Offline Reinforcement Learning promises to learn effective policies from previously-collected, static datasets without the need for exploration. However, existing Q-learning and actor-critic based off-policy RL algorithms fail when bootstrapping from out-of-distribution (OOD) actions or states. We hypothesize that a key missing ingredient from the existing methods is a proper treatment of uncertainty in the offline setting. We propose Uncertainty Weighted Actor-Critic (UWAC), an algorithm that detects OOD state-action pairs and down-weights their contribution in the training objectives accordingly. Implementation-wise, we adopt a practical and effective dropout-based uncertainty estimation method that introduces very little overhead over existing RL algorithms. Empirically, we observe that UWAC substantially improves model stability during training. In addition, UWAC out-performs existing offline RL methods on a variety of competitive tasks, and achieves significant performance gains over the state-of-the-art baseline on datasets with sparse demonstrations collected from human experts.

----

## [1030] Towards Open-World Recommendation: An Inductive Model-based Collaborative Filtering Approach

**Authors**: *Qitian Wu, Hengrui Zhang, Xiaofeng Gao, Junchi Yan, Hongyuan Zha*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/wu21j.html](http://proceedings.mlr.press/v139/wu21j.html)

**Abstract**:

Recommendation models can effectively estimate underlying user interests and predict one’s future behaviors by factorizing an observed user-item rating matrix into products of two sets of latent factors. However, the user-specific embedding factors can only be learned in a transductive way, making it difficult to handle new users on-the-fly. In this paper, we propose an inductive collaborative filtering framework that contains two representation models. The first model follows conventional matrix factorization which factorizes a group of key users’ rating matrix to obtain meta latents. The second model resorts to attention-based structure learning that estimates hidden relations from query to key users and learns to leverage meta latents to inductively compute embeddings for query users via neural message passing. Our model enables inductive representation learning for users and meanwhile guarantees equivalent representation capacity as matrix factorization. Experiments demonstrate that our model achieves promising results for recommendation on few-shot users with limited training ratings and new unseen users which are commonly encountered in open-world recommender systems.

----

## [1031] Data-efficient Hindsight Off-policy Option Learning

**Authors**: *Markus Wulfmeier, Dushyant Rao, Roland Hafner, Thomas Lampe, Abbas Abdolmaleki, Tim Hertweck, Michael Neunert, Dhruva Tirumala, Noah Y. Siegel, Nicolas Heess, Martin A. Riedmiller*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/wulfmeier21a.html](http://proceedings.mlr.press/v139/wulfmeier21a.html)

**Abstract**:

We introduce Hindsight Off-policy Options (HO2), a data-efficient option learning algorithm. Given any trajectory, HO2 infers likely option choices and backpropagates through the dynamic programming inference procedure to robustly train all policy components off-policy and end-to-end. The approach outperforms existing option learning methods on common benchmarks. To better understand the option framework and disentangle benefits from both temporal and action abstraction, we evaluate ablations with flat policies and mixture policies with comparable optimization. The results highlight the importance of both types of abstraction as well as off-policy training and trust-region constraints, particularly in challenging, simulated 3D robot manipulation tasks from raw pixel inputs. Finally, we intuitively adapt the inference step to investigate the effect of increased temporal abstraction on training with pre-trained options and from scratch.

----

## [1032] A Bit More Bayesian: Domain-Invariant Learning with Uncertainty

**Authors**: *Zehao Xiao, Jiayi Shen, Xiantong Zhen, Ling Shao, Cees Snoek*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/xiao21a.html](http://proceedings.mlr.press/v139/xiao21a.html)

**Abstract**:

Domain generalization is challenging due to the domain shift and the uncertainty caused by the inaccessibility of target domain data. In this paper, we address both challenges with a probabilistic framework based on variational Bayesian inference, by incorporating uncertainty into neural network weights. We couple domain invariance in a probabilistic formula with the variational Bayesian inference. This enables us to explore domain-invariant learning in a principled way. Specifically, we derive domain-invariant representations and classifiers, which are jointly established in a two-layer Bayesian neural network. We empirically demonstrate the effectiveness of our proposal on four widely used cross-domain visual recognition benchmarks. Ablation studies validate the synergistic benefits of our Bayesian treatment when jointly learning domain-invariant representations and classifiers for domain generalization. Further, our method consistently delivers state-of-the-art mean accuracy on all benchmarks.

----

## [1033] On the Optimality of Batch Policy Optimization Algorithms

**Authors**: *Chenjun Xiao, Yifan Wu, Jincheng Mei, Bo Dai, Tor Lattimore, Lihong Li, Csaba Szepesvári, Dale Schuurmans*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/xiao21b.html](http://proceedings.mlr.press/v139/xiao21b.html)

**Abstract**:

Batch policy optimization considers leveraging existing data for policy construction before interacting with an environment. Although interest in this problem has grown significantly in recent years, its theoretical foundations remain under-developed. To advance the understanding of this problem, we provide three results that characterize the limits and possibilities of batch policy optimization in the finite-armed stochastic bandit setting. First, we introduce a class of confidence-adjusted index algorithms that unifies optimistic and pessimistic principles in a common framework, which enables a general analysis. For this family, we show that any confidence-adjusted index algorithm is minimax optimal, whether it be optimistic, pessimistic or neutral. Our analysis reveals that instance-dependent optimality, commonly used to establish optimality of on-line stochastic bandit algorithms, cannot be achieved by any algorithm in the batch setting. In particular, for any algorithm that performs optimally in some environment, there exists another environment where the same algorithm suffers arbitrarily larger regret. Therefore, to establish a framework for distinguishing algorithms, we introduce a new weighted-minimax criterion that considers the inherent difficulty of optimal value prediction. We demonstrate how this criterion can be used to justify commonly used pessimistic principles for batch policy optimization.

----

## [1034] CRFL: Certifiably Robust Federated Learning against Backdoor Attacks

**Authors**: *Chulin Xie, Minghao Chen, Pin-Yu Chen, Bo Li*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/xie21a.html](http://proceedings.mlr.press/v139/xie21a.html)

**Abstract**:

Federated Learning (FL) as a distributed learning paradigm that aggregates information from diverse clients to train a shared global model, has demonstrated great success. However, malicious clients can perform poisoning attacks and model replacement to introduce backdoors into the trained global model. Although there have been intensive studies designing robust aggregation methods and empirical robust federated training protocols against backdoors, existing approaches lack robustness certification. This paper provides the first general framework, Certifiably Robust Federated Learning (CRFL), to train certifiably robust FL models against backdoors. Our method exploits clipping and smoothing on model parameters to control the global model smoothness, which yields a sample-wise robustness certification on backdoors with limited magnitude. Our certification also specifies the relation to federated learning parameters, such as poisoning ratio on instance level, number of attackers, and training iterations. Practically, we conduct comprehensive experiments across a range of federated datasets, and provide the first benchmark for certified robustness against backdoor attacks in federated learning. Our code is publicaly available at https://github.com/AI-secure/CRFL.

----

## [1035] RNNRepair: Automatic RNN Repair via Model-based Analysis

**Authors**: *Xiaofei Xie, Wenbo Guo, Lei Ma, Wei Le, Jian Wang, Lingjun Zhou, Yang Liu, Xinyu Xing*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/xie21b.html](http://proceedings.mlr.press/v139/xie21b.html)

**Abstract**:

Deep neural networks are vulnerable to adversarial attacks. Due to their black-box nature, it is rather challenging to interpret and properly repair these incorrect behaviors. This paper focuses on interpreting and repairing the incorrect behaviors of Recurrent Neural Networks (RNNs). We propose a lightweight model-based approach (RNNRepair) to help understand and repair incorrect behaviors of an RNN. Specifically, we build an influence model to characterize the stateful and statistical behaviors of an RNN over all the training data and to perform the influence analysis for the errors. Compared with the existing techniques on influence function, our method can efficiently estimate the influence of existing or newly added training samples for a given prediction at both sample level and segmentation level. Our empirical evaluation shows that the proposed influence model is able to extract accurate and understandable features. Based on the influence model, our proposed technique could effectively infer the influential instances from not only an entire testing sequence but also a segment within that sequence. Moreover, with the sample-level and segment-level influence relations, RNNRepair could further remediate two types of incorrect predictions at the sample level and segment level.

----

## [1036] Deep Reinforcement Learning amidst Continual Structured Non-Stationarity

**Authors**: *Annie Xie, James Harrison, Chelsea Finn*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/xie21c.html](http://proceedings.mlr.press/v139/xie21c.html)

**Abstract**:

As humans, our goals and our environment are persistently changing throughout our lifetime based on our experiences, actions, and internal and external drives. In contrast, typical reinforcement learning problem set-ups consider decision processes that are stationary across episodes. Can we develop reinforcement learning algorithms that can cope with the persistent change in the former, more realistic problem settings? While on-policy algorithms such as policy gradients in principle can be extended to non-stationary settings, the same cannot be said for more efficient off-policy algorithms that replay past experiences when learning. In this work, we formalize this problem setting, and draw upon ideas from the online learning and probabilistic inference literature to derive an off-policy RL algorithm that can reason about and tackle such lifelong non-stationarity. Our method leverages latent variable models to learn a representation of the environment from current and past experiences, and performs off-policy RL with this representation. We further introduce several simulation environments that exhibit lifelong non-stationarity, and empirically find that our approach substantially outperforms approaches that do not reason about environment shift.

----

## [1037] Batch Value-function Approximation with Only Realizability

**Authors**: *Tengyang Xie, Nan Jiang*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/xie21d.html](http://proceedings.mlr.press/v139/xie21d.html)

**Abstract**:

We make progress in a long-standing problem of batch reinforcement learning (RL): learning Q* from an exploratory and polynomial-sized dataset, using a realizable and otherwise arbitrary function class. In fact, all existing algorithms demand function-approximation assumptions stronger than realizability, and the mounting negative evidence has led to a conjecture that sample-efficient learning is impossible in this setting (Chen & Jiang, 2019). Our algorithm, BVFT, breaks the hardness conjecture (albeit under a stronger notion of exploratory data) via a tournament procedure that reduces the learning problem to pairwise comparison, and solves the latter with the help of a state-action-space partition constructed from the compared functions. We also discuss how BVFT can be applied to model selection among other extensions and open problems.

----

## [1038] Interaction-Grounded Learning

**Authors**: *Tengyang Xie, John Langford, Paul Mineiro, Ida Momennejad*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/xie21e.html](http://proceedings.mlr.press/v139/xie21e.html)

**Abstract**:

Consider a prosthetic arm, learning to adapt to its user’s control signals. We propose \emph{Interaction-Grounded Learning} for this novel setting, in which a learner’s goal is to interact with the environment with no grounding or explicit reward to optimize its policies. Such a problem evades common RL solutions which require an explicit reward. The learning agent observes a multidimensional \emph{context vector}, takes an \emph{action}, and then observes a multidimensional \emph{feedback vector}. This multidimensional feedback vector has \emph{no} explicit reward information. In order to succeed, the algorithm must learn how to evaluate the feedback vector to discover a latent reward signal, with which it can ground its policies without supervision. We show that in an Interaction-Grounded Learning setting, with certain natural assumptions, a learner can discover the latent reward and ground its policy for successful interaction. We provide theoretical guarantees and a proof-of-concept empirical evaluation to demonstrate the effectiveness of our proposed approach.

----

## [1039] Composed Fine-Tuning: Freezing Pre-Trained Denoising Autoencoders for Improved Generalization

**Authors**: *Sang Michael Xie, Tengyu Ma, Percy Liang*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/xie21f.html](http://proceedings.mlr.press/v139/xie21f.html)

**Abstract**:

We focus on prediction problems with structured outputs that are subject to output validity constraints, e.g. pseudocode-to-code translation where the code must compile. While labeled input-output pairs are expensive to obtain, "unlabeled" outputs, i.e. outputs without corresponding inputs, are freely available (e.g. code on GitHub) and provide information about output validity. Pre-training captures this structure by training a denoiser to denoise corrupted versions of unlabeled outputs. We first show that standard fine-tuning after pre-training destroys some of this structure. We then propose composed fine-tuning, which trains a predictor composed with the pre-trained denoiser. Importantly, the denoiser is fixed to preserve output structure. Like standard fine-tuning, the predictor is also initialized with the pre-trained denoiser. We prove for two-layer ReLU networks that composed fine-tuning significantly reduces the complexity of the predictor, thus improving generalization. Empirically, we show that composed fine-tuning improves over standard fine-tuning on two pseudocode-to-code translation datasets (3% and 6% relative). The improvement is magnified on out-of-distribution (OOD) examples (4% and 25% relative), suggesting that reducing predictor complexity improves OOD extrapolation.

----

## [1040] Learning While Playing in Mean-Field Games: Convergence and Optimality

**Authors**: *Qiaomin Xie, Zhuoran Yang, Zhaoran Wang, Andreea Minca*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/xie21g.html](http://proceedings.mlr.press/v139/xie21g.html)

**Abstract**:

We study reinforcement learning in mean-field games. To achieve the Nash equilibrium, which consists of a policy and a mean-field state, existing algorithms require obtaining the optimal policy while fixing any mean-field state. In practice, however, the policy and the mean-field state evolve simultaneously, as each agent is learning while playing. To bridge such a gap, we propose a fictitious play algorithm, which alternatively updates the policy (learning) and the mean-field state (playing) by one step of policy optimization and gradient descent, respectively. Despite the nonstationarity induced by such an alternating scheme, we prove that the proposed algorithm converges to the Nash equilibrium with an explicit convergence rate. To the best of our knowledge, it is the first provably efficient algorithm that achieves learning while playing via alternating updates.

----

## [1041] Positive-Negative Momentum: Manipulating Stochastic Gradient Noise to Improve Generalization

**Authors**: *Zeke Xie, Li Yuan, Zhanxing Zhu, Masashi Sugiyama*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/xie21h.html](http://proceedings.mlr.press/v139/xie21h.html)

**Abstract**:

It is well-known that stochastic gradient noise (SGN) acts as implicit regularization for deep learning and is essentially important for both optimization and generalization of deep networks. Some works attempted to artificially simulate SGN by injecting random noise to improve deep learning. However, it turned out that the injected simple random noise cannot work as well as SGN, which is anisotropic and parameter-dependent. For simulating SGN at low computational costs and without changing the learning rate or batch size, we propose the Positive-Negative Momentum (PNM) approach that is a powerful alternative to conventional Momentum in classic optimizers. The introduced PNM method maintains two approximate independent momentum terms. Then, we can control the magnitude of SGN explicitly by adjusting the momentum difference. We theoretically prove the convergence guarantee and the generalization advantage of PNM over Stochastic Gradient Descent (SGD). By incorporating PNM into the two conventional optimizers, SGD with Momentum and Adam, our extensive experiments empirically verified the significant advantage of the PNM-based variants over the corresponding conventional Momentum-based optimizers. Code: \url{https://github.com/zeke-xie/Positive-Negative-Momentum}.

----

## [1042] A Hybrid Variance-Reduced Method for Decentralized Stochastic Non-Convex Optimization

**Authors**: *Ran Xin, Usman A. Khan, Soummya Kar*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/xin21a.html](http://proceedings.mlr.press/v139/xin21a.html)

**Abstract**:

This paper considers decentralized stochastic optimization over a network of $n$ nodes, where each node possesses a smooth non-convex local cost function and the goal of the networked nodes is to find an $\epsilon$-accurate first-order stationary point of the sum of the local costs. We focus on an online setting, where each node accesses its local cost only by means of a stochastic first-order oracle that returns a noisy version of the exact gradient. In this context, we propose a novel single-loop decentralized hybrid variance-reduced stochastic gradient method, called GT-HSGD, that outperforms the existing approaches in terms of both the oracle complexity and practical implementation. The GT-HSGD algorithm implements specialized local hybrid stochastic gradient estimators that are fused over the network to track the global gradient. Remarkably, GT-HSGD achieves a network topology-independent oracle complexity of $O(n^{-1}\epsilon^{-3})$ when the required error tolerance $\epsilon$ is small enough, leading to a linear speedup with respect to the centralized optimal online variance-reduced approaches that operate on a single node. Numerical experiments are provided to illustrate our main technical results.

----

## [1043] Explore Visual Concept Formation for Image Classification

**Authors**: *Shengzhou Xiong, Yihua Tan, Guoyou Wang*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/xiong21a.html](http://proceedings.mlr.press/v139/xiong21a.html)

**Abstract**:

Human beings acquire the ability of image classification through visual concept learning, in which the process of concept formation involves intertwined searches of common properties and concept descriptions. However, in most image classification algorithms using deep convolutional neural network (ConvNet), the representation space is constructed under the premise that concept descriptions are fixed as one-hot codes, which limits the mining of properties and the ability of identifying unseen samples. Inspired by this, we propose a learning strategy of visual concept formation (LSOVCF) based on the ConvNet, in which the two intertwined parts of concept formation, i.e. feature extraction and concept description, are learned together. First, LSOVCF takes sample response in the last layer of ConvNet to induct concept description being assumed as Gaussian distribution, which is part of the training process. Second, the exploration and experience loss is designed for optimization, which adopts experience cache pool to speed up convergence. Experiments show that LSOVCF improves the ability of identifying unseen samples on cifar10, STL10, flower17 and ImageNet based on several backbones, from the classic VGG to the SOTA Ghostnet. The code is available at \url{https://github.com/elvintanhust/LSOVCF}.

----

## [1044] CRPO: A New Approach for Safe Reinforcement Learning with Convergence Guarantee

**Authors**: *Tengyu Xu, Yingbin Liang, Guanghui Lan*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/xu21a.html](http://proceedings.mlr.press/v139/xu21a.html)

**Abstract**:

In safe reinforcement learning (SRL) problems, an agent explores the environment to maximize an expected total reward and meanwhile avoids violation of certain constraints on a number of expected total costs. In general, such SRL problems have nonconvex objective functions subject to multiple nonconvex constraints, and hence are very challenging to solve, particularly to provide a globally optimal policy. Many popular SRL algorithms adopt a primal-dual structure which utilizes the updating of dual variables for satisfying the constraints. In contrast, we propose a primal approach, called constraint-rectified policy optimization (CRPO), which updates the policy alternatingly between objective improvement and constraint satisfaction. CRPO provides a primal-type algorithmic framework to solve SRL problems, where each policy update can take any variant of policy optimization step. To demonstrate the theoretical performance of CRPO, we adopt natural policy gradient (NPG) for each policy update step and show that CRPO achieves an $\mathcal{O}(1/\sqrt{T})$ convergence rate to the global optimal policy in the constrained policy set and an $\mathcal{O}(1/\sqrt{T})$ error bound on constraint satisfaction. This is the first finite-time analysis of primal SRL algorithms with global optimality guarantee. Our empirical results demonstrate that CRPO can outperform the existing primal-dual baseline algorithms significantly.

----

## [1045] To be Robust or to be Fair: Towards Fairness in Adversarial Training

**Authors**: *Han Xu, Xiaorui Liu, Yaxin Li, Anil K. Jain, Jiliang Tang*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/xu21b.html](http://proceedings.mlr.press/v139/xu21b.html)

**Abstract**:

Adversarial training algorithms have been proved to be reliable to improve machine learning models’ robustness against adversarial examples. However, we find that adversarial training algorithms tend to introduce severe disparity of accuracy and robustness between different groups of data. For instance, PGD adversarially trained ResNet18 model on CIFAR-10 has 93% clean accuracy and 67% PGD l_infty-8 adversarial accuracy on the class ”automobile” but only 65% and 17% on class ”cat”. This phenomenon happens in balanced datasets and does not exist in naturally trained models when only using clean samples. In this work, we empirically and theoretically show that this phenomenon can generally happen under adversarial training algorithms which minimize DNN models’ robust errors. Motivated by these findings, we propose a Fair-Robust-Learning (FRL) framework to mitigate this unfairness problem when doing adversarial defenses and experimental results validate the effectiveness of FRL.

----

## [1046] Interpretable Stein Goodness-of-fit Tests on Riemannian Manifold

**Authors**: *Wenkai Xu, Takeru Matsuda*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/xu21c.html](http://proceedings.mlr.press/v139/xu21c.html)

**Abstract**:

In many applications, we encounter data on Riemannian manifolds such as torus and rotation groups. Standard statistical procedures for multivariate data are not applicable to such data. In this study, we develop goodness-of-fit testing and interpretable model criticism methods for general distributions on Riemannian manifolds, including those with an intractable normalization constant. The proposed methods are based on extensions of kernel Stein discrepancy, which are derived from Stein operators on Riemannian manifolds. We discuss the connections between the proposed tests with existing ones and provide a theoretical analysis of their asymptotic Bahadur efficiency. Simulation results and real data applications show the validity and usefulness of the proposed methods.

----

## [1047] Rethinking Neural vs Matrix-Factorization Collaborative Filtering: the Theoretical Perspectives

**Authors**: *Da Xu, Chuanwei Ruan, Evren Körpeoglu, Sushant Kumar, Kannan Achan*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/xu21d.html](http://proceedings.mlr.press/v139/xu21d.html)

**Abstract**:

The recent work by Rendle et al. (2020), based on empirical observations, argues that matrix-factorization collaborative filtering (MCF) compares favorably to neural collaborative filtering (NCF), and conjectures the dot product’s superiority over the feed-forward neural network as similarity function. In this paper, we address the comparison rigorously by answering the following questions: 1. what is the limiting expressivity of each model; 2. under the practical gradient descent, to which solution does each optimization path converge; 3. how would the models generalize under the inductive and transductive learning setting. Our results highlight the similar expressivity for the overparameterized NCF and MCF as kernelized predictors, and reveal the relation between their optimization paths. We further show their different generalization behaviors, where MCF and NCF experience specific tradeoff and comparison in the transductive and inductive collaborative filtering setting. Lastly, by showing a novel generalization result, we reveal the critical role of correcting exposure bias for model evaluation in the inductive setting. Our results explain some of the previously observed conflicts, and we provide synthetic and real-data experiments to shed further insights to this topic.

----

## [1048] Dash: Semi-Supervised Learning with Dynamic Thresholding

**Authors**: *Yi Xu, Lei Shang, Jinxing Ye, Qi Qian, Yu-Feng Li, Baigui Sun, Hao Li, Rong Jin*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/xu21e.html](http://proceedings.mlr.press/v139/xu21e.html)

**Abstract**:

While semi-supervised learning (SSL) has received tremendous attentions in many machine learning tasks due to its successful use of unlabeled data, existing SSL algorithms use either all unlabeled examples or the unlabeled examples with a fixed high-confidence prediction during the training progress. However, it is possible that too many correct/wrong pseudo labeled examples are eliminated/selected. In this work we develop a simple yet powerful framework, whose key idea is to select a subset of training examples from the unlabeled data when performing existing SSL methods so that only the unlabeled examples with pseudo labels related to the labeled data will be used to train models. The selection is performed at each updating iteration by only keeping the examples whose losses are smaller than a given threshold that is dynamically adjusted through the iteration. Our proposed approach, Dash, enjoys its adaptivity in terms of unlabeled data selection and its theoretical guarantee. Specifically, we theoretically establish the convergence rate of Dash from the view of non-convex optimization. Finally, we empirically demonstrate the effectiveness of the proposed method in comparison with state-of-the-art over benchmarks.

----

## [1049] An End-to-End Framework for Molecular Conformation Generation via Bilevel Programming

**Authors**: *Minkai Xu, Wujie Wang, Shitong Luo, Chence Shi, Yoshua Bengio, Rafael Gómez-Bombarelli, Jian Tang*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/xu21f.html](http://proceedings.mlr.press/v139/xu21f.html)

**Abstract**:

Predicting molecular conformations (or 3D structures) from molecular graphs is a fundamental problem in many applications. Most existing approaches are usually divided into two steps by first predicting the distances between atoms and then generating a 3D structure through optimizing a distance geometry problem. However, the distances predicted with such two-stage approaches may not be able to consistently preserve the geometry of local atomic neighborhoods, making the generated structures unsatisfying. In this paper, we propose an end-to-end solution for molecular conformation prediction called ConfVAE based on the conditional variational autoencoder framework. Specifically, the molecular graph is first encoded in a latent space, and then the 3D structures are generated by solving a principled bilevel optimization program. Extensive experiments on several benchmark data sets prove the effectiveness of our proposed approach over existing state-of-the-art approaches. Code is available at \url{https://github.com/MinkaiXu/ConfVAE-ICML21}.

----

## [1050] Self-supervised Graph-level Representation Learning with Local and Global Structure

**Authors**: *Minghao Xu, Hang Wang, Bingbing Ni, Hongyu Guo, Jian Tang*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/xu21g.html](http://proceedings.mlr.press/v139/xu21g.html)

**Abstract**:

This paper studies unsupervised/self-supervised whole-graph representation learning, which is critical in many tasks such as molecule properties prediction in drug and material discovery. Existing methods mainly focus on preserving the local similarity structure between different graph instances but fail to discover the global semantic structure of the entire data set. In this paper, we propose a unified framework called Local-instance and Global-semantic Learning (GraphLoG) for self-supervised whole-graph representation learning. Specifically, besides preserving the local similarities, GraphLoG introduces the hierarchical prototypes to capture the global semantic clusters. An efficient online expectation-maximization (EM) algorithm is further developed for learning the model. We evaluate GraphLoG by pre-training it on massive unlabeled graphs followed by fine-tuning on downstream tasks. Extensive experiments on both chemical and biological benchmark data sets demonstrate the effectiveness of the proposed approach.

----

## [1051] Conformal prediction interval for dynamic time-series

**Authors**: *Chen Xu, Yao Xie*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/xu21h.html](http://proceedings.mlr.press/v139/xu21h.html)

**Abstract**:

We develop a method to construct distribution-free prediction intervals for dynamic time-series, called \Verb|EnbPI| that wraps around any bootstrap ensemble estimator to construct sequential prediction intervals. \Verb|EnbPI| is closely related to the conformal prediction (CP) framework but does not require data exchangeability. Theoretically, these intervals attain finite-sample, \textit{approximately valid} marginal coverage for broad classes of regression functions and time-series with strongly mixing stochastic errors. Computationally, \Verb|EnbPI| avoids overfitting and requires neither data-splitting nor training multiple ensemble estimators; it efficiently aggregates bootstrap estimators that have been trained. In general, \Verb|EnbPI| is easy to implement, scalable to producing arbitrarily many prediction intervals sequentially, and well-suited to a wide range of regression functions. We perform extensive real-data analyses to demonstrate its effectiveness.

----

## [1052] Learner-Private Convex Optimization

**Authors**: *Jiaming Xu, Kuang Xu, Dana Yang*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/xu21i.html](http://proceedings.mlr.press/v139/xu21i.html)

**Abstract**:

Convex optimization with feedback is a framework where a learner relies on iterative queries and feedback to arrive at the minimizer of a convex function. The paradigm has gained significant popularity recently thanks to its scalability in large-scale optimization and machine learning. The repeated interactions, however, expose the learner to privacy risks from eavesdropping adversaries that observe the submitted queries. In this paper, we study how to optimally obfuscate the learner’s queries in convex optimization with first-order feedback, so that their learned optimal value is provably difficult to estimate for the eavesdropping adversary. We consider two formulations of learner privacy: a Bayesian formulation in which the convex function is drawn randomly, and a minimax formulation in which the function is fixed and the adversary’s probability of error is measured with respect to a minimax criterion. We show that, if the learner wants to ensure the probability of the adversary estimating accurately be kept below 1/L, then the overhead in query complexity is additive in L in the minimax formulation, but multiplicative in L in the Bayesian formulation. Compared to existing learner-private sequential learning models with binary feedback, our results apply to the significantly richer family of general convex functions with full-gradient feedback. Our proofs are largely enabled by tools from the theory of Dirichlet processes, as well as more sophisticated lines of analysis aimed at measuring the amount of information leakage under a full-gradient oracle.

----

## [1053] Doubly Robust Off-Policy Actor-Critic: Convergence and Optimality

**Authors**: *Tengyu Xu, Zhuoran Yang, Zhaoran Wang, Yingbin Liang*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/xu21j.html](http://proceedings.mlr.press/v139/xu21j.html)

**Abstract**:

Designing off-policy reinforcement learning algorithms is typically a very challenging task, because a desirable iteration update often involves an expectation over an on-policy distribution. Prior off-policy actor-critic (AC) algorithms have introduced a new critic that uses the density ratio for adjusting the distribution mismatch in order to stabilize the convergence, but at the cost of potentially introducing high biases due to the estimation errors of both the density ratio and value function. In this paper, we develop a doubly robust off-policy AC (DR-Off-PAC) for discounted MDP, which can take advantage of learned nuisance functions to reduce estimation errors. Moreover, DR-Off-PAC adopts a single timescale structure, in which both actor and critics are updated simultaneously with constant stepsize, and is thus more sample efficient than prior algorithms that adopt either two timescale or nested-loop structure. We study the finite-time convergence rate and characterize the sample complexity for DR-Off-PAC to attain an $\epsilon$-accurate optimal policy. We also show that the overall convergence of DR-Off-PAC is doubly robust to the approximation errors that depend only on the expressive power of approximation functions. To the best of our knowledge, our study establishes the first overall sample complexity analysis for single time-scale off-policy AC algorithm.

----

## [1054] Optimization of Graph Neural Networks: Implicit Acceleration by Skip Connections and More Depth

**Authors**: *Keyulu Xu, Mozhi Zhang, Stefanie Jegelka, Kenji Kawaguchi*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/xu21k.html](http://proceedings.mlr.press/v139/xu21k.html)

**Abstract**:

Graph Neural Networks (GNNs) have been studied through the lens of expressive power and generalization. However, their optimization properties are less well understood. We take the first step towards analyzing GNN training by studying the gradient dynamics of GNNs. First, we analyze linearized GNNs and prove that despite the non-convexity of training, convergence to a global minimum at a linear rate is guaranteed under mild assumptions that we validate on real-world graphs. Second, we study what may affect the GNNs’ training speed. Our results show that the training of GNNs is implicitly accelerated by skip connections, more depth, and/or a good label distribution. Empirical results confirm that our theoretical results for linearized GNNs align with the training behavior of nonlinear GNNs. Our results provide the first theoretical support for the success of GNNs with skip connections in terms of optimization, and suggest that deep GNNs with skip connections would be promising in practice.

----

## [1055] Group-Sparse Matrix Factorization for Transfer Learning of Word Embeddings

**Authors**: *Kan Xu, Xuanyi Zhao, Hamsa Bastani, Osbert Bastani*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/xu21l.html](http://proceedings.mlr.press/v139/xu21l.html)

**Abstract**:

Sparse regression has recently been applied to enable transfer learning from very limited data. We study an extension of this approach to unsupervised learning—in particular, learning word embeddings from unstructured text corpora using low-rank matrix factorization. Intuitively, when transferring word embeddings to a new domain, we expect that the embeddings change for only a small number of words—e.g., the ones with novel meanings in that domain. We propose a novel group-sparse penalty that exploits this sparsity to perform transfer learning when there is very little text data available in the target domain—e.g., a single article of text. We prove generalization bounds for our algorithm. Furthermore, we empirically evaluate its effectiveness, both in terms of prediction accuracy in downstream tasks as well as in terms of interpretability of the results.

----

## [1056] KNAS: Green Neural Architecture Search

**Authors**: *Jingjing Xu, Liang Zhao, Junyang Lin, Rundong Gao, Xu Sun, Hongxia Yang*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/xu21m.html](http://proceedings.mlr.press/v139/xu21m.html)

**Abstract**:

Many existing neural architecture search (NAS) solutions rely on downstream training for architecture evaluation, which takes enormous computations. Considering that these computations bring a large carbon footprint, this paper aims to explore a green (namely environmental-friendly) NAS solution that evaluates architectures without training. Intuitively, gradients, induced by the architecture itself, directly decide the convergence and generalization results. It motivates us to propose the gradient kernel hypothesis: Gradients can be used as a coarse-grained proxy of downstream training to evaluate random-initialized networks. To support the hypothesis, we conduct a theoretical analysis and find a practical gradient kernel that has good correlations with training loss and validation performance. According to this hypothesis, we propose a new kernel based architecture search approach KNAS. Experiments show that KNAS achieves competitive results with orders of magnitude faster than “train-then-test” paradigms on image classification tasks. Furthermore, the extremely low search cost enables its wide applications. The searched network also outperforms strong baseline RoBERTA-large on two text classification tasks.

----

## [1057] Structured Convolutional Kernel Networks for Airline Crew Scheduling

**Authors**: *Yassine Yaakoubi, François Soumis, Simon Lacoste-Julien*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/yaakoubi21a.html](http://proceedings.mlr.press/v139/yaakoubi21a.html)

**Abstract**:

Motivated by the needs from an airline crew scheduling application, we introduce structured convolutional kernel networks (Struct-CKN), which combine CKNs from Mairal et al. (2014) in a structured prediction framework that supports constraints on the outputs. CKNs are a particular kind of convolutional neural networks that approximate a kernel feature map on training data, thus combining properties of deep learning with the non-parametric flexibility of kernel methods. Extending CKNs to structured outputs allows us to obtain useful initial solutions on a flight-connection dataset that can be further refined by an airline crew scheduling solver. More specifically, we use a flight-based network modeled as a general conditional random field capable of incorporating local constraints in the learning process. Our experiments demonstrate that this approach yields significant improvements for the large-scale crew pairing problem (50,000 flights per month) over standard approaches, reducing the solution cost by 17% (a gain of millions of dollars) and the cost of global constraints by 97%.

----

## [1058] Mediated Uncoupled Learning: Learning Functions without Direct Input-output Correspondences

**Authors**: *Ikko Yamane, Junya Honda, Florian Yger, Masashi Sugiyama*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/yamane21a.html](http://proceedings.mlr.press/v139/yamane21a.html)

**Abstract**:

Ordinary supervised learning is useful when we have paired training data of input $X$ and output $Y$. However, such paired data can be difficult to collect in practice. In this paper, we consider the task of predicting $Y$ from $X$ when we have no paired data of them, but we have two separate, independent datasets of $X$ and $Y$ each observed with some mediating variable $U$, that is, we have two datasets $S_X = \{(X_i, U_i)\}$ and $S_Y = \{(U’_j, Y’_j)\}$. A naive approach is to predict $U$ from $X$ using $S_X$ and then $Y$ from $U$ using $S_Y$, but we show that this is not statistically consistent. Moreover, predicting $U$ can be more difficult than predicting $Y$ in practice, e.g., when $U$ has higher dimensionality. To circumvent the difficulty, we propose a new method that avoids predicting $U$ but directly learns $Y = f(X)$ by training $f(X)$ with $S_{X}$ to predict $h(U)$ which is trained with $S_{Y}$ to approximate $Y$. We prove statistical consistency and error bounds of our method and experimentally confirm its practical usefulness.

----

## [1059] EL-Attention: Memory Efficient Lossless Attention for Generation

**Authors**: *Yu Yan, Jiusheng Chen, Weizhen Qi, Nikhil Bhendawade, Yeyun Gong, Nan Duan, Ruofei Zhang*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/yan21a.html](http://proceedings.mlr.press/v139/yan21a.html)

**Abstract**:

Transformer model with multi-head attention requires caching intermediate results for efficient inference in generation tasks. However, cache brings new memory-related costs and prevents leveraging larger batch size for faster speed. We propose memory-efficient lossless attention (called EL-attention) to address this issue. It avoids heavy operations for building multi-head keys and values, cache for them is not needed. EL-attention constructs an ensemble of attention results by expanding query while keeping key and value shared. It produces the same result as multi-head attention with less GPU memory and faster inference speed. We conduct extensive experiments on Transformer, BART, and GPT-2 for summarization and question generation tasks. The results show EL-attention speeds up existing models by 1.6x to 5.3x without accuracy loss.

----

## [1060] Link Prediction with Persistent Homology: An Interactive View

**Authors**: *Zuoyu Yan, Tengfei Ma, Liangcai Gao, Zhi Tang, Chao Chen*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/yan21b.html](http://proceedings.mlr.press/v139/yan21b.html)

**Abstract**:

Link prediction is an important learning task for graph-structured data. In this paper, we propose a novel topological approach to characterize interactions between two nodes. Our topological feature, based on the extended persistent homology, encodes rich structural information regarding the multi-hop paths connecting nodes. Based on this feature, we propose a graph neural network method that outperforms state-of-the-arts on different benchmarks. As another contribution, we propose a novel algorithm to more efficiently compute the extended persistence diagrams for graphs. This algorithm can be generally applied to accelerate many other topological methods for graph learning tasks.

----

## [1061] CATE: Computation-aware Neural Architecture Encoding with Transformers

**Authors**: *Shen Yan, Kaiqiang Song, Fei Liu, Mi Zhang*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/yan21c.html](http://proceedings.mlr.press/v139/yan21c.html)

**Abstract**:

Recent works (White et al., 2020a; Yan et al., 2020) demonstrate the importance of architecture encodings in Neural Architecture Search (NAS). These encodings encode either structure or computation information of the neural architectures. Compared to structure-aware encodings, computation-aware encodings map architectures with similar accuracies to the same region, which improves the downstream architecture search performance (Zhang et al., 2019; White et al., 2020a). In this work, we introduce a Computation-Aware Transformer-based Encoding method called CATE. Different from existing computation-aware encodings based on fixed transformation (e.g. path encoding), CATE employs a pairwise pre-training scheme to learn computation-aware encodings using Transformers with cross-attention. Such learned encodings contain dense and contextualized computation information of neural architectures. We compare CATE with eleven encodings under three major encoding-dependent NAS subroutines in both small and large search spaces. Our experiments show that CATE is beneficial to the downstream search, especially in the large search space. Moreover, the outside search space experiment demonstrates its superior generalization ability beyond the search space on which it was trained. Our code is available at: https://github.com/MSU-MLSys-Lab/CATE.

----

## [1062] On Perceptual Lossy Compression: The Cost of Perceptual Reconstruction and An Optimal Training Framework

**Authors**: *Zeyu Yan, Fei Wen, Rendong Ying, Chao Ma, Peilin Liu*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/yan21d.html](http://proceedings.mlr.press/v139/yan21d.html)

**Abstract**:

Lossy compression algorithms are typically designed to achieve the lowest possible distortion at a given bit rate. However, recent studies show that pursuing high perceptual quality would lead to increase of the lowest achievable distortion (e.g., MSE). This paper provides nontrivial results theoretically revealing that, 1) the cost of achieving perfect perception quality is exactly a doubling of the lowest achievable MSE distortion, 2) an optimal encoder for the “classic” rate-distortion problem is also optimal for the perceptual compression problem, 3) distortion loss is unnecessary for training a perceptual decoder. Further, we propose a novel training framework to achieve the lowest MSE distortion under perfect perception constraint at a given bit rate. This framework uses a GAN with discriminator conditioned on an MSE-optimized encoder, which is superior over the traditional framework using distortion plus adversarial loss. Experiments are provided to verify the theoretical finding and demonstrate the superiority of the proposed training framework.

----

## [1063] CIFS: Improving Adversarial Robustness of CNNs via Channel-wise Importance-based Feature Selection

**Authors**: *Hanshu Yan, Jingfeng Zhang, Gang Niu, Jiashi Feng, Vincent Y. F. Tan, Masashi Sugiyama*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/yan21e.html](http://proceedings.mlr.press/v139/yan21e.html)

**Abstract**:

We investigate the adversarial robustness of CNNs from the perspective of channel-wise activations. By comparing normally trained and adversarially trained models, we observe that adversarial training (AT) robustifies CNNs by aligning the channel-wise activations of adversarial data with those of their natural counterparts. However, the channels that are \textit{negatively-relevant} (NR) to predictions are still over-activated when processing adversarial data. Besides, we also observe that AT does not result in similar robustness for all classes. For the robust classes, channels with larger activation magnitudes are usually more \textit{positively-relevant} (PR) to predictions, but this alignment does not hold for the non-robust classes. Given these observations, we hypothesize that suppressing NR channels and aligning PR ones with their relevances further enhances the robustness of CNNs under AT. To examine this hypothesis, we introduce a novel mechanism, \textit{i.e.}, \underline{C}hannel-wise \underline{I}mportance-based \underline{F}eature \underline{S}election (CIFS). The CIFS manipulates channels’ activations of certain layers by generating non-negative multipliers to these channels based on their relevances to predictions. Extensive experiments on benchmark datasets including CIFAR10 and SVHN clearly verify the hypothesis and CIFS’s effectiveness of robustifying CNNs.

----

## [1064] Exact Gap between Generalization Error and Uniform Convergence in Random Feature Models

**Authors**: *Zitong Yang, Yu Bai, Song Mei*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/yang21a.html](http://proceedings.mlr.press/v139/yang21a.html)

**Abstract**:

Recent work showed that there could be a large gap between the classical uniform convergence bound and the actual test error of zero-training-error predictors (interpolators) such as deep neural networks. To better understand this gap, we study the uniform convergence in the nonlinear random feature model and perform a precise theoretical analysis on how uniform convergence depends on the sample size and the number of parameters. We derive and prove analytical expressions for three quantities in this model: 1) classical uniform convergence over norm balls, 2) uniform convergence over interpolators in the norm ball (recently proposed by \citet{zhou2021uniform}), and 3) the risk of minimum norm interpolator. We show that, in the setting where the classical uniform convergence bound is vacuous (diverges to $\infty$), uniform convergence over the interpolators still gives a non-trivial bound of the test error of interpolating solutions. We also showcase a different setting where classical uniform convergence bound is non-vacuous, but uniform convergence over interpolators can give an improved sample complexity guarantee. Our result provides a first exact comparison between the test errors and uniform convergence bounds for interpolators beyond simple linear models.

----

## [1065] Learning Optimal Auctions with Correlated Valuations from Samples

**Authors**: *Chunxue Yang, Xiaohui Bei*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/yang21b.html](http://proceedings.mlr.press/v139/yang21b.html)

**Abstract**:

In single-item auction design, it is well known due to Cremer and McLean that when bidders’ valuations are drawn from a correlated prior distribution, the auctioneer can extract full social surplus as revenue. However, in most real-world applications, the prior is usually unknown and can only be learned from historical data. In this work, we investigate the robustness of the optimal auction with correlated valuations via sample complexity analysis. We prove upper and lower bounds on the number of samples from the unknown prior required to learn a (1-epsilon)-approximately optimal auction. Our results reinforce the common belief that optimal correlated auctions are sensitive to the distribution parameters and hard to learn unless the prior distribution is well-behaved.

----

## [1066] Tensor Programs IV: Feature Learning in Infinite-Width Neural Networks

**Authors**: *Greg Yang, Edward J. Hu*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/yang21c.html](http://proceedings.mlr.press/v139/yang21c.html)

**Abstract**:

As its width tends to infinity, a deep neural network’s behavior under gradient descent can become simplified and predictable (e.g. given by the Neural Tangent Kernel (NTK)), if it is parametrized appropriately (e.g. the NTK parametrization). However, we show that the standard and NTK parametrizations of a neural network do not admit infinite-width limits that can *learn* features, which is crucial for pretraining and transfer learning such as with BERT. We propose simple modifications to the standard parametrization to allow for feature learning in the limit. Using the *Tensor Programs* technique, we derive explicit formulas for such limits. On Word2Vec and few-shot learning on Omniglot via MAML, two canonical tasks that rely crucially on feature learning, we compute these limits exactly. We find that they outperform both NTK baselines and finite-width networks, with the latter approaching the infinite-width feature learning performance as width increases.

----

## [1067] LARNet: Lie Algebra Residual Network for Face Recognition

**Authors**: *Xiaolong Yang, Xiaohong Jia, Dihong Gong, Dong-Ming Yan, Zhifeng Li, Wei Liu*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/yang21d.html](http://proceedings.mlr.press/v139/yang21d.html)

**Abstract**:

Face recognition is an important yet challenging problem in computer vision. A major challenge in practical face recognition applications lies in significant variations between profile and frontal faces. Traditional techniques address this challenge either by synthesizing frontal faces or by pose invariant learning. In this paper, we propose a novel method with Lie algebra theory to explore how face rotation in the 3D space affects the deep feature generation process of convolutional neural networks (CNNs). We prove that face rotation in the image space is equivalent to an additive residual component in the feature space of CNNs, which is determined solely by the rotation. Based on this theoretical finding, we further design a Lie Algebraic Residual Network (LARNet) for tackling pose robust face recognition. Our LARNet consists of a residual subnet for decoding rotation information from input face images, and a gating subnet to learn rotation magnitude for controlling the strength of the residual component contributing to the feature learning process. Comprehensive experimental evaluations on both frontal-profile face datasets and general face recognition datasets convincingly demonstrate that our method consistently outperforms the state-of-the-art ones.

----

## [1068] BASGD: Buffered Asynchronous SGD for Byzantine Learning

**Authors**: *Yi-Rui Yang, Wu-Jun Li*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/yang21e.html](http://proceedings.mlr.press/v139/yang21e.html)

**Abstract**:

Distributed learning has become a hot research topic due to its wide application in cluster-based large-scale learning, federated learning, edge computing and so on. Most traditional distributed learning methods typically assume no failure or attack. However, many unexpected cases, such as communication failure and even malicious attack, may happen in real applications. Hence, Byzantine learning (BL), which refers to distributed learning with failure or attack, has recently attracted much attention. Most existing BL methods are synchronous, which are impractical in some applications due to heterogeneous or offline workers. In these cases, asynchronous BL (ABL) is usually preferred. In this paper, we propose a novel method, called buffered asynchronous stochastic gradient descent (BASGD), for ABL. To the best of our knowledge, BASGD is the first ABL method that can resist malicious attack without storing any instances on server. Compared with those methods which need to store instances on server, BASGD has a wider scope of application. BASGD is proved to be convergent, and be able to resist failure or attack. Empirical results show that BASGD significantly outperforms vanilla asynchronous stochastic gradient descent (ASGD) and other ABL baselines when there exists failure or attack on workers.

----

## [1069] Tensor Programs IIb: Architectural Universality Of Neural Tangent Kernel Training Dynamics

**Authors**: *Greg Yang, Etai Littwin*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/yang21f.html](http://proceedings.mlr.press/v139/yang21f.html)

**Abstract**:

Yang (2020) recently showed that the Neural Tangent Kernel (NTK) at initialization has an infinite-width limit for a large class of architectures including modern staples such as ResNet and Transformers. However, their analysis does not apply to training. Here, we show the same neural networks (in the so-called NTK parametrization) during training follow a kernel gradient descent dynamics in function space, where the kernel is the infinite-width NTK. This completes the proof of the architectural universality of NTK behavior. To achieve this result, we apply the Tensor Programs technique: Write the entire SGD dynamics inside a Tensor Program and analyze it via the Master Theorem. To facilitate this proof, we develop a graphical notation for Tensor Programs, which we believe is also an important contribution toward the pedagogy and exposition of the Tensor Programs technique.

----

## [1070] Graph Neural Networks Inspired by Classical Iterative Algorithms

**Authors**: *Yongyi Yang, Tang Liu, Yangkun Wang, Jinjing Zhou, Quan Gan, Zhewei Wei, Zheng Zhang, Zengfeng Huang, David Wipf*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/yang21g.html](http://proceedings.mlr.press/v139/yang21g.html)

**Abstract**:

Despite the recent success of graph neural networks (GNN), common architectures often exhibit significant limitations, including sensitivity to oversmoothing, long-range dependencies, and spurious edges, e.g., as can occur as a result of graph heterophily or adversarial attacks. To at least partially address these issues within a simple transparent framework, we consider a new family of GNN layers designed to mimic and integrate the update rules of two classical iterative algorithms, namely, proximal gradient descent and iterative reweighted least squares (IRLS). The former defines an extensible base GNN architecture that is immune to oversmoothing while nonetheless capturing long-range dependencies by allowing arbitrary propagation steps. In contrast, the latter produces a novel attention mechanism that is explicitly anchored to an underlying end-to-end energy function, contributing stability with respect to edge uncertainty. When combined we obtain an extremely simple yet robust model that we evaluate across disparate scenarios including standardized benchmarks, adversarially-perturbated graphs, graphs with heterophily, and graphs involving long-range dependencies. In doing so, we compare against SOTA GNN approaches that have been explicitly designed for the respective task, achieving competitive or superior node classification accuracy. Our code is available at https://github.com/FFTYYY/TWIRLS. And for an extended version of this work, please see https://arxiv.org/abs/2103.06064.

----

## [1071] Representation Matters: Offline Pretraining for Sequential Decision Making

**Authors**: *Mengjiao Yang, Ofir Nachum*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/yang21h.html](http://proceedings.mlr.press/v139/yang21h.html)

**Abstract**:

The recent success of supervised learning methods on ever larger offline datasets has spurred interest in the reinforcement learning (RL) field to investigate whether the same paradigms can be translated to RL algorithms. This research area, known as offline RL, has largely focused on offline policy optimization, aiming to find a return-maximizing policy exclusively from offline data. In this paper, we consider a slightly different approach to incorporating offline data into sequential decision-making. We aim to answer the question, what unsupervised objectives applied to offline datasets are able to learn state representations which elevate performance on downstream tasks, whether those downstream tasks be online RL, imitation learning from expert demonstrations, or even offline policy optimization based on the same offline dataset? Through a variety of experiments utilizing standard offline RL datasets, we find that the use of pretraining with unsupervised learning objectives can dramatically improve the performance of policy learning algorithms that otherwise yield mediocre performance on their own. Extensive ablations further provide insights into what components of these unsupervised objectives {–} e.g., reward prediction, continuous or discrete representations, pretraining or finetuning {–} are most important and in which settings.

----

## [1072] Accelerating Safe Reinforcement Learning with Constraint-mismatched Baseline Policies

**Authors**: *Tsung-Yen Yang, Justinian Rosca, Karthik Narasimhan, Peter J. Ramadge*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/yang21i.html](http://proceedings.mlr.press/v139/yang21i.html)

**Abstract**:

We consider the problem of reinforcement learning when provided with (1) a baseline control policy and (2) a set of constraints that the learner must satisfy. The baseline policy can arise from demonstration data or a teacher agent and may provide useful cues for learning, but it might also be sub-optimal for the task at hand, and is not guaranteed to satisfy the specified constraints, which might encode safety, fairness or other application-specific requirements. In order to safely learn from baseline policies, we propose an iterative policy optimization algorithm that alternates between maximizing expected return on the task, minimizing distance to the baseline policy, and projecting the policy onto the constraint-satisfying set. We analyze our algorithm theoretically and provide a finite-time convergence guarantee. In our experiments on five different control tasks, our algorithm consistently outperforms several state-of-the-art baselines, achieving 10 times fewer constraint violations and 40% higher reward on average.

----

## [1073] Voice2Series: Reprogramming Acoustic Models for Time Series Classification

**Authors**: *Chao-Han Huck Yang, Yun-Yun Tsai, Pin-Yu Chen*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/yang21j.html](http://proceedings.mlr.press/v139/yang21j.html)

**Abstract**:

Learning to classify time series with limited data is a practical yet challenging problem. Current methods are primarily based on hand-designed feature extraction rules or domain-specific data augmentation. Motivated by the advances in deep speech processing models and the fact that voice data are univariate temporal signals, in this paper we propose Voice2Serie (V2S), a novel end-to-end approach that reprograms acoustic models for time series classification, through input transformation learning and output label mapping. Leveraging the representation learning power of a large-scale pre-trained speech processing model, on 31 different time series tasks we show that V2S outperforms or is on part with state-of-the-art methods on 22 tasks, and improves their average accuracy by 1.72%. We further provide theoretical justification of V2S by proving its population risk is upper bounded by the source risk and a Wasserstein distance accounting for feature alignment via reprogramming. Our results offer new and effective means to time series classification.

----

## [1074] When All We Need is a Piece of the Pie: A Generic Framework for Optimizing Two-way Partial AUC

**Authors**: *Zhiyong Yang, Qianqian Xu, Shilong Bao, Yuan He, Xiaochun Cao, Qingming Huang*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/yang21k.html](http://proceedings.mlr.press/v139/yang21k.html)

**Abstract**:

The Area Under the ROC Curve (AUC) is a crucial metric for machine learning, which evaluates the average performance over all possible True Positive Rates (TPRs) and False Positive Rates (FPRs). Based on the knowledge that a skillful classifier should simultaneously embrace a high TPR and a low FPR, we turn to study a more general variant called Two-way Partial AUC (TPAUC), where only the region with $\mathsf{TPR} \ge \alpha, \mathsf{FPR} \le \beta$ is included in the area. Moreover, a recent work shows that the TPAUC is essentially inconsistent with the existing Partial AUC metrics where only the FPR range is restricted, opening a new problem to seek solutions to leverage high TPAUC. Motivated by this, we present the first trial in this paper to optimize this new metric. The critical challenge along this course lies in the difficulty of performing gradient-based optimization with end-to-end stochastic training, even with a proper choice of surrogate loss. To address this issue, we propose a generic framework to construct surrogate optimization problems, which supports efficient end-to-end training with deep-learning. Moreover, our theoretical analyses show that: 1) the objective function of the surrogate problems will achieve an upper bound of the original problem under mild conditions, and 2) optimizing the surrogate problems leads to good generalization performance in terms of TPAUC with a high probability. Finally, empirical studies over several benchmark datasets speak to the efficacy of our framework.

----

## [1075] Rethinking Rotated Object Detection with Gaussian Wasserstein Distance Loss

**Authors**: *Xue Yang, Junchi Yan, Qi Ming, Wentao Wang, Xiaopeng Zhang, Qi Tian*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/yang21l.html](http://proceedings.mlr.press/v139/yang21l.html)

**Abstract**:

Boundary discontinuity and its inconsistency to the final detection metric have been the bottleneck for rotating detection regression loss design. In this paper, we propose a novel regression loss based on Gaussian Wasserstein distance as a fundamental approach to solve the problem. Specifically, the rotated bounding box is converted to a 2-D Gaussian distribution, which enables to approximate the indifferentiable rotational IoU induced loss by the Gaussian Wasserstein distance (GWD) which can be learned efficiently by gradient back-propagation. GWD can still be informative for learning even there is no overlapping between two rotating bounding boxes which is often the case for small object detection. Thanks to its three unique properties, GWD can also elegantly solve the boundary discontinuity and square-like problem regardless how the bounding box is defined. Experiments on five datasets using different detectors show the effectiveness of our approach, and codes are available at https://github.com/yangxue0827/RotationDetection.

----

## [1076] Delving into Deep Imbalanced Regression

**Authors**: *Yuzhe Yang, Kaiwen Zha, Ying-Cong Chen, Hao Wang, Dina Katabi*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/yang21m.html](http://proceedings.mlr.press/v139/yang21m.html)

**Abstract**:

Real-world data often exhibit imbalanced distributions, where certain target values have significantly fewer observations. Existing techniques for dealing with imbalanced data focus on targets with categorical indices, i.e., different classes. However, many tasks involve continuous targets, where hard boundaries between classes do not exist. We define Deep Imbalanced Regression (DIR) as learning from such imbalanced data with continuous targets, dealing with potential missing data for certain target values, and generalizing to the entire target range. Motivated by the intrinsic difference between categorical and continuous label space, we propose distribution smoothing for both labels and features, which explicitly acknowledges the effects of nearby targets, and calibrates both label and learned feature distributions. We curate and benchmark large-scale DIR datasets from common real-world tasks in computer vision, natural language processing, and healthcare domains. Extensive experiments verify the superior performance of our strategies. Our work fills the gap in benchmarks and techniques for practical imbalanced regression problems. Code and data are available at: https://github.com/YyzHarry/imbalanced-regression.

----

## [1077] Backpropagated Neighborhood Aggregation for Accurate Training of Spiking Neural Networks

**Authors**: *Yukun Yang, Wenrui Zhang, Peng Li*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/yang21n.html](http://proceedings.mlr.press/v139/yang21n.html)

**Abstract**:

While Backpropagation (BP) has been applied to spiking neural networks (SNNs) achieving encouraging results, a key challenge involved is to backpropagate a differentiable continuous-valued loss over layers of spiking neurons exhibiting discontinuous all-or-none firing activities. Existing methods deal with this difficulty by introducing compromises that come with their own limitations, leading to potential performance degradation. We propose a novel BP-like method, called neighborhood aggregation (NA), which computes accurate error gradients guiding weight updates that may lead to discontinuous modifications of firing activities. NA achieves this goal by aggregating the error gradient over multiple spike trains in the neighborhood of the present spike train of each neuron. The employed aggregation is based on a generalized finite difference approximation with a proposed distance metric quantifying the similarity between a given pair of spike trains. Our experiments show that the proposed NA algorithm delivers state-of-the-art performance for SNN training on several datasets including CIFAR10.

----

## [1078] SimAM: A Simple, Parameter-Free Attention Module for Convolutional Neural Networks

**Authors**: *Lingxiao Yang, Ru-Yuan Zhang, Lida Li, Xiaohua Xie*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/yang21o.html](http://proceedings.mlr.press/v139/yang21o.html)

**Abstract**:

In this paper, we propose a conceptually simple but very effective attention module for Convolutional Neural Networks (ConvNets). In contrast to existing channel-wise and spatial-wise attention modules, our module instead infers 3-D attention weights for the feature map in a layer without adding parameters to the original networks. Specifically, we base on some well-known neuroscience theories and propose to optimize an energy function to find the importance of each neuron. We further derive a fast closed-form solution for the energy function, and show that the solution can be implemented in less than ten lines of code. Another advantage of the module is that most of the operators are selected based on the solution to the defined energy function, avoiding too many efforts for structure tuning. Quantitative evaluations on various visual tasks demonstrate that the proposed module is flexible and effective to improve the representation ability of many ConvNets. Our code is available at Pytorch-SimAM.

----

## [1079] HAWQ-V3: Dyadic Neural Network Quantization

**Authors**: *Zhewei Yao, Zhen Dong, Zhangcheng Zheng, Amir Gholami, Jiali Yu, Eric Tan, Leyuan Wang, Qijing Huang, Yida Wang, Michael W. Mahoney, Kurt Keutzer*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/yao21a.html](http://proceedings.mlr.press/v139/yao21a.html)

**Abstract**:

Current low-precision quantization algorithms often have the hidden cost of conversion back and forth from floating point to quantized integer values. This hidden cost limits the latency improvement realized by quantizing Neural Networks. To address this, we present HAWQ-V3, a novel mixed-precision integer-only quantization framework. The contributions of HAWQ-V3 are the following: (i) An integer-only inference where the entire computational graph is performed only with integer multiplication, addition, and bit shifting, without any floating point operations or even integer division; (ii) A novel hardware-aware mixed-precision quantization method where the bit-precision is calculated by solving an integer linear programming problem that balances the trade-off between model perturbation and other constraints, e.g., memory footprint and latency; (iii) Direct hardware deployment and open source contribution for 4-bit uniform/mixed-precision quantization in TVM, achieving an average speed up of 1.45x for uniform 4-bit, as compared to uniform 8-bit for ResNet50 on T4 GPUs; and (iv) extensive evaluation of the proposed methods on ResNet18/50 and InceptionV3, for various model compression levels with/without mixed precision. For ResNet50, our INT8 quantization achieves an accuracy of 77.58%, which is 2.68% higher than prior integer-only work, and our mixed-precision INT4/8 quantization can reduce INT8 latency by 23% and still achieve 76.73% accuracy. Our framework and the TVM implementation have been open sourced (HAWQ, 2020).

----

## [1080] Improving Generalization in Meta-learning via Task Augmentation

**Authors**: *Huaxiu Yao, Long-Kai Huang, Linjun Zhang, Ying Wei, Li Tian, James Zou, Junzhou Huang, Zhenhui Li*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/yao21b.html](http://proceedings.mlr.press/v139/yao21b.html)

**Abstract**:

Meta-learning has proven to be a powerful paradigm for transferring the knowledge from previous tasks to facilitate the learning of a novel task. Current dominant algorithms train a well-generalized model initialization which is adapted to each task via the support set. The crux lies in optimizing the generalization capability of the initialization, which is measured by the performance of the adapted model on the query set of each task. Unfortunately, this generalization measure, evidenced by empirical results, pushes the initialization to overfit the meta-training tasks, which significantly impairs the generalization and adaptation to novel tasks. To address this issue, we actively augment a meta-training task with “more data” when evaluating the generalization. Concretely, we propose two task augmentation methods, including MetaMix and Channel Shuffle. MetaMix linearly combines features and labels of samples from both the support and query sets. For each class of samples, Channel Shuffle randomly replaces a subset of their channels with the corresponding ones from a different class. Theoretical studies show how task augmentation improves the generalization of meta-learning. Moreover, both MetaMix and Channel Shuffle outperform state-of-the-art results by a large margin across many datasets and are compatible with existing meta-learning algorithms.

----

## [1081] Deep Learning for Functional Data Analysis with Adaptive Basis Layers

**Authors**: *Junwen Yao, Jonas Mueller, Jane-Ling Wang*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/yao21c.html](http://proceedings.mlr.press/v139/yao21c.html)

**Abstract**:

Despite their widespread success, the application of deep neural networks to functional data remains scarce today. The infinite dimensionality of functional data means standard learning algorithms can be applied only after appropriate dimension reduction, typically achieved via basis expansions. Currently, these bases are chosen a priori without the information for the task at hand and thus may not be effective for the designated task. We instead propose to adaptively learn these bases in an end-to-end fashion. We introduce neural networks that employ a new Basis Layer whose hidden units are each basis functions themselves implemented as a micro neural network. Our architecture learns to apply parsimonious dimension reduction to functional inputs that focuses only on information relevant to the target rather than irrelevant variation in the input function. Across numerous classification/regression tasks with functional data, our method empirically outperforms other types of neural networks, and we prove that our approach is statistically consistent with low generalization error.

----

## [1082] Addressing Catastrophic Forgetting in Few-Shot Problems

**Authors**: *Pau Ching Yap, Hippolyt Ritter, David Barber*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/yap21a.html](http://proceedings.mlr.press/v139/yap21a.html)

**Abstract**:

Neural networks are known to suffer from catastrophic forgetting when trained on sequential datasets. While there have been numerous attempts to solve this problem in large-scale supervised classification, little has been done to overcome catastrophic forgetting in few-shot classification problems. We demonstrate that the popular gradient-based model-agnostic meta-learning algorithm (MAML) indeed suffers from catastrophic forgetting and introduce a Bayesian online meta-learning framework that tackles this problem. Our framework utilises Bayesian online learning and meta-learning along with Laplace approximation and variational inference to overcome catastrophic forgetting in few-shot classification problems. The experimental evaluations demonstrate that our framework can effectively achieve this goal in comparison with various baselines. As an additional utility, we also demonstrate empirically that our framework is capable of meta-learning on sequentially arriving few-shot tasks from a stationary task distribution.

----

## [1083] Reinforcement Learning with Prototypical Representations

**Authors**: *Denis Yarats, Rob Fergus, Alessandro Lazaric, Lerrel Pinto*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/yarats21a.html](http://proceedings.mlr.press/v139/yarats21a.html)

**Abstract**:

Learning effective representations in image-based environments is crucial for sample efficient Reinforcement Learning (RL). Unfortunately, in RL, representation learning is confounded with the exploratory experience of the agent – learning a useful representation requires diverse data, while effective exploration is only possible with coherent representations. Furthermore, we would like to learn representations that not only generalize across tasks but also accelerate downstream exploration for efficient task-specific training. To address these challenges we propose Proto-RL, a self-supervised framework that ties representation learning with exploration through prototypical representations. These prototypes simultaneously serve as a summarization of the exploratory experience of an agent as well as a basis for representing observations. We pre-train these task-agnostic representations and prototypes on environments without downstream task information. This enables state-of-the-art downstream policy learning on a set of difficult continuous control tasks.

----

## [1084] Elementary superexpressive activations

**Authors**: *Dmitry Yarotsky*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/yarotsky21a.html](http://proceedings.mlr.press/v139/yarotsky21a.html)

**Abstract**:

We call a finite family of activation functions \emph{superexpressive} if any multivariate continuous function can be approximated by a neural network that uses these activations and has a fixed architecture only depending on the number of input variables (i.e., to achieve any accuracy we only need to adjust the weights, without increasing the number of neurons). Previously, it was known that superexpressive activations exist, but their form was quite complex. We give examples of very simple superexpressive families: for example, we prove that the family $\{sin, arcsin\}$ is superexpressive. We also show that most practical activations (not involving periodic functions) are not superexpressive.

----

## [1085] Break-It-Fix-It: Unsupervised Learning for Program Repair

**Authors**: *Michihiro Yasunaga, Percy Liang*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/yasunaga21a.html](http://proceedings.mlr.press/v139/yasunaga21a.html)

**Abstract**:

We consider repair tasks: given a critic (e.g., compiler) that assesses the quality of an input, the goal is to train a fixer that converts a bad example (e.g., code with syntax errors) into a good one (e.g., code with no errors). Existing works create training data consisting of (bad, good) pairs by corrupting good examples using heuristics (e.g., dropping tokens). However, fixers trained on this synthetically-generated data do not extrapolate well to the real distribution of bad inputs. To bridge this gap, we propose a new training approach, Break-It-Fix-It (BIFI), which has two key ideas: (i) we use the critic to check a fixer’s output on real bad inputs and add good (fixed) outputs to the training data, and (ii) we train a breaker to generate realistic bad code from good code. Based on these ideas, we iteratively update the breaker and the fixer while using them in conjunction to generate more paired data. We evaluate BIFI on two code repair datasets: GitHub-Python, a new dataset we introduce where the goal is to repair Python code with AST parse errors; and DeepFix, where the goal is to repair C code with compiler errors. BIFI outperforms existing methods, obtaining 90.5% repair accuracy on GitHub-Python (+28.5%) and 71.7% on DeepFix (+5.6%). Notably, BIFI does not require any labeled data; we hope it will be a strong starting point for unsupervised learning of various repair tasks.

----

## [1086] Improving Gradient Regularization using Complex-Valued Neural Networks

**Authors**: *Eric C. Yeats, Yiran Chen, Hai Li*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/yeats21a.html](http://proceedings.mlr.press/v139/yeats21a.html)

**Abstract**:

Gradient regularization is a neural network defense technique that requires no prior knowledge of an adversarial attack and that brings only limited increase in training computational complexity. A form of complex-valued neural network (CVNN) is proposed to improve the performance of gradient regularization on classification tasks of real-valued input in adversarial settings. The activation derivatives of each layer of the CVNN are dependent on the combination of inputs to the layer, and locally stable representations can be learned for inputs the network is trained on. Furthermore, the properties of the CVNN parameter derivatives resist decrease of performance on the standard objective that is caused by competition with the gradient regularization objective. Experimental results show that the performance of gradient regularized CVNN surpasses that of real-valued neural networks with comparable storage and computational complexity. Moreover, gradient regularized complex-valued networks exhibit robust performance approaching that of real-valued networks trained with multi-step adversarial training.

----

## [1087] Neighborhood Contrastive Learning Applied to Online Patient Monitoring

**Authors**: *Hugo Yèche, Gideon Dresdner, Francesco Locatello, Matthias Hüser, Gunnar Rätsch*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/yeche21a.html](http://proceedings.mlr.press/v139/yeche21a.html)

**Abstract**:

Intensive care units (ICU) are increasingly looking towards machine learning for methods to provide online monitoring of critically ill patients. In machine learning, online monitoring is often formulated as a supervised learning problem. Recently, contrastive learning approaches have demonstrated promising improvements over competitive supervised benchmarks. These methods rely on well-understood data augmentation techniques developed for image data which do not apply to online monitoring. In this work, we overcome this limitation by supplementing time-series data augmentation techniques with a novel contrastive learning objective which we call neighborhood contrastive learning (NCL). Our objective explicitly groups together contiguous time segments from each patient while maintaining state-specific information. Our experiments demonstrate a marked improvement over existing work applying contrastive methods to medical time-series.

----

## [1088] From Local Structures to Size Generalization in Graph Neural Networks

**Authors**: *Gilad Yehudai, Ethan Fetaya, Eli A. Meirom, Gal Chechik, Haggai Maron*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/yehudai21a.html](http://proceedings.mlr.press/v139/yehudai21a.html)

**Abstract**:

Graph neural networks (GNNs) can process graphs of different sizes, but their ability to generalize across sizes, specifically from small to large graphs, is still not well understood. In this paper, we identify an important type of data where generalization from small to large graphs is challenging: graph distributions for which the local structure depends on the graph size. This effect occurs in multiple important graph learning domains, including social and biological networks. We first prove that when there is a difference between the local structures, GNNs are not guaranteed to generalize across sizes: there are "bad" global minima that do well on small graphs but fail on large graphs. We then study the size-generalization problem empirically and demonstrate that when there is a discrepancy in local structure, GNNs tend to converge to non-generalizing solutions. Finally, we suggest two approaches for improving size generalization, motivated by our findings. Notably, we propose a novel Self-Supervised Learning (SSL) task aimed at learning meaningful representations of local structures that appear in large graphs. Our SSL task improves classification accuracy on several popular datasets.

----

## [1089] Improved OOD Generalization via Adversarial Training and Pretraing

**Authors**: *Mingyang Yi, Lu Hou, Jiacheng Sun, Lifeng Shang, Xin Jiang, Qun Liu, Zhiming Ma*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/yi21a.html](http://proceedings.mlr.press/v139/yi21a.html)

**Abstract**:

Recently, learning a model that generalizes well on out-of-distribution (OOD) data has attracted great attention in the machine learning community. In this paper, after defining OOD generalization by Wasserstein distance, we theoretically justify that a model robust to input perturbation also generalizes well on OOD data. Inspired by previous findings that adversarial training helps improve robustness, we show that models trained by adversarial training have converged excess risk on OOD data. Besides, in the paradigm of pre-training then fine-tuning, we theoretically justify that the input perturbation robust model in the pre-training stage provides an initialization that generalizes well on downstream OOD data. Finally, various experiments conducted on image classification and natural language understanding tasks verify our theoretical findings.

----

## [1090] Regret and Cumulative Constraint Violation Analysis for Online Convex Optimization with Long Term Constraints

**Authors**: *Xinlei Yi, Xiuxian Li, Tao Yang, Lihua Xie, Tianyou Chai, Karl Henrik Johansson*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/yi21b.html](http://proceedings.mlr.press/v139/yi21b.html)

**Abstract**:

This paper considers online convex optimization with long term constraints, where constraints can be violated in intermediate rounds, but need to be satisfied in the long run. The cumulative constraint violation is used as the metric to measure constraint violations, which excludes the situation that strictly feasible constraints can compensate the effects of violated constraints. A novel algorithm is first proposed and it achieves an $\mathcal{O}(T^{\max\{c,1-c\}})$ bound for static regret and an $\mathcal{O}(T^{(1-c)/2})$ bound for cumulative constraint violation, where $c\in(0,1)$ is a user-defined trade-off parameter, and thus has improved performance compared with existing results. Both static regret and cumulative constraint violation bounds are reduced to $\mathcal{O}(\log(T))$ when the loss functions are strongly convex, which also improves existing results. %In order to bound the regret with respect to any comparator sequence, In order to achieve the optimal regret with respect to any comparator sequence, another algorithm is then proposed and it achieves the optimal $\mathcal{O}(\sqrt{T(1+P_T)})$ regret and an $\mathcal{O}(\sqrt{T})$ cumulative constraint violation, where $P_T$ is the path-length of the comparator sequence. Finally, numerical simulations are provided to illustrate the effectiveness of the theoretical results.

----

## [1091] Continuous-time Model-based Reinforcement Learning

**Authors**: *Çagatay Yildiz, Markus Heinonen, Harri Lähdesmäki*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/yildiz21a.html](http://proceedings.mlr.press/v139/yildiz21a.html)

**Abstract**:

Model-based reinforcement learning (MBRL) approaches rely on discrete-time state transition models whereas physical systems and the vast majority of control tasks operate in continuous-time. To avoid time-discretization approximation of the underlying process, we propose a continuous-time MBRL framework based on a novel actor-critic method. Our approach also infers the unknown state evolution differentials with Bayesian neural ordinary differential equations (ODE) to account for epistemic uncertainty. We implement and test our method on a new ODE-RL suite that explicitly solves continuous-time control systems. Our experiments illustrate that the model is robust against irregular and noisy data, and can solve classic control problems in a sample-efficient manner.

----

## [1092] Distributed Nyström Kernel Learning with Communications

**Authors**: *Rong Yin, Yong Liu, Weiping Wang, Dan Meng*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/yin21a.html](http://proceedings.mlr.press/v139/yin21a.html)

**Abstract**:

We study the statistical performance for distributed kernel ridge regression with Nyström (DKRR-NY) and with Nyström and iterative solvers (DKRR-NY-PCG) and successfully derive the optimal learning rates, which can improve the ranges of the number of local processors $p$ to the optimal in existing state-of-art bounds. More precisely, our theoretical analysis show that DKRR-NY and DKRR-NY-PCG achieve the same learning rates as the exact KRR requiring essentially $\mathcal{O}(|D|^{1.5})$ time and $\mathcal{O}(|D|)$ memory with relaxing the restriction on $p$ in expectation, where $|D|$ is the number of data, which exhibits the average effectiveness of multiple trials. Furthermore, for showing the generalization performance in a single trial, we deduce the learning rates for DKRR-NY and DKRR-NY-PCG in probability. Finally, we propose a novel algorithm DKRR-NY-CM based on DKRR-NY, which employs a communication strategy to further improve the learning performance, whose effectiveness of communications is validated in theoretical and experimental analysis.

----

## [1093] Path Planning using Neural A* Search

**Authors**: *Ryo Yonetani, Tatsunori Taniai, Mohammadamin Barekatain, Mai Nishimura, Asako Kanezaki*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/yonetani21a.html](http://proceedings.mlr.press/v139/yonetani21a.html)

**Abstract**:

We present Neural A*, a novel data-driven search method for path planning problems. Despite the recent increasing attention to data-driven path planning, machine learning approaches to search-based planning are still challenging due to the discrete nature of search algorithms. In this work, we reformulate a canonical A* search algorithm to be differentiable and couple it with a convolutional encoder to form an end-to-end trainable neural network planner. Neural A* solves a path planning problem by encoding a problem instance to a guidance map and then performing the differentiable A* search with the guidance map. By learning to match the search results with ground-truth paths provided by experts, Neural A* can produce a path consistent with the ground truth accurately and efficiently. Our extensive experiments confirmed that Neural A* outperformed state-of-the-art data-driven planners in terms of the search optimality and efficiency trade-off. Furthermore, Neural A* successfully predicted realistic human trajectories by directly performing search-based planning on natural image inputs.

----

## [1094] SinIR: Efficient General Image Manipulation with Single Image Reconstruction

**Authors**: *Jihyeong Yoo, Qifeng Chen*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/yoo21a.html](http://proceedings.mlr.press/v139/yoo21a.html)

**Abstract**:

We propose SinIR, an efficient reconstruction-based framework trained on a single natural image for general image manipulation, including super-resolution, editing, harmonization, paint-to-image, photo-realistic style transfer, and artistic style transfer. We train our model on a single image with cascaded multi-scale learning, where each network at each scale is responsible for image reconstruction. This reconstruction objective greatly reduces the complexity and running time of training, compared to the GAN objective. However, the reconstruction objective also exacerbates the output quality. Therefore, to solve this problem, we further utilize simple random pixel shuffling, which also gives control over manipulation, inspired by the Denoising Autoencoder. With quantitative evaluation, we show that SinIR has competitive performance on various image manipulation tasks. Moreover, with a much simpler training objective (i.e., reconstruction), SinIR is trained 33.5 times faster than SinGAN (for 500x500 images) that solves similar tasks. Our code is publicly available at github.com/YooJiHyeong/SinIR.

----

## [1095] Conditional Temporal Neural Processes with Covariance Loss

**Authors**: *Boseon Yoo, Jiwoo Lee, Janghoon Ju, Seijun Chung, Soyeon Kim, Jaesik Choi*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/yoo21b.html](http://proceedings.mlr.press/v139/yoo21b.html)

**Abstract**:

We introduce a novel loss function, Covariance Loss, which is conceptually equivalent to conditional neural processes and has a form of regularization so that is applicable to many kinds of neural networks. With the proposed loss, mappings from input variables to target variables are highly affected by dependencies of target variables as well as mean activation and mean dependencies of input and target variables. This nature enables the resulting neural networks to become more robust to noisy observations and recapture missing dependencies from prior information. In order to show the validity of the proposed loss, we conduct extensive sets of experiments on real-world datasets with state-of-the-art models and discuss the benefits and drawbacks of the proposed Covariance Loss.

----

## [1096] Adversarial Purification with Score-based Generative Models

**Authors**: *Jongmin Yoon, Sung Ju Hwang, Juho Lee*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/yoon21a.html](http://proceedings.mlr.press/v139/yoon21a.html)

**Abstract**:

While adversarial training is considered as a standard defense method against adversarial attacks for image classifiers, adversarial purification, which purifies attacked images into clean images with a standalone purification, model has shown promises as an alternative defense method. Recently, an EBM trained with MCMC has been highlighted as a purification model, where an attacked image is purified by running a long Markov-chain using the gradients of the EBM. Yet, the practicality of the adversarial purification using an EBM remains questionable because the number of MCMC steps required for such purification is too large. In this paper, we propose a novel adversarial purification method based on an EBM trained with DSM. We show that an EBM trained with DSM can quickly purify attacked images within a few steps. We further introduce a simple yet effective randomized purification scheme that injects random noises into images before purification. This process screens the adversarial perturbations imposed on images by the random noises and brings the images to the regime where the EBM can denoise well. We show that our purification method is robust against various attacks and demonstrate its state-of-the-art performances.

----

## [1097] Federated Continual Learning with Weighted Inter-client Transfer

**Authors**: *Jaehong Yoon, Wonyong Jeong, Giwoong Lee, Eunho Yang, Sung Ju Hwang*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/yoon21b.html](http://proceedings.mlr.press/v139/yoon21b.html)

**Abstract**:

There has been a surge of interest in continual learning and federated learning, both of which are important in deep neural networks in real-world scenarios. Yet little research has been done regarding the scenario where each client learns on a sequence of tasks from a private local data stream. This problem of federated continual learning poses new challenges to continual learning, such as utilizing knowledge from other clients, while preventing interference from irrelevant knowledge. To resolve these issues, we propose a novel federated continual learning framework, Federated Weighted Inter-client Transfer (FedWeIT), which decomposes the network weights into global federated parameters and sparse task-specific parameters, and each client receives selective knowledge from other clients by taking a weighted combination of their task-specific parameters. FedWeIT minimizes interference between incompatible tasks, and also allows positive knowledge transfer across clients during learning. We validate our FedWeIT against existing federated learning and continual learning methods under varying degrees of task similarity across clients, and our model significantly outperforms them with a large reduction in the communication cost.

----

## [1098] Autoencoding Under Normalization Constraints

**Authors**: *Sangwoong Yoon, Yung-Kyun Noh, Frank Chongwoo Park*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/yoon21c.html](http://proceedings.mlr.press/v139/yoon21c.html)

**Abstract**:

Likelihood is a standard estimate for outlier detection. The specific role of the normalization constraint is to ensure that the out-of-distribution (OOD) regime has a small likelihood when samples are learned using maximum likelihood. Because autoencoders do not possess such a process of normalization, they often fail to recognize outliers even when they are obviously OOD. We propose the Normalized Autoencoder (NAE), a normalized probabilistic model constructed from an autoencoder. The probability density of NAE is defined using the reconstruction error of an autoencoder, which is differently defined in the conventional energy-based model. In our model, normalization is enforced by suppressing the reconstruction of negative samples, significantly improving the outlier detection performance. Our experimental results confirm the efficacy of NAE, both in detecting outliers and in generating in-distribution samples.

----

## [1099] Accelerated Algorithms for Smooth Convex-Concave Minimax Problems with O(1/k^2) Rate on Squared Gradient Norm

**Authors**: *Taeho Yoon, Ernest K. Ryu*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/yoon21d.html](http://proceedings.mlr.press/v139/yoon21d.html)

**Abstract**:

In this work, we study the computational complexity of reducing the squared gradient magnitude for smooth minimax optimization problems. First, we present algorithms with accelerated $\mathcal{O}(1/k^2)$ last-iterate rates, faster than the existing $\mathcal{O}(1/k)$ or slower rates for extragradient, Popov, and gradient descent with anchoring. The acceleration mechanism combines extragradient steps with anchoring and is distinct from Nesterov’s acceleration. We then establish optimality of the $\mathcal{O}(1/k^2)$ rate through a matching lower bound.

----

## [1100] Lower-Bounded Proper Losses for Weakly Supervised Classification

**Authors**: *Shuhei M. Yoshida, Takashi Takenouchi, Masashi Sugiyama*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/yoshida21a.html](http://proceedings.mlr.press/v139/yoshida21a.html)

**Abstract**:

This paper discusses the problem of weakly supervised classification, in which instances are given weak labels that are produced by some label-corruption process. The goal is to derive conditions under which loss functions for weak-label learning are proper and lower-bounded—two essential requirements for the losses used in class-probability estimation. To this end, we derive a representation theorem for proper losses in supervised learning, which dualizes the Savage representation. We use this theorem to characterize proper weak-label losses and find a condition for them to be lower-bounded. From these theoretical findings, we derive a novel regularization scheme called generalized logit squeezing, which makes any proper weak-label loss bounded from below, without losing properness. Furthermore, we experimentally demonstrate the effectiveness of our proposed approach, as compared to improper or unbounded losses. The results highlight the importance of properness and lower-boundedness.

----

## [1101] Graph Contrastive Learning Automated

**Authors**: *Yuning You, Tianlong Chen, Yang Shen, Zhangyang Wang*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/you21a.html](http://proceedings.mlr.press/v139/you21a.html)

**Abstract**:

Self-supervised learning on graph-structured data has drawn recent interest for learning generalizable, transferable and robust representations from unlabeled graphs. Among many, graph contrastive learning (GraphCL) has emerged with promising representation learning performance. Unfortunately, unlike its counterpart on image data, the effectiveness of GraphCL hinges on ad-hoc data augmentations, which have to be manually picked per dataset, by either rules of thumb or trial-and-errors, owing to the diverse nature of graph data. That significantly limits the more general applicability of GraphCL. Aiming to fill in this crucial gap, this paper proposes a unified bi-level optimization framework to automatically, adaptively and dynamically select data augmentations when performing GraphCL on specific graph data. The general framework, dubbed JOint Augmentation Optimization (JOAO), is instantiated as min-max optimization. The selections of augmentations made by JOAO are shown to be in general aligned with previous "best practices" observed from handcrafted tuning: yet now being automated, more flexible and versatile. Moreover, we propose a new augmentation-aware projection head mechanism, which will route output features through different projection heads corresponding to different augmentations chosen at each training step. Extensive experiments demonstrate that JOAO performs on par with or sometimes better than the state-of-the-art competitors including GraphCL, on multiple graph datasets of various scales and types, yet without resorting to any laborious dataset-specific tuning on augmentation selection. We release the code at https://github.com/Shen-Lab/GraphCL_Automated.

----

## [1102] LogME: Practical Assessment of Pre-trained Models for Transfer Learning

**Authors**: *Kaichao You, Yong Liu, Jianmin Wang, Mingsheng Long*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/you21b.html](http://proceedings.mlr.press/v139/you21b.html)

**Abstract**:

This paper studies task adaptive pre-trained model selection, an underexplored problem of assessing pre-trained models for the target task and select best ones from the model zoo \emph{without fine-tuning}. A few pilot works addressed the problem in transferring supervised pre-trained models to classification tasks, but they cannot handle emerging unsupervised pre-trained models or regression tasks. In pursuit of a practical assessment method, we propose to estimate the maximum value of label evidence given features extracted by pre-trained models. Unlike the maximum likelihood, the maximum evidence is \emph{immune to over-fitting}, while its expensive computation can be dramatically reduced by our carefully designed algorithm. The Logarithm of Maximum Evidence (LogME) can be used to assess pre-trained models for transfer learning: a pre-trained model with a high LogME value is likely to have good transfer performance. LogME is \emph{fast, accurate, and general}, characterizing itself as the first practical method for assessing pre-trained models. Compared with brute-force fine-tuning, LogME brings at most $3000\times$ speedup in wall-clock time and requires only $1%$ memory footprint. It outperforms prior methods by a large margin in their setting and is applicable to new settings. It is general enough for diverse pre-trained models (supervised pre-trained and unsupervised pre-trained), downstream tasks (classification and regression), and modalities (vision and language). Code is available at this repository: \href{https://github.com/thuml/LogME}{https://github.com/thuml/LogME}.

----

## [1103] Exponentially Many Local Minima in Quantum Neural Networks

**Authors**: *Xuchen You, Xiaodi Wu*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/you21c.html](http://proceedings.mlr.press/v139/you21c.html)

**Abstract**:

Quantum Neural Networks (QNNs), or the so-called variational quantum circuits, are important quantum applications both because of their similar promises as classical neural networks and because of the feasibility of their implementation on near-term intermediate-size noisy quantum machines (NISQ). However, the training task of QNNs is challenging and much less understood. We conduct a quantitative investigation on the landscape of loss functions of QNNs and identify a class of simple yet extremely hard QNN instances for training. Specifically, we show for typical under-parameterized QNNs, there exists a dataset that induces a loss function with the number of spurious local minima depending exponentially on the number of parameters. Moreover, we show the optimality of our construction by providing an almost matching upper bound on such dependence. While local minima in classical neural networks are due to non-linear activations, in quantum neural networks local minima appear as a result of the quantum interference phenomenon. Finally, we empirically confirm that our constructions can indeed be hard instances in practice with typical gradient-based optimizers, which demonstrates the practical value of our findings.

----

## [1104] DAGs with No Curl: An Efficient DAG Structure Learning Approach

**Authors**: *Yue Yu, Tian Gao, Naiyu Yin, Qiang Ji*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/yu21a.html](http://proceedings.mlr.press/v139/yu21a.html)

**Abstract**:

Recently directed acyclic graph (DAG) structure learning is formulated as a constrained continuous optimization problem with continuous acyclicity constraints and was solved iteratively through subproblem optimization. To further improve efficiency, we propose a novel learning framework to model and learn the weighted adjacency matrices in the DAG space directly. Specifically, we first show that the set of weighted adjacency matrices of DAGs are equivalent to the set of weighted gradients of graph potential functions, and one may perform structure learning by searching in this equivalent set of DAGs. To instantiate this idea, we propose a new algorithm, DAG-NoCurl, which solves the optimization problem efficiently with a two-step procedure: $1)$ first we find an initial non-acyclic solution to the optimization problem, and $2)$ then we employ the Hodge decomposition of graphs and learn an acyclic graph by projecting the non-acyclic graph to the gradient of a potential function. Experimental studies on benchmark datasets demonstrate that our method provides comparable accuracy but better efficiency than baseline DAG structure learning methods on both linear and generalized structural equation models, often by more than one order of magnitude.

----

## [1105] Provably Efficient Algorithms for Multi-Objective Competitive RL

**Authors**: *Tiancheng Yu, Yi Tian, Jingzhao Zhang, Suvrit Sra*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/yu21b.html](http://proceedings.mlr.press/v139/yu21b.html)

**Abstract**:

We study multi-objective reinforcement learning (RL) where an agent’s reward is represented as a vector. In settings where an agent competes against opponents, its performance is measured by the distance of its average return vector to a target set. We develop statistically and computationally efficient algorithms to approach the associated target set. Our results extend Blackwell’s approachability theorem \citep{blackwell1956analog} to tabular RL, where strategic exploration becomes essential. The algorithms presented are adaptive; their guarantees hold even without Blackwell’s approachability condition. If the opponents use fixed policies, we give an improved rate of approaching the target set while also tackling the more ambitious goal of simultaneously minimizing a scalar cost function. We discuss our analysis for this special case by relating our results to previous works on constrained RL. To our knowledge, this work provides the first provably efficient algorithms for vector-valued Markov games and our theoretical guarantees are near-optimal.

----

## [1106] Whittle Networks: A Deep Likelihood Model for Time Series

**Authors**: *Zhongjie Yu, Fabrizio Ventola, Kristian Kersting*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/yu21c.html](http://proceedings.mlr.press/v139/yu21c.html)

**Abstract**:

While probabilistic circuits have been extensively explored for tabular data, less attention has been paid to time series. Here, the goal is to estimate joint densities among the entire time series and, in turn, determining, for instance, conditional independence relations between them. To this end, we propose the first probabilistic circuits (PCs) approach for modeling the joint distribution of multivariate time series, called Whittle sum-product networks (WSPNs). WSPNs leverage the Whittle approximation, casting the likelihood in the frequency domain, and place a complex-valued sum-product network, the most prominent PC, over the frequencies. The conditional independence relations among the time series can then be determined efficiently in the spectral domain. Moreover, WSPNs can naturally be placed into the deep neural learning stack for time series, resulting in Whittle Networks, opening the likelihood toolbox for training deep neural models and inspecting their behaviour. Our experiments show that Whittle Networks can indeed capture complex dependencies between time series and provide a useful measure of uncertainty for neural networks.

----

## [1107] Deep Latent Graph Matching

**Authors**: *Tianshu Yu, Runzhong Wang, Junchi Yan, Baoxin Li*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/yu21d.html](http://proceedings.mlr.press/v139/yu21d.html)

**Abstract**:

Deep learning for graph matching (GM) has emerged as an important research topic due to its superior performance over traditional methods and insights it provides for solving other combinatorial problems on graph. While recent deep methods for GM extensively investigated effective node/edge feature learning or downstream GM solvers given such learned features, there is little existing work questioning if the fixed connectivity/topology typically constructed using heuristics (e.g., Delaunay or k-nearest) is indeed suitable for GM. From a learning perspective, we argue that the fixed topology may restrict the model capacity and thus potentially hinder the performance. To address this, we propose to learn the (distribution of) latent topology, which can better support the downstream GM task. We devise two latent graph generation procedures, one deterministic and one generative. Particularly, the generative procedure emphasizes the across-graph consistency and thus can be viewed as a matching-guided co-generative model. Our methods deliver superior performance over previous state-of-the-arts on public benchmarks, hence supporting our hypothesis.

----

## [1108] Learning Generalized Intersection Over Union for Dense Pixelwise Prediction

**Authors**: *Jiaqian Yu, Jingtao Xu, Yiwei Chen, Weiming Li, Qiang Wang, ByungIn Yoo, Jae-Joon Han*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/yu21e.html](http://proceedings.mlr.press/v139/yu21e.html)

**Abstract**:

Intersection over union (IoU) score, also named Jaccard Index, is one of the most fundamental evaluation methods in machine learning. The original IoU computation cannot provide non-zero gradients and thus cannot be directly optimized by nowadays deep learning methods. Several recent works generalized IoU for bounding box regression, but they are not straightforward to adapt for pixelwise prediction. In particular, the original IoU fails to provide effective gradients for the non-overlapping and location-deviation cases, which results in performance plateau. In this paper, we propose PixIoU, a generalized IoU for pixelwise prediction that is sensitive to the distance for non-overlapping cases and the locations in prediction. We provide proofs that PixIoU holds many nice properties as the original IoU. To optimize the PixIoU, we also propose a loss function that is proved to be submodular, hence we can apply the Lovász functions, the efficient surrogates for submodular functions for learning this loss. Experimental results show consistent performance improvements by learning PixIoU over the original IoU for several different pixelwise prediction tasks on Pascal VOC, VOT-2020 and Cityscapes.

----

## [1109] Large Scale Private Learning via Low-rank Reparametrization

**Authors**: *Da Yu, Huishuai Zhang, Wei Chen, Jian Yin, Tie-Yan Liu*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/yu21f.html](http://proceedings.mlr.press/v139/yu21f.html)

**Abstract**:

We propose a reparametrization scheme to address the challenges of applying differentially private SGD on large neural networks, which are 1) the huge memory cost of storing individual gradients, 2) the added noise suffering notorious dimensional dependence. Specifically, we reparametrize each weight matrix with two \emph{gradient-carrier} matrices of small dimension and a \emph{residual weight} matrix. We argue that such reparametrization keeps the forward/backward process unchanged while enabling us to compute the projected gradient without computing the gradient itself. To learn with differential privacy, we design \emph{reparametrized gradient perturbation (RGP)} that perturbs the gradients on gradient-carrier matrices and reconstructs an update for the original weight from the noisy gradients. Importantly, we use historical updates to find the gradient-carrier matrices, whose optimality is rigorously justified under linear regression and empirically verified with deep learning tasks. RGP significantly reduces the memory cost and improves the utility. For example, we are the first able to apply differential privacy on the BERT model and achieve an average accuracy of $83.9%$ on four downstream tasks with $\epsilon=8$, which is within $5%$ loss compared to the non-private baseline but enjoys much lower privacy leakage risk.

----

## [1110] Federated Deep AUC Maximization for Hetergeneous Data with a Constant Communication Complexity

**Authors**: *Zhuoning Yuan, Zhishuai Guo, Yi Xu, Yiming Ying, Tianbao Yang*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/yuan21a.html](http://proceedings.mlr.press/v139/yuan21a.html)

**Abstract**:

Deep AUC (area under the ROC curve) Maximization (DAM) has attracted much attention recently due to its great potential for imbalanced data classification. However, the research on Federated Deep AUC Maximization (FDAM) is still limited. Compared with standard federated learning (FL) approaches that focus on decomposable minimization objectives, FDAM is more complicated due to its minimization objective is non-decomposable over individual examples. In this paper, we propose improved FDAM algorithms for heterogeneous data by solving the popular non-convex strongly-concave min-max formulation of DAM in a distributed fashion, which can also be applied to a class of non-convex strongly-concave min-max problems. A striking result of this paper is that the communication complexity of the proposed algorithm is a constant independent of the number of machines and also independent of the accuracy level, which improves an existing result by orders of magnitude. The experiments have demonstrated the effectiveness of our FDAM algorithm on benchmark datasets, and on medical chest X-ray images from different organizations. Our experiment shows that the performance of FDAM using data from multiple hospitals can improve the AUC score on testing data from a single hospital for detecting life-threatening diseases based on chest radiographs.

----

## [1111] Neural Tangent Generalization Attacks

**Authors**: *Chia-Hung Yuan, Shan-Hung Wu*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/yuan21b.html](http://proceedings.mlr.press/v139/yuan21b.html)

**Abstract**:

The remarkable performance achieved by Deep Neural Networks (DNNs) in many applications is followed by the rising concern about data privacy and security. Since DNNs usually require large datasets to train, many practitioners scrape data from external sources such as the Internet. However, an external data owner may not be willing to let this happen, causing legal or ethical issues. In this paper, we study the generalization attacks against DNNs, where an attacker aims to slightly modify training data in order to spoil the training process such that a trained network lacks generalizability. These attacks can be performed by data owners and protect data from unexpected use. However, there is currently no efficient generalization attack against DNNs due to the complexity of a bilevel optimization involved. We propose the Neural Tangent Generalization Attack (NTGA) that, to the best of our knowledge, is the first work enabling clean-label, black-box generalization attack against DNNs. We conduct extensive experiments, and the empirical results demonstrate the effectiveness of NTGA. Our code and perturbed datasets are available at: https://github.com/lionelmessi6410/ntga.

----

## [1112] On Explainability of Graph Neural Networks via Subgraph Explorations

**Authors**: *Hao Yuan, Haiyang Yu, Jie Wang, Kang Li, Shuiwang Ji*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/yuan21c.html](http://proceedings.mlr.press/v139/yuan21c.html)

**Abstract**:

We consider the problem of explaining the predictions of graph neural networks (GNNs), which otherwise are considered as black boxes. Existing methods invariably focus on explaining the importance of graph nodes or edges but ignore the substructures of graphs, which are more intuitive and human-intelligible. In this work, we propose a novel method, known as SubgraphX, to explain GNNs by identifying important subgraphs. Given a trained GNN model and an input graph, our SubgraphX explains its predictions by efficiently exploring different subgraphs with Monte Carlo tree search. To make the tree search more effective, we propose to use Shapley values as a measure of subgraph importance, which can also capture the interactions among different subgraphs. To expedite computations, we propose efficient approximation schemes to compute Shapley values for graph data. Our work represents the first attempt to explain GNNs via identifying subgraphs explicitly and directly. Experimental results show that our SubgraphX achieves significantly improved explanations, while keeping computations at a reasonable level.

----

## [1113] Federated Composite Optimization

**Authors**: *Honglin Yuan, Manzil Zaheer, Sashank J. Reddi*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/yuan21d.html](http://proceedings.mlr.press/v139/yuan21d.html)

**Abstract**:

Federated Learning (FL) is a distributed learning paradigm that scales on-device learning collaboratively and privately. Standard FL algorithms such as FEDAVG are primarily geared towards smooth unconstrained settings. In this paper, we study the Federated Composite Optimization (FCO) problem, in which the loss function contains a non-smooth regularizer. Such problems arise naturally in FL applications that involve sparsity, low-rank, monotonicity, or more general constraints. We first show that straightforward extensions of primal algorithms such as FedAvg are not well-suited for FCO since they suffer from the "curse of primal averaging," resulting in poor convergence. As a solution, we propose a new primal-dual algorithm, Federated Dual Averaging (FedDualAvg), which by employing a novel server dual averaging procedure circumvents the curse of primal averaging. Our theoretical analysis and empirical experiments demonstrate that FedDualAvg outperforms the other baselines.

----

## [1114] Three Operator Splitting with a Nonconvex Loss Function

**Authors**: *Alp Yurtsever, Varun Mangalick, Suvrit Sra*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/yurtsever21a.html](http://proceedings.mlr.press/v139/yurtsever21a.html)

**Abstract**:

We consider the problem of minimizing the sum of three functions, one of which is nonconvex but differentiable, and the other two are convex but possibly nondifferentiable. We investigate the Three Operator Splitting method (TOS) of Davis & Yin (2017) with an aim to extend its theoretical guarantees for this nonconvex problem template. In particular, we prove convergence of TOS with nonasymptotic bounds on its nonstationarity and infeasibility errors. In contrast with the existing work on nonconvex TOS, our guarantees do not require additional smoothness assumptions on the terms comprising the objective; hence they cover instances of particular interest where the nondifferentiable terms are indicator functions. We also extend our results to a stochastic setting where we have access only to an unbiased estimator of the gradient. Finally, we illustrate the effectiveness of the proposed method through numerical experiments on quadratic assignment problems.

----

## [1115] Grey-box Extraction of Natural Language Models

**Authors**: *Santiago Zanella Béguelin, Shruti Tople, Andrew Paverd, Boris Köpf*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/zanella-beguelin21a.html](http://proceedings.mlr.press/v139/zanella-beguelin21a.html)

**Abstract**:

Model extraction attacks attempt to replicate a target machine learning model by querying its inference API. State-of-the-art attacks are learning-based and construct replicas by supervised training on the target model’s predictions, but an emerging class of attacks exploit algebraic properties to obtain high-fidelity replicas using orders of magnitude fewer queries. So far, these algebraic attacks have been limited to neural networks with few hidden layers and ReLU activations. In this paper we present algebraic and hybrid algebraic/learning-based attacks on large-scale natural language models. We consider a grey-box setting, targeting models with a pre-trained (public) encoder followed by a single (private) classification layer. Our key findings are that (i) with a frozen encoder, high-fidelity extraction is possible with a small number of in-distribution queries, making extraction attacks indistinguishable from legitimate use; (ii) when the encoder is fine-tuned, a hybrid learning-based/algebraic attack improves over the learning-based state-of-the-art without requiring additional queries.

----

## [1116] Exponential Lower Bounds for Batch Reinforcement Learning: Batch RL can be Exponentially Harder than Online RL

**Authors**: *Andrea Zanette*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/zanette21a.html](http://proceedings.mlr.press/v139/zanette21a.html)

**Abstract**:

Several practical applications of reinforcement learning involve an agent learning from past data without the possibility of further exploration. Often these applications require us to 1) identify a near optimal policy or to 2) estimate the value of a target policy. For both tasks we derive exponential information-theoretic lower bounds in discounted infinite horizon MDPs with a linear function representation for the action value function even if 1) realizability holds, 2) the batch algorithm observes the exact reward and transition functions, and 3) the batch algorithm is given the best a priori data distribution for the problem class. Our work introduces a new ‘oracle + batch algorithm’ framework to prove lower bounds that hold for every distribution. The work shows an exponential separation between batch and online reinforcement learning.

----

## [1117] Learning Binary Decision Trees by Argmin Differentiation

**Authors**: *Valentina Zantedeschi, Matt J. Kusner, Vlad Niculae*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/zantedeschi21a.html](http://proceedings.mlr.press/v139/zantedeschi21a.html)

**Abstract**:

We address the problem of learning binary decision trees that partition data for some downstream task. We propose to learn discrete parameters (i.e., for tree traversals and node pruning) and continuous parameters (i.e., for tree split functions and prediction functions) simultaneously using argmin differentiation. We do so by sparsely relaxing a mixed-integer program for the discrete parameters, to allow gradients to pass through the program to continuous parameters. We derive customized algorithms to efficiently compute the forward and backward passes. This means that our tree learning procedure can be used as an (implicit) layer in arbitrary deep networks, and can be optimized with arbitrary loss functions. We demonstrate that our approach produces binary trees that are competitive with existing single tree and ensemble approaches, in both supervised and unsupervised settings. Further, apart from greedy approaches (which do not have competitive accuracies), our method is faster to train than all other tree-learning baselines we compare with.

----

## [1118] Barlow Twins: Self-Supervised Learning via Redundancy Reduction

**Authors**: *Jure Zbontar, Li Jing, Ishan Misra, Yann LeCun, Stéphane Deny*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/zbontar21a.html](http://proceedings.mlr.press/v139/zbontar21a.html)

**Abstract**:

Self-supervised learning (SSL) is rapidly closing the gap with supervised methods on large computer vision benchmarks. A successful approach to SSL is to learn embeddings which are invariant to distortions of the input sample. However, a recurring issue with this approach is the existence of trivial constant solutions. Most current methods avoid such solutions by careful implementation details. We propose an objective function that naturally avoids collapse by measuring the cross-correlation matrix between the outputs of two identical networks fed with distorted versions of a sample, and making it as close to the identity matrix as possible. This causes the embedding vectors of distorted versions of a sample to be similar, while minimizing the redundancy between the components of these vectors. The method is called Barlow Twins, owing to neuroscientist H. Barlow’s redundancy-reduction principle applied to a pair of identical networks. Barlow Twins does not require large batches nor asymmetry between the network twins such as a predictor network, gradient stopping, or a moving average on the weight updates. Intriguingly it benefits from very high-dimensional output vectors. Barlow Twins outperforms previous methods on ImageNet for semi-supervised classification in the low-data regime, and is on par with current state of the art for ImageNet classification with a linear classifier head, and for transfer tasks of classification and object detection.

----

## [1119] You Only Sample (Almost) Once: Linear Cost Self-Attention Via Bernoulli Sampling

**Authors**: *Zhanpeng Zeng, Yunyang Xiong, Sathya N. Ravi, Shailesh Acharya, Glenn Moo Fung, Vikas Singh*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/zeng21a.html](http://proceedings.mlr.press/v139/zeng21a.html)

**Abstract**:

Transformer-based models are widely used in natural language processing (NLP). Central to the transformer model is the self-attention mechanism, which captures the interactions of token pairs in the input sequences and depends quadratically on the sequence length. Training such models on longer sequences is expensive. In this paper, we show that a Bernoulli sampling attention mechanism based on Locality Sensitive Hashing (LSH), decreases the quadratic complexity of such models to linear. We bypass the quadratic cost by considering self-attention as a sum of individual tokens associated with Bernoulli random variables that can, in principle, be sampled at once by a single hash (although in practice, this number may be a small constant). This leads to an efficient sampling scheme to estimate self-attention which relies on specific modifications of LSH (to enable deployment on GPU architectures). We evaluate our algorithm on the GLUE benchmark with standard 512 sequence length where we see favorable performance relative to a standard pretrained Transformer. On the Long Range Arena (LRA) benchmark, for evaluating performance on long sequences, our method achieves results consistent with softmax self-attention but with sizable speed-ups and memory savings and often outperforms other efficient self-attention methods. Our code is available at https://github.com/mlpen/YOSO.

----

## [1120] DouZero: Mastering DouDizhu with Self-Play Deep Reinforcement Learning

**Authors**: *Daochen Zha, Jingru Xie, Wenye Ma, Sheng Zhang, Xiangru Lian, Xia Hu, Ji Liu*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/zha21a.html](http://proceedings.mlr.press/v139/zha21a.html)

**Abstract**:

Games are abstractions of the real world, where artificial agents learn to compete and cooperate with other agents. While significant achievements have been made in various perfect- and imperfect-information games, DouDizhu (a.k.a. Fighting the Landlord), a three-player card game, is still unsolved. DouDizhu is a very challenging domain with competition, collaboration, imperfect information, large state space, and particularly a massive set of possible actions where the legal actions vary significantly from turn to turn. Unfortunately, modern reinforcement learning algorithms mainly focus on simple and small action spaces, and not surprisingly, are shown not to make satisfactory progress in DouDizhu. In this work, we propose a conceptually simple yet effective DouDizhu AI system, namely DouZero, which enhances traditional Monte-Carlo methods with deep neural networks, action encoding, and parallel actors. Starting from scratch in a single server with four GPUs, DouZero outperformed all the existing DouDizhu AI programs in days of training and was ranked the first in the Botzone leaderboard among 344 AI agents. Through building DouZero, we show that classic Monte-Carlo methods can be made to deliver strong results in a hard domain with a complex action space. The code and an online demo are released at https://github.com/kwai/DouZero with the hope that this insight could motivate future work.

----

## [1121] DORO: Distributional and Outlier Robust Optimization

**Authors**: *Runtian Zhai, Chen Dan, J. Zico Kolter, Pradeep Ravikumar*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/zhai21a.html](http://proceedings.mlr.press/v139/zhai21a.html)

**Abstract**:

Many machine learning tasks involve subpopulation shift where the testing data distribution is a subpopulation of the training distribution. For such settings, a line of recent work has proposed the use of a variant of empirical risk minimization(ERM) known as distributionally robust optimization (DRO). In this work, we apply DRO to real, large-scale tasks with subpopulation shift, and observe that DRO performs relatively poorly, and moreover has severe instability. We identify one direct cause of this phenomenon: sensitivity of DRO to outliers in the datasets. To resolve this issue, we propose the framework of DORO, for Distributional and Outlier Robust Optimization. At the core of this approach is a refined risk function which prevents DRO from overfitting to potential outliers. We instantiate DORO for the Cressie-Read family of Rényi divergence, and delve into two specific instances of this family: CVaR and $\chi^2$-DRO. We theoretically prove the effectiveness of the proposed method, and empirically show that DORO improves the performance and stability of DRO with experiments on large modern datasets, thereby positively addressing the open question raised by Hashimoto et al., 2018. Codes are available at https://github.com/RuntianZ/doro.

----

## [1122] Can Subnetwork Structure Be the Key to Out-of-Distribution Generalization?

**Authors**: *Dinghuai Zhang, Kartik Ahuja, Yilun Xu, Yisen Wang, Aaron C. Courville*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/zhang21a.html](http://proceedings.mlr.press/v139/zhang21a.html)

**Abstract**:

Can models with particular structure avoid being biased towards spurious correlation in out-of-distribution (OOD) generalization? Peters et al. (2016) provides a positive answer for linear cases. In this paper, we use a functional modular probing method to analyze deep model structures under OOD setting. We demonstrate that even in biased models (which focus on spurious correlation) there still exist unbiased functional subnetworks. Furthermore, we articulate and confirm the functional lottery ticket hypothesis: the full network contains a subnetwork with proper structure that can achieve better OOD performance. We then propose Modular Risk Minimization to solve the subnetwork selection problem. Our algorithm learns the functional structure from a given dataset, and can be combined with any other OOD regularization methods. Experiments on various OOD generalization tasks corroborate the effectiveness of our method.

----

## [1123] Towards Certifying L-infinity Robustness using Neural Networks with L-inf-dist Neurons

**Authors**: *Bohang Zhang, Tianle Cai, Zhou Lu, Di He, Liwei Wang*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/zhang21b.html](http://proceedings.mlr.press/v139/zhang21b.html)

**Abstract**:

It is well-known that standard neural networks, even with a high classification accuracy, are vulnerable to small $\ell_\infty$-norm bounded adversarial perturbations. Although many attempts have been made, most previous works either can only provide empirical verification of the defense to a particular attack method, or can only develop a certified guarantee of the model robustness in limited scenarios. In this paper, we seek for a new approach to develop a theoretically principled neural network that inherently resists $\ell_\infty$ perturbations. In particular, we design a novel neuron that uses $\ell_\infty$-distance as its basic operation (which we call $\ell_\infty$-dist neuron), and show that any neural network constructed with $\ell_\infty$-dist neurons (called $\ell_{\infty}$-dist net) is naturally a 1-Lipschitz function with respect to $\ell_\infty$-norm. This directly provides a rigorous guarantee of the certified robustness based on the margin of prediction outputs. We then prove that such networks have enough expressive power to approximate any 1-Lipschitz function with robust generalization guarantee. We further provide a holistic training strategy that can greatly alleviate optimization difficulties. Experimental results show that using $\ell_{\infty}$-dist nets as basic building blocks, we consistently achieve state-of-the-art performance on commonly used datasets: 93.09% certified accuracy on MNIST ($\epsilon=0.3$), 35.42% on CIFAR-10 ($\epsilon=8/255$) and 16.31% on TinyImageNet ($\epsilon=1/255$).

----

## [1124] Efficient Lottery Ticket Finding: Less Data is More

**Authors**: *Zhenyu Zhang, Xuxi Chen, Tianlong Chen, Zhangyang Wang*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/zhang21c.html](http://proceedings.mlr.press/v139/zhang21c.html)

**Abstract**:

The lottery ticket hypothesis (LTH) reveals the existence of winning tickets (sparse but critical subnetworks) for dense networks, that can be trained in isolation from random initialization to match the latter’s accuracies. However, finding winning tickets requires burdensome computations in the train-prune-retrain process, especially on large-scale datasets (e.g., ImageNet), restricting their practical benefits. This paper explores a new perspective on finding lottery tickets more efficiently, by doing so only with a specially selected subset of data, called Pruning-Aware Critical set (PrAC set), rather than using the full training set. The concept of PrAC set was inspired by the recent observation, that deep networks have samples that are either hard to memorize during training, or easy to forget during pruning. A PrAC set is thus hypothesized to capture those most challenging and informative examples for the dense model. We observe that a high-quality winning ticket can be found with training and pruning the dense network on the very compact PrAC set, which can substantially save training iterations for the ticket finding process. Extensive experiments validate our proposal across diverse datasets and network architectures. Specifically, on CIFAR-10, CIFAR-100, and Tiny ImageNet, we locate effective PrAC sets at 35.32% 78.19% of their training set sizes. On top of them, we can obtain the same competitive winning tickets for the corresponding dense networks, yet saving up to 82.85% 92.77%, 63.54% 74.92%, and 76.14% 86.56% training iterations, respectively. Crucially, we show that a PrAC set found is reusable across different network architectures, which can amortize the extra cost of finding PrAC sets, yielding a practical regime for efficient lottery ticket finding.

----

## [1125] Robust Policy Gradient against Strong Data Corruption

**Authors**: *Xuezhou Zhang, Yiding Chen, Xiaojin Zhu, Wen Sun*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/zhang21d.html](http://proceedings.mlr.press/v139/zhang21d.html)

**Abstract**:

We study the problem of robust reinforcement learning under adversarial corruption on both rewards and transitions. Our attack model assumes an \textit{adaptive} adversary who can arbitrarily corrupt the reward and transition at every step within an episode, for at most $\epsilon$-fraction of the learning episodes. Our attack model is strictly stronger than those considered in prior works. Our first result shows that no algorithm can find a better than $O(\epsilon)$-optimal policy under our attack model. Next, we show that surprisingly the natural policy gradient (NPG) method retains a natural robustness property if the reward corruption is bounded, and can find an $O(\sqrt{\epsilon})$-optimal policy. Consequently, we develop a Filtered Policy Gradient (FPG) algorithm that can tolerate even unbounded reward corruption and can find an $O(\epsilon^{1/4})$-optimal policy. We emphasize that FPG is the first that can achieve a meaningful learning guarantee when a constant fraction of episodes are corrupted. Complimentary to the theoretical results, we show that a neural implementation of FPG achieves strong robust learning performance on the MuJoCo continuous control benchmarks.

----

## [1126] Near Optimal Reward-Free Reinforcement Learning

**Authors**: *Zihan Zhang, Simon S. Du, Xiangyang Ji*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/zhang21e.html](http://proceedings.mlr.press/v139/zhang21e.html)

**Abstract**:

We study the reward-free reinforcement learning framework, which is particularly suitable for batch reinforcement learning and scenarios where one needs policies for multiple reward functions. This framework has two phases: in the exploration phase, the agent collects trajectories by interacting with the environment without using any reward signal; in the planning phase, the agent needs to return a near-optimal policy for arbitrary reward functions. %This framework is suitable for batch RL setting and the setting where there are multiple reward functions of interes We give a new efficient algorithm, \textbf{S}taged \textbf{S}ampling + \textbf{T}runcated \textbf{P}lanning (\algoname), which interacts with the environment at most $O\left( \frac{S^2A}{\epsilon^2}\poly\log\left(\frac{SAH}{\epsilon}\right) \right)$ episodes in the exploration phase, and guarantees to output a near-optimal policy for arbitrary reward functions in the planning phase, where $S$ is the size of state space, $A$ is the size of action space, $H$ is the planning horizon, and $\epsilon$ is the target accuracy relative to the total reward. Notably, our sample complexity scales only \emph{logarithmically} with $H$, in contrast to all existing results which scale \emph{polynomially} with $H$. Furthermore, this bound matches the minimax lower bound $\Omega\left(\frac{S^2A}{\epsilon^2}\right)$ up to logarithmic factors. Our results rely on three new techniques : 1) A new sufficient condition for the dataset to plan for an $\epsilon$-suboptimal policy % for any totally bounded reward function ; 2) A new way to plan efficiently under the proposed condition using soft-truncated planning; 3) Constructing extended MDP to maximize the truncated accumulative rewards efficiently.

----

## [1127] Bayesian Attention Belief Networks

**Authors**: *Shujian Zhang, Xinjie Fan, Bo Chen, Mingyuan Zhou*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/zhang21f.html](http://proceedings.mlr.press/v139/zhang21f.html)

**Abstract**:

Attention-based neural networks have achieved state-of-the-art results on a wide range of tasks. Most such models use deterministic attention while stochastic attention is less explored due to the optimization difficulties or complicated model design. This paper introduces Bayesian attention belief networks, which construct a decoder network by modeling unnormalized attention weights with a hierarchy of gamma distributions, and an encoder network by stacking Weibull distributions with a deterministic-upward-stochastic-downward structure to approximate the posterior. The resulting auto-encoding networks can be optimized in a differentiable way with a variational lower bound. It is simple to convert any models with deterministic attention, including pretrained ones, to the proposed Bayesian attention belief networks. On a variety of language understanding tasks, we show that our method outperforms deterministic attention and state-of-the-art stochastic attention in accuracy, uncertainty estimation, generalization across domains, and robustness to adversarial attacks. We further demonstrate the general applicability of our method on neural machine translation and visual question answering, showing great potential of incorporating our method into various attention-related tasks.

----

## [1128] Understanding Failures in Out-of-Distribution Detection with Deep Generative Models

**Authors**: *Lily H. Zhang, Mark Goldstein, Rajesh Ranganath*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/zhang21g.html](http://proceedings.mlr.press/v139/zhang21g.html)

**Abstract**:

Deep generative models (DGMs) seem a natural fit for detecting out-of-distribution (OOD) inputs, but such models have been shown to assign higher probabilities or densities to OOD images than images from the training distribution. In this work, we explain why this behavior should be attributed to model misestimation. We first prove that no method can guarantee performance beyond random chance without assumptions on which out-distributions are relevant. We then interrogate the typical set hypothesis, the claim that relevant out-distributions can lie in high likelihood regions of the data distribution, and that OOD detection should be defined based on the data distribution’s typical set. We highlight the consequences implied by assuming support overlap between in- and out-distributions, as well as the arbitrariness of the typical set for OOD detection. Our results suggest that estimation error is a more plausible explanation than the misalignment between likelihood-based OOD detection and out-distributions of interest, and we illustrate how even minimal estimation error can lead to OOD detection failures, yielding implications for future work in deep generative modeling and OOD detection.

----

## [1129] Poolingformer: Long Document Modeling with Pooling Attention

**Authors**: *Hang Zhang, Yeyun Gong, Yelong Shen, Weisheng Li, Jiancheng Lv, Nan Duan, Weizhu Chen*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/zhang21h.html](http://proceedings.mlr.press/v139/zhang21h.html)

**Abstract**:

In this paper, we introduce a two-level attention schema, Poolingformer, for long document modeling. Its first level uses a smaller sliding window pattern to aggregate information from neighbors. Its second level employs a larger window to increase receptive fields with pooling attention to reduce both computational cost and memory consumption. We first evaluate Poolingformer on two long sequence QA tasks: the monolingual NQ and the multilingual TyDi QA. Experimental results show that Poolingformer sits atop three official leaderboards measured by F1, outperforming previous state-of-the-art models by 1.9 points (79.8 vs. 77.9) on NQ long answer, 1.9 points (79.5 vs. 77.6) on TyDi QA passage answer, and 1.6 points (67.6 vs. 66.0) on TyDi QA minimal answer. We further evaluate Poolingformer on a long sequence summarization task. Experimental results on the arXiv benchmark continue to demonstrate its superior performance.

----

## [1130] Probabilistic Generating Circuits

**Authors**: *Honghua Zhang, Brendan Juba, Guy Van den Broeck*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/zhang21i.html](http://proceedings.mlr.press/v139/zhang21i.html)

**Abstract**:

Generating functions, which are widely used in combinatorics and probability theory, encode function values into the coefficients of a polynomial. In this paper, we explore their use as a tractable probabilistic model, and propose probabilistic generating circuits (PGCs) for their efficient representation. PGCs are strictly more expressive efficient than many existing tractable probabilistic models, including determinantal point processes (DPPs), probabilistic circuits (PCs) such as sum-product networks, and tractable graphical models. We contend that PGCs are not just a theoretical framework that unifies vastly different existing models, but also show great potential in modeling realistic data. We exhibit a simple class of PGCs that are not trivially subsumed by simple combinations of PCs and DPPs, and obtain competitive performance on a suite of density estimation benchmarks. We also highlight PGCs’ connection to the theory of strongly Rayleigh distributions.

----

## [1131] PAPRIKA: Private Online False Discovery Rate Control

**Authors**: *Wanrong Zhang, Gautam Kamath, Rachel Cummings*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/zhang21j.html](http://proceedings.mlr.press/v139/zhang21j.html)

**Abstract**:

In hypothesis testing, a \emph{false discovery} occurs when a hypothesis is incorrectly rejected due to noise in the sample. When adaptively testing multiple hypotheses, the probability of a false discovery increases as more tests are performed. Thus the problem of \emph{False Discovery Rate (FDR) control} is to find a procedure for testing multiple hypotheses that accounts for this effect in determining the set of hypotheses to reject. The goal is to minimize the number (or fraction) of false discoveries, while maintaining a high true positive rate (i.e., correct discoveries). In this work, we study False Discovery Rate (FDR) control in multiple hypothesis testing under the constraint of differential privacy for the sample. Unlike previous work in this direction, we focus on the \emph{online setting}, meaning that a decision about each hypothesis must be made immediately after the test is performed, rather than waiting for the output of all tests as in the offline setting. We provide new private algorithms based on state-of-the-art results in non-private online FDR control. Our algorithms have strong provable guarantees for privacy and statistical performance as measured by FDR and power. We also provide experimental results to demonstrate the efficacy of our algorithms in a variety of data environments.

----

## [1132] Learning from Noisy Labels with No Change to the Training Process

**Authors**: *Mingyuan Zhang, Jane Lee, Shivani Agarwal*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/zhang21k.html](http://proceedings.mlr.press/v139/zhang21k.html)

**Abstract**:

There has been much interest in recent years in developing learning algorithms that can learn accurate classifiers from data with noisy labels. A widely-studied noise model is that of \emph{class-conditional noise} (CCN), wherein a label $y$ is flipped to a label $\tilde{y}$ with some associated noise probability that depends on both $y$ and $\tilde{y}$. In the multiclass setting, all previously proposed algorithms under the CCN model involve changing the training process, by introducing a ‘noise-correction’ to the surrogate loss to be minimized over the noisy training examples. In this paper, we show that this is really unnecessary: one can simply perform class probability estimation (CPE) on the noisy examples, e.g. using a standard (multiclass) logistic regression algorithm, and then apply noise-correction only in the final prediction step. This means that the training algorithm itself does not need any change, and one can simply use standard off-the-shelf implementations with no modification to the code for training. Our approach can handle general multiclass loss matrices, including the usual 0-1 loss but also other losses such as those used for ordinal regression problems. We also provide a quantitative regret transfer bound, which bounds the target regret on the true distribution in terms of the CPE regret on the noisy distribution; in doing so, we extend the notion of strong properness introduced for binary losses by Agarwal (2014) to the multiclass case. Our bound suggests that the sample complexity of learning under CCN increases as the noise matrix approaches singularity. We also provide fixes and potential improvements for noise estimation methods that involve computing anchor points. Our experiments confirm our theoretical findings.

----

## [1133] Progressive-Scale Boundary Blackbox Attack via Projective Gradient Estimation

**Authors**: *Jiawei Zhang, Linyi Li, Huichen Li, Xiaolu Zhang, Shuang Yang, Bo Li*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/zhang21l.html](http://proceedings.mlr.press/v139/zhang21l.html)

**Abstract**:

Boundary based blackbox attack has been recognized as practical and effective, given that an attacker only needs to access the final model prediction. However, the query efficiency of it is in general high especially for high dimensional image data. In this paper, we show that such efficiency highly depends on the scale at which the attack is applied, and attacking at the optimal scale significantly improves the efficiency. In particular, we propose a theoretical framework to analyze and show three key characteristics to improve the query efficiency. We prove that there exists an optimal scale for projective gradient estimation. Our framework also explains the satisfactory performance achieved by existing boundary black-box attacks. Based on our theoretical framework, we propose Progressive-Scale enabled projective Boundary Attack (PSBA) to improve the query efficiency via progressive scaling techniques. In particular, we employ Progressive-GAN to optimize the scale of projections, which we call PSBA-PGAN. We evaluate our approach on both spatial and frequency scales. Extensive experiments on MNIST, CIFAR-10, CelebA, and ImageNet against different models including a real-world face recognition API show that PSBA-PGAN significantly outperforms existing baseline attacks in terms of query efficiency and attack success rate. We also observe relatively stable optimal scales for different models and datasets. The code is publicly available at https://github.com/AI-secure/PSBA.

----

## [1134] FOP: Factorizing Optimal Joint Policy of Maximum-Entropy Multi-Agent Reinforcement Learning

**Authors**: *Tianhao Zhang, Yueheng Li, Chen Wang, Guangming Xie, Zongqing Lu*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/zhang21m.html](http://proceedings.mlr.press/v139/zhang21m.html)

**Abstract**:

Value decomposition recently injects vigorous vitality into multi-agent actor-critic methods. However, existing decomposed actor-critic methods cannot guarantee the convergence of global optimum. In this paper, we present a novel multi-agent actor-critic method, FOP, which can factorize the optimal joint policy induced by maximum-entropy multi-agent reinforcement learning (MARL) into individual policies. Theoretically, we prove that factorized individual policies of FOP converge to the global optimum. Empirically, in the well-known matrix game and differential game, we verify that FOP can converge to the global optimum for both discrete and continuous action spaces. We also evaluate FOP on a set of StarCraft II micromanagement tasks, and demonstrate that FOP substantially outperforms state-of-the-art decomposed value-based and actor-critic methods.

----

## [1135] Learning Noise Transition Matrix from Only Noisy Labels via Total Variation Regularization

**Authors**: *Yivan Zhang, Gang Niu, Masashi Sugiyama*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/zhang21n.html](http://proceedings.mlr.press/v139/zhang21n.html)

**Abstract**:

Many weakly supervised classification methods employ a noise transition matrix to capture the class-conditional label corruption. To estimate the transition matrix from noisy data, existing methods often need to estimate the noisy class-posterior, which could be unreliable due to the overconfidence of neural networks. In this work, we propose a theoretically grounded method that can estimate the noise transition matrix and learn a classifier simultaneously, without relying on the error-prone noisy class-posterior estimation. Concretely, inspired by the characteristics of the stochastic label corruption process, we propose total variation regularization, which encourages the predicted probabilities to be more distinguishable from each other. Under mild assumptions, the proposed method yields a consistent estimator of the transition matrix. We show the effectiveness of the proposed method through experiments on benchmark and real-world datasets.

----

## [1136] Quantile Bandits for Best Arms Identification

**Authors**: *Mengyan Zhang, Cheng Soon Ong*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/zhang21o.html](http://proceedings.mlr.press/v139/zhang21o.html)

**Abstract**:

We consider a variant of the best arm identification task in stochastic multi-armed bandits. Motivated by risk-averse decision-making problems, our goal is to identify a set of $m$ arms with the highest $\tau$-quantile values within a fixed budget. We prove asymmetric two-sided concentration inequalities for order statistics and quantiles of random variables that have non-decreasing hazard rate, which may be of independent interest. With these inequalities, we analyse a quantile version of Successive Accepts and Rejects (Q-SAR). We derive an upper bound for the probability of arm misidentification, the first justification of a quantile based algorithm for fixed budget multiple best arms identification. We show illustrative experiments for best arm identification.

----

## [1137] Towards Better Robust Generalization with Shift Consistency Regularization

**Authors**: *Shufei Zhang, Zhuang Qian, Kaizhu Huang, Qiufeng Wang, Rui Zhang, Xinping Yi*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/zhang21p.html](http://proceedings.mlr.press/v139/zhang21p.html)

**Abstract**:

While adversarial training becomes one of the most promising defending approaches against adversarial attacks for deep neural networks, the conventional wisdom through robust optimization may usually not guarantee good generalization for robustness. Concerning with robust generalization over unseen adversarial data, this paper investigates adversarial training from a novel perspective of shift consistency in latent space. We argue that the poor robust generalization of adversarial training is owing to the significantly dispersed latent representations generated by training and test adversarial data, as the adversarial perturbations push the latent features of natural examples in the same class towards diverse directions. This is underpinned by the theoretical analysis of the robust generalization gap, which is upper-bounded by the standard one over the natural data and a term of feature inconsistent shift caused by adversarial perturbation {–} a measure of latent dispersion. Towards better robust generalization, we propose a new regularization method {–} shift consistency regularization (SCR) {–} to steer the same-class latent features of both natural and adversarial data into a common direction during adversarial training. The effectiveness of SCR in adversarial training is evaluated through extensive experiments over different datasets, such as CIFAR-10, CIFAR-100, and SVHN, against several competitive methods.

----

## [1138] On-Policy Deep Reinforcement Learning for the Average-Reward Criterion

**Authors**: *Yiming Zhang, Keith W. Ross*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/zhang21q.html](http://proceedings.mlr.press/v139/zhang21q.html)

**Abstract**:

We develop theory and algorithms for average-reward on-policy Reinforcement Learning (RL). We first consider bounding the difference of the long-term average reward for two policies. We show that previous work based on the discounted return (Schulman et al. 2015, Achiam et al. 2017) results in a non-meaningful lower bound in the average reward setting. By addressing the average-reward criterion directly, we then derive a novel bound which depends on the average divergence between the policies and on Kemeny’s constant. Based on this bound, we develop an iterative procedure which produces a sequence of monotonically improved policies for the average reward criterion. This iterative procedure can then be combined with classic Deep Reinforcement Learning (DRL) methods, resulting in practical DRL algorithms that target the long-run average reward criterion. In particular, we demonstrate that Average-Reward TRPO (ATRPO), which adapts the on-policy TRPO algorithm to the average-reward criterion, significantly outperforms TRPO in the most challenging MuJuCo environments.

----

## [1139] Differentiable Dynamic Quantization with Mixed Precision and Adaptive Resolution

**Authors**: *Zhaoyang Zhang, Wenqi Shao, Jinwei Gu, Xiaogang Wang, Ping Luo*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/zhang21r.html](http://proceedings.mlr.press/v139/zhang21r.html)

**Abstract**:

Model quantization is challenging due to many tedious hyper-parameters such as precision (bitwidth), dynamic range (minimum and maximum discrete values) and stepsize (interval between discrete values). Unlike prior arts that carefully tune these values, we present a fully differentiable approach to learn all of them, named Differentiable Dynamic Quantization (DDQ), which has several benefits. (1) DDQ is able to quantize challenging lightweight architectures like MobileNets, where different layers prefer different quantization parameters. (2) DDQ is hardware-friendly and can be easily implemented using low-precision matrix-vector multiplication, making it capable in many hardware such as ARM. (3) Extensive experiments show that DDQ outperforms prior arts on many networks and benchmarks, especially when models are already efficient and compact. e.g., DDQ is the first approach that achieves lossless 4-bit quantization for MobileNetV2 on ImageNet.

----

## [1140] iDARTS: Differentiable Architecture Search with Stochastic Implicit Gradients

**Authors**: *Miao Zhang, Steven W. Su, Shirui Pan, Xiaojun Chang, M. Ehsan Abbasnejad, Reza Haffari*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/zhang21s.html](http://proceedings.mlr.press/v139/zhang21s.html)

**Abstract**:

Differentiable ARchiTecture Search(DARTS) has recently become the mainstream in the neural architecture search (NAS) due to its efficiency and simplicity. With a gradient-based bi-level optimization, DARTS alternately optimizes the inner model weights and the outer architecture parameter in a weight-sharing supernet. A key challenge to the scalability and quality of the learned architectures is the need for differentiating through the inner-loop optimisation. While much has been discussed about several potentially fatal factors in DARTS, the architecture gradient, a.k.a. hypergradient, has received less attention. In this paper, we tackle the hypergradient computation in DARTS based on the implicit function theorem, making it only depends on the obtained solution to the inner-loop optimization and agnostic to the optimization path. To further reduce the computational requirements, we formulate a stochastic hypergradient approximation for differentiable NAS, and theoretically show that the architecture optimization with the proposed method is expected to converge to a stationary point. Comprehensive experiments on two NAS benchmark search spaces and the common NAS search space verify the effectiveness of our proposed method. It leads to architectures outperforming, with large margins, those learned by the baseline methods.

----

## [1141] Deep Coherent Exploration for Continuous Control

**Authors**: *Yijie Zhang, Herke van Hoof*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/zhang21t.html](http://proceedings.mlr.press/v139/zhang21t.html)

**Abstract**:

In policy search methods for reinforcement learning (RL), exploration is often performed by injecting noise either in action space at each step independently or in parameter space over each full trajectory. In prior work, it has been shown that with linear policies, a more balanced trade-off between these two exploration strategies is beneficial. However, that method did not scale to policies using deep neural networks. In this paper, we introduce deep coherent exploration, a general and scalable exploration framework for deep RL algorithms for continuous control, that generalizes step-based and trajectory-based exploration. This framework models the last layer parameters of the policy network as latent variables and uses a recursive inference step within the policy update to handle these latent variables in a scalable manner. We find that deep coherent exploration improves the speed and stability of learning of A2C, PPO, and SAC on several continuous control tasks.

----

## [1142] Average-Reward Off-Policy Policy Evaluation with Function Approximation

**Authors**: *Shangtong Zhang, Yi Wan, Richard S. Sutton, Shimon Whiteson*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/zhang21u.html](http://proceedings.mlr.press/v139/zhang21u.html)

**Abstract**:

We consider off-policy policy evaluation with function approximation (FA) in average-reward MDPs, where the goal is to estimate both the reward rate and the differential value function. For this problem, bootstrapping is necessary and, along with off-policy learning and FA, results in the deadly triad (Sutton & Barto, 2018). To address the deadly triad, we propose two novel algorithms, reproducing the celebrated success of Gradient TD algorithms in the average-reward setting. In terms of estimating the differential value function, the algorithms are the first convergent off-policy linear function approximation algorithms. In terms of estimating the reward rate, the algorithms are the first convergent off-policy linear function approximation algorithms that do not require estimating the density ratio. We demonstrate empirically the advantage of the proposed algorithms, as well as their nonlinear variants, over a competitive density-ratio-based approach, in a simple domain as well as challenging robot simulation tasks.

----

## [1143] Matrix Sketching for Secure Collaborative Machine Learning

**Authors**: *Mengjiao Zhang, Shusen Wang*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/zhang21v.html](http://proceedings.mlr.press/v139/zhang21v.html)

**Abstract**:

Collaborative learning allows participants to jointly train a model without data sharing. To update the model parameters, the central server broadcasts model parameters to the clients, and the clients send updating directions such as gradients to the server. While data do not leave a client device, the communicated gradients and parameters will leak a client’s privacy. Attacks that infer clients’ privacy from gradients and parameters have been developed by prior work. Simple defenses such as dropout and differential privacy either fail to defend the attacks or seriously hurt test accuracy. We propose a practical defense which we call Double-Blind Collaborative Learning (DBCL). The high-level idea is to apply random matrix sketching to the parameters (aka weights) and re-generate random sketching after each iteration. DBCL prevents clients from conducting gradient-based privacy inferences which are the most effective attacks. DBCL works because from the attacker’s perspective, sketching is effectively random noise that outweighs the signal. Notably, DBCL does not much increase computation and communication costs and does not hurt test accuracy at all.

----

## [1144] MetaCURE: Meta Reinforcement Learning with Empowerment-Driven Exploration

**Authors**: *Jin Zhang, Jianhao Wang, Hao Hu, Tong Chen, Yingfeng Chen, Changjie Fan, Chongjie Zhang*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/zhang21w.html](http://proceedings.mlr.press/v139/zhang21w.html)

**Abstract**:

Meta reinforcement learning (meta-RL) extracts knowledge from previous tasks and achieves fast adaptation to new tasks. Despite recent progress, efficient exploration in meta-RL remains a key challenge in sparse-reward tasks, as it requires quickly finding informative task-relevant experiences in both meta-training and adaptation. To address this challenge, we explicitly model an exploration policy learning problem for meta-RL, which is separated from exploitation policy learning, and introduce a novel empowerment-driven exploration objective, which aims to maximize information gain for task identification. We derive a corresponding intrinsic reward and develop a new off-policy meta-RL framework, which efficiently learns separate context-aware exploration and exploitation policies by sharing the knowledge of task inference. Experimental evaluation shows that our meta-RL method significantly outperforms state-of-the-art baselines on various sparse-reward MuJoCo locomotion tasks and more complex sparse-reward Meta-World tasks.

----

## [1145] World Model as a Graph: Learning Latent Landmarks for Planning

**Authors**: *Lunjun Zhang, Ge Yang, Bradly C. Stadie*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/zhang21x.html](http://proceedings.mlr.press/v139/zhang21x.html)

**Abstract**:

Planning, the ability to analyze the structure of a problem in the large and decompose it into interrelated subproblems, is a hallmark of human intelligence. While deep reinforcement learning (RL) has shown great promise for solving relatively straightforward control tasks, it remains an open problem how to best incorporate planning into existing deep RL paradigms to handle increasingly complex environments. One prominent framework, Model-Based RL, learns a world model and plans using step-by-step virtual rollouts. This type of world model quickly diverges from reality when the planning horizon increases, thus struggling at long-horizon planning. How can we learn world models that endow agents with the ability to do temporally extended reasoning? In this work, we propose to learn graph-structured world models composed of sparse, multi-step transitions. We devise a novel algorithm to learn latent landmarks that are scattered (in terms of reachability) across the goal space as the nodes on the graph. In this same graph, the edges are the reachability estimates distilled from Q-functions. On a variety of high-dimensional continuous control tasks ranging from robotic manipulation to navigation, we demonstrate that our method, named L3P, significantly outperforms prior work, and is oftentimes the only method capable of leveraging both the robustness of model-free RL and generalization of graph-search algorithms. We believe our work is an important step towards scalable planning in reinforcement learning.

----

## [1146] Breaking the Deadly Triad with a Target Network

**Authors**: *Shangtong Zhang, Hengshuai Yao, Shimon Whiteson*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/zhang21y.html](http://proceedings.mlr.press/v139/zhang21y.html)

**Abstract**:

The deadly triad refers to the instability of a reinforcement learning algorithm when it employs off-policy learning, function approximation, and bootstrapping simultaneously. In this paper, we investigate the target network as a tool for breaking the deadly triad, providing theoretical support for the conventional wisdom that a target network stabilizes training. We first propose and analyze a novel target network update rule which augments the commonly used Polyak-averaging style update with two projections. We then apply the target network and ridge regularization in several divergent algorithms and show their convergence to regularized TD fixed points. Those algorithms are off-policy with linear function approximation and bootstrapping, spanning both policy evaluation and control, as well as both discounted and average-reward settings. In particular, we provide the first convergent linear $Q$-learning algorithms under nonrestrictive and changing behavior policies without bi-level optimization.

----

## [1147] Multiscale Invertible Generative Networks for High-Dimensional Bayesian Inference

**Authors**: *Shumao Zhang, Pengchuan Zhang, Thomas Y. Hou*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/zhang21z.html](http://proceedings.mlr.press/v139/zhang21z.html)

**Abstract**:

We propose a Multiscale Invertible Generative Network (MsIGN) and associated training algorithm that leverages multiscale structure to solve high-dimensional Bayesian inference. To address the curse of dimensionality, MsIGN exploits the low-dimensional nature of the posterior, and generates samples from coarse to fine scale (low to high dimension) by iteratively upsampling and refining samples. MsIGN is trained in a multi-stage manner to minimize the Jeffreys divergence, which avoids mode dropping in high-dimensional cases. On two high-dimensional Bayesian inverse problems, we show superior performance of MsIGN over previous approaches in posterior approximation and multiple mode capture. On the natural image synthesis task, MsIGN achieves superior performance in bits-per-dimension over baseline models and yields great interpret-ability of its neurons in intermediate layers.

----

## [1148] Meta Learning for Support Recovery in High-dimensional Precision Matrix Estimation

**Authors**: *Qian Zhang, Yilin Zheng, Jean Honorio*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/zhang21aa.html](http://proceedings.mlr.press/v139/zhang21aa.html)

**Abstract**:

In this paper, we study meta learning for support (i.e., the set of non-zero entries) recovery in high-dimensional precision matrix estimation where we reduce the sufficient sample complexity in a novel task with the information learned from other auxiliary tasks. In our setup, each task has a different random true precision matrix, each with a possibly different support. We assume that the union of the supports of all the true precision matrices (i.e., the true support union) is small in size. We propose to pool all the samples from different tasks, and \emph{improperly} estimate a single precision matrix by minimizing the $\ell_1$-regularized log-determinant Bregman divergence. We show that with high probability, the support of the \emph{improperly} estimated single precision matrix is equal to the true support union, provided a sufficient number of samples per task $n \in O((\log N)/K)$, for $N$-dimensional vectors and $K$ tasks. That is, one requires less samples per task when more tasks are available. We prove a matching information-theoretic lower bound for the necessary number of samples, which is $n \in \Omega((\log N)/K)$, and thus, our algorithm is minimax optimal. Then for the novel task, we prove that the minimization of the $\ell_1$-regularized log-determinant Bregman divergence with the additional constraint that the support is a subset of the estimated support union could reduce the sufficient sample complexity of successful support recovery to $O(\log(|S_{\text{off}}|))$ where $|S_{\text{off}}|$ is the number of off-diagonal elements in the support union and is much less than $N$ for sparse matrices. We also prove a matching information-theoretic lower bound of $\Omega(\log(|S_{\text{off}}|))$ for the necessary number of samples.

----

## [1149] Model-Free Reinforcement Learning: from Clipped Pseudo-Regret to Sample Complexity

**Authors**: *Zihan Zhang, Yuan Zhou, Xiangyang Ji*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/zhang21ab.html](http://proceedings.mlr.press/v139/zhang21ab.html)

**Abstract**:

In this paper we consider the problem of learning an $\epsilon$-optimal policy for a discounted Markov Decision Process (MDP). Given an MDP with $S$ states, $A$ actions, the discount factor $\gamma \in (0,1)$, and an approximation threshold $\epsilon > 0$, we provide a model-free algorithm to learn an $\epsilon$-optimal policy with sample complexity $\tilde{O}(\frac{SA\ln(1/p)}{\epsilon^2(1-\gamma)^{5.5}})$ \footnote{In this work, the notation $\tilde{O}(\cdot)$ hides poly-logarithmic factors of $S,A,1/(1-\gamma)$, and $1/\epsilon$.} and success probability $(1-p)$. For small enough $\epsilon$, we show an improved algorithm with sample complexity $\tilde{O}(\frac{SA\ln(1/p)}{\epsilon^2(1-\gamma)^{3}})$. While the first bound improves upon all known model-free algorithms and model-based ones with tight dependence on $S$, our second algorithm beats all known sample complexity bounds and matches the information theoretic lower bound up to logarithmic factors.

----

## [1150] Learning to Rehearse in Long Sequence Memorization

**Authors**: *Zhu Zhang, Chang Zhou, Jianxin Ma, Zhijie Lin, Jingren Zhou, Hongxia Yang, Zhou Zhao*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/zhang21ac.html](http://proceedings.mlr.press/v139/zhang21ac.html)

**Abstract**:

Existing reasoning tasks often have an important assumption that the input contents can be always accessed while reasoning, requiring unlimited storage resources and suffering from severe time delay on long sequences. To achieve efficient reasoning on long sequences with limited storage resources, memory augmented neural networks introduce a human-like write-read memory to compress and memorize the long input sequence in one pass, trying to answer subsequent queries only based on the memory. But they have two serious drawbacks: 1) they continually update the memory from current information and inevitably forget the early contents; 2) they do not distinguish what information is important and treat all contents equally. In this paper, we propose the Rehearsal Memory (RM) to enhance long-sequence memorization by self-supervised rehearsal with a history sampler. To alleviate the gradual forgetting of early information, we design self-supervised rehearsal training with recollection and familiarity tasks. Further, we design a history sampler to select informative fragments for rehearsal training, making the memory focus on the crucial information. We evaluate the performance of our rehearsal memory by the synthetic bAbI task and several downstream tasks, including text/video question answering and recommendation on long sequences.

----

## [1151] Dataset Condensation with Differentiable Siamese Augmentation

**Authors**: *Bo Zhao, Hakan Bilen*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/zhao21a.html](http://proceedings.mlr.press/v139/zhao21a.html)

**Abstract**:

In many machine learning problems, large-scale datasets have become the de-facto standard to train state-of-the-art deep networks at the price of heavy computation load. In this paper, we focus on condensing large training sets into significantly smaller synthetic sets which can be used to train deep neural networks from scratch with minimum drop in performance. Inspired from the recent training set synthesis methods, we propose Differentiable Siamese Augmentation that enables effective use of data augmentation to synthesize more informative synthetic images and thus achieves better performance when training networks with augmentations. Experiments on multiple image classification benchmarks demonstrate that the proposed method obtains substantial gains over the state-of-the-art, 7% improvements on CIFAR10 and CIFAR100 datasets. We show with only less than 1% data that our method achieves 99.6%, 94.9%, 88.5%, 71.5% relative performance on MNIST, FashionMNIST, SVHN, CIFAR10 respectively. We also explore the use of our method in continual learning and neural architecture search, and show promising results.

----

## [1152] Joining datasets via data augmentation in the label space for neural networks

**Authors**: *Junbo Zhao, Mingfeng Ou, Linji Xue, Yunkai Cui, Sai Wu, Gang Chen*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/zhao21b.html](http://proceedings.mlr.press/v139/zhao21b.html)

**Abstract**:

Most, if not all, modern deep learning systems restrict themselves to a single dataset for neural network training and inference. In this article, we are interested in systematic ways to join datasets that are made of similar purposes. Unlike previous published works that ubiquitously conduct the dataset joining in the uninterpretable latent vectorial space, the core to our method is an augmentation procedure in the label space. The primary challenge to address the label space for dataset joining is the discrepancy between labels: non-overlapping label annotation sets, different labeling granularity or hierarchy and etc. Notably we propose a new technique leveraging artificially created knowledge graph, recurrent neural networks and policy gradient that successfully achieve the dataset joining in the label space. Empirical results on both image and text classification justify the validity of our approach.

----

## [1153] Calibrate Before Use: Improving Few-shot Performance of Language Models

**Authors**: *Zihao Zhao, Eric Wallace, Shi Feng, Dan Klein, Sameer Singh*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/zhao21c.html](http://proceedings.mlr.press/v139/zhao21c.html)

**Abstract**:

GPT-3 can perform numerous tasks when provided a natural language prompt that contains a few training examples. We show that this type of few-shot learning can be unstable: the choice of prompt format, training examples, and even the order of the examples can cause accuracy to vary from near chance to near state-of-the-art. We demonstrate that this instability arises from the bias of language models towards predicting certain answers, e.g., those that are placed near the end of the prompt or are common in the pre-training data. To mitigate this, we first estimate the model’s bias towards each answer by asking for its prediction when given a training prompt and a content-free test input such as "N/A". We then fit calibration parameters that cause the prediction for this input to be uniform across answers. On a diverse set of tasks, this contextual calibration procedure substantially improves GPT-3 and GPT-2’s accuracy (up to 30.0% absolute) across different choices of the prompt, while also making learning considerably more stable.

----

## [1154] Few-Shot Neural Architecture Search

**Authors**: *Yiyang Zhao, Linnan Wang, Yuandong Tian, Rodrigo Fonseca, Tian Guo*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/zhao21d.html](http://proceedings.mlr.press/v139/zhao21d.html)

**Abstract**:

Efficient evaluation of a network architecture drawn from a large search space remains a key challenge in Neural Architecture Search (NAS). Vanilla NAS evaluates each architecture by training from scratch, which gives the true performance but is extremely time-consuming. Recently, one-shot NAS substantially reduces the computation cost by training only one supernetwork, a.k.a. supernet, to approximate the performance of every architecture in the search space via weight-sharing. However, the performance estimation can be very inaccurate due to the co-adaption among operations. In this paper, we propose few-shot NAS that uses multiple supernetworks, called sub-supernet, each covering different regions of the search space to alleviate the undesired co-adaption. Compared to one-shot NAS, few-shot NAS improves the accuracy of architecture evaluation with a small increase of evaluation cost. With only up to 7 sub-supernets, few-shot NAS establishes new SoTAs: on ImageNet, it finds models that reach 80.5% top-1 accuracy at 600 MB FLOPS and 77.5% top-1 accuracy at 238 MFLOPS; on CIFAR10, it reaches 98.72% top-1 accuracy without using extra data or transfer learning. In Auto-GAN, few-shot NAS outperforms the previously published results by up to 20%. Extensive experiments show that few-shot NAS significantly improves various one-shot methods, including 4 gradient-based and 6 search-based methods on 3 different tasks in NasBench-201 and NasBench1-shot-1.

----

## [1155] Expressive 1-Lipschitz Neural Networks for Robust Multiple Graph Learning against Adversarial Attacks

**Authors**: *Xin Zhao, Zeru Zhang, Zijie Zhang, Lingfei Wu, Jiayin Jin, Yang Zhou, Ruoming Jin, Dejing Dou, Da Yan*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/zhao21e.html](http://proceedings.mlr.press/v139/zhao21e.html)

**Abstract**:

Recent findings have shown multiple graph learning models, such as graph classification and graph matching, are highly vulnerable to adversarial attacks, i.e. small input perturbations in graph structures and node attributes can cause the model failures. Existing defense techniques often defend specific attacks on particular multiple graph learning tasks. This paper proposes an attack-agnostic graph-adaptive 1-Lipschitz neural network, ERNN, for improving the robustness of deep multiple graph learning while achieving remarkable expressive power. A K_l-Lipschitz Weibull activation function is designed to enforce the gradient norm as K_l at layer l. The nearest matrix orthogonalization and polar decomposition techniques are utilized to constraint the weight norm as 1/K_l and make the norm-constrained weight close to the original weight. The theoretical analysis is conducted to derive lower and upper bounds of feasible K_l under the 1-Lipschitz constraint. The combination of norm-constrained weight and activation function leads to the 1-Lipschitz neural network for expressive and robust multiple graph learning.

----

## [1156] Fused Acoustic and Text Encoding for Multimodal Bilingual Pretraining and Speech Translation

**Authors**: *Renjie Zheng, Junkun Chen, Mingbo Ma, Liang Huang*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/zheng21a.html](http://proceedings.mlr.press/v139/zheng21a.html)

**Abstract**:

Recently, representation learning for text and speech has successfully improved many language related tasks. However, all existing methods suffer from two limitations: (a) they only learn from one input modality, while a unified representation for both speech and text is needed by tasks such as end-to-end speech translation, and as a result, (b) they can not exploit various large-scale text and speech data and their performance is limited by the scarcity of parallel speech translation data. To address these problems, we propose a Fused Acoustic and Text Masked Language Model (FAT-MLM) which jointly learns a unified representation for both acoustic and text input from various types of corpora including parallel data for speech recognition and machine translation, and even pure speech and text data. Within this cross-modal representation learning framework, we further present an end-to-end model for Fused Acoustic and Text Speech Translation (FAT-ST). Experiments on three translation directions show that by fine-tuning from FAT-MLM, our proposed speech translation models substantially improve translation quality by up to +5.9 BLEU.

----

## [1157] Two Heads are Better Than One: Hypergraph-Enhanced Graph Reasoning for Visual Event Ratiocination

**Authors**: *Wenbo Zheng, Lan Yan, Chao Gou, Fei-Yue Wang*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/zheng21b.html](http://proceedings.mlr.press/v139/zheng21b.html)

**Abstract**:

Even with a still image, humans can ratiocinate various visual cause-and-effect descriptions before, at present, and after, as well as beyond the given image. However, it is challenging for models to achieve such task–the visual event ratiocination, owing to the limitations of time and space. To this end, we propose a novel multi-modal model, Hypergraph-Enhanced Graph Reasoning. First it represents the contents from the same modality as a semantic graph and mines the intra-modality relationship, therefore breaking the limitations in the spatial domain. Then, we introduce the Graph Self-Attention Enhancement. On the one hand, this enables semantic graph representations from different modalities to enhance each other and captures the inter-modality relationship along the line. On the other hand, it utilizes our built multi-modal hypergraphs in different moments to boost individual semantic graph representations, and breaks the limitations in the temporal domain. Our method illustrates the case of "two heads are better than one" in the sense that semantic graph representations with the help of the proposed enhancement mechanism are more robust than those without. Finally, we re-project these representations and leverage their outcomes to generate textual cause-and-effect descriptions. Experimental results show that our model achieves significantly higher performance in comparison with other state-of-the-arts.

----

## [1158] How Framelets Enhance Graph Neural Networks

**Authors**: *Xuebin Zheng, Bingxin Zhou, Junbin Gao, Yuguang Wang, Pietro Lió, Ming Li, Guido Montúfar*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/zheng21c.html](http://proceedings.mlr.press/v139/zheng21c.html)

**Abstract**:

This paper presents a new approach for assembling graph neural networks based on framelet transforms. The latter provides a multi-scale representation for graph-structured data. We decompose an input graph into low-pass and high-pass frequencies coefficients for network training, which then defines a framelet-based graph convolution. The framelet decomposition naturally induces a graph pooling strategy by aggregating the graph feature into low-pass and high-pass spectra, which considers both the feature values and geometry of the graph data and conserves the total information. The graph neural networks with the proposed framelet convolution and pooling achieve state-of-the-art performance in many node and graph prediction tasks. Moreover, we propose shrinkage as a new activation for the framelet convolution, which thresholds high-frequency information at different scales. Compared to ReLU, shrinkage activation improves model performance on denoising and signal compression: noises in both node and structure can be significantly reduced by accurately cutting off the high-pass coefficients from framelet decomposition, and the signal can be compressed to less than half its original size with well-preserved prediction performance.

----

## [1159] Probabilistic Sequential Shrinking: A Best Arm Identification Algorithm for Stochastic Bandits with Corruptions

**Authors**: *Zixin Zhong, Wang Chi Cheung, Vincent Y. F. Tan*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/zhong21a.html](http://proceedings.mlr.press/v139/zhong21a.html)

**Abstract**:

We consider a best arm identification (BAI) problem for stochastic bandits with adversarial corruptions in the fixed-budget setting of T steps. We design a novel randomized algorithm, Probabilistic Sequential Shrinking(u) (PSS(u)), which is agnostic to the amount of corruptions. When the amount of corruptions per step (CPS) is below a threshold, PSS(u) identifies the best arm or item with probability tending to 1 as T{\rightarrow}$\infty$. Otherwise, the optimality gap of the identified item degrades gracefully with the CPS.We argue that such a bifurcation is necessary. In PSS(u), the parameter u serves to balance between the optimality gap and success probability. The injection of randomization is shown to be essential to mitigate the impact of corruptions. To demonstrate this, we design two attack strategies that are applicable to any algorithm. We apply one of them to a deterministic analogue of PSS(u) known as Successive Halving (SH) by Karnin et al. (2013). The attack strategy results in a high failure probability for SH, but PSS(u) remains robust. In the absence of corruptions, PSS(2)’s performance guarantee matches SH’s. We show that when the CPS is sufficiently large, no algorithm can achieve a BAI probability tending to 1 as T{\rightarrow}$\infty$. Numerical experiments corroborate our theoretical findings.

----

## [1160] Towards Distraction-Robust Active Visual Tracking

**Authors**: *Fangwei Zhong, Peng Sun, Wenhan Luo, Tingyun Yan, Yizhou Wang*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/zhong21b.html](http://proceedings.mlr.press/v139/zhong21b.html)

**Abstract**:

In active visual tracking, it is notoriously difficult when distracting objects appear, as distractors often mislead the tracker by occluding the target or bringing a confusing appearance. To address this issue, we propose a mixed cooperative-competitive multi-agent game, where a target and multiple distractors form a collaborative team to play against a tracker and make it fail to follow. Through learning in our game, diverse distracting behaviors of the distractors naturally emerge, thereby exposing the tracker’s weakness, which helps enhance the distraction-robustness of the tracker. For effective learning, we then present a bunch of practical methods, including a reward function for distractors, a cross-modal teacher-student learning strategy, and a recurrent attention mechanism for the tracker. The experimental results show that our tracker performs desired distraction-robust active visual tracking and can be well generalized to unseen environments. We also show that the multi-agent game can be used to adversarially test the robustness of trackers.

----

## [1161] Provably Efficient Reinforcement Learning for Discounted MDPs with Feature Mapping

**Authors**: *Dongruo Zhou, Jiafan He, Quanquan Gu*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/zhou21a.html](http://proceedings.mlr.press/v139/zhou21a.html)

**Abstract**:

Modern tasks in reinforcement learning have large state and action spaces. To deal with them efficiently, one often uses predefined feature mapping to represent states and actions in a low dimensional space. In this paper, we study reinforcement learning for discounted Markov Decision Processes (MDPs), where the transition kernel can be parameterized as a linear function of certain feature mapping. We propose a novel algorithm which makes use of the feature mapping and obtains a $\tilde O(d\sqrt{T}/(1-\gamma)^2)$ regret, where $d$ is the dimension of the feature space, $T$ is the time horizon and $\gamma$ is the discount factor of the MDP. To the best of our knowledge, this is the first polynomial regret bound without accessing a generative model or making strong assumptions such as ergodicity of the MDP. By constructing a special class of MDPs, we also show that for any algorithms, the regret is lower bounded by $\Omega(d\sqrt{T}/(1-\gamma)^{1.5})$. Our upper and lower bound results together suggest that the proposed reinforcement learning algorithm is near-optimal up to a $(1-\gamma)^{-0.5}$ factor.

----

## [1162] Amortized Conditional Normalized Maximum Likelihood: Reliable Out of Distribution Uncertainty Estimation

**Authors**: *Aurick Zhou, Sergey Levine*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/zhou21b.html](http://proceedings.mlr.press/v139/zhou21b.html)

**Abstract**:

While deep neural networks provide good performance for a range of challenging tasks, calibration and uncertainty estimation remain major challenges, especially under distribution shift. In this paper, we propose the amortized conditional normalized maximum likelihood (ACNML) method as a scalable general-purpose approach for uncertainty estimation, calibration, and out-of-distribution robustness with deep networks. Our algorithm builds on the conditional normalized maximum likelihood (CNML) coding scheme, which has minimax optimal properties according to the minimum description length principle, but is computationally intractable to evaluate exactly for all but the simplest of model classes. We propose to use approximate Bayesian inference technqiues to produce a tractable approximation to the CNML distribution. Our approach can be combined with any approximate inference algorithm that provides tractable posterior densities over model parameters. We demonstrate that ACNML compares favorably to a number of prior techniques for uncertainty estimation in terms of calibration when faced with distribution shift.

----

## [1163] Optimal Estimation of High Dimensional Smooth Additive Function Based on Noisy Observations

**Authors**: *Fan Zhou, Ping Li*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/zhou21c.html](http://proceedings.mlr.press/v139/zhou21c.html)

**Abstract**:

Given $\bx_j = \btheta + \bepsilon_j$, $j=1,...,n$ where $\btheta \in \RR^d$ is an unknown parameter and $\bepsilon_j$ are i.i.d. Gaussian noise vectors, we study the estimation of $f(\btheta)$ for a given smooth function $f:\RR^d \rightarrow \RR$ equipped with an additive structure. We inherit the idea from a recent work which introduced an effective bias reduction technique through iterative bootstrap and derive a bias-reducing estimator. By establishing its normal approximation results, we show that the proposed estimator can achieve asymptotic normality with a looser constraint on smoothness compared with general smooth function due to the additive structure. Such results further imply that the proposed estimator is asymptotically efficient. Both upper and lower bounds on mean squared error are proved which shows the proposed estimator is minimax optimal for the smooth class considered. Numerical simulation results are presented to validate our analysis and show its superior performance of the proposed estimator over the plug-in approach in terms of bias reduction and building confidence intervals.

----

## [1164] Incentivized Bandit Learning with Self-Reinforcing User Preferences

**Authors**: *Tianchen Zhou, Jia Liu, Chaosheng Dong, Jingyuan Deng*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/zhou21d.html](http://proceedings.mlr.press/v139/zhou21d.html)

**Abstract**:

In this paper, we investigate a new multi-armed bandit (MAB) online learning model that considers real-world phenomena in many recommender systems: (i) the learning agent cannot pull the arms by itself and thus has to offer rewards to users to incentivize arm-pulling indirectly; and (ii) if users with specific arm preferences are well rewarded, they induce a "self-reinforcing" effect in the sense that they will attract more users of similar arm preferences. Besides addressing the tradeoff of exploration and exploitation, another key feature of this new MAB model is to balance reward and incentivizing payment. The goal of the agent is to maximize the total reward over a fixed time horizon $T$ with a low total payment. Our contributions in this paper are two-fold: (i) We propose a new MAB model with random arm selection that considers the relationship of users’ self-reinforcing preferences and incentives; and (ii) We leverage the properties of a multi-color Polya urn with nonlinear feedback model to propose two MAB policies termed "At-Least-$n$ Explore-Then-Commit" and "UCB-List". We prove that both policies achieve $O(log T)$ expected regret with $O(log T)$ expected payment over a time horizon $T$. We conduct numerical simulations to demonstrate and verify the performances of these two policies and study their robustness under various settings.

----

## [1165] Towards Defending against Adversarial Examples via Attack-Invariant Features

**Authors**: *Dawei Zhou, Tongliang Liu, Bo Han, Nannan Wang, Chunlei Peng, Xinbo Gao*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/zhou21e.html](http://proceedings.mlr.press/v139/zhou21e.html)

**Abstract**:

Deep neural networks (DNNs) are vulnerable to adversarial noise. Their adversarial robustness can be improved by exploiting adversarial examples. However, given the continuously evolving attacks, models trained on seen types of adversarial examples generally cannot generalize well to unseen types of adversarial examples. To solve this problem, in this paper, we propose to remove adversarial noise by learning generalizable invariant features across attacks which maintain semantic classification information. Specifically, we introduce an adversarial feature learning mechanism to disentangle invariant features from adversarial noise. A normalization term has been proposed in the encoded space of the attack-invariant features to address the bias issue between the seen and unseen types of attacks. Empirical evaluations demonstrate that our method could provide better protection in comparison to previous state-of-the-art approaches, especially against unseen types of attacks and adaptive attacks.

----

## [1166] Asymmetric Loss Functions for Learning with Noisy Labels

**Authors**: *Xiong Zhou, Xianming Liu, Junjun Jiang, Xin Gao, Xiangyang Ji*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/zhou21f.html](http://proceedings.mlr.press/v139/zhou21f.html)

**Abstract**:

Robust loss functions are essential for training deep neural networks with better generalization power in the presence of noisy labels. Symmetric loss functions are confirmed to be robust to label noise. However, the symmetric condition is overly restrictive. In this work, we propose a new class of loss functions, namely asymmetric loss functions, which are robust to learning from noisy labels for arbitrary noise type. Subsequently, we investigate general theoretical properties of asymmetric loss functions, including classification-calibration, excess risk bound, and noise-tolerance. Meanwhile, we introduce the asymmetry ratio to measure the asymmetry of a loss function, and the empirical results show that a higher ratio will provide better robustness. Moreover, we modify several common loss functions, and establish the necessary and sufficient conditions for them to be asymmetric. Experiments on benchmark datasets demonstrate that asymmetric loss functions can outperform state-of-the-art methods.

----

## [1167] Examining and Combating Spurious Features under Distribution Shift

**Authors**: *Chunting Zhou, Xuezhe Ma, Paul Michel, Graham Neubig*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/zhou21g.html](http://proceedings.mlr.press/v139/zhou21g.html)

**Abstract**:

A central goal of machine learning is to learn robust representations that capture the fundamental relationship between inputs and output labels. However, minimizing training errors over finite or biased datasets results in models latching on to spurious correlations between the training input/output pairs that are not fundamental to the problem at hand. In this paper, we define and analyze robust and spurious representations using the information-theoretic concept of minimal sufficient statistics. We prove that even when there is only bias of the input distribution (i.e. covariate shift), models can still pick up spurious features from their training data. Group distributionally robust optimization (DRO) provides an effective tool to alleviate covariate shift by minimizing the worst-case training losses over a set of pre-defined groups. Inspired by our analysis, we demonstrate that group DRO can fail when groups do not directly account for various spurious correlations that occur in the data. To address this, we further propose to minimize the worst-case losses over a more flexible set of distributions that are defined on the joint distribution of groups and instances, instead of treating each group as a whole at optimization time. Through extensive experiments on one image and two language tasks, we show that our model is significantly more robust than comparable baselines under various partitions.

----

## [1168] Sparse and Imperceptible Adversarial Attack via a Homotopy Algorithm

**Authors**: *Mingkang Zhu, Tianlong Chen, Zhangyang Wang*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/zhu21a.html](http://proceedings.mlr.press/v139/zhu21a.html)

**Abstract**:

Sparse adversarial attacks can fool deep neural networks (DNNs) by only perturbing a few pixels (regularized by $\ell_0$ norm). Recent efforts combine it with another $\ell_\infty$ imperceptible on the perturbation magnitudes. The resultant sparse and imperceptible attacks are practically relevant, and indicate an even higher vulnerability of DNNs that we usually imagined. However, such attacks are more challenging to generate due to the optimization difficulty by coupling the $\ell_0$ regularizer and box constraints with a non-convex objective. In this paper, we address this challenge by proposing a homotopy algorithm, to jointly tackle the sparsity and the perturbation bound in one unified framework. Each iteration, the main step of our algorithm is to optimize an $\ell_0$-regularized adversarial loss, by leveraging the nonmonotone Accelerated Proximal Gradient Method (nmAPG) for nonconvex programming; it is followed by an $\ell_0$ change control step, and an optional post-attack step designed to escape bad local minima. We also extend the algorithm to handling the structural sparsity regularizer. We extensively examine the effectiveness of our proposed \textbf{homotopy attack} for both targeted and non-targeted attack scenarios, on CIFAR-10 and ImageNet datasets. Compared to state-of-the-art methods, our homotopy attack leads to significantly fewer perturbations, e.g., reducing 42.91% on CIFAR-10 and 75.03% on ImageNet (average case, targeted attack), at similar maximal perturbation magnitudes, when still achieving 100% attack success rates. Our codes are available at: {\small\url{https://github.com/VITA-Group/SparseADV_Homotopy}}.

----

## [1169] Data-Free Knowledge Distillation for Heterogeneous Federated Learning

**Authors**: *Zhuangdi Zhu, Junyuan Hong, Jiayu Zhou*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/zhu21b.html](http://proceedings.mlr.press/v139/zhu21b.html)

**Abstract**:

Federated Learning (FL) is a decentralized machine-learning paradigm, in which a global server iteratively averages the model parameters of local users without accessing their data. User heterogeneity has imposed significant challenges to FL, which can incur drifted global models that are slow to converge. Knowledge Distillation has recently emerged to tackle this issue, by refining the server model using aggregated knowledge from heterogeneous users, other than directly averaging their model parameters. This approach, however, depends on a proxy dataset, making it impractical unless such a prerequisite is satisfied. Moreover, the ensemble knowledge is not fully utilized to guide local model learning, which may in turn affect the quality of the aggregated model. Inspired by the prior art, we propose a data-free knowledge distillation approach to address heterogeneous FL, where the server learns a lightweight generator to ensemble user information in a data-free manner, which is then broadcasted to users, regulating local training using the learned knowledge as an inductive bias. Empirical studies powered by theoretical implications show that our approach facilitates FL with better generalization performance using fewer communication rounds, compared with the state-of-the-art.

----

## [1170] Spectral vertex sparsifiers and pair-wise spanners over distributed graphs

**Authors**: *Chunjiang Zhu, Qinqing Liu, Jinbo Bi*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/zhu21c.html](http://proceedings.mlr.press/v139/zhu21c.html)

**Abstract**:

Graph sparsification is a powerful tool to approximate an arbitrary graph and has been used in machine learning over graphs. As real-world networks are becoming very large and naturally distributed, distributed graph sparsification has drawn considerable attention. In this work, we design communication-efficient distributed algorithms for constructing spectral vertex sparsifiers, which closely preserve effective resistance distances on a subset of vertices of interest in the original graphs, under the well-established message passing communication model. We prove that the communication cost approximates the lower bound with only a small gap. We further provide algorithms for constructing pair-wise spanners which approximate the shortest distances between each pair of vertices in a target set, instead of all pairs, and incur communication costs that are much smaller than those of existing algorithms in the message passing model. Experiments are performed to validate the communication efficiency of the proposed algorithms under the guarantee that the constructed sparsifiers have a good approximation quality.

----

## [1171] Few-shot Language Coordination by Modeling Theory of Mind

**Authors**: *Hao Zhu, Graham Neubig, Yonatan Bisk*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/zhu21d.html](http://proceedings.mlr.press/v139/zhu21d.html)

**Abstract**:

No man is an island. Humans develop the ability to communicate with a large community by coordinating with different interlocutors within short conversations. This ability is largely understudied by the research on building neural language communicative agents. We study the task of few-shot language coordination: agents quickly adapting to their conversational partners’ language abilities. Different from current communicative agents trained with self-play, we in- investigate this more general paradigm by requiring the lead agent to coordinate with a population of agents each of whom has different linguistic abilities. This leads to a general agent able to quickly adapt to communicating with unseen agents in the population. Unlike prior work, success here requires the ability to model the partner’s beliefs, a vital component of human communication. Drawing inspiration from the study of theory-of-mind (ToM; Premack & Woodruff (1978)), we study the effect of the speaker explicitly modeling the listener’s mental state. Learning by communicating with a population, the speakers, as shown in our experiments, acquire the ability to learn to predict the reactions of their partner upon various messages on-the-fly. The speaker’s predictions for the future actions help it generate the best instructions in order to maximize communicative goal with message costs. To examine our hypothesis that the instructions generated with ToM modeling yield better communication per- performance, we employ our agents in both a referential game and a language navigation task. Positive results from our experiments also hint at the importance of explicitly modeling language acquisition as a socio-pragmatic progress.

----

## [1172] Clusterability as an Alternative to Anchor Points When Learning with Noisy Labels

**Authors**: *Zhaowei Zhu, Yiwen Song, Yang Liu*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/zhu21e.html](http://proceedings.mlr.press/v139/zhu21e.html)

**Abstract**:

The label noise transition matrix, characterizing the probabilities of a training instance being wrongly annotated, is crucial to designing popular solutions to learning with noisy labels. Existing works heavily rely on finding “anchor points” or their approximates, defined as instances belonging to a particular class almost surely. Nonetheless, finding anchor points remains a non-trivial task, and the estimation accuracy is also often throttled by the number of available anchor points. In this paper, we propose an alternative option to the above task. Our main contribution is the discovery of an efficient estimation procedure based on a clusterability condition. We prove that with clusterable representations of features, using up to third-order consensuses of noisy labels among neighbor representations is sufficient to estimate a unique transition matrix. Compared with methods using anchor points, our approach uses substantially more instances and benefits from a much better sample complexity. We demonstrate the estimation accuracy and advantages of our estimates using both synthetic noisy labels (on CIFAR-10/100) and real human-level noisy labels (on Clothing1M and our self-collected human-annotated CIFAR-10). Our code and human-level noisy CIFAR-10 labels are available at https://github.com/UCSC-REAL/HOC.

----

## [1173] Commutative Lie Group VAE for Disentanglement Learning

**Authors**: *Xinqi Zhu, Chang Xu, Dacheng Tao*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/zhu21f.html](http://proceedings.mlr.press/v139/zhu21f.html)

**Abstract**:

We view disentanglement learning as discovering an underlying structure that equivariantly reflects the factorized variations shown in data. Traditionally, such a structure is fixed to be a vector space with data variations represented by translations along individual latent dimensions. We argue this simple structure is suboptimal since it requires the model to learn to discard the properties (e.g. different scales of changes, different levels of abstractness) of data variations, which is an extra work than equivariance learning. Instead, we propose to encode the data variations with groups, a structure not only can equivariantly represent variations, but can also be adaptively optimized to preserve the properties of data variations. Considering it is hard to conduct training on group structures, we focus on Lie groups and adopt a parameterization using Lie algebra. Based on the parameterization, some disentanglement learning constraints are naturally derived. A simple model named Commutative Lie Group VAE is introduced to realize the group-based disentanglement learning. Experiments show that our model can effectively learn disentangled representations without supervision, and can achieve state-of-the-art performance without extra constraints.

----

## [1174] Accumulated Decoupled Learning with Gradient Staleness Mitigation for Convolutional Neural Networks

**Authors**: *Huiping Zhuang, Zhenyu Weng, Fulin Luo, Kar-Ann Toj, Haizhou Li, Zhiping Lin*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/zhuang21a.html](http://proceedings.mlr.press/v139/zhuang21a.html)

**Abstract**:

Gradient staleness is a major side effect in decoupled learning when training convolutional neural networks asynchronously. Existing methods that ignore this effect might result in reduced generalization and even divergence. In this paper, we propose an accumulated decoupled learning (ADL), which includes a module-wise gradient accumulation in order to mitigate the gradient staleness. Unlike prior arts ignoring the gradient staleness, we quantify the staleness in such a way that its mitigation can be quantitatively visualized. As a new learning scheme, the proposed ADL is theoretically shown to converge to critical points in spite of its asynchronism. Extensive experiments on CIFAR-10 and ImageNet datasets are conducted, demonstrating that ADL gives promising generalization results while the state-of-the-art methods experience reduced generalization and divergence. In addition, our ADL is shown to have the fastest training speed among the compared methods.

----

## [1175] Demystifying Inductive Biases for (Beta-)VAE Based Architectures

**Authors**: *Dominik Zietlow, Michal Rolínek, Georg Martius*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/zietlow21a.html](http://proceedings.mlr.press/v139/zietlow21a.html)

**Abstract**:

The performance of Beta-Variational-Autoencoders and their variants on learning semantically meaningful, disentangled representations is unparalleled. On the other hand, there are theoretical arguments suggesting the impossibility of unsupervised disentanglement. In this work, we shed light on the inductive bias responsible for the success of VAE-based architectures. We show that in classical datasets the structure of variance, induced by the generating factors, is conveniently aligned with the latent directions fostered by the VAE objective. This builds the pivotal bias on which the disentangling abilities of VAEs rely. By small, elaborate perturbations of existing datasets, we hide the convenient correlation structure that is easily exploited by a variety of architectures. To demonstrate this, we construct modified versions of standard datasets in which (i) the generative factors are perfectly preserved; (ii) each image undergoes a mild transformation causing a small change of variance; (iii) the leading VAE-based disentanglement architectures fail to produce disentangled representations whilst the performance of a non-variational method remains unchanged.

----

## [1176] Recovering AES Keys with a Deep Cold Boot Attack

**Authors**: *Itamar Zimerman, Eliya Nachmani, Lior Wolf*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/zimerman21a.html](http://proceedings.mlr.press/v139/zimerman21a.html)

**Abstract**:

Cold boot attacks inspect the corrupted random access memory soon after the power has been shut down. While most of the bits have been corrupted, many bits, at random locations, have not. Since the keys in many encryption schemes are being expanded in memory into longer keys with fixed redundancies, the keys can often be restored. In this work we combine a deep error correcting code technique together with a modified SAT solver scheme in order to apply the attack to AES keys. Even though AES consists Rijndael SBOX elements, that are specifically designed to be resistant to linear and differential cryptanalysis, our method provides a novel formalization of the AES key scheduling as a computational graph, which is implemented by neural message passing network. Our results show that our methods outperform the state of the art attack methods by a very large gap.

----

## [1177] Learning Fair Policies in Decentralized Cooperative Multi-Agent Reinforcement Learning

**Authors**: *Matthieu Zimmer, Claire Glanois, Umer Siddique, Paul Weng*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/zimmer21a.html](http://proceedings.mlr.press/v139/zimmer21a.html)

**Abstract**:

We consider the problem of learning fair policies in (deep) cooperative multi-agent reinforcement learning (MARL). We formalize it in a principled way as the problem of optimizing a welfare function that explicitly encodes two important aspects of fairness: efficiency and equity. We provide a theoretical analysis of the convergence of policy gradient for this problem. As a solution method, we propose a novel neural network architecture, which is composed of two sub-networks specifically designed for taking into account these two aspects of fairness. In experiments, we demonstrate the importance of the two sub-networks for fair optimization. Our overall approach is general as it can accommodate any (sub)differentiable welfare function. Therefore, it is compatible with various notions of fairness that have been proposed in the literature (e.g., lexicographic maximin, generalized Gini social welfare function, proportional fairness). Our method is generic and can be implemented in various MARL settings: centralized training and decentralized execution, or fully decentralized. Finally, we experimentally validate our approach in various domains and show that it can perform much better than previous methods, both in terms of efficiency and equity.

----

## [1178] Contrastive Learning Inverts the Data Generating Process

**Authors**: *Roland S. Zimmermann, Yash Sharma, Steffen Schneider, Matthias Bethge, Wieland Brendel*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/zimmermann21a.html](http://proceedings.mlr.press/v139/zimmermann21a.html)

**Abstract**:

Contrastive learning has recently seen tremendous success in self-supervised learning. So far, however, it is largely unclear why the learned representations generalize so effectively to a large variety of downstream tasks. We here prove that feedforward models trained with objectives belonging to the commonly used InfoNCE family learn to implicitly invert the underlying generative model of the observed data. While the proofs make certain statistical assumptions about the generative model, we observe empirically that our findings hold even if these assumptions are severely violated. Our theory highlights a fundamental connection between contrastive learning, generative modeling, and nonlinear independent component analysis, thereby furthering our understanding of the learned representations as well as providing a theoretical foundation to derive more effective contrastive losses.

----

## [1179] Exploration in Approximate Hyper-State Space for Meta Reinforcement Learning

**Authors**: *Luisa M. Zintgraf, Leo Feng, Cong Lu, Maximilian Igl, Kristian Hartikainen, Katja Hofmann, Shimon Whiteson*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/zintgraf21a.html](http://proceedings.mlr.press/v139/zintgraf21a.html)

**Abstract**:

To rapidly learn a new task, it is often essential for agents to explore efficiently - especially when performance matters from the first timestep. One way to learn such behaviour is via meta-learning. Many existing methods however rely on dense rewards for meta-training, and can fail catastrophically if the rewards are sparse. 	Without a suitable reward signal, the need for exploration during meta-training is exacerbated. To address this, we propose HyperX, which uses novel reward bonuses for meta-training to explore in approximate hyper-state space (where hyper-states represent the environment state and the agent’s task belief). We show empirically that HyperX meta-learns better task-exploration and adapts more successfully to new tasks than existing methods.

----

## [1180] Provable Robustness of Adversarial Training for Learning Halfspaces with Noise

**Authors**: *Difan Zou, Spencer Frei, Quanquan Gu*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/zou21a.html](http://proceedings.mlr.press/v139/zou21a.html)

**Abstract**:

We analyze the properties of adversarial training for learning adversarially robust halfspaces in the presence of agnostic label noise. Denoting $\mathsf{OPT}_{p,r}$ as the best classification error achieved by a halfspace that is robust to perturbations of $\ell^{p}$ balls of radius $r$, we show that adversarial training on the standard binary cross-entropy loss yields adversarially robust halfspaces up to classification error $\tilde O(\sqrt{\mathsf{OPT}_{2,r}})$ for $p=2$, and $\tilde O(d^{1/4} \sqrt{\mathsf{OPT}_{\infty, r}})$ when $p=\infty$. Our results hold for distributions satisfying anti-concentration properties enjoyed by log-concave isotropic distributions among others. We additionally show that if one instead uses a non-convex sigmoidal loss, adversarial training yields halfspaces with an improved robust classification error of $O(\mathsf{OPT}_{2,r})$ for $p=2$, and $O(d^{1/4} \mathsf{OPT}_{\infty, r})$ when $p=\infty$. To the best of our knowledge, this is the first work showing that adversarial training provably yields robust classifiers in the presence of noise.

----

## [1181] On the Convergence of Hamiltonian Monte Carlo with Stochastic Gradients

**Authors**: *Difan Zou, Quanquan Gu*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/zou21b.html](http://proceedings.mlr.press/v139/zou21b.html)

**Abstract**:

Hamiltonian Monte Carlo (HMC), built based on the Hamilton’s equation, has been witnessed great success in sampling from high-dimensional posterior distributions. However, it also suffers from computational inefficiency, especially for large training datasets. One common idea to overcome this computational bottleneck is using stochastic gradients, which only queries a mini-batch of training data in each iteration. However, unlike the extensive studies on the convergence analysis of HMC using full gradients, few works focus on establishing the convergence guarantees of stochastic gradient HMC algorithms. In this paper, we propose a general framework for proving the convergence rate of HMC with stochastic gradient estimators, for sampling from strongly log-concave and log-smooth target distributions. We show that the convergence to the target distribution in $2$-Wasserstein distance can be guaranteed as long as the stochastic gradient estimator is unbiased and its variance is upper bounded along the algorithm trajectory. We further apply the proposed framework to analyze the convergence rates of HMC with four standard stochastic gradient estimators: mini-batch stochastic gradient (SG), stochastic variance reduced gradient (SVRG), stochastic average gradient (SAGA), and control variate gradient (CVG). Theoretical results explain the inefficiency of mini-batch SG, and suggest that SVRG and SAGA perform better in the tasks with high-precision requirements, while CVG performs better for large dataset. Experiment results verify our theoretical findings.

----

## [1182] A Functional Perspective on Learning Symmetric Functions with Neural Networks

**Authors**: *Aaron Zweig, Joan Bruna*

**Conference**: *icml 2021*

**URL**: [http://proceedings.mlr.press/v139/zweig21a.html](http://proceedings.mlr.press/v139/zweig21a.html)

**Abstract**:

Symmetric functions, which take as input an unordered, fixed-size set, are known to be universally representable by neural networks that enforce permutation invariance. These architectures only give guarantees for fixed input sizes, yet in many practical applications, including point clouds and particle physics, a relevant notion of generalization should include varying the input size. In this work we treat symmetric functions (of any size) as functions over probability measures, and study the learning and representation of neural networks defined on measures. By focusing on shallow architectures, we establish approximation and generalization bounds under different choices of regularization (such as RKHS and variation norms), that capture a hierarchy of functional spaces with increasing degree of non-linear learning. The resulting models can be learned efficiently and enjoy generalization guarantees that extend across input sizes, as we verify empirically.

----



[Go to the previous page](ICML-2021-list05.md)

[Go to the catalog section](README.md)