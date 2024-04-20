## [1800] Towards a Theoretical Framework of Out-of-Distribution Generalization

        **Authors**: *Haotian Ye, Chuanlong Xie, Tianle Cai, Ruichen Li, Zhenguo Li, Liwei Wang*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/c5c1cb0bebd56ae38817b251ad72bedb-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/c5c1cb0bebd56ae38817b251ad72bedb-Abstract.html)

        **Abstract**:

        Generalization to out-of-distribution (OOD) data is one of the central problems in modern machine learning. Recently, there is a surge of attempts to propose algorithms that mainly build upon the idea of extracting invariant features. Although intuitively reasonable, theoretical understanding of what kind of invariance can guarantee OOD generalization is still limited, and generalization to arbitrary out-of-distribution is clearly impossible. In this work, we take the first step towards rigorous and quantitative definitions of 1) what is OOD; and 2) what does it mean by saying an OOD problem is learnable. We also introduce a new concept of expansion function, which characterizes to what extent the variance is amplified in the test domains over the training domains, and therefore give a quantitative meaning of invariant features. Based on these, we prove an OOD generalization error bound. It turns out that OOD generalization largely depends on the expansion function. As recently pointed out by Gulrajani & Lopez-Paz (2020), any OOD learning algorithm without a model selection module is incomplete. Our theory naturally induces a model selection criterion. Extensive experiments on benchmark OOD datasets demonstrate that our model selection criterion has a significant advantage over baselines.

        ----

        ## [1801] Slice Sampling Reparameterization Gradients

        **Authors**: *David M. Zoltowski, Diana Cai, Ryan P. Adams*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/c5c3d4fe6b2cc463c7d7ecba17cc9de7-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/c5c3d4fe6b2cc463c7d7ecba17cc9de7-Abstract.html)

        **Abstract**:

        Many probabilistic modeling problems in machine learning use gradient-based optimization in which the objective takes the form of an expectation. These problems can be challenging when the parameters to be optimized determine the probability distribution under which the expectation is being taken, as the na\"ive Monte Carlo procedure is not differentiable. Reparameterization gradients make it possible to efficiently perform optimization of these Monte Carlo objectives by transforming the expectation to be differentiable, but the approach is typically limited to distributions with simple forms and tractable normalization constants. Here we describe how to differentiate samples from slice sampling to compute \textit{slice sampling reparameterization gradients}, enabling a richer class of Monte Carlo objective functions to be optimized. Slice sampling is a Markov chain Monte Carlo algorithm for simulating samples from probability distributions; it only requires a density function that can be evaluated point-wise up to a normalization constant, making it applicable to a variety of inference problems and unnormalized models. Our approach is based on the observation that when the slice endpoints are known, the sampling path is a deterministic and differentiable function of the pseudo-random variables, since the algorithm is rejection-free. We evaluate the method on synthetic examples and apply it to a variety of applications with reparameterization of unnormalized probability distributions.

        ----

        ## [1802] Multi-Label Learning with Pairwise Relevance Ordering

        **Authors**: *Ming-Kun Xie, Sheng-Jun Huang*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/c5d215777c229704a7862de577d40a73-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/c5d215777c229704a7862de577d40a73-Abstract.html)

        **Abstract**:

        Precisely annotating objects with multiple labels is costly and has become a critical bottleneck in real-world multi-label classification tasks. Instead, deciding the relative order of label pairs is obviously less laborious than collecting exact labels. However, the supervised information of pairwise relevance ordering is less informative than exact labels. It is thus an important challenge to effectively learn with such weak supervision. In this paper, we formalize this problem as a novel learning framework, called multi-label learning with pairwise relevance ordering (PRO). We show that the unbiased estimator of classification risk can be derived with a cost-sensitive loss only from PRO examples. Theoretically, we provide the estimation error bound for the proposed estimator and further prove that it is consistent with respective to the commonly used ranking loss. Empirical studies on multiple datasets and metrics validate the effectiveness of the proposed method.

        ----

        ## [1803] Sampling with Trusthworthy Constraints: A Variational Gradient Framework

        **Authors**: *Xingchao Liu, Xin Tong, Qiang Liu*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/c61aed648da48aa3893fb3eaadd88a7f-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/c61aed648da48aa3893fb3eaadd88a7f-Abstract.html)

        **Abstract**:

        Sampling-based inference and learning techniques, especially Bayesian inference, provide an essential approach to handling uncertainty in machine learning (ML). As these techniques are increasingly used in daily life, it becomes essential to safeguard the ML systems with various trustworthy-related constraints, such as fairness, safety, interpretability. Mathematically, enforcing these constraints in probabilistic inference can be cast into sampling from intractable distributions subject to general nonlinear constraints, for which practical efficient algorithms are still largely missing. In this work, we propose a family of constrained sampling algorithms which generalize Langevin Dynamics (LD) and Stein Variational Gradient Descent (SVGD) to incorporate a moment constraint specified by a general nonlinear function. By exploiting the gradient flow structure of LD and SVGD, we derive two types of algorithms for handling constraints, including a primal-dual gradient approach and the constraint controlled gradient descent approach. We investigate the continuous-time mean-field limit of these algorithms and show that they have O(1/t) convergence under mild conditions. Moreover, the LD variant converges linearly assuming that a log Sobolev like inequality holds. Various numerical experiments are conducted to demonstrate the efficiency of our algorithms in trustworthy settings.

        ----

        ## [1804] Robust and Decomposable Average Precision for Image Retrieval

        **Authors**: *Elias Ramzi, Nicolas Thome, Clément Rambour, Nicolas Audebert, Xavier Bitot*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/c622c085c04eadc473f08541b255320e-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/c622c085c04eadc473f08541b255320e-Abstract.html)

        **Abstract**:

        In image retrieval, standard evaluation metrics rely on score ranking, e.g. average precision (AP). In this paper, we introduce a method for robust and decomposable average precision (ROADMAP) addressing two major challenges for end-to-end training of deep neural networks with AP: non-differentiability and non-decomposability.Firstly, we propose a new differentiable approximation of the rank function, which provides an upper bound of the AP loss and ensures robust training. Secondly, we design a simple yet effective loss function to reduce the decomposability gap between the AP in the whole training set and its averaged batch approximation, for which we provide theoretical guarantees.Extensive experiments conducted on three image retrieval datasets show that ROADMAP outperforms several recent AP approximation methods and highlight the importance of our two contributions. Finally, using ROADMAP for training deep models yields very good performances, outperforming state-of-the-art results on the three datasets.Code and instructions to reproduce our results will be made publicly available at https://github.com/elias-ramzi/ROADMAP.

        ----

        ## [1805] Fast rates for prediction with limited expert advice

        **Authors**: *El Mehdi Saad, Gilles Blanchard*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/c688defd45ad6638febd469adb09ddf7-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/c688defd45ad6638febd469adb09ddf7-Abstract.html)

        **Abstract**:

        We investigate the problem of minimizing the excess generalization error with respect to the best expert prediction in a finite family in the stochastic setting,  under limited access to information. We consider that the learner has only access to a limited number of expert advices per training round, as well as for prediction. Assuming that the loss function is Lipschitz and strongly convex, we show that if we are allowed to see the advice of only one expert per round in the training phase, or to use the advice of only one expert for prediction in the test phase, the worst-case excess risk is ${\Omega}(1/\sqrt{T})$ with probability lower bounded by a constant. However, if we are allowed to see at least two actively chosen expert advices per training round and use at least two experts for prediction, the fast rate $\mathcal{O}(1/T)$ can be achieved. We design novel algorithms achieving this rate in this setting, and in the setting where the learner have a budget constraint on the total number of observed experts advices,  and give precise instance-dependent bounds on the number of training rounds needed to achieve a given generalization error precision.

        ----

        ## [1806] Probabilistic Transformer For Time Series Analysis

        **Authors**: *Binh Tang, David S. Matteson*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/c68bd9055776bf38d8fc43c0ed283678-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/c68bd9055776bf38d8fc43c0ed283678-Abstract.html)

        **Abstract**:

        Generative modeling of multivariate time series has remained challenging partly due to the complex, non-deterministic dynamics across long-distance timesteps. In this paper, we propose deep probabilistic methods that combine state-space models (SSMs) with transformer architectures. In contrast to previously proposed SSMs, our approaches use attention mechanism to model non-Markovian dynamics in the latent space and avoid recurrent neural networks entirely. We also extend our models to include several layers of stochastic variables organized in a hierarchy for further expressiveness. Compared to transformer models, ours are probabilistic, non-autoregressive, and capable of generating diverse long-term forecasts with uncertainty estimates. Extensive experiments show that our models consistently outperform competitive baselines on various tasks and datasets, including time series forecasting and human motion prediction.

        ----

        ## [1807] A Hierarchical Reinforcement Learning Based Optimization Framework for Large-scale Dynamic Pickup and Delivery Problems

        **Authors**: *Yi Ma, Xiaotian Hao, Jianye Hao, Jiawen Lu, Xing Liu, Xialiang Tong, Mingxuan Yuan, Zhigang Li, Jie Tang, Zhaopeng Meng*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/c6a01432c8138d46ba39957a8250e027-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/c6a01432c8138d46ba39957a8250e027-Abstract.html)

        **Abstract**:

        The Dynamic Pickup and Delivery Problem (DPDP) is an essential problem in the logistics domain, which is NP-hard. The objective is to dynamically schedule vehicles among multiple sites to serve the online generated orders such that the overall transportation cost could be minimized. The critical challenge of DPDP is the orders are not known a priori, i.e., the orders are dynamically generated in real-time. To address this problem, existing methods partition the overall DPDP into fixed-size sub-problems by caching online generated orders and solve each sub-problem, or on this basis to utilize the predicted future orders to optimize each sub-problem further. However, the solution quality and efficiency of these methods are unsatisfactory, especially when the problem scale is very large. In this paper, we propose a novel hierarchical optimization framework to better solve large-scale DPDPs. Specifically, we design an upper-level agent to dynamically partition the DPDP into a series of sub-problems with different scales to optimize vehicles routes towards globally better solutions. Besides, a lower-level agent is designed to efficiently solve each sub-problem by incorporating the strengths of classical operational research-based methods with reinforcement learning-based policies. To verify the effectiveness of the proposed framework, real historical data is collected from the order dispatching system of Huawei Supply Chain Business Unit and used to build a functional simulator. Extensive offline simulation and online testing conducted on the industrial order dispatching system justify the superior performance of our framework over existing baselines.

        ----

        ## [1808] Spatio-Temporal Variational Gaussian Processes

        **Authors**: *Oliver Hamelijnck, William J. Wilkinson, Niki A. Loppi, Arno Solin, Theodoros Damoulas*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/c6b8c8d762da15fa8dbbdfb6baf9e260-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/c6b8c8d762da15fa8dbbdfb6baf9e260-Abstract.html)

        **Abstract**:

        We introduce a scalable approach to Gaussian process inference that combines spatio-temporal filtering with natural gradient variational inference, resulting in a non-conjugate GP method for multivariate data that scales linearly with respect to time. Our natural gradient approach enables application of parallel filtering and smoothing, further reducing the temporal span complexity to be logarithmic in the number of time steps. We derive a sparse approximation that constructs a state-space model over a reduced set of spatial inducing points, and show that for separable Markov kernels the full and sparse cases exactly recover the standard variational GP, whilst exhibiting favourable computational properties. To further improve the spatial scaling we propose a mean-field assumption of independence between spatial locations which, when coupled with sparsity and parallelisation, leads to an efficient and accurate method for large spatio-temporal problems.

        ----

        ## [1809] MERLOT: Multimodal Neural Script Knowledge Models

        **Authors**: *Rowan Zellers, Ximing Lu, Jack Hessel, Youngjae Yu, Jae Sung Park, Jize Cao, Ali Farhadi, Yejin Choi*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/c6d4eb15f1e84a36eff58eca3627c82e-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/c6d4eb15f1e84a36eff58eca3627c82e-Abstract.html)

        **Abstract**:

        As humans, we understand events in the visual world contextually, performing multimodal reasoning across time to make inferences about the past, present, and future. We introduce MERLOT, a model that learns multimodal script knowledge by watching millions of YouTube videos with transcribed speech -- in an entirely label-free, self-supervised manner. By pretraining with a mix of both frame-level (spatial) and video-level (temporal) objectives, our model not only learns to match images to temporally corresponding words, but also to contextualize what is happening globally over time. As a result, MERLOT exhibits strong out-of-the-box representations of temporal commonsense, and achieves state-of-the-art performance on 12 different video QA datasets when finetuned. It also transfers well to the world of static images, allowing models to reason about the dynamic context behind visual scenes. On Visual Commonsense Reasoning, MERLOT~answers questions correctly with 80.6\% accuracy, outperforming state-of-the-art models of similar size by over 3\%, even those that make heavy use of auxiliary supervised data (like object bounding boxes).Ablation analyses demonstrate the complementary importance of: 1) training on videos versus static images; 2) scaling the magnitude and diversity of the pretraining video corpus; and 3) using diverse objectives that encourage full-stack multimodal reasoning, from the recognition to cognition level.

        ----

        ## [1810] Fast Approximate Dynamic Programming for Infinite-Horizon Markov Decision Processes

        **Authors**: *Mohamad Amin Sharifi Kolarijani, G. F. Max, Peyman Mohajerin Esfahani*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/c6f798b844366ccd65d99bc7f31e0e02-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/c6f798b844366ccd65d99bc7f31e0e02-Abstract.html)

        **Abstract**:

        In this study, we consider the infinite-horizon, discounted cost, optimal control of stochastic nonlinear systems with separable cost and constraints in the state and input variables. Using the linear-time Legendre transform, we propose a novel numerical scheme for implementation of the corresponding value iteration (VI) algorithm in the conjugate domain. Detailed analyses of the convergence, time complexity, and error of the proposed algorithm are provided. In particular, with a discretization of size $X$ and $U$ for the state and input spaces, respectively, the proposed approach reduces the time complexity of each iteration in the VI algorithm from $O(XU)$ to $O(X+U)$, by replacing the minimization operation in the primal domain with a simple addition in the conjugate domain.

        ----

        ## [1811] Adaptive Risk Minimization: Learning to Adapt to Domain Shift

        **Authors**: *Marvin Zhang, Henrik Marklund, Nikita Dhawan, Abhishek Gupta, Sergey Levine, Chelsea Finn*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/c705112d1ec18b97acac7e2d63973424-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/c705112d1ec18b97acac7e2d63973424-Abstract.html)

        **Abstract**:

        A fundamental assumption of most machine learning algorithms is that the training and test data are drawn from the same underlying distribution. However, this assumption is violated in almost all practical applications: machine learning systems are regularly tested under distribution shift, due to changing temporal correlations, atypical end users, or other factors. In this work, we consider the problem setting of domain generalization, where the training data are structured into domains and there may be multiple test time shifts, corresponding to new domains or domain distributions. Most prior methods aim to learn a single robust model or invariant feature space that performs well on all domains. In contrast, we aim to learn models that adapt at test time to domain shift using unlabeled test points. Our primary contribution is to introduce the framework of adaptive risk minimization (ARM), in which models are directly optimized for effective adaptation to shift by learning to adapt on the training domains. Compared to prior methods for robustness, invariance, and adaptation, ARM methods provide performance gains of 1-4% test accuracy on a number of image classification problems exhibiting domain shift.

        ----

        ## [1812] Learning State Representations from Random Deep Action-conditional Predictions

        **Authors**: *Zeyu Zheng, Vivek Veeriah, Risto Vuorio, Richard L. Lewis, Satinder Singh*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/c71df24045cfddab4a963d3ac9bdc9a3-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/c71df24045cfddab4a963d3ac9bdc9a3-Abstract.html)

        **Abstract**:

        Our main contribution in this work is an empirical finding that random General Value Functions (GVFs), i.e., deep action-conditional predictions---random both in what feature of observations they predict as well as in the sequence of actions the predictions are conditioned upon---form good auxiliary tasks for reinforcement learning (RL) problems. In particular, we show that random deep action-conditional predictions when used as auxiliary tasks yield state representations that produce control performance competitive with state-of-the-art hand-crafted auxiliary tasks like value prediction, pixel control, and CURL in both Atari and DeepMind Lab tasks. In another set of experiments we stop the gradients from the RL part of the network to the state representation learning part of the network and show, perhaps surprisingly, that the auxiliary tasks alone are sufficient to learn state representations good enough to outperform an end-to-end trained actor-critic baseline. We opensourced our code at https://github.com/Hwhitetooth/random_gvfs.

        ----

        ## [1813] Mixability made efficient: Fast online multiclass logistic regression

        **Authors**: *Rémi Jézéquel, Pierre Gaillard, Alessandro Rudi*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/c74214a3877c4d8297ac96217d5189b7-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/c74214a3877c4d8297ac96217d5189b7-Abstract.html)

        **Abstract**:

        Mixability has been shown to be a powerful tool to obtain algorithms with optimal regret. However, the resulting methods often suffer from high computational complexity which has reduced their practical applicability. For example, in the case of multiclass logistic regression, the aggregating forecaster (see Foster et al. 2018) achieves a regret of $O(\log(Bn))$ whereas Online Newton Step achieves $O(e^B\log(n))$ obtaining a double exponential gain in $B$ (a bound on the norm of comparative functions). However, this high statistical performance is at the price of a prohibitive computational complexity $O(n^{37})$.In this paper, we use quadratic surrogates to make aggregating forecasters more efficient. We show that the resulting algorithm has still high statistical performance for a large class of losses. In particular, we derive an algorithm for multiclass regression with a regret bounded by $O(B\log(n))$ and computational complexity of only $O(n^4)$.

        ----

        ## [1814] Tracking People with 3D Representations

        **Authors**: *Jathushan Rajasegaran, Georgios Pavlakos, Angjoo Kanazawa, Jitendra Malik*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/c74c4bf0dad9cbae3d80faa054b7d8ca-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/c74c4bf0dad9cbae3d80faa054b7d8ca-Abstract.html)

        **Abstract**:

        We present a novel approach for tracking multiple people in video. Unlike past approaches which employ 2D representations, we focus on using 3D representations of people, located in three-dimensional space. To this end, we develop a method, Human Mesh and Appearance Recovery (HMAR) which in addition to extracting the 3D geometry of the person as a SMPL mesh, also extracts appearance as a texture map on the triangles of the mesh. This serves as a 3D representation for appearance that is robust to viewpoint and pose changes. Given a video clip, we first detect bounding boxes corresponding to people, and for each one, we extract 3D appearance, pose, and location information using HMAR. These embedding vectors are then sent to a transformer, which performs spatio-temporal aggregation of the representations over the duration of the sequence. The similarity of the resulting representations is used to solve for associations that assigns each person to a tracklet. We evaluate our approach on the Posetrack, MuPoTs and AVA datasets.  We find that 3D representations are more effective than 2D representations for tracking in these settings, and we obtain state-of-the-art performance. Code and results are available at: https://brjathu.github.io/T3DP.

        ----

        ## [1815] Off-Policy Risk Assessment in Contextual Bandits

        **Authors**: *Audrey Huang, Liu Leqi, Zachary C. Lipton, Kamyar Azizzadenesheli*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/c7502c55f8db540625b59d9a42638520-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/c7502c55f8db540625b59d9a42638520-Abstract.html)

        **Abstract**:

        Even when unable to run experiments, practitioners can evaluate prospective policies, using previously logged data. However, while the bandits literature has adopted a diverse set of objectives, most research on off-policy evaluation to date focuses on the expected reward. In this paper, we introduce Lipschitz risk functionals, a broad class of objectives that subsumes conditional value-at-risk (CVaR), variance, mean-variance, many distorted risks, and CPT risks, among others. We propose Off-Policy Risk Assessment (OPRA), a framework that first estimates a target policy's CDF and then generates plugin estimates for any collection of Lipschitz risks, providing finite sample guarantees that hold simultaneously over the entire class. We instantiate OPRA with both importance sampling and doubly robust estimators. Our primary theoretical contributions are (i) the first uniform concentration inequalities for both CDF estimators in contextual bandits and (ii) error bounds on our Lipschitz risk estimates, which all converge at a rate of $O(1/\sqrt{n})$.

        ----

        ## [1816] Adaptive Denoising via GainTuning

        **Authors**: *Sreyas Mohan, Joshua L. Vincent, Ramon Manzorro, Peter A. Crozier, Carlos Fernandez-Granda, Eero P. Simoncelli*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/c7558e9d1f956b016d1fdba7ea132378-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/c7558e9d1f956b016d1fdba7ea132378-Abstract.html)

        **Abstract**:

        Deep convolutional neural networks (CNNs) for image denoising are typically trained on large datasets. These models achieve the current state of the art, but they do not generalize well to data that deviate from the training distribution. Recent work has shown that it is possible to train denoisers on a single noisy image. These models adapt to the features of the test image, but their performance is limited by the small amount of information used to train them. Here we propose "GainTuning'', a methodology by which CNN models pre-trained on large datasets can be adaptively and selectively adjusted for individual test images. To avoid overfitting, GainTuning optimizes a single multiplicative scaling parameter (the “Gain”) of each channel in the convolutional layers of the CNN. We show that GainTuning improves state-of-the-art CNNs on standard image-denoising benchmarks, boosting their denoising performance on nearly every image in a held-out test set. These adaptive improvements are even more substantial for test images differing systematically from the training data, either in noise level or image type. We illustrate the potential of adaptive GainTuning in a scientific application to transmission-electron-microscope images, using a CNN that is pre-trained on synthetic data. In contrast to the existing methodology, GainTuning is able to faithfully reconstruct the structure of catalytic nanoparticles from these data at extremely low signal-to-noise ratios.

        ----

        ## [1817] Optimal Sketching for Trace Estimation

        **Authors**: *Shuli Jiang, Hai Pham, David P. Woodruff, Qiuyi (Richard) Zhang*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/c77bfda61a0204d445185053e6a9a8fe-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/c77bfda61a0204d445185053e6a9a8fe-Abstract.html)

        **Abstract**:

        Matrix trace estimation is ubiquitous in machine learning applications and has traditionally relied on Hutchinson's method, which requires $O(\log(1/\delta)/\epsilon^2)$ matrix-vector product queries to achieve a $(1 \pm \epsilon)$-multiplicative approximation to $\text{trace}(A)$ with failure probability $\delta$ on positive-semidefinite input matrices $A$. Recently, the Hutch++ algorithm was proposed, which reduces the number of matrix-vector queries from $O(1/\epsilon^2)$ to the optimal $O(1/\epsilon)$, and the algorithm succeeds with constant probability. However, in the high probability setting, the non-adaptive Hutch++ algorithm suffers an extra $O(\sqrt{\log(1/\delta)})$ multiplicative factor in its query complexity. Non-adaptive methods are important, as they correspond to sketching algorithms, which are mergeable, highly parallelizable, and provide low-memory streaming algorithms as well as low-communication distributed protocols. In this work, we close the gap between non-adaptive and adaptive algorithms, showing that even non-adaptive algorithms can achieve $O(\sqrt{\log(1/\delta)}/\epsilon + \log(1/\delta))$ matrix-vector products. In addition, we prove matching lower bounds demonstrating that, up to a $\log \log(1/\delta)$ factor, no further improvement in the dependence on $\delta$ or $\epsilon$ is possible by any non-adaptive algorithm. Finally, our experiments demonstrate the superior performance of our sketch over the adaptive Hutch++ algorithm, which is less parallelizable, as well as over the non-adaptive Hutchinson's method.

        ----

        ## [1818] Estimating Multi-cause Treatment Effects via Single-cause Perturbation

        **Authors**: *Zhaozhi Qian, Alicia Curth, Mihaela van der Schaar*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/c793b3be8f18731f2a4c627fb3c6c63d-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/c793b3be8f18731f2a4c627fb3c6c63d-Abstract.html)

        **Abstract**:

        Most existing methods for conditional average treatment effect estimation are designed to estimate the effect of a single cause - only one variable can be intervened on at one time. However, many applications involve simultaneous intervention on multiple variables, which leads to multi-cause treatment effect problems. The multi-cause problem is challenging because one needs to overcome the confounding bias for a large number of treatment groups, each with a different cause combination. The combinatorial nature of the problem also leads to severe data scarcity - we only observe one factual outcome out of many potential outcomes. In this work, we propose Single-cause Perturbation (SCP), a novel two-step procedure to estimate the multi-cause treatment effect. SCP starts by augmenting the observational dataset with the estimated potential outcomes under single-cause interventions. It then performs covariate adjustment on the augmented dataset to obtain the estimator. SCP is agnostic to the exact choice of algorithm in either step. We show formally that the procedure is valid under standard assumptions in causal inference. We demonstrate the performance gain of SCP on extensive synthetic and semi-synthetic experiments.

        ----

        ## [1819] Be Confident! Towards Trustworthy Graph Neural Networks via Confidence Calibration

        **Authors**: *Xiao Wang, Hongrui Liu, Chuan Shi, Cheng Yang*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/c7a9f13a6c0940277d46706c7ca32601-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/c7a9f13a6c0940277d46706c7ca32601-Abstract.html)

        **Abstract**:

        Despite Graph Neural Networks (GNNs) have achieved remarkable accuracy, whether the results are trustworthy is still unexplored. Previous studies suggest that many modern neural networks are over-confident on the predictions, however, surprisingly, we discover that GNNs are primarily in the opposite direction, i.e., GNNs are under-confident. Therefore, the confidence calibration for GNNs is highly desired. In this paper, we propose a novel trustworthy GNN model by designing a topology-aware post-hoc calibration function. Specifically, we first verify that the confidence distribution in a graph has homophily property, and this finding inspires us to design a calibration GNN model (CaGCN) to learn the calibration function. CaGCN is able to obtain a unique transformation from logits of GNNs to the calibrated confidence for each node, meanwhile, such transformation is able to preserve the order between classes, satisfying the accuracy-preserving property. Moreover, we apply the calibration GNN to self-training framework, showing that more trustworthy pseudo labels can be obtained with the calibrated confidence and further improve the performance. Extensive experiments demonstrate the effectiveness of our proposed model in terms of both calibration and accuracy.

        ----

        ## [1820] Learning Riemannian metric for disease progression modeling

        **Authors**: *Samuel Gruffaz, Pierre-Emmanuel Poulet, Etienne Maheux, Bruno Jedynak, Stanley Durrleman*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/c7b90b0fc23725f299b47c5224e6ec0d-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/c7b90b0fc23725f299b47c5224e6ec0d-Abstract.html)

        **Abstract**:

        Linear mixed-effect models provide a natural baseline for estimating disease progression using longitudinal data. They provide interpretable models at the cost of modeling assumptions on the progression profiles and their variability across subjects. A significant improvement is to embed the data in a Riemannian manifold and learn patient-specific trajectories distributed around a central geodesic. A few interpretable parameters characterize subject trajectories at the cost of a prior choice of the metric, which determines the shape of the trajectories. We extend this approach by learning the metric from the data allowing more flexibility while keeping the interpretability. Specifically, we learn the metric as the push-forward of the Euclidean metric by a diffeomorphism. This diffeomorphism is estimated iteratively as the composition of radial basis functions belonging to a reproducible kernel Hilbert space. The metric update allows us to improve the forecasting of imaging and clinical biomarkers in the Alzheimerâ€™s Disease Neuroimaging Initiative (ADNI) cohort. Our results compare favorably to the 56 methods benchmarked in the TADPOLE challenge.

        ----

        ## [1821] Bias and variance of the Bayesian-mean decoder

        **Authors**: *Arthur Prat-Carrabin, Michael Woodford*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/c7c3e78e3c9d26cc1158a8735d548eaa-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/c7c3e78e3c9d26cc1158a8735d548eaa-Abstract.html)

        **Abstract**:

        Perception, in theoretical neuroscience, has been modeled as the encoding of external stimuli into internal signals, which are then decoded. The Bayesian mean is an important decoder, as it is optimal for purposes of both estimation and discrimination. We present widely-applicable approximations to the bias and to the variance of the Bayesian mean, obtained under the minimal and biologically-relevant assumption that the encoding results from a series of independent, though not necessarily identically-distributed, signals. Simulations substantiate the accuracy of our approximations in the small-noise regime. The bias of the Bayesian mean comprises two components: one driven by the prior, and one driven by the precision of the encoding. If the encoding is 'efficient', the two components have opposite effects; their relative strengths are determined by the objective that the encoding optimizes. The experimental literature on perception reports both 'Bayesian' biases directed towards prior expectations, and opposite, 'anti-Bayesian' biases. We show that different tasks are indeed predicted to yield such contradictory biases, under a consistently-optimal encoding-decoding model. Moreover, we recover Wei and Stocker's "law of human perception", a relation between the bias of the Bayesian mean and the derivative of its variance, and show how the coefficient of proportionality in this law depends on the task at hand. Our results provide a parsimonious theory of optimal perception under constraints, in which encoding and decoding are adapted both to the prior and to the task faced by the observer.

        ----

        ## [1822] MIRACLE: Causally-Aware Imputation via Learning Missing Data Mechanisms

        **Authors**: *Trent Kyono, Yao Zhang, Alexis Bellot, Mihaela van der Schaar*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/c80bcf42c220b8f5c41f85344242f1b0-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/c80bcf42c220b8f5c41f85344242f1b0-Abstract.html)

        **Abstract**:

        Missing data is an important problem in machine learning practice. Starting from the premise that imputation methods should preserve the causal structure of the data, we develop a regularization scheme that encourages any baseline imputation method to be causally consistent with the underlying data generating mechanism. Our proposal is a causally-aware imputation algorithm (MIRACLE). MIRACLE iteratively refines the imputation of a baseline by simultaneously modeling the missingness generating mechanism, encouraging imputation to be consistent with the causal structure of the data. We conduct extensive experiments on synthetic and a variety of publicly available datasets to show that MIRACLE is able to consistently improve imputation over a variety of benchmark methods across all three missingness scenarios: at random, completely at random, and not at random.

        ----

        ## [1823] Efficient Training of Visual Transformers with Small Datasets

        **Authors**: *Yahui Liu, Enver Sangineto, Wei Bi, Nicu Sebe, Bruno Lepri, Marco De Nadai*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/c81e155d85dae5430a8cee6f2242e82c-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/c81e155d85dae5430a8cee6f2242e82c-Abstract.html)

        **Abstract**:

        Visual Transformers (VTs) are emerging as an architectural paradigm alternative to Convolutional networks (CNNs). Differently from CNNs, VTs can capture global relations between image elements and they potentially have a larger representation capacity. However, the lack of the typical convolutional inductive bias makes these models more data hungry than common CNNs. In fact, some local properties of the visual domain which are embedded in the CNN architectural design, in VTs should be learned from samples. In this paper, we empirically analyse different VTs, comparing their robustness in a small training set regime, and we show that, despite having a comparable accuracy when trained on ImageNet, their performance on smaller datasets can be largely different. Moreover, we propose an auxiliary self-supervised task which can extract additional information from images with only a negligible computational overhead. This task encourages the VTs to learn  spatial relations within an image and makes the VT training much more robust when training data is scarce. Our task is used jointly with the standard (supervised) training and it does not depend on specific architectural choices, thus it can be easily plugged in the existing VTs. Using an extensive evaluation with different VTs and datasets, we show that our method can improve (sometimes dramatically) the final accuracy of the VTs. Our code is available at: https://github.com/yhlleo/VTs-Drloc.

        ----

        ## [1824] Small random initialization is akin to spectral learning: Optimization and generalization guarantees for overparameterized low-rank matrix reconstruction

        **Authors**: *Dominik Stöger, Mahdi Soltanolkotabi*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/c82836ed448c41094025b4a872c5341e-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/c82836ed448c41094025b4a872c5341e-Abstract.html)

        **Abstract**:

        Recently there has been significant theoretical progress on understanding the convergence and generalization of gradient-based methods on nonconvex losses with overparameterized models. Nevertheless, many aspects of optimization and generalization and in particular the critical role of small random initialization are not fully understood. In this paper, we take a step towards demystifying this role by proving that small random initialization followed by a few iterations of gradient descent behaves akin to popular spectral methods. We also show that this implicit spectral bias from small random initialization, which is provably more prominent for overparameterized models, also puts the gradient descent iterations on a particular trajectory towards solutions that are not only globally optimal but also generalize well. Concretely, we focus on the problem of reconstructing a low-rank matrix from a few measurements via a natural nonconvex formulation. In this setting, we show that the trajectory of the gradient descent iterations from small random initialization can be approximately decomposed into three phases: (I) a spectral or alignment phase where we show that that the iterates have an implicit spectral bias akin to spectral initialization allowing us to show that at the end of this phase the column space of the iterates and the underlying low-rank matrix are sufficiently aligned, (II) a saddle avoidance/refinement phase where we show that the trajectory of the gradient iterates moves away from certain degenerate saddle points, and (III) a local refinement phase where we show that after avoiding the saddles the iterates converge quickly to the underlying low-rank matrix. Underlying our analysis are insights for the analysis of overparameterized nonconvex optimization schemes that may have implications for computational problems beyond low-rank reconstruction.

        ----

        ## [1825] Efficient Combination of Rematerialization and Offloading for Training DNNs

        **Authors**: *Olivier Beaumont, Lionel Eyraud-Dubois, Alena Shilova*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/c8461bf13fca8a2b9912ab2eb1668e4b-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/c8461bf13fca8a2b9912ab2eb1668e4b-Abstract.html)

        **Abstract**:

        Rematerialization and offloading are two well known strategies to save memory during the training phase of deep neural networks, allowing data scientists to consider larger models, batch sizes or higher resolution data. Rematerialization trades memory for computation time, whereas Offloading trades memory for data movements. As these two resources are independent, it is appealing to consider the simultaneous combination of both strategies to save even more memory. We precisely model the costs and constraints corresponding to Deep Learning frameworks such as PyTorch or Tensorflow, we propose optimal algorithms to find a valid sequence of memory-constrained operations and finally, we evaluate the performance of proposed algorithms on realistic networks and computation platforms. Our experiments show that the possibility to offload can remove one third of the overhead of rematerialization, and that together they can reduce the memory used for activations by a factor 4 to 6, with an overhead below 20%.

        ----

        ## [1826] Particle Cloud Generation with Message Passing Generative Adversarial Networks

        **Authors**: *Raghav Kansal, Javier M. Duarte, Hao Su, Breno Orzari, Thiago Tomei, Maurizio Pierini, Mary Touranakou, Jean-Roch Vlimant, Dimitrios Gunopulos*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/c8512d142a2d849725f31a9a7a361ab9-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/c8512d142a2d849725f31a9a7a361ab9-Abstract.html)

        **Abstract**:

        In high energy physics (HEP), jets are collections of correlated particles produced ubiquitously in particle collisions such as those at the CERN Large Hadron Collider (LHC). Machine learning (ML)-based generative models, such as generative adversarial networks (GANs), have the potential to significantly accelerate LHC jet simulations. However, despite jets having a natural representation as a set of particles in momentum-space, a.k.a. a particle cloud, there exist no generative models applied to such a dataset. In this work, we introduce a new particle cloud dataset (JetNet), and apply to it existing point cloud GANs. Results are evaluated using (1) 1-Wasserstein distances between high- and low-level feature distributions, (2) a newly developed Fr√©chet ParticleNet Distance, and (3) the coverage and (4) minimum matching distance metrics. Existing GANs are found to be inadequate for physics applications, hence we develop a new message passing GAN (MPGAN), which outperforms existing point cloud GANs on virtually every metric and shows promise for use in HEP. We propose JetNet as a novel point-cloud-style dataset for the ML community to experiment with, and set MPGAN as a benchmark to improve upon for future generative models. Additionally, to facilitate research and improve accessibility and reproducibility in this area, we release the open-source JetNet Python package with interfaces for particle cloud datasets, implementations for evaluation and loss metrics, and more tools for ML in HEP development.

        ----

        ## [1827] CoFiNet: Reliable Coarse-to-fine Correspondences for Robust PointCloud Registration

        **Authors**: *Hao Yu, Fu Li, Mahdi Saleh, Benjamin Busam, Slobodan Ilic*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/c85b2ea9a678e74fdc8bafe5d0707c31-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/c85b2ea9a678e74fdc8bafe5d0707c31-Abstract.html)

        **Abstract**:

        We study the problem of extracting correspondences between a pair of point clouds for registration. For correspondence retrieval, existing works benefit from matching sparse keypoints detected from dense points but usually struggle to guarantee their repeatability. To address this issue, we present CoFiNet - Coarse-to-Fine Network which extracts hierarchical correspondences from coarse to fine without keypoint detection. On a coarse scale and guided by a weighting scheme, our model firstly learns to match down-sampled nodes whose vicinity points share more overlap, which significantly shrinks the search space of a consecutive stage. On a finer scale, node proposals are consecutively expanded to patches that consist of groups of points together with associated descriptors. Point correspondences are then refined from the overlap areas of corresponding patches, by a density-adaptive matching module capable to deal with varying point density. Extensive evaluation of CoFiNet on both indoor and outdoor standard benchmarks shows our superiority over existing methods. Especially on 3DLoMatch where point clouds share less overlap, CoFiNet significantly outperforms state-of-the-art approaches by at least 5% on Registration Recall, with at most two-third of their parameters.

        ----

        ## [1828] Partial success in closing the gap between human and machine vision

        **Authors**: *Robert Geirhos, Kantharaju Narayanappa, Benjamin Mitzkus, Tizian Thieringer, Matthias Bethge, Felix A. Wichmann, Wieland Brendel*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/c8877cff22082a16395a57e97232bb6f-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/c8877cff22082a16395a57e97232bb6f-Abstract.html)

        **Abstract**:

        A few years ago, the first CNN surpassed human performance on ImageNet. However, it soon became clear that machines lack robustness on more challenging test cases, a major obstacle towards deploying machines "in the wild" and towards obtaining better computational models of human visual perception. Here we ask: Are we making progress in closing the gap between human and machine vision? To answer this question, we tested human observers on a broad range of out-of-distribution (OOD) datasets, recording 85,120 psychophysical trials across 90 participants. We then investigated a range of promising machine learning developments that crucially deviate from standard supervised CNNs along three axes: objective function (self-supervised, adversarially trained, CLIP language-image training), architecture (e.g. vision transformers), and dataset size (ranging from 1M to 1B).Our findings are threefold. (1.) The longstanding distortion robustness gap between humans and CNNs is closing, with the best models now exceeding human feedforward performance on most of the investigated OOD datasets. (2.) There is still a substantial image-level consistency gap, meaning that humans make different errors than models. In contrast, most models systematically agree in their categorisation errors, even substantially different ones like contrastive self-supervised vs. standard supervised models. (3.) In many cases, human-to-model consistency improves when training dataset size is increased by one to three orders of magnitude. Our results give reason for cautious optimism: While there is still much room for improvement, the behavioural difference between human and machine vision is narrowing. In order to measure future progress, 17 OOD datasets with image-level human behavioural data and evaluation code are provided as a toolbox and benchmark at: https://github.com/bethgelab/model-vs-human/

        ----

        ## [1829] LLC: Accurate, Multi-purpose Learnt Low-dimensional Binary Codes

        **Authors**: *Aditya Kusupati, Matthew Wallingford, Vivek Ramanujan, Raghav Somani, Jae Sung Park, Krishna Pillutla, Prateek Jain, Sham M. Kakade, Ali Farhadi*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/c88d8d0a6097754525e02c2246d8d27f-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/c88d8d0a6097754525e02c2246d8d27f-Abstract.html)

        **Abstract**:

        Learning binary representations of instances and classes is a classical problem with several high potential applications. In modern settings, the compression of high-dimensional neural representations to low-dimensional binary codes is a challenging task and often require large bit-codes to be accurate. In this work, we propose a novel method for $\textbf{L}$earning $\textbf{L}$ow-dimensional binary $\textbf{C}$odes $(\textbf{LLC})$ for instances as well as classes. Our method does ${\textit{not}}$ require any side-information, like annotated attributes or label meta-data, and learns extremely low-dimensional binary codes ($\approx 20$ bits for ImageNet-1K). The learnt codes are super-efficient while still ensuring $\textit{nearly optimal}$ classification accuracy for ResNet50 on ImageNet-1K. We demonstrate that the learnt codes capture intrinsically important features in the data, by discovering an intuitive taxonomy over classes. We further quantitatively measure the quality of our codes by applying it to the efficient image retrieval as well as out-of-distribution (OOD) detection problems. For ImageNet-100 retrieval problem, our learnt binary codes outperform $16$ bit HashNet using only $10$ bits and also are as accurate as $10$ dimensional real representations. Finally, our learnt binary codes can perform OOD detection, out-of-the-box, as accurately as a baseline that needs $\approx3000$ samples to tune its threshold, while we require ${\textit{none}}$. Code is open-sourced at https://github.com/RAIVNLab/LLC.

        ----

        ## [1830] Analytic Insights into Structure and Rank of Neural Network Hessian Maps

        **Authors**: *Sidak Pal Singh, Gregor Bachmann, Thomas Hofmann*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/c900ced7451da79502d29aa37ebb7b60-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/c900ced7451da79502d29aa37ebb7b60-Abstract.html)

        **Abstract**:

        The Hessian of a neural network captures parameter interactions through second-order derivatives of the loss. It is a fundamental object of study, closely tied to various problems in deep learning, including model design, optimization, and generalization. Most prior work has been empirical, typically focusing on low-rank approximations and heuristics that are blind to the network structure.  In contrast, we develop theoretical tools to analyze the range of the Hessian map, which provide us with a precise understanding of its rank deficiency and the structural reasons behind it. This yields exact formulas and tight upper bounds for the Hessian rank of deep linear networks --- allowing for an elegant interpretation in terms of rank deficiency. Moreover, we demonstrate that our bounds remain faithful as an estimate of the numerical Hessian rank, for a larger class of models such as rectified and hyperbolic tangent networks. Further, we also investigate the implications of model architecture (e.g.~width, depth, bias) on the rank deficiency. Overall, our work provides novel insights into the source and extent of redundancy in overparameterized neural networks.

        ----

        ## [1831] Well-tuned Simple Nets Excel on Tabular Datasets

        **Authors**: *Arlind Kadra, Marius Lindauer, Frank Hutter, Josif Grabocka*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/c902b497eb972281fb5b4e206db38ee6-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/c902b497eb972281fb5b4e206db38ee6-Abstract.html)

        **Abstract**:

        Tabular datasets are the last "unconquered castle" for deep learning, with traditional ML methods like Gradient-Boosted Decision Trees still performing strongly even against recent specialized neural architectures. In this paper, we hypothesize that the key to boosting the performance of neural networks lies in rethinking the joint and simultaneous application of a large set of modern regularization techniques. As a result, we propose regularizing plain Multilayer Perceptron (MLP) networks by searching for the optimal combination/cocktail of 13 regularization techniques for each dataset using a joint optimization over the decision on which regularizers to apply and their subsidiary hyperparameters. We empirically assess the impact of these regularization cocktails for MLPs in a large-scale empirical study comprising 40 tabular datasets and demonstrate that (i) well-regularized plain MLPs significantly outperform recent state-of-the-art specialized neural network architectures, and (ii) they even outperform strong traditional ML methods, such as XGBoost.

        ----

        ## [1832] POODLE: Improving Few-shot Learning via Penalizing Out-of-Distribution Samples

        **Authors**: *Duong H. Le, Khoi D. Nguyen, Khoi Nguyen, Quoc-Huy Tran, Rang Nguyen, Binh-Son Hua*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/c91591a8d461c2869b9f535ded3e213e-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/c91591a8d461c2869b9f535ded3e213e-Abstract.html)

        **Abstract**:

        In this work, we propose to use out-of-distribution samples, i.e., unlabeled samples coming from outside the target classes, to improve few-shot learning. Specifically, we exploit the easily available out-of-distribution samples to drive the classifier to avoid irrelevant features by maximizing the distance from prototypes to out-of-distribution samples while minimizing that of in-distribution samples (i.e., support, query data). Our approach is simple to implement, agnostic to feature extractors, lightweight without any additional cost for pre-training, and applicable to both inductive and transductive settings. Extensive experiments on various standard benchmarks demonstrate that the proposed method consistently improves the performance of pretrained networks with different architectures.

        ----

        ## [1833] Combinatorial Pure Exploration with Bottleneck Reward Function

        **Authors**: *Yihan Du, Yuko Kuroki, Wei Chen*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/c92a10324374fac681719d63979d00fe-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/c92a10324374fac681719d63979d00fe-Abstract.html)

        **Abstract**:

        In this paper, we study the Combinatorial Pure Exploration problem with the Bottleneck reward function (CPE-B) under the fixed-confidence (FC) and fixed-budget (FB) settings.In CPE-B, given a set of base arms and a collection of subsets of base arms (super arms) following a certain combinatorial constraint, a learner sequentially plays a base arm and observes its random reward, with the objective of finding the optimal super arm with the maximum bottleneck value, defined as the minimum expected reward of the base arms contained in the super arm.CPE-B captures a variety of practical scenarios such as network routing in communication networks, and its unique challenges fall on how to utilize the bottleneck property to save samples and achieve the statistical optimality. None of the existing CPE studies (most of them assume linear rewards) can be adapted to solve such challenges, and thus we develop brand-new techniques to handle them.For the FC setting, we propose novel algorithms with optimal sample complexity for a broad family of instances and establish a matching lower bound to demonstrate the optimality (within a logarithmic factor).For the FB setting, we design an algorithm which achieves the state-of-the-art error probability guarantee and is the first to run efficiently on fixed-budget path instances, compared to existing CPE algorithms. Our experimental results on the top-$k$, path and matching instances validate the empirical superiority of the proposed algorithms over their baselines.

        ----

        ## [1834] Densely connected normalizing flows

        **Authors**: *Matej Grcic, Ivan Grubisic, Sinisa Segvic*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/c950cde9b3f83f41721788e3315a14a3-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/c950cde9b3f83f41721788e3315a14a3-Abstract.html)

        **Abstract**:

        Normalizing flows are bijective mappings between inputs and latent representations with a fully factorized distribution. They are very attractive due to exact likelihood evaluation and efficient sampling. However, their effective capacity is often insufficient since the bijectivity constraint limits the model width. We address this issue by incrementally padding intermediate representations with noise. We precondition the noise in accordance with previous invertible units, which we describe as cross-unit coupling. Our invertible glow-like modules increase the model expressivity by fusing a densely connected block with Nyström self-attention. We refer to our architecture as DenseFlow since both cross-unit and intra-module couplings rely on dense connectivity. Experiments show significant improvements due to the proposed contributions and reveal state-of-the-art density estimation under moderate computing budgets.

        ----

        ## [1835] Snowflake: Scaling GNNs to high-dimensional continuous control via parameter freezing

        **Authors**: *Charlie Blake, Vitaly Kurin, Maximilian Igl, Shimon Whiteson*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/c952ce98517ac529c60744ac28364b03-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/c952ce98517ac529c60744ac28364b03-Abstract.html)

        **Abstract**:

        Recent research has shown that graph neural networks (GNNs) can learn policies for locomotion control that are as effective as a typical multi-layer perceptron (MLP), with superior transfer and multi-task performance. However, results have so far been limited to training on small agents, with the performance of GNNs deteriorating rapidly as the number of sensors and actuators grows. A key motivation for the use of GNNs in the supervised learning setting is their applicability to large graphs, but this benefit has not yet been realised for locomotion control. We show that poor scaling in GNNs is a result of increasingly unstable policy updates, caused by overfitting in parts of the network during training. To combat this, we introduce Snowflake, a GNN training method for high-dimensional continuous control that freezes parameters in selected parts of the network. Snowflake significantly boosts the performance of GNNs for locomotion control on large agents, now matching the performance of MLPs while offering superior transfer properties.

        ----

        ## [1836] Subgame solving without common knowledge

        **Authors**: *Brian Hu Zhang, Tuomas Sandholm*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/c96c08f8bb7960e11a1239352a479053-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/c96c08f8bb7960e11a1239352a479053-Abstract.html)

        **Abstract**:

        In imperfect-information games, subgame solving is significantly more challenging than in perfect-information games, but in the last few years, such techniques have been developed. They were the key ingredient to the milestone of superhuman play in no-limit Texas hold'em poker. Current subgame-solving techniques analyze the entire common-knowledge closure of the player's current information set, that is, the smallest set of nodes within which it is common knowledge that the current node lies. While this is acceptable in games like poker where the common-knowledge closure is relatively small, many practical games have more complex information structure, which renders the common-knowledge closure impractically large to enumerate or even reasonably approximate. We introduce an approach that overcomes this obstacle, by instead working with only low-order knowledge. Our approach allows an agent, upon arriving at an infoset, to basically prune any node that is no longer reachable, thereby massively reducing the game tree size relative to the common-knowledge subgame. We prove that, as is, our approach can increase exploitability compared to the blueprint strategy. However, we develop three avenues by which safety can be guaranteed. First, safety is guaranteed if the results of subgame solves are incorporated back into the blueprint. Second, we provide a method where safety is achieved by limiting the infosets at which subgame solving is performed. Third, we prove that our approach, when applied at every infoset reached during play, achieves a weaker notion of equilibrium, which we coin affine equilibrium, and which may be of independent interest. We show that affine equilibria cannot be exploited by any Nash strategy of the opponent, so an opponent who wishes to exploit must open herself to counter-exploitation. Even without the safety-guaranteeing additions, experiments on medium-sized games show that our approach always reduced exploitability in practical games even when applied at every infoset, and a depth-limited version of it led to---to our knowledge---the first strong AI for the challenge problem dark chess.

        ----

        ## [1837] Fair Algorithms for Multi-Agent Multi-Armed Bandits

        **Authors**: *Safwan Hossain, Evi Micha, Nisarg Shah*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/c96ebeee051996333b6d70b2da6191b0-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/c96ebeee051996333b6d70b2da6191b0-Abstract.html)

        **Abstract**:

        We propose a multi-agent variant of the classical multi-armed bandit problem, in which there are $N$ agents and $K$ arms, and pulling an arm generates a (possibly different) stochastic reward for each agent. Unlike the classical multi-armed bandit problem, the goal is not to learn the "best arm"; indeed, each agent may perceive a different arm to be the best for her personally. Instead, we seek to learn a fair distribution over the arms. Drawing on a long line of research in economics and computer science, we use the Nash social welfare as our notion of fairness. We design multi-agent variants of three classic multi-armed bandit algorithms and show that they achieve sublinear regret, which is now measured in terms of the lost Nash social welfare. We also extend a classical lower bound, establishing the optimality of one of our algorithms.

        ----

        ## [1838] VAST: Value Function Factorization with Variable Agent Sub-Teams

        **Authors**: *Thomy Phan, Fabian Ritz, Lenz Belzner, Philipp Altmann, Thomas Gabor, Claudia Linnhoff-Popien*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/c97e7a5153badb6576d8939469f58336-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/c97e7a5153badb6576d8939469f58336-Abstract.html)

        **Abstract**:

        Value function factorization (VFF) is a popular approach to cooperative multi-agent reinforcement learning in order to learn local value functions from global rewards. However, state-of-the-art VFF is limited to a handful of agents in most domains. We hypothesize that this is due to the flat factorization scheme, where the VFF operator becomes a performance bottleneck with an increasing number of agents. Therefore, we propose VFF with variable agent sub-teams (VAST). VAST approximates a factorization for sub-teams which can be defined in an arbitrary way and vary over time, e.g., to adapt to different situations. The sub-team values are then linearly decomposed for all sub-team members. Thus, VAST can learn on a more focused and compact input representation of the original VFF operator. We evaluate VAST in three multi-agent domains and show that VAST can significantly outperform state-of-the-art VFF, when the number of agents is sufficiently large.

        ----

        ## [1839] On the Stochastic Stability of Deep Markov Models

        **Authors**: *Ján Drgona, Sayak Mukherjee, Jiaxin Zhang, Frank Liu, Mahantesh Halappanavar*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/c9dd73f5cb96486f5e1e0680e841a550-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/c9dd73f5cb96486f5e1e0680e841a550-Abstract.html)

        **Abstract**:

        Deep Markov models (DMM) are generative models which are scalable and expressive generalization of Markov models for representation, learning, and inference problems. However, the fundamental stochastic stability guarantees of such models have not been thoroughly investigated. In this paper, we present a novel stability analysis method and provide sufficient conditions of DMM's stochastic stability.  The proposed stability analysis is based on the contraction of probabilistic maps modeled by deep neural networks. We make connections between the spectral properties of neural network's weights and different types of used activation function on the stability and overall dynamic behavior of DMMs with Gaussian distributions. Based on the theory, we propose a few practical methods for designing constrained DMMs with guaranteed stability. We empirically substantiate our theoretical results via intuitive numerical experiments using the proposed stability constraints.

        ----

        ## [1840] Multiwavelet-based Operator Learning for Differential Equations

        **Authors**: *Gaurav Gupta, Xiongye Xiao, Paul Bogdan*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/c9e5c2b59d98488fe1070e744041ea0e-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/c9e5c2b59d98488fe1070e744041ea0e-Abstract.html)

        **Abstract**:

        The solution of a partial differential equation can be obtained by computing the inverse operator map between the input and the solution space. Towards this end, we introduce a $\textit{multiwavelet-based neural operator learning scheme}$ that compresses the associated operator's kernel using fine-grained wavelets. By explicitly embedding the inverse multiwavelet filters, we learn the projection of the kernel onto fixed multiwavelet polynomial bases. The projected kernel is trained at multiple scales derived from using repeated computation of multiwavelet transform. This allows learning the complex dependencies at various scales and results in a resolution-independent scheme. Compare to the prior works, we exploit the fundamental properties of the operator's kernel which enable numerically efficient representation. We perform experiments on the Korteweg-de Vries (KdV) equation, Burgers' equation, Darcy Flow, and Navier-Stokes equation. Compared with the existing neural operator approaches, our model shows significantly higher accuracy and achieves state-of-the-art in a range of datasets. For the time-varying equations, the proposed method exhibits a ($2X-10X$) improvement ($0.0018$ ($0.0033$) relative $L2$ error for Burgers' (KdV) equation). By learning the mappings between function spaces, the proposed method has the ability to find the solution of a high-resolution input after learning from lower-resolution data.

        ----

        ## [1841] Intermediate Layers Matter in Momentum Contrastive Self Supervised Learning

        **Authors**: *Aakash Kaku, Sahana Upadhya, Narges Razavian*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/c9f06258da6455f5bf50c5b9260efeff-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/c9f06258da6455f5bf50c5b9260efeff-Abstract.html)

        **Abstract**:

        We show that bringing intermediate layers' representations of two augmented versions of an image closer together in self-supervised learning helps to improve the momentum contrastive (MoCo) method. To this end, in addition to the contrastive loss, we minimize the mean squared error between the intermediate layer representations or make their cross-correlation matrix closer to an identity matrix. Both loss objectives either outperform standard MoCo, or achieve similar performances on three diverse medical imaging datasets: NIH-Chest Xrays, Breast Cancer Histopathology, and Diabetic Retinopathy. The gains of the improved MoCo are especially large in a low-labeled data regime (e.g. 1% labeled data) with an average gain of 5% across three datasets. We analyze the models trained using our novel approach via feature similarity analysis and layer-wise probing. Our analysis reveals that models trained via our approach have higher feature reuse compared to a standard MoCo and learn informative features earlier in the network. Finally, by comparing the output probability distribution of models fine-tuned on small versus large labeled data, we conclude that our proposed method of pre-training leads to lower Kolmogorovâ€“Smirnov distance, as compared to a standard MoCo. This provides additional evidence that our proposed method learns more informative features in the pre-training phase which could be leveraged in a low-labeled data regime.

        ----

        ## [1842] An Efficient Pessimistic-Optimistic Algorithm for Stochastic Linear Bandits with General Constraints

        **Authors**: *Xin Liu, Bin Li, Pengyi Shi, Lei Ying*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/ca460332316d6da84b08b9bcf39b687b-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/ca460332316d6da84b08b9bcf39b687b-Abstract.html)

        **Abstract**:

        This paper considers stochastic linear bandits with general nonlinear constraints. The objective is to maximize the expected cumulative reward over horizon $T$ subject to a set of constraints in each round $\tau\leq T$. We propose a pessimistic-optimistic algorithm for this problem, which is efficient in two aspects. First, the algorithm yields $\tilde{\cal O}\left(\left(\frac{K^{0.75}}{\delta}+d\right)\sqrt{\tau}\right)$ (pseudo) regret in round $\tau\leq T,$ where $K$ is the number of constraints, $d$ is the dimension of the reward feature space, and $\delta$ is a Slater's constant; and  {\em zero}  constraint violation in any round $\tau>\tau',$ where $\tau'$ is  {\em independent} of horizon $T.$ Second, the algorithm is computationally efficient. Our algorithm is based on the primal-dual approach in optimization and includes two components. The primal component is similar to unconstrained stochastic linear bandits (our algorithm uses the linear upper confidence bound algorithm (LinUCB)). The computational complexity of the dual component depends on the number of constraints, but is independent of the sizes of the contextual space, the action space, and the feature space. Thus, the computational complexity of our algorithm is similar to LinUCB for unconstrained stochastic linear bandits.

        ----

        ## [1843] Efficiently Learning One Hidden Layer ReLU Networks From Queries

        **Authors**: *Sitan Chen, Adam R. Klivans, Raghu Meka*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/ca4b5656b7e193e6bb9064c672ac8dce-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/ca4b5656b7e193e6bb9064c672ac8dce-Abstract.html)

        **Abstract**:

        While the problem of PAC learning neural networks from samples has received considerable attention in recent years, in certain settings like model extraction attacks, it is reasonable to imagine having more than just the ability to observe random labeled examples. Motivated by this, we consider the following problem: given \emph{black-box query access} to a neural network $F$, recover $F$ up to some error. Formally, we show that if $F$ is an arbitrary one hidden layer neural network with ReLU activations, there is an algorithm with query complexity and runtime polynomial in all parameters which outputs a network $Fâ€™$ achieving low square loss relative to $F$ with respect to the Gaussian measure. While a number of works in the security literature have proposed and empirically demonstrated the effectiveness of certain algorithms for this problem, ours is to the best of our knowledge the first provable guarantee in this vein.

        ----

        ## [1844] Learning Nonparametric Volterra Kernels with Gaussian Processes

        **Authors**: *Magnus Ross, Michael T. Smith, Mauricio A. Álvarez*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/ca5fbbbddd0c0ff6c01f782c60c9d1b5-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/ca5fbbbddd0c0ff6c01f782c60c9d1b5-Abstract.html)

        **Abstract**:

        This paper introduces a method for the nonparametric Bayesian learning of nonlinear operators, through the use of the Volterra series with kernels represented using Gaussian processes (GPs), which we term the nonparametric Volterra kernels model (NVKM). When the input function to the operator is unobserved and has a GP prior, the NVKM constitutes a powerful method for both single and multiple output regression, and can be viewed as a nonlinear and nonparametric latent force model. When the input function is observed, the NVKM can be used to perform Bayesian system identification. We use recent advances in efficient sampling of explicit functions from GPs to map process realisations through the Volterra series without resorting to numerical integration, allowing scalability through doubly stochastic variational inference, and avoiding the need for Gaussian approximations of the output processes. We demonstrate the performance of the model for both multiple output regression and system identification using standard benchmarks.

        ----

        ## [1845] DiBS: Differentiable Bayesian Structure Learning

        **Authors**: *Lars Lorch, Jonas Rothfuss, Bernhard Schölkopf, Andreas Krause*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/ca6ab34959489659f8c3776aaf1f8efd-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/ca6ab34959489659f8c3776aaf1f8efd-Abstract.html)

        **Abstract**:

        Bayesian structure learning allows inferring Bayesian network structure from data while reasoning about the epistemic uncertainty---a key element towards enabling active causal discovery and designing interventions in real world systems. In this work, we propose a general, fully differentiable framework for Bayesian structure learning (DiBS) that operates in the continuous space of a latent probabilistic graph representation. Contrary to existing work, DiBS is agnostic to the form of the local conditional distributions and allows for joint posterior inference of both the graph structure and the conditional distribution parameters. This makes our formulation directly applicable to posterior inference of nonstandard Bayesian network models, e.g., with nonlinear dependencies encoded by neural networks. Using DiBS, we devise an efficient, general purpose variational inference method for approximating distributions over structural models. In evaluations on simulated and real-world data, our method significantly outperforms related approaches to joint posterior inference.

        ----

        ## [1846] Nonparametric estimation of continuous DPPs with kernel methods

        **Authors**: *Michaël Fanuel, Rémi Bardenet*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/ca8a2d76a5bcc212226417361a5f0740-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/ca8a2d76a5bcc212226417361a5f0740-Abstract.html)

        **Abstract**:

        Determinantal Point Process (DPPs) are statistical models for repulsive point patterns. Both sampling and inference are tractable for DPPs, a rare feature among models with negative dependence that explains their popularity in machine learning and spatial statistics. Parametric and nonparametric inference methods have been proposed in the finite case, i.e. when the point patterns live in a finite ground set. In the continuous case, only parametric methods have been investigated, while nonparametric maximum likelihood for DPPs -- an optimization problem over trace-class operators -- has remained an open question. In this paper, we show that a restricted version of this maximum likelihood (MLE) problem falls within the scope of a recent representer theorem for nonnegative functions in an RKHS. This leads to a finite-dimensional problem, with strong statistical ties to the original MLE. Moreover, we propose, analyze, and demonstrate a fixed point algorithm to solve this finite-dimensional problem. Finally, we also provide a controlled estimate of the correlation kernel of the DPP, thus providing more interpretability.

        ----

        ## [1847] FINE Samples for Learning with Noisy Labels

        **Authors**: *Taehyeon Kim, Jongwoo Ko, Sangwook Cho, Jinhwan Choi, Se-Young Yun*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/ca91c5464e73d3066825362c3093a45f-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/ca91c5464e73d3066825362c3093a45f-Abstract.html)

        **Abstract**:

        Modern deep neural networks (DNNs) become frail when the datasets contain noisy (incorrect) class labels. Robust techniques in the presence of noisy labels can be categorized into two folds: developing noise-robust functions or using noise-cleansing methods by detecting the noisy data. Recently, noise-cleansing methods have been considered as the most competitive noisy-label learning algorithms. Despite their success, their noisy label detectors are often based on heuristics more than a theory, requiring a robust classifier to predict the noisy data with loss values. In this paper, we propose a novel detector for filtering label noise. Unlike most existing methods, we focus on each data's latent representation dynamics and measure the alignment between the latent distribution and each representation using the eigen decomposition of the data gram matrix. Our framework, coined as filtering noisy instances via their eigenvectors (FINE), provides a robust detector with derivative-free simple methods having theoretical guarantees. Under our framework, we propose three applications of the FINE: sample-selection approach, semi-supervised learning approach, and collaboration with noise-robust loss functions. Experimental results show that the proposed methods consistently outperform corresponding baselines for all three applications on various benchmark datasets.

        ----

        ## [1848] Residual2Vec: Debiasing graph embedding with random graphs

        **Authors**: *Sadamori Kojaku, Jisung Yoon, Isabel Constantino, Yong-Yeol Ahn*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/ca9541826e97c4530b07dda2eba0e013-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/ca9541826e97c4530b07dda2eba0e013-Abstract.html)

        **Abstract**:

        Graph embedding maps a graph into a convenient vector-space representation for graph analysis and machine learning applications. Many graph embedding methods hinge on a sampling of context nodes based on random walks. However, random walks can be a biased sampler due to the structural properties of graphs. Most notably, random walks are biased by the degree of each node, where a node is sampled proportionally to its degree. The implication of such biases has not been clear, particularly in the context of graph representation learning. Here, we investigate the impact of the random walks' bias on graph embedding and propose residual2vec, a general graph embedding method that can debias various structural biases in graphs by using random graphs. We demonstrate that this debiasing not only improves link prediction and clustering performance but also allows us to explicitly model salient structural properties in graph embedding.

        ----

        ## [1849] Benign Overfitting in Multiclass Classification: All Roads Lead to Interpolation

        **Authors**: *Ke Wang, Vidya Muthukumar, Christos Thrampoulidis*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/caaa29eab72b231b0af62fbdff89bfce-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/caaa29eab72b231b0af62fbdff89bfce-Abstract.html)

        **Abstract**:

        The growing literature on "benign overfitting" in overparameterized models has been mostly restricted to regression or binary classification settings; however, most success stories of modern machine learning have been recorded in multiclass settings. Motivated by this discrepancy, we study benign overfitting in multiclass linear classification. Specifically, we consider the following popular training algorithms on separable data: (i) empirical risk minimization (ERM) with cross-entropy loss, which converges to the multiclass support vector machine (SVM) solution; (ii) ERM with least-squares loss, which converges to the min-norm interpolating (MNI) solution; and, (iii) the one-vs-all SVM classifier. Our first key finding is that under a simple sufficient condition, all three algorithms lead to classifiers that interpolate the training data and have equal accuracy. When the data is generated from Gaussian mixtures or a multinomial logistic model, this condition holds under high enough effective overparameterization. Second, we derive novel error bounds on the accuracy of the MNI classifier, thereby showing that all three training algorithms lead to benign overfitting under sufficient overparameterization. Ultimately, our analysis shows that good generalization is possible for SVM solutions beyond the realm in which typical margin-based bounds apply.

        ----

        ## [1850] Instance-Dependent Bounds for Zeroth-order Lipschitz Optimization with Error Certificates

        **Authors**: *François Bachoc, Tommaso Cesari, Sébastien Gerchinovitz*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/cacbf64b8a464fa1974da1eb0aa92851-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/cacbf64b8a464fa1974da1eb0aa92851-Abstract.html)

        **Abstract**:

        We study the problem of zeroth-order (black-box) optimization of a Lipschitz function $f$ defined on a compact subset $\mathcal{X}$ of $\mathbb{R}^d$, with the additional constraint that algorithms must certify the accuracy of their recommendations. We characterize the optimal number of evaluations of any Lipschitz function $f$ to find and certify an approximate maximizer of $f$ at accuracy $\varepsilon$. Under a weak assumption on $\mathcal{X}$, this optimal sample complexity is shown to be nearly proportional to the integral $\int_{\mathcal{X}} \mathrm{d}\boldsymbol{x}/( \max(f) - f(\boldsymbol{x}) + \varepsilon )^d$. This result, which was only (and partially) known in dimension $d=1$, solves an open problem dating back to 1991. In terms of techniques, our upper bound relies on a packing bound by Bouttier et al. (2020) for the Piyavskii-Shubert algorithm that we link to the above integral. We also show that a certified version of the computationally tractable DOO algorithm matches these packing and integral bounds. Our instance-dependent lower bound differs from traditional worst-case lower bounds in the Lipschitz setting and relies on a local worst-case analysis that could likely prove useful for other learning tasks.

        ----

        ## [1851] Training Neural Networks with Fixed Sparse Masks

        **Authors**: *Yi-Lin Sung, Varun Nair, Colin Raffel*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/cb2653f548f8709598e8b5156738cc51-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/cb2653f548f8709598e8b5156738cc51-Abstract.html)

        **Abstract**:

        During typical gradient-based training of deep neural networks, all of the model's parameters are updated at each iteration. Recent work has shown that it is possible to update only a small subset of the model's parameters during training, which can alleviate storage and communication requirements. In this paper, we show that it is possible to induce a fixed sparse mask on the modelâ€™s parameters that selects a subset to update over many iterations. Our method constructs the mask out of the $k$ parameters with the largest Fisher information as a simple approximation as to which parameters are most important for the task at hand. In experiments on parameter-efficient transfer learning and distributed training, we show that our approach matches or exceeds the performance of other methods for training with sparse updates while being more efficient in terms of memory usage and communication costs. We release our code publicly to promote further applications of our approach.

        ----

        ## [1852] VATT: Transformers for Multimodal Self-Supervised Learning from Raw Video, Audio and Text

        **Authors**: *Hassan Akbari, Liangzhe Yuan, Rui Qian, Wei-Hong Chuang, Shih-Fu Chang, Yin Cui, Boqing Gong*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/cb3213ada48302953cb0f166464ab356-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/cb3213ada48302953cb0f166464ab356-Abstract.html)

        **Abstract**:

        We present a framework for learning multimodal representations from unlabeled data using convolution-free Transformer architectures. Specifically, our Video-Audio-Text Transformer (VATT) takes raw signals as inputs and extracts multimodal representations that are rich enough to benefit a variety of downstream tasks. We train VATT end-to-end from scratch using multimodal contrastive losses and evaluate its performance by the downstream tasks of video action recognition, audio event classification, image classification, and text-to-video retrieval. Furthermore, we study a modality-agnostic single-backbone Transformer by sharing weights among the three modalities. We show that the convolution-free VATT outperforms state-of-the-art ConvNet-based architectures in the downstream tasks. Especially, VATT's vision Transformer achieves the top-1 accuracy of 82.1% on Kinetics-400, 83.6% on Kinetics-600, 72.7% on Kinetics-700, and 41.1% on Moments in Time, new records while avoiding supervised pre-training. Transferring to image classification leads to 78.7% top-1 accuracy on ImageNet compared to 64.7% by training the same Transformer from scratch, showing the generalizability of our model despite the domain gap between videos and images. VATT's audio Transformer also sets a new record on waveform-based audio event recognition by achieving the mAP of 39.4% on AudioSet without any supervised pre-training.

        ----

        ## [1853] Analyzing the Generalization Capability of SGLD Using Properties of Gaussian Channels

        **Authors**: *Hao Wang, Yizhe Huang, Rui Gao, Flávio P. Calmon*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/cb77649f5d53798edfa0ff40dae46322-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/cb77649f5d53798edfa0ff40dae46322-Abstract.html)

        **Abstract**:

        Optimization is a key component for training machine learning models and has a strong impact on their generalization. In this paper, we consider a particular optimization method---the stochastic gradient Langevin dynamics (SGLD) algorithm---and investigate the generalization of models trained by SGLD. We derive a new generalization bound by connecting SGLD with Gaussian channels found in information and communication theory. Our bound can be computed from the training data and incorporates the variance of gradients for quantifying a particular kind of "sharpness" of the loss landscape. We also consider a closely related algorithm with SGLD, namely differentially private SGD (DP-SGD). We prove that the generalization capability of DP-SGD can be amplified by iteration. Specifically, our bound can be sharpened by including a time-decaying factor if the DP-SGD algorithm outputs the last iterate while keeping other iterates hidden. This decay factor enables the contribution of early iterations to our bound to reduce with time and is established by strong data processing inequalities---a fundamental tool in information theory. We demonstrate our bound through numerical experiments, showing that it can predict the behavior of the true generalization gap.

        ----

        ## [1854] Learning to Schedule Heuristics in Branch and Bound

        **Authors**: *Antonia Chmiela, Elias B. Khalil, Ambros M. Gleixner, Andrea Lodi, Sebastian Pokutta*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/cb7c403aa312160380010ee3dd4bfc53-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/cb7c403aa312160380010ee3dd4bfc53-Abstract.html)

        **Abstract**:

        Primal heuristics play a crucial role in exact solvers for Mixed Integer Programming (MIP). While solvers are guaranteed to find optimal solutions given sufficient time, real-world applications typically require finding good solutions early on in the search to enable fast decision-making. While much of MIP research focuses on designing effective heuristics, the question of how to manage multiple MIP heuristics in a solver has not received equal attention. Generally, solvers follow hard-coded rules derived from empirical testing on broad sets of instances. Since the performance of heuristics is problem-dependent, using these general rules for a particular problem might not yield the best performance. In this work, we propose the first data-driven framework for scheduling heuristics in an exact MIP solver. By learning from data describing the performance of primal heuristics, we obtain a problem-specific schedule of heuristics that collectively find many solutions at minimal cost. We formalize the learning task and propose an efficient algorithm for computing such a schedule. Compared to the default settings of a state-of-the-art academic MIP solver, we are able to reduce the average primal integral by up to 49% on two classes of challenging instances.

        ----

        ## [1855] On Training Implicit Models

        **Authors**: *Zhengyang Geng, Xin-Yu Zhang, Shaojie Bai, Yisen Wang, Zhouchen Lin*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/cb8da6767461f2812ae4290eac7cbc42-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/cb8da6767461f2812ae4290eac7cbc42-Abstract.html)

        **Abstract**:

        This paper focuses on training implicit models of infinite layers. Specifically, previous works employ implicit differentiation and solve the exact gradient for the backward propagation. However, is it necessary to compute such an exact but expensive gradient for training? In this work, we propose a novel gradient estimate for implicit models, named phantom gradient, that 1) forgoes the costly computation of the exact gradient; and 2) provides an update direction empirically preferable to the implicit model training. We theoretically analyze the condition under which an ascent direction of the loss landscape could be found and provide two specific instantiations of the phantom gradient based on the damped unrolling and Neumann series. Experiments on large-scale tasks demonstrate that these lightweight phantom gradients significantly accelerate the backward passes in training implicit models by roughly 1.7 $\times$ and even boost the performance over approaches based on the exact gradient on ImageNet.

        ----

        ## [1856] MLP-Mixer: An all-MLP Architecture for Vision

        **Authors**: *Ilya O. Tolstikhin, Neil Houlsby, Alexander Kolesnikov, Lucas Beyer, Xiaohua Zhai, Thomas Unterthiner, Jessica Yung, Andreas Steiner, Daniel Keysers, Jakob Uszkoreit, Mario Lucic, Alexey Dosovitskiy*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/cba0a4ee5ccd02fda0fe3f9a3e7b89fe-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/cba0a4ee5ccd02fda0fe3f9a3e7b89fe-Abstract.html)

        **Abstract**:

        Convolutional Neural Networks (CNNs) are the go-to model for computer vision. Recently, attention-based networks, such as the Vision Transformer, have also become popular. In this paper we show that while convolutions and attention are both sufficient for good performance, neither of them are necessary. We present MLP-Mixer, an architecture based exclusively on multi-layer perceptrons (MLPs). MLP-Mixer contains two types of layers: one with MLPs applied independently to image patches (i.e. "mixing" the per-location features), and one with MLPs applied across patches (i.e. "mixing" spatial information). When trained on large datasets, or with modern regularization schemes, MLP-Mixer attains competitive scores on image classification benchmarks, with pre-training and inference cost comparable to state-of-the-art models. We hope that these results spark further research beyond the realms of well established CNNs and Transformers.

        ----

        ## [1857] A Framework to Learn with Interpretation

        **Authors**: *Jayneel Parekh, Pavlo Mozharovskyi, Florence d'Alché-Buc*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/cbb6a3b884f4f88b3a8e3d44c636cbd8-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/cbb6a3b884f4f88b3a8e3d44c636cbd8-Abstract.html)

        **Abstract**:

        To tackle interpretability in deep learning, we present a novel framework to jointly learn a predictive model and its associated interpretation model. The interpreter provides both local and global interpretability about the predictive model in terms of human-understandable high level attribute functions, with minimal loss of accuracy. This is achieved by a dedicated architecture and well chosen regularization penalties. We seek for a small-size dictionary of high level attribute functions that take as inputs the outputs of selected hidden layers and whose outputs feed a linear classifier. We impose strong conciseness on the activation of attributes with an entropy-based criterion while enforcing fidelity to both inputs and outputs of the predictive model. A detailed pipeline to visualize the learnt features is also developed. Moreover, besides generating interpretable models by design, our approach can be specialized to provide post-hoc interpretations for a pre-trained neural network. We validate our approach against several state-of-the-art methods on multiple datasets and show its efficacy on both kinds of tasks.

        ----

        ## [1858] One Loss for All: Deep Hashing with a Single Cosine Similarity based Learning Objective

        **Authors**: *Jiun Tian Hoe, Kam Woh Ng, Tianyu Zhang, Chee Seng Chan, Yi-Zhe Song, Tao Xiang*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/cbcb58ac2e496207586df2854b17995f-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/cbcb58ac2e496207586df2854b17995f-Abstract.html)

        **Abstract**:

        A deep hashing model typically has two main learning objectives: to make the learned binary hash codes discriminative and to minimize a quantization error. With further constraints such as bit balance and code orthogonality, it is not uncommon for existing models to employ a large number (>4) of losses. This leads to difficulties in model training and subsequently impedes their effectiveness. In this work, we propose a novel deep hashing model with only $\textit{a single learning objective}$. Specifically,  we show that maximizing the cosine similarity between the continuous codes and their corresponding $\textit{binary orthogonal codes}$ can ensure both hash code discriminativeness and quantization error minimization. Further, with this learning objective, code balancing can be achieved by simply using a  Batch Normalization (BN) layer and multi-label classification is also straightforward with label smoothing. The result is a one-loss deep hashing model that removes all the hassles of tuning the weights of various losses. Importantly,  extensive experiments show that our model is highly effective, outperforming the state-of-the-art multi-loss hashing models on three large-scale instance retrieval benchmarks, often by significant margins.

        ----

        ## [1859] Fast and accurate randomized algorithms for low-rank tensor decompositions

        **Authors**: *Linjian Ma, Edgar Solomonik*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/cbef46321026d8404bc3216d4774c8a9-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/cbef46321026d8404bc3216d4774c8a9-Abstract.html)

        **Abstract**:

        Low-rank Tucker and CP tensor decompositions are powerful tools in data analytics. The widely used alternating least squares (ALS) method, which solves a sequence of over-determined least squares subproblems, is costly for large and sparse tensors. We propose a fast and accurate sketched ALS algorithm for Tucker decomposition, which solves a sequence of sketched rank-constrained linear least squares subproblems. Theoretical sketch size upper bounds are provided to achieve $O(\epsilon)$ relative error for each subproblem with two sketching techniques, TensorSketch and leverage score sampling. Experimental results show that this new ALS algorithm, combined with a new initialization scheme based on the randomized range finder, yields decomposition accuracy comparable to the standard higher-order orthogonal iteration (HOOI) algorithm. The new algorithm achieves up to $22.0\%$ relative decomposition residual improvement compared to the state-of-the-art sketched randomized algorithm for Tucker decomposition of various synthetic and real datasets. This Tucker-ALS algorithm is further used to accelerate CP decomposition, by using randomized Tucker compression followed by CP decomposition of the Tucker core tensor. Experimental results show that this algorithm not only converges faster, but also yields more accurate CP decompositions.

        ----

        ## [1860] Communication-efficient SGD: From Local SGD to One-Shot Averaging

        **Authors**: *Artin Spiridonoff, Alex Olshevsky, Yannis Paschalidis*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/cc06a6150b92e17dd3076a0f0f9d2af4-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/cc06a6150b92e17dd3076a0f0f9d2af4-Abstract.html)

        **Abstract**:

        We consider speeding up stochastic gradient descent (SGD) by parallelizing it across multiple workers. We assume the same data set is shared among $N$ workers, who can take SGD steps and coordinate with a central server. While it is possible to obtain a linear reduction in the variance by averaging all the stochastic gradients at every step, this requires a lot of communication between the workers and the server, which can dramatically reduce the gains from parallelism.The Local SGD method, proposed and analyzed in the earlier literature, suggests machines should make many local steps between such communications. While the initial analysis of Local SGD showed it needs $\Omega ( \sqrt{T} )$ communications for $T$ local gradient steps in order for the error to scale proportionately to $1/(NT)$, this has been successively improved in a string of papers, with the state of the art requiring  $\Omega \left( N \left( \mbox{ poly} (\log T) \right) \right)$ communications. In this paper, we suggest a Local SGD scheme that communicates less overall by communicating less frequently as the number of iterations grows.  Our analysis shows that this can achieve an error that scales as $1/(NT)$ with a number of communications that is completely independent of $T$. In particular, we show that $\Omega(N)$ communications are sufficient. Empirical evidence suggests this bound is close to tight as we further show that $\sqrt{N}$ or $N^{3/4}$ communications fail to achieve linear speed-up in simulations. Moreover, we show that under mild assumptions, the main of which is twice differentiability on any neighborhood of the optimal solution, one-shot averaging which only uses a single round of communication can also achieve the optimal convergence rate asymptotically.

        ----

        ## [1861] Memory Efficient Meta-Learning with Large Images

        **Authors**: *John Bronskill, Daniela Massiceti, Massimiliano Patacchiola, Katja Hofmann, Sebastian Nowozin, Richard E. Turner*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/cc1aa436277138f61cda703991069eaf-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/cc1aa436277138f61cda703991069eaf-Abstract.html)

        **Abstract**:

        Meta learning approaches to few-shot classification are computationally efficient at test time, requiring just a few optimization steps or single forward pass to learn a new task, but they remain highly memory-intensive to train. This limitation arises because a task's entire support set, which can contain up to 1000 images, must be processed before an optimization step can be taken. Harnessing the performance gains offered by large images thus requires either parallelizing the meta-learner across multiple GPUs, which may not be available, or trade-offs between task and image size when memory constraints apply. We improve on both options by proposing LITE, a general and memory efficient episodic training scheme that enables meta-training on large tasks composed of large images on a single GPU. We achieve this by observing that the gradients for a task can be decomposed into a sum of gradients over the task's training images. This enables us to perform a forward pass on a task's entire training set but realize significant memory savings by back-propagating only a random subset of these images which we show is an unbiased approximation of the full gradient. We use LITE to train meta-learners and demonstrate new state-of-the-art accuracy on the real-world ORBIT benchmark and 3 of the 4 parts of the challenging VTAB+MD benchmark relative to leading meta-learners. LITE also enables meta-learners to be competitive with transfer learning approaches but at a fraction of the test-time computational cost, thus serving as a counterpoint to the recent narrative that transfer learning is all you need for few-shot classification.

        ----

        ## [1862] On the Power of Differentiable Learning versus PAC and SQ Learning

        **Authors**: *Emmanuel Abbe, Pritish Kamath, Eran Malach, Colin Sandon, Nathan Srebro*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/cc225865b743ecc91c4743259813f604-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/cc225865b743ecc91c4743259813f604-Abstract.html)

        **Abstract**:

        We study the power of learning via mini-batch stochastic gradient descent (SGD) on the loss of a differentiable model or neural network, and ask what learning problems can be learnt using this paradigm. We show that SGD can always simulate learning with statistical queries (SQ), but its ability to go beyond that depends on the precision $\rho$ of the gradients and the minibatch size $b$. With fine enough precision relative to minibatch size, namely when $b \rho$ is small enough, SGD can go beyond SQ learning and simulate any sample-based learning algorithm and thus its learning power is equivalent to that of PAC learning; this extends prior work that achieved this result for $b=1$. Moreover, with polynomially many bits of precision (i.e. when $\rho$ is exponentially small), SGD can simulate PAC learning regardless of the batch size. On the other hand, when $b \rho^2$ is large enough, the power of SGD is equivalent to that of SQ learning.

        ----

        ## [1863] Can we globally optimize cross-validation loss? Quasiconvexity in ridge regression

        **Authors**: *William T. Stephenson, Zachary Frangella, Madeleine Udell, Tamara Broderick*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/cc298d5bc587e1b650f80e10449ee9d5-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/cc298d5bc587e1b650f80e10449ee9d5-Abstract.html)

        **Abstract**:

        Models like LASSO and ridge regression are extensively used in practice due to their interpretability, ease of use, and strong theoretical guarantees. Cross-validation (CV) is widely used for hyperparameter tuning in these models, but do practical methods minimize the true out-of-sample loss? A recent line of research promises to show that the optimum of the CV loss matches the optimum of the out-of-sample loss (possibly after simple corrections). It remains to show how tractable it is to minimize the CV loss.In the present paper, we show that, in the case of ridge regression, the CV loss may fail to be quasiconvex and thus may have multiple local optima. We can guarantee that the CV loss is quasiconvex in at least one case: when the spectrum of the covariate matrix is nearly flat and the noise in the observed responses is not too high. More generally, we show that quasiconvexity status is independent of many properties of the observed data (response norm, covariate-matrix right singular vectors and singular-value scaling) and has a complex dependence on the few that remain. We empirically confirm our theory using simulated experiments.

        ----

        ## [1864] Adaptive Proximal Gradient Methods for Structured Neural Networks

        **Authors**: *Jihun Yun, Aurélie C. Lozano, Eunho Yang*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/cc3f5463bc4d26bc38eadc8bcffbc654-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/cc3f5463bc4d26bc38eadc8bcffbc654-Abstract.html)

        **Abstract**:

        We consider the training of structured neural networks where the regularizer can be non-smooth and possibly non-convex. While popular machine learning libraries have resorted to stochastic (adaptive) subgradient approaches, the use of proximal gradient methods in the stochastic setting has been little explored and warrants further study, in particular regarding the incorporation of adaptivity. Towards this goal, we present a general framework of stochastic proximal gradient descent methods that allows for arbitrary positive preconditioners and lower semi-continuous regularizers. We derive two important instances of our framework: (i) the first proximal version of \textsc{Adam}, one of the most popular adaptive SGD algorithm, and (ii) a revised version of ProxQuant for quantization-specific regularizers, which improves upon the original approach by incorporating the effect of preconditioners in the proximal mapping computations. We provide convergence guarantees for our framework and show that adaptive gradient methods can have faster convergence in terms of constant than vanilla SGD for sparse data. Lastly, we demonstrate the superiority of stochastic proximal methods compared to subgradient-based approaches via extensive experiments. Interestingly, our results indicate that the benefit of proximal approaches over sub-gradient counterparts is more pronounced for non-convex regularizers than for convex ones.

        ----

        ## [1865] Discovering and Achieving Goals via World Models

        **Authors**: *Russell Mendonca, Oleh Rybkin, Kostas Daniilidis, Danijar Hafner, Deepak Pathak*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/cc4af25fa9d2d5c953496579b75f6f6c-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/cc4af25fa9d2d5c953496579b75f6f6c-Abstract.html)

        **Abstract**:

        How can artificial agents learn to solve many diverse tasks in complex visual environments without any supervision? We decompose this question into two challenges: discovering new goals and learning to reliably achieve them. Our proposed agent, Latent Explorer Achiever (LEXA), addresses both challenges by learning a world model from image inputs and using it to train an explorer and an achiever policy via imagined rollouts. Unlike prior methods that explore by reaching previously visited states, the explorer plans to discover unseen surprising states through foresight, which are then used as diverse targets for the achiever to practice. After the unsupervised phase, LEXA solves tasks specified as goal images zero-shot without any additional learning. LEXA substantially outperforms previous approaches to unsupervised goal reaching, both on prior benchmarks and on a new challenging benchmark with 40 test tasks spanning across four robotic manipulation and locomotion domains. LEXA further achieves goals that require interacting with multiple objects in sequence. Project page: https://orybkin.github.io/lexa/

        ----

        ## [1866] Understanding and Improving Early Stopping for Learning with Noisy Labels

        **Authors**: *Yingbin Bai, Erkun Yang, Bo Han, Yanhua Yang, Jiatong Li, Yinian Mao, Gang Niu, Tongliang Liu*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/cc7e2b878868cbae992d1fb743995d8f-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/cc7e2b878868cbae992d1fb743995d8f-Abstract.html)

        **Abstract**:

        The memorization effect of deep neural network (DNN) plays a pivotal role in many state-of-the-art label-noise learning methods.  To exploit this property, the early stopping trick, which stops the optimization at the early stage of training, is usually adopted. Current methods generally decide the early stopping point by considering a DNN as a whole. However, a DNN can be considered as a composition of a series of layers, and we find that the latter layers in a DNN are much more sensitive to label noise, while their former counterparts are quite robust. Therefore, selecting a stopping point for the whole network may make different DNN layers antagonistically affect each other, thus degrading the final performance. In this paper, we propose to separate a DNN into different parts and progressively train them to address this problem. Instead of the early stopping which trains a whole DNN all at once, we initially train former DNN layers by optimizing the DNN with a relatively large number of epochs. During training, we progressively train the latter DNN layers by using a smaller number of epochs with the preceding layers fixed to counteract the impact of noisy labels. We term the proposed method as progressive early stopping (PES). Despite its simplicity, compared with the traditional early stopping, PES can help to obtain more promising and stable results. Furthermore, by combining PES with existing approaches on noisy label training, we achieve state-of-the-art performance on image classification benchmarks. The code is made public at https://github.com/tmllab/PES.

        ----

        ## [1867] Distributionally Robust Imitation Learning

        **Authors**: *Mohammad Ali Bashiri, Brian D. Ziebart, Xinhua Zhang*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/cc8090c4d2791cdd9cd2cb3c24296190-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/cc8090c4d2791cdd9cd2cb3c24296190-Abstract.html)

        **Abstract**:

        We consider the imitation learning problem of learning a policy in a Markov Decision Process (MDP) setting where the reward function is not given, but demonstrations from experts are available. Although the goal of imitation learning is to learn a policy that produces behaviors nearly as good as the experts’ for a desired task, assumptions of consistent optimality for demonstrated behaviors are often violated in practice. Finding a policy that is distributionally robust against noisy demonstrations based on an adversarial construction potentially solves this problem by avoiding optimistic generalizations of the demonstrated data. This paper studies Distributionally Robust Imitation Learning (DRoIL) and establishes a close connection between DRoIL and Maximum Entropy Inverse Reinforcement Learning. We show that DRoIL can be seen as a framework that maximizes a generalized concept of entropy. We develop a novel approach to transform the objective function into a convex optimization problem over a polynomial number of variables for a class of loss functions that are additive over state and action spaces. Our approach lets us optimize both stationary and non-stationary policies and, unlike prevalent previous methods, it does not require repeatedly solving an inner reinforcement learning problem. We experimentally show the significant benefits of DRoIL’s new optimization method on synthetic data and a highway driving environment.

        ----

        ## [1868] On the Power of Edge Independent Graph Models

        **Authors**: *Sudhanshu Chanpuriya, Cameron Musco, Konstantinos Sotiropoulos, Charalampos E. Tsourakakis*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/cc9b3c69b56df284846bf2432f1cba90-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/cc9b3c69b56df284846bf2432f1cba90-Abstract.html)

        **Abstract**:

        Why do many modern neural-network-based graph generative models fail to reproduce typical real-world network characteristics, such as high triangle density?  In this work we study the limitations  of $edge\ independent\ random\ graph\ models$, in which  each edge is added to the graph independently with some probability. Such models include both the classic Erdos-Renyi and stochastic block models, as well as  modern generative models such as NetGAN, variational graph autoencoders, and CELL.  We prove that subject to a $bounded\  overlap$ condition, which ensures that the model does not simply memorize a single graph, edge independent models are inherently limited in their ability to generate graphs with high triangle and other subgraph densities. Notably, such high densities are known to appear in real-world social networks and other graphs. We complement our negative results with a simple generative model that balances overlap and accuracy, performing comparably to more complex models in reconstructing many graph statistics.

        ----

        ## [1869] Stochastic Online Linear Regression: the Forward Algorithm to Replace Ridge

        **Authors**: *Reda Ouhamma, Odalric-Ambrym Maillard, Vianney Perchet*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/cca289d2a4acd14c1cd9a84ffb41dd29-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/cca289d2a4acd14c1cd9a84ffb41dd29-Abstract.html)

        **Abstract**:

        We consider the problem of online linear regression in the stochastic setting. We derive high probability regret bounds for online $\textit{ridge}$ regression and the $\textit{forward}$ algorithm. This enables us to compare online regression algorithms more accurately and eliminate assumptions of bounded observations and predictions. Our study advocates for the use of the forward algorithm in lieu of ridge due to its enhanced bounds and robustness to the regularization parameter. Moreover, we explain how to integrate it in algorithms involving linear function approximation to remove a boundedness assumption without deteriorating theoretical bounds. We showcase this modification in linear bandit settings where it yields improved regret bounds. Last, we provide numerical experiments to illustrate our results and endorse our intuitions.

        ----

        ## [1870] Dr Jekyll & Mr Hyde: the strange case of off-policy policy updates

        **Authors**: *Romain Laroche, Remi Tachet des Combes*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/ccb421d5f36c5a412816d494b15ca9f6-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/ccb421d5f36c5a412816d494b15ca9f6-Abstract.html)

        **Abstract**:

        The policy gradient theorem states that the policy should only be updated in states that are visited by the current policy, which leads to insufficient planning in the off-policy states, and thus to convergence to suboptimal policies. We tackle this planning issue by extending the policy gradient theory to policy updates with respect to any state density. Under these generalized policy updates, we show convergence to optimality under a necessary and sufficient condition on the updates’ state densities, and thereby solve the aforementioned planning issue. We also prove asymptotic convergence rates that significantly improve those in the policy gradient literature. To implement the principles prescribed by our theory, we propose an agent, Dr Jekyll & Mr Hyde (J&H), with a double personality: Dr Jekyll purely exploits while Mr Hyde purely explores. J&H’s independent policies allow to record two separate replay buffers: one on-policy (Dr Jekyll’s) and one off-policy (Mr Hyde’s), and therefore to update J&H’s models with a mixture of on-policy and off-policy updates. More than an algorithm, J&H defines principles for actor-critic algorithms to satisfy the requirements we identify in our analysis. We extensively test on finite MDPs where J&H demonstrates a superior ability to recover from converging to a suboptimal policy without impairing its speed of convergence. We also implement a deep version of the algorithm and test it on a simple problem where it shows promising results.

        ----

        ## [1871] Understanding Adaptive, Multiscale Temporal Integration In Deep Speech Recognition Systems

        **Authors**: *Menoua Keshishian, Samuel Norman-Haignere, Nima Mesgarani*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/ccce2fab7336b8bc8362d115dec2d5a2-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/ccce2fab7336b8bc8362d115dec2d5a2-Abstract.html)

        **Abstract**:

        Natural signals such as speech are hierarchically structured across many different timescales, spanning tens (e.g., phonemes) to hundreds (e.g., words) of milliseconds, each of which is highly variable and context-dependent. While deep neural networks (DNNs) excel at recognizing complex patterns from natural signals, relatively little is known about how DNNs flexibly integrate across multiple timescales. Here, we show how a recently developed method for studying temporal integration in biological neural systems – the temporal context invariance (TCI) paradigm – can be used to understand temporal integration in DNNs. The method is simple: we measure responses to a large number of stimulus segments presented in two different contexts and estimate the smallest segment duration needed to achieve a context invariant response. We applied our method to understand how the popular DeepSpeech2 model learns to integrate across time in speech. We find that nearly all of the model units, even in recurrent layers, have a compact integration window within which stimuli substantially alter the response and outside of which stimuli have little effect. We show that training causes these integration windows to shrink at early layers and expand at higher layers, creating a hierarchy of integration windows across the network. Moreover, by measuring integration windows for time-stretched/compressed speech, we reveal a transition point, midway through the trained network, where integration windows become yoked to the duration of stimulus structures (e.g., phonemes or words) rather than absolute time. Similar phenomena were observed in a purely recurrent and purely convolutional network although structure-yoked integration was more prominent in the recurrent network. These findings suggest that deep speech recognition systems use a common motif to encode the hierarchical structure of speech: integrating across short, time-yoked windows at early layers and long, structure-yoked windows at later layers. Our method provides a straightforward and general-purpose toolkit for understanding temporal integration in black-box machine learning models.

        ----

        ## [1872] VidLanKD: Improving Language Understanding via Video-Distilled Knowledge Transfer

        **Authors**: *Zineng Tang, Jaemin Cho, Hao Tan, Mohit Bansal*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/ccdf3864e2fa9089f9eca4fc7a48ea0a-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/ccdf3864e2fa9089f9eca4fc7a48ea0a-Abstract.html)

        **Abstract**:

        Since visual perception can give rich information beyond text descriptions for world understanding, there has been increasing interest in leveraging visual grounding for language learning. Recently, vokenization (Tan and Bansal, 2020) has attracted attention by using the predictions of a text-to-image retrieval model as labels for language model supervision. Despite its success, the method suffers from approximation error of using finite image labels and the lack of vocabulary diversity of a small image-text dataset. To overcome these limitations, we present VidLanKD, a video-language knowledge distillation method for improving language understanding. We train a multi-modal teacher model on a video-text dataset, and then transfer its knowledge to a student language model with a text dataset. To avoid approximation error, we propose to use different knowledge distillation objectives. In addition, the use of a large-scale video-text dataset helps learn diverse and richer vocabularies. In our experiments, VidLanKD achieves consistent improvements over text-only language models and vokenization models, on several downstream language understanding tasks including GLUE, SQuAD, and SWAG. We also demonstrate the improved world knowledge, physical reasoning, and temporal reasoning capabilities of our model by evaluating on the GLUE-diagnostics, PIQA, and TRACIE datasets. Lastly, we present comprehensive ablation studies as well as visualizations of the learned text-to-video grounding results of our teacher and student language models.

        ----

        ## [1873] Detecting Individual Decision-Making Style: Exploring Behavioral Stylometry in Chess

        **Authors**: *Reid McIlroy-Young, Yu Wang, Siddhartha Sen, Jon M. Kleinberg, Ashton Anderson*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/ccf8111910291ba472b385e9c5f59099-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/ccf8111910291ba472b385e9c5f59099-Abstract.html)

        **Abstract**:

        The advent of machine learning models that surpass human decision-making ability in complex domains has initiated a movement towards building AI systems that interact with humans. Many building blocks are essential for this activity, with a central one being the algorithmic characterization of human behavior. While much of the existing work focuses on aggregate human behavior, an important long-range goal is to develop behavioral models that specialize to individual people and can differentiate among them.To formalize this process, we study the problem of behavioral stylometry, in which the task is to identify a decision-maker from their decisions alone. We present a transformer-based approach to behavioral stylometry in the context of chess, where one attempts to identify the player who played a set of games. Our method operates in a few-shot classification framework, and can correctly identify a player from among thousands of candidate players with 98% accuracy given only 100 labeled games. Even when trained on amateur play, our method generalises to out-of-distribution samples of Grandmaster players, despite the dramatic differences between amateur and world-class players. Finally, we consider more broadly what our resulting embeddings reveal about human style in chess, as well as the potential ethical implications of powerful methods for identifying individuals from behavioral data.

        ----

        ## [1874] Coupled Gradient Estimators for Discrete Latent Variables

        **Authors**: *Zhe Dong, Andriy Mnih, George Tucker*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/cd0b43eac0392accf3624b7372dec36e-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/cd0b43eac0392accf3624b7372dec36e-Abstract.html)

        **Abstract**:

        Training models with discrete latent variables is challenging due to the high variance of unbiased gradient estimators. While low-variance reparameterization gradients of a continuous relaxation can provide an effective solution, a continuous relaxation is not always available or tractable. Dong et al. (2020) and Yin et al. (2020) introduced a performant estimator that does not rely on continuous relaxations; however, it is limited to binary random variables. We introduce a novel derivation of their estimator based on importance sampling and statistical couplings, which we extend to the categorical setting. Motivated by the construction of a stick-breaking coupling, we introduce gradient estimators based on reparameterizing categorical variables as sequences of binary variables and Rao-Blackwellization. In systematic experiments, we show that our proposed categorical gradient estimators provide state-of-the-art performance, whereas even with additional Rao-Blackwellization previous estimators (Yin et al., 2019) underperform a simpler REINFORCE with a leave-one-out-baseline estimator (Kool et al., 2019).

        ----

        ## [1875] AutoGEL: An Automated Graph Neural Network with Explicit Link Information

        **Authors**: *Zhili Wang, Shimin Di, Lei Chen*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/cd3afef9b8b89558cd56638c3631868a-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/cd3afef9b8b89558cd56638c3631868a-Abstract.html)

        **Abstract**:

        Recently, Graph Neural Networks (GNNs) have gained popularity in a variety of real-world scenarios. Despite the great success, the architecture design of GNNs heavily relies on manual labor. Thus, automated graph neural network (AutoGNN) has attracted interest and attention from the research community, which makes significant performance improvements in recent years. However, existing AutoGNN works mainly adopt an implicit way to model and leverage the link information in the graphs, which is not well regularized to the link prediction task on graphs, and limits the performance of AutoGNN for other graph tasks. In this paper, we present a novel AutoGNN work that explicitly models the link information, abbreviated to AutoGEL. In such a way, AutoGEL can handle the link prediction task and improve the performance of AutoGNNs on the node classification and graph classification task. Moreover, AutoGEL proposes a novel search space containing various design dimensions at both intra-layer and inter-layer designs and adopts a more robust differentiable search algorithm to further improve efficiency and effectiveness. Experimental results on benchmark data sets demonstrate the superiority of AutoGEL on several tasks.

        ----

        ## [1876] RL for Latent MDPs: Regret Guarantees and a Lower Bound

        **Authors**: *Jeongyeol Kwon, Yonathan Efroni, Constantine Caramanis, Shie Mannor*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/cd755a6c6b699f3262bcc2aa46ab507e-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/cd755a6c6b699f3262bcc2aa46ab507e-Abstract.html)

        **Abstract**:

        In this work, we consider the regret minimization problem for reinforcement learning in latent Markov Decision Processes (LMDP). In an LMDP, an MDP is randomly drawn from a set of $M$ possible MDPs at the beginning of the interaction, but the identity of the chosen MDP is not revealed to the agent. We first show that a general instance of LMDPs requires at least $\Omega((SA)^M)$ episodes to even approximate the optimal policy. Then, we consider sufficient assumptions under which learning good policies requires polynomial number of episodes. We show that the key link is a notion of separation between the MDP system dynamics. With sufficient separation, we provide an efficient algorithm with local guarantee, {\it i.e.,} providing a sublinear regret guarantee when we are given a good initialization. Finally, if we are given standard statistical sufficiency assumptions common in the Predictive State Representation (PSR) literature (e.g., \cite{boots2011online}) and a reachability assumption, we show that the need for initialization can be removed.

        ----

        ## [1877] Adaptive Sampling for Minimax Fair Classification

        **Authors**: *Shubhanshu Shekhar, Greg Fields, Mohammad Ghavamzadeh, Tara Javidi*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/cd7c230fc5deb01ff5f7b1be1acef9cf-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/cd7c230fc5deb01ff5f7b1be1acef9cf-Abstract.html)

        **Abstract**:

        Machine learning models trained on uncurated datasets can often end up adversely affecting inputs belonging to underrepresented groups. To address this issue, we consider the problem of adaptively constructing training sets which allow us to learn classifiers that are fair in a {\em minimax} sense. We first propose an adaptive sampling algorithm based on the principle of \emph{optimism}, and derive theoretical bounds on its performance. We also propose heuristic extensions of this algorithm suitable for application to large scale, practical problems. Next, by deriving algorithm independent lower-bounds for a specific class of problems, we show that the performance achieved by our adaptive scheme cannot be improved in general. We then validate the benefits of adaptively constructing training sets via experiments on synthetic tasks with logistic regression classifiers, as well as on several real-world tasks using convolutional neural networks (CNNs).

        ----

        ## [1878] Structured in Space, Randomized in Time: Leveraging Dropout in RNNs for Efficient Training

        **Authors**: *Anup Sarma, Sonali Singh, Huaipan Jiang, Rui Zhang, Mahmut T. Kandemir, Chita R. Das*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/cd81cfd0a3397761fac44ddbe5ec3349-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/cd81cfd0a3397761fac44ddbe5ec3349-Abstract.html)

        **Abstract**:

        Recurrent Neural Networks (RNNs), more specifically their Long Short-Term Memory (LSTM) variants, have been widely used as a deep learning tool for tackling sequence-based learning tasks in text and speech. Training of such LSTM applications is computationally intensive due to the recurrent nature of hidden state computation that repeats for each time step. While sparsity in Deep Neural Nets has been widely seen as an opportunity for reducing computation time in both training and inference phases, the usage of non-ReLU activation in LSTM RNNs renders the opportunities for such dynamic sparsity associated with neuron activation and gradient values to be limited or non-existent. In this work, we identify dropout induced sparsity for LSTMs as a suitable mode of computation reduction. Dropout is a widely used regularization mechanism, which randomly drops computed neuron values during each iteration of training. We propose to structure dropout patterns, by dropping out the same set of physical neurons within a batch, resulting in column (row) level hidden state sparsity, which are well amenable to computation reduction at run-time in general-purpose SIMD hardware as well as systolic arrays. We provide a detailed analysis of how the dropout-induced sparsity propagates through the different stages of network training and how it can be leveraged in each stage. More importantly, our proposed approach works as a direct replacement for existing dropout-based application settings. We conduct our experiments for three representative NLP tasks: language modelling on the PTB dataset, OpenNMT based machine translation using the IWSLT De-En and En-Vi datasets, and named entity recognition sequence labelling using the CoNLL-2003 shared task. We demonstrate that our proposed approach can be used to translate dropout-based computation reduction into reduced training time, with improvement ranging from 1.23$\times$ to 1.64$\times$, without sacrificing the target metric.

        ----

        ## [1879] Variational Continual Bayesian Meta-Learning

        **Authors**: *Qiang Zhang, Jinyuan Fang, Zaiqiao Meng, Shangsong Liang, Emine Yilmaz*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/cdd0500dc0ef6682fa6ec6d2e6b577c4-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/cdd0500dc0ef6682fa6ec6d2e6b577c4-Abstract.html)

        **Abstract**:

        Conventional meta-learning considers a set of tasks from a stationary distribution. In contrast, this paper focuses on a more complex online setting, where tasks arrive sequentially and follow a non-stationary distribution. Accordingly, we propose a Variational Continual Bayesian Meta-Learning (VC-BML) algorithm. VC-BML maintains a Dynamic Gaussian Mixture Model for meta-parameters, with the number of component distributions determined by a Chinese Restaurant Process. Dynamic mixtures at the meta-parameter level increase the capability to adapt to diverse tasks due to a larger parameter space, alleviating the negative knowledge transfer problem. To infer posteriors of model parameters, compared to the previously used point estimation method, we develop a more robust posterior approximation method -- structured variational inference for the sake of avoiding forgetting knowledge. Experiments on tasks from non-stationary distributions show that VC-BML is superior in transferring knowledge among diverse tasks and alleviating catastrophic forgetting in an online setting.

        ----

        ## [1880] Recognizing Vector Graphics without Rasterization

        **Authors**: *Xinyang Jiang, Lu Liu, Caihua Shan, Yifei Shen, Xuanyi Dong, Dongsheng Li*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/cdf1035c34ec380218a8cc9a43d438f9-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/cdf1035c34ec380218a8cc9a43d438f9-Abstract.html)

        **Abstract**:

        In this paper, we consider a different data format for images: vector graphics. In contrast to raster graphics which are widely used in image recognition, vector graphics can be scaled up or down into any resolution without aliasing or information loss, due to the analytic representation of the primitives in the document. Furthermore, vector graphics are able to give extra structural information on how low-level elements group together to form high level shapes or structures. These merits of graphic vectors have not been fully leveraged in existing methods.  To explore this data format, we target on the fundamental recognition tasks: object localization and classification. We propose an efficient CNN-free pipeline that does not render the graphic into pixels (i.e. rasterization), and takes textual document of the vector graphics as input, called YOLaT (You Only Look at Text). YOLaT builds multi-graphs to model the structural and spatial information in vector graphics, and a dual-stream graph neural network is proposed to detect objects from the graph. Our experiments show that by directly operating on vector graphics, YOLaT outperforms raster-graphic based object detection baselines in terms of both average precision and efficiency. Code is available at https://github.com/microsoft/YOLaT-VectorGraphicsRecognition.

        ----

        ## [1881] On Episodes, Prototypical Networks, and Few-Shot Learning

        **Authors**: *Steinar Laenen, Luca Bertinetto*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/cdfa4c42f465a5a66871587c69fcfa34-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/cdfa4c42f465a5a66871587c69fcfa34-Abstract.html)

        **Abstract**:

        Episodic learning is a popular practice among researchers and practitioners interested in few-shot learning.It consists of organising training in a series of learning problems (or episodes), each divided into a small training and validation subset to mimic the circumstances encountered during evaluation.But is this always necessary?In this paper, we investigate the usefulness of episodic learning in methods which use nonparametric approaches, such as nearest neighbours, at the level of the episode.For these methods, we not only show how the constraints imposed by episodic learning are not necessary, but that they in fact lead to a data-inefficient way of exploiting training batches.We conduct a wide range of ablative experiments with Matching and Prototypical Networks, two of the most popular methods that use nonparametric approaches at the level of the episode.Their "non-episodic'' counterparts are considerably simpler, have less hyperparameters, and improve their performance in multiple few-shot classification datasets.

        ----

        ## [1882] Pointwise Bounds for Distribution Estimation under Communication Constraints

        **Authors**: *Wei-Ning Chen, Peter Kairouz, Ayfer Özgür*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/ce4449660c6523b377b22a1dc2da5556-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/ce4449660c6523b377b22a1dc2da5556-Abstract.html)

        **Abstract**:

        We consider the problem of estimating a $d$-dimensional discrete distribution from its samples observed under a $b$-bit communication constraint. In contrast to most previous results that largely focus on the global minimax error, we study the local behavior of the estimation error and provide \emph{pointwise} bounds that depend on the target distribution $p$. In particular, we show that the $\ell_2$ error decays with $O\left(\frac{\lVert p\rVert_{1/2}}{n2^b}\vee \frac{1}{n}\right)$ when $n$ is sufficiently large, hence it is governed by the \emph{half-norm} of $p$ instead of the ambient dimension $d$. For the achievability result, we propose a two-round sequentially interactive estimation scheme that achieves this error rate uniformly over all $p$. Our scheme is based on a novel local refinement idea, where we first use a standard global minimax scheme to localize $p$ and then use the remaining samples to locally refine our estimate.We also develop a new local minimax lower bound with (almost) matching $\ell_2$ error, showing that any interactive scheme must admit a $\Omega\left( \frac{\lVert p \rVert_{{(1+\delta)}/{2}}}{n2^b}\right)$ $\ell_2$ error for any $\delta > 0$ when $n$ is sufficiently large. The lower bound is derived by first finding the best parametric sub-model containing $p$, and then upper bounding the quantized Fisher information under this model. Our upper and lower bounds together indicate that the $\mathsf{H}_{1/2}(p) = \log(\lVert p \rVert_{{1}/{2}})$ bits of communication is both sufficient and necessary to achieve the optimal (centralized) performance, where $\mathsf{H}_{{1}/{2}}(p)$ is the R\'enyi entropy of order $2$. Therefore, under the $\ell_2$ loss, the correct measure of the local communication complexity at $p$ is its R\'enyi entropy.

        ----

        ## [1883] CHIP: CHannel Independence-based Pruning for Compact Neural Networks

        **Authors**: *Yang Sui, Miao Yin, Yi Xie, Huy Phan, Saman A. Zonouz, Bo Yuan*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/ce6babd060aa46c61a5777902cca78af-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/ce6babd060aa46c61a5777902cca78af-Abstract.html)

        **Abstract**:

        Filter pruning has been widely used for neural network compression because of its enabled practical acceleration. To date, most of the existing filter pruning works explore the importance of filters via using intra-channel information. In this paper, starting from an inter-channel perspective, we propose to perform efficient filter pruning using Channel Independence, a metric that measures the correlations among different feature maps. The less independent feature map is interpreted as containing less useful information$/$knowledge, and hence its corresponding filter can be pruned without affecting model capacity. We systematically investigate the quantification metric, measuring scheme and sensitiveness$/$reliability of channel independence in the context of filter pruning. Our evaluation results for different models on various datasets show the superior performance of our approach. Notably, on CIFAR-10 dataset our solution can bring $0.75\%$ and $0.94\%$ accuracy increase over baseline ResNet-56 and ResNet-110 models, respectively, and meanwhile the model size and FLOPs are reduced by  $42.8\%$ and  $47.4\%$ (for ResNet-56) and $48.3\%$ and $52.1\%$ (for ResNet-110), respectively. On ImageNet dataset, our approach can achieve $40.8\%$ and $44.8\%$ storage and computation reductions, respectively, with $0.15\%$ accuracy increase over the baseline ResNet-50 model. The code is available at https://github.com/Eclipsess/CHIP_NeurIPS2021.

        ----

        ## [1884] Federated Split Task-Agnostic Vision Transformer for COVID-19 CXR Diagnosis

        **Authors**: *Sangjoon Park, Gwanghyun Kim, Jeongsol Kim, Boah Kim, Jong Chul Ye*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/ceb0595112db2513b9325a85761b7310-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/ceb0595112db2513b9325a85761b7310-Abstract.html)

        **Abstract**:

        Federated learning, which shares the weights of the neural network across clients, is gaining attention in the healthcare sector as it enables training on a large corpus of decentralized data while maintaining data privacy. For example, this enables neural network training for COVID-19 diagnosis on chest X-ray (CXR) images without collecting patient CXR data across multiple hospitals. Unfortunately, the exchange of the weights quickly consumes the network bandwidth if highly expressive network architecture is employed. So-called split learning partially solves this problem by dividing a neural network into a client and a server part, so that the client part of the network takes up less extensive computation resources and bandwidth. However, it is not clear how to find the optimal split without sacrificing the overall network performance. To amalgamate these methods and thereby maximize their distinct strengths, here we show that the Vision Transformer, a recently developed deep learning architecture with straightforward decomposable configuration, is ideally suitable for split learning without sacrificing performance. Even under the non-independent and identically distributed data distribution which emulates a real collaboration between hospitals using CXR datasets from multiple sources, the proposed framework was able to attain performance comparable to data-centralized training. In addition, the proposed framework along with heterogeneous multi-task clients also improves individual task performances including the diagnosis of COVID-19, eliminating the need for sharing large weights with innumerable parameters. Our results affirm the suitability of Transformer for collaborative learning in medical imaging and pave the way forward for future real-world implementations.

        ----

        ## [1885] Active Offline Policy Selection

        **Authors**: *Ksenia Konyushkova, Yutian Chen, Thomas Paine, Çaglar Gülçehre, Cosmin Paduraru, Daniel J. Mankowitz, Misha Denil, Nando de Freitas*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/cec2346566ba8ecd04bfd992fd193fb3-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/cec2346566ba8ecd04bfd992fd193fb3-Abstract.html)

        **Abstract**:

        This paper addresses the problem of policy selection in domains with abundant logged data, but with a restricted interaction budget. Solving this problem would enable safe evaluation and deployment of offline reinforcement learning policies in industry, robotics, and recommendation domains among others. Several off-policy evaluation (OPE) techniques have been proposed to assess the value of policies using only logged data. However, there is still a big gap between the evaluation by OPE and the full online evaluation in the real environment. Yet, large amounts of online interactions are often not possible in practice. To overcome this problem, we introduce active offline policy selection --- a novel sequential decision approach that combines logged data with online interaction to identify the best policy. This approach uses OPE estimates to warm start the online evaluation. Then, in order to utilize the limited environment interactions wisely we decide which policy to evaluate next based on a Bayesian optimization method with a kernel function that represents policy similarity. We use multiple benchmarks with a large number of candidate policies to show that the proposed approach improves upon state-of-the-art OPE estimates and pure online policy evaluation.

        ----

        ## [1886] Unsupervised Representation Transfer for Small Networks: I Believe I Can Distill On-the-Fly

        **Authors**: *Hee Min Choi, Hyoa Kang, Dokwan Oh*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/cecd845e3577efdaaf24eea03af4c033-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/cecd845e3577efdaaf24eea03af4c033-Abstract.html)

        **Abstract**:

        A current remarkable improvement of unsupervised visual representation learning is based on heavy networks with large-batch training. While recent methods have greatly reduced the gap between supervised and unsupervised performance of deep models such as ResNet-50, this development has been relatively limited for small models. In this work, we propose a novel unsupervised learning framework for small networks that combines deep self-supervised representation learning and knowledge distillation within one-phase training. In particular, a teacher model is trained to produce consistent cluster assignments between different views of the same image. Simultaneously, a student model is encouraged to mimic the prediction of on-the-fly self-supervised teacher. For effective knowledge transfer, we adopt the idea of domain classifier so that student training is guided by discriminative features invariant to the representational space shift between teacher and student. We also introduce a network driven multi-view generation paradigm to capture rich feature information contained in the network itself. Extensive experiments show that our student models surpass state-of-the-art offline distilled networks even from stronger self-supervised teachers as well as top-performing self-supervised models. Notably, our ResNet-18, trained with ResNet-50 teacher, achieves 68.3% ImageNet Top-1 accuracy on frozen feature linear evaluation, which is only 1.5% below the supervised baseline.

        ----

        ## [1887] Understanding Bandits with Graph Feedback

        **Authors**: *Houshuang Chen, Zengfeng Huang, Shuai Li, Chihao Zhang*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/cf004fdc76fa1a4f25f62e0eb5261ca3-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/cf004fdc76fa1a4f25f62e0eb5261ca3-Abstract.html)

        **Abstract**:

        The bandit problem with graph feedback, proposed in [Mannor and Shamir, NeurIPS 2011], is modeled by a directed graph $G=(V,E)$ where $V$ is the collection of bandit arms, and once an arm is triggered, all its incident arms are observed. A fundamental question is how the structure of the graph affects the min-max regret. We propose the notions of the fractional weak domination number $\delta^*$ and the $k$-packing independence number capturing upper bound and lower bound for the regret respectively.  We show that the two notions are inherently connected via aligning them with the linear program of the weakly dominating set and its dual --- the fractional vertex packing set respectively. Based on this connection, we utilize the strong duality theorem to prove a general regret upper bound $O\left(\left(\delta^*\log  |V|\right)^{\frac{1}{3}}T^{\frac{2}{3}}\right)$ and a lower bound $\Omega\left(\left(\delta^*/\alpha\right)^{\frac{1}{3}}T^{\frac{2}{3}}\right)$ where $\alpha$ is the integrality gap of the dual linear program. Therefore, our bounds are tight up to a $\left(\log |V|\right)^{\frac{1}{3}}$ factor on graphs with bounded integrality gap for the vertex packing problem including trees and graphs with bounded degree. Moreover, we show that for several special families of graphs, we can get rid of the $\left(\log |V|\right)^{\frac{1}{3}}$ factor and establish optimal regret.

        ----

        ## [1888] Information-theoretic generalization bounds for black-box learning algorithms

        **Authors**: *Hrayr Harutyunyan, Maxim Raginsky, Greg Ver Steeg, Aram Galstyan*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/cf0d02ec99e61a64137b8a2c3b03e030-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/cf0d02ec99e61a64137b8a2c3b03e030-Abstract.html)

        **Abstract**:

        We derive information-theoretic generalization bounds for supervised learning algorithms based on the information contained in predictions rather than in the output of the training algorithm. These bounds improve over the existing information-theoretic bounds, are applicable to a wider range of algorithms, and solve two key challenges: (a) they give meaningful results for deterministic algorithms and (b) they are significantly easier to estimate. We show experimentally that the proposed bounds closely follow the generalization gap in practical scenarios for deep learning.

        ----

        ## [1889] Trash or Treasure? An Interactive Dual-Stream Strategy for Single Image Reflection Separation

        **Authors**: *Qiming Hu, Xiaojie Guo*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/cf1f78fe923afe05f7597da2be7a3da8-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/cf1f78fe923afe05f7597da2be7a3da8-Abstract.html)

        **Abstract**:

        Single image reflection separation (SIRS), as a representative blind source separation task, aims to recover two layers, $\textit{i.e.}$, transmission and reflection, from one mixed observation, which is challenging due to the highly ill-posed nature. Existing deep learning based solutions typically restore the target layers individually, or with some concerns at the end of the output, barely taking into account the interaction across the two streams/branches. In order to utilize information more efficiently, this work presents a general yet simple interactive strategy, namely $\textit{your trash is my treasure}$ (YTMT), for constructing dual-stream decomposition networks. To be specific, we explicitly enforce the two streams to communicate with each other block-wisely. Inspired by the additive property between the two components, the interactive path can be easily built via transferring, instead of discarding, deactivated information by the ReLU rectifier from one stream to the other. Both ablation studies and experimental results on widely-used SIRS datasets are conducted to demonstrate the efficacy of YTMT, and reveal its superiority over other state-of-the-art alternatives. The implementation is quite simple and our code is publicly available at https://github.com/mingcv/YTMT-Strategy.

        ----

        ## [1890] Rot-Pro: Modeling Transitivity by Projection in Knowledge Graph Embedding

        **Authors**: *Tengwei Song, Jie Luo, Lei Huang*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/cf2f3fe19ffba462831d7f037a07fc83-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/cf2f3fe19ffba462831d7f037a07fc83-Abstract.html)

        **Abstract**:

        Knowledge graph embedding models learn the representations of entities and relations in the knowledge graphs for predicting missing links (relations) between entities. Their effectiveness are deeply affected by the ability of modeling and inferring different relation patterns such as symmetry, asymmetry, inversion, composition and transitivity. Although existing models are already able to model many of these relations patterns, transitivity, a very common relation pattern, is still not been fully supported. In this paper, we first theoretically show that the transitive relations can be modeled with projections. We then propose the Rot-Pro model which combines the projection and relational rotation together. We prove that Rot-Pro can infer all the above relation patterns. Experimental results show that the proposed Rot-Pro model  effectively learns the transitivity pattern and achieves the state-of-the-art results on the link prediction task in the datasets containing transitive relations.

        ----

        ## [1891] Planning from Pixels in Environments with Combinatorially Hard Search Spaces

        **Authors**: *Marco Bagatella, Miroslav Olsák, Michal Rolínek, Georg Martius*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/cf708fc1decf0337aded484f8f4519ae-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/cf708fc1decf0337aded484f8f4519ae-Abstract.html)

        **Abstract**:

        The ability to form complex plans based on raw visual input is a litmus test for current capabilities of artificial intelligence, as it requires a seamless combination of visual processing and abstract algorithmic execution, two traditionally separate areas of computer science. A recent surge of interest in this field brought advances that yield good performance in tasks ranging from arcade games to continuous control; these methods however do not come without significant issues, such as limited generalization capabilities and difficulties when dealing with combinatorially hard planning instances. Our contribution is two-fold: (i) we present a method that learns to represent its environment as a latent graph and leverages state reidentification to reduce the complexity of finding a good policy from exponential to linear (ii) we introduce a set of lightweight environments with an underlying discrete combinatorial structure in which planning is challenging even for humans. Moreover, we show that our methods achieves strong empirical generalization to variations in the environment, even across highly disadvantaged regimes, such as “one-shot” planning, or in an offline RL paradigm which only provides low-quality trajectories.

        ----

        ## [1892] PLUGIn: A simple algorithm for inverting generative models with recovery guarantees

        **Authors**: *Babhru Joshi, Xiaowei Li, Yaniv Plan, Özgür Yilmaz*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/cf77e1f8490495e9f8dedceaf372f969-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/cf77e1f8490495e9f8dedceaf372f969-Abstract.html)

        **Abstract**:

        We consider the problem of recovering an unknown latent code vector under a known generative model. For a $d$-layer deep generative network $\mathcal{G}:\mathbb{R}^{n_0}\rightarrow \mathbb{R}^{n_d}$ with ReLU activation functions, let the observation be $\mathcal{G}(x)+\epsilon$ where $\epsilon$ is noise. We introduce a simple novel algorithm, Partially Linearized Update for Generative Inversion (PLUGIn), to estimate $x$ (and thus $\mathcal{G}(x)$). We prove that, when weights are Gaussian and layer widths $n_i \gtrsim 5^i n_0$ (up to log factors), the algorithm converges geometrically to a neighbourhood of $x$ with high probability. Note the inequality on layer widths allows $n_i>n_{i+1}$ when $i\geq 1$. To our knowledge, this is the first such result for networks with some contractive layers. After a sufficient number of iterations, the estimation errors for both $x$ and $\mathcal{G}(x)$ are at most in the order of $\sqrt{4^dn_0/n_d} \|\epsilon\|$. Thus, the algorithm can denoise when the expansion ratio $n_d/n_0$ is large. Numerical experiments on synthetic data and real data are provided to validate our theoretical results and to illustrate that the algorithm can effectively remove artifacts in an image.

        ----

        ## [1893] Modular Gaussian Processes for Transfer Learning

        **Authors**: *Pablo Moreno-Muñoz, Antonio Artés-Rodríguez, Mauricio A. Álvarez*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/cf79ae6addba60ad018347359bd144d2-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/cf79ae6addba60ad018347359bd144d2-Abstract.html)

        **Abstract**:

        We present a framework for transfer learning based on modular variational Gaussian processes (GP). We develop a module-based method that having a dictionary of well fitted GPs, each model being characterised by its hyperparameters, pseudo-inputs and their corresponding posterior densities, one could build ensemble GP models without revisiting any data. Our method avoids undesired data centralisation, reduces rising computational costs and allows the transfer of learned uncertainty metrics after training. We exploit the augmentation of high-dimensional integral operators based on the Kullback-Leibler divergence between stochastic processes to introduce an efficient lower bound under all the sparse variational GPs, with different complexity and even likelihood distribution. The method is also valid for multi-output GPs, learning correlations a posteriori between independent modules. Extensive results illustrate the usability of our framework in large-scale and multi-task experiments, also compared with the exact inference methods in the literature.

        ----

        ## [1894] Neural Human Performer: Learning Generalizable Radiance Fields for Human Performance Rendering

        **Authors**: *Youngjoong Kwon, Dahun Kim, Duygu Ceylan, Henry Fuchs*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/cf866614b6b18cda13fe699a3a65661b-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/cf866614b6b18cda13fe699a3a65661b-Abstract.html)

        **Abstract**:

        In this paper, we aim at synthesizing a free-viewpoint video of an arbitrary human performance using sparse multi-view cameras. Recently, several works have addressed this problem by learning person-specific neural radiance fields (NeRF) to capture the appearance of a particular human. In parallel, some work proposed to use pixel-aligned features to generalize radiance fields to arbitrary new scenes and objects. Adopting such generalization approaches to humans, however, is highly challenging due to the heavy occlusions and dynamic articulations of body parts. To tackle this, we propose Neural Human Performer, a novel approach that learns generalizable neural radiance fields based on a parametric human body model for robust performance capture. Specifically, we first introduce a temporal transformer that aggregates tracked visual features based on the skeletal body motion over time. Moreover, a multi-view transformer is proposed to perform cross-attention between the temporally-fused features and the pixel-aligned features at each time step to integrate observations on the fly from multiple views. Experiments on the ZJU-MoCap and AIST datasets show that our method significantly outperforms recent generalizable NeRF methods on unseen identities and poses.

        ----

        ## [1895] Locally differentially private estimation of functionals of discrete distributions

        **Authors**: *Cristina Butucea, Yann Issartel*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/cf8c9be2a4508a24ae92c9d3d379131d-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/cf8c9be2a4508a24ae92c9d3d379131d-Abstract.html)

        **Abstract**:

        We study the  problem of estimating non-linear functionals of discrete distributions in the context of local differential privacy. The initial data $x_1,\ldots,x_n \in[K]$ are supposed i.i.d. and distributed according to an unknown discrete distribution $p = (p_1,\ldots,p_K)$. Only $\alpha$-locally differentially private (LDP) samples $z_1,...,z_n$ are publicly available, where the term 'local' means that each $z_i$ is produced using one individual attribute $x_i$. We exhibit privacy mechanisms (PM) that are interactive (i.e. they are allowed to use already published confidential data) or non-interactive. We describe the behavior of the quadratic risk for estimating the power sum functional $F_{\gamma} = \sum_{k=1}^K p_k^{\gamma}$, $\gamma >0$ as a function of $K, \, n$ and $\alpha$. In the non-interactive case, we study twol plug-in type estimators of $F_{\gamma}$, for all $\gamma >0$, that are similar to the MLE analyzed by Jiao et al. (2017) in the multinomial model. However, due to the privacy constraint the rates we attain are slower and similar to those obtained in the Gaussian model by Collier et al. (2020). In the sequentially interactive case, we introduce for all $\gamma >1$ a two-step procedure which attains the parametric rate $(n \alpha^2)^{-1/2}$ when $\gamma \geq 2$.  We give lower bounds results over all $\alpha-$LDP mechanisms and over all estimators using the private samples.

        ----

        ## [1896] Asymptotics of representation learning in finite Bayesian neural networks

        **Authors**: *Jacob A. Zavatone-Veth, Abdulkadir Canatar, Benjamin S. Ruben, Cengiz Pehlevan*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/cf9dc5e4e194fc21f397b4cac9cc3ae9-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/cf9dc5e4e194fc21f397b4cac9cc3ae9-Abstract.html)

        **Abstract**:

        Recent works have suggested that finite Bayesian neural networks may sometimes outperform their infinite cousins because finite networks can flexibly adapt their internal representations. However, our theoretical understanding of how the learned hidden layer representations of finite networks differ from the fixed representations of infinite networks remains incomplete. Perturbative finite-width corrections to the network prior and posterior have been studied, but the asymptotics of learned features have not been fully characterized. Here, we argue that the leading finite-width corrections to the average feature kernels for any Bayesian network with linear readout and Gaussian likelihood have a largely universal form. We illustrate this explicitly for three tractable network architectures: deep linear fully-connected and convolutional networks, and networks with a single nonlinear hidden layer. Our results begin to elucidate how task-relevant learning signals shape the hidden layer representations of wide Bayesian neural networks.

        ----

        ## [1897] Adaptive Ensemble Q-learning: Minimizing Estimation Bias via Error Feedback

        **Authors**: *Hang Wang, Sen Lin, Junshan Zhang*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/cfa45151ccad6bf11ea146ed563f2119-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/cfa45151ccad6bf11ea146ed563f2119-Abstract.html)

        **Abstract**:

        The ensemble method is a promising way to mitigate the overestimation issue in Q-learning, where multiple function approximators are used to estimate the action values. It is known that the estimation bias hinges heavily on the ensemble size (i.e., the number of  Q-function approximators used in the target), and that determining the 'right' ensemble size is highly nontrivial, because of the time-varying nature of the function approximation errors during the learning process. To tackle this challenge, we first derive an upper bound and a lower bound on the  estimation bias, based on which the ensemble size is  adapted to drive the bias to be nearly zero, thereby coping with the impact of the time-varying approximation errors accordingly. Motivated by the theoretic findings, we advocate that the ensemble method can be combined with Model Identification Adaptive Control (MIAC) for effective ensemble size adaptation. Specifically, we devise Adaptive Ensemble Q-learning (AdaEQ), a generalized ensemble method with two key steps: (a) approximation error characterization which serves as the feedback for flexibly controlling the ensemble size, and (b) ensemble size adaptation tailored towards minimizing the estimation bias.   Extensive experiments are carried out to show that AdaEQ can  improve the learning performance than the existing methods for the MuJoCo benchmark.

        ----

        ## [1898] Domain Adaptation with Invariant Representation Learning: What Transformations to Learn?

        **Authors**: *Petar Stojanov, Zijian Li, Mingming Gong, Ruichu Cai, Jaime G. Carbonell, Kun Zhang*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/cfc5d9422f0c8f8ad796711102dbe32b-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/cfc5d9422f0c8f8ad796711102dbe32b-Abstract.html)

        **Abstract**:

        Unsupervised domain adaptation, as a prevalent transfer learning setting, spans many real-world applications. With the increasing representational power and applicability of neural networks, state-of-the-art domain adaptation methods make use of deep architectures to map the input features $X$ to a latent representation $Z$ that has the same marginal  distribution across domains. This has been shown to be insufficient for generating optimal representation for classification, and to find conditionally invariant representations, usually strong assumptions are needed. We provide reasoning why when the supports of the source and target data from overlap, any map of $X$ that is fixed across domains may not be suitable for domain adaptation via invariant features. Furthermore, we develop an efficient technique in which  the optimal map from $X$ to $Z$ also takes domain-specific information as input, in addition to the features $X$. By using the property of minimal changes of causal mechanisms across domains, our model also takes into account the domain-specific information to ensure that the latent representation $Z$ does not discard valuable information about $Y$. We demonstrate the efficacy of our method via synthetic and real-world data experiments. The code is available at: \texttt{https://github.com/DMIRLAB-Group/DSAN}.

        ----

        ## [1899] CSDI: Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation

        **Authors**: *Yusuke Tashiro, Jiaming Song, Yang Song, Stefano Ermon*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/cfe8504bda37b575c70ee1a8276f3486-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/cfe8504bda37b575c70ee1a8276f3486-Abstract.html)

        **Abstract**:

        The imputation of missing values in time series has many applications in healthcare and finance. While autoregressive models are natural candidates for time series imputation, score-based diffusion models have recently outperformed existing counterparts including autoregressive models in many tasks such as image generation and audio synthesis, and would be promising for time series imputation. In this paper, we propose Conditional Score-based Diffusion model (CSDI), a novel time series imputation method that utilizes score-based diffusion models conditioned on observed data. Unlike existing score-based approaches, the conditional diffusion model is explicitly trained for imputation and can exploit correlations between observed values. On healthcare and environmental data, CSDI improves by 40-65% over existing probabilistic imputation methods on popular performance metrics. In addition, deterministic imputation by CSDI reduces the error by 5-20% compared to the state-of-the-art deterministic imputation methods. Furthermore, CSDI can also be applied to time series interpolation and probabilistic forecasting, and is competitive with existing baselines. The code is available at https://github.com/ermongroup/CSDI.

        ----

        ## [1900] Causal Bandits with Unknown Graph Structure

        **Authors**: *Yangyi Lu, Amirhossein Meisami, Ambuj Tewari*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/d010396ca8abf6ead8cacc2c2f2f26c7-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/d010396ca8abf6ead8cacc2c2f2f26c7-Abstract.html)

        **Abstract**:

        In causal bandit problems the action set consists of interventions on variables of a causal graph. Several researchers have recently studied such bandit problems and pointed out their practical applications. However, all existing works rely on a restrictive and impractical assumption that the learner is given full knowledge of the causal graph structure upfront. In this paper, we develop novel causal bandit algorithms without knowing the causal graph. Our algorithms work well for causal trees, causal forests and a general class of causal graphs. The regret guarantees of our algorithms greatly improve upon those of  standard multi-armed bandit (MAB) algorithms under mild conditions. Lastly, we prove our mild conditions are necessary: without them one cannot do better than standard MAB algorithms.

        ----

        ## [1901] Piper: Multidimensional Planner for DNN Parallelization

        **Authors**: *Jakub Tarnawski, Deepak Narayanan, Amar Phanishayee*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/d01eeca8b24321cd2fe89dd85b9beb51-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/d01eeca8b24321cd2fe89dd85b9beb51-Abstract.html)

        **Abstract**:

        The rapid increase in sizes of state-of-the-art DNN models, and consequently the increase in the compute and memory requirements of model training, has led to the development of many execution schemes such as data parallelism, pipeline model parallelism, tensor (intra-layer) model parallelism, and various memory-saving optimizations. However, no prior work has tackled the highly complex problem of optimally partitioning the DNN computation graph across many accelerators while combining all these parallelism modes and optimizations.In this work, we introduce Piper, an efficient optimization algorithm for this problem that is based on a two-level dynamic programming approach. Our two-level approach is driven by the insight that being given tensor-parallelization techniques for individual layers (e.g., Megatron-LM's splits for transformer layers) significantly reduces the search space and makes the global problem tractable, compared to considering tensor-parallel configurations for the entire DNN operator graph.

        ----

        ## [1902] Causal Effect Inference for Structured Treatments

        **Authors**: *Jean Kaddour, Yuchen Zhu, Qi Liu, Matt J. Kusner, Ricardo Silva*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/d02e9bdc27a894e882fa0c9055c99722-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/d02e9bdc27a894e882fa0c9055c99722-Abstract.html)

        **Abstract**:

        We address the estimation of conditional average treatment effects (CATEs) for structured treatments (e.g., graphs, images, texts). Given a weak condition on the effect, we propose the generalized Robinson decomposition, which (i) isolates the causal estimand (reducing regularization bias), (ii) allows one to plug in arbitrary models for learning, and (iii) possesses a quasi-oracle convergence guarantee under mild assumptions. In experiments with small-world and molecular graphs we demonstrate that our approach outperforms prior work in CATE estimation.

        ----

        ## [1903] Efficient hierarchical Bayesian inference for spatio-temporal regression models in neuroimaging

        **Authors**: *Ali Hashemi, Yijing Gao, Chang Cai, Sanjay Ghosh, Klaus-Robert Müller, Srikantan S. Nagarajan, Stefan Haufe*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/d03a857a23b5285736c4d55e0bb067c8-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/d03a857a23b5285736c4d55e0bb067c8-Abstract.html)

        **Abstract**:

        Several problems in neuroimaging and beyond require inference on the parameters of multi-task sparse hierarchical regression models. Examples include M/EEG inverse problems, neural encoding models for task-based fMRI analyses, and climate science. In these domains, both the model parameters to be inferred and the measurement noise may exhibit a complex spatio-temporal structure. Existing work either neglects the temporal structure or leads to computationally demanding inference schemes. Overcoming these limitations, we devise a novel flexible hierarchical Bayesian framework within which the spatio-temporal dynamics of model parameters and noise are modeled to have Kronecker product covariance structure. Inference in our framework is based on majorization-minimization optimization and has guaranteed convergence properties. Our highly efficient algorithms exploit the intrinsic Riemannian geometry of temporal autocovariance matrices. For stationary dynamics described by Toeplitz matrices, the theory of circulant embeddings is employed. We prove convex bounding properties and derive update rules of the resulting algorithms. On both synthetic and real neural data from M/EEG, we demonstrate that our methods lead to improved performance.

        ----

        ## [1904] Topological Attention for Time Series Forecasting

        **Authors**: *Sebastian Zeng, Florian Graf, Christoph D. Hofer, Roland Kwitt*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/d062f3e278a1fbba2303ff5a22e8c75e-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/d062f3e278a1fbba2303ff5a22e8c75e-Abstract.html)

        **Abstract**:

        The problem of (point) forecasting univariate time series is considered. Most approaches, ranging from traditional statistical methods to recent learning-based techniques with neural networks, directly operate on raw time series observations. As an extension, we study whether local topological properties, as captured via persistent homology, can serve as a reliable signal that provides complementary information for learning to forecast. To this end, we propose topological attention, which allows attending to local topological features within a time horizon of historical data. Our approach easily integrates into existing end-to-end trainable forecasting models, such as N-BEATS, and, in combination with the latter exhibits state-of-the-art performance on the large-scale M4 benchmark dataset of 100,000 diverse time series from different domains. Ablation experiments, as well as a comparison to recent techniques in a setting where only a single time series is available for training, corroborate the beneficial nature of including local topological information through an attention mechanism.

        ----

        ## [1905] Local Signal Adaptivity: Provable Feature Learning in Neural Networks Beyond Kernels

        **Authors**: *Stefani Karp, Ezra Winston, Yuanzhi Li, Aarti Singh*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/d064bf1ad039ff366564f352226e7640-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/d064bf1ad039ff366564f352226e7640-Abstract.html)

        **Abstract**:

        Neural networks have been shown to outperform kernel methods in practice (including neural tangent kernels). Most theoretical explanations of this performance gap focus on learning a complex hypothesis class; in some cases, it is unclear whether this hypothesis class captures realistic data. In this work, we propose a related, but alternative, explanation for this performance gap in the image classification setting, based on finding a sparse signal in the presence of noise. Specifically, we prove that, for a simple data distribution with sparse signal amidst high-variance noise, a simple convolutional neural network trained using stochastic gradient descent learns to threshold out the noise and find the signal. On the other hand, the corresponding neural tangent kernel, with a fixed set of predetermined features, is unable to adapt to the signal in this manner. We supplement our theoretical results by demonstrating this phenomenon empirically: in CIFAR-10 and MNIST images with various backgrounds, as the background noise increases in intensity, a CNN's performance stays relatively robust, whereas its corresponding neural tangent kernel sees a notable drop in performance. We therefore propose the "local signal adaptivity" (LSA) phenomenon as one explanation for the superiority of neural networks over kernel methods.

        ----

        ## [1906] IA-RED$^2$: Interpretability-Aware Redundancy Reduction for Vision Transformers

        **Authors**: *Bowen Pan, Rameswar Panda, Yifan Jiang, Zhangyang Wang, Rogério Feris, Aude Oliva*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/d072677d210ac4c03ba046120f0802ec-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/d072677d210ac4c03ba046120f0802ec-Abstract.html)

        **Abstract**:

        The self-attention-based model, transformer, is recently becoming the leading backbone in the field of computer vision. In spite of the impressive success made by transformers in a variety of vision tasks, it still suffers from heavy computation and intensive memory costs. To address this limitation, this paper presents an Interpretability-Aware REDundancy REDuction framework (IA-RED$^2$). We start by observing a large amount of redundant computation, mainly spent on uncorrelated input patches, and then introduce an interpretable module to dynamically and gracefully drop these redundant patches. This novel framework is then extended to a hierarchical structure, where uncorrelated tokens at different stages are gradually removed, resulting in a considerable shrinkage of computational cost. We include extensive experiments on both image and video tasks, where our method could deliver up to 1.4x speed-up for state-of-the-art models like DeiT and TimeSformer, by only sacrificing less than 0.7% accuracy. More importantly, contrary to other acceleration approaches, our method is inherently interpretable with substantial visual evidence, making vision transformer closer to a more human-understandable architecture while being lighter. We demonstrate that the interpretability that naturally emerged in our framework can outperform the raw attention learned by the original visual transformer, as well as those generated by off-the-shelf interpretation methods, with both qualitative and quantitative results. Project Page: http://people.csail.mit.edu/bpan/ia-red/.

        ----

        ## [1907] Symbolic Regression via Deep Reinforcement Learning Enhanced Genetic Programming Seeding

        **Authors**: *T. Nathan Mundhenk, Mikel Landajuela, Ruben Glatt, Cláudio P. Santiago, Daniel M. Faissol, Brenden K. Petersen*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/d073bb8d0c47f317dd39de9c9f004e9d-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/d073bb8d0c47f317dd39de9c9f004e9d-Abstract.html)

        **Abstract**:

        Symbolic regression is the process of identifying mathematical expressions that fit observed output from a black-box process. It is a discrete optimization problem generally believed to be NP-hard. Prior approaches to solving the problem include neural-guided search (e.g. using reinforcement learning) and genetic programming. In this work, we introduce a hybrid neural-guided/genetic programming approach to symbolic regression and other combinatorial optimization problems. We propose a neural-guided component used to seed the starting population of a random restart genetic programming component, gradually learning better starting populations. On a number of common benchmark tasks to recover underlying expressions from a dataset, our method recovers 65% more expressions than a recently published top-performing model using the same experimental setup. We demonstrate that running many genetic programming generations without interdependence on the neural-guided component performs better for symbolic regression than alternative formulations where the two are more strongly coupled. Finally, we introduce a new set of 22 symbolic regression benchmark problems with increased difficulty over existing benchmarks. Source code is provided at www.github.com/brendenpetersen/deep-symbolic-optimization.

        ----

        ## [1908] Choose a Transformer: Fourier or Galerkin

        **Authors**: *Shuhao Cao*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/d0921d442ee91b896ad95059d13df618-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/d0921d442ee91b896ad95059d13df618-Abstract.html)

        **Abstract**:

        In this paper, we apply the self-attention from the state-of-the-art Transformer in Attention Is All You Need for the first time to a data-driven operator learning problem related to partial differential equations. An effort is put together to explain the heuristics of, and to improve the efficacy of the attention mechanism. By employing the operator approximation theory in Hilbert spaces, it is demonstrated for the first time that the softmax normalization in the scaled dot-product attention is sufficient but not necessary. Without softmax, the approximation capacity of a linearized Transformer variant can be proved to be comparable to a Petrov-Galerkin projection layer-wise, and the estimate is independent with respect to the sequence length. A new layer normalization scheme mimicking the Petrov-Galerkin projection is proposed to allow a scaling to propagate through attention layers, which helps the model achieve remarkable accuracy in operator learning tasks with unnormalized data. Finally, we present three operator learning experiments, including the viscid Burgers' equation, an interface Darcy flow, and an inverse interface coefficient identification problem. The newly proposed simple attention-based operator learner, Galerkin Transformer, shows significant improvements in both training cost and evaluation accuracy over its softmax-normalized counterparts.

        ----

        ## [1909] A Causal Lens for Controllable Text Generation

        **Authors**: *Zhiting Hu, Li Erran Li*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/d0f5edad9ac19abed9e235c0fe0aa59f-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/d0f5edad9ac19abed9e235c0fe0aa59f-Abstract.html)

        **Abstract**:

        Controllable text generation concerns two fundamental tasks of wide applications, namely generating text of given attributes (i.e., attribute-conditional generation), and minimally editing existing text to possess desired attributes (i.e., text attribute transfer). Extensive prior work has largely studied the two problems separately, and developed different conditional models which, however, are prone to producing biased text (e.g., various gender stereotypes). This paper proposes to formulate controllable text generation from a principled causal perspective which models the two tasks with a unified framework. A direct advantage of the causal formulation is the use of  rich causality tools to mitigate generation biases and improve control. We treat the two tasks as interventional and counterfactual causal inference based on a structural causal model, respectively. We then apply the framework to the challenging practical setting where confounding factors (that induce spurious correlations) are observable only on a small fraction of data. Experiments show significant superiority of the causal approach over previous conditional models for improved control accuracy and reduced bias.

        ----

        ## [1910] Differentially Private Multi-Armed Bandits in the Shuffle Model

        **Authors**: *Jay Tenenbaum, Haim Kaplan, Yishay Mansour, Uri Stemmer*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/d14388bb836687ff2b16b7bee6bab182-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/d14388bb836687ff2b16b7bee6bab182-Abstract.html)

        **Abstract**:

        We give an $(\varepsilon,\delta)$-differentially private algorithm for the Multi-Armed Bandit (MAB) problem in the shuffle model with a distribution-dependent regret of $O\left(\left(\sum_{a:\Delta_a>0}\frac{\log T}{\Delta_a}\right)+\frac{k\sqrt{\log\frac{1}{\delta}}\log T}{\varepsilon}\right)$, and a distribution-independent regret of $O\left(\sqrt{kT\log T}+\frac{k\sqrt{\log\frac{1}{\delta}}\log T}{\varepsilon}\right)$, where $T$ is the number of rounds, $\Delta_a$ is the suboptimality gap of the action $a$, and $k$ is the total number of actions. Our upper bound almost matches the regret of the best known algorithms for the centralized model, and significantly outperforms the best known algorithm in the local model.

        ----

        ## [1911] Dual Adaptivity: A Universal Algorithm for Minimizing the Adaptive Regret of Convex Functions

        **Authors**: *Lijun Zhang, Guanghui Wang, Wei-Wei Tu, Wei Jiang, Zhi-Hua Zhou*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/d1588e685562af341ff2448de4b674d1-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/d1588e685562af341ff2448de4b674d1-Abstract.html)

        **Abstract**:

        To deal with changing environments, a new performance measureâ€”adaptive regret, defined as the maximum static regret over any interval, was proposed in online learning. Under the setting of online convex optimization, several algorithms have been successfully developed to minimize the adaptive regret. However, existing algorithms lack universality in the sense that they can only handle one type of convex functions and need apriori knowledge of parameters. By contrast, there exist universal algorithms, such as MetaGrad, that attain optimal static regret for multiple types of convex functions simultaneously. Along this line of research, this paper presents the first universal algorithm for minimizing the adaptive regret of convex functions. Specifically, we borrow the idea of maintaining multiple learning rates in MetaGrad to handle the uncertainty of functions, and utilize the technique of sleeping experts to capture changing environments. In this way, our algorithm automatically adapts to the property of functions (convex, exponentially concave, or strongly convex), as well as the nature of environments (stationary or changing). As a by product, it also allows the type of functions to switch between rounds.

        ----

        ## [1912] Learning Hard Optimization Problems: A Data Generation Perspective

        **Authors**: *James Kotary, Ferdinando Fioretto, Pascal Van Hentenryck*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/d1942a3ab01eb59220e2b3a46e7ef09d-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/d1942a3ab01eb59220e2b3a46e7ef09d-Abstract.html)

        **Abstract**:

        Optimization problems are ubiquitous in our societies and are present in almost every segment of the economy. Most of these optimization problems are NP-hard and computationally demanding, often requiring approximate solutions for large-scale instances. Machine learning frameworks that learn to approximate solutions to such hard optimization problems are a potentially promising avenue to address these difficulties, particularly when many closely related problem instances must be solved repeatedly. Supervised learning frameworks can train a model using the outputs of pre-solved instances. However, when the outputs are themselves approximations, when the optimization problem has symmetric solutions, and/or when the solver uses randomization, solutions to closely related instances may exhibit large differences and the learning task can become inherently more difficult. This paper demonstrates this critical challenge, connects the volatility of the training data to the ability of a model to approximate it, and proposes a method for producing (exact or approximate) solutions to optimization problems that are more amenable to supervised learning tasks.  The effectiveness of the method is tested on hard non-linear nonconvex and discrete combinatorial problems.

        ----

        ## [1913] Canonical Capsules: Self-Supervised Capsules in Canonical Pose

        **Authors**: *Weiwei Sun, Andrea Tagliasacchi, Boyang Deng, Sara Sabour, Soroosh Yazdani, Geoffrey E. Hinton, Kwang Moo Yi*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/d1ee59e20ad01cedc15f5118a7626099-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/d1ee59e20ad01cedc15f5118a7626099-Abstract.html)

        **Abstract**:

        We propose a self-supervised capsule architecture for 3D point clouds. We compute capsule decompositions of objects through permutation-equivariant attention, and self-supervise the process by training with pairs of randomly rotated objects. Our key idea is to aggregate the attention masks into semantic keypoints, and use these to supervise a decomposition that satisfies the capsule invariance/equivariance properties. This not only enables the training of a semantically consistent decomposition, but also allows us to learn a canonicalization operation that enables object-centric reasoning. To train our neural network we require neither classification labels nor manually-aligned training datasets. Yet, by learning an object-centric representation in a self-supervised manner, our method outperforms the state-of-the-art on 3D point cloud reconstruction, canonicalization, and unsupervised classification.

        ----

        ## [1914] Characterizing Generalization under Out-Of-Distribution Shifts in Deep Metric Learning

        **Authors**: *Timo Milbich, Karsten Roth, Samarth Sinha, Ludwig Schmidt, Marzyeh Ghassemi, Björn Ommer*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/d1f255a373a3cef72e03aa9d980c7eca-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/d1f255a373a3cef72e03aa9d980c7eca-Abstract.html)

        **Abstract**:

        Deep Metric Learning (DML) aims to find representations suitable for zero-shot transfer to a priori unknown test distributions. However, common evaluation protocols only test a single, fixed data split in which train and test classes are assigned randomly. More realistic evaluations should consider a broad spectrum of distribution shifts with potentially varying degree and difficulty.In this work, we systematically construct train-test splits of increasing difficulty and present the ooDML benchmark to characterize generalization under out-of-distribution shifts in DML. ooDML is designed to probe the generalization performance on much more challenging, diverse train-to-test distribution shifts. Based on our new benchmark, we conduct a thorough empirical analysis of state-of-the-art DML methods. We find that while generalization tends to consistently degrade with difficulty, some methods are better at retaining performance as the distribution shift increases. Finally, we propose few-shot DML as an efficient way to consistently improve generalization in response to unknown test shifts presented in ooDML.

        ----

        ## [1915] Dynamics-regulated kinematic policy for egocentric pose estimation

        **Authors**: *Zhengyi Luo, Ryo Hachiuma, Ye Yuan, Kris Kitani*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/d1fe173d08e959397adf34b1d77e88d7-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/d1fe173d08e959397adf34b1d77e88d7-Abstract.html)

        **Abstract**:

        We propose a method for object-aware 3D egocentric pose estimation that tightly integrates kinematics modeling, dynamics modeling, and scene object information. Unlike prior kinematics or dynamics-based approaches where the two components are used disjointly, we synergize the two approaches via dynamics-regulated training. At each timestep, a kinematic model is used to provide a target pose using video evidence and simulation state. Then, a prelearned dynamics model attempts to mimic the kinematic pose in a physics simulator. By comparing the pose instructed by the kinematic model against the pose generated by the dynamics model, we can use their misalignment to further improve the kinematic model. By factoring in the 6DoF pose of objects (e.g., chairs, boxes) in the scene, we demonstrate for the first time, the ability to estimate physically-plausible 3D human-object interactions using a single wearable camera. We evaluate our egocentric pose estimation method in both controlled laboratory settings and real-world scenarios.

        ----

        ## [1916] Never Go Full Batch (in Stochastic Convex Optimization)

        **Authors**: *Idan Amir, Yair Carmon, Tomer Koren, Roi Livni*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/d27b95cac4c27feb850aaa4070cc4675-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/d27b95cac4c27feb850aaa4070cc4675-Abstract.html)

        **Abstract**:

        We study the generalization performance of $\text{\emph{full-batch}}$ optimization algorithms for stochastic convex optimization: these are first-order methods that only access the exact gradient of the empirical risk (rather than gradients with respect to individual data points), that include a wide range of algorithms such as gradient descent, mirror descent, and their regularized and/or accelerated variants. We provide a new separation result showing that, while algorithms such as stochastic gradient descent can generalize and optimize the population risk to within $\epsilon$ after $O(1/\epsilon^2)$ iterations, full-batch methods either need at least $\Omega(1/\epsilon^4)$ iterations or exhibit a dimension-dependent sample complexity.

        ----

        ## [1917] Collaborative Learning in the Jungle (Decentralized, Byzantine, Heterogeneous, Asynchronous and Nonconvex Learning)

        **Authors**: *El-Mahdi El-Mhamdi, Sadegh Farhadkhani, Rachid Guerraoui, Arsany Guirguis, Lê-Nguyên Hoang, Sébastien Rouault*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/d2cd33e9c0236a8c2d8bd3fa91ad3acf-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/d2cd33e9c0236a8c2d8bd3fa91ad3acf-Abstract.html)

        **Abstract**:

        We study \emph{Byzantine collaborative learning}, where $n$ nodes seek to collectively learn from each others' local data. The data distribution may vary from one node to another. No node is trusted, and $f < n$ nodes can behave arbitrarily. We prove that collaborative learning is equivalent to a new form of agreement, which we call \emph{averaging agreement}. In this problem, nodes start each with an initial vector and seek to approximately agree on a common vector, which is close to the average of honest nodes' initial vectors.  We present two asynchronous solutions to averaging agreement, each we prove optimal according to some dimension. The first, based on the minimum-diameter averaging, requires $n \geq 6f+1$, but achieves asymptotically the best-possible averaging constant up to a multiplicative constant. The second, based on reliable broadcast and coordinate-wise trimmed mean, achieves optimal Byzantine resilience, i.e.,  $n \geq 3f+1$. Each of these algorithms induces an optimal Byzantine collaborative learning protocol. In particular, our equivalence yields new impossibility theorems on what any collaborative learning algorithm can achieve in adversarial and heterogeneous environments.

        ----

        ## [1918] Not All Low-Pass Filters are Robust in Graph Convolutional Networks

        **Authors**: *Heng Chang, Yu Rong, Tingyang Xu, Yatao Bian, Shiji Zhou, Xin Wang, Junzhou Huang, Wenwu Zhu*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/d30960ce77e83d896503d43ba249caf7-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/d30960ce77e83d896503d43ba249caf7-Abstract.html)

        **Abstract**:

        Graph Convolutional Networks (GCNs) are promising deep learning approaches in learning representations for graph-structured data. Despite the proliferation of such methods, it is well known that they are vulnerable to carefully crafted adversarial attacks on the graph structure. In this paper, we first conduct an adversarial vulnerability analysis based on matrix perturbation theory. We prove that the low- frequency components of the symmetric normalized Laplacian, which is usually used as the convolutional filter in GCNs, could be more robust against structural perturbations when their eigenvalues fall into a certain robust interval. Our results indicate that not all low-frequency components are robust to adversarial attacks and provide a deeper understanding of the relationship between graph spectrum and robustness of GCNs. Motivated by the theory, we present GCN-LFR, a general robust co-training paradigm for GCN-based models, that encourages transferring the robustness of low-frequency components with an auxiliary neural network. To this end, GCN-LFR could enhance the robustness of various kinds of GCN-based models against poisoning structural attacks in a plug-and-play manner. Extensive experiments across five benchmark datasets and five GCN-based models also confirm that GCN-LFR is resistant to the adversarial attacks without compromising on performance in the benign situation.

        ----

        ## [1919] Counterfactual Maximum Likelihood Estimation for Training Deep Networks

        **Authors**: *Xinyi Wang, Wenhu Chen, Michael Saxon, William Yang Wang*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/d30d0f522a86b3665d8e3a9a91472e28-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/d30d0f522a86b3665d8e3a9a91472e28-Abstract.html)

        **Abstract**:

        Although deep learning models have driven state-of-the-art performance on a wide array of tasks, they are prone to spurious correlations that should not be learned as predictive clues. To mitigate this problem, we propose a causality-based training framework to reduce the spurious correlations caused by observed confounders. We give theoretical analysis on the underlying general Structural Causal Model (SCM) and propose to perform Maximum Likelihood Estimation (MLE) on the interventional distribution instead of the observational distribution, namely Counterfactual Maximum Likelihood Estimation (CMLE). As the interventional distribution, in general, is hidden from the observational data, we then derive two different upper bounds of the expected negative log-likelihood and propose two general algorithms, Implicit CMLE and Explicit CMLE, for causal predictions of deep learning models using observational data. We conduct experiments on both simulated data and two real-world tasks: Natural Language Inference (NLI) and Image Captioning. The results show that CMLE methods outperform the regular MLE method in terms of out-of-domain generalization performance and reducing spurious correlations, while maintaining comparable performance on the regular evaluations.

        ----

        ## [1920] Robust Optimization for Multilingual Translation with Imbalanced Data

        **Authors**: *Xian Li, Hongyu Gong*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/d324a0cc02881779dcda44a675fdcaaa-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/d324a0cc02881779dcda44a675fdcaaa-Abstract.html)

        **Abstract**:

        Multilingual models are parameter-efficient and especially effective in improving low-resource languages by leveraging crosslingual transfer. Despite recent advance in massive multilingual translation with ever-growing model and data, how to effectively train multilingual models has not been well understood. In this paper, we show that a common situation in multilingual training, data imbalance among languages, poses optimization tension between high resource and low resource languages where the found multilingual solution is often sub-optimal for low resources. We show that common training method which upsamples low resources can not robustly optimize population loss with risks of either underfitting high resource languages or overfitting low resource ones. Drawing on recent findings on the geometry of loss landscape and its effect on generalization, we propose a principled optimization algorithm, Curvature Aware Task Scaling (CATS), which adaptively rescales gradients from different tasks with a meta objective of guiding multilingual training to low-curvature neighborhoods with uniformly low loss for all languages. We ran experiments on common benchmarks (TED, WMT and OPUS-100) with varying degrees of data imbalance. CATS effectively improved multilingual optimization and as a result demonstrated consistent gains on low resources ($+0.8$ to $+2.2$ BLEU) without hurting high resources. In addition, CATS is robust to overparameterization and large batch size training, making it a promising training method for massive multilingual models that truly improve low resource languages.

        ----

        ## [1921] A/B/n Testing with Control in the Presence of Subpopulations

        **Authors**: *Yoan Russac, Christina Katsimerou, Dennis Bohle, Olivier Cappé, Aurélien Garivier, Wouter M. Koolen*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/d35a29602005cb55aa57a5f683c8e0c2-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/d35a29602005cb55aa57a5f683c8e0c2-Abstract.html)

        **Abstract**:

        Motivated by A/B/n testing applications, we consider a finite set of distributions (called \emph{arms}), one of which is treated as a \emph{control}. We assume that the population is stratified into homogeneous subpopulations. At every time step, a subpopulation is sampled and an arm is chosen: the resulting observation is an independent draw from the arm conditioned on the subpopulation. The quality of each arm is assessed through a weighted combination of its subpopulation means. We propose a strategy for sequentially choosing one arm per time step so as to discover as fast as possible which arms, if any, have higher weighted expectation than the control. This strategy is shown to be asymptotically optimal in the following sense: if $\tau_\delta$ is the first time when the strategy ensures that it is able to output the correct answer with probability at least $1-\delta$, then $\mathbb{E}[\tau_\delta]$ grows linearly with $\log(1/\delta)$ at the exact optimal rate. This rate is identified in the paper in three different settings: (1) when the experimenter does not observe the subpopulation information, (2) when the subpopulation of each sample is observed but not chosen, and (3) when the experimenter can select the subpopulation from which each response is sampled. We illustrate the efficiency of the proposed strategy with numerical simulations on synthetic and real data collected from an A/B/n experiment.

        ----

        ## [1922] Using Random Effects to Account for High-Cardinality Categorical Features and Repeated Measures in Deep Neural Networks

        **Authors**: *Giora Simchoni, Saharon Rosset*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/d35b05a832e2bb91f110d54e34e2da79-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/d35b05a832e2bb91f110d54e34e2da79-Abstract.html)

        **Abstract**:

        High-cardinality categorical features are a major challenge for machine learning methods in general and for deep learning in particular. Existing solutions such as one-hot encoding and entity embeddings can be hard to scale when the cardinality is very high, require much space, are hard to interpret or may overfit the data. A special scenario of interest is that of repeated measures, where the categorical feature is the identity of the individual or object, and each object is measured several times, possibly under different conditions (values of the other features). We propose accounting for high-cardinality categorical features as random effects variables in a regression setting, and consequently adopt the corresponding negative log likelihood loss from the linear mixed models (LMM) statistical literature and integrate it in a deep learning framework. We test our model which we call LMMNN on simulated as well as real datasets with a single categorical feature with high cardinality, using various baseline neural networks architectures such as convolutional networks and LSTM, and various applications in e-commerce, healthcare and computer vision. Our results show that treating high-cardinality categorical features as random effects leads to a significant improvement in prediction performance compared to state of the art alternatives. Potential extensions such as accounting for multiple categorical features and classification settings are discussed. Our code and simulations are available at https://github.com/gsimchoni/lmmnn.

        ----

        ## [1923] Learning Debiased Representation via Disentangled Feature Augmentation

        **Authors**: *Jungsoo Lee, Eungyeup Kim, Juyoung Lee, Jihyeon Lee, Jaegul Choo*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/d360a502598a4b64b936683b44a5523a-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/d360a502598a4b64b936683b44a5523a-Abstract.html)

        **Abstract**:

        Image classification models tend to make decisions based on peripheral attributes of data items that have strong correlation with a target variable (i.e., dataset bias). These biased models suffer from the poor generalization capability when evaluated on unbiased datasets. Existing approaches for debiasing often identify and emphasize those samples with no such correlation (i.e., bias-conflicting) without defining the bias type in advance. However, such bias-conflicting samples are significantly scarce in biased datasets, limiting the debiasing capability of these approaches. This paper first presents an empirical analysis revealing that training with "diverse" bias-conflicting samples beyond a given training set is crucial for debiasing as well as the generalization capability. Based on this observation, we propose a novel feature-level data augmentation technique in order to synthesize diverse bias-conflicting samples.  To this end, our method learns the disentangled representation of (1) the intrinsic attributes (i.e., those inherently defining a certain class) and (2) bias attributes (i.e., peripheral attributes causing the bias), from a large number of bias-aligned samples, the bias attributes of which have strong correlation with the target variable.  Using the disentangled representation, we synthesize bias-conflicting samples that contain the diverse intrinsic attributes of bias-aligned samples by swapping their latent features. By utilizing these diversified bias-conflicting features during the training, our approach achieves superior classification accuracy and debiasing results against the existing baselines on both synthetic and real-world datasets.

        ----

        ## [1924] Scallop: From Probabilistic Deductive Databases to Scalable Differentiable Reasoning

        **Authors**: *Jiani Huang, Ziyang Li, Binghong Chen, Karan Samel, Mayur Naik, Le Song, Xujie Si*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/d367eef13f90793bd8121e2f675f0dc2-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/d367eef13f90793bd8121e2f675f0dc2-Abstract.html)

        **Abstract**:

        Deep learning and symbolic reasoning are complementary techniques for an intelligent system. However, principled combinations of these techniques have limited scalability, rendering them ill-suited for real-world applications. We propose Scallop, a system that builds upon probabilistic deductive databases, to bridge this gap. The key insight underlying Scallop is a provenance framework that introduces a tunable parameter to specify the level of reasoning granularity. Scallop thereby i) generalizes exact probabilistic reasoning, ii) asymptotically reduces computational cost, and iii) provides relative accuracy guarantees. On a suite of tasks that involve mathematical and logical reasoning, Scallop scales significantly better without sacrificing accuracy compared to DeepProbLog, a principled neural logic programming approach. We also create and evaluate on a real-world Visual Question Answering (VQA) benchmark that requires multi-hop reasoning. Scallop outperforms two VQA-tailored models, a Neural Module Networks based and a transformer based model, by 12.42% and 21.66% respectively.

        ----

        ## [1925] Learning to Synthesize Programs as Interpretable and Generalizable Policies

        **Authors**: *Dweep Trivedi, Jesse Zhang, Shao-Hua Sun, Joseph J. Lim*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/d37124c4c79f357cb02c655671a432fa-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/d37124c4c79f357cb02c655671a432fa-Abstract.html)

        **Abstract**:

        Recently, deep reinforcement learning (DRL) methods have achieved impressive performance on tasks in a variety of domains. However, neural network policies produced with DRL methods are not human-interpretable and often have difficulty generalizing to novel scenarios. To address these issues, prior works explore learning programmatic policies that are more interpretable and structured for generalization. Yet, these works either employ limited policy representations (e.g. decision trees, state machines, or predefined program templates) or require stronger supervision (e.g. input/output state pairs or expert demonstrations). We present a framework that instead learns to synthesize a program, which details the procedure to solve a task in a flexible and expressive manner, solely from reward signals. To alleviate the difficulty of learning to compose programs to induce the desired agent behavior from scratch, we propose to first learn a program embedding space that continuously parameterizes diverse behaviors in an unsupervised manner and then search over the learned program embedding space to yield a program that maximizes the return for a given task. Experimental results demonstrate that the proposed framework not only learns to reliably synthesize task-solving programs but also outperforms DRL and program synthesis baselines while producing interpretable and more generalizable policies. We also justify the necessity of the proposed two-stage learning scheme as well as analyze various methods for learning the program embedding. Website at https://clvrai.com/leaps.

        ----

        ## [1926] The functional specialization of visual cortex emerges from training parallel pathways with self-supervised predictive learning

        **Authors**: *Shahab Bakhtiari, Patrick J. Mineault, Timothy P. Lillicrap, Christopher C. Pack, Blake A. Richards*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/d384dec9f5f7a64a36b5c8f03b8a6d92-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/d384dec9f5f7a64a36b5c8f03b8a6d92-Abstract.html)

        **Abstract**:

        The visual system of mammals is comprised of parallel, hierarchical specialized pathways. Different pathways are specialized in so far as they use representations that are more suitable for supporting specific downstream behaviours. In particular, the clearest example is the specialization of the ventral ("what") and dorsal ("where") pathways of the visual cortex. These two pathways support behaviours related to visual recognition and movement, respectively. To-date, deep neural networks have mostly been used as models of the ventral, recognition pathway. However, it is unknown whether both pathways can be modelled with a single deep ANN. Here, we ask whether a single model with a single loss function can capture the properties of both the ventral and the dorsal pathways. We explore this question using data from mice, who like other mammals, have specialized pathways that appear to support recognition and movement behaviours. We show that when we train a deep neural network architecture with two parallel pathways using a self-supervised predictive loss function, we can outperform other models in fitting mouse visual cortex. Moreover, we can model both the dorsal and ventral pathways. These results demonstrate that a self-supervised predictive learning approach applied to parallel pathway architectures can account for some of the functional specialization seen in mammalian visual systems.

        ----

        ## [1927] Adversarial Training Helps Transfer Learning via Better Representations

        **Authors**: *Zhun Deng, Linjun Zhang, Kailas Vodrahalli, Kenji Kawaguchi, James Y. Zou*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/d3aeec875c479e55d1cdeea161842ec6-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/d3aeec875c479e55d1cdeea161842ec6-Abstract.html)

        **Abstract**:

        Transfer learning aims to leverage models pre-trained on source data to efficiently adapt to target setting, where only limited data are available for model fine-tuning. Recent works empirically demonstrate that adversarial training in the source data can improve the ability of models to transfer to new domains. However, why this happens is not known. In this paper, we provide a theoretical model to rigorously analyze how adversarial training helps transfer learning. We show that adversarial training in the source data generates provably better representations, so fine-tuning on top of this representation leads to a more accurate predictor of the target data.  We further demonstrate both theoretically and empirically that semi-supervised learning in the source data can also improve transfer learning by similarly improving the representation. Moreover, performing adversarial training on top of semi-supervised learning can further improve transferability, suggesting that the two approaches have complementary benefits on representations.  We support our theories with experiments on popular data sets and deep learning architectures.

        ----

        ## [1928] Improving Coherence and Consistency in Neural Sequence Models with Dual-System, Neuro-Symbolic Reasoning

        **Authors**: *Maxwell I. Nye, Michael Henry Tessler, Joshua B. Tenenbaum, Brenden M. Lake*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/d3e2e8f631bd9336ed25b8162aef8782-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/d3e2e8f631bd9336ed25b8162aef8782-Abstract.html)

        **Abstract**:

        Human reasoning can be understood as an interplay between two systems: the intuitive and associative ("System 1") and the deliberative and logical ("System 2"). Neural sequence models---which have been increasingly successful at performing complex, structured tasks---exhibit the advantages and failure modes of System 1: they are fast and learn patterns from data, but are often inconsistent and incoherent. In this work, we seek a lightweight, training-free means of improving existing System 1-like sequence models by adding System 2-inspired logical reasoning. We explore several variations on this theme in which candidate generations from a neural sequence model are examined for logical consistency by a symbolic reasoning module, which can either accept or reject the generations. Our approach uses neural inference to mediate between the neural System 1 and the logical System 2. Results in robust story generation and grounded instruction-following show that this approach can increase the coherence and accuracy of neurally-based generations.

        ----

        ## [1929] Learning the optimal Tikhonov regularizer for inverse problems

        **Authors**: *Giovanni S. Alberti, Ernesto De Vito, Matti Lassas, Luca Ratti, Matteo Santacesaria*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/d3e6cd9f66f2c1d3840ade4161cf7406-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/d3e6cd9f66f2c1d3840ade4161cf7406-Abstract.html)

        **Abstract**:

        In this work, we consider the linear inverse problem $y=Ax+\varepsilon$, where $A\colon X\to Y$ is a known linear operator between the separable Hilbert spaces $X$ and $Y$, $x$ is a random variable in $X$ and $\epsilon$ is a zero-mean random process in $Y$. This setting covers several inverse problems in imaging including denoising, deblurring, and X-ray tomography. Within the classical framework of regularization, we focus on the case where the regularization functional is not given a priori, but learned from data. Our first result is a characterization of the optimal generalized Tikhonov regularizer, with respect to the mean squared error. We find that it is completely independent of the forward operator $A$ and depends only on the mean and covariance of $x$.Then, we consider the problem of learning the regularizer from a finite training set in two different frameworks: one supervised, based on samples of both $x$ and $y$, and one unsupervised, based only on samples of $x$. In both cases, we prove generalization bounds, under some weak assumptions on the distribution of $x$ and $\varepsilon$, including the case of sub-Gaussian variables. Our bounds hold in infinite-dimensional spaces, thereby showing that finer and finer discretizations do not make this learning problem harder. The results are validated through numerical simulations.

        ----

        ## [1930] NovelD: A Simple yet Effective Exploration Criterion

        **Authors**: *Tianjun Zhang, Huazhe Xu, Xiaolong Wang, Yi Wu, Kurt Keutzer, Joseph E. Gonzalez, Yuandong Tian*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/d428d070622e0f4363fceae11f4a3576-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/d428d070622e0f4363fceae11f4a3576-Abstract.html)

        **Abstract**:

        Efficient exploration under sparse rewards remains a key challenge in deep reinforcement learning. Previous exploration methods (e.g., RND) have achieved strong results in multiple hard tasks. However, if there are multiple novel areas to explore, these methods often focus quickly on one without sufficiently trying others (like a depth-wise first search manner). In some scenarios (e.g., four corridor environment in Sec 4.2), we observe they explore in one corridor for long and fail to cover all the states. On the other hand, in theoretical RL, with optimistic initialization and the inverse square root of visitation count as a bonus, it won't suffer from this and explores different novel regions alternatively (like a breadth-first search manner). In this paper, inspired by this, we propose a simple but effective criterion called NovelD by weighting every novel area approximately equally. Our algorithm is very simple but yet shows comparable performance or even outperforms multiple SOTA exploration methods in many hard exploration tasks. Specifically, NovelD solves all the static procedurally-generated tasks in Mini-Grid with just 120M environment steps, without any curriculum learning. In comparison, the previous SOTA only solves 50% of them. NovelD also achieves SOTA on multiple tasks in NetHack, a rogue-like game that contains more challenging procedurally-generated environments. In multiple Atari games (e.g., MonteZuma's Revenge, Venture, Gravitar), NovelD outperforms RND. We analyze NovelD thoroughly in MiniGrid and found that empirically it helps the agent explore the environment more uniformly with a focus on exploring beyond the boundary.

        ----

        ## [1931] On Margin-Based Cluster Recovery with Oracle Queries

        **Authors**: *Marco Bressan, Nicolò Cesa-Bianchi, Silvio Lattanzi, Andrea Paudice*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/d46e1fcf4c07ce4a69ee07e4134bcef1-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/d46e1fcf4c07ce4a69ee07e4134bcef1-Abstract.html)

        **Abstract**:

        We study an active cluster recovery problem where, given a set of $n$ points and an oracle answering queries like ``are these two points in the same cluster?'', the task is to recover exactly all clusters using as few queries as possible. We begin by introducing a simple but general notion of margin between clusters that captures, as special cases, the margins used in previous works, the classic SVM margin, and standard notions of stability for center-based clusterings. Under our margin assumptions we design algorithms that, in a variety of settings, recover all clusters exactly using only $O(\log n)$ queries. For $\mathbb{R}^m$, we give an algorithm that recovers \emph{arbitrary} convex clusters, in polynomial time, and with a number of queries that is lower than the best existing algorithm by $\Theta(m^m)$ factors. For general pseudometric spaces, where clusters might not be convex or might not have any notion of shape, we give an algorithm that achieves the $O(\log n)$ query bound, and is provably near-optimal as a function of the packing number of the space. Finally, for clusterings realized by binary concept classes, we give a combinatorial characterization of recoverability with $O(\log n)$ queries, and we show that, for many concept classes in $\mathbb{R}^m$, this characterization is equivalent to our margin condition. Our results show a deep connection between cluster margins and active cluster recoverability.

        ----

        ## [1932] Multi-Scale Representation Learning on Proteins

        **Authors**: *Vignesh Ram Somnath, Charlotte Bunne, Andreas Krause*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/d494020ff8ec181ef98ed97ac3f25453-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/d494020ff8ec181ef98ed97ac3f25453-Abstract.html)

        **Abstract**:

        Proteins are fundamental biological entities mediating key roles in cellular function and disease. This paper introduces a multi-scale graph construction of a protein –HoloProt– connecting surface to structure and sequence. The surface captures coarser details of the protein, while sequence as primary component and structure –comprising secondary and tertiary components– capture finer details. Our graph encoder then learns a multi-scale representation by allowing each level to integrate the encoding from level(s) below with the graph at that level. We test the learned representation on different tasks, (i.) ligand binding affinity (regression), and (ii.) protein function prediction (classification).On the regression task, contrary to previous methods, our model performs consistently and reliably across different dataset splits, outperforming all baselines on most splits. On the classification task, it achieves a performance close to the top-performing model while using 10x fewer parameters. To improve the memory efficiency of our construction, we segment the multiplex protein surface manifold into molecular superpixels and substitute the surface with these superpixels at little to no performance loss.

        ----

        ## [1933] Sparse Quadratic Optimisation over the Stiefel Manifold with Application to Permutation Synchronisation

        **Authors**: *Florian Bernard, Daniel Cremers, Johan Thunberg*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/d4bad256c73a6b25b86cc9c1a77255b1-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/d4bad256c73a6b25b86cc9c1a77255b1-Abstract.html)

        **Abstract**:

        We address the non-convex optimisation problem of finding a sparse matrix on the Stiefel manifold (matrices with mutually orthogonal columns of unit length) that maximises (or minimises) a quadratic objective function. Optimisation problems on the Stiefel manifold occur for example in spectral relaxations of various combinatorial problems, such as graph matching, clustering, or permutation synchronisation. Although sparsity is a desirable property in such settings, it is mostly neglected in spectral formulations since existing solvers, e.g. based on eigenvalue decomposition, are unable to account for sparsity while at the same time maintaining global optimality guarantees. We fill this gap and propose a simple yet effective sparsity-promoting modification of the Orthogonal Iteration algorithm for finding the dominant eigenspace of a matrix. By doing so, we can guarantee that our method finds a Stiefel matrix that is globally optimal with respect to the quadratic objective function, while in addition being sparse. As a motivating application we consider the task of permutation synchronisation, which can be understood as a constrained clustering problem that has particular relevance for matching multiple images or 3D shapes in computer vision, computer graphics, and beyond. We demonstrate that the proposed approach outperforms previous methods in this domain.

        ----

        ## [1934] Second-Order Neural ODE Optimizer

        **Authors**: *Guan-Horng Liu, Tianrong Chen, Evangelos A. Theodorou*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/d4c2e4a3297fe25a71d030b67eb83bfc-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/d4c2e4a3297fe25a71d030b67eb83bfc-Abstract.html)

        **Abstract**:

        We propose a novel second-order optimization framework for training the emerging deep continuous-time models, specifically the Neural Ordinary Differential Equations (Neural ODEs). Since their training already involves expensive gradient computation by solving a backward ODE, deriving efficient second-order methods becomes highly nontrivial. Nevertheless, inspired by the recent Optimal Control (OC) interpretation of training deep networks, we show that a specific continuous-time OC methodology, called Differential Programming, can be adopted to derive backward ODEs for higher-order derivatives at the same O(1) memory cost. We further explore a low-rank representation of the second-order derivatives and show that it leads to efficient preconditioned updates with the aid of Kronecker-based factorization. The resulting method – named SNOpt – converges much faster than first-order baselines in wall-clock time, and the improvement remains consistent across various applications, e.g. image classification, generative flow, and time-series prediction. Our framework also enables direct architecture optimization, such as the integration time of Neural ODEs, with second-order feedback policies, strengthening the OC perspective as a principled tool of analyzing optimization in deep learning. Our code is available at https://github.com/ghliu/snopt.

        ----

        ## [1935] Graph Neural Networks with Local Graph Parameters

        **Authors**: *Pablo Barceló, Floris Geerts, Juan L. Reutter, Maksimilian Ryschkov*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/d4d8d1ac7e00e9105775a6b660dd3cbb-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/d4d8d1ac7e00e9105775a6b660dd3cbb-Abstract.html)

        **Abstract**:

        Various recent proposals increase the distinguishing power of Graph Neural Networks (GNNs) by propagating features between k-tuples of vertices. The distinguishing power of these “higher-order” GNNs is known to be bounded by the k-dimensional Weisfeiler-Leman (WL) test, yet their O(n^k) memory requirements limit their applicability. Other proposals infuse GNNs with local higher-order graph structural information from the start, hereby inheriting the desirable O(n) memory requirement from GNNs at the cost of a one-time, possibly non-linear, preprocessing step. We propose local graph parameter enabled GNNs as a framework for studying the latter kind of approaches and precisely characterize their distinguishing power, in terms of a variant of the WL test, and in terms of the graph structural properties that they can take into account. Local graph parameters can be added to any GNN architecture, and are cheap to compute. In terms of expressive power, our proposal lies in the middle of GNNs and their higher-order counterparts. Further, we propose several techniques to aide in choosing the right local graph parameters. Our results connect GNNs with deep results in finite model theory and finite variable logics. Our experimental evaluation shows that adding local graph parameters often has a positive effect for a variety of GNNs, datasets and graph learning tasks.

        ----

        ## [1936] Closing the Gap: Tighter Analysis of Alternating Stochastic Gradient Methods for Bilevel Problems

        **Authors**: *Tianyi Chen, Yuejiao Sun, Wotao Yin*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/d4dd111a4fd973394238aca5c05bebe3-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/d4dd111a4fd973394238aca5c05bebe3-Abstract.html)

        **Abstract**:

        Stochastic nested optimization, including stochastic compositional, min-max, and bilevel optimization, is gaining popularity in many machine learning applications. While the three problems share a nested structure, existing works often treat them separately, thus developing problem-specific algorithms and analyses. Among various exciting developments, simple SGD-type updates (potentially on multiple variables) are still prevalent in solving this class of nested problems, but they are believed to have a slower convergence rate than non-nested problems. This paper unifies several SGD-type updates for stochastic nested problems into a single SGD approach that we term ALternating Stochastic gradient dEscenT (ALSET) method. By leveraging the hidden smoothness of the problem, this paper presents a tighter analysis of ALSET for stochastic nested problems. Under the new analysis, to achieve an $\epsilon$-stationary point of the nested problem, it requires ${\cal O}(\epsilon^{-2})$ samples in total. Under certain regularity conditions, applying our results to stochastic compositional, min-max, and reinforcement learning problems either improves or matches the best-known sample complexity in the respective cases. Our results explain why simple SGD-type algorithms in stochastic nested problems all work very well in practice without the need for further modifications.

        ----

        ## [1937] Dense Unsupervised Learning for Video Segmentation

        **Authors**: *Nikita Araslanov, Simone Schaub-Meyer, Stefan Roth*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/d516b13671a4179d9b7b458a6ebdeb92-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/d516b13671a4179d9b7b458a6ebdeb92-Abstract.html)

        **Abstract**:

        We present a novel approach to unsupervised learning for video object segmentation (VOS). Unlike previous work, our formulation allows to learn dense feature representations directly in a fully convolutional regime. We rely on uniform grid sampling to extract a set of anchors and train our model to disambiguate between them on both inter- and intra-video levels. However, a naive scheme to train such a model results in a degenerate solution. We propose to prevent this with a simple regularisation scheme, accommodating the equivariance property of the segmentation task to similarity transformations. Our training objective admits efficient implementation and exhibits fast training convergence. On established VOS benchmarks, our approach exceeds the segmentation accuracy of previous work despite using significantly less training data and compute power.

        ----

        ## [1938] Charting and Navigating the Space of Solutions for Recurrent Neural Networks

        **Authors**: *Elia Turner, Kabir V. Dabholkar, Omri Barak*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/d530d454337fb09964237fecb4bea6ce-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/d530d454337fb09964237fecb4bea6ce-Abstract.html)

        **Abstract**:

        In recent years Recurrent Neural Networks (RNNs) were successfully used to model the way neural activity drives task-related behavior in animals, operating under the implicit assumption that the obtained solutions are universal. Observations in both neuroscience and machine learning challenge this assumption. Animals can approach a given task with a variety of strategies, and training machine learning algorithms introduces the phenomenon of underspecification. These observations imply that every task is associated with a space of solutions. To date, the structure of this space is not understood, limiting the approach of comparing RNNs with neural data.Here, we characterize the space of solutions associated with various tasks. We first study a simple two-neuron network on a task that leads to multiple solutions. We trace the nature of the final solution back to the networkâ€™s initial connectivity and identify discrete dynamical regimes that underlie this diversity. We then examine three neuroscience-inspired tasks: Delayed discrimination, Interval discrimination, and Time reproduction. For each task, we find a rich set of solutions. One layer of variability can be found directly in the neural activity of the networks. An additional layer is uncovered by testing the trained networks' ability to extrapolate, as a perturbation to a system often reveals hidden structure. Furthermore, we relate extrapolation patterns to specific dynamical objects and effective algorithms found by the networks. We introduce a tool to derive the reduced dynamics of networks by generating a compact directed graph describing the essence of the dynamics with regards to behavioral inputs and outputs. Using this representation, we can partition the solutions to each task into a handful of types and show that neural features can partially predict them.Taken together, our results shed light on the concept of the space of solutions and its uses both in Machine learning and in Neuroscience.

        ----

        ## [1939] Fast Training Method for Stochastic Compositional Optimization Problems

        **Authors**: *Hongchang Gao, Heng Huang*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/d5397f1497b5cdaad7253fdc92db610b-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/d5397f1497b5cdaad7253fdc92db610b-Abstract.html)

        **Abstract**:

        The stochastic compositional optimization problem covers a wide range of machine learning models, such as sparse additive models and model-agnostic meta-learning.  Thus, it is necessary to develop efficient methods for its optimization. Existing methods for the stochastic compositional optimization problem only focus on the single machine scenario, which is far from satisfactory when data are distributed on different devices. To address this problem, we propose novel decentralized stochastic compositional gradient descent methods to efficiently train the large-scale stochastic compositional optimization problem. To the best of our knowledge, our work is the first one facilitating decentralized training for this kind of problem. Furthermore, we provide the convergence analysis for our methods, which shows that the convergence rate of our methods can achieve linear speedup with respect to the number of devices. At last, we apply our decentralized training methods to the model-agnostic meta-learning problem, and the experimental results confirm the superior performance of our methods.

        ----

        ## [1940] Dual-stream Network for Visual Recognition

        **Authors**: *Mingyuan Mao, Peng Gao, Renrui Zhang, Honghui Zheng, Teli Ma, Yan Peng, Errui Ding, Baochang Zhang, Shumin Han*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/d56b9fc4b0f1be8871f5e1c40c0067e7-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/d56b9fc4b0f1be8871f5e1c40c0067e7-Abstract.html)

        **Abstract**:

        Transformers with remarkable global representation capacities achieve competitive results for visual tasks, but fail to consider high-level local pattern information in input images. In this paper, we present a generic Dual-stream Network  (DS-Net) to fully explore the representation capacity of local and global pattern features for image classification.  Our DS-Net can simultaneously  calculate fine-grained and integrated features and efficiently fuse them. Specifically,  we propose an Intra-scale Propagation module to process two different resolutions in each block and an Inter-Scale Alignment module to perform information interaction across features at dual scales. Besides, we also design a Dual-stream FPN (DS-FPN) to further enhance contextual information for downstream dense predictions. Without bells and whistles, the proposed DS-Net outperforms DeiT-Small by 2.4\% in terms of top-1 accuracy on ImageNet-1k and achieves state-of-the-art performance over other Vision Transformers and ResNets. For object detection and instance segmentation, DS-Net-Small respectively outperforms ResNet-50 by 6.4\% and 5.5 \% in terms of mAP on MSCOCO 2017, and surpasses the previous state-of-the-art scheme, which significantly demonstrates its potential to be a general backbone in vision tasks. The code will be released soon.

        ----

        ## [1941] Estimating High Order Gradients of the Data Distribution by Denoising

        **Authors**: *Chenlin Meng, Yang Song, Wenzhe Li, Stefano Ermon*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/d582ac40970f9885836a61d7b2c662e4-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/d582ac40970f9885836a61d7b2c662e4-Abstract.html)

        **Abstract**:

        The first order derivative of a data density can be estimated efficiently by denoising score matching, and has become an important component in many applications, such as image generation and audio synthesis. Higher order derivatives provide additional local information about the data distribution and enable new applications. Although they can be estimated via automatic differentiation of a learned density model, this can amplify estimation errors and is expensive in high dimensional settings. To overcome these limitations, we propose a method to directly estimate high order derivatives (scores) of a data density from samples. We first show that denoising score matching can be interpreted as a particular case of Tweedie’s formula. By leveraging Tweedie’s formula on higher order moments, we generalize denoising score matching to estimate higher order derivatives. We demonstrate empirically that models trained with the proposed method can approximate second order derivatives more efficiently and accurately than via automatic differentiation. We show that our models can be used to quantify uncertainty in denoising and to improve the mixing speed of Langevin dynamics via Ozaki discretization for sampling synthetic data and natural images.

        ----

        ## [1942] Machine versus Human Attention in Deep Reinforcement Learning Tasks

        **Authors**: *Sihang Guo, Ruohan Zhang, Bo Liu, Yifeng Zhu, Dana H. Ballard, Mary M. Hayhoe, Peter Stone*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/d58e2f077670f4de9cd7963c857f2534-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/d58e2f077670f4de9cd7963c857f2534-Abstract.html)

        **Abstract**:

        Deep reinforcement learning (RL) algorithms are powerful tools for solving visuomotor decision tasks. However, the trained models are often difficult to interpret, because they are represented as end-to-end deep neural networks.  In this paper, we shed light on the inner workings of such trained models by analyzing the pixels that they attend to during task execution, and comparing them with the pixels attended to by humans executing the same tasks. To this end, we investigate the following two questions that, to the best of our knowledge, have not been previously studied. 1) How similar are the visual representations learned by RL agents and humans when performing the same task? and, 2) How do similarities and differences in these learned representations explain RL agents' performance on these tasks? Specifically, we compare the saliency maps of RL agents against visual attention models of human experts when learning to play Atari games. Further, we analyze how hyperparameters of the deep RL algorithm affect the learned representations and saliency maps of the trained agents. The insights provided have the potential to inform novel algorithms for closing the performance gap between human experts and RL agents.

        ----

        ## [1943] Reusing Combinatorial Structure: Faster Iterative Projections over Submodular Base Polytopes

        **Authors**: *Jai Moondra, Hassan Mortagy, Swati Gupta*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/d58f36f7679f85784d8b010ff248f898-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/d58f36f7679f85784d8b010ff248f898-Abstract.html)

        **Abstract**:

        Optimization algorithms such as projected Newton's method, FISTA,  mirror descent and its variants enjoy near-optimal regret bounds and convergence rates, but suffer from a computational bottleneck of computing ``projections" in potentially each iteration (e.g., $O(T^{1/2})$ regret of online mirror descent). On the other hand, conditional gradient variants solve a linear optimization in each iteration, but result in suboptimal rates (e.g., $O(T^{3/4})$ regret of online Frank-Wolfe). Motivated by this trade-off in runtime v/s convergence rates, we consider iterative projections of close-by points over widely-prevalent submodular base polytopes $B(f)$. We develop a toolkit to speed up the computation of projections using both discrete and continuous perspectives. We subsequently adapt the away-step Frank-Wolfe algorithm to use this information and enable early termination. For the special case of cardinality based submodular polytopes, we improve the runtime of computing certain Bregman projections by a factor of $\Omega(n/\log(n))$. Our theoretical results show orders of magnitude reduction in runtime in preliminary computational experiments.

        ----

        ## [1944] Constrained Optimization to Train Neural Networks on Critical and Under-Represented Classes

        **Authors**: *Sara Sangalli, Ertunc Erdil, Andreas M. Hötker, Olivio Donati, Ender Konukoglu*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/d5ade38a2c9f6f073d69e1bc6b6e64c1-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/d5ade38a2c9f6f073d69e1bc6b6e64c1-Abstract.html)

        **Abstract**:

        Deep neural networks (DNNs) are notorious for making more mistakes for the classes that have substantially fewer samples than the others during training. Such class imbalance is ubiquitous in clinical applications and very crucial to handle because the classes with fewer samples most often correspond to critical cases (e.g., cancer) where misclassifications can have severe consequences.Not to miss such cases, binary classifiers need to be operated at high True Positive Rates (TPRs) by setting a higher threshold, but this comes at the cost of very high False Positive Rates (FPRs) for problems with class imbalance. Existing methods for learning under class imbalance most often do not take this into account. We argue that prediction accuracy should be improved by emphasizing the reduction of FPRs at high TPRs for problems where misclassification of the positive, i.e. critical, class samples are associated with higher cost.To this end, we pose the training of a DNN for binary classification as a constrained optimization problem and introduce a novel constraint that can be used with existing loss functions to enforce maximal area under the ROC curve (AUC) through prioritizing FPR reduction at high TPR. We solve the resulting constrained optimization problem using an Augmented Lagrangian method (ALM).Going beyond binary, we also propose two possible extensions of the proposed constraint for multi-class classification problems.We present experimental results for image-based binary and multi-class classification applications using an in-house medical imaging dataset, CIFAR10, and CIFAR100. Our results demonstrate that the proposed method improves the baselines in majority of the cases by attaining higher accuracy on critical classes while reducing the misclassification rate for the non-critical class samples.

        ----

        ## [1945] Collapsed Variational Bounds for Bayesian Neural Networks

        **Authors**: *Marcin Tomczak, Siddharth Swaroop, Andrew Y. K. Foong, Richard E. Turner*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/d5b03d3acb580879f82271ab4885ee5e-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/d5b03d3acb580879f82271ab4885ee5e-Abstract.html)

        **Abstract**:

        Recent interest in learning large variational Bayesian Neural Networks (BNNs) has been partly hampered by poor predictive performance caused by underfitting, and their performance is known to be very sensitive to the prior over weights. Current practice often fixes the prior parameters to standard values or tunes them using heuristics or cross-validation. In this paper, we treat prior parameters in a distributional way by extending the model and collapsing the variational bound with respect to their posteriors. This leads to novel and tighter Evidence Lower Bounds (ELBOs) for performing variational inference (VI) in BNNs. Our experiments show that the new bounds significantly improve the performance of Gaussian mean-field VI applied to BNNs on a variety of data sets, demonstrating that mean-field VI works well even in deep models. We also find that the tighter ELBOs can be good optimization targets for learning the hyperparameters of hierarchical priors.

        ----

        ## [1946] Consistent Estimation for PCA and Sparse Regression with Oblivious Outliers

        **Authors**: *Tommaso d'Orsi, Chih-Hung Liu, Rajai Nasser, Gleb Novikov, David Steurer, Stefan Tiegel*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/d5b3d8dadd770c460b1cde910a711987-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/d5b3d8dadd770c460b1cde910a711987-Abstract.html)

        **Abstract**:

        We develop machinery to design efficiently computable and \emph{consistent} estimators, achieving estimation error approaching zero as the number of observations grows, when facing an oblivious adversary that may corrupt responses in all but an $\alpha$ fraction of the samples.As concrete examples, we investigate two problems: sparse regression and principal component analysis (PCA).For sparse regression, we achieve consistency for optimal sample size $n\gtrsim (k\log d)/\alpha^2$ and optimal error rate $O(\sqrt{(k\log d)/(n\cdot \alpha^2)})$where $n$ is the number of observations, $d$ is the number of dimensions and $k$ is the sparsity of the parameter vector, allowing the fraction of inliers to be inverse-polynomial in the number of samples.Prior to this work, no estimator was known to be consistent when the fraction of inliers $\alpha$ is $o(1/\log \log n)$, even for (non-spherical) Gaussian design matrices.Results holding under weak design assumptions and in the presence of such general noise have only been shown in dense setting (i.e., general linear regression) very recently by d'Orsi et al.~\cite{ICML-linear-regression}.In the context of PCA, we attain optimal error guarantees under broad spikiness assumptions on the parameter matrix (usually used in matrix completion). Previous works could obtain non-trivial guarantees only under the assumptions that the measurement noise corresponding to the inliers is polynomially small in $n$ (e.g., Gaussian with variance $1/n^2$).To devise our estimators, we equip the Huber loss with non-smooth regularizers such as the $\ell_1$ norm or the nuclear norm, and extend d'Orsi et al.'s approach~\cite{ICML-linear-regression} in a novel way to analyze the loss function.Our machinery appears to be easily applicable to a wide range of estimation problems.We complement these algorithmic results with statistical lower bounds showing that the fraction of inliers that our PCA estimator can deal with is optimal up to a constant factor.

        ----

        ## [1947] Offline Constrained Multi-Objective Reinforcement Learning via Pessimistic Dual Value Iteration

        **Authors**: *Runzhe Wu, Yufeng Zhang, Zhuoran Yang, Zhaoran Wang*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/d5c8e1ab6fc0bfeb5f29aafa999cdb29-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/d5c8e1ab6fc0bfeb5f29aafa999cdb29-Abstract.html)

        **Abstract**:

        In constrained multi-objective RL, the goal is to learn a policy that achieves the best performance specified by a multi-objective preference function under a constraint. We focus on the offline setting where the RL agent aims to learn the optimal policy from a given dataset. This scenario is common in real-world applications where interactions with the environment are expensive and the constraint violation is dangerous. For such a setting, we transform the original constrained problem into a  primal-dual formulation, which is solved via dual gradient ascent. Moreover, we propose to combine such an approach with pessimism to overcome the uncertainty in offline data, which leads to our Pessimistic Dual Iteration (PEDI). We establish upper bounds on both the suboptimality and constraint violation for the policy learned by PEDI based on an arbitrary dataset, which proves that PEDI is provably sample efficient. We also specialize PEDI to the setting with linear function approximation. To the best of our knowledge, we propose the first provably efficient constrained multi-objective RL algorithm with offline data without any assumption on the coverage of the dataset.

        ----

        ## [1948] Absolute Neighbour Difference based Correlation Test for Detecting Heteroscedastic Relationships

        **Authors**: *Lifeng Zhang*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/d5cfead94f5350c12c322b5b664544c1-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/d5cfead94f5350c12c322b5b664544c1-Abstract.html)

        **Abstract**:

        It is a challenge to detect complicated data relationships thoroughly. Here, we propose a new statistical measure, named the absolute neighbour difference based neighbour correlation coefficient, to detect the associations between variables through examining the heteroscedasticity of the unpredictable variation of dependent variables. Different from previous studies, the new method concentrates on measuring nonfunctional relationships rather than functional or mixed associations. Either used alone or in combination with other measures, it enables not only a convenient test of heteroscedasticity, but also measuring functional and nonfunctional relationships separately that obviously leads to a deeper insight into the data associations. The method is concise and easy to implement that does not rely on explicitly estimating the regression residuals or the dependencies between variables so that it is not restrict to any kind of model assumption. The mechanisms of the correlation test are proved in theory and demonstrated with numerical analyses.

        ----

        ## [1949] Batch Multi-Fidelity Bayesian Optimization with Deep Auto-Regressive Networks

        **Authors**: *Shibo Li, Robert M. Kirby, Shandian Zhe*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/d5e2c0adad503c91f91df240d0cd4e49-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/d5e2c0adad503c91f91df240d0cd4e49-Abstract.html)

        **Abstract**:

        Bayesian optimization (BO) is a powerful approach for optimizing black-box, expensive-to-evaluate functions. To enable a flexible trade-off between the cost and accuracy, many applications allow the function to be evaluated at different fidelities.  In order to reduce the optimization cost while maximizing the benefit-cost ratio,  in this paper we propose Batch Multi-fidelity Bayesian Optimization with Deep Auto-Regressive Networks (BMBO-DARN). We use a set of Bayesian neural networks to construct a fully auto-regressive model, which is expressive enough to capture strong yet complex relationships across all the fidelities, so as to improve the surrogate learning and optimization performance. Furthermore, to enhance the quality and diversity of queries, we develop a simple yet efficient batch querying method, without any combinatorial search over the fidelities. We propose a batch acquisition function based on Max-value Entropy Search (MES) principle, which penalizes highly correlated queries and encourages diversity. We use posterior samples and moment matching to fulfill efficient computation of the acquisition function, and conduct alternating optimization over every fidelity-input pair, which guarantees an improvement at each step.  We demonstrate the advantage of our approach on four real-world  hyperparameter optimization applications.

        ----

        ## [1950] Mastering Atari Games with Limited Data

        **Authors**: *Weirui Ye, Shaohuai Liu, Thanard Kurutach, Pieter Abbeel, Yang Gao*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/d5eca8dc3820cad9fe56a3bafda65ca1-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/d5eca8dc3820cad9fe56a3bafda65ca1-Abstract.html)

        **Abstract**:

        Reinforcement learning has achieved great success in many applications. However, sample efficiency remains a key challenge, with prominent methods requiring millions (or even billions) of environment steps to train.  Recently, there has been significant progress in sample efficient image-based RL algorithms; however, consistent human-level performance on the Atari game benchmark remains an elusive goal. We propose a sample efficient model-based visual RL algorithm built on MuZero, which we name EfficientZero. Our method achieves 194.3% mean human performance and 109.0% median performance on the Atari 100k benchmark with only two hours of real-time game experience and outperforms the state SAC in some tasks on the DMControl 100k benchmark. This is the first time an algorithm achieves super-human performance on Atari games with such little data. EfficientZero's performance is also close to DQN's performance at 200 million frames while we consume 500 times less data. EfficientZero's low sample complexity and high performance can bring RL closer to real-world applicability. We implement our algorithm in an easy-to-understand manner and it is available at https://github.com/YeWR/EfficientZero. We hope it will accelerate the research of MCTS-based RL algorithms in the wider community.

        ----

        ## [1951] Dealing With Misspecification In Fixed-Confidence Linear Top-m Identification

        **Authors**: *Clémence Réda, Andrea Tirinzoni, Rémy Degenne*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/d5fcc35c94879a4afad61cacca56192c-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/d5fcc35c94879a4afad61cacca56192c-Abstract.html)

        **Abstract**:

        We study the problem of the identification of m arms with largest means under a fixed error rate $\delta$ (fixed-confidence Top-m identification), for misspecified linear bandit models. This problem is motivated by practical applications, especially in medicine and recommendation systems, where linear models are popular due to their simplicity and the existence of efficient algorithms, but in which data inevitably deviates from linearity. In this work, we first derive a tractable lower bound on the sample complexity of any $\delta$-correct algorithm for the general Top-m identification problem. We show that knowing the scale of the deviation from linearity is necessary to exploit the structure of the problem. We then describe the first algorithm for this setting, which is both practical and adapts to the amount of misspecification. We derive an upper bound to its sample complexity which confirms this adaptivity and that matches the lower bound when $\delta \rightarrow 0$. Finally, we evaluate our algorithm on both synthetic and real-world data, showing competitive performance with respect to existing baselines.

        ----

        ## [1952] Why Generalization in RL is Difficult: Epistemic POMDPs and Implicit Partial Observability

        **Authors**: *Dibya Ghosh, Jad Rahme, Aviral Kumar, Amy Zhang, Ryan P. Adams, Sergey Levine*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/d5ff135377d39f1de7372c95c74dd962-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/d5ff135377d39f1de7372c95c74dd962-Abstract.html)

        **Abstract**:

        Generalization is a central challenge for the deployment of reinforcement learning (RL) systems in the real world. In this paper, we show that the sequential structure of the RL problem necessitates new approaches to generalization beyond the well-studied techniques used in supervised learning. While supervised learning methods can generalize effectively without explicitly accounting for epistemic uncertainty, we describe why appropriate uncertainty handling can actually be essential in RL. We show that generalization to unseen test conditions from a limited number of training conditions induces a kind of implicit partial observability, effectively turning even fully-observed MDPs into POMDPs. Informed by this observation, we recast the problem of generalization in RL as solving the induced partially observed Markov decision process, which we call the epistemic POMDP. We demonstrate the failure modes of algorithms that do not appropriately handle this partial observability, and suggest a simple ensemble-based technique for approximately solving the partially observed problem. Empirically, we demonstrate that our simple algorithm derived from the epistemic POMDP achieves significant gains in generalization over current methods on the Procgen benchmark suite.

        ----

        ## [1953] Set Prediction in the Latent Space

        **Authors**: *Konpat Preechakul, Chawan Piansaddhayanon, Burin Naowarat, Tirasan Khandhawit, Sira Sriswasdi, Ekapol Chuangsuwanich*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/d61e9e58ae1058322bc169943b39f1d8-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/d61e9e58ae1058322bc169943b39f1d8-Abstract.html)

        **Abstract**:

        Set prediction tasks require the matching between predicted set and ground truth set in order to propagate the gradient signal. Recent works have performed this matching in the original feature space thus requiring predefined distance functions. We propose a method for learning the distance function by performing the matching in the latent space learned from encoding networks. This method enables the use of teacher forcing which was not possible previously since matching in the feature space must be computed after the entire output sequence is generated. Nonetheless, a naive implementation of latent set prediction might not converge due to permutation instability. To address this problem, we provide sufficient conditions for permutation stability which begets an algorithm to improve the overall model convergence. Experiments on several set prediction tasks, including image captioning and object detection, demonstrate the effectiveness of our method.

        ----

        ## [1954] Best of Both Worlds: Practical and Theoretically Optimal Submodular Maximization in Parallel

        **Authors**: *Yixin Chen, Tonmoy Dey, Alan Kuhnle*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/d63fbf8c3173730f82b150c5ef38b8ff-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/d63fbf8c3173730f82b150c5ef38b8ff-Abstract.html)

        **Abstract**:

        For the problem of maximizing a monotone, submodular function with respect to a cardinality constraint $k$ on a ground set of size $n$, we provide an algorithm that achieves the state-of-the-art in both its empirical performance and its theoretical properties, in terms of adaptive complexity, query complexity, and approximation ratio; that is, it obtains, with high probability, query complexity of $O(n)$ in expectation, adaptivity of $O(\log(n))$, and approximation ratio of nearly $1-1/e$. The main algorithm is assembled from two components which may be of independent interest. The first component of our algorithm, LINEARSEQ, is useful as a preprocessing algorithm to improve the query complexity of many algorithms. Moreover, a variant of LINEARSEQ is shown to have adaptive complexity of $O( \log (n / k) )$ which is smaller than that of any previous algorithm in the literature. The second component is a parallelizable thresholding procedure THRESHOLDSEQ for adding elements with gain above a constant threshold. Finally, we demonstrate that our main algorithm empirically outperforms, in terms of runtime, adaptive rounds, total queries, and objective values, the previous state-of-the-art algorithm FAST in a comprehensive evaluation with six submodular objective functions.

        ----

        ## [1955] Fine-grained Generalization Analysis of Inductive Matrix Completion

        **Authors**: *Antoine Ledent, Rodrigo Alves, Yunwen Lei, Marius Kloft*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/d6428eecbe0f7dff83fc607c5044b2b9-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/d6428eecbe0f7dff83fc607c5044b2b9-Abstract.html)

        **Abstract**:

        In this paper, we bridge the gap between the state-of-the-art theoretical results for matrix completion with the nuclear norm and their equivalent in \textit{inductive matrix completion}: (1) In the distribution-free setting, we prove bounds improving the previously best scaling of $O(rd^2)$ to $\widetilde{O}(d^{3/2}\sqrt{r})$, where $d$ is the dimension of the side information and $r$ is the rank. (2) We introduce the (smoothed) \textit{adjusted trace-norm minimization} strategy, an inductive analogue of the weighted trace norm, for which we show guarantees of the order $\widetilde{O}(dr)$ under arbitrary sampling. In the inductive case, a similar rate was previously achieved only under uniform sampling and for exact recovery. Both our results align with the state of the art in the particular case of standard (non-inductive) matrix completion, where they are known to be tight up to log terms. Experiments further confirm that our strategy outperforms standard inductive matrix completion on various synthetic datasets and real problems, justifying its place as an important tool in the arsenal of methods for matrix completion using side information.

        ----

        ## [1956] Learning Frequency Domain Approximation for Binary Neural Networks

        **Authors**: *Yixing Xu, Kai Han, Chang Xu, Yehui Tang, Chunjing Xu, Yunhe Wang*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/d645920e395fedad7bbbed0eca3fe2e0-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/d645920e395fedad7bbbed0eca3fe2e0-Abstract.html)

        **Abstract**:

        Binary neural networks (BNNs) represent original full-precision weights and activations into 1-bit with sign function. Since the gradient of the conventional sign function is almost zero everywhere which cannot be used for back-propagation, several attempts have been proposed to alleviate the optimization difficulty by using approximate gradient. However, those approximations corrupt the main direction of factual gradient. To this end, we propose to estimate the gradient of sign function in the Fourier frequency domain using the combination of sine functions for training BNNs, namely frequency domain approximation (FDA). The proposed approach does not affect the low-frequency information of the original sign function which occupies most of the overall energy, and high-frequency coefficients will be ignored to avoid the huge computational overhead. In addition, we embed a noise adaptation module into the training phase to compensate the approximation error. The experiments on several benchmark datasets and neural architectures illustrate that the binary network learned using our method achieves the state-of-the-art accuracy. Code will be available at https://gitee.com/mindspore/models/tree/master/research/cv/FDA-BNN.

        ----

        ## [1957] Reformulating Zero-shot Action Recognition for Multi-label Actions

        **Authors**: *Alec Kerrigan, Kevin Duarte, Yogesh S. Rawat, Mubarak Shah*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/d6539d3b57159babf6a72e106beb45bd-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/d6539d3b57159babf6a72e106beb45bd-Abstract.html)

        **Abstract**:

        The goal of zero-shot action recognition (ZSAR) is to classify action classes which were not previously seen during training. Traditionally, this is achieved by training a network to map, or regress, visual inputs to a semantic space where a nearest neighbor classifier is used to select the closest target class. We argue that this approach is sub-optimal due to the use of nearest neighbor on static semantic space and is ineffective when faced with multi-label videos - where two semantically distinct co-occurring action categories cannot be predicted with high confidence. To overcome these limitations, we propose a ZSAR framework which does not rely on nearest neighbor classification, but rather consists of a pairwise scoring function. Given a video and a set of action classes, our method predicts a set of confidence scores for each class independently. This allows for the prediction of several semantically distinct classes within one video input. Our evaluations show that our method not only achieves strong performance on three single-label action classification datasets (UCF-101, HMDB, and RareAct), but also outperforms previous ZSAR approaches on a challenging multi-label dataset (AVA) and a real-world surprise activity detection dataset (MEVA).

        ----

        ## [1958] Optimal Best-Arm Identification Methods for Tail-Risk Measures

        **Authors**: *Shubhada Agrawal, Wouter M. Koolen, Sandeep Juneja*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/d69c7ebb6a253532b266151eac6591af-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/d69c7ebb6a253532b266151eac6591af-Abstract.html)

        **Abstract**:

        Conditional value-at-risk (CVaR) and value-at-risk (VaR) are popular tail-risk measures in finance and insurance industries as well as in highly reliable, safety-critical uncertain environments where often the underlying probability distributions are heavy-tailed. We use the multi-armed bandit best-arm identification framework and consider the problem of identifying the arm from amongst finitely many that has the smallest CVaR, VaR, or weighted sum of CVaR and mean. The latter captures the risk-return trade-off common in finance. Our main contribution is an optimal $\delta$-correct algorithm that acts on general arms, including heavy-tailed distributions, and matches the lower bound on the expected number of samples needed, asymptotically (as $ \delta$ approaches $0$). The algorithm requires solving a non-convex optimization problem in the space of probability measures, that requires delicate analysis. En-route, we develop new non-asymptotic, anytime-valid, empirical-likelihood-based concentration inequalities for tail-risk measures.

        ----

        ## [1959] SyMetric: Measuring the Quality of Learnt Hamiltonian Dynamics Inferred from Vision

        **Authors**: *Irina Higgins, Peter Wirnsberger, Andrew Jaegle, Aleksandar Botev*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/d6ef5f7fa914c19931a55bb262ec879c-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/d6ef5f7fa914c19931a55bb262ec879c-Abstract.html)

        **Abstract**:

        A recently proposed class of models attempts to learn latent dynamics from high-dimensional observations, like images, using priors informed by Hamiltonian mechanics. While these models have important potential applications in areas like robotics or autonomous driving, there is currently no good way to evaluate their performance: existing methods primarily rely on image reconstruction quality, which does not always reflect the quality of the learnt latent dynamics. In this work, we empirically highlight the problems with the existing measures and develop a set of new measures, including a binary indicator of whether the underlying Hamiltonian dynamics have been faithfully captured, which we call Symplecticity Metric or SyMetric. Our measures take advantage of the known properties of Hamiltonian dynamics and are more discriminative of the model's ability to capture the underlying dynamics than reconstruction error. Using SyMetric, we identify a set of architectural choices that significantly improve the performance of a previously proposed model for inferring latent dynamics from pixels, the Hamiltonian Generative Network (HGN). Unlike the original HGN, the new SyMetric is able to discover an interpretable phase space with physically meaningful latents on some datasets. Furthermore, it is stable for significantly longer rollouts on a diverse range of 13 datasets,  producing rollouts of essentially infinite length both forward and backwards in time with no degradation in quality on a subset of the datasets.

        ----

        ## [1960] Learning with Holographic Reduced Representations

        **Authors**: *Ashwinkumar Ganesan, Hang Gao, Sunil Gandhi, Edward Raff, Tim Oates, James Holt, Mark McLean*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/d71dd235287466052f1630f31bde7932-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/d71dd235287466052f1630f31bde7932-Abstract.html)

        **Abstract**:

        Holographic Reduced Representations (HRR) are a method for performing symbolic AI on top of real-valued vectors by associating each vector with an abstract concept, and providing mathematical operations to manipulate vectors as if they were classic symbolic objects. This method has seen little use outside of older symbolic AI work and cognitive science. Our goal is to revisit this approach to understand if it is viable for enabling a hybrid neural-symbolic  approach to learning as a differential component of a deep learning architecture. HRRs today are not effective in a differential solution due to numerical instability, a problem we solve by introducing a projection step that forces the vectors to exist in a well behaved point in space. In doing so we improve the concept retrieval efficacy of HRRs by over $100\times$. Using multi-label classification we demonstrate how to leverage the symbolic HRR properties to develop a output layer and loss function that is able to learn effectively, and allows us to investigate some of the pros and cons of an HRR neuro-symbolic learning approach.

        ----

        ## [1961] Learning Barrier Certificates: Towards Safe Reinforcement Learning with Zero Training-time Violations

        **Authors**: *Yuping Luo, Tengyu Ma*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/d71fa38b648d86602d14ac610f2e6194-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/d71fa38b648d86602d14ac610f2e6194-Abstract.html)

        **Abstract**:

        Training-time safety violations have been a major concern when we deploy reinforcement learning algorithms in the real world.This paper explores the possibility of safe RL algorithms with zero training-time safety violations in the challenging setting where we are only given a safe but trivial-reward initial policy without any prior knowledge of the dynamics and additional offline data.We propose an algorithm, Co-trained Barrier Certificate for Safe RL (CRABS), which iteratively learns barrier certificates, dynamics models, and policies. The barrier certificates are learned via adversarial training and ensure the policy's safety assuming calibrated learned dynamics. We also add a regularization term to encourage larger certified regions to enable better exploration. Empirical simulations show that zero safety violations are already challenging for a suite of simple environments with only 2-4 dimensional state space, especially if high-reward policies have to visit regions near the safety boundary.  Prior methods require hundreds of violations to achieve decent rewards on these tasks,  whereas our proposed algorithms incur zero violations.

        ----

        ## [1962] On the Second-order Convergence Properties of Random Search Methods

        **Authors**: *Aurélien Lucchi, Antonio Orvieto, Adamos Solomou*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/d757719ed7c2b66dd17dcee2a3cb29f4-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/d757719ed7c2b66dd17dcee2a3cb29f4-Abstract.html)

        **Abstract**:

        We study the theoretical convergence properties of random-search methods when optimizing non-convex objective functions without having access to derivatives. We prove that standard random-search methods that do not rely on second-order information converge to a second-order stationary point. However, they suffer from an exponential complexity in terms of the input dimension of the problem. In order to address this issue, we propose a novel variant of random search that exploits negative curvature by only relying on function evaluations. We prove that this approach converges to a second-order stationary point at a much faster rate than vanilla methods: namely, the complexity in terms of the number of function evaluations is only linear in the problem dimension. We test our algorithm empirically and find good agreements with our theoretical results.

        ----

        ## [1963] Noether's Learning Dynamics: Role of Symmetry Breaking in Neural Networks

        **Authors**: *Hidenori Tanaka, Daniel Kunin*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/d76d8deea9c19cc9aaf2237d2bf2f785-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/d76d8deea9c19cc9aaf2237d2bf2f785-Abstract.html)

        **Abstract**:

        In nature, symmetry governs regularities, while symmetry breaking brings texture. In artificial neural networks, symmetry has been a central design principle to efficiently capture regularities in the world, but the role of symmetry breaking is not well understood. Here, we develop a theoretical framework to study the "geometry of learning dynamics" in neural networks, and reveal a key mechanism of explicit symmetry breaking behind the efficiency and stability of modern neural networks. To build this understanding, we model the discrete learning dynamics of gradient descent using a continuous-time Lagrangian formulation, in which the learning rule corresponds to the kinetic energy and the loss function corresponds to the potential energy. Then, we identify "kinetic symmetry breaking" (KSB), the condition when the kinetic energy explicitly breaks the symmetry of the potential function. We generalize Noether’s theorem known in physics to take into account KSB and derive the resulting motion of the Noether charge: "Noether's Learning Dynamics" (NLD). Finally, we apply NLD to neural networks with normalization layers and reveal how KSB introduces a mechanism of implicit adaptive optimization, establishing an analogy between learning dynamics induced by normalization layers and RMSProp. Overall, through the lens of Lagrangian mechanics, we have established a theoretical foundation to discover geometric design principles for the learning dynamics of neural networks.

        ----

        ## [1964] A Theory of the Distortion-Perception Tradeoff in Wasserstein Space

        **Authors**: *Dror Freirich, Tomer Michaeli, Ron Meir*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/d77e68596c15c53c2a33ad143739902d-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/d77e68596c15c53c2a33ad143739902d-Abstract.html)

        **Abstract**:

        The lower the distortion of an estimator, the more the distribution of its outputs generally deviates from the distribution of the signals it attempts to estimate. This phenomenon, known as the perception-distortion tradeoff, has captured significant attention in image restoration, where it implies that fidelity to ground truth images comes on the expense of perceptual quality (deviation from statistics of natural images). However, despite the increasing popularity of performing comparisons on the perception-distortion plane, there remains an important open question: what is the minimal distortion that can be achieved under a given perception constraint? In this paper, we derive a closed form expression for this distortion-perception (DP) function for the mean squared-error (MSE) distortion and Wasserstein-2 perception index. We prove that the DP function is always quadratic, regardless of the underlying distribution. This stems from the fact that estimators on the DP curve form a geodesic in Wasserstein space. In the Gaussian setting, we further provide a closed form expression for such estimators. For general distributions, we show how these estimators can be constructed from the estimators at the two extremes of the tradeoff: The global MSE minimizer, and a  minimizer of the MSE under a perfect perceptual quality constraint. The latter can be obtained as a stochastic transformation of the former.

        ----

        ## [1965] Neural Production Systems

        **Authors**: *Aniket Didolkar, Anirudh Goyal, Nan Rosemary Ke, Charles Blundell, Philippe Beaudoin, Nicolas Heess, Michael Mozer, Yoshua Bengio*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/d785bf9067f8af9e078b93cf26de2b54-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/d785bf9067f8af9e078b93cf26de2b54-Abstract.html)

        **Abstract**:

        Visual environments are structured, consisting of distinct  objects or entities. These entities have properties---visible or latent---that determine the manner in which they interact with one another. To partition images into entities, deep-learning researchers have proposed structural inductive biases such as slot-based architectures. To model interactions among entities, equivariant graph neural nets (GNNs) are used, but these are not particularly well suited to the task for two reasons. First, GNNs do not predispose interactions to be sparse, as relationships among independent entities are likely to be.  Second, GNNs do not factorize knowledge about  interactions in an entity-conditional manner. As an alternative, we take inspiration from cognitive science and resurrect a classic approach, production systems, which consist of a set of rule templates that are applied by binding placeholder  variables in the rules to specific entities. Rules are scored on their match to entities, and the best fitting rules are applied to update entity properties. In a series of experiments, we demonstrate that this architecture achieves a flexible, dynamic flow of control and serves to factorize entity-specific and rule-based information. This disentangling of knowledge achieves robust future-state prediction in rich visual environments, outperforming state-of-the-art methods using GNNs, and allows for the extrapolation from simple (few object) environments to more complex environments.

        ----

        ## [1966] Smoothness Matrices Beat Smoothness Constants: Better Communication Compression Techniques for Distributed Optimization

        **Authors**: *Mher Safaryan, Filip Hanzely, Peter Richtárik*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/d79c6256b9bdac53a55801a066b70da3-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/d79c6256b9bdac53a55801a066b70da3-Abstract.html)

        **Abstract**:

        Large scale distributed optimization has become the default tool for the training of supervised machine learning models with a large number of parameters and training data. Recent advancements in the field provide several mechanisms for speeding up the training, including {\em compressed communication}, {\em variance reduction} and {\em acceleration}. However, none of these methods is capable of exploiting the inherently rich data-dependent smoothness structure of the local losses beyond standard smoothness constants. In this paper, we argue that when training supervised models,  {\em smoothness matrices}---information-rich generalizations of the ubiquitous smoothness constants---can and should be exploited for further dramatic gains, both in theory and practice. In order to further alleviate the communication burden inherent in distributed optimization, we propose a novel communication sparsification strategy that can take full advantage of the smoothness matrices associated with local losses. To showcase the power of this tool, we describe how our sparsification technique can be adapted to three distributed optimization algorithms---DCGD, DIANA and ADIANA---yielding significant savings in terms of communication complexity.  The new methods always outperform the baselines, often dramatically so.

        ----

        ## [1967] Increasing Liquid State Machine Performance with Edge-of-Chaos Dynamics Organized by Astrocyte-modulated Plasticity

        **Authors**: *Vladimir A. Ivanov, Konstantinos P. Michmizos*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/d79c8788088c2193f0244d8f1f36d2db-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/d79c8788088c2193f0244d8f1f36d2db-Abstract.html)

        **Abstract**:

        The liquid state machine (LSM) combines low training complexity and biological plausibility, which has made it an attractive machine learning framework for edge and neuromorphic computing paradigms. Originally proposed as a model of brain computation, the LSM tunes its internal weights without backpropagation of gradients, which results in lower performance compared to multi-layer neural networks. Recent findings in neuroscience suggest that astrocytes, a long-neglected non-neuronal brain cell, modulate synaptic plasticity and brain dynamics, tuning brain networks to the vicinity of the computationally optimal critical phase transition between order and chaos. Inspired by this disruptive understanding of how brain networks self-tune, we propose the neuron-astrocyte liquid state machine (NALSM) that addresses under-performance through self-organized near-critical dynamics. Similar to its biological counterpart, the astrocyte model integrates neuronal activity and provides global feedback to spike-timing-dependent plasticity (STDP), which self-organizes NALSM dynamics around a critical branching factor that is associated with the edge-of-chaos. We demonstrate that NALSM achieves state-of-the-art accuracy versus comparable LSM methods, without the need for data-specific hand-tuning. With a top accuracy of $97.61\%$ on MNIST, $97.51\%$ on N-MNIST, and $85.84\%$ on Fashion-MNIST, NALSM achieved comparable performance to current fully-connected multi-layer spiking neural networks trained via backpropagation. Our findings suggest that the further development of brain-inspired machine learning methods has the potential to reach the performance of deep learning, with the added benefits of supporting robust and energy-efficient neuromorphic computing on the edge.

        ----

        ## [1968] Fair Sortition Made Transparent

        **Authors**: *Bailey Flanigan, Gregory Kehne, Ariel D. Procaccia*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/d7b431b1a0cc5f032399870ff4710743-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/d7b431b1a0cc5f032399870ff4710743-Abstract.html)

        **Abstract**:

        Sortition is an age-old democratic paradigm, widely manifested today through the random selection of citizens' assemblies. Recently-deployed algorithms select assemblies \textit{maximally fairly}, meaning that subject to demographic quotas, they give all potential participants as equal a chance as possible of being chosen.  While these fairness gains can bolster the legitimacy of citizens' assemblies and facilitate their uptake, existing algorithms remain limited by their lack of transparency. To overcome this hurdle, in this work we focus on panel selection by uniform lottery, which is easy to realize in an observable way. By this approach, the final assembly is selected by uniformly sampling some pre-selected set of $m$ possible assemblies.We provide theoretical guarantees on the fairness attainable via this type of uniform lottery, as compared to the existing maximally fair but opaque algorithms, for two different fairness objectives. We complement these results with experiments on real-world instances that demonstrate the viability of the uniform lottery approach as a method of selecting assemblies both fairly and transparently.

        ----

        ## [1969] A Max-Min Entropy Framework for Reinforcement Learning

        **Authors**: *Seungyul Han, Youngchul Sung*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/d7b76edf790923bf7177f7ebba5978df-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/d7b76edf790923bf7177f7ebba5978df-Abstract.html)

        **Abstract**:

        In this paper, we propose a max-min entropy framework for reinforcement learning (RL) to overcome the limitation of the soft actor-critic (SAC) algorithm implementing the maximum entropy RL in model-free sample-based learning. Whereas the maximum entropy RL guides learning for policies to reach states with high entropy in the future, the proposed max-min entropy framework aims to learn to visit states with low entropy and maximize the entropy of these low-entropy states to promote better exploration. For general Markov decision processes (MDPs), an efficient algorithm is constructed under the proposed max-min entropy framework based on disentanglement of exploration and exploitation. Numerical results show that the proposed algorithm yields drastic performance improvement over the current state-of-the-art RL algorithms.

        ----

        ## [1970] Reward is enough for convex MDPs

        **Authors**: *Tom Zahavy, Brendan O'Donoghue, Guillaume Desjardins, Satinder Singh*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/d7e4cdde82a894b8f633e6d61a01ef15-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/d7e4cdde82a894b8f633e6d61a01ef15-Abstract.html)

        **Abstract**:

        Maximising a cumulative reward function that is Markov and stationary, i.e., defined over state-action pairs and independent of time, is sufficient to capture many kinds of goals in a Markov decision process (MDP). However, not all goals can be captured in this manner. In this paper we study convex MDPs in which goals are expressed as convex functions of the stationary distribution and show that they cannot be formulated using stationary reward functions. Convex MDPs generalize the standard reinforcement learning (RL) problem formulation to a larger framework that includes many supervised and unsupervised RL problems, such as apprenticeship learning, constrained MDPs, and so-called pure exploration'. Our approach is to reformulate the convex MDP problem as a min-max game involving policy and cost (negative reward)players', using Fenchel duality. We propose a meta-algorithm for solving this problem and show that it unifies many existing algorithms in the literature.

        ----

        ## [1971] Fast Doubly-Adaptive MCMC to Estimate the Gibbs Partition Function with Weak Mixing Time Bounds

        **Authors**: *Shahrzad Haddadan, Yue Zhuang, Cyrus Cousins, Eli Upfal*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/d7f14b4988c30cc40e5e7b7d157bc018-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/d7f14b4988c30cc40e5e7b7d157bc018-Abstract.html)

        **Abstract**:

        We present a novel method for reducing the computational complexity of rigorously estimating the partition functions of Gibbs (or Boltzmann) distributions, which arise ubiquitously in probabilistic graphical models. A major obstacle to applying the Gibbs distribution in practice is the need to estimate their partition function (normalizing constant).  The state of the art in addressing this problem is multi-stage algorithms which consist of a cooling schedule and a mean estimator in each step of the schedule.  While the cooling schedule in these algorithms is adaptive, the mean estimate computations use MCMC as a black-box to draw approximately-independent samples. Here we develop a doubly adaptive approach, combining the adaptive cooling schedule with an adaptive MCMC mean estimator, whose number of Markov chain steps adapts dynamically to the underlying chain. Through rigorous theoretical analysis, we prove that our method outperforms the state of the art algorithms in several factors: (1) The computational complexity of our method is smaller; (2) Our method is less sensitive to loose bounds on mixing times, an inherent components in these algorithms; and (3) The improvement obtained by our method is particularly significant in the most challenging regime of high precision estimates. We demonstrate the advantage of our method in experiments run on classic factor graphs, such as voting models and Ising models.

        ----

        ## [1972] Does enforcing fairness mitigate biases caused by subpopulation shift?

        **Authors**: *Subha Maity, Debarghya Mukherjee, Mikhail Yurochkin, Yuekai Sun*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/d800149d2f947ad4d64f34668f8b20f6-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/d800149d2f947ad4d64f34668f8b20f6-Abstract.html)

        **Abstract**:

        Many instances of algorithmic bias are caused by subpopulation shifts. For example, ML models often perform worse on demographic groups that are underrepresented in the training data. In this paper, we study whether enforcing algorithmic fairness during training improves the performance of the trained model in the \emph{target domain}. On one hand, we conceive scenarios in which enforcing fairness does not improve performance in the target domain. In fact, it may even harm performance. On the other hand, we derive necessary and sufficient conditions under which enforcing algorithmic fairness leads to the Bayes model in the target domain. We also illustrate the practical implications of our theoretical results in simulations and on real data.

        ----

        ## [1973] Implicit Deep Adaptive Design: Policy-Based Experimental Design without Likelihoods

        **Authors**: *Desi R. Ivanova, Adam Foster, Steven Kleinegesse, Michael U. Gutmann, Thomas Rainforth*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/d811406316b669ad3d370d78b51b1d2e-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/d811406316b669ad3d370d78b51b1d2e-Abstract.html)

        **Abstract**:

        We introduce implicit Deep Adaptive Design (iDAD), a new method for performing adaptive experiments in real-time with implicit models. iDAD amortizes the cost of Bayesian optimal experimental design (BOED) by learning a design policy network upfront, which can then be deployed quickly at the time of the experiment. The iDAD network can be trained on any model which simulates differentiable samples, unlike previous design policy work that requires a closed form likelihood and conditionally independent experiments. At deployment, iDAD allows design decisions to be made in milliseconds, in contrast to traditional BOED approaches that require heavy computation during the experiment itself. We illustrate the applicability of iDAD on a number of experiments, and show that it provides a fast and effective mechanism for performing adaptive design with implicit models.

        ----

        ## [1974] Sample-Efficient Learning of Stackelberg Equilibria in General-Sum Games

        **Authors**: *Yu Bai, Chi Jin, Huan Wang, Caiming Xiong*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/d82118376df344b0010f53909b961db3-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/d82118376df344b0010f53909b961db3-Abstract.html)

        **Abstract**:

        Real world applications such as economics and policy making often involve solving multi-agent games with two unique features: (1) The agents are inherently asymmetric and partitioned into leaders and followers; (2) The agents have different reward functions, thus the game is general-sum. The majority of existing results in this field focuses on either symmetric solution concepts (e.g. Nash equilibrium) or zero-sum games. It remains open how to learn the Stackelberg equilibrium---an asymmetric analog of the Nash equilibrium---in general-sum games efficiently from noisy samples.  This paper initiates the theoretical study of sample-efficient learning of the Stackelberg equilibrium, in the bandit feedback setting where we only observe noisy samples of the reward. We consider three representative two-player general-sum games: bandit games, bandit-reinforcement learning (bandit-RL) games, and linear bandit games. In all these games, we identify a fundamental gap between the exact value of the Stackelberg equilibrium and its estimated version using finitely many noisy samples, which can not be closed information-theoretically regardless of the algorithm. We then establish sharp positive results on sample-efficient learning of Stackelberg equilibrium with value optimal up to the gap identified above, with matching lower bounds in the dependency on the gap, error tolerance, and the size of the action spaces. Overall, our results unveil unique challenges in learning Stackelberg equilibria under noisy bandit feedback, which we hope could shed light on future research on this topic.

        ----

        ## [1975] Non-approximate Inference for Collective Graphical Models on Path Graphs via Discrete Difference of Convex Algorithm

        **Authors**: *Yasunori Akagi, Naoki Marumo, Hideaki Kim, Takeshi Kurashima, Hiroyuki Toda*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/d827f12e35eae370ba9c65b7f6026695-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/d827f12e35eae370ba9c65b7f6026695-Abstract.html)

        **Abstract**:

        The importance of aggregated count data, which is calculated from the data of multiple individuals, continues to increase. Collective Graphical Model (CGM) is a probabilistic approach to the analysis of aggregated data. One of the most important operations in CGM is maximum a posteriori (MAP) inference of unobserved variables under given observations. Because the MAP inference problem for general CGMs has been shown to be NP-hard, an approach that solves an approximate problem has been proposed. However, this approach has two major drawbacks. First, the quality of the solution deteriorates when the values in the count tables are small, because the approximation becomes inaccurate. Second, since continuous relaxation is applied, the integrality constraints of the output are violated. To resolve these problems, this paper proposes a new method for MAP inference for CGMs on path graphs. Our method is based on the Difference of Convex Algorithm (DCA), which is a general methodology to minimize a function represented as the sum of a convex function and a concave function. In our algorithm, important subroutines in DCA can be efficiently calculated by minimum convex cost flow algorithms. Experiments show that the proposed method outputs higher quality solutions than the conventional approach.

        ----

        ## [1976] Implicit Task-Driven Probability Discrepancy Measure for Unsupervised Domain Adaptation

        **Authors**: *Mao Li, Kaiqi Jiang, Xinhua Zhang*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/d82f9436247aa0049767b776dceab4ed-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/d82f9436247aa0049767b776dceab4ed-Abstract.html)

        **Abstract**:

        Probability discrepancy measure is a fundamental construct for numerous machine learning models such as weakly supervised learning and generative modeling.  However, most measures overlook the fact that the distributions are not the end-product of learning, but are the basis of downstream predictor.  Therefore it is important to warp the probability discrepancy measure towards the end tasks, and we hence propose a new bi-level optimization based approach so that the two distributions are compared not uniformly against the entire hypothesis space, but only with respect to the optimal predictor for the downstream end task.  When applied to margin disparity discrepancy and contrastive domain discrepancy, our method significantly improves the performance in unsupervised domain adaptation, and enjoys a much more principled training process.

        ----

        ## [1977] SBO-RNN: Reformulating Recurrent Neural Networks via Stochastic Bilevel Optimization

        **Authors**: *Ziming Zhang, Yun Yue, Guojun Wu, Yanhua Li, Haichong K. Zhang*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/d87ca511e2a8593c8039ef732f5bffed-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/d87ca511e2a8593c8039ef732f5bffed-Abstract.html)

        **Abstract**:

        In this paper we consider the training stability of recurrent neural networks (RNNs) and propose a family of RNNs, namely SBO-RNN, that can be formulated using stochastic bilevel optimization (SBO). With the help of stochastic gradient descent (SGD), we manage to convert the SBO problem into an RNN where the feedforward and backpropagation solve the lower and upper-level optimization for learning hidden states and their hyperparameters, respectively. We prove that under mild conditions there is no vanishing or exploding gradient in training SBO-RNN. Empirically we demonstrate our approach with superior performance on several benchmark datasets, with fewer parameters, less training data, and much faster convergence. Code is available at https://zhang-vislab.github.io.

        ----

        ## [1978] Navigating to the Best Policy in Markov Decision Processes

        **Authors**: *Aymen Al Marjani, Aurélien Garivier, Alexandre Proutière*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/d9896106ca98d3d05b8cbdf4fd8b13a1-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/d9896106ca98d3d05b8cbdf4fd8b13a1-Abstract.html)

        **Abstract**:

        We investigate the classical active pure exploration problem in Markov Decision Processes, where the agent sequentially selects actions and, from the resulting system trajectory, aims at identifying the best policy as fast as possible. We propose a problem-dependent lower bound on the average number of steps required before a correct answer can be given with probability at least $1-\delta$. We further provide the first algorithm with an instance-specific sample complexity in this setting. This algorithm addresses the general case of communicating MDPs; we also propose a variant with a reduced exploration rate (and hence faster convergence) under an additional ergodicity assumption. This work extends previous results relative to the \emph{generative setting}~\cite{pmlr-v139-marjani21a}, where the agent could at each step query the random outcome of any (state, action) pair. In contrast, we show here how to deal with the \emph{navigation constraints}, induced by the \emph{online setting}. Our analysis relies on an ergodic theorem for non-homogeneous Markov chains which we consider of wide interest in the analysis of Markov Decision Processes.

        ----

        ## [1979] A Faster Decentralized Algorithm for Nonconvex Minimax Problems

        **Authors**: *Wenhan Xian, Feihu Huang, Yanfu Zhang, Heng Huang*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/d994e3728ba5e28defb88a3289cd7ee8-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/d994e3728ba5e28defb88a3289cd7ee8-Abstract.html)

        **Abstract**:

        In this paper, we study the nonconvex-strongly-concave minimax optimization problem on decentralized setting. The minimax problems are attracting increasing attentions because of their popular practical applications such as policy evaluation and adversarial training. As training data become larger, distributed training has been broadly adopted in machine learning tasks. Recent research works show that the decentralized distributed data-parallel training techniques are specially promising, because they can achieve the efficient communications and avoid the bottleneck problem on the central node or the latency of low bandwidth network. However, the decentralized minimax problems were seldom studied in literature and the existing methods suffer from very high gradient complexity. To address this challenge, we propose a new faster decentralized algorithm, named as DM-HSGD, for nonconvex minimax problems by using the variance reduced technique of hybrid stochastic gradient descent. We prove that our DM-HSGD algorithm achieves stochastic first-order oracle (SFO) complexity of $O(\kappa^3 \epsilon^{-3})$ for decentralized stochastic nonconvex-strongly-concave problem to search an $\epsilon$-stationary point, which improves the exiting best theoretical results. Moreover, we also prove that our algorithm achieves linear speedup with respect to the number of workers. Our experiments on decentralized settings show the superior performance of our new algorithm.

        ----

        ## [1980] Generalization Bounds For Meta-Learning: An Information-Theoretic Analysis

        **Authors**: *Qi Chen, Changjian Shui, Mario Marchand*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/d9d347f57ae11f34235b4555710547d8-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/d9d347f57ae11f34235b4555710547d8-Abstract.html)

        **Abstract**:

        We derive a novel information-theoretic analysis of the generalization property of meta-learning algorithms. Concretely, our analysis proposes a generic understanding in both the conventional learning-to-learn framework \citep{amit2018meta} and the modern model-agnostic meta-learning (MAML) algorithms \citep{finn2017model}.Moreover, we provide a data-dependent generalization bound for the stochastic variant of MAML, which is \emph{non-vacuous} for deep few-shot learning. As compared to previous bounds that depend on the square norms of gradients, empirical validations on both simulated data and a well-known few-shot benchmark show that our bound is orders of magnitude tighter in most conditions.

        ----

        ## [1981] ReLU Regression with Massart Noise

        **Authors**: *Ilias Diakonikolas, Jong Ho Park, Christos Tzamos*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/d9d3837ee7981e8c064774da6cdd98bf-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/d9d3837ee7981e8c064774da6cdd98bf-Abstract.html)

        **Abstract**:

        We study the fundamental problem of ReLU regression, where the goal is to fit Rectified Linear Units (ReLUs) to data. This supervised learning task is efficiently solvable in the realizable setting, but is known to be computationally hard with adversarial label noise. In this work, we focus on ReLU regression in the Massart noise model,  a natural and well-studied semi-random noise model. In this model, the label of every point is generated according to a function in the class, but an adversary is allowed to change this value arbitrarily with some probability, which is {\em at most} $\eta < 1/2$. We develop an efficient algorithm that achieves exact parameter recovery in this model under mild anti-concentration assumptions on the underlying distribution. Such assumptions are necessary for exact recovery to be information-theoretically possible. We demonstrate that our algorithm significantly outperforms naive applications of $\ell_1$ and $\ell_2$ regression on both synthetic and real data.

        ----

        ## [1982] Identification of the Generalized Condorcet Winner in Multi-dueling Bandits

        **Authors**: *Björn Haddenhorst, Viktor Bengs, Eyke Hüllermeier*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/d9de6a144a3cc26cb4b3c47b206a121a-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/d9de6a144a3cc26cb4b3c47b206a121a-Abstract.html)

        **Abstract**:

        The reliable identification of the “best” arm while keeping the sample complexity as low as possible is a common task in the field of multi-armed bandits. In the multi-dueling variant of multi-armed bandits, where feedback is provided in the form of a winning arm among as set of k chosen ones, a reasonable notion of best arm is the generalized Condorcet winner (GCW). The latter is an the arm that has the greatest probability of being the winner in each subset containing it. In this paper, we derive lower bounds on the sample complexity for the task of identifying the GCW under various assumptions. As a by-product, our lower bound results provide new insights for the special case of dueling bandits (k = 2). We propose the Dvoretzky–Kiefer–Wolfowitz tournament (DKWT) algorithm, which we prove to be nearly optimal. In a numerical study, we show that DKWT empirically outperforms current state-of-the-art algorithms, even in the special case of dueling bandits or under a Plackett-Luce assumption on the feedback mechanism.

        ----

        ## [1983] Robust Inverse Reinforcement Learning under Transition Dynamics Mismatch

        **Authors**: *Luca Viano, Yu-Ting Huang, Parameswaran Kamalaruban, Adrian Weller, Volkan Cevher*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/d9e74f47610385b11e295eec4c58d473-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/d9e74f47610385b11e295eec4c58d473-Abstract.html)

        **Abstract**:

        We study the inverse reinforcement learning (IRL) problem under a transition dynamics mismatch between the expert and the learner. Specifically, we consider the Maximum Causal Entropy (MCE) IRL learner model and provide a tight upper bound on the learner's performance degradation based on the $\ell_1$-distance between the transition dynamics of the expert and the learner. Leveraging insights from the Robust RL literature, we propose a robust MCE IRL algorithm, which is a principled approach to help with this mismatch. Finally, we empirically demonstrate the stable performance of our algorithm compared to the standard MCE IRL algorithm under transition dynamics mismatches in both finite and continuous MDP problems.

        ----

        ## [1984] Re-ranking for image retrieval and transductive few-shot classification

        **Authors**: *Xi Shen, Yang Xiao, Shell Xu Hu, Othman Sbai, Mathieu Aubry*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/d9fc0cdb67638d50f411432d0d41d0ba-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/d9fc0cdb67638d50f411432d0d41d0ba-Abstract.html)

        **Abstract**:

        In the problems of image retrieval and few-shot classification, the mainstream approaches focus on learning a better feature representation. However, directly tackling the distance or similarity measure between images could also be efficient. To this end, we revisit the idea of re-ranking the top-k retrieved images in the context of image retrieval (e.g., the k-reciprocal nearest neighbors) and generalize this idea to transductive few-shot learning. We propose to meta-learn the re-ranking updates such that the similarity graph converges towards the target similarity graph induced by the image labels. Specifically, the re-ranking module takes as input an initial similarity graph between the query image and the contextual images using a pre-trained feature extractor, and predicts an improved similarity graph by leveraging the structure among the involved images. We show that our re-ranking approach can be applied to unseen images and can further boost existing approaches for both image retrieval and few-shot learning problems. Our approach operates either independently or in conjunction with classical re-ranking approaches, yielding clear and consistent improvements on image retrieval (CUB, Cars, SOP, rOxford5K and rParis6K) and transductive few-shot classification (Mini-ImageNet, tiered-ImageNet and CIFAR-FS) benchmarks. Our code is available at https://imagine.enpc.fr/~shenx/SSR/.

        ----

        ## [1985] Post-processing for Individual Fairness

        **Authors**: *Felix Petersen, Debarghya Mukherjee, Yuekai Sun, Mikhail Yurochkin*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/d9fea4ca7e4a74c318ec27c1deb0796c-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/d9fea4ca7e4a74c318ec27c1deb0796c-Abstract.html)

        **Abstract**:

        Post-processing in algorithmic fairness is a versatile approach for correcting bias in ML systems that are already used in production. The main appeal of post-processing is that it avoids expensive retraining. In this work, we propose general post-processing algorithms for individual fairness (IF). We consider a setting where the learner only has access to the predictions of the original model and a similarity graph between individuals, guiding the desired fairness constraints. We cast the IF post-processing problem as a graph smoothing problem corresponding to graph Laplacian regularization that preserves the desired "treat similar individuals similarly" interpretation. Our theoretical results demonstrate the connection of the new objective function to a local relaxation of the original individual fairness. Empirically, our post-processing algorithms correct individual biases in large-scale NLP models such as BERT, while preserving accuracy.

        ----

        ## [1986] OpenMatch: Open-Set Semi-supervised Learning with Open-set Consistency Regularization

        **Authors**: *Kuniaki Saito, Donghyun Kim, Kate Saenko*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/da11e8cd1811acb79ccf0fd62cd58f86-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/da11e8cd1811acb79ccf0fd62cd58f86-Abstract.html)

        **Abstract**:

        Semi-supervised learning (SSL) is an effective means to leverage unlabeled data to improve a modelâ€™s performance. Typical SSL methods like FixMatch assume that labeled and unlabeled data share the same label space. However, in practice, unlabeled data can contain categories unseen in the labeled set, i.e., outliers, which can significantly harm the performance of SSL algorithms.  To address this problem, we propose a novel Open-set Semi-Supervised Learning (OSSL) approach called OpenMatch.Learning representations of inliers while rejecting outliers is essential for the success of OSSL. To this end, OpenMatch unifies FixMatch with novelty detection based on one-vs-all (OVA) classifiers. The OVA-classifier outputs the confidence score of a sample being an inlier, providing a threshold to detect outliers. Another key contribution is an open-set soft-consistency regularization loss, which enhances the smoothness of the OVA-classifier with respect to input transformations and greatly improves outlier detection. \ours achieves state-of-the-art performance on three datasets, and even outperforms a fully supervised model in detecting outliers unseen in unlabeled data on CIFAR10. The code is available at \url{https://github.com/VisionLearningGroup/OP_Match}.

        ----

        ## [1987] End-to-End Training of Multi-Document Reader and Retriever for Open-Domain Question Answering

        **Authors**: *Devendra Singh Sachan, Siva Reddy, William L. Hamilton, Chris Dyer, Dani Yogatama*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/da3fde159d754a2555eaa198d2d105b2-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/da3fde159d754a2555eaa198d2d105b2-Abstract.html)

        **Abstract**:

        We present an end-to-end differentiable training method for retrieval-augmented open-domain question answering systems that combine information from multiple retrieved documents when generating answers. We model retrieval decisions as latent variables over sets of relevant documents. Since marginalizing over sets of retrieved documents is computationally hard, we approximate this using an expectation-maximization algorithm. We iteratively estimate the value of our latent variable (the set of relevant documents for a given question) and then use this estimate to update the retriever and reader parameters. We hypothesize that such end-to-end training allows training signals to flow to the reader and then to the retriever better than staged-wise training. This results in a retriever that is able to select more relevant documents for a question and a reader that is trained on more accurate documents to generate an answer. Experiments on three benchmark datasets demonstrate that our proposed method outperforms all existing approaches of comparable size by 2-3% absolute exact match points, achieving new state-of-the-art results. Our results also demonstrate the feasibility of learning to retrieve to improve answer generation without explicit supervision of retrieval decisions.

        ----

        ## [1988] Fast Algorithms for $L_\infty$-constrained S-rectangular Robust MDPs

        **Authors**: *Bahram Behzadian, Marek Petrik, Chin Pang Ho*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/da4fb5c6e93e74d3df8527599fa62642-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/da4fb5c6e93e74d3df8527599fa62642-Abstract.html)

        **Abstract**:

        Robust Markov decision processes (RMDPs) are a useful building block of robust reinforcement learning algorithms but can be hard to solve. This paper proposes a fast, exact algorithm for computing the Bellman operator for S-rectangular robust Markov decision processes with $L_\infty$-constrained rectangular ambiguity sets. The algorithm combines a novel homotopy continuation method with a bisection method to solve S-rectangular ambiguity in quasi-linear time in the number of states and actions. The algorithm improves on the cubic time required by leading general linear programming methods. Our experimental results confirm the practical viability of our method and show that it outperforms a leading commercial optimization package by several orders of magnitude.

        ----

        ## [1989] Instance-optimal Mean Estimation Under Differential Privacy

        **Authors**: *Ziyue Huang, Yuting Liang, Ke Yi*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/da54dd5a0398011cdfa50d559c2c0ef8-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/da54dd5a0398011cdfa50d559c2c0ef8-Abstract.html)

        **Abstract**:

        Mean estimation under differential privacy is a fundamental problem, but worst-case optimal mechanisms do not offer meaningful utility guarantees in practice when the global sensitivity is very large.  Instead, various heuristics have been proposed to reduce the error on real-world data that do not resemble the worst-case instance.  This paper takes a principled approach, yielding a mechanism that is instance-optimal in a strong sense.  In addition to its theoretical optimality, the mechanism is also simple and practical, and adapts to a variety of data characteristics without the need of parameter tuning.  It easily extends to the local and shuffle model as well.

        ----

        ## [1990] Look at the Variance! Efficient Black-box Explanations with Sobol-based Sensitivity Analysis

        **Authors**: *Thomas Fel, Rémi Cadène, Mathieu Chalvidal, Matthieu Cord, David Vigouroux, Thomas Serre*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/da94cbeff56cfda50785df477941308b-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/da94cbeff56cfda50785df477941308b-Abstract.html)

        **Abstract**:

        We describe a novel attribution method which is grounded in Sensitivity Analysis and uses  Sobol indices. Beyond modeling the individual contributions of image regions,  Sobol indices provide an efficient way to capture higher-order interactions between image regions and their contributions to a neural network's prediction through the lens of variance.We describe an approach that makes the computation of these indices efficient for high-dimensional problems by using perturbation masks coupled with efficient estimators to handle the high dimensionality of images.Importantly, we show that the proposed method leads to favorable scores on standard benchmarks for vision (and language models) while drastically reducing the computing time compared to other black-box methods -- even surpassing the accuracy of state-of-the-art white-box methods which require access to internal representations. Our code is freely available:github.com/fel-thomas/Sobol-Attribution-Method.

        ----

        ## [1991] PatchGame: Learning to Signal Mid-level Patches in Referential Games

        **Authors**: *Kamal Gupta, Gowthami Somepalli, Anubhav Gupta, Vinoj Yasanga Jayasundara Magalle Hewa, Matthias Zwicker, Abhinav Shrivastava*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/dac32839a9f0baae954b41abee610cc0-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/dac32839a9f0baae954b41abee610cc0-Abstract.html)

        **Abstract**:

        We study a referential game (a type of signaling game) where two agents communicate with each other via a discrete bottleneck to achieve a common goal. In our referential game, the goal of the speaker is to compose a message or a symbolic representation of "important" image patches, while the task for the listener is to match the speaker's message to a different view of the same image. We show that it is indeed possible for the two agents to develop a communication protocol without explicit or implicit supervision. We further investigate the developed protocol and show the applications in speeding up recent Vision Transformers by using only important patches, and as pre-training for downstream recognition tasks (e.g., classification).

        ----

        ## [1992] Implicit Generative Copulas

        **Authors**: *Tim Janke, Mohamed Ghanmi, Florian Steinke*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/dac4a67bdc4a800113b0f1ad67ed696f-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/dac4a67bdc4a800113b0f1ad67ed696f-Abstract.html)

        **Abstract**:

        Copulas are a powerful tool for modeling multivariate distributions as they allow to separately estimate the univariate marginal distributions and the joint dependency structure. However, known parametric copulas offer limited flexibility especially in high dimensions, while commonly used non-parametric methods suffer from the curse of dimensionality. A popular remedy is to construct a tree-based hierarchy of conditional bivariate copulas.In this paper, we propose a flexible, yet conceptually simple alternative based on implicit generative neural networks.The key challenge is to ensure marginal uniformity of the estimated copula distribution.We achieve this by learning a multivariate latent distribution with unspecified marginals but the desired dependency structure.By applying the probability integral transform, we can then obtain samples from the high-dimensional copula distribution without relying on parametric assumptions or the need to find a suitable tree structure.Experiments on synthetic and real data from finance, physics, and image generation demonstrate the performance of this approach.

        ----

        ## [1993] Tensor Normal Training for Deep Learning Models

        **Authors**: *Yi Ren, Donald Goldfarb*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/dae3312c4c6c7000a37ecfb7b0aeb0e4-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/dae3312c4c6c7000a37ecfb7b0aeb0e4-Abstract.html)

        **Abstract**:

        Despite the predominant use of first-order methods for training deep learning models, second-order methods, and in particular, natural gradient methods, remain of interest because of their potential for accelerating training through the use of curvature information. Several methods with non-diagonal preconditioning matrices, including KFAC, Shampoo, and K-BFGS, have been proposed and shown to be effective. Based on the so-called tensor normal (TN) distribution, we propose and analyze a brand new approximate natural gradient method, Tensor Normal Training (TNT), which like Shampoo, only requires knowledge of the shape of the training parameters. By approximating the probabilistically based Fisher matrix, as opposed to the empirical Fisher matrix, our method uses the block-wise covariance of the sampling based gradient as the pre-conditioning matrix. Moreover, the assumption that the sampling-based (tensor) gradient follows a TN distribution, ensures that its covariance has a Kronecker separable structure, which leads to a tractable approximation to the Fisher matrix. Consequently, TNT's memory requirements and per-iteration computational costs are only slightly higher than those for first-order methods. In our experiments, TNT exhibited superior optimization performance to state-of-the-art first-order methods, and comparable optimization performance to the state-of-the-art second-order methods KFAC and Shampoo. Moreover, TNT demonstrated its ability to generalize as well as first-order methods, while using fewer epochs.

        ----

        ## [1994] Unintended Selection: Persistent Qualification Rate Disparities and Interventions

        **Authors**: *Reilly Raab, Yang Liu*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/db00f1b7fdf48fd26b5fb5f309e9afaf-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/db00f1b7fdf48fd26b5fb5f309e9afaf-Abstract.html)

        **Abstract**:

        Realistically---and equitably---modeling the dynamics of group-level disparities in machine learning remains an open problem. In particular, we desire models that do not suppose inherent differences between artificial groups of people---but rather endogenize disparities by appeal to unequal initial conditions of insular subpopulations. In this paper, agents each have a real-valued feature $X$ (e.g., credit score) informed by a ``true'' binary label $Y$ representing qualification (e.g., for a loan). Each agent alternately (1) receives a binary classification label $\hat{Y}$ (e.g., loan approval) from a Bayes-optimal machine learning classifier observing $X$ and (2) may update their qualification $Y$ by imitating successful strategies (e.g., seek a raise) within an isolated group $G$ of agents to which they belong. We consider the disparity of qualification rates $\Pr(Y=1)$ between different groups and how this disparity changes subject to a sequence of Bayes-optimal classifiers repeatedly retrained on the global population. We model the evolving qualification rates of each subpopulation (group) using the replicator equation, which derives from a class of imitation processes. We show that differences in qualification rates between subpopulations can persist indefinitely for a set of non-trivial equilibrium states due to uniformed classifier deployments, even when groups are identical in all aspects except initial qualification densities. We next simulate the effects of commonly proposed fairness interventions on this dynamical system along with a new feedback control mechanism capable of permanently eliminating group-level qualification rate disparities. We conclude by discussing the limitations of our model and findings and by outlining potential future work.

        ----

        ## [1995] Revisiting 3D Object Detection From an Egocentric Perspective

        **Authors**: *Boyang Deng, Charles R. Qi, Mahyar Najibi, Thomas A. Funkhouser, Yin Zhou, Dragomir Anguelov*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/db182d2552835bec774847e06406bfa2-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/db182d2552835bec774847e06406bfa2-Abstract.html)

        **Abstract**:

        3D object detection is a key module for safety-critical robotics applications such as autonomous driving. For these applications, we care most about how the detections affect the ego-agent’s behavior and safety (the egocentric perspective). Intuitively, we seek more accurate descriptions of object geometry when it’s more likely to interfere with the ego-agent’s motion trajectory. However, current detection metrics, based on box Intersection-over-Union (IoU), are object-centric and aren’t designed to capture the spatio-temporal relationship between objects and the ego-agent. To address this issue, we propose a new egocentric measure to evaluate 3D object detection,  namely Support Distance Error (SDE). Our analysis based on SDE reveals that the egocentric detection quality is bounded by the coarse geometry of the bounding boxes. Given the insight that SDE would benefit from more accurate geometry descriptions, we propose to represent objects as amodal contours, specifically amodal star-shaped polygons, and devise a simple model, StarPoly, to predict such contours. Our experiments on the large-scale Waymo Open Dataset show that SDE better reflects the impact of detection quality on the ego-agent’s safety compared to IoU; and the estimated contours from StarPoly consistently improve the egocentric detection quality over recent 3D object detectors.

        ----

        ## [1996] Optimizing Information-theoretical Generalization Bound via Anisotropic Noise of SGLD

        **Authors**: *Bohan Wang, Huishuai Zhang, Jieyu Zhang, Qi Meng, Wei Chen, Tie-Yan Liu*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/db2b4182156b2f1f817860ac9f409ad7-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/db2b4182156b2f1f817860ac9f409ad7-Abstract.html)

        **Abstract**:

        Recently, the information-theoretical framework has been proven to be able to obtain non-vacuous generalization bounds for large models trained by Stochastic Gradient Langevin Dynamics (SGLD) with isotropic noise.  In this paper, we optimize the information-theoretical generalization bound by manipulating the noise structure in SGLD. We prove that with constraint to guarantee low empirical risk, the optimal noise covariance is the square root of the expected gradient covariance if both the prior and the posterior are jointly optimized. This validates that the optimal noise is quite close to the empirical gradient covariance.  Technically, we develop a new information-theoretical bound that enables such an optimization analysis. We then apply matrix analysis to derive the form of optimal noise covariance. Presented constraint and results are validated by the empirical observations.

        ----

        ## [1997] Addressing Algorithmic Disparity and Performance Inconsistency in Federated Learning

        **Authors**: *Sen Cui, Weishen Pan, Jian Liang, Changshui Zhang, Fei Wang*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/db8e1af0cb3aca1ae2d0018624204529-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/db8e1af0cb3aca1ae2d0018624204529-Abstract.html)

        **Abstract**:

        Federated learning (FL) has gain growing interests for its capability of learning from distributed data sources collectively without the need of accessing the raw data samples across different sources. So far FL research has mostly focused on improving the performance, how the algorithmic disparity will be impacted for the model learned from FL and the impact of algorithmic disparity on the utility inconsistency are largely unexplored. In this paper, we propose an FL framework to jointly consider performance consistency and algorithmic fairness across different local clients (data sources). We derive our framework from a constrained multi-objective optimization perspective, in which we learn a model satisfying fairness constraints on all clients with consistent performance. Specifically, we treat the algorithm prediction loss at each local client as an objective and maximize the worst-performing client with fairness constraints through optimizing a surrogate maximum function with all objectives involved. A gradient-based procedure is employed to achieve the Pareto optimality of this optimization problem. Theoretical analysis is provided to prove that our method can converge to a Pareto solution that achieves the min-max performance with fairness constraints on all clients. Comprehensive experiments on synthetic and real-world datasets demonstrate the superiority that our approach over baselines and its effectiveness in achieving both fairness and consistency across all local clients.

        ----

        ## [1998] A Mathematical Framework for Quantifying Transferability in Multi-source Transfer Learning

        **Authors**: *Xinyi Tong, Xiangxiang Xu, Shao-Lun Huang, Lizhong Zheng*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/db9ad56c71619aeed9723314d1456037-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/db9ad56c71619aeed9723314d1456037-Abstract.html)

        **Abstract**:

        Current transfer learning algorithm designs mainly focus on the similarities between source and target tasks, while the impacts of the sample sizes of these tasks are often not sufficiently addressed. This paper proposes a mathematical framework for quantifying the transferability in multi-source transfer learning problems, with both the task similarities and the sample complexity of learning models taken into account. In particular, we consider the setup where the models learned from different tasks are linearly combined for learning the target task, and use the optimal combining coefficients to measure the transferability. Then, we demonstrate the analytical expression of this transferability measure, characterized by the sample sizes, model complexity, and the similarities between source and target tasks, which provides fundamental insights of the knowledge transferring mechanism and the guidance for algorithm designs. Furthermore, we apply our analyses for practical learning tasks, and establish a quantifiable transferability measure by exploiting a parameterized model. In addition, we develop an alternating iterative algorithm to implement our theoretical results for training deep neural networks in multi-source transfer learning tasks. Finally, experiments on image classification tasks show that our approach outperforms existing transfer learning algorithms in multi-source and few-shot scenarios.

        ----

        ## [1999] Morié Attack (MA): A New Potential Risk of Screen Photos

        **Authors**: *Dantong Niu, Ruohao Guo, Yisen Wang*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/db9eeb7e678863649bce209842e0d164-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/db9eeb7e678863649bce209842e0d164-Abstract.html)

        **Abstract**:

        Images, captured by a camera, play a critical role in training Deep Neural Networks (DNNs). Usually, we assume the images acquired by cameras are consistent with the ones perceived by human eyes. However, due to the different physical mechanisms between human-vision and computer-vision systems, the final perceived images could be very different in some cases, for example shooting on digital monitors. In this paper, we find a special phenomenon in digital image processing, the moiré effect, that could cause unnoticed security threats to DNNs. Based on it, we propose a Moiré Attack (MA) that generates the physical-world moiré pattern adding to the images by mimicking the shooting process of digital devices. Extensive experiments demonstrate that our proposed digital Moiré Attack (MA) is a perfect camouflage for attackers to tamper with DNNs with a high success rate ($100.0\%$ for untargeted and $97.0\%$ for targeted attack with the noise budget $\epsilon=4$), high transferability rate across different models, and high robustness under various defenses. Furthermore, MA owns great stealthiness because the moiré effect is unavoidable due to the camera's inner physical structure, which therefore hardly attracts the awareness of humans. Our code is available at https://github.com/Dantong88/Moire_Attack.

        ----

        

[Go to the previous page](NIPS-2021-list09.md)

[Go to the next page](NIPS-2021-list11.md)

[Go to the catalog section](README.md)