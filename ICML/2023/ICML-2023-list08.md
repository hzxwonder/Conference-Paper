## [1400] Understanding Self-Predictive Learning for Reinforcement Learning

        **Authors**: *Yunhao Tang, Zhaohan Daniel Guo, Pierre Harvey Richemond, Bernardo Ávila Pires, Yash Chandak, Rémi Munos, Mark Rowland, Mohammad Gheshlaghi Azar, Charline Le Lan, Clare Lyle, András György, Shantanu Thakoor, Will Dabney, Bilal Piot, Daniele Calandriello, Michal Valko*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/tang23d.html](https://proceedings.mlr.press/v202/tang23d.html)

        **Abstract**:

        We study the learning dynamics of self-predictive learning for reinforcement learning, a family of algorithms that learn representations by minimizing the prediction error of their own future latent representations. Despite its recent empirical success, such algorithms have an apparent defect: trivial representations (such as constants) minimize the prediction error, yet it is obviously undesirable to converge to such solutions. Our central insight is that careful designs of the optimization dynamics are critical to learning meaningful representations. We identify that a faster paced optimization of the predictor and semi-gradient updates on the representation, are crucial to preventing the representation collapse. Then in an idealized setup, we show self-predictive learning dynamics carries out spectral decomposition on the state transition matrix, effectively capturing information of the transition dynamics. Building on the theoretical insights, we propose bidirectional self-predictive learning, a novel self-predictive algorithm that learns two representations simultaneously. We examine the robustness of our theoretical insights with a number of small-scale experiments and showcase the promise of the novel representation learning algorithm with large-scale experiments.

        ----

        ## [1401] DoMo-AC: Doubly Multi-step Off-policy Actor-Critic Algorithm

        **Authors**: *Yunhao Tang, Tadashi Kozuno, Mark Rowland, Anna Harutyunyan, Rémi Munos, Bernardo Ávila Pires, Michal Valko*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/tang23e.html](https://proceedings.mlr.press/v202/tang23e.html)

        **Abstract**:

        Multi-step learning applies lookahead over multiple time steps and has proved valuable in policy evaluation settings. However, in the optimal control case, the impact of multi-step learning has been relatively limited despite a number of prior efforts. Fundamentally, this might be because multi-step policy improvements require operations that cannot be approximated by stochastic samples, hence hindering the widespread adoption of such methods in practice. To address such limitations, we introduce doubly multi-step off-policy VI (DoMo-VI), a novel oracle algorithm that combines multi-step policy improvements and policy evaluations. DoMo-VI enjoys guaranteed convergence speed-up to the optimal policy and is applicable in general off-policy learning settings. We then propose doubly multi-step off-policy actor-critic (DoMo-AC), a practical instantiation of the DoMo-VI algorithm. DoMo-AC introduces a bias-variance trade-off that ensures improved policy gradient estimates. When combined with the IMPALA architecture, DoMo-AC has showed improvements over the baseline algorithm on Atari-57 game benchmarks.

        ----

        ## [1402] Towards Understanding Generalization of Graph Neural Networks

        **Authors**: *Huayi Tang, Yong Liu*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/tang23f.html](https://proceedings.mlr.press/v202/tang23f.html)

        **Abstract**:

        Graph neural networks (GNNs) are widely used in machine learning for graph-structured data. Even though GNNs have achieved remarkable success in real-world applications, understanding their working mechanism in theory is still on primary stage. In this paper, we move towards this goal from the perspective of generalization. Specifically, with consideration of stochastic optimization, we establish high probability bounds of generalization gap and gradients for transductive learning algorithms. After that, we provide high probability bounds of generalization gap for popular GNNs and analyze the factors affecting their generalization capability. These theoretical results reveal how the network architecture impacts the generalization gap. Experiments on benchmark datasets validate the theoretical findings. Our results provide new insights into understanding generalization of GNNs.

        ----

        ## [1403] Towards a better understanding of representation dynamics under TD-learning

        **Authors**: *Yunhao Tang, Rémi Munos*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/tang23g.html](https://proceedings.mlr.press/v202/tang23g.html)

        **Abstract**:

        TD-learning is a foundation reinforcement learning (RL) algorithm for value prediction. Critical to the accuracy of value predictions is the quality of state representations. In this work, we consider the question: how does end-to-end TD-learning impact the representation over time? Complementary to prior work, we provide a set of analysis that sheds further light on the representation dynamics under TD-learning. We first show that when the environments are reversible, end-to-end TD-learning strictly decreases the value approximation error over time. Under further assumptions on the environments, we can connect the representation dynamics with spectral decomposition over the transition matrix. This latter finding establishes fitting multiple value functions from randomly generated rewards as a useful auxiliary task for representation learning, as we empirically validate on both tabular and Atari game suites.

        ----

        ## [1404] VA-learning as a more efficient alternative to Q-learning

        **Authors**: *Yunhao Tang, Rémi Munos, Mark Rowland, Michal Valko*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/tang23h.html](https://proceedings.mlr.press/v202/tang23h.html)

        **Abstract**:

        In reinforcement learning, the advantage function is critical for policy improvement, but is often extracted from a learned Q-function. A natural question is: Why not learn the advantage function directly? In this work, we introduce VA-learning, which directly learns advantage function and value function using bootstrapping, without explicit reference to Q-functions. VA-learning learns off-policy and enjoys similar theoretical guarantees as Q-learning. Thanks to the direct learning of advantage function and value function, VA-learning improves the sample efficiency over Q-learning both in tabular implementations and deep RL agents on Atari-57 games. We also identify a close connection between VA-learning and the dueling architecture, which partially explains why a simple architectural change to DQN agents tends to improve performance.

        ----

        ## [1405] Defects of Convolutional Decoder Networks in Frequency Representation

        **Authors**: *Ling Tang, Wen Shen, Zhanpeng Zhou, Yuefeng Chen, Quanshi Zhang*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/tang23i.html](https://proceedings.mlr.press/v202/tang23i.html)

        **Abstract**:

        In this paper, we prove the representation defects of a cascaded convolutional decoder network, considering the capacity of representing different frequency components of an input sample. We conduct the discrete Fourier transform on each channel of the feature map in an intermediate layer of the decoder network. Then, we extend the 2D circular convolution theorem to represent the forward and backward propagations through convolutional layers in the frequency domain. Based on this, we prove three defects in representing feature spectrums. First, we prove that the convolution operation, the zero-padding operation, and a set of other settings all make a convolutional decoder network more likely to weaken high-frequency components. Second, we prove that the upsampling operation generates a feature spectrum, in which strong signals repetitively appear at certain frequencies. Third, we prove that if the frequency components in the input sample and frequency components in the target output for regression have a small shift, then the decoder usually cannot be effectively learned.

        ----

        ## [1406] Difference-in-Differences Meets Tree-based Methods: Heterogeneous Treatment Effects Estimation with Unmeasured Confounding

        **Authors**: *Caizhi Tang, Huiyuan Wang, Xinyu Li, Qing Cui, Longfei Li, Jun Zhou*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/tang23j.html](https://proceedings.mlr.press/v202/tang23j.html)

        **Abstract**:

        This study considers the estimation of conditional causal effects in the presence of unmeasured confounding for a balanced panel with treatment imposed at the last time point. To address this, we combine Difference-in-differences (DiD) and tree-based methods and propose a new identification assumption that allows for the violation of the (conditional) parallel trends assumption adopted by most existing DiD methods. Under this new assumption, we prove partial identifiability of the conditional average treatment effect on the treated group (CATT). Our proposed method estimates CATT through a tree-based causal approach, guided by a novel splitting rule that avoids model misspecification and unnecessary auxiliary parameter estimation. The splitting rule measures both the error of fitting observed data and the violation of conditional parallel trends simultaneously. We also develop an ensemble of multiple trees via gradient boosting to further enhance performance. Experimental results on both synthetic and real-world datasets validate the effectiveness of our proposed method.

        ----

        ## [1407] End-to-end Training of Deep Boltzmann Machines by Unbiased Contrastive Divergence with Local Mode Initialization

        **Authors**: *Shohei Taniguchi, Masahiro Suzuki, Yusuke Iwasawa, Yutaka Matsuo*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/taniguchi23a.html](https://proceedings.mlr.press/v202/taniguchi23a.html)

        **Abstract**:

        We address the problem of biased gradient estimation in deep Boltzmann machines (DBMs). The existing method to obtain an unbiased estimator uses a maximal coupling based on a Gibbs sampler, but when the state is high-dimensional, it takes a long time to converge. In this study, we propose to use a coupling based on the Metropolis-Hastings (MH) and to initialize the state around a local mode of the target distribution. Because of the propensity of MH to reject proposals, the coupling tends to converge in only one step with a high probability, leading to high efficiency. We find that our method allows DBMs to be trained in an end-to-end fashion without greedy pretraining. We also propose some practical techniques to further improve the performance of DBMs. We empirically demonstrate that our training algorithm enables DBMs to show comparable generative performance to other deep generative models, achieving the FID score of 10.33 for MNIST.

        ----

        ## [1408] POUF: Prompt-Oriented Unsupervised Fine-tuning for Large Pre-trained Models

        **Authors**: *Korawat Tanwisuth, Shujian Zhang, Huangjie Zheng, Pengcheng He, Mingyuan Zhou*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/tanwisuth23a.html](https://proceedings.mlr.press/v202/tanwisuth23a.html)

        **Abstract**:

        Through prompting, large-scale pre-trained models have become more expressive and powerful, gaining significant attention in recent years. Though these big models have zero-shot capabilities, in general, labeled data are still required to adapt them to downstream tasks. To overcome this critical limitation, we propose an unsupervised fine-tuning framework to directly fine-tune the model or prompt on the unlabeled target data. We demonstrate how to apply our method to both language-augmented vision and masked-language models, by aligning the discrete distributions extracted from the prompts and target data. To verify our approach’s applicability, we conduct extensive experiments on image classification, sentiment analysis, and natural language inference tasks. Across 13 image-related tasks and 15 language-related ones, the proposed approach achieves consistent improvements over the baselines. PyTorch code is available at https://github.com/korawat-tanwisuth/POUF.

        ----

        ## [1409] Dual Focal Loss for Calibration

        **Authors**: *Linwei Tao, Minjing Dong, Chang Xu*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/tao23a.html](https://proceedings.mlr.press/v202/tao23a.html)

        **Abstract**:

        The use of deep neural networks in real-world applications require well-calibrated networks with confidence scores that accurately reflect the actual probability. However, it has been found that these networks often provide over-confident predictions, which leads to poor calibration. Recent efforts have sought to address this issue by focal loss to reduce over-confidence, but this approach can also lead to under-confident predictions. While different variants of focal loss have been explored, it is difficult to find a balance between over-confidence and under-confidence. In our work, we propose a new loss function by focusing on dual logits. Our method not only considers the ground truth logit, but also take into account the highest logit ranked after the ground truth logit. By maximizing the gap between these two logits, our proposed dual focal loss can achieve a better balance between over-confidence and under-confidence. We provide theoretical evidence to support our approach and demonstrate its effectiveness through evaluations on multiple models and datasets, where it achieves state-of-the-art performance. Code is available at https://github.com/Linwei94/DualFocalLoss

        ----

        ## [1410] Abstract-to-Executable Trajectory Translation for One-Shot Task Generalization

        **Authors**: *Stone Tao, Xiaochen Li, Tongzhou Mu, Zhiao Huang, Yuzhe Qin, Hao Su*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/tao23b.html](https://proceedings.mlr.press/v202/tao23b.html)

        **Abstract**:

        Training long-horizon robotic policies in complex physical environments is essential for many applications, such as robotic manipulation. However, learning a policy that can generalize to unseen tasks is challenging. In this work, we propose to achieve one-shot task generalization by decoupling plan generation and plan execution. Specifically, our method solves complex long-horizon tasks in three steps: build a paired abstract environment by simplifying geometry and physics, generate abstract trajectories, and solve the original task by an abstract-to-executable trajectory translator. In the abstract environment, complex dynamics such as physical manipulation are removed, making abstract trajectories easier to generate. However, this introduces a large domain gap between abstract trajectories and the actual executed trajectories as abstract trajectories lack low-level details and are not aligned frame-to-frame with the executed trajectory. In a manner reminiscent of language translation, our approach leverages a seq-to-seq model to overcome the large domain gap between the abstract and executable trajectories, enabling the low-level policy to follow the abstract trajectory. Experimental results on various unseen long-horizon tasks with different robot embodiments demonstrate the practicability of our methods to achieve one-shot task generalization.

        ----

        ## [1411] Data Feedback Loops: Model-driven Amplification of Dataset Biases

        **Authors**: *Rohan Taori, Tatsunori Hashimoto*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/taori23a.html](https://proceedings.mlr.press/v202/taori23a.html)

        **Abstract**:

        Datasets scraped from the internet have been critical to large-scale machine learning. Yet, its success puts the utility of future internet-derived datasets at potential risk, as model outputs begin to replace human annotations as a source of supervision. In this work, we formalize a system where interactions with one model are recorded as history and scraped as training data in the future. We then analyze its stability over time by tracking changes to a test-time bias statistic (e.g. gender bias of model predictions). We find that the degree of bias amplification is closely linked to whether the model’s outputs behave like samples from the training distribution, a behavior which we characterize and define as uniform faithfulness. Experiments in three conditional prediction scenarios – image classification, visual role-labeling, and language generation – demonstrate that models that exhibit a sampling-like behavior are more faithful and thus more stable. Based on this insight, we propose an intervention to help mitigate and stabilize unstable feedback systems.

        ----

        ## [1412] Deep Regression Unlearning

        **Authors**: *Ayush Kumar Tarun, Vikram Singh Chundawat, Murari Mandal, Mohan S. Kankanhalli*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/tarun23a.html](https://proceedings.mlr.press/v202/tarun23a.html)

        **Abstract**:

        With the introduction of data protection and privacy regulations, it has become crucial to remove the lineage of data on demand from a machine learning (ML) model. In the last few years, there have been notable developments in machine unlearning to remove the information of certain training data efficiently and effectively from ML models. In this work, we explore unlearning for the regression problem, particularly in deep learning models. Unlearning in classification and simple linear regression has been considerably investigated. However, unlearning in deep regression models largely remains an untouched problem till now. In this work, we introduce deep regression unlearning methods that generalize well and are robust to privacy attacks. We propose the Blindspot unlearning method which uses a novel weight optimization process. A randomly initialized model, partially exposed to the retain samples and a copy of the original model are used together to selectively imprint knowledge about the data that we wish to keep and scrub off the information of the data we wish to forget. We also propose a Gaussian fine tuning method for regression unlearning. The existing unlearning metrics for classification are not directly applicable to regression unlearning. Therefore, we adapt these metrics for the regression setting. We conduct regression unlearning experiments for computer vision, natural language processing and forecasting applications. Our methods show excellent performance for all these datasets across all the metrics. Source code: https://github.com/ayu987/deep-regression-unlearning

        ----

        ## [1413] How to Trust Your Diffusion Model: A Convex Optimization Approach to Conformal Risk Control

        **Authors**: *Jacopo Teneggi, Matthew Tivnan, J. Webster Stayman, Jeremias Sulam*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/teneggi23a.html](https://proceedings.mlr.press/v202/teneggi23a.html)

        **Abstract**:

        Score-based generative modeling, informally referred to as diffusion models, continue to grow in popularity across several important domains and tasks. While they provide high-quality and diverse samples from empirical distributions, important questions remain on the reliability and trustworthiness of these sampling procedures for their responsible use in critical scenarios. Conformal prediction is a modern tool to construct finite-sample, distribution-free uncertainty guarantees for any black-box predictor. In this work, we focus on image-to-image regression tasks and we present a generalization of the Risk-Controlling Prediction Sets (RCPS) procedure, that we term $K$-RCPS, which allows to $(i)$ provide entrywise calibrated intervals for future samples of any diffusion model, and $(ii)$ control a certain notion of risk with respect to a ground truth image with minimal mean interval length. Differently from existing conformal risk control procedures, ours relies on a novel convex optimization approach that allows for multidimensional risk control while provably minimizing the mean interval length. We illustrate our approach on two real-world image denoising problems: on natural images of faces as well as on computed tomography (CT) scans of the abdomen, demonstrating state of the art performance.

        ----

        ## [1414] Concurrent Shuffle Differential Privacy Under Continual Observation

        **Authors**: *Jay Tenenbaum, Haim Kaplan, Yishay Mansour, Uri Stemmer*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/tenenbaum23a.html](https://proceedings.mlr.press/v202/tenenbaum23a.html)

        **Abstract**:

        We introduce the concurrent shuffle model of differential privacy. In this model we have multiple concurrent shufflers permuting messages from different, possibly overlapping, batches of users. Similarly to the standard (single) shuffler model, the privacy requirement is that the concatenation of all shuffled messages should be differentially private. We study the private continual summation problem (a.k.a. the counter problem) and show that the concurrent shuffle model allows for significantly improved error compared to a standard (single) shuffler model. Specifically, we give a summation algorithm with error $\tilde{O}(n^{1/(2k+1)})$ with $k$ concurrent shufflers on a sequence of length $n$. Furthermore, we prove that this bound is tight for any $k$, even if the algorithm can choose the sizes of the batches adaptively. For $k=\log n$ shufflers, the resulting error is polylogarithmic, much better than $\tilde{\Theta}(n^{1/3})$ which we show is the smallest possible with a single shuffler. We use our online summation algorithm to get algorithms with improved regret bounds for the contextual linear bandit problem. In particular we get optimal $\tilde{O}(\sqrt{n})$ regret with $k= \tilde{\Omega}(\log n)$ concurrent shufflers.

        ----

        ## [1415] Finding Generalization Measures by Contrasting Signal and Noise

        **Authors**: *Jiaye Teng, Bohang Zhang, Ruichen Li, Haowei He, Yequan Wang, Yan Tian, Yang Yuan*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/teng23a.html](https://proceedings.mlr.press/v202/teng23a.html)

        **Abstract**:

        Generalization is one of the most fundamental challenges in deep learning, aiming to predict model performances on unseen data. Empirically, such predictions usually rely on a validation set, while recent works showed that an unlabeled validation set also works. Without validation sets, it is extremely difficult to obtain non-vacuous generalization bounds, which leads to a weaker task of finding generalization measures that monotonically relate to generalization error. In this paper, we propose a new generalization measure REF Complexity (RElative Fitting degree between signal and noise), motivated by the intuition that a given model-algorithm pair may generalize well if it fits signal (e.g., true labels) fast while fitting noise (e.g., random labels) slowly. Empirically, REF Complexity monotonically relates to test accuracy in real-world datasets without accessing additional validation sets, achieving -0.988 correlation on CIFAR-10 and -0.960 correlation on CIFAR-100. We further theoretically verify the utility of REF Complexity under three different cases, including convex and smooth regimes with stochastic gradient descent, smooth regimes (not necessarily convex) with stochastic gradient Langevin dynamics, and linear regimes with gradient descent. The code is available at https://github.com/962086838/REF-complexity.

        ----

        ## [1416] Reinforcement Learning with History Dependent Dynamic Contexts

        **Authors**: *Guy Tennenholtz, Nadav Merlis, Lior Shani, Martin Mladenov, Craig Boutilier*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/tennenholtz23a.html](https://proceedings.mlr.press/v202/tennenholtz23a.html)

        **Abstract**:

        We introduce Dynamic Contextual Markov Decision Processes (DCMDPs), a novel reinforcement learning framework for history-dependent environments that generalizes the contextual MDP framework to handle non-Markov environments, where contexts change over time. We consider special cases of the model, with a focus on logistic DCMDPs, which break the exponential dependence on history length by leveraging aggregation functions to determine context transitions. This special structure allows us to derive an upper-confidence-bound style algorithm for which we establish regret bounds. Motivated by our theoretical results, we introduce a practical model-based algorithm for logistic DCMDPs that plans in a latent space and uses optimism over history-dependent features. We demonstrate the efficacy of our approach on a recommendation task (using MovieLens data) where user behavior dynamics evolve in response to recommendations.

        ----

        ## [1417] PWSHAP: A Path-Wise Explanation Model for Targeted Variables

        **Authors**: *Lucile Ter-Minassian, Oscar Clivio, Karla DiazOrdaz, Robin J. Evans, Christopher C. Holmes*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/ter-minassian23a.html](https://proceedings.mlr.press/v202/ter-minassian23a.html)

        **Abstract**:

        Predictive black-box models can exhibit high-accuracy but their opaque nature hinders their uptake in safety-critical deployment environments. Explanation methods (XAI) can provide confidence for decision-making through increased transparency. However, existing XAI methods are not tailored towards models in sensitive domains where one predictor is of special interest, such as a treatment effect in a clinical model, or ethnicity in policy models. We introduce Path-Wise Shapley effects (PWSHAP), a framework for assessing the targeted effect of a binary (e.g. treatment) variable from a complex outcome model. Our approach augments the predictive model with a user-defined directed acyclic graph (DAG). The method then uses the graph alongside on-manifold Shapley values to identify effects along causal pathways whilst maintaining robustness to adversarial attacks. We establish error bounds for the identified path-wise Shapley effects and for Shapley values. We show PWSHAP can perform local bias and mediation analyses with faithfulness to the model. Further, if the targeted variable is randomised we can quantify local effect modification. We demonstrate the resolution, interpretability and true locality of our approach on examples and a real-world experiment.

        ----

        ## [1418] On the Estimation of Gaussian Mixture Copula Models

        **Authors**: *Ashutosh Tewari*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/tewari23a.html](https://proceedings.mlr.press/v202/tewari23a.html)

        **Abstract**:

        This paper revisits Gaussian Mixture Copula Model (GMCM), a more expressive alternative to the widely used Gaussian Mixture Model (GMM), with the goal to make its parameter estimation tractable. Both the Expectation Maximization and the direct Likelihood Maximization frameworks for GMCM have to grapple with a likelihood function that lacks a closed form. This has led to a few approximation schemes that alleviate the problem, nonetheless leaving the issue still unresolved. Additionally, past works have alluded to an additional challenge of parameter non-identifiability, but none has offered a rigorous treatment and a commensurate solution framework to overcome the same. This work offers solutions to each of these issues in an attempt to help GMCM realize its full potential. The source of non-identifiability is not only proven but also suitable priors are proposed that eliminate the problem. Additionally, an efficient numerical framework is proposed to evaluate the intractable likelihood function, while also providing its analytical derivatives. Finally, a view of GMCM as a series of bijective mappings from a base distribution is presented, which paves the way to synthesize GMCM using modern, probabilistic programming languages (PPLs). The main claims of this work are supported by empirical evidence gathered on synthetic and real-world datasets.

        ----

        ## [1419] Target-Aware Generative Augmentations for Single-Shot Adaptation

        **Authors**: *Kowshik Thopalli, Rakshith Subramanyam, Pavan K. Turaga, Jayaraman J. Thiagarajan*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/thopalli23a.html](https://proceedings.mlr.press/v202/thopalli23a.html)

        **Abstract**:

        In this paper, we address the problem of adapting models from a source domain to a target domain, a task that has become increasingly important due to the brittle generalization of deep neural networks. While several test-time adaptation techniques have emerged, they typically rely on synthetic toolbox data augmentations in cases of limited target data availability. We consider the challenging setting of single-shot adaptation and explore the design of augmentation strategies. We argue that augmentations utilized by existing methods are insufficient to handle large distribution shifts, and hence propose a new approach SiSTA, which first fine-tunes a generative model from the source domain using a single-shot target, and then employs novel sampling strategies for curating synthetic target data. Using experiments on a variety of benchmarks, distribution shifts and image corruptions, we find that SiSTA produces significantly improved generalization over existing baselines in face attribute detection and multi-class object recognition. Furthermore, SiSTA performs competitively to models obtained by training on larger target datasets. Our codes can be accessed at https://github.com/Rakshith-2905/SiSTA

        ----

        ## [1420] ELSA: Efficient Label Shift Adaptation through the Lens of Semiparametric Models

        **Authors**: *Qinglong Tian, Xin Zhang, Jiwei Zhao*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/tian23a.html](https://proceedings.mlr.press/v202/tian23a.html)

        **Abstract**:

        We study the domain adaptation problem with label shift in this work. Under the label shift context, the marginal distribution of the label varies across the training and testing datasets, while the conditional distribution of features given the label is the same. Traditional label shift adaptation methods either suffer from large estimation errors or require cumbersome post-prediction calibrations. To address these issues, we first propose a moment-matching framework for adapting the label shift based on the geometry of the influence function. Under such a framework, we propose a novel method named $\underline{\mathrm{E}}$fficient $\underline{\mathrm{L}}$abel $\underline{\mathrm{S}}$hift $\underline{\mathrm{A}}$daptation (ELSA), in which the adaptation weights can be estimated by solving linear systems. Theoretically, the ELSA estimator is $\sqrt{n}$-consistent ($n$ is the sample size of the source data) and asymptotically normal. Empirically, we show that ELSA can achieve state-of-the-art estimation performances without post-prediction calibrations, thus, gaining computational efficiency.

        ----

        ## [1421] Spherical Inducing Features for Orthogonally-Decoupled Gaussian Processes

        **Authors**: *Louis C. Tiao, Vincent Dutordoir, Victor Picheny*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/tiao23a.html](https://proceedings.mlr.press/v202/tiao23a.html)

        **Abstract**:

        Despite their many desirable properties, Gaussian processes (GPs) are often compared unfavorably to deep neural networks (NNs) for lacking the ability to learn representations. Recent efforts to bridge the gap between GPs and deep NNs have yielded a new class of inter-domain variational GPs in which the inducing variables correspond to hidden units of a feedforward NN. In this work, we examine some practical issues associated with this approach and propose an extension that leverages the orthogonal decomposition of GPs to mitigate these limitations. In particular, we introduce spherical inter-domain features to construct more flexible data-dependent basis functions for both the principal and orthogonal components of the GP approximation and show that incorporating NN activation features under this framework not only alleviates these shortcomings but is more scalable than alternative strategies. Experiments on multiple benchmark datasets demonstrate the effectiveness of our approach.

        ----

        ## [1422] Fast Rates for Maximum Entropy Exploration

        **Authors**: *Daniil Tiapkin, Denis Belomestny, Daniele Calandriello, Eric Moulines, Rémi Munos, Alexey Naumov, Pierre Perrault, Yunhao Tang, Michal Valko, Pierre Ménard*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/tiapkin23a.html](https://proceedings.mlr.press/v202/tiapkin23a.html)

        **Abstract**:

        We address the challenge of exploration in reinforcement learning (RL) when the agent operates in an unknown environment with sparse or no rewards. In this work, we study the maximum entropy exploration problem of two different types. The first type is visitation entropy maximization previously considered by Hazan et al. (2019) in the discounted setting. For this type of exploration, we propose a game-theoretic algorithm that has $\widetilde{\mathcal{O}}(H^3S^2A/\varepsilon^2)$ sample complexity thus improving the $\varepsilon$-dependence upon existing results, where $S$ is a number of states, $A$ is a number of actions, $H$ is an episode length, and $\varepsilon$ is a desired accuracy. The second type of entropy we study is the trajectory entropy. This objective function is closely related to the entropy-regularized MDPs, and we propose a simple algorithm that has a sample complexity of order $\widetilde{\mathcal{O}}(\mathrm{poly}(S,A,H)/\varepsilon)$. Interestingly, it is the first theoretical result in RL literature that establishes the potential statistical advantage of regularized MDPs for exploration. Finally, we apply developed regularization techniques to reduce sample complexity of visitation entropy maximization to $\widetilde{\mathcal{O}}(H^2SA/\varepsilon^2)$, yielding a statistical separation between maximum entropy exploration and reward-free exploration.

        ----

        ## [1423] Margin-based sampling in high dimensions: When being active is less efficient than staying passive

        **Authors**: *Alexandru Tifrea, Jacob Clarysse, Fanny Yang*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/tifrea23a.html](https://proceedings.mlr.press/v202/tifrea23a.html)

        **Abstract**:

        It is widely believed that given the same labeling budget, active learning (AL) algorithms like margin-based active learning achieve better predictive performance than passive learning (PL), albeit at a higher computational cost. Recent empirical evidence suggests that this added cost might be in vain, as margin-based AL can sometimes perform even worse than PL. While existing works offer different explanations in the low-dimensional regime, this paper shows that the underlying mechanism is entirely different in high dimensions: we prove for logistic regression that PL outperforms margin-based AL even for noiseless data and when using the Bayes optimal decision boundary for sampling. Insights from our proof indicate that this high-dimensional phenomenon is exacerbated when the separation between the classes is small. We corroborate this intuition with experiments on 20 high-dimensional datasets spanning a diverse range of applications, from finance and histology to chemistry and computer vision.

        ----

        ## [1424] Differentiable Multi-Target Causal Bayesian Experimental Design

        **Authors**: *Panagiotis Tigas, Yashas Annadani, Desi R. Ivanova, Andrew Jesson, Yarin Gal, Adam Foster, Stefan Bauer*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/tigas23a.html](https://proceedings.mlr.press/v202/tigas23a.html)

        **Abstract**:

        We introduce a gradient-based approach for the problem of Bayesian optimal experimental design to learn causal models in a batch setting — a critical component for causal discovery from finite data where interventions can be costly or risky. Existing methods rely on greedy approximations to construct a batch of experiments while using black-box methods to optimize over a single target-state pair to intervene with. In this work, we completely dispose of the black-box optimization techniques and greedy heuristics and instead propose a conceptually simple end-to-end gradient-based optimization procedure to acquire a set of optimal intervention target-value pairs. Such a procedure enables parameterization of the design space to efficiently optimize over a batch of multi-target-state interventions, a setting which has hitherto not been explored due to its complexity. We demonstrate that our proposed method outperforms baselines and existing acquisition strategies in both single-target and multi-target settings across a number of synthetic datasets.

        ----

        ## [1425] PCA-based Multi-Task Learning: a Random Matrix Approach

        **Authors**: *Malik Tiomoko, Romain Couillet, Frédéric Pascal*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/tiomoko23a.html](https://proceedings.mlr.press/v202/tiomoko23a.html)

        **Abstract**:

        The article proposes and theoretically analyses a computationally efficient multi-task learning (MTL) extension of popular principal component analysis (PCA)-based supervised learning schemes. The analysis reveals that (i) by default, learning may dramatically fail by suffering from negative transfer, but that (ii) simple counter-measures on data labels avert negative transfer and necessarily result in improved performances. Supporting experiments on synthetic and real data benchmarks show that the proposed method achieves comparable performance with state-of-the-art MTL methods but at a significantly reduced computational cost.

        ----

        ## [1426] Perturbation Analysis of Neural Collapse

        **Authors**: *Tom Tirer, Haoxiang Huang, Jonathan Niles-Weed*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/tirer23a.html](https://proceedings.mlr.press/v202/tirer23a.html)

        **Abstract**:

        Training deep neural networks for classification often includes minimizing the training loss beyond the zero training error point. In this phase of training, a "neural collapse" behavior has been observed: the variability of features (outputs of the penultimate layer) of within-class samples decreases and the mean features of different classes approach a certain tight frame structure. Recent works analyze this behavior via idealized unconstrained features models where all the minimizers exhibit exact collapse. However, with practical networks and datasets, the features typically do not reach exact collapse, e.g., because deep layers cannot arbitrarily modify intermediate features that are far from being collapsed. In this paper, we propose a richer model that can capture this phenomenon by forcing the features to stay in the vicinity of a predefined features matrix (e.g., intermediate features). We explore the model in the small vicinity case via perturbation analysis and establish results that cannot be obtained by the previously studied models. For example, we prove reduction in the within-class variability of the optimized features compared to the predefined input features (via analyzing gradient flow on the "central-path" with minimal assumptions), analyze the minimizers in the near-collapse regime, and provide insights on the effect of regularization hyperparameters on the closeness to collapse. We support our theory with experiments in practical deep learning settings.

        ----

        ## [1427] Overcoming Simplicity Bias in Deep Networks using a Feature Sieve

        **Authors**: *Rishabh Tiwari, Pradeep Shenoy*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/tiwari23a.html](https://proceedings.mlr.press/v202/tiwari23a.html)

        **Abstract**:

        Simplicity bias is the concerning tendency of deep networks to over-depend on simple, weakly predictive features, to the exclusion of stronger, more complex features. This causes biased, incorrect model predictions in many real-world applications, exacerbated by incomplete training data containing spurious feature-label correlations. We propose a direct, interventional method for addressing simplicity bias in DNNs, which we call the feature sieve. We aim to automatically identify and suppress easily-computable spurious features in lower layers of the network, thereby allowing the higher network levels to extract and utilize richer, more meaningful representations. We provide concrete evidence of this differential suppression & enhancement of relevant features on both controlled datasets and real-world images, and report substantial gains on many real-world debiasing benchmarks (11.4% relative gain on Imagenet-A; 3.2% on BAR, etc). Crucially, we outperform many baselines that incorporate knowledge about known spurious or biased attributes, despite our method not using any such information. We believe that our feature sieve work opens up exciting new research directions in automated adversarial feature extraction & representation learning for deep networks.

        ----

        ## [1428] Beyond In-Domain Scenarios: Robust Density-Aware Calibration

        **Authors**: *Christian Tomani, Futa Kai Waseda, Yuesong Shen, Daniel Cremers*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/tomani23a.html](https://proceedings.mlr.press/v202/tomani23a.html)

        **Abstract**:

        Calibrating deep learning models to yield uncertainty-aware predictions is crucial as deep neural networks get increasingly deployed in safety-critical applications. While existing post-hoc calibration methods achieve impressive results on in-domain test datasets, they are limited by their inability to yield reliable uncertainty estimates in domain-shift and out-of-domain (OOD) scenarios. We aim to bridge this gap by proposing DAC, an accuracy-preserving as well as Density-Aware Calibration method based on k-nearest-neighbors (KNN). In contrast to existing post-hoc methods, we utilize hidden layers of classifiers as a source for uncertainty-related information and study their importance. We show that DAC is a generic method that can readily be combined with state-of-the-art post-hoc methods. DAC boosts the robustness of calibration performance in domain-shift and OOD, while maintaining excellent in-domain predictive uncertainty estimates. We demonstrate that DAC leads to consistently better calibration across a large number of model architectures, datasets, and metrics. Additionally, we show that DAC improves calibration substantially on recent large-scale neural networks pre-trained on vast amounts of data.

        ----

        ## [1429] Distribution Free Domain Generalization

        **Authors**: *Peifeng Tong, Wu Su, He Li, Jialin Ding, Zhan Haoxiang, Song Xi Chen*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/tong23a.html](https://proceedings.mlr.press/v202/tong23a.html)

        **Abstract**:

        Accurate prediction of the out-of-distribution data is desired for a learning algorithm. In domain generalization, training data from source domains tend to have different distributions from that of the target domain, while the target data are absence in the training process. We propose a Distribution Free Domain Generalization (DFDG) procedure for classification by conducting standardization to avoid the dominance of a few domains in the training process. The essence of the DFDG is its reformulating the cross domain/class discrepancy by pairwise two sample test statistics, and equally weights their importance or the covariance structures to avoid dominant domain/class. A theoretical generalization bound is established for the multi-class classification problem. The DFDG is shown to offer a superior performance in empirical studies with fewer hyperparameters, which means faster and easier implementation.

        ----

        ## [1430] Extending Kernel PCA through Dualization: Sparsity, Robustness and Fast Algorithms

        **Authors**: *Francesco Tonin, Alex Lambert, Panagiotis Patrinos, Johan A. K. Suykens*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/tonin23a.html](https://proceedings.mlr.press/v202/tonin23a.html)

        **Abstract**:

        The goal of this paper is to revisit Kernel Principal Component Analysis (KPCA) through dualization of a difference of convex functions. This allows to naturally extend KPCA to multiple objective functions and leads to efficient gradient-based algorithms avoiding the expensive SVD of the Gram matrix. Particularly, we consider objective functions that can be written as Moreau envelopes, demonstrating how to promote robustness and sparsity within the same framework. The proposed method is evaluated on synthetic and realworld benchmarks, showing significant speedup in KPCA training time as well as highlighting the benefits in terms of robustness and sparsity.

        ----

        ## [1431] Robust Weak Supervision with Variational Auto-Encoders

        **Authors**: *Francesco Tonolini, Nikolaos Aletras, Yunlong Jiao, Gabriella Kazai*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/tonolini23a.html](https://proceedings.mlr.press/v202/tonolini23a.html)

        **Abstract**:

        Recent advances in weak supervision (WS) techniques allow to mitigate the enormous cost and effort of human data annotation for supervised machine learning by automating it using simple rule-based labelling functions (LFs). However, LFs need to be carefully designed, often requiring expert domain knowledge and extensive validation for existing WS methods to be effective. To tackle this, we propose the Weak Supervision Variational Auto-Encoder (WS-VAE), a novel framework that combines unsupervised representation learning and weak labelling to reduce the dependence of WS on expert and manual engineering of LFs. Our technique learns from inputs and weak labels jointly to capture the input signals distribution with a latent space. The unsupervised representation component of the WS-VAE regularises the inference of weak labels, while a specifically designed decoder allows the model to learn the relevance of LFs for each input. These unique features lead to considerably improved robustness to the quality of LFs, compared to existing methods. An extensive empirical evaluation on a standard WS benchmark shows that our WS-VAE is competitive to state-of-the-art methods and substantially more robust to LF engineering.

        ----

        ## [1432] Fully Bayesian Autoencoders with Latent Sparse Gaussian Processes

        **Authors**: *Ba-Hien Tran, Babak Shahbaba, Stephan Mandt, Maurizio Filippone*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/tran23a.html](https://proceedings.mlr.press/v202/tran23a.html)

        **Abstract**:

        We present a fully Bayesian autoencoder model that treats both local latent variables and global decoder parameters in a Bayesian fashion. This approach allows for flexible priors and posterior approximations while keeping the inference costs low. To achieve this, we introduce an amortized MCMC approach by utilizing an implicit stochastic network to learn sampling from the posterior over local latent variables. Furthermore, we extend the model by incorporating a Sparse Gaussian Process prior over the latent space, allowing for a fully Bayesian treatment of inducing points and kernel hyperparameters and leading to improved scalability. Additionally, we enable Deep Gaussian Process priors on the latent space and the handling of missing data. We evaluate our model on a range of experiments focusing on dynamic representation learning and generative modeling, demonstrating the strong performance of our approach in comparison to existing methods that combine Gaussian Processes and autoencoders.

        ----

        ## [1433] Discrete Key-Value Bottleneck

        **Authors**: *Frederik Träuble, Anirudh Goyal, Nasim Rahaman, Michael Curtis Mozer, Kenji Kawaguchi, Yoshua Bengio, Bernhard Schölkopf*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/trauble23a.html](https://proceedings.mlr.press/v202/trauble23a.html)

        **Abstract**:

        Deep neural networks perform well on classification tasks where data streams are i.i.d. and labeled data is abundant. Challenges emerge with non-stationary training data streams such as continual learning. One powerful approach that has addressed this challenge involves pre-training of large encoders on volumes of readily available data, followed by task-specific tuning. Given a new task, however, updating the weights of these encoders is challenging as a large number of weights needs to be fine-tuned, and as a result, they forget information about the previous tasks. In the present work, we propose a model architecture to address this issue, building upon a discrete bottleneck containing pairs of separate and learnable key-value codes. Our paradigm will be to encode; process the representation via a discrete bottleneck; and decode. Here, the input is fed to the pre-trained encoder, the output of the encoder is used to select the nearest keys, and the corresponding values are fed to the decoder to solve the current task. The model can only fetch and re-use a sparse number of these key-value pairs during inference, enabling localized and context-dependent model updates. We theoretically investigate the ability of the discrete key-value bottleneck to minimize the effect of learning under distribution shifts and show that it reduces the complexity of the hypothesis class. We empirically verify the proposed method under challenging class-incremental learning scenarios and show that the proposed model — without any task boundaries — reduces catastrophic forgetting across a wide variety of pre-trained models, outperforming relevant baselines on this task.

        ----

        ## [1434] Mimetic Initialization of Self-Attention Layers

        **Authors**: *Asher Trockman, J. Zico Kolter*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/trockman23a.html](https://proceedings.mlr.press/v202/trockman23a.html)

        **Abstract**:

        It is notoriously difficult to train Transformers on small datasets; typically, large pre-trained models are instead used as the starting point. We explore the weights of such pre-trained Transformers (particularly for vision) to attempt to find reasons for this discrepancy. Surprisingly, we find that simply initializing the weights of self-attention layers so that they "look" more like their pre-trained counterparts allows us to train vanilla Transformers faster and to higher final accuracies, particularly on vision tasks such as CIFAR-10 and ImageNet classification, where we see gains in accuracy of over 5% and 4%, respectively. Our initialization scheme is closed form, learning-free, and very simple: we set the product of the query and key weights to be approximately the identity, and the product of the value and projection weights to approximately the negative identity. As this mimics the patterns we saw in pre-trained Transformers, we call the technique "mimetic initialization".

        ----

        ## [1435] Representer Point Selection for Explaining Regularized High-dimensional Models

        **Authors**: *Che-Ping Tsai, Jiong Zhang, Hsiang-Fu Yu, Eli Chien, Cho-Jui Hsieh, Pradeep Kumar Ravikumar*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/tsai23a.html](https://proceedings.mlr.press/v202/tsai23a.html)

        **Abstract**:

        We introduce a novel class of sample-based explanations we term high-dimensional representers, that can be used to explain the predictions of a regularized high-dimensional model in terms of importance weights for each of the training samples. Our workhorse is a novel representer theorem for general regularized high-dimensional models, which decomposes the model prediction in terms of contributions from each of the training samples: with positive (negative) values corresponding to positive (negative) impact training samples to the model’s prediction. We derive consequences for the canonical instances of $\ell_1$ regularized sparse models and nuclear norm regularized low-rank models. As a case study, we further investigate the application of low-rank models in the context of collaborative filtering, where we instantiate high-dimensional representers for specific popular classes of models. Finally, we study the empirical performance of our proposed methods on three real-world binary classification datasets and two recommender system datasets. We also showcase the utility of high-dimensional representers in explaining model recommendations.

        ----

        ## [1436] Expected Gradients of Maxout Networks and Consequences to Parameter Initialization

        **Authors**: *Hanna Tseran, Guido Montúfar*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/tseran23a.html](https://proceedings.mlr.press/v202/tseran23a.html)

        **Abstract**:

        We study the gradients of a maxout network with respect to inputs and parameters and obtain bounds for the moments depending on the architecture and the parameter distribution. We observe that the distribution of the input-output Jacobian depends on the input, which complicates a stable parameter initialization. Based on the moments of the gradients, we formulate parameter initialization strategies that avoid vanishing and exploding gradients in wide networks. Experiments with deep fully-connected and convolutional networks show that this strategy improves SGD and Adam training of deep maxout networks. In addition, we obtain refined bounds on the expected number of linear regions, results on the expected curve length distortion, and results on the NTK.

        ----

        ## [1437] Provable Data Subset Selection For Efficient Neural Networks Training

        **Authors**: *Murad Tukan, Samson Zhou, Alaa Maalouf, Daniela Rus, Vladimir Braverman, Dan Feldman*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/tukan23a.html](https://proceedings.mlr.press/v202/tukan23a.html)

        **Abstract**:

        Radial basis function neural networks (RBFNN) are well-known for their capability to approximate any continuous function on a closed bounded set with arbitrary precision given enough hidden neurons. In this paper, we introduce the first algorithm to construct coresets for RBFNNs, i.e., small weighted subsets that approximate the loss of the input data on any radial basis function network and thus approximate any function defined by an RBFNN on the larger input data. In particular, we construct coresets for radial basis and Laplacian loss functions. We then use our coresets to obtain a provable data subset selection algorithm for training deep neural networks. Since our coresets approximate every function, they also approximate the gradient of each weight in a neural network, which is a particular function on the input. We then perform empirical evaluations on function approximation and dataset subset selection on popular network architectures and data sets, demonstrating the efficacy and accuracy of our coreset construction.

        ----

        ## [1438] Jump-Start Reinforcement Learning

        **Authors**: *Ikechukwu Uchendu, Ted Xiao, Yao Lu, Banghua Zhu, Mengyuan Yan, Joséphine Simon, Matthew Bennice, Chuyuan Fu, Cong Ma, Jiantao Jiao, Sergey Levine, Karol Hausman*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/uchendu23a.html](https://proceedings.mlr.press/v202/uchendu23a.html)

        **Abstract**:

        Reinforcement learning (RL) provides a theoretical framework for continuously improving an agent’s behavior via trial and error. However, efficiently learning policies from scratch can be very difficult, particularly for tasks that present exploration challenges. In such settings, it might be desirable to initialize RL with an existing policy, offline data, or demonstrations. However, naively performing such initialization in RL often works poorly, especially for value-based methods. In this paper, we present a meta algorithm that can use offline data, demonstrations, or a pre-existing policy to initialize an RL policy, and is compatible with any RL approach. In particular, we propose Jump-Start Reinforcement Learning (JSRL), an algorithm that employs two policies to solve tasks: a guide-policy, and an exploration-policy. By using the guide-policy to form a curriculum of starting states for the exploration-policy, we are able to efficiently improve performance on a set of simulated robotic tasks. We show via experiments that it is able to significantly outperform existing imitation and reinforcement learning algorithms, particularly in the small-data regime. In addition, we provide an upper bound on the sample complexity of JSRL and show that with the help of a guide-policy, one can improve the sample complexity for non-optimism exploration methods from exponential in horizon to polynomial.

        ----

        ## [1439] Submodular Order Functions and Assortment Optimization

        **Authors**: *Rajan Udwani*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/udwani23a.html](https://proceedings.mlr.press/v202/udwani23a.html)

        **Abstract**:

        We define a new class of set functions that in addition to being monotone and subadditive, also admit a very limited form of submodularity defined over a permutation of the ground set. We refer to this permutation as a submodular order. We give fast algorithms with strong approximation guarantees for maximizing submodular order functions under a variety of constraints. Applying this new notion to the problem of constrained assortment optimization in fundamental choice models, we obtain new algorithms that are both faster and have stronger approximation guarantees (in some cases, first algorithm with constant factor guarantee). We also show an intriguing connection to the maximization of monotone submodular functions in the streaming model, where we recover best known approximation guarantees as a corollary of our results.

        ----

        ## [1440] Computationally Efficient PAC RL in POMDPs with Latent Determinism and Conditional Embeddings

        **Authors**: *Masatoshi Uehara, Ayush Sekhari, Jason D. Lee, Nathan Kallus, Wen Sun*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/uehara23a.html](https://proceedings.mlr.press/v202/uehara23a.html)

        **Abstract**:

        We study reinforcement learning with function approximation for large-scale Partially Observable Markov Decision Processes (POMDPs) where the state space and observation space are large or even continuous. Particularly, we consider Hilbert space embeddings of POMDP where the feature of latent states and the feature of observations admit a conditional Hilbert space embedding of the observation emission process, and the latent state transition is deterministic. Under the function approximation setup where the optimal latent state-action $Q$-function is linear in the state feature, and the optimal $Q$-function has a gap in actions, we provide a computationally and statistically efficient algorithm for finding the exact optimal policy. We show our algorithm’s computational and statistical complexities scale polynomially with respect to the horizon and the intrinsic dimension of the feature on the observation space. Furthermore, we show both the deterministic latent transitions and gap assumptions are necessary to avoid statistical complexity exponential in horizon or dimension. Since our guarantee does not have an explicit dependence on the size of the state and observation spaces, our algorithm provably scales to large-scale POMDPs.

        ----

        ## [1441] From Adaptive Query Release to Machine Unlearning

        **Authors**: *Enayat Ullah, Raman Arora*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/ullah23a.html](https://proceedings.mlr.press/v202/ullah23a.html)

        **Abstract**:

        We formalize the problem of machine unlearning as design of efficient unlearning algorithms corresponding to learning algorithms which perform a selection of adaptive queries from structured query classes. We give efficient unlearning algorithms for linear and prefix-sum query classes. As applications, we show that unlearning in many problems, in particular, stochastic convex optimization (SCO), can be reduced to the above, yielding improved guarantees for the problem. In particular, for smooth Lipschitz losses and any $\rho>0$, our results yield an unlearning algorithm with excess population risk of $\tilde O\big(\frac{1}{\sqrt{n}}+\frac{\sqrt{d}}{n\rho}\big)$ with unlearning query (gradient) complexity $\tilde O(\rho \cdot \text{Retraining Complexity})$, where $d$ is the model dimensionality and $n$ is the initial number of samples. For non-smooth Lipschitz losses, we give an unlearning algorithm with excess population risk $\tilde O\big(\frac{1}{\sqrt{n}}+\big(\frac{\sqrt{d}}{n\rho}\big)^{1/2}\big)$ with the same unlearning query (gradient) complexity. Furthermore, in the special case of Generalized Linear Models (GLMs), such as those in linear and logistic regression, we get dimension-independent rates of $\tilde O\big(\frac{1}{\sqrt{n}} +\frac{1}{(n\rho)^{2/3}}\big)$ and $\tilde O\big(\frac{1}{\sqrt{n}} +\frac{1}{(n\rho)^{1/3}}\big)$ for smooth Lipschitz and non-smooth Lipschitz losses respectively. Finally, we give generalizations of the above from one unlearning request to dynamic streams consisting of insertions and deletions.

        ----

        ## [1442] Private Federated Learning with Autotuned Compression

        **Authors**: *Enayat Ullah, Christopher A. Choquette-Choo, Peter Kairouz, Sewoong Oh*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/ullah23b.html](https://proceedings.mlr.press/v202/ullah23b.html)

        **Abstract**:

        We propose new techniques for reducing communication in private federated learning without the need for setting or tuning compression rates. Our on-the-fly methods automatically adjust the compression rate based on the error induced during training, while maintaining provable privacy guarantees through the use of secure aggregation and differential privacy. Our techniques are provably instance-optimal for mean estimation, meaning that they can adapt to the “hardness of the problem” with minimal interactivity. We demonstrate the effectiveness of our approach on real-world datasets by achieving favorable compression rates without the need for tuning.

        ----

        ## [1443] The Monge Gap: A Regularizer to Learn All Transport Maps

        **Authors**: *Théo Uscidda, Marco Cuturi*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/uscidda23a.html](https://proceedings.mlr.press/v202/uscidda23a.html)

        **Abstract**:

        Optimal transport (OT) theory has been used in machine learning to study and characterize maps that can push-forward efficiently a probability measure onto another. Recent works have drawn inspiration from Brenier’s theorem, which states that when the ground cost is the squared-Euclidean distance, the “best” map to morph a continuous measure in $\mathcal{P}(\mathbb{R}^d)$ into another must be the gradient of a convex function. To exploit that result, Makkuva et. al (2020); Korotin et. al (2020) consider maps $T=\nabla f_\theta$, where $f_\theta$ is an input convex neural network (ICNN), as defined by Amos et. al (2017), and fit $\theta$ with SGD using samples. Despite their mathematical elegance, fitting OT maps with ICNNs raises many challenges, due notably to the many constraints imposed on $\theta$; the need to approximate the conjugate of $f_\theta$; or the limitation that they only work for the squared-Euclidean cost. More generally, we question the relevance of using Brenier’s result, which only applies to densities, to constrain the architecture of candidate maps fitted on samples. Motivated by these limitations, we propose a radically different approach to estimating OT maps: Given a cost $c$ and a reference measure $\rho$, we introduce a regularizer, the Monge gap $\mathcal{M}^c_{\rho}(T)$ of a map $T$. That gap quantifies how far a map $T$ deviates from the ideal properties we expect from a $c$-OT map. In practice, we drop all architecture requirements for $T$ and simply minimize a distance (e.g., the Sinkhorn divergence) between $T\sharp\mu$ and $\nu$, regularized by $\mathcal{M}^c_\rho(T)$. We study $\mathcal{M}^c_{\rho}$ and show how our simple pipeline significantly outperforms other baselines in practice.

        ----

        ## [1444] Semi-Dual Unbalanced Quadratic Optimal Transport: fast statistical rates and convergent algorithm

        **Authors**: *Adrien Vacher, François-Xavier Vialard*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/vacher23a.html](https://proceedings.mlr.press/v202/vacher23a.html)

        **Abstract**:

        In this paper, we derive a semi-dual formulation for the problem of unbalanced quadratic optimal transport and we study its stability properties, namely we give upper and lower bounds for the Bregman divergence of the new objective that hold globally. We observe that the new objective gains even more convexity than in the balanced case. We use this formulation to prove the first results on statistical estimation of UOT potentials and we leverage the extra convexity to recover super-parametric rates. Interestingly, unlike in the balanced case, we do not require the potentials to be smooth. Then, use variable metric descent to solve the semi-dual problem for which we prove convergence at a $1/k$ rate for strongly convex potentials and exponential convergence in the balanced case when potentials are also smooth. We emphasize that our convergence results has an interest on its own as it generalizes previous convergence results to non-equivalent metrics. Last, we instantiate a proof-of-concept tractable version of our theoretical algorithm that we benchmark on a 2D experiment in the balanced case and on a medium dimension synthetic experiment in the unbalanced case.

        ----

        ## [1445] Random Grid Neural Processes for Parametric Partial Differential Equations

        **Authors**: *Arnaud Vadeboncoeur, Ieva Kazlauskaite, Yanni Papandreou, Fehmi Cirak, Mark Girolami, Ömer Deniz Akyildiz*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/vadeboncoeur23a.html](https://proceedings.mlr.press/v202/vadeboncoeur23a.html)

        **Abstract**:

        We introduce a new class of spatially stochastic physics and data informed deep latent models for parametric partial differential equations (PDEs) which operate through scalable variational neural processes. We achieve this by assigning probability measures to the spatial domain, which allows us to treat collocation grids probabilistically as random variables to be marginalised out. Adapting this spatial statistics view, we solve forward and inverse problems for parametric PDEs in a way that leads to the construction of Gaussian process models of solution fields. The implementation of these random grids poses a unique set of challenges for inverse physics informed deep learning frameworks and we propose a new architecture called Grid Invariant Convolutional Networks (GICNets) to overcome these challenges. We further show how to incorporate noisy data in a principled manner into our physics informed model to improve predictions for problems where data may be available but whose measurement location does not coincide with any fixed mesh or grid. The proposed method is tested on a nonlinear Poisson problem, Burgers equation, and Navier-Stokes equations, and we provide extensive numerical comparisons. We demonstrate significant computational advantages over current physics informed neural learning methods for parametric PDEs while improving the predictive capabilities and flexibility of these models.

        ----

        ## [1446] Delayed Feedback in Kernel Bandits

        **Authors**: *Sattar Vakili, Danyal Ahmed, Alberto Bernacchia, Ciara Pike-Burke*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/vakili23a.html](https://proceedings.mlr.press/v202/vakili23a.html)

        **Abstract**:

        Black box optimisation of an unknown function from expensive and noisy evaluations is a ubiquitous problem in machine learning, academic research and industrial production. An abstraction of the problem can be formulated as a kernel based bandit problem (also known as Bayesian optimisation), where a learner aims at optimising a kernelized function through sequential noisy observations. The existing work predominantly assumes feedback is immediately available; an assumption which fails in many real world situations, including recommendation systems, clinical trials and hyperparameter tuning. We consider a kernel bandit problem under stochastically delayed feedback, and propose an algorithm with $\tilde{\mathcal{O}}\left(\sqrt{\Gamma_k(T) T}+\mathbb{E}[\tau]\right)$ regret, where $T$ is the number of time steps, $\Gamma_k(T)$ is the maximum information gain of the kernel with $T$ observations, and $\tau$ is the delay random variable. This represents a significant improvement over the state of the art regret bound of $\tilde{\mathcal{O}}\left(\Gamma_k(T)\sqrt{ T}+\mathbb{E}[\tau]\Gamma_k(T)\right)$ reported in (Verma et al., 2022). In particular, for very non-smooth kernels, the information gain grows almost linearly in time, trivializing the existing results. We also validate our theoretical results with simulations.

        ----

        ## [1447] Synthetic Data, Real Errors: How (Not) to Publish and Use Synthetic Data

        **Authors**: *Boris van Breugel, Zhaozhi Qian, Mihaela van der Schaar*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/van-breugel23a.html](https://proceedings.mlr.press/v202/van-breugel23a.html)

        **Abstract**:

        Generating synthetic data through generative models is gaining interest in the ML community and beyond, promising a future where datasets can be tailored to individual needs. Unfortunately, synthetic data is usually not perfect, resulting in potential errors in downstream tasks. In this work we explore how the generative process affects the downstream ML task. We show that the naive synthetic data approach—using synthetic data as if it is real—leads to downstream models and analyses that do not generalize well to real data. As a first step towards better ML in the synthetic data regime, we introduce Deep Generative Ensemble (DGE)—a framework inspired by Deep Ensembles that aims to implicitly approximate the posterior distribution over the generative process model parameters. DGE improves downstream model training, evaluation, and uncertainty quantification, vastly outperforming the naive approach on average. The largest improvements are achieved for minority classes and low-density regions of the original data, for which the generative uncertainty is largest.

        ----

        ## [1448] Trading-Off Payments and Accuracy in Online Classification with Paid Stochastic Experts

        **Authors**: *Dirk van der Hoeven, Ciara Pike-Burke, Hao Qiu, Nicolò Cesa-Bianchi*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/van-der-hoeven23a.html](https://proceedings.mlr.press/v202/van-der-hoeven23a.html)

        **Abstract**:

        We investigate online classification with paid stochastic experts. Here, before making their prediction, each expert must be paid. The amount that we pay each expert directly influences the accuracy of their prediction through some unknown Lipschitz “productivity” function. In each round, the learner must decide how much to pay each expert and then make a prediction. They incur a cost equal to a weighted sum of the prediction error and upfront payments for all experts. We introduce an online learning algorithm whose total cost after $T$ rounds exceeds that of a predictor which knows the productivity of all experts in advance by at most $\mathcal{O}\big(K^2(\ln T)\sqrt{T}\big)$ where $K$ is the number of experts. In order to achieve this result, we combine Lipschitz bandits and online classification with surrogate losses. These tools allow us to improve upon the bound of order $T^{2/3}$ one would obtain in the standard Lipschitz bandit setting. Our algorithm is empirically evaluated on synthetic data.

        ----

        ## [1449] Causal Isotonic Calibration for Heterogeneous Treatment Effects

        **Authors**: *Lars van der Laan, Ernesto Ulloa-Pérez, Marco Carone, Alex Luedtke*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/van-der-laan23a.html](https://proceedings.mlr.press/v202/van-der-laan23a.html)

        **Abstract**:

        We propose causal isotonic calibration, a novel nonparametric method for calibrating predictors of heterogeneous treatment effects. Furthermore, we introduce cross-calibration, a data-efficient variant of calibration that eliminates the need for hold-out calibration sets. Cross-calibration leverages cross-fitted predictors and generates a single calibrated predictor using all available data. Under weak conditions that do not assume monotonicity, we establish that both causal isotonic calibration and cross-calibration achieve fast doubly-robust calibration rates, as long as either the propensity score or outcome regression is estimated accurately in a suitable sense. The proposed causal isotonic calibrator can be wrapped around any black-box learning algorithm, providing robust and distribution-free calibration guarantees while preserving predictive performance.

        ----

        ## [1450] Accounting For Informative Sampling When Learning to Forecast Treatment Outcomes Over Time

        **Authors**: *Toon Vanderschueren, Alicia Curth, Wouter Verbeke, Mihaela van der Schaar*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/vanderschueren23a.html](https://proceedings.mlr.press/v202/vanderschueren23a.html)

        **Abstract**:

        Machine learning (ML) holds great potential for accurately forecasting treatment outcomes over time, which could ultimately enable the adoption of more individualized treatment strategies in many practical applications. However, a significant challenge that has been largely overlooked by the ML literature on this topic is the presence of informative sampling in observational data. When instances are observed irregularly over time, sampling times are typically not random, but rather informative–depending on the instance’s characteristics, past outcomes, and administered treatments. In this work, we formalize informative sampling as a covariate shift problem and show that it can prohibit accurate estimation of treatment outcomes if not properly accounted for. To overcome this challenge, we present a general framework for learning treatment outcomes in the presence of informative sampling using inverse intensity-weighting, and propose a novel method, TESAR-CDE, that instantiates this framework using Neural CDEs. Using a simulation environment based on a clinical use case, we demonstrate the effectiveness of our approach in learning under informative sampling.

        ----

        ## [1451] Best Arm Identification in Multi-Agent Multi-Armed Bandits

        **Authors**: *Filippo Vannella, Alexandre Proutière, Jaeseong Jeong*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/vannella23a.html](https://proceedings.mlr.press/v202/vannella23a.html)

        **Abstract**:

        We investigate the problem of best arm identification in Multi-Agent Multi-Armed Bandits (MAMABs) where the rewards are defined through a factor graph. The objective is to find an optimal global action with a prescribed level of confidence and minimal sample complexity. We derive a tight instance-specific lower bound of the sample complexity and characterize the corresponding optimal sampling strategy. Unfortunately, this bound is obtained by solving a combinatorial optimization problem with a number of variables and constraints exponentially growing with the number of agents. We leverage Mean Field (MF) techniques to obtain, in a computationally efficient manner, an approximation of the lower bound. The approximation scales at most as $\rho K^d$ (where $\rho$, $K$, and $d$ denote the number of factors in the graph, the number of possible actions per agent, and the maximal degree of the factor graph). We devise MF-TaS (Mean-Field-Track-and-Stop), an algorithm whose sample complexity provably matches our approximated lower bound. We illustrate the performance of MF-TaS numerically using both synthetic and real-world experiments (e.g., to solve the antenna tilt optimization problem in radio communication networks).

        ----

        ## [1452] Conditional Tree Matching for Inference-Time Adaptation of Tree Prediction Models

        **Authors**: *Harshit Varma, Abhijeet Awasthi, Sunita Sarawagi*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/varma23a.html](https://proceedings.mlr.press/v202/varma23a.html)

        **Abstract**:

        We present CTreeOT, a convergent, differentiable algorithm for matching two trees when each tree is conditioned on some input. Such conditional tree matching is useful for light-weight, few-shot adaptation of tree prediction models without parameter fine-tuning. CTreeOT includes an alignment algorithm that extends the popular Sinkhorn algorithm for matching tree nodes while supporting constraints on tree edges. The algorithm involves alternating between matrix rescaling and message passing updates, and can be efficiently expressed as GPU tensor operations. The second part of CTreeOT is fine-grained relevance-based reweighting of nodes that makes the match scores useful for prediction tasks. We demonstrate the usefulness of CTreeOT for cross-schema adaptation of Text-to-SQL, a popular semantic parsing task. We show that compared to state-of-the-art methods, we achieve significant increase in adaptation accuracy.

        ----

        ## [1453] Optimal LP Rounding and Linear-Time Approximation Algorithms for Clustering Edge-Colored Hypergraphs

        **Authors**: *Nate Veldt*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/veldt23a.html](https://proceedings.mlr.press/v202/veldt23a.html)

        **Abstract**:

        We study the approximability of an existing framework for clustering edge-colored hypergraphs, which is closely related to chromatic correlation clustering and is motivated by machine learning and data mining applications where the goal is to cluster a set of objects based on multiway interactions of different categories or types. We present improved approximation guarantees based on linear programming, and show they are tight by proving a matching integrality gap. Our results also include new approximation hardness results, a combinatorial 2-approximation whose runtime is linear in the hypergraph size, and several new connections to well-studied objectives such as vertex cover and hypergraph multiway cut.

        ----

        ## [1454] Fast (1+ε)-Approximation Algorithms for Binary Matrix Factorization

        **Authors**: *Ameya Velingker, Maximilian Vötsch, David P. Woodruff, Samson Zhou*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/velingker23a.html](https://proceedings.mlr.press/v202/velingker23a.html)

        **Abstract**:

        We introduce efficient $(1+\varepsilon)$-approximation algorithms for the binary matrix factorization (BMF) problem, where the inputs are a matrix $\mathbf{A}\in\{0,1\}^{n\times d}$, a rank parameter $k>0$, as well as an accuracy parameter $\varepsilon>0$, and the goal is to approximate $\mathbf{A}$ as a product of low-rank factors $\mathbf{U}\in\{0,1\}^{n\times k}$ and $\mathbf{V}\in\{0,1\}^{k\times d}$. Equivalently, we want to find $\mathbf{U}$ and $\mathbf{V}$ that minimize the Frobenius loss $\|\mathbf{U}\mathbf{V} - \mathbf{A}\|_F^2$. Before this work, the state-of-the-art for this problem was the approximation algorithm of Kumar et. al. [ICML 2019], which achieves a $C$-approximation for some constant $C\ge 576$. We give the first $(1+\varepsilon)$-approximation algorithm using running time singly exponential in $k$, where $k$ is typically a small integer. Our techniques generalize to other common variants of the BMF problem, admitting bicriteria $(1+\varepsilon)$-approximation algorithms for $L_p$ loss functions and the setting where matrix operations are performed in $\mathbb{F}_2$. Our approach can be implemented in standard big data models, such as the streaming or distributed models.

        ----

        ## [1455] The Virtues of Laziness in Model-based RL: A Unified Objective and Algorithms

        **Authors**: *Anirudh Vemula, Yuda Song, Aarti Singh, Drew Bagnell, Sanjiban Choudhury*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/vemula23a.html](https://proceedings.mlr.press/v202/vemula23a.html)

        **Abstract**:

        We propose a novel approach to addressing two fundamental challenges in Model-based Reinforcement Learning (MBRL): the computational expense of repeatedly finding a good policy in the learned model, and the objective mismatch between model fitting and policy computation. Our "lazy" method leverages a novel unified objective, Performance Difference via Advantage in Model, to capture the performance difference between the learned policy and expert policy under the true dynamics. This objective demonstrates that optimizing the expected policy advantage in the learned model under an exploration distribution is sufficient for policy computation, resulting in a significant boost in computational efficiency compared to traditional planning methods. Additionally, the unified objective uses a value moment matching term for model fitting, which is aligned with the model’s usage during policy computation. We present two no-regret algorithms to optimize the proposed objective, and demonstrate their statistical and computational gains compared to existing MBRL methods through simulated benchmarks.

        ----

        ## [1456] Learning the Right Layers a Data-Driven Layer-Aggregation Strategy for Semi-Supervised Learning on Multilayer Graphs

        **Authors**: *Sara Venturini, Andrea Cristofari, Francesco Rinaldi, Francesco Tudisco*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/venturini23a.html](https://proceedings.mlr.press/v202/venturini23a.html)

        **Abstract**:

        Clustering (or community detection) on multilayer graphs poses several additional complications with respect to standard graphs as different layers may be characterized by different structures and types of information. One of the major challenges is to establish the extent to which each layer contributes to the cluster assignment in order to effectively take advantage of the multilayer structure and improve upon the classification obtained using the individual layers or their union. However, making an informed a-priori assessment about the clustering information content of the layers can be very complicated. In this work, we assume a semi-supervised learning setting, where the class of a small percentage of nodes is initially provided, and we propose a parameter-free Laplacian-regularized model that learns an optimal nonlinear combination of the different layers from the available input labels. The learning algorithm is based on a Frank-Wolfe optimization scheme with inexact gradient, combined with a modified Label Propagation iteration. We provide a detailed convergence analysis of the algorithm and extensive experiments on synthetic and real-world datasets, showing that the proposed method compares favourably with a variety of baselines and outperforms each individual layer when used in isolation.

        ----

        ## [1457] Multi-Environment Pretraining Enables Transfer to Action Limited Datasets

        **Authors**: *David Venuto, Sherry Yang, Pieter Abbeel, Doina Precup, Igor Mordatch, Ofir Nachum*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/venuto23a.html](https://proceedings.mlr.press/v202/venuto23a.html)

        **Abstract**:

        Using massive datasets to train large-scale models has emerged as a dominant approach for broad generalization in natural language and vision applications. In reinforcement learning, however, a key challenge is that available data of sequential decision making is often not annotated with actions - for example, videos of game-play are much more available than sequences of frames paired with their logged game controls. We propose to circumvent this challenge by combining large but sparsely-annotated datasets from a target environment of interest with fully-annotated datasets from various other source environments. Our method, Action Limited PreTraining (ALPT), leverages the generalization capabilities of inverse dynamics modelling (IDM) to label missing action data in the target environment. We show that utilizing even one additional environment dataset of labelled data during IDM pretraining gives rise to substantial improvements in generating action labels for unannotated sequences. We evaluate our method on benchmark game-playing environments and show that we can significantly improve game performance and generalization capability compared to other approaches, using annotated datasets equivalent to only $12$ minutes of gameplay. Highlighting the power of IDM, we show that these benefits remain even when target and source environments share no common actions.

        ----

        ## [1458] AbODE: Ab initio antibody design using conjoined ODEs

        **Authors**: *Yogesh Verma, Markus Heinonen, Vikas Garg*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/verma23a.html](https://proceedings.mlr.press/v202/verma23a.html)

        **Abstract**:

        Antibodies are Y-shaped proteins that neutralize pathogens and constitute the core of our adaptive immune system. De novo generation of new antibodies that target specific antigens holds the key to accelerating vaccine discovery. However, this co-design of the amino acid sequence and the 3D structure subsumes and accentuates, some central challenges from multiple tasks including protein folding (sequence to structure), inverse folding (structure to sequence), and docking (binding). We strive to surmount these challenges with a new generative model AbODE that extends graph PDEs to accommodate both contextual information and external interactions. Unlike existing approaches, AbODE uses a single round of full-shot decoding, and elicits continuous differential attention that encapsulates, and evolves with, latent interactions within the antibody as well as those involving the antigen. We unravel fundamental connections between AbODE and temporal networks as well as graph-matching networks. The proposed model significantly outperforms existing methods on standard metrics across benchmarks.

        ----

        ## [1459] TabLeak: Tabular Data Leakage in Federated Learning

        **Authors**: *Mark Vero, Mislav Balunovic, Dimitar Iliev Dimitrov, Martin T. Vechev*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/vero23a.html](https://proceedings.mlr.press/v202/vero23a.html)

        **Abstract**:

        While federated learning (FL) promises to preserve privacy, recent works in the image and text domains have shown that training updates leak private client data. However, most high-stakes applications of FL (e.g., in healthcare and finance) use tabular data, where the risk of data leakage has not yet been explored. A successful attack for tabular data must address two key challenges unique to the domain: (i) obtaining a solution to a high-variance mixed discrete-continuous optimization problem, and (ii) enabling human assessment of the reconstruction as unlike for image and text data, direct human inspection is not possible. In this work we address these challenges and propose TabLeak, the first comprehensive reconstruction attack on tabular data. TabLeak is based on two key contributions: (i) a method which leverages a softmax relaxation and pooled ensembling to solve the optimization problem, and (ii) an entropy-based uncertainty quantification scheme to enable human assessment. We evaluate TabLeak on four tabular datasets for both FedSGD and FedAvg training protocols, and show that it successfully breaks several settings previously deemed safe. For instance, we extract large subsets of private data at $>$90% accuracy even at the large batch size of 128. Our findings demonstrate that current high-stakes tabular FL is excessively vulnerable to leakage attacks.

        ----

        ## [1460] Low-Variance Gradient Estimation in Unrolled Computation Graphs with ES-Single

        **Authors**: *Paul Vicol*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/vicol23a.html](https://proceedings.mlr.press/v202/vicol23a.html)

        **Abstract**:

        We propose an evolution strategies-based algorithm for estimating gradients in unrolled computation graphs, called ES-Single. Similarly to the recently-proposed Persistent Evolution Strategies (PES), ES-Single is unbiased, and overcomes chaos arising from recursive function applications by smoothing the meta-loss landscape. ES-Single samples a single perturbation per particle, that is kept fixed over the course of an inner problem (e.g., perturbations are not re-sampled for each partial unroll). Compared to PES, ES-Single is simpler to implement and has lower variance: the variance of ES-Single is constant with respect to the number of truncated unrolls, removing a key barrier in applying ES to long inner problems using short truncations. We show that ES-Single is unbiased for quadratic inner problems, and demonstrate empirically that its variance can be substantially lower than that of PES. ES-Single consistently outperforms PES on a variety of tasks, including a synthetic benchmark task, hyperparameter optimization, training recurrent neural networks, and training learned optimizers.

        ----

        ## [1461] Arithmetic Sampling: Parallel Diverse Decoding for Large Language Models

        **Authors**: *Luke Vilnis, Yury Zemlyanskiy, Patrick Murray, Alexandre Tachard Passos, Sumit Sanghai*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/vilnis23a.html](https://proceedings.mlr.press/v202/vilnis23a.html)

        **Abstract**:

        Decoding methods for large language models often trade-off between diversity of outputs and parallelism of computation. Methods such as beam search and Gumbel top-k sampling can guarantee a different output for each element of the beam, but are not easy to parallelize. Alternatively, methods such as temperature sampling and its modifications (top-k sampling, nucleus sampling, typical decoding, and others), are embarrassingly parallel, but have no guarantees about duplicate samples. We present a framework for sampling according to an arithmetic code book implicitly defined by a large language model, compatible with common sampling variations, with provable beam diversity under certain conditions, as well as being embarrassingly parallel and providing unbiased and consistent expectations from the original model. We demonstrate the effectiveness of our approach on WMT machine translation, more than halving the standard deviation when estimating expected BLEU score reward, and closing the BLEU score gap between independent sampling and beam search by up to 63%.

        ----

        ## [1462] Eventual Discounting Temporal Logic Counterfactual Experience Replay

        **Authors**: *Cameron Voloshin, Abhinav Verma, Yisong Yue*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/voloshin23a.html](https://proceedings.mlr.press/v202/voloshin23a.html)

        **Abstract**:

        Linear temporal logic (LTL) offers a simplified way of specifying tasks for policy optimization that may otherwise be difficult to describe with scalar reward functions. However, the standard RL framework can be too myopic to find maximally LTL satisfying policies. This paper makes two contributions. First, we develop a new value-function based proxy, using a technique we call eventual discounting, under which one can find policies that satisfy the LTL specification with highest achievable probability. Second, we develop a new experience replay method for generating off-policy data from on-policy rollouts via counterfactual reasoning on different ways of satisfying the LTL specification. Our experiments, conducted in both discrete and continuous state-action spaces, confirm the effectiveness of our counterfactual experience replay approach.

        ----

        ## [1463] Transformers Learn In-Context by Gradient Descent

        **Authors**: *Johannes von Oswald, Eyvind Niklasson, Ettore Randazzo, João Sacramento, Alexander Mordvintsev, Andrey Zhmoginov, Max Vladymyrov*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/von-oswald23a.html](https://proceedings.mlr.press/v202/von-oswald23a.html)

        **Abstract**:

        At present, the mechanisms of in-context learning in Transformers are not well understood and remain mostly an intuition. In this paper, we suggest that training Transformers on auto-regressive objectives is closely related to gradient-based meta-learning formulations. We start by providing a simple weight construction that shows the equivalence of data transformations induced by 1) a single linear self-attention layer and by 2) gradient-descent (GD) on a regression loss. Motivated by that construction, we show empirically that when training self-attention-only Transformers on simple regression tasks either the models learned by GD and Transformers show great similarity or, remarkably, the weights found by optimization match the construction. Thus we show how trained Transformers become mesa-optimizers i.e. learn models by gradient descent in their forward pass. This allows us, at least in the domain of regression problems, to mechanistically understand the inner workings of in-context learning in optimized Transformers. Building on this insight, we furthermore identify how Transformers surpass the performance of plain gradient descent by learning an iterative curvature correction and learn linear models on deep data representations to solve non-linear regression tasks. Finally, we discuss intriguing parallels to a mechanism identified to be crucial for in-context learning termed induction-head (Olsson et al., 2022) and show how it could be understood as a specific case of in-context learning by gradient descent learning within Transformers.

        ----

        ## [1464] Topological Singularity Detection at Multiple Scales

        **Authors**: *Julius von Rohrscheidt, Bastian Rieck*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/von-rohrscheidt23a.html](https://proceedings.mlr.press/v202/von-rohrscheidt23a.html)

        **Abstract**:

        The manifold hypothesis, which assumes that data lies on or close to an unknown manifold of low intrinsic dimension, is a staple of modern machine learning research. However, recent work has shown that real-world data exhibits distinct non-manifold structures, i.e. singularities, that can lead to erroneous findings. Detecting such singularities is therefore crucial as a precursor to interpolation and inference tasks. We address this issue by developing a topological framework that (i) quantifies the local intrinsic dimension, and (ii) yields a Euclidicity score for assessing the ’manifoldness’ of a point along multiple scales. Our approach identifies singularities of complex spaces, while also capturing singular structures and local geometric complexity in image data.

        ----

        ## [1465] Improving l1-Certified Robustness via Randomized Smoothing by Leveraging Box Constraints

        **Authors**: *Václav Vorácek, Matthias Hein*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/voracek23a.html](https://proceedings.mlr.press/v202/voracek23a.html)

        **Abstract**:

        Randomized smoothing is a popular method to certify robustness of image classifiers to adversarial input perturbations. It is the only certification technique which scales directly to datasets of higher dimension such as ImageNet. However, current techniques are not able to utilize the fact that any adversarial example has to lie in the image space, that is $[0,1]^d$; otherwise, one can trivially detect it. To address this suboptimality, we derive new certification formulae which lead to significant improvements in the certified $\ell_1$-robustness without the need of adapting the classifiers or change of smoothing distributions. The code is released at https://github.com/vvoracek/L1-smoothing

        ----

        ## [1466] Vector Quantized Wasserstein Auto-Encoder

        **Authors**: *Long Tung Vuong, Trung Le, He Zhao, Chuanxia Zheng, Mehrtash Harandi, Jianfei Cai, Dinh Q. Phung*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/vuong23a.html](https://proceedings.mlr.press/v202/vuong23a.html)

        **Abstract**:

        Learning deep discrete latent presentations offers a promise of better symbolic and summarized abstractions that are more useful to subsequent downstream tasks. Inspired by the seminal Vector Quantized Variational Auto-Encoder (VQ-VAE), most of work in learning deep discrete representations has mainly focused on improving the original VQ-VAE form and none of them has studied learning deep discrete representations from the generative viewpoint. In this work, we study learning deep discrete representations from the generative viewpoint. Specifically, we endow discrete distributions over sequences of codewords and learn a deterministic decoder that transports the distribution over the sequences of codewords to the data distribution via minimizing a WS distance between them. We develop further theories to connect it with the clustering viewpoint of WS distance, allowing us to have a better and more controllable clustering solution. Finally, we empirically evaluate our method on several well-known benchmarks, where it achieves better qualitative and quantitative performances than the other VQ-VAE variants in terms of the codebook utilization and image reconstruction/generation.

        ----

        ## [1467] Competitive Gradient Optimization

        **Authors**: *Abhijeet Vyas, Brian Bullins, Kamyar Azizzadenesheli*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/vyas23a.html](https://proceedings.mlr.press/v202/vyas23a.html)

        **Abstract**:

        We study the problem of convergence to a stationary point in zero-sum games. We propose competitive gradient optimization (CGO), a gradient-based method that incorporates the interactions between two players in zero-sum games for its iterative updates. We provide a continuous-time analysis of CGO and its convergence properties while showing that in the continuous limit, previous methods degenerate to their gradient descent ascent (GDA) variants. We further provide a rate of convergence to stationary points in the discrete-time setting. We propose a generalized class of $\alpha$-coherent functions and show that for strictly $\alpha$-coherent functions, CGO ensures convergence to a saddle point. Moreover, we propose optimistic CGO (oCGO), an optimistic variant, for which we show a convergence rate of $O(\frac{1}{n})$ to saddle points for $\alpha$-coherent functions.

        ----

        ## [1468] On Provable Copyright Protection for Generative Models

        **Authors**: *Nikhil Vyas, Sham M. Kakade, Boaz Barak*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/vyas23b.html](https://proceedings.mlr.press/v202/vyas23b.html)

        **Abstract**:

        There is a growing concern that learned conditional generative models may output samples that are substantially similar to some copyrighted data $C$ that was in their training set. We give a formal definition of near access-freeness (NAF) and prove bounds on the probability that a model satisfying this definition outputs a sample similar to $C$, even if $C$ is included in its training set. Roughly speaking, a generative model $p$ is $k$-NAF if for every potentially copyrighted data $C$, the output of $p$ diverges by at most $k$-bits from the output of a model $q$ that did not access $C$ at all. We also give generative model learning algorithms, which efficiently modify the original generative model learning algorithm in a black box manner, that output generative models with strong bounds on the probability of sampling protected content. Furthermore, we provide promising experiments for both language (transformers) and image (diffusion) generative models, showing minimal degradation in output quality while ensuring strong protections against sampling protected content.

        ----

        ## [1469] Leveraging Offline Data in Online Reinforcement Learning

        **Authors**: *Andrew Wagenmaker, Aldo Pacchiano*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/wagenmaker23a.html](https://proceedings.mlr.press/v202/wagenmaker23a.html)

        **Abstract**:

        Two central paradigms have emerged in the reinforcement learning (RL) community: online RL and offline RL. In the online RL setting, the agent has no prior knowledge of the environment, and must interact with it in order to find an $\epsilon$-optimal policy. In the offline RL setting, the learner instead has access to a fixed dataset to learn from, but is unable to otherwise interact with the environment, and must obtain the best policy it can from this offline data. Practical scenarios often motivate an intermediate setting: if we have some set of offline data and may also interact with the environment, how can we best use the offline data to minimize the number of online interactions necessary to learn an $\epsilon$-optimal policy. In this work, we consider this setting, which we call the FineTuneRL setting, for MDPs with linear structure. We characterize the necessary number of online samples needed in this setting given access to some offline dataset, and develop an algorithm, FTPedel, which is provably optimal, up to $H$ factors. We show through an explicit example that combining offline data with online interactions can lead to a provable improvement over either purely offline or purely online RL. Finally, our results illustrate the distinction between verifiable learning, the typical setting considered in online RL, and unverifiable learning, the setting often considered in offline RL, and show that there is a formal separation between these regimes.

        ----

        ## [1470] Fast Private Kernel Density Estimation via Locality Sensitive Quantization

        **Authors**: *Tal Wagner, Yonatan Naamad, Nina Mishra*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/wagner23a.html](https://proceedings.mlr.press/v202/wagner23a.html)

        **Abstract**:

        We study efficient mechanisms for differentially private kernel density estimation (DP-KDE). Prior work for the Gaussian kernel described algorithms that run in time exponential in the number of dimensions $d$. This paper breaks the exponential barrier, and shows how the KDE can privately be approximated in time linear in $d$, making it feasible for high-dimensional data. We also present improved bounds for low-dimensional data. Our results are obtained through a general framework, which we term Locality Sensitive Quantization (LSQ), for constructing private KDE mechanisms where existing KDE approximation techniques can be applied. It lets us leverage several efficient non-private KDE methods—like Random Fourier Features, the Fast Gauss Transform, and Locality Sensitive Hashing—and “privatize” them in a black-box manner. Our experiments demonstrate that our resulting DP-KDE mechanisms are fast and accurate on large datasets in both high and low dimensions.

        ----

        ## [1471] Investigating the Role of Model-Based Learning in Exploration and Transfer

        **Authors**: *Jacob C. Walker, Eszter Vértes, Yazhe Li, Gabriel Dulac-Arnold, Ankesh Anand, Theophane Weber, Jessica B. Hamrick*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/walker23a.html](https://proceedings.mlr.press/v202/walker23a.html)

        **Abstract**:

        State of the art reinforcement learning has enabled training agents on tasks of ever increasing complexity. However, the current paradigm tends to favor training agents from scratch on every new task or on collections of tasks with a view towards generalizing to novel task configurations. The former suffers from poor data efficiency while the latter is difficult when test tasks are out-of-distribution. Agents that can effectively transfer their knowledge about the world pose a potential solution to these issues. In this paper, we investigate transfer learning in the context of model-based agents. Specifically, we aim to understand where exactly environment models have an advantage and why. We find that a model-based approach outperforms controlled model-free baselines for transfer learning. Through ablations, we show that both the policy and dynamics model learnt through exploration matter for successful transfer. We demonstrate our results across three domains which vary in their requirements for transfer: in-distribution procedural (Crafter), in-distribution identical (RoboDesk), and out-of-distribution (Meta-World). Our results show that intrinsic exploration combined with environment models present a viable direction towards agents that are self-supervised and able to generalize to novel reward functions.

        ----

        ## [1472] UPSCALE: Unconstrained Channel Pruning

        **Authors**: *Alvin Wan, Hanxiang Hao, Kaushik Patnaik, Yueyang Xu, Omer Hadad, David Güera, Zhile Ren, Qi Shan*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/wan23a.html](https://proceedings.mlr.press/v202/wan23a.html)

        **Abstract**:

        As neural networks grow in size and complexity, inference speeds decline. To combat this, one of the most effective compression techniques – channel pruning – removes channels from weights. However, for multi-branch segments of a model, channel removal can introduce inference-time memory copies. In turn, these copies increase inference latency – so much so that the pruned model can be slower than the unpruned model. As a workaround, pruners conventionally constrain certain channels to be pruned together. This fully eliminates memory copies but, as we show, significantly impairs accuracy. We now have a dilemma: Remove constraints but increase latency, or add constraints and impair accuracy. In response, our insight is to reorder channels at export time, (1) reducing latency by reducing memory copies and (2) improving accuracy by removing constraints. Using this insight, we design a generic algorithm UPSCALE to prune models with any pruning pattern. By removing constraints from existing pruners, we improve ImageNet accuracy for post-training pruned models by 2.1 points on average – benefiting DenseNet (+16.9), EfficientNetV2 (+7.9), and ResNet (+6.2). Furthermore, by reordering channels, UPSCALE improves inference speeds by up to 2x over a baseline export.

        ----

        ## [1473] Poisoning Language Models During Instruction Tuning

        **Authors**: *Alexander Wan, Eric Wallace, Sheng Shen, Dan Klein*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/wan23b.html](https://proceedings.mlr.press/v202/wan23b.html)

        **Abstract**:

        Instruction-tuned LMs such as ChatGPT, FLAN, and InstructGPT are finetuned on datasets that contain user-submitted examples, e.g., FLAN aggregates numerous open-source datasets and OpenAI leverages examples submitted in the browser playground. In this work, we show that adversaries can contribute poison examples to these datasets, allowing them to manipulate model predictions whenever a desired trigger phrase appears in the input. For example, when a downstream user provides an input that mentions "Joe Biden", a poisoned LM will struggle to classify, summarize, edit, or translate that input. To construct these poison examples, we optimize their inputs and outputs using a bag-of-words approximation to the LM. We evaluate our method on open-source instruction-tuned LMs. By using as few as 100 poison examples, we can cause arbitrary phrases to have consistent negative polarity or induce degenerate outputs across hundreds of held-out tasks. Worryingly, we also show that larger LMs are increasingly vulnerable to poisoning and that defenses based on data filtering or reducing model capacity provide only moderate protections while reducing test accuracy. Notice: This paper contains tasks with obscene content.

        ----

        ## [1474] SeMAIL: Eliminating Distractors in Visual Imitation via Separated Models

        **Authors**: *Shenghua Wan, Yucen Wang, Minghao Shao, Ruying Chen, De-Chuan Zhan*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/wan23c.html](https://proceedings.mlr.press/v202/wan23c.html)

        **Abstract**:

        Model-based imitation learning (MBIL) is a popular reinforcement learning method that improves sample efficiency on high-dimension input sources, such as images and videos. Following the convention of MBIL research, existing algorithms are highly deceptive by task-irrelevant information, especially moving distractors in videos. To tackle this problem, we propose a new algorithm - named Separated Model-based Adversarial Imitation Learning (SeMAIL) - decoupling the environment dynamics into two parts by task-relevant dependency, which is determined by agent actions, and training separately. In this way, the agent can imagine its trajectories and imitate the expert behavior efficiently in task-relevant state space. Our method achieves near-expert performance on various visual control tasks with complex observations and the more challenging tasks with different backgrounds from expert observations.

        ----

        ## [1475] Multiplier Bootstrap-based Exploration

        **Authors**: *Runzhe Wan, Haoyu Wei, Branislav Kveton, Rui Song*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/wan23d.html](https://proceedings.mlr.press/v202/wan23d.html)

        **Abstract**:

        Despite the great interest in the bandit problem, designing efficient algorithms for complex models remains challenging, as there is typically no analytical way to quantify uncertainty. In this paper, we propose Multiplier Bootstrap-based Exploration (MBE), a novel exploration strategy that is applicable to any reward model amenable to weighted loss minimization. We prove both instance-dependent and instance-independent rate-optimal regret bounds for MBE in sub-Gaussian multi-armed bandits. With extensive simulation and real-data experiments, we show the generality and adaptivity of MBE.

        ----

        ## [1476] Bandit Multi-linear DR-Submodular Maximization and Its Applications on Adversarial Submodular Bandits

        **Authors**: *Zongqi Wan, Jialin Zhang, Wei Chen, Xiaoming Sun, Zhijie Zhang*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/wan23e.html](https://proceedings.mlr.press/v202/wan23e.html)

        **Abstract**:

        We investigate the online bandit learning of the monotone multi-linear DR-submodular functions, designing the algorithm $\mathtt{BanditMLSM}$ that attains $O(T^{2/3}\log T)$ of $(1-1/e)$-regret. Then we reduce submodular bandit with partition matroid constraint and bandit sequential monotone maximization to the online bandit learning of the monotone multi-linear DR-submodular functions, attaining $O(T^{2/3}\log T)$ of $(1-1/e)$-regret in both problems, which improve the existing results. To the best of our knowledge, we are the first to give a sublinear regret algorithm for the submodular bandit with partition matroid constraint. A special case of this problem is studied by Streeter et al.(2009). They prove a $O(T^{4/5})$ $(1-1/e)$-regret upper bound. For the bandit sequential submodular maximization, the existing work proves an $O(T^{2/3})$ regret with a suboptimal $1/2$ approximation ratio (Niazadeh et al. 2021).

        ----

        ## [1477] Tight Regret Bounds for Single-pass Streaming Multi-armed Bandits

        **Authors**: *Chen Wang*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/wang23a.html](https://proceedings.mlr.press/v202/wang23a.html)

        **Abstract**:

        Regret minimization in streaming multi-armed bandits (MABs) has been studied extensively, and recent work has shown that algorithms with $o(K)$ memory have to incur $\Omega(T^{2/3})$ regret, where $K$ and $T$ are the numbers of arms and trials. However, the previous best regret upper bound is still $O(K^{1/3} T^{2/3}\log^{1/3}(T))$, which is achieved by the simple uniform exploration algorithm. In this paper, we close this gap and complete the picture of regret minimization in single-pass streaming MABs. We first improve the regret lower bound to $\Omega(K^{1/3}T^{2/3})$ for algorithms with $o(K)$ memory. We then show that the $\log^{1/3}(T)$ factor is not necessary by designing algorithms with at most $O(\log^*(K))$-arm memory and achieve $O(K^{1/3}T^{2/3})$ expected regret based on streaming $\varepsilon$-best arm algorithms. We further tested the empirical performances of our algorithms on simulated MABs instances, where the proposed algorithms outperform the benchmark uniform exploration algorithm by a large margin and, on occasion, reduce the regret by up to 70%.

        ----

        ## [1478] Improved Active Multi-Task Representation Learning via Lasso

        **Authors**: *Yiping Wang, Yifang Chen, Kevin Jamieson, Simon Shaolei Du*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/wang23b.html](https://proceedings.mlr.press/v202/wang23b.html)

        **Abstract**:

        To leverage the copious amount of data from source tasks and overcome the scarcity of the target task samples, representation learning based on multi-task pretraining has become a standard approach in many applications. However, up until now, most existing works design a source task selection strategy from a purely empirical perspective. Recently, Chen et al., 2022 gave the first active multi-task representation learning (A-MTRL) algorithm which adaptively samples from source tasks and can provably reduce the total sample complexity using the L2-regularized-target-source-relevance parameter $\nu^2$. But their work is theoretically suboptimal in terms of total source sample complexity and is less practical in some real-world scenarios where sparse training source task selection is desired. In this paper, we address both issues. Specifically, we show the strict dominance of the L1-regularized-relevance-based ($\nu^1$-based) strategy by giving a lower bound for the $\nu^2$-based strategy. When $\nu^1$ is unknown, we propose a practical algorithm that uses the LASSO program to estimate $\nu^1$. Our algorithm successfully recovers the optimal result in the known case. In addition to our sample complexity results, we also characterize the potential of our $\nu^1$-based strategy in sample-cost-sensitive settings. Finally, we provide experiments on real-world computer vision datasets to illustrate the effectiveness of our proposed method.

        ----

        ## [1479] Tilted Sparse Additive Models

        **Authors**: *Yingjie Wang, Hong Chen, Weifeng Liu, Fengxiang He, Tieliang Gong, Youcheng Fu, Dacheng Tao*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/wang23c.html](https://proceedings.mlr.press/v202/wang23c.html)

        **Abstract**:

        Additive models have been burgeoning in data analysis due to their flexible representation and desirable interpretability. However, most existing approaches are constructed under empirical risk minimization (ERM), and thus perform poorly in situations where average performance is not a suitable criterion for the problems of interest, e.g., data with complex non-Gaussian noise, imbalanced labels or both of them. In this paper, a novel class of sparse additive models is proposed under tilted empirical risk minimization (TERM), which addresses the deficiencies in ERM by imposing tilted impact on individual losses, and is flexibly capable of achieving a variety of learning objectives, e.g., variable selection, robust estimation, imbalanced classification and multiobjective learning. On the theoretical side, a learning theory analysis which is centered around the generalization bound and function approximation error bound (under some specific data distributions) is conducted rigorously. On the practical side, an accelerated optimization algorithm is designed by integrating Prox-SVRG and random Fourier acceleration technique. The empirical assessments verify the competitive performance of our approach on both synthetic and real data.

        ----

        ## [1480] From Hypergraph Energy Functions to Hypergraph Neural Networks

        **Authors**: *Yuxin Wang, Quan Gan, Xipeng Qiu, Xuanjing Huang, David Wipf*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/wang23d.html](https://proceedings.mlr.press/v202/wang23d.html)

        **Abstract**:

        Hypergraphs are a powerful abstraction for representing higher-order interactions between entities of interest. To exploit these relationships in making downstream predictions, a variety of hypergraph neural network architectures have recently been proposed, in large part building upon precursors from the more traditional graph neural network (GNN) literature. Somewhat differently, in this paper we begin by presenting an expressive family of parameterized, hypergraph-regularized energy functions. We then demonstrate how minimizers of these energies effectively serve as node embeddings that, when paired with a parameterized classifier, can be trained end-to-end via a supervised bilevel optimization process. Later, we draw parallels between the implicit architecture of the predictive models emerging from the proposed bilevel hypergraph optimization, and existing GNN architectures in common use. Empirically, we demonstrate state-of-the-art results on various hypergraph node classification benchmarks. Code is available at https://github.com/yxzwang/PhenomNN.

        ----

        ## [1481] A Closer Look at Self-Supervised Lightweight Vision Transformers

        **Authors**: *Shaoru Wang, Jin Gao, Zeming Li, Xiaoqin Zhang, Weiming Hu*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/wang23e.html](https://proceedings.mlr.press/v202/wang23e.html)

        **Abstract**:

        Self-supervised learning on large-scale Vision Transformers (ViTs) as pre-training methods has achieved promising downstream performance. Yet, how much these pre-training paradigms promote lightweight ViTs’ performance is considerably less studied. In this work, we develop and benchmark several self-supervised pre-training methods on image classification tasks and some downstream dense prediction tasks. We surprisingly find that if proper pre-training is adopted, even vanilla lightweight ViTs show comparable performance to previous SOTA networks with delicate architecture design. It breaks the recently popular conception that vanilla ViTs are not suitable for vision tasks in lightweight regimes. We also point out some defects of such pre-training, e.g., failing to benefit from large-scale pre-training data and showing inferior performance on data-insufficient downstream tasks. Furthermore, we analyze and clearly show the effect of such pre-training by analyzing the properties of the layer representation and attention maps for related models. Finally, based on the above analyses, a distillation strategy during pre-training is developed, which leads to further downstream performance improvement for MAE-based pre-training. Code is available at https://github.com/wangsr126/mae-lite.

        ----

        ## [1482] PreNAS: Preferred One-Shot Learning Towards Efficient Neural Architecture Search

        **Authors**: *Haibin Wang, Ce Ge, Hesen Chen, Xiuyu Sun*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/wang23f.html](https://proceedings.mlr.press/v202/wang23f.html)

        **Abstract**:

        The wide application of pre-trained models is driving the trend of once-for-all training in one-shot neural architecture search (NAS). However, training within a huge sample space damages the performance of individual subnets and requires much computation to search for a optimal model. In this paper, we present PreNAS, a search-free NAS approach that accentuates target models in one-shot training. Specifically, the sample space is dramatically reduced in advance by a zero-cost selector, and weight-sharing one-shot training is performed on the preferred architectures to alleviate update conflicts. Extensive experiments have demonstrated that PreNAS consistently outperforms state-of-the-art one-shot NAS competitors for both Vision Transformer and convolutional architectures, and importantly, enables instant specialization with zero search cost. Our code is available at https://github.com/tinyvision/PreNAS.

        ----

        ## [1483] Adversarial Policies Beat Superhuman Go AIs

        **Authors**: *Tony Tong Wang, Adam Gleave, Tom Tseng, Kellin Pelrine, Nora Belrose, Joseph Miller, Michael D. Dennis, Yawen Duan, Viktor Pogrebniak, Sergey Levine, Stuart Russell*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/wang23g.html](https://proceedings.mlr.press/v202/wang23g.html)

        **Abstract**:

        We attack the state-of-the-art Go-playing AI system KataGo by training adversarial policies against it, achieving a $>$97% win rate against KataGo running at superhuman settings. Our adversaries do not win by playing Go well. Instead, they trick KataGo into making serious blunders. Our attack transfers zero-shot to other superhuman Go-playing AIs, and is comprehensible to the extent that human experts can implement it without algorithmic assistance to consistently beat superhuman AIs. The core vulnerability uncovered by our attack persists even in KataGo agents adversarially trained to defend against our attack. Our results demonstrate that even superhuman AI systems may harbor surprising failure modes. Example games are available https://goattack.far.ai/.

        ----

        ## [1484] On Regularization and Inference with Label Constraints

        **Authors**: *Kaifu Wang, Hangfeng He, Tin D. Nguyen, Piyush Kumar, Dan Roth*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/wang23h.html](https://proceedings.mlr.press/v202/wang23h.html)

        **Abstract**:

        Prior knowledge and symbolic rules in machine learning are often expressed in the form of label constraints, especially in structured prediction problems. In this work, we compare two common strategies for encoding label constraints in a machine learning pipeline, regularization with constraints and constrained inference, by quantifying their impact on model performance. For regularization, we show that it narrows the generalization gap by precluding models that are inconsistent with the constraints. However, its preference for small violations introduces a bias toward a suboptimal model. For constrained inference, we show that it reduces the population risk by correcting a model’s violation, and hence turns the violation into an advantage. Given these differences, we further explore the use of two approaches together and propose conditions for constrained inference to compensate for the bias introduced by regularization, aiming to improve both the model complexity and optimal risk.

        ----

        ## [1485] Policy Gradient in Robust MDPs with Global Convergence Guarantee

        **Authors**: *Qiuhao Wang, Chin Pang Ho, Marek Petrik*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/wang23i.html](https://proceedings.mlr.press/v202/wang23i.html)

        **Abstract**:

        Robust Markov decision processes (RMDPs) provide a promising framework for computing reliable policies in the face of model errors. Many successful reinforcement learning algorithms build on variations of policy-gradient methods, but adapting these methods to RMDPs has been challenging. As a result, the applicability of RMDPs to large, practical domains remains limited. This paper proposes a new Double-Loop Robust Policy Gradient (DRPG), the first generic policy gradient method for RMDPs. In contrast with prior robust policy gradient algorithms, DRPG monotonically reduces approximation errors to guarantee convergence to a globally optimal policy in tabular RMDPs. We introduce a novel parametric transition kernel and solve the inner loop robust policy via a gradient-based method. Finally, our numerical results demonstrate the utility of our new algorithm and confirm its global convergence properties.

        ----

        ## [1486] Adaptive Smoothing Gradient Learning for Spiking Neural Networks

        **Authors**: *Ziming Wang, Runhao Jiang, Shuang Lian, Rui Yan, Huajin Tang*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/wang23j.html](https://proceedings.mlr.press/v202/wang23j.html)

        **Abstract**:

        Spiking neural networks (SNNs) with biologically inspired spatio-temporal dynamics demonstrate superior energy efficiency on neuromorphic architectures. Error backpropagation in SNNs is prohibited by the all-or-none nature of spikes. The existing solution circumvents this problem by a relaxation on the gradient calculation using a continuous function with a constant relaxation de- gree, so-called surrogate gradient learning. Nevertheless, such a solution introduces additional smoothing error on spike firing which leads to the gradients being estimated inaccurately. Thus, how to adaptively adjust the relaxation degree and eliminate smoothing error progressively is crucial. Here, we propose a methodology such that training a prototype neural network will evolve into training an SNN gradually by fusing the learnable relaxation degree into the network with random spike noise. In this way, the network learns adaptively the accurate gradients of loss landscape in SNNs. The theoretical analysis further shows optimization on such a noisy network could be evolved into optimization on the embedded SNN with shared weights progressively. Moreover, The experiments on static images, dynamic event streams, speech, and instrumental sounds show the proposed method achieves state-of-the-art performance across all the datasets with remarkable robustness on different relaxation degrees.

        ----

        ## [1487] CircuitNet: A Generic Neural Network to Realize Universal Circuit Motif Modeling

        **Authors**: *Yansen Wang, Xinyang Jiang, Kan Ren, Caihua Shan, Xufang Luo, Dongqi Han, Kaitao Song, Yifei Shen, Dongsheng Li*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/wang23k.html](https://proceedings.mlr.press/v202/wang23k.html)

        **Abstract**:

        The successes of artificial neural networks (ANNs) are largely attributed to mimicking the human brain structures. Recent advances in neuroscience revealed that neurons interact with each other through various kinds of connectivity patterns to process information, in which the common connectivity patterns are also called circuit motifs. However, many existing ANNs can only model one or two circuit motifs in their architectures, so that their performance may drastically vary among different types of machine learning tasks. In this paper, we propose a new type of neural network inspired by the architectures of neuronal circuits, namely Circuit Neural Network (CircuitNet). In CircuitNet, a group of densely connected neurons, namely circuit motif unit (CMU), form the basic unit of the network, which is capable of modeling universal circuit motifs by adjusting the weights within the CMUs. Compared with traditional feed-forward networks, CircuitNet has the ability to model more types of neuron connections such as feed-back and lateral motifs. Inspired by the locally dense and globally sparse structure of the human brain, several iterations of signal transmission among different CMUs are achieved by sparse connections through the input ports and output ports of different CMUs. Experiments have demonstrated that CircuitNet can outperform popular neural network architectures in function approximation, reinforcement learning, image classification, and time series forecasting tasks.

        ----

        ## [1488] Generalized Polyak Step Size for First Order Optimization with Momentum

        **Authors**: *Xiaoyu Wang, Mikael Johansson, Tong Zhang*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/wang23l.html](https://proceedings.mlr.press/v202/wang23l.html)

        **Abstract**:

        In machine learning applications, it is well known that carefully designed learning rate (step size) schedules can significantly improve the convergence of commonly used first-order optimization algorithms. Therefore how to set step size adaptively becomes an important research question. A popular and effective method is the Polyak step size, which sets step size adaptively for gradient descent or stochastic gradient descent without the need to estimate the smoothness parameter of the objective function. However, there has not been a principled way to generalize the Polyak step size for algorithms with momentum accelerations. This paper presents a general framework to set the learning rate adaptively for first-order optimization methods with momentum, motivated by the derivation of Polyak step size. It is shown that the resulting techniques are much less sensitive to the choice of momentum parameter and may avoid the oscillation of the heavy-ball method on ill-conditioned problems. These adaptive step sizes are further extended to the stochastic settings, which are attractive choices for stochastic gradient descent with momentum. Our methods are demonstrated to be more effective for stochastic gradient methods than prior adaptive step size algorithms in large-scale machine learning tasks.

        ----

        ## [1489] Near-Minimax-Optimal Risk-Sensitive Reinforcement Learning with CVaR

        **Authors**: *Kaiwen Wang, Nathan Kallus, Wen Sun*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/wang23m.html](https://proceedings.mlr.press/v202/wang23m.html)

        **Abstract**:

        In this paper, we study risk-sensitive Reinforcement Learning (RL), focusing on the objective of Conditional Value at Risk (CVaR) with risk tolerance $\tau$. Starting with multi-arm bandits (MABs), we show the minimax CVaR regret rate is $\Omega(\sqrt{\tau^{-1}AK})$, where $A$ is the number of actions and $K$ is the number of episodes, and that it is achieved by an Upper Confidence Bound algorithm with a novel Bernstein bonus. For online RL in tabular Markov Decision Processes (MDPs), we show a minimax regret lower bound of $\Omega(\sqrt{\tau^{-1}SAK})$ (with normalized cumulative rewards), where $S$ is the number of states, and we propose a novel bonus-driven Value Iteration procedure. We show that our algorithm achieves the optimal regret of $\widetilde O(\sqrt{\tau^{-1}SAK})$ under a continuity assumption and in general attains a near-optimal regret of $\widetilde O(\tau^{-1}\sqrt{SAK})$, which is minimax-optimal for constant $\tau$. This improves on the best available bounds. By discretizing rewards appropriately, our algorithms are computationally efficient.

        ----

        ## [1490] FedHPO-Bench: A Benchmark Suite for Federated Hyperparameter Optimization

        **Authors**: *Zhen Wang, Weirui Kuang, Ce Zhang, Bolin Ding, Yaliang Li*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/wang23n.html](https://proceedings.mlr.press/v202/wang23n.html)

        **Abstract**:

        Research in the field of hyperparameter optimization (HPO) has been greatly accelerated by existing HPO benchmarks. Nonetheless, existing efforts in benchmarking all focus on HPO for traditional learning paradigms while ignoring federated learning (FL), a promising paradigm for collaboratively learning models from dispersed data. In this paper, we first identify some uniqueness of federated hyperparameter optimization (FedHPO) from various aspects, showing that existing HPO benchmarks no longer satisfy the need to study FedHPO methods. To facilitate the research of FedHPO, we propose and implement a benchmark suite FedHPO-Bench that incorporates comprehensive FedHPO problems, enables flexible customization of the function evaluations, and eases continuing extensions. We conduct extensive experiments based on FedHPO-Bench to provide the community with more insights into FedHPO. We open-sourced FedHPO-Bench at https://github.com/alibaba/FederatedScope/tree/master/benchmark/FedHPOBench.

        ----

        ## [1491] A/B Testing in Network Data with Covariate-Adaptive Randomization

        **Authors**: *Jialu Wang, Ping Li, Feifang Hu*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/wang23o.html](https://proceedings.mlr.press/v202/wang23o.html)

        **Abstract**:

        Users linked together through a network often tend to have similar behaviors. This phenomenon is usually known as network interaction. Users’ characteristics, the covariates, are often correlated with their outcomes. Therefore, one should incorporate both the covariates and the network information in a carefully designed randomization to improve the estimation of the average treatment effect (ATE) in network A/B testing. In this paper, we propose a new adaptive procedure to balance both the network and the covariates. We show that the imbalance measures with respect to the covariates and the network are $O_p(1)$. We also demonstrate the relationships between the improved balances and the increased efficiency in terms of the mean square error (MSE). Numerical studies demonstrate the advanced performance of the proposed procedure regarding the greater comparability of the treatment groups and the reduction of MSE for estimating the ATE.

        ----

        ## [1492] Learning Belief Representations for Partially Observable Deep RL

        **Authors**: *Andrew Wang, Andrew C. Li, Toryn Q. Klassen, Rodrigo Toro Icarte, Sheila A. McIlraith*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/wang23p.html](https://proceedings.mlr.press/v202/wang23p.html)

        **Abstract**:

        Many important real-world Reinforcement Learning (RL) problems involve partial observability and require policies with memory. Unfortunately, standard deep RL algorithms for partially observable settings typically condition on the full history of interactions and are notoriously difficult to train. We propose a novel deep, partially observable RL algorithm based on modelling belief states — a technique typically used when solving tabular POMDPs, but that has traditionally been difficult to apply to more complex environments. Our approach simplifies policy learning by leveraging state information at training time, that may not be available at deployment time. We do so in two ways: first, we decouple belief state modelling (via unsupervised learning) from policy optimization (via RL); and second, we propose a representation learning approach to capture a compact set of reward-relevant features of the state. Experiments demonstrate the efficacy of our approach on partially observable domains requiring information seeking and long-term memory.

        ----

        ## [1493] Warm-Start Actor-Critic: From Approximation Error to Sub-optimality Gap

        **Authors**: *Hang Wang, Sen Lin, Junshan Zhang*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/wang23q.html](https://proceedings.mlr.press/v202/wang23q.html)

        **Abstract**:

        Warm-Start reinforcement learning (RL), aided by a prior policy obtained from offline training, is emerging as a promising RL approach for practical applications. Recent empirical studies have demonstrated that the performance of Warm-Start RL can be improved quickly in some cases but become stagnant in other cases, especially when the function approximation is used. To this end, the primary objective of this work is to build a fundamental understanding on ”whether and when online learning can be significantly accelerated by a warm-start policy from offline RL?”. Specifically, we consider the widely used Actor-Critic (A-C) method with a prior policy. We first quantify the approximation errors in the Actor update and the Critic update, respectively. Next, we cast the Warm-Start A-C algorithm as Newton’s method with perturbation, and study the impact of the approximation errors on the finite-time learning performance with inaccurate Actor/Critic updates. Under some general technical conditions, we derive the upper bounds, which shed light on achieving the desired finite-learning performance in the Warm-Start A-C algorithm. In particular, our findings reveal that it is essential to reduce the algorithm bias in online learning. We also obtain lower bounds on the sub-optimality gap of the Warm-Start A-C algorithm to quantify the impact of the bias and error propagation.

        ----

        ## [1494] Slot-VAE: Object-Centric Scene Generation with Slot Attention

        **Authors**: *Yanbo Wang, Letao Liu, Justin Dauwels*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/wang23r.html](https://proceedings.mlr.press/v202/wang23r.html)

        **Abstract**:

        Slot attention has shown remarkable object-centric representation learning performance in computer vision tasks without requiring any supervision. Despite its object-centric binding ability brought by compositional modelling, as a deterministic module, slot attention lacks the ability to generate novel scenes. In this paper, we propose the Slot-VAE, a generative model that integrates slot attention with the hierarchical VAE framework for object-centric structured scene generation. For each image, the model simultaneously infers a global scene representation to capture high-level scene structure and object-centric slot representations to embed individual object components. During generation, slot representations are generated from the global scene representation to ensure coherent scene structures. Our extensive evaluation of the scene generation ability indicates that Slot-VAE outperforms slot representation-based generative baselines in terms of sample quality and scene structure accuracy.

        ----

        ## [1495] DIVISION: Memory Efficient Training via Dual Activation Precision

        **Authors**: *Guanchu Wang, Zirui Liu, Zhimeng Jiang, Ninghao Liu, Na Zou, Xia Ben Hu*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/wang23s.html](https://proceedings.mlr.press/v202/wang23s.html)

        **Abstract**:

        Activation compressed training provides a solution towards reducing the memory cost of training deep neural networks (DNNs). However, state-of-the-art work combines a search of quantization bit-width with the training, which makes the procedure complicated and less transparent. To this end, we propose a simple and effective method to compress DNN training. Our method is motivated by an instructive observation: DNN backward propagation mainly utilizes the low-frequency component (LFC) of the activation maps, while the majority of memory is for caching the high-frequency component (HFC) during the training. This indicates the HFC of activation maps is highly redundant and compressible, which inspires our proposed Dual Activation Precision (DIVISION). During the training, DIVISION preserves a high-precision copy of LFC and compresses the HFC into a light-weight copy with low numerical precision. This can significantly reduce the memory cost while maintaining a competitive model accuracy. Experiment results show DIVISION has better comprehensive performance than state-of-the-art methods, including over 10x compression of activation maps and competitive training throughput, without loss of model accuracy. The source code is available at https://github.com/guanchuwang/division.

        ----

        ## [1496] CocktailSGD: Fine-tuning Foundation Models over 500Mbps Networks

        **Authors**: *Jue Wang, Yucheng Lu, Binhang Yuan, Beidi Chen, Percy Liang, Christopher De Sa, Christopher Ré, Ce Zhang*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/wang23t.html](https://proceedings.mlr.press/v202/wang23t.html)

        **Abstract**:

        Distributed training of foundation models, especially large language models (LLMs), is communication-intensive and so has heavily relied on centralized data centers with fast interconnects. Can we train on slow networks and unlock the potential of decentralized infrastructure for foundation models? In this paper, we propose CocktailSGD, a novel communication-efficient training framework that combines three distinct compression techniques – random sparsification, top-K sparsification, and quantization – to achieve much greater compression than each individual technique alone. We justify the benefit of such a hybrid approach through a theoretical analysis of convergence. Empirically, we show that CocktailSGD achieves up to 117$\times$ compression in fine-tuning LLMs up to 20 billion parameters without hurting convergence. On a 500Mbps network, CocktailSGD only incurs $\sim$1.2$\times$ slowdown compared with data center networks.

        ----

        ## [1497] Magneto: A Foundation Transformer

        **Authors**: *Hongyu Wang, Shuming Ma, Shaohan Huang, Li Dong, Wenhui Wang, Zhiliang Peng, Yu Wu, Payal Bajaj, Saksham Singhal, Alon Benhaim, Barun Patra, Zhun Liu, Vishrav Chaudhary, Xia Song, Furu Wei*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/wang23u.html](https://proceedings.mlr.press/v202/wang23u.html)

        **Abstract**:

        A big convergence of model architectures across language, vision, speech, and multimodal is emerging. However, under the same name ”Transformers”, the above areas use different implementations for better performance, e.g., Post-LayerNorm for BERT, and Pre-LayerNorm for GPT and vision Transformers. We call for the development of Foundation Transformer for true general-purpose modeling, which serves as a go-to architecture for various tasks and modalities with guaranteed training stability. In this work, we introduce a Transformer variant, named Magneto, to fulfill the goal. Specifically, we propose Sub-LayerNorm for good expressivity, and the initialization strategy theoretically derived from DeepNet for stable scaling up. Extensive experiments demonstrate its superior performance and better stability than the de facto Transformer variants designed for various applications, including language modeling (i.e., BERT, and GPT), machine translation, vision pretraining (i.e., BEiT), speech recognition, and multimodal pretraining (i.e., BEiT-3).

        ----

        ## [1498] Direct Parameterization of Lipschitz-Bounded Deep Networks

        **Authors**: *Ruigang Wang, Ian R. Manchester*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/wang23v.html](https://proceedings.mlr.press/v202/wang23v.html)

        **Abstract**:

        This paper introduces a new parameterization of deep neural networks (both fully-connected and convolutional) with guaranteed $\ell^2$ Lipschitz bounds, i.e. limited sensitivity to input perturbations. The Lipschitz guarantees are equivalent to the tightest-known bounds based on certification via a semidefinite program (SDP). We provide a “direct” parameterization, i.e., a smooth mapping from $\mathbb R^N$ onto the set of weights satisfying the SDP-based bound. Moreover, our parameterization is complete, i.e. a neural network satisfies the SDP bound if and only if it can be represented via our parameterization. This enables training using standard gradient methods, without any inner approximation or computationally intensive tasks (e.g. projections or barrier terms) for the SDP constraint. The new parameterization can equivalently be thought of as either a new layer type (the sandwich layer), or a novel parameterization of standard feedforward networks with parameter sharing between neighbouring layers. A comprehensive set of experiments on image classification shows that sandwich layers outperform previous approaches on both empirical and certified robust accuracy. Code is available at https://github.com/acfr/LBDN.

        ----

        ## [1499] Tighter Information-Theoretic Generalization Bounds from Supersamples

        **Authors**: *Ziqiao Wang, Yongyi Mao*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/wang23w.html](https://proceedings.mlr.press/v202/wang23w.html)

        **Abstract**:

        In this work, we present a variety of novel information-theoretic generalization bounds for learning algorithms, from the supersample setting of Steinke & Zakynthinou (2020)—the setting of the “conditional mutual information” framework. Our development exploits projecting the loss pair (obtained from a training instance and a testing instance) down to a single number and correlating loss values with a Rademacher sequence (and its shifted variants). The presented bounds include square-root bounds, fast-rate bounds, including those based on variance and sharpness, and bounds for interpolating algorithms etc. We show theoretically or empirically that these bounds are tighter than all information-theoretic bounds known to date on the same supersample setting.

        ----

        ## [1500] NP-SemiSeg: When Neural Processes meet Semi-Supervised Semantic Segmentation

        **Authors**: *Jianfeng Wang, Daniela Massiceti, Xiaolin Hu, Vladimir Pavlovic, Thomas Lukasiewicz*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/wang23x.html](https://proceedings.mlr.press/v202/wang23x.html)

        **Abstract**:

        Semi-supervised semantic segmentation involves assigning pixel-wise labels to unlabeled images at training time. This is useful in a wide range of real-world applications where collecting pixel-wise labels is not feasible in time or cost. Current approaches to semi-supervised semantic segmentation work by predicting pseudo-labels for each pixel from a class-wise probability distribution output by a model. If this predicted probability distribution is incorrect, however, it leads to poor segmentation results which can have knock-on consequences in safety critical systems, like medical images or self-driving cars. It is, therefore, important to understand what a model does not know, which is mainly achieved by uncertainty quantification. Recently, neural processes (NPs) have been explored in semi-supervised image classification, and they have been a computationally efficient and effective method for uncertainty quantification. In this work, we move one step forward by adapting NPs to semi-supervised semantic segmentation, resulting in a new model called NP-SemiSeg. We experimentally evaluated NP-SemiSeg on the public benchmarks PASCAL VOC 2012 and Cityscapes, with different training settings, and the results verify its effectiveness.

        ----

        ## [1501] GC-Flow: A Graph-Based Flow Network for Effective Clustering

        **Authors**: *Tianchun Wang, Farzaneh Mirzazadeh, Xiang Zhang, Jie Chen*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/wang23y.html](https://proceedings.mlr.press/v202/wang23y.html)

        **Abstract**:

        Graph convolutional networks (GCNs) are discriminative models that directly model the class posterior $p(y|\mathbf{x})$ for semi-supervised classification of graph data. While being effective, as a representation learning approach, the node representations extracted from a GCN often miss useful information for effective clustering, because the objectives are different. In this work, we design normalizing flows that replace GCN layers, leading to a generative model that models both the class conditional likelihood $p(\mathbf{x}|y)$ and the class prior $p(y)$. The resulting neural network, GC-Flow, retains the graph convolution operations while being equipped with a Gaussian mixture representation space. It enjoys two benefits: it not only maintains the predictive power of GCN, but also produces well-separated clusters, due to the structuring of the representation space. We demonstrate these benefits on a variety of benchmark data sets. Moreover, we show that additional parameterization, such as that on the adjacency matrix used for graph convolutions, yields additional improvement in clustering.

        ----

        ## [1502] Curriculum Co-disentangled Representation Learning across Multiple Environments for Social Recommendation

        **Authors**: *Xin Wang, Zirui Pan, Yuwei Zhou, Hong Chen, Chendi Ge, Wenwu Zhu*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/wang23z.html](https://proceedings.mlr.press/v202/wang23z.html)

        **Abstract**:

        There exist complex patterns behind the decision-making processes of different individuals across different environments. For instance, in a social recommender system, various user behaviors are driven by highly entangled latent factors from two environments, i.e., consuming environment where users consume items and social environment where users connect with each other. Uncovering the disentanglement of these latent factors for users can benefit in enhanced explainability and controllability for recommendation. However, in literature there has been no work on social recommendation capable of disentangling user representations across consuming and social environments. To solve this problem, we study co-disentangled representation learning across different environments via proposing the curriculum co-disentangled representation learning (CurCoDis) model to disentangle the hidden factors for users across both consuming and social environments. To co-disentangle joint representations for user-item consumption and user-user social graph simultaneously, we partition the social graph into equal-size sub-graphs with minimum number of edges being cut, and design a curriculum weighing strategy for subgraph training through measuring the complexity of subgraphs via Descartes’ rule of signs. We further develop the prototype-routing optimization mechanism, which achieves co-disentanglement of user representations across consuming and social environments. Extensive experiments for social recommendation demonstrate that our proposed CurCoDis model can significantly outperform state-of-the-art methods on several real-world datasets.

        ----

        ## [1503] Data Efficient Neural Scaling Law via Model Reusing

        **Authors**: *Peihao Wang, Rameswar Panda, Zhangyang Wang*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/wang23aa.html](https://proceedings.mlr.press/v202/wang23aa.html)

        **Abstract**:

        The number of parameters in large transformers has been observed to grow exponentially. Despite notable performance improvements, concerns have been raised that such a growing model size will run out of data in the near future. As manifested in the neural scaling law, modern learning backbones are not data-efficient. To maintain the utility of the model capacity, training data should be increased proportionally. In this paper, we study the neural scaling law under the previously overlooked data scarcity regime, focusing on the more challenging situation where we need to train a gigantic model with a disproportionately limited supply of available training data. We find that the existing power laws underestimate the data inefficiency of large transformers. Their performance will drop significantly if the training set is insufficient. Fortunately, we discover another blessing - such a data-inefficient scaling law can be restored through a model reusing approach that warm-starts the training of a large model by initializing it using smaller models. Our empirical study shows that model reusing can effectively reproduce the power law under the data scarcity regime. When progressively applying model reusing to expand the model size, we also observe consistent performance improvement in large transformers. We release our code at: https://github.com/VITA-Group/Data-Efficient-Scaling.

        ----

        ## [1504] Deep Temporal Sets with Evidential Reinforced Attentions for Unique Behavioral Pattern Discovery

        **Authors**: *Dingrong Wang, Deep Shankar Pandey, Krishna Prasad Neupane, Zhiwei Yu, Ervine Zheng, Zhi Zheng, Qi Yu*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/wang23ab.html](https://proceedings.mlr.press/v202/wang23ab.html)

        **Abstract**:

        Machine learning-driven human behavior analysis is gaining attention in behavioral/mental healthcare, due to its potential to identify behavioral patterns that cannot be recognized by traditional assessments. Real-life applications, such as digital behavioral biomarker identification, often require the discovery of complex spatiotemporal patterns in multimodal data, which is largely under-explored. To fill this gap, we propose a novel model that integrates uniquely designed Deep Temporal Sets (DTS) with Evidential Reinforced Attentions (ERA). DTS captures complex temporal relationships in the input and generates a set-based representation, while ERA captures the policy network’s uncertainty and conducts evidence-aware exploration to locate attentive regions in behavioral data. Using child-computer interaction data as a testing platform, we demonstrate the effectiveness of DTS-ERA in differentiating children with Autism Spectrum Disorder and typically developing children based on sequential multimodal visual and touch behaviors. Comparisons with baseline methods show that our model achieves superior performance and has the potential to provide objective, quantitative, and precise analysis of complex human behaviors.

        ----

        ## [1505] Active Learning based Structural Inference

        **Authors**: *Aoran Wang, Jun Pang*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/wang23ac.html](https://proceedings.mlr.press/v202/wang23ac.html)

        **Abstract**:

        In this paper, we propose a novel framework, Active Learning based Structural Inference (ALaSI), to infer the existence of directed connections from observed agents’ states over a time period in a dynamical system. With the help of deep active learning, ALaSI is competent in learning the representation of connections with a relatively small pool of prior knowledge. Moreover, based on information theory, the proposed inter- and out-of-scope message learning pipelines are remarkably beneficial to structural inference for large dynamical systems. We evaluate ALaSI on various large datasets including simulated systems and real-world networks, to demonstrate that ALaSI is able to outperform previous methods in precisely inferring the existence of connections in large systems under either supervised learning or unsupervised learning.

        ----

        ## [1506] Better Diffusion Models Further Improve Adversarial Training

        **Authors**: *Zekai Wang, Tianyu Pang, Chao Du, Min Lin, Weiwei Liu, Shuicheng Yan*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/wang23ad.html](https://proceedings.mlr.press/v202/wang23ad.html)

        **Abstract**:

        It has been recognized that the data generated by the denoising diffusion probabilistic model (DDPM) improves adversarial training. After two years of rapid development in diffusion models, a question naturally arises: can better diffusion models further improve adversarial training? This paper gives an affirmative answer by employing the most recent diffusion model which has higher efficiency ($\sim 20$ sampling steps) and image quality (lower FID score) compared with DDPM. Our adversarially trained models achieve state-of-the-art performance on RobustBench using only generated data (no external datasets). Under the $\ell_\infty$-norm threat model with $\epsilon=8/255$, our models achieve $70.69\\%$ and $42.67\\%$ robust accuracy on CIFAR-10 and CIFAR-100, respectively, i.e. improving upon previous state-of-the-art models by $+4.58\\%$ and $+8.03\\%$. Under the $\ell_2$-norm threat model with $\epsilon=128/255$, our models achieve $84.86\\%$ on CIFAR-10 ($+4.44\\%$). These results also beat previous works that use external data. We also provide compelling results on the SVHN and TinyImageNet datasets. Our code is at https://github.com/wzekai99/DM-Improves-AT.

        ----

        ## [1507] Polarity Is All You Need to Learn and Transfer Faster

        **Authors**: *Qingyang Wang, Michael Alan Powell, Eric W. Bridgeford, Ali Geisa, Joshua T. Vogelstein*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/wang23ae.html](https://proceedings.mlr.press/v202/wang23ae.html)

        **Abstract**:

        Natural intelligences (NIs) thrive in a dynamic world - they learn quickly, sometimes with only a few samples. In contrast, artificial intelligences (AIs) typically learn with a prohibitive number of training samples and computational power. What design principle difference between NI and AI could contribute to such a discrepancy? Here, we investigate the role of weight polarity: development processes initialize NIs with advantageous polarity configurations; as NIs grow and learn, synapse magnitudes update, yet polarities are largely kept unchanged. We demonstrate with simulation and image classification tasks that if weight polarities are adequately set a priori, then networks learn with less time and data. We also explicitly illustrate situations in which a priori setting the weight polarities is disadvantageous for networks. Our work illustrates the value of weight polarities from the perspective of statistical and computational efficiency during learning.

        ----

        ## [1508] Projected Tensor Power Method for Hypergraph Community Recovery

        **Authors**: *Jinxin Wang, Yuen-Man Pun, Xiaolu Wang, Peng Wang, Anthony Man-Cho So*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/wang23af.html](https://proceedings.mlr.press/v202/wang23af.html)

        **Abstract**:

        This paper investigates the problem of exact community recovery in the symmetric $d$-uniform $(d \geq 2)$ hypergraph stochastic block model ($d$-HSBM). In this model, a $d$-uniform hypergraph with $n$ nodes is generated by first partitioning the $n$ nodes into $K\geq 2$ equal-sized disjoint communities and then generating hyperedges with a probability that depends on the community memberships of $d$ nodes. Despite the non-convex and discrete nature of the maximum likelihood estimation problem, we develop a simple yet efficient iterative method, called the projected tensor power method, to tackle it. As long as the initialization satisfies a partial recovery condition in the logarithmic degree regime of the problem, we show that our proposed method can exactly recover the hidden community structure down to the information-theoretic limit with high probability. Moreover, our proposed method exhibits a competitive time complexity of $\mathcal{O}(n\log^2n/\log\log n)$ when the aforementioned initialization condition is met. We also conduct numerical experiments to validate our theoretical findings.

        ----

        ## [1509] Estimating Possible Causal Effects with Latent Variables via Adjustment

        **Authors**: *Tian-Zuo Wang, Tian Qin, Zhi-Hua Zhou*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/wang23ag.html](https://proceedings.mlr.press/v202/wang23ag.html)

        **Abstract**:

        Causal effect identification is a fundamental task in artificial intelligence. A most ideal scenario for causal effect identification is that there is a directed acyclic graph as a prior causal graph encoding the causal relations of all relevant variables. In real tasks, however, the prior causal graph is usually not available, and some relevant variables may be latent as well. With observational data, we can only learn a partial ancestral graph (PAG), which contains some indeterminate causal relations. Since many causal graphs can correspond to one PAG, they are possibly associated with different causal effects. The aim of this paper is to estimate these possible causal effects via covariate adjustment given a PAG. This task is challenging because the number of causal graphs corresponding to a PAG grows super-exponentially with the number of variables. We propose a new graphical characterization for possible adjustment sets, and based on this, we develop the first method to determine the set of possible causal effects that are consistent with the given PAG without enumerating any causal graphs. Our method can output the same set as the enumeration method with super-exponentially less complexity. Experiments validate the effectiveness and tremendous efficiency improvement of the proposed method.

        ----

        ## [1510] InfoDiffusion: Representation Learning Using Information Maximizing Diffusion Models

        **Authors**: *Yingheng Wang, Yair Schiff, Aaron Gokaslan, Weishen Pan, Fei Wang, Christopher De Sa, Volodymyr Kuleshov*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/wang23ah.html](https://proceedings.mlr.press/v202/wang23ah.html)

        **Abstract**:

        While diffusion models excel at generating high-quality samples, their latent variables typically lack semantic meaning and are not suitable for representation learning. Here, we propose InfoDiffusion, an algorithm that augments diffusion models with low-dimensional latent variables that capture high-level factors of variation in the data. InfoDiffusion relies on a learning objective regularized with the mutual information between observed and hidden variables, which improves latent space quality and prevents the latents from being ignored by expressive diffusion-based decoders. Empirically, we find that InfoDiffusion learns disentangled and human-interpretable latent representations that are competitive with state-of-the-art generative and contrastive methods, while retaining the high sample quality of diffusion models. Our method enables manipulating the attributes of generated images and has the potential to assist tasks that require exploring a learned latent space to generate quality samples, e.g., generative design.

        ----

        ## [1511] A Robust Test for the Stationarity Assumption in Sequential Decision Making

        **Authors**: *Jitao Wang, Chengchun Shi, Zhenke Wu*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/wang23ai.html](https://proceedings.mlr.press/v202/wang23ai.html)

        **Abstract**:

        Reinforcement learning (RL) is a powerful technique that allows an autonomous agent to learn an optimal policy to maximize the expected return. The optimality of various RL algorithms relies on the stationarity assumption, which requires time-invariant state transition and reward functions. However, deviations from stationarity over extended periods often occur in real-world applications like robotics control, health care and digital marketing, resulting in suboptimal policies learned under stationary assumptions. In this paper, we propose a model-based doubly robust procedure for testing the stationarity assumption and detecting change points in offline RL settings with certain degree of homogeneity. Our proposed testing procedure is robust to model misspecifications and can effectively control type-I error while achieving high statistical power, especially in high-dimensional settings. Extensive comparative simulations and a real-world interventional mobile health example illustrate the advantages of our method in detecting change points and optimizing long-term rewards in high-dimensional, non-stationary environments.

        ----

        ## [1512] GEAR: A GPU-Centric Experience Replay System for Large Reinforcement Learning Models

        **Authors**: *Hanjing Wang, Man-Kit Sit, Congjie He, Ying Wen, Weinan Zhang, Jun Wang, Yaodong Yang, Luo Mai*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/wang23aj.html](https://proceedings.mlr.press/v202/wang23aj.html)

        **Abstract**:

        This paper introduces a distributed, GPU-centric experience replay system, GEAR, designed to perform scalable reinforcement learning (RL) with large sequence models (such as transformers). With such models, existing systems such as Reverb face considerable bottlenecks in memory, computation, and communication. GEAR, however, optimizes memory efficiency by enabling the memory resources on GPU servers (including host memory and device memory) to manage trajectory data. Furthermore, it facilitates decentralized GPU devices to expedite various trajectory selection strategies, circumventing computational bottlenecks. GEAR is equipped with GPU kernels capable of collecting trajectories using zero-copy access to host memory, along with remote-directed-memory access over InfiniBand, improving communication efficiency. Cluster experiments have shown that GEAR can achieve performance levels up to 6× greater than Reverb when training state-of-the-art large RL models. GEAR is open-sourced at https:// github.com/bigrl-team/gear.

        ----

        ## [1513] Effective and Efficient Structural Inference with Reservoir Computing

        **Authors**: *Aoran Wang, Tsz Pan Tong, Jun Pang*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/wang23ak.html](https://proceedings.mlr.press/v202/wang23ak.html)

        **Abstract**:

        In this paper, we present an effective and efficient structural inference approach by integrating a Reservoir Computing (RC) network into a Variational Auto-encoder-based (VAE-based) structural inference framework. With the help of Bi-level Optimization, the backbone VAE-based method follows the Information Bottleneck principle and infers a general adjacency matrix in its latent space; the RC net substitutes the partial role of the decoder and encourages the whole approach to perform further steps of gradient descent based on limited available data. The experimental results on various datasets including biological networks, simulated fMRI data, and physical simulations show the effectiveness and efficiency of our proposed method for structural inference, either with much fewer trajectories or with much shorter trajectories compared with previous works.

        ----

        ## [1514] Optimal Goal-Reaching Reinforcement Learning via Quasimetric Learning

        **Authors**: *Tongzhou Wang, Antonio Torralba, Phillip Isola, Amy Zhang*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/wang23al.html](https://proceedings.mlr.press/v202/wang23al.html)

        **Abstract**:

        In goal-reaching reinforcement learning (RL), the optimal value function has a particular geometry, called quasimetrics structure. This paper introduces Quasimetric Reinforcement Learning (QRL), a new RL method that utilizes quasimetric models to learn optimal value functions. Distinct from prior approaches, the QRL objective is specifically designed for quasimetrics, and provides strong theoretical recovery guarantees. Empirically, we conduct thorough analyses on a discretized MountainCar environment, identifying properties of QRL and its advantages over alternatives. On offline and online goal-reaching benchmarks, QRL also demonstrates improved sample efficiency and performance, across both state-based and image-based observations.

        ----

        ## [1515] Model-Free Robust Average-Reward Reinforcement Learning

        **Authors**: *Yue Wang, Alvaro Velasquez, George K. Atia, Ashley Prater-Bennette, Shaofeng Zou*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/wang23am.html](https://proceedings.mlr.press/v202/wang23am.html)

        **Abstract**:

        Robust Markov decision processes (MDPs) address the challenge of model uncertainty by optimizing the worst-case performance over an uncertainty set of MDPs. In this paper, we focus on the robust average-reward MDPs under the model-free setting. We first theoretically characterize the structure of solutions to the robust average-reward Bellman equation, which is essential for our later convergence analysis. We then design two model-free algorithms, robust relative value iteration (RVI) TD and robust RVI Q-learning, and theoretically prove their convergence to the optimal solution. We provide several widely used uncertainty sets as examples, including those defined by the contamination model, total variation, Chi-squared divergence, Kullback-Leibler (KL) divergence, and Wasserstein distance.

        ----

        ## [1516] Live in the Moment: Learning Dynamics Model Adapted to Evolving Policy

        **Authors**: *Xiyao Wang, Wichayaporn Wongkamjan, Ruonan Jia, Furong Huang*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/wang23an.html](https://proceedings.mlr.press/v202/wang23an.html)

        **Abstract**:

        Model-based reinforcement learning (RL) often achieves higher sample efficiency in practice than model-free RL by learning a dynamics model to generate samples for policy learning. Previous works learn a dynamics model that fits under the empirical state-action visitation distribution for all historical policies, i.e., the sample replay buffer. However, in this paper, we observe that fitting the dynamics model under the distribution for all historical policies does not necessarily benefit model prediction for the current policy since the policy in use is constantly evolving over time. The evolving policy during training will cause state-action visitation distribution shifts. We theoretically analyze how this distribution shift over historical policies affects the model learning and model rollouts. We then propose a novel dynamics model learning method, named Policy-adapted Dynamics Model Learning (PDML). PDML dynamically adjusts the historical policy mixture distribution to ensure the learned model can continually adapt to the state-action visitation distribution of the evolving policy. Experiments on a range of continuous control environments in MuJoCo show that PDML achieves significant improvement in sample efficiency and higher asymptotic performance combined with the state-of-the-art model-based RL methods.

        ----

        ## [1517] Learning to Bid in Repeated First-Price Auctions with Budgets

        **Authors**: *Qian Wang, Zongjun Yang, Xiaotie Deng, Yuqing Kong*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/wang23ao.html](https://proceedings.mlr.press/v202/wang23ao.html)

        **Abstract**:

        Budget management strategies in repeated auctions have received growing attention in online advertising markets. However, previous work on budget management in online bidding mainly focused on second-price auctions. The rapid shift from second-price auctions to first-price auctions for online ads in recent years has motivated the challenging question of how to bid in repeated first-price auctions while controlling budgets. In this work, we study the problem of learning in repeated first-price auctions with budgets. We design a dual-based algorithm that can achieve a near-optimal $\widetilde{O}(\sqrt{T})$ regret with full information feedback where the maximum competing bid is always revealed after each auction. We further consider the setting with one-sided information feedback where only the winning bid is revealed after each auction. We show that our modified algorithm can still achieve an $\widetilde{O}(\sqrt{T})$ regret with mild assumptions on the bidder’s value distribution. Finally, we complement the theoretical results with numerical experiments to confirm the effectiveness of our budget management policy.

        ----

        ## [1518] Network Effects in Performative Prediction Games

        **Authors**: *Xiaolu Wang, Chung-Yiu Yau, Hoi-To Wai*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/wang23ap.html](https://proceedings.mlr.press/v202/wang23ap.html)

        **Abstract**:

        This paper studies the multi-agent performative prediction (Multi-PP) games over multiplex networks. We consider a distributed learning setting where agents partially cooperate on an agent network, while during learning, the data samples drawn depend on the prediction models of the agent itself and neighboring agents on a population network. The dynamics of Multi-PP games is hence affected by the interplay between both networks. This paper concentrates on this Multi-PP game with the following contributions. Firstly, we analyze sufficient conditions for the existence of the performative stable equilibrium (PSE) and Nash equilibrium (NE) of the Multi-PP games. Secondly, we analyze the changes to the equilibrium induced by perturbed data distributions, and derive the closed-form solutions where the network topologies are explicit. Our results connect the existence of PSE/NE with strengths of agents’ cooperation, and the changes of equilibrium solutions across agents with their node centrality, etc. Lastly, we show that a stochastic gradient descent (SGD) based distributed learning procedure finds the PSE under the said sufficient condition. Numerical illustrations on the network effects in Multi-PP games corroborate our findings.

        ----

        ## [1519] Robustly Learning a Single Neuron via Sharpness

        **Authors**: *Puqian Wang, Nikos Zarifis, Ilias Diakonikolas, Jelena Diakonikolas*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/wang23aq.html](https://proceedings.mlr.press/v202/wang23aq.html)

        **Abstract**:

        We study the problem of learning a single neuron with respect to the $L_2^2$-loss in the presence of adversarial label noise. We give an efficient algorithm that, for a broad family of activations including ReLUs, approximates the optimal $L_2^2$-error within a constant factor. Notably, our algorithm succeeds under much milder distributional assumptions compared to prior work. The key ingredient enabling our results is a novel connection to local error bounds from optimization theory.

        ----

        ## [1520] DualHSIC: HSIC-Bottleneck and Alignment for Continual Learning

        **Authors**: *Zifeng Wang, Zheng Zhan, Yifan Gong, Yucai Shao, Stratis Ioannidis, Yanzhi Wang, Jennifer G. Dy*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/wang23ar.html](https://proceedings.mlr.press/v202/wang23ar.html)

        **Abstract**:

        Rehearsal-based approaches are a mainstay of continual learning (CL). They mitigate the catastrophic forgetting problem by maintaining a small fixed-size buffer with a subset of data from past tasks. While most rehearsal-based approaches exploit the knowledge from buffered past data, little attention is paid to inter-task relationships and to critical task-specific and task-invariant knowledge. By appropriately leveraging inter-task relationships, we propose a novel CL method, named DualHSIC, to boost the performance of existing rehearsal-based methods in a simple yet effective way. DualHSIC consists of two complementary components that stem from the so-called Hilbert Schmidt independence criterion (HSIC): HSIC-Bottleneck for Rehearsal (HBR) lessens the inter-task interference and HSIC Alignment (HA) promotes task-invariant knowledge sharing. Extensive experiments show that DualHSIC can be seamlessly plugged into existing rehearsal-based methods for consistent performance improvements, outperforming recent state-of-the-art regularization-enhanced rehearsal methods.

        ----

        ## [1521] Enforcing Hard Constraints with Soft Barriers: Safe Reinforcement Learning in Unknown Stochastic Environments

        **Authors**: *Yixuan Wang, Simon Sinong Zhan, Ruochen Jiao, Zhilu Wang, Wanxin Jin, Zhuoran Yang, Zhaoran Wang, Chao Huang, Qi Zhu*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/wang23as.html](https://proceedings.mlr.press/v202/wang23as.html)

        **Abstract**:

        It is quite challenging to ensure the safety of reinforcement learning (RL) agents in an unknown and stochastic environment under hard constraints that require the system state not to reach certain specified unsafe regions. Many popular safe RL methods such as those based on the Constrained Markov Decision Process (CMDP) paradigm formulate safety violations in a cost function and try to constrain the expectation of cumulative cost under a threshold. However, it is often difficult to effectively capture and enforce hard reachability-based safety constraints indirectly with such constraints on safety violation cost. In this work, we leverage the notion of barrier function to explicitly encode the hard safety chance constraints, and given that the environment is unknown, relax them to our design of generative-model-based soft barrier functions. Based on such soft barriers, we propose a novel safe RL approach with bi-level optimization that can jointly learn the unknown environment and optimize the control policy, while effectively avoiding the unsafe region with safety probability optimization. Experiments on a set of examples demonstrate that our approach can effectively enforce hard safety chance constraints and significantly outperform CMDP-based baseline methods in system safe rates measured via simulations.

        ----

        ## [1522] LinSATNet: The Positive Linear Satisfiability Neural Networks

        **Authors**: *Runzhong Wang, Yunhao Zhang, Ziao Guo, Tianyi Chen, Xiaokang Yang, Junchi Yan*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/wang23at.html](https://proceedings.mlr.press/v202/wang23at.html)

        **Abstract**:

        Encoding constraints into neural networks is attractive. This paper studies how to introduce the popular positive linear satisfiability to neural networks. We propose the first differentiable satisfiability layer based on an extension of the classic Sinkhorn algorithm for jointly encoding multiple sets of marginal distributions. We further theoretically characterize the convergence property of the Sinkhorn algorithm for multiple marginals, and the underlying formulation is also derived. In contrast to the sequential decision e.g. reinforcement learning-based solvers, we showcase our technique in solving constrained (specifically satisfiability) problems by one-shot neural networks, including i) a neural routing solver learned without supervision of optimal solutions; ii) a partial graph matching network handling graphs with unmatchable outliers on both sides; iii) a predictive network for financial portfolios with continuous constraints. To our knowledge, there exists no one-shot neural solver for these scenarios when they are formulated as satisfiability problems. Source code is available at https://github.com/Thinklab-SJTU/LinSATNet.

        ----

        ## [1523] Offline Meta Reinforcement Learning with In-Distribution Online Adaptation

        **Authors**: *Jianhao Wang, Jin Zhang, Haozhe Jiang, Junyu Zhang, Liwei Wang, Chongjie Zhang*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/wang23au.html](https://proceedings.mlr.press/v202/wang23au.html)

        **Abstract**:

        Recent offline meta-reinforcement learning (meta-RL) methods typically utilize task-dependent behavior policies (e.g., training RL agents on each individual task) to collect a multi-task dataset. However, these methods always require extra information for fast adaptation, such as offline context for testing tasks. To address this problem, we first formally characterize a unique challenge in offline meta-RL: transition-reward distribution shift between offline datasets and online adaptation. Our theory finds that out-of-distribution adaptation episodes may lead to unreliable policy evaluation and that online adaptation with in-distribution episodes can ensure adaptation performance guarantee. Based on these theoretical insights, we propose a novel adaptation framework, called In-Distribution online Adaptation with uncertainty Quantification (IDAQ), which generates in-distribution context using a given uncertainty quantification and performs effective task belief inference to address new tasks. We find a return-based uncertainty quantification for IDAQ that performs effectively. Experiments show that IDAQ achieves state-of-the-art performance on the Meta-World ML1 benchmark compared to baselines with/without offline adaptation.

        ----

        ## [1524] Reachability-Aware Laplacian Representation in Reinforcement Learning

        **Authors**: *Kaixin Wang, Kuangqi Zhou, Jiashi Feng, Bryan Hooi, Xinchao Wang*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/wang23av.html](https://proceedings.mlr.press/v202/wang23av.html)

        **Abstract**:

        In Reinforcement Learning (RL), Laplacian Representation (LapRep) is a task-agnostic state representation that encodes the geometry of the environment. A desirable property of LapRep stated in prior works is that the Euclidean distance in the LapRep space roughly reflects the reachability between states, which motivates the usage of this distance for reward shaping. However, we find that LapRep does not necessarily have this property in general: two states having a small distance under LapRep can actually be far away in the environment. Such a mismatch would impede the learning process in reward shaping. To fix this issue, we introduce a Reachability-Aware Laplacian Representation (RA-LapRep), by properly scaling each dimension of LapRep. Despite the simplicity, we demonstrate that RA-LapRep can better capture the inter-state reachability as compared to LapRep, through both theoretical explanations and experimental results. Additionally, we show that this improvement yields a significant boost in reward shaping performance and benefits bottleneck state discovery.

        ----

        ## [1525] PPG Reloaded: An Empirical Study on What Matters in Phasic Policy Gradient

        **Authors**: *Kaixin Wang, Daquan Zhou, Jiashi Feng, Shie Mannor*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/wang23aw.html](https://proceedings.mlr.press/v202/wang23aw.html)

        **Abstract**:

        In model-free reinforcement learning, recent methods based on a phasic policy gradient (PPG) framework have shown impressive improvements in sample efficiency and zero-shot generalization on the challenging Procgen benchmark. In PPG, two design choices are believed to be the key contributing factors to its superior performance over PPO: the high level of value sample reuse and the low frequency of feature distillation. However, through an extensive empirical study, we unveil that policy regularization and data diversity are what actually matters. In particular, we can achieve the same level of performance with low value sample reuse and frequent feature distillation, as long as the policy regularization strength and data diversity are preserved. In addition, we can maintain the high performance of PPG while reducing the computational cost to a similar level as PPO. Our comprehensive study covers all 16 Procgen games in both sample efficiency and generalization setups. We hope it can advance the understanding of PPG and provide insights for future works.

        ----

        ## [1526] On Heterogeneous Treatment Effects in Heterogeneous Causal Graphs

        **Authors**: *Richard A. Watson, Hengrui Cai, Xinming An, Samuel A. McLean, Rui Song*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/watson23a.html](https://proceedings.mlr.press/v202/watson23a.html)

        **Abstract**:

        Heterogeneity and comorbidity are two interwoven challenges associated with various healthcare problems that greatly hampered research on developing effective treatment and understanding of the underlying neurobiological mechanism. Very few studies have been conducted to investigate heterogeneous causal effects (HCEs) in graphical contexts due to the lack of statistical methods. To characterize this heterogeneity, we first conceptualize heterogeneous causal graphs (HCGs) by generalizing the causal graphical model with confounder-based interactions and multiple mediators. Such confounders with an interaction with the treatment are known as moderators. This allows us to flexibly produce HCGs given different moderators and explicitly characterize HCEs from the treatment or potential mediators on the outcome. We establish the theoretical forms of HCEs and derive their properties at the individual level in both linear and nonlinear models. An interactive structural learning is developed to estimate the complex HCGs and HCEs with confidence intervals provided. Our method is empirically justified by extensive simulations and its practical usefulness is illustrated by exploring causality among psychiatric disorders for trauma survivors. Code implementing the proposed algorithm is open-source and publicly available at: https://github.com/richard-watson/ISL.

        ----

        ## [1527] Nonparametric Extensions of Randomized Response for Private Confidence Sets

        **Authors**: *Ian Waudby-Smith, Zhiwei Steven Wu, Aaditya Ramdas*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/waudby-smith23a.html](https://proceedings.mlr.press/v202/waudby-smith23a.html)

        **Abstract**:

        This work derives methods for performing nonparametric, nonasymptotic statistical inference for population means under the constraint of local differential privacy (LDP). Given bounded observations $(X_1, …, X_n)$ with mean $\mu^\star$ that are privatized into $(Z_1, …, Z_n)$, we present confidence intervals (CI) and time-uniform confidence sequences (CS) for $\mu^\star$ when only given access to the privatized data. To achieve this, we introduce a nonparametric and sequentially interactive generalization of Warner’s famous “randomized response” mechanism, satisfying LDP for arbitrary bounded random variables, and then provide CIs and CSs for their means given access to the resulting privatized observations. For example, our results yield private analogues of Hoeffding’s inequality in both fixed-time and time-uniform regimes. We extend these Hoeffding-type CSs to capture time-varying (non-stationary) means, and conclude by illustrating how these methods can be used to conduct private online A/B tests.

        ----

        ## [1528] Global optimality for Euclidean CCCP under Riemannian convexity

        **Authors**: *Melanie Weber, Suvrit Sra*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/weber23a.html](https://proceedings.mlr.press/v202/weber23a.html)

        **Abstract**:

        We study geodesically convex (g-convex) problems that can be written as a difference of Euclidean convex functions. This structure arises in key applications such as matrix scaling, M- estimators of scatter matrices, and Brascamp-Lieb inequalities. In particular, we exploit this structure to make use of the Convex-Concave Procedure (CCCP), which helps us bypass potentially expensive Riemannian operations and leads to very competitive solvers. Importantly, unlike existing theory for CCCP that ensures convergence to stationary points, we exploit the overall g-convexity structure and provide iteration complexity results for global optimality. We illustrate our results by specializing them to a few concrete optimization problems that have been previously studied in the machine learning literature. We hope our work spurs the study of mixed Euclidean-Riemannian optimization algorithms.

        ----

        ## [1529] A Universal Unbiased Method for Classification from Aggregate Observations

        **Authors**: *Zixi Wei, Lei Feng, Bo Han, Tongliang Liu, Gang Niu, Xiaofeng Zhu, Heng Tao Shen*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/wei23a.html](https://proceedings.mlr.press/v202/wei23a.html)

        **Abstract**:

        In conventional supervised classification, true labels are required for individual instances. However, it could be prohibitive to collect the true labels for individual instances, due to privacy concerns or unaffordable annotation costs. This motivates the study on classification from aggregate observations (CFAO), where the supervision is provided to groups of instances, instead of individual instances. CFAO is a generalized learning framework that contains various learning problems, such as multiple-instance learning and learning from label proportions. The goal of this paper is to present a novel universal method of CFAO, which holds an unbiased estimator of the classification risk for arbitrary losses—previous research failed to achieve this goal. Practically, our method works by weighing the importance of each instance and each label in the group, which provides purified supervision for the classifier to learn. Theoretically, our proposed method not only guarantees the risk consistency due to the unbiased risk estimator but also can be compatible with arbitrary losses. Extensive experiments on various problems of CFAO demonstrate the superiority of our proposed method.

        ----

        ## [1530] NTK-approximating MLP Fusion for Efficient Language Model Fine-tuning

        **Authors**: *Tianxin Wei, Zeming Guo, Yifan Chen, Jingrui He*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/wei23b.html](https://proceedings.mlr.press/v202/wei23b.html)

        **Abstract**:

        Fine-tuning a pre-trained language model (PLM) emerges as the predominant strategy in many natural language processing applications. However, even fine-tuning the PLMs and doing inference are expensive, especially on edge devices with low computing power. Some general approaches (e.g. quantization and distillation) have been widely studied to reduce the compute/memory of PLM fine-tuning, while very few one-shot compression techniques are explored. In this paper, we investigate the neural tangent kernel (NTK)–which reveals the gradient descent dynamics of neural networks–of the multilayer perceptrons (MLP) modules in a PLM and propose to coin a lightweight PLM through NTK-approximating MLP fusion. To achieve this, we reconsider the MLP as a bundle of sub-MLPs, and cluster them into a given number of centroids, which can then be restored as a compressed MLP and surprisingly shown to well approximate the NTK of the original PLM. Extensive experiments of PLM fine-tuning on both natural language understanding (NLU) and generation (NLG) tasks are provided to verify the effectiveness of the proposed method MLP fusion. Our code is available at https://github.com/weitianxin/MLP_Fusion.

        ----

        ## [1531] Boosting Graph Contrastive Learning via Graph Contrastive Saliency

        **Authors**: *Chunyu Wei, Yu Wang, Bing Bai, Kai Ni, David Brady, Lu Fang*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/wei23c.html](https://proceedings.mlr.press/v202/wei23c.html)

        **Abstract**:

        Graph augmentation plays a crucial role in achieving good generalization for contrastive graph self-supervised learning. However, mainstream Graph Contrastive Learning (GCL) often favors random graph augmentations, by relying on random node dropout or edge perturbation on graphs. Random augmentations may inevitably lead to semantic information corruption during the training, and force the network to mistakenly focus on semantically irrelevant environmental background structures. To address these limitations and to improve generalization, we propose a novel self-supervised learning framework for GCL, which can adaptively screen the semantic-related substructure in graphs by capitalizing on the proposed gradient-based Graph Contrastive Saliency (GCS). The goal is to identify the most semantically discriminative structures of a graph via contrastive learning, such that we can generate semantically meaningful augmentations by leveraging on saliency. Empirical evidence on 16 benchmark datasets demonstrates the exclusive merits of the GCS-based framework. We also provide rigorous theoretical justification for GCS’s robustness properties. Code is available at https://github.com/GCS2023/GCS .

        ----

        ## [1532] Set-membership Belief State-based Reinforcement Learning for POMDPs

        **Authors**: *Wei Wei, Lijun Zhang, Lin Li, Huizhong Song, Jiye Liang*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/wei23d.html](https://proceedings.mlr.press/v202/wei23d.html)

        **Abstract**:

        Reinforcement learning (RL) has made significant progress in areas such as Atari games and robotic control, where the agents have perfect sensing capabilities. However, in many real-world sequential decision-making tasks, the observation data could be noisy or incomplete due to the intrinsic low quality of the sensors or unexpected malfunctions; that is, the agent’s perceptions are rarely perfect. The current POMDP RL methods, such as particle-based and Gaussian-based, can only provide a probability estimate of hidden states rather than certain belief regions, which may lead to inefficient and even wrong decision-making. This paper proposes a novel algorithm called Set-membership Belief state-based Reinforcement Learning (SBRL), which consists of two parts: a Set-membership Belief state learning Model (SBM) for learning bounded belief state sets and an RL controller for making decisions based on SBM. We prove that our belief estimation method can provide a series of belief state sets that always contain the true states under the unknown-but-bounded (UBB) noise. The effectiveness of the proposed method is verified on a collection of benchmark tasks, and the results show that our method outperforms the state-of-the-art methods.

        ----

        ## [1533] Mitigating Memorization of Noisy Labels by Clipping the Model Prediction

        **Authors**: *Hongxin Wei, Huiping Zhuang, Renchunzi Xie, Lei Feng, Gang Niu, Bo An, Yixuan Li*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/wei23e.html](https://proceedings.mlr.press/v202/wei23e.html)

        **Abstract**:

        In the presence of noisy labels, designing robust loss functions is critical for securing the generalization performance of deep neural networks. Cross Entropy (CE) loss has been shown to be not robust to noisy labels due to its unboundedness. To alleviate this issue, existing works typically design specialized robust losses with the symmetric condition, which usually lead to the underfitting issue. In this paper, our key idea is to induce a loss bound at the logit level, thus universally enhancing the noise robustness of existing losses. Specifically, we propose logit clipping (LogitClip), which clamps the norm of the logit vector to ensure that it is upper bounded by a constant. In this manner, CE loss equipped with our LogitClip method is effectively bounded, mitigating the overfitting to examples with noisy labels. Moreover, we present theoretical analyses to certify the noise-tolerant ability of LogitClip. Extensive experiments show that LogitClip not only significantly improves the noise robustness of CE loss, but also broadly enhances the generalization performance of popular robust losses.

        ----

        ## [1534] Graphically Structured Diffusion Models

        **Authors**: *Christian Dietrich Weilbach, William Harvey, Frank Wood*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/weilbach23a.html](https://proceedings.mlr.press/v202/weilbach23a.html)

        **Abstract**:

        We introduce a framework for automatically defining and learning deep generative models with problem-specific structure. We tackle problem domains that are more traditionally solved by algorithms such as sorting, constraint satisfaction for Sudoku, and matrix factorization. Concretely, we train diffusion models with an architecture tailored to the problem specification. This problem specification should contain a graphical model describing relationships between variables, and often benefits from explicit representation of subcomputations. Permutation invariances can also be exploited. Across a diverse set of experiments we improve the scaling relationship between problem dimension and our model’s performance, in terms of both training time and final accuracy. Our code can be found at https://github.com/plai-group/gsdm.

        ----

        ## [1535] Expectation-Complete Graph Representations with Homomorphisms

        **Authors**: *Pascal Welke, Maximilian Thiessen, Fabian Jogl, Thomas Gärtner*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/welke23a.html](https://proceedings.mlr.press/v202/welke23a.html)

        **Abstract**:

        We investigate novel random graph embeddings that can be computed in expected polynomial time and that are able to distinguish all non-isomorphic graphs in expectation. Previous graph embeddings have limited expressiveness and either cannot distinguish all graphs or cannot be computed efficiently for every graph. To be able to approximate arbitrary functions on graphs, we are interested in efficient alternatives that become arbitrarily expressive with increasing resources. Our approach is based on Lovász’ characterisation of graph isomorphism through an infinite dimensional vector of homomorphism counts. Our empirical evaluation shows competitive results on several benchmark graph learning tasks.

        ----

        ## [1536] A Conditional Normalizing Flow for Accelerated Multi-Coil MR Imaging

        **Authors**: *Jeffrey Wen, Rizwan Ahmad, Philip Schniter*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/wen23a.html](https://proceedings.mlr.press/v202/wen23a.html)

        **Abstract**:

        Accelerated magnetic resonance (MR) imaging attempts to reduce acquisition time by collecting data below the Nyquist rate. As an ill-posed inverse problem, many plausible solutions exist, yet the majority of deep learning approaches generate only a single solution. We instead focus on sampling from the posterior distribution, which provides more comprehensive information for downstream inference tasks. To do this, we design a novel conditional normalizing flow (CNF) that infers the signal component in the measurement operator’s nullspace, which is later combined with measured data to form complete images. Using fastMRI brain and knee data, we demonstrate fast inference and accuracy that surpasses recent posterior sampling techniques for MRI. Code is available at https://github.com/jwen307/mri_cnf

        ----

        ## [1537] Optimizing Mode Connectivity for Class Incremental Learning

        **Authors**: *Haitao Wen, Haoyang Cheng, Heqian Qiu, Lanxiao Wang, Lili Pan, Hongliang Li*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/wen23b.html](https://proceedings.mlr.press/v202/wen23b.html)

        **Abstract**:

        Class incremental learning (CIL) is one of the most challenging scenarios in continual learning. Existing work mainly focuses on strategies like memory replay, regularization, or dynamic architecture but ignores a crucial aspect: mode connectivity. Recent studies have shown that different minima can be connected by a low-loss valley, and ensembling over the valley shows improved performance and robustness. Motivated by this, we try to investigate the connectivity in CIL and find that the high-loss ridge exists along the linear connection between two adjacent continual minima. To dodge the ridge, we propose parameter-saving OPtimizing Connectivity (OPC) based on Fourier series and gradient projection for finding the low-loss path between minima. The optimized path provides infinite low-loss solutions. We further propose EOPC to ensemble points within a local bent cylinder to improve performance on learned tasks. Our scheme can serve as a plug-in unit, extensive experiments on CIFAR-100, ImageNet-100, and ImageNet-1K show consistent improvements when adapting EOPC to existing representative CIL methods. Our code is available at https://github.com/HaitaoWen/EOPC.

        ----

        ## [1538] Towards Learning Geometric Eigen-Lengths Crucial for Fitting Tasks

        **Authors**: *Yijia Weng, Kaichun Mo, Ruoxi Shi, Yanchao Yang, Leonidas J. Guibas*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/weng23a.html](https://proceedings.mlr.press/v202/weng23a.html)

        **Abstract**:

        Some extremely low-dimensional yet crucial geometric eigen-lengths often determine the success of some geometric tasks. For example, the height of an object is important to measure to check if it can fit between the shelves of a cabinet, while the width of a couch is crucial when trying to move it through a doorway. Humans have materialized such crucial geometric eigen-lengths in common sense since they are very useful in serving as succinct yet effective, highly interpretable, and universal object representations. However, it remains obscure and underexplored if learning systems can be equipped with similar capabilities of automatically discovering such key geometric quantities from doing tasks. In this work, we therefore for the first time formulate and propose a novel learning problem on this question and set up a benchmark suite including tasks, data, and evaluation metrics for studying the problem. We focus on a family of common fitting tasks as the testbed for the proposed learning problem. We explore potential solutions and demonstrate the feasibility of learning eigen-lengths from simply observing successful and failed fitting trials. We also attempt geometric grounding for more accurate eigen-length measurement and study the reusability of the learned geometric eigen-lengths across multiple tasks. Our work marks the first exploratory step toward learning crucial geometric eigen-lengths and we hope it can inspire future research in tackling this important yet underexplored problem.

        ----

        ## [1539] Open-VCLIP: Transforming CLIP to an Open-vocabulary Video Model via Interpolated Weight Optimization

        **Authors**: *Zejia Weng, Xitong Yang, Ang Li, Zuxuan Wu, Yu-Gang Jiang*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/weng23b.html](https://proceedings.mlr.press/v202/weng23b.html)

        **Abstract**:

        Contrastive Language-Image Pretraining (CLIP) has demonstrated impressive zero-shot learning abilities for image understanding, yet limited effort has been made to investigate CLIP for zero-shot video recognition. We introduce Open-VCLIP, a simple yet effective approach that transforms CLIP into a strong zero-shot video classifier that can recognize unseen actions and events at test time. Our framework extends CLIP with minimal modifications to model spatial-temporal relationships in videos, making it a specialized video classifier, while striving for generalization. We formally show that training an Open-VCLIP is equivalent to continual learning with zero historical data. To address this problem, we propose Interpolated Weight Optimization, which utilizes the benefit of weight interpolation in both training and test time. We evaluate our method on three popular and challenging action recognition datasets following various zero-shot evaluation protocols and we demonstrate our approach outperforms state-of-the-art methods by clear margins. In particular, we achieve 87.9%, 58.3%, 81.1% zero-shot accuracy on UCF, HMDB and Kinetics-600 respectively, outperforming state-of-the-art methods by 8.3%, 7.8% and 12.2%. Code is released at https://github.com/wengzejia1/Open-VCLIP.

        ----

        ## [1540] Fully-Adaptive Composition in Differential Privacy

        **Authors**: *Justin Whitehouse, Aaditya Ramdas, Ryan Rogers, Steven Wu*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/whitehouse23a.html](https://proceedings.mlr.press/v202/whitehouse23a.html)

        **Abstract**:

        Composition is a key feature of differential privacy. Well-known advanced composition theorems allow one to query a private database quadratically more times than basic privacy composition would permit. However, these results require that the privacy parameters of all algorithms be fixed before interacting with the data. To address this, Rogers et al. introduced fully adaptive composition, wherein both algorithms and their privacy parameters can be selected adaptively. They defined two probabilistic objects to measure privacy in adaptive composition: privacy filters, which provide differential privacy guarantees for composed interactions, and privacy odometers, time-uniform bounds on privacy loss. There are substantial gaps between advanced composition and existing filters and odometers. First, existing filters place stronger assumptions on the algorithms being composed. Second, these odometers and filters suffer from large constants, making them impractical. We construct filters that match the rates of advanced composition, including constants, despite allowing for adaptively chosen privacy parameters. En route we also derive a privacy filter for approximate zCDP. We also construct several general families of odometers. These odometers match the tightness of advanced composition at an arbitrary, preselected point in time, or at all points in time simultaneously, up to a doubly-logarithmic factor. We obtain our results by leveraging advances in martingale concentration. In sum, we show that fully adaptive privacy is obtainable at almost no loss.

        ----

        ## [1541] Scalable Set Encoding with Universal Mini-Batch Consistency and Unbiased Full Set Gradient Approximation

        **Authors**: *Jeffrey Willette, Seanie Lee, Bruno Andreis, Kenji Kawaguchi, Juho Lee, Sung Ju Hwang*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/willette23a.html](https://proceedings.mlr.press/v202/willette23a.html)

        **Abstract**:

        Recent work on mini-batch consistency (MBC) for set functions has brought attention to the need for sequentially processing and aggregating chunks of a partitioned set while guaranteeing the same output for all partitions. However, existing constraints on MBC architectures lead to models with limited expressive power. Additionally, prior work has not addressed how to deal with large sets during training when the full set gradient is required. To address these issues, we propose a Universally MBC (UMBC) class of set functions which can be used in conjunction with arbitrary non-MBC components while still satisfying MBC, enabling a wider range of function classes to be used in MBC settings. Furthermore, we propose an efficient MBC training algorithm which gives an unbiased approximation of the full set gradient and has a constant memory overhead for any set size for both train- and test-time. We conduct extensive experiments including image completion, text classification, unsupervised clustering, and cancer detection on high-resolution images to verify the efficiency and efficacy of our scalable set encoding framework. Our code is available at github.com/jeffwillette/umbc

        ----

        ## [1542] Flexible Phase Dynamics for Bio-Plausible Contrastive Learning

        **Authors**: *Ezekiel Williams, Colin Bredenberg, Guillaume Lajoie*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/williams23a.html](https://proceedings.mlr.press/v202/williams23a.html)

        **Abstract**:

        Many learning algorithms used as normative models in neuroscience or as candidate approaches for learning on neuromorphic chips learn by contrasting one set of network states with another. These Contrastive Learning (CL) algorithms are traditionally implemented with rigid, temporally non-local, and periodic learning dynamics, that could limit the range of physical systems capable of harnessing CL. In this study, we build on recent work exploring how CL might be implemented by biological or neurmorphic systems and show that this form of learning can be made temporally local, and can still function even if many of the dynamical requirements of standard training procedures are relaxed. Thanks to a set of general theorems corroborated by numerical experiments across several CL models, our results provide theoretical foundations for the study and development of CL methods for biological and neuromorphic neural networks.

        ----

        ## [1543] Approximate Stein Classes for Truncated Density Estimation

        **Authors**: *Daniel J. Williams, Song Liu*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/williams23b.html](https://proceedings.mlr.press/v202/williams23b.html)

        **Abstract**:

        Estimating truncated density models is difficult, as these models have intractable normalising constants and hard to satisfy boundary conditions. Score matching can be adapted to solve the truncated density estimation problem, but requires a continuous weighting function which takes zero at the boundary and is positive elsewhere. Evaluation of such a weighting function (and its gradient) often requires a closed-form expression of the truncation boundary and finding a solution to a complicated optimisation problem. In this paper, we propose approximate Stein classes, which in turn leads to a relaxed Stein identity for truncated density estimation. We develop a novel discrepancy measure, truncated kernelised Stein discrepancy (TKSD), which does not require fixing a weighting function in advance, and can be evaluated using only samples on the boundary. We estimate a truncated density model by minimising the Lagrangian dual of TKSD. Finally, experiments show the accuracy of our method to be an improvement over previous works even without the explicit functional form of the boundary.

        ----

        ## [1544] Theoretical Behavior of XAI Methods in the Presence of Suppressor Variables

        **Authors**: *Rick Wilming, Leo Kieslich, Benedict Clark, Stefan Haufe*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/wilming23a.html](https://proceedings.mlr.press/v202/wilming23a.html)

        **Abstract**:

        In recent years, the community of ’explainable artificial intelligence’ (XAI) has created a vast body of methods to bridge a perceived gap between model ’complexity’ and ’interpretability’. However, a concrete problem to be solved by XAI methods has not yet been formally stated. As a result, XAI methods are lacking theoretical and empirical evidence for the ’correctness’ of their explanations, limiting their potential use for quality-control and transparency purposes. At the same time, Haufe et al. (2014) showed, using simple toy examples, that even standard interpretations of linear models can be highly misleading. Specifically, high importance may be attributed to so-called suppressor variables lacking any statistical relation to the prediction target. This behavior has been confirmed empirically for a large array of XAI methods in Wilming et al. (2022). Here, we go one step further by deriving analytical expressions for the behavior of a variety of popular XAI methods on a simple two-dimensional binary classification problem involving Gaussian class-conditional distributions. We show that the majority of the studied approaches will attribute non-zero importance to a non-class-related suppressor feature in the presence of correlated noise. This poses important limitations on the interpretations and conclusions that the outputs of these XAI methods can afford.

        ----

        ## [1545] Marginalization is not Marginal: No Bad VAE Local Minima when Learning Optimal Sparse Representations

        **Authors**: *David Wipf*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/wipf23a.html](https://proceedings.mlr.press/v202/wipf23a.html)

        **Abstract**:

        Although the variational autoencoder (VAE) represents a widely-used deep generative model, the underlying energy function when applied to continuous data remains poorly understood. In fact, most prior theoretical analysis has assumed a simplified affine decoder such that the model collapses to probabilistic PCA, a restricted regime whereby existing classical algorithms can also be trivially applied to guarantee globally optimal solutions. To push our understanding into more complex, practically-relevant settings, this paper instead adopts a deceptively sophisticated single-layer decoder that nonetheless allows the VAE to address the fundamental challenge of learning optimally sparse representations of continuous data originating from popular multiple-response regression models. In doing so, we can then examine VAE properties within the non-trivial context of solving difficult, NP-hard inverse problems. More specifically, we prove rigorous conditions which guarantee that any minimum of the VAE energy (local or global) will produce the optimally sparse latent representation, meaning zero reconstruction error using a minimal number of active latent dimensions. This is ultimately possible because VAE marginalization over the latent posterior selectively smooths away bad local minima as has been conjectured but not actually proven in prior work. We then discuss how equivalent-capacity deterministic autoencoders, even with appropriate sparsity-promoting regularization of the latent space, maintain bad local minima that do not correspond with such parsimonious representations. Overall, these results serve to elucidate key properties of the VAE loss surface relative to finding low-dimensional structure in data.

        ----

        ## [1546] Uncertainty Estimation for Molecules: Desiderata and Methods

        **Authors**: *Tom Wollschläger, Nicholas Gao, Bertrand Charpentier, Mohamed Amine Ketata, Stephan Günnemann*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/wollschlager23a.html](https://proceedings.mlr.press/v202/wollschlager23a.html)

        **Abstract**:

        Graph Neural Networks (GNNs) are promising surrogates for quantum mechanical calculations as they establish unprecedented low errors on collections of molecular dynamics (MD) trajectories. Thanks to their fast inference times they promise to accelerate computational chemistry applications. Unfortunately, despite low in-distribution (ID) errors, such GNNs might be horribly wrong for out-of-distribution (OOD) samples. Uncertainty estimation (UE) may aid in such situations by communicating the model’s certainty about its prediction. Here, we take a closer look at the problem and identify six key desiderata for UE in molecular force fields, three ’physics-informed’ and three ’application-focused’ ones. To overview the field, we survey existing methods from the field of UE and analyze how they fit to the set desiderata. By our analysis, we conclude that none of the previous works satisfies all criteria. To fill this gap, we propose Localized Neural Kernel (LNK) a Gaussian Process (GP)-based extension to existing GNNs satisfying the desiderata. In our extensive experimental evaluation, we test four different UE with three different backbones across two datasets. In out-of-equilibrium detection, we find LNK yielding up to 2.5 and 2.1 times lower errors in terms of AUC-ROC score than dropout or evidential regression-based methods while maintaining high predictive performance.

        ----

        ## [1547] The Blessing of Heterogeneity in Federated Q-Learning: Linear Speedup and Beyond

        **Authors**: *Jiin Woo, Gauri Joshi, Yuejie Chi*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/woo23a.html](https://proceedings.mlr.press/v202/woo23a.html)

        **Abstract**:

        In this paper, we consider federated Q-learning, which aims to learn an optimal Q-function by periodically aggregating local Q-estimates trained on local data alone. Focusing on infinite-horizon tabular Markov decision processes, we provide sample complexity guarantees for both the synchronous and asynchronous variants of federated Q-learning. In both cases, our bounds exhibit a linear speedup with respect to the number of agents and sharper dependencies on other salient problem parameters. Moreover, existing approaches to federated Q-learning adopt an equally-weighted averaging of local Q-estimates, which can be highly sub-optimal in the asynchronous setting since the local trajectories can be highly heterogeneous due to different local behavior policies. Existing sample complexity scales inverse proportionally to the minimum entry of the stationary state-action occupancy distributions over all agents, requiring that every agent covers the entire state-action space. Instead, we propose a novel importance averaging algorithm, giving larger weights to more frequently visited state-action pairs. The improved sample complexity scales inverse proportionally to the minimum entry of the average stationary state-action occupancy distribution of all agents, thus only requiring the agents collectively cover the entire state-action space, unveiling the blessing of heterogeneity.

        ----

        ## [1548] Learning Deep Time-index Models for Time Series Forecasting

        **Authors**: *Gerald Woo, Chenghao Liu, Doyen Sahoo, Akshat Kumar, Steven C. H. Hoi*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/woo23b.html](https://proceedings.mlr.press/v202/woo23b.html)

        **Abstract**:

        Deep learning has been actively applied to time series forecasting, leading to a deluge of new methods, belonging to the class of historical-value models. Yet, despite the attractive properties of time-index models, such as being able to model the continuous nature of underlying time series dynamics, little attention has been given to them. Indeed, while naive deep time-index models are far more expressive than the manually predefined function representations of classical time-index models, they are inadequate for forecasting, being unable to generalize to unseen time steps due to the lack of inductive bias. In this paper, we propose DeepTime, a meta-optimization framework to learn deep time-index models which overcome these limitations, yielding an efficient and accurate forecasting model. Extensive experiments on real world datasets in the long sequence time-series forecasting setting demonstrate that our approach achieves competitive results with state-of-the-art methods, and is highly efficient. Code is available at https://github.com/salesforce/DeepTime.

        ----

        ## [1549] Sharper Bounds for ℓp Sensitivity Sampling

        **Authors**: *David P. Woodruff, Taisuke Yasuda*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/woodruff23a.html](https://proceedings.mlr.press/v202/woodruff23a.html)

        **Abstract**:

        In large scale machine learning, random sampling is a popular way to approximate datasets by a small representative subset of examples. In particular, sensitivity sampling is an intensely studied technique which provides provable guarantees on the quality of approximation, while reducing the number of examples to the product of the VC dimension $d$ and the total sensitivity $\mathfrak{S}$ in remarkably general settings. However, guarantees going beyond this general bound of $\mathfrak{S} d$ are known in perhaps only one setting, for $\ell_2$ subspace embeddings, despite intense study of sensitivity sampling in prior work. In this work, we show the first bounds for sensitivity sampling for $\ell_p$ subspace embeddings for $p\neq 2$ that improve over the general $\mathfrak{S} d$ bound, achieving a bound of roughly $\mathfrak{S}^{2/p}$ for $1\leq p<2$ and $\mathfrak{S}^{2-2/p}$ for $2root leverage score sampling algorithm achieves a bound of roughly $d$ for $1\leq p<2$, and that a combination of leverage score and sensitivity sampling achieves an improved bound of roughly $d^{2/p}\mathfrak{S}^{2-4/p}$ for $2
Cite this Paper



    BibTeX
  



@InProceedings{pmlr-v202-woodruff23a,
  title = 	 {Sharper Bounds for $\ell_p$ Sensitivity Sampling},
  author =       {Woodruff, David and Yasuda, Taisuke},
  booktitle = 	 {Proceedings of the 40th International Conference on Machine Learning},
  pages = 	 {37238--37272},
  year = 	 {2023},
  editor = 	 {Krause, Andreas and Brunskill, Emma and Cho, Kyunghyun and Engelhardt, Barbara and Sabato, Sivan and Scarlett, Jonathan},
  volume = 	 {202},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {23--29 Jul},
  publisher =    {PMLR},
  pdf = 	 {https://proceedings.mlr.press/v202/woodruff23a/woodruff23a.pdf},
  url = 	 {https://proceedings.mlr.press/v202/woodruff23a.html},
  abstract = 	 {In large scale machine learning, random sampling is a popular way to approximate datasets by a small representative subset of examples. In particular, sensitivity sampling is an intensely studied technique which provides provable guarantees on the quality of approximation, while reducing the number of examples to the product of the VC dimension $d$ and the total sensitivity $\mathfrak{S}$ in remarkably general settings. However, guarantees going beyond this general bound of $\mathfrak{S} d$ are known in perhaps only one setting, for $\ell_2$ subspace embeddings, despite intense study of sensitivity sampling in prior work. In this work, we show the first bounds for sensitivity sampling for $\ell_p$ subspace embeddings for $p\neq 2$ that improve over the general $\mathfrak{S} d$ bound, achieving a bound of roughly $\mathfrak{S}^{2/p}$ for $1\leq p<2$ and $\mathfrak{S}^{2-2/p}$ for $2root leverage score sampling algorithm achieves a bound of roughly $d$ for $1\leq p<2$, and that a combination of leverage score and sensitivity sampling achieves an improved bound of roughly $d^{2/p}\mathfrak{S}^{2-4/p}$ for $2

Copy to Clipboard
Download




    Endnote
  


%0 Conference Paper
%T Sharper Bounds for $\ell_p$ Sensitivity Sampling
%A David Woodruff
%A Taisuke Yasuda
%B Proceedings of the 40th International Conference on Machine Learning
%C Proceedings of Machine Learning Research
%D 2023
%E Andreas Krause
%E Emma Brunskill
%E Kyunghyun Cho
%E Barbara Engelhardt
%E Sivan Sabato
%E Jonathan Scarlett	
%F pmlr-v202-woodruff23a
%I PMLR
%P 37238--37272
%U https://proceedings.mlr.press/v202/woodruff23a.html
%V 202
%X In large scale machine learning, random sampling is a popular way to approximate datasets by a small representative subset of examples. In particular, sensitivity sampling is an intensely studied technique which provides provable guarantees on the quality of approximation, while reducing the number of examples to the product of the VC dimension $d$ and the total sensitivity $\mathfrak{S}$ in remarkably general settings. However, guarantees going beyond this general bound of $\mathfrak{S} d$ are known in perhaps only one setting, for $\ell_2$ subspace embeddings, despite intense study of sensitivity sampling in prior work. In this work, we show the first bounds for sensitivity sampling for $\ell_p$ subspace embeddings for $p\neq 2$ that improve over the general $\mathfrak{S} d$ bound, achieving a bound of roughly $\mathfrak{S}^{2/p}$ for $1\leq p<2$ and $\mathfrak{S}^{2-2/p}$ for $2root leverage score sampling algorithm achieves a bound of roughly $d$ for $1\leq p<2$, and that a combination of leverage score and sensitivity sampling achieves an improved bound of roughly $d^{2/p}\mathfrak{S}^{2-4/p}$ for $2

Copy to Clipboard
Download




    APA
  



Woodruff, D. & Yasuda, T.. (2023). Sharper Bounds for $\ell_p$ Sensitivity Sampling. Proceedings of the 40th International Conference on Machine Learning, in Proceedings of Machine Learning Research 202:37238-37272 Available from https://proceedings.mlr.press/v202/woodruff23a.html.



Copy to Clipboard
Download



Related Material


Download PDF
OpenReview

        ----

        ## [1550] Two Losses Are Better Than One: Faster Optimization Using a Cheaper Proxy

        **Authors**: *Blake E. Woodworth, Konstantin Mishchenko, Francis R. Bach*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/woodworth23a.html](https://proceedings.mlr.press/v202/woodworth23a.html)

        **Abstract**:

        We present an algorithm for minimizing an objective with hard-to-compute gradients by using a related, easier-to-access function as a proxy. Our algorithm is based on approximate proximal-point iterations on the proxy combined with relatively few stochastic gradients from the objective. When the difference between the objective and the proxy is $\delta$-smooth, our algorithm guarantees convergence at a rate matching stochastic gradient descent on a $\delta$-smooth objective, which can lead to substantially better sample efficiency. Our algorithm has many potential applications in machine learning, and provides a principled means of leveraging synthetic data, physics simulators, mixed public and private data, and more.

        ----

        ## [1551] SEGA: Structural Entropy Guided Anchor View for Graph Contrastive Learning

        **Authors**: *Junran Wu, Xueyuan Chen, Bowen Shi, Shangzhe Li, Ke Xu*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/wu23a.html](https://proceedings.mlr.press/v202/wu23a.html)

        **Abstract**:

        In contrastive learning, the choice of "view" controls the information that the representation captures and influences the performance of the model. However, leading graph contrastive learning methods generally produce views via random corruption or learning, which could lead to the loss of essential information and alteration of semantic information. An anchor view that maintains the essential information of input graphs for contrastive learning has been hardly investigated. In this paper, based on the theory of graph information bottleneck, we deduce the definition of this anchor view; put differently, the anchor view with essential information of input graph is supposed to have the minimal structural uncertainty. Furthermore, guided by structural entropy, we implement the anchor view, termed SEGA, for graph contrastive learning. We extensively validate the proposed anchor view on various benchmarks regarding graph classification under unsupervised, semi-supervised, and transfer learning and achieve significant performance boosts compared to the state-of-the-art methods.

        ----

        ## [1552] Causal Proxy Models for Concept-based Model Explanations

        **Authors**: *Zhengxuan Wu, Karel D'Oosterlinck, Atticus Geiger, Amir Zur, Christopher Potts*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/wu23b.html](https://proceedings.mlr.press/v202/wu23b.html)

        **Abstract**:

        Explainability methods for NLP systems encounter a version of the fundamental problem of causal inference: for a given ground-truth input text, we never truly observe the counterfactual texts necessary for isolating the causal effects of model representations on outputs. In response, many explainability methods make no use of counterfactual texts, assuming they will be unavailable. In this paper, we show that robust causal explainability methods can be created using approximate counterfactuals, which can be written by humans to approximate a specific counterfactual or simply sampled using metadata-guided heuristics. The core of our proposal is the Causal Proxy Model (CPM). A CPM explains a black-box model $\mathcal{N}$ because it is trained to have the same actual input/output behavior as $\mathcal{N}$ while creating neural representations that can be intervened upon to simulate the counterfactual input/output behavior of $\mathcal{N}$. Furthermore, we show that the best CPM for $\mathcal{N}$ performs comparably to $\mathcal{N}$ in making factual predictions, which means that the CPM can simply replace $\mathcal{N}$, leading to more explainable deployed models.

        ----

        ## [1553] Effective Neural Topic Modeling with Embedding Clustering Regularization

        **Authors**: *Xiaobao Wu, Xinshuai Dong, Thong Thanh Nguyen, Anh Tuan Luu*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/wu23c.html](https://proceedings.mlr.press/v202/wu23c.html)

        **Abstract**:

        Topic models have been prevalent for decades with various applications. However, existing topic models commonly suffer from the notorious topic collapsing: discovered topics semantically collapse towards each other, leading to highly repetitive topics, insufficient topic discovery, and damaged model interpretability. In this paper, we propose a new neural topic model, Embedding Clustering Regularization Topic Model (ECRTM). Besides the existing reconstruction error, we propose a novel Embedding Clustering Regularization (ECR), which forces each topic embedding to be the center of a separately aggregated word embedding cluster in the semantic space. This enables each produced topic to contain distinct word semantics, which alleviates topic collapsing. Regularized by ECR, our ECRTM generates diverse and coherent topics together with high-quality topic distributions of documents. Extensive experiments on benchmark datasets demonstrate that ECRTM effectively addresses the topic collapsing issue and consistently surpasses state-of-the-art baselines in terms of topic quality, topic distributions of documents, and downstream classification tasks.

        ----

        ## [1554] Adaptive Compositional Continual Meta-Learning

        **Authors**: *Bin Wu, Jinyuan Fang, Xiangxiang Zeng, Shangsong Liang, Qiang Zhang*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/wu23d.html](https://proceedings.mlr.press/v202/wu23d.html)

        **Abstract**:

        This paper focuses on continual meta-learning, where few-shot tasks are heterogeneous and sequentially available. Recent works use a mixture model for meta-knowledge to deal with the heterogeneity. However, these methods suffer from parameter inefficiency caused by two reasons: (1) the underlying assumption of mutual exclusiveness among mixture components hinders sharing meta-knowledge across heterogeneous tasks. (2) they only allow increasing mixture components and cannot adaptively filter out redundant components. In this paper, we propose an Adaptive Compositional Continual Meta-Learning (ACML) algorithm, which employs a compositional premise to associate a task with a subset of mixture components, allowing meta-knowledge sharing among heterogeneous tasks. Moreover, to adaptively adjust the number of mixture components, we propose a component sparsification method based on evidential theory to filter out redundant components. Experimental results show ACML outperforms strong baselines, showing the effectiveness of our compositional meta-knowledge, and confirming that ACML can adaptively learn meta-knowledge.

        ----

        ## [1555] Anchor Sampling for Federated Learning with Partial Client Participation

        **Authors**: *Feijie Wu, Song Guo, Zhihao Qu, Shiqi He, Ziming Liu, Jing Gao*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/wu23e.html](https://proceedings.mlr.press/v202/wu23e.html)

        **Abstract**:

        Compared with full client participation, partial client participation is a more practical scenario in federated learning, but it may amplify some challenges in federated learning, such as data heterogeneity. The lack of inactive clients’ updates in partial client participation makes it more likely for the model aggregation to deviate from the aggregation based on full client participation. Training with large batches on individual clients is proposed to address data heterogeneity in general, but their effectiveness under partial client participation is not clear. Motivated by these challenges, we propose to develop a novel federated learning framework, referred to as FedAMD, for partial client participation. The core idea is anchor sampling, which separates partial participants into anchor and miner groups. Each client in the anchor group aims at the local bullseye with the gradient computation using a large batch. Guided by the bullseyes, clients in the miner group steer multiple near-optimal local updates using small batches and update the global model. By integrating the results of the two groups, FedAMD is able to accelerate the training process and improve the model performance. Measured by $\epsilon$-approximation and compared to the state-of-the-art methods, FedAMD achieves the convergence by up to $O(1/\epsilon)$ fewer communication rounds under non-convex objectives. Empirical studies on real-world datasets validate the effectiveness of FedAMD and demonstrate the superiority of the proposed algorithm: Not only does it considerably save computation and communication costs, but also the test accuracy significantly improves.

        ----

        ## [1556] Solving High-Dimensional PDEs with Latent Spectral Models

        **Authors**: *Haixu Wu, Tengge Hu, Huakun Luo, Jianmin Wang, Mingsheng Long*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/wu23f.html](https://proceedings.mlr.press/v202/wu23f.html)

        **Abstract**:

        Deep models have achieved impressive progress in solving partial differential equations (PDEs). A burgeoning paradigm is learning neural operators to approximate the input-output mappings of PDEs. While previous deep models have explored the multiscale architectures and various operator designs, they are limited to learning the operators as a whole in the coordinate space. In real physical science problems, PDEs are complex coupled equations with numerical solvers relying on discretization into high-dimensional coordinate space, which cannot be precisely approximated by a single operator nor efficiently learned due to the curse of dimensionality. We present Latent Spectral Models (LSM) toward an efficient and precise solver for high-dimensional PDEs. Going beyond the coordinate space, LSM enables an attention-based hierarchical projection network to reduce the high-dimensional data into a compact latent space in linear time. Inspired by classical spectral methods in numerical analysis, we design a neural spectral block to solve PDEs in the latent space that approximates complex input-output mappings via learning multiple basis operators, enjoying nice theoretical guarantees for convergence and approximation. Experimentally, LSM achieves consistent state-of-the-art and yields a relative gain of 11.5% averaged on seven benchmarks covering both solid and fluid physics. Code is available at https://github.com/thuml/Latent-Spectral-Models.

        ----

        ## [1557] A Law of Robustness beyond Isoperimetry

        **Authors**: *Yihan Wu, Heng Huang, Hongyang Zhang*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/wu23g.html](https://proceedings.mlr.press/v202/wu23g.html)

        **Abstract**:

        We study the robust interpolation problem of arbitrary data distributions supported on a bounded space and propose a two-fold law of robustness. Robust interpolation refers to the problem of interpolating $n$ noisy training data points in $R^d$ by a Lipschitz function. Although this problem has been well understood when the samples are drawn from an isoperimetry distribution, much remains unknown concerning its performance under generic or even the worst-case distributions. We prove a Lipschitzness lower bound $\Omega(\sqrt{n/p})$ of the interpolating neural network with $p$ parameters on arbitrary data distributions. With this result, we validate the law of robustness conjecture in prior work by Bubeck, Li and Nagaraj on two-layer neural networks with polynomial weights. We then extend our result to arbitrary interpolating approximators and prove a Lipschitzness lower bound $\Omega(n^{1/d})$ for robust interpolation. Our results demonstrate a two-fold law of robustness: a) we show the potential benefit of overparametrization for smooth data interpolation when $n=poly(d)$, and b) we disprove the potential existence of an $O(1)$-Lipschitz robust interpolating function when $n=\exp(\omega(d))$.

        ----

        ## [1558] Uncovering Adversarial Risks of Test-Time Adaptation

        **Authors**: *Tong Wu, Feiran Jia, Xiangyu Qi, Jiachen T. Wang, Vikash Sehwag, Saeed Mahloujifar, Prateek Mittal*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/wu23h.html](https://proceedings.mlr.press/v202/wu23h.html)

        **Abstract**:

        Recently, test-time adaptation (TTA) has been proposed as a promising solution for addressing distribution shifts. It allows a base model to adapt to an unforeseen distribution during inference by leveraging the information from the batch of (unlabeled) test data. However, we uncover a novel security vulnerability of TTA based on the insight that predictions on benign samples can be impacted by malicious samples in the same batch. To exploit this vulnerability, we propose Distribution Invading Attack (DIA), which injects a small fraction of malicious data into the test batch. DIA causes models using TTA to misclassify benign and unperturbed test data, providing an entirely new capability for adversaries that is infeasible in canonical machine learning pipelines. Through comprehensive evaluations, we demonstrate the high effectiveness of our attack on multiple benchmarks across six TTA methods. In response, we investigate two countermeasures to robustify the existing insecure TTA implementations, following the principle of security by design. Together, we hope our findings can make the community aware of the utility-security tradeoffs in deploying TTA and provide valuable insights for developing robust TTA approaches.

        ----

        ## [1559] Stable Estimation of Heterogeneous Treatment Effects

        **Authors**: *Anpeng Wu, Kun Kuang, Ruoxuan Xiong, Bo Li, Fei Wu*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/wu23i.html](https://proceedings.mlr.press/v202/wu23i.html)

        **Abstract**:

        Estimating heterogeneous treatment effects (HTE) is crucial for identifying the variation of treatment effects across individuals or subgroups. Most existing methods estimate HTE by removing the confounding bias from imbalanced treatment assignments. However, these methods may produce unreliable estimates of treatment effects and potentially allocate suboptimal treatment arms for underrepresented populations. To improve the estimation accuracy of HTE for underrepresented populations, we propose a novel Stable CounterFactual Regression (StableCFR) to smooth the population distribution and upsample the underrepresented subpopulations, while balancing confounders between treatment and control groups. Specifically, StableCFR upsamples the underrepresented data using uniform sampling, where each disjoint subpopulation is weighted proportional to the Lebesgue measure of its support. Moreover, StableCFR balances covariates by using an epsilon-greedy matching approach. Empirical results on both synthetic and real-world datasets demonstrate the superior performance of our StableCFR on estimating HTE for underrepresented populations.

        ----

        ## [1560] Rethinking Explaining Graph Neural Networks via Non-parametric Subgraph Matching

        **Authors**: *Fang Wu, Siyuan Li, Xurui Jin, Yinghui Jiang, Dragomir Radev, Zhangming Niu, Stan Z. Li*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/wu23j.html](https://proceedings.mlr.press/v202/wu23j.html)

        **Abstract**:

        The success of graph neural networks (GNNs) provokes the question about explainability: “Which fraction of the input graph is the most determinant of the prediction?” Particularly, parametric explainers prevail in existing approaches because of their more robust capability to decipher the black-box (i.e., target GNNs). In this paper, based on the observation that graphs typically share some common motif patterns, we propose a novel non-parametric subgraph matching framework, dubbed MatchExplainer, to explore explanatory subgraphs. It couples the target graph with other counterpart instances and identifies the most crucial joint substructure by minimizing the node corresponding-based distance. Moreover, we note that present graph sampling or node-dropping methods usually suffer from the false positive sampling problem. To alleviate this issue, we design a new augmentation paradigm named MatchDrop. It takes advantage of MatchExplainer to fix the most informative portion of the graph and merely operates graph augmentations on the rest less informative part. Extensive experiments on synthetic and real-world datasets show the effectiveness of our MatchExplainer by outperforming all state-of-the-art parametric baselines with significant margins. Results also demonstrate that MatchDrop is a general scheme to be equipped with GNNs for enhanced performance. The code is available at https://github.com/smiles724/MatchExplainer.

        ----

        ## [1561] Understanding Int4 Quantization for Language Models: Latency Speedup, Composability, and Failure Cases

        **Authors**: *Xiaoxia Wu, Cheng Li, Reza Yazdani Aminabadi, Zhewei Yao, Yuxiong He*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/wu23k.html](https://proceedings.mlr.press/v202/wu23k.html)

        **Abstract**:

        Improving the deployment efficiency of transformer-based language models has been challenging given their high computation and memory cost. While INT8 quantization has recently been shown to be effective in reducing both the memory cost and latency while preserving model accuracy, it remains unclear whether we can leverage INT4 (which doubles peak hardware throughput) to achieve further latency improvement. In this study, we explore the feasibility of employing INT4 weight and activation (W4A4) quantization for language models. Our findings indicate that W4A4 quantization introduces no to negligible accuracy degradation for encoder-only and encoder-decoder models, but causes a significant accuracy drop for decoder-only models. To materialize the performance gain using W4A4, we develop a highly-optimized end-to-end W4A4 encoder inference pipeline supporting different quantization strategies. Our INT4 pipeline is $8.5\times$ faster for latency-oriented scenarios and up to $3\times$ for throughput-oriented scenarios compared to the inference of FP16, and improves the SOTA BERT INT8 performance from FasterTransformer by up to $1.7\times$. We provide insights into the failure cases when applying W4A4 to decoder-only models, and further explore the compatibility of INT4 quantization with other compression methods, like pruning and layer reduction.

        ----

        ## [1562] Towards Understanding Generalization of Macro-AUC in Multi-label Learning

        **Authors**: *Guoqiang Wu, Chongxuan Li, Yilong Yin*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/wu23l.html](https://proceedings.mlr.press/v202/wu23l.html)

        **Abstract**:

        Macro-AUC is the arithmetic mean of the class-wise AUCs in multi-label learning and is commonly used in practice. However, its theoretical understanding is far lacking. Toward solving it, we characterize the generalization properties of various learning algorithms based on the corresponding surrogate losses w.r.t. Macro-AUC. We theoretically identify a critical factor of the dataset affecting the generalization bounds: the label-wise class imbalance. Our results on the imbalance-aware error bounds show that the widely-used univariate loss-based algorithm is more sensitive to the label-wise class imbalance than the proposed pairwise and reweighted loss-based ones, which probably implies its worse performance. Moreover, empirical results on various datasets corroborate our theory findings. To establish it, technically, we propose a new (and more general) McDiarmid-type concentration inequality, which may be of independent interest.

        ----

        ## [1563] Quantifying the Knowledge in GNNs for Reliable Distillation into MLPs

        **Authors**: *Lirong Wu, Haitao Lin, Yufei Huang, Stan Z. Li*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/wu23m.html](https://proceedings.mlr.press/v202/wu23m.html)

        **Abstract**:

        To bridge the gaps between topology-aware Graph Neural Networks (GNNs) and inference-efficient Multi-Layer Perceptron (MLPs), GLNN proposes to distill knowledge from a well-trained teacher GNN into a student MLP. Despite their great progress, comparatively little work has been done to explore the reliability of different knowledge points (nodes) in GNNs, especially their roles played during distillation. In this paper, we first quantify the knowledge reliability in GNN by measuring the invariance of their information entropy to noise perturbations, from which we observe that different knowledge points (1) show different distillation speeds (temporally); (2) are differentially distributed in the graph (spatially). To achieve reliable distillation, we propose an effective approach, namely Knowledge-inspired Reliable Distillation (KRD), that models the probability of each node being an informative and reliable knowledge point, based on which we sample a set of additional reliable knowledge points as supervision for training student MLPs. Extensive experiments show that KRD improves over the vanilla MLPs by 12.62% and outperforms its corresponding teacher GNNs by 2.16% averaged over 7 datasets and 3 GNN architectures. Codes are publicly available at: https://github.com/LirongWu/RKD.

        ----

        ## [1564] Delay-agnostic Asynchronous Coordinate Update Algorithm

        **Authors**: *Xuyang Wu, Changxin Liu, Sindri Magnússon, Mikael Johansson*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/wu23n.html](https://proceedings.mlr.press/v202/wu23n.html)

        **Abstract**:

        We propose a delay-agnostic asynchronous coordinate update algorithm (DEGAS) for computing operator fixed points, with applications to asynchronous optimization. DEGAS includes novel asynchronous variants of ADMM and block-coordinate descent as special cases. We prove that DEGAS converges with both bounded and unbounded delays under delay-free parameter conditions. We also validate by theory and experiments that DEGAS adapts well to the actual delays. The effectiveness of DEGAS is demonstrated by numerical experiments on classification problems.

        ----

        ## [1565] Masked Trajectory Models for Prediction, Representation, and Control

        **Authors**: *Philipp Wu, Arjun Majumdar, Kevin Stone, Yixin Lin, Igor Mordatch, Pieter Abbeel, Aravind Rajeswaran*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/wu23o.html](https://proceedings.mlr.press/v202/wu23o.html)

        **Abstract**:

        We introduce Masked Trajectory Models (MTM) as a generic abstraction for sequential decision making. MTM takes a trajectory, such as a state-action sequence, and aims to reconstruct the trajectory conditioned on random subsets of the same trajectory. By training with a highly randomized masking pattern, MTM learns versatile networks that can take on different roles or capabilities, by simply choosing appropriate masks at inference time. For example, the same MTM network can be used as a forward dynamics model, inverse dynamics model, or even an offline RL agent. Through extensive experiments in several continuous control tasks, we show that the same MTM network – i.e. same weights – can match or outperform specialized networks trained for the aforementioned capabilities. Additionally, we find that state representations learned by MTM can significantly accelerate the learning speed of traditional RL algorithms. Finally, in offline RL benchmarks, we find that MTM is competitive with specialized offline RL algorithms, despite MTM being a generic self-supervised learning method without any explicit RL components. Code is available at https://github.com/facebookresearch/mtm.

        ----

        ## [1566] Disentangled Multi-Fidelity Deep Bayesian Active Learning

        **Authors**: *Dongxia Wu, Ruijia Niu, Matteo Chinazzi, Yian Ma, Rose Yu*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/wu23p.html](https://proceedings.mlr.press/v202/wu23p.html)

        **Abstract**:

        To balance quality and cost, various domain areas of science and engineering run simulations at multiple levels of sophistication. Multi-fidelity active learning aims to learn a direct mapping from input parameters to simulation outputs at the highest fidelity by actively acquiring data from multiple fidelity levels. However, existing approaches based on Gaussian processes are hardly scalable to high-dimensional data. Deep learning-based methods often impose a hierarchical structure in hidden representations, which only supports passing information from low-fidelity to high-fidelity. These approaches can lead to the undesirable propagation of errors from low-fidelity representations to high-fidelity ones. We propose a novel framework called Disentangled Multi-fidelity Deep Bayesian Active Learning (D-MFDAL), which learns the surrogate models conditioned on the distribution of functions at multiple fidelities. On benchmark tasks of learning deep surrogates of partial differential equations including heat equation, Poisson’s equation and fluid simulations, our approach significantly outperforms state-of-the-art in prediction accuracy and sample efficiency.

        ----

        ## [1567] Tight Data Access Bounds for Private Top-k Selection

        **Authors**: *Hao Wu, Olga Ohrimenko, Anthony Wirth*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/wu23q.html](https://proceedings.mlr.press/v202/wu23q.html)

        **Abstract**:

        We study the top-$k$ selection problem under the differential privacy model: $m$ items are rated according to votes of a set of clients. We consider a setting in which algorithms can retrieve data via a sequence of accesses, each either a random access or a sorted access; the goal is to minimize the total number of data accesses. Our algorithm requires only $O(\sqrt{mk})$ expected accesses: to our knowledge, this is the first sublinear data-access upper bound for this problem. Our analysis also shows that the well-known exponential mechanism requires only $O(\sqrt{m})$ expected accesses. Accompanying this, we develop the first lower bounds for the problem, in three settings: only random accesses; only sorted accesses; a sequence of accesses of either kind. We show that, to avoid $\Omega(m)$ access cost, supporting both kinds of access is necessary, and that in this case our algorithm’s access cost is optimal.

        ----

        ## [1568] The Implicit Regularization of Dynamical Stability in Stochastic Gradient Descent

        **Authors**: *Lei Wu, Weijie J. Su*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/wu23r.html](https://proceedings.mlr.press/v202/wu23r.html)

        **Abstract**:

        In this paper, we study the implicit regularization of stochastic gradient descent (SGD) through the lens of dynamical stability (Wu et al., 2018). We start by revising existing stability analyses of SGD, showing how the Frobenius norm and trace of Hessian relate to different notions of stability. Notably, if a global minimum is linearly stable for SGD, then the trace of Hessian must be less than or equal to $2/\eta$, where $\eta$ denotes the learning rate. By contrast, for gradient descent (GD), the stability imposes a similar constraint but only on the largest eigenvalue of Hessian. We then turn to analyze the generalization properties of these stable minima, focusing specifically on two-layer ReLU networks and diagonal linear networks. Notably, we establish the equivalence between these metrics of sharpness and certain parameter norms for the two models, which allows us to show that the stable minima of SGD provably generalize well. By contrast, the stability-induced regularization of GD is provably too weak to ensure satisfactory generalization. This discrepancy provides an explanation of why SGD often generalizes better than GD. Note that the learning rate (LR) plays a pivotal role in the strength of stability-induced regularization. As the LR increases, the regularization effect becomes more pronounced, elucidating why SGD with a larger LR consistently demonstrates superior generalization capabilities. Additionally, numerical experiments are provided to support our theoretical findings.

        ----

        ## [1569] Distributional Offline Policy Evaluation with Predictive Error Guarantees

        **Authors**: *Runzhe Wu, Masatoshi Uehara, Wen Sun*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/wu23s.html](https://proceedings.mlr.press/v202/wu23s.html)

        **Abstract**:

        We study the problem of estimating the distribution of the return of a policy using an offline dataset that is not generated from the policy, i.e., distributional offline policy evaluation (OPE). We propose an algorithm called Fitted Likelihood Estimation (FLE), which conducts a sequence of Maximum Likelihood Estimation (MLE) and has the flexibility of integrating any state-of-the-art probabilistic generative models as long as it can be trained via MLE. FLE can be used for both finite-horizon and infinite-horizon discounted settings where rewards can be multi-dimensional vectors. Our theoretical results show that for both finite-horizon and infinite-horizon discounted settings, FLE can learn distributions that are close to the ground truth under total variation distance and Wasserstein distance, respectively. Our theoretical results hold under the conditions that the offline data covers the test policy’s traces and that the supervised learning MLE procedures succeed. Experimentally, we demonstrate the performance of FLE with two generative models, Gaussian mixture models and diffusion models. For the multi-dimensional reward setting, FLE with diffusion models is capable of estimating the complicated distribution of the return of a test policy.

        ----

        ## [1570] π-Tuning: Transferring Multimodal Foundation Models with Optimal Multi-task Interpolation

        **Authors**: *Chengyue Wu, Teng Wang, Yixiao Ge, Zeyu Lu, Ruisong Zhou, Ying Shan, Ping Luo*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/wu23t.html](https://proceedings.mlr.press/v202/wu23t.html)

        **Abstract**:

        Foundation models have achieved great advances in multi-task learning with a unified interface of unimodal and multimodal tasks. However, the potential of such multi-task learners has not been exploited during transfer learning. In this work, we present a universal parameter-efficient transfer learning method, termed Predict-Interpolate Tuning ($\pi$-Tuning), for vision, language, and vision-language tasks. It aggregates the parameters of lightweight task-specific experts learned from similar tasks to aid the target downstream task. The task similarities are predicted in a unified modality-independent space, yielding a scalable graph to demonstrate task relationships. $\pi$-Tuning has several appealing benefits. First, it flexibly explores both intra- and inter-modal transferability between similar tasks to improve the accuracy and robustness of transfer learning, especially in data-scarce scenarios. Second, it offers a systematical solution for transfer learning with multi-task prediction-and-then-interpolation, compatible with diverse types of parameter-efficient experts, such as prompt and adapter. Third, an extensive study of task-level mutual benefits on 14 unimodal and 6 multimodal datasets shows that $\pi$-Tuning surpasses fine-tuning and other parameter-efficient transfer learning methods both in full-shot and low-shot regimes. The task graph also enables an in-depth interpretable analysis of task transferability across modalities. The code will be available at https://github.com/TencentARC/pi-Tuning.

        ----

        ## [1571] Learning Functional Distributions with Private Labels

        **Authors**: *Changlong Wu, Yifan Wang, Ananth Grama, Wojciech Szpankowski*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/wu23u.html](https://proceedings.mlr.press/v202/wu23u.html)

        **Abstract**:

        We study the problem of learning functional distributions in the presence of noise. A functional is a map from the space of features to distributions over a set of labels, and is often assumed to belong to a known class of hypotheses $\mathcal{F}$. Features are generated by a general random process and labels are sampled independently from feature-dependent distributions. In privacy sensitive applications, labels are passed through a noisy kernel. We consider online learning, where at each time step, a predictor attempts to predict the actual (label) distribution given only the features and noisy labels in prior steps. The performance of the predictor is measured by the expected KL-risk that compares the predicted distributions to the underlying truth. We show that the minimax expected KL-risk is of order $\tilde{\Theta}(\sqrt{T\log|\mathcal{F}|})$ for finite hypothesis class $\mathcal{F}$ and any non-trivial noise level. We then extend this result to general infinite classes via the concept of stochastic sequential covering and provide matching lower and upper bounds for a wide range of natural classes.

        ----

        ## [1572] QuantumDARTS: Differentiable Quantum Architecture Search for Variational Quantum Algorithms

        **Authors**: *Wenjie Wu, Ge Yan, Xudong Lu, Kaisen Pan, Junchi Yan*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/wu23v.html](https://proceedings.mlr.press/v202/wu23v.html)

        **Abstract**:

        With the arrival of the Noisy Intermediate-Scale Quantum (NISQ) era and the fast development of machine learning, variational quantum algorithms (VQA) including Variational Quantum Eigensolver (VQE) and quantum neural network (QNN) have received increasing attention with wide potential applications in foreseeable near future. We study the problem of quantum architecture search (QAS) for VQA to automatically design parameterized quantum circuits (PQC). We devise a differentiable searching algorithm based on Gumbel-Softmax in contrast to peer methods that often require numerous circuit sampling and evaluation. Two versions of our algorithm are provided, namely macro search and micro search, where macro search directly searches for the whole circuit like other literature while the innovative micro search is able to infer the sub-circuit structure from a small-scale and then transfer that to a large-scale problem. We conduct intensive experiments on unweighted Max-Cut, ground state energy estimation, and image classification. The superior performance shows the efficiency and capability of macro search, which requires little prior knowledge. Moreover, the experiments on micro search show the potential of our algorithm for large-scale QAS problems.

        ----

        ## [1573] Discover and Cure: Concept-aware Mitigation of Spurious Correlation

        **Authors**: *Shirley Wu, Mert Yüksekgönül, Linjun Zhang, James Zou*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/wu23w.html](https://proceedings.mlr.press/v202/wu23w.html)

        **Abstract**:

        Deep neural networks often rely on spurious correlations to make predictions, which hinders generalization beyond training environments. For instance, models that associate cats with bed backgrounds can fail to predict the existence of cats in other environments without beds. Mitigating spurious correlations is crucial in building trustworthy models. However, the existing works lack transparency to offer insights into the mitigation process. In this work, we propose an interpretable framework, Discover and Cure (DISC), to tackle the issue. With human-interpretable concepts, DISC iteratively 1) discovers unstable concepts across different environments as spurious attributes, then 2) intervenes on the training data using the discovered concepts to reduce spurious correlation. Across systematic experiments, DISC provides superior generalization ability and interpretability than the existing approaches. Specifically, it outperforms the state-of-the-art methods on an object recognition task and a skin-lesion classification task by 7.5% and 9.6%, respectively. Additionally, we offer theoretical analysis and guarantees to understand the benefits of models trained by DISC. Code and data are available at https://github.com/Wuyxin/DISC.

        ----

        ## [1574] On the Training Instability of Shuffling SGD with Batch Normalization

        **Authors**: *David Xing Wu, Chulhee Yun, Suvrit Sra*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/wu23x.html](https://proceedings.mlr.press/v202/wu23x.html)

        **Abstract**:

        We uncover how SGD interacts with batch normalization and can exhibit undesirable training dynamics such as divergence. More precisely, we study how Single Shuffle (SS) and Random Reshuffle (RR)—two widely used variants of SGD—interact surprisingly differently in the presence of batch normalization: RR leads to much more stable evolution of training loss than SS. As a concrete example, for regression using a linear network with batch normalized inputs, we prove that SS and RR converge to distinct global optima that are “distorted” away from gradient descent. Thereafter, for classification we characterize conditions under which training divergence for SS and RR can, and cannot occur. We present explicit constructions to show how SS leads to distorted optima in regression and divergence for classification, whereas RR avoids both distortion and divergence. We validate our results empirically in realistic settings, and conclude that the separation between SS and RR used with batch normalization is relevant in practice.

        ----

        ## [1575] dugMatting: Decomposed-Uncertainty-Guided Matting

        **Authors**: *Jiawei Wu, Changqing Zhang, Zuoyong Li, Huazhu Fu, Xi Peng, Joey Tianyi Zhou*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/wu23y.html](https://proceedings.mlr.press/v202/wu23y.html)

        **Abstract**:

        Cutting out an object and estimating its opacity mask, known as image matting, is a key task in image and video editing. Due to the highly ill-posed issue, additional inputs, typically user-defined trimaps or scribbles, are usually needed to reduce the uncertainty. Although effective, it is either time consuming or only suitable for experienced users who know where to place the strokes. In this work, we propose a decomposed-uncertainty-guided matting (dugMatting) algorithm, which explores the explicitly decomposed uncertainties to efficiently and effectively improve the results. Basing on the characteristic of these uncertainties, the epistemic uncertainty is reduced in the process of guiding interaction (which introduces prior knowledge), while the aleatoric uncertainty is reduced in modeling data distribution (which introduces statistics for both data and possible noise). The proposed matting framework relieves the requirement for users to determine the interaction areas by using simple and efficient labeling. Extensively quantitative and qualitative results validate that the proposed method significantly improves the original matting algorithms in terms of both efficiency and efficacy.

        ----

        ## [1576] Personalized Federated Learning under Mixture of Distributions

        **Authors**: *Yue Wu, Shuaicheng Zhang, Wenchao Yu, Yanchi Liu, Quanquan Gu, Dawei Zhou, Haifeng Chen, Wei Cheng*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/wu23z.html](https://proceedings.mlr.press/v202/wu23z.html)

        **Abstract**:

        The recent trend towards Personalized Federated Learning (PFL) has garnered significant attention as it allows for the training of models that are tailored to each client while maintaining data privacy. However, current PFL techniques primarily focus on modeling the conditional distribution heterogeneity (i.e. concept shift), which can result in suboptimal performance when the distribution of input data across clients diverges (i.e. covariate shift). Additionally, these techniques often lack the ability to adapt to unseen data, further limiting their effectiveness in real-world scenarios. To address these limitations, we propose a novel approach, FedGMM, which utilizes Gaussian mixture models (GMM) to effectively fit the input data distributions across diverse clients. The model parameters are estimated by maximum likelihood estimation utilizing a federated Expectation-Maximization algorithm, which is solved in closed form and does not assume gradient similarity. Furthermore, FedGMM possesses an additional advantage of adapting to new clients with minimal overhead, and it also enables uncertainty quantification. Empirical evaluations on synthetic and benchmark datasets demonstrate the superior performance of our method in both PFL classification and novel sample detection.

        ----

        ## [1577] Differentially Private Episodic Reinforcement Learning with Heavy-tailed Rewards

        **Authors**: *Yulian Wu, Xingyu Zhou, Sayak Ray Chowdhury, Di Wang*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/wu23aa.html](https://proceedings.mlr.press/v202/wu23aa.html)

        **Abstract**:

        In this paper we study the problem of (finite horizon tabular) Markov decision processes (MDPs) with heavy-tailed rewards under the constraint of differential privacy (DP). Compared with the previous studies for private reinforcement learning that typically assume rewards are sampled from some bounded or sub-Gaussian distributions to ensure DP, we consider the setting where reward distributions have only finite $(1+v)$-th moments with some $v \in (0,1]$. By resorting to robust mean estimators for rewards, we first propose two frameworks for heavy-tailed MDPs, i.e., one is for value iteration and another is for policy optimization. Under each framework, we consider both joint differential privacy (JDP) and local differential privacy (LDP) models. Based on our frameworks, we provide regret upper bounds for both JDP and LDP cases, and show that the moment of distributions and privacy budget have significant impact on regrets. Finally, we establish a lower bound of regret minimization for heavy-tailed MDPs in JDP model by reducing it to the instance-independent lower bound of heavy-tailed multi-armed bandits in DP model. We also show the lower bound for the problem in LDP by adopting some private minimax methods. Our results reveal that there are fundamental differences between the problem of private RL with sub-Gaussian and that with heavy-tailed rewards.

        ----

        ## [1578] Finite-Sample Analysis of Learning High-Dimensional Single ReLU Neuron

        **Authors**: *Jingfeng Wu, Difan Zou, Zixiang Chen, Vladimir Braverman, Quanquan Gu, Sham M. Kakade*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/wu23ab.html](https://proceedings.mlr.press/v202/wu23ab.html)

        **Abstract**:

        This paper considers the problem of learning single ReLU neuron with squared loss (a.k.a., ReLU regression) in the overparameterized regime, where the input dimension can exceed the number of samples. We analyze a Perceptron-type algorithm called GLM-tron [Kakade et al. 2011], and provide its dimension-free risk upper bounds for high-dimensional ReLU regression in both well-specified and misspecified settings. Our risk bounds recover several existing results as special cases. Moreover, in the well-specified setting, we also provide an instance-wise matching risk lower bound for GLM-tron. Our upper and lower risk bounds provide a sharp characterization of the high-dimensional ReLU regression problems that can be learned via GLM-tron. On the other hand, we provide some negative results for stochastic gradient descent (SGD) for ReLU regression with symmetric Bernoulli data: if the model is well-specified, the excess risk of SGD is provably no better than that of GLM-tron ignoring constant factors, for each problem instance; and in the noiseless case, GLM-tron can achieve a small risk while SGD unavoidably suffers from a constant risk in expectation. These results together suggest that GLM-tron might be more preferable than SGD for high-dimensional ReLU regression.

        ----

        ## [1579] Understanding Backdoor Attacks through the Adaptability Hypothesis

        **Authors**: *Xun Xian, Ganghua Wang, Jayanth Srinivasa, Ashish Kundu, Xuan Bi, Mingyi Hong, Jie Ding*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/xian23a.html](https://proceedings.mlr.press/v202/xian23a.html)

        **Abstract**:

        A poisoning backdoor attack is a rising security concern for deep learning. This type of attack can result in the backdoored model functioning normally most of the time but exhibiting abnormal behavior when presented with inputs containing the backdoor trigger, making it difficult to detect and prevent. In this work, we propose the adaptability hypothesis to understand when and why a backdoor attack works for general learning models, including deep neural networks, based on the theoretical investigation of classical kernel-based learning models. The adaptability hypothesis postulates that for an effective attack, the effect of incorporating a new dataset on the predictions of the original data points will be small, provided that the original data points are distant from the new dataset. Experiments on benchmark image datasets and state-of-the-art backdoor attacks for deep neural networks are conducted to corroborate the hypothesis. Our finding provides insight into the factors that affect the attack’s effectiveness and has implications for the design of future attacks and defenses.

        ----

        ## [1580] Fair and Optimal Classification via Post-Processing

        **Authors**: *Ruicheng Xian, Lang Yin, Han Zhao*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/xian23b.html](https://proceedings.mlr.press/v202/xian23b.html)

        **Abstract**:

        To mitigate the bias exhibited by machine learning models, fairness criteria can be integrated into the training process to ensure fair treatment across all demographics, but it often comes at the expense of model performance. Understanding such tradeoffs, therefore, underlies the design of fair algorithms. To this end, this paper provides a complete characterization of the inherent tradeoff of demographic parity on classification problems, under the most general multi-group, multi-class, and noisy setting. Specifically, we show that the minimum error rate achievable by randomized and attribute-aware fair classifiers is given by the optimal value of a Wasserstein-barycenter problem. On the practical side, our findings lead to a simple post-processing algorithm that derives fair classifiers from score functions, which yields the optimal fair classifier when the score is Bayes optimal. We provide suboptimality analysis and sample complexity for our algorithm, and demonstrate its effectiveness on benchmark datasets.

        ----

        ## [1581] UMD: Unsupervised Model Detection for X2X Backdoor Attacks

        **Authors**: *Zhen Xiang, Zidi Xiong, Bo Li*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/xiang23a.html](https://proceedings.mlr.press/v202/xiang23a.html)

        **Abstract**:

        Backdoor (Trojan) attack is a common threat to deep neural networks, where samples from one or more source classes embedded with a backdoor trigger will be misclassified to adversarial target classes. Existing methods for detecting whether a classifier is backdoor attacked are mostly designed for attacks with a single adversarial target (e.g., all-to-one attack). To the best of our knowledge, without supervision, no existing methods can effectively address the more general X2X attack with an arbitrary number of source classes, each paired with an arbitrary target class. In this paper, we propose UMD, the first Unsupervised Model Detection method that effectively detects X2X backdoor attacks via a joint inference of the adversarial (source, target) class pairs. In particular, we first define a novel transferability statistic to measure and select a subset of putative backdoor class pairs based on a proposed clustering approach. Then, these selected class pairs are jointly assessed based on an aggregation of their reverse-engineered trigger size for detection inference, using a robust and unsupervised anomaly detector we proposed. We conduct comprehensive evaluations on CIFAR-10, GTSRB, and Imagenette dataset, and show that our unsupervised UMD outperforms SOTA detectors (even with supervision) by 17%, 4%, and 8%, respectively, in terms of the detection accuracy against diverse X2X attacks. We also show the strong detection performance of UMD against several strong adaptive attacks.

        ----

        ## [1582] Random Shuffle Transformer for Image Restoration

        **Authors**: *Jie Xiao, Xueyang Fu, Man Zhou, Hongjian Liu, Zheng-Jun Zha*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/xiao23a.html](https://proceedings.mlr.press/v202/xiao23a.html)

        **Abstract**:

        Non-local interactions play a vital role in boosting performance for image restoration. However, local window Transformer has been preferred due to its efficiency for processing high-resolution images. The superiority in efficiency comes at the cost of sacrificing the ability to model non-local interactions. In this paper, we present that local window Transformer can also function as modeling non-local interactions. The counterintuitive function is based on the permutation-equivariance of self-attention. The basic principle is quite simple: by randomly shuffling the input, local self-attention also has the potential to model non-local interactions without introducing extra parameters. Our random shuffle strategy enjoys elegant theoretical guarantees in extending the local scope. The resulting Transformer dubbed ShuffleFormer is capable of processing high-resolution images efficiently while modeling non-local interactions. Extensive experiments demonstrate the effectiveness of ShuffleFormer across a variety of image restoration tasks, including image denoising, deraining, and deblurring. Code is available at https://github.com/jiexiaou/ShuffleFormer.

        ----

        ## [1583] Communication-Efficient Federated Hypergradient Computation via Aggregated Iterative Differentiation

        **Authors**: *Peiyao Xiao, Kaiyi Ji*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/xiao23b.html](https://proceedings.mlr.press/v202/xiao23b.html)

        **Abstract**:

        Federated bilevel optimization has attracted increasing attention due to emerging machine learning and communication applications. The biggest challenge lies in computing the gradient of the upper-level objective function (i.e., hypergradient) in the federated setting due to the nonlinear and distributed construction of a series of global Hessian matrices. In this paper, we propose a novel communication-efficient federated hypergradient estimator via aggregated iterative differentiation (AggITD). AggITD is simple to implement and significantly reduces the communication cost by conducting the federated hypergradient estimation and the lower-level optimization simultaneously. We show that the proposed AggITD-based algorithm achieves the same sample complexity as existing approximate implicit differentiation (AID)-based approaches with much fewer communication rounds in the presence of data heterogeneity. Our results also shed light on the great advantage of ITD over AID in the federated/distributed hypergradient estimation. This differs from the comparison in the non-distributed bilevel optimization, where ITD is less efficient than AID. Our extensive experiments demonstrate the great effectiveness and communication efficiency of the proposed method.

        ----

        ## [1584] SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models

        **Authors**: *Guangxuan Xiao, Ji Lin, Mickaël Seznec, Hao Wu, Julien Demouth, Song Han*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/xiao23c.html](https://proceedings.mlr.press/v202/xiao23c.html)

        **Abstract**:

        Large language models (LLMs) show excellent performance but are compute- and memory-intensive. Quantization can reduce memory and accelerate inference. However, existing methods cannot maintain accuracy and hardware efficiency at the same time. We propose SmoothQuant, a training-free, accuracy-preserving, and general-purpose post-training quantization (PTQ) solution to enable 8-bit weight, 8-bit activation (W8A8) quantization for LLMs. Based on the fact that weights are easy to quantize while activations are not, SmoothQuant smooths the activation outliers by offline migrating the quantization difficulty from activations to weights with a mathematically equivalent transformation. SmoothQuant enables an INT8 quantization of both weights and activations for all the matrix multiplications in LLMs, including OPT, BLOOM, GLM, MT-NLG, and LLaMA family. We demonstrate up to 1.56$\times$ speedup and 2$\times$ memory reduction for LLMs with negligible loss in accuracy. SmoothQuant enables serving 530B LLM within a single node. Our work offers a turn-key solution that reduces hardware costs and democratizes LLMs.

        ----

        ## [1585] On the Forward Invariance of Neural ODEs

        **Authors**: *Wei Xiao, Tsun-Hsuan Wang, Ramin M. Hasani, Mathias Lechner, Yutong Ban, Chuang Gan, Daniela Rus*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/xiao23d.html](https://proceedings.mlr.press/v202/xiao23d.html)

        **Abstract**:

        We propose a new method to ensure neural ordinary differential equations (ODEs) satisfy output specifications by using invariance set propagation. Our approach uses a class of control barrier functions to transform output specifications into constraints on the parameters and inputs of the learning system. This setup allows us to achieve output specification guarantees simply by changing the constrained parameters/inputs both during training and inference. Moreover, we demonstrate that our invariance set propagation through data-controlled neural ODEs not only maintains generalization performance but also creates an additional degree of robustness by enabling causal manipulation of the system’s parameters/inputs. We test our method on a series of representation learning tasks, including modeling physical dynamics and convexity portraits, as well as safe collision avoidance for autonomous vehicles.

        ----

        ## [1586] COMCAT: Towards Efficient Compression and Customization of Attention-Based Vision Models

        **Authors**: *Jinqi Xiao, Miao Yin, Yu Gong, Xiao Zang, Jian Ren, Bo Yuan*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/xiao23e.html](https://proceedings.mlr.press/v202/xiao23e.html)

        **Abstract**:

        Attention-based vision models, such as Vision Transformer (ViT) and its variants, have shown promising performance in various computer vision tasks. However, these emerging architectures suffer from large model sizes and high computational costs, calling for efficient model compression solutions. To date, pruning ViTs has been well studied, while other compression strategies that have been widely applied in CNN compression, e.g., model factorization, is little explored in the context of ViT compression. This paper explores an efficient method for compressing vision transformers to enrich the toolset for obtaining compact attention-based vision models. Based on the new insight on the multi-head attention layer, we develop a highly efficient ViT compression solution, which outperforms the state-of-the-art pruning methods. For compressing DeiT-small and DeiT-base models on ImageNet, our proposed approach can achieve $0.45%$ and $0.76%$ higher top-1 accuracy even with fewer parameters. Our finding can also be applied to improve the customization efficiency of text-to-image diffusion models, with much faster training (up to $2.6\times$ speedup) and lower extra storage cost (up to $1927.5\times$ reduction) than the existing works.

        ----

        ## [1587] Improving Bi-level Optimization Based Methods with Inspiration from Humans' Classroom Study Techniques

        **Authors**: *Pengtao Xie*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/xie23a.html](https://proceedings.mlr.press/v202/xie23a.html)

        **Abstract**:

        In humans’ classroom learning, many effective study techniques (e.g., the Feynman technique, peer questioning, etc.) have been developed to improve learning outcomes. We are interested in investigating whether these techniques can inspire the development of ML training strategies to improve bi-level optimization (BLO) based methods. Towards this goal, we develop a general framework, Skillearn, which consists of basic elements such as learners, interaction functions, learning stages, etc. These elements can be flexibly configured to create various training strategies, each emulating a study technique of humans. In case studies, we apply Skillearn to create new training strategies, by emulating the Feynman technique and peer questioning, which are two broadly adopted techniques in humans’ classroom learning. These training strategies are used for improving two BLO based applications including neural architecture search and data weighting. Experiments on various datasets demonstrate the effectiveness of our methods.

        ----

        ## [1588] Future-conditioned Unsupervised Pretraining for Decision Transformer

        **Authors**: *Zhihui Xie, Zichuan Lin, Deheng Ye, Qiang Fu, Yang Wei, Shuai Li*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/xie23b.html](https://proceedings.mlr.press/v202/xie23b.html)

        **Abstract**:

        Recent research in offline reinforcement learning (RL) has demonstrated that return-conditioned supervised learning is a powerful paradigm for decision-making problems. While promising, return conditioning is limited to training data labeled with rewards and therefore faces challenges in learning from unsupervised data. In this work, we aim to utilize generalized future conditioning to enable efficient unsupervised pretraining from reward-free and sub-optimal offline data. We propose Pretrained Decision Transformer (PDT), a conceptually simple approach for unsupervised RL pretraining. PDT leverages future trajectory information as a privileged context to predict actions during training. The ability to make decisions based on both present and future factors enhances PDT’s capability for generalization. Besides, this feature can be easily incorporated into a return-conditioned framework for online finetuning, by assigning return values to possible futures and sampling future embeddings based on their respective values. Empirically, PDT outperforms or performs on par with its supervised pretraining counterpart, especially when dealing with sub-optimal data. Further analysis reveals that PDT can extract diverse behaviors from offline data and controllably sample high-return behaviors by online finetuning. Code is available at here.

        ----

        ## [1589] DeSRA: Detect and Delete the Artifacts of GAN-based Real-World Super-Resolution Models

        **Authors**: *Liangbin Xie, Xintao Wang, Xiangyu Chen, Gen Li, Ying Shan, Jiantao Zhou, Chao Dong*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/xie23c.html](https://proceedings.mlr.press/v202/xie23c.html)

        **Abstract**:

        Image super-resolution (SR) with generative adversarial networks (GAN) has achieved great success in restoring realistic details. However, it is notorious that GAN-based SR models will inevitably produce unpleasant and undesirable artifacts, especially in practical scenarios. Previous works typically suppress artifacts with an extra loss penalty in the training phase. They only work for in-distribution artifact types generated during training. When applied in real-world scenarios, we observe that those improved methods still generate obviously annoying artifacts during inference. In this paper, we analyze the cause and characteristics of the GAN artifacts produced in unseen test data without ground-truths. We then develop a novel method, namely, DeSRA, to Detect and then “Delete” those SR Artifacts in practice. Specifically, we propose to measure a relative local variance distance from MSE-SR results and GAN-SR results, and locate the problematic areas based on the above distance and semantic-aware thresholds. After detecting the artifact regions, we develop a finetune procedure to improve GAN-based SR models with a few samples, so that they can deal with similar types of artifacts in more unseen real data. Equipped with our DeSRA, we can successfully eliminate artifacts from inference and improve the ability of SR models to be applied in real-world scenarios. The code will be available at https://github.com/TencentARC/DeSRA.

        ----

        ## [1590] Semiparametrically Efficient Off-Policy Evaluation in Linear Markov Decision Processes

        **Authors**: *Chuhan Xie, Wenhao Yang, Zhihua Zhang*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/xie23d.html](https://proceedings.mlr.press/v202/xie23d.html)

        **Abstract**:

        We study semiparametrically efficient estimation in off-policy evaluation (OPE) where the underlying Markov decision process (MDP) is linear with a known feature map. We characterize the variance lower bound for regular estimators in the linear MDP setting and propose an efficient estimator whose variance achieves that lower bound. Consistency and asymptotic normality of our estimator are established under mild conditions, which merely requires the only infinite-dimensional nuisance parameter to be estimated at a $n^{-1/4}$ convergence rate. We also construct an asymptotically valid confidence interval for statistical inference and conduct simulation studies to validate our results. To our knowledge, this is the first work that concerns efficient estimation in the presence of a known structure of MDPs in the OPE literature.

        ----

        ## [1591] A Critical View of Vision-Based Long-Term Dynamics Prediction Under Environment Misalignment

        **Authors**: *Hanchen Xie, Jiageng Zhu, Mahyar Khayatkhoei, Jiazhi Li, Mohamed E. Hussein, Wael AbdAlmageed*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/xie23e.html](https://proceedings.mlr.press/v202/xie23e.html)

        **Abstract**:

        Dynamics prediction, which is the problem of predicting future states of scene objects based on current and prior states, is drawing increasing attention as an instance of learning physics. To solve this problem, Region Proposal Convolutional Interaction Network (RPCIN), a vision-based model, was proposed and achieved state-of-the-art performance in long-term prediction. RPCIN only takes raw images and simple object descriptions, such as the bounding box and segmentation mask of each object, as input. However, despite its success, the model’s capability can be compromised under conditions of environment misalignment. In this paper, we investigate two challenging conditions for environment misalignment: Cross-Domain and Cross-Context by proposing four datasets that are designed for these challenges: SimB-Border, SimB-Split, BlenB-Border, and BlenB-Split. The datasets cover two domains and two contexts. Using RPCIN as a probe, experiments conducted on the combinations of the proposed datasets reveal potential weaknesses of the vision-based long-term dynamics prediction model. Furthermore, we propose a promising direction to mitigate the Cross-Domain challenge and provide concrete evidence supporting such a direction, which provides dramatic alleviation of the challenge on the proposed datasets.

        ----

        ## [1592] Controlling Type Confounding in Ad Hoc Teamwork with Instance-wise Teammate Feedback Rectification

        **Authors**: *Dong Xing, Pengjie Gu, Qian Zheng, Xinrun Wang, Shanqi Liu, Longtao Zheng, Bo An, Gang Pan*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/xing23a.html](https://proceedings.mlr.press/v202/xing23a.html)

        **Abstract**:

        Ad hoc teamwork requires an agent to cooperate with unknown teammates without prior coordination. Many works propose to abstract teammate instances into high-level representation of types and then pre-train the best response for each type. However, most of them do not consider the distribution of teammate instances within a type. This could expose the agent to the hidden risk of type confounding. In the worst case, the best response for an abstract teammate type could be the worst response for all specific instances of that type. This work addresses the issue from the lens of causal inference. We first theoretically demonstrate that this phenomenon is due to the spurious correlation brought by uncontrolled teammate distribution. Then, we propose our solution, CTCAT, which disentangles such correlation through an instance-wise teammate feedback rectification. This operation reweights the interaction of teammate instances within a shared type to reduce the influence of type confounding. The effect of CTCAT is evaluated in multiple domains, including classic ad hoc teamwork tasks and real-world scenarios. Results show that CTCAT is robust to the influence of type confounding, a practical issue that directly hazards the robustness of our trained agents but was unnoticed in previous works.

        ----

        ## [1593] Universal Morphology Control via Contextual Modulation

        **Authors**: *Zheng Xiong, Jacob Beck, Shimon Whiteson*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/xiong23a.html](https://proceedings.mlr.press/v202/xiong23a.html)

        **Abstract**:

        Learning a universal policy across different robot morphologies can significantly improve learning efficiency and generalization in continuous control. However, it poses a challenging multi-task reinforcement learning problem, as the optimal policy may be quite different across robots and critically depend on the morphology. Existing methods utilize graph neural networks or transformers to handle heterogeneous state and action spaces across different morphologies, but pay little attention to the dependency of a robot’s control policy on its morphology context. In this paper, we propose a hierarchical architecture to better model this dependency via contextual modulation, which includes two key submodules: (1) Instead of enforcing hard parameter sharing across robots, we use hypernetworks to generate morphology-dependent control parameters; (2) We propose a fixed attention mechanism that solely depends on the morphology to modulate the interactions between different limbs in a robot. Experimental results show that our method not only improves learning performance on a diverse set of training robots, but also generalizes better to unseen morphologies in a zero-shot fashion. The code is publicly available at https://github.com/MasterXiong/ModuMorph.

        ----

        ## [1594] Relevant Walk Search for Explaining Graph Neural Networks

        **Authors**: *Ping Xiong, Thomas Schnake, Michael Gastegger, Grégoire Montavon, Klaus-Robert Müller, Shinichi Nakajima*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/xiong23b.html](https://proceedings.mlr.press/v202/xiong23b.html)

        **Abstract**:

        Graph Neural Networks (GNNs) have become important machine learning tools for graph analysis, and its explainability is crucial for safety, fairness, and robustness. Layer-wise relevance propagation for GNNs (GNN-LRP) evaluates the relevance of walks to reveal important information flows in the network, and provides higher-order explanations, which have been shown to be superior to the lower-order, i.e., node-/edge-level, explanations. However, identifying relevant walks by GNN-LRP requires exponential computational complexity with respect to the network depth, which we will remedy in this paper. Specifically, we propose polynomial-time algorithms for finding top-$K$ relevant walks, which drastically reduces the computation and thus increases the applicability of GNN-LRP to large-scale problems. Our proposed algorithms are based on the max-product algorithm—a common tool for finding the maximum likelihood configurations in probabilistic graphical models—and can find the most relevant walks exactly at the neuron level and approximately at the node level. Our experiments demonstrate the performance of our algorithms at scale and their utility across application domains, i.e., on epidemiology, molecular, and natural language benchmarks. We provide our codes under github.com/xiong-ping/rel_walk_gnnlrp.

        ----

        ## [1595] Why do Nearest Neighbor Language Models Work?

        **Authors**: *Frank F. Xu, Uri Alon, Graham Neubig*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/xu23a.html](https://proceedings.mlr.press/v202/xu23a.html)

        **Abstract**:

        Language models (LMs) compute the probability of a text by sequentially computing a representation of an already-seen context and using this representation to predict the next word. Currently, most LMs calculate these representations through a neural network consuming the immediate previous context. However recently, retrieval-augmented LMs have shown to improve over standard neural LMs, by accessing information retrieved from a large datastore, in addition to their standard, parametric, next-word prediction. In this paper, we set out to understand why retrieval-augmented language models, and specifically why k-nearest neighbor language models (kNN-LMs) perform better than standard parametric LMs, even when the k-nearest neighbor component retrieves examples from the same training set that the LM was originally trained on. To this end, we perform analysis of various dimensions over which kNN-LM diverges from standard LMs, and investigate these dimensions one by one. Empirically, we identify three main reasons why kNN-LM performs better than standard LMs: using a different input representation for predicting the next tokens, approximate kNN search, and the importance of softmax temperature for the kNN distribution. Further, we incorporate some insights into the standard parametric LM, improving performance without the need for an explicit retrieval component. The code is available at https://github.com/frankxu2004/knnlm-why.

        ----

        ## [1596] MixFlows: principled variational inference via mixed flows

        **Authors**: *Zuheng Xu, Naitong Chen, Trevor Campbell*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/xu23b.html](https://proceedings.mlr.press/v202/xu23b.html)

        **Abstract**:

        This work presents mixed variational flows (MixFlows), a new variational family that consists of a mixture of repeated applications of a map to an initial reference distribution. First, we provide efficient algorithms for i.i.d. sampling, density evaluation, and unbiased ELBO estimation. We then show that MixFlows have MCMC-like convergence guarantees when the flow map is ergodic and measure-preserving, and provide bounds on the accumulation of error for practical implementations where the flow map is approximated. Finally, we develop an implementation of MixFlows based on uncorrected discretized Hamiltonian dynamics combined with deterministic momentum refreshment. Simulated and real data experiments show that MixFlows can provide more reliable posterior approximations than several black-box normalizing flows, as well as samples of comparable quality to those obtained from state-of-the-art MCMC methods.

        ----

        ## [1597] Bit Allocation using Optimization

        **Authors**: *Tongda Xu, Han Gao, Chenjian Gao, Yuanyuan Wang, Dailan He, Jinyong Pi, Jixiang Luo, Ziyu Zhu, Mao Ye, Hongwei Qin, Yan Wang, Jingjing Liu, Ya-Qin Zhang*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/xu23c.html](https://proceedings.mlr.press/v202/xu23c.html)

        **Abstract**:

        In this paper, we consider the problem of bit allocation in Neural Video Compression (NVC). First, we reveal a fundamental relationship between bit allocation in NVC and Semi-Amortized Variational Inference (SAVI). Specifically, we show that SAVI with GoP (Group-of-Picture)-level likelihood is equivalent to pixel-level bit allocation with precise rate & quality dependency model. Based on this equivalence, we establish a new paradigm of bit allocation using SAVI. Different from previous bit allocation methods, our approach requires no empirical model and is thus optimal. Moreover, as the original SAVI using gradient ascent only applies to single-level latent, we extend the SAVI to multi-level such as NVC by recursively applying back-propagating through gradient ascent. Finally, we propose a tractable approximation for practical implementation. Our method can be applied to scenarios where performance outweights encoding speed, and serves as an empirical bound on the R-D performance of bit allocation. Experimental results show that current state-of-the-art bit allocation algorithms still have a room of $\approx 0.5$ dB PSNR to improve compared with ours. Code is available at https://github.com/tongdaxu/Bit-Allocation-Using-Optimization.

        ----

        ## [1598] Regret Bounds for Markov Decision Processes with Recursive Optimized Certainty Equivalents

        **Authors**: *Wenhao Xu, Xuefeng Gao, Xuedong He*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/xu23d.html](https://proceedings.mlr.press/v202/xu23d.html)

        **Abstract**:

        The optimized certainty equivalent (OCE) is a family of risk measures that cover important examples such as entropic risk, conditional value-at-risk and mean-variance models. In this paper, we propose a new episodic risk-sensitive reinforcement learning formulation based on tabular Markov decision processes with recursive OCEs. We design an efficient learning algorithm for this problem based on value iteration and upper confidence bound. We derive an upper bound on the regret of the proposed algorithm, and also establish a minimax lower bound. Our bounds show that the regret rate achieved by our proposed algorithm has optimal dependence on the number of episodes and the number of actions.

        ----

        ## [1599] Probabilistic Categorical Adversarial Attack and Adversarial Training

        **Authors**: *Han Xu, Pengfei He, Jie Ren, Yuxuan Wan, Zitao Liu, Hui Liu, Jiliang Tang*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/xu23e.html](https://proceedings.mlr.press/v202/xu23e.html)

        **Abstract**:

        The studies on adversarial attacks and defenses have greatly improved the robustness of Deep Neural Networks (DNNs). Most advanced approaches have been overwhelmingly designed for continuous data such as images. However, these achievements are still hard to be generalized to categorical data. To bridge this gap, we propose a novel framework, Probabilistic Categorical Adversarial Attack (or PCAA). It transfers the discrete optimization problem of finding categorical adversarial examples to a continuous problem that can be solved via gradient-based methods. We analyze the optimality (attack success rate) and time complexity of PCAA to demonstrate its significant advantage over current search-based attacks. More importantly, through extensive empirical studies, we demonstrate that the well-established defenses for continuous data, such as adversarial training and TRADES, can be easily accommodated to defend DNNs for categorical data.

        ----

        

[Go to the previous page](ICML-2023-list07.md)

[Go to the next page](ICML-2023-list09.md)

[Go to the catalog section](README.md)