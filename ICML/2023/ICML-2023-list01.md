## [0] Data Structures for Density Estimation

        **Authors**: *Anders Aamand, Alexandr Andoni, Justin Y. Chen, Piotr Indyk, Shyam Narayanan, Sandeep Silwal*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/aamand23a.html](https://proceedings.mlr.press/v202/aamand23a.html)

        **Abstract**:

        We study statistical/computational tradeoffs for the following density estimation problem: given $k$ distributions $v_1, \ldots, v_k$ over a discrete domain of size $n$, and sampling access to a distribution $p$, identify $v_i$ that is "close" to $p$. Our main result is the first data structure that, given a sublinear (in $n$) number of samples from $p$, identifies $v_i$ in time sublinear in $k$. We also give an improved version of the algorithm of Acharya et al. (2018) that reports $v_i$ in time linear in $k$. The experimental evaluation of the latter algorithm shows that it achieves a significant reduction in the number of operations needed to achieve a given accuracy compared to prior work.

        ----

        ## [1] ClusterFuG: Clustering Fully connected Graphs by Multicut

        **Authors**: *Ahmed Abbas, Paul Swoboda*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/abbas23a.html](https://proceedings.mlr.press/v202/abbas23a.html)

        **Abstract**:

        We propose a graph clustering formulation based on multicut (a.k.a. weighted correlation clustering) on the complete graph. Our formulation does not need specification of the graph topology as in the original sparse formulation of multicut, making our approach simpler and potentially better performing. In contrast to unweighted correlation clustering we allow for a more expressive weighted cost structure. In dense multicut, the clustering objective is given in a factorized form as inner products of node feature vectors. This allows for an efficient formulation and inference in contrast to multicut/weighted correlation clustering, which has at least quadratic representation and computation complexity when working on the complete graph. We show how to rewrite classical greedy algorithms for multicut in our dense setting and how to modify them for greater efficiency and solution quality. In particular, our algorithms scale to graphs with tens of thousands of nodes. Empirical evidence on instance segmentation on Cityscapes and clustering of ImageNet datasets shows the merits of our approach.

        ----

        ## [2] Generalization on the Unseen, Logic Reasoning and Degree Curriculum

        **Authors**: *Emmanuel Abbe, Samy Bengio, Aryo Lotfi, Kevin Rizk*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/abbe23a.html](https://proceedings.mlr.press/v202/abbe23a.html)

        **Abstract**:

        This paper considers the learning of logical (Boolean) functions with focus on the generalization on the unseen (GOTU) setting, a strong case of out-of-distribution generalization. This is motivated by the fact that the rich combinatorial nature of data in certain reasoning tasks (e.g., arithmetic/logic) makes representative data sampling challenging, and learning successfully under GOTU gives a first vignette of an ’extrapolating’ or ’reasoning’ learner. We then study how different network architectures trained by (S)GD perform under GOTU and provide both theoretical and experimental evidence that for a class of network models including instances of Transformers, random features models, and diagonal linear networks, a min-degree-interpolator is learned on the unseen. We also provide evidence that other instances with larger learning rates or mean-field networks reach leaky min-degree solutions. These findings lead to two implications: (1) we provide an explanation to the length generalization problem (e.g., Anil et al. 2022); (2) we introduce a curriculum learning algorithm called Degree-Curriculum that learns monomials more efficiently by incrementing supports.

        ----

        ## [3] Toward Large Kernel Models

        **Authors**: *Amirhesam Abedsoltan, Mikhail Belkin, Parthe Pandit*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/abedsoltan23a.html](https://proceedings.mlr.press/v202/abedsoltan23a.html)

        **Abstract**:

        Recent studies indicate that kernel machines can often perform similarly or better than deep neural networks (DNNs) on small datasets. The interest in kernel machines has been additionally bolstered by the discovery of their equivalence to wide neural networks in certain regimes. However, a key feature of DNNs is their ability to scale the model size and training data size independently, whereas in traditional kernel machines model size is tied to data size. Because of this coupling, scaling kernel machines to large data has been computationally challenging. In this paper, we provide a way forward for constructing large-scale general kernel models, which are a generalization of kernel machines that decouples the model and data, allowing training on large datasets. Specifically, we introduce EigenPro 3.0, an algorithm based on projected dual preconditioned SGD and show scaling to model and data sizes which have not been possible with existing kernel methods. We provide a PyTorch based implementation which can take advantage of multiple GPUs.

        ----

        ## [4] Expertise Trees Resolve Knowledge Limitations in Collective Decision-Making

        **Authors**: *Axel Abels, Tom Lenaerts, Vito Trianni, Ann Nowé*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/abels23a.html](https://proceedings.mlr.press/v202/abels23a.html)

        **Abstract**:

        Experts advising decision-makers are likely to display expertise which varies as a function of the problem instance. In practice, this may lead to sub-optimal or discriminatory decisions against minority cases. In this work, we model such changes in depth and breadth of knowledge as a partitioning of the problem space into regions of differing expertise. We provide here new algorithms that explicitly consider and adapt to the relationship between problem instances and experts’ knowledge. We first propose and highlight the drawbacks of a naive approach based on nearest neighbor queries. To address these drawbacks we then introduce a novel algorithm — expertise trees — that constructs decision trees enabling the learner to select appropriate models. We provide theoretical insights and empirically validate the improved performance of our novel approach on a range of problems for which existing methods proved to be inadequate.

        ----

        ## [5] Comparison of meta-learners for estimating multi-valued treatment heterogeneous effects

        **Authors**: *Naoufal Acharki, Ramiro Lugo, Antoine Bertoncello, Josselin Garnier*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/acharki23a.html](https://proceedings.mlr.press/v202/acharki23a.html)

        **Abstract**:

        Conditional Average Treatment Effects (CATE) estimation is one of the main challenges in causal inference with observational data. In addition to Machine Learning based-models, nonparametric estimators called meta-learners have been developed to estimate the CATE with the main advantage of not restraining the estimation to a specific supervised learning method. This task becomes, however, more complicated when the treatment is not binary as some limitations of the naive extensions emerge. This paper looks into meta-learners for estimating the heterogeneous effects of multi-valued treatments. We consider different meta-learners, and we carry out a theoretical analysis of their error upper bounds as functions of important parameters such as the number of treatment levels, showing that the naive extensions do not always provide satisfactory results. We introduce and discuss meta-learners that perform well as the number of treatments increases. We empirically confirm the strengths and weaknesses of those methods with synthetic and semi-synthetic datasets.

        ----

        ## [6] BNN-DP: Robustness Certification of Bayesian Neural Networks via Dynamic Programming

        **Authors**: *Steven Adams, Andrea Patane, Morteza Lahijanian, Luca Laurenti*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/adams23a.html](https://proceedings.mlr.press/v202/adams23a.html)

        **Abstract**:

        In this paper, we introduce BNN-DP, an efficient algorithmic framework for analysis of adversarial robustness of Bayesian Neural Networks (BNNs). Given a compact set of input points $T\subset \mathbb{R}^n$, BNN-DP computes lower and upper bounds on the BNN’s predictions for all the points in $T$. The framework is based on an interpretation of BNNs as stochastic dynamical systems, which enables the use of Dynamic Programming (DP) algorithms to bound the prediction range along the layers of the network. Specifically, the method uses bound propagation techniques and convex relaxations to derive a backward recursion procedure to over-approximate the prediction range of the BNN with piecewise affine functions. The algorithm is general and can handle both regression and classification tasks. On a set of experiments on various regression and classification tasks and BNN architectures, we show that BNN-DP outperforms state-of-the-art methods by up to four orders of magnitude in both tightness of the bounds and computational efficiency.

        ----

        ## [7] SAM operates far from home: eigenvalue regularization as a dynamical phenomenon

        **Authors**: *Atish Agarwala, Yann N. Dauphin*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/agarwala23a.html](https://proceedings.mlr.press/v202/agarwala23a.html)

        **Abstract**:

        The Sharpness Aware Minimization (SAM) optimization algorithm has been shown to control large eigenvalues of the loss Hessian and provide generalization benefits in a variety of settings. The original motivation for SAM was a modified loss function which penalized sharp minima; subsequent analyses have also focused on the behavior near minima. However, our work reveals that SAM provides a strong regularization of the eigenvalues throughout the learning trajectory. We show that in a simplified setting, SAM dynamically induces a stabilization related to the edge of stability (EOS) phenomenon observed in large learning rate gradient descent. Our theory predicts the largest eigenvalue as a function of the learning rate and SAM radius parameters. Finally, we show that practical models can also exhibit this EOS stabilization, and that understanding SAM must account for these dynamics far away from any minima.

        ----

        ## [8] Second-order regression models exhibit progressive sharpening to the edge of stability

        **Authors**: *Atish Agarwala, Fabian Pedregosa, Jeffrey Pennington*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/agarwala23b.html](https://proceedings.mlr.press/v202/agarwala23b.html)

        **Abstract**:

        Recent studies of gradient descent with large step sizes have shown that there is often a regime with an initial increase in the largest eigenvalue of the loss Hessian (progressive sharpening), followed by a stabilization of the eigenvalue near the maximum value which allows convergence (edge of stability). These phenomena are intrinsically non-linear and do not happen for models in the constant Neural Tangent Kernel (NTK) regime, for which the predictive function is approximately linear in the parameters. As such, we consider the next simplest class of predictive models, namely those that are quadratic in the parameters, which we call second-order regression models. For quadratic objectives in two dimensions, we prove that this second-order regression model exhibits progressive sharpening of the NTK eigenvalue towards a value that differs slightly from the edge of stability, which we explicitly compute. In higher dimensions, the model generically shows similar behavior, even without the specific structure of a neural network, suggesting that progressive sharpening and edge-of-stability behavior aren’t unique features of neural networks, and could be a more general property of discrete learning algorithms in high-dimensional non-linear models.

        ----

        ## [9] Global optimality of Elman-type RNNs in the mean-field regime

        **Authors**: *Andrea Agazzi, Jianfeng Lu, Sayan Mukherjee*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/agazzi23a.html](https://proceedings.mlr.press/v202/agazzi23a.html)

        **Abstract**:

        We analyze Elman-type recurrent neural networks (RNNs) and their training in the mean-field regime. Specifically, we show convergence of gradient descent training dynamics of the RNN to the corresponding mean-field formulation in the large width limit. We also show that the fixed points of the limiting infinite-width dynamics are globally optimal, under some assumptions on the initialization of the weights. Our results establish optimality for feature-learning with wide RNNs in the mean-field regime.

        ----

        ## [10] SemSup-XC: Semantic Supervision for Zero and Few-shot Extreme Classification

        **Authors**: *Pranjal Aggarwal, Ameet Deshpande, Karthik R. Narasimhan*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/aggarwal23a.html](https://proceedings.mlr.press/v202/aggarwal23a.html)

        **Abstract**:

        Extreme classification (XC) involves predicting over large numbers of classes (thousands to millions), with real-world applications like news article classification and e-commerce product tagging. The zero-shot version of this task requires generalization to novel classes without additional supervision. In this paper, we develop SemSup-XC, a model that achieves state-of-the-art zero-shot and few-shot performance on three XC datasets derived from legal, e-commerce, and Wikipedia data. To develop SemSup-XC, we use automatically collected semantic class descriptions to represent classes and facilitate generalization through a novel hybrid matching module that matches input instances to class descriptions using a combination of semantic and lexical similarity. Trained with contrastive learning, SemSup-XC significantly outperforms baselines and establishes state-of-the-art performance on all three datasets considered, gaining up to 12 precision points on zero-shot and more than 10 precision points on one-shot tests, with similar gains for recall@10. Our ablation studies highlight the relative importance of our hybrid matching module and automatically collected class descriptions.

        ----

        ## [11] Adaptive IMLE for Few-shot Pretraining-free Generative Modelling

        **Authors**: *Mehran Aghabozorgi, Shichong Peng, Ke Li*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/aghabozorgi23a.html](https://proceedings.mlr.press/v202/aghabozorgi23a.html)

        **Abstract**:

        Despite their success on large datasets, GANs have been difficult to apply in the few-shot setting, where only a limited number of training examples are provided. Due to mode collapse, GANs tend to ignore some training examples, causing overfitting to a subset of the training dataset, which is small in the first place. A recent method called Implicit Maximum Likelihood Estimation (IMLE) is an alternative to GAN that tries to address this issue. It uses the same kind of generators as GANs but trains it with a different objective that encourages mode coverage. However, the theoretical guarantees of IMLE hold under a restrictive condition that the optimal likelihood at all data points is the same. In this paper, we present a more generalized formulation of IMLE which includes the original formulation as a special case, and we prove that the theoretical guarantees hold under weaker conditions. Using this generalized formulation, we further derive a new algorithm, which we dub Adaptive IMLE, which can adapt to the varying difficulty of different training examples. We demonstrate on multiple few-shot image synthesis datasets that our method significantly outperforms existing methods. Our code is available at https://github.com/mehranagh20/AdaIMLE.

        ----

        ## [12] Scaling Laws for Generative Mixed-Modal Language Models

        **Authors**: *Armen Aghajanyan, Lili Yu, Alexis Conneau, Wei-Ning Hsu, Karen Hambardzumyan, Susan Zhang, Stephen Roller, Naman Goyal, Omer Levy, Luke Zettlemoyer*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/aghajanyan23a.html](https://proceedings.mlr.press/v202/aghajanyan23a.html)

        **Abstract**:

        Generative language models define distributions over sequences of tokens that can represent essentially any combination of data modalities (e.g., any permutation of image tokens from VQ-VAEs, speech tokens from HuBERT, BPE tokens for language or code, and so on). To better understand the scaling properties of such mixed-modal models, we conducted over 250 experiments using seven different modalities and model sizes ranging from 8 million to 30 billion, trained on 5-100 billion tokens. We report new mixed-modal scaling laws that unify the contributions of individual modalities and the interactions between them. Specifically, we explicitly model the optimal synergy and competition due to data and model size as an additive term to previous uni-modal scaling laws. We also find four empirical phenomena observed during the training, such as emergent coordinate-ascent style training that naturally alternates between modalities, guidelines for selecting critical hyper-parameters, and connections between mixed-modal competition and training stability. Finally, we test our scaling law by training a 30B speech-text model, which significantly outperforms the corresponding unimodal models. Overall, our research provides valuable insights into the design and training of mixed-modal generative models, an important new class of unified models that have unique distributional properties.

        ----

        ## [13] Hypothesis Transfer Learning with Surrogate Classification Losses: Generalization Bounds through Algorithmic Stability

        **Authors**: *Anass Aghbalou, Guillaume Staerman*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/aghbalou23a.html](https://proceedings.mlr.press/v202/aghbalou23a.html)

        **Abstract**:

        Hypothesis transfer learning (HTL) contrasts domain adaptation by allowing for a previous task leverage, named the source, into a new one, the target, without requiring access to the source data. Indeed, HTL relies only on a hypothesis learnt from such source data, relieving the hurdle of expansive data storage and providing great practical benefits. Hence, HTL is highly beneficial for real-world applications relying on big data. The analysis of such a method from a theoretical perspective faces multiple challenges, particularly in classification tasks. This paper deals with this problem by studying the learning theory of HTL through algorithmic stability, an attractive theoretical framework for machine learning algorithms analysis. In particular, we are interested in the statistical behavior of the regularized empirical risk minimizers in the case of binary classification. Our stability analysis provides learning guarantees under mild assumptions. Consequently, we derive several complexity-free generalization bounds for essential statistical quantities like the training error, the excess risk and cross-validation estimates. These refined bounds allow understanding the benefits of transfer learning and comparing the behavior of standard losses in different scenarios, leading to valuable insights for practitioners.

        ----

        ## [14] Constrained Causal Bayesian Optimization

        **Authors**: *Virginia Aglietti, Alan Malek, Ira Ktena, Silvia Chiappa*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/aglietti23a.html](https://proceedings.mlr.press/v202/aglietti23a.html)

        **Abstract**:

        We propose constrained causal Bayesian optimization (cCBO), an approach for finding interventions in a known causal graph that optimize a target variable under some constraints. cCBO first reduces the search space by exploiting the graph structure and, if available, an observational dataset; and then solves the restricted optimization problem by modelling target and constraint quantities using Gaussian processes and by sequentially selecting interventions via a constrained expected improvement acquisition function. We propose different surrogate models that enable to integrate observational and interventional data while capturing correlation among effects with increasing levels of sophistication. We evaluate cCBO on artificial and real-world causal graphs showing successful trade off between fast convergence and percentage of feasible interventions.

        ----

        ## [15] Explaining the effects of non-convergent MCMC in the training of Energy-Based Models

        **Authors**: *Elisabeth Agoritsas, Giovanni Catania, Aurélien Decelle, Beatriz Seoane*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/agoritsas23a.html](https://proceedings.mlr.press/v202/agoritsas23a.html)

        **Abstract**:

        In this paper, we quantify the impact of using non-convergent Markov chains to train Energy-Based models (EBMs). In particular, we show analytically that EBMs trained with non-persistent short runs to estimate the gradient can perfectly reproduce a set of empirical statistics of the data, not at the level of the equilibrium measure, but through a precise dynamical process. Our results provide a first-principles explanation for the observations of recent works proposing the strategy of using short runs starting from random initial conditions as an efficient way to generate high-quality samples in EBMs, and lay the groundwork for using EBMs as diffusion models. After explaining this effect in generic EBMs, we analyze two solvable models in which the effect of the non-convergent sampling in the trained parameters can be described in detail. Finally, we test these predictions numerically on a ConvNet EBM and a Boltzmann machine.

        ----

        ## [16] Using Large Language Models to Simulate Multiple Humans and Replicate Human Subject Studies

        **Authors**: *Gati V. Aher, Rosa I. Arriaga, Adam Tauman Kalai*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/aher23a.html](https://proceedings.mlr.press/v202/aher23a.html)

        **Abstract**:

        We introduce a new type of test, called a Turing Experiment (TE), for evaluating to what extent a given language model, such as GPT models, can simulate different aspects of human behavior. A TE can also reveal consistent distortions in a language model’s simulation of a specific human behavior. Unlike the Turing Test, which involves simulating a single arbitrary individual, a TE requires simulating a representative sample of participants in human subject research. We carry out TEs that attempt to replicate well-established findings from prior studies. We design a methodology for simulating TEs and illustrate its use to compare how well different language models are able to reproduce classic economic, psycholinguistic, and social psychology experiments: Ultimatum Game, Garden Path Sentences, Milgram Shock Experiment, and Wisdom of Crowds. In the first three TEs, the existing findings were replicated using recent models, while the last TE reveals a “hyper-accuracy distortion” present in some language models (including ChatGPT and GPT-4), which could affect downstream applications in education and the arts.

        ----

        ## [17] Interventional Causal Representation Learning

        **Authors**: *Kartik Ahuja, Divyat Mahajan, Yixin Wang, Yoshua Bengio*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/ahuja23a.html](https://proceedings.mlr.press/v202/ahuja23a.html)

        **Abstract**:

        Causal representation learning seeks to extract high-level latent factors from low-level sensory data. Most existing methods rely on observational data and structural assumptions (e.g., conditional independence) to identify the latent factors. However, interventional data is prevalent across applications. Can interventional data facilitate causal representation learning? We explore this question in this paper. The key observation is that interventional data often carries geometric signatures of the latent factors’ support (i.e. what values each latent can possibly take). For example, when the latent factors are causally connected, interventions can break the dependency between the intervened latents’ support and their ancestors’. Leveraging this fact, we prove that the latent causal factors can be identified up to permutation and scaling given data from perfect do interventions. Moreover, we can achieve block affine identification, namely the estimated latent factors are only entangled with a few other latents if we have access to data from imperfect interventions. These results highlight the unique power of interventional data in causal representation learning; they can enable provable identification of latent factors without any assumptions about their distributions or dependency structure.

        ----

        ## [18] Sequential Underspecified Instrument Selection for Cause-Effect Estimation

        **Authors**: *Elisabeth Ailer, Jason S. Hartford, Niki Kilbertus*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/ailer23a.html](https://proceedings.mlr.press/v202/ailer23a.html)

        **Abstract**:

        Instrumental variable (IV) methods are used to estimate causal effects in settings with unobserved confounding, where we cannot directly experiment on the treatment variable. Instruments are variables which only affect the outcome indirectly via the treatment variable(s). Most IV applications focus on low-dimensional treatments and crucially require at least as many instruments as treatments. This assumption is restrictive: in the natural sciences we often seek to infer causal effects of high-dimensional treatments (e.g., the effect of gene expressions or microbiota on health and disease), but can only run few experiments with a limited number of instruments (e.g., drugs or antibiotics). In such under-specified problems, the full treatment effect is not identifiable in a single experiment even in the linear case. We show that one can still reliably recover the projection of the treatment effect onto the instrumented subspace and develop techniques to consistently combine such partial estimates from different sets of instruments. We then leverage our combined estimators in an algorithm that iteratively proposes the most informative instruments at each round of experimentation to maximize the overall information about the full causal effect.

        ----

        ## [19] Atari-5: Distilling the Arcade Learning Environment down to Five Games

        **Authors**: *Matthew Aitchison, Penny Sweetser, Marcus Hutter*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/aitchison23a.html](https://proceedings.mlr.press/v202/aitchison23a.html)

        **Abstract**:

        The Arcade Learning Environment (ALE) has become an essential benchmark for assessing the performance of reinforcement learning algorithms. However, the computational cost of generating results on the entire 57-game dataset limits ALE’s use and makes the reproducibility of many results infeasible. We propose a novel solution to this problem in the form of a principled methodology for selecting small but representative subsets of environments within a benchmark suite. We applied our method to identify a subset of five ALE games, we call Atari-5, which produces 57-game median score estimates within 10% of their true values. Extending the subset to 10-games recovers 80% of the variance for log-scores for all games within the 57-game set. We show this level of compression is possible due to a high degree of correlation between many of the games in ALE.

        ----

        ## [20] Towards credible visual model interpretation with path attribution

        **Authors**: *Naveed Akhtar, Mohammad A. A. K. Jalwana*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/akhtar23a.html](https://proceedings.mlr.press/v202/akhtar23a.html)

        **Abstract**:

        With its inspirational roots in game-theory, path attribution framework stands out among the post-hoc model interpretation techniques due to its axiomatic nature. However, recent developments show that despite being axiomatic, path attribution methods can compute counter-intuitive feature attributions. Not only that, for deep visual models, the methods may also not conform to the original game-theoretic intuitions that are the basis of their axiomatic nature. To address these issues, we perform a systematic investigation of the path attribution framework. We first pinpoint the conditions in which the counter-intuitive attributions of deep visual models can be avoided under this framework. Then, we identify a mechanism of integrating the attributions over the paths such that they computationally conform to the original insights of game-theory. These insights are eventually combined into a method, which provides intuitive and reliable feature attributions. We also establish the findings empirically by evaluating the method on multiple datasets, models and evaluation metrics. Extensive experiments show a consistent quantitative and qualitative gain in the results over the baselines.

        ----

        ## [21] Convergence of First-Order Methods for Constrained Nonconvex Optimization with Dependent Data

        **Authors**: *Ahmet Alacaoglu, Hanbaek Lyu*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/alacaoglu23a.html](https://proceedings.mlr.press/v202/alacaoglu23a.html)

        **Abstract**:

        We focus on analyzing the classical stochastic projected gradient methods under a general dependent data sampling scheme for constrained smooth nonconvex optimization. We show the worst-case rate of convergence $\tilde{O}(t^{-1/4})$ and complexity $\tilde{O}(\varepsilon^{-4})$ for achieving an $\varepsilon$-near stationary point in terms of the norm of the gradient of Moreau envelope and gradient mapping. While classical convergence guarantee requires i.i.d. data sampling from the target distribution, we only require a mild mixing condition of the conditional distribution, which holds for a wide class of Markov chain sampling algorithms. This improves the existing complexity for the constrained smooth nonconvex optimization with dependent data from $\tilde{O}(\varepsilon^{-8})$ to $\tilde{O}(\varepsilon^{-4})$ with a significantly simpler analysis. We illustrate the generality of our approach by deriving convergence results with dependent data for stochastic proximal gradient methods, adaptive stochastic gradient algorithm AdaGrad and stochastic gradient algorithm with heavy ball momentum. As an application, we obtain first online nonnegative matrix factorization algorithms for dependent data based on stochastic projected gradient methods with adaptive step sizes and optimal rate of convergence.

        ----

        ## [22] Recasting Self-Attention with Holographic Reduced Representations

        **Authors**: *Mohammad Mahmudul Alam, Edward Raff, Stella Biderman, Tim Oates, James Holt*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/alam23a.html](https://proceedings.mlr.press/v202/alam23a.html)

        **Abstract**:

        In recent years, self-attention has become the dominant paradigm for sequence modeling in a variety of domains. However, in domains with very long sequence lengths the $\mathcal{O}(T^2)$ memory and $\mathcal{O}(T^2 H)$ compute costs can make using transformers infeasible. Motivated by problems in malware detection, where sequence lengths of $T \geq 100,000$ are a roadblock to deep learning, we re-cast self-attention using the neuro-symbolic approach of Holographic Reduced Representations (HRR). In doing so we perform the same high-level strategy of the standard self-attention: a set of queries matching against a set of keys, and returning a weighted response of the values for each key. Implemented as a “Hrrformer” we obtain several benefits including $\mathcal{O}(T H \log H)$ time complexity, $\mathcal{O}(T H)$ space complexity, and convergence in $10\times$ fewer epochs. Nevertheless, the Hrrformer achieves near state-of-the-art accuracy on LRA benchmarks and we are able to learn with just a single layer. Combined, these benefits make our Hrrformer the first viable Transformer for such long malware classification sequences and up to $280\times$ faster to train on the Long Range Arena benchmark.

        ----

        ## [23] The Saddle-Point Method in Differential Privacy

        **Authors**: *Wael Alghamdi, Juan Felipe Gómez, Shahab Asoodeh, Flávio P. Calmon, Oliver Kosut, Lalitha Sankar*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/alghamdi23a.html](https://proceedings.mlr.press/v202/alghamdi23a.html)

        **Abstract**:

        We characterize the differential privacy guarantees of privacy mechanisms in the large-composition regime, i.e., when a privacy mechanism is sequentially applied a large number of times to sensitive data. Via exponentially tilting the privacy loss random variable, we derive a new formula for the privacy curve expressing it as a contour integral over an integration path that runs parallel to the imaginary axis with a free real-axis intercept. Then, using the method of steepest descent from mathematical physics, we demonstrate that the choice of saddle-point as the real-axis intercept yields closed-form accurate approximations of the desired contour integral. This procedure—dubbed the saddle-point accountant (SPA)—yields a constant-time accurate approximation of the privacy curve. Theoretically, our results can be viewed as a refinement of both Gaussian Differential Privacy and the moments accountant method found in Rényi Differential Privacy. In practice, we demonstrate through numerical experiments that the SPA provides a precise approximation of privacy guarantees competitive with purely numerical-based methods (such as FFT-based accountants), while enjoying closed-form mathematical expressions.

        ----

        ## [24] Nonlinear Advantage: Trained Networks Might Not Be As Complex as You Think

        **Authors**: *Christian H. X. Ali Mehmeti-Göpel, Jan Disselhoff*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/ali-mehmeti-gopel23a.html](https://proceedings.mlr.press/v202/ali-mehmeti-gopel23a.html)

        **Abstract**:

        We perform an empirical study of the behaviour of deep networks when fully linearizing some of its feature channels through a sparsity prior on the overall number of nonlinear units in the network. In experiments on image classification and machine translation tasks, we investigate how much we can simplify the network function towards linearity before performance collapses. First, we observe a significant performance gap when reducing nonlinearity in the network function early on as opposed to late in training, in-line with recent observations on the time-evolution of the data-dependent NTK. Second, we find that after training, we are able to linearize a significant number of nonlinear units while maintaining a high performance, indicating that much of a network’s expressivity remains unused but helps gradient descent in early stages of training. To characterize the depth of the resulting partially linearized network, we introduce a measure called average path length, representing the average number of active nonlinearities encountered along a path in the network graph. Under sparsity pressure, we find that the remaining nonlinear units organize into distinct structures, forming core-networks of near constant effective depth and width, which in turn depend on task difficulty.

        ----

        ## [25] A Simple Zero-shot Prompt Weighting Technique to Improve Prompt Ensembling in Text-Image Models

        **Authors**: *James Urquhart Allingham, Jie Ren, Michael W. Dusenberry, Xiuye Gu, Yin Cui, Dustin Tran, Jeremiah Zhe Liu, Balaji Lakshminarayanan*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/allingham23a.html](https://proceedings.mlr.press/v202/allingham23a.html)

        **Abstract**:

        Contrastively trained text-image models have the remarkable ability to perform zero-shot classification, that is, classifying previously unseen images into categories that the model has never been explicitly trained to identify. However, these zero-shot classifiers need prompt engineering to achieve high accuracy. Prompt engineering typically requires hand-crafting a set of prompts for individual downstream tasks. In this work, we aim to automate this prompt engineering and improve zero-shot accuracy through prompt ensembling. In particular, we ask “Given a large pool of prompts, can we automatically score the prompts and ensemble those that are most suitable for a particular downstream dataset, without needing access to labeled validation data?". We demonstrate that this is possible. In doing so, we identify several pathologies in a naive prompt scoring method where the score can be easily overconfident due to biases in pre-training and test data, and we propose a novel prompt scoring method that corrects for the biases. Using our proposed scoring method to create a weighted average prompt ensemble, our method overall outperforms equal average ensemble, as well as hand-crafted prompts, on ImageNet, 4 of its variants, and 11 fine-grained classification benchmarks. while being fully automatic, optimization-free, and not requiring access to labeled validation data.

        ----

        ## [26] On the Privacy-Robustness-Utility Trilemma in Distributed Learning

        **Authors**: *Youssef Allouah, Rachid Guerraoui, Nirupam Gupta, Rafael Pinot, John Stephan*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/allouah23a.html](https://proceedings.mlr.press/v202/allouah23a.html)

        **Abstract**:

        The ubiquity of distributed machine learning (ML) in sensitive public domain applications calls for algorithms that protect data privacy, while being robust to faults and adversarial behaviors. Although privacy and robustness have been extensively studied independently in distributed ML, their synthesis remains poorly understood. We present the first tight analysis of the error incurred by any algorithm ensuring robustness against a fraction of adversarial machines, as well as differential privacy (DP) for honest machines’ data against any other curious entity. Our analysis exhibits a fundamental trade-off between privacy, robustness, and utility. To prove our lower bound, we consider the case of mean estimation, subject to distributed DP and robustness constraints, and devise reductions to centralized estimation of one-way marginals. We prove our matching upper bound by presenting a new distributed ML algorithm using a high-dimensional robust aggregation rule. The latter amortizes the dependence on the dimension in the error (caused by adversarial workers and DP), while being agnostic to the statistical properties of the data.

        ----

        ## [27] Differentially Private Distributed Bayesian Linear Regression with MCMC

        **Authors**: *Baris Alparslan, Sinan Yildirim, S. Ilker Birbil*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/alparslan23a.html](https://proceedings.mlr.press/v202/alparslan23a.html)

        **Abstract**:

        We propose a novel Bayesian inference framework for distributed differentially private linear regression. We consider a distributed setting where multiple parties hold parts of the data and share certain summary statistics of their portions in privacy-preserving noise. We develop a novel generative statistical model for privately shared statistics, which exploits a useful distributional relation between the summary statistics of linear regression. We propose Bayesian estimation of the regression coefficients, mainly using Markov chain Monte Carlo algorithms, while we also provide a fast version that performs approximate Bayesian estimation in one iteration. The proposed methods have computational advantages over their competitors. We provide numerical results on both real and simulated data, which demonstrate that the proposed algorithms provide well-rounded estimation and prediction.

        ----

        ## [28] Robust and Scalable Bayesian Online Changepoint Detection

        **Authors**: *Matías Altamirano, François-Xavier Briol, Jeremias Knoblauch*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/altamirano23a.html](https://proceedings.mlr.press/v202/altamirano23a.html)

        **Abstract**:

        This paper proposes an online, provably robust, and scalable Bayesian approach for changepoint detection. The resulting algorithm has key advantages over previous work: it provides provable robustness by leveraging the generalised Bayesian perspective, and also addresses the scalability issues of previous attempts. Specifically, the proposed generalised Bayesian formalism leads to conjugate posteriors whose parameters are available in closed form by leveraging diffusion score matching. The resulting algorithm is exact, can be updated through simple algebra, and is more than 10 times faster than its closest competitor.

        ----

        ## [29] Neural Wasserstein Gradient Flows for Discrepancies with Riesz Kernels

        **Authors**: *Fabian Altekrüger, Johannes Hertrich, Gabriele Steidl*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/altekruger23a.html](https://proceedings.mlr.press/v202/altekruger23a.html)

        **Abstract**:

        Wasserstein gradient flows of maximum mean discrepancy (MMD) functionals with non-smooth Riesz kernels show a rich structure as singular measures can become absolutely continuous ones and conversely. In this paper we contribute to the understanding of such flows. We propose to approximate the backward scheme of Jordan, Kinderlehrer and Otto for computing such Wasserstein gradient flows as well as a forward scheme for so-called Wasserstein steepest descent flows by neural networks (NNs). Since we cannot restrict ourselves to absolutely continuous measures, we have to deal with transport plans and velocity plans instead of usual transport maps and velocity fields. Indeed, we approximate the disintegration of both plans by generative NNs which are learned with respect to appropriate loss functions. In order to evaluate the quality of both neural schemes, we benchmark them on the interaction energy. Here we provide analytic formulas for Wasserstein schemes starting at a Dirac measure and show their convergence as the time step size tends to zero. Finally, we illustrate our neural MMD flows by numerical examples.

        ----

        ## [30] Distributed Contextual Linear Bandits with Minimax Optimal Communication Cost

        **Authors**: *Sanae Amani, Tor Lattimore, András György, Lin Yang*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/amani23a.html](https://proceedings.mlr.press/v202/amani23a.html)

        **Abstract**:

        We study distributed contextual linear bandits with stochastic contexts, where $N$ agents/learners act cooperatively to solve a linear bandit-optimization problem with $d$-dimensional features over the course of $T$ rounds. For this problem, we derive the first ever information-theoretic lower bound $\Omega(dN)$ on the communication cost of any algorithm that performs optimally in a regret minimization setup. We then propose a distributed batch elimination version of the LinUCB algorithm, DisBE-LUCB, where the agents share information among each other through a central server. We prove that the communication cost of DisBE-LUCB, matches our lower bound up to logarithmic factors. In particular, for scenarios with known context distribution, the communication cost of DisBE-LUCB is only $\tilde{\mathcal{O}}(dN)$ and its regret is $\tilde{\mathcal{O}}(\sqrt{dNT})$, which is of the same order as that incurred by an optimal single-agent algorithm for $NT$ rounds. We also provide similar bounds for practical settings where the context distribution can only be estimated. Therefore, our proposed algorithm is nearly minimax optimal in terms of both regret and communication cost. Finally, we propose DecBE-LUCB, a fully decentralized version of DisBE-LUCB, which operates without a central server, where agents share information with their immediate neighbors through a carefully designed consensus procedure.

        ----

        ## [31] A Kernelized Stein Discrepancy for Biological Sequences

        **Authors**: *Alan Nawzad Amin, Eli N. Weinstein, Debora Susan Marks*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/amin23a.html](https://proceedings.mlr.press/v202/amin23a.html)

        **Abstract**:

        Generative models of biological sequences are a powerful tool for learning from complex sequence data, predicting the effects of mutations, and designing novel biomolecules with desired properties. To evaluate generative models it is important to accurately measure differences between high-dimensional distributions. In this paper we propose the “KSD-B”, a novel divergence measure for distributions over biological sequences that is based on the kernelized Stein discrepancy (KSD). The KSD-B can be evaluated even when the normalizing constant of the model is unknown; it allows for variable length sequences and can take into account biological notions of sequence distance. Unlike previous KSDs over discrete spaces the KSD-B (a) is theoretically guaranteed to detect convergence and non-convergence of distributions over sequence space and (b) can be efficiently estimated in practice. We demonstrate the advantages of the KSD-B on problems with synthetic and real data, and apply it to measure the fit of state-of-the-art machine learning models. Overall, the KSD-B enables rigorous evaluation of generative biological sequence models, allowing the accuracy of models, sampling procedures, and library designs to be checked reliably.

        ----

        ## [32] The Optimal Approximation Factors in Misspecified Off-Policy Value Function Estimation

        **Authors**: *Philip Amortila, Nan Jiang, Csaba Szepesvári*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/amortila23a.html](https://proceedings.mlr.press/v202/amortila23a.html)

        **Abstract**:

        Theoretical guarantees in reinforcement learning (RL) are known to suffer multiplicative blow-up factors with respect to the misspecification error of function approximation. Yet, the nature of such approximation factors—especially their optimal form in a given learning problem—is poorly understood. In this paper we study this question in linear off-policy value function estimation, where many open questions remain. We study the approximation factor in a broad spectrum of settings, such as presence vs. absence of state aliasing and full vs. partial coverage of the state space. Our core results include instance-dependent upper bounds on the approximation factors with respect to both the weighted $L_2$-norm (where the weighting is the offline state distribution) and the $L_\infty$ norm. We show that these approximation factors are optimal (in an instance-dependent sense) for a number of these settings. In other cases, we show that the instance-dependent parameters which appear in the upper bounds are necessary, and that the finiteness of either alone cannot guarantee a finite approximation factor even in the limit of infinite data.

        ----

        ## [33] Meta Optimal Transport

        **Authors**: *Brandon Amos, Giulia Luise, Samuel Cohen, Ievgen Redko*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/amos23a.html](https://proceedings.mlr.press/v202/amos23a.html)

        **Abstract**:

        We study the use of amortized optimization to predict optimal transport (OT) maps from the input measures, which we call Meta OT. This helps repeatedly solve similar OT problems between different measures by leveraging the knowledge and information present from past problems to rapidly predict and solve new problems. Otherwise, standard methods ignore the knowledge of the past solutions and suboptimally re-solve each problem from scratch. We instantiate Meta OT models in discrete and continuous settings between grayscale images, spherical data, classification labels, and color palettes and use them to improve the computational time of standard OT solvers. Our source code is available at http://github.com/facebookresearch/meta-ot

        ----

        ## [34] Near-Optimal Φ-Regret Learning in Extensive-Form Games

        **Authors**: *Ioannis Anagnostides, Gabriele Farina, Tuomas Sandholm*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/anagnostides23a.html](https://proceedings.mlr.press/v202/anagnostides23a.html)

        **Abstract**:

        In this paper, we establish efficient and uncoupled learning dynamics so that, when employed by all players in multiplayer perfect-recall imperfect-information extensive-form games, the trigger regret of each player grows as $O(\log T)$ after $T$ repetitions of play. This improves exponentially over the prior best known trigger-regret bound of $O(T^{1/4})$, and settles a recent open question by Bai et al. (2022). As an immediate consequence, we guarantee convergence to the set of extensive-form correlated equilibria and coarse correlated equilibria at a near-optimal rate of $\frac{\log T}{T}$. Building on prior work, at the heart of our construction lies a more general result regarding fixed points deriving from rational functions with polynomial degree, a property that we establish for the fixed points of (coarse) trigger deviation functions. Moreover, our construction leverages a refined regret circuit for the convex hull, which—unlike prior guarantees—preserves the RVU property introduced by Syrgkanis et al. (NIPS, 2015); this observation has an independent interest in establishing near-optimal regret under learning dynamics based on a CFR-type decomposition of the regret.

        ----

        ## [35] A Modern Look at the Relationship between Sharpness and Generalization

        **Authors**: *Maksym Andriushchenko, Francesco Croce, Maximilian Müller, Matthias Hein, Nicolas Flammarion*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/andriushchenko23a.html](https://proceedings.mlr.press/v202/andriushchenko23a.html)

        **Abstract**:

        Sharpness of minima is a promising quantity that can correlate with generalization in deep networks and, when optimized during training, can improve generalization. However, standard sharpness is not invariant under reparametrizations of neural networks, and, to fix this, reparametrization-invariant sharpness definitions have been proposed, most prominently adaptive sharpness (Kwon et al., 2021). But does it really capture generalization in modern practical settings? We comprehensively explore this question in a detailed study of various definitions of adaptive sharpness in settings ranging from training from scratch on ImageNet and CIFAR-10 to fine-tuning CLIP on ImageNet and BERT on MNLI. We focus mostly on transformers for which little is known in terms of sharpness despite their widespread usage. Overall, we observe that sharpness does not correlate well with generalization but rather with some training parameters like the learning rate that can be positively or negatively correlated with generalization depending on the setup. Interestingly, in multiple cases, we observe a consistent negative correlation of sharpness with OOD generalization implying that sharper minima can generalize better. Finally, we illustrate on a simple model that the right sharpness measure is highly data-dependent, and that we do not understand well this aspect for realistic data distributions.

        ----

        ## [36] SGD with Large Step Sizes Learns Sparse Features

        **Authors**: *Maksym Andriushchenko, Aditya Vardhan Varre, Loucas Pillaud-Vivien, Nicolas Flammarion*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/andriushchenko23b.html](https://proceedings.mlr.press/v202/andriushchenko23b.html)

        **Abstract**:

        We showcase important features of the dynamics of the Stochastic Gradient Descent (SGD) in the training of neural networks. We present empirical observations that commonly used large step sizes (i) may lead the iterates to jump from one side of a valley to the other causing loss stabilization, and (ii) this stabilization induces a hidden stochastic dynamics that biases it implicitly toward simple predictors. Furthermore, we show empirically that the longer large step sizes keep SGD high in the loss landscape valleys, the better the implicit regularization can operate and find sparse representations. Notably, no explicit regularization is used: the regularization effect comes solely from the SGD dynamics influenced by the large step sizes schedule. Therefore, these observations unveil how, through the step size schedules, both gradient and noise drive together the SGD dynamics through the loss landscape of neural networks. We justify these findings theoretically through the study of simple neural network models as well as qualitative arguments inspired from stochastic processes. This analysis allows us to shed new light on some common practices and observed phenomena when training deep networks.

        ----

        ## [37] Neural Continuous-Discrete State Space Models for Irregularly-Sampled Time Series

        **Authors**: *Abdul Fatir Ansari, Alvin Heng, Andre Lim, Harold Soh*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/ansari23a.html](https://proceedings.mlr.press/v202/ansari23a.html)

        **Abstract**:

        Learning accurate predictive models of real-world dynamic phenomena (e.g., climate, biological) remains a challenging task. One key issue is that the data generated by both natural and artificial processes often comprise time series that are irregularly sampled and/or contain missing observations. In this work, we propose the Neural Continuous-Discrete State Space Model (NCDSSM) for continuous-time modeling of time series through discrete-time observations. NCDSSM employs auxiliary variables to disentangle recognition from dynamics, thus requiring amortized inference only for the auxiliary variables. Leveraging techniques from continuous-discrete filtering theory, we demonstrate how to perform accurate Bayesian inference for the dynamic states. We propose three flexible parameterizations of the latent dynamics and an efficient training objective that marginalizes the dynamic states during inference. Empirical results on multiple benchmark datasets across various domains show improved imputation and forecasting performance of NCDSSM over existing models.

        ----

        ## [38] Paging with Succinct Predictions

        **Authors**: *Antonios Antoniadis, Joan Boyar, Marek Eliás, Lene Monrad Favrholdt, Ruben Hoeksma, Kim S. Larsen, Adam Polak, Bertrand Simon*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/antoniadis23a.html](https://proceedings.mlr.press/v202/antoniadis23a.html)

        **Abstract**:

        Paging is a prototypical problem in the area of online algorithms. It has also played a central role in the development of learning-augmented algorithms. Previous work on learning-augmented paging has investigated predictions on (i) when the current page will be requested again (reoccurrence predictions), (ii) the current state of the cache in an optimal algorithm (state predictions), (iii) all requests until the current page gets requested again, and (iv) the relative order in which pages are requested. We study learning-augmented paging from the new perspective of requiring the least possible amount of predicted information. More specifically, the predictions obtained alongside each page request are limited to one bit only. We develop algorithms satisfy all three desirable properties of learning-augmented algorithms – that is, they are consistent, robust and smooth – despite being limited to a one-bit prediction per request. We also present lower bounds establishing that our algorithms are essentially best possible.

        ----

        ## [39] Mixing Predictions for Online Metric Algorithms

        **Authors**: *Antonios Antoniadis, Christian Coester, Marek Eliás, Adam Polak, Bertrand Simon*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/antoniadis23b.html](https://proceedings.mlr.press/v202/antoniadis23b.html)

        **Abstract**:

        A major technique in learning-augmented online algorithms is combining multiple algorithms or predictors. Since the performance of each predictor may vary over time, it is desirable to use not the single best predictor as a benchmark, but rather a dynamic combination which follows different predictors at different times. We design algorithms that combine predictions and are competitive against such dynamic combinations for a wide class of online problems, namely, metrical task systems. Against the best (in hindsight) unconstrained combination of $\ell$ predictors, we obtain a competitive ratio of $O(\ell^2)$, and show that this is best possible. However, for a benchmark with slightly constrained number of switches between different predictors, we can get a $(1+\epsilon)$-competitive algorithm. Moreover, our algorithms can be adapted to access predictors in a bandit-like fashion, querying only one predictor at a time. An unexpected implication of one of our lower bounds is a new structural insight about covering formulations for the $k$-server problem.

        ----

        ## [40] Exponential Smoothing for Off-Policy Learning

        **Authors**: *Imad Aouali, Victor-Emmanuel Brunel, David Rohde, Anna Korba*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/aouali23a.html](https://proceedings.mlr.press/v202/aouali23a.html)

        **Abstract**:

        Off-policy learning (OPL) aims at finding improved policies from logged bandit data, often by minimizing the inverse propensity scoring (IPS) estimator of the risk. In this work, we investigate a smooth regularization for IPS, for which we derive a two-sided PAC-Bayes generalization bound. The bound is tractable, scalable, interpretable and provides learning certificates. In particular, it is also valid for standard IPS without making the assumption that the importance weights are bounded. We demonstrate the relevance of our approach and its favorable performance through a set of learning tasks. Since our bound holds for standard IPS, we are able to provide insight into when regularizing IPS is useful. Namely, we identify cases where regularization might not be needed. This goes against the belief that, in practice, clipped IPS often enjoys favorable performance than standard IPS in OPL.

        ----

        ## [41] Polynomial Time and Private Learning of Unbounded Gaussian Mixture Models

        **Authors**: *Jamil Arbas, Hassan Ashtiani, Christopher Liaw*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/arbas23a.html](https://proceedings.mlr.press/v202/arbas23a.html)

        **Abstract**:

        We study the problem of privately estimating the parameters of $d$-dimensional Gaussian Mixture Models (GMMs) with $k$ components. For this, we develop a technique to reduce the problem to its non-private counterpart. This allows us to privatize existing non-private algorithms in a blackbox manner, while incurring only a small overhead in the sample complexity and running time. As the main application of our framework, we develop an $(\varepsilon, \delta)$-differentially private algorithm to learn GMMs using the non-private algorithm of Moitra and Valiant (2010) as a blackbox. Consequently, this gives the first sample complexity upper bound and first polynomial time algorithm for privately learning GMMs without any boundedness assumptions on the parameters. As part of our analysis, we prove a tight (up to a constant factor) lower bound on the total variation distance of high-dimensional Gaussians which can be of independent interest.

        ----

        ## [42] Principled Acceleration of Iterative Numerical Methods Using Machine Learning

        **Authors**: *Sohei Arisaka, Qianxiao Li*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/arisaka23a.html](https://proceedings.mlr.press/v202/arisaka23a.html)

        **Abstract**:

        Iterative methods are ubiquitous in large-scale scientific computing applications, and a number of approaches based on meta-learning have been recently proposed to accelerate them. However, a systematic study of these approaches and how they differ from meta-learning is lacking. In this paper, we propose a framework to analyze such learning-based acceleration approaches, where one can immediately identify a departure from classical meta-learning. We theoretically show that this departure may lead to arbitrary deterioration of model performance, and at the same time, we identify a methodology to ameliorate it by modifying the loss objective, leading to a novel training method for learning-based acceleration of iterative algorithms. We demonstrate the significant advantage and versatility of the proposed approach through various numerical applications.

        ----

        ## [43] Faster Rates of Convergence to Stationary Points in Differentially Private Optimization

        **Authors**: *Raman Arora, Raef Bassily, Tomás González, Cristóbal Guzmán, Michael Menart, Enayat Ullah*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/arora23a.html](https://proceedings.mlr.press/v202/arora23a.html)

        **Abstract**:

        We study the problem of approximating stationary points of Lipschitz and smooth functions under $(\varepsilon,\delta)$-differential privacy (DP) in both the finite-sum and stochastic settings. A point $\widehat{w}$ is called an $\alpha$-stationary point of a function $F:\mathbb{R}^d\rightarrow\mathbb{R}$ if $\|\nabla F(\widehat{w})\|\leq \alpha$. We give a new construction that improves over the existing rates in the stochastic optimization setting, where the goal is to find approximate stationary points of the population risk given $n$ samples. Our construction finds a $\tilde{O}\big(\frac{1}{n^{1/3}} + \big[\frac{\sqrt{d}}{n\varepsilon}\big]^{1/2}\big)$-stationary point of the population risk in time linear in $n$. We also provide an efficient algorithm that finds an $\tilde{O}\big(\big[\frac{\sqrt{d}}{n\varepsilon}\big]^{2/3}\big)$-stationary point in the finite-sum setting. This improves on the previous best rate of $\tilde{O}\big(\big[\frac{\sqrt{d}}{n\varepsilon}\big]^{1/2}\big)$. Furthermore, under the additional assumption of convexity, we completely characterize the sample complexity of finding stationary points of the population risk (up to polylog factors) and show that the optimal rate on population stationarity is $\tilde \Theta\big(\frac{1}{\sqrt{n}}+\frac{\sqrt{d}}{n\varepsilon}\big)$. Finally, we show that our methods can be used to provide dimension-independent rates of $O\big(\frac{1}{\sqrt{n}}+\min\big(\big[\frac{\sqrt{rank}}{n\varepsilon}\big]^{2/3},\frac{1}{(n\varepsilon)^{2/5}}\big)\big)$ on population stationarity for Generalized Linear Models (GLM), where $rank$ is the rank of the design matrix, which improves upon the previous best known rate.

        ----

        ## [44] Prototype-Sample Relation Distillation: Towards Replay-Free Continual Learning

        **Authors**: *Nader Asadi, MohammadReza Davari, Sudhir Mudur, Rahaf Aljundi, Eugene Belilovsky*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/asadi23a.html](https://proceedings.mlr.press/v202/asadi23a.html)

        **Abstract**:

        In Continual learning (CL) balancing effective adaptation while combating catastrophic forgetting is a central challenge. Many of the recent best-performing methods utilize various forms of prior task data, e.g. a replay buffer, to tackle the catastrophic forgetting problem. Having access to previous task data can be restrictive in many real-world scenarios, for example when task data is sensitive or proprietary. To overcome the necessity of using previous tasks’ data, in this work, we start with strong representation learning methods that have been shown to be less prone to forgetting. We propose a holistic approach to jointly learn the representation and class prototypes while maintaining the relevance of old class prototypes and their embedded similarities. Specifically, samples are mapped to an embedding space where the representations are learned using a supervised contrastive loss. Class prototypes are evolved continually in the same latent space, enabling learning and prediction at any point. To continually adapt the prototypes without keeping any prior task data, we propose a novel distillation loss that constrains class prototypes to maintain relative similarities as compared to new task data. This method yields state-of-the-art performance in the task-incremental setting, outperforming methods relying on large amounts of data, and provides strong performance in the class-incremental setting without using any stored data points.

        ----

        ## [45] Near-Optimal Algorithms for Private Online Optimization in the Realizable Regime

        **Authors**: *Hilal Asi, Vitaly Feldman, Tomer Koren, Kunal Talwar*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/asi23a.html](https://proceedings.mlr.press/v202/asi23a.html)

        **Abstract**:

        We consider online learning problems in the realizable setting, where there is a zero-loss solution, and propose new Differentially Private (DP) algorithms that obtain near-optimal regret bounds. For the problem of online prediction from experts, we design new algorithms that obtain near-optimal regret $O \big( \varepsilon^{-1} \mathsf{poly}(\log{d}) \big)$ where $d$ is the number of experts. This significantly improves over the best existing regret bounds for the DP non-realizable setting which are $O \big( \varepsilon^{-1} \min\big\{d, \sqrt{T\log d}\big\} \big)$. We also develop an adaptive algorithm for the small-loss setting with regret $(L^\star+ \varepsilon^{-1}) \cdot O(\mathsf{poly}(\log{d}))$ where $L^\star$ is the total loss of the best expert. Additionally, we consider DP online convex optimization in the realizable setting and propose an algorithm with near-optimal regret $O \big(\varepsilon^{-1} \mathsf{poly}(d) \big)$, as well as an algorithm for the smooth case with regret $O \big( (\sqrt{Td}/\varepsilon)^{2/3} \big)$, both significantly improving over existing bounds in the non-realizable regime.

        ----

        ## [46] From Robustness to Privacy and Back

        **Authors**: *Hilal Asi, Jonathan R. Ullman, Lydia Zakynthinou*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/asi23b.html](https://proceedings.mlr.press/v202/asi23b.html)

        **Abstract**:

        We study the relationship between two desiderata of algorithms in statistical inference and machine learning—differential privacy and robustness to adversarial data corruptions. Their conceptual similarity was first observed by Dwork and Lei (STOC 2009), who observed that private algorithms satisfy robustness, and gave a general method for converting robust algorithms to private ones. However, all general methods for transforming robust algorithms into private ones lead to suboptimal error rates. Our work gives the first black-box transformation that converts any adversarially robust algorithm into one that satisfies pure differential privacy. Moreover, we show that for any low-dimensional estimation task, applying our transformation to an optimal robust estimator results in an optimal private estimator. Thus, we conclude that for any low-dimensional task, the optimal error rate for $\varepsilon$-differentially private estimators is essentially the same as the optimal error rate for estimators that are robust to adversarially corrupting $1/\varepsilon$ training samples. We apply our transformation to obtain new optimal private estimators for several high-dimensional statistical tasks, including Gaussian linear regression and PCA. Finally, we present an extension of our transformation that leads to approximately differentially private algorithms whose error does not depend on the range of the output space, which is impossible under pure differential privacy.

        ----

        ## [47] SGD with AdaGrad Stepsizes: Full Adaptivity with High Probability to Unknown Parameters, Unbounded Gradients and Affine Variance

        **Authors**: *Amit Attia, Tomer Koren*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/attia23a.html](https://proceedings.mlr.press/v202/attia23a.html)

        **Abstract**:

        We study Stochastic Gradient Descent with AdaGrad stepsizes: a popular adaptive (self-tuning) method for first-order stochastic optimization. Despite being well studied, existing analyses of this method suffer from various shortcomings: they either assume some knowledge of the problem parameters, impose strong global Lipschitz conditions, or fail to give bounds that hold with high probability. We provide a comprehensive analysis of this basic method without any of these limitations, in both the convex and non-convex (smooth) cases, that additionally supports a general “affine variance” noise model and provides sharp rates of convergence in both the low-noise and high-noise regimes.

        ----

        ## [48] Adversarially Robust PAC Learnability of Real-Valued Functions

        **Authors**: *Idan Attias, Steve Hanneke*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/attias23a.html](https://proceedings.mlr.press/v202/attias23a.html)

        **Abstract**:

        We study robustness to test-time adversarial attacks in the regression setting with $\ell_p$ losses and arbitrary perturbation sets. We address the question of which function classes are PAC learnable in this setting. We show that classes of finite fat-shattering dimension are learnable in both the realizable and agnostic settings. Moreover, for convex function classes, they are even properly learnable. In contrast, some non-convex function classes provably require improper learning algorithms. Our main technique is based on a construction of an adversarially robust sample compression scheme of a size determined by the fat-shattering dimension. Along the way, we introduce a novel agnostic sample compression scheme for real-valued functions, which may be of independent interest.

        ----

        ## [49] Infusing Lattice Symmetry Priors in Attention Mechanisms for Sample-Efficient Abstract Geometric Reasoning

        **Authors**: *Mattia Atzeni, Mrinmaya Sachan, Andreas Loukas*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/atzeni23a.html](https://proceedings.mlr.press/v202/atzeni23a.html)

        **Abstract**:

        The Abstraction and Reasoning Corpus (ARC) (Chollet, 2019) and its most recent language-complete instantiation (LARC) has been postulated as an important step towards general AI. Yet, even state-of-the-art machine learning models struggle to achieve meaningful performance on these problems, falling behind non-learning based approaches. We argue that solving these tasks requires extreme generalization that can only be achieved by proper accounting for core knowledge priors. As a step towards this goal, we focus on geometry priors and introduce LatFormer, a model that incorporates lattice symmetry priors in attention masks. We show that, for any transformation of the hypercubic lattice, there exists a binary attention mask that implements that group action. Hence, our study motivates a modification to the standard attention mechanism, where attention weights are scaled using soft masks generated by a convolutional network. Experiments on synthetic geometric reasoning show that LatFormer requires 2 orders of magnitude fewer data than standard attention and transformers. Moreover, our results on ARC and LARC tasks that incorporate geometric priors provide preliminary evidence that these complex datasets do not lie out of the reach of deep learning models.

        ----

        ## [50] Learning to Initiate and Reason in Event-Driven Cascading Processes

        **Authors**: *Yuval Atzmon, Eli A. Meirom, Shie Mannor, Gal Chechik*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/atzmon23a.html](https://proceedings.mlr.press/v202/atzmon23a.html)

        **Abstract**:

        Training agents to control a dynamic environment is a fundamental task in AI. In many environments, the dynamics can be summarized by a small set of events that capture the semantic behavior of the system. Typically, these events form chains or cascades. We often wish to change the system behavior using a single intervention that propagates through the cascade. For instance, one may trigger a biochemical cascade to switch the state of a cell or, in logistics, reroute a truck to meet an unexpected, urgent delivery. We introduce a new supervised learning setup called Cascade. An agent observes a system with known dynamics evolving from some initial state. The agent is given a structured semantic instruction and needs to make an intervention that triggers a cascade of events, such that the system reaches an alternative (counterfactual) behavior. We provide a test-bed for this problem, consisting of physical objects. We combine semantic tree search with an event-driven forward model and devise an algorithm that learns to efficiently search in exponentially large semantic trees. We demonstrate that our approach learns to follow instructions to intervene in new complex scenes. When provided with an observed cascade of events, it can also reason about alternative outcomes.

        ----

        ## [51] On the convergence of the MLE as an estimator of the learning rate in the Exp3 algorithm

        **Authors**: *Julien Aubert, Luc Lehéricy, Patricia Reynaud-Bouret*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/aubert23a.html](https://proceedings.mlr.press/v202/aubert23a.html)

        **Abstract**:

        When fitting the learning data of an individual to algorithm-like learning models, the observations are so dependent and non-stationary that one may wonder what the classical Maximum Likelihood Estimator (MLE) could do, even if it is the usual tool applied to experimental cognition. Our objective in this work is to show that the estimation of the learning rate cannot be efficient if the learning rate is constant in the classical Exp3 (Exponential weights for Exploration and Exploitation) algorithm. Secondly, we show that if the learning rate decreases polynomially with the sample size, then the prediction error and in some cases the estimation error of the MLE satisfy bounds in probability that decrease at a polynomial rate.

        ----

        ## [52] Dirichlet Diffusion Score Model for Biological Sequence Generation

        **Authors**: *Pavel Avdeyev, Chenlai Shi, Yuhao Tan, Kseniia Dudnyk, Jian Zhou*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/avdeyev23a.html](https://proceedings.mlr.press/v202/avdeyev23a.html)

        **Abstract**:

        Designing biological sequences is an important challenge that requires satisfying complex constraints and thus is a natural problem to address with deep generative modeling. Diffusion generative models have achieved considerable success in many applications. Score-based generative stochastic differential equations (SDE) model is a continuous-time diffusion model framework that enjoys many benefits, but the originally proposed SDEs are not naturally designed for modeling discrete data. To develop generative SDE models for discrete data such as biological sequences, here we introduce a diffusion process defined in the probability simplex space with stationary distribution being the Dirichlet distribution. This makes diffusion in continuous space natural for modeling discrete data. We refer to this approach as Dirchlet diffusion score model. We demonstrate that this technique can generate samples that satisfy hard constraints using a Sudoku generation task. This generative model can also solve Sudoku, including hard puzzles, without additional training. Finally, we applied this approach to develop the first human promoter DNA sequence design model and showed that designed sequences share similar properties with natural promoter sequences.

        ----

        ## [53] Gradient Descent Converges Linearly for Logistic Regression on Separable Data

        **Authors**: *Kyriakos Axiotis, Maxim Sviridenko*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/axiotis23a.html](https://proceedings.mlr.press/v202/axiotis23a.html)

        **Abstract**:

        We show that running gradient descent with variable learning rate guarantees loss $f(x) ≤ 1.1 \cdot f(x^*)+\epsilon$ for the logistic regression objective, where the error $\epsilon$ decays exponentially with the number of iterations and polynomially with the magnitude of the entries of an arbitrary fixed solution $x$. This is in contrast to the common intuition that the absence of strong convexity precludes linear convergence of first-order methods, and highlights the importance of variable learning rates for gradient descent. We also apply our ideas to sparse logistic regression, where they lead to an exponential improvement of the sparsity-error tradeoff.

        ----

        ## [54] Naive imputation implicitly regularizes high-dimensional linear models

        **Authors**: *Alexis Ayme, Claire Boyer, Aymeric Dieuleveut, Erwan Scornet*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/ayme23a.html](https://proceedings.mlr.press/v202/ayme23a.html)

        **Abstract**:

        Two different approaches exist to handle missing values for prediction: either imputation, prior to fitting any predictive algorithms, or dedicated methods able to natively incorporate missing values. While imputation is widely (and easily) use, it is unfortunately biased when low-capacity predictors (such as linear models) are applied afterward. However, in practice, naive imputation exhibits good predictive performance. In this paper, we study the impact of imputation in a high-dimensional linear model with MCAR missing data. We prove that zero imputation performs an implicit regularization closely related to the ridge method, often used in high-dimensional problems. Leveraging on this connection, we establish that the imputation bias is controlled by a ridge bias, which vanishes in high dimension. As a predictor, we argue in favor of the averaged SGD strategy, applied to zero-imputed data. We establish an upper bound on its generalization error, highlighting that imputation is benign in the $d \gg \sqrt{n}$ regime. Experiments illustrate our findings.

        ----

        ## [55] Half-Hop: A graph upsampling approach for slowing down message passing

        **Authors**: *Mehdi Azabou, Venkataramana Ganesh, Shantanu Thakoor, Chi-Heng Lin, Lakshmi Sathidevi, Ran Liu, Michal Valko, Petar Velickovic, Eva L. Dyer*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/azabou23a.html](https://proceedings.mlr.press/v202/azabou23a.html)

        **Abstract**:

        Message passing neural networks have shown a lot of success on graph-structured data. However, there are many instances where message passing can lead to over-smoothing or fail when neighboring nodes belong to different classes. In this work, we introduce a simple yet general framework for improving learning in message passing neural networks. Our approach essentially upsamples edges in the original graph by adding "slow nodes" at each edge that can mediate communication between a source and a target node. Our method only modifies the input graph, making it plug-and-play and easy to use with existing models. To understand the benefits of slowing down message passing, we provide theoretical and empirical analyses. We report results on several supervised and self-supervised benchmarks, and show improvements across the board, notably in heterophilic conditions where adjacent nodes are more likely to have different labels. Finally, we show how our approach can be used to generate augmentations for self-supervised learning, where slow nodes are randomly introduced into different edges in the graph to generate multi-scale views with variable path lengths.

        ----

        ## [56] CLUTR: Curriculum Learning via Unsupervised Task Representation Learning

        **Authors**: *Abdus Salam Azad, Izzeddin Gur, Jasper Emhoff, Nathaniel Alexis, Aleksandra Faust, Pieter Abbeel, Ion Stoica*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/azad23a.html](https://proceedings.mlr.press/v202/azad23a.html)

        **Abstract**:

        Reinforcement Learning (RL) algorithms are often known for sample inefficiency and difficult generalization. Recently, Unsupervised Environment Design (UED) emerged as a new paradigm for zero-shot generalization by simultaneously learning a task distribution and agent policies on the generated tasks. This is a non-stationary process where the task distribution evolves along with agent policies; creating an instability over time. While past works demonstrated the potential of such approaches, sampling effectively from the task space remains an open challenge, bottlenecking these approaches. To this end, we introduce CLUTR: a novel unsupervised curriculum learning algorithm that decouples task representation and curriculum learning into a two-stage optimization. It first trains a recurrent variational autoencoder on randomly generated tasks to learn a latent task manifold. Next, a teacher agent creates a curriculum by maximizing a minimax REGRET-based objective on a set of latent tasks sampled from this manifold. Using the fixed-pretrained task manifold, we show that CLUTR successfully overcomes the non-stationarity problem and improves stability. Our experimental results show CLUTR outperforms PAIRED, a principled and popular UED method, in the challenging CarRacing and navigation environments: achieving 10.6X and 45% improvement in zero-shot generalization, respectively. CLUTR also performs comparably to the non-UED state-of-the-art for CarRacing, while requiring 500X fewer environment interactions. We open source our code at https://github.com/clutr/clutr.

        ----

        ## [57] Personalized Subgraph Federated Learning

        **Authors**: *Jinheon Baek, Wonyong Jeong, Jiongdao Jin, Jaehong Yoon, Sung Ju Hwang*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/baek23a.html](https://proceedings.mlr.press/v202/baek23a.html)

        **Abstract**:

        Subgraphs of a larger global graph may be distributed across multiple devices, and only locally accessible due to privacy restrictions, although there may be links between subgraphs. Recently proposed subgraph Federated Learning (FL) methods deal with those missing links across local subgraphs while distributively training Graph Neural Networks (GNNs) on them. However, they have overlooked the inevitable heterogeneity between subgraphs comprising different communities of a global graph, consequently collapsing the incompatible knowledge from local GNN models. To this end, we introduce a new subgraph FL problem, personalized subgraph FL, which focuses on the joint improvement of the interrelated local GNNs rather than learning a single global model, and propose a novel framework, FEDerated Personalized sUBgraph learning (FED-PUB), to tackle it. Since the server cannot access the subgraph in each client, FED-PUB utilizes functional embeddings of the local GNNs using random graphs as inputs to compute similarities between them, and use the similarities to perform weighted averaging for server-side aggregation. Further, it learns a personalized sparse mask at each client to select and update only the subgraph-relevant subset of the aggregated parameters. We validate our FED-PUB for its subgraph FL performance on six datasets, considering both non-overlapping and overlapping subgraphs, on which it significantly outperforms relevant baselines. Our code is available at https://github.com/JinheonBaek/FED-PUB.

        ----

        ## [58] Efficient Self-supervised Learning with Contextualized Target Representations for Vision, Speech and Language

        **Authors**: *Alexei Baevski, Arun Babu, Wei-Ning Hsu, Michael Auli*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/baevski23a.html](https://proceedings.mlr.press/v202/baevski23a.html)

        **Abstract**:

        Current self-supervised learning algorithms are often modality-specific and require large amounts of computational resources. To address these issues, we increase the training efficiency of data2vec, a learning objective that generalizes across several modalities. We do not encode masked tokens, use a fast convolutional decoder and amortize the effort to build teacher representations. data2vec 2.0 benefits from the rich contextualized target representations introduced in data2vec which enable a fast self-supervised learner. Experiments on ImageNet-1K image classification show that data2vec 2.0 matches the accuracy of Masked Autoencoders in 16.4x lower pre-training time, on Librispeech speech recognition it performs as well as wav2vec 2.0 in 10.6x less time, and on GLUE natural language understanding it matches a retrained RoBERTa model in half the time. Trading some speed for accuracy results in ImageNet-1K top-1 accuracy of 86.8% with a ViT-L model trained for 150 epochs.

        ----

        ## [59] Efficient preconditioned stochastic gradient descent for estimation in latent variable models

        **Authors**: *Charlotte Baey, Maud Delattre, Estelle Kuhn, Jean-Benoist Leger, Sarah Lemler*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/baey23a.html](https://proceedings.mlr.press/v202/baey23a.html)

        **Abstract**:

        Latent variable models are powerful tools for modeling complex phenomena involving in particular partially observed data, unobserved variables or underlying complex unknown structures. Inference is often difficult due to the latent structure of the model. To deal with parameter estimation in the presence of latent variables, well-known efficient methods exist, such as gradient-based and EM-type algorithms, but with practical and theoretical limitations. In this paper, we propose as an alternative for parameter estimation an efficient preconditioned stochastic gradient algorithm. Our method includes a preconditioning step based on a positive definite Fisher information matrix estimate. We prove convergence results for the proposed algorithm under mild assumptions for very general latent variables models. We illustrate through relevant simulations the performance of the proposed methodology in a nonlinear mixed effects model and in a stochastic block model.

        ----

        ## [60] Feed Two Birds with One Scone: Exploiting Wild Data for Both Out-of-Distribution Generalization and Detection

        **Authors**: *Haoyue Bai, Gregory Canal, Xuefeng Du, Jeongyeol Kwon, Robert D. Nowak, Yixuan Li*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/bai23a.html](https://proceedings.mlr.press/v202/bai23a.html)

        **Abstract**:

        Modern machine learning models deployed in the wild can encounter both covariate and semantic shifts, giving rise to the problems of out-of-distribution (OOD) generalization and OOD detection respectively. While both problems have received significant research attention lately, they have been pursued independently. This may not be surprising, since the two tasks have seemingly conflicting goals. This paper provides a new unified approach that is capable of simultaneously generalizing to covariate shifts while robustly detecting semantic shifts. We propose a margin-based learning framework that exploits freely available unlabeled data in the wild that captures the environmental test-time OOD distributions under both covariate and semantic shifts. We show both empirically and theoretically that the proposed margin constraint is the key to achieving both OOD generalization and detection. Extensive experiments show the superiority of our framework, outperforming competitive baselines that specialize in either OOD generalization or OOD detection. Code is publicly available at https://github.com/deeplearning-wisc/scone.

        ----

        ## [61] Answering Complex Logical Queries on Knowledge Graphs via Query Computation Tree Optimization

        **Authors**: *Yushi Bai, Xin Lv, Juanzi Li, Lei Hou*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/bai23b.html](https://proceedings.mlr.press/v202/bai23b.html)

        **Abstract**:

        Answering complex logical queries on incomplete knowledge graphs is a challenging task, and has been widely studied. Embedding-based methods require training on complex queries and may not generalize well to out-of-distribution query structures. Recent work frames this task as an end-to-end optimization problem, and it only requires a pretrained link predictor. However, due to the exponentially large combinatorial search space, the optimal solution can only be approximated, limiting the final accuracy. In this work, we propose QTO (Query Computation Tree Optimization) that can efficiently find the exact optimal solution. QTO finds the optimal solution by a forward-backward propagation on the tree-like computation graph, i.e., query computation tree. In particular, QTO utilizes the independence encoded in the query computation tree to reduce the search space, where only local computations are involved during the optimization procedure. Experiments on 3 datasets show that QTO obtains state-of-the-art performance on complex query answering, outperforming previous best results by an average of 22%. Moreover, QTO can interpret the intermediate solutions for each of the one-hop atoms in the query with over 90% accuracy.

        ----

        ## [62] Linear optimal partial transport embedding

        **Authors**: *Yikun Bai, Ivan Vladimir Medri, Rocio Diaz Martin, Rana Muhammad Shahroz Khan, Soheil Kolouri*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/bai23c.html](https://proceedings.mlr.press/v202/bai23c.html)

        **Abstract**:

        Optimal transport (OT) has gained popularity due to its various applications in fields such as machine learning, statistics, and signal processing. However, the balanced mass requirement limits its performance in practical problems. To address these limitations, variants of the OT problem, including unbalanced OT, Optimal partial transport (OPT), and Hellinger Kantorovich (HK), have been proposed. In this paper, we propose the Linear optimal partial transport (LOPT) embedding, which extends the (local) linearization technique on OT and HK to the OPT problem. The proposed embedding allows for faster computation of OPT distance between pairs of positive measures. Besides our theoretical contributions, we demonstrate the LOPT embedding technique in point-cloud interpolation and PCA analysis. Our code is available at https://github.com/Baio0/LinearOPT.

        ----

        ## [63] Implicit Graph Neural Networks: A Monotone Operator Viewpoint

        **Authors**: *Justin Baker, Qingsong Wang, Cory D. Hauck, Bao Wang*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/baker23a.html](https://proceedings.mlr.press/v202/baker23a.html)

        **Abstract**:

        Implicit graph neural networks (IGNNs) – that solve a fixed-point equilibrium equation using Picard iteration for representation learning – have shown remarkable performance in learning long-range dependencies (LRD) in the underlying graphs. However, IGNNs suffer from several issues, including 1) their expressivity is limited by their parameterizations for the well-posedness guarantee, 2) IGNNs are unstable in learning LRD, and 3) IGNNs become computationally inefficient when learning LRD. In this paper, we provide a new well-posedness characterization for IGNNs leveraging monotone operator theory, resulting in a much more expressive parameterization than the existing one. We also propose an orthogonal parameterization for IGNN based on Cayley transform to stabilize learning LRD. Furthermore, we leverage Anderson-accelerated operator splitting schemes to efficiently solve for the fixed point of the equilibrium equation of IGNN with monotone or orthogonal parameterization. We verify the computational efficiency and accuracy of the new models over existing IGNNs on various graph learning tasks at both graph and node levels.

        ----

        ## [64] Tensor Decompositions Meet Control Theory: Learning General Mixtures of Linear Dynamical Systems

        **Authors**: *Ainesh Bakshi, Allen Liu, Ankur Moitra, Morris Yau*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/bakshi23a.html](https://proceedings.mlr.press/v202/bakshi23a.html)

        **Abstract**:

        Recently Chen and Poor initiated the study of learning mixtures of linear dynamical systems. While linear dynamical systems already have wide-ranging applications in modeling time-series data, using mixture models can lead to a better fit or even a richer understanding of underlying subpopulations represented in the data. In this work we give a new approach to learning mixtures of linear dynamical systems that is based on tensor decompositions. As a result, our algorithm succeeds without strong separation conditions on the components, and can be used to compete with the Bayes optimal clustering of the trajectories. Moreover our algorithm works in the challenging partially-observed setting. Our starting point is the simple but powerful observation that the classic Ho-Kalman algorithm is a relative of modern tensor decomposition methods for learning latent variable models. This gives us a playbook for how to extend it to work with more complicated generative models.

        ----

        ## [65] Block Subsampled Randomized Hadamard Transform for Nyström Approximation on Distributed Architectures

        **Authors**: *Oleg Balabanov, Matthias Beaupère, Laura Grigori, Victor Lederer*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/balabanov23a.html](https://proceedings.mlr.press/v202/balabanov23a.html)

        **Abstract**:

        This article introduces a novel structured random matrix composed blockwise from subsampled randomized Hadamard transforms (SRHTs). The block SRHT is expected to outperform well-known dimension reduction maps, including SRHT and Gaussian matrices on distributed architectures. We prove that a block SRHT with enough rows is an oblivious subspace embedding, i.e., an approximate isometry for an arbitrary low-dimensional subspace with high probability. Our estimate of the required number of rows is similar to that of the standard SRHT. This suggests that the two transforms should provide the same accuracy of approximation in the algorithms. The block SRHT can be readily incorporated into randomized methods for computing a low-rank approximation of a large-scale matrix, such as the Nyström method. For completeness, we revisit this method with a discussion of its implementation on distributed architectures.

        ----

        ## [66] Efficient Online Reinforcement Learning with Offline Data

        **Authors**: *Philip J. Ball, Laura M. Smith, Ilya Kostrikov, Sergey Levine*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/ball23a.html](https://proceedings.mlr.press/v202/ball23a.html)

        **Abstract**:

        Sample efficiency and exploration remain major challenges in online reinforcement learning (RL). A powerful approach that can be applied to address these issues is the inclusion of offline data, such as prior trajectories from a human expert or a sub-optimal exploration policy. Previous methods have relied on extensive modifications and additional complexity to ensure the effective use of this data. Instead, we ask: can we simply apply existing off-policy methods to leverage offline data when learning online? In this work, we demonstrate that the answer is yes; however, a set of minimal but important changes to existing off-policy RL algorithms are required to achieve reliable performance. We extensively ablate these design choices, demonstrating the key factors that most affect performance, and arrive at a set of recommendations that practitioners can readily apply, whether their data comprise a small number of expert demonstrations or large volumes of sub-optimal trajectories. We see that correct application of these simple recommendations can provide a $\mathbf{2.5\times}$ improvement over existing approaches across a diverse set of competitive benchmarks, with no additional computational overhead.

        ----

        ## [67] Mirror Sinkhorn: Fast Online Optimization on Transport Polytopes

        **Authors**: *Marin Ballu, Quentin Berthet*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/ballu23a.html](https://proceedings.mlr.press/v202/ballu23a.html)

        **Abstract**:

        Optimal transport is an important tool in machine learning, allowing to capture geometric properties of the data through a linear program on transport polytopes. We present a single-loop optimization algorithm for minimizing general convex objectives on these domains, utilizing the principles of Sinkhorn matrix scaling and mirror descent. The proposed algorithm is robust to noise, and can be used in an online setting. We provide theoretical guarantees for convex objectives and experimental results showcasing it effectiveness on both synthetic and real-world data.

        ----

        ## [68] On the Functional Similarity of Robust and Non-Robust Neural Representations

        **Authors**: *András Balogh, Márk Jelasity*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/balogh23a.html](https://proceedings.mlr.press/v202/balogh23a.html)

        **Abstract**:

        Model stitching—where the internal representations of two neural networks are aligned linearly—helped demonstrate that the representations of different neural networks for the same task are surprisingly similar in a functional sense. At the same time, the representations of adversarially robust networks are considered to be different from non-robust representations. For example, robust image classifiers are invertible, while non-robust networks are not. Here, we investigate the functional similarity of robust and non-robust representations for image classification with the help of model stitching. We find that robust and non-robust networks indeed have different representations. However, these representations are compatible regarding accuracy. From the point of view of robust accuracy, compatibility decreases quickly after the first few layers but the representations become compatible again in the last layers, in the sense that the properties of the front model can be recovered. Moreover, this is true even in the case of cross-task stitching. Our results suggest that stitching in the initial, preprocessing layers and the final, abstract layers test different kinds of compatibilities. In particular, the final layers are easy to match, because their representations depend mostly on the same abstract task specification, in our case, the classification of the input into $n$ classes.

        ----

        ## [69] Robust Budget Pacing with a Single Sample

        **Authors**: *Santiago R. Balseiro, Rachitesh Kumar, Vahab Mirrokni, Balasubramanian Sivan, Di Wang*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/balseiro23a.html](https://proceedings.mlr.press/v202/balseiro23a.html)

        **Abstract**:

        Major Internet advertising platforms offer budget pacing tools as a standard service for advertisers to manage their ad campaigns. Given the inherent non-stationarity in an advertiser’s value and also competing advertisers’ values over time, a commonly used approach is to learn a target expenditure plan that specifies a target spend as a function of time, and then run a controller that tracks this plan. This raises the question: how many historical samples are required to learn a good expenditure plan? We study this question by considering an advertiser repeatedly participating in $T$ second-price auctions, where the tuple of her value and the highest competing bid is drawn from an unknown time-varying distribution. The advertiser seeks to maximize her total utility subject to her budget constraint. Prior work has shown the sufficiency of $T\log T$ samples per distribution to achieve the optimal $O(\sqrt{T})$-regret. We dramatically improve this state-of-the-art and show that just one sample per distribution is enough to achieve the near-optimal $\tilde O(\sqrt{T})$-regret, while still being robust to noise in the sampling distributions.

        ----

        ## [70] Dynamic Constrained Submodular Optimization with Polylogarithmic Update Time

        **Authors**: *Kiarash Banihashem, Leyla Biabani, Samira Goudarzi, MohammadTaghi Hajiaghayi, Peyman Jabbarzade, Morteza Monemizadeh*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/banihashem23a.html](https://proceedings.mlr.press/v202/banihashem23a.html)

        **Abstract**:

        Maximizing a monotone submodular function under cardinality constraint $k$ is a core problem in machine learning and database with many basic applications, including video and data summarization, recommendation systems, feature extraction, exemplar clustering, and coverage problems. We study this classic problem in the fully dynamic model where a stream of insertions and deletions of elements of an underlying ground set is given and the goal is to maintain an approximate solution using a fast update time. A recent paper at NeurIPS’20 by Lattanzi, Mitrovic, Norouzi-Fard, Tarnawski, Zadimoghaddam claims to obtain a dynamic algorithm for this problem with a $(\frac{1}{2} -\epsilon)$ approximation ratio and a query complexity bounded by $\mathrm{poly}(\log(n),\log(k),\epsilon^{-1})$. However, as we explain in this paper, the analysis has some important gaps. Having a dynamic algorithm for the problem with polylogarithmic update time is even more important in light of a recent result by Chen and Peng at STOC’22 who show a matching lower bound for the problem – any randomized algorithm with a $\frac{1}{2}+\epsilon$ approximation ratio must have an amortized query complexity that is polynomial in $n$. In this paper, we develop a simpler algorithm for the problem that maintains a $(\frac{1}{2}-\epsilon)$-approximate solution for submodular maximization under cardinality constraint $k$ using a polylogarithmic amortized update time.

        ----

        ## [71] One Transformer Fits All Distributions in Multi-Modal Diffusion at Scale

        **Authors**: *Fan Bao, Shen Nie, Kaiwen Xue, Chongxuan Li, Shi Pu, Yaole Wang, Gang Yue, Yue Cao, Hang Su, Jun Zhu*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/bao23a.html](https://proceedings.mlr.press/v202/bao23a.html)

        **Abstract**:

        This paper proposes a unified diffusion framework (dubbed UniDiffuser) to fit all distributions relevant to a set of multi-modal data in one model. Our key insight is – learning diffusion models for marginal, conditional, and joint distributions can be unified as predicting the noise in the perturbed data, where the perturbation levels (i.e. timesteps) can be different for different modalities. Inspired by the unified view, UniDiffuser learns all distributions simultaneously with a minimal modification to the original diffusion model – perturbs data in all modalities instead of a single modality, inputs individual timesteps in different modalities, and predicts the noise of all modalities instead of a single modality. UniDiffuser is parameterized by a transformer for diffusion models to handle input types of different modalities. Implemented on large-scale paired image-text data, UniDiffuser is able to perform image, text, text-to-image, image-to-text, and image-text pair generation by setting proper timesteps without additional overhead. In particular, UniDiffuser is able to produce perceptually realistic samples in all tasks and its quantitative results (e.g., the FID and CLIP score) are not only superior to existing general-purpose models but also comparable to the bespoken models (e.g., Stable Diffusion and DALL-E 2) in representative tasks (e.g., text-to-image generation).

        ----

        ## [72] Optimizing the Collaboration Structure in Cross-Silo Federated Learning

        **Authors**: *Wenxuan Bao, Haohan Wang, Jun Wu, Jingrui He*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/bao23b.html](https://proceedings.mlr.press/v202/bao23b.html)

        **Abstract**:

        In federated learning (FL), multiple clients collaborate to train machine learning models together while keeping their data decentralized. Through utilizing more training data, FL suffers from the potential negative transfer problem: the global FL model may even perform worse than the models trained with local data only. In this paper, we propose FedCollab, a novel FL framework that alleviates negative transfer by clustering clients into non-overlapping coalitions based on their distribution distances and data quantities. As a result, each client only collaborates with the clients having similar data distributions, and tends to collaborate with more clients when it has less data. We evaluate our framework with a variety of datasets, models, and types of non-IIDness. Our results demonstrate that FedCollab effectively mitigates negative transfer across a wide range of FL algorithms and consistently outperforms other clustered FL algorithms.

        ----

        ## [73] MultiDiffusion: Fusing Diffusion Paths for Controlled Image Generation

        **Authors**: *Omer Bar-Tal, Lior Yariv, Yaron Lipman, Tali Dekel*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/bar-tal23a.html](https://proceedings.mlr.press/v202/bar-tal23a.html)

        **Abstract**:

        Recent advances in text-to-image generation with diffusion models present transformative capabilities in image quality. However, user controllability of the generated image, and fast adaptation to new tasks still remains an open challenge, currently mostly addressed by costly and long re-training and fine-tuning or ad-hoc adaptations to specific image generation tasks. In this work, we present MultiDiffusion, a unified framework that enables versatile and controllable image generation, using a pre-trained text-to-image diffusion model, without any further training or finetuning. At the center of our approach is a new generation process, based on an optimization task that binds together multiple diffusion generation processes with a shared set of parameters or constraints. We show that MultiDiffusion can be readily applied to generate high quality and diverse images that adhere to user-provided controls, such as desired aspect ratio (e.g., panorama), and spatial guiding signals, ranging from tight segmentation masks to bounding boxes.

        ----

        ## [74] Reinforcement Learning with General Utilities: Simpler Variance Reduction and Large State-Action Space

        **Authors**: *Anas Barakat, Ilyas Fatkhullin, Niao He*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/barakat23a.html](https://proceedings.mlr.press/v202/barakat23a.html)

        **Abstract**:

        We consider the reinforcement learning (RL) problem with general utilities which consists in maximizing a function of the state-action occupancy measure. Beyond the standard cumulative reward RL setting, this problem includes as particular cases constrained RL, pure exploration and learning from demonstrations among others. For this problem, we propose a simpler single-loop parameter-free normalized policy gradient algorithm. Implementing a recursive momentum variance reduction mechanism, our algorithm achieves $\tilde{\mathcal{O}}(\epsilon^{-3})$ and $\tilde{\mathcal{O}}(\epsilon^{-2})$ sample complexities for $\epsilon$-first-order stationarity and $\epsilon$-global optimality respectively, under adequate assumptions. We further address the setting of large finite state action spaces via linear function approximation of the occupancy measure and show a $\tilde{\mathcal{O}}(\epsilon^{-4})$ sample complexity for a simple policy gradient method with a linear regression subroutine.

        ----

        ## [75] Interpretable Neural-Symbolic Concept Reasoning

        **Authors**: *Pietro Barbiero, Gabriele Ciravegna, Francesco Giannini, Mateo Espinosa Zarlenga, Lucie Charlotte Magister, Alberto Tonda, Pietro Lio, Frédéric Precioso, Mateja Jamnik, Giuseppe Marra*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/barbiero23a.html](https://proceedings.mlr.press/v202/barbiero23a.html)

        **Abstract**:

        Deep learning methods are highly accurate, yet their opaque decision process prevents them from earning full human trust. Concept-based models aim to address this issue by learning tasks based on a set of human-understandable concepts. However, state-of-the-art concept-based models rely on high-dimensional concept embedding representations which lack a clear semantic meaning, thus questioning the interpretability of their decision process. To overcome this limitation, we propose the Deep Concept Reasoner (DCR), the first interpretable concept-based model that builds upon concept embeddings. In DCR, neural networks do not make task predictions directly, but they build syntactic rule structures using concept embeddings. DCR then executes these rules on meaningful concept truth degrees to provide a final interpretable and semantically-consistent prediction in a differentiable manner. Our experiments show that DCR: (i) improves up to +25% w.r.t. state-of-the-art interpretable concept-based models on challenging benchmarks (ii) discovers meaningful logic rules matching known ground truths even in the absence of concept supervision during training, and (iii), facilitates the generation of counterfactual examples providing the learnt rules as guidance.

        ----

        ## [76] Moccasin: Efficient Tensor Rematerialization for Neural Networks

        **Authors**: *Burak Bartan, Haoming Li, Harris Teague, Christopher Lott, Bistra Dilkina*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/bartan23a.html](https://proceedings.mlr.press/v202/bartan23a.html)

        **Abstract**:

        The deployment and training of neural networks on edge computing devices pose many challenges. The low memory nature of edge devices is often one of the biggest limiting factors encountered in the deployment of large neural network models. Tensor rematerialization or recompute is a way to address high memory requirements for neural network training and inference. In this paper we consider the problem of execution time minimization of compute graphs subject to a memory budget. In particular, we develop a new constraint programming formulation called Moccasin with only $O(n)$ integer variables, where $n$ is the number of nodes in the compute graph. This is a significant improvement over the works in the recent literature that propose formulations with $O(n^2)$ Boolean variables. We present numerical studies that show that our approach is up to an order of magnitude faster than recent work especially for large-scale graphs.

        ----

        ## [77] User-level Private Stochastic Convex Optimization with Optimal Rates

        **Authors**: *Raef Bassily, Ziteng Sun*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/bassily23a.html](https://proceedings.mlr.press/v202/bassily23a.html)

        **Abstract**:

        We study the problem of differentially private (DP) stochastic convex optimization (SCO) under the notion of user-level differential privacy. In this problem, there are $n$ users, each contributing $m>1$ samples to the input dataset of the private SCO algorithm, and the notion of indistinguishability embedded in DP is w.r.t. replacing the entire local dataset of any given user. Under smoothness conditions of the loss, we establish the optimal rates for user-level DP-SCO in both the central and local models of DP. In particular, we show, roughly, that the optimal rate is $\frac{1}{\sqrt{nm}}+\frac{\sqrt{d}}{\varepsilon n \sqrt{m}}$ in the central setting and is $\frac{\sqrt{d}}{\varepsilon \sqrt{nm}}$ in the local setting, where $d$ is the dimensionality of the problem and $\varepsilon$ is the privacy parameter. Our algorithms combine new user-level DP mean estimation techniques with carefully designed first-order stochastic optimization methods. For the central DP setting, our optimal rate improves over the rate attained for the same setting in Levy et al. (2021) by $\sqrt{d}$ factor. One of the main ingredients that enabled such an improvement is a novel application of the generalization properties of DP in the context of multi-pass stochastic gradient methods.

        ----

        ## [78] A Statistical Perspective on Retrieval-Based Models

        **Authors**: *Soumya Basu, Ankit Singh Rawat, Manzil Zaheer*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/basu23a.html](https://proceedings.mlr.press/v202/basu23a.html)

        **Abstract**:

        Many modern high-performing machine learning models increasingly rely on scaling up models, e.g., transformer networks. Simultaneously, a parallel line of work aims to improve the model performance by augmenting an input instance with other (labeled) instances during inference. Examples of such augmentations include task-specific prompts and similar examples retrieved from the training data by a nonparametric component. Despite a growing literature showcasing the promise of these retrieval-based models, their theoretical underpinnings %for such models remain under-explored. In this paper, we present a formal treatment of retrieval-based models to characterize their performance via a novel statistical perspective. In particular, we study two broad classes of retrieval-based classification approaches: First, we analyze a local learning framework that employs an explicit local empirical risk minimization based on retrieved examples for each input instance. Interestingly, we show that breaking down the underlying learning task into local sub-tasks enables the model to employ a low complexity parametric component to ensure good overall performance. The second class of retrieval-based approaches we explore learns a global model using kernel methods to directly map an input instance and retrieved examples to a prediction, without explicitly solving a local learning task.

        ----

        ## [79] Human-Timescale Adaptation in an Open-Ended Task Space

        **Authors**: *Jakob Bauer, Kate Baumli, Feryal M. P. Behbahani, Avishkar Bhoopchand, Nathalie Bradley-Schmieg, Michael Chang, Natalie Clay, Adrian Collister, Vibhavari Dasagi, Lucy Gonzalez, Karol Gregor, Edward Hughes, Sheleem Kashem, Maria Loks-Thompson, Hannah Openshaw, Jack Parker-Holder, Shreya Pathak, Nicolas Perez Nieves, Nemanja Rakicevic, Tim Rocktäschel, Yannick Schroecker, Satinder Singh, Jakub Sygnowski, Karl Tuyls, Sarah York, Alexander Zacherl, Lei M. Zhang*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/bauer23a.html](https://proceedings.mlr.press/v202/bauer23a.html)

        **Abstract**:

        Foundation models have shown impressive adaptation and scalability in supervised and self-supervised learning problems, but so far these successes have not fully translated to reinforcement learning (RL). In this work, we demonstrate that training an RL agent at scale leads to a general in-context learning algorithm that can adapt to open-ended novel embodied 3D problems as quickly as humans. In a vast space of held-out environment dynamics, our adaptive agent (AdA) displays on-the-fly hypothesis-driven exploration, efficient exploitation of acquired knowledge, and can successfully be prompted with first-person demonstrations. Adaptation emerges from three ingredients: (1) meta-reinforcement learning across a vast, smooth and diverse task distribution, (2) a policy parameterised as a large-scale attention-based memory architecture, and (3) an effective automated curriculum that prioritises tasks at the frontier of an agent’s capabilities. We demonstrate characteristic scaling laws with respect to network size, memory length, and richness of the training task distribution. We believe our results lay the foundation for increasingly general and adaptive RL agents that perform well across ever-larger open-ended domains.

        ----

        ## [80] A Kernel Stein Test of Goodness of Fit for Sequential Models

        **Authors**: *Jerome Baum, Heishiro Kanagawa, Arthur Gretton*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/baum23a.html](https://proceedings.mlr.press/v202/baum23a.html)

        **Abstract**:

        We propose a goodness-of-fit measure for probability densities modeling observations with varying dimensionality, such as text documents of differing lengths or variable-length sequences. The proposed measure is an instance of the kernel Stein discrepancy (KSD), which has been used to construct goodness-of-fit tests for unnormalized densities. The KSD is defined by its Stein operator: current operators used in testing apply to fixed-dimensional spaces. As our main contribution, we extend the KSD to the variable-dimension setting by identifying appropriate Stein operators, and propose a novel KSD goodness-of-fit test. As with the previous variants, the proposed KSD does not require the density to be normalized, allowing the evaluation of a large class of models. Our test is shown to perform well in practice on discrete sequential data benchmarks.

        ----

        ## [81] Individually Fair Learning with One-Sided Feedback

        **Authors**: *Yahav Bechavod, Aaron Roth*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/bechavod23a.html](https://proceedings.mlr.press/v202/bechavod23a.html)

        **Abstract**:

        We consider an online learning problem with one-sided feedback, in which the learner is able to observe the true label only for positively predicted instances. On each round, $k$ instances arrive and receive classification outcomes according to a randomized policy deployed by the learner, whose goal is to maximize accuracy while deploying individually fair policies. We first present a novel auditing scheme, capable of utilizing feedback from dynamically-selected panels of multiple, possibly inconsistent, auditors regarding fairness violations. In particular, we show how our proposed auditing scheme allows for algorithmically exploring the resulting accuracy-fairness frontier, with no need for additional feedback from auditors. We then present an efficient reduction from our problem of online learning with one-sided feedback and a panel reporting fairness violations to the contextual combinatorial semi-bandit problem (Cesa-Bianchi & Lugosi, 2009; Gyorgy et al., 2007), allowing us to leverage algorithms for contextual combinatorial semi-bandits to establish multi-criteria no regret guarantees in our setting, simultaneously for accuracy and fairness. Our results eliminate two potential sources of bias from prior work: the “hidden outcomes” that are not available to an algorithm operating in the full information setting, and human biases that might be present in any single human auditor, but can be mitigated by selecting a well-chosen panel.

        ----

        ## [82] Predicting Ordinary Differential Equations with Transformers

        **Authors**: *Sören Becker, Michal Klein, Alexander Neitz, Giambattista Parascandolo, Niki Kilbertus*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/becker23a.html](https://proceedings.mlr.press/v202/becker23a.html)

        **Abstract**:

        We develop a transformer-based sequence-to-sequence model that recovers scalar ordinary differential equations (ODEs) in symbolic form from irregularly sampled and noisy observations of a single solution trajectory. We demonstrate in extensive empirical evaluations that our model performs better or on par with existing methods in terms of accurate recovery across various settings. Moreover, our method is efficiently scalable: after one-time pretraining on a large set of ODEs, we can infer the governing law of a new observed solution in a few forward passes of the model.

        ----

        ## [83] Explaining Reinforcement Learning with Shapley Values

        **Authors**: *Daniel Beechey, Thomas M. S. Smith, Özgür Simsek*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/beechey23a.html](https://proceedings.mlr.press/v202/beechey23a.html)

        **Abstract**:

        For reinforcement learning systems to be widely adopted, their users must understand and trust them. We present a theoretical analysis of explaining reinforcement learning using Shapley values, following a principled approach from game theory for identifying the contribution of individual players to the outcome of a cooperative game. We call this general framework Shapley Values for Explaining Reinforcement Learning (SVERL). Our analysis exposes the limitations of earlier uses of Shapley values in reinforcement learning. We then develop an approach that uses Shapley values to explain agent performance. In a variety of domains, SVERL produces meaningful explanations that match and supplement human intuition.

        ----

        ## [84] TIDE: Time Derivative Diffusion for Deep Learning on Graphs

        **Authors**: *Maysam Behmanesh, Maximilian Krahn, Maks Ovsjanikov*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/behmanesh23a.html](https://proceedings.mlr.press/v202/behmanesh23a.html)

        **Abstract**:

        A prominent paradigm for graph neural networks is based on the message-passing framework. In this framework, information communication is realized only between neighboring nodes. The challenge of approaches that use this paradigm is to ensure efficient and accurate long-distance communication between nodes, as deep convolutional networks are prone to over smoothing. In this paper, we present a novel method based on time derivative graph diffusion (TIDE) to overcome these structural limitations of the message-passing framework. Our approach allows for optimizing the spatial extent of diffusion across various tasks and network channels, thus enabling medium and long-distance communication efficiently. Furthermore, we show that our architecture design also enables local message-passing and thus inherits from the capabilities of local message-passing approaches. We show that on both widely used graph benchmarks and synthetic mesh and graph datasets, the proposed framework outperforms state-of-the-art methods by a significant margin.

        ----

        ## [85] Fast as CHITA: Neural Network Pruning with Combinatorial Optimization

        **Authors**: *Riade Benbaki, Wenyu Chen, Xiang Meng, Hussein Hazimeh, Natalia Ponomareva, Zhe Zhao, Rahul Mazumder*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/benbaki23a.html](https://proceedings.mlr.press/v202/benbaki23a.html)

        **Abstract**:

        The sheer size of modern neural networks makes model serving a serious computational challenge. A popular class of compression techniques overcomes this challenge by pruning or sparsifying the weights of pretrained networks. While useful, these techniques often face serious tradeoffs between computational requirements and compression quality. In this work, we propose a novel optimization-based pruning framework that considers the combined effect of pruning (and updating) multiple weights subject to a sparsity constraint. Our approach, CHITA, extends the classical Optimal Brain Surgeon framework and results in significant improvements in speed, memory, and performance over existing optimization-based approaches for network pruning. CHITA’s main workhorse performs combinatorial optimization updates on a memory-friendly representation of local quadratic approximation(s) of the loss function. On a standard benchmark of pretrained models and datasets, CHITA leads to superior sparsity-accuracy tradeoffs than competing methods. For example, for MLPNet with only 2% of the weights retained, our approach improves the accuracy by 63% relative to the state of the art. Furthermore, when used in conjunction with fine-tuning SGD steps, our method achieves significant accuracy gains over state-of-the-art approaches. Our code is publicly available at: https://github.com/mazumder-lab/CHITA .

        ----

        ## [86] Continuously Parameterized Mixture Models

        **Authors**: *Christopher M. Bender, Yifeng Shi, Marc Niethammer, Junier Oliva*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/bender23a.html](https://proceedings.mlr.press/v202/bender23a.html)

        **Abstract**:

        Mixture models are universal approximators of smooth densities but are difficult to utilize in complicated datasets due to restrictions on typically available modes and challenges with initialiations. We show that by continuously parameterizing a mixture of factor analyzers using a learned ordinary differential equation, we can improve the fit of mixture models over direct methods. Once trained, the mixture components can be extracted and the neural ODE can be discarded, leaving us with an effective, but low-resource model. We additionally explore the use of a training curriculum from an easy-to-model latent space extracted from a normalizing flow to the more complex input space and show that the smooth curriculum helps to stabilize and improve results with and without the continuous parameterization. Finally, we introduce a hierarchical version of the model to enable more flexible, robust classification and clustering, and show substantial improvements against traditional parameterizations of GMMs.

        ----

        ## [87] Controllable Neural Symbolic Regression

        **Authors**: *Tommaso Bendinelli, Luca Biggio, Pierre-Alexandre Kamienny*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/bendinelli23a.html](https://proceedings.mlr.press/v202/bendinelli23a.html)

        **Abstract**:

        In symbolic regression, the objective is to find an analytical expression that accurately fits experimental data with the minimal use of mathematical symbols such as operators, variables, and constants. However, the combinatorial space of possible expressions can make it challenging for traditional evolutionary algorithms to find the correct expression in a reasonable amount of time. To address this issue, Neural Symbolic Regression (NSR) algorithms have been developed that can quickly identify patterns in the data and generate analytical expressions. However, these methods, in their current form, lack the capability to incorporate user-defined prior knowledge, which is often required in natural sciences and engineering fields. To overcome this limitation, we propose a novel neural symbolic regression method, named Neural Symbolic Regression with Hypothesis (NSRwH) that enables the explicit incorporation of assumptions about the expected structure of the ground-truth expression into the prediction process. Our experiments demonstrate that the proposed conditioned deep learning model outperforms its unconditioned counterparts in terms of accuracy while also providing control over the predicted expression structure.

        ----

        ## [88] On Second-Order Scoring Rules for Epistemic Uncertainty Quantification

        **Authors**: *Viktor Bengs, Eyke Hüllermeier, Willem Waegeman*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/bengs23a.html](https://proceedings.mlr.press/v202/bengs23a.html)

        **Abstract**:

        It is well known that accurate probabilistic predictors can be trained through empirical risk minimisation with proper scoring rules as loss functions. While such learners capture so-called aleatoric uncertainty of predictions, various machine learning methods have recently been developed with the goal to let the learner also represent its epistemic uncertainty, i.e., the uncertainty caused by a lack of knowledge and data. An emerging branch of the literature proposes the use of a second-order learner that provides predictions in terms of distributions on probability distributions. However, recent work has revealed serious theoretical shortcomings for second-order predictors based on loss minimisation. In this paper, we generalise these findings and prove a more fundamental result: There seems to be no loss function that provides an incentive for a second-order learner to faithfully represent its epistemic uncertainty in the same manner as proper scoring rules do for standard (first-order) learners. As a main mathematical tool to prove this result, we introduce the generalised notion of second-order scoring rules.

        ----

        ## [89] Certified Robust Neural Networks: Generalization and Corruption Resistance

        **Authors**: *M. Amine Bennouna, Ryan Lucas, Bart P. G. Van Parys*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/bennouna23a.html](https://proceedings.mlr.press/v202/bennouna23a.html)

        **Abstract**:

        Recent work have demonstrated that robustness (to "corruption") can be at odds with generalization. Adversarial training, for instance, aims to reduce the problematic susceptibility of modern neural networks to small data perturbations. Surprisingly, overfitting is a major concern in adversarial training despite being mostly absent in standard training. We provide here theoretical evidence for this peculiar “robust overfitting” phenomenon. Subsequently, we advance a novel distributionally robust loss function bridging robustness and generalization. We demonstrate both theoretically as well as empirically the loss to enjoy a certified level of robustness against two common types of corruption|data evasion and poisoning attacks|while ensuring guaranteed generalization. We show through careful numerical experiments that our resulting holistic robust (HR) training procedure yields SOTA performance. Finally, we indicate that HR training can be interpreted as a direct extension of adversarial training and comes with a negligible additional computational burden. A ready-to-use python library implementing our algorithm is available at https://github.com/RyanLucas3/HR_Neural_Networks.

        ----

        ## [90] Gaussian processes at the Helm(holtz): A more fluid model for ocean currents

        **Authors**: *Renato Berlinghieri, Brian L. Trippe, David R. Burt, Ryan James Giordano, Kaushik Srinivasan, Tamay M. Özgökmen, Junfei Xia, Tamara Broderick*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/berlinghieri23a.html](https://proceedings.mlr.press/v202/berlinghieri23a.html)

        **Abstract**:

        Oceanographers are interested in predicting ocean currents and identifying divergences in a current vector field based on sparse observations of buoy velocities. Since we expect current dynamics to be smooth but highly non-linear, Gaussian processes (GPs) offer an attractive model. But we show that applying a GP with a standard stationary kernel directly to buoy data can struggle at both current prediction and divergence identification – due to some physically unrealistic prior assumptions. To better reflect known physical properties of currents, we propose to instead put a standard stationary kernel on the divergence and curl-free components of a vector field obtained through a Helmholtz decomposition. We show that, because this decomposition relates to the original vector field just via mixed partial derivatives, we can still perform inference given the original data with only a small constant multiple of additional computational expense. We illustrate the benefits of our method on synthetic and real oceans data.

        ----

        ## [91] Optimal Rates and Efficient Algorithms for Online Bayesian Persuasion

        **Authors**: *Martino Bernasconi, Matteo Castiglioni, Andrea Celli, Alberto Marchesi, Francesco Trovò, Nicola Gatti*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/bernasconi23a.html](https://proceedings.mlr.press/v202/bernasconi23a.html)

        **Abstract**:

        Bayesian persuasion studies how an informed sender should influence beliefs of rational receivers that take decisions through Bayesian updating of a common prior. We focus on the online Bayesian persuasion framework, in which the sender repeatedly faces one or more receivers with unknown and adversarially selected types. First, we show how to obtain a tight $\tilde O(T^{1/2})$ regret bound in the case in which the sender faces a single receiver and has bandit feedback, improving over the best previously known bound of $\tilde O(T^{4/5})$. Then, we provide the first no-regret guarantees for the multi-receiver setting under bandit feedback. Finally, we show how to design no-regret algorithms with polynomial per-iteration running time by exploiting type reporting, thereby circumventing known complexity results on online Bayesian persuasion. We provide efficient algorithms guaranteeing a $O(T^{1/2})$ regret upper bound both in the single- and multi-receiver scenario when type reporting is allowed.

        ----

        ## [92] Constrained Phi-Equilibria

        **Authors**: *Martino Bernasconi, Matteo Castiglioni, Alberto Marchesi, Francesco Trovò, Nicola Gatti*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/bernasconi23b.html](https://proceedings.mlr.press/v202/bernasconi23b.html)

        **Abstract**:

        The computational study of equilibria involving constraints on players’ strategies has been largely neglected. However, in real-world applications, players are usually subject to constraints ruling out the feasibility of some of their strategies, such as, e.g., safety requirements and budget caps. Computational studies on constrained versions of the Nash equilibrium have lead to some results under very stringent assumptions, while finding constrained versions of the correlated equilibrium (CE) is still unexplored. In this paper, we introduce and computationally characterize constrained Phi-equilibria—a more general notion than constrained CEs—in normal-form games. We show that computing such equilibria is in general computationally intractable, and also that the set of the equilibria may not be convex, providing a sharp divide with unconstrained CEs. Nevertheless, we provide a polynomial-time algorithm for computing a constrained (approximate) Phi-equilibrium maximizing a given linear function, when either the number of constraints or that of players’ actions is fixed. Moreover, in the special case in which a player’s constraints do not depend on other players’ strategies, we show that an exact, function-maximizing equilibrium can be computed in polynomial time, while one (approximate) equilibrium can be found with an efficient decentralized no-regret learning algorithm.

        ----

        ## [93] Differentiable and Transportable Structure Learning

        **Authors**: *Jeroen Berrevoets, Nabeel Seedat, Fergus Imrie, Mihaela van der Schaar*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/berrevoets23a.html](https://proceedings.mlr.press/v202/berrevoets23a.html)

        **Abstract**:

        Directed acyclic graphs (DAGs) encode a lot of information about a particular distribution in their structure. However, compute required to infer these structures is typically super-exponential in the number of variables, as inference requires a sweep of a combinatorially large space of potential structures. That is, until recent advances made it possible to search this space using a differentiable metric, drastically reducing search time. While this technique— named NOTEARS —is widely considered a seminal work in DAG-discovery, it concedes an important property in favour of differentiability: transportability. To be transportable, the structures discovered on one dataset must apply to another dataset from the same domain. We introduce D-Struct which recovers transportability in the discovered structures through a novel architecture and loss function while remaining fully differentiable. Because D-Struct remains differentiable, our method can be easily adopted in existing differentiable architectures, as was previously done with NOTEARS. In our experiments, we empirically validate D-Struct with respect to edge accuracy and structural Hamming distance in a variety of settings.

        ----

        ## [94] Polyhedral Complex Extraction from ReLU Networks using Edge Subdivision

        **Authors**: *Arturs Berzins*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/berzins23a.html](https://proceedings.mlr.press/v202/berzins23a.html)

        **Abstract**:

        A neural network consisting of piecewise affine building blocks, such as fully-connected layers and ReLU activations, is itself a piecewise affine function supported on a polyhedral complex. This complex has been previously studied to characterize theoretical properties of neural networks, but, in practice, extracting it remains a challenge due to its high combinatorial complexity. A natural idea described in previous works is to subdivide the regions via intersections with hyperplanes induced by each neuron. However, we argue that this view leads to computational redundancy. Instead of regions, we propose to subdivide edges, leading to a novel method for polyhedral complex extraction. A key to this are sign-vectors, which encode the combinatorial structure of the complex. Our approach allows to use standard tensor operations on a GPU, taking seconds for millions of cells on a consumer grade machine. Motivated by the growing interest in neural shape representation, we use the speed and differentiablility of our method to optimize geometric properties of the complex. The code is available at https://github.com/arturs-berzins/relu_edge_subdivision.

        ----

        ## [95] Robust One-Class Classification with Signed Distance Function using 1-Lipschitz Neural Networks

        **Authors**: *Louis Béthune, Paul Novello, Guillaume Coiffier, Thibaut Boissin, Mathieu Serrurier, Quentin Vincenot, Andres Troya-Galvis*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/bethune23a.html](https://proceedings.mlr.press/v202/bethune23a.html)

        **Abstract**:

        We propose a new method, dubbed One Class Signed Distance Function (OCSDF), to perform One Class Classification (OCC) by provably learning the Signed Distance Function (SDF) to the boundary of the support of any distribution. The distance to the support can be interpreted as a normality score, and its approximation using 1-Lipschitz neural networks provides robustness bounds against $l2$ adversarial attacks, an under-explored weakness of deep learning-based OCC algorithms. As a result, OCSDF comes with a new metric, certified AUROC, that can be computed at the same cost as any classical AUROC. We show that OCSDF is competitive against concurrent methods on tabular and image data while being way more robust to adversarial attacks, illustrating its theoretical properties. Finally, as exploratory research perspectives, we theoretically and empirically show how OCSDF connects OCC with image generation and implicit neural surface parametrization.

        ----

        ## [96] Neural Algorithmic Reasoning with Causal Regularisation

        **Authors**: *Beatrice Bevilacqua, Kyriacos Nikiforou, Borja Ibarz, Ioana Bica, Michela Paganini, Charles Blundell, Jovana Mitrovic, Petar Velickovic*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/bevilacqua23a.html](https://proceedings.mlr.press/v202/bevilacqua23a.html)

        **Abstract**:

        Recent work on neural algorithmic reasoning has investigated the reasoning capabilities of neural networks, effectively demonstrating they can learn to execute classical algorithms on unseen data coming from the train distribution. However, the performance of existing neural reasoners significantly degrades on out-of-distribution (OOD) test data, where inputs have larger sizes. In this work, we make an important observation: there are many different inputs for which an algorithm will perform certain intermediate computations identically. This insight allows us to develop data augmentation procedures that, given an algorithm’s intermediate trajectory, produce inputs for which the target algorithm would have exactly the same next trajectory step. We ensure invariance in the next-step prediction across such inputs, by employing a self-supervised objective derived by our observation, formalised in a causal graph. We prove that the resulting method, which we call Hint-ReLIC, improves the OOD generalisation capabilities of the reasoner. We evaluate our method on the CLRS algorithmic reasoning benchmark, where we show up to 3x improvements on the OOD test data.

        ----

        ## [97] Optimally-weighted Estimators of the Maximum Mean Discrepancy for Likelihood-Free Inference

        **Authors**: *Ayush Bharti, Masha Naslidnyk, Oscar Key, Samuel Kaski, François-Xavier Briol*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/bharti23a.html](https://proceedings.mlr.press/v202/bharti23a.html)

        **Abstract**:

        Likelihood-free inference methods typically make use of a distance between simulated and real data. A common example is the maximum mean discrepancy (MMD), which has previously been used for approximate Bayesian computation, minimum distance estimation, generalised Bayesian inference, and within the nonparametric learning framework. The MMD is commonly estimated at a root-$m$ rate, where $m$ is the number of simulated samples. This can lead to significant computational challenges since a large $m$ is required to obtain an accurate estimate, which is crucial for parameter estimation. In this paper, we propose a novel estimator for the MMD with significantly improved sample complexity. The estimator is particularly well suited for computationally expensive smooth simulators with low- to mid-dimensional inputs. This claim is supported through both theoretical results and an extensive simulation study on benchmark simulators.

        ----

        ## [98] Bandit Online Linear Optimization with Hints and Queries

        **Authors**: *Aditya Bhaskara, Ashok Cutkosky, Ravi Kumar, Manish Purohit*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/bhaskara23a.html](https://proceedings.mlr.press/v202/bhaskara23a.html)

        **Abstract**:

        We study variants of the online linear optimization (OLO) problem with bandit feedback, where the algorithm has access to external information about the unknown cost vector. Our motivation is the recent body of work on using such “hints” towards improving regret bounds for OLO problems in the full-information setting. Unlike in the full-information OLO setting, with bandit feedback, we first show that one cannot improve the standard regret bounds of $\tilde{O}(\sqrt{T})$ by using hints, even if they are always well-correlated with the cost vector. In contrast, if the algorithm is empowered to issue queries and if all the responses are correct, then we show $O(\log T)$ regret is achievable. We then show how to make this result more robust—when some of the query responses can be adversarial—by using a little feedback on the quality of the responses.

        ----

        ## [99] Improved Online Conformal Prediction via Strongly Adaptive Online Learning

        **Authors**: *Aadyot Bhatnagar, Huan Wang, Caiming Xiong, Yu Bai*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/bhatnagar23a.html](https://proceedings.mlr.press/v202/bhatnagar23a.html)

        **Abstract**:

        We study the problem of uncertainty quantification via prediction sets, in an online setting where the data distribution may vary arbitrarily over time. Recent work develops online conformal prediction techniques that leverage regret minimization algorithms from the online learning literature to learn prediction sets with approximately valid coverage and small regret. However, standard regret minimization is insufficient for handling changing environments, where performance guarantees may be desired not only over the full time horizon but also in all (sub-)intervals of time. We develop new online conformal prediction methods that minimize the strongly adaptive regret, which measures the worst-case regret over all intervals of a fixed length. We prove that our methods achieve near-optimal strongly adaptive regret for all interval lengths simultaneously, and approximately valid coverage. Experiments show that our methods consistently obtain better coverage and smaller prediction sets than existing methods on real-world tasks such as time series forecasting and image classification under distribution shift.

        ----

        ## [100] Data-Copying in Generative Models: A Formal Framework

        **Authors**: *Robi Bhattacharjee, Sanjoy Dasgupta, Kamalika Chaudhuri*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/bhattacharjee23a.html](https://proceedings.mlr.press/v202/bhattacharjee23a.html)

        **Abstract**:

        There has been some recent interest in detecting and addressing memorization of training data by deep neural networks. A formal framework for memorization in generative models, called “data-copying” was proposed by Meehan et. al (2020). We build upon their work to show that their framework may fail to detect certain kinds of blatant memorization. Motivated by this and the theory of non-parametric methods, we provide an alternative definition of data-copying that applies more locally. We provide a method to detect data-copying, and provably show that it works with high probability when enough data is available. We also provide lower bounds that characterize the sample requirement for reliable detection.

        ----

        ## [101] Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling

        **Authors**: *Stella Biderman, Hailey Schoelkopf, Quentin Gregory Anthony, Herbie Bradley, Kyle O'Brien, Eric Hallahan, Mohammad Aflah Khan, Shivanshu Purohit, USVSN Sai Prashanth, Edward Raff, Aviya Skowron, Lintang Sutawika, Oskar van der Wal*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/biderman23a.html](https://proceedings.mlr.press/v202/biderman23a.html)

        **Abstract**:

        How do large language models (LLMs) develop and evolve over the course of training? How do these patterns change as models scale? To answer these questions, we introduce Pythia, a suite of 16 LLMs all trained on public data seen in the exact same order and ranging in size from 70M to 12B parameters. We provide public access to 154 checkpoints for each one of the 16 models, alongside tools to download and reconstruct their exact training dataloaders for further study. We intend Pythia to facilitate research in many areas, and we present several case studies including novel results in memorization, term frequency effects on few-shot performance, and reducing gender bias. We demonstrate that this highly controlled setup can be used to yield novel insights toward LLMs and their training dynamics. Trained models, analysis code, training code, and training data can be found at https://github.com/EleutherAI/pythia.

        ----

        ## [102] StriderNet: A Graph Reinforcement Learning Approach to Optimize Atomic Structures on Rough Energy Landscapes

        **Authors**: *Vaibhav Bihani, Sahil Manchanda, Srikanth Sastry, Sayan Ranu, N. M. Anoop Krishnan*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/bihani23a.html](https://proceedings.mlr.press/v202/bihani23a.html)

        **Abstract**:

        Optimization of atomic structures presents a challenging problem, due to their highly rough and non-convex energy landscape, with wide applications in the fields of drug design, materials discovery, and mechanics. Here, we present a graph reinforcement learning approach, StriderNet, that learns a policy to displace the atoms towards low energy configurations. We evaluate the performance of StriderNet on three complex atomic systems, namely, binary Lennard-Jones particles, calcium silicate hydrates gel, and disordered silicon. We show that StriderNet outperforms all classical optimization algorithms and enables the discovery of a lower energy minimum. In addition, StriderNet exhibits a higher rate of reaching minima with energies, as confirmed by the average over multiple realizations. Finally, we show that StriderNet exhibits inductivity to unseen system sizes that are an order of magnitude different from the training system. All the codes and datasets are available at https://github.com/M3RG-IITD/StriderNET.

        ----

        ## [103] Modeling Temporal Data as Continuous Functions with Stochastic Process Diffusion

        **Authors**: *Marin Bilos, Kashif Rasul, Anderson Schneider, Yuriy Nevmyvaka, Stephan Günnemann*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/bilos23a.html](https://proceedings.mlr.press/v202/bilos23a.html)

        **Abstract**:

        Temporal data such as time series can be viewed as discretized measurements of the underlying function. To build a generative model for such data we have to model the stochastic process that governs it. We propose a solution by defining the denoising diffusion model in the function space which also allows us to naturally handle irregularly-sampled observations. The forward process gradually adds noise to functions, preserving their continuity, while the learned reverse process removes the noise and returns functions as new samples. To this end, we define suitable noise sources and introduce novel denoising and score-matching models. We show how our method can be used for multivariate probabilistic forecasting and imputation, and how our model can be interpreted as a neural process.

        ----

        ## [104] In or Out? Fixing ImageNet Out-of-Distribution Detection Evaluation

        **Authors**: *Julian Bitterwolf, Maximilian Müller, Matthias Hein*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/bitterwolf23a.html](https://proceedings.mlr.press/v202/bitterwolf23a.html)

        **Abstract**:

        Out-of-distribution (OOD) detection is the problem of identifying inputs which are unrelated to the in-distribution task. The OOD detection performance when the in-distribution (ID) is ImageNet-1K is commonly being tested on a small range of test OOD datasets. We find that most of the currently used test OOD datasets, including datasets from the open set recognition (OSR) literature, have severe issues: In some cases more than 50$%$ of the dataset contains objects belonging to one of the ID classes. These erroneous samples heavily distort the evaluation of OOD detectors. As a solution, we introduce with NINCO a novel test OOD dataset, each sample checked to be ID free, which with its fine-grained range of OOD classes allows for a detailed analysis of an OOD detector’s strengths and failure modes, particularly when paired with a number of synthetic “OOD unit-tests”. We provide detailed evaluations across a large set of architectures and OOD detection methods on NINCO and the unit-tests, revealing new insights about model weaknesses and the effects of pretraining on OOD detection performance. We provide code and data at https://github.com/j-cb/NINCO.

        ----

        ## [105] Invariant Slot Attention: Object Discovery with Slot-Centric Reference Frames

        **Authors**: *Ondrej Biza, Sjoerd van Steenkiste, Mehdi S. M. Sajjadi, Gamaleldin Fathy Elsayed, Aravindh Mahendran, Thomas Kipf*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/biza23a.html](https://proceedings.mlr.press/v202/biza23a.html)

        **Abstract**:

        Automatically discovering composable abstractions from raw perceptual data is a long-standing challenge in machine learning. Recent slot-based neural networks that learn about objects in a self-supervised manner have made exciting progress in this direction. However, they typically fall short at adequately capturing spatial symmetries present in the visual world, which leads to sample inefficiency, such as when entangling object appearance and pose. In this paper, we present a simple yet highly effective method for incorporating spatial symmetries via slot-centric reference frames. We incorporate equivariance to per-object pose transformations into the attention and generation mechanism of Slot Attention by translating, scaling, and rotating position encodings. These changes result in little computational overhead, are easy to implement, and can result in large gains in terms of data efficiency and overall improvements to object discovery. We evaluate our method on a wide range of synthetic object discovery benchmarks namely CLEVR, Tetrominoes, CLEVRTex, Objects Room and MultiShapeNet, and show promising improvements on the challenging real-world Waymo Open dataset.

        ----

        ## [106] Understanding Oversquashing in GNNs through the Lens of Effective Resistance

        **Authors**: *Mitchell Black, Zhengchao Wan, Amir Nayyeri, Yusu Wang*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/black23a.html](https://proceedings.mlr.press/v202/black23a.html)

        **Abstract**:

        Message passing graph neural networks (GNNs) are a popular learning architectures for graph-structured data. However, one problem GNNs experience is oversquashing, where a GNN has difficulty sending information between distant nodes. Understanding and mitigating oversquashing has recently received significant attention from the research community. In this paper, we continue this line of work by analyzing oversquashing through the lens of the effective resistance between nodes in the input graph. Effective resistance intuitively captures the “strength” of connection between two nodes by paths in the graph, and has a rich literature spanning many areas of graph theory. We propose to use total effective resistance as a bound of the total amount of oversquashing in a graph and provide theoretical justification for its use. We further develop an algorithm to identify edges to be added to an input graph to minimize the total effective resistance, thereby alleviating oversquashing. We provide empirical evidence of the effectiveness of our total effective resistance based rewiring strategies for improving the performance of GNNs.

        ----

        ## [107] Unit Scaling: Out-of-the-Box Low-Precision Training

        **Authors**: *Charlie Blake, Douglas Orr, Carlo Luschi*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/blake23a.html](https://proceedings.mlr.press/v202/blake23a.html)

        **Abstract**:

        We present unit scaling, a paradigm for designing deep learning models that simplifies the use of low-precision number formats. Training in FP16 or the recently proposed FP8 formats offers substantial efficiency gains, but can lack sufficient range for out-of-the-box training. Unit scaling addresses this by introducing a principled approach to model numerics: seeking unit variance of all weights, activations and gradients at initialisation. Unlike alternative methods, this approach neither requires multiple training runs to find a suitable scale nor has significant computational overhead. We demonstrate the efficacy of unit scaling across a range of models and optimisers. We further show that existing models can be adapted to be unit-scaled, training BERT-Large in FP16 and then FP8 with no degradation in accuracy.

        ----

        ## [108] FLEX: an Adaptive Exploration Algorithm for Nonlinear Systems

        **Authors**: *Matthieu Blanke, Marc Lelarge*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/blanke23a.html](https://proceedings.mlr.press/v202/blanke23a.html)

        **Abstract**:

        Model-based reinforcement learning is a powerful tool, but collecting data to fit an accurate model of the system can be costly. Exploring an unknown environment in a sample-efficient manner is hence of great importance. However, the complexity of dynamics and the computational limitations of real systems make this task challenging. In this work, we introduce FLEX, an exploration algorithm for nonlinear dynamics based on optimal experimental design. Our policy maximizes the information of the next step and results in an adaptive exploration algorithm, compatible with arbitrary parametric learning models, and requiring minimal computing resources. We test our method on a number of nonlinear environments covering different settings, including time-varying dynamics. Keeping in mind that exploration is intended to serve an exploitation objective, we also test our algorithm on downstream model-based classical control tasks and compare it to other state-of-the-art model-based and model-free approaches. The performance achieved by FLEX is competitive and its computational cost is low.

        ----

        ## [109] Not all Strongly Rayleigh Distributions Have Small Probabilistic Generating Circuits

        **Authors**: *Markus Bläser*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/blaser23a.html](https://proceedings.mlr.press/v202/blaser23a.html)

        **Abstract**:

        Probabilistic modeling is a central task in machine learning. Probabilistic models should be tractable, i.e., allowing tractable probabilistic inference, but also efficient, i.e., being able to represent a large set of probability distributions. Zhang et al. (ICML 2021) recently proposed a new model, probabilistic generating circuits. They raised the question whether every strongly Rayleigh distribution can be efficiently represented by such circuits. We prove that this question has a negative answer, there are strongly Rayleigh distributions that cannot be represented by polynomial-sized probabilistic generating circuits, assuming a widely accepted complexity theoretic conjecture.

        ----

        ## [110] Learning the Dynamics of Sparsely Observed Interacting Systems

        **Authors**: *Linus Bleistein, Adeline Fermanian, Anne-Sophie Jannot, Agathe Guilloux*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/bleistein23a.html](https://proceedings.mlr.press/v202/bleistein23a.html)

        **Abstract**:

        We address the problem of learning the dynamics of an unknown non-parametric system linking a target and a feature time series. The feature time series is measured on a sparse and irregular grid, while we have access to only a few points of the target time series. Once learned, we can use these dynamics to predict values of the target from the previous values of the feature time series. We frame this task as learning the solution map of a controlled differential equation (CDE). By leveraging the rich theory of signatures, we are able to cast this non-linear problem as a high-dimensional linear regression. We provide an oracle bound on the prediction error which exhibits explicit dependencies on the individual-specific sampling schemes. Our theoretical results are illustrated by simulations which show that our method outperforms existing algorithms for recovering the full time series while being computationally cheap. We conclude by demonstrating its potential on real-world epidemiological data.

        ----

        ## [111] Subset Selection Based On Multiple Rankings in the Presence of Bias: Effectiveness of Fairness Constraints for Multiwinner Voting Score Functions

        **Authors**: *Niclas Boehmer, L. Elisa Celis, Lingxiao Huang, Anay Mehrotra, Nisheeth K. Vishnoi*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/boehmer23a.html](https://proceedings.mlr.press/v202/boehmer23a.html)

        **Abstract**:

        We consider the problem of subset selection where one is given multiple rankings of items and the goal is to select the highest "quality" subset. Score functions from the multiwinner voting literature have been used to aggregate rankings into quality scores for subsets. We study this setting of subset selection problems when, in addition, rankings may contain systemic or unconscious biases toward a group of items. For a general model of input rankings and biases, we show that requiring the selected subset to satisfy group fairness constraints can improve the quality of the selection with respect to unbiased rankings. Importantly, we show that for fairness constraints to be effective, different multiwinner score functions may require a drastically different number of rankings: While for some functions, fairness constraints need an exponential number of rankings to recover a close-to-optimal solution, for others, this dependency is only polynomial. This result relies on a novel notion of "smoothness" of submodular functions in this setting that quantifies how well a function can "correctly" assess the quality of items in the presence of bias. The results in this paper can be used to guide the choice of multiwinner score functions for the subset selection setting considered here; we additionally provide a tool to empirically enable this.

        ----

        ## [112] Properties of the Mallows Model Depending on the Number of Alternatives: A Warning for an Experimentalist

        **Authors**: *Niclas Boehmer, Piotr Faliszewski, Sonja Kraiczy*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/boehmer23b.html](https://proceedings.mlr.press/v202/boehmer23b.html)

        **Abstract**:

        The Mallows model is a popular distribution for ranked data. We empirically and theoretically analyze how the properties of rankings sampled from the Mallows model change when increasing the number of alternatives. We find that real-world data behaves differently from the Mallows model, yet is in line with its recent variant proposed by Boehmer et al. [IJCAI ’21]. As part of our study, we issue several warnings about using the classic Mallows model. For instance, we find that one should be extremely careful when using the Mallows model to generate data for experiments with a varying number of alternatives, as observed trends in such experiments might be due to the changing nature of the generated data.

        ----

        ## [113] A Robust Optimisation Perspective on Counterexample-Guided Repair of Neural Networks

        **Authors**: *David Boetius, Stefan Leue, Tobias Sutter*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/boetius23a.html](https://proceedings.mlr.press/v202/boetius23a.html)

        **Abstract**:

        Counterexample-guided repair aims at creating neural networks with mathematical safety guarantees, facilitating the application of neural networks in safety-critical domains. However, whether counterexample-guided repair is guaranteed to terminate remains an open question. We approach this question by showing that counterexample-guided repair can be viewed as a robust optimisation algorithm. While termination guarantees for neural network repair itself remain beyond our reach, we prove termination for more restrained machine learning models and disprove termination in a general setting. We empirically study the practical implications of our theoretical results, demonstrating the suitability of common verifiers and falsifiers for repair despite a disadvantageous theoretical result. Additionally, we use our theoretical insights to devise a novel algorithm for repairing linear regression models based on quadratic programming, surpassing existing approaches.

        ----

        ## [114] Beyond the Universal Law of Robustness: Sharper Laws for Random Features and Neural Tangent Kernels

        **Authors**: *Simone Bombari, Shayan Kiyani, Marco Mondelli*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/bombari23a.html](https://proceedings.mlr.press/v202/bombari23a.html)

        **Abstract**:

        Machine learning models are vulnerable to adversarial perturbations, and a thought-provoking paper by Bubeck and Sellke has analyzed this phenomenon through the lens of over-parameterization: interpolating smoothly the data requires significantly more parameters than simply memorizing it. However, this "universal" law provides only a necessary condition for robustness, and it is unable to discriminate between models. In this paper, we address these gaps by focusing on empirical risk minimization in two prototypical settings, namely, random features and the neural tangent kernel (NTK). We prove that, for random features, the model is not robust for any degree of over-parameterization, even when the necessary condition coming from the universal law of robustness is satisfied. In contrast, for even activations, the NTK model meets the universal lower bound, and it is robust as soon as the necessary condition on over-parameterization is fulfilled. This also addresses a conjecture in prior work by Bubeck, Li and Nagaraj. Our analysis decouples the effect of the kernel of the model from an "interaction matrix", which describes the interaction with the test data and captures the effect of the activation. Our theoretical results are corroborated by numerical evidence on both synthetic and standard datasets (MNIST, CIFAR-10).

        ----

        ## [115] Sliced-Wasserstein on Symmetric Positive Definite Matrices for M/EEG Signals

        **Authors**: *Clément Bonet, Benoît Malézieux, Alain Rakotomamonjy, Lucas Drumetz, Thomas Moreau, Matthieu Kowalski, Nicolas Courty*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/bonet23a.html](https://proceedings.mlr.press/v202/bonet23a.html)

        **Abstract**:

        When dealing with electro or magnetoencephalography records, many supervised prediction tasks are solved by working with covariance matrices to summarize the signals. Learning with these matrices requires the usage of Riemanian geometry to account for their structure. In this paper, we propose a new method to deal with distributions of covariance matrices, and demonstrate its computational efficiency on M/EEG multivariate time series. More specifically, we define a Sliced-Wasserstein distance between measures of symmetric positive definite matrices that comes with strong theoretical guarantees. Then, we take advantage of its properties and kernel methods to apply this discrepancy to brain-age prediction from MEG data, and compare it to state-of-the-art algorithms based on Riemannian geometry. Finally, we show that it is an efficient surrogate to the Wasserstein distance in domain adaptation for Brain Computer Interface applications.

        ----

        ## [116] Spherical Fourier Neural Operators: Learning Stable Dynamics on the Sphere

        **Authors**: *Boris Bonev, Thorsten Kurth, Christian Hundt, Jaideep Pathak, Maximilian Baust, Karthik Kashinath, Anima Anandkumar*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/bonev23a.html](https://proceedings.mlr.press/v202/bonev23a.html)

        **Abstract**:

        Fourier Neural Operators (FNOs) have proven to be an efficient and effective method for resolution-independent operator learning in a broad variety of application areas across scientific machine learning. A key reason for their success is their ability to accurately model long-range dependencies in spatio-temporal data by learning global convolutions in a computationally efficient manner. To this end, FNOs rely on the discrete Fourier transform (DFT), however, DFTs cause visual and spectral artifacts as well as pronounced dissipation when learning operators in spherical coordinates by incorrectly assuming flat geometry. To overcome this limitation, we generalize FNOs on the sphere, introducing Spherical FNOs (SFNOs) for learning operators on spherical geometries. We apply SFNOs to forecasting atmo- spheric dynamics, and demonstrate stable autoregressive rollouts for a year of simulated time (1,460 steps), while retaining physically plausible dynamics. The SFNO has important implications for machine learning-based simulation of climate dynamics that could eventually help accelerate our response to climate change.

        ----

        ## [117] The Regret of Exploration and the Control of Bad Episodes in Reinforcement Learning

        **Authors**: *Victor Boone, Bruno Gaujal*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/boone23a.html](https://proceedings.mlr.press/v202/boone23a.html)

        **Abstract**:

        The first contribution of this paper is the introduction of a new performance measure of a RL algorithm that is more discriminating than the regret, that we call the regret of exploration that measures the asymptotic cost of exploration. The second contribution is a new performance test (PT) to end episodes in RL optimistic algorithms. This test is based on the performance of the current policy with respect to the best policy over the current confidence set. This is in contrast with all existing RL algorithms whose episode lengths are only based on the number of visits to the states. This modification does not harm the regret and brings an additional property. We show that while all current episodic RL algorithms have a linear regret of exploration, our method has a $O(\log{T})$ regret of exploration for non-degenerate deterministic MDPs.

        ----

        ## [118] Model-agnostic Measure of Generalization Difficulty

        **Authors**: *Akhilan Boopathy, Kevin Liu, Jaedong Hwang, Shu Ge, Asaad Mohammedsaleh, Ila Fiete*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/boopathy23a.html](https://proceedings.mlr.press/v202/boopathy23a.html)

        **Abstract**:

        The measure of a machine learning algorithm is the difficulty of the tasks it can perform, and sufficiently difficult tasks are critical drivers of strong machine learning models. However, quantifying the generalization difficulty of machine learning benchmarks has remained challenging. We propose what is to our knowledge the first model-agnostic measure of the inherent generalization difficulty of tasks. Our inductive bias complexity measure quantifies the total information required to generalize well on a task minus the information provided by the data. It does so by measuring the fractional volume occupied by hypotheses that generalize on a task given that they fit the training data. It scales exponentially with the intrinsic dimensionality of the space over which the model must generalize but only polynomially in resolution per dimension, showing that tasks which require generalizing over many dimensions are drastically more difficult than tasks involving more detail in fewer dimensions. Our measure can be applied to compute and compare supervised learning, reinforcement learning and meta-learning generalization difficulties against each other. We show that applied empirically, it formally quantifies intuitively expected trends, e.g. that in terms of required inductive bias, MNIST $<$ CIFAR10 $<$ Imagenet and fully observable Markov decision processes (MDPs) $<$ partially observable MDPs. Further, we show that classification of complex images $<$ few-shot meta-learning with simple images. Our measure provides a quantitative metric to guide the construction of more complex tasks requiring greater inductive bias, and thereby encourages the development of more sophisticated architectures and learning algorithms with more powerful generalization capabilities.

        ----

        ## [119] Returning The Favour: When Regression Benefits From Probabilistic Causal Knowledge

        **Authors**: *Shahine Bouabid, Jake Fawkes, Dino Sejdinovic*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/bouabid23a.html](https://proceedings.mlr.press/v202/bouabid23a.html)

        **Abstract**:

        A directed acyclic graph (DAG) provides valuable prior knowledge that is often discarded in regression tasks in machine learning. We show that the independences arising from the presence of collider structures in DAGs provide meaningful inductive biases, which constrain the regression hypothesis space and improve predictive performance. We introduce collider regression, a framework to incorporate probabilistic causal knowledge from a collider in a regression problem. When the hypothesis space is a reproducing kernel Hilbert space, we prove a strictly positive generalisation benefit under mild assumptions and provide closed-form estimators of the empirical risk minimiser. Experiments on synthetic and climate model data demonstrate performance gains of the proposed methodology.

        ----

        ## [120] In Search for a Generalizable Method for Source Free Domain Adaptation

        **Authors**: *Malik Boudiaf, Tom Denton, Bart van Merrienboer, Vincent Dumoulin, Eleni Triantafillou*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/boudiaf23a.html](https://proceedings.mlr.press/v202/boudiaf23a.html)

        **Abstract**:

        Source-free domain adaptation (SFDA) is compelling because it allows adapting an off-the-shelf model to a new domain using only unlabelled data. In this work, we apply existing SFDA techniques to a challenging set of naturally-occurring distribution shifts in bioacoustics, which are very different from the ones commonly studied in computer vision. We find existing methods perform differently relative to each other than observed in vision benchmarks, and sometimes perform worse than no adaptation at all. We propose a new simple method which outperforms the existing methods on our new shifts while exhibiting strong performance on a range of vision datasets. Our findings suggest that existing SFDA methods are not as generalizable as previously thought and that considering diverse modalities can be a useful avenue for designing more robust models.

        ----

        ## [121] Quantum Speedups for Zero-Sum Games via Improved Dynamic Gibbs Sampling

        **Authors**: *Adam Bouland, Yosheb M. Getachew, Yujia Jin, Aaron Sidford, Kevin Tian*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/bouland23a.html](https://proceedings.mlr.press/v202/bouland23a.html)

        **Abstract**:

        We give a quantum algorithm for computing an $\epsilon$-approximate Nash equilibrium of a zero-sum game in a $m \times n$ payoff matrix with bounded entries. Given a standard quantum oracle for accessing the payoff matrix our algorithm runs in time $\widetilde{O}(\sqrt{m + n}\cdot \epsilon^{-2.5} + \epsilon^{-3})$ and outputs a classical representation of the $\epsilon$-approximate Nash equilibrium. This improves upon the best prior quantum runtime of $\widetilde{O}(\sqrt{m + n} \cdot \epsilon^{-3})$ obtained by [van Apeldoorn, Gilyen ’19] and the classical $\widetilde{O}((m + n) \cdot \epsilon^{-2})$ runtime due to [Grigoradis, Khachiyan ’95] whenever $\epsilon = \Omega((m +n)^{-1})$. We obtain this result by designing new quantum data structures for efficiently sampling from a slowly-changing Gibbs distribution.

        ----

        ## [122] Diffusion Models as Artists: Are we Closing the Gap between Humans and Machines?

        **Authors**: *Victor Boutin, Thomas Fel, Lakshya Singhal, Rishav Mukherji, Akash Nagaraj, Julien Colin, Thomas Serre*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/boutin23a.html](https://proceedings.mlr.press/v202/boutin23a.html)

        **Abstract**:

        An important milestone for AI is the development of algorithms that can produce drawings that are indistinguishable from those of humans. Here, we adapt the ”diversity vs. recognizability” scoring framework from Boutin et al (2022) and find that one-shot diffusion models have indeed started to close the gap between humans and machines. However, using a finer-grained measure of the originality of individual samples, we show that strengthening the guidance of diffusion models helps improve the humanness of their drawings, but they still fall short of approximating the originality and recognizability of human drawings. Comparing human category diagnostic features, collected through an online psychophysics experiment, against those derived from diffusion models reveals that humans rely on fewer and more localized features. Overall, our study suggests that diffusion models have significantly helped improve the quality of machine-generated drawings; however, a gap between humans and machines remains – in part explainable by discrepancies in visual strategies.

        ----

        ## [123] Settling the Reward Hypothesis

        **Authors**: *Michael Bowling, John D. Martin, David Abel, Will Dabney*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/bowling23a.html](https://proceedings.mlr.press/v202/bowling23a.html)

        **Abstract**:

        The reward hypothesis posits that, "all of what we mean by goals and purposes can be well thought of as maximization of the expected value of the cumulative sum of a received scalar signal (reward)." We aim to fully settle this hypothesis. This will not conclude with a simple affirmation or refutation, but rather specify completely the implicit requirements on goals and purposes under which the hypothesis holds.

        ----

        ## [124] ILLUME: Rationalizing Vision-Language Models through Human Interactions

        **Authors**: *Manuel Brack, Patrick Schramowski, Björn Deiseroth, Kristian Kersting*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/brack23a.html](https://proceedings.mlr.press/v202/brack23a.html)

        **Abstract**:

        Bootstrapping from pre-trained language models has been proven to be an efficient approach for building vision-language models (VLM) for tasks such as image captioning or visual question answering. However, outputs of these models rarely align with user’s rationales for specific answers. In order to improve this alignment and reinforce commonsense reasons, we propose a tuning paradigm based on human interactions with machine-generated data. Our ILLUME executes the following loop: Given an image-question-answer prompt, the VLM samples multiple candidate rationales, and a human critic provides feedback via preference selection, used for fine-tuning. This loop increases the training data and gradually carves out the VLM’s rationalization capabilities that are aligned with human intent. Our exhaustive experiments demonstrate that ILLUME is competitive with standard supervised finetuning while using significantly fewer training data and only requiring minimal feedback.

        ----

        ## [125] Provably Learning Object-Centric Representations

        **Authors**: *Jack Brady, Roland S. Zimmermann, Yash Sharma, Bernhard Schölkopf, Julius von Kügelgen, Wieland Brendel*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/brady23a.html](https://proceedings.mlr.press/v202/brady23a.html)

        **Abstract**:

        Learning structured representations of the visual world in terms of objects promises to significantly improve the generalization abilities of current machine learning models. While recent efforts to this end have shown promising empirical progress, a theoretical account of when unsupervised object-centric representation learning is possible is still lacking. Consequently, understanding the reasons for the success of existing object-centric methods as well as designing new theoretically grounded methods remains challenging. In the present work, we analyze when object-centric representations can provably be learned without supervision. To this end, we first introduce two assumptions on the generative process for scenes comprised of several objects, which we call compositionality and irreducibility. Under this generative process, we prove that the ground-truth object representations can be identified by an invertible and compositional inference model, even in the presence of dependencies between objects. We empirically validate our results through experiments on synthetic data. Finally, we provide evidence that our theory holds predictive power for existing object-centric models by showing a close correspondence between models’ compositionality and invertibility and their empirical identifiability.

        ----

        ## [126] Quantifying Human Priors over Social and Navigation Networks

        **Authors**: *Gecia Bravo Hermsdorff*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/bravo-hermsdorff23a.html](https://proceedings.mlr.press/v202/bravo-hermsdorff23a.html)

        **Abstract**:

        Human knowledge is largely implicit and relational — do we have a friend in common? can I walk from here to there? In this work, we leverage the combinatorial structure of graphs to quantify human priors over such relational data. Our experiments focus on two domains that have been continuously relevant over evolutionary timescales: social interaction and spatial navigation. We find that some features of the inferred priors are remarkably consistent, such as the tendency for sparsity as a function of graph size. Other features are domain-specific, such as the propensity for triadic closure in social interactions. More broadly, our work demonstrates how nonclassical statistical analysis of indirect behavioral experiments can be used to efficiently model latent biases in the data.

        ----

        ## [127] Critical Points and Convergence Analysis of Generative Deep Linear Networks Trained with Bures-Wasserstein Loss

        **Authors**: *Pierre Bréchet, Katerina Papagiannouli, Jing An, Guido Montúfar*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/brechet23a.html](https://proceedings.mlr.press/v202/brechet23a.html)

        **Abstract**:

        We consider a deep matrix factorization model of covariance matrices trained with the Bures-Wasserstein distance. While recent works have made advances in the study of the optimization problem for overparametrized low-rank matrix approximation, much emphasis has been placed on discriminative settings and the square loss. In contrast, our model considers another type of loss and connects with the generative setting. We characterize the critical points and minimizers of the Bures-Wasserstein distance over the space of rank-bounded matrices. The Hessian of this loss at low-rank matrices can theoretically blow up, which creates challenges to analyze convergence of gradient optimization methods. We establish convergence results for gradient flow using a smooth perturbative version of the loss as well as convergence results for finite step size gradient descent under certain assumptions on the initial weights.

        ----

        ## [128] Emergence of Sparse Representations from Noise

        **Authors**: *Trenton Bricken, Rylan Schaeffer, Bruno A. Olshausen, Gabriel Kreiman*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/bricken23a.html](https://proceedings.mlr.press/v202/bricken23a.html)

        **Abstract**:

        A hallmark of biological neural networks, which distinguishes them from their artificial counterparts, is the high degree of sparsity in their activations. This discrepancy raises three questions our work helps to answer: (i) Why are biological networks so sparse? (ii) What are the benefits of this sparsity? (iii) How can these benefits be utilized by deep learning models? Our answers to all of these questions center around training networks to handle random noise. Surprisingly, we discover that noisy training introduces three implicit loss terms that result in sparsely firing neurons specializing to high variance features of the dataset. When trained to reconstruct noisy-CIFAR10, neurons learn biological receptive fields. More broadly, noisy training presents a new approach to potentially increase model interpretability with additional benefits to robustness and computational efficiency.

        ----

        ## [129] Differentially Private Optimization on Large Model at Small Cost

        **Authors**: *Zhiqi Bu, Yu-Xiang Wang, Sheng Zha, George Karypis*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/bu23a.html](https://proceedings.mlr.press/v202/bu23a.html)

        **Abstract**:

        Differentially private (DP) optimization is the standard paradigm to learn large neural networks that are accurate and privacy-preserving. The computational cost for DP deep learning, however, is notoriously heavy due to the per-sample gradient clipping. Existing DP implementations are 2$\sim$1000$\times$ more costly in time and space complexity than the standard (non-private) training. In this work, we develop a novel Book-Keeping (BK) technique that implements existing DP optimizers (thus achieving the same accuracy), with a substantial improvement on the computational cost. Specifically, BK enables DP training on large models and high dimensional data to be roughly as fast and memory-saving as the standard training, whereas previous DP algorithms can be inefficient or incapable of training due to memory error. The computational advantage of BK is supported by the complexity analysis as well as extensive experiments on vision and language tasks. Our implementation achieves state-of-the-art (SOTA) accuracy with very small extra cost: on GPT2 and at almost the same memory cost ($<$1% overhead), BK has 1.03$\times$ the time complexity of the standard training (0.83$\times$ training speed in practice), and 0.61$\times$ the time complexity of the most efficient DP implementation (1.36$\times$ training speed in practice). We open-source the codebase for the BK algorithm at https://github.com/awslabs/fast-differential-privacy.

        ----

        ## [130] Machine Learning Force Fields with Data Cost Aware Training

        **Authors**: *Alexander Bukharin, Tianyi Liu, Shengjie Wang, Simiao Zuo, Weihao Gao, Wen Yan, Tuo Zhao*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/bukharin23a.html](https://proceedings.mlr.press/v202/bukharin23a.html)

        **Abstract**:

        Machine learning force fields (MLFF) have been proposed to accelerate molecular dynamics (MD) simulation, which finds widespread applications in chemistry and biomedical research. Even for the most data-efficient MLFFs, reaching chemical accuracy can require hundreds of frames of force and energy labels generated by expensive quantum mechanical algorithms, which may scale as $O(n^3)$ to $O(n^7)$, with $n$ proportional to the number of basis functions. To address this issue, we propose a multi-stage computational framework – ASTEROID, which lowers the data cost of MLFFs by leveraging a combination of cheap inaccurate data and expensive accurate data. The motivation behind ASTEROID is that inaccurate data, though incurring large bias, can help capture the sophisticated structures of the underlying force field. Therefore, we first train a MLFF model on a large amount of inaccurate training data, employing a bias-aware loss function to prevent the model from overfitting the potential bias of this data. We then fine-tune the obtained model using a small amount of accurate training data, which preserves the knowledge learned from the inaccurate training data while significantly improving the model’s accuracy. Moreover, we propose a variant of ASTEROID based on score matching for the setting where the inaccurate training data are unlabeled. Extensive experiments on MD datasets and downstream tasks validate the efficacy of ASTEROID. Our code and data are available at https://github.com/abukharin3/asteroid.

        ----

        ## [131] Label differential privacy and private training data release

        **Authors**: *Róbert Istvan Busa-Fekete, Andrés Muñoz Medina, Umar Syed, Sergei Vassilvitskii*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/busa-fekete23a.html](https://proceedings.mlr.press/v202/busa-fekete23a.html)

        **Abstract**:

        We study differentially private mechanisms for sharing training data in machine learning settings. Our goal is to enable learning of an accurate predictive model while protecting the privacy of each user’s label. Previous work established privacy guarantees that assumed the features are public and given exogenously, a setting known as label differential privacy. In some scenarios, this can be a strong assumption that removes the interplay between features and labels from the privacy analysis. We relax this approach and instead assume the features are drawn from a distribution that depends on the private labels. We first show that simply adding noise to the label, as in previous work, can lead to an arbitrarily weak privacy guarantee, and also present methods for estimating this privacy loss from data. We then present a new mechanism that replaces some training examples with synthetically generated data, and show that our mechanism has a much better privacy-utility tradeoff if the synthetic data is ‘realistic’, in a certain quantifiable sense. Finally, we empirically validate our theoretical analysis.

        ----

        ## [132] The SSL Interplay: Augmentations, Inductive Bias, and Generalization

        **Authors**: *Vivien Cabannes, Bobak Toussi Kiani, Randall Balestriero, Yann LeCun, Alberto Bietti*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/cabannes23a.html](https://proceedings.mlr.press/v202/cabannes23a.html)

        **Abstract**:

        Self-supervised learning (SSL) has emerged as a powerful framework to learn representations from raw data without supervision. Yet in practice, engineers face issues such as instability in tuning optimizers and collapse of representations during training. Such challenges motivate the need for a theory to shed light on the complex interplay between the choice of data augmentation, network architecture, and training algorithm. % on the resulting performance in downstream tasks. We study such an interplay with a precise analysis of generalization performance on both pretraining and downstream tasks in kernel regimes, and highlight several insights for SSL practitioners that arise from our theory.

        ----

        ## [133] Online Mechanism Design for Information Acquisition

        **Authors**: *Federico Cacciamani, Matteo Castiglioni, Nicola Gatti*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/cacciamani23a.html](https://proceedings.mlr.press/v202/cacciamani23a.html)

        **Abstract**:

        We study the problem of designing mechanisms for information acquisition scenarios. This setting models strategic interactions between a uniformed receiver and a set of informed senders. In our model the senders receive information about the underlying state of nature and communicate their observation (either truthfully or not) to the receiver, which, based on this information, selects an action. Our goal is to design mechanisms maximizing the receiver’s utility while incentivizing the senders to report truthfully their information. First, we provide an algorithm that efficiently computes an optimal incentive compatible (IC) mechanism. Then, we focus on the online problem in which the receiver sequentially interacts in an unknown game, with the objective of minimizing the cumulative regret w.r.t. the optimal IC mechanism, and the cumulative violation of the incentive compatibility constraints. We investigate two different online scenarios, i.e., the full and bandit feedback settings. For the full feedback problem, we propose an algorithm that guarantees $\tilde{O}(\sqrt{T})$ regret and violation, while for the bandit feedback setting we present an algorithm that attains $\tilde{O}(T^{\alpha})$ regret and $\tilde{O}(T^{1-\alpha/2})$ violation for any $\alpha \in [1/2, 1]$. Finally, we complement our results providing a tight lower bound.

        ----

        ## [134] MyoDex: A Generalizable Prior for Dexterous Manipulation

        **Authors**: *Vittorio Caggiano, Sudeep Dasari, Vikash Kumar*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/caggiano23a.html](https://proceedings.mlr.press/v202/caggiano23a.html)

        **Abstract**:

        Human dexterity is a hallmark of motor control behaviors. Our hands can rapidly synthesize new behaviors despite the complexity (multi-articular and multi-joints, with 23 joints controlled by more than 40 muscles) of mosculoskeletal control. In this work, we take inspiration from how human dexterity builds on a diversity of prior experiences, instead of being acquired through a single task. Motivated by this observation, we set out to develop agents that can build upon previous experience to quickly acquire new (previously unattainable) behaviors. Specifically, our approach leverages multi-task learning to implicitly capture a task-agnostic behavioral priors (MyoDex) for human-like dexterity, using a physiologically realistic human hand model – MyoHand. We demonstrate MyoDex’s effectiveness in few-shot generalization as well as positive transfer to a large repertoire of unseen dexterous manipulation tasks. MyoDex can solve approximately 3x more tasks and it can accelerate the achievement of solutions by about 4x in comparison to a distillation baseline. While prior work has synthesized single musculoskeletal control behaviors, MyoDex is the first generalizable manipulation prior that catalyzes the learning of dexterous physiological control across a large variety of contact-rich behaviors.

        ----

        ## [135] What Can Be Learnt With Wide Convolutional Neural Networks?

        **Authors**: *Francesco Cagnetta, Alessandro Favero, Matthieu Wyart*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/cagnetta23a.html](https://proceedings.mlr.press/v202/cagnetta23a.html)

        **Abstract**:

        Understanding how convolutional neural networks (CNNs) can efficiently learn high-dimensional functions remains a fundamental challenge. A popular belief is that these models harness the local and hierarchical structure of natural data such as images. Yet, we lack a quantitative understanding of how such structure affects performance, e.g., the rate of decay of the generalisation error with the number of training samples. In this paper, we study infinitely-wide deep CNNs in the kernel regime. First, we show that the spectrum of the corresponding kernel inherits the hierarchical structure of the network, and we characterise its asymptotics. Then, we use this result together with generalisation bounds to prove that deep CNNs adapt to the spatial scale of the target function. In particular, we find that if the target function depends on low-dimensional subsets of adjacent input variables, then the decay of the error is controlled by the effective dimensionality of these subsets. Conversely, if the target function depends on the full set of input variables, then the error decay is controlled by the input dimension. We conclude by computing the generalisation error of a deep CNN trained on the output of another deep CNN with randomly-initialised parameters. Interestingly, we find that, despite their hierarchical structure, the functions generated by infinitely-wide deep CNNs are too rich to be efficiently learnable in high dimension.

        ----

        ## [136] Causal Discovery with Latent Confounders Based on Higher-Order Cumulants

        **Authors**: *Ruichu Cai, Zhiyi Huang, Wei Chen, Zhifeng Hao, Kun Zhang*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/cai23a.html](https://proceedings.mlr.press/v202/cai23a.html)

        **Abstract**:

        Causal discovery with latent confounders is an important but challenging task in many scientific areas. Despite the success of some overcomplete independent component analysis (OICA) based methods in certain domains, they are computationally expensive and can easily get stuck into local optima. We notice that interestingly, by making use of higher-order cumulants, there exists a closed-form solution to OICA in specific cases, e.g., when the mixing procedure follows the One-Latent-Component structure. In light of the power of the closed-form solution to OICA corresponding to the One-Latent-Component structure, we formulate a way to estimate the mixing matrix using the higher-order cumulants, and further propose the testable One-Latent-Component condition to identify the latent variables and determine causal orders. By iteratively removing the share identified latent components, we successfully extend the results on the One-Latent-Component structure to the Multi-Latent-Component structure and finally provide a practical and asymptotically correct algorithm to learn the causal structure with latent variables. Experimental results illustrate the asymptotic correctness and effectiveness of the proposed method.

        ----

        ## [137] On the Connection Between MPNN and Graph Transformer

        **Authors**: *Chen Cai, Truong Son Hy, Rose Yu, Yusu Wang*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/cai23b.html](https://proceedings.mlr.press/v202/cai23b.html)

        **Abstract**:

        Graph Transformer (GT) recently has emerged as a new paradigm of graph learning algorithms, outperforming the previously popular Message Passing Neural Network (MPNN) on multiple benchmarks. Previous work shows that with proper position embedding, GT can approximate MPNN arbitrarily well, implying that GT is at least as powerful as MPNN. In this paper, we study the inverse connection and show that MPNN with virtual node (VN), a commonly used heuristic with little theoretical understanding, is powerful enough to arbitrarily approximate the self-attention layer of GT. In particular, we first show that if we consider one type of linear transformer, the so-called Performer/Linear Transformer, then MPNN + VN with only $\mathcal{O}(1)$ depth and $\mathcal{O}(1)$ width can approximate a self-attention layer in Performer/Linear Transformer. Next, via a connection between MPNN + VN and DeepSets, we prove the MPNN + VN with $\mathcal{O}(n^d)$ width and $\mathcal{O}(1)$ depth can approximate the self-attention layer arbitrarily well, where $d$ is the input feature dimension. Lastly, under some assumptions, we provide an explicit construction of MPNN + VN with $\mathcal{O}(1)$ width and $\mathcal{O}(n)$ depth approximating the self-attention layer in GT arbitrarily well. On the empirical side, we demonstrate that 1) MPNN + VN is a surprisingly strong baseline, outperforming GT on the recently proposed Long Range Graph Benchmark (LRGB) dataset, 2) our MPNN + VN improves over early implementation on a wide range of OGB datasets and 3) MPNN + VN outperforms Linear Transformer and MPNN on the climate modeling task.

        ----

        ## [138] Ske2Grid: Skeleton-to-Grid Representation Learning for Action Recognition

        **Authors**: *Dongqi Cai, Yangyuxuan Kang, Anbang Yao, Yurong Chen*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/cai23c.html](https://proceedings.mlr.press/v202/cai23c.html)

        **Abstract**:

        This paper presents Ske2Grid, a new representation learning framework for improved skeleton-based action recognition. In Ske2Grid, we define a regular convolution operation upon a novel grid representation of human skeleton, which is a compact image-like grid patch constructed and learned through three novel designs. Specifically, we propose a graph-node index transform (GIT) to construct a regular grid patch through assigning the nodes in the skeleton graph one by one to the desired grid cells. To ensure that GIT is a bijection and enrich the expressiveness of the grid representation, an up-sampling transform (UPT) is learned to interpolate the skeleton graph nodes for filling the grid patch to the full. To resolve the problem when the one-step UPT is aggressive and further exploit the representation capability of the grid patch with increasing spatial size, a progressive learning strategy (PLS) is proposed which decouples the UPT into multiple steps and aligns them to multiple paired GITs through a compact cascaded design learned progressively. We construct networks upon prevailing graph convolution networks and conduct experiments on six mainstream skeleton-based action recognition datasets. Experiments show that our Ske2Grid significantly outperforms existing GCN-based solutions under different benchmark settings, without bells and whistles. Code and models are available at https://github.com/OSVAI/Ske2Grid.

        ----

        ## [139] Extrapolated Random Tree for Regression

        **Authors**: *Yuchao Cai, Yuheng Ma, Yiwei Dong, Hanfang Yang*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/cai23d.html](https://proceedings.mlr.press/v202/cai23d.html)

        **Abstract**:

        In this paper, we propose a novel tree-based algorithm named Extrapolated Random Tree for Regression (ERTR) that adapts to arbitrary smoothness of the regression function while maintaining the interpretability of the tree. We first put forward the homothetic random tree for regression (HRTR) that converges to the target function as the homothetic ratio approaches zero. Then ERTR uses a linear regression model to extrapolate HRTR estimations with different ratios to the ratio zero. From the theoretical perspective, we for the first time establish the optimal convergence rates for ERTR when the target function resides in the general Hölder space $C^{k,\alpha}$ for $k\in \mathbb{N}$, whereas the lower bound of the convergence rate of the random tree for regression (RTR) is strictly slower than ERTR in the space $C^{k,\alpha}$ for $k\geq 1$. This shows that ERTR outperforms RTR for the target function with high-order smoothness due to the extrapolation. In the experiments, we compare ERTR with state-of-the-art tree algorithms on real datasets to show the superior performance of our model. Moreover, promising improvements are brought by using the extrapolated trees as base learners in the extension of ERTR to ensemble methods.

        ----

        ## [140] Cyclic Block Coordinate Descent With Variance Reduction for Composite Nonconvex Optimization

        **Authors**: *Xufeng Cai, Chaobing Song, Stephen J. Wright, Jelena Diakonikolas*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/cai23e.html](https://proceedings.mlr.press/v202/cai23e.html)

        **Abstract**:

        Nonconvex optimization is central in solving many machine learning problems, in which block-wise structure is commonly encountered. In this work, we propose cyclic block coordinate methods for nonconvex optimization problems with non-asymptotic gradient norm guarantees. Our convergence analysis is based on a gradient Lipschitz condition with respect to a Mahalanobis norm, inspired by a recent progress on cyclic block coordinate methods. In deterministic settings, our convergence guarantee matches the guarantee of (full-gradient) gradient descent, but with the gradient Lipschitz constant being defined w.r.t. a Mahalanobis norm. In stochastic settings, we use recursive variance reduction to decrease the per-iteration cost and match the arithmetic operation complexity of current optimal stochastic full-gradient methods, with a unified analysis for both finite-sum and infinite-sum cases. We prove a faster linear convergence result when a Polyak-Łojasiewicz (PŁ) condition holds. To our knowledge, this work is the first to provide non-asymptotic convergence guarantees — variance-reduced or not — for a cyclic block coordinate method in general composite (smooth + nonsmooth) nonconvex settings. Our experimental results demonstrate the efficacy of the proposed cyclic scheme in training deep neural nets.

        ----

        ## [141] Robust Weight Signatures: Gaining Robustness as Easy as Patching Weights?

        **Authors**: *Ruisi Cai, Zhenyu Zhang, Zhangyang Wang*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/cai23f.html](https://proceedings.mlr.press/v202/cai23f.html)

        **Abstract**:

        Given a robust model trained to be resilient to one or multiple types of distribution shifts (e.g., natural image corruptions), how is that "robustness" encoded in the model weights, and how easily can it be disentangled and/or "zero-shot" transferred to some other models? This paper empirically suggests a surprisingly simple answer: linearly - by straightforward model weight arithmetic! We start by drawing several key observations: (i) assuming that we train the same model architecture on both a clean dataset and its corrupted version, a comparison between the two resultant models shows their weights to mostly differ in shallow layers; (ii) the weight difference after projection, which we call "Robust Weight Signature" (RWS), appears to be discriminative and indicative of different corruption types; (iii) perhaps most strikingly, for the same corruption type, the RWSs obtained by one model architecture are highly consistent and transferable across different datasets. Based on those RWS observations, we propose a minimalistic model robustness "patching" framework that carries a model trained on clean data together with its pre-extracted RWSs. In this way, injecting certain robustness to the model is reduced to directly adding the corresponding RWS to its weight. We experimentally verify our proposed framework to be remarkably (1) lightweight. since RWSs concentrate on the shallowest few layers and we further show they can be painlessly quantized, storing an RWS is up to 13 x more compact than storing the full weight copy; (2) in-situ adjustable. RWSs can be appended as needed and later taken off to restore the intact clean model. We further demonstrate one can linearly re-scale the RWS to control the patched robustness strength; (3) composable. Multiple RWSs can be added simultaneously to patch more comprehensive robustness at once; and (4) transferable. Even when the clean model backbone is continually adapted or updated, RWSs remain as effective patches due to their outstanding cross-dataset transferability.

        ----

        ## [142] Doubly Optimal No-Regret Learning in Monotone Games

        **Authors**: *Yang Cai, Weiqiang Zheng*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/cai23g.html](https://proceedings.mlr.press/v202/cai23g.html)

        **Abstract**:

        We consider online learning in multi-player smooth monotone games. Existing algorithms have limitations such as (1) being only applicable to strongly monotone games; (2) lacking the no-regret guarantee; (3) having only asymptotic or slow $\mathcal{O}(\frac{1}{\sqrt{T}})$ last-iterate convergence rate to a Nash equilibrium. While the $\mathcal{O}(\frac{1}{\sqrt{T}})$ rate is tight for a large class of algorithms including the well-studied extragradient algorithm and optimistic gradient algorithm, it is not optimal for all gradient-based algorithms. We propose the accelerated optimistic gradient (AOG) algorithm, the first doubly optimal no-regret learning algorithm for smooth monotone games. Namely, our algorithm achieves both (i) the optimal $\mathcal{O}(\sqrt{T})$ regret in the adversarial setting under smooth and convex loss functions and (ii) the optimal $\mathcal{O}(\frac{1}{T})$ last-iterate convergence rate to a Nash equilibrium in multi-player smooth monotone games. As a byproduct of the accelerated last-iterate convergence rate, we further show that each player suffers only an $\mathcal{O}(\log T)$ individual worst-case dynamic regret, providing an exponential improvement over the previous state-of-the-art $\mathcal{O}(\sqrt{T})$ bound.

        ----

        ## [143] Multi-Agent Learning from Learners

        **Authors**: *Mine Melodi Caliskan, Francesco Chini, Setareh Maghsudi*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/caliskan23a.html](https://proceedings.mlr.press/v202/caliskan23a.html)

        **Abstract**:

        A large body of the "Inverse Reinforcement Learning" (IRL) literature focuses on recovering the reward function from a set of demonstrations of an expert agent who acts optimally or noisily optimally. Nevertheless, some recent works move away from the optimality assumption to study the "Learning from a Learner (LfL)" problem, where the challenge is inferring the reward function of a learning agent from a sequence of demonstrations produced by progressively improving policies. In this work, we take one of the initial steps in addressing the multi-agent version of this problem and propose a new algorithm, MA-LfL (Multiagent Learning from a Learner). Unlike the state-of-the-art literature, which recovers the reward functions from trajectories produced by agents in some equilibrium, we study the problem of inferring the reward functions of interacting agents in a general sum stochastic game without assuming any equilibrium state. The MA-LfL algorithm is rigorously built on a theoretical result that ensures its validity in the case of agents learning according to a multi-agent soft policy iteration scheme. We empirically test MA-LfL and we observe high positive correlation between the recovered reward functions and the ground truth.

        ----

        ## [144] Efficient Learning of Mesh-Based Physical Simulation with Bi-Stride Multi-Scale Graph Neural Network

        **Authors**: *Yadi Cao, Menglei Chai, Minchen Li, Chenfanfu Jiang*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/cao23a.html](https://proceedings.mlr.press/v202/cao23a.html)

        **Abstract**:

        Learning the long-range interactions on large-scale mesh-based physical systems with flat Graph Neural Networks (GNNs) and stacking Message Passings (MPs) is challenging due to the scaling complexity w.r.t. the number of nodes and over-smoothing. Therefore, there has been growing interest in the community to introduce multi-scale structures to GNNs for physics simulation. However, current state-of-the-art methods are limited by their reliance on the labor-heavy drawing of coarser meshes or building coarser levels based on spatial proximity, which can introduce wrong edges across geometry boundaries. Inspired by the bipartite graph determination, we propose a novel pooling strategy, bi-stride to tackle the aforementioned limitations. Bi-stride pools nodes on every other frontier of the Breadth-First-Search (BFS), without the need for the manual drawing of coarser meshes and, avoid wrong edges introduced by spatial proximity. Additionally, it enables a reduced number of MP times on each level and the non-parametrized pooling and unpooling by interpolations, similar to convolutional Neural Networks (CNNs), which significantly reduces computational requirements. Experiments show that the proposed framework, BSMS-GNN, significantly outperforms existing methods in terms of both accuracy and computational efficiency in representative physics-based simulation scenarios.

        ----

        ## [145] Variational Sparse Inverse Cholesky Approximation for Latent Gaussian Processes via Double Kullback-Leibler Minimization

        **Authors**: *Jian Cao, Myeongjong Kang, Felix Jimenez, Huiyan Sang, Florian Tobias Schaefer, Matthias Katzfuss*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/cao23b.html](https://proceedings.mlr.press/v202/cao23b.html)

        **Abstract**:

        To achieve scalable and accurate inference for latent Gaussian processes, we propose a variational approximation based on a family of Gaussian distributions whose covariance matrices have sparse inverse Cholesky (SIC) factors. We combine this variational approximation of the posterior with a similar and efficient SIC-restricted Kullback-Leibler-optimal approximation of the prior. We then focus on a particular SIC ordering and nearest-neighbor-based sparsity pattern resulting in highly accurate prior and posterior approximations. For this setting, our variational approximation can be computed via stochastic gradient descent in polylogarithmic time per iteration. We provide numerical comparisons showing that the proposed double-Kullback-Leibler-optimal Gaussian-process approximation (DKLGP) can sometimes be vastly more accurate for stationary kernels than alternative approaches such as inducing-point and mean-field approximations at similar computational complexity.

        ----

        ## [146] Learning Lightweight Object Detectors via Multi-Teacher Progressive Distillation

        **Authors**: *Shengcao Cao, Mengtian Li, James Hays, Deva Ramanan, Yu-Xiong Wang, Liangyan Gui*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/cao23c.html](https://proceedings.mlr.press/v202/cao23c.html)

        **Abstract**:

        Resource-constrained perception systems such as edge computing and vision-for-robotics require vision models to be both accurate and lightweight in computation and memory usage. While knowledge distillation is a proven strategy to enhance the performance of lightweight classification models, its application to structured outputs like object detection and instance segmentation remains a complicated task, due to the variability in outputs and complex internal network modules involved in the distillation process. In this paper, we propose a simple yet surprisingly effective sequential approach to knowledge distillation that progressively transfers the knowledge of a set of teacher detectors to a given lightweight student. To distill knowledge from a highly accurate but complex teacher model, we construct a sequence of teachers to help the student gradually adapt. Our progressive strategy can be easily combined with existing detection distillation mechanisms to consistently maximize student performance in various settings. To the best of our knowledge, we are the first to successfully distill knowledge from Transformer-based teacher detectors to convolution-based students, and unprecedentedly boost the performance of ResNet-50 based RetinaNet from 36.5% to 42.0% AP and Mask R-CNN from 38.2% to 42.5% AP on the MS COCO benchmark. Code available at https://github.com/Shengcao-Cao/MTPD.

        ----

        ## [147] One-sided Matrix Completion from Two Observations Per Row

        **Authors**: *Steven Cao, Percy Liang, Gregory Valiant*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/cao23d.html](https://proceedings.mlr.press/v202/cao23d.html)

        **Abstract**:

        Given only a few observed entries from a low-rank matrix $X$, matrix completion is the problem of imputing the missing entries, and it formalizes a wide range of real-world settings that involve estimating missing data. However, when there are too few observed entries to complete the matrix, what other aspects of the underlying matrix can be reliably recovered? We study one such problem setting, that of “one-sided” matrix completion, where our goal is to recover the right singular vectors of $X$, even in the regime where recovering the left singular vectors is impossible, which arises when there are more rows than columns and very few observations. We propose a natural algorithm that involves imputing the missing values of the matrix $X^TX$ and show that even with only two observations per row in $X$, we can provably recover $X^TX$ as long as we have at least $\Omega(r^2 d \log d)$ rows, where $r$ is the rank and $d$ is the number of columns. We evaluate our algorithm on one-sided recovery of synthetic data and low-coverage genome sequencing. In these settings, our algorithm substantially outperforms standard matrix completion and a variety of direct factorization methods.

        ----

        ## [148] State and parameter learning with PARIS particle Gibbs

        **Authors**: *Gabriel Cardoso, Yazid Janati El Idrissi, Sylvain Le Corff, Eric Moulines, Jimmy Olsson*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/cardoso23a.html](https://proceedings.mlr.press/v202/cardoso23a.html)

        **Abstract**:

        Non-linear state-space models, also known as general hidden Markov models (HMM), are ubiquitous in statistical machine learning, being the most classical generative models for serial data and sequences. Learning in HMM, either via Maximum Likelihood Estimation (MLE) or Markov Score Climbing (MSC) requires the estimation of the- smoothing expectation of some additive functionals. Controlling the bias and the variance of this estimation is crucial to establish the convergence of learning algorithms. Our first contribution is to design a novel additive smoothing algorithm, the Parisian particle Gibbs (PPG) sampler, which can be viewed as a PaRIS (Olsson, Westerborn 2017) algorithm driven by conditional SMC moves, resulting in bias-reduced estimates of the targeted quantities. We substantiate the PPG algorithm with theoretical results, including new bounds on bias and variance as well as deviation inequalities. We then establish, in the learning context, and under standard assumptions, non-asymptotic bounds highlighting the value of bias reduction and the implicit Rao–Blackwellization of PPG. These are the first non-asymptotic results of this kind in this setting. We illustrate our theoretical results with numerical experiments supporting our claims.

        ----

        ## [149] Grounding Large Language Models in Interactive Environments with Online Reinforcement Learning

        **Authors**: *Thomas Carta, Clément Romac, Thomas Wolf, Sylvain Lamprier, Olivier Sigaud, Pierre-Yves Oudeyer*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/carta23a.html](https://proceedings.mlr.press/v202/carta23a.html)

        **Abstract**:

        Recent works successfully leveraged Large Language Models’ (LLM) abilities to capture abstract knowledge about world’s physics to solve decision-making problems. Yet, the alignment between LLMs’ knowledge and the environment can be wrong and limit functional competence due to lack of grounding. In this paper, we study an approach (named GLAM) to achieve this alignment through functional grounding: we consider an agent using an LLM as a policy that is progressively updated as the agent interacts with the environment, leveraging online Reinforcement Learning to improve its performance to solve goals. Using an interactive textual environment designed to study higher-level forms of functional grounding, and a set of spatial and navigation tasks, we study several scientific questions: 1) Can LLMs boost sample efficiency for online learning of various RL tasks? 2) How can it boost different forms of generalization? 3) What is the impact of online learning? We study these questions by functionally grounding several variants (size, architecture) of FLAN-T5.

        ----

        ## [150] Stein Variational Goal Generation for adaptive Exploration in Multi-Goal Reinforcement Learning

        **Authors**: *Nicolas Castanet, Olivier Sigaud, Sylvain Lamprier*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/castanet23a.html](https://proceedings.mlr.press/v202/castanet23a.html)

        **Abstract**:

        In multi-goal Reinforcement Learning, an agent can share experience between related training tasks, resulting in better generalization for new tasks at test time. However, when the goal space has discontinuities and the reward is sparse, a majority of goals are difficult to reach. In this context, a curriculum over goals helps agents learn by adapting training tasks to their current capabilities. In this work, we propose Stein Variational Goal Generation (SVGG), which samples goals of intermediate difficulty for the agent, by leveraging a learned predictive model of its goal reaching capabilities. The distribution of goals is modeled with particles that are attracted in areas of appropriate difficulty using Stein Variational Gradient Descent. We show that SVGG outperforms state-of-the-art multi-goal Reinforcement Learning methods in terms of success coverage in hard exploration problems, and demonstrate that it is endowed with a useful recovery property when the environment changes.

        ----

        ## [151] Scalable Safe Policy Improvement via Monte Carlo Tree Search

        **Authors**: *Alberto Castellini, Federico Bianchi, Edoardo Zorzi, Thiago D. Simão, Alessandro Farinelli, Matthijs T. J. Spaan*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/castellini23a.html](https://proceedings.mlr.press/v202/castellini23a.html)

        **Abstract**:

        Algorithms for safely improving policies are important to deploy reinforcement learning approaches in real-world scenarios. In this work, we propose an algorithm, called MCTS-SPIBB, that computes safe policy improvement online using a Monte Carlo Tree Search based strategy. We theoretically prove that the policy generated by MCTS-SPIBB converges, as the number of simulations grows, to the optimal safely improved policy generated by Safe Policy Improvement with Baseline Bootstrapping (SPIBB), a popular algorithm based on policy iteration. Moreover, our empirical analysis performed on three standard benchmark domains shows that MCTS-SPIBB scales to significantly larger problems than SPIBB because it computes the policy online and locally, i.e., only in the states actually visited by the agent.

        ----

        ## [152] LESS-VFL: Communication-Efficient Feature Selection for Vertical Federated Learning

        **Authors**: *Timothy Castiglia, Yi Zhou, Shiqiang Wang, Swanand Kadhe, Nathalie Baracaldo, Stacy Patterson*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/castiglia23a.html](https://proceedings.mlr.press/v202/castiglia23a.html)

        **Abstract**:

        We propose LESS-VFL, a communication-efficient feature selection method for distributed systems with vertically partitioned data. We consider a system of a server and several parties with local datasets that share a sample ID space but have different feature sets. The parties wish to collaboratively train a model for a prediction task. As part of the training, the parties wish to remove unimportant features in the system to improve generalization, efficiency, and explainability. In LESS-VFL, after a short pre-training period, the server optimizes its part of the global model to determine the relevant outputs from party models. This information is shared with the parties to then allow local feature selection without communication. We analytically prove that LESS-VFL removes spurious features from model training. We provide extensive empirical evidence that LESS-VFL can achieve high accuracy and remove spurious features at a fraction of the communication cost of other feature selection approaches.

        ----

        ## [153] On the Robustness of Text Vectorizers

        **Authors**: *Rémi Catellier, Samuel Vaiter, Damien Garreau*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/catellier23a.html](https://proceedings.mlr.press/v202/catellier23a.html)

        **Abstract**:

        A fundamental issue in machine learning is the robustness of the model with respect to changes in the input. In natural language processing, models typically contain a first embedding layer, transforming a sequence of tokens into vector representations. While the robustness with respect to changes of continuous inputs is well-understood, the situation is less clear when considering discrete changes, for instance replacing a word by another in an input sentence. Our work formally proves that popular embedding schemes, such as concatenation, TF-IDF, and Paragraph Vector (a.k.a. doc2vec), exhibit robustness in the Hölder or Lipschitz sense with respect to the Hamming distance. We provide quantitative bounds for these schemes and demonstrate how the constants involved are affected by the length of the document. These findings are exemplified through a series of numerical examples.

        ----

        ## [154] Learning Globally Smooth Functions on Manifolds

        **Authors**: *Juan Cerviño, Luiz F. O. Chamon, Benjamin David Haeffele, René Vidal, Alejandro Ribeiro*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/cervino23a.html](https://proceedings.mlr.press/v202/cervino23a.html)

        **Abstract**:

        Smoothness and low dimensional structures play central roles in improving generalization and stability in learning and statistics. This work combines techniques from semi-infinite constrained learning and manifold regularization to learn representations that are globally smooth on a manifold. To do so, it shows that under typical conditions the problem of learning a Lipschitz continuous function on a manifold is equivalent to a dynamically weighted manifold regularization problem. This observation leads to a practical algorithm based on a weighted Laplacian penalty whose weights are adapted using stochastic gradient techniques. It is shown that under mild conditions, this method estimates the Lipschitz constant of the solution, learning a globally smooth solution as a byproduct. Experiments on real world data illustrate the advantages of the proposed method relative to existing alternatives. Our code is available at https://github.com/JuanCervino/smoothbench.

        ----

        ## [155] Tighter Lower Bounds for Shuffling SGD: Random Permutations and Beyond

        **Authors**: *Jaeyoung Cha, Jaewook Lee, Chulhee Yun*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/cha23a.html](https://proceedings.mlr.press/v202/cha23a.html)

        **Abstract**:

        We study convergence lower bounds of without-replacement stochastic gradient descent (SGD) for solving smooth (strongly-)convex finite-sum minimization problems. Unlike most existing results focusing on final iterate lower bounds in terms of the number of components $n$ and the number of epochs $K$, we seek bounds for arbitrary weighted average iterates that are tight in all factors including the condition number $\kappa$. For SGD with Random Reshuffling, we present lower bounds that have tighter $\kappa$ dependencies than existing bounds. Our results are the first to perfectly close the gap between lower and upper bounds for weighted average iterates in both strongly-convex and convex cases. We also prove weighted average iterate lower bounds for arbitrary permutation-based SGD, which apply to all variants that carefully choose the best permutation. Our bounds improve the existing bounds in factors of $n$ and $\kappa$ and thereby match the upper bounds shown for a recently proposed algorithm called GraB.

        ----

        ## [156] Orthogonality-Enforced Latent Space in Autoencoders: An Approach to Learning Disentangled Representations

        **Authors**: *Jaehoon Cha, Jeyan Thiyagalingam*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/cha23b.html](https://proceedings.mlr.press/v202/cha23b.html)

        **Abstract**:

        Noting the importance of factorizing (or disentangling) the latent space, we propose a novel, non-probabilistic disentangling framework for autoencoders, based on the principles of symmetry transformations that are independent of one another. To the best of our knowledge, this is the first deterministic model that is aiming to achieve disentanglement based on autoencoders using only a reconstruction loss without pairs of images or labels, by explicitly introducing inductive biases into a model architecture through Euler encoding. The proposed model is then compared with a number of state-of-the-art models, relevant to disentanglement, including symmetry-based models and generative models. Our evaluation using six different disentanglement metrics, including the unsupervised disentanglement metric we propose here in this paper, shows that the proposed model can offer better disentanglement, especially when variances of the features are different, where other methods may struggle. We believe that this model opens several opportunities for linear disentangled representation learning based on deterministic autoencoders.

        ----

        ## [157] STEERING : Stein Information Directed Exploration for Model-Based Reinforcement Learning

        **Authors**: *Souradip Chakraborty, Amrit S. Bedi, Alec Koppel, Mengdi Wang, Furong Huang, Dinesh Manocha*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/chakraborty23a.html](https://proceedings.mlr.press/v202/chakraborty23a.html)

        **Abstract**:

        Directed Exploration is a crucial challenge in reinforcement learning (RL), especially when rewards are sparse. Information-directed sampling (IDS), which optimizes the information ratio, seeks to do so by augmenting regret with information gain. However, estimating information gain is computationally intractable or relies on restrictive assumptions which prohibit its use in many practical instances. In this work, we posit an alternative exploration incentive in terms of the integral probability metric (IPM) between a current estimate of the transition model and the unknown optimal, which under suitable conditions, can be computed in closed form with the kernelized Stein discrepancy (KSD). Based on KSD, we develop a novel algorithm STEERING: STEin information dirEcted exploration for model-based Reinforcement LearnING. To enable its derivation, we develop fundamentally new variants of KSD for discrete conditional distributions. We further establish that STEERING archives sublinear Bayesian regret, improving upon prior learning rates of information-augmented MBRL, IDS included. Experimentally, we show that the proposed algorithm is computationally affordable and outperforms several prior approaches.

        ----

        ## [158] Thompson Sampling for High-Dimensional Sparse Linear Contextual Bandits

        **Authors**: *Sunrit Chakraborty, Saptarshi Roy, Ambuj Tewari*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/chakraborty23b.html](https://proceedings.mlr.press/v202/chakraborty23b.html)

        **Abstract**:

        We consider the stochastic linear contextual bandit problem with high-dimensional features. We analyze the Thompson sampling algorithm using special classes of sparsity-inducing priors (e.g., spike-and-slab) to model the unknown parameter and provide a nearly optimal upper bound on the expected cumulative regret. To the best of our knowledge, this is the first work that provides theoretical guarantees of Thompson sampling in high-dimensional and sparse contextual bandits. For faster computation, we use variational inference instead of Markov Chain Monte Carlo (MCMC) to approximate the posterior distribution. Extensive simulations demonstrate the improved performance of our proposed algorithm over existing ones.

        ----

        ## [159] Representations and Exploration for Deep Reinforcement Learning using Singular Value Decomposition

        **Authors**: *Yash Chandak, Shantanu Thakoor, Zhaohan Daniel Guo, Yunhao Tang, Rémi Munos, Will Dabney, Diana L. Borsa*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/chandak23a.html](https://proceedings.mlr.press/v202/chandak23a.html)

        **Abstract**:

        Representation learning and exploration are among the key challenges for any deep reinforcement learning agent. In this work, we provide a singular value decomposition based method that can be used to obtain representations that preserve the underlying transition structure in the domain. Perhaps interestingly, we show that these representations also capture the relative frequency of state visitations, thereby providing an estimate for pseudo-counts for free. To scale this decomposition method to large-scale domains, we provide an algorithm that never requires building the transition matrix, can make use of deep networks, and also permits mini-batch training. Further, we draw inspiration from predictive state representations and extend our decomposition method to partially observable environments. With experiments on multi-task settings with partially observable domains, we show that the proposed method can not only learn useful representation on DM-Lab-30 environments (that have inputs involving language instructions, pixel images, rewards, among others) but it can also be effective at hard exploration tasks in DM-Hard-8 environments.

        ----

        ## [160] Memory-Based Dual Gaussian Processes for Sequential Learning

        **Authors**: *Paul Edmund Chang, Prakhar Verma, S. T. John, Arno Solin, Mohammad Emtiyaz Khan*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/chang23a.html](https://proceedings.mlr.press/v202/chang23a.html)

        **Abstract**:

        Sequential learning with Gaussian processes (GPs) is challenging when access to past data is limited, for example, in continual and active learning. In such cases, errors can accumulate over time due to inaccuracies in the posterior, hyperparameters, and inducing points, making accurate learning challenging. Here, we present a method to keep all such errors in check using the recently proposed dual sparse variational GP. Our method enables accurate inference for generic likelihoods and improves learning by actively building and updating a memory of past data. We demonstrate its effectiveness in several applications involving Bayesian optimization, active learning, and continual learning.

        ----

        ## [161] Muse: Text-To-Image Generation via Masked Generative Transformers

        **Authors**: *Huiwen Chang, Han Zhang, Jarred Barber, Aaron Maschinot, José Lezama, Lu Jiang, Ming-Hsuan Yang, Kevin Patrick Murphy, William T. Freeman, Michael Rubinstein, Yuanzhen Li, Dilip Krishnan*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/chang23b.html](https://proceedings.mlr.press/v202/chang23b.html)

        **Abstract**:

        We present Muse, a text-to-image Transformermodel that achieves state-of-the-art image genera-tion performance while being significantly moreefficient than diffusion or autoregressive models.Muse is trained on a masked modeling task indiscrete token space: given the text embeddingextracted from a pre-trained large language model(LLM), Muse learns to predict randomly maskedimage tokens. Compared to pixel-space diffusionmodels, such as Imagen and DALL-E 2, Muse issignificantly more efficient due to the use of dis-crete tokens and requires fewer sampling itera-tions; compared to autoregressive models such asParti, Muse is more efficient due to the use of par-allel decoding. The use of a pre-trained LLM en-ables fine-grained language understanding, whichtranslates to high-fidelity image generation andthe understanding of visual concepts such as ob-jects, their spatial relationships, pose, cardinalityetc. Our 900M parameter model achieves a newSOTA on CC3M, with an FID score of 6.06. TheMuse 3B parameter model achieves an FID of7.88 on zero-shot COCO evaluation, along with aCLIP score of 0.32. Muse also directly enables anumber of image editing applications without theneed to fine-tune or invert the model: inpainting,outpainting, and mask-free editing. More resultsand videos demonstrating editing are available at https://muse-icml.github.io/

        ----

        ## [162] On Investigating the Conservative Property of Score-Based Generative Models

        **Authors**: *Chen-Hao Chao, Wei-Fang Sun, Bo-Wun Cheng, Chun-Yi Lee*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/chao23a.html](https://proceedings.mlr.press/v202/chao23a.html)

        **Abstract**:

        Existing Score-Based Models (SBMs) can be categorized into constrained SBMs (CSBMs) or unconstrained SBMs (USBMs) according to their parameterization approaches. CSBMs model probability density functions as Boltzmann distributions, and assign their predictions as the negative gradients of some scalar-valued energy functions. On the other hand, USBMs employ flexible architectures capable of directly estimating scores without the need to explicitly model energy functions. In this paper, we demonstrate that the architectural constraints of CSBMs may limit their modeling ability. In addition, we show that USBMs’ inability to preserve the property of conservativeness may lead to degraded performance in practice. To address the above issues, we propose Quasi-Conservative Score-Based Models (QCSBMs) for keeping the advantages of both CSBMs and USBMs. Our theoretical derivations demonstrate that the training objective of QCSBMs can be efficiently integrated into the training processes by leveraging the Hutchinson’s trace estimator. In addition, our experimental results on the CIFAR-10, CIFAR-100, ImageNet, and SVHN datasets validate the effectiveness of QCSBMs. Finally, we justify the advantage of QCSBMs using an example of a one-layered autoencoder.

        ----

        ## [163] Robust and private stochastic linear bandits

        **Authors**: *Vasileios Charisopoulos, Hossein Esfandiari, Vahab Mirrokni*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/charisopoulos23a.html](https://proceedings.mlr.press/v202/charisopoulos23a.html)

        **Abstract**:

        In this paper, we study the stochastic linear bandit problem under the additional requirements of differential privacy, robustness and batched observations. In particular, we assume an adversary randomly chooses a constant fraction of the observed rewards in each batch, replacing them with arbitrary numbers. We present differentially private and robust variants of the arm elimination algorithm using logarithmic batch queries under two privacy models and provide regret bounds in both settings. In the first model, every reward in each round is reported by a potentially different client, which reduces to standard local differential privacy (LDP). In the second model, every action is "owned" by a different client, who may aggregate the rewards over multiple queries and privatize the aggregate response instead. To the best of our knowledge, our algorithms are the first simultaneously providing differential privacy and adversarial robustness in the stochastic linear bandits problem.

        ----

        ## [164] Streaming Submodular Maximization with Differential Privacy

        **Authors**: *Anamay Chaturvedi, Huy L. Nguyen, Thy Dinh Nguyen*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/chaturvedi23a.html](https://proceedings.mlr.press/v202/chaturvedi23a.html)

        **Abstract**:

        In this work, we study the problem of privately maximizing a submodular function in the streaming setting. Extensive work has been done on privately maximizing submodular functions in the general case when the function depends upon the private data of individuals. However, when the size of the data stream drawn from the domain of the objective function is large or arrives very fast, one must privately optimize the objective within the constraints of the streaming setting. We establish fundamental differentially private baselines for this problem and then derive better trade-offs between privacy and utility for the special case of decomposable submodular functions. A submodular function is decomposable when it can be written as a sum of submodular functions; this structure arises naturally when each summand function models the utility of an individual and the goal is to study the total utility of the whole population as in the well-known Combinatorial Public Projects Problem. Finally, we complement our theoretical analysis with experimental corroboration.

        ----

        ## [165] Why does Throwing Away Data Improve Worst-Group Error?

        **Authors**: *Kamalika Chaudhuri, Kartik Ahuja, Martín Arjovsky, David Lopez-Paz*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/chaudhuri23a.html](https://proceedings.mlr.press/v202/chaudhuri23a.html)

        **Abstract**:

        When facing data with imbalanced classes or groups, practitioners follow an intriguing strategy to achieve best results. They throw away examples until the classes or groups are balanced in size, and then perform empirical risk minimization on the reduced training set. This opposes common wisdom in learning theory, where the expected error is supposed to decrease as the dataset grows in size. In this work, we leverage extreme value theory to address this apparent contradiction. Our results show that the tails of the data distribution play an important role in determining the worst-group-accuracy of linear classifiers. When learning on data with heavy tails, throwing away data restores the geometric symmetry of the resulting classifier, and therefore improves its worst-group generalization.

        ----

        ## [166] Collaborative Multi-Agent Heterogeneous Multi-Armed Bandits

        **Authors**: *Ronshee Chawla, Daniel Vial, Sanjay Shakkottai, R. Srikant*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/chawla23a.html](https://proceedings.mlr.press/v202/chawla23a.html)

        **Abstract**:

        The study of collaborative multi-agent bandits has attracted significant attention recently. In light of this, we initiate the study of a new collaborative setting, consisting of $N$ agents such that each agent is learning one of $M$ stochastic multi-armed bandits to minimize their group cumulative regret. We develop decentralized algorithms which facilitate collaboration between the agents under two scenarios. We characterize the performance of these algorithms by deriving the per agent cumulative regret and group regret upper bounds. We also prove lower bounds for the group regret in this setting, which demonstrates the near-optimal behavior of the proposed algorithms.

        ----

        ## [167] Correcting discount-factor mismatch in on-policy policy gradient methods

        **Authors**: *Fengdi Che, Gautham Vasan, A. Rupam Mahmood*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/che23a.html](https://proceedings.mlr.press/v202/che23a.html)

        **Abstract**:

        The policy gradient theorem gives a convenient form of the policy gradient in terms of three factors: an action value, a gradient of the action likelihood, and a state distribution involving discounting called the discounted stationary distribution. But commonly used on-policy methods based on the policy gradient theorem ignores the discount factor in the state distribution, which is technically incorrect and may even cause degenerate learning behavior in some environments. An existing solution corrects this discrepancy by using $\gamma^t$ as a factor in the gradient estimate. However, this solution is not widely adopted and does not work well in tasks where the later states are similar to earlier states. We introduce a novel distribution correction to account for the discounted stationary distribution that can be plugged into many existing gradient estimators. Our correction circumvents the performance degradation associated with the $\gamma^t$ correction with a lower variance. Importantly, compared to the uncorrected estimators, our algorithm provides improved state emphasis to evade suboptimal policies in certain environments and consistently matches or exceeds the original performance on several OpenAI gym and DeepMind suite benchmarks.

        ----

        ## [168] Fast Federated Machine Unlearning with Nonlinear Functional Theory

        **Authors**: *Tianshi Che, Yang Zhou, Zijie Zhang, Lingjuan Lyu, Ji Liu, Da Yan, Dejing Dou, Jun Huan*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/che23b.html](https://proceedings.mlr.press/v202/che23b.html)

        **Abstract**:

        Federated machine unlearning (FMU) aims to remove the influence of a specified subset of training data upon request from a trained federated learning model. Despite achieving remarkable performance, existing FMU techniques suffer from inefficiency due to two sequential operations of training and retraining/unlearning on large-scale datasets. Our prior study, PCMU, was proposed to improve the efficiency of centralized machine unlearning (CMU) with certified guarantees, by simultaneously executing the training and unlearning operations. This paper proposes a fast FMU algorithm, FFMU, for improving the FMU efficiency while maintaining the unlearning quality. The PCMU method is leveraged to train a local machine learning (MU) model on each edge device. We propose to employ nonlinear functional analysis techniques to refine the local MU models as output functions of a Nemytskii operator. We conduct theoretical analysis to derive that the Nemytskii operator has a global Lipschitz constant, which allows us to bound the difference between two MU models regarding the distance between their gradients. Based on the Nemytskii operator and average smooth local gradients, the global MU model on the server is guaranteed to achieve close performance to each local MU model with the certified guarantees.

        ----

        ## [169] On the Statistical Benefits of Temporal Difference Learning

        **Authors**: *David Cheikhi, Daniel Russo*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/cheikhi23a.html](https://proceedings.mlr.press/v202/cheikhi23a.html)

        **Abstract**:

        Given a dataset on actions and resulting long-term rewards, a direct estimation approach fits value functions that minimize prediction error on the training data. Temporal difference learning (TD) methods instead fit value functions by minimizing the degree of temporal inconsistency between estimates made at successive time-steps. Focusing on finite state Markov chains, we provide a crisp asymptotic theory of the statistical advantages of this approach. First, we show that an intuitive inverse trajectory pooling coefficient completely characterizes the percent reduction in mean-squared error of value estimates. Depending on problem structure, the reduction could be enormous or nonexistent. Next, we prove that there can be dramatic improvements in estimates of the difference in value-to-go for two states: TD’s errors are bounded in terms of a novel measure – the problem’s trajectory crossing time – which can be much smaller than the problem’s time horizon.

        ----

        ## [170] Multi-Layer Neural Networks as Trainable Ladders of Hilbert Spaces

        **Authors**: *Zhengdao Chen*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/chen23a.html](https://proceedings.mlr.press/v202/chen23a.html)

        **Abstract**:

        To characterize the functions spaces explored by multi-layer neural networks (NNs), we introduce Neural Hilbert Ladders (NHLs), a collection of reproducing kernel Hilbert spaces (RKHSes) that are defined iteratively and adaptive to training. First, we prove a correspondence between functions expressed by L-layer NNs and those belonging to L-level NHLs. Second, we prove generalization guarantees for learning the NHL based on a new complexity measure. Third, corresponding to the training of multi-layer NNs in the infinite-width mean-field limit, we derive an evolution of the NHL characterized by the dynamics of multiple random fields. Finally, we examine linear and shallow NNs from the new perspective and complement the theory with numerical results.

        ----

        ## [171] Beyond the Edge of Stability via Two-step Gradient Updates

        **Authors**: *Lei Chen, Joan Bruna*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/chen23b.html](https://proceedings.mlr.press/v202/chen23b.html)

        **Abstract**:

        Gradient Descent (GD) is a powerful workhorse of modern machine learning thanks to its scalability and efficiency in high-dimensional spaces. Its ability to find local minimisers is only guaranteed for losses with Lipschitz gradients, where it can be seen as a ’bona-fide’ discretisation of an underlying gradient flow. Yet, many ML setups involving overparametrised models do not fall into this problem class, which has motivated research beyond the so-called ”Edge of Stability” (EoS), where the step-size crosses the admissibility threshold inversely proportional to the Lipschitz constant above. Perhaps surprisingly, GD has been empirically observed to still converge regardless of local instability and oscillatory behavior. The incipient theoretical analysis of this phenomena has mainly focused in the overparametrised regime, where the effect of choosing a large learning rate may be associated to a ‘Sharpness-Minimisation’ implicit regularisation within the manifold of minimisers, under appropriate asymptotic limits. In contrast, in this work we directly examine the conditions for such unstable convergence, focusing on simple, yet representative, learning problems, via analysis of two-step gradient updates. Specifically, we characterize a local condition involving third-order derivatives that guarantees existence and convergence to fixed points of the two-step updates, and leverage such property in a teacher-student setting, under population loss. Finally, starting from Matrix Factorization, we provide observations of period-2 orbit of GD in high-dimensional settings with intuition of its dynamics, along with exploration into more general settings.

        ----

        ## [172] Trompt: Towards a Better Deep Neural Network for Tabular Data

        **Authors**: *Kuan-Yu Chen, Ping-Han Chiang, Hsin-Rung Chou, Ting-Wei Chen, Tien-Hao Chang*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/chen23c.html](https://proceedings.mlr.press/v202/chen23c.html)

        **Abstract**:

        Tabular data is arguably one of the most commonly used data structures in various practical domains, including finance, healthcare and e-commerce. The inherent heterogeneity allows tabular data to store rich information. However, based on a recently published tabular benchmark, we can see deep neural networks still fall behind tree-based models on tabular datasets. In this paper, we propose Trompt–which stands for Tabular Prompt–a novel architecture inspired by prompt learning of language models. The essence of prompt learning is to adjust a large pre-trained model through a set of prompts outside the model without directly modifying the model. Based on this idea, Trompt separates the learning strategy of tabular data into two parts. The first part, analogous to pre-trained models, focus on learning the intrinsic information of a table. The second part, analogous to prompts, focus on learning the variations among samples. Trompt is evaluated with the benchmark mentioned above. The experimental results demonstrate that Trompt outperforms state-of-the-art deep neural networks and is comparable to tree-based models.

        ----

        ## [173] Differentially Private Stochastic Convex Optimization under a Quantile Loss Function

        **Authors**: *Du Chen, Geoffrey A. Chua*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/chen23d.html](https://proceedings.mlr.press/v202/chen23d.html)

        **Abstract**:

        We study $(\varepsilon,\delta)$-differentially private (DP) stochastic convex optimization under an $r$-th quantile loss function taking the form $c(u) = ru^+ + (1-r)(-u)^+$. The function is non-smooth, and we propose to approximate it with a smooth function obtained by convolution smoothing, which enjoys both structure and bandwidth flexibility and can address outliers. This leads to a better approximation than those obtained from existing methods such as Moreau Envelope. We then design private algorithms based on DP stochastic gradient descent and objective perturbation, and show that both algorithms achieve (near) optimal excess generalization risk $O(\max\{\frac{1}{\sqrt{n}}, \frac{\sqrt{d\ln(1/\delta)}}{n\varepsilon}\})$. Through objective perturbation, we further derive an upper bound $O(\max\{\sqrt{\frac{d}{n}}, \sqrt{\frac{d\ln(1/\delta)}{n\varepsilon}}\})$ on the parameter estimation error under mild assumptions on data generating processes. Some applications in private quantile regression and private inventory control will be discussed.

        ----

        ## [174] Restoration-Degradation Beyond Linear Diffusions: A Non-Asymptotic Analysis For DDIM-type Samplers

        **Authors**: *Sitan Chen, Giannis Daras, Alex Dimakis*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/chen23e.html](https://proceedings.mlr.press/v202/chen23e.html)

        **Abstract**:

        We develop a framework for non-asymptotic analysis of deterministic samplers used for diffusion generative modeling. Several recent works have analyzed stochastic samplers using tools like Girsanov’s theorem and a chain rule variant of the interpolation argument. Unfortunately, these techniques give vacuous bounds when applied to deterministic samplers. We give a new operational interpretation for deterministic sampling by showing that one step along the probability flow ODE can be expressed as two steps: 1) a restoration step that runs gradient ascent on the conditional log-likelihood at some infinitesimally previous time, and 2) a degradation step that runs the forward process using noise pointing back towards the current iterate. This perspective allows us to extend denoising diffusion implicit models to general, non-linear forward processes. We then develop the first polynomial convergence bounds for these samplers under mild conditions on the data distribution.

        ----

        ## [175] Provably Convergent Schrödinger Bridge with Applications to Probabilistic Time Series Imputation

        **Authors**: *Yu Chen, Wei Deng, Shikai Fang, Fengpei Li, Nicole Tianjiao Yang, Yikai Zhang, Kashif Rasul, Shandian Zhe, Anderson Schneider, Yuriy Nevmyvaka*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/chen23f.html](https://proceedings.mlr.press/v202/chen23f.html)

        **Abstract**:

        The Schrödinger bridge problem (SBP) is gaining increasing attention in generative modeling and showing promising potential even in comparison with the score-based generative models (SGMs). SBP can be interpreted as an entropy-regularized optimal transport problem, which conducts projections onto every other marginal alternatingly. However, in practice, only approximated projections are accessible and their convergence is not well understood. To fill this gap, we present a first convergence analysis of the Schrödinger bridge algorithm based on approximated projections. As for its practical applications, we apply SBP to probabilistic time series imputation by generating missing values conditioned on observed data. We show that optimizing the transport cost improves the performance and the proposed algorithm achieves the state-of-the-art result in healthcare and environmental data while exhibiting the advantage of exploring both temporal and feature patterns in probabilistic time series imputation.

        ----

        ## [176] ED-Batch: Efficient Automatic Batching of Dynamic Neural Networks via Learned Finite State Machines

        **Authors**: *Siyuan Chen, Pratik Pramod Fegade, Tianqi Chen, Phillip B. Gibbons, Todd C. Mowry*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/chen23g.html](https://proceedings.mlr.press/v202/chen23g.html)

        **Abstract**:

        Batching has a fundamental influence on the efficiency of deep neural network (DNN) execution. However, for dynamic DNNs, efficient batching is particularly challenging as the dataflow graph varies per input instance. As a result, state-of-the-art frameworks use heuristics that result in suboptimal batching decisions. Further, batching puts strict restrictions on memory adjacency and can lead to high data movement costs. In this paper, we provide an approach for batching dynamic DNNs based on finite state machines, which enables the automatic discovery of batching policies specialized for each DNN via reinforcement learning. Moreover, we find that memory planning that is aware of the batching policy can save significant data movement overheads, which is automated by a PQ tree-based algorithm we introduce. Experimental results show that our framework speeds up state-of-the-art frameworks by on average 1.15x, 1.39x, and 2.45x for chain-based, tree-based, and lattice-based DNNs across CPU and GPU. The framework is open-sourced at https://github.com/gulang2019/ED-Batch.git.

        ----

        ## [177] Is Learning Summary Statistics Necessary for Likelihood-free Inference?

        **Authors**: *Yanzhi Chen, Michael U. Gutmann, Adrian Weller*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/chen23h.html](https://proceedings.mlr.press/v202/chen23h.html)

        **Abstract**:

        Likelihood-free inference (LFI) is a set of techniques for inference in implicit statistical models. A longstanding question in LFI has been how to design or learn good summary statistics of data, but this might now seem unnecessary due to the advent of recent end-to-end (i.e. neural network-based) LFI methods. In this work, we rethink this question with a new method for learning summary statistics. We show that learning sufficient statistics may be easier than direct posterior inference, as the former problem can be reduced to a set of low-dimensional, easy-to-solve learning problems. This suggests us to explicitly decouple summary statistics learning from posterior inference in LFI. Experiments on diverse inference tasks with different data types validate our hypothesis.

        ----

        ## [178] Subequivariant Graph Reinforcement Learning in 3D Environments

        **Authors**: *Runfa Chen, Jiaqi Han, Fuchun Sun, Wenbing Huang*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/chen23i.html](https://proceedings.mlr.press/v202/chen23i.html)

        **Abstract**:

        Learning a shared policy that guides the locomotion of different agents is of core interest in Reinforcement Learning (RL), which leads to the study of morphology-agnostic RL. However, existing benchmarks are highly restrictive in the choice of starting point and target point, constraining the movement of the agents within 2D space. In this work, we propose a novel setup for morphology-agnostic RL, dubbed Subequivariant Graph RL in 3D environments (3D-SGRL). Specifically, we first introduce a new set of more practical yet challenging benchmarks in 3D space that allows the agent to have full Degree-of-Freedoms to explore in arbitrary directions starting from arbitrary configurations. Moreover, to optimize the policy over the enlarged state-action space, we propose to inject geometric symmetry, i.e., subequivariance, into the modeling of the policy and Q-function such that the policy can generalize to all directions, improving exploration efficiency. This goal is achieved by a novel SubEquivariant Transformer (SET) that permits expressive message exchange. Finally, we evaluate the proposed method on the proposed benchmarks, where our method consistently and significantly outperforms existing approaches on single-task, multi-task, and zero-shot generalization scenarios. Extensive ablations are also conducted to verify our design.

        ----

        ## [179] GuardHFL: Privacy Guardian for Heterogeneous Federated Learning

        **Authors**: *Hanxiao Chen, Meng Hao, Hongwei Li, Kangjie Chen, Guowen Xu, Tianwei Zhang, Xilin Zhang*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/chen23j.html](https://proceedings.mlr.press/v202/chen23j.html)

        **Abstract**:

        Heterogeneous federated learning (HFL) enables clients with different computation and communication capabilities to collaboratively train their own customized models via a query-response paradigm on auxiliary datasets. However, such a paradigm raises serious privacy concerns due to the leakage of highly sensitive query samples and response predictions. We put forth GuardHFL, the first-of-its-kind efficient and privacy-preserving HFL framework. GuardHFL is equipped with a novel HFL-friendly secure querying scheme built on lightweight secret sharing and symmetric-key techniques. The core of GuardHFL is two customized multiplication and comparison protocols, which substantially boost the execution efficiency. Extensive evaluations demonstrate that GuardHFL significantly outperforms the alternative instantiations based on existing state-of-the-art techniques in both runtime and communication cost.

        ----

        ## [180] Efficient and Degree-Guided Graph Generation via Discrete Diffusion Modeling

        **Authors**: *Xiaohui Chen, Jiaxing He, Xu Han, Liping Liu*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/chen23k.html](https://proceedings.mlr.press/v202/chen23k.html)

        **Abstract**:

        Diffusion-based generative graph models have been proven effective in generating high-quality small graphs. However, they need to be more scalable for generating large graphs containing thousands of nodes desiring graph statistics. In this work, we propose EDGE, a new diffusion-based generative graph model that addresses generative tasks with large graphs. To improve computation efficiency, we encourage graph sparsity by using a discrete diffusion process that randomly removes edges at each time step and finally obtains an empty graph. EDGE only focuses on a portion of nodes in the graph at each denoising step. It makes much fewer edge predictions than previous diffusion-based models. Moreover, EDGE admits explicitly modeling the node degrees of the graphs, further improving the model performance. The empirical study shows that EDGE is much more efficient than competing methods and can generate large graphs with thousands of nodes. It also outperforms baseline models in generation quality: graphs generated by our approach have more similar graph statistics to those of the training graphs.

        ----

        ## [181] Evolving Semantic Prototype Improves Generative Zero-Shot Learning

        **Authors**: *Shiming Chen, Wenjin Hou, Ziming Hong, Xiaohan Ding, Yibing Song, Xinge You, Tongliang Liu, Kun Zhang*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/chen23l.html](https://proceedings.mlr.press/v202/chen23l.html)

        **Abstract**:

        In zero-shot learning (ZSL), generative methods synthesize class-related sample features based on predefined semantic prototypes. They advance the ZSL performance by synthesizing unseen class sample features for better training the classifier. We observe that each class’s predefined semantic prototype (also referred to as semantic embedding or condition) does not accurately match its real semantic prototype. So the synthesized visual sample features do not faithfully represent the real sample features, limiting the classifier training and existing ZSL performance. In this paper, we formulate this mismatch phenomenon as the visual-semantic domain shift problem. We propose a dynamic semantic prototype evolving (DSP) method to align the empirically predefined semantic prototypes and the real prototypes for class-related feature synthesis. The alignment is learned by refining sample features and semantic prototypes in a unified framework and making the synthesized visual sample features approach real sample features. After alignment, synthesized sample features from unseen classes are closer to the real sample features and benefit DSP to improve existing generative ZSL methods by 8.5%, 8.0%, and 9.7% on the standard CUB, SUN AWA2 datasets, the significant performance improvement indicates that evolving semantic prototype explores a virgin field in ZSL.

        ----

        ## [182] Explore and Exploit the Diverse Knowledge in Model Zoo for Domain Generalization

        **Authors**: *Yimeng Chen, Tianyang Hu, Fengwei Zhou, Zhenguo Li, Zhi-Ming Ma*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/chen23m.html](https://proceedings.mlr.press/v202/chen23m.html)

        **Abstract**:

        The proliferation of pretrained models, as a result of advancements in pretraining techniques, has led to the emergence of a vast zoo of publicly available models. Effectively utilizing these resources to obtain models with robust out-of-distribution generalization capabilities for downstream tasks has become a crucial area of research. Previous research has primarily focused on identifying the most powerful models within the model zoo, neglecting to fully leverage the diverse inductive biases contained within. This paper argues that the knowledge contained in weaker models is valuable and presents a method for leveraging the diversity within the model zoo to improve out-of-distribution generalization capabilities. Specifically, we investigate the behaviors of various pretrained models across different domains of downstream tasks by characterizing the variations in their encoded representations in terms of two dimensions: diversity shift and correlation shift. This characterization enables us to propose a new algorithm for integrating diverse pretrained models, not limited to the strongest models, in order to achieve enhanced out-of-distribution generalization performance. Our proposed method demonstrates state-of-the-art empirical results on a variety of datasets, thus validating the benefits of utilizing diverse knowledge.

        ----

        ## [183] Decentralized Stochastic Bilevel Optimization with Improved per-Iteration Complexity

        **Authors**: *Xuxing Chen, Minhui Huang, Shiqian Ma, Krishna Balasubramanian*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/chen23n.html](https://proceedings.mlr.press/v202/chen23n.html)

        **Abstract**:

        Bilevel optimization recently has received tremendous attention due to its great success in solving important machine learning problems like meta learning, reinforcement learning, and hyperparameter optimization. Extending single-agent training on bilevel problems to the decentralized setting is a natural generalization, and there has been a flurry of work studying decentralized bilevel optimization algorithms. However, it remains unknown how to design the distributed algorithm with sample complexity and convergence rate comparable to SGD for stochastic optimization, and at the same time without directly computing the exact Hessian or Jacobian matrices. In this paper we propose such an algorithm. More specifically, we propose a novel decentralized stochastic bilevel optimization (DSBO) algorithm that only requires first order stochastic oracle, Hessian-vector product and Jacobian-vector product oracle. The sample complexity of our algorithm matches the currently best known results for DSBO, while our algorithm does not require estimating the full Hessian and Jacobian matrices, thereby possessing to improved per-iteration complexity.

        ----

        ## [184] Score Approximation, Estimation and Distribution Recovery of Diffusion Models on Low-Dimensional Data

        **Authors**: *Minshuo Chen, Kaixuan Huang, Tuo Zhao, Mengdi Wang*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/chen23o.html](https://proceedings.mlr.press/v202/chen23o.html)

        **Abstract**:

        Diffusion models achieve state-of-the-art performance in various generation tasks. However, their theoretical foundations fall far behind. This paper studies score approximation, estimation, and distribution recovery of diffusion models, when data are supported on an unknown low-dimensional linear subspace. Our result provides sample complexity bounds for distribution estimation using diffusion models. We show that with a properly chosen neural network architecture, the score function can be both accurately approximated and efficiently estimated. Further, the generated distribution based on the estimated score function captures the data geometric structures and converges to a close vicinity of the data distribution. The convergence rate depends on subspace dimension, implying that diffusion models can circumvent the curse of data ambient dimensionality.

        ----

        ## [185] Sample Complexity of Probability Divergences under Group Symmetry

        **Authors**: *Ziyu Chen, Markos A. Katsoulakis, Luc Rey-Bellet, Wei Zhu*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/chen23p.html](https://proceedings.mlr.press/v202/chen23p.html)

        **Abstract**:

        We rigorously quantify the improvement in the sample complexity of variational divergence estimations for group-invariant distributions. In the cases of the Wasserstein-1 metric and the Lipschitz-regularized $\alpha$-divergences, the reduction of sample complexity is proportional to an ambient-dimension-dependent power of the group size. For the maximum mean discrepancy (MMD), the improvement of sample complexity is more nuanced, as it depends on not only the group size but also the choice of kernel. Numerical simulations verify our theories.

        ----

        ## [186] Improved Analysis of Score-based Generative Modeling: User-Friendly Bounds under Minimal Smoothness Assumptions

        **Authors**: *Hongrui Chen, Holden Lee, Jianfeng Lu*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/chen23q.html](https://proceedings.mlr.press/v202/chen23q.html)

        **Abstract**:

        We give an improved theoretical analysis of score-based generative modeling. Under a score estimate with small $L^2$ error (averaged across timesteps), we provide efficient convergence guarantees for any data distribution with second-order moment, by either employing early stopping or assuming smoothness condition on the score function of the data distribution. Our result does not rely on any log-concavity or functional inequality assumption and has a logarithmic dependence on the smoothness. In particular, we show that under only a finite second moment condition, approximating the following in reverse KL divergence in $\epsilon$-accuracy can be done in $\tilde O\left(\frac{d \log (1/\delta)}{\epsilon}\right)$ steps: 1) the variance-$\delta$ Gaussian perturbation of any data distribution; 2) data distributions with $1/\delta$-smooth score functions. Our analysis also provides a quantitative comparison between different discrete approximations and may guide the choice of discretization points in practice.

        ----

        ## [187] Bidirectional Looking with A Novel Double Exponential Moving Average to Adaptive and Non-adaptive Momentum Optimizers

        **Authors**: *Yineng Chen, Zuchao Li, Lefei Zhang, Bo Du, Hai Zhao*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/chen23r.html](https://proceedings.mlr.press/v202/chen23r.html)

        **Abstract**:

        Optimizer is an essential component for the success of deep learning, which guides the neural network to update the parameters according to the loss on the training set. SGD and Adam are two classical and effective optimizers on which researchers have proposed many variants, such as SGDM and RAdam. In this paper, we innovatively combine the backward-looking and forward-looking aspects of the optimizer algorithm and propose a novel Admeta (A Double exponential Moving averagE To Adaptive and non-adaptive momentum) optimizer framework. For backward-looking part, we propose a DEMA variant scheme, which is motivated by a metric in the stock market, to replace the common exponential moving average scheme. While in the forward-looking part, we present a dynamic lookahead strategy which asymptotically approaches a set value, maintaining its speed at early stage and high convergence performance at final stage. Based on this idea, we provide two optimizer implementations, AdmetaR and AdmetaS, the former based on RAdam and the latter based on SGDM. Through extensive experiments on diverse tasks, we find that the proposed Admeta optimizer outperforms our base optimizers and shows advantages over recently proposed competitive optimizers. We also provide theoretical proof of these two algorithms, which verifies the convergence of our proposed Admeta.

        ----

        ## [188] HarsanyiNet: Computing Accurate Shapley Values in a Single Forward Propagation

        **Authors**: *Lu Chen, Siyu Lou, Keyan Zhang, Jin Huang, Quanshi Zhang*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/chen23s.html](https://proceedings.mlr.press/v202/chen23s.html)

        **Abstract**:

        The Shapley value is widely regarded as a trustworthy attribution metric. However, when people use Shapley values to explain the attribution of input variables of a deep neural network (DNN), it usually requires a very high computational cost to approximate relatively accurate Shapley values in real-world applications. Therefore, we propose a novel network architecture, the HarsanyiNet, which makes inferences on the input sample and simultaneously computes the exact Shapley values of the input variables in a single forward propagation. The HarsanyiNet is designed on the theoretical foundation that the Shapley value can be reformulated as the redistribution of Harsanyi interactions encoded by the network.

        ----

        ## [189] Generalized Implicit Follow-The-Regularized-Leader

        **Authors**: *Keyi Chen, Francesco Orabona*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/chen23t.html](https://proceedings.mlr.press/v202/chen23t.html)

        **Abstract**:

        We propose a new class of online learning algorithms, generalized implicit Follow-The-Regularized-Leader (FTRL), that expands the scope of FTRL framework. Generalized implicit FTRL can recover known algorithms, such as FTRL with linearized losses and implicit FTRL, and it allows the design of new update rules, as extensions of aProx and Mirror-Prox to FTRL. Our theory is constructive in the sense that it provides a simple unifying framework to design updates that directly improve the worst-case upper bound on the regret. The key idea is substituting the linearization of the losses with a Fenchel-Young inequality. We show the flexibility of the framework by proving that some known algorithms, like the Mirror-Prox updates, are instantiations of the generalized implicit FTRL. Finally, the new framework allows us to recover the temporal variation bound of implicit OMD, with the same computational complexity.

        ----

        ## [190] Fisher Information Embedding for Node and Graph Learning

        **Authors**: *Dexiong Chen, Paolo Pellizzoni, Karsten M. Borgwardt*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/chen23u.html](https://proceedings.mlr.press/v202/chen23u.html)

        **Abstract**:

        Attention-based graph neural networks (GNNs), such as graph attention networks (GATs), have become popular neural architectures for processing graph-structured data and learning node embeddings. Despite their empirical success, these models rely on labeled data and the theoretical properties of these models have yet to be fully understood. In this work, we propose a novel attention-based node embedding framework for graphs. Our framework builds upon a hierarchical kernel for multisets of subgraphs around nodes (e.g. neighborhoods) and each kernel leverages the geometry of a smooth statistical manifold to compare pairs of multisets, by “projecting” the multisets onto the manifold. By explicitly computing node embeddings with a manifold of Gaussian mixtures, our method leads to a new attention mechanism for neighborhood aggregation. We provide theoretical insights into generalizability and expressivity of our embeddings, contributing to a deeper understanding of attention-based GNNs. We propose both efficient unsupervised and supervised methods for learning the embeddings. Through experiments on several node classification benchmarks, we demonstrate that our proposed method outperforms existing attention-based graph models like GATs. Our code is available at https://github.com/BorgwardtLab/fisher_information_embedding.

        ----

        ## [191] Rethinking Visual Reconstruction: Experience-Based Content Completion Guided by Visual Cues

        **Authors**: *Jiaxuan Chen, Yu Qi, Gang Pan*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/chen23v.html](https://proceedings.mlr.press/v202/chen23v.html)

        **Abstract**:

        Decoding seen images from brain activities has been an absorbing field. However, the reconstructed images still suffer from low quality with existing studies. This can be because our visual system is not like a camera that ”remembers” every pixel. Instead, only part of the information can be perceived with our selective attention, and the brain ”guesses” the rest to form what we think we see. Most existing approaches ignored the brain completion mechanism. In this work, we propose to reconstruct seen images with both the visual perception and the brain completion process, and design a simple, yet effective visual decoding framework to achieve this goal. Specifically, we first construct a shared discrete representation space for both brain signals and images. Then, a novel self-supervised token-to-token inpainting network is designed to implement visual content completion by building context and prior knowledge about the visual objects from the discrete latent space. Our approach improved the quality of visual reconstruction significantly and achieved state-of-the-art.

        ----

        ## [192] Stratified Adversarial Robustness with Rejection

        **Authors**: *Jiefeng Chen, Jayaram Raghuram, Jihye Choi, Xi Wu, Yingyu Liang, Somesh Jha*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/chen23w.html](https://proceedings.mlr.press/v202/chen23w.html)

        **Abstract**:

        Recently, there is an emerging interest in adversarially training a classifier with a rejection option (also known as a selective classifier) for boosting adversarial robustness. While rejection can incur a cost in many applications, existing studies typically associate zero cost with rejecting perturbed inputs, which can result in the rejection of numerous slightly-perturbed inputs that could be correctly classified. In this work, we study adversarially-robust classification with rejection in the stratified rejection setting, where the rejection cost is modeled by rejection loss functions monotonically non-increasing in the perturbation magnitude. We theoretically analyze the stratified rejection setting and propose a novel defense method – Adversarial Training with Consistent Prediction-based Rejection (CPR) – for building a robust selective classifier. Experiments on image datasets demonstrate that the proposed method significantly outperforms existing methods under strong adaptive attacks. For instance, on CIFAR-10, CPR reduces the total robust loss (for different rejection losses) by at least 7.3% under both seen and unseen attacks.

        ----

        ## [193] Multi-task Hierarchical Adversarial Inverse Reinforcement Learning

        **Authors**: *Jiayu Chen, Dipesh Tamboli, Tian Lan, Vaneet Aggarwal*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/chen23x.html](https://proceedings.mlr.press/v202/chen23x.html)

        **Abstract**:

        Multi-task Imitation Learning (MIL) aims to train a policy capable of performing a distribution of tasks based on multi-task expert demonstrations, which is essential for general-purpose robots. Existing MIL algorithms suffer from low data efficiency and poor performance on complex long-horizontal tasks. We develop Multi-task Hierarchical Adversarial Inverse Reinforcement Learning (MH-AIRL) to learn hierarchically-structured multi-task policies, which is more beneficial for compositional tasks with long horizons and has higher expert data efficiency through identifying and transferring reusable basic skills across tasks. To realize this, MH-AIRL effectively synthesizes context-based multi-task learning, AIRL (an IL approach), and hierarchical policy learning. Further, MH-AIRL can be adopted to demonstrations without the task or skill annotations (i.e., state-action pairs only) which are more accessible in practice. Theoretical justifications are provided for each module of MH-AIRL, and evaluations on challenging multi-task settings demonstrate superior performance and transferability of the multi-task policies learned with MH-AIRL as compared to SOTA MIL baselines.

        ----

        ## [194] Model Transferability with Responsive Decision Subjects

        **Authors**: *Yatong Chen, Zeyu Tang, Kun Zhang, Yang Liu*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/chen23y.html](https://proceedings.mlr.press/v202/chen23y.html)

        **Abstract**:

        Given an algorithmic predictor that is accurate on some source population consisting of strategic human decision subjects, will it remain accurate if the population respond to it? In our setting, an agent or a user corresponds to a sample $(X,Y)$ drawn from a distribution $\cal{D}$ and will face a model $h$ and its classification result $h(X)$. Agents can modify $X$ to adapt to $h$, which will incur a distribution shift on $(X,Y)$. Our formulation is motivated by applications where the deployed machine learning models are subjected to human agents, and will ultimately face responsive and interactive data distributions. We formalize the discussions of the transferability of a model by studying how the performance of the model trained on the available source distribution (data) would translate to the performance on its induced domain. We provide both upper bounds for the performance gap due to the induced domain shift, as well as lower bounds for the trade-offs that a classifier has to suffer on either the source training distribution or the induced target distribution. We provide further instantiated analysis for two popular domain adaptation settings, including covariate shift and target shift.

        ----

        ## [195] Layered State Discovery for Incremental Autonomous Exploration

        **Authors**: *Liyu Chen, Andrea Tirinzoni, Alessandro Lazaric, Matteo Pirotta*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/chen23z.html](https://proceedings.mlr.press/v202/chen23z.html)

        **Abstract**:

        We study the autonomous exploration (AX) problem proposed by Lim & Auer (2012). In this setting, the objective is to discover a set of $\epsilon$-optimal policies reaching a set $\mathcal{S}_L^{\rightarrow}$ of incrementally $L$-controllable states. We introduce a novel layered decomposition of the set of incrementally $L$-controllable states that is based on the iterative application of a state-expansion operator. We leverage these results to design Layered Autonomous Exploration (LAE), a novel algorithm for AX that attains a sample complexity of $\tilde{\mathcal{O}}(LS^{\rightarrow}_{L(1+\epsilon)}\Gamma_{L(1+\epsilon)} A \ln^{12}(S^{\rightarrow}_{L(1+\epsilon)})/\epsilon^2)$, where $S^{\rightarrow}_{L(1+\epsilon)}$ is the number of states that are incrementally $L(1+\epsilon)$-controllable, $A$ is the number of actions, and $\Gamma_{L(1+\epsilon)}$ is the branching factor of the transitions over such states. LAE improves over the algorithm of Tarbouriech et al. (2020a) by a factor of $L^2$ and it is the first algorithm for AX that works in a countably-infinite state space. Moreover, we show that, under a certain identifiability assumption, LAE achieves minimax-optimal sample complexity of $\tilde{\mathcal{O}}(LS^{\rightarrow}_{L}A\ln^{12}(S^{\rightarrow}_{L})/\epsilon^2)$, outperforming existing algorithms and matching for the first time the lower bound proved by Cai et al. (2022) up to logarithmic factors.

        ----

        ## [196] Optimistic Online Mirror Descent for Bridging Stochastic and Adversarial Online Convex Optimization

        **Authors**: *Sijia Chen, Wei-Wei Tu, Peng Zhao, Lijun Zhang*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/chen23aa.html](https://proceedings.mlr.press/v202/chen23aa.html)

        **Abstract**:

        Stochastically Extended Adversarial (SEA) model is introduced by Sachs et al. (2022) as an interpolation between stochastic and adversarial online convex optimization. Under the smoothness condition, they demonstrate that the expected regret of optimistic follow-the-regularized-leader (FTRL) depends on the cumulative stochastic variance $\sigma_{1:T}^2$ and the cumulative adversarial variation $\Sigma_{1:T}^2$ for convex functions. They also provide a slightly weaker bound based on the maximal stochastic variance $\sigma_{\max}^2$ and the maximal adversarial variation $\Sigma_{\max}^2$ for strongly convex functions. Inspired by their work, we investigate the theoretical guarantees of optimistic online mirror descent (OMD) for the SEA model. For convex and smooth functions, we obtain the same $\mathcal{O}(\sqrt{\sigma_{1:T}^2}+\sqrt{\Sigma_{1:T}^2})$ regret bound, without the convexity requirement of individual functions. For strongly convex and smooth functions, we establish an $\mathcal{O}(\min\{\log (\sigma_{1:T}^2+\Sigma_{1:T}^2), (\sigma_{\max}^2 + \Sigma_{\max}^2) \log T\})$ bound, better than their $\mathcal{O}((\sigma_{\max}^2 + \Sigma_{\max}^2) \log T)$ result. For exp-concave and smooth functions, we achieve a new $\mathcal{O}(d\log(\sigma_{1:T}^2+\Sigma_{1:T}^2))$ bound. Owing to the OMD framework, we further establish dynamic regret for convex and smooth functions, which is more favorable in non-stationary online scenarios.

        ----

        ## [197] Learning to Optimize Differentiable Games

        **Authors**: *Xuxi Chen, Nelson Vadori, Tianlong Chen, Zhangyang Wang*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/chen23ab.html](https://proceedings.mlr.press/v202/chen23ab.html)

        **Abstract**:

        Many machine learning problems can be abstracted in solving game theory formulations and boil down to optimizing nested objectives, such as generative adversarial networks (GANs) and multi-agent reinforcement learning. Solving these games requires finding their stable fixed points or Nash equilibrium. However, existing algorithms for solving games suffer from empirical instability, hence demanding heavy ad-hoc tuning in practice. To tackle these challenges, we resort to the emerging scheme of Learning to Optimize (L2O), which discovers problem-specific efficient optimization algorithms through data-driven training. Our customized L2O framework for differentiable game theory problems, dubbed “Learning to Play Games" (L2PG), seeks a stable fixed point solution, by predicting the fast update direction from the past trajectory, with a novel gradient stability-aware, sign-based loss function. We further incorporate curriculum learning and self-learning to strengthen the empirical training stability and generalization of L2PG. On test problems including quadratic games and GANs, L2PG can substantially accelerate the convergence, and demonstrates a remarkably more stable trajectory. Codes are available at https://github.com/VITA-Group/L2PG.

        ----

        ## [198] Coordinated Dynamic Bidding in Repeated Second-Price Auctions with Budgets

        **Authors**: *Yurong Chen, Qian Wang, Zhijian Duan, Haoran Sun, Zhaohua Chen, Xiang Yan, Xiaotie Deng*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/chen23ac.html](https://proceedings.mlr.press/v202/chen23ac.html)

        **Abstract**:

        In online ad markets, a rising number of advertisers are employing bidding agencies to participate in ad auctions. These agencies are specialized in designing online algorithms and bidding on behalf of their clients. Typically, an agency usually has information on multiple advertisers, so she can potentially coordinate bids to help her clients achieve higher utilities than those under independent bidding. In this paper, we study coordinated online bidding algorithms in repeated second-price auctions with budgets. We propose algorithms that guarantee every client a higher utility than the best she can get under independent bidding. We show that these algorithms achieve maximal social welfare and discuss bidders’ incentives to misreport their budgets, in symmetric cases. Our proofs combine the techniques of online learning and equilibrium analysis, overcoming the difficulty of competing with a multi-dimensional benchmark. The performance of our algorithms is further evaluated by experiments on both synthetic and real data. To the best of our knowledge, we are the first to consider bidder coordination in online repeated auctions with constraints.

        ----

        ## [199] Semi-Offline Reinforcement Learning for Optimized Text Generation

        **Authors**: *Changyu Chen, Xiting Wang, Yiqiao Jin, Victor Ye Dong, Li Dong, Jie Cao, Yi Liu, Rui Yan*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/chen23ad.html](https://proceedings.mlr.press/v202/chen23ad.html)

        **Abstract**:

        Existing reinforcement learning (RL) mainly utilize online or offline settings. The online methods explore the environment with expensive time cost, and the offline methods efficiently obtain reward signals by sacrificing the exploration capability. We propose semi-offline RL, a novel paradigm that can smoothly transit from the offline setting to the online setting, balances the exploration capability and training cost, and provides a theoretical foundation for comparing different RL settings. Based on the semi-offline MDP formulation, we present the RL setting that is optimal in terms of optimization cost, asymptotic error, and overfitting error bound. Extensive experiments show that our semi-offline RL approach is effective in various text generation tasks and datasets, and yields comparable or usually better performance compared with the state-of-the-art methods.

        ----

        

[Go to the next page](ICML-2023-list02.md)

[Go to the catalog section](README.md)