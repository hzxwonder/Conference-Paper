## [1200] Bayesian Neural Networks Avoid Encoding Complex and Perturbation-Sensitive Concepts

        **Authors**: *Qihan Ren, Huiqi Deng, Yunuo Chen, Siyu Lou, Quanshi Zhang*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/ren23a.html](https://proceedings.mlr.press/v202/ren23a.html)

        **Abstract**:

        In this paper, we focus on mean-field variational Bayesian Neural Networks (BNNs) and explore the representation capacity of such BNNs by investigating which types of concepts are less likely to be encoded by the BNN. It has been observed and studied that a relatively small set of interactive concepts usually emerge in the knowledge representation of a sufficiently-trained neural network, and such concepts can faithfully explain the network output. Based on this, our study proves that compared to standard deep neural networks (DNNs), it is less likely for BNNs to encode complex concepts. Experiments verify our theoretical proofs. Note that the tendency to encode less complex concepts does not necessarily imply weak representation power, considering that complex concepts exhibit low generalization power and high adversarial vulnerability. The code is available at https://github.com/sjtu-xai-lab/BNN-concepts.

        ----

        ## [1201] Escaping saddle points in zeroth-order optimization: the power of two-point estimators

        **Authors**: *Zhaolin Ren, Yujie Tang, Na Li*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/ren23b.html](https://proceedings.mlr.press/v202/ren23b.html)

        **Abstract**:

        Two-point zeroth order methods are important in many applications of zeroth-order optimization arising in robotics, wind farms, power systems, online optimization, and adversarial robustness to black-box attacks in deep neural networks, where the problem can be high-dimensional and/or time-varying. Furthermore, such problems may be nonconvex and contain saddle points. While existing works have shown that zeroth-order methods utilizing $\Omega(d)$ function valuations per iteration (with $d$ denoting the problem dimension) can escape saddle points efficiently, it remains an open question if zeroth-order methods based on two-point estimators can escape saddle points. In this paper, we show that by adding an appropriate isotropic perturbation at each iteration, a zeroth-order algorithm based on $2m$ (for any $1 \leq m \leq d$) function evaluations per iteration can not only find $\epsilon$-second order stationary points polynomially fast, but do so using only $\tilde{O}(\frac{d}{m\epsilon^{2}\bar{\psi}})$ function evaluations, where $\bar{\psi} \geq \tilde{\Omega}(\sqrt{\epsilon})$ is a parameter capturing the extent to which the function of interest exhibits the strict saddle property.

        ----

        ## [1202] Dimension-independent Certified Neural Network Watermarks via Mollifier Smoothing

        **Authors**: *Jiaxiang Ren, Yang Zhou, Jiayin Jin, Lingjuan Lyu, Da Yan*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/ren23c.html](https://proceedings.mlr.press/v202/ren23c.html)

        **Abstract**:

        Certified_Watermarks is the first to provide a watermark certificate against $l_2$-norm watermark removal attacks, by leveraging the randomized smoothing techniques for certified robustness to adversarial attacks. However, the randomized smoothing techniques suffer from hardness of certified robustness in high-dimensional space against $l_p$-norm attacks for large $p$ ($p>2$). The certified watermark method based on the randomized smoothing is no exception, i.e., fails to provide meaningful certificates in high-dimensional space against the $l_p$-norm watermark removal attacks ($p>2$). By leveraging mollifier theory, this paper proposes a mollifier smoothing method with dimension-independent certified radius of our proposed smooth classifier, for conducting the certified watermark problem against the $l_p$-norm watermark removal attacks ($1 \leq p \leq \infty$) for high parameter dimension $d$. Based on partial differential equation (PDE) theory, an approximation of mollifier smoothing is developed to alleviate the inefficiency of sampling and prediction in the randomized smoothing as well as numerical integration in the mollifier smoothing, while maintaining the certified watermark against the $l_p$-norm watermark removal attacks ($1 \leq p \leq \infty$).

        ----

        ## [1203] Feature Programming for Multivariate Time Series Prediction

        **Authors**: *Alex Daniel Reneau, Jerry Yao-Chieh Hu, Ammar Gilani, Han Liu*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/reneau23a.html](https://proceedings.mlr.press/v202/reneau23a.html)

        **Abstract**:

        We introduce the concept of programmable feature engineering for time series modeling and propose a feature programming framework. This framework generates large amounts of predictive features for noisy multivariate time series while allowing users to incorporate their inductive bias with minimal effort. The key motivation of our framework is to view any multivariate time series as a cumulative sum of fine-grained trajectory increments, with each increment governed by a novel spin-gas dynamical Ising model. This fine-grained perspective motivates the development of a parsimonious set of operators that summarize multivariate time series in an abstract fashion, serving as the foundation for large-scale automated feature engineering. Numerically, we validate the efficacy of our method on several synthetic and real-world noisy time series datasets.

        ----

        ## [1204] Run-off Election: Improved Provable Defense against Data Poisoning Attacks

        **Authors**: *Keivan Rezaei, Kiarash Banihashem, Atoosa Malemir Chegini, Soheil Feizi*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/rezaei23a.html](https://proceedings.mlr.press/v202/rezaei23a.html)

        **Abstract**:

        In data poisoning attacks, an adversary tries to change a model’s prediction by adding, modifying, or removing samples in the training data. Recently, ensemble-based approaches for obtaining provable defenses against data poisoning have been proposed where predictions are done by taking a majority vote across multiple base models. In this work, we show that merely considering the majority vote in ensemble defenses is wasteful as it does not effectively utilize available information in the logits layers of the base models. Instead, we propose Run-Off Election (ROE), a novel aggregation method based on a two-round election across the base models: In the first round, models vote for their preferred class and then a second, Run-Off election is held between the top two classes in the first round. Based on this approach, we propose DPA+ROE and FA+ROE defense methods based on Deep Partition Aggregation (DPA) and Finite Aggregation (FA) approaches from prior work. We evaluate our methods on MNIST, CIFAR-10, and GTSRB and obtain improvements in certified accuracy by up to $3%$-$4%$. Also, by applying ROE on a boosted version of DPA, we gain improvements around $12%$-$27%$ comparing to the current state-of-the-art, establishing a new state-of-the-art in (pointwise) certified robustness against data poisoning. In many cases, our approach outperforms the state-of-the-art, even when using 32 times less computational power.

        ----

        ## [1205] Learning Control-Oriented Dynamical Structure from Data

        **Authors**: *Spencer M. Richards, Jean-Jacques E. Slotine, Navid Azizan, Marco Pavone*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/richards23a.html](https://proceedings.mlr.press/v202/richards23a.html)

        **Abstract**:

        Even for known nonlinear dynamical systems, feedback controller synthesis is a difficult problem that often requires leveraging the particular structure of the dynamics to induce a stable closed-loop system. For general nonlinear models, including those fit to data, there may not be enough known structure to reliably synthesize a stabilizing feedback controller. In this paper, we discuss a state-dependent nonlinear tracking controller formulation based on a state-dependent Riccati equation for general nonlinear control-affine systems. This formulation depends on a nonlinear factorization of the system of vector fields defining the control-affine dynamics, which always exists under mild smoothness assumptions. We propose a method for learning this factorization from a finite set of data. On a variety of simulated nonlinear dynamical systems, we empirically demonstrate the efficacy of learned versions of this controller in stable trajectory tracking. Alongside our learning method, we evaluate recent ideas in jointly learning a controller and stabilizability certificate for known dynamical systems; we show experimentally that such methods can be frail in comparison.

        ----

        ## [1206] The Edge of Orthogonality: A Simple View of What Makes BYOL Tick

        **Authors**: *Pierre Harvey Richemond, Allison C. Tam, Yunhao Tang, Florian Strub, Bilal Piot, Felix Hill*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/richemond23a.html](https://proceedings.mlr.press/v202/richemond23a.html)

        **Abstract**:

        Self-predictive unsupervised learning methods such as BYOL or SimSIAM have shown impressive results, and counter-intuitively, do not collapse to trivial representations. In this work, we aim at exploring the simplest possible mathematical arguments towards explaining the underlying mechanisms behind self-predictive unsupervised learning. We start with the observation that those methods crucially rely on the presence of a predictor network (and stop-gradient). With simple linear algebra, we show that when using a linear predictor, the optimal predictor is close to an orthogonal projection, and propose a general framework based on orthonormalization that enables to interpret and give intuition on why BYOL works. In addition, this framework demonstrates the crucial role of the exponential moving average and stop-gradient operator in BYOL as an efficient orthonormalization mechanism. We use these insights to propose four new closed-form predictor variants of BYOL to support our analysis. Our closed-form predictors outperform standard linear trainable predictor BYOL at 100 and 300 epochs (top-1 linear accuracy on ImageNet).

        ----

        ## [1207] Multi-Agent Best Arm Identification with Private Communications

        **Authors**: *Alexandre Rio, Merwan Barlier, Igor Colin, Marta Soare*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/rio23a.html](https://proceedings.mlr.press/v202/rio23a.html)

        **Abstract**:

        We address multi-agent best arm identification with privacy guarantees. In this setting, agents collaborate by communicating to find the optimal arm. To avoid leaking sensitive data through messages, we consider two notions of privacy withholding different kinds of information: differential privacy and $(\epsilon, \eta)$-privacy. For each privacy definition, we propose an algorithm based on a two-level successive elimination scheme. We provide theoretical guarantees for the privacy level, accuracy and sample complexity of our algorithms. Experiments on various settings support our theoretical findings.

        ----

        ## [1208] A Two-Stage Active Learning Algorithm for k-Nearest Neighbors

        **Authors**: *Nicholas Rittler, Kamalika Chaudhuri*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/rittler23a.html](https://proceedings.mlr.press/v202/rittler23a.html)

        **Abstract**:

        $k$-nearest neighbor classification is a popular non-parametric method because of desirable properties like automatic adaption to distributional scale changes. Unfortunately, it has thus far proved difficult to design active learning strategies for the training of local voting-based classifiers that naturally retain these desirable properties, and hence active learning strategies for $k$-nearest neighbor classification have been conspicuously missing from the literature. In this work, we introduce a simple and intuitive active learning algorithm for the training of $k$-nearest neighbor classifiers, the first in the literature which retains the concept of the $k$-nearest neighbor vote at prediction time. We provide consistency guarantees for a modified $k$-nearest neighbors classifier trained on samples acquired via our scheme, and show that when the conditional probability function $\mathbb{P}(Y=y|X=x)$ is sufficiently smooth and the Tsybakov noise condition holds, our actively trained classifiers converge to the Bayes optimal classifier at a faster asymptotic rate than passively trained $k$-nearest neighbor classifiers.

        ----

        ## [1209] Lowering the Pre-training Tax for Gradient-based Subset Training: A Lightweight Distributed Pre-Training Toolkit

        **Authors**: *Yeonju Ro, Zhangyang Wang, Vijay Chidambaram, Aditya Akella*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/ro23a.html](https://proceedings.mlr.press/v202/ro23a.html)

        **Abstract**:

        Training data and model sizes are increasing exponentially. One way to reduce training time and resources is to train with a carefully selected subset of the full dataset. Prior work uses the gradient signals obtained during a warm-up or “pre-training" phase over the full dataset, for determining the core subset; if the pre-training phase is too small, the gradients obtained are chaotic and unreliable. As a result, the pre-training phase itself incurs significant time/resource overhead, and prior work has not gone beyond hyperparameter search to reduce pre-training time. Our work explicitly aims to reduce this $\textbf{pre-training tax}$ in gradient-based subset training. We develop a principled, scalable approach for pre-training in a distributed setup. Our approach is $\textit{lightweight}$ and $\textit{minimizes communication}$ between distributed worker nodes. It is the first to utilize the concept of model-soup based distributed training $\textit{at initialization}$. The key idea is to minimally train an ensemble of models on small, disjointed subsets of the data; we further employ data-driven sparsity and data augmentation for local worker training to boost ensemble diversity. The centralized model, obtained at the end of pre-training by merging the per-worker models, is found to offer stabilized gradient signals to select subsets, on which the main model is further trained. We have validated the effectiveness of our method through extensive experiments on CIFAR-10/100, and ImageNet, using ResNet and WideResNet models. For example, our approach is shown to achieve $\textbf{15.4$\times$}$ pre-training speedup and $\textbf{2.8$\times$}$ end-to-end speedup on CIFAR10 and ResNet18 without loss of accuracy. The code is at https://github.com/moonbucks/LiPT.git.

        ----

        ## [1210] The Role of Entropy and Reconstruction in Multi-View Self-Supervised Learning

        **Authors**: *Borja Rodríguez Gálvez, Arno Blaas, Pau Rodríguez, Adam Golinski, Xavier Suau, Jason Ramapuram, Dan Busbridge, Luca Zappella*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/rodri-guez-galvez23a.html](https://proceedings.mlr.press/v202/rodri-guez-galvez23a.html)

        **Abstract**:

        The mechanisms behind the success of multi-view self-supervised learning (MVSSL) are not yet fully understood. Contrastive MVSSL methods have been studied through the lens of InfoNCE, a lower bound of the Mutual Information (MI). However, the relation between other MVSSL methods and MI remains unclear. We consider a different lower bound on the MI consisting of an entropy and a reconstruction term (ER), and analyze the main MVSSL families through its lens. Through this ER bound, we show that clustering-based methods such as DeepCluster and SwAV maximize the MI. We also re-interpret the mechanisms of distillation-based approaches such as BYOL and DINO, showing that they explicitly maximize the reconstruction term and implicitly encourage a stable entropy, and we confirm this empirically. We show that replacing the objectives of common MVSSL methods with this ER bound achieves competitive performance, while making them stable when training with smaller batch sizes or smaller exponential moving average (EMA) coefficients.

        ----

        ## [1211] RLang: A Declarative Language for Describing Partial World Knowledge to Reinforcement Learning Agents

        **Authors**: *Rafael Rodríguez-Sánchez, Benjamin Adin Spiegel, Jennifer Wang, Roma Patel, Stefanie Tellex, George Konidaris*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/rodriguez-sanchez23a.html](https://proceedings.mlr.press/v202/rodriguez-sanchez23a.html)

        **Abstract**:

        We introduce RLang, a domain-specific language (DSL) for communicating domain knowledge to an RL agent. Unlike existing RL DSLs that ground to $\textit{single}$ elements of a decision-making formalism (e.g., the reward function or policy), RLang can specify information about every element of a Markov decision process. We define precise syntax and grounding semantics for RLang, and provide a parser that grounds RLang programs to an algorithm-agnostic $\textit{partial}$ world model and policy that can be exploited by an RL agent. We provide a series of example RLang programs demonstrating how different RL methods can exploit the resulting knowledge, encompassing model-free and model-based tabular algorithms, policy gradient and value-based methods, hierarchical approaches, and deep methods.

        ----

        ## [1212] Improving Fair Training under Correlation Shifts

        **Authors**: *Yuji Roh, Kangwook Lee, Steven Euijong Whang, Changho Suh*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/roh23a.html](https://proceedings.mlr.press/v202/roh23a.html)

        **Abstract**:

        Model fairness is an essential element for Trustworthy AI. While many techniques for model fairness have been proposed, most of them assume that the training and deployment data distributions are identical, which is often not true in practice. In particular, when the bias between labels and sensitive groups changes, the fairness of the trained model is directly influenced and can worsen. We make two contributions for solving this problem. First, we analytically show that existing in-processing fair algorithms have fundamental limits in accuracy and group fairness. We utilize the notion of correlation shifts between labels and groups, which can explicitly capture the change of the above bias. Second, we propose a novel pre-processing step that samples the input data to reduce correlation shifts and thus enables the in-processing approaches to overcome their limitations. We formulate an optimization problem for adjusting the data ratio among labels and sensitive groups to reflect the shifted correlation. A key benefit of our approach lies in decoupling the roles of pre- and in-processing approaches: correlation adjustment via pre-processing and unfairness mitigation on the processed data via in-processing. Experiments show that our framework effectively improves existing in-processing fair algorithms w.r.t. accuracy and fairness, both on synthetic and real datasets.

        ----

        ## [1213] The Statistical Benefits of Quantile Temporal-Difference Learning for Value Estimation

        **Authors**: *Mark Rowland, Yunhao Tang, Clare Lyle, Rémi Munos, Marc G. Bellemare, Will Dabney*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/rowland23a.html](https://proceedings.mlr.press/v202/rowland23a.html)

        **Abstract**:

        We study the problem of temporal-difference-based policy evaluation in reinforcement learning. In particular, we analyse the use of a distributional reinforcement learning algorithm, quantile temporal-difference learning (QTD), for this task. We reach the surprising conclusion that even if a practitioner has no interest in the return distribution beyond the mean, QTD (which learns predictions about the full distribution of returns) may offer performance superior to approaches such as classical TD learning, which predict only the mean return, even in the tabular setting.

        ----

        ## [1214] Robust Satisficing MDPs

        **Authors**: *Haolin Ruan, Siyu Zhou, Zhi Chen, Chin Pang Ho*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/ruan23a.html](https://proceedings.mlr.press/v202/ruan23a.html)

        **Abstract**:

        Despite being a fundamental building block for reinforcement learning, Markov decision processes (MDPs) often suffer from ambiguity in model parameters. Robust MDPs are proposed to overcome this challenge by optimizing the worst-case performance under ambiguity. While robust MDPs can provide reliable policies with limited data, their worst-case performances are often overly conservative, and so they do not offer practical insights into the actual performance of these reliable policies. This paper proposes robust satisficing MDPs (RSMDPs), where the expected returns of feasible policies are softly-constrained to achieve a user-specified target under ambiguity. We derive a tractable reformulation for RSMDPs and develop a first-order method for solving large instances. Experimental results demonstrate that RSMDPs can prescribe policies to achieve their targets, which are much higher than the optimal worst-case returns computed by robust MDPs. Moreover, the average and percentile performances of our model are competitive among other models. We also demonstrate the scalability of the proposed algorithm compared with a state-of-the-art commercial solver.

        ----

        ## [1215] Infinite Action Contextual Bandits with Reusable Data Exhaust

        **Authors**: *Mark Rucker, Yinglun Zhu, Paul Mineiro*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/rucker23a.html](https://proceedings.mlr.press/v202/rucker23a.html)

        **Abstract**:

        For infinite action contextual bandits, smoothed regret and reduction to regression results in state-of-the-art online performance with computational cost independent of the action set: unfortunately, the resulting data exhaust does not have well-defined importance-weights. This frustrates the execution of downstream data science processes such as offline model selection. In this paper we describe an online algorithm with an equivalent smoothed regret guarantee, but which generates well-defined importance weights: in exchange, the online computational cost increases, but only to order smoothness (i.e., still independent of the action set). This removes a key obstacle to adoption of smoothed regret in production scenarios.

        ----

        ## [1216] Function-Space Regularization in Neural Networks: A Probabilistic Perspective

        **Authors**: *Tim G. J. Rudner, Sanyam Kapoor, Shikai Qiu, Andrew Gordon Wilson*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/rudner23a.html](https://proceedings.mlr.press/v202/rudner23a.html)

        **Abstract**:

        Parameter-space regularization in neural network optimization is a fundamental tool for improving generalization. However, standard parameter-space regularization methods make it challenging to encode explicit preferences about desired predictive functions into neural network training. In this work, we approach regularization in neural networks from a probabilistic perspective and show that by viewing parameter-space regularization as specifying an empirical prior distribution over the model parameters, we can derive a probabilistically well-motivated regularization technique that allows explicitly encoding information about desired predictive functions into neural network training. This method—which we refer to as function-space empirical Bayes (FS-EB)—includes both parameter- and function-space regularization, is mathematically simple, easy to implement, and incurs only minimal computational overhead compared to standard regularization techniques. We evaluate the utility of this regularization technique empirically and demonstrate that the proposed method leads to near-perfect semantic shift detection, highly-calibrated predictive uncertainty estimates, successful task adaption from pre-trained models, and improved generalization under covariate shift.

        ----

        ## [1217] A New PHO-rmula for Improved Performance of Semi-Structured Networks

        **Authors**: *David Rügamer*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/rugamer23a.html](https://proceedings.mlr.press/v202/rugamer23a.html)

        **Abstract**:

        Recent advances to combine structured regression models and deep neural networks for better interpretability, more expressiveness, and statistically valid uncertainty quantification demonstrate the versatility of semi-structured neural networks (SSNs). We show that techniques to properly identify the contributions of the different model components in SSNs, however, lead to suboptimal network estimation, slower convergence, and degenerated or erroneous predictions. In order to solve these problems while preserving favorable model properties, we propose a non-invasive post-hoc orthogonalization (PHO) that guarantees identifiability of model components and provides better estimation and prediction quality. Our theoretical findings are supported by numerical experiments, a benchmark comparison as well as a real-world application to COVID-19 infections.

        ----

        ## [1218] Geometric Clifford Algebra Networks

        **Authors**: *David Ruhe, Jayesh K. Gupta, Steven De Keninck, Max Welling, Johannes Brandstetter*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/ruhe23a.html](https://proceedings.mlr.press/v202/ruhe23a.html)

        **Abstract**:

        We propose Geometric Clifford Algebra Networks (GCANs) for modeling dynamical systems. GCANs are based on symmetry group transformations using geometric (Clifford) algebras. We first review the quintessence of modern (plane-based) geometric algebra, which builds on isometries encoded as elements of the $\mathrm{Pin}(p,q,r)$ group. We then propose the concept of group action layers, which linearly combine object transformations using pre-specified group actions. Together with a new activation and normalization scheme, these layers serve as adjustable geometric templates that can be refined via gradient descent. Theoretical advantages are strongly reflected in the modeling of three-dimensional rigid body transformations as well as large-scale fluid dynamics simulations, showing significantly improved performance over traditional methods.

        ----

        ## [1219] Constrained Monotonic Neural Networks

        **Authors**: *Davor Runje, Sharath M. Shankaranarayana*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/runje23a.html](https://proceedings.mlr.press/v202/runje23a.html)

        **Abstract**:

        Wider adoption of neural networks in many critical domains such as finance and healthcare is being hindered by the need to explain their predictions and to impose additional constraints on them. Monotonicity constraint is one of the most requested properties in real-world scenarios and is the focus of this paper. One of the oldest ways to construct a monotonic fully connected neural network is to constrain signs on its weights. Unfortunately, this construction does not work with popular non-saturated activation functions as it can only approximate convex functions. We show this shortcoming can be fixed by constructing two additional activation functions from a typical unsaturated monotonic activation function and employing each of them on the part of neurons. Our experiments show this approach of building monotonic neural networks has better accuracy when compared to other state-of-the-art methods, while being the simplest one in the sense of having the least number of parameters, and not requiring any modifications to the learning procedure or post-learning steps. Finally, we prove it can approximate any continuous monotone function on a compact subset of $\mathbb{R}^n$.

        ----

        ## [1220] Differential Privacy, Linguistic Fairness, and Training Data Influence: Impossibility and Possibility Theorems for Multilingual Language Models

        **Authors**: *Phillip Rust, Anders Søgaard*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/rust23a.html](https://proceedings.mlr.press/v202/rust23a.html)

        **Abstract**:

        Language models such as mBERT, XLM-R, and BLOOM aim to achieve multilingual generalization or compression to facilitate transfer to a large number of (potentially unseen) languages. However, these models should ideally also be private, linguistically fair, and transparent, by relating their predictions to training data. Can these requirements be simultaneously satisfied? We show that multilingual compression and linguistic fairness are compatible with differential privacy, but that differential privacy is at odds with training data influence sparsity, an objective for transparency. We further present a series of experiments on two common NLP tasks and evaluate multilingual compression and training data influence sparsity under different privacy guarantees, exploring these trade-offs in more detail. Our results suggest that we need to develop ways to jointly optimize for these objectives in order to find practical trade-offs.

        ----

        ## [1221] Intrinsic Sliced Wasserstein Distances for Comparing Collections of Probability Distributions on Manifolds and Graphs

        **Authors**: *Raif M. Rustamov, Subhabrata Majumdar*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/rustamov23a.html](https://proceedings.mlr.press/v202/rustamov23a.html)

        **Abstract**:

        Collections of probability distributions arise in a variety of applications ranging from user activity pattern analysis to brain connectomics. In practice these distributions can be defined over diverse domain types including finite intervals, circles, cylinders, spheres, other manifolds, and graphs. This paper introduces an approach for detecting differences between two collections of distributions over such general domains. To this end, we propose the intrinsic slicing construction that yields a novel class of Wasserstein distances on manifolds and graphs. These distances are Hilbert embeddable, allowing us to reduce the distribution collection comparison problem to a more familiar mean testing problem in a Hilbert space. We provide two testing procedures one based on resampling and another on combining p-values from coordinate-wise tests. Our experiments in various synthetic and real data settings show that the resulting tests are powerful and the p-values are well-calibrated.

        ----

        ## [1222] SWARM Parallelism: Training Large Models Can Be Surprisingly Communication-Efficient

        **Authors**: *Max Ryabinin, Tim Dettmers, Michael Diskin, Alexander Borzunov*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/ryabinin23a.html](https://proceedings.mlr.press/v202/ryabinin23a.html)

        **Abstract**:

        Many deep learning applications benefit from using large models with billions of parameters. Training these models is notoriously expensive due to the need for specialized HPC clusters. In this work, we consider alternative setups for training large models: using cheap “preemptible” instances or pooling existing resources from multiple regions. We analyze the performance of existing model-parallel algorithms in these conditions and find configurations where training larger models becomes less communication-intensive. Based on these findings, we propose SWARM Parallelism (Stochastically Wired Adaptively Rebalanced Model Parallelism), a model-parallel training algorithm designed for poorly connected, heterogeneous and unreliable devices. SWARM creates temporary randomized pipelines between nodes that are rebalanced in case of failure. We empirically validate our findings and compare SWARM Parallelism with existing large-scale training approaches. Finally, we combine our insights with compression strategies to train a large Transformer language model with 1B shared parameters ($\approx$13B before sharing) on preemptible T4 GPUs with less than 200 Mb/s network.

        ----

        ## [1223] Hiera: A Hierarchical Vision Transformer without the Bells-and-Whistles

        **Authors**: *Chaitanya Ryali, Yuan-Ting Hu, Daniel Bolya, Chen Wei, Haoqi Fan, Po-Yao Huang, Vaibhav Aggarwal, Arkabandhu Chowdhury, Omid Poursaeed, Judy Hoffman, Jitendra Malik, Yanghao Li, Christoph Feichtenhofer*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/ryali23a.html](https://proceedings.mlr.press/v202/ryali23a.html)

        **Abstract**:

        Modern hierarchical vision transformers have added several vision-specific components in the pursuit of supervised classification performance. While these components lead to effective accuracies and attractive FLOP counts, the added complexity actually makes these transformers slower than their vanilla ViT counterparts. In this paper, we argue that this additional bulk is unnecessary. By pretraining with a strong visual pretext task (MAE), we can strip out all the bells-and-whistles from a state-of-the-art multi-stage vision transformer without losing accuracy. In the process, we create Hiera, an extremely simple hierarchical vision transformer that is more accurate than previous models while being significantly faster both at inference and during training. We evaluate Hiera on a variety of tasks for image and video recognition. Our code and models are available at https://github.com/facebookresearch/hiera.

        ----

        ## [1224] End-to-End Learning for Stochastic Optimization: A Bayesian Perspective

        **Authors**: *Yves Rychener, Daniel Kuhn, Tobias Sutter*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/rychener23a.html](https://proceedings.mlr.press/v202/rychener23a.html)

        **Abstract**:

        We develop a principled approach to end-to-end learning in stochastic optimization. First, we show that the standard end-to-end learning algorithm admits a Bayesian interpretation and trains a posterior Bayes action map. Building on the insights of this analysis, we then propose new end-to-end learning algorithms for training decision maps that output solutions of empirical risk minimization and distributionally robust optimization problems, two dominant modeling paradigms in optimization under uncertainty. Numerical results for a synthetic newsvendor problem illustrate the key differences between alternative training schemes. We also investigate an economic dispatch problem based on real data to showcase the impact of the neural network architecture of the decision maps on their test performance.

        ----

        ## [1225] Sequential Monte Carlo Learning for Time Series Structure Discovery

        **Authors**: *Feras Saad, Brian Patton, Matthew Douglas Hoffman, Rif A. Saurous, Vikash Mansinghka*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/saad23a.html](https://proceedings.mlr.press/v202/saad23a.html)

        **Abstract**:

        This paper presents a new approach to automatically discovering accurate models of complex time series data. Working within a Bayesian nonparametric prior over a symbolic space of Gaussian process time series models, we present a novel structure learning algorithm that integrates sequential Monte Carlo (SMC) and involutive MCMC for highly effective posterior inference. Our method can be used both in "online” settings, where new data is incorporated sequentially in time, and in “offline” settings, by using nested subsets of historical data to anneal the posterior. Empirical measurements on real-world time series show that our method can deliver 10x–100x runtime speedups over previous MCMC and greedy-search structure learning algorithms targeting the same model family. We use our method to perform the first large-scale evaluation of Gaussian process time series structure learning on a prominent benchmark of 1,428 econometric datasets. The results show that our method discovers sensible models that deliver more accurate point forecasts and interval forecasts over multiple horizons as compared to widely used statistical and neural baselines that struggle on this challenging data.

        ----

        ## [1226] Active Ranking of Experts Based on their Performances in Many Tasks

        **Authors**: *El Mehdi Saad, Nicolas Verzelen, Alexandra Carpentier*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/saad23b.html](https://proceedings.mlr.press/v202/saad23b.html)

        **Abstract**:

        We consider the problem of ranking n experts based on their performances on d tasks. We make a monotonicity assumption stating that for each pair of experts, one outperforms the other on all tasks. We consider the sequential setting where in each round the learner has access to noisy evaluations of actively chosen pair of expert-task, given the information available up to the actual round. Given a confidence parameter $\delta \in (0, 1)$, we provide strategies allowing to recover the correct ranking of experts and develop a bound on the total number of queries made by our algorithm that hold with probability at least $1-\delta$. We show that our strategy is adaptive to the complexity of the problem (our bounds are instance dependent), and develop matching lower bounds up to a ploy-logarithmic factor. Finally, we adapt our strategy to the relaxed problem of best expert identification and provide numerical simulation consistent with our theoretical results

        ----

        ## [1227] Sample Complexity Bounds for Learning High-dimensional Simplices in Noisy Regimes

        **Authors**: *Seyed Amir Hossein Saberi, Amir Najafi, Abolfazl S. Motahari, Babak H. Khalaj*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/saberi23a.html](https://proceedings.mlr.press/v202/saberi23a.html)

        **Abstract**:

        In this paper, we propose sample complexity bounds for learning a simplex from noisy samples. A dataset of size $n$ is given which includes i.i.d. samples drawn from a uniform distribution over an unknown arbitrary simplex in $\mathbb{R}^K$, where samples are assumed to be corrupted by a multi-variate additive Gaussian noise of an arbitrary magnitude. We prove the existence of an algorithm that with high probability outputs a simplex having a $\ell_2$ distance of at most $\varepsilon$ from the true simplex (for any $\varepsilon>0$). Also, we theoretically show that in order to achieve this bound, it is sufficient to have $n\ge\tilde{\Omega}\left(K^2/\varepsilon^2\right)e^{\Omega\left(K/\mathrm{SNR}^2\right)}$ samples, where $\mathrm{SNR}$ stands for the signal-to-noise ratio and is defined as the ratio of the maximum component-wise standard deviation of the simplex (signal) to that of the noise vector. This result solves an important open problem in this area of research, and shows as long as $\mathrm{SNR}\ge\Omega\left(\sqrt{K}\right)$ the sample complexity of the noisy regime has the same order to that of the noiseless case. Our proofs are a combination of the so-called sample compression technique in (Ashtiani et al., 2018), mathematical tools from high-dimensional geometry, and Fourier analysis. In particular, we have proposed a general Fourier-based technique for recovery of a more general class of distribution families from additive Gaussian noise, which can be further used in a variety of other related problems.

        ----

        ## [1228] Global Selection of Contrastive Batches via Optimization on Sample Permutations

        **Authors**: *Vin Sachidananda, Ziyi Yang, Chenguang Zhu*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/sachidananda23a.html](https://proceedings.mlr.press/v202/sachidananda23a.html)

        **Abstract**:

        Contrastive Learning has recently achieved state-of-the-art performance in a wide range of unimodal and multimodal tasks. Many contrastive learning approaches use mined hard negatives to make batches more informative during training but these approaches are inefficient as they increase epoch length proportional to the number of mined negatives and require frequent updates of nearest neighbor indices or mining from recent batches. In this work, we provide an alternative to hard negative mining, Global Contrastive Batch Sampling (GCBS), an efficient approximation to the batch assignment problem that upper bounds the gap between the global and training losses, $\mathcal{L}^{Global} - \mathcal{L}^{Train}$, in contrastive learning settings. Through experimentation we find GCBS improves state-of-the-art performance in sentence embedding and code-search tasks. Additionally, GCBS is easy to implement as it requires only a few additional lines of code, does not maintain external data structures such as nearest neighbor indices, is more computationally efficient than the most minimal hard negative mining approaches, and makes no changes to the model being trained. Code is available at https://github.com/vinayak1/GCBS.

        ----

        ## [1229] High-Probability Bounds for Stochastic Optimization and Variational Inequalities: the Case of Unbounded Variance

        **Authors**: *Abdurakhmon Sadiev, Marina Danilova, Eduard Gorbunov, Samuel Horváth, Gauthier Gidel, Pavel E. Dvurechensky, Alexander V. Gasnikov, Peter Richtárik*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/sadiev23a.html](https://proceedings.mlr.press/v202/sadiev23a.html)

        **Abstract**:

        During the recent years the interest of optimization and machine learning communities in high-probability convergence of stochastic optimization methods has been growing. One of the main reasons for this is that high-probability complexity bounds are more accurate and less studied than in-expectation ones. However, SOTA high-probability non-asymptotic convergence results are derived under strong assumptions such as boundedness of the gradient noise variance or of the objective’s gradient itself. In this paper, we propose several algorithms with high-probability convergence results under less restrictive assumptions. In particular, we derive new high-probability convergence results under the assumption that the gradient/operator noise has bounded central $\alpha$-th moment for $\alpha \in (1,2]$ in the following setups: (i) smooth non-convex / Polyak-Lojasiewicz / convex / strongly convex / quasi-strongly convex minimization problems, (ii) Lipschitz / star-cocoercive and monotone / quasi-strongly monotone variational inequalities. These results justify the usage of the considered methods for solving problems that do not fit standard functional classes studied in stochastic optimization.

        ----

        ## [1230] End-to-end Differentiable Clustering with Associative Memories

        **Authors**: *Bishwajit Saha, Dmitry Krotov, Mohammed J. Zaki, Parikshit Ram*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/saha23a.html](https://proceedings.mlr.press/v202/saha23a.html)

        **Abstract**:

        Clustering is a widely used unsupervised learning technique involving an intensive discrete optimization problem. Associative Memory models or AMs are differentiable neural networks defining a recursive dynamical system, which have been integrated with various deep learning architectures. We uncover a novel connection between the AM dynamics and the inherent discrete assignment necessary in clustering to propose a novel unconstrained continuous relaxation of the discrete clustering problem, enabling end-to-end differentiable clustering with AM, dubbed ClAM. Leveraging the pattern completion ability of AMs, we further develop a novel self-supervised clustering loss. Our evaluations on varied datasets demonstrate that ClAM benefits from the self-supervision, and significantly improves upon both the traditional Lloyd’s k-means algorithm, and more recent continuous clustering relaxations (by upto 60% in terms of the Silhouette Coefficient).

        ----

        ## [1231] Learning to Suggest Breaks: Sustainable Optimization of Long-Term User Engagement

        **Authors**: *Eden Saig, Nir Rosenfeld*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/saig23a.html](https://proceedings.mlr.press/v202/saig23a.html)

        **Abstract**:

        Optimizing user engagement is a key goal for modern recommendation systems, but blindly pushing users towards increased consumption risks burn-out, churn, or even addictive habits. To promote digital well-being, most platforms now offer a service that periodically prompts users to take breaks. These, however, must be set up manually, and so may be suboptimal for both users and the system. In this paper, we study the role of breaks in recommendation, and propose a framework for learning optimal breaking policies that promote and sustain long-term engagement. Based on the notion that recommendation dynamics are susceptible to both positive and negative feedback, we cast recommendation as a Lotka-Volterra dynamical system, where breaking reduces to a problem of optimal control. We then give an efficient learning algorithm, provide theoretical guarantees, and empirically demonstrate the utility of our approach on semi-synthetic data.

        ----

        ## [1232] Multi-class Graph Clustering via Approximated Effective p-Resistance

        **Authors**: *Shota Saito, Mark Herbster*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/saito23a.html](https://proceedings.mlr.press/v202/saito23a.html)

        **Abstract**:

        This paper develops an approximation to the (effective) $p$-resistance and applies it to multi-class clustering. Spectral methods based on the graph Laplacian and its generalization to the graph $p$-Laplacian have been a backbone of non-euclidean clustering techniques. The advantage of the $p$-Laplacian is that the parameter $p$ induces a controllable bias on cluster structure. The drawback of $p$-Laplacian eigenvector based methods is that the third and higher eigenvectors are difficult to compute. Thus, instead, we are motivated to use the $p$-resistance induced by the $p$-Laplacian for clustering. For $p$-resistance, small $p$ biases towards clusters with high internal connectivity while large $p$ biases towards clusters of small “extent,” that is a preference for smaller shortest-path distances between vertices in the cluster. However, the $p$-resistance is expensive to compute. We overcome this by developing an approximation to the $p$-resistance. We prove upper and lower bounds on this approximation and observe that it is exact when the graph is a tree. We also provide theoretical justification for the use of $p$-resistance for clustering. Finally, we provide experiments comparing our approximated $p$-resistance clustering to other $p$-Laplacian based methods.

        ----

        ## [1233] Off-Policy Evaluation for Large Action Spaces via Conjunct Effect Modeling

        **Authors**: *Yuta Saito, Qingyang Ren, Thorsten Joachims*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/saito23b.html](https://proceedings.mlr.press/v202/saito23b.html)

        **Abstract**:

        We study off-policy evaluation (OPE) of contextual bandit policies for large discrete action spaces where conventional importance-weighting approaches suffer from excessive variance. To circumvent this variance issue, we propose a new estimator, called OffCEM, that is based on the conjunct effect model (CEM), a novel decomposition of the causal effect into a cluster effect and a residual effect. OffCEM applies importance weighting only to action clusters and addresses the residual causal effect through model-based reward estimation. We show that the proposed estimator is unbiased under a new assumption, called local correctness, which only requires that the residual-effect model preserves the relative expected reward differences of the actions within each cluster. To best leverage the CEM and local correctness, we also propose a new two-step procedure for performing model-based estimation that minimizes bias in the first step and variance in the second step. We find that the resulting OffCEM estimator substantially improves bias and variance compared to a range of conventional estimators. Experiments demonstrate that OffCEM provides substantial improvements in OPE especially in the presence of many actions.

        ----

        ## [1234] Rethinking Warm-Starts with Predictions: Learning Predictions Close to Sets of Optimal Solutions for Faster L-/L♮-Convex Function Minimization

        **Authors**: *Shinsaku Sakaue, Taihei Oki*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/sakaue23a.html](https://proceedings.mlr.press/v202/sakaue23a.html)

        **Abstract**:

        An emerging line of work has shown that machine-learned predictions are useful to warm-start algorithms for discrete optimization problems, such as bipartite matching. Previous studies have shown time complexity bounds proportional to some distance between a prediction and an optimal solution, which we can approximately minimize by learning predictions from past optimal solutions. However, such guarantees may not be meaningful when multiple optimal solutions exist. Indeed, the dual problem of bipartite matching and, more generally, $\text{L}$-/$\text{L}^\\natural$-convex function minimization have arbitrarily many optimal solutions, making such prediction-dependent bounds arbitrarily large. To resolve this theoretically critical issue, we present a new warm-start-with-prediction framework for $\text{L}$-/$\text{L}^\\natural$-convex function minimization. Our framework offers time complexity bounds proportional to the distance between a prediction and the set of all optimal solutions. The main technical difficulty lies in learning predictions that are provably close to sets of all optimal solutions, for which we present an online-gradient-descent-based method. We thus give the first polynomial-time learnability of predictions that can provably warm-start algorithms regardless of multiple optimal solutions.

        ----

        ## [1235] PAC-Bayesian Offline Contextual Bandits With Guarantees

        **Authors**: *Otmane Sakhi, Pierre Alquier, Nicolas Chopin*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/sakhi23a.html](https://proceedings.mlr.press/v202/sakhi23a.html)

        **Abstract**:

        This paper introduces a new principled approach for off-policy learning in contextual bandits. Unlike previous work, our approach does not derive learning principles from intractable or loose bounds. We analyse the problem through the PAC-Bayesian lens, interpreting policies as mixtures of decision rules. This allows us to propose novel generalization bounds and provide tractable algorithms to optimize them. We prove that the derived bounds are tighter than their competitors, and can be optimized directly to confidently improve upon the logging policy offline. Our approach learns policies with guarantees, uses all available data and does not require tuning additional hyperparameters on held-out sets. We demonstrate through extensive experiments the effectiveness of our approach in providing performance guarantees in practical scenarios.

        ----

        ## [1236] Provably and Practically Efficient Neural Contextual Bandits

        **Authors**: *Sudeep Salgia*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/salgia23a.html](https://proceedings.mlr.press/v202/salgia23a.html)

        **Abstract**:

        We consider the neural contextual bandit problem. In contrast to the existing work which primarily focuses on ReLU neural nets, we consider a general set of smooth activation functions. Under this more general setting, (i) we derive non-asymptotic error bounds on the difference between an overparameterized neural net and its corresponding neural tangent kernel, (ii) we propose an algorithm with a provable sublinear regret bound that is also efficient in the finite regime as demonstrated by empirical studies. The non-asymptotic error bounds may be of broader interests as a tool to establish the relation between the smoothness of the activation functions in neural contextual bandits and the smoothness of the kernels in kernel bandits.

        ----

        ## [1237] Distributed Linear Bandits under Communication Constraints

        **Authors**: *Sudeep Salgia, Qing Zhao*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/salgia23b.html](https://proceedings.mlr.press/v202/salgia23b.html)

        **Abstract**:

        We consider distributed linear bandits where $M$ agents learn collaboratively to minimize the overall cumulative regret incurred by all agents. Information exchange is facilitated by a central server, and both the uplink and downlink communications are carried over channels with fixed capacity, which limits the amount of information that can be transmitted in each use of the channels. We investigate the regret-communication trade-off by (i) establishing information-theoretic lower bounds on the required communications (in terms of bits) for achieving a sublinear regret order; (ii) developing an efficient algorithm that achieves the minimum sublinear regret order offered by centralized learning using the minimum order of communications dictated by the information-theoretic lower bounds. For sparse linear bandits, we show a variant of the proposed algorithm offers better regret-communication trade-off by leveraging the sparsity of the problem.

        ----

        ## [1238] Optimizing Hyperparameters with Conformal Quantile Regression

        **Authors**: *David Salinas, Jacek Golebiowski, Aaron Klein, Matthias W. Seeger, Cédric Archambeau*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/salinas23a.html](https://proceedings.mlr.press/v202/salinas23a.html)

        **Abstract**:

        Many state-of-the-art hyperparameter optimization (HPO) algorithms rely on model-based optimizers that learn surrogate models of the target function to guide the search. Gaussian processes are the de facto surrogate model due to their ability to capture uncertainty. However, they make strong assumptions about the observation noise, which might not be warranted in practice. In this work, we propose to leverage conformalized quantile regression which makes minimal assumptions about the observation noise and, as a result, models the target function in a more realistic and robust fashion which translates to quicker HPO convergence on empirical benchmarks. To apply our method in a multi-fidelity setting, we propose a simple, yet effective, technique that aggregates observed results across different resource levels and outperforms conventional methods across many empirical tasks.

        ----

        ## [1239] Raising the Cost of Malicious AI-Powered Image Editing

        **Authors**: *Hadi Salman, Alaa Khaddaj, Guillaume Leclerc, Andrew Ilyas, Aleksander Madry*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/salman23a.html](https://proceedings.mlr.press/v202/salman23a.html)

        **Abstract**:

        We present an approach to mitigating the risks of malicious image editing posed by large diffusion models. The key idea is to immunize images so as to make them resistant to manipulation by these models. This immunization relies on injection of imperceptible adversarial perturbations designed to disrupt the operation of the targeted diffusion models, forcing them to generate unrealistic images. We provide two methods for crafting such perturbations, and then demonstrate their efficacy. Finally, we discuss a policy component necessary to make our approach fully effective and practical—one that involves the organizations developing diffusion models, rather than individual users, to implement (and support) the immunization process.

        ----

        ## [1240] Fast, Differentiable and Sparse Top-k: a Convex Analysis Perspective

        **Authors**: *Michael Eli Sander, Joan Puigcerver, Josip Djolonga, Gabriel Peyré, Mathieu Blondel*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/sander23a.html](https://proceedings.mlr.press/v202/sander23a.html)

        **Abstract**:

        The top-$k$ operator returns a $k$-sparse vector, where the non-zero values correspond to the $k$ largest values of the input. Unfortunately, because it is a discontinuous function, it is difficult to incorporate in neural networks trained end-to-end with backpropagation. Recent works have considered differentiable relaxations, based either on regularization or perturbation techniques. However, to date, no approach is fully differentiable and sparse. In this paper, we propose new differentiable and sparse top-$k$ operators. We view the top-$k$ operator as a linear program over the permutahedron, the convex hull of permutations. We then introduce a $p$-norm regularization term to smooth out the operator, and show that its computation can be reduced to isotonic optimization. Our framework is significantly more general than the existing one and allows for example to express top-$k$ operators that select values in magnitude. On the algorithmic side, in addition to pool adjacent violator (PAV) algorithms, we propose a new GPU/TPU-friendly Dykstra algorithm to solve isotonic optimization problems. We successfully use our operators to prune weights in neural networks, to fine-tune vision transformers, and as a router in sparse mixture of experts.

        ----

        ## [1241] TAN Without a Burn: Scaling Laws of DP-SGD

        **Authors**: *Tom Sander, Pierre Stock, Alexandre Sablayrolles*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/sander23b.html](https://proceedings.mlr.press/v202/sander23b.html)

        **Abstract**:

        Differentially Private methods for training Deep Neural Networks (DNNs) have progressed recently, in particular with the use of massive batches and aggregated data augmentations for a large number of training steps. These techniques require much more computing resources than their non-private counterparts, shifting the traditional privacy-accuracy trade-off to a privacy-accuracy-compute trade-off and making hyper-parameter search virtually impossible for realistic scenarios. In this work, we decouple privacy analysis and experimental behavior of noisy training to explore the trade-off with minimal computational requirements. We first use the tools of Renyi Differential Privacy (RDP) to highlight that the privacy budget, when not overcharged, only depends on the total amount of noise (TAN) injected throughout training. We then derive scaling laws for training models with DP-SGD to optimize hyper-parameters with more than a $100\times$ reduction in computational budget. We apply the proposed method on CIFAR-10 and ImageNet and, in particular, strongly improve the state-of-the-art on ImageNet with a $+9$ points gain in top-1 accuracy for a privacy budget $\varepsilon=8$.

        ----

        ## [1242] Discrete Continuous Optimization Framework for Simultaneous Clustering and Training in Mixture Models

        **Authors**: *Parth Vipul Sangani, Arjun Shashank Kashettiwar, Pritish Chakraborty, Bhuvan Reddy Gangula, Durga Sivasubramanian, Ganesh Ramakrishnan, Rishabh K. Iyer, Abir De*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/sangani23a.html](https://proceedings.mlr.press/v202/sangani23a.html)

        **Abstract**:

        We study a new framework of learning mixture models via automatic clustering called PRESTO, wherein we optimize a joint objective function on the model parameters and the partitioning, with each model tailored to perform well on its specific cluster. In contrast to prior work, we do not assume any generative model for the data. We convert our training problem to a joint parameter estimation cum a subset selection problem, subject to a matroid span constraint. This allows us to reduce our problem into a constrained set function minimization problem, where the underlying objective is monotone and approximately submodular. We then propose a new joint discrete-continuous optimization algorithm that achieves a bounded approximation guarantee for our problem. We show that PRESTO outperforms several alternative methods. Finally, we study PRESTO in the context of resource-efficient deep learning, where we train smaller resource-constrained models on each partition and show that it outperforms existing data partitioning and model pruning/knowledge distillation approaches, which in contrast to PRESTO, require large initial (teacher) models.

        ----

        ## [1243] Whose Opinions Do Language Models Reflect?

        **Authors**: *Shibani Santurkar, Esin Durmus, Faisal Ladhak, Cinoo Lee, Percy Liang, Tatsunori Hashimoto*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/santurkar23a.html](https://proceedings.mlr.press/v202/santurkar23a.html)

        **Abstract**:

        Language models (LMs) are increasingly being used in open-ended contexts, where the opinions they reflect in response to subjective queries can have a profound impact, both on user satisfaction, and shaping the views of society at large. We put forth a quantitative framework to investigate the opinions reflected by LMs – by leveraging high-quality public opinion polls. Using this framework, we create OpinionQA, a dataset for evaluating the alignment of LM opinions with those of 60 US demographic groups over topics ranging from abortion to automation. Across topics, we find substantial misalignment between the views reflected by current LMs and those of US demographic groups: on par with the Democrat-Republican divide on climate change. Notably, this misalignment persists even after explicitly steering the LMs towards particular groups. Our analysis not only confirms prior observations about the left-leaning tendencies of some human feedback-tuned LMs, but also surfaces groups whose opinions are poorly reflected by current LMs (e.g., 65+ and widowed individuals).

        ----

        ## [1244] Streaming Active Learning with Deep Neural Networks

        **Authors**: *Akanksha Saran, Safoora Yousefi, Akshay Krishnamurthy, John Langford, Jordan T. Ash*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/saran23a.html](https://proceedings.mlr.press/v202/saran23a.html)

        **Abstract**:

        Active learning is perhaps most naturally posed as an online learning problem. However, prior active learning approaches with deep neural networks assume offline access to the entire dataset ahead of time. This paper proposes VeSSAL, a new algorithm for batch active learning with deep neural networks in streaming settings, which samples groups of points to query for labels at the moment they are encountered. Our approach trades off between uncertainty and diversity of queried samples to match a desired query rate without requiring any hand-tuned hyperparameters. Altogether, we expand the applicability of deep neural networks to realistic active learning scenarios, such as applications relevant to HCI and large, fractured datasets.

        ----

        ## [1245] Random Teachers are Good Teachers

        **Authors**: *Felix Sarnthein, Gregor Bachmann, Sotiris Anagnostidis, Thomas Hofmann*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/sarnthein23a.html](https://proceedings.mlr.press/v202/sarnthein23a.html)

        **Abstract**:

        In this work, we investigate the implicit regularization induced by teacher-student learning dynamics in self-distillation. To isolate its effect, we describe a simple experiment where we consider teachers at random initialization instead of trained teachers. Surprisingly, when distilling a student into such a random teacher, we observe that the resulting model and its representations already possess very interesting characteristics; (1) we observe a strong improvement of the distilled student over its teacher in terms of probing accuracy. (2) The learned representations are data-dependent and transferable between different tasks but deteriorate strongly if trained on random inputs. (3) The student checkpoint contains sparse subnetworks, so-called lottery tickets, and lies on the border of linear basins in the supervised loss landscape. These observations have interesting consequences for several important areas in machine learning: (1) Self-distillation can work solely based on the implicit regularization present in the gradient dynamics without relying on any dark knowledge, (2) self-supervised learning can learn features even in the absence of data augmentation and (3) training dynamics during the early phase of supervised training do not necessarily require label information. Finally, we shed light on an intriguing local property of the loss landscape: the process of feature learning is strongly amplified if the student is initialized closely to the teacher. These results raise interesting questions about the nature of the landscape that have remained unexplored so far. Code is available at https://github.com/safelix/dinopl.

        ----

        ## [1246] Posterior Sampling for Deep Reinforcement Learning

        **Authors**: *Remo Sasso, Michelangelo Conserva, Paulo E. Rauber*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/sasso23a.html](https://proceedings.mlr.press/v202/sasso23a.html)

        **Abstract**:

        Despite remarkable successes, deep reinforcement learning algorithms remain sample inefficient: they require an enormous amount of trial and error to find good policies. Model-based algorithms promise sample efficiency by building an environment model that can be used for planning. Posterior Sampling for Reinforcement Learning is such a model-based algorithm that has attracted significant interest due to its performance in the tabular setting. This paper introduces Posterior Sampling for Deep Reinforcement Learning (PSDRL), the first truly scalable approximation of Posterior Sampling for Reinforcement Learning that retains its model-based essence. PSDRL combines efficient uncertainty quantification over latent state space models with a specially tailored incremental planning algorithm based on value-function approximation. Extensive experiments on the Atari benchmark show that PSDRL significantly outperforms previous state-of-the-art attempts at scaling up posterior sampling while being competitive with a state-of-the-art (model-based) reinforcement learning method, both in sample efficiency and computational efficiency.

        ----

        ## [1247] Graph Neural Networks can Recover the Hidden Features Solely from the Graph Structure

        **Authors**: *Ryoma Sato*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/sato23a.html](https://proceedings.mlr.press/v202/sato23a.html)

        **Abstract**:

        Graph Neural Networks (GNNs) are popular models for graph learning problems. GNNs show strong empirical performance in many practical tasks. However, the theoretical properties have not been completely elucidated. In this paper, we investigate whether GNNs can exploit the graph structure from the perspective of the expressive power of GNNs. In our analysis, we consider graph generation processes that are controlled by hidden (or latent) node features, which contain all information about the graph structure. A typical example of this framework is kNN graphs constructed from the hidden features. In our main results, we show that GNNs can recover the hidden node features from the input graph alone, even when all node features, including the hidden features themselves and any indirect hints, are unavailable. GNNs can further use the recovered node features for downstream tasks. These results show that GNNs can fully exploit the graph structure by themselves, and in effect, GNNs can use both the hidden and explicit node features for downstream tasks. In the experiments, we confirm the validity of our results by showing that GNNs can accurately recover the hidden features using a GNN architecture built based on our theoretical analysis.

        ----

        ## [1248] Existence and Estimation of Critical Batch Size for Training Generative Adversarial Networks with Two Time-Scale Update Rule

        **Authors**: *Naoki Sato, Hideaki Iiduka*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/sato23b.html](https://proceedings.mlr.press/v202/sato23b.html)

        **Abstract**:

        Previous results have shown that a two time-scale update rule (TTUR) using different learning rates, such as different constant rates or different decaying rates, is useful for training generative adversarial networks (GANs) in theory and in practice. Moreover, not only the learning rate but also the batch size is important for training GANs with TTURs and they both affect the number of steps needed for training. This paper studies the relationship between batch size and the number of steps needed for training GANs with TTURs based on constant learning rates. We theoretically show that, for a TTUR with constant learning rates, the number of steps needed to find stationary points of the loss functions of both the discriminator and generator decreases as the batch size increases and that there exists a critical batch size minimizing the stochastic first-order oracle (SFO) complexity. Then, we use the Fréchet inception distance (FID) as the performance measure for training and provide numerical results indicating that the number of steps needed to achieve a low FID score decreases as the batch size increases and that the SFO complexity increases once the batch size exceeds the measured critical batch size. Moreover, we show that measured critical batch sizes are close to the sizes estimated from our theoretical results.

        ----

        ## [1249] StyleGAN-T: Unlocking the Power of GANs for Fast Large-Scale Text-to-Image Synthesis

        **Authors**: *Axel Sauer, Tero Karras, Samuli Laine, Andreas Geiger, Timo Aila*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/sauer23a.html](https://proceedings.mlr.press/v202/sauer23a.html)

        **Abstract**:

        Text-to-image synthesis has recently seen significant progress thanks to large pretrained language models, large-scale training data, and the introduction of scalable model families such as diffusion and autoregressive models. However, the best-performing models require iterative evaluation to generate a single sample. In contrast, generative adversarial networks (GANs) only need a single forward pass. They are thus much faster, but they currently remain far behind the state-of-the-art in large-scale text-to-image synthesis. This paper aims to identify the necessary steps to regain competitiveness. Our proposed model, StyleGAN-T, addresses the specific requirements of large-scale text-to-image synthesis, such as large capacity, stable training on diverse datasets, strong text alignment, and controllable variation vs. text alignment tradeoff. StyleGAN-T significantly improves over previous GANs and outperforms distilled diffusion models - the previous state-of-the-art in fast text-to-image synthesis - in terms of sample quality and speed.

        ----

        ## [1250] Facial Expression Recognition with Adaptive Frame Rate based on Multiple Testing Correction

        **Authors**: *Andrey V. Savchenko*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/savchenko23a.html](https://proceedings.mlr.press/v202/savchenko23a.html)

        **Abstract**:

        In this paper, we consider the problem of the high computational complexity of video-based facial expression recognition. A novel sequential procedure is proposed with an adaptive frame rate selection in a short video fragment to speed up decision-making. We automatically adjust the frame rate and process fewer frames with a low frame rate for more straightforward videos and more frames for complex ones. To determine the frame rate at which an inference is sufficiently reliable, the Benjamini-Hochberg procedure from multiple comparisons theory is employed to control the false discovery rate. The main advantages of our method are an improvement of the trustworthiness of decision-making by maintaining only one hyper-parameter (false acceptance rate) and its applicability with arbitrary neural network models used as facial feature extractors without the need to re-train these models. An experimental study on datasets from ABAW and EmotiW challenges proves the superior performance (1.5-40 times faster) of the proposed approach compared to processing all frames and existing techniques with early exiting and adaptive frame selection.

        ----

        ## [1251] Off-Policy Average Reward Actor-Critic with Deterministic Policy Search

        **Authors**: *Naman Saxena, Subhojyoti Khastagir, Shishir Kolathaya, Shalabh Bhatnagar*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/saxena23a.html](https://proceedings.mlr.press/v202/saxena23a.html)

        **Abstract**:

        The average reward criterion is relatively less studied as most existing works in the Reinforcement Learning literature consider the discounted reward criterion. There are few recent works that present on-policy average reward actor-critic algorithms, but average reward off-policy actor-critic is relatively less explored. In this work, we present both on-policy and off-policy deterministic policy gradient theorems for the average reward performance criterion. Using these theorems, we also present an Average Reward Off-Policy Deep Deterministic Policy Gradient (ARO-DDPG) Algorithm. We first show asymptotic convergence analysis using the ODE-based method. Subsequently, we provide a finite time analysis of the resulting stochastic approximation scheme with linear function approximator and obtain an $\epsilon$-optimal stationary policy with a sample complexity of $\Omega(\epsilon^{-2.5})$. We compare the average reward performance of our proposed ARO-DDPG algorithm and observe better empirical performance compared to state-of-the-art on-policy average reward actor-critic algorithms over MuJoCo-based environments.

        ----

        ## [1252] Gibbsian Polar Slice Sampling

        **Authors**: *Philip Schär, Michael Habeck, Daniel Rudolf*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/schar23a.html](https://proceedings.mlr.press/v202/schar23a.html)

        **Abstract**:

        Polar slice sampling (Roberts & Rosenthal, 2002) is a Markov chain approach for approximate sampling of distributions that is difficult, if not impossible, to implement efficiently, but behaves provably well with respect to the dimension. By updating the directional and radial components of chain iterates separately, we obtain a family of samplers that mimic polar slice sampling, and yet can be implemented efficiently. Numerical experiments in a variety of settings indicate that our proposed algorithm outperforms the two most closely related approaches, elliptical slice sampling (Murray et al., 2010) and hit-and-run uniform slice sampling (MacKay, 2003). We prove the well-definedness and convergence of our methods under suitable assumptions on the target distribution.

        ----

        ## [1253] Identifiability and Generalizability in Constrained Inverse Reinforcement Learning

        **Authors**: *Andreas Schlaginhaufen, Maryam Kamgarpour*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/schlaginhaufen23a.html](https://proceedings.mlr.press/v202/schlaginhaufen23a.html)

        **Abstract**:

        Two main challenges in Reinforcement Learning (RL) are designing appropriate reward functions and ensuring the safety of the learned policy. To address these challenges, we present a theoretical framework for Inverse Reinforcement Learning (IRL) in constrained Markov decision processes. From a convex-analytic perspective, we extend prior results on reward identifiability and generalizability to both the constrained setting and a more general class of regularizations. In particular, we show that identifiability up to potential shaping (Cao et al., 2021) is a consequence of entropy regularization and may generally no longer hold for other regularizations or in the presence of safety constraints. We also show that to ensure generalizability to new transition laws and constraints, the true reward must be identified up to a constant. Additionally, we derive a finite sample guarantee for the suboptimality of the learned rewards, and validate our results in a gridworld environment.

        ----

        ## [1254] Learning Expressive Priors for Generalization and Uncertainty Estimation in Neural Networks

        **Authors**: *Dominik Schnaus, Jongseok Lee, Daniel Cremers, Rudolph Triebel*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/schnaus23a.html](https://proceedings.mlr.press/v202/schnaus23a.html)

        **Abstract**:

        In this work, we propose a novel prior learning method for advancing generalization and uncertainty estimation in deep neural networks. The key idea is to exploit scalable and structured posteriors of neural networks as informative priors with generalization guarantees. Our learned priors provide expressive probabilistic representations at large scale, like Bayesian counterparts of pre-trained models on ImageNet, and further produce non-vacuous generalization bounds. We also extend this idea to a continual learning framework, where the favorable properties of our priors are desirable. Major enablers are our technical contributions: (1) the sums-of-Kronecker-product computations, and (2) the derivations and optimizations of tractable objectives that lead to improved generalization bounds. Empirically, we exhaustively show the effectiveness of this method for uncertainty estimation and generalization.

        ----

        ## [1255] Deterministic equivalent and error universality of deep random features learning

        **Authors**: *Dominik Schröder, Hugo Cui, Daniil Dmitriev, Bruno Loureiro*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/schroder23a.html](https://proceedings.mlr.press/v202/schroder23a.html)

        **Abstract**:

        This manuscript considers the problem of learning a random Gaussian network function using a fully connected network with frozen intermediate layers and trainable readout layer. This problem can be seen as a natural generalization of the widely studied random features model to deeper architectures. First, we prove Gaussian universality of the test error in a ridge regression setting where the learner and target networks share the same intermediate layers, and provide a sharp asymptotic formula for it. Establishing this result requires proving a deterministic equivalent for traces of the deep random features sample covariance matrices which can be of independent interest. Second, we conjecture the asymptotic Gaussian universality of the test error in the more general setting of arbitrary convex losses and generic learner/target architectures. We provide extensive numerical evidence for this conjecture, which requires the derivation of closed-form expressions for the layer-wise post-activation population covariances. In light of our results, we investigate the interplay between architecture design and implicit regularization.

        ----

        ## [1256] The Acquisition of Physical Knowledge in Generative Neural Networks

        **Authors**: *Luca M. Schulze Buschoff, Eric Schulz, Marcel Binz*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/schulze-buschoff23a.html](https://proceedings.mlr.press/v202/schulze-buschoff23a.html)

        **Abstract**:

        As children grow older, they develop an intuitive understanding of the physical processes around them. Their physical understanding develops in stages, moving along developmental trajectories which have been mapped out extensively in previous empirical research. Here, we investigate how the learning trajectories of deep generative neural networks compare to children’s developmental trajectories using physical understanding as a testbed. We outline an approach that allows us to examine two distinct hypotheses of human development – stochastic optimization and complexity increase. We find that while our models are able to accurately predict a number of physical processes, their learning trajectories under both hypotheses do not follow the developmental trajectories of children.

        ----

        ## [1257] Modality-Agnostic Variational Compression of Implicit Neural Representations

        **Authors**: *Jonathan Richard Schwarz, Jihoon Tack, Yee Whye Teh, Jaeho Lee, Jinwoo Shin*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/schwarz23a.html](https://proceedings.mlr.press/v202/schwarz23a.html)

        **Abstract**:

        We introduce a modality-agnostic neural compression algorithm based on a functional view of data and parameterised as an Implicit Neural Representation (INR). Bridging the gap between latent coding and sparsity, we obtain compact latent representations non-linearly mapped to a soft gating mechanism. This allows the specialisation of a shared INR network to each data item through subnetwork selection. After obtaining a dataset of such latent representations, we directly optimise the rate/distortion trade-off in a modality-agnostic space using neural compression. Variational Compression of Implicit Neural Representations (VC-INR) shows improved performance given the same representational capacity pre quantisation while also outperforming previous quantisation schemes used for other INR techniques.Our experiments demonstrate strong results over a large set of diverse modalities using the same algorithm without any modality-specific inductive biases. We show results on images, climate data, 3D shapes and scenes as well as audio and video, introducing VC-INR as the first INR-based method to outperform codecs as well-known and diverse as JPEG 2000, MP3 and AVC/HEVC on their respective modalities.

        ----

        ## [1258] Bigger, Better, Faster: Human-level Atari with human-level efficiency

        **Authors**: *Max Schwarzer, Johan Samir Obando-Ceron, Aaron C. Courville, Marc G. Bellemare, Rishabh Agarwal, Pablo Samuel Castro*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/schwarzer23a.html](https://proceedings.mlr.press/v202/schwarzer23a.html)

        **Abstract**:

        We introduce a value-based RL agent, which we call BBF, that achieves super-human performance in the Atari 100K benchmark. BBF relies on scaling the neural networks used for value estimation, as well as a number of other design choices that enable this scaling in a sample-efficient manner. We conduct extensive analyses of these design choices and provide insights for future work. We end with a discussion about updating the goalposts for sample-efficient RL research on the ALE. We make our code and data publicly available at https://github.com/google-research/google-research/tree/master/bigger_better_faster.

        ----

        ## [1259] Dissecting the Effects of SGD Noise in Distinct Regimes of Deep Learning

        **Authors**: *Antonio Sclocchi, Mario Geiger, Matthieu Wyart*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/sclocchi23a.html](https://proceedings.mlr.press/v202/sclocchi23a.html)

        **Abstract**:

        Understanding when the noise in stochastic gradient descent (SGD) affects generalization of deep neural networks remains a challenge, complicated by the fact that networks can operate in distinct training regimes. Here we study how the magnitude of this noise $T$ affects performance as the size of the training set $P$ and the scale of initialization $\alpha$ are varied. For gradient descent, $\alpha$ is a key parameter that controls if the network is lazy’ ($\alpha\gg1$) or instead learns features ($\alpha\ll1$). For classification of MNIST and CIFAR10 images, our central results are: *(i)* obtaining phase diagrams for performance in the $(\alpha,T)$ plane. They show that SGD noise can be detrimental or instead useful depending on the training regime. Moreover, although increasing $T$ or decreasing $\alpha$ both allow the net to escape the lazy regime, these changes can have opposite effects on performance. *(ii)* Most importantly, we find that the characteristic temperature $T_c$ where the noise of SGD starts affecting the trained model (and eventually performance) is a power law of $P$. We relate this finding with the observation that key dynamical quantities, such as the total variation of weights during training, depend on both $T$ and $P$ as power laws. These results indicate that a key effect of SGD noise occurs late in training, by affecting the stopping process whereby all data are fitted. Indeed, we argue that due to SGD noise, nets must develop a strongersignal’, i.e. larger informative weights, to fit the data, leading to a longer training time. A stronger signal and a longer training time are also required when the size of the training set $P$ increases. We confirm these views in the perceptron model, where signal and noise can be precisely measured. Interestingly, exponents characterizing the effect of SGD depend on the density of data near the decision boundary, as we explain.

        ----

        ## [1260] A Fast Optimistic Method for Monotone Variational Inequalities

        **Authors**: *Michael Sedlmayer, Dang-Khoa Nguyen, Radu Ioan Bot*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/sedlmayer23a.html](https://proceedings.mlr.press/v202/sedlmayer23a.html)

        **Abstract**:

        We study monotone variational inequalities that can arise as optimality conditions for constrained convex optimization or convex-concave minimax problems and propose a novel algorithm that uses only one gradient/operator evaluation and one projection onto the constraint set per iteration. The algorithm, which we call fOGDA-VI, achieves a $o(\frac{1}{k})$ rate of convergence in terms of the restricted gap function as well as the natural residual for the last iterate. Moreover, we provide a convergence guarantee for the sequence of iterates to a solution of the variational inequality. These are the best theoretical convergence results for numerical methods for (only) monotone variational inequalities reported in the literature. To empirically validate our algorithm we investigate a two-player matrix game with mixed strategies of the two players. Concluding, we show promising results regarding the application of fOGDA-VI to the training of generative adversarial nets.

        ----

        ## [1261] Double-Weighting for Covariate Shift Adaptation

        **Authors**: *José Ignacio Segovia-Martín, Santiago Mazuelas, Anqi Liu*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/segovia-martin23a.html](https://proceedings.mlr.press/v202/segovia-martin23a.html)

        **Abstract**:

        Supervised learning is often affected by a covariate shift in which the marginal distributions of instances (covariates $x$) of training and testing samples $p_\text{tr}(x)$ and $p_\text{te}(x)$ are different but the label conditionals coincide. Existing approaches address such covariate shift by either using the ratio $p_\text{te}(x)/p_\text{tr}(x)$ to weight training samples (reweighted methods) or using the ratio $p_\text{tr}(x)/p_\text{te}(x)$ to weight testing samples (robust methods). However, the performance of such approaches can be poor under support mismatch or when the above ratios take large values. We propose a minimax risk classification (MRC) approach for covariate shift adaptation that avoids such limitations by weighting both training and testing samples. In addition, we develop effective techniques that obtain both sets of weights and generalize the conventional kernel mean matching method. We provide novel generalization bounds for our method that show a significant increase in the effective sample size compared with reweighted methods. The proposed method also achieves enhanced classification performance in both synthetic and empirical experiments.

        ----

        ## [1262] Enhancing Activity Prediction Models in Drug Discovery with the Ability to Understand Human Language

        **Authors**: *Philipp Seidl, Andreu Vall, Sepp Hochreiter, Günter Klambauer*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/seidl23a.html](https://proceedings.mlr.press/v202/seidl23a.html)

        **Abstract**:

        Activity and property prediction models are the central workhorses in drug discovery and materials sciences, but currently, they have to be trained or fine-tuned for new tasks. Without training or fine-tuning, scientific language models could be used for such low-data tasks through their announced zero- and few-shot capabilities. However, their predictive quality at activity prediction is lacking. In this work, we envision a novel type of activity prediction model that is able to adapt to new prediction tasks at inference time, via understanding textual information describing the task. To this end, we propose a new architecture with separate modules for chemical and natural language inputs, and a contrastive pretraining objective on data from large biochemical databases. In extensive experiments, we show that our method CLAMP yields improved predictive performance on few-shot learning benchmarks and zero-shot problems in drug discovery. We attribute the advances of our method to the modularized architecture and to our pre-training objective.

        ----

        ## [1263] Variational Autoencoding Neural Operators

        **Authors**: *Jacob H. Seidman, Georgios Kissas, George J. Pappas, Paris Perdikaris*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/seidman23a.html](https://proceedings.mlr.press/v202/seidman23a.html)

        **Abstract**:

        Unsupervised learning with functional data is an emerging paradigm of machine learning research with applications to computer vision, climate modeling and physical systems. A natural way of modeling functional data is by learning operators between infinite dimensional spaces, leading to discretization invariant representations that scale independently of the sample grid resolution. Here we present Variational Autoencoding Neural Operators (VANO), a general strategy for making a large class of operator learning architectures act as variational autoencoders. For this purpose, we provide a novel rigorous mathematical formulation of the variational objective in function spaces for training. VANO first maps an input function to a distribution over a latent space using a parametric encoder and then decodes a sample from the latent distribution to reconstruct the input, as in classic variational autoencoders. We test VANO with different model set-ups and architecture choices for a variety of benchmarks. We start from a simple Gaussian random field where we can analytically track what the model learns and progressively transition to more challenging benchmarks including modeling phase separation in Cahn-Hilliard systems and real world satellite data for measuring Earth surface deformation.

        ----

        ## [1264] Neural Markov Jump Processes

        **Authors**: *Patrick Seifner, Ramsés J. Sánchez*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/seifner23a.html](https://proceedings.mlr.press/v202/seifner23a.html)

        **Abstract**:

        Markov jump processes are continuous-time stochastic processes with a wide range of applications in both natural and social sciences. Despite their widespread use, inference in these models is highly non-trivial and typically proceeds via either Monte Carlo or expectation-maximization methods. In this work we introduce an alternative, variational inference algorithm for Markov jump processes which relies on neural ordinary differential equations, and is trainable via back-propagation. Our methodology learns neural, continuous-time representations of the observed data, that are used to approximate the initial distribution and time-dependent transition probability rates of the posterior Markov jump process. The time-independent rates of the prior process are in contrast trained akin to generative adversarial networks. We test our approach on synthetic data sampled from ground-truth Markov jump processes, experimental switching ion channel data and molecular dynamics simulations. Source code to reproduce our experiments is available online.

        ----

        ## [1265] Bayesian online change point detection with Hilbert space approximate Student-t process

        **Authors**: *Jeremy Sellier, Petros Dellaportas*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/sellier23a.html](https://proceedings.mlr.press/v202/sellier23a.html)

        **Abstract**:

        In this paper, we introduce a variant of Bayesian online change point detection with a reducedrank Student-t process (TP) and dependent Student-t noise, as a nonparametric time series model. Our method builds and improves upon the state-of-the-art Gaussian process (GP) change point model benchmark of Saatci et al. (2010). The Student-t process generalizes the concept of a GP and hence yields a more flexible alternative. Additionally, unlike a GP, the predictive variance explicitly depends on the training observations, while the use of an entangled Student-t noise model preserves analytical tractability. Our approach also uses a Hilbert space reduced-rank representation of the TP kernel, derived from an eigenfunction expansion of the Laplace operator (Solin & Sarkka, 2020), to alleviate its computational complexity. Improvements in prediction and training time are demonstrated with real-world data-sets

        ----

        ## [1266] Incentivizing Exploration with Linear Contexts and Combinatorial Actions

        **Authors**: *Mark Sellke*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/sellke23a.html](https://proceedings.mlr.press/v202/sellke23a.html)

        **Abstract**:

        We advance the study of incentivized bandit exploration, in which arm choices are viewed as recommendations and are required to be Bayesian incentive compatible. Recent work of Sellke-Slivkins (Operations Research 2022) has shown that for the special case of independent arms, after collecting enough initial samples, the popular Thompson sampling algorithm becomes incentive compatible. This was generalized to the combinatorial semibandit in Hu-Ngo-Slivkins-Wu (NeurIPS 2022). We give an analog of this result for linear bandits, where the independence of the prior is replaced by a natural convexity condition. This opens up the possibility of efficient and regret-optimal incentivized exploration in high-dimensional action spaces. In the semibandit model, we also improve the sample complexity for the pre-Thompson sampling phase of initial data collection.

        ----

        ## [1267] Explainability as statistical inference

        **Authors**: *Hugo Henri Joseph Sénétaire, Damien Garreau, Jes Frellsen, Pierre-Alexandre Mattei*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/senetaire23a.html](https://proceedings.mlr.press/v202/senetaire23a.html)

        **Abstract**:

        A wide variety of model explanation approaches have been proposed in recent years, all guided by very different rationales and heuristics. In this paper, we take a new route and cast interpretability as a statistical inference problem. We propose a general deep probabilistic model designed to produce interpretable predictions. The model’s parameters can be learned via maximum likelihood, and the method can be adapted to any predictor network architecture, and any type of prediction problem. Our model is akin to amortized interpretability methods, where a neural network is used as a selector to allow for fast interpretation at inference time. Several popular interpretability methods are shown to be particular cases of regularized maximum likelihood for our general model. Using our framework, we identify imputation as a common issue of these models. We propose new datasets with ground truth selection which allow for the evaluation of the features importance map and show experimentally that multiple imputation provides more reasonable interpretations.

        ----

        ## [1268] Multi-View Masked World Models for Visual Robotic Manipulation

        **Authors**: *Younggyo Seo, Junsu Kim, Stephen James, Kimin Lee, Jinwoo Shin, Pieter Abbeel*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/seo23a.html](https://proceedings.mlr.press/v202/seo23a.html)

        **Abstract**:

        Visual robotic manipulation research and applications often use multiple cameras, or views, to better perceive the world. How else can we utilize the richness of multi-view data? In this paper, we investigate how to learn good representations with multi-view data and utilize them for visual robotic manipulation. Specifically, we train a multi-view masked autoencoder which reconstructs pixels of randomly masked viewpoints and then learn a world model operating on the representations from the autoencoder. We demonstrate the effectiveness of our method in a range of scenarios, including multi-view control and single-view control with auxiliary cameras for representation learning. We also show that the multi-view masked autoencoder trained with multiple randomized viewpoints enables training a policy with strong viewpoint randomization and transferring the policy to solve real-robot tasks without camera calibration and an adaptation procedure. Video demonstrations are available at: https://sites.google.com/view/mv-mwm.

        ----

        ## [1269] One-Shot Compression of Large Edge-Exchangeable Graphs using Bits-Back Coding

        **Authors**: *Daniel Severo, James Townsend, Ashish J. Khisti, Alireza Makhzani*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/severo23a.html](https://proceedings.mlr.press/v202/severo23a.html)

        **Abstract**:

        We present a one-shot method for compressing large labeled graphs called Random Edge Coding. When paired with a parameter-free model based on Pólya’s Urn, the worst-case computational and memory complexities scale quasi-linearly and linearly with the number of observed edges, making it efficient on sparse graphs, and requires only integer arithmetic. Key to our method is bits-back coding, which is used to sample edges and vertices without replacement from the edge-list in a way that preserves the structure of the graph. Optimality is proven under a class of random graph models that are invariant to permutations of the edges and of vertices within an edge. Experiments indicate Random Edge Coding can achieve competitive compression performance on real-world network datasets and scales to graphs with millions of nodes and edges.

        ----

        ## [1270] ModelDiff: A Framework for Comparing Learning Algorithms

        **Authors**: *Harshay Shah, Sung Min Park, Andrew Ilyas, Aleksander Madry*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/shah23a.html](https://proceedings.mlr.press/v202/shah23a.html)

        **Abstract**:

        We study the problem of (learning) algorithm comparison, where the goal is to find differences between models trained with two different learning algorithms. We begin by formalizing this goal as one of finding distinguishing feature transformations, i.e., input transformations that change the predictions of models trained with one learning algorithm but not the other. We then present ModelDiff, a method that leverages the datamodels framework (Ilyas et al., 2022) to compare learning algorithms based on how they use their training data. We demonstrate ModelDiff through three case studies, comparing models trained with/without data augmentation, with/without pre-training, and with different SGD hyperparameters.

        ----

        ## [1271] Auxiliary Learning as an Asymmetric Bargaining Game

        **Authors**: *Aviv Shamsian, Aviv Navon, Neta Glazer, Kenji Kawaguchi, Gal Chechik, Ethan Fetaya*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/shamsian23a.html](https://proceedings.mlr.press/v202/shamsian23a.html)

        **Abstract**:

        Auxiliary learning is an effective method for enhancing the generalization capabilities of trained models, particularly when dealing with small datasets. However, this approach may present several difficulties: (i) optimizing multiple objectives can be more challenging, and (ii) how to balance the auxiliary tasks to best assist the main task is unclear. In this work, we propose a novel approach, named AuxiNash, for balancing tasks in auxiliary learning by formalizing the problem as generalized bargaining game with asymmetric task bargaining power. Furthermore, we describe an efficient procedure for learning the bargaining power of tasks based on their contribution to the performance of the main task and derive theoretical guarantees for its convergence. Finally, we evaluate AuxiNash on multiple multi-task benchmarks and find that it consistently outperforms competing methods.

        ----

        ## [1272] Synthetic Prompting: Generating Chain-of-Thought Demonstrations for Large Language Models

        **Authors**: *Zhihong Shao, Yeyun Gong, Yelong Shen, Minlie Huang, Nan Duan, Weizhu Chen*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/shao23a.html](https://proceedings.mlr.press/v202/shao23a.html)

        **Abstract**:

        Large language models can perform various reasoning tasks by using chain-of-thought prompting, which guides them to find answers through step-by-step demonstrations. However, the quality of the prompts depends on the demonstrations given to the models, and creating many of them by hand is costly. We introduce Synthetic prompting, a method that leverages a few handcrafted examples to prompt the model to generate more examples by itself, and selects effective demonstrations to elicit better reasoning. Our method alternates between a backward and forward process to generate new examples. The backward process generates a question that match a sampled reasoning chain, so that the question is solvable and clear. The forward process produces a more detailed reasoning chain for the question, improving the quality of the example. We evaluate our method on numerical, symbolic, and algorithmic reasoning tasks, and show that it outperforms existing prompting techniques.

        ----

        ## [1273] Complementary Attention for Multi-Agent Reinforcement Learning

        **Authors**: *Jianzhun Shao, Hongchang Zhang, Yun Qu, Chang Liu, Shuncheng He, Yuhang Jiang, Xiangyang Ji*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/shao23b.html](https://proceedings.mlr.press/v202/shao23b.html)

        **Abstract**:

        In cooperative multi-agent reinforcement learning, centralized training with decentralized execution (CTDE) shows great promise for a trade-off between independent Q-learning and joint action learning. However, vanilla CTDE methods assumed a fixed number of agents could hardly adapt to real-world scenarios where dynamic team compositions typically suffer from dramatically variant partial observability. Specifically, agents with extensive sight ranges are prone to be affected by trivial environmental substrates, dubbed the "distracted attention" issue; ones with limited observation can hardly sense their teammates, degrading the cooperation quality. In this paper, we propose Complementary Attention for Multi-Agent reinforcement learning (CAMA), which applies a divide-and-conquer strategy on input entities accompanied with the complementary attention of enhancement and replenishment. Concretely, to tackle the distracted attention issue, highly contributed entities’ attention is enhanced by the execution-related representation extracted via action prediction with an inverse model. For better out-of-sight-range cooperation, the lowly contributed ones are compressed to brief messages with a conditional mutual information estimator. Our CAMA facilitates stable and sustainable teamwork, which is justified by the impressive results reported on the challenging StarCraftII, MPE, and Traffic Junction benchmarks.

        ----

        ## [1274] Regularization-free Diffeomorphic Temporal Alignment Nets

        **Authors**: *Ron Shapira Weber, Oren Freifeld*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/shapira-weber23a.html](https://proceedings.mlr.press/v202/shapira-weber23a.html)

        **Abstract**:

        In time-series analysis, nonlinear temporal misalignment is a major problem that forestalls even simple averaging. An effective learning-based solution for this problem is the Diffeomorphic Temporal Alignment Net (DTAN), that, by relying on a diffeomorphic temporal transformer net and the amortization of the joint-alignment task, eliminates drawbacks of traditional alignment methods. Unfortunately, existing DTAN formulations crucially depend on a regularization term whose optimal hyperparameters are dataset-specific and usually searched via a large number of experiments. Here we propose a regularization-free DTAN that obviates the need to perform such an expensive, and often impractical, search. Concretely, we propose a new well-behaved loss that we call the Inverse Consistency Averaging Error (ICAE), as well as a related new triplet loss. Extensive experiments on 128 UCR datasets show that the proposed method outperforms contemporary methods despite not using a regularization. Moreover, ICAE also gives rise to the first DTAN that supports variable-length signals. Our code is available at https://github.com/BGU-CS-VIL/RF-DTAN.

        ----

        ## [1275] Toward Efficient Gradient-Based Value Estimation

        **Authors**: *Arsalan Sharifnassab, Richard S. Sutton*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/sharifnassab23a.html](https://proceedings.mlr.press/v202/sharifnassab23a.html)

        **Abstract**:

        Gradient-based methods for value estimation in reinforcement learning have favorable stability properties, but they are typically much slower than Temporal Difference (TD) learning methods. We study the root causes of this slowness and show that Mean Square Bellman Error (MSBE) is an ill-conditioned loss function in the sense that its Hessian has large condition-number. To resolve the adverse effect of poor conditioning of MSBE on gradient based methods, we propose a low complexity batch-free proximal method that approximately follows the Gauss-Newton direction and is asymptotically robust to parameterization. Our main algorithm, called RANS, is efficient in the sense that it is significantly faster than the residual gradient methods while having almost the same computational complexity, and is competitive with TD on the classic problems that we tested.

        ----

        ## [1276] Coin Sampling: Gradient-Based Bayesian Inference without Learning Rates

        **Authors**: *Louis Sharrock, Christopher Nemeth*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/sharrock23a.html](https://proceedings.mlr.press/v202/sharrock23a.html)

        **Abstract**:

        In recent years, particle-based variational inference (ParVI) methods such as Stein variational gradient descent (SVGD) have grown in popularity as scalable methods for Bayesian inference. Unfortunately, the properties of such methods invariably depend on hyperparameters such as the learning rate, which must be carefully tuned by the practitioner in order to ensure convergence to the target measure at a suitable rate. In this paper, we introduce a suite of new particle-based methods for scalable Bayesian inference based on coin betting, which are entirely learning-rate free. We illustrate the performance of our approach on a range of numerical examples, including several high-dimensional models and datasets, demonstrating comparable performance to other ParVI algorithms with no need to tune a learning rate.

        ----

        ## [1277] On Kinetic Optimal Probability Paths for Generative Models

        **Authors**: *Neta Shaul, Ricky T. Q. Chen, Maximilian Nickel, Matthew Le, Yaron Lipman*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/shaul23a.html](https://proceedings.mlr.press/v202/shaul23a.html)

        **Abstract**:

        Recent successful generative models are trained by fitting a neural network to an a-priori defined tractable probability density path taking noise to training examples. In this paper we investigate the space of Gaussian probability paths, which includes diffusion paths as an instance, and look for an optimal member in some useful sense. In particular, minimizing the Kinetic Energy (KE) of a path is known to make particles’ trajectories simple, hence easier to sample, and empirically improve performance in terms of likelihood of unseen data and sample generation quality. We investigate Kinetic Optimal (KO) Gaussian paths and offer the following observations: (i) We show the KE takes a simplified form on the space of Gaussian paths, where the data is incorporated only through a single, one dimensional scalar function, called the data separation function. (ii) We characterize the KO solutions with a one dimensional ODE. (iii) We approximate data-dependent KO paths by approximating the data separation function and minimizing the KE. (iv) We prove that the data separation function converges to $1$ in the general case of arbitrary normalized dataset consisting of $n$ samples in $d$ dimension as $n/\sqrt{d}\rightarrow 0$. A consequence of this result is that the Conditional Optimal Transport (Cond-OT) path becomes kinetic optimal as $n/\sqrt{d}\rightarrow 0$. We further support this theory with empirical experiments on ImageNet.

        ----

        ## [1278] Sequential Changepoint Detection via Backward Confidence Sequences

        **Authors**: *Shubhanshu Shekhar, Aaditya Ramdas*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/shekhar23a.html](https://proceedings.mlr.press/v202/shekhar23a.html)

        **Abstract**:

        We present a simple reduction from sequential estimation to sequential changepoint detection (SCD). In short, suppose we are interested in detecting changepoints in some parameter or functional $\theta$ of the underlying distribution. We demonstrate that if we can construct a confidence sequence (CS) for $\theta$, then we can also successfully perform SCD for $\theta$. This is accomplished by checking if two CSs — one forwards and the other backwards — ever fail to intersect. Since the literature on CSs has been rapidly evolving recently, the reduction provided in this paper immediately solves several old and new change detection problems. Further, our “backward CS”, constructed by reversing time, is new and potentially of independent interest. We provide strong nonasymptotic guarantees on the frequency of false alarms and detection delay, and demonstrate numerical effectiveness on several problems.

        ----

        ## [1279] Cold Analysis of Rao-Blackwellized Straight-Through Gumbel-Softmax Gradient Estimator

        **Authors**: *Alexander Shekhovtsov*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/shekhovtsov23a.html](https://proceedings.mlr.press/v202/shekhovtsov23a.html)

        **Abstract**:

        Many problems in machine learning require an estimate of the gradient of an expectation in discrete random variables with respect to the sampling distribution. This work is motivated by the development of the Gumbel-Softmax family of estimators, which use a temperature-controlled relaxation of discrete variables. The state-of-the art in this family, the Gumbel-Rao estimator uses an extra internal sampling to reduce the variance, which may be costly. We analyze this estimator and show that it possesses a zero temperature limit with a surprisingly simple closed form. The limit estimator, called ZGR, has favorable bias and variance properties, it is easy to implement and computationally inexpensive. It decomposes as the average of the straight through (ST) estimator and DARN estimator — two basic but not very well performing on their own estimators. We demonstrate that the simple ST–ZGR family of estimators practically dominates in the bias-variance tradeoffs the whole GR family while also outperforming SOTA unbiased estimators.

        ----

        ## [1280] Towards Understanding and Improving GFlowNet Training

        **Authors**: *Max W. Shen, Emmanuel Bengio, Ehsan Hajiramezanali, Andreas Loukas, Kyunghyun Cho, Tommaso Biancalani*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/shen23a.html](https://proceedings.mlr.press/v202/shen23a.html)

        **Abstract**:

        Generative flow networks (GFlowNets) are a family of algorithms that learn a generative policy to sample discrete objects $x$ with non-negative reward $R(x)$. Learning objectives guarantee the GFlowNet samples $x$ from the target distribution $p^*(x) \propto R(x)$ when loss is globally minimized over all states or trajectories, but it is unclear how well they perform with practical limits on training resources. We introduce an efficient evaluation strategy to compare the learned sampling distribution to the target reward distribution. As flows can be underdetermined given training data, we clarify the importance of learned flows to generalization and matching $p^*(x)$ in practice. We investigate how to learn better flows, and propose (i) prioritized replay training of high-reward $x$, (ii) relative edge flow policy parametrization, and (iii) a novel guided trajectory balance objective, and show how it can solve a substructure credit assignment problem. We substantially improve sample efficiency on biochemical design tasks.

        ----

        ## [1281] On Balancing Bias and Variance in Unsupervised Multi-Source-Free Domain Adaptation

        **Authors**: *Maohao Shen, Yuheng Bu, Gregory W. Wornell*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/shen23b.html](https://proceedings.mlr.press/v202/shen23b.html)

        **Abstract**:

        Due to privacy, storage, and other constraints, there is a growing need for unsupervised domain adaptation techniques in machine learning that do not require access to the data used to train a collection of source models. Existing methods for multi-source-free domain adaptation (MSFDA) typically train a target model using pseudo-labeled data produced by the source models, which focus on improving the pseudo-labeling techniques or proposing new training objectives. Instead, we aim to analyze the fundamental limits of MSFDA. In particular, we develop an information-theoretic bound on the generalization error of the resulting target model, which illustrates an inherent bias-variance trade-off. We then provide insights on how to balance this trade-off from three perspectives, including domain aggregation, selective pseudo-labeling, and joint feature alignment, which leads to the design of novel algorithms. Experiments on multiple datasets validate our theoretical analysis and demonstrate the state-of-art performance of the proposed algorithm, especially on some of the most challenging datasets, including Office-Home and DomainNet.

        ----

        ## [1282] On Penalty-based Bilevel Gradient Descent Method

        **Authors**: *Han Shen, Tianyi Chen*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/shen23c.html](https://proceedings.mlr.press/v202/shen23c.html)

        **Abstract**:

        Bilevel optimization enjoys a wide range of applications in hyper-parameter optimization, meta-learning and reinforcement learning. However, bilevel problems are difficult to solve and recent progress on scalable bilevel algorithms mainly focuses on bilevel optimization problems where the lower-level objective is either strongly convex or unconstrained. In this work, we tackle the bilevel problem through the lens of the penalty method. We show that under certain conditions, the penalty reformulation recovers the solutions of the original bilevel problem. Further, we propose the penalty-based bilevel gradient descent algorithm and establish its finite-time convergence for the constrained bilevel problem without lower-level strong convexity. The experimental results showcase the efficiency of the proposed algorithm.

        ----

        ## [1283] Non-autoregressive Conditional Diffusion Models for Time Series Prediction

        **Authors**: *Lifeng Shen, James T. Kwok*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/shen23d.html](https://proceedings.mlr.press/v202/shen23d.html)

        **Abstract**:

        Recently, denoising diffusion models have led to significant breakthroughs in the generation of images, audio and text. However, it is still an open question on how to adapt their strong modeling ability to model time series. In this paper, we propose TimeDiff, a non-autoregressive diffusion model that achieves high-quality time series prediction with the introduction of two novel conditioning mechanisms: future mixup and autoregressive initialization. Similar to teacher forcing, future mixup allows parts of the ground-truth future predictions for conditioning, while autoregressive initialization helps better initialize the model with basic time series patterns such as short-term trends. Extensive experiments are performed on nine real-world datasets. Results show that TimeDiff consistently outperforms existing time series diffusion models, and also achieves the best overall performance across a variety of the existing strong baselines (including transformers and FiLM).

        ----

        ## [1284] Cross-Modal Fine-Tuning: Align then Refine

        **Authors**: *Junhong Shen, Liam Li, Lucio M. Dery, Corey Staten, Mikhail Khodak, Graham Neubig, Ameet Talwalkar*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/shen23e.html](https://proceedings.mlr.press/v202/shen23e.html)

        **Abstract**:

        Fine-tuning large-scale pretrained models has led to tremendous progress in well-studied modalities such as vision and NLP. However, similar gains have not been observed in many other modalities due to a lack of relevant pretrained models. In this work, we propose ORCA, a general cross-modal fine-tuning framework that extends the applicability of a single large-scale pretrained model to diverse modalities. ORCA adapts to a target task via an align-then-refine workflow: given the target input, ORCA first learns an embedding network that aligns the embedded feature distribution with the pretraining modality. The pretrained model is then fine-tuned on the embedded data to exploit the knowledge shared across modalities. Through extensive experiments, we show that ORCA obtains state-of-the-art results on 3 benchmarks containing over 60 datasets from 12 modalities, outperforming a wide range of hand-designed, AutoML, general-purpose, and task-specific cross-modal methods. We highlight the importance of data alignment via a series of ablation studies and exemplify ORCA’s utility in data-limited regimes.

        ----

        ## [1285] Auxiliary Modality Learning with Generalized Curriculum Distillation

        **Authors**: *Yu Shen, Xijun Wang, Peng Gao, Ming C. Lin*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/shen23f.html](https://proceedings.mlr.press/v202/shen23f.html)

        **Abstract**:

        Driven by the need from real-world applications, Auxiliary Modality Learning (AML) offers the possibility to utilize more information from auxiliary data in training, while only requiring data from one or fewer modalities in test, to save the overall computational cost and reduce the amount of input data for inferencing. In this work, we formally define “Auxiliary Modality Learning” (AML), systematically classify types of auxiliary modality (in visual computing) and architectures for AML, and analyze their performance. We also analyze the conditions under which AML works well from the optimization and data distribution perspectives. To guide various choices to achieve optimal performance using AML, we propose a novel method to assist in choosing the best auxiliary modality and estimating an upper bound performance before executing AML. In addition, we propose a new AML method using generalized curriculum distillation to enable more effective curriculum learning. Our method achieves the best performance compared to other SOTA methods.

        ----

        ## [1286] TGRL: An Algorithm for Teacher Guided Reinforcement Learning

        **Authors**: *Idan Shenfeld, Zhang-Wei Hong, Aviv Tamar, Pulkit Agrawal*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/shenfeld23a.html](https://proceedings.mlr.press/v202/shenfeld23a.html)

        **Abstract**:

        We consider solving sequential decision-making problems in the scenario where the agent has access to two supervision sources: $\textit{reward signal}$ and a $\textit{teacher}$ that can be queried to obtain a $\textit{good}$ action for any state encountered by the agent. Learning solely from rewards, or reinforcement learning, is data inefficient and may not learn high-reward policies in challenging scenarios involving sparse rewards or partial observability. On the other hand, learning from a teacher may sometimes be infeasible. For instance, the actions provided by a teacher with privileged information may be unlearnable by an agent with limited information (i.e., partial observability). In other scenarios, the teacher might be sub-optimal, and imitating their actions can limit the agent’s performance. To overcome these challenges, prior work proposed to jointly optimize imitation and reinforcement learning objectives but relied on heuristics and problem-specific hyper-parameter tuning to balance the two objectives. We introduce Teacher Guided Reinforcement Learning (TGRL), a principled approach to dynamically balance following the teacher’s guidance and leveraging RL. TGRL outperforms strong baselines across diverse domains without hyperparameter tuning.

        ----

        ## [1287] FlexGen: High-Throughput Generative Inference of Large Language Models with a Single GPU

        **Authors**: *Ying Sheng, Lianmin Zheng, Binhang Yuan, Zhuohan Li, Max Ryabinin, Beidi Chen, Percy Liang, Christopher Ré, Ion Stoica, Ce Zhang*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/sheng23a.html](https://proceedings.mlr.press/v202/sheng23a.html)

        **Abstract**:

        The high computational and memory requirements of large language model (LLM) inference make it feasible only with multiple high-end accelerators. Motivated by the emerging demand for latency-insensitive tasks with batched processing, this paper initiates the study of high-throughput LLM inference using limited resources, such as a single commodity GPU. We present FlexGen, a high-throughput generation engine for running LLMs with limited GPU memory. FlexGen can be flexibly configured under various hardware resource constraints by aggregating memory and computation from the GPU, CPU, and disk. By solving a linear programming problem, it searches for efficient patterns to store and access tensors. FlexGen further compresses the weights and the attention cache to 4 bits with negligible accuracy loss. These techniques enable FlexGen to have a larger space of batch size choices and thus significantly increase maximum throughput. As a result, when running OPT-175B on a single 16GB GPU, FlexGen achieves significantly higher throughput compared to state-of-the-art offloading systems, reaching a generation throughput of 1 token/s for the first time with an effective batch size of 144. On the HELM benchmark, FlexGen can benchmark a 30B model with a 16GB GPU on 7 representative sub-scenarios in 21 hours. The code is available at https://github.com/FMInference/FlexGen.

        ----

        ## [1288] Improved Regret for Efficient Online Reinforcement Learning with Linear Function Approximation

        **Authors**: *Uri Sherman, Tomer Koren, Yishay Mansour*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/sherman23a.html](https://proceedings.mlr.press/v202/sherman23a.html)

        **Abstract**:

        We study reinforcement learning with linear function approximation and adversarially changing cost functions, a setup that has mostly been considered under simplifying assumptions such as full information feedback or exploratory conditions. We present a computationally efficient policy optimization algorithm for the challenging general setting of unknown dynamics and bandit feedback, featuring a combination of mirror-descent and least squares policy evaluation in an auxiliary MDP used to compute exploration bonuses. Our algorithm obtains an $\widetilde O(K^{6/7})$ regret bound, improving significantly over previous state-of-the-art of $\widetilde O (K^{14/15})$ in this setting. In addition, we present a version of the same algorithm under the assumption a simulator of the environment is available to the learner (but otherwise no exploratory assumptions are made), and prove it obtains state-of-the-art regret of $\widetilde O (K^{2/3})$.

        ----

        ## [1289] Fundamental Limits of Two-layer Autoencoders, and Achieving Them with Gradient Methods

        **Authors**: *Aleksandr Shevchenko, Kevin Kögler, Hamed Hassani, Marco Mondelli*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/shevchenko23a.html](https://proceedings.mlr.press/v202/shevchenko23a.html)

        **Abstract**:

        Autoencoders are a popular model in many branches of machine learning and lossy data compression. However, their fundamental limits, the performance of gradient methods and the features learnt during optimization remain poorly understood, even in the two-layer setting. In fact, earlier work has considered either linear autoencoders or specific training regimes (leading to vanishing or diverging compression rates). Our paper addresses this gap by focusing on non-linear two-layer autoencoders trained in the challenging proportional regime in which the input dimension scales linearly with the size of the representation. Our results characterize the minimizers of the population risk, and show that such minimizers are achieved by gradient methods; their structure is also unveiled, thus leading to a concise description of the features obtained via training. For the special case of a sign activation function, our analysis establishes the fundamental limits for the lossy compression of Gaussian sources via (shallow) autoencoders. Finally, while the results are proved for Gaussian data, numerical simulations on standard datasets display the universality of the theoretical predictions.

        ----

        ## [1290] Large Language Models Can Be Easily Distracted by Irrelevant Context

        **Authors**: *Freda Shi, Xinyun Chen, Kanishka Misra, Nathan Scales, David Dohan, Ed H. Chi, Nathanael Schärli, Denny Zhou*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/shi23a.html](https://proceedings.mlr.press/v202/shi23a.html)

        **Abstract**:

        Large language models have achieved impressive performance on various natural language processing tasks. However, so far they have been evaluated primarily on benchmarks where all information in the input context is relevant for solving the task. In this work, we investigate the distractibility of large language models, i.e., how the model prediction can be distracted by irrelevant context. In particular, we introduce Grade-School Math with Irrelevant Context (GSM-IC), an arithmetic reasoning dataset with irrelevant information in the problem description. We use this benchmark to measure the distractibility of different prompting techniques for large language models, and find that the model is easily distracted by irrelevant information. We also identify several approaches for mitigating this deficiency, such as decoding with self-consistency and adding to the prompt an instruction that tells the language model to ignore the irrelevant information.

        ----

        ## [1291] Everyone's Preference Changes Differently: A Weighted Multi-Interest Model For Retrieval

        **Authors**: *Hui Shi, Yupeng Gu, Yitong Zhou, Bo Zhao, Sicun Gao, Jishen Zhao*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/shi23b.html](https://proceedings.mlr.press/v202/shi23b.html)

        **Abstract**:

        User embeddings (vectorized representations of a user) are essential in recommendation systems. Numerous approaches have been proposed to construct a representation for the user in order to find similar items for retrieval tasks, and they have been proven effective in industrial recommendation systems. Recently people have discovered the power of using multiple embeddings to represent a user, with the hope that each embedding represents the user’s interest in a certain topic. With multi-interest representation, it’s important to model the user’s preference over the different topics and how the preference changes with time. However, existing approaches either fail to estimate the user’s affinity to each interest or unreasonably assume every interest of every user fades at an equal rate with time, thus hurting the performance of candidate retrieval. In this paper, we propose the Multi-Interest Preference (MIP) model, an approach that not only produces multi-interest for users by using the user’s sequential engagement more effectively but also automatically learns a set of weights to represent the preference over each embedding so that the candidates can be retrieved from each interest proportionally. Extensive experiments have been done on various industrial-scale datasets to demonstrate the effectiveness of our approach.

        ----

        ## [1292] A Near-Optimal Algorithm for Safe Reinforcement Learning Under Instantaneous Hard Constraints

        **Authors**: *Ming Shi, Yingbin Liang, Ness B. Shroff*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/shi23c.html](https://proceedings.mlr.press/v202/shi23c.html)

        **Abstract**:

        In many applications of Reinforcement Learning (RL), it is critically important that the algorithm performs safely, such that instantaneous hard constraints are satisfied at each step, and unsafe states and actions are avoided. However, existing algorithms for “safe” RL are often designed under constraints that either require expected cumulative costs to be bounded or assume all states are safe. Thus, such algorithms could violate instantaneous hard constraints and traverse unsafe states (and actions) in practice. Hence, in this paper, we develop the first near-optimal safe RL algorithm for episodic Markov Decision Processes with unsafe states and actions under instantaneous hard constraints and the linear mixture model. It achieves a regret $\tilde{O}(\frac{d H^3 \sqrt{d K}}{\Delta_c})$ that nearly matches the state-of-the-art regret in the setting with only unsafe actions and that in the unconstrained setting, and is safe at each step, where $d$ is the feature-mapping dimension, $K$ is the number of episodes, $H$ is the episode length, and $\Delta_c$ is a safety-related parameter. We also provide a lower bound $\tilde{\Omega}(\max\{d H \sqrt{K}, \frac{H}{\Delta_c^2}\})$, which indicates that the dependency on $\Delta_c$ is necessary. Further, both our algorithm design and regret analysis involve several novel ideas, which may be of independent interest.

        ----

        ## [1293] Improving the Model Consistency of Decentralized Federated Learning

        **Authors**: *Yifan Shi, Li Shen, Kang Wei, Yan Sun, Bo Yuan, Xueqian Wang, Dacheng Tao*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/shi23d.html](https://proceedings.mlr.press/v202/shi23d.html)

        **Abstract**:

        To mitigate the privacy leakages and communication burdens of Federated Learning (FL), decentralized FL (DFL) discards the central server and each client only communicates with its neighbors in a decentralized communication network. However, existing DFL suffers from high inconsistency among local clients, which results in severe distribution shift and inferior performance compared with centralized FL (CFL), especially on heterogeneous data or sparse communication topologies. To alleviate this issue, we propose two DFL algorithms named DFedSAM and DFedSAM-MGS to improve the performance of DFL. Specifically, DFedSAM leverages gradient perturbation to generate local flat models via Sharpness Aware Minimization (SAM), which searches for models with uniformly low loss values. DFedSAM-MGS further boosts DFedSAM by adopting Multiple Gossip Steps (MGS) for better model consistency, which accelerates the aggregation of local flat models and better balances communication complexity and generalization. Theoretically, we present improved convergence rates $\small \mathcal{O}\big(\frac{1}{\sqrt{KT}}+\frac{1}{T}+\frac{1}{K^{1/2}T^{3/2}(1-\lambda)^2}\big)$ and $\small \mathcal{O}\big(\frac{1}{\sqrt{KT}}+\frac{1}{T}+\frac{\lambda^Q+1}{K^{1/2}T^{3/2}(1-\lambda^Q)^2}\big)$ in non-convex setting for DFedSAM and DFedSAM-MGS, respectively, where $1-\lambda$ is the spectral gap of gossip matrix and $Q$ is the number of MGS. Empirically, our methods can achieve competitive performance compared with CFL methods and outperform existing DFL methods.

        ----

        ## [1294] UPop: Unified and Progressive Pruning for Compressing Vision-Language Transformers

        **Authors**: *Dachuan Shi, Chaofan Tao, Ying Jin, Zhendong Yang, Chun Yuan, Jiaqi Wang*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/shi23e.html](https://proceedings.mlr.press/v202/shi23e.html)

        **Abstract**:

        Real-world data contains a vast amount of multimodal information, among which vision and language are the two most representative modalities. Moreover, increasingly heavier models, e.g., Transformers, have attracted the attention of researchers to model compression. However, how to compress multimodal models, especially vison-language Transformers, is still under-explored. This paper proposes the Unified and Progressive Pruning (UPop) as a universal vison-language Transformer compression framework, which incorporates 1) unifiedly searching multimodal subnets in a continuous optimization space from the original model, which enables automatic assignment of pruning ratios among compressible modalities and structures; 2) progressively searching and retraining the subnet, which maintains convergence between the search and retrain to attain higher compression ratios. Experiments on various tasks, datasets, and model architectures demonstrate the effectiveness and versatility of the proposed UPop framework. The code is available at https://github.com/sdc17/UPop.

        ----

        ## [1295] Sequence Modeling with Multiresolution Convolutional Memory

        **Authors**: *Jiaxin Shi, Ke Alexander Wang, Emily B. Fox*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/shi23f.html](https://proceedings.mlr.press/v202/shi23f.html)

        **Abstract**:

        Efficiently capturing the long-range patterns in sequential data sources salient to a given task—such as classification and generative modeling—poses a fundamental challenge. Popular approaches in the space tradeoff between the memory burden of brute-force enumeration and comparison, as in transformers, the computational burden of complicated sequential dependencies, as in recurrent neural networks, or the parameter burden of convolutional networks with many or large filters. We instead take inspiration from wavelet-based multiresolution analysis to define a new building block for sequence modeling, which we call a MultiresLayer. The key component of our model is the multiresolution convolution, capturing multiscale trends in the input sequence. Our MultiresConv can be implemented with shared filters across a dilated causal convolution tree. Thus it garners the computational advantages of convolutional networks and the principled theoretical motivation of wavelet decompositions. Our MultiresLayer is straightforward to implement, requires significantly fewer parameters, and maintains at most a $O(N \log N)$ memory footprint for a length $N$ sequence. Yet, by stacking such layers, our model yields state-of-the-art performance on a number of sequence classification and autoregressive density estimation tasks using CIFAR-10, ListOps, and PTB-XL datasets.

        ----

        ## [1296] Statistical Inference on Multi-armed Bandits with Delayed Feedback

        **Authors**: *Lei Shi, Jingshen Wang, Tianhao Wu*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/shi23g.html](https://proceedings.mlr.press/v202/shi23g.html)

        **Abstract**:

        Multi armed bandit (MAB) algorithms have been increasingly used to complement or integrate with A/B tests and randomized clinical trials in e-commerce, healthcare, and policymaking. Recent developments incorporate possible delayed feedback. While existing MAB literature often focuses on maximizing the expected cumulative reward outcomes (or, equivalently, regret minimization), few efforts have been devoted to establish valid statistical inference approaches to quantify the uncertainty of learned policies. We attempt to fill this gap by providing a unified statistical inference framework for policy evaluation where a target policy is allowed to differ from the data collecting policy, and our framework allows delay to be associated with the treatment arms. We present an adaptively weighted estimator that on one hand incorporates the arm-dependent delaying mechanism to achieve consistency, and on the other hand mitigates the variance inflation across stages due to vanishing sampling probability. In particular, our estimator does not critically depend on the ability to estimate the unknown delay mechanism. Under appropriate conditions, we prove that our estimator converges to a normal distribution as the number of time points goes to infinity, which provides guarantees for large-sample statistical inference. We illustrate the finite-sample performance of our approach through Monte Carlo experiments.

        ----

        ## [1297] Provably Efficient Offline Reinforcement Learning with Perturbed Data Sources

        **Authors**: *Chengshuai Shi, Wei Xiong, Cong Shen, Jing Yang*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/shi23h.html](https://proceedings.mlr.press/v202/shi23h.html)

        **Abstract**:

        Existing theoretical studies on offline reinforcement learning (RL) mostly consider a dataset sampled directly from the target task. In practice, however, data often come from several heterogeneous but related sources. Motivated by this gap, this work aims at rigorously understanding offline RL with multiple datasets that are collected from randomly perturbed versions of the target task instead of from itself. An information-theoretic lower bound is derived, which reveals a necessary requirement on the number of involved sources in addition to that on the number of data samples. Then, a novel HetPEVI algorithm is proposed, which simultaneously considers the sample uncertainties from a finite number of data samples per data source and the source uncertainties due to a finite number of available data sources. Theoretical analyses demonstrate that HetPEVI can solve the target task as long as the data sources collectively provide a good data coverage. Moreover, HetPEVI is demonstrated to be optimal up to a polynomial factor of the horizon length. Finally, the study is extended to offline Markov games and offline robust RL, which demonstrates the generality of the proposed designs and theoretical analyses.

        ----

        ## [1298] On the Complexity of Bayesian Generalization

        **Authors**: *Yu-Zhe Shi, Manjie Xu, John E. Hopcroft, Kun He, Joshua B. Tenenbaum, Song-Chun Zhu, Ying Nian Wu, Wenjuan Han, Yixin Zhu*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/shi23i.html](https://proceedings.mlr.press/v202/shi23i.html)

        **Abstract**:

        We examine concept generalization at a large scale in the natural visual spectrum. Established computational modes (i.e., rule-based or similarity-based) are primarily studied isolated, focusing on confined and abstract problem spaces. In this work, we study these two modes when the problem space scales up and when the complexity of concepts becomes diverse. At the representational level, we investigate how the complexity varies when a visual concept is mapped to the representation space. Prior literature has shown that two types of complexities (Griffiths & Tenenbaum, 2003) build an inverted-U relation (Donderi, 2006; Sun & Firestone, 2021). Leveraging Representativeness of Attribute (RoA), we computationally confirm: Models use attributes with high RoA to describe visual concepts, and the description length falls in an inverted-U relation with the increment in visual complexity. At the computational level, we examine how the complexity of representation affects the shift between the rule- and similarity-based generalization. We hypothesize that category-conditioned visual modeling estimates the co-occurrence frequency between visual and categorical attributes, thus potentially serving as the prior for the natural visual world. Experimental results show that representations with relatively high subjective complexity outperform those with relatively low subjective complexity in rule-based generalization, while the trend is the opposite in similarity-based generalization.

        ----

        ## [1299] Understanding and Generalizing Contrastive Learning from the Inverse Optimal Transport Perspective

        **Authors**: *Liangliang Shi, Gu Zhang, Haoyu Zhen, Jintao Fan, Junchi Yan*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/shi23j.html](https://proceedings.mlr.press/v202/shi23j.html)

        **Abstract**:

        Previous research on contrastive learning (CL) has primarily focused on pairwise views to learn representations by attracting positive samples and repelling negative ones. In this work, we aim to understand and generalize CL from a point set matching perspective, instead of the comparison between two points. Specifically, we formulate CL as a form of inverse optimal transport (IOT), which involves a bilevel optimization procedure for learning where the outter minimization aims to learn the representations and the inner is to learn the coupling (i.e. the probability of matching matrix) between the point sets. Specifically, by adjusting the relaxation degree of constraints in the inner minimization, we obtain three contrastive losses and show that the dominant contrastive loss in literature InfoNCE falls into one of these losses. This reveals a new and more general algorithmic framework for CL. Additionally, the soft matching scheme in IOT induces a uniformity penalty to enhance representation learning which is akin to the CL’s uniformity. Results on vision benchmarks show the effectiveness of our derived loss family and the new uniformity term.

        ----

        ## [1300] Long Horizon Temperature Scaling

        **Authors**: *Andy Shih, Dorsa Sadigh, Stefano Ermon*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/shih23a.html](https://proceedings.mlr.press/v202/shih23a.html)

        **Abstract**:

        Temperature scaling is a popular technique for tuning the sharpness of a model distribution. It is used extensively for sampling likely generations and calibrating model uncertainty, and even features as a controllable parameter to many large language models in deployment. However, autoregressive models rely on myopic temperature scaling that greedily optimizes the next token. To address this, we propose Long Horizon Temperature Scaling (LHTS), a novel approach for sampling from temperature-scaled joint distributions. LHTS is compatible with all likelihood-based models, and optimizes for the long-horizon likelihood of samples. We derive a temperature-dependent LHTS objective, and show that fine-tuning a model on a range of temperatures produces a single model capable of generation with a controllable long-horizon temperature parameter. We experiment with LHTS on image diffusion models and character/language autoregressive models, demonstrating its advantages over myopic temperature scaling in likelihood and sample quality, and showing improvements in accuracy of a multiple choice analogy by $10$%.

        ----

        ## [1301] Gradient Descent in Neural Networks as Sequential Learning in Reproducing Kernel Banach Space

        **Authors**: *Alistair Shilton, Sunil Gupta, Santu Rana, Svetha Venkatesh*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/shilton23a.html](https://proceedings.mlr.press/v202/shilton23a.html)

        **Abstract**:

        The study of Neural Tangent Kernels (NTKs) has provided much needed insight into convergence and generalization properties of neural networks in the over-parametrized (wide) limit by approximating the network using a first-order Taylor expansion with respect to its weights in the neighborhood of their initialization values. This allows neural network training to be analyzed from the perspective of reproducing kernel Hilbert spaces (RKHS), which is informative in the over-parametrized regime, but a poor approximation for narrower networks as the weights change more during training. Our goal is to extend beyond the limits of NTK toward a more general theory. We construct an exact power-series representation of the neural network in a finite neighborhood of the initial weights as an inner product of two feature maps, respectively from data and weight-step space, to feature space, allowing neural network training to be analyzed from the perspective of reproducing kernel Banach space (RKBS). We prove that, regardless of width, the training sequence produced by gradient descent can be exactly replicated by regularized sequential learning in RKBS. Using this, we present novel bound on uniform convergence where the iterations count and learning rate play a central role, giving new theoretical insight into neural network training.

        ----

        ## [1302] SNeRL: Semantic-aware Neural Radiance Fields for Reinforcement Learning

        **Authors**: *Dongseok Shim, Seungjae Lee, H. Jin Kim*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/shim23a.html](https://proceedings.mlr.press/v202/shim23a.html)

        **Abstract**:

        As previous representations for reinforcement learning cannot effectively incorporate a human-intuitive understanding of the 3D environment, they usually suffer from sub-optimal performances. In this paper, we present Semantic-aware Neural Radiance Fields for Reinforcement Learning (SNeRL), which jointly optimizes semantic-aware neural radiance fields (NeRF) with a convolutional encoder to learn 3D-aware neural implicit representation from multi-view images. We introduce 3D semantic and distilled feature fields in parallel to the RGB radiance fields in NeRF to learn semantic and object-centric representation for reinforcement learning. SNeRL outperforms not only previous pixel-based representations but also recent 3D-aware representations both in model-free and model-based reinforcement learning.

        ----

        ## [1303] A Closer Look at the Intervention Procedure of Concept Bottleneck Models

        **Authors**: *Sungbin Shin, Yohan Jo, Sungsoo Ahn, Namhoon Lee*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/shin23a.html](https://proceedings.mlr.press/v202/shin23a.html)

        **Abstract**:

        Concept bottleneck models (CBMs) are a class of interpretable neural network models that predict the target response of a given input based on its high-level concepts. Unlike the standard end-to-end models, CBMs enable domain experts to intervene on the predicted concepts and rectify any mistakes at test time, so that more accurate task predictions can be made at the end. While such intervenability provides a powerful avenue of control, many aspects of the intervention procedure remain rather unexplored. In this work, we develop various ways of selecting intervening concepts to improve the intervention effectiveness and conduct an array of in-depth analyses as to how they evolve under different circumstances. Specifically, we find that an informed intervention strategy can reduce the task error more than ten times compared to the current baseline under the same amount of intervention counts in realistic settings, and yet, this can vary quite significantly when taking into account different intervention granularity. We verify our findings through comprehensive evaluations, not only on the standard real datasets, but also on synthetic datasets that we generate based on a set of different causal graphs. We further discover some major pitfalls of the current practices which, without a proper addressing, raise concerns on reliability and fairness of the intervention procedure.

        ----

        ## [1304] MetricGAN-OKD: Multi-Metric Optimization of MetricGAN via Online Knowledge Distillation for Speech Enhancement

        **Authors**: *WooSeok Shin, Byung Hoon Lee, Jin Sob Kim, Hyun Joon Park, Sung Won Han*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/shin23b.html](https://proceedings.mlr.press/v202/shin23b.html)

        **Abstract**:

        In speech enhancement, MetricGAN-based approaches reduce the discrepancy between the $L_p$ loss and evaluation metrics by utilizing a non-differentiable evaluation metric as the objective function. However, optimizing multiple metrics simultaneously remains challenging owing to the problem of confusing gradient directions. In this paper, we propose an effective multi-metric optimization method in MetricGAN via online knowledge distillation—MetricGAN-OKD. MetricGAN-OKD, which consists of multiple generators and target metrics, related by a one-to-one correspondence, enables generators to learn with respect to a single metric reliably while improving performance with respect to other metrics by mimicking other generators. Experimental results on speech enhancement and listening enhancement tasks reveal that the proposed method significantly improves performance in terms of multiple metrics compared to existing multi-metric optimization methods. Further, the good performance of MetricGAN-OKD is explained in terms of network generalizability and correlation between metrics.

        ----

        ## [1305] Improved Learning-Augmented Algorithms for the Multi-Option Ski Rental Problem via Best-Possible Competitive Analysis

        **Authors**: *Yongho Shin, Changyeol Lee, Gukryeol Lee, Hyung-Chan An*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/shin23c.html](https://proceedings.mlr.press/v202/shin23c.html)

        **Abstract**:

        In this paper, we present improved learning-augmented algorithms for the multi-option ski rental problem. Learning-augmented algorithms take ML predictions as an added part of the input and incorporates these predictions in solving the given problem. Due to their unique strength that combines the power of ML predictions with rigorous performance guarantees, they have been extensively studied in the context of online optimization problems. Even though ski rental problems are one of the canonical problems in the field of online optimization, only deterministic algorithms were previously known for multi-option ski rental, with or without learning augmentation. We present the first randomized learning-augmented algorithm for this problem, surpassing previous performance guarantees given by deterministic algorithms. Our learning-augmented algorithm is based on a new, provably best-possible randomized competitive algorithm for the problem. Our results are further complemented by lower bounds for deterministic and randomized algorithms, and computational experiments evaluating our algorithms’ performance improvements.

        ----

        ## [1306] One-shot Imitation in a Non-Stationary Environment via Multi-Modal Skill

        **Authors**: *Sangwoo Shin, Daehee Lee, Minjong Yoo, Woo Kyung Kim, Honguk Woo*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/shin23d.html](https://proceedings.mlr.press/v202/shin23d.html)

        **Abstract**:

        One-shot imitation is to learn a new task from a single demonstration, yet it is a challenging problem to adopt it for complex tasks with the high domain diversity inherent in a non-stationary environment. To tackle the problem, we explore the compositionality of complex tasks, and present a novel skill-based imitation learning framework enabling one-shot imitation and zero-shot adaptation; from a single demonstration for a complex unseen task, a semantic skill sequence is inferred and then each skill in the sequence is converted into an action sequence optimized for environmental hidden dynamics that can vary over time. Specifically, we leverage a vision-language model to learn a semantic skill set from offline video datasets, where each skill is represented on the vision-language embedding space, and adapt meta-learning with dynamics inference to enable zero-shot skill adaptation. We evaluate our framework with various one-shot imitation scenarios for extended multi-stage Meta-world tasks, showing its superiority in learning complex tasks, generalizing to dynamics changes, and extending to different demonstration conditions and modalities, compared to other baselines.

        ----

        ## [1307] Context Consistency Regularization for Label Sparsity in Time Series

        **Authors**: *Yooju Shin, Susik Yoon, Hwanjun Song, Dongmin Park, Byunghyun Kim, Jae-Gil Lee, Byung Suk Lee*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/shin23e.html](https://proceedings.mlr.press/v202/shin23e.html)

        **Abstract**:

        Labels are typically sparse in real-world time series due to the high annotation cost. Recently, consistency regularization techniques have been used to generate artificial labels from unlabeled augmented instances. To fully exploit the sequential characteristic of time series in consistency regularization, we propose a novel method of data augmentation called context-attached augmentation, which adds preceding and succeeding instances to a target instance to form its augmented instance. Unlike the existing augmentation techniques that modify a target instance by directly perturbing its attributes, the context-attached augmentation generates instances augmented with varying contexts while maintaining the target instance. Based on our augmentation method, we propose a context consistency regularization framework, which first adds different contexts to a target instance sampled from a given time series and then shares unitary reliability-based cross-window labels across the augmented instances to maintain consistency. We demonstrate that the proposed framework outperforms the existing state-of-the-art consistency regularization frameworks through comprehensive experiments on real-world time-series datasets.

        ----

        ## [1308] Generative Causal Representation Learning for Out-of-Distribution Motion Forecasting

        **Authors**: *Shayan Shirahmad Gale Bagi, Zahra Gharaee, Oliver Schulte, Mark Crowley*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/shirahmad-gale-bagi23a.html](https://proceedings.mlr.press/v202/shirahmad-gale-bagi23a.html)

        **Abstract**:

        Conventional supervised learning methods typically assume i.i.d samples and are found to be sensitive to out-of-distribution (OOD) data. We propose Generative Causal Representation Learning (GCRL) which leverages causality to facilitate knowledge transfer under distribution shifts. While we evaluate the effectiveness of our proposed method in human trajectory prediction models, GCRL can be applied to other domains as well. First, we propose a novel causal model that explains the generative factors in motion forecasting datasets using features that are common across all environments and with features that are specific to each environment. Selection variables are used to determine which parts of the model can be directly transferred to a new environment without fine-tuning. Second, we propose an end-to-end variational learning paradigm to learn the causal mechanisms that generate observations from features. GCRL is supported by strong theoretical results that imply identifiability of the causal model under certain assumptions. Experimental results on synthetic and real-world motion forecasting datasets show the robustness and effectiveness of our proposed method for knowledge transfer under zero-shot and low-shot settings by substantially outperforming the prior motion forecasting models on out-of-distribution prediction.

        ----

        ## [1309] Exphormer: Sparse Transformers for Graphs

        **Authors**: *Hamed Shirzad, Ameya Velingker, Balaji Venkatachalam, Danica J. Sutherland, Ali Kemal Sinop*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/shirzad23a.html](https://proceedings.mlr.press/v202/shirzad23a.html)

        **Abstract**:

        Graph transformers have emerged as a promising architecture for a variety of graph learning and representation tasks. Despite their successes, though, it remains challenging to scale graph transformers to large graphs while maintaining accuracy competitive with message-passing networks. In this paper, we introduce Exphormer, a framework for building powerful and scalable graph transformers. Exphormer consists of a sparse attention mechanism based on two mechanisms: virtual global nodes and expander graphs, whose mathematical characteristics, such as spectral expansion, pseduorandomness, and sparsity, yield graph transformers with complexity only linear in the size of the graph, while allowing us to prove desirable theoretical properties of the resulting transformer models. We show that incorporating Exphormer into the recently-proposed GraphGPS framework produces models with competitive empirical results on a wide variety of graph datasets, including state-of-the-art results on three datasets. We also show that Exphormer can scale to datasets on larger graphs than shown in previous graph transformer architectures.

        ----

        ## [1310] Synthetic data for model selection

        **Authors**: *Alon Shoshan, Nadav Bhonker, Igor Kviatkovsky, Matan Fintz, Gérard G. Medioni*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/shoshan23a.html](https://proceedings.mlr.press/v202/shoshan23a.html)

        **Abstract**:

        Recent breakthroughs in synthetic data generation approaches made it possible to produce highly photorealistic images which are hardly distinguishable from real ones. Furthermore, synthetic generation pipelines have the potential to generate an unlimited number of images. The combination of high photorealism and scale turn synthetic data into a promising candidate for improving various machine learning (ML) pipelines. Thus far, a large body of research in this field has focused on using synthetic images for training, by augmenting and enlarging training data. In contrast to using synthetic data for training, in this work we explore whether synthetic data can be beneficial for model selection. Considering the task of image classification, we demonstrate that when data is scarce, synthetic data can be used to replace the held out validation set, thus allowing to train on a larger dataset. We also introduce a novel method to calibrate the synthetic error estimation to fit that of the real domain. We show that such calibration significantly improves the usefulness of synthetic data for model selection.

        ----

        ## [1311] Probabilistic Attention-to-Influence Neural Models for Event Sequences

        **Authors**: *Xiao Shou, Debarun Bhattacharjya, Tian Gao, Dharmashankar Subramanian, Oktie Hassanzadeh, Kristin P. Bennett*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/shou23a.html](https://proceedings.mlr.press/v202/shou23a.html)

        **Abstract**:

        Discovering knowledge about which types of events influence others, using datasets of event sequences without time stamps, has several practical applications. While neural sequence models are able to capture complex and potentially long-range historical dependencies, they often lack the interpretability of simpler models for event sequence dynamics. We provide a novel neural framework in such a setting - a probabilistic attention-to-influence neural model - which not only captures complex instance-wise interactions between events but also learns influencers for each event type of interest. Given event sequence data and a prior distribution on type-wise influence, we efficiently learn an approximate posterior for type-wise influence by an attention-to-influence transformation using variational inference. Our method subsequently models the conditional likelihood of sequences by sampling the above posterior to focus attention on influencing event types. We motivate our general framework and show improved performance in experiments compared to existing baselines on synthetic data as well as real-world benchmarks, for tasks involving prediction and influencing set identification.

        ----

        ## [1312] Causal Bounds in Quasi-Markovian Graphs

        **Authors**: *Madhumitha Shridharan, Garud Iyengar*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/shridharan23a.html](https://proceedings.mlr.press/v202/shridharan23a.html)

        **Abstract**:

        We consider the problem of computing bounds for causal queries on quasi-Markovian graphs with unobserved confounders and discrete valued observed variables, where identifiability does not hold. Existing non-parametric approaches for computing such bounds use multilinear programming (MP) formulations that are often intractable for existing solvers when the degree of the polynomial objective is greater than two. Hence, one often has to resort to either fast approximate heuristics which are not guaranteed to contain the true query value, or more accurate but computationally intensive procedures. We show how to construct an equivalent MP with a polynomial objective of lower degree. In particular, the degree of the objective in the new MP is equal to only the number of C-components that are intervened upon, instead of the total number of C-components. As a result, we can compute exact bounds for significantly larger causal inference problems as compared to what is possible using existing techniques. We also propose a very efficient Frank-Wolfe heuristic that produces very high quality bounds, and scales to large multilinear problems of higher degree.

        ----

        ## [1313] Repository-Level Prompt Generation for Large Language Models of Code

        **Authors**: *Disha Shrivastava, Hugo Larochelle, Daniel Tarlow*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/shrivastava23a.html](https://proceedings.mlr.press/v202/shrivastava23a.html)

        **Abstract**:

        With the success of large language models (LLMs) of code and their use as code assistants (e.g. Codex used in GitHub Copilot), techniques for introducing domain-specific knowledge in the prompt design process become important. In this work, we propose a framework called Repo-Level Prompt Generator that learns to generate example-specific prompts using prompt proposals. The prompt proposals take context from the entire repository, thereby incorporating both the structure of the repository and the context from other relevant files (e.g. imports, parent class files). Our technique doesn’t require any access to the weights of the LLM, making it applicable in cases where we only have black-box access to the LLM. We conduct experiments on the task of single-line code auto-completion using code repositories taken from Google Code archives. We demonstrate that an oracle constructed from our prompt proposals gives a relative improvement of 36% over Codex, showing the quality of these proposals. Further, we show that when we train a model to predict a prompt proposal, we can achieve significant performance gains over Codex and other baselines. We release our code, data, and trained checkpoints at: https://github.com/shrivastavadisha/repo_level_prompt_generation.

        ----

        ## [1314] CLIPood: Generalizing CLIP to Out-of-Distributions

        **Authors**: *Yang Shu, Xingzhuo Guo, Jialong Wu, Ximei Wang, Jianmin Wang, Mingsheng Long*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/shu23a.html](https://proceedings.mlr.press/v202/shu23a.html)

        **Abstract**:

        Out-of-distribution (OOD) generalization, where the model needs to handle distribution shifts from training, is a major challenge of machine learning. Contrastive language-image pre-training (CLIP) models have shown impressive zero-shot ability, but the further adaptation of CLIP on downstream tasks undesirably degrades OOD performances. This paper aims at generalizing CLIP to out-of-distribution test data on downstream tasks. We propose CLIPood, a fine-tuning method that can adapt CLIP models to OOD situations where both domain shifts and open classes may occur on the unseen test data. To exploit the semantic relations between classes from the text modality, CLIPood introduces a new training objective, margin metric softmax (MMS), with class adaptive margins for fine-tuning. To incorporate both pre-trained zero-shot model and fine-tuned task-adaptive model, CLIPood leverages a new optimization strategy, Beta moving average (BMA), to maintain a temporal ensemble weighted by Beta distribution. Experiments on diverse datasets with different OOD scenarios show that CLIPood consistently outperforms existing generalization techniques.

        ----

        ## [1315] Semi-Autoregressive Energy Flows: Exploring Likelihood-Free Training of Normalizing Flows

        **Authors**: *Phillip Si, Zeyi Chen, Subham Sekhar Sahoo, Yair Schiff, Volodymyr Kuleshov*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/si23a.html](https://proceedings.mlr.press/v202/si23a.html)

        **Abstract**:

        Training normalizing flow generative models can be challenging due to the need to calculate computationally expensive determinants of Jacobians. This paper studies the likelihood-free training of flows and proposes the energy objective, an alternative sample-based loss based on proper scoring rules. The energy objective is determinant-free and supports flexible model architectures that are not easily compatible with maximum likelihood training, including semi-autoregressive energy flows, a novel model family that interpolates between fully autoregressive and non-autoregressive models. Energy flows feature competitive sample quality, posterior inference, and generation speed relative to likelihood-based flows; this performance is decorrelated from the quality of log-likelihood estimates, which are generally very poor. Our findings question the use of maximum likelihood as an objective or a metric, and contribute to a scientific study of its role in generative modeling. Code is available at https://github.com/ps789/SAEF.

        ----

        ## [1316] Unearthing InSights into Mars: Unsupervised Source Separation with Limited Data

        **Authors**: *Ali Siahkoohi, Rudy Morel, Maarten V. de Hoop, Erwan Allys, Grégory Sainton, Taichi Kawamura*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/siahkoohi23a.html](https://proceedings.mlr.press/v202/siahkoohi23a.html)

        **Abstract**:

        Source separation involves the ill-posed problem of retrieving a set of source signals that have been observed through a mixing operator. Solving this problem requires prior knowledge, which is commonly incorporated by imposing regularity conditions on the source signals, or implicitly learned through supervised or unsupervised methods from existing data. While data-driven methods have shown great promise in source separation, they often require large amounts of data, which rarely exists in planetary space missions. To address this challenge, we propose an unsupervised source separation scheme for domains with limited data access that involves solving an optimization problem in the wavelet scattering covariance representation space—an interpretable, low-dimensional representation of stationary processes. We present a real-data example in which we remove transient, thermally-induced microtilts—known as glitches—from data recorded by a seismometer during NASA’s InSight mission on Mars. Thanks to the wavelet scattering covariances’ ability to capture non-Gaussian properties of stochastic processes, we are able to separate glitches using only a few glitch-free data snippets.

        ----

        ## [1317] Quantitative Universal Approximation Bounds for Deep Belief Networks

        **Authors**: *Julian Sieber, Johann Gehringer*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/sieber23a.html](https://proceedings.mlr.press/v202/sieber23a.html)

        **Abstract**:

        We show that deep belief networks with binary hidden units can approximate any multivariate probability density under very mild integrability requirements on the parental density of the visible nodes. The approximation is measured in the $L^q$-norm for $q\in[1,\infty]$ ($q=\infty$ corresponding to the supremum norm) and in Kullback-Leibler divergence. Furthermore, we establish sharp quantitative bounds on the approximation error in terms of the number of hidden units.

        ----

        ## [1318] Pricing Experimental Design: Causal Effect, Expected Revenue and Tail Risk

        **Authors**: *David Simchi-Levi, Chonghuan Wang*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/simchi-levi23a.html](https://proceedings.mlr.press/v202/simchi-levi23a.html)

        **Abstract**:

        When launching a new product, historical sales data is often not available, leaving price as a crucial experimental instrument for sellers to gauge market response. When designing pricing experiments, there are three fundamental objectives: estimating the causal effect of price (i.e., price elasticity), maximizing the expected revenue through the experiment, and controlling the tail risk suffering from a very huge loss. In this paper, we reveal the relationship among such three objectives. Under a linear structural model, we investigate the trade-offs between causal inference and expected revenue maximization, as well as between expected revenue maximization and tail risk control. Furthermore, we propose an optimal pricing experimental design, which can flexibly adapt to different desired levels of trade-offs. Through the optimal design, we also explore the relationship between causal inference and tail risk control.

        ----

        ## [1319] Statistical Learning under Heterogenous Distribution Shift

        **Authors**: *Max Simchowitz, Anurag Ajay, Pulkit Agrawal, Akshay Krishnamurthy*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/simchowitz23a.html](https://proceedings.mlr.press/v202/simchowitz23a.html)

        **Abstract**:

        This paper studies the prediction of a target $\mathbf{z}$ from a pair of random variables $(\mathbf{x},\mathbf{y})$, where the ground-truth predictor is additive $\mathbb{E}[\mathbf{z} \mid \mathbf{x},\mathbf{y}] = f_\star(\mathbf{x}) +g_{\star}(\mathbf{y})$. We study the performance of empirical risk minimization (ERM) over functions $f+g$, $f \in \mathcal{F}$ and $g \in \mathcal{G}$, fit on a given training distribution, but evaluated on a test distribution which exhibits covariate shift. We show that, when the class $\mathcal{F}$ is "simpler" than $\mathcal{G}$ (measured, e.g., in terms of its metric entropy), our predictor is more resilient to heterogeneous covariate shifts in which the shift in $\mathbf{x}$ is much greater than that in $\mathbf{y}$. These results rely on a novel Hölder style inequality for the Dudley integral which may be of independent interest. Moreover, we corroborate our theoretical findings with experiments demonstrating improved resilience to shifts in "simpler" features across numerous domains.

        ----

        ## [1320] On the Stepwise Nature of Self-Supervised Learning

        **Authors**: *James B. Simon, Maksis Knutins, Ziyin Liu, Daniel Geisz, Abraham J. Fetterman, Joshua Albrecht*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/simon23a.html](https://proceedings.mlr.press/v202/simon23a.html)

        **Abstract**:

        We present a simple picture of the training process of self-supervised learning methods with dual deep networks. In our picture, these methods learn their high-dimensional embeddings one dimension at a time in a sequence of discrete, well-separated steps. We arrive at this picture via the study of a linear toy model of Barlow Twins, applicable to the case in which the trained network is infinitely wide. We solve the training dynamics of our toy model from small initialization, finding that the model learns the top eigenmodes of a certain contrastive kernel in a discrete, stepwise fashion, and find a closed-form expression for the final learned representations. Remarkably, we see the same stepwise learning phenomenon when training deep ResNets using the Barlow Twins, SimCLR, and VICReg losses. This stepwise picture partially demystifies the process of self-supervised training.

        ----

        ## [1321] Hindsight Learning for MDPs with Exogenous Inputs

        **Authors**: *Sean R. Sinclair, Felipe Vieira Frujeri, Ching-An Cheng, Luke Marshall, Hugo De Oliveira Barbalho, Jingling Li, Jennifer Neville, Ishai Menache, Adith Swaminathan*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/sinclair23a.html](https://proceedings.mlr.press/v202/sinclair23a.html)

        **Abstract**:

        Many resource management problems require sequential decision-making under uncertainty, where the only uncertainty affecting the decision outcomes are exogenous variables outside the control of the decision-maker. We model these problems as Exo-MDPs (Markov Decision Processes with Exogenous Inputs) and design a class of data-efficient algorithms for them termed Hindsight Learning (HL). Our HL algorithms achieve data efficiency by leveraging a key insight: having samples of the exogenous variables, past decisions can be revisited in hindsight to infer counterfactual consequences that can accelerate policy improvements. We compare HL against classic baselines in the multi-secretary and airline revenue management problems. We also scale our algorithms to a business-critical cloud resource management problem – allocating Virtual Machines (VMs) to physical machines, and simulate their performance with real datasets from a large public cloud provider. We find that HL algorithms outperform domain-specific heuristics, as well as state-of-the-art reinforcement learning methods.

        ----

        ## [1322] Text-To-4D Dynamic Scene Generation

        **Authors**: *Uriel Singer, Shelly Sheynin, Adam Polyak, Oron Ashual, Iurii Makarov, Filippos Kokkinos, Naman Goyal, Andrea Vedaldi, Devi Parikh, Justin Johnson, Yaniv Taigman*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/singer23a.html](https://proceedings.mlr.press/v202/singer23a.html)

        **Abstract**:

        We present MAV3D (Make-A-Video3D), a method for generating three-dimensional dynamic scenes from text descriptions. Our approach uses a 4D dynamic Neural Radiance Field (NeRF), which is optimized for scene appearance, density, and motion consistency by querying a Text-to-Video (T2V) diffusion-based model. The dynamic video output generated from the provided text can be viewed from any camera location and angle, and can be composited into any 3D environment. MAV3D does not require any 3D or 4D data and the T2V model is trained only on Text-Image pairs and unlabeled videos. We demonstrate the effectiveness of our approach using comprehensive quantitative and qualitative experiments and show an improvement over previously established internal baselines. To the best of our knowledge, our method is the first to generate 3D dynamic scenes given a text description. Generated samples can be viewed at make-a-video3d.github.io

        ----

        ## [1323] The Hessian perspective into the Nature of Convolutional Neural Networks

        **Authors**: *Sidak Pal Singh, Thomas Hofmann, Bernhard Schölkopf*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/singh23a.html](https://proceedings.mlr.press/v202/singh23a.html)

        **Abstract**:

        While Convolutional Neural Networks (CNNs) have long been investigated and applied, as well as theorized, we aim to provide a slightly different perspective into their nature — through the perspective of their Hessian maps. The reason is that the loss Hessian captures the pairwise interaction of parameters and therefore forms a natural ground to probe how the architectural aspects of CNNs get manifested in their structure and properties. We develop a framework relying on Toeplitz representation of CNNs, and then utilize it to reveal the Hessian structure and, in particular, its rank. We prove tight upper bounds (with linear activations), which closely follow the empirical trend of the Hessian rank and in practice also hold for more general settings. Overall, our work generalizes and further establishes the key insight that the Hessian rank grows as the square root of the number of parameters, even in CNNs.

        ----

        ## [1324] When do Minimax-fair Learning and Empirical Risk Minimization Coincide?

        **Authors**: *Harvineet Singh, Matthäus Kleindessner, Volkan Cevher, Rumi Chunara, Chris Russell*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/singh23b.html](https://proceedings.mlr.press/v202/singh23b.html)

        **Abstract**:

        Minimax-fair machine learning minimizes the error for the worst-off group. However, empirical evidence suggests that when sophisticated models are trained with standard empirical risk minimization (ERM), they often have the same performance on the worst-off group as a minimax-trained model. Our work makes this counter-intuitive observation concrete. We prove that if the hypothesis class is sufficiently expressive and the group information is recoverable from the features, ERM and minimax-fairness learning formulations indeed have the same performance on the worst-off group. We provide additional empirical evidence of how this observation holds on a wide range of datasets and hypothesis classes. Since ERM is fundamentally easier than minimax optimization, our findings have implications on the practice of fair machine learning.

        ----

        ## [1325] Differentiable Simulations for Enhanced Sampling of Rare Events

        **Authors**: *Martin Sípka, Johannes C. B. Dietschreit, Lukás Grajciar, Rafael Gómez-Bombarelli*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/sipka23a.html](https://proceedings.mlr.press/v202/sipka23a.html)

        **Abstract**:

        Simulating rare events, such as the transformation of a reactant into a product in a chemical reaction typically requires enhanced sampling techniques that rely on heuristically chosen collective variables (CVs). We propose using differentiable simulations (DiffSim) for the discovery and enhanced sampling of chemical transformations without a need to resort to preselected CVs, using only a distance metric. Reaction path discovery and estimation of the biasing potential that enhances the sampling are merged into a single end-to-end problem that is solved by path-integral optimization. This is achieved by introducing multiple improvements over standard DiffSim such as partial backpropagation and graph mini-batching making DiffSim training stable and efficient. The potential of DiffSim is demonstrated in the successful discovery of transition paths for the Muller-Brown model potential as well as a benchmark chemical system - alanine dipeptide.

        ----

        ## [1326] Preprocessors Matter! Realistic Decision-Based Attacks on Machine Learning Systems

        **Authors**: *Chawin Sitawarin, Florian Tramèr, Nicholas Carlini*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/sitawarin23a.html](https://proceedings.mlr.press/v202/sitawarin23a.html)

        **Abstract**:

        Decision-based attacks construct adversarial examples against a machine learning (ML) model by making only hard-label queries. These attacks have mainly been applied directly to standalone neural networks. However, in practice, ML models are just one component of a larger learning system. We find that by adding a single preprocessor in front of a classifier, state-of-the-art query-based attacks are up to seven× less effective at attacking a prediction pipeline than at attacking the model alone. We explain this discrepancy by the fact that most preprocessors introduce some notion of invariance to the input space. Hence, attacks that are unaware of this invariance inevitably waste a large number of queries to re-discover or overcome it. We, therefore, develop techniques to (i) reverse-engineer the preprocessor and then (ii) use this extracted information to attack the end-to-end system. Our preprocessors extraction method requires only a few hundred queries, and our preprocessor-aware attacks recover the same efficacy as when attacking the model alone. The code can be found at https://github.com/google-research/preprocessor-aware-black-box-attack.

        ----

        ## [1327] Invariance in Policy Optimisation and Partial Identifiability in Reward Learning

        **Authors**: *Joar Max Viktor Skalse, Matthew Farrugia-Roberts, Stuart Russell, Alessandro Abate, Adam Gleave*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/skalse23a.html](https://proceedings.mlr.press/v202/skalse23a.html)

        **Abstract**:

        It is often very challenging to manually design reward functions for complex, real-world tasks. To solve this, one can instead use reward learning to infer a reward function from data. However, there are often multiple reward functions that fit the data equally well, even in the infinite-data limit. This means that the reward function is only partially identifiable. In this work, we formally characterise the partial identifiability of the reward function given several popular reward learning data sources, including expert demonstrations and trajectory comparisons. We also analyse the impact of this partial identifiability for several downstream tasks, such as policy optimisation. We unify our results in a framework for comparing data sources and downstream tasks by their invariances, with implications for the design and selection of data sources for reward learning.

        ----

        ## [1328] A Game-Theoretic Framework for Managing Risk in Multi-Agent Systems

        **Authors**: *Oliver Slumbers, David Henry Mguni, Stefano B. Blumberg, Stephen Marcus McAleer, Yaodong Yang, Jun Wang*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/slumbers23a.html](https://proceedings.mlr.press/v202/slumbers23a.html)

        **Abstract**:

        In order for agents in multi-agent systems (MAS) to be safe, they need to take into account the risks posed by the actions of other agents. However, the dominant paradigm in game theory (GT) assumes that agents are not affected by risk from other agents and only strive to maximise their expected utility. For example, in hybrid human-AI driving systems, it is necessary to limit large deviations in reward resulting from car crashes. Although there are equilibrium concepts in game theory that take into account risk aversion, they either assume that agents are risk-neutral with respect to the uncertainty caused by the actions of other agents, or they are not guaranteed to exist. We introduce a new GT-based Risk-Averse Equilibrium (RAE) that always produces a solution that minimises the potential variance in reward accounting for the strategy of other agents. Theoretically and empirically, we show RAE shares many properties with a Nash Equilibrium (NE), establishing convergence properties and generalising to risk-dominant NE in certain cases. To tackle large-scale problems, we extend RAE to the PSRO multi-agent reinforcement learning (MARL) framework. We empirically demonstrate the minimum reward variance benefits of RAE in matrix games with high-risk outcomes. Results on MARL experiments show RAE generalises to risk-dominant NE in a trust dilemma game and that it reduces instances of crashing by 7x in an autonomous driving setting versus the best performing baseline.

        ----

        ## [1329] On the Effectiveness of Offline RL for Dialogue Response Generation

        **Authors**: *Paloma Sodhi, Felix Wu, Ethan R. Elenberg, Kilian Q. Weinberger, Ryan McDonald*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/sodhi23a.html](https://proceedings.mlr.press/v202/sodhi23a.html)

        **Abstract**:

        A common training technique for language models is teacher forcing (TF). TF attempts to match human language exactly, even though identical meanings can be expressed in different ways. This motivates use of sequence-level objectives for dialogue response generation. In this paper, we study the efficacy of various offline reinforcement learning (RL) methods to maximize such objectives. We present a comprehensive evaluation across multiple datasets, models, and metrics. Offline RL shows a clear performance improvement over teacher forcing while not inducing training instability or sacrificing practical training budgets.

        ----

        ## [1330] Fair Densities via Boosting the Sufficient Statistics of Exponential Families

        **Authors**: *Alexander Soen, Hisham Husain, Richard Nock*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/soen23a.html](https://proceedings.mlr.press/v202/soen23a.html)

        **Abstract**:

        We introduce a boosting algorithm to pre-process data for fairness. Starting from an initial fair but inaccurate distribution, our approach shifts towards better data fitting while still ensuring a minimal fairness guarantee. To do so, it learns the sufficient statistics of an exponential family with boosting-compliant convergence. Importantly, we are able to theoretically prove that the learned distribution will have a representation rate and statistical rate data fairness guarantee. Unlike recent optimization based pre-processing methods, our approach can be easily adapted for continuous domain features. Furthermore, when the weak learners are specified to be decision trees, the sufficient statistics of the learned distribution can be examined to provide clues on sources of (un)fairness. Empirical results are present to display the quality of result on real-world data.

        ----

        ## [1331] The Dormant Neuron Phenomenon in Deep Reinforcement Learning

        **Authors**: *Ghada Sokar, Rishabh Agarwal, Pablo Samuel Castro, Utku Evci*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/sokar23a.html](https://proceedings.mlr.press/v202/sokar23a.html)

        **Abstract**:

        In this work we identify the dormant neuron phenomenon in deep reinforcement learning, where an agent’s network suffers from an increasing number of inactive neurons, thereby affecting network expressivity. We demonstrate the presence of this phenomenon across a variety of algorithms and environments, and highlight its effect on learning. To address this issue, we propose a simple and effective method (ReDo) that Recycles Dormant neurons throughout training. Our experiments demonstrate that ReDo maintains the expressive power of networks by reducing the number of dormant neurons and results in improved performance.

        ----

        ## [1332] Abstracting Imperfect Information Away from Two-Player Zero-Sum Games

        **Authors**: *Samuel Sokota, Ryan D'Orazio, Chun Kai Ling, David J. Wu, J. Zico Kolter, Noam Brown*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/sokota23a.html](https://proceedings.mlr.press/v202/sokota23a.html)

        **Abstract**:

        In their seminal work, Nayyar et al. (2013) showed that imperfect information can be abstracted away from common-payoff games by having players publicly announce their policies as they play. This insight underpins sound solvers and decision-time planning algorithms for common-payoff games. Unfortunately, a naive application of the same insight to two-player zero-sum games fails because Nash equilibria of the game with public policy announcements may not correspond to Nash equilibria of the original game. As a consequence, existing sound decision-time planning algorithms require complicated additional mechanisms that have unappealing properties. The main contribution of this work is showing that certain regularized equilibria do not possess the aforementioned non-correspondence problem—thus, computing them can be treated as perfect-information problems. Because these regularized equilibria can be made arbitrarily close to Nash equilibria, our result opens the door to a new perspective to solving two-player zero-sum games and yields a simplified framework for decision-time planning in two-player zero-sum games, void of the unappealing properties that plague existing decision-time planning approaches.

        ----

        ## [1333] Meta-SAGE: Scale Meta-Learning Scheduled Adaptation with Guided Exploration for Mitigating Scale Shift on Combinatorial Optimization

        **Authors**: *Jiwoo Son, Minsu Kim, Hyeonah Kim, Jinkyoo Park*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/son23a.html](https://proceedings.mlr.press/v202/son23a.html)

        **Abstract**:

        This paper proposes Meta-SAGE, a novel approach for improving the scalability of deep reinforcement learning models for combinatorial optimization (CO) tasks. Our method adapts pre-trained models to larger-scale problems in test time by suggesting two components: a scale meta-learner (SML) and scheduled adaptation with guided exploration (SAGE). First, SML transforms the context embedding for subsequent adaptation of SAGE based on scale information. Then, SAGE adjusts the model parameters dedicated to the context embedding for a specific instance. SAGE introduces locality bias, which encourages selecting nearby locations to determine the next location. The locality bias gradually decays as the model is adapted to the target instance. Results show that Meta-SAGE outperforms previous adaptation methods and significantly improves scalability in representative CO tasks. Our source code is available at https://github.com/kaist-silab/meta-sage.

        ----

        ## [1334] Consistency Models

        **Authors**: *Yang Song, Prafulla Dhariwal, Mark Chen, Ilya Sutskever*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/song23a.html](https://proceedings.mlr.press/v202/song23a.html)

        **Abstract**:

        Diffusion models have significantly advanced the fields of image, audio, and video generation, but they depend on an iterative sampling process that causes slow generation. To overcome this limitation, we propose consistency models, a new family of models that generate high quality samples by directly mapping noise to data. They support fast one-step generation by design, while still allowing multistep sampling to trade compute for sample quality. They also support zero-shot data editing, such as image inpainting, colorization, and super-resolution, without requiring explicit training on these tasks. Consistency models can be trained either by distilling pre-trained diffusion models, or as standalone generative models altogether. Through extensive experiments, we demonstrate that they outperform existing distillation techniques for diffusion models in one- and few-step sampling, achieving the new state-of-the-art FID of 3.55 on CIFAR-10 and 6.20 on ImageNet 64x64 for one-step generation. When trained in isolation, consistency models become a new family of generative models that can outperform existing one-step, non-adversarial generative models on standard benchmarks such as CIFAR-10, ImageNet 64x64 and LSUN 256x256.

        ----

        ## [1335] LipsNet: A Smooth and Robust Neural Network with Adaptive Lipschitz Constant for High Accuracy Optimal Control

        **Authors**: *Xujie Song, Jingliang Duan, Wenxuan Wang, Shengbo Eben Li, Chen Chen, Bo Cheng, Bo Zhang, Junqing Wei, Xiaoming Simon Wang*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/song23b.html](https://proceedings.mlr.press/v202/song23b.html)

        **Abstract**:

        Deep reinforcement learning (RL) is a powerful approach for solving optimal control problems. However, RL-trained policies often suffer from the action fluctuation problem, where the consecutive actions significantly differ despite only slight state variations. This problem results in mechanical components’ wear and tear and poses safety hazards. The action fluctuation is caused by the high Lipschitz constant of actor networks. To address this problem, we propose a neural network named LipsNet. We propose the Multi-dimensional Gradient Normalization (MGN) method, to constrain the Lipschitz constant of networks with multi-dimensional input and output. Benefiting from MGN, LipsNet achieves Lipschitz continuity, allowing smooth actions while preserving control performance by adjusting Lipschitz constant. LipsNet addresses the action fluctuation problem at network level rather than algorithm level, which can serve as actor networks in most RL algorithms, making it more flexible and user-friendly than previous works. Experiments demonstrate that LipsNet has good landscape smoothness and noise robustness, resulting in significantly smoother action compared to the Multilayer Perceptron.

        ----

        ## [1336] Deep Perturbation Learning: Enhancing the Network Performance via Image Perturbations

        **Authors**: *Zifan Song, Xiao Gong, Guosheng Hu, Cairong Zhao*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/song23c.html](https://proceedings.mlr.press/v202/song23c.html)

        **Abstract**:

        Image perturbation technique is widely used to generate adversarial examples to attack networks, greatly decreasing the performance of networks. Unlike the existing works, in this paper, we introduce a novel framework Deep Perturbation Learning (DPL), the new insights into understanding image perturbations, to enhance the performance of networks rather than decrease the performance. Specifically, we learn image perturbations to amend the data distribution of training set to improve the performance of networks. This optimization w.r.t data distribution is non-trivial. To approach this, we tactfully construct a differentiable optimization target w.r.t. image perturbations via minimizing the empirical risk. Then we propose an alternating optimization of the network weights and perturbations. DPL can easily be adapted to a wide spectrum of downstream tasks and backbone networks. Extensive experiments demonstrate the effectiveness of our DPL on 6 datasets (CIFAR-10, CIFAR100, ImageNet, MS-COCO, PASCAL VOC, and SBD) over 3 popular vision tasks (image classification, object detection, and semantic segmentation) with different backbone architectures (e.g., ResNet, MobileNet, and ViT).

        ----

        ## [1337] Latent Traversals in Generative Models as Potential Flows

        **Authors**: *Yue Song, T. Anderson Keller, Nicu Sebe, Max Welling*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/song23d.html](https://proceedings.mlr.press/v202/song23d.html)

        **Abstract**:

        Despite the significant recent progress in deep generative models, the underlying structure of their latent spaces is still poorly understood, thereby making the task of performing semantically meaningful latent traversals an open research challenge. Most prior work has aimed to solve this challenge by modeling latent structures linearly, and finding corresponding linear directions which result in ‘disentangled’ generations. In this work, we instead propose to model latent structures with a learned dynamic potential landscape, thereby performing latent traversals as the flow of samples down the landscape’s gradient. Inspired by physics, optimal transport, and neuroscience, these potential landscapes are learned as physically realistic partial differential equations, thereby allowing them to flexibly vary over both space and time. To achieve disentanglement, multiple potentials are learned simultaneously, and are constrained by a classifier to be distinct and semantically self-consistent. Experimentally, we demonstrate that our method achieves both more qualitatively and quantitatively disentangled trajectories than state-of-the-art baselines. Further, we demonstrate that our method can be integrated as a regularization term during training, thereby acting as an inductive bias towards the learning of structured representations, ultimately improving model likelihood on similarly structured data. Code is available at https://github.com/KingJamesSong/PDETraversal.

        ----

        ## [1338] FedAvg Converges to Zero Training Loss Linearly for Overparameterized Multi-Layer Neural Networks

        **Authors**: *Bingqing Song, Prashant Khanduri, Xinwei Zhang, Jinfeng Yi, Mingyi Hong*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/song23e.html](https://proceedings.mlr.press/v202/song23e.html)

        **Abstract**:

        Federated Learning (FL) is a distributed learning paradigm that allows multiple clients to learn a joint model by utilizing privately held data at each client. Significant research efforts have been devoted to develop advanced algorithms that deal with the situation where the data at individual clients have heterogeneous distributions. In this work, we show that data heterogeneity can be dealt from a different perspective. That is, by utilizing a certain overparameterized multi-layer neural network at each client, even the vanilla FedAvg (a.k.a. the Local SGD) algorithm can accurately optimize the training problem: When each client has a neural network with one wide layer of size $N$ (where $N$ is the number of total training samples), followed by layers of smaller widths, FedAvg converges linearly to a solution that achieves (almost) zero training loss, without requiring any assumptions on the clients’ data distributions. To our knowledge, this is the first work that demonstrates such resilience to data heterogeneity for FedAvg when trained on multi-layer neural networks. Our experiments also confirm that, neural networks of large size can achieve better and more stable performance for FL problems.

        ----

        ## [1339] RGE: A Repulsive Graph Rectification for Node Classification via Influence

        **Authors**: *Jaeyun Song, Sungyub Kim, Eunho Yang*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/song23f.html](https://proceedings.mlr.press/v202/song23f.html)

        **Abstract**:

        In real-world graphs, noisy connections are inevitable, which makes it difficult to obtain unbiased node representations. Among various attempts to resolve this problem, a method of estimating the counterfactual effects of these connectivities has recently attracted attention, which mainly uses influence functions for single graph elements (i.e., node and edge). However, in this paper, we argue that there is a strongly interacting group effect between the influences of graph elements due to their connectivity. In the same vein, we observe that edge groups connecting to the same train node exhibit significant differences in their influences, hence no matter how negative each is, removing them at once may have a rather negative effect as a group. Based on this motivation, we propose a new edge-removing strategy, Repulsive edge Group Elimination (RGE), that preferentially removes edges with no interference in groups. Empirically, we demonstrate that RGE consistently outperforms existing methods on the various benchmark datasets.

        ----

        ## [1340] Importance Weighted Expectation-Maximization for Protein Sequence Design

        **Authors**: *Zhenqiao Song, Lei Li*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/song23g.html](https://proceedings.mlr.press/v202/song23g.html)

        **Abstract**:

        Designing protein sequences with desired biological function is crucial in biology and chemistry. Recent machine learning methods use a surrogate sequence-function model to replace the expensive wet-lab validation. How can we efficiently generate diverse and novel protein sequences with high fitness? In this paper, we propose IsEM-Pro, an approach to generate protein sequences towards a given fitness criterion. At its core, IsEM-Pro is a latent generative model, augmented by combinatorial structure features from a separately learned Markov random fields (MRFs). We develop an Monte Carlo Expectation-Maximization method (MCEM) to learn the model. During inference, sampling from its latent space enhances diversity while its MRFs features guide the exploration in high fitness regions. Experiments on eight protein sequence design tasks show that our IsEM-Pro outperforms the previous best methods by at least 55% on average fitness score and generates more diverse and novel protein sequences.

        ----

        ## [1341] Sketching for First Order Method: Efficient Algorithm for Low-Bandwidth Channel and Vulnerability

        **Authors**: *Zhao Song, Yitan Wang, Zheng Yu, Lichen Zhang*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/song23h.html](https://proceedings.mlr.press/v202/song23h.html)

        **Abstract**:

        Sketching is one of the most fundamental tools in large-scale machine learning. It enables runtime and memory saving via randomly compressing the original large problem into lower dimensions. In this paper, we propose a novel sketching scheme for the first order method in large-scale distributed learning setting, such that the communication costs between distributed agents are saved while the convergence of the algorithms is still guaranteed. Given gradient information in a high dimension $d$, the agent passes the compressed information processed by a sketching matrix $R\in \mathbb{R}^{s\times d}$ with $s\ll d$, and the receiver de-compressed via the de-sketching matrix $R^\top$ to “recover” the information in original dimension. Using such a framework, we develop algorithms for federated learning with lower communication costs. However, such random sketching does not protect the privacy of local data directly. We show that the gradient leakage problem still exists after applying the sketching technique by presenting a specific gradient attack method. As a remedy, we prove rigorously that the algorithm will be differentially private by adding additional random noises in gradient information, which results in a both communication-efficient and differentially private first order approach for federated learning tasks. Our sketching scheme can be further generalized to other learning settings and might be of independent interest itself.

        ----

        ## [1342] Sketching Meets Differential Privacy: Fast Algorithm for Dynamic Kronecker Projection Maintenance

        **Authors**: *Zhao Song, Xin Yang, Yuanyuan Yang, Lichen Zhang*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/song23i.html](https://proceedings.mlr.press/v202/song23i.html)

        **Abstract**:

        Projection maintenance is one of the core data structure tasks. Efficient data structures for projection maintenance have led to recent breakthroughs in many convex programming algorithms. In this work, we further extend this framework to the Kronecker product structure. Given a constraint matrix ${\sf A}$ and a positive semi-definite matrix $W\in \mathbb{R}^{n\times n}$ with a sparse eigenbasis, we consider the task of maintaining the projection in the form of ${\sf B}^\top({\sf B}{\sf B}^\top)^{-1}{\sf B}$, where ${\sf B}={\sf A}(W\otimes I)$ or ${\sf B}={\sf A}(W^{1/2}\otimes W^{1/2})$. At each iteration, the weight matrix $W$ receives a low rank change and we receive a new vector $h$. The goal is to maintain the projection matrix and answer the query ${\sf B}^\top({\sf B}{\sf B}^\top)^{-1}{\sf B}h$ with good approximation guarantees. We design a fast dynamic data structure for this task and it is robust against an adaptive adversary. Following the beautiful and pioneering work of [Beimel, Kaplan, Mansour, Nissim, Saranurak and Stemmer, STOC’22], we use tools from differential privacy to reduce the randomness required by the data structure and further improve the running time.

        ----

        ## [1343] A Nearly-Optimal Bound for Fast Regression with ℓ∞ Guarantee

        **Authors**: *Zhao Song, Mingquan Ye, Junze Yin, Lichen Zhang*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/song23j.html](https://proceedings.mlr.press/v202/song23j.html)

        **Abstract**:

        Given a matrix $A\in \mathbb{R}^{n\times d}$ and a vector $b\in \mathbb{R}^n$, we consider the regression problem with $\ell_\infty$ guarantees: finding a vector $x’\in \mathbb{R}^d$ such that $||x’-x^* ||_\infty \leq \frac{\epsilon}{\sqrt{d}}\cdot ||Ax^*-b||_2\cdot ||A^\dagger||$ with $x^*$ being the optimal solution to the regression $||Ax-b||_2$. One popular approach for solving $\ell_2$ regression problem is via sketching: picking a structured random matrix $S\in \mathbb{R}^{m\times n}$ with $m\ll n$ and $SA$ can be quickly computed, solve the “sketched” regression problem $x’=\mathrm{argmin} ||SAx-Sb||_2$. In this paper, we show that in order to obtain such $\ell_\infty$ guarantee for $\ell_2$ regression, one has to use sketching matrices that are dense. To the best of our knowledge, this is the first user case in which dense sketching matrices are necessary. On the algorithmic side, we prove that, there exists a distribution of dense sketching matrices with $m=\epsilon^{-2}d\log^3(n/\delta)$ such that solving the sketched regression problem gives the $\ell_\infty$ guarantee, with probability at least $1-\delta$. Moreover, the matrix $SA$ can be computed in time $O(nd\log n)$. Our row count is nearly-optimal up to logarithmic factors, and significantly improves the result in [Price, Song and Woodruff, ICALP’17], in which $m=\Omega(\epsilon^{-2}d^{1+\gamma})$ for $\gamma\in (0, 1)$ is required. Moreover, we develop a novel analytical framework for $\ell_\infty$ guarantee regression that utilizes the Oblivious Coordinate-wise Embedding (OCE) property introduced in [Song and Yu, ICML’21]. Our analysis is much simpler and more general than that of [Price, Song and Woodruff, ICALP’17]. Leveraging this framework, we extend the $\ell_\infty$ guarantee regression result to dense sketching matrices for computing fast tensor product of vectors.

        ----

        ## [1344] Loss-Guided Diffusion Models for Plug-and-Play Controllable Generation

        **Authors**: *Jiaming Song, Qinsheng Zhang, Hongxu Yin, Morteza Mardani, Ming-Yu Liu, Jan Kautz, Yongxin Chen, Arash Vahdat*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/song23k.html](https://proceedings.mlr.press/v202/song23k.html)

        **Abstract**:

        We consider guiding denoising diffusion models with general differentiable loss functions in a plug-and-play fashion, enabling controllable generation without additional training. This paradigm, termed Loss-Guided Diffusion (LGD), can easily be integrated into all diffusion models and leverage various efficient samplers. Despite the benefits, the resulting guidance term is, unfortunately, an intractable integral and needs to be approximated. Existing methods compute the guidance term based on a point estimate. However, we show that such approaches have significant errors over the scale of the approximations. To address this issue, we propose a Monte Carlo method that uses multiple samples from a suitable distribution to reduce bias. Our method is effective in various synthetic and real-world settings, including image super-resolution, text or label-conditional image generation, and controllable motion synthesis. Notably, we show how our method can be applied to control a pretrained motion diffusion model to follow certain paths and avoid obstacles that are proven challenging to prior methods.

        ----

        ## [1345] Differentiable Tree Operations Promote Compositional Generalization

        **Authors**: *Paul Soulos, Edward J. Hu, Kate McCurdy, Yunmo Chen, Roland Fernandez, Paul Smolensky, Jianfeng Gao*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/soulos23a.html](https://proceedings.mlr.press/v202/soulos23a.html)

        **Abstract**:

        In the context of structure-to-structure transformation tasks, learning sequences of discrete symbolic operations poses significant challenges due to their non-differentiability. To facilitate the learning of these symbolic sequences, we introduce a differentiable tree interpreter that compiles high-level symbolic tree operations into subsymbolic matrix operations on tensors. We present a novel Differentiable Tree Machine (DTM) architecture that integrates our interpreter with an external memory and an agent that learns to sequentially select tree operations to execute the target transformation in an end-to-end manner. With respect to out-of-distribution compositional generalization on synthetic semantic parsing and language generation tasks, DTM achieves 100% while existing baselines such as Transformer, Tree Transformer, LSTM, and Tree2Tree LSTM achieve less than 30%. DTM remains highly interpretable in addition to its perfect performance.

        ----

        ## [1346] Are labels informative in semi-supervised learning? Estimating and leveraging the missing-data mechanism

        **Authors**: *Aude Sportisse, Hugo Schmutz, Olivier Humbert, Charles Bouveyron, Pierre-Alexandre Mattei*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/sportisse23a.html](https://proceedings.mlr.press/v202/sportisse23a.html)

        **Abstract**:

        Semi-supervised learning is a powerful technique for leveraging unlabeled data to improve machine learning models, but it can be affected by the presence of “informative" labels, which occur when some classes are more likely to be labeled than others. In the missing data literature, such labels are called missing not at random. In this paper, we propose a novel approach to address this issue by estimating the missing-data mechanism and using inverse propensity weighting to debias any SSL algorithm, including those using data augmentation. We also propose a likelihood ratio test to assess whether or not labels are indeed informative. Finally, we demonstrate the performance of the proposed methods on different datasets, in particular on two medical datasets for which we design pseudo-realistic missing data scenarios.

        ----

        ## [1347] Linear Causal Disentanglement via Interventions

        **Authors**: *Chandler Squires, Anna Seigal, Salil S. Bhate, Caroline Uhler*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/squires23a.html](https://proceedings.mlr.press/v202/squires23a.html)

        **Abstract**:

        Causal disentanglement seeks a representation of data involving latent variables that are related via a causal model. A representation is identifiable if both the latent model and the transformation from latent to observed variables are unique. In this paper, we study observed variables that are a linear transformation of a linear latent causal model. Data from interventions are necessary for identifiability: if one latent variable is missing an intervention, we show that there exist distinct models that cannot be distinguished. Conversely, we show that a single intervention on each latent variable is sufficient for identifiability. Our proof uses a generalization of the RQ decomposition of a matrix that replaces the usual orthogonal and upper triangular conditions with analogues depending on a partial order on the rows of the matrix, with partial order determined by a latent causal model. We corroborate our theoretical results with a method for causal disentanglement. We show that the method accurately recovers a latent causal model on synthetic and semi-synthetic data and we illustrate a use case on a dataset of single-cell RNA sequencing measurements.

        ----

        ## [1348] Generating Language Corrections for Teaching Physical Control Tasks

        **Authors**: *Megha Srivastava, Noah D. Goodman, Dorsa Sadigh*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/srivastava23a.html](https://proceedings.mlr.press/v202/srivastava23a.html)

        **Abstract**:

        AI assistance continues to help advance applications in education, from language learning to intelligent tutoring systems, yet current methods for providing students feedback are still quite limited. Most automatic feedback systems either provide binary correctness feedback, which may not help a student understand how to improve, or require hand-coding feedback templates, which may not generalize to new domains. This can be particularly challenging for physical control tasks, where the rich diversity in student behavior and specialized domains make it challenging to leverage general-purpose assistive tools for providing feedback. We design and build CORGI, a model trained to generate language corrections for physical control tasks, such as learning to ride a bike. CORGI takes in as input a pair of student and expert trajectories, and then generates natural language corrections to help the student improve. We collect and train CORGI over data from three diverse physical control tasks (drawing, steering, and joint movement). Through both automatic and human evaluations, we show that CORGI can (i) generate valid feedback for novel student trajectories, (ii) outperform baselines on domains with novel control dynamics, and (iii) improve student learning in an interactive drawing task.

        ----

        ## [1349] FaDIn: Fast Discretized Inference for Hawkes Processes with General Parametric Kernels

        **Authors**: *Guillaume Staerman, Cédric Allain, Alexandre Gramfort, Thomas Moreau*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/staerman23a.html](https://proceedings.mlr.press/v202/staerman23a.html)

        **Abstract**:

        Temporal point processes (TPP) are a natural tool for modeling event-based data. Among all TPP models, Hawkes processes have proven to be the most widely used, mainly due to their adequate modeling for various applications, particularly when considering exponential or non-parametric kernels. Although non-parametric kernels are an option, such models require large datasets. While exponential kernels are more data efficient and relevant for specific applications where events immediately trigger more events, they are ill-suited for applications where latencies need to be estimated, such as in neuroscience. This work aims to offer an efficient solution to TPP inference using general parametric kernels with finite support. The developed solution consists of a fast $\ell_2$ gradient-based solver leveraging a discretized version of the events. After theoretically supporting the use of discretization, the statistical and computational efficiency of the novel approach is demonstrated through various numerical experiments. Finally, the method’s effectiveness is evaluated by modeling the occurrence of stimuli-induced patterns from brain signals recorded with magnetoencephalography (MEG). Given the use of general parametric kernels, results show that the proposed approach leads to an improved estimation of pattern latency than the state-of-the-art.

        ----

        ## [1350] Partial Optimality in Cubic Correlation Clustering

        **Authors**: *David Stein, Silvia Di Gregorio, Bjoern Andres*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/stein23a.html](https://proceedings.mlr.press/v202/stein23a.html)

        **Abstract**:

        The higher-order correlation clustering problem is an expressive model, and recently, local search heuristics have been proposed for several applications. Certifying optimality, however, is NP-hard and practically hampered already by the complexity of the problem statement. Here, we focus on establishing partial optimality conditions for the special case of complete graphs and cubic objective functions. In addition, we define and implement algorithms for testing these conditions and examine their effect numerically, on two datasets.

        ----

        ## [1351] MODeL: Memory Optimizations for Deep Learning

        **Authors**: *Benoit Steiner, Mostafa Elhoushi, Jacob Kahn, James Hegarty*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/steiner23a.html](https://proceedings.mlr.press/v202/steiner23a.html)

        **Abstract**:

        The size of deep neural networks has grown exponentially in recent years. Unfortunately, hardware devices have not kept pace with the rapidly increasing memory requirements. To cope with this, researchers have proposed various techniques including spilling, rematerialization, reduced precision training, model pruning, and so on. However, these approaches suffer from various limitations, such as increasing training time, affecting model accuracy, or requiring extensive manual modifications to the neural networks. We present MODeL, an algorithm that optimizes the lifetime and memory location of the tensors used to train neural networks. Our method automatically reduces the memory usage of existing neural networks without any of the drawbacks of other techniques. We formulate the problem as a joint integer linear program (ILP). We present several techniques to simplify the encoding of the problem, and enable our approach to scale to the size of state-of-the-art neural networks using an off-the-shelf ILP solver. We experimentally demonstrate that MODeL only takes seconds to allow the training of neural networks using 30% less memory on average.

        ----

        ## [1352] Improving Expert Predictions with Conformal Prediction

        **Authors**: *Eleni Straitouri, Lequn Wang, Nastaran Okati, Manuel Gomez Rodriguez*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/straitouri23a.html](https://proceedings.mlr.press/v202/straitouri23a.html)

        **Abstract**:

        Automated decision support systems promise to help human experts solve multiclass classification tasks more efficiently and accurately. However, existing systems typically require experts to understand when to cede agency to the system or when to exercise their own agency. Otherwise, the experts may be better off solving the classification tasks on their own. In this work, we develop an automated decision support system that, by design, does not require experts to understand when to trust the system to improve performance. Rather than providing (single) label predictions and letting experts decide when to trust these predictions, our system provides sets of label predictions constructed using conformal prediction—prediction sets—and forcefully asks experts to predict labels from these sets. By using conformal prediction, our system can precisely trade-off the probability that the true label is not in the prediction set, which determines how frequently our system will mislead the experts, and the size of the prediction set, which determines the difficulty of the classification task the experts need to solve using our system. In addition, we develop an efficient and near-optimal search method to find the conformal predictor under which the experts benefit the most from using our system. Simulation experiments using synthetic and real expert predictions demonstrate that our system may help experts make more accurate predictions and is robust to the accuracy of the classifier the conformal predictor relies on.

        ----

        ## [1353] Lookahead When It Matters: Adaptive Non-causal Transformers for Streaming Neural Transducers

        **Authors**: *Grant P. Strimel, Yi Xie, Brian John King, Martin Radfar, Ariya Rastrow, Athanasios Mouchtaris*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/strimel23a.html](https://proceedings.mlr.press/v202/strimel23a.html)

        **Abstract**:

        Streaming speech recognition architectures are employed for low-latency, real-time applications. Such architectures are often characterized by their causality. Causal architectures emit tokens at each frame, relying only on current and past signal, while non-causal models are exposed to a window of future frames at each step to increase predictive accuracy. This dichotomy amounts to a trade-off for real-time Automatic Speech Recognition (ASR) system design: profit from the low-latency benefit of strictly-causal architectures while accepting predictive performance limitations, or realize the modeling benefits of future-context models accompanied by their higher latency penalty. In this work, we relax the constraints of this choice and present the Adaptive Non-Causal Attention Transducer (ANCAT). Our architecture is non-causal in the traditional sense, but executes in a low-latency, streaming manner by dynamically choosing when to rely on future context and to what degree within the audio stream. The resulting mechanism, when coupled with our novel regularization algorithms, delivers comparable accuracy to non-causal configurations while improving significantly upon latency, closing the gap with their causal counterparts. We showcase our design experimentally by reporting comparative ASR task results with measures of accuracy and latency on both publicly accessible and production-scale, voice-assistant datasets.

        ----

        ## [1354] Kernel QuantTree

        **Authors**: *Diego Stucchi, Paolo Rizzo, Nicolò Folloni, Giacomo Boracchi*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/stucchi23a.html](https://proceedings.mlr.press/v202/stucchi23a.html)

        **Abstract**:

        We present Kernel QuantTree (KQT), a non-parametric change detection algorithm that monitors multivariate data through a histogram. KQT constructs a nonlinear partition of the input space that matches pre-defined target probabilities and specifically promotes compact bins adhering to the data distribution, resulting in a powerful detection algorithm. We prove two key theoretical advantages of KQT: i) statistics defined over the KQT histogram do not depend on the stationary data distribution $\phi_0$, so detection thresholds can be set a priori to control false positive rate, and ii) thanks to the kernel functions adopted, the KQT monitoring scheme is invariant to the roto-translation of the input data. Consequently, KQT does not require any preprocessing step like PCA. Our experiments show that KQT achieves superior detection power than non-parametric state-of-the-art change detection methods, and can reliably control the false positive rate.

        ----

        ## [1355] Topologically Faithful Image Segmentation via Induced Matching of Persistence Barcodes

        **Authors**: *Nico Stucki, Johannes C. Paetzold, Suprosanna Shit, Bjoern H. Menze, Ulrich Bauer*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/stucki23a.html](https://proceedings.mlr.press/v202/stucki23a.html)

        **Abstract**:

        Segmentation models predominantly optimize pixel-overlap-based loss, an objective that is actually inadequate for many segmentation tasks. In recent years, their limitations fueled a growing interest in topology-aware methods, which aim to recover the topology of the segmented structures. However, so far, existing methods only consider global topological properties, ignoring the need to preserve topological features spatially, which is crucial for accurate segmentation. We introduce the concept of induced matchings from persistent homology to achieve a spatially correct matching between persistence barcodes in a segmentation setting. Based on this concept, we define the Betti matching error as an interpretable, topologically and feature-wise accurate metric for image segmentations, which resolves the limitations of the Betti number error. Our Betti matching error is differentiable and efficient to use as a loss function. We demonstrate that it improves the topological performance of segmentation networks significantly across six diverse datasets while preserving the performance with respect to traditional scores. Our code is publicly available (https://github.com/nstucki/Betti-matching/).

        ----

        ## [1356] Towards Robust Graph Incremental Learning on Evolving Graphs

        **Authors**: *Junwei Su, Difan Zou, Zijun Zhang, Chuan Wu*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/su23a.html](https://proceedings.mlr.press/v202/su23a.html)

        **Abstract**:

        Incremental learning is a machine learning approach that involves training a model on a sequence of tasks, rather than all tasks at once. This ability to learn incrementally from a stream of tasks is crucial for many real-world applications. However, incremental learning is a challenging problem on graph-structured data, as many graph-related problems involve prediction tasks for each individual node, known as Node-wise Graph Incremental Learning (NGIL). This introduces non-independent and non-identically distributed characteristics in the sample data generation process, making it difficult to maintain the performance of the model as new tasks are added. In this paper, we focus on the inductive NGIL problem, which accounts for the evolution of graph structure (structural shift) induced by emerging tasks. We provide a formal formulation and analysis of the problem, and propose a novel regularization-based technique called Structural-Shift-Risk-Mitigation (SSRM) to mitigate the impact of the structural shift on catastrophic forgetting of the inductive NGIL problem. We show that the structural shift can lead to a shift in the input distribution for the existing tasks, and further lead to an increased risk of catastrophic forgetting. Through comprehensive empirical studies with several benchmark datasets, we demonstrate that our proposed method, Structural-Shift-Risk-Mitigation (SSRM), is flexible and easy to adapt to improve the performance of state-of-the-art GNN incremental learning frameworks in the inductive setting.

        ----

        ## [1357] DUET: 2D Structured and Approximately Equivariant Representations

        **Authors**: *Xavier Suau, Federico Danieli, T. Anderson Keller, Arno Blaas, Chen Huang, Jason Ramapuram, Dan Busbridge, Luca Zappella*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/suau23a.html](https://proceedings.mlr.press/v202/suau23a.html)

        **Abstract**:

        Multiview Self-Supervised Learning (MSSL) is based on learning invariances with respect to a set of input transformations. However, invariance partially or totally removes transformation-related information from the representations, which might harm performance for specific downstream tasks that require such information. We propose 2D strUctured and EquivarianT representations (coined DUET), which are 2d representations organized in a matrix structure, and equivariant with respect to transformations acting on the input data. DUET representations maintain information about an input transformation, while remaining semantically expressive. Compared to SimCLR (Chen et al., 2020) (unstructured and invariant) and ESSL (Dangovski et al., 2022) (unstructured and equivariant), the structured and equivariant nature of DUET representations enables controlled generation with lower reconstruction error, while controllability is not possible with SimCLR or ESSL. DUET also achieves higher accuracy for several discriminative tasks, and improves transfer learning.

        ----

        ## [1358] Long-Tailed Recognition by Mutual Information Maximization between Latent Features and Ground-Truth Labels

        **Authors**: *Min-Kook Suh, Seung-Woo Seo*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/suh23a.html](https://proceedings.mlr.press/v202/suh23a.html)

        **Abstract**:

        Although contrastive learning methods have shown prevailing performance on a variety of representation learning tasks, they encounter difficulty when the training dataset is long-tailed. Many researchers have combined contrastive learning and a logit adjustment technique to address this problem, but the combinations are done ad-hoc and a theoretical background has not yet been provided. The goal of this paper is to provide the background and further improve the performance. First, we show that the fundamental reason contrastive learning methods struggle with long-tailed tasks is that they try to maximize the mutual information between latent features and input data. As ground-truth labels are not considered in the maximization, they are not able to address imbalances between classes. Rather, we interpret the long-tailed recognition task as a mutual information maximization between latent features and ground-truth labels. This approach integrates contrastive learning and logit adjustment seamlessly to derive a loss function that shows state-of-the-art performance on long-tailed recognition benchmarks. It also demonstrates its efficacy in image segmentation tasks, verifying its versatility beyond image classification. Code is available at https://github.com/bluecdm/Long-tailed-recognition.

        ----

        ## [1359] Adversarial Learning of Distributional Reinforcement Learning

        **Authors**: *Yang Sui, Yukun Huang, Hongtu Zhu, Fan Zhou*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/sui23a.html](https://proceedings.mlr.press/v202/sui23a.html)

        **Abstract**:

        Reinforcement learning (RL) has made significant advancements in artificial intelligence. However, its real-world applications are limited due to differences between simulated environments and the actual world. Consequently, it is crucial to systematically analyze how each component of the RL system can affect the final model performance. In this study, we propose an adversarial learning framework for distributional reinforcement learning, which adopts the concept of influence measure from the statistics community. This framework enables us to detect performance loss caused by either the internal policy structure or the external state observation. The proposed influence measure is based on information geometry and has desirable properties of invariance. We demonstrate that the influence measure is useful for three diagnostic tasks: identifying fragile states in trajectories, determining the instability of the policy architecture, and pinpointing anomalously sensitive policy parameters.

        ----

        ## [1360] Distilling Internet-Scale Vision-Language Models into Embodied Agents

        **Authors**: *Theodore R. Sumers, Kenneth Marino, Arun Ahuja, Rob Fergus, Ishita Dasgupta*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/sumers23a.html](https://proceedings.mlr.press/v202/sumers23a.html)

        **Abstract**:

        Instruction-following agents must ground language into their observation and action spaces. Learning to ground language is challenging, typically requiring domain-specific engineering or large quantities of human interaction data. To address this challenge, we propose using pretrained vision-language models (VLMs) to supervise embodied agents. We combine ideas from model distillation and hindsight experience replay (HER), using a VLM to retroactively generate language describing the agent’s behavior. Simple prompting allows us to control the supervision signal, teaching an agent to interact with novel objects based on their names (e.g., planes) or their features (e.g., colors) in a 3D rendered environment. Fewshot prompting lets us teach abstract category membership, including pre-existing categories (food vs toys) and ad-hoc ones (arbitrary preferences over objects). Our work outlines a new and effective way to use internet-scale VLMs, repurposing the generic language grounding acquired by such models to teach task-relevant groundings to embodied agents.

        ----

        ## [1361] Vector-Valued Control Variates

        **Authors**: *Zhuo Sun, Alessandro Barp, François-Xavier Briol*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/sun23a.html](https://proceedings.mlr.press/v202/sun23a.html)

        **Abstract**:

        Control variates are variance reduction tools for Monte Carlo estimators. They can provide significant variance reduction, but usually require a large number of samples, which can be prohibitive when sampling or evaluating the integrand is computationally expensive. Furthermore, there are many scenarios where we need to compute multiple related integrals simultaneously or sequentially, which can further exacerbate computational costs. In this paper, we propose vector-valued control variates, an extension of control variates which can be used to reduce the variance of multiple Monte Carlo estimators jointly. This allows for the transfer of information across integration tasks, and hence reduces the need for a large number of samples. We focus on control variates based on kernel interpolants and our novel construction is obtained through a generalised Stein identity and the development of novel matrix-valued Stein reproducing kernels. We demonstrate our methodology on a range of problems including multifidelity modelling, Bayesian inference for dynamical systems, and model evidence computation through thermodynamic integration.

        ----

        ## [1362] MetaModulation: Learning Variational Feature Hierarchies for Few-Shot Learning with Fewer Tasks

        **Authors**: *Wenfang Sun, Yingjun Du, Xiantong Zhen, Fan Wang, Ling Wang, Cees G. M. Snoek*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/sun23b.html](https://proceedings.mlr.press/v202/sun23b.html)

        **Abstract**:

        Meta-learning algorithms are able to learn a new task using previously learned knowledge, but they often require a large number of meta-training tasks which may not be readily available. To address this issue, we propose a method for few-shot learning with fewer tasks, which we call MetaModulation. The key idea is to use a neural network to increase the density of the meta-training tasks by modulating batch normalization parameters during meta-training. Additionally, we modify parameters at various neural network levels, rather than just a single layer, to increase task diversity. To account for the uncertainty caused by the reduced number of training tasks, we propose a variational MetaModulation where the modulation parameters are treated as latent variables. We also introduce learning variational feature hierarchies by the variational MetaModulation, which modulates features at all layers and can take into account task uncertainty and generate more diverse tasks. The ablation studies illustrate the advantages of utilizing a learnable task modulation at different levels and demonstrate the benefit of incorporating probabilistic variants in few-task meta-learning. Our MetaModulation and its variational variants consistently outperform state-of-the-art alternatives on four few-task meta-learning benchmarks.

        ----

        ## [1363] Revisiting Sampling for Combinatorial Optimization

        **Authors**: *Haoran Sun, Katayoon Goshvadi, Azade Nova, Dale Schuurmans, Hanjun Dai*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/sun23c.html](https://proceedings.mlr.press/v202/sun23c.html)

        **Abstract**:

        Sampling approaches like Markov chain Monte Carlo were once popular for combinatorial optimization, but the inefficiency of classical methods and the need for problem-specific designs curtailed ongoing development. Recent work has favored data-driven approaches that mitigate the need for hand-craft heuristics, but these are often not usable as out-of-the-box solvers due to dependence on in-distribution training and limited scalability to large instances. In this paper, we revisit the idea of using sampling for combinatorial optimization, motivated by the significant recent advances of gradient-based discrete MCMC and new techniques for parallel neighborhood exploration on accelerators. Remarkably, we find that modern sampling strategies can leverage landscape information to provide general-purpose solvers that require no training and yet are competitive with state of the art combinatorial solvers. In particular, experiments on cover vertex selection, graph partition and routing demonstrate better speed-quality trade-offs over current learning based approaches, and sometimes even superior performance to commercial solvers and specialized algorithms.

        ----

        ## [1364] What Makes Entities Similar? A Similarity Flooding Perspective for Multi-sourced Knowledge Graph Embeddings

        **Authors**: *Zequn Sun, Jiacheng Huang, Xiaozhou Xu, Qijin Chen, Weijun Ren, Wei Hu*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/sun23d.html](https://proceedings.mlr.press/v202/sun23d.html)

        **Abstract**:

        Joint representation learning over multi-sourced knowledge graphs (KGs) yields transferable and expressive embeddings that improve downstream tasks. Entity alignment (EA) is a critical step in this process. Despite recent considerable research progress in embedding-based EA, how it works remains to be explored. In this paper, we provide a similarity flooding perspective to explain existing translation-based and aggregation-based EA models. We prove that the embedding learning process of these models actually seeks a fixpoint of pairwise similarities between entities. We also provide experimental evidence to support our theoretical analysis. We propose two simple but effective methods inspired by the fixpoint computation in similarity flooding, and demonstrate their effectiveness on benchmark datasets. Our work bridges the gap between recent embedding-based models and the conventional similarity flooding algorithm. It would improve our understanding of and increase our faith in embedding-based EA.

        ----

        ## [1365] Maximum Optimality Margin: A Unified Approach for Contextual Linear Programming and Inverse Linear Programming

        **Authors**: *Chunlin Sun, Shang Liu, Xiaocheng Li*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/sun23e.html](https://proceedings.mlr.press/v202/sun23e.html)

        **Abstract**:

        In this paper, we study the predict-then-optimize problem where the output of a machine learning prediction task is used as the input of some downstream optimization problem, say, the objective coefficient vector of a linear program. The problem is also known as predictive analytics or contextual linear programming. The existing approaches largely suffer from either (i) optimization intractability (a non-convex objective function)/statistical inefficiency (a suboptimal generalization bound) or (ii) requiring strong condition(s) such as no constraint or loss calibration. We develop a new approach to the problem called maximum optimality margin which designs the machine learning loss function by the optimality condition of the downstream optimization. The max-margin formulation enjoys both computational efficiency and good theoretical properties for the learning procedure. More importantly, our new approach only needs the observations of the optimal solution in the training data rather than the objective function, which makes it a new and natural approach to the inverse linear programming problem under both contextual and context-free settings; we also analyze the proposed method under both offline and online settings, and demonstrate its performance using numerical experiments.

        ----

        ## [1366] Tensor Gaussian Process with Contraction for Multi-Channel Imaging Analysis

        **Authors**: *Hu Sun, Ward Manchester, Meng Jin, Yang Liu, Yang Chen*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/sun23f.html](https://proceedings.mlr.press/v202/sun23f.html)

        **Abstract**:

        Multi-channel imaging data is a prevalent data format in scientific fields such as astronomy and biology. The structured information and the high dimensionality of these 3-D tensor data makes the analysis an intriguing but challenging topic for statisticians and practitioners. The low-rank scalar-on-tensor regression model, in particular, has received widespread attention and has been re-formulated as a tensor Gaussian Process (Tensor-GP) model with multi-linear kernel in Yu et al. (2018). In this paper, we extend the Tensor-GP model by introducing an integrative dimensionality reduction technique, called tensor contraction, with a Tensor-GP for a scalar-on-tensor regression task with multi-channel imaging data. This is motivated by the solar flare forecasting problem with high dimensional multi-channel imaging data. We first estimate a latent, reduced-size tensor for each data tensor and then apply a multi-linear Tensor-GP on the latent tensor data for prediction. We introduce an anisotropic total-variation regularization when conducting the tensor contraction to obtain a sparse and smooth latent tensor. We then propose an alternating proximal gradient descent algorithm for estimation. We validate our approach via extensive simulation studies and applying it to the solar flare forecasting problem.

        ----

        ## [1367] MABe22: A Multi-Species Multi-Task Benchmark for Learned Representations of Behavior

        **Authors**: *Jennifer J. Sun, Markus Marks, Andrew Wesley Ulmer, Dipam Chakraborty, Brian Geuther, Edward Hayes, Heng Jia, Vivek Kumar, Sebastian Oleszko, Zachary Partridge, Milan Peelman, Alice Robie, Catherine E. Schretter, Keith Sheppard, Chao Sun, Param Uttarwar, Julian Morgan Wagner, Erik Werner, Joseph Parker, Pietro Perona, Yisong Yue, Kristin Branson, Ann Kennedy*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/sun23g.html](https://proceedings.mlr.press/v202/sun23g.html)

        **Abstract**:

        We introduce MABe22, a large-scale, multi-agent video and trajectory benchmark to assess the quality of learned behavior representations. This dataset is collected from a variety of biology experiments, and includes triplets of interacting mice (4.7 million frames video+pose tracking data, 10 million frames pose only), symbiotic beetle-ant interactions (10 million frames video data), and groups of interacting flies (4.4 million frames of pose tracking data). Accompanying these data, we introduce a panel of real-life downstream analysis tasks to assess the quality of learned representations by evaluating how well they preserve information about the experimental conditions (e.g. strain, time of day, optogenetic stimulation) and animal behavior. We test multiple state-of-the-art self-supervised video and trajectory representation learning methods to demonstrate the use of our benchmark, revealing that methods developed using human action datasets do not fully translate to animal datasets. We hope that our benchmark and dataset encourage a broader exploration of behavior representation learning methods across species and settings.

        ----

        ## [1368] Dynamic Regularized Sharpness Aware Minimization in Federated Learning: Approaching Global Consistency and Smooth Landscape

        **Authors**: *Yan Sun, Li Shen, Shixiang Chen, Liang Ding, Dacheng Tao*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/sun23h.html](https://proceedings.mlr.press/v202/sun23h.html)

        **Abstract**:

        In federated learning (FL), a cluster of local clients are chaired under the coordination of the global server and cooperatively train one model with privacy protection. Due to the multiple local updates and the isolated non-iid dataset, clients are prone to overfit into their own optima, which extremely deviates from the global objective and significantly undermines the performance. Most previous works only focus on enhancing the consistency between the local and global objectives to alleviate this prejudicial client drifts from the perspective of the optimization view, whose performance would be prominently deteriorated on the high heterogeneity. In this work, we propose a novel and general algorithm FedSMOO by jointly considering the optimization and generalization targets to efficiently improve the performance in FL. Concretely, FedSMOO adopts a dynamic regularizer to guarantee the local optima towards the global objective, which is meanwhile revised by the global Sharpness Aware Minimization (SAM) optimizer to search for the consistent flat minima. Our theoretical analysis indicates that FedSMOO achieves fast $\mathcal{O}(1/T)$ convergence rate with low generalization bound. Extensive numerical studies are conducted on the real-world dataset to verify its peerless efficiency and excellent generality.

        ----

        ## [1369] When and How Does Known Class Help Discover Unknown Ones? Provable Understanding Through Spectral Analysis

        **Authors**: *Yiyou Sun, Zhenmei Shi, Yingyu Liang, Yixuan Li*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/sun23i.html](https://proceedings.mlr.press/v202/sun23i.html)

        **Abstract**:

        Novel Class Discovery (NCD) aims at inferring novel classes in an unlabeled set by leveraging prior knowledge from a labeled set with known classes. Despite its importance, there is a lack of theoretical foundations for NCD. This paper bridges the gap by providing an analytical framework to formalize and investigate when and how known classes can help discover novel classes. Tailored to the NCD problem, we introduce a graph-theoretic representation that can be learned by a novel NCD Spectral Contrastive Loss (NSCL). Minimizing this objective is equivalent to factorizing the graph’s adjacency matrix, which allows us to derive a provable error bound and provide the sufficient and necessary condition for NCD. Empirically, NSCL can match or outperform several strong baselines on common benchmark datasets, which is appealing for practical usage while enjoying theoretical guarantees.

        ----

        ## [1370] Learning Prescriptive ReLU Networks

        **Authors**: *Wei Sun, Asterios Tsiourvas*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/sun23j.html](https://proceedings.mlr.press/v202/sun23j.html)

        **Abstract**:

        We study the problem of learning optimal policy from a set of discrete treatment options using observational data. We propose a piecewise linear neural network model that can balance strong prescriptive performance and interpretability, which we refer to as the prescriptive ReLU network, or P-ReLU. We show analytically that this model (i) partitions the input space into disjoint polyhedra, where all instances that belong to the same partition receive the same treatment, and (ii) can be converted into an equivalent prescriptive tree with hyperplane splits for interpretability. We demonstrate the flexibility of the P-ReLU network as constraints can be easily incorporated with minor modifications to the architecture. Through experiments, we validate the superior prescriptive accuracy of P-ReLU against competing benchmarks. Lastly, we present examples of prescriptive trees extracted from trained P-ReLUs using a real-world dataset, for both the unconstrained and constrained scenarios.

        ----

        ## [1371] All in a Row: Compressed Convolution Networks for Graphs

        **Authors**: *Junshu Sun, Shuhui Wang, Xinzhe Han, Zhe Xue, Qingming Huang*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/sun23k.html](https://proceedings.mlr.press/v202/sun23k.html)

        **Abstract**:

        Compared to Euclidean convolution, existing graph convolution methods generally fail to learn diverse convolution operators under limited parameter scales and depend on additional treatments of multi-scale feature extraction. The challenges of generalizing Euclidean convolution to graphs arise from the irregular structure of graphs. To bridge the gap between Euclidean space and graph space, we propose a differentiable method for regularization on graphs that applies permutations to the input graphs. The permutations constrain all nodes in a row regardless of their input order and therefore enable the flexible generalization of Euclidean convolution. Based on the regularization of graphs, we propose Compressed Convolution Network (CoCN) for hierarchical graph representation learning. CoCN follows the local feature learning and global parameter sharing mechanisms of Convolution Neural Networks. The whole model can be trained end-to-end and is able to learn both individual node features and the corresponding structure features. We validate CoCN on several node classification and graph classification benchmarks. CoCN achieves superior performance over competitive convolutional GNNs and graph pooling models. Codes are available at https://github.com/sunjss/CoCN.

        ----

        ## [1372] Momentum Ensures Convergence of SIGNSGD under Weaker Assumptions

        **Authors**: *Tao Sun, Qingsong Wang, Dongsheng Li, Bao Wang*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/sun23l.html](https://proceedings.mlr.press/v202/sun23l.html)

        **Abstract**:

        Sign Stochastic Gradient Descent (signSGD) is a communication-efficient stochastic algorithm that only uses the sign information of the stochastic gradient to update the model’s weights. However, the existing convergence theory of signSGD either requires increasing batch sizes during training or assumes the gradient noise is symmetric and unimodal. Error feedback has been used to guarantee the convergence of signSGD under weaker assumptions at the cost of communication overhead. This paper revisits the convergence of signSGD and proves that momentum can remedy signSGD under weaker assumptions than previous techniques; in particular, our convergence theory does not require the assumption of bounded stochastic gradient or increased batch size. Our results resonate with echoes of previous empirical results where, unlike signSGD, signSGD with momentum maintains good performance even with small batch sizes. Another new result is that signSGD with momentum can achieve an improved convergence rate when the objective function is second-order smooth. We further extend our theory to signSGD with major vote and federated learning.

        ----

        ## [1373] A Critical Revisit of Adversarial Robustness in 3D Point Cloud Recognition with Diffusion-Driven Purification

        **Authors**: *Jiachen Sun, Jiongxiao Wang, Weili Nie, Zhiding Yu, Zhuoqing Mao, Chaowei Xiao*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/sun23m.html](https://proceedings.mlr.press/v202/sun23m.html)

        **Abstract**:

        3D point clouds serve as a crucial data representation in numerous real-world applications such as autonomous driving, robotics, and medical imaging. While the advancements in deep learning have spurred the utilization of 3D point clouds, deep models are notoriously vulnerable to adversarial attacks. Various defense solutions have been proposed to build robust models against adversarial attacks. In this work, we pinpoint a major limitation of the leading empirical defense, adversarial training, when applied to 3D point cloud models: gradient obfuscation, which significantly hampers robustness against potent attacks. To bridge the gap, we propose PointDP, a purification strategy that leverages diffusion models to defend against 3D adversarial attacks. Since PointDP does not rely on predefined adversarial examples for training, it can defend against a variety of threats. We conduct a comprehensive evaluation of PointDP across six representative 3D point cloud architectures, employing sixteen strong and adaptive attacks to manifest its foundational robustness. Our evaluation shows that PointDP achieves significantly better (i.e., 12.6%-40.3%) adversarial robustness than state-of-the-art methods under strong attacks bounded by different $\ell_p$ norms.

        ----

        ## [1374] SDDM: Score-Decomposed Diffusion Models on Manifolds for Unpaired Image-to-Image Translation

        **Authors**: *Shikun Sun, Longhui Wei, Junliang Xing, Jia Jia, Qi Tian*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/sun23n.html](https://proceedings.mlr.press/v202/sun23n.html)

        **Abstract**:

        Recent score-based diffusion models (SBDMs) show promising results in unpaired image-to-image translation (I2I). However, existing methods, either energy-based or statistically-based, provide no explicit form of the interfered intermediate generative distributions. This work presents a new score-decomposed diffusion model (SDDM) on manifolds to explicitly optimize the tangled distributions during image generation. SDDM derives manifolds to make the distributions of adjacent time steps separable and decompose the score function or energy guidance into an image "denoising" part and a content "refinement" part. To refine the image in the same noise level, we equalize the refinement parts of the score function and energy guidance, which permits multi-objective optimization on the manifold. We also leverage the block adaptive instance normalization module to construct manifolds with lower dimensions but still concentrated with the perturbed reference image. SDDM outperforms existing SBDM-based methods with much fewer diffusion steps on several I2I benchmarks.

        ----

        ## [1375] A Neural PDE Solver with Temporal Stencil Modeling

        **Authors**: *Zhiqing Sun, Yiming Yang, Shinjae Yoo*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/sun23o.html](https://proceedings.mlr.press/v202/sun23o.html)

        **Abstract**:

        Numerical simulation of non-linear partial differential equations plays a crucial role in modeling physical science and engineering phenomena, such as weather, climate, and aerodynamics. Recent Machine Learning (ML) models trained on low-resolution spatio-temporal signals have shown new promises in capturing important dynamics in high-resolution signals, under the condition that the models can effectively recover the missing details. However, this study shows that significant information is often lost in the low-resolution down-sampled features. To address such issues, we propose a new approach, namely Temporal Stencil Modeling (TSM), which combines the strengths of advanced time-series sequence modeling (with the HiPPO features) and state-of-the-art neural PDE solvers (with learnable stencil modeling). TSM aims to recover the lost information from the PDE trajectories and can be regarded as a temporal generalization of classic finite volume methods such as WENO. Our experimental results show that TSM achieves the new state-of-the-art simulation accuracy for 2-D incompressible Navier-Stokes turbulent flows: it significantly outperforms the previously reported best results by 19.9% in terms of the highly-correlated duration time, and reduces the inference latency into 80%. We also show a strong generalization ability of the proposed method to various out-of-distribution turbulent flow settings, as well as lower resolution or 1-D / 3-D settings. Our code is available at https://github.com/Edward-Sun/TSM-PDE .

        ----

        ## [1376] Feature Expansion for Graph Neural Networks

        **Authors**: *Jiaqi Sun, Lin Zhang, Guangyi Chen, Peng Xu, Kun Zhang, Yujiu Yang*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/sun23p.html](https://proceedings.mlr.press/v202/sun23p.html)

        **Abstract**:

        Graph neural networks aim to learn representations for graph-structured data and show impressive performance in node classification. Recently, many methods have studied the representations of GNNs from the perspective of optimization goals and spectral graph theory. However, the feature space that dominates representation learning has not been systematically studied in graph neural networks. In this paper, we propose to fill this gap by analyzing the feature space of both spatial and spectral models. We decompose graph neural networks into determined feature spaces and trainable weights, providing the convenience of studying the feature space explicitly using matrix space analysis. In particular, we find theoretically that the feature space tends to be linearly correlated due to repeated aggregations. In this case, the feature space is bounded by the poor representation of shared weights or the limited dimensionality of node attributes in existing models, leading to poor performance. Motivated by these findings, we propose 1) feature subspaces flattening and 2) structural principal components to expand the feature space. Extensive experiments verify the effectiveness of our proposed more comprehensive feature space, with comparable inference time to the baseline, and demonstrate its efficient convergence capability.

        ----

        ## [1377] Model-Bellman Inconsistency for Model-based Offline Reinforcement Learning

        **Authors**: *Yihao Sun, Jiaji Zhang, Chengxing Jia, Haoxin Lin, Junyin Ye, Yang Yu*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/sun23q.html](https://proceedings.mlr.press/v202/sun23q.html)

        **Abstract**:

        For offline reinforcement learning (RL), model-based methods are expected to be data-efficient as they incorporate dynamics models to generate more data. However, due to inevitable model errors, straightforwardly learning a policy in the model typically fails in the offline setting. Previous studies have incorporated conservatism to prevent out-of-distribution exploration. For example, MOPO penalizes rewards through uncertainty measures from predicting the next states, which we have discovered are loose bounds of the ideal uncertainty, i.e., the Bellman error. In this work, we propose MOdel-Bellman Inconsistency penalized offLinE Policy Optimization (MOBILE), a novel uncertainty-driven offline RL algorithm. MOBILE conducts uncertainty quantification through the inconsistency of Bellman estimations under an ensemble of learned dynamics models, which can be a better approximator to the true Bellman error, and penalizes the Bellman estimation based on this uncertainty. Empirically we have verified that our proposed uncertainty quantification can be significantly closer to the true Bellman error than the compared methods. Consequently, MOBILE outperforms prior offline RL approaches on most tasks of D4RL and NeoRL benchmarks.

        ----

        ## [1378] Inflow, Outflow, and Reciprocity in Machine Learning

        **Authors**: *Mukund Sundararajan, Walid Krichene*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/sundararajan23a.html](https://proceedings.mlr.press/v202/sundararajan23a.html)

        **Abstract**:

        Data is pooled across entities (individuals or enterprises) to create machine learning models, and sometimes, the entities that contribute the data also benefit from the models. Consider for instance a recommender system (e.g. Spotify, Instagram or YouTube), a health care app that predicts the risk for some disease, or a service built by pooling data across enterprises. In this work we propose a framework to study this value exchange, i.e., we model and measure contributions (outflows), benefits (inflows) and the balance between contributions and benefits (the degree of reciprocity). We show theoretically, and via experiments that under certain distributional assumptions, some classes of models are approximately reciprocal. These results only scratch the surface; we conclude with several open directions.

        ----

        ## [1379] When Personalization Harms Performance: Reconsidering the Use of Group Attributes in Prediction

        **Authors**: *Vinith Menon Suriyakumar, Marzyeh Ghassemi, Berk Ustun*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/suriyakumar23a.html](https://proceedings.mlr.press/v202/suriyakumar23a.html)

        **Abstract**:

        Machine learning models are often personalized with categorical attributes that define groups. In this work, we show that personalization with group attributes can inadvertently reduce performance at a group level – i.e., groups may receive unnecessarily inaccurate predictions by sharing their personal characteristics. We present formal conditions to ensure the fair use of group attributes in a prediction task, and describe how they can be checked by training one additional model. We characterize how fair use conditions be violated due to standard practices in model development, and study the prevalence of fair use violations in clinical prediction tasks. Our results show that personalization often fails to produce a tailored performance gain for every group who reports personal data, and underscore the need to evaluate fair use when personalizing models with characteristics that are protected, sensitive, self-reported, or costly to acquire.

        ----

        ## [1380] Tuning Computer Vision Models With Task Rewards

        **Authors**: *André Susano Pinto, Alexander Kolesnikov, Yuge Shi, Lucas Beyer, Xiaohua Zhai*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/susano-pinto23a.html](https://proceedings.mlr.press/v202/susano-pinto23a.html)

        **Abstract**:

        Misalignment between model predictions and intended usage can be detrimental for the deployment of computer vision models. The issue is exacerbated when the task involves complex structured outputs, as it becomes harder to design procedures which address this misalignment. In natural language processing, this is often addressed using reinforcement learning techniques that align models with a task reward. We adopt this approach and show its surprising effectiveness to improve generic models pretrained to imitate example outputs across multiple computer vision tasks, such as object detection, panoptic segmentation, colorization and image captioning. We believe this approach has the potential to be widely useful for better aligning models with a diverse range of computer vision tasks.

        ----

        ## [1381] Beyond Exponentially Fast Mixing in Average-Reward Reinforcement Learning via Multi-Level Monte Carlo Actor-Critic

        **Authors**: *Wesley A. Suttle, Amrit S. Bedi, Bhrij Patel, Brian M. Sadler, Alec Koppel, Dinesh Manocha*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/suttle23a.html](https://proceedings.mlr.press/v202/suttle23a.html)

        **Abstract**:

        Many existing reinforcement learning (RL) methods employ stochastic gradient iteration on the back end, whose stability hinges upon a hypothesis that the data-generating process mixes exponentially fast with a rate parameter that appears in the step-size selection. Unfortunately, this assumption is violated for large state spaces or settings with sparse rewards, and the mixing time is unknown, making the step size inoperable. In this work, we propose an RL methodology attuned to the mixing time by employing a multi-level Monte Carlo estimator for the critic, the actor, and the average reward embedded within an actor-critic (AC) algorithm. This method, which we call Multi-level Actor-Critic (MAC), is developed specifically for infinite-horizon average-reward settings and neither relies on oracle knowledge of the mixing time in its parameter selection nor assumes its exponential decay; it is therefore readily applicable to applications with slower mixing times. Nonetheless, it achieves a convergence rate comparable to SOTA actor-critic algorithms. We experimentally show that these alleviated restrictions on the technical conditions required for stability translate to superior performance in practice for RL problems with sparse rewards.

        ----

        ## [1382] Tight and fast generalization error bound of graph embedding in metric space

        **Authors**: *Atsushi Suzuki, Atsushi Nitanda, Taiji Suzuki, Jing Wang, Feng Tian, Kenji Yamanishi*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/suzuki23a.html](https://proceedings.mlr.press/v202/suzuki23a.html)

        **Abstract**:

        Recent studies have experimentally shown that we can achieve in non-Euclidean metric space effective and efficient graph embedding, which aims to obtain the vertices’ representations reflecting the graph’s structure in the metric space. Specifically, graph embedding in hyperbolic space has experimentally succeeded in embedding graphs with hierarchical-tree structure, e.g., data in natural languages, social networks, and knowledge bases. However, recent theoretical analyses have shown a much higher upper bound on non-Euclidean graph embedding’s generalization error than Euclidean one’s, where a high generalization error indicates that the incompleteness and noise in the data can significantly damage learning performance. It implies that the existing bound cannot guarantee the success of graph embedding in non-Euclidean metric space in a practical training data size, which can prevent non-Euclidean graph embedding’s application in real problems. This paper provides a novel upper bound of graph embedding’s generalization error by evaluating the local Rademacher complexity of the model as a function set of the distances of representation couples. Our bound clarifies that the performance of graph embedding in non-Euclidean metric space, including hyperbolic space, is better than the existing upper bounds suggest. Specifically, our new upper bound is polynomial in the metric space’s geometric radius $R$ and can be $O(\frac{1}{S})$ at the fastest, where $S$ is the training data size. Our bound is significantly tighter and faster than the existing one, which can be exponential to $R$ and $O(\frac{1}{\sqrt{S}})$ at the fastest. Specific calculations on example cases show that graph embedding in non-Euclidean metric space can outperform that in Euclidean space with much smaller training data than the existing bound has suggested.

        ----

        ## [1383] Proximal Causal Learning of Conditional Average Treatment Effects

        **Authors**: *Erik Sverdrup, Yifan Cui*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/sverdrup23a.html](https://proceedings.mlr.press/v202/sverdrup23a.html)

        **Abstract**:

        Efficiently and flexibly estimating treatment effect heterogeneity is an important task in a wide variety of settings ranging from medicine to marketing, and there are a considerable number of promising conditional average treatment effect estimators currently available. These, however, typically rely on the assumption that the measured covariates are enough to justify conditional exchangeability. We propose the P-learner, motivated by the R- and DR-learner, a tailored two-stage loss function for learning heterogeneous treatment effects in settings where exchangeability given observed covariates is an implausible assumption, and we wish to rely on proxy variables for causal inference. Our proposed estimator can be implemented by off-the-shelf loss-minimizing machine learning methods, which in the case of kernel regression satisfies an oracle bound on the estimated error as long as the nuisance components are estimated reasonably well.

        ----

        ## [1384] Inverse Reinforcement Learning without Reinforcement Learning

        **Authors**: *Gokul Swamy, David Wu, Sanjiban Choudhury, Drew Bagnell, Zhiwei Steven Wu*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/swamy23a.html](https://proceedings.mlr.press/v202/swamy23a.html)

        **Abstract**:

        Inverse Reinforcement Learning (IRL) is a powerful set of techniques for imitation learning that aims to learn a reward function that rationalizes expert demonstrations. Unfortunately, traditional IRL methods suffer from a computational weakness: they require repeatedly solving a hard reinforcement learning (RL) problem as a subroutine. This is counter-intuitive from the viewpoint of reductions: we have reduced the easier problem of imitation learning to repeatedly solving the harder problem of RL. Another thread of work has proved that access to the side-information of the distribution of states where a strong policy spends time can dramatically reduce the sample and computational complexities of solving an RL problem. In this work, we demonstrate for the first time a more informed imitation learning reduction where we utilize the state distribution of the expert to alleviate the global exploration component of the RL subroutine, providing an exponential speedup in theory. In practice, we find that we are able to significantly speed up the prior art on continuous control tasks.

        ----

        ## [1385] Von Mises Mixture Distributions for Molecular Conformation Generation

        **Authors**: *Kirk Swanson, Jake Lawrence Williams, Eric M. Jonas*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/swanson23a.html](https://proceedings.mlr.press/v202/swanson23a.html)

        **Abstract**:

        Molecules are frequently represented as graphs, but the underlying 3D molecular geometry (the locations of the atoms) ultimately determines most molecular properties. However, most molecules are not static and at room temperature adopt a wide variety of geometries or $\textit{conformations}$. The resulting distribution on geometries $p(x)$ is known as the Boltzmann distribution, and many molecular properties are expectations computed under this distribution. Generating accurate samples from the Boltzmann distribution is therefore essential for computing these expectations accurately. Traditional sampling-based methods are computationally expensive, and most recent machine learning-based methods have focused on identifying $\textit{modes}$ in this distribution rather than generating true $\textit{samples}$. Generating such samples requires capturing conformational variability, and it has been widely recognized that the majority of conformational variability in molecules arises from rotatable bonds. In this work, we present VonMisesNet, a new graph neural network that captures conformational variability via a variational approximation of rotatable bond torsion angles as a mixture of von Mises distributions. We demonstrate that VonMisesNet can generate conformations for arbitrary molecules in a way that is both physically accurate with respect to the Boltzmann distribution and orders of magnitude faster than existing sampling methods.

        ----

        ## [1386] Optimal randomized multilevel Monte Carlo for repeatedly nested expectations

        **Authors**: *Yasa Syed, Guanyang Wang*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/syed23a.html](https://proceedings.mlr.press/v202/syed23a.html)

        **Abstract**:

        The estimation of repeatedly nested expectations is a challenging task that arises in many real-world systems. However, existing methods generally suffer from high computational costs when the number of nestings becomes large. Fix any non-negative integer $D$ for the total number of nestings. Standard Monte Carlo methods typically cost at least $\mathcal{O}(\varepsilon^{-(2+D)})$ and sometimes $\mathcal {O}(\varepsilon^{-2(1+D)})$ to obtain an estimator up to $\varepsilon$-error. More advanced methods, such as multilevel Monte Carlo, currently only exist for $D = 1$. In this paper, we propose a novel Monte Carlo estimator called $\mathsf{READ}$, which stands for “Recursive Estimator for Arbitrary Depth.” Our estimator has an optimal computational cost of $\mathcal{O}(\varepsilon^{-2})$ for every fixed $D$ under suitable assumptions, and a nearly optimal computational cost of $\mathcal{O}(\varepsilon^{-2(1 + \delta)})$ for any $0 < \delta < \frac12$ under much more general assumptions. Our estimator is also unbiased, which makes it easy to parallelize. The key ingredients in our construction are an observation of the problem’s recursive structure and the recursive use of the randomized multilevel Monte Carlo method.

        ----

        ## [1387] Adaptive Coordination in Social Embodied Rearrangement

        **Authors**: *Andrew Szot, Unnat Jain, Dhruv Batra, Zsolt Kira, Ruta Desai, Akshara Rai*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/szot23a.html](https://proceedings.mlr.press/v202/szot23a.html)

        **Abstract**:

        We present the task of "Social Rearrangement", consisting of cooperative everyday tasks like setting up the dinner table, tidying a house or unpacking groceries in a simulated multi-agent environment. In Social Rearrangement, two robots coordinate to complete a long-horizon task, using onboard sensing and egocentric observations, and no privileged information about the environment. We study zero-shot coordination (ZSC) in this task, where an agent collaborates with a new partner, emulating a scenario where a robot collaborates with a new human partner. Prior ZSC approaches struggle to generalize in our complex and visually rich setting, and on further analysis, we find that they fail to generate diverse coordination behaviors at training time. To counter this, we propose Behavior Diversity Play (BDP), a novel ZSC approach that encourages diversity through a discriminability objective. Our results demonstrate that BDP learns adaptive agents that can tackle visual coordination, and zero-shot generalize to new partners in unseen environments, achieving 35% higher success and 32% higher efficiency compared to baselines.

        ----

        ## [1388] MG-GNN: Multigrid Graph Neural Networks for Learning Multilevel Domain Decomposition Methods

        **Authors**: *Ali Taghibakhshi, Nicolas Nytko, Tareq Uz Zaman, Scott P. MacLachlan, Luke N. Olson, Matthew West*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/taghibakhshi23a.html](https://proceedings.mlr.press/v202/taghibakhshi23a.html)

        **Abstract**:

        Domain decomposition methods (DDMs) are popular solvers for discretized systems of partial differential equations (PDEs), with one-level and multilevel variants. These solvers rely on several algorithmic and mathematical parameters, prescribing overlap, subdomain boundary conditions, and other properties of the DDM. While some work has been done on optimizing these parameters, it has mostly focused on the one-level setting or special cases such as structured-grid discretizations with regular subdomain construction. In this paper, we propose multigrid graph neural networks (MG-GNN), a novel GNN architecture for learning optimized parameters in two-level DDMs. We train MG-GNN using a new unsupervised loss function, enabling effective training on small problems that yields robust performance on unstructured grids that are orders of magnitude larger than those in the training set. We show that MG-GNN outperforms popular hierarchical graph network architectures for this optimization and that our proposed loss function is critical to achieving this improved performance.

        ----

        ## [1389] Learning Mixtures of Gaussians with Censored Data

        **Authors**: *Wai Ming Tai, Bryon Aragam*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/tai23a.html](https://proceedings.mlr.press/v202/tai23a.html)

        **Abstract**:

        We study the problem of learning mixtures of Gaussians with censored data. Statistical learning with censored data is a classical problem, with numerous practical applications, however, finite-sample guarantees for even simple latent variable models such as Gaussian mixtures are missing. Formally, we are given censored data from a mixture of univariate Gaussians $ \sum_{i=1}^k w_i \mathcal{N}(\mu_i,\sigma^2), $ i.e. the sample is observed only if it lies inside a set $S$. The goal is to learn the weights $w_i$ and the means $\mu_i$. We propose an algorithm that takes only $\frac{1}{\varepsilon^{O(k)}}$ samples to estimate the weights $w_i$ and the means $\mu_i$ within $\varepsilon$ error.

        ----

        ## [1390] Approximation and Estimation Ability of Transformers for Sequence-to-Sequence Functions with Infinite Dimensional Input

        **Authors**: *Shokichi Takakura, Taiji Suzuki*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/takakura23a.html](https://proceedings.mlr.press/v202/takakura23a.html)

        **Abstract**:

        Despite the great success of Transformer networks in various applications such as natural language processing and computer vision, their theoretical aspects are not well understood. In this paper, we study the approximation and estimation ability of Transformers as sequence-to-sequence functions with infinite dimensional inputs. Although inputs and outputs are both infinite dimensional, we show that when the target function has anisotropic smoothness, Transformers can avoid the curse of dimensionality due to their feature extraction ability and parameter sharing property. In addition, we show that even if the smoothness changes depending on each input, Transformers can estimate the importance of features for each input and extract important features dynamically. Then, we proved that Transformers achieve similar convergence rate as in the case of the fixed smoothness. Our theoretical results support the practical success of Transformers for high dimensional data.

        ----

        ## [1391] Learning Neural PDE Solvers with Parameter-Guided Channel Attention

        **Authors**: *Makoto Takamoto, Francesco Alesiani, Mathias Niepert*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/takamoto23a.html](https://proceedings.mlr.press/v202/takamoto23a.html)

        **Abstract**:

        Scientific Machine Learning (SciML) is concerned with the development of learned emulators of physical systems governed by partial differential equations (PDE). In application domains such as weather forecasting, molecular dynamics, and inverse design, ML-based surrogate models are increasingly used to augment or replace inefficient and often non-differentiable numerical simulation algorithms. While a number of ML-based methods for approximating the solutions of PDEs have been proposed in recent years, they typically do not adapt to the parameters of the PDEs, making it difficult to generalize to PDE parameters not seen during training. We propose a Channel Attention guided by PDE Parameter Embeddings (CAPE) component for neural surrogate models and a simple yet effective curriculum learning strategy. The CAPE module can be combined with any neural PDE solvers allowing them to adapt to unseen PDE parameters. The curriculum learning strategy provides a seamless transition between teacher-forcing and fully auto-regressive training. We compare CAPE in conjunction with the curriculum learning strategy using a PDE benchmark and obtain consistent and significant improvements over the baseline models. The experiments also show several advantages of CAPE, such as its increased ability to generalize to unseen PDE parameters without large increases inference time and parameter count. An implementation of the method and experiments are available at https://anonymous.4open.science/r/CAPE-ML4Sci-145B.

        ----

        ## [1392] Contextual Conservative Interleaving Bandits

        **Authors**: *Kei Takemura*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/takemura23a.html](https://proceedings.mlr.press/v202/takemura23a.html)

        **Abstract**:

        The performance of a bandit algorithm is usually measured by the cumulative rewards of the actions chosen by the algorithm. However, in many real-world applications, the rewards in each round should be good enough for reasons such as safety and fairness. In this paper, we investigate the contextual conservative interleaving bandit problem, which has a performance constraint that requires the chosen actions to be not much worse than given baseline actions in each round. This work is the first to simultaneously consider the following practical situations: (1) multiple actions are chosen in a round, (2) the feature vectors associated with given actions depend on the round, and (3) the performance constraints in each round that depend only on the actions chosen in that round. We propose a meta-algorithm, Greedy on Confidence Widths (GCW), that satisfies the performance constraints with high probability. GCW uses a standard bandit algorithm and achieves minimax optimal regret up to logarithmic factors if the algorithm used is also minimax optimal. We improve the existing analyses for the C${}^2$UCB algorithm and the Thompson sampling to combine with GCW. We show that these algorithms achieve near-optimal regret when the feasible sets of given actions are the bases of a matroid. Our numerical experiments on a real-world dataset demonstrate that GCW with the standard bandit algorithms efficiently improves performance while satisfying the performance constraints.

        ----

        ## [1393] Randomized Gaussian Process Upper Confidence Bound with Tighter Bayesian Regret Bounds

        **Authors**: *Shion Takeno, Yu Inatsu, Masayuki Karasuyama*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/takeno23a.html](https://proceedings.mlr.press/v202/takeno23a.html)

        **Abstract**:

        Gaussian process upper confidence bound (GP-UCB) is a theoretically promising approach for black-box optimization; however, the confidence parameter $\beta$ is considerably large in the theorem and chosen heuristically in practice. Then, randomized GP-UCB (RGP-UCB) uses a randomized confidence parameter, which follows the Gamma distribution, to mitigate the impact of manually specifying $\beta$. This study first generalizes the regret analysis of RGP-UCB to a wider class of distributions, including the Gamma distribution. Furthermore, we propose improved RGP-UCB (IRGP-UCB) based on a two-parameter exponential distribution, which achieves tighter Bayesian regret bounds. IRGP-UCB does not require an increase in the confidence parameter in terms of the number of iterations, which avoids over-exploration in the later iterations. Finally, we demonstrate the effectiveness of IRGP-UCB through extensive experiments.

        ----

        ## [1394] Towards Practical Preferential Bayesian Optimization with Skew Gaussian Processes

        **Authors**: *Shion Takeno, Masahiro Nomura, Masayuki Karasuyama*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/takeno23b.html](https://proceedings.mlr.press/v202/takeno23b.html)

        **Abstract**:

        We study preferential Bayesian optimization (BO) where reliable feedback is limited to pairwise comparison called duels. An important challenge in preferential BO, which uses the preferential Gaussian process (GP) model to represent flexible preference structure, is that the posterior distribution is a computationally intractable skew GP. The most widely used approach for preferential BO is Gaussian approximation, which ignores the skewness of the true posterior. Alternatively, Markov chain Monte Carlo (MCMC) based preferential BO is also proposed. In this work, we first verify the accuracy of Gaussian approximation, from which we reveal the critical problem that the predictive probability of duels can be inaccurate. This observation motivates us to improve the MCMC-based estimation for skew GP, for which we show the practical efficiency of Gibbs sampling and derive the low variance MC estimator. However, the computational time of MCMC can still be a bottleneck in practice. Towards building a more practical preferential BO, we develop a new method that achieves both high computational efficiency and low sample complexity, and then demonstrate its effectiveness through extensive numerical experiments.

        ----

        ## [1395] Robust Explanation for Free or At the Cost of Faithfulness

        **Authors**: *Zeren Tan, Yang Tian*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/tan23a.html](https://proceedings.mlr.press/v202/tan23a.html)

        **Abstract**:

        Devoted to interpreting the explicit behaviors of machine learning models, explanation methods can identify implicit characteristics of models to improve trustworthiness. However, explanation methods are shown as vulnerable to adversarial perturbations, implying security concerns in high-stakes domains. In this paper, we investigate when robust explanations are necessary and what they cost. We prove that the robustness of explanations is determined by the robustness of the model to be explained. Therefore, we can have robust explanations for free for a robust model. To have robust explanations for a non-robust model, composing the original model with a kernel is proved as an effective way that returns strictly more robust explanations. Nevertheless, we argue that this also incurs a robustness-faithfulness trade-off, i.e., contrary to common expectations, an explanation method may also become less faithful when it becomes more robust. This argument holds for any model. We are the first to introduce this trade-off and theoretically prove its existence for SmoothGrad. Theoretical findings are verified by empirical evidence on six state-of-the-art explanation methods and four backbones.

        ----

        ## [1396] Provably Invariant Learning without Domain Information

        **Authors**: *Xiaoyu Tan, Lin Yong, Shengyu Zhu, Chao Qu, Xihe Qiu, Yinghui Xu, Peng Cui, Yuan Qi*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/tan23b.html](https://proceedings.mlr.press/v202/tan23b.html)

        **Abstract**:

        Typical machine learning applications always assume the data follows independent and identically distributed (IID) assumptions. In contrast, this assumption is frequently violated in real-world circumstances, leading to the Out-of-Distribution (OOD) generalization problem and a major drop in model robustness. To mitigate this issue, the invariant learning technique is leveraged to distinguish between spurious features and invariant features among all input features and to train the model purely on the basis of the invariant features. Numerous invariant learning strategies imply that the training data should contain domain information. Such information includes the environment index or auxiliary information acquired from prior knowledge. However, acquiring these information is typically impossible in practice. In this study, we present TIVA for environment-independent invariance learning, which requires no environment-specific information in training data. We discover and prove that, given certain mild data conditions, it is possible to train an environment partitioning policy based on attributes that are independent of the targets and then conduct invariant risk minimization. We examine our method in comparison to other baseline methods, which demonstrate superior performance and excellent robustness under OOD, using multiple benchmarks.

        ----

        ## [1397] Auto-Differentiation of Relational Computations for Very Large Scale Machine Learning

        **Authors**: *Yuxin Tang, Zhimin Ding, Dimitrije Jankov, Binhang Yuan, Daniel Bourgeois, Chris Jermaine*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/tang23a.html](https://proceedings.mlr.press/v202/tang23a.html)

        **Abstract**:

        The relational data model was designed to facilitate large-scale data management and analytics. We consider the problem of how to differentiate computations expressed relationally. We show experimentally that a relational engine running an auto-differentiated relational algorithm can easily scale to very large datasets, and is competitive with state-of-the-art, special-purpose systems for large-scale distributed machine learning.

        ----

        ## [1398] Regret-Minimizing Double Oracle for Extensive-Form Games

        **Authors**: *Xiaohang Tang, Le Cong Dinh, Stephen Marcus McAleer, Yaodong Yang*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/tang23b.html](https://proceedings.mlr.press/v202/tang23b.html)

        **Abstract**:

        By incorporating regret minimization, double oracle methods have demonstrated rapid convergence to Nash Equilibrium (NE) in normal-form games and extensive-form games, through algorithms such as online double oracle (ODO) and extensive-form double oracle (XDO), respectively. In this study, we further examine the theoretical convergence rate and sample complexity of such regret minimization-based double oracle methods, utilizing a unified framework called Regret-Minimizing Double Oracle. Based on this framework, we extend ODO to extensive-form games and determine its sample complexity. Moreover, we demonstrate that the sample complexity of XDO can be exponential in the number of information sets $|S|$, owing to the exponentially decaying stopping threshold of restricted games. To solve this problem, we propose the Periodic Double Oracle (PDO) method, which has the lowest sample complexity among regret minimization-based double oracle methods, being only polynomial in $|S|$. Empirical evaluations on multiple poker and board games show that PDO achieves significantly faster convergence than previous double oracle algorithms and reaches a competitive level with state-of-the-art regret minimization methods.

        ----

        ## [1399] From Perception to Programs: Regularize, Overparameterize, and Amortize

        **Authors**: *Hao Tang, Kevin Ellis*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/tang23c.html](https://proceedings.mlr.press/v202/tang23c.html)

        **Abstract**:

        We develop techniques for synthesizing neurosymbolic programs. Such programs mix discrete symbolic processing with continuous neural computation. We relax this mixed discrete/continuous problem and jointly learn all modules with gradient descent, and also incorporate amortized inference, overparameterization, and a differentiable strategy for penalizing lengthy programs. Collectedly this toolbox improves the stability of gradient-guided program search, and suggests ways of learning both how to parse continuous input into discrete abstractions, and how to process those abstractions via symbolic code.

        ----

        

[Go to the previous page](ICML-2023-list06.md)

[Go to the next page](ICML-2023-list08.md)

[Go to the catalog section](README.md)