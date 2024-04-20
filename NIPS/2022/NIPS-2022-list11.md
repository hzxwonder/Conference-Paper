## [2000] QUARK: Controllable Text Generation with Reinforced Unlearning

        **Authors**: *Ximing Lu, Sean Welleck, Jack Hessel, Liwei Jiang, Lianhui Qin, Peter West, Prithviraj Ammanabrolu, Yejin Choi*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/b125999bde7e80910cbdbd323087df8f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/b125999bde7e80910cbdbd323087df8f-Abstract-Conference.html)

        **Abstract**:

        Large-scale language models often learn behaviors that are misaligned with user expectations. Generated text may contain offensive or toxic language, contain significant repetition, or be of a different sentiment than desired by the user. We consider the task of unlearning these misalignments by fine-tuning the language model on signals of what not to do. We introduce Quantized Reward Konditioning (Quark), an algorithm for optimizing a reward function that quantifies an (un)wanted property, while not straying too far from the original model. Quark alternates between (i) collecting samples with the current language model, (ii) sorting them into quantiles based on reward, with each quantile identified by a reward token prepended to the language modelâ€™s input, and (iii) using a standard language modeling loss on samples from each quantile conditioned on its reward token, while remaining nearby the original language model via a KL-divergence penalty. By conditioning on a high-reward token at generation time, the model generates text that exhibits less of the unwanted property. For unlearning toxicity, negative sentiment, and repetition, our experiments show that Quark outperforms both strong baselines and state-of-the-art reinforcement learning methods like PPO, while relying only on standard language modeling primitives.

        ----

        ## [2001] Safe Opponent-Exploitation Subgame Refinement

        **Authors**: *Mingyang Liu, Chengjie Wu, Qihan Liu, Yansen Jing, Jun Yang, Pingzhong Tang, Chongjie Zhang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/b12a1d1014e952e676f5d6931d03241a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/b12a1d1014e952e676f5d6931d03241a-Abstract-Conference.html)

        **Abstract**:

        In zero-sum games, an NE strategy tends to be overly conservative confronted with opponents of limited rationality, because it does not actively exploit their weaknesses. From another perspective, best responding to an estimated opponent model is vulnerable to estimation errors and lacks safety guarantees. Inspired by the recent success of real-time search algorithms in developing superhuman AI, we investigate the dilemma of safety and opponent exploitation and present a novel real-time search framework, called Safe Exploitation Search (SES), which continuously interpolates between the two extremes of online strategy refinement. We provide SES with a theoretically upper-bounded exploitability and a lower-bounded evaluation performance. Additionally, SES enables computationally efficient online adaptation to a possibly updating opponent model, while previous safe exploitation methods have to recompute for the whole game. Empirical results show that SES significantly outperforms NE baselines and previous algorithms while keeping exploitability low at the same time.

        ----

        ## [2002] Parameter-free Dynamic Graph Embedding for Link Prediction

        **Authors**: *Jiahao Liu, Dongsheng Li, Hansu Gu, Tun Lu, Peng Zhang, Ning Gu*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/b14d7175755b180dc2163e15e3110cb6-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/b14d7175755b180dc2163e15e3110cb6-Abstract-Conference.html)

        **Abstract**:

        Dynamic interaction graphs have been widely adopted to model the evolution of user-item interactions over time. There are two crucial factors when modelling user preferences for link prediction in dynamic interaction graphs: 1) collaborative relationship among users and 2) user personalized interaction patterns. Existing methods often implicitly consider these two factors together, which may lead to noisy user modelling when the two factors diverge. In addition, they usually require time-consuming parameter learning with back-propagation, which is prohibitive for real-time user preference modelling. To this end, this paper proposes FreeGEM, a parameter-free dynamic graph embedding method for link prediction. Firstly, to take advantage of the collaborative relationships, we propose an incremental graph embedding engine to obtain user/item embeddings, which is an Online-Monitor-Offline architecture consisting of an Online module to approximately embed users/items over time, a Monitor module to estimate the approximation error in real time and an Offline module to calibrate the user/item embeddings when the online approximation errors exceed a threshold. Meanwhile, we integrate attribute information into the model, which enables FreeGEM to better model users belonging to some under represented groups. Secondly, we design a personalized dynamic interaction pattern modeller, which combines dynamic time decay with attention mechanism to model user short-term interests. Experimental results on two link prediction tasks show that FreeGEM can outperform the state-of-the-art methods in accuracy while achieving over 36X improvement in efficiency. All code and datasets can be found in https://github.com/FudanCISL/FreeGEM.

        ----

        ## [2003] Towards a Unified Framework for Uncertainty-aware Nonlinear Variable Selection with Theoretical Guarantees

        **Authors**: *Wenying Deng, Beau Coker, Rajarshi Mukherjee, Jeremiah Z. Liu, Brent A. Coull*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/b153f11554e8f7926189e3f21185f00f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/b153f11554e8f7926189e3f21185f00f-Abstract-Conference.html)

        **Abstract**:

        We develop a simple and unified framework for nonlinear variable importance estimation that incorporates uncertainty in the prediction function and is compatible with a wide range of machine learning models (e.g., tree ensembles, kernel methods, neural networks, etc). In particular, for a learned nonlinear model $f(\mathbf{x})$, we consider quantifying the importance of an input variable $\mathbf{x}^j$ using the integrated partial derivative $\Psi_j = \Vert \frac{\partial}{\partial \mathbf{x}^j} f(\mathbf{x})\Vert^2_{P_\mathcal{X}}$. We then (1) provide a principled approach for quantifying uncertainty in variable importance by deriving its posterior distribution, and (2) show that the approach is generalizable even to non-differentiable models such as tree ensembles. Rigorous Bayesian nonparametric theorems are derived to guarantee the posterior consistency and asymptotic uncertainty of the proposed approach. Extensive  simulations and experiments on healthcare benchmark datasets confirm that the proposed algorithm outperforms existing classical and recent variable selection methods.

        ----

        ## [2004] Non-Markovian Reward Modelling from Trajectory Labels via Interpretable Multiple Instance Learning

        **Authors**: *Joseph Early, Tom Bewley, Christine Evers, Sarvapali D. Ramchurn*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/b157cfde6794e93b2353b9712bbd45a5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/b157cfde6794e93b2353b9712bbd45a5-Abstract-Conference.html)

        **Abstract**:

        We generalise the problem of reward modelling (RM) for reinforcement learning (RL) to handle non-Markovian rewards. Existing work assumes that human evaluators observe each step in a trajectory independently when providing feedback on agent behaviour. In this work, we remove this assumption, extending RM to capture temporal dependencies in human assessment of trajectories. We show how RM can be approached as a multiple instance learning (MIL) problem, where trajectories are treated as bags with return labels, and steps within the trajectories are instances with unseen reward labels. We go on to develop new MIL models that are able to capture the time dependencies in labelled trajectories. We demonstrate on a range of RL tasks that our novel MIL models can reconstruct reward functions to a high level of accuracy, and can be used to train high-performing agent policies.

        ----

        ## [2005] Explaining Preferences with Shapley Values

        **Authors**: *Robert Hu, Siu Lun Chau, Jaime Ferrando Huertas, Dino Sejdinovic*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/b1656d20067ca7c84a33785c4083a75e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/b1656d20067ca7c84a33785c4083a75e-Abstract-Conference.html)

        **Abstract**:

        While preference modelling is becoming one of the pillars of machine learning, the problem of preference explanation remains challenging and underexplored. In this paper, we propose \textsc{Pref-SHAP}, a Shapley value-based model explanation framework for pairwise comparison data. We derive the appropriate value functions for preference models and further extend the framework to model and explain \emph{context specific} information, such as the surface type in a tennis game. To demonstrate the utility of \textsc{Pref-SHAP}, we apply our method to a variety of synthetic and real-world datasets and show that richer and more insightful explanations can be obtained over the baseline.

        ----

        ## [2006] Structuring Uncertainty for Fine-Grained Sampling in Stochastic Segmentation Networks

        **Authors**: *Frank Nussbaum, Jakob Gawlikowski, Julia Niebling*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/b1a77a501bf32f8c7348fe39da2cf8c6-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/b1a77a501bf32f8c7348fe39da2cf8c6-Abstract-Conference.html)

        **Abstract**:

        In image segmentation, the classic approach of learning a deterministic segmentation neither accounts for noise and ambiguity in the data nor for expert disagreements about the correct segmentation. This has been addressed by architectures that predict heteroscedastic (input-dependent) segmentation uncertainty, which indicates regions of segmentations that should be treated with care. What is missing are structural insights into the uncertainty, which would be desirable for interpretability and systematic adjustments. In the context of state-of-the-art stochastic segmentation networks (SSNs), we solve this issue by dismantling the overall predicted uncertainty into smaller uncertainty components. We obtain them directly from the low-rank Gaussian distribution for the logits in the network head of SSNs, based on a previously unconsidered view of this distribution as a factor model. The rank subsequently encodes a number of latent variables, each of which controls an individual uncertainty component. Hence, we can use the latent variables (called factors) for fine-grained sample control, thereby solving an open problem from previous work. There is one caveat though--factors are only unique up to orthogonal rotations. Factor rotations allow us to structure the uncertainty in a way that endorses simplicity, non-redundancy, and separation among the individual uncertainty components. To make the overall and factor-specific uncertainties at play comprehensible, we introduce flow probabilities that quantify deviations from the mean prediction and can also be used for uncertainty visualization. We show on medical-imaging, earth-observation, and traffic-scene data that rotation criteria based on factor-specific flow probabilities consistently yield the best factors for fine-grained sampling.

        ----

        ## [2007] Fair Bayes-Optimal Classifiers Under Predictive Parity

        **Authors**: *Xianli Zeng, Edgar Dobriban, Guang Cheng*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/b1d9c7e7bd265d81aae8d74a7a6bd7f1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/b1d9c7e7bd265d81aae8d74a7a6bd7f1-Abstract-Conference.html)

        **Abstract**:

        Increasing concerns about disparate effects of AI have motivated a great deal of work on fair machine learning. Existing works mainly focus on independence- and separation-based measures (e.g., demographic parity, equality of opportunity, equalized odds), while sufficiency-based measures such as predictive parity are much less studied. This paper considers predictive parity, which requires equalizing the probability of success given a positive prediction among different protected groups. We prove that, if the overall performances of different groups vary only moderately, all fair Bayes-optimal classifiers under predictive parity are group-wise thresholding rules. Perhaps surprisingly, this may not hold if group performance levels vary widely; in this case, we find that predictive parity among protected groups may lead to within-group unfairness. We then propose an algorithm we call FairBayes-DPP, aiming to ensure predictive parity when our condition is satisfied. FairBayes-DPP is an adaptive thresholding algorithm that aims to achieve predictive parity, while also seeking to maximize test accuracy. We provide supporting experiments conducted on synthetic and empirical data.

        ----

        ## [2008] Pre-Train Your Loss: Easy Bayesian Transfer Learning with Informative Priors

        **Authors**: *Ravid Shwartz-Ziv, Micah Goldblum, Hossein Souri, Sanyam Kapoor, Chen Zhu, Yann LeCun, Andrew Gordon Wilson*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/b1e7f61f40d68b2177857bfcb195a507-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/b1e7f61f40d68b2177857bfcb195a507-Abstract-Conference.html)

        **Abstract**:

        Deep learning is increasingly moving towards a transfer learning paradigm whereby large foundation models are fine-tuned on downstream tasks, starting from an initialization learned on the source task. But an initialization contains relatively little information about the source task, and does not reflect the belief that our knowledge of the source task should affect the locations and shape of optima on the downstream task.Instead, we show that we can learn highly informative posteriors from the source task, through supervised or self-supervised approaches, which then serve as the basis for priors that modify the whole loss surface on the downstream task. This simple modular approach enables significant performance gains and more data-efficient learning on a variety of downstream classification and segmentation tasks, serving as a drop-in replacement for standard pre-training strategies. These highly informative priors also can be saved for future use, similar to pre-trained weights, and stand in contrast to the zero-mean isotropic uninformative priors that are typically used in Bayesian deep learning.

        ----

        ## [2009] Unsupervised Adaptation from Repeated Traversals for Autonomous Driving

        **Authors**: *Yurong You, Cheng Perng Phoo, Katie Luo, Travis Zhang, Wei-Lun Chao, Bharath Hariharan, Mark Campbell, Kilian Q. Weinberger*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/b1eb88348ee19a33c81cf5bc3fb8e9d2-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/b1eb88348ee19a33c81cf5bc3fb8e9d2-Abstract-Conference.html)

        **Abstract**:

        For a self-driving car to operate reliably, its perceptual system must generalize to the end-user's environment --- ideally without additional annotation efforts. One potential solution is to leverage unlabeled data (e.g., unlabeled LiDAR point clouds) collected from the end-users' environments (i.e. target domain) to adapt the system to the difference between training and testing environments. While extensive research has been done on such an unsupervised domain adaptation problem, one fundamental problem lingers: there is no reliable signal in the target domain to supervise the adaptation process. To overcome this issue we observe that it is easy to collect unsupervised data from multiple traversals of repeated routes. While different from conventional unsupervised domain adaptation, this assumption is extremely realistic since many drivers share the same roads. We show that this simple additional assumption is sufficient to obtain a potent signal that allows us to perform iterative self-training of 3D object detectors on the target domain. Concretely, we generate pseudo-labels with the out-of-domain detector but reduce false positives by removing detections of supposedly mobile objects that are persistent across traversals. Further, we reduce false negatives by encouraging predictions in regions that are not persistent. We experiment with our approach on two large-scale driving datasets and show remarkable improvement in 3D object detection of cars, pedestrians, and cyclists, bringing us a step closer to generalizable autonomous driving.

        ----

        ## [2010] Training language models to follow instructions with human feedback

        **Authors**: *Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul F. Christiano, Jan Leike, Ryan Lowe*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/b1efde53be364a73914f58805a001731-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/b1efde53be364a73914f58805a001731-Abstract-Conference.html)

        **Abstract**:

        Making language models bigger does not inherently make them better at following a user's intent. For example, large language models can generate outputs that are untruthful, toxic, or simply not helpful to the user. In other words, these models are not aligned with their users. In this paper, we show an avenue for aligning language models with user intent on a wide range of tasks by fine-tuning with human feedback. Starting with a set of labeler-written prompts and prompts submitted through a language model API, we collect a dataset of labeler demonstrations of the desired model behavior, which we use to fine-tune GPT-3 using supervised learning. We then collect a dataset of rankings of model outputs, which we use to further fine-tune this supervised model using reinforcement learning from human feedback. We call the resulting models InstructGPT. In human evaluations on our prompt distribution, outputs from the 1.3B parameter InstructGPT model are preferred to outputs from the 175B GPT-3, despite having 100x fewer parameters. Moreover, InstructGPT models show improvements in truthfulness and reductions in toxic output generation while having minimal performance regressions on public NLP datasets. Even though InstructGPT still makes simple mistakes, our results show that fine-tuning with human feedback is a promising direction for aligning language models with human intent.

        ----

        ## [2011] Byzantine Spectral Ranking

        **Authors**: *Arnhav Datar, Arun Rajkumar, John Augustine*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/b1f7288854d3bd476c17725c2d85967f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/b1f7288854d3bd476c17725c2d85967f-Abstract-Conference.html)

        **Abstract**:

        We study the problem of rank aggregation where the goal is to obtain a global ranking by aggregating pair-wise comparisons of voters over a set of items. We consider an adversarial setting where the voters are partitioned into two sets. The first set votes in a stochastic manner according to the popular score-based Bradley-Terry-Luce (BTL) model for pairwise comparisons. The second set comprises malicious Byzantine voters trying to deteriorate the ranking. We consider a strongly-adversarial scenario where the Byzantine voters know the BTL scores, the votes of the good voters, the algorithm, and can collude with each other. We first show that the popular spectral ranking based Rank-Centrality algorithm, though optimal for the BTL model, does not perform well even when a small constant fraction of the voters are Byzantine.We introduce the Byzantine Spectral Ranking Algorithm (and a faster variant of it), which produces a reliable ranking when the number of good voters exceeds the number of Byzantine voters. We show that no algorithm can produce a satisfactory ranking with probability > 1/2 for all BTL weights when there are more Byzantine voters than good voters, showing that our algorithm works for all possible population fractions. We support our theoretical results with experimental results on synthetic and real datasets to demonstrate the failure of the Rank-Centrality algorithm under several adversarial scenarios and how the proposed Byzantine Spectral Ranking algorithm is robust in obtaining good rankings.

        ----

        ## [2012] Non-rigid Point Cloud Registration with Neural Deformation Pyramid

        **Authors**: *Yang Li, Tatsuya Harada*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/b2077e6d66da612fcb701589efa9ce88-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/b2077e6d66da612fcb701589efa9ce88-Abstract-Conference.html)

        **Abstract**:

        Non-rigid point cloud registration is a key component in many computer vision and computer graphics applications. The high complexity of the unknown non-rigid motion make this task a challenging problem. In this paper, we break down this problem via hierarchical motion decomposition. Our method called Neural Deformation Pyramid (NDP) represents non-rigid motion using a pyramid architecture. Each pyramid level, denoted by a Multi-Layer Perception (MLP), takes as input a sinusoidally encoded 3D point and outputs its motion increments from the previous level. The sinusoidal function starts with a low input frequency and gradually increases when the pyramid level goes down. This allows a multi-level rigid to nonrigid motion decomposition and also speeds up the solving by Ã—50 times compared to the existing MLP-based approach. Our method achieves advanced partial-to-partial non-rigid point cloud registration results on the 4DMatch/4DLoMatchbenchmark under both no-learned and supervised settings.

        ----

        ## [2013] Provably tuning the ElasticNet across instances

        **Authors**: *Maria-Florina Balcan, Misha Khodak, Dravyansh Sharma, Ameet Talwalkar*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/b21a34c4e8dba253f05f4a5adc68ba73-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/b21a34c4e8dba253f05f4a5adc68ba73-Abstract-Conference.html)

        **Abstract**:

        An important unresolved challenge in the theory of regularization is to set the regularization coefficients of popular techniques like the ElasticNet with general provable guarantees. We consider the problem of tuning the regularization parameters of Ridge regression, LASSO, and the ElasticNet across multiple problem instances, a setting that encompasses both cross-validation and multi-task hyperparameter optimization. We obtain a novel structural result for the ElasticNet which characterizes the loss as a function of the tuning parameters as a piecewise-rational function with algebraic boundaries. We use this to bound the structural complexity of the regularized loss functions and show generalization guarantees for tuning the ElasticNet regression coefficients in the statistical setting. We also consider the more challenging online learning setting, where we show vanishing average expected regret relative to the optimal parameter pair. We further extend our results to tuning classification algorithms obtained by thresholding regression fits regularized by Ridge, LASSO, or ElasticNet. Our results are the first general learning-theoretic guarantees for this important class of problems that avoid strong assumptions on the data distribution. Furthermore, our guarantees hold for both validation and popular information criterion objectives.

        ----

        ## [2014] PlasticityNet: Learning to Simulate Metal, Sand, and Snow for Optimization Time Integration

        **Authors**: *Xuan Li, Yadi Cao, Minchen Li, Yin Yang, Craig A. Schroeder, Chenfanfu Jiang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/b235f0b417e8bf270c0cb19fe0b82c1e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/b235f0b417e8bf270c0cb19fe0b82c1e-Abstract-Conference.html)

        **Abstract**:

        In this paper, we propose a neural network-based approach for learning to represent the behavior of plastic solid materials ranging from rubber and metal to sand and snow. Unlike elastic forces such as spring forces, these plastic forces do not result from the positional gradient of any potential energy, imposing great challenges on the stability and flexibility of their simulation. Our method effectively resolves this issue by learning a generalizable plastic energy whose derivative closely matches the analytical behavior of plastic forces. Our method, for the first time, enables the simulation of a wide range of arbitrary elasticity-plasticity combinations using time step-independent, unconditionally stable optimization-based time integrators. We demonstrate the efficacy of our method by learning and producing challenging 2D and 3D effects of metal, sand, and snow with complex dynamics.

        ----

        ## [2015] Synthetic Model Combination: An Instance-wise Approach to Unsupervised Ensemble Learning

        **Authors**: *Alex J. Chan, Mihaela van der Schaar*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/b24426d44bfbef35e24812c996752ceb-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/b24426d44bfbef35e24812c996752ceb-Abstract-Conference.html)

        **Abstract**:

        Consider making a prediction over new test data without any opportunity to learn from a training set of labelled data - instead given access to a set of expert models and their predictions alongside some limited information about the dataset used to train them. In scenarios from finance to the medical sciences, and even consumer practice, stakeholders have developed models on private data they either cannot, or do not want to, share. Given the value and legislation surrounding personal information, it is not surprising that only the models, and not the data, will be released - the pertinent question becoming: how best to use these models? Previous work has focused on global model selection or ensembling, with the result of a single final model across the feature space. Machine learning models perform notoriously poorly on data outside their training domain however, and so we argue that when ensembling models the weightings for individual instances must reflect their respective domains - in other words models that are more likely to have seen information on that instance should have more attention paid to them. We introduce a method for such an instance-wise ensembling of models, including a novel representation learning step for handling sparse high-dimensional domains. Finally, we demonstrate the need and generalisability of our method on classical machine learning tasks as well as highlighting a real world use case in the pharmacological setting of vancomycin precision dosing.

        ----

        ## [2016] Power and limitations of single-qubit native quantum neural networks

        **Authors**: *Zhan Yu, Hongshun Yao, Mujin Li, Xin Wang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/b250de41980b58d34d6aadc3f4aedd4c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/b250de41980b58d34d6aadc3f4aedd4c-Abstract-Conference.html)

        **Abstract**:

        Quantum neural networks (QNNs) have emerged as a leading strategy to establish applications in machine learning, chemistry, and optimization. While the applications of QNN have been widely investigated, its theoretical foundation remains less understood. In this paper, we formulate a theoretical framework for the expressive ability of data re-uploading quantum neural networks that consist of interleaved encoding circuit blocks and trainable circuit blocks. First, we prove that single-qubit quantum neural networks can approximate any univariate function by mapping the model to a partial Fourier series. We in particular establish the exact correlations between the parameters of the trainable gates and the Fourier coefficients, resolving an open problem on the universal approximation property of QNN. Second, we discuss the limitations of single-qubit native QNNs on approximating multivariate functions by analyzing the frequency spectrum and the flexibility of Fourier coefficients. We further demonstrate the expressivity and limitations of single-qubit native QNNs via numerical experiments. We believe these results would improve our understanding of QNNs and provide a helpful guideline for designing powerful QNNs for machine learning tasks.

        ----

        ## [2017] Hedging as Reward Augmentation in Probabilistic Graphical Models

        **Authors**: *Debarun Bhattacharjya, Radu Marinescu*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/b2647998bc953781490049fe2ac28bf0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/b2647998bc953781490049fe2ac28bf0-Abstract-Conference.html)

        **Abstract**:

        Most people associate the term `hedging' exclusively with financial applications, particularly the use of financial derivatives. We argue that hedging is an activity that human and machine agents should engage in more broadly, even when the agent's value is not necessarily in monetary units. In this paper, we propose a decision-theoretic view of hedging based on augmenting a probabilistic graphical model -- specifically a Bayesian network or an influence diagram -- with a reward. Hedging is therefore posed as a particular kind of graph manipulation, and can be viewed as analogous to control/intervention and information gathering related analysis. Effective hedging occurs when a risk-averse agent finds opportunity to balance uncertain rewards in their current situation. We illustrate the concepts with examples and counter-examples, and conduct experiments to demonstrate the properties and applicability of the proposed computational tools that enable agents to proactively identify potential hedging opportunities in real-world situations.

        ----

        ## [2018] Debiased, Longitudinal and Coordinated Drug Recommendation through Multi-Visit Clinic Records

        **Authors**: *Hongda Sun, Shufang Xie, Shuqi Li, Yuhan Chen, Ji-Rong Wen, Rui Yan*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/b295b3a940706f431076c86b78907757-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/b295b3a940706f431076c86b78907757-Abstract-Conference.html)

        **Abstract**:

        AI-empowered drug recommendation has become an important task in healthcare research areas, which offers an additional perspective to assist human doctors with more accurate and more efficient drug prescriptions. Generally, drug recommendation is based on patients' diagnosis results in the electronic health records. We assume that there are three key factors to be addressed in drug recommendation: 1) elimination of recommendation bias due to limitations of observable information, 2) better utilization of historical health condition and 3) coordination of multiple drugs to control safety. To this end, we propose DrugRec, a causal inference based drug recommendation model. The causal graphical model can identify and deconfound the recommendation bias with front-door adjustment. Meanwhile, we model the multi-visit in the causal graph to characterize a patient's historical health conditions. Finally, we model the drug-drug interactions (DDIs) as the propositional satisfiability (SAT) problem, and solving the SAT problem can help better coordinate the recommendation. Comprehensive experiment results show that our proposed model achieves state-of-the-art performance on the widely used datasets MIMIC-III and MIMIC-IV, demonstrating the effectiveness and safety of our method.

        ----

        ## [2019] Disentangling Causal Effects from Sets of Interventions in the Presence of Unobserved Confounders

        **Authors**: *Olivier Jeunen, Ciarán M. Gilligan-Lee, Rishabh Mehrotra, Mounia Lalmas*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/b29ec434e049fb96f3c4245a405ee976-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/b29ec434e049fb96f3c4245a405ee976-Abstract-Conference.html)

        **Abstract**:

        The ability to answer causal questions is crucial in many domains, as causal inference allows one to understand the impact of interventions. In many applications, only a single intervention is possible at a given time. However, in some important areas, multiple interventions are concurrently applied. Disentangling the effects of single interventions from jointly applied interventions is a challenging task---especially as simultaneously applied interventions can interact. This problem is made harder still by unobserved confounders, which influence both treatments and outcome. We address this challenge by aiming to learn the  effect of a single-intervention  from  both observational data and sets of interventions. We prove that this is not generally possible, but provide identification proofs demonstrating that it can be achieved under non-linear continuous structural causal models with additive, multivariate Gaussian noise---even when unobserved confounders are present. Importantly, we show how to incorporate observed covariates and learn heterogeneous treatment effects. Based on the identifiability proofs, we provide an algorithm that learns the causal model parameters by pooling data from different regimes and jointly maximising the combined likelihood. The effectiveness of our method is empirically demonstrated on both synthetic and real-world data.

        ----

        ## [2020] MATE: Benchmarking Multi-Agent Reinforcement Learning in Distributed Target Coverage Control

        **Authors**: *Xuehai Pan, Mickel Liu, Fangwei Zhong, Yaodong Yang, Song-Chun Zhu, Yizhou Wang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/b2a1c152f14a4b842a9ddb3bd84c62a1-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2022/hash/b2a1c152f14a4b842a9ddb3bd84c62a1-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        We introduce the Multi-Agent Tracking Environment (MATE), a novel multi-agent environment simulates the target coverage control problems in the real world. MATE hosts an asymmetric cooperative-competitive game consisting of two groups of learning agents--"cameras" and "targets"--with opposing interests. Specifically, "cameras", a group of directional sensors, are mandated to actively control the directional perception area to maximize the coverage rate of targets. On the other side, "targets" are mobile agents that aim to transport cargo between multiple randomly assigned warehouses while minimizing the exposure to the camera sensor networks. To showcase the practicality of MATE, we benchmark the multi-agent reinforcement learning (MARL) algorithms from different aspects, including cooperation, communication, scalability, robustness, and asymmetric self-play. We start by reporting results for cooperative tasks using MARL algorithms (MAPPO, IPPO, QMIX, MADDPG) and the results after augmenting with multi-agent communication protocols (TarMAC, I2C). We then evaluate the effectiveness of the popular self-play techniques (PSRO, fictitious self-play) in an asymmetric zero-sum competitive game. This process of co-evolution between cameras and targets helps to realize a less exploitable camera network. We also observe the emergence of different roles of the target agents while incorporating I2C into target-target communication. MATE is written purely in Python and integrated with OpenAI Gym API to enhance user-friendliness. Our project is released at https://github.com/UnrealTracking/mate.

        ----

        ## [2021] Receding Horizon Inverse Reinforcement Learning

        **Authors**: *Yiqing Xu, Wei Gao, David Hsu*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/b2b781badeeb49896c4b324c466ec442-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/b2b781badeeb49896c4b324c466ec442-Abstract-Conference.html)

        **Abstract**:

        Inverse reinforcement learning (IRL) seeks to infer a cost function that explains the underlying goals and  preferences of  expert demonstrations. This paper presents Receding Horizon Inverse Reinforcement Learning (RHIRL), a new IRL algorithm for high-dimensional, noisy, continuous systems with black-box dynamic models. RHIRL addresses two key challenges of IRL: scalability and robustness. To handle high-dimensional continuous systems, RHIRL matches the induced optimal trajectories with expert demonstrations locally in a receding horizon manner and stitches'' together the local solutions to learn the cost; it thereby avoids thecurse of dimensionality''. This contrasts sharply with  earlier algorithms that match with expert demonstrations globally over the entire high-dimensional state space. To be robust against imperfect expert demonstrations and control noise, RHIRL learns a state-dependent cost function ``disentangled'' from system dynamics under mild conditions. Experiments on benchmark tasks show that RHIRL outperforms several leading IRL algorithms in most instances. We also prove that the cumulative error of RHIRL grows linearly with the task duration.

        ----

        ## [2022] A Reparametrization-Invariant Sharpness Measure Based on Information Geometry

        **Authors**: *Cheongjae Jang, Sungyoon Lee, Frank C. Park, Yung-Kyun Noh*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/b2ba568effcc3ab221912db2fb095ea9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/b2ba568effcc3ab221912db2fb095ea9-Abstract-Conference.html)

        **Abstract**:

        It has been observed that the generalization performance of neural networks correlates with the sharpness of their loss landscape. Dinh et al. (2017) have observed that existing formulations of sharpness measures fail to be invariant with respect to scaling and reparametrization. While some scale-invariant measures have recently been proposed, reparametrization-invariant measures are still lacking. Moreover, they often do not provide any theoretical insights into generalization performance nor lead to practical use to improve the performance. Based on an information geometric analysis of the neural network parameter space, in this paper we propose a reparametrization-invariant sharpness measure that captures the change in loss with respect to changes in the probability distribution modeled by neural networks, rather than with respect to changes in the parameter values. We reveal some theoretical connections of our measure to generalization performance. In particular, experiments confirm that using our measure as a regularizer in neural network training significantly improves performance.

        ----

        ## [2023] Hyper-Representations as Generative Models: Sampling Unseen Neural Network Weights

        **Authors**: *Konstantin Schürholt, Boris Knyazev, Xavier Giró-i-Nieto, Damian Borth*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/b2c4b7d34b3d96b9dc12f7bce424b7ae-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/b2c4b7d34b3d96b9dc12f7bce424b7ae-Abstract-Conference.html)

        **Abstract**:

        Learning representations of neural network weights given a model zoo is an emerg- ing and challenging area with many potential applications from model inspection, to neural architecture search or knowledge distillation. Recently, an autoencoder trained on a model zoo was able to learn a hyper-representation, which captures intrinsic and extrinsic properties of the models in the zoo. In this work, we ex- tend hyper-representations for generative use to sample new model weights. We propose layer-wise loss normalization which we demonstrate is key to generate high-performing models and several sampling methods based on the topology of hyper-representations. The models generated using our methods are diverse, per- formant and capable to outperform strong baselines as evaluated on several down- stream tasks: initialization, ensemble sampling and transfer learning. Our results indicate the potential of knowledge aggregation from model zoos to new models via hyper-representations thereby paving the avenue for novel research directions.

        ----

        ## [2024] Multi-Game Decision Transformers

        **Authors**: *Kuang-Huei Lee, Ofir Nachum, Mengjiao Yang, Lisa Lee, Daniel Freeman, Sergio Guadarrama, Ian Fischer, Winnie Xu, Eric Jang, Henryk Michalewski, Igor Mordatch*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/b2cac94f82928a85055987d9fd44753f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/b2cac94f82928a85055987d9fd44753f-Abstract-Conference.html)

        **Abstract**:

        A longstanding goal of the field of AI is a method for learning a highly capable, generalist agent from diverse experience. In the subfields of vision and language, this was largely achieved by scaling up transformer-based models and training them on large, diverse datasets. Motivated by this progress, we investigate whether the same strategy can be used to produce generalist reinforcement learning agents. Specifically, we show that a single transformer-based model – with a single set of weights – trained purely offline can play a suite of up to 46 Atari games simultaneously at close-to-human performance. When trained and evaluated appropriately, we find that the same trends observed in language and vision hold, including scaling of performance with model size and rapid adaptation to new games via fine-tuning. We compare several approaches in this multi-game setting, such as online and offline RL methods and behavioral cloning, and find that our Multi-Game Decision Transformer models offer the best scalability and performance. We release the pre-trained models and code to encourage further research in this direction.

        ----

        ## [2025] Improving Transformer with an Admixture of Attention Heads

        **Authors**: *Tan Nguyen, Tam Nguyen, Hai Do, Khai Nguyen, Vishwanath Saragadam, Minh Pham, Duy Khuong Nguyen, Nhat Ho, Stanley J. Osher*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/b2e4edd53059e24002a0c916d75cc9a3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/b2e4edd53059e24002a0c916d75cc9a3-Abstract-Conference.html)

        **Abstract**:

        Transformers with multi-head self-attention have achieved remarkable success in sequence modeling and beyond. However, they suffer from high computational and memory complexities for computing the attention matrix at each head. Recently, it has been shown that those attention matrices lie on a low-dimensional manifold and, thus, are redundant. We propose the Transformer with a Finite Admixture of Shared Heads (FiSHformers), a novel class of efficient and flexible transformers that allow the sharing of attention matrices between attention heads. At the core of FiSHformer is a novel finite admixture model of shared heads (FiSH) that samples attention matrices from a set of global attention matrices. The number of global attention matrices is much smaller than the number of local attention matrices generated. FiSHformers directly learn these global attention matrices rather than the local ones as in other transformers, thus significantly improving the computational and memory efficiency of the model. We empirically verify the advantages of the FiSHformer over the baseline transformers in a wide range of practical applications including language modeling, machine translation, and image classification. On the WikiText-103,  IWSLT'14 De-En and WMT'14 En-De, FiSHformers use much fewer floating-point operations per second (FLOPs), memory, and parameters compared to the baseline transformers.

        ----

        ## [2026] Flexible Diffusion Modeling of Long Videos

        **Authors**: *William Harvey, Saeid Naderiparizi, Vaden Masrani, Christian Weilbach, Frank Wood*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/b2fe1ee8d936ac08dd26f2ff58986c8f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/b2fe1ee8d936ac08dd26f2ff58986c8f-Abstract-Conference.html)

        **Abstract**:

        We present a framework for video modeling based on denoising diffusion probabilistic models that produces long-duration video completions in a variety of realistic environments. We introduce a generative model that can at test-time sample any arbitrary subset of video frames conditioned on any other subset and present an architecture adapted for this purpose. Doing so allows us to efficiently compare and optimize a variety of schedules for the order in which frames in a long video are sampled and use selective sparse and long-range conditioning on previously sampled frames.  We demonstrate improved video modeling over prior work on a number of datasets and sample temporally coherent videos over 25 minutes in length.  We additionally release a new video modeling dataset and semantically meaningful metrics based on videos generated in the CARLA autonomous driving simulator.

        ----

        ## [2027] Towards Reasonable Budget Allocation in Untargeted Graph Structure Attacks via Gradient Debias

        **Authors**: *Zihan Liu, Yun Luo, Lirong Wu, Zicheng Liu, Stan Z. Li*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/b31aec087b4c9be97d7148dfdf6e062d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/b31aec087b4c9be97d7148dfdf6e062d-Abstract-Conference.html)

        **Abstract**:

        It has become cognitive inertia to employ cross-entropy loss function in classification related tasks. In the untargeted attacks on graph structure, the gradients derived from the attack objective are the attacker's basis for evaluating a perturbation scheme. Previous methods use negative cross-entropy loss as the attack objective in attacking node-level classification models. However, the suitability of the cross-entropy function for constructing the untargeted attack objective has yet been discussed in previous works. This paper argues about the previous unreasonable attack objective from the perspective of budget allocation. We demonstrate theoretically and empirically that negative cross-entropy tends to produce more significant gradients from nodes with lower confidence in the labeled classes, even if the predicted classes of these nodes have been misled. To free up these inefficient attack budgets, we propose a simple attack model for untargeted attacks on graph structure based on a novel attack objective which generates unweighted gradients on graph structures that are not affected by the node confidence. By conducting experiments in gray-box poisoning attack scenarios, we demonstrate that a reasonable budget allocation can significantly improve the effectiveness of gradient-based edge perturbations without any extra hyper-parameter.

        ----

        ## [2028] ToDD: Topological Compound Fingerprinting in Computer-Aided Drug Discovery

        **Authors**: *Andac Demir, Baris Coskunuzer, Yulia R. Gel, Ignacio Segovia-Dominguez, Yuzhou Chen, Bulent Kiziltan*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/b31f6d65f2584b3c4347148db36fe07f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/b31f6d65f2584b3c4347148db36fe07f-Abstract-Conference.html)

        **Abstract**:

        In computer-aided drug discovery (CADD), virtual screening (VS) is used for comparing a library of compounds against known active ligands to identify the drug candidates that are most likely to bind to a molecular target. Most VS methods to date have focused on using canonical compound representations (e.g., SMILES strings, Morgan fingerprints) or generating alternative fingerprints of the compounds by training progressively more complex variational autoencoders (VAEs) and graph neural networks (GNNs). Although VAEs and GNNs led to significant improvements in VS performance, these methods suffer from reduced performance when scaling to large virtual compound datasets. The performance of these methods has shown only incremental improvements in the past few years. To address this problem, we developed a novel method using multiparameter persistence (MP) homology that produces topological fingerprints of the compounds as multidimensional vectors. Our primary contribution is framing the VS process as a new topology-based graph ranking problem by partitioning a compound into chemical substructures informed by the periodic properties of its atoms and extracting their persistent homology features at multiple resolution levels. We show that the margin loss fine-tuning of pretrained Triplet networks attains highly competitive results in differentiating between compounds in the embedding space and ranking their likelihood of becoming effective drug candidates. We further establish theoretical guarantees for the stability properties of our proposed MP signatures, and demonstrate that our models, enhanced by the MP signatures, outperform state-of-the-art methods on benchmark datasets by a wide and highly statistically significant margin (e.g., 93\% gain for Cleves-Jain and 54\% gain for DUD-E Diverse dataset).

        ----

        ## [2029] Geo-SIC: Learning Deformable Geometric Shapes in Deep Image Classifiers

        **Authors**: *Jian Wang, Miaomiao Zhang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/b328c5bd9ff8e3a5e1be74baf4a7a456-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/b328c5bd9ff8e3a5e1be74baf4a7a456-Abstract-Conference.html)

        **Abstract**:

        Deformable shapes provide important and complex geometric features of objects presented in images. However, such information is oftentimes missing or underutilized as implicit knowledge in many image analysis tasks. This paper presents Geo-SIC, the first deep learning model to learn deformable shapes in a deformation space for an improved performance of image classification. We introduce a newly designed framework that (i) simultaneously derives features from both image and latent shape spaces with large intra-class variations; and (ii) gains increased model interpretability by allowing direct access to the underlying geometric features of image data. In particular, we develop a boosted classification network, equipped with an unsupervised learning of geometric shape representations characterized by diffeomorphic transformations within each class. In contrast to previous approaches using pre-extracted shapes, our model provides a more fundamental approach by naturally learning the most relevant shape features jointly with an image classifier. We demonstrate the effectiveness of our method on both simulated 2D images and real 3D brain magnetic resonance (MR) images. Experimental results show that our model substantially improves the image classification accuracy with an additional benefit of increased model interpretability.  Our code is publicly available at https://github.com/jw4hv/Geo-SIC.

        ----

        ## [2030] A Survey and Datasheet Repository of Publicly Available US Criminal Justice Datasets

        **Authors**: *Miri Zilka, Bradley Butcher, Adrian Weller*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/b3640c2d3e58f716c67066046318db0f-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2022/hash/b3640c2d3e58f716c67066046318db0f-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        Criminal justice is an increasingly important application domain for machine learning and algorithmic fairness, as predictive tools are becoming widely used in police, courts, and prison systems worldwide. A few relevant benchmarks have received significant attention, e.g., the COMPAS dataset, often without proper consideration of the domain context. To raise awareness of publicly available criminal justice datasets and encourage their responsible use, we conduct a survey, consider contexts, highlight potential uses, and identify gaps and limitations. We provide datasheets for 15 datasets and upload them to a public repository. We compare the datasets across several dimensions, including size, coverage of the population, and potential use, highlighting concerns. We hope that this work can provide a useful starting point for researchers looking for appropriate datasets related to criminal justice, and that the repository will continue to grow as a community effort.

        ----

        ## [2031] Segmenting Moving Objects via an Object-Centric Layered Representation

        **Authors**: *Junyu Xie, Weidi Xie, Andrew Zisserman*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/b37aa1d677970f2f56d0d17410c52b3b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/b37aa1d677970f2f56d0d17410c52b3b-Abstract-Conference.html)

        **Abstract**:

        The objective of this paper is a model that is able to discover, track and segment multiple moving objects in a video. We make four contributions: First, we introduce an object-centric segmentation model with a depth-ordered layer representation. This is implemented using a variant of the transformer architecture that ingests optical flow, where each query vector specifies an object and its layer for the entire video. The model can effectively discover multiple moving objects and handle mutual occlusions; Second, we introduce a scalable pipeline for generating multi-object synthetic training data via layer compositions, that is used to train the proposed model, significantly reducing the requirements for labour-intensive annotations, and supporting Sim2Real generalisation; Third, we conduct thorough ablation studies, showing that the model is able to learn object permanence and temporal shape consistency, and is able to predict amodal segmentation masks; Fourth, we evaluate our model, trained only on synthetic data, on standard video segmentation benchmarks, DAVIS, MoCA, SegTrack, FBMS-59, and achieve state-of-the-art performance among existing methods that do not rely on any manual annotations. With test-time adaptation, we observe further performance boosts.

        ----

        ## [2032] NAS-Bench-Suite-Zero: Accelerating Research on Zero Cost Proxies

        **Authors**: *Arjun Krishnakumar, Colin White, Arber Zela, Renbo Tu, Mahmoud Safari, Frank Hutter*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/b3835dd49b7d5bb062aecccc14d8a675-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2022/hash/b3835dd49b7d5bb062aecccc14d8a675-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        Zero-cost proxies (ZC proxies) are a recent architecture performance prediction technique aiming to significantly speed up algorithms for neural architecture search (NAS). Recent work has shown that these techniques show great promise, but certain aspects, such as evaluating and exploiting their complementary strengths, are under-studied. In this work, we create NAS-Bench-Suite: we evaluate 13 ZC proxies across 28 tasks, creating by far the largest dataset (and unified codebase) for ZC proxies, enabling orders-of-magnitude faster experiments on ZC proxies, while avoiding confounding factors stemming from different implementations. To demonstrate the usefulness of NAS-Bench-Suite, we run a large-scale analysis of ZC proxies, including a bias analysis, and the first information-theoretic analysis which concludes that ZC proxies capture substantial complementary information. Motivated by these findings, we present a procedure to improve the performance of ZC proxies by reducing biases such as cell size, and we also show that incorporating all 13 ZC proxies into the surrogate models used by NAS algorithms can improve their predictive performance by up to 42%. Our code and datasets are available at https://github.com/automl/naslib/tree/zerocost.

        ----

        ## [2033] On the Strong Correlation Between Model Invariance and Generalization

        **Authors**: *Weijian Deng, Stephen Gould, Liang Zheng*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/b3847cda0c8cc0cfcdacf462dc122214-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/b3847cda0c8cc0cfcdacf462dc122214-Abstract-Conference.html)

        **Abstract**:

        Generalization and invariance are two essential properties of  machine learning models. Generalization captures a model's ability to classify unseen data while invariance measures consistency of model predictions on transformations of the data. Existing research suggests a positive relationship: a model generalizing well should be invariant to certain visual factors. Building on this qualitative implication we make two contributions. First, we introduce effective invariance (EI), a simple and reasonable measure of model invariance which does not rely on image labels. Given predictions on a test image and its transformed version, EI measures how well the predictions agree and with what level of confidence. Second, using invariance scores computed by EI, we perform large-scale quantitative correlation studies between generalization and invariance, focusing on rotation and grayscale transformations. From a model-centric view, we observe generalization and invariance of different models exhibit a strong linear relationship, on both in-distribution and out-of-distribution datasets. From a dataset-centric view, we find a certain model's accuracy and invariance linearly correlated on different test sets. Apart from these major findings, other minor but interesting insights are also discussed.

        ----

        ## [2034] On Scalable Testing of Samplers

        **Authors**: *Yash Pote, Kuldeep S. Meel*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/b3875605f2e35714fc8a807cadf8a5e8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/b3875605f2e35714fc8a807cadf8a5e8-Abstract-Conference.html)

        **Abstract**:

        In this paper we study the problem of testing of constrained samplers over high-dimensional distributions with $(\varepsilon,\eta,\delta)$ guarantees. Samplers are increasingly used in a wide range of safety-critical ML applications, and hence the testing problem has gained importance. For $n$-dimensional distributions, the existing state-of-the-art algorithm, $\mathsf{Barbarik2}$, has a worst case query complexity of exponential in $n$ and hence is not ideal for use in practice. Our primary contribution is an exponentially faster algorithm, $\mathsf{Barbarik3}$, that has a query complexity linear in $n$ and hence can easily scale to larger instances. We demonstrate our claim by implementing our algorithm and then comparing it against $\mathsf{Barbarik2}$. Our experiments on the samplers $\mathsf{wUnigen3}$ and $\mathsf{wSTS}$, find that $\mathsf{Barbarik3}$ requires $10\times$ fewer samples for $\mathsf{wUnigen3}$ and $450\times$ fewer samples for $\mathsf{wSTS}$ as compared to $\mathsf{Barbarik2}$.

        ----

        ## [2035] How and Why to Manipulate Your Own Agent: On the Incentives of Users of Learning Agents

        **Authors**: *Yoav Kolumbus, Noam Nisan*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/b39fcf2e88dad4c38386b3af6edf88c7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/b39fcf2e88dad4c38386b3af6edf88c7-Abstract-Conference.html)

        **Abstract**:

        The usage of automated learning agents is becoming increasingly prevalent in many online economic applications such as online auctions and automated trading. Motivated by such applications, this paper is dedicated to fundamental modeling and analysis of the strategic situations that the users of automated learning agents are facing. We consider strategic settings where several users engage in a repeated online interaction, assisted by regret-minimizing learning agents that repeatedly play a "game" on their behalf. We propose to view the outcomes of the agents' dynamics as inducing a "meta-game" between the users. Our main focus is on whether users can benefit in this meta-game from "manipulating" their own agents by misreporting their parameters to them. We define a general framework to model and analyze these strategic interactions between users of learning agents for general games and analyze the equilibria induced between the users in three classes of games. We show that, generally, users have incentives to misreport their parameters to their own agents, and that such strategic user behavior can lead to very different outcomes than those anticipated by standard analysis.

        ----

        ## [2036] LIPS - Learning Industrial Physical Simulation benchmark suite

        **Authors**: *Milad Leyli-Abadi, Antoine Marot, Jérôme Picault, David Danan, Mouadh Yagoubi, Benjamin Donnot, Seif Attoui, Pavel Dimitrov, Asma Farjallah, Clement Etienam*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/b3ac9866f6333beaa7d38926101b7e1c-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2022/hash/b3ac9866f6333beaa7d38926101b7e1c-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        Physical simulations are at the core of many critical industrial systems. However, today's physical simulators  have some limitations such as computation time, dealing with missing or uncertain data, or even  non-convergence for some feasible cases. Recently, the use of data-driven approaches to learn complex physical simulations has been considered as a promising approach to address those issues. However, this comes often at the cost of some accuracy which may hinder the industrial use. To drive this new research topic towards a better real-world applicability, we propose a new benchmark suite "Learning Industrial Physical Simulations"(LIPS) to meet the need of developing efficient, industrial application-oriented, augmented simulators. To define how to assess such benchmark performance, we propose a set of four generic categories of criteria. The proposed benchmark suite is a modular and configurable framework that can deal with different physical problems. To demonstrate this ability, we propose in this paper to investigate two distinct use-cases with different physical simulations, namely: the power grid and the pneumatic. For each use case, several benchmarks are described and assessed with existing models. None of the models perform well under all expected criteria, inviting the community to develop  new industry-applicable solutions and possibly showcase their performance publicly upon online LIPS instance on Codabench.

        ----

        ## [2037] FlyView: a bio-informed optical flow truth dataset for visual navigation using panoramic stereo vision

        **Authors**: *Alix Leroy, Graham K. Taylor*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/b4005da5affc3ba527dcb992495ecd20-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2022/hash/b4005da5affc3ba527dcb992495ecd20-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        Flying at speed through complex environments is a challenging task that has been performed successfully by insects since the Carboniferous, but which remains a challenge for robotic and autonomous systems. Insects navigate the world using optical flow sensed by their compound eyes, which they process using a deep neural network weighing just a few milligrams. Deploying an insect-inspired network architecture in computer vision could therefore enable more efficient and effective ways of estimating structure and self-motion using optical flow. Training a bio-informed deep network to implement these tasks requires biologically relevant training, test, and validation data. To this end, we introduce FlyView, a novel bio-informed truth dataset for visual navigation. This simulated dataset is rendered using open source 3D scenes in which the observer's position is known at every frame, and is accompanied by truth data on depth, self-motion, and motion flow. This dataset comprising 42,475 frames has several key features that are missing from existing optical flow datasets, including: (i) panoramic cameras with a monocular and binocular field of view matched to that of a fly's compound eyes; (ii) dynamically meaningful self-motion modelled on motion primitives, or the 3D trajectories of drones and flies; and (iii) complex natural and indoor environments including reflective surfaces.

        ----

        ## [2038] Controllable Text Generation with Neurally-Decomposed Oracle

        **Authors**: *Tao Meng, Sidi Lu, Nanyun Peng, Kai-Wei Chang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/b40d5797756800c97f3d525c2e4c8357-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/b40d5797756800c97f3d525c2e4c8357-Abstract-Conference.html)

        **Abstract**:

        We propose a general and efficient framework to control auto-regressive generation models with NeurAlly-Decomposed Oracle (NADO). Given a pre-trained base language model and a sequence-level boolean oracle function, we aim to decompose the oracle function into token-level guidance to steer the base model in text generation. Specifically, the token-level guidance is provided by NADO, a neural model trained with examples sampled from the base model, demanding no additional auxiliary labeled data. Based on posterior regularization, we present the close-form optimal solution to incorporate the decomposed token-level guidance into the base model for controllable generation. We further discuss how the neural approximation affects the quality of the solution. These experiments conducted on two different applications: (1) text generation with lexical constraints and (2) machine translation with formality control demonstrate that our framework efficiently guides the base model towards the given oracle while keeping high generation quality.

        ----

        ## [2039] Active Learning Helps Pretrained Models Learn the Intended Task

        **Authors**: *Alex Tamkin, Dat Nguyen, Salil Deshpande, Jesse Mu, Noah D. Goodman*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/b43a0e8a35b1c044b18cd843b9771915-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/b43a0e8a35b1c044b18cd843b9771915-Abstract-Conference.html)

        **Abstract**:

        Models can fail in unpredictable ways during deployment due to task ambiguity, when multiple behaviors are consistent with the provided training data. An example is an object classifier trained on red squares and blue circles: when encountering blue squares, the intended behavior is undefined. We investigate whether pretrained models are better active learners, capable of disambiguating between the possible tasks a user may be trying to specify. Intriguingly, we find that better active learning is an emergent property of the pretraining process: pretrained models require up to 5 times fewer labels when using uncertainty-based active learning, while non-pretrained models see no or even negative benefit. We find these gains come from an ability to select examples with attributes that disambiguate the intended behavior, such as rare product categories or atypical backgrounds. These attributes are far more linearly separable in pretrained model's representation spaces vs non-pretrained models, suggesting a possible mechanism for this behavior.

        ----

        ## [2040] Pragmatically Learning from Pedagogical Demonstrations in Multi-Goal Environments

        **Authors**: *Hugo Caselles-Dupré, Olivier Sigaud, Mohamed Chetouani*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/b44ae90136013a8d0e2d24f6015b6097-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/b44ae90136013a8d0e2d24f6015b6097-Abstract-Conference.html)

        **Abstract**:

        Learning from demonstration methods usually leverage close to optimal demonstrations to accelerate training. By contrast, when demonstrating a task, human teachers deviate from optimal demonstrations and pedagogically modify their behavior by giving demonstrations that best disambiguate the goal they want to demonstrate. Analogously, human learners excel at pragmatically inferring the intent of the teacher, facilitating communication between the two agents. These mechanisms are critical in the few demonstrations regime, where inferring the goal is more difficult. In this paper, we implement pedagogy and pragmatism mechanisms by leveraging a Bayesian model of Goal Inference from demonstrations. We highlight the benefits of this model in multi-goal teacher-learner setups with two artificial agents that learn with goal-conditioned Reinforcement Learning. We show that combining BGI-agents (a pedagogical teacher and a pragmatic learner) results in faster learning and reduced goal ambiguity over standard learning from demonstrations, especially in the few demonstrations regime.

        ----

        ## [2041] Beyond black box densities: Parameter learning for the deviated components

        **Authors**: *Dat Do, Nhat Ho, XuanLong Nguyen*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/b4779c2d7130d5e4c29b5a233c34fbdb-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/b4779c2d7130d5e4c29b5a233c34fbdb-Abstract-Conference.html)

        **Abstract**:

        As we collect additional samples from a data population for which a known density function estimate may have been previously obtained by a black box method, the increased complexity of the data set may result in the true density being deviated from the known estimate by a mixture distribution. To model this phenomenon, we consider the \emph{deviating mixture model} $(1-\lambda^{*})h_0 + \lambda^{*} (\sum_{i = 1}^{k} p_{i}^{*} f(x|\theta_{i}^{*}))$, where $h_0$ is a known density function, while the deviated proportion $\lambda^{*}$ and latent mixing measure $G_{*} = \sum_{i = 1}^{k} p_{i}^{*} \delta_{\theta_i^{*}}$ associated with the mixture distribution are unknown. Via a novel notion of distinguishability between the known density $h_{0}$ and the deviated mixture distribution, we establish rates of convergence for the maximum likelihood estimates of $\lambda^{*}$ and $G^{*}$ under Wasserstein metric. Simulation studies are carried out to illustrate the theory.

        ----

        ## [2042] Statistical, Robustness, and Computational Guarantees for Sliced Wasserstein Distances

        **Authors**: *Sloan Nietert, Ziv Goldfeld, Ritwik Sadhu, Kengo Kato*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/b4bc180bf09d513c34ecf66e53101595-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/b4bc180bf09d513c34ecf66e53101595-Abstract-Conference.html)

        **Abstract**:

        Sliced Wasserstein distances preserve properties of classic Wasserstein distances while being more scalable for computation and estimation in high dimensions. The goal of this work is to quantify this scalability from three key aspects: (i) empirical convergence rates; (ii) robustness to data contamination; and (iii) efficient computational methods. For empirical convergence, we derive fast rates with explicit dependence of constants on dimension, subject to log-concavity of the population distributions. For robustness, we characterize minimax optimal, dimension-free robust estimation risks, and show an equivalence between robust sliced 1-Wasserstein estimation and robust mean estimation. This enables lifting statistical and algorithmic guarantees available for the latter to the sliced 1-Wasserstein setting. Moving on to computational aspects, we analyze the Monte Carlo estimator for the average-sliced distance, demonstrating that larger dimension can result in faster convergence of the numerical integration error. For the max-sliced distance, we focus on a subgradient-based local optimization algorithm that is frequently used in practice, albeit without formal guarantees, and establish an $O(\epsilon^{-4})$ computational complexity bound for it. Our theory is validated by numerical experiments, which altogether provide a comprehensive quantitative account of the scalability question.

        ----

        ## [2043] Oracle Inequalities for Model Selection in Offline Reinforcement Learning

        **Authors**: *Jonathan N. Lee, George Tucker, Ofir Nachum, Bo Dai, Emma Brunskill*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/b51693c2ba5b5ddf67429966576fb962-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/b51693c2ba5b5ddf67429966576fb962-Abstract-Conference.html)

        **Abstract**:

        In offline reinforcement learning (RL), a learner leverages prior logged data to learn a good policy without interacting with the environment. A major challenge in applying such methods in practice is the lack of both theoretically principled and practical tools for model selection and evaluation. To address this, we study the problem of model selection in offline RL with value function approximation. The learner is given a nested sequence of model classes to minimize squared Bellman error and must select among these to achieve a balance between approximation and estimation error of the classes. We propose the first model selection algorithm for offline RL that achieves minimax rate-optimal oracle inequalities up to logarithmic factors. The algorithm, ModBE, takes as input a collection of candidate model classes and a generic base offline RL algorithm. By successively eliminating model classes using a novel one-sided generalization test, ModBE returns a policy with regret scaling with the complexity of the minimally complete model class. In addition to its theoretical guarantees, it is conceptually simple and computationally efficient, amounting to solving a series of square loss regression problems and then comparing relative square loss between classes. We conclude with several numerical simulations showing it is capable of reliably selecting a good model class.

        ----

        ## [2044] Model-Based Opponent Modeling

        **Authors**: *Xiaopeng Yu, Jiechuan Jiang, Wanpeng Zhang, Haobin Jiang, Zongqing Lu*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/b528459c99e929718a7d7e1697253d7f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/b528459c99e929718a7d7e1697253d7f-Abstract-Conference.html)

        **Abstract**:

        When one agent interacts with a multi-agent environment, it is challenging to deal with various opponents unseen before. Modeling the behaviors, goals, or beliefs of opponents could help the agent adjust its policy to adapt to different opponents. In addition, it is also important to consider opponents who are learning simultaneously or capable of reasoning. However, existing work usually tackles only one of the aforementioned types of opponents. In this paper, we propose model-based opponent modeling (MBOM), which employs the environment model to adapt to all kinds of opponents. MBOM simulates the recursive reasoning process in the environment model and imagines a set of improving opponent policies. To effectively and accurately represent the opponent policy, MBOM further mixes the imagined opponent policies according to the similarity with the real behaviors of opponents. Empirically, we show that MBOM achieves more effective adaptation than existing methods in a variety of tasks, respectively with different types of opponents, i.e., fixed policy, naive learner, and reasoning learner.

        ----

        ## [2045] What is Where by Looking: Weakly-Supervised Open-World Phrase-Grounding without Text Inputs

        **Authors**: *Tal Shaharabany, Yoad Tewel, Lior Wolf*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/b54e0146a82945f01e69c2e3309ba925-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/b54e0146a82945f01e69c2e3309ba925-Abstract-Conference.html)

        **Abstract**:

        Given an input image, and nothing else, our method returns the bounding boxes of objects in the image and phrases that describe the objects. This is achieved within an open world paradigm, in which the objects in the input image may not have been encountered during the training of the localization mechanism. Moreover, training takes place in a weakly supervised setting, where no bounding boxes are provided. To achieve this, our method combines two pre-trained networks: the CLIP image-to-text matching score and the BLIP image captioning tool. Training takes place on COCO images and their captions and is based on CLIP. Then, during inference, BLIP is used to generate a hypothesis regarding various regions of the current image. Our work generalizes weakly supervised segmentation and phrase grounding and is shown empirically to outperform the state of the art in both domains. It also shows very convincing results in the novel task of weakly-supervised open-world purely visual phrase-grounding presented in our work.For example, on the datasets used for benchmarking phrase-grounding, our method results in a very modest degradation in comparison to methods that employ human captions as an additional input.

        ----

        ## [2046] DeepMed: Semiparametric Causal Mediation Analysis with Debiased Deep Learning

        **Authors**: *Siqi Xu, Lin Liu, Zhonghua Liu*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/b57939005a3cbe40f49b66a0efd6fc8c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/b57939005a3cbe40f49b66a0efd6fc8c-Abstract-Conference.html)

        **Abstract**:

        Causal mediation analysis can unpack the black box of causality and is therefore a powerful tool for disentangling causal pathways in biomedical and social sciences, and also for evaluating machine learning fairness. To reduce bias for estimating Natural Direct and Indirect Effects in mediation analysis, we propose a new method called DeepMed that uses deep neural networks (DNNs) to cross-fit the infinite-dimensional nuisance functions in the efficient influence functions. We obtain novel theoretical results that our DeepMed method (1) can achieve semiparametric efficiency bound without imposing sparsity constraints on the DNN architecture and (2) can adapt to certain low dimensional structures of the nuisance functions, significantly advancing the existing literature on DNN-based semiparametric causal inference. Extensive synthetic experiments are conducted to support our findings and also expose the gap between theory and practice. As a proof of concept, we apply DeepMed to analyze two real datasets on machine learning fairness and reach conclusions consistent with previous findings.

        ----

        ## [2047] On the Learning Mechanisms in Physical Reasoning

        **Authors**: *Shiqian Li, Kewen Wu, Chi Zhang, Yixin Zhu*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/b595fb83245adb57ce1a0bdc2e12681d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/b595fb83245adb57ce1a0bdc2e12681d-Abstract-Conference.html)

        **Abstract**:

        Is dynamics prediction indispensable for physical reasoning? If so, what kind of roles do the dynamics prediction modules play during the physical reasoning process? Most studies focus on designing dynamics prediction networks and treating physical reasoning as a downstream task without investigating the questions above, taking for granted that the designed dynamics prediction would undoubtedly help the reasoning process. In this work, we take a closer look at this assumption, exploring this fundamental hypothesis by comparing two learning mechanisms: Learning from Dynamics (LfD) and Learning from Intuition (LfI). In the first experiment, we directly examine and compare these two mechanisms. Results show a surprising finding: Simple LfI is better than or on par with state-of-the-art LfD. This observation leads to the second experiment with Ground-truth Dynamics (GD), the ideal case of LfD wherein dynamics are obtained directly from a simulator. Results show that dynamics, if directly given instead of approximated, would achieve much higher performance than LfI alone on physical reasoning; this essentially serves as the performance upper bound. Yet practically, LfD mechanism can only predict Approximate Dynamics (AD) using dynamics learning modules that mimic the physical laws, making the following downstream physical reasoning modules degenerate into the LfI paradigm; see the third experiment. We note that this issue is hard to mitigate, as dynamics prediction errors inevitably accumulate in the long horizon. Finally, in the fourth experiment, we note that LfI, the extremely simpler strategy when done right, is more effective in learning to solve physical reasoning problems. Taken together, the results on the challenging benchmark of PHYRE show that LfI is, if not better, as good as LfD with bells and whistles for dynamics prediction. However, the potential improvement from LfD, though challenging, remains lucrative.

        ----

        ## [2048] A Continuous Time Framework for Discrete Denoising Models

        **Authors**: *Andrew Campbell, Joe Benton, Valentin De Bortoli, Thomas Rainforth, George Deligiannidis, Arnaud Doucet*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/b5b528767aa35f5b1a60fe0aaeca0563-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/b5b528767aa35f5b1a60fe0aaeca0563-Abstract-Conference.html)

        **Abstract**:

        We provide the first complete continuous time framework for denoising diffusion models of discrete data. This is achieved by formulating the forward noising process and corresponding reverse time generative process as Continuous Time Markov Chains (CTMCs). The model can be efficiently trained using a continuous time version of the ELBO. We simulate the high dimensional CTMC using techniques developed in chemical physics and exploit our continuous time framework to derive high performance samplers that we show can outperform discrete time methods for discrete data. The continuous time treatment also enables us to derive a novel theoretical result bounding the error between the generated sample distribution and the true data distribution.

        ----

        ## [2049] Simple Mechanisms for Welfare Maximization in Rich Advertising Auctions

        **Authors**: *Gagan Aggarwal, Kshipra Bhawalkar, Aranyak Mehta, Divyarthi Mohan, Alexandros Psomas*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/b5b66077d016c037576cc56a82f97f66-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/b5b66077d016c037576cc56a82f97f66-Abstract-Conference.html)

        **Abstract**:

        Internet ad auctions have evolved from a few lines of text to richer informational layouts that include images, sitelinks, videos, etc. Ads in these new formats occupy varying amounts of space, and an advertiser can provide multiple formats, only one of which can be shown.The seller is now faced with a multi-parameter mechanism design problem.Computing an efficient allocation is computationally intractable, and therefore the standard Vickrey-Clarke-Groves (VCG) auction, while truthful and welfare-optimal, is impractical. In this paper, we tackle a fundamental problem in the design of modern ad auctions. We adopt a ``Myersonian'' approach and study allocation rules that are monotone both in the bid and set of rich ads. We show that such rules can be paired with a payment function to give a truthful auction. Our main technical challenge is designing a monotone rule that yields a good approximation to the optimal welfare. Monotonicity doesn't hold for standard algorithms, e.g. the incremental bang-per-buck order, that give good approximations to ``knapsack-like'' problems such as ours. In fact, we show that no deterministic monotone rule can approximate the optimal welfare within a factor better than $2$ (while there is a non-monotone FPTAS). Our main result is a new, simple, greedy and monotone allocation rule that guarantees a $3$ approximation. In ad auctions in practice, monotone allocation rules are often paired with the so-called \emph{Generalized Second Price (GSP)} payment rule, which charges the minimum threshold price below which the allocation changes. We prove that, even though our monotone allocation rule paired with GSP is not truthful, its Price of Anarchy (PoA) is bounded. Under standard no-overbidding assumptions, we prove bounds on the a pure and Bayes-Nash PoA. Finally, we experimentally test our algorithms on real-world data.

        ----

        ## [2050] Continuously Tempered PDMP samplers

        **Authors**: *Matthew Sutton, Robert Salomone, Augustin Chevallier, Paul Fearnhead*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/b5b939436789f76f08b9d0da5e81af7c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/b5b939436789f76f08b9d0da5e81af7c-Abstract-Conference.html)

        **Abstract**:

        New sampling algorithms based on simulating continuous-time stochastic processes called piece-wise deterministic Markov processes (PDMPs) have shown considerable promise. However, these methods can struggle to sample from multi-modal or heavy-tailed distributions. We show how tempering ideas can improve the mixing of PDMPs in such cases. We introduce an extended distribution defined over the state of the posterior distribution and an inverse temperature, which interpolates between a tractable distribution when the inverse temperature is 0 and the posterior when the inverse temperature is 1. The marginal distribution of the inverse temperature is a mixture of a continuous distribution on $[0,1)$ and a point mass at 1: which means that we obtain samples when the inverse temperature is 1, and these are draws from the posterior, but sampling algorithms will also explore distributions at lower temperatures which will improve mixing. We show how PDMPs, and particularly the Zig-Zag sampler, can be implemented to sample from such an extended distribution. The resulting algorithm is easy to implement and we show empirically that it can outperform existing PDMP-based samplers on challenging multimodal posteriors.

        ----

        ## [2051] Distributed Influence-Augmented Local Simulators for Parallel MARL in Large Networked Systems

        **Authors**: *Miguel Suau, Jinke He, Mustafa Mert Çelikok, Matthijs T. J. Spaan, Frans A. Oliehoek*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/b5c8c1c117618267944b2617add0a766-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/b5c8c1c117618267944b2617add0a766-Abstract-Conference.html)

        **Abstract**:

        Due to its high sample complexity, simulation is, as of today, critical for the successful application of reinforcement learning. Many real-world problems, however, exhibit overly complex dynamics, making their full-scale simulation computationally slow. In this paper, we show how to factorize large networked systems of many agents into multiple local regions such that we can build separate simulators that run independently and in parallel. To monitor the influence that the different local regions exert on one another, each of these simulators is equipped with a learned model that is periodically trained on real trajectories. Our empirical results reveal that distributing the simulation among different processes not only makes it possible to train large multi-agent systems in just a few hours but also helps mitigate the negative effects of simultaneous learning.

        ----

        ## [2052] Tight Mutual Information Estimation With Contrastive Fenchel-Legendre Optimization

        **Authors**: *Qing Guo, Junya Chen, Dong Wang, Yuewei Yang, Xinwei Deng, Jing Huang, Lawrence Carin, Fan Li, Chenyang Tao*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/b5cc526f12164b2144bb2e06f2e84864-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/b5cc526f12164b2144bb2e06f2e84864-Abstract-Conference.html)

        **Abstract**:

        Successful applications of InfoNCE (Information Noise-Contrastive Estimation) and its variants have popularized the use of contrastive variational mutual information (MI) estimators in machine learning . While featuring superior stability, these estimators crucially depend on costly large-batch training, and they sacrifice bound tightness for variance reduction. To overcome these limitations, we revisit the mathematics of popular variational MI bounds from the lens of unnormalized statistical modeling and convex optimization. Our investigation yields a new unified theoretical framework encompassing popular variational MI bounds, and leads to a novel, simple, and powerful contrastive MI estimator we name FLO. Theoretically, we show that the FLO estimator is tight, and it converges under stochastic gradient descent. Empirically, the proposed FLO estimator overcomes the limitations of its predecessors and learns more efficiently. The utility of FLO is verified using extensive benchmarks, and we further inspire the community with novel applications in meta-learning. Our presentation underscores the foundational importance of variational MI estimation in data-efficient learning.

        ----

        ## [2053] Exploring the Algorithm-Dependent Generalization of AUPRC Optimization with List Stability

        **Authors**: *Peisong Wen, Qianqian Xu, Zhiyong Yang, Yuan He, Qingming Huang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/b5dc49f44db2fadc5c4d717c57f4a424-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/b5dc49f44db2fadc5c4d717c57f4a424-Abstract-Conference.html)

        **Abstract**:

        Stochastic optimization of the Area Under the Precision-Recall Curve (AUPRC) is a crucial problem for machine learning. Although various algorithms have been extensively studied for AUPRC optimization, the generalization is only guaranteed in the multi-query case. In this work, we present the first trial in the single-query generalization of stochastic AUPRC optimization. For sharper generalization bounds, we focus on algorithm-dependent generalization. There are both algorithmic and theoretical obstacles to our destination. From an algorithmic perspective, we notice that the majority of existing stochastic estimators are biased when the sampling strategy is biased, and is leave-one-out unstable due to the non-decomposability. To address these issues, we propose a sampling-rate-invariant unbiased stochastic estimator with superior stability. On top of this, the AUPRC optimization is formulated as a composition optimization problem, and a stochastic algorithm is proposed to solve this problem. From a theoretical perspective, standard techniques of the algorithm-dependent generalization analysis cannot be directly applied to such a listwise compositional optimization problem. To fill this gap, we extend the model stability from instancewise losses to listwise losses and bridge the corresponding generalization and stability. Additionally, we construct state transition matrices to describe the recurrence of the stability, and simplify calculations by matrix spectrum. Practically, experimental results on three image retrieval datasets on speak to the effectiveness and soundness of our framework.

        ----

        ## [2054] LTMD: Learning Improvement of Spiking Neural Networks with Learnable Thresholding Neurons and Moderate Dropout

        **Authors**: *Siqi Wang, Tee Hiang Cheng, Meng-Hiot Lim*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/b5fd95d6b16d3172e307103a97f19e1b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/b5fd95d6b16d3172e307103a97f19e1b-Abstract-Conference.html)

        **Abstract**:

        Spiking Neural Networks (SNNs) have shown substantial promise in processing spatio-temporal data, mimicking biological neuronal mechanisms, and saving computational power. However, most SNNs use fixed model regardless of their locations in the network. This limits SNNsâ€™ capability of transmitting precise information in the network, which becomes worse for deeper SNNs. Some researchers try to use specified parametric models in different network layers or regions, but most still use preset or suboptimal parameters. Inspired by the neuroscience observation that different neuronal mechanisms exist in disparate brain regions, we propose a new spiking neuronal mechanism, named learnable thresholding, to address this issue. Utilizing learnable threshold values, learnable thresholding enables flexible neuronal mechanisms across layers, proper information flow within the network, and fast network convergence. In addition, we propose a moderate dropout method to serve as an enhancement technique to minimize inconsistencies between independent dropout runs. Finally, we evaluate the robustness of the proposed learnable thresholding and moderate dropout for image classification with different initial thresholds for various types of datasets. Our proposed methods produce superior results compared to other approaches for almost all datasets with fewer timesteps. Our codes are available at https://github.com/sq117/LTMD.git.

        ----

        ## [2055] TransBoost: Improving the Best ImageNet Performance using Deep Transduction

        **Authors**: *Omer Belhasin, Guy Bar-Shalom, Ran El-Yaniv*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/b60161e93f3e0e4207081a3b4ef5e8d8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/b60161e93f3e0e4207081a3b4ef5e8d8-Abstract-Conference.html)

        **Abstract**:

        This paper deals with deep transductive learning, and proposes TransBoost as a procedure for fine-tuning any deep neural model to improve its performance on any (unlabeled) test set provided at training time. TransBoost is inspired by a large margin principle and is efficient and simple to use. Our method significantly improves the ImageNet classification performance on a wide range of architectures, such as ResNets, MobileNetV3-L, EfficientNetB0, ViT-S, and ConvNext-T, leading to state-of-the-art transductive performance.Additionally we show that TransBoost is effective on a wide variety of image classification datasets. The implementation of TransBoost is provided at: https://github.com/omerb01/TransBoost .

        ----

        ## [2056] Sparse Probabilistic Circuits via Pruning and Growing

        **Authors**: *Meihua Dang, Anji Liu, Guy Van den Broeck*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/b6089408f4893289296ad0499783b3a6-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/b6089408f4893289296ad0499783b3a6-Abstract-Conference.html)

        **Abstract**:

        Probabilistic circuits (PCs) are a tractable representation of probability distributions allowing for exact and efficient computation of likelihoods and marginals. There has been significant recent progress on improving the scale and expressiveness of PCs. However, PC training performance plateaus as model size increases. We discover that most capacity in existing large PC structures is wasted: fully-connected parameter layers are only sparsely used. We propose two operations: pruning and growing, that exploit the sparsity of PC structures. Specifically, the pruning operation removes unimportant sub-networks of the PC for model compression and comes with theoretical guarantees. The growing operation increases model capacity by increasing the dimensions of latent states. By alternatingly applying pruning and growing, we increase the capacity that is meaningfully used, allowing us to significantly scale up PC learning. Empirically, our learner achieves state-of-the-art likelihoods on MNIST-family image datasets and an Penn Tree Bank language data compared to other PC learners and less tractable deep generative models such as flow-based models and variational autoencoders (VAEs).

        ----

        ## [2057] Adam Can Converge Without Any Modification On Update Rules

        **Authors**: *Yushun Zhang, Congliang Chen, Naichen Shi, Ruoyu Sun, Zhi-Quan Luo*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/b6260ae5566442da053e5ab5d691067a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/b6260ae5566442da053e5ab5d691067a-Abstract-Conference.html)

        **Abstract**:

        Ever since \citet{reddi2019convergence} pointed out the divergence issue of Adam, many new variants have been designed to obtain convergence. However, vanilla Adam remains exceptionally popular and it works well in practice. Why is there a gap between theory and practice? We point out there is a mismatch between the settings of theory and practice: \citet{reddi2019convergence} pick the problem after picking the hyperparameters of Adam, i.e., $(\beta_1,\beta_2)$; while practical applications often fix the problem first and then tune $(\beta_1,\beta_2)$.   Due to this observation, we conjecture that the empirical convergence can be theoretically justified, only if we change the order of picking the problem and hyperparameter.  In this work, we confirm this conjecture.  We prove that, when the 2nd-order momentum parameter $\beta_2$ is large and 1st-order momentum parameter $\beta_1 < \sqrt{\beta_2}<1$, Adam converges to the neighborhood of critical points. The size of the neighborhood is propositional to the variance of stochastic gradients. Under an extra condition (strong growth condition), Adam converges to critical points. It is worth mentioning that our results cover a wide range of hyperparameters: as $\beta_2$  increases, our convergence result can cover any $\beta_1 \in [0,1)$ including $\beta_1=0.9$, which is the default setting in deep learning libraries. To our knowledge, this is the first result showing that Adam can converge {\it without any modification} on its update rules. Further, our analysis does not require assumptions of bounded gradients or bounded 2nd-order momentum. When $\beta_2$ is small, we further point out a large region of  $(\beta_1,\beta_2)$ combinations where  Adam can diverge to infinity. Our divergence result considers the same setting (fixing the optimization problem ahead) as our convergence result, indicating that there is a phase transition from divergence to convergence when increasing $\beta_2$. These positive and negative results provide suggestions on how to tune Adam hyperparameters: for instance,  when Adam does not work well, we suggest tuning up $\beta_2$ and trying $\beta_1< \sqrt{\beta_2}$.

        ----

        ## [2058] Nonlinear MCMC for Bayesian Machine Learning

        **Authors**: *James Vuckovic*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/b6341525cd84f3be0ef203e4d7cd8556-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/b6341525cd84f3be0ef203e4d7cd8556-Abstract-Conference.html)

        **Abstract**:

        We explore the application of a nonlinear MCMC technique first introduced in [1] to problems in Bayesian machine learning. We provide a convergence guarantee in total variation that uses novel results for long-time convergence and large-particle (``propagation of chaos'') convergence. We apply this nonlinear MCMC technique to sampling problems including a Bayesian neural network on CIFAR10.

        ----

        ## [2059] Continual learning: a feature extraction formalization, an efficient algorithm, and fundamental obstructions

        **Authors**: *Binghui Peng, Andrej Risteski*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/b63a24a1832bd14fa945c71f535c0095-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/b63a24a1832bd14fa945c71f535c0095-Abstract-Conference.html)

        **Abstract**:

        Continual learning is an emerging paradigm in machine learning, wherein a model is exposed in an online fashion to data from multiple different distributions (i.e. environments), and is expected to adapt to the distribution change. Precisely, the goal is to perform well in the new environment, while simultaneously retaining the performance on the previous environments (i.e. avoid ``catastrophic forgetting'').While this setup has enjoyed a lot of attention in the applied community, there hasn't be theoretical work that even formalizes the desired guarantees. In this paper, we propose a framework for continual learning through the framework of feature extraction---namely, one in which features, as well as a classifier, are being trained with each environment. When the features are linear, we design an efficient gradient-based algorithm $\mathsf{DPGrad}$, that is guaranteed to perform well on the current environment, as well as avoid catastrophic forgetting. In the general case, when the features are non-linear, we show such an algorithm cannot exist, whether efficient or not.

        ----

        ## [2060] IKEA-Manual: Seeing Shape Assembly Step by Step

        **Authors**: *Ruocheng Wang, Yunzhi Zhang, Jiayuan Mao, Ran Zhang, Chin-Yi Cheng, Jiajun Wu*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/b645d1a085bcb39bece5c03703b62464-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2022/hash/b645d1a085bcb39bece5c03703b62464-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        Human-designed visual manuals are crucial components in shape assembly activities. They provide step-by-step guidance on how we should move and connect different parts in a convenient and physically-realizable way. While there has been an ongoing effort in building agents that perform assembly tasks, the information in human-design manuals has been largely overlooked. We identify that this is due to 1) a lack of realistic 3D assembly objects that have paired manuals and 2) the difficulty of extracting structured information from purely image-based manuals. Motivated by this observation, we present IKEA-Manual, a dataset consisting of 102 IKEA objects paired with assembly manuals. We provide fine-grained annotations on the IKEA objects and assembly manuals, including decomposed assembly parts, assembly plans, manual segmentation, and 2D-3D correspondence between 3D parts and visual manuals. We illustrate the broad application of our dataset on four tasks related to shape assembly: assembly plan generation, part segmentation, pose estimationand 3D part assembly.

        ----

        ## [2061] M³ViT: Mixture-of-Experts Vision Transformer for Efficient Multi-task Learning with Model-Accelerator Co-design

        **Authors**: *Hanxue Liang, Zhiwen Fan, Rishov Sarkar, Ziyu Jiang, Tianlong Chen, Kai Zou, Yu Cheng, Cong Hao, Zhangyang Wang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/b653f34d576d1790481e3797cb740214-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/b653f34d576d1790481e3797cb740214-Abstract-Conference.html)

        **Abstract**:

        Multi-task learning (MTL) encapsulates multiple learned tasks in a single model and often lets those tasks learn better jointly. Multi-tasking models have become successful and often essential for many sophisticated systems such as autonomous driving and indoor robots. However, when deploying MTL onto those real-world systems that are often resource-constrained or latency-sensitive, two prominent challenges arise: (i) during training, simultaneously optimizing all tasks is often difficult due to gradient conflicts across tasks, and the challenge is amplified when a growing number of tasks have to be squeezed into one compact model; (ii) at inference, current MTL regimes have to activate nearly the entire model even to just execute a single task. Yet most real systems demand only one or two tasks at each moment, while flexibly switching between tasks per need: therefore such “all tasks activated” inference is also highly inefficient and non-scalable in practice. In this paper, we present a model-accelerator co-design framework to enable efficient on-device MTL, that tackles both training and inference bottlenecks. Our framework, dubbed M³ViT, customizes mixture-of-experts (MoE) layers into a vision transformer (ViT) backbone for MTL, and sparsely activates task-specific experts during training, which effectively disentangles the parameter spaces to avoid different tasks’ training conflicts. Then at inference with any task of interest, the same design allows for activating only the task-corresponding sparse “expert” pathway, instead of the full model. Our new model design is further enhanced by hardware-level innovations, in particular, a novel computation reordering scheme tailored for memory-constrained MTL that achieves zero-overhead switching between tasks and can scale to any number of experts. Extensive experiments on PASCAL-Context and NYUD-v2 datasets at both software and hardware levels are conducted to demonstrate the effectiveness of the proposed design. When executing the practical scenario of single-task inference, M³ViT achieves higher accuracies than encoder-focused MTL methods, while significantly reducing 88% inference FLOPs. When implemented on a hardware platform of one Xilinx ZCU104 FPGA, our co-design framework reduces the memory requirement by 2.40×, while achieving energy efficiency (as the product of latency and power) up to 9.23× times higher than a comparable FPGA baseline.

        ----

        ## [2062] When to Make Exceptions: Exploring Language Models as Accounts of Human Moral Judgment

        **Authors**: *Zhijing Jin, Sydney Levine, Fernando Gonzalez Adauto, Ojasv Kamal, Maarten Sap, Mrinmaya Sachan, Rada Mihalcea, Josh Tenenbaum, Bernhard Schölkopf*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/b654d6150630a5ba5df7a55621390daf-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/b654d6150630a5ba5df7a55621390daf-Abstract-Conference.html)

        **Abstract**:

        AI systems are becoming increasingly intertwined with human life. In order to effectively collaborate with humans and ensure safety, AI systems need to be able to understand, interpret and predict human moral judgments and decisions. Human moral judgments are often guided by rules, but not always. A central challenge for AI safety is capturing the flexibility of the human moral mind — the ability to determine when a rule should be broken, especially in novel or unusual situations. In this paper, we present a novel challenge set consisting of moral exception question answering (MoralExceptQA) of cases that involve potentially permissible moral exceptions – inspired by recent moral psychology studies. Using a state-of-the-art large language model (LLM) as a basis, we propose a novel moral chain of thought (MoralCoT) prompting strategy that combines the strengths of LLMs with theories of moral reasoning developed in cognitive science to predict human moral judgments. MoralCoT outperforms seven existing LLMs by 6.2% F1, suggesting that modeling human reasoning might be necessary to capture the flexibility of the human moral mind. We also conduct a detailed error analysis to suggest directions for future work to improve AI safety using MoralExceptQA. Our data is open-sourced at https://huggingface.co/datasets/feradauto/MoralExceptQA and code at https://github.com/feradauto/MoralCoT.

        ----

        ## [2063] Exponential Family Model-Based Reinforcement Learning via Score Matching

        **Authors**: *Gene Li, Junbo Li, Anmol Kabra, Nati Srebro, Zhaoran Wang, Zhuoran Yang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/b693a240cf1009bff9fa4422141c9392-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/b693a240cf1009bff9fa4422141c9392-Abstract-Conference.html)

        **Abstract**:

        We propose an optimistic model-based algorithm, dubbed SMRL, for finite-horizon episodic reinforcement learning (RL) when the transition model is specified by exponential family distributions with $d$ parameters and the reward is bounded and known. SMRL uses score matching, an unnormalized density estimation technique that enables efficient estimation of the model parameter by ridge regression. Under standard regularity assumptions, SMRL achieves $\tilde O(d\sqrt{H^3T})$ online regret, where $H$ is the length of each episode and $T$ is the total number of interactions (ignoring polynomial dependence on structural scale parameters).

        ----

        ## [2064] Monte Carlo Tree Search based Variable Selection for High Dimensional Bayesian Optimization

        **Authors**: *Lei Song, Ke Xue, Xiaobin Huang, Chao Qian*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/b6a171867138c80de2a35a6125d6757c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/b6a171867138c80de2a35a6125d6757c-Abstract-Conference.html)

        **Abstract**:

        Bayesian optimization (BO) is a class of popular methods for expensive black-box optimization, and has been widely applied to many scenarios. However, BO suffers from the curse of dimensionality, and scaling it to high-dimensional problems is still a challenge. In this paper, we propose a variable selection method MCTS-VS based on Monte Carlo tree search (MCTS), to iteratively select and optimize a subset of variables. That is, MCTS-VS constructs a low-dimensional subspace via MCTS and optimizes in the subspace with any BO algorithm. We give a theoretical analysis of the general variable selection method to reveal how it can work. Experiments on high-dimensional synthetic functions and real-world problems (e.g., MuJoCo locomotion tasks) show that MCTS-VS equipped with a proper BO optimizer can achieve state-of-the-art performance.

        ----

        ## [2065] OpenSRH: optimizing brain tumor surgery using intraoperative stimulated Raman histology

        **Authors**: *Cheng Jiang, Asadur Chowdury, Xinhai Hou, Akhil Kondepudi, Christian W. Freudiger, Kyle Conway, Sandra Camelo-Piragua, Daniel A. Orringer, Honglak Lee, Todd C. Hollon*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/b6b5f50a2001ad1cbccca96e693c4ab4-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2022/hash/b6b5f50a2001ad1cbccca96e693c4ab4-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        Accurate intraoperative diagnosis is essential for providing safe and effective care during brain tumor surgery. Our standard-of-care diagnostic methods are time, resource, and labor intensive, which restricts access to optimal surgical treatments. To address these limitations, we propose an alternative workflow that combines stimulated Raman histology (SRH), a rapid optical imaging method, with deep learning-based automated interpretation of SRH images for intraoperative brain tumor diagnosis and real-time surgical decision support. Here, we present OpenSRH, the first public dataset of clinical SRH images from 300+ brain tumors patients and 1300+ unique whole slide optical images. OpenSRH contains data from the most common brain tumors diagnoses, full pathologic annotations, whole slide tumor segmentations, raw and processed optical imaging data for end-to-end model development and validation. We provide a framework for patch-based whole slide SRH classification and inference using weak (i.e. patient-level) diagnostic labels. Finally, we benchmark two computer vision tasks: multi-class histologic brain tumor classification and patch-based contrastive representation learning. We hope OpenSRH will facilitate the clinical translation of rapid optical imaging and real-time ML-based surgical decision support in order to improve the access, safety, and efficacy of cancer surgery in the era of precision medicine.

        ----

        ## [2066] Assistive Teaching of Motor Control Tasks to Humans

        **Authors**: *Megha Srivastava, Erdem Biyik, Suvir Mirchandani, Noah D. Goodman, Dorsa Sadigh*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/b6fa3ed9624c184bd73e435123bd576a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/b6fa3ed9624c184bd73e435123bd576a-Abstract-Conference.html)

        **Abstract**:

        Recent works on shared autonomy and assistive-AI technologies, such as assistive robotic teleoperation, seek to model and help human users with limited ability in a fixed task. However, these approaches often fail to account for humans' ability to adapt and eventually learn how to execute a control task themselves. Furthermore, in applications where it may be desirable for a human to intervene, these methods may have inhibited their ability to learn how to succeed with full self-control. In this paper, we focus on the problem of assistive teaching of motor control tasks such as parking a car or landing an aircraft. Despite their ubiquitous role in humans' daily activities and occupations, motor tasks are rarely taught in a uniform way due to their high complexity and variance. We propose an AI-assisted teaching algorithm that leverages skill discovery methods from reinforcement learning (RL) literature to (i) break down any motor control task into teachable skills, (ii) construct novel drill sequences, and (iii) individualize curricula to students with different capabilities. Through an extensive mix of synthetic and user studies on two motor control tasks - parking a car with a joystick and writing  characters from the Balinese alphabet - we show that assisted teaching with skills improve student performance by around 40% compared to practicing full trajectories without skills, and practicing with individualized drills can result in up to 25% further improvement.

        ----

        ## [2067] Retrospective Adversarial Replay for Continual Learning

        **Authors**: *Lilly Kumari, Shengjie Wang, Tianyi Zhou, Jeff A. Bilmes*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/b6ffbbacbe2e56f2ec9a0da907382b4a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/b6ffbbacbe2e56f2ec9a0da907382b4a-Abstract-Conference.html)

        **Abstract**:

        Continual learning is an emerging research challenge in machine learning that addresses the problem where models quickly fit the most recently trained-on data but suffer from catastrophic forgetting of previous data due to distribution shifts --- it does this by maintaining a small historical replay buffer in replay-based methods. To avoid these problems, this paper proposes a method, ``Retrospective Adversarial Replay (RAR)'', that synthesizes adversarial samples near the forgetting boundary. RAR perturbs a buffered sample towards its nearest neighbor drawn from the current task in a latent representation space. By replaying such samples, we are able to refine the boundary between previous and current tasks, hence combating forgetting and reducing bias towards the current task. To mitigate the severity of a small replay buffer, we develop a novel MixUp-based strategy to increase replay variation by replaying mixed augmentations. Combined with RAR, this achieves a holistic framework that helps to alleviate catastrophic forgetting. We show that this excels on broadly-used benchmarks and outperforms other continual learning baselines especially when only a small buffer is available. We conduct a thorough ablation study over each key component as well as a hyperparameter sensitivity analysis to demonstrate the effectiveness and robustness of RAR.

        ----

        ## [2068] Unsupervised Image-to-Image Translation with Density Changing Regularization

        **Authors**: *Shaoan Xie, Qirong Ho, Kun Zhang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/b7032a9d960ebb6bcf1ce9d73b5861f0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/b7032a9d960ebb6bcf1ce9d73b5861f0-Abstract-Conference.html)

        **Abstract**:

        Unpaired image-to-image translation aims to translate an input image to another domain such that the output image looks like an image from another domain while important semantic information are preserved. Inferring the optimal mapping with unpaired data is impossible without making any assumptions.   In this paper, we make a density changing assumption where image patches of high probability density should be mapped to patches of high probability density in another domain. Then we propose an efficient way to enforce this assumption: we train the flows as density estimators and penalize the variance of density changes. Despite its simplicity, our method achieves the best performance on benchmark datasets and needs only $56-86\%$ of training time of the existing state-of-the-art method. The training and evaluation code are avaliable at $$\url{https://github.com/Mid-Push/Decent}.$$

        ----

        ## [2069] CASA: Category-agnostic Skeletal Animal Reconstruction

        **Authors**: *Yuefan Wu, Zeyuan Chen, Shaowei Liu, Zhongzheng Ren, Shenlong Wang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/b709131d0a67f743915e12bc57947ddb-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/b709131d0a67f743915e12bc57947ddb-Abstract-Conference.html)

        **Abstract**:

        Recovering a skeletal shape from a monocular video is a longstanding challenge. Prevailing nonrigid animal reconstruction methods often adopt a control-point driven animation model and optimize bone transforms individually without considering skeletal topology, yielding unsatisfactory shape and articulation. In contrast, humans can easily infer the articulation structure of an unknown character by associating it with a seen articulated object in their memory.  Inspired by this fact, we present CASA, a novel category-agnostic articulated animal reconstruction method. Our method consists of two components, a video-to-shape retrieval process and a neural inverse graphics framework. During inference, CASA first finds a matched articulated shape from a 3D character assets bank so that the input video scores highly with the rendered image, according to a pretrained image-language model. It then integrates the retrieved character into an inverse graphics framework and jointly infers the shape deformation, skeleton structure, and skinning weights through optimization. Experiments validate the efficacy of our method in shape reconstruction and articulation. We further show that we can use the resulting skeletal-animated character for re-animation.

        ----

        ## [2070] Regret Bounds for Information-Directed Reinforcement Learning

        **Authors**: *Botao Hao, Tor Lattimore*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/b733cdd80ed2ae7e3156d8c33108c5d5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/b733cdd80ed2ae7e3156d8c33108c5d5-Abstract-Conference.html)

        **Abstract**:

        Information-directed sampling (IDS) has revealed its potential as a data-efficient algorithm for reinforcement learning (RL). However, theoretical understanding of IDS for Markov Decision Processes (MDPs) is still limited. We develop novel information-theoretic tools to bound the information ratio and cumulative information gain about the learning target. Our theoretical results shed light on the importance of choosing the learning target such that the practitioners can balance the computation and regret bounds. As a consequence, we derive prior-free Bayesian regret bounds for vanilla-IDS which learns the whole environment under tabular finite-horizon MDPs. In addition, we propose a computationally-efficient regularized-IDS that maximizes an additive form rather than the ratio form and show that it enjoys the same regret bound as vanilla-IDS. With the aid of rate-distortion theory, we improve the regret bound by learning a surrogate, less informative environment. Furthermore, we extend our analysis to linear MDPs and prove similar regret bounds for Thompson sampling as a by-product.

        ----

        ## [2071] Adversarial Reprogramming Revisited

        **Authors**: *Matthias Englert, Ranko Lazic*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/b734c30b9c955c535e333f0301f5e45c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/b734c30b9c955c535e333f0301f5e45c-Abstract-Conference.html)

        **Abstract**:

        Adversarial reprogramming, introduced by Elsayed, Goodfellow, and Sohl-Dickstein, seeks to repurpose a neural network to perform a different task, by manipulating its input without modifying its weights.  We prove that two-layer ReLU neural networks with random weights can be adversarially reprogrammed to achieve arbitrarily high accuracy on Bernoulli data models over hypercube vertices, provided the network width is no greater than its input dimension.  We also substantially strengthen a recent result of Phuong and Lampert on directional convergence of gradient flow, and obtain as a corollary that training two-layer ReLU neural networks on orthogonally separable datasets can cause their adversarial reprogramming to fail.  We support these theoretical results by experiments that demonstrate that, as long as batch normalisation layers are suitably initialised, even untrained networks with random weights are susceptible to adversarial reprogramming.  This is in contrast to observations in several recent works that suggested that adversarial reprogramming is not possible for untrained networks to any degree of reliability.

        ----

        ## [2072] Coded Residual Transform for Generalizable Deep Metric Learning

        **Authors**: *Shichao Kan, Yixiong Liang, Min Li, Yigang Cen, Jianxin Wang, Zhihai He*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/b74a8de47d2b3c928360e0a011f48351-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/b74a8de47d2b3c928360e0a011f48351-Abstract-Conference.html)

        **Abstract**:

        A fundamental challenge in deep metric learning is the generalization capability of the  feature embedding network model since the embedding network learned on training classes need to be evaluated on new test classes. To address this challenge, in this paper, we introduce a new method called coded residual transform (CRT) for deep metric learning to significantly improve its generalization capability. Specifically, we learn a set of diversified prototype features, project the feature map onto each prototype, and then encode its features using their projection residuals weighted by their correlation coefficients with each prototype. The proposed CRT method has the following two unique characteristics. First, it represents and encodes the feature map from a set of complimentary perspectives based on projections onto diversified prototypes. Second, unlike existing transformer-based feature representation approaches which encode the original values of features based on global correlation analysis, the proposed coded residual transform encodes the relative differences between the original features and their projected prototypes. Embedding space density and spectral decay analysis show that this multi perspective projection onto diversified prototypes and coded residual representation  are able to achieve significantly improved generalization capability in metric learning. Finally, to further enhance the generalization performance, we propose to enforce the consistency on their feature similarity matrices between  coded residual transforms with different sizes of projection prototypes and embedding dimensions. Our extensive experimental results and ablation studies demonstrate that the proposed CRT method outperform the state-of-the-art deep metric learning methods by large margins and improving upon the current best method by up to 4.28% on the CUB dataset.

        ----

        ## [2073] When Does Differentially Private Learning Not Suffer in High Dimensions?

        **Authors**: *Xuechen Li, Daogao Liu, Tatsunori B. Hashimoto, Huseyin A. Inan, Janardhan Kulkarni, Yin Tat Lee, Abhradeep Guha Thakurta*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/b75ce884441c983f7357a312ffa02a3c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/b75ce884441c983f7357a312ffa02a3c-Abstract-Conference.html)

        **Abstract**:

        Large pretrained models can be fine-tuned with differential privacy to achieve performance approaching that of non-private models. A common theme in these results is the surprising observation that high-dimensional models can achieve favorable privacy-utility trade-offs. This seemingly contradicts known results on the model-size dependence of differentially private convex learning and raises the following research question: When does the performance of differentially private learning not degrade with increasing model size? We identify that the magnitudes of gradients projected onto subspaces is a key factor that determines performance. To precisely characterize this for private convex learning, we introduce a condition on the objective that we term restricted Lipschitz continuity and derive improved bounds for the excess empirical and population risks that are dimension- independent under additional conditions. We empirically show that in private fine-tuning of large language models, gradients obtained during fine-tuning are mostly controlled by a few principal components. This behavior is similar to conditions under which we obtain dimension-independent bounds in convex settings. Our theoretical and empirical results together provide a possible explanation for the recent success of large-scale private fine-tuning. Code to reproduce our results can be found at https://github.com/lxuechen/private-transformers/tree/main/examples/classification/spectral_analysis.

        ----

        ## [2074] Nearly Optimal Best-of-Both-Worlds Algorithms for Online Learning with Feedback Graphs

        **Authors**: *Shinji Ito, Taira Tsuchiya, Junya Honda*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/b7aea253ab34a773967f1e4cdea9e4fb-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/b7aea253ab34a773967f1e4cdea9e4fb-Abstract-Conference.html)

        **Abstract**:

        This study considers online learning with general directed feedback graphs. For this problem, we present best-of-both-worlds algorithms that achieve nearly tight regret bounds for adversarial environments as well as poly-logarithmic regret bounds for stochastic environments. As Alon et al. [2015] have shown, tight regret bounds depend on the structure of the feedback graph: strongly observable graphs yield minimax regret of $\tilde{\Theta}( \alpha^{1/2} T^{1/2} )$, while weakly observable graphs induce minimax regret of $\tilde{\Theta}( \delta^{1/3} T^{2/3} )$, where $\alpha$ and $\delta$, respectively, represent the independence number of the graph and the domination number of a certain portion of the graph. Our proposed algorithm for strongly observable graphs has a regret bound of $\tilde{O}( \alpha^{1/2} T^{1/2} )$ for adversarial environments, as well as of  $ {O} ( \frac{\alpha (\ln T)^3 }{\Delta_{\min}} ) $ for stochastic environments, where $\Delta_{\min}$ expresses the minimum suboptimality gap. This result resolves an open question raised by Erez and Koren [2021]. We also provide an algorithm for weakly observable graphs that achieves a regret bound of $\tilde{O}( \delta^{1/3}T^{2/3} )$ for adversarial environments and poly-logarithmic regret for stochastic environments. The proposed algorithms are based on the follow-the-regularized-leader approach combined with newly designed update rules for learning rates.

        ----

        ## [2075] Few-shot Task-agnostic Neural Architecture Search for Distilling Large Language Models

        **Authors**: *Dongkuan Xu, Subhabrata Mukherjee, Xiaodong Liu, Debadeepta Dey, Wenhui Wang, Xiang Zhang, Ahmed Hassan Awadallah, Jianfeng Gao*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/b7c12689a89e98a61bcaa65285a41b7c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/b7c12689a89e98a61bcaa65285a41b7c-Abstract-Conference.html)

        **Abstract**:

        Traditional knowledge distillation (KD) methods manually design student architectures to compress large models given pre-specified computational cost. This requires several trials to find viable students, and repeating the process with change in computational budget. We use Neural Architecture Search (NAS) to automatically distill several compressed students with variable cost from a large model. Existing NAS methods train a single SuperLM consisting of millions of subnetworks with weight-sharing, resulting in interference between subnetworks of different sizes. Additionally, many of these works are task-specific requiring task labels for SuperLM training. Our framework AutoDistil addresses above challenges with the following steps: (a) Incorporates inductive bias and heuristics to partition Transformer search space into K compact sub-spaces (e.g., K=3 can generate typical student sizes of base, small and tiny); (b) Trains one SuperLM for each sub-space using task-agnostic objective (e.g., self-attention distillation) with weight-sharing of students; (c) Lightweight search for the optimal student without re-training. Task-agnostic training and search allow students to be reused for fine-tuning on any downstream task. Experiments on GLUE benchmark demonstrate AutoDistil to outperform state-of-the-art KD and NAS methods with upto 3x reduction in computational cost and negligible loss in task performance. Code and model checkpoints are available at https://github.com/microsoft/autodistil.

        ----

        ## [2076] Universal Rates for Interactive Learning

        **Authors**: *Steve Hanneke, Amin Karbasi, Shay Moran, Grigoris Velegkas*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/b8362385b08348d21162310c5b4e9541-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/b8362385b08348d21162310c5b4e9541-Abstract-Conference.html)

        **Abstract**:

        Consider the task of learning an unknown concept from a given concept class;  to what extent does interacting with a domain expert accelerate the learning process? It is common to measure the effectiveness of learning algorithms by plotting the "learning curve",  that is, the decay of the error rate as a function of the algorithm's resources (examples, queries, etc). Thus, the overarching question in this work is whether (and which kind of) interaction accelerates the learning curve. Previous work in interactive learning focused on uniform bounds on the learning rates which only capture the upper envelope of the learning curves over families of data distributions. We thus formalize our overarching question within the distribution dependent framework of universal learning, which aims to understand the performance of learning algorithms on every data distribution, but without requiring a single upper bound which applies uniformly to all distributions. Our main result reveals a fundamental trichotomy of interactive learning rates, thus providing a complete characterization of universal interactive learning. As a corollary we deduce a strong affirmative answer to our overarching question, showing that interaction is beneficial. Remarkably, we show that in important cases such benefits are realized with label queries, that is, by active learning algorithms. On the other hand, our lower bounds apply to arbitrary binary queries and, hence, they hold in any interactive learning setting.

        ----

        ## [2077] Learning Active Camera for Multi-Object Navigation

        **Authors**: *Peihao Chen, Dongyu Ji, Kunyang Lin, Weiwen Hu, Wenbing Huang, Thomas H. Li, Mingkui Tan, Chuang Gan*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/b84b9d1fe05c5e74d8f9466f063327a5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/b84b9d1fe05c5e74d8f9466f063327a5-Abstract-Conference.html)

        **Abstract**:

        Getting robots to navigate to multiple objects autonomously is essential yet difficult in robot applications. One of the key challenges is how to explore environments efficiently with camera sensors only. Existing navigation methods mainly focus on fixed cameras and few attempts have been made to navigate with active cameras. As a result, the agent may take a very long time to perceive the environment due to limited camera scope. In contrast, humans typically gain a larger field of view by looking around for a better perception of the environment. How to make robots perceive the environment as efficiently as humans is a fundamental problem in robotics. In this paper, we consider navigating to multiple objects more efficiently with active cameras. Specifically, we cast moving camera to a Markov Decision Process and reformulate the active camera problem as a reinforcement learning problem. However, we have to address two new challenges: 1) how to learn a good camera policy in complex environments and 2) how to coordinate it with the navigation policy. To address these, we carefully design a reward function to encourage the agent to explore more areas by moving camera actively. Moreover, we exploit human experience to infer a rule-based camera action to guide the learning process. Last, to better coordinate two kinds of policies, the camera policy takes navigation actions into account when making camera moving decisions. Experimental results show our camera policy consistently improves the performance of multi-object navigation over four baselines on two datasets.

        ----

        ## [2078] Knowledge Distillation: Bad Models Can Be Good Role Models

        **Authors**: *Gal Kaplun, Eran Malach, Preetum Nakkiran, Shai Shalev-Shwartz*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/b88edf805e96654a4f9e7b783e854ae3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/b88edf805e96654a4f9e7b783e854ae3-Abstract-Conference.html)

        **Abstract**:

        Large neural networks trained in the overparameterized regime are able to fit noise to zero train error. Recent work of Nakkiran and Bansal has empirically observed that such networks behave as “conditional samplers” from the noisy distribution. That is, they replicate the noise in the train data to unseen examples. We give a theoretical framework for studying this conditional sampling behavior in the context of learning theory. We relate the notion of such samplers to knowledge distillation, where a student network imitates the outputs of a teacher on unlabeled data. We show that samplers, while being bad classifiers, can be good teachers. Concretely, we prove that distillation from samplers is guaranteed to produce a student which approximates the Bayes optimal classifier. Finally, we show that some common learning algorithms (e.g., Nearest-Neighbours and Kernel Machines) can often generate samplers when applied in the overparameterized regime.

        ----

        ## [2079] On Computing Probabilistic Explanations for Decision Trees

        **Authors**: *Marcelo Arenas, Pablo Barceló, Miguel A. Romero Orth, Bernardo Subercaseaux*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/b8963f6a0a72e686dfa98ac3e7260f73-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/b8963f6a0a72e686dfa98ac3e7260f73-Abstract-Conference.html)

        **Abstract**:

        Formal XAI (explainable AI) is a growing area that focuses on computing explanations with mathematical guarantees for the decisions made by ML models. Inside formal XAI, one of the most studied cases is that of explaining the choices taken by decision trees, as they are traditionally deemed as one of the most interpretable classes of models. Recent work has focused on studying the computation of sufficient reasons, a kind of explanation in which given a decision tree $T$ and an instance $x$, one explains the decision $T(x)$ by providing a subset $y$ of the features of $x$ such that for any other instance $z$ compatible with $y$, it holds that  $T(z) = T(x)$, intuitively meaning that the features in $y$ are already enough to fully justify the classification of $x$ by $T$. It has been argued, however, that sufficient reasons constitute a restrictive notion of explanation. For such a reason, the community has started to study their probabilistic counterpart, in which one requires that the probability of $T(z) = T(x)$ must be at least some value $\delta \in (0, 1]$, where $z$ is a random instance that is compatible with $y$. Our paper settles the computational complexity of $\delta$-sufficient-reasons over decision trees, showing that both (1) finding $\delta$-sufficient-reasons  that are minimal in size, and (2) finding $\delta$-sufficient-reasons that are minimal inclusion-wise, do not admit polynomial-time algorithms (unless P = NP).   This is in stark contrast with the deterministic case ($\delta = 1$) where inclusion-wise minimal sufficient-reasons are easy to compute. By doing this, we answer two open problems originally raised by Izza et al., and extend the hardness of explanations for Boolean circuits presented by W{\"a}ldchen et al. to the more restricted case of decision trees. On the positive side, we identify structural restrictions of decision trees that make the problem tractable, and show how SAT solvers might be able to tackle these problems in practical settings.

        ----

        ## [2080] Masked Autoencoders that Listen

        **Authors**: *Po-Yao Huang, Hu Xu, Juncheng Li, Alexei Baevski, Michael Auli, Wojciech Galuba, Florian Metze, Christoph Feichtenhofer*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/b89d5e209990b19e33b418e14f323998-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/b89d5e209990b19e33b418e14f323998-Abstract-Conference.html)

        **Abstract**:

        This paper studies a simple extension of image-based Masked Autoencoders (MAE) to self-supervised representation learning from audio spectrograms. Following the Transformer encoder-decoder design in MAE, our Audio-MAE first encodes audio spectrogram patches with a high masking ratio, feeding only the non-masked tokens through encoder layers. The decoder then re-orders and decodes the encoded context padded with mask tokens, in order to reconstruct the input spectrogram. We find it beneficial to incorporate local window attention in the decoder, as audio spectrograms are highly correlated in local time and frequency bands. We then fine-tune the encoder with a lower masking ratio on target datasets. Empirically, Audio-MAE sets new state-of-the-art performance on six audio and speech classification tasks, outperforming other recent models that use external supervised pre-training. Our code and models is available at https://github.com/facebookresearch/AudioMAE.

        ----

        ## [2081] AUTOMATA: Gradient Based Data Subset Selection for Compute-Efficient Hyper-parameter Tuning

        **Authors**: *KrishnaTeja Killamsetty, Guttu Sai Abhishek, Aakriti, Ganesh Ramakrishnan, Alexandre V. Evfimievski, Lucian Popa, Rishabh K. Iyer*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/b8ab7288e7d5aefc695175f22bbddead-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/b8ab7288e7d5aefc695175f22bbddead-Abstract-Conference.html)

        **Abstract**:

        Deep neural networks have seen great success in recent years; however, training a deep model is often challenging as its performance heavily depends on the hyper-parameters used. In addition, finding the optimal hyper-parameter configuration, even with state-of-the-art (SOTA) hyper-parameter optimization (HPO) algorithms, can be time-consuming, requiring multiple training runs over the entire datasetfor different possible sets of hyper-parameters. Our central insight is that using an informative subset of the dataset for model training runs involved in hyper-parameter optimization, allows us to find the optimal hyper-parameter configuration significantly faster. In this work, we propose AUTOMATA, a gradient-based subset selection framework for hyper-parameter tuning. We empirically evaluate the effectiveness of AUTOMATA in hyper-parameter tuning through several experiments on real-world datasets in the text, vision, and tabular domains. Our experiments show that using gradient-based data subsets for hyper-parameter tuning achieves significantly faster turnaround times and speedups of 3×-30× while achieving comparable performance to the hyper-parameters found using the entire dataset.

        ----

        ## [2082] DeepTOP: Deep Threshold-Optimal Policy for MDPs and RMABs

        **Authors**: *Khaled Nakhleh, I-Hong Hou*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/b8bf2c0dd0b48511889b7d3b2c5fc8f5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/b8bf2c0dd0b48511889b7d3b2c5fc8f5-Abstract-Conference.html)

        **Abstract**:

        We consider the problem of learning the optimal threshold policy for control problems. Threshold policies make control decisions by evaluating whether an element of the system state exceeds a certain threshold, whose value is determined by other elements of the system state. By leveraging the monotone property of threshold policies, we prove that their policy gradients have a surprisingly simple expression. We use this simple expression to build an off-policy actor-critic algorithm for learning the optimal threshold policy. Simulation results show that our policy significantly outperforms other reinforcement learning algorithms due to its ability to exploit the monotone property.In addition, we show that the Whittle index, a powerful tool for restless multi-armed bandit problems, is equivalent to the optimal threshold policy for an alternative problem. This observation leads to a simple algorithm that finds the Whittle index by learning the optimal threshold policy in the alternative problem. Simulation results show that our algorithm learns the Whittle index much faster than several recent studies that learn the Whittle index through indirect means.

        ----

        ## [2083] Is a Modular Architecture Enough?

        **Authors**: *Sarthak Mittal, Yoshua Bengio, Guillaume Lajoie*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/b8d1d741f137d9b6ac4f3c1683791e4a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/b8d1d741f137d9b6ac4f3c1683791e4a-Abstract-Conference.html)

        **Abstract**:

        Inspired from human cognition, machine learning systems are gradually revealing advantages of sparser and more modular architectures. Recent work demonstrates that not only do some modular architectures generalize well, but they also lead to better out of distribution generalization, scaling properties, learning speed, and interpretability. A key intuition behind the success of such systems is that the data generating system for most real-world settings is considered to consist of sparse modular connections, and endowing models with similar inductive biases will be helpful. However, the field has been lacking in a rigorous quantitative assessment of such systems because these real-world data distributions are complex and unknown. In this work, we provide a thorough assessment of common modular architectures, through the lens of simple and known modular data distributions. We highlight the benefits of modularity and sparsity and reveal insights on the challenges faced while optimizing modular systems. In doing so, we propose evaluation metrics that highlight the benefits of modularity, the regimes in which these benefits are substantial, as well as the sub-optimality of current end-to-end learned modular systems as opposed to their claimed potential.

        ----

        ## [2084] Exploration via Planning for Information about the Optimal Trajectory

        **Authors**: *Viraj Mehta, Ian Char, Joseph Abbate, Rory Conlin, Mark D. Boyer, Stefano Ermon, Jeff Schneider, Willie Neiswanger*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/b90cb10d4dae058dd167388e76168c1b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/b90cb10d4dae058dd167388e76168c1b-Abstract-Conference.html)

        **Abstract**:

        Many potential applications of reinforcement learning (RL) are stymied by the large numbers of samples required to learn an effective policy. This is especially true when applying RL to real-world control tasks, e.g. in the sciences or robotics, where executing a policy in the environment is costly. In popular RL algorithms, agents typically explore either by adding stochasticity to a reward-maximizing policy or by attempting to gather maximal information about environment dynamics without taking the given task into account. In this work, we develop a method that allows us to plan for exploration while taking both the task and the current knowledge about the dynamics into account.  The key insight to our approach is to plan an action sequence that maximizes the expected information gain about the optimal trajectory for the task at hand. We demonstrate that our method learns strong policies with 2x fewer samples than strong exploration baselines and 200x fewer samples than model free methods on a diverse set of low-to-medium dimensional control tasks in both the open-loop and closed-loop control settings.

        ----

        ## [2085] Subquadratic Kronecker Regression with Applications to Tensor Decomposition

        **Authors**: *Matthew Fahrbach, Gang Fu, Mehrdad Ghadiri*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/b9121bbb3112975d33c527f046ae68f2-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/b9121bbb3112975d33c527f046ae68f2-Abstract-Conference.html)

        **Abstract**:

        Kronecker regression is a highly-structured least squares problem $\min_{\mathbf{x}} \lVert \mathbf{K}\mathbf{x} - \mathbf{b} \rVert_{2}^2$, where the design matrix $\mathbf{K} = \mathbf{A}^{(1)} \otimes \cdots \otimes \mathbf{A}^{(N)}$ is a Kronecker product of factor matrices. This regression problem arises in each step of the widely-used alternating least squares (ALS) algorithm for computing the Tucker decomposition of a tensor. We present the first subquadratic-time algorithm for solving Kronecker regression to a $(1+\varepsilon)$-approximation that avoids the exponential term $O(\varepsilon^{-N})$ in the running time. Our techniques combine leverage score sampling and iterative methods. By extending our approach to block-design matrices where one block is a Kronecker product, we also achieve subquadratic-time algorithms for (1) Kronecker ridge regression and (2) updating the factor matrix of a Tucker decomposition in ALS, which is not a pure Kronecker regression problem, thereby improving the running time of all steps of Tucker ALS. We demonstrate the speed and accuracy of this Kronecker regression algorithm on synthetic data and real-world image tensors.

        ----

        ## [2086] Robust Anytime Learning of Markov Decision Processes

        **Authors**: *Marnix Suilen, Thiago D. Simão, David Parker, Nils Jansen*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/b931c44c35ce09e942edab7003eb3daa-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/b931c44c35ce09e942edab7003eb3daa-Abstract-Conference.html)

        **Abstract**:

        Markov decision processes (MDPs) are formal models commonly used in sequential decision-making. MDPs capture the stochasticity that may arise, for instance, from imprecise actuators via probabilities in the transition function. However, in data-driven applications, deriving precise probabilities from (limited) data introduces statistical errors that may lead to unexpected or undesirable outcomes.Uncertain MDPs (uMDPs) do not require precise probabilities but instead use so-called uncertainty sets in the transitions, accounting for such limited data.Tools from the formal verification community efficiently compute robust policies that provably adhere to formal specifications, like safety constraints, under the worst-case instance in the uncertainty set. We continuously learn the transition probabilities of an MDP in a robust anytime-learning approach that combines a dedicated Bayesian inference scheme with the computation of robust policies. In particular, our method (1) approximates probabilities as intervals, (2) adapts to new data that may be inconsistent with an intermediate model, and (3) may be stopped at any time to compute a robust policy on the uMDP that faithfully captures the data so far. Furthermore, our method is capable of adapting to changes in the environment. We show the effectiveness of our approach and compare it to robust policies computed on uMDPs learned by the UCRL2 reinforcement learning algorithm in an experimental evaluation on several benchmarks.

        ----

        ## [2087] Discovering Design Concepts for CAD Sketches

        **Authors**: *Yuezhi Yang, Hao Pan*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/b9432d0f94275f0571c6cc99cf8b1664-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/b9432d0f94275f0571c6cc99cf8b1664-Abstract-Conference.html)

        **Abstract**:

        Sketch design concepts are recurring patterns found in parametric CAD sketches. Though rarely explicitly formalized by the CAD designers, these concepts are implicitly used in design for modularity and regularity. In this paper, we propose a learning based approach that discovers the modular concepts by induction over raw sketches. We propose the dual implicit-explicit representation of concept structures that allows implicit detection and explicit generation, and the separation of structure generation and parameter instantiation for parameterized concept generation, to learn modular concepts by end-to-end training. We demonstrate the design concept learning on a large scale CAD sketch dataset and show its applications for design intent interpretation and auto-completion.

        ----

        ## [2088] Learning Distributed and Fair Policies for Network Load Balancing as Markov Potential Game

        **Authors**: *Zhiyuan Yao, Zihan Ding*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/b94d8b035e2183e47afef9e2f299ba47-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/b94d8b035e2183e47afef9e2f299ba47-Abstract-Conference.html)

        **Abstract**:

        This paper investigates the network load balancing problem in data centers (DCs) where multiple load balancers (LBs) are deployed, using the multi-agent reinforcement learning (MARL) framework. The challenges of this problem consist of the heterogeneous processing architecture and dynamic environments, as well as limited and partial observability of each LB agent in distributed networking systems, which can largely degrade the performance of in-production load balancing algorithms in real-world setups. Centralised training and distributed execution (CTDE) RL scheme has been proposed to improve MARL performance, yet it incurs -- especially in distributed networking systems, which prefer distributed and plug-and-play design schemes -- additional communication and management overhead among agents. We formulate the multi-agent load balancing problem as a Markov potential game, with a carefully and properly designed workload distribution fairness as the potential function. A fully distributed MARL algorithm is proposed to approximate the Nash equilibrium of the game. Experimental evaluations involve both an event-driven simulator and a real-world system, where the proposed MARL load balancing algorithm shows close-to-optimal performance in simulations and superior results over in-production LBs in the real-world system.

        ----

        ## [2089] Para-CFlows: $C^k$-universal diffeomorphism approximators as superior neural surrogates

        **Authors**: *Junlong Lyu, Zhitang Chen, Chang Feng, Wenjing Cun, Shengyu Zhu, Yanhui Geng, Zhijie Xu, Chen Yongwei*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/b9523d484af624986c2e0c630ac44ecb-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/b9523d484af624986c2e0c630ac44ecb-Abstract-Conference.html)

        **Abstract**:

        Invertible neural networks based on Coupling Flows (CFlows) have various applications such as image synthesis and data compression. The approximation universality for CFlows is of paramount importance to ensure the model expressiveness. In this paper, we prove that CFlows}can approximate any diffeomorphism in $C^k$-norm if its layers can approximate certain single-coordinate transforms. Specifically, we derive that a composition of affine coupling layers and invertible linear transforms achieves this universality. Furthermore, in parametric cases where the diffeomorphism depends on some extra parameters, we prove the corresponding approximation theorems for parametric coupling flows named Para-CFlows. In practice, we apply Para-CFlows as a neural surrogate model in contextual Bayesian optimization tasks, to demonstrate its superiority over other neural surrogate models in terms of optimization performance and gradient approximations.

        ----

        ## [2090] NCP: Neural Correspondence Prior for Effective Unsupervised Shape Matching

        **Authors**: *Souhaib Attaiki, Maks Ovsjanikov*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/b95c7e24501f5d1dddbc5e8526cda7ae-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/b95c7e24501f5d1dddbc5e8526cda7ae-Abstract-Conference.html)

        **Abstract**:

        We present Neural Correspondence Prior (NCP), a new paradigm for computing correspondences between 3D shapes. Our approach is fully unsupervised and can lead to high quality correspondences even in challenging cases such as sparse point clouds or non-isometric meshes, where current methods fail. Our first key observation is that, in line with neural priors observed in other domains, recent network architectures on 3D data, even without training, tend to produce pointwise features that induce plausible maps between rigid or non-rigid shapes. Secondly, we show that given a noisy map as input, training a feature extraction network with the input map as supervision, tends to remove artifacts from the input and can act as a powerful correspondence denoising mechanism, both between individual pairs and within a collection. With  these observations in hand, we propose a two-stage unsupervised paradigm for shape matching, by (i) performing unsupervised training by adapting an existing approach to obtain an initial set of noisy matches, (ii) using these matches to train a network in a supervised manner. We demonstrate that this approach significantly improves the accuracy of the maps, especially when trained within a collection. We show that NCP is data-efficient, fast, and achieves state-of-the-art results on many tasks. Our code will be released after publication.

        ----

        ## [2091] Efficient Spatially Sparse Inference for Conditional GANs and Diffusion Models

        **Authors**: *Muyang Li, Ji Lin, Chenlin Meng, Stefano Ermon, Song Han, Jun-Yan Zhu*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/b9603de9e49d0838e53b6c9cf9d06556-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/b9603de9e49d0838e53b6c9cf9d06556-Abstract-Conference.html)

        **Abstract**:

        During image editing, existing deep generative models tend to re-synthesize the entire output from scratch, including the unedited regions. This leads to a significant waste of computation, especially for minor editing operations. In this work, we present Spatially Sparse Inference (SSI), a general-purpose technique that selectively performs computation for edited regions and accelerates various generative models, including both conditional GANs and diffusion models. Our key observation is that users tend to make gradual changes to the input image. This motivates us to cache and reuse the feature maps of the original image. Given an edited image, we sparsely apply the convolutional filters to the edited regions while reusing the cached features for the unedited regions. Based on our algorithm, we further propose Sparse Incremental Generative Engine (SIGE) to convert the computation reduction to latency reduction on off-the-shelf hardware. With 1.2%-area edited regions, our method reduces the computation of DDIM by $7.5\times$ and GauGAN by $18\times$  while preserving the visual fidelity. With SIGE, we accelerate the inference time of DDIM by $3.0\times$ on RTX 3090 and $6.6\times$ on Apple M1 Pro CPU, and GauGAN by $4.2\times$ on RTX 3090 and $14\times$ on Apple M1 Pro CPU.

        ----

        ## [2092] A Greek Parliament Proceedings Dataset for Computational Linguistics and Political Analysis

        **Authors**: *Konstantina Dritsa, Aikaterini Thoma, Ioannis Pavlopoulos, Panos Louridas*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/b96ce67b2f2d45e4ab315e13a6b5b9c5-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2022/hash/b96ce67b2f2d45e4ab315e13a6b5b9c5-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        Large, diachronic datasets of political discourse are hard to come across, especially for resource-lean languages such as Greek. In this paper, we introduce a curated dataset of the Greek Parliament Proceedings that extends chronologically from 1989 up to 2020. It consists of more than 1 million speeches with extensive meta-data, extracted from 5,355 parliamentary sitting record files. We explain how it was constructed and the challenges that had to be overcome. The dataset can be used for both computational linguistics and political analysis---ideally, combining the two. We present such an application, showing (i) how the dataset can be used to study the change of word usage through time, (ii) between significant historical events and political parties, (iii) by evaluating and employing algorithms for detecting semantic shifts.

        ----

        ## [2093] Multi-objective Deep Data Generation with Correlated Property Control

        **Authors**: *Shiyu Wang, Xiaojie Guo, Xuanyang Lin, Bo Pan, Yuanqi Du, Yinkai Wang, Yanfang Ye, Ashley Ann Petersen, Austin Leitgeb, Saleh AlKhalifa, Kevin Minbiole, William M. Wuest, Amarda Shehu, Liang Zhao*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/b9c2e8a0bbed5fcfaf62856a3a719ada-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/b9c2e8a0bbed5fcfaf62856a3a719ada-Abstract-Conference.html)

        **Abstract**:

        Developing deep generative models has been an emerging field due to the ability to model and generate complex data for various purposes, such as image synthesis and molecular design. However, the advance of deep generative models is limited by the challenges to generate objects that possess multiple desired properties because: 1) the existence of complex correlation among real-world properties is common but hard to identify; 2) controlling individual property enforces an implicit partially control of its correlated properties, which is difficult to model; 3) controlling multiple properties under variour manners simultaneously is hard and underexplored. We address these challenges by proposing a novel deep generative framework that recovers semantics and correlation of properties through disentangled latent vectors. The correlation is handled via an explainable mask pooling layer, and properties are precisely retained by the generated objects via the mutual dependence between latent vectors and properties. Our generative model preserves properties of interest while handles correlation and conflicts of properties under a multi-objective optimization framework. The experiments demonstrate our model's superior performance in generating objects with desired properties.

        ----

        ## [2094] Domain Adaptation meets Individual Fairness And they get along

        **Authors**: *Debarghya Mukherjee, Felix Petersen, Mikhail Yurochkin, Yuekai Sun*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/b9e0ceee9751ae8b5c6603c029e4ca42-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/b9e0ceee9751ae8b5c6603c029e4ca42-Abstract-Conference.html)

        **Abstract**:

        Many instances of algorithmic bias are caused by distributional shifts. For example, machine learning (ML) models often perform worse on demographic groups that are underrepresented in the training data. In this paper, we leverage this connection between algorithmic fairness and distribution shifts to show that algorithmic fairness interventions can help ML models overcome distribution shifts, and that domain adaptation methods (for overcoming distribution shifts) can mitigate algorithmic biases. In particular, we show that (i) enforcing suitable notions of individual fairness (IF) can improve the out-of-distribution accuracy of ML models under the covariate shift assumption and that (ii) it is possible to adapt representation alignment methods for domain adaptation to enforce individual fairness. The former is unexpected because IF interventions were not developed with distribution shifts in mind. The latter is also unexpected because representation alignment is not a common approach in the individual fairness literature.

        ----

        ## [2095] An Asymptotically Optimal Batched Algorithm for the Dueling Bandit Problem

        **Authors**: *Arpit Agarwal, Rohan Ghuge, Viswanath Nagarajan*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/b9e472cd579c83e2f6aa3459f46aac28-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/b9e472cd579c83e2f6aa3459f46aac28-Abstract-Conference.html)

        **Abstract**:

        We study the $K$-armed dueling bandit problem, a variation of the traditional multi-armed bandit problem in which feedback is obtained in the form of pairwise comparisons. Previous learning algorithms have focused on the fully adaptive setting, where the algorithm can make updates after every comparison. The "batched" dueling bandit problem is motivated by large-scale applications like web search ranking and recommendation systems, where  performing sequential updates may be infeasible. In this work, we ask: is there a solution using only a few adaptive rounds that matches the asymptotic regret bounds of the best sequential algorithms for $K$-armed dueling bandits? We answer this in the affirmative under the Condorcet condition, a standard setting of the $K$-armed dueling bandit problem. We obtain asymptotic regret of $O(K^2\log^2(K))$ + $O(K\log(T))$ in $O(\log(T))$ rounds, where $T$ is the time horizon. Our regret bounds nearly match  the best regret bounds known in the fully sequential setting under the Condorcet condition. Finally, in computational experiments  over a variety of real-world datasets, we observe that our algorithm using $O(\log(T))$ rounds achieves almost the same performance as fully sequential algorithms (that use $T$ rounds).

        ----

        ## [2096] Enhanced Bilevel Optimization via Bregman Distance

        **Authors**: *Feihu Huang, Junyi Li, Shangqian Gao, Heng Huang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/b9e98316cb72fee82cc1160da5810abc-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/b9e98316cb72fee82cc1160da5810abc-Abstract-Conference.html)

        **Abstract**:

        Bilevel optimization has been recently used in many machine learning problems such as hyperparameter optimization, policy optimization, and meta learning. Although many bilevel optimization methods have been proposed, they still suffer from the high computational complexities and do not consider the more general bilevel problems with nonsmooth regularization. In the paper, thus, we propose a class of enhanced bilevel optimization methods with using Bregman distance to solve bilevel optimization problems, where the outer subproblem is nonconvex and possibly nonsmooth, and the inner subproblem is strongly convex. Specifically, we propose a bilevel optimization method based on Bregman distance (BiO-BreD) to solve deterministic bilevel problems, which achieves a lower computational complexity than the best known results. Meanwhile, we also propose a stochastic bilevel optimization method (SBiO-BreD) to solve stochastic bilevel problems based on stochastic approximated gradients and Bregman distance. Moreover, we further propose an accelerated version of SBiO-BreD method (ASBiO-BreD) using the variance-reduced technique, which can achieve a lower computational complexity than the best known computational complexities with respect to condition number $\kappa$ and target accuracy $\epsilon$ for finding an $\epsilon$-stationary point. We conduct data hyper-cleaning task and hyper-representation learning task to demonstrate that our new algorithms outperform related bilevel optimization approaches.

        ----

        ## [2097] SAVi++: Towards End-to-End Object-Centric Learning from Real-World Videos

        **Authors**: *Gamaleldin F. Elsayed, Aravindh Mahendran, Sjoerd van Steenkiste, Klaus Greff, Michael C. Mozer, Thomas Kipf*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/ba1a6ba05319e410f0673f8477a871e3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/ba1a6ba05319e410f0673f8477a871e3-Abstract-Conference.html)

        **Abstract**:

        The visual world can be parsimoniously characterized in terms of distinct entities with sparse interactions. Discovering this compositional structure in dynamic visual scenes has proven challenging for end-to-end computer vision approaches unless explicit instance-level supervision is provided. Slot-based models leveraging motion cues have recently shown great promise in learning to represent, segment, and track objects without direct supervision, but they still fail to scale to complex real-world multi-object videos. In an effort to bridge this gap, we take inspiration from human development and hypothesize that information about scene geometry in the form of depth signals can facilitate object-centric learning. We introduce SAVi++, an object-centric video model which is trained to predict depth signals from a slot-based video representation. By further leveraging best practices for model scaling, we are able to train SAVi++ to segment complex dynamic scenes recorded with moving cameras, containing both static and moving objects of diverse appearance on naturalistic backgrounds, without the need for segmentation supervision. Finally, we demonstrate that by using sparse depth signals obtained from LiDAR, SAVi++ is able to learn emergent object segmentation and tracking from videos in the real-world Waymo Open dataset.

        ----

        ## [2098] Reincarnating Reinforcement Learning: Reusing Prior Computation to Accelerate Progress

        **Authors**: *Rishabh Agarwal, Max Schwarzer, Pablo Samuel Castro, Aaron C. Courville, Marc G. Bellemare*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/ba1c5356d9164bb64c446a4b690226b0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/ba1c5356d9164bb64c446a4b690226b0-Abstract-Conference.html)

        **Abstract**:

        Learning tabula rasa, that is without any prior knowledge, is the prevalent workflow in reinforcement learning (RL) research. However, RL systems, when applied to large-scale settings, rarely operate tabula rasa. Such large-scale systems undergo multiple design or algorithmic changes during their development cycle and use ad hoc approaches for incorporating these changes without re-training from scratch, which would have been prohibitively expensive. Additionally, the inefficiency of deep RL typically excludes researchers without access to industrial-scale resources from tackling computationally-demanding problems. To address these issues, we present reincarnating RL as an alternative workflow or class of problem settings, where prior computational work (e.g., learned policies) is reused or transferred between design iterations of an RL agent, or from one RL agent to another. As a step towards enabling reincarnating RL from any agent to any other agent, we focus on the specific setting of efficiently transferring an existing sub-optimal policy to a standalone value-based RL agent. We find that existing approaches fail in this setting and propose a simple algorithm to address their limitations. Equipped with this algorithm, we demonstrate reincarnating RL's gains over tabula rasa RL on Atari 2600 games, a challenging locomotion task, and the real-world problem of navigating stratospheric balloons. Overall, this work argues for an alternative approach to RL research, which we believe could significantly improve real-world RL adoption and help democratize it further. Open-sourced code and trained agents at https://agarwl.github.io/reincarnating_rl.

        ----

        ## [2099] Global Linear and Local Superlinear Convergence of IRLS for Non-Smooth Robust Regression

        **Authors**: *Liangzu Peng, Christian Kümmerle, René Vidal*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/ba3354bcfeae4f166a8bfe75443ac8f7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/ba3354bcfeae4f166a8bfe75443ac8f7-Abstract-Conference.html)

        **Abstract**:

        We advance both the theory and practice of robust $\ell_p$-quasinorm regression for $p \in (0,1]$ by using novel variants of iteratively reweighted least-squares (IRLS) to solve the underlying non-smooth problem. In the convex case, $p=1$, we prove that this IRLS variant converges globally at a linear rate under a mild, deterministic condition on the feature matrix called the stable range space property. In the non-convex case, $p\in(0,1)$, we prove that under a similar condition, IRLS converges locally to the global minimizer at a superlinear rate of order $2-p$; the rate becomes quadratic as $p\to 0$. We showcase the proposed methods in three applications: real phase retrieval, regression without correspondences, and robust face restoration. The results show that (1) IRLS can handle a larger number of outliers than other methods, (2) it is faster than competing methods at the same level of accuracy, (3) it restores a sparsely corrupted face image with satisfactory visual quality.

        ----

        ## [2100] Rashomon Capacity: A Metric for Predictive Multiplicity in Classification

        **Authors**: *Hsiang Hsu, Flávio P. Calmon*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/ba4caa85ecdcafbf9102ab8ec384182d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/ba4caa85ecdcafbf9102ab8ec384182d-Abstract-Conference.html)

        **Abstract**:

        Predictive multiplicity occurs when classification models with statistically indistinguishable performances assign conflicting predictions to individual samples. When used for decision-making in applications of consequence (e.g., lending, education, criminal justice), models developed without regard for predictive multiplicity may result in unjustified and arbitrary decisions for specific individuals. We introduce a new metric, called Rashomon Capacity, to measure predictive multiplicity in probabilistic classification. Prior metrics for predictive multiplicity focus on classifiers that output thresholded (i.e., 0-1) predicted classes. In contrast, Rashomon Capacity applies to probabilistic classifiers, capturing more nuanced score variations for individual samples. We provide a rigorous derivation for Rashomon Capacity, argue its intuitive appeal, and demonstrate how to estimate it in practice. We show that Rashomon Capacity yields principled strategies for disclosing conflicting models to stakeholders. Our numerical experiments illustrate how Rashomon Capacity captures predictive multiplicity in various datasets and learning models, including neural networks. The tools introduced in this paper can help data scientists measure and report predictive multiplicity prior to model deployment.

        ----

        ## [2101] Inducing Equilibria via Incentives: Simultaneous Design-and-Play Ensures Global Convergence

        **Authors**: *Boyi Liu, Jiayang Li, Zhuoran Yang, Hoi-To Wai, Mingyi Hong, Yu Nie, Zhaoran Wang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/ba5f85ce126aad12075a3ffa68a3e969-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/ba5f85ce126aad12075a3ffa68a3e969-Abstract-Conference.html)

        **Abstract**:

        To regulate a social system comprised of self-interested agents, economic incentives are often required to induce a desirable outcome. This incentive design problem naturally possesses a bilevel structure, in which a designer modifies the payoffs of the agents with incentives while anticipating the response of the agents, who play a non-cooperative game that converges to an equilibrium. The existing bilevel optimization algorithms raise a dilemma when applied to this problem: anticipating how incentives affect the agents at equilibrium requires solving the equilibrium problem repeatedly, which is computationally inefficient; bypassing the time-consuming step of equilibrium-finding can reduce the computational cost, but may lead the designer to a sub-optimal solution. To address such a dilemma, we propose a method that tackles the designer’s and agents’ problems simultaneously in a single loop.  Specifically, at each iteration, both the designer and the agents only move one step. Nevertheless, we allow the designer to gradually learn the overall influence of the incentives on the agents, which guarantees optimality after convergence. The convergence rate of the proposed scheme is also established for a broad class of games.

        ----

        ## [2102] Learning with convolution and pooling operations in kernel methods

        **Authors**: *Theodor Misiakiewicz, Song Mei*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/ba8aee784ffe0813890288b334444eda-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/ba8aee784ffe0813890288b334444eda-Abstract-Conference.html)

        **Abstract**:

        Recent empirical work has shown that hierarchical convolutional kernels inspired by convolutional neural networks (CNNs) signiﬁcantly improve the performance of kernel methods in image classiﬁcation tasks. A widely accepted explanation for their success is that these architectures encode hypothesis classes that are suitable for natural images. However, understanding the precise interplay between approximation and generalization in convolutional architectures remains a challenge. In this paper, we consider the stylized setting of covariates (image pixels) uniformly distributed on the hypercube, and characterize exactly the RKHS of kernels composed of single layers of convolution, pooling, and downsampling operations. We use this characterization to compute sharp asymptotics of the generalization error for any given function in high-dimension. In particular, we quantify the gain in sample complexity brought by enforcing locality with the convolution operation and approximate translation invariance with average pooling. Notably, these results provide a precise description of how convolution and pooling operations trade off approximation with generalization power in one layer convolutional kernels.

        ----

        ## [2103] In Differential Privacy, There is Truth: on Vote-Histogram Leakage in Ensemble Private Learning

        **Authors**: *Jiaqi Wang, Roei Schuster, Ilia Shumailov, David Lie, Nicolas Papernot*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/ba8d1b46292c5e82cbfb3b3dc3b968af-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/ba8d1b46292c5e82cbfb3b3dc3b968af-Abstract-Conference.html)

        **Abstract**:

        When learning from sensitive data, care must be taken to ensure that training algorithms address privacy concerns. The canonical Private Aggregation of Teacher Ensembles, or PATE, computes output labels by aggregating the predictions of a (possibly distributed) collection of teacher models via a voting mechanism. The mechanism adds noise to attain a differential privacy guarantee with respect to the teachers' training data. In this work, we observe that this use of noise, which makes PATE predictions stochastic, enables new forms of leakage of sensitive information. For a given input, our adversary exploits this stochasticity to extract high-fidelity histograms of the votes submitted by the underlying teachers. From these histograms, the adversary can learn sensitive attributes of the input such as race, gender, or age. Although this attack does not directly violate the differential privacy guarantee, it clearly violates privacy norms and expectations, and would not be possible $\textit{at all}$ without the noise inserted to obtain differential privacy. In fact, counter-intuitively, the attack $\textbf{becomes easier as we add more noise}$ to provide stronger differential privacy. We hope this encourages future work to consider privacy holistically rather than treat differential privacy as a panacea.

        ----

        ## [2104] Estimating graphical models for count data with applications to single-cell gene network

        **Authors**: *Feiyi Xiao, Junjie Tang, Huaying Fang, Ruibin Xi*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/ba92705991cfbbcedc26e27e833ebbae-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/ba92705991cfbbcedc26e27e833ebbae-Abstract-Conference.html)

        **Abstract**:

        Graphical models such as Gaussian graphical models have been widely applied for direct interaction inference in many different areas. In many modern applications, such as single-cell RNA sequencing (scRNA-seq) studies, the observed data are counts and often contain many small counts.  Traditional graphical models for continuous data are inappropriate for network inference of count data. We consider the Poisson log-normal (PLN) graphical model for count data and the precision matrix of the latent normal distribution represents the network. We propose a two-step method PLNet to estimate the precision matrix. PLNet first estimates the latent covariance matrix using the maximum marginal likelihood estimator (MMLE) and then estimates the precision matrix by minimizing the lasso-penalized D-trace loss function. We establish the convergence rate of the MMLE of the covariance matrix and further establish the convergence rate and the sign consistency of the proposed PLNet estimator of the precision matrix in the high dimensional setting. Importantly, although the PLN model is not sub-Gaussian, we show that the PLNet estimator is consistent even if the model dimension goes to infinity exponentially as the sample size increases. The performance of PLNet is evaluated and compared with available methods using simulation and gene regulatory network analysis of real scRNA-seq data.

        ----

        ## [2105] Online Minimax Multiobjective Optimization: Multicalibeating and Other Applications

        **Authors**: *Daniel Lee, Georgy Noarov, Mallesh M. Pai, Aaron Roth*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/ba942323c447c9bbb9d4b638eadefab9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/ba942323c447c9bbb9d4b638eadefab9-Abstract-Conference.html)

        **Abstract**:

        We introduce a simple but general online learning framework in which a learner plays against an adversary in a vector-valued game that changes every round. Even though the learner's objective is not convex-concave (and so the minimax theorem does not apply), we give a simple algorithm that can compete with the setting in which the adversary must announce their action first, with optimally diminishing regret. We demonstrate the power of our framework by using it to (re)derive optimal bounds and efficient algorithms across a variety of domains, ranging from multicalibration to a large set of no-regret algorithms, to a variant of Blackwell's approachability theorem for polytopes with fast convergence rates. As a new application, we show how to ``(multi)calibeat'' an arbitrary collection of forecasters --- achieving an exponentially improved dependence on the number of models we are competing against, compared to prior work.

        ----

        ## [2106] Roadblocks for Temporarily Disabling Shortcuts and Learning New Knowledge

        **Authors**: *Hongjing Niu, Hanting Li, Feng Zhao, Bin Li*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/baaa7b5b5bbaadca5023e1ab909b8af5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/baaa7b5b5bbaadca5023e1ab909b8af5-Abstract-Conference.html)

        **Abstract**:

        Deep learning models have been found with a tendency of relying on shortcuts, i.e., decision rules that perform well on standard benchmarks but fail when transferred to more challenging testing conditions. Such reliance may hinder deep learning models from learning other task-related features and seriously affect their performance and robustness. Although recent studies have shown some characteristics of shortcuts, there are few investigations on how to help the deep learning models to solve shortcut problems. This paper proposes a framework to address this issue by setting up roadblocks on shortcuts. Specifically, roadblocks are placed when the model is urged to learn to complete a gently modified task to ensure that the learned knowledge, including shortcuts, is insufficient the complete the task. Therefore, the model trained on the modified task will no longer over-rely on shortcuts. Extensive experiments demonstrate that the proposed framework significantly improves the training of networks on both synthetic and real-world datasets in terms of both classification accuracy and feature diversity. Moreover, the visualization results show that the mechanism behind the proposed our method is consistent with our expectations. In summary, our approach can effectively disable the shortcuts and thus learn more robust features.

        ----

        ## [2107] Random Rank: The One and Only Strategyproof and Proportionally Fair Randomized Facility Location Mechanism

        **Authors**: *Haris Aziz, Alexander Lam, Mashbat Suzuki, Toby Walsh*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/bac4d92b3f6decfe47eab9a5893dd1f6-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/bac4d92b3f6decfe47eab9a5893dd1f6-Abstract-Conference.html)

        **Abstract**:

        Proportionality is an attractive fairness concept that has been applied to a range of problems including the facility location problem, a classic problem in social choice. In our work, we propose a concept called Strong Proportionality, which ensures that when there are two groups of agents at different locations, both groups incur the same total cost. We show that although Strong Proportionality is a well-motivated and basic axiom, there is no deterministic strategyproof mechanism satisfying the property. We then identify a randomized mechanism called Random Rank (which uniformly selects a number $k$ between $1$ to $n$ and locates the facility at the $k$'th highest agent location) which satisfies Strong Proportionality in expectation. Our main theorem characterizes  Random Rank as the unique mechanism that achieves universal truthfulness, universal anonymity, and Strong Proportionality in expectation among all randomized mechanisms. Finally, we show via the AverageOrRandomRank mechanism that even stronger ex-post fairness guarantees can be achieved by weakening universal truthfulness to strategyproofness in expectation.

        ----

        ## [2108] Distributionally robust weighted k-nearest neighbors

        **Authors**: *Shixiang Zhu, Liyan Xie, Minghe Zhang, Rui Gao, Yao Xie*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/bb0d37c6210f84abfa0fd7709edb30cb-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/bb0d37c6210f84abfa0fd7709edb30cb-Abstract-Conference.html)

        **Abstract**:

        Learning a robust classifier from a few samples remains a key challenge in machine learning. A major thrust of research has been focused on developing k-nearest neighbor (k-NN) based algorithms combined with metric learning that captures similarities between samples. When the samples are limited, robustness is especially crucial to ensure the generalization capability of the classifier. In this paper, we study a minimax distributionally robust formulation of weighted k-nearest neighbors, which aims to find the optimal weighted k-NN classifiers that hedge against feature uncertainties. We develop an algorithm, Dr.k-NN, that efficiently solves this functional optimization problem and features in assigning minimax optimal weights to training samples when performing classification. These weights are class-dependent, and are determined by the similarities of sample features under the least favorable scenarios. When the size of the uncertainty set is properly tuned, the robust classifier has a smaller Lipschitz norm than the vanilla k-NN, and thus improves the generalization capability. We also couple our framework with neural-network-based feature embedding. We demonstrate the competitive performance of our algorithm compared to the state-of-the-art in the few-training-sample setting with various real-data experiments.

        ----

        ## [2109] A Character-Level Length-Control Algorithm for Non-Autoregressive Sentence Summarization

        **Authors**: *Puyuan Liu, Xiang Zhang, Lili Mou*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/bb0f9af6a4881ccb6e14c11b8b4be710-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/bb0f9af6a4881ccb6e14c11b8b4be710-Abstract-Conference.html)

        **Abstract**:

        Sentence summarization aims at compressing a long sentence into a short one that keeps the main gist, and has extensive real-world applications such as headline generation. In previous work, researchers have developed various approaches to improve the ROUGE score, which is the main evaluation metric for summarization, whereas controlling the summary length has not drawn much attention. In our work, we address a new problem of explicit character-level length control for summarization, and propose a dynamic programming algorithm based on the Connectionist Temporal Classification (CTC) model. Results show that our approach not only achieves higher ROUGE scores but also yields more complete sentences.

        ----

        ## [2110] Neural Lyapunov Control of Unknown Nonlinear Systems with Stability Guarantees

        **Authors**: *Ruikun Zhou, Thanin Quartz, Hans De Sterck, Jun Liu*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/bba3160c42f4c1d23175b5abf04347f5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/bba3160c42f4c1d23175b5abf04347f5-Abstract-Conference.html)

        **Abstract**:

        Learning for control of dynamical systems with formal guarantees remains a challenging task. This paper proposes a learning framework to simultaneously stabilize an unknown nonlinear system with a neural controller and learn a neural Lyapunov function to certify a region of attraction (ROA) for the closed-loop system with provable guarantees. The algorithmic structure consists of two neural networks and a satisfiability modulo theories (SMT) solver. The first neural network is responsible for learning the unknown dynamics. The second neural network aims to identify a valid Lyapunov function and a provably stabilizing nonlinear controller. The SMT solver verifies the candidate Lyapunov function satisfies the Lyapunov conditions. We further provide theoretical guarantees of the proposed learning framework and show that the obtained Lyapunov function indeed verifies for the unknown nonlinear system under mild assumptions. We illustrate the effectiveness of the results with a few numerical experiments.

        ----

        ## [2111] Minimax Regret for Cascading Bandits

        **Authors**: *Daniel Vial, Sujay Sanghavi, Sanjay Shakkottai, R. Srikant*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/bbb397739546de027bc81a2c2a8fb119-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/bbb397739546de027bc81a2c2a8fb119-Abstract-Conference.html)

        **Abstract**:

        Cascading bandits is a natural and popular model that frames the task of learning to rank from Bernoulli click feedback in a bandit setting. For the case of unstructured rewards, we prove matching upper and lower bounds for the problem-independent (i.e., gap-free) regret, both of which strictly improve the best known. A key observation is that the hard instances of this problem are those with small mean rewards, i.e., the small click-through rates that are most relevant in practice. Based on this, and the fact that small mean implies small variance for Bernoullis, our key technical result shows that variance-aware confidence sets derived from the Bernstein and Chernoff bounds lead to optimal algorithms (up to log terms), whereas Hoeffding-based algorithms suffer order-wise suboptimal regret. This sharply contrasts with the standard (non-cascading) bandit setting, where the variance-aware algorithms only improve constants. In light of this and as an additional contribution, we propose a variance-aware algorithm for the structured case of linear rewards and show its regret strictly improves the state-of-the-art.

        ----

        ## [2112] Mean Estimation with User-level Privacy under Data Heterogeneity

        **Authors**: *Rachel Cummings, Vitaly Feldman, Audra McMillan, Kunal Talwar*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/bbba680acd2826f23928c6675a19f0e7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/bbba680acd2826f23928c6675a19f0e7-Abstract-Conference.html)

        **Abstract**:

        A key challenge in many modern data analysis tasks is that user data is heterogeneous. Different users may possess vastly different numbers of data points. More importantly, it cannot be assumed that all users sample from the same underlying distribution.  This is true, for example in language data, where different speech styles result in data heterogeneity. In this work we propose a simple model of heterogeneous user data that differs in both distribution and quantity of data, and we provide a method for estimating the population-level mean while preserving user-level differential privacy. We demonstrate asymptotic optimality of our estimator and also prove general lower bounds on the error achievable in our problem.

        ----

        ## [2113] Proppo: a Message Passing Framework for Customizable and Composable Learning Algorithms

        **Authors**: *Paavo Parmas, Takuma Seno*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/bbc9d480a8257889d2af88983e8b126a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/bbc9d480a8257889d2af88983e8b126a-Abstract-Conference.html)

        **Abstract**:

        While existing automatic differentiation (AD) frameworks allow flexibly composing model architectures, they do not provide the same flexibility for composing learning algorithms---everything has to be implemented in terms of back propagation. To address this gap, we invent Automatic Propagation (AP) software, which generalizes AD, and allows custom and composable construction of complex learning algorithms. The framework allows packaging custom learning algorithms into propagators that automatically implement the necessary computations, and can be reused across different computation graphs. We implement Proppo, a prototype AP software package built on top of the Pytorch AD framework. To demonstrate the utility of Proppo, we use it to implement Monte Carlo gradient estimation techniques, such as reparameterization and likelihood ratio gradients, as well as the total propagation algorithm and Gaussian shaping gradients, which were previously used in model-based reinforcement learning, but do not have any publicly available implementation. Finally, in minimalistic experiments, we show that these methods allow increasing the gradient accuracy by orders of magnitude, particularly when the machine learning system is at the edge of chaos.

        ----

        ## [2114] A Lower Bound of Hash Codes' Performance

        **Authors**: *Xiaosu Zhu, Jingkuan Song, Yu Lei, Lianli Gao, Hengtao Shen*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/bbd7d8bd780fcf7143add2317ba04638-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/bbd7d8bd780fcf7143add2317ba04638-Abstract-Conference.html)

        **Abstract**:

        As a crucial approach for compact representation learning, hashing has achieved great success in effectiveness and efficiency. Numerous heuristic Hamming space metric learning objectives are designed to obtain high-quality hash codes. Nevertheless, a theoretical analysis of criteria for learning good hash codes remains largely unexploited. In this paper, we prove that inter-class distinctiveness and intra-class compactness among hash codes determine the lower bound of hash codes' performance. Promoting these two characteristics could lift the bound and improve hash learning. We then propose a surrogate model to fully exploit the above objective by estimating the posterior of hash codes and controlling it, which results in a low-bias optimization. Extensive experiments reveal the effectiveness of the proposed method. By testing on a series of hash-models, we obtain performance improvements among all of them, with an up to $26.5\%$ increase in mean Average Precision and an up to $20.5\%$ increase in accuracy. Our code is publicly available at https://github.com/VL-Group/LBHash.

        ----

        ## [2115] Self-Supervised Image Restoration with Blurry and Noisy Pairs

        **Authors**: *Zhilu Zhang, Rongjian Xu, Ming Liu, Zifei Yan, Wangmeng Zuo*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/bc12914d66b41b6bfc2d3a5decdb498b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/bc12914d66b41b6bfc2d3a5decdb498b-Abstract-Conference.html)

        **Abstract**:

        When taking photos under an environment with insufficient light, the exposure time and the sensor gain usually require to be carefully chosen to obtain images with satisfying visual quality. For example, the images with high ISO usually have inescapable noise, while the long-exposure ones may be blurry due to camera shake or object motion. Existing solutions generally suggest to seek a balance between noise and blur, and learn denoising or deblurring models under either full- or self-supervision. However, the real-world training pairs are difficult to collect, and the self-supervised methods merely rely on blurry or noisy images are limited in performance. In this work, we tackle this problem by jointly leveraging the short-exposure noisy image and the long-exposure blurry image for better image restoration. Such setting is practically feasible due to that short-exposure and long-exposure images can be either acquired by two individual cameras or synthesized by a long burst of images. Moreover, the short-exposure images are hardly blurry, and the long-exposure ones have negligible noise. Their complementarity makes it feasible to learn restoration model in a self-supervised manner. Specifically, the noisy images can be used as the supervision information for deblurring, while the sharp areas in the blurry images can be utilized as the auxiliary supervision information for self-supervised denoising. By learning in a collaborative manner, the deblurring and denoising tasks in our method can benefit each other. Experiments on synthetic and real-world images show the effectiveness and practicality of the proposed method. Codes are available at https://github.com/cszhilu1998/SelfIR.

        ----

        ## [2116] Embracing Consistency: A One-Stage Approach for Spatio-Temporal Video Grounding

        **Authors**: *Yang Jin, Yongzhi Li, Zehuan Yuan, Yadong Mu*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/bc18c538d983cea434f9281148d43e1e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/bc18c538d983cea434f9281148d43e1e-Abstract-Conference.html)

        **Abstract**:

        Spatio-Temporal video grounding (STVG) focuses on retrieving the spatio-temporal tube of a specific object depicted by a free-form textual expression. Existing approaches mainly treat this complicated task as a parallel frame-grounding problem and thus suffer from two types of inconsistency drawbacks: feature alignment inconsistency and prediction inconsistency. In this paper, we present an end-to-end one-stage framework, termed Spatio-Temporal Consistency-Aware Transformer (STCAT), to alleviate these issues. Specially, we introduce a novel multi-modal template as the global objective to address this task, which explicitly constricts the grounding region and associates the predictions among all video frames. Moreover, to generate the above template under sufficient video-textual perception, an encoder-decoder architecture is proposed for effective global context modeling. Thanks to these critical designs, STCAT enjoys more consistent cross-modal feature alignment and tube prediction without reliance on any pre-trained object detectors. Extensive experiments show that our method outperforms previous state-of-the-arts with clear margins on two challenging video benchmarks (VidSTG and HC-STVG), illustrating the superiority of the proposed framework to better understanding the association between vision and natural language. Code is publicly available at https://github.com/jy0205/STCAT.

        ----

        ## [2117] Pitfalls of Epistemic Uncertainty Quantification through Loss Minimisation

        **Authors**: *Viktor Bengs, Eyke Hüllermeier, Willem Waegeman*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/bc1d640f841f752c689aae20b31198c1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/bc1d640f841f752c689aae20b31198c1-Abstract-Conference.html)

        **Abstract**:

        Uncertainty quantification has received increasing attention in machine learning in the recent past. In particular, a distinction between aleatoric and epistemic uncertainty has been found useful in this regard. The latter refers to the learner's (lack of) knowledge and appears to be especially difficult to measure and quantify. In this paper, we analyse a recent proposal based on the idea of a second-order learner, which yields predictions in the form of distributions over probability distributions. While standard (first-order) learners can be trained to predict accurate probabilities, namely by minimising suitable loss functions on sample data, we show that loss minimisation does not work for second-order predictors: The loss functions proposed for inducing such predictors do not incentivise the learner to represent its epistemic uncertainty in a faithful way.

        ----

        ## [2118] Pile of Law: Learning Responsible Data Filtering from the Law and a 256GB Open-Source Legal Dataset

        **Authors**: *Peter Henderson, Mark S. Krass, Lucia Zheng, Neel Guha, Christopher D. Manning, Dan Jurafsky, Daniel E. Ho*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/bc218a0c656e49d4b086975a9c785f47-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2022/hash/bc218a0c656e49d4b086975a9c785f47-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        One concern with the rise of large language models lies with their potential for significant harm, particularly from pretraining on biased, obscene, copyrighted, and private information. Emerging ethical approaches have attempted to filter pretraining material, but such approaches have been ad hoc and failed to take context into account. We offer an approach to filtering grounded in law, which has directly addressed the tradeoffs in filtering material. First, we gather and make available the Pile of Law, a ~256GB (and growing) dataset of open-source English-language legal and administrative data, covering court opinions, contracts, administrative rules, and legislative records. Pretraining on the Pile of Law may help with legal tasks that have the promise to improve access to justice. Second, we distill the legal norms that governments have developed to constrain the inclusion of toxic or private content into actionable lessons for researchers and discuss how our dataset reflects these norms. Third, we show how the Pile of Law offers researchers the opportunity to learn such filtering rules directly from the data, providing an exciting new research direction in model-based processing.

        ----

        ## [2119] Active-Passive SimStereo - Benchmarking the Cross-Generalization Capabilities of Deep Learning-based Stereo Methods

        **Authors**: *Laurent Valentin Jospin, Allen Antony, Lian Xu, Hamid Laga, Farid Boussaïd, Mohammed Bennamoun*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/bc3a68a20e5c8ba5cbefc1ecf74bfaaa-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2022/hash/bc3a68a20e5c8ba5cbefc1ecf74bfaaa-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        In stereo vision, self-similar or bland regions can make it difficult to match patches between two images. Active stereo-based methods mitigate this problem by projecting a pseudo-random pattern on the scene so that each patch of an image pair can be identified without ambiguity. However, the projected pattern significantly alters the appearance of the image. If this pattern acts as a form of adversarial noise, it could negatively impact the performance of deep learning-based methods, which are now the de-facto standard for dense stereo vision. In this paper, we propose the Active-Passive SimStereo dataset and a corresponding benchmark to evaluate the performance gap between passive and active stereo images for stereo matching algorithms. Using the proposed benchmark and an additional ablation study, we show that the feature extraction and matching modules of a selection of twenty selected deep learning-based stereo matching methods generalize to active stereo without a problem. However, the disparity refinement modules of three of the twenty architectures (ACVNet, CascadeStereo, and StereoNet) are negatively affected by the active stereo patterns due to their reliance on the appearance of the input images.

        ----

        ## [2120] Back Razor: Memory-Efficient Transfer Learning by Self-Sparsified Backpropagation

        **Authors**: *Ziyu Jiang, Xuxi Chen, Xueqin Huang, Xianzhi Du, Denny Zhou, Zhangyang Wang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/bc6a1f968f8b1dae3e880f3f723d7d46-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/bc6a1f968f8b1dae3e880f3f723d7d46-Abstract-Conference.html)

        **Abstract**:

        Transfer learning from the model trained on large datasets to customized downstream tasks has been widely used as the pre-trained model can greatly boost the generalizability. However, the increasing sizes of pre-trained models also lead to a prohibitively large memory footprints for downstream transferring, making them unaffordable for personal devices. Previous work recognizes the bottleneck of the footprint to be the activation, and hence proposes various solutions such as injecting specific lite modules. In this work, we present a novel memory-efficient transfer framework called Back Razor, that can be plug-and-play applied to any pre-trained network without changing its architecture. The key idea of Back Razor is asymmetric sparsifying: pruning the activation stored for back-propagation, while keeping the forward activation dense. It is based on the observation that the stored activation, that dominates the memory footprint, is only needed for backpropagation. Such asymmetric pruning avoids affecting the precision of forward computation, thus making more aggressive pruning possible. Furthermore, we conduct the theoretical analysis for the convergence rate of Back Razor, showing that under mild conditions, our method retains the similar convergence rate as vanilla SGD. Extensive transfer learning experiments on both Convolutional Neural Networks and Vision Transformers with classification, dense prediction, and language modeling tasks show that Back Razor could yield up to 97% sparsity, saving 9.2x memory usage, without losing accuracy. The code is available at: https://github.com/VITA-Group/BackRazor_Neurips22.

        ----

        ## [2121] Patching open-vocabulary models by interpolating weights

        **Authors**: *Gabriel Ilharco, Mitchell Wortsman, Samir Yitzhak Gadre, Shuran Song, Hannaneh Hajishirzi, Simon Kornblith, Ali Farhadi, Ludwig Schmidt*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/bc6cddcd5d325e1c0f826066c1ad0215-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/bc6cddcd5d325e1c0f826066c1ad0215-Abstract-Conference.html)

        **Abstract**:

        Open-vocabulary models like CLIP achieve high accuracy across many image classification tasks. However, there are still settings where their zero-shot performance is far from optimal. We study model patching, where the goal is to improve accuracy on specific tasks without degrading accuracy on tasks where performance is already adequate. Towards this goal, we introduce PAINT, a patching method that uses interpolations between the weights of a model before fine-tuning and the weights after fine-tuning on a task to be patched. On nine tasks where zero-shot CLIP performs poorly, PAINT increases accuracy by 15 to 60 percentage points while preserving accuracy on ImageNet within one percentage point of the zero-shot model. PAINT also allows a single model to be patched on multiple tasks and improves with model scale. Furthermore, we identify cases of broad transfer, where patching on one task increases accuracy on other tasks even when the tasks have disjoint classes. Finally, we investigate applications beyond common benchmarks such as counting or reducing the impact of typographic attacks on CLIP. Our findings demonstrate that it is possible to expand the set of tasks on which open-vocabulary models achieve high accuracy without re-training them from scratch.

        ----

        ## [2122] Training stochastic stabilized supralinear networks by dynamics-neutral growth

        **Authors**: *Wayne Soo, Máté Lengyel*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/bc827452450356f9f558f4e4568d553b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/bc827452450356f9f558f4e4568d553b-Abstract-Conference.html)

        **Abstract**:

        There continues to be a trade-off between the biological realism and performance of neural networks. Contemporary deep learning techniques allow neural networks to be trained to perform challenging computations at (near) human-level, but these networks typically violate key biological constraints. More detailed models of biological neural networks can incorporate many of these constraints but typically suffer from subpar performance and trainability. Here, we narrow this gap by developing an effective method for training a canonical model of cortical neural circuits, the stabilized supralinear network (SSN), that in previous work had to be constructed manually or trained with undue constraints. SSNs are particularly challenging to train for the same reasons that make them biologically realistic: they are characterized by strongly-connected excitatory cells and expansive firing rate non-linearities that together make them prone to dynamical instabilities unless stabilized by appropriately tuned recurrent inhibition. Our method avoids such instabilities by initializing a small network and gradually increasing network size via the dynamics-neutral addition of neurons during training. We first show how SSNs can be trained to perform typical machine learning tasks by training an SSN on MNIST classification. We then demonstrate the effectiveness of our method by training an SSN on the challenging task of performing amortized Markov chain Monte Carlo-based inference under a Gaussian scale mixture generative model of natural image patches with a rich and diverse set of basis functions -- something that was not possible with previous methods. These results open the way to training realistic cortical-like neural networks on challenging tasks at scale.

        ----

        ## [2123] Post-hoc estimators for learning to defer to an expert

        **Authors**: *Harikrishna Narasimhan, Wittawat Jitkrittum, Aditya Krishna Menon, Ankit Singh Rawat, Sanjiv Kumar*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/bc8f76d9caadd48f77025b1c889d2e2d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/bc8f76d9caadd48f77025b1c889d2e2d-Abstract-Conference.html)

        **Abstract**:

        Many practical settings allow a learner to defer predictions to one or more costly experts. For example, the learning to defer paradigm allows a learner to defer to a human expert, at some monetary cost. Similarly, the adaptive inference paradigm allows a base model to defer to one or more large models, at some computational cost. The goal in these settings is to learn classification and deferral mechanisms to optimise a suitable accuracy-cost tradeoff. To achieve this, a central issue studied in prior work is the design of a coherent loss function for both mechanisms. In this work, we demonstrate that existing losses have two subtle limitations: they can encourage underfitting when there is a high cost of deferring, and the deferral function can have a weak dependence on the base model predictions. To resolve these issues, we propose a post-hoc training scheme: we train a deferral function on top of a base model, with the objective of predicting to defer when the base model's error probability exceeds the cost of the expert model. This may be viewed as applying a partial surrogate to the ideal deferral loss, which can lead to a tighter approximation and thus better performance. Empirically, we verify the efficacy of post-hoc training on benchmarks for learning to defer and adaptive inference.

        ----

        ## [2124] Learning Generalizable Part-based Feature Representation for 3D Point Clouds

        **Authors**: *Xin Wei, Xiang Gu, Jian Sun*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/bc943cd038a5531d5433b1431c822c01-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/bc943cd038a5531d5433b1431c822c01-Abstract-Conference.html)

        **Abstract**:

        Deep networks on 3D point clouds have achieved remarkable success in 3D classification, while they are vulnerable to geometry variations caused by inconsistent data acquisition procedures. This results in a challenging 3D domain generalization (3DDG) problem, that is to generalize a model trained on source domain to an unseen target domain. Based on the observation that local geometric structures are more generalizable than the whole shape, we propose to reduce the geometry shift by a generalizable part-based feature representation and design a novel part-based domain generalization network (PDG) for 3D point cloud classification. Specifically, we build a part-template feature space shared by source and target domains. Shapes from distinct domains are first organized to part-level features and then represented by part-template features. The transformed part-level features, dubbed aligned part-based representations, are then aggregated by a part-based feature aggregation module. To improve the robustness of the part-based representations, we further propose a contrastive learning framework upon part-based shape representation. Experiments and ablation studies on 3DDA and 3DDG benchmarks justify the efficacy of the proposed approach for domain generalization, compared with the previous state-of-the-art methods. Our code will be available on http://github.com/weixmath/PDG.

        ----

        ## [2125] FourierFormer: Transformer Meets Generalized Fourier Integral Theorem

        **Authors**: *Tan Nguyen, Minh Pham, Tam Nguyen, Khai Nguyen, Stanley J. Osher, Nhat Ho*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/bc968adbdff4a2551649d464b83f264a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/bc968adbdff4a2551649d464b83f264a-Abstract-Conference.html)

        **Abstract**:

        Multi-head attention empowers the recent success of transformers, the state-of-the-art models that have achieved remarkable success in sequence modeling and beyond. These attention mechanisms compute the pairwise dot products between the queries and keys, which results from the use of unnormalized Gaussian kernels with the assumption that the queries follow a mixture of Gaussian distribution. There is no guarantee that this assumption is valid in practice. In response, we first interpret attention in transformers as a nonparametric kernel regression. We then propose the FourierFormer, a new class of transformers in which the dot-product kernels are replaced by the novel generalized Fourier integral kernels. Different from the dot-product kernels, where we need to choose a good covariance matrix to capture the dependency of the features of data, the generalized Fourier integral kernels can automatically capture such dependency and remove the need to tune the covariance matrix. We theoretically prove that our proposed Fourier integral kernels can efficiently approximate any key and query distributions. Compared to the conventional transformers with dot-product attention, FourierFormers attain better accuracy and reduce the redundancy between attention heads. We empirically corroborate the advantages of FourierFormers over the baseline transformers in a variety of practical applications including language modeling and image classification.

        ----

        ## [2126] Learning Multi-resolution Functional Maps with Spectral Attention for Robust Shape Matching

        **Authors**: *Lei Li, Nicolas Donati, Maks Ovsjanikov*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/bcade016e3004543b289b33e7deb7472-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/bcade016e3004543b289b33e7deb7472-Abstract-Conference.html)

        **Abstract**:

        In this work, we present a novel non-rigid shape matching framework based on multi-resolution functional maps with spectral attention. Existing functional map learning methods all rely on the critical choice of the spectral resolution hyperparameter, which can severely affect the overall accuracy or lead to overfitting, if not chosen carefully. In this paper, we show that spectral resolution tuning can be alleviated by introducing spectral attention. Our framework is applicable in both supervised and unsupervised settings, and we show that it is possible to train the network so that it can adapt the spectral resolution, depending on the given shape input. More specifically, we propose to compute multi-resolution functional maps that characterize correspondence across a range of spectral resolutions, and introduce a spectral attention network that helps to combine this representation into a single coherent final correspondence. Our approach is not only accurate with near-isometric input, for which a high spectral resolution is typically preferred, but also robust and able to produce reasonable matching even in the presence of significant non-isometric distortion, which poses great challenges to existing methods. We demonstrate the superior performance of our approach through experiments on a suite of challenging near-isometric and non-isometric shape matching benchmarks.

        ----

        ## [2127] Label-invariant Augmentation for Semi-Supervised Graph Classification

        **Authors**: *Han Yue, Chunhui Zhang, Chuxu Zhang, Hongfu Liu*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/bcc0b4fc47c21e143461bb98c1d55926-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/bcc0b4fc47c21e143461bb98c1d55926-Abstract-Conference.html)

        **Abstract**:

        Recently, contrastiveness-based augmentation surges a new climax in the computer vision domain, where some operations, including rotation, crop, and flip, combined with dedicated algorithms, dramatically increase the model generalization and robustness. Following this trend, some pioneering attempts employ the similar idea to graph data. Nevertheless, unlike images, it is much more difficult to design reasonable augmentations without changing the nature of graphs. Although exciting, the current graph contrastive learning does not achieve as promising performance as visual contrastive learning. We conjecture the current performance of graph contrastive learning might be limited by the violation of the label-invariant augmentation assumption. In light of this, we propose a label-invariant augmentation for graph-structured data to address this challenge. Different from the node/edge modification and subgraph extraction, we conduct the augmentation in the representation space and generate the augmented samples in the most difficult direction while keeping the label of augmented data the same as the original samples. In the semi-supervised scenario, we demonstrate our proposed method outperforms the classical graph neural network based methods and recent graph contrastive learning on eight benchmark graph-structured data, followed by several in-depth experiments to further explore the label-invariant augmentation in several aspects.

        ----

        ## [2128] Practical Adversarial Multivalid Conformal Prediction

        **Authors**: *Osbert Bastani, Varun Gupta, Christopher Jung, Georgy Noarov, Ramya Ramalingam, Aaron Roth*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/bcdaaa1aec3ae2aa39542acefdec4e4b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/bcdaaa1aec3ae2aa39542acefdec4e4b-Abstract-Conference.html)

        **Abstract**:

        We give a simple, generic conformal prediction method for sequential prediction that achieves target empirical coverage guarantees on adversarial data. It is computationally lightweight --- comparable to split conformal prediction --- but does not require having a held-out validation set, and so all data can be used for training models from which to derive a conformal score. Furthermore, it gives stronger than marginal coverage guarantees in two ways. First, it gives threshold-calibrated prediction sets that have correct empirical coverage even conditional on the threshold used to form the prediction set from the conformal score. Second, the user can specify an arbitrary collection of subsets of the feature space --- possibly intersecting --- and the coverage guarantees will also hold conditional on membership in each of these subsets. We call our algorithm MVP, short for MultiValid Prediction. We give both theory and an extensive set of empirical evaluations.

        ----

        ## [2129] Test-Time Training with Masked Autoencoders

        **Authors**: *Yossi Gandelsman, Yu Sun, Xinlei Chen, Alexei A. Efros*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/bcdec1c2d60f94a93b6e36f937aa0530-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/bcdec1c2d60f94a93b6e36f937aa0530-Abstract-Conference.html)

        **Abstract**:

        Test-time training adapts to a new test distribution on the fly by optimizing a model for each test input using self-supervision.In this paper, we use masked autoencoders for this one-sample learning problem.Empirically, our simple method improves generalization on many visual benchmarks for distribution shifts.Theoretically, we characterize this improvement in terms of the bias-variance trade-off.

        ----

        ## [2130] Constraining Gaussian Processes to Systems of Linear Ordinary Differential Equations

        **Authors**: *Andreas Besginow, Markus Lange-Hegermann*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/bcef27c5825d1ed8757290f237b2d851-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/bcef27c5825d1ed8757290f237b2d851-Abstract-Conference.html)

        **Abstract**:

        Data in many applications follows systems of Ordinary Differential Equations (ODEs).This paper presents a novel algorithmic and symbolic construction for covariance functions of Gaussian Processes (GPs) with realizations strictly following a system of linear homogeneous ODEs with constant coefficients, which we call LODE-GPs. Introducing this strong inductive bias into a GP improves modelling of such data. Using smith normal form algorithms, a symbolic technique, we overcome two current restrictions in the state of the art: (1) the need for certain uniqueness conditions in the set of solutions, typically assumed in classical ODE solvers and their probabilistic counterparts, and (2) the restriction to controllable systems, typically assumed when encoding differential equations in covariance functions. We show the effectiveness of LODE-GPs in a number of experiments, for example learning physically interpretable parameters by maximizing the likelihood.

        ----

        ## [2131] So3krates: Equivariant attention for interactions on arbitrary length-scales in molecular systems

        **Authors**: *J. Thorben Frank, Oliver T. Unke, Klaus-Robert Müller*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/bcf4ca90a8d405201d29dd47d75ac896-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/bcf4ca90a8d405201d29dd47d75ac896-Abstract-Conference.html)

        **Abstract**:

        The application of machine learning methods in quantum chemistry has enabled the study of numerous chemical phenomena, which are computationally intractable with traditional ab-initio methods. However, some quantum mechanical properties of molecules and materials depend on non-local electronic effects, which are often neglected due to the difficulty of modeling them efficiently. This work proposes a modified attention mechanism adapted to the underlying physics, which allows to recover the relevant non-local effects. Namely, we introduce spherical harmonic coordinates (SPHCs) to reflect higher-order geometric information for each atom in a molecule, enabling a non-local formulation of attention in the SPHC space. Our proposed model So3krates - a self-attention based message passing neural network - uncouples geometric information from atomic features, making them independently amenable to attention mechanisms. Thereby we construct spherical filters, which extend the concept of continuous filters in Euclidean space to SPHC space and serve as foundation for a spherical self-attention mechanism. We show that in contrast to other published methods, So3krates is able to describe non-local quantum mechanical effects over arbitrary length scales. Further, we find evidence that the inclusion of higher-order geometric correlations increases data efficiency and improves generalization. So3krates matches or exceeds state-of-the-art performance on popular benchmarks, notably, requiring a significantly lower number of parameters (0.25 - 0.4x) while at the same time giving a substantial speedup (6 - 14x for training and 2 - 11x for inference) compared to other models.

        ----

        ## [2132] HyperDomainNet: Universal Domain Adaptation for Generative Adversarial Networks

        **Authors**: *Aibek Alanov, Vadim Titov, Dmitry P. Vetrov*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/bd1fc5cbedfe4d90d0ac2d23966fa27e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/bd1fc5cbedfe4d90d0ac2d23966fa27e-Abstract-Conference.html)

        **Abstract**:

        Domain adaptation framework of GANs has achieved great progress in recent years as a main successful approach of training contemporary GANs in the case of very limited training data. In this work, we significantly improve this framework by proposing an extremely compact parameter space for fine-tuning the generator. We introduce a novel domain-modulation technique that allows to optimize only 6 thousand-dimensional vector instead of 30 million weights of StyleGAN2 to adapt to a target domain. We apply this parameterization to the state-of-art domain adaptation methods and show that it has almost the same expressiveness as the full parameter space. Additionally, we propose a new regularization loss that considerably enhances the diversity of the fine-tuned generator. Inspired by the reduction in the size of the optimizing parameter space we consider the problem of multi-domain adaptation of GANs, i.e. setting when the same model can adapt to several domains depending on the input query. We propose the HyperDomainNet that is a hypernetwork that predicts our parameterization given the target domain. We empirically confirm that it can successfully learn a number of domains at once and may even generalize to unseen domains. Source code can be found at https://github.com/MACderRu/HyperDomainNet

        ----

        ## [2133] Multiagent Q-learning with Sub-Team Coordination

        **Authors**: *Wenhan Huang, Kai Li, Kun Shao, Tianze Zhou, Matthew E. Taylor, Jun Luo, Dongge Wang, Hangyu Mao, Jianye Hao, Jun Wang, Xiaotie Deng*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/bd31bfd4caa85bffe07a35568182cdfa-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/bd31bfd4caa85bffe07a35568182cdfa-Abstract-Conference.html)

        **Abstract**:

        In many real-world cooperative multiagent reinforcement learning (MARL) tasks, teams of agents can rehearse together before deployment, but then communication constraints may force individual agents to execute independently when deployed. Centralized training and decentralized execution (CTDE) is increasingly popular in recent years, focusing mainly on this setting. In the value-based MARL branch, credit assignment mechanism is typically used to factorize the team reward into each individual’s reward — individual-global-max (IGM) is a condition on the factorization ensuring that agents’ action choices coincide with team’s optimal joint action. However, current architectures fail to consider local coordination within sub-teams that should be exploited for more effective factorization, leading to faster learning. We propose a novel value factorization framework, called multiagent Q-learning with sub-team coordination (QSCAN), to flexibly represent sub-team coordination while honoring the IGM condition. QSCAN encompasses the full spectrum of sub-team coordination according to sub-team size, ranging from the monotonic value function class to the entire IGM function class, with familiar methods such as QMIX and QPLEX located at the respective extremes of the spectrum. Experimental results show that QSCAN’s performance dominates state-of-the-art methods in matrix games, predator-prey tasks, the Switch challenge in MA-Gym. Additionally, QSCAN achieves comparable performances to those methods in a selection of StarCraft II micro-management tasks.

        ----

        ## [2134] CLiMB: A Continual Learning Benchmark for Vision-and-Language Tasks

        **Authors**: *Tejas Srinivasan, Ting-Yun Chang, Leticia Leonor Pinto Alva, Georgios Chochlakis, Mohammad Rostami, Jesse Thomason*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/bd3611971089d466ab4ca96a20f7ab13-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2022/hash/bd3611971089d466ab4ca96a20f7ab13-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        Current state-of-the-art vision-and-language models are evaluated on tasks either individually or in a multi-task setting, overlooking the challenges of continually learning (CL) tasks as they arrive. Existing CL benchmarks have facilitated research on task adaptation and mitigating "catastrophic forgetting", but are limited to vision-only and language-only tasks. We present CLiMB, a benchmark to study the challenge of learning multimodal tasks in a CL setting, and to systematically evaluate how upstream continual learning can rapidly generalize to new multimodal and unimodal tasks. CLiMB includes implementations of several CL algorithms and a modified Vision-Language Transformer (ViLT) model that can be deployed on both multimodal and unimodal tasks. We find that common CL methods can help mitigate forgetting during multimodal task learning, but do not enable cross-task knowledge transfer. We envision that CLiMB will facilitate research on a new class of CL algorithms for this challenging multimodal setting.

        ----

        ## [2135] Bidirectional Learning for Offline Infinite-width Model-based Optimization

        **Authors**: *Can Chen, Yingxue Zhang, Jie Fu, Xue (Steve) Liu, Mark Coates*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/bd391cf5bdc4b63674d6da3edc1bde0d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/bd391cf5bdc4b63674d6da3edc1bde0d-Abstract-Conference.html)

        **Abstract**:

        In offline model-based optimization, we strive to maximize a black-box objective function by only leveraging a static dataset of designs and their scores. This problem setting arises in numerous fields including the design of materials, robots, DNAs, proteins, etc. Recent approaches train a deep neural network (DNN) model on the static dataset to act as a proxy function, and then perform gradient ascent on the existing designs to obtain potentially high-scoring designs. This methodology frequently suffers from the out-of-distribution problem where the proxy function often returns adversarial designs. To mitigate this problem, we propose $\textit{\textbf{B}i\textbf{D}irectional learning for offline \textbf{I}nfinite-width model-based optimization}~(\textbf{BDI})$. BDI consists of two mappings: the forward mapping leverages the static dataset to predict the scores of the high-scoring designs, and the backward mapping leverages the high-scoring designs to predict the scores of the static dataset. The backward mapping, neglected in previous work, can distill more information of the static dataset into the high-scoring designs, which effectively mitigates the out-of-distribution problem. Yet, for a finite-width DNN model, the loss function of the backward mapping is intractable and only has an approximate form, which leads to a significant deterioration of the design quality. We thus adopt an infinite-width DNN model and propose to employ the corresponding neural tangent kernel to yield a closed-form loss for more accurate design updates. Experiments on various tasks verify the effectiveness of BDI. The code is available [here](https://github.com/GGchen1997/BDI).

        ----

        ## [2136] Differentially Private Model Compression

        **Authors**: *Fatemehsadat Mireshghallah, Arturs Backurs, Huseyin A. Inan, Lukas Wutschitz, Janardhan Kulkarni*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/bd6bb13e78da078d8adcabbe6d9ca737-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/bd6bb13e78da078d8adcabbe6d9ca737-Abstract-Conference.html)

        **Abstract**:

        Recent papers have shown that large pre-trained language models (LLMs) such as BERT, GPT-2 can be fine-tuned on private data to achieve performance comparable to non-private models for many downstream Natural Language Processing (NLP) tasks while simultaneously guaranteeing differential privacy. The inference cost of these models -- which consist of hundreds of millions of parameters -- however, can be prohibitively large.  Hence, often in practice, LLMs are compressed before they are deployed in specific applications. In this paper, we initiate the study of differentially private model compression and propose frameworks for achieving 50% sparsity levels while maintaining nearly full performance. We demonstrate these ideas on standard GLUE benchmarks using BERT models, setting benchmarks for future research on this topic.

        ----

        ## [2137] Dual-Curriculum Contrastive Multi-Instance Learning for Cancer Prognosis Analysis with Whole Slide Images

        **Authors**: *Chao Tu, Yu Zhang, Zhenyuan Ning*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/bd8b52c2fefdb37e3b3953a37408e9dc-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/bd8b52c2fefdb37e3b3953a37408e9dc-Abstract-Conference.html)

        **Abstract**:

        The multi-instance learning (MIL) has advanced cancer prognosis analysis with whole slide images (WSIs). However, current MIL methods for WSI analysis still confront unique challenges. Previous methods typically generate instance representations via a pre-trained model or a model trained by the instances with bag-level annotations, which, however, may not generalize well to the downstream task due to the introduction of excessive label noises and the lack of fine-grained information across multi-magnification WSIs. Additionally, existing methods generally aggregate instance representations as bag ones for prognosis prediction and have no consideration of intra-bag redundancy and inter-bag discrimination. To address these issues, we propose a dual-curriculum contrastive MIL method for cancer prognosis analysis with WSIs. The proposed method consists of two curriculums, i.e., saliency-guided weakly-supervised instance encoding with cross-scale tiles and contrastive-enhanced soft-bag prognosis inference. Extensive experiments on three public datasets demonstrate that our method outperforms state-of-the-art methods in this field. The code is available at https://github.com/YuZhang-SMU/Cancer-Prognosis-Analysis/tree/main/DC_MIL%20Code.

        ----

        ## [2138] Meta-Complementing the Semantics of Short Texts in Neural Topic Models

        **Authors**: *Delvin Ce Zhang, Hady W. Lauw*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/bda5c35eded86adaf0231748e3ce071c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/bda5c35eded86adaf0231748e3ce071c-Abstract-Conference.html)

        **Abstract**:

        Topic models infer latent topic distributions based on observed word co-occurrences in a text corpus. While typically a corpus contains documents of variable lengths, most previous topic models treat documents of different lengths uniformly, assuming that each document is sufficiently informative. However, shorter documents may have only a few word co-occurrences, resulting in inferior topic quality.  Some other previous works assume that all documents are short, and leverage external auxiliary data, e.g., pretrained word embeddings and document connectivity. Orthogonal to existing works, we remedy this problem within the corpus itself by proposing a Meta-Complement Topic Model, which improves topic quality of short texts by transferring the semantic knowledge learned on long documents to complement semantically limited short texts. As a self-contained module, our framework is agnostic to auxiliary data and can be further improved by flexibly integrating them into our framework. Specifically, when incorporating document connectivity, we further extend our framework to complement documents with limited edges. Experiments demonstrate the advantage of our framework.

        ----

        ## [2139] Unified Optimal Transport Framework for Universal Domain Adaptation

        **Authors**: *Wanxing Chang, Ye Shi, Hoang Tuan, Jingya Wang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/bda6843dbbca0b09b8769122e0928fad-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/bda6843dbbca0b09b8769122e0928fad-Abstract-Conference.html)

        **Abstract**:

        Universal Domain Adaptation (UniDA) aims to transfer knowledge from a source domain to a target domain without any constraints on label sets. Since both domains may hold private classes, identifying target common samples for domain alignment is an essential issue in UniDA. Most existing methods require manually specified or hand-tuned threshold values to detect common samples thus they are hard to extend to more realistic UniDA because of the diverse ratios of common classes. Moreover, they cannot recognize different categories among target-private samples as these private samples are treated as a whole. In this paper, we propose to use Optimal Transport (OT) to handle these issues under a unified framework, namely UniOT. First, an OT-based partial alignment with adaptive filling is designed to detect common classes without any predefined threshold values for realistic UniDA. It can automatically discover the intrinsic difference between common and private classes based on the statistical information of the assignment matrix obtained from OT. Second, we propose an OT-based target representation learning that encourages both global discrimination and local consistency of samples to avoid the over-reliance on the source. Notably, UniOT is the first method with the capability to automatically discover and recognize private categories in the target domain for UniDA. Accordingly, we introduce a new metric H^3-score to evaluate the performance in terms of both accuracy of common samples and clustering performance of private ones. Extensive experiments clearly demonstrate the advantages of UniOT over a wide range of state-of-the-art methods in UniDA.

        ----

        ## [2140] GBA: A Tuning-free Approach to Switch between Synchronous and Asynchronous Training for Recommendation Models

        **Authors**: *Wenbo Su, Yuanxing Zhang, Yufeng Cai, Kaixu Ren, Pengjie Wang, Huimin Yi, Yue Song, Jing Chen, Hongbo Deng, Jian Xu, Lin Qu, Bo Zheng*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/be0a8ecf8b2743a4117557c5eca0fb79-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/be0a8ecf8b2743a4117557c5eca0fb79-Abstract-Conference.html)

        **Abstract**:

        High-concurrency asynchronous training upon parameter server (PS) architecture and high-performance synchronous training upon all-reduce (AR) architecture are the most commonly deployed distributed training modes for recommendation models. Although synchronous AR training is designed to have higher training efficiency, asynchronous PS training would be a better choice for training speed when there are stragglers (slow workers) in the shared cluster, especially under limited computing resources. An ideal way to take full advantage of these two training modes is to switch between them upon the cluster status. However, switching training modes often requires tuning hyper-parameters, which is extremely time- and resource-consuming. We find two obstacles to a tuning-free approach: the different distribution of the gradient values and the stale gradients from the stragglers. This paper proposes Global Batch gradients Aggregation (GBA) over PS, which aggregates and applies gradients with the same global batch size as the synchronous training. A token-control process is implemented to assemble the gradients and decay the gradients with severe staleness. We provide the convergence analysis to reveal that GBA has comparable convergence properties with the synchronous training, and demonstrate the robustness of GBA the recommendation models against the gradient staleness. Experiments on three industrial-scale recommendation tasks show that GBA is an effective tuning-free approach for switching. Compared to the state-of-the-art derived asynchronous training, GBA achieves up to 0.2% improvement on the AUC metric, which is significant for the recommendation models. Meanwhile, under the strained hardware resource, GBA speeds up at least 2.4x compared to synchronous training.

        ----

        ## [2141] TUSK: Task-Agnostic Unsupervised Keypoints

        **Authors**: *Yuhe Jin, Weiwei Sun, Jan Hosang, Eduard Trulls, Kwang Moo Yi*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/be53aed1708d5828441087e5c7b97440-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/be53aed1708d5828441087e5c7b97440-Abstract-Conference.html)

        **Abstract**:

        Existing unsupervised methods for keypoint learning rely heavily on the assumption that a specific keypoint type (e.g. elbow, digit, abstract geometric shape) appears only once in an image. This greatly limits their applicability, as each instance must be isolated before applying the method—an issue that is never discussed or evaluated. We thus propose a novel method to learn Task-agnostic, UnSupervised Keypoints (TUSK) which can deal with multiple instances. To achieve this, instead of the commonly-used strategy of detecting multiple heatmaps, each dedicated to a specific keypoint type, we use a single heatmap for detection, and enable unsupervised learning of keypoint types through clustering. Specifically, we encode semantics into the keypoints by teaching them to reconstruct images from a sparse set of keypoints and their descriptors, where the descriptors are forced to form distinct clusters in feature space around learned prototypes. This makes our approach amenable to a wider range of tasks than any previous unsupervised keypoint method: we show experiments on multiple-instance detection and classification, object discovery, and landmark detection—all unsupervised—with performance on par with the state of the art, while also being able to deal with multiple instances.

        ----

        ## [2142] Multi-block Min-max Bilevel Optimization with Applications in Multi-task Deep AUC Maximization

        **Authors**: *Quanqi Hu, Yongjian Zhong, Tianbao Yang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/be76ca290f1b30bd16cef178bfa8adbe-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/be76ca290f1b30bd16cef178bfa8adbe-Abstract-Conference.html)

        **Abstract**:

        In this paper, we study multi-block min-max bilevel optimization problems, where the upper level is non-convex strongly-concave minimax objective and the lower level is a strongly convex objective, and there are multiple blocks of dual variables and lower level problems. Due to the intertwined  multi-block min-max bilevel structure, the computational cost at each iteration could be prohibitively high, especially with a large number of blocks. To tackle this challenge, we present two single-loop randomized stochastic algorithms, which require updates for only a constant number of blocks at each iteration. Under some mild assumptions on the problem, we establish their sample complexity of $\mathcal{O}(1/\epsilon^4)$ for finding an $\epsilon$-stationary point. This matches the optimal complexity order for solving stochastic nonconvex optimization under a general unbiased stochastic oracle model. Moreover, we provide two applications of the proposed method in multi-task deep AUC (area under ROC curve) maximization. Experimental results validate our theory and demonstrate the effectiveness of our method.

        ----

        ## [2143] Coresets for Vertical Federated Learning: Regularized Linear Regression and $K$-Means Clustering

        **Authors**: *Lingxiao Huang, Zhize Li, Jialin Sun, Haoyu Zhao*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/be7b70477c8fca697f14b1dbb1c086d1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/be7b70477c8fca697f14b1dbb1c086d1-Abstract-Conference.html)

        **Abstract**:

        Vertical federated learning (VFL), where data features are stored in multiple parties distributively, is an important area in machine learning. However, the communication complexity for VFL is typically very high. In this paper, we propose a unified framework by constructing \emph{coresets} in a distributed fashion for communication-efficient VFL. We study two important learning tasks in the VFL setting: regularized linear regression and $k$-means clustering, and apply our coreset framework to both problems. We theoretically show that using coresets can drastically alleviate the communication complexity, while nearly maintain the solution quality. Numerical experiments are conducted to corroborate our theoretical findings.

        ----

        ## [2144] Class-Aware Adversarial Transformers for Medical Image Segmentation

        **Authors**: *Chenyu You, Ruihan Zhao, Fenglin Liu, Siyuan Dong, Sandeep Chinchali, Ufuk Topcu, Lawrence H. Staib, James S. Duncan*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/be99227ef4a4de84bb45d7dc7b53f808-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/be99227ef4a4de84bb45d7dc7b53f808-Abstract-Conference.html)

        **Abstract**:

        Transformers have made remarkable progress towards modeling long-range dependencies within the medical image analysis domain. However, current transformer-based models suffer from several disadvantages: (1) existing methods fail to capture the important features of the images due to the naive tokenization scheme; (2) the models suffer from information loss because they only consider single-scale feature representations; and (3) the segmentation label maps generated by the models are not accurate enough without considering rich semantic contexts and anatomical textures. In this work, we present CASTformer, a novel type of adversarial transformers, for 2D medical image segmentation. First, we take advantage of the pyramid structure to construct multi-scale representations and handle multi-scale variations. We then design a novel class-aware transformer module to better learn the discriminative regions of objects with semantic structures. Lastly, we utilize an adversarial training strategy that boosts segmentation accuracy and correspondingly allows a transformer-based discriminator to capture high-level semantically correlated contents and low-level anatomical features. Our experiments demonstrate that CASTformer dramatically outperforms previous state-of-the-art transformer-based approaches on three benchmarks, obtaining 2.54%-5.88% absolute improvements in Dice over previous models. Further qualitative experiments provide a more detailed picture of the modelâ€™s inner workings, shed light on the challenges in improved transparency, and demonstrate that transfer learning can greatly improve performance and reduce the size of medical image datasets in training, making CASTformer a strong starting point for downstream medical image analysis tasks.

        ----

        ## [2145] MCL-GAN: Generative Adversarial Networks with Multiple Specialized Discriminators

        **Authors**: *Jinyoung Choi, Bohyung Han*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/beac6bfb7eac3d651307c16ac747df01-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/beac6bfb7eac3d651307c16ac747df01-Abstract-Conference.html)

        **Abstract**:

        We propose a framework of generative adversarial networks with multiple discriminators, which collaborate to represent a real dataset more effectively. Our approach facilitates learning a generator consistent with the underlying data distribution based on real images and thus mitigates the chronic mode collapse problem. From the inspiration of multiple choice learning, we guide each discriminator to have expertise in a subset of the entire data and allow the generator to find reasonable correspondences between the latent and real data spaces automatically without extra supervision for training examples. Despite the use of multiple discriminators, the backbone networks are shared across the discriminators and the increase in training cost is marginal. We demonstrate the effectiveness of our algorithm using multiple evaluation metrics in the standard datasets for diverse tasks.

        ----

        ## [2146] Unsupervised Visual Representation Learning via Mutual Information Regularized Assignment

        **Authors**: *Dong Hoon Lee, Sungik Choi, Hyunwoo J. Kim, Sae-Young Chung*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/bedc61a9936af18cb51b7c5e8f3b89a3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/bedc61a9936af18cb51b7c5e8f3b89a3-Abstract-Conference.html)

        **Abstract**:

        This paper proposes Mutual Information Regularized Assignment (MIRA), a pseudo-labeling algorithm for unsupervised representation learning inspired by information maximization. We formulate online pseudo-labeling as an optimization problem to find pseudo-labels that maximize the mutual information between the label and data while being close to a given model probability. We derive a fixed-point iteration method and prove its convergence to the optimal solution. In contrast to baselines, MIRA combined with pseudo-label prediction enables a simple yet effective clustering-based representation learning without incorporating extra training techniques or artificial constraints such as sampling strategy, equipartition constraints, etc. With relatively small training epochs, representation learned by MIRA achieves state-of-the-art performance on various downstream tasks, including the linear/${\it k}$-NN evaluation and transfer learning. Especially, with only 400 epochs, our method applied to ImageNet dataset with ResNet-50 architecture achieves 75.6% linear evaluation accuracy.

        ----

        ## [2147] Mind Reader: Reconstructing complex images from brain activities

        **Authors**: *Sikun Lin, Thomas Sprague, Ambuj K. Singh*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/bee5125b773414d3d6eeb4334fbc5453-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/bee5125b773414d3d6eeb4334fbc5453-Abstract-Conference.html)

        **Abstract**:

        Understanding how the brain encodes external stimuli and how these stimuli can be decoded from the measured brain activities are long-standing and challenging questions in neuroscience. In this paper, we focus on reconstructing the complex image stimuli from fMRI (functional magnetic resonance imaging) signals. Unlike previous works that reconstruct images with single objects or simple shapes, our work aims to reconstruct image stimuli that are rich in semantics, closer to everyday scenes, and can reveal more perspectives. However, data scarcity of fMRI datasets is the main obstacle to applying state-of-the-art deep learning models to this problem. We find that incorporating an additional text modality is beneficial for the reconstruction problem compared to directly translating brain signals to images. Therefore, the modalities involved in our method are: (i) voxel-level fMRI signals, (ii) observed images that trigger the brain signals, and (iii) textual description of the images. To further address data scarcity, we leverage an aligned vision-language latent space pre-trained on massive datasets. Instead of training models from scratch to find a latent space shared by the three modalities, we encode fMRI signals into this pre-aligned latent space. Then, conditioned on embeddings in this space, we reconstruct images with a generative model. The reconstructed images from our pipeline balance both naturalness and fidelity: they are photo-realistic and capture the ground truth image contents well.

        ----

        ## [2148] A Few Expert Queries Suffices for Sample-Efficient RL with Resets and Linear Value Approximation

        **Authors**: *Philip Amortila, Nan Jiang, Dhruv Madeka, Dean P. Foster*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/bef8e5620c699630405adafaa86cb038-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/bef8e5620c699630405adafaa86cb038-Abstract-Conference.html)

        **Abstract**:

        The current paper studies sample-efficient Reinforcement Learning (RL) in settings where only the optimal value function is assumed to be linearly-realizable. It has recently been understood that, even under this seemingly strong assumption and access to a generative model, worst-case sample complexities can be prohibitively (i.e., exponentially) large. We investigate the setting where the learner additionally has access to interactive demonstrations from an expert policy, and we present a statistically and computationally efficient algorithm (Delphi) for blending exploration with expert queries. In particular, Delphi requires $\tilde O(d)$ expert queries and a $\texttt{poly}(d,H,|A|,1/\varepsilon)$ amount of exploratory samples to provably recover an $\varepsilon$-suboptimal policy. Compared to pure RL approaches, this corresponds to an exponential improvement in sample complexity with surprisingly-little expert input. Compared to prior imitation learning (IL) approaches, our required number of expert demonstrations is independent of $H$ and logarithmic in $1/\varepsilon$, whereas all prior work required at least linear factors of both in addition to the same dependence on $d$. Towards establishing the minimal amount of expert queries needed, we show that, in the same setting, any learner whose exploration budget is \textit{polynomially-bounded} (in terms of $d,H,$ and $|A|$) will require \textit{at least} $\tilde\Omega(\sqrt{d})$ oracle calls to recover a policy competing with the expert's value function. Under the weaker assumption that the expert's policy is linear, we show that the lower bound increases to $\tilde\Omega(d)$.

        ----

        ## [2149] Subsidiary Prototype Alignment for Universal Domain Adaptation

        **Authors**: *Jogendra Nath Kundu, Suvaansh Bhambri, Akshay R. Kulkarni, Hiran Sarkar, Varun Jampani, Venkatesh Babu R.*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/bf121b033db3bac31c3193e8a0dcbf66-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/bf121b033db3bac31c3193e8a0dcbf66-Abstract-Conference.html)

        **Abstract**:

        Universal Domain Adaptation (UniDA) deals with the problem of knowledge transfer between two datasets with domain-shift as well as category-shift. The goal is to categorize unlabeled target samples, either into one of the "known" categories or into a single "unknown" category. A major problem in UniDA is negative transfer, i.e. misalignment of "known" and "unknown" classes. To this end, we first uncover an intriguing tradeoff between negative-transfer-risk and domain-invariance exhibited at different layers of a deep network. It turns out we can strike a balance between these two metrics at a mid-level layer. Towards designing an effective framework based on this insight, we draw motivation from Bag-of-visual-Words (BoW). Word-prototypes in a BoW-like representation of a mid-level layer would represent lower-level visual primitives that are likely to be unaffected by the category-shift in the high-level features. We develop modifications that encourage learning of word-prototypes followed by word-histogram based classification. Following this, subsidiary prototype-space alignment (SPA) can be seen as a closed-set alignment problem, thereby avoiding negative transfer. We realize this with a novel word-histogram-related pretext task to enable closed-set SPA, operating in conjunction with goal task UniDA. We demonstrate the efficacy of our approach on top of existing UniDA techniques, yielding state-of-the-art performance across three standard UniDA and Open-Set DA object recognition benchmarks.

        ----

        ## [2150] Dynamic Inverse Reinforcement Learning for Characterizing Animal Behavior

        **Authors**: *Zoe Ashwood, Aditi Jha, Jonathan W. Pillow*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/bf215fa7fe70a38c5e967e59c44a99d0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/bf215fa7fe70a38c5e967e59c44a99d0-Abstract-Conference.html)

        **Abstract**:

        Understanding decision-making is a core goal in both neuroscience and psychology, and computational models have often been helpful in the pursuit of this goal. While many models have been developed for characterizing behavior in binary decision-making and bandit tasks, comparatively little work has focused on animal decision-making in more complex tasks, such as navigation through a maze. Inverse reinforcement learning (IRL) is a promising approach  for understanding such behavior, as it aims to infer the unknown reward function of an agent from its observed trajectories through state space. However, IRL has yet to be widely applied in neuroscience. One potential reason for this is that existing IRL frameworks assume that an agent's reward function is fixed over time. To address this shortcoming, we introduce dynamic inverse reinforcement learning (DIRL), a novel IRL framework that allows for time-varying intrinsic rewards. Our method parametrizes the unknown reward function as a time-varying linear combination of spatial reward maps (which we refer to as "goal maps"). We develop an efficient inference method for recovering this dynamic reward function from behavioral data.  We demonstrate DIRL in simulated experiments and then apply it to a dataset of mice exploring a labyrinth. Our method returns interpretable reward functions for two separate cohorts of mice, and provides a novel characterization of exploratory behavior. We expect DIRL to have broad applicability in neuroscience, and to facilitate the design of biologically-inspired reward functions for training artificial agents.

        ----

        ## [2151] FedRolex: Model-Heterogeneous Federated Learning with Rolling Sub-Model Extraction

        **Authors**: *Samiul Alam, Luyang Liu, Ming Yan, Mi Zhang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/bf5311df07f3efce97471921e6d2f159-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/bf5311df07f3efce97471921e6d2f159-Abstract-Conference.html)

        **Abstract**:

        Most cross-device federated learning (FL) studies focus on the model-homogeneous setting where the global server model and local client models are identical. However, such constraint not only excludes low-end clients who would otherwise make unique contributions to model training but also restrains clients from training large models due to on-device resource bottlenecks. In this work, we propose FedRolex, a partial training (PT)-based approach that enables model-heterogeneous FL and can train a global server model larger than the largest client model. At its core, FedRolex employs a rolling sub-model extraction scheme that allows different parts of the global server model to be evenly trained, which mitigates the client drift induced by the inconsistency between individual client models and server model architectures. Empirically, we show that FedRolex outperforms state-of-the-art PT-based model-heterogeneous FL methods (e.g. Federated Dropout) and reduces the gap between model-heterogeneous and model-homogeneous FL, especially under the large-model large-dataset regime. In addition, we provide theoretical statistical analysis on its advantage over Federated Dropout. Lastly, we evaluate FedRolex on an emulated real-world device distribution to show that FedRolex can enhance the inclusiveness of FL and boost the performance of low-end devices that would otherwise not benefit from FL. Our code is available at: https://github.com/AIoT-MLSys-Lab/FedRolex.

        ----

        ## [2152] Multi-Lingual Acquisition on Multimodal Pre-training for Cross-modal Retrieval

        **Authors**: *Liang Zhang, Anwen Hu, Qin Jin*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/bfadef437ed27372648714c930c3a77a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/bfadef437ed27372648714c930c3a77a-Abstract-Conference.html)

        **Abstract**:

        Vision and diverse languages are important information sources in our living world. A model that understands multi-modalities and multi-languages can be applied to a wider range of real-life scenarios. To build such a multimodal and multilingual model, existing works try to ensemble vision-language data from multiple languages in pre-training. However, due to the large number of languages, these works often require huge computing resources and cannot be flexibly extended to new languages. In this work, we propose a MultiLingual Acquisition (MLA) framework that can easily empower a monolingual Vision-Language Pre-training (VLP) model with multilingual capability. Specifically, we design a lightweight language acquisition encoder based on state-of-the-art monolingual VLP models. We further propose a two-stage training strategy to optimize the language acquisition encoder, namely the Native Language Transfer stage and the Language Exposure stage. With much less multilingual training data and computing resources, our model achieves state-of-the-art performance on multilingual image-text and video-text retrieval benchmarks.

        ----

        ## [2153] Manifold Interpolating Optimal-Transport Flows for Trajectory Inference

        **Authors**: *Guillaume Huguet, Daniel Sumner Magruder, Alexander Tong, Oluwadamilola Fasina, Manik Kuchroo, Guy Wolf, Smita Krishnaswamy*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/bfc03f077688d8885c0a9389d77616d0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/bfc03f077688d8885c0a9389d77616d0-Abstract-Conference.html)

        **Abstract**:

        We present a method called Manifold Interpolating Optimal-Transport Flow (MIOFlow) that learns stochastic, continuous population dynamics from static snapshot samples taken at sporadic timepoints. MIOFlow combines dynamic models,  manifold learning, and optimal transport by training neural ordinary differential equations (Neural ODE) to interpolate between static population snapshots as penalized by optimal transport with manifold ground distance. Further, we ensure that the flow follows the geometry by operating in the latent space of an autoencoder that we call a geodesic autoencoder (GAE). In GAE the latent space distance between points is regularized to match a novel multiscale geodesic distance on the data manifold that we define. We show that this method is superior to normalizing flows, Schr\"odinger bridges and other generative models that are designed to flow from noise to data in terms of interpolating between populations. Theoretically, we link these trajectories with dynamic optimal transport. We evaluate our method on simulated data with bifurcations and merges, as well as scRNA-seq data from embryoid body differentiation, and acute myeloid leukemia treatment.

        ----

        ## [2154] Depth is More Powerful than Width with Prediction Concatenation in Deep Forest

        **Authors**: *Shen-Huan Lyu, Yi-Xiao He, Zhi-Hua Zhou*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/c017e92288b5056c578bb6b0b69d9e76-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/c017e92288b5056c578bb6b0b69d9e76-Abstract-Conference.html)

        **Abstract**:

        Random Forest (RF) is an ensemble learning algorithm proposed by \citet{breiman2001random} that constructs a large number of randomized decision trees individually and aggregates their predictions by naive averaging. \citet{zhou2019deep} further propose Deep Forest (DF) algorithm with multi-layer feature transformation, which significantly outperforms random forest in various application fields. The prediction concatenation (PreConc) operation is crucial for the multi-layer feature transformation in deep forest, though little has been known about its theoretical property. In this paper, we analyze the influence of Preconc on the consistency of deep forest. Especially when the individual tree is inconsistent (as in practice, the individual tree is often set to be fully grown, i.e., there is only one sample at each leaf node), we find that the convergence rate of two-layer DF \textit{w.r.t.} the number of trees $M$ can reach $\mathcal{O}(1/M^2)$ under some mild conditions, while the convergence rate of RF is $\mathcal{O}(1/M)$. Therefore, with the help of PreConc, DF with deeper layer will be more powerful than the shallower layer. Experiments confirm theoretical advantages.

        ----

        ## [2155] Logical Activation Functions: Logit-space equivalents of Probabilistic Boolean Operators

        **Authors**: *Scott C. Lowe, Robert Earle, Jason d'Eon, Thomas Trappenberg, Sageev Oore*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/c05144b635df16ac9bbf8246bbbd55ca-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/c05144b635df16ac9bbf8246bbbd55ca-Abstract-Conference.html)

        **Abstract**:

        The choice of activation functions and their motivation is a long-standing issue within the neural network community. Neuronal representations within artificial neural networks are commonly understood as logits, representing the log-odds score of presence of features within the stimulus. We derive logit-space operators equivalent to probabilistic Boolean logic-gates AND, OR, and XNOR for independent probabilities. Such theories are important to formalize more complex dendritic operations in real neurons, and these operations can be used as activation functions within a neural network, introducing probabilistic Boolean-logic as the core operation of the neural network. Since these functions involve taking multiple exponents and logarithms, they are computationally expensive and not well suited to be directly used within neural networks. Consequently, we construct efficient approximations named $\text{AND}_\text{AIL}$ (the AND operator Approximate for Independent Logits), $\text{OR}_\text{AIL}$, and $\text{XNOR}_\text{AIL}$, which utilize only comparison and addition operations, have well-behaved gradients, and can be deployed as activation functions in neural networks. Like MaxOut, $\text{AND}_\text{AIL}$ and $\text{OR}_\text{AIL}$ are generalizations of ReLU to two-dimensions. While our primary aim is to formalize dendritic computations within a logit-space probabilistic-Boolean framework, we deploy these new activation functions, both in isolation and in conjunction to demonstrate their effectiveness on a variety of tasks including tabular classification, image classification, transfer learning, abstract reasoning, and compositional zero-shot learning.

        ----

        ## [2156] An Investigation into Whitening Loss for Self-supervised Learning

        **Authors**: *Xi Weng, Lei Huang, Lei Zhao, Rao Muhammad Anwer, Salman H. Khan, Fahad Shahbaz Khan*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/c057cb81b8d3c67093427bf1c16a4e9f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/c057cb81b8d3c67093427bf1c16a4e9f-Abstract-Conference.html)

        **Abstract**:

        A desirable objective in self-supervised learning (SSL) is to avoid feature collapse.  Whitening loss guarantees collapse avoidance by minimizing the distance between embeddings of positive pairs under the conditioning that the embeddings from different views are whitened. In this paper, we propose a framework with an informative indicator to analyze whitening loss, which provides a clue to demystify several interesting phenomena as well as a pivoting point connecting to other SSL methods. We reveal that batch whitening (BW) based methods do not impose whitening constraints on the embedding, but they only require the embedding to be full-rank. This full-rank constraint is also sufficient to avoid dimensional collapse. Based on our analysis, we propose channel whitening with random group partition (CW-RGP), which exploits the advantages of BW-based methods in preventing collapse and avoids their disadvantages requiring large batch size.  Experimental results on ImageNet classification and COCO object detection reveal that the proposed CW-RGP possesses a promising potential for learning good representations. The code is available at https://github.com/winci-ai/CW-RGP.

        ----

        ## [2157] Geometric Knowledge Distillation: Topology Compression for Graph Neural Networks

        **Authors**: *Chenxiao Yang, Qitian Wu, Junchi Yan*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/c06f788963f0ce069f5b2dbf83fe7822-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/c06f788963f0ce069f5b2dbf83fe7822-Abstract-Conference.html)

        **Abstract**:

        We study a new paradigm of knowledge transfer that aims at encoding graph topological information into graph neural networks (GNNs) by distilling knowledge from a teacher GNN model trained on a complete graph to a student GNN model operating on a smaller or sparser graph. To this end, we revisit the connection between thermodynamics and the behavior of GNN, based on which we propose Neural Heat Kernel (NHK) to encapsulate the geometric property of the underlying manifold concerning the architecture of GNNs. A fundamental and principled solution is derived by aligning NHKs on teacher and student models, dubbed as Geometric Knowledge Distillation. We develop non- and parametric instantiations and demonstrate their efficacy in various experimental settings for knowledge distillation regarding different types of privileged topological information and teacher-student schemes.

        ----

        ## [2158] A Benchmark for Compositional Visual Reasoning

        **Authors**: *Aimen Zerroug, Mohit Vaishnav, Julien Colin, Sebastian Musslick, Thomas Serre*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/c08ee8fe3d19521f3bfa4102898329fd-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2022/hash/c08ee8fe3d19521f3bfa4102898329fd-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        A fundamental component of human vision is our ability to parse complex visual scenes and judge the relations between their constituent objects. AI benchmarks for visual reasoning have driven rapid progress in recent years with state-of-the-art systems now reaching human accuracy on some of these benchmarks. Yet, there remains a major gap between humans and AI systems in terms of the sample efficiency with which they learn new visual reasoning tasks. Humans' remarkable efficiency at learning has been at least partially attributed to their ability to harness compositionality -- allowing them to efficiently take advantage of previously gained knowledge when learning new tasks. Here, we introduce a novel visual reasoning benchmark, Compositional Visual Relations (CVR), to drive progress towards the development of more data-efficient learning algorithms. We take inspiration from fluidic intelligence and non-verbal reasoning tests and describe a novel method for creating compositions of abstract rules and generating image datasets corresponding to these rules at scale. Our proposed benchmark includes measures of sample efficiency, generalization, compositionality, and transfer across task rules. We systematically evaluate modern neural architectures and find that convolutional architectures surpass transformer-based architectures across all performance measures in most data regimes. However, all computational models are much less data efficient than humans, even after learning informative visual representations using self-supervision. Overall, we hope our challenge will spur interest in developing neural architectures that can learn to harness compositionality for more efficient learning.

        ----

        ## [2159] Learning Articulated Rigid Body Dynamics with Lagrangian Graph Neural Network

        **Authors**: *Ravinder Bhattoo, Sayan Ranu, N. M. Anoop Krishnan*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/c0a9c840d651c295c095dad40e06fed9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/c0a9c840d651c295c095dad40e06fed9-Abstract-Conference.html)

        **Abstract**:

        Lagrangian  and Hamiltonian neural networks LNN and HNNs, respectively) encode strong inductive biases that allow them to outperform other models of physical systems significantly. However, these models have, thus far, mostly been limited to simple systems such as pendulums and springs or a single rigid body such as a gyroscope or a rigid rotor. Here, we present a Lagrangian graph neural network (LGNN) that can learn the dynamics of articulated rigid bodies by exploiting their topology. We demonstrate the performance of LGNN by learning the dynamics of ropes, chains, and trusses with the bars modeled as rigid bodies. LGNN also exhibits generalizability---LGNN trained on chains with a few segments exhibits generalizability to simulate a chain with large number of links and arbitrary link length. We also show that the LGNN can simulate unseen hybrid systems including bars and chains, on which they have not been trained on. Specifically, we show that the LGNN can be used to model the dynamics of complex real-world structures such as the stability of tensegrity structures. Finally, we discuss the non-diagonal nature of the mass matrix and its ability to generalize in complex systems.

        ----

        ## [2160] Myriad: a real-world testbed to bridge trajectory optimization and deep learning

        **Authors**: *Nikolaus H. R. Howe, Simon Dufort-Labbé, Nitarshan Rajkumar, Pierre-Luc Bacon*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/c0b91f9a3587bf35287f41dba5d20233-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2022/hash/c0b91f9a3587bf35287f41dba5d20233-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        We present Myriad, a testbed written in JAX which enables machine learning researchers to benchmark imitation learning and reinforcement learning algorithms against trajectory optimization-based methods in real-world environments. Myriad contains 17 optimal control problems presented in continuous time which span medicine, ecology, epidemiology, and engineering. As such, Myriad strives to serve as a stepping stone towards application of modern machine learning techniques for impactful real-world tasks. The repository also provides machine learning practitioners access to trajectory optimization techniques, not only for standalone use, but also for integration within a typical automatic differentiation workflow. Indeed, the combination of classical control theory and deep learning in a fully GPU-compatible package unlocks potential for new algorithms to arise. We present one such novel approach for use in dynamics learning and control tasks. Trained in a fully end-to-end fashion, our model leverages an implicit planning module over neural ordinary differential equations, enabling simultaneous learning and planning with unknown environment dynamics. All environments, optimizers and tools are available in the software package at \url{https://github.com/nikihowe/myriad}.

        ----

        ## [2161] Batch Bayesian optimisation via density-ratio estimation with guarantees

        **Authors**: *Rafael Oliveira, Louis C. Tiao, Fabio T. Ramos*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/c0d5a28eb3949efbedbe3e41751e3ffc-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/c0d5a28eb3949efbedbe3e41751e3ffc-Abstract-Conference.html)

        **Abstract**:

        Bayesian optimisation (BO) algorithms have shown remarkable success in applications involving expensive black-box functions. Traditionally BO has been set as a sequential decision-making process which estimates the utility of query points via an acquisition function and a prior over functions, such as a Gaussian process. Recently, however, a reformulation of BO via density-ratio estimation (BORE) allowed reinterpreting the acquisition function as a probabilistic binary classifier, removing the need for an explicit prior over functions and increasing scalability. In this paper, we present a theoretical analysis of BORE's regret and an extension of the algorithm with improved uncertainty estimates. We also show that BORE can be naturally extended to a batch optimisation setting by recasting the problem as approximate Bayesian inference. The resulting algorithms come equipped with theoretical performance guarantees and are assessed against other batch and sequential BO baselines in a series of experiments.

        ----

        ## [2162] Amplifying Membership Exposure via Data Poisoning

        **Authors**: *Yufei Chen, Chao Shen, Yun Shen, Cong Wang, Yang Zhang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/c0f240bb986df54b38026398da1ae72a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/c0f240bb986df54b38026398da1ae72a-Abstract-Conference.html)

        **Abstract**:

        As in-the-wild data are increasingly involved in the training stage, machine learning applications become more susceptible to data poisoning attacks. Such attacks typically lead to test-time accuracy degradation or controlled misprediction. In this paper, we investigate the third type of exploitation of data poisoning - increasing the risks of privacy leakage of benign training samples. To this end, we demonstrate a set of data poisoning attacks to amplify the membership exposure of the targeted class. We first propose a generic dirty-label attack for supervised classification algorithms. We then propose an optimization-based clean-label attack in the transfer learning scenario, whereby the poisoning samples are correctly labeled and look "natural" to evade human moderation. We extensively evaluate our attacks on computer vision benchmarks. Our results show that the proposed attacks can substantially increase the membership inference precision with minimum overall test-time model performance degradation. To mitigate the potential negative impacts of our attacks, we also investigate feasible countermeasures.

        ----

        ## [2163] Boosting the Transferability of Adversarial Attacks with Reverse Adversarial Perturbation

        **Authors**: *Zeyu Qin, Yanbo Fan, Yi Liu, Li Shen, Yong Zhang, Jue Wang, Baoyuan Wu*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/c0f9419caa85d7062c7e6d621a335726-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/c0f9419caa85d7062c7e6d621a335726-Abstract-Conference.html)

        **Abstract**:

        Deep neural networks (DNNs) have been shown to be vulnerable to adversarial examples, which can produce erroneous predictions by injecting imperceptible perturbations. In this work, we study the transferability of adversarial examples, which is significant due to its threat to real-world applications where model architecture or parameters are usually unknown. Many existing works reveal that the adversarial examples are likely to overfit the surrogate model that they are generated from, limiting its transfer attack performance against different target models. To mitigate the overfitting of the surrogate model, we propose a novel attack method, dubbed reverse adversarial perturbation (RAP). Specifically, instead of minimizing the loss of a single adversarial point, we advocate seeking adversarial example located at a region with unified low loss value, by injecting the worst-case perturbation (the reverse adversarial perturbation) for each step of the optimization procedure. The adversarial attack with RAP is formulated as a min-max bi-level optimization problem.  By integrating RAP into the iterative process for attacks, our method can find more stable adversarial examples which are less sensitive to the changes of decision boundary, mitigating the overfitting of the surrogate model.  Comprehensive experimental comparisons demonstrate that RAP can significantly boost adversarial transferability. Furthermore, RAP can be naturally combined with many existing black-box attack techniques, to further boost the transferability. When attacking a real-world image recognition system, Google Cloud Vision API, we obtain 22% performance improvement of targeted attacks over the compared method. Our codes are available at https://github.com/SCLBD/TransferattackRAP.

        ----

        ## [2164] Extra-Newton: A First Approach to Noise-Adaptive Accelerated Second-Order Methods

        **Authors**: *Kimon Antonakopoulos, Ali Kavis, Volkan Cevher*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/c10804702be5a0cca89331315413f1a2-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/c10804702be5a0cca89331315413f1a2-Abstract-Conference.html)

        **Abstract**:

        In this work, we propose a universal and adaptive second-order method for minimization of second-order smooth, convex functions. Precisely, our algorithm achieves $O(\sigma / \sqrt{T})$ when the oracle feedback is stochastic with variance $\sigma$, and obtains the improved $O( 1 / T^3)$ convergence with deterministic oracles. Our method achieves this rate interpolation without knowing the nature of the oracle a priori, which was enabled by a parameter-free step-size that is oblivious to the knowledge of smoothness modulus, variance bounds and the diameter of the constrained set. To our knowledge, this is the first universal algorithm that achieves the aforementioned global guarantees within second-order convex optimization literature.

        ----

        ## [2165] FETA: Towards Specializing Foundational Models for Expert Task Applications

        **Authors**: *Amit Alfassy, Assaf Arbelle, Oshri Halimi, Sivan Harary, Roei Herzig, Eli Schwartz, Rameswar Panda, Michele Dolfi, Christoph Auer, Peter W. J. Staar, Kate Saenko, Rogério Feris, Leonid Karlinsky*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/c12dd3034259fc000d80db823041c187-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2022/hash/c12dd3034259fc000d80db823041c187-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        Foundational Models (FMs) have demonstrated unprecedented capabilities including zero-shot learning, high fidelity data synthesis, and out of domain generalization. However, the parameter capacity of FMs is still limited, leading to poor out-of-the-box performance of FMs on many expert tasks (e.g. retrieval of car manuals technical illustrations from language queries), data for which is either unseen or belonging to a long-tail part of the data distribution of the huge datasets used for FM pre-training. This underlines the necessity to explicitly evaluate and finetune FMs on such expert tasks, arguably ones that appear the most in practical real-world applications. In this paper, we propose a first of its kind FETA benchmark built around the task of teaching FMs to understand technical documentation, via learning to match their graphical illustrations to corresponding language descriptions. Our FETA benchmark focuses on text-to-image and image-to-text retrieval in public car manuals and sales catalogue brochures. FETA is equipped with a procedure for completely automatic annotation extraction (code would be released upon acceptance), allowing easy extension of FETA to more documentation types and application domains in the future. Our automatic annotation leads to an automated performance metric shown to be consistent with metrics computed on human-curated annotations (also released). We provide multiple baselines and analysis of popular FMs on FETA leading to several interesting findings that we believe would be very valuable to the FM community, paving the way towards real-world application of FMs for many practical expert tasks currently being `overlooked' by standard benchmarks focusing on common objects.

        ----

        ## [2166] Global Convergence of Federated Learning for Mixed Regression

        **Authors**: *Lili Su, Jiaming Xu, Pengkun Yang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/c132c02176577c4319a878f6417a331a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/c132c02176577c4319a878f6417a331a-Abstract-Conference.html)

        **Abstract**:

        This paper studies the problem of model training under Federated Learning when clients exhibit cluster structure. We contextualize this problem in mixed regression, where each client has limited local data generated from one of $k$ unknown regression models. We design an algorithm that achieves global convergence from any initialization, and works even when local data volume is highly unbalanced  -- there could exist clients that contain $O(1)$ data points only. Our algorithm first runs moment descent on a few anchor clients (each with $\tilde{\Omega}(k)$ data points) to obtain coarse model estimates. Then each client alternately estimates its cluster labels  and refines the model estimates based on FedAvg or FedProx. A key innovation in our analysis is a uniform estimate on the clustering errors, which we prove by bounding the VC dimension of general polynomial concept classes based on the theory of algebraic geometry.

        ----

        ## [2167] BayesPCN: A Continually Learnable Predictive Coding Associative Memory

        **Authors**: *Jinsoo Yoo, Frank Wood*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/c13d5a10028586fdc15ee7da97b7563f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/c13d5a10028586fdc15ee7da97b7563f-Abstract-Conference.html)

        **Abstract**:

        Associative memory plays an important role in human intelligence and its mechanisms have been linked to attention in machine learning. While the machine learning community's interest in associative memories has recently been rekindled, most work has focused on memory recall ($read$) over memory learning ($write$). In this paper, we present BayesPCN, a hierarchical associative memory capable of performing continual one-shot memory writes without meta-learning. Moreover, BayesPCN is able to gradually forget past observations ($forget$) to free its memory. Experiments show that BayesPCN can recall corrupted i.i.d. high-dimensional data observed hundreds to a thousand ``timesteps'' ago without a large drop in recall ability compared to the state-of-the-art offline-learned parametric memory models.

        ----

        ## [2168] Optimizing Data Collection for Machine Learning

        **Authors**: *Rafid Mahmood, James Lucas, José M. Álvarez, Sanja Fidler, Marc T. Law*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/c1449acc2e64050d79c2830964f8515f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/c1449acc2e64050d79c2830964f8515f-Abstract-Conference.html)

        **Abstract**:

        Modern deep learning systems require huge data sets to achieve impressive performance, but there is little guidance on how much or what kind of data to collect. Over-collecting data incurs unnecessary present costs, while under-collecting may incur future costs and delay workflows. We propose a new paradigm for modeling the data collection workflow as a formal optimal data collection problem that allows designers to specify performance targets, collection costs, a time horizon, and penalties for failing to meet the targets. Additionally, this formulation generalizes to tasks requiring multiple data sources, such as labeled and unlabeled data used in semi-supervised learning. To solve our problem, we develop Learn-Optimize-Collect (LOC), which minimizes expected future collection costs. Finally, we numerically compare our framework to the conventional baseline of estimating data requirements by extrapolating from neural scaling laws. We significantly reduce the risks of failing to meet desired performance targets on several classification, segmentation, and detection tasks, while maintaining low total collection costs.

        ----

        ## [2169] DP-PCA: Statistically Optimal and Differentially Private PCA

        **Authors**: *Xiyang Liu, Weihao Kong, Prateek Jain, Sewoong Oh*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/c150ebe1b9d1ca0eb61502bf979fa87d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/c150ebe1b9d1ca0eb61502bf979fa87d-Abstract-Conference.html)

        **Abstract**:

        We study the canonical statistical task of  computing the principal component from    i.i.d.~data under differential privacy. Although extensively studied in literature,  existing solutions fall short on two key aspects: ($i$) even for Gaussian data, existing private algorithms   require the number of samples  $n$ to scale super-linearly with $d$, i.e., $n=\Omega(d^{3/2})$, to obtain non-trivial results while non-private PCA  requires only $n=O(d)$, and ($ii$) existing techniques suffer from a large error even when the variance in each data point is small. We propose DP-PCA method that uses a single-pass minibatch gradient descent style algorithm to overcome the above  limitations. For sub-Gaussian data, we provide nearly optimal statistical error rates even for $n=O(d \log d)$.

        ----

        ## [2170] Semantic Probabilistic Layers for Neuro-Symbolic Learning

        **Authors**: *Kareem Ahmed, Stefano Teso, Kai-Wei Chang, Guy Van den Broeck, Antonio Vergari*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/c182ec594f38926b7fcb827635b9a8f4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/c182ec594f38926b7fcb827635b9a8f4-Abstract-Conference.html)

        **Abstract**:

        We design a predictive layer for structured-output prediction (SOP) that can be plugged into any neural network guaranteeing its predictions are consistent with a set of predefined symbolic constraints. Our Semantic Probabilistic Layer (SPL) can model intricate correlations, and hard constraints, over a structured output space all while being amenable to end-to-end learning via maximum likelihood.SPLs combine exact probabilistic inference with logical reasoning in a clean and modular way, learning complex distributions and restricting their support to solutions of the constraint. As such, they can faithfully, and efficiently, model complex SOP tasks beyond the reach of alternative neuro-symbolic approaches. We empirically demonstrate that SPLs outperform these competitors in terms of accuracy on challenging SOP tasks such as hierarchical multi-label classification, pathfinding and preference learning, while retaining perfect constraint satisfaction.

        ----

        ## [2171] Adapting to Online Label Shift with Provable Guarantees

        **Authors**: *Yong Bai, Yu-Jie Zhang, Peng Zhao, Masashi Sugiyama, Zhi-Hua Zhou*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/c19c4def293b6a63db1ff27143dd4f10-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/c19c4def293b6a63db1ff27143dd4f10-Abstract-Conference.html)

        **Abstract**:

        The standard supervised learning paradigm works effectively when training data shares the same distribution as the upcoming testing samples. However, this stationary assumption is often violated in real-world applications, especially when testing data appear in an online fashion. In this paper, we formulate and investigate the problem of \emph{online label shift} (OLaS): the learner trains an initial model from the labeled offline data and then deploys it to an unlabeled online environment where the underlying label distribution changes over time but the label-conditional density does not. The non-stationarity nature and the lack of supervision make the problem challenging to be tackled. To address the difficulty, we construct a new unbiased risk estimator that utilizes the unlabeled data, which exhibits many benign properties albeit with potential non-convexity. Building upon that, we propose novel online ensemble algorithms to deal with the non-stationarity of the environments. Our approach enjoys optimal \emph{dynamic regret}, indicating that the performance is competitive with a clairvoyant who knows the online environments in hindsight and then chooses the best decision for each round. The obtained dynamic regret bound scales with the intensity and pattern of label distribution shift, hence exhibiting the adaptivity in the OLaS problem. Extensive experiments are conducted to validate the effectiveness and support our theoretical findings.

        ----

        ## [2172] CAGroup3D: Class-Aware Grouping for 3D Object Detection on Point Clouds

        **Authors**: *Haiyang Wang, Lihe Ding, Shaocong Dong, Shaoshuai Shi, Aoxue Li, Jianan Li, Zhenguo Li, Liwei Wang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/c1aaf7c3f306fe94f77236dc0756d771-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/c1aaf7c3f306fe94f77236dc0756d771-Abstract-Conference.html)

        **Abstract**:

        We present a novel two-stage fully sparse convolutional 3D object detection framework, named CAGroup3D. Our proposed method first generates some high-quality 3D proposals by leveraging the class-aware local group strategy on the object surface voxels with the same semantic predictions, which considers semantic consistency and diverse locality abandoned in previous bottom-up approaches. Then, to recover the features of missed voxels due to incorrect voxel-wise segmentation, we build a fully sparse convolutional RoI pooling module to directly aggregate fine-grained spatial information from backbone for further proposal refinement. It is memory-and-computation efficient and can better encode the geometry-specific features of each 3D proposal. Our model achieves state-of-the-art 3D detection performance with remarkable gains of +3.6% on ScanNet V2 and +2.6%  on SUN RGB-D in term of mAP@0.25. Code will be available at https://github.com/Haiyang-W/CAGroup3D.

        ----

        ## [2173] Revisiting Injective Attacks on Recommender Systems

        **Authors**: *Haoyang Li, Shimin Di, Lei Chen*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/c1bb0e3b062f0a443f2cc8a4ec4bb30d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/c1bb0e3b062f0a443f2cc8a4ec4bb30d-Abstract-Conference.html)

        **Abstract**:

        Recent studies have demonstrated that recommender systems (RecSys) are vulnerable to injective attacks.Given a limited fake user budget, attackers can inject fake users with carefully designed behaviors into the open platforms, making RecSys recommend a target item to more real users for profits. In this paper, we first revisit existing attackers and reveal that they suffer from the difficulty-agnostic and diversity-deficit issues. Existing attackers concentrate their efforts on difficult users who have low tendencies toward the target item, thus reducing their effectiveness. Moreover, they are incapable of affecting the target RecSys to recommend the target item to real users in a diverse manner, because their generated fake user behaviors are dominated  by large communities. To alleviate these two issues, we propose a difficulty and diversity aware attacker, namely DADA. We design the difficulty-aware and diversity-aware objectives to enable easy users from various communities to contribute more weights when optimizing attackers. By incorporating these two objectives, the proposed attacker DADA can concentrate on easy users while also affecting a broader range of real users simultaneously, thereby boosting the effectiveness. Extensive experiments on three real-world datasets demonstrate the effectiveness of our proposed attacker.

        ----

        ## [2174] Learning-based Motion Planning in Dynamic Environments Using GNNs and Temporal Encoding

        **Authors**: *Ruipeng Zhang, Chenning Yu, Jingkai Chen, Chuchu Fan, Sicun Gao*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/c1d4798259250f2b4fe38614b48f8996-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/c1d4798259250f2b4fe38614b48f8996-Abstract-Conference.html)

        **Abstract**:

        Learning-based methods have shown promising performance for accelerating motion planning, but mostly in the setting of static environments. For the more challenging problem of planning in dynamic environments, such as multi-arm assembly tasks and human-robot interaction, motion planners need to consider the trajectories of the dynamic obstacles and reason about temporal-spatial interactions in very large state spaces. We propose a GNN-based approach that uses temporal encoding and imitation learning with data aggregation for learning both the embeddings and the edge prioritization policies. Experiments show that the proposed methods can significantly accelerate online planning over state-of-the-art complete dynamic planning algorithms. The learned models can often reduce costly collision checking operations by more than 1000x, and thus accelerating planning by up to 95%, while achieving high success rates on hard instances as well.

        ----

        ## [2175] An empirical analysis of compute-optimal large language model training

        **Authors**: *Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, Tom Hennigan, Eric Noland, Katherine Millican, George van den Driessche, Bogdan Damoc, Aurelia Guy, Simon Osindero, Karen Simonyan, Erich Elsen, Oriol Vinyals, Jack W. Rae, Laurent Sifre*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/c1e2faff6f588870935f114ebe04a3e5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/c1e2faff6f588870935f114ebe04a3e5-Abstract-Conference.html)

        **Abstract**:

        We investigate the optimal model size and number of tokens for training a transformer language model under a given compute budget. We find that current large language models are significantly undertrained, a consequence of the recent focus on scaling language models whilst keeping the amount of training data constant. By training over 400 language models ranging from 70 million to over 16 billion parameters on 5 to 500 billion tokens, we find that for compute-optimal training, the model size and the number of training tokens should be scaled equally: for every doubling of model size the number of training tokens should also be doubled. We test this hypothesis by training a predicted compute-optimal model, Chinchilla, that uses the same compute budget as Gopher but with 70B parameters and 4$\times$ more data. Chinchilla uniformly and significantly outperformsGopher (280B), GPT-3 (175B), Jurassic-1 (178B), and Megatron-Turing NLG (530B) on a large range of downstream evaluation tasks. This also means that Chinchilla uses substantially less compute for fine-tuning and inference, greatly facilitating downstream usage. As a highlight, Chinchilla reaches a state-of-the-art average accuracy of 67.5% on the MMLU benchmark, a 7% improvement over Gopher.

        ----

        ## [2176] Robust Imitation via Mirror Descent Inverse Reinforcement Learning

        **Authors**: *Dong-Sig Han, Hyunseo Kim, Hyundo Lee, Je-Hwan Ryu, Byoung-Tak Zhang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/c1f7b1ed763e9c75e4db74b49b76db5f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/c1f7b1ed763e9c75e4db74b49b76db5f-Abstract-Conference.html)

        **Abstract**:

        Recently, adversarial imitation learning has shown a scalable reward acquisition method for inverse reinforcement learning (IRL) problems. However, estimated reward signals often become uncertain and fail to train a reliable statistical model since the existing methods tend to solve hard optimization problems directly. Inspired by a first-order optimization method called mirror descent, this paper proposes to predict a sequence of reward functions, which are iterative solutions for a constrained convex problem. IRL solutions derived by mirror descent are tolerant to the uncertainty incurred by target density estimation since the amount of reward learning is regulated with respect to local geometric constraints. We prove that the proposed mirror descent update rule ensures robust minimization of a Bregman divergence in terms of a rigorous regret bound of $\mathcal{O}(1/T)$ for step sizes $\{\eta_t\}_{t=1}^{T}$. Our IRL method was applied on top of an adversarial framework, and it outperformed existing adversarial methods in an extensive suite of benchmarks.

        ----

        ## [2177] Characterizing Datapoints via Second-Split Forgetting

        **Authors**: *Pratyush Maini, Saurabh Garg, Zachary C. Lipton, J. Zico Kolter*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/c20447998d6c624b4b97d4466a3bfff5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/c20447998d6c624b4b97d4466a3bfff5-Abstract-Conference.html)

        **Abstract**:

        Researchers investigating example hardness have increasingly focused on the dynamics by which neural networks learn and forget examples throughout training. Popular metrics derived from these dynamics include (i) the epoch at which examples are first correctly classified; (ii) the number of times their predictions flip during training; and (iii) whether their prediction flips if they are held out. However, these metrics do not distinguish among examples that are hard for distinct reasons, such as membership in a rare subpopulation, being mislabeled, or belonging to a complex subpopulation. In this paper, we propose second-split forgetting time (SSFT), a complementary metric that tracks the epoch (if any) after which an original training example is forgotten as the network is fine-tuned on a randomly held out partition of the data.  Across multiple benchmark datasets and modalities,  we demonstrate that mislabeled examples are forgotten quickly, and seemingly rare examples are forgotten comparatively slowly.  By contrast, metrics only considering the first split learning dynamics struggle to differentiate the two.  At large learning rates, SSFT tends to be robust across architectures,  optimizers, and random seeds. From a practical standpoint, the SSFT can (i) help to identify mislabeled samples, the removal of which improves generalization; and (ii) provide insights about failure modes.  Through theoretical analysis addressing overparameterized linear models, we provide insights into how the observed phenomena may arise.

        ----

        ## [2178] VoiceBlock: Privacy through Real-Time Adversarial Attacks with Audio-to-Audio Models

        **Authors**: *Patrick O'Reilly, Andreas Bugler, Keshav Bhandari, Max Morrison, Bryan Pardo*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/c204d12afa0175285e5aac65188808b4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/c204d12afa0175285e5aac65188808b4-Abstract-Conference.html)

        **Abstract**:

        As governments and corporations adopt deep learning systems to collect and analyze user-generated audio data, concerns about security and privacy naturally emerge in areas such as automatic speaker recognition. While audio adversarial examples offer one route to mislead or evade these invasive systems, they are typically crafted through time-intensive offline optimization, limiting their usefulness in streaming contexts. Inspired by architectures for audio-to-audio tasks such as denoising and speech enhancement, we propose a neural network model capable of adversarially modifying a user's audio stream in real-time. Our model learns to apply a time-varying finite impulse response (FIR) filter to outgoing audio, allowing for effective and inconspicuous perturbations on a small fixed delay suitable for streaming tasks. We demonstrate our model is highly effective at de-identifying user speech from speaker recognition and able to transfer to an unseen recognition system. We conduct a perceptual study and find that our method produces perturbations significantly less perceptible than baseline anonymization methods, when controlling for effectiveness. Finally, we provide an implementation of our model capable of running in real-time on a single CPU thread. Audio examples and code can be found at https://interactiveaudiolab.github.io/project/voiceblock.html.

        ----

        ## [2179] A Theoretical View on Sparsely Activated Networks

        **Authors**: *Cenk Baykal, Nishanth Dikkala, Rina Panigrahy, Cyrus Rashtchian, Xin Wang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/c2201e444d2b22a10ca50116a522b9a9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/c2201e444d2b22a10ca50116a522b9a9-Abstract-Conference.html)

        **Abstract**:

        Deep and wide neural networks successfully fit very complex functions today, but dense models are starting to be prohibitively expensive for inference. To mitigate this, one promising research direction is networks that activate a sparse subgraph of the network. The subgraph is chosen by a data-dependent routing function, enforcing a fixed mapping of inputs to subnetworks (e.g., the Mixture of Experts (MoE) paradigm in Switch Transformers). However, there is no theoretical grounding for these sparsely activated models. As our first contribution, we present a formal model of data-dependent sparse networks that captures salient aspects of popular architectures. Then, we show how to construct sparse networks that provably match the approximation power and total size of dense networks on Lipschitz functions. The sparse networks use much fewer inference operations than dense networks, leading to a faster forward pass. The key idea is to use locality sensitive hashing on the input vectors and then interpolate the function in subregions of the input space. This offers a theoretical insight into why sparse networks work well in practice. Finally, we present empirical findings that support our theory; compared to dense networks, sparse networks give a favorable trade-off between number of active units and approximation quality.

        ----

        ## [2180] Invariance Learning based on Label Hierarchy

        **Authors**: *Shoji Toyota, Kenji Fukumizu*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/c23ccf9eedf87e4380e92b75b24955bb-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/c23ccf9eedf87e4380e92b75b24955bb-Abstract-Conference.html)

        **Abstract**:

        Deep Neural Networks inherit spurious correlations embedded in training data and hence may fail to predict desired labels on unseen domains (or environments), which have different distributions from the domain to provide training data. Invariance Learning (IL) has been developed recently to overcome this shortcoming; using training data in many domains, IL estimates such a predictor that is invariant to a change of domain.  However, the requirement of training data in multiple domains is a strong restriction of using IL, since it demands expensive annotation. We propose a novel IL framework to overcome this problem. Assuming the availability of data from multiple domains for a higher level of classification task, for which the labeling cost is lower, we estimate an invariant predictor for the target classification task with training data gathered in a single domain.  Additionally, we propose two cross-validation methods for selecting hyperparameters of invariance regularization, which has not been addressed properly in existing IL methods.  The effectiveness of the proposed framework, including the cross-validation, is demonstrated empirically. Theoretical analysis reveals that our framework can estimate the desirable invariant predictor with a hyperparameter fixed correctly, and that such a preferable hyperparameter is chosen by the proposed CV methods under some conditions.

        ----

        ## [2181] Effects of Data Geometry in Early Deep Learning

        **Authors**: *Saket Tiwari, George Konidaris*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/c23f3852601f6dd7f0b39223d031806f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/c23f3852601f6dd7f0b39223d031806f-Abstract-Conference.html)

        **Abstract**:

        Deep neural networks can approximate functions on different types of data, from images to graphs, with varied underlying structure. This underlying structure can be viewed as the geometry of the data manifold. By extending recent advances in the theoretical understanding of neural networks, we study how a randomly initialized neural network with piecewise linear activation splits the data manifold into regions where the neural network behaves as a linear function.   We derive bounds on the density of boundary of linear regions and the distance to these boundaries on the data manifold. This leads to insights into the expressivity of randomly initialized deep neural networks on non-Euclidean data sets. We empirically corroborate our theoretical results using a toy supervised learning problem. Our experiments demonstrate that number of linear regions varies across manifolds and the results hold with changing neural network architectures. We further demonstrate how the complexity of linear regions is different on the low dimensional manifold of images as compared to the Euclidean space, using the MetFaces dataset.

        ----

        ## [2182] When to Intervene: Learning Optimal Intervention Policies for Critical Events

        **Authors**: *Niranjan Damera Venkata, Chiranjib Bhattacharyya*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/c26a8494fe31695db965ae8b7244b7c1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/c26a8494fe31695db965ae8b7244b7c1-Abstract-Conference.html)

        **Abstract**:

        Providing a timely intervention before the onset of a critical event, such as a system failure, is of importance in many industrial settings. Before the onset of the critical event, systems typically exhibit behavioral changes which often manifest as stochastic co-variate observations which may be leveraged to trigger intervention. In this paper, for the first time, we formulate the problem of finding an optimally timed intervention (OTI) policy as minimizing the expected residual time to event, subject to a constraint on the probability of missing the event. Existing machine learning approaches to intervention on critical events focus on predicting event occurrence within a pre-defined window (a classification problem) or predicting time-to-event (a regression problem). Interventions are then triggered by setting model thresholds. These are heuristic-driven, lacking guarantees regarding optimality. To model the evolution of system behavior, we introduce the concept of a hazard rate process. We show that the OTI problem is equivalent to an optimal stopping problem on the associated hazard rate process. This key link has not been explored in literature. Under Markovian assumptions on the hazard rate process, we show that an OTI policy at any time can be analytically determined from the conditional hazard rate function at that time. Further, we show that our theory includes, as a special case, the important class of neural hazard rate processes generated by recurrent neural networks (RNNs). To model such processes, we propose a dynamic deep recurrent survival analysis (DDRSA) architecture, introducing an RNN encoder into the static DRSA setting. Finally, we demonstrate RNN-based OTI policies with experiments and show that they outperform popular intervention methods

        ----

        ## [2183] Draft-and-Revise: Effective Image Generation with Contextual RQ-Transformer

        **Authors**: *Doyup Lee, Chiheon Kim, Saehoon Kim, Minsu Cho, Wook-Shin Han*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/c276c3303c0723c83a43b95a44a1fcbf-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/c276c3303c0723c83a43b95a44a1fcbf-Abstract-Conference.html)

        **Abstract**:

        Although autoregressive models have achieved promising results on image generation, their unidirectional generation process prevents the resultant images from fully reflecting global contexts. To address the issue, we propose an effective image generation framework of \emph{Draft-and-Revise} with \emph{Contextual RQ-transformer} to consider global contexts during the generation process. As a generalized VQ-VAE, RQ-VAE first represents a high-resolution image as a sequence of discrete code stacks. After code stacks in the sequence are randomly masked, Contextual RQ-Transformer is trained to infill the masked code stacks based on the unmasked contexts of the image. Then, we propose the two-phase decoding, Draft-and-Revise, for Contextual RQ-Transformer to generates an image, while fully exploiting the global contexts of the image during the generation process. Specifically. in the \emph{draft} phase, our model first focuses on generating diverse images despite rather low quality. Then, in the \emph{revise} phase, the model iteratively improves the quality of images, while preserving the global contexts of generated images. In experiments, our method achieves state-of-the-art results on conditional image generation. We also validate that the Draft-and-Revise decoding can achieve high performance by effectively controlling the quality-diversity trade-off in image generation.

        ----

        ## [2184] Finite-Sample Maximum Likelihood Estimation of Location

        **Authors**: *Shivam Gupta, Jasper C. H. Lee, Eric Price, Paul Valiant*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/c27cfb05a2e9eb579698419b25234ffb-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/c27cfb05a2e9eb579698419b25234ffb-Abstract-Conference.html)

        **Abstract**:

        We consider 1-dimensional location estimation, where we estimate a parameter $\lambda$ from $n$ samples $\lambda + \eta_i$, with each $\eta_i$ drawn i.i.d. from a known distribution $f$. For fixed $f$ the maximum-likelihood estimate (MLE) is well-known to be optimal in the limit as $n \to \infty$: it is asymptotically normal with variance matching the Cramer-Rao lower bound of $\frac{1}{n\mathcal{I}}$, where $\mathcal{I}$ is the Fisher information of $f$. However, this bound does not hold for finite $n$, or when $f$ varies with $n$. We show for arbitrary $f$ and $n$ that one can recover a similar theory based on the Fisher information of a smoothed version of $f$, where the smoothing radius decays with $n$.

        ----

        ## [2185] GENIE: Higher-Order Denoising Diffusion Solvers

        **Authors**: *Tim Dockhorn, Arash Vahdat, Karsten Kreis*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/c281c5a17ad2e55e1ac1ca825071f991-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/c281c5a17ad2e55e1ac1ca825071f991-Abstract-Conference.html)

        **Abstract**:

        Denoising diffusion models (DDMs) have emerged as a powerful class of generative models. A forward diffusion process slowly perturbs the data, while a deep model learns to gradually denoise. Synthesis amounts to solving a differential equation (DE) defined by the learnt model. Solving the DE requires slow iterative solvers for high-quality generation. In this work, we propose Higher-Order Denoising Diffusion Solvers (GENIE): Based on truncated Taylor methods, we derive a novel higher-order solver that significantly accelerates synthesis. Our solver relies on higher-order gradients of the perturbed data distribution, that is, higher-order score functions. In practice, only Jacobian-vector products (JVPs) are required and we propose to extract them from the first-order score network via automatic differentiation. We then distill the JVPs into a separate neural network that allows us to efficiently compute the necessary higher-order terms for our novel sampler during synthesis. We only need to train a small additional head on top of the first-order score network. We validate GENIE on multiple image generation benchmarks and demonstrate that GENIE outperforms all previous solvers. Unlike recent methods that fundamentally alter the generation process in DDMs, our GENIE solves the true generative DE and still enables applications such as encoding and guided sampling. Project page and code: https://nv-tlabs.github.io/GENIE.

        ----

        ## [2186] Provably Adversarially Robust Detection of Out-of-Distribution Data (Almost) for Free

        **Authors**: *Alexander Meinke, Julian Bitterwolf, Matthias Hein*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/c2c62117283dda155db754e54dbe8d71-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/c2c62117283dda155db754e54dbe8d71-Abstract-Conference.html)

        **Abstract**:

        The application of machine learning in safety-critical systems requires a reliable assessment of uncertainty.However, deep neural networks are known to produce highly overconfident predictions on out-of-distribution (OOD) data.Even if trained to be non-confident on OOD data, one can still adversarially manipulate OOD data so that the classifier again assigns high confidence to the manipulated samples.We show that two previously published defenses can be broken by better adapted attacks, highlighting the importance of robustness guarantees around OOD data.Since the existing method for this task is hard to train and significantly limits accuracy, we construct a classifier that can simultaneously achieve provably adversarially robust OOD detection and high clean accuracy.Moreover, by slightly modifying the classifier's architecture our method provably avoids the asymptotic overconfidence problem of standard neural networks.We provide code for all our experiments.

        ----

        ## [2187] Efficient Graph Similarity Computation with Alignment Regularization

        **Authors**: *Wei Zhuo, Guang Tan*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/c2ce2f2701c10a2b2f2ea0bfa43cfaa3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/c2ce2f2701c10a2b2f2ea0bfa43cfaa3-Abstract-Conference.html)

        **Abstract**:

        We consider the graph similarity computation (GSC) task based on graph edit distance (GED) estimation. State-of-the-art methods treat GSC as a learning-based prediction task using Graph Neural Networks (GNNs). To capture fine-grained interactions between pair-wise graphs, these methods mostly contain a node-level matching module in the end-to-end learning pipeline, which causes high computational costs in both the training and inference stages. We show that the expensive node-to-node matching module is not necessary for GSC, and high-quality learning can be attained with a simple yet powerful regularization technique, which we call the Alignment Regularization (AReg). In the training stage, the AReg term imposes a node-graph correspondence constraint on the GNN encoder. In the inference stage, the graph-level representations learned by the GNN encoder are directly used to compute the similarity score without using AReg again to speed up inference. We further propose a multi-scale GED discriminator to enhance the expressive ability of the learned representations. Extensive experiments on real-world datasets demonstrate the effectiveness, efficiency and transferability of our approach.

        ----

        ## [2188] Tsetlin Machine for Solving Contextual Bandit Problems

        **Authors**: *Raihan Seraj, Jivitesh Sharma, Ole-Christoffer Granmo*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/c2d550cf3b2e177deb2d1720fb1e2710-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/c2d550cf3b2e177deb2d1720fb1e2710-Abstract-Conference.html)

        **Abstract**:

        This paper introduces an interpretable contextual bandit algorithm using Tsetlin Machines, which solves complex pattern recognition tasks using  propositional (Boolean) logic. The proposed bandit learning algorithm relies on straightforward bit manipulation, thus simplifying computation and interpretation. We then present a mechanism for performing Thompson sampling with Tsetlin Machine, given its non-parametric nature. Our empirical analysis shows that Tsetlin Machine as a base contextual bandit learner outperforms other popular base learners on eight out of nine datasets. We further analyze the interpretability of our learner, investigating how arms are selected based on propositional expressions that model the context.

        ----

        ## [2189] Prune and distill: similar reformatting of image information along rat visual cortex and deep neural networks

        **Authors**: *Paolo Muratore, Sina Tafazoli, Eugenio Piasini, Alessandro Laio, Davide Zoccolan*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/c2d82a425af4c18a35049899fea5ee82-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/c2d82a425af4c18a35049899fea5ee82-Abstract-Conference.html)

        **Abstract**:

        Visual object recognition has been extensively studied in both neuroscience and computer vision. Recently, the most popular class of artificial systems for this task, deep convolutional neural networks (CNNs), has been shown to provide excellent models for its functional analogue in the brain, the ventral stream in visual cortex. This has prompted questions on what, if any, are the common principles underlying the reformatting of visual information as it flows through a CNN or the ventral stream. Here we consider some prominent statistical patterns that are known to exist in the internal representations of either CNNs or the visual cortex and look for them in the other system. We show that intrinsic dimensionality (ID) of object representations along the rat homologue of the ventral stream presents two distinct expansion-contraction phases, as previously shown for CNNs. Conversely, in CNNs, we show that training results in both distillation and active pruning (mirroring the increase in ID) of low- to middle-level image information in single units, as representations gain the ability to support invariant discrimination, in agreement with previous observations in rat visual cortex. Taken together, our findings suggest that CNNs and visual cortex share a similarly tight relationship between dimensionality expansion/reduction of object representations and reformatting of image information.

        ----

        ## [2190] Graph Scattering beyond Wavelet Shackles

        **Authors**: *Christian Koke, Gitta Kutyniok*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/c2f2230abc7ccf669f403be881d3ffb7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/c2f2230abc7ccf669f403be881d3ffb7-Abstract-Conference.html)

        **Abstract**:

        This work develops a flexible and mathematically sound framework for the design and analysis of graph scattering networks with variable branching ratios and generic functional calculus filters.Spectrally-agnostic stability guarantees for node- and graph-level perturbations are derived; the vertex-set non-preserving case is treated by utilizing recently developed mathematical-physics based tools. Energy propagation through the network layers is investigated and related to truncation stability. New methods of graph-level feature aggregation are introduced and stability of the resulting composite scattering architectures is established. Finally, scattering transforms are extended to edge- and higher order tensorial input. Theoretical results are complemented by numerical investigations: Suitably chosen scattering networks conforming to the developed theory perform better than traditional graph-wavelet based scattering approaches in social network graph classification tasks andsignificantly outperform other graph-based learning approaches to regression of quantum-chemical energies on QM$7$.

        ----

        ## [2191] Matryoshka Representation Learning

        **Authors**: *Aditya Kusupati, Gantavya Bhatt, Aniket Rege, Matthew Wallingford, Aditya Sinha, Vivek Ramanujan, William Howard-Snyder, Kaifeng Chen, Sham M. Kakade, Prateek Jain, Ali Farhadi*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/c32319f4868da7613d78af9993100e42-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/c32319f4868da7613d78af9993100e42-Abstract-Conference.html)

        **Abstract**:

        Learned representations are a central component in modern ML systems, serving a multitude of downstream tasks. When training such representations, it is often the case that computational and statistical constraints for each downstream task are unknown. In this context rigid, fixed capacity representations can be either over or under-accommodating to the task at hand. This leads us to ask: can we design a flexible representation that can adapt to multiple downstream tasks with varying computational resources? Our main contribution is Matryoshka Representation Learning (MRL) which encodes information at different granularities and allows a single embedding to adapt to the computational constraints of downstream tasks. MRL minimally modifies existing representation learning pipelines and imposes no additional cost during inference and deployment. MRL learns coarse-to-fine representations that are at least as accurate and rich as independently trained low-dimensional representations. The flexibility within the learned Matryoshka Representations offer: (a) up to $\mathbf{14}\times$ smaller embedding size for ImageNet-1K classification at the same level of accuracy; (b) up to $\mathbf{14}\times$ real-world speed-ups for large-scale retrieval on ImageNet-1K and 4K; and (c) up to $\mathbf{2}\%$ accuracy improvements for long-tail few-shot classification, all while being as robust as the original representations. Finally, we show that MRL extends seamlessly to web-scale datasets (ImageNet, JFT) across various modalities -- vision (ViT, ResNet), vision + language (ALIGN) and language (BERT). MRL code and pretrained models are open-sourced at https://github.com/RAIVNLab/MRL.

        ----

        ## [2192] Off-Policy Evaluation with Deficient Support Using Side Information

        **Authors**: *Nicolò Felicioni, Maurizio Ferrari Dacrema, Marcello Restelli, Paolo Cremonesi*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/c32be49c09eec3aad1f2bb587543e7f6-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/c32be49c09eec3aad1f2bb587543e7f6-Abstract-Conference.html)

        **Abstract**:

        The Off-Policy Evaluation (OPE) problem consists in evaluating the performance of new policies from the data collected by another one. OPE is crucial when evaluating a new policy online is too expensive or risky. Many of the state-of-the-art OPE estimators are based on the Inverse Propensity Scoring (IPS) technique, which provides an unbiased estimator when the full support assumption holds, i.e., when the logging policy assigns a non-zero probability to each action. However, there are several scenarios where this assumption does not hold in practice, i.e., there is deficient support, and the IPS estimator is biased in the general case.In this paper, we consider two alternative estimators for the deficient support OPE problem. We first show how to adapt an estimator that was originally proposed for a different domain to the deficient support setting.Then, we propose another estimator, which is a novel contribution of this paper.These estimators exploit additional information about the actions, which we call side information, in order to make reliable estimates on the unsupported actions. Under alternative assumptions that do not require full support, we show that the considered estimators are unbiased.We also provide a theoretical analysis of the concentration when relaxing all the assumptions. Finally, we provide an experimental evaluation showing how the considered estimators are better suited for the deficient support setting compared to the baselines.

        ----

        ## [2193] The Nature of Temporal Difference Errors in Multi-step Distributional Reinforcement Learning

        **Authors**: *Yunhao Tang, Rémi Munos, Mark Rowland, Bernardo Ávila Pires, Will Dabney, Marc G. Bellemare*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/c32de883c5fe94d33a20a717fad53971-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/c32de883c5fe94d33a20a717fad53971-Abstract-Conference.html)

        **Abstract**:

        We study the multi-step off-policy learning approach to distributional RL. Despite the apparent similarity between value-based RL and distributional RL, our study reveals intriguing and fundamental differences between the two cases in the multi-step setting. We identify a novel notion of path-dependent distributional TD error, which is indispensable for principled multi-step distributional RL. The distinction from the value-based case bears important implications on concepts such as backward-view algorithms. Our work provides the first theoretical guarantees on multi-step off-policy distributional RL algorithms, including results that apply to the small number of existing approaches to multi-step distributional RL. In addition, we derive a novel algorithm, Quantile Regression-Retrace, which leads to a deep RL agent QR-DQN-Retrace that shows empirical improvements over QR-DQN on the Atari-57 benchmark. Collectively, we shed light on how unique challenges in multi-step distributional RL can be addressed both in theory and practice.

        ----

        ## [2194] GraphDE: A Generative Framework for Debiased Learning and Out-of-Distribution Detection on Graphs

        **Authors**: *Zenan Li, Qitian Wu, Fan Nie, Junchi Yan*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/c34262c35aa5f8c1a091822cbb2020c2-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/c34262c35aa5f8c1a091822cbb2020c2-Abstract-Conference.html)

        **Abstract**:

        Despite the remarkable success of graph neural networks (GNNs) for graph representation learning, they are generally built on the (unreliable) i.i.d. assumption across training and testing data. However, real-world graph data are universally comprised of outliers in training set and out-of-distribution (OOD) testing samples from unseen domains, which solicits effective models for i) debiased learning and ii) OOD detection, towards general trustworthy purpose. In this paper, we first mathematically formulate the two challenging problems for graph data and take an initiative on tackling them under a unified probabilistic model. Specifically, we model the graph generative process to characterize the distribution shifts of graph data together with an additionally introduced latent environment variable as an indicator. We then define a variational distribution, i.e., a recognition model, to infer the environment during training of GNN. By instantiating the generative models as two-component mixtures, we derive a tractable learning objective and theoretically justify that the model can i) automatically identify and down-weight outliers in the training procedure, and ii) induce an effective OOD detector simultaneously. Experiments on diverse datasets with different types of OOD data prove that our model consistently outperforms strong baselines for both debiasing and OOD detection tasks. The source code has been made publicly available at https://github.com/Emiyalzn/GraphDE.

        ----

        ## [2195] Expectation-Maximization Contrastive Learning for Compact Video-and-Language Representations

        **Authors**: *Peng Jin, Jinfa Huang, Fenglin Liu, Xian Wu, Shen Ge, Guoli Song, David A. Clifton, Jie Chen*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/c355566ce402de341c3320cf69a10750-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/c355566ce402de341c3320cf69a10750-Abstract-Conference.html)

        **Abstract**:

        Most video-and-language representation learning approaches employ contrastive learning, e.g., CLIP, to project the video and text features into a common latent space according to the semantic similarities of text-video pairs. However, such learned shared latent spaces are not often optimal, and the modality gap between visual and textual representation can not be fully eliminated. In this paper, we propose Expectation-Maximization Contrastive Learning (EMCL) to learn compact video-and-language representations. Specifically, we use the Expectation-Maximization algorithm to find a compact set of bases for the latent space, where the features could be concisely represented as the linear combinations of these bases. Such feature decomposition of video-and-language representations reduces the rank of the latent space, resulting in increased representing power for the semantics. Extensive experiments on three benchmark text-video retrieval datasets prove that our EMCL can learn more discriminative video-and-language representations than previous methods, and significantly outperform previous state-of-the-art methods across all metrics. More encouragingly, the proposed method can be applied to boost the performance of existing approaches either as a jointly training layer or an out-of-the-box inference module with no extra training, making it easy to be incorporated into any existing methods.

        ----

        ## [2196] Accelerating Sparse Convolution with Column Vector-Wise Sparsity

        **Authors**: *Yijun Tan, Kai Han, Kang Zhao, Xianzhi Yu, Zidong Du, Yunji Chen, Yunhe Wang, Jun Yao*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/c383e44d9a878d1982d9abb838bd5d8a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/c383e44d9a878d1982d9abb838bd5d8a-Abstract-Conference.html)

        **Abstract**:

        Weight sparsity is a promising approach to reducing the model size and computation cost of convolutional neural networks (CNNs). Nevertheless, non-zero weights often distribute randomly in sparse CNN models, introducing enormous difficulty in obtaining actual speedup on common hardware (e.g., GPU) over their dense counterparts. Existing acceleration solutions either require hardware modifications for irregular memory access support or rely on a partially structured sparsity pattern. Neither of these methods is capable of achieving fruitful speedup on convolution layers.In this work, we propose an algorithm-software co-designed sparse convolution based on a novel out-vector-wise (OVW) sparse pattern. Building on the insight that vertical vector integrity can preserve continuous memory access in IM2COL,  the OVW pattern treats a $V\times1$ vector as an entirety. To reduce the error caused by sparsity, we propose an equivalent transformation process, i.e., clustering-based channel permutation, to gather similar rows together. Experimental evaluations demonstrate that our method achieves a $1.7\times$ and $3.2\times$ speedup over the SOTA solution and the dense convolution of ResNet50 on NVIDIA V100 at 75\% sparsity, respectively, with only negligible accuracy loss. Moreover, compared to the SOTA solution that achieves speedups only on data with 60\% sparsity or more, our method begins to obtain speedups on data with only 10\% sparsity.

        ----

        ## [2197] GPT3int8(): 8-bit Matrix Multiplication for Transformers at Scale

        **Authors**: *Tim Dettmers, Mike Lewis, Younes Belkada, Luke Zettlemoyer*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/c3ba4962c05c49636d4c6206a97e9c8a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/c3ba4962c05c49636d4c6206a97e9c8a-Abstract-Conference.html)

        **Abstract**:

        Large language models have been widely adopted but require significant GPU memory for inference. We develop a procedure for Int8 matrix multiplication for feed-forward and attention projection layers in transformers, which cut the memory needed for inference by half while retaining full precision performance. With our method, a 175B parameter 16/32-bit checkpoint can be loaded, converted to Int8, and used immediately without performance degradation. This is made possible by understanding and working around properties of highly systematic emergent features in transformer language models that dominate attention and transformer predictive performance. To cope with these features, we develop a two-part quantization procedure, {\bf LLM.int8()}. We first use vector-wise quantization with separate normalization constants for each inner product in the matrix multiplication, to quantize most of the features. However, for the emergent outliers, we also include a new mixed-precision decomposition scheme, which isolates the outlier feature dimensions into a 16-bit matrix multiplication while still more than 99.9\% of values are multiplied in 8-bit. Using LLM.int8(), we show empirically it is possible to perform inference in LLMs with up to 175B parameters without any performance degradation. This result makes such models much more accessible, for example making it possible to use OPT-175B/BLOOM on a single server with consumer GPUs. We open source our software.

        ----

        ## [2198] Posterior Refinement Improves Sample Efficiency in Bayesian Neural Networks

        **Authors**: *Agustinus Kristiadi, Runa Eschenhagen, Philipp Hennig*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/c3f9c42965b7572d4f06ec0e983df0aa-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/c3f9c42965b7572d4f06ec0e983df0aa-Abstract-Conference.html)

        **Abstract**:

        Monte Carlo (MC) integration is the de facto method for approximating the predictive distribution of Bayesian neural networks (BNNs). But, even with many MC samples, Gaussian-based BNNs could still yield bad predictive performance due to the posterior approximation's error. Meanwhile, alternatives to MC integration are expensive. In this work, we experimentally show that the key to good MC-approximated predictive distributions is the quality of the approximate posterior itself. However, previous methods for obtaining accurate posterior approximations are expensive and non-trivial to implement. We, therefore, propose to refine Gaussian approximate posteriors with normalizing flows. When applied to last-layer BNNs, it yields a simple, cost-efficient, post hoc method for improving pre-existing parametric approximations. We show that the resulting posterior approximation is competitive with even the gold-standard full-batch Hamiltonian Monte Carlo.

        ----

        ## [2199] Micro and Macro Level Graph Modeling for Graph Variational Auto-Encoders

        **Authors**: *Kiarash Zahirnia, Oliver Schulte, Parmis Naddaf, Ke Li*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/c400474e8a36d0812fdee52739288b12-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/c400474e8a36d0812fdee52739288b12-Abstract-Conference.html)

        **Abstract**:

        Generative models for graph data are an important research topic in machine learning. Graph data comprise two levels that are typically analyzed separately: node-level properties such as the existence of a link between a pair of nodes, and global aggregate graph-level statistics, such as motif counts.This paper proposes a new multi-level framework that jointly models node-level properties and graph-level statistics, as mutually reinforcing sources of information.  We introduce a new micro-macro training objective for graph generation that combines node-level and graph-level losses.  We utilize the micro-macro objective to improve graph generation with a GraphVAE, a well-established model based on graph-level latent variables, that provides fast training and generation time for medium-sized graphs. Our experiments show that adding micro-macro modeling to the GraphVAE model improves graph quality scores up to 2 orders of magnitude on five benchmark datasets, while maintaining the GraphVAE generation speed advantage.

        ----

        

[Go to the previous page](NIPS-2022-list10.md)

[Go to the next page](NIPS-2022-list12.md)

[Go to the catalog section](README.md)