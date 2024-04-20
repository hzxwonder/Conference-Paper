## [600] Learning to Boost Training by Periodic Nowcasting Near Future Weights

        **Authors**: *Jinhyeok Jang, Woo-han Yun, Won Hwa Kim, Youngwoo Yoon, Jaehong Kim, Jaeyeon Lee, ByungOk Han*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/jang23b.html](https://proceedings.mlr.press/v202/jang23b.html)

        **Abstract**:

        Recent complicated problems require large-scale datasets and complex model architectures, however, it is difficult to train such large networks due to high computational issues. Significant efforts have been made to make the training more efficient such as momentum, learning rate scheduling, weight regularization, and meta-learning. Based on our observations on 1) high correlation between past eights and future weights, 2) conditions for beneficial weight prediction, and 3) feasibility of weight prediction, we propose a more general framework by intermittently skipping a handful of epochs by periodically forecasting near future weights, i.e., a Weight Nowcaster Network (WNN). As an add-on module, WNN predicts the future weights to make the learning process faster regardless of tasks and architectures. Experimental results show that WNN can significantly save actual time cost for training with an additional marginal time to train WNN. We validate the generalization capability of WNN under various tasks, and demonstrate that it works well even for unseen tasks. The code and pre-trained model are available at https://github.com/jjh6297/WNN.

        ----

        ## [601] Unscented Autoencoder

        **Authors**: *Faris Janjos, Lars Rosenbaum, Maxim Dolgov, J. Marius Zoellner*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/janjos23a.html](https://proceedings.mlr.press/v202/janjos23a.html)

        **Abstract**:

        The Variational Autoencoder (VAE) is a seminal approach in deep generative modeling with latent variables. Interpreting its reconstruction process as a nonlinear transformation of samples from the latent posterior distribution, we apply the Unscented Transform (UT) – a well-known distribution approximation used in the Unscented Kalman Filter (UKF) from the field of filtering. A finite set of statistics called sigma points, sampled deterministically, provides a more informative and lower-variance posterior representation than the ubiquitous noise-scaling of the reparameterization trick, while ensuring higher-quality reconstruction. We further boost the performance by replacing the Kullback-Leibler (KL) divergence with the Wasserstein distribution metric that allows for a sharper posterior. Inspired by the two components, we derive a novel, deterministic-sampling flavor of the VAE, the Unscented Autoencoder (UAE), trained purely with regularization-like terms on the per-sample posterior. We empirically show competitive performance in Fréchet Inception Distance scores over closely-related models, in addition to a lower training variance than the VAE.

        ----

        ## [602] Curiosity in Hindsight: Intrinsic Exploration in Stochastic Environments

        **Authors**: *Daniel Jarrett, Corentin Tallec, Florent Altché, Thomas Mesnard, Rémi Munos, Michal Valko*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/jarrett23a.html](https://proceedings.mlr.press/v202/jarrett23a.html)

        **Abstract**:

        Consider the problem of exploration in sparse-reward or reward-free environments, such as in Montezuma’s Revenge. In the curiosity-driven paradigm, the agent is rewarded for how much each realized outcome differs from their predicted outcome. But using predictive error as intrinsic motivation is fragile in stochastic environments, as the agent may become trapped by high-entropy areas of the state-action space, such as a "noisy TV". In this work, we study a natural solution derived from structural causal models of the world: Our key idea is to learn representations of the future that capture precisely the unpredictable aspects of each outcome—which we use as additional input for predictions, such that intrinsic rewards only reflect the predictable aspects of world dynamics. First, we propose incorporating such hindsight representations into models to disentangle "noise" from "novelty", yielding Curiosity in Hindsight: a simple and scalable generalization of curiosity that is robust to stochasticity. Second, we instantiate this framework for the recently introduced BYOL-Explore algorithm as our prime example, resulting in the noise-robust BYOL-Hindsight. Third, we illustrate its behavior under a variety of different stochasticities in a grid world, and find improvements over BYOL-Explore in hard-exploration Atari games with sticky actions. Notably, we show state-of-the-art results in exploring Montezuma’s Revenge with sticky actions, while preserving performance in the non-sticky setting.

        ----

        ## [603] BiRT: Bio-inspired Replay in Vision Transformers for Continual Learning

        **Authors**: *Kishaan Jeeveswaran, Prashant Shivaram Bhat, Bahram Zonooz, Elahe Arani*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/jeeveswaran23a.html](https://proceedings.mlr.press/v202/jeeveswaran23a.html)

        **Abstract**:

        The ability of deep neural networks to continually learn and adapt to a sequence of tasks has remained challenging due to catastrophic forgetting of previously learned tasks. Humans, on the other hand, have a remarkable ability to acquire, assimilate, and transfer knowledge across tasks throughout their lifetime without catastrophic forgetting. The versatility of the brain can be attributed to the rehearsal of abstract experiences through a complementary learning system. However, representation rehearsal in vision transformers lacks diversity, resulting in overfitting and consequently, performance drops significantly compared to raw image rehearsal. Therefore, we propose BiRT, a novel representation rehearsal-based continual learning approach using vision transformers. Specifically, we introduce controllable noises at various stages of the vision transformer and enforce consistency in predictions with respect to an exponential moving average of the working model. Our method provides consistent performance gain over raw image and vanilla representation rehearsal on several challenging CL benchmarks while being memory efficient and robust to natural and adversarial corruptions.

        ----

        ## [604] Recovering Top-Two Answers and Confusion Probability in Multi-Choice Crowdsourcing

        **Authors**: *Hyeonsu Jeong, Hye Won Chung*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/jeong23a.html](https://proceedings.mlr.press/v202/jeong23a.html)

        **Abstract**:

        Crowdsourcing has emerged as an effective platform for labeling large amounts of data in a cost- and time-efficient manner. Most previous work has focused on designing an efficient algorithm to recover only the ground-truth labels of the data. In this paper, we consider multi-choice crowdsourcing tasks with the goal of recovering not only the ground truth, but also the most confusing answer and the confusion probability. The most confusing answer provides useful information about the task by revealing the most plausible answer other than the ground truth and how plausible it is. To theoretically analyze such scenarios, we propose a model in which there are the top two plausible answers for each task, distinguished from the rest of the choices. Task difficulty is quantified by the probability of confusion between the top two, and worker reliability is quantified by the probability of giving an answer among the top two. Under this model, we propose a two-stage inference algorithm to infer both the top two answers and the confusion probability. We show that our algorithm achieves the minimax optimal convergence rate. We conduct both synthetic and real data experiments and demonstrate that our algorithm outperforms other recent algorithms. We also show the applicability of our algorithms in inferring the difficulty of tasks and in training neural networks with top-two soft labels.

        ----

        ## [605] Leveraging Label Non-Uniformity for Node Classification in Graph Neural Networks

        **Authors**: *Feng Ji, See Hian Lee, Hanyang Meng, Kai Zhao, Jielong Yang, Wee Peng Tay*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/ji23a.html](https://proceedings.mlr.press/v202/ji23a.html)

        **Abstract**:

        In node classification using graph neural networks (GNNs), a typical model generates logits for different class labels at each node. A softmax layer often outputs a label prediction based on the largest logit. We demonstrate that it is possible to infer hidden graph structural information from the dataset using these logits. We introduce the key notion of label non-uniformity, which is derived from the Wasserstein distance between the softmax distribution of the logits and the uniform distribution. We demonstrate that nodes with small label non-uniformity are harder to classify correctly. We theoretically analyze how the label non-uniformity varies across the graph, which provides insights into boosting the model performance: increasing training samples with high non-uniformity or dropping edges to reduce the maximal cut size of the node set of small non-uniformity. These mechanisms can be easily added to a base GNN model. Experimental results demonstrate that our approach improves the performance of many benchmark base models.

        ----

        ## [606] Bidirectional Adaptation for Robust Semi-Supervised Learning with Inconsistent Data Distributions

        **Authors**: *Lin-Han Jia, Lan-Zhe Guo, Zhi Zhou, Jie-Jing Shao, Yuke Xiang, Yu-Feng Li*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/jia23a.html](https://proceedings.mlr.press/v202/jia23a.html)

        **Abstract**:

        Semi-supervised learning (SSL) suffers from severe performance degradation when labeled and unlabeled data come from inconsistent data distributions. However, there is still a lack of sufficient theoretical guidance on how to alleviate this problem. In this paper, we propose a general theoretical framework that demonstrates how distribution discrepancies caused by pseudo-label predictions and target predictions can lead to severe generalization errors. Through theoretical analysis, we identify three main reasons why previous SSL algorithms cannot perform well with inconsistent distributions: coupling between the pseudo-label predictor and the target predictor, biased pseudo labels, and restricted sample weights. To address these challenges, we introduce a practical framework called Bidirectional Adaptation that can adapt to the distribution of unlabeled data for debiased pseudo-label prediction and to the target distribution for debiased target prediction, thereby mitigating these shortcomings. Extensive experimental results demonstrate the effectiveness of our proposed framework.

        ----

        ## [607] Short-lived High-volume Bandits

        **Authors**: *Su Jia, Nishant Oli, Ian Anderson, Paul Duff, Andrew A. Li, R. Ravi*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/jia23b.html](https://proceedings.mlr.press/v202/jia23b.html)

        **Abstract**:

        Modern platforms leverage randomized experiments to make informed decisions from a given set of alternatives. As a particularly challenging scenario, these alternatives can potentially have (i) high volume, with thousands of new items being released each hour, and (ii) short lifetime, either due to the contents’ transient nature, or some underlying non-stationarity that impels the learner to treat the same item as non-identical copies across time. We consider a multiplay bandits model. In each round a set of $k=n^\rho$ actions that will be available for $w$ rounds arrives, each of whose mean reward is drawn from a fixed known distribution. The learner selects a multiset of $n$ actions at a time. We propose an $\ell$-Layered Sieve Policy that recursively refines the action space for $\ell\leq w$ times. We show that for any given $\rho>0$, with suitable $\ell$, the policy achieves $\tilde O (n^{-\min \{\rho, \frac 12 (1+\frac 1w)^{-1}\}})$ regret. We also complement this result with an $\Omega (n^{-\min \{\rho, \frac 12\}})$ lower bound. We further validate the effectiveness of our Sieve Policy via numerical simulations and a field experiment in a large content card serving platform.

        ----

        ## [608] Smooth Non-stationary Bandits

        **Authors**: *Su Jia, Qian Xie, Nathan Kallus, Peter I. Frazier*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/jia23c.html](https://proceedings.mlr.press/v202/jia23c.html)

        **Abstract**:

        In many applications of online decision making, the environment is non-stationary and it is therefore crucial to use bandit algorithms that handle changes. Most existing approaches are designed to protect against non-smooth changes, constrained only by total variation or Lipschitzness over time, where they guarantee $T^{2/3}$ regret. However, in practice environments are often changing smoothly, so such algorithms may incur higher-than-necessary regret in these settings and do not leverage information on the rate of change. In this paper, we study a non-stationary two-arm bandit problem where we assume an arm’s mean reward is a $\beta$-Hölder function over (normalized) time, meaning it is $(\beta-1)$-times Lipschitz-continuously differentiable. We show the first separation between the smooth and non-smooth regimes by presenting a policy with $T^{3/5}$ regret for $\beta=2$. We complement this result by a $T^{\frac{\beta+1}{2\beta+1}}$ lower bound for any integer $\beta\ge 1$, which matches our upper bound for $\beta=2$.

        ----

        ## [609] A Unified Optimization Framework of ANN-SNN Conversion: Towards Optimal Mapping from Activation Values to Firing Rates

        **Authors**: *Haiyan Jiang, Srinivas Anumasa, Giulia De Masi, Huan Xiong, Bin Gu*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/jiang23a.html](https://proceedings.mlr.press/v202/jiang23a.html)

        **Abstract**:

        Spiking Neural Networks (SNNs) have gained significant attention for their energy-efficient and fast-inference capabilities, but training SNNs from scratch can be challenging due to the discrete nature of spikes. One alternative method is to convert an Artificial Neural Network (ANN) into an SNN, known as ANN-SNN conversion. Currently, existing ANN-SNN conversion methods often involve redesigning the ANN with a new activation function, rather than utilizing the traditional ReLU, and converting it to an SNN. However, these methods do not take into account the potential performance loss between the regular ANN with ReLU and the tailored ANN. In this work, we propose a unified optimization framework for ANN-SNN conversion that considers both performance loss and conversion error. To achieve this, we introduce the SlipReLU activation function, which is a weighted sum of the threshold-ReLU and the step function. Theoretical analysis demonstrates that conversion error can be zero on a range of shift values $\delta \in [-0.5,0.5]$ rather than a fixed shift term 0.5. We evaluate our SlipReLU method on CIFAR datasets, which shows that SlipReLU outperforms current ANN-SNN conversion methods and supervised training methods in terms of accuracy and latency. To the best of our knowledge, this is the first ANN-SNN conversion method that enables SNN inference using only 1 time step. Code is available at https://github.com/HaiyanJiang/SNN_Conversion_unified.

        ----

        ## [610] VIMA: Robot Manipulation with Multimodal Prompts

        **Authors**: *Yunfan Jiang, Agrim Gupta, Zichen Zhang, Guanzhi Wang, Yongqiang Dou, Yanjun Chen, Li Fei-Fei, Anima Anandkumar, Yuke Zhu, Linxi Fan*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/jiang23b.html](https://proceedings.mlr.press/v202/jiang23b.html)

        **Abstract**:

        Prompt-based learning has emerged as a successful paradigm in natural language processing, where a single general-purpose language model can be instructed to perform any task specified by input prompts. Yet task specification in robotics comes in various forms, such as imitating one-shot demonstrations, following language instructions, and reaching visual goals. They are often considered different tasks and tackled by specialized models. We show that a wide spectrum of robot manipulation tasks can be expressed with multimodal prompts, interleaving textual and visual tokens. Accordingly, we develop a new simulation benchmark that consists of thousands of procedurally-generated tabletop tasks with multimodal prompts, 600K+ expert trajectories for imitation learning, and a four-level evaluation protocol for systematic generalization. We design a transformer-based robot agent, VIMA, that processes these prompts and outputs motor actions autoregressively. VIMA features a recipe that achieves strong model scalability and data efficiency. It outperforms alternative designs in the hardest zero-shot generalization setting by up to $2.9\times$ task success rate given the same training data. With $10\times$ less training data, VIMA still performs $2.7\times$ better than the best competing variant. Code and video demos are available at https://vimalabs.github.io

        ----

        ## [611] Estimating Causal Effects using a Multi-task Deep Ensemble

        **Authors**: *Ziyang Jiang, Zhuoran Hou, Yiling Liu, Yiman Ren, Keyu Li, David E. Carlson*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/jiang23c.html](https://proceedings.mlr.press/v202/jiang23c.html)

        **Abstract**:

        A number of methods have been proposed for causal effect estimation, yet few have demonstrated efficacy in handling data with complex structures, such as images. To fill this gap, we propose Causal Multi-task Deep Ensemble (CMDE), a novel framework that learns both shared and group-specific information from the study population. We provide proofs demonstrating equivalency of CDME to a multi-task Gaussian process (GP) with a coregionalization kernel a priori. Compared to multi-task GP, CMDE efficiently handles high-dimensional and multi-modal covariates and provides pointwise uncertainty estimates of causal effects. We evaluate our method across various types of datasets and tasks and find that CMDE outperforms state-of-the-art methods on a majority of these tasks.

        ----

        ## [612] Online Restless Bandits with Unobserved States

        **Authors**: *Bowen Jiang, Bo Jiang, Jian Li, Tao Lin, Xinbing Wang, Chenghu Zhou*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/jiang23d.html](https://proceedings.mlr.press/v202/jiang23d.html)

        **Abstract**:

        We study the online restless bandit problem, where each arm evolves according to a Markov chain independently, and the reward of pulling an arm depends on both the current state of the corresponding Markov chain and the pulled arm. The agent (decision maker) does not know the transition functions and reward functions, and cannot observe the states of arms even after pulling. The goal is to sequentially choose which arms to pull so as to maximize the expected cumulative rewards collected. In this paper, we propose TSEETC, a learning algorithm based on Thompson Sampling with Episodic Explore-Then-Commit. The algorithm proceeds in episodes of increasing length and each episode is divided into exploration and exploitation phases. During the exploration phase, samples of action-reward pairs are collected in a round-robin fashion and utilized to update the posterior distribution as a mixture of Dirichlet distributions. At the beginning of the exploitation phase, TSEETC generates a sample from the posterior distribution as true parameters. It then follows the optimal policy for the sampled model for the rest of the episode. We establish the Bayesian regret bound $\tilde {\mathcal{O}}(\sqrt{T})$ for TSEETC, where $T$ is the time horizon. We show through simulations that TSEETC outperforms existing algorithms in regret.

        ----

        ## [613] Detecting Out-of-distribution Data through In-distribution Class Prior

        **Authors**: *Xue Jiang, Feng Liu, Zhen Fang, Hong Chen, Tongliang Liu, Feng Zheng, Bo Han*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/jiang23e.html](https://proceedings.mlr.press/v202/jiang23e.html)

        **Abstract**:

        Given a pre-trained in-distribution (ID) model, the inference-time out-of-distribution (OOD) detection aims to recognize OOD data during the inference stage. However, some representative methods share an unproven assumption that the probability that OOD data belong to every ID class should be the same, i.e., these OOD-to-ID probabilities actually form a uniform distribution. In this paper, we show that this assumption makes the above methods incapable when the ID model is trained with class-imbalanced data.Fortunately, by analyzing the causal relations between ID/OOD classes and features, we identify several common scenarios where the OOD-to-ID probabilities should be the ID-class-prior distribution and propose two strategies to modify existing inference-time detection methods: 1) replace the uniform distribution with the ID-class-prior distribution if they explicitly use the uniform distribution; 2) otherwise, reweight their scores according to the similarity between the ID-class-prior distribution and the softmax outputs of the pre-trained model. Extensive experiments show that both strategies can improve the OOD detection performance when the ID model is pre-trained with imbalanced data, reflecting the importance of ID-class prior in OOD detection.

        ----

        ## [614] Towards Stable and Efficient Adversarial Training against l1 Bounded Adversarial Attacks

        **Authors**: *Yulun Jiang, Chen Liu, Zhichao Huang, Mathieu Salzmann, Sabine Süsstrunk*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/jiang23f.html](https://proceedings.mlr.press/v202/jiang23f.html)

        **Abstract**:

        We address the problem of stably and efficiently training a deep neural network robust to adversarial perturbations bounded by an $l_1$ norm. We demonstrate that achieving robustness against $l_1$-bounded perturbations is more challenging than in the $l_2$ or $l_\infty$ cases, because adversarial training against $l_1$-bounded perturbations is more likely to suffer from catastrophic overfitting and yield training instabilities. Our analysis links these issues to the coordinate descent strategy used in existing methods. We address this by introducing Fast-EG-$l_1$, an efficient adversarial training algorithm based on Euclidean geometry and free of coordinate descent. Fast-EG-$l_1$ comes with no additional memory costs and no extra hyper-parameters to tune. Our experimental results on various datasets demonstrate that Fast-EG-$l_1$ yields the best and most stable robustness against $l_1$-bounded adversarial attacks among the methods of comparable computational complexity. Code and the checkpoints are available at https://github.com/IVRL/FastAdvL.

        ----

        ## [615] Learning Unnormalized Statistical Models via Compositional Optimization

        **Authors**: *Wei Jiang, Jiayu Qin, Lingyu Wu, Changyou Chen, Tianbao Yang, Lijun Zhang*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/jiang23g.html](https://proceedings.mlr.press/v202/jiang23g.html)

        **Abstract**:

        Learning unnormalized statistical models (e.g., energy-based models) is computationally challenging due to the complexity of handling the partition function. To eschew this complexity, noise-contrastive estimation (NCE) has been proposed by formulating the objective as the logistic loss of the real data and the artificial noise. However, as found in previous works, NCE may perform poorly in many tasks due to its flat loss landscape and slow convergence. In this paper, we study a direct approach for optimizing the negative log-likelihood of unnormalized models from the perspective of compositional optimization. To tackle the partition function, a noise distribution is introduced such that the log partition function can be written as a compositional function whose inner function can be estimated with stochastic samples. Hence, the objective can be optimized by stochastic compositional optimization algorithms. Despite being a simple method, we demonstrate that it is more favorable than NCE by (1) establishing a fast convergence rate and quantifying its dependence on the noise distribution through the variance of stochastic estimators; (2) developing better results for one-dimensional Gaussian mean estimation by showing our objective has a much favorable loss landscape and hence our method enjoys faster convergence; (3) demonstrating better performance on multiple applications, including density estimation, out-of-distribution detection, and real image generation.

        ----

        ## [616] Approximate Causal Effect Identification under Weak Confounding

        **Authors**: *Ziwei Jiang, Lai Wei, Murat Kocaoglu*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/jiang23h.html](https://proceedings.mlr.press/v202/jiang23h.html)

        **Abstract**:

        Causal effect estimation has been studied by many researchers when only observational data is available. Sound and complete algorithms have been developed for pointwise estimation of identifiable causal queries. For non-identifiable causal queries, researchers developed polynomial programs to estimate tight bounds on causal effect. However, these are computationally difficult to optimize for variables with large support sizes. In this paper, we analyze the effect of "weak confounding’" on causal estimands. More specifically, under the assumption that the unobserved confounders that render a query non-identifiable have small entropy, we propose an efficient linear program to derive the upper and lower bounds of the causal effect. We show that our bounds are consistent in the sense that as the entropy of unobserved confounders goes to zero, the gap between the upper and lower bound vanishes. Finally, we conduct synthetic and real data simulations to compare our bounds with the bounds obtained by the existing work that cannot incorporate such entropy constraints and show that our bounds are tighter for the setting with weak confounders.

        ----

        ## [617] MEWL: Few-shot multimodal word learning with referential uncertainty

        **Authors**: *Guangyuan Jiang, Manjie Xu, Shiji Xin, Wei Liang, Yujia Peng, Chi Zhang, Yixin Zhu*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/jiang23i.html](https://proceedings.mlr.press/v202/jiang23i.html)

        **Abstract**:

        Without explicit feedback, humans can rapidly learn the meaning of words. Children can acquire a new word after just a few passive exposures, a process known as fast mapping. This word learning capability is believed to be the most fundamental building block of multimodal understanding and reasoning. Despite recent advancements in multimodal learning, a systematic and rigorous evaluation is still missing for human-like word learning in machines. To fill in this gap, we introduce the MachinE Word Learning (MEWL) benchmark to assess how machines learn word meaning in grounded visual scenes. MEWL covers human’s core cognitive toolkits in word learning: cross-situational reasoning, bootstrapping, and pragmatic learning. Specifically, MEWL is a few-shot benchmark suite consisting of nine tasks for probing various word learning capabilities. These tasks are carefully designed to be aligned with the children’s core abilities in word learning and echo the theories in the developmental literature. By evaluating multimodal and unimodal agents’ performance with a comparative analysis of human performance, we notice a sharp divergence in human and machine word learning. We further discuss these differences between humans and machines and call for human-like few-shot word learning in machines.

        ----

        ## [618] NeuralSlice: Neural 3D Triangle Mesh Reconstruction via Slicing 4D Tetrahedral Meshes

        **Authors**: *Chenbo Jiang, Jie Yang, Shwai He, Yu-Kun Lai, Lin Gao*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/jiang23j.html](https://proceedings.mlr.press/v202/jiang23j.html)

        **Abstract**:

        Learning-based high-fidelity reconstruction of 3D shapes with varying topology is a fundamental problem in computer vision and computer graphics. Recent advances in learning 3D shapes using explicit and implicit representations have achieved impressive results in 3D modeling. However, the template-based explicit representation is limited by fixed topology, and the implicit representation, although flexible with arbitrary topology, requires a large number of sampled points to regress the surface, which is computationally expensive. In this work, we propose a novel 3D shape representation named NeuralSlice, which represents a 3D shape as the intersection of a 4D tetrahedral mesh and a 4D hyperplane. A novel network is designed to incorporate the proposed representation flexibly, which learns a deformable 4D template and a parameter for slicing 4D hyperplane to reconstruct the 3D object. To learn the local deformation of the 4D template, we further propose a spatial-aware network to locate the 4D points within the 3D feature volume of input shape via positional encoding, which leverages the local geometrical feature to guide the 4D deformation. By addressing the 3D problem in a higher 4D space, our method supports flexible topology changes while being highly efficient. Our method is guaranteed to produce manifold meshes. NeuralSlice outperforms the state-of-the-art explicit-based approaches in terms of reconstruction quality. Compared with implicit approaches, by avoiding point sampling, our method is 10 times faster than the implicit approaches, and better preserves thin structures. NeuralSlice has the capability of representing various shapes and topologies using a single 4D tetrahedral mesh. The corresponding code can be found on GitHub at https://github.com/IGLICT/NEURALSLICE

        ----

        ## [619] Effective Structured Prompting by Meta-Learning and Representative Verbalizer

        **Authors**: *Weisen Jiang, Yu Zhang, James T. Kwok*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/jiang23k.html](https://proceedings.mlr.press/v202/jiang23k.html)

        **Abstract**:

        Prompt tuning for pre-trained masked language models (MLM) has shown promising performance in natural language processing tasks with few labeled examples. It tunes a prompt for the downstream task, and a verbalizer is used to bridge the predicted token and label prediction. Due to the limited training data, prompt initialization is crucial for prompt tuning. Recently, MetaPrompting (Hou et al., 2022) uses meta-learning to learn a shared initialization for all task-specific prompts. However, a single initialization is insufficient to obtain good prompts for all tasks and samples when the tasks are complex. Moreover, MetaPrompting requires tuning the whole MLM, causing a heavy burden on computation and memory as the MLM is usually large. To address these issues, we use a prompt pool to extract more task knowledge and construct instance-dependent prompts via attention. We further propose a novel soft verbalizer (RepVerb) which constructs label embedding from feature embeddings directly. Combining meta-learning the prompt pool and RepVerb, we propose MetaPrompter for effective structured prompting. MetaPrompter is parameter-efficient as only the pool is required to be tuned. Experimental results demonstrate that MetaPrompter performs better than the recent state-of-the-arts and RepVerb outperforms existing soft verbalizers.

        ----

        ## [620] Understanding Incremental Learning of Gradient Descent: A Fine-grained Analysis of Matrix Sensing

        **Authors**: *Jikai Jin, Zhiyuan Li, Kaifeng Lyu, Simon Shaolei Du, Jason D. Lee*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/jin23a.html](https://proceedings.mlr.press/v202/jin23a.html)

        **Abstract**:

        It is believed that Gradient Descent (GD) induces an implicit bias towards good generalization in training machine learning models. This paper provides a fine-grained analysis of the dynamics of GD for the matrix sensing problem, whose goal is to recover a low-rank ground-truth matrix from near-isotropic linear measurements. It is shown that GD with small initialization behaves similarly to the greedy low-rank learning heuristics and follows an incremental learning procedure: GD sequentially learns solutions with increasing ranks until it recovers the ground truth matrix. Compared to existing works which only analyze the first learning phase for rank-1 solutions, our result provides characterizations for the whole learning process. Moreover, besides the over-parameterized regime that many prior works focused on, our analysis of the incremental learning procedure also applies to the under-parameterized regime. Finally, we conduct numerical experiments to confirm our theoretical findings.

        ----

        ## [621] Thompson Sampling with Less Exploration is Fast and Optimal

        **Authors**: *Tianyuan Jin, Xianglin Yang, Xiaokui Xiao, Pan Xu*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/jin23b.html](https://proceedings.mlr.press/v202/jin23b.html)

        **Abstract**:

        We propose $\epsilon$-Exploring Thompson Sampling ($\epsilon$-TS), a modified version of the Thompson Sampling (TS) algorithm for multi-armed bandits. In $\epsilon$-TS, arms are selected greedily based on empirical mean rewards with probability $1-\epsilon$, and based on posterior samples obtained from TS with probability $\epsilon$. Here, $\epsilon\in(0,1)$ is a user-defined constant. By reducing exploration, $\epsilon$-TS improves computational efficiency compared to TS while achieving better regret bounds. We establish that $\epsilon$-TS is both minimax optimal and asymptotically optimal for various popular reward distributions, including Gaussian, Bernoulli, Poisson, and Gamma. A key technical advancement in our analysis is the relaxation of the requirement for a stringent anti-concentration bound of the posterior distribution, which was necessary in recent analyses that achieved similar bounds. As a result, $\epsilon$-TS maintains the posterior update structure of TS while minimizing alterations, such as clipping the sampling distribution or solving the inverse of the Kullback-Leibler (KL) divergence between reward distributions, as done in previous work. Furthermore, our algorithm is as easy to implement as TS, but operates significantly faster due to reduced exploration. Empirical evaluations confirm the efficiency and optimality of $\epsilon$-TS.

        ----

        ## [622] R-U-SURE? Uncertainty-Aware Code Suggestions By Maximizing Utility Across Random User Intents

        **Authors**: *Daniel D. Johnson, Daniel Tarlow, Christian Walder*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/johnson23a.html](https://proceedings.mlr.press/v202/johnson23a.html)

        **Abstract**:

        Large language models show impressive results at predicting structured text such as code, but also commonly introduce errors and hallucinations in their output. When used to assist software developers, these models may make mistakes that users must go back and fix, or worse, introduce subtle bugs that users may miss entirely. We propose Randomized Utility-driven Synthesis of Uncertain REgions (R-U-SURE), an approach for building uncertainty-aware suggestions based on a decision-theoretic model of goal-conditioned utility, using random samples from a generative model as a proxy for the unobserved possible intents of the end user. Our technique combines minimum-Bayes-risk decoding, dual decomposition, and decision diagrams in order to efficiently produce structured uncertainty summaries, given only sample access to an arbitrary generative model of code and an optional AST parser. We demonstrate R-U-SURE on three developer-assistance tasks, and show that it can be applied different user interaction patterns without retraining the model and leads to more accurate uncertainty estimates than token-probability baselines. We also release our implementation as an open-source library at https://github.com/google-research/r_u_sure.

        ----

        ## [623] Automatically Auditing Large Language Models via Discrete Optimization

        **Authors**: *Erik Jones, Anca D. Dragan, Aditi Raghunathan, Jacob Steinhardt*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/jones23a.html](https://proceedings.mlr.press/v202/jones23a.html)

        **Abstract**:

        Auditing large language models for unexpected behaviors is critical to preempt catastrophic deployments, yet remains challenging. In this work, we cast auditing as an optimization problem, where we automatically search for input-output pairs that match a desired target behavior. For example, we might aim to find a non-toxic input that starts with “Barack Obama” that a model maps to a toxic output. This optimization problem is difficult to solve as the set of feasible points is sparse, the space is discrete, and the language models we audit are non-linear and high-dimensional. To combat these challenges, we introduce a discrete optimization algorithm, ARCA, that jointly and efficiently optimizes over inputs and outputs. Our approach automatically uncovers derogatory completions about celebrities (e.g. "Barack Obama is a legalized unborn" –$>$ "child murderer"), produces French inputs that complete to English outputs, and finds inputs that generate a specific name. Our work offers a promising new tool to uncover models’ failure-modes before deployment. Content Warning: This paper contains examples that may be offensive in nature.

        ----

        ## [624] On the Expressive Power of Geometric Graph Neural Networks

        **Authors**: *Chaitanya K. Joshi, Cristian Bodnar, Simon V. Mathis, Taco Cohen, Pietro Lio*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/joshi23a.html](https://proceedings.mlr.press/v202/joshi23a.html)

        **Abstract**:

        The expressive power of Graph Neural Networks (GNNs) has been studied extensively through the Weisfeiler-Leman (WL) graph isomorphism test. However, standard GNNs and the WL framework are inapplicable for geometric graphs embedded in Euclidean space, such as biomolecules, materials, and other physical systems. In this work, we propose a geometric version of the WL test (GWL) for discriminating geometric graphs while respecting the underlying physical symmetries: permutations, rotation, reflection, and translation. We use GWL to characterise the expressive power of geometric GNNs that are invariant or equivariant to physical symmetries in terms of distinguishing geometric graphs. GWL unpacks how key design choices influence geometric GNN expressivity: (1) Invariant layers have limited expressivity as they cannot distinguish one-hop identical geometric graphs; (2) Equivariant layers distinguish a larger class of graphs by propagating geometric information beyond local neighbourhoods; (3) Higher order tensors and scalarisation enable maximally powerful geometric GNNs; and (4) GWL’s discrimination-based perspective is equivalent to universal approximation. Synthetic experiments supplementing our results are available at https://github.com/chaitjo/geometric-gnn-dojo

        ----

        ## [625] Data-Efficient Contrastive Self-supervised Learning: Most Beneficial Examples for Supervised Learning Contribute the Least

        **Authors**: *Siddharth Joshi, Baharan Mirzasoleiman*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/joshi23b.html](https://proceedings.mlr.press/v202/joshi23b.html)

        **Abstract**:

        Self-supervised learning (SSL) learns high-quality representations from large pools of unlabeled training data. As datasets grow larger, it becomes crucial to identify the examples that contribute the most to learning such representations. This enables efficient SSL by reducing the volume of data required. Nevertheless, quantifying the value of examples for SSL has remained an open question. In this work, we address this problem for the first time, by proving that examples that contribute the most to contrastive SSL are those that have the most similar augmentations to other examples, in expectation. We provide rigorous guarantees for the generalization performance of contrastive learning on such subsets. Through extensive experiments, we show that we can safely exclude 20% of examples from CIFAR100 and 40% from STL10 and TinyImageNet, without affecting downstream task performance. In general, subsets selected by our method outperform random subsets by over 3% across these datasets. Interestingly, we also discover the subsets that contribute the most to contrastive learning are those that contribute the least to supervised learning.

        ----

        ## [626] Robust Subtask Learning for Compositional Generalization

        **Authors**: *Kishor Jothimurugan, Steve Hsu, Osbert Bastani, Rajeev Alur*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/jothimurugan23a.html](https://proceedings.mlr.press/v202/jothimurugan23a.html)

        **Abstract**:

        Compositional reinforcement learning is a promising approach for training policies to perform complex long-horizon tasks. Typically, a high-level task is decomposed into a sequence of subtasks and a separate policy is trained to perform each subtask. In this paper, we focus on the problem of training subtask policies in a way that they can be used to perform any task; here, a task is given by a sequence of subtasks. We aim to maximize the worst-case performance over all tasks as opposed to the average-case performance. We formulate the problem as a two agent zero-sum game in which the adversary picks the sequence of subtasks. We propose two RL algorithms to solve this game: one is an adaptation of existing multi-agent RL algorithms to our setting and the other is an asynchronous version which enables parallel training of subtask policies. We evaluate our approach on two multi-task environments with continuous states and actions and demonstrate that our algorithms outperform state-of-the-art baselines.

        ----

        ## [627] On Bridging the Gap between Mean Field and Finite Width Deep Random Multilayer Perceptron with Batch Normalization

        **Authors**: *Amir Joudaki, Hadi Daneshmand, Francis R. Bach*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/joudaki23a.html](https://proceedings.mlr.press/v202/joudaki23a.html)

        **Abstract**:

        Mean-field theory is widely used in theoretical studies of neural networks. In this paper, we analyze the role of depth in the concentration of mean-field predictions for Gram matrices of hidden representations in deep multilayer perceptron (MLP) with batch normalization (BN) at initialization. It is postulated that the mean-field predictions suffer from layer-wise errors that amplify with depth. We demonstrate that BN avoids this error amplification with depth. When the chain of hidden representations is rapidly mixing, we establish a concentration bound for a mean-field model of Gram matrices. To our knowledge, this is the first concentration bound that does not become vacuous with depth for standard MLPs with a finite width.

        ----

        ## [628] FARE: Provably Fair Representation Learning with Practical Certificates

        **Authors**: *Nikola Jovanovic, Mislav Balunovic, Dimitar Iliev Dimitrov, Martin T. Vechev*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/jovanovic23a.html](https://proceedings.mlr.press/v202/jovanovic23a.html)

        **Abstract**:

        Fair representation learning (FRL) is a popular class of methods aiming to produce fair classifiers via data preprocessing. Recent regulatory directives stress the need for FRL methods that provide practical certificates, i.e., provable upper bounds on the unfairness of any downstream classifier trained on preprocessed data, which directly provides assurance in a practical scenario. Creating such FRL methods is an important challenge that remains unsolved. In this work, we address that challenge and introduce FARE (Fairness with Restricted Encoders), the first FRL method with practical fairness certificates. FARE is based on our key insight that restricting the representation space of the encoder enables the derivation of practical guarantees, while still permitting favorable accuracy-fairness tradeoffs for suitable instantiations, such as one we propose based on fair trees. To produce a practical certificate, we develop and apply a statistical procedure that computes a finite sample high-confidence upper bound on the unfairness of any downstream classifier trained on FARE embeddings. In our comprehensive experimental evaluation, we demonstrate that FARE produces practical certificates that are tight and often even comparable with purely empirical results obtained by prior methods, which establishes the practical value of our approach.

        ----

        ## [629] Scaling of Class-wise Training Losses for Post-hoc Calibration

        **Authors**: *Seungjin Jung, Seungmo Seo, Yonghyun Jeong, Jongwon Choi*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/jung23a.html](https://proceedings.mlr.press/v202/jung23a.html)

        **Abstract**:

        The class-wise training losses often diverge as a result of the various levels of intra-class and inter-class appearance variation, and we find that the diverging class-wise training losses cause the uncalibrated prediction with its reliability. To resolve the issue, we propose a new calibration method to synchronize the class-wise training losses. We design a new training loss to alleviate the variance of class-wise training losses by using multiple class-wise scaling factors. Since our framework can compensate the training losses of overfitted classes with those of under-fitted classes, the integrated training loss is preserved, preventing the performance drop even after the model calibration. Furthermore, our method can be easily employed in the post-hoc calibration methods, allowing us to use the pre-trained model as an initial model and reduce the additional computation for model calibration. We validate the proposed framework by employing it in the various post-hoc calibration methods, which generally improves calibration performance while preserving accuracy, and discover through the investigation that our approach performs well with unbalanced datasets and untuned hyperparameters.

        ----

        ## [630] Fighting Fire with Fire: Contrastive Debiasing without Bias-free Data via Generative Bias-transformation

        **Authors**: *Yeonsung Jung, Hajin Shim, June Yong Yang, Eunho Yang*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/jung23b.html](https://proceedings.mlr.press/v202/jung23b.html)

        **Abstract**:

        Deep neural networks (DNNs), despite their ability to generalize with over-capacity networks, often rely heavily on the malignant bias as shortcuts instead of task-related information for discriminative tasks. This can lead to poor performance on real-world inputs, particularly when the majority of the sample is biased. To address the highly biased issue, recent studies either exploit auxiliary information which is rarely obtainable in practice or sift handful bias-free samples to emphasize them for debiasing. However, these methods are not always guaranteed to work due to unmet presumptions. In this paper, we propose Contrastive Debiasing via Generative Bias-transformation (CDvG) which is capable of operating without explicitly exploiting bias labels and bias-free samples. Motivated by our observation that not only discriminative models but also image translation models tend to focus on the malignant bias, CDvG employs an image translation model to transform the bias to another mode of bias while preserving task-relevant information. Through contrastive learning, the bias-transformed views are set against each other to learn bias-invariant representations. Our method shows a better debiasing effect when bias is more malignant as opposed to previous methods, and can also be integrated with the methods that focus on bias-free samples in a plug-and-play manner for further improvement. Experimental results on diverse datasets demonstrate that the proposed method outperforms the state-of-the-art, especially when bias-free samples are extremely scarce or absent.

        ----

        ## [631] Estimating Joint Treatment Effects by Combining Multiple Experiments

        **Authors**: *Yonghan Jung, Jin Tian, Elias Bareinboim*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/jung23c.html](https://proceedings.mlr.press/v202/jung23c.html)

        **Abstract**:

        Estimating the effects of multi-dimensional treatments (i.e., joint treatment effects) is critical in many data-intensive domains, including genetics and drug evaluation. The main challenges for studying the joint treatment effects include the need for large sample sizes to explore different treatment combinations as well as potentially unsafe treatment interactions. In this paper, we develop machinery for estimating joint treatment effects by combining data from multiple experimental datasets. In particular, first, we develop new identification conditions for determining whether a joint treatment effect can be computed in terms of multiple interventional distributions under various scenarios. Further, we develop estimators with statistically appealing properties, including consistency and robustness to model misspecification and slow convergence. Finally, we perform simulation studies, which corroborate the effectiveness of the proposed methods.

        ----

        ## [632] The Catalog Problem: Clustering and Ordering Variable-Sized Sets

        **Authors**: *Mateusz Maria Jurewicz, Graham W. Taylor, Leon Derczynski*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/jurewicz23a.html](https://proceedings.mlr.press/v202/jurewicz23a.html)

        **Abstract**:

        Prediction of a $\textbf{varying number}$ of $\textbf{ordered clusters}$ from sets of $\textbf{any cardinality}$ is a challenging task for neural networks, combining elements of set representation, clustering and learning to order. This task arises in many diverse areas, ranging from medical triage and early discharge, through machine part management and multi-channel signal analysis for petroleum exploration to product catalog structure prediction. This paper focuses on that last area, which exemplifies a number of challenges inherent to adaptive ordered clustering, referred to further as the eponymous $\textit{Catalog Problem}$. These include learning variable cluster constraints, exhibiting relational reasoning and managing combinatorial complexity. Despite progress in both neural clustering and set-to-sequence methods, no joint, fully differentiable model exists to-date. We develop such a modular architecture, referred to further as Neural Ordered Clusters (NOC), enhance it with a specific mechanism for learning cluster-level cardinality constraints, and provide a robust comparison of its performance in relation to alternative models. We test our method on three datasets, including synthetic catalog structures and PROCAT, a dataset of real-world catalogs consisting of over 1.5M products, achieving state-of-the-art results on a new, more challenging formulation of the underlying problem, which has not been addressed before. Additionally, we examine the network’s ability to learn higher-order interactions.

        ----

        ## [633] Equivariance with Learned Canonicalization Functions

        **Authors**: *Sékou-Oumar Kaba, Arnab Kumar Mondal, Yan Zhang, Yoshua Bengio, Siamak Ravanbakhsh*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/kaba23a.html](https://proceedings.mlr.press/v202/kaba23a.html)

        **Abstract**:

        Symmetry-based neural networks often constrain the architecture in order to achieve invariance or equivariance to a group of transformations. In this paper, we propose an alternative that avoids this architectural constraint by learning to produce canonical representations of the data. These canonicalization functions can readily be plugged into non-equivariant backbone architectures. We offer explicit ways to implement them for some groups of interest. We show that this approach enjoys universality while providing interpretable insights. Our main hypothesis, supported by our empirical results, is that learning a small neural network to perform canonicalization is better than using predefined heuristics. Our experiments show that learning the canonicalization function is competitive with existing techniques for learning equivariant functions across many tasks, including image classification, $N$-body dynamics prediction, point cloud classification and part segmentation, while being faster across the board.

        ----

        ## [634] Biases in Evaluation of Molecular Optimization Methods and Bias Reduction Strategies

        **Authors**: *Hiroshi Kajino, Kohei Miyaguchi, Takayuki Osogami*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/kajino23a.html](https://proceedings.mlr.press/v202/kajino23a.html)

        **Abstract**:

        We are interested in an evaluation methodology for molecular optimization. Given a sample of molecules and their properties of our interest, we wish not only to train a generator of molecules optimized with respect to a target property but also to evaluate its performance accurately. A common practice is to train a predictor of the target property using the sample and apply it to both training and evaluating the generator. However, little is known about its statistical properties, and thus, we are not certain about whether this performance estimate is reliable or not. We theoretically investigate this evaluation methodology and show that it potentially suffers from two biases; one is due to misspecification of the predictor and the other to reusing the same finite sample for training and evaluation. We discuss bias reduction methods for each of the biases, and empirically investigate their effectiveness.

        ----

        ## [635] Statistical Indistinguishability of Learning Algorithms

        **Authors**: *Alkis Kalavasis, Amin Karbasi, Shay Moran, Grigoris Velegkas*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/kalavasis23a.html](https://proceedings.mlr.press/v202/kalavasis23a.html)

        **Abstract**:

        When two different parties use the same learning rule on their own data, how can we test whether the distributions of the two outcomes are similar? In this paper, we study the similarity of outcomes of learning rules through the lens of the Total Variation (TV) distance of distributions. We say that a learning rule is TV indistinguishable if the expected TV distance between the posterior distributions of its outputs, executed on two training data sets drawn independently from the same distribution, is small. We first investigate the learnability of hypothesis classes using TV indistinguishable learners. Our main results are information-theoretic equivalences between TV indistinguishability and existing algorithmic stability notions such as replicability and approximate differential privacy. Then, we provide statistical amplification and boosting algorithms for TV indistinguishable learners.

        ----

        ## [636] Identifying Interpretable Subspaces in Image Representations

        **Authors**: *Neha Mukund Kalibhat, Shweta Bhardwaj, C. Bayan Bruss, Hamed Firooz, Maziar Sanjabi, Soheil Feizi*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/kalibhat23a.html](https://proceedings.mlr.press/v202/kalibhat23a.html)

        **Abstract**:

        We propose Automatic Feature Explanation using Contrasting Concepts (FALCON), an interpretability framework to explain features of image representations. For a target feature, FALCON captions its highly activating cropped images using a large captioning dataset (like LAION-400m) and a pre-trained vision-language model like CLIP. Each word among the captions is scored and ranked leading to a small number of shared, human-understandable concepts that closely describe the target feature. FALCON also applies contrastive interpretation using lowly activating (counterfactual) images, to eliminate spurious concepts. Although many existing approaches interpret features independently, we observe in state-of-the-art self-supervised and supervised models, that less than 20% of the representation space can be explained by individual features. We show that features in larger spaces become more interpretable when studied in groups and can be explained with high-order scoring concepts through FALCON. We discuss how extracted concepts can be used to explain and debug failures in downstream tasks. Finally, we present a technique to transfer concepts from one (explainable) representation space to another unseen representation space by learning a simple linear transformation.

        ----

        ## [637] Nonlinear Causal Discovery with Latent Confounders

        **Authors**: *David Kaltenpoth, Jilles Vreeken*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/kaltenpoth23a.html](https://proceedings.mlr.press/v202/kaltenpoth23a.html)

        **Abstract**:

        Causal discovery, the task of discovering the causal graph over a set of observed variables $X_1,\ldots,X_m$, is a challenging problem. One of the cornerstone assumptions is that of causal sufficiency: that all common causes of all measured variables have been observed. When it does not hold, causal discovery algorithms making this assumption return networks with many spurious edges. In this paper, we propose a nonlinear causal model involving hidden confounders. We show that it is identifiable from only the observed data and propose an efficient method for recovering this causal model. At the heart of our approach is a variational autoencoder which parametrizes both the causal interactions between observed variables as well as the influence of the unobserved confounders. Empirically we show that it outperforms other state-of-the-art methods for causal discovery under latent confounding on synthetic and real-world data.

        ----

        ## [638] Deep Generative Symbolic Regression with Monte-Carlo-Tree-Search

        **Authors**: *Pierre-Alexandre Kamienny, Guillaume Lample, Sylvain Lamprier, Marco Virgolin*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/kamienny23a.html](https://proceedings.mlr.press/v202/kamienny23a.html)

        **Abstract**:

        Symbolic regression (SR) is the problem of learning a symbolic expression from numerical data. Recently, deep neural models trained on procedurally-generated synthetic datasets showed competitive performance compared to more classical Genetic Programming (GP) ones. Unlike their GP counterparts, these neural approaches are trained to generate expressions from datasets given as context. This allows them to produce accurate expressions in a single forward pass at test time. However, they usually do not benefit from search abilities, which result in low performance compared to GP on out-of-distribution datasets. In this paper, we propose a novel method which provides the best of both worlds, based on a Monte-Carlo Tree Search procedure using a context-aware neural mutation model, which is initially pre-trained to learn promising mutations, and further refined from successful experiences in an online fashion. The approach demonstrates state-of-the-art performance on the well-known SRBench benchmark.

        ----

        ## [639] One-vs-the-Rest Loss to Focus on Important Samples in Adversarial Training

        **Authors**: *Sekitoshi Kanai, Shin'ya Yamaguchi, Masanori Yamada, Hiroshi Takahashi, Kentaro Ohno, Yasutoshi Ida*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/kanai23a.html](https://proceedings.mlr.press/v202/kanai23a.html)

        **Abstract**:

        This paper proposes a new loss function for adversarial training. Since adversarial training has difficulties, e.g., necessity of high model capacity, focusing on important data points by weighting cross-entropy loss has attracted much attention. However, they are vulnerable to sophisticated attacks, e.g., Auto-Attack. This paper experimentally reveals that the cause of their vulnerability is their small margins between logits for the true label and the other labels. Since neural networks classify the data points based on the logits, logit margins should be large enough to avoid flipping the largest logit by the attacks. Importance-aware methods do not increase logit margins of important samples but decrease those of less-important samples compared with cross-entropy loss. To increase logit margins of important samples, we propose switching one-vs-the-rest loss (SOVR), which switches from cross-entropy to one-vs-the-rest loss for important samples that have small logit margins. We prove that one-vs-the-rest loss increases logit margins two times larger than the weighted cross-entropy loss for a simple problem. We experimentally confirm that SOVR increases logit margins of important samples unlike existing methods and achieves better robustness against Auto-Attack than importance-aware methods.

        ----

        ## [640] Large Language Models Struggle to Learn Long-Tail Knowledge

        **Authors**: *Nikhil Kandpal, Haikang Deng, Adam Roberts, Eric Wallace, Colin Raffel*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/kandpal23a.html](https://proceedings.mlr.press/v202/kandpal23a.html)

        **Abstract**:

        The Internet contains a wealth of knowledge—from the birthdays of historical figures to tutorials on how to code—all of which may be learned by language models. However, while certain pieces of information are ubiquitous on the web, others appear extremely rarely. In this paper, we study the relationship between the knowledge memorized by large language models and the information in pre-training datasets scraped from the web. In particular, we show that a language model’s ability to answer a fact-based question relates to how many documents associated with that question were seen during pre-training. We identify these relevant documents by entity linking pre-training datasets and counting documents that contain the same entities as a given question-answer pair. Our results demonstrate strong correlational and causal relationships between accuracy and relevant document count for numerous question answering datasets (e.g., TriviaQA), pre-training corpora (e.g., ROOTS), and model sizes (e.g., 176B parameters). Moreover, while larger models are better at learning long-tail knowledge, we estimate that today’s models must be scaled by many orders of magnitude to reach competitive QA performance on questions with little support in the pre-training data. Finally, we show that retrieval-augmentation can reduce the dependence on relevant pre-training information, presenting a promising approach for capturing the long-tail.

        ----

        ## [641] Git-Theta: A Git Extension for Collaborative Development of Machine Learning Models

        **Authors**: *Nikhil Kandpal, Brian Lester, Mohammed Muqeeth, Anisha Mascarenhas, Monty Evans, Vishal Baskaran, Tenghao Huang, Haokun Liu, Colin Raffel*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/kandpal23b.html](https://proceedings.mlr.press/v202/kandpal23b.html)

        **Abstract**:

        Currently, most machine learning models are trained by centralized teams and are rarely updated. In contrast, open-source software development involves the iterative development of a shared artifact through distributed collaboration using a version control system. In the interest of enabling collaborative and continual improvement of machine learning models (Raffel, 2023), we introduce Git-Theta, a version control system for machine learning models. Git-Theta is an extension to Git, the most widely used version control software, that allows fine-grained tracking of changes to model parameters alongside code and other artifacts. Unlike existing version control systems that treat a model checkpoint as a blob of data, Git-Theta leverages the structure of checkpoints to support communication-efficient updates, automatic model merges, and meaningful reporting about the difference between two versions of a model. In addition, Git-Theta includes a plug-in system that enables users to easily add support for new functionality. In this paper, we introduce Git-Theta’s design and features and include an example use-case of Git-Theta where a pre-trained model is continually adapted and modified. We publicly release Git-Theta in hopes of kickstarting a new era of collaborative model development. https://github.com/r-three/git-theta/

        ----

        ## [642] A Deep Conjugate Direction Method for Iteratively Solving Linear Systems

        **Authors**: *Ayano Kaneda, Osman Akar, Jingyu Chen, Victoria Alicia Trevino Kala, David Hyde, Joseph Teran*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/kaneda23a.html](https://proceedings.mlr.press/v202/kaneda23a.html)

        **Abstract**:

        We present a novel deep learning approach to approximate the solution of large, sparse, symmetric, positive-definite linear systems of equations. Motivated by the conjugate gradients algorithm that iteratively selects search directions for minimizing the matrix norm of the approximation error, we design an approach that utilizes a deep neural network to accelerate convergence via data-driven improvement of the search direction at each iteration. Our method leverages a carefully chosen convolutional network to approximate the action of the inverse of the linear operator up to an arbitrary constant. We demonstrate the efficacy of our approach on spatially discretized Poisson equations, which arise in computational fluid dynamics applications, with millions of degrees of freedom. Unlike state-of-the-art learning approaches, our algorithm is capable of reducing the linear system residual to a given tolerance in a small number of iterations, independent of the problem size. Moreover, our method generalizes effectively to various systems beyond those encountered during training.

        ----

        ## [643] Leveraging Proxy of Training Data for Test-Time Adaptation

        **Authors**: *Juwon Kang, Nayeong Kim, Donghyeon Kwon, Jungseul Ok, Suha Kwak*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/kang23a.html](https://proceedings.mlr.press/v202/kang23a.html)

        **Abstract**:

        We consider test-time adaptation (TTA), the task of adapting a trained model to an arbitrary test domain using unlabeled input data on-the-fly during testing. A common practice of TTA is to disregard data used in training due to large memory demand and privacy leakage. However, the training data are the only source of supervision. This motivates us to investigate a proper way of using them while minimizing the side effects. To this end, we propose two lightweight yet informative proxies of the training data and a TTA method fully exploiting them. One of the proxies is composed of a small number of images synthesized (hence, less privacy-sensitive) by data condensation which minimizes their domain-specificity to capture a general underlying structure over a wide spectrum of domains. Then, in TTA, they are translated into labeled test data by stylizing them to match styles of unlabeled test samples. This enables virtually supervised test-time training. The other proxy is inter-class relations of training data, which are transferred to target model during TTA. On four public benchmarks, our method outperforms the state-of-the-art ones at remarkably less computation and memory.

        ----

        ## [644] Beyond Reward: Offline Preference-guided Policy Optimization

        **Authors**: *Yachen Kang, Diyuan Shi, Jinxin Liu, Li He, Donglin Wang*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/kang23b.html](https://proceedings.mlr.press/v202/kang23b.html)

        **Abstract**:

        This study focuses on the topic of offline preference-based reinforcement learning (PbRL), a variant of conventional reinforcement learning that dispenses with the need for online interaction or specification of reward functions. Instead, the agent is provided with fixed offline trajectories and human preferences between pairs of trajectories to extract the dynamics and task information, respectively. Since the dynamics and task information are orthogonal, a naive approach would involve using preference-based reward learning followed by an off-the-shelf offline RL algorithm. However, this requires the separate learning of a scalar reward function, which is assumed to be an information bottleneck of the learning process. To address this issue, we propose the offline preference-guided policy optimization (OPPO) paradigm, which models offline trajectories and preferences in a one-step process, eliminating the need for separately learning a reward function. OPPO achieves this by introducing an offline hindsight information matching objective for optimizing a contextual policy and a preference modeling objective for finding the optimal context. OPPO further integrates a well-performing decision policy by optimizing the two objectives iteratively. Our empirical results demonstrate that OPPO effectively models offline preferences and outperforms prior competing baselines, including offline RL algorithms performed over either true or pseudo reward function specifications. Our code is available on the project website: https://sites.google.com/view/oppo-icml-2023.

        ----

        ## [645] Poisoning Generative Replay in Continual Learning to Promote Forgetting

        **Authors**: *Siteng Kang, Zhan Shi, Xinhua Zhang*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/kang23c.html](https://proceedings.mlr.press/v202/kang23c.html)

        **Abstract**:

        Generative models have grown into the workhorse of many state-of-the-art machine learning methods. However, their vulnerability under poisoning attacks has been largely understudied. In this work, we investigate this issue in the context of continual learning, where generative replayers are utilized to tackle catastrophic forgetting. By developing a novel customization of dirty-label input-aware backdoors to the online setting, our attacker manages to stealthily promote forgetting while retaining high accuracy at the current task and sustaining strong defenders. Our approach taps into an intriguing property of generative models, namely that they cannot well capture input-dependent triggers. Experiments on four standard datasets corroborate the poisoner’s effectiveness.

        ----

        ## [646] Node Embedding from Neural Hamiltonian Orbits in Graph Neural Networks

        **Authors**: *Qiyu Kang, Kai Zhao, Yang Song, Sijie Wang, Wee Peng Tay*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/kang23d.html](https://proceedings.mlr.press/v202/kang23d.html)

        **Abstract**:

        In the graph node embedding problem, embedding spaces can vary significantly for different data types, leading to the need for different GNN model types. In this paper, we model the embedding update of a node feature as a Hamiltonian orbit over time. Since the Hamiltonian orbits generalize the exponential maps, this approach allows us to learn the underlying manifold of the graph in training, in contrast to most of the existing literature that assumes a fixed graph embedding manifold with a closed exponential map solution. Our proposed node embedding strategy can automatically learn, without extensive tuning, the underlying geometry of any given graph dataset even if it has diverse geometries. We test Hamiltonian functions of different forms and verify the performance of our approach on two graph node embedding downstream tasks: node classification and link prediction. Numerical experiments demonstrate that our approach adapts better to different types of graph datasets than popular state-of-the-art graph node embedding GNNs. The code is available at https://github.com/zknus/Hamiltonian-GNN.

        ----

        ## [647] Understanding Gradient Regularization in Deep Learning: Efficient Finite-Difference Computation and Implicit Bias

        **Authors**: *Ryo Karakida, Tomoumi Takase, Tomohiro Hayase, Kazuki Osawa*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/karakida23a.html](https://proceedings.mlr.press/v202/karakida23a.html)

        **Abstract**:

        Gradient regularization (GR) is a method that penalizes the gradient norm of the training loss during training. While some studies have reported that GR can improve generalization performance, little attention has been paid to it from the algorithmic perspective, that is, the algorithms of GR that efficiently improve the performance. In this study, we first reveal that a specific finite-difference computation, composed of both gradient ascent and descent steps, reduces the computational cost of GR. Next, we show that the finite-difference computation also works better in the sense of generalization performance. We theoretically analyze a solvable model, a diagonal linear network, and clarify that GR has a desirable implicit bias to so-called rich regime and finite-difference computation strengthens this bias. Furthermore, finite-difference GR is closely related to some other algorithms based on iterative ascent and descent steps for exploring flat minima. In particular, we reveal that the flooding method can perform finite-difference GR in an implicit way. Thus, this work broadens our understanding of GR for both practice and theory.

        ----

        ## [648] Langevin Thompson Sampling with Logarithmic Communication: Bandits and Reinforcement Learning

        **Authors**: *Amin Karbasi, Nikki Lijing Kuang, Yi-An Ma, Siddharth Mitra*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/karbasi23a.html](https://proceedings.mlr.press/v202/karbasi23a.html)

        **Abstract**:

        Thompson sampling (TS) is widely used in sequential decision making due to its ease of use and appealing empirical performance. However, many existing analytical and empirical results for TS rely on restrictive assumptions on reward distributions, such as belonging to conjugate families, which limits their applicability in realistic scenarios. Moreover, sequential decision making problems are often carried out in a batched manner, either due to the inherent nature of the problem or to serve the purpose of reducing communication and computation costs. In this work, we jointly study these problems in two popular settings, namely, stochastic multi-armed bandits (MABs) and infinite-horizon reinforcement learning (RL), where TS is used to learn the unknown reward distributions and transition dynamics, respectively. We propose batched Langevin Thompson Sampling algorithms that leverage MCMC methods to sample from approximate posteriors with only logarithmic communication costs in terms of batches. Our algorithms are computationally efficient and maintain the same order-optimal regret guarantees of $\mathcal{O}(\log T)$ for stochastic MABs, and $\mathcal{O}(\sqrt{T})$ for RL. We complement our theoretical findings with experimental results.

        ----

        ## [649] On the Relationship Between Explanation and Prediction: A Causal View

        **Authors**: *Amir-Hossein Karimi, Krikamol Muandet, Simon Kornblith, Bernhard Schölkopf, Been Kim*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/karimi23a.html](https://proceedings.mlr.press/v202/karimi23a.html)

        **Abstract**:

        Being able to provide explanations for a model’s decision has become a central requirement for the development, deployment, and adoption of machine learning models. However, we are yet to understand what explanation methods can and cannot do. How do upstream factors such as data, model prediction, hyperparameters, and random initialization influence downstream explanations? While previous work raised concerns that explanations (E) may have little relationship with the prediction (Y), there is a lack of conclusive study to quantify this relationship. Our work borrows tools from causal inference to systematically assay this relationship. More specifically, we study the relationship between E and Y by measuring the treatment effect when intervening on their causal ancestors, i.e., on hyperparameters and inputs used to generate saliency-based Es or Ys. Our results suggest that the relationships between E and Y is far from ideal. In fact, the gap between ’ideal’ case only increase in higher-performing models — models that are likely to be deployed. Our work is a promising first step towards providing a quantitative measure of the relationship between E and Y, which could also inform the future development of methods for E with a quantitative metric.

        ----

        ## [650] Cocktail Party Attack: Breaking Aggregation-Based Privacy in Federated Learning Using Independent Component Analysis

        **Authors**: *Sanjay Kariyappa, Chuan Guo, Kiwan Maeng, Wenjie Xiong, G. Edward Suh, Moinuddin K. Qureshi, Hsien-Hsin S. Lee*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/kariyappa23a.html](https://proceedings.mlr.press/v202/kariyappa23a.html)

        **Abstract**:

        Federated learning (FL) aims to perform privacy-preserving machine learning on distributed data held by multiple data owners. To this end, FL requires the data owners to perform training locally and share the gradients or weight updates (instead of the private inputs) with the central server, which are then securely aggregated over multiple data owners. Although aggregation by itself does not offer provable privacy protection, prior work suggested that if the batch size is sufficiently large the aggregation may be secure enough. In this paper, we propose the Cocktail Party Attack (CPA) that, contrary to prior belief, is able to recover the private inputs from gradients/weight updates aggregated over as many as 1024 samples. CPA leverages the crucial insight that aggregate gradients from a fully connected (FC) layer is a linear combination of its inputs, which allows us to frame gradient inversion as a blind source separation (BSS) problem. We adapt independent component analysis (ICA)—a classic solution to the BSS problem—to recover private inputs for FC and convolutional networks, and show that CPA significantly outperforms prior gradient inversion attacks, scales to ImageNet-sized inputs, and works on large batch sizes of up to 1024.

        ----

        ## [651] General Sequential Episodic Memory Model

        **Authors**: *Arjun Karuvally, Terrence J. Sejnowski, Hava T. Siegelmann*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/karuvally23a.html](https://proceedings.mlr.press/v202/karuvally23a.html)

        **Abstract**:

        The state-of-the-art memory model is the General Associative Memory Model, a generalization of the classical Hopfield network. Like its ancestor, the general associative memory has a well-defined state-dependant energy surface, and its memories correlate with its fixed points. This is unlike human memories, which are commonly sequential rather than separated fixed points. In this paper, we introduce a class of General Sequential Episodic Memory Models (GSEMM) that, in the adiabatic limit, exhibit a dynamic energy surface, leading to a series of meta-stable states capable of encoding memory sequences. A multiple-timescale architecture enables the dynamic nature of the energy surface with newly introduced asymmetric synapses and signal propagation delays. We demonstrate its dense capacity under polynomial activation functions. GSEMM combines separate memories, short and long sequential episodic memories, under a unified theoretical framework, demonstrating how energy-based memory modeling can provide richer, human-like episodes.

        ----

        ## [652] Regression with Sensor Data Containing Incomplete Observations

        **Authors**: *Takayuki Katsuki, Takayuki Osogami*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/katsuki23a.html](https://proceedings.mlr.press/v202/katsuki23a.html)

        **Abstract**:

        This paper addresses a regression problem in which output label values are the results of sensing the magnitude of a phenomenon. A low value of such labels can mean either that the actual magnitude of the phenomenon was low or that the sensor made an incomplete observation. This leads to a bias toward lower values in labels and the resultant learning because labels may have lower values due to incomplete observations, even if the actual magnitude of the phenomenon was high. Moreover, because an incomplete observation does not provide any tags indicating incompleteness, we cannot eliminate or impute them. To address this issue, we propose a learning algorithm that explicitly models incomplete observations corrupted with an asymmetric noise that always has a negative value. We show that our algorithm is unbiased as if it were learned from uncorrupted data that does not involve incomplete observations. We demonstrate the advantages of our algorithm through numerical experiments.

        ----

        ## [653] Data Representations' Study of Latent Image Manifolds

        **Authors**: *Ilya Kaufman, Omri Azencot*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/kaufman23a.html](https://proceedings.mlr.press/v202/kaufman23a.html)

        **Abstract**:

        Deep neural networks have been demonstrated to achieve phenomenal success in many domains, and yet their inner mechanisms are not well understood. In this paper, we investigate the curvature of image manifolds, i.e., the manifold deviation from being flat in its principal directions. We find that state-of-the-art trained convolutional neural networks for image classification have a characteristic curvature profile along layers: an initial steep increase, followed by a long phase of a plateau, and followed by another increase. In contrast, this behavior does not appear in untrained networks in which the curvature flattens. We also show that the curvature gap between the last two layers has a strong correlation with the generalization capability of the network. Moreover, we find that the intrinsic dimension of latent codes is not necessarily indicative of curvature. Finally, we observe that common regularization methods such as mixup yield flatter representations when compared to other methods. Our experiments show consistent results over a variety of deep learning architectures and multiple data sets.

        ----

        ## [654] Multi-Modal Classifiers for Open-Vocabulary Object Detection

        **Authors**: *Prannay Kaul, Weidi Xie, Andrew Zisserman*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/kaul23a.html](https://proceedings.mlr.press/v202/kaul23a.html)

        **Abstract**:

        The goal of this paper is open-vocabulary object detection (OVOD) — building a model that can detect objects beyond the set of categories seen at training, thus enabling the user to specify categories of interest at inference without the need for model retraining. We adopt a standard two- stage object detector architecture, and explore three ways for specifying novel categories: via language descriptions, via image exemplars, or via a combination of the two. We make three contributions: first, we prompt a large language model (LLM) to generate informative language descriptions for object classes, and construct powerful text-based classifiers; second, we employ a visual aggregator on image exemplars that can ingest any number of images as input, forming vision-based classifiers; and third, we provide a simple method to fuse information from language descriptions and image exemplars, yield- ing a multi-modal classifier. When evaluating on the challenging LVIS open-vocabulary bench- mark we demonstrate that: (i) our text-based classifiers outperform all previous OVOD works; (ii) our vision-based classifiers perform as well as text-based classifiers in prior work; (iii) using multi-modal classifiers perform better than either modality alone; and finally, (iv) our text-based and multi-modal classifiers yield better performance than a fully-supervised detector.

        ----

        ## [655] Learning Mixtures of Markov Chains and MDPs

        **Authors**: *Chinmaya Kausik, Kevin Tan, Ambuj Tewari*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/kausik23a.html](https://proceedings.mlr.press/v202/kausik23a.html)

        **Abstract**:

        We present an algorithm for learning mixtures of Markov chains and Markov decision processes (MDPs) from short unlabeled trajectories. Specifically, our method handles mixtures of Markov chains with optional control input by going through a multi-step process, involving (1) a subspace estimation step, (2) spectral clustering of trajectories using "pairwise distance estimators," along with refinement using the EM algorithm, (3) a model estimation step, and (4) a classification step for predicting labels of new trajectories. We provide end-to-end performance guarantees, where we only explicitly require the length of trajectories to be linear in the number of states and the number of trajectories to be linear in a mixing time parameter. Experimental results support these guarantees, where we attain 96.6% average accuracy on a mixture of two MDPs in gridworld, outperforming the EM algorithm with random initialization (73.2% average accuracy). We also significantly outperform the EM algorithm on real data from the LastFM song dataset.

        ----

        ## [656] Curious Replay for Model-based Adaptation

        **Authors**: *Isaac Kauvar, Chris Doyle, Linqi Zhou, Nick Haber*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/kauvar23a.html](https://proceedings.mlr.press/v202/kauvar23a.html)

        **Abstract**:

        Agents must be able to adapt quickly as an environment changes. We find that existing model-based reinforcement learning agents are unable to do this well, in part because of how they use past experiences to train their world model. Here, we present Curious Replay—a form of prioritized experience replay tailored to model-based agents through use of a curiosity-based priority signal. Agents using Curious Replay exhibit improved performance in an exploration paradigm inspired by animal behavior and on the Crafter benchmark. DreamerV3 with Curious Replay surpasses state-of-the-art performance on Crafter, achieving a mean score of 19.4 that substantially improves on the previous high score of 14.5 by DreamerV3 with uniform replay, while also maintaining similar performance on the Deepmind Control Suite. Code for Curious Replay is available at github.com/AutonomousAgentsLab/curiousreplay.

        ----

        ## [657] How Does Information Bottleneck Help Deep Learning?

        **Authors**: *Kenji Kawaguchi, Zhun Deng, Xu Ji, Jiaoyang Huang*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/kawaguchi23a.html](https://proceedings.mlr.press/v202/kawaguchi23a.html)

        **Abstract**:

        Numerous deep learning algorithms have been inspired by and understood via the notion of information bottleneck, where unnecessary information is (often implicitly) minimized while task-relevant information is maximized. However, a rigorous argument for justifying why it is desirable to control information bottlenecks has been elusive. In this paper, we provide the first rigorous learning theory for justifying the benefit of information bottleneck in deep learning by mathematically relating information bottleneck to generalization errors. Our theory proves that controlling information bottleneck is one way to control generalization errors in deep learning, although it is not the only or necessary way. We investigate the merit of our new mathematical findings with experiments across a range of architectures and learning settings. In many cases, generalization errors are shown to correlate with the degree of information bottleneck: i.e., the amount of the unnecessary information at hidden layers. This paper provides a theoretical foundation for current and future methods through the lens of information bottleneck. Our new generalization bounds scale with the degree of information bottleneck, unlike the previous bounds that scale with the number of parameters, VC dimension, Rademacher complexity, stability or robustness. Our code is publicly available at: https://github.com/xu-ji/information-bottleneck

        ----

        ## [658] Instrumental Variable Estimation of Average Partial Causal Effects

        **Authors**: *Yuta Kawakami, Manabu Kuroki, Jin Tian*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/kawakami23a.html](https://proceedings.mlr.press/v202/kawakami23a.html)

        **Abstract**:

        Instrumental variable (IV) analysis is a powerful tool widely used to elucidate causal relationships. We study the problem of estimating the average partial causal effect (APCE) of a continuous treatment in an IV setting. Specifically, we develop new methods for estimating APCE based on a recent identification condition via an integral equation. We develop two families of methods, nonparametric and parametric - the former uses the Picard iteration to solve the integral equation; the latter parameterizes APCE using a linear basis function model. We analyze the statistical and computational properties of the proposed methods and illustrate them on synthetic and real data.

        ----

        ## [659] The Test of Tests: A Framework for Differentially Private Hypothesis Testing

        **Authors**: *Zeki Kazan, Kaiyan Shi, Adam Groce, Andrew P. Bray*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/kazan23a.html](https://proceedings.mlr.press/v202/kazan23a.html)

        **Abstract**:

        We present a generic framework for creating differentially private versions of any hypothesis test in a black-box way. We analyze the resulting tests analytically and experimentally. Most crucially, we show good practical performance for small data sets, showing that at ε = 1 we only need 5-6 times as much data as in the fully public setting. We compare our work to the one existing framework of this type, as well as to several individually-designed private hypothesis tests. Our framework is higher power than other generic solutions and at least competitive with (and often better than) individually-designed tests.

        ----

        ## [660] Exact Inference in High-order Structured Prediction

        **Authors**: *Chuyang Ke, Jean Honorio*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/ke23a.html](https://proceedings.mlr.press/v202/ke23a.html)

        **Abstract**:

        In this paper, we study the problem of inference in high-order structured prediction tasks. In the context of Markov random fields, the goal of a high-order inference task is to maximize a score function on the space of labels, and the score function can be decomposed into sum of unary and high-order potentials. We apply a generative model approach to study the problem of high-order inference, and provide a two-stage convex optimization algorithm for exact label recovery. We also provide a new class of hypergraph structural properties related to hyperedge expansion that drives the success in general high-order inference problems. Finally, we connect the performance of our algorithm and the hyperedge expansion property using a novel hypergraph Cheeger-type inequality.

        ----

        ## [661] Neural Wave Machines: Learning Spatiotemporally Structured Representations with Locally Coupled Oscillatory Recurrent Neural Networks

        **Authors**: *T. Anderson Keller, Max Welling*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/keller23a.html](https://proceedings.mlr.press/v202/keller23a.html)

        **Abstract**:

        Traveling waves have been measured at a diversity of regions and scales in the brain, however a consensus as to their computational purpose has yet to be reached. An intriguing hypothesis is that traveling waves serve to structure neural representations both in space and time, thereby acting as an inductive bias towards natural data. In this work, we investigate this hypothesis by introducing the Neural Wave Machine (NWM) – a locally coupled oscillatory recurrent neural network capable of exhibiting traveling waves in its hidden state. After training on simple dynamic sequences, we show that this model indeed learns static spatial structure such as topographic organization, and further uses complex spatiotemporal structure such as traveling waves to encode observed transformations. To measure the computational implications of this structure, we use a suite of sequence classification and physical dynamics modeling tasks to show that the NWM is both more parameter efficient, and is able to forecast future trajectories of simple physical dynamical systems more accurately than existing state of the art counterparts.

        ----

        ## [662] Homomorphism AutoEncoder - Learning Group Structured Representations from Observed Transitions

        **Authors**: *Hamza Keurti, Hsiao-Ru Pan, Michel Besserve, Benjamin F. Grewe, Bernhard Schölkopf*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/keurti23a.html](https://proceedings.mlr.press/v202/keurti23a.html)

        **Abstract**:

        How can agents learn internal models that veridically represent interactions with the real world is a largely open question. As machine learning is moving towards representations containing not just observational but also interventional knowledge, we study this problem using tools from representation learning and group theory. We propose methods enabling an agent acting upon the world to learn internal representations of sensory information that are consistent with actions that modify it. We use an autoencoder equipped with a group representation acting on its latent space, trained using an equivariance-derived loss in order to enforce a suitable homomorphism property on the group representation. In contrast to existing work, our approach does not require prior knowledge of the group and does not restrict the set of actions the agent can perform. We motivate our method theoretically, and show empirically that it can learn a group representation of the actions, thereby capturing the structure of the set of transformations applied to the environment. We further show that this allows agents to predict the effect of sequences of future actions with improved accuracy.

        ----

        ## [663] Rethinking Backdoor Attacks

        **Authors**: *Alaa Khaddaj, Guillaume Leclerc, Aleksandar Makelov, Kristian Georgiev, Hadi Salman, Andrew Ilyas, Aleksander Madry*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/khaddaj23a.html](https://proceedings.mlr.press/v202/khaddaj23a.html)

        **Abstract**:

        In a backdoor attack, an adversary inserts maliciously constructed backdoor examples into a training set to make the resulting model vulnerable to manipulation. Defending against such attacks involves viewing inserted examples as outliers in the training set and using techniques from robust statistics to detect and remove them. In this work, we present a different approach to the backdoor attack problem. Specifically, we show that without structural information about the training data distribution, backdoor attacks are indistinguishable from naturally-occuring features in the data—and thus impossible to "detect" in a general sense. Then, guided by this observation, we revisit existing defenses against backdoor attacks and characterize the (often latent) assumptions they make, and on which they depend. Finally, we explore an alternative perspective on backdoor attacks: one that assumes these attacks correspond to the strongest feature in the training data. Under this assumption (which we make formal) we develop a new primitive for detecting backdoor attacks. Our primitive naturally gives rise to a detection algorithm that comes with theoretical guarantees, and is effective in practice.

        ----

        ## [664] PAC Prediction Sets for Large Language Models of Code

        **Authors**: *Adam Khakhar, Stephen Mell, Osbert Bastani*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/khakhar23a.html](https://proceedings.mlr.press/v202/khakhar23a.html)

        **Abstract**:

        Prediction sets have recently been shown to be a promising strategy for quantifying the uncertainty of deep neural networks in a way that provides theoretical guarantees. However, existing techniques have largely targeted settings where the space of labels is simple, so prediction sets can be arbitrary subsets of labels. For structured prediction problems where the space of labels is exponential in size, even prediction sets containing a small fraction of all labels can be exponentially large. In the context of code generation, we propose a solution that considers a restricted set of prediction sets that can compactly be represented as partial programs, which are programs with portions replaced with holes. Given a trained code generation model, our algorithm leverages a programming language’s abstract syntax tree to generate a set of programs such that the correct program is in the set with high-confidence. Valuable applications of our algorithm include a Codex-style code generator with holes in uncertain parts of the generated code, which provides a partial program with theoretical guarantees. We evaluate our approach on PICARD (a T5 model for SQL semantic parsing) and Codex (a GPT model for over a dozen programming languages, including Python), demonstrating that our approach generates compact PAC prediction sets. This is the first research contribution that generates PAC prediction sets for generative code models.

        ----

        ## [665] Accelerated Primal-Dual Methods for Convex-Strongly-Concave Saddle Point Problems

        **Authors**: *Mohammad Khalafi, Digvijay Boob*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/khalafi23a.html](https://proceedings.mlr.press/v202/khalafi23a.html)

        **Abstract**:

        We investigate a primal-dual (PD) method for the saddle point problem (SPP) that uses a linear approximation of the primal function instead of the standard proximal step, resulting in a linearized PD (LPD) method. For convex-strongly concave SPP, we observe that the LPD method has a suboptimal dependence on the Lipschitz constant of the primal function. To fix this issue, we combine features of Accelerated Gradient Descent with the LPD method resulting in a single-loop Accelerated Linearized Primal-Dual (ALPD) method. ALPD method achieves the optimal gradient complexity when the SPP has a semi-linear coupling function. We also present an inexact ALPD method for SPPs with a general nonlinear coupling function that maintains the optimal gradient evaluations of the primal parts and significantly improves the gradient evaluations of the coupling term compared to the ALPD method. We verify our findings with numerical experiments.

        ----

        ## [666] Loss Balancing for Fair Supervised Learning

        **Authors**: *Mohammad Mahdi Khalili, Xueru Zhang, Mahed Abroshan*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/khalili23a.html](https://proceedings.mlr.press/v202/khalili23a.html)

        **Abstract**:

        Supervised learning models have been used in various domains such as lending, college admission, face recognition, natural language processing, etc. However, they may inherit pre-existing biases from training data and exhibit discrimination against protected social groups. Various fairness notions have been proposed to address unfairness issues. In this work, we focus on Equalized Loss (EL), a fairness notion that requires the expected loss to be (approximately) equalized across different groups. Imposing EL on the learning process leads to a non-convex optimization problem even if the loss function is convex, and the existing fair learning algorithms cannot properly be adopted to find the fair predictor under the EL constraint. This paper introduces an algorithm that can leverage off-the-shelf convex programming tools (e.g., CVXPY (Diamond and Boyd, 2016; Agrawal et al., 2018)) to efficiently find the global optimum of this non-convex optimization. In particular, we propose the ELminimizer algorithm, which finds the optimal fair predictor under EL by reducing the non-convex optimization to a sequence of convex optimization problems. We theoretically prove that our algorithm finds the global optimal solution under certain conditions. Then, we support our theoretical results through several empirical studies

        ----

        ## [667] Linearly Constrained Bilevel Optimization: A Smoothed Implicit Gradient Approach

        **Authors**: *Prashant Khanduri, Ioannis C. Tsaknakis, Yihua Zhang, Jia Liu, Sijia Liu, Jiawei Zhang, Mingyi Hong*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/khanduri23a.html](https://proceedings.mlr.press/v202/khanduri23a.html)

        **Abstract**:

        This work develops analysis and algorithms for solving a class of bilevel optimization problems where the lower-level (LL) problems have linear constraints. Most of the existing approaches for constrained bilevel problems rely on value function-based approximate reformulations, which suffer from issues such as non-convex and non-differentiable constraints. In contrast, in this work, we develop an implicit gradient-based approach, which is easy to implement, and is suitable for machine learning applications. We first provide an in-depth understanding of the problem, by showing that the implicit objective for such problems is in general non-differentiable. However, if we add some small (linear) perturbation to the LL objective, the resulting implicit objective becomes differentiable almost surely. This key observation opens the door for developing (deterministic and stochastic) gradient-based algorithms similar to the state-of-the-art ones for unconstrained bi-level problems. We show that when the implicit function is assumed to be strongly-convex, convex, and weakly-convex, the resulting algorithms converge with guaranteed rate. Finally, we experimentally corroborate the theoretical findings and evaluate the performance of the proposed framework on numerical and adversarial learning problems.

        ----

        ## [668] Emergent Asymmetry of Precision and Recall for Measuring Fidelity and Diversity of Generative Models in High Dimensions

        **Authors**: *Mahyar Khayatkhoei, Wael Abd-Almageed*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/khayatkhoei23a.html](https://proceedings.mlr.press/v202/khayatkhoei23a.html)

        **Abstract**:

        Precision and Recall are two prominent metrics of generative performance, which were proposed to separately measure the fidelity and diversity of generative models. Given their central role in comparing and improving generative models, understanding their limitations are crucially important. To that end, in this work, we identify a critical flaw in the common approximation of these metrics using k-nearest-neighbors, namely, that the very interpretations of fidelity and diversity that are assigned to Precision and Recall can fail in high dimensions, resulting in very misleading conclusions. Specifically, we empirically and theoretically show that as the number of dimensions grows, two model distributions with supports at equal point-wise distance from the support of the real distribution, can have vastly different Precision and Recall regardless of their respective distributions, hence an emergent asymmetry in high dimensions. Based on our theoretical insights, we then provide simple yet effective modifications to these metrics to construct symmetric metrics regardless of the number of dimensions. Finally, we provide experiments on real-world datasets to illustrate that the identified flaw is not merely a pathological case, and that our proposed metrics are effective in alleviating its impact.

        ----

        ## [669] Learning-augmented private algorithms for multiple quantile release

        **Authors**: *Mikhail Khodak, Kareem Amin, Travis Dick, Sergei Vassilvitskii*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/khodak23a.html](https://proceedings.mlr.press/v202/khodak23a.html)

        **Abstract**:

        When applying differential privacy to sensitive data, we can often improve performance using external information such as other sensitive data, public data, or human priors. We propose to use the learning-augmented algorithms (or algorithms with predictions) framework—previously applied largely to improve time complexity or competitive ratios—as a powerful way of designing and analyzing privacy-preserving methods that can take advantage of such external information to improve utility. This idea is instantiated on the important task of multiple quantile release, for which we derive error guarantees that scale with a natural measure of prediction quality while (almost) recovering state-of-the-art prediction-independent guarantees. Our analysis enjoys several advantages, including minimal assumptions about the data, a natural way of adding robustness, and the provision of useful surrogate losses for two novel ”meta” algorithms that learn predictions from other (potentially sensitive) data. We conclude with experiments on challenging tasks demonstrating that learning predictions across one or more instances can lead to large error reductions while preserving privacy.

        ----

        ## [670] CrossSplit: Mitigating Label Noise Memorization through Data Splitting

        **Authors**: *Jihye Kim, Aristide Baratin, Yan Zhang, Simon Lacoste-Julien*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/kim23a.html](https://proceedings.mlr.press/v202/kim23a.html)

        **Abstract**:

        We approach the problem of improving robustness of deep learning algorithms in the presence of label noise. Building upon existing label correction and co-teaching methods, we propose a novel training procedure to mitigate the memorization of noisy labels, called CrossSplit, which uses a pair of neural networks trained on two disjoint parts of the labeled dataset. CrossSplit combines two main ingredients: (i) Cross-split label correction. The idea is that, since the model trained on one part of the data cannot memorize example-label pairs from the other part, the training labels presented to each network can be smoothly adjusted by using the predictions of its peer network; (ii) Cross-split semi-supervised training. A network trained on one part of the data also uses the unlabeled inputs of the other part. Extensive experiments on CIFAR-10, CIFAR-100, Tiny-ImageNet and mini-WebVision datasets demonstrate that our method can outperform the current state-of-the-art in a wide range of noise ratios. The project page is at https://rlawlgul.github.io/.

        ----

        ## [671] Trainability, Expressivity and Interpretability in Gated Neural ODEs

        **Authors**: *Timothy Doyeon Kim, Tankut Can, Kamesh Krishnamurthy*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/kim23b.html](https://proceedings.mlr.press/v202/kim23b.html)

        **Abstract**:

        Understanding how the dynamics in biological and artificial neural networks implement the computations required for a task is a salient open question in machine learning and neuroscience. In particular, computations requiring complex memory storage and retrieval pose a significant challenge for these networks to implement or learn. Recently, a family of models described by neural ordinary differential equations (nODEs) has emerged as powerful dynamical neural network models capable of capturing complex dynamics. Here, we extend nODEs by endowing them with adaptive timescales using gating interactions. We refer to these as gated neural ODEs (gnODEs). Using a task that requires memory of continuous quantities, we demonstrate the inductive bias of the gnODEs to learn (approximate) continuous attractors. We further show how reduced-dimensional gnODEs retain their modeling power while greatly improving interpretability, even allowing explicit visualization of the structure of learned attractors. We introduce a novel measure of expressivity which probes the capacity of a neural network to generate complex trajectories. Using this measure, we explore how the phase-space dimension of the nODEs and the complexity of the function modeling the flow field contribute to expressivity. We see that a more complex function for modeling the flow field allows a lower-dimensional nODE to capture a given target dynamics. Finally, we demonstrate the benefit of gating in nODEs on several real-world tasks.

        ----

        ## [672] SAAL: Sharpness-Aware Active Learning

        **Authors**: *Yoon-Yeong Kim, Youngjae Cho, JoonHo Jang, Byeonghu Na, Yeongmin Kim, Kyungwoo Song, Wanmo Kang, Il-Chul Moon*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/kim23c.html](https://proceedings.mlr.press/v202/kim23c.html)

        **Abstract**:

        While deep neural networks play significant roles in many research areas, they are also prone to overfitting problems under limited data instances. To overcome overfitting, this paper introduces the first active learning method to incorporate the sharpness of loss space into the acquisition function. Specifically, our proposed method, Sharpness-Aware Active Learning (SAAL), constructs its acquisition function by selecting unlabeled instances whose perturbed loss becomes maximum. Unlike the Sharpness-Aware learning with fully-labeled datasets, we design a pseudo-labeling mechanism to anticipate the perturbed loss w.r.t. the ground-truth label, which we provide the theoretical bound for the optimization. We conduct experiments on various benchmark datasets for vision-based tasks in image classification, object detection, and domain adaptive semantic segmentation. The experimental results confirm that SAAL outperforms the baselines by selecting instances that have the potentially maximal perturbation on the loss. The code is available at https://github.com/YoonyeongKim/SAAL.

        ----

        ## [673] Demonstration-free Autonomous Reinforcement Learning via Implicit and Bidirectional Curriculum

        **Authors**: *Jigang Kim, Daesol Cho, H. Jin Kim*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/kim23d.html](https://proceedings.mlr.press/v202/kim23d.html)

        **Abstract**:

        While reinforcement learning (RL) has achieved great success in acquiring complex skills solely from environmental interactions, it assumes that resets to the initial state are readily available at the end of each episode. Such an assumption hinders the autonomous learning of embodied agents due to the time-consuming and cumbersome workarounds for resetting in the physical world. Hence, there has been a growing interest in autonomous RL (ARL) methods that are capable of learning from non-episodic interactions. However, existing works on ARL are limited by their reliance on prior data and are unable to learn in environments where task-relevant interactions are sparse. In contrast, we propose a demonstration-free ARL algorithm via Implicit and Bi-directional Curriculum (IBC). With an auxiliary agent that is conditionally activated upon learning progress and a bidirectional goal curriculum based on optimal transport, our method outperforms previous methods, even the ones that leverage demonstrations.

        ----

        ## [674] Improved Algorithms for Multi-period Multi-class Packing Problems with Bandit Feedback

        **Authors**: *Wonyoung Kim, Garud Iyengar, Assaf Zeevi*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/kim23e.html](https://proceedings.mlr.press/v202/kim23e.html)

        **Abstract**:

        We consider the linear contextual multi-class multi-period packing problem (LMMP) where the goal is to pack items such that the total vector of consumption is below a given budget vector and the total value is as large as possible. We consider the setting where the reward and the consumption vector associated with each action is a class-dependent linear function of the context, and the decision-maker receives bandit feedback. LMMP includes linear contextual bandits with knapsacks and online revenue management as special cases. We establish a new estimator which guarantees a faster convergence rate, and consequently, a lower regret in LMMP. We propose a bandit policy that is a closed-form function of said estimated parameters. When the contexts are non-degenerate, the regret of the proposed policy is sublinear in the context dimension, the number of classes, and the time horizon $T$ when the budget grows at least as $\sqrt{T}$. We also resolve an open problem posed in Agrawal & Devanur (2016) and extend the result to a multi-class setting. Our numerical experiments clearly demonstrate that the performance of our policy is superior to other benchmarks in the literature.

        ----

        ## [675] Efficient Latency-Aware CNN Depth Compression via Two-Stage Dynamic Programming

        **Authors**: *Jinuk Kim, Yeonwoo Jeong, Deokjae Lee, Hyun Oh Song*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/kim23f.html](https://proceedings.mlr.press/v202/kim23f.html)

        **Abstract**:

        Recent works on neural network pruning advocate that reducing the depth of the network is more effective in reducing run-time memory usage and accelerating inference latency than reducing the width of the network through channel pruning. In this regard, some recent works propose depth compression algorithms that merge convolution layers. However, the existing algorithms have a constricted search space and rely on human-engineered heuristics. In this paper, we propose a novel depth compression algorithm which targets general convolution operations. We propose a subset selection problem that replaces inefficient activation layers with identity functions and optimally merges consecutive convolution operations into shallow equivalent convolution operations for efficient end-to-end inference latency. Since the proposed subset selection problem is NP-hard, we formulate a surrogate optimization problem that can be solved exactly via two-stage dynamic programming within a few seconds. We evaluate our methods and baselines by TensorRT for a fair inference latency comparison. Our method outperforms the baseline method with higher accuracy and faster inference speed in MobileNetV2 on the ImageNet dataset. Specifically, we achieve $1.41\times$ speed-up with $0.11$%p accuracy gain in MobileNetV2-1.0 on the ImageNet.

        ----

        ## [676] Probabilistic Concept Bottleneck Models

        **Authors**: *Eunji Kim, Dahuin Jung, Sangha Park, Siwon Kim, Sungroh Yoon*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/kim23g.html](https://proceedings.mlr.press/v202/kim23g.html)

        **Abstract**:

        Interpretable models are designed to make decisions in a human-interpretable manner. Representatively, Concept Bottleneck Models (CBM) follow a two-step process of concept prediction and class prediction based on the predicted concepts. CBM provides explanations with high-level concepts derived from concept predictions; thus, reliable concept predictions are important for trustworthiness. In this study, we address the ambiguity issue that can harm reliability. While the existence of a concept can often be ambiguous in the data, CBM predicts concepts deterministically without considering this ambiguity. To provide a reliable interpretation against this ambiguity, we propose Probabilistic Concept Bottleneck Models (ProbCBM). By leveraging probabilistic concept embeddings, ProbCBM models uncertainty in concept prediction and provides explanations based on the concept and its corresponding uncertainty. This uncertainty enhances the reliability of the explanations. Furthermore, as class uncertainty is derived from concept uncertainty in ProbCBM, we can explain class uncertainty by means of concept uncertainty. Code is publicly available at https://github.com/ejkim47/prob-cbm.

        ----

        ## [677] DevFormer: A Symmetric Transformer for Context-Aware Device Placement

        **Authors**: *Haeyeon Kim, Minsu Kim, Federico Berto, Joungho Kim, Jinkyoo Park*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/kim23h.html](https://proceedings.mlr.press/v202/kim23h.html)

        **Abstract**:

        In this paper, we present DevFormer, a novel transformer-based architecture for addressing the complex and computationally demanding problem of hardware design optimization. Despite the demonstrated efficacy of transformers in domains including natural language processing and computer vision, their use in hardware design has been limited by the scarcity of offline data. Our approach addresses this limitation by introducing strong inductive biases such as relative positional embeddings and action-permutation symmetricity that effectively capture the hardware context and enable efficient design optimization with limited offline data. We apply DevFormer to the problem of decoupling capacitor placement and show that it outperforms state-of-the-art methods in both simulated and real hardware, leading to improved performances while reducing the number of components by more than 30%. Finally, we show that our approach achieves promising results in other offline contextual learning-based combinatorial optimization tasks.

        ----

        ## [678] Refining Generative Process with Discriminator Guidance in Score-based Diffusion Models

        **Authors**: *Dongjun Kim, Yeongmin Kim, Se Jung Kwon, Wanmo Kang, Il-Chul Moon*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/kim23i.html](https://proceedings.mlr.press/v202/kim23i.html)

        **Abstract**:

        The proposed method, Discriminator Guidance, aims to improve sample generation of pre-trained diffusion models. The approach introduces a discriminator that gives explicit supervision to a denoising sample path whether it is realistic or not. Unlike GANs, our approach does not require joint training of score and discriminator networks. Instead, we train the discriminator after score training, making discriminator training stable and fast to converge. In sample generation, we add an auxiliary term to the pre-trained score to deceive the discriminator. This term corrects the model score to the data score at the optimal discriminator, which implies that the discriminator helps better score estimation in a complementary way. Using our algorithm, we achive state-of-the-art results on ImageNet 256x256 with FID 1.83 and recall 0.64, similar to the validation data’s FID (1.68) and recall (0.66). We release the code at https://github.com/alsdudrla10/DG.

        ----

        ## [679] Robust Non-Linear Feedback Coding via Power-Constrained Deep Learning

        **Authors**: *Junghoon Kim, Taejoon Kim, David J. Love, Christopher G. Brinton*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/kim23j.html](https://proceedings.mlr.press/v202/kim23j.html)

        **Abstract**:

        The design of codes for feedback-enabled communications has been a long-standing open problem. Recent research on non-linear, deep learning-based coding schemes have demonstrated significant improvements in communication reliability over linear codes, but are still vulnerable to the presence of forward and feedback noise over the channel. In this paper, we develop a new family of non-linear feedback codes that greatly enhance robustness to channel noise. Our autoencoder-based architecture is designed to learn codes based on consecutive blocks of bits, which obtains de-noising advantages over bit-by-bit processing to help overcome the physical separation between the encoder and decoder over a noisy channel. Moreover, we develop a power control layer at the encoder to explicitly incorporate hardware constraints into the learning optimization, and prove that the resulting average power constraint is satisfied asymptotically. Numerical experiments demonstrate that our scheme outperforms state-of-the-art feedback codes by wide margins over practical forward and feedback noise regimes, and provide information-theoretic insights on the behavior of our non-linear codes. Moreover, we observe that, in a long blocklength regime, canonical error correction codes are still preferable to feedback codes when the feedback noise becomes high. Our code is available at https://anonymous.4open.science/r/RCode1.

        ----

        ## [680] LESSON: Learning to Integrate Exploration Strategies for Reinforcement Learning via an Option Framework

        **Authors**: *Woojun Kim, Jeonghye Kim, Youngchul Sung*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/kim23k.html](https://proceedings.mlr.press/v202/kim23k.html)

        **Abstract**:

        In this paper, a unified framework for exploration in reinforcement learning (RL) is proposed based on an option-critic architecture. The proposed framework learns to integrate a set of diverse exploration strategies so that the agent can adaptively select the most effective exploration strategy to realize an effective exploration-exploitation trade-off for each given task. The effectiveness of the proposed exploration framework is demonstrated by various experiments in the MiniGrid and Atari environments.

        ----

        ## [681] BPipe: Memory-Balanced Pipeline Parallelism for Training Large Language Models

        **Authors**: *Taebum Kim, Hyoungjoo Kim, Gyeong-In Yu, Byung-Gon Chun*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/kim23l.html](https://proceedings.mlr.press/v202/kim23l.html)

        **Abstract**:

        Pipeline parallelism is a key technique for training large language models within GPU clusters. However, it often leads to a memory imbalance problem, where certain GPUs face high memory pressure while others underutilize their capacity. This imbalance results in suboptimal training performance, even when the overall GPU memory capacity is sufficient for more efficient setups. To address this inefficiency, we propose BPipe, a novel approach for achieving memory balance in pipeline parallelism. BPipe employs an activation balancing method to transfer intermediate activations between GPUs during training, enabling all GPUs to utilize comparable amounts of memory. With balanced memory utilization, BPipe enhances the training efficiency of large language models like GPT-3 by eliminating redundant recomputations or increasing the micro-batch size. Our evaluation conducted on 48 A100 GPUs across six nodes interconnected with HDR InfiniBand shows that BPipe accelerates the training of GPT-3 96B and GPT-3 134B models by 1.25x-2.17x compared to Megatron-LM, a state-of-the-art framework for training large language models.

        ----

        ## [682] Probabilistic Imputation for Time-series Classification with Missing Data

        **Authors**: *Seunghyun Kim, Hyunsu Kim, Eunggu Yun, Hwangrae Lee, Jaehun Lee, Juho Lee*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/kim23m.html](https://proceedings.mlr.press/v202/kim23m.html)

        **Abstract**:

        Multivariate time series data for real-world applications typically contain a significant amount of missing values. The dominant approach for classification with such missing values is to impute them heuristically with specific values (zero, mean, values of adjacent time-steps) or learnable parameters. However, these simple strategies do not take the data generative process into account, and more importantly, do not effectively capture the uncertainty in prediction due to the multiple possibilities for the missing values. In this paper, we propose a novel probabilistic framework for classification with multivariate time series data with missing values. Our model consists of two parts; a deep generative model for missing value imputation and a classifier. Extending the existing deep generative models to better capture structures of time-series data, our deep generative model part is trained to impute the missing values in multiple plausible ways, effectively modeling the uncertainty of the imputation. The classifier part takes the time series data along with the imputed missing values and classifies signals, and is trained to capture the predictive uncertainty due to the multiple possibilities of imputations. Importantly, we show that naïvely combining the generative model and the classifier could result in trivial solutions where the generative model does not produce meaningful imputations. To resolve this, we present a novel regularization technique that can promote the model to produce useful imputation values that help classification. Through extensive experiments on real-world time series data with missing values, we demonstrate the effectiveness of our method.

        ----

        ## [683] Variational Curriculum Reinforcement Learning for Unsupervised Discovery of Skills

        **Authors**: *Seongun Kim, Kyowoon Lee, Jaesik Choi*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/kim23n.html](https://proceedings.mlr.press/v202/kim23n.html)

        **Abstract**:

        Mutual information-based reinforcement learning (RL) has been proposed as a promising framework for retrieving complex skills autonomously without a task-oriented reward function through mutual information (MI) maximization or variational empowerment. However, learning complex skills is still challenging, due to the fact that the order of training skills can largely affect sample efficiency. Inspired by this, we recast variational empowerment as curriculum learning in goal-conditioned RL with an intrinsic reward function, which we name Variational Curriculum RL (VCRL). From this perspective, we propose a novel approach to unsupervised skill discovery based on information theory, called Value Uncertainty Variational Curriculum (VUVC). We prove that, under regularity conditions, VUVC accelerates the increase of entropy in the visited states compared to the uniform curriculum. We validate the effectiveness of our approach on complex navigation and robotic manipulation tasks in terms of sample efficiency and state coverage speed. We also demonstrate that the skills discovered by our method successfully complete a real-world robot navigation task in a zero-shot setup and that incorporating these skills with a global planner further increases the performance.

        ----

        ## [684] Margin-based Neural Network Watermarking

        **Authors**: *Byungjoo Kim, Suyoung Lee, Seanie Lee, Sooel Son, Sung Ju Hwang*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/kim23o.html](https://proceedings.mlr.press/v202/kim23o.html)

        **Abstract**:

        As Machine Learning as a Service (MLaaS) platforms become prevalent, deep neural network (DNN) watermarking techniques are gaining increasing attention, which enables one to verify the ownership of a target DNN model in a black-box scenario. Unfortunately, previous watermarking methods are vulnerable to functionality stealing attacks, thus allowing an adversary to falsely claim the ownership of a DNN model stolen from its original owner. In this work, we propose a novel margin-based DNN watermarking approach that is robust to the functionality stealing attacks based on model extraction and distillation. Specifically, during training, our method maximizes the margins of watermarked samples by using projected gradient ascent on them so that their predicted labels cannot change without compromising the accuracy of the model that the attacker tries to steal. We validate our method on multiple benchmarks and show that our watermarking method successfully defends against model extraction attacks, outperforming recent baselines.

        ----

        ## [685] Regularizing Towards Soft Equivariance Under Mixed Symmetries

        **Authors**: *Hyunsu Kim, Hyungi Lee, Hongseok Yang, Juho Lee*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/kim23p.html](https://proceedings.mlr.press/v202/kim23p.html)

        **Abstract**:

        Datasets often have their intrinsic symmetries, and particular deep-learning models called equivariant or invariant models have been developed to exploit these symmetries. However, if some or all of these symmetries are only approximate, which frequently happens in practice, these models may be suboptimal due to the architectural restrictions imposed on them. We tackle this issue of approximate symmetries in a setup where symmetries are mixed, i.e., they are symmetries of not single but multiple different types and the degree of approximation varies across these types. Instead of proposing a new architectural restriction as in most of the previous approaches, we present a regularizer-based method for building a model for a dataset with mixed approximate symmetries. The key component of our method is what we call equivariance regularizer for a given type of symmetries, which measures how much a model is equivariant with respect to the symmetries of the type. Our method is trained with these regularizers, one per each symmetry type, and the strength of the regularizers is automatically tuned during training, leading to the discovery of the approximation levels of some candidate symmetry types without explicit supervision. Using synthetic function approximation and motion forecasting tasks, we demonstrate that our method achieves better accuracy than prior approaches while discovering the approximate symmetry levels correctly.

        ----

        ## [686] Model-based Offline Reinforcement Learning with Count-based Conservatism

        **Authors**: *Byeongchan Kim, Min Hwan Oh*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/kim23q.html](https://proceedings.mlr.press/v202/kim23q.html)

        **Abstract**:

        In this paper, we present a model-based offline reinforcement learning method that integrates count-based conservatism, named $\texttt{Count-MORL}$. Our method utilizes the count estimates of state-action pairs to quantify model estimation error, marking the first algorithm of demonstrating the efficacy of count-based conservatism in model-based offline deep RL to the best of our knowledge. For our proposed method, we first show that the estimation error is inversely proportional to the frequency of state-action pairs. Secondly, we demonstrate that the learned policy under the count-based conservative model offers near-optimality performance guarantees. Through extensive numerical experiments, we validate that $\texttt{Count-MORL}$ with hash code implementation significantly outperforms existing offline RL algorithms on the D4RL benchmark datasets. The code is accessible at https://github.com/oh-lab/Count-MORL.

        ----

        ## [687] Transformer-based Stagewise Decomposition for Large-Scale Multistage Stochastic Optimization

        **Authors**: *Chanyeong Kim, Jongwoong Park, Hyunglip Bae, Woo Chang Kim*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/kim23r.html](https://proceedings.mlr.press/v202/kim23r.html)

        **Abstract**:

        Solving large-scale multistage stochastic programming (MSP) problems poses a significant challenge as commonly used stagewise decomposition algorithms, including stochastic dual dynamic programming (SDDP), face growing time complexity as the subproblem size and problem count increase. Traditional approaches approximate the value functions as piecewise linear convex functions by incrementally accumulating subgradient cutting planes from the primal and dual solutions of stagewise subproblems. Recognizing these limitations, we introduce TranSDDP, a novel Transformer-based stagewise decomposition algorithm. This innovative approach leverages the structural advantages of the Transformer model, implementing a sequential method for integrating subgradient cutting planes to approximate the value function. Through our numerical experiments, we affirm TranSDDP’s effectiveness in addressing MSP problems. It efficiently generates a piecewise linear approximation for the value function, significantly reducing computation time while preserving solution quality, thus marking a promising progression in the treatment of large-scale multistage stochastic programming problems.

        ----

        ## [688] SurProGenes: Survival Risk-Ordered Representation of Cancer Patients and Genes for the Identification of Prognostic Genes

        **Authors**: *Junetae Kim, Kyoungsuk Park, Hanseok Jeong, Youngwook Kim, Jeongseon Kim, Sun-Young Kim*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/kim23s.html](https://proceedings.mlr.press/v202/kim23s.html)

        **Abstract**:

        Identifying prognostic genes associated with patient survival is an important goal in cancer genomics, as this information could inform treatment approaches and improve patient outcomes. However, the identification of prognostic genes is complicated by the high dimensionality of genetic data, which makes their identification computationally intensive. Furthermore, most cancer genomics studies lack appropriate low-risk groups against which to compare. To address these issues, we present a framework that identifies candidate prognostic genes by integrating representation learning and statistical analysis approaches. Specifically, we propose a collaborative filtering-derived mechanism to represent patients in order of their survival risk, facilitating their dichotomization. We also propose a mechanism that allows embedded gene vectors to be polarized on the extremities of, or centered on, both reference axes to facilitate recommendations. Restricting our analysis to a few representative genes within each cluster allowed for the efficient identification of prognostic genes. Finally, we demonstrate the potential of this proposed framework for identifying prognostic genes.

        ----

        ## [689] Stable and Consistent Prediction of 3D Characteristic Orientation via Invariant Residual Learning

        **Authors**: *Seungwook Kim, Chunghyun Park, Yoonwoo Jeong, Jaesik Park, Minsu Cho*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/kim23t.html](https://proceedings.mlr.press/v202/kim23t.html)

        **Abstract**:

        Learning to predict reliable characteristic orientations of 3D point clouds is an important yet challenging problem, as different point clouds of the same class may have largely varying appearances. In this work, we introduce a novel method to decouple the shape geometry and semantics of the input point cloud to achieve both stability and consistency. The proposed method integrates shape-geometry-based SO(3)-equivariant learning and shape-semantics-based SO(3)-invariant residual learning, where a final characteristic orientation is obtained by calibrating an SO(3)-equivariant orientation hypothesis using an SO(3)-invariant residual rotation. In experiments, the proposed method not only demonstrates superior stability and consistency but also exhibits state-of-the-art performances when applied to point cloud part segmentation, given randomly rotated inputs.

        ----

        ## [690] Prefer to Classify: Improving Text Classifiers via Auxiliary Preference Learning

        **Authors**: *Jaehyung Kim, Jinwoo Shin, Dongyeop Kang*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/kim23u.html](https://proceedings.mlr.press/v202/kim23u.html)

        **Abstract**:

        The development of largely human-annotated benchmarks has driven the success of deep neural networks in various NLP tasks. To enhance the effectiveness of existing benchmarks, collecting new additional input-output pairs is often too costly and challenging, particularly considering their marginal impact on improving the current model accuracy. Instead, additional or complementary annotations on the existing input texts in the benchmarks can be preferable as an efficient way to pay the additional human cost. In this paper, we investigate task-specific preferences between pairs of input texts as a new alternative way for such auxiliary data annotation. From pair-wise comparisons with respect to the task, the auxiliary preference learning enables the model to learn an additional informative training signal that cannot be captured with instance-wise task labels. To this end, we propose a novel multi-task learning framework, called prefer-to-classify (P2C), which can enjoy the cooperative effect of learning both the given classification task and the auxiliary preferences. Here, we provide three different ways to collect preference signals in practice: (a) implicitly extracting from annotation records (for free, but often unavailable), (b) collecting explicitly from crowd workers (high paid), or (c) pre-trained large language models such as GPT-3 (low paid). Given existing classification NLP benchmarks, we demonstrate that the proposed auxiliary preference learning via P2C on them is effective in improving text classifiers. Our codes are publicly available.

        ----

        ## [691] An Adaptive Entropy-Regularization Framework for Multi-Agent Reinforcement Learning

        **Authors**: *Woojun Kim, Youngchul Sung*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/kim23v.html](https://proceedings.mlr.press/v202/kim23v.html)

        **Abstract**:

        In this paper, we propose an adaptive entropy-regularization framework (ADER) for multi-agent reinforcement learning (RL) to learn the adequate amount of exploration of each agent for entropy-based exploration. In order to derive a metric for the proper level of exploration entropy for each agent, we disentangle the soft value function into two types: one for pure return and the other for entropy. By applying multi-agent value factorization to the disentangled value function of pure return, we obtain a metric to determine the relevant level of exploration entropy for each agent, given by the partial derivative of the pure-return value function with respect to (w.r.t.) the policy entropy of each agent. Based on this metric, we propose the ADER algorithm based on maximum entropy RL, which controls the necessary level of exploration across agents over time by learning the proper target entropy for each agent. Experimental results show that the proposed scheme significantly outperforms current state-of-the-art multi-agent RL algorithms.

        ----

        ## [692] Practical and Matching Gradient Variance Bounds for Black-Box Variational Bayesian Inference

        **Authors**: *Kyurae Kim, Kaiwen Wu, Jisu Oh, Jacob R. Gardner*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/kim23w.html](https://proceedings.mlr.press/v202/kim23w.html)

        **Abstract**:

        Understanding the gradient variance of black-box variational inference (BBVI) is a crucial step for establishing its convergence and developing algorithmic improvements. However, existing studies have yet to show that the gradient variance of BBVI satisfies the conditions used to study the convergence of stochastic gradient descent (SGD), the workhorse of BBVI. In this work, we show that BBVI satisfies a matching bound corresponding to the ABC condition used in the SGD literature when applied to smooth and quadratically-growing log-likelihoods. Our results generalize to nonlinear covariance parameterizations widely used in the practice of BBVI. Furthermore, we show that the variance of the mean-field parameterization has provably superior dimensional dependence.

        ----

        ## [693] Learnability and Algorithm for Continual Learning

        **Authors**: *Gyuhak Kim, Changnan Xiao, Tatsuya Konishi, Bing Liu*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/kim23x.html](https://proceedings.mlr.press/v202/kim23x.html)

        **Abstract**:

        This paper studies the challenging continual learning (CL) setting of Class Incremental Learning (CIL). CIL learns a sequence of tasks consisting of disjoint sets of concepts or classes. At any time, a single model is built that can be applied to predict/classify test instances of any classes learned thus far without providing any task related information for each test instance. Although many techniques have been proposed for CIL, they are mostly empirical. It has been shown recently that a strong CIL system needs a strong within-task prediction (WP) and a strong out-of-distribution (OOD) detection for each task. However, it is still not known whether CIL is actually learnable. This paper shows that CIL is learnable. Based on the theory, a new CIL algorithm is also proposed. Experimental results demonstrate its effectiveness.

        ----

        ## [694] Unifying Nesterov's Accelerated Gradient Methods for Convex and Strongly Convex Objective Functions

        **Authors**: *Jungbin Kim, Insoon Yang*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/kim23y.html](https://proceedings.mlr.press/v202/kim23y.html)

        **Abstract**:

        Although Nesterov’s accelerated gradient method (AGM) has been studied from various perspectives, it remains unclear why the most popular forms of AGMs must handle convex and strongly convex objective functions separately. To address this inconsistency, we propose a novel unified framework for Lagrangians, ordinary differential equation (ODE) models, and algorithms. As a special case, our new simple momentum algorithm, which we call the unified AGM, seamlessly bridges the gap between the two most popular forms of Nesterov’s AGM and has a superior convergence guarantee compared to existing algorithms for non-strongly convex objective functions. This property is beneficial in practice when considering ill-conditioned $\mu$-strongly convex objective functions (with small $\mu$). Furthermore, we generalize this algorithm and the corresponding ODE model to the higher-order non-Euclidean setting. Last but not least, our unified framework is used to construct the unified AGM-G ODE, a novel ODE model for minimizing the gradient norm of strongly convex functions.

        ----

        ## [695] Denoising MCMC for Accelerating Diffusion-Based Generative Models

        **Authors**: *Beomsu Kim, Jong Chul Ye*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/kim23z.html](https://proceedings.mlr.press/v202/kim23z.html)

        **Abstract**:

        The sampling process of diffusion models can be interpreted as solving the reverse stochastic differential equation (SDE) or the ordinary differential equation (ODE) of the diffusion process, which often requires up to thousands of discretization steps to generate a single image. This has sparked a great interest in developing efficient integration techniques for reverse-S/ODEs. Here, we propose an orthogonal approach to accelerating score-based sampling: Denoising MCMC (DMCMC). DMCMC first uses MCMC to produce initialization points for reverse-S/ODE in the product space of data and diffusion time. Then, a reverse-S/ODE integrator is used to denoise the initialization points. Since MCMC traverses close to the data manifold, the cost of producing a clean sample for DMCMC is much less than that of producing a clean sample from noise. Denoising Langevin Gibbs, an instance of DMCMC, successfully accelerates all six reverse-S/ODE integrators considered in this work, and achieves state-of-the-art results: in the limited number of score function evaluation (NFE) setting on CIFAR10, we have $3.25$ FID with $\approx 10$ NFE and $2.49$ FID with $\approx 16$ NFE. On CelebA-HQ-256, we have $6.99$ FID with $\approx 160$ NFE, which beats the current best record of Kim et al. (2022) among score-based models, $7.16$ FID with $4000$ NFE. Code: https://github.com/1202kbs/DMCMC

        ----

        ## [696] Structure Learning of Latent Factors via Clique Search on Correlation Thresholded Graphs

        **Authors**: *Dale Kim, Qing Zhou*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/kim23aa.html](https://proceedings.mlr.press/v202/kim23aa.html)

        **Abstract**:

        Despite the widespread application of latent factor analysis, existing methods suffer from the following weaknesses: requiring the number of factors to be known, lack of theoretical guarantees for learning the model structure, and nonidentifiability of the parameters due to rotation invariance properties of the likelihood. We address these concerns by proposing a fast correlation thresholding (CT) algorithm that simultaneously learns the number of latent factors and a rotationally identifiable model structure. Our novel approach translates this structure learning problem into the search for so-called independent maximal cliques in a thresholded correlation graph that can be easily constructed from the observed data. Our clique analysis technique scales well up to thousands of variables, while competing methods are not applicable in a reasonable amount of running time. We establish a finite-sample error bound and high-dimensional consistency for the structure learning of our method. Through a series of simulation studies and a real data example, we show that the CT algorithm is an accurate method for learning the structure of factor analysis models and is robust to violations of its assumptions.

        ----

        ## [697] Fair and Robust Estimation of Heterogeneous Treatment Effects for Policy Learning

        **Authors**: *Kwangho Kim, José R. Zubizarreta*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/kim23ab.html](https://proceedings.mlr.press/v202/kim23ab.html)

        **Abstract**:

        We propose a simple and general framework for nonparametric estimation of heterogeneous treatment effects under fairness constraints. Under standard regularity conditions, we show that the resulting estimators possess the double robustness property. We use this framework to characterize the trade-off between fairness and the maximum welfare achievable by the optimal policy. We evaluate the methods in a simulation study and illustrate them in a real-world case study.

        ----

        ## [698] Proper Losses for Discrete Generative Models

        **Authors**: *Dhamma Kimpara, Rafael M. Frongillo, Bo Waggoner*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/kimpara23a.html](https://proceedings.mlr.press/v202/kimpara23a.html)

        **Abstract**:

        We initiate the study of proper losses for evaluating generative models in the discrete setting. Unlike traditional proper losses, we treat both the generative model and the target distribution as black-boxes, only assuming ability to draw i.i.d. samples. We define a loss to be black-box proper if the generative distribution that minimizes expected loss is equal to the target distribution. Using techniques from statistical estimation theory, we give a general construction and characterization of black-box proper losses: they must take a polynomial form, and the number of draws from the model and target distribution must exceed the degree of the polynomial. The characterization rules out a loss whose expectation is the cross-entropy between the target distribution and the model. By extending the construction to arbitrary sampling schemes such as Poisson sampling, however, we show that one can construct such a loss.

        ----

        ## [699] Controlling Posterior Collapse by an Inverse Lipschitz Constraint on the Decoder Network

        **Authors**: *Yuri Kinoshita, Kenta Oono, Kenji Fukumizu, Yuichi Yoshida, Shin-ichi Maeda*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/kinoshita23a.html](https://proceedings.mlr.press/v202/kinoshita23a.html)

        **Abstract**:

        Variational autoencoders (VAEs) are one of the deep generative models that have experienced enormous success over the past decades. However, in practice, they suffer from a problem called posterior collapse, which occurs when the posterior distribution coincides, or collapses, with the prior taking no information from the latent structure of the input data into consideration. In this work, we introduce an inverse Lipschitz neural network into the decoder and, based on this architecture, provide a new method that can control in a simple and clear manner the degree of posterior collapse for a wide range of VAE models equipped with a concrete theoretical guarantee. We also illustrate the effectiveness of our method through several numerical experiments.

        ----

        ## [700] A Watermark for Large Language Models

        **Authors**: *John Kirchenbauer, Jonas Geiping, Yuxin Wen, Jonathan Katz, Ian Miers, Tom Goldstein*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/kirchenbauer23a.html](https://proceedings.mlr.press/v202/kirchenbauer23a.html)

        **Abstract**:

        Potential harms of large language models can be mitigated by watermarking model output, i.e., embedding signals into generated text that are invisible to humans but algorithmically detectable from a short span of tokens. We propose a watermarking framework for proprietary language models. The watermark can be embedded with negligible impact on text quality, and can be detected using an efficient open-source algorithm without access to the language model API or parameters. The watermark works by selecting a randomized set of "green" tokens before a word is generated, and then softly promoting use of green tokens during sampling. We propose a statistical test for detecting the watermark with interpretable p-values, and derive an information-theoretic framework for analyzing the sensitivity of the watermark. We test the watermark using a multi-billion parameter model from the Open Pretrained Transformer (OPT) family, and discuss robustness and security.

        ----

        ## [701] Probabilistic Contrastive Learning Recovers the Correct Aleatoric Uncertainty of Ambiguous Inputs

        **Authors**: *Michael Kirchhof, Enkelejda Kasneci, Seong Joon Oh*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/kirchhof23a.html](https://proceedings.mlr.press/v202/kirchhof23a.html)

        **Abstract**:

        Contrastively trained encoders have recently been proven to invert the data-generating process: they encode each input, e.g., an image, into the true latent vector that generated the image (Zimmermann et al., 2021). However, real-world observations often have inherent ambiguities. For instance, images may be blurred or only show a 2D view of a 3D object, so multiple latents could have generated them. This makes the true posterior for the latent vector probabilistic with heteroscedastic uncertainty. In this setup, we extend the common InfoNCE objective and encoders to predict latent distributions instead of points. We prove that these distributions recover the correct posteriors of the data-generating process, including its level of aleatoric uncertainty, up to a rotation of the latent space. In addition to providing calibrated uncertainty estimates, these posteriors allow the computation of credible intervals in image retrieval. They comprise images with the same latent as a given query, subject to its uncertainty. Code is at https://github.com/mkirchhof/Probabilistic_Contrastive_Learning .

        ----

        ## [702] Training Normalizing Flows from Dependent Data

        **Authors**: *Matthias Kirchler, Christoph Lippert, Marius Kloft*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/kirchler23a.html](https://proceedings.mlr.press/v202/kirchler23a.html)

        **Abstract**:

        Normalizing flows are powerful non-parametric statistical models that function as a hybrid between density estimators and generative models. Current learning algorithms for normalizing flows assume that data points are sampled independently, an assumption that is frequently violated in practice, which may lead to erroneous density estimation and data generation. We propose a likelihood objective of normalizing flows incorporating dependencies between the data points, for which we derive a flexible and efficient learning algorithm suitable for different dependency structures. We show that respecting dependencies between observations can improve empirical results on both synthetic and real-world data, and leads to higher statistical power in a downstream application to genome-wide association studies.

        ----

        ## [703] IncDSI: Incrementally Updatable Document Retrieval

        **Authors**: *Varsha Kishore, Chao Wan, Justin Lovelace, Yoav Artzi, Kilian Q. Weinberger*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/kishore23a.html](https://proceedings.mlr.press/v202/kishore23a.html)

        **Abstract**:

        Differentiable Search Index is a recently proposed paradigm for document retrieval, that encodes information about a corpus of documents within the parameters of a neural network and directly maps queries to corresponding documents. These models have achieved state-of-the-art performances for document retrieval across many benchmarks. These kinds of models have a significant limitation: it is not easy to add new documents after a model is trained. We propose IncDSI, a method to add documents in real time (about 20-50ms per document), without retraining the model on the entire dataset (or even parts thereof). Instead we formulate the addition of documents as a constrained optimization problem that makes minimal changes to the network parameters. Although orders of magnitude faster, our approach is competitive with re-training the model on the whole dataset and enables the development of document retrieval systems that can be updated with new information in real-time. Our code for IncDSI is available at https://github.com/varshakishore/IncDSI.

        ----

        ## [704] Regularization and Variance-Weighted Regression Achieves Minimax Optimality in Linear MDPs: Theory and Practice

        **Authors**: *Toshinori Kitamura, Tadashi Kozuno, Yunhao Tang, Nino Vieillard, Michal Valko, Wenhao Yang, Jincheng Mei, Pierre Ménard, Mohammad Gheshlaghi Azar, Rémi Munos, Olivier Pietquin, Matthieu Geist, Csaba Szepesvári, Wataru Kumagai, Yutaka Matsuo*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/kitamura23a.html](https://proceedings.mlr.press/v202/kitamura23a.html)

        **Abstract**:

        Mirror descent value iteration (MDVI), an abstraction of Kullback-Leibler (KL) and entropy-regularized reinforcement learning (RL), has served as the basis for recent high-performing practical RL algorithms. However, despite the use of function approximation in practice, the theoretical understanding of MDVI has been limited to tabular Markov decision processes (MDPs). We study MDVI with linear function approximation through its sample complexity required to identify an $\varepsilon$-optimal policy with probability $1-\delta$ under the settings of an infinite-horizon linear MDP, generative model, and G-optimal design. We demonstrate that least-squares regression weighted by the variance of an estimated optimal value function of the next state is crucial to achieving minimax optimality. Based on this observation, we present Variance-Weighted Least-Squares MDVI (VWLS-MDVI), the first theoretical algorithm that achieves nearly minimax optimal sample complexity for infinite-horizon linear MDPs. Furthermore, we propose a practical VWLS algorithm for value-based deep RL, Deep Variance Weighting (DVW). Our experiments demonstrate that DVW improves the performance of popular value-based deep RL algorithms on a set of MinAtar benchmarks.

        ----

        ## [705] Drug Discovery under Covariate Shift with Domain-Informed Prior Distributions over Functions

        **Authors**: *Leo Klarner, Tim G. J. Rudner, Michael Reutlinger, Torsten Schindler, Garrett M. Morris, Charlotte M. Deane, Yee Whye Teh*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/klarner23a.html](https://proceedings.mlr.press/v202/klarner23a.html)

        **Abstract**:

        Accelerating the discovery of novel and more effective therapeutics is an important pharmaceutical problem in which deep learning is playing an increasingly significant role. However, real-world drug discovery tasks are often characterized by a scarcity of labeled data and significant covariate shift—a setting that poses a challenge to standard deep learning methods. In this paper, we present Q-SAVI, a probabilistic model able to address these challenges by encoding explicit prior knowledge of the data-generating process into a prior distribution over functions, presenting researchers with a transparent and probabilistically principled way to encode data-driven modeling preferences. Building on a novel, gold-standard bioactivity dataset that facilitates a meaningful comparison of models in an extrapolative regime, we explore different approaches to induce data shift and construct a challenging evaluation setup. We then demonstrate that using Q-SAVI to integrate contextualized prior knowledge of drug-like chemical space into the modeling process affords substantial gains in predictive accuracy and calibration, outperforming a broad range of state-of-the-art self-supervised pre-training and domain adaptation techniques.

        ----

        ## [706] Deep Laplacian-based Options for Temporally-Extended Exploration

        **Authors**: *Martin Klissarov, Marlos C. Machado*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/klissarov23a.html](https://proceedings.mlr.press/v202/klissarov23a.html)

        **Abstract**:

        Selecting exploratory actions that generate a rich stream of experience for better learning is a fundamental challenge in reinforcement learning (RL). An approach to tackle this problem consists in selecting actions according to specific policies for an extended period of time, also known as options. A recent line of work to derive such exploratory options builds upon the eigenfunctions of the graph Laplacian. Importantly, until now these methods have been mostly limited to tabular domains where (1) the graph Laplacian matrix was either given or could be fully estimated, (2) performing eigendecomposition on this matrix was computationally tractable, and (3) value functions could be learned exactly. Additionally, these methods required a separate option discovery phase. These assumptions are fundamentally not scalable. In this paper we address these limitations and show how recent results for directly approximating the eigenfunctions of the Laplacian can be leveraged to truly scale up options-based exploration. To do so, we introduce a fully online deep RL algorithm for discovering Laplacian-based options and evaluate our approach on a variety of pixel-based tasks. We compare to several state-of-the-art exploration methods and show that our approach is effective, general, and especially promising in non-stationary settings.

        ----

        ## [707] Generalized Reductions: Making any Hierarchical Clustering Fair and Balanced with Low Cost

        **Authors**: *Marina Knittel, Max Springer, John P. Dickerson, MohammadTaghi Hajiaghayi*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/knittel23a.html](https://proceedings.mlr.press/v202/knittel23a.html)

        **Abstract**:

        Clustering is a fundamental building block of modern statistical analysis pipelines. Fair clustering has seen much attention from the machine learning community in recent years. We are some of the first to study fairness in the context of hierarchical clustering, after the results of Ahmadian et al. from NeurIPS in 2020. We evaluate our results using Dasgupta’s cost function, perhaps one of the most prevalent theoretical metrics for hierarchical clustering evaluation. Our work vastly improves the previous $O(n^{5/6}poly\log(n))$ fair approximation for cost to a near polylogarithmic $O(n^\delta poly\log(n))$ fair approximation for any constant $\delta\in(0,1)$. This result establishes a cost fairness tradeoff and extends to broader fairness constraints than the previous work. We also show how to alter existing hierarchical clusterings to guarantee fairness and cluster balance across any level in the hierarchy.

        ----

        ## [708] Can We Scale Transformers to Predict Parameters of Diverse ImageNet Models?

        **Authors**: *Boris Knyazev, Doha Hwang, Simon Lacoste-Julien*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/knyazev23a.html](https://proceedings.mlr.press/v202/knyazev23a.html)

        **Abstract**:

        Pretraining a neural network on a large dataset is becoming a cornerstone in machine learning that is within the reach of only a few communities with large-resources. We aim at an ambitious goal of democratizing pretraining. Towards that goal, we train and release a single neural network that can predict high quality ImageNet parameters of other neural networks. By using predicted parameters for initialization we are able to boost training of diverse ImageNet models available in PyTorch. When transferred to other datasets, models initialized with predicted parameters also converge faster and reach competitive final performance.

        ----

        ## [709] Online Learning with Feedback Graphs: The True Shape of Regret

        **Authors**: *Tomás Kocák, Alexandra Carpentier*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/kocak23a.html](https://proceedings.mlr.press/v202/kocak23a.html)

        **Abstract**:

        Sequential learning with feedback graphs is a natural extension of the multi-armed bandit problem where the problem is equipped with an underlying graph structure that provides additional information - playing an action reveals the losses of all the neighbors of the action. This problem was introduced by Mannor & Shamir (2011) and received considerable attention in recent years. It is generally stated in the literature that the minimax regret rate for this problem is of order $\sqrt{\alpha T}$, where $\alpha$ is the independence number of the graph, and $T$ is the time horizon. However, this is proven only when the number of rounds $T$ is larger than $\alpha^3$, which poses a significant restriction for the usability of this result in large graphs. In this paper, we define a new quantity $R^*$, called the problem complexity, and prove that the minimax regret is proportional to $R^*$ for any graph and time horizon $T$. Introducing an intricate exploration strategy, we define the Exp3-EX algorithm that achieves the minimax optimal regret bound and becomes the first provably optimal algorithm for this setting, even if $T$ is smaller than $\alpha^3$.

        ----

        ## [710] Grounding Language Models to Images for Multimodal Inputs and Outputs

        **Authors**: *Jing Yu Koh, Ruslan Salakhutdinov, Daniel Fried*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/koh23a.html](https://proceedings.mlr.press/v202/koh23a.html)

        **Abstract**:

        We propose an efficient method to ground pretrained text-only language models to the visual domain, enabling them to process arbitrarily interleaved image-and-text data, and generate text interleaved with retrieved images. Our method leverages the abilities of language models learnt from large scale text-only pretraining, such as in-context learning and free-form text generation. We keep the language model frozen, and finetune input and output linear layers to enable cross-modality interactions. This allows our model to process arbitrarily interleaved image-and-text inputs, and generate free-form text interleaved with retrieved images. We achieve strong zero-shot performance on grounded tasks such as contextual image retrieval and multimodal dialogue, and showcase compelling interactive abilities. Our approach works with any off-the-shelf language model and paves the way towards an effective, general solution for leveraging pretrained language models in visually grounded settings.

        ----

        ## [711] Rigid Body Flows for Sampling Molecular Crystal Structures

        **Authors**: *Jonas Köhler, Michele Invernizzi, Pim de Haan, Frank Noé*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/kohler23a.html](https://proceedings.mlr.press/v202/kohler23a.html)

        **Abstract**:

        Normalizing flows (NF) are a class of powerful generative models that have gained popularity in recent years due to their ability to model complex distributions with high flexibility and expressiveness. In this work, we introduce a new type of normalizing flow that is tailored for modeling positions and orientations of multiple objects in three-dimensional space, such as molecules in a crystal. Our approach is based on two key ideas: first, we define smooth and expressive flows on the group of unit quaternions, which allows us to capture the continuous rotational motion of rigid bodies; second, we use the double cover property of unit quaternions to define a proper density on the rotation group. This ensures that our model can be trained using standard likelihood-based methods or variational inference with respect to a thermodynamic target density. We evaluate the method by training Boltzmann generators for two molecular examples, namely the multi-modal density of a tetrahedral system in an external field and the ice XI phase in the TIP4P water model. Our flows can be combined with flows operating on the internal degrees of freedom of molecules and constitute an important step towards the modeling of distributions of many interacting molecules.

        ----

        ## [712] Enabling First-Order Gradient-Based Learning for Equilibrium Computation in Markets

        **Authors**: *Nils Kohring, Fabian Raoul Pieroth, Martin Bichler*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/kohring23a.html](https://proceedings.mlr.press/v202/kohring23a.html)

        **Abstract**:

        Understanding and analyzing markets is crucial, yet analytical equilibrium solutions remain largely infeasible. Recent breakthroughs in equilibrium computation rely on zeroth-order policy gradient estimation. These approaches commonly suffer from high variance and are computationally expensive. The use of fully differentiable simulators would enable more efficient gradient estimation. However, the discrete allocation of goods in economic simulations is a non-differentiable operation. This renders the first-order Monte Carlo gradient estimator inapplicable and the learning feedback systematically misleading. We propose a novel smoothing technique that creates a surrogate market game, in which first-order methods can be applied. We provide theoretical bounds on the resulting bias which justifies solving the smoothed game instead. These bounds also allow choosing the smoothing strength a priori such that the resulting estimate has low variance. Furthermore, we validate our approach via numerous empirical experiments. Our method theoretically and empirically outperforms zeroth-order methods in approximation quality and computational efficiency.

        ----

        ## [713] Revisiting Gradient Clipping: Stochastic bias and tight convergence guarantees

        **Authors**: *Anastasia Koloskova, Hadrien Hendrikx, Sebastian U. Stich*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/koloskova23a.html](https://proceedings.mlr.press/v202/koloskova23a.html)

        **Abstract**:

        Gradient clipping is a popular modification to standard (stochastic) gradient descent, at every iteration limiting the gradient norm to a certain value $c >0$. It is widely used for example for stabilizing the training of deep learning models (Goodfellow et al., 2016), or for enforcing differential privacy (Abadi et al., 2016). Despite popularity and simplicity of the clipping mechanism, its convergence guarantees often require specific values of $c$ and strong noise assumptions. In this paper, we give convergence guarantees that show precise dependence on arbitrary clipping thresholds $c$ and show that our guarantees are tight with both deterministic and stochastic gradients. In particular, we show that (i) for deterministic gradient descent, the clipping threshold only affects the higher-order terms of convergence, (ii) in the stochastic setting convergence to the true optimum cannot be guaranteed under the standard noise assumption, even under arbitrary small step-sizes. We give matching upper and lower bounds for convergence of the gradient norm when running clipped SGD, and illustrate these results with experiments.

        ----

        ## [714] On Computing Optimal Tree Ensembles

        **Authors**: *Christian Komusiewicz, Pascal Kunz, Frank Sommer, Manuel Sorge*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/komusiewicz23a.html](https://proceedings.mlr.press/v202/komusiewicz23a.html)

        **Abstract**:

        Random forests and, more generally, (decision-)tree ensembles are widely used methods for classification and regression. Recent algorithmic advances allow to compute decision trees that are optimal for various measures such as their size or depth. We are not aware of such research for tree ensembles and aim to contribute to this area. Mainly, we provide two novel algorithms and corresponding lower bounds. First, we are able to carry over and substantially improve on tractability results for decision trees, obtaining a $(6\delta D S)^S \cdot \mathrm{poly}$-time algorithm, where $S$ is the number of cuts in the tree ensemble, $D$ the largest domain size, and $\delta$ is the largest number of features in which two examples differ. To achieve this, we introduce the witness-tree technique which also seems promising for practice. Second, we show that dynamic programming, which has been successful for decision trees, may also be viable for tree ensembles, providing an $\ell^n \cdot \mathrm{poly}$-time algorithm, where $\ell$ is the number of trees and $n$ the number of examples. Finally, we compare the number of cuts necessary to classify training data sets for decision trees and tree ensembles, showing that ensembles may need exponentially fewer cuts for increasing number of trees.

        ----

        ## [715] GOAT: A Global Transformer on Large-scale Graphs

        **Authors**: *Kezhi Kong, Jiuhai Chen, John Kirchenbauer, Renkun Ni, C. Bayan Bruss, Tom Goldstein*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/kong23a.html](https://proceedings.mlr.press/v202/kong23a.html)

        **Abstract**:

        Graph transformers have been competitive on graph classification tasks, but they fail to outperform Graph Neural Networks (GNNs) on node classification, which is a common task performed on large-scale graphs for industrial applications. Meanwhile, existing GNN architectures are limited in their ability to perform equally well on both homophilious and heterophilious graphs as their inductive biases are generally tailored to only one setting. To address these issues, we propose GOAT, a scalable global graph transformer. In GOAT, each node conceptually attends to all the nodes in the graph and homophily/heterophily relationships can be learnt adaptively from the data. We provide theoretical justification for our approximate global self-attention scheme, and show it to be scalable to large-scale graphs. We demonstrate the competitiveness of GOAT on both heterophilious and homophilious graphs with millions of nodes.

        ----

        ## [716] Autoregressive Diffusion Model for Graph Generation

        **Authors**: *Lingkai Kong, Jiaming Cui, Haotian Sun, Yuchen Zhuang, B. Aditya Prakash, Chao Zhang*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/kong23b.html](https://proceedings.mlr.press/v202/kong23b.html)

        **Abstract**:

        Diffusion-based graph generative models have recently obtained promising results for graph generation. However, existing diffusion-based graph generative models are mostly one-shot generative models that apply Gaussian diffusion in the dequantized adjacency matrix space. Such a strategy can suffer from difficulty in model training, slow sampling speed, and incapability of incorporating constraints. We propose an autoregressive diffusion model for graph generation. Unlike existing methods, we define a node-absorbing diffusion process that operates directly in the discrete graph space. For forward diffusion, we design a diffusion ordering network, which learns a data-dependent node absorbing ordering from graph topology. For reverse generation, we design a denoising network that uses the reverse node ordering to efficiently reconstruct the graph by predicting the node type of the new node and its edges with previously denoised nodes at a time. Based on the permutation invariance of graph, we show that the two networks can be jointly trained by optimizing a simple lower bound of data likelihood. Our experiments on six diverse generic graph datasets and two molecule datasets show that our model achieves better or comparable generation performance with previous state-of-the-art, and meanwhile enjoys fast generation speed.

        ----

        ## [717] End-to-End Full-Atom Antibody Design

        **Authors**: *Xiangzhe Kong, Wenbing Huang, Yang Liu*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/kong23c.html](https://proceedings.mlr.press/v202/kong23c.html)

        **Abstract**:

        Antibody design is an essential yet challenging task in various domains like therapeutics and biology. There are two major defects in current learning-based methods: 1) tackling only a certain subtask of the whole antibody design pipeline, making them suboptimal or resource-intensive. 2) omitting either the framework regions or side chains, thus incapable of capturing the full-atom geometry. To address these pitfalls, we propose dynamic Multi-channel Equivariant grAph Network (dyMEAN), an end-to-end full-atom model for E(3)-equivariant antibody design given the epitope and the incomplete sequence of the antibody. Specifically, we first explore structural initialization as a knowledgeable guess of the antibody structure and then propose shadow paratope to bridge the epitope-antibody connections. Both 1D sequences and 3D structures are updated via an adaptive multi-channel equivariant encoder that is able to process protein residues of variable sizes when considering full atoms. Finally, the updated antibody is docked to the epitope via the alignment of the shadow paratope. Experiments on epitope-binding CDR-H3 design, complex structure prediction, and affinity optimization demonstrate the superiority of our end-to-end framework and full-atom modeling.

        ----

        ## [718] Covariate balancing using the integral probability metric for causal inference

        **Authors**: *Insung Kong, Yuha Park, Joonhyuk Jung, Kwonsang Lee, Yongdai Kim*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/kong23d.html](https://proceedings.mlr.press/v202/kong23d.html)

        **Abstract**:

        Weighting methods in causal inference have been widely used to achieve a desirable level of covariate balancing. However, the existing weighting methods have desirable theoretical properties only when a certain model, either the propensity score or outcome regression model, is correctly specified. In addition, the corresponding estimators do not behave well for finite samples due to large variance even when the model is correctly specified. In this paper, we consider to use the integral probability metric (IPM), which is a metric between two probability measures, for covariate balancing. Optimal weights are determined so that weighted empirical distributions for the treated and control groups have the smallest IPM value for a given set of discriminators. We prove that the corresponding estimator can be consistent without correctly specifying any model (neither the propensity score nor the outcome regression model). In addition, we empirically show that our proposed method outperforms existing weighting methods with large margins for finite samples.

        ----

        ## [719] Masked Bayesian Neural Networks : Theoretical Guarantee and its Posterior Inference

        **Authors**: *Insung Kong, Dongyoon Yang, Jongjin Lee, Ilsang Ohn, Gyuseung Baek, Yongdai Kim*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/kong23e.html](https://proceedings.mlr.press/v202/kong23e.html)

        **Abstract**:

        Bayesian approaches for learning deep neural networks (BNN) have been received much attention and successfully applied to various applications. Particularly, BNNs have the merit of having better generalization ability as well as better uncertainty quantification. For the success of BNN, search an appropriate architecture of the neural networks is an important task, and various algorithms to find good sparse neural networks have been proposed. In this paper, we propose a new node-sparse BNN model which has good theoretical properties and is computationally feasible. We prove that the posterior concentration rate to the true model is near minimax optimal and adaptive to the smoothness of the true model. In particular the adaptiveness is the first of its kind for node-sparse BNNs. In addition, we develop a novel MCMC algorithm which makes the Bayesian inference of the node-sparse BNN model feasible in practice.

        ----

        ## [720] Parameter-Level Soft-Masking for Continual Learning

        **Authors**: *Tatsuya Konishi, Mori Kurokawa, Chihiro Ono, Zixuan Ke, Gyuhak Kim, Bing Liu*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/konishi23a.html](https://proceedings.mlr.press/v202/konishi23a.html)

        **Abstract**:

        Existing research on task incremental learning in continual learning has primarily focused on preventing catastrophic forgetting (CF). Although several techniques have achieved learning with no CF, they attain it by letting each task monopolize a sub-network in a shared network, which seriously limits knowledge transfer (KT) and causes over-consumption of the network capacity, i.e., as more tasks are learned, the performance deteriorates. The goal of this paper is threefold: (1) overcoming CF, (2) encouraging KT, and (3) tackling the capacity problem. A novel technique (called SPG) is proposed that soft-masks (partially blocks) parameter updating in training based on the importance of each parameter to old tasks. Each task still uses the full network, i.e., no monopoly of any part of the network by any task, which enables maximum KT and reduction in capacity usage. To our knowledge, this is the first work that soft-masks a model at the parameter-level for continual learning. Extensive experiments demonstrate the effectiveness of SPG in achieving all three objectives. More notably, it attains significant transfer of knowledge not only among similar tasks (with shared knowledge) but also among dissimilar tasks (with little shared knowledge) while mitigating CF.

        ----

        ## [721] Pretraining Language Models with Human Preferences

        **Authors**: *Tomasz Korbak, Kejian Shi, Angelica Chen, Rasika Vinayak Bhalerao, Christopher L. Buckley, Jason Phang, Samuel R. Bowman, Ethan Perez*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/korbak23a.html](https://proceedings.mlr.press/v202/korbak23a.html)

        **Abstract**:

        Language models (LMs) are pretrained to imitate text from large and diverse datasets that contain content that would violate human preferences if generated by an LM: falsehoods, offensive comments, personally identifiable information, low-quality or buggy code, among others. Here, we explore alternative objectives for pretraining LMs in a way that also guides them to generate text aligned with human preferences. We benchmark five objectives for pretraining with human feedback across three tasks and study how they affect the alignment and capabilities of pretrained LMs. We find a Pareto-optimal and simple approach among those we explored: conditional training, or learning distribution over tokens conditional on their human preference scores. Conditional training reduces the rate of undesirable content by up to an order of magnitude, both when generating without a prompt and with an adversarially-chosen prompt. Moreover, conditional training maintains the downstream task performance of standard LM pretraining, both before and after task-specific finetuning. Pretraining with human feedback results in much better preference satisfaction than standard LM pretraining followed by finetuning with feedback, i.e., learning and then unlearning undesirable behavior. Our results suggest that we should move beyond imitation learning when pretraining LMs and incorporate human preferences from the start of training.

        ----

        ## [722] Detecting Adversarial Directions in Deep Reinforcement Learning to Make Robust Decisions

        **Authors**: *Ezgi Korkmaz, Jonah Brown-Cohen*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/korkmaz23a.html](https://proceedings.mlr.press/v202/korkmaz23a.html)

        **Abstract**:

        Learning in MDPs with highly complex state representations is currently possible due to multiple advancements in reinforcement learning algorithm design. However, this incline in complexity, and furthermore the increase in the dimensions of the observation came at the cost of volatility that can be taken advantage of via adversarial attacks (i.e. moving along worst-case directions in the observation space). To solve this policy instability problem we propose a novel method to detect the presence of these non-robust directions via local quadratic approximation of the deep neural policy loss. Our method provides a theoretical basis for the fundamental cut-off between safe observations and adversarial observations. Furthermore, our technique is computationally efficient, and does not depend on the methods used to produce the worst-case directions. We conduct extensive experiments in the Arcade Learning Environment with several different adversarial attack techniques. Most significantly, we demonstrate the effectiveness of our approach even in the setting where non-robust directions are explicitly optimized to circumvent our proposed method.

        ----

        ## [723] Ewald-based Long-Range Message Passing for Molecular Graphs

        **Authors**: *Arthur Kosmala, Johannes Gasteiger, Nicholas Gao, Stephan Günnemann*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/kosmala23a.html](https://proceedings.mlr.press/v202/kosmala23a.html)

        **Abstract**:

        Neural architectures that learn potential energy surfaces from molecular data have undergone fast improvement in recent years. A key driver of this success is the Message Passing Neural Network (MPNN) paradigm. Its favorable scaling with system size partly relies upon a spatial distance limit on messages. While this focus on locality is a useful inductive bias, it also impedes the learning of long-range interactions such as electrostatics and van der Waals forces. To address this drawback, we propose Ewald message passing: a nonlocal Fourier space scheme which limits interactions via a cutoff on frequency instead of distance, and is theoretically well-founded in the Ewald summation method. It can serve as an augmentation on top of existing MPNN architectures as it is computationally inexpensive and agnostic to architectural details. We test the approach with four baseline models and two datasets containing diverse periodic (OC20) and aperiodic structures (OE62). Across all models and datasets, we observe robust improvements in energy mean absolute errors, averaging 10% on OC20 and 16% on OE62. Our analysis shows an outsize impact of these improvements on structures with high long-range contributions to the ground-truth energy.

        ----

        ## [724] TabDDPM: Modelling Tabular Data with Diffusion Models

        **Authors**: *Akim Kotelnikov, Dmitry Baranchuk, Ivan Rubachev, Artem Babenko*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/kotelnikov23a.html](https://proceedings.mlr.press/v202/kotelnikov23a.html)

        **Abstract**:

        Denoising diffusion probabilistic models are becoming the leading generative modeling paradigm for many important data modalities. Being the most prevalent in the computer vision community, diffusion models have recently gained some attention in other domains, including speech, NLP, and graph-like data. In this work, we investigate if the framework of diffusion models can be advantageous for general tabular problems, where data points are typically represented by vectors of heterogeneous features. The inherent heterogeneity of tabular data makes it quite challenging for accurate modeling since the individual features can be of a completely different nature, i.e., some of them can be continuous and some can be discrete. To address such data types, we introduce TabDDPM — a diffusion model that can be universally applied to any tabular dataset and handles any feature types. We extensively evaluate TabDDPM on a wide set of benchmarks and demonstrate its superiority over existing GAN/VAE alternatives, which is consistent with the advantage of diffusion models in other fields.

        ----

        ## [725] Randomized Schur Complement Views for Graph Contrastive Learning

        **Authors**: *Vignesh Kothapalli*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/kothapalli23a.html](https://proceedings.mlr.press/v202/kothapalli23a.html)

        **Abstract**:

        We introduce a randomized topological augmentor based on Schur complements for Graph Contrastive Learning (GCL). Given a graph laplacian matrix, the technique generates unbiased approximations of its Schur complements and treats the corresponding graphs as augmented views. We discuss the benefits of our approach, provide theoretical justifications and present connections with graph diffusion. Unlike previous efforts, we study the empirical effectiveness of the augmentor in a controlled fashion by varying the design choices for subsequent GCL phases, such as encoding and contrasting. Extensive experiments on node and graph classification benchmarks demonstrate that our technique consistently outperforms pre-defined and adaptive augmentation approaches to achieve state-of-the-art results.

        ----

        ## [726] Benign Overfitting in Two-layer ReLU Convolutional Neural Networks

        **Authors**: *Yiwen Kou, Zixiang Chen, Yuanzhou Chen, Quanquan Gu*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/kou23a.html](https://proceedings.mlr.press/v202/kou23a.html)

        **Abstract**:

        Modern deep learning models with great expressive power can be trained to overfit the training data but still generalize well. This phenomenon is referred to as benign overfitting. Recently, a few studies have attempted to theoretically understand benign overfitting in neural networks. However, these works are either limited to neural networks with smooth activation functions or to the neural tangent kernel regime. How and when benign overfitting can occur in ReLU neural networks remains an open problem. In this work, we seek to answer this question by establishing algorithm-dependent risk bounds for learning two-layer ReLU convolutional neural networks with label-flipping noise. We show that, under mild conditions, the neural network trained by gradient descent can achieve near-zero training loss and Bayes optimal test risk. Our result also reveals a sharp transition between benign and harmful overfitting under different conditions on data distribution in terms of test risk. Experiments on synthetic data back up our theory.

        ----

        ## [727] Variational Mixture of HyperGenerators for Learning Distributions over Functions

        **Authors**: *Batuhan Koyuncu, Pablo Sánchez-Martín, Ignacio Peis, Pablo M. Olmos, Isabel Valera*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/koyuncu23a.html](https://proceedings.mlr.press/v202/koyuncu23a.html)

        **Abstract**:

        Recent approaches build on implicit neural representations (INRs) to propose generative models over function spaces. However, they are computationally costly when dealing with inference tasks, such as missing data imputation, or directly cannot tackle them. In this work, we propose a novel deep generative model, named VaMoH. VaMoH combines the capabilities of modeling continuous functions using INRs and the inference capabilities of Variational Autoencoders (VAEs). In addition, VaMoH relies on a normalizing flow to define the prior, and a mixture of hypernetworks to parametrize the data log-likelihood. This gives VaMoH a high expressive capability and interpretability. Through experiments on a diverse range of data types, such as images, voxels, and climate data, we show that VaMoH can effectively learn rich distributions over continuous functions. Furthermore, it can perform inference-related tasks, such as conditional super-resolution generation and in-painting, as well or better than previous approaches, while being less computationally demanding.

        ----

        ## [728] Gradient Descent Monotonically Decreases the Sharpness of Gradient Flow Solutions in Scalar Networks and Beyond

        **Authors**: *Itai Kreisler, Mor Shpigel Nacson, Daniel Soudry, Yair Carmon*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/kreisler23a.html](https://proceedings.mlr.press/v202/kreisler23a.html)

        **Abstract**:

        Recent research shows that when Gradient Descent (GD) is applied to neural networks, the loss almost never decreases monotonically. Instead, the loss oscillates as gradient descent converges to its “Edge of Stability” (EoS). Here, we find a quantity that does decrease monotonically throughout GD training: the sharpness attained by the gradient flow solution (GFS)—the solution that would be obtained if, from now until convergence, we train with an infinitesimal step size. Theoretically, we analyze scalar neural networks with the squared loss, perhaps the simplest setting where the EoS phenomena still occur. In this model, we prove that the GFS sharpness decreases monotonically. Using this result, we characterize settings where GD provably converges to the EoS in scalar networks. Empirically, we show that GD monotonically decreases the GFS sharpness in a squared regression model as well as practical neural network architectures.

        ----

        ## [729] Estimation Beyond Data Reweighting: Kernel Method of Moments

        **Authors**: *Heiner Kremer, Yassine Nemmour, Bernhard Schölkopf, Jia-Jie Zhu*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/kremer23a.html](https://proceedings.mlr.press/v202/kremer23a.html)

        **Abstract**:

        Moment restrictions and their conditional counterparts emerge in many areas of machine learning and statistics ranging from causal inference to reinforcement learning. Estimators for these tasks, generally called methods of moments, include the prominent generalized method of moments (GMM) which has recently gained attention in causal inference. GMM is a special case of the broader family of empirical likelihood estimators which are based on approximating a population distribution by means of minimizing a $\varphi$-divergence to an empirical distribution. However, the use of $\varphi$-divergences effectively limits the candidate distributions to reweightings of the data samples. We lift this long-standing limitation and provide a method of moments that goes beyond data reweighting. This is achieved by defining an empirical likelihood estimator based on maximum mean discrepancy which we term the kernel method of moments (KMM). We provide a variant of our estimator for conditional moment restrictions and show that it is asymptotically first-order optimal for such problems. Finally, we show that our method achieves competitive performance on several conditional moment restriction tasks.

        ----

        ## [730] Multi-Task Differential Privacy Under Distribution Skew

        **Authors**: *Walid Krichene, Prateek Jain, Shuang Song, Mukund Sundararajan, Abhradeep Guha Thakurta, Li Zhang*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/krichene23a.html](https://proceedings.mlr.press/v202/krichene23a.html)

        **Abstract**:

        We study the problem of multi-task learning under user-level differential privacy, in which n users contribute data to m tasks, each involving a subset of users. One important aspect of the problem, that can significantly impact quality, is the distribution skew among tasks. Tasks that have much fewer data samples than others are more susceptible to the noise added for privacy. It is natural to ask whether algorithms can adapt to this skew to improve the overall utility. We give a systematic analysis of the problem, by studying how to optimally allocate a user’s privacy budget among tasks. We propose a generic algorithm, based on an adaptive reweighting of the empirical loss, and show that in the presence of distribution skew, this gives a quantifiable improvement of excess empirical risk. Experimental studies on recommendation problems that exhibit a long tail of small tasks, demonstrate that our methods significantly improve utility, achieving the state of the art on two standard benchmarks.

        ----

        ## [731] Towards Bridging the Gaps between the Right to Explanation and the Right to be Forgotten

        **Authors**: *Satyapriya Krishna, Jiaqi Ma, Himabindu Lakkaraju*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/krishna23a.html](https://proceedings.mlr.press/v202/krishna23a.html)

        **Abstract**:

        The Right to Explanation and the Right to be Forgotten are two important principles outlined to regulate algorithmic decision making and data usage in real-world applications. While the right to explanation allows individuals to request an actionable explanation for an algorithmic decision, the right to be forgotten grants them the right to ask for their data to be deleted from all the databases and models of an organization. Intuitively, enforcing the right to be forgotten may trigger model updates which in turn invalidate previously provided explanations, thus violating the right to explanation. In this work, we investigate the technical implications arising due to the interference between the two aforementioned regulatory principles, and propose the first algorithmic framework to resolve the tension between them. To this end, we formulate a novel optimization problem to generate explanations that are robust to model updates due to the removal of training data instances by data deletion requests. We then derive an efficient approximation algorithm to handle the combinatorial complexity of this optimization problem. We theoretically demonstrate that our method generates explanations that are provably robust to worst-case data deletion requests with bounded costs in case of linear models and certain classes of non-linear models. Extensive experimentation with real-world datasets demonstrates the efficacy of the proposed framework.

        ----

        ## [732] Graph Neural Tangent Kernel: Convergence on Large Graphs

        **Authors**: *Sanjukta Krishnagopal, Luana Ruiz*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/krishnagopal23a.html](https://proceedings.mlr.press/v202/krishnagopal23a.html)

        **Abstract**:

        Graph neural networks (GNNs) achieve remarkable performance in graph machine learning tasks but can be hard to train on large-graph data, where their learning dynamics are not well understood. We investigate the training dynamics of large-graph GNNs using graph neural tangent kernels (GNTKs) and graphons. In the limit of large width, optimization of an overparametrized NN is equivalent to kernel regression on the NTK. Here, we investigate how the GNTK evolves as another independent dimension is varied: the graph size. We use graphons to define limit objects—graphon NNs for GNNs, and graphon NTKs for GNTKs—, and prove that, on a sequence of graphs, the GNTKs converge to the graphon NTK. We further prove that the spectrum of the GNTK, which is related to the problem’s learning directions, converges to the spectrum of the GNTK. This implies that in the large-graph limit, the GNTK fitted on a graph of moderate size can be used to solve the same task on the large graph, and to infer the learning dynamics of the large-graph GNN. These results are verified empirically on node regression and classification tasks.

        ----

        ## [733] Diffusion Models for Black-Box Optimization

        **Authors**: *Siddarth Krishnamoorthy, Satvik Mehul Mashkaria, Aditya Grover*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/krishnamoorthy23a.html](https://proceedings.mlr.press/v202/krishnamoorthy23a.html)

        **Abstract**:

        The goal of offline black-box optimization (BBO) is to optimize an expensive black-box function using a fixed dataset of function evaluations. Prior works consider forward approaches that learn surrogates to the black-box function and inverse approaches that directly map function values to corresponding points in the input domain of the black-box function. These approaches are limited by the quality of the offline dataset and the difficulty in learning one-to-many mappings in high dimensions, respectively. We propose Denoising Diffusion Optimization Models (DDOM), a new inverse approach for offline black-box optimization based on diffusion models. Given an offline dataset, DDOM learns a conditional generative model over the domain of the black-box function conditioned on the function values. We investigate several design choices in DDOM, such as reweighting the dataset to focus on high function values and the use of classifier-free guidance at test-time to enable generalization to function values that can even exceed the dataset maxima. Empirically, we conduct experiments on the Design-Bench benchmark (Trabucco et al., 2022) and show that DDOM achieves results competitive with state-of-the-art baselines.

        ----

        ## [734] Learning to Design Analog Circuits to Meet Threshold Specifications

        **Authors**: *Dmitrii Krylov, Pooya Khajeh, Junhan Ouyang, Thomas Reeves, Tongkai Liu, Hiba Ajmal, Hamidreza Aghasi, Roy Fox*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/krylov23a.html](https://proceedings.mlr.press/v202/krylov23a.html)

        **Abstract**:

        Automated design of analog and radio-frequency circuits using supervised or reinforcement learning from simulation data has recently been studied as an alternative to manual expert design. It is straightforward for a design agent to learn an inverse function from desired performance metrics to circuit parameters. However, it is more common for a user to have threshold performance criteria rather than an exact target vector of feasible performance measures. In this work, we propose a method for generating from simulation data a dataset on which a system can be trained via supervised learning to design circuits to meet threshold specifications. We moreover perform the to-date most extensive evaluation of automated analog circuit design, including experimenting in a significantly more diverse set of circuits than in prior work, covering linear, nonlinear, and autonomous circuit configurations, and show that our method consistently reaches success rate better than 90% at 5% error margin, while also improving data efficiency by upward of an order of magnitude.

        ----

        ## [735] Variance Control for Distributional Reinforcement Learning

        **Authors**: *Qi Kuang, Zhoufan Zhu, Liwen Zhang, Fan Zhou*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/kuang23a.html](https://proceedings.mlr.press/v202/kuang23a.html)

        **Abstract**:

        Although distributional reinforcement learning (DRL) has been widely examined in the past few years, very few studies investigate the validity of the obtained Q-function estimator in the distributional setting. To fully understand how the approximation errors of the Q-function affect the whole training process, we do some error analysis and theoretically show how to reduce both the bias and the variance of the error terms. With this new understanding, we construct a new estimator Quantiled Expansion Mean (QEM) and introduce a new DRL algorithm (QEMRL) from the statistical perspective. We extensively evaluate our QEMRL algorithm on a variety of Atari and Mujoco benchmark tasks and demonstrate that QEMRL achieves significant improvement over baseline algorithms in terms of sample efficiency and convergence performance.

        ----

        ## [736] Hierarchical Imitation Learning with Vector Quantized Models

        **Authors**: *Kalle Kujanpää, Joni Pajarinen, Alexander Ilin*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/kujanpaa23a.html](https://proceedings.mlr.press/v202/kujanpaa23a.html)

        **Abstract**:

        The ability to plan actions on multiple levels of abstraction enables intelligent agents to solve complex tasks effectively. However, learning the models for both low and high-level planning from demonstrations has proven challenging, especially with higher-dimensional inputs. To address this issue, we propose to use reinforcement learning to identify subgoals in expert trajectories by associating the magnitude of the rewards with the predictability of low-level actions given the state and the chosen subgoal. We build a vector-quantized generative model for the identified subgoals to perform subgoal-level planning. In experiments, the algorithm excels at solving complex, long-horizon decision-making problems outperforming state-of-the-art. Because of its ability to plan, our algorithm can find better trajectories than the ones in the training set.

        ----

        ## [737] SinDDM: A Single Image Denoising Diffusion Model

        **Authors**: *Vladimir Kulikov, Shahar Yadin, Matan Kleiner, Tomer Michaeli*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/kulikov23a.html](https://proceedings.mlr.press/v202/kulikov23a.html)

        **Abstract**:

        Denoising diffusion models (DDMs) have led to staggering performance leaps in image generation, editing and restoration. However, existing DDMs use very large datasets for training. Here, we introduce a framework for training a DDM on a single image. Our method, which we coin SinDDM, learns the internal statistics of the training image by using a multi-scale diffusion process. To drive the reverse diffusion process, we use a fully-convolutional light-weight denoiser, which is conditioned on both the noise level and the scale. This architecture allows generating samples of arbitrary dimensions, in a coarse-to-fine manner. As we illustrate, SinDDM generates diverse high-quality samples, and is applicable in a wide array of tasks, including style transfer and harmonization. Furthermore, it can be easily guided by external supervision. Particularly, we demonstrate text-guided generation from a single image using a pre-trained CLIP model.

        ----

        ## [738] Towards Explaining Distribution Shifts

        **Authors**: *Sean Kulinski, David I. Inouye*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/kulinski23a.html](https://proceedings.mlr.press/v202/kulinski23a.html)

        **Abstract**:

        A distribution shift can have fundamental consequences such as signaling a change in the operating environment or significantly reducing the accuracy of downstream models. Thus, understanding distribution shifts is critical for examining and hopefully mitigating the effect of such a shift. Most prior work has focused on merely detecting if a shift has occurred and assumes any detected shift can be understood and handled appropriately by a human operator. We hope to aid in these manual mitigation tasks by explaining the distribution shift using interpretable transportation maps from the original distribution to the shifted one. We derive our interpretable mappings from a relaxation of the optimal transport problem, where the candidate mappings are restricted to a set of interpretable mappings. We then use a wide array of quintessential examples of distribution shift in real-world tabular, text, and image cases to showcase how our explanatory mappings provide a better balance between detail and interpretability than baseline explanations by both visual inspection and our PercentExplained metric.

        ----

        ## [739] Featured Graph Coarsening with Similarity Guarantees

        **Authors**: *Manoj Kumar, Anurag Sharma, Shashwat Saxena, Sandeep Kumar*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/kumar23a.html](https://proceedings.mlr.press/v202/kumar23a.html)

        **Abstract**:

        Graph coarsening is a dimensionality reduction technique that aims to learn a smaller-tractable graph while preserving the properties of the original input graph. However, many real-world graphs also have features or contexts associated with each node. The existing graph coarsening methods do not consider the node features and rely solely on a graph matrix(e.g., adjacency and Laplacian) to coarsen graphs. However, some recent deep learning-based graph coarsening methods are designed for specific tasks considering both node features and graph matrix. In this paper, we introduce a novel optimization-based framework for graph coarsening that takes both the graph matrix and the node features as the input and jointly learns the coarsened graph matrix and the coarsened feature matrix while ensuring desired properties. To the best of our knowledge, this is the first work that guarantees that the learned coarsened graph is $\epsilon\in[0,1)$ similar to the original graph. Extensive experiments with both real and synthetic benchmark datasets elucidate the proposed framework’s efficacy and applicability for numerous graph-based applications, including graph clustering, node classification, stochastic block model identification, and graph summarization.

        ----

        ## [740] Modeling Dynamic Environments with Scene Graph Memory

        **Authors**: *Andrey Kurenkov, Michael Lingelbach, Tanmay Agarwal, Emily Jin, Chengshu Li, Ruohan Zhang, Li Fei-Fei, Jiajun Wu, Silvio Savarese, Roberto Martín-Martín*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/kurenkov23a.html](https://proceedings.mlr.press/v202/kurenkov23a.html)

        **Abstract**:

        Embodied AI agents that search for objects in large environments such as households often need to make efficient decisions by predicting object locations based on partial information. We pose this as a new type of link prediction problem: link prediction on partially observable dynamic graphs Our graph is a representation of a scene in which rooms and objects are nodes, and their relationships are encoded in the edges; only parts of the changing graph are known to the agent at each timestep. This partial observability poses a challenge to existing link prediction approaches, which we address. We propose a novel state representation – Scene Graph Memory (SGM) – with captures the agent’s accumulated set of observations, as well as a neural net architecture called a Node Edge Predictor (NEP) that extracts information from the SGM to search efficiently. We evaluate our method in the Dynamic House Simulator, a new benchmark that creates diverse dynamic graphs following the semantic patterns typically seen at homes, and show that NEP can be trained to predict the locations of objects in a variety of environments with diverse object movement dynamics, outperforming baselines both in terms of new scene adaptability and overall accuracy. The codebase and more can be found www.scenegraphmemory.com.

        ----

        ## [741] Tied-Augment: Controlling Representation Similarity Improves Data Augmentation

        **Authors**: *Emirhan Kurtulus, Zichao Li, Yann N. Dauphin, Ekin Dogus Cubuk*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/kurtulus23a.html](https://proceedings.mlr.press/v202/kurtulus23a.html)

        **Abstract**:

        Data augmentation methods have played an important role in the recent advance of deep learning models, and have become an indispensable component of state-of-the-art models in semi-supervised, self-supervised, and supervised training for vision. Despite incurring no additional latency at test time, data augmentation often requires more epochs of training to be effective. For example, even the simple flips-and-crops augmentation requires training for more than 5 epochs to improve performance, whereas RandAugment requires more than 90 epochs. We propose a general framework called Tied-Augment, which improves the efficacy of data augmentation in a wide range of applications by adding a simple term to the loss that can control the similarity of representations under distortions. Tied-Augment can improve state-of-the-art methods from data augmentation (e.g. RandAugment, mixup), optimization (e.g. SAM), and semi-supervised learning (e.g. FixMatch). For example, Tied-RandAugment can outperform RandAugment by 2.0% on ImageNet. Notably, using Tied-Augment, data augmentation can be made to improve generalization even when training for a few epochs and when fine-tuning. We open source our code at https://github.com/ekurtulus/tied-augment/tree/main.

        ----

        ## [742] Cooperation in the Latent Space: The Benefits of Adding Mixture Components in Variational Autoencoders

        **Authors**: *Oskar Kviman, Ricky Molén, Alexandra Hotti, Semih Kurt, Víctor Elvira, Jens Lagergren*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/kviman23a.html](https://proceedings.mlr.press/v202/kviman23a.html)

        **Abstract**:

        In this paper, we show how the mixture components cooperate when they jointly adapt to maximize the ELBO. We build upon recent advances in the multiple and adaptive importance sampling literature. We then model the mixture components using separate encoder networks and show empirically that the ELBO is monotonically non-decreasing as a function of the number of mixture components. These results hold for a range of different VAE architectures on the MNIST, FashionMNIST, and CIFAR-10 datasets. In this work, we also demonstrate that increasing the number of mixture components improves the latent-representation capabilities of the VAE on both image and single-cell datasets. This cooperative behavior motivates that using Mixture VAEs should be considered a standard approach for obtaining more flexible variational approximations. Finally, Mixture VAEs are here, for the first time, compared and combined with normalizing flows, hierarchical models and/or the VampPrior in an extensive ablation study. Multiple of our Mixture VAEs achieve state-of-the-art log-likelihood results for VAE architectures on the MNIST and FashionMNIST datasets. The experiments are reproducible using our code, provided https://github.com/Lagergren-Lab/MixtureVAEs.

        ----

        ## [743] GeCoNeRF: Few-shot Neural Radiance Fields via Geometric Consistency

        **Authors**: *Minseop Kwak, Jiuhn Song, Seungryong Kim*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/kwak23a.html](https://proceedings.mlr.press/v202/kwak23a.html)

        **Abstract**:

        We present a novel framework to regularize Neural Radiance Field (NeRF) in a few-shot setting with a geometry-aware consistency regularization. The proposed approach leverages a rendered depth map at unobserved viewpoint to warp sparse input images to the unobserved viewpoint and impose them as pseudo ground truths to facilitate learning of NeRF. By encouraging such geometry-aware consistency at a feature-level instead of using pixel-level reconstruction loss, we regularize the NeRF at semantic and structural levels while allowing for modeling view dependent radiance to account for color variations across viewpoints. We also propose an effective method to filter out erroneous warped solutions, along with training strategies to stabilize training during optimization. We show that our model achieves competitive results compared to state-of-the-art few-shot NeRF models.

        ----

        ## [744] Rotation and Translation Invariant Representation Learning with Implicit Neural Representations

        **Authors**: *Sehyun Kwon, Joo Young Choi, Ernest K. Ryu*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/kwon23a.html](https://proceedings.mlr.press/v202/kwon23a.html)

        **Abstract**:

        In many computer vision applications, images are acquired with arbitrary or random rotations and translations, and in such setups, it is desirable to obtain semantic representations disentangled from the image orientation. Examples of such applications include semiconductor wafer defect inspection, plankton microscope images, and inference on single-particle cryo-electron microscopy (cryo-EM) micro-graphs. In this work, we propose Invariant Representation Learning with Implicit Neural Representation (IRL-INR), which uses an implicit neural representation (INR) with a hypernetwork to obtain semantic representations disentangled from the orientation of the image. We show that IRL-INR can effectively learn disentangled semantic representations on more complex images compared to those considered in prior works and show that these semantic representations synergize well with SCAN to produce state-of-the-art unsupervised clustering results.

        ----

        ## [745] Reward-Mixing MDPs with Few Latent Contexts are Learnable

        **Authors**: *Jeongyeol Kwon, Yonathan Efroni, Constantine Caramanis, Shie Mannor*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/kwon23b.html](https://proceedings.mlr.press/v202/kwon23b.html)

        **Abstract**:

        We consider episodic reinforcement learning in reward-mixing Markov decision processes (RMMDPs): at the beginning of every episode nature randomly picks a latent reward model among $M$ candidates and an agent interacts with the MDP throughout the episode for $H$ time steps. Our goal is to learn a near-optimal policy that nearly maximizes the $H$ time-step cumulative rewards in such a model. Prior work established an upper bound for RMMDPs with $M=2$. In this work, we resolve several open questions for the general RMMDP setting. We consider an arbitrary $M\ge2$ and provide a sample-efficient algorithm–$EM^2$–that outputs an $\epsilon$-optimal policy using $O \left(\epsilon^{-2} \cdot S^d A^d \cdot \text{poly}(H, Z)^d \right)$ episodes, where $S, A$ are the number of states and actions respectively, $H$ is the time-horizon, $Z$ is the support size of reward distributions and $d=O(\min(M,H))$. We also provide a $(SA)^{\Omega(\sqrt{M})} / \epsilon^{2}$ lower bound, supporting that super-polynomial sample complexity in $M$ is necessary.

        ----

        ## [746] A Fully First-Order Method for Stochastic Bilevel Optimization

        **Authors**: *Jeongyeol Kwon, Dohyun Kwon, Stephen Wright, Robert D. Nowak*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/kwon23c.html](https://proceedings.mlr.press/v202/kwon23c.html)

        **Abstract**:

        We consider stochastic unconstrained bilevel optimization problems when only the first-order gradient oracles are available. While numerous optimization methods have been proposed for tackling bilevel problems, existing methods either tend to require possibly expensive calculations regarding Hessians of lower-level objectives, or lack rigorous finite-time performance guarantees. In this work, we propose a Fully First-order Stochastic Approximation (F2SA) method, and study its non-asymptotic convergence properties. Specifically, we show that F2SA converges to an $\epsilon$-stationary solution of the bilevel problem after $\epsilon^{-7/2}, \epsilon^{-5/2}$, and $\epsilon^{-3/2}$ iterations (each iteration using $O(1)$ samples) when stochastic noises are in both level objectives, only in the upper-level objective, and not present (deterministic settings), respectively. We further show that if we employ momentum-assisted gradient estimators, the iteration complexities can be improved to $\epsilon^{-5/2}, \epsilon^{-4/2}$, and $\epsilon^{-3/2}$, respectively. We demonstrate even superior practical performance of the proposed method over existing second-order based approaches on MNIST data-hypercleaning experiments.

        ----

        ## [747] Complexity of Block Coordinate Descent with Proximal Regularization and Applications to Wasserstein CP-dictionary Learning

        **Authors**: *Dohyun Kwon, Hanbaek Lyu*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/kwon23d.html](https://proceedings.mlr.press/v202/kwon23d.html)

        **Abstract**:

        We consider the block coordinate descent methods of Gauss-Seidel type with proximal regularization (BCD-PR), which is a classical method of minimizing general nonconvex objectives under constraints that has a wide range of practical applications. We theoretically establish the worst-case complexity bound for this algorithm. Namely, we show that for general nonconvex smooth objectives with block-wise constraints, the classical BCD-PR algorithm converges to an epsilon-stationary point within O(1/epsilon) iterations. Under a mild condition, this result still holds even if the algorithm is executed inexactly in each step. As an application, we propose a provable and efficient algorithm for ‘Wasserstein CP-dictionary learning’, which seeks a set of elementary probability distributions that can well-approximate a given set of d-dimensional joint probability distributions. Our algorithm is a version of BCD-PR that operates in the dual space, where the primal problem is regularized both entropically and proximally.

        ----

        ## [748] Data-OOB: Out-of-bag Estimate as a Simple and Efficient Data Value

        **Authors**: *Yongchan Kwon, James Zou*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/kwon23e.html](https://proceedings.mlr.press/v202/kwon23e.html)

        **Abstract**:

        Data valuation is a powerful framework for providing statistical insights into which data are beneficial or detrimental to model training. Many Shapley-based data valuation methods have shown promising results in various downstream tasks, however, they are well known to be computationally challenging as it requires training a large number of models. As a result, it has been recognized as infeasible to apply to large datasets. To address this issue, we propose Data-OOB, a new data valuation method for a bagging model that utilizes the out-of-bag estimate. The proposed method is computationally efficient and can scale to millions of data by reusing trained weak learners. Specifically, Data-OOB takes less than $2.25$ hours on a single CPU processor when there are $10^6$ samples to evaluate and the input dimension is $100$. Furthermore, Data-OOB has solid theoretical interpretations in that it identifies the same important data point as the infinitesimal jackknife influence function when two different points are compared. We conduct comprehensive experiments using 12 classification datasets, each with thousands of sample sizes. We demonstrate that the proposed method significantly outperforms existing state-of-the-art data valuation methods in identifying mislabeled data and finding a set of helpful (or harmful) data points, highlighting the potential for applying data values in real-world applications.

        ----

        ## [749] Emergence of Adaptive Circadian Rhythms in Deep Reinforcement Learning

        **Authors**: *Aqeel Labash, Florian Stelzer, Daniel Majoral, Raul Vicente Zafra*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/labash23a.html](https://proceedings.mlr.press/v202/labash23a.html)

        **Abstract**:

        Adapting to regularities of the environment is critical for biological organisms to anticipate events and plan. A prominent example is the circadian rhythm corresponding to the internalization by organisms of the $24$-hour period of the Earth’s rotation. In this work, we study the emergence of circadian-like rhythms in deep reinforcement learning agents. In particular, we deployed agents in an environment with a reliable periodic variation while solving a foraging task. We systematically characterize the agent’s behavior during learning and demonstrate the emergence of a rhythm that is endogenous and entrainable. Interestingly, the internal rhythm adapts to shifts in the phase of the environmental signal without any re-training. Furthermore, we show via bifurcation and phase response curve analyses how artificial neurons develop dynamics to support the internalization of the environmental rhythm. From a dynamical systems view, we demonstrate that the adaptation proceeds by the emergence of a stable periodic orbit in the neuron dynamics with a phase response that allows an optimal phase synchronisation between the agent’s dynamics and the environmental rhythm.

        ----

        ## [750] Synergies between Disentanglement and Sparsity: Generalization and Identifiability in Multi-Task Learning

        **Authors**: *Sébastien Lachapelle, Tristan Deleu, Divyat Mahajan, Ioannis Mitliagkas, Yoshua Bengio, Simon Lacoste-Julien, Quentin Bertrand*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/lachapelle23a.html](https://proceedings.mlr.press/v202/lachapelle23a.html)

        **Abstract**:

        Although disentangled representations are often said to be beneficial for downstream tasks, current empirical and theoretical understanding is limited. In this work, we provide evidence that disentangled representations coupled with sparse task-specific predictors improve generalization. In the context of multi-task learning, we prove a new identifiability result that provides conditions under which maximally sparse predictors yield disentangled representations. Motivated by this theoretical result, we propose a practical approach to learn disentangled representations based on a sparsity-promoting bi-level optimization problem. Finally, we explore a meta-learning version of this algorithm based on group Lasso multiclass SVM predictors, for which we derive a tractable dual formulation. It obtains competitive results on standard few-shot classification benchmarks, while each task is using only a fraction of the learned representations.

        ----

        ## [751] Nearly-Optimal Hierarchical Clustering for Well-Clustered Graphs

        **Authors**: *Steinar Laenen, Bogdan-Adrian Manghiuc, He Sun*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/laenen23a.html](https://proceedings.mlr.press/v202/laenen23a.html)

        **Abstract**:

        This paper presents two efficient hierarchical clustering (HC) algorithms with respect to Dasgupta’s cost function. For any input graph $G$ with a clear cluster-structure, our designed algorithms run in nearly-linear time in the input size of $G$, and return an $O(1)$-approximate HC tree with respect to Dasgupta’s cost function. We compare the performance of our algorithm against the previous state-of-the-art on synthetic and real-world datasets and show that our designed algorithm produces comparable or better HC trees with much lower running time.

        ----

        ## [752] Hybrid Energy Based Model in the Feature Space for Out-of-Distribution Detection

        **Authors**: *Marc Lafon, Elias Ramzi, Clément Rambour, Nicolas Thome*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/lafon23a.html](https://proceedings.mlr.press/v202/lafon23a.html)

        **Abstract**:

        Out-of-distribution (OOD) detection is a critical requirement for the deployment of deep neural networks. This paper introduces the HEAT model, a new post-hoc OOD detection method estimating the density of in-distribution (ID) samples using hybrid energy-based models (EBM) in the feature space of a pre-trained backbone. HEAT complements prior density estimators of the ID density, e.g. parametric models like the Gaussian Mixture Model (GMM), to provide an accurate yet robust density estimation. A second contribution is to leverage the EBM framework to provide a unified density estimation and to compose several energy terms. Extensive experiments demonstrate the significance of the two contributions. HEAT sets new state-of-the-art OOD detection results on the CIFAR-10 / CIFAR-100 benchmark as well as on the large-scale Imagenet benchmark. The code is available at: https://github.com/MarcLafon/heatood.

        ----

        ## [753] A theory of continuous generative flow networks

        **Authors**: *Salem Lahlou, Tristan Deleu, Pablo Lemos, Dinghuai Zhang, Alexandra Volokhova, Alex Hernández-García, Léna Néhale Ezzine, Yoshua Bengio, Nikolay Malkin*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/lahlou23a.html](https://proceedings.mlr.press/v202/lahlou23a.html)

        **Abstract**:

        Generative flow networks (GFlowNets) are amortized variational inference algorithms that are trained to sample from unnormalized target distributions over compositional objects. A key limitation of GFlowNets until this time has been that they are restricted to discrete spaces. We present a theory for generalized GFlowNets, which encompasses both existing discrete GFlowNets and ones with continuous or hybrid state spaces, and perform experiments with two goals in mind. First, we illustrate critical points of the theory and the importance of various assumptions. Second, we empirically demonstrate how observations about discrete GFlowNets transfer to the continuous case and show strong results compared to non-GFlowNet baselines on several previously studied tasks. This work greatly widens the perspectives for the application of GFlowNets in probabilistic inference and various modeling settings.

        ----

        ## [754] Automatically marginalized MCMC in probabilistic programming

        **Authors**: *Jinlin Lai, Javier Burroni, Hui Guan, Daniel Sheldon*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/lai23a.html](https://proceedings.mlr.press/v202/lai23a.html)

        **Abstract**:

        Hamiltonian Monte Carlo (HMC) is a powerful algorithm to sample latent variables from Bayesian models. The advent of probabilistic programming languages (PPLs) frees users from writing inference algorithms and lets users focus on modeling. However, many models are difficult for HMC to solve directly, and often require tricks like model reparameterization. We are motivated by the fact that many of those models could be simplified by marginalization. We propose to use automatic marginalization as part of the sampling process using HMC in a graphical model extracted from a PPL, which substantially improves sampling from real-world hierarchical models.

        ----

        ## [755] DS-1000: A Natural and Reliable Benchmark for Data Science Code Generation

        **Authors**: *Yuhang Lai, Chengxi Li, Yiming Wang, Tianyi Zhang, Ruiqi Zhong, Luke Zettlemoyer, Wen-Tau Yih, Daniel Fried, Sida I. Wang, Tao Yu*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/lai23b.html](https://proceedings.mlr.press/v202/lai23b.html)

        **Abstract**:

        We introduce DS-1000, a code generation benchmark with a thousand data science problems spanning seven Python libraries, such as Numpy and Pandas. Compared to prior works, DS-1000 incorporates three core features. First, our problems reflect diverse, realistic, and practical use cases since we collected them from StackOverflow. Second, our automatic evaluation is highly specific (reliable) – across all Codex-002-predicted solutions that our evaluation accepts, only 1.8% of them are incorrect; we achieve this with multi-criteria metrics, checking both functional correctness by running test cases and surface-form constraints by restricting API usages or keywords. Finally, we proactively defend against memorization by slightly modifying our problems to be different from the original StackOverflow source; consequently, models cannot answer them correctly by memorizing the solutions from pre-training. The current best public system (Codex-002) achieves 43.3% accuracy, leaving ample room for improvement. We release our benchmark at https://ds1000-code-gen.github.io.

        ----

        ## [756] ChiPFormer: Transferable Chip Placement via Offline Decision Transformer

        **Authors**: *Yao Lai, Jinxin Liu, Zhentao Tang, Bin Wang, Jianye Hao, Ping Luo*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/lai23c.html](https://proceedings.mlr.press/v202/lai23c.html)

        **Abstract**:

        Placement is a critical step in modern chip design, aiming to determine the positions of circuit modules on the chip canvas. Recent works have shown that reinforcement learning (RL) can improve human performance in chip placement. However, such an RL-based approach suffers from long training time and low transfer ability in unseen chip circuits. To resolve these challenges, we cast the chip placement as an offline RL formulation and present ChiPFormer that enables learning a transferable placement policy from fixed offline data. ChiPFormer has several advantages that prior arts do not have. First, ChiPFormer can exploit offline placement designs to learn transferable policies more efficiently in a multi-task setting. Second, ChiPFormer can promote effective finetuning for unseen chip circuits, reducing the placement runtime from hours to minutes. Third, extensive experiments on 32 chip circuits demonstrate that ChiPFormer achieves significantly better placement quality while reducing the runtime by 10x compared to recent state-of-the-art approaches in both public benchmarks and realistic industrial tasks. The deliverables are released at https://sites.google.com/view/chipformer/home.

        ----

        ## [757] FP-Diffusion: Improving Score-based Diffusion Models by Enforcing the Underlying Score Fokker-Planck Equation

        **Authors**: *Chieh-Hsin Lai, Yuhta Takida, Naoki Murata, Toshimitsu Uesaka, Yuki Mitsufuji, Stefano Ermon*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/lai23d.html](https://proceedings.mlr.press/v202/lai23d.html)

        **Abstract**:

        Score-based generative models (SGMs) learn a family of noise-conditional score functions corresponding to the data density perturbed with increasingly large amounts of noise. These perturbed data densities are linked together by the Fokker-Planck equation (FPE), a partial differential equation (PDE) governing the spatial-temporal evolution of a density undergoing a diffusion process. In this work, we derive a corresponding equation called the score FPE that characterizes the noise-conditional scores of the perturbed data densities (i.e., their gradients). Surprisingly, despite the impressive empirical performance, we observe that scores learned through denoising score matching (DSM) fail to fulfill the underlying score FPE, which is an inherent self-consistency property of the ground truth score. We prove that satisfying the score FPE is desirable as it improves the likelihood and the degree of conservativity. Hence, we propose to regularize the DSM objective to enforce satisfaction of the score FPE, and we show the effectiveness of this approach across various datasets.

        ----

        ## [758] Private Statistical Estimation of Many Quantiles

        **Authors**: *Clément Lalanne, Aurélien Garivier, Rémi Gribonval*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/lalanne23a.html](https://proceedings.mlr.press/v202/lalanne23a.html)

        **Abstract**:

        This work studies the estimation of many statistical quantiles under differential privacy. More precisely, given a distribution and access to i.i.d. samples from it, we study the estimation of the inverse of its cumulative distribution function (the quantile function) at specific points. For instance, this task is of key importance in private data generation. We present two different approaches. The first one consists in privately estimating the empirical quantiles of the samples and using this result as an estimator of the quantiles of the distribution. In particular, we study the statistical properties of the recently published algorithm introduced by (Kaplan et al., 2022) that privately estimates the quantiles recursively. The second approach is to use techniques of density estimation in order to uniformly estimate the quantile function on an interval. In particular, we show that there is a tradeoff between the two methods. When we want to estimate many quantiles, it is better to estimate the density rather than estimating the quantile function at specific points.

        ----

        ## [759] Bootstrap in High Dimension with Low Computation

        **Authors**: *Henry Lam, Zhenyuan Liu*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/lam23a.html](https://proceedings.mlr.press/v202/lam23a.html)

        **Abstract**:

        The bootstrap is a popular data-driven method to quantify statistical uncertainty, but for modern high-dimensional problems, it could suffer from huge computational costs due to the need to repeatedly generate resamples and refit models. We study the use of bootstraps in high-dimensional environments with a small number of resamples. In particular, we show that with a recent "cheap" bootstrap perspective, using a number of resamples as small as one could attain valid coverage even when the dimension grows closely with the sample size, thus strongly supporting the implementability of the bootstrap for large-scale problems. We validate our theoretical results and compare the performance of our approach with other benchmarks via a range of experiments.

        ----

        ## [760] LegendreTron: Uprising Proper Multiclass Loss Learning

        **Authors**: *Kevin H. Lam, Christian J. Walder, Spiridon Penev, Richard Nock*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/lam23b.html](https://proceedings.mlr.press/v202/lam23b.html)

        **Abstract**:

        Loss functions serve as the foundation of supervised learning and are often chosen prior to model development. To avoid potentially ad hoc choices of losses, statistical decision theory describes a desirable property for losses known as properness, which asserts that Bayes’ rule is optimal. Recent works have sought to learn losses and models jointly. Existing methods do this by fitting an inverse canonical link function which monotonically maps $\mathbb{R}$ to $[0,1]$ to estimate probabilities for binary problems. In this paper, we extend monotonicity to maps between $\mathbb{R}^{C-1}$ and the projected probability simplex $\tilde{\Delta}^{C-1}$ by using monotonicity of gradients of convex functions. We present LegendreTron as a novel and practical method that jointly learns proper canonical losses and probabilities for multiclass problems. Tested on a benchmark of domains with up to 1,000 classes, our experimental results show that our method consistently outperforms the natural multiclass baseline under a $t$-test at 99% significance on all datasets with greater than $10$ classes.

        ----

        ## [761] Metagenomic Binning using Connectivity-constrained Variational Autoencoders

        **Authors**: *Andre Lamurias, Alessandro Tibo, Katja Hose, Mads Albertsen, Thomas Dyhre Nielsen*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/lamurias23a.html](https://proceedings.mlr.press/v202/lamurias23a.html)

        **Abstract**:

        Current state-of-the-art techniques for metagenomic binning only utilize local features for the individual DNA sequences (contigs), neglecting additional information such as the assembly graph, in which the contigs are connected according to overlapping reads, and gene markers identified in the contigs. In this paper, we propose the use of a Variational AutoEncoder (VAE) tailored to leverage auxiliary structural information about contig relations when learning contig representations for subsequent metagenomic binning. Our method, CCVAE, improves on previous work that used VAEs for learning latent representations of the individual contigs, by constraining these representations according to the connectivity information from the assembly graph. Additionally, we incorporate into the model additional information in the form of marker genes to better differentiate contigs from different genomes. Our experiments on both simulated and real-world datasets demonstrate that CCVAE outperforms current state-of-the-art techniques, thus providing a more effective method for metagenomic binning.

        ----

        ## [762] Delay-Adapted Policy Optimization and Improved Regret for Adversarial MDP with Delayed Bandit Feedback

        **Authors**: *Tal Lancewicki, Aviv Rosenberg, Dmitry Sotnikov*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/lancewicki23a.html](https://proceedings.mlr.press/v202/lancewicki23a.html)

        **Abstract**:

        Policy Optimization (PO) is one of the most popular methods in Reinforcement Learning (RL). Thus, theoretical guarantees for PO algorithms have become especially important to the RL community. In this paper, we study PO in adversarial MDPs with a challenge that arises in almost every real-world application – delayed bandit feedback. We give the first near-optimal regret bounds for PO in tabular MDPs, and may even surpass state-of-the-art (which uses less efficient methods). Our novel Delay-Adapted PO (DAPO) is easy to implement and to generalize, allowing us to extend our algorithm to: (i) infinite state space under the assumption of linear $Q$-function, proving the first regret bounds for delayed feedback with function approximation. (ii) deep RL, demonstrating its effectiveness in experiments on MuJoCo domains.

        ----

        ## [763] Lottery Tickets in Evolutionary Optimization: On Sparse Backpropagation-Free Trainability

        **Authors**: *Robert Tjarko Lange, Henning Sprekeler*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/lange23a.html](https://proceedings.mlr.press/v202/lange23a.html)

        **Abstract**:

        Is the lottery ticket phenomenon an idiosyncrasy of gradient-based training or does it generalize to evolutionary optimization? In this paper we establish the existence of highly sparse trainable initializations for evolution strategies (ES) and characterize qualitative differences compared to gradient descent (GD)-based sparse training. We introduce a novel signal-to-noise iterative pruning procedure, which incorporates loss curvature information into the network pruning step. This can enable the discovery of even sparser trainable network initializations when using black-box evolution as compared to GD-based optimization. Furthermore, we find that these initializations encode an inductive bias, which transfers across different ES, related tasks and even to GD-based training. Finally, we compare the local optima resulting from the different optimization paradigms and sparsity levels. In contrast to GD, ES explore diverse and flat local optima and do not preserve linear mode connectivity across sparsity levels and independent runs. The results highlight qualitative differences between evolution and gradient-based learning dynamics, which can be uncovered by the study of iterative pruning procedures.

        ----

        ## [764] On the Occupancy Measure of Non-Markovian Policies in Continuous MDPs

        **Authors**: *Romain Laroche, Remi Tachet des Combes*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/laroche23a.html](https://proceedings.mlr.press/v202/laroche23a.html)

        **Abstract**:

        The state-action occupancy measure of a policy is the expected (discounted or undiscounted) number of times a state-action couple is visited in a trajectory. For decades, RL books have been reporting the occupancy equivalence between Markovian and non-Markovian policies in countable state-action spaces under mild conditions. This equivalence states that the occupancy of any non-Markovian policy can be equivalently obtained by a Markovian policy, i.e. a memoryless probability distribution, conditioned only on its current state. While expected, for technical reasons, the translation of this result to continuous state space has resisted until now. Our main contribution is to fill this gap and to provide a general measure-theoretic treatment of the problem, permitting, in particular, its extension to continuous MDPs. Furthermore, we show that when the occupancy is infinite, we may encounter some non-trivial cases where the result does not hold anymore.

        ----

        ## [765] Minimalistic Predictions to Schedule Jobs with Online Precedence Constraints

        **Authors**: *Alexandra Anna Lassota, Alexander Lindermayr, Nicole Megow, Jens Schlöter*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/lassota23a.html](https://proceedings.mlr.press/v202/lassota23a.html)

        **Abstract**:

        We consider non-clairvoyant scheduling with online precedence constraints, where an algorithm is oblivious to any job dependencies and learns about a job only if all of its predecessors have been completed. Given strong impossibility results in classical competitive analysis, we investigate the problem in a learning-augmented setting, where an algorithm has access to predictions without any quality guarantee. We discuss different prediction models: novel problem-specific models as well as general ones, which have been proposed in previous works. We present lower bounds and algorithmic upper bounds for different precedence topologies, and thereby give a structured overview on which and how additional (possibly erroneous) information helps for designing better algorithms. Along the way, we also improve bounds on traditional competitive ratios for existing algorithms.

        ----

        ## [766] Speeding Up Bellman Ford via Minimum Violation Permutations

        **Authors**: *Silvio Lattanzi, Ola Svensson, Sergei Vassilvitskii*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/lattanzi23a.html](https://proceedings.mlr.press/v202/lattanzi23a.html)

        **Abstract**:

        The Bellman-Ford algorithm is a basic primitive for computing single source shortest paths in graphs with negative weight edges. Its running time is governed by the order the algorithm examines vertices for iterative updates on the value of their shortest path. In this work we study this problem through the lens of ’Algorithms with predictions,’ and show how to leverage auxiliary information from similar instances to improve the running time. We do this by identifying the key problem of Minimum Violation Permutations, and give algorithms with strong approximation guarantees as well as formal lower bounds. We complement the theoretical analysis with an empirical evaluation, showing that this approach can lead to a significant speed up in practice.

        ----

        ## [767] Who Needs to Know? Minimal Knowledge for Optimal Coordination

        **Authors**: *Niklas Lauffer, Ameesh Shah, Micah Carroll, Michael D. Dennis, Stuart Russell*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/lauffer23a.html](https://proceedings.mlr.press/v202/lauffer23a.html)

        **Abstract**:

        To optimally coordinate with others in cooperative games, it is often crucial to have information about one’s collaborators: successful driving requires understanding which side of the road to drive on. However, not every feature of collaborators is strategically relevant: the fine-grained acceleration of drivers may be ignored while maintaining optimal coordination. We show that there is a well-defined dichotomy between strategically relevant and irrelevant information. Moreover, we show that, in dynamic games, this dichotomy has a compact representation that can be efficiently computed via a Bellman backup operator. We apply this algorithm to analyze the strategically relevant information for tasks in both a standard and a partially observable version of the Overcooked environment. Theoretical and empirical results show that our algorithms are significantly more efficient than baselines. Videos are available at https://minknowledge.github.io.

        ----

        ## [768] Target-based Surrogates for Stochastic Optimization

        **Authors**: *Jonathan Wilder Lavington, Sharan Vaswani, Reza Babanezhad Harikandeh, Mark Schmidt, Nicolas Le Roux*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/lavington23a.html](https://proceedings.mlr.press/v202/lavington23a.html)

        **Abstract**:

        We consider minimizing functions for which it is expensive to compute the (possibly stochastic) gradient. Such functions are prevalent in reinforcement learning, imitation learning and adversarial training. Our target optimization framework uses the (expensive) gradient computation to construct surrogate functions in a target space (e.g. the logits output by a linear model for classification) that can be minimized efficiently. This allows for multiple parameter updates to the model, amortizing the cost of gradient computation. In the full-batch setting, we prove that our surrogate is a global upper-bound on the loss, and can be (locally) minimized using a black-box optimization algorithm. We prove that the resulting majorization-minimization algorithm ensures convergence to a stationary point of the loss. Next, we instantiate our framework in the stochastic setting and propose the $SSO$ algorithm, which can be viewed as projected stochastic gradient descent in the target space. This connection enables us to prove theoretical guarantees for $SSO$ when minimizing convex functions. Our framework allows the use of standard stochastic optimization algorithms to construct surrogates which can be minimized by any deterministic optimization method. To evaluate our framework, we consider a suite of supervised learning and imitation learning problems. Our experiments indicate the benefits of target optimization and the effectiveness of $SSO$.

        ----

        ## [769] Cluster Explanation via Polyhedral Descriptions

        **Authors**: *Connor Lawless, Oktay Günlük*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/lawless23a.html](https://proceedings.mlr.press/v202/lawless23a.html)

        **Abstract**:

        This paper focuses on the cluster description problem where, given a dataset and its partition into clusters, the task is to explain the clusters. We introduce a new approach to explain clusters by constructing a polyhedron around each cluster while minimizing either the complexity of the resulting polyhedra or the number of features used in the description. We formulate the cluster description problem as an integer program and present a column generation approach to search over an exponential number of candidate half-spaces that can be used to build the polyhedra. To deal with large datasets, we introduce a novel grouping scheme that first forms smaller groups of data points and then builds the polyhedra around the grouped data, a strategy which out-performs the common approach of sub-sampling data. Compared to state of the art cluster description algorithms, our approach is able to achieve competitive interpretability with improved description accuracy.

        ----

        ## [770] Pre-training for Speech Translation: CTC Meets Optimal Transport

        **Authors**: *Phuong-Hang Le, Hongyu Gong, Changhan Wang, Juan Pino, Benjamin Lecouteux, Didier Schwab*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/le23a.html](https://proceedings.mlr.press/v202/le23a.html)

        **Abstract**:

        The gap between speech and text modalities is a major challenge in speech-to-text translation (ST). Different methods have been proposed to reduce this gap, but most of them require architectural changes in ST training. In this work, we propose to mitigate this issue at the pre-training stage, requiring no change in the ST model. First, we show that the connectionist temporal classification (CTC) loss can reduce the modality gap by design. We provide a quantitative comparison with the more common cross-entropy loss, showing that pre-training with CTC consistently achieves better final ST accuracy. Nevertheless, CTC is only a partial solution and thus, in our second contribution, we propose a novel pre-training method combining CTC and optimal transport to further reduce this gap. Our method pre-trains a Siamese-like model composed of two encoders, one for acoustic inputs and the other for textual inputs, such that they produce representations that are close to each other in the Wasserstein space. Extensive experiments on the standard CoVoST-2 and MuST-C datasets show that our pre-training method applied to the vanilla encoder-decoder Transformer achieves state-of-the-art performance under the no-external-data setting, and performs on par with recent strong multi-task learning systems trained with external data. Finally, our method can also be applied on top of these multi-task systems, leading to further improvements for these models.

        ----

        ## [771] Bootstrapped Representations in Reinforcement Learning

        **Authors**: *Charline Le Lan, Stephen Tu, Mark Rowland, Anna Harutyunyan, Rishabh Agarwal, Marc G. Bellemare, Will Dabney*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/le-lan23a.html](https://proceedings.mlr.press/v202/le-lan23a.html)

        **Abstract**:

        In reinforcement learning (RL), state representations are key to dealing with large or continuous state spaces. While one of the promises of deep learning algorithms is to automatically construct features well-tuned for the task they try to solve, such a representation might not emerge from end-to-end training of deep RL agents. To mitigate this issue, auxiliary objectives are often incorporated into the learning process and help shape the learnt state representation. Bootstrapping methods are today’s method of choice to make these additional predictions. Yet, it is unclear which features these algorithms capture and how they relate to those from other auxiliary-task-based approaches. In this paper, we address this gap and provide a theoretical characterization of the state representation learnt by temporal difference learning (Sutton, 1988). Surprisingly, we find that this representation differs from the features learned by Monte Carlo and residual gradient algorithms for most transition structures of the environment in the policy evaluation setting. We describe the efficacy of these representations for policy evaluation, and use our theoretical analysis to design new auxiliary learning rules. We complement our theoretical results with an empirical comparison of these learning rules for different cumulant functions on classic domains such as the four-room domain (Sutton et al, 1999) and Mountain Car (Moore, 1990).

        ----

        ## [772] Strategic Classification with Unknown User Manipulations

        **Authors**: *Tosca Lechner, Ruth Urner, Shai Ben-David*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/lechner23a.html](https://proceedings.mlr.press/v202/lechner23a.html)

        **Abstract**:

        In many human-centric applications for Machine Learning instances will adapt to a classifier after its deployment. The field of strategic classification deals with this issue by aiming for a classifier that balances the trade-off between correctness and robustness to manipulation. This task is made harder if the underlying manipulation structure (i.e. the set of manipulations available at every instance) is unknown to the learner. We propose a novel batch-learning setting in which we use unlabeled data from previous rounds to estimate the manipulation structure. We show that in this batch-learning setting it is possible to learn a close-to-optimal classifier in terms of the strategic loss even without knowing the feasible manipulations beforehand. In line with recent advances in the strategic classification literature, we do not assume a best-response from agents but only require that observed manipulations are feasible.

        ----

        ## [773] Learning in POMDPs is Sample-Efficient with Hindsight Observability

        **Authors**: *Jonathan Lee, Alekh Agarwal, Christoph Dann, Tong Zhang*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/lee23a.html](https://proceedings.mlr.press/v202/lee23a.html)

        **Abstract**:

        POMDPs capture a broad class of decision making problems, but hardness results suggest that learning is intractable even in simple settings due to the inherent partial observability. However, in many realistic problems, more information is either revealed or can be computed during some point of the learning process. Motivated by diverse applications ranging from robotics to data center scheduling, we formulate a Hindsight Observable Markov Decision Process (HOMDP) as a POMDP where the latent states are revealed to the learner in hindsight and only during training. We introduce new algorithms for the tabular and function approximation settings that are provably sample-efficient with hindsight observability, even in POMDPs that would otherwise be statistically intractable. We give a lower bound showing that the tabular algorithm is optimal in its dependence on latent state and observation cardinalities.

        ----

        ## [774] Towards Deep Attention in Graph Neural Networks: Problems and Remedies

        **Authors**: *Soo Yong Lee, Fanchen Bu, Jaemin Yoo, Kijung Shin*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/lee23b.html](https://proceedings.mlr.press/v202/lee23b.html)

        **Abstract**:

        Graph neural networks (GNNs) learn the representation of graph-structured data, and their expressiveness can be further enhanced by inferring node relations for propagation. Attention-based GNNs infer neighbor importance to manipulate the weight of its propagation. Despite their popularity, the discussion on deep graph attention and its unique challenges has been limited. In this work, we investigate some problematic phenomena related to deep graph attention, including vulnerability to over-smoothed features and smooth cumulative attention. Through theoretical and empirical analyses, we show that various attention-based GNNs suffer from these problems. Motivated by our findings, we propose AERO-GNN, a novel GNN architecture designed for deep graph attention. AERO-GNN provably mitigates the proposed problems of deep graph attention, which is further empirically demonstrated with (a) its adaptive and less smooth attention functions and (b) higher performance at deep layers (up to 64). On 9 out of 12 node classification benchmarks, AERO-GNN outperforms the baseline GNNs, highlighting the advantages of deep graph attention. Our code is available at https://github.com/syleeheal/AERO-GNN.

        ----

        ## [775] InGram: Inductive Knowledge Graph Embedding via Relation Graphs

        **Authors**: *Jaejun Lee, Chanyoung Chung, Joyce Jiyoung Whang*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/lee23c.html](https://proceedings.mlr.press/v202/lee23c.html)

        **Abstract**:

        Inductive knowledge graph completion has been considered as the task of predicting missing triplets between new entities that are not observed during training. While most inductive knowledge graph completion methods assume that all entities can be new, they do not allow new relations to appear at inference time. This restriction prohibits the existing methods from appropriately handling real-world knowledge graphs where new entities accompany new relations. In this paper, we propose an INductive knowledge GRAph eMbedding method, InGram, that can generate embeddings of new relations as well as new entities at inference time. Given a knowledge graph, we define a relation graph as a weighted graph consisting of relations and the affinity weights between them. Based on the relation graph and the original knowledge graph, InGram learns how to aggregate neighboring embeddings to generate relation and entity embeddings using an attention mechanism. Experimental results show that InGram outperforms 14 different state-of-the-art methods on varied inductive learning scenarios.

        ----

        ## [776] Optimality of Thompson Sampling with Noninformative Priors for Pareto Bandits

        **Authors**: *Jongyeong Lee, Junya Honda, Chao-Kai Chiang, Masashi Sugiyama*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/lee23d.html](https://proceedings.mlr.press/v202/lee23d.html)

        **Abstract**:

        In the stochastic multi-armed bandit problem, a randomized probability matching policy called Thompson sampling (TS) has shown excellent performance in various reward models. In addition to the empirical performance, TS has been shown to achieve asymptotic problem-dependent lower bounds in several models. However, its optimality has been mainly addressed under light-tailed or one-parameter models that belong to exponential families. In this paper, we consider the optimality of TS for the Pareto model that has a heavy tail and is parameterized by two unknown parameters. Specifically, we discuss the optimality of TS with probability matching priors that include the Jeffreys prior and the reference priors. We first prove that TS with certain probability matching priors can achieve the optimal regret bound. Then, we show the suboptimality of TS with other priors, including the Jeffreys and the reference priors. Nevertheless, we find that TS with the Jeffreys and reference priors can achieve the asymptotic lower bound if one uses a truncation procedure. These results suggest carefully choosing noninformative priors to avoid suboptimality and show the effectiveness of truncation procedures in TS-based policies.

        ----

        ## [777] Conditional Graph Information Bottleneck for Molecular Relational Learning

        **Authors**: *Namkyeong Lee, Dongmin Hyun, Gyoung S. Na, Sungwon Kim, Junseok Lee, Chanyoung Park*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/lee23e.html](https://proceedings.mlr.press/v202/lee23e.html)

        **Abstract**:

        Molecular relational learning, whose goal is to learn the interaction behavior between molecular pairs, got a surge of interest in molecular sciences due to its wide range of applications. Recently, graph neural networks have recently shown great success in molecular relational learning by modeling a molecule as a graph structure, and considering atom-level interactions between two molecules. Despite their success, existing molecular relational learning methods tend to overlook the nature of chemistry, i.e., a chemical compound is composed of multiple substructures such as functional groups that cause distinctive chemical reactions. In this work, we propose a novel relational learning framework, called CGIB, that predicts the interaction behavior between a pair of graphs by detecting core subgraphs therein. The main idea is, given a pair of graphs, to find a subgraph from a graph that contains the minimal sufficient information regarding the task at hand conditioned on the paired graph based on the principle of conditional graph information bottleneck. We argue that our proposed method mimics the nature of chemical reactions, i.e., the core substructure of a molecule varies depending on which other molecule it interacts with. Extensive experiments on various tasks with real-world datasets demonstrate the superiority of CGIB over state-of-the-art baselines. Our code is available at https://github.com/Namkyeong/CGIB.

        ----

        ## [778] Exploring Chemical Space with Score-based Out-of-distribution Generation

        **Authors**: *Seul Lee, Jaehyeong Jo, Sung Ju Hwang*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/lee23f.html](https://proceedings.mlr.press/v202/lee23f.html)

        **Abstract**:

        A well-known limitation of existing molecular generative models is that the generated molecules highly resemble those in the training set. To generate truly novel molecules that may have even better properties for de novo drug discovery, more powerful exploration in the chemical space is necessary. To this end, we propose Molecular Out-Of-distribution Diffusion(MOOD), a score-based diffusion scheme that incorporates out-of-distribution (OOD) control in the generative stochastic differential equation (SDE) with simple control of a hyperparameter, thus requires no additional costs. Since some novel molecules may not meet the basic requirements of real-world drugs, MOOD performs conditional generation by utilizing the gradients from a property predictor that guides the reverse-time diffusion process to high-scoring regions according to target properties such as protein-ligand interactions, drug-likeness, and synthesizability. This allows MOOD to search for novel and meaningful molecules rather than generating unseen yet trivial ones. We experimentally validate that MOOD is able to explore the chemical space beyond the training distribution, generating molecules that outscore ones found with existing methods, and even the top 0.01% of the original training pool. Our code is available at https://github.com/SeulLee05/MOOD.

        ----

        ## [779] Pix2Struct: Screenshot Parsing as Pretraining for Visual Language Understanding

        **Authors**: *Kenton Lee, Mandar Joshi, Iulia Raluca Turc, Hexiang Hu, Fangyu Liu, Julian Martin Eisenschlos, Urvashi Khandelwal, Peter Shaw, Ming-Wei Chang, Kristina Toutanova*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/lee23g.html](https://proceedings.mlr.press/v202/lee23g.html)

        **Abstract**:

        Visually-situated language is ubiquitous—sources range from textbooks with diagrams to web pages with images and tables, to mobile apps with buttons and forms. Perhaps due to this diversity, previous work has typically relied on domain-specific recipes with limited sharing of the underlying data, model architectures, and objectives. We present Pix2Struct, a pretrained image-to-text model for purely visual language understanding, which can be finetuned on tasks containing visually-situated language. Pix2Struct is pretrained by learning to parse masked screenshots of web pages into simplified HTML. The web, with its richness of visual elements cleanly reflected in the HTML structure, provides a large source of pretraining data well suited to the diversity of downstream tasks. Intuitively, this objective subsumes common pretraining signals such as OCR, language modeling, and image captioning. In addition to the novel pretraining strategy, we introduce a variable-resolution input representation and a more flexible integration of language and vision inputs, where language prompts such as questions are rendered directly on top of the input image. For the first time, we show that a single pretrained model can achieve state-of-the-art results in six out of nine tasks across four domains: documents, illustrations, user interfaces, and natural images.

        ----

        ## [780] FlexRound: Learnable Rounding based on Element-wise Division for Post-Training Quantization

        **Authors**: *Jung Hyun Lee, Jeonghoon Kim, Se Jung Kwon, Dongsoo Lee*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/lee23h.html](https://proceedings.mlr.press/v202/lee23h.html)

        **Abstract**:

        Post-training quantization (PTQ) has been gaining popularity for the deployment of deep neural networks on resource-limited devices since unlike quantization-aware training, neither a full training dataset nor end-to-end training is required at all. As PTQ schemes based on reconstructing each layer or block output turn out to be effective to enhance quantized model performance, recent works have developed algorithms to devise and learn a new weight-rounding scheme so as to better reconstruct each layer or block output. In this work, we propose a simple yet effective new weight-rounding mechanism for PTQ, coined FlexRound, based on element-wise division instead of typical element-wise addition such that FlexRound enables jointly learning a common quantization grid size as well as a different scale for each pre-trained weight. Thanks to the reciprocal rule of derivatives induced by element-wise division, FlexRound is inherently able to exploit pre-trained weights when updating their corresponding scales, and thus, flexibly quantize pre-trained weights depending on their magnitudes. We empirically validate the efficacy of FlexRound on a wide range of models and tasks. To the best of our knowledge, our work is the first to carry out comprehensive experiments on not only image classification and natural language understanding but also natural language generation, assuming a per-tensor uniform PTQ setting. Moreover, we demonstrate, for the first time, that large language models can be efficiently quantized, with only a negligible impact on performance compared to half-precision baselines, achieved by reconstructing the output in a block-by-block manner.

        ----

        ## [781] CoDi: Co-evolving Contrastive Diffusion Models for Mixed-type Tabular Synthesis

        **Authors**: *Chaejeong Lee, Jayoung Kim, Noseong Park*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/lee23i.html](https://proceedings.mlr.press/v202/lee23i.html)

        **Abstract**:

        With growing attention to tabular data these days, the attempt to apply a synthetic table to various tasks has been expanded toward various scenarios. Owing to the recent advances in generative modeling, fake data generated by tabular data synthesis models become sophisticated and realistic. However, there still exists a difficulty in modeling discrete variables (columns) of tabular data. In this work, we propose to process continuous and discrete variables separately (but being conditioned on each other) by two diffusion models. The two diffusion models are co-evolved during training by reading conditions from each other. In order to further bind the diffusion models, moreover, we introduce a contrastive learning method with a negative sampling method. In our experiments with 11 real-world tabular datasets and 8 baseline methods, we prove the efficacy of the proposed method, called $\texttt{CoDi}$. Our code is available at https://github.com/ChaejeongLee/CoDi.

        ----

        ## [782] Minimizing Trajectory Curvature of ODE-based Generative Models

        **Authors**: *Sangyun Lee, Beomsu Kim, Jong Chul Ye*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/lee23j.html](https://proceedings.mlr.press/v202/lee23j.html)

        **Abstract**:

        Recent ODE/SDE-based generative models, such as diffusion models, rectified flows, and flow matching, define a generative process as a time reversal of a fixed forward process. Even though these models show impressive performance on large-scale datasets, numerical simulation requires multiple evaluations of a neural network, leading to a slow sampling speed. We attribute the reason to the high curvature of the learned generative trajectories, as it is directly related to the truncation error of a numerical solver. Based on the relationship between the forward process and the curvature, here we present an efficient method of training the forward process to minimize the curvature of generative trajectories without any ODE/SDE simulation. Experiments show that our method achieves a lower curvature than previous models and, therefore, decreased sampling costs while maintaining competitive performance. Code is available at https://github.com/sangyun884/fast-ode.

        ----

        ## [783] H-Likelihood Approach to Deep Neural Networks with Temporal-Spatial Random Effects for High-Cardinality Categorical Features

        **Authors**: *Hangbin Lee, Youngjo Lee*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/lee23k.html](https://proceedings.mlr.press/v202/lee23k.html)

        **Abstract**:

        Deep Neural Networks (DNNs) are one of the most powerful tools for prediction, but many of them implicitly assume that the data are statistically independent. However, in the real world, it is common for large-scale data to be clustered with temporal-spatial correlation structures. Variational approaches and integrated likelihood approaches have been proposed to obtain approximate maximum likelihood estimators (MLEs) for correlated data. However, due to the large size of data, they cannot provide exact MLEs. In this study, we propose a new hierarchical likelihood approach to DNNs with correlated random effects for clustered data. By jointly optimizing the the negative h-likelihood loss, we can provide exact MLEs for both mean and dispersion parameters, as well as the best linear unbiased predictors for the random effects. Moreover, the hierarchical likelihood allows a computable procedure for restricted maximum likelihood estimators of dispersion parameters. The proposed two-step algorithm enables online learning for the neural networks, whereas the integrated likelihood cannot decompose like a widely-used loss function in DNNs. The proposed h-likelihood approach offers several advantages, which we demonstrate through numerical studies and real data analyses.

        ----

        ## [784] On the Importance of Feature Decorrelation for Unsupervised Representation Learning in Reinforcement Learning

        **Authors**: *Hojoon Lee, Koanho Lee, Dongyoon Hwang, Hyunho Lee, Byungkun Lee, Jaegul Choo*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/lee23l.html](https://proceedings.mlr.press/v202/lee23l.html)

        **Abstract**:

        Recently, unsupervised representation learning (URL) has improved the sample efficiency of Reinforcement Learning (RL) by pretraining a model from a large unlabeled dataset. The underlying principle of these methods is to learn temporally predictive representations by predicting future states in the latent space. However, an important challenge of this approach is the representational collapse, where the subspace of the latent representations collapses into a low-dimensional manifold. To address this issue, we propose a novel URL framework that causally predicts future states while increasing the dimension of the latent manifold by decorrelating the features in the latent space. Through extensive empirical studies, we demonstrate that our framework effectively learns predictive representations without collapse, which significantly improves the sample efficiency of state-of-the-art URL methods on the Atari 100k benchmark. The code is available at https://github.com/dojeon-ai/SimTPR.

        ----

        ## [785] HETAL: Efficient Privacy-preserving Transfer Learning with Homomorphic Encryption

        **Authors**: *Seewoo Lee, Garam Lee, Jung Woo Kim, Junbum Shin, Mun-Kyu Lee*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/lee23m.html](https://proceedings.mlr.press/v202/lee23m.html)

        **Abstract**:

        Transfer learning is a de facto standard method for efficiently training machine learning models for data-scarce problems by adding and fine-tuning new classification layers to a model pre-trained on large datasets. Although numerous previous studies proposed to use homomorphic encryption to resolve the data privacy issue in transfer learning in the machine learning as a service setting, most of them only focused on encrypted inference. In this study, we present HETAL, an efficient Homomorphic Encryption based Transfer Learning algorithm, that protects the client’s privacy in training tasks by encrypting the client data using the CKKS homomorphic encryption scheme. HETAL is the first practical scheme that strictly provides encrypted training, adopting validation-based early stopping and achieving the accuracy of nonencrypted training. We propose an efficient encrypted matrix multiplication algorithm, which is 1.8 to 323 times faster than prior methods, and a highly precise softmax approximation algorithm with increased coverage. The experimental results for five well-known benchmark datasets show total training times of 567–3442 seconds, which is less than an hour.

        ----

        ## [786] QASA: Advanced Question Answering on Scientific Articles

        **Authors**: *Yoonjoo Lee, Kyungjae Lee, Sunghyun Park, Dasol Hwang, Jaehyeon Kim, Hong-In Lee, Moontae Lee*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/lee23n.html](https://proceedings.mlr.press/v202/lee23n.html)

        **Abstract**:

        Reasoning is the crux of intellectual thinking. While question answering (QA) tasks are prolific with various computational models and benchmark datasets, they mostly tackle factoid or shallow QA without asking deeper understanding. Dual process theory asserts that human reasoning consists of associative thinking to collect relevant pieces of knowledge and logical reasoning to consciously conclude grounding on evidential rationale. Based on our intensive think-aloud study that revealed the three types of questions: surface, testing, and deep questions, we first propose the QASA benchmark that consists of 1798 novel question answering pairs that require full-stack reasoning on scientific articles in AI and ML fields. Then we propose the QASA approach that tackles the full-stack reasoning with large language models via associative selection, evidential rationale-generation, and systematic composition. Our experimental results show that QASA’s full-stack inference outperforms the state-of-the-art InstructGPT by a big margin. We also find that rationale-generation is critical for the performance gain, claiming how we should rethink advanced question answering. The dataset is available at https://github.com/lgresearch/QASA.

        ----

        ## [787] Demystifying Disagreement-on-the-Line in High Dimensions

        **Authors**: *Donghwan Lee, Behrad Moniri, Xinmeng Huang, Edgar Dobriban, Hamed Hassani*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/lee23o.html](https://proceedings.mlr.press/v202/lee23o.html)

        **Abstract**:

        Evaluating the performance of machine learning models under distribution shifts is challenging, especially when we only have unlabeled data from the shifted (target) domain, along with labeled data from the original (source) domain. Recent work suggests that the notion of disagreement, the degree to which two models trained with different randomness differ on the same input, is a key to tackling this problem. Experimentally, disagreement and prediction error have been shown to be strongly connected, which has been used to estimate model performance. Experiments have led to the discovery of the disagreement-on-the-line phenomenon, whereby the classification error under the target domain is often a linear function of the classification error under the source domain; and whenever this property holds, disagreement under the source and target domain follow the same linear relation. In this work, we develop a theoretical foundation for analyzing disagreement in high-dimensional random features regression; and study under what conditions the disagreement-on-the-line phenomenon occurs in our setting. Experiments on CIFAR-10-C, Tiny ImageNet-C, and Camelyon17 are consistent with our theory and support the universality of the theoretical findings.

        ----

        ## [788] On the Correctness of Automatic Differentiation for Neural Networks with Machine-Representable Parameters

        **Authors**: *Wonyeol Lee, Sejun Park, Alex Aiken*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/lee23p.html](https://proceedings.mlr.press/v202/lee23p.html)

        **Abstract**:

        Recent work has shown that forward- and reverse- mode automatic differentiation (AD) over the reals is almost always correct in a mathematically precise sense. However, actual programs work with machine-representable numbers (e.g., floating-point numbers), not reals. In this paper, we study the correctness of AD when the parameter space of a neural network consists solely of machine-representable numbers. In particular, we analyze two sets of parameters on which AD can be incorrect: the incorrect set on which the network is differentiable but AD does not compute its derivative, and the non-differentiable set on which the network is non-differentiable. For a neural network with bias parameters, we first prove that the incorrect set is always empty. We then prove a tight bound on the size of the non-differentiable set, which is linear in the number of non-differentiabilities in activation functions, and give a simple necessary and sufficient condition for a parameter to be in this set. We further prove that AD always computes a Clarke subderivative even on the non-differentiable set. We also extend these results to neural networks possibly without bias parameters.

        ----

        ## [789] Implicit Jacobian regularization weighted with impurity of probability output

        **Authors**: *Sungyoon Lee, Jinseong Park, Jaewook Lee*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/lee23q.html](https://proceedings.mlr.press/v202/lee23q.html)

        **Abstract**:

        The success of deep learning is greatly attributed to stochastic gradient descent (SGD), yet it remains unclear how SGD finds well-generalized models. We demonstrate that SGD has an implicit regularization effect on the logit-weight Jacobian norm of neural networks. This regularization effect is weighted with the impurity of the probability output, and thus it is active in a certain phase of training. Moreover, based on these findings, we propose a novel optimization method that explicitly regularizes the Jacobian norm, which leads to similar performance as other state-of-the-art sharpness-aware optimization methods.

        ----

        ## [790] Unsupervised Skill Discovery for Learning Shared Structures across Changing Environments

        **Authors**: *Sang-Hyun Lee, Seung-Woo Seo*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/lee23r.html](https://proceedings.mlr.press/v202/lee23r.html)

        **Abstract**:

        Learning shared structures across changing environments enables an agent to efficiently retain obtained knowledge and transfer it between environments. A skill is a promising concept to represent shared structures. Several recent works proposed unsupervised skill discovery algorithms that can discover useful skills without a reward function. However, they focused on discovering skills in stationary environments or assumed that a skill being trained is fixed within an episode, which is insufficient to learn and represent shared structures. In this paper, we introduce a new unsupervised skill discovery algorithm that discovers a set of skills that can represent shared structures across changing environments. Our algorithm trains incremental skills and encourages a new skill to expand state coverage obtained with compositions of previously learned skills. We also introduce a skill evaluation process to prevent our skills from containing redundant skills, a common issue in previous work. Our experimental results show that our algorithm acquires skills that represent shared structures across changing maze navigation and locomotion environments. Furthermore, we demonstrate that our skills are more useful than baselines on downstream tasks.

        ----

        ## [791] Generalization Analysis for Contrastive Representation Learning

        **Authors**: *Yunwen Lei, Tianbao Yang, Yiming Ying, Ding-Xuan Zhou*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/lei23a.html](https://proceedings.mlr.press/v202/lei23a.html)

        **Abstract**:

        Recently, contrastive learning has found impressive success in advancing the state of the art in solving various machine learning tasks. However, the existing generalization analysis is very limited or even not meaningful. In particular, the existing generalization error bounds depend linearly on the number $k$ of negative examples while it was widely shown in practice that choosing a large $k$ is necessary to guarantee good generalization of contrastive learning in downstream tasks. In this paper, we establish novel generalization bounds for contrastive learning which do not depend on $k$, up to logarithmic terms. Our analysis uses structural results on empirical covering numbers and Rademacher complexities to exploit the Lipschitz continuity of loss functions. For self-bounding Lipschitz loss functions, we further improve our results by developing optimistic bounds which imply fast rates in a low noise condition. We apply our results to learning with both linear representation and nonlinear representation by deep neural networks, for both of which we derive Rademacher complexity bounds to get improved generalization bounds.

        ----

        ## [792] Learning Control by Iterative Inversion

        **Authors**: *Gal Leibovich, Guy Jacob, Or Avner, Gal Novik, Aviv Tamar*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/leibovich23a.html](https://proceedings.mlr.press/v202/leibovich23a.html)

        **Abstract**:

        We propose iterative inversion - an algorithm for learning an inverse function without input-output pairs, but only with samples from the desired output distribution and access to the forward function. The key challenge is a distribution shift between the desired outputs and the outputs of an initial random guess, and we prove that iterative inversion can steer the learning correctly, under rather strict conditions on the function. We apply iterative inversion to learn control. Our input is a set of demonstrations of desired behavior, given as video embeddings of trajectories (without actions), and our method iteratively learns to imitate trajectories generated by the current policy, perturbed by random exploration noise. Our approach does not require rewards, and only employs supervised learning, which can be easily scaled to use state-of-the-art trajectory embedding techniques and policy representations. Indeed, with a VQ-VAE embedding, and a transformer-based policy, we demonstrate non-trivial continuous control on several tasks (videos available at https://sites.google.com/view/iter-inver). Further, we report an improved performance on imitating diverse behaviors compared to reward based methods.

        ----

        ## [793] Sampling-Based Accuracy Testing of Posterior Estimators for General Inference

        **Authors**: *Pablo Lemos, Adam Coogan, Yashar Hezaveh, Laurence Perreault Levasseur*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/lemos23a.html](https://proceedings.mlr.press/v202/lemos23a.html)

        **Abstract**:

        Parameter inference, i.e. inferring the posterior distribution of the parameters of a statistical model given some data, is a central problem to many scientific disciplines. Posterior inference with generative models is an alternative to methods such as Markov Chain Monte Carlo, both for likelihood-based and simulation-based inference. However, assessing the accuracy of posteriors encoded in generative models is not straightforward. In this paper, we introduce "Tests of Accuracy with Random Points" (TARP) coverage testing as a method to estimate coverage probabilities of generative posterior estimators. Our method differs from previously-existing coverage-based methods, which require posterior evaluations. We prove that our approach is necessary and sufficient to show that a posterior estimator is accurate. We demonstrate the method on a variety of synthetic examples, and show that TARP can be used to test the results of posterior inference analyses in high-dimensional spaces. We also show that our method can detect inaccurate inferences in cases where existing methods fail.

        ----

        ## [794] Fast Inference from Transformers via Speculative Decoding

        **Authors**: *Yaniv Leviathan, Matan Kalman, Yossi Matias*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/leviathan23a.html](https://proceedings.mlr.press/v202/leviathan23a.html)

        **Abstract**:

        Inference from large autoregressive models like Transformers is slow - decoding K tokens takes K serial runs of the model. In this work we introduce speculative decoding - an algorithm to sample from autoregressive models faster without any changes to the outputs, by computing several tokens in parallel. At the heart of our approach lie the observations that (1) hard language-modeling tasks often include easier subtasks that can be approximated well by more efficient models, and (2) using speculative execution and a novel sampling method, we can make exact decoding from the large models faster, by running them in parallel on the outputs of the approximation models, potentially generating several tokens concurrently, and without changing the distribution. Our method can accelerate existing off-the-shelf models without retraining or architecture changes. We demonstrate it on T5-XXL and show a 2X-3X acceleration compared to the standard T5X implementation, with identical outputs.

        ----

        ## [795] Efficient Rate Optimal Regret for Adversarial Contextual MDPs Using Online Function Approximation

        **Authors**: *Orin Levy, Alon Cohen, Asaf B. Cassel, Yishay Mansour*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/levy23a.html](https://proceedings.mlr.press/v202/levy23a.html)

        **Abstract**:

        We present the OMG-CMDP! algorithm for regret minimization in adversarial Contextual MDPs. The algorithm operates under the minimal assumptions of realizable function class and access to online least squares and log loss regression oracles. Our algorithm is efficient (assuming efficient online regression oracles), simple and robust to approximation errors. It enjoys an $\widetilde{O}(H^{2.5} \sqrt{ T|S||A| ( \mathcal{R}_{TH}(\mathcal{O}) + H \log(\delta^{-1}) )})$ regret guarantee, with $T$ being the number of episodes, $S$ the state space, $A$ the action space, $H$ the horizon and $\mathcal{R}_{TH}(\mathcal{O}) = \mathcal{R}_{TH}(\mathcal{O}_{sq}^\mathcal{F}) + \mathcal{R}_{TH}(\mathcal{O}_{log}^\mathcal{P})$ is the sum of the square and log-loss regression oracles’ regret, used to approximate the context-dependent rewards and dynamics, respectively. To the best of our knowledge, our algorithm is the first efficient rate optimal regret minimization algorithm for adversarial CMDPs that operates under the minimal standard assumption of online function approximation.

        ----

        ## [796] GLOBE-CE: A Translation Based Approach for Global Counterfactual Explanations

        **Authors**: *Dan Ley, Saumitra Mishra, Daniele Magazzeni*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/ley23a.html](https://proceedings.mlr.press/v202/ley23a.html)

        **Abstract**:

        Counterfactual explanations have been widely studied in explainability, with a range of application dependent methods prominent in fairness, recourse and model understanding. The major shortcoming associated with these methods, however, is their inability to provide explanations beyond the local or instance-level. While many works touch upon the notion of a global explanation, typically suggesting to aggregate masses of local explanations in the hope of ascertaining global properties, few provide frameworks that are both reliable and computationally tractable. Meanwhile, practitioners are requesting more efficient and interactive explainability tools. We take this opportunity to propose Global & Efficient Counterfactual Explanations (GLOBE-CE), a flexible framework that tackles the reliability and scalability issues associated with current state-of-the-art, particularly on higher dimensional datasets and in the presence of continuous features. Furthermore, we provide a unique mathematical analysis of categorical feature translations, utilising it in our method. Experimental evaluation with publicly available datasets and user studies demonstrate that GLOBE-CE performs significantly better than the current state-of-the-art across multiple metrics (e.g., speed, reliability).

        ----

        ## [797] TIPS: Topologically Important Path Sampling for Anytime Neural Networks

        **Authors**: *Guihong Li, Kartikeya Bhardwaj, Yuedong Yang, Radu Marculescu*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/li23a.html](https://proceedings.mlr.press/v202/li23a.html)

        **Abstract**:

        Anytime neural networks (AnytimeNNs) are a promising solution to adaptively adjust the model complexity at runtime under various hardware resource constraints. However, the manually-designed AnytimeNNs are biased by designers’ prior experience and thus provide sub-optimal solutions. To address the limitations of existing hand-crafted approaches, we first model the training process of AnytimeNNs as a discrete-time Markov chain (DTMC) and use it to identify the paths that contribute the most to the training of AnytimeNNs. Based on this new DTMC-based analysis, we further propose TIPS, a framework to automatically design AnytimeNNs under various hardware constraints. Our experimental results show that TIPS can improve the convergence rate and test accuracy of AnytimeNNs. Compared to the existing AnytimeNNs approaches, TIPS improves the accuracy by 2%-6.6% on multiple datasets and achieves SOTA accuracy-FLOPs tradeoffs.

        ----

        ## [798] MAHALO: Unifying Offline Reinforcement Learning and Imitation Learning from Observations

        **Authors**: *Anqi Li, Byron Boots, Ching-An Cheng*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/li23b.html](https://proceedings.mlr.press/v202/li23b.html)

        **Abstract**:

        We study a new paradigm for sequential decision making, called offline policy learning from observations (PLfO). Offline PLfO aims to learn policies using datasets with substandard qualities: 1) only a subset of trajectories is labeled with rewards, 2) labeled trajectories may not contain actions, 3) labeled trajectories may not be of high quality, and 4) the data may not have full coverage. Such imperfection is common in real-world learning scenarios, and offline PLfO encompasses many existing offline learning setups, including offline imitation learning (IL), offline IL from observations (ILfO), and offline reinforcement learning (RL). In this work, we present a generic approach to offline PLfO, called Modality-agnostic Adversarial Hypothesis Adaptation for Learning from Observations (MAHALO). Built upon the pessimism concept in offline RL, MAHALO optimizes the policy using a performance lower bound that accounts for uncertainty due to the dataset’s insufficient coverage. We implement this idea by adversarially training data-consistent critic and reward functions, which forces the learned policy to be robust to data deficiency. We show that MAHALO consistently outperforms or matches specialized algorithms across a variety of offline PLfO tasks in theory and experiments. Our code is available at https://github.com/AnqiLi/mahalo.

        ----

        ## [799] Internet Explorer: Targeted Representation Learning on the Open Web

        **Authors**: *Alexander Cong Li, Ellis Langham Brown, Alexei A. Efros, Deepak Pathak*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/li23c.html](https://proceedings.mlr.press/v202/li23c.html)

        **Abstract**:

        Vision models typically rely on fine-tuning general-purpose models pre-trained on large, static datasets. These general-purpose models only capture the knowledge within their pre-training datasets, which are tiny, out-of-date snapshots of the Internet—where billions of images are uploaded each day. We suggest an alternate approach: rather than hoping our static datasets transfer to our desired tasks after large-scale pre-training, we propose dynamically utilizing the Internet to quickly train a small-scale model that does extremely well on a target dataset. Our approach, called Internet Explorer, explores the web in a self-supervised manner to progressively find relevant examples that improve performance on a desired target dataset. It cycles between searching for images on the Internet with text queries, self-supervised training on downloaded images, determining which images were useful, and prioritizing what to search for next. We evaluate Internet Explorer across several datasets and show that it outperforms or matches CLIP oracle performance using just a single GPU desktop to actively query the Internet for 30-40 hours.

        ----

        

[Go to the previous page](ICML-2023-list03.md)

[Go to the next page](ICML-2023-list05.md)

[Go to the catalog section](README.md)