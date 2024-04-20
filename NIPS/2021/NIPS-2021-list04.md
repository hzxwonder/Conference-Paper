## [600] The future is log-Gaussian: ResNets and their infinite-depth-and-width limit at initialization

        **Authors**: *Mufan (Bill) Li, Mihai Nica, Daniel M. Roy*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/412758d043dd247bddea07c7ec558c31-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/412758d043dd247bddea07c7ec558c31-Abstract.html)

        **Abstract**:

        Theoretical results show that neural networks can be approximated by Gaussian processes in the infinite-width limit. However, for fully connected networks, it has been previously shown that for any fixed network width, $n$, the Gaussian approximation gets worse as the network depth, $d$, increases. Given that modern networks are deep, this raises the question of how well modern architectures, like ResNets, are captured by the infinite-width limit. To provide a better approximation, we study ReLU ResNets in the infinite-depth-and-width limit, where \emph{both} depth and width tend to infinity as their ratio, $d/n$, remains constant. In contrast to the Gaussian infinite-width limit, we show theoretically that the network exhibits log-Gaussian behaviour at initialization in the infinite-depth-and-width limit, with parameters depending on the ratio $d/n$. Using Monte Carlo simulations, we demonstrate that even basic properties of standard ResNet architectures are poorly captured by the Gaussian limit, but remarkably well captured by our log-Gaussian limit. Moreover, our analysis reveals that ReLU ResNets at initialization are hypoactivated: fewer than half of the ReLUs are activated. Additionally, we calculate the interlayer correlations, which have the effect of exponentially increasing the variance of the network output. Based on our analysis, we introduce \emph{Balanced ResNets}, a simple architecture modification, which eliminates hypoactivation and interlayer correlations and is more amenable to theoretical analysis.

        ----

        ## [601] Grammar-Based Grounded Lexicon Learning

        **Authors**: *Jiayuan Mao, Freda Shi, Jiajun Wu, Roger Levy, Josh Tenenbaum*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/4158f6d19559955bae372bb00f6204e4-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/4158f6d19559955bae372bb00f6204e4-Abstract.html)

        **Abstract**:

        We present Grammar-Based Grounded Language Learning (G2L2), a lexicalist approach toward learning a compositional and grounded meaning representation of language from grounded data, such as paired images and texts. At the core of G2L2 is a collection of lexicon entries, which map each word to a tuple of a syntactic type and a neuro-symbolic semantic program. For example, the word shiny has a syntactic type of adjective; its neuro-symbolic semantic program has the symbolic form $\lambda x.\textit{filter}(x, \textbf{SHINY})$, where the concept SHINY is associated with a neural network embedding, which will be used to classify shiny objects. Given an input sentence, G2L2 first looks up the lexicon entries associated with each token. It then derives the meaning of the sentence as an executable neuro-symbolic program by composing lexical meanings based on syntax. The recovered meaning programs can be executed on grounded inputs. To facilitate learning in an exponentially-growing compositional space, we introduce a joint parsing and expected execution algorithm, which does local marginalization over derivations to reduce the training time. We evaluate G2L2 on two domains: visual reasoning and language-driven navigation. Results show that G2L2 can generalize from small amounts of data to novel compositions of words.

        ----

        ## [602] Distributed Deep Learning In Open Collaborations

        **Authors**: *Michael Diskin, Alexey Bukhtiyarov, Max Ryabinin, Lucile Saulnier, Quentin Lhoest, Anton Sinitsin, Dmitry Popov, Dmitry V. Pyrkin, Maxim Kashirin, Alexander Borzunov, Albert Villanova del Moral, Denis Mazur, Ilia Kobelev, Yacine Jernite, Thomas Wolf, Gennady Pekhimenko*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/41a60377ba920919939d83326ebee5a1-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/41a60377ba920919939d83326ebee5a1-Abstract.html)

        **Abstract**:

        Modern deep learning applications require increasingly more compute to train state-of-the-art models. To address this demand, large corporations and institutions use dedicated High-Performance Computing clusters, whose construction and maintenance are both environmentally costly and well beyond the budget of most organizations. As a result, some research directions become the exclusive domain of a few large industrial and even fewer academic actors. To alleviate this disparity, smaller groups may pool their computational resources and run collaborative experiments that benefit all participants. This paradigm, known as grid- or volunteer computing, has seen successful applications in numerous scientific areas. However, using this approach for machine learning is difficult due to high latency, asymmetric bandwidth, and several challenges unique to volunteer computing. In this work, we carefully analyze these constraints and propose a novel algorithmic framework designed specifically for collaborative training. We demonstrate the effectiveness of our approach for SwAV and ALBERT pretraining in realistic conditions and achieve performance comparable to traditional setups at a fraction of the cost. Finally, we provide a detailed report of successful collaborative language model pretraining with nearly 50 participants.

        ----

        ## [603] Neural Ensemble Search for Uncertainty Estimation and Dataset Shift

        **Authors**: *Sheheryar Zaidi, Arber Zela, Thomas Elsken, Chris C. Holmes, Frank Hutter, Yee Whye Teh*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/41a6fd31aa2e75c3c6d427db3d17ea80-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/41a6fd31aa2e75c3c6d427db3d17ea80-Abstract.html)

        **Abstract**:

        Ensembles of neural networks achieve superior performance compared to standalone networks in terms of accuracy, uncertainty calibration and robustness to dataset shift. Deep ensembles, a state-of-the-art method for uncertainty estimation, only ensemble random initializations of a fixed architecture. Instead, we propose two methods for automatically constructing ensembles with varying architectures, which implicitly trade-off individual architectures’ strengths against the ensemble’s diversity and exploit architectural variation as a source of diversity. On a variety of classification tasks and modern architecture search spaces, we show that the resulting ensembles outperform deep ensembles not only in terms of accuracy but also uncertainty calibration and robustness to dataset shift. Our further analysis and ablation studies provide evidence of higher ensemble diversity due to architectural variation, resulting in ensembles that can outperform deep ensembles, even when having weaker average base learners. To foster reproducibility, our code is available: https://github.com/automl/nes

        ----

        ## [604] Finding Bipartite Components in Hypergraphs

        **Authors**: *Peter Macgregor, He Sun*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/41bacf567aefc61b3076c74d8925128f-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/41bacf567aefc61b3076c74d8925128f-Abstract.html)

        **Abstract**:

        Hypergraphs are important objects to model ternary or higher-order relations of objects, and have a number of applications in analysing many complex datasets occurring in practice.  In this work we study a new heat diffusion process in hypergraphs, and employ this process to design a polynomial-time algorithm that approximately finds bipartite components in a hypergraph.  We theoretically prove the performance of our proposed algorithm, and compare it against the previous state-of-the-art through extensive experimental analysis on both synthetic and real-world datasets. We find that our new algorithm consistently and significantly outperforms the previous state-of-the-art across a wide range of hypergraphs.

        ----

        ## [605] Hit and Lead Discovery with Explorative RL and Fragment-based Molecule Generation

        **Authors**: *Soojung Yang, Doyeong Hwang, Seul Lee, Seongok Ryu, Sung Ju Hwang*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/41da609c519d77b29be442f8c1105647-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/41da609c519d77b29be442f8c1105647-Abstract.html)

        **Abstract**:

        Recently, utilizing reinforcement learning (RL) to generate molecules with desired properties has been highlighted as a promising strategy for drug design. Molecular docking program -- a physical simulation that estimates protein-small molecule binding affinity -- can be an ideal reward scoring function for RL, as it is a straightforward proxy of the therapeutic potential. Still, two imminent challenges exist for this task. First, the models often fail to generate chemically realistic and pharmacochemically acceptable molecules. Second, the docking score optimization is a difficult exploration problem that involves many local optima and less smooth surface with respect to molecular structure. To tackle these challenges, we propose a novel RL framework that generates pharmacochemically acceptable molecules with large docking scores. Our method -- Fragment-based generative RL with Explorative Experience replay for Drug design (FREED) -- constrains the generated molecules to a realistic and qualified chemical space and effectively explores the space to find drugs by coupling our fragment-based generation method and a novel error-prioritized experience replay (PER). We also show that our model performs well on both de novo and scaffold-based schemes. Our model produces molecules of higher quality compared to existing methods while achieving state-of-the-art performance on two of three targets in terms of the docking scores of the generated molecules. We further show with ablation studies that our method, predictive error-PER (FREED(PE)), significantly improves the model performance.

        ----

        ## [606] Proxy Convexity: A Unified Framework for the Analysis of Neural Networks Trained by Gradient Descent

        **Authors**: *Spencer Frei, Quanquan Gu*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/42299f06ee419aa5d9d07798b56779e2-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/42299f06ee419aa5d9d07798b56779e2-Abstract.html)

        **Abstract**:

        Although the optimization objectives for learning neural networks are highly non-convex, gradient-based methods have been wildly successful at learning neural networks in practice. This juxtaposition has led to a number of recent studies on provable guarantees for neural networks trained by gradient descent. Unfortunately, the techniques in these works are often highly specific to the particular setup in each problem, making it difficult to generalize across different settings. To address this drawback in the literature, we propose a unified non-convex optimization framework for the analysis of neural network training. We introduce the notions of proxy convexity and proxy Polyak-Lojasiewicz (PL) inequalities, which are satisfied if the original objective function induces a proxy objective function that is implicitly minimized when using gradient methods. We show that stochastic gradient descent (SGD) on objectives satisfying proxy convexity or the proxy PL inequality leads to efficient guarantees for proxy objective functions. We further show that many existing guarantees for neural networks trained by gradient descent can be unified through proxy convexity and proxy PL inequalities.

        ----

        ## [607] Covariance-Aware Private Mean Estimation Without Private Covariance Estimation

        **Authors**: *Gavin Brown, Marco Gaboardi, Adam D. Smith, Jonathan R. Ullman, Lydia Zakynthinou*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/42778ef0b5805a96f9511e20b5611fce-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/42778ef0b5805a96f9511e20b5611fce-Abstract.html)

        **Abstract**:

        We present two sample-efficient differentially private mean estimators for $d$-dimensional (sub)Gaussian distributions with unknown covariance. Informally, given $n \gtrsim d/\alpha^2$ samples from such a distribution with mean $\mu$ and covariance $\Sigma$, our estimators output $\tilde\mu$ such that $\| \tilde\mu - \mu \|_{\Sigma} \leq \alpha$, where $\| \cdot \|_{\Sigma}$ is the \emph{Mahalanobis distance}. All previous estimators with the same guarantee either require strong a priori bounds on the covariance matrix or require $\Omega(d^{3/2})$ samples.     Each of our estimators is based on a simple, general approach to designing differentially private mechanisms, but with novel technical steps to make the estimator private and sample-efficient. Our first estimator samples a point with approximately maximum Tukey depth using the exponential mechanism, but restricted to the set of points of large Tukey depth. Proving that this mechanism is private requires a novel analysis. Our second estimator perturbs the empirical mean of the data set with noise calibrated to the empirical covariance. Only the mean is released, however; the covariance is only used internally. Its sample complexity guarantees hold more generally for subgaussian distributions, albeit with a slightly worse dependence on the privacy parameter. For both estimators, careful preprocessing of the data is required to satisfy differential privacy.

        ----

        ## [608] Label consistency in overfitted generalized $k$-means

        **Authors**: *Linfan Zhang, Arash A. Amini*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/427e3427c5f38a41bb9cb26525b22fba-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/427e3427c5f38a41bb9cb26525b22fba-Abstract.html)

        **Abstract**:

        We provide theoretical guarantees for label consistency in generalized $k$-means problems, with an emphasis on the overfitted case where the number of clusters used by the algorithm is more than the ground truth. We provide conditions under which the estimated labels are close to a refinement of the true cluster labels. We consider both exact and approximate recovery of the labels. Our results hold for any constant-factor approximation to the $k$-means problem. The results are also model-free and only based on bounds on the maximum or average distance of the data points to the true cluster centers. These centers themselves are loosely defined and can be taken to be any set of points for which the aforementioned distances can be controlled. We show the usefulness of the results with applications to some manifold clustering problems.

        ----

        ## [609] Open-set Label Noise Can Improve Robustness Against Inherent Label Noise

        **Authors**: *Hongxin Wei, Lue Tao, Renchunzi Xie, Bo An*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/428fca9bc1921c25c5121f9da7815cde-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/428fca9bc1921c25c5121f9da7815cde-Abstract.html)

        **Abstract**:

        Learning with noisy labels is a practically challenging problem in weakly supervised learning. In the existing literature, open-set noises are always considered to be poisonous for generalization, similar to closed-set noises. In this paper, we empirically show that open-set noisy labels can be non-toxic and even benefit the robustness against inherent noisy labels. Inspired by the observations, we propose a simple yet effective regularization by introducing Open-set samples with Dynamic Noisy Labels (ODNL) into training. With ODNL, the extra capacity of the neural network can be largely consumed in a way that does not interfere with learning patterns from clean data. Through the lens of SGD noise, we show that the noises induced by our method are random-direction, conflict-free and biased, which may help the model converge to a flat minimum with superior stability and enforce the model to produce conservative predictions on Out-of-Distribution instances. Extensive experimental results on benchmark datasets with various types of noisy labels demonstrate that the proposed method not only enhances the performance of many existing robust algorithms but also achieves significant improvement on Out-of-Distribution detection tasks even in the label noise setting.

        ----

        ## [610] The Complexity of Sparse Tensor PCA

        **Authors**: *Davin Choo, Tommaso d'Orsi*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/42a6845a557bef704ad8ac9cb4461d43-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/42a6845a557bef704ad8ac9cb4461d43-Abstract.html)

        **Abstract**:

        We study the problem of sparse tensor principal component analysis: given a tensor $\pmb Y = \pmb W + \lambda x^{\otimes p}$ with $\pmb W \in \otimes^p \mathbb{R}^n$ having i.i.d. Gaussian entries, the goal is to recover the $k$-sparse unit vector $x \in \mathbb{R}^n$. The model captures both sparse PCA (in its Wigner form)  and tensor PCA.For the highly sparse regime of $k \leq \sqrt{n}$, we present a family of algorithms that smoothly interpolates between a simple polynomial-time algorithm and the exponential-time exhaustive search algorithm. For any $1 \leq t \leq k$, our algorithms recovers the sparse vector for signal-to-noise ratio $\lambda \geq \tilde{\mathcal{O}} (\sqrt{t} \cdot (k/t)^{p/2})$ in time $\tilde{\mathcal{O}}(n^{p+t})$, capturing the state-of-the-art guarantees for the matrix settings (in both the polynomial-time and sub-exponential time regimes).Our results naturally extend to the case of $r$ distinct $k$-sparse signals with disjoint supports, with guarantees that are independent of the number of spikes. Even in the restricted case of sparse PCA, known algorithms only recover the sparse vectors for $\lambda \geq \tilde{\mathcal{O}}(k \cdot r)$ while our algorithms require $\lambda \geq \tilde{\mathcal{O}}(k)$.Finally, by analyzing the low-degree likelihood ratio, we complement these algorithmic results with rigorous evidence illustrating the trade-offs between signal-to-noise ratio and running time. This lower bound captures the known lower bounds for both sparse  PCA and tensor PCA. In this general model, we observe a more intricate three-way trade-off between the number of samples $n$, the sparsity $k$, and the tensor power $p$.

        ----

        ## [611] Learning to Elect

        **Authors**: *Cem Anil, Xuchan Bao*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/42d6c7d61481d1c21bd1635f59edae05-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/42d6c7d61481d1c21bd1635f59edae05-Abstract.html)

        **Abstract**:

        Voting systems have a wide range of applications including recommender systems, web search, product design and elections. Limited by the lack of general-purpose analytical tools, it is difficult to hand-engineer desirable voting rules for each use case. For this reason, it is appealing to automatically discover voting rules geared towards each scenario. In this paper, we show that set-input neural network architectures such as Set Transformers, fully-connected graph networks and DeepSets are both theoretically and empirically well-suited for learning voting rules. In particular, we show that these network models can not only mimic a number of existing voting rules to compelling accuracy --- both position-based (such as Plurality and Borda) and comparison-based (such as Kemeny, Copeland and Maximin) --- but also discover near-optimal voting rules that maximize different social welfare functions. Furthermore, the learned voting rules generalize well to different voter utility distributions and election sizes unseen during training.

        ----

        ## [612] KALE Flow: A Relaxed KL Gradient Flow for Probabilities with Disjoint Support

        **Authors**: *Pierre Glaser, Michael Arbel, Arthur Gretton*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/433a6ea5429d6d75f0be9bf9da26e24c-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/433a6ea5429d6d75f0be9bf9da26e24c-Abstract.html)

        **Abstract**:

        We study the gradient flow for a relaxed approximation to the Kullback-Leibler (KL) divergencebetween a moving source and a fixed target distribution.This approximation, termed theKALE (KL approximate lower-bound estimator), solves a regularized version ofthe Fenchel dual problem defining the KL over a restricted class of functions.When using a Reproducing Kernel Hilbert Space (RKHS) to define the functionclass, we show that the KALE continuously interpolates between the KL and theMaximum Mean Discrepancy (MMD). Like the MMD and other Integral ProbabilityMetrics, the KALE remains well defined for mutually singulardistributions. Nonetheless, the KALE inherits from the limiting KL a greater sensitivity to mismatch in the support of the distributions, compared with the MMD. These two properties make theKALE gradient flow particularly well suited when the target distribution is supported on a low-dimensional manifold. Under an assumption of sufficient smoothness of the trajectories, we show the global convergence of the KALE flow. We propose a particle implementation of the flow given initial samples from the source and the target distribution, which we use to empirically confirm the KALE's properties.

        ----

        ## [613] When Is Generalizable Reinforcement Learning Tractable?

        **Authors**: *Dhruv Malik, Yuanzhi Li, Pradeep Ravikumar*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/437d46a857214c997956eaf0e3b21a55-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/437d46a857214c997956eaf0e3b21a55-Abstract.html)

        **Abstract**:

        Agents trained by reinforcement learning (RL) often fail to generalize beyond the environment they were trained in, even when presented with new scenarios that seem similar to the training environment. We study the query complexity required to train RL agents that generalize to multiple environments. Intuitively, tractable generalization is only possible when the environments are similar or close in some sense. To capture this, we introduce Weak Proximity, a natural structural condition that requires the environments to have highly similar transition and reward functions and share a policy providing optimal value. Despite such shared structure, we prove that tractable generalization is impossible in the worst case. This holds even when each individual environment can be efficiently solved to obtain an optimal linear policy, and when the agent possesses a generative model. Our lower bound applies to the more complex task of representation learning for efficient generalization to multiple environments. On the positive side, we introduce Strong Proximity, a strengthened condition which we prove is sufficient for efficient generalization.

        ----

        ## [614] Relational Self-Attention: What's Missing in Attention for Video Understanding

        **Authors**: *Manjin Kim, Heeseung Kwon, Chunyu Wang, Suha Kwak, Minsu Cho*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/4392e631da381761421d5e1e0c3de25f-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/4392e631da381761421d5e1e0c3de25f-Abstract.html)

        **Abstract**:

        Convolution has been arguably the most important feature transform for modern neural networks, leading to the advance of deep learning.  Recent emergence of Transformer networks, which replace convolution layers with self-attention blocks,  has revealed the limitation of stationary convolution kernels and opened the door to the era of dynamic feature transforms. The existing dynamic transforms, including self-attention, however, are all limited for video understanding where correspondence relations in space and time, i.e., motion information, are crucial for effective representation. In this work, we introduce a relational feature transform, dubbed the relational self-attention (RSA), that leverages rich structures of spatio-temporal relations in videos by dynamically generating relational kernels and aggregating relational contexts. Our experiments and ablation studies show that the RSA network substantially outperforms convolution and self-attention counterparts, achieving the state of the art on the standard motion-centric benchmarks for video action recognition, such as Something-Something-V1&V2, Diving48, and FineGym.

        ----

        ## [615] Towards Enabling Meta-Learning from Target Models

        **Authors**: *Su Lu, Han-Jia Ye, Le Gan, De-Chuan Zhan*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/43baa6762fa81bb43b39c62553b2970d-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/43baa6762fa81bb43b39c62553b2970d-Abstract.html)

        **Abstract**:

        Meta-learning can extract an inductive bias from previous learning experience and assist the training of new tasks. It is often realized through optimizing a meta-model with the evaluation loss of task-specific solvers. Most existing algorithms sample non-overlapping $\mathit{support}$ sets and $\mathit{query}$ sets to train and evaluate the solvers respectively due to simplicity ($\mathcal{S}$/$\mathcal{Q}$ protocol). Different from $\mathcal{S}$/$\mathcal{Q}$ protocol, we can also evaluate a task-specific solver by comparing it to a target model $\mathcal{T}$, which is the optimal model for this task or a model that behaves well enough on this task ($\mathcal{S}$/$\mathcal{T}$ protocol). Although being short of research, $\mathcal{S}$/$\mathcal{T}$ protocol has unique advantages such as offering more informative supervision, but it is computationally expensive. This paper looks into this special evaluation method and takes a step towards putting it into practice. We find that with a small ratio of tasks armed with target models, classic meta-learning algorithms can be improved a lot without consuming many resources. We empirically verify the effectiveness of $\mathcal{S}$/$\mathcal{T}$ protocol in a typical application of meta-learning, $\mathit{i.e.}$, few-shot learning. In detail, after constructing target models by fine-tuning the pre-trained network on those hard tasks, we match the task-specific solvers and target models via knowledge distillation.

        ----

        ## [616] A Near-Optimal Algorithm for Debiasing Trained Machine Learning Models

        **Authors**: *Ibrahim M. Alabdulmohsin, Mario Lucic*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/43c656628a4a479e108ed86f7a28a010-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/43c656628a4a479e108ed86f7a28a010-Abstract.html)

        **Abstract**:

        We present a scalable post-processing algorithm for debiasing trained models, including deep neural networks (DNNs), which we prove to be near-optimal by bounding its excess Bayes risk.  We empirically validate its advantages on standard benchmark datasets across both classical algorithms as well as modern DNN architectures and demonstrate that it outperforms previous post-processing methods while performing on par with in-processing. In addition, we show that the proposed algorithm is particularly effective for models trained at scale where post-processing is a natural and practical choice.

        ----

        ## [617] GENESIS-V2: Inferring Unordered Object Representations without Iterative Refinement

        **Authors**: *Martin Engelcke, Oiwi Parker Jones, Ingmar Posner*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/43ec517d68b6edd3015b3edc9a11367b-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/43ec517d68b6edd3015b3edc9a11367b-Abstract.html)

        **Abstract**:

        Advances in unsupervised learning of object-representations have culminated in the development of a broad range of methods for unsupervised object segmentation and interpretable object-centric scene generation. These methods, however, are limited to simulated and real-world datasets with limited visual complexity. Moreover, object representations are often inferred using RNNs which do not scale well to large images or iterative refinement which avoids imposing an unnatural ordering on objects in an image but requires the a priori initialisation of a fixed number of object representations. In contrast to established paradigms, this work proposes an embedding-based approach in which embeddings of pixels are clustered in a differentiable fashion using a stochastic stick-breaking process. Similar to iterative refinement, this clustering procedure also leads to randomly ordered object representations, but without the need of initialising a fixed number of clusters a priori. This is used to develop a new model, GENESIS-v2, which can infer a variable number of object representations without using RNNs or iterative refinement. We show that GENESIS-v2 performs strongly in comparison to recent baselines in terms of unsupervised image segmentation and object-centric scene generation on established synthetic datasets as well as more complex real-world datasets.

        ----

        ## [618] How Data Augmentation affects Optimization for Linear Regression

        **Authors**: *Boris Hanin, Yi Sun*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/442b548e816f05640dec68f497ca38ac-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/442b548e816f05640dec68f497ca38ac-Abstract.html)

        **Abstract**:

        Though data augmentation has rapidly emerged as a key tool for optimization in modern machine learning, a clear picture of how augmentation schedules affect optimization and interact with optimization hyperparameters such as learning rate is nascent. In the spirit of classical convex optimization and recent work on implicit bias, the present work analyzes the effect of augmentation on optimization in the simple convex setting of linear regression with MSE loss.We find joint schedules for learning rate and data augmentation scheme under which augmented gradient descent provably converges and characterize the resulting minimum. Our results apply to arbitrary augmentation schemes, revealing complex interactions between learning rates and augmentations even in the convex setting. Our approach interprets augmented (S)GD as a stochastic optimization method for a time-varying sequence of proxy losses. This gives a unified way to analyze learning rate, batch size, and augmentations ranging from additive noise to random projections. From this perspective, our results, which also give rates of convergence, can be viewed as Monro-Robbins type conditions for augmented (S)GD.

        ----

        ## [619] An Exact Characterization of the Generalization Error for the Gibbs Algorithm

        **Authors**: *Gholamali Aminian, Yuheng Bu, Laura Toni, Miguel R. D. Rodrigues, Gregory W. Wornell*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/445e24b5f22cacb9d51a837c10e91a3f-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/445e24b5f22cacb9d51a837c10e91a3f-Abstract.html)

        **Abstract**:

        Various approaches have been developed to upper bound the generalization error of a supervised learning algorithm. However, existing bounds are often loose and lack of guarantees. As a result, they may fail to characterize the exact generalization ability of a learning algorithm.Our main contribution is an exact characterization of the expected generalization error of the well-known Gibbs algorithm (a.k.a. Gibbs posterior) using symmetrized KL information between the input training samples and the output hypothesis. Our result can be applied to tighten existing expected generalization error and PAC-Bayesian bounds. Our approach is versatile, as it also characterizes the generalization error of the Gibbs algorithm with data-dependent regularizer and that of the Gibbs algorithm in the asymptotic regime, where it converges to the empirical risk minimization algorithm. Of particular relevance, our results highlight the role the symmetrized KL information plays in controlling the generalization error of the Gibbs algorithm.

        ----

        ## [620] Subgaussian and Differentiable Importance Sampling for Off-Policy Evaluation and Learning

        **Authors**: *Alberto Maria Metelli, Alessio Russo, Marcello Restelli*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/4476b929e30dd0c4e8bdbcc82c6ba23a-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/4476b929e30dd0c4e8bdbcc82c6ba23a-Abstract.html)

        **Abstract**:

        Importance Sampling (IS) is a widely used building block for a large variety of off-policy estimation and learning algorithms. However, empirical and theoretical studies have progressively shown that vanilla IS leads to poor estimations whenever the behavioral and target policies are too dissimilar. In this paper, we analyze the theoretical properties of the IS estimator by deriving a novel anticoncentration bound that formalizes the intuition behind its undesired behavior. Then, we propose a new class of IS transformations, based on the notion of power mean. To the best of our knowledge, the resulting estimator is the first to achieve, under certain conditions, two key properties: (i) it displays a subgaussian concentration rate; (ii) it preserves the differentiability in the target distribution. Finally, we provide numerical simulations on both synthetic examples and contextual bandits, in comparison with off-policy evaluation and learning baselines.

        ----

        ## [621] Rethinking gradient sparsification as total error minimization

        **Authors**: *Atal Narayan Sahu, Aritra Dutta, Ahmed M. Abdelmoniem, Trambak Banerjee, Marco Canini, Panos Kalnis*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/447b0408b80078338810051bb38b177f-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/447b0408b80078338810051bb38b177f-Abstract.html)

        **Abstract**:

        Gradient compression is a widely-established remedy to tackle the communication bottleneck in distributed training of large deep neural networks (DNNs). Under the error-feedback framework, Top-$k$ sparsification, sometimes with $k$ as little as 0.1% of the gradient size, enables training to the same model quality as the uncompressed case for a similar iteration count. From the optimization perspective, we find that Top-$k$ is the communication-optimal sparsifier given a per-iteration $k$ element budget.We argue that to further the benefits of gradient sparsification, especially for DNNs, a different perspective is necessary — one that moves from per-iteration optimality to consider optimality for the entire training.We identify that the total error — the sum of the compression errors for all iterations — encapsulates sparsification throughout training. Then, we propose a communication complexity model that minimizes the total error under a communication budget for the entire training. We find that the hard-threshold sparsifier, a variant of the Top-$k$ sparsifier with $k$ determined by a constant hard-threshold, is the optimal sparsifier for this model. Motivated by this, we provide convex and non-convex convergence analyses for the hard-threshold sparsifier with error-feedback. We show that hard-threshold has the same asymptotic convergence and linear speedup property as SGD in both the case, and unlike with Top-$k$ sparsifier, has no impact due to data-heterogeneity. Our diverse experiments on various DNNs and a logistic regression model demonstrate that the hard-threshold sparsifier is more communication-efficient than Top-$k$.

        ----

        ## [622] Approximate optimization of convex functions with outlier noise

        **Authors**: *Anindya De, Sanjeev Khanna, Huan Li, MohammadHesam NikpeySalekde*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/44b422a6d1df1d47db5d50a8d0aaca5d-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/44b422a6d1df1d47db5d50a8d0aaca5d-Abstract.html)

        **Abstract**:

        We study the problem of minimizing a convex function given by a zeroth order oracle that is possibly corrupted by {\em outlier noise}. Specifically, we assume the function values at some points of the domain are corrupted arbitrarily by an adversary, with the only restriction being that the total volume of corrupted points is bounded. The goal then is to find a point close to the function's minimizer using access to the corrupted oracle.We first prove a lower bound result showing that, somewhat surprisingly, one cannot hope to approximate the minimizer {\em nearly as well} as one might expect, even if one is allowed {\em an unbounded number} of queries to the oracle. Complementing this negative result, we then develop an efficient algorithm that outputs a point close to the minimizer of the convex function, where the specific distance matches {\em exactly}, up to constant factors, the distance bound shown in our lower bound result.

        ----

        ## [623] Fair Classification with Adversarial Perturbations

        **Authors**: *L. Elisa Celis, Anay Mehrotra, Nisheeth K. Vishnoi*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/44e207aecc63505eb828d442de03f2e9-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/44e207aecc63505eb828d442de03f2e9-Abstract.html)

        **Abstract**:

        We study fair classification in the presence of an omniscient adversary that, given an $\eta$, is allowed to choose an arbitrary $\eta$-fraction of the training samples and arbitrarily perturb their protected attributes. The motivation comes from settings in which protected attributes can be incorrect due to strategic misreporting, malicious actors, or errors in imputation; and prior approaches that make stochastic or independence assumptions on errors may not satisfy their guarantees in this adversarial setting. Our main contribution is an optimization framework to learn fair classifiers in this adversarial setting that comes with provable guarantees on accuracy and fairness. Our framework works with multiple and non-binary protected attributes, is designed for the large class of linear-fractional fairness metrics, and can also handle perturbations besides protected attributes. We prove near-tightness of our framework's guarantees for natural hypothesis classes: no algorithm can have significantly better accuracy and any algorithm with better fairness must have lower accuracy. Empirically, we evaluate the classifiers produced by our framework for statistical rate on real-world and synthetic datasets for a family of adversaries.

        ----

        ## [624] Distributed Saddle-Point Problems Under Data Similarity

        **Authors**: *Aleksandr Beznosikov, Gesualdo Scutari, Alexander Rogozin, Alexander V. Gasnikov*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/44e65d3e9bc2f88b2b3d566de51a5381-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/44e65d3e9bc2f88b2b3d566de51a5381-Abstract.html)

        **Abstract**:

        We study solution methods for (strongly-)convex-(strongly)-concave Saddle-Point Problems (SPPs) over networks of two type--master/workers (thus centralized) architectures  and  mesh (thus decentralized) networks. The local functions at each node are assumed to be \textit{similar}, due to statistical data similarity or otherwise. We establish lower complexity bounds for a fairly general  class of algorithms solving the SPP. We show that   a given suboptimality $\epsilon>0$ is achieved over master/workers networks in $\Omega\big(\Delta\cdot  \delta/\mu\cdot \log (1/\varepsilon)\big)$ rounds of communications, where $\delta>0$ measures the degree of similarity of the local functions, $\mu$ is their strong convexity constant, and $\Delta$ is the diameter of the network. The lower communication complexity bound over mesh networks reads     $\Omega\big(1/{\sqrt{\rho}} \cdot  {\delta}/{\mu}\cdot\log (1/\varepsilon)\big)$, where $\rho$ is the (normalized) eigengap of the gossip matrix used for the communication between neighbouring nodes.  We then propose algorithms matching the lower bounds over either  types of networks (up to log-factors). We assess the effectiveness of the proposed algorithms on a robust regression  problem.

        ----

        ## [625] Combining Latent Space and Structured Kernels for Bayesian Optimization over Combinatorial Spaces

        **Authors**: *Aryan Deshwal, Janardhan Rao Doppa*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/44e76e99b5e194377e955b13fb12f630-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/44e76e99b5e194377e955b13fb12f630-Abstract.html)

        **Abstract**:

        We consider the problem of optimizing combinatorial spaces (e.g., sequences, trees, and graphs) using expensive black-box function evaluations. For example, optimizing molecules for drug design using physical lab experiments. Bayesian optimization (BO) is an efficient framework for solving such problems by intelligently selecting the inputs with high utility guided by a learned surrogate model. A recent BO approach for combinatorial spaces is through a reduction to BO over continuous spaces by learning a latent representation of structures using deep generative models (DGMs). The selected input from the continuous space is decoded into a discrete structure for performing function evaluation. However, the surrogate model over the latent space only uses the information learned by the DGM, which may not have the desired inductive bias to approximate the target black-box function. To overcome this drawback, this paper proposes a principled approach referred as LADDER. The key idea is to define a novel structure-coupled kernel that explicitly integrates the structural information from decoded structures with the learned latent space representation for better surrogate modeling. Our experiments on real-world benchmarks show that LADDER significantly improves over the BO over latent space method, and performs better or similar to state-of-the-art methods.

        ----

        ## [626] Gradual Domain Adaptation without Indexed Intermediate Domains

        **Authors**: *Hong-You Chen, Wei-Lun Chao*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/45017f6511f91be700fda3d118034994-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/45017f6511f91be700fda3d118034994-Abstract.html)

        **Abstract**:

        The effectiveness of unsupervised domain adaptation degrades when there is a large discrepancy between the source and target domains. Gradual domain adaption (GDA) is one promising way to mitigate such an issue, by leveraging additional unlabeled data that gradually shift from the source to the target. Through sequentially adapting the model along the "indexed" intermediate domains, GDA substantially improves the overall adaptation performance. In practice, however, the extra unlabeled data may not be separated into intermediate domains and indexed properly, limiting the applicability of GDA. In this paper, we investigate how to discover the sequence of intermediate domains when it is not already available. Concretely, we propose a coarse-to-fine framework, which starts with a coarse domain discovery step via progressive domain discriminator training. This coarse domain sequence then undergoes a fine indexing step via a novel cycle-consistency loss, which encourages the next intermediate domain to preserve sufficient discriminative knowledge of the current intermediate domain. The resulting domain sequence can then be used by a GDA algorithm. On benchmark data sets of GDA, we show that our approach, which we name Intermediate DOmain Labeler (IDOL), can lead to comparable or even better adaptation performance compared to the pre-defined domain sequence, making GDA more applicable and robust to the quality of domain sequences. Codes are available at https://github.com/hongyouc/IDOL.

        ----

        ## [627] K-level Reasoning for Zero-Shot Coordination in Hanabi

        **Authors**: *Brandon Cui, Hengyuan Hu, Luis Pineda, Jakob N. Foerster*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/4547dff5fd7604f18c8ee32cf3da41d7-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/4547dff5fd7604f18c8ee32cf3da41d7-Abstract.html)

        **Abstract**:

        The standard problem setting in cooperative multi-agent settings is \emph{self-play} (SP), where the goal is to train a \emph{team} of agents that works well together.     However, optimal SP policies commonly contain arbitrary conventions  (``handshakes'') and are not compatible with other, independently trained agents or humans.     This latter desiderata was recently formalized by \cite{Hu2020-OtherPlay} as the \emph{zero-shot coordination} (ZSC) setting and partially addressed with their \emph{Other-Play} (OP) algorithm, which showed improved ZSC and human-AI performance in the card game Hanabi.     OP assumes access to the symmetries of the environment and prevents agents from breaking these in a mutually \emph{incompatible} way during training. However, as the authors point out, discovering symmetries for a given environment is a computationally hard problem.    Instead, we show that through a simple adaption of k-level reasoning (KLR) \cite{Costa-Gomes2006-K-level}, synchronously training all levels, we can obtain competitive ZSC and ad-hoc teamplay performance in Hanabi, including when paired with a human-like proxy bot. We also introduce a new method, synchronous-k-level reasoning with a best response (SyKLRBR), which further improves performance on our synchronous KLR by co-training a best response.

        ----

        ## [628] Learning Markov State Abstractions for Deep Reinforcement Learning

        **Authors**: *Cameron Allen, Neev Parikh, Omer Gottesman, George Konidaris*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/454cecc4829279e64d624cd8a8c9ddf1-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/454cecc4829279e64d624cd8a8c9ddf1-Abstract.html)

        **Abstract**:

        A fundamental assumption of reinforcement learning in Markov decision processes (MDPs) is that the relevant decision process is, in fact, Markov. However, when MDPs have rich observations, agents typically learn by way of an abstract state representation, and such representations are not guaranteed to preserve the Markov property. We introduce a novel set of conditions and prove that they are sufficient for learning a Markov abstract state representation. We then describe a practical training procedure that combines inverse model estimation and temporal contrastive learning to learn an abstraction that approximately satisfies these conditions. Our novel training objective is compatible with both online and offline training: it does not require a reward signal, but agents can capitalize on reward information when available. We empirically evaluate our approach on a visual gridworld domain and a set of continuous control benchmarks. Our approach learns representations that capture the underlying structure of the domain and lead to improved sample efficiency over state-of-the-art deep reinforcement learning with visual features---often matching or exceeding the performance achieved with hand-designed compact state information.

        ----

        ## [629] Towards Deeper Deep Reinforcement Learning with Spectral Normalization

        **Authors**: *Johan Bjorck, Carla P. Gomes, Kilian Q. Weinberger*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/4588e674d3f0faf985047d4c3f13ed0d-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/4588e674d3f0faf985047d4c3f13ed0d-Abstract.html)

        **Abstract**:

        In computer vision and natural language processing, innovations in model architecture that increase model capacity have reliably translated into gains in performance. In stark contrast with this trend, state-of-the-art reinforcement learning (RL) algorithms often use small MLPs, and gains in performance typically originate from algorithmic innovations. It is natural to hypothesize that small datasets in RL necessitate simple models to avoid overfitting; however, this hypothesis is untested. In this paper we investigate how RL agents are affected by exchanging the small MLPs with larger modern networks with skip connections and normalization, focusing specifically on actor-critic algorithms. We empirically verify that naively adopting such architectures leads to instabilities and poor performance, likely contributing to the popularity of simple models in practice. However, we show that dataset size is not the limiting factor, and instead argue that instability from taking gradients through the critic is the culprit. We demonstrate that spectral normalization (SN) can mitigate this issue and enable stable training with large modern architectures. After smoothing with SN, larger models yield significant performance improvements --- suggesting that more ``easy'' gains may be had by focusing on model architectures in addition to algorithmic innovations.

        ----

        ## [630] Functionally Regionalized Knowledge Transfer for Low-resource Drug Discovery

        **Authors**: *Huaxiu Yao, Ying Wei, Long-Kai Huang, Ding Xue, Junzhou Huang, Zhenhui Li*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/459a4ddcb586f24efd9395aa7662bc7c-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/459a4ddcb586f24efd9395aa7662bc7c-Abstract.html)

        **Abstract**:

        More recently, there has been a surge of interest in employing machine learning approaches to expedite the drug discovery process where virtual screening for hit discovery and ADMET prediction for lead optimization play essential roles. One of the main obstacles to the wide success of machine learning approaches in these two tasks is that the number of compounds labeled with activities or ADMET properties is too small to build an effective predictive model. This paper seeks to remedy the problem by transferring the knowledge from previous assays, namely in-vivo experiments, by different laboratories and against various target proteins. To accommodate these wildly different assays and capture the similarity between assays, we propose a functional rationalized meta-learning algorithm FRML for such knowledge transfer. FRML constructs the predictive model with layers of neural sub-networks or so-called functional regions. Building on this, FRML shares an initialization for the weights of the predictive model across all assays, while customizes it to each assay with a region localization network choosing the pertinent regions. The compositionality of the model improves the capacity of generalization to various and even out-of-distribution tasks. Empirical results on both virtual screening and ADMET prediction validate the superiority of FRML over state-of-the-art baselines powered with interpretability in assay relationship.

        ----

        ## [631] Memory-Efficient Approximation Algorithms for Max-k-Cut and Correlation Clustering

        **Authors**: *Nimita Shinde, Vishnu Narayanan, James Saunderson*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/45c166d697d65080d54501403b433256-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/45c166d697d65080d54501403b433256-Abstract.html)

        **Abstract**:

        Max-k-Cut and correlation clustering are fundamental graph partitioning problems. For a graph $G=(V,E)$ with $n$ vertices, the methods with the best approximation guarantees for Max-k-Cut and the Max-Agree variant of correlation clustering involve solving SDPs with $\mathcal{O}(n^2)$ constraints and variables. Large-scale instances of SDPs, thus, present a memory bottleneck. In this paper, we develop simple polynomial-time Gaussian sampling-based algorithms for these two problems that use $\mathcal{O}(n+|E|)$ memory and nearly achieve the best existing approximation guarantees. For dense graphs arriving in a stream, we eliminate the dependence on $|E|$ in the storage complexity at the cost of a slightly worse approximation ratio by combining our approach with sparsification.

        ----

        ## [632] Panoptic 3D Scene Reconstruction From a Single RGB Image

        **Authors**: *Manuel Dahnert, Ji Hou, Matthias Nießner, Angela Dai*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/46031b3d04dc90994ca317a7c55c4289-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/46031b3d04dc90994ca317a7c55c4289-Abstract.html)

        **Abstract**:

        Richly segmented 3D scene reconstructions are an integral basis for many high-level scene understanding tasks, such as for robotics, motion planning, or augmented reality. Existing works in 3D perception from a single RGB image tend to focus on geometric reconstruction only, or geometric reconstruction with semantic  segmentation or instance segmentation.Inspired by 2D panoptic segmentation, we propose to unify the tasks of geometric reconstruction, 3D semantic segmentation, and 3D instance segmentation into the task of panoptic 3D scene reconstruction -- from a single RGB image, predicting the complete geometric reconstruction of the scene in the camera frustum of the image, along with semantic and instance segmentations.We propose a new approach for holistic 3D scene understanding from a single RGB image which learns to lift and propagate 2D features from an input image to a 3D volumetric scene representation.Our panoptic 3D reconstruction metric evaluates both geometric reconstruction quality as well as panoptic segmentation.Our experiments demonstrate that our approach for panoptic 3D scene reconstruction outperforms alternative approaches for this task.

        ----

        ## [633] Measuring Generalization with Optimal Transport

        **Authors**: *Ching-Yao Chuang, Youssef Mroueh, Kristjan H. Greenewald, Antonio Torralba, Stefanie Jegelka*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/4607f7fff0dce694258e1c637512aa9d-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/4607f7fff0dce694258e1c637512aa9d-Abstract.html)

        **Abstract**:

        Understanding the generalization of deep neural networks is one of the most important tasks in deep learning. Although much progress has been made, theoretical error bounds still often behave disparately from empirical observations. In this work, we develop margin-based generalization bounds, where the margins are normalized with optimal transport costs between independent random subsets sampled from the training distribution. In particular, the optimal transport cost can be interpreted as a generalization of variance which captures the structural properties of the learned feature space. Our bounds robustly predict the generalization error, given training data and network parameters, on large scale datasets. Theoretically, we demonstrate that the concentration and separation of features play crucial roles in generalization, supporting empirical results in the literature.

        ----

        ## [634] Uniform Concentration Bounds toward a Unified Framework for Robust Clustering

        **Authors**: *Debolina Paul, Saptarshi Chakraborty, Swagatam Das, Jason Q. Xu*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/460b491b917d4185ed1f5be97229721a-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/460b491b917d4185ed1f5be97229721a-Abstract.html)

        **Abstract**:

        Recent advances in center-based clustering continue to improve upon the drawbacks of Lloyd's celebrated $k$-means algorithm over $60$ years after its introduction. Various methods seek to address poor local minima, sensitivity to outliers, and data that are not well-suited to Euclidean measures of fit, but many are supported largely empirically. Moreover, combining such approaches in a piecemeal manner can result in ad hoc methods, and the limited theoretical results supporting each individual contribution may no longer hold. Toward addressing these issues in a principled way, this paper proposes a cohesive robust framework for center-based clustering under a general class of dissimilarity measures. In particular, we present a rigorous theoretical treatment within a Median-of-Means (MoM) estimation framework, showing that it subsumes several popular $k$-means variants. In addition to unifying existing methods, we derive uniform concentration bounds that complete their analyses, and bridge these results to the MoM framework via Dudley's chaining arguments. Importantly, we neither require any assumptions on the distribution of the outlying observations nor on the relative number of observations $n$ to features $p$. We establish strong consistency and an error rate of $O(n^{-1/2})$ under mild conditions, surpassing the best-known results in the literature. The methods are empirically validated thoroughly on real and synthetic datasets.

        ----

        ## [635] Learning Signal-Agnostic Manifolds of Neural Fields

        **Authors**: *Yilun Du, Katie Collins, Josh Tenenbaum, Vincent Sitzmann*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/4639475d6782a08c1e964f9a4329a254-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/4639475d6782a08c1e964f9a4329a254-Abstract.html)

        **Abstract**:

        Deep neural networks have been used widely to learn the latent structure of datasets, across modalities such as images, shapes, and audio signals. However, existing models are generally modality-dependent, requiring custom architectures and objectives to process different classes of signals. We leverage neural fields to capture the underlying structure in image, shape, audio and cross-modal audiovisual domains in a modality-independent manner. We cast our task as one of learning a manifold, where we aim to infer a low-dimensional, locally linear subspace in which our data resides. By enforcing coverage of the manifold, local linearity, and local isometry, our model -- dubbed GEM -- learns to capture the underlying structure of datasets across modalities. We can then travel along linear regions of our manifold to obtain perceptually consistent interpolations between samples, and can further use GEM to recover points on our manifold and glean not only diverse completions of input images, but cross-modal hallucinations of audio or image signals.  Finally, we show that by walking across the underlying manifold of GEM, we may generate new samples in our signal domains.

        ----

        ## [636] Low-dimensional Structure in the Space of Language Representations is Reflected in Brain Responses

        **Authors**: *Richard Antonello, Javier S. Turek, Vy Ai Vo, Alexander Huth*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/464074179972cbbd75a39abc6954cd12-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/464074179972cbbd75a39abc6954cd12-Abstract.html)

        **Abstract**:

        How related are the representations learned by neural language models, translation models, and language tagging tasks? We answer this question by adapting an encoder-decoder transfer learning method from computer vision to investigate the structure among 100 different feature spaces extracted from hidden representations of various networks trained on language tasks.This method reveals a low-dimensional structure where language models and translation models smoothly interpolate between word embeddings, syntactic and semantic tasks, and future word embeddings. We call this low-dimensional structure a language representation embedding because it encodes the relationships between representations needed to process language for a variety of NLP tasks. We find that this representation embedding can predict how well each individual feature space maps to human brain responses to natural language stimuli recorded using fMRI. Additionally, we find that the principal dimension of this structure can be used to create a metric which highlights the brain's natural language processing hierarchy. This suggests that the embedding captures some part of the brain's natural language representation structure.

        ----

        ## [637] On the Suboptimality of Thompson Sampling in High Dimensions

        **Authors**: *Raymond Zhang, Richard Combes*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/46489c17893dfdcf028883202cefd6d1-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/46489c17893dfdcf028883202cefd6d1-Abstract.html)

        **Abstract**:

        In this paper we consider Thompson Sampling for combinatorial semi-bandits. We demonstrate that, perhaps surprisingly, Thompson Sampling is sub-optimal for this problem in the sense that its regret scales exponentially in the ambient dimension, and its minimax regret scales almost linearly. This phenomenon occurs under a wide variety of assumptions including both non-linear and linear reward functions in the Bernoulli distribution setting. We also show that including a fixed amount of forced exploration to Thompson Sampling does not alleviate the problem. We complement our theoretical results with numerical results and show that in practice Thompson Sampling indeed can perform very poorly in some high dimension situations.

        ----

        ## [638] Learning Debiased and Disentangled Representations for Semantic Segmentation

        **Authors**: *Sanghyeok Chu, Dongwan Kim, Bohyung Han*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/465636eb4a7ff4b267f3b765d07a02da-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/465636eb4a7ff4b267f3b765d07a02da-Abstract.html)

        **Abstract**:

        Deep neural networks are susceptible to learn biased models with entangled feature representations, which may lead to subpar performances on various downstream tasks. This is particularly true for under-represented classes, where a lack of diversity in the data exacerbates the tendency. This limitation has been addressed mostly in classification tasks, but there is little study on additional challenges that may appear in more complex dense prediction problems including semantic segmentation. To this end, we propose a model-agnostic and stochastic training scheme for semantic segmentation, which facilitates the learning of debiased and disentangled representations. For each class, we first extract class-specific information from the highly entangled feature map. Then, information related to a randomly sampled class is suppressed by a feature selection process in the feature space. By randomly eliminating certain class information in each training iteration, we effectively reduce feature dependencies among classes, and the model is able to learn more debiased and disentangled feature representations. Models trained with our approach demonstrate strong results on multiple semantic segmentation benchmarks, with especially notable performance gains on under-represented classes.

        ----

        ## [639] Diversity Matters When Learning From Ensembles

        **Authors**: *Giung Nam, Jongmin Yoon, Yoonho Lee, Juho Lee*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/466473650870501e3600d9a1b4ee5d44-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/466473650870501e3600d9a1b4ee5d44-Abstract.html)

        **Abstract**:

        Deep ensembles excel in large-scale image classification tasks both in terms of prediction accuracy and calibration. Despite being simple to train, the computation and memory cost of deep ensembles limits their practicability. While some recent works propose to distill an ensemble model into a single model to reduce such costs, there is still a performance gap between the ensemble and distilled models. We propose a simple approach for reducing this gap, i.e., making the distilled performance close to the full ensemble. Our key assumption is that a distilled model should absorb as much function diversity inside the ensemble as possible. We first empirically show that the typical distillation procedure does not effectively transfer such diversity, especially for complex models that achieve near-zero training error. To fix this, we propose a perturbation strategy for distillation that reveals diversity by seeking inputs for which ensemble member outputs disagree. We empirically show that a model distilled with such perturbed samples indeed exhibits enhanced diversity, leading to improved performance.

        ----

        ## [640] Locally Valid and Discriminative Prediction Intervals for Deep Learning Models

        **Authors**: *Zhen Lin, Shubhendu Trivedi, Jimeng Sun*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/46c7cb50b373877fb2f8d5c4517bb969-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/46c7cb50b373877fb2f8d5c4517bb969-Abstract.html)

        **Abstract**:

        Crucial for building trust in deep learning models for critical real-world applications is efficient and theoretically sound uncertainty quantification, a task that continues to be challenging. Useful uncertainty information is expected to have two key properties: It should be valid (guaranteeing coverage) and discriminative (more uncertain when the expected risk is high). Moreover, when combined with deep learning (DL) methods, it should be scalable and affect the DL model performance minimally. Most existing Bayesian methods lack frequentist coverage guarantees and usually affect model performance. The few available frequentist methods are rarely discriminative and/or violate coverage guarantees due to unrealistic assumptions. Moreover, many methods are expensive or require substantial modifications to the base neural network. Building upon recent advances in conformal prediction [13, 33] and leveraging the classical idea of kernel regression, we propose Locally Valid and Discriminative prediction intervals (LVD), a simple, efficient, and lightweight method to construct discriminative prediction intervals (PIs) for almost any DL model. With no assumptions on the data distribution, such PIs also offer finite-sample local coverage guarantees (contrasted to the simpler marginal coverage). We empirically verify, using diverse datasets, that besides being the only locally valid method for DL, LVD also exceeds or matches the performance (including coverage rate and prediction accuracy) of existing uncertainty quantification methods, while offering additional benefits in scalability and flexibility.

        ----

        ## [641] Personalized Federated Learning With Gaussian Processes

        **Authors**: *Idan Achituve, Aviv Shamsian, Aviv Navon, Gal Chechik, Ethan Fetaya*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/46d0671dd4117ea366031f87f3aa0093-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/46d0671dd4117ea366031f87f3aa0093-Abstract.html)

        **Abstract**:

        Federated learning aims to learn a global model that performs well on client devices with limited cross-client communication. Personalized federated learning (PFL) further extends this setup to handle data heterogeneity between clients by learning personalized models. A key challenge in this setting is to learn effectively across clients even though each client has unique data that is often limited in size. Here we present pFedGP, a solution to PFL that is based on Gaussian processes (GPs) with deep kernel learning. GPs are highly expressive models that work well in the low data regime due to their Bayesian nature.However, applying GPs to PFL raises multiple challenges. Mainly, GPs performance depends heavily on access to a good kernel function, and learning a kernel requires a large training set. Therefore, we propose learning a shared kernel function across all clients, parameterized by a neural network, with a personal GP classifier for each client. We further extend pFedGP to include inducing points using two novel methods, the first helps to improve generalization in the low data regime and the second reduces the computational cost. We derive a PAC-Bayes generalization bound on novel clients and empirically show that it gives non-vacuous guarantees. Extensive experiments on standard PFL benchmarks with CIFAR-10, CIFAR-100, and CINIC-10, and on a new setup of learning under input noise show that pFedGP achieves well-calibrated predictions while significantly outperforming baseline methods, reaching up to 21% in accuracy gain.

        ----

        ## [642] Risk Bounds for Over-parameterized Maximum Margin Classification on Sub-Gaussian Mixtures

        **Authors**: *Yuan Cao, Quanquan Gu, Mikhail Belkin*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/46e0eae7d5217c79c3ef6b4c212b8c6f-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/46e0eae7d5217c79c3ef6b4c212b8c6f-Abstract.html)

        **Abstract**:

        Modern machine learning systems such as deep neural networks are often highly over-parameterized so that they can fit the noisy training data exactly, yet they can still achieve small test errors in practice. In this paper, we study this "benign overfitting" phenomenon of the maximum margin classifier for linear classification problems. Specifically, we consider data generated from sub-Gaussian mixtures, and provide a tight risk bound for the maximum margin linear classifier in the over-parameterized setting. Our results precisely characterize the condition under which benign overfitting can occur in linear classification problems, and improve on previous work. They also have direct implications for over-parameterized logistic regression.

        ----

        ## [643] Implicit SVD for Graph Representation Learning

        **Authors**: *Sami Abu-El-Haija, Hesham Mostafa, Marcel Nassar, Valentino Crespi, Greg Ver Steeg, Aram Galstyan*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/46fc943ecd56441056a560ba37d0b9e8-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/46fc943ecd56441056a560ba37d0b9e8-Abstract.html)

        **Abstract**:

        Recent improvements in the performance of state-of-the-art (SOTA) methods for Graph Representational Learning (GRL) have come at the cost of significant computational resource requirements for training, e.g., for calculating gradients via backprop over many data epochs. Meanwhile, Singular Value Decomposition (SVD) can find closed-form solutions to convex problems, using merely a handful of epochs. In this paper, we make GRL more computationally tractable for those with modest hardware. We design a framework that computes SVD of *implicitly* defined matrices, and apply this framework to several GRL tasks. For each task, we derive first-order approximation of a SOTA model, where we design (expensive-to-store) matrix $\mathbf{M}$ and train the model, in closed-form, via SVD of $\mathbf{M}$, without calculating entries of $\mathbf{M}$. By converging to a unique point in one step, and without calculating gradients, our models show competitive empirical test performance over various graphs such as article citation and biological interaction networks. More importantly, SVD can initialize a deeper model, that is architected to be non-linear almost everywhere, though behaves linearly when its parameters reside on a hyperplane, onto which SVD initializes. The deeper model can then be fine-tuned within only a few epochs. Overall, our algorithm trains hundreds of times faster than state-of-the-art methods, while competing on test empirical performance. We open-source our implementation at: https://github.com/samihaija/isvd

        ----

        ## [644] Offline Model-based Adaptable Policy Learning

        **Authors**: *Xiong-Hui Chen, Yang Yu, Qingyang Li, Fan-Ming Luo, Zhiwei (Tony) Qin, Wenjie Shang, Jieping Ye*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/470e7a4f017a5476afb7eeb3f8b96f9b-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/470e7a4f017a5476afb7eeb3f8b96f9b-Abstract.html)

        **Abstract**:

        In reinforcement learning, a promising direction to avoid online trial-and-error costs is learning from an offline dataset. Current offline reinforcement learning methods commonly learn in the policy space constrained to in-support regions by the offline dataset, in order to ensure the robustness of the outcome policies. Such constraints, however, also limit the potential of the outcome policies. In this paper, to release the potential of offline policy learning, we investigate the decision-making problems in out-of-support regions directly and propose offline Model-based Adaptable Policy LEarning (MAPLE). By this approach, instead of learning in in-support regions, we learn an adaptable policy that can adapt its behavior in out-of-support regions when deployed. We conduct experiments on MuJoCo controlling tasks with offline datasets. The results show that the proposed method can make robust decisions in out-of-support regions and achieve better performance than SOTA algorithms.

        ----

        ## [645] Multilingual Pre-training with Universal Dependency Learning

        **Authors**: *Kailai Sun, Zuchao Li, Hai Zhao*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/473803f0f2ebd77d83ee60daaa61f381-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/473803f0f2ebd77d83ee60daaa61f381-Abstract.html)

        **Abstract**:

        The pre-trained language model (PrLM) demonstrates domination in downstream natural language processing tasks, in which multilingual PrLM takes advantage of language universality to alleviate the issue of limited resources for low-resource languages. Despite its successes, the performance of multilingual PrLM is still unsatisfactory, when multilingual PrLMs only focus on plain text and ignore obvious universal linguistic structure clues. Existing PrLMs have shown that monolingual linguistic structure knowledge may bring about better performance. Thus we propose a novel multilingual PrLM that supports both explicit universal dependency parsing and implicit language modeling. Syntax in terms of universal dependency parse serves as not only pre-training objective but also learned representation in our model, which brings unprecedented PrLM interpretability and convenience in downstream task use. Our model outperforms two popular multilingual PrLM, multilingual-BERT and XLM-R, on cross-lingual natural language understanding (NLU) benchmarks and linguistic structure parsing datasets, demonstrating the effectiveness and stronger cross-lingual modeling capabilities of our approach.

        ----

        ## [646] Parameter-free HE-friendly Logistic Regression

        **Authors**: *Junyoung Byun, Woojin Lee, Jaewook Lee*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/477bdb55b231264bb53a7942fd84254d-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/477bdb55b231264bb53a7942fd84254d-Abstract.html)

        **Abstract**:

        Privacy in machine learning has been widely recognized as an essential ethical and legal issue, because the data used for machine learning may contain sensitive information. Homomorphic encryption has recently attracted attention as a key solution to preserve privacy in machine learning applications. However, current approaches on the training of encrypted machine learning have relied heavily on hyperparameter selection, which should be avoided owing to the extreme difficulty of conducting validation on encrypted data. In this study, we propose an effective privacy-preserving logistic regression method that is free from the approximation of the sigmoid function and hyperparameter selection. In our framework, a logistic regression model can be transformed into the corresponding ridge regression for the logit function. We provide a theoretical background for our framework by suggesting a new generalization error bound on the encrypted data. Experiments on various real-world data show that our framework achieves better classification results while reducing latency by $\sim68\%$, compared to the previous models.

        ----

        ## [647] Active clustering for labeling training data

        **Authors**: *Quentin Lutz, Elie de Panafieu, Maya Stein, Alex Scott*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/47841cc9e552bd5c40164db7073b817b-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/47841cc9e552bd5c40164db7073b817b-Abstract.html)

        **Abstract**:

        Gathering training data is a key step of any supervised learning task, and it is both critical and expensive. Critical, because the quantity and quality of the training data has a high impact on the performance of the learned function. Expensive, because most practical cases rely on humans-in-the-loop to label the data. The process of determining the correct labels is much more expensive than comparing two items to see whether they belong to the same class. Thus motivated, we propose a setting for training data gathering where the human experts perform the comparatively cheap task of answering pairwise queries, and the computer groups the items into classes (which can be labeled cheaply at the very end of the process). Given the items, we consider two random models for the classes: one where the set partition they form is drawn uniformly, the other one where each item chooses its class independently following a fixed distribution. In the first model, we characterize the algorithms that minimize the average number of queries required to cluster the items and analyze their complexity. In the second model, we analyze a specific algorithm family, propose as a conjecture that they reach the minimum average number of queries and compare their performance to a random approach. We also propose solutions to handle errors or inconsistencies in the experts' answers.

        ----

        ## [648] Exploring Social Posterior Collapse in Variational Autoencoder for Interaction Modeling

        **Authors**: *Chen Tang, Wei Zhan, Masayoshi Tomizuka*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/47951a40efc0d2f7da8ff1ecbfde80f4-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/47951a40efc0d2f7da8ff1ecbfde80f4-Abstract.html)

        **Abstract**:

        Multi-agent behavior modeling and trajectory forecasting are crucial for the safe navigation of autonomous agents in interactive scenarios. Variational Autoencoder (VAE) has been widely applied in multi-agent interaction modeling to generate diverse behavior and learn a low-dimensional representation for interacting systems. However, existing literature did not formally discuss if a VAE-based model can properly encode interaction into its latent space. In this work, we argue that one of the typical formulations of VAEs in multi-agent modeling suffers from an issue we refer to as social posterior collapse, i.e., the model is prone to ignoring historical social context when predicting the future trajectory of an agent. It could cause significant prediction errors and poor generalization performance. We analyze the reason behind this under-explored phenomenon and propose several measures to tackle it. Afterward, we implement the proposed framework and experiment on real-world datasets for multi-agent trajectory prediction. In particular, we propose a novel sparse graph attention message-passing (sparse-GAMP) layer, which helps us detect social posterior collapse in our experiments. In the experiments, we verify that social posterior collapse indeed occurs. Also, the proposed measures are effective in alleviating the issue. As a result, the model attains better generalization performance when historical social context is informative for prediction.

        ----

        ## [649] Ensembling Graph Predictions for AMR Parsing

        **Authors**: *Thanh Lam Hoang, Gabriele Picco, Yufang Hou, Young-Suk Lee, Lam M. Nguyen, Dzung T. Phan, Vanessa López, Ramón Fernandez Astudillo*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/479b4864e55e12e0fb411eadb115c095-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/479b4864e55e12e0fb411eadb115c095-Abstract.html)

        **Abstract**:

        In many machine learning tasks, models are trained to predict structure data such as graphs. For example, in natural language processing, it is very common to parse texts into dependency trees or abstract meaning representation (AMR) graphs. On the other hand, ensemble methods combine predictions from multiple models to create a new one that is more robust and accurate than individual predictions. In the literature, there are many ensembling techniques proposed for classification or regression problems, however, ensemble graph prediction has not been studied thoroughly. In this work, we formalize this problem as mining the largest graph that is the most supported by a collection of graph predictions. As the problem is NP-Hard, we propose an efficient heuristic algorithm to approximate the optimal solution. To validate our approach, we carried out experiments in AMR parsing problems. The experimental results demonstrate that the proposed approach can combine the strength of state-of-the-art AMR parsers to create new predictions that are more accurate than any individual models in five standard benchmark datasets.

        ----

        ## [650] On the interplay between data structure and loss function in classification problems

        **Authors**: *Stéphane d'Ascoli, Marylou Gabrié, Levent Sagun, Giulio Biroli*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/47a5feca4ce02883a5643e295c7ce6cd-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/47a5feca4ce02883a5643e295c7ce6cd-Abstract.html)

        **Abstract**:

        One of the central features of modern machine learning models, including deep neural networks, is their generalization ability on structured data in the over-parametrized regime. In this work, we consider an analytically solvable setup to investigate how properties of data impact learning in classification problems, and compare the results obtained for quadratic loss and logistic loss. Using methods from statistical physics, we obtain a precise asymptotic expression for the train and test errors of random feature models trained on a simple model of structured data. The input covariance is built from independent blocks allowing us to tune the saliency of low-dimensional structures and their alignment with respect to the target function.Our results show in particular that in the over-parametrized regime, the impact of data structure on both train and test error curves is greater for logistic loss than for mean-squared loss: the easier the task, the wider the gap in performance between the two losses at the advantage of the logistic. Numerical experiments on MNIST and CIFAR10 confirm our insights.

        ----

        ## [651] Near-optimal Offline and Streaming Algorithms for Learning Non-Linear Dynamical Systems

        **Authors**: *Suhas S. Kowshik, Dheeraj Nagaraj, Prateek Jain, Praneeth Netrapalli*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/47a658229eb2368a99f1d032c8848542-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/47a658229eb2368a99f1d032c8848542-Abstract.html)

        **Abstract**:

        We consider the setting of vector valued non-linear dynamical systems $X_{t+1} = \phi(A^{*} X_t) + \eta_t$, where $\eta_t$ is unbiased noise and $\phi : \mathbb{R} \to \mathbb{R}$ is a known link function that satisfies certain {\em expansivity property}. The goal is to learn $A^{*}$ from a single trajectory $X_1,\cdots , X_T$ of {\em dependent or correlated} samples.While the problem is well-studied in the linear case, where $\phi$ is identity, with optimal error rates even for non-mixing systems, existing results in the non-linear case hold only for mixing systems. In this work, we improve existing results for learning nonlinear systems in a number of ways: a) we provide the first offline algorithm that can learn non-linear dynamical systems without the mixing assumption, b) we significantly improve upon the sample complexity of existing results for mixing systems, c) in the much harder one-pass, streaming setting we study a SGD with Reverse Experience Replay (SGD-RER) method, and demonstrate that for mixing systems, it achieves the same sample complexity as our offline algorithm, d) we justify the expansivity assumption by showing that for the popular ReLU  link function --- a non-expansive but easy to learn link function with i.i.d. samples --- any method would require exponentially many samples (with respect to dimension of $X_t$) from the dynamical system. We validate our results via. simulations and  demonstrate that a naive application of SGD can be highly sub-optimal. Indeed, our work demonstrates that for correlated data, specialized  methods designed for the dependency structure in data can  significantly outperform  standard SGD based methods.

        ----

        ## [652] Mixture Proportion Estimation and PU Learning: A Modern Approach

        **Authors**: *Saurabh Garg, Yifan Wu, Alexander J. Smola, Sivaraman Balakrishnan, Zachary C. Lipton*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/47b4f1bfdf6d298682e610ad74b37dca-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/47b4f1bfdf6d298682e610ad74b37dca-Abstract.html)

        **Abstract**:

        Given only positive examples and unlabeled examples (from both positive and negative classes), we might hope nevertheless to estimate an accurate positive-versus-negative classifier. Formally, this task is broken down into two subtasks: (i) Mixture Proportion Estimation (MPE)---determining the fraction of positive examples in the unlabeled data; and (ii) PU-learning---given such an estimate, learning the desired positive-versus-negative classifier. Unfortunately, classical methods for both problems break down in high-dimensional settings. Meanwhile, recently proposed heuristics lack theoretical coherence and depend precariously on hyperparameter tuning. In this paper, we propose two simple techniques: Best Bin Estimation (BBE) (for MPE); and Conditional Value Ignoring Risk (CVIR), a simple objective for PU-learning. Both methods dominate previous approaches empirically, and for BBE, we establish formal guarantees that hold whenever we can train a model to cleanly separate out a small subset of positive examples. Our final algorithm (TED)$^n$, alternates between the two procedures, significantly improving both our mixture proportion estimator and classifier

        ----

        ## [653] Escape saddle points by a simple gradient-descent based algorithm

        **Authors**: *Chenyi Zhang, Tongyang Li*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/47bd8ac1becf213f155a82244b4a696a-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/47bd8ac1becf213f155a82244b4a696a-Abstract.html)

        **Abstract**:

        Escaping saddle points is a central research topic in nonconvex optimization. In this paper, we propose a simple gradient-based algorithm such that for a smooth function $f\colon\mathbb{R}^n\to\mathbb{R}$, it outputs an $\epsilon$-approximate second-order stationary point in $\tilde{O}(\log n/\epsilon^{1.75})$ iterations. Compared to the previous state-of-the-art algorithms by Jin et al. with $\tilde{O}(\log^4 n/\epsilon^{2})$ or $\tilde{O}(\log^6 n/\epsilon^{1.75})$ iterations, our algorithm is polynomially better in terms of $\log n$ and matches their complexities in terms of $1/\epsilon$. For the stochastic setting, our algorithm outputs an $\epsilon$-approximate second-order stationary point in $\tilde{O}(\log^{2} n/\epsilon^{4})$ iterations. Technically, our main contribution is an idea of implementing a robust Hessian power method using only gradients, which can find negative curvature near saddle points and achieve the polynomial speedup in $\log n$ compared to the perturbed gradient descent methods. Finally, we also perform numerical experiments that support our results.

        ----

        ## [654] AC/DC: Alternating Compressed/DeCompressed Training of Deep Neural Networks

        **Authors**: *Alexandra Peste, Eugenia Iofinova, Adrian Vladu, Dan Alistarh*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/48000647b315f6f00f913caa757a70b3-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/48000647b315f6f00f913caa757a70b3-Abstract.html)

        **Abstract**:

        The increasing computational requirements of deep neural networks (DNNs) have led to significant interest in obtaining DNN models that are sparse, yet accurate. Recent work has investigated the even harder case of sparse training, where the DNN weights are, for as much as possible, already sparse to reduce computational costs during training. Existing sparse training methods are often empirical and can have lower accuracy relative to the dense baseline. In this paper, we present a general approach called Alternating Compressed/DeCompressed (AC/DC) training of DNNs, demonstrate convergence for a variant of the algorithm,  and show that AC/DC outperforms existing sparse training methods in accuracy at similar computational budgets; at high sparsity levels, AC/DC even outperforms existing methods that rely on accurate pre-trained dense models. An important property of AC/DC is that it allows co-training of dense and sparse models, yielding accurate sparse-dense model pairs at the end of the training process. This is useful in practice, where compressed variants may be desirable for deployment in resource-constrained settings without re-doing the entire training flow, and also provides us with insights into the accuracy gap between dense and compressed models.

        ----

        ## [655] HyperSPNs: Compact and Expressive Probabilistic Circuits

        **Authors**: *Andy Shih, Dorsa Sadigh, Stefano Ermon*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/481fbfa59da2581098e841b7afc122f1-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/481fbfa59da2581098e841b7afc122f1-Abstract.html)

        **Abstract**:

        Probabilistic circuits (PCs) are a family of generative models which allows for the computation of exact likelihoods and marginals of its probability distributions. PCs are both expressive and tractable, and serve as popular choices for discrete density estimation tasks. However, large PCs are susceptible to overfitting, and only a few regularization strategies (e.g., dropout, weight-decay) have been explored. We propose HyperSPNs: a new paradigm of generating the mixture weights of large PCs using a small-scale neural network. Our framework can be viewed as a soft weight-sharing strategy, which combines the greater expressiveness of large models with the better generalization and memory-footprint properties of small models.  We show the merits of our regularization strategy on two state-of-the-art PC families introduced in recent literature -- RAT-SPNs and EiNETs -- and demonstrate generalization improvements in both models on a suite of density estimation benchmarks in both discrete and continuous domains.

        ----

        ## [656] Scaling Vision with Sparse Mixture of Experts

        **Authors**: *Carlos Riquelme, Joan Puigcerver, Basil Mustafa, Maxim Neumann, Rodolphe Jenatton, André Susano Pinto, Daniel Keysers, Neil Houlsby*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/48237d9f2dea8c74c2a72126cf63d933-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/48237d9f2dea8c74c2a72126cf63d933-Abstract.html)

        **Abstract**:

        Sparsely-gated Mixture of Experts networks (MoEs) have demonstrated excellent scalability in Natural Language Processing. In Computer Vision, however, almost all performant networks are "dense", that is, every input is processed by every parameter. We present a Vision MoE (V-MoE), a sparse version of the Vision Transformer, that is scalable and competitive with the largest dense networks. When applied to image recognition, V-MoE matches the performance of state-of-the-art networks, while requiring as little as half of the compute at inference time. Further, we propose an extension to the routing algorithm that can prioritize subsets of each input across the entire batch, leading to adaptive per-image compute. This allows V-MoE to trade-off performance and compute smoothly at test-time. Finally, we demonstrate the potential of V-MoE to scale vision models, and train a 15B parameter model that attains 90.35% on ImageNet.

        ----

        ## [657] Two-sided fairness in rankings via Lorenz dominance

        **Authors**: *Virginie Do, Sam Corbett-Davies, Jamal Atif, Nicolas Usunier*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/48259990138bc03361556fb3f94c5d45-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/48259990138bc03361556fb3f94c5d45-Abstract.html)

        **Abstract**:

        We consider the problem of generating rankings that are fair towards both users and item producers in recommender systems. We address both usual recommendation (e.g., of music or movies) and reciprocal recommendation (e.g., dating). Following concepts of distributive justice in welfare economics, our notion of fairness aims at increasing the utility of the worse-off individuals, which we formalize using the criterion of Lorenz efficiency. It guarantees that rankings are Pareto efficient, and that they maximally redistribute utility from better-off to worse-off, at a given level of overall utility. We propose to generate rankings by maximizing concave welfare functions, and develop an efficient inference procedure based on the Frank-Wolfe algorithm. We prove that unlike existing approaches based on fairness constraints, our approach always produces fair rankings. Our experiments also show that it increases the utility of the worse-off at lower costs in terms of overall utility.

        ----

        ## [658] Stability & Generalisation of Gradient Descent for Shallow Neural Networks without the Neural Tangent Kernel

        **Authors**: *Dominic Richards, Ilja Kuzborskij*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/483101a6bc4e6c46a86222eb65fbcb6a-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/483101a6bc4e6c46a86222eb65fbcb6a-Abstract.html)

        **Abstract**:

        We revisit on-average algorithmic stability of Gradient Descent (GD) for training  overparameterised  shallow  neural  networks  and prove  new  generalisation and  excess  risk  bounds  without  the  Neural  Tangent  Kernel  (NTK)  or  Polyak-≈Åojasiewicz (PL) assumptions. In particular, we show oracle type bounds which reveal that the generalisation and excess risk of GD is controlled by an interpolating network with the shortest GD path from initialisation (in a sense, an interpolating network with the smallest relative norm).  While this was known for kernelised interpolants, our proof applies directly to networks trained by GD without intermediate kernelisation. At the same time, by relaxing oracle inequalities developed here we recover existing NTK-based risk bounds in a straightforward way, which demonstrates that our analysis is tighter. Finally, unlike most of the NTK-based analyses we focus on regression with label noise and show that GD with early stopping is consistent

        ----

        ## [659] Adversarial Intrinsic Motivation for Reinforcement Learning

        **Authors**: *Ishan Durugkar, Mauricio Tec, Scott Niekum, Peter Stone*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/486c0401c56bf7ec2daa9eba58907da9-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/486c0401c56bf7ec2daa9eba58907da9-Abstract.html)

        **Abstract**:

        Learning with an objective to minimize the mismatch with a reference distribution has been shown to be useful for generative modeling and imitation learning. In this paper, we investigate whether one such objective, the Wasserstein-1 distance between a policy's state visitation distribution and a target distribution, can be utilized effectively for reinforcement learning (RL) tasks. Specifically, this paper focuses on goal-conditioned reinforcement learning where the idealized (unachievable) target distribution has full measure at the goal. This paper introduces a quasimetric specific to Markov Decision Processes (MDPs) and uses this quasimetric to estimate the above Wasserstein-1 distance. It further shows that the policy that minimizes this Wasserstein-1 distance is the policy that reaches the goal in as few steps as possible. Our approach, termed Adversarial Intrinsic Motivation (AIM), estimates this Wasserstein-1 distance through its dual objective and uses it to compute a supplemental reward function. Our experiments show that this reward function changes smoothly with respect to transitions in the MDP and directs the agent's exploration to find the goal efficiently. Additionally, we combine AIM with Hindsight Experience Replay (HER) and show that the resulting algorithm accelerates learning significantly on several simulated robotics tasks when compared to other rewards that encourage exploration or accelerate learning.

        ----

        ## [660] Machine Learning for Variance Reduction in Online Experiments

        **Authors**: *Yongyi Guo, Dominic Coey, Mikael Konutgan, Wenting Li, Chris Schoener, Matt Goldman*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/488b084119a1c7a4950f00706ec7ea16-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/488b084119a1c7a4950f00706ec7ea16-Abstract.html)

        **Abstract**:

        We consider the problem of variance reduction in randomized controlled trials, through the use of covariates correlated with the outcome but independent of the treatment. We propose a machine learning regression-adjusted treatment effect estimator, which we call MLRATE. MLRATE uses machine learning predictors of the outcome to reduce estimator variance. It employs cross-fitting to avoid overfitting biases, and we prove consistency and asymptotic normality under general conditions. MLRATE is robust to poor predictions from the machine learning step: if the predictions are uncorrelated with the outcomes, the estimator performs asymptotically no worse than the standard difference-in-means estimator, while if predictions are highly correlated with outcomes, the efficiency gains are large. In A/A tests, for a set of 48 outcome metrics commonly monitored in Facebook experiments, the estimator has over $70\%$ lower variance than the simple difference-in-means estimator, and about $19\%$ lower variance than the common univariate procedure which adjusts only for pre-experiment values of the outcome.

        ----

        ## [661] L2ight: Enabling On-Chip Learning for Optical Neural Networks via Efficient in-situ Subspace Optimization

        **Authors**: *Jiaqi Gu, Hanqing Zhu, Chenghao Feng, Zixuan Jiang, Ray T. Chen, David Z. Pan*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/48aedb8880cab8c45637abc7493ecddd-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/48aedb8880cab8c45637abc7493ecddd-Abstract.html)

        **Abstract**:

        Silicon-photonics-based optical neural network (ONN) is a promising hardware platform that could represent a paradigm shift in efficient AI with its CMOS-compatibility, flexibility, ultra-low execution latency, and high energy efficiency. In-situ training on the online programmable photonic chips is appealing but still encounters challenging issues in on-chip implementability, scalability, and efficiency. In this work, we propose a closed-loop ONN on-chip learning framework L2ight to enable scalable ONN mapping and efficient in-situ learning. L2ight adopts a three-stage learning flow that first calibrates the complicated photonic circuit states under challenging physical constraints, then performs photonic core mapping via combined analytical solving and zeroth-order optimization. A subspace learning procedure with multi-level sparsity is integrated into L2ight to enable in-situ gradient evaluation and fast adaptation, unleashing the power of optics for real on-chip intelligence. Extensive experiments demonstrate our proposed L2ight outperforms prior ONN training protocols with 3-order-of-magnitude higher scalability and over 30x better efficiency, when benchmarked on various models and learning tasks. This synergistic framework is the first scalable on-chip learning solution that pushes this emerging field from intractable to scalable and further to efficient for next-generation self-learnable photonic neural chips. From a co-design perspective, L2ight also provides essential insights for hardware-restricted unitary subspace optimization and efficient sparse training. We open-source our framework at the link.

        ----

        ## [662] Towards Gradient-based Bilevel Optimization with Non-convex Followers and Beyond

        **Authors**: *Risheng Liu, Yaohua Liu, Shangzhi Zeng, Jin Zhang*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/48bea99c85bcbaaba618ba10a6f69e44-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/48bea99c85bcbaaba618ba10a6f69e44-Abstract.html)

        **Abstract**:

        In recent years, Bi-Level Optimization (BLO) techniques have received extensive attentions from both learning and vision communities. A variety of BLO models in complex and practical tasks are of non-convex follower structure in nature (a.k.a., without Lower-Level Convexity, LLC for short). However, this challenging class of BLOs is lack of developments on both efficient solution strategies and solid theoretical guarantees. In this work, we propose a new algorithmic framework, named Initialization Auxiliary and Pessimistic Trajectory Truncated Gradient Method (IAPTT-GM), to partially address the above issues. In particular, by introducing an auxiliary as initialization to guide the optimization dynamics and designing a pessimistic trajectory truncation operation, we construct a reliable approximate version of the original BLO in the absence of LLC hypothesis. Our theoretical investigations establish the convergence of solutions returned by IAPTT-GM towards those of the original BLO without LLC. As an additional bonus, we also theoretically justify the quality of our IAPTT-GM embedded with Nesterov's accelerated dynamics under LLC. The experimental results confirm both the convergence of our algorithm without LLC, and the theoretical findings under LLC.

        ----

        ## [663] Multi-Facet Clustering Variational Autoencoders

        **Authors**: *Fabian Falck, Haoting Zhang, Matthew Willetts, George Nicholson, Christopher Yau, Chris C. Holmes*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/48cb136b65a69e8c2aa22913a0d91b2f-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/48cb136b65a69e8c2aa22913a0d91b2f-Abstract.html)

        **Abstract**:

        Work in deep clustering focuses on finding a single partition of data. However, high-dimensional data, such as images, typically feature multiple interesting characteristics one could cluster over. For example, images of objects against a background could be clustered over the shape of the object and separately by the colour of the background. In this paper, we introduce Multi-Facet Clustering Variational Autoencoders (MFCVAE), a novel class of variational autoencoders with a hierarchy of latent variables, each with a Mixture-of-Gaussians prior, that learns multiple clusterings simultaneously, and is trained fully unsupervised and end-to-end. MFCVAE uses a progressively-trained ladder architecture which leads to highly stable performance. We provide novel theoretical results for optimising the ELBO analytically with respect to the categorical variational posterior distribution, correcting earlier influential theoretical work. On image benchmarks, we demonstrate that our approach separates out and clusters over different aspects of the data in a disentangled manner. We also show other advantages of our model: the compositionality of its latent space and that it provides controlled generation of samples.

        ----

        ## [664] Synthetic Design: An Optimization Approach to Experimental Design with Synthetic Controls

        **Authors**: *Nick Doudchenko, Khashayar Khosravi, Jean Pouget-Abadie, Sébastien Lahaie, Miles Lubin, Vahab S. Mirrokni, Jann Spiess, Guido Imbens*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/48d23e87eb98cc2227b5a8c33fa00680-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/48d23e87eb98cc2227b5a8c33fa00680-Abstract.html)

        **Abstract**:

        We investigate the optimal design of experimental studies that have pre-treatment outcome data available.  The average treatment effect is estimated as the difference between the weighted average outcomes of the treated and control units. A number of commonly used approaches fit this formulation, including the difference-in-means estimator and a variety of synthetic-control techniques. We propose several methods for choosing the set of treated units in conjunction with the weights. Observing the NP-hardness of the problem, we introduce a mixed-integer programming formulation which selects both the treatment and control sets and unit weightings. We prove that these proposed approaches lead to qualitatively different experimental units being selected for treatment. We use simulations based on publicly available data from the US Bureau of Labor Statistics that show improvements in terms of mean squared error and statistical power when compared to simple and commonly used alternatives such as randomized trials.

        ----

        ## [665] Ranking Policy Decisions

        **Authors**: *Hadrien Pouget, Hana Chockler, Youcheng Sun, Daniel Kroening*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/48db71587df6c7c442e5b76cc723169a-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/48db71587df6c7c442e5b76cc723169a-Abstract.html)

        **Abstract**:

        Policies trained via Reinforcement Learning (RL) without human intervention are often needlessly complex, making them difficult to analyse and interpret. In a run with $n$ time steps, a policy will make $n$ decisions on actions to take; we conjecture that only a small subset of these decisions delivers value over selecting a simple default action. Given a trained policy, we propose a novel black-box method based on statistical fault localisation that ranks the states of the environment according to the importance of decisions made in those states. We argue that among other things, the ranked list of states can help explain and understand the policy. As the ranking method is statistical, a direct evaluation of its quality is hard. As a proxy for quality, we use the ranking to create new, simpler policies from the original ones by pruning decisions identified as unimportant (that is, replacing them by default actions) and measuring the impact on performance. Our experimental results on a diverse set of standard benchmarks demonstrate that pruned policies can perform on a level comparable to the original policies. We show that naive approaches for ranking policies, e.g. ranking based on the frequency of visiting a state, do not result in high-performing pruned policies. To the best of our knowledge, there are no similar techniques for ranking RL policies' decisions.

        ----

        ## [666] Searching the Search Space of Vision Transformer

        **Authors**: *Minghao Chen, Kan Wu, Bolin Ni, Houwen Peng, Bei Liu, Jianlong Fu, Hongyang Chao, Haibin Ling*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/48e95c45c8217961bf6cd7696d80d238-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/48e95c45c8217961bf6cd7696d80d238-Abstract.html)

        **Abstract**:

        Vision Transformer has shown great visual representation power in substantial vision tasks such as recognition and detection, and thus been attracting fast-growing efforts on manually designing more effective architectures. In this paper, we propose to use neural architecture search to automate this process, by searching not only the architecture but also the search space. The central idea is to gradually evolve different search dimensions guided by their E-T Error computed using a weight-sharing supernet. Moreover, we provide design guidelines of general vision transformers with extensive analysis according to the space searching process, which could promote the understanding of vision transformer. Remarkably, the searched models, named S3 (short for Searching the Search Space), from the searched space achieve superior performance to recently proposed models, such as Swin, DeiT and ViT, when evaluated on ImageNet. The effectiveness of S3 is also illustrated on object detection, semantic segmentation and visual question answering, demonstrating its generality to downstream vision and vision-language tasks. Code and models will be available at https://github.com/microsoft/Cream.

        ----

        ## [667] Relative stability toward diffeomorphisms indicates performance in deep nets

        **Authors**: *Leonardo Petrini, Alessandro Favero, Mario Geiger, Matthieu Wyart*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/497476fe61816251905e8baafdf54c23-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/497476fe61816251905e8baafdf54c23-Abstract.html)

        **Abstract**:

        Understanding why deep nets can classify data in large dimensions remains a challenge. It has been proposed that they do so by becoming stable to diffeomorphisms, yet existing empirical measurements support that it is often not the case. We revisit this question by defining a maximum-entropy distribution on diffeomorphisms, that allows to study typical diffeomorphisms of a given norm. We confirm that stability toward diffeomorphisms does not strongly correlate to performance on benchmark data sets of images. By contrast, we find that the stability toward diffeomorphisms relative to that of generic transformations $R_f$ correlates remarkably with the test error $\epsilon_t$. It is of order unity at initialization but decreases by several decades during training for state-of-the-art architectures. For CIFAR10 and 15 known architectures, we find $\epsilon_t\approx 0.2\sqrt{R_f}$, suggesting that obtaining a small $R_f$ is important to achieve good performance. We study how $R_f$ depends on the size of the training set and compare it to a simple model of invariant learning.

        ----

        ## [668] Raw Nav-merge Seismic Data to Subsurface Properties with MLP based Multi-Modal Information Unscrambler

        **Authors**: *Aditya Desai, Zhaozhuo Xu, Menal Gupta, Anu Chandran, Antoine Vial-Aussavy, Anshumali Shrivastava*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/498f2c21688f6451d9f5fd09d53edda7-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/498f2c21688f6451d9f5fd09d53edda7-Abstract.html)

        **Abstract**:

        Traditional seismic inversion (SI) maps the hundreds of terabytes of raw-field data to subsurface properties in gigabytes.  This inversion process is expensive, requiring over a year of human and computational effort. Recently, data-driven approaches equipped with Deep learning (DL) are envisioned to improve SI efficiency.  However, these improvements are restricted to data with highly reduced scale and complexity. To extend these approaches to real-scale seismic data, researchers need to process raw nav-merge seismic data into an image and perform convolution. We argue that this convolution-based way of SI is not only computationally expensive but also conceptually problematic. Seismic data is not naturally an image and need not be processed as images. In this work, we go beyond convolution and propose a novel SI method. We solve the scalability of SI by proposing a new auxiliary learning paradigm for SI (Aux-SI). This paradigm breaks the SI into local inversion tasks, which predicts each small chunk of subsurface properties using surrounding seismic data. Aux-SI combines these local predictions to obtain the entire subsurface model. However, even this local inversion is still challenging due to: (1) high-dimensional, spatially irregular multi-modal seismic data, (2) there is no concrete spatial mapping (or alignment) between subsurface properties and raw data. To handle these challenges, we propose an all-MLP architecture,  Multi-Modal Information Unscrambler (MMI-Unscrambler), that unscrambles seismic information by ingesting all available multi-modal data. The experiment shows that MMI-Unscrambler outperforms both SOTA U-Net and Transformer models on simulation data. We also scale MMI-Unscrambler to raw-field nav-merge data on Gulf-of-Mexico to obtain a geologically sound velocity model with an SSIM score of 0.8. To the best of our knowledge, this is the first successful demonstration of the DL approach on SI for real, large-scale, and complicated raw field data.

        ----

        ## [669] Inverse Problems Leveraging Pre-trained Contrastive Representations

        **Authors**: *Sriram Ravula, Georgios Smyrnis, Matt Jordan, Alexandros G. Dimakis*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/498f940d9b933c529b06aa96d18f7eda-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/498f940d9b933c529b06aa96d18f7eda-Abstract.html)

        **Abstract**:

        We study a new family of inverse problems for recovering representations of corrupted data. We assume access to a pre-trained representation learning network R(x) that operates on clean images, like CLIP. The problem is to recover the representation of an image R(x), if we are only given a corrupted version A(x), for some known forward operator A. We propose a supervised inversion method that uses a contrastive objective to obtain excellent representations for highly corrupted images. Using a linear probe on our robust representations, we achieve a higher accuracy than end-to-end supervised baselines when classifying images with various types of distortions, including blurring, additive noise, and random pixel masking. We evaluate on a subset of ImageNet and observe that our method is robust to varying levels of distortion. Our method outperforms end-to-end baselines even with a fraction of the labeled data in a wide range of forward operators.

        ----

        ## [670] The Unbalanced Gromov Wasserstein Distance: Conic Formulation and Relaxation

        **Authors**: *Thibault Séjourné, François-Xavier Vialard, Gabriel Peyré*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/4990974d150d0de5e6e15a1454fe6b0f-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/4990974d150d0de5e6e15a1454fe6b0f-Abstract.html)

        **Abstract**:

        Comparing metric measure spaces (i.e. a metric space endowed with a probability distribution) is at the heart of many machine learning problems. The most popular distance between such metric measure spaces is the Gromov-Wasserstein (GW) distance, which is the solution of a quadratic assignment problem. The GW distance is however limited to the comparison of metric measure spaces endowed with a \emph{probability} distribution. To alleviate this issue, we introduce two Unbalanced Gromov-Wasserstein formulations: a distance and a more tractable upper-bounding relaxation.  They both allow the comparison of metric spaces equipped with arbitrary positive measures up to isometries. The first formulation is a positive and definite divergence based on a relaxation of the mass conservation constraint using a novel type of quadratically-homogeneous divergence. This divergence works hand in hand with the entropic regularization approach which is popular to solve large scale optimal transport problems. We show that the underlying non-convex optimization problem can be efficiently tackled using a highly parallelizable and GPU-friendly iterative scheme. The second formulation is a distance between mm-spaces up to isometries based on a conic lifting.  Lastly, we provide numerical experiments on synthetic and domain adaptation data with a Positive-Unlabeled learning task to highlight the salient features of the unbalanced divergence and its potential applications in ML.

        ----

        ## [671] Diffusion Models Beat GANs on Image Synthesis

        **Authors**: *Prafulla Dhariwal, Alexander Quinn Nichol*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/49ad23d1ec9fa4bd8d77d02681df5cfa-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/49ad23d1ec9fa4bd8d77d02681df5cfa-Abstract.html)

        **Abstract**:

        We show that diffusion models can achieve image sample quality superior to the current state-of-the-art generative models. We achieve this on unconditional image synthesis by finding a better architecture through a series of ablations. For conditional image synthesis, we further improve sample quality with classifier guidance: a simple, compute-efficient method for trading off diversity for fidelity using gradients from a classifier. We achieve an FID of 2.97 on ImageNet 128$\times$128, 4.59 on ImageNet 256$\times$256, and 7.72 on ImageNet 512$\times$512, and we match BigGAN-deep even with as few as 25 forward passes per sample, all while maintaining better coverage of the distribution. Finally, we find that classifier guidance combines well with upsampling diffusion models, further improving FID to 3.94 on ImageNet 256$\times$256 and 3.85 on ImageNet 512$\times$512.

        ----

        ## [672] Learning MDPs from Features: Predict-Then-Optimize for Sequential Decision Making by Reinforcement Learning

        **Authors**: *Kai Wang, Sanket Shah, Haipeng Chen, Andrew Perrault, Finale Doshi-Velez, Milind Tambe*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/49e863b146f3b5470ee222ee84669b1c-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/49e863b146f3b5470ee222ee84669b1c-Abstract.html)

        **Abstract**:

        In the predict-then-optimize framework, the objective is to train a predictive model, mapping from environment features to parameters of an optimization problem, which maximizes decision quality when the optimization is subsequently solved. Recent work on decision-focused learning shows that embedding the optimization problem in the training pipeline can improve decision quality and help generalize better to unseen tasks compared to relying on an intermediate loss function for evaluating prediction quality. We study the predict-then-optimize framework in the context of sequential decision problems (formulated as MDPs) that are solved via reinforcement learning. In particular, we are given environment features and a set of trajectories from training MDPs, which we use to train a predictive model that generalizes to unseen test MDPs without trajectories. Two significant computational challenges arise in applying decision-focused learning to MDPs: (i) large state and action spaces make it infeasible for existing techniques to differentiate through MDP problems, and (ii) the high-dimensional policy space, as parameterized by a neural network, makes differentiating through a policy expensive. We resolve the first challenge by sampling provably unbiased derivatives to approximate and differentiate through optimality conditions, and the second challenge by using a low-rank approximation to the high-dimensional sample-based derivatives. We implement both Bellman-based and policy gradient-based decision-focused learning on three different MDP problems with missing parameters, and show that decision-focused learning performs better in generalization to unseen tasks.

        ----

        ## [673] A Closer Look at the Worst-case Behavior of Multi-armed Bandit Algorithms

        **Authors**: *Anand Kalvit, Assaf Zeevi*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/49ef08ad6e7f26d7f200e1b2b9e6e4ac-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/49ef08ad6e7f26d7f200e1b2b9e6e4ac-Abstract.html)

        **Abstract**:

        One of the key drivers of complexity in the classical (stochastic) multi-armed bandit (MAB) problem is the difference between mean rewards in the top two arms, also known as the instance gap. The celebrated Upper Confidence Bound (UCB) policy is among the simplest optimism-based MAB algorithms that naturally adapts to this gap: for a horizon of play n, it achieves optimal O(log n) regret in instances with "large" gaps, and a near-optimal O(\sqrt{n log n}) minimax regret when the gap can be arbitrarily "small." This paper provides new results on the arm-sampling behavior of UCB, leading to several important insights. Among these, it is shown that arm-sampling rates under UCB are asymptotically deterministic, regardless of the problem complexity. This discovery facilitates new sharp asymptotics and a novel alternative proof for the O(\sqrt{n log n}) minimax regret of UCB. Furthermore, the paper also provides the first complete process-level characterization of the MAB problem in the conventional diffusion scaling. Among other things, the "small" gap worst-case lens adopted in this paper also reveals profound distinctions between the behavior of UCB and Thompson Sampling, such as an "incomplete learning" phenomenon characteristic of the latter.

        ----

        ## [674] SAPE: Spatially-Adaptive Progressive Encoding for Neural Optimization

        **Authors**: *Amir Hertz, Or Perel, Raja Giryes, Olga Sorkine-Hornung, Daniel Cohen-Or*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/4a06d868d044c50af0cf9bc82d2fc19f-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/4a06d868d044c50af0cf9bc82d2fc19f-Abstract.html)

        **Abstract**:

        Multilayer-perceptrons (MLP)  are known to struggle learning functions of high-frequencies, and in particular, instances of wide frequency bands.We present a progressive mapping scheme for input signals of MLP networks, enabling them to better fit a wide range of frequencies without sacrificing training stability or requiring any domain specific preprocessing. We introduce Spatially Adaptive Progressive Encoding (SAPE) layers, which gradually unmask signal components with increasing frequencies as a function of time and space. The progressive exposure of frequencies is monitored by a feedback loop throughout the neural optimization process, allowing changes to propagate at different rates among local spatial portions of the signal space. We demonstrate the advantage of our method on variety of domains and applications: regression of low dimensional signals and images, representation learning of occupancy networks, and a geometric task of mesh transfer between 3D shapes.

        ----

        ## [675] A Biased Graph Neural Network Sampler with Near-Optimal Regret

        **Authors**: *Qingru Zhang, David Wipf, Quan Gan, Le Song*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/4a08142c38dbe374195d41c04562d9f8-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/4a08142c38dbe374195d41c04562d9f8-Abstract.html)

        **Abstract**:

        Graph neural networks (GNN) have recently emerged as a vehicle for applying deep network architectures to graph and relational data.  However, given the increasing size of industrial datasets, in many practical situations, the message passing computations required for sharing information across GNN layers are no longer scalable. Although various sampling methods have been introduced to approximate full-graph training within a tractable budget, there remain unresolved complications such as high variances and limited theoretical guarantees.  To address these issues, we build upon existing work and treat GNN neighbor sampling as a multi-armed bandit problem but with a newly-designed reward function that introduces some degree of bias designed to reduce variance and avoid unstable, possibly-unbounded pay outs.  And unlike prior bandit-GNN use cases, the resulting policy leads to near-optimal regret while accounting for the GNN training dynamics introduced by SGD. From a practical standpoint, this translates into lower variance estimates and competitive or superior test accuracy across several benchmarks.

        ----

        ## [676] Equilibrium Refinement for the Age of Machines: The One-Sided Quasi-Perfect Equilibrium

        **Authors**: *Gabriele Farina, Tuomas Sandholm*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/4a3050ae2c77da4f9c90e2e58e8e520f-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/4a3050ae2c77da4f9c90e2e58e8e520f-Abstract.html)

        **Abstract**:

        In two-player zero-sum extensive-form games, Nash equilibrium prescribes optimal strategies against perfectly rational opponents. However, it does not guarantee rational play in parts of the game tree that can only be reached by the players making mistakes. This can be problematic when operationalizing equilibria in the real world among imperfect players. Trembling-hand refinements are a sound remedy to this issue, and are subsets of Nash equilibria that are designed to handle the possibility that any of the players may make mistakes. In this paper, we initiate the study of equilibrium refinements for settings where one of the players is perfectly rational (the ``machine'') and the other may make mistakes. As we show, this endeavor has many pitfalls: many intuitively appealing approaches to refinement fail in various ways. On the positive side, we introduce a modification of the classical quasi-perfect equilibrium (QPE) refinement, which we call the one-sided quasi-perfect equilibrium. Unlike QPE, one-sided QPE only accounts for mistakes from one player and assumes that no mistakes will be made by the machine. We present experiments on standard benchmark games and an endgame from the famous man-machine match where the AI Libratus was the first to beat top human specialist professionals in heads-up no-limit Texas hold'em poker. We show that one-sided QPE can be computed more efficiently than all known prior refinements, paving the way to wider adoption of Nash equilibrium refinements in settings with perfectly rational machines (or humans perfectly actuating machine-generated strategies) that interact with players prone to mistakes. We also show that one-sided QPE tends to play better than a Nash equilibrium strategy against imperfect opponents.

        ----

        ## [677] Interpreting Representation Quality of DNNs for 3D Point Cloud Processing

        **Authors**: *Wen Shen, Qihan Ren, Dongrui Liu, Quanshi Zhang*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/4a3e00961a08879c34f91ca0070ea2f5-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/4a3e00961a08879c34f91ca0070ea2f5-Abstract.html)

        **Abstract**:

        In this paper, we evaluate the quality of knowledge representations encoded in deep neural networks (DNNs) for 3D point cloud processing. We propose a method to disentangle the overall model vulnerability into the sensitivity to the rotation, the translation, the scale, and local 3D structures. Besides, we also propose metrics to evaluate the spatial smoothness of encoding 3D structures, and the representation complexity of the DNN. Based on such analysis, experiments expose representation problems with classic DNNs, and explain the utility of the adversarial training. The code will be released when this paper is accepted.

        ----

        ## [678] How Fine-Tuning Allows for Effective Meta-Learning

        **Authors**: *Kurtland Chua, Qi Lei, Jason D. Lee*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/4a533591763dfa743a13affab1a85793-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/4a533591763dfa743a13affab1a85793-Abstract.html)

        **Abstract**:

        Representation learning has served as a key tool for meta-learning, enabling rapid learning of new tasks. Recent works like MAML learn task-specific representations by finding an initial representation requiring minimal per-task adaptation (i.e. a fine-tuning-based objective). We present a theoretical framework for analyzing a MAML-like algorithm, assuming all available tasks require approximately the same representation. We then provide risk bounds on predictors found by fine-tuning via gradient descent, demonstrating that the method provably leverages the shared structure. We illustrate these bounds in the logistic regression and neural network settings. In contrast, we establish settings where learning one representation for all tasks (i.e. using a "frozen representation" objective) fails. Notably, any such algorithm cannot outperform directly learning the target task with no other information, in the worst case. This separation underscores the benefit of fine-tuning-based over “frozen representation” objectives in few-shot learning.

        ----

        ## [679] Cooperative Stochastic Bandits with Asynchronous Agents and Constrained Feedback

        **Authors**: *Lin Yang, Yu-Zhen Janice Chen, Stephen Pasteris, Mohammad H. Hajiesmaili, John C. S. Lui, Don Towsley*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/4a5876b450b45371f6cfe5047ac8cd45-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/4a5876b450b45371f6cfe5047ac8cd45-Abstract.html)

        **Abstract**:

        This paper studies a cooperative multi-armed bandit problem with $M$ agents cooperating together to solve the same instance of a $K$-armed stochastic bandit problem with the goal of maximizing the cumulative reward of agents. The agents are heterogeneous in (i) their limited access to a local subset of arms; and (ii) their decision-making rounds, i.e., agents are asynchronous with different decision-making gaps. The goal is to find the global optimal arm and agents are able to pull any arm, however, they observe the reward only when the selected arm is local.The challenge is a tradeoff for agents between pulling a local arm with the possibility of observing the feedback, or relying on the observations of other agents that might occur at different rates. Naive extensions of traditional algorithms lead to an arbitrarily poor regret as a function of aggregate action frequency of any $\textit{suboptimal}$ arm located in slow agents. We resolve this issue by proposing a novel two-stage learning algorithm, called $\texttt{CO-LCB}$ algorithm, whose regret is a function of aggregate action frequency of agents containing the $\textit{optimal}$ arm. We also show that the regret of $\texttt{CO-LCB}$ matches the regret lower bound up to a small factor.

        ----

        ## [680] Multiple Descent: Design Your Own Generalization Curve

        **Authors**: *Lin Chen, Yifei Min, Mikhail Belkin, Amin Karbasi*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/4ae67a7dd7e491f8fb6f9ea0cf25dfdb-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/4ae67a7dd7e491f8fb6f9ea0cf25dfdb-Abstract.html)

        **Abstract**:

        This paper explores the generalization loss of linear regression in variably parameterized families of models, both under-parameterized and over-parameterized. We show that the generalization curve can have an arbitrary number of peaks, and moreover, the locations of those peaks can be explicitly controlled. Our results highlight the fact that both the classical U-shaped generalization curve and the recently observed double descent curve are not intrinsic properties of the model family. Instead, their emergence is due to the interaction between the properties of the data and the inductive biases of learning algorithms.

        ----

        ## [681] On Empirical Risk Minimization with Dependent and Heavy-Tailed Data

        **Authors**: *Abhishek Roy, Krishnakumar Balasubramanian, Murat A. Erdogdu*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/4afa19649ae378da31a423bcd78a97c8-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/4afa19649ae378da31a423bcd78a97c8-Abstract.html)

        **Abstract**:

        In this work, we establish risk bounds for Empirical Risk Minimization (ERM) with both dependent and heavy-tailed data-generating processes. We do so by extending the seminal works~\cite{pmlr-v35-mendelson14, mendelson2018learning} on the analysis of ERM with heavy-tailed but independent and identically distributed observations, to the strictly stationary exponentially $\beta$-mixing case. We allow for the interaction between the noise and inputs to be even polynomially heavy-tailed, which covers a significantly large class of heavy-tailed models beyond what is analyzed in the learning theory literature. We illustrate our theoretical results by obtaining rates of convergence for high-dimensional linear regression with dependent and heavy-tailed data.

        ----

        ## [682] Gone Fishing: Neural Active Learning with Fisher Embeddings

        **Authors**: *Jordan T. Ash, Surbhi Goel, Akshay Krishnamurthy, Sham M. Kakade*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/4afe044911ed2c247005912512ace23b-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/4afe044911ed2c247005912512ace23b-Abstract.html)

        **Abstract**:

        There is an increasing need for effective active learning algorithms that are compatible with deep neural networks. This paper motivates and revisits a classic, Fisher-based active selection objective, and proposes BAIT, a practical, tractable, and high-performing algorithm that makes it viable for use with neural models. BAIT draws inspiration from the theoretical analysis of maximum likelihood estimators (MLE) for parametric models. It selects batches of samples by optimizing a bound on the MLE error in terms of the Fisher information, which we show can be implemented efficiently at scale by exploiting linear-algebraic structure especially amenable to execution on modern hardware. Our experiments demonstrate that BAIT outperforms the previous state of the art on both classification and regression problems, and is flexible enough to be used with a variety of model architectures.

        ----

        ## [683] On Riemannian Optimization over Positive Definite Matrices with the Bures-Wasserstein Geometry

        **Authors**: *Andi Han, Bamdev Mishra, Pratik Kumar Jawanpuria, Junbin Gao*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/4b04b0dcd2ade339a3d7ce13252a29d4-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/4b04b0dcd2ade339a3d7ce13252a29d4-Abstract.html)

        **Abstract**:

        In this paper, we comparatively analyze the Bures-Wasserstein (BW) geometry with the popular Affine-Invariant (AI) geometry for Riemannian optimization on the symmetric positive definite (SPD) matrix manifold. Our study begins with an observation that the BW metric has a linear dependence on SPD matrices in contrast to the quadratic dependence of the AI metric. We build on this to show that the BW metric is a more suitable and robust choice for several Riemannian optimization problems over ill-conditioned SPD matrices. We show that the BW geometry has a non-negative curvature, which further improves convergence rates of algorithms over the non-positively curved AI geometry. Finally, we verify that several popular cost functions, which are known to be geodesic convex under the AI geometry, are also geodesic convex under the BW geometry. Extensive experiments on various applications support our findings.

        ----

        ## [684] Refining Language Models with Compositional Explanations

        **Authors**: *Huihan Yao, Ying Chen, Qinyuan Ye, Xisen Jin, Xiang Ren*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/4b26dc4663ccf960c8538d595d0a1d3a-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/4b26dc4663ccf960c8538d595d0a1d3a-Abstract.html)

        **Abstract**:

        Pre-trained language models have been successful on text classification tasks, but are prone to learning spurious correlations from biased datasets, and are thus vulnerable when making inferences in a new domain. Prior work reveals such spurious patterns via post-hoc explanation algorithms which compute the importance of input features. Further, the model is regularized to align the importance scores with human knowledge, so that the unintended model behaviors are eliminated. However, such a regularization technique lacks flexibility and coverage, since only importance scores towards a pre-defined list of features are adjusted, while more complex human knowledge such as feature interaction and pattern generalization can hardly be incorporated. In this work, we propose to refine a learned language model for a target domain by collecting human-provided compositional explanations regarding observed biases. By parsing these explanations into executable logic rules, the human-specified refinement advice from a small set of explanations can be generalized to more training examples. We additionally introduce a regularization term allowing adjustments for both importance and interaction of features to better rectify model behavior. We demonstrate the effectiveness of the proposed approach on two text classification tasks by showing improved performance in target domain as well as improved model fairness after refinement.

        ----

        ## [685] Going Beyond Linear RL: Sample Efficient Neural Function Approximation

        **Authors**: *Baihe Huang, Kaixuan Huang, Sham M. Kakade, Jason D. Lee, Qi Lei, Runzhe Wang, Jiaqi Yang*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/4b4edc2630fe75800ddc29a7b4070add-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/4b4edc2630fe75800ddc29a7b4070add-Abstract.html)

        **Abstract**:

        Deep Reinforcement Learning (RL) powered by neural net approximation of the Q function has had enormous empirical success. While the theory of RL has traditionally focused on linear function approximation (or eluder dimension) approaches, little is known about nonlinear RL with neural net approximations of the Q functions. This is the focus of this work, where we study function approximation with two-layer neural networks (considering both ReLU and polynomial activation functions).  Our first result is a computationally and statistically efficient algorithm in the generative model setting under completeness for two-layer neural networks. Our second result considers this setting but under only realizability of the neural net function class.  Here, assuming deterministic dynamics, the sample complexity scales linearly in the algebraic dimension. In all cases, our results significantly improve upon what can be attained with linear (or eluder dimension) methods.

        ----

        ## [686] Scalable Neural Data Server: A Data Recommender for Transfer Learning

        **Authors**: *Tianshi Cao, Sasha Doubov, David Acuna, Sanja Fidler*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/4b55df75e2e804bab559aa885be40310-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/4b55df75e2e804bab559aa885be40310-Abstract.html)

        **Abstract**:

        Absence of large-scale labeled data in the practitioner's target domain can be a bottleneck to applying machine learning algorithms in practice. Transfer learning is a popular strategy for leveraging additional data to improve the downstream performance, but finding the most relevant data to transfer from can be challenging. Neural Data Server (NDS), a search engine that recommends relevant data for a given downstream task, has been previously proposed to address this problem (Yan et al., 2020). NDS uses a mixture of experts trained on data sources to estimate similarity between each source and the downstream task. Thus, the computational cost to each user grows with the number of sources and requires an expensive training step for each data provider.To address these issues, we propose Scalable Neural Data Server (SNDS), a large-scale search engine that can theoretically index thousands of datasets to serve relevant ML data to end users. SNDS trains the mixture of experts on intermediary datasets during initialization, and represents both data sources and downstream tasks by their proximity to the intermediary datasets. As such, computational cost incurred by users of SNDS remains fixed as new datasets are added to the server, without pre-training for the data providers.We validate SNDS on a plethora of real world tasks and find that data recommended by SNDS improves downstream task performance over baselines. We also demonstrate the scalability of our system by demonstrating its ability to select relevant data for transfer outside of the natural image setting.

        ----

        ## [687] What can linearized neural networks actually say about generalization?

        **Authors**: *Guillermo Ortiz-Jiménez, Seyed-Mohsen Moosavi-Dezfooli, Pascal Frossard*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/4b5deb9a14d66ab0acc3b8a2360cde7c-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/4b5deb9a14d66ab0acc3b8a2360cde7c-Abstract.html)

        **Abstract**:

        For certain infinitely-wide neural networks, the neural tangent kernel (NTK) theory fully characterizes generalization, but for the networks used in practice, the empirical NTK only provides a rough first-order approximation. Still, a growing body of work keeps leveraging this approximation to successfully analyze important deep learning phenomena and design algorithms for new applications. In our work, we provide strong empirical evidence to determine the practical validity of such approximation by conducting a systematic comparison of the behavior of different neural networks and their linear approximations on different tasks. We show that the linear approximations can indeed rank the learning complexity of certain tasks for neural networks, even when they achieve very different performances. However, in contrast to what was previously reported, we discover that neural networks do not always perform better than their kernel approximations, and reveal that the performance gap heavily depends on architecture, dataset size and training task. We discover that networks overfit to these tasks mostly due to the evolution of their kernel during training, thus, revealing a new type of implicit bias.

        ----

        ## [688] CATs: Cost Aggregation Transformers for Visual Correspondence

        **Authors**: *Seokju Cho, Sunghwan Hong, Sangryul Jeon, Yunsung Lee, Kwanghoon Sohn, Seungryong Kim*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/4b6538a44a1dfdc2b83477cd76dee98e-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/4b6538a44a1dfdc2b83477cd76dee98e-Abstract.html)

        **Abstract**:

        We propose a novel cost aggregation network, called Cost Aggregation Transformers (CATs), to find dense correspondences between semantically similar images with additional challenges posed by large intra-class appearance and geometric variations. Cost aggregation is a highly important process in matching tasks, which the matching accuracy depends on the quality of its output. Compared to hand-crafted or CNN-based methods addressing the cost aggregation, in that either lacks robustness to severe deformations or inherit the limitation of CNNs that fail to discriminate incorrect matches due to limited receptive fields, CATs explore global consensus among initial correlation map with the help of some architectural designs that allow us to fully leverage self-attention mechanism. Specifically, we include appearance affinity modeling to aid the cost aggregation process in order to disambiguate the noisy initial correlation maps and propose multi-level aggregation to efficiently capture different semantics from hierarchical feature representations. We then combine with swapping self-attention technique and residual connections not only to enforce consistent matching, but also to ease the learning process, which we find that these result in an apparent performance boost. We conduct experiments to demonstrate the effectiveness of the proposed model over the latest methods and provide extensive ablation studies. Code and trained models are available at https://sunghwanhong.github.io/CATs/.

        ----

        ## [689] Asynchronous Stochastic Optimization Robust to Arbitrary Delays

        **Authors**: *Alon Cohen, Amit Daniely, Yoel Drori, Tomer Koren, Mariano Schain*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/4b85256c4881edb6c0776df5d81f6236-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/4b85256c4881edb6c0776df5d81f6236-Abstract.html)

        **Abstract**:

        We consider the problem of stochastic optimization with delayed gradients in which, at each time step $t$, the algorithm makes an update using a stale stochastic gradient from step $t - d_t$ for some arbitrary delay $d_t$.   This setting abstracts asynchronous distributed optimization where a central server receives gradient updates computed by worker machines. These machines can experience computation and communication loads that might vary significantly over time.   In the general non-convex smooth optimization setting, we give a simple and efficient algorithm that requires $O( \sigma^2/\epsilon^4 + \tau/\epsilon^2 )$ steps for finding an $\epsilon$-stationary point $x$.   Here, $\tau$ is the \emph{average} delay $\frac{1}{T}\sum_{t=1}^T d_t$ and $\sigma^2$ is the variance of the stochastic gradients.   This improves over previous work, which showed that stochastic gradient decent achieves the same rate but with respect to the \emph{maximal} delay $\max_{t} d_t$, that can be significantly larger than the average delay especially in heterogeneous distributed systems.   Our experiments demonstrate the efficacy and robustness of our algorithm in cases where the delay distribution is skewed or heavy-tailed.

        ----

        ## [690] Consistent Non-Parametric Methods for Maximizing Robustness

        **Authors**: *Robi Bhattacharjee, Kamalika Chaudhuri*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/4bb236de7787ceedafdff83bb8ea4710-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/4bb236de7787ceedafdff83bb8ea4710-Abstract.html)

        **Abstract**:

        Learning classifiers that are robust to adversarial examples has received a great deal of recent attention. A major drawback of the standard robust learning framework is the imposition of an artificial robustness radius $r$ that applies to all inputs, and ignores the fact that data may be highly heterogeneous. In particular, it is plausible that robustness regions should be larger in some regions of data, and smaller in other. In this paper, we address this limitation by proposing a new limit classifier, called the neighborhood optimal classifier, that extends the Bayes optimal classifier outside its support by using the label of the closest in-support point. We then argue that this classifier maximizes the size of its robustness regions subject to the constraint of having accuracy equal to the Bayes optimal. We then present sufficient conditions under which general non-parametric methods that can be represented as weight functions converge towards this limit object, and show that both nearest neighbors and kernel classifiers (under certain assumptions) suffice.

        ----

        ## [691] Generalizable Multi-linear Attention Network

        **Authors**: *Tao Jin, Zhou Zhao*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/4bbdcc0e821637155ac4217bdab70d2e-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/4bbdcc0e821637155ac4217bdab70d2e-Abstract.html)

        **Abstract**:

        The majority of existing multimodal sequential learning methods focus on how to obtain effective representations and ignore the importance of multimodal fusion. Bilinear attention network (BAN) is a commonly used fusion method, which leverages tensor operations to associate the features of different modalities. However, BAN has a poor compatibility for more modalities, since the computational complexity of the attention map increases exponentially with the number of modalities. Based on this concern, we propose a new method called generalizable multi-linear attention network (MAN), which can associate as many modalities as possible in linear complexity with hierarchical approximation decomposition (HAD). Besides, considering the fact that softmax attention kernels cannot be decomposed as linear operation directly, we adopt the addition random features (ARF) mechanism to approximate the non-linear softmax functions with enough theoretical analysis. We conduct extensive experiments on four datasets of three tasks (multimodal sentiment analysis, multimodal speaker traits recognition, and video retrieval), the experimental results show that MAN could achieve competitive results compared with the state-of-the-art methods, showcasing the effectiveness of the approximation decomposition and addition random features mechanism.

        ----

        ## [692] Labeling Trick: A Theory of Using Graph Neural Networks for Multi-Node Representation Learning

        **Authors**: *Muhan Zhang, Pan Li, Yinglong Xia, Kai Wang, Long Jin*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/4be49c79f233b4f4070794825c323733-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/4be49c79f233b4f4070794825c323733-Abstract.html)

        **Abstract**:

        In this paper, we provide a theory of using graph neural networks (GNNs) for multi-node representation learning (where we are interested in learning a representation for a set of more than one node, such as link). We know that GNN is designed to learn single-node representations. When we want to learn a node set representation involving multiple nodes, a common practice in previous works is to directly aggregate the single-node representations obtained by a GNN into a joint node set representation. In this paper, we show a fundamental constraint of such an approach, namely the inability to capture the dependence between nodes in the node set, and argue that directly aggregating individual node representations does not lead to an effective joint representation for multiple nodes. Then, we notice that a few previous successful works for multi-node representation learning, including SEAL, Distance Encoding, and ID-GNN, all used node labeling. These methods first label nodes in the graph according to their relationships with the target node set before applying a GNN. Then, the node representations obtained in the labeled graph are aggregated into a node set representation. By investigating their inner mechanisms, we unify these node labeling techniques into a single and most general form---labeling trick. We prove that with labeling trick a sufficiently expressive GNN learns the most expressive node set representations, thus in principle solves any joint learning tasks over node sets. Experiments on one important two-node representation learning task, link prediction, verified our theory. Our work explains the superior performance of previous node-labeling-based methods, and establishes a theoretical foundation of using GNNs for multi-node representation learning.

        ----

        ## [693] SUPER-ADAM: Faster and Universal Framework of Adaptive Gradients

        **Authors**: *Feihu Huang, Junyi Li, Heng Huang*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/4be5a36cbaca8ab9d2066debfe4e65c1-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/4be5a36cbaca8ab9d2066debfe4e65c1-Abstract.html)

        **Abstract**:

        Adaptive gradient methods have shown excellent performances for solving many machine learning problems. Although multiple adaptive gradient  methods were recently studied, they mainly focus on either empirical or theoretical aspects and also only work for specific problems by using some  specific adaptive learning rates. Thus, it is desired to design  a  universal  framework for practical algorithms of adaptive gradients with theoretical guarantee to solve general problems. To fill this gap, we propose a faster and universal framework of adaptive gradients (i.e., SUPER-ADAM) by introducing a universal adaptive matrix that includes most existing adaptive gradient forms. Moreover, our framework can flexibly integrate the momentum and variance reduced techniques. In particular, our novel framework provides the convergence analysis support for adaptive gradient methods under the nonconvex setting. In theoretical analysis, we prove that our SUPER-ADAM algorithm can achieve the best known gradient (i.e., stochastic first-order oracle (SFO)) complexity of $\tilde{O}(\epsilon^{-3})$ for finding an $\epsilon$-stationary point of nonconvex optimization, which matches the lower bound for stochastic smooth nonconvex optimization. In numerical experiments, we employ various deep learning tasks to validate that our algorithm consistently outperforms the existing adaptive algorithms. Code is available at https://github.com/LIJUNYI95/SuperAdam

        ----

        ## [694] General Nonlinearities in SO(2)-Equivariant CNNs

        **Authors**: *Daniel Franzen, Michael Wand*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/4bfbd52f4e8466dc12aaf30b7e057b66-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/4bfbd52f4e8466dc12aaf30b7e057b66-Abstract.html)

        **Abstract**:

        Invariance under symmetry is an important problem in machine learning. Our paper looks specifically at equivariant neural networks where transformations of inputs yield homomorphic transformations of outputs. Here, steerable CNNs have emerged as the standard solution. An inherent problem of steerable representations is that general nonlinear layers break equivariance, thus restricting architectural choices. Our paper applies harmonic distortion analysis to illuminate the effect of nonlinearities on Fourier representations of SO(2). We develop a novel FFT-based algorithm for computing representations of non-linearly transformed activations while maintaining band-limitation. It yields exact equivariance for polynomial (approximations of) nonlinearities, as well as approximate solutions with tunable accuracy for general functions. We apply the approach to build a fully E(3)-equivariant network for sampled 3D surface data. In experiments with 2D and 3D data, we obtain results that compare favorably to the state-of-the-art in terms of accuracy while permitting continuous symmetry and exact equivariance.

        ----

        ## [695] Denoising Normalizing Flow

        **Authors**: *Christian Horvat, Jean-Pascal Pfister*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/4c07fe24771249c343e70c32289c1192-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/4c07fe24771249c343e70c32289c1192-Abstract.html)

        **Abstract**:

        Normalizing flows (NF) are expressive as well as tractable density estimation methods whenever the support of the density is diffeomorphic to the entire data-space. However, real-world data sets typically live on (or very close to) low-dimensional manifolds thereby challenging the applicability of standard NF on real-world problems. Here we propose a novel method - called Denoising Normalizing Flow (DNF) - that estimates the density on the low-dimensional manifold while learning the manifold as well. The DNF works in 3 steps. First, it inflates the manifold - making it diffeomorphic to the entire data-space. Secondly, it learns an NF on the inflated manifold and finally it learns a denoising mapping - similarly to denoising autoencoders. The DNF relies on a single cost function and does not require to alternate between a density estimation phase and a manifold learning phase - as it is the case with other recent methods. Furthermore, we show that the DNF can learn meaningful low-dimensional representations from naturalistic images as well as generate high-quality samples.

        ----

        ## [696] Attention over Learned Object Embeddings Enables Complex Visual Reasoning

        **Authors**: *David Ding, Felix Hill, Adam Santoro, Malcolm Reynolds, Matt M. Botvinick*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/4c26774d852f62440fc746ea4cdd57f6-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/4c26774d852f62440fc746ea4cdd57f6-Abstract.html)

        **Abstract**:

        Neural networks have achieved success in a wide array of perceptual tasks but often fail at tasks involving both perception and higher-level reasoning. On these more challenging tasks, bespoke approaches (such as modular symbolic components, independent dynamics models or semantic parsers) targeted towards that specific type of task have typically performed better. The downside to these targeted approaches, however, is that they can be more brittle than general-purpose neural networks, requiring significant modification or even redesign according to the particular task at hand. Here, we propose a more general neural-network-based approach to dynamic visual reasoning problems that obtains state-of-the-art performance on three different domains, in each case outperforming bespoke modular approaches tailored specifically to the task. Our method relies on learned object-centric representations, self-attention and self-supervised dynamics learning, and all three elements together are required for strong performance to emerge. The success of this combination suggests that there may be no need to trade off flexibility for performance on problems involving spatio-temporal or causal-style reasoning. With the right soft biases and learning objectives in a neural network we may be able to attain the best of both worlds.

        ----

        ## [697] Differentially Private Federated Bayesian Optimization with Distributed Exploration

        **Authors**: *Zhongxiang Dai, Bryan Kian Hsiang Low, Patrick Jaillet*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/4c27cea8526af8cfee3be5e183ac9605-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/4c27cea8526af8cfee3be5e183ac9605-Abstract.html)

        **Abstract**:

        Bayesian optimization (BO) has recently been extended to the federated learning (FL) setting by the federated Thompson sampling (FTS) algorithm, which has promising applications such as federated hyperparameter tuning. However, FTS is not equipped with a rigorous privacy guarantee which is an important consideration in FL. Recent works have incorporated differential privacy (DP) into the training of deep neural networks through a general framework for adding DP to iterative algorithms. Following this general DP framework, our work here integrates DP into FTS to preserve user-level privacy. We also leverage the ability of this general DP framework to handle different parameter vectors, as well as the technique of local modeling for BO, to further improve the utility of our algorithm through distributed exploration (DE). The resulting differentially private FTS with DE (DP-FTS-DE) algorithm is endowed with theoretical guarantees for both the privacy and utility and is amenable to interesting theoretical insights about the privacy-utility trade-off. We also use real-world experiments to show that DP-FTS-DE achieves high utility (competitive performance) with a strong privacy guarantee (small privacy loss) and induces a trade-off between privacy and utility.

        ----

        ## [698] Differentiable Learning Under Triage

        **Authors**: *Nastaran Okati, Abir De, Manuel Gomez-Rodriguez*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/4c4c937b67cc8d785cea1e42ccea185c-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/4c4c937b67cc8d785cea1e42ccea185c-Abstract.html)

        **Abstract**:

        Multiple lines of evidence suggest that predictive models may benefit from algorithmic triage. Under algorithmic triage, a predictive model does not predict all instances but instead defers some of them to human experts. However, the interplay between the prediction accuracy of the model and the human experts under algorithmic triage is not well understood. In this work, we start by formally characterizing under which circumstances a predictive model may benefit from algorithmic triage. In doing so, we also demonstrate that models trained for full automation may be suboptimal under triage. Then, given any model and desired level of triage, we show that the optimal triage policy is a deterministic threshold rule in which triage decisions are derived deterministically by thresholding the difference between the model and human errors on a per-instance level. Building upon these results, we introduce a practical gradient-based algorithm that is guaranteed to find a sequence of predictive models and triage policies of increasing performance. Experiments on a wide variety of supervised learning tasks using synthetic and real data from two important applications---content moderation and scientific discovery---illustrate our theoretical results and show that the models and triage policies provided by our gradient-based algorithm outperform those provided by several competitive baselines.

        ----

        ## [699] A New Theoretical Framework for Fast and Accurate Online Decision-Making

        **Authors**: *Nicolò Cesa-Bianchi, Tommaso Cesari, Yishay Mansour, Vianney Perchet*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/4c4ea5258ef3fb3fb1fc48fee9b4408c-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/4c4ea5258ef3fb3fb1fc48fee9b4408c-Abstract.html)

        **Abstract**:

        We introduce a novel theoretical framework for Return On Investment (ROI) maximization in repeated decision-making. Our setting is motivated by the use case of companies that regularly receive proposals for technological innovations and want to quickly decide whether they are worth implementing. We design an algorithm for learning ROI-maximizing decision-making policies over a sequence of innovation proposals. Our algorithm provably converges to an optimal policy in class $\Pi$ at a rate of order $\min\big\{1/(N\Delta^2),N^{-1/3}\}$, where $N$ is the number of innovations and $\Delta$ is the suboptimality gap in $\Pi$. A significant hurdle of our formulation, which sets it aside from other online learning problems such as bandits, is that running a policy does not provide an unbiased estimate of its performance.

        ----

        ## [700] When Expressivity Meets Trainability: Fewer than $n$ Neurons Can Work

        **Authors**: *Jiawei Zhang, Yushun Zhang, Mingyi Hong, Ruoyu Sun, Zhi-Quan Luo*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/4c7a167bb329bd92580a99ce422d6fa6-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/4c7a167bb329bd92580a99ce422d6fa6-Abstract.html)

        **Abstract**:

        Modern neural networks are often quite wide, causing large memory and computation costs. It is thus of great interest to train a narrower network. However, training narrow neural nets remains a challenging task. We ask two theoretical questions: Can narrow networks have as strong expressivity as wide ones? If so, does the loss function exhibit a  benign optimization landscape? In this work, we provide partially affirmative answers to both questions for 1-hidden-layer networks with fewer than $n$ (sample size) neurons when the activation is smooth.  First, we prove that as long as the width $m \geq 2n/d$ (where $d$ is the input dimension), its expressivity is strong, i.e., there exists at least one global minimizer with zero training loss. Second, we identify a nice local region with no local-min or saddle points. Nevertheless, it is not clear whether gradient descent can stay in this nice region. Third, we consider a constrained optimization formulation where the feasible region is the nice local region, and prove that every KKT point is a nearly global minimizer. It is expected that projected gradient methods converge to KKT points under mild technical conditions, but we leave the rigorous convergence analysis to future work. Thorough numerical results show that projected gradient methods on this constrained formulation significantly outperform SGD for training narrow neural nets.

        ----

        ## [701] Analyzing the Confidentiality of Undistillable Teachers in Knowledge Distillation

        **Authors**: *Souvik Kundu, Qirui Sun, Yao Fu, Massoud Pedram, Peter A. Beerel*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/4ca82782c5372a547c104929f03fe7a9-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/4ca82782c5372a547c104929f03fe7a9-Abstract.html)

        **Abstract**:

        Knowledge distillation (KD) has recently been identified as a method that can unintentionally leak private information regarding the details of a teacher model to an unauthorized student. Recent research in developing undistillable nasty teachers that can protect model confidentiality has gained significant attention. However, the level of protection these nasty models offer has been largely untested. In this paper, we show that transferring knowledge to a shallow sub-section of a student can largely reduce a teacher’s influence. By exploring the depth of the shallow subsection, we then present a distillation technique that enables a skeptical student model to learn even from a nasty teacher. To evaluate the efficacy of our skeptical students, we conducted experiments with several models with KD on both training data-available and data-free scenarios for various datasets. While distilling from nasty teachers, compared to the normal student models, skeptical students consistently provide superior classification performance of up to ∼59.5%. Moreover, similar to normal students, skeptical students maintain high classification accuracy when distilled from a normal teacher, showing their efficacy irrespective of the teacher being nasty or not. We believe the ability of skeptical students to largely diminish the KD-immunity of potentially nasty teachers will motivate the research community to create more robust mechanisms for model confidentiality. We have open-sourced the code at https://github.com/ksouvik52/Skeptical2021

        ----

        ## [702] High Probability Complexity Bounds for Line Search Based on Stochastic Oracles

        **Authors**: *Billy Jin, Katya Scheinberg, Miaolan Xie*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/4cb811134b9d39fc3104bd06ce75abad-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/4cb811134b9d39fc3104bd06ce75abad-Abstract.html)

        **Abstract**:

        We consider a line-search method for continuous optimization under a stochastic setting where the function values and gradients are available only through inexact probabilistic zeroth and first-order oracles. These oracles capture multiple standard settings including expected loss minimization and zeroth-order optimization. Moreover, our framework is very general and allows the function and gradient estimates to be biased.  The proposed algorithm is simple to describe, easy to implement, and uses these oracles in a similar way as the standard deterministic line search uses exact function and gradient values.  Under fairly general conditions on the oracles, we derive a high probability tail bound on the iteration complexity of the algorithm when applied to non-convex smooth functions. These results are stronger than those for other existing stochastic line search methods and apply in more general settings.

        ----

        ## [703] Pay Attention to MLPs

        **Authors**: *Hanxiao Liu, Zihang Dai, David R. So, Quoc V. Le*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/4cc05b35c2f937c5bd9e7d41d3686fff-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/4cc05b35c2f937c5bd9e7d41d3686fff-Abstract.html)

        **Abstract**:

        Transformers have become one of the most important architectural innovations in deep learning and have enabled many breakthroughs over the past few years. Here we propose a simple network architecture, gMLP, based solely on MLPs with gating, and show that it can perform as well as Transformers in key language and vision applications. Our comparisons show that self-attention is not critical for Vision Transformers, as gMLP can achieve the same accuracy. For BERT, our model achieves parity with Transformers on pretraining perplexity and is better on some downstream NLP tasks. On finetuning tasks where gMLP performs worse, making the gMLP model substantially larger can close the gap with Transformers. In general, our experiments show that gMLP can scale as well as Transformers over increased data and compute.

        ----

        ## [704] An Image is Worth More Than a Thousand Words: Towards Disentanglement in The Wild

        **Authors**: *Aviv Gabbay, Niv Cohen, Yedid Hoshen*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/4cef5b5e6ff1b3445db4c013f1d452e0-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/4cef5b5e6ff1b3445db4c013f1d452e0-Abstract.html)

        **Abstract**:

        Unsupervised disentanglement has been shown to be theoretically impossible without inductive biases on the models and the data. As an alternative approach, recent methods rely on limited supervision to disentangle the factors of variation and allow their identifiability. While annotating the true generative factors is only required for a limited number of observations, we argue that it is infeasible to enumerate all the factors of variation that describe a real-world image distribution. To this end, we propose a method for disentangling a set of factors which are only partially labeled, as well as separating the complementary set of residual factors that are never explicitly specified. Our success in this challenging setting, demonstrated on synthetic benchmarks, gives rise to leveraging off-the-shelf image descriptors to partially annotate a subset of attributes in real image domains (e.g. of human faces) with minimal manual effort. Specifically, we use a recent language-image embedding model (CLIP) to annotate a set of attributes of interest in a zero-shot manner and demonstrate state-of-the-art disentangled image manipulation results.

        ----

        ## [705] Dynamics of Stochastic Momentum Methods on Large-scale, Quadratic Models

        **Authors**: *Courtney Paquette, Elliot Paquette*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/4cf0ed8641cfcbbf46784e620a0316fb-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/4cf0ed8641cfcbbf46784e620a0316fb-Abstract.html)

        **Abstract**:

        We analyze a class of stochastic gradient algorithms with momentum on a high-dimensional random least squares problem. Our framework, inspired by random matrix theory, provides an exact (deterministic) characterization for the sequence of function values produced by these algorithms which is expressed only in terms of the eigenvalues of the Hessian. This leads to simple expressions for nearly-optimal hyperparameters, a description of the limiting neighborhood, and average-case complexity. As a consequence, we show that (small-batch) stochastic heavy-ball momentum with a fixed momentum parameter provides no actual performance improvement over SGD when step sizes are adjusted correctly. For contrast, in the non-strongly convex setting, it is possible to get a large improvement over SGD using momentum. By introducing hyperparameters that depend on the number of samples, we propose a new algorithm sDANA (stochastic dimension adjusted Nesterov acceleration) which obtains an asymptotically optimal average-case complexity while remaining linearly convergent in the strongly convex setting without adjusting parameters.

        ----

        ## [706] Adversarial Examples in Multi-Layer Random ReLU Networks

        **Authors**: *Peter L. Bartlett, Sébastien Bubeck, Yeshwanth Cherapanamjeri*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/4d19b37a2c399deace9082d464930022-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/4d19b37a2c399deace9082d464930022-Abstract.html)

        **Abstract**:

        We consider the phenomenon of adversarial examples in ReLU networks with independent Gaussian parameters.  For networks of constant depth and with a large range of widths (for instance, it suffices if the width of each layer is polynomial in that of any other layer), small perturbations of input vectors lead to large changes of outputs.  This generalizes results of Daniely and Schacham (2020) for networks of rapidly decreasing width and of Bubeck et al (2021) for two-layer networks. Our proof shows that adversarial examples arise in these networks because the functions they compute are \emph{locally} very similar to random linear functions. Bottleneck layers play a key role: the minimal width up to some point in the network determines scales and sensitivities of mappings computed up to that point.  The main result is for networks with constant depth, but we also show that some constraint on depth is necessary for a result of this kind, because there are suitably deep networks that, with constant probability, compute a function that is close to constant.

        ----

        ## [707] Efficient Statistical Assessment of Neural Network Corruption Robustness

        **Authors**: *Karim Tit, Teddy Furon, Mathias Rousset*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/4d215ab7508a3e089af43fb605dd27d1-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/4d215ab7508a3e089af43fb605dd27d1-Abstract.html)

        **Abstract**:

        We quantify the robustness of a trained network to input uncertainties with a stochastic simulation inspired by the field of Statistical Reliability Engineering. The robustness assessment is cast as a statistical hypothesis test: the network is deemed as locally robust if the estimated probability of failure is lower than a critical level.The procedure is based on an Importance Splitting simulation generating samples of rare events. We derive theoretical guarantees that are non-asymptotic w.r.t. sample size. Experiments tackling large scale networks outline the efficiency of our method making a low number of calls to the network function.

        ----

        ## [708] A Highly-Efficient Group Elastic Net Algorithm with an Application to Function-On-Scalar Regression

        **Authors**: *Tobia Boschi, Matthew Reimherr, Francesca Chiaromonte*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/4d410063822cd9be28f86701c0bc3a31-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/4d410063822cd9be28f86701c0bc3a31-Abstract.html)

        **Abstract**:

        Feature Selection and Functional Data Analysis are two dynamic areas of research, with important applications in the analysis of large and complex data sets. Straddling these two areas, we propose a new highly efficient algorithm to perform Group Elastic Net with application to function-on-scalar feature selection, where a functional response is modeled against a very large number of potential scalar predictors. First, we introduce a new algorithm to solve Group Elastic Net in ultra-high dimensional settings, which exploits the sparsity structure of the Augmented Lagrangian to greatly reduce the computational burden. Next, taking advantage of the properties of Functional Principal Components, we extend our algorithm to the function-on-scalar regression framework. We use simulations to demonstrate the CPU time gains afforded by our approach compared to its best existing competitors, and present an application to data from a Genome-Wide Association Study on childhood obesity.

        ----

        ## [709] Hierarchical Clustering: O(1)-Approximation for Well-Clustered Graphs

        **Authors**: *Bogdan-Adrian Manghiuc, He Sun*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/4d68e143defa221fead61c84de7527a3-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/4d68e143defa221fead61c84de7527a3-Abstract.html)

        **Abstract**:

        Hierarchical clustering  studies a recursive partition of a data set into clusters of successively smaller size, and is a fundamental problem in data analysis. In this work we study the cost function for hierarchical clustering introduced by Dasgupta, and present two polynomial-time approximation algorithms: Our first result is an $O(1)$-approximation algorithm for graphs of high conductance. Our simple construction bypasses complicated recursive routines of finding sparse cuts known in the literature. Our second and main result is an $O(1)$-approximation algorithm for a wide family of graphs that exhibit a well-defined structure of clusters. This result generalises the previous state-of-the-art, which holds only for graphs generated from stochastic models. The significance of our work is demonstrated by the empirical analysis on both synthetic and real-world data sets, on which our presented algorithm outperforms  the previously proposed algorithm for graphs with a well-defined cluster structure.

        ----

        ## [710] Realistic evaluation of transductive few-shot learning

        **Authors**: *Olivier Veilleux, Malik Boudiaf, Pablo Piantanida, Ismail Ben Ayed*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/4d7a968bb636e25818ff2a3941db08c1-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/4d7a968bb636e25818ff2a3941db08c1-Abstract.html)

        **Abstract**:

        Transductive inference is widely used in few-shot learning, as it leverages the statistics of the unlabeled query set of a few-shot task, typically yielding substantially better performances than its inductive counterpart.  The current few-shot benchmarks use perfectly class-balanced tasks at inference. We argue that such an artificial regularity is unrealistic, as it assumes that the marginal label probability of the testing samples is known and fixed to the uniform distribution. In fact, in realistic scenarios, the unlabeled query sets come with arbitrary and unknown label marginals.   We introduce and study the effect of arbitrary class distributions within the query sets of few-shot tasks at inference,  removing the class-balance artefact. Specifically, we model the marginal probabilities of the classes as Dirichlet-distributed random variables, which yields a principled and realistic sampling within the simplex.  This leverages the current few-shot benchmarks, building testing tasks with arbitrary class distributions. We evaluate experimentally state-of-the-art transductive methods over 3 widely used data sets, and observe, surprisingly, substantial performance drops, even below inductive methods in some cases. Furthermore, we propose a generalization of the mutual-information loss, based on α-divergences, which can handle effectively class-distribution variations. Empirically, we show that our transductive α-divergence optimization outperforms state-of-the-art methods across several data sets, models and few-shot settings.

        ----

        ## [711] Qu-ANTI-zation: Exploiting Quantization Artifacts for Achieving Adversarial Outcomes

        **Authors**: *Sanghyun Hong, Michael-Andrei Panaitescu-Liess, Yigitcan Kaya, Tudor Dumitras*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/4d8bd3f7351f4fee76ba17594f070ddd-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/4d8bd3f7351f4fee76ba17594f070ddd-Abstract.html)

        **Abstract**:

        Quantization is a popular technique that transforms the parameter representation of a neural network from floating-point numbers into lower-precision ones (e.g., 8-bit integers). It reduces the memory footprint and the computational cost at inference, facilitating the deployment of resource-hungry models. However, the parameter perturbations caused by this transformation result in behavioral disparities between the model before and after quantization. For example, a quantized model can misclassify some test-time samples that are otherwise classified correctly. It is not known whether such differences lead to a new security vulnerability. We hypothesize that an adversary may control this disparity to introduce specific behaviors that activate upon quantization. To study this hypothesis, we weaponize quantization-aware training and propose a new training framework to implement adversarial quantization outcomes. Following this framework, we present three attacks we carry out with quantization: (i) an indiscriminate attack for significant accuracy loss; (ii) a targeted attack against specific samples; and (iii) a backdoor attack for controlling the model with an input trigger. We further show that a single compromised model defeats multiple quantization schemes, including robust quantization techniques. Moreover, in a federated learning scenario, we demonstrate that a set of malicious participants who conspire can inject our quantization-activated backdoor. Lastly, we discuss potential counter-measures and show that only re-training consistently removes the attack artifacts. Our code is available at https://github.com/Secure-AI-Systems-Group/Qu-ANTI-zation

        ----

        ## [712] Differentially Private Stochastic Optimization: New Results in Convex and Non-Convex Settings

        **Authors**: *Raef Bassily, Cristóbal Guzmán, Michael Menart*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/4ddb5b8d603f88e9de689f3230234b47-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/4ddb5b8d603f88e9de689f3230234b47-Abstract.html)

        **Abstract**:

        We study differentially private stochastic optimization in convex and non-convex settings. For the convex case, we focus on the family of non-smooth generalized linear losses (GLLs). Our algorithm for the $\ell_2$ setting achieves optimal excess population risk in near-linear time, while the best known differentially private algorithms for general convex losses run in super-linear time. Our algorithm for the $\ell_1$ setting has nearly-optimal excess population risk $\tilde{O}\big(\sqrt{\frac{\log{d}}{n}}\big)$, and circumvents the dimension dependent lower bound of \cite{Asi:2021} for general non-smooth convex losses. In the differentially private non-convex setting, we provide several new algorithms for approximating stationary points of the population risk. For the $\ell_1$-case with smooth losses and polyhedral constraint, we provide the first nearly dimension independent rate, $\tilde O\big(\frac{\log^{2/3}{d}}{{n^{1/3}}}\big)$ in linear time. For the constrained $\ell_2$-case, with smooth losses, we obtain a linear-time algorithm with rate $\tilde O\big(\frac{1}{n^{3/10}d^{1/10}}+\big(\frac{d}{n^2}\big)^{1/5}\big)$.   Finally, for the $\ell_2$-case we provide the first method  for {\em non-smooth weakly convex} stochastic optimization with rate $\tilde O\big(\frac{1}{n^{1/4}}+\big(\frac{d}{n^2}\big)^{1/6}\big)$ which matches the best existing non-private algorithm when $d= O(\sqrt{n})$. We also extend all our results above for the non-convex $\ell_2$ setting to the $\ell_p$ setting, where $1 < p \leq 2$, with only polylogarithmic (in the dimension) overhead in the rates.

        ----

        ## [713] TacticZero: Learning to Prove Theorems from Scratch with Deep Reinforcement Learning

        **Authors**: *Minchao Wu, Michael Norrish, Christian Walder, Amir Dezfouli*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/4dea382d82666332fb564f2e711cbc71-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/4dea382d82666332fb564f2e711cbc71-Abstract.html)

        **Abstract**:

        We propose a novel approach to interactive theorem-proving (ITP) using deep reinforcement learning. The proposed framework is able to learn proof search strategies as well as tactic and arguments prediction in an end-to-end manner. We formulate the process of ITP as a Markov decision process (MDP) in which each state represents a set of potential derivation paths. This structure allows us to introduce a novel backtracking mechanism which enables the agent to efficiently discard (predicted) dead-end derivations and restart the derivation from promising alternatives. We implement the framework in the HOL theorem prover. Experimental results show that the framework using learned search strategies outperforms existing automated theorem provers (i.e., hammers) available in HOL when evaluated on unseen problems. We further elaborate the role of key components of the framework using ablation studies.

        ----

        ## [714] Integrating Tree Path in Transformer for Code Representation

        **Authors**: *Han Peng, Ge Li, Wenhan Wang, Yunfei Zhao, Zhi Jin*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/4e0223a87610176ef0d24ef6d2dcde3a-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/4e0223a87610176ef0d24ef6d2dcde3a-Abstract.html)

        **Abstract**:

        Learning distributed representation of source code requires modelling its syntax and semantics. Recent state-of-the-art models leverage highly structured source code representations, such as the syntax trees and paths therein. In this paper, we investigate two representative path encoding methods shown in previous research work and integrate them into the attention module of Transformer. We draw inspiration from the ideas of positional encoding and modify them to incorporate these path encoding. Specifically, we encode both the pairwise path between tokens of source code and the path from the leaf node to the tree root for each token in the syntax tree. We explore the interaction between these two kinds of paths by integrating them into the unified Transformer framework. The detailed empirical study for path encoding methods also leads to our novel state-of-the-art representation model TPTrans, which finally outperforms strong baselines. Extensive experiments and ablation studies on code summarization across four different languages demonstrate the effectiveness of our approaches. We release our code at \url{https://github.com/AwdHanPeng/TPTrans}.

        ----

        ## [715] Twins: Revisiting the Design of Spatial Attention in Vision Transformers

        **Authors**: *Xiangxiang Chu, Zhi Tian, Yuqing Wang, Bo Zhang, Haibing Ren, Xiaolin Wei, Huaxia Xia, Chunhua Shen*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/4e0928de075538c593fbdabb0c5ef2c3-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/4e0928de075538c593fbdabb0c5ef2c3-Abstract.html)

        **Abstract**:

        Very recently, a variety of vision transformer architectures for dense prediction tasks have been proposed and they show that the design of spatial attention is critical to their success in these tasks. In this work, we revisit the design of the spatial attention and demonstrate that a carefully devised yet simple spatial attention mechanism performs favorably against the state-of-the-art schemes. As a result, we propose two vision transformer architectures, namely, Twins- PCPVT and Twins-SVT. Our proposed architectures are highly efficient and easy to implement, only involving matrix multiplications that are highly optimized in modern deep learning frameworks. More importantly, the proposed architectures achieve excellent performance on a wide range of visual tasks including image-level classification as well as dense detection and segmentation. The simplicity and strong performance suggest that our proposed architectures may serve as stronger backbones for many vision tasks.

        ----

        ## [716] Evaluating State-of-the-Art Classification Models Against Bayes Optimality

        **Authors**: *Ryan Theisen, Huan Wang, Lav R. Varshney, Caiming Xiong, Richard Socher*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/4e0ccd2b894f717df5ebc12f4282ee70-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/4e0ccd2b894f717df5ebc12f4282ee70-Abstract.html)

        **Abstract**:

        Evaluating the inherent difficulty of a given data-driven classification problem is important for establishing absolute benchmarks and evaluating progress in the field. To this end, a natural quantity to consider is the \emph{Bayes error}, which measures the optimal classification error theoretically achievable for a given data distribution.  While generally an intractable quantity, we show that we can compute the exact Bayes error of generative models learned using normalizing flows. Our technique relies on a fundamental result, which states that the Bayes error is invariant under invertible transformation. Therefore, we can compute the exact Bayes error of the learned flow models by computing it for Gaussian base distributions, which can be done efficiently using Holmes-Diaconis-Ross integration. Moreover, we show that by varying the temperature of the learned flow models, we can generate synthetic datasets that closely resemble standard benchmark datasets, but with almost any desired Bayes error. We use our approach to conduct a thorough investigation of state-of-the-art classification models, and find that in some --- but not all --- cases, these models are capable of obtaining accuracy very near optimal. Finally, we use our method to evaluate the intrinsic "hardness" of standard benchmark datasets.

        ----

        ## [717] Data-Efficient Instance Generation from Instance Discrimination

        **Authors**: *Ceyuan Yang, Yujun Shen, Yinghao Xu, Bolei Zhou*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/4e0d67e54ad6626e957d15b08ae128a6-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/4e0d67e54ad6626e957d15b08ae128a6-Abstract.html)

        **Abstract**:

        Generative Adversarial Networks (GANs) have significantly advanced image synthesis, however, the synthesis quality drops significantly given a limited amount of training data. To improve the data efficiency of GAN training, prior work typically employs data augmentation to mitigate the overfitting of the discriminator yet still learn the discriminator with a bi-classification ($\textit{i.e.}$, real $\textit{vs.}$ fake) task. In this work, we propose a data-efficient Instance Generation ($\textit{InsGen}$) method based on instance discrimination. Concretely, besides differentiating the real domain from the fake domain, the discriminator is required to distinguish every individual image, no matter it comes from the training set or from the generator. In this way, the discriminator can benefit from the infinite synthesized samples for training, alleviating the overfitting problem caused by insufficient training data. A noise perturbation strategy is further introduced to improve its discriminative power. Meanwhile, the learned instance discrimination capability from the discriminator is in turn exploited to encourage the generator for diverse generation. Extensive experiments demonstrate the effectiveness of our method on a variety of datasets and training settings. Noticeably, on the setting of $2K$ training images from the FFHQ dataset, we outperform the state-of-the-art approach with 23.5\% FID improvement.

        ----

        ## [718] Reliable Post hoc Explanations: Modeling Uncertainty in Explainability

        **Authors**: *Dylan Slack, Anna Hilgard, Sameer Singh, Himabindu Lakkaraju*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/4e246a381baf2ce038b3b0f82c7d6fb4-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/4e246a381baf2ce038b3b0f82c7d6fb4-Abstract.html)

        **Abstract**:

        As black box explanations are increasingly being employed to establish model credibility in high stakes settings, it is important to ensure that these explanations are accurate and reliable. However, prior work demonstrates that explanations generated by state-of-the-art techniques are inconsistent, unstable, and provide very little insight into their correctness and reliability. In addition, these methods are also computationally inefficient, and require significant hyper-parameter tuning. In this paper, we address the aforementioned challenges by developing a novel Bayesian framework for generating local explanations along with their associated uncertainty. We instantiate this framework to obtain Bayesian versions of LIME and KernelSHAP which  output credible intervals for the feature importances, capturing the associated uncertainty. The resulting explanations not only enable us to make concrete inferences about their quality (e.g., there is a 95% chance that the feature importance lies within the given range), but are also highly consistent and stable. We carry out a detailed theoretical analysis that leverages the aforementioned uncertainty to estimate how many perturbations to sample, and how to sample for faster convergence.  This work makes the first attempt at addressing several critical issues with popular explanation methods in one shot, thereby generating consistent, stable, and reliable explanations with guarantees in a computationally efficient manner. Experimental evaluation with multiple real world datasets and user studies demonstrate that the efficacy of the proposed framework.

        ----

        ## [719] Learning Graph Models for Retrosynthesis Prediction

        **Authors**: *Vignesh Ram Somnath, Charlotte Bunne, Connor W. Coley, Andreas Krause, Regina Barzilay*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/4e2a6330465c8ffcaa696a5a16639176-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/4e2a6330465c8ffcaa696a5a16639176-Abstract.html)

        **Abstract**:

        Retrosynthesis prediction is a fundamental problem in organic synthesis, where the task is to identify precursor molecules that can be used to synthesize a target molecule. A key consideration in building neural models for this task is aligning model design with strategies adopted by chemists. Building on this viewpoint, this paper introduces a graph-based approach that capitalizes on the idea that the graph topology of precursor molecules is largely unaltered during a chemical reaction. The model first predicts the set of graph edits transforming the target into incomplete molecules called synthons. Next, the model learns to expand synthons into complete molecules by attaching relevant leaving groups. This decomposition simplifies the architecture, making its predictions more interpretable, and also amenable to manual correction. Our model achieves a top-1 accuracy of 53.7%, outperforming previous template-free and semi-template-based methods.

        ----

        ## [720] Differentiable Equilibrium Computation with Decision Diagrams for Stackelberg Models of Combinatorial Congestion Games

        **Authors**: *Shinsaku Sakaue, Kengo Nakamura*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/4e4b5fbbbb602b6d35bea8460aa8f8e5-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/4e4b5fbbbb602b6d35bea8460aa8f8e5-Abstract.html)

        **Abstract**:

        We address Stackelberg models of combinatorial congestion games (CCGs); we aim to optimize the parameters of CCGs so that the selfish behavior of non-atomic players attains desirable equilibria. This model is essential for designing such social infrastructures as traffic and communication networks. Nevertheless, computational approaches to the model have not been thoroughly studied due to two difficulties: (I) bilevel-programming structures and (II) the combinatorial nature of CCGs. We tackle them by carefully combining (I) the idea of \textit{differentiable} optimization and (II) data structures called \textit{zero-suppressed binary decision diagrams} (ZDDs), which can compactly represent sets of combinatorial strategies. Our algorithm numerically approximates the equilibria of CCGs, which we can differentiate with respect to parameters of CCGs by automatic differentiation. With the resulting derivatives, we can apply gradient-based methods to Stackelberg models of CCGs. Our method is tailored to induce Nesterov's acceleration and can fully utilize the empirical compactness of ZDDs. These technical advantages enable us to deal with CCGs with a vast number of combinatorial strategies. Experiments on real-world network design instances demonstrate the practicality of our method.

        ----

        ## [721] Inverse Optimal Control Adapted to the Noise Characteristics of the Human Sensorimotor System

        **Authors**: *Matthias Schultheis, Dominik Straub, Constantin A. Rothkopf*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/4e55139e019a58e0084f194f758ffdea-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/4e55139e019a58e0084f194f758ffdea-Abstract.html)

        **Abstract**:

        Computational level explanations based on optimal feedback control with signal-dependent noise have been able to account for a vast array of phenomena in human sensorimotor behavior. However, commonly a cost function needs to be assumed for a task and the optimality of human behavior is evaluated by comparing observed and predicted trajectories. Here, we introduce inverse optimal control with signal-dependent noise, which allows inferring the cost function from observed behavior. To do so, we formalize the problem as a partially observable Markov decision process and distinguish between the agent’s and the experimenter’s inference problems.  Specifically, we derive a probabilistic formulation of the evolution of states and belief states and an approximation to the propagation equation in the linear-quadratic Gaussian problem with signal-dependent noise. We extend the model to the case of partial observability of state variables from the point of view of the experimenter. We show the feasibility of the approach through validation on synthetic data and application to experimental data. Our approach enables recovering the costs and benefits implicit in human sequential sensorimotor behavior, thereby reconciling normative and descriptive approaches in a computational framework.

        ----

        ## [722] Deep Neural Networks as Point Estimates for Deep Gaussian Processes

        **Authors**: *Vincent Dutordoir, James Hensman, Mark van der Wilk, Carl Henrik Ek, Zoubin Ghahramani, Nicolas Durrande*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/4e6cd95227cb0c280e99a195be5f6615-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/4e6cd95227cb0c280e99a195be5f6615-Abstract.html)

        **Abstract**:

        Neural networks and Gaussian processes are complementary in their strengths and weaknesses. Having a better understanding of their relationship comes with the promise to make each method benefit from the strengths of the other. In this work, we establish an equivalence between the forward passes of neural networks and (deep) sparse Gaussian process models. The theory we develop is based on interpreting activation functions as interdomain inducing features through a rigorous analysis of the interplay between activation functions and kernels. This results in models that can either be seen as neural networks with improved uncertainty prediction or deep Gaussian processes with increased prediction accuracy. These claims are supported by experimental results on regression and classification datasets.

        ----

        ## [723] Locality defeats the curse of dimensionality in convolutional teacher-student scenarios

        **Authors**: *Alessandro Favero, Francesco Cagnetta, Matthieu Wyart*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/4e8eaf897c638d519710b1691121f8cb-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/4e8eaf897c638d519710b1691121f8cb-Abstract.html)

        **Abstract**:

        Convolutional neural networks perform a local and translationally-invariant treatment of the data: quantifying which of these two aspects is central to their success remains a challenge. We study this problem within a teacher-student framework for kernel regression, using 'convolutional' kernels inspired by the neural tangent kernel of simple convolutional architectures of given filter size. Using heuristic methods from physics, we find in the ridgeless case that locality is key in determining the learning curve exponent $\beta$ (that relates the test error $\epsilon_t\sim P^{-\beta}$ to the size of the training set $P$), whereas translational invariance is not. In particular, if the filter size of the teacher $t$ is smaller than that of the student $s$, $\beta$ is a function of $s$ only and does not depend on the input dimension. We confirm our predictions on $\beta$ empirically. We conclude by proving, under a natural universality assumption, that performing kernel regression with a ridge that decreases with the size of the training set leads to similar learning curve exponents to those we obtain in the ridgeless case.

        ----

        ## [724] Causal Identification with Matrix Equations

        **Authors**: *Sanghack Lee, Elias Bareinboim*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/4ea06fbc83cdd0a06020c35d50e1e89a-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/4ea06fbc83cdd0a06020c35d50e1e89a-Abstract.html)

        **Abstract**:

        Causal effect identification is concerned with determining whether a causal effect is computable from a combination of qualitative assumptions about the underlying system (e.g., a causal graph) and distributions collected from this system. Many identification algorithms exclusively rely on graphical criteria made of a non-trivial combination of probability axioms, do-calculus, and refined c-factorization (e.g., Lee & Bareinboim, 2020). In a sequence of increasingly sophisticated results, it has been shown how proxy variables can be used to identify certain effects that would not be otherwise recoverable in challenging scenarios through solving matrix equations (e.g., Kuroki & Pearl, 2014; Miao et al., 2018). In this paper, we develop a new causal identification algorithm which utilizes both graphical criteria and matrix equations. Specifically, we first characterize the relationships between certain graphically-driven formulae and matrix multiplications. With such characterizations, we broaden the spectrum of proxy variable based identification conditions and further propose novel intermediary criteria based on the pseudoinverse of a matrix. Finally, we devise a causal effect identification algorithm, which accepts as input a collection of marginal, conditional, and interventional distributions, integrating enriched matrix-based criteria into a graphical identification approach.

        ----

        ## [725] Private and Non-private Uniformity Testing for Ranking Data

        **Authors**: *Róbert Busa-Fekete, Dimitris Fotakis, Emmanouil Zampetakis*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/4eb0194ddf4d6c7a72dca4fd3149e92e-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/4eb0194ddf4d6c7a72dca4fd3149e92e-Abstract.html)

        **Abstract**:

        We study the problem of uniformity testing for statistical data that consists of rankings over $m$ items where the alternative class is restricted to Mallows models with single parameter. Testing ranking data is challenging because of the size of the large domain that is factorial in $m$, therefore the tester needs to take advantage of some structure of the alternative class. We show that uniform distribution can be distinguished from Mallows model with $O(m^{-1/2})$ samples based on simple pairwise statistics, which allows us to test uniformity using only two samples, if $m$ is large enough. We also consider uniformity testing with central and locally differential private (DP) constraints. We present a central DP algorithm that requires $O\left(\max \{ 1/\epsilon_0, 1/\sqrt{m} \} \right)$ where $\epsilon_0$ is the privacy budget parameter. Interestingly, our uniformity testing algorithm is  straightforward to apply in the local DP scenario by its nature, since it works with binary statistics that is extracted from the ranking data. We carry out large-scale experiments, including $m=10000$, to show that these testing algorithms scales very gracefully with the number of items.

        ----

        ## [726] Model-Based Reinforcement Learning via Imagination with Derived Memory

        **Authors**: *Yao Mu, Yuzheng Zhuang, Bin Wang, Guangxiang Zhu, Wulong Liu, Jianyu Chen, Ping Luo, Shengbo Li, Chongjie Zhang, Jianye Hao*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/4ebccfb3e317c7789f04f7a558df4537-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/4ebccfb3e317c7789f04f7a558df4537-Abstract.html)

        **Abstract**:

        Model-based reinforcement learning aims to improve the sample efficiency of policy learning by modeling the dynamics of the environment. Recently, the latent dynamics model is further developed to enable fast planning in a compact space. It summarizes the high-dimensional experiences of an agent, which mimics the memory function of humans.  Learning policies via imagination with the latent model shows great potential for solving complex tasks. However, only considering memories from the true experiences in the process of imagination could limit its advantages. Inspired by the memory prosthesis proposed by neuroscientists, we present a novel model-based reinforcement learning framework called Imagining with Derived Memory (IDM). It enables the agent to learn policy from enriched diverse imagination with prediction-reliability weight, thus improving sample efficiency and policy robustness. Experiments on various high-dimensional visual control tasks in the DMControl benchmark demonstrate that IDM outperforms previous state-of-the-art methods in terms of policy robustness and further improves the sample efficiency of the model-based method.

        ----

        ## [727] Compositional Transformers for Scene Generation

        **Authors**: *Dor Arad Hudson, Larry Zitnick*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/4eff0720836a198b6174eecf02cbfdbf-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/4eff0720836a198b6174eecf02cbfdbf-Abstract.html)

        **Abstract**:

        We introduce the GANformer2 model, an iterative object-oriented transformer, explored for the task of generative modeling. The network incorporates strong and explicit structural priors, to reflect the compositional nature of visual scenes, and synthesizes images through a sequential process. It operates in two stages: a fast and lightweight planning phase, where we draft a high-level scene layout, followed by an attention-based execution phase, where the layout is being refined, evolving into a rich and detailed picture. Our model moves away from conventional black-box GAN architectures that feature a flat and monolithic latent space towards a transparent design that encourages efficiency, controllability and interpretability. We demonstrate GANformer2's strengths and qualities through a careful evaluation over a range of datasets, from multi-object CLEVR scenes to the challenging COCO images, showing it successfully achieves state-of-the-art performance in terms of visual quality, diversity and consistency. Further experiments demonstrate the model's disentanglement and provide a deeper insight into its generative process, as it proceeds step-by-step from a rough initial sketch, to a detailed layout that accounts for objects' depths and dependencies, and up to the final high-resolution depiction of vibrant and intricate real-world scenes. See https://github.com/dorarad/gansformer for model implementation.

        ----

        ## [728] An Exponential Lower Bound for Linearly Realizable MDP with Constant Suboptimality Gap

        **Authors**: *Yuanhao Wang, Ruosong Wang, Sham M. Kakade*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/4f00921114932db3f8662a41b44ee68f-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/4f00921114932db3f8662a41b44ee68f-Abstract.html)

        **Abstract**:

        A fundamental question in the theory of reinforcement learning is: suppose the optimal $Q$-function lies in the linear span of a given $d$ dimensional feature mapping, is sample-efficient reinforcement learning (RL) possible? The recent and remarkable result of Weisz et al. (2020) resolves this question in the negative, providing an exponential (in $d$) sample size lower bound, which holds even if the agent has access to a generative model of the environment. One may hope that such a lower can be circumvented with an even stronger assumption that there is a \emph{constant gap} between the optimal $Q$-value of the best action and that of the second-best action (for all states); indeed, the construction in Weisz et al. (2020) relies on having an exponentially small gap. This work resolves this subsequent question, showing that an exponential sample complexity lower bound still holds even if a constant gap is assumed.  Perhaps surprisingly, this result implies an exponential separation between the online RL setting and the generative model setting, where sample-efficient RL is in fact possible in the latter setting with a constant gap. Complementing our negative hardness result, we give two positive results showing that provably sample-efficient RL is possible either under an additional low-variance assumption or under a novel hypercontractivity assumption.

        ----

        ## [729] Combating Noise: Semi-supervised Learning by Region Uncertainty Quantification

        **Authors**: *Zhenyu Wang, Ya-Li Li, Ye Guo, Shengjin Wang*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/4f16c818875d9fcb6867c7bdc89be7eb-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/4f16c818875d9fcb6867c7bdc89be7eb-Abstract.html)

        **Abstract**:

        Semi-supervised learning aims to leverage a large amount of unlabeled data for performance boosting. Existing works primarily focus on image classification. In this paper, we delve into semi-supervised learning for object detection, where labeled data are more labor-intensive to collect. Current methods are easily distracted by noisy regions generated by pseudo labels. To combat the noisy labeling, we propose noise-resistant semi-supervised learning by quantifying the region uncertainty. We first investigate the adverse effects brought by different forms of noise associated with pseudo labels. Then we propose to quantify the uncertainty of regions by identifying the noise-resistant properties of regions over different strengths. By importing the region uncertainty quantification and promoting multi-peak probability distribution output, we introduce uncertainty into training and further achieve noise-resistant learning. Experiments on both PASCAL VOC and MS COCO demonstrate the extraordinary performance of our method.

        ----

        ## [730] Reducing the Covariate Shift by Mirror Samples in Cross Domain Alignment

        **Authors**: *Yin Zhao, Minquan Wang, Longjun Cai*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/4f284803bd0966cc24fa8683a34afc6e-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/4f284803bd0966cc24fa8683a34afc6e-Abstract.html)

        **Abstract**:

        Eliminating the covariate shift cross domains is one of the common methods to deal with the issue of domain shift in visual unsupervised domain adaptation. However, current alignment methods, especially the prototype based or sample-level based methods neglect the structural properties of the underlying distribution and even break the condition of covariate shift. To relieve the limitations and conflicts, we introduce a novel concept named (virtual) mirror, which represents the equivalent sample in another domain. The equivalent sample pairs, named mirror pairs reflect the natural correspondence of the empirical distributions. Then a mirror loss, which aligns the mirror pairs cross domains, is constructed to enhance the alignment of the domains. The proposed method does not distort the internal structure of the underlying distribution. We also provide theoretical proof that the mirror samples and mirror loss have better asymptotic properties in reducing the domain shift. By applying the virtual mirror and mirror loss to the generic unsupervised domain adaptation model, we achieved consistently superior performance on several mainstream benchmarks.

        ----

        ## [731] Permutation-Invariant Variational Autoencoder for Graph-Level Representation Learning

        **Authors**: *Robin Winter, Frank Noé, Djork-Arné Clevert*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/4f3d7d38d24b740c95da2b03dc3a2333-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/4f3d7d38d24b740c95da2b03dc3a2333-Abstract.html)

        **Abstract**:

        Recently, there has been great success in applying deep neural networks on graph structured data. Most work, however, focuses on either node- or graph-level supervised learning, such as node, link or graph classification or node-level unsupervised learning (e.g. node clustering). Despite its wide range of possible applications, graph-level unsupervised learning has not received much attention yet. This might be mainly attributed to the high representation complexity of graphs, which can be represented by $n!$ equivalent adjacency matrices, where $n$ is the number of nodes.In this work we address this issue by proposing a permutation-invariant variational autoencoder for graph structured data. Our proposed model indirectly learns to match the node ordering of input and output graph, without imposing a particular node ordering or performing expensive graph matching. We demonstrate the effectiveness of our proposed model for graph reconstruction, generation and interpolation and evaluate the expressive power of extracted representations for downstream graph-level classification and regression.

        ----

        ## [732] Causal Abstractions of Neural Networks

        **Authors**: *Atticus Geiger, Hanson Lu, Thomas Icard, Christopher Potts*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/4f5c422f4d49a5a807eda27434231040-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/4f5c422f4d49a5a807eda27434231040-Abstract.html)

        **Abstract**:

        Structural analysis methods (e.g., probing and feature attribution) are increasingly important tools for neural network analysis. We propose a new structural analysis method grounded in a formal theory of causal abstraction that provides rich characterizations of model-internal representations and their roles in input/output behavior. In this method, neural representations are aligned with variables in interpretable causal models, and then interchange interventions are used to experimentally verify that the neural representations have the causal properties of their aligned variables. We apply this method in a case study to analyze neural models trained on Multiply Quantified Natural Language Inference (MQNLI) corpus, a highly complex NLI dataset that was constructed with a tree-structured natural logic causal model. We discover that a BERT-based model with state-of-the-art performance successfully realizes parts of the natural logic modelâ€™s causal structure, whereas a simpler baseline model fails to show any such structure, demonstrating that neural representations encode the compositional structure of MQNLI examples.

        ----

        ## [733] Conic Blackwell Algorithm: Parameter-Free Convex-Concave Saddle-Point Solving

        **Authors**: *Julien Grand-Clément, Christian Kroer*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/4f87658ef0de194413056248a00ce009-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/4f87658ef0de194413056248a00ce009-Abstract.html)

        **Abstract**:

        We develop new parameter-free and scale-free algorithms for solving convex-concave saddle-point problems. Our results are based on a new simple regret minimizer, the Conic Blackwell Algorithm$^+$ (CBA$^+$), which attains $O(1/\sqrt{T})$ average regret. Intuitively, our approach generalizes to other decision sets of interest ideas from the Counterfactual Regret minimization (CFR$^+$) algorithm, which has very strong practical performance for solving sequential games on simplexes.We show how to implement CBA$^+$ for the simplex, $\ell_{p}$ norm balls, and ellipsoidal confidence regions in the simplex, and we present numerical experiments for solving matrix games and distributionally robust optimization problems.Our empirical results show that CBA$^+$ is a simple algorithm that outperforms state-of-the-art methods on synthetic data and real data instances, without the need for any choice of step sizes or other algorithmic parameters.

        ----

        ## [734] 3DP3: 3D Scene Perception via Probabilistic Programming

        **Authors**: *Nishad Gothoskar, Marco F. Cusumano-Towner, Ben Zinberg, Matin Ghavamizadeh, Falk Pollok, Austin Garrett, Josh Tenenbaum, Dan Gutfreund, Vikash K. Mansinghka*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/4fc66104f8ada6257fa55f29a2a567c7-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/4fc66104f8ada6257fa55f29a2a567c7-Abstract.html)

        **Abstract**:

        We present 3DP3, a framework for inverse graphics that uses inference in a structured generative model of objects, scenes, and images. 3DP3 uses (i) voxel models to represent the 3D shape of objects, (ii) hierarchical scene graphs to decompose scenes into objects and the contacts between them, and (iii) depth image likelihoods based on real-time graphics. Given an observed RGB-D image, 3DP3's inference algorithm infers the underlying latent 3D scene, including the object poses and a parsimonious joint parametrization of these poses, using fast bottom-up pose proposals, novel involutive MCMC updates of the scene graph structure, and, optionally, neural object detectors and pose estimators. We show that 3DP3 enables scene understanding that is aware of 3D shape, occlusion, and contact structure. Our results demonstrate that 3DP3 is more accurate at 6DoF object pose estimation from real images than deep learning baselines and shows better generalization to challenging scenes with novel viewpoints, contact, and partial observability.

        ----

        ## [735] Novel Upper Bounds for the Constrained Most Probable Explanation Task

        **Authors**: *Tahrima Rahman, Sara Rouhani, Vibhav Gogate*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/4fc7e9c4df30aafd8b7e1ab324f27712-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/4fc7e9c4df30aafd8b7e1ab324f27712-Abstract.html)

        **Abstract**:

        We propose several schemes for upper bounding the optimal value of the constrained most probable explanation (CMPE) problem. Given a set of discrete random variables, two probabilistic graphical models defined over them and a real number $q$, this problem involves finding an assignment of values to all the variables such that the probability of the assignment is maximized according to the first model and is bounded by $q$ w.r.t. the second model. In prior work, it was shown that CMPE is a unifying problem with several applications and special cases including the nearest assignment problem, the decision preserving most probable explanation task and robust estimation. It was also shown that CMPE is NP-hard even on tractable models such as bounded treewidth networks and is hard for integer linear programming methods because it includes a dense global constraint. The main idea in our approach is to simplify the problem via Lagrange relaxation and decomposition to yield either a knapsack problem or the unconstrained most probable explanation (MPE) problem, and then solving the two problems, respectively using specialized knapsack algorithms and mini-buckets based upper bounding schemes. We evaluate our proposed scheme along several dimensions including quality of the bounds and computation time required on various benchmark graphical models and how it can be used to find heuristic, near-optimal feasible solutions in an example application pertaining to robust estimation and adversarial attacks on classifiers.

        ----

        ## [736] Why Spectral Normalization Stabilizes GANs: Analysis and Improvements

        **Authors**: *Zinan Lin, Vyas Sekar, Giulia Fanti*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/4ffb0d2ba92f664c2281970110a2e071-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/4ffb0d2ba92f664c2281970110a2e071-Abstract.html)

        **Abstract**:

        Spectral normalization (SN) is a widely-used technique for improving the stability and sample quality of Generative Adversarial Networks (GANs). However, current understanding of SN's efficacy is limited. In this work, we show that SN controls two important failure modes of GAN training: exploding and vanishing gradients. Our proofs illustrate a (perhaps unintentional) connection with the successful LeCun initialization. This connection helps to explain why the most popular implementation of SN for GANs requires no hyper-parameter tuning, whereas stricter implementations of SN have poor empirical performance out-of-the-box. Unlike LeCun initialization which only controls gradient vanishing at the beginning of training, SN preserves this property throughout training. Building on this theoretical understanding, we propose a new spectral normalization technique: Bidirectional Scaled Spectral Normalization (BSSN), which incorporates insights from later improvements to LeCun initialization: Xavier initialization and Kaiming initialization. Theoretically, we show that BSSN gives better gradient control than SN. Empirically, we demonstrate that it outperforms SN in sample quality and training stability on several benchmark datasets.

        ----

        ## [737] $(\textrm{Implicit})^2$: Implicit Layers for Implicit Representations

        **Authors**: *Zhichun Huang, Shaojie Bai, J. Zico Kolter*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/4ffbd5c8221d7c147f8363ccdc9a2a37-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/4ffbd5c8221d7c147f8363ccdc9a2a37-Abstract.html)

        **Abstract**:

        Recent research in deep learning has investigated two very different forms of ''implicitness'': implicit representations model high-frequency data such as images or 3D shapes directly via a low-dimensional neural network (often using e.g., sinusoidal bases or nonlinearities); implicit layers, in contrast, refer to techniques where the forward pass of a network is computed via non-linear dynamical systems, such as fixed-point or differential equation solutions, with the backward pass computed via the implicit function theorem.  In this work, we demonstrate that these two seemingly orthogonal concepts are remarkably well-suited for each other. In particular, we show that by exploiting fixed-point implicit layer to model implicit representations, we can substantially improve upon the performance of the conventional explicit-layer-based approach. Additionally, as implicit representation networks are typically trained in large-batch settings, we propose to leverage the property of implicit layers to amortize the cost of fixed-point forward/backward passes over training steps -- thereby addressing one of the primary challenges with implicit layers (that many iterations are required for the black-box fixed-point solvers). We empirically evaluated our method on learning multiple implicit representations for images, videos and audios, showing that our $(\textrm{Implicit})^2$ approach substantially improve upon existing models while being both faster to train and much more memory efficient.

        ----

        ## [738] Best Arm Identification in Contaminated Stochastic Bandits

        **Authors**: *Arpan Mukherjee, Ali Tajer, Pin-Yu Chen, Payel Das*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/500ee9106e0e4d8f769fadfdf9f2837e-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/500ee9106e0e4d8f769fadfdf9f2837e-Abstract.html)

        **Abstract**:

        This paper investigates the problem of best arm identification in {\sl contaminated} stochastic multi-arm bandits. In this setting, the rewards obtained from any arm are replaced by samples from an adversarial model with probability $\varepsilon$. A fixed confidence (infinite-horizon) setting is considered, where the goal of the learner is to identify the arm with the largest mean. Owing to the adversarial contamination of the rewards, each arm's mean is only partially identifiable. This paper proposes two algorithms, a gap-based algorithm and one based on the successive elimination, for best arm identification in sub-Gaussian bandits. These algorithms involve mean estimates that achieve the optimal error guarantee on the deviation of the true mean from the estimate asymptotically. Furthermore, these algorithms asymptotically achieve the optimal sample complexity. Specifically, for the gap-based algorithm, the sample complexity is asymptotically optimal up to constant factors, while for the successive elimination-based algorithm, it is optimal up to logarithmic factors. Finally, numerical experiments are provided to illustrate the gains of the algorithms compared to the existing baselines.

        ----

        ## [739] MADE: Exploration via Maximizing Deviation from Explored Regions

        **Authors**: *Tianjun Zhang, Paria Rashidinejad, Jiantao Jiao, Yuandong Tian, Joseph E. Gonzalez, Stuart Russell*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/5011bf6d8a37692913fce3a15a51f070-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/5011bf6d8a37692913fce3a15a51f070-Abstract.html)

        **Abstract**:

        In online reinforcement learning (RL), efficient exploration remains particularly challenging in high-dimensional environments with sparse rewards. In low-dimensional environments, where tabular parameterization is possible, count-based upper confidence bound (UCB) exploration methods achieve minimax near-optimal rates. However, it remains unclear how to efficiently implement UCB in realistic RL tasks that involve non-linear function approximation. To address this, we propose a new exploration approach via maximizing the deviation of the occupancy of the next policy from the explored regions. We add this term as an adaptive regularizer to the standard RL objective to balance exploration vs. exploitation.  We pair the new objective with a provably convergent algorithm, giving rise to a new intrinsic reward that adjusts existing bonuses. The proposed intrinsic reward is easy to implement and combine with other existing RL algorithms to conduct exploration. As a proof of concept, we evaluate the new intrinsic reward on tabular examples across a variety of model-based and model-free algorithms, showing improvements over count-only exploration strategies. When tested on navigation and locomotion tasks from MiniGrid and DeepMind Control Suite benchmarks, our approach significantly improves sample efficiency over state-of-the-art methods.

        ----

        ## [740] Variational Automatic Curriculum Learning for Sparse-Reward Cooperative Multi-Agent Problems

        **Authors**: *Jiayu Chen, Yuanxin Zhang, Yuanfan Xu, Huimin Ma, Huazhong Yang, Jiaming Song, Yu Wang, Yi Wu*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/503e7dbbd6217b9a591f3322f39b5a6c-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/503e7dbbd6217b9a591f3322f39b5a6c-Abstract.html)

        **Abstract**:

        We introduce an automatic curriculum algorithm, Variational Automatic Curriculum Learning (VACL), for solving challenging goal-conditioned cooperative multi-agent reinforcement learning problems. We motivate our curriculum learning paradigm through a variational perspective, where the learning objective can be decomposed into two terms: task learning on the current curriculum, and curriculum update to a new task distribution. Local optimization over the second term suggests that the curriculum should gradually expand the training tasks from easy to hard. Our VACL algorithm implements this variational paradigm with two practical components, task expansion and entity curriculum, which produces a series of training tasks over both the task configurations as well as the number of entities in the task. Experiment results show that VACL solves a collection of sparse-reward problems with a large number of agents. Particularly, using a single desktop machine, VACL achieves 98% coverage rate with 100 agents in the simple-spread benchmark and reproduces the ramp-use behavior originally shown in OpenAIâ€™s hide-and-seek project.

        ----

        ## [741] Align before Fuse: Vision and Language Representation Learning with Momentum Distillation

        **Authors**: *Junnan Li, Ramprasaath R. Selvaraju, Akhilesh Gotmare, Shafiq R. Joty, Caiming Xiong, Steven Chu-Hong Hoi*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/505259756244493872b7709a8a01b536-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/505259756244493872b7709a8a01b536-Abstract.html)

        **Abstract**:

        Large-scale vision and language representation learning has shown promising improvements on various vision-language tasks. Most existing methods employ a transformer-based multimodal encoder to jointly model visual tokens (region-based image features) and word tokens. Because the visual tokens and word tokens are unaligned, it is challenging for the multimodal encoder to learn image-text interactions. In this paper, we introduce a contrastive loss to ALign the image and text representations BEfore Fusing (ALBEF) them through cross-modal attention, which enables more grounded vision and language representation learning. Unlike most existing methods, our method does not require bounding box annotations nor high-resolution images. In order to improve learning from noisy web data, we propose momentum distillation, a self-training method which learns from pseudo-targets produced by a momentum model. We provide a theoretical analysis of ALBEF from a mutual information maximization perspective, showing that different training tasks can be interpreted as different ways to generate views for an image-text pair. ALBEF achieves state-of-the-art performance on multiple downstream vision-language tasks. On image-text retrieval, ALBEF outperforms methods that are pre-trained on orders of magnitude larger datasets. On VQA and NLVR$^2$, ALBEF achieves absolute improvements of 2.37% and 3.84% compared to the state-of-the-art, while enjoying faster inference speed. Code and models are available at https://github.com/salesforce/ALBEF.

        ----

        ## [742] Variational Model Inversion Attacks

        **Authors**: *Kuan-Chieh Wang, Yan Fu, Ke Li, Ashish Khisti, Richard S. Zemel, Alireza Makhzani*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/50a074e6a8da4662ae0a29edde722179-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/50a074e6a8da4662ae0a29edde722179-Abstract.html)

        **Abstract**:

        Given the ubiquity of deep neural networks, it is important that these models do not reveal information about sensitive data that they have been trained on. In model inversion attacks, a malicious user attempts to recover the private dataset used to train a supervised neural network. A successful model inversion attack should generate realistic and diverse samples that accurately describe each of the classes in the private dataset. In this work, we provide a probabilistic interpretation of model inversion attacks, and formulate a variational objective that accounts for both diversity and accuracy. In order to optimize this variational objective, we choose a variational family defined in the code space of a deep generative model, trained on a public auxiliary dataset that shares some structural similarity with the target dataset.  Empirically, our method substantially improves performance in terms of target attack accuracy, sample realism, and diversity on datasets of faces and chest X-ray images.

        ----

        ## [743] Graph Neural Networks with Adaptive Residual

        **Authors**: *Xiaorui Liu, Jiayuan Ding, Wei Jin, Han Xu, Yao Ma, Zitao Liu, Jiliang Tang*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/50abc3e730e36b387ca8e02c26dc0a22-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/50abc3e730e36b387ca8e02c26dc0a22-Abstract.html)

        **Abstract**:

        Graph neural networks (GNNs) have shown the power in graph representation learning for numerous tasks. In this work, we discover an interesting phenomenon that although residual connections in the message passing of GNNs help improve the performance, they immensely amplify GNNs' vulnerability against abnormal node features. This is undesirable because in real-world applications, node features in graphs could often be abnormal such as being naturally noisy or adversarially manipulated. We analyze possible reasons to understand this phenomenon and aim to design GNNs with stronger resilience to abnormal features. Our understandings motivate us to propose and derive a simple, efficient, interpretable, and adaptive message passing scheme, leading to a novel GNN with Adaptive Residual, AirGNN. Extensive experiments under various abnormal feature scenarios demonstrate the effectiveness of the proposed algorithm.

        ----

        ## [744] Efficient Active Learning for Gaussian Process Classification by Error Reduction

        **Authors**: *Guang Zhao, Edward R. Dougherty, Byung-Jun Yoon, Francis J. Alexander, Xiaoning Qian*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/50d2e70cdf7dd05be85e1b8df3f8ced4-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/50d2e70cdf7dd05be85e1b8df3f8ced4-Abstract.html)

        **Abstract**:

        Active learning sequentially selects the best instance for labeling by optimizing an acquisition function to enhance data/label efficiency. The selection can be either from a discrete instance set (pool-based scenario) or a continuous instance space (query synthesis scenario). In this work, we study both active learning scenarios for Gaussian Process Classification (GPC). The existing active learning strategies that  maximize the Estimated Error Reduction (EER) aim at reducing the classification error after training with the new acquired instance in a one-step-look-ahead manner. The computation of EER-based acquisition functions is typically prohibitive as it requires retraining the GPC with every new query. Moreover, as the EER is not smooth, it can not be combined with gradient-based optimization techniques to efficiently explore the continuous instance space for query synthesis. To overcome these critical limitations, we develop computationally efficient algorithms for EER-based active learning with GPC. We derive the joint predictive distribution of label pairs as a one-dimensional integral, as a result of which the computation of the  acquisition function avoids retraining the GPC for each query, remarkably reducing the computational overhead. We also derive the gradient chain rule to efficiently calculate the gradient of the acquisition function, which leads to the first query synthesis active learning algorithm implementing EER-based strategies. Our experiments clearly demonstrate the computational efficiency of the proposed algorithms. We also benchmark our algorithms on both synthetic and real-world datasets, which show superior performance in terms of sampling efficiency compared to the existing state-of-the-art algorithms.

        ----

        ## [745] Non-Asymptotic Analysis for Two Time-scale TDC with General Smooth Function Approximation

        **Authors**: *Yue Wang, Shaofeng Zou, Yi Zhou*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/50e207ab6946b5d78b377ae0144b9e07-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/50e207ab6946b5d78b377ae0144b9e07-Abstract.html)

        **Abstract**:

        Temporal-difference learning with gradient correction (TDC) is a two time-scale algorithm for policy evaluation in reinforcement learning. This algorithm was initially proposed with linear function approximation, and was later extended to the one with general smooth function approximation. The asymptotic convergence for the on-policy setting with general smooth function approximation was established in [Bhatnagar et al., 2009], however, the non-asymptotic convergence analysis remains unsolved due to challenges in the non-linear and two-time-scale update structure, non-convex objective function and the projection onto a time-varying tangent plane. In this paper, we develop novel techniques to address the above challenges and explicitly characterize the non-asymptotic error bound for the general off-policy setting with i.i.d. or Markovian samples, and show that it converges as fast as $\mathcal O(1/\sqrt T)$ (up to a factor of $\mathcal O(\log T)$). Our approach can be applied to a wide range of value-based reinforcement learning algorithms with general smooth function approximation.

        ----

        ## [746] A Little Robustness Goes a Long Way: Leveraging Robust Features for Targeted Transfer Attacks

        **Authors**: *Jacob M. Springer, Melanie Mitchell, Garrett T. Kenyon*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/50f3f8c42b998a48057e9d33f4144b8b-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/50f3f8c42b998a48057e9d33f4144b8b-Abstract.html)

        **Abstract**:

        Adversarial examples for neural network image classifiers are known to be transferable: examples optimized to be misclassified by a source classifier are often misclassified as well by classifiers with different architectures. However, targeted adversarial examples—optimized to be classified as a chosen target class—tend to be less transferable between architectures. While prior research on constructing transferable targeted attacks has focused on improving the optimization procedure, in this work we examine the role of the source classifier. Here, we show that training the source classifier to be "slightly robust"—that is, robust to small-magnitude adversarial examples—substantially improves the transferability of class-targeted and representation-targeted adversarial attacks, even between architectures as different as convolutional neural networks and transformers. The results we present provide insight into the nature of adversarial examples as well as the mechanisms underlying so-called "robust" classifiers.

        ----

        ## [747] TriBERT: Human-centric Audio-visual Representation Learning

        **Authors**: *Tanzila Rahman, Mengyu Yang, Leonid Sigal*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/51200d29d1fc15f5a71c1dab4bb54f7c-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/51200d29d1fc15f5a71c1dab4bb54f7c-Abstract.html)

        **Abstract**:

        The recent success of transformer models in language, such as BERT, has motivated the use of such architectures for multi-modal feature learning and tasks. However, most multi-modal variants (e.g., ViLBERT) have limited themselves to visual-linguistic data. Relatively few have explored its use in audio-visual modalities, and none, to our knowledge, illustrate them in the context of granular audio-visual detection or segmentation tasks such as sound source separation and localization. In this work, we introduce TriBERT -- a transformer-based architecture, inspired by ViLBERT, which enables contextual feature learning across three modalities: vision, pose, and audio, with the use of flexible co-attention. The use of pose keypoints is inspired by recent works that illustrate that such representations can significantly boost performance in many audio-visual scenarios where often one or more persons are responsible for the sound explicitly (e.g., talking) or implicitly (e.g., sound produced as a function of human manipulating an object). From a technical perspective, as part of the TriBERT architecture, we introduce a learned visual tokenization scheme based on spatial attention and leverage weak-supervision to allow granular cross-modal interactions for visual and pose modalities. Further, we supplement learning with sound-source separation loss formulated across all three streams. We pre-train our model on the large MUSIC21 dataset and demonstrate improved performance in audio-visual sound source separation on that dataset as well as other datasets through fine-tuning. In addition, we show that the learned TriBERT representations are generic and significantly improve performance on other audio-visual tasks such as cross-modal audio-visual-pose retrieval by as much as 66.7% in top-1 accuracy.

        ----

        ## [748] How does a Neural Network's Architecture Impact its Robustness to Noisy Labels?

        **Authors**: *Jingling Li, Mozhi Zhang, Keyulu Xu, John Dickerson, Jimmy Ba*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/51311013e51adebc3c34d2cc591fefee-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/51311013e51adebc3c34d2cc591fefee-Abstract.html)

        **Abstract**:

        Noisy labels are inevitable in large real-world datasets. In this work, we explore an area understudied by previous works --- how the network's architecture impacts its robustness to noisy labels. We provide a formal framework connecting the robustness of a network to the alignments between its architecture and target/noise functions. Our framework measures a network's robustness via the predictive power in its representations --- the test performance of a linear model trained on the learned representations using a small set of clean labels. We hypothesize that a network is more robust to noisy labels if its architecture is more aligned with the target function than the noise. To support our hypothesis, we provide both theoretical and empirical evidence across various neural network architectures and different domains. We also find that when the network is well-aligned with the target function, its predictive power in representations could improve upon state-of-the-art (SOTA) noisy-label-training methods in terms of test accuracy and even outperform sophisticated methods that use clean labels.

        ----

        ## [749] Calibration and Consistency of Adversarial Surrogate Losses

        **Authors**: *Pranjal Awasthi, Natalie Frank, Anqi Mao, Mehryar Mohri, Yutao Zhong*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/514a70448c235ccb8b6842ef5e02ad3b-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/514a70448c235ccb8b6842ef5e02ad3b-Abstract.html)

        **Abstract**:

        Adversarial robustness is an increasingly critical property of classifiers in applications. The design of robust algorithms relies on surrogate losses since the optimization of the adversarial loss with most hypothesis sets is NP-hard. But, which surrogate losses should be used and when do they benefit from theoretical guarantees? We present an extensive study of this question, including a detailed analysis of the $\mathcal{H}$-calibration and $\mathcal{H}$-consistency of adversarial surrogate losses. We show that convex loss functions, or the supremum-based convex losses often used in applications, are not $\mathcal{H}$-calibrated for common hypothesis sets used in machine learning. We then give a characterization of $\mathcal{H}$-calibration and prove that some surrogate losses are indeed $\mathcal{H}$-calibrated for the adversarial zero-one loss, with common hypothesis sets. In particular, we fix some calibration results presented in prior work for a family of linear models and significantly generalize the results to the nonlinear hypothesis sets. Next, we show that $\mathcal{H}$-calibration is not sufficient to guarantee consistency and prove that, in the absence of any distributional assumption, no continuous surrogate loss is consistent in the adversarial setting. This, in particular, proves that a claim made in prior work is inaccurate. Next, we identify natural conditions under which some surrogate losses that we describe in detail are $\mathcal{H}$-consistent. We also report a series of empirical results which show that many $\mathcal{H}$-calibrated surrogate losses are indeed not $\mathcal{H}$-consistent, and validate our theoretical assumptions. Our adversarial $\mathcal{H}$-consistency results are novel, even for the case where $\mathcal{H}$ is the family of all measurable functions.

        ----

        ## [750] The Value of Information When Deciding What to Learn

        **Authors**: *Dilip Arumugam, Benjamin Van Roy*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/517da335fd0ec2f4a25ea139d5494163-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/517da335fd0ec2f4a25ea139d5494163-Abstract.html)

        **Abstract**:

        All sequential decision-making agents explore so as to acquire knowledge about a particular target. It is often the responsibility of the agent designer to construct this target which, in rich and complex environments, constitutes a onerous burden; without full knowledge of the environment itself, a designer may forge a sub-optimal learning target that poorly balances the amount of information an agent must acquire to identify the target against the target's associated performance shortfall. While recent work has developed a connection between learning targets and rate-distortion theory to address this challenge and empower agents that decide what to learn in an automated fashion, the proposed algorithm does not optimally tackle the equally important challenge of efficient information acquisition. In this work, building upon the seminal design principle of information-directed sampling (Russo & Van Roy, 2014), we address this shortcoming directly to couple optimal information acquisition with the optimal design of learning targets. Along the way, we offer new insights into learning targets from the literature on rate-distortion theory before turning to empirical results that confirm the value of information when deciding what to learn.

        ----

        ## [751] Co-Adaptation of Algorithmic and Implementational Innovations in Inference-based Deep Reinforcement Learning

        **Authors**: *Hiroki Furuta, Tadashi Kozuno, Tatsuya Matsushima, Yutaka Matsuo, Shixiang Shane Gu*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/517f24c02e620d5a4dac1db388664a63-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/517f24c02e620d5a4dac1db388664a63-Abstract.html)

        **Abstract**:

        Recently many algorithms were devised for reinforcement learning (RL) with function approximation. While they have clear algorithmic distinctions, they also have many implementation differences that are algorithm-independent and sometimes under-emphasized. Such mixing of algorithmic novelty and implementation craftsmanship makes rigorous analyses of the sources of performance improvements across algorithms difficult. In this work, we focus on a series of off-policy inference-based actor-critic algorithms -- MPO, AWR, and SAC -- to decouple their algorithmic innovations and implementation decisions. We present unified derivations through a single control-as-inference objective, where we can categorize each algorithm as based on either Expectation-Maximization (EM) or direct Kullback-Leibler (KL) divergence minimization and treat the rest of specifications as implementation details. We performed extensive ablation studies, and identified substantial performance drops whenever implementation details are mismatched for algorithmic choices. These results show which implementation or code details are co-adapted and co-evolved with algorithms, and which are transferable across algorithms: as examples, we identified that tanh Gaussian policy and network sizes are highly adapted to algorithmic types, while layer normalization and ELU are critical for MPO's performances but also transfer to noticeable gains in SAC. We hope our work can inspire future work to further demystify sources of performance improvements across multiple algorithms and allow researchers to build on one another's both algorithmic and implementational innovations.

        ----

        ## [752] Can fMRI reveal the representation of syntactic structure in the brain?

        **Authors**: *Aniketh Janardhan Reddy, Leila Wehbe*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/51a472c08e21aef54ed749806e3e6490-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/51a472c08e21aef54ed749806e3e6490-Abstract.html)

        **Abstract**:

        While studying semantics in the brain, neuroscientists use two approaches. One is to identify areas that are correlated with semantic processing load. Another is to find areas that are predicted by the semantic representation of the stimulus words. However, most studies of syntax have focused only on identifying areas correlated with syntactic processing load.  One possible reason for this discrepancy is that representing syntactic structure in an embedding space such that it can be used to model brain activity is a non-trivial computational problem. Another possible reason is that it is unclear if the low signal-to-noise ratio of neuroimaging tools such as functional Magnetic Resonance Imaging (fMRI) can allow us to reveal the correlates of complex (and perhaps subtle) syntactic representations. In this study, we propose novel multi-dimensional features that encode information about the syntactic structure of sentences. Using these features and fMRI recordings of participants reading a natural text, we model the brain representation of syntax. First, we find that our syntactic structure-based features explain additional variance in the brain activity of various parts of the language system, even after controlling for complexity metrics that capture processing load. At the same time, we see that regions well-predicted by syntactic features are distributed in the language system and are not distinguishable from those processing semantics. Our code and data will be available at https://github.com/anikethjr/brainsyntacticrepresentations.

        ----

        ## [753] Robust Implicit Networks via Non-Euclidean Contractions

        **Authors**: *Saber Jafarpour, Alexander Davydov, Anton V. Proskurnikov, Francesco Bullo*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/51a6ce0252d8fa6e913524bdce8db490-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/51a6ce0252d8fa6e913524bdce8db490-Abstract.html)

        **Abstract**:

        Implicit neural networks, a.k.a., deep equilibrium networks, are a class of implicit-depth learning models where function evaluation is performed by solving a fixed point equation. They generalize classic feedforward models and are equivalent to infinite-depth weight-tied feedforward networks. While implicit models show improved accuracy and significant reduction in memory consumption, they can suffer from ill-posedness and convergence instability.This paper provides a new framework, which we call Non-Euclidean Monotone Operator Network (NEMON), to design well-posed and robust implicit neural networks based upon contraction theory for the non-Euclidean norm $\ell_\infty$. Our framework includes (i) a novel condition for well-posedness based on one-sided Lipschitz constants, (ii) an average iteration for computing fixed-points, and (iii) explicit estimates on input-output Lipschitz constants. Additionally, we design a training problem with the well-posedness condition and the average iteration as constraints and, to achieve robust models, with the input-output Lipschitz constant as a regularizer. Our $\ell_\infty$ well-posedness condition leads to a larger polytopic training search space than existing conditions and our average iteration enjoys accelerated convergence. Finally, we evaluate our framework in image classification through the MNIST and the CIFAR-10 datasets. Our numerical results demonstrate improved accuracy and robustness of the implicit models with smaller input-output Lipschitz bounds. Code is available at https://github.com/davydovalexander/Non-Euclidean_Mon_Op_Net.

        ----

        ## [754] A Kernel-based Test of Independence for Cluster-correlated Data

        **Authors**: *Hongjiao Liu, Anna M. Plantinga, Yunhua Xiang, Michael C. Wu*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/51be2fed6c55f5aa0c16ff14c140b187-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/51be2fed6c55f5aa0c16ff14c140b187-Abstract.html)

        **Abstract**:

        The Hilbert-Schmidt Independence Criterion (HSIC) is a powerful kernel-based statistic for assessing the generalized dependence between two multivariate variables. However, independence testing based on the HSIC is not directly possible for cluster-correlated data. Such a correlation pattern among the observations arises in many practical situations, e.g., family-based and longitudinal data, and requires proper accommodation. Therefore, we propose a novel HSIC-based independence test to evaluate the dependence between two multivariate variables based on cluster-correlated data. Using the previously proposed empirical HSIC as our test statistic, we derive its asymptotic distribution under the null hypothesis of independence between the two variables but in the presence of sample correlation. Based on both simulation studies and real data analysis, we show that, with clustered data, our approach effectively controls type I error and has a higher statistical power than competing methods.

        ----

        ## [755] Efficient methods for Gaussian Markov random fields under sparse linear constraints

        **Authors**: *David Bolin, Jonas Wallin*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/51e6d6e679953c6311757004d8cbbba9-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/51e6d6e679953c6311757004d8cbbba9-Abstract.html)

        **Abstract**:

        Methods for inference and simulation of linearly constrained Gaussian Markov Random Fields (GMRF) are computationally prohibitive when the number of constraints is large. In some cases, such as for intrinsic GMRFs, they may even be unfeasible. We propose a new class of methods to overcome these challenges in the common case of sparse constraints, where one has a large number of constraints and each only involves a few elements. Our methods rely on a basis transformation into blocks of constrained versus non-constrained subspaces, and we show that the methods greatly outperform existing alternatives in terms of computational cost. By combining the proposed methods with the stochastic partial differential equation approach for Gaussian random fields, we also show how to formulate Gaussian process regression with linear constraints in a GMRF setting to reduce computational cost. This is illustrated in two applications with simulated data.

        ----

        ## [756] Sparse is Enough in Scaling Transformers

        **Authors**: *Sebastian Jaszczur, Aakanksha Chowdhery, Afroz Mohiuddin, Lukasz Kaiser, Wojciech Gajewski, Henryk Michalewski, Jonni Kanerva*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/51f15efdd170e6043fa02a74882f0470-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/51f15efdd170e6043fa02a74882f0470-Abstract.html)

        **Abstract**:

        Large Transformer models yield impressive results on many tasks, but are expensive to train, or even fine-tune, and so slow at decoding that their use and study becomes out of reach. We address this problem by leveraging sparsity. We study sparse variants for all layers in the Transformer and propose Scaling Transformers, a family of next generation Transformer models that use sparse layers to scale efficiently and perform unbatched decoding much faster than the standard Transformer as we scale up the model size. Surprisingly, the sparse layers are enough to obtain the same perplexity as the standard Transformer with the same number of parameters. We also integrate with prior sparsity approaches to attention and enable fast inference on long sequences even with limited memory. This results in performance competitive to the state-of-the-art on long text summarization.

        ----

        ## [757] Sparse Training via Boosting Pruning Plasticity with Neuroregeneration

        **Authors**: *Shiwei Liu, Tianlong Chen, Xiaohan Chen, Zahra Atashgahi, Lu Yin, Huanyu Kou, Li Shen, Mykola Pechenizkiy, Zhangyang Wang, Decebal Constantin Mocanu*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/5227b6aaf294f5f027273aebf16015f2-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/5227b6aaf294f5f027273aebf16015f2-Abstract.html)

        **Abstract**:

        Works on lottery ticket hypothesis (LTH) and single-shot network pruning (SNIP) have raised a lot of attention currently on post-training pruning (iterative magnitude pruning), and before-training pruning (pruning at initialization). The former method suffers from an extremely large computation cost and the latter usually struggles with insufficient performance. In comparison, during-training pruning, a class of pruning methods that simultaneously enjoys the training/inference efficiency and the comparable performance, temporarily, has been less explored. To better understand during-training pruning, we quantitatively study the effect of pruning throughout training from the perspective of pruning plasticity (the ability of the pruned networks to recover the original performance). Pruning plasticity can help explain several other empirical observations about neural network pruning in literature. We further find that pruning plasticity can be substantially improved by injecting a brain-inspired mechanism called neuroregeneration, i.e., to regenerate the same number of connections as pruned. We design a novel gradual magnitude pruning (GMP) method, named gradual pruning with zero-cost neuroregeneration (GraNet), that advances state of the art. Perhaps most impressively, its sparse-to-sparse version for the first time boosts the sparse-to-sparse training performance over various dense-to-sparse methods with ResNet-50 on ImageNet without extending the training time. We release all codes in https://github.com/Shiweiliuiiiiiii/GraNet.

        ----

        ## [758] Low-Fidelity Video Encoder Optimization for Temporal Action Localization

        **Authors**: *Mengmeng Xu, Juan-Manuel Pérez-Rúa, Xiatian Zhu, Bernard Ghanem, Brais Martínez*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/522a9ae9a99880d39e5daec35375e999-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/522a9ae9a99880d39e5daec35375e999-Abstract.html)

        **Abstract**:

        Most existing temporal action localization (TAL) methods rely on a transfer learning pipeline: by first optimizing a video encoder on a large action classification dataset (i.e., source domain), followed by freezing the encoder and training a TAL head on the action localization dataset (i.e., target domain). This results in a task discrepancy problem for the video encoder â€“ trained for action classification, but used for TAL. Intuitively, joint optimization with both the video encoder and TAL head is a strong baseline solution to this discrepancy. However, this is not operable for TAL subject to the GPU memory constraints, due to the prohibitive computational cost in processing long untrimmed videos. In this paper, we resolve this challenge by introducing a novel low-fidelity (LoFi) video encoder optimization method. Instead of always using the full training configurations in TAL learning, we propose to reduce the mini-batch composition in terms of temporal, spatial, or spatio-temporal resolution so that jointly optimizing the video encoder and TAL head becomes operable under the same memory conditions of a mid-range hardware budget. Crucially, this enables the gradients to flow backwards through the video encoder conditioned on a TAL supervision loss, favourably solving the task discrepancy problem and providing more effective feature representations. Extensive experiments show that the proposed LoFi optimization approach can significantly enhance the performance of existing TAL methods. Encouragingly, even with a lightweight ResNet18 based video encoder in a single RGB stream, our method surpasses two-stream (RGB + optical-flow) ResNet50 based alternatives, often by a good margin. Our code is publicly available at https://github.com/saic-fi/lofiactionlocalization.

        ----

        ## [759] On Provable Benefits of Depth in Training Graph Convolutional Networks

        **Authors**: *Weilin Cong, Morteza Ramezani, Mehrdad Mahdavi*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/524265e8b942930fbbe8a5d979d29205-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/524265e8b942930fbbe8a5d979d29205-Abstract.html)

        **Abstract**:

        Graph Convolutional Networks (GCNs) are known to suffer from performance degradation as the number of layers increases, which is usually attributed to over-smoothing. Despite the apparent consensus, we observe that there exists a discrepancy between the theoretical understanding of over-smoothing and the practical capabilities of GCNs. Specifically, we argue that over-smoothing does not necessarily happen in practice, a deeper model is provably expressive, can converge to global optimum with linear convergence rate, and achieve very high training accuracy as long as properly trained. Despite being capable of achieving high training accuracy, empirical results show that the deeper models generalize poorly on the testing stage and existing theoretical understanding of such behavior remains elusive. To achieve better understanding, we carefully analyze the generalization capability of GCNs, and show that the training strategies to achieve high training accuracy significantly deteriorate the generalization capability of GCNs. Motivated by these findings, we propose a decoupled structure for GCNs that detaches weight matrices from feature propagation to preserve the expressive power and ensure good generalization performance. We conduct empirical evaluations on various synthetic and real-world datasets to validate the correctness of our theory.

        ----

        ## [760] Practical Near Neighbor Search via Group Testing

        **Authors**: *Joshua Engels, Benjamin Coleman, Anshumali Shrivastava*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/5248e5118c84beea359b6ea385393661-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/5248e5118c84beea359b6ea385393661-Abstract.html)

        **Abstract**:

        We present a new algorithm for the approximate near neighbor problem that combines classical ideas from group testing with locality-sensitive hashing (LSH). We reduce the near neighbor search problem to a group testing problem by designating neighbors as "positives," non-neighbors as "negatives," and approximate membership queries as group tests. We instantiate this framework using distance-sensitive Bloom Filters to Identify Near-Neighbor Groups (FLINNG). We prove that FLINNG has sub-linear query time and show that our algorithm comes with a variety of practical advantages. For example, FLINNG can be constructed in a single pass through the data, consists entirely of efficient integer operations, and does not require any distance computations. We conduct large-scale experiments on high-dimensional search tasks such as genome search, URL similarity search, and embedding search over the massive YFCC100M dataset. In our comparison with leading algorithms such as HNSW and FAISS, we find that FLINNG can provide up to a 10x query speedup with substantially smaller indexing time and memory.

        ----

        ## [761] Baby Intuitions Benchmark (BIB): Discerning the goals, preferences, and actions of others

        **Authors**: *Kanishk Gandhi, Gala Stojnic, Brenden M. Lake, Moira R. Dillon*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/525b8410cc8612283c9ecaf9a319f8ed-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/525b8410cc8612283c9ecaf9a319f8ed-Abstract.html)

        **Abstract**:

        To achieve human-like common sense about everyday life, machine learning systems must understand and reason about the goals, preferences, and actions of other agents in the environment. By the end of their first year of life, human infants intuitively achieve such common sense, and these cognitive achievements lay the foundation for humans' rich and complex understanding of the mental states of others. Can machines achieve generalizable, commonsense reasoning about other agents like human infants? The Baby Intuitions Benchmark (BIB) challenges machines to predict the plausibility of an agent's behavior based on the underlying causes of its actions. Because BIB's content and paradigm are adopted from developmental cognitive science, BIB allows for direct comparison between human and machine performance. Nevertheless, recently proposed, deep-learning-based agency reasoning models fail to show infant-like reasoning, leaving BIB an open challenge.

        ----

        ## [762] Neural Hybrid Automata: Learning Dynamics With Multiple Modes and Stochastic Transitions

        **Authors**: *Michael Poli, Stefano Massaroli, Luca Scimeca, Sanghyuk Chun, Seong Joon Oh, Atsushi Yamashita, Hajime Asama, Jinkyoo Park, Animesh Garg*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/5291822d0636dc429e80e953c58b6a76-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/5291822d0636dc429e80e953c58b6a76-Abstract.html)

        **Abstract**:

        Effective control and prediction of dynamical systems require appropriate handling of continuous-time and discrete, event-triggered processes. Stochastic hybrid systems (SHSs), common across engineering domains, provide a formalism for dynamical systems subject to discrete, possibly stochastic, state jumps and multi-modal continuous-time flows. Despite the versatility and importance of SHSs across applications, a general procedure for the explicit learning of both discrete events and multi-mode continuous dynamics remains an open problem. This work introduces Neural Hybrid Automata (NHAs), a recipe for learning SHS dynamics without a priori knowledge on the number, mode parameters, and inter-modal transition dynamics. NHAs provide a systematic inference method based on normalizing flows, neural differential equations, and self-supervision. We showcase NHAs on several tasks, including mode recovery and flow learning in systems with stochastic transitions, and end-to-end learning of hierarchical robot controllers.

        ----

        ## [763] Fast Projection onto the Capped Simplex with Applications to Sparse Regression in Bioinformatics

        **Authors**: *Andersen Man Shun Ang, Jianzhu Ma, Nianjun Liu, Kun Huang, Yijie Wang*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/52aaa62e71f829d41d74892a18a11d59-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/52aaa62e71f829d41d74892a18a11d59-Abstract.html)

        **Abstract**:

        We consider the problem of projecting a vector onto the so-called k-capped simplex, which is a hyper-cube cut by a hyperplane.For an n-dimensional input vector with bounded elements, we found that a simple algorithm based on Newton's method is able to solve the projection problem to high precision with a complexity roughly about O(n), which has a much lower computational cost compared with the existing sorting-based methods proposed in the literature.We provide a theory for partial explanation and justification of the method.We demonstrate that the proposed algorithm can produce a solution of the projection problem with high precision on large scale datasets, and the algorithm is able to significantly outperform the state-of-the-art methods in terms of runtime (about 6-8 times faster than a commercial software with respect to CPU time for input vector with 1 million variables or more).We further illustrate the effectiveness of the proposed algorithm on solving sparse regression in a bioinformatics problem.Empirical results on the GWAS dataset (with 1,500,000 single-nucleotide polymorphisms) show that, when using the proposed method to accelerate the Projected Quasi-Newton (PQN) method, the accelerated PQN algorithm is able to handle huge-scale regression problem and it is more efficient (about 3-6 times faster) than the current state-of-the-art methods.

        ----

        ## [764] The Many Faces of Adversarial Risk

        **Authors**: *Muni Sreenivas Pydi, Varun S. Jog*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/52c4608c2f126708211b9e0a60eaf050-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/52c4608c2f126708211b9e0a60eaf050-Abstract.html)

        **Abstract**:

        Adversarial risk quantifies the performance of classifiers on adversarially perturbed data. Numerous definitions of adversarial risk---not all mathematically rigorous and differing subtly in the details---have appeared in the literature. In this paper, we revisit these definitions, make them rigorous, and critically examine their similarities and differences. Our technical tools derive from optimal transport, robust statistics, functional analysis, and game theory. Our contributions include the following: generalizing Strassenâ€™s theorem to the unbalanced optimal transport setting with applications to adversarial classification with unequal priors; showing an equivalence between adversarial robustness and robust hypothesis testing with $\infty$-Wasserstein uncertainty sets; proving the existence of a pure Nash equilibrium in the two-player game between the adversary and the algorithm; and characterizing adversarial risk by the minimum Bayes error between distributions belonging to the $\infty$-Wasserstein uncertainty sets. Our results generalize and deepen recently discovered connections between optimal transport and adversarial robustness and reveal new connections to Choquet capacities and game theory.

        ----

        ## [765] Meta-Adaptive Nonlinear Control: Theory and Algorithms

        **Authors**: *Guanya Shi, Kamyar Azizzadenesheli, Michael O'Connell, Soon-Jo Chung, Yisong Yue*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/52fc2aee802efbad698503d28ebd3a1f-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/52fc2aee802efbad698503d28ebd3a1f-Abstract.html)

        **Abstract**:

        We present an online multi-task learning approach for adaptive nonlinear control, which we call Online Meta-Adaptive Control (OMAC). The goal is to control a nonlinear system subject to adversarial disturbance and unknown \emph{environment-dependent} nonlinear dynamics, under the assumption that the environment-dependent dynamics can be well captured with some shared representation. Our approach is motivated by robot control, where a robotic system encounters a sequence of new environmental conditions that it must quickly adapt to. A key emphasis is to integrate online representation learning with established methods from control theory, in order to arrive at a unified framework that yields both control-theoretic and learning-theoretic guarantees. We provide instantiations of our approach under varying conditions, leading to the first non-asymptotic end-to-end convergence guarantee for multi-task nonlinear control. OMAC can also be integrated with deep representation learning. Experiments show that OMAC significantly outperforms conventional adaptive control approaches which do not learn the shared representation, in inverted pendulum and 6-DoF drone control tasks under varying wind conditions.

        ----

        ## [766] Compositional Reinforcement Learning from Logical Specifications

        **Authors**: *Kishor Jothimurugan, Suguman Bansal, Osbert Bastani, Rajeev Alur*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/531db99cb00833bcd414459069dc7387-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/531db99cb00833bcd414459069dc7387-Abstract.html)

        **Abstract**:

        We study the problem of learning control policies for complex tasks given by logical specifications. Recent approaches automatically generate a reward function from a given specification and use a suitable reinforcement learning algorithm to learn a policy that maximizes the expected reward. These approaches, however, scale poorly to complex tasks that require high-level planning. In this work, we develop a compositional learning approach, called DIRL, that interleaves high-level planning and reinforcement learning. First, DIRL encodes the specification as an abstract graph; intuitively, vertices and edges of the graph correspond to regions of the state space and simpler sub-tasks, respectively. Our approach then incorporates reinforcement learning to learn neural network policies for each edge (sub-task) within a Dijkstra-style planning algorithm to compute a high-level plan in the graph. An evaluation of the proposed approach on a set of challenging control benchmarks with continuous state and action spaces demonstrates that it outperforms state-of-the-art baselines.

        ----

        ## [767] Differentiable Quality Diversity

        **Authors**: *Matthew C. Fontaine, Stefanos Nikolaidis*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/532923f11ac97d3e7cb0130315b067dc-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/532923f11ac97d3e7cb0130315b067dc-Abstract.html)

        **Abstract**:

        Quality diversity (QD) is a growing branch of stochastic optimization research that studies the problem of generating an archive of solutions that maximize a given objective function but are also diverse with respect to a set of specified measure functions. However, even when these functions are differentiable, QD algorithms treat them as "black boxes", ignoring gradient information. We present the differentiable quality diversity (DQD) problem, a special case of QD, where both the objective and measure functions are first order differentiable. We then present MAP-Elites via a Gradient Arborescence (MEGA), a DQD algorithm that leverages gradient information to efficiently explore the joint range of the objective and measure functions. Results in two QD benchmark domains and in searching the latent space of a StyleGAN show that MEGA significantly outperforms state-of-the-art QD algorithms, highlighting DQD's promise for efficient quality diversity optimization when gradient information is available. Source code is available at https://github.com/icaros-usc/dqd.

        ----

        ## [768] Credit Assignment Through Broadcasting a Global Error Vector

        **Authors**: *David G. Clark, L. F. Abbott, SueYeon Chung*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/532b81fa223a1b1ec74139a5b8151d12-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/532b81fa223a1b1ec74139a5b8151d12-Abstract.html)

        **Abstract**:

        Backpropagation (BP) uses detailed, unit-specific feedback to train deep neural networks (DNNs) with remarkable success. That biological neural circuits appear to perform credit assignment, but cannot implement BP, implies the existence of other powerful learning algorithms. Here, we explore the extent to which a globally broadcast learning signal, coupled with local weight updates, enables training of DNNs. We present both a learning rule, called global error-vector broadcasting (GEVB), and a class of DNNs, called vectorized nonnegative networks (VNNs), in which this learning rule operates. VNNs have vector-valued units and nonnegative weights past the first layer. The GEVB learning rule generalizes three-factor Hebbian learning, updating each weight by an amount proportional to the inner product of the presynaptic activation and a globally broadcast error vector when the postsynaptic unit is active. We prove that these weight updates are matched in sign to the gradient, enabling accurate credit assignment. Moreover, at initialization, these updates are exactly proportional to the gradient in the limit of infinite network width. GEVB matches the performance of BP in VNNs, and in some cases outperforms direct feedback alignment (DFA) applied in conventional networks. Unlike DFA, GEVB successfully trains convolutional layers. Altogether, our theoretical and empirical results point to a surprisingly powerful role for a global learning signal in training DNNs.

        ----

        ## [769] An Online Method for A Class of Distributionally Robust Optimization with Non-convex Objectives

        **Authors**: *Qi Qi, Zhishuai Guo, Yi Xu, Rong Jin, Tianbao Yang*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/533fa796b43291fc61a9e812a50c3fb6-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/533fa796b43291fc61a9e812a50c3fb6-Abstract.html)

        **Abstract**:

        In this paper, we propose a practical online method for solving a class of distributional robust optimization (DRO) with non-convex objectives, which has important applications in machine learning for improving the robustness of neural networks. In the literature, most methods for solving DRO are based on stochastic primal-dual methods. However, primal-dual methods for DRO suffer from several drawbacks: (1) manipulating a high-dimensional dual variable corresponding to the size of data is time expensive; (2) they are not friendly to online learning where data is coming sequentially. To address these issues, we consider a class of DRO with an KL divergence regularization on the dual variables, transform the min-max problem into a compositional minimization problem, and propose practical duality-free online stochastic methods without requiring a large mini-batch size. We establish the state-of-the-art complexities of the proposed methods with and without a Polyak-Łojasiewicz (PL) condition of the objective. Empirical studies on large-scale deep learning tasks (i) demonstrate that our method can speed up the training by more than 2 times than baseline methods and save days of training time on a large-scale dataset with ∼ 265K images, and (ii) verify the supreme performance of DRO over Empirical Risk Minimization (ERM) on imbalanced datasets. Of independent interest, the proposed method can be also used for solving a family of stochastic compositional problems with state-of-the-art complexities.

        ----

        ## [770] A single gradient step finds adversarial examples on random two-layers neural networks

        **Authors**: *Sébastien Bubeck, Yeshwanth Cherapanamjeri, Gauthier Gidel, Remi Tachet des Combes*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/536eecee295b92db6b32194e269541f8-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/536eecee295b92db6b32194e269541f8-Abstract.html)

        **Abstract**:

        Daniely and Schacham recently showed that gradient descent finds adversarial examples on random undercomplete two-layers ReLU neural networks. The term “undercomplete” refers to the fact that their proof only holds when the number of neurons is a vanishing fraction of the ambient dimension. We extend their result to the overcomplete case, where the number of neurons is larger than the dimension (yet also subexponential in the dimension). In fact we prove that a single step of gradient descent suffices. We also show this result for any subexponential width random neural network with smooth activation function.

        ----

        ## [771] Parameterized Knowledge Transfer for Personalized Federated Learning

        **Authors**: *Jie Zhang, Song Guo, Xiaosong Ma, Haozhao Wang, Wenchao Xu, Feijie Wu*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/5383c7318a3158b9bc261d0b6996f7c2-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/5383c7318a3158b9bc261d0b6996f7c2-Abstract.html)

        **Abstract**:

        In recent years, personalized federated learning (pFL) has attracted increasing attention for its potential in dealing with statistical heterogeneity among clients. However, the state-of-the-art pFL methods rely on model parameters aggregation at the server side, which require all models to have the same structure and size, and thus limits the application for more heterogeneous scenarios. To deal with such model constraints, we exploit the potentials of heterogeneous model settings and propose a novel training framework to employ personalized models for different clients. Specifically, we formulate the aggregation procedure in original pFL into a personalized group knowledge transfer training algorithm, namely, KT-pFL, which enables each client to maintain a personalized soft prediction at the server side to guide the others' local training.  KT-pFL updates the personalized soft prediction of each client by a linear combination of all local soft predictions using a knowledge coefficient matrix, which can adaptively reinforce the collaboration among clients who own similar data distribution. Furthermore, to quantify the contributions of each client to others' personalized training, the knowledge coefficient matrix is parameterized so that it can be trained simultaneously with the models.  The knowledge coefficient matrix and the model parameters are alternatively updated in each round following the gradient descent way. Extensive experiments on various datasets (EMNIST, Fashion_MNIST, CIFAR-10) are conducted under different settings (heterogeneous models and data distributions). It is demonstrated that the proposed framework is the first federated learning paradigm that realizes personalized model training via parameterized group knowledge transfer while achieving significant performance gain comparing with state-of-the-art algorithms.

        ----

        ## [772] Contrastively Disentangled Sequential Variational Autoencoder

        **Authors**: *Junwen Bai, Weiran Wang, Carla P. Gomes*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/53c5b2affa12eed84dfec9bfd83550b1-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/53c5b2affa12eed84dfec9bfd83550b1-Abstract.html)

        **Abstract**:

        Self-supervised disentangled representation learning is a critical task in sequence modeling. The learnt representations contribute to better model interpretability as well as the data generation, and improve the sample efficiency for downstream tasks. We propose a novel sequence representation learning method, named Contrastively Disentangled Sequential Variational Autoencoder (C-DSVAE), to extract and separate the static (time-invariant) and dynamic (time-variant) factors in the latent space. Different from previous sequential variational autoencoder methods, we use a novel evidence lower bound which maximizes the mutual information between the input and the latent factors, while penalizes the mutual information between the static and dynamic factors. We leverage contrastive estimations of the mutual information terms in training, together with simple yet effective augmentation techniques, to introduce additional inductive biases. Our experiments show that C-DSVAE significantly outperforms the previous state-of-the-art methods on multiple metrics.

        ----

        ## [773] Recursive Causal Structure Learning in the Presence of Latent Variables and Selection Bias

        **Authors**: *Sina Akbari, Ehsan Mokhtarian, AmirEmad Ghassami, Negar Kiyavash*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/53edebc543333dfbf7c5933af792c9c4-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/53edebc543333dfbf7c5933af792c9c4-Abstract.html)

        **Abstract**:

        We consider the problem of learning the causal MAG of a system from observational data in the presence of latent variables and selection bias. Constraint-based methods are one of the main approaches for solving this problem, but the existing methods are either computationally impractical when dealing with large graphs or lacking completeness guarantees. We propose a novel computationally efficient recursive constraint-based method that is sound and complete. The key idea of our approach is that at each iteration a specific type of variable is identified and removed. This allows us to learn the structure efficiently and recursively, as this technique reduces both the number of required conditional independence (CI) tests and the size of the conditioning sets. The former substantially reduces the computational complexity, while the latter results in more reliable CI tests. We provide an upper bound on the number of required CI tests in the worst case. To the best of our knowledge, this is the tightest bound in the literature. We further provide a lower bound on the number of CI tests required by any constraint-based method.  The upper bound of our proposed approach and the lower bound at most differ by a factor equal to the number of variables in the worst case. We provide experimental results to compare the proposed approach with the state of the art on both synthetic and real-world structures.

        ----

        ## [774] Generalization Error Rates in Kernel Regression: The Crossover from the Noiseless to Noisy Regime

        **Authors**: *Hugo Cui, Bruno Loureiro, Florent Krzakala, Lenka Zdeborová*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/543bec10c8325987595fcdc492a525f4-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/543bec10c8325987595fcdc492a525f4-Abstract.html)

        **Abstract**:

        In this manuscript we consider Kernel Ridge Regression (KRR) under the Gaussian design. Exponents for the decay of the excess generalization error of KRR have been reported in various works under the assumption of power-law decay of eigenvalues of the features co-variance. These decays were, however, provided for sizeably different setups, namely in the noiseless case with constant regularization and in the noisy optimally regularized case. Intermediary settings have been left substantially uncharted. In this work, we unify and extend this line of work, providing characterization of all regimes and excess error decay rates that can be observed in terms of the interplay of noise and regularization. In particular, we show the existence of a transition in the noisy setting between the noiseless exponents to its noisy values as the sample complexity is increased. Finally, we illustrate how this crossover can also be observed on real data sets.

        ----

        ## [775] Learning Gaussian Mixtures with Generalized Linear Models: Precise Asymptotics in High-dimensions

        **Authors**: *Bruno Loureiro, Gabriele Sicuro, Cédric Gerbelot, Alessandro Pacco, Florent Krzakala, Lenka Zdeborová*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/543e83748234f7cbab21aa0ade66565f-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/543e83748234f7cbab21aa0ade66565f-Abstract.html)

        **Abstract**:

        Generalised linear models for multi-class classification problems are one of the fundamental building blocks of modern machine learning tasks. In this manuscript, we characterise the learning of a mixture of $K$ Gaussians with generic means and covariances via empirical risk minimisation (ERM) with any convex loss and regularisation. In particular, we prove exact asymptotics characterising the ERM estimator in high-dimensions, extending several previous results about Gaussian mixture classification in the literature. We exemplify our result in two tasks of interest in statistical learning: a) classification for a mixture with sparse means, where we study the efficiency of $\ell_1$ penalty with respect to $\ell_2$; b) max-margin multi-class classification, where we characterise the phase transition on the existence of the multi-class logistic maximum likelihood estimator for $K>2$. Finally, we discuss how our theory can be applied beyond the scope of synthetic data, showing that in different cases Gaussian mixtures capture closely the learning curve of classification tasks in real data sets.

        ----

        ## [776] Spectral embedding for dynamic networks with stability guarantees

        **Authors**: *Ian Gallagher, Andrew Jones, Patrick Rubin-Delanchy*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/5446f217e9504bc593ad9dcf2ec88dda-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/5446f217e9504bc593ad9dcf2ec88dda-Abstract.html)

        **Abstract**:

        We consider the problem of embedding a dynamic network, to obtain time-evolving vector representations of each node, which can then be used to describe changes in behaviour of individual nodes, communities, or the entire graph. Given this open-ended remit, we argue that two types of stability in the spatio-temporal positioning of nodes are desirable: to assign the same position, up to noise, to nodes behaving similarly at a given time (cross-sectional stability) and a constant position, up to noise, to a single node behaving similarly across different times (longitudinal stability). Similarity in behaviour is defined formally using notions of exchangeability under a dynamic latent position network model. By showing how this model can be recast as a multilayer random dot product graph, we demonstrate that unfolded adjacency spectral embedding satisfies both stability conditions. We also show how two alternative methods, omnibus and independent spectral embedding, alternately lack one or the other form of stability.

        ----

        ## [777] Infinite Time Horizon Safety of Bayesian Neural Networks

        **Authors**: *Mathias Lechner, Dorde Zikelic, Krishnendu Chatterjee, Thomas A. Henzinger*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/544defa9fddff50c53b71c43e0da72be-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/544defa9fddff50c53b71c43e0da72be-Abstract.html)

        **Abstract**:

        Bayesian neural networks (BNNs) place distributions over the weights of a neural network to model uncertainty in the data and the network's prediction.We consider the problem of verifying safety when running a Bayesian neural network policy in a feedback loop with infinite time horizon systems.Compared to the existing sampling-based approaches, which are inapplicable to the infinite time horizon setting, we train a separate deterministic neural network that serves as an infinite time horizon safety certificate.In particular, we show that the certificate network guarantees the safety of the system over a subset of the BNN weight posterior's support. Our method first computes a safe weight set and then alters the BNN's weight posterior to reject samples outside this set. Moreover, we show how to extend our approach to a safe-exploration reinforcement learning setting, in order to avoid unsafe trajectories during the training of the policy. We evaluate our approach on a series of reinforcement learning benchmarks, including non-Lyapunovian safety specifications.

        ----

        ## [778] Towards understanding retrosynthesis by energy-based models

        **Authors**: *Ruoxi Sun, Hanjun Dai, Li Li, Steven Kearnes, Bo Dai*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/5470abe68052c72afb19be45bb418d02-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/5470abe68052c72afb19be45bb418d02-Abstract.html)

        **Abstract**:

        Retrosynthesis is the process of identifying a set of reactants to synthesize a target molecule. It is of vital importance to material design and drug discovery. Existing machine learning approaches based on language models and graph neural networks have achieved encouraging results. However, the inner connections of these models are rarely discussed, and rigorous evaluations of these models are largely in need. In this paper, we propose a framework that unifies sequence- and graph-based methods as energy-based models (EBMs) with different energy functions. This unified view establishes connections and reveals the differences between models, thereby enhancing our understanding of model design. We also provide a comprehensive assessment of performance to the community. Moreover, we present a novel dual variant within the framework that performs consistent training to induce the agreement between forward- and backward-prediction. This model improves the state-of-the-art of template-free methods with or without reaction types.

        ----

        ## [779] List-Decodable Mean Estimation in Nearly-PCA Time

        **Authors**: *Ilias Diakonikolas, Daniel Kane, Daniel Kongsgaard, Jerry Li, Kevin Tian*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/547b85f3fafdf30856386753dc21c4e1-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/547b85f3fafdf30856386753dc21c4e1-Abstract.html)

        **Abstract**:

        Robust statistics has traditionally focused on designing estimators tolerant to a minority of contaminated data. {\em List-decodable learning}~\cite{CharikarSV17} studies the more challenging regime where only a minority $\tfrac 1 k$ fraction of the dataset, $k \geq 2$, is drawn from the distribution of interest, and no assumptions are made on the remaining data. We study the fundamental task of list-decodable mean estimation in high dimensions. Our main result is a new algorithm for bounded covariance distributions with optimal sample complexity and near-optimal error guarantee, running in {\em nearly-PCA time}. Assuming the ground truth distribution on $\mathbb{R}^d$ has identity-bounded covariance, our algorithm outputs $O(k)$ candidate means, one of which is within distance $O(\sqrt{k\log k})$ from the truth.    Our algorithm runs in time $\widetilde{O}(ndk)$, where $n$ is the dataset size. This runtime nearly matches the cost of performing $k$-PCA on the data, a natural bottleneck of known algorithms for (very) special cases of our problem, such as clustering well-separated mixtures. Prior to our work, the fastest runtimes were $\widetilde{O}(n^2 d k^2)$~\cite{DiakonikolasKK20}, and $\widetilde{O}(nd k^C)$ \cite{CherapanamjeriMY20} for an unspecified constant $C \geq 6$. Our approach builds on a novel soft downweighting method we term SIFT, arguably the simplest known polynomial-time mean estimator in the list-decodable setting. To develop our fast algorithms, we boost the computational cost of SIFT via a careful ``win-win-win'' analysis of an approximate Ky Fan matrix multiplicative weights procedure we develop, which may be of independent interest.

        ----

        ## [780] Distributed Zero-Order Optimization under Adversarial Noise

        **Authors**: *Arya Akhavan, Massimiliano Pontil, Alexandre B. Tsybakov*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/5487e79fa0ccd0b79e5d4a4c8ced005d-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/5487e79fa0ccd0b79e5d4a4c8ced005d-Abstract.html)

        **Abstract**:

        We study the problem of distributed zero-order optimization for a class of strongly convex functions. They are formed by the average of local objectives, associated to different nodes in a prescribed network. We propose a distributed zero-order projected gradient descent algorithm to solve the problem. Exchange of information within the network is permitted only between neighbouring nodes. An important feature of our procedure is that it can query only function values, subject to a general noise model, that does not require zero mean or independent errors.  We derive upper bounds for the average cumulative regret and optimization error of the algorithm  which highlight the role played by a network connectivity parameter, the number of variables, the noise level, the strong convexity parameter, and smoothness properties of the local objectives. The bounds indicate some key improvements of our method over the state-of-the-art, both in the distributed and standard zero-order optimization settings.

        ----

        ## [781] Reliable Estimation of KL Divergence using a Discriminator in Reproducing Kernel Hilbert Space

        **Authors**: *Sandesh Ghimire, Aria Masoomi, Jennifer G. Dy*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/54a367d629152b720749e187b3eaa11b-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/54a367d629152b720749e187b3eaa11b-Abstract.html)

        **Abstract**:

        Estimating Kullbackâ€“Leibler (KL) divergence from samples of two distributions is essential in many machine learning problems. Variational methods using neural network discriminator have been proposed to achieve this task in a scalable manner. However, we noticed that most of these methods using neural network discriminators suffer from high fluctuations (variance) in estimates and instability in training. In this paper, we look at this issue from statistical learning theory and function space complexity perspective to understand why this happens and how to solve it. We argue that the cause of these pathologies is lack of control over the complexity of the neural network discriminator function and could be mitigated by controlling it. To achieve this objective, we 1) present a novel construction of the discriminator in the Reproducing Kernel Hilbert Space (RKHS), 2) theoretically relate the error probability bound of the KL estimates to the complexity of the discriminator in the RKHS space, 3) present a scalable way to control the complexity (RKHS norm) of the discriminator for a reliable estimation of KL divergence, and 4) prove the consistency of the proposed estimator. In three different applications of KL divergence -- estimation of KL, estimation of mutual information and Variational Bayes -- we show that by controlling the complexity as developed in the theory, we are able to reduce the variance of KL estimates and stabilize the training.

        ----

        ## [782] Latent Matters: Learning Deep State-Space Models

        **Authors**: *Alexej Klushyn, Richard Kurle, Maximilian Soelch, Botond Cseke, Patrick van der Smagt*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/54b2b21af94108d83c2a909d5b0a6a50-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/54b2b21af94108d83c2a909d5b0a6a50-Abstract.html)

        **Abstract**:

        Deep state-space models (DSSMs) enable temporal predictions by learning the underlying dynamics of observed sequence data. They are often trained by maximising the evidence lower bound. However, as we show, this does not ensure the model actually learns the underlying dynamics. We therefore propose a constrained optimisation framework as a general approach for training DSSMs. Building upon this, we introduce the extended Kalman VAE (EKVAE), which combines amortised variational inference with classic Bayesian filtering/smoothing to model dynamics more accurately than RNN-based DSSMs. Our results show that the constrained optimisation framework significantly improves system identification and prediction accuracy on the example of established state-of-the-art DSSMs. The EKVAE outperforms previous models w.r.t. prediction accuracy, achieves remarkable results in identifying dynamical systems, and can furthermore successfully learn state-space representations where static and dynamic features are disentangled.

        ----

        ## [783] On the Estimation Bias in Double Q-Learning

        **Authors**: *Zhizhou Ren, Guangxiang Zhu, Hao Hu, Beining Han, Jianglun Chen, Chongjie Zhang*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/54e8912427a8d007ece906c577fdca60-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/54e8912427a8d007ece906c577fdca60-Abstract.html)

        **Abstract**:

        Double Q-learning is a classical method for reducing overestimation bias, which is caused by taking maximum estimated values in the Bellman operation. Its variants in the deep Q-learning paradigm have shown great promise in producing reliable value prediction and improving learning performance. However, as shown by prior work, double Q-learning is not fully unbiased and suffers from underestimation bias. In this paper, we show that such underestimation bias may lead to multiple non-optimal fixed points under an approximate Bellman operator. To address the concerns of converging to non-optimal stationary solutions, we propose a simple but effective approach as a partial fix for the underestimation bias in double Q-learning. This approach leverages an approximate dynamic programming to bound the target value. We extensively evaluate our proposed method in the Atari benchmark tasks and demonstrate its significant improvement over baseline algorithms.

        ----

        ## [784] Mitigating Forgetting in Online Continual Learning with Neuron Calibration

        **Authors**: *Haiyan Yin, Peng Yang, Ping Li*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/54ee290e80589a2a1225c338a71839f5-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/54ee290e80589a2a1225c338a71839f5-Abstract.html)

        **Abstract**:

        Inspired by human intelligence, the research on online continual learning aims to push the limits of the machine learning models to constantly learn from sequentially encountered tasks, with the data from each task being observed in an online fashion. Though recent studies have achieved remarkable progress in improving the online continual learning performance empowered by the deep neural networks-based models, many of today's approaches still suffer a lot from catastrophic forgetting, a persistent challenge for continual learning. In this paper, we present a novel method which attempts to mitigate catastrophic forgetting in online continual learning from a new perspective, i.e., neuron calibration. In particular, we model the neurons in the deep neural networks-based models as calibrated units under a general formulation. Then we formalize a learning framework to effectively train the calibrated model, where neuron calibration could give ubiquitous benefit to balance the stability and plasticity of online continual learning algorithms through influencing both their forward inference path and backward optimization path.  Our proposed formulation for neuron calibration is lightweight and applicable to general feed-forward neural networks-based models. We perform extensive experiments to evaluate our method on four benchmark continual learning datasets. The results show that neuron calibration plays a vital role in improving online continual learning performance and our method could substantially improve the state-of-the-art performance on all~the~evaluated~datasets.

        ----

        ## [785] Escaping Saddle Points with Compressed SGD

        **Authors**: *Dmitrii Avdiukhin, Grigory Yaroslavtsev*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/54eea69746513c0b90bbe6227b6f46c3-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/54eea69746513c0b90bbe6227b6f46c3-Abstract.html)

        **Abstract**:

        Stochastic gradient descent (SGD) is a prevalent optimization technique for large-scale distributed machine learning. While SGD computation can be efficiently divided between multiple machines, communication typically becomes a bottleneck in the distributed setting. Gradient compression methods can be used to alleviate this problem, and a recent line of work shows that SGD augmented with gradient compression converges to an $\varepsilon$-first-order stationary point. In this paper we extend these results to convergence to an $\varepsilon$-second-order stationary point ($\varepsilon$-SOSP), which is to the best of our knowledge the first result of this type. In addition, we show that, when the stochastic gradient is not Lipschitz, compressed SGD with RandomK compressor converges to an $\varepsilon$-SOSP with the same number of iterations as uncompressed SGD [Jin et al.,2021] (JACM), while improving the total communication by a factor of $\tilde \Theta(\sqrt{d} \varepsilon^{-3/4})$, where $d$ is the dimension of the optimization problem. We present additional results for the cases when the compressor is arbitrary and when the stochastic gradient is Lipschitz.

        ----

        ## [786] Non-Gaussian Gaussian Processes for Few-Shot Regression

        **Authors**: *Marcin Sendera, Jacek Tabor, Aleksandra Nowak, Andrzej Bedychaj, Massimiliano Patacchiola, Tomasz Trzcinski, Przemyslaw Spurek, Maciej Zieba*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/54f3bc04830d762a3b56a789b6ff62df-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/54f3bc04830d762a3b56a789b6ff62df-Abstract.html)

        **Abstract**:

        Gaussian Processes (GPs) have been widely used in machine learning to model distributions over functions, with applications including multi-modal regression, time-series prediction, and few-shot learning. GPs are particularly useful in the last application since they rely on Normal distributions and enable closed-form computation of the posterior probability function. Unfortunately, because the resulting posterior is not flexible enough to capture complex distributions, GPs assume high similarity between subsequent tasks - a requirement rarely met in real-world conditions. In this work, we address this limitation by leveraging the flexibility of Normalizing Flows to modulate the posterior predictive distribution of the GP. This makes the GP posterior locally non-Gaussian, therefore we name our method Non-Gaussian Gaussian Processes (NGGPs). More precisely, we propose an invertible ODE-based mapping that operates on each component of the random variable vectors and shares the parameters across all of them. We empirically tested the flexibility of NGGPs on various few-shot learning regression datasets, showing that the mapping can incorporate context embedding information to model different noise levels for periodic functions. As a result, our method shares the structure of the problem between subsequent tasks, but the contextualization allows for adaptation to dissimilarities. NGGPs outperform the competing state-of-the-art approaches on a diversified set of benchmarks and applications.

        ----

        ## [787] Believe What You See: Implicit Constraint Approach for Offline Multi-Agent Reinforcement Learning

        **Authors**: *Yiqin Yang, Xiaoteng Ma, Chenghao Li, Zewu Zheng, Qiyuan Zhang, Gao Huang, Jun Yang, Qianchuan Zhao*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/550a141f12de6341fba65b0ad0433500-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/550a141f12de6341fba65b0ad0433500-Abstract.html)

        **Abstract**:

        Learning from datasets without interaction with environments (Offline Learning) is an essential step to apply Reinforcement Learning (RL) algorithms in real-world scenarios.However, compared with the single-agent counterpart, offline multi-agent RL introduces more agents with the larger state and action space, which is more challenging but attracts little attention. We demonstrate current offline RL algorithms are ineffective in multi-agent systems due to the accumulated extrapolation error. In this paper, we propose a novel offline RL algorithm, named Implicit Constraint Q-learning (ICQ), which effectively alleviates the extrapolation error by only trusting the state-action pairs given in the dataset for value estimation.  Moreover, we extend ICQ to multi-agent tasks by decomposing the joint-policy under the implicit constraint.  Experimental results demonstrate that the extrapolation error is successfully controlled within a reasonable range and insensitive to the number of agents. We further show that ICQ achieves the state-of-the-art performance in the challenging multi-agent offline tasks (StarCraft II). Our code is public online at https://github.com/YiqinYang/ICQ.

        ----

        ## [788] Online Learning in Periodic Zero-Sum Games

        **Authors**: *Tanner Fiez, Ryann Sim, Stratis Skoulakis, Georgios Piliouras, Lillian J. Ratliff*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/55563844bcd4bba067fe86ac1f008c7e-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/55563844bcd4bba067fe86ac1f008c7e-Abstract.html)

        **Abstract**:

        A seminal result in game theory is von Neumann's minmax theorem, which states that zero-sum games admit an essentially unique equilibrium solution. Classical learning results build on this theorem to show that online no-regret dynamics converge to an equilibrium in a time-average sense in zero-sum games. In the past several years, a key research direction has focused on characterizing the transient behavior of such dynamics. General results in this direction show that broad classes of online learning dynamics are cyclic, and formally Poincar\'{e} recurrent, in zero-sum games. We analyze the robustness of these online learning behaviors in the case of periodic zero-sum games with a time-invariant equilibrium. This model generalizes the usual repeated game formulation while also being a realistic and natural model of a repeated competition between players that depends on exogenous environmental variations such as time-of-day effects, week-to-week trends, and seasonality. Interestingly, time-average convergence may fail even in the simplest such settings, in spite of the equilibrium being fixed. In contrast, using novel analysis methods, we show that Poincar\'{e} recurrence provably generalizes despite the complex, non-autonomous nature of these dynamical systems.

        ----

        ## [789] K-Net: Towards Unified Image Segmentation

        **Authors**: *Wenwei Zhang, Jiangmiao Pang, Kai Chen, Chen Change Loy*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/55a7cf9c71f1c9c495413f934dd1a158-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/55a7cf9c71f1c9c495413f934dd1a158-Abstract.html)

        **Abstract**:

        Semantic, instance, and panoptic segmentations have been addressed using different and specialized frameworks despite their underlying connections. This paper presents a unified, simple, and effective framework for these essentially similar tasks. The framework, named K-Net, segments both instances and semantic categories consistently by a group of learnable kernels, where each kernel is responsible for generating a mask for either a potential instance or a stuff class. To remedy the difficulties of distinguishing various instances, we propose a kernel update strategy that enables each kernel dynamic and conditional on its meaningful group in the input image. K-Net can be trained in an end-to-end manner with bipartite matching, and its training and inference are naturally NMS-free and box-free. Without bells and whistles, K-Net surpasses all previous published state-of-the-art single-model results of panoptic segmentation on MS COCO test-dev split and semantic segmentation on ADE20K val split with 55.2% PQ and 54.3% mIoU, respectively. Its instance segmentation performance is also on par with Cascade Mask R-CNN on MS COCO with 60%-90% faster inference speeds. Code and models will be released at https://github.com/ZwwWayne/K-Net/.

        ----

        ## [790] Pareto-Optimal Learning-Augmented Algorithms for Online Conversion Problems

        **Authors**: *Bo Sun, Russell Lee, Mohammad H. Hajiesmaili, Adam Wierman, Danny H. K. Tsang*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/55a988dfb00a914717b3000a3374694c-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/55a988dfb00a914717b3000a3374694c-Abstract.html)

        **Abstract**:

        This paper leverages machine-learned predictions to design competitive algorithms for online conversion problems with the goal of improving the competitive ratio when predictions are accurate (i.e., consistency), while also guaranteeing a worst-case competitive ratio regardless of the prediction quality (i.e., robustness). We unify the algorithmic design of both integral and fractional conversion problems, which are also known as the 1-max-search and one-way trading problems, into a class of online threshold-based algorithms (OTA). By incorporating predictions into design of OTA, we achieve the Pareto-optimal trade-off of consistency and robustness, i.e., no online algorithm can achieve a better consistency guarantee given for a robustness guarantee. We demonstrate the performance of OTA using numerical experiments on Bitcoin conversion.

        ----

        ## [791] Dynaboard: An Evaluation-As-A-Service Platform for Holistic Next-Generation Benchmarking

        **Authors**: *Zhiyi Ma, Kawin Ethayarajh, Tristan Thrush, Somya Jain, Ledell Wu, Robin Jia, Christopher Potts, Adina Williams, Douwe Kiela*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/55b1927fdafef39c48e5b73b5d61ea60-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/55b1927fdafef39c48e5b73b5d61ea60-Abstract.html)

        **Abstract**:

        We introduce Dynaboard, an evaluation-as-a-service framework for hosting benchmarks and conducting holistic model comparison, integrated with the Dynabench platform. Our platform evaluates NLP models directly instead of relying on self-reported metrics or predictions on a single dataset. Under this paradigm, models are submitted to be evaluated in the cloud, circumventing the issues of reproducibility, accessibility, and backwards compatibility that often hinder benchmarking in NLP. This allows users to interact with uploaded models in real time to assess their quality, and permits the collection of additional metrics such as memory use, throughput, and robustness, which -- despite their importance to practitioners -- have traditionally been absent from leaderboards. On each task, models are ranked according to the Dynascore, a novel utility-based aggregation of these statistics, which users can customize to better reflect their preferences, placing more/less weight on a particular axis of evaluation or dataset. As state-of-the-art NLP models push the limits of traditional benchmarks, Dynaboard offers a standardized solution for a more diverse and comprehensive evaluation of model quality.

        ----

        ## [792] NTopo: Mesh-free Topology Optimization using Implicit Neural Representations

        **Authors**: *Jonas Zehnder, Yue Li, Stelian Coros, Bernhard Thomaszewski*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/55d99a37b2e1badba7c8df4ccd506a88-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/55d99a37b2e1badba7c8df4ccd506a88-Abstract.html)

        **Abstract**:

        Recent advances in implicit neural representations show great promise when it comes to generating numerical solutions to partial differential equations. Compared to conventional alternatives, such representations employ parameterized neural networks to define, in a mesh-free manner, signals that are highly-detailed, continuous, and fully differentiable. In this work, we present a novel machine learning approach for topology optimization---an important class of inverse problems with high-dimensional parameter spaces and highly nonlinear objective landscapes. To effectively leverage neural representations in the context of mesh-free topology optimization, we use multilayer perceptrons to parameterize both density and displacement fields. Our experiments indicate that our method is highly competitive for minimizing  structural compliance objectives, and it enables self-supervised learning of continuous solution spaces for topology optimization problems.

        ----

        ## [793] Generalization Bounds for (Wasserstein) Robust Optimization

        **Authors**: *Yang An, Rui Gao*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/55fd1368113e5a675e868c5653a7bb9e-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/55fd1368113e5a675e868c5653a7bb9e-Abstract.html)

        **Abstract**:

        (Distributionally) robust optimization has gained momentum in machine learning community recently, due to its promising applications in developing generalizable learning paradigms. In this paper, we derive generalization bounds for robust optimization and Wasserstein robust optimization for Lipschitz and piecewise HÃ¶lder smooth loss functions under both stochastic and adversarial setting, assuming that the underlying data distribution satisfies transportation-information inequalities. The proofs are built on new generalization bounds for variation regularization (such as Lipschitz or gradient regularization) and its connection with robustness.

        ----

        ## [794] Faster Matchings via Learned Duals

        **Authors**: *Michael Dinitz, Sungjin Im, Thomas Lavastida, Benjamin Moseley, Sergei Vassilvitskii*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/5616060fb8ae85d93f334e7267307664-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/5616060fb8ae85d93f334e7267307664-Abstract.html)

        **Abstract**:

        A recent line of research investigates how algorithms can be augmented with machine-learned predictions to overcome worst case lower bounds.  This area has revealed interesting algorithmic insights into problems, with particular success in the design of competitive online algorithms.  However, the question of improving algorithm running times with predictions has largely been unexplored.  We take a first step in this direction by combining the idea of machine-learned predictions with the idea of ``warm-starting" primal-dual algorithms. We consider one of the most important primitives in combinatorial optimization: weighted bipartite matching and its generalization to $b$-matching. We identify three key challenges when using learned dual variables in a primal-dual algorithm.  First, predicted duals may be infeasible, so we give an algorithm that efficiently maps predicted infeasible duals to nearby feasible solutions.  Second, once the duals are feasible, they may not be optimal, so we show that they can be used to quickly find an optimal solution. Finally, such predictions are useful only if they can be learned, so we show that the problem of learning duals for matching has low sample complexity.  We validate our theoretical findings through experiments on both real and synthetic data.  As a result we give a rigorous, practical, and empirically effective method to compute bipartite matchings.

        ----

        ## [795] Online learning in MDPs with linear function approximation and bandit feedback

        **Authors**: *Gergely Neu, Julia Olkhovskaya*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/5631e6ee59a4175cd06c305840562ff3-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/5631e6ee59a4175cd06c305840562ff3-Abstract.html)

        **Abstract**:

        We consider the problem of online learning in an episodic Markov decision process, where the reward function is allowed to change between episodes in an adversarial manner and the learner only observes the rewards associated with its actions. We assume that rewards and the transition function can be represented as linear functions in terms of a known low-dimensional feature map, which allows us to consider the setting where  the state space is arbitrarily large. We also assume that the learner has a perfect knowledge of the MDP dynamics. Our main contribution is developing an algorithm whose expected regret after $T$ episodes is bounded by $\widetilde{\mathcal{O}}(\sqrt{dHT})$, where $H$ is the number of steps in each episode and $d$ is the dimensionality of the feature map.

        ----

        ## [796] Learning Collaborative Policies to Solve NP-hard Routing Problems

        **Authors**: *Minsu Kim, Jinkyoo Park, Joungho Kim*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/564127c03caab942e503ee6f810f54fd-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/564127c03caab942e503ee6f810f54fd-Abstract.html)

        **Abstract**:

        Recently, deep reinforcement learning (DRL) frameworks have shown potential for solving NP-hard routing problems such as the traveling salesman problem (TSP) without problem-specific expert knowledge. Although DRL can be used to solve complex problems, DRL frameworks still struggle to compete with state-of-the-art heuristics showing a substantial performance gap. This paper proposes a novel hierarchical problem-solving strategy, termed learning collaborative policies (LCP), which can effectively find the near-optimum solution using two iterative DRL policies: the seeder and reviser. The seeder generates as diversified candidate solutions as possible (seeds) while being dedicated to exploring over the full combinatorial action space (i.e., sequence of assignment action). To this end, we train the seeder's policy using a simple yet effective entropy regularization reward to encourage the seeder to find diverse solutions. On the other hand, the reviser modifies each candidate solution generated by the seeder; it partitions the full trajectory into sub-tours and simultaneously revises each sub-tour to minimize its traveling distance. Thus, the reviser is trained to improve the candidate solution's quality, focusing on the reduced solution space (which is beneficial for exploitation). Extensive experiments demonstrate that the proposed two-policies collaboration scheme improves over single-policy DRL framework on various NP-hard routing problems, including TSP, prize collecting TSP (PCTSP), and capacitated vehicle routing problem (CVRP).

        ----

        ## [797] Efficient Mirror Descent Ascent Methods for Nonsmooth Minimax Problems

        **Authors**: *Feihu Huang, Xidong Wu, Heng Huang*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/56503192b14190d3826780d47c0d3bf3-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/56503192b14190d3826780d47c0d3bf3-Abstract.html)

        **Abstract**:

        In the paper, we propose a class of  efficient mirror descent ascent methods to solve the nonsmooth nonconvex-strongly-concave minimax problems by using dynamic mirror functions, and introduce a convergence analysis framework to conduct rigorous theoretical analysis for our mirror descent ascent methods. For our stochastic algorithms, we first prove that the mini-batch stochastic mirror descent ascent (SMDA) method obtains a gradient  complexity of $O(\kappa^3\epsilon^{-4})$ for finding an $\epsilon$-stationary point, where $\kappa$ denotes the condition number. Further, we propose an accelerated stochastic mirror descent ascent (VR-SMDA) method based on  the variance reduced technique. We prove that our VR-SMDA method achieves a lower gradient complexity of  $O(\kappa^3\epsilon^{-3})$. For our deterministic algorithm, we prove that our deterministic mirror descent ascent (MDA) achieves a lower gradient complexity of $O(\sqrt{\kappa}\epsilon^{-2})$ under mild conditions, which matches the best known complexity in solving smooth nonconvex-strongly-concave minimax optimization. We conduct the experiments on fair classifier and robust neural network training tasks to demonstrate the efficiency of our new algorithms.

        ----

        ## [798] CO-PILOT: COllaborative Planning and reInforcement Learning On sub-Task curriculum

        **Authors**: *Shuang Ao, Tianyi Zhou, Guodong Long, Qinghua Lu, Liming Zhu, Jing Jiang*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/56577889b3c1cd083b6d7b32d32f99d5-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/56577889b3c1cd083b6d7b32d32f99d5-Abstract.html)

        **Abstract**:

        Goal-conditioned reinforcement learning (RL) usually suffers from sparse reward and inefficient exploration in long-horizon tasks. Planning can find the shortest path to a distant goal that provides dense reward/guidance but is inaccurate without a precise environment model. We show that RL and planning can collaboratively learn from each other to overcome their own drawbacks. In ''CO-PILOT'', a learnable path-planner and an RL agent produce dense feedback to train each other on a curriculum of tree-structured sub-tasks. Firstly, the planner recursively decomposes a long-horizon task to a tree of sub-tasks in a top-down manner, whose layers construct coarse-to-fine sub-task sequences as plans to complete the original task. The planning policy is trained to minimize the RL agent's cost of completing the sequence in each layer from top to bottom layers, which gradually increases the sub-tasks and thus forms an easy-to-hard curriculum for the planner.  Next, a bottom-up traversal of the tree trains the RL agent from easier sub-tasks with denser rewards on bottom layers to harder ones on top layers and collects its cost on each sub-task train the planner in the next episode. CO-PILOT repeats this mutual training for multiple episodes before switching to a new task, so the RL agent and planner are fully optimized to facilitate each other's training. We compare CO-PILOT with RL (SAC, HER, PPO), planning (RRT*, NEXT, SGT), and their combination (SoRB) on navigation and continuous control tasks. CO-PILOT significantly improves the success rate and sample efficiency.

        ----

        ## [799] Modality-Agnostic Topology Aware Localization

        **Authors**: *Farhad Ghazvinian Zanjani, Ilia Karmanov, Hanno Ackermann, Daniel Dijkman, Simone Merlin, Max Welling, Fatih Porikli*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/569ff987c643b4bedf504efda8f786c2-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/569ff987c643b4bedf504efda8f786c2-Abstract.html)

        **Abstract**:

        This work presents a data-driven approach for the indoor localization of an observer on a 2D topological map of the environment. State-of-the-art techniques may yield accurate estimates only when they are tailor-made for a specific data modality like camera-based system that prevents their applicability to broader domains. Here, we establish a modality-agnostic framework (called OT-Isomap) and formulate the localization problem in the context of parametric manifold learning while leveraging optimal transportation. This framework allows jointly learning a low-dimensional embedding as well as correspondences with a topological map. We examine the generalizability of the proposed algorithm by applying it to data from diverse modalities such as image sequences and radio frequency signals. The experimental results demonstrate decimeter-level accuracy for localization using different sensory inputs.

        ----

        

[Go to the previous page](NIPS-2021-list03.md)

[Go to the next page](NIPS-2021-list05.md)

[Go to the catalog section](README.md)