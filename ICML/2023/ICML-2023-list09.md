## [1600] Hierarchical Neural Coding for Controllable CAD Model Generation

        **Authors**: *Xiang Xu, Pradeep Kumar Jayaraman, Joseph George Lambourne, Karl D. D. Willis, Yasutaka Furukawa*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/xu23f.html](https://proceedings.mlr.press/v202/xu23f.html)

        **Abstract**:

        This paper presents a novel generative model for Computer Aided Design (CAD) that 1) represents high-level design concepts of a CAD model as a three-level hierarchical tree of neural codes, from global part arrangement down to local curve geometry; and 2) controls the generation or completion of CAD models by specifying the target design using a code tree. Concretely, a novel variant of a vector quantized VAE with "masked skip connection" extracts design variations as neural codebooks at three levels. Two-stage cascaded auto-regressive transformers learn to generate code trees from incomplete CAD models and then complete CAD models following the intended design. Extensive experiments demonstrate superior performance on conventional tasks such as unconditional generation while enabling novel interaction capabilities on conditional generation tasks. The code is available at https://github.com/samxuxiang/hnc-cad.

        ----

        ## [1601] Efficient Sequence Transduction by Jointly Predicting Tokens and Durations

        **Authors**: *Hainan Xu, Fei Jia, Somshubra Majumdar, He Huang, Shinji Watanabe, Boris Ginsburg*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/xu23g.html](https://proceedings.mlr.press/v202/xu23g.html)

        **Abstract**:

        This paper introduces a novel Token-and-Duration Transducer (TDT) architecture for sequence-to-sequence tasks. TDT extends conventional RNN-Transducer architectures by jointly predicting both a token and its duration, i.e. the number of input frames covered by the emitted token. This is achieved by using a joint network with two outputs which are independently normalized to generate distributions over tokens and durations. During inference, TDT models can skip input frames guided by the predicted duration output, which makes them significantly faster than conventional Transducers which process the encoder output frame by frame. TDT models achieve both better accuracy and significantly faster inference than conventional Transducers on different sequence transduction tasks. TDT models for Speech Recognition achieve better accuracy and up to 2.82X faster inference than conventional Transducers. TDT models for Speech Translation achieve an absolute gain of over 1 BLEU on the MUST-C test compared with conventional Transducers, and its inference is 2.27X faster. In Speech Intent Classification and Slot Filling tasks, TDT models improve the intent accuracy by up to over 1% (absolute) over conventional Transducers, while running up to 1.28X faster. Our implementation of the TDT model will be open-sourced with the NeMo (https://github.com/NVIDIA/NeMo) toolkit.

        ----

        ## [1602] Constrained Efficient Global Optimization of Expensive Black-box Functions

        **Authors**: *Wenjie Xu, Yuning Jiang, Bratislav Svetozarevic, Colin N. Jones*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/xu23h.html](https://proceedings.mlr.press/v202/xu23h.html)

        **Abstract**:

        We study the problem of constrained efficient global optimization, where both the objective and constraints are expensive black-box functions that can be learned with Gaussian processes. We propose CONFIG (CONstrained efFIcient Global Optimization), a simple and effective algorithm to solve it. Under certain regularity assumptions, we show that our algorithm enjoys the same cumulative regret bound as that in the unconstrained case and similar cumulative constraint violation upper bounds. For commonly used Matern and Squared Exponential kernels, our bounds are sublinear and allow us to derive a convergence rate to the optimal solution of the original constrained problem. In addition, our method naturally provides a scheme to declare infeasibility when the original black-box optimization problem is infeasible. Numerical experiments on sampled instances from the Gaussian process, artificial numerical problems, and a black-box building controller tuning problem all demonstrate the competitive performance of our algorithm. Compared to the other state-of-the-art methods, our algorithm significantly improves the theoretical guarantees while achieving competitive empirical performance.

        ----

        ## [1603] Pareto Regret Analyses in Multi-objective Multi-armed Bandit

        **Authors**: *Mengfan Xu, Diego Klabjan*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/xu23i.html](https://proceedings.mlr.press/v202/xu23i.html)

        **Abstract**:

        We study Pareto optimality in multi-objective multi-armed bandit by providing a formulation of adversarial multi-objective multi-armed bandit and defining its Pareto regrets that can be applied to both stochastic and adversarial settings. The regrets do not rely on any scalarization functions and reflect Pareto optimality compared to scalarized regrets. We also present new algorithms assuming both with and without prior information of the multi-objective multi-armed bandit setting. The algorithms are shown optimal in adversarial settings and nearly optimal up to a logarithmic factor in stochastic settings simultaneously by our established upper bounds and lower bounds on Pareto regrets. Moreover, the lower bound analyses show that the new regrets are consistent with the existing Pareto regret for stochastic settings and extend an adversarial attack mechanism from bandit to the multi-objective one.

        ----

        ## [1604] Diverse and Faithful Knowledge-Grounded Dialogue Generation via Sequential Posterior Inference

        **Authors**: *Yan Xu, Deqian Kong, Dehong Xu, Ziwei Ji, Bo Pang, Pascale Fung, Ying Nian Wu*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/xu23j.html](https://proceedings.mlr.press/v202/xu23j.html)

        **Abstract**:

        The capability to generate responses with diversity and faithfulness using factual knowledge is paramount for creating a human-like, trustworthy dialogue system. Common strategies either adopt a two-step paradigm, which optimizes knowledge selection and response generation separately, and may overlook the inherent correlation between these two tasks, or leverage conditional variational method to jointly optimize knowledge selection and response generation by employing an inference network. In this paper, we present an end-to-end learning framework, termed Sequential Posterior Inference (SPI), capable of selecting knowledge and generating dialogues by approximately sampling from the posterior distribution. Unlike other methods, SPI does not require the inference network or assume a simple geometry of the posterior distribution. This straightforward and intuitive inference procedure of SPI directly queries the response generation model, allowing for accurate knowledge selection and generation of faithful responses. In addition to modeling contributions, our experimental results on two common dialogue datasets (Wizard of Wikipedia and Holl-E) demonstrate that SPI outperforms previous strong baselines according to both automatic and human evaluation metrics.

        ----

        ## [1605] Quantifying the Variability Collapse of Neural Networks

        **Authors**: *Jing Xu, Haoxiong Liu*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/xu23k.html](https://proceedings.mlr.press/v202/xu23k.html)

        **Abstract**:

        Recent studies empirically demonstrate the positive relationship between the transferability of neural networks and the in-class variation of the last layer features. The recently discovered Neural Collapse (NC) phenomenon provides a new perspective of understanding such last layer geometry of neural networks. In this paper, we propose a novel metric, named Variability Collapse Index (VCI), to quantify the variability collapse phenomenon in the NC paradigm. The VCI metric is well-motivated and intrinsically related to the linear probing loss on the last layer features. Moreover, it enjoys desired theoretical and empirical properties, including invariance under invertible linear transformations and numerical stability, that distinguishes it from previous metrics. Our experiments verify that VCI is indicative of the variability collapse and the transferability of pretrained neural networks.

        ----

        ## [1606] Progressive Purification for Instance-Dependent Partial Label Learning

        **Authors**: *Ning Xu, Biao Liu, Jiaqi Lv, Congyu Qiao, Xin Geng*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/xu23l.html](https://proceedings.mlr.press/v202/xu23l.html)

        **Abstract**:

        Partial label learning (PLL) aims to train multiclass classifiers from the examples each annotated with a set of candidate labels where a fixed but unknown candidate label is correct. In the last few years, the instance-independent generation process of candidate labels has been extensively studied, on the basis of which many theoretical advances have been made in PLL. Nevertheless, the candidate labels are always instance-dependent in practice and there is no theoretical guarantee that the model trained on the instance-dependent PLL examples can converge to an ideal one. In this paper, a theoretically grounded and practically effective approach named POP, i.e. PrOgressive Purification for instance-dependent partial label learning, is proposed. Specifically, POP updates the learning model and purifies each candidate label set progressively in every epoch. Theoretically, we prove that POP enlarges the region appropriately fast where the model is reliable, and eventually approximates the Bayes optimal classifier with mild assumptions. Technically, POP is flexible with arbitrary PLL losses and could improve the performance of the previous PLL losses in the instance-dependent case. Experiments on the benchmark datasets and the real-world datasets validate the effectiveness of the proposed method.

        ----

        ## [1607] PFGM++: Unlocking the Potential of Physics-Inspired Generative Models

        **Authors**: *Yilun Xu, Ziming Liu, Yonglong Tian, Shangyuan Tong, Max Tegmark, Tommi S. Jaakkola*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/xu23m.html](https://proceedings.mlr.press/v202/xu23m.html)

        **Abstract**:

        We introduce a new family of physics-inspired generative models termed PFGM++ that unifies diffusion models and Poisson Flow Generative Models (PFGM). These models realize generative trajectories for N dimensional data by embedding paths in N+D dimensional space while still controlling the progression with a simple scalar norm of the D additional variables. The new models reduce to PFGM when D=1 and to diffusion models when D$\to\infty$. The flexibility of choosing D allows us to trade off robustness against rigidity as increasing D results in more concentrated coupling between the data and the additional variable norms. We dispense with the biased large batch field targets used in PFGM and instead provide an unbiased perturbation-based objective similar to diffusion models. To explore different choices of D, we provide a direct alignment method for transferring well-tuned hyperparameters from diffusion models (D$\to\infty$) to any finite D values. Our experiments show that models with finite D can be superior to previous state-of-the-art diffusion models on CIFAR-10/FFHQ 64$\times$64 datasets/LSUN Churches 256$\times$256, with median Ds. In class-conditional setting, D=2048 yields current state-of-the-art FID of 1.74 on CIFAR-10 without additional training. Furthermore, we demonstrate that models with smaller $D$ exhibit improved robustness against modeling errors. Code is available at https://github.com/Newbeeer/pfgmpp

        ----

        ## [1608] Geometric Latent Diffusion Models for 3D Molecule Generation

        **Authors**: *Minkai Xu, Alexander S. Powers, Ron O. Dror, Stefano Ermon, Jure Leskovec*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/xu23n.html](https://proceedings.mlr.press/v202/xu23n.html)

        **Abstract**:

        Generative models, especially diffusion models (DMs), have achieved promising results for generating feature-rich geometries and advancing foundational science problems such as molecule design. Inspired by the recent huge success of Stable (latent) Diffusion models, we propose a novel and principled method for 3D molecule generation named Geometric Latent Diffusion Models (GeoLDM). GeoLDM is the first latent DM model for the molecular geometry domain, composed of autoencoders encoding structures into continuous latent codes and DMs operating in the latent space. Our key innovation is that for modeling the 3D molecular geometries, we capture its critical roto-translational equivariance constraints by building a point-structured latent space with both invariant scalars and equivariant tensors. Extensive experiments demonstrate that GeoLDM can consistently achieve better performance on multiple molecule generation benchmarks, with up to 7% improvement for the valid percentage of large biomolecules. Results also demonstrate GeoLDM’s higher capacity for controllable generation thanks to the latent modeling. Code is provided at https://github.com/MinkaiXu/GeoLDM.

        ----

        ## [1609] The Power of Preconditioning in Overparameterized Low-Rank Matrix Sensing

        **Authors**: *Xingyu Xu, Yandi Shen, Yuejie Chi, Cong Ma*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/xu23o.html](https://proceedings.mlr.press/v202/xu23o.html)

        **Abstract**:

        We propose $\textsf{ScaledGD($\lambda$)}$, a preconditioned gradient descent method to tackle the low-rank matrix sensing problem when the true rank is unknown, and when the matrix is possibly ill-conditioned. Using overparametrized factor representations, $\textsf{ScaledGD($\lambda$)}$ starts from a small random initialization, and proceeds by gradient descent with a specific form of preconditioning with a fixed damping term to combat overparameterization. At the expense of light computational overhead incurred by preconditioners, $\textsf{ScaledGD($\lambda$)}$ is remarkably robust to ill-conditioning compared to vanilla gradient descent ($\mathsf{GD}$). Specifically, we show that, under the Gaussian design, $\textsf{ScaledGD($\lambda$)}$ converges to the true low-rank matrix at a constant linear rate that is independent of the condition number (apart from a short nearly dimension-free burdening period), with near-optimal sample complexity. This significantly improves upon the convergence rate of vanilla $\mathsf{GD}$ which suffers from a polynomial dependency with the condition number. Our work provides evidence on the power of preconditioning in accelerating the convergence without hurting generalization in overparameterized learning.

        ----

        ## [1610] Fascinating Supervisory Signals and Where to Find Them: Deep Anomaly Detection with Scale Learning

        **Authors**: *Hongzuo Xu, Yijie Wang, Juhui Wei, Songlei Jian, Yizhou Li, Ning Liu*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/xu23p.html](https://proceedings.mlr.press/v202/xu23p.html)

        **Abstract**:

        Due to the unsupervised nature of anomaly detection, the key to fueling deep models is finding supervisory signals. Different from current reconstruction-guided generative models and transformation-based contrastive models, we devise novel data-driven supervision for tabular data by introducing a characteristic – scale – as data labels. By representing varied sub-vectors of data instances, we define scale as the relationship between the dimensionality of original sub-vectors and that of representations. Scales serve as labels attached to transformed representations, thus offering ample labeled data for neural network training. This paper further proposes a scale learning-based anomaly detection method. Supervised by the learning objective of scale distribution alignment, our approach learns the ranking of representations converted from varied subspaces of each data instance. Through this proxy task, our approach models inherent regularities and patterns within data, which well describes data "normality". Abnormal degrees of testing instances are obtained by measuring whether they fit these learned patterns. Extensive experiments show that our approach leads to significant improvement over state-of-the-art generative/contrastive anomaly detection methods.

        ----

        ## [1611] Competing for Shareable Arms in Multi-Player Multi-Armed Bandits

        **Authors**: *Renzhe Xu, Haotian Wang, Xingxuan Zhang, Bo Li, Peng Cui*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/xu23q.html](https://proceedings.mlr.press/v202/xu23q.html)

        **Abstract**:

        Competitions for shareable and limited resources have long been studied with strategic agents. In reality, agents often have to learn and maximize the rewards of the resources at the same time. To design an individualized competing policy, we model the competition between agents in a novel multi-player multi-armed bandit (MPMAB) setting where players are selfish and aim to maximize their own rewards. In addition, when several players pull the same arm, we assume that these players averagely share the arms’ rewards by expectation. Under this setting, we first analyze the Nash equilibrium when arms’ rewards are known. Subsequently, we propose a novel Selfish MPMAB with Averaging Allocation (SMAA) approach based on the equilibrium. We theoretically demonstrate that SMAA could achieve a good regret guarantee for each player when all players follow the algorithm. Additionally, we establish that no single selfish player can significantly increase their rewards through deviation, nor can they detrimentally affect other players’ rewards without incurring substantial losses for themselves. We finally validate the effectiveness of the method in extensive synthetic experiments.

        ----

        ## [1612] Sequential Predictive Conformal Inference for Time Series

        **Authors**: *Chen Xu, Yao Xie*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/xu23r.html](https://proceedings.mlr.press/v202/xu23r.html)

        **Abstract**:

        We present a new distribution-free conformal prediction algorithm for sequential data (e.g., time series), called the sequential predictive conformal inference (SPCI). We specifically account for the nature that time series data are non-exchangeable, and thus many existing conformal prediction algorithms are not applicable. The main idea is to adaptively re-estimate the conditional quantile of non-conformity scores (e.g., prediction residuals), upon exploiting the temporal dependence among them. More precisely, we cast the problem of conformal prediction interval as predicting the quantile of a future residual, given a user-specified point prediction algorithm. Theoretically, we establish asymptotic valid conditional coverage upon extending consistency analyses in quantile regression. Using simulation and real-data experiments, we demonstrate a significant reduction in interval width of SPCI compared to other existing methods under the desired empirical coverage.

        ----

        ## [1613] mPLUG-2: A Modularized Multi-modal Foundation Model Across Text, Image and Video

        **Authors**: *Haiyang Xu, Qinghao Ye, Ming Yan, Yaya Shi, Jiabo Ye, Yuanhong Xu, Chenliang Li, Bin Bi, Qi Qian, Wei Wang, Guohai Xu, Ji Zhang, Songfang Huang, Fei Huang, Jingren Zhou*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/xu23s.html](https://proceedings.mlr.press/v202/xu23s.html)

        **Abstract**:

        Recent years have witnessed a big convergence of language, vision, and multi-modal pretraining. In this work, we present mPLUG-2, a new unified paradigm with modularized design for multi-modal pretraining, which can benefit from modality collaboration while addressing the problem of modality entanglement. In contrast to predominant paradigms of solely relying on sequence-to-sequence generation or encoder-based instance discrimination, mPLUG-2 introduces a multi-module composition network by sharing common universal modules for modality collaboration and disentangling different modality modules to deal with modality entanglement. It is flexible to select different modules for different understanding and generation tasks across all modalities including text, image, and video. Empirical study shows that mPLUG-2 achieves state-of-the-art or competitive results on a broad range of over 30 downstream tasks, spanning multi-modal tasks of image-text and video-text understanding and generation, and uni-modal tasks of text-only, image-only, and video-only understanding. Notably, mPLUG-2 shows new state-of-the-art results of 48.0 top-1 accuracy and 80.3 CIDEr on the challenging MSRVTT video QA and video caption tasks with a far smaller model size and data scale. It also demonstrates strong zero-shot transferability on vision-language and video-language tasks. Code and models will be released in https://github.com/X-PLUG/mPLUG-2.

        ----

        ## [1614] ProtST: Multi-Modality Learning of Protein Sequences and Biomedical Texts

        **Authors**: *Minghao Xu, Xinyu Yuan, Santiago Miret, Jian Tang*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/xu23t.html](https://proceedings.mlr.press/v202/xu23t.html)

        **Abstract**:

        Current protein language models (PLMs) learn protein representations mainly based on their sequences, thereby well capturing co-evolutionary information, but they are unable to explicitly acquire protein functions, which is the end goal of protein representation learning. Fortunately, for many proteins, their textual property descriptions are available, where their various functions are also described. Motivated by this fact, we first build the ProtDescribe dataset to augment protein sequences with text descriptions of their functions and other important properties. Based on this dataset, we propose the ProtST framework to enhance Protein Sequence pre-training and understanding by biomedical Texts. During pre-training, we design three types of tasks, i.e., unimodal mask prediction, multimodal representation alignment and multimodal mask prediction, to enhance a PLM with protein property information with different granularities and, at the same time, preserve the PLM’s original representation power. On downstream tasks, ProtST enables both supervised learning and zero-shot prediction. We verify the superiority of ProtST-induced PLMs over previous ones on diverse representation learning benchmarks. Under the zero-shot setting, we show the effectiveness of ProtST on zero-shot protein classification, and ProtST also enables functional protein retrieval from a large-scale database without any function annotation.

        ----

        ## [1615] Bayesian Design Principles for Frequentist Sequential Learning

        **Authors**: *Yunbei Xu, Assaf Zeevi*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/xu23u.html](https://proceedings.mlr.press/v202/xu23u.html)

        **Abstract**:

        We develop a general theory to optimize the frequentist regret for sequential learning problems, where efficient bandit and reinforcement learning algorithms can be derived from unified Bayesian principles. We propose a novel optimization approach to create "algorithmic beliefs" at each round, and use Bayesian posteriors to make decisions. This is the first approach to make Bayesian-type algorithms prior-free and applicable to adversarial settings, in a generic and optimal manner. Moreover, the algorithms are simple and often efficient to implement. As a major application, we present a novel algorithm for multi-armed bandits that achieves the "best-of-all-worlds" empirical performance in the stochastic, adversarial, and non-stationary environments. And we illustrate how these principles can be used in linear bandits, convex bandits, and reinforcement learning.

        ----

        ## [1616] SLAMB: Accelerated Large Batch Training with Sparse Communication

        **Authors**: *Hang Xu, Wenxuan Zhang, Jiawei Fei, Yuzhe Wu, Tingwen Xie, Jun Huang, Yuchen Xie, Mohamed Elhoseiny, Panos Kalnis*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/xu23v.html](https://proceedings.mlr.press/v202/xu23v.html)

        **Abstract**:

        Distributed training of large deep neural networks requires frequent exchange of massive data between machines, thus communication efficiency is a major concern. Existing compressed communication methods are either not compatible with large batch optimization algorithms, or do not provide sufficient speedup in large scale. In this paper, we combine sparsification-based gradient compression with the layer-wise adaptive moments optimizer for large batch training (LAMB). We propose SLAMB, a novel communication-efficient optimizer that supports large batch sizes and scales to thousands of GPUs. SLAMB employs momentum masking, local error compensation, and element-wise adaptive rescaling to achieve accurate layer-wise weight updates, which translates to fast convergence for very large batches. Our empirical results show that, compared to the state-of-the-art, SLAMB transmits half the amount of data in large-batch BERT pre-training, without sacrificing accuracy. Moreover, SLAMB achieves excellent scalability in large computing infrastructures. For instance, SLAMB with 128 GPUs reduces the training time of Swin Transformer pre-training on ImageNet to 5.35 hours, which is 2 hours faster than the state-of-the-art. At the extreme, we trained BERT-XL (2.8B parameters) on 1,024 NVIDIA A100 GPUs, where SLAMB achieved 90% scaling efficiency.

        ----

        ## [1617] Do Not Train It: A Linear Neural Architecture Search of Graph Neural Networks

        **Authors**: *Peng Xu, Lin Zhang, Xuanzhou Liu, Jiaqi Sun, Yue Zhao, Haiqin Yang, Bei Yu*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/xu23w.html](https://proceedings.mlr.press/v202/xu23w.html)

        **Abstract**:

        Neural architecture search (NAS) for Graph neural networks (GNNs), called NAS-GNNs, has achieved significant performance over manually designed GNN architectures. However, these methods inherit issues from the conventional NAS methods, such as high computational cost and optimization difficulty. More importantly, previous NAS methods have ignored the uniqueness of GNNs, where GNNs possess expressive power without training. With the randomly-initialized weights, we can then seek the optimal architecture parameters via the sparse coding objective and derive a novel NAS-GNNs method, namely neural architecture coding (NAC). Consequently, our NAC holds a no-update scheme on GNNs and can efficiently compute in linear time. Empirical evaluations on multiple GNN benchmark datasets demonstrate that our approach leads to state-of-the-art performance, which is up to $200\times$ faster and $18.8%$ more accurate than the strong baselines.

        ----

        ## [1618] An Instrumental Variable Approach to Confounded Off-Policy Evaluation

        **Authors**: *Yang Xu, Jin Zhu, Chengchun Shi, Shikai Luo, Rui Song*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/xu23x.html](https://proceedings.mlr.press/v202/xu23x.html)

        **Abstract**:

        Off-policy evaluation (OPE) aims to estimate the return of a target policy using some pre-collected observational data generated by a potentially different behavior policy. In many cases, there exist unmeasured variables that confound the action-reward or action-next-state relationships, rendering many existing OPE approaches ineffective. This paper develops an instrumental variable (IV)-based method for consistent OPE in confounded sequential decision making. Similar to single-stage decision making, we show that IV enables us to correctly identify the target policy’s value in infinite horizon settings as well. Furthermore, we propose a number of policy value estimators and illustrate their effectiveness through extensive simulations and real data analysis from a world-leading short-video platform.

        ----

        ## [1619] Near-Optimal Quantum Coreset Construction Algorithms for Clustering

        **Authors**: *Yecheng Xue, Xiaoyu Chen, Tongyang Li, Shaofeng H.-C. Jiang*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/xue23a.html](https://proceedings.mlr.press/v202/xue23a.html)

        **Abstract**:

        $k$-Clustering in $\mathbb{R}^d$ (e.g., $k$-median and $k$-means) is a fundamental machine learning problem. While near-linear time approximation algorithms were known in the classical setting for a dataset with cardinality $n$, it remains open to find sublinear-time quantum algorithms. We give quantum algorithms that find coresets for $k$-clustering in $\mathbb{R}^d$ with $\tilde{O}(\sqrt{nk}d^{3/2})$ query complexity. Our coreset reduces the input size from $n$ to $\mathrm{poly}(k\epsilon^{-1}d)$, so that existing $\alpha$-approximation algorithms for clustering can run on top of it and yield $(1 + \epsilon)\alpha$-approximation. This eventually yields a quadratic speedup for various $k$-clustering approximation algorithms. We complement our algorithm with a nearly matching lower bound, that any quantum algorithm must make $\Omega(\sqrt{nk})$ queries in order to achieve even $O(1)$-approximation for $k$-clustering.

        ----

        ## [1620] A Study on Transformer Configuration and Training Objective

        **Authors**: *Fuzhao Xue, Jianghai Chen, Aixin Sun, Xiaozhe Ren, Zangwei Zheng, Xiaoxin He, Yongming Chen, Xin Jiang, Yang You*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/xue23b.html](https://proceedings.mlr.press/v202/xue23b.html)

        **Abstract**:

        Transformer-based models have delivered impressive results on many tasks, particularly vision and language tasks. In many model training situations, conventional configurations are often adopted. For example, we usually set the base model with hidden size (i.e. model width) to be 768 and the number of transformer layers (i.e. model depth) to be 12. In this paper, we revisit these conventional configurations by studying the the relationship between transformer configuration and training objective. We show that the optimal transformer configuration is closely related to the training objective. Specifically, compared with the simple classification objective, the masked autoencoder is effective in alleviating the over-smoothing issue in deep transformer training. Based on this finding, we propose “Bamboo”, a notion of using deeper and narrower transformer configurations, for masked autoencoder training. On ImageNet, with such a simple change in configuration, the re-designed Base-level transformer achieves 84.2% top-1 accuracy and outperforms SoTA models like MAE by $0.9%$. On language tasks, re-designed model outperforms BERT with the default setting by 1.1 points on average, on GLUE benchmark with 8 datasets.

        ----

        ## [1621] LazyGNN: Large-Scale Graph Neural Networks via Lazy Propagation

        **Authors**: *Rui Xue, Haoyu Han, MohamadAli Torkamani, Jian Pei, Xiaorui Liu*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/xue23c.html](https://proceedings.mlr.press/v202/xue23c.html)

        **Abstract**:

        Recent works have demonstrated the benefits of capturing long-distance dependency in graphs by deeper graph neural networks (GNNs). But deeper GNNs suffer from the long-lasting scalability challenge due to the neighborhood explosion problem in large-scale graphs. In this work, we propose to capture long-distance dependency in graphs by shallower models instead of deeper models, which leads to a much more efficient model, LazyGNN, for graph representation learning. Moreover, we demonstrate that LazyGNN is compatible with existing scalable approaches (such as sampling methods) for further accelerations through the development of mini-batch LazyGNN. Comprehensive experiments demonstrate its superior prediction performance and scalability on large-scale benchmarks. The implementation of LazyGNN is available at https: //github.com/RXPHD/Lazy_GNN.

        ----

        ## [1622] Which Features are Learnt by Contrastive Learning? On the Role of Simplicity Bias in Class Collapse and Feature Suppression

        **Authors**: *Yihao Xue, Siddharth Joshi, Eric Gan, Pin-Yu Chen, Baharan Mirzasoleiman*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/xue23d.html](https://proceedings.mlr.press/v202/xue23d.html)

        **Abstract**:

        Contrastive learning (CL) has emerged as a powerful technique for representation learning, with or without label supervision. However, supervised CL is prone to collapsing representations of subclasses within a class by not capturing all their features, and unsupervised CL may suppress harder class-relevant features by focusing on learning easy class-irrelevant features; both significantly compromise representation quality. Yet, there is no theoretical understanding of class collapse or feature suppression at test time. We provide the first unified theoretically rigorous framework to determine which features are learnt by CL. Our analysis indicate that, perhaps surprisingly, bias of (stochastic) gradient descent towards finding simpler solutions is a key factor in collapsing subclass representations and suppressing harder class-relevant features. Moreover, we present increasing embedding dimensionality and improving the quality of data augmentations as two theoretically motivated solutions to feature suppression. We also provide the first theoretical explanation for why employing supervised and unsupervised CL together yields higher-quality representations, even when using commonly-used stochastic gradient methods.

        ----

        ## [1623] Adaptive Computation with Elastic Input Sequence

        **Authors**: *Fuzhao Xue, Valerii Likhosherstov, Anurag Arnab, Neil Houlsby, Mostafa Dehghani, Yang You*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/xue23e.html](https://proceedings.mlr.press/v202/xue23e.html)

        **Abstract**:

        Humans have the ability to adapt the type of information they use, the procedure they employ, and the amount of time they spend when solving problems. However, most standard neural networks have a fixed function type and computation budget regardless of the sample’s nature or difficulty. Adaptivity is a powerful paradigm as it not only imbues practitioners with flexibility pertaining to the downstream usage of these models but can also serve as a powerful inductive bias for solving certain challenging classes of problems. In this work, we introduce a new approach called AdaTape, which allows for dynamic computation in neural networks through adaptive tape tokens. AdaTape utilizes an elastic input sequence by equipping an architecture with a dynamic read-and-write tape. Specifically, we adaptively generate input sequences using tape tokens obtained from a tape bank which can be either trainable or derived from input data. We examine the challenges and requirements to obtain dynamic sequence content and length, and propose the Adaptive Tape Reading (ATR) algorithm to achieve both goals. Through extensive experiments on image recognition tasks, we show that AdaTape can achieve better performance while maintaining the computational cost. To facilitate further research, we have released code at https://github.com/google-research/scenic/tree/main/scenic/projects/adatape.

        ----

        ## [1624] Q-learning Decision Transformer: Leveraging Dynamic Programming for Conditional Sequence Modelling in Offline RL

        **Authors**: *Taku Yamagata, Ahmed Khalil, Raúl Santos-Rodríguez*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/yamagata23a.html](https://proceedings.mlr.press/v202/yamagata23a.html)

        **Abstract**:

        Recent works have shown that tackling offline reinforcement learning (RL) with a conditional policy produces promising results. The Decision Transformer (DT) combines the conditional policy approach and a transformer architecture, showing competitive performance against several benchmarks. However, DT lacks stitching ability – one of the critical abilities for offline RL to learn the optimal policy from sub-optimal trajectories. This issue becomes particularly significant when the offline dataset only contains sub-optimal trajectories. On the other hand, the conventional RL approaches based on Dynamic Programming (such as Q-learning) do not have the same limitation; however, they suffer from unstable learning behaviours, especially when they rely on function approximation in an off-policy learning setting. In this paper, we propose the Q-learning Decision Transformer (QDT) to address the shortcomings of DT by leveraging the benefits of Dynamic Programming (Q-learning). It utilises the Dynamic Programming results to relabel the return-to-go in the training data to then train the DT with the relabelled data. Our approach efficiently exploits the benefits of these two approaches and compensates for each other’s shortcomings to achieve better performance.

        ----

        ## [1625] Quantum Ridgelet Transform: Winning Lottery Ticket of Neural Networks with Quantum Computation

        **Authors**: *Hayata Yamasaki, Sathyawageeswar Subramanian, Satoshi Hayakawa, Sho Sonoda*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/yamasaki23a.html](https://proceedings.mlr.press/v202/yamasaki23a.html)

        **Abstract**:

        A significant challenge in the field of quantum machine learning (QML) is to establish applications of quantum computation to accelerate common tasks in machine learning such as those for neural networks. Ridgelet transform has been a fundamental mathematical tool in the theoretical studies of neural networks, but the practical applicability of ridgelet transform to conducting learning tasks was limited since its numerical implementation by conventional classical computation requires an exponential runtime $\exp(O(D))$ as data dimension $D$ increases. To address this problem, we develop a quantum ridgelet transform (QRT), which implements the ridgelet transform of a quantum state within a linear runtime $O(D)$ of quantum computation. As an application, we also show that one can use QRT as a fundamental subroutine for QML to efficiently find a sparse trainable subnetwork of large shallow wide neural networks without conducting large-scale optimization of the original network. This application discovers an efficient way in this regime to demonstrate the lottery ticket hypothesis on finding such a sparse trainable neural network. These results open an avenue of QML for accelerating learning tasks with commonly used classical neural networks.

        ----

        ## [1626] Compressed Decentralized Proximal Stochastic Gradient Method for Nonconvex Composite Problems with Heterogeneous Data

        **Authors**: *Yonggui Yan, Jie Chen, Pin-Yu Chen, Xiaodong Cui, Songtao Lu, Yangyang Xu*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/yan23a.html](https://proceedings.mlr.press/v202/yan23a.html)

        **Abstract**:

        We first propose a decentralized proximal stochastic gradient tracking method (DProxSGT) for nonconvex stochastic composite problems, with data heterogeneously distributed on multiple workers in a decentralized connected network. To save communication cost, we then extend DProxSGT to a compressed method by compressing the communicated information. Both methods need only $\mathcal{O}(1)$ samples per worker for each proximal update, which is important to achieve good generalization performance on training deep neural networks. With a smoothness condition on the expected loss function (but not on each sample function), the proposed methods can achieve an optimal sample complexity result to produce a near-stationary point. Numerical experiments on training neural networks demonstrate the significantly better generalization performance of our methods over large-batch training methods and momentum variance-reduction methods and also, the ability of handling heterogeneous data by the gradient tracking scheme.

        ----

        ## [1627] Temporally Consistent Transformers for Video Generation

        **Authors**: *Wilson Yan, Danijar Hafner, Stephen James, Pieter Abbeel*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/yan23b.html](https://proceedings.mlr.press/v202/yan23b.html)

        **Abstract**:

        To generate accurate videos, algorithms have to understand the spatial and temporal dependencies in the world. Current algorithms enable accurate predictions over short horizons but tend to suffer from temporal inconsistencies. When generated content goes out of view and is later revisited, the model invents different content instead. Despite this severe limitation, no established benchmarks exist for video generation with long temporal dependencies. In this paper, we curate 3 challenging video datasets with long-range dependencies by rendering walks through 3D scenes of procedural mazes, Minecraft worlds, and indoor scans. We perform a comprehensive evaluation of current models and observe their limitations in temporal consistency. Moreover, we introduce the Temporally Consistent Transformer (TECO), a generative model that substantially improves long-term consistency while also reducing sampling time. By compressing its input sequence into fewer embeddings, applying a temporal transformer, and expanding back using a spatial MaskGit, TECO outperforms existing models across many metrics. Videos are available on the website: https://wilson1yan.github.io/teco

        ----

        ## [1628] Distortion and Uncertainty Aware Loss for Panoramic Depth Completion

        **Authors**: *Zhiqiang Yan, Xiang Li, Kun Wang, Shuo Chen, Jun Li, Jian Yang*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/yan23c.html](https://proceedings.mlr.press/v202/yan23c.html)

        **Abstract**:

        Standard MSE or MAE loss function is commonly used in limited field-of-vision depth completion, treating each pixel equally under a basic assumption that all pixels have same contribution during optimization. Recently, with the rapid rise of panoramic photography, panoramic depth completion (PDC) has raised increasing attention in 3D computer vision. However, the assumption is inapplicable to panoramic data due to its latitude-wise distortion and high uncertainty nearby textures and edges. To handle these challenges, we propose distortion and uncertainty aware loss (DUL) that consists of a distortion-aware loss and an uncertainty-aware loss. The distortion-aware loss is designed to tackle the panoramic distortion caused by equirectangular projection, whose coordinate transformation relation is used to adaptively calculate the weight of the latitude-wise distortion, distributing uneven importance instead of the equal treatment for each pixel. The uncertainty-aware loss is presented to handle the inaccuracy in non-smooth regions. Specifically, we characterize uncertainty into PDC solutions under Bayesian deep learning framework, where a novel consistent uncertainty estimation constraint is designed to learn the consistency between multiple uncertainty maps of a single panorama. This consistency constraint allows model to produce more precise uncertainty estimation that is robust to feature deformation. Extensive experiments show the superiority of our method over standard loss functions, reaching the state of the art.

        ----

        ## [1629] Self-Interpretable Time Series Prediction with Counterfactual Explanations

        **Authors**: *Jingquan Yan, Hao Wang*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/yan23d.html](https://proceedings.mlr.press/v202/yan23d.html)

        **Abstract**:

        Interpretable time series prediction is crucial for safety-critical areas such as healthcare and autonomous driving. Most existing methods focus on interpreting predictions by assigning important scores to segments of time series. In this paper, we take a different and more challenging route and aim at developing a self-interpretable model, dubbed Counterfactual Time Series (CounTS), which generates counterfactual and actionable explanations for time series predictions. Specifically, we formalize the problem of time series counterfactual explanations, establish associated evaluation protocols, and propose a variational Bayesian deep learning model equipped with counterfactual inference capability of time series abduction, action, and prediction. Compared with state-of-the-art baselines, our self-interpretable model can generate better counterfactual explanations while maintaining comparable prediction accuracy.

        ----

        ## [1630] Quantum 3D Graph Learning with Applications to Molecule Embedding

        **Authors**: *Ge Yan, Huaijin Wu, Junchi Yan*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/yan23e.html](https://proceedings.mlr.press/v202/yan23e.html)

        **Abstract**:

        Learning 3D graph with spatial position as well as node attributes has been recently actively studied, for its utility in different applications e.g. 3D molecules. Quantum computing is known a promising direction for its potential theoretical supremacy for large-scale graph and combinatorial problem as well as the increasing evidence for the availability to physical quantum devices in the near term. In this paper, for the first time to our best knowledge, we propose a quantum 3D embedding ansatz that learns the latent representation of 3D structures from the Hilbert space composed of the Bloch sphere of each qubit. Specifically, the 3D Cartesian coordinates of nodes are converted into rotation and torsion angles and then encode them into the form of qubits. Moreover, Parameterized Quantum Circuit (PQC) is applied to serve as the trainable layers and the output of the PQC is adopted as the final node embedding. Experimental results on two downstream tasks, molecular property prediction and 3D molecular geometries generation, demonstrate the effectiveness of our model. We show the capacity and capability of our model with the evaluation on the QM9 dataset (134k molecules) with very few parameters, and its potential to be executed on a real quantum device.

        ----

        ## [1631] Fast Rates in Time-Varying Strongly Monotone Games

        **Authors**: *Yu-Hu Yan, Peng Zhao, Zhi-Hua Zhou*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/yan23f.html](https://proceedings.mlr.press/v202/yan23f.html)

        **Abstract**:

        Multi-player online games depict the interaction of multiple players with each other over time. Strongly monotone games are of particular interest since they have benign properties and also relate to many classic games that have applications in real life. Existing works mainly focus on the time-invariant case with provable guarantees established. However, the research of the more general time-varying games in changing environments is underexplored and the best-known result cannot match the guarantees in the time-invariant case. In this work, we present a new decentralized online algorithm for time-varying strongly monotone games, which greatly improves existing results and obtains fast rates, matching the best time-invariant guarantee without knowing the environmental non-stationarity. Furthermore, to achieve faster rates, we generalize the RVU property with smoothness and establish a series of problem-dependent bounds that also match the best time-invariant one. To realize all those results, we make a comprehensive use of the techniques in non-stationary and universal online learning.

        ----

        ## [1632] Proper Scoring Rules for Survival Analysis

        **Authors**: *Hiroki Yanagisawa*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/yanagisawa23a.html](https://proceedings.mlr.press/v202/yanagisawa23a.html)

        **Abstract**:

        Survival analysis is the problem of estimating probability distributions for future event times, which can be seen as a problem in uncertainty quantification. Although there are fundamental theories on strictly proper scoring rules for uncertainty quantification, little is known about those for survival analysis. In this paper, we investigate extensions of four major strictly proper scoring rules for survival analysis and we prove that these extensions are proper under certain conditions, which arise from the discretization of the estimation of probability distributions. We also compare the estimation performances of these extended scoring rules by using real datasets, and the extensions of the logarithmic score and the Brier score performed the best.

        ----

        ## [1633] Behavior Contrastive Learning for Unsupervised Skill Discovery

        **Authors**: *Rushuai Yang, Chenjia Bai, Hongyi Guo, Siyuan Li, Bin Zhao, Zhen Wang, Peng Liu, Xuelong Li*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/yang23a.html](https://proceedings.mlr.press/v202/yang23a.html)

        **Abstract**:

        In reinforcement learning, unsupervised skill discovery aims to learn diverse skills without extrinsic rewards. Previous methods discover skills by maximizing the mutual information (MI) between states and skills. However, such an MI objective tends to learn simple and static skills and may hinder exploration. In this paper, we propose a novel unsupervised skill discovery method through contrastive learning among behaviors, which makes the agent produce similar behaviors for the same skill and diverse behaviors for different skills. Under mild assumptions, our objective maximizes the MI between different behaviors based on the same skill, which serves as an upper bound of the previous MI objective. Meanwhile, our method implicitly increases the state entropy to obtain better state coverage. We evaluate our method on challenging mazes and continuous control tasks. The results show that our method generates diverse and far-reaching skills, and also obtains competitive performance in downstream tasks compared to the state-of-the-art methods.

        ----

        ## [1634] Nested Elimination: A Simple Algorithm for Best-Item Identification From Choice-Based Feedback

        **Authors**: *Junwen Yang, Yifan Feng*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/yang23b.html](https://proceedings.mlr.press/v202/yang23b.html)

        **Abstract**:

        We study the problem of best-item identification from choice-based feedback. In this problem, a company sequentially and adaptively shows display sets to a population of customers and collects their choices. The objective is to identify the most preferred item with the least number of samples and at a high confidence level. We propose an elimination-based algorithm, namely Nested Elimination (NE), which is inspired by the nested structure implied by the information-theoretic lower bound. NE is simple in structure, easy to implement, and has a strong theoretical guarantee for sample complexity. Specifically, NE utilizes an innovative elimination criterion and circumvents the need to solve any complex combinatorial optimization problem. We provide an instance-specific and non-asymptotic bound on the expected sample complexity of NE. We also show NE achieves high-order worst-case asymptotic optimality. Finally, numerical experiments from both synthetic and real data corroborate our theoretical findings.

        ----

        ## [1635] Towards Better Graph Representation Learning with Parameterized Decomposition & Filtering

        **Authors**: *Mingqi Yang, Wenjie Feng, Yanming Shen, Bryan Hooi*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/yang23c.html](https://proceedings.mlr.press/v202/yang23c.html)

        **Abstract**:

        Proposing an effective and flexible matrix to represent a graph is a fundamental challenge that has been explored from multiple perspectives, e.g., filtering in Graph Fourier Transforms. In this work, we develop a novel and general framework which unifies many existing GNN models from the view of parameterized decomposition and filtering, and show how it helps to enhance the flexibility of GNNs while alleviating the smoothness and amplification issues of existing models. Essentially, we show that the extensively studied spectral graph convolutions with learnable polynomial filters are constrained variants of this formulation, and releasing these constraints enables our model to express the desired decomposition and filtering simultaneously. Based on this generalized framework, we develop models that are simple in implementation but achieve significant improvements and computational efficiency on a variety of graph learning tasks. Code is available at https://github.com/qslim/PDF.

        ----

        ## [1636] Weighted Flow Diffusion for Local Graph Clustering with Node Attributes: an Algorithm and Statistical Guarantees

        **Authors**: *Shenghao Yang, Kimon Fountoulakis*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/yang23d.html](https://proceedings.mlr.press/v202/yang23d.html)

        **Abstract**:

        Local graph clustering methods aim to detect small clusters in very large graphs without the need to process the whole graph. They are fundamental and scalable tools for a wide range of tasks such as local community detection, node ranking and node embedding. While prior work on local graph clustering mainly focuses on graphs without node attributes, modern real-world graph datasets typically come with node attributes that provide valuable additional information. We present a simple local graph clustering algorithm for graphs with node attributes, based on the idea of diffusing mass locally in the graph while accounting for both structural and attribute proximities. Using high-dimensional concentration results, we provide statistical guarantees on the performance of the algorithm for the recovery of a target cluster with a single seed node. We give conditions under which a target cluster generated from a fairly general contextual random graph model, which includes both the stochastic block model and the planted cluster model as special cases, can be fully recovered with bounded false positives. Empirically, we validate all theoretical claims using synthetic data, and we show that incorporating node attributes leads to superior local clustering performances using real-world graph datasets.

        ----

        ## [1637] Chemically Transferable Generative Backmapping of Coarse-Grained Proteins

        **Authors**: *Soojung Yang, Rafael Gómez-Bombarelli*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/yang23e.html](https://proceedings.mlr.press/v202/yang23e.html)

        **Abstract**:

        Coarse-graining (CG) accelerates molecular simulations of protein dynamics by simulating sets of atoms as singular beads. Backmapping is the opposite operation of bringing lost atomistic details back from the CG representation. While machine learning (ML) has produced accurate and efficient CG simulations of proteins, fast and reliable backmapping remains a challenge. Rule-based methods produce poor all-atom geometries, needing computationally costly refinement through additional simulations. Recently proposed ML approaches outperform traditional baselines but are not transferable between proteins and sometimes generate unphysical atom placements with steric clashes and implausible torsion angles. This work addresses both issues to build a fast, transferable, and reliable generative backmapping tool for CG protein representations. We achieve generalization and reliability through a combined set of innovations: representation based on internal coordinates; an equivariant encoder/prior; a custom loss function that helps ensure local structure, global structure, and physical constraints; and expert curation of high-quality out-of-equilibrium protein data for training. Our results pave the way for out-of-the-box backmapping of coarse-grained simulations for arbitrary proteins.

        ----

        ## [1638] Data Poisoning Attacks Against Multimodal Encoders

        **Authors**: *Ziqing Yang, Xinlei He, Zheng Li, Michael Backes, Mathias Humbert, Pascal Berrang, Yang Zhang*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/yang23f.html](https://proceedings.mlr.press/v202/yang23f.html)

        **Abstract**:

        Recently, the newly emerged multimodal models, which leverage both visual and linguistic modalities to train powerful encoders, have gained increasing attention. However, learning from a large-scale unlabeled dataset also exposes the model to the risk of potential poisoning attacks, whereby the adversary aims to perturb the model’s training data to trigger malicious behaviors in it. In contrast to previous work, only poisoning visual modality, in this work, we take the first step to studying poisoning attacks against multimodal models in both visual and linguistic modalities. Specially, we focus on answering two questions: (1) Is the linguistic modality also vulnerable to poisoning attacks? and (2) Which modality is most vulnerable? To answer the two questions, we propose three types of poisoning attacks against multimodal models. Extensive evaluations on different datasets and model architectures show that all three attacks can achieve significant attack performance while maintaining model utility in both visual and linguistic modalities. Furthermore, we observe that the poisoning effect differs between different modalities. To mitigate the attacks, we propose both pre-training and post-training defenses. We empirically show that both defenses can significantly reduce the attack performance while preserving the model’s utility. Our code is available at https://github.com/zqypku/mm_poison/.

        ----

        ## [1639] Towards Sustainable Learning: Coresets for Data-efficient Deep Learning

        **Authors**: *Yu Yang, Hao Kang, Baharan Mirzasoleiman*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/yang23g.html](https://proceedings.mlr.press/v202/yang23g.html)

        **Abstract**:

        To improve the efficiency and sustainability of learning deep models, we propose CREST, the first scalable framework with rigorous theoretical guarantees to identify the most valuable examples for training non-convex models, particularly deep networks. To guarantee convergence to a stationary point of a non-convex function, CREST models the non-convex loss as a series of quadratic functions and extracts a coreset for each quadratic sub-region. In addition, to ensure faster convergence of stochastic gradient methods such as (mini-batch) SGD, CREST iteratively extracts multiple mini-batch coresets from larger random subsets of training data, to ensure nearly-unbiased gradients with small variances. Finally, to further improve scalability and efficiency, CREST identifies and excludes the examples that are learned from the coreset selection pipeline. Our extensive experiments on several deep networks trained on vision and NLP datasets, including CIFAR-10, CIFAR-100, TinyImageNet, and SNLI, confirm that CREST speeds up training deep networks on very large datasets, by 1.7x to 2.5x with minimum loss in the performance. By analyzing the learning difficulty of the subsets selected by CREST, we show that deep models benefit the most by learning from subsets of increasing difficulty levels.

        ----

        ## [1640] Improving Adversarial Robustness by Putting More Regularizations on Less Robust Samples

        **Authors**: *Dongyoon Yang, Insung Kong, Yongdai Kim*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/yang23h.html](https://proceedings.mlr.press/v202/yang23h.html)

        **Abstract**:

        Adversarial training, which is to enhance robustness against adversarial attacks, has received much attention because it is easy to generate human-imperceptible perturbations of data to deceive a given deep neural network. In this paper, we propose a new adversarial training algorithm that is theoretically well motivated and empirically superior to other existing algorithms. A novel feature of the proposed algorithm is to apply more regularization to data vulnerable to adversarial attacks than other existing regularization algorithms do. Theoretically, we show that our algorithm can be understood as an algorithm of minimizing a newly derived upper bound of the robust risk. Numerical experiments illustrate that our proposed algorithm improves the generalization (accuracy on examples) and robustness (accuracy on adversarial attacks) simultaneously to achieve the state-of-the-art performance.

        ----

        ## [1641] Improving Adversarial Robustness of Deep Equilibrium Models with Explicit Regulations Along the Neural Dynamics

        **Authors**: *Zonghan Yang, Peng Li, Tianyu Pang, Yang Liu*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/yang23i.html](https://proceedings.mlr.press/v202/yang23i.html)

        **Abstract**:

        Deep equilibrium (DEQ) models replace the multiple-layer stacking of conventional deep networks with a fixed-point iteration of a single-layer transformation. Having been demonstrated to be competitive in a variety of real-world scenarios, the adversarial robustness of general DEQs becomes increasingly crucial for their reliable deployment. Existing works improve the robustness of general DEQ models with the widely-used adversarial training (AT) framework, but they fail to exploit the structural uniquenesses of DEQ models. To this end, we interpret DEQs through the lens of neural dynamics and find that AT under-regulates intermediate states. Besides, the intermediate states typically provide predictions with a high prediction entropy. Informed by the correlation between the entropy of dynamical systems and their stability properties, we propose reducing prediction entropy by progressively updating inputs along the neural dynamics. During AT, we also utilize random intermediate states to compute the loss function. Our methods regulate the neural dynamics of DEQ models in this manner. Extensive experiments demonstrate that our methods substantially increase the robustness of DEQ models and even outperform the strong deep network baselines.

        ----

        ## [1642] Mitigating Spurious Correlations in Multi-modal Models during Fine-tuning

        **Authors**: *Yu Yang, Besmira Nushi, Hamid Palangi, Baharan Mirzasoleiman*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/yang23j.html](https://proceedings.mlr.press/v202/yang23j.html)

        **Abstract**:

        Spurious correlations that degrade model generalization or lead the model to be right for the wrong reasons are one of the main robustness concerns for real-world deployments. However, mitigating these correlations during pre-training for large-scale models can be costly and impractical, particularly for those without access to high-performance computing resources. This paper proposes a novel approach to address spurious correlations during fine-tuning for a given domain of interest. With a focus on multi-modal models (e.g., CLIP), the proposed method leverages different modalities in these models to detect and explicitly set apart spurious attributes from the affected class, achieved through a multi-modal contrastive loss function that expresses spurious relationships through language. Our experimental results and in-depth visualizations on CLIP show that such an intervention can effectively i) improve the model’s accuracy when spurious attributes are not present, and ii) directs the model’s activation maps towards the actual class rather than the spurious attribute when present. In particular, on the Waterbirds dataset, our algorithm achieved a worst-group accuracy 23% higher than ERM on CLIP with a ResNet-50 backbone, and 32% higher on CLIP with a ViT backbone, while maintaining the same average accuracy as ERM.

        ----

        ## [1643] A theory of representation learning gives a deep generalisation of kernel methods

        **Authors**: *Adam X. Yang, Maxime Robeyns, Edward Milsom, Ben Anson, Nandi Schoots, Laurence Aitchison*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/yang23k.html](https://proceedings.mlr.press/v202/yang23k.html)

        **Abstract**:

        The successes of modern deep machine learning methods are founded on their ability to transform inputs across multiple layers to build good high-level representations. It is therefore critical to understand this process of representation learning. However, standard theoretical approaches (formally NNGPs) involving infinite width limits eliminate representation learning. We therefore develop a new infinite width limit, the Bayesian representation learning limit, that exhibits representation learning mirroring that in finite-width models, yet at the same time, retains some of the simplicity of standard infinite-width limits. In particular, we show that Deep Gaussian processes (DGPs) in the Bayesian representation learning limit have exactly multivariate Gaussian posteriors, and the posterior covariances can be obtained by optimizing an interpretable objective combining a log-likelihood to improve performance with a series of KL-divergences which keep the posteriors close to the prior. We confirm these results experimentally in wide but finite DGPs. Next, we introduce the possibility of using this limit and objective as a flexible, deep generalisation of kernel methods, that we call deep kernel machines (DKMs). Like most naive kernel methods, DKMs scale cubically in the number of datapoints. We therefore use methods from the Gaussian process inducing point literature to develop a sparse DKM that scales linearly in the number of datapoints. Finally, we extend these approaches to NNs (which have non-Gaussian posteriors) in the Appendices.

        ----

        ## [1644] Efficient Algorithms for Exact Graph Matching on Correlated Stochastic Block Models with Constant Correlation

        **Authors**: *Joonhyuk Yang, Dongpil Shin, Hye Won Chung*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/yang23l.html](https://proceedings.mlr.press/v202/yang23l.html)

        **Abstract**:

        We consider the problem of graph matching, or learning vertex correspondence, between two correlated stochastic block models (SBMs). The graph matching problem arises in various fields, including computer vision, natural language processing and bioinformatics, and in particular, matching graphs with inherent community structure has significance related to de-anonymization of correlated social networks. Compared to the correlated Erdos-Renyi (ER) model, where various efficient algorithms have been developed, among which a few algorithms have been proven to achieve the exact matching with constant edge correlation, no low-order polynomial algorithm has been known to achieve exact matching for the correlated SBMs with constant correlation. In this work, we propose an efficient algorithm for matching graphs with community structure, based on the comparison between partition trees rooted from each vertex, by extending the idea of Mao et al. (2021) to graphs with communities. The partition tree divides the large neighborhoods of each vertex into disjoint subsets using their edge statistics to different communities. Our algorithm is the first low-order polynomial-time algorithm achieving exact matching between two correlated SBMs with high probability in dense graphs.

        ----

        ## [1645] Are Neurons Actually Collapsed? On the Fine-Grained Structure in Neural Representations

        **Authors**: *Yongyi Yang, Jacob Steinhardt, Wei Hu*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/yang23m.html](https://proceedings.mlr.press/v202/yang23m.html)

        **Abstract**:

        Recent work has observed an intriguing "Neural Collapse” phenomenon in well-trained neural networks, where the last-layer representations of training samples with the same label collapse into each other. This appears to suggest that the last-layer representations are completely determined by the labels, and do not depend on the intrinsic structure of input distribution. We provide evidence that this is not a complete description, and that the apparent collapse hides important fine-grained structure in the representations. Specifically, even when representations apparently collapse, the small amount of remaining variation can still faithfully and accurately captures the intrinsic structure of input distribution. As an example, if we train on CIFAR-10 using only 5 coarse-grained labels (by combining two classes into one super-class) until convergence, we can reconstruct the original 10-class labels from the learned representations via unsupervised clustering. The reconstructed labels achieve 93% accuracy on the CIFAR-10 test set, nearly matching the normal CIFAR-10 accuracy for the same architecture. We also provide an initial theoretical result showing the fine-grained representation structure in a simplified synthetic setting. Our results show concretely how the structure of input data can play a significant role in determining the fine-grained structure of neural representations, going beyond what Neural Collapse predicts.

        ----

        ## [1646] Generative Adversarial Symmetry Discovery

        **Authors**: *Jianke Yang, Robin Walters, Nima Dehmamy, Rose Yu*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/yang23n.html](https://proceedings.mlr.press/v202/yang23n.html)

        **Abstract**:

        Despite the success of equivariant neural networks in scientific applications, they require knowing the symmetry group a priori. However, it may be difficult to know which symmetry to use as an inductive bias in practice. Enforcing the wrong symmetry could even hurt the performance. In this paper, we propose a framework, LieGAN, to automatically discover equivariances from a dataset using a paradigm akin to generative adversarial training. Specifically, a generator learns a group of transformations applied to the data, which preserve the original distribution and fool the discriminator. LieGAN represents symmetry as interpretable Lie algebra basis and can discover various symmetries such as the rotation group $\mathrm{SO}(n)$, restricted Lorentz group $\mathrm{SO}(1,3)^+$ in trajectory prediction and top-quark tagging tasks. The learned symmetry can also be readily used in several existing equivariant neural networks to improve accuracy and generalization in prediction.

        ----

        ## [1647] Boosting Offline Reinforcement Learning with Action Preference Query

        **Authors**: *Qisen Yang, Shenzhi Wang, Matthieu Gaetan Lin, Shiji Song, Gao Huang*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/yang23o.html](https://proceedings.mlr.press/v202/yang23o.html)

        **Abstract**:

        Training practical agents usually involve offline and online reinforcement learning (RL) to balance the policy’s performance and interaction costs. In particular, online fine-tuning has become a commonly used method to correct the erroneous estimates of out-of-distribution data learned in the offline training phase. However, even limited online interactions can be inaccessible or catastrophic for high-stake scenarios like healthcare and autonomous driving. In this work, we introduce an interaction-free training scheme dubbed Offline-with-Action-Preferences (OAP). The main insight is that, compared to online fine-tuning, querying the preferences between pre-collected and learned actions can be equally or even more helpful to the erroneous estimate problem. By adaptively encouraging or suppressing policy constraint according to action preferences, OAP could distinguish overestimation from beneficial policy improvement and thus attains a more accurate evaluation of unseen data. Theoretically, we prove a lower bound of the behavior policy’s performance improvement brought by OAP. Moreover, comprehensive experiments on the D4RL benchmark and state-of-the-art algorithms demonstrate that OAP yields higher (29% on average) scores, especially on challenging AntMaze tasks (98% higher).

        ----

        ## [1648] Towards Controlled Data Augmentations for Active Learning

        **Authors**: *Jianan Yang, Haobo Wang, Sai Wu, Gang Chen, Junbo Zhao*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/yang23p.html](https://proceedings.mlr.press/v202/yang23p.html)

        **Abstract**:

        The mission of active learning is to identify the most valuable data samples, thus attaining decent performance with much fewer samples. The data augmentation techniques seem straightforward yet promising to enhance active learning by extending the exploration of the input space, which helps locate more valuable samples. In this work, we thoroughly study the coupling of data augmentation and active learning, thereby proposing Controllable Augmentation ManiPulator for Active Learning. In contrast to the few prior works that touched on this line, CAMPAL emphasizes a purposeful, tighten, and better-controlled integration of data augmentation into active learning in three folds: (i)-carefully designed augmentation policies applied separately on labeled and unlabeled data pools; (ii)-controlled and quantifiably optimizable augmentation strengths; (iii)-full and flexible coverage for most (if not all) active learning schemes. Theories are proposed and associated with the development of key components in CAMPAL. Through extensive empirical experiments, we bring the performance of active learning methods to a new level: an absolute performance boost of 16.99% on CIFAR-10 and 12.25 on SVHN with 1,000 annotated samples. Codes are available at https://github.com/jnzju/CAMPAL.

        ----

        ## [1649] What is Essential for Unseen Goal Generalization of Offline Goal-conditioned RL?

        **Authors**: *Rui Yang, Lin Yong, Xiaoteng Ma, Hao Hu, Chongjie Zhang, Tong Zhang*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/yang23q.html](https://proceedings.mlr.press/v202/yang23q.html)

        **Abstract**:

        Offline goal-conditioned RL (GCRL) offers a way to train general-purpose agents from fully offline datasets. In addition to being conservative within the dataset, the generalization ability to achieve unseen goals is another fundamental challenge for offline GCRL. However, to the best of our knowledge, this problem has not been well studied yet. In this paper, we study out-of-distribution (OOD) generalization of offline GCRL both theoretically and empirically to identify factors that are important. In a number of experiments, we observe that weighted imitation learning enjoys better generalization than pessimism-based offline RL method. Based on this insight, we derive a theory for OOD generalization, which characterizes several important design choices. We then propose a new offline GCRL method, Generalizable Offline goAl-condiTioned RL (GOAT), by combining the findings from our theoretical and empirical studies. On a new benchmark containing 9 independent identically distributed (IID) tasks and 17 OOD tasks, GOAT outperforms current state-of-the-art methods by a large margin.

        ----

        ## [1650] Neural Prediction Errors enable Analogical Visual Reasoning in Human Standard Intelligence Tests

        **Authors**: *Lingxiao Yang, Hongzhi You, Zonglei Zhen, Dahui Wang, Xiaohong Wan, Xiaohua Xie, Ru-Yuan Zhang*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/yang23r.html](https://proceedings.mlr.press/v202/yang23r.html)

        **Abstract**:

        Deep neural networks have long been criticized for lacking the ability to perform analogical visual reasoning. Here, we propose a neural network model to solve Raven’s Progressive Matrices (RPM) - one of the standard intelligence tests in human psychology. Specifically, we design a reasoning block based on the well-known concept of prediction error (PE) in neuroscience. Our reasoning block uses convolution to extract abstract rules from high-level visual features of the 8 context images and generates the features of a predicted answer. PEs are then calculated between the predicted features and those of the 8 candidate answers, and are then passed to the next stage. We further integrate our novel reasoning blocks into a residual network and build a new Predictive Reasoning Network (PredRNet). Extensive experiments show that our proposed PredRNet achieves state-of-the-art average performance on several important RPM benchmarks. PredRNet also shows good generalization abilities in a variety of out-of-distribution scenarios and other visual reasoning tasks. Most importantly, our PredRNet forms low-dimensional representations of abstract rules and minimizes hierarchical prediction errors during model training, supporting the critical role of PE minimization in visual reasoning. Our work highlights the potential of using neuroscience theories to solve abstract visual reasoning problems in artificial intelligence. The code is available at https://github.com/ZjjConan/AVR-PredRNet.

        ----

        ## [1651] Change is Hard: A Closer Look at Subpopulation Shift

        **Authors**: *Yuzhe Yang, Haoran Zhang, Dina Katabi, Marzyeh Ghassemi*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/yang23s.html](https://proceedings.mlr.press/v202/yang23s.html)

        **Abstract**:

        Machine learning models often perform poorly on subgroups that are underrepresented in the training data. Yet, little is understood on the variation in mechanisms that cause subpopulation shifts, and how algorithms generalize across such diverse shifts at scale. In this work, we provide a fine-grained analysis of subpopulation shift. We first propose a unified framework that dissects and explains common shifts in subgroups. We then establish a comprehensive benchmark of 20 state-of-the-art algorithms evaluated on 12 real-world datasets in vision, language, and healthcare domains. With results obtained from training over 10,000 models, we reveal intriguing observations for future progress in this space. First, existing algorithms only improve subgroup robustness over certain types of shifts but not others. Moreover, while current algorithms rely on group-annotated validation data for model selection, we find that a simple selection criterion based on worst-class accuracy is surprisingly effective even without any group information. Finally, unlike existing works that solely aim to improve worst-group accuracy (WGA), we demonstrate the fundamental tradeoff between WGA and other important metrics, highlighting the need to carefully choose testing metrics. Code and data are available at: https://github.com/YyzHarry/SubpopBench.

        ----

        ## [1652] Continual Task Allocation in Meta-Policy Network via Sparse Prompting

        **Authors**: *Yijun Yang, Tianyi Zhou, Jing Jiang, Guodong Long, Yuhui Shi*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/yang23t.html](https://proceedings.mlr.press/v202/yang23t.html)

        **Abstract**:

        How to train a generalizable meta-policy by continually learning a sequence of tasks? It is a natural human skill yet challenging to achieve by current reinforcement learning: the agent is expected to quickly adapt to new tasks (plasticity) meanwhile retaining the common knowledge from previous tasks (stability). We address it by "Continual Task Allocation via Sparse Prompting (CoTASP)", which learns over-complete dictionaries to produce sparse masks as prompts extracting a sub-network for each task from a meta-policy network. CoTASP trains a policy for each task by optimizing the prompts and the sub-network weights alternatively. The dictionary is then updated to align the optimized prompts with tasks’ embedding, thereby capturing tasks’ semantic correlations. Hence, relevant tasks share more neurons in the meta-policy network due to similar prompts while cross-task interference causing forgetting is effectively restrained. Given a meta-policy and dictionaries trained on previous tasks, new task adaptation reduces to highly efficient sparse prompting and sub-network finetuning. In experiments, CoTASP achieves a promising plasticity-stability trade-off without storing or replaying any past tasks’ experiences. It outperforms existing continual and multi-task RL methods on all seen tasks, forgetting reduction, and generalization to unseen tasks.

        ----

        ## [1653] Hyperbolic Representation Learning: Revisiting and Advancing

        **Authors**: *Menglin Yang, Min Zhou, Rex Ying, Yankai Chen, Irwin King*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/yang23u.html](https://proceedings.mlr.press/v202/yang23u.html)

        **Abstract**:

        The non-Euclidean geometry of hyperbolic spaces has recently garnered considerable attention in the realm of representation learning. Current endeavors in hyperbolic representation largely presuppose that the underlying hierarchies can be automatically inferred and preserved through the adaptive optimization process. This assumption, however, is questionable and requires further validation. In this work, we first introduce a position-tracking mechanism to scrutinize existing prevalent hyperbolic models, revealing that the learned representations are sub-optimal and unsatisfactory. To address this, we propose a simple yet effective method, hyperbolic informed embedding (HIE), by incorporating cost-free hierarchical information deduced from the hyperbolic distance of the node to the origin (i.e., induced hyperbolic norm) to advance existing hyperbolic models. The proposed method HIE is both task-agnostic and model-agnostic, enabling its seamless integration with a broad spectrum of models and tasks. Extensive experiments across various models and different tasks demonstrate the versatility and adaptability of the proposed method. Remarkably, our method achieves a remarkable improvement of up to 21.4% compared to the competing baselines.

        ----

        ## [1654] Which is Better for Learning with Noisy Labels: The Semi-supervised Method or Modeling Label Noise?

        **Authors**: *Yu Yao, Mingming Gong, Yuxuan Du, Jun Yu, Bo Han, Kun Zhang, Tongliang Liu*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/yao23a.html](https://proceedings.mlr.press/v202/yao23a.html)

        **Abstract**:

        In real life, accurately annotating large-scale datasets is sometimes difficult. Datasets used for training deep learning models are likely to contain label noise. To make use of the dataset containing label noise, two typical methods have been proposed. One is to employ the semi-supervised method by exploiting labeled confident examples and unlabeled unconfident examples. The other one is to model label noise and design statistically consistent classifiers. A natural question remains unsolved: which one should be used for a specific real-world application? In this paper, we answer the question from the perspective of causal data generative process. Specifically, the performance of the semi-supervised based method depends heavily on the data generative process while the method modeling label-noise is not influenced by the generation process. For example, for a given dataset, if it has a causal generative structure that the features cause the label, the semi-supervised based method would not be helpful. When the causal structure is unknown, we provide an intuitive method to discover the causal structure for a given dataset containing label noise.

        ----

        ## [1655] How Bad is Top-K Recommendation under Competing Content Creators?

        **Authors**: *Fan Yao, Chuanhao Li, Denis Nekipelov, Hongning Wang, Haifeng Xu*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/yao23b.html](https://proceedings.mlr.press/v202/yao23b.html)

        **Abstract**:

        This study explores the impact of content creators’ competition on user welfare in recommendation platforms, as well as the long-term dynamics of relevance-driven recommendations. We establish a model of creator competition, under the setting where the platform uses a top-$K$ recommendation policy, user decisions are guided by the Random Utility model, and creators, in absence of explicit utility functions, employ arbitrary no-regret learning algorithms for strategy updates. We study the user welfare guarantee through the lens of Price of Anarchy and show that the fraction of user welfare loss due to creator competition is always upper bounded by a small constant depending on $K$ and randomness in user decisions; we also prove the tightness of this bound. Our result discloses an intrinsic merit of the relevance-driven recommendation policy, as long as users’ decisions involve randomness and the platform provides reasonably many alternatives to its users.

        ----

        ## [1656] MultiAdam: Parameter-wise Scale-invariant Optimizer for Multiscale Training of Physics-informed Neural Networks

        **Authors**: *Jiachen Yao, Chang Su, Zhongkai Hao, Songming Liu, Hang Su, Jun Zhu*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/yao23c.html](https://proceedings.mlr.press/v202/yao23c.html)

        **Abstract**:

        Physics-informed Neural Networks (PINNs) have recently achieved remarkable progress in solving Partial Differential Equations (PDEs) in various fields by minimizing a weighted sum of PDE loss and boundary loss. However, there are several critical challenges in the training of PINNs, including the lack of theoretical frameworks and the imbalance between PDE loss and boundary loss. In this paper, we present an analysis of second-order non-homogeneous PDEs, which are classified into three categories and applicable to various common problems. We also characterize the connections between the training loss and actual error, guaranteeing convergence under mild conditions. The theoretical analysis inspires us to further propose MultiAdam, a scale-invariant optimizer that leverages gradient momentum to parameter-wisely balance the loss terms. Extensive experiment results on multiple problems from different physical domains demonstrate that our MultiAdam solver can improve the predictive accuracy by 1-2 orders of magnitude compared with strong baselines.

        ----

        ## [1657] Policy Mirror Ascent for Efficient and Independent Learning in Mean Field Games

        **Authors**: *Batuhan Yardim, Semih Cayci, Matthieu Geist, Niao He*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/yardim23a.html](https://proceedings.mlr.press/v202/yardim23a.html)

        **Abstract**:

        Mean-field games have been used as a theoretical tool to obtain an approximate Nash equilibrium for symmetric and anonymous $N$-player games. However, limiting applicability, existing theoretical results assume variations of a “population generative model”, which allows arbitrary modifications of the population distribution by the learning algorithm. Moreover, learning algorithms typically work on abstract simulators with population instead of the $N$-player game. Instead, we show that $N$ agents running policy mirror ascent converge to the Nash equilibrium of the regularized game within $\widetilde{\mathcal{O}}(\varepsilon^{-2})$ samples from a single sample trajectory without a population generative model, up to a standard $\mathcal{O}(\frac{1}{\sqrt{N}})$ error due to the mean field. Taking a divergent approach from the literature, instead of working with the best-response map we first show that a policy mirror ascent map can be used to construct a contractive operator having the Nash equilibrium as its fixed point. We analyze single-path TD learning for $N$-agent games, proving sample complexity guarantees by only using a sample path from the $N$-agent simulator without a population generative model. Furthermore, we demonstrate that our methodology allows for independent learning by $N$ agents with finite sample guarantees.

        ----

        ## [1658] Retrieval-Augmented Multimodal Language Modeling

        **Authors**: *Michihiro Yasunaga, Armen Aghajanyan, Weijia Shi, Richard James, Jure Leskovec, Percy Liang, Mike Lewis, Luke Zettlemoyer, Wen-Tau Yih*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/yasunaga23a.html](https://proceedings.mlr.press/v202/yasunaga23a.html)

        **Abstract**:

        Recent multimodal models such as DALL-E and CM3 have achieved remarkable progress in text-to-image and image-to-text generation. However, these models store all their knowledge (e.g., the appearance of the Eiffel Tower) in the model parameters, requiring increasingly larger models and training data to capture more knowledge. To integrate knowledge in a more scalable and modular way, we propose a retrieval-augmented multimodal model, which enables a base multimodal model (generator) to refer to relevant text and images fetched by a retriever from external memory (e.g., documents on the web). Specifically, for the retriever, we use a pretrained CLIP, and for the generator, we train a CM3 Transformer on the LAION dataset. Our resulting model, named Retrieval-Augmented CM3 (RA-CM3), is the first multimodal model that can retrieve and generate both text and images. We show that RA-CM3 significantly outperforms baseline multimodal models such as DALL-E and CM3 on both image and caption generation tasks (12 FID and 17 CIDEr improvements on MS-COCO), while requiring much less compute for training ($<$30% of DALL-E). Moreover, we show that RA-CM3 exhibits novel capabilities such as faithful image generation and multimodal in-context learning (e.g., image generation from demonstrations).

        ----

        ## [1659] On the Power of Pre-training for Generalization in RL: Provable Benefits and Hardness

        **Authors**: *Haotian Ye, Xiaoyu Chen, Liwei Wang, Simon Shaolei Du*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/ye23a.html](https://proceedings.mlr.press/v202/ye23a.html)

        **Abstract**:

        Generalization in Reinforcement Learning (RL) aims to train an agent during training that generalizes to the target environment. In this work, we first point out that RL generalization is fundamentally different from the generalization in supervised learning, and fine-tuning on the target environment is necessary for good test performance. Therefore, we seek to answer the following question: how much can we expect pre-training over training environments to be helpful for efficient and effective fine-tuning? On one hand, we give a surprising result showing that asymptotically, the improvement from pre-training is at most a constant factor. On the other hand, we show that pre-training can be indeed helpful in the non-asymptotic regime by designing a policy collection-elimination (PCE) algorithm and proving a distribution-dependent regret bound that is independent of the state-action space. We hope our theoretical results can provide insight towards understanding pre-training and generalization in RL.

        ----

        ## [1660] Personalized Federated Learning with Inferred Collaboration Graphs

        **Authors**: *Rui Ye, Zhenyang Ni, Fangzhao Wu, Siheng Chen, Yanfeng Wang*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/ye23b.html](https://proceedings.mlr.press/v202/ye23b.html)

        **Abstract**:

        Personalized federated learning (FL) aims to collaboratively train a personalized model for each client. Previous methods do not adaptively determine who to collaborate at a fine-grained level, making them difficult to handle diverse data heterogeneity levels and those cases where malicious clients exist. To address this issue, our core idea is to learn a collaboration graph, which models the benefits from each pairwise collaboration and allocates appropriate collaboration strengths. Based on this, we propose a novel personalized FL algorithm, pFedGraph, which consists of two key modules: (1) inferring the collaboration graph based on pairwise model similarity and dataset size at server to promote fine-grained collaboration and (2) optimizing local model with the assistance of aggregated model at client to promote personalization. The advantage of pFedGraph is flexibly adaptive to diverse data heterogeneity levels and model poisoning attacks, as the proposed collaboration graph always pushes each client to collaborate more with similar and beneficial clients. Extensive experiments show that pFedGraph consistently outperforms the other $14$ baseline methods across various heterogeneity levels and multiple cases where malicious clients exist. Code will be available at https://github.com/MediaBrain-SJTU/pFedGraph.

        ----

        ## [1661] Compositional Exemplars for In-context Learning

        **Authors**: *Jiacheng Ye, Zhiyong Wu, Jiangtao Feng, Tao Yu, Lingpeng Kong*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/ye23c.html](https://proceedings.mlr.press/v202/ye23c.html)

        **Abstract**:

        Large pretrained language models (LMs) have shown impressive In-Context Learning (ICL) ability, where the model learns to do an unseen task simply by conditioning on a prompt consisting of input-output examples as demonstration, without any parameter updates. The performance of ICL is highly dominated by the quality of the selected in-context examples. However, previous selection methods are mostly based on simple heuristics, leading to sub-optimal performance. In this work, we systematically formulate in-context example selection as a subset selection problem, and optimize it in an end-to-end fashion. We propose CEIL (Compositional Exemplars for In-context Learning), which is instantiated by Determinantal Point Processes (DPPs) to model the interaction between the given input and in-context examples, and optimized through carefully-designed contrastive learning to obtain preference from LMs. We validate CEIL on 12 classification and generation datasets from 7 distinct NLP tasks, including sentiment analysis, phraphrase detection, natural language inference, commonsense reasoning, open-domain question answering, code generation and semantic parsing. Extensive experiments demonstrate the effectiveness, transferability, compositionality of CEIL, shedding new lights on in-context leaning. Our code is released at https://github.com/HKUNLP/icl-ceil.

        ----

        ## [1662] Corruption-Robust Algorithms with Uncertainty Weighting for Nonlinear Contextual Bandits and Markov Decision Processes

        **Authors**: *Chenlu Ye, Wei Xiong, Quanquan Gu, Tong Zhang*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/ye23d.html](https://proceedings.mlr.press/v202/ye23d.html)

        **Abstract**:

        Despite the significant interest and progress in reinforcement learning (RL) problems with adversarial corruption, current works are either confined to the linear setting or lead to an undesired $\tilde{\mathcal O}(\sqrt{T}\zeta)$ regret bound, where $T$ is the number of rounds and $\zeta$ is the total amount of corruption. In this paper, we consider contextual bandits with general function approximation and propose a computationally efficient algorithm to achieve a regret of $\tilde{\mathcal O}(\sqrt{T}+\zeta)$. The proposed algorithm relies on the recently developed uncertainty-weighted least-squares regression from linear contextual bandits (He et al., 2022) and a new weighted estimator of uncertainty for the general function class. In contrast to the existing analysis for the sum of uncertainty that is heavily based on the linear structure, we develop a novel technique to control the sum of weighted uncertainty, thus establishing the final regret bound. We then generalize our algorithm to the episodic MDP and first achieve an additive dependence on the corruption level $\zeta$ in the scenario of general function approximation. Notably, our algorithms achieve regret bounds that either nearly match the lower bound or improve the performance of existing methods for all the corruption levels in both known and unknown $\zeta$ cases.

        ----

        ## [1663] GNN&GBDT-Guided Fast Optimizing Framework for Large-scale Integer Programming

        **Authors**: *Huigen Ye, Hua Xu, Hongyan Wang, Chengming Wang, Yu Jiang*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/ye23e.html](https://proceedings.mlr.press/v202/ye23e.html)

        **Abstract**:

        The latest two-stage optimization framework based on graph neural network (GNN) and large neighborhood search (LNS) is the most popular framework in solving large-scale integer programs (IPs). However, the framework can not effectively use the embedding spatial information in GNN and still highly relies on large-scale solvers in LNS, resulting in the scale of IP being limited by the ability of the current solver and performance bottlenecks. To handle these issues, this paper presents a GNN&GBDT-guided fast optimizing framework for large-scale IPs that only uses a small-scale optimizer to solve large-scale IPs efficiently. Specifically, the proposed framework can be divided into three stages: Multi-task GNN Embedding to generate the embedding space, GBDT Prediction to effectively use the embedding spatial information, and Neighborhood Optimization to solve large-scale problems fast using the small-scale optimizer. Extensive experiments show that the proposed framework can solve IPs with millions of scales and surpass SCIP and Gurobi in the specified wall-clock time using only a small-scale optimizer with 30% of the problem size. It also shows that the proposed framework can save 99% of running time in achieving the same solution quality as SCIP, which verifies the effectiveness and efficiency of the proposed framework in solving large-scale IPs.

        ----

        ## [1664] FedDisco: Federated Learning with Discrepancy-Aware Collaboration

        **Authors**: *Rui Ye, Mingkai Xu, Jianyu Wang, Chenxin Xu, Siheng Chen, Yanfeng Wang*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/ye23f.html](https://proceedings.mlr.press/v202/ye23f.html)

        **Abstract**:

        This work considers the category distribution heterogeneity in federated learning. This issue is due to biased labeling preferences at multiple clients and is a typical setting of data heterogeneity. To alleviate this issue, most previous works consider either regularizing local models or fine-tuning the global model, while they ignore the adjustment of aggregation weights and simply assign weights based on the dataset size. However, based on our empirical observations and theoretical analysis, we find that the dataset size is not optimal and the discrepancy between local and global category distributions could be a beneficial and complementary indicator for determining aggregation weights. We thus propose a novel aggregation method, Federated Learning with Discrepancy-Aware Collaboration (FedDisco), whose aggregation weights not only involve both the dataset size and the discrepancy value, but also contribute to a tighter theoretical upper bound of the optimization error. FedDisco can promote utility and modularity in a communication- and computation-efficient way. Extensive experiments show that our FedDisco outperforms several state-of-the-art methods and can be easily incorporated with many existing methods to further enhance the performance. Our code will be available at https://github.com/MediaBrain-SJTU/FedDisco.

        ----

        ## [1665] Towards Quantum Machine Learning for Constrained Combinatorial Optimization: a Quantum QAP Solver

        **Authors**: *Xinyu Ye, Ge Yan, Junchi Yan*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/ye23g.html](https://proceedings.mlr.press/v202/ye23g.html)

        **Abstract**:

        Combinatorial optimization (CO) on the graph is a crucial but challenging research topic. Recent quantum algorithms provide a new perspective for solving CO problems and have the potential to demonstrate quantum advantage. Quantum Approximate Optimization Algorithm (QAOA) is a well-known quantum heuristic for CO constructed by a parametric quantum circuit. However, QAOA is originally designed for unconstrained problems and the circuit parameters and solutions are jointly solved with time-consuming iterations. In this paper, we propose a novel quantum neural network (QNN) for learning CO problems in a supervised manner to achieve better and faster results. We focus on the Quadratic Assignment Problem (QAP) with matching constraints and the node permutation invariance property. To this end, a quantum neural network called QAP-QNN is devised to translate the QAP into a constrained vertex classification task. Moreover, we study two QAP tasks: Graph Matching and Traveling Salesman Problem on TorchQauntum simulators, and empirically show the effectiveness of our approach.

        ----

        ## [1666] Temporal Label Smoothing for Early Event Prediction

        **Authors**: *Hugo Yèche, Alizée Pace, Gunnar Rätsch, Rita Kuznetsova*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/yeche23a.html](https://proceedings.mlr.press/v202/yeche23a.html)

        **Abstract**:

        Models that can predict the occurrence of events ahead of time with low false-alarm rates are critical to the acceptance of decision support systems in the medical community. This challenging task is typically treated as a simple binary classification, ignoring temporal dependencies between samples, whereas we propose to exploit this structure. We first introduce a common theoretical framework unifying dynamic survival analysis and early event prediction. Following an analysis of objectives from both fields, we propose Temporal Label Smoothing (TLS), a simpler, yet best-performing method that preserves prediction monotonicity over time. By focusing the objective on areas with a stronger predictive signal, TLS improves performance over all baselines on two large-scale benchmark tasks. Gains are particularly notable along clinically relevant measures, such as event recall at low false-alarm rates. TLS reduces the number of missed events by up to a factor of two over previously used approaches in early event prediction.

        ----

        ## [1667] From Temporal to Contemporaneous Iterative Causal Discovery in the Presence of Latent Confounders

        **Authors**: *Raanan Y. Yehezkel Rohekar, Shami Nisimov, Yaniv Gurwicz, Gal Novik*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/yehezkel-rohekar23a.html](https://proceedings.mlr.press/v202/yehezkel-rohekar23a.html)

        **Abstract**:

        We present a constraint-based algorithm for learning causal structures from observational time-series data, in the presence of latent confounders. We assume a discrete-time, stationary structural vector autoregressive process, with both temporal and contemporaneous causal relations. One may ask if temporal and contemporaneous relations should be treated differently. The presented algorithm gradually refines a causal graph by learning long-term temporal relations before short-term ones, where contemporaneous relations are learned last. This ordering of causal relations to be learnt leads to a reduction in the required number of statistical tests. We validate this reduction empirically and demonstrate that it leads to higher accuracy for synthetic data and more plausible causal graphs for real-world data compared to state-of-the-art algorithms.

        ----

        ## [1668] Doubly Adversarial Federated Bandits

        **Authors**: *Jialin Yi, Milan Vojnovic*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/yi23a.html](https://proceedings.mlr.press/v202/yi23a.html)

        **Abstract**:

        We study a new non-stochastic federated multiarmed bandit problem with multiple agents collaborating via a communication network. The losses of the arms are assigned by an oblivious adversary that specifies the loss of each arm not only for each time step but also for each agent, which we call doubly adversarial. In this setting, different agents may choose the same arm in the same time step but observe different feedback. The goal of each agent is to find a globally best arm in hindsight that has the lowest cumulative loss averaged over all agents, which necessities the communication among agents. We provide regret lower bounds for any federated bandit algorithm under different settings, when agents have access to full-information feedback, or the bandit feedback. For the bandit feedback setting, we propose a near-optimal federated bandit algorithm called FEDEXP3. Our algorithm gives a positive answer to an open question proposed in (Cesa-Bianchi et al., 2016): FEDEXP3 can guarantee a sub-linear regret without exchanging sequences of selected arm identities or loss sequences among agents. We also provide numerical evaluations of our algorithm to validate our theoretical results and demonstrate its effectiveness on synthetic and real-world datasets.

        ----

        ## [1669] Online Prototype Alignment for Few-shot Policy Transfer

        **Authors**: *Qi Yi, Rui Zhang, Shaohui Peng, Jiaming Guo, Yunkai Gao, Kaizhao Yuan, Ruizhi Chen, Siming Lan, Xing Hu, Zidong Du, Xishan Zhang, Qi Guo, Yunji Chen*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/yi23b.html](https://proceedings.mlr.press/v202/yi23b.html)

        **Abstract**:

        Domain adaptation in RL mainly deals with the changes of observation when transferring the policy to a new environment. Many traditional approaches of domain adaptation in RL manage to learn a mapping function between the source and target domain in explicit or implicit ways. However, they typically require access to abundant data from the target domain. Besides, they often rely on visual clues to learn the mapping function and may fail when the source domain looks quite different from the target domain. To address these problems, in this paper, we propose a novel framework Online Prototype Alignment (OPA) to learn the mapping function based on the functional similarity of elements and is able to achieve few-shot policy transfer within only several episodes. The key insight of OPA is to introduce an exploration mechanism that can interact with the unseen elements of the target domain in an efficient and purposeful manner, and then connect them with the seen elements in the source domain according to their functionalities (instead of visual clues). Experimental results show that when the target domain looks visually different from the source domain, OPA can achieve better transfer performance even with much fewer samples from the target domain, outperforming prior methods.

        ----

        ## [1670] MonoFlow: Rethinking Divergence GANs via the Perspective of Wasserstein Gradient Flows

        **Authors**: *Mingxuan Yi, Zhanxing Zhu, Song Liu*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/yi23c.html](https://proceedings.mlr.press/v202/yi23c.html)

        **Abstract**:

        The conventional understanding of adversarial training in generative adversarial networks (GANs) is that the discriminator is trained to estimate a divergence, and the generator learns to minimize this divergence. We argue that despite the fact that many variants of GANs were developed following this paradigm, the current theoretical understanding of GANs and their practical algorithms are inconsistent. In this paper, we leverage Wasserstein gradient flows which characterize the evolution of particles in the sample space, to gain theoretical insights and algorithmic inspiration of GANs. We introduce a unified generative modeling framework – MonoFlow: the particle evolution is rescaled via a monotonically increasing mapping of the log density ratio. Under our framework, adversarial training can be viewed as a procedure first obtaining MonoFlow’s vector field via training the discriminator and the generator learns to draw the particle flow defined by the corresponding vector field. We also reveal the fundamental difference between variational divergence minimization and adversarial training. This analysis helps us to identify what types of generator loss functions can lead to the successful training of GANs and suggest that GANs may have more loss designs beyond the literature (e.g., non-saturated loss), as long as they realize MonoFlow. Consistent empirical studies are included to validate the effectiveness of our framework.

        ----

        ## [1671] SE(3) diffusion model with application to protein backbone generation

        **Authors**: *Jason Yim, Brian L. Trippe, Valentin De Bortoli, Emile Mathieu, Arnaud Doucet, Regina Barzilay, Tommi S. Jaakkola*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/yim23a.html](https://proceedings.mlr.press/v202/yim23a.html)

        **Abstract**:

        The design of novel protein structures remains a challenge in protein engineering for applications across biomedicine and chemistry. In this line of work, a diffusion model over rigid bodies in 3D (referred to as frames) has shown success in generating novel, functional protein backbones that have not been observed in nature. However, there exists no principled methodological framework for diffusion on SE(3), the space of orientation preserving rigid motions in R3, that operates on frames and confers the group invariance. We address these shortcomings by developing theoretical foundations of SE(3) invariant diffusion models on multiple frames followed by a novel framework, FrameDiff, for estimating the SE(3) equivariant score over multiple frames. We apply FrameDiff on monomer backbone generation and find it can generate designable monomers up to 500 amino acids without relying on a pretrained protein structure prediction network that has been integral to previous methods. We find our samples are capable of generalizing beyond any known protein structure.

        ----

        ## [1672] CoCo: A Coupled Contrastive Framework for Unsupervised Domain Adaptive Graph Classification

        **Authors**: *Nan Yin, Li Shen, Mengzhu Wang, Long Lan, Zeyu Ma, Chong Chen, Xian-Sheng Hua, Xiao Luo*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/yin23a.html](https://proceedings.mlr.press/v202/yin23a.html)

        **Abstract**:

        Although graph neural networks (GNNs) have achieved impressive achievements in graph classification, they often need abundant task-specific labels, which could be extensively costly to acquire. A credible solution is to explore additional labeled graphs to enhance unsupervised learning on the target domain. However, how to apply GNNs to domain adaptation remains unsolved owing to the insufficient exploration of graph topology and the significant domain discrepancy. In this paper, we propose Coupled Contrastive Graph Representation Learning (CoCo), which extracts the topological information from coupled learning branches and reduces the domain discrepancy with coupled contrastive learning. CoCo contains a graph convolutional network branch and a hierarchical graph kernel network branch, which explore graph topology in implicit and explicit manners. Besides, we incorporate coupled branches into a holistic multi-view contrastive learning framework, which not only incorporates graph representations learned from complementary views for enhanced understanding, but also encourages the similarity between cross-domain example pairs with the same semantics for domain alignment. Extensive experiments on popular datasets show that our CoCo outperforms these competing baselines in different settings generally.

        ----

        ## [1673] Adaptive Estimation of Graphical Models under Total Positivity

        **Authors**: *Jiaxi Ying, José Vinícius de Miranda Cardoso, Daniel P. Palomar*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/ying23a.html](https://proceedings.mlr.press/v202/ying23a.html)

        **Abstract**:

        We consider the problem of estimating (diagonally dominant) M-matrices as precision matrices in Gaussian graphical models. Such models have shown interesting properties, e.g., the maximum likelihood estimator exists with as little as two observations in the case of M-matrices, and exists even with one observation in the case of diagonally dominant M-matrices. We propose an adaptive multiple-stage estimation method, which refines the estimate by solving a weighted $\ell_1$-regularized problem in each stage. We further design a unified framework based on gradient projection method to solve the regularized problem, equipped with different projections to handle the constraints of M-matrices and diagonally dominant M-matrices. Theoretical analysis of the estimation error is established. The proposed method outperforms state-of-the-art methods in estimating precision matrices and identifying graph edges, as evidenced by synthetic and financial time-series data sets.

        ----

        ## [1674] Improving Visual Prompt Tuning for Self-supervised Vision Transformers

        **Authors**: *Seungryong Yoo, Eunji Kim, Dahuin Jung, Jungbeom Lee, Sungroh Yoon*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/yoo23a.html](https://proceedings.mlr.press/v202/yoo23a.html)

        **Abstract**:

        Visual Prompt Tuning (VPT) is an effective tuning method for adapting pretrained Vision Transformers (ViTs) to downstream tasks. It leverages extra learnable tokens, known as prompts, which steer the frozen pretrained ViTs. Although VPT has demonstrated its applicability with supervised vision transformers, it often underperforms with self-supervised ones. Through empirical observations, we deduce that the effectiveness of VPT hinges largely on the ViT blocks with which the prompt tokens interact. Specifically, VPT shows improved performance on image classification tasks for MAE and MoCo v3 when the prompt tokens are inserted into later blocks rather than the first block. These observations suggest that there exists an optimal location of blocks for the insertion of prompt tokens. Unfortunately, identifying the optimal blocks for prompts within each self-supervised ViT for diverse future scenarios is a costly process. To mitigate this problem, we propose a simple yet effective method that learns a gate for each ViT block to adjust its intervention into the prompt tokens. With our method, prompt tokens are selectively influenced by blocks that require steering for task adaptation. Our method outperforms VPT variants in FGVC and VTAB image classification and ADE20K semantic segmentation. The code is available at https://github.com/ryongithub/GatedPromptTuning.

        ----

        ## [1675] End-to-End Multi-Object Detection with a Regularized Mixture Model

        **Authors**: *Jaeyoung Yoo, Hojun Lee, Seunghyeon Seo, Inseop Chung, Nojun Kwak*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/yoo23b.html](https://proceedings.mlr.press/v202/yoo23b.html)

        **Abstract**:

        Recent end-to-end multi-object detectors simplify the inference pipeline by removing hand-crafted processes such as non-maximum suppression (NMS). However, during training, they still heavily rely on heuristics and hand-crafted processes which deteriorate the reliability of the predicted confidence score. In this paper, we propose a novel framework to train an end-to-end multi-object detector consisting of only two terms: negative log-likelihood (NLL) and a regularization term. In doing so, the multi-object detection problem is treated as density estimation of the ground truth bounding boxes utilizing a regularized mixture density model. The proposed end-to-end multi-object Detection with a Regularized Mixture Model (D-RMM) is trained by minimizing the NLL with the proposed regularization term, maximum component maximization (MCM) loss, preventing duplicate predictions. Our method reduces the heuristics of the training process and improves the reliability of the predicted confidence score. Moreover, our D-RMM outperforms the previous end-to-end detectors on MS COCO dataset. Code is available at https://github.com/lhj815/D-RMM.

        ----

        ## [1676] EM-Network: Oracle Guided Self-distillation for Sequence Learning

        **Authors**: *Ji Won Yoon, Sunghwan Ahn, Hyeonseung Lee, Minchan Kim, Seok Min Kim, Nam Soo Kim*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/yoon23a.html](https://proceedings.mlr.press/v202/yoon23a.html)

        **Abstract**:

        We introduce EM-Network, a novel self-distillation approach that effectively leverages target information for supervised sequence-to-sequence (seq2seq) learning. In contrast to conventional methods, it is trained with oracle guidance, which is derived from the target sequence. Since the oracle guidance compactly represents the target-side context that can assist the sequence model in solving the task, the EM-Network achieves a better prediction compared to using only the source input. To allow the sequence model to inherit the promising capability of the EM-Network, we propose a new self-distillation strategy, where the original sequence model can benefit from the knowledge of the EM-Network in a one-stage manner. We conduct comprehensive experiments on two types of seq2seq models: connectionist temporal classification (CTC) for speech recognition and attention-based encoder-decoder (AED) for machine translation. Experimental results demonstrate that the EM-Network significantly advances the current state-of-the-art approaches, improving over the best prior work on speech recognition and establishing state-of-the-art performance on WMT’14 and IWSLT’14.

        ----

        ## [1677] Continual Learners are Incremental Model Generalizers

        **Authors**: *Jaehong Yoon, Sung Ju Hwang, Yue Cao*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/yoon23b.html](https://proceedings.mlr.press/v202/yoon23b.html)

        **Abstract**:

        Motivated by the efficiency and rapid convergence of pre-trained models for solving downstream tasks, this paper extensively studies the impact of Continual Learning (CL) models as pre-trainers. We find that, in both supervised and unsupervised CL, the transfer quality of representations does not show a noticeable degradation of fine-tuning performance but rather increases gradually. This is because CL models can learn improved task-general features when easily forgetting task-specific knowledge. Based on this observation, we suggest a new unsupervised CL framework with masked modeling, which aims to capture fluent task-generic representation during training. Furthermore, we propose a new fine-tuning scheme, GLobal Attention Discretization (GLAD), that preserves rich task-generic representation during solving downstream tasks. The model fine-tuned with GLAD achieves competitive performance and can also be used as a good pre-trained model itself. We believe this paper breaks the barriers between pre-training and fine-tuning steps and leads to a sustainable learning framework in which the continual learner incrementally improves model generalization, yielding better transfer to unseen tasks.

        ----

        ## [1678] An Investigation into Pre-Training Object-Centric Representations for Reinforcement Learning

        **Authors**: *Jaesik Yoon, Yi-Fu Wu, Heechul Bae, Sungjin Ahn*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/yoon23c.html](https://proceedings.mlr.press/v202/yoon23c.html)

        **Abstract**:

        Unsupervised object-centric representation (OCR) learning has recently drawn attention as a new paradigm of visual representation. This is because of its potential of being an effective pre-training technique for various downstream tasks in terms of sample efficiency, systematic generalization, and reasoning. Although image-based reinforcement learning (RL) is one of the most important and thus frequently mentioned such downstream tasks, the benefit in RL has surprisingly not been investigated systematically thus far. Instead, most of the evaluations have focused on rather indirect metrics such as segmentation quality and object property prediction accuracy. In this paper, we investigate the effectiveness of OCR pre-training for image-based reinforcement learning via empirical experiments. For systematic evaluation, we introduce a simple object-centric visual RL benchmark and conduct experiments to answer questions such as "Does OCR pre-training improve performance on object-centric tasks?" and "Can OCR pre-training help with out-of-distribution generalization?". Our results provide empirical evidence for valuable insights into the effectiveness of OCR pre-training for RL and the potential limitations of its use in certain scenarios. Additionally, this study also examines the critical aspects of incorporating OCR pre-training in RL, including performance in a visually complex environment and the appropriate pooling layer to aggregate the object representations.

        ----

        ## [1679] Graph Generative Model for Benchmarking Graph Neural Networks

        **Authors**: *Minji Yoon, Yue Wu, John Palowitch, Bryan Perozzi, Russ Salakhutdinov*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/yoon23d.html](https://proceedings.mlr.press/v202/yoon23d.html)

        **Abstract**:

        As the field of Graph Neural Networks (GNN) continues to grow, it experiences a corresponding increase in the need for large, real-world datasets to train and test new GNN models on challenging, realistic problems. Unfortunately, such graph datasets are often generated from online, highly privacy-restricted ecosystems, which makes research and development on these datasets hard, if not impossible. This greatly reduces the amount of benchmark graphs available to researchers, causing the field to rely only on a handful of publicly-available datasets. To address this problem, we introduce a novel graph generative model, Computation Graph Transformer (CGT) that learns and reproduces the distribution of real-world graphs in a privacy-controlled way. More specifically, CGT (1) generates effective benchmark graphs on which GNNs show similar task performance as on the source graphs, (2) scales to process large-scale graphs, (3) incorporates off-the-shelf privacy modules to guarantee end-user privacy of the generated graph. Extensive experiments across a vast body of graph generative models show that only our model can successfully generate privacy-controlled, synthetic substitutes of large-scale real-world graphs that can be effectively used to benchmark GNN models.

        ----

        ## [1680] Analyzing Convergence in Quantum Neural Networks: Deviations from Neural Tangent Kernels

        **Authors**: *Xuchen You, Shouvanik Chakrabarti, Boyang Chen, Xiaodi Wu*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/you23a.html](https://proceedings.mlr.press/v202/you23a.html)

        **Abstract**:

        A quantum neural network (QNN) is a parameterized mapping efficiently implementable on near-term Noisy Intermediate-Scale Quantum (NISQ) computers. It can be used for supervised learning when combined with classical gradient-based optimizers. Despite the existing empirical and theoretical investigations, the convergence of QNN training is not fully understood. Inspired by the success of the neural tangent kernels (NTKs) in probing into the dynamics of classical neural networks, a recent line of works proposes to study over-parameterized QNNs by examining a quantum version of tangent kernels. In this work, we study the dynamics of QNNs and show that contrary to popular belief it is qualitatively different from that of any kernel regression: due to the unitarity of quantum operations, there is a non-negligible deviation from the tangent kernel regression derived at the random initialization. As a result of the deviation, we prove the at-most sublinear convergence for QNNs with Pauli measurements, which is beyond the explanatory power of any kernel regression dynamics. We then present the actual dynamics of QNNs in the limit of over-parameterization. The new dynamics capture the change of convergence rate during training and implies that the range of measurements is crucial to the fast QNN convergence.

        ----

        ## [1681] Entropy-driven Unsupervised Keypoint Representation Learning in Videos

        **Authors**: *Ali Younes, Simone Schaub-Meyer, Georgia Chalvatzaki*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/younes23a.html](https://proceedings.mlr.press/v202/younes23a.html)

        **Abstract**:

        Extracting informative representations from videos is fundamental for effectively learning various downstream tasks. We present a novel approach for unsupervised learning of meaningful representations from videos, leveraging the concept of image spatial entropy (ISE) that quantifies the per-pixel information in an image. We argue that local entropy of pixel neighborhoods and their temporal evolution create valuable intrinsic supervisory signals for learning prominent features. Building on this idea, we abstract visual features into a concise representation of keypoints that act as dynamic information transmitters, and design a deep learning model that learns, purely unsupervised, spatially and temporally consistent representations directly from video frames. Two original information-theoretic losses, computed from local entropy, guide our model to discover consistent keypoint representations; a loss that maximizes the spatial information covered by the keypoints and a loss that optimizes the keypoints’ information transportation over time. We compare our keypoint representation to strong baselines for various downstream tasks, e.g., learning object dynamics. Our empirical results show superior performance for our information-driven keypoints that resolve challenges like attendance to static and dynamic objects or objects abruptly entering and leaving the scene.

        ----

        ## [1682] The Benefits of Model-Based Generalization in Reinforcement Learning

        **Authors**: *Kenny John Young, Aditya Ramesh, Louis Kirsch, Jürgen Schmidhuber*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/young23a.html](https://proceedings.mlr.press/v202/young23a.html)

        **Abstract**:

        Model-Based Reinforcement Learning (RL) is widely believed to have the potential to improve sample efficiency by allowing an agent to synthesize large amounts of imagined experience. Experience Replay (ER) can be considered a simple kind of model, which has proved effective at improving the stability and efficiency of deep RL. In principle, a learned parametric model could improve on ER by generalizing from real experience to augment the dataset with additional plausible experience. However, given that learned value functions can also generalize, it is not immediately obvious why model generalization should be better. Here, we provide theoretical and empirical insight into when, and how, we can expect data generated by a learned model to be useful. First, we provide a simple theorem motivating how learning a model as an intermediate step can narrow down the set of possible value functions more than learning a value function directly from data using the Bellman equation. Second, we provide an illustrative example showing empirically how a similar effect occurs in a more concrete setting with neural network function approximation. Finally, we provide extensive experiments showing the benefit of model-based learning for online RL in environments with combinatorial complexity, but factored structure that allows a learned model to generalize. In these experiments, we take care to control for other factors in order to isolate, insofar as possible, the benefit of using experience generated by a learned model relative to ER alone.

        ----

        ## [1683] COLA: Orchestrating Error Coding and Learning for Robust Neural Network Inference Against Hardware Defects

        **Authors**: *Anlan Yu, Ning Lyu, Jieming Yin, Zhiyuan Yan, Wujie Wen*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/yu23a.html](https://proceedings.mlr.press/v202/yu23a.html)

        **Abstract**:

        Error correcting output codes (ECOCs) have been proposed to improve the robustness of deep neural networks (DNNs) against hardware defects of DNN hardware accelerators. Unfortunately, existing efforts suffer from drawbacks that would greatly impact their practicality: 1) robust accuracy (with defects) improvement at the cost of degraded clean accuracy (without defects); 2) no guarantee on better robust or clean accuracy using stronger ECOCs. In this paper, we first shed light on the connection between these drawbacks and error correlation, and then propose a novel comprehensive error decorrelation framework, namely COLA. Specifically, we propose to reduce inner layer feature error correlation by 1) adopting a separated architecture, where the last portions of the paths to all output nodes are separated, and 2) orthogonalizing weights in common DNN layers so that the intermediate features are orthogonal with each other. We also propose a regularization technique based on total correlation to mitigate overall error correlation at the outputs. The effectiveness of COLA is first analyzed theoretically, and then evaluated experimentally, e.g. up to 6.7% clean accuracy improvement compared with the original DNNs and up to 40% robust accuracy improvement compared to the state-of-the-art ECOC-enhanced DNNs.

        ----

        ## [1684] Delving into Noisy Label Detection with Clean Data

        **Authors**: *Chenglin Yu, Xinsong Ma, Weiwei Liu*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/yu23b.html](https://proceedings.mlr.press/v202/yu23b.html)

        **Abstract**:

        A critical element of learning with noisy labels is noisy label detection. Notably, numerous previous works assume that no source of labels can be clean in a noisy label detection context. In this work, we relax this assumption and assume that a small subset of the training data is clean, which enables substantial noisy label detection performance gains. Specifically, we propose a novel framework that leverages clean data by framing the problem of noisy label detection with clean data as a multiple hypothesis testing problem. Moreover, we propose BHN, a simple yet effective approach for noisy label detection that integrates the Benjamini-Hochberg (BH) procedure into deep neural networks. BHN achieves $\textit{state-of-the-art}$ performance and outperforms baselines by $\textbf{28.48}$% in terms of false discovery rate (FDR) and by $\textbf{18.99}$% in terms of F1 on CIFAR-10. Extensive ablation studies further demonstrate the superiority of BHN. Our code is available at https://github.com/ChenglinYu/BHN.

        ----

        ## [1685] Bag of Tricks for Training Data Extraction from Language Models

        **Authors**: *Weichen Yu, Tianyu Pang, Qian Liu, Chao Du, Bingyi Kang, Yan Huang, Min Lin, Shuicheng Yan*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/yu23c.html](https://proceedings.mlr.press/v202/yu23c.html)

        **Abstract**:

        With the advance of language models, privacy protection is receiving more attention. Training data extraction is therefore of great importance, as it can serve as a potential tool to assess privacy leakage. However, due to the difficulty of this task, most of the existing methods are proof-of-concept and still not effective enough. In this paper, we investigate and benchmark tricks for improving training data extraction using a publicly available dataset. Because most existing extraction methods use a pipeline of generating-then-ranking, i.e., generating text candidates as potential training data and then ranking them based on specific criteria, our research focuses on the tricks for both text generation (e.g., sampling strategy) and text ranking (e.g., token-level criteria). The experimental results show that several previously overlooked tricks can be crucial to the success of training data extraction. Based on the GPT-Neo 1.3B evaluation results, our proposed tricks outperform the baseline by a large margin in most cases, providing a much stronger baseline for future research. The code is available at https://github.com/weichen-yu/LM-Extraction.

        ----

        ## [1686] Discover-Then-Rank Unlabeled Support Vectors in the Dual Space for Multi-Class Active Learning

        **Authors**: *Dayou Yu, Weishi Shi, Qi Yu*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/yu23d.html](https://proceedings.mlr.press/v202/yu23d.html)

        **Abstract**:

        We propose to approach active learning (AL) from a novel perspective of discovering and then ranking potential support vectors by leveraging the key properties of the dual space of a sparse kernel max-margin predictor. We theoretically analyze the change of a hinge loss in the dual form and provide both the upper and lower bounds that are deeply connected to the key geometric properties induced by the dual space, which then help us identify various types of important data samples for AL. These bounds inform the design of a novel sampling strategy that leverages class-wise evidence as a key vehicle, formed through an affine combination of dual variables and kernel evaluation. We construct two distinct types of sampling functions, including discovery and ranking. The former focuses on samples with low total evidence from all classes, which signifies their potential to support exploration; the latter exploits the current decision boundary to identify the most conflicting regions for sampling, aiming to further refine the decision boundary. These two functions, which are complementary to each other, are automatically arranged into a two-phase active sampling process that starts with the discovery and then transitions to the ranking of data points to most effectively balance exploration and exploitation. Experiments on various real-world data demonstrate the state-of-the-art AL performance achieved by our model.

        ----

        ## [1687] Long-Term Rhythmic Video Soundtracker

        **Authors**: *Jiashuo Yu, Yaohui Wang, Xinyuan Chen, Xiao Sun, Yu Qiao*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/yu23e.html](https://proceedings.mlr.press/v202/yu23e.html)

        **Abstract**:

        We consider the problem of generating musical soundtracks in sync with rhythmic visual cues. Most existing works rely on pre-defined music representations, leading to the incompetence of generative flexibility and complexity. Other methods directly generating video-conditioned waveforms suffer from limited scenarios, short lengths, and unstable generation quality. To this end, we present Long-Term Rhythmic Video Soundtracker (LORIS), a novel framework to synthesize long-term conditional waveforms. Specifically, our framework consists of a latent conditional diffusion probabilistic model to perform waveform synthesis. Furthermore, a series of context-aware conditioning encoders are proposed to take temporal information into consideration for a long-term generation. Notably, we extend our model’s applicability from dances to multiple sports scenarios such as floor exercise and figure skating. To perform comprehensive evaluations, we establish a benchmark for rhythmic video soundtracks including the pre-processed dataset, improved evaluation metrics, and robust generative baselines. Extensive experiments show that our model generates long-term soundtracks with state-of-the-art musical quality and rhythmic correspondence. Codes are available at https://github.com/OpenGVLab/LORIS.

        ----

        ## [1688] Adversarial Parameter Attack on Deep Neural Networks

        **Authors**: *Lijia Yu, Yihan Wang, Xiao-Shan Gao*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/yu23f.html](https://proceedings.mlr.press/v202/yu23f.html)

        **Abstract**:

        The parameter perturbation attack is a safety threat to deep learning, where small parameter perturbations are made such that the attacked network gives wrong or desired labels of the adversary to specified inputs. However, such attacks could be detected by the user, because the accuracy of the attacked network will reduce and the network cannot work normally. To make the attack more stealthy, in this paper, the adversarial parameter attack is proposed, in which small perturbations to the parameters of the network are made such that the accuracy of the attacked network does not decrease much, but its robustness against adversarial example attacks becomes much lower. As a consequence, the attacked network performs normally on standard samples, but is much more vulnerable to adversarial attacks. The existence of nearly perfect adversarial parameters under $L_\infty$ norm and $L_0$ norm is proved under reasonable conditions. Algorithms are given which can be used to produce high quality adversarial parameters for the commonly used networks trained with various robust training methods, in that the robustness of the attacked networks decreases significantly when they are evaluated using various adversarial attack methods.

        ----

        ## [1689] CodeIPPrompt: Intellectual Property Infringement Assessment of Code Language Models

        **Authors**: *Zhiyuan Yu, Yuhao Wu, Ning Zhang, Chenguang Wang, Yevgeniy Vorobeychik, Chaowei Xiao*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/yu23g.html](https://proceedings.mlr.press/v202/yu23g.html)

        **Abstract**:

        Recent advances in large language models (LMs) have facilitated their ability to synthesize programming code. However, they have also raised concerns about intellectual property (IP) rights violations. Despite the significance of this issue, it has been relatively less explored. In this paper, we aim to bridge the gap by presenting CodeIPPrompt, a platform for automatic evaluation of the extent to which code language models may reproduce licensed programs. It comprises two key components: prompts constructed from a licensed code database to elicit LMs to generate IP-violating code, and a measurement tool to evaluate the extent of IP violation of code LMs. We conducted an extensive evaluation of existing open-source code LMs and commercial products and revealed the prevalence of IP violations in all these models. We further identified that the root cause is the substantial proportion of training corpus subject to restrictive licenses, resulting from both intentional inclusion and inconsistent license practice in the real world. To address this issue, we also explored potential mitigation strategies, including fine-tuning and dynamic token filtering. Our study provides a testbed for evaluating the IP violation issues of the existing code generation platforms and stresses the need for a better mitigation strategy.

        ----

        ## [1690] SeedGNN: Graph Neural Network for Supervised Seeded Graph Matching

        **Authors**: *Liren Yu, Jiaming Xu, Xiaojun Lin*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/yu23h.html](https://proceedings.mlr.press/v202/yu23h.html)

        **Abstract**:

        There is a growing interest in designing Graph Neural Networks (GNNs) for seeded graph matching, which aims to match two unlabeled graphs using only topological information and a small set of seed nodes. However, most previous GNNs for this task use a semi-supervised approach, which requires a large number of seeds and cannot learn knowledge that is transferable to unseen graphs. In contrast, this paper proposes a new supervised approach that can learn from a training set how to match unseen graphs with only a few seeds. Our SeedGNN architecture incorporates several novel designs, inspired by theoretical studies of seeded graph matching: 1) it can learn to compute and use witness-like information from different hops, in a way that can be generalized to graphs of different sizes; 2) it can use easily-matched node-pairs as new seeds to improve the matching in subsequent layers. We evaluate SeedGNN on synthetic and real-world graphs and demonstrate significant performance improvements over both non-learning and learning algorithms in the existing literature. Furthermore, our experiments confirm that the knowledge learned by SeedGNN from training graphs can be generalized to test graphs of different sizes and categories.

        ----

        ## [1691] Efficient and Equivariant Graph Networks for Predicting Quantum Hamiltonian

        **Authors**: *Haiyang Yu, Zhao Xu, Xiaofeng Qian, Xiaoning Qian, Shuiwang Ji*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/yu23i.html](https://proceedings.mlr.press/v202/yu23i.html)

        **Abstract**:

        We consider the prediction of the Hamiltonian matrix, which finds use in quantum chemistry and condensed matter physics. Efficiency and equivariance are two important, but conflicting factors. In this work, we propose a SE(3)-equivariant network, named QHNet, that achieves efficiency and equivariance. Our key advance lies at the innovative design of QHNet architecture, which not only obeys the underlying symmetries, but also enables the reduction of number of tensor products by 92%. In addition, QHNet prevents the exponential growth of channel dimension when more atom types are involved. We perform experiments on MD17 datasets, including four molecular systems. Experimental results show that our QHNet can achieve comparable performance to the state of the art methods at a significantly faster speed. Besides, our QHNet consumes 50% less memory due to its streamlined architecture. Our code is publicly available as part of the AIRS library (https://github.com/divelab/AIRS).

        ----

        ## [1692] On the Global Convergence of Risk-Averse Policy Gradient Methods with Expected Conditional Risk Measures

        **Authors**: *Xian Yu, Lei Ying*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/yu23j.html](https://proceedings.mlr.press/v202/yu23j.html)

        **Abstract**:

        Risk-sensitive reinforcement learning (RL) has become a popular tool to control the risk of uncertain outcomes and ensure reliable performance in various sequential decision-making problems. While policy gradient methods have been developed for risk-sensitive RL, it remains unclear if these methods enjoy the same global convergence guarantees as in the risk-neutral case. In this paper, we consider a class of dynamic time-consistent risk measures, called Expected Conditional Risk Measures (ECRMs), and derive policy gradient updates for ECRM-based objective functions. Under both constrained direct parameterization and unconstrained softmax parameterization, we provide global convergence and iteration complexities of the corresponding risk-averse policy gradient algorithms. We further test risk-averse variants of REINFORCE and actor-critic algorithms to demonstrate the efficacy of our method and the importance of risk control.

        ----

        ## [1693] Actor-Critic Alignment for Offline-to-Online Reinforcement Learning

        **Authors**: *Zishun Yu, Xinhua Zhang*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/yu23k.html](https://proceedings.mlr.press/v202/yu23k.html)

        **Abstract**:

        Deep offline reinforcement learning has recently demonstrated considerable promises in leveraging offline datasets, providing high-quality models that significantly reduce the online interactions required for fine-tuning. However, such a benefit is often diminished due to the marked state-action distribution shift, which causes significant bootstrap error and wipes out the good initial policy. Existing solutions resort to constraining the policy shift or balancing the sample replay based on their online-ness. However, they require online estimation of distribution divergence or density ratio. To avoid such complications, we propose deviating from existing actor-critic approaches that directly transfer the state-action value functions. Instead, we post-process them by aligning with the offline learned policy, so that the $Q$-values for actions outside the offline policy are also tamed. As a result, the online fine-tuning can be simply performed as in the standard actor-critic algorithms. We show empirically that the proposed method improves the performance of the fine-tuned robotic agents on various simulated tasks.

        ----

        ## [1694] Master-ASR: Achieving Multilingual Scalability and Low-Resource Adaptation in ASR with Modular Learning

        **Authors**: *Zhongzhi Yu, Yang Zhang, Kaizhi Qian, Cheng Wan, Yonggan Fu, Yongan Zhang, Yingyan Celine Lin*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/yu23l.html](https://proceedings.mlr.press/v202/yu23l.html)

        **Abstract**:

        Despite the impressive performance recently achieved by automatic speech recognition (ASR), we observe two primary challenges that hinder its broader applications: (1) The difficulty of introducing scalability into the model to support more languages with limited training, inference, and storage overhead; (2) The low-resource adaptation ability that enables effective low-resource adaptation while avoiding over fitting and catastrophic forgetting issues. Inspired by recent findings, we hypothesize that we can address the above challenges with modules widely shared across languages. To this end, we propose an ASR framework, dubbed Master-ASR, that, for the first time, simultaneously achieves strong multilingual scalability and low-resource adaptation ability thanks to its modularize-then-assemble strategy. Specifically, Master-ASR learns a small set of generalizable sub-modules and adaptively assembles them for different languages to reduce the multilingual overhead and enable effective knowledge transfer for low-resource adaptation. Extensive experiments and visualizations demonstrate that Master-ASR can effectively discover language similarity and improve multilingual and low-resource ASR performance over state-of-the-art (SOTA) methods, e.g., under multilingual-ASR, our framework achieves a 0.13∼2.41 lower character error rate (CER) with 30% smaller inference overhead over SOTA solutions on multilingual ASR and a comparable CER with nearly 100 times fewer trainable parameters over SOTA solutions on low-resource tuning, respectively.

        ----

        ## [1695] Coordinate Descent Methods for Fractional Minimization

        **Authors**: *Ganzhao Yuan*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/yuan23a.html](https://proceedings.mlr.press/v202/yuan23a.html)

        **Abstract**:

        We consider a class of structured fractional minimization problems, in which the numerator part of the objective is the sum of a differentiable convex function and a convex non-smooth function, while the denominator part is a convex or concave function. This problem is difficult to solve since it is non-convex. By exploiting the structure of the problem, we propose two Coordinate Descent (CD) methods for solving this problem. The proposed methods iteratively solve a one-dimensional subproblem globally, and they are guaranteed to converge to coordinate-wise stationary points. In the case of a convex denominator, under a weak locally bounded non-convexity condition, we prove that the optimality of coordinate-wise stationary point is stronger than that of the standard critical point and directional point. Under additional suitable conditions, CD methods converge Q-linearly to coordinate-wise stationary points. In the case of a concave denominator, we show that any critical point is a global minimum, and CD methods converge to the global minimum with a sublinear convergence rate. We demonstrate the applicability of the proposed methods to some machine learning and signal processing models. Our experiments on real-world data have shown that our method significantly and consistently outperforms existing methods in terms of accuracy.

        ----

        ## [1696] On the Power of Foundation Models

        **Authors**: *Yang Yuan*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/yuan23b.html](https://proceedings.mlr.press/v202/yuan23b.html)

        **Abstract**:

        With infinitely many high-quality data points, infinite computational power, an infinitely large foundation model with a perfect training algorithm and guaranteed zero generalization error on the pretext task, can the model be used for everything? This question cannot be answered by the existing theory of representation, optimization or generalization, because the issues they mainly investigate are assumed to be nonexistent here. In this paper, we show that category theory provides powerful machinery to answer this question. We have proved three results. The first one limits the power of prompt-based learning, saying that the model can solve a downstream task with prompts if and only if the task is representable. The second one says fine tuning does not have this limit, as a foundation model with the minimum required power (up to symmetry) can theoretically solve downstream tasks for the category defined by pretext task, with fine tuning and enough resources. Our final result can be seen as a new type of generalization theorem, showing that the foundation model can generate unseen objects from the target category (e.g., images) using the structural information from the source category (e.g., texts). Along the way, we provide a categorical framework for supervised and self-supervised learning, which might be of independent interest.

        ----

        ## [1697] Automatic Intrinsic Reward Shaping for Exploration in Deep Reinforcement Learning

        **Authors**: *Mingqi Yuan, Bo Li, Xin Jin, Wenjun Zeng*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/yuan23c.html](https://proceedings.mlr.press/v202/yuan23c.html)

        **Abstract**:

        We present AIRS: Automatic Intrinsic Reward Shaping that intelligently and adaptively provides high-quality intrinsic rewards to enhance exploration in reinforcement learning (RL). More specifically, AIRS selects shaping function from a predefined set based on the estimated task return in real-time, providing reliable exploration incentives and alleviating the biased objective problem. Moreover, we develop an intrinsic reward toolkit to provide efficient and reliable implementations of diverse intrinsic reward approaches. We test AIRS on various tasks of MiniGrid, Procgen, and DeepMind Control Suite. Extensive simulation demonstrates that AIRS can outperform the benchmarking schemes and achieve superior performance with simple architecture.

        ----

        ## [1698] Traversing Between Modes in Function Space for Fast Ensembling

        **Authors**: *Eunggu Yun, Hyungi Lee, Giung Nam, Juho Lee*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/yun23a.html](https://proceedings.mlr.press/v202/yun23a.html)

        **Abstract**:

        Deep ensemble is a simple yet powerful way to improve the performance of deep neural networks. Under this motivation, recent works on mode connectivity have shown that parameters of ensembles are connected by low-loss subspaces, and one can efficiently collect ensemble parameters in those subspaces. While this provides a way to efficiently train ensembles, for inference, multiple forward passes should still be executed using all the ensemble parameters, which often becomes a serious bottleneck for real-world deployment. In this work, we propose a novel framework to reduce such costs. Given a low-loss subspace connecting two modes of a neural network, we build an additional neural network that predicts the output of the original neural network evaluated at a certain point in the low-loss subspace. The additional neural network, which we call a “bridge”, is a lightweight network that takes minimal features from the original network and predicts outputs for the low-loss subspace without forward passes through the original network. We empirically demonstrate that we can indeed train such bridge networks and significantly reduce inference costs with the help of bridge networks.

        ----

        ## [1699] Conformal Prediction with Missing Values

        **Authors**: *Margaux Zaffran, Aymeric Dieuleveut, Julie Josse, Yaniv Romano*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/zaffran23a.html](https://proceedings.mlr.press/v202/zaffran23a.html)

        **Abstract**:

        Conformal prediction is a theoretically grounded framework for constructing predictive intervals. We study conformal prediction with missing values in the covariates – a setting that brings new challenges to uncertainty quantification. We first show that the marginal coverage guarantee of conformal prediction holds on imputed data for any missingness distribution and almost all imputation functions. However, we emphasize that the average coverage varies depending on the pattern of missing values: conformal methods tend to construct prediction intervals that under-cover the response conditionally to some missing patterns. This motivates our novel generalized conformalized quantile regression framework, missing data augmentation, which yields prediction intervals that are valid conditionally to the patterns of missing values, despite their exponential number. We then show that a universally consistent quantile regression algorithm trained on the imputed data is Bayes optimal for the pinball risk, thus achieving valid coverage conditionally to any given data point. Moreover, we examine the case of a linear model, which demonstrates the importance of our proposal in overcoming the heteroskedasticity induced by missing values. Using synthetic and data from critical care, we corroborate our theory and report improved performance of our methods.

        ----

        ## [1700] KDEformer: Accelerating Transformers via Kernel Density Estimation

        **Authors**: *Amir Zandieh, Insu Han, Majid Daliri, Amin Karbasi*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/zandieh23a.html](https://proceedings.mlr.press/v202/zandieh23a.html)

        **Abstract**:

        Dot-product attention mechanism plays a crucial role in modern deep architectures (e.g., Transformer) for sequence modeling, however, naïve exact computation of this model incurs quadratic time and memory complexities in sequence length, hindering the training of long-sequence models. Critical bottlenecks are due to the computation of partition functions in the denominator of softmax function as well as the multiplication of the softmax matrix with the matrix of values. Our key observation is that the former can be reduced to a variant of the kernel density estimation (KDE) problem, and an efficient KDE solver can be further utilized to accelerate the latter via subsampling-based fast matrix products. Our proposed KDEformer can approximate the attention in sub-quadratic time with provable spectral norm bounds, while all prior results merely provide entry-wise error bounds. Empirically, we verify that KDEformer outperforms other attention approximations in terms of accuracy, memory, and arithmetic operations on various pre-trained models. For instance, on BigGAN image generation we achieve better generative scores than the exact computation with over 4× speedup. For ImageNet classification with T2T-ViT, KDEformer shows over 18× speedup while the accuracy drop is less than 0.5%.

        ----

        ## [1701] Bayesian Estimation of Differential Privacy

        **Authors**: *Santiago Zanella Béguelin, Lukas Wutschitz, Shruti Tople, Ahmed Salem, Victor Rühle, Andrew Paverd, Mohammad Naseri, Boris Köpf, Daniel Jones*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/zanella-beguelin23a.html](https://proceedings.mlr.press/v202/zanella-beguelin23a.html)

        **Abstract**:

        Algorithms such as Differentially Private SGD enable training machine learning models with formal privacy guarantees. However, because these guarantees hold with respect to unrealistic adversaries, the protection afforded against practical attacks is typically much better. An emerging strand of work empirically estimates the protection afforded by differentially private training as a confidence interval for the privacy budget $\hat{\varepsilon}$ spent with respect to specific threat models. Existing approaches derive confidence intervals for $\hat{\varepsilon}$ from confidence intervals for false positive and false negative rates of membership inference attacks, which requires training an impractically large number of models to get intervals that can be acted upon. We propose a novel, more efficient Bayesian approach that brings privacy estimates within the reach of practitioners. Our approach reduces sample size by computing a posterior for $\hat{\varepsilon}$ (not just a confidence interval) from the joint posterior of the false positive and false negative rates of membership inference attacks. We implement an end-to-end system for privacy estimation that integrates our approach and state-of-the-art membership inference attacks, and evaluate it on text and vision classification tasks. For the same number of samples, we see a reduction in interval width of up to 40% compared to prior work.

        ----

        ## [1702] When is Realizability Sufficient for Off-Policy Reinforcement Learning?

        **Authors**: *Andrea Zanette*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/zanette23a.html](https://proceedings.mlr.press/v202/zanette23a.html)

        **Abstract**:

        Understanding when reinforcement learning algorithms can make successful off-policy predictions—and when the may fail to do so–remains an open problem. Typically, model-free algorithms for reinforcement learning are analyzed under a condition called Bellman completeness when they operate off-policy with function approximation, unless additional conditions are met. However, Bellman completeness is a requirement that is much stronger than realizability and that is deemed to be too strong to hold in practice. In this work, we relax this structural assumption and analyze the statistical complexity of off-policy reinforcement learning when only realizability holds for the prescribed function class. We establish finite-sample guarantees for off-policy reinforcement learning that are free of the approximation error term known as inherent Bellman error, and that depend on the interplay of three factors. The first two are well known: they are the metric entropy of the function class and the concentrability coefficient that represents the cost of learning off-policy. The third factor is new, and it measures the violation of Bellman completeness, namely the mis-alignment between the chosen function class and its image through the Bellman operator. Our analysis directly applies to the solution found by temporal difference algorithms when they converge.

        ----

        ## [1703] On Distribution Dependent Sub-Logarithmic Query Time of Learned Indexing

        **Authors**: *Sepanta Zeighami, Cyrus Shahabi*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/zeighami23a.html](https://proceedings.mlr.press/v202/zeighami23a.html)

        **Abstract**:

        A fundamental problem in data management is to find the elements in an array that match a query. Recently, learned indexes are being extensively used to solve this problem, where they learn a model to predict the location of the items in the array. They are empirically shown to outperform non-learned methods (e.g., B-trees or binary search that answer queries in $O(\log n)$ time) by orders of magnitude. However, success of learned indexes has not been theoretically justified. Only existing attempt shows the same query time of $O(\log n)$, but with a constant factor improvement in space complexity over non-learned methods, under some assumptions on data distribution. In this paper, we significantly strengthen this result, showing that under mild assumptions on data distribution, and the same space complexity as non-learned methods, learned indexes can answer queries in $O(\log\log n)$ expected query time. We also show that allowing for slightly larger but still near-linear space overhead, a learned index can achieve $O(1)$ expected query time. Our results theoretically prove learned indexes are orders of magnitude faster than non-learned methods, theoretically grounding their empirical success.

        ----

        ## [1704] Sequential Counterfactual Risk Minimization

        **Authors**: *Houssam Zenati, Eustache Diemert, Matthieu Martin, Julien Mairal, Pierre Gaillard*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/zenati23a.html](https://proceedings.mlr.press/v202/zenati23a.html)

        **Abstract**:

        Counterfactual Risk Minimization (CRM) is a framework for dealing with the logged bandit feedback problem, where the goal is to improve a logging policy using offline data. In this paper, we explore the case where it is possible to deploy learned policies multiple times and acquire new data. We extend the CRM principle and its theory to this scenario, which we call "Sequential Counterfactual Risk Minimization (SCRM)." We introduce a novel counterfactual estimator and identify conditions that can improve the performance of CRM in terms of excess risk and regret rates, by using an analysis similar to restart strategies in accelerated optimization methods. We also provide an empirical evaluation of our method in both discrete and continuous action settings, and demonstrate the benefits of multiple deployments of CRM.

        ----

        ## [1705] LookupFFN: Making Transformers Compute-lite for CPU inference

        **Authors**: *Zhanpeng Zeng, Michael Davies, Pranav Pulijala, Karthikeyan Sankaralingam, Vikas Singh*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/zeng23a.html](https://proceedings.mlr.press/v202/zeng23a.html)

        **Abstract**:

        While GPU clusters are the de facto choice for training large deep neural network (DNN) models today, several reasons including ease of workflow, security and cost have led to efforts investigating whether CPUs may be viable for inference in routine use in many sectors of the industry. But the imbalance between the compute capabilities of GPUs and CPUs is huge. Motivated by these considerations, we study a module which is a workhorse within modern DNN architectures, GEMM based Feed Forward Networks (FFNs), and assess the extent to which it can be made compute- (or FLOP-) lite. Specifically, we propose an alternative formulation (we call it LookupFFN) to GEMM based FFNs inspired by the recent studies of using Locality Sensitive Hashing (LSH) to approximate FFNs. Our formulation recasts most essential operations as a memory look-up, leveraging the trade-off between the two resources on any platform: compute and memory (since CPUs offer it in abundance). For RoBERTa language model pretraining, our formulation achieves similar performance compared to GEMM based FFNs, while dramatically reducing the required FLOP. Our development is complemented with a detailed hardware profiling of strategies that will maximize efficiency – not just on contemporary hardware but on products that will be offered in the near/medium term future. Code is avaiable at https://github.com/mlpen/LookupFFN.

        ----

        ## [1706] Attribute-Efficient PAC Learning of Low-Degree Polynomial Threshold Functions with Nasty Noise

        **Authors**: *Shiwei Zeng, Jie Shen*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/zeng23b.html](https://proceedings.mlr.press/v202/zeng23b.html)

        **Abstract**:

        The concept class of low-degree polynomial threshold functions (PTFs) plays a fundamental role in machine learning. In this paper, we study PAC learning of $K$-sparse degree-$d$ PTFs on $\mathbb{R}^n$, where any such concept depends only on $K$ out of $n$ attributes of the input. Our main contribution is a new algorithm that runs in time $({nd}/{\epsilon})^{O(d)}$ and under the Gaussian marginal distribution, PAC learns the class up to error rate $\epsilon$ with $O(\frac{K^{4d}}{\epsilon^{2d}} \cdot \log^{5d} n)$ samples even when an $\eta \leq O(\epsilon^d)$ fraction of them are corrupted by the nasty noise of Bshouty et al. (2002), possibly the strongest corruption model. Prior to this work, attribute-efficient robust algorithms are established only for the special case of sparse homogeneous halfspaces. Our key ingredients are: 1) a structural result that translates the attribute sparsity to a sparsity pattern of the Chow vector under the basis of Hermite polynomials, and 2) a novel attribute-efficient robust Chow vector estimation algorithm which uses exclusively a restricted Frobenius norm to either certify a good approximation or to validate a sparsity-induced degree-$2d$ polynomial as a filter to detect corrupted samples.

        ----

        ## [1707] Generative Graph Dictionary Learning

        **Authors**: *Zhichen Zeng, Ruike Zhu, Yinglong Xia, Hanqing Zeng, Hanghang Tong*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/zeng23c.html](https://proceedings.mlr.press/v202/zeng23c.html)

        **Abstract**:

        Dictionary learning, which approximates data samples by a set of shared atoms, is a fundamental task in representation learning. However, dictionary learning over graphs, namely graph dictionary learning (GDL), is much more challenging than vectorial data as graphs lie in disparate metric spaces. The sparse literature on GDL formulates the problem from the reconstructive view and often learns linear graph embeddings with a high computational cost. In this paper, we propose a Fused Gromov-Wasserstein (FGW) Mixture Model named FraMe to address the GDL problem from the generative view. Equipped with the graph generation function based on the radial basis function kernel and FGW distance, FraMe generates nonlinear embedding spaces, which, as we theoretically proved, provide a good approximation of the original graph spaces. A fast solution is further proposed on top of the expectation-maximization algorithm with guaranteed convergence. Extensive experiments demonstrate the effectiveness of the obtained node and graph embeddings, and our algorithm achieves significant improvements over the state-of-the-art methods.

        ----

        ## [1708] Stabilizing Transformer Training by Preventing Attention Entropy Collapse

        **Authors**: *Shuangfei Zhai, Tatiana Likhomanenko, Etai Littwin, Dan Busbridge, Jason Ramapuram, Yizhe Zhang, Jiatao Gu, Joshua M. Susskind*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/zhai23a.html](https://proceedings.mlr.press/v202/zhai23a.html)

        **Abstract**:

        Training stability is of great importance to Transformers. In this work, we investigate the training dynamics of Transformers by examining the evolution of the attention layers. In particular, we track the attention entropy for each attention head during the course of training, which is a proxy for model sharpness. We identify a common pattern across different architectures and tasks, where low attention entropy is accompanied by high training instability, which can take the form of oscillating loss or divergence. We denote the pathologically low attention entropy, corresponding to highly concentrated attention scores, as $\textit{entropy collapse}$. As a remedy, we propose $\sigma$Reparam, a simple and efficient solution where we reparametrize all linear layers with spectral normalization and an additional learned scalar. We demonstrate that $\sigma$Reparam successfully prevents entropy collapse in the attention layers, promoting more stable training. Additionally, we prove a tight lower bound of the attention entropy, which decreases exponentially fast with the spectral norm of the attention logits, providing additional motivation for our approach. We conduct experiments with $\sigma$Reparam on image classification, image self-supervised learning, machine translation, speech recognition, and language modeling tasks. We show that $\sigma$Reparam provides stability and robustness with respect to the choice of hyperparameters, going so far as enabling training (a) a Vision Transformer to competitive performance without warmup, weight decay, layer normalization or adaptive optimizers; (b) deep architectures in machine translation and (c) speech recognition to competitive performance without warmup and adaptive optimizers. Code is available at https://github.com/apple/ml-sigma-reparam.

        ----

        ## [1709] Offline Learning in Markov Games with General Function Approximation

        **Authors**: *Yuheng Zhang, Yu Bai, Nan Jiang*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/zhang23a.html](https://proceedings.mlr.press/v202/zhang23a.html)

        **Abstract**:

        We study offline multi-agent reinforcement learning (RL) in Markov games, where the goal is to learn an approximate equilibrium—such as Nash equilibrium and (Coarse) Correlated Equilibrium—from an offline dataset pre-collected from the game. Existing works consider relatively restricted tabular or linear models and handle each equilibria separately. In this work, we provide the first framework for sample-efficient offline learning in Markov games under general function approximation, handling all 3 equilibria in a unified manner. By using Bellman-consistent pessimism, we obtain interval estimation for policies’ returns, and use both the upper and the lower bounds to obtain a relaxation on the gap of a candidate policy, which becomes our optimization objective. Our results generalize prior works and provide several additional insights. Importantly, we require a data coverage condition that improves over the recently proposed “unilateral concentrability”. Our condition allows selective coverage of deviation policies that optimally trade-off between their greediness (as approximate best responses) and coverage, and we show scenarios where this leads to significantly better guarantees. As a new connection, we also show how our algorithmic framework can subsume seemingly different solution concepts designed for the special case of two-player zero-sum games.

        ----

        ## [1710] Learning useful representations for shifting tasks and distributions

        **Authors**: *Jianyu Zhang, Léon Bottou*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/zhang23b.html](https://proceedings.mlr.press/v202/zhang23b.html)

        **Abstract**:

        Does the dominant approach to learn representations (as a side effect of optimizing an expected cost for a single training distribution) remain a good approach when we are dealing with multiple distributions? Our thesis is that such scenarios are better served by representations that are richer than those obtained with a single optimization episode. We support this thesis with simple theoretical arguments and with experiments utilizing an apparently näive ensembling technique: concatenating the representations obtained from multiple training episodes using the same data, model, algorithm, and hyper-parameters, but different random seeds. These independently trained networks perform similarly. Yet, in a number of scenarios involving new distributions, the concatenated representation performs substantially better than an equivalently sized network trained with a single training run. This proves that the representations constructed by multiple training episodes are in fact different. Although their concatenation carries little additional information about the training task under the training distribution, it becomes substantially more informative when tasks or distributions change. Meanwhile, a single training episode is unlikely to yield such a redundant representation because the optimization process has no reason to accumulate features that do not incrementally improve the training performance.

        ----

        ## [1711] Nonparametric Iterative Machine Teaching

        **Authors**: *Chen Zhang, Xiaofeng Cao, Weiyang Liu, Ivor W. Tsang, James T. Kwok*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/zhang23c.html](https://proceedings.mlr.press/v202/zhang23c.html)

        **Abstract**:

        In this paper, we consider the problem of Iterative Machine Teaching (IMT), where the teacher provides examples to the learner iteratively such that the learner can achieve fast convergence to a target model. However, existing IMT algorithms are solely based on parameterized families of target models. They mainly focus on convergence in the parameter space, resulting in difficulty when the target models are defined to be functions without dependency on parameters. To address such a limitation, we study a more general task – Nonparametric Iterative Machine Teaching (NIMT), which aims to teach nonparametric target models to learners in an iterative fashion. Unlike parametric IMT that merely operates in the parameter space, we cast NIMT as a functional optimization problem in the function space. To solve it, we propose both random and greedy functional teaching algorithms. We obtain the iterative teaching dimension (ITD) of the random teaching algorithm under proper assumptions, which serves as a uniform upper bound of ITD in NIMT. Further, the greedy teaching algorithm has a significantly lower ITD, which reaches a tighter upper bound of ITD in NIMT. Finally, we verify the correctness of our theoretical findings with extensive experiments in nonparametric scenarios.

        ----

        ## [1712] Matrix Estimation for Individual Fairness

        **Authors**: *Cindy Y. Zhang, Sarah Huiyi Cen, Devavrat Shah*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/zhang23d.html](https://proceedings.mlr.press/v202/zhang23d.html)

        **Abstract**:

        In recent years, multiple notions of algorithmic fairness have arisen. One such notion is individual fairness (IF), which requires that individuals who are similar receive similar treatment. In parallel, matrix estimation (ME) has emerged as a natural paradigm for handling noisy data with missing values. In this work, we connect the two concepts. We show that pre-processing data using ME can improve an algorithm’s IF without sacrificing performance. Specifically, we show that using a popular ME method known as singular value thresholding (SVT) to pre-process the data provides a strong IF guarantee under appropriate conditions. We then show that, under analogous conditions, SVT pre-processing also yields estimates that are consistent and approximately minimax optimal. As such, the ME pre-processing step does not, under the stated conditions, increase the prediction error of the base algorithm, i.e., does not impose a fairness-performance trade-off. We verify these results on synthetic and real data.

        ----

        ## [1713] Graph Contrastive Backdoor Attacks

        **Authors**: *Hangfan Zhang, Jinghui Chen, Lu Lin, Jinyuan Jia, Dinghao Wu*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/zhang23e.html](https://proceedings.mlr.press/v202/zhang23e.html)

        **Abstract**:

        Graph Contrastive Learning (GCL) has attracted considerable interest due to its impressive node representation learning capability. Despite the wide application of GCL techniques, little attention has been paid to the security of GCL. In this paper, we systematically study the vulnerability of GCL in the presence of malicious backdoor adversaries. In particular, we propose GCBA, the first backdoor attack for graph contrastive learning. GCBA incorporates three attacks: poisoning, crafting, and natural backdoor, each targeting one stage of the GCL pipeline. We formulate our attacks as optimization problems and solve them with a novel discrete optimization technique to overcome the discrete nature of graph-structured data. By extensively evaluating GCBA on multiple datasets and GCL methods, we show that our attack can achieve high attack success rates while preserving stealthiness. We further consider potential countermeasures to our attack and conclude that existing defenses are insufficient to mitigate GCBA. We show that as a complex paradigm involving data and model republishing, GCL is vulnerable to backdoor attacks, and specifically designed defenses are needed to mitigate the backdoor attacks on GCL.

        ----

        ## [1714] Effective Minkowski Dimension of Deep Nonparametric Regression: Function Approximation and Statistical Theories

        **Authors**: *Zixuan Zhang, Minshuo Chen, Mengdi Wang, Wenjing Liao, Tuo Zhao*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/zhang23f.html](https://proceedings.mlr.press/v202/zhang23f.html)

        **Abstract**:

        Existing theories on deep nonparametric regression have shown that when the input data lie on a low-dimensional manifold, deep neural networks can adapt to the intrinsic data structures. In real world applications, such an assumption of data lying exactly on a low dimensional manifold is stringent. This paper introduces a relaxed assumption that the input data are concentrated around a subset of $\mathbb{R}^d$ denoted by $\mathcal{S}$, and the intrinsic dimension of $\mathcal{S}$ can be characterized by a new complexity notation – effective Minkowski dimension. We prove that, the sample complexity of deep nonparametric regression only depends on the effective Minkowski dimension of $\mathcal{S}$ denoted by $p$. We further illustrate our theoretical findings by considering nonparametric regression with an anisotropic Gaussian random design $N(0,\Sigma)$, where $\Sigma$ is full rank. When the eigenvalues of $\Sigma$ have an exponential or polynomial decay, the effective Minkowski dimension of such an Gaussian random design is $p=\mathcal{O}(\sqrt{\log n})$ or $p=\mathcal{O}(n^\gamma)$, respectively, where $n$ is the sample size and $\gamma\in(0,1)$ is a small constant depending on the polynomial decay rate. Our theory shows that, when the manifold assumption does not hold, deep neural networks can still adapt to the effective Minkowski dimension of the data, and circumvent the curse of the ambient dimensionality for moderate sample sizes.

        ----

        ## [1715] Tractable Control for Autoregressive Language Generation

        **Authors**: *Honghua Zhang, Meihua Dang, Nanyun Peng, Guy Van den Broeck*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/zhang23g.html](https://proceedings.mlr.press/v202/zhang23g.html)

        **Abstract**:

        Despite the success of autoregressive large language models in text generation, it remains a major challenge to generate text that satisfies complex constraints: sampling from the conditional distribution ${\Pr}(\text{text} | \alpha)$ is intractable for even the simplest lexical constraints $\alpha$. To overcome this challenge, we propose to use tractable probabilistic models (TPMs) to impose lexical constraints in autoregressive text generation models, which we refer to as GeLaTo (Generating Language with Tractable Constraints). To demonstrate the effectiveness of this framework, we use distilled hidden Markov models, where we can efficiently compute ${\Pr}(\text{text} | \alpha)$, to guide autoregressive generation from GPT2. GeLaTo achieves state-of-the-art performance on challenging benchmarks for constrained text generation (e.g., CommonGen), beating various strong baselines by a large margin. Our work not only opens up new avenues for controlling large language models but also motivates the development of more expressive TPMs.

        ----

        ## [1716] CataBEEM: Integrating Latent Interaction Categories in Node-wise Community Detection Models for Network Data

        **Authors**: *Yuhua Zhang, Walter H. Dempsey*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/zhang23h.html](https://proceedings.mlr.press/v202/zhang23h.html)

        **Abstract**:

        Community detection is a fundamental task in network analysis. Learning underlying network structures has brought deep insights into the understanding of complex systems. While many methods have focused on clustering nodes into blocks, few accounts for the fact that interactions may exhibit edge-level clustering, which we call categories. Real network data often arise via a series of interactions. Interactions in complex systems can often be clustered into different categories and node-level community structures that depend on the category. In this paper, we introduce a category-and-block edge exchangeable model (CataBEEM) to study interaction networks with joint latent interaction-level category and node-level community structures. In particular, the proposed method models the network from the interaction process perspective and allows the incorporation of prior knowledge from auxiliary interaction-wise information. We derive an efficient variational inference algorithm that can be applied to networks consisting of millions of interactions and provide the theoretical bound of the misspecification rate. We demonstrate the effectiveness of our method in various simulation settings and apply the method to TalkLife data, a large-scale online peer-to-peer support network. We show CataBEEM detects more temporally consistent community structures and has better predictions than other methods.

        ----

        ## [1717] Rethink DARTS Search Space and Renovate a New Benchmark

        **Authors**: *Jiuling Zhang, Zhiming Ding*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/zhang23i.html](https://proceedings.mlr.press/v202/zhang23i.html)

        **Abstract**:

        DARTS search space (DSS) has become a canonical benchmark for NAS whereas some emerging works pointed out the issue of narrow accuracy range and claimed it would hurt the method ranking. We observe some recent studies already suffer from this issue that overshadows the meaning of scores. In this work, we first propose and orchestrate a suite of improvements to frame a larger and harder DSS, termed LHD, while retaining high efficiency in search. We step forward to renovate a LHD-based new benchmark, taking care of both discernibility and accessibility. Specifically, we re-implement twelve baselines and evaluate them across twelve conditions by combining two underexpolored influential factors: transductive robustness and discretization policy, to reasonably construct a benchmark upon multi-condition evaluation. Considering that the tabular benchmarks are always insufficient to adequately evaluate the methods of neural architecture search (NAS), our work can serve as a crucial basis for the future progress of NAS.

        ----

        ## [1718] Team Belief DAG: Generalizing the Sequence Form to Team Games for Fast Computation of Correlated Team Max-Min Equilibria via Regret Minimization

        **Authors**: *Brian Hu Zhang, Gabriele Farina, Tuomas Sandholm*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/zhang23j.html](https://proceedings.mlr.press/v202/zhang23j.html)

        **Abstract**:

        A classic result in the theory of extensive-form games asserts that the set of strategies available to any perfect-recall player is strategically equivalent to a low-dimensional convex polytope, called the sequence-form polytope. Online convex optimization tools operating on this polytope are the current state-of-the-art for computing several notions of equilibria in games, and have been crucial in landmark applications of computational game theory. However, when optimizing over the joint strategy space of a team of players, one cannot use the sequence form to obtain a strategically-equivalent convex description of the strategy set of the team. In this paper, we provide new complexity results on the computation of optimal strategies for teams, and propose a new representation, coined team belief DAG (TB-DAG), that describes team strategies as a convex set. The TB-DAG enjoys state-of-the-art parameterized complexity bounds, while at the same time enjoying the advantages of efficient regret minimization techniques. We show that TB-DAG can be exponentially smaller and can be computed exponentially faster than all other known representations, and that the converse is never true. Experimentally, we show that the TB-DAG, when paired with learning techniques, yields state of the art on a wide variety of benchmark team games.

        ----

        ## [1719] A Complete Expressiveness Hierarchy for Subgraph GNNs via Subgraph Weisfeiler-Lehman Tests

        **Authors**: *Bohang Zhang, Guhao Feng, Yiheng Du, Di He, Liwei Wang*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/zhang23k.html](https://proceedings.mlr.press/v202/zhang23k.html)

        **Abstract**:

        Recently, subgraph GNNs have emerged as an important direction for developing expressive graph neural networks (GNNs). While numerous architectures have been proposed, so far there is still a limited understanding of how various design paradigms differ in terms of expressive power, nor is it clear what design principle achieves maximal expressiveness with minimal architectural complexity. To address these fundamental questions, this paper conducts a systematic study of general node-based subgraph GNNs through the lens of Subgraph Weisfeiler-Lehman Tests (SWL). Our central result is to build a complete hierarchy of SWL with strictly growing expressivity. Concretely, we prove that any node-based subgraph GNN falls into one of the six SWL equivalence classes, among which $\mathsf{SSWL}$ achieves the maximal expressive power. We also study how these equivalence classes differ in terms of their practical expressiveness such as encoding graph distance and biconnectivity. In addition, we give a tight expressivity upper bound of all SWL algorithms by establishing a close relation with localized versions of WL and Folklore WL (FWL) tests. Overall, our results provide insights into the power of existing subgraph GNNs, guide the design of new architectures, and point out their limitations by revealing an inherent gap with the 2-FWL test. Finally, experiments demonstrate that $\mathsf{SSWL}$-inspired subgraph GNNs can significantly outperform prior architectures on multiple benchmarks despite great simplicity.

        ----

        ## [1720] Crafting Training Degradation Distribution for the Accuracy-Generalization Trade-off in Real-World Super-Resolution

        **Authors**: *Ruofan Zhang, Jinjin Gu, Haoyu Chen, Chao Dong, Yulun Zhang, Wenming Yang*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/zhang23l.html](https://proceedings.mlr.press/v202/zhang23l.html)

        **Abstract**:

        Super-resolution (SR) techniques designed for real-world applications commonly encounter two primary challenges: generalization performance and restoration accuracy. We demonstrate that when methods are trained using complex, large-range degradations to enhance generalization, a decline in accuracy is inevitable. However, since the degradation in a certain real-world applications typically exhibits a limited variation range, it becomes feasible to strike a trade-off between generalization performance and testing accuracy within this scope. In this work, we introduce a novel approach to craft training degradation distributions using a small set of reference images. Our strategy is founded upon the binned representation of the degradation space and the Frechet distance between degradation distributions. Our results indicate that the proposed technique significantly improves the performance of test images while preserving generalization capabilities in real-world applications.

        ----

        ## [1721] Prompting Large Language Model for Machine Translation: A Case Study

        **Authors**: *Biao Zhang, Barry Haddow, Alexandra Birch*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/zhang23m.html](https://proceedings.mlr.press/v202/zhang23m.html)

        **Abstract**:

        Research on prompting has shown excellent performance with little or even no supervised training across many tasks. However, prompting for machine translation is still under-explored in the literature. We fill this gap by offering a systematic study on prompting strategies for translation, examining various factors for prompt template and demonstration example selection. We further explore the use of monolingual data and the feasibility of cross-lingual, cross-domain, and sentence-to-document transfer learning in prompting. Extensive experiments with GLM-130B (Zeng et al., 2022) as the testbed show that 1) the number and the quality of prompt examples matter, where using suboptimal examples degenerates translation; 2) several features of prompt examples, such as semantic similarity, show significant Spearman correlation with their prompting performance; yet, none of the correlations are strong enough; 3) using pseudo parallel prompt examples constructed from monolingual data via zero-shot prompting could improve translation; and 4) improved performance is achievable by transferring knowledge from prompt examples selected in other settings. We finally provide an analysis on the model outputs and discuss several problems that prompting still suffers from.

        ----

        ## [1722] On the Interplay Between Misspecification and Sub-optimality Gap in Linear Contextual Bandits

        **Authors**: *Weitong Zhang, Jiafan He, Zhiyuan Fan, Quanquan Gu*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/zhang23n.html](https://proceedings.mlr.press/v202/zhang23n.html)

        **Abstract**:

        We study linear contextual bandits in the misspecified setting, where the expected reward function can be approximated by a linear function class up to a bounded misspecification level $\zeta>0$. We propose an algorithm based on a novel data selection scheme, which only selects the contextual vectors with large uncertainty for online regression. We show that, when the misspecification level $\zeta$ is dominated by $\tilde O(\Delta / \sqrt{d})$ with $\Delta$ being the minimal sub-optimality gap and $d$ being the dimension of the contextual vectors, our algorithm enjoys the same gap-dependent regret bound $\tilde O ({d^2} /{\Delta})$ as in the well-specified setting up to logarithmic factors. Given this result, we show that the existing SupLinUCB algorithm (Chu et al., 2011) can also achieve a gap-dependent constant regret bound without the knowledge of sub-optimality gap $\Delta$. Together with a lower bound adapted from Lattimore et al. (2020), our result suggests an interplay between the misspecification level and the sub-optimality gap: (1) the linear contextual bandit model is efficiently learnable when $\zeta \leq \tilde O({\Delta} / \sqrt{d})$; and (2) it is not efficiently learnable when $\zeta \geq \tilde \Omega({\Delta} / {\sqrt{d}})$. Experiments on both synthetic and real-world datasets corroborate our theoretical results.

        ----

        ## [1723] When Sparsity Meets Contrastive Models: Less Graph Data Can Bring Better Class-Balanced Representations

        **Authors**: *Chunhui Zhang, Chao Huang, Yijun Tian, Qianlong Wen, Zhongyu Ouyang, Youhuan Li, Yanfang Ye, Chuxu Zhang*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/zhang23o.html](https://proceedings.mlr.press/v202/zhang23o.html)

        **Abstract**:

        Graph Neural Networks (GNNs) are powerful models for non-Euclidean data, but their training is often accentuated by massive unnecessary computation: on the one hand, training on non-Euclidean data has relatively high computational cost due to its irregular density properties; on the other hand, the class imbalance property often associated with non-Euclidean data cannot be alleviated by the massiveness of the data, thus hindering the generalisation of the models. To address the above issues, theoretically, we start with a hypothesis about the effectiveness of using a subset of training data for GNNs, which is guaranteed by the gradient distance between the subset and the full set. Empirically, we also observe that a subset of the data can provide informative gradients for model optimization and which changes over time dynamically. We name this phenomenon dynamic data sparsity. Additionally, we find that pruned sparse contrastive models may miss valuable information, leading to a large loss value on the informative subset. Motivated by the above findings, we develop a unified data model dynamic sparsity framework called Data Decantation (DataDec) to address the above challenges. The key idea of DataDec is to identify the informative subset dynamically during the training process by applying sparse graph contrastive learning. The effectiveness of DataDec is comprehensively evaluated on graph benchmark datasets and we also verify its generalizability on image data.

        ----

        ## [1724] Spatial-Temporal Graph Learning with Adversarial Contrastive Adaptation

        **Authors**: *Qianru Zhang, Chao Huang, Lianghao Xia, Zheng Wang, Siu Ming Yiu, Ruihua Han*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/zhang23p.html](https://proceedings.mlr.press/v202/zhang23p.html)

        **Abstract**:

        Spatial-temporal graph learning has emerged as the state-of-the-art solution for modeling structured spatial-temporal data in learning region representations for various urban sensing tasks (e.g., crime forecasting, traffic flow prediction). However, most existing models are vulnerable to the quality of the generated region graph due to the inartistic graph-structured information aggregation schema. The ubiquitous spatial-temporal data noise and incompleteness in real-life scenarios bring difficulties to generate high-quality region representations. In this paper, we propose a Spatial-Temporal Adversarial Graph contrastive learning model (STAG) to tackle this challenge for adaptive self-supervised graph augmentation. Specifically, we propose a learnable contrastive learning function that enables the automated distillation of important multi-view self-supervised signals for adaptive spatial-temporal graph augmentation. To enhance the representation discrimination ability and robustness, the designed adversarial contrastive learning mechanism empowers STAG to adaptively identify hard samples for better self-supervision. Finally, a cross-view contrastive learning paradigm is introduced to model the inter-dependencies across view-specific region representations and preserve the underlying relation heterogeneity. We verify the superiority of our STAG method in various spatial-temporal prediction tasks on several benchmark datasets.

        ----

        ## [1725] Towards Coherent Image Inpainting Using Denoising Diffusion Implicit Models

        **Authors**: *Guanhua Zhang, Jiabao Ji, Yang Zhang, Mo Yu, Tommi S. Jaakkola, Shiyu Chang*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/zhang23q.html](https://proceedings.mlr.press/v202/zhang23q.html)

        **Abstract**:

        Image inpainting refers to the task of generating a complete, natural image based on a partially revealed reference image. Recently, many research interests have been focused on addressing this problem using fixed diffusion models. These approaches typically directly replace the revealed region of the intermediate or final generated images with that of the reference image or its variants. However, since the unrevealed regions are not directly modified to match the context, it results in incoherence between revealed and unrevealed regions. To address the incoherence problem, a small number of methods introduce a rigorous Bayesian framework, but they tend to introduce mismatches between the generated and the reference images due to the approximation errors in computing the posterior distributions. In this paper, we propose CoPaint, which can coherently inpaint the whole image without introducing mismatches. CoPaint also uses the Bayesian framework to jointly modify both revealed and unrevealed regions but approximates the posterior distribution in a way that allows the errors to gradually drop to zero throughout the denoising steps, thus strongly penalizing any mismatches with the reference image. Our experiments verify that CoPaint can outperform the existing diffusion-based methods under both objective and subjective metrics.

        ----

        ## [1726] CAB: Comprehensive Attention Benchmarking on Long Sequence Modeling

        **Authors**: *Jun Zhang, Shuyang Jiang, Jiangtao Feng, Lin Zheng, Lingpeng Kong*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/zhang23r.html](https://proceedings.mlr.press/v202/zhang23r.html)

        **Abstract**:

        Transformer has achieved remarkable success in language, image, and speech processing. Recently, various efficient attention architectures have been proposed to improve transformer’s efficiency while largely preserving its efficacy, especially in modeling long sequences. A widely-used benchmark to test these efficient methods’ capability on long-range modeling is Long Range Arena (LRA). However, LRA only focuses on the standard bidirectional (or noncausal) self attention, and completely ignores cross attentions and unidirectional (or causal) attentions, which are equally important to downstream applications. In this paper, we propose Comprehensive Attention Benchmark (CAB) under a fine-grained attention taxonomy with four distinguishable attention patterns, namely, noncausal self, causal self, noncausal cross, and causal cross attentions. CAB collects seven real-world tasks from different research areas to evaluate efficient attentions under the four attention patterns. Among these tasks, CAB validates efficient attentions in eight backbone networks to show their generalization across neural architectures. We conduct exhaustive experiments to benchmark the performances of nine widely-used efficient attention architectures designed with different philosophies on CAB. Extensive experimental results also shed light on the fundamental problems of efficient attentions, such as efficiency length against vanilla attention, performance consistency across attention patterns, the benefit of attention mechanisms, and interpolation/extrapolation on long-context language modeling.

        ----

        ## [1727] Adaptive Barrier Smoothing for First-Order Policy Gradient with Contact Dynamics

        **Authors**: *Shenao Zhang, Wanxin Jin, Zhaoran Wang*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/zhang23s.html](https://proceedings.mlr.press/v202/zhang23s.html)

        **Abstract**:

        Differentiable physics-based simulators have witnessed remarkable success in robot learning involving contact dynamics, benefiting from their improved accuracy and efficiency in solving the underlying complementarity problem. However, when utilizing the First-Order Policy Gradient (FOPG) method, our theory indicates that the complementarity-based systems suffer from stiffness, leading to an explosion in the gradient variance of FOPG. As a result, optimization becomes challenging due to chaotic and non-smooth loss landscapes. To tackle this issue, we propose a novel approach called Adaptive Barrier Smoothing (ABS), which introduces a class of softened complementarity systems that correspond to barrier-smoothed objectives. With a contact-aware adaptive central-path parameter, ABS reduces the FOPG gradient variance while controlling the gradient bias. We justify the adaptive design by analyzing the roots of the system’s stiffness. Additionally, we establish the convergence of FOPG and show that ABS achieves a reasonable trade-off between the gradient variance and bias by providing their upper bounds. Moreover, we present a variant of FOPG based on complementarity modeling that efficiently fits the contact dynamics by learning the physical parameters. Experimental results on various robotic tasks are provided to support our theory and method.

        ----

        ## [1728] One-Step Estimator for Permuted Sparse Recovery

        **Authors**: *Hang Zhang, Ping Li*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/zhang23t.html](https://proceedings.mlr.press/v202/zhang23t.html)

        **Abstract**:

        This paper considers the unlabeled sparse recovery under multiple measurements, i.e., ${\mathbf{Y}} = {\mathbf{\Pi}}^{\natural} {\mathbf{X}} {\mathbf{B}}^{\natural} + {\mathbf{W}}$, where ${\mathbf{Y}} \in \mathbb{R}^{n\times m}, {\mathbf{\Pi}}^{\natural}\in \mathbb{R}^{n\times n}, {\mathbf{X}} \in \mathbb{R}^{n\times p}, {\mathbf{B}} ^{\natural}\in \mathbb{R}^{p\times m}, {\mathbf{W}}\in \mathbb{R}^{n\times m}$ represents the observations, missing (or incomplete) correspondence information, sensing matrix, sparse signals, and additive sensing noise, respectively. Different from the previous works on multiple measurements ($m > 1$) which all focus on the sufficient samples regime, namely, $n > p$, we consider a sparse matrix $\mathbf{B}^{\natural}$ and investigate the insufficient samples regime (i.e., $n \ll p$) for the first time. To begin with, we establish the lower bound on the sample number and signal-to-noise ratio ($ {\mathsf{SNR}}$) for the correct permutation recovery. Moreover, we present a simple yet effective estimator. Under mild conditions, we show that our estimator can restore the correct correspondence information with high probability. Numerical experiments are presented to corroborate our theoretical claims.

        ----

        ## [1729] Quantum Lower Bounds for Finding Stationary Points of Nonconvex Functions

        **Authors**: *Chenyi Zhang, Tongyang Li*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/zhang23u.html](https://proceedings.mlr.press/v202/zhang23u.html)

        **Abstract**:

        Quantum computing is an emerging technology that has been rapidly advancing in the past decades. In this paper, we conduct a systematic study of quantum lower bounds on finding $\epsilon$-approximate stationary points of nonconvex functions, and we consider the following two important settings: 1) having access to $p$-th order derivatives; or 2) having access to stochastic gradients. The classical query lower bounds are $\Omega\big(\epsilon^{-\frac{1+p}{p}}\big)$ regarding the first setting and $\Omega(\epsilon^{-4})$ regarding the second setting (or $\Omega(\epsilon^{-3})$ if the stochastic gradient function is mean-squared smooth). In this paper, we extend all these classical lower bounds to the quantum setting. They match the classical algorithmic results respectively, demonstrating that there is no quantum speedup for finding $\epsilon$-stationary points of nonconvex functions with $p$-th order derivative inputs or stochastic gradient inputs, whether with or without the mean-squared smoothness assumption. Technically, we prove our quantum lower bounds by showing that the sequential nature of classical hard instances in all these settings also applies to quantum queries, preventing any quantum speedup other than revealing information of the stationary points sequentially.

        ----

        ## [1730] Improving Medical Predictions by Irregular Multimodal Electronic Health Records Modeling

        **Authors**: *Xinlu Zhang, Shiyang Li, Zhiyu Chen, Xifeng Yan, Linda Ruth Petzold*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/zhang23v.html](https://proceedings.mlr.press/v202/zhang23v.html)

        **Abstract**:

        Health conditions among patients in intensive care units (ICUs) are monitored via electronic health records (EHRs), composed of numerical time series and lengthy clinical note sequences, both taken at $\textit{irregular}$ time intervals. Dealing with such irregularity in every modality, and integrating irregularity into multimodal representations to improve medical predictions, is a challenging problem. Our method first addresses irregularity in each single modality by (1) modeling irregular time series by dynamically incorporating hand-crafted imputation embeddings into learned interpolation embeddings via a gating mechanism, and (2) casting a series of clinical note representations as multivariate irregular time series and tackling irregularity via a time attention mechanism. We further integrate irregularity in multimodal fusion with an interleaved attention mechanism across temporal steps. To the best of our knowledge, this is the first work to thoroughly model irregularity in multimodalities for improving medical predictions. Our proposed methods for two medical prediction tasks consistently outperforms state-of-the-art (SOTA) baselines in each single modality and multimodal fusion scenarios. Specifically, we observe relative improvements of 6.5%, 3.6%, and 4.3% in F1 for time series, clinical notes, and multimodal fusion, respectively. These results demonstrate the effectiveness of our methods and the importance of considering irregularity in multimodal EHRs.

        ----

        ## [1731] FedCR: Personalized Federated Learning Based on Across-Client Common Representation with Conditional Mutual Information Regularization

        **Authors**: *Hao Zhang, Chenglin Li, Wenrui Dai, Junni Zou, Hongkai Xiong*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/zhang23w.html](https://proceedings.mlr.press/v202/zhang23w.html)

        **Abstract**:

        In personalized federated learning (PFL), multiple clients train customized models to fulfill their personal objectives, which, however, are prone to overfitting to local data due to the heterogeneity and scarcity of local data. To address this, we propose from the information-theoretic perspective a personalized federated learning framework based on the common representation learned across clients, named FedCR. Specifically, we introduce to the local client update a regularizer that aims at minimizing the discrepancy between local and global conditional mutual information (CMI), such that clients are encouraged to learn and exploit the common representation. Upon this, each client learns individually a customized predictor (head), while the extractor (body) remains to be aggregated by the server. Our CMI regularizer leads to a theoretically sound alignment between the local and global stochastic feature distributions in terms of their Kullback-Leibler (KL) divergence. More importantly, by modeling the global joint feature distribution as a product of multiple local feature distributions, clients can efficiently extract diverse information from the global data but without need of the raw data from other clients. We further show that noise injection via feature alignment and ensemble of local predictors in FedCR would help enhance its generalization capability. Experiments on benchmark datasets demonstrate a consistent performance gain and better generalization behavior of FedCR.

        ----

        ## [1732] On the Optimality of Misspecified Kernel Ridge Regression

        **Authors**: *Haobo Zhang, Yicheng Li, Weihao Lu, Qian Lin*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/zhang23x.html](https://proceedings.mlr.press/v202/zhang23x.html)

        **Abstract**:

        In the misspecified kernel ridge regression problem, researchers usually assume the underground true function $f_{\rho}^{\star} \in [\mathcal{H}]^{s}$, a less-smooth interpolation space of a reproducing kernel Hilbert space (RKHS) $\mathcal{H}$ for some $s\in (0,1)$. The existing minimax optimal results require $\left\Vert f_{\rho}^{\star} \right \Vert_{L^{\infty}} < \infty$ which implicitly requires $s > \alpha_{0}$ where $\alpha_{0} \in (0,1) $ is the embedding index, a constant depending on $\mathcal{H}$. Whether the KRR is optimal for all $s\in (0,1)$ is an outstanding problem lasting for years. In this paper, we show that KRR is minimax optimal for any $s\in (0,1)$ when the $\mathcal{H}$ is a Sobolev RKHS.

        ----

        ## [1733] Fed-CBS: A Heterogeneity-Aware Client Sampling Mechanism for Federated Learning via Class-Imbalance Reduction

        **Authors**: *Jianyi Zhang, Ang Li, Minxue Tang, Jingwei Sun, Xiang Chen, Fan Zhang, Changyou Chen, Yiran Chen, Hai Li*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/zhang23y.html](https://proceedings.mlr.press/v202/zhang23y.html)

        **Abstract**:

        Due to the often limited communication bandwidth of edge devices, most existing federated learning (FL) methods randomly select only a subset of devices to participate in training at each communication round. Compared with engaging all the available clients, such a random-selection mechanism could lead to significant performance degradation on non-IID (independent and identically distributed) data. In this paper, we present our key observation that the essential reason resulting in such performance degradation is the class-imbalance of the grouped data from randomly selected clients. Based on this observation, we design an efficient heterogeneity-aware client sampling mechanism, namely, Federated Class-balanced Sampling (Fed-CBS), which can effectively reduce class-imbalance of the grouped dataset from the intentionally selected clients. We first propose a measure of class-imbalance which can be derived in a privacy-preserving way. Based on this measure, we design a computation-efficient client sampling strategy such that the actively selected clients will generate a more class-balanced grouped dataset with theoretical guarantees. Experimental results show that Fed-CBS outperforms the status quo approaches in terms of test accuracy and the rate of convergence while achieving comparable or even better performance than the ideal setting where all the available clients participate in the FL training.

        ----

        ## [1734] Learning Subpocket Prototypes for Generalizable Structure-based Drug Design

        **Authors**: *Zaixi Zhang, Qi Liu*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/zhang23z.html](https://proceedings.mlr.press/v202/zhang23z.html)

        **Abstract**:

        Generating molecules with high binding affinities to target proteins (a.k.a. structure-based drug design) is a fundamental and challenging task in drug discovery. Recently, deep generative models have achieved remarkable success in generating 3D molecules conditioned on the protein pocket. However, most existing methods consider molecular generation for protein pockets independently while neglecting the underlying connections such as subpocket-level similarities. Subpockets are the local protein environments of ligand fragments and pockets with similar subpockets may bind the same molecular fragment (motif) even though their overall structures are different. Therefore, the trained models can hardly generalize to unseen protein pockets in real-world applications. In this paper, we propose a novel method DrugGPS for generalizable structure-based drug design. With the biochemical priors, we propose to learn subpocket prototypes and construct a global interaction graph to model the interactions between subpocket prototypes and molecular motifs. Moreover, a hierarchical graph transformer encoder and motif-based 3D molecule generation scheme are used to improve the model’s performance. The experimental results show that our model consistently outperforms baselines in generating realistic drug candidates with high affinities in challenging out-of-distribution settings.

        ----

        ## [1735] No One Idles: Efficient Heterogeneous Federated Learning with Parallel Edge and Server Computation

        **Authors**: *Feilong Zhang, Xianming Liu, Shiyi Lin, Gang Wu, Xiong Zhou, Junjun Jiang, Xiangyang Ji*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/zhang23aa.html](https://proceedings.mlr.press/v202/zhang23aa.html)

        **Abstract**:

        Federated learning suffers from a latency bottleneck induced by network stragglers, which hampers the training efficiency significantly. In addition, due to the heterogeneous data distribution and security requirements, simple and fast averaging aggregation is not feasible anymore. Instead, complicated aggregation operations, such as knowledge distillation, are required. The time cost for complicated aggregation becomes a new bottleneck that limits the computational efficiency of FL. In this work, we claim that the root cause of training latency actually lies in the aggregation-then-broadcasting workflow of the server. By swapping the computational order of aggregation and broadcasting, we propose a novel and efficient parallel federated learning (PFL) framework that unlocks the edge nodes during global computation and the central server during local computation. This fully asynchronous and parallel pipeline enables handling complex aggregation and network stragglers, allowing flexible device participation as well as achieving scalability in computation. We theoretically prove that synchronous and asynchronous PFL can achieve a similar convergence rate as vanilla FL. Extensive experiments empirically show that our framework brings up to $5.56\times$ speedup compared with traditional FL. Code is available at: https://github.com/Hypervoyager/PFL.

        ----

        ## [1736] The Wisdom of Hindsight Makes Language Models Better Instruction Followers

        **Authors**: *Tianjun Zhang, Fangchen Liu, Justin Wong, Pieter Abbeel, Joseph E. Gonzalez*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/zhang23ab.html](https://proceedings.mlr.press/v202/zhang23ab.html)

        **Abstract**:

        Reinforcement learning has seen wide success in finetuning large language models to better align with instructions via human feedback. The so-called algorithm, Reinforcement Learning with Human Feedback (RLHF) demonstrates impressive performance on the GPT series models. However, the underlying reinforcement learning algorithm is complex and requires additional training for reward and value networks. In this paper, we consider an alternative approach: converting feedback to instruction by relabeling the original one and training the model for better alignment in a supervised manner. Such an algorithm doesn’t require any additional parameters except for the original language model and maximally reuses the pretraining pipeline. To achieve this, we formulate instruction alignment problem for language models as a goal-reaching problem in decision making. We propose Hindsight Instruction Relabeling (HIR), a novel algorithm for aligning language models with instructions. The resulting two-stage algorithm shed light to a family of reward-free approaches that utilize the hindsightly relabeled instructions based on feedback. We evaluate the performance of HIR extensively on 12 challenging BigBench reasoning tasks and show that HIR outperforms the baseline algorithms and is comparable to or even surpasses supervised fine-tuning. The implementation of HIR is available at https://github.com/tianjunz/HIR.

        ----

        ## [1737] Detecting Adversarial Data by Probing Multiple Perturbations Using Expected Perturbation Score

        **Authors**: *Shuhai Zhang, Feng Liu, Jiahao Yang, Yifan Yang, Changsheng Li, Bo Han, Mingkui Tan*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/zhang23ac.html](https://proceedings.mlr.press/v202/zhang23ac.html)

        **Abstract**:

        Adversarial detection aims to determine whether a given sample is an adversarial one based on the discrepancy between natural and adversarial distributions. Unfortunately, estimating or comparing two data distributions is extremely difficult, especially in high-dimension spaces. Recently, the gradient of log probability density (a.k.a., score) w.r.t. the sample is used as an alternative statistic to compute. However, we find that the score is sensitive in identifying adversarial samples due to insufficient information with one sample only. In this paper, we propose a new statistic called expected perturbation score (EPS), which is essentially the expected score of a sample after various perturbations. Specifically, to obtain adequate information regarding one sample, we perturb it by adding various noises to capture its multi-view observations. We theoretically prove that EPS is a proper statistic to compute the discrepancy between two samples under mild conditions. In practice, we can use a pre-trained diffusion model to estimate EPS for each sample. Last, we pro- pose an EPS-based adversarial detection (EPS- AD) method, in which we develop EPS-based maximum mean discrepancy (MMD) as a metric to measure the discrepancy between the test sample and natural samples. We also prove that the EPS-based MMD between natural and adversarial samples is larger than that among natural samples. Extensive experiments show the superior adversarial detection performance of our EPS-AD.

        ----

        ## [1738] On Enhancing Expressive Power via Compositions of Single Fixed-Size ReLU Network

        **Authors**: *Shijun Zhang, Jianfeng Lu, Hongkai Zhao*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/zhang23ad.html](https://proceedings.mlr.press/v202/zhang23ad.html)

        **Abstract**:

        This paper explores the expressive power of deep neural networks through the framework of function compositions. We demonstrate that the repeated compositions of a single fixed-size ReLU network exhibit surprising expressive power, despite the limited expressive capabilities of the individual network itself. Specifically, we prove by construction that $\mathcal{L}_2\circ \boldsymbol{g}^{\circ r}\circ \boldsymbol{\mathcal{L}}_1$ can approximate $1$-Lipschitz continuous functions on $[0,1]^d$ with an error $\mathcal{O}(r^{-1/d})$, where $\boldsymbol{g}$ is realized by a fixed-size ReLU network, $\boldsymbol{\mathcal{L}}_1$ and $\mathcal{L}_2$ are two affine linear maps matching the dimensions, and $\boldsymbol{g}^{\circ r}$ denotes the $r$-times composition of $\boldsymbol{g}$. Furthermore, we extend such a result to generic continuous functions on $[0,1]^d$ with the approximation error characterized by the modulus of continuity. Our results reveal that a continuous-depth network generated via a dynamical system has immense approximation power even if its dynamics function is time-independent and realized by a fixed-size ReLU network.

        ----

        ## [1739] Bi-directional Masks for Efficient N: M Sparse Training

        **Authors**: *Yuxin Zhang, Yiting Luo, Mingbao Lin, Yunshan Zhong, Jingjing Xie, Fei Chao, Rongrong Ji*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/zhang23ae.html](https://proceedings.mlr.press/v202/zhang23ae.html)

        **Abstract**:

        We focus on addressing the dense backward propagation issue for training efficiency of N:M fine-grained sparsity that preserves at most N out of M consecutive weights and achieves practical speedups supported by the N:M sparse tensor core. Therefore, we present a novel method of Bi-directional Masks (Bi-Mask) with its two central innovations in: 1) Separate sparse masks in the two directions of forward and backward propagation to obtain training acceleration. It disentangles the forward and backward weight sparsity and overcomes the very dense gradient computation. 2) An efficient weight row permutation method to maintain performance. It picks up the permutation candidate with the most eligible N:M weight blocks in the backward to minimize the gradient gap between traditional unidirectional masks and our bi-directional masks. Compared with existing uni-directional scenario that applies a transposable mask and enables backward acceleration, our Bi-Mask is experimentally demonstrated to be more superior in performance. Also, our Bi-Mask performs on par with or even better than methods that fail to achieve backward acceleration. Project of this paper is available at https://github.com/zyxxmu/Bi-Mask.

        ----

        ## [1740] Towards Unbiased Training in Federated Open-world Semi-supervised Learning

        **Authors**: *Jie Zhang, Xiaosong Ma, Song Guo, Wenchao Xu*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/zhang23af.html](https://proceedings.mlr.press/v202/zhang23af.html)

        **Abstract**:

        Federated Semi-supervised Learning (FedSSL) has emerged as a new paradigm for allowing distributed clients to collaboratively train a machine learning model over scarce labeled data and abundant unlabeled data. However, existing works for FedSSL rely on a closed-world assumption that all local training data and global testing data are from seen classes observed in the labeled dataset. It is crucial to go one step further: adapting FL models to an open-world setting, where unseen classes exist in the unlabeled data. In this paper, we propose a novel Federatedopen-world Semi-Supervised Learning (FedoSSL) framework, which can solve the key challenge in distributed and open-world settings, i.e., the biased training process for heterogeneously distributed unseen classes. Specifically, since the advent of a certain unseen class depends on a client basis, the locally unseen classes (exist in multiple clients) are likely to receive differentiated superior aggregation effects than the globally unseen classes (exist only in one client). We adopt an uncertainty-aware suppressed loss to alleviate the biased training between locally unseen and globally unseen classes. Besides, we enable a calibration module supplementary to the global aggregation to avoid potential conflicting knowledge transfer caused by inconsistent data distribution among different clients. The proposed FedoSSL can be easily adapted to state-of-the-art FL methods, which is also validated via extensive experiments on benchmarks and real-world datasets (CIFAR-10, CIFAR-100 and CINIC-10).

        ----

        ## [1741] Interactive Object Placement with Reinforcement Learning

        **Authors**: *Shengping Zhang, Quanling Meng, Qinglin Liu, Liqiang Nie, Bineng Zhong, Xiaopeng Fan, Rongrong Ji*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/zhang23ag.html](https://proceedings.mlr.press/v202/zhang23ag.html)

        **Abstract**:

        Object placement aims to insert a foreground object into a background image with a suitable location and size to create a natural composition. To predict a diverse distribution of placements, existing methods usually establish a one-to-one mapping from random vectors to the placements. However, these random vectors are not interpretable, which prevents users from interacting with the object placement process. To address this problem, we propose an Interactive Object Placement method with Reinforcement Learning, dubbed IOPRE, to make sequential decisions for producing a reasonable placement given an initial location and size of the foreground. We first design a novel action space to flexibly and stably adjust the location and size of the foreground while preserving its aspect ratio. Then, we propose a multi-factor state representation learning method, which integrates composition image features and sinusoidal positional embeddings of the foreground to make decisions for selecting actions. Finally, we design a hybrid reward function that combines placement assessment and the number of steps to ensure that the agent learns to place objects in the most visually pleasing and semantically appropriate location. Experimental results on the OPA dataset demonstrate that the proposed method achieves state-of-the-art performance in terms of plausibility and diversity.

        ----

        ## [1742] Optimal Shrinkage for Distributed Second-Order Optimization

        **Authors**: *Fangzhao Zhang, Mert Pilanci*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/zhang23ah.html](https://proceedings.mlr.press/v202/zhang23ah.html)

        **Abstract**:

        In this work, we address the problem of Hessian inversion bias in distributed second-order optimization algorithms. We introduce a novel shrinkage-based estimator for the resolvent of gram matrices which is asymptotically unbiased, and characterize its non-asymptotic convergence rate in the isotropic case. We apply this estimator to bias correction of Newton steps in distributed second-order optimization algorithms, as well as randomized sketching based methods. We examine the bias present in the naive averaging-based distributed Newton’s method using analytical expressions and contrast it with our proposed biasfree approach. Our approach leads to significant improvements in convergence rate compared to standard baselines and recent proposals, as shown through experiments on both real and synthetic datasets.

        ----

        ## [1743] "Why did the Model Fail?": Attributing Model Performance Changes to Distribution Shifts

        **Authors**: *Haoran Zhang, Harvineet Singh, Marzyeh Ghassemi, Shalmali Joshi*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/zhang23ai.html](https://proceedings.mlr.press/v202/zhang23ai.html)

        **Abstract**:

        Machine learning models frequently experience performance drops under distribution shifts. The underlying cause of such shifts may be multiple simultaneous factors such as changes in data quality, differences in specific covariate distributions, or changes in the relationship between label and features. When a model does fail during deployment, attributing performance change to these factors is critical for the model developer to identify the root cause and take mitigating actions. In this work, we introduce the problem of attributing performance differences between environments to distribution shifts in the underlying data generating mechanisms. We formulate the problem as a cooperative game where the players are distributions. We define the value of a set of distributions to be the change in model performance when only this set of distributions has changed between environments, and derive an importance weighting method for computing the value of an arbitrary set of distributions. The contribution of each distribution to the total performance change is then quantified as its Shapley value. We demonstrate the correctness and utility of our method on synthetic, semi-synthetic, and real-world case studies, showing its effectiveness in attributing performance changes to a wide range of distribution shifts.

        ----

        ## [1744] Learning Regions of Interest for Bayesian Optimization with Adaptive Level-Set Estimation

        **Authors**: *Fengxue Zhang, Jialin Song, James C. Bowden, Alexander Ladd, Yisong Yue, Thomas Desautels, Yuxin Chen*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/zhang23aj.html](https://proceedings.mlr.press/v202/zhang23aj.html)

        **Abstract**:

        We study Bayesian optimization (BO) in high-dimensional and non-stationary scenarios. Existing algorithms for such scenarios typically require extensive hyperparameter tuning, which limits their practical effectiveness. We propose a framework, called BALLET, which adaptively filters for a high-confidence region of interest (ROI) as a superlevel-set of a nonparametric probabilistic model such as a Gaussian process (GP). Our approach is easy to tune, and is able to focus on local region of the optimization space that can be tackled by existing BO methods. The key idea is to use two probabilistic models: a coarse GP to identify the ROI, and a localized GP for optimization within the ROI. We show theoretically that BALLET can efficiently shrink the search space, and can exhibit a tighter regret bound than standard BO without ROI filtering. We demonstrate empirically the effectiveness of BALLET on both synthetic and real-world optimization tasks.

        ----

        ## [1745] A Category-theoretical Meta-analysis of Definitions of Disentanglement

        **Authors**: *Yivan Zhang, Masashi Sugiyama*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/zhang23ak.html](https://proceedings.mlr.press/v202/zhang23ak.html)

        **Abstract**:

        Disentangling the factors of variation in data is a fundamental concept in machine learning and has been studied in various ways by different researchers, leading to a multitude of definitions. Despite the numerous empirical studies, more theoretical research is needed to fully understand the defining properties of disentanglement and how different definitions relate to each other. This paper presents a meta-analysis of existing definitions of disentanglement, using category theory as a unifying and rigorous framework. We propose that the concepts of the cartesian and monoidal products should serve as the core of disentanglement. With these core concepts, we show the similarities and crucial differences in dealing with (i) functions, (ii) equivariant maps, (iii) relations, and (iv) stochastic maps. Overall, our meta-analysis deepens our understanding of disentanglement and its various formulations and can help researchers navigate different definitions and choose the most appropriate one for their specific context.

        ----

        ## [1746] On the Convergence of SARSA with Linear Function Approximation

        **Authors**: *Shangtong Zhang, Remi Tachet des Combes, Romain Laroche*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/zhang23al.html](https://proceedings.mlr.press/v202/zhang23al.html)

        **Abstract**:

        SARSA, a classical on-policy control algorithm for reinforcement learning, is known to chatter when combined with linear function approximation: SARSA does not diverge but oscillates in a bounded region. However, little is known about how fast SARSA converges to that region and how large the region is. In this paper, we make progress towards this open problem by showing the convergence rate of projected SARSA to a bounded region. Importantly, the region is much smaller than the region that we project into, provided that the the magnitude of the reward is not too large. Existing works regarding the convergence of linear SARSA to a fixed point all require the Lipschitz constant of SARSA’s policy improvement operator to be sufficiently small; our analysis instead applies to arbitrary Lipschitz constants and thus characterizes the behavior of linear SARSA for a new regime.

        ----

        ## [1747] AdaNPC: Exploring Non-Parametric Classifier for Test-Time Adaptation

        **Authors**: *Yifan Zhang, Xue Wang, Kexin Jin, Kun Yuan, Zhang Zhang, Liang Wang, Rong Jin, Tieniu Tan*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/zhang23am.html](https://proceedings.mlr.press/v202/zhang23am.html)

        **Abstract**:

        Many recent machine learning tasks focus to develop models that can generalize to unseen distributions. Domain generalization (DG) has become one of the key topics in various fields. Several literatures show that DG can be arbitrarily hard without exploiting target domain information. To address this issue, test-time adaptive (TTA) methods are proposed. Existing TTA methods require offline target data or extra sophisticated optimization procedures during the inference stage. In this work, we adopt Non-Parametric Classifier to perform the test-time Adaptation (AdaNPC). In particular, we construct a memory that contains the feature and label pairs from training domains. During inference, given a test instance, AdaNPC first recalls $k$ closed samples from the memory to vote for the prediction, and then the test feature and predicted label are added to the memory. In this way, the sample distribution in the memory can be gradually changed from the training distribution towards the test distribution with very little extra computation cost. We theoretically justify the rationality behind the proposed method. Besides, we test our model on extensive numerical experiments. AdaNPC significantly outperforms competitive baselines on various DG benchmarks. In particular, when the adaptation target is a series of domains, the adaptation accuracy of AdaNPC is $50$% higher than advanced TTA methods.

        ----

        ## [1748] On the Generalization of Multi-modal Contrastive Learning

        **Authors**: *Qi Zhang, Yifei Wang, Yisen Wang*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/zhang23an.html](https://proceedings.mlr.press/v202/zhang23an.html)

        **Abstract**:

        Multi-modal contrastive learning (MMCL) has recently garnered considerable interest due to its superior performance in visual tasks, achieved by embedding multi-modal data, such as visual-language pairs. However, there still lack theoretical understandings of how MMCL extracts useful visual representation from multi-modal pairs, and particularly, how MMCL outperforms previous approaches like self-supervised contrastive learning (SSCL). In this paper, by drawing an intrinsic connection between MMCL and asymmetric matrix factorization, we establish the first generalization guarantees of MMCL for visual downstream tasks. Based on this framework, we further unify MMCL and SSCL by showing that MMCL implicitly performs SSCL with (pseudo) positive pairs induced by text pairs. Through this unified perspective, we characterize the advantage of MMCL by showing that text pairs induce more semantically consistent and diverse positive pairs, which, according to our analysis, provably benefit downstream generalization. Inspired by this finding, we propose several methods to significantly improve the downstream performance of SSCL on ImageNet by leveraging multi-modal information. Code is available at https://github.com/PKU-ML/CLIP-Help-SimCLR.

        ----

        ## [1749] ConCerNet: A Contrastive Learning Based Framework for Automated Conservation Law Discovery and Trustworthy Dynamical System Prediction

        **Authors**: *Wang Zhang, Tsui-Wei Weng, Subhro Das, Alexandre Megretski, Luca Daniel, Lam M. Nguyen*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/zhang23ao.html](https://proceedings.mlr.press/v202/zhang23ao.html)

        **Abstract**:

        Deep neural networks (DNN) have shown great capacity of modeling a dynamical system; nevertheless, they usually do not obey physics constraints such as conservation laws. This paper proposes a new learning framework named $\textbf{ConCerNet}$ to improve the trustworthiness of the DNN based dynamics modeling to endow the invariant properties. $\textbf{ConCerNet}$ consists of two steps: (i) a contrastive learning method to automatically capture the system invariants (i.e. conservation properties) along the trajectory observations; (ii) a neural projection layer to guarantee that the learned dynamics models preserve the learned invariants. We theoretically prove the functional relationship between the learned latent representation and the unknown system invariant function. Experiments show that our method consistently outperforms the baseline neural networks in both coordinate error and conservation metrics by a large margin. With neural network based parameterization and no dependence on prior knowledge, our method can be extended to complex and large-scale dynamics by leveraging an autoencoder.

        ----

        ## [1750] Towards Trustworthy Explanation: On Causal Rationalization

        **Authors**: *Wenbo Zhang, Tong Wu, Yunlong Wang, Yong Cai, Hengrui Cai*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/zhang23ap.html](https://proceedings.mlr.press/v202/zhang23ap.html)

        **Abstract**:

        With recent advances in natural language processing, rationalization becomes an essential self-explaining diagram to disentangle the black box by selecting a subset of input texts to account for the major variation in prediction. Yet, existing association-based approaches on rationalization cannot identify true rationales when two or more snippets are highly inter-correlated and thus provide a similar contribution to prediction accuracy, so-called spuriousness. To address this limitation, we novelly leverage two causal desiderata, non-spuriousness and efficiency, into rationalization from the causal inference perspective. We formally define a series of probabilities of causation based on a newly proposed structural causal model of rationalization, with its theoretical identification established as the main component of learning necessary and sufficient rationales. The superior performance of the proposed causal rationalization is demonstrated on real-world review and medical datasets with extensive experiments compared to state-of-the-art methods.

        ----

        ## [1751] Demystifying Uneven Vulnerability of Link Stealing Attacks against Graph Neural Networks

        **Authors**: *He Zhang, Bang Wu, Shuo Wang, Xiangwen Yang, Minhui Xue, Shirui Pan, Xingliang Yuan*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/zhang23aq.html](https://proceedings.mlr.press/v202/zhang23aq.html)

        **Abstract**:

        While graph neural networks (GNNs) dominate the state-of-the-art for exploring graphs in real-world applications, they have been shown to be vulnerable to a growing number of privacy attacks. For instance, link stealing is a well-known membership inference attack (MIA) on edges that infers the presence of an edge in a GNN’s training graph. Recent studies on independent and identically distributed data (e.g., images) have empirically demonstrated that individuals from different groups suffer from different levels of privacy risks to MIAs, i.e., uneven vulnerability. However, theoretical evidence of such uneven vulnerability is missing. In this paper, we first present theoretical evidence of the uneven vulnerability of GNNs to link stealing attacks, which lays the foundation for demystifying such uneven risks among different groups of edges. We further demonstrate a group-based attack paradigm to expose the practical privacy harm to GNN users derived from the uneven vulnerability of edges. Finally, we empirically validate the existence of obvious uneven vulnerability on nine real-world datasets (e.g., about 25% AUC difference between different groups in the Credit graph). Compared with existing methods, the outperformance of our group-based attack paradigm confirms that customising different strategies for different groups results in more effective privacy attacks.

        ----

        ## [1752] Provable Dynamic Fusion for Low-Quality Multimodal Data

        **Authors**: *Qingyang Zhang, Haitao Wu, Changqing Zhang, Qinghua Hu, Huazhu Fu, Joey Tianyi Zhou, Xi Peng*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/zhang23ar.html](https://proceedings.mlr.press/v202/zhang23ar.html)

        **Abstract**:

        The inherent challenge of multimodal fusion is to precisely capture the cross-modal correlation and flexibly conduct cross-modal interaction. To fully release the value of each modality and mitigate the influence of low-quality multimodal data, dynamic multimodal fusion emerges as a promising learning paradigm. Despite its widespread use, theoretical justifications in this field are still notably lacking. Can we design a provably robust multimodal fusion method? This paper provides theoretical understandings to answer this question under a most popular multimodal fusion framework from the generalization perspective. We proceed to reveal that several uncertainty estimation solutions are naturally available to achieve robust multimodal fusion. Then a novel multimodal fusion framework termed Quality-aware Multimodal Fusion (QMF) is proposed, which can improve the performance in terms of classification accuracy and model robustness. Extensive experimental results on multiple benchmarks can support our findings.

        ----

        ## [1753] ReDi: Efficient Learning-Free Diffusion Inference via Trajectory Retrieval

        **Authors**: *Kexun Zhang, Xianjun Yang, William Yang Wang, Lei Li*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/zhang23as.html](https://proceedings.mlr.press/v202/zhang23as.html)

        **Abstract**:

        Diffusion models show promising generation capability for a variety of data. Despite their high generation quality, the inference for diffusion models is still time-consuming due to the numerous sampling iterations required. To accelerate the inference, we propose ReDi, a simple yet learning-free Retrieval-based Diffusion sampling framework. From a precomputed knowledge base, ReDi retrieves a trajectory similar to the partially generated trajectory at an early stage of generation, skips a large portion of intermediate steps, and continues sampling from a later step in the retrieved trajectory. We theoretically prove that the generation performance of ReDi is guaranteed. Our experiments demonstrate that ReDi improves the model inference efficiency by 2$\times$ speedup. Furthermore, ReDi is able to generalize well in zero-shot cross-domain image generation such as image stylization. The code and demo for ReDi is available at https://github.com/zkx06111/ReDiffusion.

        ----

        ## [1754] Nearly Optimal Competitive Ratio for Online Allocation Problems with Two-sided Resource Constraints and Finite Requests

        **Authors**: *Qixin Zhang, Wenbing Ye, Zaiyi Chen, Haoyuan Hu, Enhong Chen, Yu Yang*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/zhang23at.html](https://proceedings.mlr.press/v202/zhang23at.html)

        **Abstract**:

        In this paper, we investigate the online allocation problem of maximizing the overall revenue subject to both lower and upper bound constraints. Compared to the extensively studied online problems with only resource upper bounds, the two-sided constraints affect the prospects of resource consumption more severely. As a result, only limited violations of constraints or pessimistic competitive bounds could be guaranteed. To tackle the challenge, we define a measure of feasibility $\xi^*$ to evaluate the hardness of this problem, and estimate this measurement by an optimization routine with theoretical guarantees. We propose an online algorithm adopting a constructive framework, where we initialize a threshold price vector using the estimation, then dynamically update the price vector and use it for decision-making at each step. It can be shown that the proposed algorithm is $\big(1-O(\frac{\varepsilon}{\xi^*-\varepsilon})\big)$ or $\big(1-O(\frac{\varepsilon}{\xi^*-\sqrt{\varepsilon}})\big)$ competitive with high probability for $\xi^*$ known or unknown respectively. To the best of our knowledge, this is the first result establishing a nearly optimal competitive algorithm for solving two-sided constrained online allocation problems with a high probability of feasibility.

        ----

        ## [1755] Do You Remember? Overcoming Catastrophic Forgetting for Fake Audio Detection

        **Authors**: *Xiaohui Zhang, Jiangyan Yi, Jianhua Tao, Chenglong Wang, Chu Yuan Zhang*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/zhang23au.html](https://proceedings.mlr.press/v202/zhang23au.html)

        **Abstract**:

        Current fake audio detection algorithms have achieved promising performances on most datasets. However, their performance may be significantly degraded when dealing with audio of a different dataset. The orthogonal weight modification to overcome catastrophic forgetting does not consider the similarity of genuine audio across different datasets. To overcome this limitation, we propose a continual learning algorithm for fake audio detection to overcome catastrophic forgetting, called Regularized Adaptive Weight Modification (RAWM). When fine-tuning a detection network, our approach adaptively computes the direction of weight modification according to the ratio of genuine utterances and fake utterances. The adaptive modification direction ensures the network can effectively detect fake audio on the new dataset while preserving its knowledge of old model, thus mitigating catastrophic forgetting. In addition, genuine audio collected from quite different acoustic conditions may skew their feature distribution, so we introduce a regularization constraint to force the network to remember the old distribution in this regard. Our method can easily be generalized to related fields, like speech emotion recognition. We also evaluate our approach across multiple datasets and obtain a significant performance improvement on cross-dataset experiments.

        ----

        ## [1756] Coder Reviewer Reranking for Code Generation

        **Authors**: *Tianyi Zhang, Tao Yu, Tatsunori Hashimoto, Mike Lewis, Wen-Tau Yih, Daniel Fried, Sida Wang*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/zhang23av.html](https://proceedings.mlr.press/v202/zhang23av.html)

        **Abstract**:

        Sampling diverse programs from a code language model and reranking with model likelihood is a popular method for code generation but it is prone to preferring degenerate solutions. Inspired by collaborative programming, we propose Coder-Reviewer reranking. We augment Coder language models from past work, which generate programs given language instructions, with Reviewer models, which evaluate the likelihood of the instruction given the generated programs. We perform an extensive study across six datasets with eight models from three model families. Experimental results show that Coder-Reviewer reranking leads to consistent and significant improvement (up to 17% absolute accuracy gain) over reranking with the Coder model only. When combined with executability filtering, Coder-Reviewer reranking can often outperform the minimum Bayes risk method. Coder-Reviewer reranking is easy to implement by prompting, can generalize to different programming languages, and works well with off-the-shelf hyperparameters.

        ----

        ## [1757] DP-Fast MH: Private, Fast, and Accurate Metropolis-Hastings for Large-Scale Bayesian Inference

        **Authors**: *Wanrong Zhang, Ruqi Zhang*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/zhang23aw.html](https://proceedings.mlr.press/v202/zhang23aw.html)

        **Abstract**:

        Bayesian inference provides a principled framework for learning from complex data and reasoning under uncertainty. It has been widely applied in machine learning tasks such as medical diagnosis, drug design, and policymaking. In these common applications, data can be highly sensitive. Differential privacy (DP) offers data analysis tools with powerful worst-case privacy guarantees and has been developed as the leading approach in privacy-preserving data analysis. In this paper, we study Metropolis-Hastings (MH), one of the most fundamental MCMC methods, for large-scale Bayesian inference under differential privacy. While most existing private MCMC algorithms sacrifice accuracy and efficiency to obtain privacy, we provide the first exact and fast DP MH algorithm, using only a minibatch of data in most iterations. We further reveal, for the first time, a three-way trade-off among privacy, scalability (i.e. the batch size), and efficiency (i.e. the convergence rate), theoretically characterizing how privacy affects the utility and computational cost in Bayesian inference. We empirically demonstrate the effectiveness and efficiency of our algorithm in various experiments.

        ----

        ## [1758] Nearly-tight Bounds for Deep Kernel Learning

        **Authors**: *Yifan Zhang, Min-Ling Zhang*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/zhang23ax.html](https://proceedings.mlr.press/v202/zhang23ax.html)

        **Abstract**:

        The generalization analysis of deep kernel learning (DKL) is a crucial and open problem of kernel methods for deep learning. The implicit nonlinear mapping in DKL makes existing methods of capacity-based generalization analysis for deep learning invalid. In an attempt to overcome this challenge and make up for the gap in the generalization theory of DKL, we develop an analysis method based on the composite relationship of function classes and derive capacity-based bounds with mild dependence on the depth, which generalizes learning theory bounds to deep kernels and serves as theoretical guarantees for the generalization of DKL. In this paper, we prove novel and nearly-tight generalization bounds based on the uniform covering number and the Rademacher chaos complexity for deep (multiple) kernel machines. In addition, for some common classes, we estimate their uniform covering numbers and Rademacher chaos complexities by bounding their pseudo-dimensions and kernel pseudo-dimensions, respectively. The mild bounds without strong assumptions partially explain the good generalization ability of deep learning combined with kernel methods.

        ----

        ## [1759] OpenFE: Automated Feature Generation with Expert-level Performance

        **Authors**: *Tianping Zhang, Zheyu Aqa Zhang, Zhiyuan Fan, Haoyan Luo, Fengyuan Liu, Qian Liu, Wei Cao, Li Jian*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/zhang23ay.html](https://proceedings.mlr.press/v202/zhang23ay.html)

        **Abstract**:

        The goal of automated feature generation is to liberate machine learning experts from the laborious task of manual feature generation, which is crucial for improving the learning performance of tabular data. The major challenge in automated feature generation is to efficiently and accurately identify effective features from a vast pool of candidate features. In this paper, we present OpenFE, an automated feature generation tool that provides competitive results against machine learning experts. OpenFE achieves high efficiency and accuracy with two components: 1) a novel feature boosting method for accurately evaluating the incremental performance of candidate features and 2) a two-stage pruning algorithm that performs feature pruning in a coarse-to-fine manner. Extensive experiments on ten benchmark datasets show that OpenFE outperforms existing baseline methods by a large margin. We further evaluate OpenFE in two Kaggle competitions with thousands of data science teams participating. In the two competitions, features generated by OpenFE with a simple baseline model can beat 99.3% and 99.6% data science teams respectively. In addition to the empirical results, we provide a theoretical perspective to show that feature generation can be beneficial in a simple yet representative setting.

        ----

        ## [1760] Optimal Horizon-Free Reward-Free Exploration for Linear Mixture MDPs

        **Authors**: *Junkai Zhang, Weitong Zhang, Quanquan Gu*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/zhang23az.html](https://proceedings.mlr.press/v202/zhang23az.html)

        **Abstract**:

        We study reward-free reinforcement learning (RL) with linear function approximation, where the agent works in two phases: (1) in the exploration phase, the agent interacts with the environment but cannot access the reward; and (2) in the planning phase, the agent is given a reward function and is expected to find a near-optimal policy based on samples collected in the exploration phase. The sample complexities of existing reward-free algorithms have a polynomial dependence on the planning horizon, which makes them intractable for long planning horizon RL problems. In this paper, we propose a new reward-free algorithm for learning linear mixture Markov decision processes (MDPs), where the transition probability can be parameterized as a linear combination of known feature mappings. At the core of our algorithm is uncertainty-weighted value-targeted regression with exploration-driven pseudo-reward and a high-order moment estimator for the aleatoric and epistemic uncertainties. When the total reward is bounded by $1$, we show that our algorithm only needs to explore $\tilde O\left( d^2\varepsilon^{-2}\right)$ episodes to find an $\varepsilon$-optimal policy, where $d$ is the dimension of the feature mapping. The sample complexity of our algorithm only has a polylogarithmic dependence on the planning horizon and therefore is "horizon-free”. In addition, we provide an $\Omega\left(d^2\varepsilon^{-2}\right)$ sample complexity lower bound, which matches the sample complexity of our algorithm up to logarithmic factors, suggesting that our algorithm is optimal.

        ----

        ## [1761] Unlocking Slot Attention by Changing Optimal Transport Costs

        **Authors**: *Yan Zhang, David W. Zhang, Simon Lacoste-Julien, Gertjan J. Burghouts, Cees G. M. Snoek*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/zhang23ba.html](https://proceedings.mlr.press/v202/zhang23ba.html)

        **Abstract**:

        Slot attention is a powerful method for object-centric modeling in images and videos. However, its set-equivariance limits its ability to handle videos with a dynamic number of objects because it cannot break ties. To overcome this limitation, we first establish a connection between slot attention and optimal transport. Based on this new perspective we propose MESH (Minimize Entropy of Sinkhorn): a cross-attention module that combines the tiebreaking properties of unregularized optimal transport with the speed of regularized optimal transport. We evaluate slot attention using MESH on multiple object-centric learning benchmarks and find significant improvements over slot attention in every setting.

        ----

        ## [1762] Towards a Persistence Diagram that is Robust to Noise and Varied Densities

        **Authors**: *Hang Zhang, Kaifeng Zhang, Kai Ming Ting, Ye Zhu*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/zhang23bb.html](https://proceedings.mlr.press/v202/zhang23bb.html)

        **Abstract**:

        Recent works have identified that existing methods, which construct persistence diagrams in Topological Data Analysis (TDA), are not robust to noise and varied densities in a point cloud. We analyze the necessary properties of an approach that can address these two issues, and propose a new filter function for TDA based on a new data-dependent kernel which possesses these properties. Our empirical evaluation reveals that the proposed filter function provides a better means for t-SNE visualization and SVM classification than three existing methods of TDA.

        ----

        ## [1763] Robust Situational Reinforcement Learning in Face of Context Disturbances

        **Authors**: *Jinpeng Zhang, Yufeng Zheng, Chuheng Zhang, Li Zhao, Lei Song, Yuan Zhou, Jiang Bian*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/zhang23bc.html](https://proceedings.mlr.press/v202/zhang23bc.html)

        **Abstract**:

        In many real-world tasks, some parts of state features, called contexts, are independent of action signals, e.g., customer demand in inventory control, speed of lead car in autonomous driving, etc. One of the challenges of reinforcement learning in these applications is that the true context transitions can be easily exposed some unknown source of contamination, leading to a shift of context transitions between source domains and target domains, which could cause performance degradation for RL algorithms. However, existing methods on robust RL aim at learning robust policies against the deviations of the entire system dynamics. To tackle this problem, this paper proposes the framework of robust situational Markov decision process (RS-MDP) which captures the possible deviations of context transitions explicitly. To scale to large context space, we introduce the softmin smoothed robust Bellman operator to learn the robust Q-value approximately, and apply our RS-MDP framework to existing RL algorithm SAC to learn the desired robust policies. We conduct experiments on several robot control tasks with dynamic contexts and inventory control tasks to demonstrate that our algorithm can generalize better and more robust against deviations of context transitions, and outperform existing robust RL algorithms.

        ----

        ## [1764] Patch-level Contrastive Learning via Positional Query for Visual Pre-training

        **Authors**: *Shaofeng Zhang, Qiang Zhou, Zhibin Wang, Fan Wang, Junchi Yan*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/zhang23bd.html](https://proceedings.mlr.press/v202/zhang23bd.html)

        **Abstract**:

        Dense contrastive learning (DCL) has been recently explored for learning localized information for dense prediction tasks (e.g., detection and segmentation). It still suffers the difficulty of mining pixels/patches correspondence between two views. A simple way is inputting the same view twice and aligning the pixel/patch representation. However, it would reduce the variance of inputs, and hurts the performance. We propose a plug-in method PQCL (Positional Query for patch-level Contrastive Learning), which allows performing patch-level contrasts between two views with exact patch correspondence. Besides, by using positional queries, PQCL increases the variance of inputs, to enhance training. We apply PQCL to popular transformer-based CL frameworks (DINO and iBOT, and evaluate them on classification, detection and segmentation tasks, where our method obtains stable improvements, especially for dense tasks. It achieves new state-of-the-art in most settings. Code is available at https://github.com/Sherrylone/Query_Contrastive.

        ----

        ## [1765] Men Also Do Laundry: Multi-Attribute Bias Amplification

        **Authors**: *Dora Zhao, Jerone Theodore Alexander Andrews, Alice Xiang*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/zhao23a.html](https://proceedings.mlr.press/v202/zhao23a.html)

        **Abstract**:

        The phenomenon of $\textit{bias amplification}$ occurs when models amplify training set biases at test time. Existing metrics measure bias amplification with respect to single annotated attributes (e.g., $\texttt{computer}$). However, large-scale datasets typically consist of instances with multiple attribute annotations (e.g., $\{\texttt{computer}, \texttt{keyboard}\}$). We demonstrate models can learn to exploit correlations with respect to multiple attributes, which are not accounted for by current metrics. Moreover, we show that current metrics can give the erroneous impression that little to no bias amplification has occurred as they aggregate positive and negative bias scores. Further, these metrics lack an ideal value, making them difficult to interpret. To address these shortcomings, we propose a new metric: $\textit{Multi-Attribute Bias Amplification}$. We validate our metric’s utility through a bias amplification analysis on the COCO, imSitu, and CelebA datasets. Finally, we benchmark bias mitigation methods using our proposed metric, suggesting possible avenues for future bias mitigation efforts.

        ----

        ## [1766] Rockmate: an Efficient, Fast, Automatic and Generic Tool for Re-materialization in PyTorch

        **Authors**: *Xunyi Zhao, Théotime Le Hellard, Lionel Eyraud-Dubois, Julia Gusak, Olivier Beaumont*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/zhao23b.html](https://proceedings.mlr.press/v202/zhao23b.html)

        **Abstract**:

        We propose Rockmate to control the memory requirements when training PyTorch DNN models. Rockmate is an automatic tool that starts from the model code and generates an equivalent model, using a predefined amount of memory for activations, at the cost of a few re-computations. Rockmate automatically detects the structure of computational and data dependencies and rewrites the initial model as a sequence of complex blocks. We show that such a structure is widespread and can be found in many models in the literature (Transformer based models, ResNet, RegNets,...). This structure allows us to solve the problem in a fast and efficient way, using an adaptation of Checkmate (too slow on the whole model but general) at the level of individual blocks and an adaptation of Rotor (fast but limited to sequential models) at the level of the sequence itself. We show through experiments on many models that Rockmate is as fast as Rotor and as efficient as Checkmate, and that it allows in many cases to obtain a significantly lower memory consumption for activations (by a factor of 2 to 5) for a rather negligible overhead (of the order of 10% to 20%). Rockmate is open source and available at https://github.com/topal-team/rockmate.

        ----

        ## [1767] Revisiting Structured Variational Autoencoders

        **Authors**: *Yixiu Zhao, Scott W. Linderman*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/zhao23c.html](https://proceedings.mlr.press/v202/zhao23c.html)

        **Abstract**:

        Structured variational autoencoders (SVAEs) combine probabilistic graphical model priors on latent variables, deep neural networks to link latent variables to observed data, and structure-exploiting algorithms for approximate posterior inference. These models are particularly appealing for sequential data, where the prior can capture temporal dependencies. However, despite their conceptual elegance, SVAEs have proven difficult to implement, and more general approaches have been favored in practice. Here, we revisit SVAEs using modern machine learning tools and demonstrate their advantages over more general alternatives in terms of both accuracy and efficiency. First, we develop a modern implementation for hardware acceleration, parallelization, and automatic differentiation of the message passing algorithms at the core of the SVAE. Second, we show that by exploiting structure in the prior, the SVAE learns more accurate models and posterior distributions, which translate into improved performance on prediction tasks. Third, we show how the SVAE can naturally handle missing data, and we leverage this ability to develop a novel, self-supervised training approach. Altogether, these results show that the time is ripe to revisit structured variational autoencoders.

        ----

        ## [1768] On Pitfalls of Test-Time Adaptation

        **Authors**: *Hao Zhao, Yuejiang Liu, Alexandre Alahi, Tao Lin*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/zhao23d.html](https://proceedings.mlr.press/v202/zhao23d.html)

        **Abstract**:

        Test-Time Adaptation (TTA) has recently gained significant attention as a new paradigm for tackling distribution shifts. Despite the sheer number of existing methods, the inconsistent experimental conditions and lack of standardization in prior literature make it difficult to measure their actual efficacies and progress. To address this issue, we present a large-scale open-sourced Test-Time Adaptation Benchmark, dubbed TTAB, which includes nine state-of-the-art algorithms, a diverse array of distribution shifts, and two comprehensive evaluation protocols. Through extensive experiments, we identify three common pitfalls in prior efforts: (i) choosing appropriate hyper-parameter, especially for model selection, is exceedingly difficult due to online batch dependency; (ii) the effectiveness of TTA varies greatly depending on the quality of the model being adapted; (iii) even under optimal algorithmic conditions, existing methods still systematically struggle with certain types of distribution shifts. Our findings suggest that future research in the field should be more transparent about their experimental conditions, ensure rigorous evaluations on a broader set of models and shifts, and re-examine the assumptions underlying the potential success of TTA for practical applications.

        ----

        ## [1769] Addressing Budget Allocation and Revenue Allocation in Data Market Environments Using an Adaptive Sampling Algorithm

        **Authors**: *Boxin Zhao, Boxiang Lyu, Raul Castro Fernandez, Mladen Kolar*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/zhao23e.html](https://proceedings.mlr.press/v202/zhao23e.html)

        **Abstract**:

        High-quality machine learning models are dependent on access to high-quality training data. When the data are not already available, it is tedious and costly to obtain them. Data markets help with identifying valuable training data: model consumers pay to train a model, the market uses that budget to identify data and train the model (the budget allocation problem), and finally the market compensates data providers according to their data contribution (revenue allocation problem). For example, a bank could pay the data market to access data from other financial institutions to train a fraud detection model. Compensating data contributors requires understanding data’s contribution to the model; recent efforts to solve this revenue allocation problem based on the Shapley value are inefficient to lead to practical data markets. In this paper, we introduce a new algorithm to solve budget allocation and revenue allocation problems simultaneously in linear time. The new algorithm employs an adaptive sampling process that selects data from those providers who are contributing the most to the model. Better data means that the algorithm accesses those providers more often, and more frequent accesses corresponds to higher compensation. Furthermore, the algorithm can be deployed in both centralized and federated scenarios, boosting its applicability. We provide theoretical guarantees for the algorithm that show the budget is used efficiently and the properties of revenue allocation are similar to Shapley’s. Finally, we conduct an empirical evaluation to show the performance of the algorithm in practical scenarios and when compared to other baselines. Overall, we believe that the new algorithm paves the way for the implementation of practical data markets.

        ----

        ## [1770] X-Paste: Revisiting Scalable Copy-Paste for Instance Segmentation using CLIP and StableDiffusion

        **Authors**: *Hanqing Zhao, Dianmo Sheng, Jianmin Bao, Dongdong Chen, Dong Chen, Fang Wen, Lu Yuan, Ce Liu, Wenbo Zhou, Qi Chu, Weiming Zhang, Nenghai Yu*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/zhao23f.html](https://proceedings.mlr.press/v202/zhao23f.html)

        **Abstract**:

        Copy-Paste is a simple and effective data augmentation strategy for instance segmentation. By randomly pasting object instances onto new background images, it creates new training data for free and significantly boosts the segmentation performance, especially for rare object categories. Although diverse, high-quality object instances used in Copy-Paste result in more performance gain, previous works utilize object instances either from human-annotated instance segmentation datasets or rendered from 3D object models, and both approaches are too expensive to scale up to obtain good diversity. In this paper, we revisit Copy-Paste at scale with the power of newly emerged zero-shot recognition models (e.g., CLIP) and text2image models (e.g., StableDiffusion). We demonstrate for the first time that using a text2image model to generate images or zero-shot recognition model to filter noisily crawled images for different object categories is a feasible way to make Copy-Paste truly scalable. To make such success happen, we design a data acquisition and processing framework, dubbed “X-Paste", upon which a systematic study is conducted. On the LVIS dataset, X-Paste provides impressive improvements over the strong baseline CenterNet2 with Swin-L as the backbone. Specifically, it archives +2.6 box AP and +2.1 mask AP gains on all classes and even more significant gains with +6.8 box AP +6.5 mask AP on long-tail classes.

        ----

        ## [1771] Revisiting Simple Regret: Fast Rates for Returning a Good Arm

        **Authors**: *Yao Zhao, Connor Stephens, Csaba Szepesvári, Kwang-Sung Jun*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/zhao23g.html](https://proceedings.mlr.press/v202/zhao23g.html)

        **Abstract**:

        Simple regret is a natural and parameter-free performance criterion for pure exploration in multi-armed bandits yet is less popular than the probability of missing the best arm or an $\epsilon$-good arm, perhaps due to lack of easy ways to characterize it. In this paper, we make a significant progress on minimizing simple regret in both data-rich ($T\ge n$) and data-poor regime ($T \le n$) where $n$ is the number of arms and $T$ is the number of samples. At its heart is our improved instance-dependent analysis of the well-known Sequential Halving (SH) algorithm where we bound the probability of returning an arm whose mean reward is not within $\epsilon$ from the best (i.e., not $\epsilon$-good) for any choice of $\epsilon>0$, although $\epsilon$ is not an input to SH. Our bound not only leads to an optimal worst-case simple regret bound of $\sqrt{n/T}$ up to logarithmic factors but also essentially matches the instance-dependent lower bound for returning an $\epsilon$-good arm reported by Katz-Samuels and Jamieson (2020). For the more challenging data-poor regime, we propose Bracketing SH (BSH) that enjoys the same improvement even without sampling each arm at least once. Our empirical study shows that BSH outperforms existing methods on real-world tasks.

        ----

        ## [1772] Transformed Distribution Matching for Missing Value Imputation

        **Authors**: *He Zhao, Ke Sun, Amir Dezfouli, Edwin V. Bonilla*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/zhao23h.html](https://proceedings.mlr.press/v202/zhao23h.html)

        **Abstract**:

        We study the problem of imputing missing values in a dataset, which has important applications in many domains. The key to missing value imputation is to capture the data distribution with incomplete samples and impute the missing values accordingly. In this paper, by leveraging the fact that any two batches of data with missing values come from the same data distribution, we propose to impute the missing values of two batches of samples by transforming them into a latent space through deep invertible functions and matching them distributionally. To learn the transformations and impute the missing values simultaneously, a simple and well-motivated algorithm is proposed. Our algorithm has fewer hyperparameters to fine-tune and generates high-quality imputations regardless of how missing values are generated. Extensive experiments over a large number of datasets and competing benchmark algorithms show that our method achieves state-of-the-art performance.

        ----

        ## [1773] Protecting Language Generation Models via Invisible Watermarking

        **Authors**: *Xuandong Zhao, Yu-Xiang Wang, Lei Li*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/zhao23i.html](https://proceedings.mlr.press/v202/zhao23i.html)

        **Abstract**:

        Language generation models have been an increasingly powerful enabler to many applications. Many such models offer free or affordable API access which makes them potentially vulnerable to model extraction attacks through distillation. To protect intellectual property (IP) and make fair use of these models, various techniques such as lexical watermarking and synonym replacement have been proposed. However, these methods can be nullified by obvious countermeasures such as “synonym randomization”. To address this issue, we propose GINSW, a novel method to protect text generation models from being stolen through distillation. The key idea of our method is to inject secret signals into the probability vector of the decoding steps for each target token. We can then detect the secret message by probing a suspect model to tell if it is distilled from the protected one. Experimental results show that GINSW can effectively identify instances of IP infringement with minimal impact on the generation quality of protected APIs. Our method demonstrates an absolute improvement of 19 to 29 points on mean average precision (mAP) in detecting suspects compared to previous methods against watermark removal attacks.

        ----

        ## [1774] Local Optimization Achieves Global Optimality in Multi-Agent Reinforcement Learning

        **Authors**: *Yulai Zhao, Zhuoran Yang, Zhaoran Wang, Jason D. Lee*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/zhao23j.html](https://proceedings.mlr.press/v202/zhao23j.html)

        **Abstract**:

        Policy optimization methods with function approximation are widely used in multi-agent reinforcement learning. However, it remains elusive how to design such algorithms with statistical guarantees. Leveraging a multi-agent performance difference lemma that characterizes the landscape of multi-agent policy optimization, we find that the localized action value function serves as an ideal descent direction for each local policy. Motivated by the observation, we present a multi-agent PPO algorithm in which the local policy of each agent is updated similarly to vanilla PPO. We prove that with standard regularity conditions on the Markov game and problem-dependent quantities, our algorithm converges to the globally optimal policy at a sublinear rate. We extend our algorithm to the off-policy setting and introduce pessimism to policy evaluation, which aligns with experiments. To our knowledge, this is the first provably convergent multi-agent PPO algorithm in cooperative Markov games.

        ----

        ## [1775] Simplified Temporal Consistency Reinforcement Learning

        **Authors**: *Yi Zhao, Wenshuai Zhao, Rinu Boney, Juho Kannala, Joni Pajarinen*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/zhao23k.html](https://proceedings.mlr.press/v202/zhao23k.html)

        **Abstract**:

        Reinforcement learning (RL) is able to solve complex sequential decision-making tasks but is currently limited by sample efficiency and required computation. To improve sample efficiency, recent work focuses on model-based RL which interleaves model learning with planning. Recent methods further utilize policy learning, value estimation, and, self-supervised learning as auxiliary objectives. In this paper we show that, surprisingly, a simple representation learning approach relying only on a latent dynamics model trained by latent temporal consistency is sufficient for high-performance RL. This applies when using pure planning with a dynamics model conditioned on the representation, but, also when utilizing the representation as policy and value function features in model-free RL. In experiments, our approach learns an accurate dynamics model to solve challenging high-dimensional locomotion tasks with online planners while being 4.1$\times$ faster to train compared to ensemble-based methods. With model-free RL without planning, especially on high-dimensional tasks, such as the Deepmind Control Suite Humanoid and Dog tasks, our approach outperforms model-free methods by a large margin and matches model-based methods’ sample efficiency while training 2.4$\times$ faster.

        ----

        ## [1776] RLEG: Vision-Language Representation Learning with Diffusion-based Embedding Generation

        **Authors**: *Liming Zhao, Kecheng Zheng, Yun Zheng, Deli Zhao, Jingren Zhou*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/zhao23l.html](https://proceedings.mlr.press/v202/zhao23l.html)

        **Abstract**:

        Vision-language representation learning models (e.g., CLIP) have achieved state-of-the-art performance on various downstream tasks, which usually need large-scale training data to learn discriminative representation. Recent progress on generative diffusion models (e.g., DALL-E 2) has demonstrated that diverse high-quality samples can be synthesized by randomly sampling from generative distribution. By virtue of generative capability in this paper, we propose a novel vision-language Representation Learning method with diffusion-based Embedding Generation (RLEG), which exploits diffusion models to generate feature embedding online for learning effective vision-language representation. Specifically, we first adopt image and text encoders to extract the corresponding embeddings. Secondly, pretrained diffusion-based embedding generators are harnessed to transfer the embedding modality online between vision and language domains. The embeddings generated from the generators are then served as augmented embedding-level samples, which are applied to contrastive learning with the variant of the CLIP framework. Experimental results show that the proposed method could learn effective representation and achieve state-of-the-art performance on various tasks including image classification, image-text retrieval, object detection, semantic segmentation, and text-conditional image generation.

        ----

        ## [1777] Optimal Online Generalized Linear Regression with Stochastic Noise and Its Application to Heteroscedastic Bandits

        **Authors**: *Heyang Zhao, Dongruo Zhou, Jiafan He, Quanquan Gu*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/zhao23m.html](https://proceedings.mlr.press/v202/zhao23m.html)

        **Abstract**:

        We study the problem of online generalized linear regression in the stochastic setting, where the label is generated from a generalized linear model with possibly unbounded additive noise. We provide a sharp analysis of the classical follow-the-regularized-leader (FTRL) algorithm to cope with the label noise. More specifically, for $\sigma$-sub-Gaussian label noise, our analysis provides a regret upper bound of $O(\sigma^2 d \log T) + o(\log T)$, where $d$ is the dimension of the input vector, $T$ is the total number of rounds. We also prove an $\Omega(\sigma^2d\log(T/d))$ lower bound for stochastic online linear regression, which indicates that our upper bound is nearly optimal. In addition, we extend our analysis to a more refined Bernstein noise condition. As an application, we study generalized linear bandits with heterogeneous noise and propose an algorithm based on FTRL to achieve the first variance-aware regret bound.

        ----

        ## [1778] Does Continual Learning Equally Forget All Parameters?

        **Authors**: *Haiyan Zhao, Tianyi Zhou, Guodong Long, Jing Jiang, Chengqi Zhang*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/zhao23n.html](https://proceedings.mlr.press/v202/zhao23n.html)

        **Abstract**:

        Distribution shift (e.g., task or domain shift) in continual learning (CL) usually results in catastrophic forgetting of previously learned knowledge. Although it can be alleviated by repeatedly replaying buffered data, the every-step replay is time-consuming. In this paper, we study which modules in neural networks are more prone to forgetting by investigating their training dynamics during CL. Our proposed metrics show that only a few modules are more task-specific and sensitive to task change, while others can be shared across tasks as common knowledge. Hence, we attribute forgetting mainly to the former and find that finetuning them only on a small buffer at the end of any CL method can bring non-trivial improvement. Due to the small number of finetuned parameters, such ”Forgetting Prioritized Finetuning (FPF)” is efficient in computation. We further propose a more efficient and simpler method that entirely removes the every-step replay and replaces them by only $k$-times of FPF periodically triggered during CL. Surprisingly, this ”$k$-FPF” performs comparably to FPF and outperforms the SOTA CL methods but significantly reduces their computational overhead and cost. In experiments on several benchmarks of class- and domain-incremental CL, FPF consistently improves existing CL methods by a large margin, and $k$-FPF further excels in efficiency without degrading the accuracy. We also empirically studied the impact of buffer size, epochs per task, and finetuning modules on the cost and accuracy of our methods.

        ----

        ## [1779] Online Learning in Stackelberg Games with an Omniscient Follower

        **Authors**: *Geng Zhao, Banghua Zhu, Jiantao Jiao, Michael I. Jordan*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/zhao23o.html](https://proceedings.mlr.press/v202/zhao23o.html)

        **Abstract**:

        We study the problem of online learning in a two-player decentralized cooperative Stackelberg game. In each round, the leader first takes an action, followed by the follower who takes their action after observing the leader’s move. The goal of the leader is to learn to minimize the cumulative regret based on the history of interactions. Differing from the traditional formulation of repeated Stackelberg games, we assume the follower is omniscient, with full knowledge of the true reward, and that they always best-respond to the leader’s actions. We analyze the sample complexity of regret minimization in this repeated Stackelberg game. We show that depending on the reward structure, the existence of the omniscient follower may change the sample complexity drastically, from constant to exponential, even for linear cooperative Stackelberg games. This poses unique challenges for the learning process of the leader and the subsequent regret analysis.

        ----

        ## [1780] Structure-informed Language Models Are Protein Designers

        **Authors**: *Zaixiang Zheng, Yifan Deng, Dongyu Xue, Yi Zhou, Fei Ye, Quanquan Gu*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/zheng23a.html](https://proceedings.mlr.press/v202/zheng23a.html)

        **Abstract**:

        This paper demonstrates that language models are strong structure-based protein designers. We present LM-Design, a generic approach to reprogramming sequence-based protein language models (pLMs), that have learned massive sequential evolutionary knowledge from the universe of natural protein sequences, to acquire an immediate capability to design preferable protein sequences for given folds. We conduct a structural surgery on pLMs, where a lightweight structural adapter is implanted into pLMs and endows it with structural awareness. During inference, iterative refinement is performed to effectively optimize the generated protein sequences. Experiments show that LM-Design improves the state-of-the-art results by a large margin, leading to 4% to 12% accuracy gains in sequence recovery (e.g., 55.65%/56.63% on CATH 4.2/4.3 single-chain benchmarks, and $>$60% when designing protein complexes). We provide extensive and in-depth analyses, which verify that LM-Design can (1) indeed leverage both structural and sequential knowledge to accurately handle structurally non-deterministic regions, (2) benefit from scaling data and model size, and (3) generalize to other proteins (e.g., antibodies and de novo proteins).

        ----

        ## [1781] Semi-Supervised Offline Reinforcement Learning with Action-Free Trajectories

        **Authors**: *Qinqing Zheng, Mikael Henaff, Brandon Amos, Aditya Grover*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/zheng23b.html](https://proceedings.mlr.press/v202/zheng23b.html)

        **Abstract**:

        Natural agents can effectively learn from multiple data sources that differ in size, quality, and types of measurements. We study this heterogeneity in the context of offline reinforcement learning (RL) by introducing a new, practically motivated semi-supervised setting. Here, an agent has access to two sets of trajectories: labelled trajectories containing state, action and reward triplets at every timestep, along with unlabelled trajectories that contain only state and reward information. For this setting, we develop and study a simple meta-algorithmic pipeline that learns an inverse dynamics model on the labelled data to obtain proxy-labels for the unlabelled data, followed by the use of any offline RL algorithm on the true and proxy-labelled trajectories. Empirically, we find this simple pipeline to be highly successful — on several D4RL benchmarks (Fu et al., 2020), certain offline RL algorithms can match the performance of variants trained on a fully labelled dataset even when we label only 10% of trajectories which are highly suboptimal. To strengthen our understanding, we perform a large-scale controlled empirical study investigating the interplay of data-centric properties of the labelled and unlabelled datasets, with algorithmic design choices (e.g., choice of inverse dynamics, offline RL algorithm) to identify general trends and best practices for training RL agents on semi-supervised offline datasets.

        ----

        ## [1782] Improved Techniques for Maximum Likelihood Estimation for Diffusion ODEs

        **Authors**: *Kaiwen Zheng, Cheng Lu, Jianfei Chen, Jun Zhu*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/zheng23c.html](https://proceedings.mlr.press/v202/zheng23c.html)

        **Abstract**:

        Diffusion models have exhibited excellent performance in various domains. The probability flow ordinary differential equation (ODE) of diffusion models (i.e., diffusion ODEs) is a particular case of continuous normalizing flows (CNFs), which enables deterministic inference and exact likelihood evaluation. However, the likelihood estimation results by diffusion ODEs are still far from those of the state-of-the-art likelihood-based generative models. In this work, we propose several improved techniques for maximum likelihood estimation for diffusion ODEs, including both training and evaluation perspectives. For training, we propose velocity parameterization and explore variance reduction techniques for faster convergence. We also derive an error-bounded high-order flow matching objective for finetuning, which improves the ODE likelihood and smooths its trajectory. For evaluation, we propose a novel training-free truncated-normal dequantization to fill the training-evaluation gap commonly existing in diffusion ODEs. Building upon these techniques, we achieve state-of-the-art likelihood estimation results on image datasets (2.56 on CIFAR-10, 3.43/3.69 on ImageNet-32) without variational dequantization or data augmentation.

        ----

        ## [1783] Fast Sampling of Diffusion Models via Operator Learning

        **Authors**: *Hongkai Zheng, Weili Nie, Arash Vahdat, Kamyar Azizzadenesheli, Anima Anandkumar*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/zheng23d.html](https://proceedings.mlr.press/v202/zheng23d.html)

        **Abstract**:

        Diffusion models have found widespread adoption in various areas. However, their sampling process is slow because it requires hundreds to thousands of network evaluations to emulate a continuous process defined by differential equations. In this work, we use neural operators, an efficient method to solve the probability flow differential equations, to accelerate the sampling process of diffusion models. Compared to other fast sampling methods that have a sequential nature, we are the first to propose a parallel decoding method that generates images with only one model forward pass. We propose diffusion model sampling with neural operator (DSNO) that maps the initial condition, i.e., Gaussian distribution, to the continuous-time solution trajectory of the reverse diffusion process. To model the temporal correlations along the trajectory, we introduce temporal convolution layers that are parameterized in the Fourier space into the given diffusion model backbone. We show our method achieves state-of-the-art FID of 3.78 for CIFAR-10 and 7.83 for ImageNet-64 in the one-model-evaluation setting.

        ----

        ## [1784] Outline, Then Details: Syntactically Guided Coarse-To-Fine Code Generation

        **Authors**: *Wenqing Zheng, S. P. Sharan, Ajay Kumar Jaiswal, Kevin Wang, Yihan Xi, Dejia Xu, Zhangyang Wang*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/zheng23e.html](https://proceedings.mlr.press/v202/zheng23e.html)

        **Abstract**:

        For a complicated algorithm, its implementation by a human programmer usually starts with outlining a rough control flow followed by iterative enrichments, eventually yielding carefully generated syntactic structures and variables in a hierarchy. However, state-of-the-art large language models generate codes in a single pass, without intermediate warm-ups to reflect the structured thought process of "outline-then-detail". Inspired by the recent success of chain-of-thought prompting, we propose ChainCoder, a program synthesis language model that generates Python code progressively, i.e. from coarse to fine in multiple passes. We first decompose source code into layout frame components and accessory components via abstract syntax tree parsing to construct a hierarchical representation. We then reform our prediction target into a multi-pass objective, each pass generates a subsequence, which is concatenated in the hierarchy. Finally, a tailored transformer architecture is leveraged to jointly encode the natural language descriptions and syntactically aligned I/O data samples. Extensive evaluations show that ChainCoder outperforms state-of-the-arts, demonstrating that our progressive generation eases the reasoning procedure and guides the language model to generate higher-quality solutions. Our codes are available at: https://github.com/VITA-Group/ChainCoder.

        ----

        ## [1785] Revisiting Discriminative vs Generative Classifiers: Theory and Implications

        **Authors**: *Chenyu Zheng, Guoqiang Wu, Fan Bao, Yue Cao, Chongxuan Li, Jun Zhu*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/zheng23f.html](https://proceedings.mlr.press/v202/zheng23f.html)

        **Abstract**:

        A large-scale deep model pre-trained on massive labeled or unlabeled data transfers well to downstream tasks. Linear evaluation freezes parameters in the pre-trained model and trains a linear classifier separately, which is efficient and attractive for transfer. However, little work has investigated the classifier in linear evaluation except for the default logistic regression. Inspired by the statistical efficiency of naive Bayes, the paper revisits the classical topic on discriminative vs. generative classifiers. Theoretically, the paper considers the surrogate loss instead of the zero-one loss in analyses and generalizes the classical results from binary cases to multiclass ones. We show that, under mild assumptions, multiclass naive Bayes requires $O(\log n)$ samples to approach its asymptotic error while the corresponding multiclass logistic regression requires $O(n)$ samples, where $n$ is the feature dimension. To establish it, we present a multiclass $\mathcal{H}$-consistency bound framework and an explicit bound for logistic loss, which are of independent interests. Simulation results on a mixture of Gaussian validate our theoretical findings. Experiments on various pre-trained deep vision models show that naive Bayes consistently converges faster as the number of data increases. Besides, naive Bayes shows promise in few-shot cases and we observe the "two regimes” phenomenon in pre-trained supervised models. Our code is available at https://github.com/ML-GSAI/Revisiting-Dis-vs-Gen-Classifiers.

        ----

        ## [1786] Evidential Interactive Learning for Medical Image Captioning

        **Authors**: *Ervine Zheng, Qi Yu*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/zheng23g.html](https://proceedings.mlr.press/v202/zheng23g.html)

        **Abstract**:

        Medical image captioning alleviates the burden of physicians and possibly reduces medical errors by automatically generating text descriptions to describe image contents and convey findings. It is more challenging than conventional image captioning due to the complexity of medical images and the difficulty of aligning image regions with medical terms. In this paper, we propose an evidential interactive learning framework that leverages evidence-based uncertainty estimation and interactive machine learning to improve image captioning with limited labeled data. The interactive learning process involves three stages: keyword prediction, caption generation, and model retraining. First, the model predicts a list of keywords with evidence-based uncertainty and selects the most informative keywords to seek user feedback. Second, user-approved keywords are used as model input to guide the model to generate satisfactory captions. Third, the model is updated based on user-approved keywords and captions, where evidence-based uncertainty is used to allocate different weights to different data instances. Experiments on two medical image datasets illustrate that the proposed framework can effectively learn from human feedback and improve the model’s performance in the future.

        ----

        ## [1787] Finding the Missing-half: Graph Complementary Learning for Homophily-prone and Heterophily-prone Graphs

        **Authors**: *Yizhen Zheng, He Zhang, Vincent Cheng-Siong Lee, Yu Zheng, Xiao Wang, Shirui Pan*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/zheng23h.html](https://proceedings.mlr.press/v202/zheng23h.html)

        **Abstract**:

        Real-world graphs generally have only one kind of tendency in their connections. These connections are either homophilic-prone or heterophily-prone. While graphs with homophily-prone edges tend to connect nodes with the same class (i.e., intra-class nodes), heterophily-prone edges tend to build relationships between nodes with different classes (i.e., inter-class nodes). Existing GNNs only take the original graph as input during training. The problem with this approach is that it forgets to take into consideration the ”missing-half” structural information, that is, heterophily-prone topology for homophily-prone graphs and homophily-prone topology for heterophily-prone graphs. In our paper, we introduce Graph cOmplementAry Learning, namely GOAL, which consists of two components: graph complementation and complemented graph convolution. The first component finds the missing-half structural information for a given graph to complement it. The complemented graph has two sets of graphs including both homophily- and heterophily-prone topology. In the latter component, to handle complemented graphs, we design a new graph convolution from the perspective of optimisation. The experiment results show that GOAL consistently outperforms all baselines in eight real-world datasets.

        ----

        ## [1788] Multi-agent Online Scheduling: MMS Allocations for Indivisible Items

        **Authors**: *Shengwei Zhou, Rufan Bai, Xiaowei Wu*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/zhou23a.html](https://proceedings.mlr.press/v202/zhou23a.html)

        **Abstract**:

        We consider the problem of fairly allocating a sequence of indivisible items that arrive online in an arbitrary order to a group of $n$ agents with additive normalized valuation functions, we consider the allocation of goods and chores separately and propose algorithms for approximating maximin share (MMS) allocations for both settings. When agents have identical valuation functions the problem coincides with the semi-online machine covering problem (when items are goods) and load balancing problem (when items are chores), for both of which optimal competitive ratios have been achieved. In this paper we consider the case when agents have general additive valuation functions. For the allocation of goods we show that no competitive algorithm exists even when there are only three agents and propose an optimal $0.5$-competitive algorithm for the case of two agents. For the allocation of chores we propose a $(2-1/n)$-competitive algorithm for $n\geq 3$ agents and a $\sqrt{2}\approx 1.414$-competitive algorithm for two agents. Additionally, we show that no algorithm can do better than $15/11\approx 1.364$-competitive for two agents.

        ----

        ## [1789] Eliminating Adversarial Noise via Information Discard and Robust Representation Restoration

        **Authors**: *Dawei Zhou, Yukun Chen, Nannan Wang, Decheng Liu, Xinbo Gao, Tongliang Liu*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/zhou23b.html](https://proceedings.mlr.press/v202/zhou23b.html)

        **Abstract**:

        Deep neural networks (DNNs) are vulnerable to adversarial noise. Denoising model-based defense is a major protection strategy. However, denoising models may fail and induce negative effects in fully white-box scenarios. In this work, we start from the latent inherent properties of adversarial samples to break the limitations. Unlike solely learning a mapping from adversarial samples to natural samples, we aim to achieve denoising by destroying the spatial characteristics of adversarial noise and preserving the robust features of natural information. Motivated by this, we propose a defense based on information discard and robust representation restoration. Our method utilize complementary masks to disrupt adversarial noise and guided denoising models to restore robust-predictive representations from masked samples. Experimental results show that our method has competitive performance against white-box attacks and effectively reverses the negative effect of denoising models.

        ----

        ## [1790] Brainformers: Trading Simplicity for Efficiency

        **Authors**: *Yanqi Zhou, Nan Du, Yanping Huang, Daiyi Peng, Chang Lan, Da Huang, Siamak Shakeri, David R. So, Andrew M. Dai, Yifeng Lu, Zhifeng Chen, Quoc V. Le, Claire Cui, James Laudon, Jeff Dean*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/zhou23c.html](https://proceedings.mlr.press/v202/zhou23c.html)

        **Abstract**:

        Transformers are central to recent successes in natural language processing and computer vision. Transformers have a mostly uniform backbone where layers alternate between feed-forward and self-attention in order to build a deep network. Here we investigate this design choice and find that more complex blocks that have different permutations of layer primitives can be more efficient. Using this insight, we develop a complex block, named Brainformer, that consists of a diverse sets of layers such as sparsely gated feed-forward layers, dense feed-forward layers, attention layers, and various forms of layer normalization and activation functions. Brainformer consistently outperforms the state-of-the-art dense and sparse Transformers, in terms of both quality and efficiency. A Brainformer model with 8 billion activated parameters per token demonstrates 2x faster training convergence and 5x faster step time compared to its GLaM counterpart. In downstream task evaluation, Brainformer also demonstrates a 3% higher SuperGLUE score with fine-tuning compared to GLaM with a similar number of activated parameters. Finally, Brainformer largely outperforms a Primer dense model derived with NAS with similar computation per token on fewshot evaluations.

        ----

        ## [1791] Implicit Regularization Leads to Benign Overfitting for Sparse Linear Regression

        **Authors**: *Mo Zhou, Rong Ge*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/zhou23d.html](https://proceedings.mlr.press/v202/zhou23d.html)

        **Abstract**:

        In deep learning, often the training process finds an interpolator (a solution with 0 training loss), but the test loss is still low. This phenomenon, known as benign overfitting, is a major mystery that received a lot of recent attention. One common mechanism for benign overfitting is implicit regularization, where the training process leads to additional properties for the interpolator, often characterized by minimizing certain norms. However, even for a simple sparse linear regression problem $y = \beta^{\ast\top} x +\xi$ with sparse $\beta^{\ast}$, neither minimum $\ell_1$ or $\ell_2$ norm interpolator gives the optimal test loss. In this work, we give a different parametrization of the model which leads to a new implicit regularization effect that combines the benefit of $\ell_1$ and $\ell_2$ interpolators. We show that training our new model via gradient descent leads to an interpolator with near-optimal test loss. Our result is based on careful analysis of the training dynamics and provides another example of implicit regularization effect that goes beyond norm minimization.

        ----

        ## [1792] ODS: Test-Time Adaptation in the Presence of Open-World Data Shift

        **Authors**: *Zhi Zhou, Lan-Zhe Guo, Lin-Han Jia, Dingchu Zhang, Yu-Feng Li*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/zhou23e.html](https://proceedings.mlr.press/v202/zhou23e.html)

        **Abstract**:

        Test-time adaptation (TTA) adapts a source model to the distribution shift in testing data without using any source data. There have been plenty of algorithms concentrated on covariate shift in the last decade, i.e., $\mathcal{D}_t(X)$, the distribution of the test data is different from the source data. Nonetheless, in real application scenarios, it is necessary to consider the influence of label distribution shift, i.e., both $\mathcal{D}_t(X)$ and $\mathcal{D}_t(Y)$ are shifted, which has not been sufficiently explored yet. To remedy this, we study a new problem setup, namely, TTA with Open-world Data Shift (AODS). The goal of AODS is simultaneously adapting a model to covariate and label distribution shifts in the test phase. In this paper, we first analyze the relationship between classification error and distribution shifts. Motivated by this, we hence propose a new framework, namely ODS, which decouples the mixed distribution shift and then addresses covariate and label distribution shifts accordingly. We conduct experiments on multiple benchmarks with different types of shifts, and the results demonstrate the superior performance of our method against the state of the arts. Moreover, ODS is suitable for many TTA algorithms.

        ----

        ## [1793] Fourmer: An Efficient Global Modeling Paradigm for Image Restoration

        **Authors**: *Man Zhou, Jie Huang, Chun-Le Guo, Chongyi Li*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/zhou23f.html](https://proceedings.mlr.press/v202/zhou23f.html)

        **Abstract**:

        Global modeling-based image restoration frameworks have become popular. However, they often require a high memory footprint and do not consider task-specific degradation. Our work presents an alternative approach to global modeling that is more efficient for image restoration. The key insights which motivate our study are two-fold: 1) Fourier transform is capable of disentangling image degradation and content component to a certain extent, serving as the image degradation prior, and 2) Fourier domain innately embraces global properties, where each pixel in the Fourier space is involved with all spatial pixels. While adhering to the “spatial interaction + channel evolution” rule of previous studies, we customize the core designs with Fourier spatial interaction modeling and Fourier channel evolution. Our paradigm, Fourmer, achieves competitive performance on common image restoration tasks such as image de-raining, image enhancement, image dehazing, and guided image super-resolution, while requiring fewer computational resources. The code for Fourmer will be made publicly available.

        ----

        ## [1794] Controlled Text Generation with Natural Language Instructions

        **Authors**: *Wangchunshu Zhou, Yuchen Eleanor Jiang, Ethan Wilcox, Ryan Cotterell, Mrinmaya Sachan*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/zhou23g.html](https://proceedings.mlr.press/v202/zhou23g.html)

        **Abstract**:

        Large language models can be prompted to pro- duce fluent output for a wide range of tasks without being specifically trained to do so. Nevertheless, it is notoriously difficult to control their generation in such a way that it satisfies user-specified constraints. In this paper, we present InstructCTG, a simple controlled text generation framework that incorporates different constraints by verbalizing them as natural language instructions. We annotate natural texts through a combination of off-the-shelf NLP tools and simple heuristics with the linguistic and extra-linguistic constraints they satisfy. Then, we verbalize the constraints into natural language instructions to form weakly supervised training data, i.e., we prepend the natural language verbalizations of the constraints in front of their corresponding natural language sentences. Next, we fine-tune a pre-trained language model on the augmented corpus. Compared to existing methods, InstructCTG is more flexible in terms of the types of constraints it allows the practitioner to use. It also does not require any modification of the decoding procedure. Finally, InstructCTG allows the model to adapt to new constraints without re-training through the use of in-context learning.

        ----

        ## [1795] NNSplitter: An Active Defense Solution for DNN Model via Automated Weight Obfuscation

        **Authors**: *Tong Zhou, Yukui Luo, Shaolei Ren, Xiaolin Xu*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/zhou23h.html](https://proceedings.mlr.press/v202/zhou23h.html)

        **Abstract**:

        As a type of valuable intellectual property (IP), deep neural network (DNN) models have been protected by techniques like watermarking. However, such passive model protection cannot fully prevent model abuse. In this work, we propose an active model IP protection scheme, namely NNSplitter, which actively protects the model by splitting it into two parts: the obfuscated model that performs poorly due to weight obfuscation, and the model secrets consisting of the indexes and original values of the obfuscated weights, which can only be accessed by authorized users with the support of the trusted execution environment. Experimental results demonstrate the effectiveness of NNSplitter, e.g., by only modifying 275 out of over 11 million (i.e., 0.002%) weights, the accuracy of the obfuscated ResNet-18 model on CIFAR-10 can drop to 10%. Moreover, NNSplitter is stealthy and resilient against norm clipping and fine-tuning attacks, making it an appealing solution for DNN model protection. The code is available at: https://github.com/Tongzhou0101/NNSplitter.

        ----

        ## [1796] Deep Latent State Space Models for Time-Series Generation

        **Authors**: *Linqi Zhou, Michael Poli, Winnie Xu, Stefano Massaroli, Stefano Ermon*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/zhou23i.html](https://proceedings.mlr.press/v202/zhou23i.html)

        **Abstract**:

        Methods based on ordinary differential equations (ODEs) are widely used to build generative models of time-series. In addition to high computational overhead due to explicitly computing hidden states recurrence, existing ODE-based models fall short in learning sequence data with sharp transitions - common in many real-world systems - due to numerical challenges during optimization. In this work, we propose LS4, a generative model for sequences with latent variables evolving according to a state space ODE to increase modeling capacity. Inspired by recent deep state space models (S4), we achieve speedups by leveraging a convolutional representation of LS4 which bypasses the explicit evaluation of hidden states. We show that LS4 significantly outperforms previous continuous-time generative models in terms of marginal distribution, classification, and prediction scores on real-world datasets in the Monash Forecasting Repository, and is capable of modeling highly stochastic data with sharp temporal transitions. LS4 sets state-of-the-art for continuous-time latent generative models, with significant improvement of mean squared error and tighter variational lower bounds on irregularly-sampled datasets, while also being x100 faster than other baselines on long sequences.

        ----

        ## [1797] SlotGAT: Slot-based Message Passing for Heterogeneous Graphs

        **Authors**: *Ziang Zhou, Jieming Shi, Renchi Yang, Yuanhang Zou, Qing Li*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/zhou23j.html](https://proceedings.mlr.press/v202/zhou23j.html)

        **Abstract**:

        Heterogeneous graphs are ubiquitous to model complex data. There are urgent needs on powerful heterogeneous graph neural networks to effectively support important applications. We identify a potential semantic mixing issue in existing message passing processes, where the representations of the neighbors of a node v are forced to be transformed to the feature space of v for aggregation, though the neighbors are in different types. That is, the semantics in different node types are entangled together into node v’s representation. To address the issue, we propose SlotGAT with separate message passing processes in slots, one for each node type, to maintain the representations in their own node-type feature spaces. Moreover, in a slot-based message passing layer, we design an attention mechanism for effective slot-wise message aggregation. Further, we develop a slot attention technique after the last layer of SlotGAT, to learn the importance of different slots in downstream tasks. Our analysis indicates that the slots in SlotGAT can preserve different semantics in various feature spaces. The superiority of SlotGAT is evaluated against 13 baselines on 6 datasets for node classification and link prediction. Our code is at https://github.com/scottjiao/SlotGAT_ICML23/.

        ----

        ## [1798] Fast Online Node Labeling for Very Large Graphs

        **Authors**: *Baojian Zhou, Yifan Sun, Reza Babanezhad Harikandeh*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/zhou23k.html](https://proceedings.mlr.press/v202/zhou23k.html)

        **Abstract**:

        This paper studies the online node classification problem under a transductive learning setting. Current methods either invert a graph kernel matrix with $\mathcal{O}(n^3)$ runtime and $\mathcal{O}(n^2)$ space complexity or sample a large volume of random spanning trees, thus are difficult to scale to large graphs. In this work, we propose an improvement based on the online relaxation technique introduced by a series of works (Rakhlin et al., 2012; Rakhlin & Sridharan, 2015; 2017). We first prove an effective regret $\mathcal{O}(\sqrt{n^{1+\gamma}})$ when suitable parameterized graph kernels are chosen, then propose an approximate algorithm FastONL enjoying $\mathcal{O}(k\sqrt{n^{1+\gamma}})$ regret based on this relaxation. The key of FastONL is a generalized local push method that effectively approximates inverse matrix columns and applies to a series of popular kernels. Furthermore, the per-prediction cost is $\mathcal{O}(\operatorname{vol}{\mathcal{S}}\log 1/\epsilon)$ locally dependent on the graph with linear memory cost. Experiments show that our scalable method enjoys a better tradeoff between local and global consistency.

        ----

        ## [1799] Horizon-Free and Variance-Dependent Reinforcement Learning for Latent Markov Decision Processes

        **Authors**: *Runlong Zhou, Ruosong Wang, Simon Shaolei Du*

        **Conference**: *icml 2023*

        **URL**: [https://proceedings.mlr.press/v202/zhou23l.html](https://proceedings.mlr.press/v202/zhou23l.html)

        **Abstract**:

        We study regret minimization for reinforcement learning (RL) in Latent Markov Decision Processes (LMDPs) with context in hindsight. We design a novel model-based algorithmic framework which can be instantiated with both a model-optimistic and a value-optimistic solver. We prove an $\tilde{O}(\sqrt{\mathsf{Var}^\star M \Gamma S A K})$ regret bound where $\tilde{O}$ hides logarithm factors, $M$ is the number of contexts, $S$ is the number of states, $A$ is the number of actions, $K$ is the number of episodes, $\Gamma \le S$ is the maximum transition degree of any state-action pair, and $\mathsf{Var}^\star$ is a variance quantity describing the determinism of the LMDP. The regret bound only scales logarithmically with the planning horizon, thus yielding the first (nearly) horizon-free regret bound for LMDP. This is also the first problem-dependent regret bound for LMDP. Key in our proof is an analysis of the total variance of alpha vectors (a generalization of value functions), which is handled with a truncation method. We complement our positive result with a novel $\Omega(\sqrt{\mathsf{Var}^\star M S A K})$ regret lower bound with $\Gamma = 2$, which shows our upper bound minimax optimal when $\Gamma$ is a constant for the class of variance-bounded LMDPs. Our lower bound relies on new constructions of hard instances and an argument inspired by the symmetrization technique from theoretical computer science, both of which are technically different from existing lower bound proof for MDPs, and thus can be of independent interest.

        ----

        

[Go to the previous page](ICML-2023-list08.md)

[Go to the next page](ICML-2023-list10.md)

[Go to the catalog section](README.md)