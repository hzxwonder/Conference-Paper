## [1000] Early-Bird GCNs: Graph-Network Co-optimization towards More Efficient GCN Training and Inference via Drawing Early-Bird Lottery Tickets

**Authors**: *Haoran You, Zhihan Lu, Zijian Zhou, Yonggan Fu, Yingyan Lin*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i8.20873](https://doi.org/10.1609/aaai.v36i8.20873)

**Abstract**:

Graph Convolutional Networks (GCNs) have emerged as the state-of-the-art deep learning model for representation learning on graphs. However, it remains notoriously challenging to train and inference GCNs over large graph datasets, limiting their application to large real-world graphs and hindering the exploration of deeper and more sophisticated GCN graphs. This is because as the graph size grows, the sheer number of node features and the large adjacency matrix can easily explode the required memory and data movements. To tackle the aforementioned challenges, we explore the possibility of drawing lottery tickets when sparsifying GCN graphs, i.e., subgraphs that largely shrink the adjacency matrix yet are capable of achieving accuracy comparable to or even better than their full graphs. Specifically, we for the first time discover the existence of graph early-bird (GEB) tickets that emerge at the very early stage when sparsifying GCN graphs, and propose a simple yet effective detector to automatically identify the emergence of such GEB tickets. Furthermore, we advocate graph-model co-optimization and develop a generic efficient GCN early-bird training framework dubbed GEBT that can significantly boost the efficiency of GCN training by (1) drawing joint early-bird tickets between the GCN graphs and models and (2) enabling simultaneously sparsification of both the GCN graphs and models. Experiments on various GCN models and datasets consistently validate our GEB finding and the effectiveness of our GEBT, e.g., our GEBT achieves up to 80.2% ~ 85.6% and 84.6% ~ 87.5% savings of GCN training and inference costs while offering a comparable or even better accuracy as compared to state-of-the-art methods. Our source code and supplementary appendix are available at https://github.com/RICE-EIC/Early-Bird-GCN.

----

## [1001] Hindsight Network Credit Assignment: Efficient Credit Assignment in Networks of Discrete Stochastic Units

**Authors**: *Kenny Young*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i8.20874](https://doi.org/10.1609/aaai.v36i8.20874)

**Abstract**:

Training neural networks with discrete stochastic variables presents a unique challenge. Backpropagation is not directly applicable, nor are the reparameterization tricks used in networks with continuous stochastic variables. To address this challenge, we present Hindsight Network Credit Assignment (HNCA), a novel gradient estimation algorithm for networks of discrete stochastic units. HNCA works by assigning credit to each unit based on the degree to which its output influences its immediate children in the network. We prove that HNCA produces unbiased gradient estimates with reduced variance compared to the REINFORCE estimator, while the computational cost is similar to that of backpropagation. We first apply HNCA in a contextual bandit setting to optimize a reward function that is unknown to the agent. In this setting, we empirically demonstrate that HNCA significantly outperforms REINFORCE, indicating that the variance reduction implied by our theoretical analysis is significant and impactful. We then show how HNCA can be extended to optimize a more general function of the outputs of a network of stochastic units, where the function is known to the agent. We apply this extended version of HNCA to train a discrete variational auto-encoder and empirically show it compares favourably to other strong methods. We believe that the ideas underlying HNCA can help stimulate new ways of thinking about efficient credit assignment in stochastic compute graphs.

----

## [1002] SAIL: Self-Augmented Graph Contrastive Learning

**Authors**: *Lu Yu, Shichao Pei, Lizhong Ding, Jun Zhou, Longfei Li, Chuxu Zhang, Xiangliang Zhang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i8.20875](https://doi.org/10.1609/aaai.v36i8.20875)

**Abstract**:

This paper studies learning node representations with graph neural networks (GNNs) for unsupervised scenario. Specifically, we derive a theoretical analysis and provide an empirical demonstration about the non-steady performance of GNNs over different graph datasets, when the supervision signals are not appropriately defined. The performance of GNNs depends on both the node feature smoothness and the locality of graph structure. To smooth the discrepancy of node proximity measured by graph topology and node feature, we proposed SAIL - a novel self-augmented graph contrastive learning framework, with two complementary self-distilling regularization modules, i.e., intra- and inter-graph knowledge distillation. We demonstrate the competitive performance of SAIL on a variety of graph applications. Even with a single GNN layer, SAIL has consistently competitive or even better performance on various benchmark datasets, comparing with state-of-the-art baselines.

----

## [1003] Natural Black-Box Adversarial Examples against Deep Reinforcement Learning

**Authors**: *Mengran Yu, Shiliang Sun*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i8.20876](https://doi.org/10.1609/aaai.v36i8.20876)

**Abstract**:

Black-box attacks in deep reinforcement learning usually retrain substitute policies to mimic behaviors of target policies as well as craft adversarial examples, and attack the target policies with these transferable adversarial examples. However, the transferability of adversarial examples is not always guaranteed. Moreover, current methods of crafting adversarial examples only utilize simple pixel space metrics which neglect semantics in the whole images, and thus generate unnatural adversarial examples. To address these problems, we propose an advRL-GAN framework to directly generate semantically natural adversarial examples in the black-box setting, bypassing the transferability requirement of adversarial examples. It formalizes the black-box attack as a reinforcement learning (RL) agent, which explores natural and aggressive adversarial examples with generative adversarial networks and the feedback of target agents. To the best of our knowledge, it is the first RL-based adversarial attack on a deep RL agent. Experimental results on multiple environments demonstrate the effectiveness of advRL-GAN in terms of reward reductions and magnitudes of perturbations, and validate the sparse and targeted property of adversarial perturbations through visualization.

----

## [1004] Regularization Penalty Optimization for Addressing Data Quality Variance in OoD Algorithms

**Authors**: *Runpeng Yu, Hong Zhu, Kaican Li, Lanqing Hong, Rui Zhang, Nanyang Ye, Shao-Lun Huang, Xiuqiang He*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i8.20877](https://doi.org/10.1609/aaai.v36i8.20877)

**Abstract**:

Due to the poor generalization performance of traditional empirical risk minimization (ERM) in the case of distributional shift, Out-of-Distribution (OoD) generalization algorithms receive increasing attention. However, OoD generalization algorithms overlook the great variance in the quality of training data, which significantly compromises the accuracy of these methods. In this paper, we theoretically reveal the relationship between training data quality and algorithm performance, and analyze the optimal regularization scheme for Lipschitz regularized invariant risk minimization. A novel algorithm is proposed based on the theoretical results to alleviate the influence of low quality data at both the sample level and the domain level. The experiments on both the regression and classification benchmarks validate the effectiveness of our method with statistical significance.

----

## [1005] Low-Pass Graph Convolutional Network for Recommendation

**Authors**: *Wenhui Yu, Zixin Zhang, Zheng Qin*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i8.20878](https://doi.org/10.1609/aaai.v36i8.20878)

**Abstract**:

Spectral graph convolution is extremely time-consuming for large graphs, thus existing Graph Convolutional Networks (GCNs) reconstruct the kernel by a polynomial, which is (almost) fixed. To extract features from the graph data by learning kernels, Low-pass Collaborative Filter Network (LCFN) was proposed as a new paradigm with trainable kernels. However, there are two demerits of LCFN: (1) The hypergraphs in LCFN are constructed by mining 2-hop connections of the user-item bipartite graph, thus 1-hop connections are not used, resulting in serious information loss. (2) LCFN follows the general network structure of GCNs, which is suboptimal. To address these issues, we utilize the bipartite graph to define the graph space directly and explore the best network structure based on experiments. Comprehensive experiments on two real-world datasets demonstrate the effectiveness of the proposed model. Codes are available on https://github.com/Wenhui-Yu/LCFN.

----

## [1006] MIA-Former: Efficient and Robust Vision Transformers via Multi-Grained Input-Adaptation

**Authors**: *Zhongzhi Yu, Yonggan Fu, Sicheng Li, Chaojian Li, Yingyan Lin*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i8.20879](https://doi.org/10.1609/aaai.v36i8.20879)

**Abstract**:

Vision transformers have recently demonstrated great success in various computer vision tasks, motivating a tremendously increased interest in their deployment into many real-world IoT applications. 
 However, powerful ViTs are often too computationally expensive to be fitted onto real-world resource-constrained platforms, due to (1) their quadratically increased complexity with the number of input tokens and (2) their overparameterized self-attention heads and model depth. In parallel, different images are of varied complexity and their different regions can contain various levels of visual information, e.g., a sky background is not as informative as a foreground object in object classification tasks, indicating that treating those regions equally in terms of model complexity is unnecessary while such opportunities for trimming down ViTs' complexity have not been fully exploited.
 To this end, we propose a Multi-grained Input-Adaptive Vision Transformer framework dubbed MIA-Former that can input-adaptively adjust the structure of ViTs at three coarse-to-fine-grained granularities (i.e., model depth and the number of model heads/tokens).
 In particular, our MIA-Former adopts a low-cost network trained with a hybrid supervised and reinforcement learning method to skip the unnecessary layers, heads, and tokens in an input adaptive manner, reducing the overall computational cost. Furthermore, an interesting side effect of our MIA-Former is that its resulting ViTs are naturally equipped with improved robustness against adversarial attacks over their static counterparts, because MIA-Former's multi-grained dynamic control improves the model diversity similar to the effect of ensemble and thus increases the 
 difficulty of adversarial attacks against all its sub-models.
 Extensive experiments and ablation studies validate that the proposed MIA-Former framework can (1) effectively allocate adaptive computation budgets to the difficulty of input images, achieving state-of-the-art (SOTA) accuracy-efficiency trade-offs, e.g., up to 16.5\% computation savings with the same or even a higher accuracy compared with the SOTA dynamic transformer models, and (2) boost ViTs' robustness accuracy under various adversarial attacks over their vanilla counterparts by 2.4\% and 3.0\%, respectively. Our code is available at https://github.com/RICE-EIC/MIA-Former.

----

## [1007] Unsupervised Learning of Compositional Scene Representations from Multiple Unspecified Viewpoints

**Authors**: *Jinyang Yuan, Bin Li, Xiangyang Xue*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i8.20880](https://doi.org/10.1609/aaai.v36i8.20880)

**Abstract**:

Visual scenes are extremely rich in diversity, not only because there are infinite combinations of objects and background, but also because the observations of the same scene may vary greatly with the change of viewpoints. When observing a visual scene that contains multiple objects from multiple viewpoints, humans are able to perceive the scene in a compositional way from each viewpoint, while achieving the so-called ``object constancy'' across different viewpoints, even though the exact viewpoints are untold. This ability is essential for humans to identify the same object while moving and to learn from vision efficiently. It is intriguing to design models that have the similar ability. In this paper, we consider a novel problem of learning compositional scene representations from multiple unspecified viewpoints without using any supervision, and propose a deep generative model which separates latent representations into a viewpoint-independent part and a viewpoint-dependent part to solve this problem. To infer latent representations, the information contained in different viewpoints is iteratively integrated by neural networks. Experiments on several specifically designed synthetic datasets have shown that the proposed method is able to effectively learn from multiple unspecified viewpoints.

----

## [1008] TS2Vec: Towards Universal Representation of Time Series

**Authors**: *Zhihan Yue, Yujing Wang, Juanyong Duan, Tianmeng Yang, Congrui Huang, Yunhai Tong, Bixiong Xu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i8.20881](https://doi.org/10.1609/aaai.v36i8.20881)

**Abstract**:

This paper presents TS2Vec, a universal framework for learning representations of time series in an arbitrary semantic level. Unlike existing methods, TS2Vec performs contrastive learning in a hierarchical way over augmented context views, which enables a robust contextual representation for each timestamp. Furthermore, to obtain the representation of an arbitrary sub-sequence in the time series, we can apply a simple aggregation over the representations of corresponding timestamps. We conduct extensive experiments on time series classification tasks to evaluate the quality of time series representations. As a result, TS2Vec achieves significant improvement over existing SOTAs of unsupervised time series representation on 125 UCR datasets and 29 UEA datasets. The learned timestamp-level representations also achieve superior results in time series forecasting and anomaly detection tasks. A linear regression trained on top of the learned representations outperforms previous SOTAs of time series forecasting. Furthermore, we present a simple way to apply the learned representations for unsupervised anomaly detection, which establishes SOTA results in the literature. The source code is publicly available at https://github.com/yuezhihan/ts2vec.

----

## [1009] Fractional Adaptive Linear Units

**Authors**: *Julio Zamora, Anthony D. Rhodes, Lama Nachman*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i8.20882](https://doi.org/10.1609/aaai.v36i8.20882)

**Abstract**:

This work introduces Fractional Adaptive Linear Units (FALUs), a flexible generalization of adaptive activation functions. Leveraging principles from fractional calculus, FALUs define a diverse family of activation functions (AFs) that encompass many traditional and state-of-the-art activation functions. This family includes the Sigmoid, Gaussian, ReLU, GELU, and Swish functions, as well as a large variety of smooth interpolations between these functions. Our technique requires only a small number of additional trainable parameters, and needs no further specialized optimization or initialization procedures. For this reason, FALUs present a seamless and rich automated solution to the problem of activation function optimization.  Through experiments on a variety of conventional tasks and network architectures, we demonstrate the effectiveness of FALUs when compared to traditional and state-of-the-art AFs.  To facilitate practical use of this work, we plan to make our code publicly available

----

## [1010] SimSR: Simple Distance-Based State Representations for Deep Reinforcement Learning

**Authors**: *Hongyu Zang, Xin Li, Mingzhong Wang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i8.20883](https://doi.org/10.1609/aaai.v36i8.20883)

**Abstract**:

This work explores how to learn robust and generalizable state representation from image-based observations with deep reinforcement learning methods. Addressing the computational complexity, stringent assumptions and representation collapse challenges in existing work of bisimulation metric, we devise Simple State Representation (SimSR) operator. SimSR enables us to design a stochastic approximation method that can practically learn the mapping functions (encoders) from observations to latent representation space. In addition to the theoretical analysis and comparison with the existing work, we experimented and compared our work with recent state-of-the-art solutions in visual MuJoCo tasks. The results shows that our model generally achieves better performance and has better robustness and good generalization.

----

## [1011] Efficient Decentralized Stochastic Gradient Descent Method for Nonconvex Finite-Sum Optimization Problems

**Authors**: *Wenkang Zhan, Gang Wu, Hongchang Gao*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i8.20884](https://doi.org/10.1609/aaai.v36i8.20884)

**Abstract**:

Decentralized stochastic gradient descent methods have attracted increasing interest in recent years. Numerous methods have been proposed for the nonconvex finite-sum optimization problem.  However, existing methods have a large sample complexity,  slowing down the empirical convergence speed. To address this issue, in this paper, we proposed a novel decentralized stochastic gradient descent method for the nonconvex finite-sum optimization problem, which enjoys a better sample and communication complexity than existing methods. To the best of our knowledge, our work is the first one achieving such favorable sample and communication complexities. Finally, we have conducted extensive experiments and the experimental results have confirmed the superior performance of our proposed method.

----

## [1012] MetaNODE: Prototype Optimization as a Neural ODE for Few-Shot Learning

**Authors**: *Baoquan Zhang, Xutao Li, Shanshan Feng, Yunming Ye, Rui Ye*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i8.20885](https://doi.org/10.1609/aaai.v36i8.20885)

**Abstract**:

Few-Shot Learning (FSL) is a challenging task, i.e., how to recognize novel classes with few examples? Pre-training based methods effectively tackle the problem by pre-training a feature extractor and then predicting novel classes via a cosine nearest neighbor classifier with mean-based prototypes. Nevertheless, due to the data scarcity, the mean-based prototypes are usually biased. In this paper, we attempt to diminish the prototype bias by regarding it as a prototype optimization problem. To this end, we propose a novel meta-learning based prototype optimization framework to rectify prototypes, i.e., introducing a meta-optimizer to optimize prototypes. Although the existing meta-optimizers can also be adapted to our framework, they all overlook a crucial gradient bias issue, i.e., the mean-based gradient estimation is also biased on sparse data. To address the issue, we regard the gradient and its flow as meta-knowledge and then propose a novel Neural Ordinary Differential Equation (ODE)-based meta-optimizer to polish prototypes, called MetaNODE. In this meta-optimizer, we first view the mean-based prototypes as initial prototypes, and then model the process of prototype optimization as continuous-time dynamics specified by a Neural ODE. A gradient flow inference network is carefully designed to learn to estimate the continuous gradient flow for prototype dynamics. Finally, the optimal prototypes can be obtained by solving the Neural ODE. Extensive experiments on miniImagenet, tieredImagenet, and CUB-200-2011 show the effectiveness of our method.

----

## [1013] State Deviation Correction for Offline Reinforcement Learning

**Authors**: *Hongchang Zhang, Jianzhun Shao, Yuhang Jiang, Shuncheng He, Guanwen Zhang, Xiangyang Ji*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i8.20886](https://doi.org/10.1609/aaai.v36i8.20886)

**Abstract**:

Offline reinforcement learning aims to maximize the expected cumulative rewards with a fixed collection of data. The basic principle of current offline reinforcement learning methods is to restrict the policy to the offline dataset action space. However, they ignore the case where the dataset's trajectories fail to cover the state space completely. Especially, when the dataset's size is limited, it is likely that the agent would encounter unseen states during test time. Prior policy-constrained methods are incapable of correcting the state deviation, and may lead the agent to its unexpected regions further. In this paper, we propose the state deviation correction (SDC) method to constrain the policy's induced state distribution by penalizing the out-of-distribution states which might appear during the test period. We first perturb the states sampled from the logged dataset, then simulate noisy next states on the basis of a dynamics model and the policy. We then train the policy to minimize the distances between the noisy next states and the offline dataset. In this manner, we allow the trained policy to guide the agent to its familiar regions. Experimental results demonstrate that our proposed method is competitive with the state-of-the-art methods in a GridWorld setup, offline Mujoco control suite, and a modified offline Mujoco dataset with a finite number of valuable samples.

----

## [1014] Multi-Agent Reinforcement Learning with General Utilities via Decentralized Shadow Reward Actor-Critic

**Authors**: *Junyu Zhang, Amrit Singh Bedi, Mengdi Wang, Alec Koppel*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i8.20887](https://doi.org/10.1609/aaai.v36i8.20887)

**Abstract**:

We posit a new mechanism for cooperation in multi-agent reinforcement learning (MARL)  based upon any nonlinear function of the team's long-term state-action occupancy measure, i.e., a general utility. This subsumes the cumulative return but also allows one to incorporate risk-sensitivity, exploration, and priors.
We derive the Decentralized Shadow Reward Actor-Critic (DSAC) in which agents alternate between policy evaluation (critic), weighted averaging with neighbors (information mixing), and local gradient updates for their policy parameters (actor). DSAC augments the classic critic step by requiring agents to (i) estimate their local occupancy measure in order to (ii) estimate the derivative of the local utility with respect to their occupancy measure, i.e., the ``shadow reward". DSAC converges to ϵ-stationarity in O(1/ϵ^2.5) or faster O(1/ϵ^2) steps with high probability, depending on the amount of communications. We further establish the non-existence of spurious stationary points for this problem, that is, DSAC finds the globally optimal policy. Experiments demonstrate the merits of goals beyond the cumulative return in cooperative MARL.

----

## [1015] Co-promotion Predictions of Financing Market and Sales Market: A Cooperative-Competitive Attention Approach

**Authors**: *Lei Zhang, Wang Xiang, Chuang Zhao, Hongke Zhao, Rui Li, Runze Wu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i8.20888](https://doi.org/10.1609/aaai.v36i8.20888)

**Abstract**:

Market popularity prediction has always been a hot research topic, such as sales prediction and crowdfunding prediction. Most of these studies put the perspective on isolated markets, relying on the knowledge of certain market to maximize the prediction performance. However, these market-specific approaches are restricted by the knowledge limitation of isolated markets and incapable of the complicated and potential relations among different markets, especially some with strong dependence such as the financing market and sales market. Fortunately, we discover potentially symbiotic relations between the financing market and the sales market, which provides us with an opportunity to co-promote the popularity predictions of both markets. Thus, for bridgly learning the knowledge interactions between financing market and sales market, we propose a cross-market approach, namely CATN: Cooperative-competitive Attention Transfer Network, which could effectively transfer knowledge of financing capability from the crowdfunding market and sales prospect from the E-commerce market. Specifically, for capturing the complicated relations especially the cooperation or complement of items and enhancing the knowledge transfer between the two heterogeneous markets, we design a novel Cooperative Attention; meanwhile, for finely computing the relations of items especially the competition in specific same market, we further design Competitive Attentions for the two markets respectively. Besides, we also distinguish aligned features and unique features to adapt the cross-market predictions. With the real-world datasets collected from Indiegogo and Amazon, we construct extensive experiments on three types of datasets from the two markets and the results demonstrate the effectiveness and generalization of our CATN model.

----

## [1016] Categorical Neighbour Correlation Coefficient (CnCor) for Detecting Relationships between Categorical Variables

**Authors**: *Lifeng Zhang, Shimo Yang, Hongxun Jiang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i8.20889](https://doi.org/10.1609/aaai.v36i8.20889)

**Abstract**:

Categorical data is common and, however, special in that its possible values exist only on a nominal scale so that many statistical operations such as mean, variance, and covariance become not applicable. Following the basic idea of the neighbour correlation coefficient (nCor), in this study, we propose a new measure named the categorical nCor (CnCor) to examine the association between categorical variables through using indicator functions to reform the distance metric and product-moment correlation coefficient. The proposed measure is easy to compute, and enables a direct test of statistical dependence without the need of converting the qualitative variables to quantitative ones. Compare to previous approaches, it is much more robust and effective in dealing with multi-categorical target variables especially when highly nonlinear relationships occurs in the multivariate case. We also applied the CnCor to implementing feature selection by the scheme of backward elimination. Finally, extensive experiments performed on both synthetic and real-world datasets are conducted to demonstrate the outstanding performance of the proposed methods, and draw comparisons with state-of-the-art association measures and feature selection algorithms.

----

## [1017] Interpretable Domain Adaptation for Hidden Subdomain Alignment in the Context of Pre-trained Source Models

**Authors**: *Luxin Zhang, Pascal Germain, Yacine Kessaci, Christophe Biernacki*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i8.20890](https://doi.org/10.1609/aaai.v36i8.20890)

**Abstract**:

Domain adaptation aims to leverage source domain knowledge to predict target domain labels. Most domain adaptation methods tackle a single-source, single-target scenario, whereas source and target domain data can often be subdivided into data from different distributions in real-life applications (e.g., when the distribution of the collected data changes with time). However, such subdomains are rarely given and should be discovered automatically. 
 To this end, some recent domain adaptation works seek separations of hidden subdomains, w.r.t. a known or fixed number of subdomains. 
 In contrast, this paper introduces a new subdomain combination method that leverages a variable number of subdomains. 
 Precisely, we propose to use an inter-subdomain divergence maximization criterion to exploit hidden subdomains. 
 Besides, our proposition stands in a target-to-source domain adaptation scenario, where one exploits a pre-trained source model as a black box; thus, the proposed method is model-agnostic.
 By providing interpretability at two complementary levels (transformation and subdomain levels), our method can also be easily interpreted by practitioners with or without machine learning backgrounds.
 Experimental results over two fraud detection datasets demonstrate the efficiency of our method.

----

## [1018] Convergence and Optimality of Policy Gradient Methods in Weakly Smooth Settings

**Authors**: *Matthew Shunshi Zhang, Murat A. Erdogdu, Animesh Garg*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i8.20891](https://doi.org/10.1609/aaai.v36i8.20891)

**Abstract**:

Policy gradient methods have been frequently applied to problems in control and reinforcement learning with great success, yet existing convergence analysis still relies on non-intuitive, impractical and often opaque conditions. In particular, existing rates are achieved in limited settings, under strict regularity conditions. In this work, we establish explicit convergence rates of policy gradient methods, extending the convergence regime to weakly smooth policy classes with L2 integrable gradient. We provide intuitive examples to illustrate the insight behind these new conditions. Notably, our analysis also shows that convergence rates are achievable for both the standard policy gradient and the natural policy gradient algorithms under these assumptions. Lastly we provide performance guarantees for the converged policies.

----

## [1019] Gaussian Process Bandits with Aggregated Feedback

**Authors**: *Mengyan Zhang, Russell Tsuchida, Cheng Soon Ong*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i8.20892](https://doi.org/10.1609/aaai.v36i8.20892)

**Abstract**:

We consider the continuum-armed bandits problem, under a novel setting of recommending the best arms within a fixed budget under aggregated feedback. This is motivated by applications where the precise rewards are impossible or expensive to obtain, while an aggregated reward or feedback, such as the average over a subset, is available. We constrain the set of reward functions by assuming that they are from a Gaussian Process and propose the Gaussian Process Optimistic Optimisation (GPOO) algorithm. We adaptively construct a tree with nodes as subsets of the arm space, where the feedback is the aggregated reward of representatives of a node. We propose a new simple regret notion with respect to aggregated feedback on the recommended arms. We provide theoretical analysis for the proposed algorithm, and recover single point feedback as a special case. We illustrate GPOO and compare it with related algorithms on simulated data.

----

## [1020] Rethinking Influence Functions of Neural Networks in the Over-Parameterized Regime

**Authors**: *Rui Zhang, Shihua Zhang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i8.20893](https://doi.org/10.1609/aaai.v36i8.20893)

**Abstract**:

Understanding the black-box prediction for neural networks is challenging. To achieve this, early studies have designed influence function (IF) to measure the effect of removing a single training point on neural networks. However, the classic implicit Hessian-vector product (IHVP) method for calculating IF is fragile, and theoretical analysis of IF in the context of neural networks is still lacking. To this end, we utilize the neural tangent kernel (NTK) theory to calculate IF for the neural network trained with regularized mean-square loss, and prove that the approximation error can be arbitrarily small when the width is sufficiently large for two-layer ReLU networks. We analyze the error bound for the classic IHVP method in the over-parameterized regime to understand when and why it fails or not. In detail, our theoretical analysis reveals that (1) the accuracy of IHVP depends on the regularization term, and is pretty low under weak regularization; (2) the accuracy of IHVP has a significant correlation with the probability density of corresponding training points. We further borrow the theory from NTK to understand the IFs better, including quantifying the complexity for influential samples and depicting the variation of IFs during the training dynamics. Numerical experiments on real-world data confirm our theoretical results and demonstrate our findings.

----

## [1021] A Multi-Agent Reinforcement Learning Approach for Efficient Client Selection in Federated Learning

**Authors**: *Sai Qian Zhang, Jieyu Lin, Qi Zhang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i8.20894](https://doi.org/10.1609/aaai.v36i8.20894)

**Abstract**:

Federated learning (FL) is a training technique that enables client devices to jointly learn a shared model by aggregating locally computed models without exposing their raw data. While most of the existing work focuses on improving the FL model accuracy, in this paper, we focus on the improving the training efficiency, which is often a hurdle for adopting FL in real world applications. Specifically, we design an efficient FL framework which jointly optimizes model accuracy, processing latency and communication efficiency, all of which are primary design considerations for real implementation of FL. Inspired by the recent success of Multi Agent Reinforcement Learning (MARL) in solving complex control problems, we present FedMarl, a federated learning framework that relies on trained MARL agents to perform efficient run-time client selection. Experiments show that FedMarl can significantly improve model accuracy with much lower processing latency and communication cost.

----

## [1022] Tailor Versatile Multi-Modal Learning for Multi-Label Emotion Recognition

**Authors**: *Yi Zhang, Mingyuan Chen, Jundong Shen, Chongjun Wang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i8.20895](https://doi.org/10.1609/aaai.v36i8.20895)

**Abstract**:

Multi-modal Multi-label Emotion Recognition (MMER) aims to identify various human emotions from heterogeneous visual, audio and text modalities. Previous methods mainly focus on projecting multiple modalities into a common latent space and learning an identical representation for all labels, which neglects the diversity of each modality and fails to capture richer semantic information for each label from different perspectives. Besides, associated relationships of modalities and labels have not been fully exploited. In this paper, we propose versaTile multi-modAl learning for multI-labeL emOtion Recognition (TAILOR), aiming to refine multi-modal representations and enhance discriminative capacity of each label. Specifically, we design an adversarial multi-modal refinement module to sufficiently explore the commonality among different modalities and strengthen the diversity of each modality. To further exploit label-modal dependence, we devise a BERT-like cross-modal encoder to gradually fuse private and common modality representations in a granularity descent way, as well as a label-guided decoder to adaptively generate a tailored representation for each label with the guidance of label semantics. In addition, we conduct experiments on the benchmark MMER dataset CMU-MOSEI in both aligned and unaligned settings, which demonstrate the superiority of TAILOR over the state-of-the-arts.

----

## [1023] Fusion Multiple Kernel K-means

**Authors**: *Yi Zhang, Xinwang Liu, Jiyuan Liu, Sisi Dai, Changwang Zhang, Kai Xu, En Zhu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i8.20896](https://doi.org/10.1609/aaai.v36i8.20896)

**Abstract**:

Multiple kernel clustering aims to seek an appropriate combination of base kernels to mine inherent non-linear information for optimal clustering. Late fusion algorithms generate base partitions independently and integrate them in the following clustering procedure, improving the overall efficiency. However, the separate base partition generation leads to inadequate negotiation with the clustering procedure and a great loss of beneficial information in corresponding kernel matrices, which negatively affects the clustering performance. To address this issue, we propose a novel algorithm, termed as Fusion Multiple Kernel k-means (FMKKM), which unifies base partition learning and late fusion clustering into one single objective function, and adopts early fusion technique to capture more sufficient information in kernel matrices. Specifically, the early fusion helps base partitions keep more beneficial kernel details, and the base partitions learning further guides the generation of consensus partition in the late fusion stage, while the late fusion provides positive feedback on two former procedures.
 The close collaboration of three procedures results in a promising performance improvement. Subsequently, an alternate optimization method with promising convergence is developed to solve the resultant optimization problem. Comprehensive experimental results demonstrate that our proposed algorithm achieves state-of-the-art performance on multiple public datasets, validating its effectiveness. The code of this work is publicly available at https://github.com/ethan-yizhang/Fusion-Multiple-Kernel-K-means.

----

## [1024] Batch Active Learning with Graph Neural Networks via Multi-Agent Deep Reinforcement Learning

**Authors**: *Yuheng Zhang, Hanghang Tong, Yinglong Xia, Yan Zhu, Yuejie Chi, Lei Ying*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i8.20897](https://doi.org/10.1609/aaai.v36i8.20897)

**Abstract**:

Graph neural networks (GNNs) have achieved tremendous success in many graph learning tasks such as node classification, graph classification and link prediction. For the classification task, GNNs' performance often highly depends on the number of labeled nodes and thus could be significantly hampered due to the expensive annotation cost. The sparse literature on active learning for GNNs has primarily focused on selecting only one sample each iteration, which becomes inefficient for large scale datasets. In this paper, we study the batch active learning setting for GNNs where the learning agent can acquire labels of multiple samples at each time. We formulate batch active learning as a cooperative multi-agent reinforcement learning problem and present a novel reinforced batch-mode active learning framework BiGeNe. To avoid the combinatorial explosion of the joint action space, we introduce a value decomposition method that factorizes the total Q-value into the average of individual Q-values. Moreover, we propose a novel multi-agent Q-network consisting of a graph convolutional network (GCN) component and a gated recurrent unit (GRU) component. The GCN component takes both the informativeness and inter-dependences between nodes into account and the GRU component enables the agent to consider interactions between selected nodes in the same batch. Experimental results on multiple public datasets demonstrate the effectiveness and efficiency of our proposed method.

----

## [1025] ProtGNN: Towards Self-Explaining Graph Neural Networks

**Authors**: *Zaixi Zhang, Qi Liu, Hao Wang, Chengqiang Lu, Cheekong Lee*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i8.20898](https://doi.org/10.1609/aaai.v36i8.20898)

**Abstract**:

Despite the recent progress in Graph Neural Networks (GNNs), it remains challenging to explain the predictions
 made by GNNs. Existing explanation methods mainly focus on post-hoc explanations where another explanatory model is employed to provide explanations for a trained GNN. The fact that post-hoc methods fail to reveal the original reasoning process of GNNs raises the need of building GNNs with built-in interpretability. In this work, we propose Prototype Graph Neural Network (ProtGNN), which combines prototype learning with GNNs and provides a new perspective on the explanations of GNNs. In ProtGNN, the explanations are naturally derived from the case-based reasoning process and are actually used during classification. The prediction of ProtGNN is obtained by comparing the inputs to a few learned prototypes in the latent space.
 Furthermore, for better interpretability and higher efficiency, a novel conditional subgraph sampling module is incorporated to indicate which part of the input graph is most similar to each prototype in ProtGNN+. Finally, we evaluate our method on a wide range of datasets and perform concrete case studies. Extensive results show that ProtGNN and ProtGNN+ can provide inherent interpretability while achieving accuracy on par with the non-interpretable counterparts.

----

## [1026] Learning to Solve Travelling Salesman Problem with Hardness-Adaptive Curriculum

**Authors**: *Zeyang Zhang, Ziwei Zhang, Xin Wang, Wenwu Zhu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i8.20899](https://doi.org/10.1609/aaai.v36i8.20899)

**Abstract**:

Various neural network models have been proposed to tackle combinatorial optimization problems such as the travelling salesman problem (TSP). Existing learning-based TSP methods adopt a simple setting that the training and testing data are independent and identically distributed. However, the existing literature fails to solve TSP instances when training and testing data have different distributions.
Concretely, we find that different training and testing distribution will result in more difficult TSP instances, i.e., the solution obtained by the model has a large gap from the optimal solution.
To tackle this problem, in this work, we study learning-based TSP methods when training and testing data have different distributions using adaptive-hardness, i.e., how difficult a TSP instance can be for a solver. 
This problem is challenging because it is non-trivial to (1) define hardness measurement quantitatively; (2) efficiently and continuously generate sufficiently hard TSP instances upon model training; (3) fully utilize instances with different levels of hardness to learn a more powerful TSP solver.
To solve these challenges, we first propose a principled hardness measurement to quantify the hardness of TSP instances. Then, we propose a hardness-adaptive generator to generate instances with different hardness. We further propose a curriculum learner fully utilizing these instances to train the TSP solver. Experiments show that our hardness-adaptive generator can generate instances ten times harder than the existing methods, and our proposed method achieves significant improvement over state-of-the-art models in terms of the optimality gap. The codes are publicly available.

----

## [1027] Robust Action Gap Increasing with Clipped Advantage Learning

**Authors**: *Zhe Zhang, Yaozhong Gan, Xiaoyang Tan*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i8.20900](https://doi.org/10.1609/aaai.v36i8.20900)

**Abstract**:

Advantage Learning (AL) seeks to increase the action gap between the optimal action and its competitors, so as to improve the robustness to estimation errors. However, the method becomes problematic when the optimal action induced by the approximated value function does not agree with the true optimal action. In this paper, we present a novel method, named clipped Advantage Learning (clipped AL), to address this issue. The method is inspired by our observation that increasing the action gap blindly for all given samples while not taking their necessities into account could accumulate more errors in the performance loss bound, leading to a slow value convergence, and to avoid that, we should adjust the advantage value adaptively. We show that our simple clipped AL operator not only enjoys fast convergence guarantee but also retains proper action gaps, hence achieving a good balance between the large action gap and the fast convergence. The feasibility and effectiveness of the proposed method are verified empirically on several RL benchmarks with promising performance.

----

## [1028] Online Influence Maximization with Node-Level Feedback Using Standard Offline Oracles

**Authors**: *Zhijie Zhang, Wei Chen, Xiaoming Sun, Jialin Zhang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i8.20901](https://doi.org/10.1609/aaai.v36i8.20901)

**Abstract**:

We study the online influence maximization (OIM) problem in social networks, where in multiple rounds the learner repeatedly chooses seed nodes to generate cascades, observes the cascade feedback, and gradually learns the best seeds that generate the largest cascade. We focus on two major challenges in this paper. First, we work with node-level feedback instead of edge-level feedback. The edge-level feedback reveals all edges that pass through information in a cascade, whereas the node-level feedback only reveals the activated nodes with timestamps. The node-level feedback is arguably more realistic since in practice it is relatively easy to observe who is influenced but very difficult to observe from which relationship (edge) the influence comes. Second, we use standard offline oracles instead of offline pair-oracles. To compute a good seed set for the next round, an offline pair-oracle finds the best seed set and the best parameters within the confidence region simultaneously, and such an oracle is difficult to compute due to the combinatorial core of the OIM problem. So we focus on how to use the standard offline influence maximization oracle which finds the best seed set given the edge parameters as input. In this paper, we resolve these challenges for the famous independent cascade (IC) diffusion model. The past research only achieves edge-level feedback, while we present the first optimal regret algorithm for the node-level feedback. For the first challenge above, we apply a novel adaptation of the maximum likelihood estimation (MLE) approach to learn the graph parameters and its confidence region (a confidence ellipsoid). For the second challenge, we adjust the update procedure to dissect the confidence ellipsoid into confidence intervals on each parameter, so that the standard offline influence maximization oracle is enough.

----

## [1029] CLPA: Clean-Label Poisoning Availability Attacks Using Generative Adversarial Nets

**Authors**: *Bingyin Zhao, Yingjie Lao*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i8.20902](https://doi.org/10.1609/aaai.v36i8.20902)

**Abstract**:

Poisoning attacks are emerging threats to deep neural networks where the adversaries attempt to compromise the models by injecting malicious data points in the clean training data. Poisoning attacks target either the availability or integrity of a model. The availability attack aims to degrade the overall accuracy while the integrity attack causes misclassification only for specific instances without affecting the accuracy of clean data. Although clean-label integrity attacks are proven to be effective in recent studies, the feasibility of clean-label availability attacks remains unclear. This paper, for the first time, proposes a clean-label approach, CLPA, for the poisoning availability attack. We reveal that due to the intrinsic imperfection of classifiers, naturally misclassified inputs can be considered as a special type of poisoned data, which we refer to as "natural poisoned data''. We then propose a two-phase generative adversarial net (GAN) based poisoned data generation framework along with a triplet loss function for synthesizing clean-label poisoned samples that locate in a similar distribution as natural poisoned data. The generated poisoned data are plausible to human perception and can also bypass the singular vector decomposition (SVD) based defense. We demonstrate the effectiveness of our approach on CIFAR-10 and ImageNet dataset over a variety type of models. Codes are available at: https://github.com/bxz9200/CLPA.

----

## [1030] FedInv: Byzantine-Robust Federated Learning by Inversing Local Model Updates

**Authors**: *Bo Zhao, Peng Sun, Tao Wang, Keyu Jiang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i8.20903](https://doi.org/10.1609/aaai.v36i8.20903)

**Abstract**:

Federated learning (FL) is a privacy-preserving distributed machine learning paradigm that enables multiple clients to collaboratively train statistical models without disclosing raw training data. However, the inaccessible local training data and uninspectable local training process make FL susceptible to various Byzantine attacks (e.g., data poisoning and model poisoning attacks), aiming to manipulate the FL model training process and degrade the model performance. Most of the existing Byzantine-robust FL schemes cannot effectively defend against stealthy poisoning attacks that craft poisoned models statistically similar to benign models. Things worsen when many clients are compromised or data among clients are highly non-independent and identically distributed (non-IID). In this work, to address these issues, we propose FedInv, a novel Byzantine-robust FL framework by inversing local model updates. Specifically, in each round of local model aggregation in FedInv, the parameter server first inverses the local model updates submitted by each client to generate a corresponding dummy dataset. Then, the server identifies those dummy datasets with exceptional Wasserstein distances from others and excludes the related local model updates from model aggregation. We conduct an exhaustive experimental evaluation of FedInv. The results demonstrate that FedInv significantly outperforms the existing robust FL schemes in defending against stealthy poisoning attacks under highly non-IID data partitions.

----

## [1031] Well-Classified Examples Are Underestimated in Classification with Deep Neural Networks

**Authors**: *Guangxiang Zhao, Wenkai Yang, Xuancheng Ren, Lei Li, Yunfang Wu, Xu Sun*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i8.20904](https://doi.org/10.1609/aaai.v36i8.20904)

**Abstract**:

The conventional wisdom behind learning deep classification models is to focus on bad-classified examples and ignore well-classified examples that are far from the decision boundary. For instance, when training with cross-entropy loss, examples with higher likelihoods (i.e., well-classified examples) contribute smaller gradients in back-propagation. However, we theoretically show that this common practice hinders representation learning, energy optimization, and margin growth. To counteract this deficiency, we propose to reward well-classified examples with additive bonuses to revive their contribution to the learning process. This counterexample theoretically addresses these three issues. We empirically support this claim by directly verifying the theoretical results or significant performance improvement with our counterexample on diverse tasks, including image classification, graph classification, and machine translation. Furthermore, this paper shows that we can deal with complex scenarios, such as imbalanced classification, OOD detection, and applications under adversarial attacks because our idea can solve these three issues. Code is available at https://github.com/lancopku/well-classified-examples-are-underestimated.

----

## [1032] Error-Based Knockoffs Inference for Controlled Feature Selection

**Authors**: *Xuebin Zhao, Hong Chen, Yingjie Wang, Weifu Li, Tieliang Gong, Yulong Wang, Feng Zheng*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i8.20905](https://doi.org/10.1609/aaai.v36i8.20905)

**Abstract**:

Recently, the scheme of model-X knockoffs was proposed as a promising solution to address controlled feature selection under high-dimensional finite-sample settings. However, the procedure of model-X knockoffs depends heavily on the coefficient-based feature importance and only concerns the control of false discovery rate (FDR). To further improve its adaptivity and flexibility, in this paper, we propose an error-based knockoff inference method by integrating the knockoff features, the error-based feature importance statistics, and the stepdown procedure together. The proposed inference procedure does not require specifying a regression model and can handle feature selection with theoretical guarantees on controlling false discovery proportion (FDP), FDR, or k-familywise error rate (k-FWER). Empirical evaluations demonstrate the competitive performance of our approach on both simulated and real data.

----

## [1033] Online Missing Value Imputation and Change Point Detection with the Gaussian Copula

**Authors**: *Yuxuan Zhao, Eric Landgrebe, Eliot Shekhtman, Madeleine Udell*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i8.20906](https://doi.org/10.1609/aaai.v36i8.20906)

**Abstract**:

Missing value imputation is crucial for real-world data science workflows. Imputation is harder in the online setting, as it requires the imputation method itself to be able to evolve over time. For practical applications, imputation algorithms should produce imputations that match the true data distribution, handle data of mixed types, including ordinal, boolean, and continuous variables, and scale to large datasets. In this work we develop a new online imputation algorithm for mixed data using the Gaussian copula. The online Gaussian copula model produces meets all the desiderata: its imputations match the data distribution even for mixed data, improve over its offline counterpart on the accuracy when the streaming data has a changing distribution, and on the speed (up to an order of magnitude) especially on large scale datasets. By fitting the copula model to online data, we also provide a new method to detect change points in the multivariate dependence structure for mixed data with missing values. Experimental results on synthetic and real world data validate the performance of the proposed methods.

----

## [1034] LaSSL: Label-Guided Self-Training for Semi-supervised Learning

**Authors**: *Zhen Zhao, Luping Zhou, Lei Wang, Yinghuan Shi, Yang Gao*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i8.20907](https://doi.org/10.1609/aaai.v36i8.20907)

**Abstract**:

The key to semi-supervised learning (SSL) is to explore adequate information to leverage the unlabeled data. Current dominant approaches aim to generate pseudo-labels on weakly augmented instances and train models on their corresponding strongly augmented variants with high-confidence results. However, such methods are limited in excluding samples with low-confidence pseudo-labels and under-utilization of the label information. In this paper, we emphasize the cruciality of the label information and propose a Label-guided Self-training approach to Semi-supervised Learning (LaSSL), which improves pseudo-label generations from two mutually boosted strategies. First, with the ground-truth labels and iteratively-polished pseudo-labels, we explore instance relations among all samples and then minimize a class-aware contrastive loss to learn discriminative feature representations that make same-class samples gathered and different-class samples scattered. Second, on top of improved feature representations, we propagate the label information to the unlabeled samples across the potential data manifold at the feature-embedding level, which can further improve the labelling of samples with reference to their neighbours. These two strategies are seamlessly integrated and mutually promoted across the whole training process. We evaluate LaSSL on several classification benchmarks under partially labeled settings and demonstrate its superiority over the state-of-the-art approaches.

----

## [1035] Stackelberg Actor-Critic: Game-Theoretic Reinforcement Learning Algorithms

**Authors**: *Liyuan Zheng, Tanner Fiez, Zane Alumbaugh, Benjamin Chasnov, Lillian J. Ratliff*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i8.20908](https://doi.org/10.1609/aaai.v36i8.20908)

**Abstract**:

The hierarchical interaction between the actor and critic in actor-critic based reinforcement learning algorithms naturally lends itself to a game-theoretic interpretation. We adopt this viewpoint and model the actor and critic interaction as a two-player general-sum game with a leader-follower structure known as a Stackelberg game. Given this abstraction, we propose a meta-framework for Stackelberg actor-critic algorithms where the leader player follows the total derivative of its objective instead of the usual individual gradient. From a theoretical standpoint, we develop a policy gradient theorem for the refined update and provide a local convergence guarantee for the Stackelberg actor-critic algorithms to a local Stackelberg equilibrium. From an empirical standpoint, we demonstrate via simple examples that the learning dynamics we study mitigate cycling and accelerate convergence compared to the usual gradient dynamics given cost structures induced by actor-critic formulations. Finally, extensive experiments on OpenAI gym environments show that Stackelberg actor-critic algorithms always perform at least as well and often significantly outperform the standard actor-critic algorithm counterparts.

----

## [1036] Adaptive Pairwise Weights for Temporal Credit Assignment

**Authors**: *Zeyu Zheng, Risto Vuorio, Richard L. Lewis, Satinder Singh*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i8.20909](https://doi.org/10.1609/aaai.v36i8.20909)

**Abstract**:

How much credit (or blame) should an action taken in a state get for a future reward? This is the fundamental temporal credit assignment problem in Reinforcement Learning (RL). One of the earliest and still most widely used heuristics is to assign this credit based on a scalar coefficient, lambda (treated as a hyperparameter), raised to the power of the time interval between the state-action and the reward. In this empirical paper, we explore heuristics based on more general pairwise weightings that are functions of the state in which the action was taken, the state at the time of the reward, as well as the time interval between the two. Of course it isn't clear what these pairwise weight functions should be, and because they are too complex to be treated as hyperparameters we develop a metagradient procedure for learning these weight functions during the usual RL training of a policy. Our empirical work shows that it is often possible to learn these pairwise weight functions during learning of the policy to achieve better performance than competing approaches.

----

## [1037] Programmatic Reward Design by Example

**Authors**: *Weichao Zhou, Wenchao Li*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i8.20910](https://doi.org/10.1609/aaai.v36i8.20910)

**Abstract**:

Reward design is a fundamental problem in reinforcement learning (RL). A misspecified or poorly designed reward can result in low sample efficiency and undesired behaviors. In this paper, we propose the idea of programmatic reward design, i.e. using programs to specify the reward functions in RL environments. Programs allow human engineers to express sub-goals and complex task scenarios in a structured and interpretable way. The challenge of programmatic reward design, however, is that while humans can provide the high-level structures, properly setting the low-level details, such as the right amount of reward for a specific sub-task, remains difficult. A major contribution of this paper is a probabilistic framework that can infer the best candidate programmatic reward function from expert demonstrations. Inspired by recent generative-adversarial approaches, our framework searches for themost likely programmatic reward function under whichthe optimally generated trajectories cannot be differen-tiated from the demonstrated trajectories. Experimental results show that programmatic reward functions learned using this framework can significantly outperform those learned using existing reward learning algorithms, and enable RL agents to achieve state-of-the-art performance on highly complex tasks.

----

## [1038] Neural Piecewise-Constant Delay Differential Equations

**Authors**: *Qunxi Zhu, Yifei Shen, Dongsheng Li, Wei Lin*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i8.20911](https://doi.org/10.1609/aaai.v36i8.20911)

**Abstract**:

Continuous-depth neural networks, such as the Neural Ordinary Differential Equations (ODEs), have aroused a great deal of interest from the communities of machine learning and data science in recent years, which bridge the connection between deep neural networks and dynamical systems. In this article, we introduce a new sort of continuous-depth neural network, called the Neural Piecewise-Constant Delay Differential Equations (PCDDEs). Here, unlike the recently proposed framework of the Neural Delay Differential Equations (DDEs), we transform the single delay into the piecewise-constant delay(s). The Neural PCDDEs with such a transformation, on one hand, inherit the strength of universal approximating capability in Neural DDEs. On the other hand, the Neural PCDDEs, leveraging the contributions of the information from the multiple previous time steps, further promote the modeling capability without augmenting the network dimension. With such a promotion, we show that the Neural PCDDEs do outperform the several existing continuous-depth neural frameworks on the one-dimensional piecewise-constant delay population dynamics and real-world datasets, including MNIST, CIFAR10, and SVHN.

----

## [1039] Structural Landmarking and Interaction Modelling: A "SLIM" Network for Graph Classification

**Authors**: *Yaokang Zhu, Kai Zhang, Jun Wang, Haibin Ling, Jie Zhang, Hongyuan Zha*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i8.20912](https://doi.org/10.1609/aaai.v36i8.20912)

**Abstract**:

Graph neural networks are a promising architecture for learning and inference with graph-structured data. Yet, how to generate informative, fixed dimensional features for graphs with varying size and topology can still be challenging. Typically, this is achieved through graph-pooling, which summarizes a graph by compressing all its nodes into a single vector. Is such a “collapsing-style” graph-pooling the only choice for graph classification? From complex system’s point of view, properties of a complex system arise largely from the interaction among its components. Therefore, we speculate that preserving the interacting relation between parts, instead of pooling them together, could benefit system level prediction. To verify this, we propose SLIM, a graph neural network model for Structural Landmarking and Interaction Modelling. The main idea is to compute a set of end-to-end optimizable sub-structure landmarks, so that any input graph can be projected onto these (spatially) local structural representatives for a faithful, global characterization. By doing so, explicit interaction between component parts of a graph can be leveraged directly in generating discriminative graph representation. Encouraging results are observed on benchmark datasets for graph classification, demonstrating the value of interaction modelling in the design of graph neural networks.

----

## [1040] Invariant Action Effect Model for Reinforcement Learning

**Authors**: *Zheng-Mao Zhu, Shengyi Jiang, Yu-Ren Liu, Yang Yu, Kun Zhang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i8.20913](https://doi.org/10.1609/aaai.v36i8.20913)

**Abstract**:

Good representations can help RL agents perform concise modeling of their surroundings, and thus support effective decision-making in complex environments. 
 Previous methods learn good representations by imposing extra constraints on dynamics.
 However, in the causal perspective, the causation between the action and its effect is not fully considered in those methods, which leads to the ignorance of the underlying relations among the action effects on the transitions. 
 Based on the intuition that the same action always causes similar effects among different states, we induce such causation by taking the invariance of action effects among states as the relation.
 By explicitly utilizing such invariance, in this paper, we show that a better representation can be learned and potentially improves the sample efficiency and the generalization ability of the learned policy. 
 We propose Invariant Action Effect Model (IAEM) to capture the invariance in action effects, where the effect of an action is represented as the residual of representations from neighboring states.
 IAEM is composed of two parts:
 (1) a new contrastive-based loss to capture the underlying invariance of action effects;
 (2) an individual action effect and provides a self-adapted weighting strategy to tackle the corner cases where the invariance does not hold.
 The extensive experiments on two benchmarks, i.e. Grid-World and Atari, show that the representations learned by IAEM preserve the invariance of action effects. 
 Moreover, with the invariant action effect, IAEM can accelerate the learning process by 1.6x, rapidly generalize to new environments by fine-tuning on a few components, and outperform other dynamics-based representation methods by 1.4x in limited steps.

----

## [1041] Self-Adaptive Imitation Learning: Learning Tasks with Delayed Rewards from Sub-optimal Demonstrations

**Authors**: *Zhuangdi Zhu, Kaixiang Lin, Bo Dai, Jiayu Zhou*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i8.20914](https://doi.org/10.1609/aaai.v36i8.20914)

**Abstract**:

Reinforcement learning (RL) has demonstrated its superiority in solving sequential decision-making problems. However, heavy dependence on immediate reward feedback impedes the wide application of RL. On the other hand, imitation learning (IL) tackles RL without relying on environmental supervision by leveraging external demonstrations. In practice, however, collecting sufficient expert demonstrations can be prohibitively expensive, yet the quality of demonstrations typically limits the performance of the learning policy. To address a practical scenario, in this work, we propose Self-Adaptive Imitation Learning (SAIL), which, provided with a few demonstrations from a sub-optimal teacher, can perform well in RL tasks with extremely delayed rewards, where the only reward feedback is trajectory-wise ranking. SAIL bridges the advantages of IL and RL by interactively exploiting the demonstrations to catch up with the teacher and exploring the environment to yield demonstrations that surpass the teacher. Extensive empirical results show that not only does SAIL significantly improve the sample efficiency, but it also leads to higher asymptotic performance across different continuous control tasks, compared with the state-of-the-art.

----

## [1042] Locality Matters: A Scalable Value Decomposition Approach for Cooperative Multi-Agent Reinforcement Learning

**Authors**: *Roy Zohar, Shie Mannor, Guy Tennenholtz*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i8.20915](https://doi.org/10.1609/aaai.v36i8.20915)

**Abstract**:

Cooperative multi-agent reinforcement learning (MARL) faces significant scalability issues due to state and action spaces that are exponentially large in the number of agents. As environments grow in size, effective credit assignment becomes increasingly harder and often results in infeasible learning times. Still, in many real-world settings, there exist simplified underlying dynamics that can be leveraged for more scalable solutions. In this work, we exploit such locality structures effectively whilst maintaining global cooperation. We propose a novel, value-based multi-agent algorithm called LOMAQ, which incorporates local rewards in the Centralized Training Decentralized Execution paradigm. Additionally, we provide a direct reward decomposition method for finding these local rewards when only a global signal is provided. We test our method empirically, showing it scales well compared to other methods, significantly improving performance and convergence speed.

----

## [1043] Hedonic Games with Fixed-Size Coalitions

**Authors**: *Vittorio Bilò, Gianpiero Monaco, Luca Moscardelli*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i9.21156](https://doi.org/10.1609/aaai.v36i9.21156)

**Abstract**:

In hedonic games, a set of n agents, having preferences over all possible coalition structures, needs to agree on a stable outcome. In this work, we initiate the study of hedonic games with fixed-size coalitions, where the set of possible coalition structures is restricted as follows: there are k coalitions, each coalition has a fixed size, and the sum of the sizes of all coalitions equals n. 
We focus on the basic model of additively separable hedonic games with symmetric preferences, where an agent's preference is captured by a utility function which sums up a contribution due to any other agent in the same coalition. In this setting, an outcome is stable if no pair of agents can exchange coalitions and improve their utilities. Conditioned on the definition of improvement, three stability notions arise: swap stability under transferable utilities, which requires to improve the sum of the utilities of both agents, swap stability, which requires to improve the utility of one agent without decreasing the utility of the other one, and strict swap stability, requiring to improve the utilities of both agents simultaneously.
We analyse the fundamental questions of existence, complexity and efficiency of stable outcomes, and that of complexity of a social optimum.

----

## [1044] Partner-Aware Algorithms in Decentralized Cooperative Bandit Teams

**Authors**: *Erdem Biyik, Anusha Lalitha, Rajarshi Saha, Andrea Goldsmith, Dorsa Sadigh*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i9.21158](https://doi.org/10.1609/aaai.v36i9.21158)

**Abstract**:

When humans collaborate with each other, they often make decisions by observing others and considering the consequences that their actions may have on the entire team, instead of greedily doing what is best for just themselves. We would like our AI agents to effectively collaborate in a similar way by capturing a model of their partners. In this work, we propose and analyze a decentralized Multi-Armed Bandit (MAB) problem with coupled rewards as an abstraction of more general multi-agent collaboration. We demonstrate that naive extensions of single-agent optimal MAB algorithms fail when applied for decentralized bandit teams. Instead, we propose a Partner-Aware strategy for joint sequential decision-making that extends the well-known single-agent Upper Confidence Bound algorithm. We analytically show that our proposed strategy achieves logarithmic regret, and provide extensive experiments involving human-AI and human-robot collaboration to validate our theoretical findings. Our results show that the proposed partner-aware strategy outperforms other known methods, and our human subject studies suggest humans prefer to collaborate with AI agents implementing our partner-aware strategy.

----

## [1045] Fixation Maximization in the Positional Moran Process

**Authors**: *Joachim Brendborg, Panagiotis Karras, Andreas Pavlogiannis, Asger Ullersted Rasmussen, Josef Tkadlec*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i9.21160](https://doi.org/10.1609/aaai.v36i9.21160)

**Abstract**:

The Moran process is a classic stochastic process that models invasion dynamics on graphs. A single mutant (e.g., a new opinion, strain, social trait etc.) invades a population of residents spread over the nodes of a graph. The mutant fitness advantage δ>=0 determines how aggressively mutants propagate to their neighbors. The quantity of interest is the fixation probability, i.e., the probability that the initial mutant eventually takes over the whole population. However, in realistic settings, the invading mutant has an advantage only in certain locations. E.g., the ability to metabolize a certain sugar is an advantageous trait to bacteria only when the sugar is actually present in their surroundings. In this paper we introduce the positional Moran process, a natural generalization in which the mutant fitness advantage is only realized on specific nodes called active nodes, and study the problem of fixation maximization: given a budget k, choose a set of k active nodes that maximize the fixation probability of the invading mutant. We show that the problem is NP-hard, while the optimization function is not submodular, thus indicating strong computational hardness. We focus on two natural limits. In the limit of δ to infinity (strong selection), although the problem remains NP-hard, the optimization function becomes submodular and thus admits a constant-factor approximation using a simple greedy algorithm. In the limit of δ to 0 (weak selection), we show that we can obtain a tight approximation in O(n^{2×ω}) time, where ω is the matrix-multiplication exponent. An experimental evaluation of the new algorithms along with some proposed heuristics corroborates our results.

----

## [1046] Flex Distribution for Bounded-Suboptimal Multi-Agent Path Finding

**Authors**: *Shao-Hung Chan, Jiaoyang Li, Graeme Gange, Daniel Harabor, Peter J. Stuckey, Sven Koenig*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i9.21162](https://doi.org/10.1609/aaai.v36i9.21162)

**Abstract**:

Multi-Agent Path Finding (MAPF) is the problem of finding collision-free paths for multiple agents that minimize the sum of path costs. EECBS is a leading two-level algorithm that solves MAPF bounded-suboptimally, that is, within some factor w of the minimum sum of path costs C*. It uses focal search to find bounded-suboptimal paths on the low level and  Explicit Estimation Search (EES) to resolve collisions on the high level. EES keeps track of a lower bound LB on C* to find paths whose sum of path costs is at most w LB in order to solve MAPF bounded-suboptimally. However, the costs of many paths are often much smaller than w times their minimum path costs, meaning that the sum of path costs is much smaller than w C*. In this paper, we therefore propose Flexible EECBS (FEECBS), which uses a flex(ible) distribution of the path costs (that relaxes the requirement to find bounded-suboptimal paths on the low level) in order to reduce the number of collisions that need to be resolved on the high level while still guaranteeing to solve MAPF bounded suboptimally. We address the drawbacks of flex distribution via techniques such as restrictions on the flex distribution, restarts of the high-level search with EECBS, and low-level focal-A* search. Our empirical evaluation shows that FEECBS substantially improves the efficiency of EECBS on MAPF instances with large maps and large numbers of agents.

----

## [1047] Participatory Budgeting with Donations and Diversity Constraints

**Authors**: *Jiehua Chen, Martin Lackner, Jan Maly*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i9.21163](https://doi.org/10.1609/aaai.v36i9.21163)

**Abstract**:

Participatory budgeting (PB) is a democratic process where citizens jointly decide on how to allocate public funds to indivisible projects. In this work, we focus on PB processes where citizens may provide additional money to projects they want to see funded. We introduce a formal framework for this kind of PB with donations. Our framework also allows for diversity constraints, meaning that each project belongs to one or more types, and there are lower and upper bounds on the number of projects of the same type that can be funded. We propose three general classes of methods for aggregating the citizens’ preferences in the presence of donations and analyze their axiomatic properties. Furthermore, we investigate the computational complexity of determining the outcome of a PB process with donations and of finding a citizen’s optimal donation strategy.

----

## [1048] Pretrained Cost Model for Distributed Constraint Optimization Problems

**Authors**: *Yanchen Deng, Shufeng Kong, Bo An*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i9.21164](https://doi.org/10.1609/aaai.v36i9.21164)

**Abstract**:

Distributed Constraint Optimization Problems (DCOPs) are an important subclass of combinatorial optimization problems, where information and controls are distributed among multiple autonomous agents. Previously, Machine Learning (ML) has been largely applied to solve combinatorial optimization problems by learning effective heuristics. However, existing ML-based heuristic methods are often not generalizable to different search algorithms. Most importantly, these methods usually require full knowledge about the problems to be solved, which are not suitable for distributed settings where centralization is not realistic due to geographical limitations or privacy concerns. To address the generality issue, we propose a novel directed acyclic graph representation schema for DCOPs and leverage the Graph Attention Networks (GATs) to embed graph representations. Our model, GAT-PCM, is then pretrained with optimally labelled data in an offline manner, so as to construct effective heuristics to boost a broad range of DCOP algorithms where evaluating the quality of a partial assignment is critical, such as local search or backtracking search. Furthermore, to enable decentralized model inference, we propose a distributed embedding schema of GAT-PCM where each agent exchanges only embedded vectors, and show its soundness and complexity. Finally, we demonstrate the effectiveness of our model by combining it with a local search or a backtracking search algorithm. Extensive empirical evaluations indicate that the GAT-PCM-boosted algorithms significantly outperform the state-of-the-art methods in various benchmarks.

----

## [1049] Concentration Network for Reinforcement Learning of Large-Scale Multi-Agent Systems

**Authors**: *Qingxu Fu, Tenghai Qiu, Jianqiang Yi, Zhiqiang Pu, Shiguang Wu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i9.21165](https://doi.org/10.1609/aaai.v36i9.21165)

**Abstract**:

When dealing with a series of imminent issues, humans can naturally concentrate on a subset of these concerning issues by prioritizing them according to their contributions to motivational indices, e.g., the probability of winning a game. This idea of concentration offers insights into reinforcement learning of sophisticated Large-scale Multi-Agent Systems (LMAS) participated by hundreds of agents. In such an LMAS, each agent receives a long series of entity observations at each step, which can overwhelm existing aggregation networks such as graph attention networks and cause inefficiency. In this paper, we propose a concentration network called ConcNet. First, ConcNet scores the observed entities considering several motivational indices, e.g., expected survival time and state value of the agents, and then ranks, prunes, and aggregates the encodings of observed entities to extract features. Second, distinct from the well-known attention mechanism, ConcNet has a unique motivational subnetwork to explicitly consider the motivational indices when scoring the observed entities. Furthermore, we present a concentration policy gradient architecture that can learn effective policies in LMAS from scratch. Extensive experiments demonstrate that the presented architecture has excellent scalability and flexibility, and significantly outperforms existing methods on LMAS benchmarks.

----

## [1050] Cooperative Multi-Agent Fairness and Equivariant Policies

**Authors**: *Niko A. Grupen, Bart Selman, Daniel D. Lee*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i9.21166](https://doi.org/10.1609/aaai.v36i9.21166)

**Abstract**:

We study fairness through the lens of cooperative multi-agent learning. Our work is motivated by empirical evidence that naive maximization of team reward yields unfair outcomes for individual team members. To address fairness in multi-agent contexts, we introduce team fairness, a group-based fairness measure for multi-agent learning. We then prove that it is possible to enforce team fairness during policy optimization by transforming the team's joint policy into an equivariant map. We refer to our multi-agent learning strategy as Fairness through Equivariance (Fair-E) and demonstrate its effectiveness empirically. We then introduce Fairness through Equivariance Regularization (Fair-ER) as a soft-constraint version of Fair-E and show that it reaches higher levels of utility than Fair-E and fairer outcomes than non-equivariant policies. Finally, we present novel findings regarding the fairness-utility trade-off in multi-agent settings; showing that the magnitude of the trade-off is dependent on agent skill.

----

## [1051] Practical Fixed-Parameter Algorithms for Defending Active Directory Style Attack Graphs

**Authors**: *Mingyu Guo, Jialiang Li, Aneta Neumann, Frank Neumann, Hung Nguyen*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i9.21167](https://doi.org/10.1609/aaai.v36i9.21167)

**Abstract**:

Active Directory is the default security management system for Windows domain networks. We study the shortest path edge interdiction problem for defending Active Directory style attack graphs. The problem is formulated as a Stackelberg game between one defender and one attacker. The attack graph contains one destination node and multiple entry nodes. The attacker's entry node is chosen by nature. The defender chooses to block a set of edges limited by his budget. The attacker then picks the shortest unblocked attack path. The defender aims to maximize the expected shortest path length for the attacker, where the expectation is taken over entry nodes.
 
 We observe that practical Active Directory attack graphs have small maximum attack path length and are structurally close to trees. We first show that even if the maximum attack path length is a constant, the problem is still w[1]-hard with respect to the defender's budget. Having a small maximum attack path length and a small budget is not enough to design fixed-parameter algorithms. If we further assume that the number of entry nodes is small, then we derive a fixed-parameter tractable algorithm.
 
 We then propose two other fixed-parameter algorithms by exploiting the tree-like features. One is based on tree decomposition and requires a small tree width. The other assumes a small number of splitting nodes (nodes with multiple out-going edges). Finally, the last algorithm is converted into a graph convolutional neural network based heuristic, which scales to larger graphs with more splitting nodes.

----

## [1052] Anytime Multi-Agent Path Finding via Machine Learning-Guided Large Neighborhood Search

**Authors**: *Taoan Huang, Jiaoyang Li, Sven Koenig, Bistra Dilkina*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i9.21168](https://doi.org/10.1609/aaai.v36i9.21168)

**Abstract**:

Multi-Agent Path Finding (MAPF) is the problem of finding a set of collision-free paths for a team of agents in a common environment. MAPF is NP-hard to solve optimally and, in some cases, also bounded-suboptimally. It is thus time-consuming for (bounded-sub)optimal solvers to solve large MAPF instances. Anytime algorithms find solutions quickly for large instances and then improve them to close-to-optimal ones over time. In this paper, we improve the current state-of-the-art anytime solver MAPF-LNS, that first finds an initial solution fast and then repeatedly replans the paths of subsets of agents via Large Neighborhood Search (LNS). It generates the subsets of agents for replanning by randomized destroy heuristics, but not all of them increase the solution quality substantially. We propose to use machine learning to learn how to select a subset of agents from a collection of subsets, such that replanning increases the solution quality more. We show experimentally that our solver, MAPF-ML-LNS, significantly outperforms MAPF-LNS on the standard MAPF benchmark set in terms of both the speed of improving the solution and the final solution quality.

----

## [1053] MDPGT: Momentum-Based Decentralized Policy Gradient Tracking

**Authors**: *Zhanhong Jiang, Xian Yeow Lee, Sin Yong Tan, Kai Liang Tan, Aditya Balu, Young M. Lee, Chinmay Hegde, Soumik Sarkar*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i9.21169](https://doi.org/10.1609/aaai.v36i9.21169)

**Abstract**:

We propose a novel policy gradient method for multi-agent reinforcement learning, which leverages two different variance-reduction techniques and does not require large batches over iterations. Specifically, we propose a momentum-based decentralized policy gradient tracking (MDPGT) where a new momentum-based variance reduction technique is used to approximate the local policy gradient surrogate with importance sampling, and an intermediate parameter is adopted to track two consecutive policy gradient surrogates. MDPGT provably achieves the best available sample complexity of O(N -1 e -3) for converging to an e-stationary point of the global average of N local performance functions (possibly nonconcave). This outperforms the state-of-the-art sample complexity in decentralized model-free reinforcement learning and when initialized with a single trajectory, the sample complexity matches those obtained by the existing decentralized policy gradient methods. We further validate the theoretical claim for the Gaussian policy function. When the required error tolerance e is small enough, MDPGT leads to a linear speed up, which has been previously established in decentralized stochastic optimization, but not for reinforcement learning. Lastly, we provide empirical results on a multi-agent reinforcement learning benchmark environment to support our theoretical findings.

----

## [1054] Shard Systems: Scalable, Robust and Persistent Multi-Agent Path Finding with Performance Guarantees

**Authors**: *Christopher Leet, Jiaoyang Li, Sven Koenig*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i9.21170](https://doi.org/10.1609/aaai.v36i9.21170)

**Abstract**:

Modern multi-agent robotic systems increasingly require scalable, robust and persistent Multi-Agent Path Finding (MAPF) with performance guarantees. While many MAPF solvers that provide some of these properties exist, none provides them all. To fill this need, we propose a new MAPF framework, the shard system. A shard system partitions the workspace into geographic regions, called shards, linked by a novel system of buffers. Agents are routed optimally within a shard by a local controller to local goals set by a global controller. The buffer system novelly allows shards to plan with perfect parallelism, providing scalability. A novel global controller algorithm can rapidly generate an inter-shard routing plan for thousands of agents while minimizing the traffic routed through any shard. A novel workspace partitioning algorithm produces shards small enough to replan rapidly. These innovations allow a shard system to adjust its routing plan in real time if an agent is delayed or assigned a new goal, enabling robust, persistent MAPF. A shard system's local optimality and optimized inter-shard routing bring the sum-of-costs of its solutions to single-shot MAPF problems to < 20-60% of optimal on a diversity of workspaces. Its scalability allows it to plan paths for 1000s of agents in seconds. If any of their goals change or move actions fails, a shard system can replan in under a second.

----

## [1055] A Deeper Understanding of State-Based Critics in Multi-Agent Reinforcement Learning

**Authors**: *Xueguang Lyu, Andrea Baisero, Yuchen Xiao, Christopher Amato*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i9.21171](https://doi.org/10.1609/aaai.v36i9.21171)

**Abstract**:

Centralized Training for Decentralized Execution, where training is done in a centralized offline fashion, has become a popular solution paradigm in Multi-Agent Reinforcement Learning.
Many such methods take the form of actor-critic with state-based critics, since centralized training allows access to the true system state, which can be useful during training despite not being available at execution time.
State-based critics have become a common empirical choice, albeit one which has had limited theoretical justification or analysis.
In this paper, we show that state-based critics can introduce bias in the policy gradient estimates, potentially undermining the asymptotic guarantees of the algorithm.
We also show that, even if the state-based critics do not introduce any bias, they can still result in a larger gradient variance, contrary to the common intuition.
Finally, we show the effects of the theories in practice by comparing different forms of centralized critics on a wide range of common benchmarks, and detail how various environmental properties are related to the effectiveness of different types of critics.

----

## [1056] When Can the Defender Effectively Deceive Attackers in Security Games?

**Authors**: *Thanh Nguyen, Haifeng Xu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i9.21172](https://doi.org/10.1609/aaai.v36i9.21172)

**Abstract**:

This paper studies defender patrol deception in general Stackelberg security games (SSGs), where a defender attempts to alter the attacker's perception of the defender's patrolling intensity so as to influence the attacker's decision making. We are interested in understanding the complexity and effectiveness of optimal defender deception under different attacker behavior models. Specifically, we consider three different attacker strategies of response (to the defender's deception) with increasing sophistication, and design efficient polynomial-time algorithms to compute the equilibrium for each. Moreover, we prove formal separation for the effectiveness of patrol deception when facing an attacker of increasing sophistication, until it becomes even harmful to the defender when facing the most intelligent attacker we consider. Our results shed light on when and how deception should be used in SSGs. We conduct extensive experiments to illustrate our theoretical results in various game settings.

----

## [1057] Generalization in Mean Field Games by Learning Master Policies

**Authors**: *Sarah Perrin, Mathieu Laurière, Julien Pérolat, Romuald Élie, Matthieu Geist, Olivier Pietquin*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i9.21173](https://doi.org/10.1609/aaai.v36i9.21173)

**Abstract**:

Mean Field Games (MFGs) can potentially scale multi-agent systems to extremely large populations of agents. Yet, most of the literature assumes a single initial distribution for the agents, which limits the practical applications of MFGs. Machine Learning has the potential to solve a wider diversity of MFG problems thanks to generalizations capacities. We study how to leverage these generalization properties to learn policies enabling a typical agent to behave optimally against any population distribution. In reference to the Master equation in MFGs, we coin the term “Master policies” to describe them and we prove that a single Master policy provides a Nash equilibrium, whatever the initial distribution. We propose a method to learn such Master policies. Our approach relies on three ingredients: adding the current population distribution as part of the observation, approximating Master policies with neural networks, and training via Reinforcement Learning and Fictitious Play. We illustrate on numerical examples not only the efficiency of the learned Master policy but also its generalization capabilities beyond the distributions used for training.

----

## [1058] Finding Nontrivial Minimum Fixed Points in Discrete Dynamical Systems: Complexity, Special Case Algorithms and Heuristics

**Authors**: *Zirou Qiu, Chen Chen, Madhav V. Marathe, S. S. Ravi, Daniel J. Rosenkrantz, Richard Edwin Stearns, Anil Vullikanti*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i9.21174](https://doi.org/10.1609/aaai.v36i9.21174)

**Abstract**:

Networked discrete dynamical systems are often used to model the spread of contagions and decision-making by agents in coordination games. Fixed points of such dynamical systems represent configurations to which the system converges. In the dissemination of undesirable contagions (such as rumors and misinformation), convergence to fixed points with a small number of affected nodes is a desirable goal. Motivated by such considerations, we formulate a novel optimization problem of finding a nontrivial fixed point of the system with the minimum number of affected nodes. We establish that, unless P = NP, there is no polynomial-time algorithm for approximating a solution to this problem to within the factor n^(1 - epsilon) for any constant epsilon > 0. To cope with this computational intractability, we identify several special cases for which the problem can be solved efficiently. Further, we introduce an integer linear program to address the problem for networks of reasonable sizes. For solving the problem on larger networks, we propose a general heuristic framework along with greedy selection methods. Extensive experimental results on real-world networks demonstrate the effectiveness of the proposed heuristics. A full version of the manuscript, source code and data are
available at: https://github.com/bridgelessqiu/NMIN-FPE

----

## [1059] How Many Representatives Do We Need? The Optimal Size of a Congress Voting on Binary Issues

**Authors**: *Manon Revel, Tao Lin, Daniel Halpern*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i9.21175](https://doi.org/10.1609/aaai.v36i9.21175)

**Abstract**:

Aggregating opinions of a collection of agents is a question of interest to a broad array of researchers, ranging from ensemble-learning theorists to political scientists designing democratic institutions. This work investigates the optimal number of agents needed to decide on a binary issue under majority rule. We take an epistemic view where the issue at hand has a ground truth ``correct'' outcome and each one of n voters votes correctly with a fixed probability, known as their competence level or competence. These competencies come from a fixed distribution D. Observing the competencies, we must choose a specific group that will represent the population. Finally, voters sample a decision (either correct or not), and the group is correct as long as more than half the chosen representatives voted correctly. Assuming that we can identify the best experts, i.e., those with the highest competence, to form an epistemic congress we find that the optimal congress size should be linear in the population size. This result is striking because it holds even when allowing the top representatives to become arbitrarily accurate, choosing the correct outcome with probabilities approaching 1. We then analyze real-world data, observing that the actual sizes of representative bodies are much smaller than the optimal ones our theoretical results suggest. We conclude by examining under what conditions congresses of sub-optimal sizes would still outperform direct democracy, in which all voters vote. We find that a small congress would beat direct democracy if the rate at which the societal bias towards the ground truth decreases with the population size fast enough, and we quantify the speed needed for constant and polynomial congress sizes.

----

## [1060] Decentralized Mean Field Games

**Authors**: *Sriram Ganapathi Subramanian, Matthew E. Taylor, Mark Crowley, Pascal Poupart*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i9.21176](https://doi.org/10.1609/aaai.v36i9.21176)

**Abstract**:

Multiagent reinforcement learning algorithms have not been widely adopted in large scale environments with many agents as they often scale poorly with the number of agents. Using mean field theory to aggregate agents has been proposed as a solution to this problem. However, almost all previous methods in this area make a strong assumption of a centralized system where all the agents in the environment learn the same policy and are effectively indistinguishable from each other. In this paper, we relax this assumption about indistinguishable agents and propose a new mean field system known as Decentralized Mean Field Games, where each agent can be quite different from others. All agents learn independent policies in a decentralized fashion, based on their local observations. We define a theoretical solution concept for this system and provide a fixed point guarantee for a Q-learning based algorithm in this system. A practical consequence of our approach is that we can address a `chicken-and-egg' problem in empirical mean field reinforcement learning algorithms. Further, we provide Q-learning and actor-critic algorithms that use the decentralized mean field learning approach and give stronger performances compared to common baselines in this area. In our setting, agents do not need to be clones of each other and learn in a fully decentralized fashion. Hence, for the first time, we show the application of mean field learning methods in fully competitive environments, large-scale continuous action space environments, and other environments with heterogeneous agents. Importantly, we also apply the mean field method in a ride-sharing problem using a real-world dataset. We propose a decentralized solution to this problem, which is more practical than existing centralized training methods.

----

## [1061] Incentivizing Collaboration in Machine Learning via Synthetic Data Rewards

**Authors**: *Sebastian Shenghong Tay, Xinyi Xu, Chuan Sheng Foo, Bryan Kian Hsiang Low*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i9.21177](https://doi.org/10.1609/aaai.v36i9.21177)

**Abstract**:

This paper presents a novel collaborative generative modeling (CGM) framework that incentivizes collaboration among self-interested parties to contribute data to a pool for training a generative model (e.g., GAN), from which synthetic data are drawn and distributed to the parties as rewards commensurate to their contributions. Distributing synthetic data as rewards (instead of trained models or money) offers task- and model-agnostic benefits for downstream learning tasks and is less likely to violate data privacy regulation. To realize the framework, we firstly propose a data valuation function using maximum mean discrepancy (MMD) that values data based on its quantity and quality in terms of its closeness to the true data distribution and provide theoretical results guiding the kernel choice in our MMD-based data valuation function. Then, we formulate the reward scheme as a linear optimization problem that when solved, guarantees certain incentives such as fairness in the CGM framework. We devise a weighted sampling algorithm for generating synthetic data to be distributed to each party as reward such that the value of its data and the synthetic data combined matches its assigned reward value by the reward scheme. We empirically show using simulated and real-world datasets that the parties' synthetic data rewards are commensurate to their contributions.

----

## [1062] Learning the Optimal Recommendation from Explorative Users

**Authors**: *Fan Yao, Chuanhao Li, Denis Nekipelov, Hongning Wang, Haifeng Xu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i9.21178](https://doi.org/10.1609/aaai.v36i9.21178)

**Abstract**:

We propose a new problem setting to study the sequential interactions between a recommender system and a user. Instead of assuming the user is omniscient, static, and explicit, as the classical practice does, we sketch a more realistic user behavior model, under which the user: 1) rejects recommendations if they are clearly worse than others; 2) updates her utility estimation based on rewards from her accepted recommendations; 3) withholds realized rewards from the system. We formulate the interactions between the system and such an explorative user in a K-armed bandit framework and study the problem of learning the optimal recommendation on the system side. We show that efficient system learning is still possible but is more difficult. In particular, the system can identify the best arm with probability at least 1-delta within O(1/delta) interactions, and we prove this is tight. Our finding contrasts the result for the problem of best arm identification with fixed confidence, in which the best arm can be identified with probability 1-delta within O(log(1/delta)) interactions. This gap illustrates the inevitable cost the system has to pay when it learns from an explorative user's revealed preferences on its recommendations rather than from the realized rewards.

----

## [1063] Multi-Agent Incentive Communication via Decentralized Teammate Modeling

**Authors**: *Lei Yuan, Jianhao Wang, Fuxiang Zhang, Chenghe Wang, Zongzhang Zhang, Yang Yu, Chongjie Zhang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i9.21179](https://doi.org/10.1609/aaai.v36i9.21179)

**Abstract**:

Effective communication can improve coordination in cooperative multi-agent reinforcement learning (MARL). One popular communication scheme is exchanging agents' local observations or latent embeddings and using them to augment individual local policy input. Such a communication paradigm can reduce uncertainty for local decision-making and induce implicit coordination. However, it enlarges agents' local policy spaces and increases learning complexity, leading to poor coordination in complex settings. To handle this limitation, this paper proposes a novel framework named Multi-Agent Incentive Communication (MAIC) that allows each agent to learn to generate incentive messages and bias other agents' value functions directly, resulting in effective explicit coordination. Our method firstly learns targeted teammate models, with which each agent can anticipate the teammate's action selection and generate tailored messages to specific agents. We further introduce a novel regularization to leverage interaction sparsity and improve communication efficiency. MAIC is agnostic to specific MARL algorithms and can be flexibly integrated with different value function factorization methods. Empirical results demonstrate that our method significantly outperforms baselines and achieves excellent performance on multiple cooperative MARL tasks.

----

## [1064] MLink: Linking Black-Box Models for Collaborative Multi-Model Inference

**Authors**: *Mu Yuan, Lan Zhang, Xiang-Yang Li*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i9.21180](https://doi.org/10.1609/aaai.v36i9.21180)

**Abstract**:

The cost efficiency of model inference is critical to real-world machine learning (ML) applications, especially for delay-sensitive tasks and resource-limited devices. A typical dilemma is: in order to provide complex intelligent services (e.g. smart city), we need inference results of multiple ML models, but the cost budget (e.g. GPU memory) is not enough to run all of them. In this work, we study underlying relationships among black-box ML models and propose a novel learning task: model linking. Model linking aims to bridge the knowledge of different black-box models by learning mappings (dubbed model links) between their output spaces. Based on model links, we developed a scheduling algorithm, named MLink. Through collaborative multi-model inference enabled by model links, MLink can improve the accuracy of obtained inference results under the cost budget. We evaluated MLink on a multi-modal dataset with seven different ML models and two real-world video analytics systems with six ML models and 3,264 hours of video. Experimental results show that our proposed model links can be effectively built among various black-box models. Under the budget of GPU memory, MLink can save 66.7% inference computations while preserving 94% inference accuracy, which outperforms multi-task learning, deep reinforcement learning-based scheduler and frame filtering baselines.

----

## [1065] Equilibrium Finding in Normal-Form Games via Greedy Regret Minimization

**Authors**: *Hugh Zhang, Adam Lerer, Noam Brown*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i9.21181](https://doi.org/10.1609/aaai.v36i9.21181)

**Abstract**:

We extend the classic regret minimization framework for approximating equilibria in normal-form games by greedily weighing iterates based on regrets observed at runtime. Theoretically, our method retains all previous convergence rate guarantees. Empirically, experiments on large randomly generated games and normal-form subgames of the AI benchmark Diplomacy show that greedy weights outperforms previous methods whenever sampling is used, sometimes by several orders of magnitude.

----

## [1066] Why Fair Labels Can Yield Unfair Predictions: Graphical Conditions for Introduced Unfairness

**Authors**: *Carolyn Ashurst, Ryan Carey, Silvia Chiappa, Tom Everitt*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i9.21182](https://doi.org/10.1609/aaai.v36i9.21182)

**Abstract**:

In addition to reproducing discriminatory relationships in the training data, machine learning (ML) systems can also introduce or amplify discriminatory effects. We refer to this as introduced unfairness, and investigate the conditions under which it may arise. To this end, we propose introduced total variation as a measure of introduced unfairness, and establish graphical conditions under which it may be incentivised to occur. These criteria imply that adding the sensitive attribute as a feature removes the incentive for introduced variation under well-behaved loss functions. Additionally, taking a causal perspective, introduced path-specific effects shed light on the issue of when specific paths should be considered fair.

----

## [1067] Incorporating Item Frequency for Differentially Private Set Union

**Authors**: *Ricardo Silva Carvalho, Ke Wang, Lovedeep Singh Gondara*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i9.21183](https://doi.org/10.1609/aaai.v36i9.21183)

**Abstract**:

We study the problem of releasing the set union of users' items subject to differential privacy. Previous approaches consider only the set of items for each user as the input. We propose incorporating the item frequency, which is typically available in set union problems, to boost the utility of private mechanisms. However, using the global item frequency over all users would largely increase privacy loss. We propose to use the local item frequency of each user to approximate the global item frequency without incurring additional privacy loss. 
 Local item frequency allows us to design greedy set union mechanisms that are differentially private, which is impossible for previous greedy proposals. Moreover, while all previous works have to use uniform sampling to limit the number of items each user would contribute to, our construction eliminates the sampling step completely and allows our mechanisms to consider all of the users' items. 
 Finally, we propose to transfer the knowledge of the global item frequency from a public dataset into our mechanism, which further boosts utility even when the public and private datasets are from different domains. We evaluate the proposed methods on multiple real-life datasets.

----

## [1068] Cosine Model Watermarking against Ensemble Distillation

**Authors**: *Laurent Charette, Lingyang Chu, Yizhou Chen, Jian Pei, Lanjun Wang, Yong Zhang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i9.21184](https://doi.org/10.1609/aaai.v36i9.21184)

**Abstract**:

Many model watermarking methods have been developed to prevent valuable deployed commercial models from being stealthily stolen by model distillations.
 However, watermarks produced by most existing model watermarking methods can be easily evaded by ensemble distillation, because averaging the outputs of multiple ensembled models can significantly reduce or even erase the watermarks.
 In this paper, we focus on tackling the challenging task of defending against ensemble distillation.
 We propose a novel watermarking technique named CosWM to achieve outstanding model watermarking performance against ensemble distillation.
 CosWM is not only elegant in design, but also comes with desirable theoretical guarantees.
 Our extensive experiments on public data sets demonstrate the excellent performance of CosWM and its advantages over the state-of-the-art baselines.

----

## [1069] Towards Debiasing DNN Models from Spurious Feature Influence

**Authors**: *Mengnan Du, Ruixiang Tang, Weijie Fu, Xia Hu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i9.21185](https://doi.org/10.1609/aaai.v36i9.21185)

**Abstract**:

Recent studies indicate that deep neural networks (DNNs) are prone to show discrimination towards certain demographic groups. We observe that algorithmic discrimination can be explained by the high reliance of the models on fairness sensitive features. Motivated by this observation, we propose to achieve fairness by suppressing the DNN models from capturing the spurious correlation between those fairness sensitive features with the underlying task. Specifically, we firstly train a bias-only teacher model which is explicitly encouraged to maximally employ fairness sensitive features for prediction. The teacher model then counter-teaches a debiased student model so that the interpretation of the student model is orthogonal to the interpretation of the teacher model. The key idea is that since the teacher model relies explicitly on fairness sensitive features for prediction, the orthogonal interpretation loss enforces the student network to reduce its reliance on sensitive features and instead capture more task relevant features for prediction. Experimental analysis indicates that our framework substantially reduces the model's attention on fairness sensitive features. Experimental results on four datasets further validate that our framework has consistently improved the fairness with respect to three group fairness metrics, with a comparable or even better accuracy.

----

## [1070] Path-Specific Objectives for Safer Agent Incentives

**Authors**: *Sebastian Farquhar, Ryan Carey, Tom Everitt*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i9.21186](https://doi.org/10.1609/aaai.v36i9.21186)

**Abstract**:

We present a general framework for training safe agents whose naive incentives are unsafe. As an example, manipulative or deceptive behaviour can improve rewards but should be avoided. Most approaches fail here: agents maximize expected return by any means necessary. We formally describe settings with `delicate' parts of the state which should not be used as a means to an end. We then train agents to maximize the causal effect of actions on the expected return which is not mediated by the delicate parts of state, using Causal Influence Diagram analysis. The resulting agents have no incentive to control the delicate state. We further show how our framework unifies and generalizes existing proposals.

----

## [1071] Algorithmic Fairness Verification with Graphical Models

**Authors**: *Bishwamittra Ghosh, Debabrota Basu, Kuldeep S. Meel*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i9.21187](https://doi.org/10.1609/aaai.v36i9.21187)

**Abstract**:

In recent years, machine learning (ML) algorithms have been deployed in safety-critical and high-stake decision-making, where the fairness of algorithms is of paramount importance. Fairness in ML centers on detecting bias towards certain demographic populations induced by an ML classifier and proposes algorithmic solutions to mitigate the bias with respect to different fairness definitions. To this end, several fairness verifiers have been proposed that compute the bias in the prediction of an ML classifier—essentially beyond a finite dataset—given the probability distribution of input features. In the context of verifying linear classifiers, existing fairness verifiers are limited by accuracy due to imprecise modeling of correlations among features and scalability due to restrictive formulations of the classifiers as SSAT/SMT formulas or by sampling.


In this paper, we propose an efficient fairness verifier, called FVGM, that encodes the correlations among features as a Bayesian network. In contrast to existing verifiers, FVGM proposes a stochastic subset-sum based approach for verifying linear classifiers. Experimentally, we show that FVGM leads to an accurate and scalable assessment for more diverse families of fairness-enhancing algorithms, fairness attacks, and group/causal fairness metrics than the state-of-the-art fairness verifiers. We also demonstrate that FVGM facilitates the computation of fairness influence functions as a stepping stone to detect the source of bias induced by subsets of features.

----

## [1072] Achieving Long-Term Fairness in Sequential Decision Making

**Authors**: *Yaowei Hu, Lu Zhang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i9.21188](https://doi.org/10.1609/aaai.v36i9.21188)

**Abstract**:

In this paper, we propose a framework for achieving long-term fair sequential decision making. By conducting both the hard and soft interventions, we propose to take path-specific effects on the time-lagged causal graph as a quantitative tool for measuring long-term fairness. The problem of fair sequential decision making is then formulated as a constrained optimization problem with the utility as the objective and the long-term and short-term fairness as constraints. We show that such an optimization problem can be converted to a performative risk optimization. Finally, repeated risk minimization (RRM) is used for model training, and the convergence of RRM is theoretically analyzed. The empirical evaluation shows the effectiveness of the proposed algorithm on synthetic and semi-synthetic temporal datasets.

----

## [1073] Fairness without Imputation: A Decision Tree Approach for Fair Prediction with Missing Values

**Authors**: *Haewon Jeong, Hao Wang, Flávio P. Calmon*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i9.21189](https://doi.org/10.1609/aaai.v36i9.21189)

**Abstract**:

We investigate the fairness concerns of training a machine learning model using data with missing values. Even though there are a number of fairness intervention methods in the literature, most of them require a complete training set as input. In practice, data can have missing values, and data missing patterns can depend on group attributes (e.g. gender or race). Simply applying off-the-shelf fair learning algorithms to an imputed dataset may lead to an unfair model. In this paper, we first theoretically analyze different sources of discrimination risks when training with an imputed dataset. Then, we propose an integrated approach based on decision trees that does not require a separate process of imputation and learning. Instead, we train a tree with missing incorporated as attribute (MIA), which does not require explicit imputation, and we optimize a fairness-regularized objective function. We demonstrate that our approach outperforms existing fairness intervention methods applied to an imputed dataset, through several experiments on real-world datasets.

----

## [1074] Shaping Noise for Robust Attributions in Neural Stochastic Differential Equations

**Authors**: *Sumit Kumar Jha, Rickard Ewetz, Alvaro Velasquez, Arvind Ramanathan, Susmit Jha*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i9.21190](https://doi.org/10.1609/aaai.v36i9.21190)

**Abstract**:

Neural SDEs with Brownian motion as noise lead to smoother attributions than traditional ResNets. Various attribution methods such as saliency maps, integrated gradients, DeepSHAP and DeepLIFT have been shown to be more robust for neural SDEs than for ResNets using the recently proposed sensitivity metric. In this paper, we show that neural SDEs with adaptive attribution-driven noise lead to even more robust attributions and smaller sensitivity metrics than traditional neural SDEs with Brownian motion as noise. In particular, attribution-driven shaping of noise leads to 6.7%, 6.9% and 19.4% smaller sensitivity metric for integrated gradients computed on three discrete approximations of neural SDEs with standard Brownian motion noise: stochastic ResNet-50, WideResNet-101 and ResNeXt-101 models respectively. The neural SDE model with adaptive attribution-driven noise leads to 25.7% and 4.8% improvement in the SIC metric over traditional ResNets and Neural SDEs with Brownian motion as noise. To the best of our knowledge, we are the first to propose the use of attributions for shaping the noise injected in neural SDEs, and demonstrate that this process leads to more robust attributions than traditional neural SDEs with standard Brownian motion as noise.

----

## [1075] Certified Robustness of Nearest Neighbors against Data Poisoning and Backdoor Attacks

**Authors**: *Jinyuan Jia, Yupei Liu, Xiaoyu Cao, Neil Zhenqiang Gong*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i9.21191](https://doi.org/10.1609/aaai.v36i9.21191)

**Abstract**:

Data poisoning attacks and backdoor attacks aim to corrupt a machine learning classifier via modifying, adding, and/or removing some carefully selected training examples, such that the corrupted classifier makes incorrect predictions as the attacker desires. The key idea of state-of-the-art certified defenses against data poisoning attacks and backdoor attacks is to create a majority vote mechanism to predict the label of a testing example. Moreover, each voter is a base classifier trained on a subset of the training dataset. Classical simple learning algorithms such as k nearest neighbors (kNN) and radius nearest neighbors (rNN) have intrinsic majority vote mechanisms. In this work, we show that the intrinsic majority vote mechanisms in kNN and rNN already provide certified robustness guarantees against data poisoning attacks and backdoor attacks. Moreover, our evaluation results on MNIST and CIFAR10 show that the intrinsic certified robustness guarantees of kNN and rNN outperform those provided by state-of-the-art certified defenses. Our results serve as standard baselines for future certified defenses against data poisoning attacks and backdoor attacks.

----

## [1076] On the Fairness of Causal Algorithmic Recourse

**Authors**: *Julius von Kügelgen, Amir-Hossein Karimi, Umang Bhatt, Isabel Valera, Adrian Weller, Bernhard Schölkopf*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i9.21192](https://doi.org/10.1609/aaai.v36i9.21192)

**Abstract**:

Algorithmic fairness is typically studied from the perspective of predictions. Instead, here we investigate fairness from the perspective of recourse actions suggested to individuals to remedy an unfavourable classification. We propose two new fair-ness criteria at the group and individual level, which—unlike prior work on equalising the average group-wise distance from the decision boundary—explicitly account for causal relationships between features, thereby capturing downstream effects of recourse actions performed in the physical world. We explore how our criteria relate to others, such as counterfactual fairness, and show that fairness of recourse is complementary to fairness of prediction. We study theoretically and empirically how to enforce fair causal recourse by altering the classifier and perform a case study on the Adult dataset. Finally, we discuss whether fairness violations in the data generating process revealed by our criteria may be better addressed by societal interventions as opposed to constraints on the classifier.

----

## [1077] DeepAuth: A DNN Authentication Framework by Model-Unique and Fragile Signature Embedding

**Authors**: *Yingjie Lao, Weijie Zhao, Peng Yang, Ping Li*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i9.21193](https://doi.org/10.1609/aaai.v36i9.21193)

**Abstract**:

Along with the evolution of deep neural networks (DNNs) in many real-world applications, the complexity of model building has also dramatically increased. Therefore, it is vital to protect the intellectual property (IP) of the model builder and ensure the trustworthiness of the deployed models. Meanwhile, adversarial attacks on DNNs (e.g., backdoor and poisoning attacks) that seek to inject malicious behaviors have been investigated recently, demanding a means for verifying the integrity of the deployed model to protect the users. This paper presents a novel DNN authentication framework DeepAuth that embeds a unique and fragile signature to each protected DNN model. Our approach exploits sensitive key samples that are well crafted from the input space to latent space and then to logit space for producing signatures. After embedding, each model will respond distinctively to these key samples, which creates a model-unique signature as a strong tool for authentication and user identity. The signature embedding process is also designed to ensure the fragility of the signature, which can be used to detect malicious modifications such that an illegitimate user or an altered model should not have the intact signature. Extensive evaluations on various models over a wide range of datasets demonstrate the effectiveness and efficiency of the proposed DeepAuth.

----

## [1078] Fast Sparse Decision Tree Optimization via Reference Ensembles

**Authors**: *Hayden McTavish, Chudi Zhong, Reto Achermann, Ilias Karimalis, Jacques Chen, Cynthia Rudin, Margo I. Seltzer*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i9.21194](https://doi.org/10.1609/aaai.v36i9.21194)

**Abstract**:

Sparse decision tree optimization has been one of the most fundamental problems in AI since its inception and is a challenge at the core of interpretable machine learning. Sparse decision tree optimization is computationally hard, and despite steady effort since the 1960's, breakthroughs have been made on the problem only within the past few years, primarily on the problem of finding optimal sparse decision trees. However, current state-of-the-art algorithms often require impractical amounts of computation time and memory to find optimal or near-optimal trees for some real-world datasets, particularly those having several continuous-valued features. Given that the search spaces of these decision tree optimization problems are massive, can we practically hope to find a sparse decision tree that competes in accuracy with a black box machine learning model? We address this problem via smart guessing strategies that can be applied to any optimal branch-and-bound-based decision tree algorithm. The guesses come from knowledge gleaned from black box models. We show that by using these guesses, we can reduce the run time by multiple orders of magnitude while providing bounds on how far the resulting trees can deviate from the black box's accuracy and expressive power. Our approach enables guesses about how to bin continuous features, the size of the tree, and lower bounds on the error for the optimal decision tree. Our experiments show that in many cases we can rapidly construct sparse decision trees that match the accuracy of black box models. To summarize: when you are having trouble optimizing, just guess.

----

## [1079] Unsupervised Causal Binary Concepts Discovery with VAE for Black-Box Model Explanation

**Authors**: *Thien Q. Tran, Kazuto Fukuchi, Youhei Akimoto, Jun Sakuma*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i9.21195](https://doi.org/10.1609/aaai.v36i9.21195)

**Abstract**:

We aim to explain a black-box classifier with the form: "data X is classified as class Y because X has A, B and does not have C" in which A, B, and C are high-level concepts. The challenge is that we have to discover in an unsupervised manner a set of concepts, i.e., A, B and C, that is useful for explaining the classifier. We first introduce a structural generative model that is suitable to express and discover such concepts. We then propose a learning process that simultaneously learns the data distribution and encourages certain concepts to have a large causal influence on the classifier output. Our method also allows easy integration of user's prior knowledge to induce high interpretability of concepts. Finally, using multiple datasets, we demonstrate that the proposed method can discover useful concepts for explanation in this form.

----

## [1080] Do Feature Attribution Methods Correctly Attribute Features?

**Authors**: *Yilun Zhou, Serena Booth, Marco Túlio Ribeiro, Julie Shah*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i9.21196](https://doi.org/10.1609/aaai.v36i9.21196)

**Abstract**:

Feature attribution methods are popular in interpretable machine learning. These methods compute the attribution of each input feature to represent its importance, but there is no consensus on the definition of "attribution", leading to many competing methods with little systematic evaluation, complicated in particular by the lack of ground truth attribution. To address this, we propose a dataset modification procedure to induce such ground truth. Using this procedure, we evaluate three common methods: saliency maps, rationales, and attentions. We identify several deficiencies and add new perspectives to the growing body of evidence questioning the correctness and reliability of these methods applied on datasets in the wild. We further discuss possible avenues for remedy and recommend new attribution methods to be tested against ground truth before deployment. The code and appendix are available at https://yilunzhou.github.io/feature-attribution-evaluation/.

----

## [1081] Formal Semantics and Formally Verified Validation for Temporal Planning

**Authors**: *Mohammad Abdulaziz, Lukas Koller*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i9.21197](https://doi.org/10.1609/aaai.v36i9.21197)

**Abstract**:

We present a simple and concise semantics for temporal planning. Our semantics are developed and formalised in the logic of the interactive theorem prover Isabelle/HOL. We derive from those semantics a validation algorithm for temporal planning and show, using a formal proof in Isabelle/HOL, that this validation algorithm implements our semantics. We experimentally evaluate our verified validation algorithm and show that it is practical.

----

## [1082] Goal Recognition as Reinforcement Learning

**Authors**: *Leonardo Amado, Reuth Mirsky, Felipe Meneguzzi*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i9.21198](https://doi.org/10.1609/aaai.v36i9.21198)

**Abstract**:

Most approaches for goal recognition rely on specifications of the possible dynamics of the actor in the environment when pursuing a goal. These specifications suffer from two key issues. First, encoding these dynamics requires careful design by a domain expert, which is often not robust to noise at recognition time. Second, existing approaches often need costly real-time computations to reason about the likelihood of each potential goal. In this paper, we develop a framework that combines model-free reinforcement learning and goal recognition to alleviate the need for careful, manual domain design, and the need for costly online executions. This framework consists of two main stages: Offline learning of policies or utility functions for each potential goal, and online inference. We provide a first instance of this framework using tabular Q-learning for the learning stage, as well as three measures that can be used to perform the inference stage. The resulting instantiation achieves state-of-the-art performance against goal recognizers on standard evaluation domains and superior performance in noisy environments.

----

## [1083] Online Search with Best-Price and Query-Based Predictions

**Authors**: *Spyros Angelopoulos, Shahin Kamali, Dehou Zhang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i9.21199](https://doi.org/10.1609/aaai.v36i9.21199)

**Abstract**:

In the online (time-series) search problem, a player is presented with a sequence of prices which are revealed in an online manner. In the standard definition of the problem, for each revealed price, the player must decide irrevocably whether to accept or reject it, without knowledge of future prices (other than an upper and a lower bound on their extreme values), and the objective is to minimize the competitive ratio, namely the worst case ratio between the maximum price in the sequence and the one selected by the player. The problem formulates several applications of decision-making in the face of uncertainty on the revealed samples.
 
 Previous work on this problem has largely assumed extreme scenarios in which either the player has almost no information about the input, or the player is provided with some powerful, and error-free advice. In this work, we study learning-augmented algorithms, in which there is a potentially erroneous prediction concerning the input. Specifically, we consider two different settings: the setting in which the prediction is related to the maximum price in the sequence, as well as well as the setting in which the prediction is obtained as a response to a number of binary queries. For both settings, we provide tight, or near-tight upper and lower bounds on the worst-case performance of search algorithms as a function of the prediction error. We also provide experimental results on data obtained from stock exchange markets that confirm the theoretical analysis, and explain how our techniques can be applicable to other learning-augmented applications.

----

## [1084] Extended Goal Recognition Design with First-Order Computation Tree Logic

**Authors**: *Tsz-Chiu Au*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i9.21200](https://doi.org/10.1609/aaai.v36i9.21200)

**Abstract**:

Goal recognition design (GRD) is the task of modifying environments for aiding observers to recognize the objectives of agents during online observations. The worst case distinctiveness (WCD), a widely used performance measure in GRD research, can fail to provide useful guidance to the redesign process when some goals are too hard to be distinguished.  Moreover, the existing WCD-based approaches do not work when an agent aims for a sequence of goals instead of just one goal. The paper presents a new GRD framework called extended goal recognition design (EGRD) for goal recognition that involves multiple goals. The objective of EGRD is to modify an environment to minimize the worst case distinctiveness of a goal condition that describes how an agent can reach a set of goals.  A goal condition can be formally expressed in first-order computation tree logic (FO-CTL) that can be evaluated by model checking.  We introduce a novel graphical representation of FO-CTL sentences that is suitable for extended goal recognition.  Moreover, we present a search algorithm for EGRD with a novel caching mechanism.  Our experimental results show that the caching mechanism can greatly speed up our EGRD search algorithm by reusing the previous evaluation of FO-CTL sentences.

----

## [1085] Sampling-Based Robust Control of Autonomous Systems with Non-Gaussian Noise

**Authors**: *Thom S. Badings, Alessandro Abate, Nils Jansen, David Parker, Hasan A. Poonawala, Mariëlle Stoelinga*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i9.21201](https://doi.org/10.1609/aaai.v36i9.21201)

**Abstract**:

Controllers for autonomous systems that operate in safety-critical settings must account for stochastic disturbances. Such disturbances are often modeled as process noise, and common assumptions are that the underlying distributions are known and/or Gaussian. In practice, however, these assumptions may be unrealistic and can lead to poor approximations of the true noise distribution. We present a novel planning method that does not rely on any explicit representation of the noise distributions. In particular, we address the problem of computing a controller that provides probabilistic guarantees on safely reaching a target. First, we abstract the continuous system into a discrete-state model that captures noise by probabilistic transitions between states. As a key contribution, we adapt tools from the scenario approach to compute probably approximately correct (PAC) bounds on these transition probabilities, based on a finite number of samples of the noise. We capture these bounds in the transition probability intervals of a so-called interval Markov decision process (iMDP). This iMDP is robust against uncertainty in the transition probabilities, and the tightness of the probability intervals can be controlled through the number of samples. We use state-of-the-art verification techniques to provide guarantees on the iMDP, and compute a controller for which these guarantees carry over to the autonomous system. Realistic benchmarks show the practical applicability of our method, even when the iMDP has millions of states or transitions.

----

## [1086] Synthesis from Satisficing and Temporal Goals

**Authors**: *Suguman Bansal, Lydia E. Kavraki, Moshe Y. Vardi, Andrew M. Wells*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i9.21202](https://doi.org/10.1609/aaai.v36i9.21202)

**Abstract**:

Reactive synthesis from high-level specifications that combine hard constraints expressed in Linear Temporal Logic (LTL) with soft constraints expressed by discounted sum (DS) rewards has applications in planning and reinforcement learning. An existing approach combines techniques from LTL synthesis with optimization for the DS rewards but has failed to yield a sound algorithm. An alternative approach combining LTL synthesis with satisficing DS rewards (rewards that achieve a threshold) is sound and complete for integer discount factors, but, in practice, a fractional discount factor is desired. This work extends the existing satisficing approach, presenting the first sound algorithm for synthesis from LTL and DS rewards with fractional discount factors. The utility of our algorithm is demonstrated on robotic planning domains.

----

## [1087] Making Translations to Classical Planning Competitive with Other HTN Planners

**Authors**: *Gregor Behnke, Florian Pollitt, Daniel Höller, Pascal Bercher, Ron Alford*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i9.21203](https://doi.org/10.1609/aaai.v36i9.21203)

**Abstract**:

Translation-based approaches to planning allow for solving problems in complex and expressive formalisms via the means of highly efficient solvers for simpler formalisms.
To be effective, these translations have to be constructed appropriately. The current existing translation of the highly expressive formalism of HTN planning into the more simple formalism of classical planning is not on par with the performance of current dedicated HTN planners. With our contributions in this paper, we close this gap: we describe new versions of the translation that reach the performance of state-of-the-art dedicated HTN planners. We present new translation techniques both for the special case of totally-ordered HTNs as well as for the general partially-ordered case. In the latter, we show that our new translation generates only linearly many actions, while the previous encoding generates and exponential number of actions.

----

## [1088] PlanVerb: Domain-Independent Verbalization and Summary of Task Plans

**Authors**: *Gerard Canal, Senka Krivic, Paul Luff, Andrew Coles*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i9.21204](https://doi.org/10.1609/aaai.v36i9.21204)

**Abstract**:

For users to trust planning algorithms, they must be able to understand the planner's outputs and the reasons for each action selection. This output does not tend to be user-friendly, often consisting of sequences of parametrised actions or task networks. And these may not be practical for non-expert users who may find it easier to read natural language descriptions. In this paper, we propose PlanVerb, a domain and planner-independent method for the verbalization of task plans. It is based on semantic tagging of actions and predicates. Our method can generate natural language descriptions of plans including causal explanations. The verbalized plans can be summarized by compressing the actions that act on the same parameters. We further extend the concept of verbalization space, previously applied to robot navigation, and apply it to planning to generate different kinds of plan descriptions for different user requirements. Our method can deal with PDDL and RDDL domains, provided that they are tagged accordingly. Our user survey evaluation shows that users can read our automatically generated plan descriptions and that the explanations help them answer questions about the plan.

----

## [1089] Competing for Resources: Estimating Adversary Strategy for Effective Plan Generation

**Authors**: *Lukás Chrpa, Pavel Rytír, Rostislav Horcík, Stefan Edelkamp*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i9.21205](https://doi.org/10.1609/aaai.v36i9.21205)

**Abstract**:

Effective decision making while competing for limited resources in adversarial environments is important for many real-world applications (e.g. two Taxi companies competing for customers). Decision-making techniques such as Automated planning have to take into account possible actions of adversary (or competing) agents. That said, the agent should know what the competitor will likely do and then generate its plan accordingly. In this paper we propose a novel approach for estimating strategies of the adversary (or the competitor), sampling its actions that might hinder agent's goals by interfering with the agent's actions. The estimated competitor strategies are used in plan generation such that agent's actions have to be applied prior to the ones of the competitor, whose estimated times dictate the deadlines. We empirically evaluate our approach leveraging sampling of competitor's actions by comparing it to the naive approach optimising the make-span (not taking the competing agent into account at all) and to Nash Equilibrium (mixed) strategies.

----

## [1090] The FF Heuristic for Lifted Classical Planning

**Authors**: *Augusto B. Corrêa, Florian Pommerening, Malte Helmert, Guillem Francès*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i9.21206](https://doi.org/10.1609/aaai.v36i9.21206)

**Abstract**:

Heuristics for lifted planning are not yet as informed as the best heuristics for ground planning. Recent work introduced the idea of using Datalog programs to compute the additive heuristic over lifted tasks. Based on this work, we show how to compute the more informed FF heuristic in a lifted manner. We extend the Datalog program with executable annotations that can also be used to define other delete-relaxation heuristics. In our experiments, we show that a planner using the lifted FF implementation produces state-of-the-art results for lifted planners. It also reduces the gap to state-of-the-art ground planners in domains where grounding is feasible.

----

## [1091] Inconsistent Planning: When in Doubt, Toss a Coin!

**Authors**: *Yuriy Dementiev, Fedor V. Fomin, Artur Ignatiev*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i9.21207](https://doi.org/10.1609/aaai.v36i9.21207)

**Abstract**:

One of the most widespread human behavioral biases is the present bias -- the tendency to overestimate current costs by a bias factor. Kleinberg and Oren (2014) introduced an elegant graph-theoretical model of inconsistent planning capturing the behavior of a present-biased agent accomplishing a set of actions. The essential measure of the system introduced by Kleinberg and Oren is the cost of irrationality -- the ratio of the total cost of the actions performed by the present-biased agent to the optimal cost. This measure is vital for a task designer to estimate the aftermaths of human behavior related to time-inconsistent planning, including procrastination and abandonment. 
 As we prove in this paper, the cost of irrationality is highly susceptible to the agent's choices when faced with a few possible actions of equal estimated costs. To address this issue, we propose a modification of Kleinberg-Oren's model of inconsistent planning. In our model, when an agent selects from several options of minimum prescribed cost, he uses a randomized procedure. We explore the algorithmic complexity of computing and estimating the cost of irrationality in the new model.

----

## [1092] Robustification of Online Graph Exploration Methods

**Authors**: *Franziska Eberle, Alexander Lindermayr, Nicole Megow, Lukas Nölke, Jens Schlöter*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i9.21208](https://doi.org/10.1609/aaai.v36i9.21208)

**Abstract**:

Exploring unknown environments is a fundamental task in many domains, e.g., robot navigation, network security, and internet search. We initiate the study of a learning-augmented variant of the classical, notoriously hard online graph exploration problem by adding access to machine-learned predictions. We propose an algorithm that naturally integrates predictions into the well-known Nearest Neighbor (NN) algorithm and significantly outperforms any known online algorithm if the prediction is of high accuracy while maintaining good guarantees when the prediction is of poor quality. We provide theoretical worst-case bounds that gracefully degrade with the prediction error, and we complement them by computational experiments that confirm our results. Further, we extend our concept to a general framework to robustify algorithms. By interpolating carefully between a given algorithm and NN, we prove new performance bounds that leverage the individual good performance on particular inputs while establishing robustness to arbitrary inputs.

----

## [1093] Explainable Planner Selection for Classical Planning

**Authors**: *Patrick Ferber, Jendrik Seipp*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i9.21209](https://doi.org/10.1609/aaai.v36i9.21209)

**Abstract**:

Since no classical planner consistently outperforms all others, it is important to select a planner that works well for a given classical planning task. The two strongest approaches for planner selection use image and graph convolutional neural networks. They have the drawback that the learned models are complicated and uninterpretable. To obtain explainable models, we identify a small set of simple task features and show that elementary and interpretable machine learning techniques can use these features to solve roughly as many tasks as the complex approaches based on neural networks.

----

## [1094] Operator-Potential Heuristics for Symbolic Search

**Authors**: *Daniel Fiser, Álvaro Torralba, Jörg Hoffmann*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i9.21210](https://doi.org/10.1609/aaai.v36i9.21210)

**Abstract**:

Symbolic search, using Binary Decision Diagrams (BDDs) to represent sets of states, is a competitive approach to optimal planning. Yet heuristic search in this context remains challenging. The many advances on admissible planning heuristics are not directly applicable, as they evaluate one state at a time. Indeed, progress using heuristic functions in symbolic search has been limited and even very informed heuristics have been shown to be detrimental. Here we show how this connection can be made stronger for LP-based potential heuristics. Our key observation is that, for this family of heuristic functions, the change of heuristic value induced by each operator can be precomputed. This facilitates their smooth integration into symbolic search. Our experiments show that this can pay off significantly: we establish a new state of the art in optimal symbolic planning.

----

## [1095] Reconfiguring Shortest Paths in Graphs

**Authors**: *Kshitij Gajjar, Agastya Vibhuti Jha, Manish Kumar, Abhiruk Lahiri*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i9.21211](https://doi.org/10.1609/aaai.v36i9.21211)

**Abstract**:

Reconfiguring two shortest paths in a graph means modifying one shortest path to the other by changing one vertex at a time, so that all the intermediate paths are also shortest paths. This problem has several natural applications, namely: (a) revamping road networks, (b) rerouting data packets in a synchronous multiprocessing setting, (c) the shipping container stowage problem, and (d) the train marshalling problem.
 
 When modelled as graph problems, (a) is the most general case while (b), (c) and (d) are restrictions to different graph classes. We show that (a) is intractable, even for relaxed variants of the problem. For (b), (c) and (d), we present efficient algorithms to solve the respective problems. We also generalise the problem to when at most k (for some k >= 2) contiguous vertices on a shortest path can be changed at a time.

----

## [1096] Homomorphisms of Lifted Planning Tasks: The Case for Delete-Free Relaxation Heuristics

**Authors**: *Rostislav Horcík, Daniel Fiser, Álvaro Torralba*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i9.21212](https://doi.org/10.1609/aaai.v36i9.21212)

**Abstract**:

Classical planning tasks are modelled in PDDL which is a schematic language based on first-order logic. Most of the current planners turn this lifted representation into a propositional one via a grounding process. However, grounding may cause an exponential blowup. Therefore it is important to investigate methods for searching for plans on the lifted level. To build a lifted state-based planner, it is necessary to invent lifted heuristics. We introduce maps between PDDL tasks preserving plans allowing to transform a PDDL task into a smaller one. We propose a novel method for computing lifted (admissible) delete-free relaxed heuristics via grounding of the smaller task and computing the (admissible) delete-free relaxed heuristics there. This allows us to transfer the knowledge about relaxed heuristics from the grounded level to the lifted level.

----

## [1097] Speeding Up the RUL¯ Dynamic-Controllability-Checking Algorithm for Simple Temporal Networks with Uncertainty

**Authors**: *Luke Hunsberger, Roberto Posenato*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i9.21213](https://doi.org/10.1609/aaai.v36i9.21213)

**Abstract**:

A Simple Temporal Network with Uncertainty (STNU) includes real-valued variables, called time-points; binary difference constraints on those time-points; and contingent links that represent actions with uncertain durations. STNUs have been used for robot control, web-service composition, and business processes. The most important property of an STNU is called dynamic controllability (DC); and algorithms for checking this property are called DC-checking algorithms. The DC-checking algorithm for STNUs with the best worst-case time-complexity is the RUL¯ algorithm due to Cairo, Hunsberger and Rizzi. Its complexity is O(mn + k²n + kn log n), where n is the number of time-points, m is the number of constraints, and k is the number of contingent links. It is expected that this worst-case complexity cannot be improved upon. However, this paper provides a new algorithm, called RUL2021, that improves its performance in practice by an order of magnitude, as demonstrated by a thorough empirical evaluation.

----

## [1098] Learning to Solve Routing Problems via Distributionally Robust Optimization

**Authors**: *Yuan Jiang, Yaoxin Wu, Zhiguang Cao, Jie Zhang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i9.21214](https://doi.org/10.1609/aaai.v36i9.21214)

**Abstract**:

Recent deep models for solving routing problems always assume a single distribution of nodes for training, which severely impairs their cross-distribution generalization ability. In this paper, we exploit group distributionally robust optimization (group DRO) to tackle this issue, where we jointly optimize the weights for different groups of distributions and the parameters for the deep model in an interleaved manner during training. We also design a module based on convolutional neural network, which allows the deep model to learn more informative latent pattern among the nodes. We evaluate the proposed approach on two types of well-known deep models including GCN and POMO. The experimental results on the randomly synthesized instances and the ones from two benchmark dataset (i.e., TSPLib and CVRPLib) demonstrate that our approach could significantly improve the cross-distribution generalization performance over the original models.

----

## [1099] Learning Probably Approximately Complete and Safe Action Models for Stochastic Worlds

**Authors**: *Brendan Juba, Roni Stern*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i9.21215](https://doi.org/10.1609/aaai.v36i9.21215)

**Abstract**:

We consider the problem of learning action models for planning in unknown stochastic environments that can be defined using the Probabilistic Planning Domain Description Language (PPDDL). As input, we are given a set of previously executed trajectories, and the main challenge is to learn an action model that has a similar goal achievement probability to the policies used to create these trajectories. To this end, we introduce a variant of PPDDL in which there is uncertainty about the transition probabilities, specified by an interval for each factor that contains the respective true transition probabilities. Then, we present SAM+, an algorithm that learns such an imprecise-PPDDL environment model. SAM+ has a polynomial time and sample complexity, and guarantees that with high probability, the true environment is indeed captured by the defined intervals. We prove that the action model SAM+ outputs has a goal achievement probability that is almost as good or better than that of the policies used to produced the training trajectories. Then, we show how to produce a PPDDL model based on this imprecise-PPDDL model that has similar properties.

----

## [1100] Bounding Quality in Diverse Planning

**Authors**: *Michael Katz, Shirin Sohrabi, Octavian Udrea*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i9.21216](https://doi.org/10.1609/aaai.v36i9.21216)

**Abstract**:

Diverse planning is an important problem in automated planning with many real world applications. Recently, diverse planning has seen renewed interest, with work that defines a taxonomy of computational problems with respect to both plan quality and solution diversity. However, despite the recent advances in diverse planning, the variety of approaches and the number of available planners are still quite limited, even nonexistent for several computational problems. In this work, we aim to extend the portfolio of planners for various computational problems in diverse planning. To that end, we introduce a novel approach to finding solutions for three computational problems within diverse planning and present planners for these three problems. For one of these problems, our approach is the first one that is able to provide solutions to the problem. For another, we show that top-k and top quality planners can provide, albeit naive, solutions to the problem and we extend these planners to improve the diversity of the solution. Finally, for the third problem, we show that some existing diverse planners already provide solutions to that problem. We suggest another approach and empirically show it to compare favorably with these existing planners.

----

## [1101] A* Search and Bound-Sensitive Heuristics for Oversubscription Planning

**Authors**: *Michael Katz, Emil Keyder*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i9.21217](https://doi.org/10.1609/aaai.v36i9.21217)

**Abstract**:

Oversubscription planning (OSP) is the problem of finding plans that maximize the utility value of their end state while staying within a specified cost bound. Recently, it has been shown that OSP problems can be reformulated as classical planning problems with multiple cost functions but no utilities. Here we take advantage of this reformulation to show that OSP problems can be solved optimally using the A* search algorithm, in contrast to previous approaches that have used variations on branch-and-bound search. This allows many powerful techniques developed for classical planning to be applied to OSP problems. We also introduce novel bound-sensitive heuristics, which are able to reason about the primary cost of a solution while taking into account secondary cost functions and bounds, to provide superior guidance compared to heuristics that do not take these bounds into account. We propose two such bound-sensitive variants of existing classical planning heuristics, and show experimentally that the resulting search is significantly more informed than with comparable heuristics that do not consider bounds.

----

## [1102] NICE: Robust Scheduling through Reinforcement Learning-Guided Integer Programming

**Authors**: *Luke Kenworthy, Siddharth Nayak, Christopher Chin, Hamsa Balakrishnan*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i9.21218](https://doi.org/10.1609/aaai.v36i9.21218)

**Abstract**:

Integer programs provide a powerful abstraction for representing a wide range of real-world scheduling problems. Despite their ability to model general scheduling problems, solving large-scale integer programs (IP) remains a computational challenge in practice. The incorporation of more complex objectives such as robustness to disruptions further exacerbates the computational challenge. We present NICE (Neural network IP Coefficient Extraction), a novel technique that combines reinforcement learning and integer programming to tackle the problem of robust scheduling. More specifically, NICE uses reinforcement learning to approximately represent complex objectives in an integer programming formulation. We use NICE to determine assignments of pilots to a flight crew schedule so as to reduce the impact of disruptions. We compare NICE with (1) a baseline integer programming formulation that produces a feasible crew schedule, and (2) a robust integer programming formulation that explicitly tries to minimize the impact of disruptions. Our experiments show that, across a variety of scenarios, NICE produces schedules resulting in 33% to 48% fewer disruptions than the baseline formulation. Moreover, in more severely constrained scheduling scenarios in which the robust integer program fails to produce a schedule within 90 minutes, NICE is able to build robust schedules in less than 2 seconds on average.

----

## [1103] Planning to Avoid Side Effects

**Authors**: *Toryn Q. Klassen, Sheila A. McIlraith, Christian Muise, Jarvis Xu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i9.21219](https://doi.org/10.1609/aaai.v36i9.21219)

**Abstract**:

In sequential decision making, objective specifications are often underspecified or incomplete, neglecting to take into account potential (negative) side effects. Executing plans without consideration of their side effects can lead to catastrophic outcomes -- a concern recently raised in relation to the safety of AI. In this paper we investigate how to avoid side effects in a symbolic planning setting. We study the notion of minimizing side effects in the context of a planning environment where multiple independent agents co-exist. We define (classes of) negative side effects in terms of their effect on the agency of those other agents. Finally, we show how plans which minimize side effects of different types can be computed via compilations to cost-optimizing symbolic planning, and investigate experimentally.

----

## [1104] Sample-Efficient Iterative Lower Bound Optimization of Deep Reactive Policies for Planning in Continuous MDPs

**Authors**: *Siow Meng Low, Akshat Kumar, Scott Sanner*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i9.21220](https://doi.org/10.1609/aaai.v36i9.21220)

**Abstract**:

Recent advances in deep learning have enabled optimization of deep reactive policies (DRPs) for continuous MDP planning by encoding a parametric policy as a deep neural network and exploiting automatic differentiation in an end-to-end model-based gradient descent framework. This approach has proven effective for optimizing DRPs in nonlinear continuous MDPs, but it requires a large number of sampled trajectories to learn effectively and can suffer from high variance in solution quality. In this work, we revisit the overall model-based DRP objective and instead take a minorization-maximization perspective to iteratively optimize the DRP w.r.t. a locally tight lower-bounded objective. This novel formulation of DRP learning as iterative lower bound optimization (ILBO) is particularly appealing because (i) each step is structurally easier to optimize than the overall objective, (ii) it guarantees a monotonically improving objective under certain theoretical conditions, and (iii) it reuses samples between iterations thus lowering sample complexity. Empirical evaluation confirms that ILBO is significantly more sample-efficient than the state-of-the-art DRP planner and consistently produces better solution quality with lower variance. We additionally demonstrate that ILBO generalizes well to new problem instances (i.e., different initial states) without requiring retraining.

----

## [1105] Bridging LTLf Inference to GNN Inference for Learning LTLf Formulae

**Authors**: *Weilin Luo, Pingjia Liang, Jianfeng Du, Hai Wan, Bo Peng, Delong Zhang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i9.21221](https://doi.org/10.1609/aaai.v36i9.21221)

**Abstract**:

Learning linear temporal logic on finite traces (LTLf) formulae aims to learn a target formula that characterizes the high-level behavior of a system from observation traces in planning. Existing approaches to learning LTLf formulae, however, can hardly learn accurate LTLf formulae from noisy data. It is challenging to design an efficient search mechanism in the large search space in form of arbitrary LTLf formulae while alleviating the wrong search bias resulting from noisy data. In this paper, we tackle this problem by bridging LTLf inference to GNN inference. Our key theoretical contribution is showing that GNN inference can simulate LTLf inference to distinguish traces. Based on our theoretical result, we design a GNN-based approach, GLTLf, which combines GNN inference and parameter interpretation to seek the target formula in the large search space. Thanks to the non-deterministic learning process of GNNs, GLTLf is able to cope with noise. We evaluate GLTLf on various datasets with noise. Our experimental results confirm the effectiveness of GNN inference in learning LTLf formulae and show that GLTLf is superior to the state-of-the-art approaches.

----

## [1106] Risk-Aware Stochastic Shortest Path

**Authors**: *Tobias Meggendorfer*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i9.21222](https://doi.org/10.1609/aaai.v36i9.21222)

**Abstract**:

We treat the problem of risk-aware control for stochastic shortest path (SSP) on Markov decision processes (MDP). Typically, expectation is considered for SSP, which however is oblivious to the incurred risk. We present an alternative view, instead optimizing conditional value-at-risk (CVaR), an established risk measure. We treat both Markov chains as well as MDP and introduce, through novel insights, two algorithms, based on linear programming and value iteration, respectively. Both algorithms offer precise and provably correct solutions. Evaluation of our prototype implementation shows that risk-aware control is feasible on several moderately sized models.

----

## [1107] Differential Assessment of Black-Box AI Agents

**Authors**: *Rashmeet Kaur Nayyar, Pulkit Verma, Siddharth Srivastava*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i9.21223](https://doi.org/10.1609/aaai.v36i9.21223)

**Abstract**:

Much of the research on learning symbolic models of AI agents focuses on agents with stationary models. This assumption fails to hold in settings where the agent's capabilities may change as a result of learning, adaptation, or other post-deployment modifications. Efficient assessment of agents in such settings is critical for learning the true capabilities of an AI system and for ensuring its safe usage. In this work, we propose a novel approach to differentially assess black-box AI agents that have drifted from their previously known models. As a starting point, we consider the fully observable and deterministic setting. We leverage sparse observations of the drifted agent's current behavior and knowledge of its initial model to generate an active querying policy that selectively queries the agent and computes an updated model of its functionality. Empirical evaluation shows that our approach is much more efficient than re-learning the agent model from scratch. We also show that the cost of differential assessment using our method is proportional to the amount of drift in the agent's functionality.

----

## [1108] Solving Disjunctive Temporal Networks with Uncertainty under Restricted Time-Based Controllability Using Tree Search and Graph Neural Networks

**Authors**: *Kevin Osanlou, Jeremy Frank, Andrei Bursuc, Tristan Cazenave, Eric Jacopin, Christophe Guettier, J. Benton*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i9.21224](https://doi.org/10.1609/aaai.v36i9.21224)

**Abstract**:

Scheduling under uncertainty is an area of interest in artificial intelligence. We study the problem of Dynamic Controllability (DC) of Disjunctive Temporal Networks with Uncertainty (DTNU), which seeks a reactive scheduling strategy to satisfy temporal constraints in response to uncontrollable action durations. We introduce new semantics for reactive scheduling: Time-based Dynamic Controllability (TDC) and a restricted subset of TDC, R-TDC. We present a tree search approach to determine whether or not a DTNU is R-TDC. Moreover, we leverage the learning capability of a Graph Neural Network (GNN) as a heuristic for tree search guidance. Finally, we conduct experiments on a known benchmark on which we show R-TDC to retain significant completeness with regard to DC, while being faster to prove. This results in the tree search processing fifty percent more DTNU problems in R-TDC than the state-of-the-art DC solver does in DC with the same time budget. We also observe that GNN tree search guidance leads to substantial performance gains on benchmarks of more complex DTNUs, with up to eleven times more problems solved than the baseline tree search.

----

## [1109] Deciding Unsolvability in Temporal Planning under Action Non-Self-Overlapping

**Authors**: *Stefan Panjkovic, Andrea Micheli, Alessandro Cimatti*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i9.21225](https://doi.org/10.1609/aaai.v36i9.21225)

**Abstract**:

The field of Temporal Planning (TP) is receiving increasing interest for its many real-world applications. Most of the literature focuses on the TP problem of finding a plan, with algorithms that are not guaranteed to terminate when the problem admits no solution. In this paper, we present sound and complete decision procedures that address the dual problem of proving that no plan exists, which has important applications in oversubscription, model validation and optimization. We focus on the expressive and practically relevant semantics of action non-self-overlapping, recently proved to be PSPACE-complete. For this subclass, we propose two approaches: a reduction of the planning problem to model-checking of Timed Transition Systems, and a heuristic-search algorithm where temporal constraints are represented by Difference Bound Matrices. We implemented the approaches, and carried out an experimental evaluation against other state-of-the-art TP tools. On benchmarks that admit no plans, both approaches dramatically outperform the other planners, while the heuristic-search algorithm remains competitive on solvable benchmarks.

----

## [1110] A Distributional Framework for Risk-Sensitive End-to-End Planning in Continuous MDPs

**Authors**: *Noah Patton, Jihwan Jeong, Mike Gimelfarb, Scott Sanner*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i9.21226](https://doi.org/10.1609/aaai.v36i9.21226)

**Abstract**:

Recent advances in efficient planning in deterministic or stochastic high-dimensional domains with continuous action spaces leverage backpropagation through a model of the environment to directly optimize action sequences. However, existing methods typically do not take risk into account when optimizing in stochastic domains, which can be incorporated efficiently in MDPs by optimizing a nonlinear utility function of the return distribution. We bridge this gap by introducing Risk-Aware Planning using PyTorch (RAPTOR), a novel unified framework for risk-sensitive planning through end-to-end optimization of commonly-studied risk-sensitive utility functions such as entropic utility, mean-variance optimization and CVaR. A key technical difficulty of our approach is that direct optimization of general risk-sensitive utility functions by backpropagation is impossible due to the presence of environment stochasticity. The novelty of RAPTOR lies in leveraging reparameterization of the state distribution, leading to a unique distributional perspective of end-to-end planning where the return distribution is utilized for sampling as well as optimizing risk-aware objectives by backpropagation in a unified framework. We evaluate and compare RAPTOR on three highly stochastic MDPs, including nonlinear navigation, HVAC control, and linear reservoir control, demonstrating the ability of RAPTOR to manage risk in complex continuous domains according to different notions of risk-sensitive utility.

----

## [1111] Formula Synthesis in Propositional Dynamic Logic with Shuffle

**Authors**: *Sophie Pinchinat, Sasha Rubin, François Schwarzentruber*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i9.21227](https://doi.org/10.1609/aaai.v36i9.21227)

**Abstract**:

We introduce the formula-synthesis problem for Propositional Dynamic Logic with Shuffle (PDL || ). This problem, which generalises the model-checking problem againsts PDL || is the following: given a finite transition system  and a regular term-grammar that generates (possibly infinitely many) PDL || formulas, find a formula generated by the grammar that is true in the structure (or return that there is none). We prove that the problem is undecidable in general, but add certain restrictions on the input structure or on the input grammar to yield decidability. In particular, we prove that (1) if the grammar only generates formulas in PDL (without shuffle), then the problem is EXPTIME-complete, and a further restriction to linear grammars is PSPACE-complete, and a further restriction to non-recursive grammars is NP-complete,  and (2) if one restricts the input structure to have only simple paths then the problem is in 2-EXPTIME. This work is motivated by and opens up connections to other forms of synthesis from hierarchical descriptions, including HTN problems in Planning and Attack-tree Synthesis problems in Security.

----

## [1112] Efficient Encoding of Cost Optimal Delete-Free Planning as SAT

**Authors**: *Masood Feyzbakhsh Rankooh, Jussi Rintanen*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i9.21228](https://doi.org/10.1609/aaai.v36i9.21228)

**Abstract**:

We introduce a novel method for encoding cost optimal delete-free STRIPS Planning as SAT. Our method is based on representing relaxed plans as partial functions from the set of propositions to the set of actions. This function can map any proposition to a unique action that adds the proposition during execution of the relaxed plan. We show that a relaxed plan can be produced by maintaining acyclicity in the graph of all causal relations among propositions, represented by the mentioned partial function. We also show that by efficient encoding of action cost propagation and enforcing a series of upper bounds on the total costs of the output plan, an optimal plan can effectively be produced for a given delete-free STRIPS problem. Our empirical results indicate that this method is quite competitive with the state of the art, demonstrating a better coverage compared to that of competing methods on standard STRIPS planning benchmark problems.

----

## [1113] Optimal Admission Control for Multiclass Queues with Time-Varying Arrival Rates via State Abstraction

**Authors**: *Marc Rigter, Danial Dervovic, Parisa Hassanzadeh, Jason Long, Parisa Zehtabi, Daniele Magazzeni*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i9.21229](https://doi.org/10.1609/aaai.v36i9.21229)

**Abstract**:

We consider a novel queuing problem where the decision-maker must choose to accept or reject randomly arriving tasks into a no buffer queue which are processed by N identical servers. Each task has a price, which is a positive real number, and a class. Each class of task has a different price distribution, service rate, and arrives according to an inhomogenous Poisson process. The objective is to decide which tasks to accept so that the total price of tasks processed is maximised over a finite horizon. We formulate the problem using a discrete time Markov Decision Process (MDP) with a hybrid state space. We show that the optimal value function has a specific structure, which enables us to solve the hybrid MDP exactly. Moreover, we rigorously prove that as the gap between successive decision epochs grows smaller, the discrete time solution approaches the optimal solution to the original continuous time problem. To improve the scalability of our approach to a greater number of servers and task classes, we present an approximation based on state abstraction. We validate our approach on synthetic data, as well as a real financial fraud data set, which is the motivating application for this work.

----

## [1114] Enhancing Column Generation by a Machine-Learning-Based Pricing Heuristic for Graph Coloring

**Authors**: *Yunzhuang Shen, Yuan Sun, Xiaodong Li, Andrew C. Eberhard, Andreas T. Ernst*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i9.21230](https://doi.org/10.1609/aaai.v36i9.21230)

**Abstract**:

Column Generation (CG) is an effective method for solving large-scale optimization problems. CG starts by solving a subproblem with a subset of columns (i.e., variables) and gradually includes new columns that can improve the solution of the current subproblem. The new columns are generated as needed by repeatedly solving a pricing problem, which is often NP-hard and is a bottleneck of the CG approach. To tackle this, we propose a Machine-Learning-based Pricing Heuristic (MLPH) that can generate many high-quality columns efficiently. In each iteration of CG, our MLPH leverages an ML model to predict the optimal solution of the pricing problem, which is then used to guide a sampling method to efficiently generate multiple high-quality columns. Using the graph coloring problem, we empirically show that MLPH significantly enhances CG as compared to six state-of-the-art methods, and the improvement in CG can lead to substantially better performance of the branch-and-price exact method.

----

## [1115] Qubit Routing Using Graph Neural Network Aided Monte Carlo Tree Search

**Authors**: *Animesh Sinha, Utkarsh Azad, Harjinder Singh*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i9.21231](https://doi.org/10.1609/aaai.v36i9.21231)

**Abstract**:

Near-term quantum hardware can support two-qubit operations only on the qubits that can interact with each other. Therefore, to execute an arbitrary quantum circuit on the hardware, compilers have to first perform the task of qubit routing, i.e., to transform the quantum circuit either by inserting additional SWAP gates or by reversing existing CNOT gates to satisfy the connectivity constraints of the target topology. The depth of the transformed quantum circuits is minimized by utilizing the Monte Carlo tree search (MCTS) to perform qubit routing by making it both construct each action and search over the space of all actions. It is aided in performing these tasks by a Graph neural network that evaluates the value function and action probabilities for each state. Along with this, we propose a new method of adding mutex-lock like variables in our state representation which helps factor in the parallelization of the scheduled operations, thereby pruning the depth of the output circuit. Overall, our procedure (referred to as QRoute) performs qubit routing in a hardware agnostic manner, and it outperforms other available qubit routing implementations on various circuit benchmarks.

----

## [1116] Classical Planning with Avoid Conditions

**Authors**: *Marcel Steinmetz, Jörg Hoffmann, Alisa Kovtunova, Stefan Borgwardt*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i9.21232](https://doi.org/10.1609/aaai.v36i9.21232)

**Abstract**:

It is often natural in planning to specify conditions that should be avoided, characterizing dangerous or highly undesirable behavior. PDDL3 supports this with temporal-logic state trajectory constraints. Here we focus on the simpler case where the constraint is a non-temporal formula ? - the avoid condition - that must be false throughout the plan. We design techniques tackling such avoid conditions effectively. We show how to learn from search experience which states necessarily lead into ?, and we show how to tailor abstractions to recognize that avoiding ? will not be possible starting from a given state. We run a large-scale experiment, comparing our techniques against compilation methods and against simple state pruning using ?. The results show that our techniques are often superior.

----

## [1117] Stochastic Goal Recognition Design Problems with Suboptimal Agents

**Authors**: *Christabel Wayllace, William Yeoh*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i9.21233](https://doi.org/10.1609/aaai.v36i9.21233)

**Abstract**:

Goal Recognition Design (GRD) problems identify the minimum number of environmental modifications aiming to force an interacting agent to reveal its goal as early as possible. Researchers proposed several extensions to the original model, some of them handling stochastic agent action outcomes. While this generalization is useful, it assumes optimal acting agents, which limits its applicability to more realistic scenarios. This paper presents the Suboptimal Stochastic GRD model, where we consider boundedly rational agents that, due to limited resources, might follow a suboptimal policy. Inspired by theories on human behavior asserting that humans are (close to) optimal when making perceptual decisions, we assume the chosen policy has at most m suboptimal actions. Our contribution includes (I) Extending the stochastic goal recognition design framework by supporting suboptimal agents in cases where an observer has either full or partial observability; (ii) Presenting methods to evaluate the ambiguity of the model under these assumptions; and (iii) Evaluating our approach on a range of benchmark applications.

----

## [1118] Equity Promotion in Online Resource Allocation

**Authors**: *Pan Xu, Yifan Xu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i9.21234](https://doi.org/10.1609/aaai.v36i9.21234)

**Abstract**:

We consider online resource allocation under a typical non-profit setting, where limited or even scarce resources are administered by a not-for-profit organization like a government. We focus on the internal-equity by assuming that arriving requesters are homogeneous in terms of their external factors like demands but heterogeneous for their internal attributes like demographics. Specifically, we associate each arriving requester with one or several groups based on their demographics (i.e., race, gender, and age), and we aim to design an equitable distributing strategy such that every group of requesters can receive a fair share of resources proportional to a preset target ratio. 
 We present two LP-based sampling algorithms and investigate them both theoretically (in terms of competitive-ratio analysis) and experimentally based on real COVID-19 vaccination data maintained by the Minnesota Department of Health. Both theoretical and numerical results show that our LP-based sampling strategies can effectively promote equity, especially when the arrival population is disproportionately represented, as observed in the early stage of the COVID-19 vaccine rollout.

----

## [1119] Efficient Device Scheduling with Multi-Job Federated Learning

**Authors**: *Chendi Zhou, Ji Liu, Juncheng Jia, Jingbo Zhou, Yang Zhou, Huaiyu Dai, Dejing Dou*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i9.21235](https://doi.org/10.1609/aaai.v36i9.21235)

**Abstract**:

Recent years have witnessed a large amount of decentralized data in multiple (edge) devices of end-users, while the aggregation of the decentralized data remains difficult for machine learning jobs due to laws or regulations. Federated Learning (FL) emerges as an effective approach to handling decentralized data without sharing the sensitive raw data, while collaboratively training global machine learning models. The servers in FL need to select (and schedule) devices during the training process. However, the scheduling of devices for multiple jobs with FL remains a critical and open problem. In this paper, we propose a novel multi-job FL framework to enable the parallel training process of multiple jobs. The framework consists of a system model and two scheduling methods. In the system model, we propose a parallel training process of multiple jobs, and construct a cost model based on the training time and the data fairness of various devices during the training process of diverse jobs. We propose a reinforcement learning-based method and a Bayesian optimization-based method to schedule devices for multiple jobs while minimizing the cost. We conduct extensive experimentation with multiple jobs and datasets. The experimental results show that our proposed approaches significantly outperform baseline approaches in terms of training time (up to 8.67 times faster) and accuracy (up to 44.6% higher).

----

## [1120] MAPDP: Cooperative Multi-Agent Reinforcement Learning to Solve Pickup and Delivery Problems

**Authors**: *Zefang Zong, Meng Zheng, Yong Li, Depeng Jin*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i9.21236](https://doi.org/10.1609/aaai.v36i9.21236)

**Abstract**:

Cooperative Pickup and Delivery Problem (PDP), as a variant of the typical Vehicle Routing Problems (VRP), is an important formulation in many real-world applications, such as on-demand delivery, industrial warehousing, etc. It is of great importance to efficiently provide high-quality solutions of cooperative PDP. However, it is not trivial to provide effective solutions directly due to two major challenges: 1) the structural dependency between pickup and delivery pairs require explicit modeling and representation. 2) the cooperation between different vehicles is highly related to the solution exploration and difficult to model. In this paper, we propose a novel multi-agent reinforcement learning based framework to solve the cooperative PDP (MAPDP). First, we design a paired context embedding to well measure the dependency of different nodes considering their structural limits. Second, we utilize cooperative multi-agent decoders to leverage the decision dependence among different vehicle agents based on a special communication embedding. Third, we design a novel cooperative A2C algorithm to train the integrated model. We conduct extensive experiments on a randomly generated dataset and a real-world dataset. Experiments result shown that the proposed MAPDP outperform all other baselines by at least 1.64\% in all settings, and shows significant computation speed during solution inference.

----

## [1121] Entropy Estimation via Normalizing Flow

**Authors**: *Ziqiao Ao, Jinglai Li*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i9.21237](https://doi.org/10.1609/aaai.v36i9.21237)

**Abstract**:

Entropy estimation is an important problem in information theory and statistical science. Many popular entropy estimators suffer from fast growing estimation bias with respect to dimensionality, rendering them unsuitable for high dimensional problems. In this work we propose a transformbased method for high dimensional entropy estimation, which consists of the following two main ingredients. First by modifying the k-NN based entropy estimator, we propose a new estimator which enjoys small estimation bias for samples that are close to a uniform distribution. Second we design a normalizing flow based mapping that pushes samples toward a uniform distribution, and the relation between the entropy of the original samples and the transformed ones is also derived. As a result the entropy of a given set of samples is estimated by first transforming them toward a uniform distribution and then applying the proposed estimator to the transformed samples. Numerical experiments demonstrate the effectiveness of the method for high dimensional entropy estimation problems.

----

## [1122] Fast and More Powerful Selective Inference for Sparse High-Order Interaction Model

**Authors**: *Diptesh Das, Vo Nguyen Le Duy, Hiroyuki Hanada, Koji Tsuda, Ichiro Takeuchi*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i9.21238](https://doi.org/10.1609/aaai.v36i9.21238)

**Abstract**:

Automated high-stake decision-making, such as medical diagnosis, requires models with high interpretability and reliability. We consider the sparse high-order interaction model as an interpretable and reliable model with a good prediction ability. However, finding statistically significant high-order interactions is challenging because of the intrinsically high dimensionality of the combinatorial effects. Another problem in data-driven modeling is the effect of ``cherry-picking" (i.e., selection bias). Our main contribution is extending the recently developed parametric programming approach for selective inference to high-order interaction models. An exhaustive search over the cherry tree (all possible interactions) can be daunting and impractical, even for small-sized problems. We introduced an efficient pruning strategy and demonstrated the computational efficiency and statistical power of the proposed method using both synthetic and real data.

----

## [1123] Generalized Stochastic Matching

**Authors**: *Alireza Farhadi, Jacob Gilbert, MohammadTaghi Hajiaghayi*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i9.21239](https://doi.org/10.1609/aaai.v36i9.21239)

**Abstract**:

In this paper, we generalize the recently studied stochastic matching problem to more accurately model a significant medical process, kidney exchange, and several other applications. Up until now the stochastic matching problem that has been studied was as follows: given a graph G= (V,E), each edge is included in the realized sub-graph of G independently with probability pe, and the goal is to find a degree-bounded sub-graph Q of G that has an expected maximum matching that approximates the expected maximum matching of G. This model does not account for possibilities of vertex dropouts, which can be found in several applications, e.g. in kidney exchange when donors or patients opt out of the exchange process as well as in online freelancing and online dating when online profiles are found to be faked. Thus, we will study a more generalized model of stochastic matching in which vertices and edges are both realized independently with some probabilities pv, pe, respectively, which more accurately fits important applications than the previously studied model. 
 
We will discuss the first algorithms and analysis for this generalization of the stochastic matching model and prove that they achieve good approximation ratios. In particular, we show that the approximation factor of a natural algorithm for this problem is at least 0.6568 in unweighted graphs, and 1/2+ε in weighted graphs for some constant ε >0. We further improve our result for unweighted graphs to 2/3 using edge degree constrained sub-graphs (EDCS).

----

## [1124] Robust Tests in Online Decision-Making

**Authors**: *Gi-Soo Kim, Jane P. Kim, Hyun-Joon Yang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i9.21240](https://doi.org/10.1609/aaai.v36i9.21240)

**Abstract**:

Bandit algorithms are widely used in sequential decision problems to maximize the cumulative reward. One potential application is mobile health, where the goal is to promote the user's health through personalized interventions based on user specific information acquired through wearable devices. Important considerations include the type of, and frequency with which data is collected (e.g. GPS, or continuous monitoring), as such factors can severely impact app performance and users’ adherence. In order to balance the need to collect data that is useful with the constraint of impacting app performance, one needs to be able to assess the usefulness of variables. Bandit feedback data are sequentially correlated, so traditional testing procedures developed for independent data cannot apply. Recently, a statistical testing procedure was developed for the actor-critic bandit algorithm. An actor-critic algorithm maintains two separate models, one for the actor, the action selection policy, and the other for the critic, the reward model. The performance of the algorithm as well as the validity of the test are guaranteed only when the critic model is correctly specified. However, misspecification is frequent in practice due to incorrect functional form or missing covariates. In this work, we propose a modified actor-critic algorithm which is robust to critic misspecification and derive a novel testing procedure for the actor parameters in this case.

----

## [1125] Local Differential Privacy for Belief Functions

**Authors**: *Qiyu Li, Chunlai Zhou, Biao Qin, Zhiqiang Xu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i9.21241](https://doi.org/10.1609/aaai.v36i9.21241)

**Abstract**:

In this paper, we propose two new definitions of local differential privacy for belief functions. One is based on Shafer’s semantics of randomly coded messages and the other from the perspective of imprecise probabilities. We show that such basic properties as composition and post-processing also hold for our new definitions. Moreover, we provide a hypothesis testing framework for these definitions and study the effect of "don’t know" in the trade-off between privacy and utility in discrete distribution estimation.

----

## [1126] A Complete Criterion for Value of Information in Soluble Influence Diagrams

**Authors**: *Chris van Merwijk, Ryan Carey, Tom Everitt*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i9.21242](https://doi.org/10.1609/aaai.v36i9.21242)

**Abstract**:

Influence diagrams have recently been used to analyse the safety and fairness properties of AI systems. A key building block for this analysis is a graphical criterion for value of information (VoI). This paper establishes the first complete graphical criterion for VoI in influence diagrams with multiple decisions. Along the way, we establish two techniques for proving properties of multi-decision influence diagrams: ID homomorphisms are structure-preserving transformations of influence diagrams, while a Tree of Systems is a collection of paths that captures how information and control can flow in an influence diagram.

----

## [1127] Training-Free Uncertainty Estimation for Dense Regression: Sensitivity as a Surrogate

**Authors**: *Lu Mi, Hao Wang, Yonglong Tian, Hao He, Nir Shavit*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i9.21243](https://doi.org/10.1609/aaai.v36i9.21243)

**Abstract**:

Uncertainty estimation is an essential step in the evaluation of the robustness for deep learning models in computer vision, especially when applied in risk-sensitive areas. However, most state-of-the-art deep learning models either fail to obtain uncertainty estimation or need significant modification (e.g., formulating a proper Bayesian treatment) to obtain it. Most previous methods are not able to take an arbitrary model off the shelf and generate uncertainty estimation without retraining or redesigning it. To address this gap, we perform a systematic exploration into training-free uncertainty estimation for dense regression, an unrecognized yet important problem, and provide a theoretical construction justifying such estimations. We propose three simple and scalable methods to analyze the variance of outputs from a trained network under tolerable perturbations: infer-transformation, infer-noise, and infer-dropout. They operate solely during the inference, without the need to re-train, re-design, or fine-tune the models, as typically required by state-of-the-art uncertainty estimation methods. Surprisingly, even without involving such perturbations in training, our methods produce comparable or even better uncertainty estimation when compared to training-required state-of-the-art methods. Code is available at https://github.com/lumi9587/train-free-uncertainty.

----

## [1128] On the Impact of Spurious Correlation for Out-of-Distribution Detection

**Authors**: *Yifei Ming, Hang Yin, Yixuan Li*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i9.21244](https://doi.org/10.1609/aaai.v36i9.21244)

**Abstract**:

Modern neural networks can assign high confidence to inputs drawn from outside the training distribution, posing threats to models in real-world deployments. While much research attention has been placed on designing new out-of-distribution (OOD) detection methods, the precise definition of OOD is often left in vagueness and falls short of the desired notion of OOD in reality. In this paper, we present a new formalization and model the data shifts by taking into account both the invariant and environmental (spurious) features. Under such formalization, we systematically investigate how spurious correlation in the training set impacts OOD detection. Our results suggest that the detection performance is severely worsened when the correlation between spurious features and labels is increased in the training set. We further show insights on detection methods that are more effective in reducing the impact of spurious correlation, and provide theoretical analysis on why reliance on environmental features leads to high OOD detection error. Our work aims to facilitate better understanding of OOD samples and their formalization, as well as the exploration of methods that enhance OOD detection. Code is available at https://github.com/deeplearning-wisc/Spurious_OOD.

----

## [1129] Inference and Learning with Model Uncertainty in Probabilistic Logic Programs

**Authors**: *Victor Verreet, Vincent Derkinderen, Pedro Zuidberg Dos Martires, Luc De Raedt*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i9.21245](https://doi.org/10.1609/aaai.v36i9.21245)

**Abstract**:

An issue that has so far received only limited attention in probabilistic logic programming (PLP) is the modelling of so-called epistemic uncertainty, the uncertainty about the model itself. Accurately quantifying this model uncertainty is paramount to robust inference, learning and ultimately decision making. We introduce BetaProbLog, a PLP language that can model epistemic uncertainty. BetaProbLog has sound semantics, an effective inference algorithm that combines Monte Carlo techniques with knowledge compilation, and a parameter learning algorithm. We empirically outperform state-of-the-art methods on probabilistic inference tasks in second-order Bayesian networks, digit classification and discriminative learning in the presence of epistemic uncertainty.

----

## [1130] Domain-Lifted Sampling for Universal Two-Variable Logic and Extensions

**Authors**: *Yuanhong Wang, Timothy van Bremen, Yuyi Wang, Ondrej Kuzelka*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i9.21246](https://doi.org/10.1609/aaai.v36i9.21246)

**Abstract**:

Given a first-order sentence ? and a domain size n, how can one sample a model of ? on the domain {1, . . . , n} efficiently as n scales? We consider two variants of this problem: the uniform sampling regime, in which the goal is to sample a model uniformly at random, and the symmetric weighted sampling regime, in which models are weighted according to the number of groundings of each predicate appearing in them. Solutions to this problem have applications to the scalable generation of combinatorial structures, as well as sampling in several statistical-relational models such as Markov logic networks and probabilistic logic programs. In this paper, we identify certain classes of sentences that are domain-liftable under sampling, in the sense that they admit a sampling algorithm that runs in time polynomial in n. In particular, we prove that every sentence of the form ∀x∀y: ?(x, y) for some quantifier-free formula ?(x,y) is domain-liftable under sampling. We then further show that this result continues to hold in the presence of one or more cardinality constraints as well as a single tree axiom constraint.

----

## [1131] Identifiability of Linear AMP Chain Graph Models

**Authors**: *Yuhao Wang, Arnab Bhattacharyya*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i9.21247](https://doi.org/10.1609/aaai.v36i9.21247)

**Abstract**:

We study identifiability of linear Andersson-Madigan-Perlman (AMP) chain graph models, which are a common generalization of linear structural equation models and Gaussian graphical models. AMP models are described by DAGs on chain components which themselves are undirected graphs.
For a known chain component decomposition, we show that the DAG on the chain components is identifiable if the determinants of the residual covariance matrices of the chain components are equal (or more generally, monotone non-decreasing in topological order). This condition extends the equal variance identifiability criterion for Bayes nets, and it can be generalized from determinants to any super-additive function on positive semidefinite matrices. When the component decomposition is unknown, we describe conditions that allow recovery of the full structure using a polynomial time algorithm based on submodular function minimization. We also conduct experiments comparing our algorithm's performance against existing baselines.

----

## [1132] DeepStochLog: Neural Stochastic Logic Programming

**Authors**: *Thomas Winters, Giuseppe Marra, Robin Manhaeve, Luc De Raedt*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i9.21248](https://doi.org/10.1609/aaai.v36i9.21248)

**Abstract**:

Recent advances in neural-symbolic learning, such as DeepProbLog, extend probabilistic logic programs with neural predicates. Like graphical models, these probabilistic logic programs define a probability distribution over possible worlds, for which inference is computationally hard. We propose DeepStochLog, an alternative neural-symbolic framework based on stochastic definite clause grammars, a kind of stochastic logic program. More specifically, we introduce neural grammar rules into stochastic definite clause grammars to create a framework that can be trained end-to-end. We show that inference and learning in neural stochastic logic programming scale much better than for neural probabilistic logic programs. Furthermore, the experimental evaluation shows that DeepStochLog achieves state-of-the-art results on challenging neural-symbolic learning tasks.

----

## [1133] Towards Robust Off-Policy Learning for Runtime Uncertainty

**Authors**: *Da Xu, Yuting Ye, Chuanwei Ruan, Bo Yang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i9.21249](https://doi.org/10.1609/aaai.v36i9.21249)

**Abstract**:

Off-policy learning plays a pivotal role in optimizing and evaluating policies prior to the online deployment. However, during the real-time serving, we observe varieties of interventions and constraints that cause inconsistency between the online and offline setting, which we summarize and term as runtime uncertainty. Such uncertainty cannot be learned from the logged data due to its abnormality and rareness nature. To assert a certain level of robustness, we perturb the off-policy estimators along an adversarial direction in view of the runtime uncertainty. It allows the resulting estimators to be robust not only to observed but also unexpected runtime uncertainties. Leveraging this idea, we bring runtime-uncertainty robustness to three major off-policy learning methods: the inverse propensity score method, reward-model method, and doubly robust method. We theoretically justify the robustness of our methods to runtime uncertainty, and demonstrate their effectiveness using both the simulation and the real-world online experiments.

----

## [1134] Improving Bayesian Neural Networks by Adversarial Sampling

**Authors**: *Jiaru Zhang, Yang Hua, Tao Song, Hao Wang, Zhengui Xue, Ruhui Ma, Haibing Guan*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i9.21250](https://doi.org/10.1609/aaai.v36i9.21250)

**Abstract**:

Bayesian neural networks (BNNs) have drawn extensive interest due to the unique probabilistic representation framework.
 However, Bayesian neural networks have limited publicized deployments because of the relatively poor model performance in real-world applications.
 In this paper, we argue that the randomness of sampling in Bayesian neural networks causes errors in the updating of model parameters during training and some sampled models with poor performance in testing.
 To solve this, we propose to train Bayesian neural networks with Adversarial Distribution as a theoretical solution. 
 To avoid the difficulty of calculating Adversarial Distribution analytically, we further present the Adversarial Sampling method as an approximation in practice. 
 We conduct extensive experiments with multiple network structures on different datasets, e.g., CIFAR-10 and CIFAR-100. 
 Experimental results validate the correctness of the theoretical analysis and the effectiveness of the Adversarial Sampling on improving model performance.
 Additionally, models trained with Adversarial Sampling still keep their ability to model uncertainties and perform better when predictions are retained according to the uncertainties, which further verifies the generality of the Adversarial Sampling approach.

----

## [1135] Efficient Optimal Transport Algorithm by Accelerated Gradient Descent

**Authors**: *Dongsheng An, Na Lei, Xiaoyin Xu, Xianfeng Gu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i9.21251](https://doi.org/10.1609/aaai.v36i9.21251)

**Abstract**:

Optimal transport (OT) plays an essential role in various areas like machine learning and deep learning.
 However, computing discrete optimal transport plan for large scale problems with adequate accuracy and efficiency is still highly challenging. 
 Recently, methods based on the Sinkhorn algorithm add an entropy regularizer to the prime problem and get a trade off between efficiency and accuracy. 
 In this paper, we propose a novel algorithm to further improve the efficiency and accuracy based on Nesterov's smoothing technique. 
 Basically, the non-smooth c-transform of the Kantorovich potential is approximated by the smooth Log-Sum-Exp function, which finally smooths the original non-smooth Kantorovich dual functional. The smooth Kantorovich functional can be optimized by the fast proximal gradient algorithm (FISTA) efficiently. Theoretically, the computational complexity of the proposed method is lower than 
current estimation of the Sinkhorn algorithm in terms of the precision.
 Empirically, compared with the Sinkhorn algorithm, our experimental results demonstrate that the proposed method achieves faster convergence and better accuracy with the same parameter.

----

## [1136] Local and Global Linear Convergence of General Low-Rank Matrix Recovery Problems

**Authors**: *Yingjie Bi, Haixiang Zhang, Javad Lavaei*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i9.21252](https://doi.org/10.1609/aaai.v36i9.21252)

**Abstract**:

We study the convergence rate of gradient-based local search methods for solving low-rank matrix recovery problems with general objectives in both symmetric and asymmetric cases, under the assumption of the restricted isometry property. First, we develop a new technique to verify the Polyak-Lojasiewicz inequality in a neighborhood of the global minimizers, which leads to a local linear convergence region for the gradient descent method. Second, based on the local convergence result and a sharp strict saddle property proven in this paper, we present two new conditions that guarantee the global linear convergence of the perturbed gradient descent method. The developed local and global convergence results provide much stronger theoretical guarantees than the existing results. As a by-product, this work significantly improves the existing bounds on the RIP constant required to guarantee the non-existence of spurious solutions.

----

## [1137] A*+BFHS: A Hybrid Heuristic Search Algorithm

**Authors**: *Zhaoxing Bu, Richard E. Korf*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i9.21253](https://doi.org/10.1609/aaai.v36i9.21253)

**Abstract**:

We present a new algorithm called A*+BFHS for solving problems with unit-cost operators where A* and IDA* fail due to memory limitations and/or the existence of many distinct paths between the same pair of nodes. A*+BFHS is based on A* and breadth-first heuristic search (BFHS). A*+BFHS combines advantages from both algorithms, namely A*'s node ordering, BFHS's memory savings, and both algorithms' duplicate detection. On easy problems, A*+BFHS behaves the same as A*. On hard problems, it is slower than A* but saves a large amount of memory. Compared to BFIDA*, A*+BFHS reduces the search time and/or memory requirement by several times on a variety of planning domains.

----

## [1138] NukCP: An Improved Local Search Algorithm for Maximum k-Club Problem

**Authors**: *Jiejiang Chen, Yiyuan Wang, Shaowei Cai, Minghao Yin, Yupeng Zhou, Jieyu Wu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i9.21254](https://doi.org/10.1609/aaai.v36i9.21254)

**Abstract**:

The maximum k-club problem (MkCP) is an important clique relaxation problem with wide applications. Previous MkCP algorithms only work on small-scale instances and are not applicable for large-scale instances. For solving instances with different scales, this paper develops an efficient local search algorithm named NukCP for the MkCP which mainly includes two novel ideas. First, we propose a dynamic reduction strategy, which makes a good balance between the time efficiency and the precision effectiveness of the upper bound calculation. Second, a stratified threshold configuration checking strategy is designed by giving different priorities for the neighborhood in the different levels. Experiments on a broad range of different scale instances show that NukCP significantly outperforms the state-of-the-art MkCP algorithms on most instances.

----

## [1139] Fourier Representations for Black-Box Optimization over Categorical Variables

**Authors**: *Hamid Dadkhahi, Jesus Rios, Karthikeyan Shanmugam, Payel Das*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i9.21255](https://doi.org/10.1609/aaai.v36i9.21255)

**Abstract**:

Optimization of real-world black-box functions defined over purely categorical variables is an active area of research. In particular, optimization and design of biological sequences with specific functional or structural properties have a profound impact in medicine, materials science, and biotechnology. Standalone search algorithms, such as simulated annealing (SA) and Monte Carlo tree search (MCTS), are typically used for such optimization problems. In order to improve the performance and sample efficiency of such algorithms, we propose to use existing methods in conjunction with a surrogate model for the black-box evaluations over purely categorical variables. To this end, we present two different representations, a group-theoretic Fourier expansion and an abridged one-hot encoded Boolean Fourier expansion. To learn such representations, we consider two different settings to update our surrogate model. First, we utilize an adversarial online regression setting where Fourier characters of each representation are considered as experts and their respective coefficients are updated via an exponential weight update rule each time the black box is evaluated. Second, we consider a Bayesian setting where queries are selected via Thompson sampling and the posterior is updated via a sparse Bayesian regression model (over our proposed representation) with a regularized horseshoe prior. Numerical experiments over synthetic benchmarks as well as real-world RNA sequence optimization and design problems demonstrate the representational power of the proposed methods, which achieve competitive or superior performance compared to state-of-the-art counterparts, while improving the computation cost and/or sample efficiency, substantially.

----

## [1140] New Results in Bounded-Suboptimal Search

**Authors**: *Maximilian Fickert, Tianyi Gu, Wheeler Ruml*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i9.21256](https://doi.org/10.1609/aaai.v36i9.21256)

**Abstract**:

In bounded-suboptimal heuristic search, one attempts to find a solution that costs no more than a prespecified factor of optimal as quickly as possible. This is an important setting, as it admits faster-than-optimal solving while retaining some control over solution cost. In this paper, we investigate several new algorithms for bounded-suboptimal search, including novel variants of EES and DPS, the two most prominent previous proposals, and methods inspired by recent work in bounded-cost search that leverages uncertainty estimates of the heuristic. We perform what is, to our knowledge, the most comprehensive empirical comparison of bounded-suboptimal search algorithms to date, including both search and planning benchmarks, and we find that one of the new algorithms, a simple alternating queue scheme, significantly outperforms previous work.

----

## [1141] An Exact Algorithm with New Upper Bounds for the Maximum k-Defective Clique Problem in Massive Sparse Graphs

**Authors**: *Jian Gao, Zhenghang Xu, Ruizhi Li, Minghao Yin*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i9.21257](https://doi.org/10.1609/aaai.v36i9.21257)

**Abstract**:

The Maximum k-Defective Clique Problem (MDCP), as a clique relaxation model, has been used to solve various problems. Because it is a hard computational task, previous works can hardly solve the MDCP for massive sparse graphs derived from real-world applications. In this work, we propose a novel branch-and-bound algorithm to solve the MDCP based on several new techniques. First, we propose two new upper bounds of the MDCP as well as corresponding reduction rules to remove redundant vertices and edges. The proposed reduction rules are particularly useful for massive graphs. Second, we present another new upper bound by counting missing edges between fixed vertices and an unfixed vertex for cutting branches. We perform extensive computational experiments to evaluate our algorithm. Experimental results show that our reduction rules are very effective for removing redundant vertices and edges so that graphs are reduced greatly. Also, our algorithm can solve benchmark instances efficiently, and it has significantly better performance than state-of-the-art algorithms.

----

## [1142] Learning from Mistakes - a Framework for Neural Architecture Search

**Authors**: *Bhanu Garg, Li Zhang, Pradyumna Sridhara, Ramtin Hosseini, Eric P. Xing, Pengtao Xie*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i9.21258](https://doi.org/10.1609/aaai.v36i9.21258)

**Abstract**:

Learning from one's mistakes is an effective human learning technique where the learners focus more on the topics where mistakes were made, so as to deepen their understanding. In this paper, we investigate if this human learning strategy can be applied in machine learning. We propose a novel machine learning method called Learning From Mistakes (LFM), wherein the learner improves its ability to learn by focusing more on the mistakes during revision. We formulate LFM as a three-stage optimization problem: 1) learner learns; 2) learner re-learns focusing on the mistakes, and; 3) learner validates its learning. We develop an efficient algorithm to solve the LFM problem. We apply the LFM framework to neural architecture search on CIFAR-10, CIFAR-100, and Imagenet. Experimental results strongly demonstrate the effectiveness of our model.

----

## [1143] The Complexity of Temporal Vertex Cover in Small-Degree Graphs

**Authors**: *Thekla Hamm, Nina Klobas, George B. Mertzios, Paul G. Spirakis*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i9.21259](https://doi.org/10.1609/aaai.v36i9.21259)

**Abstract**:

Temporal graphs naturally model graphs whose underlying topology changes over time. Recently, the problems Temporal Vertex Cover (or TVC) and Sliding-Window Temporal Vertex Cover (or Delta-TVC for time-windows of a fixed-length Delta) have been established as natural extensions of the classic Vertex Cover problem on static graphs with connections to areas such as surveillance in sensor networks. In this paper we initiate a systematic study of the complexity of TVC and Delta-TVC on sparse graphs. Our main result shows that for every Delta geq 2, Delta-TVC is NP-hard even when the underlying topology is described by a path or a cycle. This resolves an open problem from literature and shows a surprising contrast between Delta-TVC and TVC for which we provide a polynomial-time algorithm in the same setting. To circumvent this hardness, we present a number of exact and approximation algorithms for temporal graphs whose underlying topologies are given by a path, that have bounded vertex degree in every time step, or that admit a small-sized temporal vertex cover.

----

## [1144] Provable Sensor Sets for Epidemic Detection over Networks with Minimum Delay

**Authors**: *Jack Heavey, Jiaming Cui, Chen Chen, B. Aditya Prakash, Anil Vullikanti*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i9.21260](https://doi.org/10.1609/aaai.v36i9.21260)

**Abstract**:

The efficient detection of outbreaks and other cascading phenomena is a fundamental problem in a number of domains, including disease spread, social networks, and infrastructure networks. In such settings, monitoring and testing a small group of pre-selected nodes from the susceptible population (i.e., a sensor set) is often the preferred testing regime. We study the problem of selecting a sensor set that minimizes the delay in detection---we refer to this as the MinDelSS problem. Prior methods for minimizing the detection time rely on greedy algorithms using submodularity. We show that this approach can sometimes lead to a worse approximation for minimizing the detection time than desired. We also show that MinDelSS is hard to approximate within an O(n^(1-1/g))-factor for any constant g greater than or equal to 2 for a graph with n nodes. This instead motivates seeking a bicriteria approximations. We present the algorithm RoundSensor, which gives a rigorous worst case O(log(n))-factor for the detection time, while violating the budget by a factor of O(log^2(n)). Our algorithm is based on the sample average approximation technique from stochastic optimization, combined with linear programming and rounding. We evaluate our algorithm on several networks, including hospital contact networks, which validates its effectiveness in real settings.

----

## [1145] Towards Automated Discovery of God-Like Folk Algorithms for Rubik's Cube

**Authors**: *Garrett E. Katz, Naveed Tahir*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i9.21261](https://doi.org/10.1609/aaai.v36i9.21261)

**Abstract**:

We present a multi-objective meta-search procedure that constructs candidate algorithms for state-space search puzzles like Rubik's cube. The candidate algorithms take the form of macro databases, i.e., rule tables that specify sequences of actions to perform in different states. Rules are repeatedly applied until the puzzle is solved. The objectives favor candidates that are god-like (solving the puzzle in fewer steps) and folk-like (having fewer rules in the macro database). We build each candidate with a non-deterministic rule table construction, and then optimize over the non-deterministic choice points to find candidates near the Pareto-optimal trades-offs between godliness and folksiness. We prove that the rule table construction is correct: it always terminates and solves every state at termination. This is verified empirically on the full 2x2x2 "pocket" cube, where correct (but unoptimized) constructions take under one hour and the total number of rules is less than 10% the number of possible states. We also empirically assess the multi-objective optimization on restricted variants of the cube with up to 29K possible states, showing relative improvements in the objectives between 14-20%. Avenues for scaling up the method in future work are discussed.

----

## [1146] MIP-GNN: A Data-Driven Framework for Guiding Combinatorial Solvers

**Authors**: *Elias B. Khalil, Christopher Morris, Andrea Lodi*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i9.21262](https://doi.org/10.1609/aaai.v36i9.21262)

**Abstract**:

Mixed-integer programming (MIP) technology offers a generic way of formulating and solving combinatorial optimization problems. While generally reliable, state-of-the-art MIP solvers base many crucial decisions on hand-crafted heuristics, largely ignoring common patterns within a given instance distribution of the problem of interest. Here, we propose MIP-GNN, a general framework for enhancing such solvers with data-driven insights. By encoding the variable-constraint interactions of a given mixed-integer linear program (MILP) as a bipartite graph, we leverage state-of-the-art graph neural network architectures to predict variable biases, i.e., component-wise averages of (near) optimal solutions, indicating how likely a variable will be set to 0 or 1 in (near) optimal solutions of binary MILPs. In turn, the predicted biases stemming from a single, once-trained model are used to guide the solver, replacing heuristic components. We integrate MIP-GNN into a state-of-the-art MIP solver, applying it to tasks such as node selection and warm-starting, showing significant improvements compared to the default setting of the solver on two classes of challenging binary MILPs. Our code and appendix are publicly available at https://github.com/lyeskhalil/mipGNN.

----

## [1147] Bandit Limited Discrepancy Search and Application to Machine Learning Pipeline Optimization

**Authors**: *Akihiro Kishimoto, Djallel Bouneffouf, Radu Marinescu, Parikshit Ram, Ambrish Rawat, Martin Wistuba, Paulito P. Palmes, Adi Botea*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i9.21263](https://doi.org/10.1609/aaai.v36i9.21263)

**Abstract**:

Optimizing a machine learning (ML) pipeline has been an important topic of AI and ML. Despite recent progress, pipeline optimization remains a challenging problem, due to potentially many combinations to consider as well as slow training and validation. We present the BLDS algorithm for optimized algorithm selection (ML operations) in a fixed ML pipeline structure. BLDS performs multi-fidelity optimization for selecting ML algorithms trained with smaller computational overhead, while controlling its pipeline search based on multi-armed bandit and limited discrepancy search. Our experiments on well-known classification benchmarks show that BLDS is superior to competing algorithms. We also combine BLDS with hyperparameter optimization, empirically showing the advantage of BLDS.

----

## [1148] PRISM: A Rich Class of Parameterized Submodular Information Measures for Guided Data Subset Selection

**Authors**: *Suraj Kothawade, Vishal Kaushal, Ganesh Ramakrishnan, Jeff A. Bilmes, Rishabh K. Iyer*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i9.21264](https://doi.org/10.1609/aaai.v36i9.21264)

**Abstract**:

With ever-increasing dataset sizes, subset selection techniques are becoming increasingly important for a plethora of tasks. It is often necessary to guide the subset selection to achieve certain desiderata, which includes focusing or targeting certain data points, while avoiding others. Examples of such problems include: i)targeted learning, where the goal is to find subsets with rare classes or rare attributes on which the model is under performing, and ii)guided summarization, where data (e.g., image collection, text, document or video) is summarized for quicker human consumption with specific additional user intent. Motivated by such applications, we present PRISM, a rich class of PaRameterIzed Submodular information Measures. Through novel functions and their parameterizations, PRISM offers a variety of modeling capabilities that enable a trade-off between desired qualities of a subset like diversity or representation and similarity/dissimilarity with a set of data points. We demonstrate how PRISM can be applied to the two real-world problems mentioned above, which require guided subset selection. In doing so, we show that PRISM interestingly generalizes some past work, therein reinforcing its broad utility. Through extensive experiments on diverse datasets, we demonstrate the superiority of PRISM over the state-of-the-art in targeted learning and in guided image-collection summarization. PRISM is available as a part of the SUBMODLIB (https://github.com/decile-team/submodlib) and TRUST (https://github.com/decile-team/trust) toolkits.

----

## [1149] Split Moves for Monte-Carlo Tree Search

**Authors**: *Jakub Kowalski, Maksymilian Mika, Wojciech Pawlik, Jakub Sutowicz, Marek Szykula, Mark H. M. Winands*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i9.21265](https://doi.org/10.1609/aaai.v36i9.21265)

**Abstract**:

In many games, moves consist of several decisions made by the player. These decisions can be viewed as separate moves, which is already a common practice in multi-action games for efficiency reasons. Such division of a player move into a sequence of simpler / lower level moves is called splitting. So far, split moves have been applied only in forementioned straightforward cases, and furthermore, there was almost no study revealing its impact on agents' playing strength. Taking the knowledge-free perspective, we aim to answer how to effectively use split moves within Monte-Carlo Tree Search (MCTS) and what is the practical impact of split design on agents' strength. This paper proposes a generalization of MCTS that works with arbitrarily split moves. We design several variations of the algorithm and try to measure the impact of split moves separately on efficiency, quality of MCTS, simulations, and action-based heuristics. The tests are carried out on a set of board games and performed using the Regular Boardgames General Game Playing formalism, where split strategies of different granularity can be automatically derived based on an abstract description of the game. The results give an overview of the behavior of agents using split design in different ways. We conclude that split design can be greatly beneficial for single- as well as multi-action games.

----

## [1150] MAPF-LNS2: Fast Repairing for Multi-Agent Path Finding via Large Neighborhood Search

**Authors**: *Jiaoyang Li, Zhe Chen, Daniel Harabor, Peter J. Stuckey, Sven Koenig*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i9.21266](https://doi.org/10.1609/aaai.v36i9.21266)

**Abstract**:

Multi-Agent Path Finding (MAPF) is the problem of planning collision-free paths for multiple agents in a shared environment. In this paper, we propose a novel algorithm MAPF-LNS2 based on large neighborhood search for solving MAPF efficiently. Starting from a set of paths that contain collisions, MAPF-LNS2 repeatedly selects a subset of colliding agents and replans their paths to reduce the number of collisions until the paths become collision-free. We compare MAPF-LNS2 against a variety of state-of-the-art MAPF algorithms, including Prioritized Planning with random restarts, EECBS, and PPS, and show that MAPF-LNS2 runs significantly faster than them while still providing near-optimal solutions in most cases. MAPF-LNS2 solves 80% of the random-scenario instances with the largest number of agents from the MAPF benchmark suite with a runtime limit of just 5 minutes, which, to our knowledge, has not been achieved by any existing algorithms.

----

## [1151] Local and Global Convergence of General Burer-Monteiro Tensor Optimizations

**Authors**: *Shuang Li, Qiuwei Li*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i9.21267](https://doi.org/10.1609/aaai.v36i9.21267)

**Abstract**:

Tensor optimization is crucial to massive machine learning and signal processing tasks. In this paper, we consider tensor optimization with a convex and well-conditioned objective function and reformulate it into a nonconvex optimization using the Burer-Monteiro type parameterization. We analyze the local convergence of applying vanilla gradient descent to the factored formulation and establish a local regularity condition under mild assumptions. We also provide a linear convergence analysis of the gradient descent algorithm started in a neighborhood of the true tensor factors.  Complementary to the local analysis, this work also characterizes the global geometry of the best rank-one tensor approximation problem and demonstrates that for orthogonally decomposable tensors the problem has no spurious local minima and all saddle points are strict except for the one at zero which is a third-order saddle point.

----

## [1152] Bi-CMR: Bidirectional Reinforcement Guided Hashing for Effective Cross-Modal Retrieval

**Authors**: *Tieying Li, Xiaochun Yang, Bin Wang, Chong Xi, Hanzhong Zheng, Xiangmin Zhou*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i9.21268](https://doi.org/10.1609/aaai.v36i9.21268)

**Abstract**:

Cross-modal hashing has attracted considerable attention for large-scale multimodal data. Recent supervised cross-modal hashing methods using multi-label networks utilize the semantics of multi-labels to enhance retrieval accuracy, where label hash codes are learned independently. However, all these methods assume that label annotations reliably reflect the relevance between their corresponding instances, which is not true in real applications. In this paper, we propose a novel framework called Bidirectional Reinforcement Guided Hashing for Effective Cross-Modal Retrieval (Bi-CMR), which exploits a bidirectional learning to relieve the negative impact of this assumption. Specifically, in the forward learning procedure, we highlight the representative labels and learn the reinforced multi-label hash codes by intra-modal semantic information, and further adjust similarity matrix. In the backward learning procedure, the reinforced multi-label hash codes and adjusted similarity matrix are used to guide the matching of instances. We construct two datasets with explicit relevance labels that reflect the semantic relevance of instance pairs based on two benchmark datasets. The Bi-CMR is evaluated by conducting extensive experiments over these two datasets. Experimental results prove the superiority of Bi-CMR over four state-of-the-art methods in terms of effectiveness.

----

## [1153] Improving Local Search Algorithms via Probabilistic Configuration Checking

**Authors**: *Weilin Luo, Rongzhen Ye, Hai Wan, Shaowei Cai, Biqing Fang, Delong Zhang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i9.21269](https://doi.org/10.1609/aaai.v36i9.21269)

**Abstract**:

Configuration checking (CC) has been confirmed to alleviate the cycling problem in local search for combinatorial optimization problems (COPs). When using CC heuristics in local search for graph problems, a critical concept is the configuration of the vertices. All existing CC variants employ either 1- or 2-level neighborhoods of a vertex as its configuration. Inspired by the idea that neighborhoods with different levels should have different contributions to solving COPs, we propose the probabilistic configuration (PC), which introduces probabilities for neighborhoods at different levels to consider the impact of neighborhoods of different levels on the CC strategy. Based on the concept of PC, we first propose probabilistic configuration checking (PCC), which can be developed in an automated and lightweight favor. We then apply PCC to two classic COPs which have been shown to achieve good results by using CC, and our preliminary results confirm that PCC improves the existing algorithms because PCC alleviates the cycling problem.

----

## [1154] PEA*+IDA*: An Improved Hybrid Memory-Restricted Algorithm

**Authors**: *Frederico Messa, André Grahl Pereira*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i9.21270](https://doi.org/10.1609/aaai.v36i9.21270)

**Abstract**:

It is well-known that the search algorithms A* and Iterative Deepening A* (IDA*) can fail to solve state-space tasks optimally due to time and memory limits. The former typically fails in memory-restricted scenarios and the latter in time-restricted scenarios. Therefore, several algorithms were proposed to solve state-space tasks optimally using less memory than A* and less time than IDA*, such as A*+IDA*, a hybrid memory-restricted algorithm that combines A* and IDA*. In this paper, we present a hybrid memory-restricted algorithm that combines Partial Expansion A* (PEA*) and IDA*. This new algorithm has two phases, the same structure as the A*+IDA* algorithm. The first phase of PEA*+IDA* runs PEA* until it reaches a memory limit, and the second phase runs IDA* without duplicate detection on each node of PEA*'s Open. First, we present a model that shows how PEA*+IDA* can perform better than A*+IDA* although pure PEA* usually makes more expansions than pure A*. Later, we perform an experimental evaluation using three memory limits and show that, compared to A*+IDA* on classical planning domains, PEA*+IDA* has higher coverage and expands fewer nodes. Finally, we experimentally analyze both algorithms and show that having higher F-limits and better priority-queue composition given by PEA* have a considerable impact on the performance of the algorithms.

----

## [1155] Search Strategies for Topological Network Optimization

**Authors**: *Michael D. Moffitt*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i9.21271](https://doi.org/10.1609/aaai.v36i9.21271)

**Abstract**:

We consider an application of combinatorial search to the optimization of topologies in series-parallel networks. We propose a recursive search over the space of decomposition trees, in which partial solutions are obtained by exploring k-way partitionings of expandable nodes. We present two complementary pruning techniques that bound the value of intermediate solutions from above and below, applying monotonic operations to the contents of unresolved leaves. We also develop a means to exploit the convexity of our objective function, so as to prevent the redundant recomputation of subcircuit configurations. Finally, we evaluate our approach on a parameterized benchmark suite of electrical circuits, demonstrating over an order of magnitude improvement in performance as compared to a baseline implementation.

----

## [1156] Hibernated Backdoor: A Mutual Information Empowered Backdoor Attack to Deep Neural Networks

**Authors**: *Rui Ning, Jiang Li, Chunsheng Xin, Hongyi Wu, Chonggang Wang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i9.21272](https://doi.org/10.1609/aaai.v36i9.21272)

**Abstract**:

We report a new neural backdoor attack, named Hibernated Backdoor, which is stealthy, aggressive and devastating. The backdoor is planted in a hibernated mode to avoid being detected. Once deployed and fine-tuned on end-devices, the hibernated backdoor turns into the active state that can be exploited by the attacker. To the best of our knowledge, this is the first hibernated neural backdoor attack. It is achieved by maximizing the mutual information (MI) between the gradients of regular and malicious data on the model. We introduce a practical algorithm to achieve MI maximization to effectively plant the hibernated backdoor. To evade adaptive defenses, we further develop a targeted hibernated backdoor, which can only be activated by specific data samples and thus achieves a higher degree of stealthiness. We show the hibernated backdoor is robust and cannot be removed by existing backdoor removal schemes. It has been fully tested on four datasets with two neural network architectures, compared to five existing backdoor attacks, and evaluated using seven backdoor detection schemes. The experiments demonstrate the effectiveness of the hibernated backdoor attack under various settings.

----

## [1157] Planning with Explanations for Finding Desired Meeting Points on Graphs

**Authors**: *Keisuke Otaki*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i9.21273](https://doi.org/10.1609/aaai.v36i9.21273)

**Abstract**:

Combinatorial optimization problems are ubiquitous for decision making in planning social infrastructures.
In real-world scenarios, a decision-maker needs to solve his/her problem iteratively until he/she satisfies solutions, but such an iterative process remains challenging.
This paper studies a new explainable framework, particularly for finding meeting points, which is a key optimization problem for designing facility locations.
Our framework automatically fills the gap between its input instance and instances from which a user could obtain the desired outcome, where computed solutions are judged by the user.
The framework also provides users with explanations, representing the difference of instances for deeply understanding the process and its inside.
Explanations are clues for users to understand their situation and implement suggested results in practice (e.g., designing a coupon for free travel).
We experimentally demonstrate that our search-based framework is promising to solve instances with generating explanations in a sequential decision-making process.

----

## [1158] A Fast Local Search Algorithm for the Latin Square Completion Problem

**Authors**: *Shiwei Pan, Yiyuan Wang, Minghao Yin*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i9.21274](https://doi.org/10.1609/aaai.v36i9.21274)

**Abstract**:

The Latin square completion (LSC) problem is an important NP-complete problem with numerous applications. Given its theoretical and practical importance, several algorithms are designed for solving the LSC problem. In this work, to further improve the performance, a fast local search algorithm is developed based on three main ideas. Firstly, a reduction reasoning technique is used to reduce the scale of search space. Secondly, we propose a novel conflict value selection heuristic, which considers the history conflicting information of vertices as a selection criterion when more than one vertex have equal values on the primary scoring function. Thirdly, during the search phase, we record previous history search information and then make use of these information to restart the candidate solution. Experimental results show that our proposed algorithm significantly outperforms the state-of-the-art heuristic algorithms on almost all instances in terms of success rate and run time.

----

## [1159] Sparsification of Decomposable Submodular Functions

**Authors**: *Akbar Rafiey, Yuichi Yoshida*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i9.21275](https://doi.org/10.1609/aaai.v36i9.21275)

**Abstract**:

Submodular functions are at the core of many machine learning and data mining tasks. The underlying submodular functions for many of these tasks are decomposable, i.e., they are sum of several simple submodular functions. In many data intensive applications, however, the number of underlying submodular functions in the original function is so large that we need prohibitively large amount of time to process it and/or it does not even fit in the main memory. To overcome this issue, we introduce the notion of sparsification for decomposable submodular functions whose objective is to obtain an accurate approximation of the original function that is a (weighted) sum of only a few submodular functions. Our main result is a polynomial-time randomized sparsification algorithm such that the expected number of functions used in the output is independent of the number of underlying submodular functions in the original function. We also study the effectiveness of our algorithm under various constraints such as matroid and cardinality constraints. We complement our theoretical analysis with an empirical study of the performance of our algorithm.

----

## [1160] Subset Approximation of Pareto Regions with Bi-objective A

**Authors**: *Nicolás Rivera, Jorge A. Baier, Carlos Hernández*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i9.21276](https://doi.org/10.1609/aaai.v36i9.21276)

**Abstract**:

In bi-objective search, we are given a graph in which each directed arc is associated with a pair of non-negative weights, and the objective is to find the Pareto-optimal solution set. Unfortunately, in many practical settings, this set is too large, and therefore its computation is very time-consuming. In addition, even though bi-objective search algorithms generate the Pareto set incrementally, they do so exhaustively. This means that early during search the solution set covers is not diverse, being concentrated in a small region of the solution set. To address this issue, we present a new approach to subset approximation of the solution set, that can be used as the basis for an anytime bi-objective search algorithm. Our approach transforms the given task into a target bi-objective search task using two real parameters. For each particular parameter setting, the solutions to the target task is a subset of the solution set of the original task. Depending on the parameters used, the solution set of the target task may be computed very quickly. This
 allows us to obtain, in challenging road map benchmarks, a rich variety of solutions in times that may be orders of magnitude smaller than the time needed to compute the solution set. We show that by running the algorithm with an appropriate sequence of parameters, we obtain a growing sequence of solutions that converges to the full solution set. We prove that our approach is correct and that Bi-Objective A* prunes at least as many nodes when run over the target task.

----

## [1161] On Probabilistic Generalization of Backdoors in Boolean Satisfiability

**Authors**: *Alexander A. Semenov, Artem Pavlenko, Daniil Chivilikhin, Stepan Kochemazov*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i9.21277](https://doi.org/10.1609/aaai.v36i9.21277)

**Abstract**:

The paper proposes a probabilistic generalization of the well-known Strong Backdoor Set (SBS) concept applied to the Boolean Satisfiability Problem (SAT). We call a set of Boolean variables B a ρ-backdoor, if for a fraction of at least ρ of possible assignments of variables from B, assigning their values to variables in a Boolean formula in Conjunctive Normal Form (CNF) results in polynomially solvable formulas. Clearly, a ρ-backdoor with ρ=1 is an SBS. For a given set B it is possible to efficiently construct an (ε, δ)-approximation of parameter ρ using the Monte Carlo method. Thus, we define an (ε, δ)-SBS as such a set B for which the conclusion "parameter ρ deviates from 1 by no more than ε" is true with probability no smaller than 1 - δ. We consider the problems of finding the minimum SBS and the minimum (ε, δ)-SBS. To solve the former problem, one can use the algorithm described by R. Williams, C. Gomes and B. Selman in 2003. In the paper we propose a new probabilistic algorithm to solve the latter problem, and show that the asymptotic estimation of the worst-case complexity of the proposed algorithm is significantly smaller than that of the algorithm by Williams et al. For practical applications, we suggest a metaheuristic optimization algorithm based on the penalty function method to seek the minimal (ε, δ)-SBS. Results of computational experiments show that the use of (ε, δ)-SBSes found by the proposed algorithm allows speeding up solving of test problems related to equivalence checking and hard crafted and combinatorial benchmarks compared to state-of-the-art SAT solvers.

----

## [1162] A Novel Approach to Solving Goal-Achieving Problems for Board Games

**Authors**: *Chung-Chin Shih, Ti-Rong Wu, Ting-Han Wei, I-Chen Wu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i9.21278](https://doi.org/10.1609/aaai.v36i9.21278)

**Abstract**:

Goal-achieving problems are puzzles that set up a specific situation with a clear objective. An example that is well-studied is the category of life-and-death (L&D) problems for Go, which helps players hone their skill of identifying region safety. Many previous methods like lambda search try null moves first, then derive so-called relevance zones (RZs), outside of which the opponent does not need to search. This paper first proposes a novel RZ-based approach, called the RZ-Based Search (RZS), to solving L&D problems for Go. RZS tries moves before determining whether they are null moves post-hoc. This means we do not need to rely on null move heuristics, resulting in a more elegant algorithm, so that it can also be seamlessly incorporated into AlphaZero's super-human level play in our solver. To repurpose AlphaZero for solving, we also propose a new training method called Faster to Life (FTL), which modifies AlphaZero to entice it to win more quickly. We use RZS and FTL to solve L&D problems on Go, namely solving 68 among 106 problems from a professional L&D book while a previous state-of-the-art program TSUMEGO-EXPLORER solves 11 only. Finally, we discuss that the approach is generic in the sense that RZS is applicable to solving many other goal-achieving problems for board games.

----

## [1163] Machine Learning for Online Algorithm Selection under Censored Feedback

**Authors**: *Alexander Tornede, Viktor Bengs, Eyke Hüllermeier*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i9.21279](https://doi.org/10.1609/aaai.v36i9.21279)

**Abstract**:

In online algorithm selection (OAS), instances of an algorithmic problem class are presented to an agent one after another, and the agent has to quickly select a presumably best algorithm from a fixed set of candidate algorithms. For decision problems such as satisfiability (SAT), quality typically refers to the algorithm's runtime. As the latter is known to exhibit a heavy-tail distribution, an algorithm is normally stopped when exceeding a predefined upper time limit. As a consequence, machine learning methods used to optimize an algorithm selection strategy in a data-driven manner need to deal with right-censored samples, a problem that has received little attention in the literature so far. In this work, we revisit multi-armed bandit algorithms for OAS and discuss their capability of dealing with the problem. Moreover, we adapt them towards runtime-oriented losses, allowing for partially censored data while keeping a space- and time-complexity independent of the time horizon. In an extensive experimental evaluation on an adapted version of the ASlib benchmark, we demonstrate that theoretically well-founded methods based on Thompson sampling perform specifically strong and improve in comparison to existing methods.

----

## [1164] Procrastinated Tree Search: Black-Box Optimization with Delayed, Noisy, and Multi-Fidelity Feedback

**Authors**: *Junxiong Wang, Debabrota Basu, Immanuel Trummer*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i9.21280](https://doi.org/10.1609/aaai.v36i9.21280)

**Abstract**:

In black-box optimization problems, we aim to maximize an unknown objective function, where the function is only accessible through feedbacks of an evaluation or simulation oracle. In real-life, the feedbacks of such oracles are often noisy and available after some unknown delay that may depend on the computation time of the oracle. Additionally, if the exact evaluations are expensive but coarse approximations are available at a lower cost, the feedbacks can have multi-fidelity. In order to address this problem, we propose a generic extension of hierarchical optimistic tree search (HOO), called ProCrastinated Tree Search (PCTS), that flexibly accommodates a delay and noise-tolerant bandit algorithm. We provide a generic proof technique to quantify regret of PCTS under delayed, noisy, and multi-fidelity feedbacks. Specifically, we derive regret bounds of PCTS enabled with delayed-UCB1 (DUCB1) and delayed-UCB-V (DUCBV) algorithms. Given a horizon T, PCTS retains the regret bound of non-delayed HOO for expected delay of O(log T), and worsens by T^((1-α)/(d+2)) for expected delays of O(T^(1-α)) for α ∈ (0,1]. We experimentally validate on multiple synthetic functions and hyperparameter tuning problems that PCTS outperforms the state-of-the-art black-box optimization methods for feedbacks with different noise levels, delays, and fidelity.

----

## [1165] DPCD: Discrete Principal Coordinate Descent for Binary Variable Problems

**Authors**: *Huan Xiong*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i9.21281](https://doi.org/10.1609/aaai.v36i9.21281)

**Abstract**:

Binary optimization, a representative subclass of discrete optimization, plays an important role in mathematical optimization and has various applications in computer vision and machine learning. Generally speaking, binary optimization problems are NP-hard and difficult to solve due to the binary constraints, especially when the number of variables is very large. Existing methods often suffer from high computational costs or large accumulated quantization errors, or are only designed for specific tasks. In this paper, we propose an efficient algorithm, named Discrete Principal Coordinate Descent (DPCD), to find effective approximate solutions for general binary optimization problems. The proposed algorithm iteratively solves optimization problems related to the linear approximation of loss functions, which leads to updating the binary variables that most impact the value of the loss functions at each step. Our method supports a wide range of empirical objective functions with/without restrictions on the numbers of 1s and -1s in the binary variables. Furthermore, the theoretical convergence of our algorithm is proven, and the explicit convergence rates are derived for objective functions with Lipschitz continuous gradients, which are commonly adopted in practice. Extensive experiments on binary hashing tasks and large-scale datasets demonstrate the superiority of the proposed algorithm over several state-of-the-art methods in terms of both effectiveness and efficiency.

----

## [1166] Optimize What You Evaluate With: Search Result Diversification Based on Metric Optimization

**Authors**: *Hai-Tao Yu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i9.21282](https://doi.org/10.1609/aaai.v36i9.21282)

**Abstract**:

Most of the existing methods for search result diversification (SRD) appeal to the greedy strategy for generating diversified results, which is formulated as a sequential process of selecting documents one-by-one, and the locally optimal choice is made at each round. Unfortunately, this strategy suffers from the following shortcomings: (1) Such a one-by-one selection process is rather time-consuming for both training and inference. (2) It works well on the premise that the preceding choices are optimal or close to the optimal solution. (3) The mismatch between the objective function used in training and the final evaluation measure used in testing has not been taken into account. We propose a novel framework through direct metric optimization for SRD (referred to as MO4SRD) based on the score-and-sort strategy. Specifically, we represent the diversity score of each document that determines its rank position based on a probability distribution. These distributions over scores naturally give rise to expectations over rank positions. Armed with this advantage, we can get the differentiable variants of the widely used diversity metrics. Thanks to this, we are able to directly optimize the evaluation measure used in testing. Moreover, we have devised a novel probabilistic neural scoring function. It jointly scores candidate documents by taking into account both cross-document interaction and permutation equivariance, which makes it possible to generate a diversified ranking via a simple sorting. The experimental results on benchmark collections show that the proposed method achieves significantly improved performance over the state-of-the-art results.

----

## [1167] A First Mathematical Runtime Analysis of the Non-dominated Sorting Genetic Algorithm II (NSGA-II)

**Authors**: *Weijie Zheng, Yufei Liu, Benjamin Doerr*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i9.21283](https://doi.org/10.1609/aaai.v36i9.21283)

**Abstract**:

The non-dominated sorting genetic algorithm II (NSGA-II) is the most intensively used multi-objective evolutionary algorithm (MOEA) in real-world applications. However, in contrast to several simple MOEAs analyzed also via mathematical means, no such study exists for the NSGA-II so far. In this work, we show that mathematical runtime analyses are feasible also for the NSGA-II. As particular results, we prove that with a population size larger than the Pareto front size by a constant factor, the NSGA-II with two classic mutation operators and three different ways to select the parents satisfies the same asymptotic runtime guarantees as the SEMO and GSEMO algorithms on the basic OneMinMax and LOTZ benchmark functions. However, if the population size is only equal to the size of the Pareto front, then the NSGA-II cannot efficiently compute the full Pareto front (for an exponential number of iterations, the population will always miss a constant fraction of the Pareto front). Our experiments confirm the above findings.

----

## [1168] Pinpointing Fine-Grained Relationships between Hateful Tweets and Replies

**Authors**: *Abdullah Albanyan, Eduardo Blanco*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i10.21284](https://doi.org/10.1609/aaai.v36i10.21284)

**Abstract**:

Recent studies in the hate and counter hate domain have provided the grounds for investigating how to detect this pervasive content in social media. These studies mostly work with synthetic replies to hateful content written by annotators on demand rather than replies written by real users. We argue that working with naturally occurring replies to hateful content is key to study the problem. Building on this motivation, we create a corpus of 5,652 hateful tweets and replies. We analyze their fine-grained relationships by indicating whether the reply (a) is hate or counter hate speech, (b) provides a justification, (c) attacks the author of the tweet, and (d) adds additional hate. We also present linguistic insights into the language people use depending on these fine-grained relationships. Experimental results show improvements (a) taking into account the hateful tweet in addition to the reply and (b) pretraining with related tasks.

----

## [1169] Cross-Modal Coherence for Text-to-Image Retrieval

**Authors**: *Malihe Alikhani, Fangda Han, Hareesh Ravi, Mubbasir Kapadia, Vladimir Pavlovic, Matthew Stone*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i10.21285](https://doi.org/10.1609/aaai.v36i10.21285)

**Abstract**:

Common image-text joint understanding techniques presume that images and the associated text can universally be characterized by a single implicit model. However, co-occurring images and text can be related in qualitatively different ways, and explicitly modeling it could improve the performance of current joint understanding models. In this paper, we train a Cross-Modal Coherence Model for text-to-image retrieval task. Our analysis shows that models trained with image–text coherence relations can retrieve images originally paired with target text more often than coherence-agnostic models. We also show via human evaluation that images retrieved by the proposed coherence-aware model are preferred over a coherence-agnostic baseline by a huge margin. Our findings provide insights into the ways that different modalities communicate and the role of coherence relations in capturing commonsense inferences in text and imagery.

----

## [1170] Enhanced Story Comprehension for Large Language Models through Dynamic Document-Based Knowledge Graphs

**Authors**: *Berkeley R. Andrus, Yeganeh Nasiri, Shilong Cui, Benjamin Cullen, Nancy Fulda*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i10.21286](https://doi.org/10.1609/aaai.v36i10.21286)

**Abstract**:

Large transformer-based language models have achieved incredible success at various tasks which require narrative comprehension, including story completion, answering questions about stories, and generating stories ex nihilo. However, due to the limitations of finite context windows, these language models struggle to produce or understand stories longer than several thousand tokens. In order to mitigate the document length limitations that come with finite context windows, we introduce a novel architecture that augments story processing with an external dynamic knowledge graph. In contrast to static commonsense knowledge graphs which hold information about the real world, these dynamic knowledge graphs reflect facts extracted from the story being processed. Our architecture uses these knowledge graphs to create information-rich prompts which better facilitate story comprehension than prompts composed only of story text. We apply our architecture to the tasks of question answering and story completion. To complement this line of research, we introduce two long-form question answering tasks, LF-SQuAD and LF-QUOREF, in which the document length exceeds the size of the language model's context window, and introduce a story completion evaluation method that bypasses the stochastic nature of language model generation. We demonstrate broad improvement over typical prompt formulation methods for both question answering and story completion using GPT-2, GPT-3 and XLNet.

----

## [1171] Diagnostics-Guided Explanation Generation

**Authors**: *Pepa Atanasova, Jakob Grue Simonsen, Christina Lioma, Isabelle Augenstein*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i10.21287](https://doi.org/10.1609/aaai.v36i10.21287)

**Abstract**:

Explanations shed light on a machine learning model's rationales and can aid in identifying deficiencies in its reasoning process. Explanation generation models are typically trained in a supervised way given human explanations. When such annotations are not available, explanations are often selected as those portions of the input that maximise a downstream task's performance, which corresponds to optimising an explanation's Faithfulness to a given model. Faithfulness is one of several so-called diagnostic properties, which prior work has identified as useful for gauging the quality of an explanation without requiring annotations. Other diagnostic properties are Data Consistency, which measures how similar explanations are for similar input instances, and Confidence Indication, which shows whether the explanation reflects the confidence of the model. In this work, we show how to directly optimise for these diagnostic properties when training a model to generate sentence-level explanations, which markedly improves explanation quality, agreement with human rationales, and downstream task performance on three complex reasoning tasks.

----

## [1172] Mitigating Reporting Bias in Semi-supervised Temporal Commonsense Inference with Probabilistic Soft Logic

**Authors**: *Bibo Cai, Xiao Ding, Bowen Chen, Li Du, Ting Liu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i10.21288](https://doi.org/10.1609/aaai.v36i10.21288)

**Abstract**:

Acquiring high-quality temporal common sense (TCS) knowledge from free-form text is a crucial but challenging problem for event-centric natural language understanding, due to the language reporting bias problem: people rarely report the commonly observed events but highlight the special cases. For example, one may rarely report "I get up from bed in 1 minute", but we can observe "It takes me an hour to get up from bed every morning'' in text. Models directly trained upon such corpus would capture distorted TCS knowledge, which could influence the model performance. Prior work addresses this issue mainly by exploiting the interactions among temporal dimensions (e.g., duration, temporal relation between events) in a multi-task view. However, this line of work suffers the limitation of implicit, inadequate and unexplainable interactions modeling. In this paper, we propose a novel neural-logic based Soft Logic Enhanced Event Temporal Reasoning (SLEER) model for acquiring unbiased TCS knowledge, in which the complementary relationship among dimensions are explicitly represented as logic rules and modeled by t-norm fuzzy logics. SLEER can utilize logic rules to regularize its inference process. Experimental results on four intrinsic evaluation datasets and two extrinsic datasets show the efficiency of our proposed method.

----

## [1173] Adversarial Training for Improving Model Robustness? Look at Both Prediction and Interpretation

**Authors**: *Hanjie Chen, Yangfeng Ji*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i10.21289](https://doi.org/10.1609/aaai.v36i10.21289)

**Abstract**:

Neural language models show vulnerability to adversarial examples which are semantically similar to their original counterparts with a few words replaced by their synonyms. A common way to improve model robustness is adversarial training which follows two steps—collecting adversarial examples by attacking a target model, and fine-tuning the model on the augmented dataset with these adversarial examples. The objective of traditional adversarial training is to make a model produce the same correct predictions on an original/adversarial example pair. However, the consistency between model decision-makings on two similar texts is ignored. We argue that a robust model should behave consistently on original/adversarial example pairs, that is making the same predictions (what) based on the same reasons (how) which can be reflected by consistent interpretations. In this work, we propose a novel feature-level adversarial training method named FLAT. FLAT aims at improving model robustness in terms of both predictions and interpretations. FLAT incorporates variational word masks in neural networks to learn global word importance and play as a bottleneck teaching the model to make predictions based on important words. FLAT explicitly shoots at the vulnerability problem caused by the mismatch between model understandings on the replaced words and their synonyms in original/adversarial example pairs by regularizing the corresponding global word importance scores. Experiments show the effectiveness of FLAT in improving the robustness with respect to both predictions and interpretations of four neural network models (LSTM, CNN, BERT, and DeBERTa) to two adversarial attacks on four text classification tasks. The models trained via FLAT also show better robustness than baseline models on unforeseen adversarial examples across different attacks.

----

## [1174] Unsupervised Editing for Counterfactual Stories

**Authors**: *Jiangjie Chen, Chun Gan, Sijie Cheng, Hao Zhou, Yanghua Xiao, Lei Li*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i10.21290](https://doi.org/10.1609/aaai.v36i10.21290)

**Abstract**:

Creating what-if stories requires reasoning about prior statements and possible outcomes of the changed conditions. One can easily generate coherent endings under new conditions, but it would be challenging for current systems to do it with minimal changes to the original story. Therefore, one major challenge is the trade-off between generating a logical story and rewriting with minimal-edits. In this paper, we propose EDUCAT, an editing-based unsupervised approach for counterfactual story rewriting. EDUCAT includes a target position detection strategy based on estimating causal effects of the what-if conditions, which keeps the causal invariant parts of the story. EDUCAT then generates the stories under fluency, coherence and minimal-edits constraints. We also propose a new metric to alleviate the shortcomings of current automatic metrics and better evaluate the trade-off. We evaluate EDUCAT on a public counterfactual story rewriting benchmark. Experiments show that EDUCAT achieves the best trade-off over unsupervised SOTA methods according to both automatic and human evaluation. The resources of EDUCAT are available at: https://github.com/jiangjiechen/EDUCAT.

----

## [1175] LOREN: Logic-Regularized Reasoning for Interpretable Fact Verification

**Authors**: *Jiangjie Chen, Qiaoben Bao, Changzhi Sun, Xinbo Zhang, Jiaze Chen, Hao Zhou, Yanghua Xiao, Lei Li*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i10.21291](https://doi.org/10.1609/aaai.v36i10.21291)

**Abstract**:

Given a natural language statement, how to verify its veracity against a large-scale textual knowledge source like Wikipedia? Most existing neural models make predictions without giving clues about which part of a false claim goes wrong. In this paper, we propose LOREN, an approach for interpretable fact verification. We decompose the verification of the whole claim at phrase-level, where the veracity of the phrases serves as explanations and can be aggregated into the final verdict according to logical rules. The key insight of LOREN is to represent claim phrase veracity as three-valued latent variables, which are regularized by aggregation logical rules. The final claim verification is based on all latent variables. Thus, LOREN enjoys the additional benefit of interpretability --- it is easy to explain how it reaches certain results with claim phrase veracity. Experiments on a public fact verification benchmark show that LOREN is competitive against previous approaches while enjoying the merit of faithful and accurate interpretability. The resources of LOREN are available at: https://github.com/jiangjiechen/LOREN.

----

## [1176] ContrastNet: A Contrastive Learning Framework for Few-Shot Text Classification

**Authors**: *Junfan Chen, Richong Zhang, Yongyi Mao, Jie Xu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i10.21292](https://doi.org/10.1609/aaai.v36i10.21292)

**Abstract**:

Few-shot text classification has recently been promoted by the meta-learning paradigm which aims to identify target classes with knowledge transferred from source classes with sets of small tasks named episodes. Despite their success, existing works building their meta-learner based on Prototypical Networks are unsatisfactory in learning discriminative text representations between similar classes, which may lead to contradictions during label prediction. In addition, the task-level and instance-level overfitting problems in few-shot text classification caused by a few training examples are not sufficiently tackled. In this work, we propose a contrastive learning framework named ContrastNet to tackle both discriminative representation and overfitting problems in few-shot text classification. ContrastNet learns to pull closer text representations belonging to the same class and push away text representations belonging to different classes, while simultaneously introducing unsupervised contrastive regularization at both task-level and instance-level to prevent overfitting. Experiments on 8 few-shot text classification datasets show that ContrastNet outperforms the current state-of-the-art models.

----

## [1177] From Good to Best: Two-Stage Training for Cross-Lingual Machine Reading Comprehension

**Authors**: *Nuo Chen, Linjun Shou, Ming Gong, Jian Pei*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i10.21293](https://doi.org/10.1609/aaai.v36i10.21293)

**Abstract**:

Cross-lingual Machine Reading Comprehension (xMRC) is a challenging task due to the lack of training data in low-resource languages. Recent approaches use training data only in a resource-rich language (such as English) to fine-tune large-scale cross-lingual pre-trained language models, which transfer knowledge from resource-rich languages (source) to low-resource languages (target). Due to the big difference between languages, the model fine-tuned only by the source language may not perform well for target languages. In our study, we make an interesting observation that while the top 1 result predicted by the previous approaches may often fail to hit the ground-truth answer, there are still good chances for the correct answer to be contained in the set of top k predicted results. Intuitively, the previous approaches have empowered the model certain level of capability to roughly distinguish good answers from bad ones. However, without sufficient training data, it is not powerful enough to capture the nuances between the accurate answer and those approximate ones. Based on this observation, we develop a two-stage approach to enhance the model performance. The first stage targets at recall; we design a hard-learning (HL) algorithm to maximize the likelihood that the top k predictions contain the accurate answer. The second stage focuses on precision, where an answer-aware contrastive learning (AA-CL) mechanism is developed to learn the minute difference between the accurate answer and other candidates. Extensive experiments show that our model significantly outperforms strong baselines on two cross-lingual MRC benchmark datasets.

----

## [1178] Probing Linguistic Information for Logical Inference in Pre-trained Language Models

**Authors**: *Zeming Chen, Qiyue Gao*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i10.21294](https://doi.org/10.1609/aaai.v36i10.21294)

**Abstract**:

Progress in pre-trained language models has led to a surge of impressive results on downstream tasks for natural language understanding. Recent work on probing pre-trained language models uncovered a wide range of linguistic properties encoded in their contextualized representations. However, it is unclear whether they encode semantic knowledge that is crucial to symbolic inference methods. We propose a methodology for probing knowledge for inference that logical systems require but often lack in pre-trained language model representations. Our probing datasets cover a list of key types of knowledge used by many symbolic inference systems. We find that (i) pre-trained language models do encode several types of knowledge for inference, but there are also some types of knowledge for inference that are not encoded, (ii) language models can effectively learn missing knowledge for inference through fine-tuning. Overall, our findings provide insights into which aspects of knowledge for inference language models and their pre-training procedures capture. Moreover, we have demonstrated language models' potential as semantic and background knowledge bases for supporting symbolic inference methods.

----

## [1179] On the Transferability of Pre-trained Language Models: A Study from Artificial Datasets

**Authors**: *David Cheng-Han Chiang, Hung-Yi Lee*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i10.21295](https://doi.org/10.1609/aaai.v36i10.21295)

**Abstract**:

Pre-training language models (LMs) on large-scale unlabeled text data makes the model much easier to achieve exceptional downstream performance than their counterparts directly trained on the downstream tasks. 
 In this work, we study what specific traits in the pre-training data, other than the semantics, make a pre-trained LM superior to their counterparts trained from scratch on downstream tasks.
 We propose to use artificially constructed datasets as the pre-training data to exclude the effect of semantics, and further control what characteristics the pre-training corpora have.
 By fine-tuning the pre-trained models on GLUE benchmark, we can learn how beneficial it is to transfer the knowledge from the model trained on the dataset possessing that specific trait.
 We define and discuss three different characteristics in the artificial dataset: 1) matching the token's uni-gram or bi-gram distribution between pre-training and downstream fine-tuning, 2) the presence of the explicit dependencies among the tokens in a sequence, 3) the length of the implicit dependencies among the tokens in a sequence. 
 Our experiments show that the explicit dependencies in the sequences of the pre-training data are critical to the downstream performance.
 Our results also reveal that models achieve better downstream performance when pre-trained on a dataset with a longer range of implicit dependencies.
 Based on our analysis, we find that models pre-trained with artificial datasets are prone to learn spurious correlation in downstream tasks.
 Our work reveals that even if the LMs are not pre-trained on natural language, they still gain transferability on certain human language downstream tasks once the LMs learn to model the token dependencies in the sequences. 
 This result helps us understand the exceptional transferability of pre-trained LMs.

----

## [1180] C2L: Causally Contrastive Learning for Robust Text Classification

**Authors**: *Seungtaek Choi, Myeongho Jeong, Hojae Han, Seung-won Hwang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i10.21296](https://doi.org/10.1609/aaai.v36i10.21296)

**Abstract**:

Despite the super-human accuracy of recent deep models in NLP tasks, their robustness is reportedly limited due to their reliance on spurious patterns. We thus aim to leverage contrastive learning and counterfactual augmentation for robustness. For augmentation, existing work either requires humans to add counterfactuals to the dataset or machines to automatically matches near-counterfactuals already in the dataset. Unlike existing augmentation is affected by spurious correlations, ours, by synthesizing “a set” of counterfactuals, and making a collective decision on the distribution of predictions on this set, can robustly supervise the causality of each term. Our empirical results show that our approach, by collective decisions, is less sensitive to task model bias of attribution-based synthesis, and thus achieves significant improvements, in diverse dimensions: 1) counterfactual robustness, 2) cross-domain generalization, and 3) generalization from scarce data.

----

## [1181] Novelty Controlled Paraphrase Generation with Retrieval Augmented Conditional Prompt Tuning

**Authors**: *Jishnu Ray Chowdhury, Yong Zhuang, Shuyi Wang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i10.21297](https://doi.org/10.1609/aaai.v36i10.21297)

**Abstract**:

Paraphrase generation is a fundamental and long-standing task in natural language processing. In this paper, we concentrate on two contributions to the task: (1) we propose Retrieval Augmented Prompt Tuning (RAPT) as a parameter-efficient method to adapt large pre-trained language models for paraphrase generation; (2) we propose Novelty Conditioned RAPT (NC-RAPT) as a simple model-agnostic method of using specialized prompt tokens for controlled paraphrase generation with varying levels of lexical novelty. By conducting extensive experiments on four datasets, we demonstrate the effectiveness of the proposed approaches for retaining the semantic content of the original text while inducing lexical novelty in the generation.

----

## [1182] Flexible Instance-Specific Rationalization of NLP Models

**Authors**: *George Chrysostomou, Nikolaos Aletras*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i10.21298](https://doi.org/10.1609/aaai.v36i10.21298)

**Abstract**:

Recent research on model interpretability in natural language processing extensively uses feature scoring methods for identifying which parts of the input are the most important for a model to make a prediction (i.e. explanation or rationale). However, previous research has shown that there is no clear best scoring method across various text classification tasks while practitioners typically have to make several other ad-hoc choices regarding the length and the type of the rationale (e.g. short or long, contiguous or not). Inspired by this, we propose a simple yet effective and flexible method that allows selecting optimally for each data instance: (1) a feature scoring method; (2) the length; and (3) the type of the rationale. Our method is inspired by input erasure approaches to interpretability which assume that the most faithful rationale for a prediction should be the one with the highest difference between the model's output distribution using the full text and the text after removing the rationale as input respectively. Evaluation on four standard text classification datasets shows that our proposed method provides more faithful, comprehensive and highly sufficient explanations compared to using a fixed feature scoring method, rationale length and type. More importantly, we demonstrate that a practitioner is not required to make any ad-hoc choices in order to extract faithful rationales using our approach.

----

## [1183] InfoLM: A New Metric to Evaluate Summarization & Data2Text Generation

**Authors**: *Pierre Jean A. Colombo, Chloé Clavel, Pablo Piantanida*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i10.21299](https://doi.org/10.1609/aaai.v36i10.21299)

**Abstract**:

Assessing the quality of natural language generation (NLG) systems through human annotation is very expensive. Additionally, human annotation campaigns are time-consuming and include non-reusable human labour. In practice, researchers rely on automatic metrics as a proxy of quality. In the last decade, many string-based metrics (e.g., BLEU or ROUGE) have been introduced. However, such metrics usually rely on exact matches and thus, do not robustly handle synonyms. In this paper, we introduce InfoLM a family of untrained metrics that can be viewed as a string-based metric that addresses the aforementioned flaws thanks to a pre-trained masked language model. This family of metrics also makes use of information measures allowing the possibility to adapt InfoLM to different evaluation criteria. Using direct assessment, we demonstrate that InfoLM achieves statistically significant improvement and two figure correlation gains in many configurations compared to existing metrics on both summarization and data2text generation tasks.

----

## [1184] Nice Perfume How Long Did You Marinate in It? Multimodal Sarcasm Explanation

**Authors**: *Poorav Desai, Tanmoy Chakraborty, Md. Shad Akhtar*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i10.21300](https://doi.org/10.1609/aaai.v36i10.21300)

**Abstract**:

Sarcasm is a pervading linguistic phenomenon and highly challenging to explain due to its subjectivity, lack of context and deeply-felt opinion. In the multimodal setup, sarcasm is conveyed through the incongruity between the text and visual entities. Although recent approaches deal with sarcasm as a classification problem, it is unclear why an online post is identified as sarcastic. Without proper explanation, end users may not be able to perceive the underlying sense of irony. In this paper, we propose a novel problem -- Multimodal Sarcasm Explanation (MuSE) -- given a multimodal sarcastic post containing an image and a caption, we aim to generate a natural language explanation to reveal the intended sarcasm. To this end, we develop MORE, a new dataset with explanation of 3510 sarcastic multimodal posts. Each explanation is a natural language (English) sentence describing the hidden irony. We benchmark MORE by employing a multimodal Transformer-based architecture. It incorporates a cross-modal attention in the Transformer's encoder which attends to the distinguishing features between the two modalities. Subsequently, a BART-based auto-regressive decoder is used as the generator. Empirical results demonstrate convincing results over various baselines (adopted for MuSE) across five evaluation metrics. We also conduct human evaluation on predictions and obtain Fleiss' Kappa score of 0.4 as a fair agreement among 25 evaluators.

----

## [1185] Zero-Shot Commonsense Question Answering with Cloze Translation and Consistency Optimization

**Authors**: *Zi-Yi Dou, Nanyun Peng*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i10.21301](https://doi.org/10.1609/aaai.v36i10.21301)

**Abstract**:

Commonsense question answering (CQA) aims to test if models can answer questions regarding commonsense knowledge that everyone knows. Prior works that incorporate external knowledge bases have shown promising results, but knowledge bases are expensive to construct and are often limited to a fixed set of relations. In this paper, we instead focus on better utilizing the implicit knowledge stored in pre-trained language models. While researchers have found that the knowledge embedded in pre-trained language models can be extracted by having them fill in the blanks of carefully designed prompts for relation extraction and text classification, it remains unclear if we can adopt this paradigm in CQA where the inputs and outputs take much more flexible forms. To this end, we investigate four translation methods that can translate natural questions into cloze-style sentences to better solicit commonsense knowledge from language models, including a syntactic-based model, an unsupervised neural model, and two supervised neural models. In addition, to combine the different translation methods, we propose to encourage consistency among model predictions on different translated questions with unlabeled data. We demonstrate the effectiveness of our methods on three CQA datasets in zero-shot settings. We show that our methods are complementary to a knowledge base improved model, and combining them can lead to state-of-the-art zero-shot performance. Analyses also reveal distinct characteristics of the different cloze translation methods and provide insights on why combining them can lead to great improvements. Code/dataset is available at https://github.com/PlusLabNLP/zero_shot_cqa.

----

## [1186] Synthetic Disinformation Attacks on Automated Fact Verification Systems

**Authors**: *Yibing Du, Antoine Bosselut, Christopher D. Manning*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i10.21302](https://doi.org/10.1609/aaai.v36i10.21302)

**Abstract**:

Automated fact-checking is a needed technology to curtail the spread of online misinformation. One current framework for such solutions proposes to verify claims by retrieving supporting or refuting evidence from related textual sources. However, the realistic use cases for fact-checkers will require verifying claims against evidence sources that could be affected by the same misinformation. Furthermore, the development of modern NLP tools that can produce coherent, fabricated content would allow malicious actors to systematically generate adversarial disinformation for fact-checkers.
  
In this work, we explore the sensitivity of automated fact-checkers to synthetic adversarial evidence in two simulated settings: ADVERSARIAL ADDITION, where we fabricate documents and add them to the evidence repository available to the fact-checking system, and ADVERSARIAL MODIFICATION, where existing evidence source documents in the repository are automatically altered. Our study across multiple models on three benchmarks demonstrates that these systems suffer significant performance drops against these attacks. Finally, we discuss the growing threat of modern NLG systems as generators of disinformation in the context of the challenges they pose to automated fact-checkers.

----

## [1187] Regularizing End-to-End Speech Translation with Triangular Decomposition Agreement

**Authors**: *Yichao Du, Zhirui Zhang, Weizhi Wang, Boxing Chen, Jun Xie, Tong Xu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i10.21303](https://doi.org/10.1609/aaai.v36i10.21303)

**Abstract**:

End-to-end speech-to-text translation (E2E-ST) is becoming increasingly popular due to the potential of its less error propagation, lower latency, and fewer parameters. Given the triplet training corpus〈speech, transcription, translation〉, the conventional high-quality E2E-ST system leverages the〈speech, transcription〉pair to pre-train the model and then utilizes the〈speech, translation〉pair to optimize it further. However, this process only involves two-tuple data at each stage, and this loose coupling fails to fully exploit the association between triplet data. In this paper, we attempt to model the joint probability of transcription and translation based on the speech input to directly leverage such triplet data. Based on that, we propose a novel regularization method for model training to improve the agreement of dual-path decomposition within triplet data, which should be equal in theory. To achieve this goal, we introduce two Kullback-Leibler divergence regularization terms into the model training objective to reduce the mismatch between output probabilities of dual-path. Then the well-trained model can be naturally transformed as the E2E-ST models by a pre-defined early stop tag. Experiments on the MuST-C benchmark demonstrate that our proposed approach significantly outperforms state-of-the-art E2E-ST baselines on all 8 language pairs while achieving better performance in the automatic speech recognition task.

----

## [1188] Play the Shannon Game with Language Models: A Human-Free Approach to Summary Evaluation

**Authors**: *Nicholas Egan, Oleg V. Vasilyev, John Bohannon*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i10.21304](https://doi.org/10.1609/aaai.v36i10.21304)

**Abstract**:

The goal of a summary is to concisely state the most important information in a document. With this principle in mind, we introduce new reference-free summary evaluation metrics that use a pretrained language model to estimate the information content shared between a document and its summary. These metrics are a modern take on the Shannon Game, a method for summary quality scoring proposed decades ago, where we replace human annotators with language models. We also view these metrics as an extension of BLANC, a recently proposed approach to summary quality measurement based on the performance of a language model with and without the help of a summary. Using transformer based language models, we empirically verify that our metrics achieve state-of-the-art correlation with human judgement of the summary quality dimensions of both coherence and relevance, as well as competitive correlation with human judgement of consistency and fluency.

----

## [1189] Fortunately, Discourse Markers Can Enhance Language Models for Sentiment Analysis

**Authors**: *Liat Ein-Dor, Ilya Shnayderman, Artem Spector, Lena Dankin, Ranit Aharonov, Noam Slonim*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i10.21305](https://doi.org/10.1609/aaai.v36i10.21305)

**Abstract**:

In recent years, pretrained language models have revolutionized the NLP world, while achieving state of the art performance in various downstream tasks. However, in many cases, these models do not perform well when labeled data is scarce and the model is expected to perform in the zero or few shot setting. Recently, several works have shown that continual pretraining or performing a second phase of pretraining (inter-training) which is better aligned with the downstream task, can lead to improved results, especially in the scarce data setting. Here, we propose to leverage sentiment-carrying discourse markers to generate large-scale weakly-labeled data, which in turn can be used to adapt language models for sentiment analysis. Extensive experimental results show the value of our approach on various benchmark datasets, including the finance domain. Code, models and data are available at https://github.com/ibm/tslm-discourse-markers.

----

## [1190] Retrieve, Caption, Generate: Visual Grounding for Enhancing Commonsense in Text Generation Models

**Authors**: *Steven Y. Feng, Kevin Lu, Zhuofu Tao, Malihe Alikhani, Teruko Mitamura, Eduard H. Hovy, Varun Gangal*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i10.21306](https://doi.org/10.1609/aaai.v36i10.21306)

**Abstract**:

We investigate the use of multimodal information contained in images as an effective method for enhancing the commonsense of Transformer models for text generation. We perform experiments using BART and T5 on concept-to-text generation, specifically the task of generative commonsense reasoning, or CommonGen. We call our approach VisCTG: Visually Grounded Concept-to-Text Generation. VisCTG involves captioning images representing appropriate everyday scenarios, and using these captions to enrich and steer the generation process. Comprehensive evaluation and analysis demonstrate that VisCTG noticeably improves model performance while successfully addressing several issues of the baseline generations, including poor commonsense, fluency, and specificity.

----

## [1191] Language Model Priming for Cross-Lingual Event Extraction

**Authors**: *Steven Fincke, Shantanu Agarwal, Scott Miller, Elizabeth Boschee*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i10.21307](https://doi.org/10.1609/aaai.v36i10.21307)

**Abstract**:

We present a novel, language-agnostic approach to "priming" language models for the task of event extraction, providing particularly effective performance in low-resource and zero-shot cross-lingual settings. With priming, we augment the input to the transformer stack's language model differently depending on the question(s) being asked of the model at runtime. For instance, if the model is being asked to identify arguments for the trigger "protested", we will provide that trigger as part of the input to the language model, allowing it to produce different representations for candidate arguments than when it is asked about arguments for the trigger "arrest" elsewhere in the same sentence. We show that by enabling the language model to better compensate for the deficits of sparse and noisy training data, our approach improves both trigger and argument detection and classification significantly over the state of the art in a zero-shot cross-lingual setting.

----

## [1192] Language Modelling via Learning to Rank

**Authors**: *Arvid Frydenlund, Gagandeep Singh, Frank Rudzicz*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i10.21308](https://doi.org/10.1609/aaai.v36i10.21308)

**Abstract**:

We consider language modelling (LM) as a multi-label structured prediction task by re-framing training from solely predicting a single ground-truth word to ranking a set of words which could continue a given context. To avoid annotating top-k ranks, we generate them using pre-trained LMs: GPT-2, BERT, and Born-Again models. This leads to a rank-based form of knowledge distillation (KD). We also develop a method using N-grams to create a non-probabilistic teacher which generates the ranks without the need of a pre-trained LM.
 
 We confirm the hypotheses: that we can treat LMing as a ranking task and that we can do so without the use of a pre-trained LM.
 We show that rank-based KD generally gives a modest improvement to perplexity (PPL) -- though often with statistical significance -- when compared to Kullback–Leibler-based KD. Surprisingly, given the naivety of the method, the N-grams act as competitive teachers and achieve similar performance as using either BERT or a Born-Again model teachers. Unsurprisingly, GPT-2 always acts as the best teacher. 
 Using it and a Transformer-XL student on Wiki-02, rank-based KD reduces a cross-entropy baseline from 65.27 to 55.94 and against a KL-based KD of 56.70.

----

## [1193] NAREOR: The Narrative Reordering Problem

**Authors**: *Varun Gangal, Steven Y. Feng, Malihe Alikhani, Teruko Mitamura, Eduard H. Hovy*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i10.21309](https://doi.org/10.1609/aaai.v36i10.21309)

**Abstract**:

Many implicit inferences exist in text depending on how it is structured that can critically impact the text's interpretation and meaning. One such structural aspect present in text with chronology is the order of its presentation. For narratives or stories, this is known as the narrative order. Reordering a narrative can impact the temporal, causal, event-based, and other inferences readers draw from it, which in turn can have strong effects both on its interpretation and interestingness. In this paper, we propose and investigate the task of Narrative Reordering (NAREOR) which involves rewriting a given story in a different narrative order while preserving its plot. We present a dataset, NAREORC, with human rewritings of stories within ROCStories in non-linear orders, and conduct a detailed analysis of it. Further, we propose novel task-specific training methods with suitable evaluation metrics. We perform experiments on NAREORC using state-of-the-art models such as BART and T5 and conduct extensive automatic and human evaluations. We demonstrate that although our models can perform decently, NAREOR is a challenging task with potential for further exploration. We also investigate two applications of NAREOR: generation of more interesting variations of stories and serving as adversarial sets for temporal/event-related tasks, besides discussing other prospective ones, such as for pedagogical setups related to language skills like essay writing and applications to medicine involving clinical narratives.

----

## [1194] UNISON: Unpaired Cross-Lingual Image Captioning

**Authors**: *Jiahui Gao, Yi Zhou, Philip L. H. Yu, Shafiq R. Joty, Jiuxiang Gu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i10.21310](https://doi.org/10.1609/aaai.v36i10.21310)

**Abstract**:

Image captioning has emerged as an interesting research field in recent years due to its broad application scenarios. The traditional paradigm of image captioning relies on paired image-caption datasets to train the model in a supervised manner. However, creating such paired datasets for every target language is prohibitively expensive, which hinders the extensibility of captioning technology and deprives a large part of the world population of its benefit. In this work, we present a novel unpaired cross-lingual method to generate image captions without relying on any caption corpus in the source or the target language. Specifically, our method consists of two phases: (1) a cross-lingual auto-encoding process, which utilizing a sentence parallel (bitext) corpus to learn the mapping from the source to the target language in the scene graph encoding space and decode sentences in the target language, and (2) a cross-modal unsupervised feature mapping, which seeks to map the encoded scene graph features from image modality to language modality. We verify the effectiveness of our proposed method on the Chinese image caption generation task. The comparisons against several existing methods demonstrate the effectiveness of our approach.

----

## [1195] AutoBERT-Zero: Evolving BERT Backbone from Scratch

**Authors**: *Jiahui Gao, Hang Xu, Han Shi, Xiaozhe Ren, Philip L. H. Yu, Xiaodan Liang, Xin Jiang, Zhenguo Li*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i10.21311](https://doi.org/10.1609/aaai.v36i10.21311)

**Abstract**:

Transformer-based pre-trained language models like BERT and its variants have recently achieved promising performance in various natural language processing (NLP) tasks. However, the conventional paradigm constructs the backbone by purely stacking the manually designed global self-attention layers, introducing inductive bias and thus leads to sub-optimal. In this work, we make the first attempt to automatically discover novel pre-trained language model (PLM) backbone on a flexible search space containing the most fundamental operations from scratch. Specifically, we propose a well-designed search space which (i) contains primitive math operations in the intra-layer level to explore novel attention structures, and (ii) leverages convolution blocks to be the supplementary for attentions in the inter-layer level to better learn local dependency. To enhance the efficiency for finding promising architectures, we propose an Operation-Priority Neural Architecture Search (OP-NAS) algorithm, which optimizes both the search algorithm and evaluation of candidate models. Specifically, we propose Operation-Priority (OP) evolution strategy to facilitate model search via balancing exploration and exploitation. Furthermore, we design a Bi-branch Weight-Sharing (BIWS) training strategy for fast model evaluation. Extensive experiments show that the searched architecture (named AutoBERT-Zero) significantly outperforms BERT and its variants of different model capacities in various downstream tasks, proving the architecture's transfer and scaling abilities. Remarkably, AutoBERT-Zero-base outperforms RoBERTa-base (using much more data) and BERT-large (with much larger model size) by 2.4 and 1.4 higher score on GLUE test set.

----

## [1196] ISEEQ: Information Seeking Question Generation Using Dynamic Meta-Information Retrieval and Knowledge Graphs

**Authors**: *Manas Gaur, Kalpa Gunaratna, Vijay Srinivasan, Hongxia Jin*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i10.21312](https://doi.org/10.1609/aaai.v36i10.21312)

**Abstract**:

Conversational Information Seeking (CIS) is a relatively new research area within conversational AI that attempts to seek information from end-users in order to understand and satisfy the users' needs. If realized, such a CIS system has far-reaching benefits in the real world; for example, CIS systems can assist clinicians in pre-screening or triaging patients in healthcare. A key open sub-problem in CIS that remains unaddressed in the literature is generating Information Seeking Questions (ISQs) based on a short initial query from the end-user. To address this open problem, we propose Information SEEking Question generator (ISEEQ), a novel approach for generating ISQs from just a short user query, given a large text corpus relevant to the user query. Firstly, ISEEQ uses a knowledge graph to enrich the user query. Secondly, ISEEQ uses the knowledge-enriched query to retrieve relevant context passages to ask coherent ISQs adhering to a conceptual flow. Thirdly, ISEEQ introduces a new deep generative-adversarial reinforcement learning-based approach for generating ISQs. We show that ISEEQ can generate high-quality ISQs to promote the development of CIS agents. ISEEQ significantly outperforms comparable baselines on five ISQ evaluation metrics across four datasets having user queries from diverse domains. Further, we argue that ISEEQ is transferable across domains for generating ISQs, as it shows the acceptable performance when trained and tested on different pairs of domains. A qualitative human evaluation confirms that ISEEQ generated ISQs are comparable in quality to human-generated questions, and it outperformed the best comparable baseline.

----

## [1197] Explainable Metaphor Identification Inspired by Conceptual Metaphor Theory

**Authors**: *Mengshi Ge, Rui Mao, Erik Cambria*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i10.21313](https://doi.org/10.1609/aaai.v36i10.21313)

**Abstract**:

Metaphor is not only a linguistic phenomenon but also reflects the concept projection between source and target domains in human cognition. Previous sequence tagging-based metaphor identification methods could not model the concept projection, resulting in a limitation that the outputs of these models are unexplainable in the predictions of the metaphoricity labels. In this work, we propose the first explainable metaphor identification model, inspired by Conceptual Metaphor Theory. The model is based on statistic learning, a lexical resource, and a novel reward mechanism. Our model can identify the metaphoricity on the word-pair level, and explain the predicted metaphoricity labels via learned concept mappings. The use of the reward mechanism allows the model to learn the optimal concept mappings without knowing their true labels. Our method is also applicable for the concepts that are out of training domains by using the lexical resource. The automatically generated concept mappings demonstrate the implicit human thoughts in metaphoric expressions. Our experiments show the effectiveness of the proposed model in metaphor identification, and concept mapping tasks, respectively.

----

## [1198] Confidence Calibration for Intent Detection via Hyperspherical Space and Rebalanced Accuracy-Uncertainty Loss

**Authors**: *Yantao Gong, Cao Liu, Fan Yang, Xunliang Cai, Guanglu Wan, Jiansong Chen, Weipeng Zhang, Houfeng Wang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i10.21314](https://doi.org/10.1609/aaai.v36i10.21314)

**Abstract**:

Data-driven methods have achieved notable performance on intent detection, which is a task to comprehend user queries. Nonetheless, they are controversial for over-confident predictions. In some scenarios, users do not only care about the accuracy but also the confidence of model. Unfortunately, mainstream neural networks are poorly calibrated, with a large gap between accuracy and confidence. To handle this problem defined as confidence calibration, we propose a model using the hyperspherical space and rebalanced accuracy-uncertainty loss. Specifically, we project the label vector onto hyperspherical space uniformly to generate a dense label representation matrix, which mitigates over-confident predictions due to overfitting sparse one-hot label matrix. Besides, we rebalance samples of different accuracy and uncertainty to better guide model training. Experiments on the open datasets verify that our model outperforms the existing calibration methods and achieves a significant improvement on the calibration metric.

----

## [1199] SSAST: Self-Supervised Audio Spectrogram Transformer

**Authors**: *Yuan Gong, Cheng-I Lai, Yu-An Chung, James R. Glass*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i10.21315](https://doi.org/10.1609/aaai.v36i10.21315)

**Abstract**:

Recently, neural networks based purely on self-attention, such as the Vision Transformer (ViT), have been shown to outperform deep learning models constructed with convolutional neural networks (CNNs) on various vision tasks, thus extending the success of Transformers, which were originally developed for language processing, to the vision domain. A recent study showed that a similar methodology can also be applied to the audio domain. Specifically, the Audio Spectrogram Transformer (AST) achieves state-of-the-art results on various audio classification benchmarks. However, pure Transformer models tend to require more training data compared to CNNs, and the success of the AST relies on supervised pretraining that requires a large amount of labeled data and a complex training pipeline, thus limiting the practical usage of AST. This paper focuses on audio and speech classification, and aims to reduce the need for large amounts of labeled data for the AST by leveraging self-supervised learning using unlabeled data. Specifically, we propose to pretrain the AST model with joint discriminative and generative masked spectrogram patch modeling (MSPM) using unlabeled audio from AudioSet and Librispeech. We evaluate our pretrained models on both audio and speech classification tasks including audio event classification, keyword spotting, emotion recognition, and speaker identification. The proposed self-supervised framework significantly boosts AST performance on all tasks, with an average improvement of 60.9%, leading to similar or even better results than a supervised pretrained AST. To the best of our knowledge, it is the first patch-based self-supervised learning framework in the audio and speech domain, and also the first self-supervised learning framework for AST.

----



[Go to the previous page](AAAI-2022-list05.md)

[Go to the next page](AAAI-2022-list07.md)

[Go to the catalog section](README.md)