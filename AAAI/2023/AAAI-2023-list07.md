## [1200] Semidefinite Programming versus Burer-Monteiro Factorization for Matrix Sensing

**Authors**: *Baturalp Yalçin, Ziye Ma, Javad Lavaei, Somayeh Sojoudi*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i9.26270](https://doi.org/10.1609/aaai.v37i9.26270)

**Abstract**:

Many fundamental low-rank optimization problems, such as matrix completion, phase retrieval, and robust PCA, can be formulated as the matrix sensing problem. Two main approaches for solving matrix sensing are based on semidefinite programming (SDP) and Burer-Monteiro (B-M) factorization. The former suffers from high computational and space complexities, whereas the latter may return a spurious solution due to the non-convexity of the problem. The existing theoretical guarantees for the success of these methods have led to similar conservative conditions, which may wrongly imply that these methods have comparable performances. In this paper, we shed light on some major differences between these two methods. First, we present a class of structured matrix completion problems for which the B-M methods fail with an overwhelming probability, while the SDP method works correctly. Second, we identify a class of highly sparse matrix completion problems for which the B-M method works and the SDP method fails. Third, we prove that although the B-M method exhibits the same performance independent of the rank of the unknown solution, the success of the SDP method is correlated to the rank of the solution and improves as the rank increases. Unlike the existing literature that has mainly focused on those instances of matrix sensing for which both SDP and B-M work, this paper offers the first result on the unique merit of each method over the alternative approach.

----

## [1201] DeFL: Defending against Model Poisoning Attacks in Federated Learning via Critical Learning Periods Awareness

**Authors**: *Gang Yan, Hao Wang, Xu Yuan, Jian Li*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i9.26271](https://doi.org/10.1609/aaai.v37i9.26271)

**Abstract**:

Federated learning (FL) is known to be susceptible to model poisoning attacks in which malicious clients hamper the accuracy of the global model by sending manipulated model updates to the central server during the FL training process.  Existing defenses mainly focus on Byzantine-robust FL aggregations, and largely ignore the impact of the underlying deep neural network (DNN) that is used to FL training.  Inspired by recent findings on critical learning periods (CLP) in DNNs, where small gradient errors have irrecoverable impact on the final model accuracy, we propose a new defense, called a CLP-aware defense against poisoning of FL (DeFL).  The key idea of DeFL is to measure fine-grained differences between DNN model updates via an easy-to-compute federated gradient norm vector (FGNV) metric.  Using FGNV, DeFL simultaneously detects malicious clients and identifies CLP, which in turn is leveraged to guide the adaptive removal of detected malicious clients from aggregation.  As a result, DeFL not only mitigates model poisoning attacks on the global model but also is robust to detection errors.  Our extensive experiments on three benchmark datasets demonstrate that DeFL produces significant performance gain over conventional defenses against state-of-the-art model poisoning attacks.

----

## [1202] T2G-FORMER: Organizing Tabular Features into Relation Graphs Promotes Heterogeneous Feature Interaction

**Authors**: *Jiahuan Yan, Jintai Chen, Yixuan Wu, Danny Z. Chen, Jian Wu*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i9.26272](https://doi.org/10.1609/aaai.v37i9.26272)

**Abstract**:

Recent development of deep neural networks (DNNs) for tabular learning has largely benefited from the capability of DNNs for automatic feature interaction. However, the heterogeneity nature of tabular features makes such features relatively independent, and developing effective methods to promote tabular feature interaction still remains an open problem. In this paper, we propose a novel Graph Estimator, which automatically estimates the relations among tabular features and builds graphs by assigning edges between related features. Such relation graphs organize independent tabular features into a kind of graph data such that interaction of nodes (tabular features) can be conducted in an orderly fashion. Based on our proposed Graph Estimator, we present a bespoke Transformer network tailored for tabular learning, called T2G-Former, which processes tabular data by performing tabular feature interaction guided by the relation graphs. A specific Cross-level Readout collects salient features predicted by the layers in T2G-Former across different levels, and attains global semantics for final prediction. Comprehensive experiments show that our T2G-Former achieves superior performance among DNNs and is competitive with non-deep Gradient Boosted Decision Tree models. The code and detailed results are available at https://github.com/jyansir/t2g-former.

----

## [1203] Computably Continuous Reinforcement-Learning Objectives Are PAC-Learnable

**Authors**: *Cambridge Yang, Michael Littman, Michael Carbin*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i9.26273](https://doi.org/10.1609/aaai.v37i9.26273)

**Abstract**:

In reinforcement learning, the classic objectives of maximizing discounted and finite-horizon cumulative rewards are PAC-learnable: There are algorithms that learn a near-optimal policy with high probability using a finite amount of samples and computation. 
In recent years, researchers have introduced objectives and corresponding reinforcement-learning algorithms beyond the classic cumulative rewards, such as objectives specified as linear temporal logic formulas. 
However, questions about the PAC-learnability of these new objectives have remained open.


This work demonstrates the PAC-learnability of general reinforcement-learning objectives through sufficient conditions for PAC-learnability in two analysis settings. 
In particular, for the analysis that considers only sample complexity, we prove that if an objective given as an oracle is uniformly continuous, then it is PAC-learnable.
Further, for the analysis that considers computational complexity, we prove that if an objective is computable, then it is PAC-learnable. 
In other words, if a procedure computes successive approximations of the objective's value, then the objective is PAC-learnable.


We give three applications of our condition on objectives from the literature with previously unknown PAC-learnability and prove that these objectives are PAC-learnable.  
Overall, our result helps verify existing objectives' PAC-learnability. 
Also, as some studied objectives that are not uniformly continuous have been shown to be not PAC-learnable, our results could guide the design of new PAC-learnable objectives.

----

## [1204] Reinforcement Causal Structure Learning on Order Graph

**Authors**: *Dezhi Yang, Guoxian Yu, Jun Wang, Zhengtian Wu, Maozu Guo*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i9.26274](https://doi.org/10.1609/aaai.v37i9.26274)

**Abstract**:

Learning directed acyclic graph (DAG) that  describes the causality of observed data is a very challenging but important task. Due to the limited quantity and quality of observed data, and non-identifiability of causal graph, it is almost impossible to infer a single precise DAG. Some methods approximate the posterior distribution of DAGs to explore the DAG space via Markov chain Monte Carlo (MCMC), but the DAG space is over the nature of super-exponential growth, accurately characterizing the whole distribution over DAGs is very intractable. In this paper, we propose Reinforcement Causal Structure Learning on Order Graph (RCL-OG) that uses order graph instead of MCMC to model different DAG topological orderings and to reduce the problem size. RCL-OG first defines reinforcement learning with a new reward mechanism to approximate the posterior distribution of orderings in an efficacy way, and uses deep Q-learning to update and transfer rewards between nodes. Next, it obtains the probability transition model of nodes on order graph, and computes the posterior probability of different orderings. In this way, we can sample on this model to obtain the ordering with high probability. Experiments on synthetic and benchmark datasets show that RCL-OG provides accurate posterior probability approximation and achieves better results than competitive causal discovery algorithms.

----

## [1205] AdaTask: A Task-Aware Adaptive Learning Rate Approach to Multi-Task Learning

**Authors**: *Enneng Yang, Junwei Pan, Ximei Wang, Haibin Yu, Li Shen, Xihua Chen, Lei Xiao, Jie Jiang, Guibing Guo*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i9.26275](https://doi.org/10.1609/aaai.v37i9.26275)

**Abstract**:

Multi-task learning (MTL) models have demonstrated impressive results in computer vision, natural language processing, and recommender systems. Even though many approaches have been proposed, how well these approaches balance different tasks on each parameter still remains unclear. In this paper, we propose to measure the task dominance degree of a parameter by the total updates of each task on this parameter. Specifically, we compute the total updates by the exponentially decaying Average of the squared Updates (AU) on a parameter from the corresponding task. Based on this novel metric, we observe that many parameters in existing MTL methods, especially those in the higher shared layers, are still dominated by one or several tasks. The dominance of AU is mainly due to the dominance of accumulative gradients from one or several tasks. Motivated by this, we propose a Task-wise Adaptive learning rate approach, AdaTask in short, to separate the accumulative gradients and hence the learning rate of each task for each parameter in adaptive learning rate approaches (e.g., AdaGrad, RMSProp, and Adam). Comprehensive experiments on computer vision and recommender system MTL datasets demonstrate that AdaTask significantly improves the performance of dominated tasks, resulting SOTA average task-wise performance.  Analysis on both synthetic and real-world datasets shows AdaTask  balance parameters in every shared layer well.

----

## [1206] WaveForM: Graph Enhanced Wavelet Learning for Long Sequence Forecasting of Multivariate Time Series

**Authors**: *Fuhao Yang, Xin Li, Min Wang, Hongyu Zang, Wei Pang, Mingzhong Wang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i9.26276](https://doi.org/10.1609/aaai.v37i9.26276)

**Abstract**:

Multivariate time series (MTS) analysis and forecasting are crucial in many real-world applications, such as smart traffic management and weather forecasting. However, most existing work either focuses on short sequence forecasting or makes predictions predominantly with time domain features, which is not effective at removing noises with irregular frequencies in MTS. Therefore, we propose WaveForM, an end-to-end graph enhanced Wavelet learning framework for long sequence FORecasting of MTS. WaveForM first utilizes Discrete Wavelet Transform (DWT) to represent MTS in the wavelet domain, which captures both frequency and time domain features with a sound theoretical basis. To enable the effective learning in the wavelet domain, we further propose a graph constructor, which learns a global graph to represent the relationships between MTS variables, and graph-enhanced prediction modules, which utilize dilated convolution and graph convolution to capture the correlations between time series and predict the wavelet coefficients at different levels. Extensive experiments on five real-world forecasting datasets show that our model can achieve considerable performance improvement over different prediction lengths against the most competitive baseline of each dataset.

----

## [1207] Layout Generation as Intermediate Action Sequence Prediction

**Authors**: *Huiting Yang, Danqing Huang, Chin-Yew Lin, Shengfeng He*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i9.26277](https://doi.org/10.1609/aaai.v37i9.26277)

**Abstract**:

Layout generation plays a crucial role in graphic design intelligence. One important characteristic of the graphic layouts is that they usually follow certain design principles. For example, the principle of repetition emphasizes the reuse of similar visual elements throughout the design. To generate a layout, previous works mainly attempt at predicting the absolute value of bounding box for each element, where such target representation has hidden the information of higher-order design operations like repetition (e.g. copy the size of the previously generated element). In this paper, we introduce a novel action schema to encode these operations for better modeling the generation process. Instead of predicting the bounding box values, our approach autoregressively outputs the intermediate action sequence, which can then be deterministically converted to the final layout. We achieve state-of-the-art performances on three datasets. Both automatic and human evaluations show that our approach generates high-quality and diverse layouts. Furthermore, we revisit the commonly used evaluation metric FID adapted in this task, and observe that previous works use different settings to train the feature extractor for obtaining real/generated data distribution, which leads to inconsistent conclusions. We conduct an in-depth analysis on this metric and settle for a more robust and reliable evaluation setting. Code is available at this website.

----

## [1208] Learning-Assisted Algorithm Unrolling for Online Optimization with Budget Constraints

**Authors**: *Jianyi Yang, Shaolei Ren*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i9.26278](https://doi.org/10.1609/aaai.v37i9.26278)

**Abstract**:

Online optimization with multiple budget constraints is challenging since the online decisions over a short time horizon are coupled together by strict inventory constraints. The existing manually-designed algorithms cannot achieve satisfactory average performance for this setting because they often need a large number of time steps for convergence and/or may violate the inventory constraints. In this paper, we propose a new machine learning (ML) assisted unrolling approach, called LAAU (Learning-Assisted Algorithm Unrolling), which unrolls the agent’s online decision pipeline and leverages an ML model for updating the Lagrangian multiplier online. For efficient training via backpropagation, we derive gradients of the decision pipeline over time. We also provide the average cost bounds for two cases when training data is available offline and collected online, respectively. Finally, we present numerical results to highlight that LAAU can outperform the existing baselines.

----

## [1209] ADEPT: A DEbiasing PrompT Framework

**Authors**: *Ke Yang, Charles Yu, Yi Ren Fung, Manling Li, Heng Ji*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i9.26279](https://doi.org/10.1609/aaai.v37i9.26279)

**Abstract**:

Several works have proven that finetuning is an applicable approach for debiasing contextualized word embeddings. Similarly, discrete prompts with semantic meanings have shown to be effective in debiasing tasks. With unfixed mathematical representation at the token level, continuous prompts usually surpass discrete ones at providing a pre-trained language model (PLM) with additional task-specific information. Despite this, relatively few efforts have been made to debias PLMs by prompt tuning with continuous prompts compared to its discrete counterpart. Furthermore, for most debiasing methods that alter a PLM's original parameters, a major problem is the need to not only decrease the bias in the PLM but also to ensure that the PLM does not lose its representation ability. Finetuning methods typically have a hard time maintaining this balance, as they tend to violently remove meanings of attribute words (like the words developing our concepts of "male" and "female" for gender), which also leads to an unstable and unpredictable training process. In this paper, we propose ADEPT, a method to debias PLMs using prompt tuning while maintaining the delicate balance between removing biases and ensuring representation ability. To achieve this, we propose a new training criterion inspired by manifold learning and equip it with an explicit debiasing term to optimize prompt tuning. In addition, we conduct several experiments with regard to the reliability, quality, and quantity of a previously proposed attribute training corpus in order to obtain a clearer prototype of a certain attribute, which indicates the attribute's position and relative distances to other words on the manifold. We evaluate ADEPT on several widely acknowledged debiasing benchmarks and downstream tasks, and find that it achieves competitive results while maintaining (and in some cases even improving) the PLM's representation ability. We further visualize words' correlation before and after debiasing a PLM, and give some possible explanations for the visible effects.

----

## [1210] Generalized Semantic Segmentation by Self-Supervised Source Domain Projection and Multi-Level Contrastive Learning

**Authors**: *Liwei Yang, Xiang Gu, Jian Sun*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i9.26280](https://doi.org/10.1609/aaai.v37i9.26280)

**Abstract**:

Deep networks trained on the source domain show degraded performance when tested on unseen target domain data. To enhance the model's generalization ability, most existing domain generalization methods learn domain invariant features by suppressing domain sensitive features. Different from them, we propose a Domain Projection and Contrastive Learning (DPCL) approach for generalized semantic segmentation, which includes two modules: Self-supervised Source Domain Projection (SSDP) and Multi-Level Contrastive Learning (MLCL). SSDP aims to reduce domain gap by projecting data to the source domain, while MLCL is a learning scheme to learn discriminative and generalizable features on the projected data. During test time, we first project the target data by SSDP to mitigate domain shift, then generate the segmentation results by the learned segmentation network based on MLCL. At test time, we can update the projected data by minimizing our proposed pixel-to-pixel contrastive loss to obtain better results. Extensive experiments for semantic segmentation demonstrate the favorable generalization capability of our method on benchmark datasets.

----

## [1211] CEM: Constrained Entropy Maximization for Task-Agnostic Safe Exploration

**Authors**: *Qisong Yang, Matthijs T. J. Spaan*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i9.26281](https://doi.org/10.1609/aaai.v37i9.26281)

**Abstract**:

In the absence of assigned tasks, a learning agent typically seeks to explore its environment efficiently. However, the pursuit of exploration will bring more safety risks.
An under-explored aspect of reinforcement learning is how to achieve safe efficient exploration when the task is unknown.
In this paper, we propose a practical Constrained Entropy Maximization (CEM) algorithm to solve task-agnostic safe exploration problems, which naturally require a finite horizon and undiscounted constraints on safety costs.
The CEM algorithm aims to learn a policy that maximizes state entropy under the premise of safety.
To avoid approximating the state density in complex domains, CEM leverages a k-nearest neighbor entropy estimator to evaluate the efficiency of exploration.
In terms of safety, CEM minimizes the safety costs, and adaptively trades off safety and exploration based on the current constraint satisfaction. The empirical analysis shows that CEM enables the acquisition of a safe exploration policy in complex environments, resulting in improved performance in both safety and sample efficiency for target tasks.

----

## [1212] Understanding Representation Learnability of Nonlinear Self-Supervised Learning

**Authors**: *Ruofeng Yang, Xiangyuan Li, Bo Jiang, Shuai Li*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i9.26282](https://doi.org/10.1609/aaai.v37i9.26282)

**Abstract**:

Self-supervised learning (SSL) has empirically shown its data representation learnability in many downstream tasks. There are only a few theoretical works on data representation learnability, and many of those focus on final data representation, treating the nonlinear neural network as a ``black box". However, the accurate learning results of neural networks are crucial for describing the data distribution features learned by SSL models. Our paper is the first to analyze the learning results of the nonlinear SSL model accurately. We consider a toy data distribution that contains two features: the label-related feature and the hidden feature. Unlike previous linear setting work that depends on closed-form solutions, we use the gradient descent algorithm to train a 1-layer nonlinear SSL model with a certain initialization region and prove that the model converges to a local minimum. Furthermore, different from the complex iterative analysis, we propose a new analysis process which uses the exact version of  Inverse Function Theorem to accurately describe the features learned by the local minimum. With this local minimum, we prove that the nonlinear SSL model can capture the label-related feature and hidden feature at the same time. In contrast, the nonlinear supervised learning (SL) model can only learn the label-related feature. We also present the learning processes and results of the nonlinear SSL and SL model via simulation experiments.

----

## [1213] Simple and Efficient Heterogeneous Graph Neural Network

**Authors**: *Xiaocheng Yang, Mingyu Yan, Shirui Pan, Xiaochun Ye, Dongrui Fan*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i9.26283](https://doi.org/10.1609/aaai.v37i9.26283)

**Abstract**:

Heterogeneous graph neural networks (HGNNs) have the powerful capability to embed rich structural and semantic information of a heterogeneous graph into node representations. Existing HGNNs inherit many mechanisms from graph neural networks (GNNs) designed for homogeneous graphs, especially the attention mechanism and the multi-layer structure. These mechanisms bring excessive complexity, but seldom work studies whether they are really effective on heterogeneous graphs. In this paper, we conduct an in-depth and detailed study of these mechanisms and propose the Simple and Efficient Heterogeneous Graph Neural Network (SeHGNN). To easily capture structural information, SeHGNN pre-computes the neighbor aggregation using a light-weight mean aggregator, which reduces complexity by removing overused neighbor attention and avoiding repeated neighbor aggregation in every training epoch. To better utilize semantic information, SeHGNN adopts the single-layer structure with long metapaths to extend the receptive field, as well as a transformer-based semantic fusion module to fuse features from different metapaths. As a result, SeHGNN exhibits the characteristics of a simple network structure, high prediction accuracy, and fast training speed. Extensive experiments on five real-world heterogeneous graphs demonstrate the superiority of SeHGNN over the state-of-the-arts on both accuracy and training speed.

----

## [1214] T-distributed Spherical Feature Representation for Imbalanced Classification

**Authors**: *Xiaoyu Yang, Yufei Chen, Xiaodong Yue, Shaoxun Xu, Chao Ma*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i9.26284](https://doi.org/10.1609/aaai.v37i9.26284)

**Abstract**:

Real-world classification tasks often show an extremely imbalanced problem. The extreme imbalance will cause a strong bias that the decision boundary of the classifier is completely dominated by the categories with abundant samples, which are also called the head categories. Current methods have alleviated the imbalanced impact from mainly three aspects: class re-balance, decoupling and domain adaptation. However, the existing criterion with the winner-take-all strategy still leads to the crowding problem in the eigenspace. The head categories with many samples can extract features more accurately, but occupy most of the eigenspace. The tail categories sharing the rest of the narrow eigenspace are too crowded together to accurately extract features. Above these issues, we propose a novel T-distributed spherical metric for equalized eigenspace in the imbalanced classification, which has the following innovations: 1) We design the T-distributed spherical metric, which has the characteristics of high kurtosis. Instead of the winner-take-all strategy, the T-distributed spherical metric produces a high logit only when the extracted feature is close enough to the category center, without a strong bias against other categories. 2) The T-distributed spherical metric is integrated into the classifier, which is able to equalize the eigenspace for alleviating the crowding issue in the imbalanced problem. The equalized eigenspace by the T-distributed spherical classifier is capable of improving the accuracy of the tail categories while maintaining the accuracy of the head, which significantly promotes the intraclass compactness and interclass separability of features. Extensive experiments on large-scale imbalanced datasets verify our method, which shows superior results in the long-tailed CIFAR-100/-10 with the imbalanced ratio IR = 100/50. Our method also achieves excellent results on the large-scale ImageNet-LT dataset and the iNaturalist dataset with various backbones. In addition, we provide a case study of the real clinical classification of pancreatic tumor subtypes with 6 categories. Among them, the largest number of PDAC accounts for 315 cases, and the least CP has only 8 cases. After 4-fold cross-validation, we achieved a top-1 accuracy of 69.04%.

----

## [1215] Cluster-Guided Contrastive Graph Clustering Network

**Authors**: *Xihong Yang, Yue Liu, Sihang Zhou, Siwei Wang, Wenxuan Tu, Qun Zheng, Xinwang Liu, Liming Fang, En Zhu*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i9.26285](https://doi.org/10.1609/aaai.v37i9.26285)

**Abstract**:

Benefiting from the intrinsic supervision information exploitation capability, contrastive learning has achieved promising performance in the field of deep graph clustering recently. However, we observe that two drawbacks of the positive and negative sample construction mechanisms limit the performance of existing algorithms from further improvement. 1) The quality of positive samples heavily depends on the carefully designed data augmentations, while inappropriate data augmentations would easily lead to the semantic drift and indiscriminative positive samples. 2) The constructed negative samples are not reliable for ignoring important clustering information. To solve these problems, we propose a Cluster-guided Contrastive deep Graph Clustering network (CCGC) by mining the intrinsic supervision information in the high-confidence clustering results. Specifically, instead of conducting complex node or edge perturbation, we construct two views of the graph by designing special Siamese encoders whose weights are not shared between the sibling sub-networks. Then, guided by the high-confidence clustering information, we carefully select and construct the positive samples from the same high-confidence cluster in two views. Moreover, to construct semantic meaningful negative sample pairs, we regard the centers of different high-confidence clusters as negative samples, thus improving the discriminative capability and reliability of the constructed sample pairs. Lastly, we design an objective function to pull close the samples from the same cluster while pushing away those from other clusters by maximizing and minimizing the cross-view cosine similarity between positive and negative samples. Extensive experimental results on six datasets demonstrate the effectiveness of CCGC compared with the existing state-of-the-art algorithms. The code of CCGC is available at https://github.com/xihongyang1999/CCGC on Github.

----

## [1216] Flow to Control: Offline Reinforcement Learning with Lossless Primitive Discovery

**Authors**: *Yiqin Yang, Hao Hu, Wenzhe Li, Siyuan Li, Jun Yang, Qianchuan Zhao, Chongjie Zhang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i9.26286](https://doi.org/10.1609/aaai.v37i9.26286)

**Abstract**:

Offline reinforcement learning (RL) enables the agent to effectively learn from logged data, which significantly extends the applicability of RL algorithms in real-world scenarios where exploration can be expensive or unsafe. Previous works have shown that extracting primitive skills from the recurring and temporally extended structures in the logged data yields better learning. However, these methods suffer greatly when the primitives have limited representation ability to recover the original policy space, especially in offline settings. In this paper, we give a quantitative characterization of the performance of offline hierarchical learning and highlight the importance of learning lossless primitives. To this end, we propose to use a flow-based structure as the representation for low-level policies. This allows us to represent the behaviors in the dataset faithfully while keeping the expression ability to recover the whole policy space. We show that such lossless primitives can drastically improve the performance of hierarchical policies. The experimental results and extensive ablation studies on the standard D4RL benchmark show that our method has a good representation ability for policies and achieves superior performance in most tasks.

----

## [1217] Prototypical Partial Optimal Transport for Universal Domain Adaptation

**Authors**: *Yucheng Yang, Xiang Gu, Jian Sun*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i9.26287](https://doi.org/10.1609/aaai.v37i9.26287)

**Abstract**:

Universal domain adaptation (UniDA) aims to transfer knowledge from a labeled source domain to an unlabeled target domain without requiring the same label sets of both domains. The existence of domain and category shift makes the task challenging and requires us to distinguish “known” samples (i.e., samples whose labels exist in both domains) and “unknown” samples (i.e., samples whose labels exist in only one domain) in both domains before reducing the domain gap. In this paper, we consider the problem from the point of view of distribution matching which we only need to align two distributions partially. A novel approach, dubbed mini-batch Prototypical Partial Optimal Transport (m-PPOT), is proposed to conduct partial distribution alignment for UniDA. In training phase, besides minimizing m-PPOT, we also leverage the transport plan of m-PPOT to reweight source prototypes and target samples, and design reweighted entropy loss and reweighted cross-entropy loss to distinguish “known” and “unknown” samples. Experiments on four benchmarks show that our method outperforms the previous state-of-the-art UniDA methods.

----

## [1218] DeCOM: Decomposed Policy for Constrained Cooperative Multi-Agent Reinforcement Learning

**Authors**: *Zhaoxing Yang, Haiming Jin, Rong Ding, Haoyi You, Guiyun Fan, Xinbing Wang, Chenghu Zhou*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i9.26288](https://doi.org/10.1609/aaai.v37i9.26288)

**Abstract**:

In recent years, multi-agent reinforcement learning (MARL) has presented impressive performance in various applications. However, physical limitations, budget restrictions, and many other factors usually impose constraints on a multi-agent system (MAS), which cannot be handled by traditional MARL frameworks. Specifically, this paper focuses on constrained MASes where agents work cooperatively to maximize the expected team-average return under various constraints on expected team-average costs, and develops a constrained cooperative MARL framework, named DeCOM, for such MASes. In particular, DeCOM decomposes the policy of each agent into two modules, which empowers information sharing among agents to achieve better cooperation. In addition, with such modularization, the training algorithm of DeCOM separates the original constrained optimization into an unconstrained optimization on reward and a constraints satisfaction problem on costs. DeCOM then iteratively solves these problems in a computationally efficient manner, which makes DeCOM highly scalable. We also provide theoretical guarantees on the convergence of DeCOM's policy update algorithm. Finally, we conduct extensive experiments to show the effectiveness of DeCOM with various types of costs in both moderate-scale and large-scale (with 500 agents) environments that originate from real-world applications.

----

## [1219] Purifier: Defending Data Inference Attacks via Transforming Confidence Scores

**Authors**: *Ziqi Yang, Lijin Wang, Da Yang, Jie Wan, Ziming Zhao, Ee-Chien Chang, Fan Zhang, Kui Ren*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i9.26289](https://doi.org/10.1609/aaai.v37i9.26289)

**Abstract**:

Neural networks are susceptible to data inference attacks such as the membership inference attack, the adversarial model inversion attack and the attribute inference attack, where the attacker could infer useful information such as the membership, the reconstruction or the sensitive attributes of a data sample from the confidence scores predicted by the target classifier. In this paper, we propose a method, namely PURIFIER, to defend against membership inference attacks. It transforms the confidence score vectors predicted by the target classifier and makes purified confidence scores indistinguishable in individual shape, statistical distribution and prediction label between members and non-members. The experimental results show that PURIFIER helps defend membership inference attacks with high effectiveness and efficiency, outperforming previous defense methods, and also incurs negligible utility loss. Besides, our further experiments show that PURIFIER is also effective in defending adversarial model inversion attacks and attribute inference attacks. For example, the inversion error is raised about 4+ times on the Facescrub530 classifier, and the attribute inference accuracy drops significantly when PURIFIER is deployed in our experiment.

----

## [1220] i-Code: An Integrative and Composable Multimodal Learning Framework

**Authors**: *Ziyi Yang, Yuwei Fang, Chenguang Zhu, Reid Pryzant, Dongdong Chen, Yu Shi, Yichong Xu, Yao Qian, Mei Gao, Yi-Ling Chen, Liyang Lu, Yujia Xie, Robert Gmyr, Noel Codella, Naoyuki Kanda, Bin Xiao, Lu Yuan, Takuya Yoshioka, Michael Zeng, Xuedong Huang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i9.26290](https://doi.org/10.1609/aaai.v37i9.26290)

**Abstract**:

Human intelligence is multimodal; we integrate visual, linguistic, and acoustic signals to maintain a holistic worldview. Most current pretraining methods, however, are limited to one or two modalities. We present i-Code, a self-supervised pretraining framework where users may flexibly combine the modalities of vision, speech, and language into unified and general-purpose vector representations. In this framework, data from each modality are first given to pretrained single-modality encoders. The encoder outputs are then integrated with a multimodal fusion network, which uses novel merge- and co-attention mechanisms to effectively combine information from the different modalities. The entire system is pretrained end-to-end with new objectives including masked modality unit modeling and cross-modality contrastive learning. Unlike previous research using only video for pretraining, the i-Code framework can dynamically process single, dual, and triple-modality data during training and inference, flexibly projecting different combinations of modalities into a single representation space. Experimental results demonstrate how i-Code can outperform state-of-the-art techniques on five multimodal understanding tasks and single-modality benchmarks, improving by as much as 11% and demonstrating the power of integrative multimodal pretraining.

----

## [1221] Learning Dynamic Latent Spaces for Lifelong Generative Modelling

**Authors**: *Fei Ye, Adrian G. Bors*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i9.26291](https://doi.org/10.1609/aaai.v37i9.26291)

**Abstract**:

Task Free Continual Learning (TFCL) aims to capture novel concepts from non-stationary data streams without forgetting previously learned knowledge. Mixture models, which add new components when certain conditions are met, have shown promising results in TFCL tasks. However, such approaches do not make use of the knowledge already accumulated for positive knowledge transfer. In this paper, we develop a new model, namely the Online Recursive Variational Autoencoder (ORVAE). ORVAE utilizes the prior knowledge by selectively incorporating the newly learnt information, by adding new components, according to the knowledge already known from the past learnt data. We introduce a new attention mechanism to regularize the structural latent space in which the most important information is reused while the information that interferes with novel samples is inactivated. The proposed attention mechanism can maximize the benefit from the forward transfer for learning novel information without forgetting previously learnt knowledge. We perform several experiments which show that ORVAE achieves state-of-the-art results under TFCL.

----

## [1222] Lifelong Compression Mixture Model via Knowledge Relationship Graph

**Authors**: *Fei Ye, Adrian G. Bors*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i9.26292](https://doi.org/10.1609/aaai.v37i9.26292)

**Abstract**:

Task-Free Continual Learning (TFCL) represents a challenging scenario for lifelong learning because the model, under this paradigm, does not access any task information. The Dynamic Expansion Model (DEM) has shown promising results in this scenario due to its scalability and generalisation power. However, DEM focuses only on addressing forgetting and ignores minimizing  the model size, which limits its deployment in practical systems. In this work, we aim to simultaneously address network forgetting and model size optimization by developing the Lifelong Compression Mixture Model (LGMM) equipped with the Maximum Mean Discrepancy (MMD) based expansion criterion for model expansion. A diversity-aware sample selection approach is proposed to selectively store a variety of samples to promote information diversity among the components of the LGMM, which allows more knowledge to be captured with an appropriate model size. In order to avoid having multiple components with similar knowledge in the LGMM, we propose a data-free component discarding mechanism that evaluates a knowledge relation graph matrix describing the relevance between each pair of components. A greedy selection procedure is proposed to identify and remove the redundant    components from the LGMM. The proposed discarding mechanism can be performed during or after the training. Experiments on different datasets show that LGMM achieves the best performance for TFCL.

----

## [1223] Lifelong Variational Autoencoder via Online Adversarial Expansion Strategy

**Authors**: *Fei Ye, Adrian G. Bors*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i9.26293](https://doi.org/10.1609/aaai.v37i9.26293)

**Abstract**:

The Variational Autoencoder (VAE) suffers from a significant loss of information when trained  on a non-stationary data distribution. This loss in VAE models, called catastrophic forgetting, has not been studied theoretically before. We analyse the forgetting behaviour of a VAE in continual generative modelling by developing a new lower bound on the data likelihood, which interprets the forgetting process as an increase in the probability distance between the generator's distribution and the evolved data distribution. The proposed bound shows that a VAE-based dynamic expansion model can achieve better performance if its capacity increases appropriately considering the shift in the data distribution. Based on this analysis, we propose a novel expansion criterion that aims to preserve the information diversity among the VAE components, while ensuring that it acquires more knowledge with fewer parameters. Specifically, we implement this expansion criterion from the perspective of a multi-player game and propose the Online Adversarial Expansion Strategy (OAES), which considers all previously learned components as well as the currently updated component as multiple players in a game, while an adversary model evaluates their performance. The proposed OAES can dynamically estimate the discrepancy between each player and the adversary without accessing task information. This leads to the gradual addition of new components while ensuring the knowledge diversity among all of them. We show theoretically and empirically that the proposed extension strategy can enable a VAE model to achieve the best performance given an appropriate model size.

----

## [1224] Continual Variational Autoencoder via Continual Generative Knowledge Distillation

**Authors**: *Fei Ye, Adrian G. Bors*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i9.26294](https://doi.org/10.1609/aaai.v37i9.26294)

**Abstract**:

Humans and other living beings have the ability of short and long-term memorization during their entire lifespan. However, most existing Continual Learning (CL) methods can only account for short-term information when training on infinite streams of data. In this paper, we develop a new unsupervised continual learning framework consisting of two memory systems using Variational Autoencoders (VAEs). We develop a Short-Term Memory (STM), and a parameterised scalable memory implemented by a Teacher model aiming to preserve the long-term information. To incrementally enrich the Teacher's knowledge during training, we propose the Knowledge Incremental Assimilation Mechanism (KIAM), which evaluates the knowledge similarity between the STM and the already accumulated information as signals to expand the Teacher's capacity. Then we train a VAE as a Student module and propose a new Knowledge Distillation (KD) approach that gradually transfers generative knowledge from the Teacher to the Student module. To ensure the quality and diversity of knowledge in KD, we propose a new expert pruning approach that selectively removes the Teacher's redundant parameters, associated with unnecessary experts which have learnt overlapping information with other experts. This mechanism further reduces the complexity of the Teacher's module while ensuring the diversity of knowledge for the KD procedure. We show theoretically and empirically that the proposed framework can train a statistically diversified Teacher module for continual VAE learning which is applicable to learning infinite data streams.

----

## [1225] Certifiable Out-of-Distribution Generalization

**Authors**: *Nanyang Ye, Lin Zhu, Jia Wang, Zhaoyu Zeng, Jiayao Shao, Chensheng Peng, Bikang Pan, Kaican Li, Jun Zhu*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i9.26295](https://doi.org/10.1609/aaai.v37i9.26295)

**Abstract**:

Machine learning methods suffer from test-time performance degeneration when faced with out-of-distribution (OoD) data whose distribution is not necessarily the same as training data distribution. Although a plethora of algorithms have been proposed to mitigate this issue, it has been demonstrated that achieving better performance than ERM simultaneously on different types of distributional shift datasets is challenging for existing approaches. Besides, it is unknown how and to what extent these methods work on any OoD datum without theoretical guarantees. In this paper, we propose a certifiable out-of-distribution generalization method that provides provable OoD generalization performance guarantees via a functional optimization framework leveraging random distributions and max-margin learning for each input datum. With this approach, the proposed algorithmic scheme can provide certified accuracy for each input datum's prediction on the semantic space and achieves better performance simultaneously on OoD datasets dominated by correlation shifts or diversity shifts. Our code is available at https://github.com/ZlatanWilliams/StochasticDisturbanceLearning.

----

## [1226] Random Walk Conformer: Learning Graph Representation from Long and Short Range

**Authors**: *Pei-Kai Yeh, Hsi-Wen Chen, Ming-Syan Chen*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i9.26296](https://doi.org/10.1609/aaai.v37i9.26296)

**Abstract**:

While graph neural networks (GNNs) have achieved notable success in various graph mining tasks, conventional GNNs only model the pairwise correlation in 1-hop neighbors without considering the long-term relations and the high-order patterns, thus limiting their performances. Recently, several works have addressed these issues by exploring the motif, i.e., frequent subgraphs. However, these methods usually require an unacceptable computational time to enumerate all possible combinations of motifs. In this paper, we introduce a new GNN framework, namely Random Walk Conformer (RWC), to exploit global correlations and local patterns based on the random walk, which is a promising method to discover the graph structure. Besides, we propose random walk encoding to help RWC capture topological information, which is proven more expressive than conventional spatial encoding. Extensive experiment results manifest that RWC achieves state-of-the-art performance on graph classification and regression tasks. The source code of RWC is available at https://github.com/b05901024/RandomWalkConformer.

----

## [1227] Lottery Pools: Winning More by Interpolating Tickets without Increasing Training or Inference Cost

**Authors**: *Lu Yin, Shiwei Liu, Meng Fang, Tianjin Huang, Vlado Menkovski, Mykola Pechenizkiy*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i9.26297](https://doi.org/10.1609/aaai.v37i9.26297)

**Abstract**:

Lottery tickets (LTs) is able to discover accurate and sparse subnetworks that could be trained in isolation to match the performance of dense networks. Ensemble, in parallel, is one of the oldest time-proven tricks in machine learning to improve performance by combining the output of multiple independent models. However, the benefits of ensemble in the context of LTs will be diluted since ensemble does not directly lead to stronger sparse subnetworks, but leverages their predictions for a better decision. In this work, we first observe that directly averaging the weights of the adjacent learned subnetworks significantly boosts the performance of LTs. Encouraged by this observation, we further propose an alternative way to perform an "ensemble'' over the subnetworks identified by iterative magnitude pruning via a simple interpolating strategy. We call our method Lottery Pools. In contrast to the naive ensemble which brings no performance gains to each single subnetwork, Lottery Pools yields much stronger sparse subnetworks than the original LTs without requiring any extra training or inference cost. Across various modern architectures on CIFAR-10/100 and ImageNet, we show that our method achieves significant performance gains in both, in-distribution and out-of-distribution scenarios. Impressively, evaluated with VGG-16 and ResNet-18, the produced sparse subnetworks outperform the original LTs by up to 1.88% on CIFAR-100 and 2.36% on CIFAR-100-C; the resulting dense network surpasses the pre-trained dense-model up to 
 2.22% on CIFAR-100 and 2.38% on CIFAR-100-C. Our source code can be found at https://github.com/luuyin/Lottery-pools.

----

## [1228] GOHSP: A Unified Framework of Graph and Optimization-Based Heterogeneous Structured Pruning for Vision Transformer

**Authors**: *Miao Yin, Burak Uzkent, Yilin Shen, Hongxia Jin, Bo Yuan*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i9.26298](https://doi.org/10.1609/aaai.v37i9.26298)

**Abstract**:

The recently proposed Vision transformers (ViTs) have shown
very impressive empirical performance in various computer vision tasks,
and they are viewed as an important type of foundation model. However, ViTs are typically constructed with large-scale sizes, which then
severely hinder their potential deployment in many practical resources constrained applications. 
To mitigate this challenging problem, structured pruning is a promising solution to compress model size and enable
practical efficiency. However, unlike its current popularity for CNNs and
RNNs, structured pruning for ViT models is little explored.
In this paper, we propose GOHSP, a unified framework of Graph and
Optimization-based Structured Pruning for ViT models. We first develop
a graph-based ranking for measuring the importance of attention heads,
and the extracted importance information is further integrated to an
optimization-based procedure to impose the heterogeneous structured
sparsity patterns on the ViT models. Experimental results show that
our proposed GOHSP demonstrates excellent compression performance.
On CIFAR-10 dataset, our approach can bring 40% parameters reduction
with no accuracy loss for ViT-Small model. On ImageNet dataset, with
30% and 35% sparsity ratio for DeiT-Tiny and DeiT-Small models, our
approach achieves 1.65% and 0.76% accuracy increase over the existing
structured pruning methods, respectively.

----

## [1229] Policy-Based Primal-Dual Methods for Convex Constrained Markov Decision Processes

**Authors**: *Donghao Ying, Mengzi Amy Guo, Yuhao Ding, Javad Lavaei, Zuo-Jun Max Shen*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i9.26299](https://doi.org/10.1609/aaai.v37i9.26299)

**Abstract**:

We study convex Constrained Markov Decision Processes (CMDPs) in which the objective is concave and the constraints are convex in the state-action occupancy measure. We propose a policy-based primal-dual algorithm that updates the primal variable via policy gradient ascent and updates the dual variable via projected sub-gradient descent. Despite the loss of additivity structure and the nonconvex nature, we establish the global convergence of the proposed algorithm by leveraging a hidden convexity in the problem, and prove the O(T^-1/3) convergence rate in terms of both optimality gap and constraint violation. When the objective is strongly concave in the occupancy measure, we prove an improved convergence rate of O(T^-1/2). By introducing a pessimistic term to the constraint, we further show that a zero constraint violation can be achieved while preserving the same convergence rate for the optimality gap. This work is the first one in the literature that establishes non-asymptotic convergence guarantees for policy-based primal-dual methods for solving infinite-horizon discounted convex CMDPs.

----

## [1230] Priori Anchor Labels Supervised Scalable Multi-View Bipartite Graph Clustering

**Authors**: *Jiali You, Zhenwen Ren, Xiaojian You, Haoran Li, Yuancheng Yao*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i9.26300](https://doi.org/10.1609/aaai.v37i9.26300)

**Abstract**:

Although multi-view clustering (MVC) has achieved remarkable performance by integrating the complementary information of views, it is inefficient when facing scalable data. Proverbially, anchor strategy can mitigate such a challenge a certain extent. However, the unsupervised dynamic strategy usually cannot obtain the optimal anchors for MVC. The main reasons are that it does not consider the fairness of different views and lacks the priori supervised guidance. To completely solve these problems, we first propose the priori anchor graph regularization (PAGG) for scalable multi-view bipartite graph clustering, dubbed as SMGC method. Specifically, SMGC learns a few representative consensus anchors to simulate the numerous view data well, and constructs a bipartite graph to bridge the affinities between the anchors and original data points. In order to largely improve the quality of anchors, PAGG predefines prior anchor labels to constrain the anchors with discriminative cluster structure and fair view allocation, such that a better bipartite graph can be obtained for fast clustering. Experimentally, abundant of experiments are accomplished on six scalable benchmark datasets, and the experimental results fully demonstrate the effectiveness and efficiency of our SMGC.

----

## [1231] STARS: Spatial-Temporal Active Re-sampling for Label-Efficient Learning from Noisy Annotations

**Authors**: *Dayou Yu, Weishi Shi, Qi Yu*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i9.26301](https://doi.org/10.1609/aaai.v37i9.26301)

**Abstract**:

Active learning (AL) aims to sample the most informative data instances for labeling, which makes the model fitting data efficient while significantly reducing the annotation cost. However, most existing AL models make a strong assumption that the annotated data instances are always assigned correct labels, which may not hold true in many practical settings.  In this paper, we develop a theoretical framework to formally analyze the impact of noisy annotations and show that systematically re-sampling guarantees to reduce the noise rate, which can lead to improved generalization capability. More importantly, the theoretical framework demonstrates the key benefit of conducting active re-sampling on label-efficient learning, which is critical for AL. The theoretical results also suggest essential properties of an active re-sampling function with a fast convergence speed and guaranteed error reduction. This inspires us to design a novel spatial-temporal active re-sampling function by leveraging the important spatial and temporal properties of maximum-margin classifiers. Extensive experiments conducted on both synthetic and real-world data clearly demonstrate the effectiveness of the proposed active re-sampling function.

----

## [1232] Boosted Dynamic Neural Networks

**Authors**: *Haichao Yu, Haoxiang Li, Gang Hua, Gao Huang, Humphrey Shi*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i9.26302](https://doi.org/10.1609/aaai.v37i9.26302)

**Abstract**:

Early-exiting dynamic neural networks (EDNN), as one type of dynamic neural networks, has been widely studied recently. A typical EDNN has multiple prediction heads at different layers of the network backbone. During inference, the model will exit at either the last prediction head or an intermediate prediction head where the prediction confidence is higher than a predefined threshold. To optimize the model, these prediction heads together with the network backbone are trained on every batch of training data. This brings a train-test mismatch problem that all the prediction heads are optimized on all types of data in training phase while the deeper heads will only see difficult inputs in testing phase. Treating training and testing inputs differently at the two phases will cause the mismatch between training and testing data distributions. To mitigate this problem, we formulate an EDNN as an additive model inspired by gradient boosting, and propose multiple training techniques to optimize the model effectively. We name our method BoostNet. Our experiments show it achieves the state-of-the-art performance on CIFAR100 and ImageNet datasets in both anytime and budgeted-batch prediction modes. Our code is released at https://github.com/SHI-Labs/Boosted-Dynamic-Networks.

----

## [1233] Stable Learning via Sparse Variable Independence

**Authors**: *Han Yu, Peng Cui, Yue He, Zheyan Shen, Yong Lin, Renzhe Xu, Xingxuan Zhang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i9.26303](https://doi.org/10.1609/aaai.v37i9.26303)

**Abstract**:

The problem of covariate-shift generalization has attracted intensive research attention. Previous stable learning algorithms employ sample reweighting schemes to decorrelate the covariates when there is no explicit domain information about training data. However, with finite samples, it is difficult to achieve the desirable weights that ensure perfect independence to get rid of the unstable variables. Besides, decorrelating within stable variables may bring about high variance of learned models because of the over-reduced effective sample size. A tremendous sample size is required for these algorithms to work. In this paper, with theoretical justification, we propose SVI (Sparse Variable Independence) for the covariate-shift generalization problem. We introduce sparsity constraint to compensate for the imperfectness of sample reweighting under the finite-sample setting in previous methods. Furthermore, we organically combine independence-based sample reweighting and sparsity-based variable selection in an iterative way to avoid decorrelating within stable variables, increasing the effective sample size to alleviate variance inflation. Experiments on both synthetic and real-world datasets demonstrate the improvement of covariate-shift generalization performance brought by SVI.

----

## [1234] Compressing Transformers: Features Are Low-Rank, but Weights Are Not!

**Authors**: *Hao Yu, Jianxin Wu*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i9.26304](https://doi.org/10.1609/aaai.v37i9.26304)

**Abstract**:

Transformer and its variants achieve excellent results in various computer vision and natural language processing tasks,  but high computational costs and reliance on large training datasets restrict their deployment in resource-constrained settings. Low-rank approximation of model weights has been effective in compressing CNN models, but its application to transformers has been less explored and is less effective. Existing methods require the complete dataset to fine-tune compressed models, which are both time-consuming and data-hungry. This paper reveals that the features (i.e., activations) are low-rank, but model weights are surprisingly not low-rank. Hence, AAFM is proposed, which adaptively determines the compressed model structure and locally compresses each linear layer's output features rather than the model weights. A second stage, GFM, optimizes the entire compressed network holistically. Both AAFM and GFM only use few training samples without labels, that is, they are few-shot, unsupervised, fast and effective. For example, with only 2K images without labels, 33% of the parameters are removed in DeiT-B with 18.8% relative throughput increase, but only a 0.23% accuracy loss for ImageNet recognition. The proposed methods are successfully applied to the language modeling task in NLP, too. Besides, the few-shot compressed models generalize well in downstream tasks.

----

## [1235] Offline Imitation Learning with Suboptimal Demonstrations via Relaxed Distribution Matching

**Authors**: *Lantao Yu, Tianhe Yu, Jiaming Song, Willie Neiswanger, Stefano Ermon*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i9.26305](https://doi.org/10.1609/aaai.v37i9.26305)

**Abstract**:

Offline imitation learning (IL) promises the ability to learn performant policies from pre-collected demonstrations without interactions with the environment. However, imitating behaviors fully offline typically requires numerous expert data. To tackle this issue, we study the setting where we have limited expert data and supplementary suboptimal data. In this case, a well-known issue is the distribution shift between the learned policy and the behavior policy that collects the offline data. Prior works mitigate this issue by regularizing the KL divergence between the stationary state-action distributions of the learned policy and the behavior policy. We argue that such constraints based on exact distribution matching can be overly conservative and hamper policy learning, especially when the imperfect offline data is highly suboptimal. To resolve this issue, we present RelaxDICE, which employs an asymmetrically-relaxed f-divergence for explicit support regularization. Specifically, instead of driving the learned policy to exactly match the behavior policy, we impose little penalty whenever the density ratio between their stationary state-action distributions is upper bounded by a constant. Note that such formulation leads to a nested min-max optimization problem, which causes instability in practice. RelaxDICE addresses this challenge by supporting a closed-form solution for the inner maximization problem. Extensive empirical study shows that our method significantly outperforms the best prior offline IL method in six standard continuous control environments with over 30% performance gain on average, across 22 settings where the imperfect dataset is highly suboptimal.

----

## [1236] High-Level Semantic Feature Matters Few-Shot Unsupervised Domain Adaptation

**Authors**: *Lei Yu, Wanqi Yang, Shengqi Huang, Lei Wang, Ming Yang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i9.26306](https://doi.org/10.1609/aaai.v37i9.26306)

**Abstract**:

In few-shot unsupervised domain adaptation (FS-UDA), most existing methods followed the few-shot learning (FSL) methods to leverage the low-level local features (learned from conventional convolutional models, e.g., ResNet) for classification. However, the goal of FS-UDA and FSL are relevant yet distinct, since FS-UDA aims to classify the samples in target domain rather than source domain. We found that the local features are insufficient to FS-UDA, which could introduce noise or bias against classification, and not be used to effectively align the domains. To address the above issues, we aim to refine the local features to be more discriminative and relevant to classification. Thus, we propose a novel task-specific semantic feature learning method (TSECS) for FS-UDA. TSECS learns high-level semantic features for image-to-class similarity measurement. Based on the high-level features, we design a cross-domain self-training strategy to leverage the few labeled samples in source domain to build the classifier in target domain. In addition, we minimize the KL divergence of the high-level feature distributions between source and target domains to shorten the distance of the samples between the two domains. Extensive experiments on DomainNet show that the proposed method significantly outperforms SOTA methods in FS-UDA by a large margin (i.e., ~10%).

----

## [1237] Coordinate Descent Methods for DC Minimization: Optimality Conditions and Global Convergence

**Authors**: *Ganzhao Yuan*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i9.26307](https://doi.org/10.1609/aaai.v37i9.26307)

**Abstract**:

Difference-of-Convex (DC) minimization, referring to the problem of minimizing the difference of two convex functions, has been found rich applications in statistical learning and studied extensively for decades. However, existing methods are primarily based on multi-stage convex relaxation, only leading to weak optimality of critical points. This paper proposes a coordinate descent method for minimizing a class of DC functions based on sequential nonconvex approximation. Our approach iteratively solves a nonconvex one-dimensional subproblem globally, and it is guaranteed to converge to a coordinate-wise stationary point. We prove that this new optimality condition is always stronger than the standard critical point condition and directional point condition under a mildlocally bounded nonconvexity assumption. For comparisons, we also include a naive variant of coordinate descent methods based on sequential convex approximation in our study. When the objective function satisfies a globally bounded nonconvexity assumption and Luo-Tseng error bound assumption, coordinate descent methods achieve Q-linear convergence rate. Also, for many applications of interest, we show that the nonconvex one-dimensional subproblem can be computed exactly and efficiently using a breakpoint searching method. Finally, we have conducted extensive experiments on several statistical learning tasks to show the superiority of our approach.

----

## [1238] CEMA - Cost-Efficient Machine-Assisted Document Annotations

**Authors**: *Guowen Yuan, Ben Kao, Tien-Hsuan Wu*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i9.26308](https://doi.org/10.1609/aaai.v37i9.26308)

**Abstract**:

We study the problem of semantically annotating textual documents that are complex in the sense that the documents are long, feature rich, and domain specific. Due to their complexity, such annotation tasks require trained human workers, which are very expensive in both time and money. We propose CEMA, a method for deploying machine learning to assist humans in complex document annotation. CEMA estimates the human cost of annotating each document and selects the set of documents to be annotated that strike the best balance between model accuracy and human cost. We conduct experiments on complex annotation tasks in which we compare CEMA against other document selection and annotation strategies. Our results show that CEMA is the most cost-efficient solution for those tasks.

----

## [1239] Joint Multimodal Entity-Relation Extraction Based on Edge-Enhanced Graph Alignment Network and Word-Pair Relation Tagging

**Authors**: *Li Yuan, Yi Cai, Jin Wang, Qing Li*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i9.26309](https://doi.org/10.1609/aaai.v37i9.26309)

**Abstract**:

Multimodal named entity recognition (MNER) and multimodal relation extraction (MRE) are two fundamental subtasks in the multimodal knowledge graph construction task. However, the existing methods usually handle two tasks independently, which ignores the bidirectional interaction between them. This paper is the first to propose jointly performing MNER and MRE as a joint multimodal entity-relation extraction (JMERE) task .
Besides, the current MNER and MRE models only consider aligning the visual objects with textual entities in visual and textual graphs but ignore the entity-entity relationships and object-object relationships. To address the above challenges, we propose an edge-enhanced graph alignment network and a word-pair relation tagging (EEGA) for the JMERE task. Specifically, we first design a word-pair relation tagging to exploit the bidirectional interaction between MNER and MRE and avoid error propagation. Then, we propose an edge-enhanced graph alignment network to enhance the JMERE task by aligning nodes and edges in the cross-graph. Compared with previous methods, the proposed method can leverage the edge information to auxiliary alignment between objects and entities and find the correlations between entity-entity relationships and object-object relationships. Experiments are conducted to show the effectiveness of our model.

----

## [1240] ODE-RSSM: Learning Stochastic Recurrent State Space Model from Irregularly Sampled Data

**Authors**: *Zhaolin Yuan, Xiaojuan Ban, Zixuan Zhang, Xiaorui Li, Hong-Ning Dai*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i9.26310](https://doi.org/10.1609/aaai.v37i9.26310)

**Abstract**:

For the complicated input-output systems with nonlinearity and stochasticity, Deep State Space Models (SSMs) are effective for identifying systems in the latent state space, which are of great significance for representation, forecasting, and planning in online scenarios. However, most SSMs are designed for discrete-time sequences and inapplicable when the observations are irregular in time. To solve the problem, we propose a novel continuous-time SSM named Ordinary Differential Equation Recurrent State Space Model (ODE-RSSM). ODE-RSSM incorporates an ordinary differential equation (ODE) network (ODE-Net) to model the continuous-time evolution of latent states between adjacent time points. Inspired from the equivalent linear transformation on integration limits, we propose an efficient reparameterization method for solving batched ODEs with non-uniform time spans in parallel for efficiently training the ODE-RSSM with irregularly sampled sequences. We also conduct extensive experiments to evaluate the proposed ODE-RSSM and the baselines on three input-output datasets, one of which is a rollout of a private industrial dataset with strong long-term delay and stochasticity. The results demonstrate that the ODE-RSSM achieves better performance than other baselines in open loop prediction even if the time spans of predicted points are uneven and the distribution of length is changeable. Code is availiable at https://github.com/yuanzhaolin/ODE-RSSM.

----

## [1241] Value-Consistent Representation Learning for Data-Efficient Reinforcement Learning

**Authors**: *Yang Yue, Bingyi Kang, Zhongwen Xu, Gao Huang, Shuicheng Yan*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i9.26311](https://doi.org/10.1609/aaai.v37i9.26311)

**Abstract**:

Deep reinforcement learning (RL) algorithms suffer severe performance degradation when the interaction data is scarce, which limits their real-world application. Recently, visual representation learning has been shown to be effective and promising for boosting sample efficiency in RL. These methods usually rely on contrastive learning and data augmentation to train a transition model, which is different from how the model is used in RL---performing value-based planning.  Accordingly, the learned representation by these visual methods may be good for recognition but not optimal for estimating state value and solving the decision problem. To address this issue, we propose a novel method, called value-consistent representation learning (VCR), to learn representations that are directly related to decision-making. More specifically, VCR trains a model to predict the future state (also referred to as the "imagined state'') based on the current one and a sequence of actions. Instead of aligning this imagined state with a real state returned by the environment, VCR applies a Q value head on both of the states and obtains two distributions of action values. Then a distance is computed and minimized to force the imagined state to produce a similar action value prediction as that by the real state. We develop two implementations of the above idea for the discrete and continuous action spaces  respectively. We conduct experiments on Atari 100k and DeepMind Control Suite benchmarks to validate their effectiveness for improving sample efficiency. It has been demonstrated that our methods achieve new state-of-the-art performance for search-free RL algorithms.

----

## [1242] Learning Conflict-Noticed Architecture for Multi-Task Learning

**Authors**: *Zhixiong Yue, Yu Zhang, Jie Liang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i9.26312](https://doi.org/10.1609/aaai.v37i9.26312)

**Abstract**:

Multi-task learning has been widely used in many applications to enable more efficient learning by sharing part of the architecture across multiple tasks. However, a major challenge is the gradient conflict when optimizing the shared parameters, where the gradients of different tasks could have opposite directions. Directly averaging those gradients will impair the performance of some tasks and cause negative transfer. Different from most existing works that manipulate gradients to mitigate the gradient conflict, in this paper, we address this problem from the perspective of architecture learning and propose a Conflict-Noticed Architecture Learning (CoNAL) method to alleviate the gradient conflict by learning architectures. By introducing purely-specific modules specific to each task in the search space, the CoNAL method can automatically learn when to switch to purely-specific modules in the tree-structured network architectures when the gradient conflict occurs. To handle multi-task problems with a large number of tasks, we propose a progressive extension of the CoNAL method. Extensive experiments on computer vision, natural language processing, and reinforcement learning benchmarks demonstrate the effectiveness of the proposed methods.

----

## [1243] Quantum Multi-Agent Meta Reinforcement Learning

**Authors**: *Won Joon Yun, Jihong Park, Joongheon Kim*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i9.26313](https://doi.org/10.1609/aaai.v37i9.26313)

**Abstract**:

Although quantum supremacy is yet to come, there has recently been an increasing interest in identifying the potential of quantum machine learning (QML) in the looming era of practical quantum computing. Motivated by this, in this article we re-design multi-agent reinforcement learning (MARL) based on the unique characteristics of quantum neural networks (QNNs) having two separate dimensions of trainable parameters: angle parameters affecting the output qubit states, and pole parameters associated with the output measurement basis. Exploiting this dyadic trainability as meta-learning capability, we propose quantum meta MARL (QM2ARL) that first applies angle training for meta-QNN learning, followed by pole training for few-shot or local-QNN training. To avoid overfitting, we develop an angle-to-pole regularization technique injecting noise into the pole domain during angle training. Furthermore, by exploiting the pole as the memory address of each trained QNN, we introduce the concept of pole memory allowing one to save and load trained QNNs using only two-parameter pole values. We theoretically prove the convergence of angle training under the angle-to-pole regularization, and by simulation corroborate the effectiveness of QM2ARL in achieving high reward and fast convergence, as well as of the pole memory in fast adaptation to a time-varying environment.

----

## [1244] Linking Sketch Patches by Learning Synonymous Proximity for Graphic Sketch Representation

**Authors**: *Sicong Zang, Shikui Tu, Lei Xu*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i9.26314](https://doi.org/10.1609/aaai.v37i9.26314)

**Abstract**:

Graphic sketch representations are effective for representing sketches. Existing methods take the patches cropped from sketches as the graph nodes, and construct the edges based on sketch's drawing order or Euclidean distances on the canvas. However, the drawing order of a sketch may not be unique, while the patches from semantically related parts of a sketch may be far away from each other on the canvas. In this paper, we propose an order-invariant, semantics-aware method for graphic sketch representations. The cropped sketch patches are linked according to their global semantics or local geometric shapes, namely the synonymous proximity, by computing the cosine similarity between the captured patch embeddings. Such constructed edges are learnable to adapt to the variation of sketch drawings, which enable the message passing among synonymous patches. Aggregating the messages from synonymous patches by graph convolutional networks plays a role of denoising, which is beneficial to produce robust patch embeddings and accurate sketch representations. Furthermore, we enforce a clustering constraint over the embeddings jointly with the network learning. The synonymous patches are self-organized as compact clusters, and their embeddings are guided to move towards their assigned cluster centroids. It raises the accuracy of the computed synonymous proximity. Experimental results show that our method significantly improves the performance on both controllable sketch synthesis and sketch healing.

----

## [1245] Neural Integro-Differential Equations

**Authors**: *Emanuele Zappala, Antonio Henrique de Oliveira Fonseca, Andrew Henry Moberly, Michael James Higley, Chadi Abdallah, Jessica A. Cardin, David van Dijk*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i9.26315](https://doi.org/10.1609/aaai.v37i9.26315)

**Abstract**:

Modeling continuous dynamical systems from discretely sampled observations is a fundamental problem in data science. Often, such dynamics are the result of non-local processes that present an integral over time. As such, these systems are modeled with Integro-Differential Equations (IDEs); generalizations of differential equations that comprise both an integral and a differential component. For example, brain dynamics are not accurately modeled by differential equations since their behavior is non-Markovian, i.e. dynamics are in part dictated by history. Here, we introduce the Neural IDE (NIDE), a novel deep learning framework based on the theory of IDEs where integral operators are learned using neural networks.  
    We test NIDE on several toy and brain activity datasets and demonstrate that NIDE outperforms other models. These tasks include time extrapolation as well as predicting dynamics from unseen initial conditions, which we test on whole-cortex activity recordings in freely behaving mice. Further, we show that NIDE can decompose dynamics into their Markovian and non-Markovian constituents, via the learned integral operator, which we test on fMRI brain activity recordings of people on ketamine. Finally, the integrand of the integral operator provides a latent space that gives insight into the underlying dynamics, which we demonstrate on wide-field brain imaging recordings. Altogether, NIDE is a novel approach that enables modeling of complex non-local dynamics with neural networks.

----

## [1246] Leveraging Structure for Improved Classification of Grouped Biased Data

**Authors**: *Daniel Zeiberg, Shantanu Jain, Predrag Radivojac*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i9.26316](https://doi.org/10.1609/aaai.v37i9.26316)

**Abstract**:

We consider semi-supervised binary classification for applications in which data points are naturally grouped (e.g., survey responses grouped by state) and the labeled data is biased (e.g., survey respondents are not representative of the population). The groups overlap in the feature space and consequently the input-output patterns are related across the groups. To model the inherent structure in such data, we assume the partition-projected class-conditional invariance across groups, defined in terms of the group-agnostic feature space. We demonstrate that under this assumption, the group carries additional information about the class, over the group-agnostic features, with provably improved area under the ROC curve. Further assuming invariance of partition-projected class-conditional distributions across both labeled and unlabeled data, we derive a semi-supervised algorithm that explicitly leverages the structure to learn an optimal, group-aware, probability-calibrated classifier, despite the bias in the labeled data. Experiments on synthetic and real data demonstrate the efficacy of our algorithm over suitable baselines and ablative models, spanning standard supervised and semi-supervised learning approaches, with and without incorporating the group directly as a feature.

----

## [1247] Are Transformers Effective for Time Series Forecasting?

**Authors**: *Ailing Zeng, Muxi Chen, Lei Zhang, Qiang Xu*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i9.26317](https://doi.org/10.1609/aaai.v37i9.26317)

**Abstract**:

Recently, there has been a surge of Transformer-based solutions for the long-term time series forecasting (LTSF) task. Despite the growing performance over the past few years, we question the validity of this line of research in this work. Specifically, Transformers is arguably the most successful solution to extract the semantic correlations among the elements in a long sequence. However, in time series modeling, we are to extract the temporal relations in an ordered set of continuous points. While employing positional encoding and using tokens to embed sub-series in Transformers facilitate preserving some ordering information, the nature of the permutation-invariant self-attention mechanism inevitably results in temporal information loss. 
To validate our claim, we introduce a set of embarrassingly simple one-layer linear models named LTSF-Linear for comparison. Experimental results on nine real-life datasets show that LTSF-Linear surprisingly outperforms existing sophisticated Transformer-based LTSF models in all cases, and often by a large margin. Moreover, we conduct comprehensive empirical studies to explore the impacts of various design elements of LTSF models on their temporal relation extraction capability. We hope this surprising finding opens up new research directions for the LTSF task. We also advocate revisiting the validity of Transformer-based solutions for other time series analysis tasks (e.g., anomaly detection) in the future.

----

## [1248] Substructure Aware Graph Neural Networks

**Authors**: *Dingyi Zeng, Wanlong Liu, Wenyu Chen, Li Zhou, Malu Zhang, Hong Qu*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i9.26318](https://doi.org/10.1609/aaai.v37i9.26318)

**Abstract**:

Despite the great achievements of Graph Neural Networks (GNNs) in graph learning, conventional GNNs struggle to break through the upper limit of the expressiveness of first-order Weisfeiler-Leman graph isomorphism test algorithm (1-WL) due to the consistency of the propagation paradigm of GNNs with the 1-WL.Based on the fact that it is easier to distinguish the original graph through subgraphs, we propose a  novel framework neural network framework called Substructure Aware Graph Neural Networks (SAGNN) to address these issues. We first propose a  Cut subgraph  which can be obtained from the original graph by continuously and selectively removing edges. Then we extend the random walk encoding paradigm to the return probability of the rooted node on the subgraph to capture the structural information and use it as a node feature to improve the expressiveness of GNNs. We theoretically prove that our framework is more powerful than 1-WL, and is superior in structure perception. Our extensive experiments demonstrate the effectiveness of our framework,  achieving state-of-the-art performance on a variety of well-proven graph tasks, and GNNs equipped with our framework perform flawlessly even in 3-WL failed graphs. Specifically, our framework achieves a maximum performance improvement of 83% compared to the base models and  32% compared to the previous state-of-the-art methods.

----

## [1249] ImGCL: Revisiting Graph Contrastive Learning on Imbalanced Node Classification

**Authors**: *Liang Zeng, Lanqing Li, Ziqi Gao, Peilin Zhao, Jian Li*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i9.26319](https://doi.org/10.1609/aaai.v37i9.26319)

**Abstract**:

Graph contrastive learning (GCL) has attracted a surge of attention due to its superior performance for learning node/graph representations without labels. However, in practice, the underlying class distribution of unlabeled nodes for the given graph is usually imbalanced. This highly imbalanced class distribution inevitably deteriorates the quality of learned node representations in GCL. Indeed, we empirically find that most state-of-the-art GCL methods cannot obtain discriminative representations and exhibit poor performance on imbalanced node classification. Motivated by this observation, we propose a principled GCL framework on Imbalanced node classification (ImGCL), which automatically and adaptively balances the representations learned from GCL without labels. Specifically, we first introduce the online clustering based progressively balanced sampling (PBS) method with theoretical rationale, which balances the training sets based on pseudo-labels obtained from learned representations in GCL. We then develop the node centrality based PBS method to better preserve the intrinsic structure of graphs, by upweighting the important nodes of the given graph. Extensive experiments on multiple imbalanced graph datasets and imbalanced settings demonstrate the effectiveness of our proposed framework, which significantly improves the performance of the recent state-of-the-art GCL methods. Further experimental ablations and analyses show that the ImGCL framework consistently improves the representation quality of nodes in under-represented (tail) classes.

----

## [1250] Foresee What You Will Learn: Data Augmentation for Domain Generalization in Non-stationary Environment

**Authors**: *Qiuhao Zeng, Wei Wang, Fan Zhou, Charles Ling, Boyu Wang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i9.26320](https://doi.org/10.1609/aaai.v37i9.26320)

**Abstract**:

Existing domain generalization aims to learn a generalizable model to perform well even on unseen domains. For many real-world machine learning applications, the data distribution often shifts gradually along domain indices. For example, a self-driving car with a vision system drives from dawn to dusk, with the sky gradually darkening. Therefore, the system must be able to adapt to changes in ambient illuminations and continue to drive safely on the road. In this paper, we formulate such problems as Evolving Domain Generalization, where a model aims to generalize well on a target domain by discovering and leveraging the evolving pattern of the environment. We then propose Directional Domain Augmentation (DDA), which simulates the unseen target features by mapping source data as augmentations through a domain transformer. Specifically, we formulate DDA as a bi-level optimization problem and solve it through a novel meta-learning approach in the representation space. We evaluate the proposed method on both synthetic datasets and real-world datasets, and empirical results show that our approach can outperform other existing methods.

----

## [1251] Acceleration of Large Transformer Model Training by Sensitivity-Based Layer Dropping

**Authors**: *Yujie Zeng, Wenlong He, Ihor Vasyltsov, Jiali Pang, Lin Chen*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i9.26321](https://doi.org/10.1609/aaai.v37i9.26321)

**Abstract**:

Transformer models are widely used in AI applications such as Natural Language Processing (NLP), Computer Vision (CV), etc. However, enormous computation workload be-comes an obstacle to train large transformer models efficiently. Recently, some methods focus on reducing the computation workload during the training by skipping some layers. How-ever, these methods use simple probability distribution and coarse-grained probability calculation, which significantly affect the model accuracy. To address the issue, in this paper we propose a novel method to accelerate training—Sensitivity-Based Layer Dropping (SBLD). SBLD uses lay-er-wise sensitivity data to switch on/off transformer layers in proper order to keep high accuracy. Besides, we adjust the probability of skipping transformer layers with a scheduler to accelerate training speed and get faster convergence. Our results show that SBLD solves the accuracy drop issue com-pared with prior layer dropping methods. Our SBLD method can decrease end-to-end training time by 19.67% during training of GPT-3 Medium model, the same time increasing the accuracy by 1.65% w.r.t. baseline. Furthermore, for SwinV2-L model the obtained Top-1 and Top-5 accuracies are also higher vs. the baseline. Thus, the proposed method is efficient and practical to improve the large transformer model training.

----

## [1252] Interventional SHAP Values and Interaction Values for Piecewise Linear Regression Trees

**Authors**: *Artjom Zern, Klaus Broelemann, Gjergji Kasneci*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i9.26322](https://doi.org/10.1609/aaai.v37i9.26322)

**Abstract**:

In recent years, game-theoretic Shapley values have gained increasing attention with respect to local model explanation by feature attributions. While the approach using Shapley values is model-independent, their (exact) computation is usually intractable, so efficient model-specific algorithms have been devised including approaches for decision trees or their ensembles in general. Our work goes further in this direction by extending the interventional TreeSHAP algorithm to piecewise linear regression trees, which gained more attention in the past few years. To this end, we introduce a decomposition of the contribution function based on decision paths, which allows a more comprehensible formulation of SHAP algorithms for tree-based models. Our algorithm can also be readily applied to computing SHAP interaction values of these models. In particular, as the main contribution of this paper, we provide a more efficient approach of interventional SHAP for tree-based models by precomputing statistics of the background data based on the tree structure.

----

## [1253] Enhanced Tensor Low-Rank and Sparse Representation Recovery for Incomplete Multi-View Clustering

**Authors**: *Chao Zhang, Huaxiong Li, Wei Lv, Zizheng Huang, Yang Gao, Chunlin Chen*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i9.26323](https://doi.org/10.1609/aaai.v37i9.26323)

**Abstract**:

Incomplete multi-view clustering (IMVC) has attracted remarkable attention due to the emergence of multi-view data with missing views in real applications. Recent methods attempt to recover the missing information to address the IMVC problem. However, they generally cannot fully explore the underlying properties and correlations of data similarities across views. This paper proposes a novel Enhanced Tensor Low-rank and Sparse Representation Recovery (ETLSRR) method, which reformulates the IMVC problem as a joint incomplete similarity graphs learning and complete tensor representation recovery problem. Specifically, ETLSRR learns the intra-view similarity graphs and constructs a 3-way tensor by stacking the graphs to explore the inter-view correlations. To alleviate the negative influence of missing views and data noise, ETLSRR decomposes the tensor into two parts: a sparse tensor and an intrinsic tensor, which models the noise and underlying true data similarities, respectively. Both global low-rank and local structured sparse characteristics of the intrinsic tensor are considered, which enhances the discrimination of similarity matrix. Moreover, instead of using the convex tensor nuclear norm, ETLSRR introduces a generalized non-convex tensor low-rank regularization to alleviate the biased approximation. Experiments on several datasets demonstrate the effectiveness of our method compared with the state-of-the-art methods.

----

## [1254] Denoising Multi-Similarity Formulation: A Self-Paced Curriculum-Driven Approach for Robust Metric Learning

**Authors**: *Chenkang Zhang, Lei Luo, Bin Gu*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i9.26324](https://doi.org/10.1609/aaai.v37i9.26324)

**Abstract**:

Deep Metric Learning (DML) is a group of techniques that aim to measure the similarity between objects through the neural network. Although the number of DML methods has rapidly increased in recent years,  most previous studies cannot effectively handle noisy data, which commonly exists in practical applications and often leads to serious performance deterioration. To overcome this limitation, in this paper, we build a connection between noisy samples and hard samples in the framework of self-paced learning, and propose a Balanced Self-Paced Metric Learning (BSPML) algorithm with a denoising multi-similarity formulation, where noisy samples are treated as extremely hard samples and adaptively excluded from the model training by  sample weighting. Especially, due to the pairwise relationship  and a new balance regularization term, the sub-problem  w.r.t.  sample weights is a  nonconvex  quadratic function.  To efficiently solve this nonconvex  quadratic problem, we propose a doubly stochastic projection coordinate gradient algorithm. Importantly, we  theoretically prove  the convergence  not only for the doubly stochastic projection coordinate gradient algorithm, but also for our BSPML algorithm. Experimental results on several standard  data sets demonstrate that our BSPML algorithm has   better generalization ability and robustness  than  the state-of-the-art   robust DML approaches.

----

## [1255] Rethinking Alignment and Uniformity in Unsupervised Image Semantic Segmentation

**Authors**: *Daoan Zhang, Chenming Li, Haoquan Li, Wenjian Huang, Lingyun Huang, Jianguo Zhang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i9.26325](https://doi.org/10.1609/aaai.v37i9.26325)

**Abstract**:

Unsupervised image segmentation aims to match low-level visual features with semantic-level representations without outer supervision. In this paper, we address the critical properties from the view of feature alignments and feature uniformity for UISS models. We also make a comparison between UISS and image-wise representation learning. Based on the analysis, we argue that the existing MI-based methods in UISS suffer from representation collapse. By this, we proposed a robust network called Semantic Attention Network(SAN), in which a new module Semantic Attention(SEAT) is proposed to generate pixel-wise and semantic features dynamically. Experimental results on multiple semantic segmentation benchmarks show that our unsupervised segmentation framework specializes in catching semantic representations, which outperforms all the unpretrained and even several pretrained methods.

----

## [1256] Behavior Estimation from Multi-Source Data for Offline Reinforcement Learning

**Authors**: *Guoxi Zhang, Hisashi Kashima*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i9.26326](https://doi.org/10.1609/aaai.v37i9.26326)

**Abstract**:

Offline reinforcement learning (RL) have received rising interest due to its appealing data efficiency. The present study addresses behavior estimation, a task that aims at estimating the data-generating policy. In particular, this work considers a scenario where data are collected from multiple sources. Neglecting data heterogeneity, existing approaches cannot provide good estimates and impede policy learning. To overcome this drawback, the present study proposes a latent variable model and a model-learning algorithm to infer a set of policies from data, which allows an agent to use as behavior policy the policy that best describes a particular trajectory. To illustrate the benefit of such a fine-grained characterization for multi-source data, this work showcases how the proposed model can be incorporated into an existing offline RL algorithm. Lastly, with extensive empirical evaluation this work confirms the risks of neglecting data heterogeneity and the efficacy of the proposed model.

----

## [1257] DARL: Distance-Aware Uncertainty Estimation for Offline Reinforcement Learning

**Authors**: *Hongchang Zhang, Jianzhun Shao, Shuncheng He, Yuhang Jiang, Xiangyang Ji*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i9.26327](https://doi.org/10.1609/aaai.v37i9.26327)

**Abstract**:

To facilitate offline reinforcement learning, uncertainty estimation is commonly used to detect out-of-distribution data. By inspecting, we show that current explicit uncertainty estimators such as Monte Carlo Dropout and model ensemble are not competent to provide trustworthy uncertainty estimation in offline reinforcement learning. Accordingly, we propose a non-parametric distance-aware uncertainty estimator which is sensitive to the change in the input space for offline reinforcement learning. Based on our new estimator, adaptive truncated quantile critics are proposed to underestimate the out-of-distribution samples. We show that the proposed distance-aware uncertainty estimator is able to offer better uncertainty estimation compared to previous methods. Experimental results demonstrate that our proposed DARL method is competitive to the state-of-the-art methods in offline evaluation tasks.

----

## [1258] When Neural Networks Fail to Generalize? A Model Sensitivity Perspective

**Authors**: *Jiajin Zhang, Hanqing Chao, Amit Dhurandhar, Pin-Yu Chen, Ali Tajer, Yangyang Xu, Pingkun Yan*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i9.26328](https://doi.org/10.1609/aaai.v37i9.26328)

**Abstract**:

Domain generalization (DG) aims to train a model to perform well in unseen domains under different distributions. This paper considers a more realistic yet more challenging scenario, namely Single Domain Generalization (Single-DG), where only a single source domain is available for training. To tackle this challenge, we first try to understand when neural networks fail to generalize? We empirically ascertain a property of a model that correlates strongly with its generalization that we coin as "model sensitivity". Based on our analysis, we propose a novel strategy of Spectral Adversarial Data Augmentation (SADA) to generate augmented images targeted at the highly sensitive frequencies. Models trained with these hard-to-learn samples can effectively suppress the sensitivity in the frequency space, which leads to improved generalization performance. Extensive experiments on multiple public datasets demonstrate the superiority of our approach, which surpasses the state-of-the-art single-DG methods by up to 2.55%. The source code is available at https://github.com/DIAL-RPI/Spectral-Adversarial-Data-Augmentation.

----

## [1259] Memorization Weights for Instance Reweighting in Adversarial Training

**Authors**: *Jianfu Zhang, Yan Hong, Qibin Zhao*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i9.26329](https://doi.org/10.1609/aaai.v37i9.26329)

**Abstract**:

Adversarial training is an effective way to defend deep neural networks (DNN) against adversarial examples. However, there are atypical samples that are rare and hard to learn, or even hurt DNNs' generalization performance on test data. In this paper, we propose a novel algorithm to reweight the training samples based on self-supervised techniques to mitigate the negative effects of the atypical samples. 
Specifically, a memory bank is built to record the popular samples as prototypes and calculate the memorization weight for each sample, evaluating the "typicalness" of a sample. All the training samples are reweigthed based on the proposed memorization weights to reduce the negative effects of atypical samples. Experimental results show the proposed method is flexible to boost state-of-the-art adversarial training methods, improving both robustness and standard accuracy of DNNs.

----

## [1260] FedALA: Adaptive Local Aggregation for Personalized Federated Learning

**Authors**: *Jianqing Zhang, Yang Hua, Hao Wang, Tao Song, Zhengui Xue, Ruhui Ma, Haibing Guan*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i9.26330](https://doi.org/10.1609/aaai.v37i9.26330)

**Abstract**:

A key challenge in federated learning (FL) is the statistical heterogeneity that impairs the generalization of the global model on each client. To address this, we propose a  method Federated learning with Adaptive Local Aggregation (FedALA) by capturing the desired information in the global model for client models in personalized FL. The key component of FedALA is an Adaptive Local Aggregation (ALA)  module, which can adaptively aggregate the downloaded global model and local model towards the local objective on each client to initialize the local model before training in each iteration.  To evaluate the effectiveness of FedALA, we conduct extensive experiments with five benchmark datasets in computer vision and natural language processing domains. FedALA outperforms eleven state-of-the-art baselines by up to 3.27% in test accuracy.  Furthermore, we also apply ALA  module to other federated learning methods and achieve up to 24.19% improvement in test accuracy. Code is available at https://github.com/TsingZ0/FedALA.

----

## [1261] Delving into the Adversarial Robustness of Federated Learning

**Authors**: *Jie Zhang, Bo Li, Chen Chen, Lingjuan Lyu, Shuang Wu, Shouhong Ding, Chao Wu*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i9.26331](https://doi.org/10.1609/aaai.v37i9.26331)

**Abstract**:

In Federated Learning (FL), models are as fragile as centrally trained models against adversarial examples. However, the adversarial robustness of federated learning remains largely unexplored. This paper casts light on the challenge of adversarial robustness of federated learning. To facilitate a better understanding of the adversarial vulnerability of the existing FL methods, we conduct comprehensive robustness evaluations on various attacks and adversarial training methods. Moreover, we reveal the negative impacts induced by directly adopting adversarial training in FL, which seriously hurts the test accuracy, especially in non-IID settings. In this work, we propose a novel algorithm called Decision Boundary based Federated Adversarial Training (DBFAT), which consists of two components (local re-weighting and global regularization) to improve both accuracy and robustness of FL systems. Extensive experiments on multiple datasets demonstrate that DBFAT consistently outperforms other baselines under both IID and non-IID settings.

----

## [1262] DRGCN: Dynamic Evolving Initial Residual for Deep Graph Convolutional Networks

**Authors**: *Lei Zhang, Xiaodong Yan, Jianshan He, Ruopeng Li, Wei Chu*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i9.26332](https://doi.org/10.1609/aaai.v37i9.26332)

**Abstract**:

Graph convolutional networks (GCNs) have been proved to be very practical to handle various graph-related tasks. It has attracted considerable research interest to study deep GCNs, due to their potential superior performance compared with shallow ones. However, simply increasing network depth will, on the contrary, hurt the performance due to the over-smoothing problem. Adding residual connection is proved to be effective for learning deep convolutional neural networks (deep CNNs), it is not trivial when applied to deep GCNs. Recent works proposed an initial residual mechanism that did alleviate the over-smoothing problem in deep GCNs. However, according to our study, their algorithms are quite sensitive to different datasets. In their setting, the personalization (dynamic) and correlation (evolving) of how residual applies are ignored. To this end, we propose a novel model called Dynamic evolving initial Residual Graph Convolutional Network (DRGCN). Firstly, we use a dynamic block for each node to adaptively fetch information from the initial representation. Secondly, we use an evolving block to model the residual evolving pattern between layers. Our experimental results show that our model effectively relieves the problem of over-smoothing in deep GCNs and outperforms the state-of-the-art (SOTA) methods on various benchmark datasets. Moreover, we develop a mini-batch version of DRGCN which can be applied to large-scale data. Coupling with several fair training techniques, our model reaches new SOTA results on the large-scale ogbn-arxiv dataset of Open Graph Benchmark (OGB). Our reproducible code is available on GitHub.

----

## [1263] Let the Data Choose: Flexible and Diverse Anchor Graph Fusion for Scalable Multi-View Clustering

**Authors**: *Pei Zhang, Siwei Wang, Liang Li, Changwang Zhang, Xinwang Liu, En Zhu, Zhe Liu, Lu Zhou, Lei Luo*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i9.26333](https://doi.org/10.1609/aaai.v37i9.26333)

**Abstract**:

In the past few years, numerous multi-view graph clustering algorithms have been proposed to enhance the clustering performance by exploring information from multiple views. Despite the superior performance, the high time and space expenditures limit their scalability. Accordingly, anchor graph learning has been introduced to alleviate the computational complexity. However, existing approaches can be further improved by the following considerations: (i) Existing anchor-based methods share the same number of anchors across views. This strategy violates the diversity and flexibility of multi-view data distribution. (ii) Searching for the optimal anchor number within hyper-parameters takes much extra tuning time, which makes existing methods impractical. (iii) How to flexibly fuse multi-view anchor graphs of diverse sizes has not been well explored in existing literature. To address the above issues, we propose a novel anchor-based method termed Flexible and Diverse Anchor Graph Fusion for Scalable Multi-view Clustering (FDAGF) in this paper. Instead of manually tuning optimal anchor with massive hyper-parameters, we propose to optimize the contribution weights of a group of pre-defined anchor numbers to avoid extra time expenditure among views. Most importantly, we propose a novel hybrid fusion strategy for multi-size anchor graphs with theoretical proof, which allows flexible and diverse anchor graph fusion. Then, an efficient linear optimization algorithm is proposed to solve the resultant problem. Comprehensive experimental results demonstrate the effectiveness and efficiency of our proposed framework. The source code is available at https://github.com/Jeaninezpp/FDAGF.

----

## [1264] Optimal Sparse Regression Trees

**Authors**: *Rui Zhang, Rui Xin, Margo I. Seltzer, Cynthia Rudin*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i9.26334](https://doi.org/10.1609/aaai.v37i9.26334)

**Abstract**:

Regression trees are one of the oldest forms of AI models, and their predictions can be made without a calculator, which makes them broadly useful, particularly for high-stakes applications. Within the large literature on regression trees, there has been little effort towards full provable optimization, mainly due to the computational hardness of the problem. This work proposes a dynamic programming-with-bounds approach to the construction of provably-optimal sparse regression trees. We leverage a novel lower bound based on an optimal solution to the k-Means clustering algorithm on one dimensional data. We are often able to find optimal sparse trees in seconds, even for challenging datasets that involve large numbers of samples and highly-correlated features.

----

## [1265] High-Dimensional Dueling Optimization with Preference Embedding

**Authors**: *Yangwenhui Zhang, Hong Qian, Xiang Shu, Aimin Zhou*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i9.26335](https://doi.org/10.1609/aaai.v37i9.26335)

**Abstract**:

In many scenarios of black-box optimization, evaluating the objective function values of solutions is expensive, while comparing a pair of solutions is relatively cheap, which yields the dueling black-box optimization. The side effect of dueling optimization is that it doubles the dimension of solution space and exacerbates the dimensionality scalability issue of black-box optimization, e.g., Bayesian optimization. To address this issue, the existing dueling optimization methods fix one solution when dueling throughout the optimization process, but it may reduce their efficacy. Fortunately, it has been observed that, in recommendation systems, the dueling results are mainly determined by the latent human preferences. In this paper, we abstract this phenomenon as the preferential intrinsic dimension and inject it into the dueling Bayesian optimization, resulting in the preferential embedding dueling Bayesian optimization (PE-DBO). PE-DBO decouples optimization and pairwise comparison via the preferential embedding matrix. Optimization is performed in the preferential intrinsic subspace with much lower dimensionality, while pairwise comparison is completed in the original dueling solution space. Theoretically, we disclose that the preference function can be approximately preserved in the lower-dimensional preferential intrinsic subspace. Experiment results verify that, on molecule discovery and web page recommendation dueling optimization tasks, the preferential intrinsic dimension exists and PE-DBO is superior in scalability compared with that of the state-of-the-art (SOTA) methods.

----

## [1266] Spectral Feature Augmentation for Graph Contrastive Learning and Beyond

**Authors**: *Yifei Zhang, Hao Zhu, Zixing Song, Piotr Koniusz, Irwin King*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i9.26336](https://doi.org/10.1609/aaai.v37i9.26336)

**Abstract**:

Although  augmentations (e.g., perturbation of graph edges, image crops)  boost the efficiency of Contrastive Learning (CL), feature level augmentation is another plausible, complementary yet not well researched strategy. Thus, we present a novel spectral feature argumentation for contrastive learning on graphs (and images). To this end, for each data view, we estimate a low-rank approximation per feature map and  subtract that approximation from the map to obtain its complement. This is achieved by the proposed herein  incomplete power iteration, a non-standard power iteration regime which enjoys two valuable byproducts (under mere one or two iterations): (i) it partially balances spectrum of the feature map, and (ii) it injects the noise into rebalanced singular values of the feature map (spectral augmentation). For two views, we align these rebalanced feature maps as such an improved alignment step can focus more on less dominant singular values of matrices of both views, whereas the spectral augmentation does not affect the spectral angle alignment (singular vectors are not perturbed). We derive the analytical form for: (i) the incomplete power iteration to capture its spectrum-balancing effect, and (ii) the variance of singular values  augmented implicitly by the noise. We also show that the spectral augmentation improves the generalization bound. Experiments on graph/image datasets show that our spectral feature augmentation  outperforms baselines, and is complementary with other  augmentation strategies and compatible with various contrastive losses.

----

## [1267] Scalable Bayesian Meta-Learning through Generalized Implicit Gradients

**Authors**: *Yilang Zhang, Bingcong Li, Shijian Gao, Georgios B. Giannakis*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i9.26337](https://doi.org/10.1609/aaai.v37i9.26337)

**Abstract**:

Meta-learning owns unique effectiveness and swiftness in tackling emerging tasks with limited data. Its broad applicability is revealed by viewing it as a bi-level optimization problem. The resultant algorithmic viewpoint however, faces scalability issues when the inner-level optimization relies on gradient-based iterations. Implicit differentiation has been considered to alleviate this challenge, but it is restricted to an isotropic Gaussian prior, and only favors deterministic meta-learning approaches. This work markedly mitigates the scalability bottleneck by cross-fertilizing the benefits of implicit differentiation to probabilistic Bayesian meta-learning. The novel implicit Bayesian meta-learning (iBaML) method not only broadens the scope of learnable priors, but also quantifies the associated uncertainty. Furthermore, the ultimate complexity is well controlled regardless of the inner-level optimization trajectory. Analytical error bounds are established to demonstrate the precision and efficiency of the generalized implicit gradient over the explicit one. Extensive numerical tests are also carried out to empirically validate the performance of the proposed method.

----

## [1268] Dynamic Heterogeneous Graph Attention Neural Architecture Search

**Authors**: *Zeyang Zhang, Ziwei Zhang, Xin Wang, Yijian Qin, Zhou Qin, Wenwu Zhu*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i9.26338](https://doi.org/10.1609/aaai.v37i9.26338)

**Abstract**:

Dynamic heterogeneous graph neural networks (DHGNNs) have been shown to be effective in handling the ubiquitous dynamic heterogeneous graphs. However, the existing DHGNNs are hand-designed, requiring extensive human efforts and failing to adapt to diverse dynamic heterogeneous graph scenarios. In this paper, we propose to automate the design of DHGNN, which faces two major challenges: 1) how to design the search space to jointly consider the spatial-temporal dependencies and heterogeneous interactions in graphs; 2) how to design an efficient search algorithm in the potentially large and complex search space. To tackle these challenges, we propose a novel Dynamic Heterogeneous Graph Attention Search (DHGAS) method. Our proposed method can automatically discover the optimal DHGNN architecture and adapt to various dynamic heterogeneous graph scenarios without human guidance. In particular, we first propose a unified dynamic heterogeneous graph attention (DHGA) framework, which enables each node to jointly attend its heterogeneous and dynamic neighbors. Based on the framework, we design a localization space to determine where the attention should be applied and a parameterization space to determine how the attention should be parameterized. Lastly, we design a multi-stage differentiable search algorithm to efficiently explore the search space. Extensive experiments on real-world dynamic heterogeneous graph datasets demonstrate that our proposed method significantly outperforms state-of-the-art baselines for tasks including link prediction, node classification and node regression. To the best of our knowledge, DHGAS is the first dynamic heterogeneous graph neural architecture search method.

----

## [1269] Dynamic Ensemble of Low-Fidelity Experts: Mitigating NAS "Cold-Start"

**Authors**: *Junbo Zhao, Xuefei Ning, Enshu Liu, Binxin Ru, Zixuan Zhou, Tianchen Zhao, Chen Chen, Jiajin Zhang, Qingmin Liao, Yu Wang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i9.26339](https://doi.org/10.1609/aaai.v37i9.26339)

**Abstract**:

Predictor-based Neural Architecture Search (NAS) employs an architecture performance predictor to improve the sample efficiency. However, predictor-based NAS suffers from the severe ``cold-start'' problem, since a large amount of architecture-performance data is required to get a working predictor. In this paper, we focus on exploiting information in cheaper-to-obtain performance estimations (i.e., low-fidelity information) to mitigate the large data requirements of predictor training. Despite the intuitiveness of this idea, we observe that using inappropriate low-fidelity information even damages the prediction ability and different search spaces have different preferences for low-fidelity information types. To solve the problem and better fuse beneficial information provided by different types of low-fidelity information, we propose a novel dynamic ensemble predictor framework that comprises two steps. In the first step, we train different sub-predictors on different types of available low-fidelity information to extract beneficial knowledge as low-fidelity experts. In the second step, we learn a gating network to dynamically output a set of weighting coefficients conditioned on each input neural architecture, which will be used to combine the predictions of different low-fidelity experts in a weighted sum. The overall predictor is optimized on a small set of actual architecture-performance data to fuse the knowledge from different low-fidelity experts to make the final prediction. We conduct extensive experiments across five search spaces with different architecture encoders under various experimental settings. For example, our methods can improve the Kendall's Tau correlation coefficient between actual performance and predicted scores from 0.2549 to 0.7064 with only 25 actual architecture-performance data on NDS-ResNet. Our method can easily be incorporated into existing predictor-based NAS frameworks to discover better architectures. Our method will be implemented in Mindspore (Huawei 2020), and the example code is published at https://github.com/A-LinCui/DELE.

----

## [1270] Tensorized Incomplete Multi-View Clustering with Intrinsic Graph Completion

**Authors**: *Shuping Zhao, Jie Wen, Lunke Fei, Bob Zhang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i9.26340](https://doi.org/10.1609/aaai.v37i9.26340)

**Abstract**:

Most of the existing incomplete multi-view clustering (IMVC) methods focus on attaining a consensus representation from different views but ignore the important information hidden in the missing views and the latent intrinsic structures in each view. To tackle these issues, in this paper, a unified and novel framework, named tensorized incomplete multi-view clustering with intrinsic graph completion (TIMVC_IGC) is proposed. Firstly, owing to the effectiveness of the low-rank representation in revealing the inherent structure of the data, we exploit it to infer the missing instances and construct the complete graph for each view. Afterwards, inspired by the structural consistency, a between-view consistency constraint is imposed to guarantee the similarity of the graphs from different views. More importantly, the TIMVC_IGC simultaneously learns the low-rank structures of the different views and explores the correlations of the different graphs in a latent manifold sub-space using a low-rank tensor constraint, such that the intrinsic graphs of the different views can be obtained. Finally, a consensus representation for each sample is gained with a co-regularization term for final clustering.  Experimental results on several real-world databases illustrates that the proposed method can outperform the other state-of-the-art related methods for incomplete multi-view clustering.

----

## [1271] Imbalanced Label Distribution Learning

**Authors**: *Xingyu Zhao, Yuexuan An, Ning Xu, Jing Wang, Xin Geng*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i9.26341](https://doi.org/10.1609/aaai.v37i9.26341)

**Abstract**:

Label distribution covers a certain number of labels, representing the degree to which each label describes an instance. The learning process on the instances labeled by label distributions is called Label Distribution Learning (LDL). Although LDL has been applied successfully to many practical applications, one problem with existing LDL methods is that they are limited to data with balanced label information. However, annotation information in real-world data often exhibits imbalanced distributions, which significantly degrades the performance of existing methods. In this paper, we investigate the Imbalanced Label Distribution Learning (ILDL) problem. To handle this challenging problem, we delve into the characteristics of ILDL and empirically find that the representation distribution shift is the underlying reason for the performance degradation of existing methods. Inspired by this finding, we present a novel method named Representation Distribution Alignment (RDA). RDA aligns the distributions of feature representations and label representations to alleviate the impact of the distribution gap between the training set and the test set caused by the imbalance issue. Extensive experiments verify the superior performance of RDA. Our work fills the gap in benchmarks and techniques for practical ILDL problems.

----

## [1272] CoopInit: Initializing Generative Adversarial Networks via Cooperative Learning

**Authors**: *Yang Zhao, Jianwen Xie, Ping Li*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i9.26342](https://doi.org/10.1609/aaai.v37i9.26342)

**Abstract**:

Numerous research efforts have been made to stabilize the training of the Generative Adversarial Networks (GANs), such as through regularization and architecture design. However, we identify the instability can also arise from the fragile balance at the early stage of adversarial learning. This paper proposes the CoopInit, a simple yet effective cooperative learning-based initialization strategy that can quickly learn a good starting point for GANs, with a very small computation overhead during training. The proposed algorithm consists of two learning stages: (i) Cooperative initialization stage: The discriminator of GAN is treated as an energy-based model (EBM) and is optimized via maximum likelihood estimation (MLE), with the help of the GAN's generator to provide synthetic data to approximate the learning gradients. The EBM also guides the MLE learning of the generator via MCMC teaching; (ii) Adversarial finalization stage: After a few iterations of initialization, the algorithm seamlessly transits to the regular mini-max adversarial training until convergence. The motivation is that the MLE-based initialization stage drives the model towards mode coverage, which is helpful in alleviating the issue of mode dropping during the adversarial learning stage. We demonstrate the effectiveness of the proposed approach on image generation and one-sided unpaired image-to-image translation tasks through extensive experiments.

----

## [1273] AutoGraph: Optimizing DNN Computation Graph for Parallel GPU Kernel Execution

**Authors**: *Yuxuan Zhao, Qi Sun, Zhuolun He, Yang Bai, Bei Yu*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i9.26343](https://doi.org/10.1609/aaai.v37i9.26343)

**Abstract**:

Deep learning frameworks optimize the computation graphs and intra-operator computations to boost the inference performance on GPUs,  while inter-operator parallelism is usually ignored.  In this paper, a unified framework, AutoGraph, is proposed to obtain highly optimized computation graphs in favor of parallel executions of GPU kernels.  A novel dynamic programming algorithm, combined with backtracking search, is adopted to explore the optimal graph optimization solution, with the fast performance estimation from the mixed critical path cost. Accurate runtime information based on GPU Multi-Stream launched with CUDA Graph is utilized to determine the convergence of the optimization. Experimental results demonstrate that our method achieves up to 3.47x speedup over existing graph optimization methods. Moreover, AutoGraph outperforms state-of-the-art parallel kernel launch frameworks by up to 1.26x.

----

## [1274] Fairness and Explainability: Bridging the Gap towards Fair Model Explanations

**Authors**: *Yuying Zhao, Yu Wang, Tyler Derr*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i9.26344](https://doi.org/10.1609/aaai.v37i9.26344)

**Abstract**:

While machine learning models have achieved unprecedented success in real-world applications, they might make biased/unfair decisions for specific demographic groups and hence result in discriminative outcomes. Although research efforts have been devoted to measuring and mitigating bias, they mainly study bias from the result-oriented perspective while neglecting the bias encoded in the decision-making procedure. This results in their inability to capture procedure-oriented bias, which therefore limits the ability to have a fully debiasing method. Fortunately, with the rapid development of explainable machine learning, explanations for predictions are now available to gain insights into the procedure. In this work, we bridge the gap between fairness and explainability by presenting a novel perspective of procedure-oriented fairness based on explanations. We identify the procedure-based bias by measuring the gap of explanation quality between different groups with Ratio-based and Value-based Explanation Fairness.  The new metrics further motivate us to design an optimization objective to mitigate the procedure-based bias where we observe that it will also mitigate bias from the prediction. Based on our designed optimization objective, we propose a Comprehensive Fairness Algorithm (CFA), which simultaneously fulfills multiple objectives - improving traditional fairness, satisfying explanation fairness, and maintaining the utility performance. Extensive experiments on real-world datasets demonstrate the effectiveness of our proposed CFA and highlight the importance of considering fairness from the explainability perspective. Our code: https://github.com/YuyingZhao/FairExplanations-CFA.

----

## [1275] Adaptive Policy Learning for Offline-to-Online Reinforcement Learning

**Authors**: *Han Zheng, Xufang Luo, Pengfei Wei, Xuan Song, Dongsheng Li, Jing Jiang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i9.26345](https://doi.org/10.1609/aaai.v37i9.26345)

**Abstract**:

Conventional reinforcement learning (RL) needs an environment to collect fresh data, which is impractical when online interactions are costly. Offline RL provides an alternative solution by directly learning from the previously collected dataset. However, it will yield unsatisfactory performance if the quality of the offline datasets is poor. In this paper, we consider an offline-to-online setting where the agent is first learned from the offline dataset and then trained online, and propose a framework called Adaptive Policy Learning for effectively taking advantage of offline and online data. Specifically, we explicitly consider the difference between the online and offline data and apply an adaptive update scheme accordingly, that is, a pessimistic update strategy for the offline dataset and an optimistic/greedy update scheme for the online dataset. Such a simple and effective method provides a way to mix the offline and online RL and achieve the best of both worlds. We further provide two detailed algorithms for implementing the framework through embedding value or policy-based RL algorithms into it. Finally, we conduct extensive experiments on popular continuous control tasks, and results show that our algorithm can learn the expert policy with high sample efficiency even when the quality of offline dataset is poor, e.g., random dataset.

----

## [1276] Multi-Level Confidence Learning for Trustworthy Multimodal Classification

**Authors**: *Xiao Zheng, Chang Tang, Zhiguo Wan, Chengyu Hu, Wei Zhang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i9.26346](https://doi.org/10.1609/aaai.v37i9.26346)

**Abstract**:

With the rapid development of various data acquisition technologies, more and more multimodal data come into being. It is important to integrate different modalities which are with high-dimensional features for boosting final multimodal data classification task. However, existing multimodal classification methods mainly focus on exploiting the complementary information of different modalities, while ignoring the learning confidence during information fusion. In this paper, we propose a trustworthy multimodal classification network via multi-level confidence learning, referred to as MLCLNet. Considering that a large number of feature dimensions could not contribute to final classification performance but disturb the discriminability of different samples, we propose a feature confidence learning mechanism to suppress some redundant features, as well as enhancing the expression of discriminative feature dimensions in each modality. In order to capture the inherent sample structure information implied in each modality, we design a graph convolutional network branch to learn the corresponding structure preserved feature representation and generate modal-specific initial classification labels. Since samples from different modalities should share consistent labels, a cross-modal label fusion module is deployed to capture the label correlations of different modalities. In addition, motivated the ideally orthogonality of final fused label matrix, we design a label confidence loss to supervise the network for learning more separable data representations. To the best of our knowledge, MLCLNet is the first work which integrates both feature and label-level confidence learning for multimodal classification. Extensive experiments on four multimodal medical datasets are conducted to validate superior performance of MLCLNet when compared to other state-of-the-art methods.

----

## [1277] CowClip: Reducing CTR Prediction Model Training Time from 12 Hours to 10 Minutes on 1 GPU

**Authors**: *Zangwei Zheng, Pengtai Xu, Xuan Zou, Da Tang, Zhen Li, Chenguang Xi, Peng Wu, Leqi Zou, Yijie Zhu, Ming Chen, Xiangzhuo Ding, Fuzhao Xue, Ziheng Qin, Youlong Cheng, Yang You*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i9.26347](https://doi.org/10.1609/aaai.v37i9.26347)

**Abstract**:

The click-through rate (CTR) prediction task is to predict whether a user will click on the recommended item. As mind-boggling amounts of data are produced online daily, accelerating CTR prediction model training is critical to ensuring an up-to-date model and reducing the training cost. One approach to increase the training speed is to apply large batch training. However, as shown in computer vision and natural language processing tasks, training with a large batch easily suffers from the loss of accuracy. Our experiments show that previous scaling rules fail in the training of CTR prediction neural networks. To tackle this problem, we first theoretically show that different frequencies of ids make it challenging to scale hyperparameters when scaling the batch size. To stabilize the training process in a large batch size setting, we develop the adaptive Column-wise Clipping (CowClip). It enables an easy and effective scaling rule for the embeddings, which keeps the learning rate unchanged and scales the L2 loss. We conduct extensive experiments with four CTR prediction networks on two real-world datasets and successfully scaled 128 times the original batch size without accuracy loss. In particular, for CTR prediction model DeepFM training on the Criteo dataset, our optimization framework enlarges the batch size from 1K to 128K with over 0.1% AUC improvement and reduces training time from 12 hours to 10 minutes on a single V100 GPU. Our code locates at github.com/bytedance/LargeBatchCTR.

----

## [1278] Data Imputation with Iterative Graph Reconstruction

**Authors**: *Jiajun Zhong, Ning Gui, Weiwei Ye*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i9.26348](https://doi.org/10.1609/aaai.v37i9.26348)

**Abstract**:

Effective data imputation demands rich latent ``structure" discovery capabilities from ``plain" tabular data. Recent advances in graph neural networks-based data imputation solutions show their structure learning potentials by translating tabular data as bipartite graphs. However, due to a lack of relations between samples, they treat all samples equally which is against one important observation: ``similar sample should give more information about missing values."  This paper presents a novel Iterative graph Generation and Reconstruction framework for Missing data imputation(IGRM). Instead of treating all samples equally, we introduce the concept: ``friend networks" to represent different relations among samples. To generate an accurate friend network with missing data, an end-to-end friend network reconstruction solution is designed to allow for continuous friend network optimization during imputation learning. The representation of the optimized friend network, in turn, is used to further optimize the data imputation process with differentiated message passing. Experiment results on eight benchmark datasets show that IGRM yields 39.13% lower mean absolute error compared with nine baselines and 9.04% lower than the second-best. Our code is available at https://github.com/G-AILab/IGRM.

----

## [1279] Does It Pay to Optimize AUC?

**Authors**: *Baojian Zhou, Steven Skiena*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i9.26349](https://doi.org/10.1609/aaai.v37i9.26349)

**Abstract**:

The Area Under the ROC Curve (AUC) is an important model metric for evaluating binary classifiers, and many algorithms have been proposed to optimize AUC approximately. It raises the question of whether the generally insignificant gains observed by previous studies are due to inherent limitations of the metric or the inadequate quality of optimization.

To better understand the value of optimizing for AUC, we present an efficient algorithm, namely AUC-opt, to find the provably optimal AUC linear classifier in R2, which runs in O(n+n- log n+n-) where n+ and n- are the number of positive and negative samples respectively. Furthermore, it can be naturally extended to Rd in O(n+n-d-1 log (n+n-)) by recursively calling AUC-opt in lower-dimensional spaces. We prove the problem is NP-complete when d is not fixed, reducing from the open hemisphere problem.

Compared with other methods, experiments show that AUC-opt achieves statistically significant improvements between 17 to 40 in R2 and 4 to 42 in R3 of 50 t-SNE training datasets. However, generally, the gain proves insignificant on most testing datasets compared to the best standard classifiers. Similar observations are found for nonlinear AUC methods under real-world datasets.

----

## [1280] SLOTH: Structured Learning and Task-Based Optimization for Time Series Forecasting on Hierarchies

**Authors**: *Fan Zhou, Chen Pan, Lintao Ma, Yu Liu, Shiyu Wang, James Zhang, Xinxin Zhu, Xuanwei Hu, Yunhua Hu, Yangfei Zheng, Lei Lei, Hu Yun*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i9.26350](https://doi.org/10.1609/aaai.v37i9.26350)

**Abstract**:

Multivariate time series forecasting with hierarchical structure
is widely used in real-world applications, e.g., sales
predictions for the geographical hierarchy formed by cities,
states, and countries. The hierarchical time series (HTS) forecasting
includes two sub-tasks, i.e., forecasting and reconciliation.
In the previous works, hierarchical information is only
integrated in the reconciliation step to maintain coherency,
but not in forecasting step for accuracy improvement. In this
paper, we propose two novel tree-based feature integration
mechanisms, i.e., top-down convolution and bottom-up attention
to leverage the information of the hierarchical structure
to improve the forecasting performance. Moreover, unlike
most previous reconciliation methods which either rely
on strong assumptions or focus on coherent constraints only,
we utilize deep neural optimization networks, which not only
achieve coherency without any assumptions, but also allow
more flexible and realistic constraints to achieve task-based
targets, e.g., lower under-estimation penalty and meaningful
decision-making loss to facilitate the subsequent downstream
tasks. Experiments on real-world datasets demonstrate that
our tree-based feature integration mechanism achieves superior
performances on hierarchical forecasting tasks compared
to the state-of-the-art methods, and our neural optimization
networks can be applied to real-world tasks effectively without
any additional effort under coherence and task-based constraints.

----

## [1281] Robust Temporal Smoothness in Multi-Task Learning

**Authors**: *Menghui Zhou, Yu Zhang, Yun Yang, Tong Liu, Po Yang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i9.26351](https://doi.org/10.1609/aaai.v37i9.26351)

**Abstract**:

Multi-task learning models based on temporal smoothness assumption, in which each time point of a sequence of time points concerns a task of prediction, assume the adjacent tasks are similar to each other. However, the effect of outliers is not taken into account.  In this paper, we show that even only one outlier task will destroy the performance of the entire model.  To solve this problem, we propose two Robust Temporal Smoothness (RoTS) frameworks. Compared with the existing models based on temporal relation, our methods not only chase the temporal smoothness information but identify outlier tasks, however, without increasing the computational complexity.  Detailed theoretical analyses are presented to evaluate the performance of our methods.  Experimental results on synthetic and real-life datasets demonstrate the effectiveness of our frameworks. We also discuss several potential specific applications and extensions of our RoTS frameworks.

----

## [1282] Combining Adversaries with Anti-adversaries in Training

**Authors**: *Xiaoling Zhou, Nan Yang, Ou Wu*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i9.26352](https://doi.org/10.1609/aaai.v37i9.26352)

**Abstract**:

Adversarial training is an effective learning technique to improve the robustness of deep neural networks. In this study, the influence of adversarial training on deep learning models in terms of fairness, robustness, and generalization is theoretically investigated under more general perturbation scope that different samples can have different perturbation directions (the adversarial and anti-adversarial directions) and varied perturbation bounds. Our theoretical explorations suggest that the combination of adversaries and anti-adversaries (samples with anti-adversarial perturbations) in training can be more effective in achieving better fairness between classes and a better tradeoff between robustness and generalization in some typical learning scenarios (e.g., noisy label learning and imbalance learning) compared with standard adversarial training. On the basis of our theoretical findings, a more general learning objective that combines adversaries and anti-adversaries with varied bounds on each training sample is presented. Meta learning is utilized to optimize the combination weights. Experiments on benchmark datasets under different learning scenarios verify our theoretical findings and the effectiveness of the proposed methodology.

----

## [1283] Gradient-Adaptive Pareto Optimization for Constrained Reinforcement Learning

**Authors**: *Zixian Zhou, Mengda Huang, Feiyang Pan, Jia He, Xiang Ao, Dandan Tu, Qing He*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i9.26353](https://doi.org/10.1609/aaai.v37i9.26353)

**Abstract**:

Constrained Reinforcement Learning (CRL) burgeons broad interest in recent years, which pursues maximizing long-term returns while constraining costs. Although CRL can be cast as a multi-objective optimization problem, it is still facing the key challenge that gradient-based Pareto optimization methods tend to stick to known Pareto-optimal solutions even when they yield poor returns (e.g., the safest self-driving car that never moves) or violate the constraints (e.g., the record-breaking racer that crashes the car). In this paper, we propose Gradient-adaptive Constrained Policy Optimization (GCPO for short), a novel Pareto optimization method for CRL with two adaptive gradient recalibration techniques. First, to find Pareto-optimal solutions with balanced performance over all targets, we propose gradient rebalancing which forces the agent to improve more on under-optimized objectives at every policy iteration. Second, to guarantee that the cost constraints are satisfied, we propose gradient perturbation that can temporarily sacrifice the returns for costs. Experiments on the SafetyGym benchmarks show that our method consistently outperforms previous CRL methods in reward while satisfying the constraints.

----

## [1284] Quantized Feature Distillation for Network Quantization

**Authors**: *Ke Zhu, Yin-Yin He, Jianxin Wu*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i9.26354](https://doi.org/10.1609/aaai.v37i9.26354)

**Abstract**:

Neural network quantization aims to accelerate and trim full-precision neural network models by using low bit approximations. Methods adopting the quantization aware training (QAT) paradigm have recently seen a rapid growth, but are often conceptually complicated. This paper proposes a novel and highly effective QAT method, quantized feature distillation (QFD). QFD first trains a quantized (or binarized) representation as the teacher, then quantize the network using knowledge distillation (KD). Quantitative results show that QFD is more flexible and effective (i.e., quantization friendly) than previous quantization methods. QFD surpasses existing methods by a noticeable margin on not only image classification but also object detection, albeit being much simpler. Furthermore, QFD quantizes ViT and Swin-Transformer on MS-COCO detection and segmentation, which verifies its potential in real world deployment. To the best of our knowledge, this is the first time that vision transformers have been quantized in object detection and image segmentation tasks.

----

## [1285] Bayesian Cross-Modal Alignment Learning for Few-Shot Out-of-Distribution Generalization

**Authors**: *Lin Zhu, Xinbing Wang, Chenghu Zhou, Nanyang Ye*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i9.26355](https://doi.org/10.1609/aaai.v37i9.26355)

**Abstract**:

Recent advances in large pre-trained models showed promising results in few-shot learning. However, their generalization ability on two-dimensional Out-of-Distribution (OoD) data, i.e., correlation shift and diversity shift, has not been thoroughly investigated. Researches have shown that even with a significant amount of training data, few methods can achieve better performance than the standard empirical risk minimization method (ERM) in OoD generalization. This few-shot OoD generalization dilemma emerges as a challenging direction in deep neural network generalization research, where the performance suffers from overfitting on few-shot examples and OoD generalization errors. In this paper, leveraging a broader supervision source, we explore a novel Bayesian cross-modal image-text alignment learning method (Bayes-CAL) to address this issue. Specifically, the model is designed as only text representations are fine-tuned via a Bayesian modelling approach with gradient orthogonalization loss and invariant risk minimization (IRM) loss. The Bayesian approach is essentially introduced to avoid overfitting the base classes observed during training and improve generalization to broader unseen classes. The dedicated loss is introduced to achieve better image-text alignment by disentangling the causal and non-casual parts of image features. Numerical experiments demonstrate that Bayes-CAL achieved state-of-the-art OoD generalization performances on two-dimensional distribution shifts. Moreover, compared with CLIP-like models, Bayes-CAL yields more stable generalization performances on unseen classes. Our code is available at https://github.com/LinLLLL/BayesCAL.

----

## [1286] ContraFeat: Contrasting Deep Features for Semantic Discovery

**Authors**: *Xinqi Zhu, Chang Xu, Dacheng Tao*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i9.26356](https://doi.org/10.1609/aaai.v37i9.26356)

**Abstract**:

StyleGAN has shown strong potential for disentangled semantic control, thanks to its special design of multi-layer intermediate latent variables. However, existing semantic discovery methods on StyleGAN rely on manual selection of modified latent layers to obtain satisfactory manipulation results, which is tedious and demanding. In this paper, we propose a model that automates this process and achieves state-of-the-art semantic discovery performance. The model consists of an attention-equipped navigator module and losses contrasting deep-feature changes. We propose two model variants, with one contrasting samples in a binary manner, and another one contrasting samples with learned prototype variation patterns. The proposed losses are computed with pretrained deep features, based on our assumption that the features implicitly possess the desired semantic variation structure including consistency and orthogonality. Additionally, we design two metrics to quantitatively evaluate the performance of semantic discovery methods on FFHQ dataset, and also show that disentangled representations can be derived via a simple training process. Experimentally, we show that our models achieve state-of-the-art semantic discovery results without relying on layer-wise manual selection, and these discovered semantics can be used to manipulate real-world images.

----

## [1287] Locate Then Generate: Bridging Vision and Language with Bounding Box for Scene-Text VQA

**Authors**: *Yongxin Zhu, Zhen Liu, Yukang Liang, Xin Li, Hao Liu, Changcun Bao, Linli Xu*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i9.26357](https://doi.org/10.1609/aaai.v37i9.26357)

**Abstract**:

In this paper, we propose a novel multi-modal framework for Scene Text Visual Question Answering (STVQA), which requires models to read scene text in images for question answering. Apart from text or visual objects, which could exist independently, scene text naturally links text and visual modalities together by conveying linguistic semantics while being a visual object in an image simultaneously. Different to conventional STVQA models which take the linguistic semantics and visual semantics in scene text as two separate features, in this paper, we propose a paradigm of "Locate Then Generate" (LTG), which explicitly unifies this two semantics with the spatial bounding box as a bridge connecting them. Specifically, at first, LTG locates the region in an image that may contain the answer words with an answer location module (ALM) consisting of a region proposal network and a language refinement network, both of which can transform to each other with one-to-one mapping via the scene text bounding box. Next, given the answer words selected by ALM, LTG generates a readable answer sequence with an answer generation module (AGM) based on a pre-trained language model. As a benefit of the explicit alignment of the visual and linguistic semantics, even without any scene text based pre-training tasks, LTG can boost the absolute accuracy by +6.06% and +6.92% on the TextVQA dataset and the ST-VQA dataset respectively, compared with a non-pre-training baseline. We further demonstrate that LTG effectively unifies visual and text modalities through the spatial bounding box connection, which is underappreciated in previous methods.

----

## [1288] ILSGAN: Independent Layer Synthesis for Unsupervised Foreground-Background Segmentation

**Authors**: *Qiran Zou, Yu Yang, Wing Yin Cheung, Chang Liu, Xiangyang Ji*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i9.26358](https://doi.org/10.1609/aaai.v37i9.26358)

**Abstract**:

Unsupervised foreground-background segmentation aims at extracting salient objects from cluttered backgrounds, where Generative Adversarial Network (GAN) approaches, especially layered GANs, show great promise. However, without human annotations, they are typically prone to produce foreground and background layers with non-negligible semantic and visual confusion, dubbed "information leakage", resulting in notable degeneration of the generated segmentation mask. To alleviate this issue, we propose a simple-yet-effective explicit layer independence modeling approach, termed Independent Layer Synthesis GAN (ILSGAN), pursuing independent foreground-background layer generation by encouraging their discrepancy. Specifically, it targets minimizing the mutual information between visible and invisible regions of the foreground and background to spur interlayer independence. Through in-depth theoretical and experimental analyses, we justify that explicit layer independence modeling is critical to suppressing information leakage and contributes to impressive segmentation performance gains. Also, our ILSGAN achieves strong state-of-the-art generation quality and segmentation performance on complex real-world data.

----

## [1289] SVP-T: A Shape-Level Variable-Position Transformer for Multivariate Time Series Classification

**Authors**: *Rundong Zuo, Guozhong Li, Byron Choi, Sourav S. Bhowmick, Daphne Ngar-yin Mah, Grace Lai-Hung Wong*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i9.26359](https://doi.org/10.1609/aaai.v37i9.26359)

**Abstract**:

Multivariate time series classiﬁcation (MTSC), one of the most fundamental time series applications, has not only gained substantial research attentions but has also emerged in many real-life applications. Recently, using transformers to solve MTSC has been reported. However, current transformer-based methods take data points of individual timestamps as inputs (timestamp-level), which only capture the temporal dependencies, not the dependencies among variables. In this
paper, we propose a novel method, called SVP-T. Specifically, we ﬁrst propose to take time series subsequences, which can be from different variables and positions (time interval), as the inputs (shape-level). The temporal and variable dependencies are both handled by capturing the long- and short-term dependencies among shapes. Second, we propose a variable-position encoding layer (VP-layer) to utilize both the variable and position information of each shape. Third, we introduce a novel VP-based (Variable-Position) self-attention mechanism to allow the enhancing the attention weights of overlapping shapes. We evaluate our method on all UEA MTS datasets. SVP-T achieves the best accuracy rank when compared with several competitive state-of-the-art methods. Furthermore, we demonstrate the effectiveness of the VP-layer and the VP-based self-attention mechanism. Finally, we present one case study to interpret the result of SVP-T.

----

## [1290] Mixed-Variable Black-Box Optimisation Using Value Proposal Trees

**Authors**: *Yan Zuo, Vu Nguyen, Amir Dezfouli, David Alexander, Benjamin Ward Muir, Iadine Chades*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i9.26360](https://doi.org/10.1609/aaai.v37i9.26360)

**Abstract**:

Many real-world optimisation problems are defined over both categorical and continuous variables, yet efficient optimisation methods such as Bayesian Optimisation (BO) are ill-equipped to handle such mixed-variable search spaces. The optimisation breadth introduced by categorical variables in the mixed-input setting has seen recent approaches operating on local trust regions, but these methods can be greedy in suboptimal regions of the search space. In this paper, we adopt a holistic view and aim to consolidate optimisation of the categorical and continuous sub-spaces under a single acquisition metric. We develop a tree-based method which retains a global view of the optimisation spaces by identifying regions in the search space with high potential candidates which we call value proposals. Our method uses these proposals to make selections on both the categorical and continuous components of the input. We show that this approach significantly outperforms existing mixed-variable optimisation approaches across several mixed-variable black-box optimisation tasks.

----

## [1291] Synchronization and Diversity of Solutions

**Authors**: *Emmanuel Arrighi, Henning Fernau, Mateus de Oliveira Oliveira, Petra Wolf*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i10.26361](https://doi.org/10.1609/aaai.v37i10.26361)

**Abstract**:

A central computational problem in the realm of automata theory is the problem of determining whether a finite automaton A has a synchronizing word. This problem has found applications in a variety of subfields of artificial intelligence, including planning, robotics, and multi-agent systems. In this work, we study this problem within the framework of diversity of solutions, an up-and-coming trend in the field of artificial intelligence where the goal is to compute a set of solutions that are sufficiently distinct from one another. We define a notion of diversity of solutions that is suitable for contexts were solutions are strings that may have distinct lengths. Using our notion of diversity, we show that for each fixed r ∈ N, each fixed finite automaton A, and each finite automaton B given at the input, the problem of determining the existence of a diverse set {w1,w2, . . . ,wr} ⊆ L(B) of words that are synchronizing for A can be solved in polynomial time. Finally, we generalize this result to the realm of conformant planning, where the goal is to devise plans that achieve a goal irrespectively of initial conditions and of nondeterminism that may occur during their execution.

----

## [1292] The Multi-Agent Transportation Problem

**Authors**: *Pascal Bachor, Rolf-David Bergdoll, Bernhard Nebel*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i10.26362](https://doi.org/10.1609/aaai.v37i10.26362)

**Abstract**:

We introduce the multi-agent transportation (MAT) problem, where agents have to transport containers from their starting positions to their designated goal positions. Movement takes place in a common environment where collisions between agents and between containers must be avoided.
In contrast to other frameworks such as multi-agent pathfinding (MAPF) or multi-agent pickup and delivery (MAPD), the agents are allowed to separate from the containers at any time, which can reduce the makespan and also allows for plans in scenarios that are unsolvable otherwise.
We present a complexity analysis establishing the problem's NP-completeness and show how the problem can be reduced to a sequence of SAT problems when optimizing for makespan.
A MAT solver is empirically evaluated with regard to varying input characteristics and movement constraints and compared to a MAPD solver that utilizes conflict-based search (CBS).

----

## [1293] Emergent Quantized Communication

**Authors**: *Boaz Carmeli, Ron Meir, Yonatan Belinkov*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i10.26363](https://doi.org/10.1609/aaai.v37i10.26363)

**Abstract**:

The field of emergent communication aims to understand the characteristics of communication as it emerges from artificial agents solving tasks that require information exchange. Communication with discrete messages is considered a desired characteristic, for scientific and applied reasons. However,  training a multi-agent system with discrete communication is not straightforward, requiring either reinforcement learning algorithms or relaxing the discreteness requirement via a continuous approximation such as the Gumbel-softmax. Both these solutions result in poor performance compared to fully continuous communication. In this work, we propose an alternative approach to achieve discrete communication -- quantization of communicated message. Using message quantization allows us to train the model end-to-end, achieving superior performance in multiple setups. Moreover, quantization is a natural framework that runs the gamut from continuous to discrete communication. Thus,  it sets the ground for a broader view of multi-agent communication in the deep learning era.

----

## [1294] Learning Explicit Credit Assignment for Cooperative Multi-Agent Reinforcement Learning via Polarization Policy Gradient

**Authors**: *Wubing Chen, Wenbin Li, Xiao Liu, Shangdong Yang, Yang Gao*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i10.26364](https://doi.org/10.1609/aaai.v37i10.26364)

**Abstract**:

Cooperative multi-agent policy gradient (MAPG) algorithms have recently attracted wide attention and are regarded as a general scheme for the multi-agent system. Credit assignment plays an important role in MAPG and can induce cooperation among multiple agents. However, most MAPG algorithms cannot achieve good credit assignment because of the game-theoretic pathology known as centralized-decentralized mismatch. To address this issue, this paper presents a novel method, Multi-Agent Polarization Policy Gradient (MAPPG). MAPPG takes a simple but efficient polarization function to transform the optimal consistency of joint and individual actions into easily realized constraints, thus enabling efficient credit assignment in MAPPG. Theoretically, we prove that individual policies of MAPPG can converge to the global optimum. Empirically, we evaluate MAPPG on the well-known matrix game and differential game, and verify that MAPPG can converge to the global optimum for both discrete and continuous action spaces. We also evaluate MAPPG on a set of StarCraft II micromanagement tasks and demonstrate that MAPPG outperforms the state-of-the-art MAPG algorithms.

----

## [1295] Zero-Shot Assistance in Sequential Decision Problems

**Authors**: *Sebastiaan De Peuter, Samuel Kaski*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i10.26365](https://doi.org/10.1609/aaai.v37i10.26365)

**Abstract**:

We consider the problem of creating assistants that can help agents solve new sequential decision problems, assuming the agent is not able to specify the reward function explicitly to the assistant. Instead of acting in place of the agent as in current automation-based approaches, we give the assistant an advisory role and keep the agent in the loop as the main decision maker. The difficulty is that we must account for potential biases of the agent which may cause it to seemingly irrationally reject advice. To do this we introduce a novel formalization of assistance that models these biases, allowing the assistant to infer and adapt to them. We then introduce a new method for planning the assistant's actions which can scale to large decision making problems. We show experimentally that our approach adapts to these agent biases, and results in higher cumulative reward for the agent than automation-based alternatives. Lastly, we show that an approach combining advice and automation outperforms advice alone at the cost of losing some safety guarantees.

----

## [1296] Multi-Unit Auctions for Allocating Chance-Constrained Resources

**Authors**: *Anna Gautier, Bruno Lacerda, Nick Hawes, Michael J. Wooldridge*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i10.26366](https://doi.org/10.1609/aaai.v37i10.26366)

**Abstract**:

Sharing scarce resources is a key challenge in multi-agent interaction, especially when individual agents are uncertain about their future consumption.  We present a new auction mechanism for preallocating multi-unit resources among agents, while limiting the chance of resource violations. By planning for a chance constraint, we strike a balance between worst-case approaches, which under-utilise resources, and expected-case approaches, which lack formal guarantees. We also present an algorithm that allows agents to generate bids via multi-objective reasoning, which are then submitted to the auction. We then discuss how the auction can be extended to non-cooperative scenarios. Finally, we demonstrate empirically that our auction outperforms state-of-the-art  techniques for chance-constrained multi-agent resource allocation in complex settings with up to hundreds of agents.

----

## [1297] Reward-Based Negotiating Agent Strategies

**Authors**: *Ryota Higa, Katsuhide Fujita, Toki Takahashi, Takumu Shimizu, Shinji Nakadai*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i10.26367](https://doi.org/10.1609/aaai.v37i10.26367)

**Abstract**:

This study proposed a novel reward-based negotiating agent strategy using an issue-based represented deep policy network. We compared the negotiation strategies with reinforcement learning (RL) by the tournaments toward heuristics-based champion agents in multi-issue negotiation. A bilateral multi-issue negotiation in which the two agents exchange offers in turn was considered. Existing RL architectures for a negotiation strategy incorporate rich utility function that provides concrete information even though the rewards of RL are considered as generalized signals in practice. Additionally, in existing reinforcement learning architectures for negotiation strategies, both the issue-based representations of the negotiation problems and the policy network to improve the scalability of negotiation domains are yet to be considered. This study proposed a novel reward-based negotiation strategy through deep RL by considering an issue-based represented deep policy network for multi-issue negotiation. Comparative studies analyzed the significant properties of negotiation strategies with RL. The results revealed that the policy-based learning agents with issue-based representations achieved comparable or higher utility than the state-of-the-art baselines with RL and heuristics, especially in the large-sized domains. Additionally, negotiation strategies with RL based on the policy network can achieve agreements by effectively using each step.

----

## [1298] Intersection Coordination with Priority-Based Search for Autonomous Vehicles

**Authors**: *Jiaoyang Li, The Anh Hoang, Eugene Lin, Hai L. Vu, Sven Koenig*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i10.26368](https://doi.org/10.1609/aaai.v37i10.26368)

**Abstract**:

The development of connected and autonomous vehicles opens an opportunity to manage intersections without signals. One promising approach is to use a central autonomous intersection manager to optimize the movement of the vehicles in the intersection. Existing work uses Mixed Integer Linear Programming (MILP) to find optimal solutions for this problem but is time-consuming and cannot be applied in real-time. On the other hand, the coordination of the vehicles is essentially a Multi-Agent Path Finding (MAPF) problem, for which dozens of efficient algorithms have been proposed in recent years. Inspired by these MAPF algorithms, we propose a three-level algorithm called PSL to solve the intersection coordination problem. Theoretically, PSL is complete and polynomial-time in the number of vehicles. Empirically, PSL runs significantly faster with only a slight compromise in the solution quality than the optimal MILP method. It also generates significantly better solutions with a slightly larger runtime than the traditional First-Come-First-Served strategy.

----

## [1299] Solving Large-Scale Pursuit-Evasion Games Using Pre-trained Strategies

**Authors**: *Shuxin Li, Xinrun Wang, Youzhi Zhang, Wanqi Xue, Jakub Cerný, Bo An*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i10.26369](https://doi.org/10.1609/aaai.v37i10.26369)

**Abstract**:

Pursuit-evasion games on graphs model the coordination of police forces chasing a fleeing felon in real-world urban settings, using the standard framework of imperfect-information extensive-form games (EFGs). In recent years, solving EFGs has been largely dominated by the Policy-Space Response Oracle (PSRO) methods due to their modularity, scalability, and favorable convergence properties. However, even these methods quickly reach their limits when facing large combinatorial strategy spaces of the pursuit-evasion games. To improve their efficiency, we integrate the pre-training and fine-tuning paradigm into the core module of PSRO -- the repeated computation of the best response. First, we pre-train the pursuer's policy base model against many different strategies of the evader. Then we proceed with the PSRO loop and fine-tune the pre-trained policy to attain the pursuer's best responses. The empirical evaluation shows that our approach significantly outperforms the baselines in terms of speed and scalability, and can solve even games on street maps of megalopolises with tens of thousands of crossroads -- a scale beyond the effective reach of previous methods.

----

## [1300] Contrastive Identity-Aware Learning for Multi-Agent Value Decomposition

**Authors**: *Shunyu Liu, Yihe Zhou, Jie Song, Tongya Zheng, Kaixuan Chen, Tongtian Zhu, Zunlei Feng, Mingli Song*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i10.26370](https://doi.org/10.1609/aaai.v37i10.26370)

**Abstract**:

Value Decomposition (VD) aims to deduce the contributions of agents for decentralized policies in the presence of only global rewards, and has recently emerged as a powerful credit assignment paradigm for tackling cooperative Multi-Agent Reinforcement Learning (MARL) problems. One of the main challenges in VD is to promote diverse behaviors among agents, while existing methods directly encourage the diversity of learned agent networks with various strategies. However, we argue that these dedicated designs for agent networks are still limited by the indistinguishable VD network, leading to homogeneous agent behaviors and thus downgrading the cooperation capability. In this paper, we propose a novel Contrastive Identity-Aware learning (CIA) method, explicitly boosting the credit-level distinguishability of the VD network to break the bottleneck of multi-agent diversity. Specifically, our approach leverages contrastive learning to maximize the mutual information between the temporal credits and identity representations of different agents, encouraging the full expressiveness of credit assignment and further the emergence of individualities. The algorithm implementation of the proposed CIA module is simple yet effective that can be readily incorporated into various VD architectures. Experiments on the SMAC benchmarks and across different VD backbones demonstrate that the proposed method yields results superior to the state-of-the-art counterparts. Our code is available at https://github.com/liushunyu/CIA.

----

## [1301] Learning to Shape Rewards Using a Game of Two Partners

**Authors**: *David Mguni, Taher Jafferjee, Jianhong Wang, Nicolas Perez Nieves, Wenbin Song, Feifei Tong, Matthew E. Taylor, Tianpei Yang, Zipeng Dai, Hui Chen, Jiangcheng Zhu, Kun Shao, Jun Wang, Yaodong Yang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i10.26371](https://doi.org/10.1609/aaai.v37i10.26371)

**Abstract**:

Reward shaping (RS) is a powerful method in reinforcement learning (RL) for overcoming the problem of sparse or uninformative rewards. However, RS typically relies on manually engineered shaping-reward functions whose construc- tion is time-consuming and error-prone. It also requires domain knowledge which runs contrary to the goal of autonomous learning. We introduce Reinforcement Learning Optimising Shaping Algorithm (ROSA), an automated reward shaping framework in which the shaping-reward function is constructed in a Markov game between two agents. A reward-shaping agent (Shaper) uses switching controls to determine which states to add shaping rewards for more efficient learning while the other agent (Controller) learns the optimal policy for the task using these shaped rewards. We prove that ROSA, which adopts existing RL algorithms, learns to construct a shaping-reward function that is beneficial to the task thus ensuring efficient convergence to high performance policies. We demonstrate ROSA’s properties in three didactic experiments and show its superior performance against state-of-the-art RS algorithms in challenging sparse reward environments.

----

## [1302] Reconstructing an Epidemic Outbreak Using Steiner Connectivity

**Authors**: *Ritwick Mishra, Jack Heavey, Gursharn Kaur, Abhijin Adiga, Anil Vullikanti*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i10.26372](https://doi.org/10.1609/aaai.v37i10.26372)

**Abstract**:

Only a subset of infections is actually observed in an outbreak, due to multiple reasons such as asymptomatic cases and under-reporting. Therefore, reconstructing an epidemic cascade given some observed cases is an important step in responding to such an outbreak. A maximum likelihood solution to this problem ( referred to as CascadeMLE ) can be shown to be a variation of the classical Steiner subgraph problem, which connects a subset of observed infections. In contrast to prior works on epidemic reconstruction, which consider the standard Steiner tree objective, we show that a solution to CascadeMLE, based on the actual MLE objective, has a very different structure. We design a logarithmic approximation algorithm for CascadeMLE, and evaluate it on multiple synthetic and social contact networks, including a contact network constructed for a hospital. Our algorithm has significantly better performance compared to a prior baseline.

----

## [1303] Formal Verification of Bayesian Mechanisms

**Authors**: *Munyque Mittelmann, Bastien Maubert, Aniello Murano, Laurent Perrussel*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i10.26373](https://doi.org/10.1609/aaai.v37i10.26373)

**Abstract**:

In this paper, for the first time, we study the formal verification of Bayesian mechanisms through strategic reasoning. We rely on the framework of Probabilistic Strategy Logic (PSL), which is well-suited for representing and verifying multi-agent systems with incomplete information. We take advantage of the recent results on the decidability of PSL model checking under memoryless strategies, and reduce the problem of formally verifying Bayesian mechanisms to PSL model checking. We show how to encode Bayesian-Nash equilibrium and economical properties, and illustrate our approach with different kinds of mechanisms.

----

## [1304] Memory-Augmented Theory of Mind Network

**Authors**: *Dung Nguyen, Phuoc Nguyen, Hung Le, Kien Do, Svetha Venkatesh, Truyen Tran*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i10.26374](https://doi.org/10.1609/aaai.v37i10.26374)

**Abstract**:

Social reasoning necessitates the capacity of theory of mind (ToM), the ability to contextualise and attribute mental states to others without having access to their internal cognitive structure. Recent machine learning approaches to ToM have demonstrated that we can train the observer to read the past and present behaviours of other agents and infer their beliefs (including false beliefs about things that no longer exist), goals, intentions and future actions. The challenges arise when the behavioural space is complex, demanding skilful space navigation for rapidly changing contexts for an extended period. We tackle the challenges by equipping the observer with novel neural memory mechanisms to encode, and hierarchical attention to selectively retrieve information about others. The memories allow rapid, selective querying of distal related past behaviours of others to deliberatively reason about their current mental state, beliefs and future behaviours. This results in ToMMY, a theory of mind model that learns to reason while making little assumptions about the underlying mental processes. We also construct a new suite of experiments to demonstrate that memories facilitate the learning process and achieve better theory of mind performance, especially for high-demand false-belief tasks that require inferring through multiple steps of changes.

----

## [1305] Socially Optimal Non-discriminatory Restrictions for Continuous-Action Games

**Authors**: *Michael Oesterle, Guni Sharon*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i10.26375](https://doi.org/10.1609/aaai.v37i10.26375)

**Abstract**:

We address the following mechanism design problem: Given a multi-player Normal-Form Game (NFG) with a continuous action space, find a non-discriminatory (i.e., identical for all players) restriction of the action space which maximizes the resulting Nash Equilibrium with respect to a fixed social utility function. First, we propose a formal model of a Restricted Game and the corresponding restriction optimization problem. We then present an algorithm to find optimal non-discriminatory restrictions under some assumptions. Our experimental results with Braess' Paradox and the Cournot Game show that this method leads to an optimized social utility of the Nash Equilibria, even when the assumptions are not guaranteed to hold. Finally, we outline a generalization of our approach to the much wider scope of Stochastic Games.

----

## [1306] Fault-Tolerant Offline Multi-Agent Path Planning

**Authors**: *Keisuke Okumura, Sébastien Tixeuil*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i10.26376](https://doi.org/10.1609/aaai.v37i10.26376)

**Abstract**:

We study a novel graph path planning problem for multiple agents that may crash at runtime, and block part of the workspace. In our setting, agents can detect neighboring crashed agents, and change followed paths at runtime. The objective is then to prepare a set of paths and switching rules for each agent, ensuring that all correct agents reach their destinations without collisions or deadlocks, despite unforeseen crashes of other agents. Such planning is attractive to build reliable multi-robot systems. We present problem formalization, theoretical analysis such as computational complexities, and how to solve this offline planning problem.

----

## [1307] LaCAM: Search-Based Algorithm for Quick Multi-Agent Pathfinding

**Authors**: *Keisuke Okumura*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i10.26377](https://doi.org/10.1609/aaai.v37i10.26377)

**Abstract**:

We propose a novel complete algorithm for multi-agent pathfinding (MAPF) called lazy constraints addition search for MAPF (LaCAM). MAPF is a problem of finding collision-free paths for multiple agents on graphs and is the foundation of multi-robot coordination. LaCAM uses a two-level search to find solutions quickly, even with hundreds of agents or more. At the low-level, it searches constraints about agents' locations. At the high-level, it searches a sequence of all agents' locations, following the constraints specified by the low-level. Our exhaustive experiments reveal that LaCAM is comparable to or outperforms state-of-the-art sub-optimal MAPF algorithms in a variety of scenarios, regarding success rate, planning time, and solution quality of sum-of-costs.

----

## [1308] Networked Anti-coordination Games Meet Graphical Dynamical Systems: Equilibria and Convergence

**Authors**: *Zirou Qiu, Chen Chen, Madhav V. Marathe, S. S. Ravi, Daniel J. Rosenkrantz, Richard Edwin Stearns, Anil Vullikanti*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i10.26378](https://doi.org/10.1609/aaai.v37i10.26378)

**Abstract**:

Evolutionary anti-coordination games on networks capture real-world strategic situations such as traffic routing and market competition. Two key problems concerning evolutionary games are the existence of a pure Nash equilibrium (NE) and the convergence time. In this work, we study these two problems for anti-coordination games under sequential and synchronous update schemes. For each update scheme, we examine two decision modes based on whether an agent considers its own previous action (self essential) or not (self non-essential) in choosing its next action. Using a relationship between games and dynamical systems, we show that for both update schemes, finding an NE can be done efficiently under the self non-essential mode but is computationally intractable under the self essential mode. We then identify special cases for which an NE can be obtained efficiently. For convergence time, we show that the dynamics converges in a polynomial number of steps under the synchronous scheme; for the sequential scheme, the convergence time is polynomial only under the self non-essential mode. Through experiments, we empirically examine the convergence time and the equilibria for both synthetic and real-world networks.

----

## [1309] Learning from Good Trajectories in Offline Multi-Agent Reinforcement Learning

**Authors**: *Qi Tian, Kun Kuang, Furui Liu, Baoxiang Wang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i10.26379](https://doi.org/10.1609/aaai.v37i10.26379)

**Abstract**:

Offline multi-agent reinforcement learning (MARL) aims to learn effective multi-agent policies from pre-collected datasets, which is an important step toward the deployment of multi-agent systems in real-world applications. However, in practice, each individual behavior policy that generates multi-agent joint trajectories usually has a different level of how well it performs. e.g., an agent is a random policy while other agents are medium policies. In the cooperative game with global reward, one agent learned by existing offline MARL often inherits this random policy, jeopardizing the utility of the entire team. In this paper, we investigate offline MARL with explicit consideration on the diversity of agent-wise trajectories and propose a novel framework called Shared Individual Trajectories (SIT) to address this problem. Specifically, an attention-based reward decomposition network assigns the credit to each agent through a differentiable key-value memory mechanism in an offline manner. These decomposed credits are then used to reconstruct the joint offline datasets into prioritized experience replay with individual trajectories, thereafter agents can share their good trajectories and conservatively train their policies with a graph attention network (GAT) based critic. We evaluate our method in both discrete control (i.e., StarCraft II and multi-agent particle environment) and continuous control (i.e., multi-agent mujoco). The results indicate that our method achieves significantly better results in complex and mixed offline multi-agent datasets, especially when the difference of data quality between individual trajectories is large.

----

## [1310] Resource Sharing through Multi-Round Matchings

**Authors**: *Yohai Trabelsi, Abhijin Adiga, Sarit Kraus, S. S. Ravi, Daniel J. Rosenkrantz*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i10.26380](https://doi.org/10.1609/aaai.v37i10.26380)

**Abstract**:

Applications such as employees sharing office spaces over a workweek
can be modeled as problems where agents are matched to resources
over multiple rounds. Agents' requirements limit the set of compatible
resources and the rounds in which they want to be matched.  Viewing such an
application as a multi-round matching problem on a bipartite compatibility
graph between agents and resources, we show that a  solution 
(i.e., a set of matchings, with one matching per round) can be found
efficiently if one exists.  To cope with situations where a solution does not exist, we consider two extensions. In
the first extension, a benefit function is defined for each agent and the
objective is to find a multi-round matching to maximize the total benefit.  For a
general class of benefit functions satisfying certain properties (including
diminishing returns), we show that this multi-round matching problem is
efficiently solvable.  This class includes utilitarian and Rawlsian welfare
functions.  
For another benefit function, we show that the maximization
problem is NP-hard.  
In the second extension, the objective is to generate advice to
each agent (i.e., a subset of requirements to be relaxed) subject to a
budget constraint so that the agent can be matched.
We show that this budget-constrained advice generation problem is NP-hard.
For this problem, we develop an integer linear programming formulation  as well
as a heuristic based on local search.
 We experimentally evaluate our algorithms on
synthetic networks and apply them to two real-world situations: shared
office spaces and matching courses to classrooms.

----

## [1311] Effective Integration of Weighted Cost-to-Go and Conflict Heuristic within Suboptimal CBS

**Authors**: *Rishi Veerapaneni, Tushar Kusnur, Maxim Likhachev*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i10.26381](https://doi.org/10.1609/aaai.v37i10.26381)

**Abstract**:

Conflict-Based Search (CBS) is a popular multi-agent path finding (MAPF) solver that employs a low-level single agent planner and a high-level constraint tree to resolve conflicts. The vast majority of modern MAPF solvers focus on improving CBS by reducing the size of this tree through various strategies with few methods modifying the low level planner. Typically low level planners in existing CBS methods use an unweighted cost-to-go heuristic, with suboptimal CBS methods also using a conflict heuristic to help the high level search. In this paper, we show that, contrary to prevailing CBS beliefs, a weighted cost-to-go heuristic can be used effectively alongside the conflict heuristic in two possible variants. In particular, one of these variants can obtain large speedups, 2-100x, across several  scenarios and suboptimal CBS methods. Importantly, we discover that performance is related not to the weighted cost-to-go heuristic but rather to the relative conflict heuristic weight's ability to effectively balance low-level and high-level work. Additionally, to the best of our knowledge, we show the first theoretical relation of prioritized planning and bounded suboptimal CBS and demonstrate that our methods are their natural generalization.

----

## [1312] DM²: Decentralized Multi-Agent Reinforcement Learning via Distribution Matching

**Authors**: *Caroline Wang, Ishan Durugkar, Elad Liebman, Peter Stone*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i10.26382](https://doi.org/10.1609/aaai.v37i10.26382)

**Abstract**:

Current approaches to multi-agent cooperation rely heavily on centralized mechanisms or explicit communication protocols to ensure convergence. This paper studies the problem of distributed multi-agent learning without resorting to centralized components or explicit communication. It examines the use of distribution matching to facilitate the coordination of independent agents. In the proposed scheme, each agent independently minimizes the distribution mismatch to the corresponding component of a target visitation distribution. The theoretical  analysis shows that under certain conditions, each agent minimizing its individual  distribution mismatch allows the convergence to the joint policy that generated the target distribution. Further, if the target distribution is from a joint policy that optimizes a cooperative task, the optimal policy for a combination of this task reward and the distribution matching reward is the same joint policy. This insight is used to formulate a practical algorithm (DM^2), in which each individual agent matches a target distribution derived from concurrently sampled trajectories from a joint expert policy. Experimental validation on the StarCraft domain shows that combining (1) a task reward, and (2) a distribution matching reward for expert demonstrations for the same task, allows agents to outperform a naive distributed baseline. Additional experiments probe the conditions under which expert demonstrations need to be sampled to obtain the learning benefits.

----

## [1313] Emergence of Punishment in Social Dilemma with Environmental Feedback

**Authors**: *Zhen Wang, Zhao Song, Chen Shen, Shuyue Hu*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i10.26383](https://doi.org/10.1609/aaai.v37i10.26383)

**Abstract**:

Altruistic punishment (or punishment) has been extensively shown as an important mechanism for promoting cooperation in human societies. In AI, the emergence of punishment has received much recent interest. In this paper, we contribute with a novel evolutionary game theoretic model to study the impacts of environmental feedback. Whereas a population of agents plays public goods games, there exists a third-party population whose payoffs depend not only on whether to punish or not, but also on the state of the environment (e.g., how cooperative the agents in a social dilemma are). Focusing on one-shot public goods games, we show that environmental feedback, by itself, can lead to the emergence of punishment. We analyze the co-evolution of punishment and cooperation, and derive conditions for their co-presence, co-dominance and co-extinction. Moreover, we show that the system can exhibit bistability as well as cyclic dynamics. Our findings provide a new explanation for the emergence of punishment.  On the other hand, our results also alert the need for careful design of implementing punishment in multi-agent systems, as the resulting evolutionary dynamics can be somewhat complex.

----

## [1314] Subspace-Aware Exploration for Sparse-Reward Multi-Agent Tasks

**Authors**: *Pei Xu, Junge Zhang, Qiyue Yin, Chao Yu, Yaodong Yang, Kaiqi Huang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i10.26384](https://doi.org/10.1609/aaai.v37i10.26384)

**Abstract**:

Exploration under sparse rewards is a key challenge for multi-agent reinforcement learning problems. One possible solution to this issue is to exploit inherent task structures for an acceleration of exploration. In this paper, we present a novel exploration approach, which encodes a special structural prior on the reward function into exploration, for sparse-reward multi-agent tasks. Specifically, a novel entropic exploration objective which encodes the structural prior is proposed to accelerate the discovery of rewards. By maximizing the lower bound of this objective, we then propose an algorithm with moderate computational cost, which can be applied to practical tasks. Under the sparse-reward setting, we show that the proposed algorithm significantly outperforms the state-of-the-art algorithms in the multiple-particle environment, the Google Research Football and StarCraft II micromanagement tasks. To the best of our knowledge, on some hard tasks (such as 27m_vs_30m}) which have relatively larger number of agents and need non-trivial strategies to defeat enemies, our method is the first to learn winning strategies under the sparse-reward setting.

----

## [1315] Consensus Learning for Cooperative Multi-Agent Reinforcement Learning

**Authors**: *Zhiwei Xu, Bin Zhang, Dapeng Li, Zeren Zhang, Guangchong Zhou, Hao Chen, Guoliang Fan*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i10.26385](https://doi.org/10.1609/aaai.v37i10.26385)

**Abstract**:

Almost all multi-agent reinforcement learning algorithms without communication follow the principle of centralized training with decentralized execution. During the centralized training, agents can be guided by the same signals, such as the global state. However, agents lack the shared signal and choose actions given local observations during execution. Inspired by viewpoint invariance and contrastive learning, we propose consensus learning for cooperative multi-agent reinforcement learning in this study. Although based on local observations, different agents can infer the same consensus in discrete spaces without communication. We feed the inferred one-hot consensus to the network of agents as an explicit input in a decentralized way, thereby fostering their cooperative spirit. With minor model modifications, our suggested framework can be extended to a variety of multi-agent reinforcement learning algorithms. Moreover, we carry out these variants on some fully cooperative tasks and get convincing results.

----

## [1316] HAVEN: Hierarchical Cooperative Multi-Agent Reinforcement Learning with Dual Coordination Mechanism

**Authors**: *Zhiwei Xu, Yunpeng Bai, Bin Zhang, Dapeng Li, Guoliang Fan*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i10.26386](https://doi.org/10.1609/aaai.v37i10.26386)

**Abstract**:

Recently, some challenging tasks in multi-agent systems have been solved by some hierarchical reinforcement learning methods. Inspired by the intra-level and inter-level coordination in the human nervous system, we propose a novel value decomposition framework HAVEN based on hierarchical reinforcement learning for fully cooperative multi-agent problems. To address the instability arising from the concurrent optimization of policies between various levels and agents, we introduce the dual coordination mechanism of inter-level and inter-agent strategies by designing reward functions in a two-level hierarchy. HAVEN does not require domain knowledge and pre-training, and can be applied to any value decomposition variant. Our method achieves desirable results on different decentralized partially observable Markov decision process domains and outperforms other popular multi-agent hierarchical reinforcement learning algorithms.

----

## [1317] Hierarchical Mean-Field Deep Reinforcement Learning for Large-Scale Multiagent Systems

**Authors**: *Chao Yu*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i10.26387](https://doi.org/10.1609/aaai.v37i10.26387)

**Abstract**:

Learning for efficient coordination in large-scale multiagent systems suffers from the problem of the curse of dimensionality due to the exponential growth of agent interactions. Mean-Field (MF)-based methods address this issue by transforming the interactions within the whole system into a single agent played with the average effect of its neighbors. However, considering the neighbors merely by their average may ignore the varying influences of each neighbor, and learning with this kind of local average effect would likely lead to inferior system performance due to lack of an efficient coordination mechanism in the whole population level. In this work, we propose a Hierarchical Mean-Field (HMF) learning framework to further improve the performance of existing MF methods. The basic idea is to approximate the average effect for a sub-group of agents by considering their different influences within the sub-group, and realize population-level coordination through the interactions among different sub-groups. Empirical studies show that HMF significantly outperforms existing baselines on both challenging cooperative and mixed cooperative-competitive tasks with different scales of agent populations.

----

## [1318] Robust Multi-Agent Coordination via Evolutionary Generation of Auxiliary Adversarial Attackers

**Authors**: *Lei Yuan, Ziqian Zhang, Ke Xue, Hao Yin, Feng Chen, Cong Guan, Lihe Li, Chao Qian, Yang Yu*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i10.26388](https://doi.org/10.1609/aaai.v37i10.26388)

**Abstract**:

Cooperative Multi-agent Reinforcement Learning (CMARL) has shown to be promising for many real-world applications. Previous works mainly focus on improving coordination ability via solving MARL-specific challenges (e.g., non-stationarity, credit assignment, scalability), but ignore the policy perturbation issue when testing in a different environment. This issue hasn't been considered in problem formulation or efficient algorithm design. To address this issue, we firstly model the problem as a Limited Policy Adversary Dec-POMDP (LPA-Dec-POMDP), where some coordinators from a team might accidentally and unpredictably encounter a limited number of malicious action attacks, but the regular coordinators still strive for the intended goal. Then, we propose Robust Multi-Agent Coordination via Evolutionary Generation of Auxiliary Adversarial Attackers (ROMANCE), which enables the trained policy to encounter diversified and strong auxiliary adversarial attacks during training, thus achieving high robustness under various 
policy perturbations. Concretely, to avoid the ego-system overfitting to a specific attacker, we maintain a set of attackers, which is optimized to guarantee the attackers high attacking quality and behavior diversity. The goal of quality is to minimize the ego-system coordination effect, and a novel diversity regularizer based on sparse action is applied to diversify the behaviors among attackers. The ego-system is then paired with a population of attackers selected from the maintained attacker set, and alternately trained against the constantly evolving attackers. Extensive experiments on multiple scenarios from SMAC indicate our ROMANCE provides comparable or better robustness and generalization ability than other baselines.

----

## [1319] DACOM: Learning Delay-Aware Communication for Multi-Agent Reinforcement Learning

**Authors**: *Tingting Yuan, Hwei-Ming Chung, Jie Yuan, Xiaoming Fu*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i10.26389](https://doi.org/10.1609/aaai.v37i10.26389)

**Abstract**:

Communication is supposed to improve multi-agent collaboration and overall performance in cooperative Multi-agent reinforcement learning (MARL). However, such improvements are prevalently limited in practice since most existing communication schemes ignore communication overheads (e.g., communication delays). In this paper, we demonstrate that ignoring communication delays has detrimental effects on collaborations, especially in delay-sensitive tasks such as autonomous driving. To mitigate this impact, we design a delay-aware multi-agent communication model (DACOM) to adapt communication to delays. Specifically, DACOM introduces a component, TimeNet, that is responsible for adjusting the waiting time of an agent to receive messages from other agents such that the uncertainty associated with delay can be addressed. Our experiments reveal that DACOM has a non-negligible performance improvement over other mechanisms by making a better trade-off between the benefits of communication and the costs of waiting for messages.

----

## [1320] Effective and Stable Role-Based Multi-Agent Collaboration by Structural Information Principles

**Authors**: *Xianghua Zeng, Hao Peng, Angsheng Li*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i10.26390](https://doi.org/10.1609/aaai.v37i10.26390)

**Abstract**:

Role-based learning is a promising approach to improving the performance of Multi-Agent Reinforcement Learning (MARL). Nevertheless, without manual assistance, current role-based methods cannot guarantee stably discovering a set of roles to effectively decompose a complex task, as they assume either a predefined role structure or practical experience for selecting hyperparameters. In this article, we propose a mathematical Structural Information principles-based Role Discovery method, namely SIRD, and then present a SIRD optimizing MARL framework, namely SR-MARL, for multi-agent collaboration. The SIRD transforms role discovery into a hierarchical action space clustering. Specifically, the SIRD consists of structuralization, sparsification, and optimization modules, where an optimal encoding tree is generated to perform abstracting to discover roles. The SIRD is agnostic to specific MARL algorithms and flexibly integrated with various value function factorization approaches. Empirical evaluations on the StarCraft II micromanagement benchmark demonstrate that, compared with state-of-the-art MARL algorithms, the SR-MARL framework improves the average test win rate by 0.17%, 6.08%, and 3.24%, and reduces the deviation by 16.67%, 30.80%, and 66.30%, under easy, hard, and super hard scenarios.

----

## [1321] Learning to Play General-Sum Games against Multiple Boundedly Rational Agents

**Authors**: *Eric Zhao, Alexander R. Trott, Caiming Xiong, Stephan Zheng*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i10.26391](https://doi.org/10.1609/aaai.v37i10.26391)

**Abstract**:

We study the problem of training a principal in a multi-agent general-sum game using reinforcement learning (RL). Learning a robust principal policy requires anticipating the worst possible strategic responses of other agents, which is generally NP-hard. However, we show that no-regret dynamics can identify these worst-case responses in poly-time in smooth games. We propose a framework that uses this policy evaluation method for efficiently learning a robust principal policy using RL. This framework can be extended to provide robustness to boundedly rational agents too. Our motivating application is automated mechanism design: we empirically demonstrate our framework learns robust mechanisms in both matrix games and complex spatiotemporal games. In particular, we learn a dynamic tax policy that improves the welfare of a simulated trade-and-barter economy by 15%, even when facing previously unseen boundedly rational RL taxpayers.

----

## [1322] Towards Robust Metrics for Concept Representation Evaluation

**Authors**: *Mateo Espinosa Zarlenga, Pietro Barbiero, Zohreh Shams, Dmitry Kazhdan, Umang Bhatt, Adrian Weller, Mateja Jamnik*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i10.26392](https://doi.org/10.1609/aaai.v37i10.26392)

**Abstract**:

Recent work on interpretability has focused on concept-based explanations, where deep learning models are explained in terms of high-level units of information, referred to as concepts. Concept learning models, however, have been shown to be prone to encoding impurities in their representations, failing to fully capture meaningful features of their inputs. While concept learning lacks metrics to measure such phenomena, the field of disentanglement learning has explored the related notion of underlying factors of variation in the data, with plenty of metrics to measure the purity of such factors. In this paper, we show that such metrics are not appropriate for concept learning and propose novel metrics for evaluating the purity of concept representations in both approaches. We show the advantage of these metrics over existing ones and demonstrate their utility in evaluating the robustness of concept representations and interventions performed on them. In addition, we show their utility for benchmarking state-of-the-art methods from both families and find that, contrary to common assumptions, supervision alone may not be sufficient for pure concept representations.

----

## [1323] On the Vulnerability of Backdoor Defenses for Federated Learning

**Authors**: *Pei Fang, Jinghui Chen*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i10.26393](https://doi.org/10.1609/aaai.v37i10.26393)

**Abstract**:

Federated learning (FL) is a popular distributed machine learning paradigm which enables jointly training a global model without sharing clients' data. However, its repetitive server-client communication gives room for possible backdoor attacks which aims to mislead the global model into a targeted misprediction when a specific trigger pattern is presented. In response to such backdoor threats on federated learning, various defense measures have been proposed. In this paper, we study whether the current defense mechanisms truly neutralize the backdoor threats from federated learning in a practical setting by proposing a new federated backdoor attack framework for possible countermeasures. Different from traditional training (on triggered data) and rescaling (the malicious client model) based backdoor injection, the proposed backdoor attack framework (1) directly modifies (a small proportion of) local model weights to inject the backdoor trigger via sign flips; (2) jointly optimize the trigger pattern with the client model,  thus is more persistent and stealthy for circumventing existing defenses. In a case study, we examine the strength and weaknesses of several recent federated backdoor defenses from three major categories and provide suggestions to the practitioners when training federated models in practice.

----

## [1324] Distributionally Robust Optimization with Probabilistic Group

**Authors**: *Soumya Suvra Ghosal, Yixuan Li*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i10.26394](https://doi.org/10.1609/aaai.v37i10.26394)

**Abstract**:

Modern machine learning models may be susceptible to learning spurious correlations that hold on average but not for the atypical group of samples. To address the problem, previous approaches minimize the empirical worst-group risk. Despite the promise, they often assume that each sample belongs to one and only one group, which does not allow expressing the uncertainty in group labeling. In this paper, we propose a novel framework PG-DRO, which explores the idea of probabilistic group membership for distributionally robust optimization. Key to our framework, we consider soft group membership instead of hard group annotations. The group probabilities can be flexibly generated using either supervised learning or zero-shot approaches. Our framework accommodates samples with group membership ambiguity, offering stronger flexibility and generality than the prior art. We comprehensively evaluate PG-DRO on both image classification and natural language processing benchmarks, establishing superior performance.

----

## [1325] Correct for Whom? Subjectivity and the Evaluation of Personalized Image Aesthetics Assessment Models

**Authors**: *Samuel Goree, Weslie Khoo, David J. Crandall*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i10.26395](https://doi.org/10.1609/aaai.v37i10.26395)

**Abstract**:

The problem of image aesthetic quality assessment is surprisingly difficult to define precisely. Most early work attempted to estimate the average aesthetic rating of a group of observers, while some recent work has shifted to an approach based on few-shot personalization. In this paper, we connect few-shot personalization, via Immanuel Kant's concept of disinterested judgment, to an argument from feminist aesthetics about the biased tendencies of objective standards for subjective pleasures. To empirically investigate this philosophical debate, we introduce PR-AADB, a relabeling of the existing AADB dataset with labels for pairs of images, and measure how well the existing groundtruth predicts our new pairwise labels. We find, consistent with the feminist critique, that both the existing groundtruth and few-shot personalized predictions represent some users' preferences significantly better than others, but that it is difficult to predict when and for whom the existing groundtruth will be correct. We thus advise against using benchmark datasets to evaluate models for personalized IAQA, and recommend caution when attempting to account for subjective difference using machine learning more generally.

----

## [1326] Covariate-Shift Generalization via Random Sample Weighting

**Authors**: *Yue He, Xinwei Shen, Renzhe Xu, Tong Zhang, Yong Jiang, Wenchao Zou, Peng Cui*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i10.26396](https://doi.org/10.1609/aaai.v37i10.26396)

**Abstract**:

Shifts in the marginal distribution of covariates from training to the test phase, named covariate-shifts, often lead to unstable prediction performance across agnostic testing data, especially under model misspecification. Recent literature on invariant learning attempts to learn an invariant predictor from heterogeneous environments. However, the performance of the learned predictor depends heavily on the availability and quality of provided environments. In this paper, we propose a simple and effective non-parametric method for generating heterogeneous environments via Random Sample Weighting (RSW). Given the training dataset from a single source environment, we randomly generate a set of covariate-determining sample weights and use each weighted training distribution to simulate an environment. We theoretically show that under appropriate conditions, such random sample weighting can produce sufficient heterogeneity to be exploited by common invariance constraints to find the invariant variables for stable prediction under covariate shifts. Extensive experiments on both simulated and real-world datasets clearly validate the effectiveness of our method.

----

## [1327] Fairness in Contextual Resource Allocation Systems: Metrics and Incompatibility Results

**Authors**: *Nathanael Jo, Bill Tang, Kathryn Dullerud, Sina Aghaei, Eric Rice, Phebe Vayanos*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i10.26397](https://doi.org/10.1609/aaai.v37i10.26397)

**Abstract**:

We study critical systems that allocate scarce resources to satisfy basic needs, such as homeless services that provide housing. These systems often support communities disproportionately affected by systemic racial, gender, or other injustices, so it is crucial to design these systems with fairness considerations in mind. To address this problem, we propose a framework for evaluating fairness in contextual resource allocation systems that is inspired by fairness metrics in machine learning. This framework can be applied to evaluate the fairness properties of a historical policy, as well as to impose constraints in the design of new (counterfactual) allocation policies. Our work culminates with a set of incompatibility results that investigate the interplay between the different fairness metrics we propose. Notably, we demonstrate that: 1) fairness in allocation and fairness in outcomes are usually incompatible; 2) policies that prioritize based on a vulnerability score will usually result in unequal outcomes across groups, even if the score is perfectly calibrated; 3) policies using contextual information beyond what is needed to characterize baseline risk and treatment effects can be fairer in their outcomes than those using just baseline risk and treatment effects; and 4) policies using group status in addition to baseline risk and treatment effects are as fair as possible given all available information. Our framework can help guide the discussion among stakeholders in deciding which fairness metrics to impose when allocating scarce resources.

----

## [1328] Improvement-Focused Causal Recourse (ICR)

**Authors**: *Gunnar König, Timo Freiesleben, Moritz Grosse-Wentrup*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i10.26398](https://doi.org/10.1609/aaai.v37i10.26398)

**Abstract**:

Algorithmic recourse recommendations inform stakeholders of how to act to revert unfavorable decisions. However, existing methods may recommend actions that lead to acceptance (i.e., revert the model's decision) but do not lead to improvement (i.e., may not revert the underlying real-world state). To recommend such actions is to recommend fooling the predictor. We introduce a novel method, Improvement-Focused Causal Recourse (ICR), which involves a conceptual shift: Firstly, we require ICR recommendations to guide toward improvement. Secondly, we do not tailor the recommendations to be accepted by a specific predictor.  Instead, we leverage causal knowledge to design decision systems that predict accurately pre- and post-recourse, such that improvement guarantees translate into acceptance guarantees. Curiously, optimal pre-recourse classifiers are robust to ICR actions and thus suitable post-recourse. In semi-synthetic experiments, we demonstrate that given correct causal knowledge ICR, in contrast to existing approaches, guides toward both acceptance and improvement.

----

## [1329] Explaining Model Confidence Using Counterfactuals

**Authors**: *Thao Le, Tim Miller, Ronal Singh, Liz Sonenberg*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i10.26399](https://doi.org/10.1609/aaai.v37i10.26399)

**Abstract**:

Displaying confidence scores in human-AI interaction has been shown to help build trust between humans and AI systems. However, most existing research uses only the confidence score as a form of communication. As confidence scores are just another model output, users may want to understand why the algorithm is confident to determine whether to accept the confidence score. In this paper, we show that counterfactual explanations of confidence scores help study participants to better understand and better trust a machine learning model's prediction. We present two methods for understanding model confidence using counterfactual explanation: (1) based on counterfactual examples; and (2) based on visualisation of the counterfactual space. Both increase understanding and trust for study participants over a baseline of no explanation, but qualitative results show that they are used quite differently, leading to recommendations of when to use each one and directions of designing better explanations.

----

## [1330] Echo of Neighbors: Privacy Amplification for Personalized Private Federated Learning with Shuffle Model

**Authors**: *Yixuan Liu, Suyun Zhao, Li Xiong, Yuhan Liu, Hong Chen*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i10.26400](https://doi.org/10.1609/aaai.v37i10.26400)

**Abstract**:

Federated Learning, as a popular paradigm for collaborative training, is vulnerable against privacy attacks. Different privacy levels regarding users' attitudes need to be satisfied locally, while a strict privacy guarantee for the global model is also required centrally. Personalized Local Differential Privacy (PLDP) is suitable for preserving users' varying local privacy, yet only provides a central privacy guarantee equivalent to the worst-case local privacy level. Thus, achieving strong central privacy as well as personalized local privacy with a utility-promising model is a challenging problem. In this work, a general framework (APES) is built up to strengthen model privacy under personalized local privacy by leveraging the privacy amplification effect of the shuffle model.  To tighten the privacy bound, we quantify the heterogeneous contributions to the central privacy user by user. The contributions are characterized by the ability of generating “echos” from the perturbation of each user,  which is carefully measured by proposed methods Neighbor Divergence and Clip-Laplace Mechanism. Furthermore, we propose a refined framework (S-APES) with the post-sparsification technique to reduce privacy loss in high-dimension scenarios. To the best of our knowledge, the impact of shuffling on personalized local privacy is considered for the first time. We provide a strong privacy amplification effect, and the bound is tighter than the baseline result based on existing methods for uniform local privacy. Experiments demonstrate that our frameworks ensure comparable or higher accuracy for the global model.

----

## [1331] XRand: Differentially Private Defense against Explanation-Guided Attacks

**Authors**: *Truc D. T. Nguyen, Phung Lai, Hai Phan, My T. Thai*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i10.26401](https://doi.org/10.1609/aaai.v37i10.26401)

**Abstract**:

Recent development in the field of explainable artificial intelligence (XAI) has helped improve trust in Machine-Learning-as-a-Service (MLaaS) systems, in which an explanation is provided together with the model prediction in response to each query. However, XAI also opens a door for adversaries to gain insights into the black-box models in MLaaS, thereby making the models more vulnerable to several attacks. For example, feature-based explanations (e.g., SHAP) could expose the top important features that a black-box model focuses on. Such disclosure has been exploited to craft effective backdoor triggers against malware classifiers. To address this trade-off, we introduce a new concept of achieving local differential privacy (LDP) in the explanations, and from that we establish a defense, called XRand, against such attacks. We show that our mechanism restricts the information that the adversary can learn about the top important features, while maintaining the faithfulness of the explanations.

----

## [1332] Mitigating Adversarial Norm Training with Moral Axioms

**Authors**: *Taylor Olson, Kenneth D. Forbus*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i10.26402](https://doi.org/10.1609/aaai.v37i10.26402)

**Abstract**:

This paper addresses the issue of adversarial attacks on ethical AI systems. We investigate using moral axioms and rules of deontic logic in a norm learning framework to mitigate adversarial norm training. This model of moral intuition and construction provides AI systems with moral guard rails yet still allows for learning conventions. We evaluate our approach by drawing inspiration from a study commonly used in moral development research. This questionnaire aims to test an agent's ability to reason to moral conclusions despite opposed testimony. Our findings suggest that our model can still correctly evaluate moral situations and learn conventions in an adversarial training environment. We conclude that adding axiomatic moral prohibitions and deontic inference rules to a norm learning model makes it less vulnerable to adversarial attacks.

----

## [1333] Equity Promotion in Public Transportation

**Authors**: *Anik Pramanik, Pan Xu, Yifan Xu*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i10.26403](https://doi.org/10.1609/aaai.v37i10.26403)

**Abstract**:

There are many news articles reporting the obstacles confronting poverty-stricken households in access to public transits. These barriers create a great deal of inconveniences for these impoverished families and more importantly, they contribute a lot of social inequalities. A typical approach addressing the issue is to build more transport infrastructure to offer more opportunities to access the public transits especially for those deprived communities. Examples include adding more bus lines connecting needy residents to railways systems and extending existing bus lines to areas with low socioeconomic status. Recently, a new strategy is proposed, which is to harness the ubiquitous ride-hailing services to connect disadvantaged households with the nearest public transportations. Compared with the former infrastructure-based solution, the ride-hailing-based strategy enjoys a few exclusive benefits such as higher effectiveness and more flexibility.

In this paper, we propose an optimization model to study how to integrate the two approaches together for equity-promotion purposes. Specifically, we aim to design a strategy of allocating a given limited budget to different candidate programs such that the overall social equity is maximized, which is defined as the minimum covering ratio among all pre-specified protected groups of households (based on race, income, etc.). We have designed a linear-programming (LP) based rounding algorithm, which proves to achieve an optimal approximation ratio of 1-1/e. Additionally, we test our algorithm against a few baselines on real data assembled by outsourcing multiple public datasets collected in the city of Chicago. Experimental results confirm our theoretical predictions and demonstrate the effectiveness of our LP-based strategy in promoting social equity, especially when the budget is insufficient.

----

## [1334] Online Platforms and the Fair Exposure Problem under Homophily

**Authors**: *Jakob Schoeffer, Alexander Ritchie, Keziah Naggita, Faidra Monachou, Jessica Finocchiaro, Marc Juarez*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i10.26404](https://doi.org/10.1609/aaai.v37i10.26404)

**Abstract**:

In the wake of increasing political extremism, online platforms have been criticized for contributing to polarization. One line of criticism has focused on echo chambers and the recommended content served to users by these platforms. In this work, we introduce the fair exposure problem: given limited intervention power of the platform, the goal is to enforce balance in the spread of content (e.g., news articles) among two groups of users through constraints similar to those imposed by the Fairness Doctrine in the United States in the past. Groups are characterized by different affiliations (e.g., political views) and have different preferences for content. We develop a stylized framework that models intra- and inter-group content propagation under homophily, and we formulate the platform's decision as an optimization problem that aims at maximizing user engagement, potentially under fairness constraints. Our main notion of fairness requires that each group see a mixture of their preferred and non-preferred content, encouraging information diversity. Promoting such information diversity is often viewed as desirable and a potential means for breaking out of harmful echo chambers. We study the solutions to both the fairness-agnostic and fairness-aware problems. We prove that a fairness-agnostic approach inevitably leads to group-homogeneous targeting  by the platform. This is only partially mitigated by imposing fairness constraints: we show that there exist optimal fairness-aware solutions which target one group with different types of content and the other group with only one type that is not necessarily the group's most preferred. Finally, using simulations with real-world data, we study the system dynamics and quantify the price of fairness.

----

## [1335] Minimax AUC Fairness: Efficient Algorithm with Provable Convergence

**Authors**: *Zhenhuan Yang, Yan Lok Ko, Kush R. Varshney, Yiming Ying*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i10.26405](https://doi.org/10.1609/aaai.v37i10.26405)

**Abstract**:

The use of machine learning models in consequential decision making often exacerbates societal inequity, in particular yielding disparate impact on members of marginalized groups defined by race and gender. The area under the ROC curve (AUC) is widely used to evaluate the performance of a scoring function in machine learning, but is studied in algorithmic fairness less than other performance metrics. Due to the pairwise nature of the AUC, defining an AUC-based group fairness metric is pairwise-dependent and may involve both intra-group and inter-group AUCs. Importantly, considering only one category of AUCs is not sufficient to mitigate unfairness in AUC optimization. In this paper, we propose a minimax learning and bias mitigation framework that incorporates both intra-group and inter-group AUCs while maintaining utility. Based on this Rawlsian framework, we design an efficient stochastic optimization algorithm and prove its convergence to the minimum group-level AUC. We conduct numerical experiments on both synthetic and real-world datasets to validate the effectiveness of the minimax framework and the proposed optimization algorithm.

----

## [1336] Faster Fair Machine via Transferring Fairness Constraints to Virtual Samples

**Authors**: *Zhou Zhai, Lei Luo, Heng Huang, Bin Gu*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i10.26406](https://doi.org/10.1609/aaai.v37i10.26406)

**Abstract**:

Fair classification is an emerging and important research topic in machine learning community. Existing methods usually formulate the fairness metrics as additional inequality constraints, and then embed them into the original objective. This makes fair classification problems unable to be effectively tackled by some solvers specific to unconstrained optimization. Although many new tailored algorithms have been designed to attempt to overcome this limitation, they often increase additional computation burden and cannot cope with all types of fairness metrics. To address these challenging issues, in this paper, we propose a novel method for fair classification. 
Specifically, we theoretically
demonstrate that all types of fairness with linear and non-linear covariance functions can be transferred to two virtual samples, which makes the existing state-of-the-art classification solvers be applicable to these cases. Meanwhile, we  generalize the proposed method to multiple fairness constraints. We take SVM as an example to show the effectiveness of our new idea. 
Empirically, we test the proposed method on real-world datasets and all results confirm its excellent performance.

----

## [1337] Learning Control Policies for Stochastic Systems with Reach-Avoid Guarantees

**Authors**: *Dorde Zikelic, Mathias Lechner, Thomas A. Henzinger, Krishnendu Chatterjee*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i10.26407](https://doi.org/10.1609/aaai.v37i10.26407)

**Abstract**:

We study the problem of learning controllers for discrete-time non-linear stochastic dynamical systems with formal reach-avoid guarantees. This work presents the first method for providing formal reach-avoid guarantees, which combine and generalize stability and safety guarantees, with a tolerable probability threshold p in [0,1] over the infinite time horizon. Our method leverages advances in machine learning literature and it represents formal certificates as neural networks. In particular, we learn a certificate in the form of a reach-avoid supermartingale (RASM), a novel notion that we introduce in this work. Our RASMs provide reachability and avoidance guarantees by imposing constraints on what can be viewed as a stochastic extension of level sets of Lyapunov functions for deterministic systems. Our approach solves several important problems -- it can be used to learn a control policy from scratch, to verify a reach-avoid specification for a fixed control policy, or to fine-tune a pre-trained policy if it does not satisfy the reach-avoid specification. We validate our approach on 3 stochastic non-linear reinforcement learning tasks.

----

## [1338] Robust Neuro-Symbolic Goal and Plan Recognition

**Authors**: *Leonardo Amado, Ramon Fraga Pereira, Felipe Meneguzzi*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i10.26408](https://doi.org/10.1609/aaai.v37i10.26408)

**Abstract**:

Goal Recognition is the task of discerning the intended goal of an agent given a sequence of observations, whereas Plan Recognition consists of identifying the plan to achieve such intended goal. Regardless of the underlying techniques, most recognition approaches are directly affected by the quality of the available observations. In this paper, we develop neuro-symbolic recognition approaches that can combine learning and planning techniques, compensating for noise and missing observations using prior data. We evaluate our approaches in standard human-designed planning domains as well as domain models automatically learned from real-world data. Empirical experimentation shows that our approaches reliably infer goals and compute correct plans in the experimental datasets. An ablation study shows that outperform approaches that rely exclusively on the domain model, or exclusively on machine learning in problems with both noisy observations and low observability.

----

## [1339] Heuristic Search for Multi-Objective Probabilistic Planning

**Authors**: *Dillon Ze Chen, Felipe W. Trevizan, Sylvie Thiébaux*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i10.26409](https://doi.org/10.1609/aaai.v37i10.26409)

**Abstract**:

Heuristic search is a powerful approach that has successfully been applied to a broad class of planning problems, including classical planning, multi-objective planning, and probabilistic planning modelled as a stochastic shortest path (SSP) problem. Here, we extend the reach of heuristic search to a more expressive class of problems, namely multi-objective stochastic shortest paths (MOSSPs), which require computing a coverage set of non-dominated policies. We design new heuristic search algorithms MOLAO* and MOLRTDP, which extend well-known SSP algorithms to the multi-objective case. We further construct a spectrum of domain-independent heuristic functions differing in their ability to take into account the stochastic and multi-objective features of the problem to guide the search. Our experiments demonstrate the benefits of these algorithms and the relative merits of the heuristics.

----

## [1340] Zero-Knowledge Proofs for Classical Planning Problems

**Authors**: *Augusto B. Corrêa, Clemens Büchner, Remo Christen*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i10.26410](https://doi.org/10.1609/aaai.v37i10.26410)

**Abstract**:

In classical planning, the aim is to find a sequence of deterministic actions leading from the initial to a goal state. In this work, we consider the scenario where a party who knows the solution to a planning task, called the prover, wants to convince a second party, the verifier, that it has the solution without revealing any information about the solution itself. This is relevant in domains where privacy is important, for example when plans contain sensitive information or when the solution should not be revealed upfront.  We achieve this by introducing a zero-knowledge protocol for plan existence. By restricting ourselves to tasks with polynomially-bounded plan length, we are able to construct a protocol that can be run efficiently by both the prover and verifier. The resulting protocol does not rely on any reduction, has a constant number of rounds, and runs in time polynomial in the size of the task.

----

## [1341] Planning with Hidden Parameter Polynomial MDPs

**Authors**: *Clarissa Costen, Marc Rigter, Bruno Lacerda, Nick Hawes*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i10.26411](https://doi.org/10.1609/aaai.v37i10.26411)

**Abstract**:

For many applications of Markov Decision Processes (MDPs), the transition function cannot be specified exactly. Bayes-Adaptive MDPs (BAMDPs) extend MDPs to consider transition probabilities governed by latent parameters. To act optimally in BAMDPs, one must maintain a belief distribution over the latent parameters. Typically, this distribution is described by a set of sample (particle) MDPs, and associated weights which represent the likelihood of a sample MDP being the true underlying MDP. However, as the number of dimensions of the latent parameter space increases, the number of sample MDPs required to sufficiently represent the belief distribution grows exponentially. Thus, maintaining an accurate belief in the form of a set of sample MDPs over complex latent spaces is computationally intensive, which in turn affects the performance of planning for these models. In this paper, we propose an alternative approach for maintaining the belief over the latent parameters. We consider a class of BAMDPs where the transition probabilities can be expressed in closed form as a polynomial of the latent parameters, and outline a method to maintain a closed-form belief distribution for the latent parameters which results in an accurate belief representation. Furthermore, the closed-form representation does away with the need to tune the number of sample MDPs required to represent the belief. We evaluate two domains and empirically show that the polynomial, closed-form, belief representation results in better plans than a sampling-based belief representation.

----

## [1342] Privacy Attacks on Schedule-Driven Data

**Authors**: *Stephan A. Fahrenkrog-Petersen, Arik Senderovich, Alexandra Tichauer, Ali Kaan Tutak, J. Christopher Beck, Matthias Weidlich*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i10.26412](https://doi.org/10.1609/aaai.v37i10.26412)

**Abstract**:

Schedules define how resources process jobs in diverse domains, reaching from healthcare to transportation, and, therefore, denote a valuable starting point for analysis of the underlying system. However, publishing a schedule may disclose private information on the considered jobs. In this paper, we provide a first threat model for published schedules, thereby defining a completely new class of data privacy problems. We then propose distance-based measures to assess the privacy loss incurred by a published schedule, and show their theoretical properties for an uninformed adversary, which can be used as a benchmark for informed attacks. We show how an informed attack on a published schedule can be phrased as an inverse scheduling problem. We instantiate this idea by formulating the inverse of a well-studied single-machine scheduling problem, namely minimizing the total weighted completion times. An empirical evaluation for synthetic scheduling problems shows the effectiveness of informed privacy attacks and compares the results to theoretical bounds on uninformed attacks.

----

## [1343] Markov Decision Processes with Time-Varying Geometric Discounting

**Authors**: *Jiarui Gan, Annika Hennes, Rupak Majumdar, Debmalya Mandal, Goran Radanovic*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i10.26413](https://doi.org/10.1609/aaai.v37i10.26413)

**Abstract**:

Canonical models of Markov decision processes (MDPs) usually consider geometric discounting based on a constant discount factor. While this standard modeling approach has led to many elegant results, some recent studies indicate the necessity of modeling time-varying discounting in certain applications. This paper studies a model of infinite-horizon MDPs with time-varying discount factors. We take a game-theoretic perspective – whereby each time step is treated as an independent decision maker with their own (fixed) discount factor – and we study the subgame perfect equilibrium (SPE) of the resulting game as well as the related algorithmic problems. We present a constructive proof of the existence of an SPE and demonstrate the EXPTIME-hardness of computing an SPE. We also turn to the approximate notion of epsilon-SPE and show that an epsilon-SPE exists under milder assumptions. An algorithm is presented to compute an epsilon-SPE, of which an upper bound of the time complexity, as a function of the convergence property of the time-varying discount factor, is provided.

----

## [1344] Learning-Augmented Algorithms for Online TSP on the Line

**Authors**: *Themistoklis Gouleakis, Konstantinos Lakis, Golnoosh Shahkarami*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i10.26414](https://doi.org/10.1609/aaai.v37i10.26414)

**Abstract**:

We study the online Traveling Salesman Problem (TSP) on the line augmented with machine-learned predictions. In the classical problem, there is a stream of requests released over time along the real line. The goal is to minimize the makespan of the algorithm. We distinguish between the open variant and the closed one, in which we additionally require the algorithm to return to the origin after serving all requests. The state of the art is a 1.64-competitive algorithm and a 2.04-competitive algorithm for the closed and open variants, respectively. In both cases, a tight lower bound is known.

In both variants, our primary prediction model involves predicted positions of the requests. We introduce algorithms that (i) obtain a tight 1.5 competitive ratio for the closed variant and a 1.66 competitive ratio for the open variant in the case of perfect predictions, (ii) are robust against unbounded prediction error, and (iii) are smooth, i.e., their performance degrades gracefully as the prediction error increases.

Moreover, we further investigate the learning-augmented setting in the open variant by additionally considering a prediction for the last request served by the optimal offline algorithm. Our algorithm for this enhanced setting obtains a 1.33 competitive ratio with perfect predictions while also being smooth and robust, beating the lower bound of 1.44 we show for our original prediction setting for the open variant. Also, we provide a lower bound of 1.25 for this enhanced setting.

----

## [1345] Networked Restless Bandits with Positive Externalities

**Authors**: *Christine Herlihy, John P. Dickerson*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i10.26415](https://doi.org/10.1609/aaai.v37i10.26415)

**Abstract**:

Restless multi-armed bandits are often used to model budget-constrained resource allocation tasks where receipt of the resource is associated with an increased probability of a favorable state transition. Prior work assumes that individual arms only benefit if they receive the resource directly. However, many allocation tasks occur within communities and can be characterized by positive externalities that allow arms to derive partial benefit when their neighbor(s) receive the resource. We thus introduce networked restless bandits, a novel multi-armed bandit setting in which arms are both restless and embedded within a directed graph. We then present Greta, a graph-aware, Whittle index-based heuristic algorithm that can be used to efficiently construct a constrained reward-maximizing action vector at each timestep. Our empirical results demonstrate that Greta outperforms comparison policies across a range of hyperparameter values and graph topologies. Code and appendices are available at https://github.com/crherlihy/networked_restless_bandits.

----

## [1346] Planning for Learning Object Properties

**Authors**: *Leonardo Lamanna, Luciano Serafini, Mohamadreza Faridghasemnia, Alessandro Saffiotti, Alessandro Saetti, Alfonso Gerevini, Paolo Traverso*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i10.26416](https://doi.org/10.1609/aaai.v37i10.26416)

**Abstract**:

Autonomous agents embedded in a physical environment need the ability to recognize objects and their properties from sensory data. Such a perceptual ability is often implemented by supervised machine learning models, which are pre-trained using a set of labelled data. In real-world, open-ended deployments, however, it is unrealistic to assume to have a pre-trained model for all possible environments. Therefore, agents need to dynamically learn/adapt/extend their perceptual abilities online, in an autonomous way, by exploring and interacting with the environment where they operate. This paper describes a way to do so, by exploiting symbolic planning. Specifically, we formalize the problem of automatically training a neural network to recognize object properties as a symbolic planning problem (using PDDL). We use planning techniques to produce a strategy for automating the training dataset creation and the learning process. Finally, we provide an experimental evaluation in both a simulated and a real environment, which shows that the proposed approach is able to successfully learn how to recognize new object properties.

----

## [1347] Fully Online Matching with Stochastic Arrivals and Departures

**Authors**: *Zihao Li, Hao Wang, Zhenzhen Yan*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i10.26417](https://doi.org/10.1609/aaai.v37i10.26417)

**Abstract**:

We study a fully online matching problem with stochastic arrivals and departures. In this model, each online arrival follows a known identical and independent distribution over a fixed set of agent types. Its sojourn time is unknown in advance and follows type-specific distributions with known expectations. The goal is to maximize the weighted reward from successful matches. To solve this problem, we first propose a linear program (LP)-based algorithm whose competitive ratio is lower bounded by 0.155 under mild conditions. We further achieve better ratios in some special cases. To demonstrate the challenges of the problem, we further establish several hardness results. In particular, we show that no online algorithm can achieve a competitive ratio better than 2/3 in this model and there is no LP-based algorithm (with respect to our proposed LP) with a competitive ratio better than 1/3. Finally, we demonstrate the effectiveness and efficiency of our algorithm numerically.

----

## [1348] Towards Automated Modeling Assistance: An Efficient Approach for Repairing Flawed Planning Domains

**Authors**: *Songtuan Lin, Alban Grastien, Pascal Bercher*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i10.26418](https://doi.org/10.1609/aaai.v37i10.26418)

**Abstract**:

Designing a planning domain is a difficult task in AI planning. Assisting tools are thus required if we want planning to be used more broadly. In this paper, we are interested in automatically correcting a flawed domain. In particular, we are concerned with the scenario where a domain contradicts a plan that is known to be valid. Our goal is to repair the domain so as to turn the plan into a solution. Specifically, we consider both grounded and lifted representations support for negative preconditions and show how to explore the space of repairs to find the optimal one efficiently. As an evidence of the efficiency of our approach, the experiment results show that all flawed domains except one in the benchmark set can be repaired optimally by our approach within one second.

----

## [1349] Was Fixing This Really That Hard? On the Complexity of Correcting HTN Domains

**Authors**: *Songtuan Lin, Pascal Bercher*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i10.26419](https://doi.org/10.1609/aaai.v37i10.26419)

**Abstract**:

Automated modeling assistance is indispensable to the AI planning being deployed in practice, notably in industry and other non-academic contexts. Yet, little progress has been made that goes beyond smart interfaces like programming environments. They focus on autocompletion, but lack intelligent support for guiding the modeler. As a theoretical foundation of a first step towards this direction, we study the computational complexity of correcting a flawed Hierarchical Task Network (HTN) planning domain. Specifically, a modeler provides a (white) list of plans that are supposed to be solutions, and likewise a (black) list of plans that shall not be solutions. We investigate the complexity of finding a set of (optimal or suboptimal) model corrections so that those plans are (resp. not) solutions to the corrected model. More specifically, we factor out each hardness source that contributes towards NP-hardness, including one that we deem important for many other complexity investigations that go beyond our specific context of application. All complexities range between NP and Sigma-2-p, rising the hope for efficient practical tools in the future.

----

## [1350] On Total-Order HTN Plan Verification with Method Preconditions - An Extension of the CYK Parsing Algorithm

**Authors**: *Songtuan Lin, Gregor Behnke, Simona Ondrcková, Roman Barták, Pascal Bercher*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i10.26420](https://doi.org/10.1609/aaai.v37i10.26420)

**Abstract**:

In this paper, we consider the plan verification problem for totally ordered (TO) HTN planning. The problem is proved to be solvable in polynomial time by recognizing its connection to the membership decision problem for context-free grammars. Currently, most HTN plan verification approaches do not have special treatments for the TO configuration, and the only one features such an optimization still relies on an exhaustive search. Hence, we will develop a new TOHTN plan verification approach in this paper by extending the standard CYK parsing algorithm which acts as the best decision procedure in general.

----

## [1351] A Dynamics and Task Decoupled Reinforcement Learning Architecture for High-Efficiency Dynamic Target Intercept

**Authors**: *Dora D. Liu, Liang Hu, Qi Zhang, Tangwei Ye, Usman Naseem, Zhong Yuan Lai*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i10.26421](https://doi.org/10.1609/aaai.v37i10.26421)

**Abstract**:

Due to the flexibility and ease of control, unmanned aerial vehicles (UAVs) have been increasingly used in various scenarios and applications in recent years. Training UAVs with reinforcement learning (RL) for a specific task is often expensive in terms of time and computation. However, it is known that the main effort of the learning process is made to fit the low-level physical dynamics systems instead of the high-level task itself. In this paper, we study to apply UAVs in the dynamic target intercept (DTI) task, where the dynamics systems equipped by different UAV models are correspondingly distinct. To this end, we propose a dynamics and task decoupled RL architecture to address the inefficient learning procedure, where the RL module focuses on modeling the DTI task without involving physical dynamics, and the design of states, actions, and rewards are completely task-oriented while the dynamics control module can adaptively convert actions from the RL module to dynamics signals to control different UAVs without retraining the RL module. We show the efficiency and efficacy of our results in comparison and ablation experiments against state-of-the-art methods.

----

## [1352] AlphaRoute: Large-Scale Coordinated Route Planning via Monte Carlo Tree Search

**Authors**: *Guiyang Luo, Yantao Wang, Hui Zhang, Quan Yuan, Jinglin Li*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i10.26422](https://doi.org/10.1609/aaai.v37i10.26422)

**Abstract**:

Comprehensive experiments are conducted on two real-world road networks as compared with several baselines to evaluate the performance, and results show that AlphaRoute achieves the lowest travel time, and is efficient and effective for coordinating large-scale routes and alleviating the traffic congestion problem. The code will be publicly available.

----

## [1353] Learning Rational Subgoals from Demonstrations and Instructions

**Authors**: *Zhezheng Luo, Jiayuan Mao, Jiajun Wu, Tomás Lozano-Pérez, Joshua B. Tenenbaum, Leslie Pack Kaelbling*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i10.26423](https://doi.org/10.1609/aaai.v37i10.26423)

**Abstract**:

We present a framework for learning useful subgoals that support efficient long-term planning to achieve novel goals. At the core of our framework is a collection of rational subgoals (RSGs), which are essentially binary classifiers over the environmental states. RSGs can be learned from weakly-annotated data, in the form of unsegmented demonstration trajectories, paired with abstract task descriptions, which are composed of terms initially unknown to the agent (e.g., collect-wood then craft-boat then go-across-river). Our framework also discovers dependencies between RSGs, e.g., the task collect-wood is a helpful subgoal for the task craft-boat. Given a goal description, the learned subgoals and the derived dependencies facilitate off-the-shelf planning algorithms, such as A* and RRT, by setting helpful subgoals as waypoints to the planner, which significantly improves performance-time efficiency. Project page: https://rsg.csail.mit.edu

----

## [1354] Learning Safe Numeric Action Models

**Authors**: *Argaman Mordoch, Brendan Juba, Roni Stern*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i10.26424](https://doi.org/10.1609/aaai.v37i10.26424)

**Abstract**:

Powerful domain-independent planners have been developed to solve various types of planning problems. 
These planners often require a model of the acting agent's actions, given in some planning domain description language. 
Yet obtaining such an action model is a notoriously hard task. 
This task is even more challenging in mission-critical domains, where a trial-and-error approach to learning how to act is not an option. 
In such domains, the action model used to generate plans must be safe, in the sense that plans generated with it must be applicable and achieve their goals. 
Learning safe action models for planning has been recently explored for domains in which states are sufficiently described with Boolean variables. 
In this work, we go beyond this limitation and propose the NSAM algorithm. 
NSAM runs in time that is polynomial in the number of observations and, under certain conditions, is guaranteed to return safe action models. 
We analyze its worst-case sample complexity, which may be intractable for some domains. Empirically, however, NSAM can quickly learn a safe action model that can solve most problems in the domain.

----

## [1355] Automated Verification of Social Laws in Numeric Settings

**Authors**: *Ronen Nir, Alexander Shleyfman, Erez Karpas*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i10.26425](https://doi.org/10.1609/aaai.v37i10.26425)

**Abstract**:

It is possible for agents operating in a shared environment to interfere with one another. One mechanism of coordination is called Social Law. Enacting such a law in a multi-agent setting restricts agents' behaviors. Robustness, in this case, ensures that the agents do not harmfully interfere with each other and that each agent achieves its goals regardless of what other agents do. Previous work on social law verification examined only the case of boolean state variables. However, many real-world problems require reasoning with numeric variables. Moreover, numeric fluents allow a more compact representation of multiple planning problems.

In this paper, we develop a method to verify whether a given social law is robust via compilation to numeric planning. A solution to this compilation constitutes a counterexample to the robustness of the problem, i.e., evidence of cross-agent conflict. Thus, the social law is robust if and only if the proposed compilation is unsolvable. We empirically verify robustness in multiple domains using state-of-the-art numeric planners. Additionally, this compilation raises a challenge by generating a set of non-trivial numeric domains where unsolvability should be either proved or disproved.

----

## [1356] Expressive Optimal Temporal Planning via Optimization Modulo Theory

**Authors**: *Stefan Panjkovic, Andrea Micheli*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i10.26426](https://doi.org/10.1609/aaai.v37i10.26426)

**Abstract**:

Temporal Planning is the problem of synthesizing a course of actions given a predictive model of a system subject to temporal constraints. This kind of planning finds natural applications in the automation of industrial processes and in robotics when the timing and deadlines are important. Finding any plan in temporal planning is often not enough as it is sometimes needed to optimize a certain objective function: particularly interesting are the minimization of the makespan and the optimization of the costs of actions. Despite the importance of the problem, only few works in the literature tackled the problem of optimal temporal planning because of the complicated intermix of planning and scheduling.
In this paper, we address the problem of optimal temporal planning for a very expressive class of problems using a reduction of the bounded planning problem to Optimization Modulo Theory (OMT) a powerful discrete/continuous optimization framework. We theoretically and empirically show the expressive power of this approach and we set a baseline for future research in this area.

----

## [1357] Flexible Budgets in Restless Bandits: A Primal-Dual Algorithm for Efficient Budget Allocation

**Authors**: *Paula Rodriguez Diaz, Jackson A. Killian, Lily Xu, Arun Sai Suggala, Aparna Taneja, Milind Tambe*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i10.26427](https://doi.org/10.1609/aaai.v37i10.26427)

**Abstract**:

Restless multi-armed bandits (RMABs) are an important model to optimize allocation of limited resources in sequential decision-making settings. Typical RMABs assume the budget --- the number of arms pulled --- to be fixed for each step in the planning horizon. However, for realistic real-world planning, resources are not necessarily limited at each planning step; we may be able to distribute surplus resources in one round to an earlier or later round. In real-world planning settings, this flexibility in budget is often constrained to within a subset of consecutive planning steps, e.g., weekly planning of a monthly budget. In this paper we define a general class of RMABs with flexible budget, which we term F-RMABs, and provide an algorithm to optimally solve for them. We derive a min-max formulation to find optimal policies for F-RMABs and leverage gradient primal-dual algorithms to solve for reward-maximizing policies with flexible budgets. We introduce a scheme to sample expected gradients to apply primal-dual algorithms to the F-RMAB setting and make an otherwise computationally expensive approach tractable. Additionally, we provide heuristics that trade off solution quality for efficiency and present experimental comparisons of different F-RMAB solution approaches.

----

## [1358] Structurally Restricted Fragments of Numeric Planning - a Complexity Analysis

**Authors**: *Alexander Shleyfman, Daniel Gnad, Peter Jonsson*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i10.26428](https://doi.org/10.1609/aaai.v37i10.26428)

**Abstract**:

Numeric planning is known to be undecidable even under severe restrictions. Prior work has investigated the decidability boundaries by restricting the expressiveness of the planning formalism in terms of the numeric functions allowed in conditions and effects. We study a well-known restricted form of Hoffmann's simple numeric planning, which is undecidable. We analyze the complexity by imposing restrictions on the causal structure, exploiting a novel method for bounding variable domain sizes. First, we show that plan existence for tasks where all numeric variables are root nodes in the causal graph is in PSPACE.
Second, we show that for tasks with only numeric leaf variables the problem is decidable, and that it is in PSPACE if the propositional state space has a fixed size. Our work lays a strong foundation for future investigations of structurally more complex tasks. From a practical perspective, our method allows to employ heuristics and methods that are geared towards finite variable domains (such as pattern database heuristics or decoupled search) to solve non-trivial families of numeric planning problems.

----

## [1359] Predicate Invention for Bilevel Planning

**Authors**: *Tom Silver, Rohan Chitnis, Nishanth Kumar, Willie McClinton, Tomás Lozano-Pérez, Leslie Pack Kaelbling, Joshua B. Tenenbaum*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i10.26429](https://doi.org/10.1609/aaai.v37i10.26429)

**Abstract**:

Efficient planning in continuous state and action spaces is fundamentally hard, even when the transition model is deterministic and known. One way to alleviate this challenge is to perform bilevel planning with abstractions, where a high-level search for abstract plans is used to guide planning in the original transition space. Previous work has shown that when state abstractions in the form of symbolic predicates are hand-designed, operators and samplers for bilevel planning can be learned from demonstrations. In this work, we propose an algorithm for learning predicates from demonstrations, eliminating the need for manually specified state abstractions. Our key idea is to learn predicates by optimizing a surrogate objective that is tractable but faithful to our real efficient-planning objective. We use this surrogate objective in a hill-climbing search over predicate sets drawn from a grammar. Experimentally, we show across four robotic planning environments that our learned abstractions are able to quickly solve held-out tasks, outperforming six baselines.

----

## [1360] Smoothed Online Combinatorial Optimization Using Imperfect Predictions

**Authors**: *Kai Wang, Zhao Song, Georgios Theocharous, Sridhar Mahadevan*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i10.26430](https://doi.org/10.1609/aaai.v37i10.26430)

**Abstract**:

Smoothed online combinatorial optimization considers a learner who repeatedly chooses a combinatorial decision to minimize an unknown changing cost function with a penalty on switching decisions in consecutive rounds. We study smoothed online combinatorial optimization problems when an imperfect predictive model is available, where the model can forecast the future cost functions with uncertainty. We show that using predictions to plan for a finite time horizon leads to regret dependent on the total predictive uncertainty and an additional switching cost. This observation suggests choosing a suitable planning window to balance between uncertainty and switching cost, which leads to an online algorithm with guarantees on the upper and lower bounds of the cumulative regret. Empirically, our algorithm shows a significant improvement in cumulative regret compared to other baselines in synthetic online distributed streaming problems.

----

## [1361] Scalable Decision-Focused Learning in Restless Multi-Armed Bandits with Application to Maternal and Child Health

**Authors**: *Kai Wang, Shresth Verma, Aditya Mate, Sanket Shah, Aparna Taneja, Neha Madhiwalla, Aparna Hegde, Milind Tambe*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i10.26431](https://doi.org/10.1609/aaai.v37i10.26431)

**Abstract**:

This paper studies restless multi-armed bandit (RMAB) problems with unknown arm transition dynamics but with known correlated arm features. The goal is to learn a model to predict transition dynamics given features, where the Whittle index policy solves the RMAB problems using predicted transitions. However, prior works often learn the model by maximizing the predictive accuracy instead of final RMAB solution quality, causing a mismatch between training and evaluation objectives. To address this shortcoming, we propose a novel approach for decision-focused learning in RMAB that directly trains the predictive model to maximize the Whittle index solution quality. We present three key contributions: (i) we establish differentiability of the Whittle index policy to support decision-focused learning; (ii) we significantly improve the scalability of decision-focused learning approaches in sequential problems, specifically RMAB problems; (iii) we apply our algorithm to a previously collected  dataset of maternal and child health to demonstrate its performance. Indeed, our algorithm is the first for decision-focused learning in RMAB that scales to real-world problem sizes.

----

## [1362] Neural TSP Solver with Progressive Distillation

**Authors**: *Dongxiang Zhang, Ziyang Xiao, Yuan Wang, Mingli Song, Gang Chen*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i10.26432](https://doi.org/10.1609/aaai.v37i10.26432)

**Abstract**:

Travelling salesman problem (TSP) is NP-Hard with exponential search space. Recently, the adoption of encoder-decoder models as neural TSP  solvers has emerged as an attractive topic because they can instantly obtain near-optimal results for small-scale instances. Nevertheless, their training efficiency and solution quality degrade dramatically when dealing with large-scale problems. To address the issue, we propose a novel progressive distillation framework, by adopting curriculum learning to train TSP samples in increasing order of their problem size and progressively distilling high-level knowledge from small models to large models via a distillation loss. In other words, the trained small models are used as the teacher network to guide action selection when training large models. To accelerate training speed, we also propose a Delaunary-graph based action mask and a new attention-based decoder to reduce decoding cost. Experimental results show that our approach  establishes clear advantages over existing encoder-decoder models in terms of training effectiveness and solution quality. In addition, we validate its usefulness as an initial solution generator for the state-of-the-art TSP solvers, whose probability of obtaining the optimal solution can be further improved in such a hybrid manner.

----

## [1363] The Linear Distance Traveling Tournament Problem Allows an EPTAS

**Authors**: *Jingyang Zhao, Mingyu Xiao*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i10.26433](https://doi.org/10.1609/aaai.v37i10.26433)

**Abstract**:

The Traveling Tournament Problem (TTP-k) is a well-known benchmark problem in tournament timetabling and has been extensively studied in the field of AI. In this problem, we are going to design a double round-robin schedule such that each pair of teams plays one game in each other's home venue, minimizing the total distance traveled by all n teams (n is even) under the constraint that each team can have at most k-consecutive home games or away games. The Linear Distance Traveling Tournament Problem (LDTTP-k), where all teams are located on a line, was introduced by Hoshino and Kawarabayashi (AAAI 2012). For LDTTP-3, they gave a 4/3-approximation algorithm for n≡4 (mod 6) teams. In this paper, we show that for any 3≤k=o(∛n), LDTTP-k allows an efficient polynomial-time approximation scheme (EPTAS).

----

## [1364] Learning Relational Causal Models with Cycles through Relational Acyclification

**Authors**: *Ragib Ahsan, David Arbour, Elena Zheleva*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i10.26434](https://doi.org/10.1609/aaai.v37i10.26434)

**Abstract**:

In real-world phenomena which involve mutual influence or causal effects between interconnected units, equilibrium states are typically represented with cycles in graphical models. An expressive class of graphical models, relational causal models, can represent and reason about complex dynamic systems exhibiting such cycles or feedback loops. Existing cyclic causal discovery algorithms for learning causal models from observational data assume that the data instances are independent and identically distributed which makes them unsuitable for relational causal models. At the same time, causal discovery algorithms for relational causal models assume acyclicity. In this work, we examine the necessary and sufficient conditions under which a constraint-based relational causal discovery algorithm is sound and complete for cyclic relational causal models. We introduce relational acyclification, an operation specifically designed for relational models that enables reasoning about the identifiability of cyclic relational causal models. We show that under the assumptions of relational acyclification and sigma-faithfulness, the relational causal discovery algorithm RCD is sound and complete for cyclic relational models. We present experimental results to support our claim.

----

## [1365] Causal Effect Identification in Cluster DAGs

**Authors**: *Tara V. Anand, Adèle H. Ribeiro, Jin Tian, Elias Bareinboim*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i10.26435](https://doi.org/10.1609/aaai.v37i10.26435)

**Abstract**:

Reasoning about the effect of interventions and counterfactuals is a fundamental task found throughout the data sciences. A collection of principles, algorithms, and tools has been developed for performing such tasks in the last decades.  One of the pervasive requirements found throughout this literature is the articulation of assumptions, which commonly appear in the form of causal diagrams. Despite the power of this approach, there are significant settings where the knowledge necessary to specify a causal diagram over all variables is not available, particularly in complex, high-dimensional domains. In this paper, we introduce a new graphical modeling tool called cluster DAGs (for short, C-DAGs) that allows for the partial specification of relationships among variables based on limited prior knowledge, alleviating the stringent requirement of specifying a full causal diagram. A C-DAG specifies relationships between clusters of variables, while the relationships between the variables within a cluster are left unspecified, and can be seen as a graphical representation of an equivalence class of causal diagrams that share the relationships among the clusters. We develop the foundations and machinery for valid inferences over C-DAGs about the clusters of variables at each layer of Pearl's Causal Hierarchy - L1 (probabilistic), L2 (interventional), and L3 (counterfactual).  In particular, we prove the soundness and completeness of d-separation for probabilistic inference in C-DAGs.  Further, we demonstrate the validity of Pearl's do-calculus rules over C-DAGs and show that the standard ID identification algorithm is sound and complete to systematically compute causal effects from observational data given a C-DAG. Finally, we show that C-DAGs are valid for performing counterfactual inferences about clusters of variables.

----

## [1366] A Simple Unified Approach to Testing High-Dimensional Conditional Independences for Categorical and Ordinal Data

**Authors**: *Ankur Ankan, Johannes Textor*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i10.26436](https://doi.org/10.1609/aaai.v37i10.26436)

**Abstract**:

Conditional independence (CI) tests underlie many approaches to model testing and structure learning in causal inference. Most existing CI tests for categorical and ordinal data stratify the sample by the conditioning variables, perform simple independence tests in each stratum, and combine the results. Unfortunately, the statistical power of this approach degrades rapidly as the number of conditioning variables increases. Here we propose a simple unified CI test for ordinal and categorical data that maintains reasonable calibration and power in high dimensions. We show that our test outperforms existing baselines in model testing and structure learning for dense directed graphical models while being comparable for sparse models. Our approach could be attractive for causal model testing because it is easy to implement, can be used with non-parametric or parametric probability models, has the symmetry property, and has reasonable computational requirements.

----

## [1367] Score-Based Learning of Graphical Event Models with Background Knowledge Augmentation

**Authors**: *Debarun Bhattacharjya, Tian Gao, Dharmashankar Subramanian, Xiao Shou*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i10.26437](https://doi.org/10.1609/aaai.v37i10.26437)

**Abstract**:

Graphical event models (GEMs) are representations of temporal point process dynamics between different event types. Many real-world applications however involve limited event stream data, making it challenging to learn GEMs from data alone. In this paper, we introduce approaches that can work together in a score-based learning paradigm, to augment data with potentially different types of background knowledge. We propose novel scores for learning an important parametric class of GEMs; in particular, we propose a Bayesian score for leveraging prior information as well as a more practical simplification that involves fewer parameters, analogous to Bayesian networks. We also introduce a framework for incorporating easily assessed qualitative background knowledge from domain experts, in the form of statements such as `event X depends on event Y' or `event Y makes event X more likely'. The proposed framework has Bayesian interpretations and can be deployed by any score-based learner. Through an extensive empirical investigation, we demonstrate the practical benefits of background knowledge augmentation while learning GEMs for applications in the low-data regime.

----

## [1368] Entropy Regularization for Population Estimation

**Authors**: *Ben Chugg, Peter Henderson, Jacob Goldin, Daniel E. Ho*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i10.26438](https://doi.org/10.1609/aaai.v37i10.26438)

**Abstract**:

Entropy regularization is known to improve exploration in sequential decision-making problems. We show that this same mechanism can also lead to nearly unbiased and lower-variance estimates of the mean reward in the optimize-and-estimate structured bandit setting. Mean reward estimation (i.e., population estimation) tasks have recently been shown to be essential for public policy settings where legal constraints often require precise estimates of population metrics. We show that leveraging entropy and KL divergence can yield a better trade-off between reward and estimator variance than existing baselines, all while remaining nearly unbiased. These properties of entropy regularization illustrate an exciting potential for bringing together the optimal exploration and estimation literature.

----

## [1369] Principled and Efficient Motif Finding for Structure Learning of Lifted Graphical Models

**Authors**: *Jonathan Feldstein, Dominic Phillips, Efthymia Tsamoura*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i10.26439](https://doi.org/10.1609/aaai.v37i10.26439)

**Abstract**:

Structure learning is a core problem in AI central to the fields of neuro-symbolic AI and statistical relational learning. It consists in automatically learning a logical theory from data. The basis for structure learning is mining repeating patterns in the data, known as structural motifs. Finding these patterns reduces the exponential search space and therefore guides the learning of formulas. Despite the importance of motif learning, it is still not well understood. We present the first principled approach for mining structural motifs in lifted graphical models, languages that blend first-order logic with probabilistic models,  which uses a stochastic process to measure the similarity of entities in the data. 

Our first contribution is an algorithm, which depends on two intuitive hyperparameters: one controlling the uncertainty in the entity similarity measure, and one controlling the softness of the resulting rules. Our second contribution is a preprocessing step where we perform hierarchical clustering on the data to reduce the search space to the most relevant data. Our third contribution is to introduce an O(n ln(n)) (in the size of the entities in the data) algorithm for clustering structurally-related data. We evaluate our approach using standard benchmarks and show that we outperform state-of-the-art structure learning approaches by up to 6% in terms of accuracy and up to 80% in terms of runtime.

----

## [1370] A Faster Practical Approximation Scheme for the Permanent

**Authors**: *Juha Harviainen, Mikko Koivisto*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i10.26440](https://doi.org/10.1609/aaai.v37i10.26440)

**Abstract**:

The permanent of a matrix has numerous applications but is notoriously hard to compute. While nonnegative matrices admit polynomial approximation schemes based on rapidly mixing Markov chains, the known practical estimators of the permanent rely on importance or rejection sampling. We advance the rejection sampling approach, which provides probabilistic accuracy guarantees, unlike importance sampling. Specifically, we give a novel class of nesting upper bounds and a simple preprocessing method that, in comparison to previous works, enable faster sampling with better acceptance rate; we demonstrate order-of-magnitude improvements with both theoretical and empirical analyses. In addition, we display instances on which our approximation scheme is competitive against state-of-the-art importance sampling based estimators.

----

## [1371] Neural Diffeomorphic Non-uniform B-spline Flows

**Authors**: *Seongmin Hong, Se Young Chun*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i10.26441](https://doi.org/10.1609/aaai.v37i10.26441)

**Abstract**:

Normalizing flows have been successfully modeling a complex probability distribution as an invertible transformation of a simple base distribution. However, there are often applications that require more than invertibility. For instance, the computation of energies and forces in physics requires the second derivatives of the transformation to be well-defined and continuous. Smooth normalizing flows employ infinitely differentiable transformation, but with the price of slow non-analytic inverse transforms. In this work, we propose diffeomorphic non-uniform B-spline flows that are at least twice continuously differentiable while bi-Lipschitz continuous, enabling efficient parametrization while retaining analytic inverse transforms based on a sufficient condition for diffeomorphism. Firstly, we investigate the sufficient condition for C(k-2)-diffeomorphic non-uniform kth-order B-spline transformations. Then, we derive an analytic inverse transformation of the non-uniform cubic B-spline transformation for neural diffeomorphic non-uniform B-spline flows. Lastly, we performed experiments on solving the force matching problem in Boltzmann generators, demonstrating that our C2-diffeomorphic non-uniform B-spline flows yielded solutions better than previous spline flows and faster than smooth normalizing flows. Our source code is publicly available at https://github.com/smhongok/Non-uniform-B-spline-Flow.

----

## [1372] Identification and Estimation of the Probabilities of Potential Outcome Types Using Covariate Information in Studies with Non-compliance

**Authors**: *Yuta Kawakami, Ryusei Shingaki, Manabu Kuroki*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i10.26442](https://doi.org/10.1609/aaai.v37i10.26442)

**Abstract**:

We propose novel identification conditions and a statistical estimation method for the probabilities of potential outcome types using covariate information in randomized trials in which the treatment assignment is randomized but subject compliance is not perfect. Different from existing studies, the proposed identification conditions do not require strict assumptions such as the assumption of monotonicity. When the probabilities of potential outcome types are identifiable through the proposed conditions, the problem of estimating the probabilities of potential outcome types is reduced to that of singular models. Thus, the probabilities cannot be evaluated using standard statistical likelihood-based estimation methods. Rather, the proposed identification conditions show that we can derive consistent estimators of the probabilities of potential outcome types via the method of moments, which leads to the asymptotic normality of the proposed estimators through the delta method under regular conditions. We also propose a new statistical estimation method based on the bounded constrained augmented Lagrangian method to derive more efficient estimators than can be derived through the method of moments.

----

## [1373] Computing Divergences between Discrete Decomposable Models

**Authors**: *Loong Kuan Lee, Nico Piatkowski, François Petitjean, Geoffrey I. Webb*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i10.26443](https://doi.org/10.1609/aaai.v37i10.26443)

**Abstract**:

There are many applications that benefit from computing the exact divergence between 2 discrete probability measures, including machine learning. Unfortunately, in the absence of any assumptions on the structure or independencies within these distributions, computing the divergence between them is an intractable problem in high dimensions. We show that we are able to compute a wide family of functionals and divergences, such as the alpha-beta divergence, between two decomposable models, i.e. chordal Markov networks, in time exponential to the treewidth of these models. The alpha-beta divergence is a family of divergences that include popular divergences such as the Kullback-Leibler divergence, the Hellinger distance, and the chi-squared divergence. Thus, we can accurately compute the exact values of any of this broad class of divergences to the extent to which we can accurately model the two distributions using decomposable models.

----

## [1374] Out-of-Distribution Generalization by Neural-Symbolic Joint Training

**Authors**: *Anji Liu, Hongming Xu, Guy Van den Broeck, Yitao Liang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i10.26444](https://doi.org/10.1609/aaai.v37i10.26444)

**Abstract**:

This paper develops a novel methodology to simultaneously learn a neural network and extract generalized logic rules. Different from prior neural-symbolic methods that require background knowledge and candidate logical rules to be provided, we aim to induce task semantics with minimal priors. This is achieved by a two-step learning framework that iterates between optimizing neural predictions of task labels and searching for a more accurate representation of the hidden task semantics. Notably, supervision works in both directions: (partially) induced task semantics guide the learning of the neural network and induced neural predictions admit an improved semantic representation. We demonstrate that our proposed framework is capable of achieving superior out-of-distribution generalization performance on two tasks: (i) learning multi-digit addition, where it is trained on short sequences of digits and tested on long sequences of digits; (ii) predicting the optimal action in the Tower of Hanoi, where the model is challenged to discover a policy independent of the number of disks in the puzzle.

----

## [1375] Novel Ordering-Based Approaches for Causal Structure Learning in the Presence of Unobserved Variables

**Authors**: *Ehsan Mokhtarian, Mohammadsadegh Khorasani, Jalal Etesami, Negar Kiyavash*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i10.26445](https://doi.org/10.1609/aaai.v37i10.26445)

**Abstract**:

We propose ordering-based approaches for learning the maximal ancestral graph (MAG) of a structural equation model (SEM) up to its Markov equivalence class (MEC) in the presence of unobserved variables. Existing ordering-based methods in the literature recover a graph through learning a causal order (c-order). We advocate for a novel order called removable order (r-order) as they are advantageous over c-orders for structure learning. This is because r-orders are the minimizers of an appropriately defined optimization problem that could be either solved exactly (using a reinforcement learning approach) or approximately (using a hill-climbing search). Moreover, the r-orders (unlike c-orders) are invariant among all the graphs in a MEC and include c-orders as a subset. Given that set of r-orders is often significantly larger than the set of c-orders, it is easier for the optimization problem to find an r-order instead of a c-order. We evaluate the performance and the scalability of our proposed approaches on both real-world and randomly generated networks.

----

## [1376] Maximizing the Probability of Fixation in the Positional Voter Model

**Authors**: *Petros Petsinis, Andreas Pavlogiannis, Panagiotis Karras*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i10.26446](https://doi.org/10.1609/aaai.v37i10.26446)

**Abstract**:

The Voter model is a well-studied stochastic process that models the invasion of a novel trait A (e.g., a new opinion, social meme, genetic mutation, magnetic spin) in a network of individuals (agents, people, genes, particles) carrying an existing resident trait B. Individuals change traits by occasionally sampling the trait of a neighbor, while an invasion bias δ ≥ 0 expresses the stochastic preference to adopt the novel trait A over the resident trait B. The strength of an invasion is measured by the probability that eventually the whole population adopts trait A, i.e., the fixation probability. In more realistic settings, however, the invasion bias is not ubiquitous, but rather manifested only in parts of the network. For instance, when modeling the spread of a social trait, the invasion bias represents localized incentives. In this paper, we generalize the standard biased Voter model to the positional Voter model, in which the invasion bias is effectuated only on an arbitrary subset of the network nodes, called biased nodes. We study the ensuing optimization problem, which is, given a budget k, to choose k biased nodes so as to maximize the fixation probability of a randomly occurring invasion. We show that the problem is NP-hard both for finite δ and when δ → ∞ (strong bias), while the objective function is not submodular in either setting, indicating strong computational hardness. On the other hand, we show that, when δ → 0 (weak bias), we can obtain a tight approximation in O(n^2ω ) time, where ω is the matrix-multiplication exponent. We complement our theoretical results with an experimental evaluation of some proposed heuristics.

----

## [1377] Certifying Fairness of Probabilistic Circuits

**Authors**: *Nikil Roashan Selvam, Guy Van den Broeck, YooJung Choi*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i10.26447](https://doi.org/10.1609/aaai.v37i10.26447)

**Abstract**:

With the increased use of machine learning systems for decision making, questions about the fairness properties of such systems start to take center stage. Most existing work on algorithmic fairness assume complete observation of features at prediction time, as is the case for popular notions like statistical parity and equal opportunity. However, this is not sufficient for models that can make predictions with partial observation as we could miss patterns of bias and incorrectly certify a model to be fair. To address this, a recently introduced notion of fairness asks whether the model exhibits any discrimination pattern, in which an individual—characterized by (partial) feature observations—receives vastly different decisions merely by disclosing one or more sensitive attributes such as gender and race. By explicitly accounting for partial observations, this provides a much more fine-grained notion of fairness.

In this paper, we propose an algorithm to search for discrimination patterns in a general class of probabilistic models, namely probabilistic circuits. Previously, such algorithms were limited to naive Bayes classifiers which make strong independence assumptions; by contrast, probabilistic circuits provide a unifying framework for a wide range of tractable probabilistic models and can even be compiled from certain classes of Bayesian networks and probabilistic programs, making our method much more broadly applicable. Furthermore, for an unfair model, it may be useful to quickly find discrimination patterns and distill them for better interpretability. As such, we also propose a sampling-based approach to more efficiently mine discrimination patterns, and introduce new classes of patterns such as minimal, maximal, and Pareto optimal patterns that can effectively summarize exponentially many discrimination patterns.

----

## [1378] Probabilities of Potential Outcome Types in Experimental Studies: Identification and Estimation Based on Proxy Covariate Information

**Authors**: *Ryusei Shingaki, Manabu Kuroki*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i10.26448](https://doi.org/10.1609/aaai.v37i10.26448)

**Abstract**:

The concept of potential outcome types is one of the fundamental components of causal inference. However, even in randomized experiments, assumptions on the data generating process, such as monotonicity, are required to evaluate the probabilities of the potential outcome types. To solve the problem without such assumptions in experimental studies, a novel identification condition based on proxy covariate information is proposed in this paper. In addition, the estimation problem of the probabilities of the potential outcome types reduces to that of singular models when they are identifiable through the proposed condition. Thus, they cannot be evaluated by standard statistical estimation methods. To overcome this difficulty, new plug-in estimators of these probabilities are presented, and the asymptotic normality of the proposed estimators is shown.

----

## [1379] Lifted Inference with Linear Order Axiom

**Authors**: *Jan Tóth, Ondrej Kuzelka*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i10.26449](https://doi.org/10.1609/aaai.v37i10.26449)

**Abstract**:

We consider the task of weighted first-order model counting (WFOMC) used for probabilistic inference in the area of statistical relational learning. Given a formula φ, domain size n and a pair of weight functions, what is the weighted sum of all models of φ over a domain of size n? It was shown that computing WFOMC of any logical sentence with at most two logical variables can be done in time polynomial in n. However, it was also shown that the task is #P1-complete once we add the third variable, which inspired the search for extensions of the two-variable fragment that would still permit a running time polynomial in n. One of such extension is the two-variable fragment with counting quantifiers. In this paper, we prove that adding a linear order axiom (which forces one of the predicates in φ to introduce a linear ordering of the domain elements in each model of φ) on top of the counting quantifiers still permits a computation time polynomial in the domain size. We present a new dynamic programming-based algorithm which can compute WFOMC with linear order in time polynomial in n, thus proving our primary claim.

----

## [1380] Vector Causal Inference between Two Groups of Variables

**Authors**: *Jonas Wahl, Urmi Ninad, Jakob Runge*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i10.26450](https://doi.org/10.1609/aaai.v37i10.26450)

**Abstract**:

Methods to identify cause-effect relationships currently mostly assume the variables to be scalar random variables. However, in many fields the objects of interest are vectors or groups of scalar variables.
We present a new constraint-based non-parametric approach for inferring the causal relationship between two vector-valued random variables from observational data. Our method employs sparsity estimates of directed and undirected graphs and is based on two new principles for groupwise causal reasoning that we justify theoretically in Pearl's graphical model-based causality framework. Our theoretical considerations are complemented by two new causal discovery algorithms for causal interactions between two random vectors which find the correct causal direction reliably in simulations even if interactions are nonlinear. We evaluate our methods empirically and compare them to other state-of-the-art techniques.

----

## [1381] Efficient Enumeration of Markov Equivalent DAGs

**Authors**: *Marcel Wienöbst, Malte Luttermann, Max Bannach, Maciej Liskiewicz*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i10.26451](https://doi.org/10.1609/aaai.v37i10.26451)

**Abstract**:

Enumerating the directed acyclic graphs (DAGs) of a Markov equivalence class (MEC) is an important primitive in causal analysis. The central resource from the perspective of computational complexity is the delay, that is, the time an algorithm that lists all members of the class requires between two consecutive outputs. Commonly used algorithms for this task utilize the rules proposed by Meek (1995) or the transformational characterization by Chickering (1995), both resulting in superlinear delay. In this paper, we present the first linear-time delay algorithm. On the theoretical side, we show that our algorithm can be generalized to enumerate DAGs represented by models that incorporate background knowledge, such as MPDAGs; on the practical side, we provide an efficient implementation and evaluate it in a series of experiments. Complementary to the linear-time delay algorithm, we also provide intriguing insights into Markov equivalence itself: All members of an MEC can be enumerated such that two successive DAGs have structural Hamming distance at most three.

----

## [1382] Differentially Private Nonlinear Causal Discovery from Numerical Data

**Authors**: *Hao Zhang, Yewei Xia, Yixin Ren, Jihong Guan, Shuigeng Zhou*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i10.26452](https://doi.org/10.1609/aaai.v37i10.26452)

**Abstract**:

Recently, several methods such as private ANM, EM-PC and Priv-PC have been proposed to perform differentially private causal discovery in various scenarios including bivariate, multivariate Gaussian and categorical cases. However, there is little effort on how to conduct private nonlinear causal discovery from numerical data. This work tries to challenge this problem. To this end, we propose a method to infer nonlinear causal relations from observed numerical data by using regression-based conditional independence test (RCIT) that consists of kernel ridge regression (KRR) and Hilbert-Schmidt independence criterion (HSIC) with permutation approximation. Sensitivity analysis for RCIT is given and a private constraint-based causal discovery framework with differential privacy guarantee is developed. Extensive simulations and real-world experiments for both conditional independence test and causal discovery are conducted, which show that our method is effective in handling nonlinear numerical cases and easy to implement. The source code of our method and data are available at https://github.com/Causality-Inference/PCD.

----

## [1383] Safe Interval Path Planning with Kinodynamic Constraints

**Authors**: *Zain Alabedeen Ali, Konstantin S. Yakovlev*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i10.26453](https://doi.org/10.1609/aaai.v37i10.26453)

**Abstract**:

Safe Interval Path Planning (SIPP) is a powerful algorithm for solving a single-agent pathfinding problem where the agent is confined to a graph and certain vertices/edges of this graph are blocked at certain time intervals due to dynamic obstacles that populate the environment. The original SIPP algorithm relies on the assumption that the agent is able to stop instantaneously. However, this assumption often does not hold in practice, e.g. a mobile robot moving at a cruising speed cannot stop immediately but rather requires gradual deceleration to a full stop that takes time. In other words, the robot is subject to kinodynamic constraints. Unfortunately, as we show in this work, in such a case, the original SIPP is incomplete. To this end, we introduce a novel variant of SIPP that is provably complete and optimal for planning with acceleration/deceleration. In the experimental evaluation, we show that the key property of the original SIPP still holds for the modified version: it performs much fewer expansions compared to A* and, as a result, is notably faster.

----

## [1384] Diversity Maximization in the Presence of Outliers

**Authors**: *Daichi Amagata*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i10.26454](https://doi.org/10.1609/aaai.v37i10.26454)

**Abstract**:

Given a set X of n points in a metric space, the problem of diversity maximization is to extract a set S of k points from X so that the diversity of S is maximized. This problem is essential in AI-related fields, such as web search, databases, recommender systems, and data mining. Although there have been extensive studies of this problem, these studies assume that X is clean. This usually does not hold, because real-world datasets usually contain outliers. The state-of-the-art algorithm for the diversity maximization problem is based on furthest point retrieval, which is too sensitive to outliers. We therefore address the problem of diversity maximization with outliers and propose two algorithms with performance guarantee. The first algorithm runs in O((k+z)n) time, guarantees 1/2-approximation, and returns no outliers, where z is the number of outliers. The second algorithm runs in O(kz) time (which is independent of n), guarantees 1/6(1+epsilon)-approximation, and returns no outliers with constant probability. We conduct experiments on real datasets to demonstrate the effectiveness and efficiency of our algorithms.

----

## [1385] Fair Short Paths in Vertex-Colored Graphs

**Authors**: *Matthias Bentert, Leon Kellerhals, Rolf Niedermeier*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i10.26455](https://doi.org/10.1609/aaai.v37i10.26455)

**Abstract**:

The computation of short paths in graphs with arc lengths is a pillar of graph algorithmics and network science. In a more diverse world, however, not every short path is equally valuable. For the setting where each vertex is assigned to a group (color), we provide a framework to model multiple natural fairness aspects. We seek to find short paths in which the number of occurrences of each color is within some given lower and upper bounds. Among other results, we prove the introduced problems to be computationally intractable (NP-hard and parameterized hard with respect to the number of colors) even in very restricted settings (such as each color should appear with exactly the same frequency), while also presenting an encouraging algorithmic result ("fixed-parameter tractability") related to the length of the sought solution path for the general problem.

----

## [1386] AC-Band: A Combinatorial Bandit-Based Approach to Algorithm Configuration

**Authors**: *Jasmin Brandt, Elias Schede, Björn Haddenhorst, Viktor Bengs, Eyke Hüllermeier, Kevin Tierney*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i10.26456](https://doi.org/10.1609/aaai.v37i10.26456)

**Abstract**:

We study the algorithm configuration (AC) problem, in which one seeks to find an optimal parameter configuration of a given target algorithm in an automated way. Although this field of research has experienced much progress recently regarding approaches satisfying strong theoretical guarantees, there is still a gap between the practical performance of these approaches and the heuristic state-of-the-art approaches. Recently, there has been significant progress in designing AC approaches that satisfy strong theoretical guarantees. However, a significant gap still remains between the practical performance of these approaches and state-of-the-art heuristic methods. To this end, we introduce AC-Band, a general approach for the AC problem based on multi-armed bandits that provides theoretical guarantees while exhibiting strong practical performance. We show that AC-Band requires significantly less computation time than other AC approaches providing theoretical guarantees while still yielding high-quality configurations.

----

## [1387] GRASMOS: Graph Signage Model Selection for Gene Regulatory Networks

**Authors**: *Angelina Brilliantova, Hannah Miller, Ivona Bezáková*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i10.26457](https://doi.org/10.1609/aaai.v37i10.26457)

**Abstract**:

Signed networks (networks with positive and negative edges) commonly arise in various domains from molecular biology to social media. 
The edge signs -- i.e., the graph signage -- represent the interaction pattern between the vertices and can provide insights into the underlying system formation process. Generative models considering signage formation are essential for testing hypotheses about the emergence of interactions and for creating synthetic datasets for algorithm benchmarking (especially in areas where obtaining real-world datasets is difficult).

In this work, we pose a novel Maximum-Likelihood-based optimization problem for modeling signages given their topology and showcase it in the context of gene regulation. Regulatory interactions of genes play a key role in the process of organism development, and when broken can lead to serious organism abnormalities and diseases.  Our contributions are threefold: First, we design a new class of signage models for a given topology, and, based on the parameter setting, we discuss its biological interpretations for gene regulatory networks (GRNs). Second, we design algorithms computing the Maximum Likelihood -- depending on the parameter setting, our algorithms range from closed-form expressions to MCMC sampling. Third, we evaluated the results of our algorithms on synthetic datasets and real-world large GRNs. Our work can lead to the prediction of unknown gene regulations, novel biological hypotheses, and realistic benchmark datasets in the realm of gene regulation.

----

## [1388] Optimal Pathfinding on Weighted Grid Maps

**Authors**: *Mark Carlson, Sajjad K. Moghadam, Daniel Damir Harabor, Peter J. Stuckey, Morteza Ebrahimi*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i10.26458](https://doi.org/10.1609/aaai.v37i10.26458)

**Abstract**:

In many computer games up to hundreds of agents navigate in real-time across a dynamically changing weighted grid map. Pathfinding in these situations is challenging because the grids are large, traversal costs are not uniform, and because each shortest path has many symmetric permutations, all of which must be considered by an optimal online search. In this work we introduce Weighted Jump Point Search (JPSW), a new type of pathfinding algorithm which breaks weighted grid symmetries by introducing a tiebreaking policy that allows us to apply effective pruning rules in symmetric regions. We show that these pruning rules preserve at least one optimal path to every grid cell and that their application can yield large performance improvements for optimal pathfinding. We give a complete theoretical description of the new algorithm, including pseudo-code. We also conduct a wide-ranging experimental evaluation, including data from real games. Results indicate JPSW is up to orders of magnitude faster than the nearest baseline, online search using A*.

----

## [1389] Warm-Starting Nested Rollout Policy Adaptation with Optimal Stopping

**Authors**: *Chen Dang, Cristina Bazgan, Tristan Cazenave, Morgan Chopin, Pierre-Henri Wuillemin*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i10.26459](https://doi.org/10.1609/aaai.v37i10.26459)

**Abstract**:

Nested Rollout Policy Adaptation (NRPA) is an approach using online learning policies in a nested structure. It has achieved a great result in a variety of difficult combinatorial optimization problems. In this paper, we propose Meta-NRPA, which combines optimal stopping theory with NRPA for warm-starting and significantly improves the performance of NRPA. We also present several exploratory techniques for NRPA which enable it to perform better exploration. We establish this for three notoriously difficult problems ranging from telecommunication, transportation and coding theory namely Minimum Congestion Shortest Path Routing,  Traveling Salesman Problem with Time Windows and Snake-in-the-Box.
We also improve the lower bounds of the Snake-in-the-Box problem for multiple dimensions.

----

## [1390] A Proof That Using Crossover Can Guarantee Exponential Speed-Ups in Evolutionary Multi-Objective Optimisation

**Authors**: *Duc-Cuong Dang, Andre Opris, Bahare Salehi, Dirk Sudholt*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i10.26460](https://doi.org/10.1609/aaai.v37i10.26460)

**Abstract**:

Evolutionary algorithms are popular algorithms for multiobjective optimisation (also called Pareto optimisation) as they use a population to store trade-offs between different objectives. Despite their popularity, the theoretical foundation of multiobjective evolutionary optimisation (EMO) is still in its early development. Fundamental questions such as the benefits of the crossover operator are still not fully understood.

We provide a theoretical analysis of well-known EMO algorithms GSEMO and NSGA-II to showcase the possible advantages of crossover. We propose a class of problems on which these EMO algorithms using crossover find the Pareto set in expected polynomial time. In sharp contrast, they and many other EMO algorithms without crossover require exponential time to even find a single Pareto-optimal point. This is the first example of an exponential performance gap through the use of crossover for the widely used NSGA-II algorithm.

----

## [1391] Runtime Analysis for the NSGA-II: Provable Speed-Ups from Crossover

**Authors**: *Benjamin Doerr, Zhongdi Qu*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i10.26461](https://doi.org/10.1609/aaai.v37i10.26461)

**Abstract**:

Very recently, the first mathematical runtime analyses for the NSGA-II, the most common multi-objective evolutionary algorithm, have been conducted. Continuing this research direction, we prove that the NSGA-II optimizes the OneJumpZeroJump benchmark asymptotically faster when crossover is employed. Together with a parallel independent work by Dang, Opris, Salehi, and Sudholt, this is the first time such an advantage of crossover is proven for the NSGA-II. Our arguments can be transferred to single-objective optimization. They then prove that crossover can speed up the (mu+1) genetic algorithm in a different way and more pronounced than known before. Our experiments confirm the added value of crossover and show that the observed advantages are even larger than what our proofs can guarantee.

----

## [1392] From Understanding the Population Dynamics of the NSGA-II to the First Proven Lower Bounds

**Authors**: *Benjamin Doerr, Zhongdi Qu*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i10.26462](https://doi.org/10.1609/aaai.v37i10.26462)

**Abstract**:

Due to the more complicated population dynamics of the NSGA-II, none of the existing runtime guarantees for this algorithm is accompanied by a non-trivial lower bound. Via a first mathematical understanding of the population dynamics of the NSGA-II, that is, by estimating the expected number of individuals having a certain objective value, we prove that the NSGA-II with suitable population size needs Omega(Nn log n) function evaluations to find the Pareto front of the OneMinMax problem and Omega(Nn^k)  evaluations on the OneJumpZeroJump problem with jump size k. These bounds are asymptotically tight (that is, they match previously shown upper bounds) and show that the NSGA-II here does not even in terms of the parallel runtime (number of iterations) profit from larger population sizes. For the OneJumpZeroJump problem and when the same sorting is used for the computation of the crowding distance contributions of the two objectives, we even obtain a runtime estimate that is tight including the leading constant.

----

## [1393] Ultrafast Euclidean Shortest Path Computation Using Hub Labeling

**Authors**: *Jinchun Du, Bojie Shen, Muhammad Aamir Cheema*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i10.26463](https://doi.org/10.1609/aaai.v37i10.26463)

**Abstract**:

Finding shortest paths in a Euclidean plane containing polygonal obstacles is a well-studied problem motivated by a variety of real-world applications. 
The state-of-the-art algorithms require finding obstacle corners visible to the source and target, and need to consider potentially a large number of candidate paths. This adversely affects their query processing cost. We address these limitations by proposing a novel adaptation of hub labeling which is the state-of-the-art approach for  shortest distance computation in road networks. Our experimental study conducted on the widely used benchmark maps shows that our approach is typically 1-2  orders of magnitude faster than two state-of-the-art algorithms.

----

## [1394] A Formal Metareasoning Model of Concurrent Planning and Execution

**Authors**: *Amihay Elboher, Ava Bensoussan, Erez Karpas, Wheeler Ruml, Shahaf S. Shperberg, Solomon Eyal Shimony*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i10.26464](https://doi.org/10.1609/aaai.v37i10.26464)

**Abstract**:

Agents that plan and act in the real world must deal with the fact that time passes as they are planning. When timing is tight, there may be insufficient time to complete the search for a plan before it is time to act.  By commencing execution before search concludes, one gains time to search by making planning and execution concurrent. However, this incurs the risk of making incorrect action choices, especially if actions are irreversible. This tradeoff between opportunity and risk is the problem addressed in this paper. Our main contribution is to formally define this setting as an abstract metareasoning problem. We find that the abstract problem is intractable.  However, we identify special cases that are solvable in polynomial time, develop greedy solution algorithms, and, through tests on instances derived from search problems, find several methods that achieve promising practical performance.  This work lays the foundation for a principled time-aware executive that concurrently plans and executes.

----

## [1395] TransPath: Learning Heuristics for Grid-Based Pathfinding via Transformers

**Authors**: *Daniil E. Kirilenko, Anton Andreychuk, Aleksandr Panov, Konstantin S. Yakovlev*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i10.26465](https://doi.org/10.1609/aaai.v37i10.26465)

**Abstract**:

Heuristic search algorithms, e.g. A*, are the commonly used tools for pathfinding on grids, i.e. graphs of regular structure that are widely employed to represent environments in robotics, video games, etc. Instance-independent heuristics for grid graphs, e.g. Manhattan distance, do not take the obstacles into account, and thus the search led by such heuristics performs poorly in obstacle-rich environments. To this end, we suggest learning the instance-dependent heuristic proxies that are supposed to notably increase the efficiency of the search. The first heuristic proxy we suggest to learn is the correction factor, i.e. the ratio between the instance-independent cost-to-go estimate and the perfect one (computed offline at the training phase). Unlike learning the absolute values of the cost-to-go heuristic function, which was known before, learning the correction factor utilizes the knowledge of the instance-independent heuristic. The second heuristic proxy is the path probability, which indicates how likely the grid cell is lying on the shortest path. This heuristic can be employed in the Focal Search framework as the secondary heuristic, allowing us to preserve the guarantees on the bounded sub-optimality of the solution. We learn both suggested heuristics in a supervised fashion with the state-of-the-art neural networks containing attention blocks (transformers). We conduct a thorough empirical evaluation on a comprehensive dataset of planning tasks, showing that the suggested techniques i) reduce the computational effort of the A* up to a factor of 4x while producing the solutions, whose costs exceed those of the optimal solutions by less than 0.3% on average; ii) outperform the competitors, which include the conventional techniques from the heuristic search, i.e. weighted A*, as well as the state-of-the-art learnable planners.

The project web-page is: https://airi-institute.github.io/TransPath/.

----

## [1396] Large-State Reinforcement Learning for Hyper-Heuristics

**Authors**: *Lucas Kletzander, Nysret Musliu*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i10.26466](https://doi.org/10.1609/aaai.v37i10.26466)

**Abstract**:

Hyper-heuristics are a domain-independent problem solving approach where the main task is to select effective chains of problem-specific low-level heuristics on the fly for an unseen instance. This task can be seen as a reinforcement learning problem, however, the information available to the hyper-heuristic is very limited, usually leading to very limited state representations. In this work, for the first time we use the trajectory of solution changes for a larger set of features for reinforcement learning in the novel hyper-heuristic LAST-RL (Large-State Reinforcement Learning). Further, we introduce a probability distribution for the exploration case in our epsilon-greedy policy that is based on the idea of Iterated Local Search to increase the chance to sample good chains of low-level heuristics. The benefit of the collaboration of our novel components is shown on the academic benchmark of the Cross Domain Heuristic Challenge 2011 consisting of six different problem domains. Our approach can provide state-of-the-art results on this benchmark where it outperforms recent hyper-heuristics based on reinforcement learning, and also demonstrates high performance on a benchmark of complex real-life personnel scheduling domains.

----

## [1397] Human Assisted Learning by Evolutionary Multi-Objective Optimization

**Authors**: *Dan-Xuan Liu, Xin Mu, Chao Qian*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i10.26467](https://doi.org/10.1609/aaai.v37i10.26467)

**Abstract**:

Machine learning models have liberated manpower greatly in many real-world tasks, but their predictions are still worse than humans on some specific instances. To improve the performance, it is natural to optimize machine learning models to take decisions for most instances while delivering a few tricky instances to humans, resulting in the problem of Human Assisted Learning (HAL). Previous works mainly formulated HAL as a constrained optimization problem that tries to find a limited subset of instances for human decision such that the sum of model and human errors can be minimized; and employed the greedy algorithms, whose performance, however, may be limited due to the greedy nature. In this paper, we propose a new framework HAL-EMO based on Evolutionary Multi-objective Optimization, which reformulates HAL as a bi-objective optimization problem that minimizes the number of selected instances for human decision and the total errors simultaneously, and employs a Multi-Objective Evolutionary Algorithm (MOEA) to solve it. We implement HAL-EMO using two MOEAs, the popular NSGA-II as well as the theoretically grounded GSEMO. We also propose a specific MOEA, called BSEMO, with biased selection and balanced mutation for HAL-EMO, and prove that for human assisted regression and classification, HAL-EMO using BSEMO can achieve better and same theoretical guarantees than previous greedy algorithms, respectively. Experiments on the tasks of medical diagnosis and content moderation show the superiority of HAL-EMO (with either NSGA-II, GSEMO or BSEMO) over previous algorithms, and that using BSEMO leads to the best performance of HAL-EMO.

----

## [1398] OPT-GAN: A Broad-Spectrum Global Optimizer for Black-Box Problems by Learning Distribution

**Authors**: *Minfang Lu, Shuai Ning, Shuangrong Liu, Fengyang Sun, Bo Zhang, Bo Yang, Lin Wang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i10.26468](https://doi.org/10.1609/aaai.v37i10.26468)

**Abstract**:

Black-box optimization (BBO) algorithms are concerned with finding the best solutions for problems with missing analytical details. Most classical methods for such problems are based on strong and fixed a priori assumptions, such as Gaussianity. However, the complex real-world problems, especially when the global optimum is desired, could be very far from the a priori assumptions because of their diversities, causing unexpected obstacles.   In this study, we propose a generative adversarial net-based broad-spectrum global optimizer (OPT-GAN) which estimates the distribution of optimum gradually, with strategies to balance exploration-exploitation trade-off. It has potential to better adapt to the regularity and structure of diversified landscapes than other methods with fixed prior, e.g., Gaussian assumption or separability. 
Experiments on diverse BBO benchmarks and high dimensional real world applications exhibit that OPT-GAN outperforms other traditional and neural net-based BBO algorithms. The code and Appendix are available at https://github.com/NBICLAB/OPT-GAN

----

## [1399] Analyzing and Improving the Use of the FastMap Embedding in Pathfinding Tasks

**Authors**: *Reza Mashayekhi, Dor Atzmon, Nathan R. Sturtevant*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i10.26469](https://doi.org/10.1609/aaai.v37i10.26469)

**Abstract**:

The FastMap algorithm has been proposed as an inexpensive metric embedding which provides admissible distance estimates between all vertices in an embedding. As an embedding, it also supports additional operations such as taking the median location of two vertices, which is important in some problems. This paper studies several aspects of FastMap embeddings, showing the relationship of FastMap to general additive heuristics. As an admissible heuristic, FastMap is not as strong as previous suggested. However, by combining FastMap with the ideas of differential heuristics, we can significantly improve the performance of FastMap heuristics. We show the impact of these ideas in both single-agent pathfinding and the Multi-Agent Meeting problem, where the performance of algorithms using our improved FastMap embedding is improved by up to a factor of two.

----



[Go to the previous page](AAAI-2023-list06.md)

[Go to the next page](AAAI-2023-list08.md)

[Go to the catalog section](README.md)