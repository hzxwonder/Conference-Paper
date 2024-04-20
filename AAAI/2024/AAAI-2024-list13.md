## [2400] Robust Stochastic Graph Generator for Counterfactual Explanations

**Authors**: *Mario Alfonso Prado-Romero, Bardh Prenkaj, Giovanni Stilo*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i19.30149](https://doi.org/10.1609/aaai.v38i19.30149)

**Abstract**:

Counterfactual Explanation (CE) techniques have garnered attention as a means to provide insights to the users engaging with AI systems. While extensively researched in domains such as medical imaging and autonomous vehicles, Graph Counterfactual Explanation (GCE) methods have been comparatively under-explored. GCEs generate a new graph similar to the original one, with a different outcome grounded on the underlying predictive model. Among these GCE techniques, those rooted in generative mechanisms have received relatively limited investigation despite demonstrating impressive accomplishments in other domains, such as artistic styles and natural language modelling. The preference for generative explainers stems from their capacity to generate counterfactual instances during inference, leveraging autonomously acquired perturbations of the input graph. Motivated by the rationales above, our study introduces RSGG-CE, a novel Robust Stochastic Graph Generator for Counterfactual Explanations able to produce counterfactual examples from the learned latent space considering a partially ordered generation sequence. Furthermore, we undertake quantitative and qualitative analyses to compare RSGG-CE's performance against SoA generative explainers, highlighting its increased ability to engendering plausible counterfactual candidates.

----

## [2401] Visual Adversarial Examples Jailbreak Aligned Large Language Models

**Authors**: *Xiangyu Qi, Kaixuan Huang, Ashwinee Panda, Peter Henderson, Mengdi Wang, Prateek Mittal*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i19.30150](https://doi.org/10.1609/aaai.v38i19.30150)

**Abstract**:

Warning: this paper contains data, prompts, and model outputs that are offensive in nature.

Recently, there has been a surge of interest in integrating vision into Large Language Models (LLMs), exemplified by Visual Language Models (VLMs) such as Flamingo and GPT-4. This paper sheds light on the security and safety implications of this trend. First, we underscore that the continuous and high-dimensional nature of the visual input makes it a weak link against adversarial attacks, representing an expanded attack surface of vision-integrated LLMs. Second, we highlight that the versatility of LLMs also presents visual attackers with a wider array of achievable adversarial objectives, extending the implications of security failures beyond mere misclassification. As an illustration, we present a case study in which we exploit visual adversarial examples to circumvent the safety guardrail of aligned LLMs with integrated vision. Intriguingly, we discover that a single visual adversarial example can universally jailbreak an aligned LLM, compelling it to heed a wide range of harmful instructions (that it otherwise would not) and generate harmful content that transcends the narrow scope of a `few-shot' derogatory corpus initially employed to optimize the adversarial example. Our study underscores the escalating adversarial risks associated with the pursuit of multimodality. Our findings also connect the long-studied adversarial vulnerabilities of neural networks to the nascent field of AI alignment. The presented attack suggests a fundamental adversarial challenge for AI alignment, especially in light of the emerging trend toward multimodality in frontier foundation models.

----

## [2402] Dissenting Explanations: Leveraging Disagreement to Reduce Model Overreliance

**Authors**: *Omer Reingold, Judy Hanwen Shen, Aditi Talati*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i19.30151](https://doi.org/10.1609/aaai.v38i19.30151)

**Abstract**:

While modern explanation methods have been shown to be inconsistent and contradictory, the explainability of black-box models nevertheless remains desirable. When the role of explanations extends from understanding models to aiding decision making, the semantics of explanations is not always fully understood – to what extent do explanations ``explain” a decision and to what extent do they merely advocate for a decision? Can we help humans gain insights from explanations accompanying correct predictions and not over-rely on incorrect predictions advocated for by explanations? With this perspective in mind, we introduce the notion of dissenting explanations: conflicting predictions with accompanying explanations. We first explore the advantage of dissenting explanations in the setting of model multiplicity, where multiple models with similar performance may have different predictions. Through a human study on the task of identifying deceptive reviews, we demonstrate that dissenting explanations reduce overreliance on model predictions, without reducing overall accuracy. Motivated by the utility of dissenting explanations we present both global and local methods for their generation.

----

## [2403] I-CEE: Tailoring Explanations of Image Classification Models to User Expertise

**Authors**: *Yao Rong, Peizhu Qian, Vaibhav V. Unhelkar, Enkelejda Kasneci*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i19.30152](https://doi.org/10.1609/aaai.v38i19.30152)

**Abstract**:

Effectively explaining decisions of black-box machine learning models is critical to responsible deployment of AI systems that rely on them. Recognizing their importance, the field of explainable AI (XAI) provides several techniques to generate these explanations. Yet, there is relatively little emphasis on the user (the explainee) in this growing body of work and most XAI techniques generate "one-size-fits-all'' explanations. To bridge this gap and achieve a step closer towards human-centered XAI, we present I-CEE, a framework that provides Image Classification Explanations tailored to User Expertise. Informed by existing work, I-CEE explains the decisions of image classification models by providing the user with an informative subset of training data (i.e., example images), corresponding local explanations, and model decisions. However, unlike prior work, I-CEE models the informativeness of the example images to depend on user expertise, resulting in different examples for different users. We posit that by tailoring the example set to user expertise, I-CEE can better facilitate users' understanding and simulatability of the model. To evaluate our approach, we conduct detailed experiments in both simulation and with human participants (N = 100) on multiple datasets. Experiments with simulated users show that I-CEE improves users' ability to accurately predict the model's decisions (simulatability) compared to baselines, providing promising preliminary results. Experiments with human participants demonstrate that our method significantly improves user simulatability accuracy, highlighting the importance of human-centered XAI.

----

## [2404] A Simple and Practical Method for Reducing the Disparate Impact of Differential Privacy

**Authors**: *Lucas Rosenblatt, Julia Stoyanovich, Christopher Musco*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i19.30153](https://doi.org/10.1609/aaai.v38i19.30153)

**Abstract**:

Differentially private (DP) mechanisms have been deployed in a variety of high-impact social settings (perhaps most notably by the U.S. Census). Since all DP mechanisms involve adding noise to results of statistical queries, they are expected to impact our ability to accurately analyze and learn from data, in effect trading off privacy with utility. Alarmingly, the impact of DP on utility can vary significantly among different sub-populations. A simple way to reduce this disparity is with stratification. First compute an independent private estimate for each group in the data set (which may be the intersection of several protected classes), then, to compute estimates of global statistics, appropriately recombine these group estimates. Our main observation is that naive stratification often yields high-accuracy estimates of population-level statistics, without the need for additional privacy budget. We support this observation theoretically and empirically. Our theoretical results center on the private mean estimation problem, while our empirical results center on extensive experiments on private data synthesis to demonstrate the effectiveness of stratification on a variety of private mechanisms. Overall, we argue that this straightforward approach provides a strong baseline against which future work on reducing utility disparities of DP mechanisms should be compared.

----

## [2405] Interpretability Benchmark for Evaluating Spatial Misalignment of Prototypical Parts Explanations

**Authors**: *Mikolaj Sacha, Bartosz Jura, Dawid Rymarczyk, Lukasz Struski, Jacek Tabor, Bartosz Zielinski*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i19.30154](https://doi.org/10.1609/aaai.v38i19.30154)

**Abstract**:

Prototypical parts-based networks are becoming increasingly popular due to their faithful self-explanations. However, their similarity maps are calculated in the penultimate network layer. Therefore, the receptive field of the prototype activation region often depends on parts of the image outside this region, which can lead to misleading interpretations. We name this undesired behavior a spatial explanation misalignment and introduce an interpretability benchmark with a set of dedicated metrics for quantifying this phenomenon. In addition, we propose a method for misalignment compensation and apply it to existing state-of-the-art models. We show the expressiveness of our benchmark and the effectiveness of the proposed compensation methodology through extensive empirical studies.

----

## [2406] Human-Guided Moral Decision Making in Text-Based Games

**Authors**: *Zijing Shi, Meng Fang, Ling Chen, Yali Du, Jun Wang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i19.30155](https://doi.org/10.1609/aaai.v38i19.30155)

**Abstract**:

Training reinforcement learning (RL) agents to achieve desired goals while also acting morally is a challenging problem. Transformer-based language models (LMs) have shown some promise in moral awareness, but their use in different contexts is problematic because of the complexity and implicitness of human morality. In this paper, we build on text-based games, which are challenging environments for current RL agents, and propose the HuMAL (Human-guided Morality Awareness Learning) algorithm, which adaptively learns personal values through human-agent collaboration with minimal manual feedback. We evaluate HuMAL on the Jiminy Cricket benchmark, a set of text-based games with various scenes and dense morality annotations, using both simulated and actual human feedback. The experimental results demonstrate that with a small amount of human feedback, HuMAL can improve task performance and reduce immoral behavior in a variety of games and is adaptable to different personal values.

----

## [2407] Towards Fairer Centroids in K-means Clustering

**Authors**: *Stanley Simoes, Deepak P, Muiris MacCarthaigh*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i19.30156](https://doi.org/10.1609/aaai.v38i19.30156)

**Abstract**:

There has been much recent interest in developing fair clustering algorithms that seek to do justice to the representation of groups defined along sensitive attributes such as race and sex. Within the centroid clustering paradigm, these algorithms are seen to generate clusterings where different groups are disadvantaged within different clusters with respect to their representativity, i.e., distance to centroid. In view of this deficiency, we propose a novel notion of cluster-level centroid fairness that targets the representativity unfairness borne by groups within each cluster, along with a metric to quantify the same. Towards operationalising this notion, we draw on ideas from political philosophy aligned with consideration for the worst-off group to develop Fair-Centroid; a new clustering method that focusses on enhancing the representativity of the worst-off group within each cluster. Our method uses an iterative optimisation paradigm wherein an initial cluster assignment is refined by reassigning objects to clusters such that the worst-off group in each cluster is benefitted. We compare our notion with a related fairness notion and show through extensive empirical evaluations on real-world datasets that our method significantly enhances cluster-level centroid fairness at low impact on cluster coherence.

----

## [2408] Toward Robustness in Multi-Label Classification: A Data Augmentation Strategy against Imbalance and Noise

**Authors**: *Hwanjun Song, Minseok Kim, Jae-Gil Lee*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i19.30157](https://doi.org/10.1609/aaai.v38i19.30157)

**Abstract**:

Multi-label classification poses challenges due to imbalanced and noisy labels in training data. In this paper, we propose a unified data augmentation method, named BalanceMix, to address these challenges. Our approach includes two samplers for imbalanced labels, generating minority-augmented instances with high diversity. It also refines multi-labels at the label-wise granularity, categorizing noisy labels as clean, re-labeled, or ambiguous for robust optimization. Extensive experiments on three benchmark datasets demonstrate that BalanceMix outperforms existing state-of-the-art methods. We release the code at https://github.com/DISL-Lab/BalanceMix.

----

## [2409] Bidirectional Contrastive Split Learning for Visual Question Answering

**Authors**: *Yuwei Sun, Hideya Ochiai*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i19.30158](https://doi.org/10.1609/aaai.v38i19.30158)

**Abstract**:

Visual Question Answering (VQA) based on multi-modal data facilitates real-life applications such as home robots and medical diagnoses. One significant challenge is to devise a robust decentralized learning framework for various client models where centralized data collection is refrained due to confidentiality concerns. This work aims to tackle privacy-preserving VQA by decoupling a multi-modal model into representation modules and a contrastive module, leveraging inter-module gradients sharing and inter-client weight sharing. To this end, we propose Bidirectional Contrastive Split Learning (BiCSL) to train a global multi-modal model on the entire data distribution of decentralized clients. We employ the contrastive loss that enables a more efficient self-supervised learning of decentralized modules. Comprehensive experiments are conducted on the VQA-v2 dataset based on five SOTA VQA models, demonstrating the effectiveness of the proposed method. Furthermore, we inspect BiCSL's robustness against a dual-key backdoor attack on VQA. Consequently, BiCSL shows significantly enhanced resilience when exposed to the multi-modal adversarial attack compared to the centralized learning method, which provides a promising approach to decentralized multi-modal learning.

----

## [2410] Quantile-Based Maximum Likelihood Training for Outlier Detection

**Authors**: *Masoud Taghikhah, Nishant Kumar, Sinisa Segvic, Abouzar Eslami, Stefan Gumhold*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i19.30159](https://doi.org/10.1609/aaai.v38i19.30159)

**Abstract**:

Discriminative learning effectively predicts true object class for image classification. However, it often results in false positives for outliers, posing critical concerns in applications like autonomous driving and video surveillance systems. Previous attempts to address this challenge involved training image classifiers through contrastive learning using actual outlier data or synthesizing outliers for self-supervised learning. Furthermore, unsupervised generative modeling of inliers in pixel space has shown limited success for outlier detection. In this work, we introduce a quantile-based maximum likelihood objective for learning the inlier distribution to improve the outlier separation during inference. Our approach fits a normalizing flow to pre-trained discriminative features and detects the outliers according to the evaluated log-likelihood. The experimental evaluation demonstrates the effectiveness of our method as it surpasses the performance of the state-of-the-art unsupervised methods for outlier detection. The results are also competitive compared with a recent self-supervised approach for outlier detection. Our work allows to reduce dependency on well-sampled negative training data, which is especially important for domains like medical diagnostics or remote sensing.

----

## [2411] Sparsity-Guided Holistic Explanation for LLMs with Interpretable Inference-Time Intervention

**Authors**: *Zhen Tan, Tianlong Chen, Zhenyu Zhang, Huan Liu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i19.30160](https://doi.org/10.1609/aaai.v38i19.30160)

**Abstract**:

Large Language Models (LLMs) have achieved unprecedented breakthroughs in various natural language processing domains. However, the enigmatic ``black-box'' nature of LLMs remains a significant challenge for interpretability, hampering transparent and accountable applications. While past approaches, such as attention visualization, pivotal subnetwork extraction, and concept-based analyses, offer some insight, they often focus on either local or global explanations within a single dimension, occasionally falling short in providing comprehensive clarity. In response, we propose a novel methodology anchored in sparsity-guided techniques, aiming to provide a holistic interpretation of LLMs. Our framework, termed SparseCBM, innovatively integrates sparsity to elucidate three intertwined layers of interpretation: input, subnetwork, and concept levels. In addition, the newly introduced dimension of interpretable inference-time intervention facilitates dynamic adjustments to the model during deployment. Through rigorous empirical evaluations on real-world datasets, we demonstrate that SparseCBM delivers a profound understanding of LLM behaviors, setting it apart in both interpreting and ameliorating model inaccuracies. Codes are provided in supplements.

----

## [2412] Toward More Generalized Malicious URL Detection Models

**Authors**: *Yun-Da Tsai, Cayon Liow, Yin Sheng Siang, Shou-De Lin*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i19.30161](https://doi.org/10.1609/aaai.v38i19.30161)

**Abstract**:

This paper reveals a data bias issue that can profoundly hinder the performance of machine learning models in malicious URL detection. We describe how such bias can be diagnosed using interpretable machine learning techniques and further argue that such biases naturally exist in the real world security data for training a classification model. To counteract these challenges, we propose a debiased training strategy that can be applied to most deep-learning based models to alleviate the negative effects of the biased features. The solution is based on the technique of adversarial training to train deep neural networks learning invariant embedding from biased data. Through extensive experimentation, we substantiate that our innovative strategy fosters superior generalization capabilities across both CNN-based and RNN-based detection models. The findings presented in this work not only expose a latent issue in the field but also provide an actionable remedy, marking a significant step forward in the pursuit of more reliable and robust malicious URL detection.

----

## [2413] Self-Supervised Likelihood Estimation with Energy Guidance for Anomaly Segmentation in Urban Scenes

**Authors**: *Yuanpeng Tu, Yuxi Li, Boshen Zhang, Liang Liu, Jiangning Zhang, Yabiao Wang, Cairong Zhao*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i19.30162](https://doi.org/10.1609/aaai.v38i19.30162)

**Abstract**:

Robust autonomous driving requires agents to accurately identify unexpected areas (anomalies) in urban scenes. To this end, some critical issues remain open: how to design advisable metric to measure anomalies, and how to properly generate training samples of anomaly data? Classical effort in anomaly detection usually resorts to pixel-wise uncertainty or sample synthesis, which ignores the contextual information and sometimes requires auxiliary data with fine-grained annotations. On the contrary, in this paper, we exploit the strong context-dependent nature of segmentation task and design an energy-guided self-supervised frameworks for anomaly segmentation, which optimizes an anomaly head by maximizing likelihood of self-generated anomaly pixels. For this purpose, we design two estimators to model anomaly likelihood, one is a task-agnostic binary estimator and the other depicts the likelihood as residual of task-oriented joint energy. Based on proposed estimators, we devise an adaptive self-supervised training framework, which exploits the contextual reliance and estimated likelihood to refine mask annotations in anomaly areas. We conduct extensive experiments on challenging Fishyscapes and Road Anomaly benchmarks, demonstrating that without any auxiliary data or synthetic models, our method can still achieves comparable performance to supervised competitors. Code is available at https://github.com/yuanpengtu/SLEEG.

----

## [2414] Pure-Past Action Masking

**Authors**: *Giovanni Varricchione, Natasha Alechina, Mehdi Dastani, Giuseppe De Giacomo, Brian Logan, Giuseppe Perelli*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i19.30163](https://doi.org/10.1609/aaai.v38i19.30163)

**Abstract**:

We present Pure-Past Action Masking (PPAM), a lightweight approach to action masking for safe reinforcement learning. In PPAM, actions are disallowed (“masked”) according to specifications expressed in Pure-Past Linear Temporal Logic (PPLTL). PPAM can enforce non-Markovian constraints, i.e., constraints based on the history of the system, rather than just the current state of the (possibly hidden) MDP. The features used in the safety constraint need not be the same as those used by the learning agent, allowing a clear separation of concerns between the safety constraints and reward specifications of the (learning) agent. We prove formally that an agent trained with PPAM can learn any optimal policy that satisfies the safety constraints, and that they are as expressive as shields, another approach to enforce non-Markovian constraints in RL. Finally, we provide empirical results showing how PPAM can guarantee constraint satisfaction in practice.

----

## [2415] Long-Term Safe Reinforcement Learning with Binary Feedback

**Authors**: *Akifumi Wachi, Wataru Hashimoto, Kazumune Hashimoto*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i19.30164](https://doi.org/10.1609/aaai.v38i19.30164)

**Abstract**:

Safety is an indispensable requirement for applying reinforcement learning (RL) to real problems. Although there has been a surge of safe RL algorithms proposed in recent years, most existing work typically 1) relies on receiving numeric safety feedback; 2) does not guarantee safety during the learning process; 3) limits the problem to a priori known, deterministic transition dynamics; and/or 4) assume the existence of a known safe policy for any states. Addressing the issues mentioned above, we thus propose Long-term Binary-feedback Safe RL (LoBiSaRL), a safe RL algorithm for constrained Markov decision processes (CMDPs) with binary safety feedback and an unknown, stochastic state transition function. LoBiSaRL optimizes a policy to maximize rewards while guaranteeing long-term safety that an agent executes only safe state-action pairs throughout each episode with high probability. Specifically, LoBiSaRL models the binary safety function via a generalized linear model (GLM) and conservatively takes only a safe action at every time step while inferring its effect on future safety under proper assumptions. Our theoretical results show that LoBiSaRL guarantees the long-term safety constraint, with high probability. Finally, our empirical results demonstrate that our algorithm is safer than existing methods without significantly compromising performance in terms of reward.

----

## [2416] Identifying Reasons for Bias: An Argumentation-Based Approach

**Authors**: *Madeleine Waller, Odinaldo Rodrigues, Oana Cocarascu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i19.30165](https://doi.org/10.1609/aaai.v38i19.30165)

**Abstract**:

As algorithmic decision-making systems become more prevalent in society, ensuring the fairness of these systems is becoming increasingly important. Whilst there has been substantial research in building fair algorithmic decision-making systems, the majority of these methods require access to the training data, including personal characteristics, and are not transparent regarding which individuals are classified unfairly. In this paper, we propose a novel model-agnostic argumentation-based method to determine why an individual is classified differently in comparison to similar individuals. Our method uses a quantitative argumentation framework to represent attribute-value pairs of an individual and of those similar to them, and uses a well-known semantics to identify the attribute-value pairs in the individual contributing most to their different classification. We evaluate our method on two datasets commonly used in the fairness literature and illustrate its effectiveness in the identification of bias.

----

## [2417] Would You Like Your Data to Be Trained? A User Controllable Recommendation Framework

**Authors**: *Lei Wang, Xu Chen, Zhenhua Dong, Quanyu Dai*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i19.30166](https://doi.org/10.1609/aaai.v38i19.30166)

**Abstract**:

Recommender systems have a significant impact on various real-world applications, shaping people's daily lives and enhancing productivity. Traditional recommender models aim to collect extensive user information to accurately estimate user preferences. However, in practical scenarios, users may not want all their behaviors to be included in the model training process. This paper introduces a novel recommendation paradigm that allows users to indicate their ``willingness'' regarding which data should contribute to model training. The models are then optimized to maximize utility, which considers the trade-off between recommendation performance and respecting user preferences. The recommendation problem is formulated as a multiplayer game, with each user acting as a player and using a selection vector to indicate their willingness to include specific interacted items in training. To efficiently solve this game, an influence function-based model is proposed to approximate recommendation performances for different actions without re-optimizing the model. Furthermore, an enhanced model leveraging multiple anchor actions for the influence function is introduced to improve performance approximation accuracy. The convergence rate of the algorithm is theoretically analyzed, and the advantages of incorporating multiple anchor actions are demonstrated. Extensive experiments on both simulated and real-world datasets validate the effectiveness of the proposed models in balancing recommendation quality and user willingness. To promote this research direction, we have released our project at https://paitesanshi.github.io/IFRQE/.

----

## [2418] Moderate Message Passing Improves Calibration: A Universal Way to Mitigate Confidence Bias in Graph Neural Networks

**Authors**: *Min Wang, Hao Yang, Jincai Huang, Qing Cheng*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i19.30167](https://doi.org/10.1609/aaai.v38i19.30167)

**Abstract**:

Confidence calibration in Graph Neural Networks (GNNs) aims to align a model's predicted confidence with its actual accuracy. Recent studies have indicated that GNNs exhibit an under-confidence bias, which contrasts the over-confidence bias commonly observed in deep neural networks. However, our deeper investigation into this topic reveals that not all GNNs exhibit this behavior. Upon closer examination of message passing in GNNs, we found a clear link between message aggregation and confidence levels. Specifically, GNNs with extensive message aggregation, often seen in deep architectures or when leveraging large amounts of labeled data, tend to exhibit overconfidence. This overconfidence can be attributed to factors like over-learning and over-smoothing. Conversely, GNNs with fewer layers, known for their balanced message passing and superior node representation, may exhibit under-confidence. To counter these confidence biases, we introduce the Adaptive Unified Label Smoothing (AU-LS) technique. Our experiments show that AU-LS outperforms existing methods, addressing both over and under-confidence in various GNN scenarios.

----

## [2419] Generating Diagnostic and Actionable Explanations for Fair Graph Neural Networks

**Authors**: *Zhenzhong Wang, Qingyuan Zeng, Wanyu Lin, Min Jiang, Kay Chen Tan*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i19.30168](https://doi.org/10.1609/aaai.v38i19.30168)

**Abstract**:

A plethora of fair graph neural networks (GNNs) have been proposed to promote algorithmic fairness for high-stake real-life contexts. Meanwhile, explainability is generally proposed to help machine learning practitioners debug models by providing human-understandable explanations. However, seldom work on explainability is made to generate explanations for fairness diagnosis in GNNs. From the explainability perspective, this paper explores the problem of what subgraph patterns cause the biased behavior of GNNs, and what actions could practitioners take to rectify the bias? By answering the two questions, this paper aims to produce compact, diagnostic, and actionable explanations that are responsible for discriminatory behavior. Specifically, we formulate the problem of generating diagnostic and actionable explanations as a multi-objective combinatorial optimization problem. To solve the problem, a dedicated multi-objective evolutionary algorithm is presented to ensure GNNs' explainability and fairness in one go. In particular, an influenced nodes-based gradient approximation is developed to boost the computation efficiency of the evolutionary algorithm. We provide a theoretical analysis to illustrate the effectiveness of the proposed framework. Extensive experiments have been conducted to demonstrate the superiority of the proposed method in terms of classification performance, fairness, and interpretability.

----

## [2420] Physics-Informed Representation and Learning: Control and Risk Quantification

**Authors**: *Zhuoyuan Wang, Reece Keller, Xiyu Deng, Kenta Hoshino, Takashi Tanaka, Yorie Nakahira*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i19.30169](https://doi.org/10.1609/aaai.v38i19.30169)

**Abstract**:

Optimal and safety-critical control are fundamental problems for stochastic systems, and are widely considered in real-world scenarios such as robotic manipulation and autonomous driving. In this paper, we consider the problem of efficiently finding optimal and safe control for high-dimensional systems. Specifically, we propose to use dimensionality reduction techniques from a comparison theorem for stochastic differential equations together with a generalizable physics-informed neural network to estimate the optimal value function and the safety probability of the system. The proposed framework results in substantial sample efficiency improvement compared to existing methods. We further develop an autoencoder-like neural network to automatically identify the low-dimensional features in the system to enhance the ease of design for system integration. We also provide experiments and quantitative analysis to validate the efficacy of the proposed method. 
Source code is available at https://github.com/jacobwang925/path-integral-PINN.

----

## [2421] Safe Reinforcement Learning with Instantaneous Constraints: The Role of Aggressive Exploration

**Authors**: *Honghao Wei, Xin Liu, Lei Ying*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i19.30170](https://doi.org/10.1609/aaai.v38i19.30170)

**Abstract**:

This paper studies safe Reinforcement Learning (safe RL) with linear function approximation and under hard instantaneous constraints where unsafe actions must be avoided at each step. Existing studies have considered safe RL with hard instantaneous constraints, but their approaches rely on several key assumptions: (i) the RL agent knows a safe action set for every state or knows a safe graph in which all the state-action-state triples are safe, and (ii) the constraint/cost functions are linear. In this paper, we consider safe RL with instantaneous hard constraints without assumption (i) and generalize (ii) to Reproducing Kernel Hilbert Space (RKHS). Our proposed algorithm, LSVI-AE, achieves O(√{d³H⁴K}) regret and O(H √{dK})  hard constraint violation when the cost function is linear and O(H?ₖ √{K}) hard constraint violation when the cost function belongs to RKHS. Here K is the learning horizon, H is the length of each episode, and ?ₖ is the information gain w.r.t the kernel used to approximate cost functions. Our results achieve the optimal dependency on the learning horizon K, matching the lower bound we provide in this paper and demonstrating the efficiency of LSVI-AE. Notably, the design of our approach encourages aggressive policy exploration, providing a unique perspective on safe RL with general cost functions and no prior knowledge of safe actions,  which may be of independent interest.

----

## [2422] Concealing Sensitive Samples against Gradient Leakage in Federated Learning

**Authors**: *Jing Wu, Munawar Hayat, Mingyi Zhou, Mehrtash Harandi*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i19.30171](https://doi.org/10.1609/aaai.v38i19.30171)

**Abstract**:

Federated Learning (FL) is a distributed learning paradigm that enhances users' privacy by eliminating the need for clients to share raw, private data with the server.
Despite the success, recent studies expose the vulnerability of FL to model inversion attacks, where adversaries reconstruct users’ private data via eavesdropping on the shared gradient information. We hypothesize that a key factor in the success of such attacks is the low entanglement among gradients per data within the batch during stochastic optimization. This creates a vulnerability that an adversary can exploit to reconstruct the sensitive data. Building upon this insight, we present a simple, yet effective defense strategy that obfuscates the gradients of the sensitive data with concealed samples. To achieve this, we propose synthesizing concealed samples to mimic the sensitive data at the gradient level while ensuring their visual dissimilarity from the actual sensitive data. Compared to the previous art, our empirical evaluations suggest that the proposed technique provides the strongest protection while simultaneously maintaining the FL performance. Code is located at https://github.com/JingWu321/DCS-2.

----

## [2423] The Evidence Contraction Issue in Deep Evidential Regression: Discussion and Solution

**Authors**: *Yuefei Wu, Bin Shi, Bo Dong, Qinghua Zheng, Hua Wei*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i19.30172](https://doi.org/10.1609/aaai.v38i19.30172)

**Abstract**:

Deep Evidential Regression (DER) places a prior on the original Gaussian likelihood and treats learning as an evidence acquisition process to quantify uncertainty. For the validity of the evidence theory, DER requires specialized activation functions to ensure that the prior parameters remain non-negative. However, such constraints will trigger evidence contraction, causing sub-optimal performance. In this paper, we analyse DER theoretically, revealing the intrinsic limitations for sub-optimal performance: the non-negativity constraints on the Normal Inverse-Gamma (NIG) prior parameter trigger the evidence contraction under the specialized activation function, which hinders the optimization of DER performance. On this basis, we design a Non-saturating Uncertainty Regularization term, which effectively ensures that the performance is further optimized in the right direction. Experiments on real-world datasets show that our proposed approach improves the performance of DER while maintaining the ability to quantify uncertainty.

----

## [2424] Byzantine-Robust Decentralized Learning via Remove-then-Clip Aggregation

**Authors**: *Caiyi Yang, Javad Ghaderi*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i19.30173](https://doi.org/10.1609/aaai.v38i19.30173)

**Abstract**:

We consider decentralized learning over a network of workers with heterogeneous datasets, in the presence of Byzantine workers. 
Byzantine workers may transmit arbitrary or malicious values to neighboring workers, leading to degradation in overall performance. The heterogeneous nature of the training data across various workers complicates the identification and mitigation of Byzantine workers.  
To address this complex problem, we introduce a resilient decentralized learning approach that combines the gradient descent algorithm with a novel robust aggregator. Specifically, we propose a remove-then-clip aggregator, whereby each benign worker meticulously filters the neighbors' values and subsequently projects the remaining values to a sphere centered at its local value, with an appropriately selected radius.
We prove that our proposed method converges to a neighborhood of a stationary point for non-convex objectives under standard assumptions. Furthermore, empirical evaluations are provided to demonstrate the superior performance of our method in comparison to existing algorithms, under various Byzantine attack models.

----

## [2425] Hypothesis Testing for Class-Conditional Noise Using Local Maximum Likelihood

**Authors**: *Weisong Yang, Rafael Poyiadzi, Niall Twomey, Raúl Santos-Rodríguez*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i19.30174](https://doi.org/10.1609/aaai.v38i19.30174)

**Abstract**:

In supervised learning, automatically assessing the quality of the labels before any learning takes place remains an open research question. In certain particular cases, hypothesis testing procedures have been proposed to assess whether a given instance-label dataset is contaminated with class-conditional label noise, as opposed to uniform label noise. The existing theory builds on the asymptotic properties of the Maximum Likelihood Estimate for parametric logistic regression. However, the parametric assumptions on top of which these approaches are constructed are often too strong and unrealistic in practice. To alleviate this problem, in this paper we propose an alternative path by showing how similar procedures can be followed when the underlying model is a product of Local Maximum Likelihood Estimation that leads to more flexible nonparametric logistic regression models, which in turn are less susceptible to model misspecification. This different view allows for wider applicability of the tests by offering users access to a richer model class. Similarly to existing works, we assume we have access to anchor points which are provided by the users. We introduce the necessary ingredients for the adaptation of the hypothesis tests to the case of nonparametric logistic regression and empirically compare against the parametric approach presenting both synthetic and real-world case studies and discussing the advantages and limitations of the proposed approach.

----

## [2426] Providing Fair Recourse over Plausible Groups

**Authors**: *Jayanth Yetukuri, Ian Hardy, Yevgeniy Vorobeychik, Berk Ustun, Yang Liu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i19.30175](https://doi.org/10.1609/aaai.v38i19.30175)

**Abstract**:

Machine learning models now automate decisions in applications where we may wish to provide recourse to adversely affected individuals. In practice, existing methods to provide recourse return actions that fail to account for latent characteristics that are not captured in the model (e.g., age, sex, marital status). In this paper, we study how the cost and feasibility of recourse can change across these latent groups. We introduce a notion of group-level plausibility to identify groups of individuals with a shared set of latent characteristics. We develop a general-purpose clustering procedure to identify groups from samples. Further, we propose a constrained optimization approach to learn models that equalize the cost of recourse over latent groups. We evaluate our approach through an empirical study on simulated and real-world datasets, showing that it can produce models that have better performance in terms of overall costs and feasibility at a group level.

----

## [2427] Representation-Based Robustness in Goal-Conditioned Reinforcement Learning

**Authors**: *Xiangyu Yin, Sihao Wu, Jiaxu Liu, Meng Fang, Xingyu Zhao, Xiaowei Huang, Wenjie Ruan*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i19.30176](https://doi.org/10.1609/aaai.v38i19.30176)

**Abstract**:

While Goal-Conditioned Reinforcement Learning (GCRL) has gained attention, its algorithmic robustness against adversarial perturbations remains unexplored. The attacks and robust representation training methods that are designed for traditional RL become less effective when applied to GCRL. To address this challenge, we first propose the Semi-Contrastive Representation attack, a novel approach inspired by the adversarial contrastive attack. Unlike existing attacks in RL, it only necessitates information from the policy function and can be seamlessly implemented during deployment. Then, to mitigate the vulnerability of existing GCRL algorithms, we introduce Adversarial Representation Tactics, which combines Semi-Contrastive Adversarial Augmentation with Sensitivity-Aware Regularizer to improve the adversarial robustness of the underlying RL agent against various types of perturbations. Extensive experiments validate the superior performance of our attack and defence methods across multiple state-of-the-art GCRL algorithms. Our code is available at https://github.com/TrustAI/ReRoGCRL.

----

## [2428] Enhancing Off-Policy Constrained Reinforcement Learning through Adaptive Ensemble C Estimation

**Authors**: *Hengrui Zhang, Youfang Lin, Shuo Shen, Sheng Han, Kai Lv*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i19.30177](https://doi.org/10.1609/aaai.v38i19.30177)

**Abstract**:

In the domain of real-world agents, the application of Reinforcement Learning (RL) remains challenging due to the necessity for safety constraints. Previously, Constrained Reinforcement Learning (CRL) has predominantly focused on on-policy algorithms. Although these algorithms exhibit a degree of efficacy, their interactivity efficiency in real-world settings is sub-optimal, highlighting the demand for more efficient off-policy methods. However, off-policy CRL algorithms grapple with challenges in precise estimation of the C-function, particularly due to the fluctuations in the constrained Lagrange multiplier. Addressing this gap, our study focuses on the nuances of C-value estimation in off-policy CRL and introduces the Adaptive Ensemble C-learning (AEC) approach to reduce these inaccuracies. Building on state-of-the-art off-policy algorithms, we propose AEC-based CRL algorithms designed for enhanced task optimization. Extensive experiments on nine constrained robotics tasks reveal the superior interaction efficiency and performance of our algorithms in comparison to preceding methods.

----

## [2429] Efficient Toxic Content Detection by Bootstrapping and Distilling Large Language Models

**Authors**: *Jiang Zhang, Qiong Wu, Yiming Xu, Cheng Cao, Zheng Du, Konstantinos Psounis*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i19.30178](https://doi.org/10.1609/aaai.v38i19.30178)

**Abstract**:

Toxic content detection is crucial for online services to remove inappropriate content that violates community standards. To automate the detection process, prior works have proposed varieties of machine learning (ML) approaches to train Language Models (LMs) for toxic content detection. However, both their accuracy and transferability across datasets are limited. Recently, Large Language Models (LLMs) have shown promise in toxic content detection due to their superior zero-shot and few-shot in-context learning ability as well as broad transferability on ML tasks.
However, efficiently designing prompts for LLMs remains challenging. Moreover, the high run-time cost of LLMs may hinder their deployments in production. To address these challenges, in this work, we propose BD-LLM, a novel and efficient approach to bootstrapping and distilling LLMs for toxic content detection. 
Specifically, we design a novel prompting method named Decision-Tree-of-Thought (DToT) to bootstrap LLMs' detection performance and extract high-quality rationales. DToT can automatically select more fine-grained context to re-prompt LLMs when their responses lack confidence. Additionally, we use the rationales extracted via DToT to fine-tune student LMs. Our experimental results on various datasets demonstrate that DToT can improve the accuracy of LLMs by up to 4.6%. Furthermore, student LMs fine-tuned with rationales extracted via DToT outperform baselines on all datasets with up to 16.9% accuracy improvement, while being more than 60x smaller than conventional LLMs. Finally, we observe that student LMs fine-tuned with rationales exhibit better cross-dataset transferability.

----

## [2430] LR-XFL: Logical Reasoning-Based Explainable Federated Learning

**Authors**: *Yanci Zhang, Han Yu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i19.30179](https://doi.org/10.1609/aaai.v38i19.30179)

**Abstract**:

Federated learning (FL) is an emerging approach for training machine learning models collaboratively while preserving data privacy. The need for privacy protection makes it difficult for FL models to achieve global transparency and explainability. To address this limitation, we incorporate logic-based explanations into FL by proposing the Logical Reasoning-based eXplainable Federated Learning (LR-XFL) approach. Under LR-XFL, FL clients create local logic rules based on their local data and send them, along with model updates, to the FL server. The FL server connects the local logic rules through a proper logical connector that is derived based on properties of client data, without requiring access to the raw data. In addition, the server also aggregates the local model updates with weight values determined by the quality of the clients’ local data as reflected by their uploaded logic rules. The results show that LR-XFL outperforms the most relevant baseline by 1.19%, 5.81% and 5.41% in terms of classification accuracy, rule accuracy and rule fidelity, respectively. The explicit rule evaluation and expression under LR-XFL enable human experts to validate and correct the rules on the server side, hence improving the global FL model’s robustness to errors. It has the potential to enhance the transparency of FL models for areas like healthcare and finance where both data privacy and explainability are important.

----

## [2431] GaLileo: General Linear Relaxation Framework for Tightening Robustness Certification of Transformers

**Authors**: *Yunruo Zhang, Lujia Shen, Shanqing Guo, Shouling Ji*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i19.30180](https://doi.org/10.1609/aaai.v38i19.30180)

**Abstract**:

Transformers based on attention mechanisms exhibit vulnerability to adversarial examples, posing a substantial threat to the security of their applications. Aiming to solve this problem, the concept of robustness certification is introduced to formally ascertain the presence of any adversarial example within a specified region surrounding a given sample. However, prior works have neglected the dependencies among inputs of softmax (the most complex function in attention mechanisms) during linear relaxations. This oversight has consequently led to imprecise certification results. In this work, we introduce GaLileo, a general linear relaxation framework designed to certify the robustness of Transformers. GaLileo effectively surmounts the trade-off between precision and efficiency in robustness certification through our innovative n-dimensional relaxation approach. Notably, our relaxation technique represents a pioneering effort as the first linear relaxation for n-dimensional functions such as softmax. Our novel approach successfully transcends the challenges posed by the curse of dimensionality inherent in linear relaxations, thereby enhancing linear bounds by incorporating input dependencies. Our evaluations encompassed a thorough analysis utilizing the SST and Yelp datasets along with diverse Transformers of different depths and widths. The experimental results demonstrate that, as compared to the baseline method CROWN-BaF, GaLileo achieves up to 3.24 times larger certified radii while requiring similar running times. Additionally, GaLileo successfully attains certification for Transformers' robustness against multi-word lp perturbations, marking a notable accomplishment in this field.

----

## [2432] A Huber Loss Minimization Approach to Byzantine Robust Federated Learning

**Authors**: *Puning Zhao, Fei Yu, Zhiguo Wan*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i19.30181](https://doi.org/10.1609/aaai.v38i19.30181)

**Abstract**:

Federated learning systems are susceptible to adversarial attacks. To combat this, we introduce a novel aggregator based on Huber loss minimization, and provide a comprehensive theoretical analysis. Under independent and identically distributed (i.i.d) assumption, our approach has several advantages compared to existing methods. Firstly, it has optimal dependence on epsilon, which stands for the ratio of attacked clients. Secondly, our approach does not need precise knowledge of epsilon. Thirdly, it allows different clients to have unequal data sizes. We then broaden our analysis to include non-i.i.d data, such that clients have slightly different distributions.

----

## [2433] Responsible Bandit Learning via Privacy-Protected Mean-Volatility Utility

**Authors**: *Shanshan Zhao, Wenhai Cui, Bei Jiang, Linglong Kong, Xiaodong Yan*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i19.30182](https://doi.org/10.1609/aaai.v38i19.30182)

**Abstract**:

For ensuring the safety of users by protecting the privacy,  the traditional privacy-preserving bandit algorithm aiming to maximize the mean reward has been widely studied in scenarios such as online ride-hailing, advertising recommendations, and personalized healthcare. However, classical bandit learning is irresponsible in such practical applications as they fail to account for risks in online decision-making and ignore external system information. This paper firstly proposes  privacy protected mean-volatility utility as the objective of bandit learning and proves its responsibility, because it aims at achieving the maximum probability of utility by considering the risk.  Theoretically, our proposed responsible bandit learning is expected to achieve the fastest convergence rate among current bandit algorithms  and generates more statistical power than classical normality-based test. Finally, simulation studies provide supporting evidence for the theoretical results and demonstrate stronger performance when using stricter privacy budgets.

----

## [2434] UMA: Facilitating Backdoor Scanning via Unlearning-Based Model Ablation

**Authors**: *Yue Zhao, Congyi Li, Kai Chen*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i19.30183](https://doi.org/10.1609/aaai.v38i19.30183)

**Abstract**:

Recent advances in backdoor attacks, like leveraging complex triggers or stealthy implanting techniques, have   introduced new challenges in backdoor scanning, limiting the usability of Deep Neural Networks (DNNs) in various scenarios. In this paper, we propose Unlearning-based Model Ablation (UMA), a novel approach to facilitate backdoor scanning and defend against advanced backdoor attacks. UMA filters out backdoor-irrelevant features by ablating the inherent features of the target class within the model and subsequently reveals the backdoor through dynamic trigger optimization. We evaluate our method on 1700 models (700 benign and 1000 trojaned) with 6 model structures, 7 different backdoor attacks and 4 datasets. Our results demonstrate that the proposed methodology effectively detect these advanced backdoors. Specifically, our method can achieve 91% AUC-ROC and 86.6% detection accuracy on average, which outperforms the baselines, including Neural Cleanse, ABS, K-Arm and MNTD.

----

## [2435] AdvST: Revisiting Data Augmentations for Single Domain Generalization

**Authors**: *Guangtao Zheng, Mengdi Huai, Aidong Zhang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i19.30184](https://doi.org/10.1609/aaai.v38i19.30184)

**Abstract**:

Single domain generalization (SDG) aims to train a robust model against unknown target domain shifts using data from a single source domain. Data augmentation has been proven an effective approach to SDG. However, the utility of standard augmentations, such as translate, or invert, has not been fully exploited in SDG; practically, these augmentations are used as a part of a data preprocessing procedure. Although it is intuitive to use many such augmentations to boost the robustness of a model to out-of-distribution domain shifts, we lack a principled approach to harvest the benefit brought from multiple these augmentations. Here,  we conceptualize standard data augmentations with learnable parameters as semantics transformations that can manipulate certain semantics of a sample, such as the geometry or color of an image. Then, we propose Adversarial learning with Semantics Transformations (AdvST) that augments the source domain data with semantics transformations and  learns a robust model with the augmented data. We theoretically show that AdvST essentially optimizes a distributionally robust optimization objective defined on a set of semantics distributions induced by the parameters of semantics transformations. We demonstrate that AdvST can produce samples that expand the coverage on target domain data. Compared with the state-of-the-art methods, AdvST, despite being a simple method, is surprisingly competitive and achieves the best average SDG performance on the Digits, PACS, and DomainNet datasets. Our code is available at https://github.com/gtzheng/AdvST.

----

## [2436] Can LLM Replace Stack Overflow? A Study on Robustness and Reliability of Large Language Model Code Generation

**Authors**: *Li Zhong, Zilong Wang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i19.30185](https://doi.org/10.1609/aaai.v38i19.30185)

**Abstract**:

Recently, large language models (LLMs) have shown an extraordinary ability to understand natural language and generate programming code. It has been a common practice for software engineers to consult LLMs when encountering coding questions. Although efforts have been made to avoid syntax errors and align the code with the intended semantics, the reliability, and robustness of the code generation from LLMs have not yet been thoroughly studied. The executable code is not equivalent to reliable and robust code, especially in the context of real-world software development. For example, the misuse of APIs in the generated code could lead to severe problems, such as resource leaks, program crashes, etc. Existing code evaluation benchmarks and datasets focus on crafting small tasks such as programming questions in coding interviews, which, however, deviates from the problem that developers would ask LLM for real-world coding help. To fill the missing piece, in this work, we propose a dataset RobustAPI for evaluating the reliability and robustness of code generated by LLMs. We collect 1208 coding questions from Stack Overflow on 18 representative Java APIs. We summarize the common misuse patterns of these APIs and evaluate them on current popular LLMs. The evaluation results show that even for GPT-4, 62% of the generated code contains API misuses, which would cause unexpected consequences if the code is introduced into real-world software.

----

## [2437] DataElixir: Purifying Poisoned Dataset to Mitigate Backdoor Attacks via Diffusion Models

**Authors**: *Jiachen Zhou, Peizhuo Lv, Yibing Lan, Guozhu Meng, Kai Chen, Hualong Ma*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i19.30186](https://doi.org/10.1609/aaai.v38i19.30186)

**Abstract**:

Dataset sanitization is a widely adopted proactive defense against poisoning-based backdoor attacks, aimed at filtering out and removing poisoned samples from training datasets. However, existing methods have shown limited efficacy in countering the ever-evolving trigger functions, and often leading to considerable degradation of benign accuracy. In this paper, we propose DataElixir, a novel sanitization approach tailored to purify poisoned datasets. We leverage diffusion models to eliminate trigger features and restore benign features, thereby turning the poisoned samples into benign ones. Specifically, with multiple iterations of the forward and reverse process, we extract intermediary images and their predicted labels for each sample in the original dataset. Then, we identify anomalous samples in terms of the presence of label transition of the intermediary images, detect the target label by quantifying distribution discrepancy, select their purified images considering pixel and feature distance, and determine their ground-truth labels by training a benign model. Experiments conducted on 9 popular attacks demonstrates that DataElixir effectively mitigates various complex attacks while exerting minimal impact on benign accuracy, surpassing the performance of baseline defense methods.

----

## [2438] Closing the Gap: Achieving Better Accuracy-Robustness Tradeoffs against Query-Based Attacks

**Authors**: *Pascal Zimmer, Sébastien Andreina, Giorgia Azzurra Marson, Ghassan Karame*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i19.30187](https://doi.org/10.1609/aaai.v38i19.30187)

**Abstract**:

Although promising, existing defenses against query-based attacks share a common limitation: they offer increased robustness against attacks at the price of a considerable accuracy drop on clean samples. In this work, we show how to efficiently establish, at test-time, a solid tradeoff between robustness and accuracy when mitigating query-based attacks. Given that these attacks necessarily explore low-confidence regions, our insight is that activating dedicated defenses, such as random noise defense and random image transformations, only for low-confidence inputs is sufficient to prevent them. Our approach is independent of training and supported by theory. We verify the effectiveness of our approach for various existing defenses by conducting extensive experiments on CIFAR-10, CIFAR-100, and ImageNet. Our results confirm that our proposal can indeed enhance these defenses by providing better tradeoffs between robustness and accuracy when compared to state-of-the-art approaches while being completely training-free.

----

## [2439] Coevolutionary Algorithm for Building Robust Decision Trees under Minimax Regret

**Authors**: *Adam Zychowski, Andrew Perrault, Jacek Mandziuk*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i19.30188](https://doi.org/10.1609/aaai.v38i19.30188)

**Abstract**:

In recent years, there has been growing interest in developing robust machine learning (ML) models that can withstand adversarial attacks, including one of the most widely adopted, efficient, and interpretable ML algorithms—decision trees (DTs). This paper proposes a novel coevolutionary algorithm (CoEvoRDT) designed to create robust DTs capable of handling noisy high-dimensional data in adversarial contexts. Motivated by the limitations of traditional DT algorithms, we leverage adaptive coevolution to allow DTs to evolve and learn from interactions with perturbed input data. CoEvoRDT alternately evolves competing populations of DTs and perturbed features, enabling construction of DTs with desired properties. CoEvoRDT is easily adaptable to various target metrics, allowing the use of tailored robustness criteria such as minimax regret. Furthermore, CoEvoRDT has potential to improve the results of other state-of-the-art methods by incorporating their outcomes (DTs they produce) into the initial population and optimize them in the process of coevolution. Inspired by the game theory, CoEvoRDT utilizes mixed Nash equilibrium to enhance convergence. The method is tested on 20 popular datasets and shows superior performance compared to 4 state-of-the-art algorithms. It outperformed all competing methods on 13 datasets with adversarial accuracy metrics, and on all 20 considered datasets with minimax regret. Strong experimental results and flexibility in choosing the error measure make CoEvoRDT a promising approach for constructing robust DTs in real-world applications.

----

## [2440] BirdCollect: A Comprehensive Benchmark for Analyzing Dense Bird Flock Attributes

**Authors**: *Kshitiz, Sonu Shreshtha, Bikash Dutta, Muskan Dosi, Mayank Vatsa, Richa Singh, Saket Anand, Sudeep Sarkar, Sevaram Mali Parihar*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i20.30189](https://doi.org/10.1609/aaai.v38i20.30189)

**Abstract**:

Automatic recognition of bird behavior from long-term, un controlled outdoor imagery can contribute to conservation efforts by enabling large-scale monitoring of bird populations. Current techniques in AI-based wildlife monitoring have focused on short-term tracking and monitoring birds individually rather than in species-rich flocks. We present Bird-Collect, a comprehensive benchmark dataset for monitoring dense bird flock attributes. It includes a unique collection of more than 6,000 high-resolution images of Demoiselle Cranes (Anthropoides virgo) feeding and nesting in the vicinity of Khichan region of Rajasthan. Particularly, each image contains an average of 190 individual birds, illustrating the complex dynamics of densely populated bird flocks on a scale that has not previously been studied. In addition, a total of 433 distinct pictures captured at Keoladeo National Park, Bharatpur provide a comprehensive representation of 34 distinct bird species belonging to various taxonomic groups. These images offer details into the diversity and the behaviour of birds in vital natural ecosystem along the migratory flyways. Additionally, we provide a set of 2,500 point-annotated samples which serve as ground truth for benchmarking various computer vision tasks like crowd counting, density estimation, segmentation, and species classification. The benchmark performance for these tasks highlight the need for tailored approaches for specific wildlife applications, which include varied conditions including views, illumination, and resolutions. With around 46.2 GBs in size encompassing data collected from two distinct nesting ground sets, it is the largest birds dataset containing detailed annotations, showcasing a substantial leap in bird research possibilities. We intend to publicly release the dataset to the research community. The database is available at: https://iab-rubric.org/resources/wildlife-dataset/birdcollect

----

## [2441] A Bayesian Spatial Model to Correct Under-Reporting in Urban Crowdsourcing

**Authors**: *Gabriel Agostini, Emma Pierson, Nikhil Garg*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i20.30190](https://doi.org/10.1609/aaai.v38i20.30190)

**Abstract**:

Decision-makers often observe the occurrence of events through a reporting process. City governments, for example, rely on resident reports to find and then resolve urban infrastructural problems such as fallen street trees, flooded basements, or rat infestations. Without additional assumptions, there is no way to distinguish events that occur but are not reported from events that truly did not occur--a fundamental problem in settings with positive-unlabeled data. Because disparities in reporting rates correlate with resident demographics, addressing incidents only on the basis of reports leads to systematic neglect in neighborhoods that are less likely to report events. We show how to overcome this challenge by leveraging the fact that events are spatially correlated. Our framework uses a Bayesian spatial latent variable model to infer event occurrence probabilities and applies it to storm-induced flooding reports in New York City, further pooling results across multiple storms. We show that a model accounting for under-reporting and spatial correlation predicts future reports more accurately than other models, and further induces a more equitable set of inspections: its allocations better reflect the population and provide equitable service to non-white, less traditionally educated, and lower-income residents. This finding reflects heterogeneous reporting behavior learned by the model: reporting rates are higher in Census tracts with higher populations, proportions of white residents, and proportions of owner-occupied households. Our work lays the groundwork for more equitable proactive government services, even with disparate reporting behavior.

----

## [2442] Automatic Interpretation of Line Probe Assay Test for Tuberculosis

**Authors**: *Jatin Agrawal, Mukul Kumar, Avtansh Tiwari, Sachin Danisetty, Soma Dhavala, Nakul Jain, Prasaanth Balraj, Niket Singh, Siddhant Shingi, Jayakrishna Kurada, Raghuram Rao, S. Anand, Nishant Kumar*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i20.30191](https://doi.org/10.1609/aaai.v38i20.30191)

**Abstract**:

Line Probe Assay (LPA) is a widely used method for diagnosing drug-resistant tuberculosis (DRTB), but it is a time-consuming and labor-intensive process that requires expert interpretation. DRTB is a significant threat to global TB control efforts and its prompt diagnosis is critical for initiating appropriate treatment. In this paper, we present an automated LPA test interpretation solution that uses computer vision techniques to extract and analyze strips from LPA sheets and uses machine learning algorithms to produce drug sensitivity and resistivity outcomes with extremely high precision and recall. We also develop OCR models to eliminate manual data entry to further reduce the overall time. Our solution comprises a rejection module that flags ambiguous and novel samples that are then referred to experienced lab technicians. This results in increased trust in the solution. To evaluate our solution, we curate an extensive and diverse dataset of LPA strips annotated by multiple microbiologists across India. Our solution achieves more than 95% accuracy for all drugs on this dataset. The proposed solution has the potential to increase the efficiency, standardization of LPA test interpretation, and fast-tracking the dissemination of results to end-users via a designated Management Information System (MIS).

----

## [2443] Physics-Informed Graph Neural Networks for Water Distribution Systems

**Authors**: *Inaam Ashraf, Janine Strotherm, Luca Hermes, Barbara Hammer*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i20.30192](https://doi.org/10.1609/aaai.v38i20.30192)

**Abstract**:

Water distribution systems (WDS) are an integral part of critical infrastructure which is pivotal to urban development. As 70% of the world's population will likely live in urban environments in 2050, efficient simulation and planning tools for WDS play a crucial role in reaching UN's sustainable developmental goal (SDG) 6 - "Clean water and sanitation for all". In this realm, we propose a novel and efficient machine learning emulator, more precisely, a physics-informed deep learning (DL) model, for hydraulic state estimation in WDS. Using a recursive approach, our model only needs a few graph convolutional neural network (GCN) layers and employs an innovative algorithm based on message passing. Unlike conventional machine learning tasks, the model uses hydraulic principles to infer two additional hydraulic state features in the process of reconstructing the available ground truth feature in an unsupervised manner. To the best of our knowledge, this is the first DL approach to emulate the popular hydraulic simulator EPANET, utilizing no additional information. Like most DL models and unlike the hydraulic simulator, our model demonstrates vastly faster emulation times that do not increase drastically with the size of the WDS. Moreover, we achieve high accuracy on the ground truth and very similar results compared to the hydraulic simulator as demonstrated through experiments on five real-world WDS datasets.

----

## [2444] Quantile-Regression-Ensemble: A Deep Learning Algorithm for Downscaling Extreme Precipitation

**Authors**: *Thomas Bailie, Yun Sing Koh, Neelesh Rampal, Peter B. Gibson*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i20.30193](https://doi.org/10.1609/aaai.v38i20.30193)

**Abstract**:

Global Climate Models (GCMs) simulate low resolution climate projections on a global scale. The native resolution of GCMs is generally too low for societal-level decision-making. To enhance the spatial resolution, downscaling is often applied to GCM output. Statistical downscaling techniques, in particular, are well-established as a cost-effective approach. They require significantly less computational time than physics-based dynamical downscaling. In recent years, deep learning has gained prominence in statistical downscaling, demonstrating significantly lower error rates compared to traditional statistical methods. However, a drawback of regression-based deep learning techniques is their tendency to overfit to the mean sample intensity. Extreme values as a result are often underestimated. Problematically, extreme events have the largest societal impact. We propose Quantile-Regression-Ensemble (QRE), an innovative deep learning algorithm inspired by boosting methods. Its primary objective is to avoid trade-offs between fitting to sample means and extreme values by training independent models on a partitioned dataset. Our QRE is robust to redundant models and not susceptible to explosive ensemble weights, ensuring a reliable training process. QRE achieves lower Mean Squared Error (MSE) compared to various baseline models. In particular, our algorithm has a lower  error for high-intensity precipitation events over New Zealand, highlighting the ability to represent extreme events accurately.

----

## [2445] Early Detection of Extreme Storm Tide Events Using Multimodal Data Processing

**Authors**: *Marcel R. de Barros, Andressa Pinto, Andres Monroy, Felipe M. Moreno, Jefferson F. Coelho, Aldomar Pietro Silva, Caio Fabricio Deberaldini Netto, José Roberto Leite, Marlon S. Mathias, Eduardo Aoun Tannuri, Artur Jordão, Edson S. Gomi, Fábio Gagliardi Cozman, Marcelo Dottori, Anna Helena Reali Costa*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i20.30194](https://doi.org/10.1609/aaai.v38i20.30194)

**Abstract**:

Sea-level rise is a well-known consequence of climate change. Several studies have estimated the social and economic impact of the increase in extreme flooding. An efficient way to mitigate its consequences is the development of a flood alert and prediction system, based on high-resolution numerical models and robust sensing networks. However, current models use various simplifying assumptions that compromise accuracy to ensure solvability within a reasonable timeframe, hindering more regular and cost-effective forecasts for various locations along the shoreline. To address these issues, this work proposes a hybrid model for multimodal data processing that combines physics-based numerical simulations, data obtained from a network of sensors, and satellite images to provide refined wave and sea-surface height forecasts, with real results obtained in a critical location within the Port of Santos (the largest port in Latin America). Our approach exhibits faster convergence than data-driven models while achieving more accurate predictions. Moreover, the model handles irregularly sampled time series and missing data without the need for complex preprocessing mechanisms or data imputation while keeping low computational costs through a combination of time encoding, recurrent and graph neural networks. Enabling raw sensor data to be easily combined with existing physics-based models opens up new possibilities for accurate extreme storm tide events forecast systems that enhance community safety and aid policymakers in their decision-making processes.

----

## [2446] Decision-Making for Land Conservation: A Derivative-Free Optimization Framework with Nonlinear Inputs

**Authors**: *Cassidy K. Buhler, Hande Y. Benson*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i20.30195](https://doi.org/10.1609/aaai.v38i20.30195)

**Abstract**:

Protected areas (PAs) are designated spaces where human activities are restricted to preserve critical habitats. Decision-makers are challenged with balancing a trade-off of financial feasibility with ecological benefit when establishing PAs. Given the long-term ramifications of these decisions and the constantly shifting environment, it is crucial that PAs are carefully selected with long-term viability in mind. 

Using AI tools like simulation and optimization is common for designating PAs, but current decision models are primarily linear. In this paper, we propose a derivative-free optimization framework paired with a nonlinear component, population viability analysis (PVA). Formulated as a mixed integer nonlinear programming (MINLP) problem, our model allows for linear and nonlinear inputs. Connectivity, competition, crowding, and other similar concerns are handled by the PVA software, rather than expressed as constraints of the optimization model. In addition, we present numerical results that serve as a proof of concept, showing our models yield PAs with similar expected risk to that of preserving every parcel in a habitat, but at a significantly lower cost. 

The overall goal is to promote interdisciplinary work by providing a new mathematical programming tool for conservationists that allows for nonlinear inputs and can be paired with existing ecological software. The code and data are available at
https://github.com/cassiebuhler/conservation-dfo.

----

## [2447] CariesXrays: Enhancing Caries Detection in Hospital-Scale Panoramic Dental X-rays via Feature Pyramid Contrastive Learning

**Authors**: *Bingzhi Chen, Sisi Fu, Yishu Liu, Jiahui Pan, Guangming Lu, Zheng Zhang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i20.30196](https://doi.org/10.1609/aaai.v38i20.30196)

**Abstract**:

Dental caries has been widely recognized as one of the most prevalent chronic diseases in the field of public health. Despite advancements in automated diagnosis across various medical domains, it remains a substantial challenge for dental caries detection due to its inherent variability and intricacies. To bridge this gap, we release a hospital-scale panoramic dental X-ray benchmark, namely “CariesXrays”, to facilitate the advancements in high-precision computer-aided diagnosis for dental caries. It comprises 6,000 panoramic dental X-ray images, with a total of 13,783 instances of dental caries, all meticulously annotated by dental professionals. In this paper, we propose a novel Feature Pyramid Contrastive Learning (FPCL) framework, that jointly incorporates feature pyramid learning and contrastive learning within a unified diagnostic paradigm for automated dental caries detection. Specifically, a robust dual-directional feature pyramid network (D2D-FPN) is designed to adaptively capture rich and informative contextual information from multi-level feature maps, thus enhancing the generalization ability of caries detection across different scales. Furthermore, our model is augmented with an effective proposals-prototype contrastive regularization learning (P2P-CRL) mechanism, which can flexibly bridge the semantic gaps among diverse dental caries with varying appearances, resulting in high-quality dental caries proposals. Extensive experiments on our newly-established CariesXrays benchmark demonstrate the potential of FPCL to make a significant social impact on caries diagnosis.

----

## [2448] Referee-Meta-Learning for Fast Adaptation of Locational Fairness

**Authors**: *Weiye Chen, Yiqun Xie, Xiaowei Jia, Erhu He, Han Bao, Bang An, Xun Zhou*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i20.30197](https://doi.org/10.1609/aaai.v38i20.30197)

**Abstract**:

When dealing with data from distinct locations, machine learning algorithms tend to demonstrate an implicit preference of some locations over the others, which constitutes biases that sabotage the spatial fairness of the algorithm. This unfairness can easily introduce biases in subsequent decision-making given broad adoptions of learning-based solutions in practice. However, locational biases in AI are largely understudied. To mitigate biases over locations, we propose a locational meta-referee (Meta-Ref) to oversee the few-shot meta-training and meta-testing of a deep neural network. Meta-Ref dynamically adjusts the learning rates for training samples of given locations to advocate a fair performance across locations, through an explicit consideration of locational biases and the characteristics of input data. We present a three-phase training framework to learn both a meta-learning-based predictor and an integrated Meta-Ref that governs the fairness of the model. Once trained with a distribution of spatial tasks, Meta-Ref is applied to samples from new spatial tasks (i.e., regions outside the training area) to promote fairness during the fine-tune step. We carried out experiments with two case studies on crop monitoring and transportation safety, which show Meta-Ref can improve locational fairness while keeping the overall prediction quality at a similar level.

----

## [2449] From Artificially Real to Real: Leveraging Pseudo Data from Large Language Models for Low-Resource Molecule Discovery

**Authors**: *Yuhan Chen, Nuwa Xi, Yanrui Du, Haochun Wang, Jianyu Chen, Sendong Zhao, Bing Qin*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i20.30198](https://doi.org/10.1609/aaai.v38i20.30198)

**Abstract**:

Molecule discovery serves as a cornerstone in numerous scientific domains, fueling the development of new materials and innovative drug designs. Recent developments of in-silico molecule discovery have highlighted the promising results of cross-modal techniques, which bridge molecular structures with their descriptive annotations. However, these cross-modal methods frequently encounter the issue of data scarcity, hampering their performance and application. In this paper, we address the low-resource challenge by utilizing artificially-real data generated by Large Language Models (LLMs). We first introduce a retrieval-based prompting strategy to construct high-quality pseudo data, then explore the optimal method to effectively leverage this pseudo data. Experiments show that using pseudo data for domain adaptation outperforms all existing methods, while also requiring a smaller model scale, reduced data size and lower training cost, highlighting its efficiency. Furthermore, our method shows a sustained improvement as the volume of pseudo data increases, revealing the great potential of pseudo data in advancing low-resource cross-modal molecule discovery.

----

## [2450] Auto311: A Confidence-Guided Automated System for Non-emergency Calls

**Authors**: *Zirong Chen, Xutong Sun, Yuanhe Li, Meiyi Ma*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i20.30199](https://doi.org/10.1609/aaai.v38i20.30199)

**Abstract**:

Emergency and non-emergency response systems are essential services provided by local governments and critical to protecting lives, the environment, and property. The effective handling of (non-)emergency calls is critical for public safety and well-being. By reducing the burden through non-emergency callers, residents in critical need of assistance through 911 will receive a fast and effective response. Collaborating with the Department of Emergency Communications (DEC) in Nashville, we analyzed 11,796 non-emergency call recordings and developed Auto311, the first automated system to handle 311 non-emergency calls, which (1) effectively and dynamically predicts ongoing non-emergency incident types to generate tailored case reports during the call; (2) itemizes essential information from dialogue contexts to complete the generated reports; and (3) strategically structures system-caller dialogues with optimized confidence. We used real-world data to evaluate the system's effectiveness and deployability. The experimental results indicate that the system effectively predicts incident type with an average F-1 score of 92.54%. Moreover, the system successfully itemizes critical information from relevant contexts to complete reports, evincing a 0.93 average consistency score compared to the ground truth. Additionally, emulations demonstrate that the system effectively decreases conversation turns as the utterance size gets more extensive and categorizes the ongoing call with 94.49% mean accuracy.

----

## [2451] Blind-Touch: Homomorphic Encryption-Based Distributed Neural Network Inference for Privacy-Preserving Fingerprint Authentication

**Authors**: *Hyunmin Choi, Simon S. Woo, Hyoungshick Kim*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i20.30200](https://doi.org/10.1609/aaai.v38i20.30200)

**Abstract**:

Fingerprint authentication is a popular security mechanism for smartphones and laptops. However, its adoption in web and cloud environments has been limited due to privacy concerns over storing and processing biometric data on servers. This paper introduces Blind-Touch, a novel machine learning-based fingerprint authentication system leveraging homomorphic encryption to address these privacy concerns. Homomorphic encryption allows computations on encrypted data without decrypting. Thus, Blind-Touch can keep fingerprint data encrypted on the server while performing machine learning operations. Blind-Touch combines three strategies to efficiently utilize homomorphic encryption in machine learning: (1) It optimizes the feature vector for a distributed architecture, processing the first fully connected layer (FC-16) in plaintext on the client side and the subsequent layer (FC-1) post-encryption on the server, thereby minimizing encrypted computations; (2) It employs a homomorphic encryption-compatible data compression technique capable of handling 8,192 authentication results concurrently; and (3) It utilizes a clustered server architecture to simultaneously process authentication results, thereby enhancing scalability with increasing user numbers. Blind-Touch achieves high accuracy on two benchmark fingerprint datasets, with a 93.6% F1- score for the PolyU dataset and a 98.2% F1-score for the SOKOTO dataset. Moreover, Blind-Touch can match a fingerprint among 5,000 in about 0.65 seconds. With its privacy-focused design, high accuracy, and efficiency, Blind-Touch is a promising alternative to conventional fingerprint authentication for web and cloud applications.

----

## [2452] Identifying Guarantors of War Veterans Using Robust-SEAL: A Case of the Korean War

**Authors**: *Jong In Choi, Won Kyung Lee, Jae Hwan Lee, So Young Sohn*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i20.30201](https://doi.org/10.1609/aaai.v38i20.30201)

**Abstract**:

Most countries provide veterans with various benefits to reward their sacrifice. Unfortunately, many veterans have failed to prove their status due to loss of military records. Thus, some governments allow the verification of those veterans through "buddy statements" obtained from the people who can vouch for the buddy's participation in the war. However, it is still challenging for veterans to find guarantors directly. With this background, we suggest to utilizing historical war records of combined operations to increase the pool of potential guarantors for the buddy statements. However, a combined operation network among troops can have missing edges and perturbations on attributes of the troop due to inaccurate information. In this study, we learn from some recorded interactions which might be incomplete and noisy, and predict missing linkages among the troops that might have interacted together in the war, by proposing Robust-SEAL (learning from Subgraphs, Embeddings, and Attributes for Link prediction). It combines two Graph Neural Network (GNN) architectures: robust Graph Convolutional Network which considers the uncertainty of node attributes with a probabilistic approach, and SEAL which improves the expressive power of the GNN with a labeling trick. Our proposed approach was applied to Korean War data with perturbations. For experimentations, we hid some actual interactions and found that Robust-SEAL restores missing interactions better than other GNN-based baselines.

----

## [2453] Fair Sampling in Diffusion Models through Switching Mechanism

**Authors**: *Yujin Choi, Jinseong Park, Hoki Kim, Jaewook Lee, Saerom Park*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i20.30202](https://doi.org/10.1609/aaai.v38i20.30202)

**Abstract**:

Diffusion models have shown their effectiveness in generation tasks by well-approximating the underlying probability distribution. However, diffusion models are known to suffer from an amplified inherent bias from the training data in terms of fairness. While the sampling process of diffusion models can be controlled by conditional guidance, previous works have attempted to find empirical guidance to achieve quantitative fairness. 
To address this limitation, we propose a fairness-aware sampling method called \textit{attribute switching} mechanism for diffusion models. Without additional training, the proposed sampling can obfuscate sensitive attributes in generated data without relying on classifiers.
We mathematically prove and experimentally demonstrate the effectiveness of the proposed method on two key aspects: (i) the generation of fair data and (ii) the preservation of the utility of the generated data.

----

## [2454] Arbitrariness and Social Prediction: The Confounding Role of Variance in Fair Classification

**Authors**: *A. Feder Cooper, Katherine Lee, Madiha Zahrah Choksi, Solon Barocas, Christopher De Sa, James Grimmelmann, Jon M. Kleinberg, Siddhartha Sen, Baobao Zhang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i20.30203](https://doi.org/10.1609/aaai.v38i20.30203)

**Abstract**:

Variance in predictions across different trained models is a significant, under-explored source of error in fair binary classification. In practice, the variance on some data examples is so large that decisions can be effectively arbitrary. To investigate this problem, we take an experimental approach and make four overarching contributions. We: 1) Define a metric called self-consistency, derived from variance, which we use as a proxy for measuring and reducing arbitrariness; 2) Develop an ensembling algorithm that abstains from classification when a prediction would be arbitrary; 3) Conduct the largest to-date empirical study of the role of variance (vis-a-vis self-consistency and arbitrariness) in fair binary classification; and, 4) Release a toolkit that makes the US Home Mortgage Disclosure Act (HMDA) datasets easily usable for future research. Altogether, our experiments reveal shocking insights about the reliability of conclusions on benchmark datasets. Most fair binary classification benchmarks are close-to-fair when taking into account the amount of arbitrariness present in predictions -- before we even try to apply any fairness interventions. This finding calls into question the practical utility of common algorithmic fairness methods, and in turn suggests that we should reconsider how we choose to measure fairness in binary classification.

----

## [2455] Finding ε and δ of Traditional Disclosure Control Systems

**Authors**: *Saswat Das, Keyu Zhu, Christine Task, Pascal Van Hentenryck, Ferdinando Fioretto*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i20.30204](https://doi.org/10.1609/aaai.v38i20.30204)

**Abstract**:

This paper analyzes the privacy of traditional Statistical Disclosure Control (SDC) systems under a differential privacy interpretation. SDCs, such as cell suppression and swapping, promise to safeguard the confidentiality of data and are routinely adopted in data analyses with profound societal and economic impacts. Through a formal analysis and empirical evaluation of demographic data from real households in the U.S., the paper shows that widely adopted SDC systems not only induce vastly larger privacy losses than classical differential privacy mechanisms, but, they may also come at a cost of larger accuracy and fairness.

----

## [2456] MedAlign: A Clinician-Generated Dataset for Instruction Following with Electronic Medical Records

**Authors**: *Scott L. Fleming, Alejandro Lozano, William J. Haberkorn, Jenelle A. Jindal, Eduardo Pontes Reis, Rahul Thapa, Louis Blankemeier, Julian Z. Genkins, Ethan Steinberg, Ashwin Nayak, Birju S. Patel, Chia-Chun Chiang, Alison Callahan, Zepeng Huo, Sergios Gatidis, Scott J. Adams, Oluseyi Fayanju, Shreya J. Shah, Thomas Savage, Ethan Goh, Akshay S. Chaudhari, Nima Aghaeepour, Christopher D. Sharp, Michael A. Pfeffer, Percy Liang, Jonathan H. Chen, Keith E. Morse, Emma P. Brunskill, Jason A. Fries, Nigam H. Shah*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i20.30205](https://doi.org/10.1609/aaai.v38i20.30205)

**Abstract**:

The ability of large language models (LLMs) to follow natural language instructions with human-level fluency suggests many opportunities in healthcare to reduce administrative burden and improve quality of care. However, evaluating LLMs on realistic text generation tasks for healthcare remains challenging. Existing question answering datasets for electronic health record (EHR) data fail to capture the complexity of information needs and documentation burdens experienced by clinicians. To address these challenges, we introduce MedAlign, a benchmark dataset of 983 natural language instructions for EHR data. MedAlign is curated by 15 clinicians (7 specialities), includes clinician-written reference responses for 303 instructions, and provides 276 longitudinal EHRs for grounding instruction-response pairs. We used MedAlign to evaluate 6 general domain LLMs, having clinicians rank the accuracy and quality of each LLM response. We found high error rates, ranging from 35% (GPT-4) to 68% (MPT-7B-Instruct), and 8.3% drop in accuracy moving from 32k to 2k context lengths for GPT-4. Finally, we report correlations between clinician rankings and automated natural language generation metrics as a way to rank LLMs without human review. We make MedAlign available under a research data use agreement to enable LLM evaluations on tasks aligned with clinician needs and preferences.

----

## [2457] CLIPSyntel: CLIP and LLM Synergy for Multimodal Question Summarization in Healthcare

**Authors**: *Akash Ghosh, Arkadeep Acharya, Raghav Jain, Sriparna Saha, Aman Chadha, Setu Sinha*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i20.30206](https://doi.org/10.1609/aaai.v38i20.30206)

**Abstract**:

In the era of modern healthcare, swiftly generating medical question summaries is crucial for informed and timely patient care. Despite the increasing complexity and volume of medical data, existing studies have focused solely on text-based summarization, neglecting the integration of visual information. Recognizing the untapped potential of combining textual queries with visual representations of medical conditions, we introduce the Multimodal Medical Question Summarization (MMQS) Dataset. This dataset, a major contribution of our work, pairs medical queries with visual aids, facilitating a richer and more nuanced understanding of patient needs. We also propose a framework, utilizing the power of Contrastive Language Image Pretraining(CLIP) and Large Language Models(LLMs), consisting of four modules that identify medical disorders, generate relevant context, filter medical concepts, and craft visually aware summaries. Our comprehensive framework harnesses the power of CLIP, a multimodal foundation model, and various general-purpose LLMs, comprising four main modules: the medical disorder identification module, the relevant context generation module, the context filtration module for distilling relevant medical concepts and knowledge, and finally, a general-purpose LLM to generate visually aware medical question summaries. Leveraging our MMQS dataset, we showcase how visual cues from images enhance the generation of medically nuanced summaries. This multimodal approach not only enhances the decision-making process in healthcare but also fosters a more nuanced understanding of patient queries, laying the groundwork for future research in personalized and responsive medical care.
Disclaimer: The article features graphic medical imagery, a result of the subject's inherent requirements.

----

## [2458] Benchmarking Cyber Harassment Dialogue Comprehension through Emotion-Informed Manifestations-Determinants Demarcation

**Authors**: *Soumitra Ghosh, Gopendra Vikram Singh, Jashn Arora, Asif Ekbal*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i20.30207](https://doi.org/10.1609/aaai.v38i20.30207)

**Abstract**:

In the digital age, cybercrimes, particularly cyber harassment, have become pressing issues, targeting vulnerable individuals like children, teenagers, and women. Understanding the experiences and needs of the victims is crucial for effective support and intervention. Online conversations between victims and virtual harassment counselors (chatbots) offer valuable insights into cyber harassment manifestations (CHMs) and determinants (CHDs). However, the distinction between CHMs and CHDs remains unclear. This research is the first to introduce concrete definitions for CHMs and CHDs, investigating their distinction through automated methods to enable efficient cyber-harassment dialogue comprehension. We present a novel dataset, Cyber-MaD that contains Cyber harassment dialogues manually annotated with Manifestations and Determinants. Additionally, we design an Emotion-informed Contextual Dual attention Convolution Transformer (E-ConDuCT) framework to extract CHMs and CHDs from cyber harassment dialogues. The framework primarily: a) utilizes inherent emotion features through adjective-noun pairs modeled by an autoencoder, b) employs a unique Contextual Dual attention Convolution Transformer to learn contextual insights; and c) incorporates a demarcation module leveraging task-specific emotional knowledge and a discriminator loss function to differentiate manifestations and determinants. E-ConDuCT outperforms the state-of-the-art systems on the Cyber-MaD corpus, showcasing its potential in the extraction of CHMs and CHDs. Furthermore, its robustness is demonstrated on the emotion cause extraction task using the CARES_CEASE-v2.0 dataset of suicide notes, confirming its efficacy across diverse cause extraction objectives. Access the code and data at 1. https://www.iitp.ac.in/~ai-nlp-ml/resources.html#E-ConDuCT-on-Cyber-MaD, 2. https://github.com/Soumitra816/Manifestations-Determinants.

----

## [2459] Grey-Box Bayesian Optimization for Sensor Placement in Assisted Living Environments

**Authors**: *Shadan Golestan, Omid Ardakanian, Pierre Boulanger*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i20.30208](https://doi.org/10.1609/aaai.v38i20.30208)

**Abstract**:

Optimizing the configuration and placement of sensors is crucial for reliable fall detection, indoor localization, and activity recognition in assisted living spaces. We propose a novel, sample-efficient approach to find a high-quality sensor placement in an arbitrary indoor space based on grey-box Bayesian optimization and simulation-based evaluation. Our key technical contribution lies in capturing domain-specific knowledge about the spatial distribution of activities and incorporating it into the iterative selection of query points in Bayesian optimization. Considering two simulated indoor environments and a real-world dataset containing human activities and sensor triggers, we show that our proposed method performs better compared to state-of-the-art black-box optimization techniques in identifying high-quality sensor placements, leading to an accurate activity recognition model in terms of F1-score, while also requiring a significantly lower (51.3% on average) number of expensive function queries.

----

## [2460] Federated Learning via Input-Output Collaborative Distillation

**Authors**: *Xuan Gong, Shanglin Li, Yuxiang Bao, Barry Yao, Yawen Huang, Ziyan Wu, Baochang Zhang, Yefeng Zheng, David S. Doermann*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i20.30209](https://doi.org/10.1609/aaai.v38i20.30209)

**Abstract**:

Federated learning (FL) is a machine learning paradigm in which distributed local nodes collaboratively train a central model without sharing individually held private data. Existing FL methods either iteratively share local model parameters or deploy co-distillation. However, the former is highly susceptible to private data leakage, and the latter design relies on the prerequisites of task-relevant real data. Instead, we propose a data-free FL framework based on local-to-central collaborative distillation with direct input and output space exploitation. Our design eliminates any requirement of recursive local parameter exchange or auxiliary task-relevant data to transfer knowledge, thereby giving direct privacy control to local users. In particular, to cope with the inherent data heterogeneity across locals, our technique learns to distill input on which each local model produces consensual yet unique results to represent each expertise. Our proposed FL framework achieves notable privacy-utility trade-offs with extensive experiments on image classification and segmentation tasks under various real-world heterogeneous federated learning settings on both natural and medical images. Code is available at  https://github.com/lsl001006/FedIOD.

----

## [2461] Scaling Up Pareto Optimization for Tree Structures with Affine Transformations: Evaluating Hybrid Floating Solar-Hydropower Systems in the Amazon

**Authors**: *Marc Grimson, Rafael Almeida, Qinru Shi, Yiwei Bai, Hector Angarita, Felipe Siqueira Pacheco, Rafael Schmitt, Alexander Flecker, Carla P. Gomes*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i20.30210](https://doi.org/10.1609/aaai.v38i20.30210)

**Abstract**:

Sustainability challenges inherently involve the consideration of multiple competing objectives. The Pareto frontier – the set of all optimal solutions that cannot be improved with respect to one objective without negatively affecting another – is a crucial decision-making tool for navigating sustainability challenges as it highlights the inherent trade-offs among conflicting objectives. Our research is motivated by the strategic planning of hydropower in the Amazon basin, one of the earth’s largest and most biodiverse river systems, where the need to increase energy production coincides with the pressing requirement of minimizing detrimental environmental impacts. We investigate an innovative strategy that pairs hydropower with Floating Photovoltaic Solar Panels (FPV). We provide a new extended multi-tree network formulation, which enables the consideration of multiple dam configurations.  To address the computational challenge of scaling up the Pareto optimization framework to tackle multiple objectives across the entire Amazon basin, we further enhance the state-of-the-art algorithm for Pareto frontiers in tree-structured networks with two improvements. We introduce affine transformations induced by the sub-frontiers to compute Pareto dominance and provide strategies for merging sub-trees,  significantly increasing the pruning of dominated solutions. Our experiments demonstrate considerable speedups, in some cases by more than an order of magnitude, while maintaining optimality guarantees, thus allowing us to more effectively approximate the Pareto frontiers. Moreover, our findings suggest significant shifts towards higher energy values in the Pareto frontier when pairing hybrid hydropower with FPV solutions, potentially amplifying energy production while mitigating adverse impacts.

----

## [2462] Fair Multivariate Adaptive Regression Splines for Ensuring Equity and Transparency

**Authors**: *Parian Haghighat, Denisa Gándara, Lulu Kang, Hadis Anahideh*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i20.30211](https://doi.org/10.1609/aaai.v38i20.30211)

**Abstract**:

Predictive analytics has been widely used in various domains, including education, to inform decision-making and improve outcomes. However, many predictive models are proprietary and inaccessible for evaluation or modification by researchers and practitioners, limiting their accountability and ethical design. Moreover, predictive models are often opaque and incomprehensible to the officials who use them, reducing their trust and utility. Furthermore, predictive models may introduce or exacerbate bias and inequity, as they have done in many sectors of society. Therefore, there is a need for transparent, interpretable, and fair predictive models that can be easily adopted and adapted by different stakeholders. In this paper, we propose a fair predictive model based on multivariate adaptive regression splines (MARS) that incorporates fairness measures in the learning process. MARS is a non-parametric regression model that performs feature selection, handles non-linear relationships, generates interpretable decision rules, and derives optimal splitting criteria on the variables. Specifically, we integrate fairness into the knot optimization algorithm and provide theoretical and empirical evidence of how it results in a fair knot placement. We apply our fairMARS model to real-world data and demonstrate its effectiveness in terms of accuracy and equity. Our paper contributes to the advancement of responsible and ethical predictive analytics for social good.

----

## [2463] Fair Graph Learning Using Constraint-Aware Priority Adjustment and Graph Masking in River Networks

**Authors**: *Erhu He, Yiqun Xie, Alexander Sun, Jacob Zwart, Jie Yang, Zhenong Jin, Yang Wang, Hassan A. Karimi, Xiaowei Jia*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i20.30212](https://doi.org/10.1609/aaai.v38i20.30212)

**Abstract**:

Accurate prediction of water quality and quantity is crucial for sustainable development and human well-being. However, existing data-driven methods often suffer from spatial biases in model performance due to heterogeneous data, limited observations, and noisy sensor data. To overcome these challenges, we propose Fair-Graph, a novel graph-based recurrent neural network that leverages interrelated knowledge from multiple rivers to predict water flow and temperature within large-scale stream networks. Additionally, we introduce node-specific graph masks for information aggregation and adaptation to enhance prediction over heterogeneous river segments. To reduce performance disparities across river segments, we introduce a centralized coordination strategy that adjusts training priorities for segments. We evaluate the prediction of water temperature within the Delaware River Basin, and the prediction of streamflow using simulated data from U.S. National Water Model in the Houston River network. The results showcase improvements in predictive performance and highlight the proposed model's ability to maintain spatial fairness over different river segments.

----

## [2464] Multi-Modal Discussion Transformer: Integrating Text, Images and Graph Transformers to Detect Hate Speech on Social Media

**Authors**: *Liam Hebert, Gaurav Sahu, Yuxuan Guo, Nanda Kishore Sreenivas, Lukasz Golab, Robin Cohen*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i20.30213](https://doi.org/10.1609/aaai.v38i20.30213)

**Abstract**:

We present the Multi-Modal Discussion Transformer (mDT), a novel method for detecting hate speech on online social networks such as Reddit discussions. In contrast to traditional comment-only methods, our approach to labelling a comment as hate speech involves a holistic analysis of text and images grounded in the discussion context. This is done by leveraging graph transformers to capture the contextual relationships in the discussion surrounding a comment and grounding the interwoven fusion layers that combine text and image embeddings instead of processing modalities separately. To evaluate our work, we present a new dataset, HatefulDiscussions, comprising complete multi-modal discussions from multiple online communities on Reddit.  We compare the performance of our model to baselines that only process individual comments and conduct extensive ablation studies.

----

## [2465] Bad Actor, Good Advisor: Exploring the Role of Large Language Models in Fake News Detection

**Authors**: *Beizhe Hu, Qiang Sheng, Juan Cao, Yuhui Shi, Yang Li, Danding Wang, Peng Qi*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i20.30214](https://doi.org/10.1609/aaai.v38i20.30214)

**Abstract**:

Detecting fake news requires both a delicate sense of diverse clues and a profound understanding of the real-world background, which remains challenging for detectors based on small language models (SLMs) due to their knowledge and capability limitations. Recent advances in large language models (LLMs) have shown remarkable performance in various tasks, but whether and how LLMs could help with fake news detection remains underexplored. In this paper, we investigate the potential of LLMs in fake news detection. First, we conduct an empirical study and find that a sophisticated LLM such as GPT 3.5 could generally expose fake news and provide desirable multi-perspective rationales but still underperforms the basic SLM, fine-tuned BERT. Our subsequent analysis attributes such a gap to the LLM's inability to select and integrate rationales properly to conclude. Based on these findings, we propose that current LLMs may not substitute fine-tuned SLMs in fake news detection but can be a good advisor for SLMs by providing multi-perspective instructive rationales. To instantiate this proposal, we design an adaptive rationale guidance network for fake news detection (ARG), in which SLMs selectively acquire insights on news analysis from the LLMs' rationales. We further derive a rationale-free version of ARG by distillation, namely ARG-D, which services cost-sensitive scenarios without inquiring LLMs. Experiments on two real-world datasets demonstrate that ARG and ARG-D outperform three types of baseline methods, including SLM-based, LLM-based, and combinations of small and large language models.

----

## [2466] Long-Term Fair Decision Making through Deep Generative Models

**Authors**: *Yaowei Hu, Yongkai Wu, Lu Zhang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i20.30215](https://doi.org/10.1609/aaai.v38i20.30215)

**Abstract**:

This paper studies long-term fair machine learning which aims to mitigate group disparity over the long term in sequential decision-making systems. To define long-term fairness, we leverage the temporal causal graph and use the 1-Wasserstein distance between the interventional distributions of different demographic groups at a sufficiently large time step as the quantitative metric. Then, we propose a three-phase learning framework where the decision model is trained on high-fidelity data generated by a deep generative model. We formulate the optimization problem as a performative risk minimization and adopt the repeated gradient descent algorithm for learning. The empirical evaluation shows the efficacy of the proposed method using both synthetic and semi-synthetic datasets.

----

## [2467] CityPulse: Fine-Grained Assessment of Urban Change with Street View Time Series

**Authors**: *Tianyuan Huang, Zejia Wu, Jiajun Wu, Jackelyn Hwang, Ram Rajagopal*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i20.30216](https://doi.org/10.1609/aaai.v38i20.30216)

**Abstract**:

Urban transformations have profound societal impact on both individuals and communities at large. Accurately assessing these shifts is essential for understanding their underlying causes and ensuring sustainable urban planning. Traditional measurements often encounter constraints in spatial and temporal granularity, failing to capture real-time physical changes. While street view imagery, capturing the heartbeat of urban spaces in a pedestrian point of view, can add as a high-definition, up-to-date, and on-the-ground visual proxy of urban change. We curate the largest street view time series dataset to date, and propose an end-to-end change detection model to effectively capture physical alterations in the built environment at scale. We demonstrate the effectiveness of our proposed method by benchmark comparisons with previous literature and implementing it at the city-wide level. Our approach has the potential to supplement existing dataset and serve as a fine-grained and accurate assessment of urban change.

----

## [2468] iTrendRNN: An Interpretable Trend-Aware RNN for Meteorological Spatiotemporal Prediction

**Authors**: *Xu Huang, Chuyao Luo, Bowen Zhang, Huiwei Lin, Xutao Li, Yunming Ye*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i20.30217](https://doi.org/10.1609/aaai.v38i20.30217)

**Abstract**:

Accurate prediction of meteorological elements, such as temperature and relative humidity, is important to human livelihood, early warning of extreme weather, and urban governance. Recently, neural network-based methods have shown impressive performance in this field. However, most of them are overcomplicated and impenetrable. In this paper, we propose a straightforward and interpretable differential framework, where the key lies in explicitly estimating the evolutionary trends. Specifically, three types of trends are exploited. (1) The proximity trend simply uses the most recent changes. It works well for approximately linear evolution. (2) The sequential trend explores the global information, aiming to capture the nonlinear dynamics. Here, we develop an attention-based trend unit to help memorize long-term features. (3) The flow trend is motivated by the nature of evolution, i.e., the heat or substance flows from one region to another. Here, we design a flow-aware attention unit. It can reflect the interactions via performing spatial attention over flow maps. Finally, we develop a trend fusion module to adaptively fuse the above three trends. Extensive experiments on two datasets demonstrate the effectiveness of our method.

----

## [2469] Where It Really Matters: Few-Shot Environmental Conservation Media Monitoring for Low-Resource Languages

**Authors**: *Sameer Jain, Sedrick Scott Keh, Shova Chettri, Karun Dewan, Pablo Izquierdo, Johanna Prussman, Pooja Shrestha, César Suárez, Zheyuan Ryan Shi, Lei Li, Fei Fang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i20.30218](https://doi.org/10.1609/aaai.v38i20.30218)

**Abstract**:

Environmental conservation organizations routinely monitor news content on conservation in protected areas to maintain situational awareness of developments that can have an environmental impact. Existing automated media monitoring systems require large amounts of data labeled by domain experts, which is only feasible at scale for high-resource languages like English. However, such tools are most needed in the global south where the news of interest is mainly in local low-resource languages, and far fewer experts are available to annotate datasets on a sustainable basis. In this paper, we propose NewsSerow, a method to automatically recognize environmental conservation content in low-resource languages. NewsSerow is a pipeline of summarization, in-context few-shot classification, and self-reflection using large language models (LLMs). Using at most 10 demonstration example news articles in Nepali, NewsSerow significantly outperforms other few-shot methods and can achieve comparable performance with models fully fine-tuned using thousands of examples. With NewsSerow, Organization X has been able to deploy the media monitoring tool in Nepal, significantly reducing their operational burden, and ensuring that AI tools for conservation actually reach the communities that need them the most. NewsSerow has also been deployed for countries with other languages like Colombia.

----

## [2470] Active Reinforcement Learning for Robust Building Control

**Authors**: *Doseok Jang, Larry Yan, Lucas Spangher, Costas J. Spanos*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i20.30219](https://doi.org/10.1609/aaai.v38i20.30219)

**Abstract**:

Reinforcement learning (RL) is a powerful tool for optimal control that has found great success in Atari games, the game of Go, robotic control, and building optimization. RL is also very brittle; agents often overfit to their training environment and fail to generalize to new settings. Unsupervised environment design (UED) has been proposed as a solution to this problem, in which the agent trains in environments that have been specially selected to help it learn.  Previous UED algorithms focus on trying to train an RL agent that generalizes across a large distribution of environments. This is not necessarily desirable when we wish to prioritize performance in one environment over others. In this work, we will be examining the setting of robust RL building control, where we wish to train an RL agent that prioritizes performing well in normal weather while still being robust to extreme weather conditions. We demonstrate a novel UED algorithm, ActivePLR, that uses uncertainty-aware neural network architectures to generate new training environments at the limit of the RL agent's ability while being able to prioritize performance in a desired base environment. We show that ActivePLR is able to outperform state-of-the-art UED algorithms in minimizing energy usage while maximizing occupant comfort in the setting of building control.

----

## [2471] Adversarial Fairness Network

**Authors**: *Taeuk Jang, Xiaoqian Wang, Heng Huang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i20.30220](https://doi.org/10.1609/aaai.v38i20.30220)

**Abstract**:

Fairness is becoming a rising concern in machine learning. Recent research has discovered that state-of-the-art models are amplifying social bias by making biased prediction towards some population groups (characterized by sensitive features like race or gender). Such unfair prediction among groups renders trust issues and ethical concerns in machine learning, especially for sensitive fields such as employment, criminal justice, and trust score assessment. In this paper, we introduce a new framework to improve machine learning fairness. The goal of our model is to minimize the influence of sensitive feature from the perspectives of both data input and predictive model. To achieve this goal, we reformulate the data input by eliminating the sensitive information and strengthen model fairness by minimizing the marginal contribution of the sensitive feature. We propose to learn the sensitive-irrelevant input via sampling among features and design an adversarial network to minimize the dependence between the reformulated input and the sensitive information. Empirical results validate that our model achieves comparable or better results than related state-of-the-art methods w.r.t. both fairness metrics and prediction performance.

----

## [2472] Unraveling Pain Levels: A Data-Uncertainty Guided Approach for Effective Pain Assessment

**Authors**: *Xinwei Ji, Xiaomin Chang, Wei Li, Albert Y. Zomaya*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i20.30221](https://doi.org/10.1609/aaai.v38i20.30221)

**Abstract**:

Pain, a primary reason for seeking medical help, requires essential pain assessment for effective management. Studies have recognized electrodermal activity (EDA) signaling's potential for automated pain assessment, but traditional algorithms often ignore the noise and uncertainty inherent in pain data. To address this, we propose a learning framework predicated on data uncertainty, introducing two forms: a) subject-level stimulation-reaction drift; b) ambiguity in self-reporting scores. We formulate an uncertainty assessment using Heart Rate Variability (HRV) features to guide the selection of responsive pain profiles and reweight subtask importance based on the vagueness of self-reported data. These methods are integrated within an end-to-end neural network learning paradigm, focusing the detector on more accurate insights within the uncertainty domain. Extensive experimentation on both the publicly available biovid dataset and the proprietary Apon dataset demonstrates our approach's effectiveness. In the biovid dataset, we achieved a 6% enhancement over the state-of-the-art methodology, and on the Apon dataset, our method outperformed baseline approaches by over 20%.

----

## [2473] Outlier Ranking for Large-Scale Public Health Data

**Authors**: *Ananya Joshi, Tina Townes, Nolan Gormley, Luke Neureiter, Roni Rosenfeld, Bryan Wilder*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i20.30222](https://doi.org/10.1609/aaai.v38i20.30222)

**Abstract**:

Disease control experts inspect public health data streams daily for outliers worth investigating, like those corresponding to data quality issues or disease outbreaks. However, they can only examine a few of the thousands of maximally-tied outliers returned by univariate outlier detection methods applied to large-scale public health data streams. To help experts distinguish the most important outliers from these thousands of tied outliers, we propose a new task for algorithms to rank the outputs of any univariate method applied to each of many streams. Our novel algorithm for this task, which leverages hierarchical networks and extreme value analysis, performed the best across traditional outlier detection metrics in a human-expert evaluation using public health data streams. Most importantly, experts have used our open-source Python implementation since April 2023 and report identifying outliers worth investigating 9.1x faster than their prior baseline. Other organizations can readily adapt this implementation to create rankings from the outputs of their tailored univariate methods across large-scale streams.

----

## [2474] Deploying ADVISER: Impact and Lessons from Using Artificial Intelligence for Child Vaccination Uptake in Nigeria

**Authors**: *Opadele Kehinde, Ruth Abdul, Bose Afolabi, Parminder Vir, Corinne Namblard, Ayan Mukhopadhyay, Abiodun Adereni*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i20.30223](https://doi.org/10.1609/aaai.v38i20.30223)

**Abstract**:

More than 5 million children under five years die from largely preventable or treatable medical conditions every year, with an overwhelmingly large proportion of deaths occurring in underdeveloped countries with low vaccination uptake. One of the United Nations' sustainable development goals (SDG 3) aims to end preventable deaths of newborns and children under five years of age. We focus on Nigeria, where the rate of infant mortality is appalling. In particular, low vaccination uptake in Nigeria is a major driver of more than 2,000 daily deaths of children under the age of five years. In this paper, we describe our collaboration with government partners in Nigeria to deploy ADVISER: AI-Driven Vaccination Intervention Optimiser. The framework, based on an integer linear program that seeks to maximize the cumulative probability of successful vaccination, is the first successful deployment of an AI-enabled toolchain for optimizing the allocation of health interventions in Nigeria. In this paper, we provide a background of the ADVISER framework and present results, lessons, and success stories of deploying ADVISER to more than 13,000 families in the state of Oyo, Nigeria.

----

## [2475] Vector Field Oriented Diffusion Model for Crystal Material Generation

**Authors**: *Astrid Klipfel, Yaël Frégier, Adlane Sayede, Zied Bouraoui*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i20.30224](https://doi.org/10.1609/aaai.v38i20.30224)

**Abstract**:

Discovering crystal structures with specific chemical properties has become an increasingly important focus in material science. However, current models are limited in their ability to generate new crystal lattices, as they only consider atomic positions or chemical composition. To address this issue, we propose a probabilistic diffusion model that utilizes a geometrically equivariant GNN to consider atomic positions and crystal lattices jointly. To evaluate the effectiveness of our model, we introduce a new generation metric inspired by Frechet Inception Distance, but based on GNN energy prediction rather than InceptionV3 used in computer vision. In addition to commonly used metrics like validity, which assesses the plausibility of a structure, this new metric offers a more comprehensive evaluation of our model's capabilities. Our experiments on existing benchmarks show the significance of our diffusion model. We also show that our method can effectively learn meaningful representations.

----

## [2476] Combining Deep Learning and Street View Imagery to Map Smallholder Crop Types

**Authors**: *Jordi Laguarta Soler, Thomas Friedel, Sherrie Wang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i20.30225](https://doi.org/10.1609/aaai.v38i20.30225)

**Abstract**:

Accurate crop type maps are an essential source of information for monitoring yield progress at scale, projecting global crop production, and planning effective policies. To date, however, crop type maps remain challenging to create in low- and middle-income countries due to a lack of ground truth labels for training machine learning models. Field surveys are the gold standard in terms of accuracy but require an often-prohibitively large amount of time, money, and statistical capacity. 
In recent years, street-level imagery, such as Google Street View, KartaView, and Mapillary, has become available around the world. Such imagery contains rich information about crop types grown at particular locations and times. 
In this work, we develop an automated system to generate crop type ground references using deep learning and Google Street View imagery. The method efficiently curates a set of street-view images containing crop fields, trains a model to predict crop types using either weakly-labeled images from disparate out-of-domain sources or zero-shot labeled street view images with GPT-4V, and combines the predicted labels with remote sensing time series to create a wall-to-wall crop type map. We show that, in Thailand, the resulting country-wide map of rice, cassava, maize, and sugarcane achieves an accuracy of 93%. We publicly release the first-ever crop type map for all of Thailand 2022 at 10m-resolution with no gaps. To our knowledge, this is the first time a 10m-resolution, multi-crop map has been created for any smallholder country. As the availability of roadside imagery expands, our pipeline provides a way to map crop types at scale around the globe, especially in underserved smallholder regions.

----

## [2477] GLH-Water: A Large-Scale Dataset for Global Surface Water Detection in Large-Size Very-High-Resolution Satellite Imagery

**Authors**: *Yansheng Li, Bo Dang, Wanchun Li, Yongjun Zhang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i20.30226](https://doi.org/10.1609/aaai.v38i20.30226)

**Abstract**:

Global surface water detection in very-high-resolution (VHR) satellite imagery can directly serve major applications such as refined flood mapping and water resource assessment. Although achievements have been made in detecting surface water in small-size satellite images corresponding to local geographic scales, datasets and methods suitable for mapping and analyzing global surface water have yet to be explored. To encourage the development of this task and facilitate the implementation of relevant applications, we propose the GLH-water dataset that consists of 250 satellite images and 40.96 billion pixels labeled surface water annotations that are distributed globally and contain water bodies exhibiting a wide variety of types (e.g. , rivers, lakes, and ponds in forests, irrigated fields, bare areas, and urban areas). Each image is of the size 12,800 × 12,800 pixels at 0.3 meter spatial resolution. To build a benchmark for GLH-water, we perform extensive experiments employing representative surface water detection models, popular semantic segmentation models, and ultra-high resolution segmentation models. Furthermore, we also design a strong baseline with the novel pyramid consistency loss (PCL) to initially explore this challenge, increasing IoU by 2.4% over the next best baseline. Finally, we implement the cross-dataset generalization and pilot area application experiments, and the superior performance illustrates the strong generalization and practical application value of GLH-water dataset. Project page: https://jack-bo1220.github.io/project/GLH-water.html

----

## [2478] AutoLTS: Automating Cycling Stress Assessment via Contrastive Learning and Spatial Post-processing

**Authors**: *Bo Lin, Shoshanna Saxe, Timothy C. Y. Chan*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i20.30227](https://doi.org/10.1609/aaai.v38i20.30227)

**Abstract**:

Cycling stress assessment, which quantifies cyclists' perceived stress imposed by the built environment and motor traffics, increasingly informs cycling infrastructure planning and cycling route recommendation. However, currently calculating cycling stress is slow and data-intensive, which hinders its broader application. In this paper, We propose a deep learning framework to support accurate, fast, and large-scale cycling stress assessments for urban road networks based on street-view images. Our framework features i) a contrastive learning approach that leverages the ordinal relationship among cycling stress labels, and ii) a post-processing technique that enforces spatial smoothness into our predictions. On a dataset of 39,153 road segments collected in Toronto, Canada, our results demonstrate the effectiveness of our deep learning framework and the value of using image data for cycling stress assessment in the absence of high-quality road geometry and motor traffic data.

----

## [2479] Depression Detection via Capsule Networks with Contrastive Learning

**Authors**: *Han Liu, Changya Li, Xiaotong Zhang, Feng Zhang, Wei Wang, Fenglong Ma, Hongyang Chen, Hong Yu, Xianchao Zhang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i20.30228](https://doi.org/10.1609/aaai.v38i20.30228)

**Abstract**:

Depression detection is a challenging and crucial task in psychological illness diagnosis. Utilizing online user posts to predict whether a user suffers from depression seems an effective and promising direction. However, existing methods suffer from either poor interpretability brought by the black-box models or underwhelming performance caused by the completely separate two-stage model structure. To alleviate these limitations, we propose a novel capsule network integrated with contrastive learning for depression detection (DeCapsNet). The highlights of DeCapsNet can be summarized as follows. First, it extracts symptom capsules from user posts by leveraging meticulously designed symptom descriptions, and then distills them into class-indicative depression capsules. The overall workflow is in an explicit hierarchical reasoning manner and can be well interpreted by the Patient Health Questionnaire-9 (PHQ9), which is one of the most widely adopted questionnaires for depression diagnosis. Second, it integrates with contrastive learning, which can facilitate the embeddings from the same class to be pulled closer, while simultaneously pushing the embeddings from different classes apart. In addition, by adopting the end-to-end training strategy, it does not necessitate additional data annotation, and mitigates the potential adverse effects from the upstream task to the downstream task. Extensive experiments on three widely-used datasets show that in both within-dataset and cross-dataset scenarios our proposed method outperforms other strong baselines significantly.

----

## [2480] On the Actionability of Outcome Prediction

**Authors**: *Lydia T. Liu, Solon Barocas, Jon M. Kleinberg, Karen Levy*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i20.30229](https://doi.org/10.1609/aaai.v38i20.30229)

**Abstract**:

Predicting future outcomes is a prevalent application of machine learning in social impact domains. Examples range from predicting student success in education to predicting disease risk in healthcare. Practitioners recognize that the ultimate goal is not just to predict but to act effectively. Increasing evidence suggests that relying on outcome predictions for downstream interventions may not have desired results. 

In most domains there exists a multitude of possible interventions for each individual, making the challenge of taking effective action more acute. Even when causal mechanisms connecting the individual's latent states to outcomes are well understood, in any given instance (a specific student or patient), practitioners still need to infer---from budgeted measurements of latent states---which of many possible interventions will be most effective for this individual. With this in mind, we ask: when are accurate predictors of outcomes helpful for identifying the most suitable intervention?

Through a simple model encompassing actions, latent states, and measurements, we demonstrate that pure outcome prediction rarely results in the most effective policy for taking actions, even when combined with other measurements. 
We find that except in cases where there is a single decisive action for improving the outcome, outcome prediction never maximizes "action value", the utility of taking actions. Making measurements of actionable latent states, where specific actions lead to desired outcomes, may considerably enhance the action value compared to outcome prediction, and the degree of improvement depends on action costs and the outcome model. This analysis emphasizes the need to go beyond generic outcome prediction in interventional settings by incorporating knowledge of plausible actions and latent states.

----

## [2481] Hear You Say You: An Efficient Framework for Marine Mammal Sounds' Classification

**Authors**: *Xiangrui Liu, Xiaoou Liu, Shan Du, Julian Cheng*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i20.30230](https://doi.org/10.1609/aaai.v38i20.30230)

**Abstract**:

Marine mammals and their ecosystem face significant threats from, for example, military active sonar and marine transportation. To mitigate this harm, early detection and classification of marine mammals are essential. While recent efforts have utilized spectrogram analysis and machine learning techniques, there remain challenges in their efficiency. Therefore, we propose a novel knowledge distillation framework, named XCFSMN, for this problem. We construct a teacher model that fuses the features extracted from an X-vector extractor, a DenseNet and Cross-Covariance attended compact Feed-Forward Sequential Memory Network (cFSMN). The teacher model transfers knowledge to a simpler cFSMN model through a temperature-cooling strategy for efficient learning. Compared to multiple convolutional neural network backbones and transformers, the proposed framework achieves state-of-the-art efficiency and performance. The improved model size is approximately 20 times smaller and the inference time can be 10 times shorter without affecting the model’s accuracy.

----

## [2482] Identifying and Addressing Disparities in Public Libraries with Bayesian Latent Variable Modeling

**Authors**: *Zhi Liu, Sarah Rankin, Nikhil Garg*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i20.30231](https://doi.org/10.1609/aaai.v38i20.30231)

**Abstract**:

Public libraries are an essential public good. We ask: are urban library systems providing equitable service to all residents, in terms of the books they have access to and check out? If not, what causes disparities: heterogeneous book collections, resident behavior and access, and/or operational policies? Existing methods leverage only system-level outcome data (such as overall checkouts per branch), and so cannot distinguish between these factors. As a result, it is difficult to use their results to guide interventions to increase equitable access. We propose a Bayesian framework to characterize book checkout behavior across multiple branches of a library system, learning heterogeneous book popularity, overall branch demand, and usage of the online hold system, while controlling for book availability.

In collaboration with the New York Public Library, we apply our framework to granular data consisting of over 400,000 checkouts during 2022. We first show that our model significantly out-performs baseline methods in predicting checkouts at the book-branch level. Next, we study spatial and socioeconomic disparities. We show that disparities are largely driven by disparate use of the online holds system, which allows library patrons to receive books from any other branch through an online portal. This system thus leads to a large outflow of popular books from branches in lower income neighborhoods to those in high income ones. Finally, we illustrate the use of our model and insights to quantify the impact of potential interventions, such as changing how books are internally routed between branches to fulfill hold requests.

----

## [2483] Interpretable Long-Form Legal Question Answering with Retrieval-Augmented Large Language Models

**Authors**: *Antoine Louis, Gijs van Dijck, Gerasimos Spanakis*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i20.30232](https://doi.org/10.1609/aaai.v38i20.30232)

**Abstract**:

Many individuals are likely to face a legal dispute at some point in their lives, but their lack of understanding of how to navigate these complex issues often renders them vulnerable. The advancement of natural language processing opens new avenues for bridging this legal literacy gap through the development of automated legal aid systems. However, existing legal question answering (LQA) approaches often suffer from a narrow scope, being either confined to specific legal domains or limited to brief, uninformative responses. In this work, we propose an end-to-end methodology designed to generate long-form answers to any statutory law questions, utilizing a "retrieve-then-read" pipeline. To support this approach, we introduce and release the Long-form Legal Question Answering (LLeQA) dataset, comprising 1,868 expert-annotated legal questions in the French language, complete with detailed answers rooted in pertinent legal provisions. Our experimental results demonstrate promising performance on automatic evaluation metrics, but a qualitative analysis uncovers areas for refinement. As one of the only comprehensive, expert-annotated long-form LQA dataset, LLeQA has the potential to not only accelerate research towards resolving a significant real-world issue, but also act as a rigorous benchmark for evaluating NLP models in specialized domains. We publicly release our code, data, and models.

----

## [2484] T-NET: Weakly Supervised Graph Learning for Combatting Human Trafficking

**Authors**: *Pratheeksha Nair, Javin Liu, Catalina Vajiac, Andreas M. Olligschlaeger, Duen Horng Chau, Mirela T. Cazzolato, Cara Jones, Christos Faloutsos, Reihaneh Rabbany*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i20.30233](https://doi.org/10.1609/aaai.v38i20.30233)

**Abstract**:

Human trafficking (HT) for forced sexual exploitation, often described as modern-day slavery, is a pervasive problem that affects millions of people worldwide. Perpetrators of this crime post advertisements (ads) on behalf of their victims on adult service websites (ASW). These websites typically contain hundreds of thousands of ads including those posted by independent escorts, massage parlor agencies and spammers (fake ads). Detecting suspicious activity in these ads is difficult and developing data-driven methods is challenging due to the  hard-to-label, complex and sensitive nature of the data. 

In this paper, we propose T-Net, which unlike previous solutions, formulates this problem as weakly supervised classification. Since it takes several months to years to investigate a case and obtain a single definitive label, we design domain-specific signals or indicators that provide weak labels. T-Net also looks into connections between ads and models the problem as a graph learning task instead of classifying ads independently. We show that T-Net outperforms all baselines on a real-world dataset of ads by 7% average weighted F1 score. Given that this data contains personally identifiable information, we also present a realistic data generator and provide the first publicly available dataset in this domain which may be leveraged by the wider research community.

----

## [2485] Promoting Fair Vaccination Strategies through Influence Maximization: A Case Study on COVID-19 Spread

**Authors**: *Nicola Neophytou, Afaf Taïk, Golnoosh Farnadi*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i20.30234](https://doi.org/10.1609/aaai.v38i20.30234)

**Abstract**:

The aftermath of the Covid-19 pandemic saw more severe outcomes for racial minority groups and economically-deprived communities. Such disparities can be explained by several factors, including unequal access to healthcare, as well as the inability of low income groups to reduce their mobility due to work or social obligations. Moreover, senior citizens were found to be more susceptible to severe symptoms, largely due to age-related health reasons. Adapting vaccine distribution strategies to consider a range of demographics is therefore essential to address these disparities. In this study, we propose a novel approach that utilizes influence maximization (IM) on mobility networks to develop vaccination strategies which incorporate demographic fairness. By considering factors such as race, social status, age, and associated risk factors, we aim to optimize vaccine distribution to achieve various fairness definitions for one or more protected attributes at a time. Through extensive experiments conducted on Covid-19 spread in three major metropolitan areas across the United States, we demonstrate the effectiveness of our proposed approach in reducing disease transmission and promoting fairness in vaccination distribution.

----

## [2486] DISCount: Counting in Large Image Collections with Detector-Based Importance Sampling

**Authors**: *Gustavo Pérez, Subhransu Maji, Daniel Sheldon*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i20.30235](https://doi.org/10.1609/aaai.v38i20.30235)

**Abstract**:

Many applications use computer vision to detect and count objects in massive image collections. However, automated methods may fail to deliver accurate counts, especially when the task is very difficult or requires a fast response time. For example, during disaster response, aid organizations aim to quickly count damaged buildings in satellite images to plan relief missions, but pre-trained building and damage detectors often perform poorly due to domain shifts. In such cases, there is a need for human-in-the-loop approaches to accurately count with minimal human effort. We propose DISCount -- a detector-based importance sampling framework for counting in large image collections. DISCount uses an imperfect detector and human screening to estimate low-variance unbiased counts. We propose techniques for counting over multiple spatial or temporal regions using a small amount of screening and estimate confidence intervals.  This enables end-users to stop screening when estimates are sufficiently accurate, which is often the goal in real-world applications.  We demonstrate our method with two applications: counting birds in radar imagery to understand responses to climate change, and counting damaged buildings in satellite imagery for damage assessment in regions struck by a natural disaster. On the technical side we develop variance reduction techniques based on control variates and prove the (conditional) unbiasedness of the estimators.  DISCount leads to a 9-12x reduction in the labeling costs to obtain the same error rates compared to naive screening for tasks we consider, and surpasses alternative covariate-based screening approaches.

----

## [2487] Discretionary Trees: Understanding Street-Level Bureaucracy via Machine Learning

**Authors**: *Gaurab Pokharel, Sanmay Das, Patrick J. Fowler*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i20.30236](https://doi.org/10.1609/aaai.v38i20.30236)

**Abstract**:

Street-level bureaucrats interact directly with people on behalf of government agencies to perform a wide range of functions, including, for example, administering social services and policing. A key feature of street-level bureaucracy is that the civil servants, while tasked with implementing agency policy, are also granted significant discretion in how they choose to apply that policy in individual cases. Using that discretion could be beneficial, as it allows for exceptions to policies based on human interactions and evaluations, but it could also allow biases and inequities to seep into important domains of societal resource allocation. In this paper, we use machine learning techniques to understand street-level bureaucrats' behavior. We leverage a rich dataset that combines demographic and other information on households with information on which homelessness interventions they were assigned during a period when assignments were not formulaic. We find that  caseworker decisions in this time are highly predictable overall, and some, but not all of this predictivity can be captured by simple decision rules. We theorize that the decisions not captured by the simple decision rules can be considered applications of caseworker discretion. These discretionary decisions are far from random in both the characteristics of such households and in terms of the outcomes of the decisions. Caseworkers typically only apply discretion to households that would be considered less vulnerable. When they do apply discretion to assign households to more intensive interventions, the marginal benefits to those households are significantly higher than would be expected if the households were chosen at random; there is no similar reduction in marginal benefit to households that are discretionarily allocated less intensive interventions, suggesting that caseworkers are using their knowledge and experience to improve outcomes for households experiencing homelessness.

----

## [2488] IndicCONAN: A Multilingual Dataset for Combating Hate Speech in Indian Context

**Authors**: *Nihar Ranja Sahoo, Gyana Prakash Beria, Pushpak Bhattacharyya*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i20.30237](https://doi.org/10.1609/aaai.v38i20.30237)

**Abstract**:

Hate speech (HS) is a growing concern in many parts of
the world, including India, where it has led to numerous instances of violence and discrimination. The development of
effective counter-narratives (CNs) is a critical step in combating hate speech, but there is a lack of research in this
area, especially in non-English languages. In this paper, we
introduce a new dataset, IndicCONAN, of counter-narratives
against hate speech in Hindi and Indian English. We propose a scalable human-in-the-loop approach for generating counter-narratives by an auto-regressive language model
through machine generation - human correction cycle, where
the model uses augmented data from previous cycles to generate new training samples. These newly generated samples
are then reviewed and edited by annotators, leading to further
model refnement. The dataset consists of over 2,500 exam- ˜
ples of counter-narratives each in both English and Hindi corresponding to various hate speeches in the Indian context. We
also present a framework for generating CNs conditioned on
specifc CN type with a mean perplexity of 3.85 for English
and 3.70 for Hindi, a mean toxicity score of 0.04 for English
and 0.06 for Hindi, and a mean diversity of 0.08 for English
and 0.14 for Hindi. Our dataset and framework provide valuable resources for researchers and practitioners working to
combat hate speech in the Indian context.

----

## [2489] Carbon Footprint Reduction for Sustainable Data Centers in Real-Time

**Authors**: *Soumyendu Sarkar, Avisek Naug, Ricardo Luna Gutierrez, Antonio Guillen, Vineet Gundecha, Sahand Ghorbanpour, Sajad Mousavi, Dejan Markovikj, Ashwin Ramesh Babu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i20.30238](https://doi.org/10.1609/aaai.v38i20.30238)

**Abstract**:

As machine learning workloads are significantly increasing energy consumption, sustainable data centers with low carbon emissions are becoming a top priority for governments and corporations worldwide. This requires a paradigm shift in optimizing power consumption in cooling and IT loads, shifting flexible loads based on the availability of renewable energy in the power grid, and leveraging battery storage from the uninterrupted power supply in data centers, using collaborative agents. The complex association between these optimization strategies and their dependencies on variable external factors like weather and the power grid carbon intensity makes this a hard problem. Currently, a real-time controller to optimize all these goals simultaneously in a dynamic real-world setting is lacking. We propose a Data Center Carbon Footprint Reduction (DC-CFR) multi-agent Reinforcement Learning (MARL) framework that optimizes data centers for the multiple objectives of carbon footprint reduction, energy consumption, and energy cost. The results show that the DC-CFR MARL agents effectively resolved the complex interdependencies in optimizing cooling, load shifting, and energy storage in real-time for various locations under real-world dynamic weather and grid carbon intensity conditions. DC-CFR significantly outperformed the industry-standard ASHRAE controller with a considerable reduction in carbon emissions (14.5%), energy usage (14.4%), and energy cost (13.7%) when evaluated over one year across multiple geographical regions.

----

## [2490] Evaluating Pre-trial Programs Using Interpretable Machine Learning Matching Algorithms for Causal Inference

**Authors**: *Travis Seale-Carlisle, Saksham Jain, Courtney Lee, Caroline Levenson, Swathi Ramprasad, Brandon Garrett, Sudeepa Roy, Cynthia Rudin, Alexander Volfovsky*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i20.30239](https://doi.org/10.1609/aaai.v38i20.30239)

**Abstract**:

After a person is arrested and charged with a crime, they may be released on bail and required to participate in a community supervision program while awaiting trial. These 'pre-trial programs' are common throughout the United States, but very little research has demonstrated their effectiveness. Researchers have emphasized the need for more rigorous program evaluation methods, which we introduce in this article. We describe a program evaluation pipeline that uses recent interpretable machine learning techniques for observational causal inference, and demonstrate these techniques in a study of a pre-trial program in Durham, North Carolina. Our findings show no evidence that the program either significantly increased or decreased the probability of new criminal charges. If these findings replicate, the criminal-legal system needs to either improve pre-trial programs or consider alternatives to them. The simplest option is to release low-risk individuals back into the community without subjecting them to any restrictions or conditions. Another option is to assign individuals to pre-trial programs that incentivize pro-social behavior.  We believe that the techniques introduced here can provide researchers the rigorous tools they need to evaluate these programs.

----

## [2491] Self-Supervised Framework Based on Subject-Wise Clustering for Human Subject Time Series Data

**Authors**: *Eunseon Seong, Harim Lee, Dong-Kyu Chae*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i20.30240](https://doi.org/10.1609/aaai.v38i20.30240)

**Abstract**:

With the widespread adoption of IoT, wearable devices, and sensors, time series data from human subjects are significantly increasing in the healthcare domain. Due to the laborious nature of manual annotation in time series data and the requirement for human experts, self-supervised learning methods are attempted to alleviate the limited label situations. While existing self-supervised methods have been successful to achieve comparable performance to the fully supervised methods, there are still some limitations that need to be addressed, considering the nature of time series data from human subjects: In real-world clinical settings, data labels (e.g., sleep stages) are usually annotated by subject-level, and there is a substantial variation in patterns between subjects. Thus, a model should be designed to deal with not only the label scarcity but also subject-wise nature of data to ensure high performance in real-world scenarios. To mitigate these issues, we propose a novel self-supervised learning framework for human subject time series data: Subject-Aware Time Series Clustering (SA-TSC). In the unsupervised representation learning phase, SA-TSC adopts a subject-wise learning strategy rather than instance-wise learning which randomly samples data instances from different subjects within the batch during training. Specifically, we generate subject-graphs with our graph construction method based on Gumbel-Softmax and perform graph spectral clustering on each subject-graph. In addition, we utilize graph neural networks to capture dependencies between channels and design our own graph learning module motivated from self-supervised loss. Experimental results show the outstanding performance of our SA-TSC with the limited & subject-wise label setting, leading to its high applicability to the healthcare industry. The code is available at: https://github.com/DILAB-HYU/SA-TSC

----

## [2492] Characterizing Information Seeking Events in Health-Related Social Discourse

**Authors**: *Omar Sharif, Madhusudan Basak, Tanzia Parvin, Ava Scharfstein, Alphonso Bradham, Jacob T. Borodovsky, Sarah E. Lord, Sarah Masud Preum*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i20.30241](https://doi.org/10.1609/aaai.v38i20.30241)

**Abstract**:

Social media sites have become a popular platform for individuals to seek and share health information. Despite the progress in natural language processing for social media mining, a gap remains in analyzing health-related texts on social discourse in the context of events. Event-driven analysis can offer insights into different facets of healthcare at an individual and collective level, including treatment options, misconceptions, knowledge gaps, etc. This paper presents a paradigm to characterize health-related information-seeking in social discourse through the lens of events. Events here are board categories defined with domain experts that capture the trajectory of the treatment/medication. To illustrate the value of this approach, we analyze Reddit posts regarding medications for Opioid Use Disorder (OUD), a critical global health concern. To the best of our knowledge, this is the first attempt to define event categories for characterizing information-seeking in OUD social discourse. Guided by domain experts, we develop TREAT-ISE, a novel multilabel treatment information-seeking event dataset to analyze online discourse on an event-based framework. This dataset contains Reddit posts on information-seeking events related to recovery from OUD, where each post is annotated based on the type of events. We also establish a strong performance benchmark (77.4% F1 score) for the task by employing several machine learning and deep learning classifiers. Finally, we thoroughly investigate the performance and errors of ChatGPT on this task, providing valuable insights into the LLM's capabilities and ongoing characterization efforts.

----

## [2493] Nowcasting Temporal Trends Using Indirect Surveys

**Authors**: *Ajitesh Srivastava, Juan Marcos Ramirez, Sergio Díaz-Aranda, José Aguilar, Antonio Fernández Anta, Antonio Ortega, Rosa Elvira Lillo*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i20.30242](https://doi.org/10.1609/aaai.v38i20.30242)

**Abstract**:

Indirect surveys, in which respondents provide information about other people they know, have been proposed for estimating (nowcasting) the size of a hidden population where privacy is important or the hidden population is hard to reach. Examples include estimating casualties in an earthquake, conditions among female sex workers, and the prevalence of drug use and infectious diseases. The Network Scale-up Method (NSUM) is the classical approach to developing estimates from indirect surveys, but it was designed for one-shot surveys. Further, it requires certain assumptions and asking for or estimating the number of individuals in each respondent's network. In recent years, surveys have been increasingly deployed online and can collect data continuously (e.g., COVID-19 surveys on Facebook during much of the pandemic). Conventional NSUM can be applied to these scenarios by analyzing the data independently at each point in time, but this misses the opportunity of leveraging the temporal dimension. We propose to use the responses from indirect surveys collected over time and develop analytical tools (i)  to prove that indirect surveys can provide better estimates for the trends of the hidden population over time, as compared to direct surveys and (ii) to identify appropriate temporal aggregations to improve the estimates. We demonstrate through extensive simulations that our approach outperforms traditional NSUM and direct surveying methods. We also empirically demonstrate the superiority of our approach on a real indirect survey dataset of COVID-19 cases.

----

## [2494] FairPlay: A Multi-Sided Fair Dynamic Pricing Policy for Hotels

**Authors**: *Errikos Streviniotis, Athina Georgara, Filippo Bistaffa, Georgios Chalkiadakis*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i20.30243](https://doi.org/10.1609/aaai.v38i20.30243)

**Abstract**:

In recent years, popular touristic destinations face overtourism. Local communities suffer from its consequences in several ways. Among others, overpricing and profiteering harms local societies and economies deeply. In this paper we focus on the problem of determining fair hotel room prices. Specifically, we put forward a dynamic pricing policy where the price of a room depends not only on the demand of the hotel it belongs to but also on the demand of: (i) similar rooms in the area and (ii) their hotels. To this purpose, we model our setting as a cooperative game and exploit an appropriate game theoretic solution concept that promotes fairness both on the customers' and the providers' side. Our simulation results involving price adjustments across real-world hotels datasets, confirm that ours is a fair dynamic pricing policy, avoiding both over- and under-pricing hotel rooms.

----

## [2495] Stable Matchings in Practice: A Constraint Programming Approach

**Authors**: *Zhaohong Sun, Naoyuki Yamada, Yoshihiro Takenami, Daisuke Moriwaki, Makoto Yokoo*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i20.30244](https://doi.org/10.1609/aaai.v38i20.30244)

**Abstract**:

We study a practical two-sided matching problem of allocating children to daycare centers, which has significant social implications. We are cooperating with several municipalities in Japan and our goal is to devise a reliable and trustworthy clearing algorithm to deal with the problem. In this paper, we describe the design of our new algorithm that minimizes the number of unmatched children while ensuring stability. We evaluate our algorithm using real-life data sets, and experimental results demonstrate that our algorithm surpasses the commercial software that currently dominates the market in terms of both the number of matched children and the number of blocking coalitions (measuring stability). Our findings have been reported to local governments, and some are considering adopting our proposed algorithm in the near future, instead of the existing solution. Moreover, our model and algorithm have broader applicability to other important matching markets, such as hospital-doctor matching with couples and school choice with siblings.

----

## [2496] Social, Legal, Ethical, Empathetic, and Cultural Rules: Compilation and Reasoning

**Authors**: *Nicolas Troquard, Martina De Sanctis, Paola Inverardi, Patrizio Pelliccione, Gian Luca Scoccia*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i20.30245](https://doi.org/10.1609/aaai.v38i20.30245)

**Abstract**:

The rise of AI-based and autonomous systems is raising concerns and apprehension due to potential negative repercussions arising from their behavior or decisions. These systems must be designed to comply with the human contexts in which they will operate. To this extent, Townsend et al. (2022) introduce the concept of SLEEC (social, legal, ethical, empathetic, or cultural) rules that aim to facilitate the formulation, verification, and enforcement of the rules AI-based and autonomous systems should obey. They lay out a methodology to elicit them and to let philosophers, lawyers, domain experts, and others to formulate them in natural language. To enable their effective use in AI systems, it is necessary to translate these rules systematically into a formal language that supports automated reasoning. In this study, we first conduct a linguistic analysis of the SLEEC rules pattern, which justifies the translation of SLEEC rules into classical logic. Then we investigate the computational complexity of reasoning about SLEEC rules and show how logical programming frameworks can be employed to implement SLEEC rules in practical scenarios. the result is a readily applicable strategy for implementing AI systems that conform to norms expressed as SLEEC rules.

----

## [2497] Preventing Eviction-Caused Homelessness through ML-Informed Distribution of Rental Assistance

**Authors**: *Catalina Vajiac, Arun Frey, Joachim Baumann, Abigail Smith, Kasun Amarasinghe, Alice Lai, Kit T. Rodolfa, Rayid Ghani*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i20.30246](https://doi.org/10.1609/aaai.v38i20.30246)

**Abstract**:

Rental assistance programs provide individuals with financial assistance to prevent housing instabilities caused by evictions and avert homelessness. Since these programs operate under resource constraints, they must decide who to prioritize. Typically, funding is distributed by a reactive allocation process that does not systematically consider risk of future homelessness. We partnered with Anonymous County (PA) to explore a proactive and preventative allocation approach that prioritizes individuals facing eviction based on their risk of future homelessness. Our ML models, trained on state and county administrative data accurately identify at-risk individuals, outperforming simpler prioritization approaches by at least 20% while meeting our equity and fairness goals across race and gender. Furthermore, our approach would reach 28% of individuals who are overlooked by the current process and end up homeless. Beyond improvements to the rental assistance program in Anonymous County, this study can inform the development of evidence-based decision support tools in similar contexts, including lessons about data needs, model design, evaluation, and field validation.

----

## [2498] RLPeri: Accelerating Visual Perimetry Test with Reinforcement Learning and Convolutional Feature Extraction

**Authors**: *Tanvi Verma, Linh Le Dinh, Nicholas Tan, Xinxing Xu, Ching-Yu Cheng, Yong Liu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i20.30247](https://doi.org/10.1609/aaai.v38i20.30247)

**Abstract**:

Visual perimetry is an important eye examination that helps detect vision problems caused by ocular or neurological conditions. During the test, a patient's gaze is fixed at a specific location while light stimuli of varying intensities are presented in central and peripheral vision. Based on the patient's responses to the stimuli, the visual field mapping and sensitivity are determined. However, maintaining high levels of concentration throughout the test can be challenging for patients, leading to increased examination times and decreased accuracy.

In this work, we present RLPeri, a reinforcement learning-based approach to optimize visual perimetry testing. By determining the optimal sequence of locations and initial stimulus values, we aim to reduce the examination time without compromising accuracy. Additionally, we incorporate reward shaping techniques to further improve the testing performance. To monitor the patient's responses over time during testing, we represent the test's state as a pair of 3D matrices. We apply two different convolutional kernels to extract spatial features across locations as well as features across different stimulus values for each location. Through experiments, we demonstrate that our approach results in a 10-20% reduction in examination time while maintaining the accuracy as compared to state-of-the-art methods. With the presented approach, we aim to make visual perimetry testing more efficient and patient-friendly, while still providing accurate results.

----

## [2499] Deep Reinforcement Learning for Early Diagnosis of Lung Cancer

**Authors**: *Yifan Wang, Qining Zhang, Lei Ying, Chuan Zhou*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i20.30248](https://doi.org/10.1609/aaai.v38i20.30248)

**Abstract**:

Lung cancer remains the leading cause of cancer-related death worldwide, and early diagnosis of lung cancer is critical for improving the survival rate of patients. Performing annual low-dose computed tomography (LDCT) screening among high-risk populations is the primary approach for early diagnosis. However, after each screening, whether to continue monitoring (with follow-up screenings) or to order a biopsy for diagnosis remains a challenging decision to make. Continuing with follow-up screenings may lead to delayed diagnosis but ordering a biopsy without sufficient evidence incurs unnecessary risk and cost. In this paper, we tackle the problem by an optimal stopping approach. Our proposed algorithm, called EarlyStop-RL, utilizes the structure of the Snell envelope for optimal stopping, and model-free deep reinforcement learning for making diagnosis decisions. Through evaluating our algorithm on a commonly used clinical trial dataset (the National Lung Screening Trial), we demonstrate that EarlyStop-RL has the potential to greatly enhance risk assessment and early diagnosis of lung cancer, surpassing the performance of two widely adopted clinical models, namely the Lung-RADS and the Brock model.

----

## [2500] SimFair: Physics-Guided Fairness-Aware Learning with Simulation Models

**Authors**: *Zhihao Wang, Yiqun Xie, Zhili Li, Xiaowei Jia, Zhe Jiang, Aolin Jia, Shuo Xu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i20.30249](https://doi.org/10.1609/aaai.v38i20.30249)

**Abstract**:

Fairness-awareness has emerged as an essential building block for the responsible use of artificial intelligence in real applications. In many cases, inequity in performance is due to the change in distribution over different regions. While techniques have been developed to improve the transferability of fairness, a solution to the problem is not always feasible with no samples from the new regions, which is a bottleneck for pure data-driven attempts. Fortunately, physics-based mechanistic models have been studied for many problems with major social impacts. We propose SimFair, a physics-guided fairness-aware learning framework, which bridges the data limitation by integrating physical-rule-based simulation and inverse modeling into the training design. Using temperature prediction as an example, we demonstrate the effectiveness of the proposed SimFair in fairness preservation.

----

## [2501] I Open at the Close: A Deep Reinforcement Learning Evaluation of Open Streets Initiatives

**Authors**: *R. Teal Witter, Lucas Rosenblatt*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i20.30250](https://doi.org/10.1609/aaai.v38i20.30250)

**Abstract**:

The open streets initiative "opens" streets to pedestrians and bicyclists by closing them to cars and trucks. The initiative, adopted by many cities across North America, increases community space in urban environments. But could open streets also make cities safer and less congested? We study this question by framing the choice of which streets to open as a reinforcement learning problem. In order to simulate the impact of opening streets, we first compare models for predicting vehicle collisions given network and temporal data. We find that a recurrent graph neural network, leveraging the graph structure and the short-term temporal dependence of the data, gives the best predictive performance. Then, with the ability to simulate collisions and traffic, we frame a reinforcement learning problem to find which streets to open. We compare the streets in the open streets initiative to those proposed by a Q-learning algorithm. We find that the streets proposed by the Q-learning algorithm have reliably better outcomes, while streets already selected by the open streets initiative have similar outcomes to randomly selected streets. We present our work as a step toward principally choosing which streets to open for safer and less congested cities.

----

## [2502] HarvestNet: A Dataset for Detecting Smallholder Farming Activity Using Harvest Piles and Remote Sensing

**Authors**: *Jonathan Xu, Amna Elmustafa, Liya Weldegebriel, Emnet Negash, Richard Lee, Chenlin Meng, Stefano Ermon, David B. Lobell*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i20.30251](https://doi.org/10.1609/aaai.v38i20.30251)

**Abstract**:

Small farms contribute to a large share of the productive land in developing countries. In regions such as sub-Saharan Africa, where 80% of farms are small (under 2 ha in size), the task of mapping smallholder cropland is an important part of tracking sustainability measures such as crop productivity. However, the visually diverse and nuanced appearance of small farms has limited the effectiveness of traditional approaches to cropland mapping. Here we introduce a new approach based on the detection of harvest piles characteristic of many smallholder systems throughout the world. We present HarvestNet, a dataset for mapping the presence of farms in the Ethiopian regions of Tigray and Amhara during 2020-2023, collected using expert knowledge and satellite images, totalling 7k hand-labeled images and 2k ground-collected labels. We also benchmark a set of baselines, including SOTA models in remote sensing, with our best models having around 80% classification performance on hand labelled data and 90% and 98% accuracy on ground truth data for Tigray and Amhara, respectively. We also perform a visual comparison with a widely used pre-existing coverage map and show that our model detects an extra 56,621 hectares of cropland in Tigray. We conclude that remote sensing of harvest piles can contribute to more timely and accurate cropland assessments in food insecure regions. The dataset can be accessed through https://figshare.com/s/45a7b45556b90a9a11d2, while the code for the dataset and benchmarks is publicly available at https://github.com/jonxuxu/harvest-piles

----

## [2503] Harnessing Network Effect for Fake News Mitigation: Selecting Debunkers via Self-Imitation Learning

**Authors**: *Xiaofei Xu, Ke Deng, Michael Dann, Xiuzhen Zhang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i20.30252](https://doi.org/10.1609/aaai.v38i20.30252)

**Abstract**:

This study aims to minimize the influence of fake news on social networks by deploying debunkers to propagate true news. This is framed as a reinforcement learning problem, where, at each stage, one user is selected to propagate true news. A challenging issue is episodic reward where the "net" effect of selecting individual debunkers cannot be discerned from the interleaving information propagation on social networks, and only the collective effect from mitigation efforts can be observed. Existing Self-Imitation Learning (SIL) methods have shown promise in learning from episodic rewards, but are ill-suited to the real-world application of fake news mitigation because of their poor sample efficiency. To learn a more effective debunker selection policy for fake news mitigation, this study proposes NAGASIL - Negative sampling and state Augmented Generative Adversarial Self-Imitation Learning, which consists of two improvements geared towards fake news mitigation: learning from negative samples, and an augmented state representation to capture the "real" environment state by integrating the current observed state with the previous state-action pairs from the same campaign. Experiments on two social networks show that NAGASIL yields superior performance to standard GASIL and state-of-the-art fake news mitigation models.

----

## [2504] Spatial-Logic-Aware Weakly Supervised Learning for Flood Mapping on Earth Imagery

**Authors**: *Zelin Xu, Tingsong Xiao, Wenchong He, Yu Wang, Zhe Jiang, Shigang Chen, Yiqun Xie, Xiaowei Jia, Da Yan, Yang Zhou*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i20.30253](https://doi.org/10.1609/aaai.v38i20.30253)

**Abstract**:

Flood mapping on Earth imagery is crucial for disaster management, but its efficacy is hampered by the lack of high-quality training labels. Given high-resolution Earth imagery with coarse and noisy training labels, a base deep neural network model, and a spatial knowledge base with label constraints, our problem is to infer the true high-resolution labels while training neural network parameters. Traditional methods are largely based on specific physical properties and thus fall short of capturing the rich domain constraints expressed by symbolic logic. Neural-symbolic models can capture rich domain knowledge, but existing methods do not address the unique spatial challenges inherent in flood mapping on high-resolution imagery. To fill this gap, we propose a spatial-logic-aware weakly supervised learning framework. Our framework integrates symbolic spatial logic inference into probabilistic learning in a weakly supervised setting. To reduce the time costs of logic inference on vast high-resolution pixels, we propose a multi-resolution spatial reasoning algorithm to infer true labels while training neural network parameters. Evaluations of real-world flood datasets show that our model outperforms several baselines in prediction accuracy. The code is available at https://github.com/spatialdatasciencegroup/SLWSL.

----

## [2505] Unveiling the Tapestry of Automated Essay Scoring: A Comprehensive Investigation of Accuracy, Fairness, and Generalizability

**Authors**: *Kaixun Yang, Mladen Rakovic, Yuyang Li, Quanlong Guan, Dragan Gasevic, Guanliang Chen*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i20.30254](https://doi.org/10.1609/aaai.v38i20.30254)

**Abstract**:

Automatic Essay Scoring (AES) is a well-established educational pursuit that employs machine learning to evaluate student-authored essays. While much effort has been made in this area, current research primarily focuses on either (i) boosting the predictive accuracy of an AES model for a specific prompt (i.e., developing prompt-specific models), which often heavily relies on the use of the labeled data from the same target prompt; or (ii) assessing the applicability of AES models developed on non-target prompts to the intended target prompt (i.e., developing the AES models in a cross-prompt setting). Given the inherent bias in machine learning and its potential impact on marginalized groups, it is imperative to investigate whether such bias exists in current AES methods and, if identified, how it intervenes with an AES model's accuracy and generalizability. Thus, our study aimed to uncover the intricate relationship between an AES model's accuracy, fairness, and generalizability, contributing practical insights for developing effective AES models in real-world education. To this end, we meticulously selected nine prominent AES methods and evaluated their performance using seven distinct metrics on an open-sourced dataset, which contains over 25,000 essays and various demographic information about students such as gender, English language learner status, and economic status. Through extensive evaluations, we demonstrated that: (1) prompt-specific models tend to outperform their cross-prompt counterparts in terms of predictive accuracy; (2) prompt-specific models frequently exhibit a greater bias towards students of different economic statuses compared to cross-prompt models; (3) in the pursuit of generalizability, traditional machine learning models (e.g., SVM) coupled with carefully engineered features hold greater potential for achieving both high accuracy and fairness than complex neural network models.

----

## [2506] Graph Bayesian Optimization for Multiplex Influence Maximization

**Authors**: *Zirui Yuan, Minglai Shao, Zhiqian Chen*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i20.30255](https://doi.org/10.1609/aaai.v38i20.30255)

**Abstract**:

Influence maximization (IM) is the problem of identifying a limited number of initial influential users within a social network to maximize the number of influenced users. However, previous research has mostly focused on individual information propagation, neglecting the simultaneous and interactive dissemination of multiple information items. In reality, when users encounter a piece of information, such as a smartphone product, they often associate it with related products in their minds, such as earphones or computers from the same brand. Additionally, information platforms frequently recommend related content to users, amplifying this cascading effect and leading to multiplex influence diffusion.

This paper first formulates the Multiplex Influence Maximization (Multi-IM) problem using multiplex diffusion models with an information association mechanism. In this problem, the seed set is a combination of influential users and information. To effectively manage the combinatorial complexity, we propose Graph Bayesian Optimization for Multi-IM (GBIM). The multiplex diffusion process is thoroughly investigated using a highly effective global kernelized attention message-passing module. This module, in conjunction with Bayesian linear regression (BLR), produces a scalable surrogate model. A data acquisition module incorporating the exploration-exploitation trade-off is developed to optimize the seed set further.
Extensive experiments on synthetic and real-world datasets have proven our proposed framework effective. The code is available at https://github.com/zirui-yuan/GBIM.

----

## [2507] Fairness-Aware Structured Pruning in Transformers

**Authors**: *Abdelrahman Zayed, Gonçalo Mordido, Samira Shabanian, Ioana Baldini, Sarath Chandar*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i20.30256](https://doi.org/10.1609/aaai.v38i20.30256)

**Abstract**:

The increasing size of large language models (LLMs) has introduced challenges in their training and inference. Removing model components is perceived as a solution to tackle the large model sizes, however, existing pruning methods solely focus on performance, without considering an essential aspect for the responsible use of LLMs: model fairness. It is crucial to address the fairness of LLMs towards diverse groups, such as women, Black people, LGBTQ+, Jewish communities, among others, as they are being deployed and available to a wide audience. In this work, first, we investigate how attention heads impact fairness and performance in pre-trained transformer-based language models. We then propose a novel method to prune the attention heads that negatively impact fairness while retaining the heads critical for performance, i.e. language modeling capabilities. Our approach is practical in terms of time and resources, as it does not require fine-tuning the final pruned, and fairer, model. Our findings demonstrate a reduction in gender bias by 19%, 19.5%, 39.5%, 34.7%, 23%, and 8% for DistilGPT-2, GPT-2, GPT-Neo of two different sizes, GPT-J, and Llama 2 models, respectively, in comparison to the biased model, with only a slight decrease in performance. WARNING: This work uses language that is offensive in nature.

----

## [2508] Estimating On-Road Transportation Carbon Emissions from Open Data of Road Network and Origin-Destination Flow Data

**Authors**: *Jinwei Zeng, Yu Liu, Jingtao Ding, Jian Yuan, Yong Li*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i20.30257](https://doi.org/10.1609/aaai.v38i20.30257)

**Abstract**:

Accounting for over 20% of the total carbon emissions, the precise estimation of on-road transportation carbon emissions is crucial for carbon emission monitoring and efficient mitigation policy formulation. However, existing estimation methods typically depend on hard-to-collect individual statistics of vehicle miles traveled to calculate emissions, thereby suffering from high data collection difficulty. To relieve this issue by utilizing the strong pattern recognition of artificial intelligence, we incorporate two sources of open data representative of the transportation demand and capacity factors, the origin-destination (OD) flow data and the road network data, to build a hierarchical heterogeneous graph learning method for on-road carbon emission estimation (HENCE). Specifically, a hierarchical graph consisting of the road network level, community level, and region level is constructed to model the multi-scale road network-based connectivity and travel connection between spatial areas. Heterogeneous graphs consisting of OD links and spatial links are further built at both the community level and region level to capture the intrinsic interactions between travel demand and road network accessibility. Extensive experiments on two large-scale real-world datasets demonstrate HENCE's effectiveness and superiority with R-squared exceeding 0.75 and outperforming baselines by 9.60% on average, validating its success in pioneering the use of artificial intelligence to empower carbon emission management and sustainability development. The implementation codes are available at this link: https://github.com/tsinghua-fib-lab/HENCE.

----

## [2509] Towards Automatic Boundary Detection for Human-AI Collaborative Hybrid Essay in Education

**Authors**: *Zijie Zeng, Lele Sha, Yuheng Li, Kaixun Yang, Dragan Gasevic, Guanliang Chen*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i20.30258](https://doi.org/10.1609/aaai.v38i20.30258)

**Abstract**:

The recent large language models (LLMs), e.g., ChatGPT, have been able to generate human-like and fluent responses when provided with specific instructions. While admitting the convenience brought by technological advancement, educators also have concerns that students might leverage LLMs to complete their writing assignments and pass them off as their original work. Although many AI content detection studies have been conducted as a result of such concerns, most of these prior studies modeled AI content detection as a classification problem, assuming that a text is either entirely human-written or entirely AI-generated. In this study, we investigated AI content detection in a rarely explored yet realistic setting where the text to be detected is collaboratively written by human and generative LLMs (termed as hybrid text for simplicity). We first formalized the detection task as identifying the transition points between human-written content and AI-generated content from a given hybrid text (boundary detection). We constructed a hybrid essay dataset by partially and randomly removing sentences from the original student-written essays and then instructing ChatGPT to fill in for the incomplete essays. Then we proposed a two-step detection approach where we (1) separated AI-generated content from human-written content during the encoder training process; and (2) calculated the distances between every two adjacent prototypes (a prototype is the mean of a set of consecutive sentences from the hybrid text in the embedding space) and assumed that the boundaries exist between the two adjacent prototypes that have the furthest distance from each other. Through extensive experiments, we observed the following main findings: (1) the proposed approach consistently outperformed the baseline methods across different experiment settings; (2) the encoder training process (i.e., step 1 of the above two-step approach) can significantly boost the performance of the proposed approach; (3) when detecting boundaries for single-boundary hybrid essays, the proposed approach could be enhanced by adopting a relatively large prototype size (i.e., the number of sentences needed to calculate a prototype), leading to a 22% improvement (against the best baseline method) in the In-Domain evaluation and an 18% improvement in the Out-of-Domain evaluation.

----

## [2510] Pre-trained Online Contrastive Learning for Insurance Fraud Detection

**Authors**: *Rui Zhang, Dawei Cheng, Jie Yang, Yi Ouyang, Xian Wu, Yefeng Zheng, Changjun Jiang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i20.30259](https://doi.org/10.1609/aaai.v38i20.30259)

**Abstract**:

Medical insurance fraud has always been a crucial challenge in the field of healthcare industry. Existing fraud detection models mostly focus on offline learning scenes. However, fraud patterns are constantly evolving, making it difficult for models trained on past data to detect newly emerging fraud patterns, posing a severe challenge in medical fraud detection. Moreover, current incremental learning models are mostly designed to address catastrophic forgetting, but often exhibit suboptimal performance in fraud detection. To address this challenge, this paper proposes an innovative online learning method for medical insurance fraud detection, named POCL. This method combines contrastive learning pre-training with online updating strategies. In the pre-training stage, we leverage contrastive learning pre-training to learn on historical data, enabling deep feature learning and obtaining rich risk representations. In the online learning stage, we adopt a Temporal Memory Aware Synapses online updating strategy, allowing the model to perform incremental learning and optimization based on continuously emerging new data. This ensures timely adaptation to fraud patterns and reduces forgetting of past knowledge. Our model undergoes extensive experiments and evaluations on real-world insurance fraud datasets. The results demonstrate our model has significant advantages in accuracy compared to the state-of-the-art baseline methods, while also exhibiting lower running time and space consumption. Our sources are released at https://github.com/finint/POCL.

----

## [2511] UV-SAM: Adapting Segment Anything Model for Urban Village Identification

**Authors**: *Xin Zhang, Yu Liu, Yuming Lin, Qingmin Liao, Yong Li*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i20.30260](https://doi.org/10.1609/aaai.v38i20.30260)

**Abstract**:

Urban villages, defined as informal residential areas in or around urban centers, are characterized by inadequate infrastructures and poor living conditions, closely related to the Sustainable Development Goals (SDGs) on poverty, adequate housing, and sustainable cities. Traditionally, governments heavily depend on field survey methods to monitor the urban villages, which however are time-consuming, labor-intensive, and possibly delayed. Thanks to widely available and timely updated satellite images, recent studies develop computer vision techniques to detect urban villages efficiently. However, existing studies either focus on simple urban village image classification or fail to provide accurate boundary information. To accurately identify urban village boundaries from satellite images, we harness the power of the vision foundation model and adapt the Segment Anything Model (SAM) to urban village segmentation, named UV-SAM. Specifically, UV-SAM first leverages a small-sized semantic segmentation model to produce mixed prompts for urban villages, including mask, bounding box, and image representations, which are then fed into SAM for fine-grained boundary identification. Extensive experimental results on two datasets in China demonstrate that UV-SAM outperforms existing baselines, and identification results over multiple years show that both the number and area of urban villages are decreasing over time, providing deeper insights into the development trends of urban villages and sheds light on the vision foundation models for sustainable cities. The dataset and codes of this study are available at https://github.com/tsinghua-fib-lab/UV-SAM.

----

## [2512] Causally Aware Generative Adversarial Networks for Light Pollution Control

**Authors**: *Yuyao Zhang, Ke Guo, Xiao Zhou*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i20.30261](https://doi.org/10.1609/aaai.v38i20.30261)

**Abstract**:

Artificial light plays an integral role in modern cities, significantly enhancing human productivity and the efficiency of civilization. However, excessive illumination can lead to light pollution, posing non-negligible threats to economic burdens, ecosystems, and human health. Despite its critical importance, the exploration of its causes remains relatively limited within the field of artificial intelligence, leaving an incomplete understanding of the factors contributing to light pollution and sustainable illumination planning distant. To address this gap, we introduce a novel framework named Causally Aware Generative Adversarial Networks (CAGAN). This innovative approach aims to uncover the fundamental drivers of light pollution within cities and offer intelligent solutions for optimal illumination resource allocation in the context of sustainable urban development. We commence by examining light pollution across 33,593 residential areas in seven global metropolises. Our findings reveal substantial influences on light pollution levels from various building types, notably grasslands, commercial centers and residential buildings as significant contributors. These discovered causal relationships are seamlessly integrated into the generative modeling framework, guiding the process of generating light pollution maps for diverse residential areas. Extensive experiments showcase CAGAN’s potential to inform and guide the implementation of effective strategies to mitigate light pollution. Our code and data are publicly available at https://github.com/zhangyuuao/Light_Pollution_CAGAN.

----

## [2513] Multiple-Source Localization from a Single-Snapshot Observation Using Graph Bayesian Optimization

**Authors**: *Zonghan Zhang, Zijian Zhang, Zhiqian Chen*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i20.30262](https://doi.org/10.1609/aaai.v38i20.30262)

**Abstract**:

Due to the significance of its various applications, source localization has garnered considerable attention as one of the most important means to confront diffusion hazards. Multi-source localization from a single-snapshot observation is especially relevant due to its prevalence. However, the inherent complexities of this problem, such as limited information, interactions among sources, and dependence on diffusion models, pose challenges to resolution. Current methods typically utilize heuristics and greedy selection, and they are usually bonded with one diffusion model. Consequently, their effectiveness is constrained.
To address these limitations, we propose a simulation-based method termed BOSouL. Bayesian optimization (BO) is adopted to approximate the results for its sample efficiency. A surrogate function models uncertainty from the limited information. It takes sets of nodes as the input instead of individual nodes. BOSouL can incorporate any diffusion model in the data acquisition process through simulations. Empirical studies demonstrate that its performance is robust across graph structures and diffusion models. The code is available at https://github.com/XGraph-Team/BOSouL.

----

## [2514] Leveraging Opposite Gender Interaction Ratio as a Path towards Fairness in Online Dating Recommendations Based on User Sexual Orientation

**Authors**: *Yuying Zhao, Yu Wang, Yi Zhang, Pamela Wisniewski, Charu C. Aggarwal, Tyler Derr*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i20.30263](https://doi.org/10.1609/aaai.v38i20.30263)

**Abstract**:

Online dating platforms have gained widespread popularity as a means for individuals to seek potential romantic relationships. While recommender systems have been designed to improve the user experience in dating platforms by providing personalized recommendations, increasing concerns about fairness have encouraged the development of fairness-aware recommender systems from various perspectives (e.g., gender and race). However, sexual orientation, which plays a significant role in finding a satisfying relationship, is under-investigated. To fill this crucial gap, we propose a novel metric, Opposite Gender Interaction Ratio (OGIR), as a way to investigate potential unfairness for users with varying preferences towards the opposite gender. We empirically analyze a real online dating dataset and observe existing recommender algorithms could suffer from group unfairness according to OGIR. We further investigate the potential causes for such gaps in recommendation quality, which lead to the challenges of group quantity imbalance and group calibration imbalance. Ultimately, we propose a fair recommender system based on re-weighting and re-ranking strategies to respectively mitigate these associated imbalance challenges. Experimental results demonstrate both strategies improve fairness while their combination achieves the best performance towards maintaining model utility while improving fairness.

----

## [2515] AI-Based Energy Transportation Safety: Pipeline Radial Threat Estimation Using Intelligent Sensing System

**Authors**: *Chengyuan Zhu, Yiyuan Yang, Kaixiang Yang, Haifeng Zhang, Qinmin Yang, C. L. Philip Chen*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i20.30264](https://doi.org/10.1609/aaai.v38i20.30264)

**Abstract**:

The application of artificial intelligence technology has greatly enhanced and fortified the safety of energy pipelines, particularly in safeguarding against external threats. The predominant methods involve the integration of intelligent sensors to detect external vibration, enabling the identification of event types and locations, thereby replacing manual detection methods. However, practical implementation has exposed a limitation in current methods - their constrained ability to accurately discern the spatial dimensions of external signals, which complicates the authentication of threat events. Our research endeavors to overcome the above issues by harnessing deep learning techniques to achieve a more fine-grained recognition and localization process. This refinement is crucial in effectively identifying genuine threats to pipelines, thus enhancing the safety of energy transportation. This paper proposes a radial threat estimation method for energy pipelines based on distributed optical fiber sensing technology. Specifically, we introduce a continuous multi-view and multi-domain feature fusion methodology to extract comprehensive signal features and construct a threat estimation and recognition network. The utilization of collected acoustic signal data is optimized, and the underlying principle is elucidated. Moreover, we incorporate the concept of transfer learning through a pre-trained model, enhancing both recognition accuracy and training efficiency. Empirical evidence gathered from real-world scenarios underscores the efficacy of our method, notably in its substantial reduction of false alarms and remarkable gains in recognition accuracy. More generally, our method exhibits versatility and can be extrapolated to a broader spectrum of recognition tasks and scenarios.

----

## [2516] TAU: Trajectory Data Augmentation with Uncertainty for Next POI Recommendation

**Authors**: *Zhuang Zhuang, Tianxin Wei, Lingbo Liu, Heng Qi, Yanming Shen, Baocai Yin*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i20.30265](https://doi.org/10.1609/aaai.v38i20.30265)

**Abstract**:

Next Point-of-Interest (POI) recommendation has been proven effective at utilizing sparse, intricate spatial-temporal trajectory data to recommend subsequent POIs to users. While existing methods commonly alleviate the problem of data sparsity by integrating spatial-temporal context information, POI category features, and social relationships, they largely overlook the fact that the trajectory sequences collected in the datasets are often incomplete. This oversight limits the model’s potential to fully leverage historical context. In light of this background, we propose Trajectory Data Augmentation with Uncertainty (TAU) for Next POI Recommendation. TAU is a general graph-based trajectory data augmentation method designed to complete user mobility patterns by marrying uncertainty estimation into the next POI recommendation task. More precisely, TAU taps into the global transition pattern graph to identify sets of intermediate nodes located between every pair of locations, effectively
leveraging edge weights as transition probabilities. During trajectory sequence construction, TAU selectively prompts intermediate nodes, chosen based on their likelihood of occurrence as pseudo-labels, to establish comprehensive trajectory sequences. Furthermore, to gauge the certainty and impact of pseudo-labels on the target location, we introduce a novel confidence-aware calibration strategy using evidence deep learning (EDL) for improved performance and reliability. The experimental results clearly indicate that our TAU method achieves consistent performance improvements over existing techniques across two real-world datasets, verifying its effectiveness as the state-of-the-art approach to the task.

----

## [2517] Recommender Ecosystems: A Mechanism Design Perspective on Holistic Modeling and Optimization

**Authors**: *Craig Boutilier, Martin Mladenov, Guy Tennenholtz*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i20.30266](https://doi.org/10.1609/aaai.v38i20.30266)

**Abstract**:

Modern recommender systems lie at the heart of complex recommender ecosystems that couple the behavior of users, content providers, vendors, advertisers, and other actors. Despite this, the focus of much recommender systems research and deployment is on the local, myopic optimization of the recommendations made to individual users. This comes at a significant cost to the long-term utility that recommender systems generate for their users. We argue that modeling the incentives and behaviors of these actors, and the interactions among them induced by the recommender systems, is needed to maximize value and improve overall ecosystem health. Moreover, we propose the use of economic mechanism design, an area largely overlooked in recommender systems research, as a framework for developing such models. That said, one cannot apply “vanilla” mechanism design to recommender ecosystem modeling optimization out of the box—the use of mechanism design raises a number of subtle and interesting research challenges. We outline a number of these in this talk (and paper), emphasizing the need to develop nonstandard approaches to mechanism design that intersect with numerous areas of research, including preference modeling, reinforcement learning and exploration, behavioral economics, and generative AI, among others.

----

## [2518] Model Reprogramming: Resource-Efficient Cross-Domain Machine Learning

**Authors**: *Pin-Yu Chen*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i20.30267](https://doi.org/10.1609/aaai.v38i20.30267)

**Abstract**:

In data-rich domains such as vision, language, and speech, deep learning prevails to deliver high-performance task-specific models and can even learn general task-agnostic representations for efficient finetuning to downstream tasks. However, deep learning in resource-limited domains still faces multiple challenges including (i) limited data, (ii) constrained model development cost, and (iii) lack of adequate pre-trained models for effective finetuning. This paper provides an overview of model reprogramming to bridge this gap. Model reprogramming enables resource-efficient cross-domain machine learning by repurposing and reusing a well-developed pre-trained model from a source domain to solve tasks in a target domain without model finetuning, where the source and target domains can be vastly different. In many applications, model reprogramming outperforms transfer learning and training from scratch. This paper elucidates the methodology of model reprogramming, summarizes existing use cases, provides a theoretical explanation of the success of model reprogramming, and concludes with a discussion on open-ended research questions and opportunities.

----

## [2519] Conversational Modeling for Constraint Satisfaction

**Authors**: *Eugene C. Freuder*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i20.30268](https://doi.org/10.1609/aaai.v38i20.30268)

**Abstract**:

Many problems, from Sudoku to factory scheduling, can be regarded as constraint satisfaction problems. A key component of real world problem solving is a conversation between a constraint programming expert and a problem domain expert to specify the problem to be solved. This presentation argues that the time is ripe for progress in automating the constraint programmer side of this conversation and suggests promising avenues for this pursuit.

----

## [2520] Integrated Systems for Computational Scientific Discovery

**Authors**: *Pat Langley*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i20.30269](https://doi.org/10.1609/aaai.v38i20.30269)

**Abstract**:

This paper poses the challenge of developing and evaluating integrated
systems for computational scientific discovery. We note some distinguishing
characteristics of discovery tasks, examine eight component abilities,
review previous successes at partial integration, and consider hurdles
the AI research community must leap to transform the vision for
integrated discovery into reality. In closing, we discuss promising
scientific domains in which to test such computational artifacts.

----

## [2521] Towards a More Burkean Approach to Computational Social Choice

**Authors**: *Omer Lev*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i20.30270](https://doi.org/10.1609/aaai.v38i20.30270)

**Abstract**:

In the last few years, a lot of the activity of the computational social choice community has focused on novel mechanisms for reaching decisions by large groups of people. While this research makes meaningful scientific contributions, many of these mechanisms are not quite useful in realistic decision-making settings. Moreover, their radicalism ignores the centuries-old experience we have with large-scale human decision-making, and what it teaches us about what works. We believe it is important the community engage with mechanisms which are widely-used in the real world, as they may hold a key to a deeper understanding of how people reach decisions and the way that helps them do that productively. Moreover, letting the community bring its analysis and understanding to these will allow for algorithmic suggestions that have some chance of being implemented (and, thus, can contribute to the public debate on these topics). In particular, we highlight the relatively less-investigated role of parties and grouping of voters and candidates, and the role of executive capacity in analyzing decision-making structures.

----

## [2522] Regeneration Learning: A Learning Paradigm for Data Generation

**Authors**: *Xu Tan, Tao Qin, Jiang Bian, Tie-Yan Liu, Yoshua Bengio*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i20.30271](https://doi.org/10.1609/aaai.v38i20.30271)

**Abstract**:

Machine learning methods for conditional data generation usually build a mapping from source conditional data X to target data Y. The target Y (e.g., text, speech, music, image, video) is usually high-dimensional and complex, and contains information that does not exist in source data, which hinders effective and efficient learning on the source-target mapping. In this paper, we present a learning paradigm called regeneration learning for data generation, which first generates Y' (an abstraction/representation of Y) from X and then generates Y from Y'. During training, Y' is obtained from Y through either handcrafted rules or self-supervised learning and is used to learn X-->Y' and Y'-->Y. Regeneration learning extends the concept of representation learning to data generation tasks, and can be regarded as a counterpart of traditional representation learning, since 1) regeneration learning handles the abstraction (Y') of the target data Y for data generation while traditional representation learning handles the abstraction (X') of source data X for data understanding; 2) both the processes of Y'-->Y in regeneration learning and X-->X' in representation learning can be learned in a self-supervised way (e.g., pre-training); 3) both the mappings from X to Y' in regeneration learning and from X' to Y in representation learning are simpler than the direct mapping from X to Y. We show that regeneration learning can be a widely-used paradigm for data generation (e.g., text generation, speech recognition, speech synthesis, music composition, image generation, and video generation) and can provide valuable insights into developing data generation methods.

----

## [2523] The Fairness Fair: Bringing Human Perception into Collective Decision-Making

**Authors**: *Hadi Hosseini*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i20.30272](https://doi.org/10.1609/aaai.v38i20.30272)

**Abstract**:

Fairness is one of the most desirable societal principles in collective decision-making. It has been extensively studied in the past decades for its axiomatic properties and has received substantial attention from the multiagent systems community in recent years for its theoretical and computational aspects in algorithmic decision-making. However, these studies are often not sufficiently rich to capture the intricacies of human perception of fairness in the ambivalent nature of the real-world problems. We argue that not only fair solutions should be deemed desirable by social planners (designers), but they should be governed by human and societal cognition, consider perceived outcomes based on human judgement, and be verifiable. We discuss how achieving this goal requires a broad transdisciplinary approach ranging from computing and AI to behavioral economics and human-AI interaction. In doing so, we identify shortcomings and long-term challenges of the current literature of fair division, describe recent efforts in addressing them, and more importantly, highlight a series of open research directions.

----

## [2524] Temporal Fairness in Multiwinner Voting

**Authors**: *Edith Elkind, Svetlana Obraztsova, Nicholas Teh*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i20.30273](https://doi.org/10.1609/aaai.v38i20.30273)

**Abstract**:

Multiwinner voting captures a wide variety of settings, from parliamentary elections in democratic systems to product placement in online shopping platforms. There is a large body of work dealing with axiomatic characterizations, computational complexity, and algorithmic analysis of multiwinner voting rules. Although many challenges remain, significant progress has been made in showing existence of fair and representative outcomes as well as efficient algorithmic solutions for many commonly studied settings. However, much of this work focuses on single-shot elections, even though in numerous real-world settings elections are held periodically and repeatedly. Hence, it is imperative to extend the study of multiwinner voting to temporal settings. Recently, there have been several efforts to address this challenge. However, these works are difficult to compare, as they model multi-period voting in very different ways. We propose a unified framework for studying temporal fairness in this domain, drawing connections with various existing bodies of work, and consolidating them within a general framework. We also identify gaps in existing literature, outline multiple opportunities for future work, and put forward a vision for the future of multiwinner voting in temporal settings.

----

## [2525] Mixed Fair Division: A Survey

**Authors**: *Shengxin Liu, Xinhang Lu, Mashbat Suzuki, Toby Walsh*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i20.30274](https://doi.org/10.1609/aaai.v38i20.30274)

**Abstract**:

The fair allocation of resources to agents is a fundamental problem in society and has received significant attention and rapid developments from the game theory and artificial intelligence communities in recent years. The majority of the fair division literature can be divided along at least two orthogonal directions: goods versus chores, and divisible versus indivisible resources. In this survey, besides describing the state of the art, we outline a number of interesting open questions in three mixed fair division settings: (i) indivisible goods and chores, (ii) divisible and indivisible goods (i.e., mixed goods), and (iii) fair division of indivisible goods with subsidy.

----

## [2526] Adventures of Trustworthy Vision-Language Models: A Survey

**Authors**: *Mayank Vatsa, Anubhooti Jain, Richa Singh*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i20.30275](https://doi.org/10.1609/aaai.v38i20.30275)

**Abstract**:

Recently, transformers have become incredibly popular in computer vision and vision-language tasks. This notable rise in their usage can be primarily attributed to the capabilities offered by attention mechanisms and the outstanding ability of transformers to adapt and apply themselves to a variety of tasks and domains. Their versatility and state-of-the-art performance have established them as indispensable tools for a wide array of applications. However, in the constantly changing landscape of machine learning, the assurance of the trustworthiness of transformers holds utmost importance. This paper conducts a thorough examination of vision-language transformers, employing three fundamental principles of responsible AI: Bias, Robustness, and Interpretability. The primary objective of this paper is to delve into the intricacies and complexities associated with the practical use of transformers, with the overarching goal of advancing our comprehension of how to enhance their reliability and accountability.

----

## [2527] Interactive Theorem Provers: Applications in AI, Opportunities, and Challenges

**Authors**: *Mohammad Abdulaziz*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i20.30276](https://doi.org/10.1609/aaai.v38i20.30276)

**Abstract**:

Interactive theorem provers (ITPs) are computer programs in which axioms and a conjecture are stated in a formal language, and a user provides the ITP with relatively high-level steps of a formal proof for the conjecture. Then, by invoking automated theorem provers, the ITP tries to generate low-level steps that fill the gaps between the steps provided by the user, thus forming a complete formal proof of the conjecture. The ITP also checks the entire formal proof against the axioms, thus confirming the soundness of all derivations in the formal proof.

In this talk, I will discuss the existing opportunities and potential benefits to applying ITPs to reason about and verify AI concepts, algorithms, and software. I will also discuss the challenges we have to being able to apply ITPs in AI and reap those benefits. I will do so by discussing a number of my previous projects on the application of ITPs to different AI concepts, algorithms, and software systems. These  projects span different areas of planning (classical planning, temporal planning, and planning under uncertainty) as well as algorithms with applications in algorithmic game theory, like general graph matching and online matching.

----

## [2528] Symbolic Reasoning Methods for AI Planning

**Authors**: *Gregor Behnke*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i20.30277](https://doi.org/10.1609/aaai.v38i20.30277)

**Abstract**:

Planning is the act of deliberative thinking before acting.
It is based on a symbolic model of the world and the options to act in it, usually defined in function-free first-order logic.
The task is to find a sequence of actions (a plan) that leads from a given current state to a desired goal state.
The basic, purely physical description may be augmented with a partially ordered grammar-like structure (a Hierarchical Task Network or HTN), which can describe expert knowledge, or practical, legal, or operational requirements.


In this talk, I will survey a variety of methods for automatically deriving plans using symbolic methods for planning -- from both my past and future research.
These symbolic methods -- in some sense -- translate planning problems into other, simpler symbolic representations and reason over them to find plans.


As a basis for these methods, I will firstly introduce relevant theoretical results on planning.
First, I will discuss the expressive power of planning formalisms (ECAI'14, ICAPS'16) and second, the computational complexity of HTN planning and related tasks such as HTN plan verification, plan modification, and plan recognition (ICAPS'15, ICAPS'16).


Based on these theoretical results, I will develop why SAT-based HTN planning is possible and how it can be implemented.
To this end, I will survey several of my publications at top-tier conferences, including papers at ICAPS'17, AAAI'18, AAAI'19, IJCAI'19, AAAI'20, and ICAPS'21 -- in which I developed an highly SAT-based planner for HTN problems including the ability to find optimal plans as well as the grounding as a preprocessing step.
Here I will also give an outlook on future developments and new ideas that I propose for SAT-based planning -- including the exploitation of structures in plan (e.g.\ landmarks or operator-counting constraints).

Next, I will present the idea of expressing lifted classical planning as SAT (ICAPS'22).
The resulting planner LiSAT was the first lifted SAT-based planner -- and proved highly efficient and outperformed all other lifted planners at the time of publication.
Notably, LiSAT was the first planner (lifted or grounded) and still is the only one to solve the challenging OrganicSynthesis benchmark -- and could even prove optimality for all plans.
I will also outline future ideas to further improve the efficiency of LiSAT.


Lastly, I introduce the notion of planning with symbolic symbolic representations (AAAI'21 and ICAPS'23).
Here one uses Binary Decision Diagrams to encode large sets of states efficiently.
For expressing the additional structure encoded by HTNs, I show how BDDs can be suitably integrated into finite automata.
Based on this representation, an efficient and optimal planning algorithm can be derived.
Additionally, I show how this algorithm can be extended to also cover oversubscription planning.

----

## [2529] Demystifying Algorithmic Fairness in an Uncertain World

**Authors**: *Lu Cheng*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i20.30278](https://doi.org/10.1609/aaai.v38i20.30278)

**Abstract**:

Significant progress in the field of fair machine learning (ML) has been made to counteract algorithmic discrimination against marginalized groups. However, fairness remains an active research area that is far from settled. One key bottleneck is the implicit assumption that environments, where ML is developed and deployed, are certain and reliable. In a world that is characterized by volatility, uncertainty, complexity, and ambiguity, whether what has been developed in algorithmic fairness can still serve its purpose is far from obvious. In this talk, I will first discuss how to improve algorithmic fairness under two kinds of predictive uncertainties, i.e., aleatoric uncertainty (i.e., randomness and ambiguity in the data) and epistemic uncertainty (i.e., a lack of data or knowledge), respectively. The former regards historical bias reflected in the data and the latter corresponds to the bias perpetuated or amplified during model training due to lack of data or knowledge. In particular, the first work studies pushing the fairness-utility trade-off through aleatoric uncertainty, and the second work investigates fair few-shot learning. The last work introduces coverage-based fairness that ensures different groups enjoy identical treatment and receive equal coverage.

----

## [2530] Data-Efficient Graph Learning

**Authors**: *Kaize Ding*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i20.30279](https://doi.org/10.1609/aaai.v38i20.30279)

**Abstract**:

My research strives to develop fundamental graph-centric learning algorithms to reduce the need for human supervision in low-resource scenarios. The focus is on achieving effective and reliable data-efficient learning on graphs, which can be summarized into three facets: (1) graph weakly-supervised learning; (2) graph few-shot learning; and (3) graph self-supervised learning.

----

## [2531] Making Natural Language Reasoning Explainable and Faithful

**Authors**: *Xinya Du*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i20.30280](https://doi.org/10.1609/aaai.v38i20.30280)

**Abstract**:

Neural models, including large language models (LLMs), achieve superior performance on logical reasoning tasks such as question answering. To elicit reasoning capabilities from LLMs, recent works propose using the chain-of-thought (CoT) mechanism to generate both the reasoning chain and the answer, which enhances the model’s capabilities in conducting reasoning. However, due to LLM’s uninterpretable nature and the extreme flexibility of free-form explanations, several challenges remain: such as struggling with inaccurate reasoning, hallucinations, and not aligning with human preferences. In this talk, we will focus on (1) our design of leveraging structured information (that is grounded to the context), for the explainable complex question answering and reasoning; (2) our multi-module interpretable framework for inductive reasoning, which conducts step-wise faithful reasoning with iterative feedback.

----

## [2532] Towards Robust Visual Understanding: from Recognition to Reasoning

**Authors**: *Tejas Gokhale*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i20.30281](https://doi.org/10.1609/aaai.v38i20.30281)

**Abstract**:

Models that learn from data are widely and rapidly being deployed today for real-world use, but they suffer from unforeseen failures due to distribution shift, adversarial attacks, noise and corruption, and data scarcity.  But many failures also occur because many modern AI tasks require reasoning beyond pattern matching -- and such reasoning abilities are difficult to formulate as data-based input-output function fitting.  The reliability problem has become increasingly important under the new paradigm of semantic ``multimodal'' learning.  My research provides avenues to develop robust and reliable computer vision systems, particularly by leveraging the interactions between vision and language. In this AAAI New Faculty highlights talk, I will cover three thematic areas of my research, ranging from robustness in computer vision, open-domain reliability in visual reasoning, and challenges and opportunities in evaluation of generative models. Readers are encouraged to refer to my website (www.tejasgokhale.com) for more details and updates from my lab's activities towards the goal of robust visual understanding.

----

## [2533] Continual Learning in an Open and Dynamic World

**Authors**: *Yunhui Guo*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i20.30282](https://doi.org/10.1609/aaai.v38i20.30282)

**Abstract**:

Building autonomous agents that can process massive amounts of real-time sensor-captured data is essential for many real-world applications including autonomous vehicles, robotics and AI in medicine. As the agent often needs to explore in a dynamic environment, it is thus a desirable as well as challenging goal to enable the agent to learn over time without performance degradation. Continual learning aims to build a continual learner which can learn new concepts over the data stream while preserving previously learnt concepts. In the talk, I will survey three pieces of my recent research on continual learning (i) supervised continual learning, (ii) unsupervised continual learning, and (iii) multi-modal continual learning. In the first work, I will discuss a supervised
continual learning algorithm called MEGA which dynamically balances the old tasks and the new task. In the second work, I will discuss unsupervised continual learning algorithms which learn representation continually without access to the labels. In the third work, I will elaborate an efficient continual learning algorithm that can learn multiple modalities continually without forgetting.

----

## [2534] Scaling Offline Evaluation of Reinforcement Learning Agents through Abstraction

**Authors**: *Josiah P. Hanna*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i20.30283](https://doi.org/10.1609/aaai.v38i20.30283)

**Abstract**:

A critical challenge for the widescale adoption of reinforcement learning (RL) is the need to give domain experts assurance that learned policies will improve decision-making -- and not lead to unacceptable behavior. To meet this challenge, my work aims to develop new methods for offline policy evaluation in real world RL domains. There has been much recent interest in offline evaluation and many advances. However, recent benchmarking efforts have also shown that there remains a substantial gap between current state-of-the-art methods and real world domains such as robotics. Towards scalable offline evaluation, my group is investigating the use of methods for abstraction and representation learning. In this new faculty highlight, I will present our recent results that show the promise of this direction for scaling offline evaluation in RL domains. I will then describe future directions in this line of that work which will further realize the promise of offline policy evaluation for increasing confidence in deployed RL.

----

## [2535] Collaborative Learning across Heterogeneous Systems with Pre-Trained Models

**Authors**: *Trong Nghia Hoang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i20.30284](https://doi.org/10.1609/aaai.v38i20.30284)

**Abstract**:

The increasingly decentralized and private nature of data in our digital society has  motivated the development of personalized, collaborative intelligent systems that enable knowledge aggregation across multiple data owners while accommodating for their data privacy and system constraints. However, collaborative learning has only been investigated in simple and limited settings: isolated task scenarios where learning begins from scratch and does not build on prior expertise; learned model is represented in task-specific forms which are not generalizable to unseen, emerging scenarios; and more often, a universal model representation is assumed across collaborators, ignoring their local compute constraints or input representations. This restricts its practicality in continual learning scenarios with limited task data, which demand continuous adaptation and knowledge transfer across different information silos, tasks, and learning models, as well as the utilization of prior solution expertises. To overcome these limitations, my research has been focused on developing effective and scalable resource-aware collaborative learning frameworks across heterogeneous systems.

----

## [2536] Understanding Surprising Generalization Phenomena in Deep Learning

**Authors**: *Wei Hu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i20.30285](https://doi.org/10.1609/aaai.v38i20.30285)

**Abstract**:

Deep learning has exhibited a number of surprising generalization phenomena that are not captured by classical statistical learning theory. This talk will survey some of my work on the theoretical characterizations of several such intriguing phenomena: (1) Implicit regularization: A major mystery in deep learning is that deep neural networks can often generalize well despite their excessive expressive capacity. Towards explaining this mystery, it has been suggested that commonly used gradient-based optimization algorithms enforce certain implicit regularization which effectively constrains the model capacity. (2) Benign overfitting: In certain scenarios, a model can perfectly fit noisily labeled training data, but still archives near-optimal test error at the same time, which is very different from the classical notion of overfitting. (3) Grokking: In certain scenarios, a model initially achieves perfect training accuracy but no generalization (i.e. no better than a random predictor), and upon further training, transitions to almost perfect generalization. Theoretically establishing these properties often involves making appropriate high-dimensional assumptions on the problem as well as a careful analysis of the training dynamics.

----

## [2537] Fostering Trustworthiness in Machine Learning Algorithms

**Authors**: *Mengdi Huai*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i20.30286](https://doi.org/10.1609/aaai.v38i20.30286)

**Abstract**:

Recent years have seen a surge in research that develops and applies machine learning algorithms to create intelligent learning systems. However, traditional machine learning algorithms have primarily focused on optimizing accuracy and efficiency, and they often fail to consider how to foster trustworthiness in their design. As a result, machine learning models usually face a trust crisis in real-world applications. Driven by these urgent concerns about trustworthiness, in this talk, I will introduce my research efforts towards the goal of making machine learning trustworthy. Specifically, I will delve into the following key research topics: security vulnerabilities and robustness, model explanations, and privacy-preserving mechanisms.

----

## [2538] Deep Learning on Graphs: A Data-Centric Exploration

**Authors**: *Wei Jin*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i20.30287](https://doi.org/10.1609/aaai.v38i20.30287)

**Abstract**:

Many learning tasks in Artificial Intelligence (AI) require dealing with graph data, ranging from biology and chemistry to finance and education. As powerful deep learning tools for graphs,  graph neural networks (GNNs) have demonstrated remarkable performance in various graph-related applications.  Despite the significant accomplishments of GNNs, recent studies have highlighted that their efficiency and effectiveness face significant challenges such as adversarial robustness and scalability, which are fundamentally linked to data. While major attention has been devoted to improving GNNs from the model perspective, the potential of directly enhancing data has often been overlooked.  It underscores a critical gap in GNN research---while model improvements are undoubtedly important, we also need to recognize and address the data-related factors contributing to the challenges. Hence, my research is to investigate solutions for these challenges from the data perspective, employing strategies such as data characterization, reduction, augmentation, transformation, and detection.

----

## [2539] Quantifying Political Polarization through the Lens of Machine Translation and Vicarious Offense

**Authors**: *Ashiqur R. KhudaBukhsh*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i20.30288](https://doi.org/10.1609/aaai.v38i20.30288)

**Abstract**:

This talk surveys three related research contributions that shed light on the current US political divide: 

1. a novel machine-translation-based framework to quantify political polarization; 
2. an analysis of disparate media portrayal of US policing in major cable news outlets; and 
3. a novel perspective of vicarious offense that examines a timely and important question --  how well do Democratic-leaning users perceive what content would be deemed as offensive by their Republican-leaning counterparts or vice-versa?

----

## [2540] Learning Representations for Robust Human-Robot Interaction

**Authors**: *Yen-Ling Kuo*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i20.30289](https://doi.org/10.1609/aaai.v38i20.30289)

**Abstract**:

For robots to robustly and flexibly interact with humans, they need to acquire skills to use across scenarios. One way to enable the generalization of skills is to learn representations that are useful for downstream tasks. Learning a representation for interactions requires an understanding of what (e.g., objects) as well as how (e.g., actions, controls, and manners) to interact with. However, most existing language or visual representations mainly focus on objects. To enable robust human-robot interactions, we need a representation that is not just grounded at the object level but to reason at the action level. The ability to reason about an agent’s own actions and other’s actions will be crucial for long-tail interactions. My research focuses on leveraging the compositional nature of language and reward functions to learn representations that generalize to novel scenarios. Together with the information from multiple modalities, the learned representation can reason about task progress, future behaviors, and the goals/beliefs of an agent. The above ideas have been demonstrated in my research on building robots to understand language and engage in social interactions.

----

## [2541] The Role of Over-Parameterization in Machine Learning - the Good, the Bad, the Ugly

**Authors**: *Fanghui Liu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i20.30290](https://doi.org/10.1609/aaai.v38i20.30290)

**Abstract**:

The conventional wisdom of simple models in machine learning misses the bigger picture, especially over-parameterized neural networks (NNs), where the number of parameters are much larger than the number of training data. Our goal is to explore the mystery behind over-parameterized models from a theoretical side.

In this talk, I will discuss the role of over-parameterization in neural networks, to theoretically understand why they can perform well. First, I will discuss the role of over-parameterization in neural networks from the perspective of models, to theoretically understand why they can genralize well. Second, the effects of over-parameterization in robustness, privacy are discussed. Third, I will talk about the over-parameterization from kernel methods to neural networks in a function space theory view.  Besides, from classical statistical learning to sequential decision making, I will talk about the benefits of over-parameterization on how deep reinforcement learning works well for function approximation. Potential future directions on theory of over-parameterization ML will also be discussed.

----

## [2542] Algorithmic Foundation of Federated Learning with Sequential Data

**Authors**: *Mingrui Liu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i20.30291](https://doi.org/10.1609/aaai.v38i20.30291)

**Abstract**:

The current analysis of federated optimization algorithms for training deep neural networks assumes that the data is non-sequential (e.g., images), which incurs a smooth loss objective. In contrast, edge devices generate lots of sequential data every day, where these sequences exhibit significant sequential correlation at different time stamps (e.g., text messages). In order to learn from such sequential data, people typically use a class of neural networks that is inherently nonsmooth, with a potentially unbounded smoothness parameter. Examples include recurrent neural networks, long-short-term memory networks, and transformers. It remains unclear how to design provably efficient algorithms for training these neural networks to learn from sequential data. My goal is to lay the algorithmic foundation of federated learning with sequential data, which contributes novel algorithms for learning from a range of real-world sequential data (e.g., natural language, electronic health record, transportation, time series, etc.) using state-of-the-art deep neural networks.


In this talk, I will first motivate the problem by showing that the transformer, which is widely used for sequential data learning, has an unbounded smooth landscape. Then, I will introduce provably efficient federated deep learning algorithms in the presence of unbounded smoothness. In particular, I will introduce a few efficient algorithms for various settings of federated learning, including homogeneous data, heterogeneous data, and partial client participation. The main result is twofold. First, we show that the designed algorithms provably small computational and communication complexities. Second, we establish fundamental hardness results in the unbounded smoothness setting. Ultimately, I will discuss the future challenges of extending our research framework from small-scale neural networks to large language models.

----

## [2543] When Causal Inference Meets Graph Machine Learning

**Authors**: *Jing Ma*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i20.30292](https://doi.org/10.1609/aaai.v38i20.30292)

**Abstract**:

Graphs (i.e., networks) are ubiquitous in daily life, as they can effectively model a plethora of real-world systems with connected units, such as social networks and biological networks. Recent years have witnessed rapid development in graph-based machine learning (GML) in various high-impact domains. Currently, the mainstream GML methods are based on statistical learning, e.g., utilizing the statistical correlations between node features, graph structure, and labels for node classification. However, statistical learning has been widely criticized for only capturing the superficial relations between variables in the data system, and consequently, rendering the lack of trustworthiness in real-world applications. Therefore, it is crucial to understand the causality in the data system and the learning process. Causal inference is the discipline that investigates the causality inside a system, for example, to identify and estimate the causal effect of a certain treatment (e.g., wearing a face mask) on an important outcome (e.g., COVID-19 infection). Involving the concepts and philosophy of causal inference in ML methods is often considered significant for human-level intelligence and can serve as the foundation of artificial intelligence (AI). However, most traditional causal inference studies rely on strong assumptions, and focus on independent and identically distributed (i.i.d.) data, while causal inference on graphs is faced with many barriers. Therefore, we aim to bridge the gap between causal inference and GML.

----

## [2544] Towards Holistic, Pragmatic and Multimodal Conversational Systems

**Authors**: *Pranava Madhyastha*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i20.30293](https://doi.org/10.1609/aaai.v38i20.30293)

**Abstract**:

Language acquisition and utilization transcend the mere exchange of lexical units. Visual cues, prosody, gestures, body movements, and context play an undeniably crucial role. Humans naturally communicate multimodally, employing multiple channels and synthesizing information from diverse modalities. My research delves into the characterization and construction of multimodal models that seamlessly integrate data from multiple independent modalities. I will cover recent work that highlights the challenges, achievements, and opportunities towards developing capable multimodal discursive models.

----

## [2545] From Statistical Relational to Neuro-Symbolic Artificial Intelligence

**Authors**: *Giuseppe Marra*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i20.30294](https://doi.org/10.1609/aaai.v38i20.30294)

**Abstract**:

The integration of learning and reasoning is one of the key challenges in artificial intelligence and machine learning today. The area of Neuro-Symbolic AI (NeSy) tackles this challenge by integrating symbolic reasoning with neural networks. In our recent work, we provided an introduction to NeSy by drawing several parallels to another field that has a rich tradition in integrating learning and reasoning, namely Statistical Relational Artificial Intelligence (StarAI).

----

## [2546] Harmonious Mobility for Robots that Work with and around People

**Authors**: *Christoforos I. Mavrogiannis*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i20.30295](https://doi.org/10.1609/aaai.v38i20.30295)

**Abstract**:

The integration of advances from machine learning and computer vision with the classical autonomy stack has brought successful robot deployments in fulfilment, manufacturing, and transportation. However, unstructured and dynamic environments such as pedestrian spaces and streets, workplaces, and homes pose additional challenges such as modeling human behavior, understanding user perceptions, and ensuring human safety and comfort. My work addresses such challenges to enable robots to fluently work with and around people to increase productivity and assist users.

----

## [2547] Recent Advancements in Inverse Reinforcement Learning

**Authors**: *Alberto Maria Metelli*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i20.30296](https://doi.org/10.1609/aaai.v38i20.30296)

**Abstract**:

Inverse reinforcement learning (IRL) has seen significant advancements in recent years. This class of approaches aims to efficiently learn the underlying reward function that rationalizes the behavior exhibited by expert agents, often represented by humans. In contrast to mere behavioral cloning, the reconstruction of a reward function yields appealing implications, as it allows for more effective interpretability of the expert’s decisions and provides a transferable specification of the expert’s objectives for application in even different environments. Unlike the well-understood field of reinforcement learning (RL) from a theoretical perspective, IRL still grapples with limited understanding, significantly constraining its applicability. A fundamental challenge in IRL is the inherent ambiguity in selecting a reward function, given the existence of multiple candidate functions, all explaining the expert’s behavior.

In this talk, I will survey three of my papers that have made notable contributions to the IRL field: “Provably Efficient Learning of Transferable Rewards”, “Towards Theoretical Understanding of Inverse Reinforcement Learning”, and “Inverse Reinforcement Learning with Sub-optimal Experts".

The central innovation introduced by the first paper is a novel formulation of the IRL problem that overcomes the issue of ambiguity. IRL is reframed as the problem of learning the feasible reward set, which is the set of all rewards that can explain the expert’s behavior. This approach postpones the selection of the reward function, thereby circumventing the ambiguity issues. Furthermore, the feasible reward set exhibits convenient geometric properties that enable the development of efficient algorithms for its computation. 

Building on this novel formulation of IRL, the second paper addresses the problem of efficiently learning the feasible reward set when the environment and the expert’s policy are not known in advance. It introduces a novel way to assess the dissimilarity between feasible reward sets based on the Hausdorff distance and presents a new PAC (probabilistic approximately correct) framework. The most significant contribution of this paper is the introduction of the first sample complexity lower bound, which highlights the challenges inherent in the IRL problem. Deriving this lower bound necessitated the development of novel technical tools. The paper also demonstrates that when a generative model of the environment is available, a uniform sampling strategy achieves a sample complexity that matches the lower bound, up to logarithmic factors.

Finally, in the third paper, the IRL problem in the presence of sub-optimal experts is investigated. Specifically, the paper assumes the availability of multiple sub-optimal experts, in addition to the expert agent, which provides additional demonstrations, associated with a known quantification of the maximum amount of sub-optimality. The paper shows that this richer information mitigates the ambiguity problem, significantly reducing the size of the feasible reward set while retaining its favorable geometric properties. Furthermore, the paper explores the associated statistical problem and derives novel lower bounds for sample complexity, along with almost matching algorithms. These selected papers represent notable advancements in IRL, contributing to the establishment of a solid theoretical foundation for IRL and extending the framework to accommodate scenarios with sub-optimal experts.

----

## [2548] Exploiting Data Geometry in Machine Learning

**Authors**: *Melanie Weber*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i20.30297](https://doi.org/10.1609/aaai.v38i20.30297)

**Abstract**:

A key challenge in Machine Learning (ML) is the identification of geometric structure in high-dimensional data. Most algorithms assume that data lives in a high-dimensional vector space; however, many applications involve non-Euclidean data, such as graphs, strings and matrices, or data whose structure is determined by symmetries in the underlying system. Here, we discuss methods for identifying geometric structure in data and how leveraging data geometry can give rise to efficient ML algorithms with provable guarantees.

----

## [2549] Towards Trustworthy Deep Learning

**Authors**: *Tsui-Wei Lily Weng*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i20.30298](https://doi.org/10.1609/aaai.v38i20.30298)

**Abstract**:

Deep neural networks (DNNs) have achieved unprecedented success across many scientific and engineering fields in the last decades. Despite its empirical success, unfortunately, recent studies have shown that there are various failure modes and blindspots in DNN models which may result in unexpected serious failures and potential harms, e.g. the existence of adversarial examples and small perturbations. This is not acceptable especially for safety critical and high stakes applications in the real-world, including healthcare, self-driving cars, aircraft control systems, hiring and malware detection protocols. Moreover, it has been challenging to understand why and when DNNs will fail due to their complicated structures and black-box behaviors. Lacking interpretability is one critical issue that may seriously hinder the deployment of DNNs in high-stake applications, which need interpretability to trust the prediction, to understand potential failures, and to be able to mitigate harms and eliminate biases in the model.


To make DNNs trustworthy and reliable for deployment, it is necessary and urgent to develop methods and tools that can (i) quantify and improve their robustness against adversarial and natural perturbations, and (ii) understand their underlying behaviors and further correct errors to prevent injuries and damages. These are the important first steps to enable Trustworthy AI and Trustworthy Machine Learning. In this talk, I will survey a series of research efforts in my lab contributed to tackling the grand challenges in (i) and (ii). In the first part of my talk, I will overview our research effort in Robust Machine Learning since 2017, where we have proposed the first attack-agnostic robustness evaluation metric, the first efficient robustness certification algorithms for various types of perturbations, and efficient robust learning algorithms across supervised learning to deep reinforcement learning. 


In the second part of my talk, I will survey a series of exciting results in my lab on accelerating interpretable machine learning and explainable AI. Specifically, I will show how we could bring interpretability into deep learning by leveraging recent advances in multi-modal models. I'll present recent works in our group on automatically dissecting neural networks with open vocabulary concepts, designing interpretable neural networks without concept labels, and briefly overview our recent efforts on demystifying black-box DNN training process, automated neuron explanations for Large Language Models and the first robustness evaluation of a family of neuron-level interpretation techniques.

----

## [2550] Towards Reliable Learning in the Wild: Generalization and Adaptation

**Authors**: *Huaxiu Yao*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i20.30299](https://doi.org/10.1609/aaai.v38i20.30299)

**Abstract**:

The real-world deployment of machine learning algorithms often poses challenges due to shifts in data distributions and tasks. These shifts can lead to a degradation in model performance, as the model may not have encountered such changes during training. Additionally, they can make it difficult for the model to generalize to new scenarios and can result in poor performance in real-world applications. In this talk, I will present our research on building machine learning models that are highly generalizable and easily adaptable to different shifts. Specifically, I will first discuss our approach to improving out-of-distribution robustness and mitigating spurious correlations by training environment-invariant models through selective augmentation and post-hoc rectification. Second, I will present our techniques for continuous and rapid adaptation of models to new tasks and environments. This includes methods to facilitate compositional generalization and adaptation by extracting relationships from historical observations and to enhance reliable adaptation even in the face of imperfect observations. Additionally, I will showcase our successful practices for addressing shifts in real-world applications, such as in the healthcare, e-commerce, and transportation industries. The talk will also touch upon the remaining challenges and outline future research directions in this area.

----

## [2551] Towards Human-like Learning from Relational Structured Data

**Authors**: *Quanming Yao*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i20.30300](https://doi.org/10.1609/aaai.v38i20.30300)

**Abstract**:

Relational structured data is a way of representing knowledge using nodes and edges, while also capturing the meaning of that knowledge in a structured form that can be used for machine learning. Compared with vision and natural language data, relational structured data represents and manipulates structured knowledge, which can be beneficial for tasks that involve reasoning or inference. On the other hand, vision and NLP deal more with unstructured data (like images and text), and they often require different types of models and algorithms to extract useful information or features from the data. Human-like Learning develops methods that can harness relational structures and learning-to-learn to rapidly acquire and generalize knowledge to new tasks and situations. With Human-like Learning, the learning algorithm is efficient and can adapt to new or unseen situations, which is crucial in real-world applications where environments may change unpredictably. Moreover, the models are easier for humans to understand and interpret, which is important for transparency and trust in AI systems. In this talk, we present our recent attempts towards human-like learning from relational structured data.

----

## [2552] Fairness with Censorship: Bridging the Gap between Fairness Research and Real-World Deployment

**Authors**: *Wenbin Zhang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i20.30301](https://doi.org/10.1609/aaai.v38i20.30301)

**Abstract**:

Recent works in artificial intelligence fairness attempt to mitigate discrimination by proposing constrained optimization programs that achieve parity for some fairness statistics. Most assume the availability of class label which is impractical in many real-world applications such as precision medicine, actuarial analysis and recidivism prediction. To this end, this talk revisits fairness and reveals idiosyncrasies of existing fairness literature assuming the availability of class label that limits their real-world utility. The primary artifacts are formulating fairness with censorship to account for scenarios where the class label is not guaranteed, and a suite of corresponding new fairness notions, algorithms, and theoretical constructs to bridge the gap between the design of a ``fair'' model in the lab and its deployment in the real-world.

----

## [2553] Fair and Optimal Prediction via Post-Processing

**Authors**: *Han Zhao*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i20.30302](https://doi.org/10.1609/aaai.v38i20.30302)

**Abstract**:

In this talk I will discuss our recent work on characterizing the inherent tradeoff between fairness and accuracy in both classification and regression problems. I will also present a post-processing algorithm that derives optimal fair predictors from Bayes score functions.

----

## [2554] Towards Reproducible, Automated, and Scalable Anomaly Detection

**Authors**: *Yue Zhao*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i20.30303](https://doi.org/10.1609/aaai.v38i20.30303)

**Abstract**:

Anomaly detection (AD), often termed outlier detection, is a key machine learning (ML) task, aiming to identify uncommon yet crucial patterns in data. With the increasing complexity of the modern world, the applications of AD span wide—from NASA's spacecraft monitoring to early patient prioritization at University of Pittsburgh Medical Center. Technology giants like Google and Amazon also leverage AD for service disruption identification. Here, I will traverse my AD works with promising new directions, particularly emphasizing reproducible benchmarks (Part 1), automated algorithms (Part 2), and scalable systems (Part 3).

----

## [2555] Combating Insider Threat in the Open-World Environments: Identification, Monitoring, and Data Augmentation

**Authors**: *Dawei Zhou*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i20.30304](https://doi.org/10.1609/aaai.v38i20.30304)

**Abstract**:

Recent years have witnessed a dramatic increase in a class of security threats known as "insider threats". These threats occur when individuals with authorized access to an organization's network engage in harmful activities, potentially leading to the disclosure of vital information or adversely affecting the organization's systems (e.g., financial loss, system crashes, and national security challenges).  Distinct from other types of terror attacks, combating insider threats exhibits several unique challenges, including (1) rarity, (2) non-separability, (3) label scarcity, (4) dynamics, and (5) heterogeneity, making themselves extremely difficult to identify and mitigate. We target the challenging problem of combating insider threats in open-world environments by leveraging a variety of data sources (e.g., internal system logs, employee networks, human trafficking, and smuggling networks). To effectively combat these intricate threats, we introduce an interactive learning mechanism that is composed of three mutually beneficial learning modules: insider identification, insider monitoring, and data augmentation. Each module plays a crucial role in enhancing our ability to detect and mitigate insider threats, thereby contributing to a more secure and resilient organizational environment.

----

## [2556] Select and Augment: Enhanced Dense Retrieval Knowledge Graph Augmentation (Abstract Reprint)

**Authors**: *Micheal Abaho, Yousef H. Alfaifi*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i20.30590](https://doi.org/10.1609/aaai.v38i20.30590)

**Abstract**:

Injecting textual information into knowledge graph (KG) entity representations has
been a worthwhile expedition in terms of improving performance in KG oriented tasks
within the NLP community. External knowledge often adopted to enhance KG embeddings
ranges from semantically rich lexical dependency parsed features to a set of relevant key
words to entire text descriptions supplied from an external corpus such as wikipedia and
many more. Despite the gains this innovation (Text-enhanced KG embeddings) has made,
the proposal in this work suggests that it can be improved even further. Instead of using
a single text description (which would not sufficiently represent an entity because of the
inherent lexical ambiguity of text), we propose a multi-task framework that jointly selects a
set of text descriptions relevant to KG entities as well as align or augment KG embeddings
with text descriptions. Different from prior work that plugs formal entity descriptions
declared in knowledge bases, this framework leverages a retriever model to selectively identify
richer or highly relevant text descriptions to use in augmenting entities. Furthermore, the
framework treats the number of descriptions to use in augmentation process as a parameter,
which allows the flexibility of enumerating across several numbers before identifying an
appropriate number. Experiment results for Link Prediction demonstrate a 5.5% and 3.5%
percentage increase in the Mean Reciprocal Rank (MRR) and Hits@10 scores respectively,
in comparison to text-enhanced knowledge graph augmentation methods using traditional
CNNs.

----

## [2557] Program Synthesis with Best-First Bottom-Up Search (Abstract Reprint)

**Authors**: *Saqib Ameen, Levi H. S. Lelis*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i20.30591](https://doi.org/10.1609/aaai.v38i20.30591)

**Abstract**:

Cost-guided bottom-up search (BUS) algorithms use a cost function to guide the search to solve program synthesis tasks. In this paper, we show that current state-of-the-art cost-guided BUS algorithms suffer from a common problem: they can lose useful information given by the model and fail to perform the search in a best-first order according to a cost function. We introduce a novel best-first bottom-up search algorithm, which we call Bee Search, that does not suffer information loss and is able to perform cost-guided bottom-up synthesis in a best-first manner. Importantly, Bee Search performs best-first search with respect to the generation of programs, i.e., it does not even create in memory programs that are more expensive than the solution program. It attains best-first ordering with respect to generation by performing a search in an abstract space of program costs. We also introduce a new cost function that better uses the information provided by an existing cost model. Empirical results on string manipulation and bit-vector tasks show that Bee Search can outperform existing cost-guided BUS approaches when employing more complex domain-specific languages (DSLs); Bee Search and previous approaches perform equally well with simpler DSLs. Furthermore, our new cost function with Bee Search outperforms previous cost functions on string manipulation tasks.

----

## [2558] Monitoring of Perception Systems: Deterministic, Probabilistic, and Learning-Based Fault Detection and Identification (Abstract Reprint)

**Authors**: *Pasquale Antonante, Heath Nilsen, Luca Carlone*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i20.30592](https://doi.org/10.1609/aaai.v38i20.30592)

**Abstract**:

This paper investigates runtime monitoring of perception systems. Perception is a critical component of high-integrity applications of robotics and autonomous systems, such as self-driving cars. In these applications, failure of perception systems may put human life at risk, and a broad adoption of these technologies requires the development of methodologies to guarantee and monitor safe operation. Despite the paramount importance of perception, currently there is no formal approach for system-level perception monitoring. In this paper, we formalize the problem of runtime fault detection and identification in perception systems and present a framework to model diagnostic information using a diagnostic graph. We then provide a set of deterministic, probabilistic, and learning-based algorithms that use diagnostic graphs to perform fault detection and identification. Moreover, we investigate fundamental limits and provide deterministic and probabilistic guarantees on the fault detection and identification results. We conclude the paper with an extensive experimental evaluation, which recreates several realistic failure modes in the LGSVL open-source autonomous driving simulator, and applies the proposed system monitors to a state-of-the-art autonomous driving software stack (Baidu's Apollo Auto). The results show that the proposed system monitors outperform baselines, have the potential of preventing accidents in realistic autonomous driving scenarios, and incur a negligible computational overhead.

----

## [2559] A General Model for Aggregating Annotations AcrossSimple, Complex, and Multi-object Annotation Tasks (Abstract Reprint)

**Authors**: *Alexander Braylan, Madalyn Marabella, Omar Alonso, Matthew Lease*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i20.30593](https://doi.org/10.1609/aaai.v38i20.30593)

**Abstract**:

Human annotations are vital to supervised learning, yet annotators often disagree on the correct label, especially as annotation tasks increase in complexity. A common strategy to improve label quality is to ask multiple annotators to label the same item and then aggregate their labels. To date, many aggregation models have been proposed for simple categorical or numerical annotation tasks, but far less work has considered more complex annotation tasks, such as those involving open-ended, multivariate, or structured responses. Similarly, while a variety of bespoke models have been proposed for specific tasks, our work is the first we are aware of to introduce aggregation methods that generalize across many, diverse complex tasks, including sequence labeling, translation, syntactic parsing, ranking, bounding boxes, and keypoints. This generality is achieved by applying readily available task-specific distance functions, then devising a task-agnostic method to model these distances between labels, rather than the labels themselves.

This article presents a unified treatment of our prior work on complex annotation modeling and extends that work with investigation of three new research questions. First, how do complex annotation task and dataset properties impact aggregation accuracy? Second, how should a task owner navigate the many modeling choices in order to maximize aggregation accuracy? Finally, what tests and diagnoses can verify that aggregation models are specified correctly for the given data? To understand how various factors impact accuracy and to inform model selection, we conduct large-scale simulation studies and broad experiments on real, complex datasets. Regarding testing, we introduce the concept of unit tests for aggregation models and present a suite of such tests to ensure that a given model is not mis-specified and exhibits expected behavior.

Beyond investigating these research questions above, we discuss the foundational concept and nature of annotation complexity, present a new aggregation model as a conceptual bridge between traditional models and our own, and contribute a new general semisupervised learning method for complex label aggregation that outperforms prior work.

----

## [2560] Temporal Logic Explanations for Dynamic Decision Systems Using Anchors and Monte Carlo Tree Search (Abstract Reprint)

**Authors**: *Tzu-Yi Chiu, Jerome Le Ny, Jean-Pierre David*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i20.30594](https://doi.org/10.1609/aaai.v38i20.30594)

**Abstract**:

For many automated perception and decision tasks, state-of-the-art performance may be obtained by algorithms that are too complex for their behavior to be completely understandable or predictable by human users, e.g., because they employ large machine learning models. To integrate these algorithms into safety-critical decision and control systems, it is particularly important to develop methods that can promote trust into their decisions and help explore their failure modes. In this article, we combine the anchors methodology with Monte Carlo Tree Search to provide local model-agnostic explanations for the behaviors of a given black-box model making decisions by processing time-varying input signals. Our approach searches for descriptive explanations for these decisions in the form of properties of the input signals, expressed in Signal Temporal Logic, which are highly likely to reproduce the observed behavior. To illustrate the methodology, we apply it in simulations to the analysis of a hybrid (continuous-discrete) control system and a collision avoidance system for unmanned aircraft (ACAS Xu) implemented by a neural network.

----

## [2561] Mimicking Behaviors in Separated Domains (Abstract Reprint)

**Authors**: *Giuseppe De Giacomo, Dror Fried, Fabio Patrizi, Shufang Zhu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i20.30595](https://doi.org/10.1609/aaai.v38i20.30595)

**Abstract**:

Devising a strategy to make a system mimic behaviors from another system is a problem that naturally arises in many areas of Computer Science. In this work, we interpret this problem in the context of intelligent agents, from the perspective of LTLf, a formalism commonly used in AI for expressing finite-trace properties. Our model consists of two separated dynamic domains, D_A and D_B, and an LTLf specification that formalizes the notion of mimicking by mapping properties on behaviors (traces) of D_A into properties on behaviors of D_B. The goal is to synthesize a strategy that step-by-step maps every behavior of D_A into a behavior of D_B so that the specification is met. We consider several forms of mapping specifications, ranging from simple ones to full LTLf, and for each, we study synthesis algorithms and computational properties.

----

## [2562] Counterfactual Explanations for Misclassified Images: How Human and Machine Explanations Differ (Abstract Reprint)

**Authors**: *Eoin Delaney, Arjun Pakrashi, Derek Greene, Mark T. Keane*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i20.30596](https://doi.org/10.1609/aaai.v38i20.30596)

**Abstract**:

Counterfactual explanations have emerged as a popular solution for the eXplainable AI (XAI) problem of elucidating the predictions of black-box deep-learning systems because people easily understand them, they apply across different problem domains and seem to be legally compliant. Although over 100 counterfactual methods exist in the XAI literature, each claiming to generate plausible explanations akin to those preferred by people, few of these methods have actually been tested on users (∼7%). Even fewer studies adopt a user-centered perspective; for instance, asking people for their counterfactual explanations to determine their perspective on a “good explanation”. This gap in the literature is addressed here using a novel methodology that (i) gathers human-generated counterfactual explanations for misclassified images, in two user studies and, then, (ii) compares these human-generated explanations to computationally-generated explanations for the same misclassifications. Results indicate that humans do not “minimally edit” images when generating counterfactual explanations. Instead, they make larger, “meaningful” edits that better approximate prototypes in the counterfactual class. An analysis based on “explanation goals” is proposed to account for this divergence between human and machine explanations. The implications of these proposals for future work are discussed.

----

## [2563] Reasoning about Causality in Games (Abstract Reprint)

**Authors**: *Lewis Hammond, James Fox, Tom Everitt, Ryan Carey, Alessandro Abate, Michael J. Wooldridge*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i20.30597](https://doi.org/10.1609/aaai.v38i20.30597)

**Abstract**:

Causal reasoning and game-theoretic reasoning are fundamental topics in artificial intelligence, among many other disciplines: this paper is concerned with their intersection. Despite their importance, a formal framework that supports both these forms of reasoning has, until now, been lacking. We offer a solution in the form of (structural) causal games, which can be seen as extending Pearl's causal hierarchy to the game-theoretic domain, or as extending Koller and Milch's multi-agent influence diagrams to the causal domain. We then consider three key questions:
i)
How can the (causal) dependencies in games – either between variables, or between strategies – be modelled in a uniform, principled manner?

ii)
How may causal queries be computed in causal games, and what assumptions does this require?

iii)
How do causal games compare to existing formalisms?

To address question i), we introduce mechanised games, which encode dependencies between agents' decision rules and the distributions governing the game. In response to question ii), we present definitions of predictions, interventions, and counterfactuals, and discuss the assumptions required for each. Regarding question iii), we describe correspondences between causal games and other formalisms, and explain how causal games can be used to answer queries that other causal or game-theoretic models do not support. Finally, we highlight possible applications of causal games, aided by an extensive open-source Python library.

----

## [2564] A Survey of Learning Criteria Going beyond the Usual Risk (Abstract Reprint)

**Authors**: *Matthew J. Holland, Kazuki Tanabe*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i20.30598](https://doi.org/10.1609/aaai.v38i20.30598)

**Abstract**:

Virtually all machine learning tasks are characterized using some form of loss function, and "good performance" is typically stated in terms of a sufficiently small average loss, taken over the random draw of test data. While optimizing for performance on average is intuitive, convenient to analyze in theory, and easy to implement in practice, such a choice brings about trade-offs. In this work, we survey and introduce a wide variety of non-traditional criteria used to design and evaluate machine learning algorithms, place the classical paradigm within the proper historical context, and propose a view of learning problems which emphasizes the question of "what makes for a desirable loss distribution?" in place of tacit use of the expected loss.

----

## [2565] Sim-to-Lab-to-Real: Safe Reinforcement Learning with Shielding and Generalization Guarantees (Abstract Reprint)

**Authors**: *Kai-Chieh Hsu, Allen Z. Ren, Duy Phuong Nguyen, Anirudha Majumdar, Jaime F. Fisac*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i20.30599](https://doi.org/10.1609/aaai.v38i20.30599)

**Abstract**:

Safety is a critical component of autonomous systems and remains a challenge for learning-based policies to be utilized in the real world. In particular, policies learned using reinforcement learning often fail to generalize to novel environments due to unsafe behavior. In this paper, we propose Sim-to-Lab-to-Real to bridge the reality gap with a probabilistically guaranteed safety-aware policy distribution. To improve safety, we apply a dual policy setup where a performance policy is trained using the cumulative task reward and a backup (safety) policy is trained by solving the Safety Bellman Equation based on Hamilton-Jacobi (HJ) reachability analysis. In Sim-to-Lab transfer, we apply a supervisory control scheme to shield unsafe actions during exploration; in Lab-to-Real transfer, we leverage the Probably Approximately Correct (PAC)-Bayes framework to provide lower bounds on the expected performance and safety of policies in unseen environments. Additionally, inheriting from the HJ reachability analysis, the bound accounts for the expectation over the worst-case safety in each environment. We empirically study the proposed framework for ego-vision navigation in two types of indoor environments with varying degrees of photorealism. We also demonstrate strong generalization performance through hardware experiments in real indoor spaces with a quadrupedal robot. See https://sites.google.com/princeton.edu/sim-to-lab-to-real for supplementary material.

----

## [2566] FlexiBO: A Decoupled Cost-Aware Multi-objective Optimization Approach for Deep Neural Networks (Abstract Reprint)

**Authors**: *Md Shahriar Iqbal, Jianhai Su, Lars Kotthoff, Pooyan Jamshidi*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i20.30600](https://doi.org/10.1609/aaai.v38i20.30600)

**Abstract**:

The design of machine learning systems often requires trading off different objectives, for example, prediction error and energy consumption for deep neural networks (DNNs). Typically, no single design performs well in all objectives; therefore, finding Pareto-optimal designs is of interest. The search for Pareto-optimal designs involves evaluating designs in an iterative process, and the measurements are used to evaluate an acquisition function that guides the search process. However, measuring different objectives incurs different costs. For example, the cost of measuring the prediction error of DNNs is orders of magnitude higher than that of measuring the energy consumption of a pre-trained DNN as it requires re-training the DNN. Current state-of-the-art methods do not consider this difference in objective evaluation cost, potentially incurring expensive evaluations of objective functions in the optimization process. In this paper, we develop a novel decoupled and cost-aware multi-objective optimization algorithm, which we call Flexible Multi-Objective Bayesian Optimization (FlexiBO) to address this issue. For evaluating each design, FlexiBO selects the objective with higher relative gain by weighting the improvement of the hypervolume of the Pareto region with the measurement cost of each objective. This strategy, therefore, balances the expense of collecting new information with the knowledge gained through objective evaluations, preventing FlexiBO from performing expensive measurements for little to no gain. We evaluate FlexiBO on seven state-of-the-art DNNs for image recognition, natural language processing (NLP), and speech-to-text translation. Our results indicate that, given the same total experimental budget, FlexiBO discovers designs with 4.8% to 12.4% lower hypervolume error than the best method in state-of-the-art multi-objective optimization.

----

## [2567] Discovering Agents (Abstract Reprint)

**Authors**: *Zachary Kenton, Ramana Kumar, Sebastian Farquhar, Jonathan Richens, Matt MacDermott, Tom Everitt*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i20.30601](https://doi.org/10.1609/aaai.v38i20.30601)

**Abstract**:

Causal models of agents have been used to analyse the safety aspects of machine learning systems. But identifying agents is non-trivial – often the causal model is just assumed by the modeller without much justification – and modelling failures can lead to mistakes in the safety analysis. This paper proposes the first formal causal definition of agents – roughly that agents are systems that would adapt their policy if their actions influenced the world in a different way. From this we derive the first causal discovery algorithm for discovering the presence of agents from empirical data, given a set of variables and under certain assumptions. We also provide algorithms for translating between causal models and game-theoretic influence diagrams. We demonstrate our approach by resolving some previous confusions caused by incorrect causal modelling of agents.

----

## [2568] Reward (Mis)design for Autonomous Driving (Abstract Reprint)

**Authors**: *W. Bradley Knox, Alessandro Allievi, Holger Banzhaf, Felix Schmitt, Peter Stone*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i20.30602](https://doi.org/10.1609/aaai.v38i20.30602)

**Abstract**:

This article considers the problem of diagnosing certain common errors in reward design. Its insights are also applicable to the design of cost functions and performance metrics more generally. To diagnose common errors, we develop 8 simple sanity checks for identifying flaws in reward functions. We survey research that is published in top-tier venues and focuses on reinforcement learning (RL) for autonomous driving (AD). Specifically, we closely examine the reported reward function in each publication and present these reward functions in a complete and standardized format in the appendix. Wherever we have sufficient information, we apply the 8 sanity checks to each surveyed reward function, revealing near-universal flaws in reward design for AD that might also exist pervasively across reward design for other tasks. Lastly, we explore promising directions that may aid the design of reward functions for AD in subsequent research, following a process of inquiry that can be adapted to other domains.

----

## [2569] The Defeat of the Winograd Schema Challenge (Abstract Reprint)

**Authors**: *Vid Kocijan, Ernest Davis, Thomas Lukasiewicz, Gary Marcus, Leora Morgenstern*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i20.30603](https://doi.org/10.1609/aaai.v38i20.30603)

**Abstract**:

The Winograd Schema Challenge—a set of twin sentences involving pronoun reference disambiguation that seem to require the use of commonsense knowledge—was proposed by Hector Levesque in 2011. By 2019, a number of AI systems, based on large pre-trained transformer-based language models and fine-tuned on these kinds of problems, achieved better than 90% accuracy. In this paper, we review the history of the Winograd Schema Challenge and discuss the lasting contributions of the flurry of research that has taken place on the WSC in the last decade. We discuss the significance of various datasets developed for WSC, and the research community's deeper understanding of the role of surrogate tasks in assessing the intelligence of an AI system.

----

## [2570] Convolutional Spectral Kernel Learning with Generalization Guarantees (Abstract Reprint)

**Authors**: *Jian Li, Yong Liu, Weiping Wang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i20.30604](https://doi.org/10.1609/aaai.v38i20.30604)

**Abstract**:

Kernel methods are powerful tools to capture nonlinear patterns behind given data but often lead to poor performance on complicated tasks compared to convolutional neural networks. The reason is that kernel methods are still shallow and fully connected models, failing to reveal hierarchical features and local interdependencies. In this paper, to acquire hierarchical and local knowledge, we incorporate kernel methods with deep architectures and convolutional operators in a spectral kernel learning framework. Based on the inverse Fourier transform and Rademacher complexity theory, we provide the generalization error bounds for the proposed model and prove that under suitable initialization, deeper networks lead to tighter error bounds. Inspired by theoretical findings, we finally completed the convolutional spectral kernel network (CSKN) with two additional regularizers and an initialization strategy. Extensive ablation results validate the effectiveness of non-stationary spectral kernel, multiple layers, additional regularizers, and the convolutional filters, which coincide with our theoretical findings. We further devise a VGG-type 8-layers CSKN, and it outperforms the existing kernel-based networks and popular CNN models on the medium-sized image classification tasks.

----

## [2571] G-LIME: Statistical Learning for Local Interpretations of Deep Neural Networks Using Global Priors (Abstract Reprint)

**Authors**: *Xuhong Li, Haoyi Xiong, Xingjian Li, Xiao Zhang, Ji Liu, Haiyan Jiang, Zeyu Chen, Dejing Dou*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i20.30605](https://doi.org/10.1609/aaai.v38i20.30605)

**Abstract**:

To explain the prediction result of a Deep Neural Network (DNN) model based on a given sample, LIME [1] and its derivatives have been proposed to approximate the local behavior of the DNN model around the data point via linear surrogates. Though these algorithms interpret the DNN by finding the key features used for classification, the random interpolations used by LIME would perturb the explanation result and cause the instability and inconsistency between repetitions of LIME computations. To tackle this issue, we propose G-LIME that extends the vanilla LIME through high-dimensional Bayesian linear regression using the sparsity and informative global priors. Specifically, with a dataset representing the population of samples (e.g., the training set), G-LIME first pursues the global explanation of the DNN model using the whole dataset. Then, with a new data point, -LIME incorporates an modified estimator of ElasticNet-alike to refine the local explanation result through balancing the distance to the global explanation and the sparsity/feature selection in the explanation. Finally, G-LIME uses Least Angle Regression (LARS) and retrieves the solution path of a modified ElasticNet under varying -regularization, to screen and rank the importance of features [2] as the explanation result. Through extensive experiments on real world tasks, we show that the proposed method yields more stable, consistent, and accurate results compared to LIME.

----

## [2572] Exploiting Action Impact Regularity and Exogenous State Variables for Offline Reinforcement Learning (Abstract Reprint)

**Authors**: *Vincent Liu, James R. Wright, Martha White*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i20.30606](https://doi.org/10.1609/aaai.v38i20.30606)

**Abstract**:

Offline reinforcement learning—learning a policy from a batch of data—is known to be hard for general MDPs. These results motivate the need to look at specific classes of MDPs where offline reinforcement learning might be feasible. In this work, we explore a restricted class of MDPs to obtain guarantees for offline reinforcement learning. The key property, which we call Action Impact Regularity (AIR), is that actions primarily impact a part of the state (an endogenous component) and have limited impact on the remaining part of the state (an exogenous component). AIR is a strong assumption, but it nonetheless holds in a number of real-world domains including financial markets. We discuss algorithms that exploit the AIR property, and provide a theoretical analysis for an algorithm based on Fitted-Q Iteration. Finally, we demonstrate that the algorithm outperforms existing offline reinforcement learning algorithms across different data collection policies in simulated and real world environments where the regularity holds.

----

## [2573] Introduction to the Special Track on Artificial Intelligence and COVID-19 (Abstract Reprint)

**Authors**: *Martin Michalowski, Robert Moskovitch, Nitesh V. Chawla*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i20.30607](https://doi.org/10.1609/aaai.v38i20.30607)

**Abstract**:

The human race is facing one of the most meaningful public health emergencies in the modern era caused by the COVID-19 pandemic. This pandemic introduced various challenges, from lock-downs with significant economic costs to fundamentally altering the way of life for many people around the world. The battle to understand and control the virus is still at its early stages yet meaningful insights have already been made. The uncertainty of why some patients are infected and experience severe symptoms, while others are infected but asymptomatic, and others are not infected at all, makes managing this pandemic very challenging. Furthermore, the development of treatments and vaccines relies on knowledge generated from an ever evolving and expanding information space. Given the availability of digital data in the modern era, artificial intelligence (AI) is a meaningful tool for addressing the various challenges introduced by this unexpected pandemic. Some of the challenges include: outbreak prediction, risk modeling including infection and symptom development, testing strategy optimization, drug development, treatment repurposing, vaccine development, and others.

----

## [2574] TEAMSTER: Model-Based Reinforcement Learning for Ad Hoc Teamwork (Abstract Reprint)

**Authors**: *João G. Ribeiro, Gonçalo Rodrigues, Alberto Sardinha, Francisco S. Melo*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i20.30608](https://doi.org/10.1609/aaai.v38i20.30608)

**Abstract**:

This paper investigates the use of model-based reinforcement learning in the context of ad hoc teamwork. We introduce a novel approach, named TEAMSTER, where we propose learning both the environment's model and the model of the teammates' behavior separately. Compared to the state-of-the-art PLASTIC algorithms, our results in four different domains from the multi-agent systems literature show that TEAMSTER is more flexible than the PLASTIC-Model, by learning the environment's model instead of assuming a perfect hand-coded model, and more robust/efficient than PLASTIC-Policy, by being able to continuously adapt to newly encountered teams, without implicitly learning a new environment model from scratch.

----

## [2575] Sequential Model-Based Diagnosis by Systematic Search (Abstract Reprint)

**Authors**: *Patrick Rodler*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i20.30609](https://doi.org/10.1609/aaai.v38i20.30609)

**Abstract**:

Model-based diagnosis aims at identifying the real cause of a system's malfunction based on a formal system model and observations of the system behavior. To discriminate between multiple fault hypotheses (diagnoses), sequential diagnosis approaches iteratively pose queries to an oracle to acquire additional knowledge about the diagnosed system. Depending on the system type, queries can capture, e.g., system tests, probes, measurements, or expert questions.

As the determination of optimal queries is NP-hard, state-of-the-art sequential diagnosis methods rely on a myopic one-step-lookahead analysis which has proven to constitute a particularly favorable trade-off between computational efficiency and diagnostic effectivity. Yet, this solves only a part of the problem, as various sources of complexity, such as the reliance on costly reasoning services and large numbers of or not explicitly given query candidates, remain.

To deal with such issues, existing approaches often make assumptions about the (i) type of diagnosed system, (ii) formalism to describe the system, (iii) inference engine, (iv) type of query to be of interest, (v) query quality criterion to be adopted, or (vi) diagnosis computation algorithm to be employed. Moreover, they (vii) often cannot deal with large or implicit query spaces or with expressive logics, or (viii) require inputs that cannot always be provided.

As a remedy, we propose a novel one-step lookahead query computation technique for sequential diagnosis that overcomes the said issues of existing methods. Our approach (1) is based on a solid theory, (2) involves a systematic search for optimal queries, (3) can operate on implicit and huge query spaces, (4) allows for a two-stage optimization of queries (wrt. their number and cost), (5) is designed to reduce expensive logical inferences to a minimum, and (6) is generally applicable. The latter means that it can deal with any type of diagnosis problem as per Reiter's theory, is applicable with any monotonic knowledge representation language, can interact with a multitude of diagnosis engines and logical reasoners, and allows for a quality optimization of queries based on any of the common criteria in the literature.

We extensively study the performance of the novel technique using a benchmark of real-world diagnosis problems. Our findings are that our approach enables the computation of optimal queries with hardly any delay, independently of the size and complexity of the considered benchmark problem. Moreover, it proves to be highly scalable, and it outperforms the state-of-the-art method in the domain of our benchmarks by orders of magnitude in terms of computation time while always returning a qualitatively as good or better query.

----

## [2576] Actor Prioritized Experience Replay (Abstract Reprint)

**Authors**: *Baturay Saglam, Furkan B. Mutlu, Dogan C. Cicek, Suleyman S. Kozat*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i20.30610](https://doi.org/10.1609/aaai.v38i20.30610)

**Abstract**:

A widely-studied deep reinforcement learning (RL) technique known as Prioritized Experience Replay (PER) allows agents to learn from transitions sampled with non-uniform probability proportional to their temporal-difference (TD) error. Although it has been shown that PER is one of the most crucial components for the overall performance of deep RL methods in discrete action domains, many empirical studies indicate that it considerably underperforms off-policy actor-critic algorithms. We theoretically show that actor networks cannot be effectively trained with transitions that have large TD errors. As a result, the approximate policy gradient computed under the Q-network diverges from the actual gradient computed under the optimal Q-function. Motivated by this, we introduce a novel experience replay sampling framework for actor-critic methods, which also regards issues with stability and recent findings behind the poor empirical performance of PER. The introduced algorithm suggests a new branch of improvements to PER and schedules effective and efficient training for both actor and critic networks. An extensive set of experiments verifies our theoretical findings, showing that our method outperforms competing approaches and achieves state-of-the-art results over the standard off-policy actor-critic algorithms.

----

## [2577] Accurate Parameter Estimation for Safety-Critical Systems with Unmodeled Dynamics (Abstract Reprint)

**Authors**: *Arnab Sarker, Peter A. Fisher, Joseph E. Gaudio, Anuradha Annaswamy*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i20.30611](https://doi.org/10.1609/aaai.v38i20.30611)

**Abstract**:

Analysis and synthesis of safety-critical autonomous systems are carried out using models which are often dynamic. Two central features of these dynamic systems are parameters and unmodeled dynamics. Much of feedback control design is parametric in nature and as such, accurate and fast estimation of the parameters in the modeled part of the dynamic system is a crucial property for designing risk-aware autonomous systems. This paper addresses the use of a spectral lines-based approach for estimating parameters of the dynamic model of an autonomous system. Existing literature has treated all unmodeled components of the dynamic system as sub-Gaussian noise and proposed parameter estimation using Gaussian noise-based exogenous signals. In contrast, we allow the unmodeled part to have deterministic unmodeled dynamics, which are almost always present in physical systems, in addition to sub-Gaussian noise. In addition, we propose a deterministic construction of the exogenous signal in order to carry out parameter estimation. We introduce a new tool kit which employs the theory of spectral lines, retains the stochastic setting, and leads to non-asymptotic bounds on the parameter estimation error. Unlike the existing stochastic approach, these bounds are tunable through an optimal choice of the spectrum of the exogenous signal leading to accurate parameter estimation. We also show that this estimation is robust to unmodeled dynamics, a property that is not assured by the existing approach. Finally, we show that under ideal conditions with no deterministic unmodeled dynamics, the proposed approach can ensure a Õ(√t) Regret, matching existing literature. Experiments are provided to support all theoretical derivations, which show that the spectral lines-based approach outperforms the Gaussian noise-based method when unmodeled dynamics are present, in terms of both parameter estimation error and Regret obtained using the parameter estimates with a Linear Quadratic Regulator in feedback.

----

## [2578] Your Prompt Is My Command: On Assessing the Human-Centred Generality of Multimodal Models (Abstract Reprint)

**Authors**: *Wout Schellaert, Fernando Martínez-Plumed, Karina Vold, John Burden, Pablo A. M. Casares, Bao Sheng Loe, Roi Reichart, Seán Ó hÉigeartaigh, Anna Korhonen, José Hernández-Orallo*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i20.30612](https://doi.org/10.1609/aaai.v38i20.30612)

**Abstract**:

Even with obvious deficiencies, large prompt-commanded multimodal models are proving to be flexible cognitive tools representing an unprecedented generality. But the directness, diversity, and degree of user interaction create a distinctive “human-centred generality” (HCG), rather than a fully autonomous one. HCG implies that —for a specific user— a system is only as general as it is effective for the user’s relevant tasks and their prevalent ways of prompting. A human-centred evaluation of general-purpose AI systems therefore needs to reflect the personal nature of interaction, tasks and cognition. We argue that the best way to understand these systems is as highly-coupled cognitive extenders, and to analyse the bidirectional cognitive adaptations between them and humans. In this paper, we give a formulation of HCG, as well as a high-level overview of the elements and trade-offs involved in the prompting process. We end the paper by outlining some essential research questions and suggestions for improving evaluation practices, which we envision as characteristic for the evaluation of general artificial intelligence in the future.

----

## [2579] Reward-Respecting Subtasks for Model-Based Reinforcement Learning (Abstract Reprint)

**Authors**: *Richard S. Sutton, Marlos C. Machado, G. Zacharias Holland, David Szepesvari, Finbarr Timbers, Brian Tanner, Adam White*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i20.30613](https://doi.org/10.1609/aaai.v38i20.30613)

**Abstract**:

To achieve the ambitious goals of artificial intelligence, reinforcement learning must
include planning with a model of the world that is abstract in state and time. Deep learning
has made progress with state abstraction, but temporal abstraction has rarely been used,
despite extensively developed theory based on the options framework. One reason for this
is that the space of possible options is immense, and the methods previously proposed
for option discovery do not take into account how the option models will be used in
planning. Options are typically discovered by posing subsidiary tasks, such as reaching a
bottleneck state or maximizing the cumulative sum of a sensory signal other than reward.
Each subtask is solved to produce an option, and then a model of the option is learned and
made available to the planning process. In most previous work, the subtasks ignore the
reward on the original problem, whereas we propose subtasks that use the original reward
plus a bonus based on a feature of the state at the time the option terminates. We show
that option models obtained from such reward-respecting subtasks are much more likely to
be useful in planning than eigenoptions, shortest path options based on bottleneck states,
or reward-respecting options generated by the option-critic. Reward respecting subtasks
strongly constrain the space of options and thereby also provide a partial solution to the
problem of option discovery. Finally, we show how values, policies, options, and models can
all be learned online and off-policy using standard algorithms and general value functions.

----

## [2580] Post-trained Convolution Networks for Single Image Super-resolution (Abstract Reprint)

**Authors**: *Seid Miad Zandavi*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i20.30614](https://doi.org/10.1609/aaai.v38i20.30614)

**Abstract**:

A new method is proposed to increase the accuracy of the state-of-the-art single image super-resolution (SISR) using novel training procedure. The proposed method, named post-trained convolutional neural network (CNN), is carried out stochastic dual simplex algorithm (SDSA) in the last reconstruction layer. The method utilizes contextual information to update the last reconstruction layer of CNN. The extracted contextual information is projected to the last reconstructed layer by optimized weights and the bias is managed through SDSA. Post-trained CNN is applied to the very deep super-resolution (VDSR) method to show its performance. The quantitative and visual results demonstrate that the proposed post-trained VDSR (PTVDSR) exhibits excellent and competitive performance when compared with the VDSR and other super-resolution methods.

----

## [2581] Flood Insights: Integrating Remote and Social Sensing Data for Flood Exposure, Damage, and Urgent Needs Mapping

**Authors**: *Zainab Akhtar, Umair Qazi, Aya El-Sakka, Rizwan Sadiq, Ferda Ofli, Muhammad Imran*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30305](https://doi.org/10.1609/aaai.v38i21.30305)

**Abstract**:

The absence of comprehensive situational awareness information poses a significant challenge for humanitarian organizations during their response efforts. We present Flood Insights, an end-to-end system that ingests data from multiple non-traditional data sources such as remote sensing, social sensing, and geospatial data. We employ state-of-the-art natural language processing and computer vision models to identify flood exposure, ground-level damage and flood reports, and most importantly, urgent needs of affected people. We deploy and test the system during a recent real-world catastrophe, the 2022 Pakistan floods, to surface critical situational and damage information at the district level. We validated the system's effectiveness through geographic regression analysis using official ground-truth data, showcasing its strong performance and explanatory power. Moreover, the system was commended by the United Nations Development Programme stationed in Pakistan, as well as local authorities, for pinpointing hard-hit districts and enhancing disaster response.

----

## [2582] Building Conversational Artifacts to Enable Digital Assistant for APIs and RPAs

**Authors**: *Jayachandu Bandlamudi, Kushal Mukherjee, Prerna Agarwal, Ritwik Chaudhuri, Rakesh Pimplikar, Sampath Dechu, Alex Straley, Anbumunee Ponniah, Renuka Sindhgatta*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30306](https://doi.org/10.1609/aaai.v38i21.30306)

**Abstract**:

In the realm of business automation, digital assistants/chatbots are emerging as the primary method for making automation software accessible to users in various business sectors. Access to automation primarily occurs through APIs and RPAs. To effectively convert APIs and RPAs into chatbots on a larger scale, it is crucial to establish an automated process for generating data and training models that can recognize user intentions, identify questions for conversational slot filling, and provide recommendations for subsequent actions. In this paper, we present a technique for enhancing and generating natural language conversational artifacts from API specifications using large language models (LLMs). The goal is to utilize LLMs in the "build" phase to assist humans in creating skills for digital assistants. As a result, the system doesn't need to rely on LLMs during conversations with business users, leading to efficient deployment. Experimental results highlight the effectiveness of our proposed approach. Our system is deployed in the IBM Watson Orchestrate product for general availability.

----

## [2583] Some Like It Small: Czech Semantic Embedding Models for Industry Applications

**Authors**: *Jirí Bednár, Jakub Náplava, Petra Barancíková, Ondrej Lisický*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30307](https://doi.org/10.1609/aaai.v38i21.30307)

**Abstract**:

This article focuses on the development and evaluation of Small-sized Czech sentence embedding models. Small models are important components for real-time industry applications in resource-constrained environments. Given the limited availability of labeled Czech data, alternative approaches, including pre-training, knowledge distillation, and unsupervised contrastive fine-tuning, are investigated. Comprehensive intrinsic and extrinsic analyses are conducted, showcasing the competitive performance of our models compared to significantly larger counterparts, with approximately 8 times smaller size and 5 times faster speed than conventional Base-sized models. To promote cooperation and reproducibility, both the models and the evaluation pipeline are made publicly accessible. Ultimately, this article presents practical applications of the developed sentence embedding models in Seznam.cz, the Czech search engine. These models have effectively replaced previous counterparts, enhancing the overall search experience for instance, in organic search, featured snippets, and image search. This transition has yielded improved performance.

----

## [2584] Check-In Desk Scheduling Optimisation at CDG International Airport

**Authors**: *Thibault Falque, Gilles Audemard, Christophe Lecoutre, Bertrand Mazure*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30308](https://doi.org/10.1609/aaai.v38i21.30308)

**Abstract**:

More than ever, air transport players (i.e., airline and airport companies) in an intensely competitive climate need to benefit from a carefully optimized management of airport resources to improve the quality of service and control the induced costs.
In this paper, we investigate the Airport Check-in Desk Assignment Problem.
We propose a Constraint Programming (CP) model for this problem, and present some promising experimental results from data coming from ADP (Aéroport de Paris).
Our works are deployed in a preprod environment since 1 year.

----

## [2585] General Commerce Intelligence: Glocally Federated NLP-Based Engine for Privacy-Preserving and Sustainable Personalized Services of Multi-Merchants

**Authors**: *Kyoung Jun Lee, Baek Jeong, Suhyeon Kim, Dam Kim, Dongju Park*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30309](https://doi.org/10.1609/aaai.v38i21.30309)

**Abstract**:

One of the most crucial capabilities in the commercial sector is a personalized prediction of a customer's next purchase. We present a novel method of creating a commerce intelligence engine that caters to multiple merchants intended for the UB Platform, managed by e-payment company Harex InfoTech. To cultivate this intelligence, we utilized payment receipt data and created a Natural Language Processing (NLP)-based commerce model using a Transformer to accommodate multinational and merchant trade. Our model, called General Commerce Intelligence (GCI), provides a range of services for merchants, including product recommendations, product brainstorming, product bundling, event promotions, collaborative marketing, target marketing, and demand fore-casting etc. To bolster user privacy and foster sustainable business collaboration, especially among micro-, small-, and medium-sized enterprises (MSMEs), the GCI model was trained through federated learning, especially with glocalization. This study delves into the structure, development, and assessment of GCI, showcasing its transformative capacity to implement User Centric AI and re-shape the global commerce landscape to benefit MSMEs.

----

## [2586] A Submodular Optimization Approach to Accountable Loan Approval

**Authors**: *Kyungsik Lee, Hana Yoo, Sumin Shin, Wooyoung Kim, Yeonung Baek, Hyunjin Kang, Jaehyun Kim, Kee-Eung Kim*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30310](https://doi.org/10.1609/aaai.v38i21.30310)

**Abstract**:

In the field of finance, the underwriting process is an essential step in evaluating every loan application. During this stage, the borrowers' creditworthiness and ability to repay the loan are assessed to ultimately decide whether to approve the loan application. One of the core components of underwriting is credit scoring, in which the probability of default is estimated. 
As such, there has been significant progress in enhancing the predictive accuracy of credit scoring models through the use of machine learning, but there still exists a need to ultimately construct an approval rule that takes into consideration additional criteria beyond the score itself. This construction process is traditionally done manually to ensure that the approval rule remains interpretable to humans.
In this paper, we outline an automated system for optimizing a rule-based system for approving loan applications, which has been deployed at Hyundai Capital Services (HCS). The main challenge lay in creating a high-quality rule base that is simultaneously simple enough to be interpretable by  risk analysts as well as customers, since the approval decision should be accountable. We addressed this challenge through principled submodular optimization. The deployment of our system has led to a 14% annual growth in the volume of loan services at HCS, while maintaining the target bad rate, and has resulted in the approval of customers who might have otherwise been rejected.

----

## [2587] Transformer-Empowered Multi-Modal Item Embedding for Enhanced Image Search in E-commerce

**Authors**: *Chang Liu, Peng Hou, Anxiang Zeng, Han Yu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30311](https://doi.org/10.1609/aaai.v38i21.30311)

**Abstract**:

Over the past decade, significant advances have been made in the field of image search for e-commerce applications. Traditional image-to-image retrieval models, which focus solely on image details such as texture, tend to overlook useful semantic information contained within the images. As a result, the retrieved products might possess similar image details, but fail to fulfil the user's search goals. Moreover, the use of image-to-image retrieval models for products containing multiple images results in significant online product feature storage overhead and complex mapping implementations. In this paper, we report the design and deployment of the proposed Multi-modal Item Embedding Model (MIEM) to address these limitations. It is capable of utilizing both textual information and multiple images about a product to construct meaningful product features. By leveraging semantic information from images, MIEM effectively supplements the image search process, improving the overall accuracy of retrieval results. MIEM has become an integral part of the Shopee image search platform. Since its deployment in March 2023, it has achieved a remarkable 9.90% increase in terms of clicks per user and a 4.23% boost in terms of orders per user for the image search feature on the Shopee e-commerce platform.

----

## [2588] High Significant Fault Detection in Azure Core Workload Insights

**Authors**: *Pranay Lohia, Laurent Boué, Sharath Ranganath, Vijay Agneeswaran*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30312](https://doi.org/10.1609/aaai.v38i21.30312)

**Abstract**:

Azure Core workload insights have time-series data with different metric units. Faults or Anomalies are observed in these time-series data owing to faults observed with respect to metric name, resources region, dimensions, and its dimension value associated with the data. For Azure Core, an important task is to highlight faults or anomalies to the user on a dashboard that they can perceive easily. The number of anomalies reported should be highly significant and in a limited number, e.g., 5-20 anomalies reported per hour. The reported anomalies will have significant user perception and high reconstruction error in any time-series forecasting model. Hence, our task is to automatically identify 'high significant anomalies' and their associated information for user perception.

----

## [2589] DCV2I: A Practical Approach for Supporting Geographers' Visual Interpretation in Dune Segmentation with Deep Vision Models

**Authors**: *Anqi Lu, Zifeng Wu, Zheng Jiang, Wei Wang, Eerdun Hasi, Yi Wang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30313](https://doi.org/10.1609/aaai.v38i21.30313)

**Abstract**:

Visual interpretation is extremely important in human geography as the primary technique for geographers to use photograph data in identifying, classifying, and quantifying geographic and topological objects or regions. However, it is also time-consuming and requires overwhelming manual effort from professional geographers. This paper describes our interdisciplinary team's efforts in integrating computer vision models with geographers' visual image interpretation process to reduce their workload in interpreting images. Focusing on the dune segmentation task, we proposed an approach featuring a deep dune segmentation model to identify dunes and label their ranges in an automated way. By developing a tool to connect our model with ArcGIS, one of the most popular workbenches for visual interpretation, geographers can further refine the automatically-generated dune segmentation on images without learning any CV or deep learning techniques. Our approach thus realized a non-invasive change to geographers' visual interpretation routines, reducing their manual efforts while incurring minimal interruptions to their work routines and tools they are familiar with. Deployment with a leading Chinese geography research institution demonstrated the potential of our approach in supporting geographers in researching and solving drylands desertification.

----

## [2590] KAMEL: Knowledge Aware Medical Entity Linkage to Automate Health Insurance Claims Processing

**Authors**: *Sheng Jie Lui, Cheng Xiang, Shonali Krishnaswamy*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30314](https://doi.org/10.1609/aaai.v38i21.30314)

**Abstract**:

Automating the processing of health insurance claims to achieve "Straight-Through Processing" is one of the holy grails that all insurance companies aim to achieve. One of the major impediments to this automation is the difficulty in establishing the relationship between the underwriting exclusions that a policy has and the incoming claim's diagnosis information. Typically, policy underwriting exclusions are captured in free-text such as "Respiratory illnesses are excluded due to a pre-existing asthma condition". A medical claim coming from a hospital would have the diagnosis represented using the International Classification of Disease (ICD) codes from the World Health Organization. The complex and labour-intensive task of establishing the relationship between free-text underwriting exclusions in health insurance policies and medical diagnosis codes from health insurance claims is critical towards determining if a claim should be rejected due to underwriting exclusions. In this work, we present a novel framework that leverages both explicit and implicit domain knowledge present in medical ontologies and pre-trained language models respectively, to effectively establish the relationship between free-text describing medical conditions present in underwriting exclusions and the ICD-10CM diagnosis codes in health insurance claims. Termed KAMEL (Knowledge Aware Medical Entity Linkage), our proposed framework addresses the limitations faced by prior approaches when evaluated on real-world health insurance claims data. Our proposed framework have been deployed in several multi-national health insurance providers to automate their health insurance claims.

----

## [2591] The Virtual Driving Instructor: Multi-Agent System Collaborating via Knowledge Graph for Scalable Driver Education

**Authors**: *Johannes Rehm, Irina Reshodko, Stian Zimmermann Børresen, Odd Erik Gundersen*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30315](https://doi.org/10.1609/aaai.v38i21.30315)

**Abstract**:

This paper introduces the design, development, and deployment of a Virtual Driving Instructor (VDI) for enhanced driver education. 
The VDI provides personalized, real-time feedback to students in a driving simulator, addressing some of the limitations of traditional driver instruction. 
Employing a hybrid AI system, the VDI combines rule-based agents, learning-based agents, knowledge graphs, and Bayesian networks to assess and monitor student performance in a comprehensive manner. 
Implemented in multiple simulators at a driving school in Norway, the system aims to leverage AI and driving simulation to improve both the learning experience and the efficiency of instruction. 
Initial feedback from students has been largely positive, highlighting the effectiveness of this integration while also pointing to areas for further improvement. 
This work marks a significant stride in infusing technology into driver education, offering a scalable and efficient approach to instruction.

----

## [2592] IBCA: An Intelligent Platform for Social Insurance Benefit Qualification Status Assessment

**Authors**: *Yuliang Shi, Lin Cheng, Cheng Jiang, Hui Zhang, Guifeng Li, Xiaoli Tang, Han Yu, Zhiqi Shen, Cyril Leung*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30316](https://doi.org/10.1609/aaai.v38i21.30316)

**Abstract**:

Social insurance benefits qualification assessment is an important task to ensure that retirees enjoy their benefits according to the regulations. It also plays a key role in curbing social security frauds. In this paper, we report the deployment of the Intelligent Benefit Certification and Analysis (IBCA) platform, an AI-empowered platform for verifying the status of retirees to ensure proper dispursement of funds in Shandong province, China. Based on an improved Gated Recurrent Unit (GRU) neural network, IBCA aggregates missing value interpolation, temporal information, and global and local feature extraction to perform accurate retiree survival rate prediction. Based on the predicted results, a reliability assessment mechanism based on Variational Auto-Encoder (VAE) and Monte-Carlo Dropout (MC Dropout) is executed to perform reliability assessment. Deployed since November 2019, the IBCA platform has been adopted by 12 cities across the Shandong province, handling over 50 terabytes of data. It has empowered human resources and social services, civil affairs, and health care institutions to collaboratively provide high-quality public services. Under the IBCA platform, the efficiency of resources utilization as well as the accuracy of benefit qualification assessment have been significantly improved. It has helped Dareway Software Co. Ltd earn over RMB 50 million of revenue.

----

## [2593] HiFi-Gas: Hierarchical Federated Learning Incentive Mechanism Enhanced Gas Usage Estimation

**Authors**: *Hao Sun, Xiaoli Tang, Chengyi Yang, Zhenpeng Yu, Xiuli Wang, Qijie Ding, Zengxiang Li, Han Yu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30317](https://doi.org/10.1609/aaai.v38i21.30317)

**Abstract**:

Gas usage estimation plays a critical role in various aspects of the power generation and delivery business, including budgeting, resource planning, and environmental preservation. Federated Learning (FL) has demonstrated its potential in enhancing the accuracy and reliability of gas usage estimation by enabling distributedly owned data to be leveraged, while ensuring privacy and confidentiality. However, to effectively motivate stakeholders to contribute their high-quality local data and computational resources for this purpose, incentive mechanism design is key. In this paper, we report our experience designing and deploying the Hierarchical FL Incentive mechanism for Gas usage estimation (HiFi-Gas) system. It is designed to cater to the unique structure of gas companies and their affiliated heating stations. HiFi-Gas provides effective incentivization in a hierarchical federated learning framework that consists of a horizontal federated learning (HFL) component for effective collaboration among gas companies and multiple vertical federated learning (VFL) components for the gas company and its affiliated heating stations. To motivate active participation and ensure fairness among gas companies and heating stations, we incorporate a multi-dimensional contribution-aware reward distribution function that considers both data quality and model contributions. Since its deployment in the ENN Group in December 2022, HiFi-Gas has successfully provided incentives for gas companies and heating stations to actively participate in FL training, resulting in more than 12% higher average gas usage estimation accuracy and substantial gas procurement cost savings. This implementation marks the first successful deployment of a hierarchical FL incentive approach in the energy industry.

----

## [2594] Promoting Research Collaboration with Open Data Driven Team Recommendation in Response to Call for Proposals

**Authors**: *Siva Likitha Valluru, Biplav Srivastava, Sai Teja Paladi, Siwen Yan, Sriraam Natarajan*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30318](https://doi.org/10.1609/aaai.v38i21.30318)

**Abstract**:

Building teams and promoting collaboration  are two very common business activities. An example of these are seen in the TeamingForFunding problem, where research institutions and researchers are interested to identify collaborative opportunities when applying to funding agencies in response to latter's calls for proposals. We describe a novel deployed system to recommend teams using a variety of AI methods, such that (1) each team achieves the highest possible  skill coverage that is demanded by the opportunity, and (2) the workload of distributing the opportunities is balanced amongst the candidate members. We address these questions by extracting skills latent in open data of proposal calls (demand) and researcher profiles (supply), normalizing them using taxonomies, and creating efficient algorithms that match demand to supply. We create teams to maximize goodness along a novel metric balancing short- and long-term objectives. We validate the success of our algorithms (1) quantitatively, by evaluating the recommended teams using a goodness score and find that more informed methods lead to recommendations of smaller number of teams but higher goodness, and (2) qualitatively, by conducting a large-scale user study at a college-wide level, and demonstrate that users overall found the tool very useful and relevant. Lastly, we evaluate our system in two diverse settings in US and India (of researchers and proposal calls) to establish generality of our approach, and deploy it at a major US university for routine use.

----

## [2595] Multi-Stage Prompting for Next Best Agent Recommendations in Adaptive Workflows

**Authors**: *Prerna Agarwal, Harshit Dave, Jayachandu Bandlamudi, Renuka Sindhgatta, Kushal Mukherjee*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30319](https://doi.org/10.1609/aaai.v38i21.30319)

**Abstract**:

Traditional business processes such as loan processing, order processing, or procurement have a series of steps that are pre-defined at design and executed by enterprise systems. Recent advancements in new-age businesses, however, focus on having adaptive and ad-hoc processes by stitching together a set of functions or steps enabled through autonomous agents. Further, to enable business users to execute a flexible set of steps, there have been works on providing a conversational interface to interact and execute automation. Often, it is necessary to guide the user through the set of possible steps in the process (or workflow). Existing work on recommending the next agent to run relies on historical data. However, with changing workflows and new automation constantly getting added, it is important to provide recommendations without historical data. Additionally, hand-crafted recommendation rules do not scale. The adaptive workflow being a combination of structured and unstructured information, makes it harder to mine. Hence, in this work, we leverage Large Language Models (LLMs) to combine process knowledge with the meta-data of agents to discover NBAs specifically at cold-start. We propose a multi-stage approach that uses existing process knowledge and agent meta-data information to prompt LLM and recommend meaningful next best agent (NBA) based on user utterances.

----

## [2596] A Virtual Driving Instructor That Generates Personalized Driving Lessons Based on Student Skill Level

**Authors**: *J. Fredrik R. Bjørnland, Yrjar Gedde, Johannes Rehm, Irina Reshodko, Odd Erik Gundersen*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30320](https://doi.org/10.1609/aaai.v38i21.30320)

**Abstract**:

Currently, students acquire driving skills by practicing in actual traffic conditions and through direct interactions with an instructor. While one-on-one interactions could be tailored to a student’s learning style and skill level, making them effective for learning, one-on-one interactions are also inefficient, potentially costly, and not standardized with limitations on which traffic situation can be safely taught. For these exact reasons Way AS has developed and commercially deployed a virtual driving instructor that educates students in high-fidelity simulators. In this paper, we present a module, the Lesson generator, that extends the virtual driving instructor to generate personalized lessons for individual students with the goal to practice in a focused and deliberately fashion the skills that need practice for the students to become proficient drivers. A case study is presented, and the path to deployment is discussed.

----

## [2597] Improving Autonomous Separation Assurance through Distributed Reinforcement Learning with Attention Networks

**Authors**: *Marc W. Brittain, Luis E. Alvarez, Kara Breeden*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30321](https://doi.org/10.1609/aaai.v38i21.30321)

**Abstract**:

Advanced Air Mobility (AAM) introduces a new, efficient mode of transportation with the use of vehicle autonomy and electrified aircraft to provide increasingly autonomous transportation between previously underserved markets. Safe and efficient navigation of low altitude aircraft through highly dense environments requires the integration of a multitude of complex observations, such as surveillance, knowledge of vehicle dynamics, and weather. The processing and reasoning on these observations pose challenges due to the various sources of uncertainty in the information while ensuring cooperation with a variable number of aircraft in the airspace. These challenges coupled with the requirement to make safety-critical decisions in real-time rule out the use of conventional separation assurance techniques. We present a decentralized reinforcement learning framework to provide autonomous self-separation capabilities within AAM corridors with the use of speed and vertical maneuvers. The problem is formulated as a Markov Decision Process and solved by developing a novel extension to the sample-efficient, off-policy soft actor-critic (SAC) algorithm. We introduce the use of attention networks for variable-length observation processing and a distributed computing architecture to achieve high training sample throughput as compared to existing approaches. A comprehensive numerical study shows that the proposed framework can ensure safe and efficient separation of aircraft in high density, dynamic environments with various sources of uncertainty.

----

## [2598] Neural Bookmarks: Information Retrieval with Deep Learning and EEG Data

**Authors**: *Glenn Bruns, Michael Haidar*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30322](https://doi.org/10.1609/aaai.v38i21.30322)

**Abstract**:

In neural memory decoding, a concept being mentally recalled is identified using brain data. Recently, the feasibility of neural memory decoding with EEG data has been demonstrated. Here we propose a new application – neural information retrieval – that uses neural memory decoding to allow a document to be retrieved merely by thinking about it. In this paper we describe neural memory decoding, define the application of neural information retrieval, present experimental results related to the practicality of the application, and discuss issues of deployment and data privacy.

----

## [2599] CHRONOS: A Schema-Based Event Understanding and Prediction System

**Authors**: *Maria Chang, Achille Fokoue, Rosario Uceda-Sosa, Parul Awasthy, Ken Barker, Sadhana Kumaravel, Oktie Hassanzadeh, Elton F. S. Soares, Tian Gao, Debarun Bhattacharjya, Radu Florian, Salim Roukos*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30323](https://doi.org/10.1609/aaai.v38i21.30323)

**Abstract**:

Chronological and Hierarchical Reasoning Over Naturally Occurring Schemas (CHRONOS) is a system that combines language model-based natural language processing with symbolic knowledge representations to analyze and make predictions about newsworthy events. CHRONOS consists of an event-centric information extraction pipeline and a complex event schema instantiation and prediction system. Resulting predictions are detailed with arguments, event types from Wikidata, schema-based justifications, and source document provenance. We evaluate our system by its ability to capture the structure of unseen events described in news articles and make plausible predictions as judged by human annotators.

----



[Go to the previous page](AAAI-2024-list12.md)

[Go to the next page](AAAI-2024-list14.md)

[Go to the catalog section](README.md)