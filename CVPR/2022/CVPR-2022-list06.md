## [1000] The Flag Median and FlagIRLS

**Authors**: *Nathan Mankovich, Emily J. King, Chris Peterson, Michael Kirby*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01009](https://doi.org/10.1109/CVPR52688.2022.01009)

**Abstract**:

Finding prototypes (e.g., mean and median) for a dataset is central to a number of common machine learning algorithms. Subspaces have been shown to provide useful, robust representations for datasets of images, videos and more. Since subspaces correspond to points on a Grassmann manifold, one is led to consider the idea of a subspace prototype for a Grassmann-valued dataset. While a number of different subspace prototypes have been described, the calculation of some of these prototypes has proven to be computationally expensive while other prototypes are affected by outliers and produce highly imperfect clustering on noisy data. This work proposes a new subspace prototype, the flag median, and introduces the FlagIRLS algorithm for its calculation. We provide evidence that the flag median is robust to outliers and can be used effectively in algorithms like Linde-Buzo-Grey (LBG) to produce improved clusterings on Grassmannians. Numerical experiments include a synthetic dataset, the MNIST handwritten digits dataset, the Mind's Eye video dataset and the UCF YouTube action dataset. The flag median is compared the other leading algorithms for computing prototypes on the Grassmannian, namely, the l2-median and to the flag mean. We find that using FlagIRLS to compute the flag median converges in 4 iterations on a synthetic dataset. We also see that Grassmannian LBG with a codebook size of 20 and using the flag median produces at least a 10% improvement in cluster purity over Grassmannian LBG using the flag mean or l2-median on the Mind's Eye dataset.

----

## [1001] Learning Fair Classifiers with Partially Annotated Group Labels

**Authors**: *Sangwon Jung, Sanghyuk Chun, Taesup Moon*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01010](https://doi.org/10.1109/CVPR52688.2022.01010)

**Abstract**:

Recently, fairness-aware learning have become increasingly crucial, but most of those methods operate by assuming the availability of fully annotated demographic group labels. We emphasize that such assumption is unrealistic for real-world applications since group label annotations are expensive and can conflict with privacy issues. In this paper, we consider a more practical scenario, dubbed as Algorithmic Group Fairness with the Partially annotated Group labels (Fair-PG). We observe that the existing methods to achieve group fairness perform even worse than the vanilla training, which simply uses full data only with target labels, under Fair-PG. To address this problem, we propose a simple Confidence-based Group Label assignment (CGL) strategy that is readily applicable to any fairness-aware learning method. CGL utilizes an auxiliary group classifier to assign pseudo group labels, where random labels are assigned to low confident samples. We first theoretically show that our method design is better than the vanilla pseudo-labeling strategy in terms of fairness criteria. Then, we empirically show on several benchmark datasets that by combining CGL and the state-of-the-art fairness-aware in-processing methods, the target accuracies and the fairness metrics can be jointly improved compared to the baselines. Furthermore, we convincingly show that CGL enables to naturally augment the given group-labeled dataset with external target label-only datasets so that both accuracy and fairness can be improved. Code is available at https://github.com/naver-ai/cgl_fairness.

----

## [1002] Estimating Structural Disparities for Face Models

**Authors**: *Shervin Ardeshir, Cristina Segalin, Nathan Kallus*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01011](https://doi.org/10.1109/CVPR52688.2022.01011)

**Abstract**:

In machine learning, disparity metrics are often defined by measuring the difference in the performance or outcome of a model, across different sub-populations (groups) of datapoints. Thus, the inputs to disparity quantification consist of a model's predictions y, the ground-truth labels for the predictions y, and group labels g for the data points. Performance of the model for each group is calculated by comparing y and y for the datapoints within a specific group, and as a result, disparity of performance across the different groups can be calculated. In many real world scenarios however, group labels (g) may not be available at scale during training and validation time, or collecting them might not be feasible or desirable as they could often be sensitive information. As a result, evaluating disparity metrics across categorical groups would not be feasible. On the other hand, in many scenarios noisy groupings may be obtainable using some form of a proxy, which would allow measuring disparity metrics across sub-populations. Here we explore performing such analysis on computer vision models trained on human faces, and on tasks such as face attribute prediction and affect estimation. Our experiments indicate that embeddings resulting from an off-the-shelf face recognition model, could meaningfully serve as a proxy for such estimation.

----

## [1003] Estimating Example Difficulty using Variance of Gradients

**Authors**: *Chirag Agarwal, Daniel D'souza, Sara Hooker*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01012](https://doi.org/10.1109/CVPR52688.2022.01012)

**Abstract**:

In machine learning, a question of great interest is understanding what examples are challenging for a model to classify. Identifying atypical examples ensures the safe de-ployment of models, isolates samples that require further human inspection and provides interpretability into model behavior. In this work, we propose Variance of Gradients (VoG) as a valuable and efficient metric to rank data by difficulty and to surface a tractable subset of the most chal-lenging examples for human-in-the-loop auditing. We show that data points with high VoG scores are far more difficult for the model to learn and over-index on corrupted or mem-orized examples. Further, restricting the evaluation to the test set instances with the lowest VoG improves the model's generalization performance. Finally, we show that VoG is a valuable and efficient ranking for out-of-distribution detection.

----

## [1004] Fairness-aware Adversarial Perturbation Towards Bias Mitigation for Deployed Deep Models

**Authors**: *Zhibo Wang, Xiaowei Dong, Henry Xue, Zhifei Zhang, Weifeng Chiu, Tao Wei, Kui Ren*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01013](https://doi.org/10.1109/CVPR52688.2022.01013)

**Abstract**:

Prioritizing fairness is of central importance in artificial intelligence (AI) systems, especially for those societal applications, e.g., hiring systems should recommend applicants equally from different demographic groups, and risk assessment systems must eliminate racism in criminal justice. Existing efforts towards the ethical development of AI systems have leveraged data science to mitigate biases in the training set or introduced fairness principles into the training process. For a deployed AI system, however, it may not allow for retraining or tuning in practice. By contrast, we propose a more flexible approach, i.e., fairness-aware adversarial perturbation (FAAP), which learns to perturb input data to blind deployed models on fairness-related features, e.g., gender and ethnicity. The key advantage is that FAAP does not modify deployed models in terms of param-eters and structures. To achieve this, we design a discriminator to distinguish fairness-related attributes based on latent representations from deployed models. Meanwhile, a perturbation generator is trained against the discriminator, such that no fairness-related features could be extracted from perturbed inputs. Exhaustive experimental evaluation demonstrates the effectiveness and superior performance of the proposed FAAP. In addition, FAAP is validated on real-world commercial deployments (inaccessible to model pa-rameters), which shows the transferability of FAAP, foreseeing the potential of black-box adaptation.

----

## [1005] Fair Contrastive Learning for Facial Attribute Classification

**Authors**: *Sungho Park, Jewook Lee, Pilhyeon Lee, Sunhee Hwang, Dohyung Kim, Hyeran Byun*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01014](https://doi.org/10.1109/CVPR52688.2022.01014)

**Abstract**:

Learning visual representation of high quality is essential for image classification. Recently, a series of contrastive representation learning methods have achieved preeminent success. Particularly, SupCon [18] outperformed the dominant methods based on cross-entropy loss in representation learning. However, we notice that there could be potential ethical risks in supervised contrastive learning. In this paper, we for the first time analyze unfairness caused by supervised contrastive learning and propose a new Fair Supervised Contrastive Loss (FSCL) for fair visual representation learning. Inheriting the philosophy of supervised contrastive learning, it encourages representation of the same class to be closer to each other than that of different classes, while ensuring fairness by penalizing the inclusion of sensitive attribute information in representation. In addition, we introduce a group-wise normalization to diminish the disparities of intra-group compactness and inter-class separability between demographic groups that arouse unfair classification. Through extensive experiments on CelebA and UTK Face, we validate that the proposed method significantly outperforms SupCon and existing state-of-the-art methods in terms of the trade-off between top-l accuracy and fairness. Moreover, our method is robust to the intensity of data bias and effectively works in incomplete supervised settings. Our code is available at https://github.com/sungho-CoolG/FSCL.

----

## [1006] Leveraging Adversarial Examples to Quantify Membership Information Leakage

**Authors**: *Ganesh Del Grosso, Hamid Jalalzai, Georg Pichler, Catuscia Palamidessi, Pablo Piantanida*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01015](https://doi.org/10.1109/CVPR52688.2022.01015)

**Abstract**:

The use of personal data for training machine learning systems comes with a privacy threat and measuring the level of privacy of a model is one of the major challenges in machine learning today. Identifying training data based on a trained model is a standard way of measuring the privacy risks induced by the model. We develop a novel approach to address the problem of membership inference in pattern recognition models, relying on information provided by adversarial examples. The strategy we propose consists of measuring the magnitude of a perturbation necessary to build an adversarial example. Indeed, we argue that this quantity reflects the likelihood of belonging to the training data. Extensive numerical experiments on multivariate data and an array of state-of-the-art target models show that our method performs comparable or even outperforms state-of-the-art strategies, but without requiring any additional training samples.

----

## [1007] Leveling Down in Computer Vision: Pareto Inefficiencies in Fair Deep Classifiers

**Authors**: *Dominik Zietlow, Michael Lohaus, Guha Balakrishnan, Matthäus Kleindessner, Francesco Locatello, Bernhard Schölkopf, Chris Russell*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01016](https://doi.org/10.1109/CVPR52688.2022.01016)

**Abstract**:

Algorithmic fairness is frequently motivated in terms of a trade-off in which overall performance is decreased so as to improve performance on disadvantaged groups where the algorithm would otherwise be less accurate. Contrary to this, we find that applying existing fairness approaches to computer vision improve fairness by degrading the performance of classifiers across all groups (with increased degradation on the best performing groups). Extending the bias-variance decomposition for classification to fairness, we theoretically explain why the majority of fairness methods designed for low capacity models should not be used in settings involving high-capacity models, a scenario common to computer vision. We corroborate this analysis with extensive experimental support that shows that many of the fairness heuristics used in computer vision also degrade performance on the most disadvantaged groups. Building on these insights, we propose an adaptive augmentation strategy that, uniquely, of all methods tested, improves performance for the disadvantaged groups.

----

## [1008] Deep Unlearning via Randomized Conditionally Independent Hessians

**Authors**: *Ronak Mehta, Sourav Pal, Vikas Singh, Sathya N. Ravi*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01017](https://doi.org/10.1109/CVPR52688.2022.01017)

**Abstract**:

Recent legislation has led to interest in machine unlearning, i. e., removing specific training samples from a predictive model as if they never existed in the training dataset. Unlearning may also be required due to corrupted/adversarial data or simply a user's updated privacy requirement. For models which require no training (k-NN), simply deleting the closest original sample can be effective. But this idea is inapplicable to models which learn richer representations. Recent ideas leveraging optimization-based updates scale poorly with the model dimension d, due to inverting the Hessian of the loss function. We use a variant of a new conditional independence coefficient, L-CODEC, to identify a subset of the model parameters with the most semantic overlap on an individual sample level. Our approach completely avoids the need to invert a (possibly) huge matrix. By utilizing a Markov blanket selection, we premise that L-CODEC is also suitable for deep unlearning, as well as other applications in vision. Compared to alternatives, L-CODEC makes approximate unlearning possible in settings that would otherwise be infeasible, including vision models used for face recognition, person reidentification and NLP models that may require unlearning samples identified for exclusion. Code is available at https://github.com/vsingh-group/LCODEC-deep-unlearning

----

## [1009] Equivariance Allows Handling Multiple Nuisance Variables When Analyzing Pooled Neuroimaging Datasets

**Authors**: *Vishnu Suresh Lokhande, Rudrasis Chakraborty, Sathya N. Ravi, Vikas Singh*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01018](https://doi.org/10.1109/CVPR52688.2022.01018)

**Abstract**:

Pooling multiple neuroimaging datasets across institutions often enables improvements in statistical power when evaluating associations (e.g., between risk factors and disease outcomes) that may otherwise be too weak to detect. When there is only a single source of variability (e.g., different scanners), domain adaptation and matching the distributions of representations may suffice in many scenarios. But in the presence of more than one nuisance variable which concurrently influence the measurements, pooling datasets poses unique challenges, e.g., variations in the data can come from both the acquisition method as well as the demographics of participants (gender, age). Invariant representation learning, by itself, is illsuited to fully model the data generation process. In this paper, we show how bringing recent results on equivariant representation learning (for studying symmetries in neural networks) instantiated on structured spaces together with simple use of classical results on causal inference provides an effective practical solution. In particular, we demonstrate how our model allows dealing with more than one nuisance variable under some assumptions and can enable analysis of pooled scientific datasets in scenarios that would otherwise entail removing a large portion of the samples. Our code is available on https://github.com/vsingh-group/DatasetPooling.

----

## [1010] A study on the distribution of social biases in self-supervised learning visual models

**Authors**: *Kirill Sirotkin, Pablo Carballeira, Marcos Escudero-Viñolo*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01019](https://doi.org/10.1109/CVPR52688.2022.01019)

**Abstract**:

Deep neural networks are efficient at learning the data distribution if it is sufficiently sampled. However, they can be strongly biased by non-relevant factors implicitly incorporated in the training data. These include operational biases, such as ineffective or uneven data sampling, but also ethical concerns, as the social biases are implicitly present—even inadvertently, in the training data or explicitly defined in unfair training schedules. In tasks having impact on human processes, the learning of social biases may produce discriminatory, unethical and untrustworthy consequences. It is often assumed that social biases stem from supervised learning on labelled data, and thus, Self-Supervised Learning (SSL) wrongly appears as an efficient and bias-free solution, as it does not require labelled data. However, it was recently proven that a popular SSL method also incorporates biases. In this paper, we study the biases of a varied set of SSL visual models, trained using ImageNet data, using a method and dataset designed by psychological experts to measure social biases. We show that there is a correlation between the type of the SSL model and the number of biases that it incorporates. Furthermore, the results also suggest that this number does not strictly depend on the model's accuracy and changes throughout the network. Finally, we conclude that a careful SSL model selection process can reduce the number of social biases in the deployed model, whilst keeping high performance. The code is available at https://github.com/vpulab/SB-SSL.

----

## [1011] Cross-Modal Perceptionist: Can Face Geometry be Gleaned from Voices?

**Authors**: *Cho-Ying Wu, Chin-Cheng Hsu, Ulrich Neumann*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01020](https://doi.org/10.1109/CVPR52688.2022.01020)

**Abstract**:

This work digs into a root question in human perception: can face geometry be gleaned from one's voices? Previous works that study this question only adopt developments in image synthesis and convert voices into face images to show correlations, but working on the image domain unavoidably involves predicting attributes that voices cannot hint, including facial textures, hairstyles, and backgrounds. We instead investigate the ability to reconstruct 3D faces to concentrate on only geometry, which is much more physiologically grounded. We propose our analysis framework, Cross-Modal Perceptionist, under both supervised and unsupervised learning. First, we construct a dataset, Voxceleb-3D, which extends Voxceleb and includes paired voices and face meshes, making supervised learning possible. Second, we use a knowledge distillation mechanism to study whether face geometry can still be gleaned from voices without paired voices and 3D face data under limited availability of 3D face scans. We break down the core question into four parts and perform visual and numerical analyses as responses to the core question. Our findings echo those in physiology and neuroscience about the correlation between voices and facial structures. The work provides future human-centric cross-modal learning with explainable foundations. See our project page.

----

## [1012] Learning Hierarchical Cross-Modal Association for Co-Speech Gesture Generation

**Authors**: *Xian Liu, Qianyi Wu, Hang Zhou, Yinghao Xu, Rui Qian, Xinyi Lin, Xiaowei Zhou, Wayne Wu, Bo Dai, Bolei Zhou*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01021](https://doi.org/10.1109/CVPR52688.2022.01021)

**Abstract**:

Generating speech-consistent body and gesture movements is a long-standing problem in virtual avatar creation. Previous studies often synthesize pose movement in a holistic manner, where poses of all joints are generated simultaneously. Such a straightforward pipeline fails to generate fine-grained co-speech gestures. One observation is that the hierarchical semantics in speech and the hierarchical structures of human gestures can be naturally described into multiple granularities and associated together. To fully utilize the rich connections between speech audio and human gestures, we propose a novel framework named Hierarchical Audio-to-Gesture (HA2G) for co-speech gesture generation. In HA2G, a Hierarchical Audio Learner extracts audio representations across semantic granularities. A Hierarchical Pose Inferer subsequently renders the entire human pose gradually in a hierarchical manner. To enhance the quality of synthesized gestures, we develop a contrastive learning strategy based on audio-text alignment for better audio representations. Extensive experiments and human evaluation demonstrate that the proposed method renders realistic co-speech gestures and out-performs previous methods in a clear margin. Project page: https://alvinliu0.github.io/projects/HA2G.

----

## [1013] SEEG: Semantic Energized Co-speech Gesture Generation

**Authors**: *Yuanzhi Liang, Qianyu Feng, Linchao Zhu, Li Hu, Pan Pan, Yi Yang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01022](https://doi.org/10.1109/CVPR52688.2022.01022)

**Abstract**:

Talking gesture generation is a practical yet challenging task that aims to synthesize gestures in line with speech. Gestures with meaningful signs can better convey useful information and arouse sympathy in the audience. Current works focus on aligning gestures with the speech rhythms, which are difficult to mine the semantics and model semantic gestures explicitly. This paper proposes a novel semantic Energized Generation (SEEG) method for semantic-aware gesture generation. Our method contains two parts: DEcoupled Mining module (DEM) and Semantic Energizing Module (SEM). DEM decouples the semantic-irrelevant information from inputs and separately mines information for the beat and semantic gestures. SEM conducts semantic learning and produces semantic gestures. Apart from representational similarity, SEM requires the predictions to express the same semantics as the ground truth. Besides, a semantic prompter is designed in SEM to leverage the semantic-aware supervision to predictions. This promotes the networks to learn and generate semantic gestures. Experimental results reported in three metrics on different benchmarks prove that SEEG efficiently mines semantic cues and generates semantic gestures. SEEG outperforms other methods in all semantic-aware evaluations on different datasets. Qualitative evaluations also indicate the superiority of SEEG in semantic expressiveness. Code is available via https://github.com/akira-l/SEEG.

----

## [1014] Mix and Localize: Localizing Sound Sources in Mixtures

**Authors**: *Xixi Hu, Ziyang Chen, Andrew Owens*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01023](https://doi.org/10.1109/CVPR52688.2022.01023)

**Abstract**:

We present a method for simultaneously localizing multiple sound sources within a visual scene. This task requires a model to both group a sound mixture into individual sources, and to associate them with a visual signal. Our method jointly solves both tasks at once, using a formulation inspired by the contrastive random walk of Jabri et al. We create a graph in which images and separated sounds correspond to nodes, and train a random walker to transition between nodes from different modalities with high return probability. The transition probabilities for this walk are determined by an audio-visual similarity metric that is learned by our model. We show through experiments with musical instruments and human speech that our model can successfully localize multiple sounds, outperforming other self-supervised methods.

----

## [1015] Reading to Listen at the Cocktail Party: Multi-Modal Speech Separation

**Authors**: *Akam Rahimi, Triantafyllos Afouras, Andrew Zisserman*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01024](https://doi.org/10.1109/CVPR52688.2022.01024)

**Abstract**:

The goal of this paper is speech separation and enhancement in multi-speaker and noisy environments using a combination of different modalities. Previous works have shown good performance when conditioning on temporal or static visual evidence such as synchronised lip movements or face identity. In this paper, we present a unified framework for multi-modal speech separation and enhancement based on synchronous or asynchronous cues. To that end we make the following contributions: (i) we design a modern Transformer-based architecture tailored to fuse different modalities to solve the speech separation task in the raw waveform domain; (ii) we propose conditioning on the textual content of a sentence alone or in combination with visual information; (iii) we demonstrate the robustness of our model to audio-visual synchronisation offsets; and, (iv) we obtain state-of-the-art performance on the well-established benchmark datasets LRS2 and LRS3.

----

## [1016] IntentVizor: Towards Generic Query Guided Interactive Video Summarization

**Authors**: *Guande Wu, Jianzhe Lin, Cláudio T. Silva*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01025](https://doi.org/10.1109/CVPR52688.2022.01025)

**Abstract**:

The target of automatic video summarization is to create a short skim of the original long video while preserving the major content/events. There is a growing interest in the integration of user queries into video summarization or query-driven video summarization. This video summarization method predicts a concise synopsis of the original video based on the user query, which is commonly represented by the input text. However, two inherent problems exist in this query-driven way. First, the text query might not be enough to describe the exact and diverse needs of the user. Second, the user cannot edit once the summaries are produced, while we assume the needs of the user should be subtle and need to be adjusted interactively. To solve these two problems, we propose IntentVizor, an interactive video summarization framework guided by generic multi-modality queries. The input query that describes the user's needs are not limited to text but also the video snippets. We further represent these multi-modality finer-grained queries as user ‘intent’, which is interpretable, interactable, editable, and can better quantify the user's needs. In this paper, we use a set of the proposed intents to represent the user query and design a new interactive visual analytic interface. Users can interactively control and adjust these mixed-initiative intents to obtain a more satisfying summary through the interface. Also, to improve the summarization quality via video understanding, a novel Granularity-Scalable Ego-Graph Convolutional Networks (GSE-GCN) is proposed. We conduct our experiments on two benchmark datasets. Comparisons with the state-of-the-art methods verify the effectiveness of the proposed framework. Code and dataset are available at https://github.com/jnzs1836/intentvizor.

----

## [1017] M3L: Language-based Video Editing via Multi-Modal Multi-Level Transformers

**Authors**: *Tsu-Jui Fu, Xin Eric Wang, Scott T. Grafton, Miguel P. Eckstein, William Yang Wang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01026](https://doi.org/10.1109/CVPR52688.2022.01026)

**Abstract**:

Video editing tools are widely used nowadays for digital design. Although the demand for these tools is high, the prior knowledge required makes it difficult for novices to get started. Systems that could follow natural language instructions to perform automatic editing would significantly improve accessibility. This paper introduces the language-based video editing (LBVE) task, which allows the model to edit, guided by text instruction, a source video into a target video. LBVE contains two features: 1) the scenario of the source video is preserved instead of generating a completely different video; 2) the semantic is presented differently in the target video, and all changes are controlled by the given instruction. We propose a Multi-Modal Multi-Level Transformer (M
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">3</sup>
L) to carry out LBVE. M
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">3</sup>
L dynamically learns the correspondence between video perception and language semantic at different levels, which benefits both the video understanding and video frame synthesis. We build three new datasets for evaluation, including two diagnostic and one from natural videos with human-labeled text. Extensive experimental results show that M
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">3</sup>
L is effective for video editing and that LBVE can lead to a new field toward vision-and-language research.

----

## [1018] Finding Fallen Objects Via Asynchronous Audio-Visual Integration

**Authors**: *Chuang Gan, Yi Gu, Siyuan Zhou, Jeremy Schwartz, Seth Alter, James Traer, Dan Gutfreund, Joshua B. Tenenbaum, Josh H. McDermott, Antonio Torralba*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01027](https://doi.org/10.1109/CVPR52688.2022.01027)

**Abstract**:

The way an object looks and sounds provide complementary reflections of its physical properties. In many settings cues from vision and audition arrive asynchronously but must be integrated, as when we hear an object dropped on the floor and then must find it. In this paper, we introduce a setting in which to study multi-modal object localization in 3D virtual environments. An object is dropped somewhere in a room. An embodied robot agent, equipped with camera and microphone, must determine what object has been dropped - and where - by combining audio and visual signals with knowledge of the underlying physics. To study this problem, we have generated a large-scale dataset - the Fallen Objects dataset - that includes 8000 instances of 30 physical object categories in 64 rooms. The dataset uses the ThreeDWorld Platform that can simulate physics-based impact sounds and complex physical interactions between objects in a photorealistic setting. As a first step toward addressing this challenge, we develop a set of embodied agent baselines, based on imitation learning, reinforcement learning, and modular planning, and perform an in-depth analysis of the challenge of this new task. This dataset is publicly available
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
Project page: http://fallen-object.csail.mit.edu.

----

## [1019] Weakly Paired Associative Learning for Sound and Image Representations via Bimodal Associative Memory

**Authors**: *Sangmin Lee, Hyung-Il Kim, Yong Man Ro*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01028](https://doi.org/10.1109/CVPR52688.2022.01028)

**Abstract**:

Data representation learning without labels has attracted increasing attention due to its nature that does not require human annotation. Recently, representation learning has been extended to bimodal data, especially sound and image which are closely related to basic human senses. Existing sound and image representation learning methods necessarily require a large number of sound and image with corresponding pairs. Therefore, it is difficult to ensure the effectiveness of the methods in the weakly paired condition, which lacks paired bimodal data. In fact, according to human cognitive studies, the cognitive functions in the human brain for a certain modality can be enhanced by receiving other modalities, even not directly paired ones. Based on the observation, we propose a new problem to deal with the weakly paired condition: How to boost a certain modal representation even by using other unpaired modal data. To address the issue, we introduce a novel bimodal associative memory (BMA-Memory) with key-value switching. It enables to build sound-image association with small paired bimodal data and to boost the built association with the eas-ily obtainable large amount of unpaired data. Through the proposed associative learning, it is possible to reinforce the representation of a certain modality (e.g., sound) even by using other unpaired modal data (e.g., images).

----

## [1020] Egocentric Deep Multi-Channel Audio-Visual Active Speaker Localization

**Authors**: *Hao Jiang, Calvin Murdock, Vamsi Krishna Ithapu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01029](https://doi.org/10.1109/CVPR52688.2022.01029)

**Abstract**:

Augmented reality devices have the potential to enhance human perception and enable other assistive functionalities in complex conversational environments. Effectively capturing the audio-visual context necessary for understanding these social interactions first requires detecting and localizing the voice activities of the device wearer and the surrounding people. These tasks are challenging due to their egocentric nature: the wearer's head motion may cause motion blur, surrounding people may appear in difficult viewing angles, and there may be occlusions, visual clutter, audio noise, and bad lighting. Under these conditions, previous state-of-the-art active speaker detection methods do not give satisfactory results. Instead, we tackle the problem from a new setting using both video and multi-channel microphone array audio. We propose a novel end-to-end deep learning approach that is able to give robust voice activity detection and localization results. In contrast to previous methods, our method localizes active speakers from all possible directions on the sphere, even outside the camera's field of view, while simultaneously detecting the device wearer's own voice activity. Our experiments show that the proposed method gives superior results, can run in real time, and is robust against noise and clutter.

----

## [1021] Audiovisual Generalised Zero-shot Learning with Cross-modal Attention and Language

**Authors**: *Otniel-Bogdan Mercea, Lukas Riesch, A. Sophia Koepke, Zeynep Akata*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01030](https://doi.org/10.1109/CVPR52688.2022.01030)

**Abstract**:

Learning to classify video data from classes not included in the training data, i.e. video-based zero-shot learning, is challenging. We conjecture that the natural alignment between the audio and visual modalities in video data provides a rich training signal for learning discriminative multi-modal representations. Focusing on the relatively underexplored task of audio-visual zero-shot learning, we propose to learn multi-modal representations from audio- visual data using cross-modal attention and exploit textual label embeddings for transferring knowledge from seen classes to unseen classes. Taking this one step further, in our generalised audio-visual zero-shot learning setting, we include all the training classes in the test-time search space which act as distractors and increase the difficulty while making the setting more realistic. Due to the lack of a unified benchmark in this domain, we introduce a (generalised) zero-shot learning benchmark on three audio-visual datasets of varying sizes and difficulty, VGGSound, UCF, and ActivityNet, ensuring that the unseen test classes do not appear in the dataset used for supervised training of the backbone deep models. Comparing multiple relevant and recent methods, we demonstrate that our proposed AVCA model achieves state-of-the-art performance on all three datasets. Code and data are available at https://github.com/ExplainableML/AVCA-GZSL.

----

## [1022] It's Time for Artistic Correspondence in Music and Video

**Authors**: *Dídac Surís, Carl Vondrick, Bryan Russell, Justin Salamon*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01031](https://doi.org/10.1109/CVPR52688.2022.01031)

**Abstract**:

We present an approach for recommending a music track for a given video, and vice versa, based on both their temporal alignment and their correspondence at an artistic level. We propose a self-supervised approach that learns this correspondence directly from data, without any need of human annotations. In order to capture the high-level concepts that are required to solve the task, we propose modeling the long-term temporal context of both the video and the music signals, using Transformer networks for each modality. Experiments show that this approach strongly outperforms alternatives that do not exploit the temporal context. The combination of our contributions improve retrieval accuracy up to 10× over prior state of the art. This strong improvement allows us to introduce a wide range of analyses and applications. For instance, we can condition music retrieval based on visually defined attributes.

----

## [1023] Self-supervised object detection from audio-visual correspondence

**Authors**: *Triantafyllos Afouras, Yuki M. Asano, Francois Fagan, Andrea Vedaldi, Florian Metze*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01032](https://doi.org/10.1109/CVPR52688.2022.01032)

**Abstract**:

We tackle the problem of learning object detectors without supervision. Differently from weakly-supervised object detection, we do not assume image-level class labels. Instead, we extract a supervisory signal from audio-visual data, using the audio component to “teach” the object detector. While this problem is related to sound source localisation, it is considerably harder because the detector must classify the objects by type, enumerate each instance of the object, and do so even when the object is silent. We tackle this problem by first designing a self-supervised framework with a contrastive objective that jointly learns to classify and localise objects. Then, without using any supervision, we simply use these self-supervised labels and boxes to train an image-based object detector. With this, we outperform previous unsupervised and weakly-supervised detectors for the task of object detection and sound source localization. We also show that we can align this detector to ground-truth classes with as little as one label per pseudo-class, and show how our method can learn to detect generic objects that go beyond instruments, such as airplanes and cats.

----

## [1024] More than Words: In-the-Wild Visually-Driven Prosody for Text-to-Speech

**Authors**: *Michael Hassid, Michelle Tadmor Ramanovich, Brendan Shillingford, Miaosen Wang, Ye Jia, Tal Remez*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01033](https://doi.org/10.1109/CVPR52688.2022.01033)

**Abstract**:

In this paper we present VDTTS, a Visually-Driven Text-to-Speech model. Motivated by dubbing, VDTTS takes ad-vantage of video frames as an additional input alongside text, and generates speech that matches the video signal. We demonstrate how this allows VDTTS to, unlike plain TTS models, generate speech that not only has prosodic variations like natural pauses and pitch, but is also synchronized to the input video. Experimentally, we show our model produces well-synchronized outputs, approaching the video-speech synchronization quality of the ground-truth, on several challenging benchmarks including “in-the-wild” content from VoxCeleb2. Supplementary demo videos demonstrating video-speech synchronization, robustness to speaker ID swapping, and prosody, presented at the project page.
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
Project page: http://google-research.github.io/lingvo-lab/vdtts

----

## [1025] ObjectFolder 20: A Multisensory Object Dataset for Sim2Real Transfer

**Authors**: *Ruohan Gao, Zilin Si, Yen-Yu Chang, Samuel Clarke, Jeannette Bohg, Li Fei-Fei, Wenzhen Yuan, Jiajun Wu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01034](https://doi.org/10.1109/CVPR52688.2022.01034)

**Abstract**:

Objects play a crucial role in our everyday activities. Though multisensory object-centric learning has shown great potential lately, the modeling of objects in prior work is rather unrealistic. ObjectFolder 1.0 is a recent dataset that introduces 100 virtualized objects with visual, acoustic, and tactile sensory data. However, the dataset is small in scale and the multisensory data is of limited quality, hampering generalization to real-world scenarios. We present ObjectFolder 2.0, a large-scale, multisensory dataset of common household objects in the form of implicit neural representations that significantly enhances ObjectFolder 1.0 in three aspects. First, our dataset is 10 times larger in the amount of objects and orders of magnitude faster in rendering time. Second, we significantly improve the multisensory rendering quality for all three modalities. Third, we show that models learned from virtual objects in our dataset successfully transfer to their real-world counterparts in three challenging tasks: object scale estimation, contact localization, and shape reconstruction. ObjectFolder 2.0 offers a new path and testbed for multisensory learning in computer vision and robotics. The dataset is available at https://github.com/rhgao/ObjectFolder.

----

## [1026] A Probabilistic Graphical Model Based on Neural-symbolic Reasoning for Visual Relationship Detection

**Authors**: *Dongran Yu, Bo Yang, Qianhao Wei, Anchen Li, Shirui Pan*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01035](https://doi.org/10.1109/CVPR52688.2022.01035)

**Abstract**:

This paper aims to leverage symbolic knowledge to improve the performance and interpretability of the Visual Relationship Detection (VRD) models. Existing VRD methods based on deep learning suffer from the problems of poor performance on insufficient labeled examples and lack of interpretability. To overcome the aforementioned weaknesses, we integrate symbolic knowledge into deep learning models and propose a bi-level probabilistic graphical reasoning framework called BPGR. Specifically, in the high-level structure, we take the objects and relationships detected by the VRD model as hidden variables (reasoning results); In the low-level structure of BPGR, we use Markov Logic Networks (MLNs) to project First-Order Logic (FOL) as observed variables (symbolic knowledge) to correct error reasoning results. We adopt a variational EM algorithm for optimization. Experiments results show that our BPGR improves the performance of the VRD models. In particular, BPGR can also provide easy-to-understand insights for reasoning results to show interpretability.

----

## [1027] Diffusion Autoencoders: Toward a Meaningful and Decodable Representation

**Authors**: *Konpat Preechakul, Nattanat Chatthee, Suttisak Wizadwongsa, Supasorn Suwajanakorn*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01036](https://doi.org/10.1109/CVPR52688.2022.01036)

**Abstract**:

Diffusion probabilistic models (DPMs) have achieved remarkable quality in image generation that rivals GANs'. But unlike GANs, DPMs use a set of latent variables that lack semantic meaning and cannot serve as a useful representation for other tasks. This paper explores the possibility of using DPMs for representation learning and seeks to extract a meaningful and decodable representation of an input image via autoencoding. Our key idea is to use a learnable encoder for discovering the high-level semantics, and a DPM as the decoder for modeling the remaining stochastic variations. Our method can encode any image into a two-part latent code where the first part is semantically meaningful and linear, and the second part captures stochastic details, allowing near-exact reconstruction. This capability enables challenging applications that currently foil GAN-based methods, such as attribute manipulation on real images. We also show that this two-level encoding improves denoising efficiency and naturally facilitates various downstream tasks including few-shot conditional sampling. Please visit our page: https://Diff-AE.github.io/

----

## [1028] Polymorphic-GAN: Generating Aligned Samples across Multiple Domains with Learned Morph Maps

**Authors**: *Seung Wook Kim, Karsten Kreis, Daiqing Li, Antonio Torralba, Sanja Fidler*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01037](https://doi.org/10.1109/CVPR52688.2022.01037)

**Abstract**:

Modern image generative models show remarkable sample quality when trained on a single domain or class of objects. In this work, we introduce a generative adversarial network that can simultaneously generate aligned image samples from multiple related domains. We leverage the fact that a variety of object classes share common attributes, with certain geometric differences. We propose Polymorphic-GAN which learns shared features across all domains and a per-domain morph layer to morph shared features according to each domain. In contrast to previous works, our framework allows simultaneous modelling of images with highly varying geometries, such as images of human faces, painted and artistic faces, as well as multiple different animal faces. We demonstrate that our model produces aligned samples for all domains and show how it can be used for applications such as segmentation transfer and cross-domain image editing, as well as training in low-data regimes. Additionally, we apply our Polymorphic-GAN on image-to-image translation tasks and show that we can greatly surpass previous approaches in cases where the geometric differences between domains are large.

----

## [1029] Polarity Sampling: Quality and Diversity Control of Pre-Trained Generative Networks via Singular Values

**Authors**: *Ahmed Imtiaz Humayun, Randall Balestriero, Richard G. Baraniuk*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01038](https://doi.org/10.1109/CVPR52688.2022.01038)

**Abstract**:

We present Polarity Sampling, a theoretically justified plug-and-play method for controlling the generation quality and diversity of any pre-trained deep generative network (DGN). Leveraging the fact that DGNs are, or can be ap-proximated by, continuous piecewise affine splines, we derive the analytical DGN output space distribution as a function of the product of the DGN's Jacobian singular values raised to a power p. We dub p the polarity param-eter and prove that p focuses the DGN sampling on the modes (p < 0) or anti-modes (p > 0) of the DGN output-space probability distribution. We demonstrate that nonzero polarity values achieve a better precision-recall (quality-diversity) Pareto frontier than standard methods, such as truncation, for a number of state-of-the-art DGNs. We also present quantitative and qualitative results on the improve-ment of overall generation quality (e.g., in terms of the Fréchet Inception Distance) for a number of state-of-the-art DGNs, including StyleGAN3, BigGAN-deep, NVAE, for different conditional and unconditional image generation tasks. In particular, Polarity Sampling redefines the state-of-the-art for StyleGAN2 on the FFHQ Dataset to FID 2.57, StyleGAN2 on the LSUN Car Dataset to FID 2.27 and Style-GAN3 on the AFHQv2 Dataset to FID 3.95. Colab Demo.

----

## [1030] Ensembling Off-the-shelf Models for GAN Training

**Authors**: *Nupur Kumari, Richard Zhang, Eli Shechtman, Jun-Yan Zhu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01039](https://doi.org/10.1109/CVPR52688.2022.01039)

**Abstract**:

The advent of large-scale training has produced a cor-nucopia of powerful visual recognition models. However, generative models, such as GANs, have traditionally been trained from scratch in an unsupervised manner. Can the collective “knowledge” from a large bank ofpretrained vision models be leveraged to improve GAN training? If so, with so many models to choose from, which one(s) should be selected, and in what manner are they most effective? We find that pretrained computer vision models can signif-icantly improve performance when used in an ensemble of discriminators. Notably, the particular subset of selected models greatly affects performance. We propose an effective selection mechanism, by probing the linear separability between real and fake samples in pretrained model embed-dings, choosing the most accurate model, and progressively adding it to the discriminator ensemble. Interestingly, our method can improve GAN training in both limited data and large-scale settings. Given only 10k training samples, our FID on LSUN Catmatches the StyleGAN2 trained on 1.6M images. On the full dataset, our method improves FID by 1.5 to 2x on cat, church, and horse categories of LSUN.

----

## [1031] Marginal Contrastive Correspondence for Guided Image Generation

**Authors**: *Fangneng Zhan, Yingchen Yu, Rongliang Wu, Jiahui Zhang, Shijian Lu, Changgong Zhang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01040](https://doi.org/10.1109/CVPR52688.2022.01040)

**Abstract**:

Exemplar-based image translation establishes dense correspondences between a conditional input and an exemplar (from two different domains) for leveraging detailed exemplar styles to achieve realistic image translation. Existing work builds the cross-domain correspondences implicitly by minimizing feature- wise distances across the two domains. Without explicit exploitation of domain-invariant features, this approach may not reduce the domain gap effectively which often leads to sub-optimal correspon-dences and image translation. We design a Marginal Contrastive Learning Network (MCL-Net) that explores contrastive learning to learn domain-invariant features for realistic exemplar-based image translation. Specifically, we design an innovative marginal contrastive loss that guides to establish dense correspondences explicitly. Nevertheless, building correspondence with domain-invariant semantics alone may impair the texture patterns and lead to degraded texture generation. We thus design a Self-Correlation Map (SCM) that incorporates scene structures as auxiliary information which improves the built correspondences substantially. Quantitative and qualitative experiments on multifarious image translation tasks show that the proposed method outperforms the state-of-the-art consistently.

----

## [1032] GRAM: Generative Radiance Manifolds for 3D-Aware Image Generation

**Authors**: *Yu Deng, Jiaolong Yang, Jianfeng Xiang, Xin Tong*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01041](https://doi.org/10.1109/CVPR52688.2022.01041)

**Abstract**:

3D-aware image generative modeling aims to generate 3D-consistent images with explicitly controllable camera poses. Recent works have shown promising results by training neural radiance field (NeRF) generators on unstructured 2D images, but still cannot generate highly-realistic images with fine details. A critical reason is that the high memory and computation cost of volumetric representation learning greatly restricts the number of point samples for radiance integration during training. Deficient sampling not only limits the expressive power of the generator to handle fine details but also impedes effective GAN training due to the noise caused by unstable Monte Carlo sampling. We propose a novel approach that regulates point sampling and radiance field learning on 2D manifolds, embodied as a set of learned implicit surfaces in the 3D volume. For each viewing ray, we calculate ray-surface intersections and accumulate their radiance generated by the network. By training and rendering such radiance mani folds, our generator can produce high quality images with realistic fine details and strong visual 3D consistency. 
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
Project page: https://yudeng.github.io/GRAM/

----

## [1033] High-Resolution Image Synthesis with Latent Diffusion Models

**Authors**: *Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, Björn Ommer*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01042](https://doi.org/10.1109/CVPR52688.2022.01042)

**Abstract**:

By decomposing the image formation process into a sequential application of denoising autoencoders, diffusion models (DMs) achieve state-of-the-art synthesis results on image data and beyond. Additionally, their formulation allows for a guiding mechanism to control the image generation process without retraining. However, since these models typically operate directly in pixel space, optimization of powerful DMs often consumes hundreds of GPU days and inference is expensive due to sequential evaluations. To enable DM training on limited computational resources while retaining their quality and flexibility, we apply them in the latent space of powerful pretrained autoencoders. In contrast to previous work, training diffusion models on such a representation allows for the first time to reach a near-optimal point between complexity reduction and detail preservation, greatly boosting visual fidelity. By introducing cross-attention layers into the model architecture, we turn diffusion models into powerful and flexible generators for general conditioning inputs such as text or bounding boxes and high-resolution synthesis becomes possible in a convolutional manner. Our latent diffusion models (LDMs) achieve new state of the art scores for image inpainting and class-conditional image synthesis and highly competitive performance on various tasks, including unconditional image generation, text-to-image synthesis, and super-resolution, while significantly reducing computational requirements compared to pixel-based DMs.

----

## [1034] Vector Quantized Diffusion Model for Text-to-Image Synthesis

**Authors**: *Shuyang Gu, Dong Chen, Jianmin Bao, Fang Wen, Bo Zhang, Dongdong Chen, Lu Yuan, Baining Guo*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01043](https://doi.org/10.1109/CVPR52688.2022.01043)

**Abstract**:

We present the vector quantized diffusion (VQ-Diffusion) model for text-to-image generation. This method is based on a vector quantized variational autoencoder (VQ-VAE) whose latent space is modeled by a conditional variant of the recently developed Denoising Diffusion Probabilistic Model (DDPM). We find that this latent-space method is well-suited for text-to-image generation tasks because it not only eliminates the unidirectional bias with existing methods but also allows us to incorporate a mask-and-replace diffusion strategy to avoid the accumulation of errors, which is a serious problem with existing methods. Our experiments show that the VQ-Diffusion produces significantly better text-to-image generation results when compared with conventional autoregressive (AR) models with similar numbers of parameters. Compared with previous GAN-based text-to-image methods, our VQ-Diffusion can handle more complex scenes and improve the synthesized image quality by a large margin. Finally, we show that the image generation computation in our method can be made highly efficient by reparameterization. With traditional AR methods, the text-to-image generation time increases linearly with the output image resolution and hence is quite time consuming even for normal size images. The VQ-Diffusion allows us to achieve a better trade-off between quality and speed. Our experiments indicate that the VQ-Diffusion model with the reparameterization is fifteen times faster than traditional AR methods while achieving a better image quality. The code and models are available at https://github.com/cientgu/VQ-Diffusion.

----

## [1035] ManiTrans: Entity-Level Text-Guided Image Manipulation via Token-wise Semantic Alignment and Generation

**Authors**: *Jianan Wang, Guansong Lu, Hang Xu, Zhenguo Li, Chunjing Xu, Yanwei Fu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01044](https://doi.org/10.1109/CVPR52688.2022.01044)

**Abstract**:

Existing text-guided image manipulation methods aim to modify the appearance of the image or to edit a few objects in a virtual or simple scenario, which is far from practical application. In this work, we study a novel task on text-guided image manipulation on the entity level in the real world. The task imposes three basic requirements, (1) to edit the entity consistent with the text descriptions, (2) to preserve the text-irrelevant regions, and (3) to merge the manipulated entity into the image naturally. To this end, we propose a new transformer-based framework based on the two-stage image synthesis method, namely ManiTrans, which can not only edit the appearance of entities but also generate new entities corresponding to the text guidance. Our framework incorporates a semantic alignment module to locate the image regions to be manipulated, and a semantic loss to help align the relationship between the vision and language. We conduct extensive experiments on the real datasets, CUB, Oxford, and COCO datasets to verify that our method can distinguish the relevant and irrelevant regions and achieve more precise and flexible manipulation compared with baseline methods.

----

## [1036] Dataset Distillation by Matching Training Trajectories

**Authors**: *George Cazenavette, Tongzhou Wang, Antonio Torralba, Alexei A. Efros, Jun-Yan Zhu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01045](https://doi.org/10.1109/CVPR52688.2022.01045)

**Abstract**:

Dataset distillation is the task of synthesizing a small dataset such that a model trained on the synthetic set will match the test accuracy of the model trained on the full dataset. In this paper, we propose a new formulation that optimizes our distilled data to guide networks to a similar state as those trained on real data across many training steps. Given a network, we train it for several iterations on our distilled data and optimize the distilled data with respect to the distance between the synthetically trained parameters and the parameters trained on real data. To efficiently obtain the initial and target network parameters for large-scale datasets, we pre-compute and store training trajectories of expert networks trained on the real dataset. Our method handily outperforms existing methods and also allows us to distill higher-resolution visual data.

----

## [1037] Continual Predictive Learning from Videos

**Authors**: *Geng Chen, Wendong Zhang, Han Lu, Siyu Gao, Yunbo Wang, Mingsheng Long, Xiaokang Yang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01046](https://doi.org/10.1109/CVPR52688.2022.01046)

**Abstract**:

Predictive learning ideally builds the world model of physical processes in one or more given environments. Typical setups assume that we can collect data from all environments at all times. In practice, however, different prediction tasks may arrive sequentially so that the environments may change persistently throughout the training procedure. Can we develop predictive learning algorithms that can deal with more realistic, non-stationary physical environments? In this paper, we study a new continual learning problem in the context of video prediction, and observe that most existing methods suffer from severe catastrophic forgetting in this setup. To tackle this problem, we propose the continual predictive learning (CPL) approach, which learns a mixture world model via predictive experience replay and performs test-time adaptation with non-parametric task inference. We construct two new benchmarks based on RoboNet and KTH, in which different tasks correspond to different physical robotic environments or human actions. Our approach is shown to effectively mitigate forgetting and remarkably outperform the naíve combinations of previous art in video prediction and continual learning.

----

## [1038] Motion-Adjustable Neural Implicit Video Representation

**Authors**: *Long Mai, Feng Liu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01047](https://doi.org/10.1109/CVPR52688.2022.01047)

**Abstract**:

Implicit neural representation (INR) has been successful in representing static images. Contemporary image-based INR, with the use of Fourier-based positional encoding, can be viewed as a mapping from sinusoidal patterns with different frequencies to image content. Inspired by that view, we hypothesize that it is possible to generate temporally varying content with a single image-based INR model by displacing its input sinusoidal patterns over time. By exploiting the relation between the phase information in sinusoidal functions and their displacements, we incorporate into the conventional image-based INR model a phase-varying positional encoding module, and couple it with a phase-shift generation module that determines the phase-shift values at each frame. The model is trained end-to-end on a video to jointly determine the phase-shift values at each time with the mapping from the phase-shifted sinusoidal functions to the corresponding frame, enabling an implicit video representation. Experiments on a wide range of videos suggest that such a model is capable of learning to interpret phase-varying positional embeddings into the corresponding time-varying content. More importantly, we found that the learned phase-shift vectors tend to capture meaningful temporal and motion information from the video. In particular, manipulating the phase-shift vectors induces meaningful changes in the temporal dynamics of the resulting video, enabling non-trivial temporal and motion editing effects such as temporal interpolation, motion magnification, motion smoothing, and video loop detection.

----

## [1039] Splicing ViT Features for Semantic Appearance Transfer

**Authors**: *Narek Tumanyan, Omer Bar-Tal, Shai Bagon, Tali Dekel*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01048](https://doi.org/10.1109/CVPR52688.2022.01048)

**Abstract**:

We present a method for semantically transferring the visual appearance of one natural image to another. Specifically, our goal is to generate an image in which objects in a source structure image are “painted” with the visual appearance of their semantically related objects in a target appearance image. Our method works by training a generator given only a single structure/appearance image pair as input. To integrate semantic information into our framework—a pivotal component in tackling this task-our key idea is to leverage a pre-trained and fixed Vision Transformer (ViT) model which serves as an external semantic prior. Specifically, we derive novel representations of structure and appearance extracted from deep ViT features, untwisting them from the learned self-attention modules. We then establish an objective function that splices the desired structure and appearance representations, interweaving them together in the space of ViT features. Our framework, which we term “Splice”, does not involve adversarial training, nor does it require any additional input information such as semantic segmentation or correspondences, and can generate high resolution results, e.g., work in HD. We demonstrate high quality results on a variety of in-the-wild image pairs, under significant variations in the number of objects, their pose and appearance.

----

## [1040] MAT: Mask-Aware Transformer for Large Hole Image Inpainting

**Authors**: *Wenbo Li, Zhe Lin, Kun Zhou, Lu Qi, Yi Wang, Jiaya Jia*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01049](https://doi.org/10.1109/CVPR52688.2022.01049)

**Abstract**:

Recent studies have shown the importance of modeling long-range interactions in the inpainting problem. To achieve this goal, existing approaches exploit either standalone attention techniques or transformers, but usually under a low resolution in consideration of computational cost. In this paper, we present a novel transformer-based model for large hole inpainting, which unifies the merits of transformers and convolutions to efficiently process high-resolution images. We carefully design each component of our framework to guarantee the high fidelity and diversity of recovered images. Specifically, we customize an inpainting-oriented transformer block, where the attention module aggregates non-local information only from partial valid tokens, indicated by a dynamic mask. Extensive experiments demonstrate the state-of-the-art performance of the new model on multiple benchmark datasets. Code is released at https://github.com/fenglinglwb/MAT.

----

## [1041] Day-to-Night Image Synthesis for Training Nighttime Neural ISPs

**Authors**: *Abhijith Punnappurath, Abdullah Abuolaim, Abdelrahman Abdelhamed, Alex Levinshtein, Michael S. Brown*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01050](https://doi.org/10.1109/CVPR52688.2022.01050)

**Abstract**:

Many flagship smartphone cameras now use a dedicated neural image signal processor (ISP) to render noisy raw sensor images to the final processed output. Training night-mode ISP networks relies on large-scale datasets of image pairs with: (1) a noisy raw image captured with a short exposure and a high ISO gain; and (2) a ground truth low-noise raw image captured with a long exposure and low ISO that has been rendered through the ISP. Capturing such image pairs is tedious and time-consuming, requiring careful setup to ensure alignment between the image pairs. In addition, ground truth images are often prone to motion blur due to the long exposure. To address this problem, we propose a method that synthesizes nighttime images from day-time images. Daytime images are easy to capture, exhibit low-noise (even on smartphone cameras) and rarely suffer from motion blur. We outline a processing framework to convert daytime raw images to have the appearance of realistic nighttime raw images with different levels of noise. Our procedure allows us to easily produce aligned noisy and clean nighttime image pairs. We show the effectiveness of our synthesis framework by training neural ISPs for nightmode rendering. Furthermore, we demonstrate that using our synthetic nighttime images together with small amounts of real data (e.g., 5% to 10%) yields performance almost on par with training exclusively on real nighttime images. Our dataset and code are available at https://github.com/SamsungLabs/day-to-night.

----

## [1042] Smooth-Swap: A Simple Enhancement for Face-Swapping with Smoothness

**Authors**: *Jiseob Kim, Jihoon Lee, Byoung-Tak Zhang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01051](https://doi.org/10.1109/CVPR52688.2022.01051)

**Abstract**:

Face-swapping models have been drawing attention for their compelling generation quality, but their complex architectures and loss functions often require careful tuning for successful training. We propose a new face-swapping model called ‘Smooth-Swap’, which excludes complex handcrafted designs and allows fast and stable training. The main idea of Smooth-Swap is to build smooth identity embedding that can provide stable gradients for identity change. Unlike the one used in previous models trained for a purely discriminative task, the proposed embedding is trained with a supervised contrastive loss promoting a smoother space. With improved smoothness, Smooth-Swap suffices to be composed of a generic U-Net-based generator and three basic loss functions, a far simpler design compared with the previous models. Extensive experiments on face-swapping benchmarks (FFHQ, 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$Face-Forensics++$</tex>
) and face images in the wild show that our model is also quantitatively and qualitatively comparable or even superior to the existing methods.

----

## [1043] Few-Shot Head Swapping in the Wild

**Authors**: *Changyong Shu, Hemao Wu, Hang Zhou, Jiaming Liu, Zhibin Hong, Changxing Ding, Junyu Han, Jingtuo Liu, Errui Ding, Jingdong Wang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01052](https://doi.org/10.1109/CVPR52688.2022.01052)

**Abstract**:

The head swapping task aims at flawlessly placing a source head onto a target body, which is of great importance to various entertainment scenarios. While face swapping has drawn much attention, the task of head swapping has rarely been explored, particularly under the few-shot setting. It is inherently challenging due to its unique needs in head modeling and background blending. In this paper, we present the Head Swapper (HeSer), which achieves few-shot head swapping in the wild through two delicately de-signed modules. Firstly, a Head2Head Aligner is devised to holistically migrate pose and expression information from the target to the source head by examining multi-scale in-formation. Secondly, to tackle the challenges of skin color variations and head-background mismatches in the swapping procedure, a Head2Scene Blender is introduced to si-multaneously modify facial skin color and fill mismatched gaps on the background around the head. Particularly, seamless blending is achieved with the help of a Semantic-Guided Color Reference Creation procedure and a Blending UNet. Extensive experiments demonstrate that the proposed method produces superior head swapping results on a variety of scenes.

----

## [1044] ClothFormer: Taming Video Virtual Try-on in All Module

**Authors**: *Jianbin Jiang, Tan Wang, He Yan, Junhui Liu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01053](https://doi.org/10.1109/CVPR52688.2022.01053)

**Abstract**:

The task of video virtual try-on aims to fit the target clothes to a person in the video with spatio-temporal consistency. Despite tremendous progress of image virtual tryon, they lead to inconsistency between frames when applied to videos. Limited work also explored the task of video-based virtual try-on but failed to produce visually pleasing and temporally coherent results. Moreover, there are two other key challenges: 1) how to generate accurate warping when occlusions appear in the clothing region; 2) how to generate clothes and non-target body parts (e.g. arms, neck) in harmony with the complicated background; To address them, we propose a novel video virtual try-on framework, ClothFormer, which successfully synthesizes realistic, harmonious, and spatio-temporal consistent results in complicated environment. In particular, ClothFormer involves three major modules. First, a two-stage anti-occlusion warping module that predicts an accurate dense flow mapping between the body regions and the clothing regions. Second, an appearance-flow tracking module utilizes ridge regression and optical flow correction to smooth the dense flow sequence and generate a temporally smooth warped clothing sequence. Third, a dual-stream transformer ex-tracts and fuses clothing textures, person features, and en-vironment information to generate realistic try-on videos. Through rigorous experiments, we demonstrate that our method highly surpasses the baselines in terms of synthesized video quality both qualitatively and quantitatively
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">†</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">†</sup>
The code and all demos are available at https://github.com/luxiangju-PersonAI/ClothFormer.

----

## [1045] A-ViT: Adaptive Tokens for Efficient Vision Transformer

**Authors**: *Hongxu Yin, Arash Vahdat, José M. Álvarez, Arun Mallya, Jan Kautz, Pavlo Molchanov*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01054](https://doi.org/10.1109/CVPR52688.2022.01054)

**Abstract**:

We introduce A - ViT, a method that adaptively adjusts the inference cost of vision transformer (ViT) for images of different complexity. A - ViT achieves this by automatically reducing the number of tokens in vision transformers that are processed in the network as inference proceeds. We refor-mulate Adaptive Computation Time (ACT [17]) for this task, extending halting to discard redundant spatial tokens. The appealing architectural properties of vision transformers enables our adaptive token reduction mechanism to speed up inference without modifying the network architecture or inference hardware. We demonstrate that A - ViT requires no extra parameters or sub-network for halting, as we base the learning of adaptive halting on the original network parameters. We further introduce distributional prior regularization that stabilizes training compared to prior ACT approaches. On the image classification task (ImageNet1K), we show that our proposed A - ViT yields high efficacy in filtering informative spatial features and cutting down on the overall compute. The proposed method improves the throughput of DeiT-Tiny by 62% and DeiT-Small by 38% with only 0.3% accuracy drop, outperforming prior art by a large margin.

----

## [1046] MetaFormer is Actually What You Need for Vision

**Authors**: *Weihao Yu, Mi Luo, Pan Zhou, Chenyang Si, Yichen Zhou, Xinchao Wang, Jiashi Feng, Shuicheng Yan*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01055](https://doi.org/10.1109/CVPR52688.2022.01055)

**Abstract**:

Transformers have shown great potential in computer vision tasks. A common belief is their attention-based token mixer module contributes most to their competence. However, recent works show the attention-based module in transformers can be replaced by spatial MLPs and the resulted models still perform quite well. Based on this observation, we hypothesize that the general architecture of the transformers, instead of the specific token mixer module, is more essential to the model's performance. To verify this, we deliberately replace the attention module in transformers with an embarrassingly simple spatial pooling operator to conduct only basic token mixing. Surprisingly, we observe that the derived model, termed as PoolFormer, achieves competitive performance on multiple computer vision tasks. For example, on ImageNet-1K, PoolFormer achieves 82.1 % top-1 accuracy, surpassing well-tuned vision transformer/MLP-like baselines DeiT-B/ResMLP-B24 by 0.3%/1.1% accuracy with 35%/52% fewer parameters and 49%/61% fewer MACs. The effectiveness of Pool-Former verifies our hypothesis and urges us to initiate the concept of “MetaFormer”, a general architecture abstracted from transformers without specifying the token mixer. Based on the extensive experiments, we argue that MetaFormer is the key player in achieving superior results for recent transformer and MLP-like models on vision tasks. This work calls for more future research dedicated to improving MetaFormer instead of focusing on the token mixer modules. Additionally, our proposed PoolFormer could serve as a starting baseline for future MetaFormer architecture design.

----

## [1047] Reversible Vision Transformers

**Authors**: *Karttikeya Mangalam, Haoqi Fan, Yanghao Li, Chao-Yuan Wu, Bo Xiong, Christoph Feichtenhofer, Jitendra Malik*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01056](https://doi.org/10.1109/CVPR52688.2022.01056)

**Abstract**:

We present Reversible Vision Transformers, a memory efficient architecture design for visual recognition. By decoupling the GPU memory footprint from the depth of the model, Reversible Vision Transformers enable memory efficient scaling of transformer architectures. We adapt two popular models, namely Vision Transformer and Multiscale Vision Transformers, to reversible variants and benchmark extensively across both model sizes and tasks of image classification, object detection and video classification. Reversible Vision Transformers achieve a reduced memory footprint of up to 15.5× at identical model complexity, parameters and accuracy, demonstrating the promise of reversible vision transformers as an efficient backbone for resource limited training regimes. Finally, we find that the additional computational burden of recomputing activations is more than overcome for deeper models, where throughput can increase up to 3.9 × over their non-reversible counterparts. Code and models are available at https://github.com/facebookresearch/mvit.

----

## [1048] Learned Queries for Efficient Local Attention

**Authors**: *Moab Arar, Ariel Shamir, Amit H. Bermano*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01057](https://doi.org/10.1109/CVPR52688.2022.01057)

**Abstract**:

Vision Transformers (ViT) serve as powerful vision models. Unlike convolutional neural networks, which dominated vision research in previous years, vision transformers enjoy the ability to capture long-range dependencies in the data. Nonetheless, an integral part of any transformer architecture, the self-attention mechanism, suffers from high latency and inefficient memory utilization, making it less suitable for high-resolution input images. To alleviate these shortcomings, hierarchical vision models locally employ self-attention on non-interleaving windows. This relaxation reduces the complexity to be linear in the input size; however, it limits the cross-window interaction, hurting the model performance. In this paper, we propose a new shift-invariant local attention layer, called query and attend (QnA), that aggregates the input locally in an overlapping manner, much like convolutions. The key idea behind QnA is to introduce learned queries, which allow fast and efficient implementation. We verify the effectiveness of our layer by incorporating it into a hierarchical vision transformer model. We show improvements in speed and memory complexity while achieving comparable accuracy with state-of-the-art models. Finally, our layer scales especially well with window size, requiring up to x10 less memory while being up to x5 faster than existing methods. The code is publicly available at https://github.com/moabarar/qna.

----

## [1049] Shunted Self-Attention via Multi-Scale Token Aggregation

**Authors**: *Sucheng Ren, Daquan Zhou, Shengfeng He, Jiashi Feng, Xinchao Wang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01058](https://doi.org/10.1109/CVPR52688.2022.01058)

**Abstract**:

Recent Vision Transformer (ViT) models have demonstrated encouraging results across various computer vision tasks, thanks to its competence in modeling long-range de-pendencies of image patches or tokens via self-attention. These models, however, usually designate the similar receptive fields of each token feature within each layer. Such a constraint inevitably limits the ability of each self-attention layer in capturing multi-scale features, thereby leading to performance degradation in handling images with multiple objects of different scales. To address this issue, we propose a novel and generic strategy, termed shunted self-attention (SSA), that allows ViTs to model the attentions at hybrid scales per attention layer. The key idea of SSA is to inject heterogeneous receptive field sizes into tokens: before computing the self-attention matrix, it selectively merges tokens to represent larger object features while keeping certain tokens to preserve fine-grained features. This novel merging scheme enables the self-attention to learn relationships between objects with different sizes, and simultaneously reduces the token numbers and the computational cost. Extensive experiments across various tasks demonstrate the superiority of SSA. Specifically, the SSA-based transformer achieve 84.0% Top-1 accuracy and out-performs the state-of-the-art Focal Transformer on Ima-geNet with only half of the model size and computation cost, and surpasses Focal Transformer by 1.3 mAP on COCO and 2.9 mIOU on ADE20K under similar parameter and computation cost. Code has been released at https://github.com/OliverRensulShunted-Transformer.

----

## [1050] Automatic Relation-aware Graph Network Proliferation

**Authors**: *Shaofei Cai, Liang Li, Xinzhe Han, Jiebo Luo, Zheng-Jun Zha, Qingming Huang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01059](https://doi.org/10.1109/CVPR52688.2022.01059)

**Abstract**:

Graph neural architecture search has sparked much attention as Graph Neural Networks (GNNs) have shown powerful reasoning capability in many relational tasks. However, the currently used graph search space overem-phasizes learning node features and neglects mining hierarchical relational information. Moreover, due to diverse mechanisms in the message passing, the graph search space is much larger than that of CNNs. This hinders the straightforward application of classical search strategies for exploring complicated graph search space. We propose Automatic Relation-aware Graph Network Proliferation (ARGNP) for efficiently searching GNNs with a relation-guided message passing mechanism. Specifically, we first devise a novel dual relation-aware graph search space that comprises both node and relation learning operations. These operations can extract hierarchical node/relational information and provide anisotropic guidance for message passing on a graph. Second, analogous to cell proliferation, we design a network proliferation search paradigm to progressively determine the GNN architectures by iteratively performing network division and differentiation. The experiments on six datasets for four graph learning tasks demonstrate that GNNs produced by our method are superior to the current state-of-the-art hand-crafted and search-based GNNs. Codes are available at https://github.com/phython96/ARGNP.

----

## [1051] β-DARTS: Beta-Decay Regularization for Differentiable Architecture Search

**Authors**: *Peng Ye, Baopu Li, Yikang Li, Tao Chen, Jiayuan Fan, Wanli Ouyang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01060](https://doi.org/10.1109/CVPR52688.2022.01060)

**Abstract**:

Neural Architecture Search (NAS) has attracted increasingly more attention in recent years because of its capability to design deep neural network automatically. Among them, differential NAS approaches such as DARTS, have gained popularity for the search efficiency. However, they suffer from two main issues, the weak robustness to the performance collapse and the poor generalization ability of the searched architectures. To solve these two problems, a simple-but-efficient regularization method, termed as Beta-Decay, is proposed to regularize the DARTS-based NAS searching process. Specifically, Beta-Decay regularization can impose constraints to keep the value and variance of activated architecture parameters from too large. Furthermore, we provide in-depth theoretical analysis on how it works and why it works. Experimental results on NAS-Bench-201 show that our proposed method can help to stabilize the searching process and makes the searched network more transferable across different datasets. In addition, our search scheme shows an outstanding property of being less dependent on training time and data. Comprehensive experiments on a variety of search spaces and datasets validate the effectiveness of the proposed method. The code is available at https://github.com/Sunshine-Ye/Beta-DARTS.

----

## [1052] Distribution Consistent Neural Architecture Search

**Authors**: *Junyi Pan, Chong Sun, Yizhou Zhou, Ying Zhang, Chen Li*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01061](https://doi.org/10.1109/CVPR52688.2022.01061)

**Abstract**:

Recent progress on neural architecture search (NAS) has demonstrated exciting results on automating deep network architecture designs. In order to overcome the unaffordable complexity of training each candidate architecture from scratch, the state-of-the-art one-shot NAS approaches adopt a weight-sharing strategy to improve training efficiency. Although the computational cost is greatly reduced, such oneshot process introduces a severe weight coupling problem that largely degrades the evaluation accuracy of each candidate. The existing approaches often address the problem by shrinking the search space, model distillation, or fewshot training. Instead, in this paper, we propose a novel distribution consistent one-shot neural architecture search algorithm. We first theoretically investigate how the weight coupling problem affects the network searching performance from a parameter distribution perspective, and then propose a novel supernet training strategy with a Distribution Consistent Constraint that can provide a good measurement for the extent to which two architectures can share weights. Our strategy optimizes the supernet through iteratively inferring network weights and corresponding local sharing states. Such joint optimization of supernet's weights and topologies can diminish the discrepancy between the weights inherited from the supernet and the ones that are trained with a stand-alone model. As a result, it enables a more accurate model evaluation phase and leads to a better searching performance. We conduct extensive experiments on benchmark datasets with multiple searching spaces. The resulting architecture achieves superior performance over the current state-of-the-art NAS algorithms with comparable search costs, which demonstrates the efficacy of our approach.

----

## [1053] Training-free Transformer Architecture Search

**Authors**: *Qinqin Zhou, Kekai Sheng, Xiawu Zheng, Ke Li, Xing Sun, Yonghong Tian, Jie Chen, Rongrong Ji*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01062](https://doi.org/10.1109/CVPR52688.2022.01062)

**Abstract**:

Recently, Vision Transformer (ViT) has achieved remarkable success in several computer vision tasks. The progresses are highly relevant to the architecture design, then it is worthwhile to propose Transformer Architecture Search (TAS) to search for better ViTs automatically. However, current TAS methods are time-consuming and existing zero-cost proxies in CNN do not generalize well to the ViT search space according to our experimental observations. In this paper, for the first time, we investigate how to conduct TAS in a training-free manner and devise an effective training-free TAS (TF-TAS) scheme. Firstly, we observe that the properties of multi-head self-attention (MSA) and multi-layer perceptron (MLP) in ViTs are quite different and that the synaptic diversity of MSA affects the performance notably. Secondly, based on the observation, we devise a modular strategy in TF-TAS that evaluates and ranks ViT architectures from two theoretical perspectives: synaptic diversity and synaptic saliency, termed as DSS-indicator. With DSS-indicator, evaluation results are strongly corre-lated with the test accuracies of ViT models. Experimental results demonstrate that our TF- TAS achieves a competitive performance against the state-of-the-art manually or automatically design ViT architectures, and it promotes the searching efficiency in ViT search space greatly: from about 24 GPU days to less than 0.5 GPU days. Moreover, the proposed DSS-indicator outperforms the existing cutting-edge zero-cost approaches (e.g., TE-score and NASWOT).

----

## [1054] TeachAugment: Data Augmentation Optimization Using Teacher Knowledge

**Authors**: *Teppei Suzuki*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01063](https://doi.org/10.1109/CVPR52688.2022.01063)

**Abstract**:

Optimization of image transformation functions for the purpose of data augmentation has been intensively studied. In particular, adversarial data augmentation strategies, which search augmentation maximizing task loss, show significant improvement in the model generalization for many tasks. However, the existing methods require careful parameter tuning to avoid excessively strong deformations that take away image features critical for acquiring generalization. In this paper, we propose a data augmentation optimization method based on the adversarial strategy called TeachAugment, which can produce informative transformed images to the model without requiring careful tuning by leveraging a teacher model. Specifically, the augmentation is searched so that augmented images are adversarial for the target model and recognizable for the teacher model. We also propose data augmentation using neural networks, which simplifies the search space design and allows for updating of the data augmentation using the gradient method. We show that TeachAugment outperforms existing methods in experiments of image classification, semantic segmentation, and unsupervised representation learning tasks.

----

## [1055] Knowledge Distillation via the Target-aware Transformer

**Authors**: *Sihao Lin, Hongwei Xie, Bing Wang, Kaicheng Yu, Xiaojun Chang, Xiaodan Liang, Gang Wang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01064](https://doi.org/10.1109/CVPR52688.2022.01064)

**Abstract**:

Knowledge distillation becomes a de facto standard to improve the performance of small neural networks. Most of the previous works propose to regress the representational features from the teacher to the student in a one-to-one spatial matching fashion. However, people tend to overlook the fact that, due to the architecture differences, the semantic information on the same spatial location usually vary. This greatly undermines the underlying assumption of the one-to-one distillation approach. To this end, we propose a novel one-to-all spatial matching knowledge distillation approach. Specifically, we allow each pixel of the teacher feature to be distilled to all spatial locations of the student features given its similarity, which is generated from a target-aware transformer. Our approach surpasses the state-of-the-art methods by a significant margin on various computer vision benchmarks, such as ImageNet, Pascal VOC and COCOStuff10k. Code is available at https://github.com/sihaoevery/TaT.

----

## [1056] Knowledge distillation: A good teacher is patient and consistent

**Authors**: *Lucas Beyer, Xiaohua Zhai, Amélie Royer, Larisa Markeeva, Rohan Anil, Alexander Kolesnikov*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01065](https://doi.org/10.1109/CVPR52688.2022.01065)

**Abstract**:

There is a growing discrepancy in computer vision between large-scale models that achieve state-of-the-art performance and models that are affordable in practical applications. In this paper we address this issue and significantly bridge the gap between these two types of models. Throughout our empirical investigation we do not aim to necessarily propose a new method, but strive to identify a robust and effective recipe for making state-of-the-art large scale models affordable in practice. We demonstrate that, when performed correctly, knowledge distillation can be a powerful tool for reducing the size of large models without compromising their performance. In particular, we uncover that there are certain implicit design choices, which may drastically affect the effectiveness of distillation. Our key contribution is the explicit identification of these design choices, which were not previously articulated in the literature. We back up our findings by a comprehensive empirical study, demonstrate compelling results on a wide range of vision datasets and, in particular, obtain a state-of-the-art ResNet-50 model for ImageNet, which achieves 82.8% top-1 accuracy.

----

## [1057] An Image Patch is a Wave: Phase-Aware Vision MLP

**Authors**: *Yehui Tang, Kai Han, Jianyuan Guo, Chang Xu, Yanxi Li, Chao Xu, Yunhe Wang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01066](https://doi.org/10.1109/CVPR52688.2022.01066)

**Abstract**:

In the field of computer vision, recent works show that a pure MLP architecture mainly stacked by fully-connected layers can achieve competing performance with CNN and transformer. An input image of vision MLP is usually split into multiple tokens (patches), while the existing MLP models directly aggregate them with fixed weights, neglecting the varying semantic information of tokens from different images. To dynamically aggregate tokens, we propose to represent each token as a wave function with two parts, amplitude and phase. Amplitude is the original feature and the phase term is a complex value changing according to the semantic contents of input images. Introducing the phase term can dynamically modulate the relationship between tokens and fixed weights in MLP. Based on the wave-like token representation, we establish a novel Wave-MLP architecture for vision tasks. Extensive experiments demonstrate that the proposed Wave-MLP is superior to the state-of-the-art MLP architectures on various vision tasks such as image classification, object detection and semantic segmentation. The source code is available at https://github.com/huawei-noah/CV-Backbones/tree/master/wavemlp_pytorch and https://gitee.com/mindspore/models/tree/master/research/cv/wave_mlp.

----

## [1058] Dynamic MLP for Fine-Grained Image Classification by Leveraging Geographical and Temporal Information

**Authors**: *Lingfeng Yang, Xiang Li, Renjie Song, Borui Zhao, Juntian Tao, Shihao Zhou, Jiajun Liang, Jian Yang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01067](https://doi.org/10.1109/CVPR52688.2022.01067)

**Abstract**:

Fine-grained image classification is a challenging computer vision task where various species share similar visual appearances, resulting in misclassification if merely based on visual clues. Therefore, it is helpful to leverage additional information, e.g., the locations and dates for data shooting, which can be easily accessible but rarely exploited. In this paper, we first demonstrate that existing multimodal methods fuse multiple features only on a single dimension, which essentially has insufficient help in feature discrimination. To fully explore the potential of multimodal information, we propose a dynamic MLP on top of the image representation, which interacts with multimodal features at a higher and broader dimension. The dynamic MLP is an efficient structure parameterized by the learned embeddings of variable locations and dates. It can be regarded as an adaptive nonlinear projection for generating more discriminative image representations in visual tasks. To our best knowledge, it is the first attempt to explore the idea of dynamic networks to exploit multimodal information in fine-grained image classification tasks. Extensive experiments demonstrate the effectiveness of our method. The t-SNE algorithm visually indicates that our technique improves the recognizability of image representations that are visually similar but with different categories. Furthermore, among published works across multiple fine-grained datasets, dynamic MLP consistently achieves SOTA results
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
https://paperswithcode.com/dataset/inaturalist and takes third place in the iNaturalist challenge at FGVC8
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">2</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">2</sup>
https://www.kaggle.com/c/inaturalist-2021/leaderboard. Code is available at httpsr//glthub.com/megvii-research/DynamicMLPForFinegrained.

----

## [1059] Controllable Dynamic Multi-Task Architectures

**Authors**: *Dripta S. Raychaudhuri, Yumin Suh, Samuel Schulter, Xiang Yu, Masoud Faraki, Amit K. Roy-Chowdhury, Manmohan Chandraker*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01068](https://doi.org/10.1109/CVPR52688.2022.01068)

**Abstract**:

Multi-task learning commonly encounters competition for resources among tasks, specifically when model capac-ity is limited. This challenge motivates models which al-low control over the relative importance of tasks and total compute cost during inference time. In this work, we pro-pose such a controllable multi-task network that dynami-cally adjusts its architecture and weights to match the de-sired task preference as well as the resource constraints. In contrast to the existing dynamic multi-task approaches that adjust only the weights within a fixed architecture, our approach affords the flexibility to dynamically control the total computational cost and match the user-preferred task importance better. We propose a disentangled training of two hype rnetwo rks, by exploiting task affinity and a novel branching regularized loss, to take input prefer-ences and accordingly predict tree-structured models with adapted weights. Experiments on three multi-task bench-marks, namely PASCAL-Context, NYU-v2, and CIFAR-100, show the efficacy of our approach. Project page is available at https://www.nec-labs.com/-mas/DYMU.

----

## [1060] Grounded Language-Image Pre-training

**Authors**: *Liunian Harold Li, Pengchuan Zhang, Haotian Zhang, Jianwei Yang, Chunyuan Li, Yiwu Zhong, Lijuan Wang, Lu Yuan, Lei Zhang, Jenq-Neng Hwang, Kai-Wei Chang, Jianfeng Gao*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01069](https://doi.org/10.1109/CVPR52688.2022.01069)

**Abstract**:

This paper presents a grounded language-image pretraining (GLIP) model for learning object-level, language-aware, and semantic-rich visual representations. GLIP unifies object detection and phrase grounding for pre-training. The unification brings two benefits: 1) it allows GLIP to learn from both detection and grounding data to improve both tasks and bootstrap a good grounding model; 2) GLIP can leverage massive image-text pairs by generating grounding boxes in a self-training fashion, making the learned representations semantic-rich. In our experiments, we pre-train GLIP on 27M grounding data, including 3M human-annotated and 24M web-crawled image-text pairs. The learned representations demonstrate strong zero-shot and few-shot transferability to various object-level recognition tasks. 1) When directly evaluated on COCO and LVIS (without seeing any images in COCO during pre-training), GLIP achieves 49.8 AP and 26.9 AP, respectively, surpassing many supervised baselines.
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
Supervised baselines on COCO object detection: Faster-RCNN w/ ResNet50 (40.2) or ResNet101 (42.0), and DyHead w/ Swin-Tiny (49.7). 2) After fine-tuned on COCO, GLIP achieves 60.8 AP on val and 61.5 AP on test-dev, surpassing prior SoTA. 3) When transferred to 13 downstream object detection tasks, a 1-shot GLIP rivals with a fully-supervised Dynamic Head. Code will be released at https://github.com/microsoft/GLIP.

----

## [1061] ZZ-Net: A Universal Rotation Equivariant Architecture for 2D Point Clouds

**Authors**: *Georg Bökman, Fredrik Kahl, Axel Flinth*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01070](https://doi.org/10.1109/CVPR52688.2022.01070)

**Abstract**:

In this paper, we are concerned with rotation equivariance on 2D point cloud data. We describe a particular set of functions able to approximate any continuous rotation equivariant and permutation invariant function. Based on this result, we propose a novel neural network architecture for processing 2D point clouds and we prove its universality for approximating functions exhibiting these symmetries. We also show how to extend the architecture to accept a set of 2D-2D correspondences as indata, while maintaining similar equivariance properties. Experiments are presented on the estimation of essential matrices in stereo vision.

----

## [1062] CADTransformer: Panoptic Symbol Spotting Transformer for CAD Drawings

**Authors**: *Zhiwen Fan, Tianlong Chen, Peihao Wang, Zhangyang Wang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01071](https://doi.org/10.1109/CVPR52688.2022.01071)

**Abstract**:

Understanding 2D computer-aided design (CAD) drawings plays a crucial role for creating 3D prototypes in architecture, engineering and construction (AEC) industries. The task of automated panoptic symbol spotting, i.e., to spot and parse both countable object instances (windows, doors, tables, etc.) and uncountable stuff (wall, railing, etc.) from CAD drawings, has recently drawn interests from the computer vision community. Unfortunately, the highly irregular ordering and orientations set major roadblocks for this task. Existing methods, based on convolutional neural networks (CNNs) and/or graph neural networks (GNNs), regress instance bounding boxes in the pixel domain and then convert the predictions into symbols. In this paper, we present a novel framework named CAD Transformer, that can painlessly modify existing vision transformer (ViT) backbones to tackle the above limitations for the panoptic symbol spotting task. CADTransformer tokenizes directly from the set of graphical primitives in CAD drawings, and correspondingly optimizes line-grained semantic and instance symbol spotting altogether by a pair of prediction heads. The backbone is further enhanced with a few plug-and-play modifications, including a neighborhood aware self-attention, hierarchical feature aggregation, and graphic entity position encoding, to bake in the structure prior while optimizing the efficiency. Besides, a new data augmentation method, termed Random Layer, is proposed by the layer-wise separation and recombination of a CAD drawing. Overall, CADTransformer significantly boosts the previous state-of-the-art from 0.595 to 0.685 in the panoptic quality (PQ) metric, on the recently released FloorPlanCAD dataset. We further demonstrate that our model can spot symbols with irregular shapes and arbitrary orientations. Our codes are available in https://github.com/VITA-Group/CADTransformer.

----

## [1063] Adversarial Parametric Pose Prior

**Authors**: *Andrey Davydov, Anastasia Remizova, Victor Constantin, Sina Honari, Mathieu Salzmann, Pascal Fua*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01072](https://doi.org/10.1109/CVPR52688.2022.01072)

**Abstract**:

The Skinned Multi-Person Linear (SMPL) model represents human bodies by mapping pose and shape parameters to body meshes. However, not all pose and shape parameter values yield physically-plausible or even realistic body meshes. In other words, SMPL is under-constrained and may yield invalid results. We propose learning a prior that restricts the SMPL parameters to values that produce realistic poses via adversarial training. We show that our learned prior covers the diversity of the real-data distribution, facilitates optimization for 3D reconstruction from 2D keypoints, and yields better pose estimates when used for regression from images. For all these tasks, it outperforms the state-of-the-art VAE-based approach to constraining the SMPL parameters. The code will be made available at https://github.com/cvlab-epfl/adv_param_pose_prior.

----

## [1064] Temporal Feature Alignment and Mutual Information Maximization for Video-Based Human Pose Estimation

**Authors**: *Zhenguang Liu, Runyang Feng, Haoming Chen, Shuang Wu, Yixing Gao, Yunjun Gao, Xiang Wang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01073](https://doi.org/10.1109/CVPR52688.2022.01073)

**Abstract**:

Multi-frame human pose estimation has long been a compelling and fundamental problem in computer vision. This task is challenging due to fast motion and pose occlusion that frequently occur in videos. State-of-the-art methods strive to incorporate additional visual evidences from neighboring frames (supporting frames) to facilitate the pose estimation of the current frame (key frame). One aspect that has been obviated so far, is the fact that current methods directly aggregate unaligned contexts across frames. The spatial-misalignment between pose features of the current frame and neighboring frames might lead to unsatisfactory results. More importantly, existing approaches build upon the straightforward pose estimation loss, which unfortunately cannot constrain the network to fully leverage useful information from neighboring frames. To tackle these problems, we present a novel hierarchical alignment framework, which leverages coarse-to-fine deformations to progressively update a neighboring frame to align with the current frame at the feature level. We further propose to explicitly supervise the knowledge extraction from neighboring frames, guaranteeing that useful complementary cues are extracted. To achieve this goal, we theoretically analyzed the mutual information between the frames and arrived at a loss that maximizes the task-relevant mutual information. These allow us to rank No.1 in the Multi-frame Person Pose Estimation Challenge on benchmark dataset PoseTrack2017, and obtain state-of-the-art performance on benchmarks Sub-JHMDB and Pose-Track2018. Our code is released at https://github.com/Pose-Group/FAMI-Pose, hoping that it will be useful to the community.

----

## [1065] PoseTriplet: Co-evolving 3D Human Pose Estimation, Imitation, and Hallucination under Self-supervision

**Authors**: *Kehong Gong, Bingbing Li, Jianfeng Zhang, Tao Wang, Jing Huang, Michael Bi Mi, Jiashi Feng, Xinchao Wang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01074](https://doi.org/10.1109/CVPR52688.2022.01074)

**Abstract**:

Existing self-supervised 3D human pose estimation schemes have largely relied on weak supervisions like consistency loss to guide the learning, which, inevitably, leads to inferior results in real-world scenarios with unseen poses. In this paper, we propose a novel self-supervised approach that allows us to explicitly generate 2D-3D pose pairs for augmenting supervision, through a self-enhancing dual-loop learning framework. This is made possible via introducing a reinforcement-learning-based imitator, which is learned jointly with a pose estimator alongside a pose hallucinator; the three components form two loops during the training process, complementing and strengthening one another. Specifically, the pose estimator transforms an input 2D pose sequence to a low-fidelity 3D output, which is then enhanced by the imitator that enforces physical constraints. The refined 3D poses are subsequently fed to the hallucinator for producing even more diverse data, which are, in turn, strengthened by the imitator and further utilized to train the pose estimator. Such a co-evolution scheme, in practice, enables training a pose estimator on self-generated motion data without relying on any given 3D data. Extensive experiments across various benchmarks demonstrate that our approach yields encouraging results significantly outperforming the state of the art and, in some cases, even on par with results of fully-supervised methods. Notably, it achieves 89.1% 3D PCK on MPI-INF-3DHP under self-supervised cross-dataset evaluation setup, improving upon the previous best self-supervised method [16], [26] by 8.6%.

----

## [1066] Generalizable Human Pose Triangulation

**Authors**: *Kristijan Bartol, David Bojanic, Tomislav Petkovic*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01075](https://doi.org/10.1109/CVPR52688.2022.01075)

**Abstract**:

We address the problem of generalizability for multi-view 3D human pose estimation. The standard approach is to first detect 2D keypoints in images and then apply triangulation from multiple views. Even though the existing methods achieve remarkably accurate 3D pose estimation on public benchmarks, most of them are limited to a single spatial camera arrangement and their number. Several methods address this limitation but demonstrate significantly degraded performance on novel views. We propose a stochastic framework for human pose triangulation and demonstrate a superior generalization across different camera arrangements on two public datasets. In addition, we apply the same approach to the fundamental matrix estimation problem, showing that the proposed method can successfully apply to other computer vision problems. The stochastic framework achieves more than 8.8% improvement on the 3D pose estimation task, compared to the state-of-the-art, and more than 30% improvement for fundamental matrix estimation, compared to a standard algorithm.

----

## [1067] GLAMR: Global Occlusion-Aware Human Mesh Recovery with Dynamic Cameras

**Authors**: *Ye Yuan, Umar Iqbal, Pavlo Molchanov, Kris Kitani, Jan Kautz*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01076](https://doi.org/10.1109/CVPR52688.2022.01076)

**Abstract**:

We present an approach for 3D global human mesh recovery from monocular videos recorded with dynamic cameras. Our approach is robust to severe and long-term occlusions and tracks human bodies even when they go outside the camera's field of view. To achieve this, we first propose a deep generative motion infiller, which autoregressively infills the body motions of occluded humans based on visible motions. Additionally, in contrast to prior work, our approach reconstructs human meshes in consistent global coordinates even with dynamic cameras. Since the joint reconstruction of human motions and camera poses is underconstrained, we propose a global trajectory predictor that generates global human trajectories based on local body movements. Using the predicted trajectories as anchors, we present a global optimization framework that refines the predicted trajectories and optimizes the camera poses to match the video evidence such as 2D keypoints. Experiments on challenging indoor and in-the-wild datasets with dynamic cameras demonstrate that the proposed approach outperforms prior methods significantly in terms of motion infilling and global mesh recovery.

----

## [1068] Bailando: 3D Dance Generation by Actor-Critic GPT with Choreographic Memory

**Authors**: *Li Siyao, Weijiang Yu, Tianpei Gu, Chunze Lin, Quan Wang, Chen Qian, Chen Change Loy, Ziwei Liu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01077](https://doi.org/10.1109/CVPR52688.2022.01077)

**Abstract**:

Driving 3D characters to dance following a piece of music is highly challenging due to the spatial constraints applied to poses by choreography norms. In addition, the generated dance sequence also needs to maintain temporal coherency with different music genres. To tackle these challenges, we propose a novel music-to-dance framework, Bailando, with two powerful components: 1) a choreographic memory that learns to summarize meaningful dancing units from 3D pose sequence to a quantized codebook, 2) an actor-critic Generative Pre-trained Transformer (GPT) that composes these units to a fluent dance coherent to the music. With the learned choreographic memory, dance generation is realized on the quantized units that meet high choreography standards, such that the generated dancing sequences are confined within the spatial constraints. To achieve synchronized alignment between diverse motion tempos and music beats, we introduce an actor-critic-based reinforcement learning scheme to the GPT with a newly-designed beat-align reward function. Extensive experiments on the standard benchmark demonstrate that our proposed framework achieves state-of-the-art performance both qualitatively and quantitatively. Notably, the learned choreographic memory is shown to discover human-interpretable dancing-style poses in an unsupervised manner. Code and video demo are available at https://github.com/lisiyao21/Bailando/

----

## [1069] Contextual Instance Decoupling for Robust Multi-Person Pose Estimation

**Authors**: *Dongkai Wang, Shiliang Zhang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01078](https://doi.org/10.1109/CVPR52688.2022.01078)

**Abstract**:

Crowded scenes make it challenging to differentiate persons and locate their pose keypoints. This paper proposes the Contextual Instance Decoupling (CID), which presents a new pipeline for multi-person pose estimation. Instead of relying on person bounding boxes to spatially differentiate persons, CID decouples persons in an image into multiple instance-aware feature maps. Each of those feature maps is hence adopted to infer keypoints for a specific person. Compared with bounding box detection, CID is differentiable and robust to detection errors. Decoupling persons into different feature maps allows to isolate distractions from other persons, and explore context cues at scales larger than the bounding box size. Experiments show that CID outperforms previous multi-person pose estimation pipelines on crowded scenes pose estimation benchmarks in both accuracy and efficiency. For instance, it achieves 71.3% AP on CrowdPose, outperforming the recent single-stage DEKR by 5.6%, the bottom-up CenterAttention by 3.7%, and the top-down JC-SPPE by 5.3%. This advantage sustains on the commonly used COCO benchmark
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">†</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">†</sup>
Code is available at https://github.com/kennethwdk/CID.

----

## [1070] End-to-End Multi-Person Pose Estimation with Transformers

**Authors**: *Dahu Shi, Xing Wei, Liangqi Li, Ye Ren, Wenming Tan*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01079](https://doi.org/10.1109/CVPR52688.2022.01079)

**Abstract**:

Current methods of multi-person pose estimation typically treat the localization and association of body joints separately. In this paper, we propose the first fully end-to-end multi-person Pose Estimation framework with TRansformers, termed PETR. Our method views pose estimation as a hierarchical set prediction problem and effectively removes the need for many hand-crafted modules like RoI cropping, NMS and grouping post-processing. In PETR, multiple pose queries are learned to directly reason a set of full-body poses. Then a joint decoder is utilized to further refine the poses by exploring the kinematic relations between body joints. With the attention mechanism, the proposed method is able to adaptively attend to the features most relevant to target keypoints, which largely overcomes the feature misalignment difficulty in pose estimation and improves the performance considerably. Extensive experiments on the MS COCO and CrowdPose benchmarks show that PETR plays favorably against state-of-the-art approaches in terms of both accuracy and efficiency. The code and models are available at https://github.com/hikvision-research/opera.

----

## [1071] Meta Agent Teaming Active Learning for Pose Estimation

**Authors**: *Jia Gong, Zhipeng Fan, Qiuhong Ke, Hossein Rahmani, Jun Liu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01080](https://doi.org/10.1109/CVPR52688.2022.01080)

**Abstract**:

The existing pose estimation approaches often require a large number of annotated images to attain good estimation performance, which are laborious to acquire. To reduce the human efforts on pose annotations, we propose a novel Meta Agent Teaming Active Learning (MATAL) framework to actively select and label informative images for effective learning. Our MATAL formulates the image selection procedure as a Markov Decision Process and learns an optimal sampling policy that directly maximizes the performance of the pose estimator based on the reward. Our framework consists of a novel state-action representation as well as a multi-agent team to enable batch sampling in the active learning procedure. The framework could be effectively optimized via Meta-Optimization to accelerate the adaptation to the gradually expanded labeled data during deployment. Finally, we show experimental results on both human hand and body pose estimation benchmark datasets and demonstrate that our method significantly outperforms all baselines continuously under the same amount of annotation budget. Moreover, to obtain similar pose estimation accuracy, our MATAL framework can save around 40% labeling efforts on average compared to state-of-the-art active learning frameworks.

----

## [1072] Keypoint Transformer: Solving Joint Identification in Challenging Hands and Object Interactions for Accurate 3D Pose Estimation

**Authors**: *Shreyas Hampali, Sayan Deb Sarkar, Mahdi Rad, Vincent Lepetit*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01081](https://doi.org/10.1109/CVPR52688.2022.01081)

**Abstract**:

We propose a robust and accurate method for estimating the 3D poses of two hands in close interaction from a single color image. This is a very challenging problem, as large occlusions and many confusions between the joints may happen. State-of-the-art methods solve this problem by regressing a heatmap for each joint, which requires solving two problems simultaneously: localizing the joints and recognizing them. In this work, we propose to separate these tasks by relying on a CNN to first localize joints as 2D keypoints, and on self-attention between the CNN features at these keypoints to associate them with the corresponding hand joint. The resulting architecture, which we call “Keypoint Transformer”, is highly efficient as it achieves state-of-the-art performance with roughly half the number of model parameters on the InterHand2.6M dataset. We also show it can be easily extended to estimate the 3D pose of an object manipulated by one or two hands with high performance. Moreover, we created a new dataset of more than 75,000 images of two hands manipulating an object fully annotated in 3D and will make it publicly available.

----

## [1073] Not All Tokens Are Equal: Human-centric Visual Analysis via Token Clustering Transformer

**Authors**: *Wang Zeng, Sheng Jin, Wentao Liu, Chen Qian, Ping Luo, Wanli Ouyang, Xiaogang Wang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01082](https://doi.org/10.1109/CVPR52688.2022.01082)

**Abstract**:

Vision transformers have achieved great successes in many computer vision tasks. Most methods generate vision tokens by splitting an image into a regular and fixed grid and treating each cell as a token. However, not all regions are equally important in human-centric vision tasks, e.g., the human body needs a fine representation with many tokens, while the image background can be modeled by a few tokens. To address this problem, we propose a novel Vision Transformer, called Token Clustering Transformer (TCFormer), which merges tokens by progressive clustering, where the tokens can be merged from different locations with flexible shapes and sizes. The tokens in TCFormer can not only focus on important areas but also adjust the token shapes to fit the semantic concept and adopt a fine resolution for regions containing critical details, which is beneficial to capturing detailed information. Extensive experiments show that TCFormer consistently outperforms its counterparts on different challenging human-centric tasks and datasets, including whole-body pose estimation on COCO-WholeBody and 3D human mesh reconstruction on 3DPW. Code is available at https://github.com/zengwang430521/TCFormer.git.

----

## [1074] Occlusion-robust Face Alignment using A Viewpoint-invariant Hierarchical Network Architecture

**Authors**: *Congcong Zhu, Xintong Wan, Shaorong Xie, Xiaoqiang Li, Yinzheng Gu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01083](https://doi.org/10.1109/CVPR52688.2022.01083)

**Abstract**:

The occlusion problem heavily degrades the localization performance of face alignment. Most current solutions for this problem focus on annotating new occlusion data, introducing boundary estimation, and stacking deeper models to improve the robustness of neural networks. However, the performance degradation of models remains under extreme occlusion (i.e. average occlusion of over 50%) because of missing a large amount of facial context information. We argue that exploring neural networks to model the facial hierarchies is a more promising method for dealing with extreme occlusion. Surprisingly, in recent studies, little effort has been devoted to representing the facial hierarchies using neural networks. This paper proposes a new network architecture called GlomFace to model the facial hierarchies against various occlusions, which draws inspiration from the viewpoint-invariant hierarchy of facial structure. Specifically, GlomFace is functionally divided into two modules: the part-whole hierarchical module and the whole-part hierarchical module. The former captures the part-whole hierarchical dependencies of facial parts to suppress multi-scale occlusion information, whereas the latter injects structural reasoning into neural networks by building the whole-part hierarchical relations among facial parts. As a result, GlomFace has a clear topological interpretation due to its correspondence to the facial hierarchies. Extensive experimental results indicate that the proposed GlomFace performs comparably to existing state-of-the-art methods, especially in cases of extreme occlusion. Models are available at https://github.com/zhuccly/GlomFace-Face-Alignment.

----

## [1075] LASER: LAtent SpacE Rendering for 2D Visual Localization

**Authors**: *Zhixiang Min, Naji Khosravan, Zachary Bessinger, Manjunath Narayana, Sing Bing Kang, Enrique Dunn, Ivaylo Boyadzhiev*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01084](https://doi.org/10.1109/CVPR52688.2022.01084)

**Abstract**:

We present LASER, an image-based Monte Carlo Localization (MCL) framework for 2D floor maps. LASER introduces the concept of latent space rendering, where 2D pose hypotheses on the floor map are directly rendered into a geometrically-structured latent space by aggregating viewing ray features. Through a tightly coupled rendering codebook scheme, the viewing ray features are dynamically determined at rendering-time based on their geometries (i.e. length, incident-angle), endowing our representation with view-dependent fine-grain variability. Our codebook scheme effectively disentangles feature encoding from rendering, allowing the latent space rendering to run at speeds above 10KHz. Moreover, through metric learning, our geometrically-structured latent space is common to both pose hypotheses and query images with arbitrary field of views. As a result, LASER achieves state-of-the-art performance on large-scale indoor localization datasets (i. e. ZInD [5] and Structured3D [38]) for both panorama and perspective image queries, while significantly outperforming existing learning-based methods in speed.

----

## [1076] Learning to Detect Scene Landmarks for Camera Localization

**Authors**: *Tien Do, Ondrej Miksik, Joseph DeGol, Hyun Soo Park, Sudipta N. Sinha*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01085](https://doi.org/10.1109/CVPR52688.2022.01085)

**Abstract**:

Modern camera localization methods that use image retrieval, feature matching, and 3D structure-based pose estimation require long-term storage of numerous scene images or a vast amount of image features. This can make them unsuitable for resource constrained VR/AR devices and also raises serious privacy concerns. We present a new learned camera localization technique that eliminates the need to store features or a detailed 3D point cloud. Our key idea is to implicitly encode the appearance of a sparse yet salient set of 3D scene points into a convolutional neural network (CNN) that can detect these scene points in query images whenever they are visible. We refer to these points as scene landmarks. We also show that a CNN can be trained to regress bearing vectors for such landmarks even when they are not within the camera's field-of-view. We demonstrate that the predicted landmarks yield accurate pose estimates and that our method outperforms DSAC*, the state-of-the-art in learned localization. Furthermore, extending HLoc (an accurate method) by combining its correspondences with our predictions boosts its accuracy even further.

----

## [1077] Geometric Transformer for Fast and Robust Point Cloud Registration

**Authors**: *Zheng Qin, Hao Yu, Changjian Wang, Yulan Guo, Yuxing Peng, Kai Xu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01086](https://doi.org/10.1109/CVPR52688.2022.01086)

**Abstract**:

We study the problem of extracting accurate correspondences for point cloud registration. Recent keypoint-free methods bypass the detection of repeatable keypoints which is difficult in low-overlap scenarios, showing great potential in registration. They seek correspondences over down-sampled superpoints, which are then propagated to dense points. Superpoints are matched based on whether their neighboring patches overlap. Such sparse and loose matching requires contextual features capturing the geometric structure of the point clouds. We propose Geometric Transformer to learn geometric feature for robust superpoint matching. It encodes pair-wise distances and triplet-wise angles, making it robust in low-overlap cases and invariant to rigid transformation. The simplistic design attains surprisingly high matching accuracy such that no RANSAC is required in the estimation of alignment transformation, leading to 100 times acceleration. Our method improves the inlier ratio by 17∼30 percentage points and the registration recall by over 7 points on the challenging 3DLoMatch benchmark. Our code and models are available at https://github.com/qinzheng93/GeoTransformer.

----

## [1078] ARCS: Accurate Rotation and Correspondence Search

**Authors**: *Liangzu Peng, Manolis C. Tsakiris, René Vidal*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01087](https://doi.org/10.1109/CVPR52688.2022.01087)

**Abstract**:

This paper is about the old Wahba problem in its more general form, which we call “simultaneous rotation and correspondence search”. In this generalization we need to find a rotation that best aligns two partially overlapping 3D point sets, of sizes 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$m$</tex>
 and 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$n$</tex>
 respectively with 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$m\geq n$</tex>
. We first propose a solver, ARCS, that i) assumes noiseless point sets in general position, ii) requires only 2 inliers, iii) uses 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$O(m\log m)$</tex>
 time and 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$O(m)$</tex>
 space, and iv) can successfully solve the problem even with, e.g., 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$m, n\approx 10^{6}$</tex>
 in about 0.1 seconds. We next robustify ARCS to noise, for which we approximately solve consensus maximization problems using ideas from robust subspace learning and interval stabbing. Thirdly, we refine the approximately found consensus set by a Riemannian subgradient descent approach over the space of unit quaternions, which we show converges globally to an 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$\varepsilon$</tex>
-stationary point in 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$O(\varepsilon^{-4})$</tex>
 iterations, or locally to the ground-truth at a linear rate in the absence of noise. We combine these algorithms into ARCS+, to simultaneously search for rotations and correspondences. Experiments show that ARCS+ achieves state-of-the-art performance on large-scale datasets with more than 10
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">6</sup>
 points with a 10
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">4</sup>
 time-speedup over alternative methods. https://github.com/liangzu/ARCS

----

## [1079] FisherMatch: Semi-Supervised Rotation Regression via Entropy-based Filtering

**Authors**: *Yingda Yin, Yingcheng Cai, He Wang, Baoquan Chen*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01088](https://doi.org/10.1109/CVPR52688.2022.01088)

**Abstract**:

Estimating the 3DoF rotation from a single RGB image is an important yet challenging problem. Recent works achieve good performance relying on a large amount of expensive-to-obtain labeled data. To reduce the amount of supervision, we for the first time propose a general framework, FisherMatch, for semi-supervised rotation regression, without assuming any domain-specific knowledge or paired data. Inspired by the popular semi-supervised approach, FixMatch, we propose to leverage pseudo label filtering to facilitate the information flow from labeled data to unlabeled data in a teacher-student mutual learning framework. However, incorporating the pseudo label filtering mechanism into semi-supervised rotation regression is highly non-trivial, mainly due to the lack of a reliable confidence measure for rotation prediction. In this work, we propose to leverage matrix Fisher distribution to build a probabilistic model of rotation and devise a matrix Fisher-based regressor for jointly predicting rotation along with its prediction uncertainty. We then propose to use the entropy of the predicted distribution as a confidence measure, which enables us to perform pseudo label filtering for rotation regression. For supervising such distribution-like pseudo labels, we further investigate the problem of how to enforce loss between two matrix Fisher distributions. Our extensive experiments show that our method can work well even under very low labeled data ratios on different benchmarks, achieving significant and consistent performance improvement over supervised learning and other semi-supervised learning baselines. Our project page is at https://yd-yin.github.io/FisherMatch.

----

## [1080] Uni6D: A Unified CNN Framework without Projection Breakdown for 6D Pose Estimation

**Authors**: *Xiaoke Jiang, Donghai Li, Hao Chen, Ye Zheng, Rui Zhao, Liwei Wu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01089](https://doi.org/10.1109/CVPR52688.2022.01089)

**Abstract**:

As RGB-D sensors become more affordable, using RGB- D images to obtain high-accuracy 6D pose estimation results becomes a better option. State-of-the-art approaches typically use different backbones to extract features for RGB and depth images. They use a 2D CNN for RGB images and a perpixel point cloud network for depth data, as well as a fusion network for feature fusion. We find that the essential reason for using two independent backbones is the “projection breakdown” problem. In the depth image plane, the projected 3D structure of the physical world is preserved by the 1D depth value and its built-in 2D pixel coordinate (UV). Any spatial transformation that modifies UV, such as resize, flip, crop, or pooling operations in the CNN pipeline, breaks the binding between the pixel value and UV coordinate. As a consequence, the 3D structure is no longer preserved by a modified depth image or feature. To address this issue, we propose a simple yet effective method denoted as Uni6D that explicitly takes the extra UV data along with RGB-D images as input. Our method has a Unified CNN framework for 6D pose estimation with a single CNN backbone. In particular, the architecture of our method is based on Mask R-CNN with two extra heads, one named RT head for directly predicting 6D pose and the other named abc head for guiding the network to map the visible points to their coordinates in the 3D model as an auxiliary module. This end-to-end approach balances simplicity and accuracy, achieving comparable accuracy with state of the arts and 7.2x faster inference speed on the YCB-Video dataset.

----

## [1081] OSSGAN: Open-Set Semi-Supervised Image Generation

**Authors**: *Kai Katsumata, Duc Minh Vo, Hideki Nakayama*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01090](https://doi.org/10.1109/CVPR52688.2022.01090)

**Abstract**:

We introduce a challenging training scheme of conditional GANs, called open-set semi-supervised image generation, where the training dataset consists of two parts: (i) labeled data and (ii) unlabeled data with samples belonging to one of the labeled data classes, namely, a closed-set, and samples not belonging to any of the labeled data classes, namely, an open-set. Unlike the existing semi-supervised image generation task, where unlabeled data only contain closed-set samples, our task is more general and lowers the data collection cost in practice by allowing open-set samples to appear. Thanks to entropy regularization, the classifier that is trained on labeled data is able to quantify sample-wise importance to the training of cGAN as confidence, allowing us to use all samples in un-labeled data. We design OSSGAN, which provides decision clues to the discriminator on the basis of whether an unlabeled image belongs to one or none of the classes of interest, smoothly integrating labeled and unlabeled data during training. The results of experiments on Tiny ImageNet and ImageNet show notable improvements over supervised Big-GAN and semi-supervised methods. Our code is available at https://github.com/raven38/OSSGAN.

----

## [1082] Attribute Group Editing for Reliable Few-shot Image Generation

**Authors**: *Guanqi Ding, Xinzhe Han, Shuhui Wang, Shuzhe Wu, Xin Jin, Dandan Tu, Qingming Huang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01091](https://doi.org/10.1109/CVPR52688.2022.01091)

**Abstract**:

Few-shot image generation is a challenging task even using the state-of-the-art Generative Adversarial Networks (GANs). Due to the unstable GAN training process and the limited training data, the generated images are often of low quality and low diversity. In this work, we propose a new “editing-based” method, i.e., Attribute Group Editing (AGE), for few-shot image generation. The basic assumption is that any image is a collection of attributes and the editing direction for a specific attribute is shared across all categories. AGE examines the internal representation learned in GANs and identifies semantically meaningful directions. Specifically, the class embedding, i.e., the mean vector of the latent codes from a specific category, is used to represent the category-relevant attributes, and the category-irrelevant attributes are learned globally by Sparse Dictionary Learning on the difference between the sample embedding and the class embedding. Given a GAN well trained on seen categories, diverse images of unseen categories can be synthesized through editing category-irrelevant attributes while keeping category-relevant attributes unchanged. Without re-training the GAN, AGE is capable of not only producing more realistic and diverse images for downstream visual applications with limited data but achieving controllable image editing with interpretable category-irrelevant directions. Code is available at https://github.com/UniBester/AGE.

----

## [1083] Few Shot Generative Model Adaption via Relaxed Spatial Structural Alignment

**Authors**: *Jiayu Xiao, Liang Li, Chaofei Wang, Zheng-Jun Zha, Qingming Huang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01092](https://doi.org/10.1109/CVPR52688.2022.01092)

**Abstract**:

Training a generative adversarial network (GAN) with limited data has been a challenging task. A feasible solution is to start with a GAN well-trained on a large scale source domain and adapt it to the target domain with a few samples, termed as few shot generative model adaption. However, existing methods are prone to model overfitting and collapse in extremely few shot setting (less than 10). To solve this problem, we propose a relaxed spatial structural alignment (RSSA) method to calibrate the target generative models during the adaption. We design a cross-domain spatial structural consistency loss comprising the self-correlation and disturbance correlation consistency loss. It helps align the spatial structural information between the synthesis image pairs of the source and target domains. To relax the cross-domain alignment, we compress the original latent space of generative models to a subspace. Image pairs generated from the subspace are pulled closer. Qualitative and quantitative experiments show that our method consistently surpasses the state-of-the-art methods in few shot setting. Our source code: https://github.com/StevenShaw1999/RSSA.

----

## [1084] Semantic-shape Adaptive Feature Modulation for Semantic Image Synthesis

**Authors**: *Zhengyao Lv, Xiaoming Li, Zhenxing Niu, Bing Cao, Wangmeng Zuo*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01093](https://doi.org/10.1109/CVPR52688.2022.01093)

**Abstract**:

Recent years have witnessed substantial progress in se-mantic image synthesis, it is still challenging in synthesizing photo-realistic images with rich details. Most previ-ous methods focus on exploiting the given semantic map, which just captures an object-level layout for an image. Obviously, a fine-grained part-level semantic layout will benefit object details generation, and it can be roughly in-ferred from an object's shape. In order to exploit the part-level layouts, we propose a Shape-aware Position Descrip-tor (SPD) to describe each pixel's positional feature, where object shape is explicitly encoded into the SP D feature. Fur-thermore, a Semantic-shape Adaptive Feature Modulation (SAFM) block is proposed to combine the given semantic map and our positional features to produce adaptively mod-ulated features. Extensive experiments demonstrate that the proposed SPD and SAFM significantly improve the gener-ation of objects with rich details. Moreover, our method performs favorably against the SOTA methods in terms of quantitative and qualitative evaluation. The source code and model are available at SAFM.

----

## [1085] Retrieval-based Spatially Adaptive Normalization for Semantic Image Synthesis

**Authors**: *Yupeng Shi, Xiao Liu, Yuxiang Wei, Zhongqin Wu, Wangmeng Zuo*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01094](https://doi.org/10.1109/CVPR52688.2022.01094)

**Abstract**:

Semantic image synthesis is a challenging task with many practical applications. Albeit remarkable progress has been made in semantic image synthesis with spatiallyadaptive normalization, existing methods usually normalize the feature activations under the coarse-level guidance (e.g., semantic class). However, different parts of a semantic object (e.g., wheel and window of car) are quite different in structures and textures, making blurry synthesis results usually inevitable due to the missing of fine-grained guidance. In this paper, we propose a novel normalization module, termed as REtrieval-based Spatially Adaptive normaLization (RESAIL), for introducing pixel level fine- grained guidance to the normalization architecture. Specifically, we first present a retrieval paradigm by finding a content patch of the same semantic class from training set with the most similar shape to each test semantic mask. Then, the retrieved patches are composited into retrieval-based guidance, which can be used by RESAIL for pixel level fine-grained modulation on feature activations, thereby greatly mitigating blurry synthesis results. Moreover, distorted ground-truth images are also utilized as alternatives of retrieval-based guidance for feature normalization, further benefiting model training and improving visual quality of generated images. Experiments on several challenging datasets show that our RESAIL performs favorably against state-of-the-arts in terms of quantitative metrics, visual quality, and subjective evaluation. The source code is available at https://github.com/Shi-Yupeng/RESAIL-For-SIS.

----

## [1086] Generative Flows with Invertible Attentions

**Authors**: *Rhea Sanjay Sukthanker, Zhiwu Huang, Suryansh Kumar, Radu Timofte, Luc Van Gool*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01095](https://doi.org/10.1109/CVPR52688.2022.01095)

**Abstract**:

Flow-based generative models have shown an excellent ability to explicitly learn the probability density function of data via a sequence of invertible transformations. Yet, learning attentions in generative flows remains understudied, while it has made breakthroughs in other domains. To fill the gap, this paper introduces two types of invertible attention mechanisms, i.e., map-based and transformer-based attentions, for both unconditional and conditional generative flows. The key idea is to exploit a masked scheme of these two attentions to learn long-range data dependencies in the context of generative flows. The masked scheme allows for invertible attention modules with tractable Jacobian determinants, enabling its seamless integration at any positions of the flow-based models. The proposed attention mechanisms lead to more efficient generative flows, due to their capability of modeling the long-term data dependencies. Evaluation on multiple image synthesis tasks shows that the proposed attention flows result in efficient models and compare favorably against the state-of-the-art unconditional and conditional generative flows.

----

## [1087] Style-Structure Disentangled Features and Normalizing Flows for Diverse Icon Colorization

**Authors**: *Yuan-kui Li, Yun-Hsuan Lien, Yu-Shuen Wang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01096](https://doi.org/10.1109/CVPR52688.2022.01096)

**Abstract**:

We present a colorization network that generates flat-color icons according to given sketches and semantic colorization styles. Our network contains a style-structure disentangled colorization module and a normalizing flow. The colorization module transforms a paired sketch image and style image into a flat-color icon. To enhance network generalization and the quality of icons, we present a pixel-wise decoder, a global style code, and a contour loss to reduce color gradients at flat regions and increase color discontinuity at boundaries. The normalizing flow maps Gaussian vectors to diverse style codes conditioned on the given semantic colorization label. This conditional sampling enables users to control attributes and obtain diverse colorization results. Compared to previous methods built upon conditional generative adversarial networks, our approach enjoys the advantages of both high image quality and diversity. To evaluate its effectiveness, we compared the flat-color icons generated by our approach and recent colorization and image-to-image translation methods on various conditions. Experiment results verify that our method out- performs state-of-the-arts qualitatively and quantitatively.

----

## [1088] SemanticStyleGAN: Learning Compositional Generative Priors for Controllable Image Synthesis and Editing

**Authors**: *Yichun Shi, Xiao Yang, Yangyue Wan, Xiaohui Shen*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01097](https://doi.org/10.1109/CVPR52688.2022.01097)

**Abstract**:

Recent studies have shown that StyleGANs provide promising prior models for downstream tasks on image synthesis and editing. However since the latent codes of StyleGANs are designed to control global styles it is hard to achieve a fine-grained control over synthesized images. We present SemanticStyleGAN where a generator is trained to model local semantic parts separately and synthesizes images in a compositional way. The structure and texture of different local parts are controlled by corresponding latent codes. Experimental results demonstrate that our model provides a strong disentanglement between different spatial areas. When combined with editing methods designed for StyleGANs it can achieve a more fine-grained control to edit synthesized or real images. The model can also be extended to other domains via transfer learning. Thus as a generic prior model with built-in disentanglement it could facilitate the development of GAN-based applications and enable more potential downstream tasks.

----

## [1089] Manifold Learning Benefits GANs

**Authors**: *Yao Ni, Piotr Koniusz, Richard I. Hartley, Richard Nock*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01098](https://doi.org/10.1109/CVPR52688.2022.01098)

**Abstract**:

In this paper
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
Code: https://qithub.com/MaxwellYaoNi/LCSAGAN., we improve Generative Adversarial Net-works by incorporating a manifold learning step into the discriminator. We consider locality-constrained linear and subspace-based manifolds
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">2</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">2</sup>
The coding spaces considered in this paper are loosely termed man-ifolds. In most cases they are not manifolds in the strict mathematical sense, but rather topological spaces such as varieties, or simplicial com-plexes. The word will be used only in an informal sense., and locality-constrained non-linear manifolds. In our design, the manifold learning and coding steps are intertwined with layers of the discrimina-tor, with the goal of attracting intermediate feature repre-sentations onto manifolds. We adaptively balance the dis-crepancy between feature representations and their mani-fold view, which is a trade-off between denoising on the manifold and refining the manifold. We find that locality-constrained non-linear manifolds outperform linear mani-folds due to their non-uniform density and smoothness. We also substantially outperform state-of-the-art baselines.

----

## [1090] DO-GAN: A Double Oracle Framework for Generative Adversarial Networks

**Authors**: *Aye Phyu Phyu Aung, Xinrun Wang, Runsheng Yu, Bo An, Senthilnath Jayavelu, Xiaoli Li*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01099](https://doi.org/10.1109/CVPR52688.2022.01099)

**Abstract**:

In this paper, we propose a new approach to train Gen-erative Adversarial Networks (GANs) where we deploy a double-oracle framework using the generator and discrim-inator oracles. GAN is essentially a two-player zero-sum game between the generator and the discriminator. Training GANs is challenging as a pure Nash equilibrium may not exist and even finding the mixed Nash equilibrium is difficult as GANs have a large-scale strategy space. In DO-GAN, we extend the double oracle framework to GANs. We first generalize the players' strategies as the trained models of generator and discriminator from the best response or-acles. We then compute the meta-strategies using a linear program. For scalability of the framework where multi-ple generators and discriminator best responses are stored in the memory, we propose two solutions: 1) pruning the weakly-dominated players' strategies to keep the oracles from becoming intractable; 2) applying continual learning to retain the previous knowledge of the networks. We apply our framework to established GAN architectures such as vanilla GAN, Deep Convolutional GAN, Spectral Normalization GAN and Stacked GAN. Finally, we conduct experiments on MNIST, CIFAR-10 and CelebA datasets and show that DO-GAN variants have significant improvements in both subjective qualitative evaluation and quantitative metrics, compared with their respective GAN architectures.

----

## [1091] Improving GAN Equilibrium by Raising Spatial Awareness

**Authors**: *Jianyuan Wang, Ceyuan Yang, Yinghao Xu, Yujun Shen, Hongdong Li, Bolei Zhou*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01100](https://doi.org/10.1109/CVPR52688.2022.01100)

**Abstract**:

The success of Generative Adversarial Networks (GANs) is largely built upon the adversarial training between a generator (G) and a discriminator (D). They are expected to reach a certain equilibrium where D cannot distinguish the generated images from the real ones. However, such an equilibrium is rarely achieved in practical GAN training, instead, D almost always surpasses G. We attribute one of its sources to the information asymmetry between D and G. We observe that D learns its own visual attention when determining whether an image is real or fake, but G has no explicit clue on which regions to focus on for a particular synthesis. To alleviate the issue of D dominating the competition in GANs, we aim to raise the spatial awareness of G. Randomly sampled multi-level heatmaps are encoded into the intermediate layers of G as an inductive bias. Thus G can purposefully improve the synthesis of certain image regions. We further propose to align the spatial awareness of G with the attention map induced from D. Through this way we effectively lessen the information gap between D and G. Extensive results show that our method pushes the two-player game in GANs closer to the equilibrium, leading to a better synthesis performance. As a byproduct, the intro-duced spatial awareness facilitates interactive editing over the output synthesis. Demo video and code are available at https://genforce.github.io/eqgan-sa/

----

## [1092] Feature Statistics Mixing Regularization for Generative Adversarial Networks

**Authors**: *Junho Kim, Yunjey Choi, Youngjung Uh*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01101](https://doi.org/10.1109/CVPR52688.2022.01101)

**Abstract**:

In generative adversarial networks, improving discriminators is one of the key components for generation performance. As image classifiers are biased toward texture and debiasing improves accuracy, we investigate 1) if the discriminators are biased, and 2) if debiasing the discriminators will improve generation performance. Indeed, we find empirical evidence that the discriminators are sensitive to the style (e.g., texture and color) of images. As a remedy, we propose feature statistics mixing regularization (FSMR) that encourages the discriminator's prediction to be invariant to the styles of input images. Specifically, we generate a mixed feature of an original and a reference image in the discriminator's feature space and we apply regularization so that the prediction for the mixed feature is consistent with the prediction for the original image. We conduct extensive experiments to demonstrate that our regularization leads to reduced sensitivity to style and consistently improves the performance of various GAN architectures on nine datasets. In addition, adding FSMR to recently-proposed augmentation-based GAN methods further improves image quality. Our code is available at https://github.com/naver-ai/FSMR.

----

## [1093] StyleSwin: Transformer-based GAN for High-resolution Image Generation

**Authors**: *Bowen Zhang, Shuyang Gu, Bo Zhang, Jianmin Bao, Dong Chen, Fang Wen, Yong Wang, Baining Guo*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01102](https://doi.org/10.1109/CVPR52688.2022.01102)

**Abstract**:

Despite the tantalizing success in a broad of vision tasks, transformers have not yet demonstrated on-par ability as ConvNets in high-resolution image generative modeling. In this paper, we seek to explore using pure transformers to build a generative adversarial network for high-resolution image synthesis. To this end, we believe that local attention is crucial to strike the balance between computational efficiency and modeling capacity. Hence, the proposed generator adopts Swin transformer in a style-based architecture. To achieve a larger receptive field, we propose double attention which simultaneously leverages the context of the local and the shifted windows, leading to improved generation quality. Moreover, we show that offering the knowledge of the absolute position that has been lost in window-based transformers greatly benefits the generation quality. The proposed StyleSwin is scalable to high resolutions, with both the coarse geometry and fine structures benefit from the strong expressivity of transformers. However, blocking artifacts occur during high-resolution synthesis because performing the local attention in a block-wise manner may break the spatial coherency. To solve this, we empirically investigate various solutions, among which we find that employing a wavelet discriminator to examine the spectral discrepancy effectively suppresses the artifacts. Extensive experiments show the superiority over prior transformer-based GANs, especially on high resolutions, e.g., 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$1024 \times$</tex>
 1024. The StyleSwin, without complex training strategies, excels over StyleGAN on CelebA-HQ 1024, and achieves on-par performance on FFHQ-1024, proving the promise of using transformers for high-resolution image generation. The code and pretrained models are available at https://github.com/microsoft/StyleSwin.

----

## [1094] MaskGIT: Masked Generative Image Transformer

**Authors**: *Huiwen Chang, Han Zhang, Lu Jiang, Ce Liu, William T. Freeman*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01103](https://doi.org/10.1109/CVPR52688.2022.01103)

**Abstract**:

Generative transformers have experienced rapid popularity growth in the computer vision community in synthesizing high-fidelity and high-resolution images. The best generative transformer models so far, however, still treat an image naively as a sequence of tokens, and decode an image sequentially following the raster scan ordering (i.e. line-by-line). We find this strategy neither optimal nor efficient. This paper proposes a novel image synthesis paradigm using a bidirectional transformer decoder, which we term MaskGIT. During training, MaskGIT learns to predict randomly masked tokens by attending to tokens in all directions. At inference time, the model begins with generating all tokens of an image simultaneously, and then refines the image iteratively conditioned on the previous generation. Our experiments demonstrate that MaskGIT significantly outperforms the state-of-the-art transformer model on the ImageNet dataset, and accelerates autoregressive decoding by up to 48x. Besides, we illustrate that MaskGIT can be easily extended to various image editing tasks, such as inpainting, extrapolation, and image manipulation. Project page: masked-generative-image-transformer.github.io.

----

## [1095] StyTr2: Image Style Transfer with Transformers

**Authors**: *Yingying Deng, Fan Tang, Weiming Dong, Chongyang Ma, Xingjia Pan, Lei Wang, Changsheng Xu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01104](https://doi.org/10.1109/CVPR52688.2022.01104)

**Abstract**:

The goal of image style transfer is to render an image with artistic features guided by a style reference while maintaining the original content. Owing to the locality in convolutional neural networks (CNNs), extracting and maintaining the global information of input images is difficult. Therefore, traditional neural style transfer methods face biased content representation. To address this critical issue, we take long-range dependencies of input images into account for image style transfer by proposing a transformer-based approach called StyTr2. In contrast with visual transformers for other vision tasks, StyTr2 contains two different transformer encoders to generate domain-specific sequences for content and style, respectively. Following the encoders, a multi-layer transformer decoder is adopted to stylize the content sequence according to the style sequence. We also analyze the deficiency of existing positional encoding methods and propose the content-aware positional encoding (CAPE), which is scale-invariant and more suitable for image style transfer tasks. Qualitative and quantitative experiments demonstrate the effectiveness of the proposed StyTr2 compared with state-of-the-art CNN-based and flow-based approaches. Code and models are available at https://github.com/diyiiyiii/StyTR-2.

----

## [1096] Style Transformer for Image Inversion and Editing

**Authors**: *Xueqi Hu, Qiusheng Huang, Zhengyi Shi, Siyuan Li, Changxin Gao, Li Sun, Qingli Li*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01105](https://doi.org/10.1109/CVPR52688.2022.01105)

**Abstract**:

Existing GAN inversion methods fail to provide latent codes for reliable reconstruction and flexible editing simultaneously. This paper presents a transformer-based image inversion and editing model for pretrained StyleGAN which is not only with less distortions, but also of high quality and flexibility for editing. The proposed model employs a CNN encoder to provide multi-scale image features as keys and values. Meanwhile it regards the style code to be determined for different layers of the generator as queries. It first initializes query tokens as learnable parameters and maps them into W+ space. Then the multi-stage alternate self-and cross-attention are utilized, updating queries with the purpose of inverting the input by the generator. Moreover, based on the inverted code, we investigate the reference-and label-based attribute editing through a pretrained latent classifier, and achieve flexible image-to-image translation with high quality results. Extensive experiments are carried out, showing better performances on both inversion and editing tasks within StyleGAN. Codes are available at https://github.com/sapphire497/style-transformer.

----

## [1097] Reduce Information Loss in Transformers for Pluralistic Image Inpainting

**Authors**: *Qiankun Liu, Zhentao Tan, Dongdong Chen, Qi Chu, Xiyang Dai, Yinpeng Chen, Mengchen Liu, Lu Yuan, Nenghai Yu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01106](https://doi.org/10.1109/CVPR52688.2022.01106)

**Abstract**:

Transformers have achieved great success in pluralistic image inpainting recently. However, we find existing transformer based solutions regard each pixel as a token, thus suffer from information loss issue from two aspects: 1) They downsample the input image into much lower resolutions for efficiency consideration, incurring information loss and extra misalignment for the boundaries of masked regions. 2) They quantize 256
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">3</sup>
 RGB pixels to a small number (such as 512) of quantized pixels. The indices of quantized pixels are used as tokens for the inputs and prediction targets of transformer. Although an extra CNN network is used to upsample and refine the low-resolution results, it is difficult to retrieve the lost information back. To keep input information as much as possible, we propose a new transformer based framework “PUT”. Specifically, to avoid input downsampling while maintaining the computation efficiency, we design a patch-based auto-encoder P-VQVAE, where the encoder converts the masked image into non-overlapped patch tokens and the decoder recovers the masked regions from the inpainted tokens while keeping the unmasked regions unchanged. To eliminate the information loss caused by quantization, an Un-Quantized Transformer (UQ-Transformer) is applied, which directly takes the features from P-VQVAE encoder as input without quantization and regards the quantized tokens only as prediction targets. Extensive experiments show that PUT greatly outperforms state-of-the-art methods on image fidelity, especially for large masked regions and complex large-scale datasets.

----

## [1098] Incremental Transformer Structure Enhanced Image Inpainting with Masking Positional Encoding

**Authors**: *Qiaole Dong, Chenjie Cao, Yanwei Fu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01107](https://doi.org/10.1109/CVPR52688.2022.01107)

**Abstract**:

Image inpainting has made significant advances in recent years. However, it is still challenging to recover corrupted images with both vivid textures and reasonable structures. Some specific methods only tackle regular textures while losing holistic structures due to the limited receptive fields of convolutional neural networks (CNNs). On the other hand, attention-based models can learn better long-range dependency for the structure recovery, but they are limited by the heavy computation for inference with large image sizes. To address these issues, we propose to leverage an additional structure restorer to facilitate the image inpainting incrementally. The proposed model restores holistic image structures with a powerful attention-based transformer model in a fixed low-resolution sketch space. Such a grayscale space is easy to be upsampled to larger scales to convey correct structural information. Our structure restorer can be integrated with other pretrained inpainting models efficiently with the zero-initialized residual addition. Furthermore, a masking positional encoding strategy is utilized to improve the performance with large irregular masks. Extensive experiments on various datasets validate the efficacy of our model compared with other competitors. Our codes are released in https://github.com/DQiaole/ZITS_inpainting.

----

## [1099] UniCoRN: A Unified Conditional Image Repainting Network

**Authors**: *Jimeng Sun, Shuchen Weng, Zheng Chang, Si Li, Boxin Shi*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01108](https://doi.org/10.1109/CVPR52688.2022.01108)

**Abstract**:

Conditional image repainting (CIR) is an advanced image editing task, which requires the model to generate visual content in user-specified regions conditioned on multiple cross-modality constraints, and composite the visual content with the provided background seamlessly. Existing methods based on two-phase architecture design assume dependency between phases and cause color-image incongruity. To solve these problems, we propose a novel Unified Conditional image Repainting Network (UniCoRN). We break the two-phase assumption in the CIR task by constructing the interaction and dependency relationship between background and other conditions. We further introduce the hierarchical structure into cross-modality similarity model to capture feature patterns at different levels and bridge the gap between visual content and color condition. A new Landscape-CIR dataset is collected and annotated to expand the application scenarios of the CIR task. Experiments show that UniCoRN achieves higher synthetic quality, better condition consistency, and more realistic compositing effect.

----

## [1100] High-Fidelity GAN Inversion for Image Attribute Editing

**Authors**: *Tengfei Wang, Yong Zhang, Yanbo Fan, Jue Wang, Qifeng Chen*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01109](https://doi.org/10.1109/CVPR52688.2022.01109)

**Abstract**:

We present a novel highfidelity generative adversarial network (GAN) inversion framework that enables attribute editing with image-specific details well-preserved (e.g., background, appearance, and illumination). We first analyze the challenges of highfidelity GAN inversion from the perspective of lossy data compression. With a low bitrate latent code, previous works have difficulties in preserving highfidelity details in reconstructed and edited images. Increasing the size of a latent code can improve the accuracy of GAN inversion but at the cost of inferior editability. To improve image fidelity without compromising editability, we propose a distortion consultation approach that employs a distortion map as a reference for highfidelity reconstruction. In the distortion consultation inversion (DCI), the distortion map is first projected to a high-rate latent map, which then complements the basic low-rate latent code with more details via consultation fusion. To achieve high-fidelity editing, we propose an adaptive distortion alignment (ADA) module with a self-supervised training scheme, which bridges the gap between the edited and inversion images. Extensive experiments in the face and car domains show a clear improvement in both inversion and editing quality. The project page is https://tengfei-wang.github.io/HFGI/.

----

## [1101] HyperInverter: Improving StyleGAN Inversion via Hypernetwork

**Authors**: *Tan M. Dinh, Anh Tuan Tran, Rang Nguyen, Binh-Son Hua*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01110](https://doi.org/10.1109/CVPR52688.2022.01110)

**Abstract**:

Real-world image manipulation has achieved fantastic progress in recent years as a result of the exploration and utilization of GAN latent spaces. GAN inversion is the first step in this pipeline, which aims to map the real image to the latent code faithfully. Unfortunately, the majority of existing GAN inversion methods fail to meet at least one of the three requirements listed below: high reconstruction quality, editability, and fast inference. We present a novel two-phase strategy in this research that fits all requirements at the same time. In the first phase, we train an encoder to map the input image to StyleGAN2 W-space, which was proven to have excellent editability but lower reconstruction quality. In the second phase, we supplement the reconstruction ability in the initial phase by leveraging a series of hypernetworks to recover the missing information during inversion. These two steps complement each other to yield high reconstruction quality thanks to the hypernetwork branch and excellent editability due to the inversion done in the W-space. Our method is entirely encoder-based, resulting in extremely fast inference. Extensive experiments on two challenging datasets demonstrate the superiority of our method.
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
Project page: https://di-mi-ta.github.io/HyperInverter

----

## [1102] Spatially-Adaptive Multilayer Selection for GAN Inversion and Editing

**Authors**: *Gaurav Parmar, Yijun Li, Jingwan Lu, Richard Zhang, Jun-Yan Zhu, Krishna Kumar Singh*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01111](https://doi.org/10.1109/CVPR52688.2022.01111)

**Abstract**:

Existing GAN inversion and editing methods work well for aligned objects with a clean background, such as portraits and animal faces, but often struggle for more difficult categories with complex scene layouts and object occlusions, such as cars, animals, and outdoor images. We propose a new method to invert and edit such complex images in the latent space of GANs, such as StyleGAN2. Our key idea is to explore inversion with a collection of layers, spatially adapting the inversion process to the difficulty of the image. We learn to predict the “invertibility” of different image segments and project each segment into a latent layer. Easier regions can be inverted into an earlier layer in the generator's latent space, while more challenging regions can be inverted into a later feature space. Experiments show that our method obtains better inversion results compared to the recent approaches on complex categories, while maintaining downstream editability. Please refer to our project page at gauravparmar.com/sam_inversion.

----

## [1103] On Aliased Resizing and Surprising Subtleties in GAN Evaluation

**Authors**: *Gaurav Parmar, Richard Zhang, Jun-Yan Zhu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01112](https://doi.org/10.1109/CVPR52688.2022.01112)

**Abstract**:

Metrics for evaluating generative models aim to measure the discrepancy between real and generated images. The often-used Fréchet Inception Distance (FID) metric, for example, extracts “high-level” features using a deep network from the two sets. However, we find that the differences in “low-level” preprocessing, specifically image resizing and compression, can induce large variations and have unforeseen consequences. For instance, when resizing an image, e.g., with a bilinear or bicubic kernel, signal processing principles mandate adjusting prefilter width depending on the downsampling factor, to antialias to the appropriate bandwidth. However, commonly-used implementations use a fixed-width prefilter, resulting in aliasing artifacts. Such aliasing leads to corruptions in the feature extraction down-stream. Next, lossy compression, such as JPEG, is commonly used to reduce the file size of an image. Although designed to minimally degrade the perceptual quality of an image, the operation also produces variations downstream. Furthermore, we show that if compression is used on real training images, FID can actually improve if the generated images are also subsequently compressed. This paper shows that choices in low-level image processing have been an under-appreciated aspect of generative modeling. We identify and characterize variations in generative modeling development pipelines, provide recommendations based on signal processing principles, and release a reference implementation to facilitate future comparisons.

----

## [1104] Dual-path Image Inpainting with Auxiliary GAN Inversion

**Authors**: *Wentao Wang, Li Niu, Jianfu Zhang, Xue Yang, Liqing Zhang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01113](https://doi.org/10.1109/CVPR52688.2022.01113)

**Abstract**:

Deep image inpainting can inpaint a corrupted image using a feed-forward inference, but still fails to handle large missing area or complex semantics. Recently, GAN inversion based inpainting methods propose to leverage semantic information in pretrained generator (e.g., StyleGAN) to solve the above issues. Different from feed-forward methods, they seek for a closest latent code to the corrupted image and feed it to a pretrained generator. However, inferring the latent code is either time-consuming or inaccurate. In this paper, we develop a dual-path inpainting network with inversion path and feed-forward path, in which inversion path provides auxiliary information to help feed-forward path. We also design a novel deformable fusion module to align the feature maps in two paths. Experiments on FFHQ and LSUN demonstrate that our method is effective in solving the aforementioned problems while producing more realistic results than state-of-the-art methods.

----

## [1105] InOut: Diverse Image Outpainting via GAN Inversion

**Authors**: *Yen-Chi Cheng, Chieh Hubert Lin, Hsin-Ying Lee, Jian Ren, Sergey Tulyakov, Ming-Hsuan Yang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01114](https://doi.org/10.1109/CVPR52688.2022.01114)

**Abstract**:

Image outpainting seeks for a semantically consistent extension of the input image beyond its available content. Compared to inpainting - filling in missing pixels in a way coherent with the neighboring pixels - outpainting can be achieved in more diverse ways since the problem is less constrained by the surrounding pixels. Existing image outpainting methods pose the problem as a conditional image-to-image translation task, often generating repetitive structures and textures by replicating the content available in the input image. In this work, we formulate the problem from the perspective of inverting generative adversarial networks. Our generator renders micro-patches conditioned on their joint latent code as well as their individual positions in the image. To outpaint an image, we seek for multiple latent codes not only recovering available patches but also synthesizing diverse outpainting by patch-based generation. This leads to richer structure and content in the outpainted regions. Furthermore, our formulation allows for outpainting conditioned on the categorical input, thereby enabling flexible user controls. Extensive experimental results demonstrate the proposed method performs favorably against existing in- and outpainting methods, featuring higher visual quality and diversity.

----

## [1106] Diverse Plausible 360-Degree Image Outpainting for Efficient 3DCG Background Creation

**Authors**: *Naofumi Akimoto, Yuhi Matsuo, Yoshimitsu Aoki*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01115](https://doi.org/10.1109/CVPR52688.2022.01115)

**Abstract**:

We address the problem of generating a 360-degree image from a single image with a narrow field of view by estimating its surroundings. Previous methods suffered from overfitting to the training resolution and deterministic generation. This paper proposes a completion method using a transformer for scene modeling and novel methods to improve the properties of a 360-degree image on the output image. Specifically, we use CompletionNets with a transformer to perform diverse completions and Adjust-mentNet to match color, stitching, and resolution with an input image, enabling inference at any resolution. To improve the properties of a 360-degree image on an output image, we also propose WS-perceptual loss and circular inference. Thorough experiments show that our method out-performs state-of-the-art (SOTA) methods both qualitatively and quantitatively. For example, compared to SOTA methods, our method completes images 16 times larger in resolution and achieves 1.7 times lower Fréchet inception distance (FID). Furthermore, we propose a pipeline that uses the completion results for lighting and background of 3DCG scenes. Our plausible background completion enables perceptually natural results in the application of inserting virtual objects with specular surfaces.

----

## [1107] Contextual Outpainting with Object-Level Contrastive Learning

**Authors**: *Jiacheng Li, Chang Chen, Zhiwei Xiong*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01116](https://doi.org/10.1109/CVPR52688.2022.01116)

**Abstract**:

We study the problem of contextual outpainting, which aims to hallucinate the missing background contents based on the remaining foreground contents. Existing image outpainting methods focus on completing object shapes or extending existing scenery textures, neglecting the semantically meaningful relationship between the missing and remaining contents. To explore the semantic cues provided by the remaining foreground contents, we propose a novel ConTextual Outpainting GAN (CTO-GAN), leveraging the semantic layout as a bridge to synthesize coherent and diverse background contents. To model the contextual correlation between foreground and background contents, we incorporate an object-level contrastive loss to regularize the learning of cross-modal representations of foreground contents and the corresponding background semantic layout, facilitating accurate semantic reasoning. Furthermore, we improve the realism of the generated background contents via detecting generated context in adversarial training. Extensive experiments demonstrate that the proposed method achieves superior performance compared with existing solutions on the challenging COCO-stuff dataset. Project page: https://ddlee-cn.github.io/cto-gan.

----

## [1108] RePaint: Inpainting using Denoising Diffusion Probabilistic Models

**Authors**: *Andreas Lugmayr, Martin Danelljan, Andrés Romero, Fisher Yu, Radu Timofte, Luc Van Gool*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01117](https://doi.org/10.1109/CVPR52688.2022.01117)

**Abstract**:

Free-form inpainting is the task of adding new content to an image in the regions specified by an arbitrary binary mask. Most existing approaches train for a certain distribution of masks, which limits their generalization capabilities to unseen mask types. Furthermore, training with pixel-wise and perceptual losses often leads to simple textural extensions towards the missing areas instead of semantically meaningful generation. In this work, we propose RePaint: A Denoising Diffusion Probabilistic Model (DDPM) based inpainting approach that is applicable to even extreme masks. We employ a pretrained unconditional DDPM as the generative prior. To condition the generation process, we only alter the reverse diffusion iterations by sampling the unmasked regions using the given image infor-mation. Since this technique does not modify or condition the original DDPM network itself, the model produces high-quality and diverse output images for any inpainting form. We validate our method for both faces and general-purpose image inpainting using standard and extreme masks. Re-Paint outperforms state-of-the-art Autoregressive, and GAN approaches for at least five out of six mask distributions. Github Repository: git.io/RePaint

----

## [1109] Perception Prioritized Training of Diffusion Models

**Authors**: *Jooyoung Choi, Jungbeom Lee, Chaehun Shin, Sungwon Kim, Hyunwoo Kim, Sungroh Yoon*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01118](https://doi.org/10.1109/CVPR52688.2022.01118)

**Abstract**:

Diffusion models learn to restore noisy data, which is corrupted with different levels of noise, by optimizing the weighted sum of the corresponding loss terms, i.e., denoising score matching loss. In this paper, we show that restoring data corrupted with certain noise levels offers a proper pretext task for the model to learn rich visual concepts. We propose to prioritize such noise levels over other levels during training, by redesigning the weighting scheme of the objective function. We show that our simple redesign of the weighting scheme significantly improves the performance of diffusion models regardless of the datasets, architectures, and sampling strategies.

----

## [1110] Dynamic Dual-Output Diffusion Models

**Authors**: *Yaniv Benny, Lior Wolf*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01119](https://doi.org/10.1109/CVPR52688.2022.01119)

**Abstract**:

Iterative denoising-based generation, also known as denoising diffusion models, has recently been shown to be comparable in quality to other classes of generative models, and even surpass them. Including, in particular, Generative Adversarial Networks, which are currently the state of the art in many subtasks of image generation. However, a major drawback of this method is that it requires hundreds of iterations to produce a competitive result. Recent works have proposed solutions that allow for faster generation with fewer iterations, but the image quality gradually deteriorates with increasingly fewer iterations being applied during generation. In this paper, we reveal some of the causes that affect the generation quality of diffusion models, especially when sampling with few iterations, and come up with a simple, yet effective, solution to mitigate them. We consider two opposite equations for the iterative denoising, the first predicts the applied noise, and the second predicts the image directly. Our solution takes the two options and learns to dynamically alternate between them through the denoising process. Our proposed solution is general and can be applied to any existing diffusion model. As we show, when applied to various SOTA architectures, our solution immediately improves their generation quality, with negligible added complexity and parameters. We experiment on multiple datasets and configurations and run an extensive ablation study to support these findings.

----

## [1111] Generating High Fidelity Data from Low-density Regions using Diffusion Models

**Authors**: *Vikash Sehwag, Caner Hazirbas, Albert Gordo, Firat Ozgenel, Cristian Canton-Ferrer*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01120](https://doi.org/10.1109/CVPR52688.2022.01120)

**Abstract**:

Our work focuses on addressing sample deficiency from low-density regions of data manifold in common image datasets. We leverage diffusion process based generative models to synthesize novel images from low-density regions. We observe that uniform sampling from diffusion models predominantly samples from high-density regions of the data manifold. Therefore, we modify the sampling process to guide it towards low-density regions while simulta-neously maintaining the fidelity of synthetic data. We rigorously demonstrate that our process successfully generates novel high fidelity samples from low-density regions. We further examine generated samples and show that the model does not memorize low-density data and indeed learns to generate novel samples from low-density regions.

----

## [1112] Global Context with Discrete Diffusion in Vector Quantised Modelling for Image Generation

**Authors**: *Minghui Hu, Yujie Wang, Tat-Jen Cham, Jianfei Yang, Ponnuthurai N. Suganthan*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01121](https://doi.org/10.1109/CVPR52688.2022.01121)

**Abstract**:

The integration of Vector Quantised Variational AutoEncoder (VQ-VAE) with autoregressive models as generation part has yielded high-quality results on image generation. However, the autoregressive models will strictly follow the progressive scanning order during the sampling phase. This leads the existing VQ series models to hardly escape the trap of lacking global information. Denoising Diffusion Probabilistic Models (DDPM) in the continuous domain have shown a capability to capture the global context, while generating high-quality images. In the discrete state space, some works have demonstrated the potential to perform text generation and low resolution image generation. We show that with the help of a content-rich discrete visual codebook from VQ-VAE, the discrete diffusion model can also generate high fidelity images with global context, which compensates for the deficiency of the classical autoregressive model along pixel space. Meanwhile, the integration of the discrete VAE with the diffusion model resolves the drawback of conventional autoregressive models being oversized, and the diffusion model which demands excessive time in the sampling process when generating images. It is found that the quality of the generated images is heavily dependent on the discrete visual codebook. Extensive experiments demonstrate that the proposed Vector Quantised Discrete Diffusion Model (VQ-DDM) is able to achieve comparable performance to top-tier methods with low complexity. It also demonstrates outstanding advantages over other vectors quantised with autoregressive models in terms of image inpainting tasks without additional training.

----

## [1113] Bridging Global Context Interactions for High-Fidelity Image Completion

**Authors**: *Chuanxia Zheng, Tat-Jen Cham, Jianfei Cai, Dinh Q. Phung*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01122](https://doi.org/10.1109/CVPR52688.2022.01122)

**Abstract**:

Bridging global context interactions correctly is important for high-fidelity image completion with large masks. Previous methods attempting this via deep or large receptive field (RF) convolutions cannot escape from the dominance of nearby interactions, which may be inferior. In this paper, we propose to treat image completion as a directionless sequence-to-sequence prediction task, and deploy a transformer to directly capture long-range depen-dence. Crucially, we employ a restrictive CNN with small and non-overlapping RF for weighted token representation, which allows the transformer to explicitly model the long-range visible context relations with equal importance in all layers, without implicitly confounding neighboring tokens when larger RFs are used. To improve appearance consistency between visible and generated regions, a novel attention-aware layer (AAL) is introduced to better exploit distantly related high-frequency features. Overall, extensive experiments demonstrate superior performance compared to state-of-the-art methods on several datasets. Code is available at https://github.com/lyndonzheng/TFill.

----

## [1114] Autoregressive Image Generation using Residual Quantization

**Authors**: *Doyup Lee, Chiheon Kim, Saehoon Kim, Minsu Cho, Wook-Shin Han*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01123](https://doi.org/10.1109/CVPR52688.2022.01123)

**Abstract**:

For autoregressive (AR) modeling of high-resolution images, vector quantization (VQ) represents an image as a sequence of discrete codes. A short sequence length is important for an AR model to reduce its computational costs to consider long-range interactions of codes. However, we postulate that previous VQ cannot shorten the code sequence and generate high-fidelity images together in terms of the rate-distortion trade-off. In this study, we propose the two-stage framework, which consists of Residual-Quantized VAE (RQ-VAE) and RQ-Transformer, to effectively generate high-resolution images. Given a fixed codebook size, RQ-VAE can precisely approximate a feature map of an image and represent the image as a stacked map of discrete codes. Then, RQ-Transformer learns to predict the quantized feature vector at the next position by predicting the next stack of codes. Thanks to the precise approximation of RQ-VAE, we can represent a 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$256\times 256$</tex>
 image as 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$8\times 8$</tex>
 resolution of the feature map, and RQ-Transformer can efficiently reduce the computational costs. Consequently, our framework out-performs the existing AR models on various benchmarks of unconditional and conditional image generation. Our approach also has a significantly faster sampling speed than previous AR models to generate high-quality images.

----

## [1115] Arbitrary-Scale Image Synthesis

**Authors**: *Evangelos Ntavelis, Mohamad Shahbazi, Iason Kastanis, Radu Timofte, Martin Danelljan, Luc Van Gool*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01124](https://doi.org/10.1109/CVPR52688.2022.01124)

**Abstract**:

Positional encodings have enabled recent works to train a single adversarial network that can generate images of different scales. However, these approaches are either limited to a set of discrete scales or struggle to maintain good perceptual quality at the scales for which the model is not trained explicitly. We propose the design of scale-consistent positional encodings invariant to our generator's layers transformations. This enables the generation of arbitrary-scale images even at scales unseen during training. Moreover, we incorporate novel inter-scale augmentations into our pipeline and partial generation training to facilitate the synthesis of consistent images at arbitrary scales. Lastly, we show competitive results for a continuum of scales on various commonly used datasets for image synthesis.

----

## [1116] Cluster-guided Image Synthesis with Unconditional Models

**Authors**: *Markos Georgopoulos, James Oldfield, Grigorios G. Chrysos, Yannis Panagakis*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01125](https://doi.org/10.1109/CVPR52688.2022.01125)

**Abstract**:

Generative Adversarial Networks (GANs) are the driving force behind the state-of-the-art in image generation. Despite their ability to synthesize high-resolution photo-realistic images, generating content with on-demand conditioning of different granularity remains a challenge. This challenge is usually tackled by annotating massive datasets with the attributes of interest, a laborious task that is not always a viable option. Therefore, it is vital to introduce control into the generation process of unsupervised generative models. In this work, we focus on controllable image generation by leveraging GANs that are well-trained in an unsupervised fashion. To this end, we discover that the representation space of intermediate layers of the generator forms a number of clusters that separate the data according to semantically meaningful attributes (e.g., hair color and pose). By conditioning on the cluster assignments, the proposed method is able to control the semantic class of the generated image. Our approach enables sampling from each cluster by Implicit Maximum Likelihood Estimation (IMLE). We showcase the efficacy of our approach on faces (CelebA-HQ and FFHQ), animals (Imagenet) and objects (LSUN) using different pre-trained generative models. The results highlight the ability of our approach to condition image generation on attributes like gender, pose and hair style on faces, as well as a variety of features on different object classes.

----

## [1117] Dynamic Prototype Convolution Network for Few-Shot Semantic Segmentation

**Authors**: *Jie Liu, Yanqi Bao, Guo-Sen Xie, Huan Xiong, Jan-Jakob Sonke, Efstratios Gavves*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01126](https://doi.org/10.1109/CVPR52688.2022.01126)

**Abstract**:

The key challenge for few-shot semantic segmentation (FSS) is how to tailor a desirable interaction among sup-port and query features and/or their prototypes, under the episodic training scenario. Most existing FSS methods im-plement such support/query interactions by solely leveraging plain operations - e.g., cosine similarity and feature concatenation - for segmenting the query objects. How-ever, these interaction approaches usually cannot well capture the intrinsic object details in the query images that are widely encountered in FSS, e.g., if the query object to be segmented has holes and slots, inaccurate segmentation al-most always happens. To this end, we propose a dynamic prototype convolution network (DPCN) to fully capture the aforementioned intrinsic details for accurate FSS. Specifi-cally, in DPCN, a dynamic convolution module (DCM) is firstly proposed to generate dynamic kernels from support foreground, then information interaction is achieved by con-volution operations over query features using these kernels. Moreover, we equip DPCN with a support activation mod-ule (SAM) and a feature filtering module (FFM) to generate pseudo mask and filter out background information for the query images, respectively. SAM and FFM together can mine enriched context information from the query features. Our DPCN is also flexible and efficient under the k-shot FSS setting. Extensive experiments on PASCAL-5
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">i</sup>
 and COCO 20
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">i</sup>
 show that DPCN yields superior performances under both 1-shot and 5-shot settings.

----

## [1118] Generalized Few-shot Semantic Segmentation

**Authors**: *Zhuotao Tian, Xin Lai, Li Jiang, Shu Liu, Michelle Shu, Hengshuang Zhao, Jiaya Jia*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01127](https://doi.org/10.1109/CVPR52688.2022.01127)

**Abstract**:

Training semantic segmentation models requires a large amount of finely annotated data, making it hard to quickly adapt to novel classes not satisfying this condition. Few- Shot Segmentation (FS-Seg) tackles this problem with many constraints. In this paper, we introduce a new benchmark, called Generalized Few-Shot Semantic Segmentation (GFS- Seg), to analyze the generalization ability of simultaneously segmenting the novel categories with very few examples and the base categories with sufficient examples. It is the first study showing that previous representative state-of-the-art FS-Seg methods fall short in GFS-Seg and the performance discrepancy mainly comes from the constrained setting of FS-Seg. To make GFS-Seg tractable, we set up a GFS-Seg baseline that achieves decent performance without structural change on the original model. Then, since context is essential for semantic segmentation, we propose the Context-Aware Prototype Learning (CAPL) that significantly improves performance by 1) leveraging the co-occurrence prior knowledge from support samples, and 2) dynamically enriching contextual information to the classifier, conditioned on the content of each query image. Both two contributions are experimentally manifested for their substantial practical merit. Extensive experiments on Pascal-Voc and COCO also show that CAPL generalizes well to FS-Seg by achieving competitive performance. Code is available at https://github.com/dvlab-research/GFS-Seg.

----

## [1119] Learning Non-target Knowledge for Few-shot Semantic Segmentation

**Authors**: *Yuanwei Liu, Nian Liu, Qinglong Cao, Xiwen Yao, Junwei Han, Ling Shao*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01128](https://doi.org/10.1109/CVPR52688.2022.01128)

**Abstract**:

Existing studies in few-shot semantic segmentation only focus on mining the target object information, however, often are hard to tell ambiguous regions, especially in non-target regions, which include background (BG) and Distracting Objects (DOs). To alleviate this problem, we propose a novel framework, namely Non-Target Region Eliminating (NTRE) network, to explicitly mine and eliminate BG and DO regions in the query. First, a BG Mining Module (BGMM) is proposed to extract the BG region via learning a general BG prototype. To this end, we design a BG loss to supervise the learning of BGMM only using the known target object segmentation ground truth. Then, a BG Eliminating Module and a DO Eliminating Module are proposed to successively filter out the BG and DO information from the query feature, based on which we can obtain a BG and DO-free target object segmentation result. Furthermore, we propose a prototypical contrastive learning algorithm to improve the model ability of distinguishing the target object from DOs. Extensive experiments on both PASCAL-5
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">i</sup>
 and COCO-20
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">i</sup>
 datasets show that our approach is effective despite its simplicity. Code is available at https://github.com/LIUYUANWEI98/NERTNet

----

## [1120] Decoupling Zero-Shot Semantic Segmentation

**Authors**: *Jian Ding, Nan Xue, Gui-Song Xia, Dengxin Dai*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01129](https://doi.org/10.1109/CVPR52688.2022.01129)

**Abstract**:

Zero-shot semantic segmentation (ZS3) aims to segment the novel categories that have not been seen in the training. Existing works formulate ZS3 as a pixel-level zeroshot classification problem, and transfer semantic knowledge from seen classes to unseen ones with the help of language models pre-trained only with texts. While simple, the pixel-level ZS3 formulation shows the limited capability to integrate vision-language models that are often pre-trained with image-text pairs and currently demonstrate great potential for vision tasks. Inspired by the observation that humans often perform segment-level semantic labeling, we propose to decouple the ZS3 into two sub-tasks: 1) a classagnostic grouping task to group the pixels into segments. 2) a zero-shot classification task on segments. The former task does not involve category information and can be directly transferred to group pixels for unseen classes. The latter task performs at segment-level and provides a natural way to leverage large-scale vision-language models pre-trained with image-text pairs (e.g. CLIP) for ZS3. Based on the decoupling formulation, we propose a simple and effective zero-shot semantic segmentation model, called ZegFormer, which outperforms the previous methods on ZS3 standard benchmarks by large margins, e.g., 22 points on the PAS-CAL VOC and 3 points on the COCO-Stuff in terms of mIoU for unseen classes. Code will be released at https://github.com/dingjiansw101/ZegFormer.

----

## [1121] Class-Balanced Pixel-Level Self-Labeling for Domain Adaptive Semantic Segmentation

**Authors**: *Ruihuang Li, Shuai Li, Chenhang He, Yabin Zhang, Xu Jia, Lei Zhang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01130](https://doi.org/10.1109/CVPR52688.2022.01130)

**Abstract**:

Domain adaptive semantic segmentation aims to learn a model with the supervision of source domain data, and produce satisfactory dense predictions on unlabeled target domain. One popular solution to this challenging task is self-training, which selects high-scoring predictions on target samples as pseudo labels for training. However, the produced pseudo labels often contain much noise because the model is biased to source domain as well as majority categories. To address the above issues, we propose to di-rectly explore the intrinsic pixel distributions of target do-main data, instead of heavily relying on the source domain. Specifically, we simultaneously cluster pixels and rectify pseudo labels with the obtained cluster assignments. This process is done in an online fashion so that pseudo labels could co-evolve with the segmentation model without extra training rounds. To overcome the class imbalance problem on long-tailed categories, we employ a distribution align-ment technique to enforce the marginal class distribution of cluster assignments to be close to that of pseudo labels. The proposed method, namely Class-balanced Pixel-level Self-Labeling (CPSL), improves the segmentation performance on target domain over state-of-the-arts by a large margin, especially on long-tailed categories. The source code is available at ht tps: / / gi thub. com/lslrh/CPSL.

----

## [1122] ContrastMask: Contrastive Learning to Segment Every Thing

**Authors**: *Xuehui Wang, Kai Zhao, Ruixin Zhang, Shouhong Ding, Yan Wang, Wei Shen*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01131](https://doi.org/10.1109/CVPR52688.2022.01131)

**Abstract**:

Partially-supervised instance segmentation is a task which requests segmenting objects from novel categories via learning on limited base categories with annotated masks thus eliminating demands of heavy annotation burden. The key to addressing this task is to build an effective class-agnostic mask segmentation model. Unlike previous methods that learn such models only on base categories, in this paper, we propose a new method, named ContrastMask, which learns a mask segmentation model on both base and novel categories under a unified pixel-level contrastive learning framework. In this framework, annotated masks of base categories and pseudo masks of novel categories serve as a prior for contrastive learning, where features from the mask regions (foreground) are pulled together, and are contrasted against those from the background, and vice versa. Through this framework, feature discrimination between foreground and background is largely improved, facilitating learning of the class-agnostic mask segmentation model. Exhaustive experiments on the COCO dataset demonstrate the superiority of our method, which outperforms previous state-of-the-arts.

----

## [1123] The Neurally-Guided Shape Parser: Grammar-based Labeling of 3D Shape Regions with Approximate Inference

**Authors**: *R. Kenny Jones, Aalia Habib, Rana Hanocka, Daniel Ritchie*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01132](https://doi.org/10.1109/CVPR52688.2022.01132)

**Abstract**:

We propose the Neurally-Guided Shape Parser (NGSP), a method that learns how to assign fine-grained semantic labels to regions of a 3D shape. NGSP solves this problem via MAP inference, modeling the posterior probability of a label assignment conditioned on an input shape with a learned likelihood function. To make this search tractable, NGSP employs a neural guide network that learns to approximate the posterior. NGSP finds high-probability label assignments by first sampling proposals with the guide network and then evaluating each proposal under the full likelihood. We evaluate NGSP on the task of fine-grained semantic segmentation of man ufactured 3D shapesfrom PartNet, where shapes have been decomposed into regions that correspond to part instance over-segmentations. We find that NGSP delivers significant performance improvements over comparison methods that (i) use regions to group per-point predictions, (ii) use regions as a self-supervisory signal or (iii) assign labels to regions under alternative formulations. Further, we show that NGSP maintains strong performance even with limited labeled data or noisy input shape regions. Finally, we demonstrate that NGSP can be directly applied to CAD shapes found in online repositories and validate its effectiveness with a perceptual study.

----

## [1124] AutoGPart: Intermediate Supervision Search for Generalizable 3D Part Segmentation

**Authors**: *Xueyi Liu, Xiaomeng Xu, Anyi Rao, Chuang Gan, Li Yi*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01133](https://doi.org/10.1109/CVPR52688.2022.01133)

**Abstract**:

Training a generalizable 3D part segmentation network is quite challenging but of great importance in real-world applications. To tackle this problem, some works design task-specific solutions by translating human understanding of the task to machine's learning process, which faces the risk of missing the optimal strategy since machines do not necessarily understand in the exact human way. Others try to use conventional task-agnostic approaches designed for domain generalization problems with no task prior knowledge considered. To solve the above issues, we propose AutoGPart, a generic method enabling training generalizable 3D part segmentation networks with the task prior considered. AutoGPart builds a supervision space with geometric prior knowledge encoded, and lets the machine to search for the optimal supervisions from the space for a specific segmentation task automatically. Extensive experiments on three generalizable 3D part segmentation tasks are conducted to demonstrate the effectiveness and versatility of AutoGPart. We demonstrate that the performance of segmentation networks using simple backbones can be significantly improved when trained with supervisions searched by our method.

----

## [1125] APES: Articulated Part Extraction from Sprite Sheets

**Authors**: *Zhan Xu, Matthew Fisher, Yang Zhou, Deepali Aneja, Rushikesh Dudhat, Li Yi, Evangelos Kalogerakis*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01134](https://doi.org/10.1109/CVPR52688.2022.01134)

**Abstract**:

Rigged puppets are one of the most prevalent representations to create 2D character animations. Creating these puppets requires partitioning characters into independently moving parts. In this work, we present a method to automatically identify such articulated parts from a small set of character poses shown in a sprite sheet, which is an illustration of the character that artists often draw before puppet creation. Our method is trained to infer articulated parts, e.g. head, torso and limbs, that can be reassembled to best reconstruct the given poses. Our results demonstrate significantly better performance than alternatives qualitatively and quantitatively. Our project page https://zhan-xu.github.io/parts/ includes our code and data.

----

## [1126] GASP, a generalized framework for agglomerative clustering of signed graphs and its application to Instance Segmentation

**Authors**: *Alberto Bailoni, Constantin Pape, Nathan Hütsch, Steffen Wolf, Thorsten Beier, Anna Kreshuk, Fred A. Hamprecht*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01135](https://doi.org/10.1109/CVPR52688.2022.01135)

**Abstract**:

We propose a theoretical framework that generalizes simple and fast algorithms for hierarchical agglomerative clustering to weighted graphs with both attractive and repulsive interactions between the nodes. This framework defines GASP, a Generalized Algorithm for Signed graph Partitioning
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
Code available at: https://github.com/abailoni/GASP, and allows us to explore many combinations of different linkage criteria and cannotlink constraints. We prove the equivalence of existing clustering methods to some of those combinations and introduce new algorithms for combinations that have not been studied before. We study both theoretical and empirical properties of these combinations and prove that some of these define an ultrametric on the graph. We conduct a systematic comparison of various instantiations of GASP on a large variety of both synthetic and existing signed clustering problems, in terms of accuracy but also efficiency and robustness to noise. Lastly, we show that some of the algorithms included in our framework, when combined with the predictions from a CNN model, result in a simple bottom-up instance segmentation pipeline. Going all the way from pixels to final segments with a simple procedure, we achieve state-of-the-art accuracy on the CREMI 2016 EM segmentation benchmark without requiring domain-specific superpixels.

----

## [1127] CycleMix: A Holistic Strategy for Medical Image Segmentation from Scribble Supervision

**Authors**: *Ke Zhang, Xiahai Zhuang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01136](https://doi.org/10.1109/CVPR52688.2022.01136)

**Abstract**:

Curating a large set of fully annotated training data can be costly, especially for the tasks of medical image segmentation. Scribble, a weaker form of annotation, is more obtainable in practice, but training segmentation models from limited supervision of scribbles is still challenging. To address the difficulties, we propose a new framework for scribble learning-based medical image segmentation, which is composed of mix augmentation and cycle consistency and thus is referred to as CycleMix. For augmentation of supervision, CycleMix adopts the mixup strategy with a dedicated design of random occlusion, to perform increments and decrements of scribbles. For regularization of supervision, CycleMix intensifies the training objective with consistency losses to penalize inconsistent segmentation, which results in significant improvement of segmentation performance. Results on two open datasets, i.e., ACDC and MSCMRseg, showed that the proposed method achieved exhilarating performance, demonstrating comparable or even better accuracy than the fully-supervised methods. The code and expert-made scribble annotationsfor MSCMRseg are publicly available at https://github.com/BWGZK/CycleMix.

----

## [1128] Cross-patch Dense Contrastive Learning for Semi-supervised Segmentation of Cellular Nuclei in Histopathologic Images

**Authors**: *Huisi Wu, Zhaoze Wang, Youyi Song, Lin Yang, Jing Qin*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01137](https://doi.org/10.1109/CVPR52688.2022.01137)

**Abstract**:

We study the semi-supervised learning problem, using a few labeled data and a large amount of unlabeled data to train the network, by developing a cross-patch dense contrastive learning framework, to segment cellular nuclei in histopathologic images. This task is motivated by the expensive burden on collecting labeled data for histopathologic image segmentation tasks. The key idea of our method is to align features of teacher and student networks, sampled from cross-image in both patch- and pixel-levels, for enforcing the intra-class compactness and inter-class separability of features that as we shown is helpful for extracting valuable knowledge from unlabeled data. We also design a novel optimization framework that combines consistency regularization and entropy minimization techniques, showing good property in eviction of gradient vanishing. We assess the proposed method on two publicly available datasets, and obtain positive results on extensive experiments, outperforming the state-of-the-art methods. Codes are available at https://github.com/zzw-szu/CDCL.

----

## [1129] C-CAM: Causal CAM for Weakly Supervised Semantic Segmentation on Medical Image

**Authors**: *Zhang Chen, Zhiqiang Tian, Jihua Zhu, Ce Li, Shaoyi Du*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01138](https://doi.org/10.1109/CVPR52688.2022.01138)

**Abstract**:

Recently, many excellent weakly supervised semantic segmentation (WSSS) works are proposed based on class activation mapping (CAM). However, there are few works that consider the characteristics of medical images. In this paper, we find that there are mainly two challenges of medical images in WSSS: i) the boundary of object foreground and background is not clear; ii) the co-occurrence phenomenon is very severe in training stage. We thus propose a Causal CAM (C-CAM) method to overcome the above challenges. Our method is motivated by two cause-effect chains including category-causality chain and anatomy-causality chain. The category-causality chain represents the image content (cause) affects the category (effect). The anatomy-causality chain represents the anatomical structure (cause) affects the organ segmentation (effect). Extensive experiments were conducted on three public medical image data sets. Our C-CAM generates the best pseudo masks with the DSC of 77.26%, 80.34% and 78.15% on ProMRI, ACDC and CHAOS compared with other CAM-like methods. The pseudo masks of C-CAM are further used to improve the segmentation performance for organ segmentation tasks. Our C-CAM achieves DSC of 83.83% on ProMRI and DSC of 87.54% on ACDC, which outperforms state-of-the-art WSSS methods. Our code is available at https://github.com/Tian-lab/C-CAM.

----

## [1130] CRIS: CLIP-Driven Referring Image Segmentation

**Authors**: *Zhaoqing Wang, Yu Lu, Qiang Li, Xunqiang Tao, Yandong Guo, Mingming Gong, Tongliang Liu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01139](https://doi.org/10.1109/CVPR52688.2022.01139)

**Abstract**:

Referring image segmentation aims to segment a referent via a natural linguistic expression. Due to the distinct data properties between text and image, it is challenging for a network to well align text and pixel-level features. Existing approaches use pretrained models to facilitate learning, yet separately transfer the language/vision knowledge from pretrained models, ignoring the multi-modal corresponding information. Inspired by the recent advance in Contrastive Language-Image Pretraining (CLIP), in this paper, we propose an end-to-end CLIP-Driven Referring Image Segmen-tation framework (CRIS). To transfer the multi-modal knowledge effectively, CRIS resorts to vision-language decoding and contrastive learning for achieving the text-to-pixel alignment. More specifically, we design a vision-language decoder to propagate fine-grained semantic information from textual representations to each pixel-level activation, which promotes consistency between the two modalities. In addition, we present text-to-pixel contrastive learning to explicitly enforce the text feature similar to the related pixel-level features and dissimilar to the irrelevances. The experimental results on three benchmark datasets demonstrate that our proposed framework significantly outperforms the state-of-the-art performance without any post-processing.

----

## [1131] MatteFormer: Transformer-Based Image Matting via Prior-Tokens

**Authors**: *Gyutae Park, Sungjoon Son, Jaeyoung Yoo, Seho Kim, Nojun Kwak*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01140](https://doi.org/10.1109/CVPR52688.2022.01140)

**Abstract**:

In this paper, we propose a transformer-based image matting model called MatteFormer, which takes full advantage of trimap information in the transformer block. Our method first introduces a prior-token which is a global representation of each trimap region (e.g. foreground, background and unknown). These prior-tokens are used as global priors and participate in the self-attention mechanism of each block. Each stage of the encoder is composed of PAST (Prior-Attentive Swin Transformer) block, which is based on the Swin Transformer block, but differs in a couple of aspects: 1) It has PA-WSA (Prior-Attentive Window Self-Attention) layer, performing self-attention not only with spatial-tokens but also with prior-tokens. 2) It has prior-memory which saves prior-tokens accumulatively from the previous blocks and transfers them to the next block. We evaluate our MatteFormer on the commonly used image matting datasets: Composition-Ik and Distinctions-646. Experiment results show that our proposed method achieves state-of-the-art performance with a large margin. Our codes are available at https://github.com/webtoon/matteformer.

----

## [1132] Boosting Robustness of Image Matting with Context Assembling and Strong Data Augmentation

**Authors**: *Yutong Dai, Brian Price, He Zhang, Chunhua Shen*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01141](https://doi.org/10.1109/CVPR52688.2022.01141)

**Abstract**:

Deep image matting methods have achieved increasingly better results on benchmarks (e.g., Composition-1k/alphamatting.com). However, the robustness, including robustness to trimaps and generalization to images from different domains, is still underexplored. Although some works propose to either refine the trimaps or adapt the algorithms to real-world images via extra data augmentation, none of them has taken both into consideration, not to mention the significant performance deterioration on benchmarks while using those data augmentation. To fill this gap, we propose an image matting method which achieves higher robustness (RMat) via multilevel context assembling and strong data augmentation targeting matting. Specifically, we first build a strong matting framework by modeling ample global information with transformer blocks in the encoder, and focusing on details in combination with convolution layers as well as a low-level feature assembling attention block in the decoder. Then, based on this strong baseline, we analyze current data augmentation and explore simple but effective strong data augmentation to boost the baseline model and contribute a more generalizable matting method. Compared with previous methods, the proposed method not only achieves state-of-the-art results on the Composition-1k benchmark (11 % improvement on SAD and 27% improvement on Grad) with smaller model size, but also shows more robust generalization results on other benchmarks, on real-world images, and also on varying coarse-to-fine trimaps with our extensive experiments.
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
This work was in part done when YD was an intern at Adobe and CS was with The University of Adelaide. CS is the corresponding author. Project page: https://dongdong93.github.io/RMat/.

----

## [1133] Pyramid Grafting Network for One-Stage High Resolution Saliency Detection

**Authors**: *Chenxi Xie, Changqun Xia, Mingcan Ma, Zhirui Zhao, Xiaowu Chen, Jia Li*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01142](https://doi.org/10.1109/CVPR52688.2022.01142)

**Abstract**:

Recent salient object detection (SOD) methods based on deep neural network have achieved remarkable performance. However, most of existing SOD models designed for low-resolution input perform poorly on high-resolution images due to the contradiction between the sampling depth and the receptive field size. Aiming at resolving this con-tradiction, we propose a novel one-stage framework called Pyramid Grafting Network (PGNet), using transformer and CNN backbone to extract features from different resolution images independently and then graft the features from transformer branch to CNN branch. An attention-based Cross-Model Grafting Module (CMGM) is proposed to en-able CNN branch to combine broken detailed information more holistically, guided by different source feature during decoding process. Moreover, we design an Attention Guided Loss (AGL) to explicitly supervise the attention matrix generated by CMGM to help the network better interact with the attention from different models. We contribute a new Ultra-High-Resolution Saliency Detection dataset UHRSD, containing 5,920 images at 4K-SK resolutions. To our knowledge, it is the largest dataset in both quantity and resolution for high-resolution SOD task, which can be used for training and testing in future research. Sufficient exper-iments on UHRSD and widely-used SOD datasets demon-strate that our method achieves superior performance compared to the state-of-the-art methods.

----

## [1134] Multi-Source Uncertainty Mining for Deep Unsupervised Saliency Detection

**Authors**: *Yifan Wang, Wenbo Zhang, Lijun Wang, Ting Liu, Huchuan Lu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01143](https://doi.org/10.1109/CVPR52688.2022.01143)

**Abstract**:

Deep learning-based image salient object detection (SOD) heavily relies on large-scale training data with pixel-wise labeling. High-quality labels involve intensive labor and are expensive to acquire. In this paper, we propose a novel multi-source uncertainty mining method to facilitate unsupervised deep learning from multiple noisy labels generated by traditional handcrafted SOD methods. We design an Uncertainty Mining Network (UMNet) which consists of multiple Merge-and-Split (MS) modules to recursively analyze the commonality and difference among multiple noisy labels and infer pixel-wise uncertainty map for each label. Meanwhile, we model the noisy labels using Gibbs distribution and propose a weighted uncertainty loss to jointly train the UMNet with the SOD network. As a consequence, our UMNet can adaptively select reliable labels for SOD network learning. Extensive experiments on benchmark datasets demonstrate that our method not only outperforms existing unsupervised methods, but also is on par with fully-supervised state-of-the-art models.

----

## [1135] Modeling Motion with Multi-Modal Features for Text-Based Video Segmentation

**Authors**: *Wangbo Zhao, Kai Wang, Xiangxiang Chu, Fuzhao Xue, Xinchao Wang, Yang You*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01144](https://doi.org/10.1109/CVPR52688.2022.01144)

**Abstract**:

Text-based video segmentation aims to segment the target object in a video based on a describing sentence. Incorporating motion information from optical flow maps with appearance and linguistic modalities is crucial yet has been largely ignored by previous work. In this paper, we design a method to fuse and align appearance, motion, and linguistic features to achieve accurate segmentation. Specifically, we propose a multi-modal video transformer, which can fuse and aggregate multi-modal and temporal features between frames. Furthermore, we design a language-guided feature fusion module to progressively fuse appearance and motion features in each feature level with guidance from linguistic features. Finally, a multi-modal alignment loss is proposed to alleviate the semantic gap between features from different modalities. Extensive experiments on A2D Sentences and J-HMDB Sentences verify the performance and the generalization ability of our method compared to the state-of-the-art methods.

----

## [1136] GAT-CADNet: Graph Attention Network for Panoptic Symbol Spotting in CAD Drawings

**Authors**: *Zhaohua Zheng, Jianfang Li, Lingjie Zhu, Honghua Li, Frank Petzold, Ping Tan*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01145](https://doi.org/10.1109/CVPR52688.2022.01145)

**Abstract**:

Spotting graphical symbols from the computer-aided design (CAD) drawings is essential to many industrial applications. Different from raster images, CAD drawings are vector graphics consisting of geometric primitives such as segments, arcs, and circles. By treating each CAD drawing as a graph, we propose a novel graph attention network GAT-CADNet to solve the panoptic symbol spotting problem: vertex features derived from the GAT branch are mapped to semantic labels, while their attention scores are cascaded and mapped to instance prediction. Our key contributions are three-fold: 1) the instance symbol spotting task is formulated as a subgraph detection problem and solved by predicting the adjacency matrix; 2) a relative spatial encoding (RSE) module explicitly encodes the relative positional and geometric relation among vertices to enhance the vertex attention; 3) a cascaded edge encoding (CEE) module extracts vertex attentions from multiple stages of GAT and treats them as edge encoding to predict the adjacency matrix. The proposed GAT-CADNet is intuitive yet effective and manages to solve the panoptic symbol spotting problem in one consolidated network. Extensive experiments and ablation studies on the public benchmark show that our graph-based approach surpasses existing state-of-the-art methods by a large margin.

----

## [1137] Bending Graphs: Hierarchical Shape Matching using Gated Optimal Transport

**Authors**: *Mahdi Saleh, Shun-Cheng Wu, Luca Cosmo, Nassir Navab, Benjamin Busam, Federico Tombari*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01146](https://doi.org/10.1109/CVPR52688.2022.01146)

**Abstract**:

Shape matching has been a long-studied problem for the computer graphics and vision community. The objective is to predict a dense correspondence between meshes that have a certain degree of deformation. Existing methods either consider the local description of sampled points or discover correspondences based on global shape information. In this work, we investigate a hierarchical learning design, to which we incorporate local patch-level information and global shape-level structures. This flexible representation enables correspondence prediction and provides rich features for the matching stage. Finally, we propose a novel optimal transport solver by recurrently updating features on non-confident nodes to learn globally consistent correspondences between the shapes. Our results on publicly available datasets suggest robust performance in presence of severe deformations without the need of extensive training or refinement.

----

## [1138] CAPRI-Net: Learning Compact CAD Shapes with Adaptive Primitive Assembly

**Authors**: *Fenggen Yu, Zhiqin Chen, Manyi Li, Aditya Sanghi, Hooman Shayani, Ali Mahdavi-Amiri, Hao Zhang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01147](https://doi.org/10.1109/CVPR52688.2022.01147)

**Abstract**:

We introduce CAPRI-Net, a self-supervised neural network for learning compact and interpretable implicit representations of 3D computer-aided design (CAD) models, in the form of adaptive primitive assemblies. Given an input 3D shape, our network reconstructs it by an assembly of quadric surface primitives via constructive solid geometry (CSG) operations. Without any ground-truth shape assemblies, our self-supervised network is trained with a reconstruction loss, leading to faithful 3D reconstructions with sharp edges and plausible CSG trees. While the parametric nature of CAD models does make them more predictable locally, at the shape level, there is much structural and topological variation, which presents a significant generalizability challenge to state-of-the-art neural models for 3D shapes. Our network addresses this challenge by adaptive training with respect to each test shape, with which we fine-tune the network that was pre-trained on a model collection. We evaluate our learning framework on both ShapeNet and ABC, the largest and most diverse CAD dataset to date, in terms of reconstruction quality, sharp edges, compactness, and interpretability, to demonstrate superiority over current alternatives for neural CAD reconstruction.

----

## [1139] RIM-Net: Recursive Implicit Fields for Unsupervised Learning of Hierarchical Shape Structures

**Authors**: *Chengjie Niu, Manyi Li, Kai Xu, Hao Zhang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01148](https://doi.org/10.1109/CVPR52688.2022.01148)

**Abstract**:

We introduce RIM-Net, a neural network which learns recursive implicit fields for unsupervised inference of hierarchical shape structures. Our network recursively decomposes an input 3D shape into two parts, resulting in a binary tree hierarchy. Each level of the tree corresponds to an assembly of shape parts, represented as implicit functions, to reconstruct the input shape. At each node of the tree, simultaneous feature decoding and shape decomposition are carried out by their respective feature and part decoders, with weight sharing across the same hierarchy level. As an implicit field decoder, the part decoder is designed to decompose a sub-shape, via a two-way branched reconstruction, where each branch predicts a set of parameters defining a Gaussian to serve as a local point distribution for shape reconstruction. With reconstruction losses accounted for at each hierarchy level and a decomposition loss at each node, our network training does not require any ground-truth segmentations, let alone hierarchies. Through extensive experiments and comparisons to state-of-the-art alternatives, we demonstrate the quality, consistency, and interpretability of hierarchical structural inference by RIM-Net.

----

## [1140] Discovering Objects that Can Move

**Authors**: *Zhipeng Bao, Pavel Tokmakov, Allan Jabri, Yu-Xiong Wang, Adrien Gaidon, Martial Hebert*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01149](https://doi.org/10.1109/CVPR52688.2022.01149)

**Abstract**:

This paper studies the problem of object discovery - separating objects from the background without manual labels. Existing approaches utilize appearance cues, such as color, texture, and location, to group pixels into object-like regions. However, by relying on appearance alone, these methods fail to separate objects from the background in cluttered scenes. This is a fundamental limitation since the definition of an object is inherently ambiguous and context-dependent. To resolve this ambiguity, we choose to focus on dynamic objects - entities that can move independently in the world. We then scale the recent auto-encoder based frameworks for unsuper-vised object discovery from toy synthetic images to complex real-world scenes. To this end, we simplify their architecture, and augment the resulting model with a weak learning signal from general motion segmentation algorithms. Our experiments demonstrate that, despite only capturing a small subset of the objects that move, this signal is enough to generalize to segment both moving and static instances of dynamic objects. We show that our model scales to a newly collected, photo- realistic synthetic dataset with street driving scenarios. Additionally, we leverage ground truth segmentation and flow annotations in this dataset for thorough ablation and evaluation. Finally, our experiments on the real-world KITTI benchmark demonstrate that the proposed approach outperforms both heuristic- and learning-based methods by capitalizing on motion cues.

----

## [1141] PatchFormer: An Efficient Point Transformer with Patch Attention

**Authors**: *Cheng Zhang, Haocheng Wan, Xinyi Shen, Zizhao Wu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01150](https://doi.org/10.1109/CVPR52688.2022.01150)

**Abstract**:

The point cloud learning community witnesses a modeling shift from CNNs to Transformers, where pure Transformer architectures have achieved top accuracy on the major learning benchmarks. However, existing point Transformers are computationally expensive since they need to generate a large attention map, which has quadratic complexity (both in space and time) with respect to input size. To solve this shortcoming, we introduce Patch ATtention (PAT) to adaptively learn a much smaller set of bases upon which the attention maps are computed. By a weighted summation upon these bases, PAT not only captures the global shape context but also achieves linear complexity to input size. In addition, we propose a lightweight Multi-Scale aTtention (MST) block to build attentions among features of different scales, providing the model with multi-scale features. Equipped with the PAT and MST, we construct our neural architecture called PatchFormer that integrates both modules into a joint framework for point cloud learning. Extensive experiments demonstrate that our network achieves comparable accuracy on general point cloud learning tasks with 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$9.2\times$</tex>
 speed-up than previous point Transformers.

----

## [1142] Panoptic-PHNet: Towards Real-Time and High-Precision LiDAR Panoptic Segmentation via Clustering Pseudo Heatmap

**Authors**: *Jinke Li, Xiao He, Yang Wen, Yuan Gao, Xiaoqiang Cheng, Dan Zhang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01151](https://doi.org/10.1109/CVPR52688.2022.01151)

**Abstract**:

As a rising task, panoptic segmentation is faced with challenges in both semantic segmentation and instance seg-mentation. However, in terms of speed and accuracy, ex-isting LiDAR methods in the field are still limited. In this paper, we propose a fast and high-performance LiDAR-basedframework, referred to as Panoptic-PHNet, with three attractive aspects: 1) We introduce a clustering pseudo heatmap as a new paradigm, which, followed by a cen-ter grouping module, yields instance centers for efficient clustering without object-level learning tasks. 2) A knn-transformer module is proposed to model the interaction among foreground points for accurate offset regression. 3) For backbone design, we fuse the fine- grained voxel features and the 2D Bird's Eye View (BEV) features with different receptive fields to utilize both detailed and global information. Extensive experiments on both SemanticKITTI dataset and nuScenes dataset show that our Panoptic-PHNet sur-passes state-of-the-art methods by remarkable margins with a real-time speed. We achieve the 1st place on the public leaderboard of SemanticKITTI and leading performance on the recently released leaderboard of nuScenes.

----

## [1143] SemAffiNet: Semantic-Affine Transformation for Point Cloud Segmentation

**Authors**: *Ziyi Wang, Yongming Rao, Xumin Yu, Jie Zhou, Jiwen Lu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01152](https://doi.org/10.1109/CVPR52688.2022.01152)

**Abstract**:

Conventional point cloud semantic segmentation methods usually employ an encoder-decoder architecture, where mid-level features are locally aggregated to extract geometric information. However, the over-reliance on these class-agnostic local geometric representations may raise confusion between local parts from different categories that are similar in appearance or spatially adjacent. To address this issue, we argue that mid-level features can be further enhanced with semantic information, and propose semantic-affine transformation that transforms features of mid-level points belonging to different categories with class-specific affine parameters. Based on this technique, we propose SemAffiNet for point cloud semantic segmentation, which utilizes the attention mechanism in the Transformer module to implicitly and explicitly capture global structural knowledge within local parts for overall comprehension of each category. We conduct extensive experiments on the ScanNetV2 and NYUv2 datasets, and evaluate semantic-affine transformation on various 3D point cloud and 2D image segmentation baselines, where both qualitative and quantitative results demonstrate the superiority and generalization ability of our proposed approach. Code is available at https://github.com/wangzy22/SemAffiNet.

----

## [1144] An MIL-Derived Transformer for Weakly Supervised Point Cloud Segmentation

**Authors**: *Cheng-Kun Yang, Ji-Jia Wu, Kai-Syun Chen, Yung-Yu Chuang, Yen-Yu Lin*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01153](https://doi.org/10.1109/CVPR52688.2022.01153)

**Abstract**:

We address weakly supervised point cloud segmentation by proposing a new model, MIL-derived transformer, to mine additional supervisory signals. First, the transformer model is derived based on multiple instance learning (MIL) to explore pair-wise cloud-level supervision, where two clouds of the same category yield a positive bag while two of different classes produce a negative bag. It leverages not only individual cloud annotations but also pair-wise cloud semantics for model optimization. Second, Adaptive global weighted pooling (AdaGWP) is integrated into our transformer model to replace max pooling and average pooling. It introduces learnable weights to re-scale logits in the class activation maps. It is more robust to noise while discovering more complete foreground points under weak supervision. Third, we perform point subsampling and enforce feature equivariance between the original and subsampled point clouds for regularization. The proposed method is end-to-end trainable and is general because it can work with different backbones with diverse types of weak supervision signals, including sparsely annotated points and cloud-level labels. The experiments show that it achieves state-of-the-art performance on the S3DIS and ScanNet benchmarks. The source code will be available at https://github.com/jimmy15923/wspss_mil_transformer.

----

## [1145] Weakly Supervised Segmentation on Outdoor 4D point clouds with Temporal Matching and Spatial Graph Propagation

**Authors**: *Hanyu Shi, Jiacheng Wei, Ruibo Li, Fayao Liu, Guosheng Lin*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01154](https://doi.org/10.1109/CVPR52688.2022.01154)

**Abstract**:

Existing point cloud segmentation methods require a large amount of annotated data, especially for the outdoor point cloud scene. Due to the complexity of the outdoor 3D scenes, manual annotations on the outdoor point cloud scene are time-consuming and expensive. In this paper, we study how to achieve scene understanding with limited annotated data. Treating 100 consecutive frames as a sequence, we divide the whole dataset into a series of sequences and annotate only 0.1% points in the first frame of each sequence to reduce the annotation requirements. This leads to a total annotation budget of 0.001%. We propose a novel temporal-spatial framework for effective weakly supervised learning to generate high-quality pseudo labels from these limited annotated data. Specifically, the frame-work contains two modules: an matching module in temporal dimension to propagate pseudo labels across different frames, and a graph propagation module in spatial dimension to propagate the information of pseudo labels to the entire point clouds in each frame. With only 0.001% annotations for training, experimental results on both SemanticKITTI and SemanticPOSS shows our weakly supervised two-stage framework is comparable to some existing fully supervised methods. We also evaluate our framework with 0.005% initial annotations on SemanticKITTI, and achieve a result close to fully supervised backbone model.

----

## [1146] Point2Cyl: Reverse Engineering 3D Objects from Point Clouds to Extrusion Cylinders

**Authors**: *Mikaela Angelina Uy, Yen-Yu Chang, Minhyuk Sung, Purvi Goel, Joseph Lambourne, Tolga Birdal, Leonidas J. Guibas*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01155](https://doi.org/10.1109/CVPR52688.2022.01155)

**Abstract**:

We propose Point2Cyl, a supervised network transforming a raw 3D point cloud to a set of extrusion cylinders. Reverse engineering from a raw geometry to a CAD model is an essential task to enable manipulation of the 3D data in shape editing software and thus expand their usages in many downstream applications. Particularly, the form of CAD models having a sequence of extrusion cylinders - a 2D sketch plus an extrusion axis and range - and their boolean combinations is not only widely used in the CAD community/software but also has great expressivity of shapes, compared to having limited types of primitives (e.g., planes, spheres, and cylinders). In this work, we introduce a neural network that solves the extrusion cylinder decomposition problem in a geometry-grounded way by first learning underlying geometric proxies. Precisely, our approach first predicts per-point segmentation, base/barrel labels and normals, then estimates for the underlying extrusion parameters in differentiable and closed-form formulations. Our experiments show that our approach demonstrates the best performance on two recent CAD datasets, Fusion Gallery and DeepCAD, and we further showcase our approach on reverse engineering and editing.

----

## [1147] Demystifying the Neural Tangent Kernel from a Practical Perspective: Can it be trusted for Neural Architecture Search without training?

**Authors**: *Jisoo Mok, Byunggook Na, Ji-Hoon Kim, Dongyoon Han, Sungroh Yoon*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01156](https://doi.org/10.1109/CVPR52688.2022.01156)

**Abstract**:

In Neural Architecture Search (NAS), reducing the cost of architecture evaluation remains one of the most crucial challenges. Among a plethora of efforts to bypass training of each candidate architecture to convergence for evaluation, the Neural Tangent Kernel (NTK) is emerging as a promising theoretical framework that can be utilized to estimate the performance of a neural architecture at initialization. In this work, we revisit several at-initialization metrics that can be derived from the NTK and reveal their key short-comings. Then, through the empirical analysis of the time evolution of NTK, we deduce that modern neural architectures exhibit highly non-linear characteristics, making the NTK-based metrics incapable of reliably estimating the performance of an architecture without some amount of training. To take such non-linear characteristics into account, we introduce Label-Gradient Alignment (LGA), a novel NTK-based metric whose inherent formulation allows it to capture the large amount of non-linear advantage present in modern neural architectures. With minimal amount of training, LGA obtains a meaningful level of rank correlation with the final test accuracy of an architecture. Lastly, we demonstrate that LGA, complemented with few epochs of training, successfully guides existing search algorithms to achieve competitive search performances with significantly less search cost. The code is available at: https://github.com/nute11amok/DemystifyingNTK.

----

## [1148] BaLeNAS: Differentiable Architecture Search via the Bayesian Learning Rule

**Authors**: *Miao Zhang, Shirui Pan, Xiaojun Chang, Steven Su, Jilin Hu, Gholamreza Haffari, Bin Yang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01157](https://doi.org/10.1109/CVPR52688.2022.01157)

**Abstract**:

Differentiable Architecture Search (DARTS) has received massive attention in recent years, mainly because it significantly reduces the computational cost through weight sharing and continuous relaxation. However, more recent works find that existing differentiable NAS techniques struggle to outperform naive baselines, yielding deteriorative architectures as the search proceeds. Rather than directly optimizing the architecture parameters, this paper formulates the neural architecture search as a distribution learning problem through relaxing the architecture weights into Gaussian distributions. By leveraging the natural-gradient variational inference (NGVI), the architecture distribution can be easily optimized based on existing codebases without incurring more memory and computational consumption. We demonstrate how the differentiable NAS benefits from Bayesian principles, enhancing exploration and improving stability. The experimental results on NAS benchmark datasets confirm the significant improvements the proposed framework can make. In addition, instead of simply applying the argmax on the learned parameters, we further leverage the recently-proposed training-free proxies in NAS to select the optimal architecture from a group architectures drawn from the optimized distribution, where we achieve state-of-the-art results on the NAS-Bench-201 and NAS-Bench-1shot1 benchmarks. Our best architecture in the DARTS search space also obtains competitive test errors with 2.37%, 15.72%, and 24.2% on CIFAR-10, CIFAR-100, and ImageNet, respectively.

----

## [1149] Arch-Graph: Acyclic Architecture Relation Predictor for Task-Transferable Neural Architecture Search

**Authors**: *Minbin Huang, Zhijian Huang, Changlin Li, Xin Chen, Hang Xu, Zhenguo Li, Xiaodan Liang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01158](https://doi.org/10.1109/CVPR52688.2022.01158)

**Abstract**:

Neural Architecture Search (NAS) aims to find efficient models for multiple tasks. Beyond seeking solutions for a single task, there are surging interests in transferring network design knowledge across multiple tasks. In this line of research, effectively modeling task correlations is vital yet highly neglected. Therefore, we propose Arch-Graph, a transferable NAS method that predicts task-specific optimal architectures with respect to given task embeddings. It leverages correlations across multiple tasks by using their embeddings as a part of the predictor's input for fast adaptation. We also formulate NAS as an architecture relation graph prediction problem, with the relational graph constructed by treating candidate architectures as nodes and their pairwise relations as edges. To enforce some basic properties such as acyclicity in the relational graph, we add additional constraints to the optimization process, converting NAS into the problem of finding a Maximal Weighted Acyclic Subgraph (MWAS). Our algorithm then strives to eliminate cycles and only establish edges in the graph if the rank results can be trusted. Through MWAS, Arch-Graph can effectively rank candidate models for each task with only a small budget to finetune the predictor. With extensive experiments on TransNAS-Bench-101, we show Arch-Graph's transferability and high sample efficiency across numerous tasks, beating many NAS methods designed for both single-task and multi-task search. It is able to find top 0.16% and 0.29% architectures on average on two search spaces under the budget of only 50 models.
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
Code: https://github.com/Centaurus982034/Arch-Graph

----

## [1150] Shapley-NAS: Discovering Operation Contribution for Neural Architecture Search

**Authors**: *Han Xiao, Ziwei Wang, Zheng Zhu, Jie Zhou, Jiwen Lu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01159](https://doi.org/10.1109/CVPR52688.2022.01159)

**Abstract**:

In this paper, we propose a Shapley value based method to evaluate operation contribution (Shapley-NAS) for neural architecture search. Differentiable architecture search (DARTS) acquires the optimal architectures by optimizing the architecture parameters with gradient descent, which significantly reduces the search cost. However, the magnitude of architecture parameters updated by gradient descent fails to reveal the actual operation importance to the task performance and therefore harms the effectiveness of obtained architectures. By contrast, we propose to evaluate the direct influence of operations on validation accuracy. To deal with the complex relationships between supernet components, we leverage Shapley value to quantify their marginal contributions by considering all possible combinations. Specifically, we iteratively optimize the supernet weights and update the architecture parameters by evaluating operation contributions via Shapley value, so that the optimal architectures are derived by selecting the operations that contribute significantly to the tasks. Since the exact computation of Shapley value is NP-hard, the Monte-Carlo sampling based algorithm with early truncation is employed for efficient approximation, and the momentum update mechanism is adopted to alleviate fluctuation of the sampling process. Extensive experiments on various datasets and various search spaces show that our Shapley-NAS outperforms the state-of-the-art methods by a considerable margin with light search cost. The code is available at https://github.com/Euphoria16/Shapley-NAS.git.

----

## [1151] GreedyNASv2: Greedier Search with a Greedy Path Filter

**Authors**: *Tao Huang, Shan You, Fei Wang, Chen Qian, Changshui Zhang, Xiaogang Wang, Chang Xu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01160](https://doi.org/10.1109/CVPR52688.2022.01160)

**Abstract**:

Training a good supernet in one-shot NAS methods is difficult since the search space is usually considerably huge 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$(\mathrm{e}.\mathrm{g}.,\ 13^{21})$</tex>
. In order to enhance the supernet's evaluation ability, one greedy strategy is to sample good paths, and let the supernet lean towards the good ones and ease its evaluation burden as a result. However, in practice the search can be still quite inefficient since the identification of good paths is not accurate enough and sampled paths still scatter around the whole search space. In this paper, we leverage an explicit path filter to capture the characteristics of paths and directly filter those weak ones, so that the search can be thus implemented on the shrunk space more greedily and efficiently. Concretely, based on the fact that good paths are much less than the weak ones in the space, we argue that the label of “weak paths” will be more confident and reliable than that of “good paths” in multi-path sampling. In this way, we thus cast the training of path filter in the positive and unlabeled (PU) learning paradigm, and also encourage a path embedding as better path/operation representation to enhance the identification capacity of the learned filter. By dint of this embedding, we can further shrink the search space by aggregating similar operations with similar embeddings, and the search can be more efficient and accurate. Extensive experiments validate the effectiveness of the proposed method GreedyNASv2. For example, our obtained GreedyNASv2-L achieves 81.1% Top-1 accuracy on ImageNet dataset, significantly outperforming the ResNet-50 strong baselines.

----

## [1152] Neural Architecture Search with Representation Mutual Information

**Authors**: *Xiawu Zheng, Xiang Fei, Lei Zhang, Chenglin Wu, Fei Chao, Jianzhuang Liu, Wei Zeng, Yonghong Tian, Rongrong Ji*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01161](https://doi.org/10.1109/CVPR52688.2022.01161)

**Abstract**:

Performance evaluation strategy is one of the most important factors that determine the effectiveness and efficiency in Neural Architecture Search (NAS). Existing strategies, such as employing standard training or performance predictor, often suffer from high computational complexity and low generality. To address this issue, we propose to rank architectures by Representation Mutual Information (RMI). Specifically, given an arbitrary architecture that has decent accuracy, architectures that have high RMI with it always yield good accuracies. As an accurate performance indicator to facilitate NAS, RMI not only generalizes well to different search spaces, but is also efficient enough to evaluate architectures using only one batch of data. Building upon RMI, we further propose a new search algorithm termed RMI-NAS, facilitating with a theorem to guarantee the global optimal of the searched architecture. In particular, RMI-NAS first randomly samples architectures from the search space, which are then effectively classified as positive or negative samples by RMI. We then use these samples to train a random forest to explore new regions, while keeping track of the distribution of positive architectures. When the sample size is sufficient, the architecture with the largest probability from the aforementioned distribution is selected, which is theoretically proved to be the optimal solution. The architectures searched by our method achieve remarkable top-1 accuracies with the magnitude times faster search process. Besides, RMI-NAS also generalizes to different datasets and search spaces. Our code has been made available at https://git.openi.org.cn/PCL_AutoML/XNAS.

----

## [1153] Performance-Aware Mutual Knowledge Distillation for Improving Neural Architecture Search

**Authors**: *Pengtao Xie, Xuefeng Du*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01162](https://doi.org/10.1109/CVPR52688.2022.01162)

**Abstract**:

Knowledge distillation has shown great effectiveness for improving neural architecture search (NAS). Mutual knowledge distillation (MKD), where a group of models mutually generate knowledge to train each other, has achieved promising results in many applications. In existing MKD methods, mutual knowledge distillation is performed between models without scrutiny: a worse-performing model is allowed to generate knowledge to train a better-performing model, which may lead to collective failures. To address this problem, we propose a performance-aware MKD (PAMKD) approach for NAS, where knowledge generated by model 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$A$</tex>
 is allowed to train model 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$B$</tex>
 only if the performance of 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$A$</tex>
 is better than B. We propose a three-level optimization framework to formulate PAMKD, where three learning stages are performed end-to-end: 1) each model trains an initial model independently; 2) the initial models are evaluated on a validation set and better-performing models generate knowledge to train worse-performing models; 3) architectures are updated by minimizing a validation loss. Experimental results on a variety of datasets demonstrate that our method is effective.

----

## [1154] Knowledge Distillation with the Reused Teacher Classifier

**Authors**: *Defang Chen, Jian-Ping Mei, Hailin Zhang, Can Wang, Yan Feng, Chun Chen*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01163](https://doi.org/10.1109/CVPR52688.2022.01163)

**Abstract**:

Knowledge distillation aims to compress a powerful yet cumbersome teacher model into a lightweight student model without much sacrifice of performance. For this purpose, various approaches have been proposed over the past few years, generally with elaborately designed knowledge rep-resentations, which in turn increase the difficulty of model development and interpretation. In contrast, we empirically show that a simple knowledge distillation technique is enough to significantly narrow down the teacher-student performance gap. We directly reuse the discriminative classifier from the pre-trained teacher model for student inference and train a student encoder through feature alignment with a single ℓ
<inf xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">2</inf>
 loss. In this way, the student model is able to achieve exactly the same performance as the teacher model provided that their extracted features are perfectly aligned. An additional projector is developed to help the student encoder match with the teacher classifier, which renders our technique applicable to various teacher and student architectures. Extensive experiments demonstrate that our technique achieves state-of-the-art results at the modest cost of compression ratio due to the added projector.

----

## [1155] Self-Distillation from the Last Mini-Batch for Consistency Regularization

**Authors**: *Yiqing Shen, Liwu Xu, Yuzhe Yang, Yaqian Li, Yandong Guo*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01164](https://doi.org/10.1109/CVPR52688.2022.01164)

**Abstract**:

Knowledge distillation (KD) shows a bright promise as a powerful regularization strategy to boost generalization ability by leveraging learned sample-level soft targets. Yet, employing a complex pre-trained teacher network or an ensemble of peer students in existing KD is both timeconsuming and computationally costly. Various self KD methods have been proposed to achieve higher distillation efficiency. However, they either require extra network architecture modification or are difficult to parallelize. To cope with these challenges, we propose an efficient and reliable self-distillation framework, named Self-Distillation from Last Mini-Batch (DLB). Specifically, we rearrange the sequential sampling by constraining half of each mini-batch coinciding with the previous iteration. Meanwhile, the rest half will coincide with the upcoming iteration. Afterwards, the former half mini-batch distills on-the-fly soft targets generated in the previous iteration. Our proposed mechanism guides the training stability and consistency, resulting in robustness to label noise. Moreover, our method is easy to implement, without taking up extra run-time memory or requiring model structure modification. Experimental results on three classification benchmarks illustrate that our approach can consistently outperform state-of-the-art self-distillation approaches with different network architectures. Additionally, our method shows strong compatibility with augmentation strategies by gaining additional performance improvement. The code is available at https://github.com/Meta-knowledge-Lab/DLB.

----

## [1156] Decoupled Knowledge Distillation

**Authors**: *Borui Zhao, Quan Cui, Renjie Song, Yiyu Qiu, Jiajun Liang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01165](https://doi.org/10.1109/CVPR52688.2022.01165)

**Abstract**:

State-of-the-art distillation methods are mainly based on distilling deep features from intermediate layers, while the significance of logit distillation is greatly overlooked. To provide a novel viewpoint to study logit distillation, we re-formulate the classical KD loss into two parts, i.e., target class knowledge distillation (TCKD) and non-target class knowledge distillation (NCKD). We empirically investigate and prove the effects of the two parts: TCKD transfers knowledge concerning the “difficulty” of training samples, while NCKD is the prominent reason why logit distillation works. More importantly, we reveal that the classical KD loss is a coupled formulation, which (1) suppresses the effectiveness of NCKD and (2) limits the flexibility to balance these two parts. To address these issues, we present Decoupled Knowledge Distillation (DKD), enabling TCKD and NCKD to play their roles more efficiently and flexibly. Compared with complex feature-based methods, our DKD achieves comparable or even better results and has better training efficiency on CIFAR-100, ImageNet, and MS-COCO datasets for image classification and object detection tasks. This paper proves the great potential of logit distillation, and we hope it will be helpful for future research. The code is available at https://github.com/megviiresearch/mdistiller.

----

## [1157] Scaling Up Your Kernels to 31×31: Revisiting Large Kernel Design in CNNs

**Authors**: *Xiaohan Ding, Xiangyu Zhang, Jungong Han, Guiguang Ding*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01166](https://doi.org/10.1109/CVPR52688.2022.01166)

**Abstract**:

We revisit large kernel design in modern convolutional neural networks (CNNs). Inspired by recent advances in vision transformers (ViTs), in this paper, we demonstrate that using a few large convolutional kernels instead of a stack of small kernels could be a more powerful paradigm. We suggested five guidelines, e.g., applying re-parameterized large depthwise convolutions, to design efficient high-performance large-kernel CNNs. Following the guidelines, we propose RepLKNet, a pure CNN architecture whose kernel size is as large as 31×31, in contrast to commonly used 3×3. RepLKNet greatly closes the performance gap between CNNs and ViTs, e.g., achieving comparable or superior results than Swin Transformer on ImageNet and a few typical downstream tasks, with lower latency. RepLKNet also shows nice scalability to big data and large models, obtaining 87.8% top-1 accuracy on ImageNet and 56.0% mIoU on ADE20K, which is very competitive among the state-of-the-arts with similar model sizes. Our study further reveals that, in contrast to small-kernel CNNs, large-kernel CNNs have much larger effective receptive fields and higher shape bias rather than texture bias. Code & models at https://github.com/megvii-research/RepLKNet.

----

## [1158] A ConvNet for the 2020s

**Authors**: *Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer, Trevor Darrell, Saining Xie*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01167](https://doi.org/10.1109/CVPR52688.2022.01167)

**Abstract**:

The “Roaring 20s” of visual recognition began with the introduction of Vision Transformers (ViTs), which quickly superseded ConvNets as the state-of-the-art image classification model. A vanilla ViT, on the other hand, faces difficulties when applied to general computer vision tasks such as object detection and semantic segmentation. It is the hierarchical Transformers (e.g., Swin Transformers) that reintroduced several ConvNet priors, making Transformers practically viable as a generic vision backbone and demonstrating remarkable performance on a wide variety of vision tasks. However, the effectiveness of such hybrid approaches is still largely credited to the intrinsic superiority of Transformers, rather than the inherent inductive biases of convolutions. In this work, we reexamine the design spaces and test the limits of what a pure ConvNet can achieve. We gradually “modernize” a standard ResNet toward the design of a vision Transformer, and discover several key components that contribute to the performance difference along the way. The outcome of this exploration is a family of pure ConvNet models dubbed ConvNeXt. Constructed entirely from standard ConvNet modules, ConvNeXts compete favorably with Transformers in terms of accuracy and scalability, achieving 87.8% ImageNet top-1 accuracy and outperforming Swin Transformers on COCO detection and ADE20K segmentation, while maintaining the simplicity and efficiency of standard ConvNets.

----

## [1159] Beyond Fixation: Dynamic Window Visual Transformer

**Authors**: *Pengzhen Ren, Changlin Li, Guangrun Wang, Yun Xiao, Qing Du, Xiaodan Liang, Xiaojun Chang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01168](https://doi.org/10.1109/CVPR52688.2022.01168)

**Abstract**:

Recently, a surge of interest in visual transformers is to reduce the computational cost by limiting the calculation of self-attention to a local window. Most current work uses a fixed single-scale window for modeling by default, ignoring the impact of window size on model performance. How-ever, this may limit the modeling potential of these window-based models for multi-scale information. In this paper, we propose a novel method, named Dynamic Window Vision Transformer (DW-ViT). The dynamic window strategy proposed by DW- ViT goes beyond the model that employs a fixed single window setting. To the best of our knowl-edge, we are the first to use dynamic multi-scale windows to explore the upper limit of the effect of window settings on model performance. In DW- ViT, multi-scale information is obtained by assigning windows of different sizes to different head groups of window multi-head self-attention. Then, the information is dynamically fused by assigning different weights to the multi-scale window branches. We con-ducted a detailed performance evaluation on three datasets, ImageNet-1K, ADE20K, and COCO. Compared with re-lated state-of-the-art (SoTA) methods, DW- ViT obtains the best performance. Specifically, compared with the current SoTA Swin Transformers [31], DW-ViT has achieved con-sistent and substantial improvements on all three datasets with similar parameters and computational costs. In addition, DW-ViT exhibits good scalability and can be easily inserted into any window-based visual transformers.
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
Code release: https://github.com/pzhren/DW-ViT. This work was done when the first author interned at Dark Matter AI..

----

## [1160] Lite Vision Transformer with Enhanced Self-Attention

**Authors**: *Chenglin Yang, Yilin Wang, Jianming Zhang, He Zhang, Zijun Wei, Zhe Lin, Alan L. Yuille*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01169](https://doi.org/10.1109/CVPR52688.2022.01169)

**Abstract**:

Despite the impressive representation capacity of vision transformer models, current light-weight vision transformer models still suffer from inconsistent and incorrect dense predictions at local regions. We suspect that the power of their self-attention mechanism is limited in shallower and thinner networks. We propose Lite Vision Transformer (LVT), a novel light-weight transformer network with two enhanced self-attention mechanisms to improve the model performances for mobile deployment. For the low-level features, we introduce Convolutional Self-Attention (CSA). Unlike previous approaches of merging convolution and self-attention, CSA introduces local self-attention into the convolution within a kernel of size 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$3\times 3$</tex>
 to enrich low-level features in the first stage of LVT. For the high-level features, we propose Recursive Atrous Self-Attention (RASA), which utilizes the multi-scale context when calculating the similarity map and a recursive mechanism to increase the representation capability with marginal extra parameter cost. The superiority of LVT is demonstrated on ImageNet recognition, ADE20K semantic segmentation, and COCO panoptic segmentation. The code is made publicly available
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
https://github.com/Chenglin-Yang/LVT.

----

## [1161] Swin Transformer V2: Scaling Up Capacity and Resolution

**Authors**: *Ze Liu, Han Hu, Yutong Lin, Zhuliang Yao, Zhenda Xie, Yixuan Wei, Jia Ning, Yue Cao, Zheng Zhang, Li Dong, Furu Wei, Baining Guo*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01170](https://doi.org/10.1109/CVPR52688.2022.01170)

**Abstract**:

We present techniques for scaling Swin Transformer [35] up to 3 billion parameters and making it capable of training with images of up to 1,536x1,536 resolution. By scaling up capacity and resolution, Swin Transformer sets new records on four representative vision benchmarks: 84.0% top-1 accuracy on ImageNet- V2 image classification, 63.1 / 54.4 box / mask mAP on COCO object detection, 59.9 mIoU on ADE20K semantic segmentation, and 86.8% top-1 accuracy on Kinetics-400 video action classification. We tackle issues of training instability, and study how to effectively transfer models pre-trained at low resolutions to higher resolution ones. To this aim, several novel technologies are proposed: 1) a residual post normalization technique and a scaled cosine attention approach to improve the stability of large vision models; 2) a log-spaced continuous position bias technique to effectively transfer models pre-trained at low-resolution images and windows to their higher-resolution counterparts. In addition, we share our crucial implementation details that lead to significant savings of GPU memory consumption and thus make it feasi-ble to train large vision models with regular GPUs. Using these techniques and self-supervised pre-training, we suc-cessfully train a strong 3 billion Swin Transformer model and effectively transfer it to various vision tasks involving high-resolution images or windows, achieving the state-of-the-art accuracy on a variety of benchmarks. Code is avail-able at https://github.com/microsoft/Swin-Transformer.

----

## [1162] The Principle of Diversity: Training Stronger Vision Transformers Calls for Reducing All Levels of Redundancy

**Authors**: *Tianlong Chen, Zhenyu Zhang, Yu Cheng, Ahmed Awadallah, Zhangyang Wang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01171](https://doi.org/10.1109/CVPR52688.2022.01171)

**Abstract**:

Vision transformers (ViTs) have gained increasing popularity as they are commonly believed to own higher mod-eling capacity and representation flexibility, than traditional convolutional networks. However, it is questionable whether such potential has been fully unleashed in prac-tice, as the learned ViTs often suffer from over-smoothening, yielding likely redundant models. Recent works made pre-liminary attempts to identify and alleviate such redundancy, e.g., via regularizing embedding similarity or re-injecting convolution-like structures. However, a “head-to-toe as-sessment” regarding the extent of redundancy in ViTs, and how much we could gain by thoroughly mitigating such, has been absent for this field. This paper, for the first time, systematically studies the ubiquitous existence of re-dundancy at all three levels: patch embedding, attention map, and weight space. In view of them, we advocate a principle of diversity for training ViTs, by presenting cor-responding regularizers that encourage the representation diversity and coverage at each of those levels, that enabling capturing more discriminative information. Extensive ex-periments on ImageNet with a number of ViT backbones validate the effectiveness of our proposals, largely eliminating the observed ViT redundancy and significantly boosting the model generalization. For example, our diversified DeiT obtains 0.70% ~ 1.76% accuracy boosts on ImageNet with highly reduced similarity. Our codes are fully available in https://github.com/VITA-Group/Diverse-ViT.

----

## [1163] MuIT: An End-to-End Multitask Learning Transformer

**Authors**: *Deblina Bhattacharjee, Tong Zhang, Sabine Süsstrunk, Mathieu Salzmann*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01172](https://doi.org/10.1109/CVPR52688.2022.01172)

**Abstract**:

We propose an end-to-end Multitask Learning Transformer framework, named MulT, to simultaneously learn multiple high-level vision tasks, including depth estimation, semantic segmentation, reshading, surface normal estimation, 2D keypoint detection, and edge detection. Based on the Swin transformer model, our framework encodes the input image into a shared representation and makes predictions for each vision task using task-specific transformer-based decoder heads. At the heart of our approach is a shared attention mechanism modeling the dependencies across the tasks. We evaluate our model on several multitask benchmarks, showing that our MulT framework outperforms both the state-of-the art multitask convolutional neural network models and all the respective single task transformer models. Our experiments further highlight the benefits of sharing attention across all the tasks, and demonstrate that our MulT model is robust and generalizes well to new domains. Our project website is at https://ivrl.github.io/MulT/.

----

## [1164] Towards Robust Vision Transformer

**Authors**: *Xiaofeng Mao, Gege Qi, Yuefeng Chen, Xiaodan Li, Ranjie Duan, Shaokai Ye, Yuan He, Hui Xue*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01173](https://doi.org/10.1109/CVPR52688.2022.01173)

**Abstract**:

Recent advances on Vision Transformer (ViT) and its improved variants have shown that self-attention-based networks surpass traditional Convolutional Neural Networks (CNNs) in most vision tasks. However, existing ViTs focus on the standard accuracy and computation cost, lacking the investigation of the intrinsic influence on model robustness and generalization. In this work, we conduct systematic evaluation on components of ViTs in terms of their impact on robustness to adversarial examples, common corruptions and distribution shifts. We find some components can be harmful to robustness. By leveraging robust components as building blocks of ViTs, we propose Robust Vision Transformer (RVT), which is a new vision transformer and has superior performance with strong robustness. Inspired by the findings during the evaluation, we further propose two new plug-and-play techniques called position-aware attention scaling and patch-wise augmentation to augment our RVT, which we abbreviate as RVT*. The experimental results of RVT on ImageNet and six robustness benchmarks demonstrate its advanced robustness and generalization ability compared with previous ViTs and state-of-the-art CNNs. Furthermore, RVT-S* achieves Top-1 rank on multiple robustness leaderboards including ImageNet-C, ImageNet-Sketch and ImageNet-R.

----

## [1165] DearKD: Data-Efficient Early Knowledge Distillation for Vision Transformers

**Authors**: *Xianing Chen, Qiong Cao, Yujie Zhong, Jing Zhang, Shenghua Gao, Dacheng Tao*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01174](https://doi.org/10.1109/CVPR52688.2022.01174)

**Abstract**:

Transformers are successfully applied to computer vision due to their powerful modeling capacity with self-attention. However, the excellent performance of transformers heavily depends on enormous training images. Thus, a data-efficient transformer solution is urgently needed. In this work, we propose an early knowledge distillation framework, which is termed as DearKD, to improve the data efficiency required by transformers. Our DearKD is a two-stage framework that first distills the inductive biases from the early intermediate layers of a CNN and then gives the transformer full play by training without distillation. Further, our DearKD can be readily applied to the extreme data-free case where no real images are available. In this case, we propose a boundary-preserving intra-divergence loss based on DeepInversion to further close the performance gap against the full-data counterpart. Extensive experiments on ImageNet, partial ImageNet, data-free setting and other downstream tasks prove the superiority of DearKD over its baselines and state-of-the-art methods.

----

## [1166] MSG-Transformer: Exchanging Local Spatial Information by Manipulating Messenger Tokens

**Authors**: *Jiemin Fang, Lingxi Xie, Xinggang Wang, Xiaopeng Zhang, Wenyu Liu, Qi Tian*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01175](https://doi.org/10.1109/CVPR52688.2022.01175)

**Abstract**:

Transformers have offered a new methodology of designing neural networks for visual recognition. Compared to convolutional networks, Transformers enjoy the ability of referring to global features at each stage, yet the attention module brings higher computational overhead that obstructs the application of Transformers to process highresolution visual data. This paper aims to alleviate the conflict between efficiency and flexibility, for which we propose a specialized token for each region that serves as a messenger (MSG). Hence, by manipulating these MSG tokens, one can flexibly exchange visual information across regions and the computational complexity is reduced. We then integrate the MSG token into a multi-scale architecture named MSG-Transformer. In standard image classification and object detection, MSG-Transformer achieves competitive performance and the inference on both GPU and CPU is accelerated. Code is available at https://github.com/hustvl/MSG-Transformer.

----

## [1167] NomMer: Nominate Synergistic Context in Vision Transformer for Visual Recognition

**Authors**: *Hao Liu, Xinghua Jiang, Xin Li, Zhimin Bao, Deqiang Jiang, Bo Ren*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01176](https://doi.org/10.1109/CVPR52688.2022.01176)

**Abstract**:

Recently, Vision Transformers (ViT), with the self-attention (SA) as the de facto ingredients, have demon-strated great potential in the computer vision community. For the sake of trade-off between efficiency and performance, a group of works merely perform SA operation within local patches, whereas the global contextual information is abandoned, which would be indispensable for visual recognition tasks. To solve the issue, the subsequent global-local ViTs take a stab at marrying local SA with global one in parallel or alternative way in the model. Nevertheless, the exhaustively combined local and global context may exist redundancy for various visual data, and the receptive field within each layer is fixed. Alternatively, a more graceful way is that global and local context can adaptively contribute per se to accommodate different visual data. To achieve this goal, we in this paper propose a novel ViT architecture, termed NomMer, which can dynamically Nominate the synergistic global-local context in vision transforMer. By investigating the working pattern of NomMer, we further explore what context information is focused. Beneficial from this “dynamic nomination” mechanism, without bells and whistles, the NomMer can not only achieve 84.5% Top-1 classification accuracy on ImageNet with only 73M parameters, but also show promising performance on dense prediction tasks, i.e., object detection and semantic segmentation. The code and models are publicly available at https://github.com/TencentYoutuResearch/VisualRecognition-NomMer.

----

## [1168] TopFormer: Token Pyramid Transformer for Mobile Semantic Segmentation

**Authors**: *Wenqiang Zhang, Zilong Huang, Guozhong Luo, Tao Chen, Xinggang Wang, Wenyu Liu, Gang Yu, Chunhua Shen*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01177](https://doi.org/10.1109/CVPR52688.2022.01177)

**Abstract**:

Although vision transformers (ViTs) have achieved great success in computer vision, the heavy computational cost hampers their applications to dense prediction tasks such as semantic segmentation on mobile devices. In this paper, we present a mobile-friendly architecture named Token Pyramid Vision Transformer (TopFormer). The proposed TopFormer takes Tokens from various scales as input to produce scale-aware semantic features, which are then in-Jected into the corresponding tokens to augment the representation. Experimental results demonstrate that our method significantly outperforms CNN- and ViT-based networks across several semantic segmentation datasets and achieves a good trade-off between accuracy and latency. On the ADE20K dataset, TopFormer achieves 5% higher accuracy in mIoU than MobileNetV3 with lower latency on an ARM-based mobile device. Furthermore, the tiny version of TopFormer achieves real-time inference on an ARM-based mobile device with competitive results. The code and models are available at: https://github.com/hustvl/TopFormer.

----

## [1169] Multi-Scale High-Resolution Vision Transformer for Semantic Segmentation

**Authors**: *Jiaqi Gu, Hyoukjun Kwon, Dilin Wang, Wei Ye, Meng Li, Yu-Hsin Chen, Liangzhen Lai, Vikas Chandra, David Z. Pan*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01178](https://doi.org/10.1109/CVPR52688.2022.01178)

**Abstract**:

Vision Transformers (ViTs) have emerged with superior performance on computer vision tasks compared to the convolutional neural network (CNN)-based models. However, ViTs mainly designed for image classification will generate single-scale low-resolution representations, which makes dense prediction tasks such as semantic segmentation challenging for ViTs. Therefore, we propose HRViT, which enhances ViTs to learn semantically-rich and spatially-precise multi-scale representations by integrating high-resolution multi-branch architectures with ViTs. We balance the model performance and efficiency of HRViT by various branch-block co-optimization techniques. Specifically, we explore heterogeneous branch designs, reduce the redundancy in linear layers, and augment the attention block with enhanced expressiveness. Those approaches enabled HRViT to push the Pareto frontier of performance and efficiency on semantic segmentation to a new level, as our evaluation results on ADE20K and Cityscapes show. HRViT achieves 50.20% mIoU on ADE20K and 83.16% mIoU on Cityscapes, surpassing state-of-the-art MiT and CSWin backbones with an average of +1.78 mIoU improvement, 28% parameter saving, and 21% FLOPs reduction, demonstrating the potential of HRViT as a strong vision backbone for semantic segmentation. Our code is publicly available 
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
https://github.com/facebookresearch/HRViT.

----

## [1170] Bridged Transformer for Vision and Point Cloud 3D Object Detection

**Authors**: *Yikai Wang, TengQi Ye, Lele Cao, Wenbing Huang, Fuchun Sun, Fengxiang He, Dacheng Tao*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01180](https://doi.org/10.1109/CVPR52688.2022.01180)

**Abstract**:

3D object detection is a crucial research topic in computer vision, which usually uses 3D point clouds as input in conventional setups. Recently, there is a trend of leveraging multiple sources of input data, such as complementing the 3D point cloud with 2D images that often have richer color and fewer noises. However, due to the heterogeneous geometrics of the 2D and 3D representations, it prevents us from applying off-the-shelf neural networks to achieve multimodal fusion. To that end, we propose Bridged Transformer (BrT), an end-to-end architecture for 3D object detection. BrT is simple and effective, which learns to identify 3D and 2D object bounding boxes from both points and image patches. A key element of BrT lies in the utilization of object queries for bridging 3D and 2D spaces, which unifies different sources of data representations in Transformer. We adopt a form of feature aggregation realized by point-to-patch projections which further strengthen the interaction between images and points. Moreover, BrT works seamlessly for fusing the point cloud with multi-view images. We experimentally show that BrT surpasses state-of-the-art methods on SUN RGB-D and ScanNetV2 datasets.

----

## [1171] CSWin Transformer: A General Vision Transformer Backbone with Cross-Shaped Windows

**Authors**: *Xiaoyi Dong, Jianmin Bao, Dongdong Chen, Weiming Zhang, Nenghai Yu, Lu Yuan, Dong Chen, Baining Guo*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01181](https://doi.org/10.1109/CVPR52688.2022.01181)

**Abstract**:

We present CSWin Transformer, an efficient and effective Transformer-based backbone for general-purpose vision tasks. A challenging issue in Transformer design is that global self-attention is very expensive to compute whereas local self-attention often limits the field of interactions of each token. To address this issue, we develop the Cross-Shaped Window self-attention mechanism for computing self-attention in the horizontal and vertical stripes in parallel that form a cross-shaped window, with each stripe obtained by splitting the input feature into stripes of equal width. We provide a mathematical analysis of the effect of the stripe width and vary the stripe width for different layers of the Transformer network which achieves strong modeling capability while limiting the computation cost. We also introduce Locally-enhanced Positional Encoding (LePE), which handles the local positional information better than existing encoding schemes. LePE naturally supports arbitrary input resolutions, and is thus especially effective and friendly for downstream tasks. Incorporated with these designs and a hierarchical structure, CSWin Transformer demonstrates competitive performance on common vision tasks. Specifically, it achieves 85.4% Top-1 accuracy on ImageNet-1K without any extra training data or label, 53.9 box AP and 46.4 mask AP on the COCO detection task, and 52.2 mIOU on the ADE20K semantic segmentation task, surpassing previous state-of-the-art Swin Transformer backbone by +1.2, +2.0, +1.4, and +2.0 respectively under the similar FLOPs setting. By further pretraining on the larger dataset ImageNet-21K, we achieve 87.5% Top-1 accuracy on ImageNet-1K and high segmentation performance on ADE20K with 55.7 mIoU. 
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
Code and pretrain model is available at https://github.com/microsoft/CSWin-Transformer

----

## [1172] TransMix: Attend to Mix for Vision Transformers

**Authors**: *Jieneng Chen, Shuyang Sun, Ju He, Philip H. S. Torr, Alan L. Yuille, Song Bai*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01182](https://doi.org/10.1109/CVPR52688.2022.01182)

**Abstract**:

Mixup-based augmentation has been found to be effective for generalizing models during training, especially for Vision Transformers (ViTs) since they can easily overfit. However, previous mixup-based methods have an underlying prior knowledge that the linearly interpolated ratio of targets should be kept the same as the ratio proposed in input interpolation. This may lead to a strange phenomenon that sometimes there is no valid object in the mixed image due to the random process in augmentation but there is still response in the label space. To bridge such gap between the input and label spaces, we propose TransMix, which mixes labels based on the attention maps of Vision Transformers. The confidence of the label will be larger if the corresponding input image is weighted higher by the attention map. TransMix is embarrassingly simple and can be implemented in just a few lines of code without introducing any extra parameters and FLOPs to ViT-based models. Experimental results show that our method can consistently improve various ViT-based models at scales on ImageNet classification. After pre-trained with TransMix on ImageNet, the ViT-based models also demonstrate better transferability to semantic segmentation, object detection and instance segmentation. TransMix also exhibits to be more robust when evaluating on 4 different benchmarks. Code is publicly available at https://github.com/Beckschen/TransMix.

----

## [1173] MiniViT: Compressing Vision Transformers with Weight Multiplexing

**Authors**: *Jinnian Zhang, Houwen Peng, Kan Wu, Mengchen Liu, Bin Xiao, Jianlong Fu, Lu Yuan*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01183](https://doi.org/10.1109/CVPR52688.2022.01183)

**Abstract**:

Vision Transformer (ViT) models have recently drawn much attention in computer vision due to their high model capability. However, ViT models suffer from huge number of parameters, restricting their applicability on devices with limited memory. To alleviate this problem, we propose MiniViT, a new compression framework, which achieves parameter reduction in vision transformers while retaining the same performance. The central idea of MiniViT is to multiplex the weights of consecutive transformer blocks. More specifically, we make the weights shared across layers, while imposing a transformation on the weights to increase diversity. Weight distillation over self-attention is also applied to transfer knowledge from large-scale ViT models to weight-multiplexed compact models. Comprehensive experiments demonstrate the efficacy of MiniViT, showing that it can reduce the size of the pre-trained Swin-B transformer by 48%, while achieving an increase of 1.0% in Top-1 accuracy on ImageNet. Moreover, using a single-layer of parameters, MiniViT is able to compress DeiT-B by 9.7 times from 86M to 9M parameters, without seriously compromising the performance. Finally, we verify the transferability of MiniViT by reporting its performance on downstream benchmarks. Code and models are available at here.

----

## [1174] Fine-tuning Image Transformers using Learnable Memory

**Authors**: *Mark Sandler, Andrey Zhmoginov, Max Vladymyrov, Andrew Jackson*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01184](https://doi.org/10.1109/CVPR52688.2022.01184)

**Abstract**:

In this paper we propose augmenting Vision Transformer models with learnable memory tokens. Our approach allows the model to adapt to new tasks, using few parameters, while optionally preserving its capabilities on previously learned tasks. At each layer we introduce a set of learnable embedding vectors that provide contextual information useful for specific datasets. We call these “memory tokens”. We show that augmenting a model with just a handful of such tokens per layer significantly improves accuracy when compared to conventional head-only fine-tuning, and performs only slightly below the significantly more expensive full fine-tuning. We then propose an attention-masking approach that enables extension to new downstream tasks, with a computation reuse. In this setup in addition to being parameters efficient, models can execute both old and new tasks as a part of single inference at a small incremental cost.

----

## [1175] Patch Slimming for Efficient Vision Transformers

**Authors**: *Yehui Tang, Kai Han, Yunhe Wang, Chang Xu, Jianyuan Guo, Chao Xu, Dacheng Tao*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01185](https://doi.org/10.1109/CVPR52688.2022.01185)

**Abstract**:

This paper studies the efficiency problem for visual transformers by excavating redundant calculation in given networks. The recent transformer architecture has demonstrated its effectiveness for achieving excellent performance on a series of computer vision tasks. However, similar to that of convolutional neural networks, the huge computational cost of vision transformers is still a severe issue. Considering that the attention mechanism aggregates different patches layer-by-layer, we present a novel patch slimming approach that discards useless patches in a topdown paradigm. We first identify the effective patches in the last layer and then use them to guide the patch selection process of previous layers. For each layer, the impact of a patch on the final output feature is approximated and patches with less impacts will be removed. Experimental results on benchmark datasets demonstrate that the proposed method can significantly reduce the computational costs of vision transformers without affecting their performances. For example, over 45% FLOPs of the ViT-Ti model can be reduced with only 0.2% top-1 accuracy drop on the ImageNet dataset.

----

## [1176] CMT: Convolutional Neural Networks Meet Vision Transformers

**Authors**: *Jianyuan Guo, Kai Han, Han Wu, Yehui Tang, Xinghao Chen, Yunhe Wang, Chang Xu*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01186](https://doi.org/10.1109/CVPR52688.2022.01186)

**Abstract**:

Vision transformers have been successfully applied to image recognition tasks due to their ability to capture long-range dependencies within an image. However, there are still gaps in both performance and computational cost between transformers and existing convolutional neural networks (CNNs). In this paper, we aim to address this issue and develop a network that can outperform not only the canonical transformers, but also the high-performance convolutional models. We propose a new transformer based hybrid network by taking advantage of transformers to capture long-range dependencies, and of CNNs to extract local information. Furthermore, we scale it to obtain a family of models, called CMTs, obtaining much better trade-off for accuracy and efficiency than previous CNN-based and transformer-based models. In particular, our CMT-S achieves 83.5% top-1 accuracy on ImageNet, while being 14x and 2x smaller on FLOPs than the existing DeiT and EfficientNet, respectively. The proposed CMT-S also generalizes well on CIFAR10 (99.2%), CIFAR100 (91.7%), Flowers (98.7%), and other challenging vision datasets such as COCO (44.3% mAP), with considerably less computational cost.

----

## [1177] Multimodal Token Fusion for Vision Transformers

**Authors**: *Yikai Wang, Xinghao Chen, Lele Cao, Wenbing Huang, Fuchun Sun, Yunhe Wang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01187](https://doi.org/10.1109/CVPR52688.2022.01187)

**Abstract**:

Many adaptations of transformers have emerged to address the single-modal vision tasks, where self-attention modules are stacked to handle input sources like images. Intuitively, feeding multiple modalities of data to vision transformers could improve the performance, yet the innermodal attentive weights may be diluted, which could thus greatly undermine the final performance. In this paper, we propose a multimodal token fusion method (TokenFusion), tailored for transformer-based vision tasks. To effectively fuse multiple modalities, TokenFusion dynamically detects uninformative tokens and substitute these tokens with projected and aggregated inter-modal features. Residual positional alignment is also adopted to enable explicit utilization of the inter-modal alignments after fusion. The design of TokenFusion allows the transformer to learn correlations among multimodal features, while the single-modal transformer architecture remains largely intact. Extensive experiments are conducted on a variety of homogeneous and heterogeneous modalities and demonstrate that TokenFusion surpasses state-of-the-art methods in three typical vision tasks: multimodal image-to-image translation, RGB-depth semantic segmentation, and 3D object detection with point cloud and images. Code will be released 
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
https://github.com/huawei-noah/noah-research 
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">2</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">2</sup>
https://gitee.com/mindspore/models/tree/master/research/cv/TokenFusion.

----

## [1178] CAFE: Learning to Condense Dataset by Aligning Features

**Authors**: *Kai Wang, Bo Zhao, Xiangyu Peng, Zheng Zhu, Shuo Yang, Shuo Wang, Guan Huang, Hakan Bilen, Xinchao Wang, Yang You*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01188](https://doi.org/10.1109/CVPR52688.2022.01188)

**Abstract**:

Dataset condensation aims at reducing the network training effort through condensing a cumbersome training set into a compact synthetic one. State-of-the-art approaches largely rely on learning the synthetic data by matching the gradients between the real and synthetic data batches. Despite the intuitive motivation and promising results, such gradient-based methods, by nature, easily overfit to a biased set of samples that produce dominant gradients, and thus lack a global supervision of data distribution. In this paper, we propose a novel scheme to Condense dataset by Aligning FEatures (CAFE), which explicitly attempts to preserve the real-feature distribution as well as the discriminant power of the resulting synthetic set, lending itself to strong generalization capability to various architectures. At the heart of our approach is an effective strategy to align features from the real and synthetic data across various scales, while accounting for the classification of real samples. Our scheme is further backed up by a novel dynamic bi-level optimization, which adaptively adjusts parameter updates to prevent over-/under-fitting. We validate the proposed CAFE across various datasets, and demonstrate that it generally outperforms the state of the art: on the SVHN dataset, for example, the performance gain is up to 11%. Extensive experiments and analysis verify the effectiveness and necessity of proposed designs.

----

## [1179] Lite-MDETR: A Lightweight Multi-Modal Detector

**Authors**: *Qian Lou, Yen-Chang Hsu, Burak Uzkent, Ting Hua, Yilin Shen, Hongxia Jin*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01189](https://doi.org/10.1109/CVPR52688.2022.01189)

**Abstract**:

Recent multi-modal detectors based on transformers and modality encoders have successfully achieved impressive results on end-to-end visual object detection conditioned on a raw text query. However, they require a large model size and an enormous amount of computations to achieve high performance, which makes it difficult to deploy mobile applications that are limited by tight hardware resources. In this paper, we present a Lightweight modulated detector, Lite-MDETR, to facilitate efficient end-to-end multi-modal understanding on mobile devices. The key primitive is that Dictionary-Lookup-Transformormations (DLT) is proposed to replace Linear Transformation (LT) in multi-modal detectors where each weight in Linear Transformation (LT) is approximately factorized into a smaller dictionary, index, and coefficient. This way, the enormous linear projection with weights is converted into efficient linear projection with dictionaries, a few lookups and scalings with indices and coefficients. DLT can be applied to any pretrained multi-modal detectors, removing the need to perform expensive training from scratch. To tackle the challenging training of DLT due to non-differentiable index, we convert the index and coefficient into a sparse matrix, train this sparse matrix during the fine-tuning phase, and recover it back to index and coefficient during the inference phase. Our experiments on phrase grounding, referring expression comprehension and segmentation, and VQA show that our Lite-MDETR achieves similar accuracy as the prior multi-modal detectors with up to ~ 4.1 × model size reduction.

----

## [1180] DeeCap: Dynamic Early Exiting for Efficient Image Captioning

**Authors**: *Zhengcong Fei, Xu Yan, Shuhui Wang, Qi Tian*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01190](https://doi.org/10.1109/CVPR52688.2022.01190)

**Abstract**:

Both accuracy and efficiency are crucial for image captioning in real-world scenarios. Although Transformer-based models have gained significant improved captioning performance, their computational cost is very high. A feasible way to reduce the time complexity is to exit the prediction early in internal decoding layers without passing the entire model. However, it is not straightforward to devise early exiting into image captioning due to the following issues. On one hand, the representation in shallow layers lacks high-level semantic and sufficient cross-modal fusion information for accurate prediction. On the other hand, the exiting decisions made by internal classifiers are unreliable sometimes. To solve these issues, we propose DeeCap framework for efficient image captioning, which dynamically selects proper-sized decoding layers from a global perspective to exit early. The key to successful early exiting lies in the specially designed imitation learning mechanism, which predicts the deep layer activation with shallow layer features. By deliberately merging the imitation learning into the whole image captioning architecture, the imitated deep layer representation can mitigate the loss brought by the missing of actual deep layers when early exiting is undertaken, resulting in significant reduction in calculation cost with small sacrifice of accuracy. Experiments on the MS COCO and Flickr30k datasets demonstrate the DeeCap can achieve competitive performances with 4× speed-up. Code is available at: https://github.com/feizc/DeeCap.

----

## [1181] Searching the Deployable Convolution Neural Networks for GPUs

**Authors**: *Linnan Wang, Chenhan Yu, Satish Salian, Slawomir Kierat, Szymon Migacz, Alex Fit-Florea*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01191](https://doi.org/10.1109/CVPR52688.2022.01191)

**Abstract**:

Customizing Convolution Neural Networks (CNN) for production use has been a challenging task for DL practitioners. This paper intends to expedite the model customization with a model hub that contains the optimized models tiered by their inference latency using Neural Architecture Search (NAS). To achieve this goal, we build a distributed NAS system to search on a novel search space that consists of prominent factors to impact latency and accuracy. Since we target GPU, we name the NAS optimized models as GPUNet, which establishes a new SOTA Pareto frontier in inference latency and accuracy. Within 1ms, GPUNet is 2x faster than EfficientNet-X and FBNetV3 with even better accuracy. We also validate GPUNet on detection tasks, and GPUNet consistently outperforms EfficientNet-X and FB-NetV3 on COCO detection tasks in both latency and accuracy. All of these data validate that our NAS system is effective and generic to handle different design tasks. With this NAS system, we expand GPUNet to cover a wide range of latency targets such that DL practitioners can deploy our models directly in different scenarios.

----

## [1182] Active Learning by Feature Mixing

**Authors**: *Amin Parvaneh, Ehsan Abbasnejad, Damien Teney, Reza Haffari, Anton van den Hengel, Javen Qinfeng Shi*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01192](https://doi.org/10.1109/CVPR52688.2022.01192)

**Abstract**:

The promise of active learning (AL) is to reduce labelling costs by selecting the most valuable examples to annotate from a pool of unlabelled data. Identifying these examples is especially challenging with high-dimensional data (e.g. images, videos) and in low-data regimes. In this paper, we propose a novel method for batch AL called ALFA-Mix. We identify unlabelled instances with sufficiently-distinct features by seeking inconsistencies in predictions resulting from interventions on their representations. We construct interpolations between representations of labelled and unlabelled instances then examine the predicted labels. We show that inconsistencies in these predictions help discovering features that the model is unable to recognise in the unlabelled instances. We derive an efficient implementation based on a closed-form solution to the optimal interpolation causing changes in predictions. Our method outperforms all recent AL approaches in 30 different settings on 12 benchmarks of images, videos, and non-visual data. The improvements are especially significant in low-data regimes and on self-trained vision transformers, where ALFA-Mix outperforms the state-of-the-art in 59% and 43% of the experiments respectively
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
The code is available at https://github.com/aminparvaneh/alpha_mix_active_learning.

----

## [1183] When to Prune? A Policy towards Early Structural Pruning

**Authors**: *Maying Shen, Pavlo Molchanov, Hongxu Yin, José M. Álvarez*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01193](https://doi.org/10.1109/CVPR52688.2022.01193)

**Abstract**:

Pruning enables appealing reductions in network memory footprint and time complexity. Conventional post-training pruning techniques lean towards efficient inference while overlooking the heavy computation for training. Recent exploration of pre-training pruning at initialization hints on training cost reduction via pruning, but suffers noticeable performance degradation. We attempt to combine the benefits of both directions and propose a policy that prunes as early as possible during training without hurting performance. Instead of pruning at initialization, our method exploits initial dense training for few epochs to quickly guide the architecture, while constantly evaluating dominant sub-networks via neuron importance ranking. This unveils dominant sub-networks whose structures turn stable, allowing conventional pruning to be pushed earlier into the training. To do this early, we further introduce an Early Pruning Indicator (EPI) that relies on sub-network architectural similarity and quickly triggers pruning when the sub-network's architecture stabilizes. Through extensive experiments on ImageNet, we show that EPI empowers a quick tracking of early training epochs suitable for pruning, offering same efficacy as an otherwise “oracle” grid-search that scans through epochs and requires orders of magnitude more compute. Our method yields 1.4% top-l accuracy boost over state-of-the-art pruning counterparts, cuts down training cost on GPU by 2.4x, hence offers a new efficiency-accuracy boundary for network pruning during training.

----

## [1184] Contrastive Dual Gating: Learning Sparse Features With Contrastive Learning

**Authors**: *Jian Meng, Li Yang, Jinwoo Shin, Deliang Fan, Jae-Sun Seo*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01194](https://doi.org/10.1109/CVPR52688.2022.01194)

**Abstract**:

Contrastive learning (or its variants) has recently become a promising direction in the self-supervised learning domain, achieving similar performance as supervised learning with minimum fine-tuning. Despite the labeling efficiency, wide and large networks are required to achieve high accuracy, which incurs a high amount of computation and hinders the pragmatic merit of self-supervised learning. To effectively reduce the computation of insignificant features or channels, recent dynamic pruning algorithms for supervised learning employed auxiliary salience predictors. However, we found that such salience predictors cannot be easily trained when they are naïvely applied to contrastive learning from scratch. To address this issue, we propose contrastive dual gating (CDG), a novel dynamic pruning algorithm that skips the uninformative features during contrastive learning without hurting the trainability of the networks. We demonstrate the superiority of CDG with ResNet models for CIFAR-10, CIFAR-100, and ImageNet-100 datasets. Compared to our implementations of state-of-the-art dynamic pruning algorithms for self-supervised learning, CDG achieves up to 15% accuracy improvement for CIFAR-10 dataset with higher computation reduction.

----

## [1185] How Well Do Sparse ImageNet Models Transfer?

**Authors**: *Eugenia Iofinova, Alexandra Peste, Mark Kurtz, Dan Alistarh*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01195](https://doi.org/10.1109/CVPR52688.2022.01195)

**Abstract**:

Transfer learning is a classic paradigm by which models pretrained on large “upstream” datasets are adapted to yield good results on “downstream” specialized datasets. Generally, more accurate models on the “upstream” dataset tend to provide better transfer accuracy “downstream”. In this work, we perform an in-depth investigation of this phenomenon in the context of convolutional neural networks (CNNs) trained on the ImageNet dataset, which have been pruned-that is, compressed by sparsifiying their connections. We consider transfer using unstructured pruned models obtained by applying several state-of-the-art pruning methods, including magnitude-based, second-order, regrowth, lottery-ticket, and regularization approaches, in the context of twelve standard transfer tasks. In a nutshell, our study shows that sparse models can match or even outperform the transfer performance of dense models, even at high sparsities, and, while doing so, can lead to significant inference and even training speedups. At the same time, we observe and analyze significant differences in the behaviour of different pruning methods. The code is available at: https://github.com/IST-DASLab/sparse-imagenet-transfer.

----

## [1186] RepNet: Efficient On-Device Learning via Feature Reprogramming

**Authors**: *Li Yang, Adnan Siraj Rakin, Deliang Fan*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01196](https://doi.org/10.1109/CVPR52688.2022.01196)

**Abstract**:

Transfer learning, where the goal is to transfer the well-trained deep learning models from a primary source task to a new task, is a crucial learning scheme for on-device machine learning, due to the fact that IoT/edge devices collect and then process massive data in our daily life. However, due to the tiny memory constraint in IoT/edge devices, such on-device learning requires ultra-small training memory footprint, bringing new challenges for memory-efficient learning. Many existing works solve this problem by reducing the number of trainable parameters. However, this doesn't directly translate to memory saving since the major bottleneck is the activations, not parameters. To develop memory-efficient on-device transfer learning, in this work, we are the first to approach the concept of transfer learning from a new perspective of intermediate feature re-programming of a pre-trained model (i.e., backbone). To perform this lightweight and memory-efficient reprogramming, we propose to train a tiny Reprogramming Network (Rep-Net) directly from the new task input data, while freezing the backbone model. The proposed Rep-Net model interchanges the features with the backbone model using an activation connector at regular intervals to mutually benefit both the backbone model and Rep-Net model features. Through extensive experiments, we validate each design specs of the proposed Rep-Net model in achieving highly memory-efficient on-device reprogramming. Our experiments establish the superior performance (i.e., low training memory and high accuracy) of Rep-Net compared to SOTA on-device transfer learning schemes across multiple benchmarks. Code is available at https://github.com/ASU-ESIC-FAN-Lab/RepNet.

----

## [1187] CHEX: CHannel EXploration for CNN Model Compression

**Authors**: *Zejiang Hou, Minghai Qin, Fei Sun, Xiaolong Ma, Kun Yuan, Yi Xu, Yen-Kuang Chen, Rong Jin, Yuan Xie, Sun-Yuan Kung*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01197](https://doi.org/10.1109/CVPR52688.2022.01197)

**Abstract**:

Channel pruning has been broadly recognized as an effective technique to reduce the computation and memory cost of deep convolutional neural networks. However, conventional pruning methods have limitations in that: they are restricted to pruning process only, and they require a fully pre-trained large model. Such limitations may lead to sub-optimal model quality as well as excessive memory and training cost. In this paper, we propose a novel Channel Exploration methodology, dubbed as CHEX, to rectify these problems. As opposed to pruning-only strategy, we propose to repeatedly prune and regrow the channels throughout the training process, which reduces the risk of pruning important channels prematurely. More exactly: From intra-Layer's aspect, we tackle the channel pruning problem via a well-known column subset selection (CSS) formulation. From inter-Layer's aspect, our regrowing stages open a path for dynamically re-allocating the number of channels across all the layers under a global channel sparsity constraint. In addition, all the exploration process is done in a single training from scratch without the need of a pre-trained large model. Experimental results demonstrate that CHEX can effectively reduce the FLOPs of diverse CNN architectures on a variety of computer vision tasks, including image classification, object detection, instance segmentation, and 3D vision. For example, our compressed ResNet-50 model on ImageNet dataset achieves 76% top-l accuracy with only 25% FLOPs of the original ResNet-50 model, outperforming previous state-of-the-art channel pruning methods. The checkpoints and code are available at here.

----

## [1188] HODEC: Towards Efficient High-Order DEcomposed Convolutional Neural Networks

**Authors**: *Miao Yin, Yang Sui, Wanzhao Yang, Xiao Zang, Yu Gong, Bo Yuan*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01198](https://doi.org/10.1109/CVPR52688.2022.01198)

**Abstract**:

High-order decomposition is a widely used model compression approach towards compact convolutional neural networks (CNNs). However, many of the existing solutions, though can efficiently reduce CNN model sizes, are very difficult to bring considerable saving for computational costs, especially when the compression ratio is not huge, thereby causing the severe computation inefficiency problem. To overcome this challenge, in this paper we propose efficient High-Order DEcomposed Convolution (HODEC). By performing systematic explorations on the underlying reason and mitigation strategy for the computation inefficiency, we develop a new decomposition and computation-efficient execution scheme, enabling simultaneous reductions in computational and storage costs. To demonstrate the effectiveness of HODEC, we perform empirical evaluations for various CNN models on different datasets. HODEC shows consistently outstanding compression and acceleration performance. For compressing ResNet-56 on CIFAR-10 dataset, HODEC brings 67% fewer parameters and 62% fewer FLOPs with 1.17% accuracy increase than the baseline model. For compressing ResNet-50 on ImageNet dataset, HODEC achieves 63% FLOPs reduction with 0.31% accuracy increase than the uncompressed model.

----

## [1189] AdaViT: Adaptive Vision Transformers for Efficient Image Recognition

**Authors**: *Lingchen Meng, Hengduo Li, Bor-Chun Chen, Shiyi Lan, Zuxuan Wu, Yu-Gang Jiang, Ser-Nam Lim*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01199](https://doi.org/10.1109/CVPR52688.2022.01199)

**Abstract**:

Built on top of self-attention mechanisms, vision transformers have demonstrated remarkable performance on a variety of tasks recently. While achieving excellent performance, they still require relatively intensive computational cost that scales up drastically as the numbers of patches, self-attention heads and transformer blocks increase. In this paper, we argue that due to the large variations among images, their need for modeling long-range dependencies between patches differ. To this end, we introduce AdaViT, an adaptive computation framework that learns to derive usage policies on which patches, self-attention heads and transformer blocks to use throughout the backbone on a per-input basis, aiming to improve inference efficiency of vision transformers with a minimal drop of accuracy for image recognition. Optimized jointly with a transformer backbone in an end-to-end manner, a light-weight decision network is attached to the backbone to produce decisions on-the-fly. Extensive experiments on ImageNet demonstrate that our method obtains more than 2 × improvement on efficiency compared to state-of-the-art vision transformers with only 0.8% drop of accuracy, achieving good efficiency/accuracy trade-offs conditioned on different computational budgets. We further conduct quantitative and qualitative analysis on learned usage polices and provide more insights on the redundancy in vision transformers. Code is available at ht tps: / / gi thub. com/MengLcool/AdaVi T.

----

## [1190] Cross-Image Relational Knowledge Distillation for Semantic Segmentation

**Authors**: *Chuanguang Yang, Helong Zhou, Zhulin An, Xue Jiang, Yongjun Xu, Qian Zhang*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01200](https://doi.org/10.1109/CVPR52688.2022.01200)

**Abstract**:

Current Knowledge Distillation (KD) methods for semantic segmentation often guide the student to mimic the teacher's structured information generated from individual data samples. However, they ignore the global semantic relations among pixels across various images that are valuable for KD. This paper proposes a novel Cross-Image Relational KD (CIRKD), which focuses on transferring structured pixel-to-pixel and pixel-to-region relations among the whole images. The motivation is that a good teacher network could construct a well-structured feature space in terms of global pixel dependencies. CIRKD makes the student mimic better structured semantic relations from the teacher, thus improving the segmentation performance. Experimental results over Cityscapes, CamVid and Pascal VOC datasets demonstrate the effectiveness of our proposed approach against state-of-the-art distillation methods. The code is available at https://github.com/winycg/CIRKD.

----

## [1191] MrBiQ: Post-Training Non-Uniform Quantization based on Minimizing the Reconstruction Error

**Authors**: *Yongkweon Jeon, Chungman Lee, Eulrang Cho, Yeonju Ro*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01201](https://doi.org/10.1109/CVPR52688.2022.01201)

**Abstract**:

Post-training quantization compresses a neural network within few hours with only a small unlabeled calibration set. However, so far it has been only discussed and empirically demonstrated in the context of uniform quantization on convolutional neural networks. We thus propose a new posttraining non-uniform quantization method, called Mr.BiQ, allowing low bit-width quantization even on Transformer models. In particular, we leverage multi-level binarization for weights while allowing activations to be represented as various data formats (e.g., INT8, bfloat16, binary-coding, and FP32). Unlike conventional methods which optimize full-precision weights first, then decompose the weights into quantization parameters, Mr.BiQ recognizes the quantization parameters (i.e., scaling factors and bit-code) as directly and jointly learnable parameters during the optimization. To verify the superiority of the proposed quantization scheme, we test Mr.BiQ on various models including convolutional neural networks and Transformer models. According to experimental results, Mr.BiQ shows significant improvement in terms of accuracy when the bit-width of weights is equal to 2: up to 5.35 p.p. improvement in CNNs, up to 4.23 p.p. improvement in Vision Transformers, and up to 3.37 point improvement in Transformers for NLP.

----

## [1192] IntraQ: Learning Synthetic Images with Intra-Class Heterogeneity for Zero-Shot Network Quantization

**Authors**: *Yunshan Zhong, Mingbao Lin, Gongrui Nan, Jianzhuang Liu, Baochang Zhang, Yonghong Tian, Rongrong Ji*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01202](https://doi.org/10.1109/CVPR52688.2022.01202)

**Abstract**:

Learning to synthesize data has emerged as a promising direction in zero-shot quantization (ZSQ), which represents neural networks by low-bit integer without accessing any of the real data. In this paper, we observe an interesting phenomenon of intra-class heterogeneity in real data and show that existing methods fail to retain this property in their synthetic images, which causes a limited performance increase. To address this issue, we propose a novel zero-shot quantization method referred to as IntraQ. First, we propose a local object reinforcement that locates the target objects at different scales and positions of the synthetic images. Second, we introduce a marginal distance constraint to form class-related features distributed in a coarse area. Lastly, we devise a soft inception loss which injects a soft prior label to prevent the synthetic images from being over-fitting to a fixed object. Our IntraQ is demonstrated to well retain the intra-class heterogeneity in the synthetic images and also observed to perform state-of-the-art. For example, compared to the advanced ZSQ, our IntraQ obtains 9.17% increase of the top-1 accuracy on ImageNet when all layers of MobileNetV1 are quantized to 4-bit. Code is at https://github.com/zysxmu/IntraQ

----

## [1193] DECORE: Deep Compression with Reinforcement Learning

**Authors**: *Manoj Alwani, Yang Wang, Vashisht Madhavan*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01203](https://doi.org/10.1109/CVPR52688.2022.01203)

**Abstract**:

Deep learning has become an increasingly popular and powerful methodology for modern pattern recognition systems. However, many deep neural networks have millions or billions of parameters, making them untenable for realworld applications due to constraints on memory size or latency requirements. As a result, efficient network compression techniques are often required for the widespread adoption of deep learning methods. We present DECORE, a reinforcement learning-based approach to automate the network compression process. DECORE assigns an agent to each channel in the network along with a light policy gradient method to learn which neurons or channels to be kept or removed. Each agent in the network has just one parameter (keep or drop) to learn, which leads to a much faster training process compared to existing approaches. DECORE provides state-of-the-art compression results on various network architectures and various datasets. For example, on the ResNet-110 architecture, it achieves a 64.8% compression and 61.8% FLOPs reduction as compared to the baseline model without any accuracy loss on the CIFAR-10 dataset. It can reduce the size of regular architectures like the VGG network by up to 99% with just a small accuracy drop of 2.28%. For a larger dataset like ImageNet with only 30 epochs of training, it can compress the ResNet-50 architecture by 44.7% and reduce FLOPs by 42.3%, with just a 0.69% drop on Top-5 accuracy of the uncompressed model. We also demonstrate that DECORE can be used to search for compressed network architectures based on various constraints, such as memory and FLOPs.

----

## [1194] Towards Efficient and Scalable Sharpness-Aware Minimization

**Authors**: *Yong Liu, Siqi Mai, Xiangning Chen, Cho-Jui Hsieh, Yang You*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01204](https://doi.org/10.1109/CVPR52688.2022.01204)

**Abstract**:

Recently, Sharpness-Aware Minimization (SAM), which connects the geometry of the loss landscape and generalization, has demonstrated a significant performance boost on training large-scale models such as vision transformers. However, the update rule of SAM requires two sequential (non-parallelizable) gradient computations at each step, which can double the computational overhead. In this paper, we propose a novel algorithm LookSAM - that only periodically calculates the inner gradient ascent, to significantly reduce the additional training cost of SAM. The empirical results illustrate that LookSAM achieves similar accuracy gains to SAM while being tremendously faster - it enjoys comparable computational complexity with first-order optimizers such as SGD or Adam. To further evaluate the performance and scalability of LookSAM, we incorporate a layer-wise modification and perform experiments in the large-batch training scenario, which is more prone to converge to sharp local minima. Equipped with the proposed algorithms, we are the first to successfully scale up the batch size when training Vision Transformers (ViTs). With a 64k batch size, we are able to train ViTs from scratch in minutes while maintaining competitive performance. The code is available here: https://github.com/yong-6/LookSAM

----

## [1195] AEGNN: Asynchronous Event-based Graph Neural Networks

**Authors**: *Simon Schaefer, Daniel Gehrig, Davide Scaramuzza*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01205](https://doi.org/10.1109/CVPR52688.2022.01205)

**Abstract**:

The best performing learning algorithms devised for event cameras work by first converting events into dense representations that are then processed using standard CNNs. However, these steps discard both the sparsity and high temporal resolution of events, leading to high computational burden and latency. For this reason, recent works have adopted Graph Neural Networks (GNNs), which process events as “static” spatio-temporal graphs, which are inherently “sparse”. We take this trend one step further by introducing Asynchronous, Event-based Graph Neural Networks (AEGNNs), a novel event-processing paradigm that generalizes standard GNNs to process events as “evolving” spatio-temporal graphs. AEGNNs follow efficient update rules that restrict recomputation of network activations only to the nodes affected by each new event, thereby significantly reducing both computation and latency for event-by-event processing. AEGNNs are easily trained on synchronous inputs and can be converted to efficient, “asynchronous” networks at test time. We thoroughly validate our method on object classification and detection tasks, where we show an up to a 200-fold reduction in computational complexity (FLOPs), with similar or even better performance than state-of-the-art asynchronous methods. This reduction in computation directly translates to an 8-fold reduction in computational latency when compared to standard GNNs, which opens the door to low-latency event-based processing.

----

## [1196] DiSparse: Disentangled Sparsification for Multitask Model Compression

**Authors**: *Xinglong Sun, Ali Hassani, Zhangyang Wang, Gao Huang, Humphrey Shi*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01206](https://doi.org/10.1109/CVPR52688.2022.01206)

**Abstract**:

Despite the popularity of Model Compression and Mul-titask Learning, how to effectively compress a multitask model has been less thoroughly analyzed due to the chal-lenging entanglement of tasks in the parameter space. In this paper, we propose DiSparse, a simple, effective, and first-of-its-kind multitask pruning and sparse training scheme. We consider each task independently by disentangling the importance measurement and take the unani-mous decisions among all tasks when performing parame-ter pruning and selection. Our experimental results demon-strate superior performance on various configurations and settings compared to popular sparse training and pruning methods. Besides the effectiveness in compression, DiS-parse also provides a powerful tool to the multitask learning community. Surprisingly, we even observed better per-formance than some dedicated multitask learning methods in several cases despite the high model sparsity enforced by DiSparse. We analyzed the pruning masks generated with DiSparse and observed strikingly similar sparse net-work architecture identified by each task even before the training starts. We also observe the existence of a “water-shed” layer where the task relatedness sharply drops, implying no benefits in continued parameters sharing. Our code and models will be available at: https://github.com/SHI-Labs/DiSparse-Multitask-Model-Compression.

----

## [1197] Multi-modal Extreme Classification

**Authors**: *Anshul Mittal, Kunal Dahiya, Shreya Malani, Janani Ramaswamy, Seba Kuruvilla, Jitendra Ajmera, Keng-hao Chang, Sumeet Agarwal, Purushottam Kar, Manik Varma*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01207](https://doi.org/10.1109/CVPR52688.2022.01207)

**Abstract**:

This paper develops the MUFIN technique for extreme classification (XC) tasks with millions of labels where data-points and labels are endowed with visual and textual de-scriptors. Applications of MUFIN to product-to-product recommendation and bid query prediction over several mil-lions of products are presented. Contemporary multi-modal methods frequently rely on purely embedding-based meth-ods. On the other hand, XC methods utilize classifier ar-chitectures to offer superior accuracies than embedding-only methods but mostly focus on text-based categorization tasks. MUFIN bridges this gap by reformulating multi-modal categorization as an XC problem with several mil-lions of labels. This presents the twin challenges of devel-oping multi-modal architectures that can offer embeddings sufficiently expressive to allow accurate categorization over millions of labels; and training and inference routines that scale logarithmically in the number of labels. MUFIN de-velops an architecture based on cross-modal attention and trains it in a modular fashion using pre-training and positive and negative mining. A novel product-to-product rec-ommendation dataset MM-AmazonTitles-300K containing over 300K products was curated from publicly available amazon.com listings with each product endowed with a title and multiple images. On the MM-AmazonTitles-300K and Polyvore datasets, and a dataset with over 4 million labels curated from click logs of the Bing search engine, MUFIN offered at least 3% higher accuracy than leading text-based, image-based and multi-modal techniques.

----

## [1198] A sampling-based approach for efficient clustering in large datasets

**Authors**: *Georgios Exarchakis, Omar Oubari, Gregor Lenz*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01208](https://doi.org/10.1109/CVPR52688.2022.01208)

**Abstract**:

We propose a simple and efficient clustering method for high-dimensional data with a large number of clusters. Our algorithm achieves high-performance by evaluating distances of datapoints with a subset of the cluster centres. Our contribution is substantially more efficient than k-means as it does not require an all to all comparison of data points and clusters. We show that the optimal solutions of our approximation are the same as in the exact solution. However, our approach is considerably more efficient at extracting these clusters compared to the state-of-the-art. We compare our approximation with the exact k-means and alternative approximation approaches on a series of standardised clustering tasks. For the evaluation, we consider the algorithmic complexity, including number of operations to convergence, and the stability of the results. An efficient implementation of the algorithm is available online.

----

## [1199] Come-Closer-Diffuse-Faster: Accelerating Conditional Diffusion Models for Inverse Problems through Stochastic Contraction

**Authors**: *Hyungjin Chung, Byeongsu Sim, Jong Chul Ye*

**Conference**: *cvpr 2022*

**URL**: [https://doi.org/10.1109/CVPR52688.2022.01209](https://doi.org/10.1109/CVPR52688.2022.01209)

**Abstract**:

Diffusion models have recently attained significant interest within the community owing to their strong performance as generative models. Furthermore, its application to inverse problems have demonstrated state-of-the-art performance. Unfortunately, diffusion models have a critical downside - they are inherently slow to sample from, needing few thousand steps of iteration to generate images from pure Gaussian noise. In this work, we show that starting from Gaussian noise is unnecessary. Instead, starting from a single forward diffusion with better initialization significantly reduces the number of sampling steps in the reverse conditional diffusion. This phenomenon is formally explained by the contraction theory of the stochastic difference equations like our conditional diffusion strategy - the alternating applications of reverse diffusion followed by a non-expansive data consistency step. The new sampling strategy, dubbed Come-Closer-Diffuse-Faster (CCDF), also reveals a new insight on how the existing feed-forward neural network approaches for inverse problems can be synergistically combined with the diffusion models. Experimental results with super-resolution, image inpainting, and compressed sensing MRI demonstrate that our method can achieve state-of-the-art reconstruction performance at significantly reduced sampling steps.

----



[Go to the previous page](CVPR-2022-list05.md)

[Go to the next page](CVPR-2022-list07.md)

[Go to the catalog section](README.md)