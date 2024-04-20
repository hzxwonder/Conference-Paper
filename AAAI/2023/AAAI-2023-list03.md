## [400] BEST: BERT Pre-training for Sign Language Recognition with Coupling Tokenization

        **Authors**: *Weichao Zhao, Hezhen Hu, Wengang Zhou, Jiaxin Shi, Houqiang Li*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i3.25470](https://doi.org/10.1609/aaai.v37i3.25470)

        **Abstract**:

        In this work, we are dedicated to leveraging the BERT pre-training success and modeling the domain-specific statistics to fertilize the sign language recognition~(SLR) model. Considering the dominance of hand and body in sign language expression, we organize them as pose triplet units and feed them into the Transformer backbone in a frame-wise manner. Pre-training is performed via reconstructing the masked triplet unit from the corrupted input sequence, which learns the hierarchical correlation context cues among internal and external triplet units. Notably, different from the highly semantic word token in BERT, the pose unit is a low-level signal originally locating in continuous space, which prevents the direct adoption of the BERT cross entropy objective. To this end, we bridge this semantic gap via coupling tokenization of the triplet unit. It adaptively extracts the discrete pseudo label from the pose triplet unit, which represents the semantic gesture / body state. After pre-training, we fine-tune the pre-trained encoder on the downstream SLR task, jointly with the newly added task-specific layer. Extensive experiments are conducted to validate the effectiveness of our proposed method, achieving new state-of-the-art performance on all four benchmarks with a notable gain.

        ----

        ## [401] MulGT: Multi-Task Graph-Transformer with Task-Aware Knowledge Injection and Domain Knowledge-Driven Pooling for Whole Slide Image Analysis

        **Authors**: *Weiqin Zhao, Shujun Wang, Maximus Yeung, Tianye Niu, Lequan Yu*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i3.25471](https://doi.org/10.1609/aaai.v37i3.25471)

        **Abstract**:

        Whole slide image (WSI) has been widely used to assist automated diagnosis under the deep learning fields. However, most previous works only discuss the SINGLE task setting which is not aligned with real clinical setting, where pathologists often conduct multiple diagnosis tasks simultaneously. Also, it is commonly recognized that the multi-task learning paradigm can improve learning efficiency by exploiting commonalities and differences across multiple tasks. To this end, we present a novel multi-task framework (i.e., MulGT) for WSI analysis by the specially designed Graph-Transformer equipped with Task-aware Knowledge Injection and Domain Knowledge-driven Graph Pooling modules. Basically, with the Graph Neural Network and Transformer as the building commons, our framework is able to learn task-agnostic low-level local information as well as task-specific high-level global representation. Considering that different tasks in WSI analysis depend on different features and properties, we also design a novel Task-aware Knowledge Injection module to transfer the task-shared graph embedding into task-specific feature spaces to learn more accurate representation for different tasks. Further, we elaborately design a novel Domain Knowledge-driven Graph Pooling module for each task to improve both the accuracy and robustness of different tasks by leveraging different diagnosis patterns of multiple tasks. We evaluated our method on two public WSI datasets from TCGA projects, i.e., esophageal carcinoma and kidney carcinoma. Experimental results show that our method outperforms single-task counterparts and the state-of-theart methods on both tumor typing and staging tasks.

        ----

        ## [402] Grouped Knowledge Distillation for Deep Face Recognition

        **Authors**: *Weisong Zhao, Xiangyu Zhu, Kaiwen Guo, Xiaoyu Zhang, Zhen Lei*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i3.25472](https://doi.org/10.1609/aaai.v37i3.25472)

        **Abstract**:

        Compared with the feature-based distillation methods, logits distillation can liberalize the requirements of consistent feature dimension between teacher and student networks, while the performance is deemed inferior in face recognition. One major challenge is that the light-weight student network has difficulty fitting the target logits due to its low model capacity, which is attributed to the significant number of identities in face recognition. Therefore, we seek to probe the target logits to extract the primary knowledge related to face identity, and discard the others, to make the distillation more achievable for the student network. Specifically, there is a tail group with near-zero values in the prediction, containing minor knowledge for distillation. To provide a clear perspective of its impact, we first partition the logits into two groups, i.e., Primary Group and Secondary Group, according to the cumulative probability of the softened prediction. Then, we reorganize the Knowledge Distillation (KD) loss of grouped logits into three parts, i.e., Primary-KD, Secondary-KD, and Binary-KD. Primary-KD refers to distilling the primary knowledge from the teacher, Secondary-KD aims to refine minor knowledge but increases the difficulty of distillation, and Binary-KD ensures the consistency of knowledge distribution between teacher and student. We experimentally found that (1) Primary-KD and Binary-KD are indispensable for KD, and (2) Secondary-KD is the culprit restricting KD at the bottleneck. Therefore, we propose a Grouped Knowledge Distillation (GKD) that retains the Primary-KD and Binary-KD but omits Secondary-KD in the ultimate KD loss calculation. Extensive experimental results on popular face recognition benchmarks demonstrate the superiority of proposed GKD over state-of-the-art methods.

        ----

        ## [403] Style-Content Metric Learning for Multidomain Remote Sensing Object Recognition

        **Authors**: *Wenda Zhao, Ruikai Yang, Yu Liu, You He*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i3.25473](https://doi.org/10.1609/aaai.v37i3.25473)

        **Abstract**:

        Previous remote sensing recognition approaches predominantly perform well on the training-testing dataset. However, due to large style discrepancies not only among multidomain datasets but also within a single domain, they suffer from obvious performance degradation when applied to unseen domains. In this paper, we propose a style-content metric learning framework to address the generalizable remote sensing object recognition issue. Specifically, we firstly design an inter-class dispersion metric to encourage the model to make decision based on content rather than the style, which is achieved by dispersing predictions generated from the contents of both positive sample and negative sample and the style of input image. Secondly, we propose an intra-class compactness metric to force the model to be less style-biased by compacting classifier's predictions from the content of input image and the styles of positive sample and negative sample. Lastly, we design an intra-class interaction metric to improve model's recognition accuracy by pulling in classifier's predictions obtained from the input image and positive sample. Extensive experiments on four datasets show that our style-content metric learning achieves superior generalization performance against the state-of-the-art competitors. Code and model are available at: https://github.com/wdzhao123/TSCM.

        ----

        ## [404] Occupancy Planes for Single-View RGB-D Human Reconstruction

        **Authors**: *Xiaoming Zhao, Yuan-Ting Hu, Zhongzheng Ren, Alexander G. Schwing*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i3.25474](https://doi.org/10.1609/aaai.v37i3.25474)

        **Abstract**:

        Single-view RGB-D human reconstruction with implicit functions is often formulated as per-point classification. Specifically,  a set of 3D locations within the view-frustum of the camera are first projected independently onto the image and a corresponding feature is subsequently extracted for each 3D location. The feature of each 3D location is then used to classify independently whether the corresponding 3D point is inside or outside the observed object. This procedure leads to sub-optimal results because correlations between predictions for neighboring locations are only taken into account implicitly via the extracted features. For more accurate results we propose the occupancy planes (OPlanes) representation, which enables to formulate single-view RGB-D human reconstruction as occupancy prediction on planes which slice through the camera's view frustum. Such a representation provides more flexibility than voxel grids and enables to better leverage correlations than per-point classification. On the challenging S3D data we observe a simple classifier based on the OPlanes representation to yield compelling results, especially in difficult situations with partial occlusions due to other objects and partial visibility, which haven't been addressed by prior work.

        ----

        ## [405] Deep Equilibrium Models for Snapshot Compressive Imaging

        **Authors**: *Yaping Zhao, Siming Zheng, Xin Yuan*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i3.25475](https://doi.org/10.1609/aaai.v37i3.25475)

        **Abstract**:

        The ability of snapshot compressive imaging (SCI) systems to efficiently capture high-dimensional (HD) data has led to an inverse problem, which consists of recovering the HD signal from the compressed and noisy measurement. While reconstruction algorithms grow fast to solve it with the recent advances of deep learning, the fundamental issue of accurate and stable recovery remains. To this end, we propose deep equilibrium models (DEQ) for video SCI, fusing data-driven regularization and stable convergence in a theoretically sound manner. Each equilibrium model implicitly learns a nonexpansive operator and analytically computes the fixed point, thus enabling unlimited iterative steps and infinite network depth with only a constant memory requirement in training and testing. Specifically, we demonstrate how DEQ can be applied to two existing models for video SCI reconstruction: recurrent neural networks (RNN) and Plug-and-Play (PnP) algorithms. On a variety of datasets and real data, both quantitative and qualitative evaluations of our results demonstrate the effectiveness and stability of our proposed method. The code and models are available at: https://github.com/IndigoPurple/DEQSCI.

        ----

        ## [406] Unsupervised Deep Video Denoising with Untrained Network

        **Authors**: *Huan Zheng, Tongyao Pang, Hui Ji*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i3.25476](https://doi.org/10.1609/aaai.v37i3.25476)

        **Abstract**:

        Deep learning has become a prominent tool for video denoising. However, most existing deep video denoising methods require supervised training using noise-free videos. Collecting noise-free videos can be costly and challenging in many applications. Therefore, this paper aims to develop an unsupervised deep learning method for video denoising that only uses a single test noisy video for training. To achieve this, an unsupervised loss function is presented that provides an unbiased estimator of its supervised counterpart defined on noise-free video. Additionally, a temporal attention mechanism is proposed to exploit redundancy among frames. The experiments on video denoising demonstrate that the proposed unsupervised method outperforms existing unsupervised methods and remains competitive against recent supervised deep learning methods.

        ----

        ## [407] Attack Can Benefit: An Adversarial Approach to Recognizing Facial Expressions under Noisy Annotations

        **Authors**: *Jiawen Zheng, Bo Li, Shengchuan Zhang, Shuang Wu, Liujuan Cao, Shouhong Ding*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i3.25477](https://doi.org/10.1609/aaai.v37i3.25477)

        **Abstract**:

        The real-world Facial Expression Recognition (FER) datasets usually exhibit complex scenarios with coupled noise annotations and imbalanced classes distribution, which undoubtedly impede the development of FER methods. To address the aforementioned issues, in this paper, we propose a novel and flexible method to spot noisy labels by leveraging adversarial attack, termed as Geometry Aware Adversarial Vulnerability Estimation (GAAVE). Different from existing state-of-the-art methods of noisy label learning (NLL), our method has no reliance on additional information and is thus easy to generalize to the large-scale real-world FER datasets. Besides, the combination of Dataset Splitting module and Subset Refactoring module mitigates the impact of class imbalance, and the Self-Annotator module facilitates the sufficient use of all training data. Extensive experiments on RAF-DB, FERPlus, AffectNet, and CIFAR-10 datasets validate the effectiveness of our method. The stabilized enhancement based on different methods demonstrates the flexibility of our proposed GAAVE.

        ----

        ## [408] Phrase-Level Temporal Relationship Mining for Temporal Sentence Localization

        **Authors**: *Minghang Zheng, Sizhe Li, Qingchao Chen, Yuxin Peng, Yang Liu*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i3.25478](https://doi.org/10.1609/aaai.v37i3.25478)

        **Abstract**:

        In this paper, we address the problem of video temporal sentence localization, which aims to localize a target moment from videos according to a given language query. We observe that existing models suffer from a sheer performance drop when dealing with simple phrases contained in the sentence. It reveals the limitation that existing models only capture the annotation bias of the datasets but lack sufficient understanding of the semantic phrases in the query. To address this problem, we propose a phrase-level Temporal Relationship Mining (TRM) framework employing the temporal relationship relevant to the phrase and the whole sentence to have a better understanding of each semantic entity in the sentence. Specifically, we use phrase-level predictions to refine the sentence-level prediction, and use Multiple Instance Learning to improve the quality of phrase-level predictions. We also exploit the consistency and exclusiveness constraints of phrase-level and sentence-level predictions to regularize the training process, thus alleviating the ambiguity of each phrase prediction. The proposed approach sheds light on how machines can understand detailed phrases in a sentence and their compositions in their generality rather than learning the annotation biases. Experiments on the ActivityNet Captions and Charades-STA datasets show the effectiveness of our method on both phrase and sentence temporal localization and enable better model interpretability and generalization when dealing with unseen compositions of seen concepts. Code can be found at https://github.com/minghangz/TRM.

        ----

        ## [409] Learning Semantic Degradation-Aware Guidance for Recognition-Driven Unsupervised Low-Light Image Enhancement

        **Authors**: *Naishan Zheng, Jie Huang, Man Zhou, Zizheng Yang, Qi Zhu, Feng Zhao*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i3.25479](https://doi.org/10.1609/aaai.v37i3.25479)

        **Abstract**:

        Low-light images suffer severe degradation of low lightness and noise corruption, causing unsatisfactory visual quality and visual recognition performance. 
To solve this problem while meeting the unavailability of paired datasets in wide-range scenarios, unsupervised low-light image enhancement (ULLIE) techniques have been developed. 
However, these methods are primarily guided to alleviate the degradation effect on visual quality rather than semantic levels, hence limiting their performance in visual recognition tasks. 
To this end, we propose to learn a Semantic Degradation-Aware Guidance (SDAG) that perceives the low-light degradation effect on semantic levels in a self-supervised manner, which is further utilized to guide the ULLIE methods.
The proposed SDAG utilizes the low-light degradation factors as augmented signals to degrade the low-light images, and then capture their degradation effect on semantic levels. 
Specifically, our SDAG employs the subsequent pre-trained recognition model extractor to extract semantic representations, and then learns to self-reconstruct the enhanced low-light image and its augmented degraded images.   
By constraining the relative reconstruction effect between the original enhanced image and the augmented formats, our SDAG learns to be aware of the degradation effect on semantic levels in a relative comparison manner. 
Moreover, our SDAG is general and can be plugged into the training paradigm of the existing ULLIE methods.  
Extensive experiments demonstrate its effectiveness for improving the ULLIE approaches on the downstream recognition tasks while maintaining a competitive visual quality. 
Code will be available at https://github.com/zheng980629/SDAG.

        ----

        ## [410] Memory-Aided Contrastive Consensus Learning for Co-salient Object Detection

        **Authors**: *Peng Zheng, Jie Qin, Shuo Wang, Tian-Zhu Xiang, Huan Xiong*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i3.25480](https://doi.org/10.1609/aaai.v37i3.25480)

        **Abstract**:

        Co-salient object detection (CoSOD) aims at detecting common salient objects within a group of relevant source images. Most of the latest works employ the attention mechanism for finding common objects. To achieve accurate CoSOD results with high-quality maps and high efficiency, we propose a novel Memory-aided Contrastive Consensus Learning (MCCL) framework, which is capable of effectively detecting co-salient objects in real time (∼150 fps). To learn better group consensus, we propose the Group Consensus Aggregation Module (GCAM) to abstract the common features of each image group; meanwhile, to make the consensus representation more discriminative, we introduce the Memory-based Contrastive Module (MCM), which saves and updates the consensus of images from different groups in a queue of memories. Finally, to improve the quality and integrity of the predicted maps, we develop an Adversarial Integrity Learning (AIL) strategy to make the segmented regions more likely composed of complete objects with less surrounding noise. Extensive experiments on all the latest CoSOD benchmarks demonstrate that our lite MCCL outperforms 13 cutting-edge models, achieving the new state of the art (∼5.9% and ∼6.2% improvement in S-measure on CoSOD3k and CoSal2015, respectively). Our source codes, saliency maps, and online demos are publicly available at https://github.com/ZhengPeng7/MCCL.

        ----

        ## [411] MaskBooster: End-to-End Self-Training for Sparsely Supervised Instance Segmentation

        **Authors**: *Shida Zheng, Chenshu Chen, Xi Yang, Wenming Tan*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i3.25481](https://doi.org/10.1609/aaai.v37i3.25481)

        **Abstract**:

        The present paper introduces sparsely supervised instance segmentation, with the datasets being fully annotated bounding boxes and sparsely annotated masks. A direct solution to this task is self-training, which is not fully explored for instance segmentation yet. In this paper, we propose MaskBooster for sparsely supervised instance segmentation (SpSIS) with comprehensive usage of pseudo masks. MaskBooster is featured with (1) dynamic and progressive pseudo masks from an online updating teacher model, (2) refining binary pseudo masks with the help of bounding box prior, (3) learning inter-class prediction distribution via knowledge distillation for soft pseudo masks. As an end-to-end and universal self-training framework, MaskBooster can empower fully supervised algorithms and boost their segmentation performance on SpSIS. Abundant experiments are conducted on COCO and BDD100K datasets and validate the effectiveness of MaskBooster. Specifically, on different COCO protocols and BDD100K, we surpass sparsely supervised baseline by a large margin for both Mask RCNN and ShapeProp. MaskBooster on SpSIS also outperforms weakly and semi-supervised instance segmentation state-of-the-art on the datasets with similar annotation budgets.

        ----

        ## [412] RSPT: Reconstruct Surroundings and Predict Trajectory for Generalizable Active Object Tracking

        **Authors**: *Fangwei Zhong, Xiao Bi, Yudi Zhang, Wei Zhang, Yizhou Wang*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i3.25482](https://doi.org/10.1609/aaai.v37i3.25482)

        **Abstract**:

        Active Object Tracking (AOT) aims to maintain a specific relation between the tracker and object(s) by autonomously controlling the motion system of a tracker given observations. It is widely used in various applications such as mobile robots and autonomous driving. However, Building a generalizable active tracker that works robustly across various scenarios remains a challenge, particularly in unstructured environments with cluttered obstacles and diverse layouts. To realize this, we argue that the key is to construct a state representation that can model the geometry structure of the surroundings and the dynamics of the target. To this end, we propose a framework called RSPT to form a structure-aware motion representation by Reconstructing Surroundings and Predicting the target Trajectory. Moreover, we further enhance the generalization of the policy network by training in the asymmetric dueling mechanism.  Empirical results show that RSPT outperforms existing methods in unseen environments, especially those with cluttered obstacles and diverse layouts. We also demonstrate good sim-to-real transfer when deploying RSPT in real-world scenarios.

        ----

        ## [413] STOA-VLP: Spatial-Temporal Modeling of Object and Action for Video-Language Pre-training

        **Authors**: *Weihong Zhong, Mao Zheng, Duyu Tang, Xuan Luo, Heng Gong, Xiaocheng Feng, Bing Qin*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i3.25483](https://doi.org/10.1609/aaai.v37i3.25483)

        **Abstract**:

        Although large-scale video-language pre-training models, which usually build a global alignment between the video and the text, have achieved remarkable progress on various downstream tasks, the idea of adopting fine-grained information during the pre-training stage is not well explored. In this work, we propose STOA-VLP, a pre-training framework that jointly models object and action information across spatial and temporal dimensions. More specifically, the model regards object trajectories across frames and multiple action features from the video as fine-grained features. Besides, We design two auxiliary tasks to better incorporate both kinds of information into the pre-training process of the video-language model. The first is the dynamic object-text alignment task, which builds a better connection between object trajectories and the relevant noun tokens. The second is the spatial-temporal action set prediction, which guides the model to generate consistent action features by predicting actions found in the text. Extensive experiments on three downstream tasks (video captioning, text-video retrieval, and video question answering) demonstrate the effectiveness of our proposed STOA-VLP (e.g. 3.7 Rouge-L improvements on MSR-VTT video captioning benchmark, 2.9% accuracy improvements on MSVD video question answering benchmark, compared to previous approaches).

        ----

        ## [414] Refined Semantic Enhancement towards Frequency Diffusion for Video Captioning

        **Authors**: *Xian Zhong, Zipeng Li, Shuqin Chen, Kui Jiang, Chen Chen, Mang Ye*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i3.25484](https://doi.org/10.1609/aaai.v37i3.25484)

        **Abstract**:

        Video captioning aims to generate natural language sentences that describe the given video accurately. Existing methods obtain favorable generation by exploring richer visual representations in encode phase or improving the decoding ability. However, the long-tailed problem hinders these attempts at low-frequency tokens, which rarely occur but carry critical semantics, playing a vital role in the detailed generation. In this paper, we introduce a novel Refined Semantic enhancement method towards Frequency Diffusion (RSFD), a captioning model that constantly perceives the linguistic representation of the infrequent tokens. Concretely, a Frequency-Aware Diffusion (FAD) module is proposed to comprehend the semantics of low-frequency tokens to break through generation limitations. In this way, the caption is refined by promoting the absorption of tokens with insufficient occurrence. Based on FAD, we design a Divergent Semantic Supervisor (DSS) module to compensate for the information loss of high-frequency tokens brought by the diffusion process, where the semantics of low-frequency tokens is further emphasized to alleviate the long-tailed problem. Extensive experiments indicate that RSFD outperforms the state-of-the-art methods on two benchmark datasets, i.e., MSR-VTT and MSVD, demonstrate that the enhancement of low-frequency tokens semantics can obtain a competitive generation effect. Code is available at https://github.com/lzp870/RSFD.

        ----

        ## [415] Aesthetically Relevant Image Captioning

        **Authors**: *Zhipeng Zhong, Fei Zhou, Guoping Qiu*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i3.25485](https://doi.org/10.1609/aaai.v37i3.25485)

        **Abstract**:

        Image aesthetic quality assessment (AQA) aims to assign numerical aesthetic ratings to images whilst image aesthetic captioning (IAC) aims to generate textual descriptions of the aesthetic aspects of images. In this paper, we study image AQA and IAC together and present a new IAC method termed Aesthetically Relevant Image Captioning (ARIC). Based on the observation that most textual comments of an image are about objects and their interactions rather than aspects of aesthetics, we first introduce the concept of Aesthetic Relevance Score (ARS) of a sentence and have developed a model to automatically label a sentence with its ARS. We then use the ARS to design the ARIC model which includes an ARS weighted IAC loss function and an ARS based diverse aesthetic caption selector (DACS). We present extensive experimental results to show the soundness of the ARS concept and the effectiveness of the ARIC model by demonstrating that texts with higher ARS’s can predict the aesthetic ratings more accurately and that the new ARIC model can generate more accurate, aesthetically more relevant and more diverse image captions. Furthermore, a large new research database containing 510K images with over 5 million comments and 350K aesthetic scores, and code for implementing ARIC, are available at https://github.com/PengZai/ARIC

        ----

        ## [416] Polarization-Aware Low-Light Image Enhancement

        **Authors**: *Chu Zhou, Minggui Teng, Youwei Lyu, Si Li, Chao Xu, Boxin Shi*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i3.25486](https://doi.org/10.1609/aaai.v37i3.25486)

        **Abstract**:

        Polarization-based vision algorithms have found uses in various applications since polarization provides additional physical constraints. However, in low-light conditions, their performance would be severely degenerated since the captured polarized images could be noisy, leading to noticeable degradation in the degree of polarization (DoP) and the angle of polarization (AoP). Existing low-light image enhancement methods cannot handle the polarized images well since they operate in the intensity domain, without effectively exploiting the information provided by polarization. In this paper, we propose a Stokes-domain enhancement pipeline along with a dual-branch neural network to handle the problem in a polarization-aware manner. Two application scenarios (reflection removal and shape from polarization) are presented to show how our enhancement can improve their results.

        ----

        ## [417] Progressive Bayesian Inference for Scribble-Supervised Semantic Segmentation

        **Authors**: *Chuanwei Zhou, Chunyan Xu, Zhen Cui*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i3.25487](https://doi.org/10.1609/aaai.v37i3.25487)

        **Abstract**:

        The scribble-supervised semantic segmentation is an important yet challenging task in the field of computer vision. To deal with the pixel-wise sparse annotation problem, we propose a Progressive Bayesian Inference (PBI) framework to boost the performance of the scribble-supervised semantic segmentation, which can effectively infer the semantic distribution of these unlabeled pixels to guide the optimization of the segmentation network.  The PBI dynamically improves the model learning from two aspects: the Bayesian inference module (i.e., semantic distribution learning) and the pixel-wise segmenter (i.e., model updating). Specifically, we effectively infer the semantic probability distribution of these unlabeled pixels with our designed Bayesian inference module, where its guidance is estimated through the Bayesian expectation maximization under the situation of partially observed data. The segmenter can be progressively improved under the joint guidance of the original scribble information and the learned semantic distribution. The segmenter optimization and semantic distribution promotion are encapsulated into a unified architecture where they could improve each other with mutual evolution in a progressive fashion. Comprehensive evaluations of several benchmark datasets demonstrate the effectiveness and superiority of our proposed PBI when compared with other state-of-the-art methods applied to the scribble-supervised semantic segmentation task.

        ----

        ## [418] Exploratory Inference Learning for Scribble Supervised Semantic Segmentation

        **Authors**: *Chuanwei Zhou, Zhen Cui, Chunyan Xu, Cao Han, Jian Yang*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i3.25488](https://doi.org/10.1609/aaai.v37i3.25488)

        **Abstract**:

        Scribble supervised semantic segmentation has achieved great advances in pseudo label exploitation, yet suffers insufficient label exploration for the mass of unannotated regions. In this work, we propose a novel exploratory inference learning (EIL) framework, which facilitates efficient probing on unlabeled pixels and promotes selecting confident candidates for boosting the evolved segmentation. The exploration of unannotated regions is formulated as an iterative decision-making process, where a policy searcher learns to infer in the unknown space and the reward to the exploratory policy is based on a contrastive measurement of candidates. In particular, we devise the contrastive reward with the intra-class attraction and the inter-class repulsion in the feature space w.r.t the pseudo labels. The unlabeled exploration and the labeled exploitation are jointly balanced to improve the segmentation, and framed in a close-looping end-to-end network. Comprehensive evaluations on the benchmark datasets (PASCAL VOC 2012 and PASCAL Context) demonstrate the superiority of our proposed EIL when compared with other state-of-the-art methods for the scribble-supervised semantic segmentation problem.

        ----

        ## [419] Dual Memory Units with Uncertainty Regulation for Weakly Supervised Video Anomaly Detection

        **Authors**: *Hang Zhou, Junqing Yu, Wei Yang*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i3.25489](https://doi.org/10.1609/aaai.v37i3.25489)

        **Abstract**:

        Learning discriminative features for effectively separating abnormal events from normality is crucial for weakly supervised video anomaly detection (WS-VAD) tasks. Existing approaches, both video and segment level label oriented, mainly focus on extracting representations for anomaly data while neglecting the implication of normal data. We observe that such a scheme is sub-optimal, i.e., for better distinguishing anomaly one needs to understand what is a normal state, and may yield a higher false alarm rate. To address this issue, we propose an Uncertainty Regulated Dual Memory Units (UR-DMU) model to learn both the representations of normal data and discriminative features of abnormal data. To be specific, inspired by the traditional global and local structure on graph convolutional networks, we introduce a Global and Local Multi-Head Self Attention (GL-MHSA) module for the Transformer network to obtain more expressive embeddings for capturing associations in videos. Then, we use two memory banks, one additional abnormal memory for tackling hard samples, to store and separate abnormal and normal prototypes and maximize the margins between the two representations. Finally, we propose an uncertainty learning scheme to learn the normal data latent space, that is robust to noise from camera switching, object changing, scene transforming, etc. Extensive experiments on XD-Violence and UCF-Crime datasets demonstrate that our method outperforms the state-of-the-art methods by a sizable margin.

        ----

        ## [420] Unsupervised Hierarchical Domain Adaptation for Adverse Weather Optical Flow

        **Authors**: *Hanyu Zhou, Yi Chang, Gang Chen, Luxin Yan*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i3.25490](https://doi.org/10.1609/aaai.v37i3.25490)

        **Abstract**:

        Optical flow estimation has made great progress, but usually suffers from degradation under adverse weather. Although semi/full-supervised methods have made good attempts, the domain shift between the synthetic and real adverse weather images would deteriorate their performance. To alleviate this issue, our start point is to unsupervisedly transfer the knowledge from source clean domain to target degraded domain. Our key insight is that adverse weather does not change the intrinsic optical flow of the scene, but causes a significant difference for the warp error between clean and degraded images. In this work, we propose the first unsupervised framework for adverse weather optical flow via hierarchical motion-boundary adaptation. Specifically, we first employ image translation to construct the transformation relationship between clean and degraded domains. In motion adaptation, we utilize the flow consistency knowledge to align the cross-domain optical flows into a motion-invariance common space, where the optical flow from clean weather is used as the guidance-knowledge to obtain a preliminary optical flow for adverse weather. Furthermore, we leverage the warp error inconsistency which measures the motion misalignment of the boundary between the clean and degraded domains, and propose a joint intra- and inter-scene boundary contrastive adaptation to refine the motion boundary. The hierarchical motion and boundary adaptation jointly promotes optical flow in a unified framework. Extensive quantitative and qualitative experiments have been performed to verify the superiority of the proposed method.

        ----

        ## [421] PASS: Patch Automatic Skip Scheme for Efficient Real-Time Video Perception on Edge Devices

        **Authors**: *Qihua Zhou, Song Guo, Jun Pan, Jiacheng Liang, Zhenda Xu, Jingren Zhou*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i3.25491](https://doi.org/10.1609/aaai.v37i3.25491)

        **Abstract**:

        Real-time video perception tasks are often challenging over the resource-constrained edge devices due to the concerns of accuracy drop and hardware overhead, where saving computations is the key to performance improvement. Existing methods either rely on domain-specific neural chips or priorly searched models, which require specialized optimization according to different task properties. In this work, we propose a general and task-independent Patch Automatic Skip Scheme (PASS), a novel end-to-end learning pipeline to support diverse video perception settings by decoupling acceleration and tasks. The gist is to capture the temporal similarity across video frames and skip the redundant computations at patch level, where the patch is a non-overlapping square block in visual. PASS equips each convolution layer with a learnable gate to selectively determine which patches could be safely skipped without degrading model accuracy. As to each layer, a desired gate needs to make flexible skip decisions based on intermediate features without any annotations, which cannot be achieved by conventional supervised learning paradigm. To address this challenge, we are the first to construct a tough self-supervisory procedure for optimizing these gates, which learns to extract contrastive representation, i.e., distinguishing similarity and difference, from frame sequence. These high-capacity gates can serve as a plug-and-play module for convolutional neural network (CNN) backbones to implement patch-skippable architectures, and automatically generate proper skip strategy to accelerate different video-based downstream tasks, e.g., outperforming the state-of-the-art MobileHumanPose (MHP) in 3D pose estimation and FairMOT in multiple object tracking, by up to 9.43 times and 12.19 times speedups, respectively. By directly processing the raw data of frames, PASS can generalize to real-time video streams on commodity edge devices, e.g., NVIDIA Jetson Nano, with efficient performance in realistic deployment.

        ----

        ## [422] Robust Feature Rectification of Pretrained Vision Models for Object Recognition

        **Authors**: *Shengchao Zhou, Gaofeng Meng, Zhaoxiang Zhang, Richard Yi Da Xu, Shiming Xiang*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i3.25492](https://doi.org/10.1609/aaai.v37i3.25492)

        **Abstract**:

        Pretrained vision models for object recognition often suffer a dramatic performance drop with degradations unseen during training. In this work, we propose a RObust FEature Rectification module (ROFER) to improve the performance of pretrained models against degradations. Specifically, ROFER first estimates the type and intensity of the degradation that corrupts the image features. Then, it leverages a Fully Convolutional Network (FCN) to rectify the features from the degradation by pulling them back to clear features. ROFER is a general-purpose module that can address various degradations simultaneously, including blur, noise, and low contrast. Besides, it can be plugged into pretrained models seamlessly to rectify the degraded features without retraining the whole model. Furthermore, ROFER can be easily extended to address composite degradations by adopting a beam search algorithm to find the composition order. Evaluations on CIFAR-10 and Tiny-ImageNet demonstrate that the accuracy of ROFER is 5% higher than that of SOTA methods on different degradations. With respect to composite degradations, ROFER improves the accuracy of a pretrained CNN by 10% and 6% on CIFAR-10 and Tiny-ImageNet respectively.

        ----

        ## [423] Video Object of Interest Segmentation

        **Authors**: *Siyuan Zhou, Chunru Zhan, Biao Wang, Tiezheng Ge, Yuning Jiang, Li Niu*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i3.25493](https://doi.org/10.1609/aaai.v37i3.25493)

        **Abstract**:

        In this work, we present a new computer vision task named video object of interest segmentation (VOIS). Given a video and a target image of interest, our objective is to simultaneously segment and track all objects in the video that are relevant to the target image. This problem combines the traditional video object segmentation task with an additional image indicating the content that users are concerned with. Since no existing dataset is perfectly suitable for this new task, we specifically construct a large-scale dataset called LiveVideos, which contains 2418 pairs of target images and live videos with instance-level annotations. In addition, we propose a transformer-based method for this task. We revisit Swin Transformer and design a dual-path structure to fuse video and image features. Then, a transformer decoder is employed to generate object proposals for segmentation and tracking from the fused features. Extensive experiments on LiveVideos dataset show the superiority of our proposed method.

        ----

        ## [424] Tree-Structured Trajectory Encoding for Vision-and-Language Navigation

        **Authors**: *Xinzhe Zhou, Yadong Mu*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i3.25494](https://doi.org/10.1609/aaai.v37i3.25494)

        **Abstract**:

        Over the past few years, the research on vision-and-language navigation (VLN) has made tremendous progress. Many previous works attempted to improve the performance from different aspects like training strategy, data augmentation, pre-training, etc. This work focuses on a rarely-explored aspect in VLN, namely the trajectory organization and encoding during the navigation. Most of existing state-of-the-art VLN models adopt a vanilla sequential strategy for encoding the trajectories. Such strategy takes the whole trajectory as a single sequence to estimate the current state,  no matter whether the agent moved smoothly or perhaps made mistakes and backtracked in the past. We show that the sequential encoding may largely lose this kind of fine-grained structure in the trajectory, which could hamper the later state estimation and decision making. In order to solve this problem, this work proposes a novel tree-structured trajectory encoding strategy. The whole trajectory is organized as a tree rooted from the starting position, and encoded using our Tree-Transformer module to fully extract the fine-grained historical information. Besides, as the spatial topology could be easily embedded in the trajectory tree, we further design a tree-based action space to allow the agent making long-range error-correction in one decision. We implement the holistic agent based on cross-modal transformer and train it with a newly-proposed Tree-nDTW reward. On the benchmark dataset R2R, our model achieves a surpassing success rate (SR) of 68% on val-unseen and 66% on test. We further conduct extensive ablation studies and analyses to provide more insights for the effectiveness our designs.

        ----

        ## [425] Self-Supervised Action Representation Learning from Partial Spatio-Temporal Skeleton Sequences

        **Authors**: *Yujie Zhou, Haodong Duan, Anyi Rao, Bing Su, Jiaqi Wang*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i3.25495](https://doi.org/10.1609/aaai.v37i3.25495)

        **Abstract**:

        Self-supervised learning has demonstrated remarkable capability in representation learning for skeleton-based action recognition. Existing methods mainly focus on applying global data augmentation to generate different views of the skeleton sequence for contrastive learning. However, due to the rich action clues in the skeleton sequences, existing methods may only take a global perspective to learn to discriminate different skeletons without thoroughly leveraging the local relationship between different skeleton joints and video frames, which is essential for real-world applications. In this work, we propose a Partial Spatio-Temporal Learning (PSTL) framework to exploit the local relationship from a partial skeleton sequences built by a unique spatio-temporal masking strategy.  Specifically, we construct a negative-sample-free triplet steam structure that is composed of an anchor stream without any masking, a spatial masking stream with Central Spatial Masking (CSM), and a temporal masking stream with Motion Attention Temporal Masking (MATM). The feature cross-correlation matrix is measured between the anchor stream and the other two masking streams, respectively. (1) Central Spatial Masking discards selected joints from the feature calculation process, where the joints with a higher degree of centrality have a higher possibility of being selected.  (2) Motion Attention Temporal Masking leverages the motion of action and remove frames that move faster with a higher possibility. Our method achieves state-of-the-art performance on NTURGB+D 60, NTURGB+D 120 and PKU-MMD under various downstream tasks. Furthermore, to simulate the real-world scenarios, a practical evaluation is performed where some skeleton joints are lost in downstream tasks.In contrast to previous methods that suffer from large performance drops, our PSTL can still achieve remarkable results under this challenging setting, validating the robustness of our method.

        ----

        ## [426] Debiased Fine-Tuning for Vision-Language Models by Prompt Regularization

        **Authors**: *Beier Zhu, Yulei Niu, Saeil Lee, Minhoe Hur, Hanwang Zhang*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i3.25496](https://doi.org/10.1609/aaai.v37i3.25496)

        **Abstract**:

        We present a new paradigm for fine-tuning large-scale vision-language pre-trained models on downstream task, dubbed Prompt Regularization (ProReg). Different from traditional fine-tuning which easily overfits to the downstream task data, ProReg uses the prediction by prompting the pretrained model to regularize the fine-tuning. The motivation is: by prompting the large model “a photo of a [CLASS]”, the fill-in answer is only dependent on the pretraining encyclopedic knowledge while independent of the task data distribution, which is usually biased. Specifically, given a training sample prediction during fine-tuning, we first calculate its Kullback-Leibler loss of the prompt prediction and Cross-Entropy loss of the ground-truth label, and then combine them with a proposed sample-wise adaptive trade- off weight, which automatically adjusts the transfer between the pretrained and downstream domains. On various out-of-distribution benchmarks, we show the consistently strong performance of ProReg compared with conventional fine-tuning, zero-shot prompt, prompt tuning, and other state-of-the-art methods.

        ----

        ## [427] Improving Scene Text Image Super-resolution via Dual Prior Modulation Network

        **Authors**: *Shipeng Zhu, Zuoyan Zhao, Pengfei Fang, Hui Xue*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i3.25497](https://doi.org/10.1609/aaai.v37i3.25497)

        **Abstract**:

        Scene text image super-resolution (STISR) aims to simultaneously increase the resolution and legibility of the text images, and the resulting images will significantly affect the performance of downstream tasks. Although numerous progress has been made, existing approaches raise two crucial issues: (1) They neglect the global structure of the text, which bounds the semantic determinism of the scene text. (2) The priors, e.g., text prior or stroke prior, employed in existing works, are extracted from pre-trained text recognizers. That said, such priors suffer from the domain gap including low resolution and blurriness caused by poor imaging conditions, leading to incorrect guidance. Our work addresses these gaps and proposes a plug-and-play module dubbed Dual Prior Modulation Network (DPMN), which leverages dual image-level priors to bring performance gain over existing approaches. Specifically, two types of prior-guided refinement modules, each using the text mask or graphic recognition result of the low-quality SR image from the preceding layer, are designed to improve the structural clarity and semantic accuracy of the text, respectively. The following attention mechanism hence modulates two quality-enhanced images to attain a superior SR result. Extensive experiments validate that our method improves the image quality and boosts the performance of downstream tasks over five typical approaches on the benchmark. Substantial visualizations and ablation studies demonstrate the advantages of the proposed DPMN. Code is available at: https://github.com/jdfxzzy/DPMN.

        ----

        ## [428] SRoUDA: Meta Self-Training for Robust Unsupervised Domain Adaptation

        **Authors**: *Wanqing Zhu, Jia-Li Yin, Bo-Hao Chen, Ximeng Liu*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i3.25498](https://doi.org/10.1609/aaai.v37i3.25498)

        **Abstract**:

        As acquiring manual labels on data could be costly, unsupervised domain adaptation (UDA), which transfers knowledge learned from a rich-label dataset to the unlabeled target dataset, is gaining increasingly more popularity. While extensive studies have been devoted to improving the model accuracy on target domain, an important issue of model robustness is neglected. To make things worse, conventional adversarial training (AT) methods for improving model robustness are inapplicable under UDA scenario since they train models on adversarial examples that are generated by supervised loss function. In this paper, we present a new meta self-training pipeline, named SRoUDA, for improving adversarial robustness of UDA models. Based on self-training paradigm, SRoUDA starts with pre-training a source model by applying UDA baseline on source labeled data and taraget unlabeled data with a developed random masked augmentation (RMA), and then alternates between adversarial target model training on pseudo-labeled target data and fine-tuning source model by a meta step. While self-training allows the direct incorporation of AT in UDA, the meta step in SRoUDA further helps in mitigating error propagation from noisy pseudo labels. Extensive experiments on various benchmark datasets demonstrate the state-of-the-art performance of SRoUDA where it achieves significant model robustness improvement without harming clean accuracy.

        ----

        ## [429] Gradient-Based Graph Attention for Scene Text Image Super-resolution

        **Authors**: *Xiangyuan Zhu, Kehua Guo, Hui Fang, Rui Ding, Zheng Wu, Gerald Schaefer*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i3.25499](https://doi.org/10.1609/aaai.v37i3.25499)

        **Abstract**:

        Scene text image super-resolution (STISR) in the wild has been shown to be beneficial to support improved vision-based text recognition from low-resolution imagery. An intuitive way to enhance STISR performance is to explore the well-structured and repetitive layout characteristics of text and exploit these as prior knowledge to guide model convergence. In this paper, we propose a novel gradient-based graph attention method to embed patch-wise text layout contexts into image feature representations for high-resolution text image reconstruction in an implicit and elegant manner. We introduce a non-local group-wise attention module to extract text features which are then enhanced by a cascaded channel attention module and a novel gradient-based graph attention module in order to obtain more effective representations by exploring correlations of regional and local patch-wise text layout properties. Extensive experiments on the benchmark TextZoom dataset convincingly demonstrate that our method supports excellent text recognition and outperforms the current state-of-the-art in STISR. The source code is available at https://github.com/xyzhu1/TSAN.

        ----

        ## [430] RGBD1K: A Large-Scale Dataset and Benchmark for RGB-D Object Tracking

        **Authors**: *Xuefeng Zhu, Tianyang Xu, Zhangyong Tang, Zucheng Wu, Haodong Liu, Xiao Yang, Xiao-Jun Wu, Josef Kittler*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i3.25500](https://doi.org/10.1609/aaai.v37i3.25500)

        **Abstract**:

        RGB-D object tracking has attracted considerable attention recently, achieving promising performance thanks to the symbiosis between visual and depth channels. However, given a limited amount of annotated RGB-D tracking data, most state-of-the-art RGB-D trackers are simple extensions of high-performance RGB-only trackers, without fully exploiting the underlying potential of the depth channel in the offline training stage. To address the dataset deficiency issue, a new RGB-D dataset named RGBD1K is released in this paper. The RGBD1K contains 1,050 sequences with about 2.5M frames in total. To demonstrate the benefits of training on a larger RGB-D data set in general, and RGBD1K in particular, we develop a transformer-based RGB-D tracker, named SPT, as a baseline for future visual object tracking studies using the new dataset. The results, of extensive experiments using the SPT tracker demonstrate the potential of the RGBD1K dataset to improve the performance of RGB-D tracking, inspiring future developments of effective tracker designs. The dataset and codes will be available on the project homepage: https://github.com/xuefeng-zhu5/RGBD1K.

        ----

        ## [431] Learn More for Food Recognition via Progressive Self-Distillation

        **Authors**: *Yaohui Zhu, Linhu Liu, Jiang Tian*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i3.25501](https://doi.org/10.1609/aaai.v37i3.25501)

        **Abstract**:

        Food recognition has a wide range of applications, such as health-aware recommendation and self-service restaurants. Most previous methods of food recognition firstly locate informative regions in some weakly-supervised manners and then aggregate their features. However, location errors of informative regions limit the effectiveness of these methods to some extent. Instead of locating multiple regions, we propose a Progressive Self-Distillation (PSD) method, which progressively enhances the ability of network to mine more details for food recognition. The training of PSD simultaneously contains multiple self-distillations, in which a teacher network and a student network share the same embedding network. Since the student network receives a modified image from its teacher network by masking some informative regions, the teacher network outputs stronger semantic representations than the student network. Guided by such teacher network with stronger semantics, the student network is encouraged to mine more useful regions from the modified image by enhancing its own ability. The ability of the teacher network is also enhanced with the shared embedding network. By using progressive training, the teacher network incrementally improves its ability to mine more discriminative regions. In inference phase, only the teacher network is used without the help of the student network. Extensive experiments on three datasets demonstrate the effectiveness of our proposed method and state-of-the-art performance.

        ----

        ## [432] Generative Image Inpainting with Segmentation Confusion Adversarial Training and Contrastive Learning

        **Authors**: *Zhiwen Zuo, Lei Zhao, Ailin Li, Zhizhong Wang, Zhanjie Zhang, Jiafu Chen, Wei Xing, Dongming Lu*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i3.25502](https://doi.org/10.1609/aaai.v37i3.25502)

        **Abstract**:

        This paper presents a new adversarial training framework for image inpainting with segmentation confusion adversarial training (SCAT) and contrastive learning. SCAT plays an adversarial game between an inpainting generator and a segmentation network, which provides pixel-level local training signals and can adapt to images with free-form holes. By combining SCAT with standard global adversarial training, the new adversarial training framework exhibits the following three advantages simultaneously: (1) the global consistency of the repaired image, (2) the local fine texture details of the repaired image, and (3) the flexibility of handling images with free-form holes. Moreover, we propose the textural and semantic contrastive learning losses to stabilize and improve our inpainting model's training by exploiting the feature representation space of the discriminator, in which the inpainting images are pulled closer to the ground truth images but pushed farther from the corrupted images. The proposed contrastive losses better guide the repaired images to move from the corrupted image data points to the real image data points in the feature representation space, resulting in more realistic completed images. We conduct extensive experiments on two benchmark datasets, demonstrating our model's effectiveness and superiority both qualitatively and quantitatively.

        ----

        ## [433] Improved Algorithms for Maximum Satisfiability and Its Special Cases

        **Authors**: *Kirill Brilliantov, Vasily Alferov, Ivan Bliznets*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25503](https://doi.org/10.1609/aaai.v37i4.25503)

        **Abstract**:

        The Maximum Satisfiability (MAXSAT) problem is an optimization version of the Satisfiability problem (SAT) in which one is given a CNF formula with n variables and needs to find the maximum number of simultaneously satisfiable clauses. Recent works achieved significant progress in proving new upper bounds on the worst-case computational complexity of MAXSAT. All these works reduce general MAXSAT to a special case of MAXSAT where each variable appears a small number of times. So, it is important to design fast algorithms for (n,k)-MAXSAT to construct an efficient exact algorithm for MAXSAT. (n,k)-MAXSAT is a special case of MAXSAT where each variable appears at most k times in the input formula. 

For the (n,3)-MAXSAT problem, we design a O*(1.1749^n) algorithm improving on the previous record running time of O*(1.191^n). For the (n,4)-MAXSAT problem, we construct a O*(1.3803^n) algorithm improving on the previous best running time of O*(1.4254^n). Using the results, we develop a O*(1.0911^L) algorithm for the MAXSAT where L is a length of the input formula which improves previous algorithm with O*(1.0927^L) running time.

        ----

        ## [434] Lifting (D)QBF Preprocessing and Solving Techniques to (D)SSAT

        **Authors**: *Che Cheng, Jie-Hong R. Jiang*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25504](https://doi.org/10.1609/aaai.v37i4.25504)

        **Abstract**:

        Dependency stochastic Boolean satisfiability (DSSAT) generalizes stochastic Boolean satisfiability (SSAT) in existential variables being Henkinized allowing their dependencies on randomized variables to be explicitly specified. It allows NEXPTIME problems of reasoning under uncertainty and partial information to be compactly encoded. To date, no decision procedure has been implemented for solving DSSAT formulas. This work provides the first such tool by converting DSSAT into SSAT with dependency elimination, similar to converting dependency quantified Boolean formula (DQBF) to quantified Boolean formula (QBF). Moreover, we extend (D)QBF preprocessing techniques and implement the first standalone (D)SSAT preprocessor. Experimental results show that solving DSSAT via dependency elimination is highly applicable and that existing SSAT solvers may benefit from preprocessing.

        ----

        ## [435] NuWLS: Improving Local Search for (Weighted) Partial MaxSAT by New Weighting Techniques

        **Authors**: *Yi Chu, Shaowei Cai, Chuan Luo*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25505](https://doi.org/10.1609/aaai.v37i4.25505)

        **Abstract**:

        Maximum Satisfiability (MaxSAT) is a prototypical constraint optimization problem, and its generalized version is the (Weighted) Partial MaxSAT problem, denoted as (W)PMS, which deals with hard and soft clauses. Considerable progress has been made on stochastic local search (SLS) algorithms for solving (W)PMS, which mainly focus on clause weighting techniques. In this work, we identify two issues of existing clause weighting techniques for (W)PMS, and propose two ideas correspondingly. First, we observe that the initial values of soft clause weights have a big effect on the performance of the SLS solver for solving (W)PMS, and propose a weight initialization method. Second, we propose a new clause weighting scheme that for the first time employs different conditions for updating hard and soft clause weights. Based on these two ideas, we develop a new SLS solver for (W)PMS named NuWLS. Through extensive experiments, NuWLS performs much better than existing SLS solvers on all 6 benchmarks from the incomplete tracks of MaxSAT Evaluations (MSEs) 2019, 2020, and 2021. In terms of the number of winning instances, NuWLS outperforms state-of-the-art SAT-based incomplete solvers on all the 6 benchmarks. More encouragingly, a hybrid solver that combines NuWLS and an SAT-based solver won all four categories in the incomplete track of the MaxSAT Evaluation 2022.

        ----

        ## [436] Separate but Equal: Equality in Belief Propagation for Single Cycle Graphs

        **Authors**: *Erel Cohen, Omer Lev, Roie Zivan*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25506](https://doi.org/10.1609/aaai.v37i4.25506)

        **Abstract**:

        Belief propagation is a widely used incomplete optimization algorithm, whose main theoretical properties hold only under the assumptions that beliefs are not equal. Nevertheless, there is much evidence that equality between beliefs does occur. A method to overcome belief equality by using unary function-nodes is assumed to resolve the problem.

We focus on Min-sum, the belief propagation version for solving constraint optimization problems. We prove that on a single cycle graph, belief equality can be avoided only when the algorithm converges to the optimal solution. In any other case, the unary function methods will not prevent equality, rendering some existing results in need of reassessment. We differentiate between belief equality, which includes equal beliefs in a single message, and assignment equality, that prevents a coherent selection of assignments to variables. We show the necessary and satisfying conditions for both.

        ----

        ## [437] Complexity of Reasoning with Cardinality Minimality Conditions

        **Authors**: *Nadia Creignou, Frédéric Olive, Johannes Schmidt*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25507](https://doi.org/10.1609/aaai.v37i4.25507)

        **Abstract**:

        Many AI-related reasoning problems are based on the problem of satisfiability of propositional formulas with some cardinality-minimality condition. While the complexity of the satisfiability problem (SAT) is well understood when considering systematically all fragments of propositional logic within Schaefer’s framework, this is not the case when such minimality condition is added. We consider the CardMinSat problem, which asks, given a formula φ and an atom x, whether x is true in some cardinality-minimal model of φ. We completely classify the computational complexity of the CardMinSat problem within Schaefer’s framework, thus paving the way for a better understanding of the tractability frontier of many AI-related reasoning problems. To this end we use advanced algebraic tools.

        ----

        ## [438] DASH: A Distributed and Parallelizable Algorithm for Size-Constrained Submodular Maximization

        **Authors**: *Tonmoy Dey, Yixin Chen, Alan Kuhnle*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25508](https://doi.org/10.1609/aaai.v37i4.25508)

        **Abstract**:

        MapReduce (MR) algorithms for maximizing monotone, submodular functions subject to a cardinality constraint (SMCC) are currently restricted to the use of the linear-adaptive (non-parallelizable) algorithm GREEDY. Low-adaptive algorithms do not satisfy the requirements of these distributed MR frameworks, thereby limiting their performance. We study the SMCC problem in a distributed setting and propose the first MR algorithms with sublinear adaptive complexity. Our algorithms, R-DASH, T-DASH and G-DASH provide 0.316 - ε, 3/8 - ε , and (1 - 1/e - ε)   approximation ratios, respectively, with nearly optimal adaptive complexity and nearly linear time complexity. Additionally, we provide a framework to increase, under some mild assumptions, the maximum permissible cardinality constraint from  O( n / ℓ^2) of prior MR algorithms to O( n / ℓ ), where n is the data size and ℓ is the number of machines; under a stronger condition on the objective function, we increase the maximum constraint value to n. Finally, we provide empirical evidence to demonstrate that our sublinear-adaptive, distributed algorithms provide orders of magnitude faster runtime compared to current state-of-the-art distributed algorithms.

        ----

        ## [439] SharpSSAT: A Witness-Generating Stochastic Boolean Satisfiability Solver

        **Authors**: *Yu-Wei Fan, Jie-Hong R. Jiang*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25509](https://doi.org/10.1609/aaai.v37i4.25509)

        **Abstract**:

        Stochastic Boolean satisfiability (SSAT) is a formalism allowing decision-making for optimization under quantitative constraints. Although SSAT solvers are under active development, existing solvers do not provide Skolem-function witnesses, which are crucial for practical applications. In this work, we develop a new witness-generating SSAT solver, SharpSSAT, which integrates techniques, including component caching, clause learning, and pure literal detection. It can generate a set of Skolem functions witnessing the attained satisfying probability of a given SSAT formula. We also equip the solver ClauSSat with witness generation capability for comparison. Experimental results show that SharpSSAT outperforms current state-of-the-art solvers and can effectively generate compact Skolem-function witnesses. The new witness-generating solver may broaden the applicability of SSAT to practical applications.

        ----

        ## [440] Submodular Maximization under the Intersection of Matroid and Knapsack Constraints

        **Authors**: *Yu-Ran Gu, Chao Bian, Chao Qian*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25510](https://doi.org/10.1609/aaai.v37i4.25510)

        **Abstract**:

        Submodular maximization arises in many applications, and has attracted a lot of research attentions from various areas such as artificial intelligence, finance and operations research. Previous studies mainly consider only one kind of constraint, while many real-world problems often involve several constraints. In this paper, we consider the problem of submodular maximization under the intersection of two commonly used constraints, i.e., k-matroid constraint and m-knapsack constraint, and propose a new algorithm SPROUT by incorporating partial enumeration into the simultaneous greedy framework. We prove that SPROUT can achieve a polynomial-time approximation guarantee better than the state-of-the-art algorithms. Then, we introduce the random enumeration and smooth techniques into SPROUT to improve its efficiency, resulting in the SPROUT++ algorithm, which can keep a similar approximation guarantee. Experiments on the applications of movie recommendation and weighted max-cut demonstrate the superiority of SPROUT++ in practice.

        ----

        ## [441] A Framework to Design Approximation Algorithms for Finding Diverse Solutions in Combinatorial Problems

        **Authors**: *Tesshu Hanaka, Masashi Kiyomi, Yasuaki Kobayashi, Yusuke Kobayashi, Kazuhiro Kurita, Yota Otachi*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25511](https://doi.org/10.1609/aaai.v37i4.25511)

        **Abstract**:

        Finding a \emph{single} best solution is the most common objective in combinatorial optimization problems. However, such a single solution may not be applicable to real-world problems as objective functions and constraints are only ``approximately'' formulated for original real-world problems. To solve this issue, finding \emph{multiple} solutions is a natural direction, and diversity of solutions is an important concept in this context. Unfortunately, finding diverse solutions is much harder than finding a single solution. To cope with the difficulty, we investigate the approximability of finding diverse solutions. As a main result, we propose a framework to design approximation algorithms for finding diverse solutions, which yields several outcomes including constant-factor approximation algorithms for finding diverse matchings in graphs and diverse common bases in two matroids and PTASes for finding diverse minimum cuts and interval schedulings.

        ----

        ## [442] An Improved Approximation Algorithm for Wage Determination and Online Task Allocation in Crowd-Sourcing

        **Authors**: *Yuya Hikima, Yasunori Akagi, Hideaki Kim, Taichi Asami*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25512](https://doi.org/10.1609/aaai.v37i4.25512)

        **Abstract**:

        Crowd-sourcing has attracted much attention due to its growing importance to society, and numerous studies have been conducted on task allocation and wage determination. Recent works have focused on optimizing task allocation and workers' wages, simultaneously. However, existing methods do not provide good solutions for real-world crowd-sourcing platforms due to the low approximation ratio or myopic problem settings. We tackle an optimization problem for wage determination and online task allocation in crowd-sourcing and propose a fast 1-1/(k+3)^(1/2)-approximation algorithm, where k is the minimum of tasks' budgets (numbers of possible assignments). This approximation ratio is greater than or equal to the existing method. The proposed method reduces the tackled problem to a non-convex multi-period continuous optimization problem by approximating the objective function. Then, the method transforms the reduced problem into a minimum convex cost flow problem, which is a well-known combinatorial optimization problem, and solves it by the capacity scaling algorithm. Synthetic experiments and simulation experiments using real crowd-sourcing data show that the proposed method solves the problem faster and outputs higher objective values than existing methods.

        ----

        ## [443] Predict+Optimize for Packing and Covering LPs with Unknown Parameters in Constraints

        **Authors**: *Xinyi Hu, Jasper C. H. Lee, Jimmy H. M. Lee*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25513](https://doi.org/10.1609/aaai.v37i4.25513)

        **Abstract**:

        Predict+Optimize is a recently proposed framework which combines machine learning and constrained optimization, tackling optimization problems that contain parameters that are unknown at solving time. The goal is to predict the unknown parameters and use the estimates to solve for an estimated optimal solution to the optimization problem. However, all prior works have focused on the case where unknown parameters appear only in the optimization objective and not the constraints, for the simple reason that if the constraints were not known exactly, the estimated optimal solution might not even be feasible under the true parameters. The contributions of this paper are two-fold. First, we propose a novel and practically relevant framework for the Predict+Optimize setting, but with unknown parameters in both the objective and the constraints. We introduce the notion of a correction function, and an additional penalty term in the loss function, modelling practical scenarios where an estimated optimal solution can be modified into a feasible solution after the true parameters are revealed, but at an additional cost. Second, we propose a corresponding algorithmic approach for our framework, which handles all packing and covering linear programs. Our approach is inspired by the prior work of Mandi and Guns, though with crucial modifications and re-derivations for our very different setting. Experimentation demonstrates the superior empirical performance of our method over classical approaches.

        ----

        ## [444] Solving Explainability Queries with Quantification: The Case of Feature Relevancy

        **Authors**: *Xuanxiang Huang, Yacine Izza, João Marques-Silva*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25514](https://doi.org/10.1609/aaai.v37i4.25514)

        **Abstract**:

        Trustable explanations of machine learning (ML) models are vital in
high-risk uses of artificial intelligence (AI). Apart from the
computation of trustable explanations, a number of explainability
queries have been identified and studied in recent work. Some of these
queries involve solving quantification problems, either in
propositional or in more expressive logics. This paper investigates
one of these quantification problems, namely the feature relevancy
problem (FRP), i.e.\ to decide whether a (possibly sensitive) feature
can occur in some explanation of a prediction. In contrast with
earlier work, that studied FRP for specific classifiers, this paper
proposes a novel algorithm for the \fprob quantification problem which
is applicable to any ML classifier that meets minor requirements.
Furthermore, the paper shows that the novel algorithm is efficient
in practice. The experimental results, obtained using random forests
(RFs) induced from well-known publicly available datasets,
demonstrate that the proposed solution outperforms existing
state-of-the-art solvers for Quantified Boolean Formulas (QBF) by
orders of magnitude. Finally, the paper also identifies a novel family
of formulas that are challenging for currently state-of-the-art QBF
solvers.

        ----

        ## [445] Second-Order Quantified Boolean Logic

        **Authors**: *Jie-Hong R. Jiang*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25515](https://doi.org/10.1609/aaai.v37i4.25515)

        **Abstract**:

        Second-order quantified Boolean formulas (SOQBFs) generalize quantified Boolean formulas (QBFs) by admitting second-order quantifiers on function variables in addition to first-order quantifiers on atomic variables. Recent endeavors establish that the complexity of SOQBF satisfiability corresponds to the exponential-time hierarchy (EXPH), similar to that of QBF satisfiability corresponding to the polynomial-time hierarchy (PH). This fact reveals the succinct expression power of SOQBFs in encoding decision problems not efficiently doable by QBFs. In this paper, we investigate the second-order quantified Boolean logic with the following main results: First, we present a procedure of quantifier elimination converting SOQBFs to QBFs and a game interpretation of SOQBF semantics. Second, we devise a sound and complete refutation-proof system for SOQBF. Third, we develop an algorithm for countermodel extraction from a refutation proof. Finally, we show potential applications of SOQBFs in system design and multi-agent planning. With these advances, we anticipate practical tools for development.

        ----

        ## [446] Learning Markov Random Fields for Combinatorial Structures via Sampling through Lovász Local Lemma

        **Authors**: *Nan Jiang, Yi Gu, Yexiang Xue*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25516](https://doi.org/10.1609/aaai.v37i4.25516)

        **Abstract**:

        Learning to generate complex combinatorial structures satisfying constraints will have transformative impacts in many application domains. However, it is beyond the capabilities of existing approaches due to the highly intractable nature of the embedded probabilistic inference. Prior works spend most of the training time learning to separate valid from invalid structures but do not learn the inductive biases of valid structures. We develop NEural Lovasz Sampler (NELSON), which embeds the sampler through Lovasz Local Lemma (LLL) as a fully differentiable neural network layer. Our NELSON-CD embeds this sampler into the contrastive divergence learning process of Markov random fields. NELSON allows us to obtain valid samples from the current model distribution. Contrastive divergence is then applied to separate these samples from those in the training set. NELSON is implemented as a fully differentiable neural net, taking advantage of the parallelism of GPUs. Experimental results on several real-world domains reveal that NELSON learns to generate 100% valid structures, while baselines either time out or cannot ensure validity. NELSON also outperforms other approaches in running time, log-likelihood, and MAP scores.

        ----

        ## [447] Fast Converging Anytime Model Counting

        **Authors**: *Yong Lai, Kuldeep S. Meel, Roland H. C. Yap*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25517](https://doi.org/10.1609/aaai.v37i4.25517)

        **Abstract**:

        Model counting is a fundamental problem which has been influential in many applications, from artificial intelligence to formal verification. Due to the intrinsic hardness of model counting, approximate techniques have been developed to solve real-world instances of model counting. This paper designs a new anytime approach called PartialKC for approximate model counting. The idea is a form of partial knowledge compilation to provide an unbiased estimate of the model count which can converge to the exact count. Our empirical analysis demonstrates that PartialKC achieves significant scalability and accuracy over prior state-of-the-art approximate counters, including satss and STS. Interestingly, the empirical results show that PartialKC reaches convergence for many instances and therefore provides exact model counting performance comparable to state-of-the-art exact counters.

        ----

        ## [448] Finding Good Partial Assignments during Restart-Based Branch and Bound Search

        **Authors**: *Hongbo Li, Jimmy H. M. Lee*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25518](https://doi.org/10.1609/aaai.v37i4.25518)

        **Abstract**:

        Restart-based Branch-and-Bound Search (BBS) is a standard algorithm for solving Constraint Optimization Problems (COPs). In this paper, we propose an approach to find good partial assignments to jumpstart search at each restart for general COPs, which are identified by comparing different best solutions found in different restart runs. We consider information extracted from historical solutions to evaluate the quality of the partial assignments. Thus the good partial assignments are dynamically updated as the current best solution evolves.  Our approach makes restart-based BBS explore different promising sub-search-spaces to find high-quality solutions.  Experiments on the MiniZinc benchmark suite show how our approach brings significant improvements to a black-box COP solver equipped with the state of the art search techniques. Our method finds better solutions and proves optimality for more instances.

        ----

        ## [449] Hybrid Learning with New Value Function for the Maximum Common Induced Subgraph Problem

        **Authors**: *Yanli Liu, Jiming Zhao, Chu-Min Li, Hua Jiang, Kun He*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25519](https://doi.org/10.1609/aaai.v37i4.25519)

        **Abstract**:

        Maximum Common Induced Subgraph (MCIS) is an important NP-hard problem with wide real-world applications. An efficient class of MCIS algorithms uses Branch-and-Bound (BnB), consisting in successively selecting vertices to match and pruning when it is discovered that a solution better than the best solution found so far does not exist. The method of selecting the vertices to match is essential for the performance of BnB. In this paper, we propose a new value function and a hybrid selection strategy used in reinforcement learning to define a new vertex selection method, and propose a new BnB algorithm, called McSplitDAL, for MCIS. Extensive experiments show that McSplitDAL significantly improves the current best BnB algorithms, McSplit+LL and McSplit+RL. An empirical analysis is also performed to illustrate why the new value function and the hybrid selection strategy are effective.

        ----

        ## [450] Self-Supervised Primal-Dual Learning for Constrained Optimization

        **Authors**: *Seonho Park, Pascal Van Hentenryck*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25520](https://doi.org/10.1609/aaai.v37i4.25520)

        **Abstract**:

        This paper studies how to train machine-learning models that directly approximate the optimal solutions of constrained optimization problems. This is an empirical risk minimization under constraints, which is challenging as training must balance optimality and feasibility conditions. Supervised learning methods often approach this challenge by training the model on a large collection of pre-solved instances. This paper takes a different route and proposes the idea of Primal-Dual Learning (PDL), a self-supervised training method that does not require a set of pre-solved instances or an optimization solver for training and inference. Instead, PDL mimics the trajectory of an Augmented Lagrangian Method (ALM) and jointly trains primal and dual neural networks. Being a primal-dual method, PDL uses instance-specific penalties of the constraint terms in the loss function used to train the primal network. Experiments show that, on a set of nonlinear optimization benchmarks, PDL typically exhibits negligible constraint violations and minor optimality gaps, and is remarkably close to the ALM optimization. PDL also demonstrated improved or similar performance in terms of the optimality gaps, constraint violations, and training times compared to existing approaches.

        ----

        ## [451] Reinforcement Learning for Branch-and-Bound Optimisation Using Retrospective Trajectories

        **Authors**: *Christopher W. F. Parsonson, Alexandre Laterre, Thomas D. Barrett*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25521](https://doi.org/10.1609/aaai.v37i4.25521)

        **Abstract**:

        Combinatorial optimisation problems framed as mixed integer linear programmes (MILPs) are ubiquitous across a range of real-world applications. The canonical branch-and-bound algorithm seeks to exactly solve MILPs by constructing a search tree of increasingly constrained sub-problems. In practice, its solving time performance is dependent on heuristics, such as the choice of the next variable to constrain ('branching'). Recently, machine learning (ML) has emerged as a promising paradigm for branching. However, prior works have struggled to apply reinforcement learning (RL), citing sparse rewards, difficult exploration, and partial observability as significant challenges. Instead, leading ML methodologies resort to approximating high quality handcrafted heuristics with imitation learning (IL), which precludes the discovery of novel policies and requires expensive data labelling. In this work, we propose retro branching; a simple yet effective approach to RL for branching. By retrospectively deconstructing the search tree into multiple paths each contained within a sub-tree, we enable the agent to learn from shorter trajectories with more predictable next states. In experiments on four combinatorial tasks, our approach enables learning-to-branch without any expert guidance or pre-training. We outperform the current state-of-the-art RL branching algorithm by 3-5x and come within 20% of the best IL method's performance on MILPs with 500 constraints and 1000 variables, with ablations verifying that our retrospectively constructed trajectories are essential to achieving these results.

        ----

        ## [452] Constraint Optimization over Semirings

        **Authors**: *Aduri Pavan, Kuldeep S. Meel, N. V. Vinodchandran, Arnab Bhattacharyya*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25522](https://doi.org/10.1609/aaai.v37i4.25522)

        **Abstract**:

        Interpretations of logical formulas over semirings (other than the Boolean semiring) have applications in various areas of computer science including logic, AI, databases, and security.  Such interpretations provide richer information beyond the truth or falsity of a statement. Examples of such semirings include Viterbi semiring, min-max or access control semiring, tropical semiring, and fuzzy semiring. 
    
    The present work investigates the complexity of constraint optimization problems over semirings. The generic optimization problem we study is the following: Given a propositional formula phi over n variable and a semiring (K,+, . ,0,1), find the maximum value over all possible interpretations of phi over K. This can be seen as a generalization of the well-known satisfiability problem (a propositional formula is satisfiable if and only if the maximum value over all interpretations/assignments over the Boolean semiring is 1).  A related problem is to find an interpretation that achieves the maximum value. In this work, we first focus on these optimization problems over the Viterbi semiring, which we call optConfVal and optConf. 
    
    We first show that for general propositional formulas in negation normal form, optConfVal and optConf are in FP^NP. We then investigate optConf when the input formula phi is represented in the conjunctive normal form.  For CNF formulae, we first derive an upper bound on the value of optConf as a function of the number of maximum satisfiable clauses. In particular, we show that if r is the maximum number of satisfiable clauses in a CNF formula with m clauses, then its optConf value is at most 1/4^(m-r). Building on this we establish that optConf for CNF formulae is hard for the complexity class FP^NP[log]. We also design polynomial-time approximation algorithms and establish an inapproximability for optConfVal. We establish similar complexity results for these optimization problems over other semirings including tropical, fuzzy, and access control semirings.

        ----

        ## [453] Generalized Confidence Constraints

        **Authors**: *Guillaume Perez, Steve Malalel, Gael Glorian, Victor Jung, Alexandre Papadopoulos, Marie Pelleau, Wijnand Suijlen, Jean-Charles Régin, Arnaud Lallouet*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25523](https://doi.org/10.1609/aaai.v37i4.25523)

        **Abstract**:

        In robust optimization, finding a solution that solely respects the constraints is not enough.
Usually, the uncertainty and unknown parameters of the model are represented by random variables.
In such conditions, a good solution is a solution robust to most-likely assignments of these random variables.
Recently, the Confidence constraint has been introduced by Mercier-Aubin et al. in order to enforce this type of robustness in constraint programming.
Unfortunately, it is restricted to a conjunction of binary inequalities
In this paper, we generalize the Confidence constraint to any constraint and propose an implementation based on Multi-valued Decision Diagrams (MDDs). 
The Confidence constraint is defined over a vector of random variables. 
For a given constraint C, and given a threshold, the Confidence constraint ensures that the probability for C to be satisfied by a sample of the random variables is greater than the threshold.
We propose to use MDDs to represent the constraints on the random variables.
MDDs are an efficient tool for representing combinatorial constraints, thanks to their exponential compression power.
Here, both random and decision variables are stored in the MDD, 
and propagation rules are proposed for removing values of decision variables
that cannot lead to robust solutions.
Furthermore, for several constraints, we show that decision variables can be omitted from the MDD because
lighter filtering algorithms are sufficient. This leads to gain an exponential factor in the MDD size.
The experimental results obtained on a chemical deliveries problem in factories –  where the chemicals consumption are uncertain – 
shows the efficiency of the proposed approach.

        ----

        ## [454] Circuit Minimization with QBF-Based Exact Synthesis

        **Authors**: *Franz-Xaver Reichl, Friedrich Slivovsky, Stefan Szeider*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25524](https://doi.org/10.1609/aaai.v37i4.25524)

        **Abstract**:

        This paper presents a rewriting method for Boolean circuits that minimizes small subcircuits with exact synthesis. Individual synthesis tasks are encoded as Quantified Boolean Formulas (QBFs) that capture the full flexibility for implementing multi-output subcircuits.
This is in contrast to SAT-based resynthesis, where "don't cares" are computed for an individual gate, and replacements are confined to the circuitry used exclusively by that gate.
An implementation of our method achieved substantial size reductions compared to state-of-the-art methods across a wide range of benchmark circuits.

        ----

        ## [455] Probabilistic Generalization of Backdoor Trees with Application to SAT

        **Authors**: *Alexander A. Semenov, Daniil Chivilikhin, Stepan Kochemazov, Ibragim Dzhiblavi*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25525](https://doi.org/10.1609/aaai.v37i4.25525)

        **Abstract**:

        The concept of Strong Backdoor Sets (SBS) for Constraint Satisfaction Problems is well known as one of the attempts to exploit structural peculiarities in hard instances. However, in practice, finding an SBS for a particular instance is often harder than solving it. Recently, a probabilistic weakened variant of the SBS was introduced: in the SBS, all subproblems must be polynomially solvable, whereas in the probabilistic SBS only a large fraction ρ of them should have this property. This new variant of backdoors called ρ-backdoors makes it possible to use the Monte Carlo method and metaheuristic optimization to find ρ-backdoors with ρ very close to 1, and relatively fast. Despite the fact that in a ρ-backdoor-based decomposition a portion of hard subproblems remain, in practice the narrowing of the search space often allows solving the problem faster with such a backdoor than without it. In this paper, we significantly improve on the concept of ρ-backdoors by extending this concept to backdoor trees: we introduce ρ-backdoor trees, show the interconnections between SBS, ρ-backdoors, and the corresponding backdoor trees, and establish some new theoretical properties of backdoor trees. In the experimental part of the paper, we show that moving from the metaheuristic search for ρ-backdoors to that of ρ-backdoor trees allows drastically reducing the time required to construct the required decompositions without compromising their quality.

        ----

        ## [456] The Expressive Power of Ad-Hoc Constraints for Modelling CSPs

        **Authors**: *Ruiwei Wang, Roland H. C. Yap*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25526](https://doi.org/10.1609/aaai.v37i4.25526)

        **Abstract**:

        Ad-hoc constraints (also called generic constraints) are important for modelling Constraint Satisfaction Problems (CSPs). Many representations have been proposed to define ad-hoc constraints, such as tables, decision diagrams, binary constraint trees, automata and context-free grammars. However, prior works mainly focus on efficient Generalized Arc Consistency (GAC) propagators of ad-hoc constraints using the representations. In this paper, we ask a more fundamental question which bears on modelling constraints in a CSP as ad-hoc constraints, how the choice of constraints and operations affect tractability. Rather than ad-hoc constraints and their GAC propagators, our focus is on their expressive power in terms of succinctness (polysize) and cost of operations/queries (polytime). We use a large set of constraint families to investigate the expressive power of 14 existing ad-hoc constraints. We show a complete map of the succinctness of the ad-hoc constraints. We also present results on the tractability of applying various operations and queries on the ad-hoc constraints. Finally, we give case studies illustrating how our results can be useful for questions in the modelling of CSPs.

        ----

        ## [457] Graphs, Constraints, and Search for the Abstraction and Reasoning Corpus

        **Authors**: *Yudong Xu, Elias B. Khalil, Scott Sanner*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25527](https://doi.org/10.1609/aaai.v37i4.25527)

        **Abstract**:

        The Abstraction and Reasoning Corpus (ARC) aims at benchmarking the performance of general artificial intelligence algorithms. The ARC's focus on broad generalization and few-shot learning has made it difficult to solve using pure machine learning. A more promising approach has been to perform program synthesis within an appropriately designed Domain Specific Language (DSL). However, these too have seen limited success. We propose Abstract Reasoning with Graph Abstractions (ARGA), a new object-centric framework that first represents images using graphs and then performs a search for a correct program in a DSL that is based on the abstracted graph space. The complexity of this combinatorial search is tamed through the use of constraint acquisition, state hashing, and Tabu search. An extensive set of experiments demonstrates the promise of ARGA in tackling some of the complicated object-centric tasks of the ARC rather efficiently, producing programs that are correct and easy to understand.

        ----

        ## [458] Eliminating the Impossible, Whatever Remains Must Be True: On Extracting and Applying Background Knowledge in the Context of Formal Explanations

        **Authors**: *Jinqiang Yu, Alexey Ignatiev, Peter J. Stuckey, Nina Narodytska, João Marques-Silva*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25528](https://doi.org/10.1609/aaai.v37i4.25528)

        **Abstract**:

        The rise of AI methods to make predictions and decisions has led to a pressing need for more explainable artificial intelligence (XAI) methods. One common approach for XAI is to produce a post-hoc explanation, explaining why a black box ML model made a certain prediction. Formal approaches to post-hoc explanations provide succinct reasons for why a prediction was made, as well as why not another prediction was made. But these approaches assume that features are independent and uniformly distributed. While this means that “why” explanations are correct, they may be longer than required. It also means the “why not” explanations may be suspect as the counterexamples they rely on may not be meaningful. In this paper, we show how one can apply background knowledge to give more succinct “why” formal explanations, that are presumably easier to interpret by humans, and give more accurate “why not” explanations.  In addition, we show how to use existing rule induction techniques to efficiently extract background information from a dataset.

        ----

        ## [459] Farsighted Probabilistic Sampling: A General Strategy for Boosting Local Search MaxSAT Solvers

        **Authors**: *Jiongzhi Zheng, Kun He, Jianrong Zhou*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25529](https://doi.org/10.1609/aaai.v37i4.25529)

        **Abstract**:

        Local search has been demonstrated as an efficient approach for two practical generalizations of the MaxSAT problem, namely Partial MaxSAT (PMS) and Weighted PMS (WPMS). In this work, we observe that most local search (W)PMS solvers usually flip a single variable per iteration. Such a mechanism may lead to relatively low-quality local optimal solutions, and may limit the diversity of search directions to escape from local optima. To address this issue, we propose a general strategy, called farsighted probabilistic sampling (FPS), to replace the single flipping mechanism so as to boost the local search (W)PMS algorithms. FPS considers the benefit of continuously flipping a pair of variables in order to find higher-quality local optimal solutions. Moreover, FPS proposes an effective approach to escape from local optima by preferring the best to flip among the best sampled single variable and the best sampled variable pair. Extensive experiments demonstrate that our proposed FPS strategy significantly improves the state-of-the-art (W)PMS solvers, and FPS has an excellent generalization capability to various local search MaxSAT solvers.

        ----

        ## [460] LANCER: A Lifetime-Aware News Recommender System

        **Authors**: *Hong-Kyun Bae, Jeewon Ahn, Dongwon Lee, Sang-Wook Kim*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25530](https://doi.org/10.1609/aaai.v37i4.25530)

        **Abstract**:

        From the observation that users reading news tend to not click outdated news, we propose the notion of 'lifetime' of news, with two hypotheses: (i) news has a shorter lifetime, compared to other types of items such as movies or e-commerce products; (ii) news only competes with other news whose lifetimes have not ended, and which has an overlapping lifetime (i.e., limited competitions). By further developing the characteristics of the lifetime of news, then we present a novel approach for news recommendation, namely, Lifetime-Aware News reCommEndeR System (LANCER) that carefully exploits the lifetime of news during training and recommendation. Using real-world news datasets (e.g., Adressa and MIND), we successfully demonstrate that state-of-the-art news recommendation models can get significantly benefited by integrating the notion of lifetime and LANCER, by up to about 40% increases in recommendation accuracy.

        ----

        ## [461] Win-Win: A Privacy-Preserving Federated Framework for Dual-Target Cross-Domain Recommendation

        **Authors**: *Gaode Chen, Xinghua Zhang, Yijun Su, Yantong Lai, Ji Xiang, Junbo Zhang, Yu Zheng*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25531](https://doi.org/10.1609/aaai.v37i4.25531)

        **Abstract**:

        Cross-domain recommendation (CDR) aims to alleviate the data sparsity by transferring knowledge from an informative source domain to the target domain, which inevitably proposes stern challenges to data privacy and transferability during the transfer process. A small amount of recent CDR works have investigated privacy protection, while they still suffer from satisfying practical requirements (e.g., limited privacy-preserving ability) and preventing the potential risk of negative transfer. To address the above challenging problems, we propose a novel and unified privacy-preserving federated framework for dual-target CDR, namely P2FCDR. We design P2FCDR as peer-to-peer federated network architecture to ensure the local data storage and privacy protection of business partners. Specifically, for the special knowledge transfer process in CDR under federated settings, we initialize an optimizable orthogonal mapping matrix to learn the embedding transformation across domains and adopt the local differential privacy technique on the transformed embedding before exchanging across domains, which provides more reliable privacy protection. Furthermore, we exploit the similarity between in-domain and cross-domain embedding, and develop a gated selecting vector to refine the information fusion for more accurate dual transfer. Extensive experiments on three real-world datasets demonstrate that P2FCDR significantly outperforms the state-of-the-art methods and effectively protects data privacy.

        ----

        ## [462] Enhanced Multi-Relationships Integration Graph Convolutional Network for Inferring Substitutable and Complementary Items

        **Authors**: *Huajie Chen, Jiyuan He, Weisheng Xu, Tao Feng, Ming Liu, Tianyu Song, Runfeng Yao, Yuanyuan Qiao*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25532](https://doi.org/10.1609/aaai.v37i4.25532)

        **Abstract**:

        Understanding the relationships between items can improve the accuracy and interpretability of recommender systems. Among these relationships, the substitute and complement relationships attract the most attention in e-commerce platforms. The substitutable items are interchangeable and might be compared with each other before purchasing, while the complementary items are used in conjunction and are usually bought together with the query item. In this paper, we focus on two issues of inferring the substitutable and complementary items: 1) how to model their mutual influence to improve the performance of downstream tasks, 2) how to further discriminate them by considering the strength of relationship for different item pairs. We propose a novel multi-task learning framework named Enhanced Multi-Relationships Integration Graph Convolutional Network (EMRIGCN). We regard the relationship inference task as a link prediction task in heterogeneous graph with different types of edges between nodes (items). To model the mutual influence between substitute and complement, EMRIGCN adopts a two-level integration module, i.e., feature and structure integration, based on experts sharing mechanism during message passing. To obtain the strength of relationship for item pairs, we build an auxiliary loss function to further increase or decrease the distances between embeddings of items with weak or strong relation in latent space. Extensive experiments on both public and industrial datasets prove that EMRIGCN significantly outperforms the state-of-the-art solutions. We also conducted A/B tests on real world recommender systems of Meituan Maicai, an online supermarket platform in China, and obtained 15.3% improvement on VBR and 15.34% improvement on RPM.

        ----

        ## [463] PaTeCon: A Pattern-Based Temporal Constraint Mining Method for Conflict Detection on Knowledge Graphs

        **Authors**: *Jianhao Chen, Junyang Ren, Wentao Ding, Yuzhong Qu*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25533](https://doi.org/10.1609/aaai.v37i4.25533)

        **Abstract**:

        Temporal facts, the facts for characterizing events that hold in specific time periods, are attracting rising attention in the knowledge graph (KG) research communities. In terms of quality management, the introduction of time restrictions brings new challenges to maintaining the temporal consistency of KGs and detecting potential temporal conflicts. Previous studies rely on manually enumerated temporal constraints to detect conflicts, which are labor-intensive and may have granularity issues. We start from the common pattern of temporal facts and constraints and propose a pattern-based temporal constraint mining method, PaTeCon. PaTeCon uses automatically determined graph patterns and their relevant statistical information over the given KG instead of human experts to generate time constraints. Specifically, PaTeCon dynamically attaches type restriction to candidate constraints according to their measuring scores. We evaluate PaTeCon on two large-scale datasets based on Wikidata and Freebase respectively, the experimental results show that pattern-based automatic constraint mining is powerful in generating valuable temporal constraints.

        ----

        ## [464] End-to-End Entity Linking with Hierarchical Reinforcement Learning

        **Authors**: *Lihan Chen, Tinghui Zhu, Jingping Liu, Jiaqing Liang, Yanghua Xiao*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25534](https://doi.org/10.1609/aaai.v37i4.25534)

        **Abstract**:

        Entity linking (EL) is the task of linking the text segments to the referring entities in the knowledge graph, typically decomposed into mention detection, and entity disambiguation. Compared to traditional methods treating the two tasks separately, recent end-to-end entity linking methods exploit the mutual dependency between mentions and entities to achieve better performance. However, existing end-to-end EL methods have problems utilizing the dependency of mentions and entities in the task. To this end, we propose to model the EL task as a hierarchical decision-making process and design a hierarchical reinforcement learning algorithm to solve the problem. We conduct extensive experiments to show that the proposed method achieves state-of-the-art performance in several EL benchmark datasets. Our code is publicly available at https://github.com/lhlclhl/he2eel.

        ----

        ## [465] Entity-Agnostic Representation Learning for Parameter-Efficient Knowledge Graph Embedding

        **Authors**: *Mingyang Chen, Wen Zhang, Zhen Yao, Yushan Zhu, Yang Gao, Jeff Z. Pan, Huajun Chen*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25535](https://doi.org/10.1609/aaai.v37i4.25535)

        **Abstract**:

        We propose an entity-agnostic representation learning method for handling the problem of inefficient parameter storage costs brought by embedding knowledge graphs. Conventional knowledge graph embedding methods map elements in a knowledge graph, including entities and relations, into continuous vector spaces by assigning them one or multiple specific embeddings  (i.e., vector representations). Thus the number of embedding parameters increases linearly as the growth of knowledge graphs. In our proposed model, Entity-Agnostic Representation Learning (EARL), we only learn the embeddings for a small set of entities and refer to them as reserved entities. To obtain the embeddings for the full set of entities, we encode their distinguishable information from their connected relations, k-nearest reserved entities, and multi-hop neighbors. We learn universal and entity-agnostic encoders for transforming distinguishable information into entity embeddings. This approach allows our proposed EARL to have a static, efficient, and lower parameter count than conventional knowledge graph embedding methods. Experimental results show that EARL uses fewer parameters and performs better on link prediction tasks than baselines, reflecting its parameter efficiency.

        ----

        ## [466] Dual Low-Rank Graph Autoencoder for Semantic and Topological Networks

        **Authors**: *Zhaoliang Chen, Zhihao Wu, Shiping Wang, Wenzhong Guo*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25536](https://doi.org/10.1609/aaai.v37i4.25536)

        **Abstract**:

        Due to the powerful capability to gather the information of neighborhood nodes, Graph Convolutional Network (GCN) has become a widely explored hotspot in recent years. As a well-established extension, Graph AutoEncoder (GAE) succeeds in mining underlying node representations via evaluating the quality of adjacency matrix reconstruction from learned features. However, limited works on GAE were devoted to leveraging both semantic and topological graphs, and they only indirectly extracted the relationships between graphs via weights shared by features. To better capture the connections between nodes from these two types of graphs, this paper proposes a graph neural network dubbed Dual Low-Rank Graph AutoEncoder (DLR-GAE), which takes both semantic and topological homophily into consideration. Differing from prior works that share common weights between GCNs, the presented DLR-GAE conducts sustained exploration of low-rank information between two distinct graphs, and reconstructs adjacency matrices from learned latent factors and embeddings. In order to obtain valid adjacency matrices that meet certain conditions, we design some surrogates and projections to restrict the learned factor matrix. We compare the proposed model with state-of-the-art methods on several datasets, which demonstrates the superior accuracy of DLR-GAE in semi-supervised classification.

        ----

        ## [467] Dynamic Multi-Behavior Sequence Modeling for Next Item Recommendation

        **Authors**: *Junsu Cho, Dongmin Hyun, Dong won Lim, Hyeon jae Cheon, Hyoung-iel Park, Hwanjo Yu*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25537](https://doi.org/10.1609/aaai.v37i4.25537)

        **Abstract**:

        Sequential Recommender Systems (SRSs) aim to predict the next item that users will consume, by modeling the user interests within their item sequences. While most existing SRSs focus on a single type of user behavior, only a few pay attention to multi-behavior sequences, although they are very common in real-world scenarios. It is challenging to effectively capture the user interests within multi-behavior sequences, because the information about user interests is entangled throughout the sequences in complex relationships. To this end, we first address the characteristics of multi-behavior sequences that should be considered in SRSs, and then propose novel methods for Dynamic Multi-behavior Sequence modeling named DyMuS, which is a light version, and DyMuS+, which is an improved version, considering the characteristics. DyMuS first encodes each behavior sequence independently, and then combines the encoded sequences using dynamic routing, which dynamically integrates information required in the final result from among many candidates, based on correlations between the sequences. DyMuS+, furthermore, applies the dynamic routing even to encoding each behavior sequence to further capture the correlations at item-level. Moreover, we release a new, large and up-to-date dataset for multi-behavior recommendation. Our experiments on DyMuS and DyMuS+ show their superiority and the significance of capturing the characteristics of multi-behavior sequences.

        ----

        ## [468] Learning Representations of Bi-level Knowledge Graphs for Reasoning beyond Link Prediction

        **Authors**: *Chanyoung Chung, Joyce Jiyoung Whang*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25538](https://doi.org/10.1609/aaai.v37i4.25538)

        **Abstract**:

        Knowledge graphs represent known facts using triplets. While existing knowledge graph embedding methods only consider the connections between entities, we propose considering the relationships between triplets. For example, let us consider two triplets T1 and T2 where T1 is (Academy_Awards, Nominates, Avatar) and T2 is (Avatar, Wins, Academy_Awards). Given these two base-level triplets, we see that T1 is a prerequisite for T2. In this paper, we define a higher-level triplet to represent a relationship between triplets, e.g.,  where PrerequisiteFor is a higher-level relation. We define a bi-level knowledge graph that consists of the base-level and the higher-level triplets. We also propose a data augmentation strategy based on the random walks on the bi-level knowledge graph to augment plausible triplets. Our model called BiVE learns embeddings by taking into account the structures of the base-level and the higher-level triplets, with additional consideration of the augmented triplets. We propose two new tasks: triplet prediction and conditional link prediction. Given a triplet T1 and a higher-level relation, the triplet prediction predicts a triplet that is likely to be connected to T1 by the higher-level relation, e.g., . The conditional link prediction predicts a missing entity in a triplet conditioned on another triplet, e.g., . Experimental results show that BiVE significantly outperforms all other methods in the two new tasks and the typical base-level link prediction in real-world bi-level knowledge graphs.

        ----

        ## [469] Lifelong Embedding Learning and Transfer for Growing Knowledge Graphs

        **Authors**: *Yuanning Cui, Yuxin Wang, Zequn Sun, Wenqiang Liu, Yiqiao Jiang, Kexin Han, Wei Hu*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25539](https://doi.org/10.1609/aaai.v37i4.25539)

        **Abstract**:

        Existing knowledge graph (KG) embedding models have primarily focused on static KGs. However, real-world KGs do not remain static, but rather evolve and grow in tandem with the development of KG applications. Consequently, new facts and previously unseen entities and relations continually emerge, necessitating an embedding model that can quickly learn and transfer new knowledge through growth. Motivated by this, we delve into an expanding field of KG embedding in this paper, i.e., lifelong KG embedding. We consider knowledge transfer and retention of the learning on growing snapshots of a KG without having to learn embeddings from scratch. The proposed model includes a masked KG autoencoder for embedding learning and update, with an embedding transfer strategy to inject the learned knowledge into the new entity and relation embeddings, and an embedding regularization method to avoid catastrophic forgetting. To investigate the impacts of different aspects of KG growth, we construct four datasets to evaluate the performance of lifelong KG embedding. Experimental results show that the proposed model outperforms the state-of-the-art inductive and lifelong embedding baselines.

        ----

        ## [470] Uniform Sequence Better: Time Interval Aware Data Augmentation for Sequential Recommendation

        **Authors**: *Yizhou Dang, Enneng Yang, Guibing Guo, Linying Jiang, Xingwei Wang, Xiaoxiao Xu, Qinghui Sun, Hong Liu*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25540](https://doi.org/10.1609/aaai.v37i4.25540)

        **Abstract**:

        Sequential recommendation is an important task to predict the next-item to access based on a sequence of interacted items. Most existing works learn user preference as the transition pattern from the previous item to the next one, ignoring the time interval between these two items. However, we observe that the time interval in a sequence may vary significantly different, and thus result in the ineffectiveness of user modeling due to the issue of preference drift. In fact, we conducted an empirical study to validate this observation, and found that a sequence with uniformly distributed time interval (denoted as uniform sequence) is more beneficial for performance improvement than that with greatly varying time interval. Therefore, we propose to augment sequence data from the perspective of time interval, which is not studied in the literature. Specifically, we design five operators (Ti-Crop, Ti-Reorder, Ti-Mask, Ti-Substitute, Ti-Insert) to transform the original non-uniform sequence to uniform sequence with the consideration of variance of time intervals. Then, we devise a control strategy to execute data augmentation on item sequences in different lengths. Finally, we implement these improvements on a state-of-the-art model CoSeRec and validate our approach on four real datasets. The experimental results show that our approach reaches significantly better performance than the other 9 competing methods.  Our implementation is available: https://github.com/KingGugu/TiCoSeRec.

        ----

        ## [471] Rule Induction in Knowledge Graphs Using Linear Programming

        **Authors**: *Sanjeeb Dash, Joao Goncalves*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25541](https://doi.org/10.1609/aaai.v37i4.25541)

        **Abstract**:

        We present a simple linear programming (LP) based method to learn compact and interpretable sets of rules encoding the facts in a knowledge graph (KG) and use these rules to solve the KG completion problem. Our LP model chooses a set of rules of bounded complexity from a list of candidate first-order logic rules and assigns weights to them. The complexity bound is enforced via explicit constraints. We combine simple rule generation heuristics with our rule selection LP to obtain predictions with accuracy comparable to state-of-the-art codes, even while generating much more compact rule sets. Furthermore, when we take as input rules generated by other codes, we often improve interpretability by reducing the number of chosen rules, while maintaining accuracy.

        ----

        ## [472] Spatio-Temporal Neural Structural Causal Models for Bike Flow Prediction

        **Authors**: *Pan Deng, Yu Zhao, Junting Liu, Xiaofeng Jia, Mulan Wang*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25542](https://doi.org/10.1609/aaai.v37i4.25542)

        **Abstract**:

        As a representative of public transportation, the fundamental issue of managing bike-sharing systems is bike flow prediction. Recent methods overemphasize the spatio-temporal correlations in the data, ignoring the effects of contextual conditions on the transportation system and the inter-regional time-varying causality. In addition, due to the disturbance of incomplete observations in the data, random contextual conditions lead to spurious correlations between data and features, making the prediction of the model ineffective in special scenarios. To overcome this issue, we propose a Spatio-temporal Neural Structure Causal Model(STNSCM) from the perspective of causality. First, we build a causal graph to describe the traffic prediction, and further analyze the causal relationship between the input data, contextual conditions, spatio-temporal states, and prediction results. Second, we propose to apply the frontdoor criterion to eliminate confounding biases in the feature extraction process. Finally, we propose a counterfactual representation reasoning module to extrapolate the spatio-temporal state under the factual scenario to future counterfactual scenarios to improve the prediction performance. Experiments on real-world datasets demonstrate the superior performance of our model, especially its resistance to fluctuations caused by the external environment. The source code and data will be released.

        ----

        ## [473] DAMix: Exploiting Deep Autoregressive Model Zoo for Improving Lossless Compression Generalization

        **Authors**: *Qishi Dong, Fengwei Zhou, Ning Kang, Chuanlong Xie, Shifeng Zhang, Jiawei Li, Heng Peng, Zhenguo Li*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25543](https://doi.org/10.1609/aaai.v37i4.25543)

        **Abstract**:

        Deep generative models have demonstrated superior performance in lossless compression on identically distributed data. However, in real-world scenarios, data to be compressed are of various distributions and usually cannot be known in advance. Thus, commercially expected neural compression must have strong Out-of-Distribution (OoD) generalization capabilities. Compared with traditional compression methods, deep learning methods have intrinsic flaws for OoD generalization. In this work, we make the attempt to tackle this challenge via exploiting a zoo of Deep Autoregressive models (DAMix). We build a model zoo consisting of autoregressive models trained on data from diverse distributions. In the test phase, we select useful expert models by a simple model evaluation score and adaptively aggregate the predictions of selected models. By assuming the outputs from each expert model are biased in favor of their training distributions, a von Mises-Fisher based filter is proposed to recover the value of unbiased predictions that provides more accurate density estimations than a single model. We derive the posterior of unbiased predictions as well as concentration parameters in the filter, and a novel temporal Stein variational gradient descent for sequential data is proposed to adaptively update the posterior distributions. We evaluate DAMix on 22 image datasets, including in-distribution and OoD data, and demonstrate that making use of unbiased predictions has up to 45.6% improvement over the single model trained on ImageNet.

        ----

        ## [474] Soft Target-Enhanced Matching Framework for Deep Entity Matching

        **Authors**: *Wenzhou Dou, Derong Shen, Xiangmin Zhou, Tiezheng Nie, Yue Kou, Hang Cui, Ge Yu*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25544](https://doi.org/10.1609/aaai.v37i4.25544)

        **Abstract**:

        Deep Entity Matching (EM) is one of the core research topics in data integration. Typical existing works construct EM models by training deep neural networks (DNNs) based on the training samples with onehot labels. However, these sharp supervision signals of onehot labels harm the generalization of EM models, causing them to overfit the training samples and perform badly in unseen datasets. To solve this problem, we first propose that the challenge of training a well-generalized EM model lies in achieving the compromise between fitting the training samples and imposing regularization, i.e., the bias-variance tradeoff. Then, we propose a novel Soft Target-EnhAnced Matching (Steam) framework, which exploits the automatically generated soft targets as label-wise regularizers to constrain the model training. Specifically, Steam regards the EM model trained in previous iteration as a virtual teacher and takes its softened output as the extra regularizer to train the EM model in the current iteration. As such, Steam effectively calibrates the obtained EM model, achieving the bias-variance tradeoff without any additional computational cost. We conduct extensive experiments over open datasets and the results show that our proposed Steam outperforms the state-of-the-art EM approaches in terms of effectiveness and label efficiency.

        ----

        ## [475] DropMessage: Unifying Random Dropping for Graph Neural Networks

        **Authors**: *Taoran Fang, Zhiqing Xiao, Chunping Wang, Jiarong Xu, Xuan Yang, Yang Yang*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25545](https://doi.org/10.1609/aaai.v37i4.25545)

        **Abstract**:

        Graph Neural Networks (GNNs) are powerful tools for graph representation learning. Despite their rapid development, GNNs also face some challenges, such as over-fitting, over-smoothing, and non-robustness. Previous works indicate that these problems can be alleviated by random dropping methods, which integrate augmented data into models by randomly masking parts of the input. However, some open problems of random dropping on GNNs remain to be solved. First, it is challenging to find a universal method that are suitable for all cases considering the divergence of different datasets and models. Second, augmented data introduced to GNNs causes the incomplete coverage of parameters and unstable training process. Third, there is no theoretical analysis on the effectiveness of random dropping methods on GNNs. In this paper, we propose a novel random dropping method called DropMessage, which performs dropping operations directly on the propagated messages during the message-passing process. More importantly, we find that DropMessage provides a unified framework for most existing random dropping methods, based on which we give theoretical analysis of their effectiveness. Furthermore, we elaborate the superiority of DropMessage: it stabilizes the training process by reducing sample variance; it keeps information diversity from the perspective of information theory, enabling it become a theoretical upper bound of other methods. To evaluate our proposed method, we conduct experiments that aims for multiple tasks on five public datasets and two industrial datasets with various backbone models. The experimental results show that DropMessage has the advantages of both effectiveness and generalization, and can significantly alleviate the problems mentioned above. A detailed version with full appendix can be found on arXiv: https://arxiv.org/abs/2204.10037.

        ----

        ## [476] Contrastive Pre-training with Adversarial Perturbations for Check-In Sequence Representation Learning

        **Authors**: *Letian Gong, Youfang Lin, Shengnan Guo, Yan Lin, Tianyi Wang, Erwen Zheng, Zeyu Zhou, Huaiyu Wan*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25546](https://doi.org/10.1609/aaai.v37i4.25546)

        **Abstract**:

        A core step of mining human mobility data is to learn accurate representations for user-generated check-in sequences. The learned representations should be able to fully describe the spatial-temporal mobility patterns of users and the high-level semantics of traveling. However, existing check-in sequence representation learning is usually implicitly achieved by end-to-end models designed for specific downstream tasks, resulting in unsatisfactory generalizable abilities and poor performance. Besides, although the sequence representation learning models that follow the contrastive learning pre-training paradigm have achieved breakthroughs in many fields like NLP, they fail to simultaneously consider the unique spatial-temporal characteristics of check-in sequences and need manual adjustments on the data augmentation strategies. So, directly applying them to check-in sequences cannot yield a meaningful pretext task. To this end, in this paper we propose a contrastive pre-training model with adversarial perturbations for check-in sequence representation learning (CACSR). Firstly, we design a novel spatial-temporal augmentation block for disturbing the spatial-temporal features of check-in sequences in the latent space to relieve the stress of designing manual data augmentation strategies. Secondly, to construct an effective contrastive pretext task, we generate “hard” positive and negative pairs for the check-in sequence by adversarial training. These two designs encourage the model to capture the high-level spatial-temporal patterns and semantics of check-in sequences while ignoring the noisy and unimportant details. We demonstrate the effectiveness and versatility of CACSR on two kinds of downstream tasks using three real-world datasets. The results show that our model outperforms both the state-of-the-art pre-training methods and the end-to-end models.

        ----

        ## [477] MA-GCL: Model Augmentation Tricks for Graph Contrastive Learning

        **Authors**: *Xumeng Gong, Cheng Yang, Chuan Shi*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25547](https://doi.org/10.1609/aaai.v37i4.25547)

        **Abstract**:

        Contrastive learning (CL), which can extract the information shared between different contrastive views, has become a popular paradigm for vision representation learning. Inspired by the success in computer vision, recent work introduces CL into graph modeling, dubbed as graph contrastive learning (GCL). However, generating contrastive views in graphs is more challenging than that in images, since we have little prior knowledge on how to significantly augment a graph without changing its labels. We argue that typical data augmentation techniques (e.g., edge dropping) in GCL cannot generate diverse enough contrastive views to filter out noises. Moreover, previous GCL methods employ two view encoders with exactly the same neural architecture and tied parameters, which further harms the diversity of augmented views. To address this limitation, we propose a novel paradigm named model augmented GCL (MA-GCL), which will focus on manipulating the architectures of view encoders instead of perturbing graph inputs. Specifically, we present three easy-to-implement model augmentation tricks for GCL, namely asymmetric, random and shuffling, which can respectively help alleviate high-frequency noises, enrich training instances and bring safer augmentations. All three tricks are compatible with typical data augmentations. Experimental results show that MA-GCL can achieve state-of-the-art performance on node classification benchmarks by applying the three tricks on a simple base model. Extensive studies also validate our motivation and the effectiveness of each trick. (Code, data and appendix are available at https://github.com/GXM1141/MA-GCL. )

        ----

        ## [478] Generic and Dynamic Graph Representation Learning for Crowd Flow Modeling

        **Authors**: *Liangzhe Han, Ruixing Zhang, Leilei Sun, Bowen Du, Yanjie Fu, Tongyu Zhu*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25548](https://doi.org/10.1609/aaai.v37i4.25548)

        **Abstract**:

        Many deep spatio-temporal learning methods have been proposed for crowd flow modeling in recent years. However, most of them focus on designing a spatial and temporal convolution mechanism to aggregate information from nearby nodes and historical observations for a pre-defined prediction task. Different from the existing research, this paper aims to provide a generic and dynamic representation learning method for crowd flow modeling. The main idea of our method is to maintain a continuous-time representation for each node, and update the representations of all nodes continuously according to the streaming observed data. Along this line, a particular encoder-decoder architecture is proposed, where the encoder converts the newly happened transactions into a timestamped message, and then the representations of related nodes are updated according to the generated message. The role of the decoder is to guide the representation learning process by reconstructing the observed transactions based on the most recent node representations. Moreover, a number of virtual nodes are added to discover macro-level spatial patterns and also share the representations among spatially-interacted stations. Experiments have been conducted on two real-world datasets for four popular prediction tasks in crowd flow modeling. The result demonstrates that our method could achieve better prediction performance for all the tasks than baseline methods.

        ----

        ## [479] Conditional Diffusion Based on Discrete Graph Structures for Molecular Graph Generation

        **Authors**: *Han Huang, Leilei Sun, Bowen Du, Weifeng Lv*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25549](https://doi.org/10.1609/aaai.v37i4.25549)

        **Abstract**:

        Learning the underlying distribution of molecular graphs and generating high-fidelity samples is a fundamental research problem in drug discovery and material science. However, accurately modeling distribution and rapidly generating novel molecular graphs remain crucial and challenging goals. To accomplish these goals, we propose a novel Conditional Diffusion model based on discrete Graph Structures (CDGS) for molecular graph generation. Specifically, we construct a forward graph diffusion process on both graph structures and inherent features through stochastic differential equations (SDE) and derive discrete graph structures as the condition for reverse generative processes. We present a specialized hybrid graph noise prediction model that extracts the global context and the local node-edge dependency from intermediate graph states. We further utilize ordinary differential equation (ODE) solvers for efficient graph sampling, based on the semi-linear structure of the probability flow ODE. We also combine the solvers with gradient guidance from the molecule property predictor for similarity-constrained molecule optimization. Experiments on diverse datasets validate the effectiveness of our framework. Particularly, the proposed method still generates high-quality molecular graphs in a limited number of steps.

        ----

        ## [480] SAH: Shifting-Aware Asymmetric Hashing for Reverse k Maximum Inner Product Search

        **Authors**: *Qiang Huang, Yanhao Wang, Anthony K. H. Tung*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25550](https://doi.org/10.1609/aaai.v37i4.25550)

        **Abstract**:

        This paper investigates a new yet challenging problem called Reverse k-Maximum Inner Product Search (RkMIPS). Given a query (item) vector, a set of item vectors, and a set of user vectors, the problem of RkMIPS aims to find a set of user vectors whose inner products with the query vector are one of the k largest among the query and item vectors. We propose the first subquadratic-time algorithm, i.e., Shifting-aware Asymmetric Hashing (SAH), to tackle the RkMIPS problem. To speed up the Maximum Inner Product Search (MIPS) on item vectors, we design a shifting-invariant asymmetric transformation and develop a novel sublinear-time Shifting-Aware Asymmetric Locality Sensitive Hashing (SA-ALSH) scheme. Furthermore, we devise a new blocking strategy based on the Cone-Tree to effectively prune user vectors (in a batch). We prove that SAH achieves a theoretical guarantee for solving the RMIPS problem. Experimental results on five real-world datasets show that SAH runs 4~8x faster than the state-of-the-art methods for RkMIPS while achieving F1-scores of over 90%. The code is available at https://github.com/HuangQiang/SAH.

        ----

        ## [481] Learned Distributed Image Compression with Multi-Scale Patch Matching in Feature Domain

        **Authors**: *Yujun Huang, Bin Chen, Shiyu Qin, Jiawei Li, Yaowei Wang, Tao Dai, Shu-Tao Xia*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25551](https://doi.org/10.1609/aaai.v37i4.25551)

        **Abstract**:

        Beyond achieving higher compression efficiency over classical image compression codecs, deep image compression is expected to be improved with additional side information, e.g., another image from a different perspective of the same scene. To better utilize the side information under the distributed compression scenario, the existing method only implements patch matching at the image domain to solve the parallax problem caused by the difference in viewing points. However, the patch matching at the image domain is not robust to the variance of scale, shape, and illumination caused by the different viewing angles, and can not make full use of the rich texture information of the side information image. To resolve this issue, we propose Multi-Scale Feature Domain Patch Matching (MSFDPM) to fully utilizes side information at the decoder of the distributed image compression model. Specifically, MSFDPM consists of a side information feature extractor, a multi-scale feature domain patch matching module, and a multi-scale feature fusion network. Furthermore, we reuse inter-patch correlation from the shallow layer to accelerate the patch matching of the deep layer. Finally, we find that our patch matching in a multi-scale feature domain further improves compression rate by about 20% compared with the patch matching method at image domain.

        ----

        ## [482] Constrained Market Share Maximization by Signal-Guided Optimization

        **Authors**: *Bo Hui, Yuchen Fang, Tian Xia, Sarp Aykent, Wei-Shinn Ku*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25552](https://doi.org/10.1609/aaai.v37i4.25552)

        **Abstract**:

        With the rapid development of the airline
industry, maximizing the market share with a
constrained budget is an urgent econometric problem for an airline. We investigate the problem by adjusting flight frequencies on different flight routes. Owing to the large search space of solutions and the difficulty of predicting the market, this problem is in general daunting to solve. This paper proposes a novel two-stage optimization method to address the challenges. On the higher level, we use a signal to guide the optimization process toward a constrained satisfying solution. On the lower level, we consider the consecutive itineraries in real scenarios and model the unseen correlations between routes in itineraries for market share prediction. In theory, we prove the convergence of our optimization approach. In the experiment, we empirically verify the superiority of both our prediction model and optimization approach over existing works with large-scale real-world data. Our code has been released at: https://github.com/codingAndBS/AirlineMarket.

        ----

        ## [483] T2-GNN: Graph Neural Networks for Graphs with Incomplete Features and Structure via Teacher-Student Distillation

        **Authors**: *Cuiying Huo, Di Jin, Yawen Li, Dongxiao He, Yu-Bin Yang, Lingfei Wu*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25553](https://doi.org/10.1609/aaai.v37i4.25553)

        **Abstract**:

        Graph Neural Networks (GNNs) have been a prevailing technique for tackling various analysis tasks on graph data. A key premise for the remarkable performance of GNNs relies on complete and trustworthy initial graph descriptions (i.e., node features and graph structure), which is often not satisfied since real-world graphs are often incomplete due to various unavoidable factors. In particular, GNNs face greater challenges when both node features and graph structure are incomplete at the same time. The existing methods either focus on feature completion or structure completion. They usually rely on the matching relationship between features and structure, or employ joint learning of node representation and feature (or structure) completion in the hope of achieving mutual benefit. However, recent studies confirm that the mutual interference between features and structure leads to the degradation of GNN performance. When both features and structure are incomplete, the mismatch between features and structure caused by the missing randomness exacerbates the interference between the two, which may trigger incorrect completions that negatively affect node representation. To this end, in this paper we propose a general GNN framework based on teacher-student distillation to improve the performance of GNNs on incomplete graphs, namely T2-GNN. To avoid the interference between features and structure, we separately design feature-level and structure-level teacher models to provide targeted guidance for student model (base GNNs, such as GCN) through distillation. Then we design two personalized methods to obtain well-trained feature and structure teachers. To ensure that the knowledge of the teacher model is comprehensively and effectively distilled to the student model, we further propose a dual distillation mode to enable the student to acquire as much expert knowledge as possible. Extensive experiments on eight benchmark datasets demonstrate the effectiveness and robustness of the new framework on graphs with incomplete features and structure.

        ----

        ## [484] Detecting Sources of Healthcare Associated Infections

        **Authors**: *Hankyu Jang, Andrew Fu, Jiaming Cui, Methun Kamruzzaman, B. Aditya Prakash, Anil Vullikanti, Bijaya Adhikari, Sriram V. Pemmaraju*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25554](https://doi.org/10.1609/aaai.v37i4.25554)

        **Abstract**:

        Healthcare acquired infections (HAIs) (e.g., Methicillin-resistant Staphylococcus aureus infection) have complex transmission pathways, spreading not just via direct person-to-person contacts, but also via contaminated surfaces. Prior work in mathematical epidemiology has led to a class of models – which we call load sharing models – that provide a discrete-time, stochastic formalization of HAI-spread on temporal contact networks. The focus of this paper is the source detection problem for the load sharing model. The source detection problem has been studied extensively in SEIR type models, but this prior work does not apply to load sharing models.
We show that a natural formulation of the source detection problem for the load sharing model is computationally hard, even to approximate. We then present two alternate formulations that are much more tractable. The tractability of our problems depends crucially on the submodularity of the expected number of infections as a function of the source set. Prior techniques for showing submodularity, such as the "live graph" technique are not applicable for the load sharing model and our key technical contribution is to use a more sophisticated "coupling" technique to show the submodularity result. We propose algorithms for our two problem formulations by extending existing algorithmic results from submodular optimization and combining these with an expectation propagation heuristic for the load sharing model that leads to orders-of-magnitude speedup. We present experimental results on temporal contact networks based on fine-grained EMR data from three different hospitals. Our results on synthetic outbreaks on these networks show that our algorithms outperform baselines by up to 5.97 times. Furthermore, case studies based on hospital outbreaks of Clostridioides difficile infection show that our algorithms identify clinically meaningful sources.

        ----

        ## [485] Spatio-Temporal Self-Supervised Learning for Traffic Flow Prediction

        **Authors**: *Jiahao Ji, Jingyuan Wang, Chao Huang, Junjie Wu, Boren Xu, Zhenhe Wu, Junbo Zhang, Yu Zheng*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25555](https://doi.org/10.1609/aaai.v37i4.25555)

        **Abstract**:

        Robust prediction of citywide traffic flows at different time periods plays a crucial role in intelligent transportation systems. While previous work has made great efforts to model spatio-temporal correlations, existing methods still suffer from two key limitations: i) Most models collectively predict all regions' flows without accounting for spatial heterogeneity, i.e., different regions may have skewed traffic flow distributions. ii) These models fail to capture the temporal heterogeneity induced by time-varying traffic patterns, as they typically model temporal correlations with a shared parameterized space for all time periods. To tackle these challenges, we propose a novel Spatio-Temporal Self-Supervised Learning (ST-SSL) traffic prediction framework which enhances the traffic pattern representations to be reflective of both spatial and temporal heterogeneity, with auxiliary self-supervised learning paradigms. Specifically, our ST-SSL is built over an integrated module with temporal and spatial convolutions for encoding the information across space and time. To achieve the adaptive spatio-temporal self-supervised learning, our ST-SSL first performs the adaptive augmentation over the traffic flow graph data at both attribute- and structure-levels. On top of the augmented traffic graph, two SSL auxiliary tasks are constructed to supplement the main traffic prediction task with spatial and temporal heterogeneity-aware augmentation. Experiments on four benchmark datasets demonstrate that ST-SSL consistently outperforms various state-of-the-art baselines. Since spatio-temporal heterogeneity widely exists in practical datasets, the proposed framework may also cast light on other spatial-temporal applications. Model implementation is available at https://github.com/Echo-Ji/ST-SSL.

        ----

        ## [486] PDFormer: Propagation Delay-Aware Dynamic Long-Range Transformer for Traffic Flow Prediction

        **Authors**: *Jiawei Jiang, Chengkai Han, Wayne Xin Zhao, Jingyuan Wang*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25556](https://doi.org/10.1609/aaai.v37i4.25556)

        **Abstract**:

        As a core technology of Intelligent Transportation System, traffic flow prediction has a wide range of applications. The fundamental challenge in traffic flow prediction is to effectively model the complex spatial-temporal dependencies in traffic data. Spatial-temporal Graph Neural Network (GNN) models have emerged as one of the most promising methods to solve this problem. However, GNN-based models have three major limitations for traffic prediction: i) Most methods model spatial dependencies in a static manner, which limits the ability to learn dynamic urban traffic patterns; ii) Most methods only consider short-range spatial information and are unable to capture long-range spatial dependencies; iii) These methods ignore the fact that the propagation of traffic conditions between locations has a time delay in traffic systems. To this end, we propose a novel Propagation Delay-aware dynamic long-range transFormer, namely PDFormer, for accurate traffic flow prediction. Specifically, we design a spatial self-attention module to capture the dynamic spatial dependencies. Then, two graph masking matrices are introduced to highlight spatial dependencies from short- and long-range views. Moreover, a traffic delay-aware feature transformation module is proposed to empower PDFormer with the capability of explicitly modeling the time delay of spatial information propagation. Extensive experimental results on six real-world public traffic datasets show that our method can not only achieve state-of-the-art performance but also exhibit competitive computational efficiency. Moreover, we visualize the learned spatial-temporal attention map to make our model highly interpretable.

        ----

        ## [487] Continuous Trajectory Generation Based on Two-Stage GAN

        **Authors**: *Wenjun Jiang, Wayne Xin Zhao, Jingyuan Wang, Jiawei Jiang*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25557](https://doi.org/10.1609/aaai.v37i4.25557)

        **Abstract**:

        Simulating the human mobility and generating large-scale trajectories are of great use in many real-world applications, such as urban planning, epidemic spreading analysis, and geographic privacy protect. Although many previous works have studied the problem of trajectory generation, the continuity of the generated trajectories has been neglected, which makes these methods useless for practical urban simulation scenarios. To solve this problem, we propose a novel two-stage generative adversarial framework to generate the continuous trajectory on the road network, namely TS-TrajGen, which efficiently integrates prior domain knowledge of human mobility with model-free learning paradigm. Specifically, we build the generator under the human mobility hypothesis of the A* algorithm to learn the human mobility behavior. For the discriminator, we combine the sequential reward with the mobility yaw reward to enhance the effectiveness of the generator. Finally, we propose a novel two-stage generation process to overcome the weak point of the existing stochastic generation process. Extensive experiments on two real-world datasets and two case studies demonstrate that our framework yields significant improvements over the state-of-the-art methods.

        ----

        ## [488] Let Graph Be the Go Board: Gradient-Free Node Injection Attack for Graph Neural Networks via Reinforcement Learning

        **Authors**: *Mingxuan Ju, Yujie Fan, Chuxu Zhang, Yanfang Ye*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25558](https://doi.org/10.1609/aaai.v37i4.25558)

        **Abstract**:

        Graph Neural Networks (GNNs) have drawn significant attentions over the years and been broadly applied to essential applications requiring solid robustness or vigorous security standards, such as product recommendation and user behavior modeling. Under these scenarios, exploiting GNN's vulnerabilities and further downgrading its performance become extremely incentive for adversaries. Previous attackers mainly focus on structural perturbations or node injections to the existing graphs, guided by gradients from the surrogate models. Although they deliver promising results, several limitations still exist. For the structural perturbation attack, to launch a proposed attack, adversaries need to manipulate the existing graph topology, which is impractical in most circumstances. Whereas for the node injection attack, though being more practical, current approaches require training surrogate models to simulate a white-box setting, which results in significant performance downgrade when the surrogate architecture diverges from the actual victim model. To bridge these gaps, in this paper, we study the problem of black-box node injection attack, without training a potentially misleading surrogate model. Specifically, we model the node injection attack as a Markov decision process and propose Gradient-free Graph Advantage Actor Critic, namely G2A2C, a reinforcement learning framework in the fashion of advantage actor critic. By directly querying the victim model, G2A2C learns to inject highly malicious nodes with extremely limited attacking budgets, while maintaining a similar node feature distribution. Through our comprehensive experiments over eight acknowledged benchmark datasets with different characteristics, we demonstrate the superior performance of our proposed G2A2C over the existing state-of-the-art attackers. Source code is publicly available at: https://github.com/jumxglhf/G2A2C.

        ----

        ## [489] GLCC: A General Framework for Graph-Level Clustering

        **Authors**: *Wei Ju, Yiyang Gu, Binqi Chen, Gongbo Sun, Yifang Qin, Xingyuming Liu, Xiao Luo, Ming Zhang*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25559](https://doi.org/10.1609/aaai.v37i4.25559)

        **Abstract**:

        This paper studies the problem of graph-level clustering, which is a novel yet challenging task. This problem is critical in a variety of real-world applications such as protein clustering and genome analysis in bioinformatics. Recent years have witnessed the success of deep clustering coupled with graph neural networks (GNNs). However, existing methods focus on clustering among nodes given a single graph, while exploring clustering on multiple graphs is still under-explored. In this paper, we propose a general graph-level clustering framework named Graph-Level Contrastive Clustering (GLCC) given multiple graphs. Specifically, GLCC first constructs an adaptive affinity graph to explore instance- and cluster-level contrastive learning (CL). Instance-level CL leverages graph Laplacian based contrastive loss to learn clustering-friendly representations while cluster-level CL captures discriminative cluster representations incorporating neighbor information of each sample. Moreover, we utilize neighbor-aware pseudo-labels to reward the optimization of representation learning. The two steps can be alternatively trained to collaborate and benefit each other. Experiments on a range of well-known datasets demonstrate the superiority of our proposed GLCC over competitive baselines.

        ----

        ## [490] Parameterized Algorithms for Colored Clustering

        **Authors**: *Leon Kellerhals, Tomohiro Koana, Pascal Kunz, Rolf Niedermeier*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25560](https://doi.org/10.1609/aaai.v37i4.25560)

        **Abstract**:

        In the Colored Clustering problem, one is asked to cluster edge-colored (hyper-)graphs whose colors represent interaction types. More specifically, the goal is to select as many edges as possible without choosing two edges that share an endpoint and are colored differently. Equivalently, the goal can also be described as assigning colors to the vertices in a way that fits the edge-coloring as well as possible.

As this problem is NP-hard, we build on previous work by studying its parameterized complexity. We give a  2ᴼ⁽ᵏ⁾·nᴼ⁽¹⁾-time algorithm where k is the number of edges to be selected and n the number of vertices. We also prove the existence of a problem kernel of size O(k⁵ᐟ²), resolving an open problem posed in the literature. We consider parameters that are smaller than k, the number of edges to be selected, and r, the number of edges that can be deleted. Such smaller parameters are obtained by considering the difference between k or r and some lower bound on these values. We give both algorithms and lower bounds for Colored Clustering with such parameterizations. Finally, we settle the parameterized complexity of Colored Clustering with respect to structural graph parameters by showing that it is W[1]-hard with respect to both vertex cover number and tree-cut width, but fixed-parameter tractable with respect to local feedback edge number.

        ----

        ## [491] Towards Reliable Item Sampling for Recommendation Evaluation

        **Authors**: *Dong Li, Ruoming Jin, Zhenming Liu, Bin Ren, Jing Gao, Zhi Liu*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25561](https://doi.org/10.1609/aaai.v37i4.25561)

        **Abstract**:

        Since Rendle and Krichene argued that commonly used sampling-based evaluation metrics are ``inconsistent'' with respect to the global metrics (even in expectation), there have been a few studies on the sampling-based recommender system evaluation. Existing methods try either mapping the sampling-based metrics to their global counterparts or more generally, learning the empirical rank distribution to estimate the top-K metrics. 
However, despite existing efforts, there is still a lack of rigorous theoretical understanding of the proposed metric estimators, and the basic item sampling also suffers from the ``blind spot'' issue, i.e., estimation accuracy to recover the top-K metrics when K is small can still be rather substantial. 
In this paper, we provide an in-depth investigation into these problems and make two innovative contributions. First, we propose a new item-sampling estimator that explicitly optimizes the error with respect to the ground truth, and theoretically highlights its subtle difference against prior work. Second, we propose a new adaptive sampling method that aims to deal with the ``blind spot'' problem and also demonstrate the
expectation-maximization (EM) algorithm can be generalized for such a setting. 
Our experimental results confirm our statistical analysis and the superiority of the proposed works. 
This study helps lay the theoretical foundation for adopting item sampling metrics for recommendation evaluation and provides strong evidence for making item sampling a powerful and reliable tool for recommendation evaluation.

        ----

        ## [492] Multiple Robust Learning for Recommendation

        **Authors**: *Haoxuan Li, Quanyu Dai, Yuru Li, Yan Lyu, Zhenhua Dong, Xiao-Hua Zhou, Peng Wu*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25562](https://doi.org/10.1609/aaai.v37i4.25562)

        **Abstract**:

        In recommender systems, a common problem is the presence of various biases in the collected data, which deteriorates the generalization ability of the recommendation models and leads to inaccurate predictions. Doubly robust (DR) learning has been studied in many tasks in RS, with the advantage that unbiased learning can be achieved when either a single imputation or a single propensity model is accurate. In this paper, we propose a multiple robust (MR) estimator that can take the advantage of multiple candidate imputation and propensity models to achieve unbiasedness. Specifically, the MR estimator is unbiased when any of the imputation or propensity models, or a linear combination of these models is accurate. Theoretical analysis shows that the proposed MR is an enhanced version of DR when only having a single imputation and propensity model, and has a smaller bias. Inspired by the generalization error bound of MR, we further propose a novel multiple robust learning approach with stabilization. We conduct extensive experiments on real-world and semi-synthetic datasets, which demonstrates the superiority of the proposed approach over state-of-the-art methods.

        ----

        ## [493] Anomaly Segmentation for High-Resolution Remote Sensing Images Based on Pixel Descriptors

        **Authors**: *Jingtao Li, Xinyu Wang, Hengwei Zhao, Shaoyu Wang, Yanfei Zhong*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25563](https://doi.org/10.1609/aaai.v37i4.25563)

        **Abstract**:

        Anomaly segmentation in high spatial resolution (HSR) remote sensing imagery is aimed at segmenting anomaly patterns of the earth deviating from normal patterns, which plays an important role in various Earth vision applications. However, it is a challenging task due to the complex distribution and the irregular shapes of objects, and the lack of abnormal samples. To tackle these problems, an anomaly segmentation model based on pixel descriptors (ASD) is proposed for anomaly segmentation in HSR imagery. Specifically, deep one-class classification is introduced for anomaly segmentation in the feature space with discriminative pixel descriptors. The ASD model incorporates the data argument for generating virtual abnormal samples, which can force the pixel descriptors to be compact for normal data and meanwhile to be diverse to avoid the model collapse problems when only positive samples participated in the training. In addition, the ASD introduced a multi-level and multi-scale feature extraction strategy for learning the low-level and semantic information to make the pixel descriptors feature-rich.  The proposed ASD model was validated using four HSR datasets and compared with the recent state-of-the-art models, showing its potential value in Earth vision applications.

        ----

        ## [494] Adaptive Low-Precision Training for Embeddings in Click-Through Rate Prediction

        **Authors**: *Shiwei Li, Huifeng Guo, Lu Hou, Wei Zhang, Xing Tang, Ruiming Tang, Rui Zhang, Ruixuan Li*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25564](https://doi.org/10.1609/aaai.v37i4.25564)

        **Abstract**:

        Embedding tables are usually huge in click-through rate (CTR) prediction models. To train and deploy the CTR models efficiently and economically, it is necessary to compress their embedding tables. To this end, we formulate a novel quantization training paradigm to compress the embeddings from the training stage, termed low-precision training (LPT). Also, we provide theoretical analysis on its convergence. The results show that stochastic weight quantization has a faster convergence rate and a smaller convergence error than deterministic weight quantization in LPT. Further, to reduce accuracy degradation, we propose adaptive low-precision training (ALPT) which learns the step size (i.e., the quantization resolution). Experiments on two real-world datasets confirm our analysis and show that ALPT can significantly improve the prediction accuracy, especially at extremely low bit width. For the first time in CTR models, we successfully train 8-bit embeddings without sacrificing prediction accuracy.

        ----

        ## [495] Signed Laplacian Graph Neural Networks

        **Authors**: *Yu Li, Meng Qu, Jian Tang, Yi Chang*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25565](https://doi.org/10.1609/aaai.v37i4.25565)

        **Abstract**:

        This paper studies learning meaningful node representations for signed graphs, where both positive and negative links exist. This problem has been widely studied by meticulously designing expressive signed graph neural networks, as well as capturing the structural information of the signed graph through traditional structure decomposition methods, e.g., spectral graph theory. In this paper, we propose a novel signed graph representation learning framework, called Signed Laplacian Graph Neural Network (SLGNN), which combines the advantages of both. Specifically, based on spectral graph theory and graph signal processing, we first design different low-pass and high-pass graph convolution filters to extract low-frequency and high-frequency information on positive and negative links, respectively, and then combine them into a unified message passing framework. To effectively model signed graphs, we further propose a self-gating mechanism to estimate the impacts of low-frequency and high-frequency information during message passing. We mathematically establish the relationship between the aggregation process in SLGNN and signed Laplacian regularization in signed graphs, and theoretically analyze the expressiveness of SLGNN. Experimental results demonstrate that SLGNN outperforms various competitive baselines and achieves state-of-the-art performance.

        ----

        ## [496] PPGenCDR: A Stable and Robust Framework for Privacy-Preserving Cross-Domain Recommendation

        **Authors**: *Xinting Liao, Weiming Liu, Xiaolin Zheng, Binhui Yao, Chaochao Chen*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25566](https://doi.org/10.1609/aaai.v37i4.25566)

        **Abstract**:

        Privacy-preserving cross-domain recommendation (PPCDR) refers to preserving the privacy of users when transferring the knowledge from source domain to target domain for better performance, which is vital for the long-term development of recommender systems. Existing work on cross-domain recommendation (CDR) reaches advanced and satisfying recommendation performance, but mostly neglects preserving privacy. To fill this gap, we propose a privacy-preserving generative cross-domain recommendation (PPGenCDR) framework for PPCDR. PPGenCDR includes two main modules, i.e., stable privacy-preserving generator module, and robust cross-domain recommendation module. Specifically, the former isolates data from different domains with a generative adversarial network (GAN) based model, which stably estimates the distribution of private data in the source domain with ́Renyi differential privacy (RDP) technique. Then the latter aims to robustly leverage the perturbed but effective knowledge from the source domain with the raw data in target domain to improve recommendation performance. Three key modules, i.e., (1) selective privacy preserver, (2) GAN stabilizer, and (3) robustness conductor, guarantee the cost-effective trade-off between utility and privacy, the stability of GAN when using RDP, and the robustness of leveraging transferable knowledge accordingly. The extensive empirical studies on Douban and Amazon datasets demonstrate that PPGenCDR significantly outperforms the state-of-the-art recommendation models while preserving privacy.

        ----

        ## [497] COLA: Improving Conversational Recommender Systems by Collaborative Augmentation

        **Authors**: *Dongding Lin, Jian Wang, Wenjie Li*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25567](https://doi.org/10.1609/aaai.v37i4.25567)

        **Abstract**:

        Conversational recommender systems (CRS) aim to employ natural language conversations to suggest suitable products to users. Understanding user preferences for prospective items and learning efficient item representations are crucial for CRS. Despite various attempts, earlier studies mostly learned item representations based on individual conversations, ignoring item popularity embodied among all others. Besides, they still need support in efficiently capturing user preferences since the information reflected in a single conversation is limited. Inspired by collaborative filtering, we propose a collaborative augmentation (COLA) method to simultaneously improve both item representation learning and user preference modeling to address these issues. We construct an interactive user-item graph from all conversations, which augments item representations with user-aware information, i.e., item popularity. To improve user preference modeling, we retrieve similar conversations from the training corpus, where the involved items and attributes that reflect the user's potential interests are used to augment the user representation through gate control. Extensive experiments on two benchmark datasets demonstrate the effectiveness of our method. Our code and data are available at https://github.com/DongdingLin/COLA.

        ----

        ## [498] Scalable and Effective Conductance-Based Graph Clustering

        **Authors**: *Longlong Lin, Ronghua Li, Tao Jia*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25568](https://doi.org/10.1609/aaai.v37i4.25568)

        **Abstract**:

        Conductance-based graph clustering has been recognized as a fundamental operator in numerous graph analysis applications. Despite the significant success of conductance-based graph clustering, existing algorithms are either hard to obtain satisfactory clustering qualities, or have high time and space complexity to achieve provable clustering qualities. To overcome these limitations, we devise a powerful peeling-based graph clustering framework PCon. We show that many existing solutions can be reduced to our framework. Namely, they first define a score function for each vertex, then iteratively remove the vertex with the smallest score.  Finally, they output the result with the smallest conductance during the peeling process. Based on our framework, we propose two novel algorithms PCon_core and PCon_de with linear time and space complexity, which can efficiently and effectively identify clusters from massive graphs with more than a few billion edges. Surprisingly, we prove that PCon_de can identify clusters with near-constant approximation ratio, resulting in an important theoretical improvement over the well-known quadratic Cheeger bound. Empirical results on real-life and synthetic datasets show that our algorithms can achieve 5~42 times speedup with a high clustering accuracy, while using 1.4~7.8 times less memory than the baseline algorithms.

        ----

        ## [499] Multi-Domain Generalized Graph Meta Learning

        **Authors**: *Mingkai Lin, Wenzhong Li, Ding Li, Yizhou Chen, Guohao Li, Sanglu Lu*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25569](https://doi.org/10.1609/aaai.v37i4.25569)

        **Abstract**:

        Graph meta learning aims to learn historical knowledge from training graph neural networks (GNNs) models and adapt it to downstream learning tasks in a target graph, which has drawn increasing attention due to its ability of knowledge transfer and fast adaptation. While existing graph meta learning approaches assume the learning tasks are from the same graph domain but lack the solution for multi-domain adaptation. In this paper, we address the multi-domain generalized graph meta learning problem, which is challenging due to non-Euclidean data, inequivalent feature spaces, and heterogeneous distributions. To this end, we propose a novel solution called MD-Gram for multi-domain graph generalization. It introduces an empirical graph generalization method that uses empirical vectors to form a unified expression of non-Euclidean graph data. Then it proposes a multi-domain graphs transformation approach to transform the learning tasks from multiple source-domain graphs with inequivalent feature spaces into a common domain, where graph meta learning is conducted to learn generalized knowledge. It further adopts a domain-specific GNN enhancement method to learn a customized GNN model to achieve fast adaptation in the unseen target domain. Extensive experiments based on four real-world graph domain datasets show that the proposed method significantly outperforms the state-of-the-art in multi-domain graph meta learning tasks.

        ----

        ## [500] IterDE: An Iterative Knowledge Distillation Framework for Knowledge Graph Embeddings

        **Authors**: *Jiajun Liu, Peng Wang, Ziyu Shang, Chenxiao Wu*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25570](https://doi.org/10.1609/aaai.v37i4.25570)

        **Abstract**:

        Knowledge distillation for knowledge graph embedding (KGE) aims to reduce the KGE model size to address the challenges of storage limitations and knowledge reasoning efficiency. However, current work still suffers from the performance drops when compressing a high-dimensional original KGE model to a low-dimensional distillation KGE model. Moreover, most work focuses on the reduction of inference time but ignores the time-consuming training process of distilling KGE models. In this paper, we propose IterDE, a novel knowledge distillation framework for KGEs. First, IterDE introduces an iterative distillation way and enables a KGE model to alternately be a student model and a teacher model during the iterative distillation process. Consequently, knowledge can be transferred in a smooth manner between high-dimensional teacher models and low-dimensional student models, while preserving good KGE performances. Furthermore, in order to optimize the training process, we consider that different optimization objects between hard label loss and soft label loss can affect the efficiency of training, and then we propose a soft-label weighting dynamic adjustment mechanism that can balance the inconsistency of optimization direction between hard and soft label loss by gradually increasing the weighting of soft label loss. Our experimental results demonstrate that IterDE achieves a new state-of-the-art distillation performance for KGEs compared to strong baselines on the link prediction task. Significantly, IterDE can reduce the training time by 50% on average. Finally, more exploratory experiments show that the soft-label weighting dynamic adjustment mechanism and more fine-grained iterations can improve distillation performance.

        ----

        ## [501] Learning by Applying: A General Framework for Mathematical Reasoning via Enhancing Explicit Knowledge Learning

        **Authors**: *Jiayu Liu, Zhenya Huang, ChengXiang Zhai, Qi Liu*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25571](https://doi.org/10.1609/aaai.v37i4.25571)

        **Abstract**:

        Mathematical reasoning is one of the crucial abilities of general artificial intelligence, which requires machines to master mathematical logic and knowledge from solving problems. However, existing approaches are not transparent (thus not interpretable) in terms of what knowledge has been learned and applied in the reasoning process. In this paper, we propose a general Learning by Applying (LeAp) framework to enhance existing models (backbones) in a principled way by explicit knowledge learning. In LeAp, we perform knowledge learning in a novel problem-knowledge-expression paradigm, with a Knowledge Encoder to acquire knowledge from problem data and a Knowledge Decoder to apply knowledge for expression reasoning. The learned mathematical knowledge, including word-word relations and word-operator relations, forms an explicit knowledge graph, which bridges the knowledge “learning” and “applying” organically. Moreover, for problem solving, we design a semantics-enhanced module and a reasoning-enhanced module that apply knowledge to improve the problem comprehension and symbol reasoning abilities of any backbone, respectively. We theoretically prove the superiority of LeAp's autonomous learning mechanism. Experiments on three real-world datasets show that LeAp improves all backbones' performances, learns accurate knowledge, and achieves a more interpretable reasoning process.

        ----

        ## [502] Low-Resource Personal Attribute Prediction from Conversations

        **Authors**: *Yinan Liu, Hu Chen, Wei Shen, Jiaoyan Chen*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25572](https://doi.org/10.1609/aaai.v37i4.25572)

        **Abstract**:

        Personal knowledge bases (PKBs) are crucial for a broad range of applications such as personalized recommendation and Web-based chatbots. A critical challenge to build PKBs is extracting personal attribute knowledge from users' conversation data. Given some users of a conversational system, a personal attribute and these users' utterances, our goal is to predict the ranking of the given personal attribute values for each user. Previous studies often rely on a relative number of resources such as labeled utterances and external data, yet the attribute knowledge embedded in unlabeled utterances is underutilized and their performance of predicting some difficult personal attributes is still unsatisfactory. In addition, it is found that some text classification methods could be employed to resolve this task directly. However, they also perform not well over those difficult personal attributes. In this paper, we propose a novel framework PEARL to predict personal attributes from conversations by leveraging the abundant personal attribute knowledge from utterances under a low-resource setting in which no labeled utterances or external data are utilized. PEARL combines the biterm semantic information with the word co-occurrence information seamlessly via employing the updated prior attribute knowledge to refine the biterm topic model's Gibbs sampling process in an iterative manner. The extensive experimental results show that PEARL outperforms all the baseline methods not only on the task of personal attribute prediction from conversations over two data sets, but also on the more general weakly supervised text classification task over one data set.

        ----

        ## [503] Beyond Smoothing: Unsupervised Graph Representation Learning with Edge Heterophily Discriminating

        **Authors**: *Yixin Liu, Yizhen Zheng, Daokun Zhang, Vincent C. S. Lee, Shirui Pan*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25573](https://doi.org/10.1609/aaai.v37i4.25573)

        **Abstract**:

        Unsupervised graph representation learning (UGRL) has drawn increasing research attention and achieved promising results in several graph analytic tasks. Relying on the homophily assumption, existing UGRL methods tend to smooth the learned node representations along all edges, ignoring the existence of heterophilic edges that connect nodes with distinct attributes. As a result, current methods are hard to generalize to heterophilic graphs where dissimilar nodes are widely connected, and also vulnerable to adversarial attacks. To address this issue, we propose a novel unsupervised Graph Representation learning method with Edge hEterophily discriminaTing (GREET) which learns representations by discriminating and leveraging homophilic edges and heterophilic edges. To distinguish two types of edges, we build an edge discriminator that infers edge homophily/heterophily from feature and structure information. We train the edge discriminator in an unsupervised way through minimizing the crafted pivot-anchored ranking loss, with randomly sampled node pairs acting as pivots. Node representations are learned through contrasting the dual-channel encodings obtained from the discriminated homophilic and heterophilic edges. With an effective interplaying scheme, edge discriminating and representation learning can mutually boost each other during the training phase. We conducted extensive experiments on 14 benchmark datasets and multiple learning scenarios to demonstrate the superiority of GREET.

        ----

        ## [504] On Generalized Degree Fairness in Graph Neural Networks

        **Authors**: *Zemin Liu, Trung-Kien Nguyen, Yuan Fang*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25574](https://doi.org/10.1609/aaai.v37i4.25574)

        **Abstract**:

        Conventional graph neural networks (GNNs) are often confronted with fairness issues that may stem from their input, including node attributes and neighbors surrounding a node. While several recent approaches have been proposed to eliminate the bias rooted in sensitive attributes, they ignore the other key input of GNNs, namely the neighbors of a node, which can introduce bias since GNNs hinge on neighborhood structures to generate node representations. In particular, the varying neighborhood structures across nodes, manifesting themselves in drastically different node degrees, give rise to the diverse behaviors of nodes and biased outcomes. In this paper, we first define and generalize the degree bias using a generalized definition of node degree as a manifestation and quantification of different multi-hop structures around different nodes. To address the bias in the context of node classification, we propose a novel GNN framework called Generalized Degree Fairness-centric Graph Neural Network (DegFairGNN). Specifically, in each GNN layer, we employ a learnable debiasing function to generate debiasing contexts, which modulate the layer-wise neighborhood aggregation to eliminate the degree bias originating from the diverse degrees among nodes. Extensive experiments on three benchmark datasets demonstrate the effectiveness of our model on both accuracy and fairness metrics.

        ----

        ## [505] Time Series Contrastive Learning with Information-Aware Augmentations

        **Authors**: *Dongsheng Luo, Wei Cheng, Yingheng Wang, Dongkuan Xu, Jingchao Ni, Wenchao Yu, Xuchao Zhang, Yanchi Liu, Yuncong Chen, Haifeng Chen, Xiang Zhang*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25575](https://doi.org/10.1609/aaai.v37i4.25575)

        **Abstract**:

        Various contrastive learning approaches have been proposed in recent years and achieve significant empirical success. While effective and prevalent, contrastive learning has been less explored for time series data. A key component of contrastive learning is to select appropriate augmentations imposing some priors to construct feasible positive samples, such that an encoder can be trained to learn robust and discriminative representations. Unlike image and language domains where "desired'' augmented samples can be generated with the rule of thumb guided by prefabricated human priors, the ad-hoc manual selection of time series augmentations is hindered by their diverse and human-unrecognizable temporal structures. How to find the desired augmentations of time series data that are meaningful for given contrastive learning tasks and datasets remains an open question. In this work, we address the problem by encouraging both high fidelity and variety based on information theory. A theoretical analysis leads to the criteria for selecting feasible data augmentations. On top of that, we propose a new contrastive learning approach with information-aware augmentations, InfoTS, that adaptively selects optimal augmentations for time series representation learning. Experiments on various datasets show highly competitive performance with up to a 12.0% reduction in MSE on forecasting tasks and up to 3.7% relative improvement in accuracy on classification tasks over the leading baselines.

        ----

        ## [506] NQE: N-ary Query Embedding for Complex Query Answering over Hyper-Relational Knowledge Graphs

        **Authors**: *Haoran Luo, Haihong E, Yuhao Yang, Gengxian Zhou, Yikai Guo, Tianyu Yao, Zichen Tang, Xueyuan Lin, Kaiyang Wan*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25576](https://doi.org/10.1609/aaai.v37i4.25576)

        **Abstract**:

        Complex query answering (CQA) is an essential task for multi-hop and logical reasoning on knowledge graphs (KGs). Currently, most approaches are limited to queries among binary relational facts and pay less attention to n-ary facts (n≥2) containing more than two entities, which are more prevalent in the real world. Moreover, previous CQA methods can only make predictions for a few given types of queries and cannot be flexibly extended to more complex logical queries, which significantly limits their applications. To overcome these challenges, in this work, we propose a novel N-ary Query Embedding (NQE) model for CQA over hyper-relational knowledge graphs (HKGs), which include massive n-ary facts. The NQE utilizes a dual-heterogeneous Transformer encoder and fuzzy logic theory to satisfy all n-ary FOL queries, including existential quantifiers (∃), conjunction (∧), disjunction (∨), and negation (¬). We also propose a parallel processing algorithm that can train or predict arbitrary n-ary FOL queries in a single batch, regardless of the kind of each query, with good flexibility and extensibility. In addition, we generate a new CQA dataset WD50K-NFOL, including diverse n-ary FOL queries over WD50K. Experimental results on WD50K-NFOL and other standard CQA datasets show that NQE is the state-of-the-art CQA method over HKGs with good generalization capability. Our code and dataset are publicly available.

        ----

        ## [507] FinalMLP: An Enhanced Two-Stream MLP Model for CTR Prediction

        **Authors**: *Kelong Mao, Jieming Zhu, Liangcai Su, Guohao Cai, Yuru Li, Zhenhua Dong*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25577](https://doi.org/10.1609/aaai.v37i4.25577)

        **Abstract**:

        Click-through rate (CTR) prediction is one of the fundamental tasks in online advertising and recommendation. Multi-layer perceptron (MLP) serves as a core component in many deep CTR prediction models, but it has been widely shown that applying a vanilla MLP network alone is ineffective in learning complex feature interactions. As such, many two-stream models (e.g., Wide&Deep, DeepFM, and DCN) have recently been proposed, aiming to integrate two parallel sub-networks to learn feature interactions from two different views for enhanced CTR prediction. In addition to one MLP stream that learns feature interactions implicitly, most of the existing research focuses on designing another stream to complement the MLP stream with explicitly enhanced feature interactions. Instead, this paper presents a simple two-stream feature interaction model, namely FinalMLP, which employs only MLPs in both streams yet achieves surprisingly strong performance. In contrast to sophisticated network design in each stream, our work enhances CTR modeling through a feature selection module, which produces differentiated feature inputs to two streams, and a group-wise bilinear fusion module, which effectively captures stream-level interactions across two streams. We show that FinalMLP achieves competitive or even better performance against many existing two-stream CTR models on four open benchmark datasets and also brings significant CTR improvements during an online A/B test in our industrial news recommender system. We envision that the simple yet effective FinalMLP model could serve as a new strong baseline for future development of two-stream CTR models. Our source code will be available at MindSpore/models and FuxiCTR/model_zoo.

        ----

        ## [508] GMDNet: A Graph-Based Mixture Density Network for Estimating Packages' Multimodal Travel Time Distribution

        **Authors**: *Xiaowei Mao, Huaiyu Wan, Haomin Wen, Fan Wu, Jianbin Zheng, Yuting Qiang, Shengnan Guo, Lixia Wu, Haoyuan Hu, Youfang Lin*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25578](https://doi.org/10.1609/aaai.v37i4.25578)

        **Abstract**:

        In the logistics network, accurately estimating packages' Travel Time Distribution (TTD) given the routes greatly benefits both consumers and platforms. Although recent works perform well in predicting an expected time or a time distribution in a road network, they could not be well applied to estimate TTD in logistics networks. Because TTD prediction in the logistics network requires modeling packages' multimodal TTD (MTTD, i.e., there can be more than one likely output with a given input) while leveraging the complex correlations in the logistics network. To this end, this work opens appealing research opportunities in studying MTTD learning conditioned on graph-structure data by investigating packages' travel time distribution in the logistics network. We propose a Graph-based Mixture Density Network, named GMDNet, which takes the benefits of both graph neural network and mixture density network for estimating MTTD conditioned on graph-structure data (i.e., the logistics network). Furthermore, we adopt the Expectation-Maximization (EM) framework in the training process to guarantee local convergence and thus obtain more stable results than gradient descent. Extensive experiments on two real-world datasets demonstrate the superiority of our proposed model.

        ----

        ## [509] Logic and Commonsense-Guided Temporal Knowledge Graph Completion

        **Authors**: *Guanglin Niu, Bo Li*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25579](https://doi.org/10.1609/aaai.v37i4.25579)

        **Abstract**:

        A temporal knowledge graph (TKG) stores the events derived from the data involving time. Predicting events is extremely challenging due to the time-sensitive property of events. Besides, the previous TKG completion (TKGC) approaches cannot represent both the timeliness and the causality properties of events, simultaneously. To address these challenges, we propose a Logic and Commonsense-Guided Embedding model (LCGE) to jointly learn the time-sensitive representation involving timeliness and causality of events, together with the time-independent representation of events from the perspective of commonsense. Specifically, we design a temporal rule learning algorithm to construct a rule-guided predicate embedding regularization strategy for learning the causality among events. Furthermore, we could accurately evaluate the plausibility of events via auxiliary commonsense knowledge. The experimental results of TKGC task illustrate the significant performance improvements of our model compared with the existing approaches. More interestingly, our model is able to provide the explainability of the predicted results in the view of causal inference. The appendix, source code and datasets of this paper are available at https://github.com/ngl567/LCGE.

        ----

        ## [510] Graph Structure Learning on User Mobility Data for Social Relationship Inference

        **Authors**: *Guangming Qin, Lexue Song, Yanwei Yu, Chao Huang, Wenzhe Jia, Yuan Cao, Junyu Dong*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25580](https://doi.org/10.1609/aaai.v37i4.25580)

        **Abstract**:

        With the prevalence of smart mobile devices and location-based services, uncovering social relationships from human mobility data is of great value in real-world spatio-temporal applications ranging from friend recommendation, advertisement targeting to transportation scheduling. While a handful of sophisticated graph embedding techniques are developed for social relationship inference, they are significantly limited to the sparse and noisy nature of user mobility data, as they all ignore the essential problem of the existence of a large amount of noisy data unrelated to social activities in such mobility data. In this work, we present Social Relationship Inference Network (SRINet), a novel Graph Neural Network (GNN) framework, to improve inference performance by learning to remove noisy data. Specifically, we first construct a multiplex user meeting graph to model the spatial-temporal interactions among users in different semantic contexts. Our proposed SRINet tactfully combines the representation learning ability of Graph Convolutional Networks (GCNs) with the power of removing noisy edges of graph structure learning, which can learn effective user embeddings on the multiplex user meeting graph in a semi-supervised manner. Extensive experiments on three real-world datasets demonstrate the superiority of SRINet against state-of-the-art techniques in inferring social relationships from user mobility data. The source code of our method is available at https://github.com/qinguangming1999/SRINet.

        ----

        ## [511] Online Random Feature Forests for Learning in Varying Feature Spaces

        **Authors**: *Christian Schreckenberger, Yi He, Stefan Lüdtke, Christian Bartelt, Heiner Stuckenschmidt*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25581](https://doi.org/10.1609/aaai.v37i4.25581)

        **Abstract**:

        In this paper, we propose a new online learning algorithm tailored for data streams described by varying feature spaces (VFS), wherein new features constantly emerge and old features may stop to be observed over various time spans. Our proposed algorithm, named Online Random Feature Forests for Feature space Variabilities (ORF3V), provides a strategy to respect such feature dynamics by generating, updating, pruning, as well as online re-weighing an ensemble of what we call feature forests, which are generated and updated based on a compressed and storage efficient representation for each observed feature. We benchmark our algorithm on 12 datasets, including one novel real-world dataset of government COVID-19 responses collected through a crowd-sensing program in Spain. The empirical results substantiate the viability and effectiveness of our ORF3V algorithm and its superior accuracy performance over the state-of-the-art rival models.

        ----

        ## [512] Scaling Law for Recommendation Models: Towards General-Purpose User Representations

        **Authors**: *Kyuyong Shin, Hanock Kwak, Su Young Kim, Max Nihlén Ramström, Jisu Jeong, Jung-Woo Ha, Kyung-Min Kim*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25582](https://doi.org/10.1609/aaai.v37i4.25582)

        **Abstract**:

        Recent advancement of large-scale pretrained models such as BERT, GPT-3, CLIP, and Gopher, has shown astonishing achievements across various task domains. Unlike vision recognition and language models, studies on general-purpose user representation at scale still remain underexplored. Here we explore the possibility of general-purpose user representation learning by training a universal user encoder at large scales. We demonstrate that the scaling law is present in user representation learning areas, where the training error scales as a power-law with the amount of computation. Our Contrastive Learning User Encoder (CLUE), optimizes task-agnostic objectives, and the resulting user embeddings stretch our expectation of what is possible to do in various downstream tasks. CLUE also shows great transferability to other domains and companies, as performances on an online experiment shows significant improvements in Click-Through-Rate (CTR).  Furthermore, we also investigate how the model performance is influenced by the scale factors, such as training data size, model capacity, sequence length, and batch size. Finally, we discuss the broader impacts of CLUE in general.

        ----

        ## [513] Cross-Domain Adaptative Learning for Online Advertisement Customer Lifetime Value Prediction

        **Authors**: *Hongzu Su, Zhekai Du, Jingjing Li, Lei Zhu, Ke Lu*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25583](https://doi.org/10.1609/aaai.v37i4.25583)

        **Abstract**:

        Accurate estimation of customer lifetime value (LTV), which reflects the potential consumption of a user over a period of time, is crucial for the revenue management of online advertising platforms. However, predicting LTV in real-world applications is not an easy task since the user consumption data is usually insufficient within a specific domain. To tackle this problem, we propose a novel cross-domain adaptative framework (CDAF) to leverage consumption data from different domains. The proposed method is able to simultaneously mitigate the data scarce problem and the distribution gap problem caused by data from different domains. To be specific, our method firstly learns a LTV prediction model from a different but related platform with sufficient data provision. Subsequently, we exploit domain-invariant information to mitigate data scarce problem by minimizing the Wasserstein discrepancy between the encoded user representations of two domains. In addition, we design a dual-predictor schema which not only enhances domain-invariant information in the semantic space but also preserves domain-specific information for accurate target prediction. The proposed framework is evaluated on five datasets collected from real historical data on the advertising platform of Tencent Games. Experimental results verify that the proposed framework is able to significantly improve the LTV prediction performance on this platform. For instance, our method can boost DCNv2 with the improvement of 13.7% in terms of AUC on dataset G2. Code: https://github.com/TL-UESTC/CDAF.

        ----

        ## [514] Self-Supervised Interest Transfer Network via Prototypical Contrastive Learning for Recommendation

        **Authors**: *Guoqiang Sun, Yibin Shen, Sijin Zhou, Xiang Chen, Hongyan Liu, Chunming Wu, Chenyi Lei, Xianhui Wei, Fei Fang*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25584](https://doi.org/10.1609/aaai.v37i4.25584)

        **Abstract**:

        Cross-domain recommendation has attracted increasing attention from industry and academia recently. However, most existing methods do not exploit the interest invariance between domains, which would yield sub-optimal solutions. In this paper, we propose a cross-domain recommendation method: Self-supervised Interest Transfer Network (SITN), which can effectively transfer invariant knowledge between domains via prototypical contrastive learning. Specifically, we perform two levels of cross-domain contrastive learning: 1) instance-to-instance contrastive learning, 2) instance-to-cluster contrastive learning. Not only that, we also take into account users' multi-granularity and multi-view interests. With this paradigm, SITN can explicitly learn the invariant knowledge of interest clusters between domains and accurately capture users' intents and preferences. We conducted extensive experiments on a public dataset and a large-scale industrial dataset collected from one of the world's leading e-commerce corporations. The experimental results indicate that SITN achieves significant improvements over state-of-the-art recommendation methods. Additionally, SITN has been deployed on a micro-video recommendation platform, and the online A/B testing results further demonstrate its practical value. Supplement is available at: https://github.com/fanqieCoffee/SITN-Supplement.

        ----

        ## [515] Opinion Optimization in Directed Social Networks

        **Authors**: *Haoxin Sun, Zhongzhi Zhang*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25585](https://doi.org/10.1609/aaai.v37i4.25585)

        **Abstract**:

        Shifting social opinions has far-reaching implications in various aspects, such as public health campaigns, product marketing, and political candidates. In this paper, we study a problem of opinion optimization based on the popular Friedkin-Johnsen (FJ) model  for opinion dynamics in an unweighted directed social network with n nodes and m edges. In the FJ model, the internal opinion of every node lies in the closed interval [0, 1], with 0 and 1 being polar opposites of opinions about a certain issue. Concretely, we focus on the problem of selecting a small number of k<

        ----

        ## [516] Self-Supervised Continual Graph Learning in Adaptive Riemannian Spaces

        **Authors**: *Li Sun, Junda Ye, Hao Peng, Feiyang Wang, Philip S. Yu*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25586](https://doi.org/10.1609/aaai.v37i4.25586)

        **Abstract**:

        Continual graph learning routinely finds its role in a variety of real-world applications where the graph data with different tasks come sequentially. Despite the success of prior works, it still faces great challenges. On the one hand, existing methods work with the zero-curvature Euclidean space, and largely ignore the fact that curvature varies over the com- ing graph sequence. On the other hand, continual learners in the literature rely on abundant labels, but labeling graph in practice is particularly hard especially for the continuously emerging graphs on-the-fly. To address the aforementioned challenges, we propose to explore a challenging yet practical problem, the self-supervised continual graph learning in adaptive Riemannian spaces. In this paper, we propose a novel self-supervised Riemannian Graph Continual Learner (RieGrace). In RieGrace, we first design an Adaptive Riemannian GCN (AdaRGCN), a unified GCN coupled with a neural curvature adapter, so that Riemannian space is shaped by the learnt curvature adaptive to each graph. Then, we present a Label-free Lorentz Distillation approach, in which we create teacher-student AdaRGCN for the graph sequence. The student successively performs intra-distillation from itself and inter-distillation from the teacher so as to consolidate knowledge without catastrophic forgetting. In particular, we propose a theoretically grounded Generalized Lorentz Projection for the contrastive distillation in Riemannian space. Extensive experiments on the benchmark datasets show the superiority of RieGrace, and additionally, we investigate on how curvature changes over the graph sequence.

        ----

        ## [517] Self-Organization Preserved Graph Structure Learning with Principle of Relevant Information

        **Authors**: *Qingyun Sun, Jianxin Li, Beining Yang, Xingcheng Fu, Hao Peng, Philip S. Yu*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25587](https://doi.org/10.1609/aaai.v37i4.25587)

        **Abstract**:

        Most Graph Neural Networks follow the message-passing paradigm, assuming the observed structure depicts the ground-truth node relationships. However, this fundamental assumption cannot always be satisfied, as real-world graphs are always incomplete, noisy, or redundant. How to reveal the inherent graph structure in a unified way remains under-explored. 
We proposed PRI-GSL, a Graph Structure Learning framework guided by the Principle of Relevant Information, providing a simple and unified framework for identifying the self-organization and revealing the hidden structure. PRI-GSL learns a structure that contains the most relevant yet least redundant information quantified by von Neumann entropy and Quantum Jensen Shannon divergence. PRI-GSL incorporates the evolution of quantum continuous walk with graph wavelets to encode node structural roles, showing in which way the nodes interplay and self-organize with the graph structure. Extensive experiments demonstrate the superior effectiveness and robustness of PRI-GSL.

        ----

        ## [518] Efficient Embeddings of Logical Variables for Query Answering over Incomplete Knowledge Graphs

        **Authors**: *Dingmin Wang, Yeyuan Chen, Bernardo Cuenca Grau*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25588](https://doi.org/10.1609/aaai.v37i4.25588)

        **Abstract**:

        The problem of answering complex First-order Logic queries over incomplete knowledge graphs  is receiving growing attention in the literature. A promising recent approach to this problem has been to exploit neural link predictors, which can be effective in identifying individual missing triples in the incomplete graph, in order to efficiently answer complex queries. A crucial advantage of this approach over other methods is that it does not require example answers to complex queries for training, as it relies only on the availability of a trained link  predictor for the knowledge graph at hand.  This approach, however, can be computationally expensive during inference,  and cannot deal with queries involving negation. 

In this paper, we propose a novel approach that addresses all of these limitations. Experiments on established benchmark datasets demonstrate that our approach offers  superior performance while  significantly reducing inference times.

        ----

        ## [519] Human-Instructed Deep Hierarchical Generative Learning for Automated Urban Planning

        **Authors**: *Dongjie Wang, Lingfei Wu, Denghui Zhang, Jingbo Zhou, Leilei Sun, Yanjie Fu*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25589](https://doi.org/10.1609/aaai.v37i4.25589)

        **Abstract**:

        The essential task of urban planning is to generate the optimal land-use configuration of a target area. However, traditional urban planning is time-consuming and labor-intensive. Deep generative learning gives us hope that we can automate this planning process and come up with the ideal urban plans. While remarkable achievements have been obtained, they have exhibited limitations in lacking awareness of: 1) the hierarchical dependencies between functional zones and spatial grids; 2) the peer dependencies among functional zones; and 3) human regulations to ensure the usability of generated configurations. To address these limitations, we develop a novel human-instructed deep hierarchical generative model. We rethink the urban planning generative task from a unique functionality perspective, where we summarize planning requirements into different functionality projections for better urban plan generation. To this end, we develop a three-stage generation process from a target area to zones to grids. The first stage is to label the grids of a target area with latent functionalities to discover functional zones. The second stage is to perceive the planning requirements to form urban functionality projections. We propose a novel module: functionalizer to project the embedding of human instructions and geospatial contexts to the zone-level plan to obtain such projections. Each projection includes the information of land-use portfolios and the structural dependencies across spatial grids in terms of a specific urban function. The third stage is to leverage multi-attentions to model the zone-zone peer dependencies of the functionality projections to generate grid-level land-use configurations. Finally, we present extensive experiments to demonstrate the effectiveness of our framework.

        ----

        ## [520] Easy Begun Is Half Done: Spatial-Temporal Graph Modeling with ST-Curriculum Dropout

        **Authors**: *Hongjun Wang, Jiyuan Chen, Tong Pan, Zipei Fan, Xuan Song, Renhe Jiang, Lingyu Zhang, Yi Xie, Zhongyi Wang, Boyuan Zhang*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25590](https://doi.org/10.1609/aaai.v37i4.25590)

        **Abstract**:

        Spatial-temporal (ST) graph modeling, such as traffic speed forecasting and taxi demand prediction, is an important task in deep learning area.
However, for the nodes in the graph, their ST patterns can vary greatly in difficulties for modeling, owning to the heterogeneous nature of ST data. We argue that unveiling the nodes to the model in a meaningful order, from easy to complex, can provide performance improvements over traditional training procedure. The idea has its root in Curriculum Learning, which suggests in the early stage of training models can be sensitive to noise and difficult samples. In this paper, we propose ST-Curriculum Dropout, a novel and easy-to-implement strategy for spatial-temporal graph modeling. Specifically, we evaluate the learning difficulty of each node in high-level feature space and drop those difficult ones out to ensure the model only needs to handle fundamental ST relations at the beginning, before gradually moving to hard ones. Our strategy can be applied to any canonical deep learning architecture without extra trainable parameters, and extensive experiments on a wide range of datasets are conducted to illustrate that, by controlling the difficulty level of ST relations as the training progresses, the model is able to capture better representation of the data and thus yields better generalization.

        ----

        ## [521] Cross-Domain Graph Anomaly Detection via Anomaly-Aware Contrastive Alignment

        **Authors**: *Qizhou Wang, Guansong Pang, Mahsa Salehi, Wray L. Buntine, Christopher Leckie*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25591](https://doi.org/10.1609/aaai.v37i4.25591)

        **Abstract**:

        Cross-domain graph anomaly detection (CD-GAD) describes the problem of detecting anomalous nodes in an unlabelled target graph using auxiliary, related source graphs with labelled anomalous and normal nodes. Although it presents a promising approach to address the notoriously high false positive issue in anomaly detection, little work has been done in this line of research. There are numerous domain adaptation methods in the literature, but it is difficult to adapt them for GAD due to the unknown distributions of the anomalies and the complex node relations embedded in graph data. To this end, we introduce a novel domain adaptation approach, namely Anomaly-aware Contrastive alignmenT (ACT), for GAD. ACT is designed to jointly optimise: (i) unsupervised contrastive learning of normal representations of nodes in the target graph, and (ii) anomaly-aware one-class alignment that aligns these contrastive node representations and the representations of labelled normal nodes in the source graph, while enforcing significant deviation of the representations of the normal nodes from the labelled anomalous nodes in the source graph. In doing so, ACT effectively transfers anomaly-informed knowledge from the source graph to learn the complex node relations of the normal class for GAD on the target graph without any specification of the anomaly distributions. Extensive experiments on eight CD-GAD settings demonstrate that our approach ACT achieves substantially improved detection performance over 10 state-of-the-art GAD methods. Code is available at https://github.com/QZ-WANG/ACT.

        ----

        ## [522] WSiP: Wave Superposition Inspired Pooling for Dynamic Interactions-Aware Trajectory Prediction

        **Authors**: *Renzhi Wang, Senzhang Wang, Hao Yan, Xiang Wang*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25592](https://doi.org/10.1609/aaai.v37i4.25592)

        **Abstract**:

        Predicting motions of surrounding vehicles is critically important to help autonomous driving systems plan a safe path and avoid collisions. Although recent social pooling based LSTM models have achieved significant performance gains by considering the motion interactions between vehicles close to each other, vehicle trajectory prediction still remains as a challenging research issue due to the dynamic and high-order interactions in the real complex driving scenarios. To this end, we propose a wave superposition inspired social pooling (Wave-pooling for short) method for dynamically aggregating the high-order interactions from both local and global neighbor vehicles. Through modeling each vehicle as a wave with the amplitude and phase, Wave-pooling can more effectively represent the dynamic motion states of vehicles and capture their high-order dynamic interactions by wave superposition. By integrating Wave-pooling, an encoder-decoder based learning framework named WSiP is also proposed. Extensive experiments conducted on two public highway datasets NGSIM and highD verify the effectiveness of WSiP by comparison with current state-of-the-art baselines. More importantly, the result of WSiP is more interpretable as the interaction strength between vehicles can be intuitively reflected by their phase difference. The code of the work is publicly available at https://github.com/Chopin0123/WSiP.

        ----

        ## [523] Beyond Graph Convolutional Network: An Interpretable Regularizer-Centered Optimization Framework

        **Authors**: *Shiping Wang, Zhihao Wu, Yuhong Chen, Yong Chen*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25593](https://doi.org/10.1609/aaai.v37i4.25593)

        **Abstract**:

        Graph convolutional networks (GCNs) have been attracting widespread attentions due to their encouraging performance and powerful generalizations. However, few work provide a general view to interpret various GCNs and guide GCNs' designs. In this paper, by revisiting the original GCN, we induce an interpretable regularizer-centerd optimization framework, in which by building appropriate regularizers we can interpret most GCNs, such as APPNP, JKNet, DAGNN, and GNN-LF/HF. Further, under the proposed framework, we devise a dual-regularizer graph convolutional network (dubbed tsGCN) to capture topological and semantic structures from graph data. Since the derived learning rule for tsGCN contains an inverse of a large matrix and thus is time-consuming, we leverage the Woodbury matrix identity and low-rank approximation tricks to successfully decrease the high computational complexity of computing infinite-order graph convolutions. Extensive experiments on eight public datasets demonstrate that tsGCN achieves superior performance against quite a few state-of-the-art competitors w.r.t. classification tasks.

        ----

        ## [524] Augmenting Affective Dependency Graph via Iterative Incongruity Graph Learning for Sarcasm Detection

        **Authors**: *Xiaobao Wang, Yiqi Dong, Di Jin, Yawen Li, Longbiao Wang, Jianwu Dang*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25594](https://doi.org/10.1609/aaai.v37i4.25594)

        **Abstract**:

        Recently, progress has been made towards improving automatic sarcasm detection in computer science. Among existing models, manually constructing static graphs for texts and then using graph neural networks (GNNs) is one of the most effective approaches for drawing long-range incongruity patterns. However, the manually constructed graph structure might be prone to errors (e.g., noisy or incomplete) and not optimal for the sarcasm detection task. Errors produced during the graph construction step cannot be remedied and may accrue to the following stages, resulting in poor performance. To surmount the above limitations, we explore a novel Iterative Augmenting Affective Graph and Dependency Graph (IAAD) framework to jointly and iteratively learn the incongruity graph structure. IAAD can alternatively update the incongruity graph structure and node representation until the learning graph structure is optimal for the metrics of sarcasm detection. More concretely, we begin with deriving an affective and a dependency graph for each instance, then an iterative incongruity graph learning module is employed to augment affective and dependency graphs for obtaining the optimal inconsistent semantic graph with the goal of optimizing the graph for the sarcasm detection task. Extensive experiments on three datasets demonstrate that the proposed model outperforms state-of-the-art baselines for sarcasm detection with significant margins.

        ----

        ## [525] Structure Aware Incremental Learning with Personalized Imitation Weights for Recommender Systems

        **Authors**: *Yuening Wang, Yingxue Zhang, Antonios Valkanas, Ruiming Tang, Chen Ma, Jianye Hao, Mark Coates*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25595](https://doi.org/10.1609/aaai.v37i4.25595)

        **Abstract**:

        Recommender systems now consume large-scale data and play a significant role in improving user experience. Graph Neural Networks (GNNs) have emerged as one of the most effective recommender system models because they model the rich relational information. The ever-growing volume of data can make training GNNs prohibitively expensive. To address this, previous attempts propose to train the GNN models incrementally as new data blocks arrive. 
Feature and structure knowledge distillation techniques have been explored to allow the GNN model to train in a fast incremental fashion while alleviating the catastrophic forgetting problem. 
However, preserving the same amount of the historical information for all users is sub-optimal since it fails to take into account the dynamics of each user's change of preferences. 
For the users whose interests shift substantially, retaining too much of the old knowledge can overly constrain the model, preventing it from quickly adapting to the users’ novel interests. 
In contrast, for users who have static preferences, model performance can benefit greatly from preserving as much of the user's long-term preferences as possible.
In this work, we propose a novel training strategy that adaptively learns personalized imitation weights for each user to balance the contribution from the recent data and the amount of knowledge to be distilled from previous time periods.
We demonstrate the effectiveness of learning imitation weights via a comparison on five diverse datasets for three state-of-art structure distillation based recommender systems. The performance shows consistent improvement over competitive incremental learning techniques.

        ----

        ## [526] Online Semi-supervised Learning with Mix-Typed Streaming Features

        **Authors**: *Di Wu, Shengda Zhuo, Yu Wang, Zhong Chen, Yi He*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25596](https://doi.org/10.1609/aaai.v37i4.25596)

        **Abstract**:

        Online learning with feature spaces that are not fixed but can vary over time renders a seemingly flexible learning paradigm thus has drawn much attention. Unfortunately, two restrictions prohibit a ubiquitous application of this learning paradigm in practice. First, whereas prior studies mainly assume a homogenous feature type, data streams generated from real applications can be heterogeneous in which Boolean, ordinal, and continuous co-exist. Existing methods that prescribe parametric distributions such as Gaussians would not suffice to model the correlation among such mixtyped features. Second, while full supervision seems to be a default setup, providing labels to all arriving data instances over a long time span is tangibly onerous, laborious, and economically unsustainable. Alas, a semi-supervised online learner that can deal with mix-typed, varying feature spaces is still missing. To fill the gap, this paper explores a novel problem, named Online Semi-supervised Learning with Mixtyped streaming Features (OSLMF), which strives to relax the restrictions on the feature type and supervision information. Our key idea to solve the new problem is to leverage copula model to align the data instances with different feature spaces so as to make their distance measurable. A geometric structure underlying data instances is then established in an online fashion based on their distances, through which the limited labeling information is propagated, from the scarce labeled instances to their close neighbors. Experimental results are documented to evidence the viability and effectiveness of our proposed approach. Code is released in https://github.com/wudi1989/OSLMF.

        ----

        ## [527] Few-Shot Composition Learning for Image Retrieval with Prompt Tuning

        **Authors**: *Junda Wu, Rui Wang, Handong Zhao, Ruiyi Zhang, Chaochao Lu, Shuai Li, Ricardo Henao*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25597](https://doi.org/10.1609/aaai.v37i4.25597)

        **Abstract**:

        We study the problem of composition learning for image retrieval, for which we learn to retrieve target images with search queries in the form of a composition of a reference image and a modification text that describes desired modifications of the image. Existing models of composition learning for image retrieval are generally built with large-scale datasets, demanding extensive training samples, i.e., query-target pairs, as supervision, which restricts their application for the scenario of few-shot learning with only few query-target pairs available. Recently, prompt tuning with frozen pretrained language models has shown remarkable performance when the amount of training data is limited. Inspired by this, we propose a prompt tuning mechanism with the pretrained CLIP model for the task of few-shot composition learning for image retrieval. Specifically, we regard the representation of the reference image as a trainable visual prompt, prefixed to the embedding of the text sequence. One challenge is to efficiently train visual prompt with few-shot samples. To deal with this issue, we further propose a self-upervised auxiliary task via ensuring that the reference image can retrieve itself when no modification information is given from the text, which facilitates training for the visual prompt, while not requiring additional annotations for query-target pairs. Experiments on multiple benchmarks show that our proposed model can yield superior performance when trained with only few query-target pairs.

        ----

        ## [528] ConTextual Masked Auto-Encoder for Dense Passage Retrieval

        **Authors**: *Xing Wu, Guangyuan Ma, Meng Lin, Zijia Lin, Zhongyuan Wang, Songlin Hu*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25598](https://doi.org/10.1609/aaai.v37i4.25598)

        **Abstract**:

        Dense passage retrieval aims to retrieve the relevant passages of a query from a large corpus based on dense representations (i.e., vectors) of the query and the passages. Recent studies have explored improving pre-trained language models to boost dense retrieval performance. This paper proposes CoT-MAE (ConTextual Masked Auto-Encoder), a simple yet effective generative pre-training method for dense passage retrieval. CoT-MAE employs an asymmetric encoder-decoder architecture that learns to compress the sentence semantics into a dense vector through self-supervised and context-supervised masked auto-encoding. Precisely, self-supervised masked auto-encoding learns to model the semantics of the tokens inside a text span, and context-supervised masked auto-encoding learns to model the semantical correlation between the text spans. We conduct experiments on large-scale passage retrieval benchmarks and show considerable improvements over strong baselines, demonstrating the high efficiency of CoT-MAE. Our code is available at  https://github.com/caskcsg/ir/tree/main/cotmae.

        ----

        ## [529] Jointly Imputing Multi-View Data with Optimal Transport

        **Authors**: *Yangyang Wu, Xiaoye Miao, Xinyu Huang, Jianwei Yin*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25599](https://doi.org/10.1609/aaai.v37i4.25599)

        **Abstract**:

        The multi-view data with incomplete information hinder the effective data analysis. Existing multi-view imputation methods that learn the mapping between complete view and completely missing view are not able to deal with the common multi-view data with missing feature information. In this paper, we propose a generative imputation model named Git with optimal transport theory to jointly impute the missing features/values, conditional on all observed values from the multi-view data. Git consists of two modules, i.e., a multi-view joint generator (MJG) and a masking energy discriminator (MED). The generator MJG incorporates a joint autoencoder with the multiple imputation rule to learn the data distribution from all observed multi-view data. The discriminator MED leverages a new masking energy divergence function to make Git differentiable for imputation enhancement. Extensive experiments on several real-world multi-view data sets demonstrate that, Git yields over 35% accuracy gain, compared to the state-of-the-art approaches.

        ----

        ## [530] Knowledge Graph Embedding by Normalizing Flows

        **Authors**: *Changyi Xiao, Xiangnan He, Yixin Cao*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25600](https://doi.org/10.1609/aaai.v37i4.25600)

        **Abstract**:

        A key to knowledge graph embedding (KGE) is to choose a proper representation space, e.g., point-wise Euclidean space and complex vector space. In this paper, we propose a unified perspective of embedding and introduce uncertainty into KGE from the view of group theory. Our model can incorporate existing models (i.e., generality), ensure the computation is tractable (i.e., efficiency) and enjoy the expressive power of complex random variables (i.e., expressiveness). The core idea is that we embed entities/relations as elements of a symmetric group, i.e., permutations of a set. Permutations of different sets can reflect different properties of embedding. And the group operation of symmetric groups is easy to compute. In specific, we show that the embedding of many existing models, point vectors, can be seen as elements of a symmetric group. To reflect uncertainty, we first embed entities/relations as permutations of a set of random variables. A permutation can transform a simple random variable into a complex random variable for greater expressiveness, called a normalizing flow. We then define scoring functions by measuring the similarity of two normalizing flows, namely NFE. We construct several instantiating models and prove that they are able to learn logical rules. Experimental results demonstrate the effectiveness of introducing uncertainty and our model. The code is available at https://github.com/changyi7231/NFE.

        ----

        ## [531] Temporal Knowledge Graph Reasoning with Historical Contrastive Learning

        **Authors**: *Yi Xu, Junjie Ou, Hui Xu, Luoyi Fu*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25601](https://doi.org/10.1609/aaai.v37i4.25601)

        **Abstract**:

        Temporal knowledge graph, serving as an effective way to store and model dynamic relations, shows promising prospects in event forecasting. However, most temporal knowledge graph reasoning methods are highly dependent on the recurrence or periodicity of events, which brings challenges to inferring future events related to entities that lack historical interaction. In fact, the current moment is often the combined effect of a small part of historical information and those unobserved underlying factors. To this end, we propose a new event forecasting model called Contrastive Event Network (CENET), based on a novel training framework of historical contrastive learning. CENET learns both the historical and non-historical dependency to distinguish the most potential entities that can best match the given query. Simultaneously, it trains representations of queries to investigate whether the current moment depends more on historical or non-historical events by launching contrastive learning. The representations further help train a binary classifier whose output is a boolean mask to indicate related entities in the search space. During the inference process, CENET employs a mask-based strategy to generate the final results. We evaluate our proposed model on five benchmark graphs. The results demonstrate that CENET significantly outperforms all existing methods in most metrics, achieving at least 8.3% relative improvement of Hits@1 over previous state-of-the-art baselines on event-based datasets.

        ----

        ## [532] SCI: A Spectrum Concentrated Implicit Neural Compression for Biomedical Data

        **Authors**: *Runzhao Yang, Tingxiong Xiao, Yuxiao Cheng, Qianni Cao, Jinyuan Qu, Jinli Suo, Qionghai Dai*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25602](https://doi.org/10.1609/aaai.v37i4.25602)

        **Abstract**:

        Massive collection and explosive growth of biomedical data, demands effective compression for efficient storage, transmission and sharing. Readily available visual data compression techniques have been studied extensively but tailored for natural images/videos, and thus show limited performance on biomedical data which are of different features and larger diversity. Emerging implicit neural representation (INR) is gaining momentum and demonstrates high promise for fitting diverse visual data in target-data-specific manner, but a general compression scheme covering diverse biomedical data is so far absent. To address this issue, we firstly derive a mathematical explanation for INR's spectrum concentration property and an analytical insight on the design of INR based compressor. Further, we propose a Spectrum Concentrated Implicit neural compression (SCI) which adaptively partitions the complex biomedical data into blocks matching INR's concentrated spectrum envelop, and design a funnel shaped neural network capable of representing each block with a small number of parameters. Based on this design, we conduct compression via optimization under given budget and allocate the available parameters with high representation accuracy. The experiments show SCI's superior performance to state-of-the-art methods including commercial compressors, data-driven ones, and INR based counterparts on diverse biomedical data. The source code can be found at https://github.com/RichealYoung/ImplicitNeuralCompression.git.

        ----

        ## [533] Unsupervised Legal Evidence Retrieval via Contrastive Learning with Approximate Aggregated Positive

        **Authors**: *Feng Yao, Jingyuan Zhang, Yating Zhang, Xiaozhong Liu, Changlong Sun, Yun Liu, Weixing Shen*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25603](https://doi.org/10.1609/aaai.v37i4.25603)

        **Abstract**:

        Verifying the facts alleged by the prosecutors before the trial requires the judges to retrieve evidence within the massive materials accompanied.
Existing Legal AI applications often assume the facts are already determined and fail to notice the difficulty of reconstructing them. To build a practical Legal AI application and free the judges from the manually searching work, we introduce the task of Legal Evidence Retrieval, which aims at automatically retrieving the precise fact-related verbal evidence within a single case. We formulate the task in a dense retrieval paradigm, and jointly learn the constrastive representations and alignments between facts and evidence. To get rid of the tedious annotations, we construct an approximated positive vector for a given fact by aggregating a set of evidence from the same case. An entropy-based denoise technique is further applied to mitigate the impact of false positive samples. We train our models on tens of thousands of unlabeled cases and evaluate them on a labeled dataset containing 919 cases and 4,336 queries. Experimental results indicate that our approach is effective and outperforms other state-of-the-art representation and retrieval models. The dataset and code are available at https://github.com/yaof20/LER.

        ----

        ## [534] One-for-All: Proposal Masked Cross-Class Anomaly Detection

        **Authors**: *Xincheng Yao, Chongyang Zhang, Ruoqi Li, Jun Sun, Zhenyu Liu*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25604](https://doi.org/10.1609/aaai.v37i4.25604)

        **Abstract**:

        One of the most challenges for anomaly detection (AD) is how to learn one unified and generalizable model to adapt to multi-class especially cross-class settings: the model is trained with normal samples from seen classes with the objective to detect anomalies from both seen and unseen classes. In this work, we propose a novel Proposal Masked Anomaly Detection (PMAD) approach for such challenging multi- and cross-class anomaly detection. The proposed PMAD can be adapted to seen and unseen classes by two key designs: MAE-based patch-level reconstruction and prototype-guided proposal masking. First, motivated by MAE (Masked AutoEncoder), we develop a patch-level reconstruction model rather than the image-level reconstruction adopted in most AD methods for this reason: the masked patches in unseen classes can be reconstructed well by using the visible patches and the adaptive reconstruction capability of MAE. Moreover, we improve MAE by ViT encoder-decoder architecture, combinational masking, and visual tokens as reconstruction objectives to make it more suitable for anomaly detection. Second, we develop a two-stage anomaly detection manner during inference. In the proposal masking stage, the prototype-guided proposal masking module is utilized to generate proposals for suspicious anomalies as much as possible, then masked patches can be generated from the proposal regions. By masking most likely anomalous patches, the “shortcut reconstruction” issue (i.e., anomalous regions can be well reconstructed) can be mostly avoided. In the reconstruction stage, these masked patches are then reconstructed by the trained patch-level reconstruction model to determine if they are anomalies. Extensive experiments show that the proposed PMAD can outperform current state-of-the-art models significantly under the multi- and especially cross-class settings. Code will be publicly available at https://github.com/xcyao00/PMAD.

        ----

        ## [535] Analogical Inference Enhanced Knowledge Graph Embedding

        **Authors**: *Zhen Yao, Wen Zhang, Mingyang Chen, Yufeng Huang, Yi Yang, Huajun Chen*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25605](https://doi.org/10.1609/aaai.v37i4.25605)

        **Abstract**:

        Knowledge graph embedding (KGE), which maps entities and relations in a knowledge graph into continuous vector spaces, has achieved great success in predicting missing links in knowledge graphs. However, knowledge graphs often contain incomplete triples that are difficult to inductively infer by KGEs. To address this challenge, we resort to analogical inference and propose a novel and general self-supervised framework AnKGE to enhance KGE models with analogical inference capability. We propose an analogical object retriever that retrieves appropriate analogical objects from entity-level, relation-level, and triple-level. And in AnKGE, we train an analogy function for each level of analogical inference with the original element embedding from a well-trained KGE model as input, which outputs the analogical object embedding. In order to combine inductive inference capability from the original KGE model and analogical inference capability enhanced by AnKGE, we interpolate the analogy score with the base model score and introduce the adaptive weights in the score function for prediction. Through extensive experiments on FB15k-237 and WN18RR datasets, we show that AnKGE achieves competitive results on link prediction task and well performs analogical inference.

        ----

        ## [536] A Noise-Tolerant Differentiable Learning Approach for Single Occurrence Regular Expression with Interleaving

        **Authors**: *Rongzhen Ye, Tianqu Zhuang, Hai Wan, Jianfeng Du, Weilin Luo, Pingjia Liang*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25606](https://doi.org/10.1609/aaai.v37i4.25606)

        **Abstract**:

        We study the problem of learning a single occurrence regular expression with interleaving (SOIRE) from a set of text strings possibly with noise. SOIRE fully supports interleaving and covers a large portion of regular expressions used in practice. Learning SOIREs is challenging because it requires heavy computation and text strings usually contain noise in practice. Most of the previous studies only learn restricted SOIREs and are not robust on noisy data. To tackle these issues, we propose a noise-tolerant differentiable learning approach SOIREDL for SOIRE. We design a neural network to simulate SOIRE matching and theoretically prove that certain assignments of the set of parameters learnt by the neural network, called faithful encodings, are one-to-one corresponding to SOIREs for a bounded size. Based on this correspondence, we interpret the target SOIRE from an assignment of the set of parameters of the neural network by exploring the nearest faithful encodings. Experimental results show that SOIREDL outperforms the state-of-the-art approaches, especially on noisy data.

        ----

        ## [537] Learning from the Wisdom of Crowds: Exploiting Similar Sessions for Session Search

        **Authors**: *Yuhang Ye, Zhonghua Li, Zhicheng Dou, Yutao Zhu, Changwang Zhang, Shangquan Wu, Zhao Cao*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25607](https://doi.org/10.1609/aaai.v37i4.25607)

        **Abstract**:

        Search engines are essential internet services, enabling users to efficiently find the information they need. Session search employs users’ session logs of queries to solve complex retrieval tasks, in which users search multiple times until interested documents are found. Most existing session search models focus on the contextual information within the current search, ignoring the evidence from historical search sessions. Considering the fact that many ongoing retrieval tasks should have already been carried out by other users with a similar intent, we argue that historical sessions with similar intents can help improve the accuracy of the current search task. We propose a novel Similar Session-enhanced Ranking (SSR) model to improve the session search performance using historical sessions with similar intents. Specifically, the candidate historical sessions are matched by query-level and session-level semantic similarity, and then query-level neighbor behaviors are aggregated by a Query-guided GNN (QGNN) while session-level neighbor behaviors are aggregated using the attention mechanism. Finally, we integrate the refined and aggregated historical neighbor information into the current search session. Experimental results on AOL and Tiangong-ST datasets show that our SSR model significantly outperforms the state-of-the-art models.

        ----

        ## [538] Next POI Recommendation with Dynamic Graph and Explicit Dependency

        **Authors**: *Feiyu Yin, Yong Liu, Zhiqi Shen, Lisi Chen, Shuo Shang, Peng Han*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25608](https://doi.org/10.1609/aaai.v37i4.25608)

        **Abstract**:

        Next Point-Of-Interest (POI) recommendation plays an important role in various location-based services. Its main objective is to predict the user's next interested POI based on her previous check-in information. Most existing methods directly use users' historical check-in trajectories to construct various graphs to assist sequential models to complete this task. However, as users' check-in data is extremely sparse, it is difficult to capture the potential relations between POIs by directly using these check-in data. To this end, we propose the Sequence-based Neighbour search and Prediction Model (SNPM) for next POI recommendation. In SNPM, the RotatE knowledge graph embedding and Eigenmap methods are used to extract POI relationships implied in check-in data, and build the POI similarity graph. Then, we enhance the model's generalized representations of POIs' general features by aggregating similar POIs. As the context is typically rich and valuable when making Next POI predictions, the sequence model selects which POIs to aggregate not only depends on the current state, but also needs to consider the previous POI sequence. Therefore, we construct a Sequence-based, Dynamic Neighbor Graph (SDNG) to find the similarity neighbourhood and develop a Multi-Step Dependency Prediction model (MSDP) inspired by RotatE, which explicitly leverage information from previous states. We evaluate the proposed model on two real-world datasets, and the experimental results show that the proposed method significantly outperforms existing state-of-the-art POI recommendation methods.

        ----

        ## [539] Predicting Temporal Sets with Simplified Fully Connected Networks

        **Authors**: *Le Yu, Zihang Liu, Tongyu Zhu, Leilei Sun, Bowen Du, Weifeng Lv*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25609](https://doi.org/10.1609/aaai.v37i4.25609)

        **Abstract**:

        Given a sequence of sets, where each set contains an arbitrary number of elements, temporal sets prediction aims to predict which elements will appear in the subsequent set. Existing methods for temporal sets prediction are developed on sophisticated components (e.g., recurrent neural networks, attention or gating mechanisms, and graph neural networks), which inevitably increase the model complexity due to more trainable parameters and higher computational costs. Moreover, the involved nonlinear activation may contribute little or even degrade the performance. In this paper, we present a succinct architecture that is solely built on the Simplified Fully Connected Networks (SFCNs) for temporal sets prediction to bring both effectiveness and efficiency together. In particular, given a user's sequence of sets, we employ SFCNs to derive representations of the user by learning inter-set temporal dependencies, intra-set element relationships, and intra-embedding channel correlations. Two families of general functions are introduced to preserve the permutation-invariant property of each set and the permutation-equivariant property of elements in each set. Moreover, we design a user representations adaptive fusing module to aggregate user representations according to each element for improving the prediction performance. Experiments on four benchmarks show the superiority of our approach over the state-of-the-art under both transductive and inductive settings. We also theoretically and empirically demonstrate that our model has lower space and time complexity than baselines. Codes and datasets are available at https://github.com/yule-BUAA/SFCNTSP.

        ----

        ## [540] Learning to Count Isomorphisms with Graph Neural Networks

        **Authors**: *Xingtong Yu, Zemin Liu, Yuan Fang, Xinming Zhang*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25610](https://doi.org/10.1609/aaai.v37i4.25610)

        **Abstract**:

        Subgraph isomorphism counting is an important problem on graphs, as many graph-based tasks exploit recurring subgraph patterns. Classical methods usually boil down to a backtracking framework that needs to navigate a huge search space with prohibitive computational cost. Some recent studies resort to graph neural networks (GNNs) to learn a low-dimensional representation for both the query and input graphs, in order to predict the number of subgraph isomorphisms on the input graph. However, typical GNNs employ a node-centric message passing scheme that receives and aggregates messages on nodes, which is inadequate in complex structure matching for isomorphism counting. Moreover, on an input graph, the space of possible query graphs is enormous, and different parts of the input graph will be triggered to match different queries. Thus, expecting a fixed representation of the input graph to match diversely structured query graphs is unrealistic. In this paper, we propose a novel GNN called Count-GNN for subgraph isomorphism counting, to deal with the above challenges. At the edge level, given that an edge is an atomic unit of encoding graph structures, we propose an edge-centric message passing scheme, where messages on edges are propagated and aggregated based on the edge adjacency to preserve fine-grained structural information. At the graph level, we modulate the input graph representation conditioned on the query, so that the input graph can be adapted to each query individually to improve their matching. Finally, we conduct extensive experiments on a number of benchmark datasets to demonstrate the superior performance of Count-GNN.

        ----

        ## [541] Untargeted Attack against Federated Recommendation Systems via Poisonous Item Embeddings and the Defense

        **Authors**: *Yang Yu, Qi Liu, Likang Wu, Runlong Yu, Sanshi Lei Yu, Zaixi Zhang*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25611](https://doi.org/10.1609/aaai.v37i4.25611)

        **Abstract**:

        Federated recommendation (FedRec) can train personalized recommenders without collecting user data, but the decentralized nature makes it susceptible to poisoning attacks. Most previous studies focus on the targeted attack to promote certain items, while the untargeted attack that aims to degrade the overall performance of the FedRec system remains less explored. In fact, untargeted attacks can disrupt the user experience and bring severe ﬁnancial loss to the service provider. However, existing untargeted attack methods are either inapplicable or ineffective against FedRec systems. In this paper, we delve into the untargeted attack and its defense for FedRec systems. (i) We propose ClusterAttack, a novel untargeted attack method. It uploads poisonous gradients that converge the item embeddings into several dense clusters, which make the recommender generate similar scores for these items in the same cluster and perturb the ranking order. (ii) We propose a uniformity-based defense mechanism (UNION) to protect FedRec systems from such attacks. We design a contrastive learning task that regularizes the item embeddings toward a uniform distribution. Then the server ﬁlters out these malicious gradients by estimating the uniformity of updated item embeddings. Experiments on two public datasets show that ClusterAttack can effectively degrade the performance of FedRec systems while circumventing many defense methods, and UNION can improve the resistance of the system against various untargeted attacks, including our ClusterAttack.

        ----

        ## [542] Practical Cross-System Shilling Attacks with Limited Access to Data

        **Authors**: *Meifang Zeng, Ke Li, Bingchuan Jiang, Liujuan Cao, Hui Li*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25612](https://doi.org/10.1609/aaai.v37i4.25612)

        **Abstract**:

        In shilling attacks, an adversarial party injects a few fake user profiles into a Recommender System (RS) so that the target item can be promoted or demoted. Although much effort has been devoted to developing shilling attack methods, we find that existing approaches are still far from practical. In this paper, we analyze the properties a practical shilling attack method should have and propose a new concept of Cross-system Attack. With the idea of Cross-system Attack, we design a Practical Cross-system Shilling Attack (PC-Attack) framework that requires little information about the victim RS model and the target RS data for conducting attacks. PC-Attack is trained to capture graph topology knowledge from public RS data in a self-supervised manner. Then, it is fine-tuned on a small portion of target data that is easy to access to construct fake profiles. Extensive experiments have demonstrated the superiority of PC-Attack over state-of-the-art baselines. Our implementation of PC-Attack is available at https://github.com/KDEGroup/PC-Attack.

        ----

        ## [543] Query-Aware Quantization for Maximum Inner Product Search

        **Authors**: *Jin Zhang, Defu Lian, Haodi Zhang, Baoyun Wang, Enhong Chen*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25613](https://doi.org/10.1609/aaai.v37i4.25613)

        **Abstract**:

        Maximum Inner Product Search (MIPS) plays an essential role in many applications ranging from information retrieval, recommender systems to natural language processing. However, exhaustive MIPS is often expensive and impractical when there are a large number of candidate items. 
The state-of-the-art quantization method of approximated MIPS is product quantization with a score-aware loss, developed by assuming that queries are uniformly distributed in the unit sphere. However, in real-world datasets, the above assumption about queries does not necessarily hold. 
To this end, we propose a quantization method based on the distribution of queries combined with sampled softmax.
Further, we introduce a general framework encompassing the proposed method and multiple quantization methods, and we develop an effective optimization for the proposed general framework. The proposed method is evaluated on three real-world datasets. The experimental results show that it outperforms the state-of-the-art baselines.

        ----

        ## [544] TOT：Topology-Aware Optimal Transport for Multimodal Hate Detection

        **Authors**: *Linhao Zhang, Li Jin, Xian Sun, Guangluan Xu, Zequn Zhang, Xiaoyu Li, Nayu Liu, Qing Liu, Shiyao Yan*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25614](https://doi.org/10.1609/aaai.v37i4.25614)

        **Abstract**:

        Multimodal hate detection, which aims to identify the harmful content online such as memes, is crucial for building a wholesome internet environment. Previous work has made enlightening exploration in detecting explicit hate remarks. However, most of their approaches neglect the analysis of implicit harm, which is particularly challenging as explicit text markers and demographic visual cues are often twisted or missing. The leveraged cross-modal attention mechanisms also suffer from the distributional modality gap and lack logical interpretability. To address these 
semantic gap issues, we propose TOT: a topology-aware optimal transport framework to decipher the implicit harm in memes scenario, which formulates the cross-modal aligning problem as solutions for optimal transportation plans. Specifically, we leverage an optimal transport kernel method to capture complementary information from multiple modalities. The kernel embedding provides a non-linear transformation ability to reproduce a kernel Hilbert space (RKHS), which reflects significance for eliminating the distributional modality gap. Moreover, we perceive the topology information based on aligned representations to conduct bipartite graph path reasoning. The newly achieved state-of-the-art performance on two publicly available benchmark datasets, together with  further visual analysis, demonstrate the superiority of TOT in capturing implicit cross-modal alignment.

        ----

        ## [545] Cross-Domain Few-Shot Graph Classification with a Reinforced Task Coordinator

        **Authors**: *Qiannan Zhang, Shichao Pei, Qiang Yang, Chuxu Zhang, Nitesh V. Chawla, Xiangliang Zhang*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25615](https://doi.org/10.1609/aaai.v37i4.25615)

        **Abstract**:

        Cross-domain graph few-shot learning attempts to address the prevalent data scarcity issue in graph mining problems. However, the utilization of cross-domain data induces another intractable domain shift issue which severely degrades the generalization ability of cross-domain graph few-shot learning models. The combat with the domain shift issue is hindered due to the coarse utilization of source domains and the ignorance of accessible prompts. To address these challenges, in this paper, we design a novel Cross-domain Task Coordinator to leverage a small set of labeled target domain data as prompt tasks, then model the association and discover the relevance between meta-tasks from the source domain and the prompt tasks. Based on the discovered relevance, our model achieves adaptive task selection and enables the optimization of a graph learner using the selected fine-grained meta-tasks. Extensive experiments conducted on molecular property prediction benchmarks validate the effectiveness of our proposed method by comparing it with state-of-the-art baselines.

        ----

        ## [546] AutoSTL: Automated Spatio-Temporal Multi-Task Learning

        **Authors**: *Zijian Zhang, Xiangyu Zhao, Hao Miao, Chunxu Zhang, Hongwei Zhao, Junbo Zhang*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25616](https://doi.org/10.1609/aaai.v37i4.25616)

        **Abstract**:

        Spatio-temporal prediction plays a critical role in smart city construction. Jointly modeling multiple spatio-temporal tasks can further promote an intelligent city life by integrating their inseparable relationship. However, existing studies fail to address this joint learning problem well, which generally solve tasks individually or a fixed task combination. The challenges lie in the tangled relation between different properties, the demand for supporting flexible combinations of tasks and the complex spatio-temporal dependency. To cope with the problems above, we propose an Automated Spatio-Temporal multi-task Learning (AutoSTL) method to handle multiple spatio-temporal tasks jointly. Firstly, we propose a scalable architecture consisting of advanced spatio-temporal operations to exploit the complicated dependency. Shared modules and feature fusion mechanism are incorporated to further capture the intrinsic relationship between tasks. Furthermore, our model automatically allocates the operations and fusion weight. Extensive experiments on benchmark datasets verified that our model achieves state-of-the-art performance. As we can know, AutoSTL is the first automated spatio-temporal multi-task learning method.

        ----

        ## [547] Fair Representation Learning for Recommendation: A Mutual Information Perspective

        **Authors**: *Chen Zhao, Le Wu, Pengyang Shao, Kun Zhang, Richang Hong, Meng Wang*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25617](https://doi.org/10.1609/aaai.v37i4.25617)

        **Abstract**:

        Recommender systems have been widely used in recent years. By exploiting historical user-item interactions, recommender systems can model personalized potential interests of users and have been widely applied to a wide range of scenarios. Despite their impressive performance, most of them may be subject to unwanted biases related to sensitive attributes (e.g., race and gender), leading to unfairness. An intuitive idea to alleviate this problem is to ensure that there is no mutual information between recommendation results and sensitive attributes. However, keeping independence conditions solely achieves fairness improvement while causing an obvious degradation of recommendation accuracy, which is not a desired result. To this end, in this paper, we re-define recommendation fairness with a novel two-fold mutual information objective. In concerned details, we define fairness as mutual information minimization between embeddings and sensitive information, and mutual information maximization between embeddings and non-sensitive information. Then, a flexible Fair Mutual Information (FairMI) framework is designed to achieve this goal. FairMI first employs a sensitive attribute encoder to capture sensitive information in the data. Then, based on results from the sensitive attribute encoder, an interest encoder is developed to generate sensitive-free embeddings, which are expected to contain rich non-sensitive information of input data. Moreover, we propose novel mutual information (upper/lower) bounds with contrastive information estimation for model optimization. Extensive experiments over two real-world datasets demonstrate the effectiveness of our proposed FairMI in reducing unfairness and improving recommendation accuracy simultaneously.

        ----

        ## [548] Deep Graph Structural Infomax

        **Authors**: *Wenting Zhao, Gongping Xu, Zhen Cui, Siqiang Luo, Cheng Long, Tong Zhang*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25618](https://doi.org/10.1609/aaai.v37i4.25618)

        **Abstract**:

        In the scene of self-supervised graph learning, Mutual Information (MI) was recently introduced for graph encoding to generate robust node embeddings. A successful representative is Deep Graph Infomax (DGI), which essentially operates on the space of node features but ignores topological structures, and just considers global graph summary. In this paper, we present an effective model called Deep Graph Structural Infomax (DGSI) to learn node representation. We explore to derive the structural mutual information from the perspective of Information Bottleneck (IB), which defines a trade-off between the sufficiency and minimality of representation on the condition of the topological structure preservation. Intuitively, the derived constraints formally maximize the structural mutual information both edge-wise and local neighborhood-wise. Besides, we develop a general framework that incorporates the global representational mutual information, local representational mutual information, and sufficient structural information into the node representation. Essentially, our DGSI extends DGI and could capture more fine-grained semantic information as well as beneficial structural information in a self-supervised manner, thereby improving node representation and further boosting the learning performance. Extensive experiments on different types of datasets demonstrate the effectiveness and superiority of the proposed method.

        ----

        ## [549] Causal Conditional Hidden Markov Model for Multimodal Traffic Prediction

        **Authors**: *Yu Zhao, Pan Deng, Junting Liu, Xiaofeng Jia, Mulan Wang*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25619](https://doi.org/10.1609/aaai.v37i4.25619)

        **Abstract**:

        Multimodal traffic flow can reflect the health of the transportation system, and its prediction is crucial to urban traffic management. Recent works overemphasize spatio-temporal correlations of traffic flow, ignoring the physical concepts that lead to the generation of observations and their causal relationship. Spatio-temporal correlations are considered unstable under the influence of different conditions, and spurious correlations may exist in observations. In this paper, we analyze the physical concepts affecting the generation of multimode traffic flow from the perspective of the observation generation principle and propose a Causal Conditional Hidden Markov Model (CCHMM) to predict multimodal traffic flow. In the latent variables inference stage, a posterior network disentangles the causal representations of the concepts of interest from conditional information and observations, and a causal propagation module mines their causal relationship. In the data generation stage, a prior network samples the causal latent variables from the prior distribution and feeds them into the generator to generate multimodal traffic flow. We use a mutually supervised training method for the prior and posterior to enhance the identifiability of the model. Experiments on real-world datasets show that CCHMM can effectively disentangle causal representations of concepts of interest and identify causality, and accurately predict multimodal traffic flow.

        ----

        ## [550] ADMoE: Anomaly Detection with Mixture-of-Experts from Noisy Labels

        **Authors**: *Yue Zhao, Guoqing Zheng, Subhabrata Mukherjee, Robert McCann, Ahmed Awadallah*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25620](https://doi.org/10.1609/aaai.v37i4.25620)

        **Abstract**:

        Existing works on anomaly detection (AD) rely on clean labels from human annotators that are expensive to acquire in practice.  In this work, we propose a method to leverage weak/noisy labels (e.g., risk scores generated by machine rules for detecting malware) that are cheaper to obtain for anomaly detection. Specifically, we propose ADMoE, the first framework for anomaly detection algorithms to learn from noisy labels. In a nutshell, ADMoE leverages mixture-of-experts (MoE) architecture to encourage specialized and scalable learning from multiple noisy sources. It captures the similarities among noisy labels by sharing most model parameters, while encouraging specialization by building "expert" sub-networks. To further juice out the signals from noisy labels, ADMoE uses them as input features to facilitate expert learning. Extensive results on eight datasets (including a proprietary enterprise security dataset) demonstrate the effectiveness of ADMoE, where it brings up to 34% performance improvement over not using it. Also, it outperforms a total of 13 leading baselines with equivalent network parameters and FLOPS. Notably, ADMoE is model-agnostic to enable any neural network-based detection methods to handle noisy labels, where we showcase its results on both multiple-layer perceptron (MLP) and the leading AD method DeepSAD.

        ----

        ## [551] A Provable Framework of Learning Graph Embeddings via Summarization

        **Authors**: *Houquan Zhou, Shenghua Liu, Danai Koutra, Huawei Shen, Xueqi Cheng*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25621](https://doi.org/10.1609/aaai.v37i4.25621)

        **Abstract**:

        Given a large graph, can we learn its node embeddings from a smaller summary graph? What is the relationship between embeddings learned from original graphs and their summary graphs? Graph representation learning plays an important role in many graph mining applications, but learning em-beddings of large-scale graphs remains a challenge. Recent works try to alleviate it via graph summarization, which typ-ically includes the three steps: reducing the graph size by combining nodes and edges into supernodes and superedges,learning the supernode embedding on the summary graph and then restoring the embeddings of the original nodes. How-ever, the justification behind those steps is still unknown.
In this work, we propose GELSUMM, a well-formulated graph embedding learning framework based on graph sum-marization, in which we show the theoretical ground of learn-ing from summary graphs and the restoration with the three well-known graph embedding approaches in a closed form.Through extensive experiments on real-world datasets, we demonstrate that our methods can learn graph embeddings with matching or better performance on downstream tasks.This work provides theoretical analysis for learning node em-beddings via summarization and helps explain and under-stand the mechanism of the existing works.

        ----

        ## [552] GraphSR: A Data Augmentation Algorithm for Imbalanced Node Classification

        **Authors**: *Mengting Zhou, Zhiguo Gong*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25622](https://doi.org/10.1609/aaai.v37i4.25622)

        **Abstract**:

        Graph neural networks (GNNs) have achieved great success in node classification tasks. However, existing GNNs naturally bias towards the majority classes with more labelled data and ignore those minority classes with relatively few labelled ones. The traditional techniques often resort over-sampling methods, but they may cause overfitting problem. More recently, some works propose to synthesize additional nodes for minority classes from the labelled nodes, however, there is no any guarantee if those generated nodes really stand for the the corresponding minority classes.  In fact, improperly synthesized nodes may result in insufficient generalization of the algorithm. To resolve the problem, in this paper we seek to automatically augment the minority classes from the massive unlabelled nodes of the graph. Specifically, we propose \textit{GraphSR}, a novel self-training strategy to augment the minority classes with significant diversity of unlabelled nodes, which is based on a Similarity-based selection module and a Reinforcement Learning(RL) selection module. The first module finds a subset of unlabelled nodes which are most similar to those labelled minority nodes, and the second one further determines the representative and reliable nodes from the subset via RL technique. 
 Furthermore, the RL-based module can adaptively determine the sampling scale according to current training data. This strategy is general and can be easily combined with different GNNs models. Our experiments demonstrate the proposed approach outperforms the state-of-the-art baselines on various class-imbalanced datasets.

        ----

        ## [553] Detecting Multivariate Time Series Anomalies with Zero Known Label

        **Authors**: *Qihang Zhou, Jiming Chen, Haoyu Liu, Shibo He, Wenchao Meng*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25623](https://doi.org/10.1609/aaai.v37i4.25623)

        **Abstract**:

        Multivariate time series anomaly detection has been extensively studied under the one-class classification setting, where a training dataset with all normal instances is required. However, preparing such a dataset is very laborious since each single data instance should be fully guaranteed to be normal. It is, therefore, desired to explore multivariate time series anomaly detection methods based on the dataset without any label knowledge. In this paper, we propose MTGFlow, an unsupervised anomaly detection approach forMultivariate Time series anomaly detection via dynamic Graph and entityaware normalizing Flow, leaning only on a widely accepted hypothesis that abnormal instances exhibit sparse densities than the normal. However, the complex interdependencies among entities and the diverse inherent characteristics of each entity pose significant challenges to density estimation, let alone to detect anomalies based on the estimated possibility distribution. To tackle these problems, we propose to learn the mutual and dynamic relations among entities via a graph structure learning model, which helps to model the accurate distribution of multivariate time series. Moreover, taking account of distinct characteristics of the individual entities, an entity-aware normalizing flow is developed to describe each entity into a parameterized normal distribution, thereby producing fine-grained density estimation. Incorporating these two strategies, MTGFlow achieves superior anomaly detection performance. Experiments on five public datasets with seven baselines are conducted, MTGFlow outperforms the SOTA methods by up to 5.0 AUROC%.

        ----

        ## [554] GRLSTM: Trajectory Similarity Computation with Graph-Based Residual LSTM

        **Authors**: *Silin Zhou, Jing Li, Hao Wang, Shuo Shang, Peng Han*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25624](https://doi.org/10.1609/aaai.v37i4.25624)

        **Abstract**:

        The computation of trajectory similarity is a crucial task in many spatial data analysis applications. However, existing methods have been designed primarily for trajectories in Euclidean space, which overlooks the fact that real-world trajectories are often generated on road networks. This paper addresses this gap by proposing a novel framework, called GRLSTM (Graph-based Residual LSTM). To jointly capture the properties of trajectories and road networks, the proposed framework incorporates knowledge graph embedding (KGE), graph neural network (GNN), and the residual network into the multi-layer LSTM (Residual-LSTM). Specifically, the framework constructs a point knowledge graph to study the multi-relation of points, as points may belong to both the trajectory and the road network. KGE is introduced to learn point embeddings and relation embeddings to build the point fusion graph, while GNN is used to capture the topology structure information of the point fusion graph. Finally, Residual-LSTM is used to learn the trajectory embeddings.To further enhance the accuracy and robustness of the final trajectory embeddings, we introduce two new neighbor-based point loss functions, namely, graph-based point loss function and trajectory-based point loss function. The GRLSTM is evaluated using two real-world trajectory datasets, and the experimental results demonstrate that GRLSTM outperforms all the state-of-the-art methods significantly.

        ----

        ## [555] Heterogeneous Region Embedding with Prompt Learning

        **Authors**: *Silin Zhou, Dan He, Lisi Chen, Shuo Shang, Peng Han*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25625](https://doi.org/10.1609/aaai.v37i4.25625)

        **Abstract**:

        The prevalence of region-based urban data has opened new possibilities for exploring correlations among regions to improve urban planning and smart-city solutions. Region embedding, which plays a critical role in this endeavor, faces significant challenges related to the varying nature of city data and the effectiveness of downstream applications. In this paper, we propose a novel framework, HREP (Heterogeneous Region Embedding with Prompt learning), which addresses both intra-region and inter-region correlations through two key modules: Heterogeneous Region Embedding (HRE) and prompt learning for different downstream tasks. The HRE module constructs a heterogeneous region graph based on three categories of data, capturing inter-region contexts such as human mobility and geographic neighbors, and intraregion contexts such as POI (Point-of-Interest) information. We use relation-aware graph embedding to learn region and relation embeddings of edge types, and introduce selfattention to capture global correlations among regions. Additionally, we develop an attention-based fusion module to integrate shared information among different types of correlations. To enhance the effectiveness of region embedding in downstream tasks, we incorporate prompt learning, specifically prefix-tuning, which guides the learning of downstream tasks and results in better prediction performance. Our experiment results on real-world datasets demonstrate that our proposed model outperforms state-of-the-art methods.

        ----

        ## [556] Show Me the Way! Bilevel Search for Synthesizing Programmatic Strategies

        **Authors**: *David S. Aleixo, Levi H. S. Lelis*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25626](https://doi.org/10.1609/aaai.v37i4.25626)

        **Abstract**:

        The synthesis of programmatic strategies requires one to search in large non-differentiable spaces of computer programs. Current search algorithms use self-play approaches to guide this search. The issue with these approaches is that the guiding function often provides a weak search signal. This is because self-play functions only measure how well a program performs against other programs. Thus, while small changes to a losing program might not transform it into a winning one, such changes might represent steps in the direction of a winning program. In this paper we introduce a bilevel search algorithm that searches concurrently in the space of programs and in a space of state features. Each iteration of the search in the space of features defines a set of target features that the search in the program space attempts to achieve (i.e., features one observes while following the strategy encoded in a program). We hypothesize the combination of a self-play function and a feature-based one provides a stronger search signal for synthesis. While both functions are used to guide the search in the program space, the self-play function is used to guide the search in the feature space, to allow for the selection of target features that are more likely to lead to winning programs. We evaluated our bilevel algorithm in MicroRTS, a real-time strategy game. Our results show that the bilevel search synthesizes stronger strategies than methods that search only in the program space. Also, the strategies our method synthesizes obtained the highest winning rate in a simulated tournament with several baseline agents, including the best agents from the two latest MicroRTS competitions.

        ----

        ## [557] Anytime User Engagement Prediction in Information Cascades for Arbitrary Observation Periods

        **Authors**: *Akshay Aravamudan, Xi Zhang, Georgios C. Anagnostopoulos*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25627](https://doi.org/10.1609/aaai.v37i4.25627)

        **Abstract**:

        Predicting user engagement -- whether a user will engage in a given information cascade -- is an important problem in the context of social media, as it is useful to online marketing and misinformation mitigation just to name a couple major applications. Based on split population multi-variate survival processes, we develop a discriminative approach that, unlike prior works, leads to a single model for predicting whether individual users of an information network will engage a given cascade for arbitrary forecast horizons and observation periods. Being probabilistic in nature, this model retains the interpretability of its generative counterpart and renders count prediction intervals in a disciplined manner. Our results indicate that our model is highly competitive, if not superior, to current approaches, when compared over varying observed cascade histories and forecast horizons.

        ----

        ## [558] Principled Data-Driven Decision Support for Cyber-Forensic Investigations

        **Authors**: *Soodeh Atefi, Sakshyam Panda, Emmanouil Panaousis, Aron Laszka*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25628](https://doi.org/10.1609/aaai.v37i4.25628)

        **Abstract**:

        In the wake of a cybersecurity incident, it is crucial to promptly discover how the threat actors breached security in order to assess the impact of the incident and to develop and deploy countermeasures that can protect against further attacks. To this end, defenders can launch a cyber-forensic investigation, which discovers the techniques that the threat actors used in the incident. A fundamental challenge in such an investigation is prioritizing the investigation of particular techniques since the investigation of each technique requires time and effort, but forensic analysts cannot know which ones were actually used before investigating them. To ensure prompt discovery, it is imperative to provide decision support that can help forensic analysts with this prioritization. A recent study demonstrated that data-driven decision support, based on a dataset of prior incidents, can provide state-of-the-art prioritization. However, this data-driven approach, called DISCLOSE, is based on a heuristic that utilizes only a subset of the available information and does not approximate optimal decisions. To improve upon this heuristic, we introduce a principled approach for data-driven decision support for cyber-forensic investigations. We formulate the decision-support problem using a Markov decision process, whose states represent the states of a forensic investigation. To solve the decision problem, we propose a Monte Carlo tree search based method, which relies on a k-NN regression over prior incidents to estimate state-transition probabilities. We evaluate our proposed approach on multiple versions of the MITRE ATT&CK dataset, which is a knowledge base of adversarial techniques and tactics based on real-world cyber incidents, and demonstrate that our approach outperforms DISCLOSE in terms of techniques discovered per effort spent.

        ----

        ## [559] BETA-CD: A Bayesian Meta-Learned Cognitive Diagnosis Framework for Personalized Learning

        **Authors**: *Haoyang Bi, Enhong Chen, Weidong He, Han Wu, Weihao Zhao, Shijin Wang, Jinze Wu*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25629](https://doi.org/10.1609/aaai.v37i4.25629)

        **Abstract**:

        Personalized learning is a promising educational approach that aims to provide high-quality personalized services for each student with minimum demands for practice data. The key to achieving that lies in the cognitive diagnosis task, which estimates the cognitive state of the student through his/her logged data of doing practice quizzes. Nevertheless, in the personalized learning scenario, existing cognitive diagnosis models suffer from the inability to (1) quickly adapt to new students using a small amount of data, and (2) measure the reliability of the diagnosis result to avoid improper services that mismatch the student's actual state. In this paper, we propose a general Bayesian mETA-learned Cognitive Diagnosis framework (BETA-CD), which addresses the two challenges by prior knowledge exploitation and model uncertainty quantification, respectively. Specifically, we firstly introduce Bayesian hierarchical modeling to associate each student's cognitive state with a shared prior distribution encoding prior knowledge and a personal posterior distribution indicating model uncertainty. Furthermore, we formulate a meta-learning objective to automatically exploit prior knowledge from historical students, and efficiently solve it with a gradient-based variational inference method. The code will be publicly available at https://github.com/AyiStar/pyat.

        ----

        ## [560] Set-to-Sequence Ranking-Based Concept-Aware Learning Path Recommendation

        **Authors**: *Xianyu Chen, Jian Shen, Wei Xia, Jiarui Jin, Yakun Song, Weinan Zhang, Weiwen Liu, Menghui Zhu, Ruiming Tang, Kai Dong, Dingyin Xia, Yong Yu*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25630](https://doi.org/10.1609/aaai.v37i4.25630)

        **Abstract**:

        With the development of the online education system, personalized education recommendation has played an essential role. In this paper, we focus on developing path recommendation systems that aim to generating and recommending an entire learning path to the given user in each session. Noticing that existing approaches fail to consider the correlations of concepts in the path, we propose a novel framework named Set-to-Sequence Ranking-based Concept-aware Learning Path Recommendation (SRC), which formulates the recommendation task under a set-to-sequence paradigm. Specifically, we first design a concept-aware encoder module which can capture the correlations among the input learning concepts. The outputs are then fed into a decoder module that sequentially generates a path through an attention mechanism that handles correlations between the learning and target concepts. Our recommendation policy is optimized by policy gradient. In addition, we also introduce an auxiliary module based on knowledge tracing to enhance the model’s stability by evaluating students’ learning effects on learning concepts. We conduct extensive experiments on two real-world public datasets and one industrial dataset, and the experimental results demonstrate the superiority and effectiveness of SRC. Code now is available at https://gitee.com/mindspore/models/tree/master/research/recommend/SRC.

        ----

        ## [561] Unsupervised Deep Embedded Fusion Representation of Single-Cell Transcriptomics

        **Authors**: *Yue Cheng, Yanchi Su, Zhuohan Yu, Yanchun Liang, Ka-Chun Wong, Xiangtao Li*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25631](https://doi.org/10.1609/aaai.v37i4.25631)

        **Abstract**:

        Cell clustering is a critical step in analyzing single-cell RNA sequencing (scRNA-seq) data, which allows us to characterize the cellular heterogeneity of transcriptional profiling at the single-cell level. Single-cell deep embedded representation models have recently become popular since they can learn feature representation and clustering simultaneously. However, the model still suffers from a variety of significant challenges, including the massive amount of data, pervasive dropout events, and complicated noise patterns in transcriptional profiling. Here, we propose a Single-Cell Deep Embedding Fusion Representation (scDEFR) model, which develop a deep embedded fusion representation to learn fused heterogeneous latent embedding that contains both the transcriptome gene-level information and the cell topology information. We first fuse them layer by layer to obtain compressed representations of intercellular relationships and transcriptome information. After that, the zero-inflated negative binomial model (ZINB)-based decoder is proposed to capture the global probabilistic structure of the data and reconstruct the final gene expression information and cell graph. Finally, by simultaneously integrating the clustering loss, crossentropy loss, ZINB loss, and the cell graph reconstruction loss,
scDEFR can optimize clustering performance and learn the latent representation in fused information under a joint mutual supervised strategy. We conducted extensive and comprehensive experiments on 15 single-cell RNA-seq datasets from different sequencing platforms to demonstrate the superiority of scDEFR over a variety of state-of-the-art methods.

        ----

        ## [562] Constrained Submodular Optimization for Vaccine Design

        **Authors**: *Zheng Dai, David K. Gifford*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25632](https://doi.org/10.1609/aaai.v37i4.25632)

        **Abstract**:

        Advances in machine learning have enabled the prediction of immune system responses to prophylactic and therapeutic vaccines. However, the engineering task of designing vaccines remains a challenge. In particular, the genetic variability of the human immune system makes it difficult to design peptide vaccines that provide widespread immunity in vaccinated populations.  We introduce a framework for evaluating and designing peptide vaccines that uses probabilistic machine learning models, and demonstrate its ability to produce designs for a SARS-CoV-2 vaccine that outperform previous designs.  We provide a theoretical analysis of the approximability, scalability, and complexity of our framework.

        ----

        ## [563] Flow-Based Robust Watermarking with Invertible Noise Layer for Black-Box Distortions

        **Authors**: *Han Fang, Yupeng Qiu, Kejiang Chen, Jiyi Zhang, Weiming Zhang, Ee-Chien Chang*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25633](https://doi.org/10.1609/aaai.v37i4.25633)

        **Abstract**:

        Deep learning-based digital watermarking frameworks have been widely studied recently. Most existing methods adopt an ``encoder-noise layer-decoder''-based architecture where the embedding and extraction processes are accomplished separately by the encoder and the decoder. However, one potential drawback of such a framework is that the encoder and the decoder may not be well coupled, resulting in the fact that the encoder may embed some redundant features into the host image thus influencing the invisibility and robustness of the whole algorithm. To address this limitation, this paper proposes a flow-based robust watermarking framework. The basic component of such framework is an invertible up-down-sampling neural block that can realize the embedding and extraction simultaneously. As a consequence, the encoded feature could keep high consistency with the feature that the decoder needed, which effectively avoids the embedding of redundant features. In addition, to ensure the robustness of black-box distortion, an invertible noise layer (INL) is designed to simulate the distortion and is served as a noise layer in the training stage. Benefiting from its reversibility, INL is also applied as a preprocessing before extraction to eliminate the distortion, which further improves the robustness of the algorithm. Extensive experiments demonstrate the superiority of the proposed framework in terms of visual quality and robustness. Compared with the state-of-the-art architecture, the visual quality (measured by PSNR) of the proposed framework improves by 2dB and the extraction accuracy after JPEG compression (QF=50) improves by more than 4%. Besides, the robustness against black-box distortions can be greatly achieved with more than 95% extraction accuracy.

        ----

        ## [564] Identifying and Eliminating Majority Illusion in Social Networks

        **Authors**: *Umberto Grandi, Lawqueen Kanesh, Grzegorz Lisowski, Ramanujan Sridharan, Paolo Turrini*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25634](https://doi.org/10.1609/aaai.v37i4.25634)

        **Abstract**:

        Majority illusion occurs in a social network when the majority of the network vertices belong to a certain type but the majority of each vertex's neighbours belong to a different type, therefore creating the wrong perception, i.e., the illusion, that the majority type is different from the actual one. From a system engineering point of view, this motivates the search for algorithms to detect and, where possible, correct this undesirable phenomenon. In this paper we initiate the computational study of majority illusion in social networks, providing NP-hardness and parametrised complexity results for its occurrence and elimination.

        ----

        ## [565] A Domain-Knowledge-Inspired Music Embedding Space and a Novel Attention Mechanism for Symbolic Music Modeling

        **Authors**: *Zixun Guo, Jaeyong Kang, Dorien Herremans*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25635](https://doi.org/10.1609/aaai.v37i4.25635)

        **Abstract**:

        Following the success of the transformer architecture in the natural language domain, transformer-like architectures have been widely applied to the domain of symbolic music recently. Symbolic music and text, however, are two different modalities. Symbolic music contains multiple attributes, both absolute attributes (e.g., pitch) and relative attributes (e.g., pitch interval). These relative attributes shape human perception of musical motifs. These important relative attributes, however, are mostly ignored in existing symbolic music modelling methods with the main reason being the lack of a musically-meaningful embedding space where both the absolute and relative embeddings of the symbolic music tokens can be efficiently represented. In this paper, we propose the Fundamental Music Embedding (FME) for symbolic music based on a bias-adjusted sinusoidal encoding within which both the absolute and the relative attributes can be embedded and the fundamental musical properties (e.g., translational invariance) are explicitly preserved. Taking advantage of the proposed FME, we further propose a novel attention mechanism based on the relative index, pitch and onset embeddings (RIPO attention) such that the musical domain knowledge can be fully utilized for symbolic music modelling. Experiment results show that our proposed model: RIPO transformer which utilizes FME and RIPO attention outperforms the state-of-the-art transformers (i.e., music transformer, linear transformer) in a melody completion task. Moreover, using the RIPO transformer in a downstream music generation task, we notice that the notorious degeneration phenomenon no longer exists and the music generated by the RIPO transformer outperforms the music generated by state-of-the-art transformer models in both subjective and objective evaluations. The code of the proposed method is available online: github.com/guozixunnicolas/FundamentalMusicEmbedding.

        ----

        ## [566] MSDC: Exploiting Multi-State Power Consumption in Non-intrusive Load Monitoring Based on a Dual-CNN Model

        **Authors**: *Jialing He, Jiamou Liu, Zijian Zhang, Yang Chen, Yiwei Liu, Bakh Khoussainov, Liehuang Zhu*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25636](https://doi.org/10.1609/aaai.v37i4.25636)

        **Abstract**:

        Non-intrusive load monitoring (NILM) aims to decompose aggregated electrical usage signal into appliance-specific power consumption and it amounts to a classical example of blind source separation tasks. Leveraging recent progress on deep learning techniques, we design a new neural NILM model {\em Multi-State Dual CNN} (MSDC). Different from previous models, MSDC explicitly extracts information about the appliance's multiple states and state transitions, which in turn regulates the prediction of signals for appliances. More specifically, we employ a dual-CNN architecture: one CNN for outputting state distributions and the other for predicting the power of each state. A new technique is invented that utilizes conditional random fields (CRF) to capture state transitions. Experiments on two real-world datasets REDD and UK-DALE demonstrate that our model significantly outperform state-of-the-art models  while having good generalization capacity, achieving 6%-10% MAE gain and 33%-51% SAE gain to unseen appliances.

        ----

        ## [567] Integrating Reward Maximization and Population Estimation: Sequential Decision-Making for Internal Revenue Service Audit Selection

        **Authors**: *Peter Henderson, Ben Chugg, Brandon R. Anderson, Kristen M. Altenburger, Alex Turk, John Guyton, Jacob Goldin, Daniel E. Ho*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25637](https://doi.org/10.1609/aaai.v37i4.25637)

        **Abstract**:

        We introduce a new setting, optimize-and-estimate structured bandits. Here, a policy must select a batch of arms, each characterized by its own context, that would allow it to both maximize reward and maintain an accurate (ideally unbiased) population estimate of the reward. This setting is inherent to many public and private sector applications and often requires handling delayed feedback, small data, and distribution shifts. We demonstrate its importance on real data from the United States Internal Revenue Service (IRS). The IRS performs yearly audits of the tax base. Two of its most important objectives are to identify suspected misreporting and to estimate the "tax gap" -- the global difference between the amount paid and true amount owed. Based on a unique collaboration with the IRS, we cast these two processes as a unified optimize-and-estimate structured bandit. We analyze optimize-and-estimate approaches to the IRS problem and propose a novel mechanism for unbiased population estimation that achieves rewards comparable to baseline approaches. This approach has the potential to improve audit efficacy, while maintaining policy-relevant estimates of the tax gap. This has important social consequences given that the current tax gap is estimated at nearly half a trillion dollars. We suggest that this problem setting is fertile ground for further research and we highlight its interesting challenges. The results of this and related research are currently being incorporated into the continual improvement of the IRS audit selection methods.

        ----

        ## [568] MGTCF: Multi-Generator Tropical Cyclone Forecasting with Heterogeneous Meteorological Data

        **Authors**: *Cheng Huang, Cong Bai, Sixian Chan, Jinglin Zhang, Yuquan Wu*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25638](https://doi.org/10.1609/aaai.v37i4.25638)

        **Abstract**:

        Accurate forecasting of tropical cyclone (TC) plays a critical role in the prevention and defense of TC disasters. We must explore a more accurate method for TC prediction. Deep learning methods are increasingly being implemented to make TC prediction more accurate. However, most existing methods lack a generic framework for adapting heterogeneous meteorological data and do not focus on the importance of the environment. Therefore, we propose a Multi-Generator Tropical Cyclone Forecasting model (MGTCF), a generic, extensible, multi-modal TC prediction model with the key modules of Generator Chooser Network (GC-Net) and Environment Net (Env-Net). The proposed method can utilize heterogeneous meteorologic data efficiently and mine environmental factors. In addition, the Multi-generator with Generator Chooser Net is proposed to tackle the drawbacks of single-generator TC prediction methods: the prediction of undesired out-of-distribution samples and the problems stemming from insufficient learning ability. To prove the effectiveness of MGTCF, we conduct extensive experiments on the China Meteorological Administration Tropical Cyclone Best Track Dataset. MGTCF obtains better performance compared with other deep learning methods and outperforms the official prediction method of the China Central Meteorological Observatory in most indexes.

        ----

        ## [569] MDM: Molecular Diffusion Model for 3D Molecule Generation

        **Authors**: *Lei Huang, Hengtong Zhang, Tingyang Xu, Ka-Chun Wong*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25639](https://doi.org/10.1609/aaai.v37i4.25639)

        **Abstract**:

        Molecule generation, especially generating 3D molecular geometries from scratch (i.e., 3D de novo generation), has become a fundamental task in drug design. Existing diffusion based 3D molecule generation methods could suffer from unsatisfactory performances, especially when generating large molecules. At the same time, the generated molecules lack enough diversity. This paper proposes a novel diffusion model to address those two challenges. 

First, interatomic relations are not included in molecules' 3D point cloud representations. Thus, it is difficult for existing generative models to capture the potential interatomic forces and abundant local constraints. 
To tackle this challenge, we propose to augment the potential interatomic forces and further involve dual equivariant encoders to encode interatomic forces of different strengths.
Second, existing diffusion-based models essentially shift elements in geometry along the gradient of data density. Such a process lacks enough exploration in the intermediate steps of the Langevin dynamics. To address this issue, we introduce a distributional controlling variable in each diffusion/reverse step to enforce thorough explorations and further improve generation diversity.

Extensive experiments on multiple benchmarks demonstrate that the proposed model significantly outperforms existing methods for both unconditional and conditional generation tasks. We also conduct case studies to help understand the physicochemical properties of the generated molecules. The codes are available at https://github.com/tencent-ailab/MDM.

        ----

        ## [570] Learning Chemical Rules of Retrosynthesis with Pre-training

        **Authors**: *Yinjie Jiang, Ying Wei, Fei Wu, Zhengxing Huang, Kun Kuang, Zhihua Wang*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25640](https://doi.org/10.1609/aaai.v37i4.25640)

        **Abstract**:

        Retrosynthesis aided by artificial intelligence has been a very active and bourgeoning area of research, for its critical role in drug discovery as well as material science. Three categories of solutions, i.e., template-based, template-free, and semi-template methods, constitute mainstream solutions to this problem. In this paper, we focus on template-free methods which are known to be less bothered by the template generalization issue and the atom mapping challenge. Among several remaining problems regarding template-free methods, failing to conform to chemical rules is pronounced. To address the issue, we seek for a pre-training solution to empower the pre-trained model with chemical rules encoded. Concretely, we enforce the atom conservation rule via a molecule reconstruction pre-training task, and the reaction rule that dictates reaction centers via a reaction type guided contrastive pre-training task. In our empirical evaluation, the proposed pre-training solution substantially improves the single-step retrosynthesis accuracies in three downstream datasets.

        ----

        ## [571] Online Symbolic Regression with Informative Query

        **Authors**: *Pengwei Jin, Di Huang, Rui Zhang, Xing Hu, Ziyuan Nan, Zidong Du, Qi Guo, Yunji Chen*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25641](https://doi.org/10.1609/aaai.v37i4.25641)

        **Abstract**:

        Symbolic regression, the task of extracting mathematical expressions from the observed data, plays a crucial role in scientific discovery. Despite the promising performance of existing methods, most of them conduct symbolic regression in an offline setting. That is, they treat the observed data points as given ones that are simply sampled from uniform distributions without exploring the expressive potential of data. However, for real-world scientific problems, the data used for symbolic regression are usually actively obtained by doing experiments, which is an online setting. Thus, how to obtain informative data that can facilitate the symbolic regression process is an important problem that remains challenging. 

In this paper, we propose QUOSR, a query-based framework for online symbolic regression that can automatically obtain informative data in an iterative manner. Specifically, at each step, QUOSR receives historical data points, generates new x, and then queries the symbolic expression to get the corresponding y, where the (x, y) serves as new data points. This process repeats until the maximum number of query steps is reached. To make the generated data points informative, we implement the framework with a neural network and train it by maximizing the mutual information between generated data points and the target expression. Through comprehensive experiments, we show that QUOSR can facilitate modern symbolic regression methods by generating informative data.

        ----

        ## [572] Repair Is Nearly Generation: Multilingual Program Repair with LLMs

        **Authors**: *Harshit Joshi, José Pablo Cambronero Sánchez, Sumit Gulwani, Vu Le, Gust Verbruggen, Ivan Radicek*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25642](https://doi.org/10.1609/aaai.v37i4.25642)

        **Abstract**:

        Most programmers make mistakes when writing code. Some of these mistakes are small and require few edits to the original program – a class of errors recently termed last mile mistakes. These errors break the flow for experienced developers and can stump novice programmers. Existing automated repair techniques targeting this class of errors are language-specific and do not easily carry over to new languages. Transferring symbolic approaches requires substantial engineering and neural approaches require data and retraining. We introduce RING, a multilingual repair engine powered by a large language model trained on code (LLMC) such as Codex. Such a multilingual engine enables a flipped model for programming assistance, one where the programmer writes code and the AI assistance suggests fixes, compared to traditional code suggestion technology. Taking inspiration from the way programmers manually fix bugs, we show that a prompt-based strategy that conceptualizes repair as localization, transformation, and candidate ranking, can successfully repair programs in multiple languages with minimal effort. We present the first results for such a multilingual repair engine by evaluating on 6 different languages and comparing performance to language-specific repair engines. We show that RING can outperform language-specific repair engines for three of these languages.

        ----

        ## [573] Heterogeneous Graph Learning for Multi-Modal Medical Data Analysis

        **Authors**: *Sein Kim, Namkyeong Lee, Junseok Lee, Dongmin Hyun, Chanyoung Park*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25643](https://doi.org/10.1609/aaai.v37i4.25643)

        **Abstract**:

        Routine clinical visits of a patient produce not only image data, but also non-image data containing clinical information regarding the patient, i.e., medical data is multi-modal in nature. Such heterogeneous modalities offer different and complementary perspectives on the same patient, resulting in more accurate clinical decisions when they are properly combined. However, despite its significance, how to effectively fuse the multi-modal medical data into a unified framework has received relatively little attention. In this paper, we propose an effective graph-based framework called HetMed (Heterogeneous Graph Learning for Multi-modal Medical Data Analysis) for fusing the multi-modal medical data.
Specifically, we construct a multiplex network that incorporates multiple types of non-image features of patients to capture the complex relationship between patients in a systematic way, which leads to more accurate clinical decisions. Extensive experiments on various real-world datasets demonstrate the superiority and practicality of HetMed. The source code for HetMed is available at https://github.com/Sein-Kim/Multimodal-Medical.

        ----

        ## [574] Rolling Horizon Based Temporal Decomposition for the Offline Pickup and Delivery Problem with Time Windows

        **Authors**: *Youngseo Kim, Danushka Edirimanna, Michael Wilbur, Philip Pugliese, Aron Laszka, Abhishek Dubey, Samitha Samaranayake*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25644](https://doi.org/10.1609/aaai.v37i4.25644)

        **Abstract**:

        The offline pickup and delivery problem with time windows (PDPTW) is a classical combinatorial optimization problem in the transportation community, which has proven to be very challenging computationally. Due to the complexity of the problem, practical problem instances can be solved only via heuristics, which trade-off solution quality for computational tractability. Among the various heuristics, a common strategy is problem decomposition, that is, the reduction of a large-scale problem into a collection of smaller sub-problems, with spatial and temporal decompositions being two natural approaches. While spatial decomposition has been successful in certain settings, effective temporal decomposition has been challenging due to the difficulty of stitching together the sub-problem solutions across the decomposition boundaries. In this work, we introduce a novel temporal decomposition scheme for solving a class of PDPTWs that have narrow time windows, for which it is able to provide both fast and high-quality solutions. We utilize techniques that have been popularized recently in the context of online dial-a-ride problems along with the general idea of rolling horizon optimization. To the best of our knowledge, this is the first attempt to solve offline PDPTWs using such an approach. To show the performance and scalability of our framework, we use the optimization of paratransit services as a motivating example. Due to the lack of benchmark solvers similar to ours (i.e., temporal decomposition with an online solver), we compare our results with an offline heuristic algorithm using Google OR-Tools. In smaller problem instances (with an average of 129 requests per instance), the baseline approach is as competitive as our framework. However, in larger problem instances (approximately 2,500 requests per instance), our framework is more scalable and can provide good solutions to problem instances of varying degrees of difficulty, while the baseline algorithm often fails to find a feasible solution within comparable compute times.

        ----

        ## [575] GRIP: Graph Representation of Immune Repertoire Using Graph Neural Network and Transformer

        **Authors**: *Yongju Lee, Hyunho Lee, Kyoungseob Shin, Sunghoon Kwon*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25645](https://doi.org/10.1609/aaai.v37i4.25645)

        **Abstract**:

        The immune repertoire is a collection of immune recep-tors that has emerged as an important biomarker for both diagnostic and therapeutic of cancer patients. In terms of deep learning, analyzing immune repertoire is a challeng-ing multiple-instance learning problem in which the im-mune repertoire of an individual is a bag, and the immune receptor is an instance. Although several deep learning methods for immune repertoire analysis are introduced, they consider the immune repertoire as a set-like struc-ture that doesn’t take account of the nature of the im-mune response. When the immune response occurs, mu-tations are introduced to the immune receptor sequence sequentially to optimize the immune response against the pathogens that enter our body. As a result, immune receptors for the specific pathogen have the lineage of evolution; thus, immune repertoire is better represented as a graph-like structure. In this work, we present our novel method graph representation of immune repertoire (GRIP), which analyzes the immune repertoire as a hier-archical graph structure and utilize the collection of graph neural network followed by graph pooling and transformer to efficiently represents the immune reper-toire as an embedding vector. We show that GRIP predict the survival probability of cancer patients better than the set-based methods and graph-based structure is critical for performance. Also, GRIP provides interpretable re-sults, which prove that GRIP adequately use the progno-sis-related immune receptor and give further possibility to use the GRIP as the novel biomarker searching tool

        ----

        ## [576] LagNet: Deep Lagrangian Mechanics for Plug-and-Play Molecular Representation Learning

        **Authors**: *Chunyan Li, Junfeng Yao, Jinsong Su, Zhaoyang Liu, Xiangxiang Zeng, Chenxi Huang*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25646](https://doi.org/10.1609/aaai.v37i4.25646)

        **Abstract**:

        Molecular representation learning is a fundamental problem in the field of drug discovery and molecular science. Whereas incorporating molecular 3D information in the representations of molecule seems beneficial, which is related to computational chemistry with the basic task of predicting stable 3D structures (conformations) of molecules. Existing machine learning methods either rely on 1D and 2D molecular properties or simulate molecular force field to use additional 3D structure information via Hamiltonian network. The former has the disadvantage of ignoring important 3D structure features, while the latter has the disadvantage that existing Hamiltonian neural network must satisfy the “canonial” constraint, which is difficult to be obeyed in many cases. In this paper, we propose a novel plug-and-play architecture LagNet by simulating molecular force field only with parameterized position coordinates, which implements Lagrangian mechanics to learn molecular representation by preserving 3D conformation without obeying any additional restrictions. LagNet is designed to generate known conformations and generalize for unknown ones from molecular SMILES. Implicit positions in LagNet are learned iteratively using discrete-time Lagrangian equations. Experimental results show that LagNet can well learn 3D molecular structure features, and outperforms previous state-of-the-art baselines related molecular representation by a significant margin.

        ----

        ## [577] Steganography of Steganographic Networks

        **Authors**: *Guobiao Li, Sheng Li, Meiling Li, Xinpeng Zhang, Zhenxing Qian*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25647](https://doi.org/10.1609/aaai.v37i4.25647)

        **Abstract**:

        Steganography is a technique for covert communication between two parties. With the rapid development of deep neural networks (DNN), more and more steganographic networks are proposed recently, which are shown to be promising to achieve good performance. Unlike the traditional handcrafted steganographic tools, a steganographic network is relatively large in size. It raises concerns on how to covertly transmit the steganographic network in public channels, which is a crucial stage in the pipeline of steganography in real world applications. To address such an issue, we propose a novel scheme for steganography of steganographic networks in this paper. Unlike the existing steganographic schemes which focus on the subtle modification of the cover data to accommodate the secrets. We propose to disguise a steganographic network (termed as the secret DNN model) into a stego DNN model which performs an ordinary machine learning task (termed as the stego task). During the model disguising, we select and tune a subset of filters in the secret DNN model to preserve its function on the secret task, where the remaining filters are reactivated according to a partial optimization strategy to disguise the whole secret DNN model into a stego DNN model. The secret DNN model can be recovered from the stego DNN model when needed. Various experiments have been conducted to demonstrate the advantage of our proposed method for covert communication of steganographic networks as well as general DNN models.

        ----

        ## [578] PEN: Prediction-Explanation Network to Forecast Stock Price Movement with Better Explainability

        **Authors**: *Shuqi Li, Weiheng Liao, Yuhan Chen, Rui Yan*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25648](https://doi.org/10.1609/aaai.v37i4.25648)

        **Abstract**:

        Nowadays explainability in stock price movement prediction is attracting increasing attention in banks, hedge funds and asset managers, primarily due to audit or regulatory reasons. Text data such as financial news and social media posts can be part of the reasons for stock price movement. To this end, we propose a novel framework of Prediction-Explanation Network (PEN) jointly modeling text streams and price streams with alignment. The key component of the PEN model is an shared representation learning module that learns which texts are possibly associated with the stock price movement by modeling the interaction between the text data and stock price data with a salient vector characterizing their correlation. In this way, the PEN model is able to predict the stock price movement by identifying and utilizing abundant messages while on the other hand, the selected text messages also explain the stock price movement. Experiments on real-world datasets demonstrate that we are able to kill two birds with one stone: in terms of accuracy, the proposed PEN model outperforms the state-of-art baseline; on explainability, the PEN model are demonstrated to be far superior to attention mechanism, capable of picking out the crucial texts with a very high confidence.

        ----

        ## [579] Decision-Making Context Interaction Network for Click-Through Rate Prediction

        **Authors**: *Xiang Li, Shuwei Chen, Jian Dong, Jin Zhang, Yongkang Wang, Xingxing Wang, Dong Wang*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25649](https://doi.org/10.1609/aaai.v37i4.25649)

        **Abstract**:

        Click-through rate (CTR) prediction is crucial in recommendation and online advertising systems. Existing methods usually model user behaviors, while ignoring the informative context which influences the user to make a click decision, e.g., click pages and pre-ranking candidates that inform inferences about user interests, leading to suboptimal performance. In this paper, we propose a Decision-Making Context Interaction Network (DCIN), which deploys a carefully designed Context Interaction Unit (CIU) to learn decision-making contexts and thus benefits CTR prediction. In addition, the relationship between different decision-making context sources is explored by the proposed Adaptive Interest Aggregation Unit (AIAU) to improve CTR prediction further. In the experiments on public and industrial datasets, DCIN significantly outperforms the state-of-the-art methods. Notably, the model has obtained the improvement of CTR+2.9%/CPM+2.1%/GMV+1.5% for online A/B testing and served the main traffic of Meituan Waimai advertising system.

        ----

        ## [580] Fine-Grained Position Helps Memorizing More, a Novel Music Compound Transformer Model with Feature Interaction Fusion

        **Authors**: *Zuchao Li, Ruhan Gong, Yineng Chen, Kehua Su*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25650](https://doi.org/10.1609/aaai.v37i4.25650)

        **Abstract**:

        Due to the particularity of the simultaneous occurrence of multiple events in music sequences, compound Transformer is proposed to deal with the challenge of long sequences. However, there are two deficiencies in the compound Transformer. First, since the order of events is more important for music than natural language, the information provided by the original absolute position embedding is not precise enough. Second, there is an important correlation between the tokens in the compound word, which is ignored by the current compound Transformer. Therefore, in this work, we propose an improved compound Transformer model for music understanding. Specifically, we propose an attribute embedding fusion module  and a novel position encoding scheme with absolute-relative consideration. In the attribute embedding fusion module, different attributes are fused through feature permutation by using a multi-head self-attention mechanism in order to capture rich interactions between attributes.
In the novel position encoding scheme, we propose RoAR position encoding, which realizes rotational absolute position encoding, relative position encoding, and absolute-relative position interactive encoding, providing clear and rich orders for musical events. 
Empirical study on four typical music understanding tasks shows that our attribute fusion approach and RoAR position encoding brings large performance gains. In addition, we further investigate the impact of masked language modeling and casual language modeling  pre-training on music understanding.

        ----

        ## [581] Zero-Shot Rumor Detection with Propagation Structure via Prompt Learning

        **Authors**: *Hongzhan Lin, Pengyao Yi, Jing Ma, Haiyun Jiang, Ziyang Luo, Shuming Shi, Ruifang Liu*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25651](https://doi.org/10.1609/aaai.v37i4.25651)

        **Abstract**:

        The spread of rumors along with breaking events seriously hinders the truth in the era of social media. Previous studies reveal that due to the lack of annotated resources, rumors presented in minority languages are hard to be detected. Furthermore, the unforeseen breaking events not involved in yesterday's news exacerbate the scarcity of data resources. In this work, we propose a novel zero-shot framework based on prompt learning to detect rumors falling in different domains or presented in different languages. More specifically, we firstly represent rumor circulated on social media as diverse propagation threads, then design a hierarchical prompt encoding mechanism to learn language-agnostic contextual representations for both prompts and rumor data. To further enhance domain adaptation, we model the domain-invariant structural features from the propagation threads, to incorporate structural position representations of influential community response. In addition, a new virtual response augmentation method is used to improve model training. Extensive experiments conducted on three real-world datasets demonstrate that our proposed model achieves much better performance than state-of-the-art methods and exhibits a superior capacity for detecting rumors at early stages.

        ----

        ## [582] On Manipulating Weight Predictions in Signed Weighted Networks

        **Authors**: *Tomasz Lizurej, Tomasz Michalak, Stefan Dziembowski*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25652](https://doi.org/10.1609/aaai.v37i4.25652)

        **Abstract**:

        Adversarial social network analysis studies how graphs can be rewired or otherwise manipulated to evade social network analysis tools. While there is ample literature on manipulating simple networks, more sophisticated network types are much less understood in this respect. In this paper, we focus on the problem of evading FGA---an edge weight prediction method for signed weighted networks by Kumar et al. 2016. Among others, this method can be used for trust prediction in reputation systems. We study the theoretical underpinnings of FGA and its computational properties in terms of manipulability. Our positive finding is that, unlike many other tools, this measure is not only difficult to manipulate optimally, but also it can be difficult to manipulate in practice.

        ----

        ## [583] Better Context Makes Better Code Language Models: A Case Study on Function Call Argument Completion

        **Authors**: *Hengzhi Pei, Jinman Zhao, Leonard Lausen, Sheng Zha, George Karypis*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25653](https://doi.org/10.1609/aaai.v37i4.25653)

        **Abstract**:

        Pretrained code language models have enabled great progress towards program synthesis. However, common approaches only consider in-file local context and thus miss information and constraints imposed by other parts of the codebase and its external dependencies. Existing code completion benchmarks also lack such context. To resolve these restrictions we curate a new dataset of permissively licensed Python packages that includes full projects and their dependencies and provide tools to extract non-local information with the help of program analyzers. We then focus on the task of function call argument completion which requires predicting the arguments to function calls. We show that existing code completion models do not yield good results on our completion task. To better solve this task, we query a program analyzer for information relevant to a given function call, and consider ways to provide the analyzer results to different code completion models during inference and training. Our experiments show that providing access to the function implementation and function usages greatly improves the argument completion performance. Our ablation study provides further insights on how different types of information available from the program analyzer and different ways of incorporating the information affect the model performance.

        ----

        ## [584] MetaTPTrans: A Meta Learning Approach for Multilingual Code Representation Learning

        **Authors**: *Weiguo Pian, Hanyu Peng, Xunzhu Tang, Tiezhu Sun, Haoye Tian, Andrew Habib, Jacques Klein, Tegawendé F. Bissyandé*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25654](https://doi.org/10.1609/aaai.v37i4.25654)

        **Abstract**:

        Representation learning of source code is essential for applying machine learning to software engineering tasks. Learning code representation from a multilingual source code dataset has been shown to be more effective than learning from single-language datasets separately, since more training data from multilingual dataset improves the model's ability to extract language-agnostic information from source code. However, existing multilingual training overlooks the language-specific information which is crucial for modeling source code across different programming languages, while only focusing on learning a unified model with shared parameters among different languages for language-agnostic information modeling. To address this problem, we propose MetaTPTrans, a meta learning approach for multilingual code representation learning. MetaTPTrans generates different parameters for the feature extractor according to the specific programming language type of the input code snippet, enabling the model to learn both language-agnostic and language-specific information with dynamic parameters in the feature extractor. We conduct experiments on the code summarization and code completion tasks to verify the effectiveness of our approach. The results demonstrate the superiority of our approach with significant improvements on state-of-the-art baselines.

        ----

        ## [585] HG-SL: Jointly Learning of Global and Local User Spreading Behavior for Fake News Early Detection

        **Authors**: *Ling Sun, Yuan Rao, Yuqian Lan, Bingcan Xia, Yangyang Li*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25655](https://doi.org/10.1609/aaai.v37i4.25655)

        **Abstract**:

        Recently, fake news forgery technology has become more and more sophisticated, and even the profiles of participants may be faked, which challenges the robustness and effectiveness of traditional detection methods involving text or user identity. Most propagation-only approaches mainly rely on neural networks to learn the diffusion pattern of individual news, which is insufficient to describe the differences in news spread ability, and also ignores the valuable global connections of news and users, limiting the performance of detection. Therefore, we propose a joint learning model named HG-SL, which is blind to news content and user identities, but capable of catching the differences between true and fake news in the early stages of propagation through global and local user spreading behavior. Specifically, we innovatively design a Hypergraph-based Global interaction learning module to capture the global preferences of users from their co-spreading relationships, and introduce node centrality encoding to complement user influence in hypergraph learning. Moreover, the designed Self-attention-based Local context learning module first introduce spread status to highlight the propagation ability of news and users, thus providing additional signals for verifying news authenticity. Experiments on real-world datasets indicate that our HG-SL, which solely relies on user behavior, outperforms SOTA baselines utilizing multidimensional features in both fake news detection and early detection task.

        ----

        ## [586] Defending against Backdoor Attacks in Natural Language Generation

        **Authors**: *Xiaofei Sun, Xiaoya Li, Yuxian Meng, Xiang Ao, Lingjuan Lyu, Jiwei Li, Tianwei Zhang*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25656](https://doi.org/10.1609/aaai.v37i4.25656)

        **Abstract**:

        The frustratingly fragile nature of neural network models make current natural  language generation (NLG) systems prone to backdoor attacks and generate malicious sequences that could be sexist or offensive. Unfortunately, little effort has been invested to how backdoor attacks can affect current NLG models and how to defend against these attacks. In this work, by giving a formal definition of backdoor attack and defense, we investigate this problem on two important NLG tasks, machine translation and dialog generation. Tailored to the inherent nature of NLG models (e.g., producing a sequence of coherent words given contexts), we design defending strategies against attacks. 
We find that testing the backward probability of generating sources given targets yields  effective defense performance against all different types of attacks, and is able to handle the one-to-many issue in many NLG tasks such as dialog generation. We hope that this work can raise the awareness of backdoor risks concealed in deep NLG systems and inspire more future work (both attack and defense) towards this direction.

        ----

        ## [587] GenéLive! Generating Rhythm Actions in Love Live!

        **Authors**: *Atsushi Takada, Daichi Yamazaki, Yudai Yoshida, Nyamkhuu Ganbat, Takayuki Shimotomai, Naoki Hamada, Likun Liu, Taiga Yamamoto, Daisuke Sakurai*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25657](https://doi.org/10.1609/aaai.v37i4.25657)

        **Abstract**:

        This article presents our generative model for rhythm action games together with applications in business operation. Rhythm action games are video games in which the player is challenged to issue commands at the right timings during a music session. The timings are rendered in the chart, which consists of visual symbols, called notes, flying through the screen. We introduce our deep generative model, GenéLive!, which outperforms the state-of-the-art model by taking into account musical structures through beats and temporal scales. Thanks to its favorable performance, GenéLive! was put into operation at KLab Inc., a Japan-based video game developer, and reduced the business cost of chart generation by as much as half. The application target included the phenomenal "Love Live!", which has more than 10 million users across Asia and beyond, and is one of the few rhythm action franchises that has led the online era of the genre. In this article, we evaluate the generative performance of GenéLive! using production datasets at KLab as well as open datasets for reproducibility, while the model continues to operate in their business. Our code and the model, tuned and trained using a supercomputer, are publicly available.

        ----

        ## [588] Deepfake Video Detection via Facial Action Dependencies Estimation

        **Authors**: *Lingfeng Tan, Yunhong Wang, Junfu Wang, Liang Yang, Xunxun Chen, Yuanfang Guo*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25658](https://doi.org/10.1609/aaai.v37i4.25658)

        **Abstract**:

        Deepfake video detection has drawn significant attention from researchers due to the security issues induced by deepfake videos. Unfortunately, most of the existing deepfake detection approaches have not competently modeled the natural structures and movements of human faces. In this paper, we formulate the deepfake video detection problem into a graph classification task, and propose a novel paradigm named Facial Action Dependencies Estimation (FADE) for deepfake video detection. We propose a Multi-Dependency Graph Module (MDGM) to capture abundant dependencies among facial action units, and extracts subtle clues in these dependencies. MDGM can be easily integrated into the existing frame-level detection schemes to provide significant performance gains. Extensive experiments demonstrate the superiority of our method against the state-of-the-art methods.

        ----

        ## [589] Contrastive Attention Networks for Attribution of Early Modern Print

        **Authors**: *Nikolai Vogler, Kartik Goyal, Kishore PV Reddy, Elizaveta Pertseva, Samuel V. Lemley, Christopher N. Warren, Max G'Sell, Taylor Berg-Kirkpatrick*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25659](https://doi.org/10.1609/aaai.v37i4.25659)

        **Abstract**:

        In this paper, we develop machine learning techniques to identify unknown printers in early modern (c.~1500--1800) English printed books.
Specifically, we focus on matching uniquely damaged character type-imprints in anonymously printed books to works with known printers in order to provide evidence of their origins.
Until now, this work has been limited to manual investigations by analytical bibliographers.
We present a Contrastive Attention-based Metric Learning approach to identify similar damage across character image pairs, which is sensitive to very subtle differences in glyph shapes, yet robust to various confounding sources of noise associated with digitized historical books. 
To overcome the scarce amount of supervised data, we design a random data synthesis procedure that aims to simulate bends, fractures, and inking variations induced by the early printing process.
Our method successfully improves downstream damaged type-imprint matching among printed works from this period, as validated by in-domain human experts. The results of our approach on two important philosophical works from the Early Modern period demonstrate potential to extend the extant historical research about the origins and content of these books.

        ----

        ## [590] AdapSafe: Adaptive and Safe-Certified Deep Reinforcement Learning-Based Frequency Control for Carbon-Neutral Power Systems

        **Authors**: *Xu Wan, Mingyang Sun, Boli Chen, Zhongda Chu, Fei Teng*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25660](https://doi.org/10.1609/aaai.v37i4.25660)

        **Abstract**:

        With the increasing penetration of inverter-based renewable energy resources, deep reinforcement learning (DRL) has been proposed as one of the most promising solutions to realize real-time and autonomous control for future carbon-neutral power systems. In particular, DRL-based frequency control approaches have been extensively investigated to overcome the limitations of model-based approaches, such as the computational cost and scalability for large-scale systems. Nevertheless, the real-world implementation of DRLbased frequency control methods is facing the following fundamental challenges: 1) safety guarantee during the learning and decision-making processes; 2) adaptability against the dynamic system operating conditions. To this end, this is the first work that proposes an Adaptive and Safe-Certified DRL (AdapSafe) algorithm for frequency control to simultaneously address the aforementioned challenges. In particular, a novel self-tuning control barrier function is designed to actively compensate the unsafe frequency control strategies under variational safety constraints and thus achieve guaranteed safety. Furthermore, the concept of meta-reinforcement learning is integrated to significantly enhance its adaptiveness in non-stationary power system environments without sacrificing the safety cost. Experiments are conducted based on GB 2030 power system, and the results demonstrate that the proposed AdapSafe exhibits superior performance in terms of its guaranteed safety in both training and test phases, as well as its considerable adaptability against the dynamics changes of system parameters.

        ----

        ## [591] Don't Predict Counterfactual Values, Predict Expected Values Instead

        **Authors**: *Jeremiasz Wolosiuk, Maciej Swiechowski, Jacek Mandziuk*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25661](https://doi.org/10.1609/aaai.v37i4.25661)

        **Abstract**:

        Counterfactual Regret Minimization algorithms are the most popular way of estimating the Nash Equilibrium in imperfect-information zero-sum games.
In particular, DeepStack -- the state-of-the-art Poker bot -- employs the so-called Deep Counterfactual Value Network (DCVN) to learn the Counterfactual Values (CFVs) associated with various states in the game.
Each CFV is a multiplication of two factors: (1) the probability that the opponent would reach a given state in a game, which can be explicitly calculated from the input data, and (2) the expected value (EV) of a payoff in that state, which is a complex function of the input data, hard to calculate.
In this paper, we propose a simple yet powerful modification to the CFVs estimation process, which consists in utilizing a deep neural network to estimate only the EV factor of CFV. This new target setting significantly simplifies the learning problem and leads to much more accurate CFVs estimation. 
A direct comparison, in terms of CFVs prediction losses, shows a significant prediction accuracy improvement of the proposed approach (DEVN) over the original DCVN formulation (relatively by 9.18-15.70% when using card abstraction, and by 3.37-8.39% without card abstraction, depending on a particular setting). 
Furthermore, the application of DEVN improves the theoretical lower bound of the error by 29.05-31.83% compared to the DCVN pipeline when card abstraction is applied.
Additionally, DEVN is able to achieve the goal using significantly smaller, and faster to infer, networks.
While the proposed modification may seem to be of a rather technical nature, it, in fact, presents a fundamentally different approach to the overall process of learning and estimating CFVs, since the distributions of the training signals differ significantly between DCVN and DEVN. The former estimates CFVs, which are biased by the probability of reaching a given game state, while training the latter relies on a direct EV estimation, regardless of the state probability. In effect, the learning signal of DEVN presents a better estimation of the true value of a given state, thus allowing more accurate CFVs estimation.

        ----

        ## [592] Molformer: Motif-Based Transformer on 3D Heterogeneous Molecular Graphs

        **Authors**: *Fang Wu, Dragomir Radev, Stan Z. Li*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25662](https://doi.org/10.1609/aaai.v37i4.25662)

        **Abstract**:

        Procuring expressive molecular representations underpins AI-driven molecule design and scientific discovery. The research mainly focuses on atom-level homogeneous molecular graphs, ignoring the rich information in subgraphs or motifs. However, it has been widely accepted that substructures play a dominant role in identifying and determining molecular properties. To address such issues, we formulate heterogeneous molecular graphs (HMGs) and introduce a novel architecture to exploit both molecular motifs and 3D geometry. Precisely, we extract functional groups as motifs for small molecules and employ reinforcement learning to adaptively select quaternary amino acids as motif candidates for proteins. Then HMGs are constructed with both atom-level and motif-level nodes. To better accommodate those HMGs, we introduce a variant of the Transformer named Molformer, which adopts a heterogeneous self-attention layer to distinguish the interactions between multi-level nodes. Besides, it is also coupled with a multi-scale mechanism to capture fine-grained local patterns with increasing contextual scales. An attentive farthest point sampling algorithm is also proposed to obtain the molecular representations. We validate Molformer across a broad range of domains, including quantum chemistry, physiology, and biophysics. Extensive experiments show that Molformer outperforms or achieves the comparable performance of several state-of-the-art baselines. Our work provides a promising way to utilize informative motifs from the perspective of multi-level graph construction. The code is available at https://github.com/smiles724/Molformer.

        ----

        ## [593] DiffMD: A Geometric Diffusion Model for Molecular Dynamics Simulations

        **Authors**: *Fang Wu, Stan Z. Li*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25663](https://doi.org/10.1609/aaai.v37i4.25663)

        **Abstract**:

        Molecular dynamics (MD) has long been the de facto choice for simulating complex atomistic systems from first principles. Recently deep learning models become a popular way to accelerate MD. Notwithstanding, existing models depend on intermediate variables such as the potential energy or force fields to update atomic positions, which requires additional computations to perform back-propagation. To waive this requirement, we propose a novel model called DiffMD by directly estimating the gradient of the log density of molecular conformations. DiffMD relies on a score-based denoising diffusion generative model that perturbs the molecular structure with a conditional noise depending on atomic accelerations and treats conformations at previous timeframes as the prior distribution for sampling. Another challenge of modeling such a conformation generation process is that a molecule is kinetic instead of static, which no prior works have strictly studied. To solve this challenge, we propose an equivariant geometric Transformer as the score function in the diffusion process to calculate corresponding gradients. It incorporates the directions and velocities of atomic motions via 3D spherical Fourier-Bessel representations. With multiple architectural improvements, we outperform state-of-the-art baselines on MD17 and isomers of C7O2H10 datasets. This work contributes to accelerating material and drug discovery.

        ----

        ## [594] Retrosynthesis Prediction with Local Template Retrieval

        **Authors**: *Shufang Xie, Rui Yan, Junliang Guo, Yingce Xia, Lijun Wu, Tao Qin*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25664](https://doi.org/10.1609/aaai.v37i4.25664)

        **Abstract**:

        Retrosynthesis, which predicts the reactants of a given target molecule, is an essential task for drug discovery. In recent years, the machine learing based retrosynthesis methods have achieved promising results. In this work, we introduce RetroKNN, a local reaction template retrieval method to further boost the performance of template-based systems with non-parametric retrieval. We first build an atom-template store and a bond-template store that contains the local templates in the training data, then retrieve from these templates with a k-nearest-neighbor (KNN) search during inference. The retrieved templates are combined with neural network predictions as the final output. Furthermore, we propose a lightweight adapter to adjust the weights when combing neural network and KNN predictions conditioned on the hidden representation and the retrieved templates. We conduct comprehensive experiments on two widely used benchmarks, the USPTO-50K and USPTO-MIT. Especially for the top-1 accuracy, we improved 7.1% on the USPTO-50K dataset and 12.0% on the USPTO-MIT dataset.These results demonstrate the effectiveness of our method.

        ----

        ## [595] Multi-Relational Contrastive Learning Graph Neural Network for Drug-Drug Interaction Event Prediction

        **Authors**: *Zhankun Xiong, Shichao Liu, Feng Huang, Ziyan Wang, Xuan Liu, Zhongfei Zhang, Wen Zhang*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25665](https://doi.org/10.1609/aaai.v37i4.25665)

        **Abstract**:

        Drug-drug interactions (DDIs) could lead to various unexpected adverse consequences, so-called DDI events. Predicting DDI events can reduce the potential risk of combinatorial therapy and improve the safety of medication use, and has attracted much attention in the deep learning community. Recently, graph neural network (GNN)-based models have aroused broad interest and achieved satisfactory results in the DDI event prediction. Most existing GNN-based models ignore either drug structural information or drug interactive information, but both aspects of information are important for DDI event prediction. Furthermore, accurately predicting rare DDI events is hindered by their inadequate labeled instances. In this paper, we propose a new method, Multi-Relational Contrastive learning Graph Neural Network, MRCGNN for brevity, to predict DDI events. Specifically, MRCGNN integrates the two aspects of information by deploying a GNN on the multi-relational DDI event graph attributed with the drug features extracted from drug molecular graphs. Moreover, we implement a multi-relational graph contrastive learning with a designed dual-view negative counterpart augmentation strategy, to capture implicit information about rare DDI events. Extensive experiments on two datasets show that MRCGNN outperforms the state-of-the-art methods. Besides, we observe that MRCGNN achieves satisfactory performance when predicting rare DDI events.

        ----

        ## [596] Tighter Robust Upper Bounds for Options via No-Regret Learning

        **Authors**: *Shan Xue, Ye Du, Liang Xu*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25666](https://doi.org/10.1609/aaai.v37i4.25666)

        **Abstract**:

        Classic option pricing models, such as the Black-Scholes formula, often depend on some rigid assumptions on the dynamics of the underlying asset prices. These assumptions are inevitably violated in practice and thus induce the model risk. To mitigate this, robust option pricing that only requires the no-arbitrage principle has attracted a great deal of attention among researchers. In this paper, we give new robust upper bounds for option prices based on a novel η-momentum trading strategy. Our bounds for European options are tighter for most common moneyness, volatility, and expiration date setups than those presented in the existing literature. Our bounds for average strike Asian options are the first closed-form robust upper bounds for those options. Numerical simulations demonstrate that our bounds significantly outperform the benchmarks for both European and Asian options.

        ----

        ## [597] KerPrint: Local-Global Knowledge Graph Enhanced Diagnosis Prediction for Retrospective and Prospective Interpretations

        **Authors**: *Kai Yang, Yongxin Xu, Peinie Zou, Hongxin Ding, Junfeng Zhao, Yasha Wang, Bing Xie*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25667](https://doi.org/10.1609/aaai.v37i4.25667)

        **Abstract**:

        While recent developments of deep learning models have led to record-breaking achievements in many areas, the lack of sufficient interpretation remains a problem for many specific applications, such as the diagnosis prediction task in healthcare. The previous knowledge graph(KG) enhanced approaches mainly focus on learning clinically meaningful representations, the importance of medical concepts, and even the knowledge paths from inputs to labels. However, it is infeasible to interpret the diagnosis prediction, which needs to consider different medical concepts, various medical relationships, and the time-effectiveness of knowledge triples in different patient contexts. More importantly, the retrospective and prospective interpretations of disease processes are valuable to clinicians for the patients' confounding diseases. We propose KerPrint, a novel KG enhanced approach for retrospective and prospective interpretations to tackle these problems. Specifically, we propose a time-aware KG attention method to solve the problem of knowledge decay over time for trustworthy retrospective interpretation. We also propose a novel element-wise attention method to select candidate global knowledge using comprehensive representations from the local KG for prospective interpretation. We validate the effectiveness of our KerPrint through an extensive experimental study on a real-world dataset and a public dataset. The results show that our proposed approach not only achieves significant improvement over knowledge-enhanced methods but also gives the interpretability of diagnosis prediction in both retrospective and prospective views.

        ----

        ## [598] Multi-Label Few-Shot ICD Coding as Autoregressive Generation with Prompt

        **Authors**: *Zhichao Yang, Sunjae Kwon, Zonghai Yao, Hong Yu*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25668](https://doi.org/10.1609/aaai.v37i4.25668)

        **Abstract**:

        Automatic International Classification of Diseases (ICD) coding aims to assign multiple ICD codes to a medical note with an average of 3,000+ tokens. This task is challenging due to the high-dimensional space of multi-label assignment (155,000+ ICD code candidates) and the long-tail challenge - Many ICD codes are infrequently assigned yet infrequent ICD codes are important clinically. This study addresses the long-tail challenge by transforming this multi-label classification task into an autoregressive generation task. Specifically, we first introduce a novel pretraining objective to generate free text diagnosis and procedure descriptions using the SOAP structure, the medical logic physicians use for note documentation. Second, instead of directly predicting the high dimensional space of ICD codes, our model generates the lower dimension of text descriptions, which then infer ICD codes. Third, we designed a novel prompt template for multi-label classification. We evaluate our Generation with Prompt (GP) model with the benchmark of all code assignment (MIMIC-III-full) and few shot ICD code assignment evaluation benchmark (MIMIC-III-few). Experiments on MIMIC-III-few show that our model performs with a marco F1 30.2, which substantially outperforms the previous MIMIC-III-full SOTA model (marco F1 4.3) and the model specifically designed for few/zero shot setting (marco F1 18.7). Finally, we design a novel ensemble learner, a cross attention reranker with prompts, to integrate previous SOTA and our best few-shot coding predictions. Experiments on MIMIC-III-full show that our ensemble learner substantially improves both macro and micro F1, from 10.4 to 14.6 and from 58.2 to 59.1, respectively.

        ----

        ## [599] DMIS: Dynamic Mesh-Based Importance Sampling for Training Physics-Informed Neural Networks

        **Authors**: *Zijiang Yang, Zhongwei Qiu, Dongmei Fu*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i4.25669](https://doi.org/10.1609/aaai.v37i4.25669)

        **Abstract**:

        Modeling dynamics in the form of partial differential equations (PDEs) is an effectual way to understand real-world physics processes. For complex physics systems, analytical solutions are not available and numerical solutions are widely-used. However, traditional numerical algorithms are computationally expensive and challenging in handling multiphysics systems. Recently, using neural networks to solve PDEs has made significant progress, called physics-informed neural networks (PINNs). PINNs encode physical laws into neural networks and learn the continuous solutions of PDEs. For the training of PINNs, existing methods suffer from the problems of inefficiency and unstable convergence, since the PDE residuals require calculating automatic differentiation. In this paper, we propose Dynamic Mesh-based Importance Sampling (DMIS) to tackle these problems. DMIS is a novel sampling scheme based on importance sampling, which constructs a dynamic triangular mesh to estimate sample weights efficiently. DMIS has broad applicability and can be easily integrated into existing methods. The evaluation of DMIS on three widely-used benchmarks shows that DMIS improves the convergence speed and accuracy in the meantime. Especially in solving the highly nonlinear Schrödinger Equation, compared with state-of-the-art methods, DMIS shows up to 46% smaller root mean square error and five times faster convergence speed. Code is available at https://github.com/MatrixBrain/DMIS.

        ----

        

[Go to the previous page](AAAI-2023-list02.md)

[Go to the next page](AAAI-2023-list04.md)

[Go to the catalog section](README.md)