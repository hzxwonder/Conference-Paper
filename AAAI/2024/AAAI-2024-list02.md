## [200] Dual-Prior Augmented Decoding Network for Long Tail Distribution in HOI Detection

**Authors**: *Jiayi Gao, Kongming Liang, Tao Wei, Wei Chen, Zhanyu Ma, Jun Guo*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i3.27949](https://doi.org/10.1609/aaai.v38i3.27949)

**Abstract**:

Human object interaction detection aims at localizing human-object pairs and recognizing their interactions. Trapped by the long-tailed distribution of the data, existing HOI detection methods often have difficulty recognizing the tail categories. Many approaches try to improve the recognition of HOI tasks by utilizing external knowledge (e.g. pre-trained visual-language models). However, these approaches mainly utilize external knowledge at the HOI combination level and achieve limited improvement in the tail categories. In this paper, we propose a dual-prior augmented decoding network by decomposing the HOI task into two sub-tasks: human-object pair detection and interaction recognition. For each subtask, we leverage external knowledge to enhance the model's ability at a finer granularity. Specifically, we acquire the prior candidates from an external classifier and embed them to assist the subsequent decoding process. Thus, the long-tail problem is mitigated from a coarse-to-fine level with the corresponding external knowledge. Our approach outperforms existing state-of-the-art models in various settings and significantly boosts the performance on the tail HOI categories. The source code is available at https://github.com/PRIS-CV/DP-ADN.

----

## [201] LAMM: Label Alignment for Multi-Modal Prompt Learning

**Authors**: *Jingsheng Gao, Jiacheng Ruan, Suncheng Xiang, Zefang Yu, Ke Ji, Mingye Xie, Ting Liu, Yuzhuo Fu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i3.27950](https://doi.org/10.1609/aaai.v38i3.27950)

**Abstract**:

With the success of pre-trained visual-language (VL) models such as CLIP in visual representation tasks, transferring pre-trained models to downstream tasks has become a crucial paradigm. Recently, the prompt tuning paradigm, which draws inspiration from natural language processing (NLP), has made significant progress in VL field. However, preceding methods mainly focus on constructing prompt templates for text and visual inputs, neglecting the gap in class label representations between the VL models and downstream tasks. To address this challenge, we introduce an innovative label alignment method named \textbf{LAMM}, which can dynamically adjust the category embeddings of downstream datasets through end-to-end training. Moreover, to achieve a more appropriate label distribution, we propose a hierarchical loss, encompassing the alignment of the parameter space, feature space, and logits space. We conduct experiments on 11 downstream vision datasets and demonstrate that our method significantly improves the performance of existing multi-modal prompt learning models in few-shot scenarios, exhibiting an average accuracy improvement of 2.31(\%) compared to the state-of-the-art methods on 16 shots. Moreover, our methodology exhibits the preeminence in continual learning compared to other prompt tuning methods. Importantly, our method is synergistic with existing prompt tuning methods and can boost the performance on top of them. Our code and dataset will be publicly available at https://github.com/gaojingsheng/LAMM.

----

## [202] Frequency-Controlled Diffusion Model for Versatile Text-Guided Image-to-Image Translation

**Authors**: *Xiang Gao, Zhengbo Xu, Junhan Zhao, Jiaying Liu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i3.27951](https://doi.org/10.1609/aaai.v38i3.27951)

**Abstract**:

Recently, text-to-image diffusion models have emerged as a powerful tool for image-to-image translation (I2I), allowing flexible image translation via user-provided text prompts. This paper proposes frequency-controlled diffusion model (FCDiffusion), an end-to-end diffusion-based framework contributing a novel solution to text-guided I2I from a frequency-domain perspective. At the heart of our framework is a feature-space frequency-domain filtering module based on Discrete Cosine Transform, which extracts image features carrying different DCT spectral bands to control the text-to-image generation process of the Latent Diffusion Model, realizing versatile I2I applications including style-guided content creation, image semantic manipulation, image scene translation, and image style translation. Different from related methods, FCDiffusion establishes a unified text-driven I2I framework suiting diverse I2I application scenarios simply by switching among different frequency control branches. The effectiveness and superiority of our method for text-guided I2I are demonstrated with extensive experiments both qualitatively and quantitatively. Our project is publicly available at: https://xianggao1102.github.io/FCDiffusion/.

----

## [203] A General Implicit Framework for Fast NeRF Composition and Rendering

**Authors**: *Xinyu Gao, Ziyi Yang, Yunlu Zhao, Yuxiang Sun, Xiaogang Jin, Changqing Zou*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i3.27952](https://doi.org/10.1609/aaai.v38i3.27952)

**Abstract**:

A variety of Neural Radiance Fields (NeRF) methods have recently achieved remarkable success in high render speed. However, current accelerating methods are specialized and incompatible with various implicit methods, preventing real-time composition over various types of NeRF works. Because NeRF relies on sampling along rays, it is possible to provide general guidance for acceleration. To that end, we propose a general implicit pipeline for composing NeRF objects quickly. Our method enables the casting of dynamic shadows within or between objects using analytical light sources while allowing multiple NeRF objects to be seamlessly placed and rendered together with any arbitrary rigid transformations. Mainly, our work introduces a new surface representation known as Neural Depth Fields (NeDF) that quickly determines the spatial relationship between objects by allowing direct intersection computation between rays and implicit surfaces. It leverages an intersection neural network to query NeRF for acceleration instead of depending on an explicit spatial structure.Our proposed method is the first to enable both the progressive and interactive composition of NeRF objects. Additionally, it also serves as a previewing plugin for a range of existing NeRF works.

----

## [204] Multi-Scene Generalized Trajectory Global Graph Solver with Composite Nodes for Multiple Object Tracking

**Authors**: *Yan Gao, Haojun Xu, Jie Li, Nannan Wang, Xinbo Gao*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i3.27953](https://doi.org/10.1609/aaai.v38i3.27953)

**Abstract**:

The global multi-object tracking (MOT) system can consider interaction, occlusion, and other ``visual blur'' scenarios to ensure effective object tracking in long videos. Among them, graph-based tracking-by-detection paradigms achieve surprising performance. However, their fully-connected nature poses storage space requirements that challenge algorithm handling long videos. Currently, commonly used methods are still generated trajectories by building one-forward associations across frames. Such matches produced under the guidance of first-order similarity information may not be optimal from a longer-time perspective. Moreover, they often lack an end-to-end scheme for correcting mismatches. This paper proposes the Composite Node Message Passing Network (CoNo-Link), a multi-scene generalized framework for modeling ultra-long frames information for association. CoNo-Link's solution is a low-storage overhead method for building constrained connected graphs. In addition to the previous method of treating objects as nodes, the network innovatively treats object trajectories as nodes for information interaction, improving the graph neural network's feature representation capability. Specifically, we formulate the graph-building problem as a top-k selection task for some reliable objects or trajectories. Our model can learn better predictions on longer-time scales by adding composite nodes. As a result, our method outperforms the state-of-the-art in several commonly used datasets.

----

## [205] A Dual Stealthy Backdoor: From Both Spatial and Frequency Perspectives

**Authors**: *Yudong Gao, Honglong Chen, Peng Sun, Junjian Li, Anqing Zhang, Zhibo Wang, Weifeng Liu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i3.27954](https://doi.org/10.1609/aaai.v38i3.27954)

**Abstract**:

Backdoor attacks pose serious security threats to deep neural networks (DNNs). Backdoored models make arbitrarily (targeted) incorrect predictions on inputs containing well-designed triggers, while behaving normally on clean inputs. Prior researches have explored the invisibility of backdoor triggers to enhance attack stealthiness. However, most of them only focus on the invisibility in the spatial domain, neglecting the generation of invisible triggers in the frequency domain. This limitation renders the generated poisoned images easily detectable by recent defense methods. To address this issue, we propose a DUal stealthy BAckdoor attack method named DUBA, which simultaneously considers the invisibility of triggers in both the spatial and frequency domains, to achieve desirable attack performance, while ensuring strong stealthiness. Specifically, we first use Wavelet Transform to embed the high-frequency information of the trigger image into the clean image to ensure attack effectiveness. Then, to attain strong stealthiness, we incorporate Fourier Transform and Cosine Transform to mix the poisoned image and clean image in the frequency domain. Moreover, DUBA adopts a novel attack strategy, training the model with weak triggers and attacking with strong triggers to further enhance attack performance and stealthiness. DUBA is evaluated extensively on four datasets against popular image classifiers, showing significant superiority over state-of-the-art backdoor attacks in attack success rate and stealthiness.

----

## [206] SoftCLIP: Softer Cross-Modal Alignment Makes CLIP Stronger

**Authors**: *Yuting Gao, Jinfeng Liu, Zihan Xu, Tong Wu, Enwei Zhang, Ke Li, Jie Yang, Wei Liu, Xing Sun*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i3.27955](https://doi.org/10.1609/aaai.v38i3.27955)

**Abstract**:

During the preceding biennium, vision-language pre-training has achieved noteworthy success on several downstream tasks. Nevertheless, acquiring high-quality image-text pairs, where the pairs are entirely exclusive of each other, remains a challenging task, and noise exists in the commonly used datasets. To address this issue, we propose SoftCLIP, a novel approach that relaxes the strict one-to-one constraint and achieves a soft cross-modal alignment by introducing a softened target, which is generated from the fine-grained intra-modal self-similarity. The intra-modal guidance is indicative to enable two pairs have some local similarities and model many-to-many relationships between the two modalities. Besides, since the positive still dominates in the softened target distribution, we disentangle the negatives in the distribution to further boost the relation alignment with the negatives in the cross-modal learning. Extensive experiments demonstrate the effectiveness of SoftCLIP. In particular, on ImageNet zero-shot classification task, using CC3M/CC12M as pre-training dataset, SoftCLIP brings a top-1 accuracy improvement of 6.8%/7.2% over the CLIP baseline.

----

## [207] Composite Sketch+Text Queries for Retrieving Objects with Elusive Names and Complex Interactions

**Authors**: *Prajwal Gatti, Kshitij Parikh, Dhriti Prasanna Paul, Manish Gupta, Anand Mishra*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i3.27956](https://doi.org/10.1609/aaai.v38i3.27956)

**Abstract**:

Non-native speakers with limited vocabulary often struggle to name specific objects despite being able to visualize them, e.g., people outside Australia searching for ‘numbats.’ Further, users may want to search for such elusive objects with difficult-to-sketch interactions, e.g., “numbat digging in the ground.” In such common but complex situations, users desire a search interface that accepts composite multimodal queries comprising hand-drawn sketches of “difficult-to-name but easy-to-draw” objects and text describing “difficult-to-sketch but easy-to-verbalize” object's attributes or interaction with the scene. This novel problem statement distinctly differs from the previously well-researched TBIR (text-based image retrieval) and SBIR (sketch-based image retrieval) problems. To study this under-explored task, we curate a dataset, CSTBIR (Composite Sketch+Text Based Image Retrieval), consisting of ~2M queries and 108K natural scene images. Further, as a solution to this problem, we propose a pretrained multimodal transformer-based baseline, STNet (Sketch+Text Network), that uses a hand-drawn sketch to localize relevant objects in the natural scene image, and encodes the text and image to perform image retrieval. In addition to contrastive learning, we propose multiple training objectives that improve the performance of our model. Extensive experiments show that our proposed method outperforms several state-of-the-art retrieval methods for text-only, sketch-only, and composite query modalities. We make the dataset and code available at: https://vl2g.github.io/projects/cstbir.

----

## [208] Neuromorphic Event Signal-Driven Network for Video De-raining

**Authors**: *Chengjie Ge, Xueyang Fu, Peng He, Kunyu Wang, Chengzhi Cao, Zheng-Jun Zha*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i3.27957](https://doi.org/10.1609/aaai.v38i3.27957)

**Abstract**:

Convolutional neural networks-based video de-raining methods commonly rely on dense intensity frames captured by CMOS sensors. However, the limited temporal resolution of these sensors hinders the capture of dynamic rainfall information, limiting further improvement in de-raining performance. This study aims to overcome this issue by incorporating the neuromorphic event signal into the video de-raining to enhance the dynamic information perception. Specifically, we first utilize the dynamic information from the event signal as prior knowledge, and integrate it into existing de-raining objectives to better constrain the solution space. We then design an optimization algorithm to solve the objective, and construct a de-raining network with CNNs as the backbone architecture using a modular strategy to mimic the optimization process. To further explore the temporal correlation of the event signal, we incorporate a spiking self-attention module into our network. By leveraging the low latency and high temporal resolution of the event signal, along with the spatial and temporal representation capabilities of convolutional and spiking neural networks, our model captures more accurate dynamic information and significantly improves de-raining performance. For example, our network achieves a 1.24dB improvement on the SynHeavy25 dataset compared to the previous state-of-the-art method, while utilizing only 39% of the parameters.

----

## [209] Beyond Prototypes: Semantic Anchor Regularization for Better Representation Learning

**Authors**: *Yanqi Ge, Qiang Nie, Ye Huang, Yong Liu, Chengjie Wang, Feng Zheng, Wen Li, Lixin Duan*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i3.27958](https://doi.org/10.1609/aaai.v38i3.27958)

**Abstract**:

One of the ultimate goals of representation learning is to achieve compactness within a class and well-separability between classes. Many outstanding metric-based and prototype-based methods following the Expectation-Maximization paradigm, have been proposed for this objective. However, they inevitably introduce biases into the learning process, particularly with long-tail distributed training data. In this paper, we reveal that the class prototype is not necessarily to be derived from training features and propose a novel perspective to use pre-defined class anchors serving as feature centroid to unidirectionally guide feature learning. However, the pre-defined anchors may have a large semantic distance from the pixel features, which prevents them from being directly applied. To address this issue and generate feature centroid independent from feature learning, a simple yet effective Semantic Anchor Regularization (SAR) is proposed. SAR ensures the inter-class separability of semantic anchors in the semantic space by employing a classifier-aware auxiliary cross-entropy loss during training via disentanglement learning. By pulling the learned features to these semantic anchors, several advantages can be attained: 1) the intra-class compactness and naturally inter-class separability, 2) induced bias or errors from feature learning can be avoided, and 3) robustness to the long-tailed problem. The proposed SAR can be used in a plug-and-play manner in the existing models. Extensive experiments demonstrate that the SAR performs better than previous sophisticated prototype-based methods.  The implementation is available at https://github.com/geyanqi/SAR.

----

## [210] Learning Multi-Scale Video-Text Correspondence for Weakly Supervised Temporal Article Gronding

**Authors**: *Wenjia Geng, Yong Liu, Lei Chen, Sujia Wang, Jie Zhou, Yansong Tang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i3.27959](https://doi.org/10.1609/aaai.v38i3.27959)

**Abstract**:

Weakly Supervised temporal Article Grounding (WSAG) is a challenging and practical task in video understanding. Specifically, given a video and a relevant article, whose sentences are at different semantic scales, WSAG aims to localize corresponding video segments for all “groundable” sentences. Compared to other grounding tasks, e.g., localizing one target segment with respect to a given sentence query, WSAG confronts an essential obstacle rooted in the intricate multi-scale information inherent within both textual and visual modalities. Existing methods overlook the modeling and alignment of such structured information present in multi-scale video segments and hierarchical textual content. To this end, we propose a Multi-Scale Video-Text Correspondence Learning (MVTCL) framework, which enhances the grounding performance in complex scenes by modeling multi-scale semantic correspondence both within and between modalities. Specifically, MVTCL initially aggregates video content spanning distinct temporal scales and leverages hierarchical textual relationships in both temporal  and  semantic  dimensions via a semantic calibration module. Then multi-scale contrastive learning module is introduced to generate more discriminative representations by  selecting  typical  contexts  and performing inter-video contrastive learning. Through the multi-scale semantic calibration architecture and supervision design, our method achieves new state-of-the-art performance on existing WSAG benchmarks.

----

## [211] PoseGen: Learning to Generate 3D Human Pose Dataset with NeRF

**Authors**: *Mohsen Gholami, Rabab Ward, Z. Jane Wang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i3.27960](https://doi.org/10.1609/aaai.v38i3.27960)

**Abstract**:

This paper proposes an end-to-end framework for generating 3D human pose datasets using Neural Radiance Fields (NeRF).  Public datasets generally have limited diversity in terms of human poses and camera viewpoints, largely due to the resource-intensive nature of collecting 3D human pose data. As a result, pose estimators trained on public datasets significantly underperform when applied to unseen out-of-distribution samples. Previous works proposed augmenting public datasets by generating 2D-3D pose pairs or rendering a large amount of random data. Such approaches either overlook image rendering or result in suboptimal datasets for pre-trained models. Here we propose PoseGen, which learns to generate a dataset (human 3D poses and images) with a feedback loss from a given pre-trained pose estimator. In contrast to prior art, our generated data is optimized to improve the robustness of the pre-trained model. The objective of PoseGen is to learn a distribution of data that maximizes the prediction error of a given pre-trained model. As the learned data distribution contains OOD samples of the pre-trained model, sampling data from such a distribution for further fine-tuning a pre-trained model improves the generalizability of the model. This is the first work that proposes NeRFs for 3D human data generation. NeRFs are data-driven and do not require 3D scans of humans. Therefore, using NeRF for data generation is a new direction for convenient user-specific data generation. Our extensive experiments show that the proposed PoseGen improves two baseline models (SPIN and HybrIK) on four datasets with an average 6% relative improvement.

----

## [212] SDAC: A Multimodal Synthetic Dataset for Anomaly and Corner Case Detection in Autonomous Driving

**Authors**: *Lei Gong, Yu Zhang, Yingqing Xia, Yanyong Zhang, Jianmin Ji*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i3.27961](https://doi.org/10.1609/aaai.v38i3.27961)

**Abstract**:

Nowadays, closed-set perception methods for autonomous driving perform well on datasets containing normal scenes. However, they still struggle to handle anomalies in the real world, such as unknown objects that have never been seen while training. The lack of public datasets to evaluate the model performance on anomaly and corner cases has hindered the development of reliable autonomous driving systems. Therefore, we propose a multimodal Synthetic Dataset for Anomaly and Corner case detection, called SDAC, which encompasses anomalies captured from multi-view cameras and the LiDAR sensor, providing a rich set of annotations for multiple mainstream perception tasks. SDAC is the first public dataset for autonomous driving that categorizes anomalies into object, scene, and scenario levels, allowing the evaluation under different anomalous conditions. Experiments show that closed-set models suffer significant performance drops on anomaly subsets in SDAC. Existing anomaly detection methods fail to achieve satisfactory performance, suggesting that anomaly detection remains a challenging problem. We anticipate that our SDAC dataset could foster the development of safe and reliable systems for autonomous driving.

----

## [213] ContactGen: Contact-Guided Interactive 3D Human Generation for Partners

**Authors**: *Dongjun Gu, Jaehyeok Shim, Jaehoon Jang, Changwoo Kang, Kyungdon Joo*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i3.27962](https://doi.org/10.1609/aaai.v38i3.27962)

**Abstract**:

Among various interactions between humans, such as eye contact and gestures, physical interactions by contact can act as an essential moment in understanding human behaviors. Inspired by this fact, given a 3D partner human with the desired interaction label, we introduce a new task of 3D human generation in terms of physical contact. Unlike previous works of interacting with static objects or scenes, a given partner human can have diverse poses and different contact regions according to the type of interaction. To handle this challenge, we propose a novel method of generating interactive 3D humans for a given partner human based on a guided diffusion framework (ContactGen in short). Specifically, we newly present a contact prediction module that adaptively estimates potential contact regions between two input humans according to the interaction label. Using the estimated potential contact regions as complementary guidances, we dynamically enforce ContactGen to generate interactive 3D humans for a given partner human within a guided diffusion model. We demonstrate ContactGen on the CHI3D dataset, where our method generates physically plausible and diverse poses compared to comparison methods.

----

## [214] AnomalyGPT: Detecting Industrial Anomalies Using Large Vision-Language Models

**Authors**: *Zhaopeng Gu, Bingke Zhu, Guibo Zhu, Yingying Chen, Ming Tang, Jinqiao Wang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i3.27963](https://doi.org/10.1609/aaai.v38i3.27963)

**Abstract**:

Large Vision-Language Models (LVLMs) such as MiniGPT-4 and LLaVA have demonstrated the capability of understanding images and achieved remarkable performance in various visual tasks. Despite their strong abilities in recognizing common objects due to extensive training datasets, they lack specific domain knowledge and have a weaker understanding of localized details within objects, which hinders their effectiveness in the Industrial Anomaly Detection (IAD) task. On the other hand, most existing IAD methods only provide anomaly scores and necessitate the manual setting of thresholds to distinguish between normal and abnormal samples, which restricts their practical implementation. In this paper, we explore the utilization of LVLM to address the IAD problem and propose AnomalyGPT, a novel IAD approach based on LVLM. We generate training data by simulating anomalous images and producing corresponding textual descriptions for each image. We also employ an image decoder to provide fine-grained semantic and design a prompt learner to fine-tune the LVLM using prompt embeddings. Our AnomalyGPT eliminates the need for manual threshold adjustments, thus directly assesses the presence and locations of anomalies. Additionally, AnomalyGPT supports multi-turn dialogues and exhibits impressive few-shot in-context learning capabilities. With only one normal shot, AnomalyGPT achieves the state-of-the-art performance with an accuracy of 86.1%, an image-level AUC of 94.1%, and a pixel-level AUC of 95.3% on the MVTec-AD dataset.

----

## [215] SeqRank: Sequential Ranking of Salient Objects

**Authors**: *Huankang Guan, Rynson W. H. Lau*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i3.27964](https://doi.org/10.1609/aaai.v38i3.27964)

**Abstract**:

Salient Object Ranking (SOR) is the process of predicting the order of an observer's attention to objects when viewing a complex scene. Existing SOR methods primarily focus on ranking various scene objects simultaneously by exploring their spatial and semantic properties. However, their solutions of simultaneously ranking all salient objects do not align with human viewing behavior, and may result in incorrect attention shift predictions. We observe that humans view a scene through a sequential and continuous process involving a cycle of foveating to objects of interest with our foveal vision while using peripheral vision to prepare for the next fixation location. For instance, when we see a flying kite, our foveal vision captures the kite itself, while our peripheral vision can help us locate the person controlling it such that we can smoothly divert our attention to it next. By repeatedly carrying out this cycle, we can gain a thorough understanding of the entire scene. Based on this observation, we propose to model the dynamic interplay between foveal and peripheral vision to predict human attention shifts sequentially. To this end, we propose a novel SOR model, SeqRank, which reproduces foveal vision to extract high-acuity visual features for accurate salient instance segmentation while also modeling peripheral vision to select the object that is likely to grab the viewer’s attention next. By incorporating both types of vision, our model can mimic human viewing behavior better and provide a more faithful ranking among various scene objects. Most notably, our model improves the SA-SOR/MAE scores by +6.1%/-13.0% on IRSR, compared with the state-of-the-art. Extensive experiments show the superior performance of our model on the SOR benchmarks. Code is available at https://github.com/guanhuankang/SeqRank.

----

## [216] Knowledge-Aware Neuron Interpretation for Scene Classification

**Authors**: *Yong Guan, Freddy Lécué, Jiaoyan Chen, Ru Li, Jeff Z. Pan*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i3.27965](https://doi.org/10.1609/aaai.v38i3.27965)

**Abstract**:

Although neural models have achieved remarkable performance, they still encounter doubts due to the intransparency. To this end, model prediction explanation is attracting more and more attentions. However, current methods rarely incorporate external knowledge and still suffer from three limitations: (1) Neglecting concept completeness. Merely selecting concepts may not sufficient for prediction. (2) Lacking concept fusion. Failure to merge semantically-equivalent concepts. (3) Difficult in manipulating model behavior. Lack of verification for explanation on original model. To address these issues, we propose a novel knowledge-aware neuron interpretation framework to explain model predictions for image scene classification. Specifically, for concept completeness, we present core concepts of a scene based on knowledge graph, ConceptNet, to gauge the completeness of concepts. Our method, incorporating complete concepts, effectively provides better prediction explanations compared to baselines. Furthermore, for concept fusion, we introduce a knowledge graph-based method known as Concept Filtering, which produces over 23% point gain on neuron behaviors for neuron interpretation. At last, we propose Model Manipulation, which aims to study whether the core concepts based on ConceptNet could be employed to manipulate model behavior. The results show that core concepts can effectively improve the performance of original model by over 26%.

----

## [217] Self-Supervised Representation Learning with Meta Comprehensive Regularization

**Authors**: *Huijie Guo, Ying Ba, Jie Hu, Lingyu Si, Wenwen Qiang, Lei Shi*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i3.27966](https://doi.org/10.1609/aaai.v38i3.27966)

**Abstract**:

Self-Supervised Learning (SSL) methods harness the concept of semantic invariance by utilizing data augmentation strategies to produce similar representations for different deformations of the same input. Essentially, the model captures the shared information among multiple augmented views of samples, while disregarding the non-shared information that may be beneficial for downstream tasks. To address this issue, we introduce a module called CompMod with Meta Comprehensive Regularization (MCR), embedded into existing self-supervised frameworks, to make the learned representations more comprehensive. Specifically, we update our proposed model through a bi-level optimization mechanism, enabling it to capture comprehensive features. Additionally, guided by the constrained extraction of features using maximum entropy coding, the self-supervised learning model learns more comprehensive features on top of learning consistent features. In addition, we provide theoretical support for our proposed method from information theory and causal counterfactual perspective. Experimental results show that our method achieves significant improvement in classification, object detection and semantic segmentation tasks on multiple benchmark datasets.

----

## [218] Graph Context Transformation Learning for Progressive Correspondence Pruning

**Authors**: *Junwen Guo, Guobao Xiao, Shiping Wang, Jun Yu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i3.27967](https://doi.org/10.1609/aaai.v38i3.27967)

**Abstract**:

Most of existing correspondence pruning methods only concentrate on gathering the context information as much as possible while neglecting effective ways to utilize such information. In order to tackle this dilemma, in this paper we propose Graph Context Transformation Network (GCT-Net) enhancing context information to conduct consensus guidance for progressive correspondence pruning. Specifically, we design the Graph Context Enhance Transformer which first generates the graph network and then transforms it into multi-branch graph contexts. Moreover, it employs self-attention and cross-attention to magnify characteristics of each graph context for emphasizing the unique as well as shared essential information. To further apply the recalibrated graph contexts to the global domain, we propose the Graph Context Guidance Transformer. This module adopts a confident-based sampling strategy to temporarily screen high-confidence vertices for guiding accurate classification by searching global consensus between screened vertices and remaining ones. The extensive experimental results on outlier removal and relative pose estimation clearly demonstrate the superior performance of GCT-Net compared to state-of-the-art methods across outdoor and indoor datasets.

----

## [219] Depth-Guided Robust and Fast Point Cloud Fusion NeRF for Sparse Input Views

**Authors**: *Shuai Guo, Qiuwen Wang, Yijie Gao, Rong Xie, Li Song*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i3.27968](https://doi.org/10.1609/aaai.v38i3.27968)

**Abstract**:

Novel-view synthesis with sparse input views is important for real-world applications like AR/VR and autonomous driving. Recent methods have integrated depth information into NeRFs for sparse input synthesis, leveraging depth prior for geometric and spatial understanding. However, most existing works tend to overlook inaccuracies within depth maps and have low time efficiency. To address these issues, we propose a depth-guided robust and fast point cloud fusion NeRF for sparse inputs. We perceive radiance fields as an explicit voxel grid of features. A point cloud is constructed for each input view, characterized within the voxel grid using matrices and vectors. We accumulate the point cloud of each input view to construct the fused point cloud of the entire scene. Each voxel determines its density and appearance by referring to the point cloud of the entire scene. Through point cloud fusion and voxel grid fine-tuning, inaccuracies in depth values are refined or substituted by those from other views. Moreover, our method can achieve faster reconstruction and greater compactness through effective vector-matrix decomposition. Experimental results underline the superior performance and time efficiency of our approach compared to state-of-the-art baselines.

----

## [220] Improving Panoptic Narrative Grounding by Harnessing Semantic Relationships and Visual Confirmation

**Authors**: *Tianyu Guo, Haowei Wang, Yiwei Ma, Jiayi Ji, Xiaoshuai Sun*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i3.27969](https://doi.org/10.1609/aaai.v38i3.27969)

**Abstract**:

Recent advancements in single-stage Panoptic Narrative Grounding (PNG) have demonstrated significant potential. These methods predict pixel-level masks by directly matching pixels and phrases. However, they often neglect the modeling of semantic and visual relationships between phrase-level instances, limiting their ability for complex multi-modal reasoning in PNG. To tackle this issue, we propose XPNG, a “differentiation-refinement-localization” reasoning paradigm for accurately locating instances or regions. In XPNG, we introduce a Semantic Context Convolution (SCC) module to leverage semantic priors for generating distinctive features. This well-crafted module employs a combination of dynamic channel-wise convolution and pixel-wise convolution to embed semantic information and establish inter-object relationships guided by semantics. Subsequently, we propose a Visual Context Verification (VCV) module to provide visual cues, eliminating potential space biases introduced by semantics and further refining the visual features generated by the previous module. Extensive experiments on PNG benchmark datasets reveal that our approach achieves state-of-the-art performance, significantly outperforming existing methods by a considerable margin and yielding a 3.9-point improvement in overall metrics. Our codes and results are available at our project webpage: https://github.com/TianyuGoGO/XPNG.

----

## [221] Learning to Manipulate Artistic Images

**Authors**: *Wei Guo, Yuqi Zhang, De Ma, Qian Zheng*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i3.27970](https://doi.org/10.1609/aaai.v38i3.27970)

**Abstract**:

Recent advancement in computer vision has significantly lowered the barriers to artistic creation. Exemplar-based image translation methods have attracted much attention due to flexibility and controllability. However, these methods hold assumptions regarding semantics or require semantic information as the input, while accurate semantics is not easy to obtain in artistic images. Besides, these methods suffer from cross-domain artifacts due to training data prior and generate imprecise structure due to feature compression in the spatial domain. In this paper, we propose an arbitrary Style Image Manipulation Network (SIM-Net), which leverages semantic-free information as guidance and a region transportation strategy in a self-supervised manner for image generation. Our method balances computational efficiency and high resolution to a certain extent. Moreover, our method facilitates zero-shot style image manipulation. Both qualitative and quantitative experiments demonstrate the superiority of our method over state-of-the-art methods.Code is available at https://github.com/SnailForce/SIM-Net.

----

## [222] PICNN: A Pathway towards Interpretable Convolutional Neural Networks

**Authors**: *Wengang Guo, Jiayi Yang, Huilin Yin, Qijun Chen, Wei Ye*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i3.27971](https://doi.org/10.1609/aaai.v38i3.27971)

**Abstract**:

Convolutional Neural Networks (CNNs) have exhibited great performance in discriminative feature learning for complex visual tasks. Besides discrimination power, interpretability is another important yet under-explored property for CNNs. One difficulty in the CNN interpretability is that filters and image classes are entangled. In this paper, we introduce a novel pathway to alleviate the entanglement between filters and image classes. The proposed pathway groups the filters in a late conv-layer of CNN into class-specific clusters. Clusters and classes are in a one-to-one relationship. Specifically, we use the Bernoulli sampling to generate the filter-cluster assignment matrix from a learnable filter-class correspondence matrix. To enable end-to-end optimization, we develop a novel reparameterization trick for handling the non-differentiable Bernoulli sampling. We evaluate the effectiveness of our method on ten widely used network architectures (including nine CNNs and a ViT) and five benchmark datasets. Experimental results have demonstrated that our method PICNN (the combination of standard CNNs with our proposed pathway) exhibits greater interpretability than standard CNNs while achieving higher or comparable discrimination power.

----

## [223] GSN: Generalisable Segmentation in Neural Radiance Field

**Authors**: *Vinayak Gupta, Rahul Goel, Dhawal Sirikonda, P. J. Narayanan*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i3.27972](https://doi.org/10.1609/aaai.v38i3.27972)

**Abstract**:

Traditional Radiance Field (RF) representations capture details of a specific scene and must be trained afresh on each scene. Semantic feature fields have been added to RFs to facilitate several segmentation tasks. Generalised RF representations learn the principles of view interpolation. A generalised RF can render new views of an unknown and untrained scene, given a few views. We present a way to distil feature fields into the generalised GNT representation. Our GSN representation generates new views of unseen scenes on the fly along with consistent, per-pixel semantic features. This enables multi-view segmentation of arbitrary new scenes. We show different semantic features being distilled into generalised RFs. Our multi-view segmentation results are on par with methods that use traditional RFs. GSN closes the gap between standard and generalisable RF methods significantly. Project Page: https://vinayak-vg.github.io/GSN/

----

## [224] AMD: Autoregressive Motion Diffusion

**Authors**: *Bo Han, Hao Peng, Minjing Dong, Yi Ren, Yixuan Shen, Chang Xu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i3.27973](https://doi.org/10.1609/aaai.v38i3.27973)

**Abstract**:

Human motion generation aims to produce plausible human motion sequences according to various conditional inputs, such as text or audio. Despite the feasibility of existing methods in generating motion based on short prompts and simple motion patterns, they encounter difficulties when dealing with long prompts or complex motions.
The challenges are two-fold: 1) the scarcity of human motion-captured data for long prompts and complex motions. 2) the high diversity of human motions in the temporal domain and the substantial divergence of distributions from conditional modalities, leading to a many-to-many mapping problem when generating motion with complex and long texts. 
In this work, we address these gaps by 1) elaborating the first dataset pairing long textual descriptions and 3D complex motions (HumanLong3D), and 2) proposing an autoregressive motion diffusion model (AMD). Specifically, AMD integrates the text prompt at the current timestep with the text prompt and action sequences at the previous timestep as conditional information to predict the current action sequences in an iterative manner.
Furthermore, we present its generalization for X-to-Motion with “No Modality Left Behind”, enabling for the first time the generation of high-definition and high-fidelity human motions based on user-defined modality input.

----

## [225] HuTuMotion: Human-Tuned Navigation of Latent Motion Diffusion Models with Minimal Feedback

**Authors**: *Gaoge Han, Shaoli Huang, Mingming Gong, Jinglei Tang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i3.27974](https://doi.org/10.1609/aaai.v38i3.27974)

**Abstract**:

We introduce HuTuMotion, an innovative approach for generating natural human motions that navigates latent motion diffusion models by leveraging few-shot human feedback. Unlike existing approaches that sample latent variables from a standard normal prior distribution, our method adapts the prior distribution to better suit the characteristics of the data, as indicated by human feedback, thus enhancing the quality of motion generation. Furthermore, our findings reveal that utilizing few-shot feedback can yield performance levels on par with those attained through extensive human feedback. This discovery emphasizes the potential and efficiency of incorporating few-shot human-guided optimization within latent diffusion models for personalized and style-aware human motion generation applications. The experimental results show the significantly superior performance of our method over existing state-of-the-art approaches.

----

## [226] MA-Net: Rethinking Neural Unit in the Light of Astrocytes

**Authors**: *Mengqiao Han, Liyuan Pan, Xiabi Liu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i3.27975](https://doi.org/10.1609/aaai.v38i3.27975)

**Abstract**:

The artificial neuron (N-N) model-based networks have accomplished extraordinary success for various vision tasks. However, as a simplification of the mammal neuron model, their structure is locked during training, resulting in overfitting and over-parameters. The astrocyte, newly explored by biologists, can adaptively modulate neuronal communication by inserting itself between neurons. The communication, between the astrocyte and neuron, is bidirectionally and shows the potential to alleviate issues raised by unidirectional communication in the N-N model. In this paper, we first elaborate on the artificial Multi-Astrocyte-Neuron (MA-N) model, which enriches the functionality of the artificial neuron model. Our MA-N model is formulated at both astrocyte- and neuron-level that mimics the bidirectional communication with temporal and joint mechanisms. Then, we construct the MA-Net network with the MA-N model, whose neural connections can be continuously and adaptively modulated during training. Experiments show that our MA-Net advances new state-of-the-art on multiple tasks while significantly reducing its parameters by connection optimization.

----

## [227] Dual-Perspective Knowledge Enrichment for Semi-supervised 3D Object Detection

**Authors**: *Yucheng Han, Na Zhao, Weiling Chen, Keng Teck Ma, Hanwang Zhang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i3.27976](https://doi.org/10.1609/aaai.v38i3.27976)

**Abstract**:

Semi-supervised 3D object detection is a promising yet under-explored direction to reduce data annotation costs, especially for cluttered indoor scenes. A few prior works, such as SESS and 3DIoUMatch, attempt to solve this task by utilizing a teacher model to generate pseudo-labels for unlabeled samples. However, the availability of unlabeled samples in the 3D domain is relatively limited compared to its 2D counterpart due to the greater effort required to collect 3D data. Moreover, the loose consistency regularization in SESS and restricted pseudo-label selection strategy in 3DIoUMatch lead to either low-quality supervision or a limited amount of pseudo labels. To address these issues, we present a novel Dual-Perspective Knowledge Enrichment approach named DPKE for semi-supervised 3D object detection. Our DPKE enriches the knowledge of limited training data, particularly unlabeled data, from two perspectives: data-perspective and feature-perspective. Specifically, from the data-perspective, we propose a class-probabilistic data augmentation method that augments the input data with additional instances based on the varying distribution of class probabilities. Our DPKE achieves feature-perspective knowledge enrichment by designing a geometry-aware feature matching method that regularizes feature-level similarity between object proposals from the student and teacher models. Extensive experiments on the two benchmark datasets demonstrate that our DPKE achieves superior performance over existing state-of-the-art approaches under various label ratio conditions. The source code and models will be made available to the public.

----

## [228] Exploiting the Social-Like Prior in Transformer for Visual Reasoning

**Authors**: *Yudong Han, Yupeng Hu, Xuemeng Song, Haoyu Tang, Mingzhu Xu, Liqiang Nie*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i3.27977](https://doi.org/10.1609/aaai.v38i3.27977)

**Abstract**:

Benefiting from instrumental global dependency modeling of self-attention (SA), transformer-based approaches have become the pivotal choices for numerous downstream visual reasoning tasks, such as visual question answering (VQA) and referring expression comprehension (REC). However, some studies have recently suggested that SA tends to suffer from rank collapse thereby inevitably leads to representation degradation as the transformer layer goes deeper. Inspired by social network theory, we attempt to make an analogy between social behavior and regional information interaction in SA, and harness two crucial notions of structural hole and degree centrality in social network to explore the possible optimization towards SA learning, which naturally deduces two plug-and-play social-like modules. Based on structural hole, the former module allows to make information interaction in SA more structured, which effectively avoids redundant information aggregation and global feature homogenization for better rank remedy, followed by latter module to comprehensively characterize and refine the representation discrimination via considering degree centrality of regions and transitivity of relations. Without bells and whistles, our model outperforms a bunch of baselines by a noticeable margin when considering our social-like prior on five benchmarks in VQA and REC tasks, and a series of explanatory results are showcased to sufficiently reveal the social-like behaviors in SA.

----

## [229] Improving Audio-Visual Segmentation with Bidirectional Generation

**Authors**: *Dawei Hao, Yuxin Mao, Bowen He, Xiaodong Han, Yuchao Dai, Yiran Zhong*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i3.27978](https://doi.org/10.1609/aaai.v38i3.27978)

**Abstract**:

The aim of audio-visual segmentation (AVS) is to precisely differentiate audible objects within videos down to the pixel level. Traditional approaches often tackle this challenge by combining information from various modalities, where the contribution of each modality is implicitly or explicitly modeled. Nevertheless, the interconnections between different modalities tend to be overlooked in audio-visual modeling. In this paper, inspired by the human ability to mentally simulate the sound of an object and its visual appearance, we introduce a bidirectional generation framework. This framework establishes robust correlations between an object's visual characteristics and its associated sound, thereby enhancing the performance of AVS. To achieve this, we employ a visual-to-audio projection component that reconstructs audio features from object segmentation masks and minimizes reconstruction errors. Moreover, recognizing that many sounds are linked to object movements, we introduce an implicit volumetric motion estimation module to handle temporal dynamics that may be challenging to capture using conventional optical flow methods. To showcase the effectiveness of our approach, we conduct comprehensive experiments and analyses on the widely recognized AVSBench benchmark. As a result, we establish a new state-of-the-art performance level in the AVS benchmark, particularly excelling in the challenging MS3 subset which involves segmenting multiple sound sources. Code is released in: https://github.com/OpenNLPLab/AVS-bidirectional.

----

## [230] Hand-Centric Motion Refinement for 3D Hand-Object Interaction via Hierarchical Spatial-Temporal Modeling

**Authors**: *Yuze Hao, Jianrong Zhang, Tao Zhuo, Fuan Wen, Hehe Fan*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i3.27979](https://doi.org/10.1609/aaai.v38i3.27979)

**Abstract**:

Hands are the main medium when people interact with the world. Generating proper 3D motion for hand-object interaction is vital for applications such as virtual reality and robotics. Although grasp tracking or object manipulation synthesis can produce coarse hand motion, this kind of motion is inevitably noisy and full of jitter. To address this problem, we propose a data-driven method for coarse motion refinement. First, we design a hand-centric representation to describe the dynamic spatial-temporal relation between hands and objects. Compared to the object-centric representation, our hand-centric representation is straightforward and does not require an ambiguous projection process that converts object-based prediction into hand motion. Second, to capture the dynamic clues of hand-object interaction, we propose a new architecture that models the spatial and temporal structure in a hierarchical manner. Extensive experiments demonstrate that our method outperforms previous methods by a noticeable margin.

----

## [231] Progressive Feature Self-Reinforcement for Weakly Supervised Semantic Segmentation

**Authors**: *Jingxuan He, Lechao Cheng, Chaowei Fang, Zunlei Feng, Tingting Mu, Mingli Song*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i3.27980](https://doi.org/10.1609/aaai.v38i3.27980)

**Abstract**:

Compared to conventional semantic segmentation with pixel-level supervision, weakly supervised semantic segmentation (WSSS) with image-level labels poses the challenge that it commonly focuses on the most discriminative regions, resulting in a disparity between weakly and fully supervision scenarios. A typical manifestation is the diminished precision on object boundaries, leading to deteriorated accuracy of WSSS. To alleviate this issue, we propose to adaptively partition the image content into certain regions (e.g., confident foreground and background) and uncertain regions (e.g., object boundaries and misclassified categories) for separate processing. For uncertain cues, we propose an adaptive masking strategy and seek to recover the local information with self-distilled knowledge. We further assume that confident regions should be robust enough to preserve the global semantics, and introduce a complementary self-distillation method that constrains semantic consistency between confident regions and an augmented view with the same class labels. Extensive experiments conducted on PASCAL VOC 2012 and MS COCO 2014 demonstrate that our proposed single-stage approach for WSSS not only outperforms state-of-the-art counterparts but also surpasses multi-stage methods that trade complexity for accuracy.

----

## [232] Prompting Multi-Modal Image Segmentation with Semantic Grouping

**Authors**: *Qibin He*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i3.27981](https://doi.org/10.1609/aaai.v38i3.27981)

**Abstract**:

Multi-modal image segmentation is one of the core issues in computer vision. The main challenge lies in integrating common information between modalities while retaining specific patterns for each modality. Existing methods typically perform full fine-tuning on RGB-based pre-trained parameters to inherit the powerful representation of the foundation model. Although effective, such paradigm is not optimal due to weak transferability and scarce downstream data. Inspired by the recent success of prompt learning in language models, we propose the Grouping Prompt Tuning Framework (GoPT), which introduces explicit semantic grouping to learn modal-related prompts, adapting the frozen pre-trained foundation model to various downstream multi-modal segmentation tasks. Specifically, a class-aware uni-modal prompter is designed to balance intra- and inter-modal semantic propagation by grouping modality-specific class tokens, thereby improving the adaptability of spatial information. Furthermore, an alignment-induced cross-modal prompter is introduced to aggregate class-aware representations and share prompt parameters among different modalities to assist in modeling common statistics. Extensive experiments show the superiority of our GoPT, which achieves SOTA performance on various downstream multi-modal image segmentation tasks by training only < 1% model parameters.

----

## [233] Low-Latency Space-Time Supersampling for Real-Time Rendering

**Authors**: *Ruian He, Shili Zhou, Yuqi Sun, Ri Cheng, Weimin Tan, Bo Yan*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i3.27982](https://doi.org/10.1609/aaai.v38i3.27982)

**Abstract**:

With the rise of real-time rendering and the evolution of display devices, there is a growing demand for post-processing methods that offer high-resolution content in a high frame rate. Existing techniques often suffer from quality and latency issues due to the disjointed treatment of frame supersampling and extrapolation. In this paper, we  recognize the shared context and mechanisms between frame supersampling and extrapolation, and present a novel framework, Space-time Supersampling (STSS). By integrating them into a unified framework, STSS can improve the overall quality with lower latency. To implement an efficient architecture, we treat the aliasing and warping holes unified as reshading regions and put forth two key components to compensate the regions, namely Random Reshading Masking (RRM) and Efficient Reshading Module (ERM). Extensive experiments demonstrate that our approach achieves superior visual fidelity compared to state-of-the-art (SOTA) methods. Notably, the performance is achieved within only 4ms, saving up to 75\% of time against the conventional two-stage pipeline that necessitates 17ms.

----

## [234] Collaborative Weakly Supervised Video Correlation Learning for Procedure-Aware Instructional Video Analysis

**Authors**: *Tianyao He, Huabin Liu, Yuxi Li, Xiao Ma, Cheng Zhong, Yang Zhang, Weiyao Lin*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i3.27983](https://doi.org/10.1609/aaai.v38i3.27983)

**Abstract**:

Video Correlation Learning (VCL), which aims to analyze the relationships between videos, has been widely studied and applied in various general video tasks. However, applying VCL to instructional videos is still quite challenging due to their intrinsic procedural temporal structure. Specifically, procedural knowledge is critical for accurate correlation analyses on instructional videos. Nevertheless, current procedure-learning methods heavily rely on step-level annotations, which are costly and not scalable. To address this problem, we introduce a weakly supervised framework called Collaborative Procedure Alignment (CPA) for procedure-aware correlation learning on instructional videos. Our framework comprises two core modules: collaborative step mining and frame-to-step alignment. The collaborative step mining module enables simultaneous and consistent step segmentation for paired videos, leveraging the semantic and temporal similarity between frames. Based on the identified steps, the frame-to-step alignment module performs alignment between the frames and steps across videos. The alignment result serves as a measurement of the correlation distance between two videos. We instantiate our framework in two distinct instructional video tasks: sequence verification and action quality assessment. Extensive experiments validate the effectiveness of our approach in providing accurate and interpretable correlation analyses for instructional videos.

----

## [235] Frequency-Adaptive Pan-Sharpening with Mixture of Experts

**Authors**: *Xuanhua He, Keyu Yan, Rui Li, Chengjun Xie, Jie Zhang, Man Zhou*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i3.27984](https://doi.org/10.1609/aaai.v38i3.27984)

**Abstract**:

Pan-sharpening involves reconstructing missing high-frequency information in multi-spectral images with low spatial resolution, using a higher-resolution panchromatic image as guidance. Although the inborn connection with frequency domain, existing pan-sharpening research has not almost investigated the potential solution upon frequency domain. To this end, we propose a novel Frequency Adaptive Mixture of Experts (FAME) learning framework for pan-sharpening, which consists of three key components: the Adaptive Frequency Separation Prediction Module, the Sub-Frequency Learning Expert Module, and the Expert Mixture Module. In detail, the first leverages the discrete cosine transform to perform frequency separation by predicting the frequency mask. On the basis of generated mask, the second with low-frequency MOE and high-frequency MOE takes account for enabling the effective low-frequency and high-frequency information reconstruction. Followed by, the final fusion module dynamically weights high frequency and low-frequency MOE knowledge to adapt to remote sensing images with significant content variations. Quantitative and qualitative experiments over multiple datasets demonstrate that our method performs the best against other state-of-the-art ones and comprises a strong generalization ability for real-world scenes. Code will be made publicly at https://github.com/alexhe101/FAME-Net.

----

## [236] Enhancing RAW-to-sRGB with Decoupled Style Structure in Fourier Domain

**Authors**: *Xuanhua He, Tao Hu, Guoli Wang, Zejin Wang, Run Wang, Qian Zhang, Keyu Yan, Ziyi Chen, Rui Li, Chengjun Xie, Jie Zhang, Man Zhou*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i3.27985](https://doi.org/10.1609/aaai.v38i3.27985)

**Abstract**:

RAW to sRGB mapping, which aims to convert RAW images from smartphones into RGB form equivalent to that of Digital Single-Lens Reflex (DSLR) cameras, has become an important area of research. However, current methods often ignore the difference between cell phone RAW images and DSLR camera RGB images, a difference that goes beyond the color matrix and extends to spatial structure due to resolution variations. Recent methods directly rebuild color mapping and spatial structure via shared deep representation, limiting optimal performance. Inspired by Image Signal Processing (ISP) pipeline, which distinguishes image restoration and enhancement, we present a novel Neural ISP framework, named FourierISP. This approach breaks the image down into style and structure within the frequency domain, allowing for independent optimization. FourierISP is comprised of three subnetworks: Phase Enhance Subnet for structural refinement, Amplitude Refine Subnet for color learning, and Color Adaptation Subnet for blending them in a smooth manner. This approach sharpens both color and structure, and extensive evaluations across varied datasets confirm that our approach realizes state-of-the-art results. Code will be available at https://github.com/alexhe101/FourierISP.

----

## [237] A User-Friendly Framework for Generating Model-Preferred Prompts in Text-to-Image Synthesis

**Authors**: *Nailei Hei, Qianyu Guo, Zihao Wang, Yan Wang, Haofen Wang, Wenqiang Zhang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i3.27986](https://doi.org/10.1609/aaai.v38i3.27986)

**Abstract**:

Well-designed prompts have demonstrated the potential to guide text-to-image models in generating amazing images. Although existing prompt engineering methods can provide high-level guidance, it is challenging for novice users to achieve the desired results by manually entering prompts due to a discrepancy between novice-user-input prompts and the model-preferred prompts. To bridge the distribution gap between user input behavior and model training datasets, we first construct a novel Coarse-Fine Granularity Prompts dataset (CFP) and propose a novel User-Friendly Fine-Grained Text Generation framework (UF-FGTG) for automated prompt optimization. For CFP, we construct a novel dataset for text-to-image tasks that combines coarse and fine-grained prompts to facilitate the development of automated prompt generation methods. For UF-FGTG, we propose a novel framework that automatically translates user-input prompts into model-preferred prompts. Specifically, we propose a prompt refiner that continually rewrites prompts to empower users to select results that align with their unique needs. Meanwhile, we integrate image-related loss functions from the text-to-image model into the training process of text generation to generate model-preferred prompts. Additionally, we propose an adaptive feature extraction module to ensure diversity in the generated results. Experiments demonstrate that our approach is capable of generating more visually appealing and diverse images than previous state-of-the-art methods, achieving an average improvement of 5% across six quality and aesthetic metrics. Data and code are available at https://github.com/Naylenv/UF-FGTG.

----

## [238] Optimize & Reduce: A Top-Down Approach for Image Vectorization

**Authors**: *Or Hirschorn, Amir Jevnisek, Shai Avidan*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i3.27987](https://doi.org/10.1609/aaai.v38i3.27987)

**Abstract**:

Vector image representation is a popular choice when editability and flexibility in resolution are desired. However, most images are only available in raster form, making raster-to-vector image conversion (vectorization) an important task. Classical methods for vectorization are either domain-specific or yield an abundance of shapes which limits editability and interpretability. Learning-based methods, that use differentiable rendering, have revolutionized vectorization, at the cost of poor generalization to out-of-training distribution domains, and optimization-based counterparts are either slow or produce non-editable and redundant shapes. In this work, we propose Optimize & Reduce (O&R), a top-down approach to vectorization that is both fast and domain-agnostic. O&R aims to attain a compact representation of input images by iteratively optimizing Bezier curve parameters and significantly reducing the number of shapes, using a devised importance measure. We contribute a benchmark of five datasets comprising images from a broad spectrum of image complexities - from emojis to natural-like images. Through extensive experiments on hundreds of images, we demonstrate that our method is domain agnostic and outperforms existing works in both reconstruction and perceptual quality for a fixed number of shapes. Moreover, we show that our algorithm is x10 faster than the state-of-the-art optimization-based method. Our code is publicly available: https://github.com/ajevnisek/optimize-and-reduce

----

## [239] MotionMix: Weakly-Supervised Diffusion for Controllable Motion Generation

**Authors**: *Nhat M. Hoang, Kehong Gong, Chuan Guo, Michael Bi Mi*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i3.27988](https://doi.org/10.1609/aaai.v38i3.27988)

**Abstract**:

Controllable generation of 3D human motions becomes an important topic as the world embraces digital transformation. Existing works, though making promising progress with the advent of diffusion models, heavily rely on meticulously captured and annotated (e.g., text) high-quality motion corpus, a resource-intensive endeavor in the real world. This motivates our proposed MotionMix, a simple yet effective weakly-supervised diffusion model that leverages both noisy and unannotated motion sequences. Specifically, we separate the denoising objectives of a diffusion model into two stages: obtaining conditional rough motion approximations in the initial T-T* steps by learning the noisy annotated motions, followed by the unconditional refinement of these preliminary motions during the last T* steps using unannotated motions. Notably, though learning from two sources of imperfect data, our model does not compromise motion generation quality compared to fully supervised approaches that access gold data. Extensive experiments on several benchmarks demonstrate that our MotionMix, as a versatile framework, consistently achieves state-of-the-art performances on text-to-motion, action-to-motion, and music-to-dance tasks.

----

## [240] Commonsense for Zero-Shot Natural Language Video Localization

**Authors**: *Meghana Holla, Ismini Lourentzou*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i3.27989](https://doi.org/10.1609/aaai.v38i3.27989)

**Abstract**:

Zero-shot Natural Language-Video Localization (NLVL) methods have exhibited promising results in training NLVL models exclusively with raw video data by dynamically generating video segments and pseudo-query annotations. However, existing pseudo-queries often lack grounding in the source video, resulting in unstructured and disjointed content. In this paper, we investigate the effectiveness of commonsense reasoning in zero-shot NLVL. Specifically, we present CORONET, a zero-shot NLVL framework that leverages commonsense to bridge the gap between videos and generated pseudo-queries via a commonsense enhancement module. CORONET employs Graph Convolution Networks (GCN) to encode commonsense information extracted from a knowledge graph, conditioned on the video, and cross-attention mechanisms to enhance the encoded video and pseudo-query representations prior to localization. Through empirical evaluations on two benchmark datasets, we demonstrate that CORONET surpasses both zero-shot and weakly supervised baselines, achieving improvements up to 32.13% across various recall thresholds and up to 6.33% in mIoU. These results underscore the significance of leveraging commonsense reasoning for zero-shot NLVL.

----

## [241] Learning Subject-Aware Cropping by Outpainting Professional Photos

**Authors**: *James Hong, Lu Yuan, Michaël Gharbi, Matthew Fisher, Kayvon Fatahalian*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i3.27990](https://doi.org/10.1609/aaai.v38i3.27990)

**Abstract**:

How to frame (or crop) a photo often depends on the image subject and its context; e.g., a human portrait. Recent works have defined the subject-aware image cropping task as a nuanced and practical version of image cropping. We propose a weakly-supervised approach (GenCrop) to learn what makes a high-quality, subject-aware crop from professional stock images. Unlike supervised prior work, GenCrop requires no new manual annotations beyond the existing stock image collection. The key challenge in learning from this data, however, is that the images are already cropped and we do not know what regions were removed. Our insight is to combine a library of stock images with a modern, pre-trained text-to-image diffusion model. The stock image collection provides diversity, and its images serve as pseudo-labels for a good crop. The text-image diffusion model is used to out-paint (i.e., outward inpainting) realistic uncropped images. Using this procedure, we are able to automatically generate a large dataset of cropped-uncropped training pairs to train a cropping model. Despite being weakly-supervised, GenCrop is competitive with state-of-the-art supervised methods and significantly better than comparable weakly-supervised baselines on quantitative and qualitative evaluation metrics.

----

## [242] High-Fidelity Diffusion-Based Image Editing

**Authors**: *Chen Hou, Guoqiang Wei, Zhibo Chen*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i3.27991](https://doi.org/10.1609/aaai.v38i3.27991)

**Abstract**:

Diffusion models have attained remarkable success in the domains of image generation and editing. It is widely recognized that employing larger inversion and denoising steps in diffusion model leads to improved image reconstruction quality. However, the editing performance of diffusion models tends to be no more satisfactory even with increasing denoising steps. The deficiency in editing could be attributed to the conditional Markovian property of the editing process, where errors accumulate throughout denoising steps.  To tackle this challenge, we first propose an innovative framework where a rectifier module is incorporated to modulate diffusion model weights with  residual features from the original images, thereby providing compensatory information to bridge the fidelity gap.  Furthermore, we introduce a novel learning paradigm aimed at minimizing error propagation during the editing process, which trains the editing procedure in a manner similar to denoising score-matching.  Extensive experiments demonstrate that our proposed framework and training strategy achieve high-fidelity reconstruction and editing results across various levels of denoising steps, meanwhile exhibits exceptional performance in terms of both quantitative metric and qualitative assessments. Lastly, we explore our model's generalization though several applications like image-to-image translation and out-of-domain image editing.

----

## [243] Domain-Hallucinated Updating for Multi-Domain Face Anti-spoofing

**Authors**: *Chengyang Hu, Ke-Yue Zhang, Taiping Yao, Shice Liu, Shouhong Ding, Xin Tan, Lizhuang Ma*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i3.27992](https://doi.org/10.1609/aaai.v38i3.27992)

**Abstract**:

Multi-Domain Face Anti-Spoofing (MD-FAS) is a practical setting that aims to update models on new domains using only novel data while ensuring that the knowledge acquired from previous domains is not forgotten.
Prior methods utilize the responses from models to represent the previous domain knowledge or map the different domains into separated feature spaces to prevent forgetting.
However, due to domain gaps, the responses of new data are not as accurate as those of previous data. 
Also, without the supervision of previous data, separated feature spaces might be destroyed by new domains while updating, leading to catastrophic forgetting.
Inspired by the challenges posed by the lack of previous data, we solve this issue from a new standpoint that generates hallucinated previous data for updating FAS model.
To this end, we propose a novel Domain-Hallucinated Updating (DHU) framework to facilitate the hallucination of data.
Specifically, Domain Information Explorer learns representative domain information of the previous domains. 
Then, Domain Information Hallucination module transfers the new domain data to pseudo-previous domain ones.
Moreover, Hallucinated Features Joint Learning module is proposed to asymmetrically align the new and pseudo-previous data for real samples via dual levels to learn more generalized features, promoting the results on all domains.
Our experimental results and visualizations demonstrate that the proposed method outperforms state-of-the-art competitors in terms of effectiveness.

----

## [244] QI-IRA: Quantum-Inspired Interactive Ranking Aggregation for Person Re-identification

**Authors**: *Chunyu Hu, Hong Zhang, Chao Liang, Hao Huang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i3.27993](https://doi.org/10.1609/aaai.v38i3.27993)

**Abstract**:

Ranking aggregation (RA), the process of aggregating multiple rankings derived from multiple search strategies, has been proved effective in person re-identification (re-ID) because of a single re-ID method can not always achieve consistent superiority for different scenarios. Existing RA research mainly focus on unsupervised and fully-supervised methods. The former lack external supervision to optimize performance, while the latter are costly because of expensive labeling effort required for training. To address the above challenges, this paper proposes a quantum-inspired interactive ranking aggregation (QI-IRA) method, which (1) utilizes quantum theory to interpret and model the generation and aggregation of multiple basic rankings, (2) approximates or even exceeds the performance of fully-supervised RA methods with much less labeling cost, even as low as only two feedbacks per query on Market1501, MARS and DukeMTMC-VideoReID datasets. Comparative experiments conducted on six public re-ID datasets validate the superiority of the proposed QI-IRA method over existing unsupervised, interactive, and fully-supervised RA approaches.

----

## [245] SpaceGTN: A Time-Agnostic Graph Transformer Network for Handwritten Diagram Recognition and Segmentation

**Authors**: *Haoxiang Hu, Cangjun Gao, Yaokun Li, Xiaoming Deng, Yu-Kun Lai, Cuixia Ma, Yong-Jin Liu, Hongan Wang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i3.27994](https://doi.org/10.1609/aaai.v38i3.27994)

**Abstract**:

Online handwriting recognition is pivotal in domains like note-taking, education, healthcare, and office tasks. Existing diagram recognition algorithms mainly rely on the temporal information of strokes, resulting in a decline in recognition performance when dealing with notes that have been modified or have no temporal information. The current datasets are drawn based on templates and cannot reflect the real free-drawing situation. To address these challenges, we present SpaceGTN, a time-agnostic Graph Transformer Network, leveraging spatial integration and removing the need for temporal data. Extensive experiments on multiple datasets have demonstrated that our method consistently outperforms existing methods and achieves state-of-the-art performance. We also propose a pipeline that seamlessly connects offline and online handwritten diagrams. By integrating a stroke restoration technique with SpaceGTN, it enables intelligent editing of previously uneditable offline diagrams at the stroke level. In addition, we have also launched the first online handwritten diagram dataset, OHSD, which is collected using a free-drawing method and comes with modification annotations.

----

## [246] Learning Explicit Contact for Implicit Reconstruction of Hand-Held Objects from Monocular Images

**Authors**: *Junxing Hu, Hongwen Zhang, Zerui Chen, Mengcheng Li, Yunlong Wang, Yebin Liu, Zhenan Sun*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i3.27995](https://doi.org/10.1609/aaai.v38i3.27995)

**Abstract**:

Reconstructing hand-held objects from monocular RGB images is an appealing yet challenging task. In this task, contacts between hands and objects provide important cues for recovering the 3D geometry of the hand-held objects. Though recent works have employed implicit functions to achieve impressive progress, they ignore formulating contacts in their frameworks, which results in producing less realistic object meshes. In this work, we explore how to model contacts in an explicit way to benefit the implicit reconstruction of hand-held objects. Our method consists of two components: explicit contact prediction and implicit shape reconstruction. In the first part, we propose a new subtask of directly estimating 3D hand-object contacts from a single image. The part-level and vertex-level graph-based transformers are cascaded and jointly learned in a coarse-to-fine manner for more accurate contact probabilities. In the second part, we introduce a novel method to diffuse estimated contact states from the hand mesh surface to nearby 3D space and leverage diffused contact probabilities to construct the implicit neural representation for the manipulated object. Benefiting from estimating the interaction patterns between the hand and the object, our method can reconstruct more realistic object meshes, especially for object parts that are in contact with hands. Extensive experiments on challenging benchmarks show that the proposed method outperforms the current state of the arts by a great margin. Our code is publicly available at https://junxinghu.github.io/projects/hoi.html.

----

## [247] DALDet: Depth-Aware Learning Based Object Detection for Autonomous Driving

**Authors**: *Ke Hu, Tongbo Cao, Yuan Li, Song Chen, Yi Kang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i3.27996](https://doi.org/10.1609/aaai.v38i3.27996)

**Abstract**:

3D object detection achieves good detection performance in autonomous driving. However, it requires substantial computational resources, which prevents its practical application. 2D object detection has less computational burden but lacks spatial and geometric information embedded in depth. Therefore, we present DALDet, an efficient depth-aware learning based 2D detector, achieving high-performance object detection for autonomous driving. We design an efficient one-stage detection framework and seamlessly integrate depth cues into convolutional neural network by introducing depth-aware convolution and depth-aware average pooling, which effectively improve the detector's ability to perceive 3D space. Moreover, we propose a depth-guided loss function for training DALDet, which effectively improves the localization ability of the detector. Due to the use of depth map, DALDet can also output the distance of the object, which is of great importance for driving applications such as obstacle avoidance. Extensive experiments demonstrate the superiority and efficiency of DALDet. In particular, our DALDet ranks 1st on both KITTI Car and Cyclist 2D detection test leaderboards among all 2D detectors with high efficiency as well as yielding competitive performance among many leading 3D detectors. Code will be available at https://github.com/hukefy/DALDet.

----

## [248] COMMA: Co-articulated Multi-Modal Learning

**Authors**: *Lianyu Hu, Liqing Gao, Zekang Liu, Chi-Man Pun, Wei Feng*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i3.27997](https://doi.org/10.1609/aaai.v38i3.27997)

**Abstract**:

Pretrained large-scale vision-language models such as CLIP have demonstrated excellent generalizability over a series of downstream tasks. However, they are sensitive to the variation of input text prompts and need a selection of prompt templates to achieve satisfactory performance. Recently, various methods have been proposed to dynamically learn the prompts as the textual inputs to avoid the requirements of laboring hand-crafted prompt engineering in the fine-tuning process. We notice that these methods are suboptimal in two aspects. First, the prompts of the vision and language branches in these methods are usually separated or uni-directionally correlated. Thus, the prompts of both branches are not fully correlated and may not provide enough guidance to align the representations of both branches. Second, it's observed that most previous methods usually achieve better performance on seen classes but cause performance degeneration on unseen classes compared to CLIP. This is because the essential generic knowledge learned in the pretraining stage is partly forgotten in the fine-tuning process. In this paper, we propose Co-Articulated Multi-Modal Learning (COMMA) to handle the above limitations. Especially, our method considers prompts from both branches to generate the prompts to enhance the representation alignment of both branches. Besides, to alleviate forgetting about the essential knowledge, we minimize the feature discrepancy between the learned prompts and the embeddings of hand-crafted prompts in the pre-trained CLIP in the late transformer layers. We evaluate our method across three representative tasks of generalization to novel classes, new target datasets and unseen domain shifts. Experimental results demonstrate the superiority of our method by exhibiting a favorable performance boost upon all tasks with high efficiency. Code is available at https://github.com/hulianyuyy/COMMA.

----

## [249] Latent Space Editing in Transformer-Based Flow Matching

**Authors**: *Vincent Tao Hu, Wei Zhang, Meng Tang, Pascal Mettes, Deli Zhao, Cees Snoek*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i3.27998](https://doi.org/10.1609/aaai.v38i3.27998)

**Abstract**:

This paper strives for image editing via generative models. Flow Matching is an emerging generative modeling technique that offers the advantage of simple and efficient training. Simultaneously, a new transformer-based U-ViT has recently been proposed to replace the commonly used UNet for better scalability and performance in generative modeling. Hence, Flow Matching with a transformer backbone offers the potential for scalable and high-quality generative modeling, but their latent structure and editing ability are as of yet unknown. Hence, we adopt this setting and explore how to edit images through latent space manipulation. We introduce an editing space, which we call u-space, that can be manipulated in a controllable, accumulative, and composable manner. Additionally, we propose a tailored sampling solution to enable sampling with the more efficient adaptive step-size ODE solvers. Lastly, we put forth a straightforward yet powerful method for achieving fine-grained and nuanced editing using text prompts. Our framework is simple and efficient, all while being highly effective at editing images while preserving the essence of the original content. Our code will be publicly available at https://taohu.me/lfm/

----

## [250] BLIVA: A Simple Multimodal LLM for Better Handling of Text-Rich Visual Questions

**Authors**: *Wenbo Hu, Yifan Xu, Yi Li, Weiyue Li, Zeyuan Chen, Zhuowen Tu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i3.27999](https://doi.org/10.1609/aaai.v38i3.27999)

**Abstract**:

Vision Language Models (VLMs), which extend Large Language Models (LLM) by incorporating visual understanding capability, have demonstrated significant advancements in addressing open-ended visual question-answering (VQA) tasks. However, these models cannot accurately interpret images infused with text, a common occurrence in real-world scenarios. Standard procedures for extracting information from images often involve learning a fixed set of query embeddings. These embeddings are designed to encapsulate image contexts and are later used as soft prompt inputs in LLMs. Yet, this process is limited to the token count, potentially curtailing the recognition of scenes with text-rich context. To improve upon them, the present study introduces BLIVA: an augmented version of InstructBLIP with Visual Assistant. BLIVA incorporates the query embeddings from InstructBLIP and also directly projects encoded patch embeddings into the LLM, a technique inspired by LLaVA. This approach assists the model to capture intricate details potentially missed during the query decoding process. Empirical evidence demonstrates that our model, BLIVA, significantly enhances performance in processing text-rich VQA benchmarks (up to 17.76% in OCR-VQA benchmark) and in undertaking general (not particularly text-rich) VQA benchmarks (up to 7.9% in Visual Spatial Reasoning benchmark), and achieved 17.72% overall improvement in a comprehensive multimodal LLM benchmark (MME), comparing to our baseline InstructBLIP. BLIVA demonstrates significant capability in decoding real-world images, irrespective of text presence. To demonstrate the broad industry applications enabled by BLIVA, we evaluate the model using a new dataset comprising YouTube thumbnails paired with question-answer sets across 11 diverse categories. For researchers interested in further exploration, our code and models are freely accessible at https://github.com/mlpc-ucsd/BLIVA.

----

## [251] A Dynamic Learning Method towards Realistic Compositional Zero-Shot Learning

**Authors**: *Xiaoming Hu, Zilei Wang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i3.28000](https://doi.org/10.1609/aaai.v38i3.28000)

**Abstract**:

To tackle the challenge of recognizing images of unseen attribute-object compositions, Compositional Zero-Shot Learning (CZSL) methods have been previously addressed. However, test images in realistic scenarios may also incorporate other forms of unknown factors, such as novel semantic concepts or novel image styles. As previous CZSL works have overlooked this critical issue, in this research, we first propose the Realistic Compositional Zero-Shot Learning (RCZSL) task which considers the various types of unknown factors in an unified experimental setting. To achieve this, we firstly conduct re-labelling on MIT-States and use the pre-trained generative models to obtain images of various domains. Then the entire dataset is split into a training set and a test set, with the latter containing images of unseen concepts, unseen compositions, unseen domains as well as their combinations. Following this, we show that the visual-semantic relationship changes on unseen images, leading us to construct two dynamic modulators to adapt the visual features and composition prototypes in accordance with the input image. We believe that such a dynamic learning method could effectively alleviate the domain shift problem caused by various types of unknown factors. We conduct extensive experiments on benchmark datasets for both the conventional CZSL setting and the proposed RCZSL setting. The effectiveness of our method has been proven by empirical results, which significantly outperformed both our baseline method and state-of-the-art approaches.

----

## [252] LF-ViT: Reducing Spatial Redundancy in Vision Transformer for Efficient Image Recognition

**Authors**: *Youbing Hu, Yun Cheng, Anqi Lu, Zhiqiang Cao, Dawei Wei, Jie Liu, Zhijun Li*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i3.28001](https://doi.org/10.1609/aaai.v38i3.28001)

**Abstract**:

The Vision Transformer (ViT) excels in accuracy when handling high-resolution images, yet it confronts the challenge of significant spatial redundancy, leading to increased computational and memory requirements. To address this, we present the Localization and Focus Vision Transformer (LF-ViT). This model operates by strategically curtailing computational demands without impinging on performance. In the Localization phase, a reduced-resolution image is processed; if a definitive prediction remains elusive, our pioneering Neighborhood Global Class Attention (NGCA) mechanism is triggered, effectively identifying and spotlighting class-discriminative regions based on initial findings. Subsequently, in the Focus phase, this designated region is used from the original image to enhance recognition. Uniquely, LF-ViT employs consistent parameters across both phases, ensuring seamless end-to-end optimization. Our empirical tests affirm LF-ViT's prowess: it remarkably decreases Deit-S's FLOPs by 63% and concurrently amplifies throughput twofold. Code of this project is at https://github.com/edgeai1/LF-ViT.git.

----

## [253] O^2-Recon: Completing 3D Reconstruction of Occluded Objects in the Scene with a Pre-trained 2D Diffusion Model

**Authors**: *Yubin Hu, Sheng Ye, Wang Zhao, Matthieu Lin, Yuze He, Yu-Hui Wen, Ying He, Yong-Jin Liu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i3.28002](https://doi.org/10.1609/aaai.v38i3.28002)

**Abstract**:

Occlusion is a common issue in 3D reconstruction from RGB-D videos, often blocking the complete reconstruction of objects and presenting an ongoing problem. In this paper, we propose a novel framework, empowered by a 2D diffusion-based in-painting model, to reconstruct complete surfaces for the hidden parts of objects.
Specifically, we utilize a pre-trained diffusion model to fill in the hidden areas of 2D images. Then we use these in-painted images to optimize a neural implicit surface representation for each instance for 3D reconstruction.
Since creating the in-painting masks needed for this process is tricky, we adopt a human-in-the-loop strategy that involves very little human engagement to generate high-quality masks.
Moreover, some parts of objects can be totally hidden because the videos are usually shot from limited perspectives. To ensure recovering these invisible areas, we develop a cascaded network architecture for predicting signed distance field, making use of different frequency bands of positional encoding and maintaining overall smoothness.
Besides the commonly used rendering loss, Eikonal loss, and silhouette loss, we adopt a CLIP-based semantic consistency loss to guide the surface from unseen camera angles. 
Experiments on ScanNet scenes show that our proposed framework achieves state-of-the-art accuracy and completeness in object-level reconstruction from scene-level RGB-D videos. Code: https://github.com/THU-LYJ-Lab/O2-Recon.

----

## [254] Arbitrary-Scale Video Super-resolution Guided by Dynamic Context

**Authors**: *Cong Huang, Jiahao Li, Lei Chu, Dong Liu, Yan Lu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i3.28003](https://doi.org/10.1609/aaai.v38i3.28003)

**Abstract**:

We propose a Dynamic Context-Guided Upsampling (DCGU) module for video super-resolution (VSR) that leverages temporal context guidance to achieve efficient and effective arbitrary-scale VSR. While most VSR research focuses on backbone design,  the importance of the upsampling part is often overlooked. Existing methods rely on pixelshuffle-based upsampling, which has limited capabilities in handling arbitrary upsampling scales. Recent attempts to replace pixelshuffle-based modules with implicit neural function-based and filter-based approaches suffer from slow inference speeds and limited representation capacity, respectively. To overcome these limitations, our DCGU module predicts non-local sampling locations and content-dependent filter weights, enabling efficient and effective arbitrary-scale VSR.  Our proposed multi-granularity location search module efficiently identifies non-local sampling locations across the entire low-resolution  grid, and the temporal bilateral filter modulation module integrates content information with the filter weight to enhance textual details.  Extensive experiments demonstrate the superiority of our method in terms of performance and speed on arbitrary-scale VSR.

----

## [255] Dynamic Weighted Combiner for Mixed-Modal Image Retrieval

**Authors**: *Fuxiang Huang, Lei Zhang, Xiaowei Fu, Suqi Song*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i3.28004](https://doi.org/10.1609/aaai.v38i3.28004)

**Abstract**:

Mixed-Modal Image Retrieval (MMIR) as a flexible search paradigm has attracted wide attention. However, previous approaches always achieve limited performance, due to two critical factors are seriously overlooked. 1) The contribution of image and text modalities is different, but incorrectly treated equally. 2) There exist inherent labeling noises in describing users' intentions with text in web datasets from diverse real-world scenarios, giving rise to overfitting. We propose a Dynamic Weighted Combiner (DWC) to tackle the above challenges, which includes three merits. First, we propose an Editable Modality De-equalizer (EMD) by taking into account the contribution disparity between modalities, containing two modality feature editors and an adaptive weighted combiner. Second, to alleviate labeling noises and data bias, we propose a dynamic soft-similarity label generator (SSG) to implicitly improve noisy supervision. Finally, to bridge modality gaps and facilitate similarity learning, we propose a CLIP-based mutual enhancement module alternately trained by a mixed-modality contrastive loss. Extensive experiments verify that our proposed model significantly outperforms state-of-the-art methods on real-world datasets. The source code is available at https://github.com/fuxianghuang1/DWC.

----

## [256] NeuSurf: On-Surface Priors for Neural Surface Reconstruction from Sparse Input Views

**Authors**: *Han Huang, Yulun Wu, Junsheng Zhou, Ge Gao, Ming Gu, Yu-Shen Liu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i3.28005](https://doi.org/10.1609/aaai.v38i3.28005)

**Abstract**:

Recently, neural implicit functions have demonstrated remarkable results in the field of multi-view reconstruction. However, most existing methods are tailored for dense views and exhibit unsatisfactory performance when dealing with sparse views. Several latest methods have been proposed for generalizing implicit reconstruction to address the sparse view reconstruction task, but they still suffer from high training costs and are merely valid under carefully selected perspectives. In this paper, we propose a novel sparse view reconstruction framework that leverages on-surface priors to achieve highly faithful surface reconstruction.  Specifically, we design several constraints on global geometry alignment and local geometry refinement for jointly optimizing coarse shapes and fine details. To achieve this, we train a neural network to learn a global implicit field from the on-surface points obtained from SfM and then leverage it as a coarse geometric constraint. To exploit local geometric consistency, we project on-surface points onto seen and unseen views, treating the consistent loss of projected features as a fine geometric constraint. The experimental results with DTU and BlendedMVS datasets in two prevalent sparse settings demonstrate significant improvements over the state-of-the-art methods.

----

## [257] Seeing Dark Videos via Self-Learned Bottleneck Neural Representation

**Authors**: *Haofeng Huang, Wenhan Yang, Lingyu Duan, Jiaying Liu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i3.28006](https://doi.org/10.1609/aaai.v38i3.28006)

**Abstract**:

Enhancing low-light videos in a supervised style presents a set of challenges, including limited data diversity, misalignment, and the domain gap introduced through the dataset construction pipeline. Our paper tackles these challenges by constructing a self-learned enhancement approach that gets rid of the reliance on any external training data. The challenge of self-supervised learning lies in fitting high-quality signal representations solely from input signals. Our work designs a bottleneck neural representation mechanism that extracts those signals. More in detail, we encode the frame-wise representation with a compact deep embedding and utilize a neural network to parameterize the video-level manifold consistently. Then, an entropy constraint is applied to the enhanced results based on the adjacent spatial-temporal context to filter out the degraded visual signals, e.g. noise and frame inconsistency. Last, a novel Chromatic Retinex decomposition is proposed to effectively align the reflectance distribution temporally. It benefits the entropy control on different components of each frame and facilitates noise-to-noise training, successfully suppressing the temporal flicker. Extensive experiments demonstrate the robustness and superior effectiveness of our proposed method. Our project is publicly available at: https://huangerbai.github.io/SLBNR/.

----

## [258] Combinatorial CNN-Transformer Learning with Manifold Constraints for Semi-supervised Medical Image Segmentation

**Authors**: *Huimin Huang, Yawen Huang, Shiao Xie, Lanfen Lin, Ruofeng Tong, Yen-Wei Chen, Yuexiang Li, Yefeng Zheng*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i3.28007](https://doi.org/10.1609/aaai.v38i3.28007)

**Abstract**:

Semi-supervised learning (SSL), as one of the dominant methods, aims at leveraging the unlabeled data to deal with the annotation dilemma of supervised learning, which has attracted much attentions in the medical image segmentation. 
Most of the existing approaches leverage a unitary network by convolutional neural networks (CNNs) with compulsory consistency of the predictions through small perturbations applied to inputs or models. 
The penalties of such a learning paradigm are that (1) CNN-based models place severe limitations on global learning; (2) rich and diverse class-level distributions are inhibited. 
In this paper, we present a novel CNN-Transformer learning framework in the manifold space for semi-supervised medical image segmentation. 
First, at intra-student level, we propose a novel class-wise consistency loss to facilitate the learning of both discriminative and compact target feature representations. 
Then, at inter-student level, we align the CNN and Transformer features using a prototype-based optimal transport method. 
Extensive experiments show that our method outperforms previous state-of-the-art methods on three public medical image segmentation benchmarks.

----

## [259] Sparse Bayesian Deep Learning for Cross Domain Medical Image Reconstruction

**Authors**: *Jiaxin Huang, Qi Wu, Yazhou Ren, Fan Yang, Aodi Yang, Qianqian Yang, Xiaorong Pu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i3.28008](https://doi.org/10.1609/aaai.v38i3.28008)

**Abstract**:

Cross domain medical image reconstruction aims to address the issue that deep learning models trained solely on one source dataset might not generalize effectively to unseen target datasets from different hospitals. Some recent methods achieve satisfactory reconstruction performance, but often at the expense of extensive parameters and time consumption. To strike a balance between cross domain image reconstruction quality and model computational efficiency, we propose a lightweight sparse Bayesian deep learning method. Notably, we apply a fixed-form variational Bayes (FFVB) approach to quantify pixel-wise uncertainty priors derived from degradation distribution of the source domain. Furthermore, by integrating the uncertainty prior into the posterior sampled through stochastic gradient Langevin dynamics (SGLD), we develop a training strategy that dynamically generates and optimizes the prior distribution on the network weights for each unseen domain. This strategy enhances generalizability and ensures robust reconstruction performance. When evaluated on medical image reconstruction tasks, our proposed approach demonstrates impressive performance across various previously unseen domains.

----

## [260] UniCell: Universal Cell Nucleus Classification via Prompt Learning

**Authors**: *Junjia Huang, Haofeng Li, Xiang Wan, Guanbin Li*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i3.28009](https://doi.org/10.1609/aaai.v38i3.28009)

**Abstract**:

The recognition of multi-class cell nuclei can significantly facilitate the process of histopathological diagnosis. Numerous pathological datasets are currently available, but their annotations are inconsistent. Most existing methods require individual training on each dataset to deduce the relevant labels and lack the use of common knowledge across datasets, consequently restricting the quality of recognition. In this paper, we propose a universal cell nucleus classification framework (UniCell), which employs a novel prompt learning mechanism to uniformly predict the corresponding categories of pathological images from different dataset domains. In particular, our framework adopts an end-to-end architecture for nuclei detection and classification, and utilizes flexible prediction heads for adapting various datasets. Moreover, we develop a Dynamic Prompt Module (DPM) that exploits the properties of multiple datasets to enhance features. The DPM first integrates the embeddings of datasets and semantic categories, and then employs the integrated prompts to refine image representations, efficiently harvesting the shared knowledge among the related cell types and data sources. Experimental results demonstrate that the proposed method effectively achieves the state-of-the-art results on four nucleus detection and classification benchmarks. Code and models are available at https://github.com/lhaof/UniCell

----

## [261] SC-NeuS: Consistent Neural Surface Reconstruction from Sparse and Noisy Views

**Authors**: *Shi-Sheng Huang, Zi-Xin Zou, Yichi Zhang, Yan-Pei Cao, Ying Shan*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i3.28010](https://doi.org/10.1609/aaai.v38i3.28010)

**Abstract**:

The recent neural surface reconstruction approaches using volume rendering have made much progress by achieving impressive surface reconstruction quality, but are still limited to dense and highly accurate posed views. To overcome such drawbacks, this paper pays special attention on the consistent surface reconstruction from sparse views with noisy camera poses. Unlike previous approaches, the key difference of this paper is to exploit the multi-view constraints directly from the explicit geometry of the neural surface, which can be used as effective regularization to jointly learn the neural surface and refine the camera poses. To build effective multi-view constraints, we introduce a fast differentiable on-surface intersection to generate on-surface points, and propose view-consistent losses on such differentiable points to regularize the neural surface learning. Based on this point, we propose a joint learning strategy, named SC-NeuS, to perform geometry-consistent surface reconstruction in an end-to-end manner. With extensive evaluation on public datasets, our SC-NeuS can achieve consistently better surface reconstruction results with fine-grained details than previous approaches, especially from sparse and noisy camera views. The source code is available at https://github.com/zouzx/sc-neus.git.

----

## [262] MFTN: Multi-Level Feature Transfer Network Based on MRI-Transformer for MR Image Super-resolution

**Authors**: *Shuying Huang, Ge Chen, Yong Yang, Xiaozheng Wang, Chenbin Liang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i3.28011](https://doi.org/10.1609/aaai.v38i3.28011)

**Abstract**:

Due to the unique environment and inherent properties of magnetic resonance imaging (MRI) instruments, MR images typically have lower resolution. Therefore, improving the resolution of MR images is beneficial for assisting doctors in diagnosing the condition. Currently, the existing MR image super-resolution (SR) methods still have the problem of insufficient detail reconstruction. To overcome this issue, this paper proposes a multi-level feature transfer network (MFTN) based on MRI-Transformer to realize SR of low-resolution MRI data. MFTN consists of a multi-scale feature reconstruction network (MFRN) and a multi-level feature extraction branch (MFEB). MFRN is constructed as a pyramid structure to gradually reconstruct image features at different scales by integrating the features obtained from MFEB, and MFEB is constructed to provide detail information at different scales for low resolution MR image SR reconstruction by constructing multiple MRI-Transformer modules. Each MRI-Transformer module is designed to learn the transfer features from the reference image by establishing feature correlations between the reference image and low-resolution MR image. In addition, a contrast learning constraint item is added to the loss function to enhance the texture details of the SR image. A large number of experiments show that our network can effectively reconstruct high-quality MR Images and achieves better performance compared to some state-of-the-art methods. The source code of this work will be released on GitHub.

----

## [263] SDGAN: Disentangling Semantic Manipulation for Facial Attribute Editing

**Authors**: *Wenmin Huang, Weiqi Luo, Jiwu Huang, Xiaochun Cao*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i3.28012](https://doi.org/10.1609/aaai.v38i3.28012)

**Abstract**:

Facial attribute editing has garnered significant attention, yet prevailing methods struggle with achieving precise attribute manipulation while preserving irrelevant details and controlling attribute styles. This challenge primarily arises from the strong correlations between different attributes and the interplay between attributes and identity. In this paper, we propose Semantic Disentangled GAN (SDGAN), a novel method addressing this challenge. SDGAN introduces two key concepts: a semantic disentanglement generator that assigns facial representations to distinct attribute-specific editing modules, enabling the decoupling of the facial attribute editing process, and a semantic mask alignment strategy that confines attribute editing to appropriate regions, thereby avoiding undesired modifications. Leveraging these concepts, SDGAN demonstrates accurate attribute editing and achieves high-quality attribute style manipulation through both latent-guided and reference-guided manners. We extensively evaluate our method on the CelebA-HQ database, providing both qualitative and quantitative analyses. Our results establish that SDGAN significantly outperforms state-of-the-art techniques, showcasing the effectiveness of our approach. To foster reproducibility and further research, we will provide the code for our method.

----

## [264] Frozen CLIP Transformer Is an Efficient Point Cloud Encoder

**Authors**: *Xiaoshui Huang, Zhou Huang, Sheng Li, Wentao Qu, Tong He, Yuenan Hou, Yifan Zuo, Wanli Ouyang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i3.28013](https://doi.org/10.1609/aaai.v38i3.28013)

**Abstract**:

The pretrain-finetune paradigm has achieved great success in NLP and 2D image fields because of the high-quality representation ability and transferability of their pretrained models. However, pretraining such a strong model is difficult in the 3D point cloud field due to the limited amount of point cloud sequences. This paper introduces Efficient Point Cloud Learning (EPCL), an effective and efficient point cloud learner for directly training high-quality point cloud models with a frozen CLIP transformer. Our EPCL connects the 2D and 3D modalities by semantically aligning the image features and point cloud features without paired 2D-3D data.  Specifically, the input point cloud is divided into a series of local patches, which are converted to token embeddings by the designed point cloud tokenizer. These token embeddings are concatenated with a task token and fed into the frozen CLIP transformer to learn point cloud representation. The intuition is that the proposed point cloud tokenizer projects the input point cloud into a unified token space that is similar to the 2D images.  Comprehensive experiments on 3D detection, semantic segmentation, classification and few-shot learning demonstrate that the CLIP transformer can serve as an efficient point cloud encoder and our method achieves promising performance on both indoor and outdoor benchmarks. In particular, performance gains brought by our EPCL are 19.7 AP50 on ScanNet V2 detection, 4.4 mIoU on S3DIS segmentation and 1.2 mIoU on SemanticKITTI segmentation compared to contemporary pretrained models. Code is available at \url{https://github.com/XiaoshuiHuang/EPCL}.

----

## [265] G2L-CariGAN: Caricature Generation from Global Structure to Local Features

**Authors**: *Xin Huang, Yunfeng Bai, Dong Liang, Feng Tian, Jinyuan Jia*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i3.28014](https://doi.org/10.1609/aaai.v38i3.28014)

**Abstract**:

Existing GAN-based approaches to caricature generation mainly focus on exaggerating a character’s global facial structure. This often leads to the failure in highlighting significant facial features such as big eyes and hook nose. To address this limitation, we propose a new approach termed as G2L-CariGAN, which uses feature maps of spatial dimensions instead of latent codes for geometric exaggeration. G2L-CariGAN first exaggerates the global facial structure of the character on a low-dimensional feature map and then exaggerates its local facial features on a high-dimensional feature map. Moreover, we develop a caricature identity loss function based on feature maps, which well retains the character's identity after exaggeration. Our experiments have demonstrated that G2L-CariGAN outperforms the state-of-arts in terms of the quality of exaggerating a character and retaining its identity.

----

## [266] 3D Visibility-Aware Generalizable Neural Radiance Fields for Interacting Hands

**Authors**: *Xuan Huang, Hanhui Li, Zejun Yang, Zhisheng Wang, Xiaodan Liang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i3.28015](https://doi.org/10.1609/aaai.v38i3.28015)

**Abstract**:

Neural radiance fields (NeRFs) are promising 3D representations for scenes, objects, and humans. However, most existing methods require multi-view inputs and per-scene training, which limits their real-life applications. Moreover, current methods focus on single-subject cases, leaving scenes of interacting hands that involve severe inter-hand occlusions and challenging view variations remain unsolved. To tackle these issues, this paper proposes a generalizable visibility-aware NeRF (VA-NeRF) framework for interacting hands. Specifically, given an image of interacting hands as input, our VA-NeRF first obtains a mesh-based representation of hands and extracts their corresponding geometric and textural features. Subsequently, a feature fusion module that exploits the visibility of query points and mesh vertices is introduced to adaptively merge features of both hands, enabling the recovery of features in unseen areas. Additionally, our VA-NeRF is optimized together with a novel discriminator within an adversarial learning paradigm. In contrast to conventional discriminators that predict a single real/fake label for the synthesized image, the proposed discriminator generates a pixel-wise visibility map, providing fine-grained supervision for unseen areas and encouraging the VA-NeRF to improve the visual quality of synthesized images. Experiments on the Interhand2.6M dataset demonstrate that our proposed VA-NeRF outperforms conventional NeRFs significantly. Project Page: https://github.com/XuanHuang0/VANeRF.

----

## [267] Sunshine to Rainstorm: Cross-Weather Knowledge Distillation for Robust 3D Object Detection

**Authors**: *Xun Huang, Hai Wu, Xin Li, Xiaoliang Fan, Chenglu Wen, Cheng Wang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i3.28016](https://doi.org/10.1609/aaai.v38i3.28016)

**Abstract**:

LiDAR-based 3D object detection models inevitably struggle under rainy conditions due to the degraded and noisy scanning signals. Previous research has attempted to address this by simulating the noise from rain to improve the robustness of detection models. However, significant disparities exist between simulated and actual rain-impacted data points. In this work, we propose a novel rain simulation method, termed DRET, that unifies Dynamics and Rainy Environment Theory to provide a cost-effective means of expanding the available realistic rain data for 3D detection training. Furthermore, we present a Sunny-to-Rainy Knowledge Distillation (SRKD) approach to enhance 3D detection under rainy conditions. Extensive experiments on the Waymo-Open-Dataset show that, when combined with the state-of-the-art DSVT model and other classical 3D detectors, our proposed framework demonstrates significant detection accuracy improvements, without losing efficiency. Remarkably, our framework also improves detection capabilities under sunny conditions, therefore offering a robust solution for 3D detection regardless of whether the weather is rainy or sunny.

----

## [268] Structure-CLIP: Towards Scene Graph Knowledge to Enhance Multi-Modal Structured Representations

**Authors**: *Yufeng Huang, Jiji Tang, Zhuo Chen, Rongsheng Zhang, Xinfeng Zhang, Weijie Chen, Zeng Zhao, Zhou Zhao, Tangjie Lv, Zhipeng Hu, Wen Zhang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i3.28017](https://doi.org/10.1609/aaai.v38i3.28017)

**Abstract**:

Large-scale vision-language pre-training has achieved significant performance in multi-modal understanding and generation tasks. However, existing methods often perform poorly on image-text matching tasks that require structured representations, i.e., representations of objects, attributes, and relations. The models cannot make a distinction between "An astronaut rides a horse" and "A horse rides an astronaut". This is because they fail to fully leverage structured knowledge when learning multi-modal representations. In this paper, we present an end-to-end framework Structure-CLIP, which integrates Scene Graph Knowledge (SGK) to enhance multi-modal structured representations. Firstly, we use scene graphs to guide the construction of semantic negative examples, which results in an increased emphasis on learning structured representations. Moreover, a Knowledge-Enhance Encoder (KEE) is proposed to leverage SGK as input to further enhance structured representations. To verify the effectiveness of the proposed framework, we pre-train our model with the aforementioned approaches and conduct experiments on downstream tasks.  Experimental results demonstrate that Structure-CLIP achieves state-of-the-art (SOTA) performance on VG-Attribution and VG-Relation datasets, with 12.5% and 4.1% ahead of the multi-modal SOTA model respectively. Meanwhile, the results on MSCOCO indicate that Structure-CLIP significantly enhances the structured representations while maintaining the ability of general representations. Our code is available at https://github.com/zjukg/Structure-CLIP.

----

## [269] Voxel or Pillar: Exploring Efficient Point Cloud Representation for 3D Object Detection

**Authors**: *Yuhao Huang, Sanping Zhou, Junjie Zhang, Jinpeng Dong, Nanning Zheng*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i3.28018](https://doi.org/10.1609/aaai.v38i3.28018)

**Abstract**:

Efficient representation of point clouds is fundamental for LiDAR-based 3D object detection. While recent grid-based detectors often encode point clouds into either voxels or pillars, the distinctions between these approaches remain underexplored. In this paper, we quantify the differences between the current encoding paradigms and highlight the limited vertical learning within. To tackle these limitations, we propose a hybrid detection framework named Voxel-Pillar Fusion (VPF), which synergistically combines the unique strengths of both voxels and pillars. To be concrete, we first develop a sparse voxel-pillar encoder that encodes point clouds into voxel and pillar features through 3D and 2D sparse convolutions respectively, and then introduce the Sparse Fusion Layer (SFL), facilitating bidirectional interaction between sparse voxel and pillar features. Our computationally efficient, fully sparse method can be seamlessly integrated into both dense and sparse detectors. Leveraging this powerful yet straightforward representation, VPF delivers competitive performance, achieving real-time inference speeds on the nuScenes and Waymo Open Dataset.

----

## [270] COMBAT: Alternated Training for Effective Clean-Label Backdoor Attacks

**Authors**: *Tran Huynh, Dang Nguyen, Tung Pham, Anh Tran*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i3.28019](https://doi.org/10.1609/aaai.v38i3.28019)

**Abstract**:

Backdoor attacks pose a critical concern to the practice of using third-party data for AI development. The data can be poisoned to make a trained model misbehave when a predefined trigger pattern appears, granting the attackers illegal benefits. While most proposed backdoor attacks are dirty-label, clean-label attacks are more desirable by keeping data labels unchanged to dodge human inspection. However, designing a working clean-label attack is a challenging task, and existing clean-label attacks show underwhelming performance. In this paper, we propose a novel mechanism to develop clean-label attacks with outstanding attack performance. The key component is a trigger pattern generator, which is trained together with a surrogate model in an alternating manner. Our proposed mechanism is flexible and customizable, allowing different backdoor trigger types and behaviors for either single or multiple target labels. Our backdoor attacks can reach near-perfect attack success rates and bypass all state-of-the-art backdoor defenses, as illustrated via comprehensive experiments on standard benchmark datasets. Our code is available at https://github.com/VinAIResearch/COMBAT.

----

## [271] MagiCapture: High-Resolution Multi-Concept Portrait Customization

**Authors**: *Junha Hyung, Jaeyo Shin, Jaegul Choo*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i3.28020](https://doi.org/10.1609/aaai.v38i3.28020)

**Abstract**:

Large-scale text-to-image models including Stable Diffusion are capable of generating high-fidelity photorealistic portrait images. There is an active research area dedicated to personalizing these models, aiming to synthesize specific subjects or styles using provided sets of reference images. However, despite the plausible results from these personalization methods, they tend to produce images that often fall short of realism and are not yet on a commercially viable level. This is particularly noticeable in portrait image generation, where any unnatural artifact in human faces is easily discernible due to our inherent human bias. To address this, we introduce MagiCapture, a personalization method for integrating subject and style concepts to generate high-resolution portrait images using just a few subject and style references. For instance, given a handful of random selfies, our fine-tuned model can generate high-quality portrait images in specific styles, such as passport or profile photos. The main challenge with this task is the absence of ground truth for the composed concepts, leading to a reduction in the quality of the final output and an identity shift of the source subject. To address these issues, we present a novel Attention Refocusing loss coupled with auxiliary priors, both of which facilitate robust learning within this weakly supervised learning setting. Our pipeline also includes additional post-processing steps to ensure the creation of highly realistic outputs. MagiCapture outperforms other baselines in both quantitative and qualitative evaluations and can also be generalized to other non-human objects.

----

## [272] Rethinking Peculiar Images by Diffusion Models: Revealing Local Minima's Role

**Authors**: *Jinhyeok Jang, Chan-Hyun Youn, Minsu Jeon, Changha Lee*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i3.28021](https://doi.org/10.1609/aaai.v38i3.28021)

**Abstract**:

Recent significant advancements in diffusion models have revolutionized image generation, enabling the synthesis of highly realistic images with text-based guidance. These breakthroughs have paved the way for constructing datasets via generative artificial intelligence (AI), offering immense potential for various applications. However, two critical challenges hinder the widespread adoption of synthesized data: computational cost and the generation of peculiar images. While computational costs have improved through various approaches, the issue of peculiar image generation remains relatively unexplored. Existing solutions rely on heuristics, extra training, or AI-based post-processing to mitigate this problem. In this paper, we present a novel approach to address both issues simultaneously. We establish that both gradient descent and diffusion sampling are specific cases of the generalized expectation maximization algorithm. We hypothesize and empirically demonstrate that peculiar image generation is akin to the local minima problem in optimization. Inspired by optimization techniques, we apply naive momentum and positive-negative momentum to diffusion sampling. Last, we propose new metrics to evaluate the peculiarity. Experimental results show momentum effectively prevents peculiar image generation without extra computation.

----

## [273] ProxyDet: Synthesizing Proxy Novel Classes via Classwise Mixup for Open-Vocabulary Object Detection

**Authors**: *Joonhyun Jeong, Geondo Park, Jayeon Yoo, Hyungsik Jung, Heesu Kim*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i3.28022](https://doi.org/10.1609/aaai.v38i3.28022)

**Abstract**:

Open-vocabulary object detection (OVOD) aims to recognize novel objects whose categories are not included in the training set. In order to classify these unseen classes during training, many OVOD frameworks leverage the zero-shot capability of largely pretrained vision and language models, such as CLIP. To further improve generalization on the unseen novel classes, several approaches proposed to additionally train with pseudo region labeling on the external data sources that contain a substantial number of novel category labels beyond the existing training data. Albeit its simplicity, these pseudo-labeling methods still exhibit limited improvement with regard to the truly unseen novel classes that were not pseudo-labeled. In this paper, we present a novel, yet simple technique that helps generalization on the overall distribution of novel classes. Inspired by our observation that numerous novel classes reside within the convex hull constructed by the base (seen) classes in the CLIP embedding space, we propose to synthesize proxy-novel classes approximating novel classes via linear mixup between a pair of base classes. By training our detector with these synthetic proxy-novel classes, we effectively explore the embedding space of novel classes. The experimental results on various OVOD benchmarks such as LVIS and COCO demonstrate superior performance on novel classes compared to the other state-of-the-art methods. Code is available at https://github.com/clovaai/ProxyDet.

----

## [274] A Diffusion Model with State Estimation for Degradation-Blind Inverse Imaging

**Authors**: *Liya Ji, Zhefan Rao, Sinno Jialin Pan, Chenyang Lei, Qifeng Chen*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i3.28023](https://doi.org/10.1609/aaai.v38i3.28023)

**Abstract**:

Solving the task of inverse imaging problems can restore unknown clean images from input measurements that have incomplete information. Utilizing powerful generative models, such as denoising diffusion models, could better tackle the ill-posed issues of inverse problems with the distribution prior of the unknown clean images. We propose a learnable state-estimator-based diffusion model to incorporate the measurements into the reconstruction process. Our method makes efficient use of the pre-trained diffusion models with computational feasibility compared to the conditional diffusion models, which need to be trained from scratch. In addition, our pipeline does not require explicit knowledge of the image degradation operator or make the assumption of its form, unlike many other works that use the pre-trained diffusion models at the test time. The experiments on three typical inverse imaging problems (both linear and non-linear), inpainting, deblurring, and JPEG compression restoration, have comparable results with the state-of-the-art methods.

----

## [275] SSMG: Spatial-Semantic Map Guided Diffusion Model for Free-Form Layout-to-Image Generation

**Authors**: *Chengyou Jia, Minnan Luo, Zhuohang Dang, Guang Dai, Xiaojun Chang, Mengmeng Wang, Jingdong Wang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i3.28024](https://doi.org/10.1609/aaai.v38i3.28024)

**Abstract**:

Despite significant progress in Text-to-Image (T2I) generative models, even lengthy and complex text descriptions still struggle to convey detailed controls. In contrast, Layout-to-Image (L2I) generation, aiming to generate realistic and complex scene images from user-specified layouts, has risen to prominence. However, existing methods transform layout information into tokens or RGB images for conditional control in the generative process, leading to insufficient spatial and semantic controllability of individual instances. To address these limitations, we propose a novel Spatial-Semantic Map Guided (SSMG) diffusion model that adopts the feature map, derived from the layout, as guidance. Owing to rich spatial and semantic information encapsulated in well-designed feature maps, SSMG achieves superior generation quality with sufficient spatial and semantic controllability compared to previous works. Additionally, we propose the Relation-Sensitive Attention (RSA) and Location-Sensitive Attention (LSA) mechanisms. The former aims to model the relationships among multiple objects within scenes while the latter is designed to heighten the model's sensitivity to the spatial information embedded in the guidance. Extensive experiments demonstrate that SSMG achieves highly promising results, setting a new state-of-the-art across a range of metrics encompassing fidelity, diversity, and controllability.

----

## [276] TiMix: Text-Aware Image Mixing for Effective Vision-Language Pre-training

**Authors**: *Chaoya Jiang, Wei Ye, Haiyang Xu, Qinghao Ye, Ming Yan, Ji Zhang, Shikun Zhang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i3.28025](https://doi.org/10.1609/aaai.v38i3.28025)

**Abstract**:

Self-supervised Multi-modal Contrastive Learning (SMCL) remarkably advances modern Vision-Language Pre-training (VLP) models by aligning visual and linguistic modalities. Due to noises in web-harvested text-image pairs, however, scaling up training data volume in SMCL presents considerable obstacles in terms of computational cost and data inefficiency. To improve data efficiency in VLP, we propose Text-aware Image Mixing (TiMix), which integrates mix-based data augmentation techniques into SMCL, yielding significant performance improvements without significantly increasing computational overhead. We provide a theoretical analysis of TiMix from a mutual information (MI) perspective, showing that mixed data samples for cross-modal contrastive learning implicitly serve as a regularizer for  the contrastive loss. The experimental results demonstrate that TiMix exhibits a comparable performance on downstream tasks, even with a reduced amount of training data and shorter training time, when benchmarked against existing methods. This work empirically and theoretically demonstrates the potential of data mixing for data-efficient and computationally viable VLP, benefiting broader VLP model adoption in practical scenarios. Our code is available on https://github.com/chaoyajiang/TiMiX/tree/main.

----

## [277] Revealing the Proximate Long-Tail Distribution in Compositional Zero-Shot Learning

**Authors**: *Chenyi Jiang, Haofeng Zhang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i3.28026](https://doi.org/10.1609/aaai.v38i3.28026)

**Abstract**:

Compositional Zero-Shot Learning (CZSL) aims to transfer knowledge from seen state-object pairs to novel unseen pairs. In this process, visual bias caused by the diverse interrelationship of state-object combinations blurs their visual features, hindering the learning of distinguishable class prototypes. Prevailing methods concentrate on disentangling states and objects directly from visual features, disregarding potential enhancements that could arise from a data viewpoint. Experimentally, we unveil the results caused by the above problem closely approximate the long-tailed distribution. As a solution, we transform CZSL into a proximate class imbalance problem. We mathematically deduce the role of class prior within the long-tailed distribution in CZSL. Building upon this insight, we incorporate visual bias caused by compositions into the classifier's training and inference by estimating it as a proximate class prior. This enhancement encourages the classifier to acquire more discernible class prototypes for each composition, thereby achieving more balanced predictions. Experimental results demonstrate that our approach elevates the model's performance to the state-of-the-art level, without introducing additional parameters.

----

## [278] MWSIS: Multimodal Weakly Supervised Instance Segmentation with 2D Box Annotations for Autonomous Driving

**Authors**: *Guangfeng Jiang, Jun Liu, Yuzhi Wu, Wenlong Liao, Tao He, Pai Peng*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i3.28027](https://doi.org/10.1609/aaai.v38i3.28027)

**Abstract**:

Instance segmentation is a fundamental research in computer vision, especially in autonomous driving. However, manual mask annotation for instance segmentation is quite time-consuming and costly. To address this problem, some prior works attempt to apply weakly supervised manner by exploring 2D or 3D boxes. However, no one has ever successfully segmented 2D and 3D instances simultaneously by only using 2D box annotations, which could further reduce the annotation cost by an order of magnitude. Thus, we propose a novel framework called Multimodal Weakly Supervised Instance Segmentation (MWSIS), which incorporates various fine-grained label correction modules for both 2D and 3D modalities, along with a new multimodal cross-supervision approach. In the 2D pseudo label generation branch, the Instance-based Pseudo Mask Generation (IPG) module utilizes predictions for self-supervised correction. Similarly, in the 3D pseudo label generation branch, the Spatial-based Pseudo Label Generation (SPG) module generates pseudo labels by incorporating the spatial prior information of the point cloud. To further refine the generated pseudo labels, the Point-based Voting Label Correction (PVC) module utilizes historical predictions for correction. Additionally, a Ring Segment-based Label Correction (RSC) module is proposed to refine the predictions by leveraging the depth prior information from the point cloud. Finally, the Consistency Sparse Cross-modal Supervision (CSCS) module reduces the inconsistency of multimodal predictions by response distillation. Particularly, transferring the 3D backbone to downstream tasks not only improves the performance of the 3D detectors, but also outperforms fully supervised instance segmentation with only 5% fully supervised annotations. On the Waymo dataset, the proposed framework demonstrates significant improvements over the baseline, especially achieving 2.59% mAP and 12.75% mAP increases for 2D and 3D instance segmentation tasks, respectively. The code is available at https://github.com/jiangxb98/mwsis-plugin.

----

## [279] Transferable Video Moment Localization by Moment-Guided Query Prompting

**Authors**: *Hao Jiang, Yang Yizhang, Yadong Mu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i3.28028](https://doi.org/10.1609/aaai.v38i3.28028)

**Abstract**:

Video moment localization stands as a crucial task within the realm of computer vision, entailing the identification of temporal moments in untrimmed videos that bear semantic relevance to the supplied natural language queries. This work delves into a relatively unexplored facet of the task: the transferability of video moment localization models. This concern is addressed by evaluating moment localization models within a cross-domain transfer setting. In this setup, we curate multiple datasets distinguished by substantial domain gaps. The model undergoes training on one of these datasets, while validation and testing are executed using the remaining datasets. To confront the challenges inherent in this scenario, we draw inspiration from the recently introduced large-scale pre-trained vision-language models. Our focus is on exploring how the strategic utilization of these resources can bolster the capabilities of a model designed for video moment localization. Nevertheless, the distribution of language queries in video moment localization usually diverges from the text used by pre-trained models, exhibiting distinctions in aspects such as length, content, expression, and more. To mitigate the gap, this work proposes a Moment-Guided Query Prompting (MGQP) method for video moment localization. Our key idea is to generate multiple distinct and complementary prompt primitives through stratification of the original queries. Our approach is comprised of a prompt primitive constructor, a multimodal prompt refiner, and a holistic prompt incorporator. We carry out extensive experiments on Charades-STA, TACoS, DiDeMo, and YouCookII datasets, and investigate the efficacy of the proposed method using various pre-trained models, such as CLIP, ActionCLIP, CLIP4Clip, and VideoCLIP. The experimental results demonstrate the effectiveness of our proposed method.

----

## [280] In-Hand 3D Object Reconstruction from a Monocular RGB Video

**Authors**: *Shijian Jiang, Qi Ye, Rengan Xie, Yuchi Huo, Xiang Li, Yang Zhou, Jiming Chen*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i3.28029](https://doi.org/10.1609/aaai.v38i3.28029)

**Abstract**:

Our work aims to reconstruct a 3D object that is held and rotated by a hand in front of a static RGB camera. Previous methods that use implicit neural representations to recover the geometry of a generic hand-held object from multi-view images achieved compelling results in the visible part of the object. However, these methods falter in accurately capturing the shape within the hand-object contact region due to occlusion. In this paper, we propose a novel method that deals with surface reconstruction under occlusion by incorporating priors of 2D occlusion elucidation and physical contact constraints. For the former, we introduce an object amodal completion network to infer the 2D complete mask of objects under occlusion. To ensure the accuracy and view consistency of the predicted 2D amodal masks, we devise a joint optimization method for both amodal mask refinement and 3D reconstruction. For the latter, we impose penetration and attraction constraints on the local geometry in contact regions. We evaluate our approach on HO3D and HOD datasets and demonstrate that it outperforms the state-of-the-art methods in terms of reconstruction surface quality, with an improvement of 52% on HO3D and 20% on HOD. Project webpage: https://east-j.github.io/ihor.

----

## [281] AACP: Aesthetics Assessment of Children's Paintings Based on Self-Supervised Learning

**Authors**: *Shiqi Jiang, Ning Li, Chen Shi, Liping Guo, Changbo Wang, Chenhui Li*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i3.28030](https://doi.org/10.1609/aaai.v38i3.28030)

**Abstract**:

The Aesthetics Assessment of Children's Paintings (AACP) is an important branch of the image aesthetics assessment (IAA), playing a significant role in children's education. This task presents unique challenges, such as limited available data and the requirement for evaluation metrics from multiple perspectives. However, previous approaches have relied on training large datasets and subsequently providing an aesthetics score to the image, which is not applicable to AACP. To solve this problem, we construct an aesthetics assessment dataset of children's paintings and a model based on self-supervised learning. 1) We build a novel dataset composed of two parts: the first part contains more than 20k unlabeled images of children's paintings; the second part contains 1.2k images of children's paintings, and each image contains eight attributes labeled by multiple design experts. 2) We design a pipeline that includes a feature extraction module, perception modules and a disentangled evaluation module. 3) We conduct both qualitative and quantitative experiments to compare our model's performance with five other methods using the AACP dataset. Our experiments reveal that our method can accurately capture aesthetic features and achieve state-of-the-art performance.

----

## [282] Exploring Self- and Cross-Triplet Correlations for Human-Object Interaction Detection

**Authors**: *Weibo Jiang, Weihong Ren, Jiandong Tian, Liangqiong Qu, Zhiyong Wang, Honghai Liu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i3.28031](https://doi.org/10.1609/aaai.v38i3.28031)

**Abstract**:

Human-Object Interaction (HOI) detection plays a vital role in scene understanding, which aims to predict the HOI triplet in the form of . Existing methods mainly extract multi-modal features (e.g., appearance, object semantics, human pose) and then fuse them together to directly predict HOI triplets. However, most of these methods focus on seeking for self-triplet aggregation, but ignore the potential cross-triplet dependencies, resulting in ambiguity of action prediction. In this work, we propose to explore Self- and Cross-Triplet Correlations (SCTC) for HOI detection. Specifically, we regard each triplet proposal as a graph where Human, Object represent nodes and Action indicates edge, to aggregate self-triplet correlation. Also, we try to explore cross-triplet dependencies by jointly considering instance-level, semantic-level, and layout-level relations. Besides, we leverage the CLIP model to assist our SCTC obtain interaction-aware feature by knowledge distillation, which provides useful action clues for HOI detection. Extensive experiments on HICO-DET and V-COCO datasets verify the effectiveness of our proposed SCTC.

----

## [283] Comprehensive Visual Grounding for Video Description

**Authors**: *Wenhui Jiang, Yibo Cheng, Linxin Liu, Yuming Fang, Yuxin Peng, Yang Liu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i3.28032](https://doi.org/10.1609/aaai.v38i3.28032)

**Abstract**:

The grounding accuracy of existing video captioners is still behind the expectation. The majority of existing methods perform grounded video captioning on sparse entity annotations, whereas the captioning accuracy often suffers from degenerated object appearances on the annotated area such as motion blur and video defocus. Moreover, these methods seldom consider the complex interactions among entities. In this paper, we propose a comprehensive visual grounding network to improve video captioning, by explicitly linking the entities and actions to the visual clues across the video frames. Specifically, the network consists of spatial-temporal entity grounding and action grounding. The proposed entity grounding encourages the attention mechanism to focus on informative spatial areas across video frames, albeit the entity is annotated in only one frame of a video. The action grounding dynamically associates the verbs to related subjects and the corresponding context, which keeps fine-grained spatial and temporal details for action prediction. Both entity grounding and action grounding are formulated as a unified task guided by a soft grounding supervision, which brings architecture simplification and improves training efficiency as well. We conduct extensive experiments on two challenging datasets, and demonstrate significant performance improvements of +2.3 CIDEr on ActivityNet-Entities and +2.2 CIDEr on MSR-VTT compared to state-of-the-arts.

----

## [284] Far3D: Expanding the Horizon for Surround-View 3D Object Detection

**Authors**: *Xiaohui Jiang, Shuailin Li, Yingfei Liu, Shihao Wang, Fan Jia, Tiancai Wang, Lijin Han, Xiangyu Zhang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i3.28033](https://doi.org/10.1609/aaai.v38i3.28033)

**Abstract**:

Recently 3D object detection from surround-view images has made notable advancements with its low deployment cost. However, most works have primarily focused on close perception range while leaving long-range detection less explored. Expanding existing methods directly to cover long distances poses challenges such as heavy computation costs and unstable convergence. To address these limitations, this paper proposes a novel sparse query-based framework, dubbed Far3D. By utilizing high-quality 2D object priors, we generate 3D adaptive queries that complement the 3D global queries. To efficiently capture discriminative features across different views and scales for long-range objects, we introduce a perspective-aware aggregation module. Additionally, we propose a range-modulated 3D denoising approach to address query error propagation and mitigate convergence issues in long-range tasks. Significantly, Far3D demonstrates SoTA performance on the challenging Argoverse 2 dataset, covering a wide range of 150 meters, surpassing several LiDAR-based approaches. The code is available at https://github.com/megvii-research/Far3D.

----

## [285] Delving into Multimodal Prompting for Fine-Grained Visual Classification

**Authors**: *Xin Jiang, Hao Tang, Junyao Gao, Xiaoyu Du, Shengfeng He, Zechao Li*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i3.28034](https://doi.org/10.1609/aaai.v38i3.28034)

**Abstract**:

Fine-grained visual classification (FGVC) involves categorizing fine subdivisions within a broader category, which poses challenges due to subtle inter-class discrepancies and large intra-class variations. However, prevailing approaches primarily focus on uni-modal visual concepts. Recent advancements in pre-trained vision-language models have demonstrated remarkable performance in various high-level vision tasks, yet the applicability of such models to FGVC tasks remains uncertain. In this paper, we aim to fully exploit the capabilities of cross-modal description to tackle FGVC tasks and propose a novel multimodal prompting solution, denoted as MP-FGVC, based on the contrastive language-image pertaining (CLIP) model. Our MP-FGVC comprises a multimodal prompts scheme and a multimodal adaptation scheme. The former includes Subcategory-specific Vision Prompt (SsVP) and Discrepancy-aware Text Prompt (DaTP), which explicitly highlights the subcategory-specific discrepancies from the perspectives of both vision and language. The latter aligns the vision and text prompting elements in a common semantic space, facilitating cross-modal collaborative reasoning through a Vision-Language Fusion Module (VLFM) for further improvement on FGVC. Moreover, we tailor a two-stage optimization strategy for MP-FGVC to fully leverage the pre-trained CLIP model and expedite efficient adaptation for FGVC. Extensive experiments conducted on four FGVC datasets demonstrate the effectiveness of our MP-FGVC.

----

## [286] MCA: Moment Channel Attention Networks

**Authors**: *Yangbo Jiang, Zhiwei Jiang, Le Han, Zenan Huang, Nenggan Zheng*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i3.28035](https://doi.org/10.1609/aaai.v38i3.28035)

**Abstract**:

Channel attention mechanisms endeavor to recalibrate channel weights to enhance representation abilities of networks. However, mainstream methods often rely solely on global average pooling as the feature squeezer, which significantly limits the overall potential of models. In this paper, we investigate the statistical moments of feature maps within a neural network. Our findings highlight the critical role of high-order moments in enhancing model capacity. Consequently, we introduce a flexible and comprehensive mechanism termed Extensive Moment Aggregation (EMA) to capture the global spatial context. Building upon this mechanism, we propose the Moment Channel Attention (MCA) framework, which efficiently incorporates multiple levels of moment-based information while minimizing additional computation costs through our Cross Moment Convolution (CMC) module. The CMC module via channel-wise convolution layer to capture multiple order moment information as well as cross channel features. The MCA block is designed to be lightweight and easily integrated into a variety of neural network architectures. Experimental results on classical image classification, object detection, and instance segmentation tasks demonstrate that our proposed method achieves state-of-the-art results, outperforming existing channel attention methods.

----

## [287] Towards Robust Image Stitching: An Adaptive Resistance Learning against Compatible Attacks

**Authors**: *Zhiying Jiang, Xingyuan Li, Jinyuan Liu, Xin Fan, Risheng Liu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i3.28036](https://doi.org/10.1609/aaai.v38i3.28036)

**Abstract**:

Image stitching seamlessly integrates images captured from varying perspectives into a single wide field-of-view image. Such integration not only broadens the captured scene but also augments holistic perception in computer vision applications. Given a pair of captured images, subtle perturbations and distortions which go unnoticed by the human visual system tend to attack the correspondence matching, impairing the performance of image stitching algorithms. In light of this challenge, this paper presents the first attempt to improve the robustness of image stitching against adversarial attacks. Specifically, we introduce a stitching-oriented attack (SoA), tailored to amplify the alignment loss within overlapping regions, thereby targeting the feature matching procedure. To establish an attack resistant model, we delve into the robustness of stitching architecture and develop an adaptive adversarial training (AAT) to balance attack resistance with stitching precision. In this way, we relieve the gap between the routine adversarial training and benign models, ensuring resilience without quality compromise. Comprehensive evaluation across real-world and synthetic datasets validate the deterioration of SoA on stitching performance. Furthermore, AAT emerges as a more robust solution against adversarial perturbations, delivering superior stitching results. Code is available at: https://github.com/Jzy2017/TRIS.

----

## [288] Instance-Aware Multi-Camera 3D Object Detection with Structural Priors Mining and Self-Boosting Learning

**Authors**: *Yang Jiao, Zequn Jie, Shaoxiang Chen, Lechao Cheng, Jingjing Chen, Lin Ma, Yu-Gang Jiang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i3.28037](https://doi.org/10.1609/aaai.v38i3.28037)

**Abstract**:

Camera-based bird-eye-view (BEV) perception paradigm has made significant progress in the autonomous driving field. Under such a paradigm, accurate BEV representation construction relies on reliable depth estimation for multi-camera images. However, existing approaches exhaustively predict depths for every pixel without prioritizing objects, which are precisely the entities requiring detection in the 3D space. To this end, we propose IA-BEV, which integrates image-plane instance awareness into the depth estimation process within a BEV-based detector. First, a category-specific structural priors mining approach is proposed for enhancing the efficacy of monocular depth generation. Besides, a self-boosting learning strategy is further proposed to encourage the model to place more emphasis on challenging objects in computation-expensive temporal stereo matching. Together they provide advanced depth estimation results for high-quality BEV features construction, benefiting the ultimate 3D detection. The proposed method achieves state-of-the-art performances on the challenging nuScenes benchmark, and extensive experimental results demonstrate the effectiveness of our designs.

----

## [289] PromptMRG: Diagnosis-Driven Prompts for Medical Report Generation

**Authors**: *Haibo Jin, Haoxuan Che, Yi Lin, Hao Chen*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i3.28038](https://doi.org/10.1609/aaai.v38i3.28038)

**Abstract**:

Automatic medical report generation (MRG) is of great research value as it has the potential to relieve radiologists from the heavy burden of report writing. Despite recent advancements, accurate MRG remains challenging due to the need for precise clinical understanding and disease identification. Moreover, the imbalanced distribution of diseases makes the challenge even more pronounced, as rare diseases are underrepresented in training data, making their diagnosis unreliable. To address these challenges, we propose diagnosis-driven prompts for medical report generation (PromptMRG), a novel framework that aims to improve the diagnostic accuracy of MRG with the guidance of diagnosis-aware prompts. Specifically, PromptMRG is based on encoder-decoder architecture with an extra disease classification branch. When generating reports, the diagnostic results from the classification branch are converted into token prompts to explicitly guide the generation process. To further improve the diagnostic accuracy, we design cross-modal feature enhancement, which retrieves similar reports from the database to assist the diagnosis of a query image by leveraging the knowledge from a pre-trained CLIP. Moreover, the disease imbalanced issue is addressed by applying an adaptive logit-adjusted loss to the classification branch based on the individual learning status of each disease, which overcomes the barrier of text decoder's inability to manipulate disease distributions. Experiments on two MRG benchmarks show the effectiveness of the proposed method, where it obtains state-of-the-art clinical efficacy performance on both datasets.

----

## [290] PCE-Palm: Palm Crease Energy Based Two-Stage Realistic Pseudo-Palmprint Generation

**Authors**: *Jianlong Jin, Lei Shen, Ruixin Zhang, Chenglong Zhao, Ge Jin, Jingyun Zhang, Shouhong Ding, Yang Zhao, Wei Jia*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i3.28039](https://doi.org/10.1609/aaai.v38i3.28039)

**Abstract**:

The lack of large-scale data seriously hinders the development of palmprint recognition. Recent approaches address this issue by generating large-scale realistic pseudo palmprints from Bézier curves. However, the significant difference between Bézier curves and real palmprints limits their effectiveness. In this paper, we divide the Bézier-Real difference into creases and texture differences, thus reducing the generation difficulty. We introduce a new palm crease energy (PCE) domain as a bridge from Bézier curves to real palmprints and propose a two-stage generation model. The first stage generates PCE images (realistic creases) from Bézier curves, and the second stage outputs realistic palmprints (realistic texture) with PCE images as input. In addition, we also design a lightweight plug-and-play line feature enhancement block to facilitate domain transfer and improve recognition performance. Extensive experimental results demonstrate that the proposed method surpasses state-of-the-art methods. Under extremely few data settings like 40 IDs (only 2.5% of the total training set), our model achieves a 29% improvement over RPG-Palm and outperforms ArcFace with 100% training set by more than 6% in terms of TAR@FAR=1e-6.

----

## [291] SwiftPillars: High-Efficiency Pillar Encoder for Lidar-Based 3D Detection

**Authors**: *Xin Jin, Kai Liu, Cong Ma, Ruining Yang, Fei Hui, Wei Wu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i3.28040](https://doi.org/10.1609/aaai.v38i3.28040)

**Abstract**:

Lidar-based 3D Detection is one of the significant components of Autonomous Driving. However, current methods over-focus on improving the performance of 3D Lidar perception, which causes the architecture of networks becoming complicated and hard to deploy. Thus, the methods are difficult to apply in Autonomous Driving for real-time processing. In this paper, we propose a high-efficiency network, SwiftPillars, which includes Swift Pillar Encoder  (SPE) and Multi-scale Aggregation Decoder  (MAD). The SPE is constructed by a concise Dual-attention Module with lightweight operators. The Dual-attention Module utilizes feature pooling, matrix multiplication, etc. to speed up point-wise and channel-wise attention extraction and fusion. The MAD interconnects multiple scale features extracted by SPE with minimal computational cost to leverage performance. In our experiments, our proposal accomplishes 61.3% NDS and 53.2% mAP in nuScenes dataset. In addition, we evaluate inference time on several platforms  (P4,  T4,  A2, MLU370, RTX3080), where SwiftPillars achieves up to 13.3ms  (75FPS) on NVIDIA Tesla T4. Compared with PointPillars, SwiftPillars is on average 26.58% faster in inference speed with equivalent GPUs and a higher mAP of approximately 3.2%  in the nuScenes dataset.

----

## [292] DeS3: Adaptive Attention-Driven Self and Soft Shadow Removal Using ViT Similarity

**Authors**: *Yeying Jin, Wei Ye, Wenhan Yang, Yuan Yuan, Robby T. Tan*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i3.28041](https://doi.org/10.1609/aaai.v38i3.28041)

**Abstract**:

Removing soft and self shadows that lack clear boundaries from a single image is still challenging. Self shadows are shadows that are cast on the object itself. Most existing methods rely on binary shadow masks, without considering the ambiguous boundaries of soft and self shadows. In this paper, we present DeS3, a method that removes hard, soft and self shadows based on adaptive attention and ViT similarity. Our novel ViT similarity loss utilizes features extracted from a pre-trained Vision Transformer. This loss helps guide the reverse sampling towards recovering scene structures. Our adaptive attention is able to differentiate shadow regions from the underlying objects, as well as shadow regions from the object casting the shadow. This capability enables DeS3 to better recover the structures of objects even when they are partially occluded by shadows. Different from existing methods that rely on constraints during the training phase, we incorporate the ViT similarity during the sampling stage. Our method outperforms state-of-the-art methods on the SRD, AISTD, LRSS, USR and UIUC datasets, removing hard, soft, and self shadows robustly. Specifically, our method outperforms the SOTA method by  16% of the RMSE of the whole image on the LRSS dataset.

----

## [293] AMD: Anatomical Motion Diffusion with Interpretable Motion Decomposition and Fusion

**Authors**: *Beibei Jing, Youjia Zhang, Zikai Song, Junqing Yu, Wei Yang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i3.28042](https://doi.org/10.1609/aaai.v38i3.28042)

**Abstract**:

Generating realistic human motion sequences from text descriptions is a challenging task that requires capturing the rich expressiveness of both natural language and human motion. Recent advances in diffusion models have enabled significant progress in human motion synthesis. However, existing methods struggle to handle text inputs that describe complex or long motions. In this paper, we propose the Adaptable Motion Diffusion (AMD) model, which leverages a Large Language Model (LLM) to parse the input text into a sequence of concise and interpretable anatomical scripts that correspond to the target motion. This process exploits the LLM’s ability to provide anatomical guidance for complex motion synthesis. We then devise a two-branch fusion scheme that balances the influence of the input text and the anatomical scripts on the inverse diffusion process, which adaptively ensures the semantic fidelity and diversity of the synthesized motion. Our method can effectively handle texts with complex or long motion descriptions, where existing methods often fail. Experiments on datasets with relatively more complex motions, such as CLCD1 and CLCD2, demonstrate that our AMD significantly outperforms existing state-of-the-art models.

----

## [294] Retrieval-Augmented Primitive Representations for Compositional Zero-Shot Learning

**Authors**: *Chenchen Jing, Yukun Li, Hao Chen, Chunhua Shen*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i3.28043](https://doi.org/10.1609/aaai.v38i3.28043)

**Abstract**:

Compositional zero-shot learning (CZSL) aims to recognize unseen attribute-object compositions by learning from seen compositions. Composing the learned knowledge of seen primitives, i.e., attributes or objects, into novel compositions is critical for CZSL. In this work, we propose to explicitly retrieve knowledge of seen primitives for compositional zero-shot learning. We present a retrieval-augmented method, which augments standard multi-path classification methods with two retrieval modules. Specifically, we construct two databases storing the attribute and object representations of training images, respectively. For an input training/testing image, we use two retrieval modules to retrieve representations of training images with the same attribute and object, respectively. The primitive representations of the input image are augmented by using the retrieved representations, for composition recognition. By referencing semantically similar images, the proposed method is capable of recalling knowledge of seen primitives for compositional generalization. Experiments on three widely-used datasets show the effectiveness of the proposed method.

----

## [295] CrossBind: Collaborative Cross-Modal Identification of Protein Nucleic-Acid-Binding Residues

**Authors**: *Linglin Jing, Sheng Xu, Yifan Wang, Yuzhe Zhou, Tao Shen, Zhigang Ji, Hui Fang, Zhen Li, Siqi Sun*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i3.28044](https://doi.org/10.1609/aaai.v38i3.28044)

**Abstract**:

Accurate identification of protein nucleic acid binding residues poses a significant challenge with important implications for various biological processes and drug design. Many typical computational methods for protein analysis rely on a single model that could ignore either the semantic context of the protein or the global 3D geometric information. Consequently, these approaches may result in incomplete or inaccurate protein analysis. To address the above issue, in this paper, we present CrossBind, a novel collaborative cross modal approach for identifying binding residues by exploiting both protein geometric structure and its sequence prior knowledge extracted from a large scale protein language model. Specifically, our multi modal approach leverages a contrastive learning technique and atom wise attention to capture the positional relationships between atoms and residues, thereby incorporating fine grained local geometric knowledge, for better binding residue prediction. Extensive experimental results demonstrate that our approach outperforms the next best state of the art methods, GraphSite and GraphBind, on DNA and RNA datasets by 10.8/17.3% in terms of the harmonic mean of precision and recall (F1 Score) and 11.9/24.8% in Matthews correlation coefficient (MCC), respectively. We release the code at https://github.com/BEAM-Labs/CrossBind.

----

## [296] X4D-SceneFormer: Enhanced Scene Understanding on 4D Point Cloud Videos through Cross-Modal Knowledge Transfer

**Authors**: *Linglin Jing, Ying Xue, Xu Yan, Chaoda Zheng, Dong Wang, Ruimao Zhang, Zhigang Wang, Hui Fang, Bin Zhao, Zhen Li*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i3.28045](https://doi.org/10.1609/aaai.v38i3.28045)

**Abstract**:

The field of 4D point cloud understanding is rapidly developing with the goal of analyzing dynamic 3D point cloud sequences. However, it remains a challenging task due to the sparsity and lack of texture in point clouds.  Moreover, the irregularity of point cloud poses a difficulty in aligning temporal information within video sequences. To address these issues, we propose a novel cross-modal knowledge transfer framework, called X4D-SceneFormer. This framework enhances 4D-Scene understanding by transferring texture priors from RGB sequences using a Transformer architecture with temporal relationship mining. Specifically, the framework is designed with a dual-branch architecture, consisting of an 4D point cloud transformer and a Gradient-aware Image Transformer (GIT). The GIT combines visual texture and temporal correlation features to offer rich semantics and dynamics for better point cloud representation. During training, we employ multiple knowledge transfer techniques, including temporal consistency losses and masked self-attention, to strengthen the knowledge transfer between modalities. This leads to enhanced performance during inference using single-modal 4D point cloud inputs. Extensive experiments demonstrate the superior performance of our framework on various 4D point cloud video understanding tasks, including action recognition, action segmentation and semantic segmentation. The results achieve 1st places, i.e., 85.3% (+7.9%) accuracy and 47.3% (+5.0%) mIoU for 4D action segmentation and semantic segmentation, on the HOI4D challenge, outperforming previous state-of-the-art by a large margin. We release the code at https://github.com/jinglinglingling/X4D.

----

## [297] VVS: Video-to-Video Retrieval with Irrelevant Frame Suppression

**Authors**: *Won Jo, Geuntaek Lim, Gwangjin Lee, Hyunwoo Kim, Byungsoo Ko, Yukyung Choi*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i3.28046](https://doi.org/10.1609/aaai.v38i3.28046)

**Abstract**:

In content-based video retrieval (CBVR), dealing with large-scale collections, efficiency is as important as accuracy; thus, several video-level feature-based studies have actively been conducted. Nevertheless, owing to the severe difficulty of embedding a lengthy and untrimmed video into a single feature, these studies have been insufficient for accurate retrieval compared to frame-level feature-based studies. In this paper, we show that appropriate suppression of irrelevant frames can provide insight into the current obstacles of the video-level approaches. Furthermore, we propose a Video-to-Video Suppression network (VVS) as a solution. VVS is an end-to-end framework that consists of an easy distractor elimination stage to identify which frames to remove and a suppression weight generation stage to determine the extent to suppress the remaining frames. This structure is intended to effectively describe an untrimmed video with varying content and meaningless information. Its efficacy is proved via extensive experiments, and we show that our approach is not only state-of-the-art in video-level approaches but also has a fast inference time despite possessing retrieval capabilities close to those of frame-level approaches. Code is available at https://github.com/sejong-rcv/VVS

----

## [298] Rethinking Robustness of Model Attributions

**Authors**: *Sandesh Kamath, Sankalp Mittal, Amit Deshpande, Vineeth N. Balasubramanian*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i3.28047](https://doi.org/10.1609/aaai.v38i3.28047)

**Abstract**:

For machine learning models to be reliable and trustworthy, their decisions must be interpretable. As these models find increasing use in safety-critical applications, it is important that not just the model predictions but also their explanations (as feature attributions) be robust to small human-imperceptible input perturbations. Recent works have shown that many attribution methods are fragile and have proposed improvements in either these methods or the model training. We observe two main causes for fragile attributions: first, the existing metrics of robustness (e.g., top-k intersection) overpenalize even reasonable local shifts in attribution, thereby making random perturbations to appear as a strong attack, and second, the attribution can be concentrated in a small region even when there are multiple important parts in an image. To rectify this, we propose simple ways to strengthen existing metrics and attribution methods that incorporate locality of pixels in robustness metrics and diversity of pixel locations in attributions. Towards the role of model training in attributional robustness, we empirically observe that adversarially trained models have more robust attributions on smaller datasets, however, this advantage disappears in larger datasets. Code is made available at https://github.com/ksandeshk/LENS.

----

## [299] Cross-Constrained Progressive Inference for 3D Hand Pose Estimation with Dynamic Observer-Decision-Adjuster Networks

**Authors**: *Zhehan Kan, Xueting Hu, Zihan Liao, Ke Yu, Zhihai He*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i3.28048](https://doi.org/10.1609/aaai.v38i3.28048)

**Abstract**:

Generalization is very important for pose estimation, especially for 3D pose estimation where small changes in the 2D images could trigger structural changes in the 3D space. To achieve  generalization, the system needs to have the capability of detecting estimation errors by double-checking the projection coherence between the 3D and 2D spaces and adapting its network inference process based on this feedback. Current pose estimation is one-time feed-forward and lacks the capability to gather feedback and adapt the inference outcome. To address this problem, we propose to explore the concept of progressive inference where the network learns an observer to continuously detect the prediction error based on constraints matching, as well as an adjuster to refine its inference outcome based on these constraints errors. Within the context of 3D hand pose estimation, we find that this observer-adjuster design is relatively unstable since the observer is operating in the 2D image domain while the adjuster is operating in the 3D domain. To address this issue, we propose to construct two sets of observers-adjusters with complementary constraints from different perspectives. They  operate in a dynamic sequential manner controlled by a decision network to progressively improve the 3D pose estimation. We refer to this method as Cross-Constrained Progressive Inference (CCPI). Our extensive experimental results on FreiHAND and HO-3D benchmark datasets demonstrate that the proposed CCPI method is able to significantly improve the generalization capability and performance of 3D hand pose estimation.

----

## [300] Catch-Up Mix: Catch-Up Class for Struggling Filters in CNN

**Authors**: *Minsoo Kang, Minkoo Kang, Suhyun Kim*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i3.28049](https://doi.org/10.1609/aaai.v38i3.28049)

**Abstract**:

Deep learning has made significant advances in computer vision, particularly in image classification tasks. Despite their high accuracy on training data, deep learning models often face challenges related to complexity and overfitting. One notable concern is that the model often relies heavily on a limited subset of filters for making predictions. This dependency can result in compromised generalization and an increased vulnerability to minor variations. While regularization techniques like weight decay, dropout, and data augmentation are commonly used to address this issue, they may not directly tackle the reliance on specific filters. Our observations reveal that the heavy reliance problem gets severe when slow-learning filters are deprived of learning opportunities due to fast-learning filters. Drawing inspiration from image augmentation research that combats over-reliance on specific image regions by removing and replacing parts of images, Our idea is to mitigate the problem of over-reliance on strong filters by substituting highly activated features. To this end, we present a novel method called Catch-up Mix, which provides learning opportunities to a wide range of filters during training, focusing on filters that may lag behind. By mixing activation maps with relatively lower norms, Catch-up Mix promotes the development of more diverse representations and reduces reliance on a small subset of filters. Experimental results demonstrate the superiority of our method in various vision classification datasets, providing enhanced robustness.

----

## [301] VLCounter: Text-Aware Visual Representation for Zero-Shot Object Counting

**Authors**: *Seunggu Kang, WonJun Moon, Euiyeon Kim, Jae-Pil Heo*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i3.28050](https://doi.org/10.1609/aaai.v38i3.28050)

**Abstract**:

Zero-Shot Object Counting~(ZSOC) aims to count referred instances of arbitrary classes in a query image without human-annotated exemplars. To deal with ZSOC, preceding studies proposed a two-stage pipeline: discovering exemplars and counting. However, there remains a challenge of vulnerability to error propagation of the sequentially designed two-stage process. In this work, we propose an one-stage baseline, Visual-Language Baseline (VLBase), exploring the implicit association of the semantic-patch embeddings of CLIP. Subsequently, we extend the VLBase to Visual-language Counter (VLCounter) by incorporating three modules devised to tailor VLBase for object counting. First, we introduce Semantic-conditioned Prompt Tuning (SPT) within the image encoder to acquire target-highlighted representations. Second, Learnable Affine Transformation (LAT) is employed to translate the semantic-patch similarity map to be appropriate for the counting task. Lastly, we transfer the layer-wisely encoded features to the decoder through Segment-aware Skip Connection (SaSC) to keep the generalization capability for unseen classes. Through extensive experiments on FSC147, CARPK, and PUCPR+, we demonstrate the benefits of our end-to-end framework, VLCounter. Code is available at https://github.com/seunggu0305/VLCounter

----

## [302] StegFormer: Rebuilding the Glory of Autoencoder-Based Steganography

**Authors**: *Xiao Ke, Huanqi Wu, Wenzhong Guo*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i3.28051](https://doi.org/10.1609/aaai.v38i3.28051)

**Abstract**:

Image hiding aims to conceal one or more secret images within a cover image of the same resolution. Due to strict capacity requirements, image hiding is commonly called large-capacity steganography. In this paper, we propose StegFormer, a novel autoencoder-based image-hiding model. StegFormer can conceal one or multiple secret images within a cover image of the same resolution while preserving the high visual quality of the stego image. In addition, to mitigate the limitations of current steganographic models in real-world scenarios, we propose a normalizing training strategy and a restrict loss to improve the reliability of the steganographic models under realistic conditions. Furthermore, we propose an efficient steganographic capacity expansion method to increase the capacity of steganography and enhance the efficiency of secret communication. Through this approach, we can increase the relative payload of StegFormer to 96 bits per pixel without any training strategy modifications. Experiments demonstrate that our StegFormer outperforms existing state-of-the-art (SOTA) models. In the case of single-image steganography, there is an improvement of more than 3 dB and 5 dB in PSNR for secret/recovery image pairs and cover/stego image pairs.

----

## [303] Expediting Contrastive Language-Image Pretraining via Self-Distilled Encoders

**Authors**: *Bumsoo Kim, Jinhyung Kim, Yeonsik Jo, Seung Hwan Kim*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i3.28052](https://doi.org/10.1609/aaai.v38i3.28052)

**Abstract**:

Recent advances in vision language pretraining (VLP) have been largely attributed to the large-scale data collected from the web. However, uncurated dataset contains weakly correlated image-text pairs, causing data inefficiency. To address the issue, knowledge distillation have been explored at the expense of extra image and text momentum encoders to generate teaching signals for misaligned image-text pairs. In this paper, our goal is to resolve the misalignment problem with an efficient distillation framework. To this end, we propose ECLIPSE: Expediting Contrastive Language-Image Pretraining with Self-distilled Encoders. ECLIPSE features a distinctive distillation architecture wherein a shared text encoder is utilized between an online image encoder and a momentum image encoder. This strategic design choice enables the distillation to operate within a unified projected space of text embedding, resulting in better performance. Based on the unified text embedding space, ECLIPSE compensates for the additional computational cost of the momentum image encoder by expediting the online image encoder. Through our extensive experiments, we validate that there is a sweet spot between expedition and distillation where the partial view from the expedited online image encoder interacts complementarily with the momentum teacher. As a result, ECLIPSE outperforms its counterparts while achieving substantial acceleration in inference speed.

----

## [304] Weakly Supervised Semantic Segmentation for Driving Scenes

**Authors**: *Dongseob Kim, Seungho Lee, Junsuk Choe, Hyunjung Shim*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i3.28053](https://doi.org/10.1609/aaai.v38i3.28053)

**Abstract**:

State-of-the-art techniques in weakly-supervised semantic segmentation (WSSS) using image-level labels exhibit severe performance degradation on driving scene datasets such as Cityscapes. To address this challenge, we develop a new WSSS framework tailored to driving scene datasets. Based on extensive analysis of dataset characteristics, we employ Contrastive Language-Image Pre-training (CLIP) as our baseline to obtain pseudo-masks. However, CLIP introduces two key challenges: (1) pseudo-masks from CLIP lack in representing small object classes, and (2) these masks contain notable noise. We propose solutions for each issue as follows. (1) We devise Global-Local View Training that seamlessly incorporates small-scale patches during model training, thereby enhancing the model's capability to handle small-sized yet critical objects in driving scenes (e.g., traffic light). (2) We introduce Consistency-Aware Region Balancing (CARB), a novel technique that discerns reliable and noisy regions through evaluating the consistency between CLIP masks and segmentation predictions. It prioritizes reliable pixels over noisy pixels via adaptive loss weighting. Notably, the proposed method achieves 51.8\% mIoU on the Cityscapes test dataset, showcasing its potential as a strong WSSS baseline on driving scene datasets. Experimental results on CamVid and WildDash2 demonstrate the effectiveness of our method across diverse datasets, even with small-scale datasets or visually challenging conditions. The code is available at https://github.com/k0u-id/CARB.

----

## [305] FPRF: Feed-Forward Photorealistic Style Transfer of Large-Scale 3D Neural Radiance Fields

**Authors**: *GeonU Kim, Kim Youwang, Tae-Hyun Oh*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i3.28054](https://doi.org/10.1609/aaai.v38i3.28054)

**Abstract**:

We present FPRF, a feed-forward photorealistic style transfer method for large-scale 3D neural radiance fields. FPRF stylizes large-scale 3D scenes with arbitrary, multiple style reference images without additional optimization while preserving multi-view appearance consistency. Prior arts required tedious per-style/-scene optimization and were limited to small-scale 3D scenes. FPRF efficiently stylizes large-scale 3D scenes by introducing a style-decomposed 3D neural radiance field, which inherits AdaIN’s feed-forward stylization machinery, supporting arbitrary style reference images. Furthermore, FPRF supports multi-reference stylization with the semantic correspondence matching and local AdaIN, which adds diverse user control for 3D scene styles. FPRF also preserves multi-view consistency by applying semantic matching and style transfer processes directly onto queried features in 3D space. In experiments, we demonstrate that FPRF achieves favorable photorealistic quality 3D scene stylization for large-scale scenes with diverse reference images.

----

## [306] Let There Be Sound: Reconstructing High Quality Speech from Silent Videos

**Authors**: *Ji-Hoon Kim, Jaehun Kim, Joon Son Chung*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i3.28055](https://doi.org/10.1609/aaai.v38i3.28055)

**Abstract**:

The goal of this work is to reconstruct high quality speech from lip motions alone, a task also known as lip-to-speech. A key challenge of lip-to-speech systems is the one-to-many mapping caused by (1) the existence of homophenes and (2) multiple speech variations, resulting in a mispronounced and over-smoothed speech. In this paper, we propose a novel lip-to-speech system that significantly improves the generation quality by alleviating the one-to-many mapping problem from multiple perspectives. Specifically, we incorporate (1) self-supervised speech representations to disambiguate homophenes, and (2) acoustic variance information to model diverse speech styles. Additionally, to better solve the aforementioned problem, we employ a flow based post-net which captures and refines the details of the generated speech. We perform extensive experiments on two datasets, and demonstrate that our method achieves the generation quality close to that of real human utterance, outperforming existing methods in terms of speech naturalness and intelligibility by a large margin. Synthesised samples are available at our demo page: https://mm.kaist.ac.kr/projects/LTBS.

----

## [307] Expand-and-Quantize: Unsupervised Semantic Segmentation Using High-Dimensional Space and Product Quantization

**Authors**: *Jiyoung Kim, Kyuhong Shim, Insu Lee, Byonghyo Shim*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i3.28056](https://doi.org/10.1609/aaai.v38i3.28056)

**Abstract**:

Unsupervised semantic segmentation (USS) aims to discover and recognize meaningful categories without any labels. 
For a successful USS, two key abilities are required: 1) information compression and 2) clustering capability.
Previous methods have relied on feature dimension reduction for information compression, however, this approach may hinder the process of clustering.
In this paper, we propose a novel USS framework called Expand-and-Quantize Unsupervised Semantic Segmentation (EQUSS), which combines the benefits of high-dimensional spaces for better clustering and product quantization for effective information compression.
Our extensive experiments demonstrate that EQUSS achieves state-of-the-art results on three standard benchmarks.
In addition, we analyze the entropy of USS features, which is the first step towards understanding USS from the perspective of information theory.

----

## [308] Sync-NeRF: Generalizing Dynamic NeRFs to Unsynchronized Videos

**Authors**: *Seoha Kim, Jeongmin Bae, Youngsik Yun, Hahyun Lee, Gun Bang, Youngjung Uh*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i3.28057](https://doi.org/10.1609/aaai.v38i3.28057)

**Abstract**:

Recent advancements in 4D scene reconstruction using neural radiance fields (NeRF) have demonstrated the ability to represent dynamic scenes from multi-view videos. However, they fail to reconstruct the dynamic scenes and struggle to fit even the training views in unsynchronized settings. It happens because they employ a single latent embedding for a frame while the multi-view images at the same frame were actually captured at different moments. To address this limitation, we introduce time offsets for individual unsynchronized videos and jointly optimize the offsets with NeRF. By design, our method is applicable for various baselines and improves them with large margins. Furthermore, finding the offsets always works as synchronizing the videos without manual effort. Experiments are conducted on the common Plenoptic Video Dataset and a newly built Unsynchronized Dynamic Blender Dataset to verify the performance of our method. Project page: https://seoha-kim.github.io/sync-nerf

----

## [309] Improving Open Set Recognition via Visual Prompts Distilled from Common-Sense Knowledge

**Authors**: *Seongyeop Kim, Hyung-Il Kim, Yong Man Ro*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i3.28058](https://doi.org/10.1609/aaai.v38i3.28058)

**Abstract**:

Open Set Recognition (OSR) poses significant challenges in distinguishing known from unknown classes. In OSR, the overconfidence problem has become a persistent obstacle, where visual recognition models often misclassify unknown objects as known objects with high confidence. This issue stems from the fact that visual recognition models often lack the integration of common-sense knowledge, a feature that is naturally present in language-based models but lacking in visual recognition systems. In this paper, we propose a novel approach to enhance OSR performance by distilling common-sense knowledge into visual prompts. Utilizing text prompts that embody common-sense knowledge about known classes, the proposed visual prompt is learned by extracting semantic common-sense features and aligning them with image features from visual recognition models. The unique aspect of this work is the training of individual visual prompts for each class to encapsulate this common-sense knowledge. Our methodology is model-agnostic, capable of enhancing OSR across various visual recognition models, and computationally light as it focuses solely on training the visual prompts. This research introduces a method for addressing OSR, aiming at a more systematic integration of visual recognition systems with common-sense knowledge. The obtained results indicate an enhancement in recognition accuracy, suggesting the applicability of this approach in practical settings.

----

## [310] Gaussian Mixture Proposals with Pull-Push Learning Scheme to Capture Diverse Events for Weakly Supervised Temporal Video Grounding

**Authors**: *Sunoh Kim, Jungchan Cho, Joonsang Yu, Youngjoon Yoo, Jin Young Choi*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i3.28059](https://doi.org/10.1609/aaai.v38i3.28059)

**Abstract**:

In the weakly supervised temporal video grounding study, previous methods use predetermined single Gaussian proposals which lack the ability to express diverse events described by the sentence query. To enhance the expression ability of a proposal, we propose a Gaussian mixture proposal (GMP) that can depict arbitrary shapes by learning importance, centroid, and range of every Gaussian in the mixture. In learning GMP, each Gaussian is not trained in a feature space but is implemented over a temporal location. Thus the conventional feature-based learning for Gaussian mixture model is not valid for our case. In our special setting, to learn moderately coupled Gaussian mixture capturing diverse events, we newly propose a pull-push learning scheme using pulling and pushing losses, each of which plays an opposite role to the other. The effects of components in our scheme are verified in-depth with extensive ablation studies and the overall scheme achieves state-of-the-art performance. Our code is available at https://github.com/sunoh-kim/pps.

----

## [311] PARSAC: Accelerating Robust Multi-Model Fitting with Parallel Sample Consensus

**Authors**: *Florian Kluger, Bodo Rosenhahn*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i3.28060](https://doi.org/10.1609/aaai.v38i3.28060)

**Abstract**:

We present a real-time method for robust estimation of multiple instances of geometric models from noisy data.
Geometric models such as vanishing points, planar homographies or fundamental matrices are essential for 3D scene analysis.
Previous approaches discover distinct model instances in an iterative manner, thus limiting their potential for speedup via parallel computation.
In contrast, our method detects all model instances independently and in parallel.
A neural network segments the input data into clusters representing potential model instances by predicting multiple sets of sample and inlier weights.
Using the predicted weights, we determine the model parameters for each potential instance separately in a RANSAC-like fashion.
We train the neural network via task-specific loss functions, i.e. we do not require a ground-truth segmentation of the input data.
As suitable training data for homography and fundamental matrix fitting is scarce, we additionally present two new synthetic datasets.
We demonstrate state-of-the-art performance on these as well as multiple established datasets, with inference times as small as five milliseconds per image.

----

## [312] Distribution Matching for Multi-Task Learning of Classification Tasks: A Large-Scale Study on Faces & Beyond

**Authors**: *Dimitrios Kollias, Viktoriia Sharmanska, Stefanos Zafeiriou*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i3.28061](https://doi.org/10.1609/aaai.v38i3.28061)

**Abstract**:

Multi-Task Learning (MTL) is a framework, where multiple related tasks are learned jointly and benefit from a shared representation space, or parameter transfer. To provide sufficient learning support, modern MTL uses annotated data with full, or sufficiently large overlap across tasks, i.e., each input sample is annotated for all, or most of the tasks. However, collecting such annotations is prohibitive in many real applications, and cannot benefit from datasets available for individual tasks. In this work, we challenge this setup and show that MTL can be successful with classification tasks with little, or non-overlapping annotations, or when there is big discrepancy in the size of labeled data per task. We explore task-relatedness for co-annotation and co-training, and propose a novel approach, where knowledge exchange is enabled between the tasks via distribution matching. To demonstrate the general applicability of our method, we conducted diverse case studies in the domains of affective computing, face recognition, species recognition, and shopping item classification using nine datasets. Our large-scale study of affective tasks for basic expression recognition and facial action unit detection illustrates that our approach is network agnostic and brings large performance improvements compared to the state-of-the-art in both tasks and across all studied databases. In all case studies, we show that co-training via task-relatedness is advantageous and prevents negative transfer (which occurs when MT model's performance is worse than that of at least one single-task model).

----

## [313] Block Image Compressive Sensing with Local and Global Information Interaction

**Authors**: *Xiaoyu Kong, Yongyong Chen, Feng Zheng, Zhenyu He*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i3.28062](https://doi.org/10.1609/aaai.v38i3.28062)

**Abstract**:

Block image compressive sensing methods, which divide a single image into small blocks for efficient sampling and reconstruction, have achieved significant success.
However, these methods process each block locally and thus disregard the global communication among different blocks in the reconstruction step.
Existing methods have attempted to address this issue with local filters or by directly reconstructing the entire image, but they have only achieved insufficient communication among adjacent pixels or bypassed the problem.
To directly confront the communication problem among blocks and effectively resolve it, we propose a novel approach called Block Reconstruction with Blocks'  Communication Network (BRBCN).
BRBCN focuses on both local and global information, while further taking their interactions into account.
Specifically, BRBCN comprises dual CNN and Transformer architectures, in which CNN is used to reconstruct each block for powerful local processing and Transformer is used to calculate the global communication among all the blocks.
Moreover, we propose a global-to-local module (G2L) and a local-to-global module (L2G) to effectively integrate the representations of CNN and Transformer, with which our BRBCN network realizes the bidirectional interaction between local and global information.
Extensive experiments show our BRBCN method outperforms existing state-of-the-art methods by a large margin.
The code is available at https://github.com/kongxiuxiu/BRBCN

----

## [314] QDETRv: Query-Guided DETR for One-Shot Object Localization in Videos

**Authors**: *Yogesh Kumar, Saswat Mallick, Anand Mishra, Sowmya Rasipuram, Anutosh Maitra, Roshni R. Ramnani*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i3.28063](https://doi.org/10.1609/aaai.v38i3.28063)

**Abstract**:

In this work, we study one-shot video object localization problem that aims to localize instances of unseen objects in the target video using a single query image of the object. Toward addressing this challenging problem, we extend a popular and successful object detection method, namely DETR (Detection Transformer), and introduce a novel approach –query-guided detection transformer for videos (QDETRv). A distinctive feature of QDETRv is its capacity to exploit information from the query image and spatio-temporal context of the target video, which significantly aids in precisely pinpointing the desired object in the video. We incorporate cross-attention mechanisms that capture temporal relationships across adjacent frames to handle the dynamic context in videos effectively. Further, to ensure strong initialization for QDETRv, we also introduce a novel unsupervised pretraining technique tailored to videos. This involves training our model on synthetic object trajectories with an analogous objective as the query-guided localization task. During this pretraining phase, we incorporate recurrent object queries and loss functions that encourage accurate patch feature reconstruction. These additions enable better temporal understanding and robust representation learning. Our experiments show that the proposed model significantly outperforms the competitive baselines on two public benchmarks, VidOR and ImageNet-VidVRD, extended for one-shot open-set localization tasks.

----

## [315] LaViP: Language-Grounded Visual Prompting

**Authors**: *Nilakshan Kunananthaseelan, Jing Zhang, Mehrtash Harandi*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i3.28064](https://doi.org/10.1609/aaai.v38i3.28064)

**Abstract**:

We introduce a language-grounded visual prompting method to adapt the visual encoder of vision-language models for downstream tasks. By capitalizing on language integration, we devise a parameter-efficient strategy to adjust the input of the visual encoder, eliminating the need to modify or add to the model's parameters. Due to this design choice, our algorithm can operate even in black-box scenarios, showcasing adaptability in situations where access to the model's parameters is constrained. We will empirically demonstrate that, compared to prior art, grounding visual prompts with language enhances both the accuracy and speed of adaptation. Moreover, our algorithm excels in base-to-novel class generalization, overcoming limitations of visual prompting and exhibiting the capacity to generalize beyond seen classes. We thoroughly assess and evaluate our method across a variety of image recognition datasets, such as EuroSAT, UCF101, DTD, and CLEVR, spanning different learning situations, including few-shot adaptation, base-to-novel class generalization, and transfer learning.

----

## [316] Towards More Faithful Natural Language Explanation Using Multi-Level Contrastive Learning in VQA

**Authors**: *Chengen Lai, Shengli Song, Shiqi Meng, Jingyang Li, Sitong Yan, Guangneng Hu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i3.28065](https://doi.org/10.1609/aaai.v38i3.28065)

**Abstract**:

Natural language explanation in visual question answer (VQA-NLE) aims to explain the decision-making process of models by generating natural language sentences to increase users' trust in the black-box systems. Existing post-hoc methods have achieved significant progress in obtaining a plausible explanation. However, such post-hoc explanations are not always aligned with human logical inference, suffering from the issues on: 1) Deductive unsatisfiability, the generated explanations do not logically lead to the answer; 2) Factual inconsistency, the model falsifies its counterfactual explanation for answers without considering the facts in images; and 3) Semantic perturbation insensitivity, the model can not recognize the semantic changes caused by small perturbations. These problems reduce the faithfulness of explanations generated by models. To address the above issues, we propose a novel self-supervised Multi-level Contrastive Learning based natural language Explanation model (MCLE) for VQA  with semantic-level, image-level, and instance-level factual and counterfactual samples. MCLE extracts discriminative features and aligns the feature spaces from explanations with visual question and answer to generate more consistent explanations. We conduct extensive experiments, ablation analysis, and case study to demonstrate the effectiveness of our method on two VQA-NLE benchmarks.

----

## [317] MatchDet: A Collaborative Framework for Image Matching and Object Detection

**Authors**: *Jinxiang Lai, Wenlong Wu, Bin-Bin Gao, Jun Liu, Jiawei Zhan, Congchong Nie, Yi Zeng, Chengjie Wang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i3.28066](https://doi.org/10.1609/aaai.v38i3.28066)

**Abstract**:

Image matching and object detection are two fundamental and challenging tasks, while many related applications consider them two individual tasks (i.e. task-individual). In this paper, a collaborative framework called MatchDet (i.e. task-collaborative) is proposed for image matching and object detection to obtain mutual improvements. To achieve the collaborative learning of the two tasks, we propose three novel modules, including a Weighted Spatial Attention Module (WSAM) for Detector, and Weighted Attention Module (WAM) and Box Filter for Matcher. Specifically, the WSAM highlights the foreground regions of target image to benefit the subsequent detector, the WAM enhances the connection between the foreground regions of pair images to ensure high-quality matches, and Box Filter mitigates the impact of false matches. We evaluate the approaches on a new benchmark with two datasets called Warp-COCO and miniScanNet. Experimental results show our approaches are effective and achieve competitive improvements.

----

## [318] ViTree: Single-Path Neural Tree for Step-Wise Interpretable Fine-Grained Visual Categorization

**Authors**: *Danning Lao, Qi Liu, Jiazi Bu, Junchi Yan, Wei Shen*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i3.28067](https://doi.org/10.1609/aaai.v38i3.28067)

**Abstract**:

As computer vision continues to advance and finds widespread applications across various domains, the need for interpretability in deep learning models becomes paramount. Existing methods often resort to post-hoc techniques or prototypes to explain the decision-making process, which can be indirect and lack intrinsic illustration. In this research, we introduce ViTree, a novel approach for fine-grained visual categorization that combines the popular vision transformer as a feature extraction backbone with neural decision trees. By traversing the tree paths, ViTree effectively selects patches from transformer-processed features to highlight informative local regions, thereby refining representations in a step-wise manner. Unlike previous tree-based models that rely on soft distributions or ensembles of paths, ViTree selects a single tree path, offering a clearer and simpler decision-making process. This patch and path selectivity enhances model interpretability of ViTree, enabling better insights into the model's inner workings. Remarkably, extensive experimentation validates that this streamlined approach surpasses various strong competitors and achieves state-of-the-art performance while maintaining exceptional interpretability which is proved by multi-perspective methods. Code can be found at https://github.com/SJTU-DeepVisionLab/ViTree.

----

## [319] MaskDiff: Modeling Mask Distribution with Diffusion Probabilistic Model for Few-Shot Instance Segmentation

**Authors**: *Minh-Quan Le, Tam V. Nguyen, Trung-Nghia Le, Thanh-Toan Do, Minh N. Do, Minh-Triet Tran*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i3.28068](https://doi.org/10.1609/aaai.v38i3.28068)

**Abstract**:

Few-shot instance segmentation extends the few-shot learning paradigm to the instance segmentation task, which tries to segment instance objects from a query image with a few annotated examples of novel categories. Conventional approaches have attempted to address the task via prototype learning, known as point estimation. However, this mechanism depends on prototypes (e.g. mean of K-shot) for prediction, leading to performance instability. To overcome the disadvantage of the point estimation mechanism, we propose a novel approach, dubbed MaskDiff, which models the underlying conditional distribution of a binary mask, which is conditioned on an object region and K-shot information. Inspired by augmentation approaches that perturb data with Gaussian noise for populating low data density regions, we model the mask distribution with a diffusion probabilistic model. We also propose to utilize classifier-free guided mask sampling to integrate category information into the binary mask generation process. Without bells and whistles, our proposed method consistently outperforms state-of-the-art methods on both base and novel classes of the COCO dataset while simultaneously being more stable than existing methods. The source code is available at: https://github.com/minhquanlecs/MaskDiff.

----

## [320] FRED: Towards a Full Rotation-Equivariance in Aerial Image Object Detection

**Authors**: *Chanho Lee, Jinsu Son, Hyounguk Shon, Yunho Jeon, Junmo Kim*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i4.28069](https://doi.org/10.1609/aaai.v38i4.28069)

**Abstract**:

Rotation-equivariance is an essential yet challenging property in oriented object detection. While general object detectors naturally leverage robustness to spatial shifts due to the translation-equivariance of the conventional CNNs, achieving rotation-equivariance remains an elusive goal. Current detectors deploy various alignment techniques to derive rotation-invariant features, but still rely on high capacity models and heavy data augmentation with all possible rotations. In this paper, we introduce a Fully Rotation-Equivariant Oriented Object Detector (FRED), whose entire process from the image to the bounding box prediction is strictly equivariant. Specifically, we decouple the invariant task (object classification) and the equivariant task (object localization) to achieve end-to-end equivariance. We represent the bounding box as a set of rotation-equivariant vectors to implement rotation-equivariant localization. Moreover, we utilized these rotation-equivariant vectors as offsets in the deformable convolution, thereby enhancing the existing advantages of spatial adaptation. Leveraging full rotation-equivariance, our FRED demonstrates higher robustness to image-level rotation compared to existing methods. Furthermore, we show that FRED is one step closer to non-axis aligned learning through our experiments. Compared to state-of-the-art methods, our proposed method delivers comparable performance on DOTA-v1.0 and outperforms by 1.5 mAP on DOTA-v1.5, all while significantly reducing the model parameters to 16%.

----

## [321] Domain Generalization with Vital Phase Augmentation

**Authors**: *Ingyun Lee, Wooju Lee, Hyun Myung*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i4.28070](https://doi.org/10.1609/aaai.v38i4.28070)

**Abstract**:

Deep neural networks have shown remarkable performance in image classification. However, their performance significantly deteriorates with corrupted input data. Domain generalization methods have been proposed to train robust models against out-of-distribution data. Data augmentation in the frequency domain is one of such approaches that enable a model to learn phase features to establish domain-invariant representations. This approach changes the amplitudes of the input data while preserving the phases. However, using fixed phases leads to susceptibility to phase fluctuations because amplitudes and phase fluctuations commonly occur in out-of-distribution. In this study, to address this problem, we introduce an approach using finite variation of the phases of input data rather than maintaining fixed phases. Based on the assumption that the degree of domain-invariant features varies for each phase, we propose a method to distinguish phases based on this degree. In addition, we propose a method called vital phase augmentation (VIPAug) that applies the variation to the phases differently according to the degree of domain-invariant features of given phases. The model depends more on the vital phases that contain more domain-invariant features for attaining robustness to amplitude and phase fluctuations. We present experimental evaluations of our proposed approach, which exhibited improved performance for both clean and corrupted data. VIPAug achieved SOTA performance on the benchmark CIFAR-10 and CIFAR-100 datasets, as well as near-SOTA performance on the ImageNet-100 and ImageNet datasets. Our code is available at https://github.com/excitedkid/vipaug.

----

## [322] Modeling Stereo-Confidence out of the End-to-End Stereo-Matching Network via Disparity Plane Sweep

**Authors**: *Jae Young Lee, Woonghyun Ka, Jaehyun Choi, Junmo Kim*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i4.28071](https://doi.org/10.1609/aaai.v38i4.28071)

**Abstract**:

We propose a novel stereo-confidence that can be measured externally to various stereo-matching networks, offering an alternative input modality choice of the cost volume for learning-based approaches, especially in safety-critical systems.
Grounded in the foundational concepts of disparity definition and the disparity plane sweep, the proposed stereo-confidence method is built upon the idea that any shift in a stereo-image pair should be updated in a corresponding amount shift in the disparity map. 
Based on this idea, the proposed stereo-confidence method can be summarized in three folds.
1) Using the disparity plane sweep, multiple disparity maps can be obtained and treated as a 3-D volume (predicted disparity volume), like the cost volume is constructed. 
2) One of these disparity maps serves as an anchor, allowing us to define a desirable (or ideal) disparity profile at every spatial point.
3) By comparing the desirable and predicted disparity profiles, we can quantify the level of matching ambiguity between left and right images for confidence measurement. 
Extensive experimental results using various stereo-matching networks and datasets demonstrate that the proposed stereo-confidence method not only shows competitive performance on its own but also consistent performance improvements when it is used as an input modality for learning-based stereo-confidence methods.

----

## [323] MFOS: Model-Free & One-Shot Object Pose Estimation

**Authors**: *JongMin Lee, Yohann Cabon, Romain Brégier, Sungjoo Yoo, Jérôme Revaud*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i4.28072](https://doi.org/10.1609/aaai.v38i4.28072)

**Abstract**:

Existing learning-based methods for object pose estimation in RGB images are mostly model-specific or category based. They lack the capability to generalize to new object categories at test time, hence severely hindering their practicability and scalability. Notably, recent attempts have been made to solve this issue, but they still require accurate 3D data of the object surface at both train and test time. In this paper, we introduce a novel approach that can estimate in a single forward pass the pose of objects never seen during training, given minimum input. In contrast to existing state-of-the-art approaches, which rely on task-specific modules, our proposed model is entirely based on a transformer architecture, which can benefit from recently proposed 3D-geometry general pretraining. We conduct extensive experiments and report state-of-the-art one-shot performance on the challenging LINEMOD benchmark. Finally, extensive ablations allow us to determine good practices with this relatively new type of architecture in the field.

----

## [324] Noise-Free Optimization in Early Training Steps for Image Super-resolution

**Authors**: *Minkyu Lee, Jae-Pil Heo*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i4.28073](https://doi.org/10.1609/aaai.v38i4.28073)

**Abstract**:

Recent deep-learning-based single image super-resolution (SISR) methods have shown impressive performance whereas typical methods train their networks by minimizing the pixel-wise distance with respect to a given high-resolution (HR) image. However, despite the basic training scheme being the predominant choice, its use in the context of ill-posed inverse problems has not been thoroughly investigated. In this work, we aim to provide a better comprehension of the underlying constituent by decomposing target HR images into two subcomponents: (1) the optimal centroid which is the expectation over multiple potential HR images, and (2) the inherent noise defined as the residual between the HR image and the centroid. Our findings show that the current training scheme cannot capture the ill-posed nature of SISR and becomes vulnerable to the inherent noise term, especially during early training steps. To tackle this issue, we propose a novel optimization method that can effectively remove the inherent noise term in the early steps of vanilla training by estimating the optimal centroid and directly optimizing toward the estimation. Experimental results show that the proposed method can effectively enhance the stability of vanilla training, leading to overall performance gain. Codes are available at github.com/2minkyulee/ECO.

----

## [325] Spectrum Translation for Refinement of Image Generation (STIG) Based on Contrastive Learning and Spectral Filter Profile

**Authors**: *Seokjun Lee, Seung-Won Jung, Hyunseok Seo*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i4.28074](https://doi.org/10.1609/aaai.v38i4.28074)

**Abstract**:

Currently, image generation and synthesis have remarkably progressed with generative models. Despite photo-realistic results, intrinsic discrepancies are still observed in the frequency domain. The spectral discrepancy appeared not only in generative adversarial networks but in diffusion models. In this study, we propose a framework to effectively mitigate the disparity in frequency domain of the generated images to improve generative performance of both GAN and diffusion models. This is realized by spectrum translation for the refinement of image generation (STIG) based on contrastive learning. We adopt theoretical logic of frequency components in various generative networks. The key idea, here, is to refine the spectrum of the generated image via the concept of image-to-image translation and contrastive learning in terms of digital signal processing. We evaluate our framework across eight fake image datasets and various cutting-edge models to demonstrate the effectiveness of STIG. Our framework outperforms other cutting-edges showing significant decreases in FID and log frequency distance of spectrum. We further emphasize that STIG improves image quality by decreasing the spectral anomaly. Additionally, validation results present that the frequency-based deepfake detector confuses more in the case where fake spectrums are manipulated by STIG.

----

## [326] Few-Shot Neural Radiance Fields under Unconstrained Illumination

**Authors**: *SeokYeong Lee, Junyong Choi, Seungryong Kim, Ig-Jae Kim, Junghyun Cho*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i4.28075](https://doi.org/10.1609/aaai.v38i4.28075)

**Abstract**:

In this paper, we introduce a new challenge for synthesizing novel view images in practical environments with limited input multi-view images and varying lighting conditions. Neural radiance fields (NeRF), one of the pioneering works for this task, demand an extensive set of multi-view images taken under constrained illumination, which is often unattainable in real-world settings. While some previous works have managed to synthesize novel views given images with different illumination, their performance still relies on a substantial number of input multi-view images. To address this problem, we suggest ExtremeNeRF, which utilizes multi-view albedo consistency, supported by geometric alignment. Specifically, we extract intrinsic image components that should be illumination-invariant across different views, enabling direct appearance comparison between the input and novel view under unconstrained illumination. We offer thorough experimental results for task evaluation, employing the newly created NeRF Extreme benchmark—the first in-the-wild benchmark for novel view synthesis under multiple viewing directions and varying illuminations.

----

## [327] Object-Aware Domain Generalization for Object Detection

**Authors**: *Wooju Lee, Dasol Hong, Hyungtae Lim, Hyun Myung*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i4.28076](https://doi.org/10.1609/aaai.v38i4.28076)

**Abstract**:

Single-domain generalization (S-DG) aims to generalize a model to unseen environments with a single-source domain. However, most S-DG approaches have been conducted in the field of classification. When these approaches are applied to object detection, the semantic features of some objects can be damaged, which can lead to imprecise object localization and misclassification. To address these problems, we propose an object-aware domain generalization (OA-DG) method for single-domain generalization in object detection. Our method consists of data augmentation and training strategy, which are called OA-Mix and OA-Loss, respectively. OA-Mix generates multi-domain data with multi-level transformation and object-aware mixing strategy. OA-Loss enables models to learn domain-invariant representations for objects and backgrounds from the original and OA-Mixed images. Our proposed method outperforms state-of-the-art works on standard benchmarks. Our code is available at https://github.com/WoojuLee24/OA-DG.

----

## [328] Attention Guided CAM: Visual Explanations of Vision Transformer Guided by Self-Attention

**Authors**: *Saebom Leem, Hyunseok Seo*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i4.28077](https://doi.org/10.1609/aaai.v38i4.28077)

**Abstract**:

Vision Transformer(ViT) is one of the most widely used models in the computer vision field with its great performance on various tasks. In order to fully utilize the ViT-based architecture in various applications, proper visualization methods with a decent localization performance are necessary, but these methods employed in CNN-based models are still not available in ViT due to its unique structure. In this work, we propose an attention-guided visualization method applied to ViT that provides a high-level semantic explanation for its decision. Our method selectively aggregates the gradients directly propagated from the classification output to each self-attention, collecting the contribution of image features extracted from each location of the input image. These gradients are additionally guided by the normalized self-attention scores, which are the pairwise patch correlation scores. They are used to supplement the gradients on the patch-level context information efficiently detected by the self-attention mechanism. This approach of our method provides elaborate high-level semantic explanations with great localization performance only with the class labels. As a result, our method outperforms the previous leading explainability methods of ViT in the weakly-supervised localization task and presents great capability in capturing the full instances of the target class object. Meanwhile, our method provides a visualization that faithfully explains the model, which is demonstrated in the perturbation comparison test.

----

## [329] Contrastive Tuning: A Little Help to Make Masked Autoencoders Forget

**Authors**: *Johannes Lehner, Benedikt Alkin, Andreas Fürst, Elisabeth Rumetshofer, Lukas Miklautz, Sepp Hochreiter*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i4.28078](https://doi.org/10.1609/aaai.v38i4.28078)

**Abstract**:

Masked Image Modeling (MIM) methods, like Masked Autoencoders (MAE), efficiently learn a rich representation of the input. However, for adapting to downstream tasks, they require a sufficient amount of labeled data since their rich features code not only objects but also less relevant image background. In contrast, Instance Discrimination (ID) methods focus on objects. In this work, we study how to combine the efficiency and scalability of MIM with the ability of ID to perform downstream classification in the absence of large amounts of labeled data. To this end, we introduce Masked Autoencoder Contrastive Tuning (MAE-CT), a sequential approach that utilizes the implicit clustering of the Nearest Neighbor Contrastive Learning (NNCLR) objective to induce abstraction in the topmost layers of a pre-trained MAE. MAE-CT tunes the rich features such that they form semantic clusters of objects without using any labels. Notably, MAE-CT does not rely on hand-crafted augmentations and frequently achieves its best performances while using only minimal augmentations (crop & flip). Further, MAE-CT is compute efficient as it requires at most 10% overhead compared to MAE re-training. Applied to large and huge Vision Transformer (ViT) models, MAE-CT excels over previous self-supervised methods trained on ImageNet in linear probing, k-NN and low-shot classification accuracy as well as in unsupervised clustering accuracy. With ViT-H/16 MAE-CT achieves a new state-of-the-art in linear probing of 82.2%.
Project page: github.com/ml-jku/MAE-CT.

----

## [330] Few-Shot Learning from Augmented Label-Uncertain Queries in Bongard-HOI

**Authors**: *Qinqian Lei, Bo Wang, Robby T. Tan*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i4.28079](https://doi.org/10.1609/aaai.v38i4.28079)

**Abstract**:

Detecting human-object interactions (HOI) in a few-shot setting remains a challenge. Existing meta-learning methods struggle to extract representative features for classification due to the limited data, while existing few-shot HOI models rely on HOI text labels for classification. Moreover, some query images may display visual similarity to those outside their class, such as similar backgrounds between different HOI classes. This makes learning more challenging, especially with limited samples. Bongard-HOI epitomizes this HOI few-shot problem, making it the benchmark we focus on in this paper. In our proposed method, we introduce novel label-uncertain query augmentation techniques to enhance the diversity of the query inputs, aiming to distinguish the positive HOI class from the negative ones. As these augmented inputs may or may not have the same class label as the original inputs, their class label is unknown. Those belonging to a different class become hard samples due to their visual similarity to the original ones. Additionally, we introduce a novel pseudo-label generation technique that enables a mean teacher model to learn from the augmented label-uncertain inputs. We propose to augment the negative support set for the student model to enrich the semantic information, fostering diversity that challenges and enhances the student’s learning. Experimental results demonstrate that our method sets a new state-of-the-art (SOTA) performance by achieving 68.74% accuracy on the Bongard-HOI benchmark, a significant improvement over the existing SOTA of 66.59%. In our evaluation on HICO-FS, a more general few-shot recognition dataset, our method achieves 73.27% accuracy, outperforming the previous SOTA of 71.20% in the 5- way 5-shot task.

----

## [331] Removing Interference and Recovering Content Imaginatively for Visible Watermark Removal

**Authors**: *Yicheng Leng, Chaowei Fang, Gen Li, Yixiang Fang, Guanbin Li*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i4.28080](https://doi.org/10.1609/aaai.v38i4.28080)

**Abstract**:

Visible watermarks, while instrumental in protecting image copyrights, frequently distort the underlying content, complicating tasks like scene interpretation and image editing. Visible watermark removal aims to eliminate the interference of watermarks and restore the background content. However, existing methods often implement watermark component removal and background restoration tasks within a singular branch, leading to residual watermarks in the predictions and ignoring cases where watermarks heavily obscure the background. To address these limitations, this study introduces the Removing Interference and Recovering Content Imaginatively (RIRCI) framework. RIRCI embodies a two-stage approach: the initial phase centers on discerning and segregating the watermark component, while the subsequent phase focuses on background content restoration. To achieve meticulous background restoration, our proposed model employs a dual-path network capable of fully exploring the intrinsic background information beneath semi-transparent watermarks and peripheral contextual information from unaffected regions. Moreover,  a Global and Local Context Interaction module is built upon multi-layer perceptrons and bidirectional feature transformation for comprehensive representation modeling in the background restoration phase. The efficacy of our approach is empirically validated across two large-scale datasets, and our findings reveal a marked enhancement over existing watermark removal techniques.

----

## [332] Data Roaming and Quality Assessment for Composed Image Retrieval

**Authors**: *Matan Levy, Rami Ben-Ari, Nir Darshan, Dani Lischinski*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i4.28081](https://doi.org/10.1609/aaai.v38i4.28081)

**Abstract**:

The task of Composed Image Retrieval (CoIR) involves queries that combine image and text modalities, allowing users to express their intent more effectively. However, current CoIR datasets are orders of magnitude smaller compared to other vision and language (V&L) datasets. Additionally, some of these datasets have noticeable issues, such as queries containing redundant modalities. To address these shortcomings, we introduce the Large Scale Composed Image Retrieval (LaSCo) dataset, a new CoIR dataset which is ten times larger than existing ones. Pre-training on our LaSCo, shows a noteworthy improvement in performance, even in zero-shot. Furthermore, we propose a new approach for analyzing CoIR datasets and methods, which detects modality redundancy or necessity, in queries.
We also introduce a new CoIR baseline, the Cross-Attention driven Shift Encoder (CASE). This baseline allows for early fusion of modalities using a cross-attention module and employs an additional auxiliary task during training. Our experiments demonstrate that this new baseline outperforms the current state-of-the-art methods on established benchmarks like FashionIQ and CIRR.

----

## [333] Point Transformer with Federated Learning for Predicting Breast Cancer HER2 Status from Hematoxylin and Eosin-Stained Whole Slide Images

**Authors**: *Bao Li, Zhenyu Liu, Lizhi Shao, Bensheng Qiu, Hong Bu, Jie Tian*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i4.28082](https://doi.org/10.1609/aaai.v38i4.28082)

**Abstract**:

Directly predicting human epidermal growth factor receptor 2 (HER2) status from widely available hematoxylin and eosin (HE)-stained whole slide images (WSIs) can reduce technical costs and expedite treatment selection. Accurately predicting HER2 requires large collections of multi-site WSIs. Federated learning enables collaborative training of these WSIs without gigabyte-size WSIs transportation and data privacy concerns. However, federated learning encounters challenges in addressing label imbalance in multi-site WSIs from the real world. Moreover, existing WSI classification methods cannot simultaneously exploit local context information and long-range dependencies in the site-end feature representation of federated learning. To address these issues, we present a point transformer with federated learning for multi-site HER2 status prediction from HE-stained WSIs. Our approach incorporates two novel designs. We propose a dynamic label distribution strategy and an auxiliary classifier, which helps to establish a well-initialized model and mitigate label distribution variations across sites. Additionally, we propose a farthest cosine sampling based on cosine distance. It can sample the most distinctive features and capture the long-range dependencies. Extensive experiments and analysis show that our method achieves state-of-the-art performance at four sites with a total of 2687 WSIs. Furthermore, we demonstrate that our model can generalize to two unseen sites with 229 WSIs. Code is available at: https://github.com/boyden/PointTransformerFL

----

## [334] Unsupervised Cross-Domain Image Retrieval via Prototypical Optimal Transport

**Authors**: *Bin Li, Ye Shi, Qian Yu, Jingya Wang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i4.28083](https://doi.org/10.1609/aaai.v38i4.28083)

**Abstract**:

Unsupervised cross-domain image retrieval (UCIR) aims to retrieve images sharing the same category across diverse domains without relying on labeled data. Prior approaches have typically decomposed the UCIR problem into two distinct tasks: intra-domain representation learning and cross-domain feature alignment. However, these segregated strategies overlook the potential synergies between these tasks. This paper introduces ProtoOT, a novel Optimal Transport formulation explicitly tailored for UCIR, which integrates intra-domain feature representation learning and cross-domain alignment into a unified framework. ProtoOT leverages the strengths of the K-means clustering method to effectively manage distribution imbalances inherent in UCIR. By utilizing K-means for generating initial prototypes and approximating class marginal distributions, we modify the constraints in Optimal Transport accordingly, significantly enhancing its performance in UCIR scenarios. Furthermore, we incorporate contrastive learning into the ProtoOT framework to further improve representation learning. This encourages local semantic consistency among features with similar semantics, while also explicitly enforcing separation between features and unmatched prototypes, thereby enhancing global discriminativeness. ProtoOT surpasses existing state-of-the-art methods by a notable margin across benchmark datasets. Notably, on DomainNet, ProtoOT achieves an average P@200 enhancement of 24.44%, and on Office-Home, it demonstrates a P@15 improvement of 12.12%. Code is available at https://github.com/HCVLAB/ProtoOT.

----

## [335] Semantic-Guided Generative Image Augmentation Method with Diffusion Models for Image Classification

**Authors**: *Bohan Li, Xiao Xu, Xinghao Wang, Yutai Hou, Yunlong Feng, Feng Wang, Xuanliang Zhang, Qingfu Zhu, Wanxiang Che*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i4.28084](https://doi.org/10.1609/aaai.v38i4.28084)

**Abstract**:

Existing image augmentation methods consist of two categories: perturbation-based methods and generative methods. Perturbation-based methods apply pre-defined perturbations to augment an original image, but only locally vary the image, thus lacking image diversity. In contrast, generative methods bring more image diversity in the augmented images but may not preserve semantic consistency, thus may incorrectly change the essential semantics of the original image.  To balance image diversity and semantic consistency in augmented images, we propose SGID, a Semantic-guided Generative Image augmentation method with Diffusion models for image classification. Specifically, SGID employs diffusion models to generate augmented images with good image diversity. More importantly, SGID takes image labels and captions as guidance to maintain semantic consistency between the augmented and original images. Experimental results show that SGID outperforms the best augmentation baseline by 1.72% on ResNet-50 (from scratch), 0.33% on ViT (ImageNet-21k), and 0.14% on CLIP-ViT (LAION-2B). Moreover, SGID can be combined with other image augmentation baselines and further improves the overall performance. We demonstrate the semantic consistency and image diversity of SGID through quantitative human and automated evaluations, as well as qualitative case studies.

----

## [336] One at a Time: Progressive Multi-Step Volumetric Probability Learning for Reliable 3D Scene Perception

**Authors**: *Bohan Li, Yasheng Sun, Jingxin Dong, Zheng Zhu, Jinming Liu, Xin Jin, Wenjun Zeng*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i4.28085](https://doi.org/10.1609/aaai.v38i4.28085)

**Abstract**:

Numerous studies have investigated the pivotal role of reliable 3D volume representation in scene perception tasks, such as multi-view stereo (MVS) and semantic scene completion (SSC). They typically construct 3D probability volumes directly with geometric correspondence, attempting to fully address the scene perception tasks in a single forward pass. However, such a single-step solution makes it hard to learn accurate and convincing volumetric probability, especially in challenging regions like unexpected occlusions and complicated light reflections. Therefore, this paper proposes to decompose the complicated 3D volume representation learning into a sequence of generative steps to facilitate fine and reliable scene perception. Considering the recent advances achieved by strong generative diffusion models, we introduce a multi-step learning framework, dubbed as VPD, dedicated to progressively refining the Volumetric Probability in a Diffusion process. Specifically, we first build a coarse probability volume from input images with the off-the-shelf scene perception baselines, which is then conditioned as the basic geometry prior before being fed into a 3D diffusion UNet, to progressively achieve accurate probability distribution modeling. To handle the corner cases in challenging areas, a Confidence-Aware Contextual Collaboration (CACC) module is developed to correct the uncertain regions for reliable volumetric learning based on multi-scale contextual contents. Moreover, an Online Filtering (OF) strategy is designed to maintain representation consistency for stable diffusion sampling. Extensive experiments are conducted on scene perception tasks including multi-view stereo (MVS) and semantic scene completion (SSC), to validate the efficacy of our method in learning reliable volumetric representations. Notably, for the SSC task, our work stands out as the first to surpass LiDAR-based methods on the SemanticKITTI dataset.

----

## [337] AE-NeRF: Audio Enhanced Neural Radiance Field for Few Shot Talking Head Synthesis

**Authors**: *Dongze Li, Kang Zhao, Wei Wang, Bo Peng, Yingya Zhang, Jing Dong, Tieniu Tan*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i4.28086](https://doi.org/10.1609/aaai.v38i4.28086)

**Abstract**:

Audio-driven talking head synthesis is a promising topic with wide applications in digital human, film making and virtual reality. Recent NeRF-based approaches have shown superiority in quality and fidelity compared to previous studies. However, when it comes to few-shot talking head generation, a practical scenario where only few seconds of talking video is available for one identity, two limitations emerge: 1) they either have no base model, which serves as a facial prior for fast convergence, or ignore the importance of audio when building the prior; 2) most of them overlook the degree of correlation between different face regions and audio, e.g., mouth is audio related, while ear is audio independent. In this paper, we present Audio Enhanced Neural Radiance Field (AE-NeRF) to tackle the above issues, which can generate realistic portraits of a new speaker with few-shot dataset. Specifically, we introduce an Audio Aware Aggregation module into the feature fusion stage of the reference scheme, where the weight is determined by the similarity of audio between reference and target image. Then, an Audio-Aligned Face Generation strategy is proposed to model the audio related and audio independent regions respectively, with a dual-NeRF framework. Extensive experiments have shown AE-NeRF surpasses the state-of-the-art on image fidelity, audio-lip synchronization, and generalization ability, even in limited training set or training iterations.

----

## [338] Monocular 3D Hand Mesh Recovery via Dual Noise Estimation

**Authors**: *Hanhui Li, Xiaojian Lin, Xuan Huang, Zejun Yang, Zhisheng Wang, Xiaodan Liang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i4.28087](https://doi.org/10.1609/aaai.v38i4.28087)

**Abstract**:

Current parametric models have made notable progress in 3D hand pose and shape estimation. However, due to the fixed hand topology and complex hand poses, current models are hard to generate meshes that are aligned with the image well. To tackle this issue, we introduce a dual noise estimation method in this paper. Given a single-view image as input, we first adopt a baseline parametric regressor to obtain the coarse hand meshes. We assume the mesh vertices and their image-plane projections are noisy, and can be associated in a unified probabilistic model. We then learn the distributions of noise to refine mesh vertices and their projections. The refined vertices are further utilized to refine camera parameters in a closed-form manner. Consequently, our method obtains well-aligned and high-quality 3D hand meshes. Extensive experiments on the large-scale Interhand2.6M dataset demonstrate that the proposed method not only improves the performance of its baseline by more than 10% but also achieves state-of-the-art performance. Project page: https://github.com/hanhuili/DNE4Hand.

----

## [339] Point2Real: Bridging the Gap between Point Cloud and Realistic Image for Open-World 3D Recognition

**Authors**: *Hanxuan Li, Bin Fu, Ruiping Wang, Xilin Chen*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i4.28088](https://doi.org/10.1609/aaai.v38i4.28088)

**Abstract**:

Recognition in open-world scenarios is an important and challenging field, where Vision-Language Pre-training paradigms have greatly impacted the 2D domain. This inspires a growing interest in introducing 2D pre-trained models, such as CLIP, into the 3D domain to enhance the ability of point cloud understanding. Considering the difference between discrete 3D point clouds and real-world 2D images, reducing the domain gap is crucial. Some recent works project point clouds onto a 2D plane to enable 3D zero-shot capabilities without training. However, this simplistic approach leads to an unclear or even distorted geometric structure, limiting the potential of 2D pre-trained models in 3D. To address the domain gap, we propose Point2Real, a training-free framework based on the realistic rendering technique to automate the transformation of the 3D point cloud domain into the Vision-Language domain. Specifically, Point2Real leverages a shape recovery module that devises an iterative ball-pivoting algorithm to convert point clouds into meshes, narrowing the gap in shape at first. To simulate photo-realistic images, a set of refined textures as candidates is applied for rendering, where the CLIP confidence is utilized to select the suitable one. Moreover, to tackle the viewpoint challenge, a heuristic multi-view adapter is implemented for feature aggregation, which exploits the depth surface as an effective indicator of view-specific discriminability for recognition. We conduct experiments on ModelNet10, ModelNet40, and ScanObjectNN datasets, and the results demonstrate that Point2Real outperforms other approaches in zero-shot and few-shot tasks by a large margin.

----

## [340] Gradual Residuals Alignment: A Dual-Stream Framework for GAN Inversion and Image Attribute Editing

**Authors**: *Hao Li, Mengqi Huang, Lei Zhang, Bo Hu, Yi Liu, Zhendong Mao*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i4.28089](https://doi.org/10.1609/aaai.v38i4.28089)

**Abstract**:

GAN-based image attribute editing firstly leverages GAN Inversion to project real images into the latent space of GAN and then manipulates corresponding latent codes. Recent inversion methods mainly utilize additional high-bit features to improve image details preservation, as low-bit codes cannot faithfully reconstruct source images, leading to the loss of details. However, during editing, existing works fail to accurately complement the lost details and suffer from poor editability. The main reason is they inject all the lost details indiscriminately at one time, which inherently induces the position and quantity of details to overfit source images, resulting in inconsistent content and artifacts in edited images. This work argues that details should be gradually injected into both the reconstruction and editing process in a multi-stage coarse-to-fine manner for better detail preservation and high editability. Therefore, a novel dual-stream framework is proposed to accurately complement details at each stage. The Reconstruction Stream is employed to embed coarse-to-fine lost details into residual features and then adaptively add them to the GAN generator. In the Editing Stream, residual features are accurately aligned by our Selective Attention mechanism and then injected into the editing process in a multi-stage manner. Extensive experiments have shown the superiority of our framework in both reconstruction accuracy and editing quality compared with existing methods.

----

## [341] Towards Automated Chinese Ancient Character Restoration: A Diffusion-Based Method with a New Dataset

**Authors**: *Haolong Li, Chenghao Du, Ziheng Jiang, Yifan Zhang, Jiawei Ma, Chen Ye*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i4.28090](https://doi.org/10.1609/aaai.v38i4.28090)

**Abstract**:

Automated Chinese ancient character restoration (ACACR) remains a challenging task due to its historical significance and aesthetic complexity. Existing methods are constrained by non-professional masks and even overfitting when training on small-scale datasets, which hinder their interdisciplinary application to traditional fields. In this paper, we are proud to introduce the Chinese Ancient Rubbing and Manuscript Character Dataset (ARMCD), which consists of 15,553 real-world ancient single-character images with 42 rubbings and manuscripts, covering the works of over 200 calligraphy artists spanning from 200 to 1,800 AD. We are also dedicated to providing professional synthetic masks by extracting localized erosion from real eroded images. Moreover, we propose DiffACR (Diffusion model for automated Chinese Ancient Character Restoration), a diffusion-based method for the ACACR task. Specifically, we regard the synthesis of eroded images as a special form of cold diffusion on uneroded ones and extract the prior mask directly from the eroded images. Our experiments demonstrate that our method comprehensively outperforms most existing methods on the proposed ARMCD. Dataset and code are available at https://github.com/lhl322001/DiffACR.

----

## [342] Learning Deformable Hypothesis Sampling for Accurate PatchMatch Multi-View Stereo

**Authors**: *Hongjie Li, Yao Guo, Xianwei Zheng, Hanjiang Xiong*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i4.28091](https://doi.org/10.1609/aaai.v38i4.28091)

**Abstract**:

This paper introduces a learnable Deformable Hypothesis Sampler (DeformSampler) to address the challenging issue of noisy depth estimation in faithful PatchMatch multi-view stereo (MVS). We observe that the heuristic depth hypothesis sampling modes employed by PatchMatch MVS solvers are insensitive to (i) the piece-wise smooth distribution of depths across the object surface and (ii) the implicit multi-modal distribution of depth prediction probabilities along the ray direction on the surface points. Accordingly, we develop DeformSampler to learn distribution-sensitive sample spaces to (i) propagate depths consistent with the scene's geometry across the object surface and (ii) fit a Laplace Mixture model that approaches the point-wise probabilities distribution of the actual depths along the ray direction. We integrate DeformSampler into a learnable PatchMatch MVS system to enhance depth estimation in challenging areas, such as piece-wise discontinuous surface boundaries and weakly-textured regions. Experimental results on DTU and Tanks & Temples datasets demonstrate its superior performance and generalization capabilities compared to state-of-the-art competitors. Code is available at https://github.com/Geo-Tell/DS-PMNet.

----

## [343] Catalyst for Clustering-Based Unsupervised Object Re-identification: Feature Calibration

**Authors**: *Huafeng Li, Qingsong Hu, Zhanxuan Hu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i4.28092](https://doi.org/10.1609/aaai.v38i4.28092)

**Abstract**:

Clustering-based methods are emerging as a ubiquitous technology in unsupervised object Re-Identification (ReID), which alternate between pseudo-label generation and representation learning. Recent advances in this field mainly fall into two groups: pseudo-label correction and robust representation learning. Differently, in this work, we improve unsupervised object ReID from feature calibration, a completely different but complementary insight from the current approaches. Specifically, we propose to insert a conceptually simple yet empirically powerful Feature Calibration Module (FCM) before pseudo-label generation. In practice, FCM calibrates the features using a nonparametric graph attention network, enforcing similar instances to move together in the feature space while allowing dissimilar instances to separate. As a result, we can generate more reliable pseudo-labels using the calibrated features and further improve subsequent representation learning. FCM is simple, effective, parameter-free, training-free, plug-and-play, and can be considered as a catalyst, increasing the ’chemical reaction’ between pseudo-label generation and representation learning. Moreover, it maintains the efficiency of testing time with negligible impact on training time. In this paper, we insert FCM into a simple baseline. Experiments across different scenarios and benchmarks show that FCM consistently improves the baseline (e.g., 8.2% mAP gain on MSMT17), and achieves the new state-of-the-art results. Code is available at: https://github.com/lhf12278/FCM-ReID.

----

## [344] EAN: An Efficient Attention Module Guided by Normalization for Deep Neural Networks

**Authors**: *Jiafeng Li, Zelin Li, Ying Wen*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i4.28093](https://doi.org/10.1609/aaai.v38i4.28093)

**Abstract**:

Deep neural networks (DNNs) have achieved remarkable success in various fields, and two powerful techniques, feature normalization and attention mechanisms, have been widely used to enhance model performance. However, they are usually considered as two separate approaches or combined in a simplistic manner.
In this paper, we investigate the intrinsic relationship between feature normalization and attention mechanisms and propose an Efficient Attention module guided by Normalization, dubbed EAN. Instead of using costly fully-connected layers for attention learning, EAN leverages the strengths of feature normalization and incorporates an Attention Generation (AG) unit to re-calibrate features. The proposed AG unit exploits the normalization component as a measure of the importance of distinct features and generates an attention mask using GroupNorm, L2 Norm, and Adaptation operations. By employing a grouping, AG unit and aggregation strategy, EAN is established, offering a unified module that harnesses the advantages of both normalization and attention, while maintaining minimal computational overhead. Furthermore, EAN serves as a plug-and-play module that can be seamlessly integrated with classic backbone architectures. Extensive quantitative evaluations on various visual tasks demonstrate that EAN achieves highly competitive performance compared to the current state-of-the-art attention methods while sustaining lower model complexity.

----

## [345] Label-Efficient Few-Shot Semantic Segmentation with Unsupervised Meta-Training

**Authors**: *Jianwu Li, Kaiyue Shi, Guo-Sen Xie, Xiaofeng Liu, Jian Zhang, Tianfei Zhou*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i4.28094](https://doi.org/10.1609/aaai.v38i4.28094)

**Abstract**:

The goal of this paper is to alleviate the training cost for few-shot semantic segmentation (FSS) models. Despite that FSS in nature improves model generalization to new concepts using only a handful of test exemplars, it relies on strong supervision from a considerable amount of labeled training data for base classes. However, collecting pixel-level annotations is notoriously expensive and time-consuming, and small-scale training datasets convey low information density that limits test-time generalization. To resolve the issue, we take a pioneering step towards label-efficient training of FSS models from fully unlabeled training data, or additionally a few labeled samples to enhance the performance. This motivates an approach based on a novel unsupervised meta-training paradigm. In particular, the approach first distills pre-trained unsupervised pixel embedding into compact semantic clusters from which a massive number of pseudo meta-tasks is constructed. To mitigate the noise in the pseudo meta-tasks, we further advocate a robust Transformer-based FSS model with a novel prototype-based cross-attention design. Extensive experiments have been conducted on two standard benchmarks, i.e., PASCAL-5i and COCO-20i, and the results show that our method produces impressive performance without any annotations, and is comparable to fully supervised competitors even using only 20% of the annotations. Our code is available at: https://github.com/SSSKYue/UMTFSS.

----

## [346] FedDiv: Collaborative Noise Filtering for Federated Learning with Noisy Labels

**Authors**: *Jichang Li, Guanbin Li, Hui Cheng, Zicheng Liao, Yizhou Yu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i4.28095](https://doi.org/10.1609/aaai.v38i4.28095)

**Abstract**:

Federated Learning with Noisy Labels (F-LNL) aims at seeking an optimal server model via collaborative distributed learning by aggregating multiple client models trained with local noisy or clean samples. On the basis of a federated learning framework, recent advances primarily adopt label noise filtering to separate clean samples from noisy ones on each client, thereby mitigating the negative impact of label noise. However, these prior methods do not learn noise filters by exploiting knowledge across all clients, leading to sub-optimal and inferior noise filtering performance and thus damaging training stability. In this paper, we present FedDiv to tackle the challenges of F-LNL. Specifically, we propose a global noise filter called Federated Noise Filter for effectively identifying samples with noisy labels on every client, thereby raising stability during local training sessions. Without sacrificing data privacy, this is achieved by modeling the global distribution of label noise across all clients. Then, in an effort to make the global model achieve higher performance, we introduce a Predictive Consistency based Sampler to identify more credible local data for local model training, thus preventing noise memorization and further boosting the training stability. Extensive experiments on CIFAR-10, CIFAR-100, and Clothing1M demonstrate that FedDiv achieves superior performance over state-of-the-art F-LNL methods under different label noise settings for both IID and non-IID data partitions. Source code is publicly available at https://github.com/lijichang/FLNL-FedDiv.

----

## [347] Fully Data-Driven Pseudo Label Estimation for Pointly-Supervised Panoptic Segmentation

**Authors**: *Jing Li, Junsong Fan, Yuran Yang, Shuqi Mei, Jun Xiao, Zhaoxiang Zhang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i4.28096](https://doi.org/10.1609/aaai.v38i4.28096)

**Abstract**:

The core of pointly-supervised panoptic segmentation is estimating accurate dense pseudo labels from sparse point labels to train the panoptic head. Previous works generate pseudo labels mainly based on hand-crafted rules, such as connecting multiple points into polygon masks, or assigning the label information of labeled pixels to unlabeled pixels based on the artificially defined traversing distance. The accuracy of pseudo labels is limited by the quality of the hand-crafted rules (polygon masks are rough at object contour regions, and the traversing distance error will result in wrong pseudo labels). To overcome the limitation of hand-crafted rules, we estimate pseudo labels with a fully data-driven pseudo label branch, which is optimized by point labels end-to-end and predicts more accurate pseudo labels than previous methods. We also train an auxiliary semantic branch with point labels, it assists the training of the pseudo label branch by transferring semantic segmentation knowledge through shared parameters. Experiments on Pascal VOC and MS COCO demonstrate that our approach is effective and shows state-of-the-art performance compared with related works. Codes are available at  https://github.com/BraveGroup/FDD.

----

## [348] FAVOR: Full-Body AR-Driven Virtual Object Rearrangement Guided by Instruction Text

**Authors**: *Kailin Li, Lixin Yang, Zenan Lin, Jian Xu, Xinyu Zhan, Yifei Zhao, Pengxiang Zhu, Wenxiong Kang, Kejian Wu, Cewu Lu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i4.28097](https://doi.org/10.1609/aaai.v38i4.28097)

**Abstract**:

Rearrangement operations form the crux of interactions between humans and their environment. The ability to generate natural, fluid sequences of this operation is of essential value in AR/VR and CG. Bridging a gap in the field, our study introduces FAVOR: a novel dataset for Full-body AR-driven Virtual Object Rearrangement that uniquely employs motion capture systems and AR eyeglasses. Comprising 3k diverse motion rearrangement sequences and 7.17 million interaction data frames, this dataset breaks new ground in research data. We also present a pipeline FAVORITE for producing digital human rearrangement motion sequences guided by instructions. Experimental results, both qualitative and quantitative, suggest that this dataset and pipeline deliver high-quality motion sequences. Our dataset, code, and appendix are available at https://kailinli.github.io/FAVOR.

----

## [349] Panoptic Scene Graph Generation with Semantics-Prototype Learning

**Authors**: *Li Li, Wei Ji, Yiming Wu, Mengze Li, You Qin, Lina Wei, Roger Zimmermann*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i4.28098](https://doi.org/10.1609/aaai.v38i4.28098)

**Abstract**:

Panoptic Scene Graph Generation (PSG) parses objects and predicts their relationships (predicate) to connect human language and visual scenes.
However, different language preferences of annotators and semantic overlaps between predicates lead to biased predicate annotations in the dataset, i.e. different predicates for the same object pairs.
Biased predicate annotations make PSG models struggle in constructing a clear decision plane among predicates, which greatly hinders the real application of PSG models.
To address the intrinsic bias above, we propose a novel framework named ADTrans to adaptively transfer biased predicate annotations to informative and unified ones. To promise consistency and accuracy during the transfer process, we propose to observe the invariance degree of representations in each predicate class, and learn unbiased prototypes of predicates with different intensities. Meanwhile, we continuously measure the distribution changes between each presentation and its prototype, and constantly screen potentially biased data. Finally, with the unbiased predicate-prototype representation embedding space, biased annotations are easily identified.
Experiments show that ADTrans significantly improves the performance of benchmark models, achieving a new state-of-the-art performance, and shows great generalization and effectiveness on multiple datasets. Our code is released at https://github.com/lili0415/PSG-biased-annotation.

----

## [350] SpectralNeRF: Physically Based Spectral Rendering with Neural Radiance Field

**Authors**: *Ru Li, Jia Liu, Guanghui Liu, Shengping Zhang, Bing Zeng, Shuaicheng Liu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i4.28099](https://doi.org/10.1609/aaai.v38i4.28099)

**Abstract**:

In this paper, we propose SpectralNeRF, an end-to-end Neural Radiance Field (NeRF)-based architecture for high-quality physically based rendering from a novel spectral perspective. We modify the classical spectral rendering into two main steps, 1) the generation of a series of spectrum maps spanning different wavelengths, 2) the combination of these spectrum maps for the RGB output. Our SpectralNeRF follows these two steps through the proposed multi-layer perceptron (MLP)-based architecture (SpectralMLP) and Spectrum Attention UNet (SAUNet). Given the ray origin and the ray direction, the SpectralMLP constructs the spectral radiance field to obtain spectrum maps of novel views, which are then sent to the SAUNet to produce RGB images of white-light illumination. Applying NeRF to build up the spectral rendering is a more physically-based way from the perspective of ray-tracing. Further, the spectral radiance fields decompose difficult scenes and improve the performance of NeRF-based methods. Comprehensive experimental results demonstrate the proposed SpectralNeRF is superior to recent NeRF-based methods when synthesizing new views on synthetic and real datasets. The codes and datasets are available at https://github.com/liru0126/SpectralNeRF.

----

## [351] GridFormer: Point-Grid Transformer for Surface Reconstruction

**Authors**: *Shengtao Li, Ge Gao, Yudong Liu, Yu-Shen Liu, Ming Gu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i4.28100](https://doi.org/10.1609/aaai.v38i4.28100)

**Abstract**:

Implicit neural networks have emerged as a crucial technology in 3D surface reconstruction. To reconstruct continuous surfaces from discrete point clouds, encoding the input points into regular grid features (plane or volume) has been commonly employed in existing approaches. However, these methods typically use the grid as an index for uniformly scattering point features. Compared with the irregular point features, the regular grid features may sacrifice some reconstruction details but improve efficiency. To take full advantage of these two types of features, we introduce a novel and high-efficiency attention mechanism between the grid and point features named Point-Grid Transformer (GridFormer). This mechanism treats the grid as a transfer point connecting the space and point cloud. Our method maximizes the spatial expressiveness of grid features and maintains computational efficiency. Furthermore, optimizing predictions over the entire space could potentially result in blurred boundaries. To address this issue, we further propose a boundary optimization strategy incorporating margin binary cross-entropy loss and boundary sampling. This approach enables us to achieve a more precise representation of the object structure. Our experiments validate that our method is effective and outperforms the state-of-the-art approaches under widely used benchmarks by producing more precise geometry reconstructions. The code is available at https://github.com/list17/GridFormer.

----

## [352] Adaptive Uncertainty-Based Learning for Text-Based Person Retrieval

**Authors**: *Shenshen Li, Chen He, Xing Xu, Fumin Shen, Yang Yang, Heng Tao Shen*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i4.28101](https://doi.org/10.1609/aaai.v38i4.28101)

**Abstract**:

Text-based person retrieval aims at retrieving a specific pedestrian image from a gallery based on textual descriptions. The primary challenge is how to overcome the inherent heterogeneous modality gap in the situation of significant intra-class variation and minimal inter-class variation. Existing approaches commonly employ vision-language pre-training or attention mechanisms to learn appropriate cross-modal alignments from noise inputs. Despite commendable progress, current methods inevitably suffer from two defects: 1) Matching ambiguity, which mainly derives from unreliable matching pairs; 2) One-sided cross-modal alignments, stemming from the absence of exploring one-to-many correspondence, i.e., coarse-grained semantic alignment. These critical issues significantly deteriorate retrieval performance. To this end, we propose a novel framework termed Adaptive Uncertainty-based Learning (AUL) for text-based person retrieval from the uncertainty perspective. Specifically, our AUL framework consists of three key components: 1) Uncertainty-aware Matching Filtration that leverages Subjective Logic to effectively mitigate the disturbance of unreliable matching pairs and select high-confidence cross-modal matches for training; 2) Uncertainty-based Alignment Refinement, which not only simulates coarse-grained alignments by constructing uncertainty representations but also performs progressive learning to incorporate coarse- and fine-grained alignments properly; 3) Cross-modal Masked Modeling that aims at exploring more comprehensive relations between vision and language. Extensive experiments demonstrate that our AUL method consistently achieves state-of-the-art performance on three benchmark datasets in supervised, weakly supervised, and domain generalization settings. Our code is available at https://github.com/CFM-MSG/Code-AUL.

----

## [353] Learning Continuous Implicit Field with Local Distance Indicator for Arbitrary-Scale Point Cloud Upsampling

**Authors**: *Shujuan Li, Junsheng Zhou, Baorui Ma, Yu-Shen Liu, Zhizhong Han*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i4.28102](https://doi.org/10.1609/aaai.v38i4.28102)

**Abstract**:

Point cloud upsampling aims to generate dense and uniformly distributed point sets from a sparse point cloud, which plays a critical role in 3D computer vision. Previous methods typically split a sparse point cloud into several local patches, upsample patch points, and merge all upsampled patches. However, these methods often produce holes, outliers or non-uniformity due to the splitting and merging process which does not maintain consistency among local patches.To address these issues, we propose a novel approach that learns an unsigned distance field guided by local priors for point cloud upsampling.  Specifically, we train a local distance indicator (LDI) that predicts the unsigned distance from a query point to a local implicit surface. Utilizing the learned LDI, we learn an unsigned distance field to represent the sparse point cloud with patch consistency. At inference time, we randomly sample queries around the sparse point cloud, and project these query points onto the zero-level set of the learned implicit field to generate a dense point cloud. We justify that the implicit field is naturally continuous, which inherently enables the application of arbitrary-scale upsampling without necessarily retraining for various scales. We conduct comprehensive experiments on both synthetic data and real scans, and report state-of-the-art results under widely used benchmarks. Project page: https://lisj575.github.io/APU-LDI

----

## [354] Long-Tailed Learning as Multi-Objective Optimization

**Authors**: *Weiqi Li, Fan Lyu, Fanhua Shang, Liang Wan, Wei Feng*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i4.28103](https://doi.org/10.1609/aaai.v38i4.28103)

**Abstract**:

Real-world data is extremely imbalanced and presents a long-tailed distribution, resulting in models biased towards classes with sufficient samples and performing poorly on rare classes. Recent methods propose to rebalance classes but they undertake the seesaw dilemma (what is increasing performance on tail classes may decrease that of head classes, and vice versa). In this paper, we argue that the seesaw dilemma is derived from the gradient imbalance of different classes, in which gradients of inappropriate classes are set to important for updating, thus prone to overcompensation or undercompensation on tail classes. To achieve ideal compensation, we formulate long-tailed recognition as a multi-objective optimization problem, which fairly respects the contributions of head and tail classes simultaneously. For efficiency, we propose a Gradient-Balancing Grouping (GBG) strategy to gather the classes with similar gradient directions, thus approximately making every update under a Pareto descent direction. Our GBG method drives classes with similar gradient directions to form a more representative gradient and provides ideal compensation to the tail classes. Moreover, we conduct extensive experiments on commonly used benchmarks in long-tailed learning and demonstrate the superiority of our method over existing SOTA methods. Our code is released at https://github.com/WickyLee1998/GBG_v1.

----

## [355] Temporal-Distributed Backdoor Attack against Video Based Action Recognition

**Authors**: *Xi Li, Songhe Wang, Ruiquan Huang, Mahanth Gowda, George Kesidis*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i4.28104](https://doi.org/10.1609/aaai.v38i4.28104)

**Abstract**:

Deep neural networks (DNNs) have achieved tremendous success in various applications including video action recognition, yet remain vulnerable to backdoor attacks (Trojans). The backdoor-compromised model will mis-classify to the target class chosen by the attacker when a test instance (from a non-target class) is embedded with a specific trigger, while maintaining high accuracy on attack-free instances. Although there are extensive studies on backdoor attacks against image data, the susceptibility of video-based systems under backdoor attacks remains largely unexplored. Current studies are direct extensions of approaches proposed for image data, e.g., the triggers are independently embedded within the frames, which tend to be detectable by existing defenses. In this paper, we introduce a simple yet effective backdoor attack against video data. Our proposed attack, adding perturbations in a transformed domain, plants an imperceptible, temporally distributed trigger across the video frames, and is shown to be resilient to existing defensive strategies. The effectiveness of the proposed attack is demonstrated by extensive experiments with various well-known models on two video recognition benchmarks, UCF101 and HMDB51, and a sign language recognition benchmark, Greek Sign Language (GSL) dataset. We delve into the impact of several influential factors on our proposed attack and identify an intriguing effect termed "collateral damage" through extensive studies.

----

## [356] DI-V2X: Learning Domain-Invariant Representation for Vehicle-Infrastructure Collaborative 3D Object Detection

**Authors**: *Xiang Li, Junbo Yin, Wei Li, Chengzhong Xu, Ruigang Yang, Jianbing Shen*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i4.28105](https://doi.org/10.1609/aaai.v38i4.28105)

**Abstract**:

Vehicle-to-Everything (V2X) collaborative perception has recently gained significant attention due to its capability to enhance scene understanding by integrating information from various agents, e.g., vehicles, and infrastructure. However, current works often treat the information from each agent equally, ignoring the inherent domain gap caused by the utilization of different LiDAR sensors of each agent, thus leading to suboptimal performance. In this paper, we propose DI-V2X, that aims to learn Domain-Invariant representations through a new distillation framework to mitigate the domain discrepancy in the context of V2X 3D object detection. DI-V2X comprises three essential components: a domain-mixing instance augmentation (DMA) module, a progressive domain-invariant distillation (PDD) module, and a domain-adaptive fusion (DAF) module. Specifically, DMA builds a domain-mixing 3D instance bank for the teacher and student models during training, resulting in aligned data representation. Next, PDD encourages the student models from different domains to gradually learn a domain-invariant feature representation towards the teacher, where the overlapping regions between agents are employed as guidance to facilitate the distillation process. Furthermore, DAF closes the domain gap between the students by incorporating calibration-aware domain-adaptive attention. Extensive experiments on the challenging DAIR-V2X and V2XSet benchmark datasets demonstrate DI-V2X achieves remarkable performance, outperforming all the previous V2X models. Code is available at https://github.com/Serenos/DI-V2X.

----

## [357] Multi-Modality Affinity Inference for Weakly Supervised 3D Semantic Segmentation

**Authors**: *Xiawei Li, Qingyuan Xu, Jing Zhang, Tianyi Zhang, Qian Yu, Lu Sheng, Dong Xu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i4.28106](https://doi.org/10.1609/aaai.v38i4.28106)

**Abstract**:

3D point cloud semantic segmentation has a wide range of applications. Recently, weakly supervised point cloud segmentation methods have been proposed, aiming to alleviate the expensive and laborious manual annotation process by leveraging scene-level labels. However, these methods have not effectively exploited the rich geometric information (such as shape and scale) and appearance information (such as color and texture) present in RGB-D scans. Furthermore, current approaches fail to fully leverage the point affinity that can be inferred from the feature extraction network, which is crucial for learning from weak scene-level labels. Additionally, previous work overlooks the detrimental effects of the long-tailed distribution of point cloud data in weakly supervised 3D semantic segmentation. To this end, this paper proposes a simple yet effective scene-level weakly supervised point cloud segmentation method with a newly introduced multi-modality point affinity inference module. The point affinity proposed in this paper is characterized by features from multiple modalities (e.g., point cloud and RGB), and is further refined by normalizing the classifier weights to alleviate the detrimental effects of long-tailed distribution without the need of the prior of category distribution. Extensive experiments on the ScanNet and S3DIS benchmarks verify the effectiveness of our proposed method, which outperforms the state-of-the-art by ~4% to ~ 6% mIoU. Codes are released at https://github.com/Sunny599/AAAI24-3DWSSG-MMA.

----

## [358] IINet: Implicit Intra-inter Information Fusion for Real-Time Stereo Matching

**Authors**: *Ximeng Li, Chen Zhang, Wanjuan Su, Wenbing Tao*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i4.28107](https://doi.org/10.1609/aaai.v38i4.28107)

**Abstract**:

Recently, there has been a growing interest in 3D CNN-based stereo matching methods due to their remarkable accuracy. However, the high complexity of 3D convolution makes it challenging to strike a balance between accuracy and speed. Notably, explicit 3D volumes contain considerable redundancy. In this study, we delve into more compact 2D implicit network to eliminate redundancy and boost real-time performance. However, simply replacing explicit 3D networks with 2D implicit networks causes issues that can lead to performance degradation, including the loss of structural information, the quality decline of inter-image information, as well as the inaccurate regression caused by low-level features. To address these issues, we first integrate intra-image information to fuse with inter-image information, facilitating propagation guided by structural cues. Subsequently, we introduce the Fast Multi-scale Score Volume (FMSV) and Confidence Based Filtering (CBF) to efficiently acquire accurate multi-scale, noise-free inter-image information. Furthermore, combined with the Residual Context-aware Upsampler (RCU), our Intra-Inter Fusing network is meticulously designed to enhance information transmission on both feature-level and disparity-level, thereby enabling accurate and robust regression. Experimental results affirm the superiority of our network in terms of both speed and accuracy compared to all other fast methods.

----

## [359] Causal Representation Learning via Counterfactual Intervention

**Authors**: *Xiutian Li, Siqi Sun, Rui Feng*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i4.28108](https://doi.org/10.1609/aaai.v38i4.28108)

**Abstract**:

Existing causal representation learning methods are based on the causal graph they build. However, due to the omission of bias within the causal graph, they essentially encourage models to learn biased causal effects in latent space. In this paper, we propose a novel causally disentangling framework that aims to learn unbiased causal effects. We first introduce inductive and dataset biases into traditional causal graph for the physical concepts of interest. Then, we eliminate the negative effects from these two biases by counterfactual intervention with reweighted loss function for learning unbiased causal effects. Finally, we employ the causal effects into the VAE to endow the latent representations with causality. In particular, we highlight that removing biases in this paper is regarded as a part of learning process for unbiased causal effects, which is crucial for causal disentanglement performance improvement. Through extensive experiments on real-world and synthetic datasets, we show that our method outperforms different baselines and obtains the state-of-the-art results for achieving causal representation learning.

----

## [360] Bi-ViT: Pushing the Limit of Vision Transformer Quantization

**Authors**: *Yanjing Li, Sheng Xu, Mingbao Lin, Xianbin Cao, Chuanjian Liu, Xiao Sun, Baochang Zhang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i4.28109](https://doi.org/10.1609/aaai.v38i4.28109)

**Abstract**:

Vision transformers (ViTs) quantization offers a promising prospect to facilitate deploying large pre-trained networks on resource-limited devices.  Fully-binarized ViTs (Bi-ViT) that pushes the quantization of ViTs to its limit remain largely unexplored and a very challenging task yet, due to their  unacceptable performance. Through extensive empirical analyses, we  identify the severe drop in ViT binarization is caused by attention distortion in self-attention, which technically stems from the gradient vanishing and ranking disorder. To address these issues, we first introduce a learnable scaling factor to reactivate the vanished gradients and illustrate its effectiveness through theoretical and experimental analyses. We then propose a ranking-aware distillation method  to rectify the disordered ranking in a teacher-student framework. Bi-ViT achieves significant improvements over popular DeiT and Swin backbones in terms of Top-1 accuracy and FLOPs. For example, with DeiT-Tiny and Swin-Tiny, our method significantly outperforms baselines by 22.1% and 21.4% respectively, while  61.5x and 56.1x theoretical acceleration  in terms of FLOPs compared with real-valued counterparts on ImageNet. Our codes and models are attached on https://github.com/YanjingLi0202/Bi-ViT/ .

----

## [361] Harnessing Edge Information for Improved Robustness in Vision Transformers

**Authors**: *Yanxi Li, Chengbin Du, Chang Xu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i4.28110](https://doi.org/10.1609/aaai.v38i4.28110)

**Abstract**:

Deep Neural Networks (DNNs) have demonstrated remarkable accuracy in vision classification tasks. However, they exhibit vulnerability to additional noises known as adversarial attacks. Previous studies hypothesize that this vulnerability might stem from the fact that high-accuracy DNNs heavily rely on irrelevant and non-robust features, such as textures and the background. In this work, we reveal that edge information extracted from images can provide relevant and robust features related to shapes and the foreground. These features assist pretrained DNNs in achieving improved adversarial robustness without compromising their accuracy on clean images. A lightweight and plug-and-play EdgeNet is proposed, which can be seamlessly integrated into existing pretrained DNNs, including Vision Transformers, a recent family of state-of-the-art models for vision classification. Our EdgeNet can process edges derived from either clean nature images or noisy adversarial images, yielding robust features which can be injected into the intermediate layers of the frozen backbone DNNs. The cost of obtaining such edges using conventional edge detection algorithms (e.g., Canny edge detector) is marginal, and the cost of training the EdgeNet is equivalent to that of fine-tuning the backbone network with techniques such as Adapter.

----

## [362] Multi-Region Text-Driven Manipulation of Diffusion Imagery

**Authors**: *Yiming Li, Peng Zhou, Jun Sun, Yi Xu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i4.28111](https://doi.org/10.1609/aaai.v38i4.28111)

**Abstract**:

Text-guided image manipulation has attracted significant attention recently. Prevailing techniques concentrate on image attribute editing for individual objects, however, encountering challenges when it comes to multi-object editing. The main reason is the lack of consistency constraints on the spatial layout. This work presents a multi-region guided image manipulation framework, enabling manipulation through region-level textual prompts. With MultiDiffusion as a baseline, we are dedicated to the automatic generation of a rational multi-object spatial distribution, where disparate regions are fused as a unified entity. To mitigate interference from regional fusion, we employ an off-the-shelf model (CLIP) to impose region-aware spatial guidance on multi-object manipulation. Moreover, when applied to the StableDiffusion, the presence of quality-related yet object-agnostic lengthy words hampers the manipulation. To ensure focus on meaningful object-specific words for efficient guidance and generation, we introduce a keyword selection method. Furthermore, we demonstrate a downstream application of our method for multi-region inversion, which is tailored for manipulating multiple objects in real images. Our approach, compatible with variants of Stable Diffusion models, is readily applicable for manipulating diverse objects in extensive images with high-quality generation, showing superb image control capabilities. Code is available at https://github.com/liyiming09/multi-region-guided-diffusion.

----

## [363] Direct May Not Be the Best: An Incremental Evolution View of Pose Generation

**Authors**: *Yuelong Li, Tengfei Xiao, Lei Geng, Jianming Wang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i4.28112](https://doi.org/10.1609/aaai.v38i4.28112)

**Abstract**:

Pose diversity is an inherent representative characteristic of 2D images. Due to the 3D to 2D projection mechanism, there is evident content discrepancy among distinct pose images. This is the main obstacle bothering pose transformation related researches. To deal with this challenge, we propose a fine-grained incremental evolution centered pose generation framework, rather than traditional direct one-to-one in a rush. Since proposed approach actually bypasses the theoretical difficulty of directly modeling dramatic non-linear variation, the incurred content distortion and blurring could be effectively constrained, at the same time the various individual pose details, especially clothes texture, could be precisely maintained. In order to systematically guide the evolution course, both global and incremental evolution constraints are elaborately designed and merged into the overall framework. And a novel triple-path knowledge fusion structure is worked out to take full advantage of all available valuable knowledge to conduct high-quality pose synthesis. In addition, our framework could generate a series of valuable by-products, namely the various intermediate poses. Extensive experiments have been conducted to verify the effectiveness of the proposed approach. Code is available at https://github.com/Xiaofei-CN/Incremental-Evolution-Pose-Generation.

----

## [364] FocalDreamer: Text-Driven 3D Editing via Focal-Fusion Assembly

**Authors**: *Yuhan Li, Yishun Dou, Yue Shi, Yu Lei, Xuanhong Chen, Yi Zhang, Peng Zhou, Bingbing Ni*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i4.28113](https://doi.org/10.1609/aaai.v38i4.28113)

**Abstract**:

While text-3D editing has made significant strides in leveraging score distillation sampling, emerging approaches still fall short in delivering separable, precise and consistent outcomes that are vital to content creation. In response, we introduce FocalDreamer, a framework that merges base shape with editable parts according to text prompts for fine-grained editing within desired regions. Specifically, equipped with geometry union and dual-path rendering, FocalDreamer assembles independent 3D parts into a complete object, tailored for convenient instance reuse and part-wise control. We propose geometric focal loss and style consistency regularization, which encourage focal fusion and congruent overall appearance. Furthermore, FocalDreamer generates high-fidelity geometry and PBR textures which are compatible with widely-used graphics engines. Extensive experiments have highlighted the superior editing capabilities of FocalDreamer in both quantitative and qualitative evaluations.

----

## [365] SAVSR: Arbitrary-Scale Video Super-Resolution via a Learned Scale-Adaptive Network

**Authors**: *Zekun Li, Hongying Liu, Fanhua Shang, Yuanyuan Liu, Liang Wan, Wei Feng*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i4.28114](https://doi.org/10.1609/aaai.v38i4.28114)

**Abstract**:

Deep learning-based video super-resolution (VSR) networks have gained significant performance improvements in recent years. However, existing VSR networks can only support a fixed integer scale super-resolution task, and when we want to perform VSR at multiple scales, we need to train several models. This implementation certainly increases the consumption of computational and storage resources, which limits the application scenarios of VSR techniques. In this paper, we propose a novel Scale-adaptive Arbitrary-scale Video Super-Resolution network (SAVSR), which is the first work focusing on spatial VSR at arbitrary scales including both non-integer and asymmetric scales. We also present an omni-dimensional scale-attention convolution, which dynamically adapts according to the scale of the input to extract inter-frame features with stronger representational power. Moreover, the proposed spatio-temporal adaptive arbitrary-scale upsampling performs VSR tasks using both temporal features and scale information. And we design an iterative bi-directional architecture for implicit feature alignment. Experiments at various scales on the benchmark datasets show that the proposed SAVSR outperforms state-of-the-art (SOTA) methods at non-integer and asymmetric scales. The source code is available at https://github.com/Weepingchestnut/SAVSR.

----

## [366] Sampling-Resilient Multi-Object Tracking

**Authors**: *Zepeng Li, Dongxiang Zhang, Sai Wu, Mingli Song, Gang Chen*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i4.28115](https://doi.org/10.1609/aaai.v38i4.28115)

**Abstract**:

Multi-Object Tracking (MOT) is a cornerstone operator for video surveillance applications. To enable real-time processing of large-scale live video streams, we study an interesting scenario called down-sampled MOT, which performs object tracking only on a small subset of video frames. The problem is challenging for state-of-the-art MOT methods, which exhibit significant performance degradation under high frame reduction ratios. In this paper, we devise a sampling-resilient tracker with a novel sparse-observation Kalman filter (SOKF). It integrates an LSTM network to capture non-linear and dynamic motion patterns caused by sparse observations. Since the LSTM-based state transition is not compatible with the original noise estimation mechanism, we propose new estimation strategies based on Bayesian neural networks and derive the optimal Kalman gain for SOKF. To associate the detected bounding boxes robustly, we also propose a comprehensive similarity metric that systematically integrates multiple spatial matching signals. Experiments on three benchmark datasets show that our proposed tracker achieves the best trade-off between efficiency and accuracy. With the same tracking accuracy, we reduce the total processing time of ByteTrack by 2× in MOT17 and 3× in DanceTrack.

----

## [367] Object-Aware Adaptive-Positivity Learning for Audio-Visual Question Answering

**Authors**: *Zhangbin Li, Dan Guo, Jinxing Zhou, Jing Zhang, Meng Wang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i4.28116](https://doi.org/10.1609/aaai.v38i4.28116)

**Abstract**:

This paper focuses on the Audio-Visual Question Answering (AVQA) task that aims to answer questions derived from untrimmed audible videos. To generate accurate answers, an AVQA model is expected to find the most informative audio-visual clues relevant to the given questions. In this paper, we propose to explicitly consider fine-grained visual objects in video frames (object-level clues) and explore the multi-modal relations (\textit{i.e.}, the object, audio, and question) in terms of feature interaction and model optimization. For the former, we present an end-to-end object-oriented network that adopts a question-conditioned clue discovery module to concentrate audio/visual modalities on respective keywords of the question and designs a modality-conditioned clue collection module to highlight closely associated audio segments or visual objects. For model optimization, we propose an object-aware adaptive-positivity learning strategy that selects the highly semantic-matched multi-modal pair as \textit{positivity}. Specifically, we design two object-aware contrastive loss functions to identify the highly relevant question-object pairs and audio-object pairs, respectively. These selected pairs are constrained to have larger similarity values than the mismatched pairs. The positivity-selecting process is adaptive as the positivity pairs selected in each video frame may be different. These two object-aware objectives help the model understand \textit{which objects are exactly relevant to the question} and \textit{which are making sounds}. Extensive experiments on the MUSIC-AVQA dataset demonstrate the proposed method is effective in finding favorable audio-visual clues and also achieves new state-of-the-art question-answering performance. The code is available at https://github.com/zhangbin-ai/APL.

----

## [368] Hypercorrelation Evolution for Video Class-Incremental Learning

**Authors**: *Sen Liang, Kai Zhu, Wei Zhai, Zhiheng Liu, Yang Cao*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i4.28117](https://doi.org/10.1609/aaai.v38i4.28117)

**Abstract**:

Video class-incremental learning aims to recognize new actions while restricting the catastrophic forgetting of old ones, whose representative samples can only be saved in limited memory. Semantically variable subactions are susceptible to class confusion due to data imbalance. While existing methods address the problem by estimating and distilling the spatio-temporal knowledge, we further explores that the refinement of hierarchical correlations is crucial for the alignment of spatio-temporal features. To enhance the adaptability on evolved actions, we proposes a hierarchical aggregation strategy, in which  hierarchical matching matrices are combined and jointly optimized to selectively store and retrieve relevant features from previous tasks. Meanwhile, a correlation refinement mechanism is presented to reinforce the bias on informative exemplars according to online hypercorrelation distribution. Experimental results demonstrate the effectiveness of the proposed method on three standard video class-incremental learning benchmarks, outperforming state-of-the-art methods. Code is available at: https://github.com/Lsen991031/HCE

----

## [369] CoSTA: End-to-End Comprehensive Space-Time Entanglement for Spatio-Temporal Video Grounding

**Authors**: *Yaoyuan Liang, Xiao Liang, Yansong Tang, Zhao Yang, Ziran Li, Jingang Wang, Wenbo Ding, Shao-Lun Huang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i4.28118](https://doi.org/10.1609/aaai.v38i4.28118)

**Abstract**:

This paper studies the spatio-temporal video grounding task, which aims to localize a spatio-temporal tube in an untrimmed video based on the given text description of an event. Existing one-stage approaches suffer from insufficient space-time interaction in two aspects: i) less precise prediction of event temporal boundaries, and ii) inconsistency in object prediction for the same event across adjacent frames. To address these issues, we propose a framework of Comprehensive Space-Time entAnglement (CoSTA) to densely entangle space-time multi-modal features for spatio-temporal localization. Specifically, we propose a space-time collaborative encoder to extract comprehensive video features and leverage Transformer to perform spatio-temporal multi-modal understanding. Our entangled decoder couples temporal boundary prediction and spatial localization via an entangled query, boasting an enhanced ability to capture object-event relationships. We conduct extensive experiments on the challenging benchmarks of HC-STVG and VidSTG, where CoSTA outperforms existing state-of-the-art methods, demonstrating its effectiveness for this task.

----

## [370] Any-Stereo: Arbitrary Scale Disparity Estimation for Iterative Stereo Matching

**Authors**: *Zhaohuai Liang, Changhe Li*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i4.28119](https://doi.org/10.1609/aaai.v38i4.28119)

**Abstract**:

Due to unaffordable computational costs, the regularized disparity in iterative stereo matching is typically maintained at a lower resolution than the input. To regress the full resolution disparity, most stereo methods resort to convolutions to decode a fixed-scale output. However, they are inadequate for recovering vital high-frequency information lost during downsampling, limiting their performance on full-resolution prediction. In this paper, we introduce AnyStereo, an accurate and efficient disparity upsampling module with implicit neural representation for the iterative stereo pipeline. By modeling the disparity as a continuous representation over 2D spatial coordinates, subtle details can emerge from the latent space at arbitrary resolution. To further complement the missing information and details in the latent code, we propose two strategies: intra-scale similarity unfolding and cross-scale feature alignment. The former unfolds the neighbor relationships, while the latter introduces the context in high-resolution feature maps. The proposed AnyStereo can seamlessly replace the upsampling module in most iterative stereo models, improving their ability to capture fine details and generate arbitrary-scale disparities even with fewer parameters. With our method, the iterative stereo pipeline establishes a new state-of-the-art performance. The code is available at https://github.com/Zhaohuai-L/Any-Stereo.

----

## [371] Impartial Adversarial Distillation: Addressing Biased Data-Free Knowledge Distillation via Adaptive Constrained Optimization

**Authors**: *Dongping Liao, Xitong Gao, Chengzhong Xu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i4.28120](https://doi.org/10.1609/aaai.v38i4.28120)

**Abstract**:

Data-Free Knowledge Distillation (DFKD) enables knowledge transfer from a pretrained teacher to a light-weighted student without original training data. Existing works are limited by a strong assumption that samples used to pretrain the teacher model are balanced, which is, however, unrealistic for many real-world tasks. In this work, we investigated a pragmatic yet under-explored problem: how to perform DFKD from a teacher model pretrained from imbalanced data. We observe a seemingly counter-intuitive phenomenon, i.e., adversarial DFKD algorithms favour minority classes, while causing a disastrous impact on majority classes. We theoretically prove that a biased teacher could cause severe disparity on different groups of synthetic data in adversarial distillation, which further exacerbates the mode collapse of a generator and consequently degenerates the overall accuracy of a distilled student model. To tackle this problem, we propose a class-adaptive regularization method, aiming to encourage impartial representation learning of a generator among different classes under a constrained learning formulation. We devise a primal-dual algorithm to solve the target optimization problem. Through extensive experiments, we show that our method mitigates the biased learning of majority classes in DFKD and improves the overall performance compared with baselines. Code will be available at https://github.com/ldpbuaa/ipad.

----

## [372] VLM2Scene: Self-Supervised Image-Text-LiDAR Learning with Foundation Models for Autonomous Driving Scene Understanding

**Authors**: *Guibiao Liao, Jiankun Li, Xiaoqing Ye*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i4.28121](https://doi.org/10.1609/aaai.v38i4.28121)

**Abstract**:

Vision and language foundation models (VLMs) have showcased impressive capabilities in 2D scene understanding. However, their latent potential in elevating the understanding of 3D autonomous driving scenes remains untapped. In this paper, we propose VLM2Scene, which exploits the potential of VLMs to enhance 3D self-supervised representation learning through our proposed image-text-LiDAR contrastive learning strategy. Specifically, in the realm of autonomous driving scenes, the inherent sparsity of LiDAR point clouds poses a notable challenge for point-level contrastive learning methods. This method often grapples with limitations tied to a restricted receptive field and the presence of noisy points. To tackle this challenge, our approach emphasizes region-level learning, leveraging regional masks without semantics derived from the vision foundation model. This approach capitalizes on valuable contextual information to enhance the learning of point cloud representations. First, we introduce Region Caption Prompts to generate fine-grained language descriptions for the corresponding regions, utilizing the language foundation model. These region prompts then facilitate the establishment of positive and negative text-point pairs within the contrastive loss framework. Second, we propose a Region Semantic Concordance Regularization, which involves a semantic-filtered region learning and a region semantic assignment strategy. The former aims to filter the false negative samples based on the semantic distance, and the latter mitigates potential inaccuracies in pixel semantics, thereby enhancing overall semantic consistency. Extensive experiments on representative autonomous driving datasets demonstrate that our self-supervised method significantly outperforms other counterparts. Codes are available at https://github.com/gbliao/VLM2Scene.

----

## [373] Text-to-Image Generation for Abstract Concepts

**Authors**: *Jiayi Liao, Xu Chen, Qiang Fu, Lun Du, Xiangnan He, Xiang Wang, Shi Han, Dongmei Zhang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i4.28122](https://doi.org/10.1609/aaai.v38i4.28122)

**Abstract**:

Recent years have witnessed the substantial progress of large-scale models across various domains, such as natural language processing and computer vision, facilitating the expression of concrete concepts. Unlike concrete concepts that are usually directly associated with physical objects, expressing abstract concepts through natural language requires considerable effort since they are characterized by intricate semantics and connotations. An alternative approach is to leverage images to convey rich visual information as a supplement. Nevertheless, existing Text-to-Image (T2I) models are primarily trained on concrete physical objects and often struggle to visualize abstract concepts. Inspired by the three-layer artwork theory that identifies critical factors, intent, object and form during artistic creation, we propose a framework of Text-to-Image generation for Abstract Concepts (TIAC). The abstract concept is clarified into a clear intent with a detailed definition to avoid ambiguity. LLMs then transform it into semantic-related physical objects, and the concept-dependent form is retrieved from an LLM-extracted form pattern set. Information from these three aspects will be integrated to generate prompts for T2I models via LLM. Evaluation results from human assessments and our newly designed metric concept score demonstrate the effectiveness of our framework in creating images that can sufficiently express abstract concepts.

----

## [374] VSFormer: Visual-Spatial Fusion Transformer for Correspondence Pruning

**Authors**: *Tangfei Liao, Xiaoqin Zhang, Li Zhao, Tao Wang, Guobao Xiao*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i4.28123](https://doi.org/10.1609/aaai.v38i4.28123)

**Abstract**:

Correspondence pruning aims to find correct matches (inliers) from an initial set of putative correspondences, which is a fundamental task for many applications. The process of finding is challenging, given the varying inlier ratios between scenes/image pairs due to significant visual differences. However, the performance of the existing methods is usually limited by the problem of lacking visual cues (e.g., texture, illumination, structure) of scenes. In this paper, we propose a Visual-Spatial Fusion Transformer (VSFormer) to identify inliers and recover camera poses accurately. Firstly, we obtain highly abstract visual cues of a scene with the cross attention between local features of two-view images. Then, we model these visual cues and correspondences by a joint visual-spatial fusion module, simultaneously embedding visual cues into correspondences for pruning. Additionally, to mine the consistency of correspondences, we also design a novel module that combines the KNN-based graph and the transformer, effectively capturing both local and global contexts. Extensive experiments have demonstrated that the proposed VSFormer outperforms state-of-the-art methods on outdoor and indoor benchmarks. Our code is provided at the following repository: https://github.com/sugar-fly/VSFormer.

----

## [375] NightRain: Nighttime Video Deraining via Adaptive-Rain-Removal and Adaptive-Correction

**Authors**: *Beibei Lin, Yeying Jin, Wending Yan, Wei Ye, Yuan Yuan, Shunli Zhang, Robby T. Tan*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i4.28124](https://doi.org/10.1609/aaai.v38i4.28124)

**Abstract**:

Existing deep-learning-based methods for nighttime video deraining rely on synthetic data due to the absence of real-world paired data. However, the intricacies of the real world, particularly with the presence of light effects and low-light regions affected by noise, create significant domain gaps, hampering synthetic-trained models in removing rain streaks properly and leading to over-saturation and color shifts. Motivated by this, we introduce NightRain, a novel nighttime video deraining method with adaptive-rain-removal and adaptive-correction. Our adaptive-rain-removal uses unlabeled rain videos to enable our model to derain real-world rain videos, particularly in regions affected by complex light effects. The idea is to allow our model to obtain rain-free regions based on the confidence scores. Once rain-free regions and the corresponding regions from our input are obtained, we can have region-based paired real data. These paired data are used to train our model using a teacher-student framework, allowing the model to iteratively learn from less challenging regions to more challenging regions. Our adaptive-correction aims to rectify errors in our model's predictions, such as over-saturation and color shifts. The idea is to learn from clear night input training videos based on the differences or distance between those input videos and their corresponding predictions. Our model learns from these differences, compelling our model to correct the errors. From extensive experiments, our method demonstrates state-of-the-art performance. It achieves a PSNR of 26.73dB, surpassing existing nighttime video deraining methods by a substantial margin of 13.7%.

----

## [376] Unsupervised Pan-Sharpening via Mutually Guided Detail Restoration

**Authors**: *Huangxing Lin, Yuhang Dong, Xinghao Ding, Tianpeng Liu, Yongxiang Liu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i4.28125](https://doi.org/10.1609/aaai.v38i4.28125)

**Abstract**:

Pan-sharpening is a task that aims to super-resolve the low-resolution multispectral (LRMS) image with the guidance of a corresponding high-resolution panchromatic (PAN) image. The key challenge in pan-sharpening is to accurately modeling the relationship between the MS and PAN images. While supervised deep learning methods are commonly employed to address this task, the unavailability of ground-truth severely limits their effectiveness. In this paper, we propose a mutually guided detail restoration method for unsupervised pan-sharpening. Specifically, we treat pan-sharpening as a blind image deblurring task, in which the blur kernel can be estimated by a CNN. Constrained by the blur kernel, the pan-sharpened image retains spectral information consistent with the LRMS image. Once the pan-sharpened image is obtained, the PAN image is blurred using a pre-defined blur operator. The pan-sharpened image, in turn, is used to guide the detail restoration of the blurred PAN image. By leveraging the mutual guidance between MS and PAN images, the pan-sharpening network can implicitly learn the spatial relationship between the two modalities. Extensive experiments show that the proposed method significantly outperforms existing unsupervised pan-sharpening methods.

----

## [377] Gramformer: Learning Crowd Counting via Graph-Modulated Transformer

**Authors**: *Hui Lin, Zhiheng Ma, Xiaopeng Hong, Qinnan Shangguan, Deyu Meng*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i4.28126](https://doi.org/10.1609/aaai.v38i4.28126)

**Abstract**:

Transformer has been popular in recent crowd counting work since it breaks the limited receptive field of traditional CNNs. However, since crowd images always contain a large number of similar patches, the self-attention mechanism in Transformer tends to find a homogenized solution where the attention maps of almost all patches are identical. In this paper, we address this problem by proposing Gramformer: a graph-modulated transformer to enhance the network by adjusting the attention and input node features respectively on the basis of two different types of graphs. Firstly, an attention graph is proposed to diverse attention maps to attend to complementary information. The graph is building upon the dissimilarities between patches, modulating the attention in an anti-similarity fashion. Secondly, a feature-based centrality encoding is proposed to discover the centrality positions or importance of nodes. We encode them with a proposed centrality indices scheme to modulate the node features and similarity relationships. Extensive experiments on four challenging crowd counting datasets have validated the competitiveness of the proposed method. Code is available at https://github.com/LoraLinH/Gramformer.

----

## [378] Weakly Supervised Open-Vocabulary Object Detection

**Authors**: *Jianghang Lin, Yunhang Shen, Bingquan Wang, Shaohui Lin, Ke Li, Liujuan Cao*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i4.28127](https://doi.org/10.1609/aaai.v38i4.28127)

**Abstract**:

Despite weakly supervised object detection (WSOD) being a promising step toward evading strong instance-level annotations, its capability is confined to closed-set categories within a single training dataset. In this paper, we propose a novel weakly supervised open-vocabulary object detection framework, namely WSOVOD, to extend traditional WSOD to detect novel concepts and utilize diverse datasets with only image-level annotations. To achieve this, we explore three vital strategies, including dataset-level feature adaptation, image-level salient object localization, and region-level vision-language alignment. First, we perform data-aware feature extraction to produce an input-conditional coefficient, which is leveraged into dataset attribute prototypes to identify dataset bias and help achieve cross-dataset generalization. Second, a customized location-oriented weakly supervised region proposal network is proposed to utilize high-level semantic layouts from the category-agnostic segment anything model to distinguish object boundaries. Lastly, we introduce a proposal-concept synchronized multiple-instance network, i.e., object mining and refinement with visual-semantic alignment, to discover objects matched to the text embeddings of concepts. Extensive experiments on Pascal VOC and MS COCO demonstrate that the proposed WSOVOD achieves new state-of-the-art compared with previous WSOD methods in both close-set object localization and detection tasks. Meanwhile, WSOVOD enables cross-dataset and open-vocabulary learning to achieve on-par or even better performance than well-established fully-supervised open-vocabulary object detection (FSOVOD).

----

## [379] Spot the Error: Non-autoregressive Graphic Layout Generation with Wireframe Locator

**Authors**: *Jieru Lin, Danqing Huang, Tiejun Zhao, Dechen Zhan, Chin-Yew Lin*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i4.28128](https://doi.org/10.1609/aaai.v38i4.28128)

**Abstract**:

Layout generation is a critical step in graphic design to achieve meaningful compositions of elements. Most previous works view it as a sequence generation problem by concatenating element attribute tokens (i.e., category, size, position). So far the autoregressive approach (AR) has achieved promising results, but is still limited in global context modeling and suffers from error propagation since it can only attend to the previously generated tokens. Recent non-autoregressive attempts (NAR) have shown competitive results, which provides a wider context range and the flexibility to refine with iterative decoding. However, current works only use simple heuristics to recognize erroneous tokens for refinement which is inaccurate. This paper first conducts an in-depth analysis to better understand the difference between the AR and NAR framework. Furthermore, based on our observation that pixel space is more sensitive in capturing spatial patterns of graphic layouts (e.g., overlap, alignment), we propose a learning-based locator to detect erroneous tokens which takes the wireframe image rendered from the generated layout sequence as input. We show that it serves as a complementary modality to the element sequence in object space and contributes greatly to the overall performance. Experiments on two public datasets show that our approach outperforms both AR and NAR baselines. Extensive studies further prove the effectiveness of different modules with interesting findings. Our code will be available at https://github.com/ffffatgoose/SpotError.

----

## [380] M2SD: Multiple Mixing Self-Distillation for Few-Shot Class-Incremental Learning

**Authors**: *Jinhao Lin, Ziheng Wu, Weifeng Lin, Jun Huang, Ronghua Luo*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i4.28129](https://doi.org/10.1609/aaai.v38i4.28129)

**Abstract**:

Few-shot Class-incremental learning (FSCIL) is a challenging task in machine learning that aims to recognize new classes from a limited number of instances while preserving the ability to classify previously learned classes without retraining the entire model. This presents challenges in updating the model with new classes using limited training data, particularly in balancing acquiring new knowledge while retaining the old. We propose a novel method named Multiple Mxing Self-Distillation (M2SD) during the training phase to address these issues. Specifically, we propose a dual-branch structure that facilitates the expansion of the entire feature space to accommodate new classes. Furthermore, we introduce a feature enhancement component that can pass additional enhanced information back to the base network by self-distillation, resulting in improved classification performance upon adding new classes. After training, we discard both structures, leaving only the primary network to classify new class instances. Extensive experiments demonstrate that our approach achieves superior performance over previous state-of-the-art methods.

----

## [381] EDA: Evolving and Distinct Anchors for Multimodal Motion Prediction

**Authors**: *Longzhong Lin, Xuewu Lin, Tianwei Lin, Lichao Huang, Rong Xiong, Yue Wang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i4.28130](https://doi.org/10.1609/aaai.v38i4.28130)

**Abstract**:

Motion prediction is a crucial task in autonomous driving, and one of its major challenges lands in the multimodality of future behaviors.
Many successful works have utilized mixture models which require identification of positive mixture components, and correspondingly fall into two main lines: prediction-based and anchor-based matching.
The prediction clustering phenomenon in prediction-based matching makes it difficult to pick representative trajectories for downstream tasks, while the anchor-based matching suffers from a limited regression capability.
In this paper, we introduce a novel paradigm, named Evolving and Distinct Anchors (EDA), to define the positive and negative components for multimodal motion prediction based on mixture models.
We enable anchors to evolve and redistribute themselves under specific scenes for an enlarged regression capacity.
Furthermore, we select distinct anchors before matching them with the ground truth, which results in impressive scoring performance.
Our approach enhances all metrics compared to the baseline MTR, particularly with a notable relative reduction of 13.5% in Miss Rate, resulting in state-of-the-art performance on the Waymo Open Motion Dataset.
Appendix and code are available at https://github.com/Longzhong-Lin/EDA.

----

## [382] PTUS: Photo-Realistic Talking Upper-Body Synthesis via 3D-Aware Motion Decomposition Warping

**Authors**: *Luoyang Lin, Zutao Jiang, Xiaodan Liang, Liqian Ma, Michael C. Kampffmeyer, Xiaochun Cao*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i4.28131](https://doi.org/10.1609/aaai.v38i4.28131)

**Abstract**:

Talking upper-body synthesis is a promising task due to its versatile potential for video creation and consists of animating the body and face from a source image with the motion from a given driving video. However, prior synthesis approaches fall short in addressing this task and have been either limited to animating heads of a target person only, or have animated the upper body but neglected the synthesis of precise facial details. To tackle this task, we propose a Photo-realistic Talking Upper-body Synthesis method via 3D-aware motion decomposition warping, named PTUS, to both precisely synthesize the upper body as well as recover the details of the face such as blinking and lip synchronization. In particular, the motion decomposition mechanism consists of a face-body motion decomposition, which decouples the 3D motion estimation of the face and body, and a local-global motion decomposition, which decomposes the 3D face motion into global and local motions resulting in the transfer of facial expression. The 3D-aware warping module transfers the large-scale and subtle 3D motions to the extracted 3D depth-aware features in a coarse-tofine manner. Moreover, we present a new dataset, Talking-UB, which includes upper-body images with high-resolution faces, addressing the limitations of prior datasets that either consist of only facial images or upper-body images with blurry faces. Experimental results demonstrate that our proposed method can synthesize high-quality videos that preserve facial details, and achieves superior results compared to state-of-the-art cross-person motion transfer approaches. Code and collected dataset are released in https://github.com/cooluoluo/PTUS.

----

## [383] Exploring Temporal Feature Correlation for Efficient and Stable Video Semantic Segmentation

**Authors**: *Matthieu Lin, Jenny Sheng, Yubin Hu, Yangguang Li, Lu Qi, Andrew Zhao, Gao Huang, Yong-Jin Liu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i4.28132](https://doi.org/10.1609/aaai.v38i4.28132)

**Abstract**:

This paper tackles the problem of efficient and stable video semantic segmentation. While stability has been under-explored, prevalent work in efficient video semantic segmentation uses the keyframe paradigm. They efficiently process videos by only recomputing the low-level features and reusing high-level features computed at selected keyframes. In addition, the reused features stabilize the predictions across frames, thereby improving video consistency. However, dynamic scenes in the video can easily lead to misalignments between reused and recomputed features, which hampers performance. Moreover, relying on feature reuse to improve prediction consistency is brittle; an erroneous alignment of the features can easily lead to unstable predictions. Therefore, the keyframe paradigm exhibits a dilemma between stability and performance. We address this efficiency and stability challenge using a novel yet simple Temporal Feature Correlation (TFC) module. It uses the cosine similarity between two frames’ low-level features to inform the semantic label’s consistency across frames. Specifically, we selectively reuse label-consistent features across frames through linear interpolation and update others through sparse multi-scale deformable attention. As a result, we no longer directly reuse features to improve stability and thus effectively solve feature misalignment. This work provides a significant step towards efficient and stable video semantic segmentation. On the VSPW dataset, our method significantly improves the prediction consistency of image-based methods while being as fast and accurate.

----

## [384] Boosting Adversarial Transferability across Model Genus by Deformation-Constrained Warping

**Authors**: *Qinliang Lin, Cheng Luo, Zenghao Niu, Xilin He, Weicheng Xie, Yuanbo Hou, Linlin Shen, Siyang Song*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i4.28133](https://doi.org/10.1609/aaai.v38i4.28133)

**Abstract**:

Adversarial examples generated by a surrogate model typically exhibit limited transferability to unknown target systems. To address this problem, many transferability enhancement approaches (e.g., input transformation and model augmentation) have been proposed. However, they show poor performances in attacking systems having different model genera from the surrogate model. In this paper, we propose a novel and generic attacking strategy, called Deformation-Constrained Warping Attack (DeCoWA), that can be effectively applied to cross model genus attack. Specifically, DeCoWA firstly augments input examples via an elastic deformation, namely Deformation-Constrained Warping (DeCoW), to obtain rich local details of the augmented input. To avoid severe distortion of global semantics led by random deformation, DeCoW further constrains the strength and direction of the warping transformation by a novel adaptive control strategy. Extensive experiments demonstrate that the transferable examples crafted by our DeCoWA on CNN surrogates can significantly hinder the performance of Transformers (and vice versa) on various tasks, including image classification, video action recognition, and audio recognition. Code is made available at https://github.com/LinQinLiang/DeCoWA.

----

## [385] A Fixed-Point Approach to Unified Prompt-Based Counting

**Authors**: *Wei Lin, Antoni B. Chan*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i4.28134](https://doi.org/10.1609/aaai.v38i4.28134)

**Abstract**:

Existing class-agnostic counting models typically rely on a single type of prompt, e.g., box annotations. This paper aims to establish a comprehensive prompt-based counting framework capable of generating density maps for concerned objects indicated by various prompt types, such as box, point, and text. To achieve this goal, we begin by converting prompts from different modalities into prompt masks without requiring training. These masks are then integrated into a class-agnostic counting methodology for predicting density maps. Furthermore, we introduce a fixed-point inference along with an associated loss function to improve counting accuracy, all without introducing new parameters. The effectiveness of this method is substantiated both theoretically and experimentally. Additionally, a contrastive training scheme is implemented to mitigate dataset bias inherent in current class-agnostic counting datasets, a strategy whose effectiveness is confirmed by our ablation study. Our model excels in prominent class-agnostic datasets and exhibits superior performance in cross-dataset adaptation tasks.

----

## [386] Boosting Multiple Instance Learning Models for Whole Slide Image Classification: A Model-Agnostic Framework Based on Counterfactual Inference

**Authors**: *Weiping Lin, Zhenfeng Zhuang, Lequan Yu, Liansheng Wang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i4.28135](https://doi.org/10.1609/aaai.v38i4.28135)

**Abstract**:

Multiple instance learning is an effective paradigm for whole slide image (WSI) classification, where labels are only provided at the bag level. However, instance-level prediction is also crucial as it offers insights into fine-grained regions of interest. Existing multiple instance learning methods either solely focus on training a bag classifier or have the insufficient capability of exploring instance prediction. In this work, we propose a novel model-agnostic framework to boost existing multiple instance learning models, to improve the WSI classification performance in both bag and instance levels. Specifically, we propose a counterfactual inference-based sub-bag assessment method and a hierarchical instance searching strategy to help to search reliable instances and obtain their accurate pseudo labels. Furthermore, an instance classifier is well-trained to produce accurate predictions. The instance embedding it generates is treated as a prompt to refine the instance feature for bag prediction. This framework is model-agnostic, capable of adapting to existing multiple instance learning models, including those without specific mechanisms like attention. Extensive experiments on three datasets demonstrate the competitive performance of our method. Code will be available at https://github.com/centurion-crawler/CIMIL.

----

## [387] Relightable and Animatable Neural Avatars from Videos

**Authors**: *Wenbin Lin, Chengwei Zheng, Jun-Hai Yong, Feng Xu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i4.28136](https://doi.org/10.1609/aaai.v38i4.28136)

**Abstract**:

Lightweight creation of 3D digital avatars is a highly desirable but challenging task. With only sparse videos of a person under unknown illumination, we propose a method to create relightable and animatable neural avatars, which can be used to synthesize photorealistic images of humans under novel viewpoints, body poses, and lighting. The key challenge here is to disentangle the geometry, material of the clothed body, and lighting, which becomes more difficult due to the complex geometry and shadow changes caused by body motions.  To solve this ill-posed problem, we propose novel techniques to better model the geometry and shadow changes.  For geometry change modeling, we propose an invertible deformation field, which helps to solve the inverse skinning problem and leads to better geometry quality. To model the spatial and temporal varying shading cues, we propose a pose-aware part-wise light visibility network to estimate light occlusion. Extensive experiments on synthetic and real datasets show that our approach reconstructs high-quality geometry and generates realistic shadows under different body poses. Code and data are available at https://wenbin-lin.github.io/RelightableAvatar-page.

----

## [388] TD²-Net: Toward Denoising and Debiasing for Video Scene Graph Generation

**Authors**: *Xin Lin, Chong Shi, Yibing Zhan, Zuopeng Yang, Yaqi Wu, Dacheng Tao*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i4.28137](https://doi.org/10.1609/aaai.v38i4.28137)

**Abstract**:

Dynamic scene graph generation (SGG) focuses on detecting objects in a video and determining their pairwise relationships. Existing dynamic SGG methods usually suffer from several issues, including 1) Contextual noise, as some frames might contain occluded and blurred objects. 2) Label bias, primarily due to the high imbalance between a few positive relationship samples and numerous negative ones. Additionally, the distribution of relationships exhibits a long-tailed pattern. To address the above problems, in this paper, we introduce a network named TD2-Net that aims at denoising and debiasing for dynamic SGG. Specifically, we first propose a denoising spatio-temporal transformer module that enhances object representation with robust contextual information. This is achieved by designing a differentiable Top-K object selector that utilizes the gumbel-softmax sampling strategy to select the relevant neighborhood for each object. 
Second, we introduce an asymmetrical reweighting loss to relieve the issue of label bias. This loss function integrates asymmetry focusing factors and the volume of samples to adjust the weights assigned to individual samples. Systematic experimental results demonstrate the superiority of our proposed TD2-Net over existing state-of-the-art approaches on Action Genome databases. In more detail, TD2-Net outperforms the second-best competitors by 12.7% on mean-Recall@10 for predicate classification.

----

## [389] Ced-NeRF: A Compact and Efficient Method for Dynamic Neural Radiance Fields

**Authors**: *Youtian Lin*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i4.28138](https://doi.org/10.1609/aaai.v38i4.28138)

**Abstract**:

Rendering photorealistic dynamic scenes has been a focus of recent research, with applications in virtual and augmented reality. While the Neural Radiance Field (NeRF) has shown remarkable rendering quality for static scenes, achieving real-time rendering of dynamic scenes remains challenging due to expansive computation for the time dimension. The incorporation of explicit-based methods, specifically voxel grids, has been proposed to accelerate the training and rendering of neural radiance fields with hybrid representation. However, employing a hybrid representation for dynamic scenes results in overfitting due to fast convergence, which can result in artifacts (e.g., floaters, noisy geometric) on novel views. To address this, we propose a compact and efficient method for dynamic neural radiance fields, namely Ced-NeRF which only require a small number of additional parameters to construct a hybrid representation of dynamic NeRF. Evaluation of dynamic scene datasets shows that our Ced-NeRF achieves fast rendering speeds while maintaining high-quality rendering results. Our method outperforms the current state-of-the-art methods in terms of quality, training and rendering speed.

----

## [390] TagCLIP: A Local-to-Global Framework to Enhance Open-Vocabulary Multi-Label Classification of CLIP without Training

**Authors**: *Yuqi Lin, Minghao Chen, Kaipeng Zhang, Hengjia Li, Mingming Li, Zheng Yang, Dongqin Lv, Binbin Lin, Haifeng Liu, Deng Cai*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i4.28139](https://doi.org/10.1609/aaai.v38i4.28139)

**Abstract**:

Contrastive Language-Image Pre-training (CLIP) has demonstrated impressive capabilities in open-vocabulary classification. The class token in the image encoder is trained to capture the global features to distinguish different text descriptions supervised by contrastive loss, making it highly effective for single-label classification. However, it shows poor performance on multi-label datasets because the global feature tends to be dominated by the most prominent class and the contrastive nature of softmax operation aggravates it.
In this study, we observe that the multi-label classification results heavily rely on discriminative local features but are overlooked by CLIP. As a result, we dissect the preservation of patch-wise spatial information in CLIP and proposed a local-to-global framework to obtain image tags. It comprises three steps: (1) patch-level classification to obtain coarse scores; (2) dual-masking attention refinement (DMAR) module to refine the coarse scores; (3) class-wise reidentification (CWR) module to remedy predictions from a global perspective. This framework is solely based on frozen CLIP and significantly enhances its multi-label classification performance on various benchmarks without dataset-specific training. Besides, to comprehensively assess the quality and practicality of generated tags, we extend their application to the downstream task, i.e., weakly supervised semantic segmentation (WSSS) with generated tags as image-level pseudo labels. Experiments demonstrate that this classify-then-segment paradigm dramatically outperforms other annotation-free segmentation methods and validates the effectiveness of generated tags. Our code is available at https://github.com/linyq2117/TagCLIP.

----

## [391] Independency Adversarial Learning for Cross-Modal Sound Separation

**Authors**: *Zhenkai Lin, Yanli Ji, Yang Yang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i4.28140](https://doi.org/10.1609/aaai.v38i4.28140)

**Abstract**:

The sound mixture separation is still challenging due to heavy sound overlapping and disturbance from noise. Unsupervised separation would significantly increase the difficulty. As sound overlapping always hinders accurate sound separation, we propose an Independency Adversarial Learning based Cross-Modal Sound Separation (IAL-CMS) approach, where IAL employs adversarial learning to minimize the correlation of separated sound elements, exploring high sound independence; CMS performs cross-modal sound separation, incorporating audio-visual consistent feature learning and interactive cross-attention learning to emphasize the semantic consistency among cross-modal features. Both audio-visual consistency and audio consistency are kept to guarantee accurate separation. The consistency and sound independence ensure the decomposition of overlapping mixtures into unrelated and distinguishable sound elements. The proposed approach is evaluated on MUSIC, VGGSound, and AudioSet. Extensive experiments certify that our approach outperforms existing approaches in supervised and unsupervised scenarios.

----

## [392] BEV-MAE: Bird's Eye View Masked Autoencoders for Point Cloud Pre-training in Autonomous Driving Scenarios

**Authors**: *Zhiwei Lin, Yongtao Wang, Shengxiang Qi, Nan Dong, Ming-Hsuan Yang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i4.28141](https://doi.org/10.1609/aaai.v38i4.28141)

**Abstract**:

Existing LiDAR-based 3D object detection methods for autonomous driving scenarios mainly adopt the training-from-scratch paradigm. Unfortunately, this paradigm heavily relies on large-scale labeled data, whose collection can be expensive and time-consuming. Self-supervised pre-training is an effective and desirable way to alleviate this dependence on extensive annotated data. In this work, we present BEV-MAE, an efficient masked autoencoder pre-training framework for LiDAR-based 3D object detection in autonomous driving. Specifically, we propose a bird's eye view (BEV) guided masking strategy to guide the 3D encoder learning feature representation in a BEV perspective and avoid complex decoder design during pre-training. Furthermore, we introduce a learnable point token to maintain a consistent receptive field size of the 3D encoder with fine-tuning for masked point cloud inputs. Based on the property of outdoor point clouds in autonomous driving scenarios, i.e., the point clouds of distant objects are more sparse, we propose point density prediction to enable the 3D encoder to learn location information, which is essential for object detection. Experimental results show that BEV-MAE surpasses prior state-of-the-art self-supervised methods and achieves a favorably pre-training efficiency. Furthermore, based on TransFusion-L, BEV-MAE achieves new state-of-the-art LiDAR-based 3D object detection results, with 73.6 NDS and 69.6 mAP on the nuScenes benchmark. The source code will be released at https://github.com/VDIGPKU/BEV-MAE.

----

## [393] Focus Stacking with High Fidelity and Superior Visual Effects

**Authors**: *Bo Liu, Bin Hu, Xiuli Bi, Weisheng Li, Bin Xiao*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i4.28142](https://doi.org/10.1609/aaai.v38i4.28142)

**Abstract**:

Focus stacking is a technique in computational photography, and it synthesizes a single all-in-focus image from different focal plane images. It is difficult for previous works to produce a high-quality all-in-focus image that meets two goals: high-fidelity to its source images and good visual effects without defects or abnormalities. This paper proposes a novel method based on optical imaging process analysis and modeling. Based on a foreground segmentation - diffusion elimination architecture, the foreground segmentation makes most of the areas in full-focus images heritage information from the source images to achieve high fidelity; diffusion elimination models the physical imaging process and is specially used to solve the transition region (TR) problem that is a long-term neglected issue and degrades visual effects of synthesized images. Based on extensive experiments on simulated dataset, existing realistic dataset and our proposed BetaFusion dataset, the results show that our proposed method can generate high-quality all-in-focus images by achieving two goals simultaneously, especially can successfully solve the TR problem and eliminate the visual effect degradation of synthesized images caused by the TR problem.

----

## [394] DeepBranchTracer: A Generally-Applicable Approach to Curvilinear Structure Reconstruction Using Multi-Feature Learning

**Authors**: *Chao Liu, Ting Zhao, Nenggan Zheng*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i4.28143](https://doi.org/10.1609/aaai.v38i4.28143)

**Abstract**:

Curvilinear structures, which include line-like continuous objects, are fundamental geometrical elements in image-based applications. Reconstructing these structures from images constitutes a pivotal research area in computer vision. However, the complex topology and ambiguous image evidence render this process a challenging task. In this paper, we introduce DeepBranchTracer, a novel method that learns both external image features and internal geometric characteristics to reconstruct curvilinear structures. Firstly, we formulate the curvilinear structures extraction as a geometric attribute estimation problem. Then, a curvilinear structure feature learning network is designed to extract essential branch attributes, including the image features of centerline and boundary, and the geometric features of direction and radius. Finally, utilizing a multi-feature fusion tracing strategy, our model iteratively traces the entire branch by integrating the extracted image and geometric features. We extensively evaluated our model on both 2D and 3D datasets, demonstrating its superior performance over existing segmentation and reconstruction methods in terms of accuracy and continuity.

----

## [395] Decoupling Degradations with Recurrent Network for Video Restoration in Under-Display Camera

**Authors**: *Chengxu Liu, Xuan Wang, Yuanting Fan, Shuai Li, Xueming Qian*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i4.28144](https://doi.org/10.1609/aaai.v38i4.28144)

**Abstract**:

Under-display camera (UDC) systems are the foundation of full-screen display devices in which the lens mounts under the display. The pixel array of light-emitting diodes used for display diffracts and attenuates incident light, causing various degradations as the light intensity changes. Unlike general video restoration which recovers video by treating different degradation factors equally, video restoration for UDC systems is more challenging that concerns removing diverse degradation over time while preserving temporal consistency. In this paper, we introduce a novel video restoration network, called D2RNet, specifically designed for UDC systems. It employs a set of Decoupling Attention Modules (DAM) that effectively separate the various video degradation factors. More specifically, a soft mask generation function is proposed to formulate each frame into flare and haze based on the diffraction arising from incident light of different intensities, followed by the proposed flare and haze removal components that leverage long- and short-term feature learning to handle the respective degradations. Such a design offers an targeted and effective solution to eliminating various types of degradation in UDC systems. We further extend our design into multi-scale to overcome the scale-changing of degradation that often occur in long-range videos. To demonstrate the superiority of D2RNet, we propose a large-scale UDC video benchmark by gathering HDR videos and generating realistically degraded videos using the point spread function measured by a commercial UDC system. Extensive quantitative and qualitative evaluations demonstrate the superiority of D2RNet compared to other state-of-the-art video restoration and UDC image restoration methods.

----

## [396] Unsupervised Domain Adaptative Temporal Sentence Localization with Mutual Information Maximization

**Authors**: *Daizong Liu, Xiang Fang, Xiaoye Qu, Jianfeng Dong, He Yan, Yang Yang, Pan Zhou, Yu Cheng*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i4.28145](https://doi.org/10.1609/aaai.v38i4.28145)

**Abstract**:

Temporal sentence localization (TSL) aims to localize a target segment in a video according to a given sentence query. Though respectable works have made decent achievements in this task, they severely rely on abundant yet expensive manual annotations for training. Moreover, these trained data-dependent models usually can not generalize well to unseen scenarios because of the inherent domain shift. To facilitate this issue, in this paper, we target another more practical but challenging setting: unsupervised domain adaptative temporal sentence localization (UDA-TSL), which explores whether the localization knowledge can be transferred from a fully-annotated data domain (source domain) to a new unannotated data domain (target domain). Particularly, we propose an effective and novel baseline for UDA-TSL to bridge the multi-modal gap across different domains and learn the potential correspondence between the video-query pairs in target domain. We first develop separate modality-specific domain adaptation modules to smoothly balance the minimization of the domain shifts in cross-dataset video and query domains. Then, to fully exploit the semantic correspondence of both modalities in target domain for unsupervised localization, we devise a mutual information learning module to adaptively align the video-query pairs which are more likely to be relevant in target domain, leading to more truly aligned target pairs and ensuring the discriminability of target features. In this way, our model can learn domain-invariant and semantic-aligned cross-modal representations. Three sets of migration experiments show that our model achieves competitive performance compared to existing methods.

----

## [397] Explicitly Perceiving and Preserving the Local Geometric Structures for 3D Point Cloud Attack

**Authors**: *Daizong Liu, Wei Hu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i4.28146](https://doi.org/10.1609/aaai.v38i4.28146)

**Abstract**:

Deep learning models for point clouds have shown to be vulnerable to adversarial attacks, which have received increasing attention in various safety-critical applications such as autonomous driving, robotics, and surveillance. Existing 3D attack methods generally employ global distance losses to implicitly constrain the point-wise perturbations for optimization. However, these simple losses are quite difficult to accurately measure and restrict the proper 3D geometry as point clouds are highly structured. Although few recent works try to exploit additional shape-aware surface knowledge to globally constrain the point position, they still fail to preserve the detailed point-to-point geometric dependency in different local regions. To this end, in this paper, we propose a novel Multi-grained Geometry-aware Attack (MGA), which explicitly captures the local topology characteristics in different 3D regions for adversarial constraint. Specifically, we first develop multi-scale spectral local filter banks adapting to different 3D object shapes to explore potential geometric structures in local regions. Considering that objects may contain complex geometries, we then extend each filter bank into multi-layer ones to gradually capture the topology contexts of the same region in a coarse-to-fine manner. Hence, the focused local geometric structures will be highlighted in the coefficients calculated by the filtering process. At last, by restricting these coefficients between benign and adversarial samples, our MGA is able to properly measure and preserve the detailed geometry contexts in the whole 3D object with trivial perturbations. Extensive experiments demonstrate that our attack can achieve superior performance on various 3D classification models, with satisfying adversarial imperceptibility and strong resistance to different defense methods.

----

## [398] Adv-Diffusion: Imperceptible Adversarial Face Identity Attack via Latent Diffusion Model

**Authors**: *Decheng Liu, Xijun Wang, Chunlei Peng, Nannan Wang, Ruimin Hu, Xinbo Gao*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i4.28147](https://doi.org/10.1609/aaai.v38i4.28147)

**Abstract**:

Adversarial attacks involve adding perturbations to the source image to cause misclassification by the target model, which demonstrates the potential of attacking face recognition models. Existing adversarial face image generation methods still can’t achieve satisfactory performance because of low transferability and high detectability. In this paper, we propose a unified framework Adv-Diffusion that can generate imperceptible adversarial identity perturbations in the latent space but not the raw pixel space, which utilizes strong inpainting capabilities of the latent diffusion model to generate realistic adversarial images. Specifically, we propose the identity-sensitive conditioned diffusion generative model to generate semantic perturbations in the surroundings. The designed adaptive strength-based adversarial perturbation algorithm can ensure both attack transferability and stealthiness. Extensive qualitative and quantitative experiments on the public FFHQ and CelebA-HQ datasets prove the proposed method achieves superior performance compared with the state-of-the-art methods without an extra generative model training process. The source code is available at https://github.com/kopper-xdu/Adv-Diffusion.

----

## [399] Multi-View Dynamic Reflection Prior for Video Glass Surface Detection

**Authors**: *Fang Liu, Yuhao Liu, Jiaying Lin, Ke Xu, Rynson W. H. Lau*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i4.28148](https://doi.org/10.1609/aaai.v38i4.28148)

**Abstract**:

Recent research has shown significant interest in image-based glass surface detection (GSD). However, detecting glass surfaces in dynamic scenes remains largely unexplored due to the lack of a high-quality dataset and an effective video glass surface detection (VGSD) method. In this paper, we propose the first VGSD approach. Our key observation is that reflections frequently appear on glass surfaces, but they change dynamically as the camera moves. Based on this observation, we propose to offset the excessive dependence on a single uncertainty reflection via joint modeling of temporal and spatial reflection cues. To this end, we propose the VGSD-Net with two novel modules: a Location-aware Reflection Extraction (LRE) module and a Context-enhanced Reflection Integration (CRI) module, for the position-aware reflection feature extraction and the spatial-temporal reflection cues integration, respectively. We have also created the first large-scale video glass surface dataset (VGSD-D), consisting of 19,166 image frames with accurately-annotated glass masks extracted from 297 videos. Extensive experiments demonstrate that VGSD-Net outperforms state-of-the-art approaches adapted from related fields. Code and dataset will be available at https://github.com/fawnliu/VGSD.

----



[Go to the previous page](AAAI-2024-list01.md)

[Go to the next page](AAAI-2024-list03.md)

[Go to the catalog section](README.md)