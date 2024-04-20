## [1000] Frame Flexible Network

        **Authors**: *Yitian Zhang, Yue Bai, Chang Liu, Huan Wang, Sheng Li, Yun Fu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01012](https://doi.org/10.1109/CVPR52729.2023.01012)

        **Abstract**:

        Existing video recognition algorithms always conduct different training pipelines for inputs with different frame numbers, which requires repetitive training operations and multiplying storage costs. If we evaluate the model using other frames which are not used in training, we observe the performance will drop significantly (see Fig. 1), which is summarized as Temporal Frequency Deviation phenomenon. To fix this issue, we propose a general frame-work, named Frame Flexible Network (FFN), which not only enables the model to be evaluated at different frames to adjust its computation, but also reduces the memory costs of storing multiple models significantly. Concretely, FFN integrates several sets of training sequences, involves Multi-Frequency Alignment (MFAL) to learn temporal frequency invariant representations, and leverages Multi-Frequency Adaptation (MFAD) to further strengthen the representation abilities. Comprehensive empirical validations using various architectures and popular benchmarks solidly demonstrate the effectiveness and generalization of FFN (e.g., 7.08/5.15/2.17% performance gain at Frame 4/8/16 on Something-Something V1 dataset over Uniformer). Code is available at https://github.com/BeSpontaneous/FFN.

        ----

        ## [1001] System-Status-Aware Adaptive Network for Online Streaming Video Understanding

        **Authors**: *Lin Geng Foo, Jia Gong, Zhipeng Fan, Jun Liu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01013](https://doi.org/10.1109/CVPR52729.2023.01013)

        **Abstract**:

        Recent years have witnessed great progress in deep neural networks for real-time applications. However, most existing works do not explicitly consider the general case where the device's state and the available resources fluctuate over time, and none of them investigate or address the impact of varying computational resources for online video understanding tasks. This paper proposes a System-status-aware Adaptive Network (SAN) that considers the device's real-time state to provide high-quality predictions with low delay. Usage of our agent's policy improves efficiency and robustness to fluctuations of the system status. On two widely used video understanding tasks, SAN obtains state-of-the-art performance while constantly keeping processing delays low. Moreover, training such an agent on various types of hardware configurations is not easy as the labeled training data might not be available, or can be computationally prohibitive. To address this challenging problem, we propose a Meta Self-supervised Adaptation (MSA) method that adapts the agent's policy to new hardware configurations at test-time, allowing for easy deployment of the model onto other unseen hardware platforms.

        ----

        ## [1002] MDQE: Mining Discriminative Query Embeddings to Segment Occluded Instances on Challenging Videos

        **Authors**: *Minghan Li, Shuai Li, Wangmeng Xiang, Lei Zhang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01014](https://doi.org/10.1109/CVPR52729.2023.01014)

        **Abstract**:

        While impressive progress has been achieved, video instance segmentation (VIS) methods with per-clip input often fail on challenging videos with occluded objects and crowded scenes. This is mainly because instance queries in these methods cannot encode well the discriminative embeddings of instances, making the query-based segmenter difficult to distinguish those ‘hard’ instances. To address these issues, we propose to mine discriminative query embeddings (MDQE) to segment occluded instances on challenging videos. First, we initialize the positional embeddings and content features of object queries by considering their spatial contextual information and the inter-frame object motion. Second, we propose an inter-instance mask repulsion loss to distance each instance from its nearby non-target instances. The proposed MDQE is the first VIS method with per-clip input that achieves state-of-the-art results on challenging videos and competitive performance on simple videos. In specific, MDQE with ResNet50 achieves 33.0% and 44.5% mask AP on OVIS and YouTube- Vis 2021, respectively. Code of MDQE can be found at https://github.com/MinghanLi/MDQE_CVPR2023.

        ----

        ## [1003] Spatio-Temporal Pixel-Level Contrastive Learning-based Source-Free Domain Adaptation for Video Semantic Segmentation

        **Authors**: *Shao-Yuan Lo, Poojan Oza, Sumanth Chennupati, Alejandro Galindo, Vishal M. Patel*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01015](https://doi.org/10.1109/CVPR52729.2023.01015)

        **Abstract**:

        Unsupervised Domain Adaptation (UDA) of semantic segmentation transfers labeled source knowledge to an un-labeled target domain by relying on accessing both the source and target data. However, the access to source data is often restricted or infeasible in real-world scenar-ios. Under the source data restrictive circumstances, UDA is less practical. To address this, recent works have ex-plored solutions under the Source-Free Domain Adaptation (SFDA) setup, which aims to adapt a source-trained model to the target domain without accessing source data. Still, existing SFDA approaches use only image-level information for adaptation, making them sub-optimal in video applications. This paper studies SFDA for Video Semantic Segmentation (VSS), where temporal information is lever-aged to address video adaptation. Specifically, we propose Spatio-Temporal Pixel-Level (STPL) contrastive learning, a novel method that takes full advantage of spatio-temporal information to tackle the absence of source data better. STPL explicitly learns semantic correlations among pixels in the spatio-temporal space, providing strong self-supervision for adaptation to the unlabeled target domain. Extensive experiments show that STPL achieves state-of-the-art performance on VSS benchmarks compared to current UDA and SFDA approaches. Code is available at: https://github.com/shaoyuanlo/STPL

        ----

        ## [1004] Taming Diffusion Models for Audio-Driven Co-Speech Gesture Generation

        **Authors**: *Lingting Zhu, Xian Liu, Xuanyu Liu, Rui Qian, Ziwei Liu, Lequan Yu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01016](https://doi.org/10.1109/CVPR52729.2023.01016)

        **Abstract**:

        Animating virtual avatars to make co-speech gestures facilitates various applications in human-machine interaction. The existing methods mainly rely on generative adversarial networks (GANs), which typically suffer from notorious mode collapse and unstable training, thus making it difficult to learn accurate audio-gesture joint distributions. In this work, we propose a novel diffusion-based framework, named Diffusion Co-Speech Gesture (DiffGesture), to effectively capture the cross-modal audio-to-gesture associations and preserve temporal coherence for high-fidelity audio-driven co-speech gesture generation. Specifically, we first establish the diffusion-conditional generation process on clips of skeleton sequences and audio to enable the whole framework. Then, a novel Diffusion Audio-Gesture Transformer is devised to better attend to the information from multiple modalities and model the long-term temporal dependency. Moreover, to eliminate temporal inconsistency, we propose an effective Diffusion Gesture Stabilizer with an annealed noise sampling strategy. Benefiting from the architectural advantages of diffusion models, we further incorporate implicit classifier-free guidance to trade off between diversity and gesture quality. Extensive experiments demonstrate that DiffGesture achieves state-of-the-art performance, which renders coherent gestures with better mode coverage and stronger audio correlations. Code is available at https://github.com/Advocate99/DiffGesture.

        ----

        ## [1005] Chat2Map: Efficient Scene Mapping from Multi-Ego Conversations

        **Authors**: *Sagnik Majumder, Hao Jiang, Pierre Moulon, Ethan Henderson, Paul Calamia, Kristen Grauman, Vamsi Krishna Ithapu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01017](https://doi.org/10.1109/CVPR52729.2023.01017)

        **Abstract**:

        Can conversational videos captured from multiple egocentric viewpoints reveal the map of a scene in a cost-efficient way? We seek to answer this question by proposing a new problem: efficiently building the map of a previously unseen 3D environment by exploiting shared information in the egocentric audio-visual observations of participants in a natural conversation. Our hypothesis is that as multiple people (“egos”) move in a scene and talk among themselves, they receive rich audio-visual cues that can help uncover the unseen areas of the scene. Given the high cost of continuously processing egocentric visual streams, we further explore how to actively coordinate the sampling of visual information, so as to minimize redundancy and reduce power use. To that end, we present an audio-visual deep reinforcement learning approach that works with our shared scene mapper to selectively turn on the camera to efficiently chart out the space. We evaluate the approach using a state-of-the-art audio-visual simulator for 3D scenes as well as real-world video. Our model outperforms previous state-of-the-art mapping methods, and achieves an excellent cost-accuracy tradeoff. Project: http://vision.cs.utexas.edu/projects/chat2map.

        ----

        ## [1006] Audio-Visual Grouping Network for Sound Localization from Mixtures

        **Authors**: *Shentong Mo, Yapeng Tian*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01018](https://doi.org/10.1109/CVPR52729.2023.01018)

        **Abstract**:

        Sound source localization is a typical and challenging task that predicts the location of sound sources in a video. Previous single-source methods mainly used the audio-visual association as clues to localize sounding objects in each image. Due to the mixed property of multiple sound sources in the original space, there exist rare multi-source approaches to localizing multiple sources simultaneously, except for one recent work using a contrastive random walk in the graph with images and separated sound as nodes. Despite their promising performance, they can only handle a fixed number of sources, and they cannot learn compact class-aware representations for individual sources. To alleviate this shortcoming, in this paper, we propose a novel audio-visual grouping network, namely AVGN, that can directly learn category-wise semantic features for each source from the input audio mixture and image to localize multiple sources simultaneously. Specifically, our AVGN leverages learnable audio-visual class tokens to aggregate class-aware source features. Then, the aggregated semantic features for each source can be used as guidance to localize the corresponding visual regions. Compared to existing multi-source methods, our new framework can localize a flexible number of sources and disentangle category-aware audio-visual representations for individual sound sources. We conduct extensive experiments on MUSIC, VGGSound-Instruments, and VGG-Sound Sources benchmarks. The results demonstrate that the proposed AVGN can achieve state-of-the-art sounding object localization performance on both single-source and multi-source scenarios. Code is available at https://github.com/stoneMo/AVGN.

        ----

        ## [1007] Language-Guided Audio-Visual Source Separation via Trimodal Consistency

        **Authors**: *Reuben Tan, Arijit Ray, Andrea Burns, Bryan A. Plummer, Justin Salamon, Oriol Nieto, Bryan Russell, Kate Saenko*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01019](https://doi.org/10.1109/CVPR52729.2023.01019)

        **Abstract**:

        We propose a self-supervised approach for learning to perform audio source separation in videos based on natu-ral language queries, using only unlabeled video and au-dio pairs as training data. A key challenge in this task is learning to associate the linguistic description of a sound-emitting object to its visual features and the corresponding components of the audio waveform, all without access to annotations during training. To overcome this challenge, we adapt off-the-shelf vision-language foundation models to provide pseudo-target supervision via two novel loss functions and encourage a stronger alignment between the audio, visual and natural language modalities. During inference, our approach can separate sounds given text, video and audio input, or given text and audio input alone. We demonstrate the effectiveness of our self-supervised approach on three audio-visual separation datasets, including MUSIC, SOLOS and AudioSet, where we outperform state-of-the-art strongly supervised approaches despite not using object detectors or text labels during training. Our project page including publicly available code can be found at https://cs-people.bu.edu/rxtan/projectsNAST.

        ----

        ## [1008] Fine-grained Audible Video Description

        **Authors**: *Xuyang Shen, Dong Li, Jinxing Zhou, Zhen Qin, Bowen He, Xiaodong Han, Aixuan Li, Yuchao Dai, Lingpeng Kong, Meng Wang, Yu Qiao, Yiran Zhong*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01020](https://doi.org/10.1109/CVPR52729.2023.01020)

        **Abstract**:

        We explore a new task for audio-visual-language modeling called fine-grained audible video description (FAVD). It aims to provide detailed textual descriptions for the given audible videos, including the appearance and spatial locations of each object, the actions of moving objects, and the sounds in videos. Existing visual-language modeling tasks often concentrate on visual cues in videos while undervaluing the language and audio modalities. On the other hand, FAVD requires not only audio-visual-language modeling skills but also paragraph-level language generation abilities. We construct the first fine-grained audible video description benchmark (FAVDBench) to facilitate this research. For each video clip, we first provide a one-sentence summary of the video, i.e., the caption, followed by 4–6 sentences describing the visual details and 1–2 audio-related descriptions at the end. The descriptions are provided in both English and Chinese. We create two new metrics for this task: an EntityScore to gauge the completeness of entities in the visual descriptions, and an AudioScore to assess the audio descriptions. As a preliminary approach to this task, we propose an audio-visual-language transformer that extends existing video captioning model with an additional audio branch. We combine the masked language modeling and auto-regressive language modeling losses to optimize our model so that it can produce paragraph-level descriptions. We illustrate the efficiency of our model in audio-visual-language modeling by evaluating it against the proposed benchmark using both conventional captioning metrics and our proposed metrics. We further put our benchmark to the test in video generation models, demonstrating that employing fine-grained video descriptions can create more intricate videos than using captions. Code and dataset are available at https://github.com/OpenNLPLab/FAVDBench. Our online benchmark is available at www.avlbench.opennlplab.cn.

        ----

        ## [1009] Neural Koopman Pooling: Control-Inspired Temporal Dynamics Encoding for Skeleton-Based Action Recognition

        **Authors**: *Xinghan Wang, Xin Xu, Yadong Mu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01021](https://doi.org/10.1109/CVPR52729.2023.01021)

        **Abstract**:

        Skeleton-based human action recognition is becoming increasingly important in a variety of fields. Most existing works train a CNN or GCN based backbone to extract spatial-temporal features, and use temporal average/max pooling to aggregate the information. However, these pooling methods fail to capture high-order dynamics information. To address the problem, we propose a plug-and-play module called Koopman pooling, which is a parameterized high-order pooling technique based on Koopman theory. The Koopman operator linearizes a non-linear dynamics system, thus providing a way to represent the complex system through the dynamics matrix, which can be used for classification. We also propose an eigenvalue normalization method to encourage the learned dynamics to be non-decaying and stable. Besides, we also show that our Koopman pooling framework can be easily extended to one-shot action recognition when combined with Dynamic Mode Decomposition. The proposed method is evaluated on three benchmark datasets, namely NTU RGB+D 60, 120 and NW-UCLA. Our experiments clearly demonstrate that Koopman pooling significantly improves the performance under both full-dataset and one-shot settings.

        ----

        ## [1010] Learning Discriminative Representations for Skeleton Based Action Recognition

        **Authors**: *Huanyu Zhou, Qingjie Liu, Yunhong Wang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01022](https://doi.org/10.1109/CVPR52729.2023.01022)

        **Abstract**:

        Human action recognition aims at classifying the category of human action from a segment of a video. Recently, people have dived into designing GCN-based models to extract features from skeletons for performing this task, because skeleton representations are much more efficient and robust than other modalities such as RGB frames. However, when employing the skeleton data, some important clues like related items are also discarded. It results in some ambiguous actions that are hard to be distinguished and tend to be misclassified. To alleviate this problem, we propose an auxiliary feature refinement head (FR Head), which consists of spatial-temporal decoupling and contrastive feature refinement, to obtain discriminative representations of skeletons. Ambiguous samples are dynamically discovered and calibrated in the feature space. Furthermore, FR Head could be imposed on different stages of GCNs to build a multi-level refinement for stronger supervision. Extensive experiments are conducted on NTU RGB+D, NTU RGB+D 120, and NW-UCLA datasets. Our proposed models obtain competitive results from state-of-the-art methods and can help to discriminate those ambiguous samples. Codes are available at https://github.com/zhysora/FR-Head.

        ----

        ## [1011] Therbligs in Action: Video Understanding through Motion Primitives

        **Authors**: *Eadom Dessalene, Michael Maynord, Cornelia Fermüller, Yiannis Aloimonos*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01023](https://doi.org/10.1109/CVPR52729.2023.01023)

        **Abstract**:

        In this paper we introduce a rule-based, compositional, and hierarchical modeling of action using Therbligs as our atoms. Introducing these atoms provides us with a consistent, expressive, contact-centered representation of action. Over the atoms we introduce a differentiable method of rule-based reasoning to regularize for logical consistency. Our approach is complementary to other approaches in that the Therblig-based representations produced by our architecture augment rather than replace existing architectures' representations. We release the first Therblig-centered an-notations over two popular video datasets - EPIC Kitchens 100 and 50-Salads. We also broadly demonstrate benefits to adopting Therblig representations through evaluation on the following tasks: action segmentation, action anticipation, and action recognition - observing an average 10.5%/7.53%/6.5% relative improvement, respectively, over EPIC Kitchens and an average 8.9%/6.63%/4.8% relative improvement, respectively, over 50 Salads. Code and data will be made publicly available.

        ----

        ## [1012] Search-Map-Search: A Frame Selection Paradigm for Action Recognition

        **Authors**: *Mingjun Zhao, Yakun Yu, Xiaoli Wang, Lei Yang, Di Niu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01024](https://doi.org/10.1109/CVPR52729.2023.01024)

        **Abstract**:

        Despite the success of deep learning in video understanding tasks, processing every frame in a video is computationally expensive and often unnecessary in real-time applications. Frame selection aims to extract the most informative and representative frames to help a model better understand video content. Existing frame selection methods either individually sample frames based on per-frame importance prediction, without considering interaction among frames, or adopt reinforcement learning agents to find representative frames in succession, which are costly to train and may lead to potential stability issues. To overcome the limitations of existing methods, we propose a Search-Map-Search learning paradigm which combines the advantages of heuristic search and supervised learning to select the best combination of frames from a video as one entity. By combining search with learning, the proposed method can better capture frame interactions while incurring a low inference overhead. Specifically, we first propose a hierarchical search method conducted on each training video to search for the optimal combination of frames with the lowest error on the downstream task. A feature mapping function is then learned to map the frames of a video to the representation of its target optimal frame combination. During inference, another search is performed on an unseen video to select a combination of frames whose feature representation is close to the projected feature representation. Extensive experiments based on several action recognition benchmarks demonstrate that our frame selection method effectively improves performance of action recognition models, and significantly outperforms a number of competitive baselines.

        ----

        ## [1013] Re2TAL: Rewiring Pretrained Video Backbones for Reversible Temporal Action Localization

        **Authors**: *Chen Zhao, Shuming Liu, Karttikeya Mangalam, Bernard Ghanem*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01025](https://doi.org/10.1109/CVPR52729.2023.01025)

        **Abstract**:

        Temporal action localization (TAL) requires long-form reasoning to predict actions of various durations and complex content. Given limited GPU memory, training TAL end to end (i.e., from videos to predictions) on long videos is a significant challenge. Most methods can only train on pre-extracted features without optimizing them for the localization problem, consequently limiting localization performance. In this work, to extend the potential in TAL networks, we propose a novel end-to-end method Re
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">2</sup>
TAL, which rewires pretrained video backbones for reversible TAL. Re
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">2</sup>
TAL builds a backbone with reversible modules, where the input can be recovered from the output such that the bulky intermediate activations can be cleared from memory during training. Instead of designing one single type of reversible module, we propose a network rewiring mechanism, to transform any module with a residual connection to a reversible module without changing any parameters. This provides two benefits: (1) a large variety of reversible networks are easily obtained from existing and even future model designs, and (2) the reversible models require much less training effort as they reuse the pre-trained parameters of their original non-reversible versions. Re
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">2</sup>
TAL, only using the RGB modality, reaches 37.01% average mAP on ActivityNet-v1.3, a new state-of-the-art record, and mAP 64.9% at tIoU=0.5 on THUMOS-14, outperforming all other RGB-only methods. Code is available at https://github.com/coolbay/Re2TAL.

        ----

        ## [1014] Boosting Weakly-Supervised Temporal Action Localization with Text Information

        **Authors**: *Guozhang Li, De Cheng, Xinpeng Ding, Nannan Wang, Xiaoyu Wang, Xinbo Gao*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01026](https://doi.org/10.1109/CVPR52729.2023.01026)

        **Abstract**:

        Due to the lack of temporal annotation, current Weakly-supervised Temporal Action Localization (WTAL) methods are generally stuck into over-complete or incomplete localization. In this paper, we aim to leverage the text information to boost WTAL from two aspects, i.e., (a) the discriminative objective to enlarge the inter-class difference, thus reducing the over-complete; (b) the generative objective to enhance the intra-class integrity, thus finding more complete temporal boundaries. For the discriminative objective, we propose a Text-Segment Mining (TSM) mechanism, which constructs a text description based on the action class label, and regards the text as the query to mine all class-related segments. Without the temporal annotation of actions, TSM compares the text query with the entire videos across the dataset to mine the best matching segments while ignoring irrelevant ones. Due to the shared sub-actions in different categories of videos, merely applying TSM is too strict to neglect the semantic-related segments, which results in incomplete localization. We further introduce a generative objective named Video-text Language Completion (VLC), which focuses on all semantic-related segments from videos to complete the text sentence. We achieve the state-of-the-art performance on THUMOS14 and ActivityNetl.3. Surprisingly, we also find our proposed method can be seamlessly applied to existing methods, and improve their performances with a clear margin. The code is available at https://github.com/lgzlIlIlI/Boosting-WTAL.

        ----

        ## [1015] Perception and Semantic Aware Regularization for Sequential Confidence Calibration

        **Authors**: *Zhenghua Peng, Yu Luo, Tianshui Chen, Keke Xu, Shuangping Huang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01027](https://doi.org/10.1109/CVPR52729.2023.01027)

        **Abstract**:

        Deep sequence recognition (DSR) models receive increasing attention due to their superior application to various applications. Most DSR models use merely the target sequences as supervision without considering other related sequences, leading to over-confidence in their predictions. The DSR models trained with label smoothing regularize labels by equally and independently smoothing each token, reallocating a small value to other tokens for mitigating overconfidence. However, they do not consider tokens/sequences correlations that may provide more effective information to regularize training and thus lead to sub-optimal performance. In this work, we find tokens/sequences with high perception and semantic correlations with the target ones contain more correlated and effective information and thus facilitate more effective regularization. To this end, we propose a Perception and Semantic aware Sequence Regularization framework, which explore perceptively and semantically correlated tokens/sequences as regularization. Specifically, we introduce a semantic context-free recognition and a language model to acquire similar sequences with high perceptive similarities and semantic correlation, respectively. Moreover, over-confidence degree varies across samples according to their difficulties. Thus, we further design an adaptive calibration intensity module to compute a difficulty score for each samples to obtain finer-grained regularization. Extensive experiments on canonical sequence recognition tasks, including scene text and speech recognition, demonstrate that our method sets novel state-of-the-art results. Code is available at https://github.com/husterpzh/PSSR.

        ----

        ## [1016] NewsNet: A Novel Dataset for Hierarchical Temporal Segmentation

        **Authors**: *Haoqian Wu, Keyu Chen, Haozhe Liu, Mingchen Zhuge, Bing Li, Ruizhi Qiao, Xiujun Shu, Bei Gan, Liangsheng Xu, Bo Ren, Mengmeng Xu, Wentian Zhang, Raghavendra Ramachandra, Chia-Wen Lin, Bernard Ghanem*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01028](https://doi.org/10.1109/CVPR52729.2023.01028)

        **Abstract**:

        Temporal video segmentation is the get-to- go automatic video analysis, which decomposes a long-form video into smaller components for the following-up understanding tasks. Recent works have studied several levels of granularity to segment a video, such as shot, event, and scene. Those segmentations can help compare the semantics in the corresponding scales, but lack a wider view of larger temporal spans, especially when the video is complex and structured. Therefore, we present two abstractive levels of temporal segmentations and study their hierarchy to the existing fine-grained levels. Accordingly, we collect NewsNet, the largest news video dataset consisting of 1,000 videos in over 900 hours, associated with several tasks for hierarchical temporal video segmentation. Each news video is a collection of stories on different topics, represented as aligned audio, visual, and textual data, along with extensive frame-wise annotations in four granularities. We assert that the study on NewsNet can advance the understanding of complex structured video and benefit more areas such as short-video creation, personalized advertisement, digital instruction, and education. Our dataset and code is publicly available at https://github.com/NewsNet-Benchmark/NewsNet.

        ----

        ## [1017] Tell Me What Happened: Unifying Text-guided Video Completion via Multimodal Masked Video Generation

        **Authors**: *Tsu-Jui Fu, Licheng Yu, Ning Zhang, Cheng-Yang Fu, Jong-Chyi Su, William Yang Wang, Sean Bell*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01029](https://doi.org/10.1109/CVPR52729.2023.01029)

        **Abstract**:

        Generating a video given the first several static frames is challenging as it anticipates reasonable future frames with temporal coherence. Besides video prediction, the ability to rewind from the last frame or infilling between the head and tail is also crucial, but they have rarely been explored for video completion. Since there could be different outcomes from the hints of just a few frames, a system that can follow natural language to perform video completion may significantly improve controllability. Inspired by this, we introduce a novel task, text-guided video completion (TVC), which requests the model to generate a video from partial frames guided by an instruction. We then propose Multimodal Masked Video Generation (MMVG) to address this TVC task. During training, MMVG discretizes the video frames into visual tokens and masks most of them to perform video completion from any time point. At inference time, a single MMVG model can address all 3 cases of TVC, including video prediction, rewind, and infilling, by applying corresponding masking conditions. We evaluate MMVG in various video scenarios, including egocentric, animation, and gaming. Extensive experimental results indicate that MMVG is effective in generating high-quality visual appearances with text guidance for TVC.

        ----

        ## [1018] Leveraging Temporal Context in Low Representational Power Regimes

        **Authors**: *Camilo Luciano Fosco, SouYoung Jin, Emilie Josephs, Aude Oliva*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01030](https://doi.org/10.1109/CVPR52729.2023.01030)

        **Abstract**:

        Computer vision models are excellent at identifying and exploiting regularities in the world. However, it is computationally costly to learn these regularities from scratch. This presents a challenge for low-parameter models, like those running on edge devices (e.g. smartphones). Can the performance of models with low representational power be improved by supplementing training with additional information about these statistical regularities? We explore this in the domains of action recognition and action anticipation, leveraging the fact that actions are typically embedded in stereotypical sequences. We introduce the Event Transition Matrix (ETM), computed from action labels in an untrimmed video dataset, which captures the temporal context of a given action, operationalized as the likelihood that it was preceded or followed by each other action in the set. We show that including information from the ETM during training improves action recognition and anticipation performance on various egocentric video datasets. Through ablation and control studies, we show that the coherent sequence of information captured by our ETM is key to this effect, and we find that the benefit of this explicit representation of temporal context is most pronounced for smaller models. Code, matrices and models are available in our project page: https://camilofosco.com/etm_website.

        ----

        ## [1019] Cap4Video: What Can Auxiliary Captions Do for Text-Video Retrieval?

        **Authors**: *Wenhao Wu, Haipeng Luo, Bo Fang, Jingdong Wang, Wanli Ouyang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01031](https://doi.org/10.1109/CVPR52729.2023.01031)

        **Abstract**:

        Most existing text-video retrieval methods focus on cross-modal matching between the visual content of videos and textual query sentences. However, in real-world scenarios, online videos are often accompanied by relevant text information such as titles, tags, and even subtitles, which can be utilized to match textual queries. This insight has motivated us to propose a novel approach to text-video retrieval, where we directly generate associated captions from videos using zero-shot video captioning with knowledge from web-scale pre-trained models (e.g., CLIP and GPT-2). Given the generated captions, a natural question arises: what benefits do they bring to text-video retrieval? To answer this, we introduce Cap4Video, a new framework that leverages captions in three ways: i) Input data: video-caption pairs can augment the training data. ii) Intermediate feature interaction: we perform cross-modal feature interaction between the video and caption to produce enhanced video representations. iii) Output score: the Query-Caption matching branch can complement the original Query-Video matching branch for text-video retrieval. We conduct comprehensive ablation studies to demonstrate the effectiveness of our approach. Without any post-processing, Cap4Video achieves state-of-the-art performance on four standard text-video retrieval benchmarks: MSR-VTT (51.4%), VATEX (66.6%), MSVD (51.8%), and DiDeMo (52.0%). The code is available at https://github.com/whwu95/Cap4Video.

        ----

        ## [1020] Vid2Seq: Large-Scale Pretraining of a Visual Language Model for Dense Video Captioning

        **Authors**: *Antoine Yang, Arsha Nagrani, Paul Hongsuck Seo, Antoine Miech, Jordi Pont-Tuset, Ivan Laptev, Josef Sivic, Cordelia Schmid*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01032](https://doi.org/10.1109/CVPR52729.2023.01032)

        **Abstract**:

        In this work, we introduce Vid2Seq, a multi-modal single-stage dense event captioning model pretrained on narrated videos which are readily-available at scale. The Vid2Seq architecture augments a language model with special time tokens, allowing it to seamlessly predict event boundaries and textual descriptions in the same output sequence. Such a unified model requires large-scale training data, which is not available in current annotated datasets. We show that it is possible to leverage unlabeled narrated videos for dense video captioning, by reformulating sentence boundaries of transcribed speech as pseudo event boundaries, and using the transcribed speech sentences as pseudo event captions. The resulting Vid2Seq model pretrained on the YT-Temporal-1B dataset improves the state of the art on a variety of dense video captioning benchmarks including YouCook2, ViTT and ActivityNet Captions. Vid2Seq also generalizes well to the tasks of video paragraph captioning and video clip captioning, and to few-shot settings. Our code is publicly available at [1].

        ----

        ## [1021] Procedure-Aware Pretraining for Instructional Video Understanding

        **Authors**: *Honglu Zhou, Roberto Martín-Martín, Mubbasir Kapadia, Silvio Savarese, Juan Carlos Niebles*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01033](https://doi.org/10.1109/CVPR52729.2023.01033)

        **Abstract**:

        Our goal is to learn a video representation that is useful for downstream procedure understanding tasks in instructional videos. Due to the small amount of available annotations, a key challenge in procedure understanding is to be able to extract from unlabeled videos the procedural knowledge such as the identity of the task (e.g., ‘make latte’), its steps (e.g., ‘pour milk’), or the potential next steps given partial progress in its execution. Our main insight is that instructional videos depict sequences of steps that repeat between instances of the same or different tasks, and that this structure can be well represented by a Procedural Knowledge Graph (PKG), where nodes are discrete steps and edges connect steps that occur sequentially in the instructional activities. This graph can then be used to generate pseudo labels to train a video representation that encodes the procedural knowledge in a more accessible form to generalize to multiple procedure understanding tasks. We build a PKG by combining information from a text-based procedural knowledge database and an unlabeled instructional video corpus and then use it to generate training pseudo labels with four novel pre-training objectives. We call this PKG-based pre-training procedure and the resulting model Paprika, Procedure-Aware PRe-training for Instructional Knowledge Acquisition. We evaluate Paprika on COIN and CrossTask for procedure understanding tasks such as task recognition, step recognition, and step forecasting. Paprika yields a video representation that improves over the state of the art: up to 11.23% gains in accuracy in 12 evaluation settings. Implementation is available at https://github.com/salesforce/paprika.

        ----

        ## [1022] VindLU: A Recipe for Effective Video-and-Language Pretraining

        **Authors**: *Feng Cheng, Xizi Wang, Jie Lei, David J. Crandall, Mohit Bansal, Gedas Bertasius*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01034](https://doi.org/10.1109/CVPR52729.2023.01034)

        **Abstract**:

        The last several years have witnessed remarkable progress in video-and-language (VidL) understanding. However, most modern VidL approaches use complex and specialized model architectures and sophisticated pretraining protocols, making the reproducibility, analysis and comparisons of these frameworks difficult. Hence, instead of proposing yet another new VidL model, this paper conducts a thorough empirical study demystifying the most important factors in the VidL model design. Among the factors that we investigate are (i) the spatiotemporal architecture design, (ii) the multimodal fusion schemes, (iii) the pretraining objectives, (iv) the choice of pretraining data, (v) pretraining and finetuning protocols, and (vi) dataset and model scaling. Our empirical study reveals that the most important design factors include: temporal modeling, video-to-text multimodal fusion, masked modeling objectives, and joint training on images and videos. Using these empirical insights, we then develop a step-by-step recipe, dubbed VindLU, for effective VidL pretraining. Our final model trained using our recipe achieves comparable or better than state-of-the-art results on several VidL tasks without relying on external CLIP pretraining. In particular, on the text-to-video retrieval task, our approach obtains 61.2% on DiDeMo, and 55.0% on ActivityNet, outperforming current SOTA by 7.8% and 6.1% respectively. Furthermore, our model also obtains state-of-the-art video question-answering results on ActivityNet-QA, MSRVTT-QA, MSRVTT-MC and TVQA. Our code and pretrained models are publicly available at: https://github.com/klauscc/VindLU.

        ----

        ## [1023] Modular Memorability: Tiered Representations for Video Memorability Prediction

        **Authors**: *Théo Dumont, Juan Segundo Hevia, Camilo Luciano Fosco*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01035](https://doi.org/10.1109/CVPR52729.2023.01035)

        **Abstract**:

        The question of how to best estimate the memorability of visual content is currently a source of debate in the memorability community. In this paper, we propose to explore how different key properties of images and videos affect their consolidation into memory. We analyze the impact of several features and develop a model that emulates the most important parts of a proposed “pathway to memory”: a simple but effective way of representing the different hurdles that new visual content needs to surpass to stay in memory. This framework leads to the construction of our M3-S model, a novel memorability network that processes input videos in a modular fashion. Each module of the network emulates one of the four key steps of the pathway to memory: raw encoding, scene understanding, event understanding and memory consolidation. We find that the different representations learned by our modules are non-trivial and substantially different from each other. Additionally, we observe that certain representations tend to perform better at the task of memorability prediction than others, and we introduce an in-depth ablation study to support our results. Our proposed approach surpasses the state of the art on the two largest video memorability datasets and opens the door to new applications in the field. Our code is available at https://github.com/tekal-ai/modular-memorability.

        ----

        ## [1024] Multivariate, Multi-Frequency and Multimodal: Rethinking Graph Neural Networks for Emotion Recognition in Conversation

        **Authors**: *Feiyu Chen, Jie Shao, Shuyuan Zhu, Heng Tao Shen*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01036](https://doi.org/10.1109/CVPR52729.2023.01036)

        **Abstract**:

        Complex relationships of high arity across modality and context dimensions is a critical challenge in the Emotion Recognition in Conversation (ERC) task. Yet, previous works tend to encode multimodal and contextual relationships in a loosely-coupled manner, which may harm relationship modelling. Recently, Graph Neural Networks (GNN) which show advantages in capturing data relations, offer a new solution for ERC. However, existing GNN-based ERC models fail to address some general limits of GNNs, including assuming pairwise formulation and erasing high-frequency signals, which may be trivial for many applications but crucial for the ERC task. In this paper, we propose a GNN-based model that explores multivariate relationships and captures the varying importance of emotion discrepancy and commonality by valuing multi-frequency signals. We empower GNNs to better capture the inherent relationships among utterances and deliver more sufficient multimodal and contextual modelling. Experimental results show that our proposed method outperforms previous state-of-the-art works on two popular multimodal ERC datasets.

        ----

        ## [1025] Distilling Cross-Temporal Contexts for Continuous Sign Language Recognition

        **Authors**: *Leming Guo, Wanli Xue, Qing Guo, Bo Liu, Kaihua Zhang, Tiantian Yuan, Shengyong Chen*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01037](https://doi.org/10.1109/CVPR52729.2023.01037)

        **Abstract**:

        Continuous sign language recognition (CSLR) aims to recognize glosses in a sign language video. State-of-the-art methods typically have two modules, a spatial perception module and a temporal aggregation module, which are jointly learned end-to-end. Existing results in [9, 20, 25, 36] have indicated that, as the frontal component of the over-all model, the spatial perception module used for spatial feature extraction tends to be insufficiently trained. In this paper, we first conduct empirical studies and show that a shallow temporal aggregation module allows more thor-ough training of the spatial perception module. However, a shallow temporal aggregation module cannot well capture both local and global temporal context information in sign language. To address this dilemma, we propose a cross-temporal context aggregation (CTCA) model. Specifically, we build a dual-path network that contains two branches for perceptions of local temporal context and global temporal context. We further design a cross-context knowledge distil-lation learning objective to aggregate the two types of con-text and the linguistic prior. The knowledge distillation en-ables the resultant one-branch temporal aggregation mod-ule to perceive local-global temporal and semantic context. This shallow temporal perception module structure facili-tates spatial perception module learning. Extensive exper-iments on challenging CSLR benchmarks demonstrate that our method outperforms all state-of-the-art methods.

        ----

        ## [1026] You Need Multiple Exiting: Dynamic Early Exiting for Accelerating Unified Vision Language Model

        **Authors**: *Shengkun Tang, Yaqing Wang, Zhenglun Kong, Tianchi Zhang, Yao Li, Caiwen Ding, Yanzhi Wang, Yi Liang, Dongkuan Xu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01038](https://doi.org/10.1109/CVPR52729.2023.01038)

        **Abstract**:

        Large-scale Transformer models bring significant improvements for various downstream vision language tasks with a unified architecture. The performance improvements come with increasing model size, resulting in slow inference speed and increased cost for severing. While some certain predictions benefit from the full computation of the large-scale model, not all of inputs need the same amount of computation to conduct, potentially leading to computation resource waste. To handle this challenge, early exiting is proposed to adaptively allocate computational power in term of input complexity to improve inference efficiency. The existing early exiting strategies usually adopt output confidence based on intermediate layers as a proxy of input complexity to incur the decision of skipping following layers. However, such strategies cannot be applied to encoder in the widely-used unified architecture with both encoder and decoder due to difficulty of output confidence estimation in the encoder layers. It is suboptimal in term of saving computation power to ignore the early exiting in encoder component. To address this issue, we propose a novel early exiting strategy for unified vision language models, which allows to dynamically skip the layers in encoder and decoder simultaneously in term of input layer-wise similarities with multiple times of early exiting, namely MuE. By decomposing the image and text modalities in the encoder, MuE is flexible and can skip different layers in term of modalities, advancing the inference efficiency while minimizing performance drop. Experiments on the SNLI-VE and MS COCO datasets show that the proposed approach MuE can reduce expected inference time by up to 50% and 40% while maintaining 99% and 96% performance respectively.

        ----

        ## [1027] Layout-based Causal Inference for Object Navigation

        **Authors**: *Sixian Zhang, Xinhang Song, Weijie Li, Yubing Bai, Xinyao Yu, Shuqiang Jiang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01039](https://doi.org/10.1109/CVPR52729.2023.01039)

        **Abstract**:

        Previous works for ObjectNav task attempt to learn the association (e.g. relation graph) between the visual inputs and the goal during training. Such association contains the prior knowledge of navigating in training environments, which is denoted as the experience. The experience performs a positive effect on helping the agent infer the likely location of the goal when the layout gap between the unseen environments of the test and the prior knowledge obtained in training is minor. However, when the layout gap is significant, the experience exerts a negative effect on navigation. Motivated by keeping the positive effect and removing the negative effect of the experience, we propose the layout-based soft Total Direct Effect (L-sTDE) framework based on the causal inference to adjust the prediction of the navigation policy. In particular, we propose to calculate the layout gap which is defined as the KL divergence between the posterior and the prior distribution of the object layout. Then the sTDE is proposed to appropriately control the effect of the experience based on the layout gap. Experimental results on AI2THOR, RoboTHOR, and Habitat demonstrate the effectiveness of our method. The code is available at https://github.com/sx-zhang/Layout-based-sTDE.git.

        ----

        ## [1028] Improving Vision-and-Language Navigation by Generating Future-View Image Semantics

        **Authors**: *Jialu Li, Mohit Bansal*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01040](https://doi.org/10.1109/CVPR52729.2023.01040)

        **Abstract**:

        Vision-and-Language Navigation (VLN) is the task that requires an agent to navigate through the environment based on natural language instructions. At each step, the agent takes the next action by selecting from a set of navigable locations. In this paper, we aim to take one step further and explore whether the agent can benefit from generating the potential future view during navigation. Intuitively, humans will have an expectation of how the future environment will look like, based on the natural language instructions and surrounding views, which will aid correct navigation. Hence, to equip the agent with this ability to generate the semantics of future navigation views, we first propose three proxy tasks during the agent's in-domain pre-training: Masked Panorama Modeling (MPM), Masked Trajectory Modeling (MTM), and Action Prediction with Image Generation (APIG). These three objectives teach the model to predict missing views in a panorama (MPM), predict missing steps in the full trajectory (MTM), and generate the next view based on the full instruction and navigation history (APIG), respectively. We then fine-tune the agent on the VLN task with an auxiliary loss that minimizes the difference between the view semantics generated by the agent and the ground truth view semantics of the next step. Empirically, our VLN-SIG achieves the new state-of-the-art on both Room-to-Room dataset and CVDN dataset. We further show that our agent learns to fill in missing patches in future views qualitatively, which brings more interpretability over agents' predicted actions. Lastly, we demonstrate that learning to predict future view semantics also enables the agent to have better performance on longer paths.
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
Code is available at https://github.com/jialuli-luka/VLN-SIG.git.

        ----

        ## [1029] A New Path: Scaling Vision-and-Language Navigation with Synthetic Instructions and Imitation Learning

        **Authors**: *Aishwarya Kamath, Peter Anderson, Su Wang, Jing Yu Koh, Alexander Ku, Austin Waters, Yinfei Yang, Jason Baldridge, Zarana Parekh*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01041](https://doi.org/10.1109/CVPR52729.2023.01041)

        **Abstract**:

        Recent studies in Vision-and-Language Navigation (VLN) train RL agents to execute natural-language navigation instructions in photorealistic environments, as a step towards robots that can follow human instructions. However, given the scarcity of human instruction data and limited diversity in the training environments, these agents still struggle with complex language grounding and spatial language understanding. Pretraining on large text and image-text datasets from the web has been extensively explored but the improvements are limited. We investigate large-scale augmentation with synthetic instructions. We take 500+ indoor environments captured in densely-sampled 360 ° panoramas, construct navigation trajectories through these panoramas, and generate a visually-grounded instruction for each trajectory using Marky [63], a high-quality multilingual navigation instruction generator. We also synthesize image observations from novel viewpoints using an image-to-image GAN [27]. The resulting dataset of 4.2M instruction-trajectory pairs is two orders of magnitude larger than existing human-annotated datasets, and contains a wider variety of environments and viewpoints. To efficiently leverage data at this scale, we train a simple transformer agent with imitation learning. On the challenging RxR dataset, our approach outperforms all existing RL agents, improving the state-of-the-art NDTW from 71.1 to 79.1 in seen environments, and from 64.6 to 66.8 in unseen test environments. Our work points to a new path to improving instruction-following agents, emphasizing large-scale training on near-human quality synthetic instructions.

        ----

        ## [1030] A-CAP: Anticipation Captioning with Commonsense Knowledge

        **Authors**: *Duc Minh Vo, Quoc-An Luong, Akihiro Sugimoto, Hideki Nakayama*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01042](https://doi.org/10.1109/CVPR52729.2023.01042)

        **Abstract**:

        Humans possess the capacity to reason about the future based on a sparse collection of visual cues acquired over time. In order to emulate this ability, we introduce a novel task called Anticipation Captioning, which generates a caption for an unseen oracle image using a sparsely temporally-ordered set of images. To tackle this new task, we propose a model called A-CAP, which incorporates commonsense knowledge into a pre-trained vision-language model, allowing it to anticipate the caption. Through both qualitative and quantitative evaluations on a customized visual storytelling dataset, A-CAP out-performs other image captioning methods and establishes a strong baseline for anticipation captioning. We also address the challenges inherent in this task.

        ----

        ## [1031] Are Deep Neural Networks SMARTer Than Second Graders?

        **Authors**: *Anoop Cherian, Kuan-Chuan Peng, Suhas Lohit, Kevin A. Smith, Joshua B. Tenenbaum*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01043](https://doi.org/10.1109/CVPR52729.2023.01043)

        **Abstract**:

        Recent times have witnessed an increasing number of applications of deep neural networks towards solving tasks that require superior cognitive abilities, e.g., playing Go, generating art, question answering (e.g., ChatGPT), etc. Such a dramatic progress raises the question: how generalizable are neural networks in solving problems that demand broad skills? To answer this question, we propose SMART: a Simple Multimodal Algorithmic Reasoning Task and the associated SMART-101 dataset
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
The SMART-101 dataset is publicly available at: https://doi.org/10.5281/zenodo.7761800, for evaluating the abstraction, deduction, and generalization abilities of neural networks in solving visuo-linguistic puzzles designed specifically for children in the 6–8 age group. Our dataset consists of 101 unique puzzles; each puzzle comprises a picture and a question, and their solution needs a mix of several elementary skills, including arithmetic, algebra, and spatial reasoning, among others. To scale our dataset towards training deep neural networks, we programmatically generate entirely new instances for each puzzle while retaining their solution algorithm. To benchmark the performance on the SMART-101 dataset, we propose a vision-and-language meta-learning model that can incorporate varied state-of-the-art neural backbones. Our experiments reveal that while powerful deep models offer reasonable performances on puzzles in a supervised setting, they are not better than random accuracy when analyzed for generalization –filling this gap may demand new multimodal learning approaches.

        ----

        ## [1032] Fusing Pre-Trained Language Models with Multimodal Prompts through Reinforcement Learning

        **Authors**: *Youngjae Yu, Jiwan Chung, Heeseung Yun, Jack Hessel, Jae Sung Park, Ximing Lu, Rowan Zellers, Prithviraj Ammanabrolu, Ronan Le Bras, Gunhee Kim, Yejin Choi*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01044](https://doi.org/10.1109/CVPR52729.2023.01044)

        **Abstract**:

        Language models are capable of commonsense reasoning: while domain-specific models can learn from explicit knowledge (e.g. commonsense graphs [6] ethical norms [25]), and larger models like GPT-3 [7] mani-fest broad commonsense reasoning capacity. Can their knowledge be extended to multimodal inputs such as images and audio without paired domain data? In this work, we propose 
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">‡</sup>
ESPER (Extending Sensory PErception with Reinforcement learning) which enables text-only pretrained models to address multimodal tasks such as visual commonsense reasoning. Our key novelty is to use rein-forcement learning to align multimodal inputs to language model generations without direct supervision: for example, our reward optimization relies only on cosine similarity derived from CLIP [52] and requires no additional paired (image, text) data. Experiments demonstrate that ESPER outperforms baselines and prior work on a variety of multimodal text generation tasks ranging from captioning to commonsense reasoning; these include a new benchmark we collect and release, the ESP dataset, which tasks models with generating the text of several different domains for each image. Our code and data are publicly released at https://github.com/JiwanChung/esper.

        ----

        ## [1033] Language Adaptive Weight Generation for Multi-Task Visual Grounding

        **Authors**: *Wei Su, Peihan Miao, Huanzhang Dou, Gaoang Wang, Liang Qiao, Zheyang Li, Xi Li*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01045](https://doi.org/10.1109/CVPR52729.2023.01045)

        **Abstract**:

        Although the impressive performance in visual grounding, the prevailing approaches usually exploit the visual backbone in a passive way, i.e., the visual backbone extracts features with fixed weights without expression-related hints. The passive perception may lead to mismatches (e.g., redundant and missing), limiting further performance improvement. Ideally, the visual backbone should actively extract visual features since the expressions already provide the blueprint of desired visual features. The active perception can take expressions as priors to extract relevant visual features, which can effectively alleviate the mismatches. Inspired by this, we propose an active perception Visual Grounding framework based on Language Adaptive Weights, called VG-LAW. The visual backbone serves as an expression-specific feature extractor through dynamic weights generated for various expressions. Benefiting from the specific and relevant visual features extracted from the language-aware visual backbone, VG-LAW does not require additional modules for cross-modal interaction. Along with a neat multi-task head, VG-LAW can be competent in referring expression comprehension and segmentation jointly. Extensive experiments on four representative datasets, i.e., RefCOCO, RefCOCO+, RefCOCOg, and ReferItGame, validate the effectiveness of the proposed framework and demonstrate state-of-the-art performance.

        ----

        ## [1034] From Images to Textual Prompts: Zero-shot Visual Question Answering with Frozen Large Language Models

        **Authors**: *Jiaxian Guo, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Boyang Li, Dacheng Tao, Steven C. H. Hoi*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01046](https://doi.org/10.1109/CVPR52729.2023.01046)

        **Abstract**:

        Large language models (LLMs) have demonstrated excellent zero-shot generalization to new language tasks. However, effective utilization of LLMs for zero-shot visual question-answering (VQA) remains challenging, primarily due to the modality disconnect and task disconnect between the LLM and VQA tasks. End-to-end training on multimodal data may bridge the disconnects, but is inflexible and computationally expensive. To address this issue, we propose Img2LLM, a plug-and-play module that provides LLM prompts to enable LLMs to perform zeroshot VQA tasks without end-to-end training. We develop LLM-agnostic models describe image content as exemplar question-answer pairs, which prove to be effective LLM prompts. Img2LLM offers the following benefits: 1) It achieves comparable or better performance than methods relying on end-to-end training. For example, we outperform Flamingo [3] by 5.6% on VQAv2. On the challenging A-OKVQA dataset, our method outperforms few-shot methods by as much as 20%. 2) It flexibly interfaces with a wide range of LLMs to perform VQA. 3) It eliminates the need to specialize LLMs using end-to-end finetuning and serve highly specialized LLMs to end users, thereby reducing cost. Code is available via the LAVIS [28] framework at https://github.com/salesforce/LAVIS/tree/main/projects/img2llm-vqa.

        ----

        ## [1035] Diversity-Aware Meta Visual Prompting

        **Authors**: *Qidong Huang, Xiaoyi Dong, Dongdong Chen, Weiming Zhang, Feifei Wang, Gang Hua, Nenghai Yu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01047](https://doi.org/10.1109/CVPR52729.2023.01047)

        **Abstract**:

        We present Diversity-Aware Meta Visual Prompting (DAM-VP), an efficient and effective prompting method for transferring pre-trained models to downstream tasks with frozen backbone. A challenging issue in visual prompting is that image datasets sometimes have a large data diversity whereas a per-dataset generic prompt can hardly handle the complex distribution shift toward the original pretraining data distribution properly. To address this issue, we propose a dataset Diversity-Aware prompting strategy whose initialization is realized by a Meta-prompt. Specifically, we cluster the downstream dataset into small homogeneity subsets in a diversity-adaptive way, with each subset has its own prompt optimized separately. Such a divide-and-conquer design reduces the optimization difficulty greatly and significantly boosts the prompting performance. Furthermore, all the prompts are initialized with a meta-prompt, which is learned across several datasets. It is a bootstrapped paradigm, with the key observation that the prompting knowledge learned from previous datasets could help the prompt to converge faster and perform better on a new dataset. During inference, we dynamically select a proper prompt for each input, based on the feature distance between the input and each subset. Through extensive experiments, our DAM-VP demonstrates superior efficiency and effectiveness, clearly surpassing previous prompting methods in a series of downstream datasets for different pretraining models. Our code is available at: https://github.com/shikiw/DAM-VP.

        ----

        ## [1036] Hierarchical Prompt Learning for Multi-Task Learning

        **Authors**: *Yajing Liu, Yuning Lu, Hao Liu, Yaozu An, Zhuoran Xu, Zhuokun Yao, Baofeng Zhang, Zhiwei Xiong, Chenguang Gui*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01048](https://doi.org/10.1109/CVPR52729.2023.01048)

        **Abstract**:

        Vision-language models (VLMs) can effectively transfer to various vision tasks via prompt learning. Real-world scenarios often require adapting a model to multiple similar yet distinct tasks. Existing methods focus on learning a specific prompt for each task, limiting the ability to exploit potentially shared information from other tasks. Naively training a task-shared prompt using a combination of all tasks ignores fine-grained task correlations. Significant discrepancies across tasks could cause negative transferring. Considering this, we present Hierarchical Prompt (HiPro) learning, a simple and effective method for jointly adapting a pre-trained VLM to multiple downstream tasks. Our method quantifies inter-task affinity and subsequently constructs a hierarchical task tree. Task-shared prompts learned by internal nodes explore the information within the corresponding task group, while task-individual prompts learned by leaf nodes obtain fine-grained information targeted at each task. The combination of hierarchical prompts provides high-quality content of different granularity. We evaluate HiPro on four multi-task learning datasets. The results demonstrate the effectiveness of our method.

        ----

        ## [1037] Task Residual for Tuning Vision-Language Models

        **Authors**: *Tao Yu, Zhihe Lu, Xin Jin, Zhibo Chen, Xinchao Wang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01049](https://doi.org/10.1109/CVPR52729.2023.01049)

        **Abstract**:

        Large-scale vision-language models (VLMs) pre-trained on billion-level data have learned general visual representations and broad visual concepts. In principle, the welllearned knowledge structure of the VLMs should be inherited appropriately when being transferred to downstream tasks with limited data. However, most existing efficient transfer learning (ETL) approaches for VLMs either damage or are excessively biased towards the prior knowledge, e.g., prompt tuning (PT) discards the pre-trained text-based classifier and builds a new one while adapter-style tuning (AT) fully relies on the pre-trained features. To address this, we propose a new efficient tuning approach for VLMs named Task Residual Tuning (TaskRes), which performs directly on the text-based classifier and explicitly decouples the prior knowledge of the pre-trained models and new knowledge regarding a target task. Specifically, TaskRes keeps the original classifier weights from the VLMs frozen and obtains a new classifier for the target task by tuning a set of prior-independent parameters as a residual to the original one, which enables reliable prior knowledge preservation and flexible task-specific knowledge exploration. The proposed TaskRes is simple yet effective, which significantly outperforms previous ETL methods (e.g., PT and AT) on 11 benchmark datasets while requiring minimal effort for the implementation. Our code is available at https://github.com/geekyutao/TaskRes.

        ----

        ## [1038] @ CREPE: Can Vision-Language Foundation Models Reason Compositionally?

        **Authors**: *Zixian Ma, Jerry Hong, Mustafa Omer Gul, Mona Gandhi, Irena Gao, Ranjay Krishna*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01050](https://doi.org/10.1109/CVPR52729.2023.01050)

        **Abstract**:

        A fundamental characteristic common to both human vision and natural language is their compositional nature. Yet, despite the performance gains contributed by large vision and language pretraining, we find that-across 7 architectures trained with 4 algorithms on massive datasets-they struggle at compositionality. To arrive at this conclusion, we introduce a new compositionality evaluation benchmark, @ CREPE, which measures two important aspects of compositionality identified by cognitive science literature: system-aticity and productivity. To measure systematicity, CREPE consists of a test dataset containing over 370K image-text pairs and three different seen-unseen splits. The three splits are designed to test models trained on three popular training datasets: CC-12M, YFCC-15M, and LAION-400M. We also generate 325K, 316K, and 309K hard negative captions for a subset of the pairs. To test productivity, CREPE contains 17 K image-text pairs with nine different complexities plus 278K hard negative captions with atomic, swapping and negation foils. The datasets are generated by repurposing the Visual Genome scene graphs and region descriptions and applying handcrafted templates and GPT-3. For systematicity, we find that model performance decreases consistently when novel compositions dominate the retrieval set, with Recall@1 dropping by up to 9%. For productivity, models' retrieval success decays as complexity increases, frequently nearing random chance at high complexity. These results hold regardless of model and training dataset size.

        ----

        ## [1039] LOCATE: Localize and Transfer Object Parts for Weakly Supervised Affordance Grounding

        **Authors**: *Gen Li, Varun Jampani, Deqing Sun, Laura Sevilla-Lara*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01051](https://doi.org/10.1109/CVPR52729.2023.01051)

        **Abstract**:

        Humans excel at acquiring knowledge through observation. For example, we can learn to use new tools by watching demonstrations. This skill is fundamental for intelligent systems to interact with the world. A key step to acquire this skill is to identify what part of the object affords each action, which is called affordance grounding. In this paper, we address this problem and propose a framework called LOCATE that can identify matching object parts across images, to transfer knowledge from images where an object is being used (exocentric images used for learning), to images where the object is inactive (egocentric ones used to test). To this end, we first find interaction areas and extract their feature embeddings. Then we learn to aggregate the embeddings into compact prototypes (human, object part, and background), and select the one representing the object part. Finally, we use the selected prototype to guide affordance grounding. We do this in a weakly supervised manner, learning only from image-level affordance and object labels. Extensive experiments demonstrate that our approach outperforms state-of-the-art methods by a large margin on both seen and unseen objects.
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
Project page: https://reagan1311.github.io/locate.

        ----

        ## [1040] Overlooked Factors in Concept-Based Explanations: Dataset Choice, Concept Learnability, and Human Capability

        **Authors**: *Vikram V. Ramaswamy, Sunnie S. Y. Kim, Ruth Fong, Olga Russakovsky*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01052](https://doi.org/10.1109/CVPR52729.2023.01052)

        **Abstract**:

        Concept-based interpretability methods aim to explain a deep neural network model's components and predictions using a pre-defined set of semantic concepts. These methods evaluate a trained model on a new, “probe” dataset and correlate the model's outputs with concepts labeled in that dataset. Despite their popularity, they suffer from limitations that are not well-understood and articulated in the literature. In this work, we identify and analyze three commonly overlooked factors in concept-based explanations. First, we find that the choice of the probe dataset has a profound impact on the generated explanations. Our analysis reveals that different probe datasets lead to very different explanations, suggesting that the generated explanations are not generalizable outside the probe dataset. Second, we find that concepts in the probe dataset are often harder to learn than the target classes they are used to explain, calling into question the correctness of the explanations. We argue that only easily learnable concepts should be used in concept-based explanations. Finally, while existing methods use hundreds or even thousands of concepts, our human studies reveal a much stricter upper bound of 32 concepts or less, beyond which the explanations are much less practically useful. We discuss the implications of our findings and provide suggestions for future development of concept-based interpretability methods. Code for our analysis and user interface can be found at https://github.com/princetonvisualai/OverlookedFactors

        ----

        ## [1041] Grounding Counterfactual Explanation of Image Classifiers to Textual Concept Space

        **Authors**: *Siwon Kim, Jinoh Oh, Sungjin Lee, Seunghak Yu, Jaeyoung Do, Tara Taghavi*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01053](https://doi.org/10.1109/CVPR52729.2023.01053)

        **Abstract**:

        Concept-based explanation aims to provide concise and human-understandable explanations of an image classifier. However, existing concept-based explanation methods typically require a significant amount of manually collected concept-annotated images. This is costly and runs the risk of human biases being involved in the explanation. In this paper, we propose Counterfactual explanation with text-driven concepts (CounTEX), where the concepts are defined only from text by leveraging a pretrained multimodal joint embedding space without additional concept-annotated datasets. A conceptual counterfactual explanation is generated with text-driven concepts. To utilize the text-driven concepts defined in the joint embedding space to interpret target classifier outcome, we present a novel projection scheme for mapping the two spaces with a simple yet effective implementation. We show that CounTEX generates faithful explanations that provide a semantic understanding of model decision rationale robust to human bias.

        ----

        ## [1042] GIVL: Improving Geographical Inclusivity of Vision-Language Models with Pre-Training Methods

        **Authors**: *Da Yin, Feng Gao, Govind Thattai, Michael Johnston, Kai-Wei Chang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01054](https://doi.org/10.1109/CVPR52729.2023.01054)

        **Abstract**:

        A key goal for the advancement of AI is to develop technologies that serve the needs not just of one group but of all communities regardless of their geographical re-gion. In fact, a significant proportion of knowledge is locally shared by people from certain regions but may not apply equally in other regions because of cultural dif-ferences. If a model is unaware of regional character-istics, it may lead to performance disparity across re-gions and result in bias against underrepresented groups. We propose GIVL, a Geographically Inclusive Vision-and-Language Pre-trained model. There are two attributes of geo-diverse visual concepts which can help to learn geo-diverse knowledge: 1) concepts under similar categories have unique knowledge and visual characteristics, 2) concepts with similar visual features may fall in completely different categories. Motivated by the attributes, we de-sign new pre-training objectives Image-Knowledge Matching (IKM) and Image Edit Checking (IEC) to pre-train GIVL. Compared with similar-size models pre-trained with similar scale of data, GIVL achieves state-of-the-art (SOTA) and more balanced performance on geo-diverse V &L tasks. Code and data are released at https://github.com/WadeYin9712/GIVL.

        ----

        ## [1043] Learning Bottleneck Concepts in Image Classification

        **Authors**: *Bowen Wang, Liangzhi Li, Yuta Nakashima, Hajime Nagahara*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01055](https://doi.org/10.1109/CVPR52729.2023.01055)

        **Abstract**:

        Interpreting and explaining the behavior of deep neural networks is critical for many tasks. Explainable AI provides a way to address this challenge, mostly by providing per-pixel relevance to the decision. Yet, interpreting such explanations may require expert knowledge. Some recent attempts toward interpretability adopt a concept-based framework, giving a higher-level relationship between some concepts and model decisions. This paper proposes Bottleneck Concept Learner (BotCL), which represents an image solely by the presence/absence of concepts learned through training over the target task without explicit supervision over the concepts. It uses self-supervision and tailored regularizers so that learned concepts can be human-understandable. Using some image classification tasks as our testbed, we demonstrate BotCL's potential to rebuild neural networks for better interpretability
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
Code is avaliable at https://github.com/wbw520/BotCL and a simple demo is available at https://botcl.liangzhili.com/.

        ----

        ## [1044] SceneTrilogy: On Human Scene-Sketch and its Complementarity with Photo and Text

        **Authors**: *Pinaki Nath Chowdhury, Ayan Kumar Bhunia, Aneeshan Sain, Subhadeep Koley, Tao Xiang, Yi-Zhe Song*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01056](https://doi.org/10.1109/CVPR52729.2023.01056)

        **Abstract**:

        In this paper, we extend scene understanding to include that of human sketch. The result is a complete trilogy of scene representation from three diverse and complementary modalities – sketch, photo, and text. Instead of learning a rigid three-way embedding and be done with it, wefocus on learning a flexible joint embedding that fully supports the “optionality” that this complementarity brings. Our embedding supports optionality on two axes: (i) optionality across modalities – use any combination of modalities as query for downstream tasks like retrieval, (ii) optionality across tasks – simultaneously utilising the embedding for either discriminative (e.g., retrieval) or generative tasks (e.g., captioning). This provides flexibility to end-users by exploiting the best of each modality, therefore serving the very purpose behind our proposal of a trilogy in the first place. First, a combination of information-bottleneck and conditional invertible neural networks disentangle the modality-specific component from modality-agnostic in sketch, photo, and text. Second, the modality-agnostic instances from sketch, photo, and text are synergised using a modified cross-attention. Once learned, we show our embedding can accommodate a multi-facet of scene-related tasks, including those enabled for the first time by the inclusion of sketch, all without any task-specific modifications. Project Page: https://pinakinathc.github.io/scenetrilogy

        ----

        ## [1045] Context-aware Alignment and Mutual Masking for 3D-Language Pre-training

        **Authors**: *Zhao Jin, Munawar Hayat, Yuwei Yang, Yulan Guo, Yinjie Lei*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01057](https://doi.org/10.1109/CVPR52729.2023.01057)

        **Abstract**:

        3D visual language reasoning plays an important role in effective human-computer interaction. The current approaches for 3D visual reasoning are task-specific, and lack pre-training methods to learn generic representations that can transfer across various tasks. Despite the encouraging progress in vision-language pre-training for image-text data, 3D-language pre-training is still an open issue due to limited 3D-language paired data, highly sparse and irregular structure of point clouds and ambiguities in spatial relations of 3D objects with viewpoint changes. In this paper, we present a generic 3D-language pre-training approach, that tackles multiple facets of 3D-language reasoning by learning universal representations. Our learning objective constitutes two main parts. 1) Context aware spatial-semantic alignment to establish fine-grained correspondence between point clouds and texts. It reduces relational ambiguities by aligning 3D spatial relationships with textual semantic context. 2) Mutual 3D-Language Masked modeling to enable cross-modality information exchange. Instead of reconstructing sparse 3D points for which language can hardly provide cues, we propose masked proposal reasoning to learn semantic class and mask-invariant representations. Our proposed 3D-language pre-training method achieves promising results once adapted to various downstream tasks, including 3D visual grounding, 3D dense captioning and 3D question answering. Our codes are available at https://github.com/leolyj/3D-VLP

        ----

        ## [1046] MaskCLIP: Masked Self-Distillation Advances Contrastive Language-Image Pretraining

        **Authors**: *Xiaoyi Dong, Jianmin Bao, Yinglin Zheng, Ting Zhang, Dongdong Chen, Hao Yang, Ming Zeng, Weiming Zhang, Lu Yuan, Dong Chen, Fang Wen, Nenghai Yu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01058](https://doi.org/10.1109/CVPR52729.2023.01058)

        **Abstract**:

        This paper presents a simple yet effective framework MaskCLIP, which incorporates a newly proposed masked self-distillation into contrastive language-image pretraining. The core idea of masked self-distillation is to distill representation from a full image to the representation predicted from a masked image. Such incorporation enjoys two vital benefits. First, masked self-distillation targets local patch representation learning, which is complementary to vision-language contrastive focusing on text-related representation. Second, masked self-distillation is also consistent with vision-language contrastive from the perspective of training objective as both utilize the visual encoder for feature aligning, and thus is able to learn local semantics getting indirect supervision from the language. We provide specially designed experiments with a comprehensive analysis to validate the two benefits. Symmetrically, we also introduce the local semantic supervision into the text branch, which further improves the pretraining performance. With extensive experiments, we show that MaskCLIP, when applied to various challenging downstream tasks, achieves superior results in linear probing, finetuning, and zeroshot performance with the guidance of the language encoder. Code will be release at https://github.com/LightDXY/MaskCLIP.

        ----

        ## [1047] CLIPPO: Image-and-Language Understanding from Pixels Only

        **Authors**: *Michael Tschannen, Basil Mustafa, Neil Houlsby*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01059](https://doi.org/10.1109/CVPR52729.2023.01059)

        **Abstract**:

        Multimodal models are becoming increasingly effective, in part due to unified components, such as the Transformer architecture. However, multimodal models still often consist of many task- and modality-specific pieces and training procedures. For example, CLIP (Radford et al., 2021) trains independent text and image towers via a contrastive loss. We explore an additional unification: the use of a pure pixel-based model to perform image, text, and multimodal tasks. Our model is trained with contrastive loss alone, so we call it CLIP-Pixels Only (CLIPPO). CLIPPO uses a single encoder that processes both regular images and text rendered as images. CLIPPO performs image-based tasks such as retrieval and zero-shot image classification almost as well as CLIP-style models, with half the number of parameters and no text-specific tower or embedding. When trained jointly via image-text contrastive learning and next-sentence contrastive learning, CLIPPO can perform well on natural language understanding tasks, without any word-level loss (language modelling or masked language modelling), outperforming pixel-based prior work. Surprisingly, CLIPPO can obtain good accuracy in visual question answering, simply by rendering the question and image together. Finally, we exploit the fact that CLIPPO does not require a tokenizer to show that it can achieve strong performance on multilingual multimodal retrieval without modifications.

        ----

        ## [1048] ViLEM: Visual-Language Error Modeling for Image-Text Retrieval

        **Authors**: *Yuxin Chen, Zongyang Ma, Ziqi Zhang, Zhongang Qi, Chunfeng Yuan, Ying Shan, Bing Li, Weiming Hu, Xiaohu Qie, Jianping Wu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01060](https://doi.org/10.1109/CVPR52729.2023.01060)

        **Abstract**:

        Dominant pre-training works for image-text retrieval adopt “dual-encoder” architecture to enable high efficiency, where two encoders are used to extract image and text representations and contrastive learning is employed for global alignment. However, coarse-grained global alignment ignores detailed semantic associations between image and text. In this work, we propose a novel proxy task, named Visual-Language Error Modeling (ViLEM), to inject detailed image-text association into “dual-encoder” model by “proofreading” each word in the text against the corresponding image. Specifically, we first edit the image-paired text to automatically generate diverse plausible negative texts with pre-trained language models. ViLEM then enforces the model to discriminate the correctness of each word in the plausible negative texts and further correct the wrong words via resorting to image information. Further-more, we propose a multi-granularity interaction framework to perform ViLEM via interacting text features with both global and local image features, which associates local text semantics with both high-level visual context and multi-level local visual information. Our method surpasses state-of-the-art “dual-encoder” methods by a large margin on the image-text retrieval task and significantly improves discriminativeness to local textual semantics. Our model can also generalize well to video-text retrieval.

        ----

        ## [1049] Non-Contrastive Learning Meets Language-Image Pre-Training

        **Authors**: *Jinghao Zhou, Li Dong, Zhe Gan, Lijuan Wang, Furu Wei*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01061](https://doi.org/10.1109/CVPR52729.2023.01061)

        **Abstract**:

        Contrastive language-image pre-training (CLIP) serves as a de-facto standard to align images and texts. Nonetheless, the loose correlation between images and texts of webcrawled data renders the contrastive objective data inefficient and craving for a large training batch size. In this work, we explore the validity of non-contrastive language-image pre-training (nCLIP), and study whether nice properties exhibited in visual self-supervised models can emerge. We empirically observe that the non-contrastive objective benefits representation learning while sufficiently underperforming under zero-shot recognition. Based on the above study, we further introduce xCLIP, a multi-tasking framework combining CLIP and nCLIP, and show that nCLIP aids CLIP in enhancing feature semantics. The synergy between two objectives lets xCLIP enjoy the best of both worlds: superior performance in both zero-shot transfer and representation learning. Systematic evaluation is conducted spanning a wide variety of downstream tasks including zero-shot classification, out-of-domain classification, retrieval, visual representation learning, and textual representation learning, showcasing a consistent performance gain and validating the effectiveness of xCLIP. The code and pre-trained models will be publicly available at https://github.com/shallowtoil/xclip.

        ----

        ## [1050] HAAV: Hierarchical Aggregation of Augmented Views for Image Captioning

        **Authors**: *Chia-Wen Kuo, Zsolt Kira*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01062](https://doi.org/10.1109/CVPR52729.2023.01062)

        **Abstract**:

        A great deal of progress has been made in image captioning, driven by research into how to encode the image using pre-trained models. This includes visual encodings (e.g. image grid features or detected objects) and more recently textual encodings (e.g. image tags or text descriptions of image regions). As more advanced encodings are available and incorporated, it is natural to ask: how to efficiently and effectively leverage the heterogeneous set of encodings? In this paper, we propose to regard the encodings as augmented views of the input image. The image captioning model encodes each view independently with a shared encoder efficiently, and a contrastive loss is incorporated across the encoded views in a novel way to improve their representation quality and the model's data efficiency. Our proposed hierarchical decoder then adaptively weighs the encoded views according to their effectiveness for caption generation by first aggregating within each view at the token level, and then across views at the view level. We demonstrate significant performance improvements of +5.6% CIDEr on MS-COCO and + 12.9% CIDEr on Flickr30k compared to state of the arts, and conduct rigorous analyses to demonstrate the importance of each part of our design.

        ----

        ## [1051] Learning Attribute and Class-Specific Representation Duet for Fine-Grained Fashion Analysis

        **Authors**: *Yang Jiao, Yan Gao, Jingjing Meng, Jin Shang, Yi Sun*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01063](https://doi.org/10.1109/CVPR52729.2023.01063)

        **Abstract**:

        Fashion representation learning involves the analysis and understanding of various visual elements at different granularities and the interactions among them. Existing works often learn fine-grained fashion representations at the attribute level without considering their relationships and inter-dependencies across different classes. In this work, we propose to learn an attribute and class-specific fashion representation duet to better model such attribute relationships and inter-dependencies by leveraging prior knowledge about the taxonomy of fashion attributes and classes. Through two sub-networks for the attributes and classes, respectively, our proposed an embedding network progressively learns and refines the visual representation of a fashion image to improve its robustness for fashion retrieval. A multi-granularity loss consisting of attribute-level and class-level losses is proposed to introduce appropriate inductive bias to learn across different granularities of the fashion representations. Experimental results on three benchmark datasets demonstrate the effectiveness of our method, which outperforms the state-of-the-art methods by a large margin.

        ----

        ## [1052] Learning Instance-Level Representation for Large-Scale Multi-Modal Pretraining in E-Commerce

        **Authors**: *Yang Jin, Yongzhi Li, Zehuan Yuan, Yadong Mu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01064](https://doi.org/10.1109/CVPR52729.2023.01064)

        **Abstract**:

        This paper aims to establish a generic multi-modal foundation model that has the scalable capability to massive downstream applications in E-commerce. Recently, large-scale vision-language pretraining approaches have achieved remarkable advances in the general domain. However, due to the significant differences between natural and product images, directly applying these frameworks for modeling image-level representations to E-commerce will be inevitably sub-optimal. To this end, we propose an instance-centric multi-modal pretraining paradigm called ECLIP in this work. In detail, we craft a decoder architecture that introduces a set of learnable instance queries to explicitly aggregate instance-level semantics. Moreover, to enable the model to focus on the desired product instance without reliance on expensive manual annotations, two specially configured pretext tasks are further proposed. Pretrained on the 100 million E-commerce-related data, ECLIP successfully extracts more generic, semantic-rich, and robust representations. Extensive experimental results show that, without further fine-tuning, ECLIP surpasses existing methods by a large margin on a broad range of downstream tasks, demonstrating the strong transferability to real-world E-commerce applications.

        ----

        ## [1053] Cross-Image-Attention for Conditional Embeddings in Deep Metric Learning

        **Authors**: *Dmytro Kotovenko, Pingchuan Ma, Timo Milbich, Björn Ommer*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01065](https://doi.org/10.1109/CVPR52729.2023.01065)

        **Abstract**:

        Learning compact image embeddings that yield seman-tic similarities between images and that generalize to un-seen test classes, is at the core of deep metric learning (DML). Finding a mapping from a rich, localized image feature map onto a compact embedding vector is challenging: Although similarity emerges between tuples of images, DML approaches marginalize out information in an individ-ual image before considering another image to which simi-larity is to be computed. Instead, we propose during training to condition the em-bedding of an image on the image we want to compare it to. Rather than embedding by a simple pooling as in standard DML, we use cross-attention so that one image can iden-tify relevant features in the other image. Consequently, the attention mechanism establishes a hierarchy of conditional embeddings that gradually incorporates information about the tuple to steer the representation of an individual image. The cross-attention layers bridge the gap between the origi-nal unconditional embedding and the final similarity and al-low backpropagtion to update encodings more directly than through a lossy pooling layer. At test time we use the re-sulting improved unconditional embeddings, thus requiring no additional parameters or computational overhead. Ex-periments on established DML benchmarks show that our cross-attention conditional embedding during training im-proves the underlying standard DML pipeline significantly so that it outperforms the state-of-the-art.

        ----

        ## [1054] Asymmetric Feature Fusion for Image Retrieval

        **Authors**: *Hui Wu, Min Wang, Wengang Zhou, Zhenbo Lu, Houqiang Li*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01066](https://doi.org/10.1109/CVPR52729.2023.01066)

        **Abstract**:

        In asymmetric retrieval systems, models with different capacities are deployed on platforms with different computational and storage resources. Despite the great progress, existing approaches still suffer from a dilemma between retrieval efficiency and asymmetric accuracy due to the limited capacity of the lightweight query model. In this work, we propose an Asymmetric Feature Fusion (AFF) paradigm, which advances existing asymmetric retrieval systems by considering the complementarity among different features just at the gallery side. Specifically, it first embeds each gallery image into various features, e.g., local features and global features. Then, a dynamic mixer is introduced to aggregate these features into compact embedding for efficient search. On the query side, only a single lightweight model is deployed for feature extraction. The query model and dynamic mixer are jointly trained by sharing a momentum-updated classifier. Notably, the proposed paradigm boosts the accuracy of asymmetric retrieval without introducing any extra overhead to the query side. Exhaustive experiments on various landmark retrieval datasets demonstrate the superiority of our paradigm.

        ----

        ## [1055] Improving Zero-shot Generalization and Robustness of Multi-Modal Models

        **Authors**: *Yunhao Ge, Jie Ren, Andrew Gallagher, Yuxiao Wang, Ming-Hsuan Yang, Hartwig Adam, Laurent Itti, Balaji Lakshminarayanan, Jiaping Zhao*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01067](https://doi.org/10.1109/CVPR52729.2023.01067)

        **Abstract**:

        Multi-modal- image-text models such as CLIP and LiT have demonstrated impressive performance on image classification benchmarks and their zero-shot generalization ability is particularly exciting. While the top-5 zero-shot accuracies of these models are very high, the top-1 accuracies are much lower (over 25% gap in some cases). We investigate the reasons for this performance gap and find that many of the failure cases are caused by ambiguity in the text prompts. First, we develop a simple and efficient zero-shot post-hoc method to identify images whose top-1 prediction is likely to be incorrect, by measuring consistency of the predictions w.r.t. multiple prompts and image transformations. We show that our procedure better predicts mistakes, outperforming the popular max logit baseline on selective prediction tasks. Next, we propose a simple and efficient way to improve accuracy on such uncertain images by making use of the WordNet hierarchy; specifically we augment the original class by incorporating its parent and children from the semantic label hierarchy, and plug the augmentation into text prompts. We conduct experiments on both CLIP and LiT models with five different ImageNet-based datasets. For CLIP, our method improves the top-1 accuracy by 17.13% on the uncertain subset and 3.6% on the entire ImageNet validation set. We also show that our method improves across ImageNet shifted datasets, four other datasets, and other model architectures such as LiT. The proposed method
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
Work carried out mainly at Google is hyperparameter-free, requires no additional model training and can be easily scaled to other large multi-modal architectures. Code is available at https://github.com/gyhandy/Hierarchy-CLIP.

        ----

        ## [1056] Hint-Aug: Drawing Hints from Foundation Vision Transformers towards Boosted Few-shot Parameter-Efficient Tuning

        **Authors**: *Zhongzhi Yu, Shang Wu, Yonggan Fu, Shunyao Zhang, Yingyan Celine Lin*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01068](https://doi.org/10.1109/CVPR52729.2023.01068)

        **Abstract**:

        Despite the growing demand for tuningfoundation vision transformers (FViTs) on downstream tasks, fully unleashing FViTs' potential under data-limited scenarios (e.g., few-shot tuning) remains a challenge due to FViTs' data-hungry nature. Common data augmentation techniques fall short in this context due to the limited features contained in the few-shot tuning data. To tackle this challenge, we first identify an opportunity for FViTs in few-shot tuning: pretrained FViTs themselves have already learned highly representative features from large-scale pretraining data, which are fully preserved during widely used parameter-efficient tuning. We thus hypothesize that leveraging those learned features to augment the tuning data can boost the effectiveness of few-shot FViT tuning. To this end, we propose a framework called Hint-based Data Augmentation (Hint-Aug), which aims to boost FViT in few-shot tuning byaugmenting the over-fitted parts of tuning samples with the learned features of pretrained FViTs. Specifically, Hint-Aug integrates two key enablers: (1) an Attentive Over-fitting Detector (AOD) to detect over-confident patches offoundation ViTs for potentially alleviating their over-fitting on the few-shot tuning data and (2) a Confusion-based Feature Infusion (CFI) module to infuse easy-to-confuse features from the pretrained FViTs with the over-confident patches detected by the above AOD in order to enhance the feature diversity during tuning. Extensive experiments and ablation studies on five datasets and three parameter-efficient tuning techniques consistently validate Hint-Aug's effectiveness: 0.04% ~ 32.91 % higher accuracy over the state-of-the-art (SOTA) data augmentation method under various low-shot settings. For example, on the Pet dataset, Hint-Aug achieves a 2.22% higher accuracy with 50% less training data over SOTA data augmentation methods.

        ----

        ## [1057] Visual DNA: Representing and Comparing Images Using Distributions of Neuron Activations

        **Authors**: *Benjamin Ramtoula, Matthew Gadd, Paul Newman, Daniele De Martini*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01069](https://doi.org/10.1109/CVPR52729.2023.01069)

        **Abstract**:

        Selecting appropriate datasets is critical in modern computer vision. However, no general-purpose tools exist to evaluate the extent to which two datasets differ. For this, we propose representing images - and by extension datasets - using Distributions of Neuron Activations (DNAs). DNAsfit distributions, such as histograms or Gaussians, to activations of neurons in a pre-trained feature extractor through which we pass the imager s) to represent. This extractor is frozen for all datasets, and we rely on its generally expressive power in feature space. By comparing two DNAs, we can evaluate the extent to which two datasets differ with granular control over the comparison attributes of interest, providing the ability to customise the way distances are measured to suit the requirements of the task at hand. Furthermore, DNAs are compact, representing datasets of any size with less than 15 megabytes. We demonstrate the value of DNAs by evaluating their applicability on several tasks, including conditional dataset comparison, synthetic image evaluation, and transfer learning, and across diverse datasets, ranging from synthetic cat images to celebrity faces and urban driving scenes.

        ----

        ## [1058] End-to-End 3D Dense Captioning with Vote2Cap-DETR

        **Authors**: *Sijin Chen, Hongyuan Zhu, Xin Chen, Yinjie Lei, Gang Yu, Tao Chen*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01070](https://doi.org/10.1109/CVPR52729.2023.01070)

        **Abstract**:

        3D dense captioning aims to generate multiple captions localized with their associated object regions. Existing methods follow a sophisticated “detect-then-describe” pipeline equipped with numerous hand-crafted components. However, these hand-crafted components would yield sub-optimal performance given cluttered object spatial and class distributions among different scenes. In this paper, we propose a simple-yet-effective transformer framework Vote2Cap-DETR based on recent popular DEtection TRansformer (DETR). Compared with prior arts, our framework has several appealing advantages: 1) Without resorting to numerous hand-crafted components, our method is based on a full transformer encoder-decoder architecture with a learnable vote query driven object decoder, and a caption decoder that produces the dense captions in a set-prediction manner. 2) In contrast to the two-stage scheme, our method can perform detection and captioning in one-stage. 3) Without bells and whistles, extensive experiments on two commonly used datasets, ScanRefer and Nr3D, demonstrate that our Vote2Cap-DETR surpasses current state-of-the-arts by 11.13% and 7.11% in CIDEr@0.5IoU, respectively. Codes will be released soon.

        ----

        ## [1059] Improving Table Structure Recognition with Visual-Alignment Sequential Coordinate Modeling

        **Authors**: *Yongshuai Huang, Ning Lu, Dapeng Chen, Yibo Li, Zecheng Xie, Shenggao Zhu, Liangcai Gao, Wei Peng*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01071](https://doi.org/10.1109/CVPR52729.2023.01071)

        **Abstract**:

        Table structure recognition aims to extract the logical and physical structure of unstructured table images into a machine-readable format. The latest end-to-end image-to-text approaches simultaneously predict the two structures by two decoders, where the prediction of the physical structure (the bounding boxes of the cells) is based on the representation of the logical structure. However, the previous methods struggle with imprecise bounding boxes as the logical representation lacks local visual information. To address this issue, we propose an end-to-end sequential modeling framework for table structure recognition called VAST. It contains a novel coordinate sequence decoder triggered by the representation of the non-empty cell from the logical structure decoder. In the coordinate sequence decoder, we model the bounding box coordinates as a language sequence, where the left, top, right and bottom coordinates are decoded sequentially to leverage the inter-coordinate dependency. Furthermore, we propose an auxiliary visual-alignment loss to enforce the logical representation of the non-empty cells to contain more local visual details, which helps produce better cell bounding boxes. Extensive experiments demonstrate that our proposed method can achieve state-of-the-art results in both logical and physical structure recognition. The ablation study also validates that the proposed coordinate sequence decoder and the visual-alignment loss are the keys to the success of our method.

        ----

        ## [1060] Region-Aware Pretraining for Open-Vocabulary Object Detection with Vision Transformers

        **Authors**: *Dahun Kim, Anelia Angelova, Weicheng Kuo*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01072](https://doi.org/10.1109/CVPR52729.2023.01072)

        **Abstract**:

        We present Region-aware Open-vocabulary Vision Transformers (RO-ViT) - a contrastive image-text pretraining recipe to bridge the gap between image-level pretraining and open-vocabulary object detection. At the pretraining phase, we propose to randomly crop and resize regions of positional embeddings instead of using the whole image positional embeddings. This better matches the use of positional embeddings at region-level in the detection finetuning phase. In addition, we replace the common softmax cross entropy loss in contrastive learning with focal loss to better learn the informative yet difficult examples. Finally, we leverage recent advances in novel object proposals to improve open-vocabulary detection finetuning. We evaluate our full model on the LVIS and COCO open-vocabulary detection benchmarks and zero-shot transfer. RO-ViT achieves a state-of-the-art 32.1 AP
<inf xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">r</inf>
 on LVIS, surpassing the best existing approach by +5.8 points in addition to competitive zero-shot transfer detection. Surprisingly, RO-ViT improves the image-level representation as well and achieves the state of the art on 9 out of 12 metrics on COCO and Flickr image-text retrieval benchmarks, outperforming competitive approaches with larger models.

        ----

        ## [1061] Mobile User Interface Element Detection Via Adaptively Prompt Tuning

        **Authors**: *Zhangxuan Gu, Zhuoer Xu, Haoxing Chen, Jun Lan, Changhua Meng, Weiqiang Wang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01073](https://doi.org/10.1109/CVPR52729.2023.01073)

        **Abstract**:

        Recent object detection approaches rely on pretrained vision-language models for image-text alignment. However, they fail to detect the Mobile User Interface (MUI) element since it contains additional OCR information, which describes its content and function but is often ignored. In this paper, we develop a new MUI element detection dataset named MUI-zh and propose an Adaptively Prompt Tuning (APT) module to take advantage of discriminating OCR information. APT is a lightweight and effective module to jointly optimize category prompts across different modalities. For every element, APT uniformly encodes its visual features and OCR descriptions to dynamically adjust the representation of frozen category prompts. We evaluate the effectiveness of our plug-and-play APT upon several existing CLIP-based detectors for both standard and open-vocabulary MUI element detection. Extensive experiments show that our method achieves considerable improvements on two datasets. The datasets is available at github.com/antmachineintelligence/MUI-zh.

        ----

        ## [1062] Learning to Generate Text-Grounded Mask for Open-World Semantic Segmentation from Only Image-Text Pairs

        **Authors**: *Junbum Cha, Jonghwan Mun, Byungseok Roh*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01074](https://doi.org/10.1109/CVPR52729.2023.01074)

        **Abstract**:

        We tackle open-world semantic segmentation, which aims at learning to segment arbitrary visual concepts in images, by using only image-text pairs without dense annotations. Existing open-world segmentation methods have shown impressive advances by employing contrastive learning (CL) to learn diverse visual concepts and transferring the learned image-level understanding to the segmentation task. However, these CL-based methods suffer from a train-test discrepancy, since it only considers image-text alignment during training, whereas segmentation requires region-text alignment during testing. In this paper, we proposed a novel Text-grounded Contrastive Learning (TCL) framework that enables a model to directly learn region-text alignment. Our method generates a segmentation mask for a given text, extracts text-grounded image embedding from the masked region, and aligns it with text embedding via TCL. By learning region-text alignment directly, our framework encourages a model to directly improve the quality of generated segmentation masks. In addition, for a rigorous and fair comparison, we present a unified evaluation protocol with widely used 8 semantic segmentation datasets. TCL achieves state-of-the-art zero-shot segmentation performances with large margins in all datasets. Code is available at https://github.com/kakaobrain/tcl.

        ----

        ## [1063] ZegCLIP: Towards Adapting CLIP for Zero-shot Semantic Segmentation

        **Authors**: *Ziqin Zhou, Yinjie Lei, Bowen Zhang, Lingqiao Liu, Yifan Liu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01075](https://doi.org/10.1109/CVPR52729.2023.01075)

        **Abstract**:

        Recently, CLIP has been applied to pixel-level zero-shot learning tasks via a two-stage scheme. The general idea is to first generate class-agnostic region proposals and then feed the cropped proposal regions to CLIP to utilize its image-level zero-shot classification capability. While effective, such a scheme requires two image encoders, one for proposal generation and one for CLIP, leading to a complicated pipeline and high computational cost. In this work, we pursue a simpler-and-efficient one-stage solution that directly extends CLIP's zero-shot prediction capability from image to pixel level. Our investigation starts with a straightforward extension as our baseline that generates semantic masks by comparing the similarity between text and patch embeddings extracted from CLIP. However, such a paradigm could heavily overfit the seen classes and fail to generalize to unseen classes. To handle this issue, we propose three simple-but-effective designs and figure out that they can significantly retain the inherent zero-shot capacity of CLIP and improve pixel-level generalization ability. Incorporating those modifications leads to an efficient zero-shot semantic segmentation system called ZegCLIP. Through extensive experiments on three public benchmarks, ZegCLIP demonstrates superior performance, outperforming the state-of-the-art methods by a large margin under both “inductive” and “transductive” zero-shot settings. In addition, compared with the two-stage method, our one-stage ZegCLIP achieves a speedup of about 5 times faster during inference. We release the code at https://github.com/ZiqinZhou66/ZegCLIP.git.

        ----

        ## [1064] Object-Aware Distillation Pyramid for Open-Vocabulary Object Detection

        **Authors**: *Luting Wang, Yi Liu, Penghui Du, Zihan Ding, Yue Liao, Qiaosong Qi, Biaolong Chen, Si Liu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01076](https://doi.org/10.1109/CVPR52729.2023.01076)

        **Abstract**:

        Open-vocabulary object detection aims to provide object detectors trained on a fixed set of object categories with the generalizability to detect objects described by arbitrary text queries. Previous methods adopt knowledge distillation to extract knowledge from Pretrained Vision-and-Language Models (PVLMs) and transfer it to detectors. However, due to the non-adaptive proposal cropping and single-level feature mimicking processes, they suffer from information destruction during knowledge extraction and inefficient knowledge transfer. To remedy these limitations, we propose an Object-Aware Distillation Pyramid (OADP) framework, including an Object-Aware Knowledge Extraction (OAKE) module and a Distillation Pyramid (DP) mechanism. When extracting object knowledge from PVLMs, the former adaptively transforms object proposals and adopts object-aware mask attention to obtain precise and complete knowledge of objects. The latter introduces global and block distillation for more comprehensive knowledge transfer to compensate for the missing relation information in object distillation. Extensive experiments show that our method achieves significant improvement compared to current methods. Especially on the MS-COCO dataset, our OADP framework reaches 35.6 mAP
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">N</sup>
<inf xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">50</inf>
, surpassing the current state-of-the-art method by 3.3 mAP
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">N</sup>
<inf xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">50</inf>
. Code is released at https://github.com/LutingWang/OADP.

        ----

        ## [1065] Learning Conditional Attributes for Compositional Zero-Shot Learning

        **Authors**: *Qingsheng Wang, Lingqiao Liu, Chenchen Jing, Hao Chen, Guoqiang Liang, Peng Wang, Chunhua Shen*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01077](https://doi.org/10.1109/CVPR52729.2023.01077)

        **Abstract**:

        Compositional Zero-Shot Learning (CZSL) aims to train models to recognize novel compositional concepts based on learned concepts such as attribute-object combinations. One of the challenges is to model attributes interacted with different objects, e.g., the attribute “wet” in “wet apple” and “wet cat” is different. As a solution, we provide analysis and argue that attributes are conditioned on the recognized object and input image and explore learning conditional attribute embeddings by a proposed attribute learning framework containing an attribute hyper learner and an attribute base learner. By encoding conditional attributes, our model enables to generate flexible attribute embeddings for generalization from seen to unseen compositions. Experiments on CZSL benchmarks, including the more challenging C-GQA dataset, demonstrate better performances compared with other state-of-the-art approaches and validate the importance of learning conditional attributes. Code
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">‡</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
Gllee:https://gitee.com/wqshmzh/canet-czsl is available at https://github.com/wqshmzh/CANet-CZSL.

        ----

        ## [1066] CLIP-S4: Language-Guided Self-Supervised Semantic Segmentation

        **Authors**: *Wenbin He, Suphanut Jamonnak, Liang Gou, Liu Ren*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01078](https://doi.org/10.1109/CVPR52729.2023.01078)

        **Abstract**:

        Existing semantic segmentation approaches are often limited by costly pixel-wise annotations and predefined classes. In this work, we present CLIP-S
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">4</sup>
 that leverages self-supervised pixel representation learning and vision-language models to enable various semantic segmentation tasks (e.g., unsupervised, transfer learning, language-driven segmentation) without any human annotations and unknown class information. We first learn pixel embeddings with pixel-segment contrastive learning from different augmented views of images. To further improve the pixel embeddings and enable language-driven semantic segmentation, we design two types of consistency guided by vision-language models: 1) embedding consistency, aligning our pixel embeddings to the joint feature space of a pre-trained vision-language model, CLIP [34]; and 2) semantic consistency, forcing our model to make the same predictions as CLIP over a set of carefully designed target classes with both known and unknown prototypes. Thus, CLIP-S
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">4</sup>
 enables a new task of class-free semantic segmentation where no unknown class information is needed during training. As a result, our approach shows consistent and substantial performance improvement over four popular benchmarks compared with the state-of-the-art unsupervised and language-driven semantic segmentation methods. More importantly, our method outperforms these methods on unknown class recognition by a large margin.

        ----

        ## [1067] StructVPR: Distill Structural Knowledge with Weighting Samples for Visual Place Recognition

        **Authors**: *Yanqing Shen, Sanping Zhou, Jingwen Fu, Ruotong Wang, Shitao Chen, Nanning Zheng*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01079](https://doi.org/10.1109/CVPR52729.2023.01079)

        **Abstract**:

        Visual place recognition (VPR) is usually considered as a specific image retrieval problem. Limited by existing training frameworks, most deep learning-based works cannot extract sufficiently stable global features from RGB images and rely on a time-consuming re-ranking step to exploit spatial structural information for better performance. In this paper, we propose StructVPR, a novel training architecture for VPR, to enhance structural knowledge in RGB global features and thus improve feature stability in a constantly changing environment. Specifically, StructVPR uses segmentation images as a more definitive source of structural knowledge input into a CNN network and applies knowledge distillation to avoid online segmentation and inference of seg-branch in testing. Considering that not all samples contain high-quality and helpful knowledge, and some even hurt the performance of distillation, we partition samples and weigh each sample's distillation loss to enhance the expected knowledge precisely. Finally, StructVPR achieves impressive performance on several benchmarks using only global retrieval and even outperforms many two-stage approaches by a large margin. After adding additional re-ranking, ours achieves state-of-the-art performance while maintaining a low computational cost.

        ----

        ## [1068] UniDAformer: Unified Domain Adaptive Panoptic Segmentation Transformer via Hierarchical Mask Calibration

        **Authors**: *Jingyi Zhang, Jiaxing Huang, Xiaoqin Zhang, Shijian Lu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01080](https://doi.org/10.1109/CVPR52729.2023.01080)

        **Abstract**:

        Domain adaptive panoptic segmentation aims to miti-gate data annotation challenge by leveraging off-the-shelf annotated data in one or multiple related source domains. However, existing studies employ two separate networks for instance segmentation and semantic segmentation which lead to excessive network parameters as well as complicated and computationally intensive training and inference processes. We design UniDAformer, a unified domain adaptive panoptic segmentation transformer that is simple but can achieve domain adaptive instance segmentation and semantic segmentation simultaneously within a single network. UniDAformer introduces Hierarchical Mask Calibration (HMC) that rectifies inaccurate predictions at the level of regions, superpixels and pixels via online self-training on the fly. It has three unique features: 1) it enables unified domain adaptive panoptic adaptation; 2) it mitigates false predictions and improves domain adaptive panoptic segmentation effectively; 3) it is end-to-end trainable with a much simpler training and inference pipeline. Exten-sive experiments over multiple public benchmarks show that UniDAformer achieves superior domain adaptive panoptic segmentation as compared with the state-of-the-art.

        ----

        ## [1069] Primitive Generation and Semantic-Related Alignment for Universal Zero-Shot Segmentation

        **Authors**: *Shuting He, Henghui Ding, Wei Jiang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01081](https://doi.org/10.1109/CVPR52729.2023.01081)

        **Abstract**:

        We study universal zero-shot segmentation in this work to achieve panoptic, instance, and semantic segmentation for novel categories without any training samples. Such zero-shot segmentation ability relies on inter-class relationships in semantic space to transfer the visual knowledge learned from seen categories to unseen ones. Thus, it is desired to well bridge semantic-visual spaces and apply the semantic relationships to visual feature learning. We introduce a generative model to synthesize features for unseen categories, which links semantic and visual spaces as well as address the issue of lack of unseen training data. Furthermore, to mitigate the domain gap between semantic and visual spaces, firstly, we enhance the vanilla generator with learned primitives, each of which contains fine-grained attributes related to categories, and synthesize unseen features by selectively assembling these primitives. Secondly, we propose to disentangle the visual feature into the semantic-related part and the semantic-unrelated part that contains useful visual classification clues but is less relevant to semantic representation. The inter-class relationships of semantic-related visual features are then required to be aligned with those in semantic space, thereby transferring semantic knowledge to visual feature learning. The proposed approach achieves impressively state-of-the-art performance on zero-shot panoptic segmentation, instance segmentation, and semantic segmentation.

        ----

        ## [1070] Inferring and Leveraging Parts from Object Shape for Improving Semantic Image Synthesis

        **Authors**: *Yuxiang Wei, Zhilong Ji, Xiaohe Wu, Jinfeng Bai, Lei Zhang, Wangmeng Zuo*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01082](https://doi.org/10.1109/CVPR52729.2023.01082)

        **Abstract**:

        Despite the progress in semantic image synthesis, it remains a challenging problem to generate photo-realistic parts from input semantic map. Integrating part segmentation map can undoubtedly benefit image synthesis, but is bothersome and inconvenient to be provided by users. To improve part synthesis, this paper presents to infer Parts from Object ShapE (iPOSE) and leverage it for improving semantic image synthesis. However, albeit several part segmentation datasets are available, part annotations are still not provided for many object categories in semantic image synthesis. To circumvent it, we resort to few-shot regime to learn a PartNet for predicting the object part map with the guidance of pre-defined support part maps. PartNet can be readily generalized to handle a new object category when a small number (e.g., 3) of support part maps for this category are provided. Furthermore, part semantic modulation is presented to incorporate both inferred part map and semantic map for image synthesis. Experiments show that our iPOSE not only generates objects with rich part details, but also enables to control the image synthesis flexibly. And our iPOSE performs favorably against the state-of-the-art methods in terms of quantitative and qualitative evaluation. Our code will be publicly available at https://github.com/csyxwei/iPOSE.

        ----

        ## [1071] Compositor: Bottom-Up Clustering and Compositing for Robust Part and Object Segmentation

        **Authors**: *Ju He, Jieneng Chen, Ming-Xian Lin, Qihang Yu, Alan L. Yuille*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01083](https://doi.org/10.1109/CVPR52729.2023.01083)

        **Abstract**:

        In this work, we present a robust approach for joint part and object segmentation. Specifically, we reformulate object and part segmentation as an optimization problem and build a hierarchical feature representation including pixel, part, and object-level embeddings to solve it in a bottom-up clustering manner. Pixels are grouped into several clusters where the part-level embeddings serve as cluster centers. Afterwards, object masks are obtained by compositing the part proposals. This bottom-up interaction is shown to be effective in integrating information from lower semantic levels to higher semantic levels. Based on that, our novel approach Compositor produces part and object segmentation masks simultaneously while improving the mask quality. Compositor achieves state-of-the-art performance on PartImageNet and Pascal-Part by outperforming previous methods by around 0.9% and 1.3% on PartImageNet, 0.4% and 1.7% on Pascal-Part in terms of part and object mIoU and demonstrates better robustness against occlusion by around 4.4% and 7.1% on part and object respectively.

        ----

        ## [1072] A Strong Baseline for Generalized Few-Shot Semantic Segmentation

        **Authors**: *Sina Hajimiri, Malik Boudiaf, Ismail Ben Ayed, Jose Dolz*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01084](https://doi.org/10.1109/CVPR52729.2023.01084)

        **Abstract**:

        This paper introduces a generalized few-shot segmentation framework with a straightforward training process and an easy-to-optimize inference phase. In particular, we propose a simple yet effective model based on the well-known InfoMax principle, where the Mutual Information (MI) between the learned feature representations and their corresponding predictions is maximized. In addition, the terms derived from our MI-based formulation are coupled with a knowledge distillation term to retain the knowledge on base classes. With a simple training process, our inference model can be applied on top of any segmentation network trained on base classes. The proposed inference yields substantial improvements on the popular few-shot segmentation benchmarks, PASCAL-5
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">i</sup>
 and COCO-20
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">i</sup>
. Particularly, for novel classes, the improvement gains range from 7% to 26% (PASCAL-5
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">i</sup>
) and from 3% to 12% (COCO-20
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">i</sup>
) in the 1-shot and 5-shot scenarios, respectively. Furthermore, we propose a more challenging setting, where performance gaps are further exacerbated. Our code is publicly available at https://github.com/sinahmr/DIaM.

        ----

        ## [1073] DynaMask: Dynamic Mask Selection for Instance Segmentation

        **Authors**: *Ruihuang Li, Chenhang He, Shuai Li, Yabin Zhang, Lei Zhang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01085](https://doi.org/10.1109/CVPR52729.2023.01085)

        **Abstract**:

        The representative instance segmentation methods mostly segment different object instances with a mask of the fixed resolution, e.g., 28 × 28 grid. However, a low-resolution mask loses rich details, while a high-resolution mask incurs quadratic computation overhead. It is a challenging task to predict the optimal binary mask for each instance. In this paper, we propose to dynamically select suitable masks for different object proposals. First, a dual-level Feature Pyramid Network (FPN) with adaptive feature aggregation is developed to gradually increase the mask grid resolution, ensuring high-quality segmentation of objects. Specifically, an efficient region-level top-down path (r-FPN) is introduced to incorporate complementary contextual and detailed information from different stages of image-level FPN (i-FPN). Then, to alleviate the increase of computation and memory costs caused by using large masks, we develop a Mask Switch Module (MSM) with negligible computational cost to select the most suitable mask resolution for each instance, achieving high efficiency while maintaining high segmentation accuracy. Without bells and whistles, the proposed method, namely DynaMask, brings consistent and noticeable performance improvements over other state-of-the-arts at a moderate computation overhead. The source code: https://github.com/lslrh/DynaMask.

        ----

        ## [1074] Focus On Details: Online Multi-Object Tracking with Diverse Fine-Grained Representation

        **Authors**: *Hao Ren, Shoudong Han, Huilin Ding, Ziwen Zhang, Hongwei Wang, Faquan Wang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01086](https://doi.org/10.1109/CVPR52729.2023.01086)

        **Abstract**:

        Discriminative representation is essential to keep a unique identifier for each target in Multiple object tracking (MOT). Some recent MOT methods extract features of the bounding box region or the center point as identity embeddings. However, when targets are occluded, these coarse-grained global representations become unreliable. To this end, we propose exploring diverse fine-grained representation, which describes appearance comprehensively from global and local perspectives. This fine-grained represen-tation requires high feature resolution and precise semantic information. To effectively alleviate the semantic misalignment caused by indiscriminate contextual information aggregation, Flow Alignment FPN (FAFPN) is proposed for multi-scale feature alignment aggregation. It generates semantic flow among feature maps from different resolutions to transform their pixel positions. Furthermore, we present a Multi-head Part Mask Generator (MPMG) to extract fine-grained representation based on the aligned feature maps. Multiple parallel branches of MPMG allow it to focus on different parts of targets to generate local masks without label supervision. The diverse details in target masks facilitate fine-grained representation. Eventually, benefiting from a Shuffle-Group Sampling (SGS) training strategy with positive and negative samples balanced, we achieve state-of-the-art performance on MOT17 and MOT20 test sets. Even on DanceTrack, where the appearance of targets is extremely similar, our method significantly outperforms Byte-Track by 5.0% on HOTA and 5.6% on IDF1. Extensive experiments have proved that diverse fine- grained representation makes Re-ID great again in MOT.

        ----

        ## [1075] Dynamic Focus-aware Positional Queries for Semantic Segmentation

        **Authors**: *Haoyu He, Jianfei Cai, Zizheng Pan, Jing Liu, Jing Zhang, Dacheng Tao, Bohan Zhuang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01087](https://doi.org/10.1109/CVPR52729.2023.01087)

        **Abstract**:

        The DETR-like segmentors have underpinned the most recent breakthroughs in semantic segmentation, which end-to-end train a set of queries representing the class prototypes or target segments. Recently, masked attention [8] is proposed to restrict each query to only attend to the foreground regions predicted by the preceding decoder block for easier optimization. Although promising, it relies on the learnable parameterized positional queries which tend to encode the dataset statistics, leading to inaccurate localization for distinct individual queries. In this paper, we propose a simple yet effective query design for semantic segmentation termed Dynamic Focus-aware Positional Queries (DFPQ), which dynamically generates positional queries conditioned on the cross-attention scores from the preceding decoder block and the positional encodings for the corresponding image features, simultaneously. Therefore, our DFPQ preserves rich localization information for the target segments and provides accurate and fine-grained positional priors. In addition, we propose to efficiently deal with high-resolution cross-attention by only aggregating the con-textual tokens based on the low-resolution cross-attention scores to perform local relation aggregation. Extensive experiments on ADE20K and Cityscapes show that with the two modifications on Mask2former, our framework achieves SOTA performance and outperforms Mask2former by clear margins of 1.1%, 1.9%, and 1.1% single-scale mIoU with ResNet-50, Swin-T, and Swin-B backbones on the ADE20K validation set, respectively. Source code is available at https://github.com/ziplab/FASeg.

        ----

        ## [1076] Beyond mAP: Towards Better Evaluation of Instance Segmentation

        **Authors**: *Rohit Jena, Lukas Zhornyak, Nehal Doiphode, Pratik Chaudhari, Vivek Buch, James C. Gee, Jianbo Shi*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01088](https://doi.org/10.1109/CVPR52729.2023.01088)

        **Abstract**:

        Correctness of instance segmentation constitutes counting the number of objects, correctly localizing all predictions and classifying each localized prediction. Average Precision is the de-facto metric used to measure all these constituents of segmentation. However, this metric does not penalize duplicate predictions in the high-recall range, and cannot distinguish instances that are localized correctly but categorized incorrectly. This weakness has inadvertently led to network designs that achieve significant gains in AP but also introduce a large number of false positives. We therefore cannot rely on AP to choose a model that provides an optimal tradeoff between false positives and high recall. To resolve this dilemma, we review alternative metrics in the literature and propose two new measures to explicitly measure the amount of both spatial and categorical duplicate predictions. We also propose a Semantic Sorting and NMS module to remove these duplicates based on a pixel occupancy matching scheme. Experiments show that modern segmentation networks have significant gains in AP, but also contain a considerable amount of duplicates. Our Semantic Sorting and NMS can be added as a plug-and-play module to mitigate hedged predictions and preserve AP.

        ----

        ## [1077] Learning Orthogonal Prototypes for Generalized Few-Shot Semantic Segmentation

        **Authors**: *Sun'ao Liu, Yiheng Zhang, Zhaofan Qiu, Hongtao Xie, Yongdong Zhang, Ting Yao*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01089](https://doi.org/10.1109/CVPR52729.2023.01089)

        **Abstract**:

        Generalized few-shot semantic segmentation (GFSS) distinguishes pixels of base and novel classes from the background simultaneously, conditioning on sufficient data of base classes and a few examples from novel class. A typical GFSS approach has two training phases: base class learning and novel class updating. Nevertheless, such a stand-alone updating process often compromises the well-learnt features and results in performance drop on base classes. In this paper, we propose a new idea of leveraging Projection onto Orthogonal Prototypes (POP), which updates features to identify novel classes without compromising base classes. POP builds a set of orthogonal prototypes, each of which represents a semantic class, and makes the prediction for each class separately based on the features projected onto its prototype. Technically, POP first learns prototypes on base data, and then extends the prototype set to novel classes. The orthogonal constraint of POP encourages the orthogonality between the learnt prototypes and thus mitigates the influence on base class features when generalizing to novel prototypes. Moreover, we capitalize on the residual of feature projection as the background representation to dynamically fit semantic shifting (i.e., background no longer includes the pixels of novel classes in updating phase). Extensive experiments on two benchmarks demonstrate that our POP achieves superior performances on novel classes without sacrificing much accuracy on base classes. Notably, POP outperforms the state-of-the-art fine-tuning by 3.93% overall mIoU on PASCAL-5
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">i</sup>
 in 5-shot scenario.

        ----

        ## [1078] Weakly Supervised Semantic Segmentation via Adversarial Learning of Classifier and Reconstructor

        **Authors**: *Hyeokjun Kweon, Sung-Hoon Yoon, Kuk-Jin Yoon*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01090](https://doi.org/10.1109/CVPR52729.2023.01090)

        **Abstract**:

        In Weakly Supervised Semantic Segmentation (WSSS), Class Activation Maps (CAMs) usually 1) do not cover the whole object and 2) be activated on irrelevant regions. To address the issues, we propose a novel WSSS framework via adversarial learning of a classifier and an image reconstructor. When an image is perfectly decomposed into class-wise segments, information (i.e., color or texture) of a single segment could not be inferred from the other segments. Therefore, inferability between the segments can represent the preciseness of segmentation. We quantify the inferability as a reconstruction quality of one segment from the other segments. If one segment could be reconstructed from the others, then the segment would be imprecise. To bring this idea into WSSS, we simultaneously train two models: a classifier generating CAMs that decompose an image into segments and a reconstructor that measures the inferability between the segments. As in GANs, while being alternatively trained in an adversarial manner, two networks provide positive feedback to each other. We verify the superiority of the proposed framework with extensive ablation studies. Our method achieves new state-of-the-art performances on both PAS-CAL VOC 2012 and MS COCO 2014. The code is available at https://github.com/sangrockEG/ACR.

        ----

        ## [1079] SemiCVT: Semi-Supervised Convolutional Vision Transformer for Semantic Segmentation

        **Authors**: *Huimin Huang, Shiao Xie, Lanfen Lin, Ruofeng Tong, Yen-Wei Chen, Yuexiang Li, Hong Wang, Yawen Huang, Yefeng Zheng*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01091](https://doi.org/10.1109/CVPR52729.2023.01091)

        **Abstract**:

        Semi-supervised learning improves data efficiency of deep models by leveraging unlabeled samples to alleviate the reliance on a large set of labeled samples. These successes concentrate on the pixel-wise consistency by using convolutional neural networks (CNNs) but fail to address both global learning capability and class-level features for unlabeled data. Recent works raise a new trend that Transformer achieves superior performance on the entire feature map in various tasks. In this paper, we unify the current dominant Mean-Teacher approaches by reconciling intra-model and inter-model properties for semi-supervised segmentation to produce a novel algorithm, SemiCVT, that absorbs the quintessence of CNNs and Transformer in a comprehensive way. Specifically, we first design a parallel CNN-Transformer architecture (CVT) with introducing an intra-model local-global interaction schema (LGI) in Fourier domain for full integration. The inter-model class-wise consistency is further presented to complement the class-level statistics of CNNs and Transformer in a cross-teaching manner. Extensive empirical evidence shows that SemiCVT yields consistent improvements over the state-of-the-art methods in two public benchmarks.

        ----

        ## [1080] Augmentation Matters: A Simple-Yet-Effective Approach to Semi-Supervised Semantic Segmentation

        **Authors**: *Zhen Zhao, Lihe Yang, Sifan Long, Jimin Pi, Luping Zhou, Jingdong Wang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01092](https://doi.org/10.1109/CVPR52729.2023.01092)

        **Abstract**:

        Recent studies on semi-supervised semantic segmentation (SSS) have seen fast progress. Despite their promising performance, current state-of-the-art methods tend to increasingly complex designs at the cost of introducing more network components and additional training procedures. Differently, in this work, we follow a standard teacher-student framework and propose AugSeg, a simple and clean approach that focuses mainly on data perturbations to boost the SSS performance. We argue that various data augmentations should be adjusted to better adapt to the semi-supervised scenarios instead of directly applying these techniques from supervised learning. Specifically, we adopt a simplified intensity-based augmentation that selects a random number of data transformations with uniformly sampling distortion strengths from a continuous space. Based on the estimated confidence of the model on different unlabeled samples, we also randomly inject labelled information to augment the unlabeled samples in an adaptive manner. Without bells and whistles, our simple AugSeg can readily achieve new state-of-the-art performance on SSS benchmarks under different partition protocols
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
Code and logs: https://github.com/zhenzhao/AugSeg..

        ----

        ## [1081] The Devil is in the Points: Weakly Semi-Supervised Instance Segmentation via Point-Guided Mask Representation

        **Authors**: *Beomyoung Kim, Joonhyun Jeong, Dongyoon Han, Sung Ju Hwang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01093](https://doi.org/10.1109/CVPR52729.2023.01093)

        **Abstract**:

        In this paper, we introduce a novel learning scheme named weakly semi-supervised instance segmentation (WS-SIS) with point labels for budget-efficient and high-performance instance segmentation. Namely, we consider a dataset setting consisting of a few fully-labeled images and a lot of point-labeled images. Motivated by the main challenge of semi-supervised approaches mainly derives from the trade-off between false-negative and false-positive instance proposals, we propose a method for WSSIS that can effectively leverage the budget-friendly point labels as a powerful weak supervision source to resolve the challenge. Furthermore, to deal with the hard case where the amount of fully-labeled data is extremely limited, we propose a MaskRefineNet that refines noise in rough masks. We conduct extensive experiments on COCO and BDD 100K datasets, and the proposed method achieves promising results comparable to those of the fully-supervised model, even with 50% of the fully labeled COCO data (38.8% vs. 39.7%). Moreover, when using as little as 5% of fully labeled COCO data, our method shows significantly superior performance over the state-of-the-art semi-supervised learning method (33.7% vs. 24.9%). The code is available at https://github.com/clovaai/PointWSSIS.

        ----

        ## [1082] Class-Incremental Exemplar Compression for Class-Incremental Learning

        **Authors**: *Zilin Luo, Yaoyao Liu, Bernt Schiele, Qianru Sun*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01094](https://doi.org/10.1109/CVPR52729.2023.01094)

        **Abstract**:

        Exemplar-based class-incremental learning (CIL) [36] finetunes the model with all samples of new classes but few-shot exemplars of old classes in each incremental phase, where the “few-shot” abides by the limited memory budget. In this paper, we break this “few-shot” limit based on a simple yet surprisingly effective idea: compressing exemplars by downsampling non-discriminative pixels and saving “many-shot” compressed exemplars in the memory. Without needing any manual annotation, we achieve this compression by generating 0–1 masks on discriminative pixels from class activation maps (CAM) [49]. We propose an adaptive mask generation model called class-incremental masking (CIM) to explicitly resolve two difficulties of using CAM: 1) transforming the heatmaps of CAM to 0–1 masks with an arbitrary threshold leads to a trade-off between the coverage on discriminative pixels and the quantity of exemplars, as the total memory is fixed; and 2) optimal thresholds vary for different object classes, which is particularly obvious in the dynamic environment of CIL. We optimize the CIM model alternatively with the conventional CIL model through a bilevel optimization problem [40]. We conduct extensive experiments on high-resolution CIL benchmarks including Food-101, ImageNet-100, and ImageNet-1000, and show that using the compressed exemplars by CIM can achieve a new state-of-the-art CIL accuracy, e.g., 4.8 percentage points higher than FOSTER [42] on 10-Phase ImageNet-1000. Our code is available at https://github.com/xfflzlICIM-CIL.

        ----

        ## [1083] Full or Weak Annotations? An Adaptive Strategy for Budget-Constrained Annotation Campaigns

        **Authors**: *Javier Gamazo Tejero, Martin S. Zinkernagel, Sebastian Wolf, Raphael Sznitman, Pablo Márquez-Neila*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01095](https://doi.org/10.1109/CVPR52729.2023.01095)

        **Abstract**:

        Annotating new datasets for machine learning tasks is tedious, time-consuming, and costly. For segmentation applications, the burden is particularly high as manual delin-eations of relevant image content are often extremely expensive or can only be done by experts with domain-specific knowledge. Thanks to developments in transfer learning and training with weak supervision, segmentation models can now also greatly benefit from annotations of different kinds. However, for any new domain application looking to use weak supervision, the dataset builder still needs to define a strategy to distribute full segmentation and other weak annotations. Doing so is challenging, however, as it is a priori unknown how to distribute an annotation budget for a given new dataset. To this end, we propose a novel approach to determine annotation strategies for segmentation datasets, whereby estimating what proportion of segmentation and classification annotations should be collected given a fixed budget. To do so, our method sequentially determines proportions of segmentation and classification annotations to collect for budget-fractions by modeling the expected improvement of the final segmentation model. We show in our experiments that our approach yields annotations that perform very close to the optimal for a number of different annotation budgets and datasets.

        ----

        ## [1084] Learning Common Rationale to Improve Self-Supervised Representation for Fine-Grained Visual Recognition Problems

        **Authors**: *Yangyang Shu, Anton van den Hengel, Lingqiao Liu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01096](https://doi.org/10.1109/CVPR52729.2023.01096)

        **Abstract**:

        Self-supervised learning (SSL) strategies have demonstrated remarkable performance in various recognition tasks. However, both our preliminary investigation and recent studies suggest that they may be less effective in learning representations for fine-grained visual recognition (FGVR) since many features helpful for optimizing SSL objectives are not suitable for characterizing the subtle differences in FGVR. To overcome this issue, we propose learning an additional screening mechanism to identify discriminative clues commonly seen across instances and classes, dubbed as common rationales in this paper. Intuitively, common rationales tend to correspond to the discriminative patterns from the key parts of foreground objects. We show that a common rationale detector can be learned by simply exploiting the GradCAM induced from the SSL objective without using any pre-trained object parts or saliency detectors, making it seamlessly to be integrated with the existing SSL process. Specifically, we fit the GradCAM with a branch with limited fitting capacity, which allows the branch to capture the common rationales and discard the less common discriminative patterns. At the test stage, the branch generates a set of spatial weights to selectively aggregate features representing an instance. Extensive experimental results on four visual tasks demonstrate that the proposed method can lead to a significant improvement in different evaluation settings.
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
The source code will be publicly available at:https://github.com/GANPerf/LCR

        ----

        ## [1085] Detection Hub: Unifying Object Detection Datasets via Query Adaptation on Language Embedding

        **Authors**: *Lingchen Meng, Xiyang Dai, Yinpeng Chen, Pengchuan Zhang, Dongdong Chen, Mengchen Liu, Jianfeng Wang, Zuxuan Wu, Lu Yuan, Yu-Gang Jiang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01097](https://doi.org/10.1109/CVPR52729.2023.01097)

        **Abstract**:

        Combining multiple datasets enables performance boost on many computer vision tasks. But similar trend has not been witnessed in object detection when combining multiple datasets due to two inconsistencies among detection datasets: taxonomy difference and domain gap. In this paper, we address these challenges by a new design (named Detection Hub) that is dataset-aware and category-aligned. It not only mitigates the dataset inconsistency but also provides coherent guidance for the detector to learn across multiple datasets. In particular, the dataset-aware design is achieved by learning a dataset embedding that is used to adapt object queries as well as convolutional kernels in detection heads. The categories across datasets are semantically aligned into a unified space by replacing one-hot category representations with word embedding and leveraging the semantic coherence of language embedding. Detection Hub fulfills the benefits of large data on object detection. Experiments demonstrate that joint training on multiple datasets achieves significant performance gains over training on each dataset alone. Detection Hub further achieves SoTA performance on UODB benchmark with wide variety of datasets.

        ----

        ## [1086] Self-supervised AutoFlow

        **Authors**: *Hsin-Ping Huang, Charles Herrmann, Junhwa Hur, Erika Lu, Kyle Sargent, Austin Stone, Ming-Hsuan Yang, Deqing Sun*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01098](https://doi.org/10.1109/CVPR52729.2023.01098)

        **Abstract**:

        Recently, AutoFlow has shown promising results on learning a training set for optical flow, but requires ground truth labels in the target domain to compute its search metric. Observing a strong correlation between the ground truth search metric and self-supervised losses, we introduce self-supervised AutoFlow to handle real-world videos without ground truth labels. Using self-supervised loss as the search metric, our self-supervised AutoFlow performs on par with AutoFlow on Sintel and KITTI where ground truth is available, and performs better on the real-world DAVIS dataset. We further explore using self-supervised AutoFlow in the (semi-)supervised setting and obtain competitive results against the state of the art.

        ----

        ## [1087] DETR with Additional Global Aggregation for Cross-domain Weakly Supervised Object Detection

        **Authors**: *Zongheng Tang, Yifan Sun, Si Liu, Yi Yang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01099](https://doi.org/10.1109/CVPR52729.2023.01099)

        **Abstract**:

        This paper presents a DETR-based method for cross-domain weakly supervised object detection (CDWSOD), aiming at adapting the detector from source to target domain through weak supervision. We think DETR has strong potential for CDWSOD due to an insight: the encoder and the decoder in DETR are both based on the attention mechanism and are thus capable of aggregating semantics across the entire image. The aggregation results, i.e., image-level predictions, can naturally exploit the weak supervision for domain alignment. Such motivated, we propose DETR with additional Global Aggregation (DETR-GA), a CDWSOD detector that simultaneously makes “instance-level + image-level” predictions and utilizes “strong + weak” supervisions. The key point of DETR-GA is very simple: for the encoder / decoder, we respectively add multiple class queries / a foreground query to aggregate the semantics into image-level predictions. Our query-based aggregation has two advantages. First, in the encoder, the weakly-supervised class queries are capable of roughly locating the corresponding positions and excluding the distraction from non-relevant regions. Second, through our design, the object queries and the foreground query in the decoder share consensus on the class semantics, therefore making the strong and weak supervision mutually benefit each other for domain alignment. Extensive experiments on four popular cross-domain benchmarks show that DETR-GA significantly improves cross-domain detection accuracy (e.g., 29.0% → 79.4% mAP on PASCAL VOC → Clipart
<inf xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">all</inf>
 dataset) and advances the states of the art.

        ----

        ## [1088] Detecting Everything in the Open World: Towards Universal Object Detection

        **Authors**: *Zhenyu Wang, Yali Li, Xi Chen, Ser-Nam Lim, Antonio Torralba, Hengshuang Zhao, Shengjin Wang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01100](https://doi.org/10.1109/CVPR52729.2023.01100)

        **Abstract**:

        In this paper, we formally address universal object detection, which aims to detect every scene and predict every category. The dependence on human annotations, the limited visual information, and the novel categories in the open world severely restrict the universality of traditional detectors. We propose UniDetector, a universal object detector that has the ability to recognize enormous categories in the open world. The critical points for the universality of UniDetector are: 1) it leverages images of multiple sources and heterogeneous label spaces for training through the alignment of image and text spaces, which guarantees sufficient information for universal representations. 2) it generalizes to the open world easily while keeping the balance between seen and unseen classes, thanks to abundant information from both vision and language modalities. 3) it further promotes the generalization ability to novel categories through our proposed decoupling training manner and probability calibration. These contributions allow UniDetector to detect over 7k categories, the largest measurable category size so far, with only about 500 classes participating in training. Our UniDetector behaves the strong zero-shot generalization ability on largevocabulary datasets - it surpasses the traditional supervised baselines by more than 4% on average without seeing any corresponding images. On 13 public detection datasets with various scenes, UniDetector also achieves state-of-the-art performance with only a 3% amount of training data. 
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
Codes are available at https://github.com/zhenyuw16/UniDetector.

        ----

        ## [1089] PROB: Probabilistic Objectness for Open World Object Detection

        **Authors**: *Orr Zohar, Kuan-Chieh Wang, Serena Yeung*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01101](https://doi.org/10.1109/CVPR52729.2023.01101)

        **Abstract**:

        Open World Object Detection (OWOD) is a new and challenging computer vision task that bridges the gap between classic object detection (OD) benchmarks and object detection in the real world. In addition to detecting and classifying seen/labeled objects, OWOD algorithms are expected to detect novel/unknown objects - which can be classified and incrementally learned. In standard OD, object proposals not overlapping with a labeled object are automatically classified as background. Therefore, simply applying OD methods to OWOD fails as unknown objects would be predicted as background. The challenge of detecting unknown objects stems from the lack of supervision in distinguishing unknown objects and background object proposals. Previous OWOD methods have attempted to overcome this issue by generating supervision using pseudo-labeling - however, unknown object detection has remained low. Probabilistic/generative models may provide a solution for this challenge. Herein, we introduce a novel probabilistic framework for objectness estimation, where we alternate between probability distribution estimation and objectness likelihood maximization of known objects in the embedded feature space - ultimately allowing us to estimate the objectness probability of different proposals. The resulting Probabilistic Objectness transformer-based open-world detector, PROB, integrates our framework into traditional object detection models, adapting them for the open-world setting. Comprehensive experiments on OWOD benchmarks show that PROB outperforms all existing OWOD methods in both unknown object detection (~ 2 × unknown recall) and known object detection (~ 10% mAP). Our code is available at https://github.com/orrzohar/PROB.

        ----

        ## [1090] Annealing-based Label-Transfer Learning for Open World Object Detection

        **Authors**: *Yuqing Ma, Hainan Li, Zhange Zhang, Jinyang Guo, Shanghang Zhang, Ruihao Gong, Xianglong Liu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01102](https://doi.org/10.1109/CVPR52729.2023.01102)

        **Abstract**:

        Open world object detection (OWOD) has attracted extensive attention due to its practicability in the real world. Previous OWOD works manually designed unknown-discover strategies to select unknown proposals from the background, suffering from uncertainties without appropriate priors. In this paper, we claim the learning of object detection could be seen as an object-level feature-entanglement process, where unknown traits are propagated to the known proposals through convolutional operations and could be distilled to benefit unknown recognition without manual selection. Therefore, we propose a simple yet effective Annealing-based Label-Transfer framework, which sufficiently explores the known proposals to alleviate the uncertainties. Specifically, a Label-Transfer Learning paradigm is introduced to decouple the known and unknown features, while a Sawtooth Annealing Scheduling strategy is further employed to rebuild the decision boundaries of the known and unknown classes, thus promoting both known and unknown recognition. Moreover, previous OWOD works neglected the trade-off of known and unknown performance, and we thus introduce a metric called Equilibrium Index to comprehensively evaluate the effectiveness of the OWOD models. To the best of our knowledge, this is the first OWOD work without manual unknown selection. Extensive experiments conducted on the common-used benchmark validate that our model achieves superior detection performance (200% unknown mAP improvement with the even higher known detection performance) compared to other state-of-the-art methods. Our code is available at https://github.com/DIG-Beihang/ALLOW.git.

        ----

        ## [1091] Learning Transformation-Predictive Representations for Detection and Description of Local Features

        **Authors**: *Zihao Wang, Chunxu Wu, Yifei Yang, Zhen Li*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01103](https://doi.org/10.1109/CVPR52729.2023.01103)

        **Abstract**:

        The task of key-points detection and description is to estimate the stable location and discriminative representation of local features, which is a fundamental task in visual applications. However, either the rough hard positive or negative labels generated from one-to-one correspondences among images may bring indistinguishable samples, like false positives or negatives, which acts as inconsistent supervision. Such resultant false samples mixed with hard samples prevent neural networks from learning descriptions for more accurate matching. To tackle this challenge, we propose to learn the transformation-predictive representations with self-supervised contrastive learning. We maximize the similarity between corresponding views of the same 3D point (landmark) by using none of the negative sample pairs and avoiding collapsing solutions. Furthermore, we adopt self-supervised generation learning and curriculum learning to soften the hard positive labels into soft continuous targets. The aggressively updated soft labels contribute to overcoming the training bottleneck (derived from the label noise of false positives) and facilitating the model training under a stronger transformation paradigm. Our self-supervised training pipeline greatly decreases the computation load and memory usage, and outperforms the sota on the standard image matching benchmarks by noticeable margins, demonstrating excellent generalization capability on multiple downstream tasks.

        ----

        ## [1092] Bridging Precision and Confidence: A Train-Time Loss for Calibrating Object Detection

        **Authors**: *Muhammad Akhtar Munir, Muhammad Haris Khan, Salman H. Khan, Fahad Shahbaz Khan*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01104](https://doi.org/10.1109/CVPR52729.2023.01104)

        **Abstract**:

        Deep neural networks (DNNs) have enabled astounding progress in several vision-based problems. Despite showing high predictive accuracy, recently, several works have revealed that they tend to provide overconfident predictions and thus are poorly calibrated. The majority of the works addressing the miscalibration of DNNs fall under the scope of classification and consider only in-domain predictions. However, there is little to no progress in studying the calibration of DNN-based object detection models, which are central to many vision-based safety-critical applications. In this paper, inspired by the train-time calibration methods, we propose a novel auxiliary loss formulation that explicitly aims to align the class confidence of bounding boxes with the accurateness of predictions (i.e. precision). Since the original formulation of our loss depends on the counts of true positives and false positives in a mini-batch, we develop a differentiable proxy of our loss that can be used during training with other application-specific loss functions. We perform extensive experiments on challenging in-domain and out-domain scenarios with six benchmark datasets including MS-COCO, Cityscapes, Sim10k, and BDD100k. Our results reveal that our train-time loss surpasses strong calibration baselines in reducing calibration error for both in and out-domain scenarios. Our source code and pre-trained models are available at https://github.com/akhtarvision/bpc_calibration

        ----

        ## [1093] 2PCNet: Two-Phase Consistency Training for Day-to-Night Unsupervised Domain Adaptive Object Detection

        **Authors**: *Mikhail Kennerley, Jian-Gang Wang, Bharadwaj Veeravalli, Robby T. Tan*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01105](https://doi.org/10.1109/CVPR52729.2023.01105)

        **Abstract**:

        Object detection at night is a challenging problem due to the absence of night image annotations. Despite several domain adaptation methods, achieving high-precision results remains an issue. False-positive error propagation is still observed in methods using the well-established student-teacher framework, particularly for small-scale and low-light objects. This paper proposes a two-phase consistency unsupervised domain adaptation network, 2PCNet, to address these issues. The network employs high-confidence bounding-box predictions from the teacher in the first phase and appends them to the student's region proposals for the teacher to re-evaluate in the second phase, resulting in a combination of high and low confidence pseudolabels. The night images and pseudo-labels are scaled-down before being used as input to the student, providing stronger small-scale pseudo-labels. To address errors that arise from low-light regions and other night-related attributes in images, we propose a night-specific augmentation pipeline called NightAug. This pipeline involves applying random augmentations, such as glare, blur, and noise, to daytime images. Experiments on publicly available datasets demonstrate that our method achieves superior results to state-of-the-art methods by 20%, and to supervised models trained directly on the target data.
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
www.github.com/mecarill/2pcnet

        ----

        ## [1094] Zero-Shot Generative Model Adaptation via Image-Specific Prompt Learning

        **Authors**: *Jiayi Guo, Chaofei Wang, You Wu, Eric J. Zhang, Kai Wang, Xingqian Xu, Shiji Song, Humphrey Shi, Gao Huang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01106](https://doi.org/10.1109/CVPR52729.2023.01106)

        **Abstract**:

        Recently, CLIP-guided image synthesis has shown appealing performance on adapting a pre-trained source-domain generator to an unseen target domain. It does not require any target-domain samples but only the textual domain labels. The training is highly efficient, e.g., a few minutes. However, existing methods still have some limitations in the quality of generated images and may suffer from the mode collapse issue. A key reason is that a fixed adaptation direction is applied for all cross-domain image pairs, which leads to identical supervision signals. To address this issue, we propose an Image-specific Prompt Learning (IPL) method, which learns specific prompt vectors for each source-domain image. This produces a more precise adaptation direction for every cross-domain image pair, endowing the target-domain generator with greatly enhanced flexibility. Qualitative and quantitative evaluations on various domains demonstrate that IPL effectively improves the quality and diversity of synthesized images and alleviates the mode collapse. Moreover, IPL is independent of the structure of the generative model, such as generative adversarial networks or diffusion models. Code is available at https://github.com/Picsart-AI-Research/IPL-Zero-Shot-Generative-Model-Adaptation.

        ----

        ## [1095] AutoLabel: CLIP-based framework for Open-Set Video Domain Adaptation

        **Authors**: *Giacomo Zara, Subhankar Roy, Paolo Rota, Elisa Ricci*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01107](https://doi.org/10.1109/CVPR52729.2023.01107)

        **Abstract**:

        Open-set Unsupervised Video Domain Adaptation (OU-VDA) deals with the task of adapting an action recognition model from a labelled source domain to an unlabelled target domain that contains “target-private” categories, which are present in the target but absent in the source. In this work we deviate from the prior work of training a specialized open-set classifier or weighted adversarial learning by proposing to use pre-trained Language and Vision Models (CLIP). The CLIP is well suited for OUVDA due to its rich representation and the zero-shot recognition capabilities. However, rejecting target-private instances with the CLIP's zero-shot protocol requires oracle knowledge about the target-private label names. To circumvent the impossibility of the knowledge of label names, we propose AutoLabel that automatically discovers and generates object-centric compositional candidate target-private class names. Despite its simplicity, we show that CLIP when equipped with AutoLabel can satisfactorily reject the target-private instances, thereby facilitating better alignment between the shared classes of the two domains. The code is available
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
https://github.com/gzaraunitn/autolabel.

        ----

        ## [1096] Bidirectional Copy-Paste for Semi-Supervised Medical Image Segmentation

        **Authors**: *Yunhao Bai, Duowen Chen, Qingli Li, Wei Shen, Yan Wang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01108](https://doi.org/10.1109/CVPR52729.2023.01108)

        **Abstract**:

        In semi-supervised medical image segmentation, there exist empirical mismatch problems between labeled and un-labeled data distribution. The knowledge learned from the labeled data may be largely discarded if treating labeled and unlabeled data separately or in an inconsistent manner. We propose a straightforward method for alleviating the problem-copy-pasting labeled and unlabeled data bidirectionally, in a simple Mean Teacher architecture. The method encourages unlabeled data to learn comprehensive common semantics from the labeled data in both inward and outward directions. More importantly, the consistent learning procedure for labeled and unlabeled data can largely reduce the empirical distribution gap. In detail, we copy-paste a random crop from a labeled image (foreground) onto an unlabeled image (background) and an unlabeled image (foreground) onto a labeled image (background), respectively. The two mixed images are fed into a Student network and supervised by the mixed supervisory signals of pseudo-labels and ground-truth. We reveal that the simple mechanism of copy-pasting bidirectionally between labeled and unlabeled data is good enough and the experiments show solid gains (e.g., over 21% Dice improvement on ACDC dataset with 5% labeled data) compared with other state-of-the-arts on various semi-supervised medical image segmentation datasets. Code is avaiable at https://github.com/DeepMed-Lab-ECNU/BCP.

        ----

        ## [1097] Directional Connectivity-based Segmentation of Medical Images

        **Authors**: *Ziyun Yang, Sina Farsiu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01109](https://doi.org/10.1109/CVPR52729.2023.01109)

        **Abstract**:

        Anatomical consistency in biomarker segmentation is crucial for many medical image analysis tasks. A promising paradigm for achieving anatomically consistent segmentation via deep networks is incorporating pixel connectivity, a basic concept in digital topology, to model inter-pixel relationships. However, previous works on connectivity modeling have ignored the rich channel-wise directional information in the latent space. In this work, we demonstrate that effective disentanglement of directional sub-space from the shared latent space can significantly enhance the feature representation in the connectivity- based network. To this end, we propose a directional connectivity modeling scheme for segmentation that decouples, tracks, and utilizes the directional information across the network. Experiments on various public medical image segmentation benchmarks show the effectiveness of our model as compared to the state-of-the-art methods. Code is available at https://github.com/Zyun-Y/DconnNet.

        ----

        ## [1098] Ambiguous Medical Image Segmentation Using Diffusion Models

        **Authors**: *Aimon Rahman, Jeya Maria Jose Valanarasu, Ilker Hacihaliloglu, Vishal M. Patel*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01110](https://doi.org/10.1109/CVPR52729.2023.01110)

        **Abstract**:

        Collective insights from a group of experts have always proven to outperform an individual's best diagnostic for clinical tasks. For the task of medical image segmentation, existing research on AI-based alternatives focuses more on developing models that can imitate the best individual rather than harnessing the power of expert groups. In this paper, we introduce a single diffusion model-based approach that produces multiple plausible outputs by learning a distribution over group insights. Our proposed model generates a distribution of segmentation masks by leveraging the inherent stochastic sampling process of diffusion using only minimal additional learning. We demonstrate on three different medical image modalities- CT, ultrasound, and MRI that our model is capable of producing several possible variants while capturing the frequencies of their occurrences. Comprehensive results show that our proposed approach outperforms existing state-of-the-art ambiguous segmentation networks in terms of accuracy while preserving naturally occurring variation. We also propose a new metric to evaluate the diversity as well as the accuracy of segmentation predictions that aligns with the interest of clinical practice of collective insights. Implementation code: https://github.com/aimansnigdha/Ambiguous-Medical-Image-Segmentation-using-Diffusion-Models.

        ----

        ## [1099] Sparse Multi-Modal Graph Transformer with Shared-Context Processing for Representation Learning of Giga-pixel Images

        **Authors**: *Ramin Nakhli, Puria Azadi Moghadam, Haoyang Mi, Hossein Farahani, Alexander Baras, C. Blake Gilks, Ali Bashashati*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01111](https://doi.org/10.1109/CVPR52729.2023.01111)

        **Abstract**:

        Processing giga-pixel whole slide histopathology images (WSI) is a computationally expensive task. Multiple instance learning (MIL) has become the conventional approach to process WSIs, in which these images are split into smaller patches for further processing. However, MIL-based techniques ignore explicit information about the individual cells within a patch. In this paper, by defining the novel concept of shared-context processing, we designed a multi-modal Graph Transformer (AMIGO) that uses the cellular graph within the tissue to provide a single representation for a patient while taking advantage of the hierarchical structure of the tissue, enabling a dynamic focus between cell-level and tissue-level information. We benchmarked the performance of our model against multiple state-of-the-art methods in survival prediction and showed that ours can significantly outperform all of them including hierarchical Vision Transformer (ViT). More importantly, we show that our model is strongly robust to missing information to an extent that it can achieve the same performance with as low as 20% of the data. Finally, in two different cancer datasets, we demonstrated that our model was able to stratify the patients into low-risk and high-risk groups while other state-of-the-art methods failed to achieve this goal. We also publish a large dataset of immunohistochemistry images (InUIT) containing 1,600 tissue microarray (TMA) cores from 188 patients along with their survival information, making it one of the largest publicly available datasets in this context.

        ----

        ## [1100] METransformer: Radiology Report Generation by Transformer with Multiple Learnable Expert Tokens

        **Authors**: *Zhanyu Wang, Lingqiao Liu, Lei Wang, Luping Zhou*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01112](https://doi.org/10.1109/CVPR52729.2023.01112)

        **Abstract**:

        In clinical scenarios, multi-specialist consultation could significantly benefit the diagnosis, especially for intricate cases. This inspires us to explore a “multi-expert joint diagnosis” mechanism to upgrade the existing “single expert” framework commonly seen in the current literature. To this end, we propose METransformer, a method to realize this idea with a transformer-based backbone. The key design of our method is the introduction of multiple learnable “expert” tokens into both the transformer encoder and decoder. In the encoder, each expert token interacts with both vision tokens and other expert tokens to learn to attend different image regions for image representation. These expert tokens are encouraged to capture complementary information by an orthogonal loss that minimizes their over-lap. In the decoder, each attended expert token guides the cross-attention between input words and visual tokens, thus influencing the generated report. A metrics-based expert voting strategy is further developed to generate the final report. By the multi-experts concept, our model enjoys the merits of an ensemble-based approach but through a manner that is computationally more efficient and supports more sophisticated interactions among experts. Experimental results demonstrate the promising performance of our proposed model on two widely used benchmarks. Last but not least, the framework-level innovation makes our work ready to incorporate advances on existing “single-expert” models to further improve its performance.

        ----

        ## [1101] Towards Trustable Skin Cancer Diagnosis via Rewriting Model's Decision

        **Authors**: *Siyuan Yan, Zhen Yu, Xuelin Zhang, Dwarikanath Mahapatra, Shekhar S. Chandra, Monika Janda, H. Peter Soyer, Zongyuan Ge*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01113](https://doi.org/10.1109/CVPR52729.2023.01113)

        **Abstract**:

        Deep neural networks have demonstrated promising performance on image recognition tasks. However, they may heavily rely on confounding factors, using irrelevant artifacts or bias within the dataset as the cue to improve performance. When a model performs decision-making based on these spurious correlations, it can become untrustable and lead to catastrophic outcomes when deployed in the realworld scene. In this paper, we explore and try to solve this problem in the context of skin cancer diagnosis. We introduce a human-in-the-loop framework in the model training process such that users can observe and correct the model's decision logic when confounding behaviors happen. Specifically, our method can automatically discover confounding factors by analyzing the co-occurrence behavior of the samples. It is capable of learning confounding concepts using easily obtained concept exemplars. By mapping the black-box model's feature representation onto an explainable concept space, human users can interpret the concept and intervene via first order-logic instruction. We systematically evaluate our method on our newly crafted, well-controlled skin lesion dataset and several public skin lesion datasets. Experiments show that our method can effectively detect and remove confounding factors from datasets without any prior knowledge about the category distribution and does not require fully annotated concept labels. We also show that our method enables the model to focus on clinical-related concepts, improving the model's performance and trustworthiness during model inference.

        ----

        ## [1102] Rethinking Out-of-distribution (OOD) Detection: Masked Image Modeling is All You Need

        **Authors**: *Jingyao Li, Pengguang Chen, Zexin He, Shaozuo Yu, Shu Liu, Jiaya Jia*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01114](https://doi.org/10.1109/CVPR52729.2023.01114)

        **Abstract**:

        The core of out-of-distribution (OOD) detection is to learn the in-distribution (ID) representation, which is distinguishable from OOD samples. Previous work applied recognition-based methods to learn the ID features, which tend to learn shortcuts instead of comprehensive representations. In this work, we find surprisingly that simply using reconstruction-based methods could boost the performance of OOD detection significantly. We deeply explore the main contributors of OOD detection and find that reconstruction-based pretext tasks have the potential to provide a generally applicable and efficacious prior, which benefits the model in learning intrinsic data distributions of the ID dataset. Specifically, we take Masked Image Modeling as a pretext task for our OOD detection framework (MOOD). Without bells and whistles, MOOD outperforms previous SOTA of one-class OOD detection by 5.7%, multi-class OOD detection by 3.0%, and near-distribution OOD detection by 2.1 %. It even defeats the 10-shot-per-class out-lier exposure OOD detection, although we do not include any OOD samples for our detection. Codes are available at https://github.com/lijingyao20010602/MOOD

        ----

        ## [1103] MetaViewer: Towards A Unified Multi-View Representation

        **Authors**: *Ren Wang, Haoliang Sun, Yuling Ma, Xiaoming Xi, Yilong Yin*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01115](https://doi.org/10.1109/CVPR52729.2023.01115)

        **Abstract**:

        Existing multi-view representation learning methods typically follow a specific-to-uniform pipeline, extracting latent features from each view and then fusing or aligning them to obtain the unified object representation. However, the manually pre-specified fusion functions and aligning criteria could potentially degrade the quality of the derived representation. To overcome them, we propose a novel uniform-to-specific multi-view learning framework from a meta-learning perspective, where the unified representation no longer involves manual manipulation but is automatically derived from a meta-learner named MetaViewer. Specifically, we formulated the extraction and fusion of view-specific latent features as a nested optimization problem and solved it by using a bi-level optimization scheme. In this way, MetaViewer automatically fuses view-specific features into a unified one and learns the optimal fusion scheme by observing reconstruction processes from the unified to the specific over all views. Extensive experimental results in downstream classification and clustering tasks demonstrate the efficiency and effectiveness of the proposed method.

        ----

        ## [1104] Deep Incomplete Multi-View Clustering with Cross-View Partial Sample and Prototype Alignment

        **Authors**: *Jiaqi Jin, Siwei Wang, Zhibin Dong, Xinwang Liu, En Zhu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01116](https://doi.org/10.1109/CVPR52729.2023.01116)

        **Abstract**:

        The success of existing multi-view clustering relies on the assumption of sample integrity across multiple views. However, in real-world scenarios, samples of multi-view are partially available due to data corruption or sensor failure, which leads to incomplete multi-view clustering study (IMVC). Although several attempts have been proposed to address IMVC, they suffer from the following draw-backs: i) Existing methods mainly adopt cross-view contrastive learning forcing the representations of each sample across views to be exactly the same, which might ignore view discrepancy and flexibility in representations; ii) Due to the absence of non-observed samples across multiple views, the obtained prototypes of clusters might be unaligned and biased, leading to incorrect fusion. To address the above issues, we propose a Cross-view Partial Sample and Prototype Alignment Network (CPSPAN) for Deep Incomplete Multi-view Clustering. Firstly, unlike existing contrastive-based methods, we adopt pair-observed data alignment as 'proxy supervised signals' to guide instance-to-instance correspondence construction among views. Then, regarding of the shifted prototypes in IMVC, we further propose a prototype alignment module to achieve incomplete distribution calibration across views. Extensive experimental results showcase the effectiveness of our proposed modules, attaining noteworthy performance improvements when compared to existing IMVC competitors on benchmark datasets.

        ----

        ## [1105] RONO: Robust Discriminative Learning with Noisy Labels for 2D-3D Cross-Modal Retrieval

        **Authors**: *Yanglin Feng, Hongyuan Zhu, Dezhong Peng, Xi Peng, Peng Hu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01117](https://doi.org/10.1109/CVPR52729.2023.01117)

        **Abstract**:

        Recently, with the advent of Metaverse and AI Generated Content, cross-modal retrieval becomes popular with a burst of 2D and 3D data. However, this problem is challenging given the heterogeneous structure and semantic discrepancies. Moreover, imperfect annotations are ubiquitous given the ambiguous 2D and 3D content, thus inevitably producing noisy labels to degrade the learning performance. To tackle the problem, this paper proposes a robust 2D-3D retrieval framework (RONO) to robustly learn from noisy multimodal data. Specifically, one novel Robust Discriminative Center Learning mechanism (RDCL) is proposed in RONO to adaptively distinguish clean and noisy samples for respectively providing them with positive and negative optimization directions, thus mitigating the negative impact of noisy labels. Besides, we present a Shared Space Consistency Learning mechanism (SSCL) to capture the intrinsic information inside the noisy data by minimizing the cross-modal and semantic discrepancy between common space and label space simultaneously. Comprehensive mathematical analyses are given to theoretically prove the noise tolerance of the proposed method. Furthermore, we conduct extensive experiments on four 3D-model multimodal datasets to verify the effectiveness of our method by comparing it with 15 state-of-the-art methods. Code is available at https://github.com/penghu-cs/RONO.

        ----

        ## [1106] Mind the Label Shift of Augmentation-based Graph OOD Generalization

        **Authors**: *Junchi Yu, Jian Liang, Ran He*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01118](https://doi.org/10.1109/CVPR52729.2023.01118)

        **Abstract**:

        Out-of-distribution (OOD) generalization is an important issue for Graph Neural Networks (GNNs). Recent works employ different graph editions to generate augmented environments and learn an invariant GNN for generalization. However, the label shift usually occurs in augmentation since graph structural edition inevitably alters the graph label. This brings inconsistent predictive relationships among augmented environments, which is harmful to generalization. To address this issue, we propose LiSA, which generates label-invariant augmentations to facilitate graph OOD generalization. Instead of resorting to graph editions, LiSA exploits Label-invariant Subgraphs of the training graphs to construct Augmented environments. Specifically, LiSA first designs the variational subgraph generators to extract locally predictive patterns and construct multiple label-invariant subgraphs efficiently. Then, the subgraphs produced by different generators are collected to build different augmented environments. To promote diversity among augmented environments, LiSA further introduces a tractable energy-based regularization to enlarge pair-wise distances between the distributions of environments. In this manner, LiSA generates diverse augmented environments with a consistent predictive relationship and facilitates learning an invariant GNN. Extensive experiments on node-level and graph-level OOD benchmarks show that LiSA achieves impressive generalization performance with different GNN backbones. Code is available on https://github.com/Samyu0304/LiSA.

        ----

        ## [1107] Zero-Shot Model Diagnosis

        **Authors**: *Jinqi Luo, Zhaoning Wang, Chen Henry Wu, Dong Huang, Fernando De la Torre*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01119](https://doi.org/10.1109/CVPR52729.2023.01119)

        **Abstract**:

        When it comes to deploying deep vision models, the behavior of these systems must be explicable to ensure confidence in their reliability and fairness. A common approach to evaluate deep learning models is to build a labeled test set with attributes of interest and assess how well it performs. However, creating a balanced test set (i.e., one that is uniformly sampled over all the important traits) is often time-consuming, expensive, and prone to mistakes. The question we try to address is: can we evaluate the sensitivity of deep learning models to arbitrary visual attributes without an annotated test set? This paper argues the case that Zero-shot Model Diagnosis (ZOOM) is possible without the need for a test set nor labeling. To avoid the need for test sets, our system relies on a generative model and CLIP. The key idea is enabling the user to select a set of prompts (relevant to the problem) and our system will automatically search for semantic counterfactual images (i.e., synthesized images that flip the prediction in the case of a binary classifier) using the generative model. We evaluate several visual tasks (classification, key-point detection, and segmentation) in multiple visual domains to demonstrate the viability of our methodology. Extensive experiments demonstrate that our method is capable of producing counterfactual images and offering sensitivity analysis for model diagnosis without the need for a test set.

        ----

        ## [1108] Protocon: Pseudo-Label Refinement via Online Clustering and Prototypical Consistency for Efficient Semi-Supervised Learning

        **Authors**: *Islam Nassar, Munawar Hayat, Ehsan Abbasnejad, Hamid Rezatofighi, Gholamreza Haffari*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01120](https://doi.org/10.1109/CVPR52729.2023.01120)

        **Abstract**:

        Confidence-based pseudo-labeling is among the dominant approaches in semi-supervised learning (SSL). It relies on including high-confidence predictions made on unlabeled data as additional targets to train the model. We propose Protocon, a novel SSL method aimed at the less-explored label-scarce SSL where such methods usually underperform. Protocon refines the pseudolabels by lever-aging their nearest neighbours' information. The neighbours are identified as the training proceeds using an online clustering approach operating in an embedding space trained via a prototypical loss to encourage well-formed clusters. The online nature of Protocon allows it to utilise the label history of the entire dataset in one training cycle to refine labels in the following cycle without the need to store image embeddings. Hence, it can seamlessly scale to larger datasets at a low cost. Finally, Protocon addresses the poor training signal in the initial phase of training (due to fewer confident predictions) by introducing an auxiliary self-supervised loss. It delivers significant gains and faster convergence over state-of-the-art across 5 datasets, including CIFARs, ImageNet and DomainNet.

        ----

        ## [1109] Fine-Grained Classification with Noisy Labels

        **Authors**: *Qi Wei, Lei Feng, Haoliang Sun, Ren Wang, Chenhui Guo, Yilong Yin*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01121](https://doi.org/10.1109/CVPR52729.2023.01121)

        **Abstract**:

        Learning with noisy labels (LNL) aims to ensure model generalization given a label-corrupted training set. In this work, we investigate a rarely studied scenario of LNL on fine-grained datasets (LNL-FG), which is more practical and challenging as large inter-class ambiguities among fine-grained classes cause more noisy labels. We empirically show that existing methods that work well for LNL fail to achieve satisfying performance for LNL-FG, arising the practical need of effective solutions for LNL-FG. To this end, we propose a novel framework called stochastic noise-tolerated supervised contrastive learning (SNSCL) that confronts label noise by encouraging distinguishable representation. Specifically, we design a noise-tolerated supervised contrastive learning loss that incorporates a weight-aware mechanism for noisy label correction and selectively updating momentum queue lists. By this mechanism, we mitigate the effects of noisy anchors and avoid inserting noisy labels into the momentum-updated queue. Besides, to avoid manually-defined augmentation strategies in contrastive learning, we propose an efficient stochastic module that samples feature embeddings from a generated distribution, which can also enhance the representation ability of deep models. SNSCL is general and compatible with prevailing robust LNL strategies to improve their performance for LNL-FG. Extensive experiments demonstrate the effectiveness of SNSCL.

        ----

        ## [1110] Twin Contrastive Learning with Noisy Labels

        **Authors**: *Zhizhong Huang, Junping Zhang, Hongming Shan*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01122](https://doi.org/10.1109/CVPR52729.2023.01122)

        **Abstract**:

        Learning from noisy data is a challenging task that sig-nificantly degenerates the model performance. In this paper, we present TCL, a novel twin contrastive learning model to learn robust representations and handle noisy labels for classification. Specifically, we construct a Gaussian mixture model (GMM) over the representations by injecting the supervised model predictions into GMM to link label- free latent variables in GMM with label-noisy annotations. Then, TCL detects the examples with wrong labels as the out- of-distribution examples by another two-component GMM, taking into account the data distribution. We further propose a cross-supervision with an entropy regularization loss that bootstraps the true targets from model predictions to handle the noisy labels. As a result, TCL can learn discriminative representations aligned with estimated labels through mixup and contrastive learning. Extensive experimental results on several standard benchmarks and real-world datasets demonstrate the superior performance of TCL. In particular, TCL achieves 7.5% improvements on CIFAR-10 with 90% noisy label-an extremely noisy scenario. The source code is available at https://github.com/Hzzone/TCL.

        ----

        ## [1111] RMLVQA: A Margin Loss Approach For Visual Question Answering with Language Biases

        **Authors**: *Abhipsa Basu, Sravanti Addepalli, R. Venkatesh Babu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01123](https://doi.org/10.1109/CVPR52729.2023.01123)

        **Abstract**:

        Visual Question Answering models have been shown to suffer from language biases, where the model learns a correlation between the question and the answer, ignoring the image. While early works attempted to use question-only models or data augmentations to reduce this bias, we propose an adaptive margin loss approach having two components. The first component considers the frequency of answers within a question type in the training data, which addresses the concern of the class-imbalance causing the language biases. However, it does not take into account the answering difficulty of the samples, which impacts their learning. We address this through the second component, where instance-specific margins are learnt, allowing the model to distinguish between samples of varying complexity. We introduce a bias-injecting component to our model, and compute the instance-specific margins from the confidence of this component. We combine these with the estimated margins to consider both answer-frequency and task-complexity in the training loss. We show that, while the margin loss is effective for out-of-distribution (ood) data, the bias-injecting component is essential for generalising to in-distribution (id) data. Our proposed approach, Robust Margin Loss for Visual Question Answering (RMLVQA) 
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
Code available at https://github.com/val-iisc/RMLVQA improves upon the existing state-of-the-art results when compared to augmentation-free methods on benchmark VQA datasets suffering from language biases, while maintaining competitive performance on id data, making our method the most robust one among all comparable methods.

        ----

        ## [1112] Generative Bias for Robust Visual Question Answering

        **Authors**: *Jae-Won Cho, Dong-Jin Kim, Hyeonggon Ryu, In So Kweon*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01124](https://doi.org/10.1109/CVPR52729.2023.01124)

        **Abstract**:

        The task of Visual Question Answering (VQA) is known to be plagued by the issue of VQA models exploiting biases within the dataset to make its final prediction. Various previous ensemble based debiasing methods have been proposed where an additional model is purposefully trained to be biased in order to train a robust target model. However, these methods compute the bias for a model simply from the label statistics of the training data or from single modal branches. In this work, in order to better learn the bias a target VQA model suffers from, we propose a generative method to train the bias model directly from the target model, called GenB. In particular, GenB employs a generative network to learn the bias in the target model through a combination of the adversarial objective and knowledge distillation. We then debias our target model with GenB as a bias model, and show through extensive experiments the effects of our method on various VQA bias datasets including VQA-CP2, VQA-CP1, GQA-OOD, and VQA-CE, and show state-of-the-art results with the LXMERT architecture on VQA-CP2.

        ----

        ## [1113] On-the-Fly Category Discovery

        **Authors**: *Ruoyi Du, Dongliang Chang, Kongming Liang, Timothy M. Hospedales, Yi-Zhe Song, Zhanyu Ma*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01125](https://doi.org/10.1109/CVPR52729.2023.01125)

        **Abstract**:

        Although machines have surpassed humans on visual recognition problems, they are still limited to providing closed-set answers. Unlike machines, humans can cognize novel categories at the first observation. Novel category discovery (NCD) techniques, transferring knowledge from seen categories to distinguish unseen categories, aim to bridge the gap. However, current NCD methods assume a transductive learning and offline inference paradigm, which restricts them to a predefined query set and renders them unable to deliver instant feedback. In this paper, we study on-the-fly category discovery (OCD) aimed at making the model instantaneously aware of novel category samples (i.e., enabling inductive learning and streaming inference). We first design a hash coding-based expandable recognition model as a practical baseline. Afterwards, noticing the sensitivity of hash codes to intra-category variance, we further propose a novel Sign-Magnitude dIsentangLEment (SMILE) architecture to alleviate the disturbance it brings. Our experimental results demonstrate the superiority of SMILE against our baseline model and prior art. Our code is available at https://github.com/PRIS-CV/On-the-fly-Category-Discovery.

        ----

        ## [1114] Co-training 2L Submodels for Visual Recognition

        **Authors**: *Hugo Touvron, Matthieu Cord, Maxime Oquab, Piotr Bojanowski, Jakob Verbeek, Hervé Jégou*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01126](https://doi.org/10.1109/CVPR52729.2023.01126)

        **Abstract**:

        We introduce submodel co-training, a regularization method related to co-training, self-distillation and stochastic depth. Given a neural network to be trained, for each sample we implicitly instantiate two altered networks, “submodels”, with stochastic depth: we activate only a subset of the layers. Each network serves as a soft teacher to the other, by providing a loss that complements the regular loss provided by the one-hot label. Our approach, dubbed “co-sub”, uses a single set of weights, and does not involve a pre-trained external model or temporal averaging. Experimentally, we show that submodel co-training is effective to train backbones for recognition tasks such as image classification and semantic segmentation. Our approach is compatible with multiple architectures, including RegNet, ViT, PiT, XCiT, Swin and ConvNext. Our training strategy improves their results in comparable settings. For instance, a ViT-B pretrained with cosub on ImageNet-21k obtains 87.4% top1 acc. @448 on ImageNet-val.

        ----

        ## [1115] Neural Dependencies Emerging from Learning Massive Categories

        **Authors**: *Ruili Feng, Kecheng Zheng, Kai Zhu, Yujun Shen, Jian Zhao, Yukun Huang, Deli Zhao, Jingren Zhou, Michael I. Jordan, Zheng-Jun Zha*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01127](https://doi.org/10.1109/CVPR52729.2023.01127)

        **Abstract**:

        This work presents two astonishing findings on neural networks learned for large-scale image classification. 1) Given a well-trained model, the logits predicted for some category can be directly obtained by linearly combining the predictions of a few other categories, which we call neural dependency. 2) Neural dependencies exist not only within a single model, but even between two independently learned models, regardless of their architectures. Towards a theoretical analysis of such phenomena, we demonstrate that identifying neural dependencies is equivalent to solving the Covariance Lasso (CovLasso) regression problem proposed in this paper. Through investigating the properties of the problem solution, we confirm that neural dependency is guaranteed by a redundant logit covariance matrix, which condition is easily met given massive categories, and that neural dependency is highly sparse, implying that one category correlates to only a few others. We further empirically show the potential of neural dependencies in understanding internal data correlations, generalizing models to unseen categories, and improving model robustness with a dependency-derived regularizer. Code to reproduce the results in this paper is available at https://github.com/RuiLiFengiNeural-Dependencies.

        ----

        ## [1116] MIC: Masked Image Consistency for Context-Enhanced Domain Adaptation

        **Authors**: *Lukas Hoyer, Dengxin Dai, Haoran Wang, Luc Van Gool*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01128](https://doi.org/10.1109/CVPR52729.2023.01128)

        **Abstract**:

        In unsupervised domain adaptation (UDA), a model trained on source data (e.g. synthetic) is adapted to target data (e.g. real-world) without access to target annotation. Most previous UDA methods struggle with classes that have a similar visual appearance on the target domain as no ground truth is available to learn the slight appearance differences. To address this problem, we propose a Masked Image Consistency (MIC) module to enhance UDA by learning spatial context relations of the target domain as additional clues for robust visual recognition. MIC enforces the consistency between predictions of masked target images, where random patches are withheld, and pseudo-labels that are generated based on the complete image by an exponential moving average teacher. To minimize the consistency loss, the network has to learn to infer the predictions of the masked regions from their context. Due to its simple and universal concept, MIC can be integrated into various UDA methods across different visual recognition tasks such as image classification, semantic segmentation, and object detection. MIC significantly improves the state-of-the-art performance across the different recognition tasks for synthetic-to-real, day-to-nighttime, and clear-toadverse-weather UDA. For instance, MIC achieves an un-precedented UDA performance of 75.9 mIoU and 92.8% on GTA→Cityscapes and VisDA-2017, respectively, which corresponds to an improvement of +2.1 and +3.0 percent points over the previous state of the art. The implementation is available at https://github.com/lhoyer/MIC.

        ----

        ## [1117] Towards Better Stability and Adaptability: Improve Online Self-Training for Model Adaptation in Semantic Segmentation

        **Authors**: *Dong Zhao, Shuang Wang, Qi Zang, Dou Quan, Xiutiao Ye, Licheng Jiao*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01129](https://doi.org/10.1109/CVPR52729.2023.01129)

        **Abstract**:

        Unsupervised domain adaptation (UDA) in semantic segmentation transfers the knowledge of the source domain to the target one to improve the adaptability of the segmentation model in the target domain. The need to access labeled source data makes UDA unable to handle adaptation scenarios involving privacy, property rights protection, and confidentiality. In this paper, we focus on unsupervised model adaptation (UMA), also called source-free domain adaptation, which adapts a source-trained model to the target domain without accessing source data. We find that the online self-training method has the potential to be deployed in UMA, but the lack of source domain loss will greatly weaken the stability and adaptability of the method. We analyze two reasons for the degradation of online self-training, i.e. inopportune updates of the teacher model and biased knowledge from the source-trained model. Based on this, we propose a dynamic teacher update mechanism and a training-consistency based resampling strategy to improve the stability and adaptability of online self-training. On multiple model adaptation benchmarks, our method obtains new state-of-the-art performance, which is comparable or even better than state-of-the-art UDA methods. The code is available at https://github.com/DZhaoXd/DT-ST.

        ----

        ## [1118] DARE-GRAM : Unsupervised Domain Adaptation Regression by Aligning Inverse Gram Matrices

        **Authors**: *Ismail Nejjar, Qin Wang, Olga Fink*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01130](https://doi.org/10.1109/CVPR52729.2023.01130)

        **Abstract**:

        Unsupervised Domain Adaptation Regression (DAR) aims to bridge the domain gap between a labeled source dataset and an unlabelled target dataset for regression problems. Recent works mostly focus on learning a deep feature encoder by minimizing the discrepancy between source and target features. In this work, we present a different perspective for the DAR problem by analyzing the closed-form ordinary least square (OLS) solution to the linear regressor in the deep domain adaptation context. Rather than aligning the original feature embedding space, we propose to align the inverse Gram matrix of the features, which is motivated by its presence in the OLS solution and the Gram matrix's ability to capture the feature correlations. Specifically, we propose a simple yet effective DAR method which leverages the pseudo-inverse low-rank property to align the scale and angle in a selected subspace generated by the pseudo-inverse Gram matrix of the two domains. We evaluate our method on three domain adaptation regression benchmarks. Experimental results demonstrate that our method achieves state-of-the-art performance. Our code is available at https://github.com/ismailnejjar/DARE-GRAM.

        ----

        ## [1119] Equiangular Basis Vectors

        **Authors**: *Yang Shen, Xuhao Sun, Xiu-Shen Wei*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01131](https://doi.org/10.1109/CVPR52729.2023.01131)

        **Abstract**:

        We propose Equiangular Basis Vectors (EBVs) for classification tasks. In deep neural networks, models usually end with a k-way fully connected layer with softmax to handle different classification tasks. The learning objective of these methods can be summarized as mapping the learned feature representations to the samples' label space. While in metric learning approaches, the main objective is to learn a transformation function that maps training data points from the original space to a new space where similar points are closer while dissimilar points become farther apart. Different from previous methods, our EBVs generate normalized vector embeddings as “predefined classifiers” which are required to not only be with the equal status between each other, but also be as orthogonal as possible. By minimizing the spherical distance of the embedding of an input between its categorical EBV in training, the predictions can be obtained by identifying the categorical EBV with the smallest distance during inference. Various experiments on the ImageNet-1K dataset and other downstream tasks demonstrate that our method outperforms the general fully connected classifier while it does not introduce huge additional computation compared with classical metric learning methods. Our EBVs won the first place in the 2022 DIGIX Global AI Challenge, and our code is open-source and available at https://github.com/NJUST-VIPGroup/Equiangular-Basis-Vectors.

        ----

        ## [1120] Enhanced Multimodal Representation Learning with Cross-modal KD

        **Authors**: *Mengxi Chen, Linyu Xing, Yu Wang, Ya Zhang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01132](https://doi.org/10.1109/CVPR52729.2023.01132)

        **Abstract**:

        This paper explores the tasks of leveraging auxiliary modalities which are only available at training to enhance multimodal representation learning through cross-modal Knowledge Distillation (KD). The widely adopted mutual information maximization-based objective leads to a short-cut solution of the weak teacher, i.e., achieving the maximum mutual information by simply making the teacher model as weak as the student model. To prevent such a weak solution, we introduce an additional objective term, i.e., the mutual information between the teacher and the auxiliary modality model. Besides, to narrow down the information gap between the student and teacher, we further propose to minimize the conditional entropy of the teacher given the student. Novel training schemes based on contrastive learning and adversarial learning are designed to optimize the mutual information and the conditional entropy, respectively. Experimental results on three popular multimodal benchmark datasets have shown that the proposed method outperforms a range of state-of-the-art approaches for video recognition, video retrieval and emotion classification.

        ----

        ## [1121] Decompose, Adjust, Compose: Effective Normalization by Playing with Frequency for Domain Generalization

        **Authors**: *Sangrok Lee, Jongseong Bae, Ha Young Kim*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01133](https://doi.org/10.1109/CVPR52729.2023.01133)

        **Abstract**:

        Domain generalization (DG) is a principal task to evaluate the robustness of computer vision models. Many previous studies have used normalization for DG. In normalization, statistics and normalized features are regarded as style and content, respectively. However, it has a content variation problem when removing style because the boundary between content and style is unclear. This study addresses this problem from the frequency domain perspective, where amplitude and phase are considered as style and content, respectively. First, we verify the quantitative phase variation of normalization through the mathematical derivation of the Fourier transform formula. Then, based on this, we propose a novel normalization method, PC Norm, which eliminates style only as the preserving content through spectral decomposition. Furthermore, we propose advanced PCNorm variants, CCNorm and SCNorm, which adjust the degrees of variations in content and style, respectively. Thus, they can learn domain-agnostic representations for DG. With the normalization methods, we propose ResNet-variant models, DAC-P and DAC-SC, which are robust to the domain gap. The proposed models outperform other recent DG methods. The DAC-SC achieves an average state-of-the-art performance of 65.6% on five datasets: PACS, VLCS, Office-Home, DomainNet, and TerraIncognita.

        ----

        ## [1122] Back to the Source: Diffusion-Driven Adaptation to Test-Time Corruption

        **Authors**: *Jin Gao, Jialing Zhang, Xihui Liu, Trevor Darrell, Evan Shelhamer, Dequan Wang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01134](https://doi.org/10.1109/CVPR52729.2023.01134)

        **Abstract**:

        Test-time adaptation harnesses test inputs to improve the accuracy of a model trained on source data when tested on shifted target data. Most methods update the source model by (re-)training on each target domain. While retraining can help, it is sensitive to the amount and order of the data and the hyperparameters for optimization. We update the target data instead, and project all test inputs toward the source domain with a generative diffusion model. Our diffusion-driven adaptation (DDA) method shares its models for classification and generation across all domains, training both on source then freezing them for all targets, to avoid expensive domain-wise retraining. We augment diffusion with image guidance and classifier self-ensembling to automatically decide how much to adapt. Input adaptation by DDA is more robust than model adaptation across a variety of corruptions, models, and data regimes on the ImageNet-C benchmark. With its input-wise updates, DDA succeeds where model adaptation degrades on too little data (small batches), on dependent data (correlated orders), or on mixed data (multiple corruptions).

        ----

        ## [1123] Deep Frequency Filtering for Domain Generalization

        **Authors**: *Shiqi Lin, Zhizheng Zhang, Zhipeng Huang, Yan Lu, Cuiling Lan, Peng Chu, Quanzeng You, Jiang Wang, Zicheng Liu, Amey Parulkar, Viraj Navkal, Zhibo Chen*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01135](https://doi.org/10.1109/CVPR52729.2023.01135)

        **Abstract**:

        Improving the generalization ability of Deep Neural Networks (DNNs) is critical for their practical uses, which has been a longstanding challenge. Some theoretical studies have uncovered that DNNs have preferences for some frequency components in the learning process and indicated that this may affect the robustness of learned features. In this paper, we propose Deep Frequency Filtering (DFF)for learning domain-generalizable features, which is the first endeavour to explicitly modulate the frequency components of different transfer difficulties across domains in the latent space during training. To achieve this, we perform Fast Fourier Transform (FFT) for the feature maps at different layers, then adopt a light-weight module to learn attention masks from the frequency representations after FFT to enhance transferable components while suppressing the components not conducive to generalization. Further, we empirically compare the effectiveness of adopting different types of attention designs for implementing DFF. Extensive experiments demonstrate the effectiveness of our proposed DFF and show that applying our DFF on a plain baseline out-performs the state-of-the-art methods on different domain generalization tasks, including close-set classification and open-set retrieval.

        ----

        ## [1124] Generalizable Implicit Neural Representations via Instance Pattern Composers

        **Authors**: *Chiheon Kim, Doyup Lee, Saehoon Kim, Minsu Cho, Wook-Shin Han*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01136](https://doi.org/10.1109/CVPR52729.2023.01136)

        **Abstract**:

        Despite recent advances in implicit neural representations (INRs), it remains challenging for a coordinate-based multi-layer perceptron (MLP) of INRs to learn a common representation across data instances and generalize it for unseen instances. In this work, we introduce a simple yet effective framework for generalizable INRs that enables a coordinate-based MLP to represent complex data instances by modulating only a small set of weights in an early MLP layer as an instance pattern composer; the remaining MLP weights learn pattern composition rules for common representations across instances. Our generalizable INR frame-work is fully compatible with existing meta-learning and hypernetworks in learning to predict the modulated weight for unseen instances. Extensive experiments demonstrate that our method achieves high performance on a wide range of domains such as an audio, image, and 3D object, while the ablation study validates our weight modulation.

        ----

        ## [1125] Train-Once-for-All Personalization

        **Authors**: *Hong-You Chen, Yandong Li, Yin Cui, Mingda Zhang, Wei-Lun Chao, Li Zhang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01137](https://doi.org/10.1109/CVPR52729.2023.01137)

        **Abstract**:

        We study the problem of how to train a “personalization-friendly” model such that given only the task descriptions, the model can be adapted to different end-users' needs, e.g., for accurately classifying different subsets of objects. One baseline approach is to train a “generic” model for classifying a wide range of objects, followed by class selection. In our experiments, we however found it suboptimal, perhaps because the model's weights are kept frozen without being personalized. To address this drawback, we propose Train-once-for-All PERsonalization (TAPER), a framework that is trained just once and can later customize a model for different end-users given their task descriptions. TAPER learns a set of “basis” models and a mixer predictor, such that given the task description, the weights (not the predictions!) of the basis models can be on the fly combined into a single “personalized” model. Via extensive experiments on multiple recognition tasks, we show that TAPER consistently outperforms the baseline methods in achieving a higher personalized accuracy. Moreover, we show that TAPER can synthesize a much smaller model to achieve comparable performance to a huge generic model, making it “deployment-friendly” to resource-limited end devices. Interestingly, even without end-users' task descriptions, TAPER can still be specialized to the deployed context based on its past predictions, making it even more “personalization-friendly”.

        ----

        ## [1126] Mod-Squad: Designing Mixtures of Experts As Modular Multi-Task Learners

        **Authors**: *Zitian Chen, Yikang Shen, Mingyu Ding, Zhenfang Chen, Hengshuang Zhao, Erik G. Learned-Miller, Chuang Gan*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01138](https://doi.org/10.1109/CVPR52729.2023.01138)

        **Abstract**:

        Optimization in multi-task learning (MTL) is more challenging than single-task learning (STL), as the gradient from different tasks can be contradictory. When tasks are related, it can be beneficial to share some parameters among them (cooperation). However, some tasks require additional parameters with expertise in a specific type of data or discrimination (specialization). To address the MTL challenge, we propose Mod-Squad, a new model that is Modularized into groups of experts (a ‘Squad’). This structure allows us to formalize cooperation and specialization as the process of matching experts and tasks. We optimize this matching process during the training of a single model. Specifically, we incorporate mixture of experts (MoE) layers into a transformer model, with a new loss that incorporates the mutual dependence between tasks and experts. As a result, only a small set of experts are activated for each task. This prevents the sharing of the entire backbone model between all tasks, which strengthens the model, especially when the training set size and the number of tasks scale up. More interestingly, for each task, we can extract the small set of experts as a standalone model that maintains the same performance as the large model. Extensive experiments on the Taskonomy dataset with 13 vision tasks and the PASCAL-Context dataset with 5 vision tasks show the superiority of our approach. The project page can be accessed at https://vis-www.cs.umass.edu/Mod-Squad.

        ----

        ## [1127] Few-Shot Class-Incremental Learning via Class-Aware Bilateral Distillation

        **Authors**: *Linglan Zhao, Jing Lu, Yunlu Xu, Zhanzhan Cheng, Dashan Guo, Yi Niu, Xiangzhong Fang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01139](https://doi.org/10.1109/CVPR52729.2023.01139)

        **Abstract**:

        Few-Shot Class-Incremental Learning (FSCIL) aims to continually learn novel classes based on only few training samples, which poses a more challenging task than the well-studied Class-Incremental Learning (CIL) due to data scarcity. While knowledge distillation, a prevailing technique in CIL, can alleviate the catastrophic forgetting of older classes by regularizing outputs between current and previous model, it fails to consider the overfitting risk of novel classes in FSCIL. To adapt the powerful distillation technique for FSCIL, we propose a novel distillation structure, by taking the unique challenge of overfitting into account. Concretely, we draw knowledge from two complementary teachers. One is the model trained on abundant data from base classes that carries rich general knowledge, which can be leveraged for easing the overfitting of current novel classes. The other is the updated model from last incremental session that contains the adapted knowledge of previous novel classes, which is used for alleviating their forgetting. To combine the guidances, an adaptive strategy conditioned on the class-wise semantic similarities is introduced. Besides, for better preserving base class knowledge when accommodating novel concepts, we adopt a two-branch network with an attention-based aggregation module to dynamically merge predictions from two complementary branches. Extensive experiments on 3 popular FSCIL datasets: mini-ImageNet, CIFAR100 and CUB200 validate the effectiveness of our method by surpassing existing works by a significant margin. Code is available at https://github.com/LinglanZhao/BiDistFSCIL.

        ----

        ## [1128] Multi-Mode Online Knowledge Distillation for Self-Supervised Visual Representation Learning

        **Authors**: *Kaiyou Song, Jin Xie, Shan Zhang, Zimeng Luo*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01140](https://doi.org/10.1109/CVPR52729.2023.01140)

        **Abstract**:

        Self-supervised learning (SSL) has made remarkable progress in visual representation learning. Some studies combine SSL with knowledge distillation (SSL-KD) to boost the representation learning performance of small models. In this study, we propose a Multi-mode Online Knowledge Distillation method (MOKD) to boost self-supervised visual representation learning. Different from existing SSL-KD methods that transfer knowledge from a static pre-trained teacher to a student, in MOKD, two different models learn collaboratively in a self-supervised manner. Specifically, MOKD consists of two distillation modes: self-distillation and cross-distillation modes. Among them, self-distillation performs self-supervised learning for each model independently, while cross-distillation realizes knowledge interaction between different models. In cross-distillation, a cross-attention feature search strategy is proposed to enhance the semantic feature alignment between different models. As a result, the two models can absorb knowledge from each other to boost their representation learning performance. Extensive experimental results on different backbones and datasets demonstrate that two heterogeneous models can benefit from MOKD and outperform their independently trained baseline. In addition, MOKD also outperforms existing SSL-KD methods for both the student and teacher models.

        ----

        ## [1129] Dense Network Expansion for Class Incremental Learning

        **Authors**: *Zhiyuan Hu, Yunsheng Li, Jiancheng Lyu, Dashan Gao, Nuno Vasconcelos*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01141](https://doi.org/10.1109/CVPR52729.2023.01141)

        **Abstract**:

        The problem of class incremental learning (CIL) is considered. State-of-the-art approaches use a dynamic architecture based on network expansion (NE), in which a task expert is added per task. While effective from a computational standpoint, these methods lead to models that grow quickly with the number of tasks. A new NE method, dense network expansion (DNE), is proposed to achieve a better trade-off between accuracy and model complexity. This is accomplished by the introduction of dense connections between the intermediate layers of the task expert networks, that enable the transfer of knowledge from old to new tasks via feature sharing and reusing. This sharing is implemented with a cross-task attention mechanism, based on a new task attention block (TAB), that fuses information across tasks. Unlike traditional attention mechanisms, TAB operates at the level of the feature mixing and is decoupled with spatial attentions. This is shown more effective than a joint spatial-and-task attention for CIL. The proposed DNE approach can strictly maintain the feature space of old classes while growing the network and feature scale at a much slower rate than previous methods. In result, it outperforms the previous SOTA methods by a margin of 4% in terms of accuracy, with similar or even smaller model scale.

        ----

        ## [1130] Class Attention Transfer Based Knowledge Distillation

        **Authors**: *Ziyao Guo, Haonan Yan, Hui Li, Xiaodong Lin*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01142](https://doi.org/10.1109/CVPR52729.2023.01142)

        **Abstract**:

        Previous knowledge distillation methods have shown their impressive performance on model compression tasks, however, it is hard to explain how the knowledge they transferred helps to improve the performance of the student network. In this work, we focus on proposing a knowledge distillation method that has both high interpretability and competitive performance. We first revisit the structure of mainstream CNN models and reveal that possessing the capacity of identifying class discriminative regions of input is critical for CNN to perform classification. Furthermore, we demonstrate that this capacity can be obtained and enhanced by transferring class activation maps. Based on our findings, we propose class attention transfer based knowledge distillation (CAT-KD). Different from previous KD methods, we explore and present several properties of the knowledge transferred by our method, which not only improve the interpretability of CAT-KD but also contribute to a better understanding of CNN. While having high interpretability, CAT-KD achieves state-of-the-art performance on multiple benchmarks. Code is available at: https://github.com/GzyAftermath/CAT-KD.

        ----

        ## [1131] Dealing with Cross-Task Class Discrimination in Online Continual Learning

        **Authors**: *Yiduo Guo, Bing Liu, Dongyan Zhao*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01143](https://doi.org/10.1109/CVPR52729.2023.01143)

        **Abstract**:

        Existing continual learning (CL) research regards catastrophic forgetting (CF) as almost the only challenge. This paper argues for another challenge in class-incremental learning (CIL), which we call cross-task class discrimination (CTCD), i.e., how to establish decision boundaries between the classes of the new task and old tasks with no (or limited) access to the old task data. CTCD is implicitly and partially dealt with by replay-based methods. A replay method saves a small amount of data (replay data) from previous tasks. When a batch of current task data arrives, the system jointly trains the new data and some sampled replay data. The replay data enables the system to partially learn the decision boundaries between the new classes and the old classes as the amount of the saved data is small. However, this paper argues that the replay approach also has a dynamic training bias issue which reduces the effectiveness of the replay data in solving the CTCD problem. A novel optimization objective with a gradient-based adaptive method is proposed to dynamically deal with the problem in the online CL process. Experimental results show that the new method achieves much better results in online CL.

        ----

        ## [1132] Real-Time Evaluation in Online Continual Learning: A New Hope

        **Authors**: *Yasir Ghunaim, Adel Bibi, Kumail Alhamoud, Motasem Alfarra, Hasan Abed Al Kader Hammoud, Ameya Prabhu, Philip H. S. Torr, Bernard Ghanem*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01144](https://doi.org/10.1109/CVPR52729.2023.01144)

        **Abstract**:

        Current evaluations of Continual Learning (CL) methods typically assume that there is no constraint on training time and computation. This is an unrealistic assumption for any real-world setting, which motivates us to propose: a practical real-time evaluation of continual learning, in which the stream does not wait for the model to complete training before revealing the next data for predictions. To do this, we evaluate current CL methods with respect to their computational costs. We conduct extensive experiments on CLOC, a large-scale dataset containing 39 million time-stamped images with geolocation labels. We show that a simple baseline outperforms state-of-the-art CL methods under this evaluation, questioning the applicability of existing methods in realistic settings. In addition, we explore various CL components commonly used in the literature, including memory sampling strategies and regularization approaches. We find that all considered methods fail to be competitive against our simple baseline. This surprisingly suggests that the majority of existing CL literature is tailored to a specific class of streams that is not practical. We hope that the evaluation we provide will be the first step towards a paradigm shift to consider the computational cost in the development of online continual learning methods.

        ----

        ## [1133] DisWOT: Student Architecture Search for Distillation WithOut Training

        **Authors**: *Peijie Dong, Lujun Li, Zimian Wei*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01145](https://doi.org/10.1109/CVPR52729.2023.01145)

        **Abstract**:

        Knowledge distillation (KD) is an effective training strategy to improve the lightweight student models under the guidance of cumbersome teachers. However, the large architecture difference across the teacher-student pairs limits the distillation gains. In contrast to previous adaptive distillation methods to reduce the teacher-student gap, we explore a novel training-free framework to search for the best student architectures for a given teacher. Our work first empirically show that the optimal model under vanilla training cannot be the winner in distillation. Secondly, we find that the similarity of feature semantics and sample relations between random-initialized teacher-student networks have good correlations with final distillation performances. Thus, we efficiently measure similarity matrixs conditioned on the semantic activation maps to select the optimal student via an evolutionary algorithm without any training. In this way, our student architecture search for Distillation WithOut Training (DisWOT) significantly improves the performance of the model in the distillation stage with at least 180 × training acceleration. Additionally, we extend similarity metrics in DisWOT as new distillers and KD-based zero-proxies. Our experiments on CIFAR, ImageNet and NAS-Bench-201 demonstrate that our technique achieves state-of-the-art results on different search spaces. Our project and code are available at https://lilujunai.github.io/DisWot-CvpR20231.

        ----

        ## [1134] CODA-Prompt: COntinual Decomposed Attention-Based Prompting for Rehearsal-Free Continual Learning

        **Authors**: *James Seale Smith, Leonid Karlinsky, Vyshnavi Gutta, Paola Cascante-Bonilla, Donghyun Kim, Assaf Arbelle, Rameswar Panda, Rogério Feris, Zsolt Kira*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01146](https://doi.org/10.1109/CVPR52729.2023.01146)

        **Abstract**:

        Computer vision models suffer from a phenomenon known as catastrophic forgetting when learning novel concepts from continuously shifting training data. Typical solutions for this continual learning problem require extensive rehearsal of previously seen data, which increases memory costs and may violate data privacy. Recently, the emergence of large-scale pre-trained vision transformer models has enabled prompting approaches as an alternative to data-rehearsal. These approaches rely on a key-query mechanism to generate prompts and have been found to be highly resistant to catastrophic forgetting in the well-established rehearsal-free continual learning setting. However, the key mechanism of these methods is not trained end-to-end with the task sequence. Our experiments show that this leads to a reduction in their plasticity, hence sacrificing new task accuracy, and inability to benefit from expanded parameter capacity. We instead propose to learn a set of prompt components which are assembled with input-conditioned weights to produce input-conditioned prompts, resulting in a novel attention-based end-to-end key-query scheme. Our experiments show that we outperform the current SOTA method DualPrompt on established benchmarks by as much as 4.5% in average final accuracy. We also outperform the state of art by as much as 4.4% accuracy on a continual learning benchmark which contains both class-incremental and domain-incremental task shifts, corresponding to many practical settings. Our code is available at https://github.com/GT-RIPL/CODA-Prompt

        ----

        ## [1135] EcoTTA: Memory-Efficient Continual Test-Time Adaptation via Self-Distilled Regularization

        **Authors**: *Junha Song, Jungsoo Lee, In So Kweon, Sungha Choi*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01147](https://doi.org/10.1109/CVPR52729.2023.01147)

        **Abstract**:

        This paper presents a simple yet effective approach that improves continual test-time adaptation (TTA) in a memory-efficient manner. TTA may primarily be conducted on edge devices with limited memory, so reducing memory is crucial but has been overlooked in previous TTA studies. In addition, long-term adaptation often leads to catastrophic forgetting and error accumulation, which hinders applying TTA in real-world deployments. Our approach consists of two components to address these issues. First, we present lightweight meta networks that can adapt the frozen original networks to the target domain. This novel architecture minimizes memory consumption by decreasing the size of intermediate activations required for backpropagation. Second, our novel self-distilled regularization controls the output of the meta networks not to deviate significantly from the output of the frozen original networks, thereby preserving well-trained knowledge from the source domain. Without additional memory, this regularization prevents error accumulation and catastrophic forgetting, resulting in stable performance even in long-term test-time adaptation. We demonstrate that our simple yet effective strategy outperforms other state-of-the-art methods on various benchmarks for image classification and semantic segmentation tasks. Notably, our proposed method with ResNet-50 and WideResNet-40 takes 86% and 80% less memory than the recent state-of-the-art method, CoTTA.

        ----

        ## [1136] Achieving a Better Stability-Plasticity Trade-off via Auxiliary Networks in Continual Learning

        **Authors**: *Sanghwan Kim, Lorenzo Noci, Antonio Orvieto, Thomas Hofmann*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01148](https://doi.org/10.1109/CVPR52729.2023.01148)

        **Abstract**:

        In contrast to the natural capabilities of humans to learn new tasks in a sequential fashion, neural networks are known to suffer from catastrophic forgetting, where the model's performances on old tasks drop dramatically after being optimized for a new task. Since then, the continual learning (CL) community has proposed several solutions aiming to equip the neural network with the ability to learn the current task (plasticity) while still achieving high accuracy on the previous tasks (stability). Despite remarkable improvements, the plasticity-stability trade-off is still far from being solved and its underlying mechanism is poorly understood. In this work, we propose Auxiliary Network Continual Learning (ANCL), a novel method that applies an additional auxiliary network which promotes plasticity to the continually learned model which mainly focuses on stability. More concretely, the proposed framework materializes in a regularizer that naturally interpolates between plasticity and stability, surpassing strong baselines on task incremental and class incremental scenarios. Through extensive analyses on ANCL solutions, we identify some essential principles beneath the stability-plasticity tradeoff. The code implementation of our work is available at https://github.com/kim-sanghwan/ANCL.

        ----

        ## [1137] PA&DA: Jointly Sampling PAth and DAta for Consistent NAS

        **Authors**: *Shun Lu, Yu Hu, Longxing Yang, Zihao Sun, Jilin Mei, Jianchao Tan, Chengru Song*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01149](https://doi.org/10.1109/CVPR52729.2023.01149)

        **Abstract**:

        Based on the weight-sharing mechanism, one-shot NAS methods train a supernet and then inherit the pre-trained weights to evaluate sub-models, largely reducing the search cost. However, several works have pointed out that the shared weights suffer from different gradient descent directions during training. And we further find that large gradient variance occurs during supernet training, which degrades the supernet ranking consistency. To mitigate this issue, we propose to explicitly minimize the gradient variance of the supernet training by jointly optimizing the sampling distributions of PAth and DAta (PA&DA). We theoretically derive the relationship between the gradient variance and the sampling distributions, and reveal that the optimal sampling probability is proportional to the normalized gradient norm of path and training data. Hence, we use the normalized gradient norm as the importance indicator for path and training data, and adopt an importance sampling strategy for the supernet training. Our method only requires negligible computation cost for optimizing the sampling distributions of path and data, but achieves lower gradient variance during supernet training and better generalization performance for the supernet, resulting in a more consistent NAS. We conduct comprehensive comparisons with other improved approaches in various search spaces. Results show that our method surpasses others with more reliable ranking performance and higher accuracy of searched architectures, showing the effectiveness of our method. Code is available at https://github.com/ShunLu91/PA-DA.

        ----

        ## [1138] Accelerating Dataset Distillation via Model Augmentation

        **Authors**: *Lei Zhang, Jie Zhang, Bowen Lei, Subhabrata Mukherjee, Xiang Pan, Bo Zhao, Caiwen Ding, Yao Li, Dongkuan Xu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01150](https://doi.org/10.1109/CVPR52729.2023.01150)

        **Abstract**:

        Dataset Distillation (DD), a newly emerging field, aims at generating much smaller but efficient synthetic training datasets from large ones. Existing DD methods based on gradient matching achieve leading performance; however, they are extremely computationally intensive as they require continuously optimizing a dataset among thousands of randomly initialized models. In this paper, we assume that training the synthetic data with diverse models leads to better generalization performance. Thus we propose two model augmentation techniques, i.e. using early-stage models and parameter perturbation to learn an informative synthetic set with significantly reduced training cost. Extensive experiments demonstrate that our method achieves up to 20× speedup and comparable performance on par with state-of-the-art methods.

        ----

        ## [1139] Multi-Agent Automated Machine Learning

        **Authors**: *Zhaozhi Wang, Kefan Su, Jian Zhang, Huizhu Jia, Qixiang Ye, Xiaodong Xie, Zongqing Lu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01151](https://doi.org/10.1109/CVPR52729.2023.01151)

        **Abstract**:

        In this paper, we propose multi-agent automated machine learning (MA2ML) with the aim to effectively handle joint optimization of modules in automated machine learning (AutoML). MA2ML takes each machine learning module, such as data augmentation (AUG), neural architecture search (NAS), or hyper-parameters (HPO), as an agent and the final performance as the reward, to formulate a multi-agent reinforcement learning problem. MA2ML explicitly assigns credit to each agent according to its marginal contribution to enhance cooperation among modules, and incorporates off-policy learning to improve search efficiency. Theoretically, MA2ML guarantees monotonic improvement of joint optimization. Extensive experiments show that MA2ML yields the state-of-the-art top-1 accuracy on ImageNet under constraints of computational cost, e.g., 79.7%/80.5% with FLOPs fewer than 600M/800M. Exten\sive ablation studies verify the benefits of credit assignment and off-policy learning of MA2ML.

        ----

        ## [1140] Transformer-Based Learned Optimization

        **Authors**: *Erik Gärtner, Luke Metz, Mykhaylo Andriluka, C. Daniel Freeman, Cristian Sminchisescu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01152](https://doi.org/10.1109/CVPR52729.2023.01152)

        **Abstract**:

        We propose a new approach to learned optimization where we represent the computation of an optimizer's update step using a neural network. The parameters of the optimizer are then learned by training on a set of optimization tasks with the objective to perform minimization efficiently. Our innovation is a new neural network architecture, Optimus, for the learned optimizer inspired by the classic BFGS algorithm. As in BFGS, we estimate a preconditioning matrix as a sum of rank-one updates but use a Transformerbased neural network to predict these updates jointly with the step length and direction. In contrast to several recent learned optimization-based approaches [24, 27], our formulation allows for conditioning across the dimensions of the parameter space of the target problem while remaining applicable to optimization tasks of variable dimensionality without retraining. We demonstrate the advantages of our approach on a benchmark composed of objective functions traditionally used for the evaluation of optimization algorithms, as well as on the real world-task of physics-based visual reconstruction of articulated 3d human motion.

        ----

        ## [1141] Solving Relaxations of MAP-MRF Problems: Combinatorial in-Face Frank-Wolfe Directions

        **Authors**: *Vladimir Kolmogorov*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01153](https://doi.org/10.1109/CVPR52729.2023.01153)

        **Abstract**:

        We consider the problem of solving LP relaxations of MAP-MRF inference problems, and in particular the method proposed recently in [16], [35]. As a key computational subroutine, it uses a variant of the Frank-Wolfe (FW) method to minimize a smooth convex function over a combinatorial polytope. We propose an efficient implementation of this subroutine based on in-face Frank-Wolfe directions, introduced in [4] in a different context. More generally, we define an abstract data structure for a combinatorial subproblem that enables in-face FW directions, and describe its specialization for tree-structured MAP-MRF inference subproblems. Experimental results indicate that the resulting method is the current state-of-art LP solver for some classes of problems. Our code is available at pub.ist.ac.at/~vnk/papers/IN-FACE-FW.html.

        ----

        ## [1142] HOTNAS: Hierarchical Optimal Transport for Neural Architecture Search

        **Authors**: *Jiechao Yang, Yong Liu, Hongteng Xu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01154](https://doi.org/10.1109/CVPR52729.2023.01154)

        **Abstract**:

        Instead of searching the entire network directly, current NAS approaches increasingly search for multiple relatively small cells to reduce search costs. A major challenge is to jointly measure the similarity of cell micro-architectures and the difference in macro-architectures between different cell-based networks. Recently, optimal transport (OT) has been successfully applied to NAS as it can capture the operational and structural similarity across various networks. However, existing OT-based NAS methods either ignore the cell similarity or focus solely on searching for a single cell architecture. To address these issues, we propose a hierarchical optimal transport metric called HOTNN for measuring the similarity of different networks. In HOTNN, the cell-level similarity computes the OT distance between cells in various networks by considering the similarity of each node and the differences in the information flow costs between node pairs within each cell in terms of operational and structural information. The network-level similarity calculates OT distance between networks by considering both the cell-level similarity and the variation in the global position of each cell within their respective networks. We then explore HOTNN in a Bayesian optimization framework called HOTNAS, and demonstrate its efficacy in diverse tasks. Extensive experiments demonstrate that H OT-NAS can discover network architectures with better performance in multiple modular cell-based search spaces.

        ----

        ## [1143] Disentangled Representation Learning for Unsupervised Neural Quantization

        **Authors**: *Haechan Noh, Sangeek Hyun, Woojin Jeong, Hanshin Lim, Jae-Pil Heo*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01155](https://doi.org/10.1109/CVPR52729.2023.01155)

        **Abstract**:

        The inverted index is a widely used data structure to avoid the infeasible exhaustive search. It accelerates retrieval significantly by splitting the database into multiple disjoint sets and restricts distance computation to a small fraction of the database. Moreover, it even improves search quality by allowing quantizers to exploit the compact distribution of residual vector space. However, we firstly point out a problem that an existing deep learning-based quantizer hardly benefits from the residual vector space, unlike conventional shallow quantizers. To cope with this problem, we introduce a novel disentangled representation learning for unsupervised neural quantization. Similar to the concept of residual vector space, the proposed method enables more compact latent space by disentangling information of the inverted index from the vectors. Experimental results on large-scale datasets confirm that our method outperforms the state-of-the-art retrieval systems by a large margin.

        ----

        ## [1144] FFCV: Accelerating Training by Removing Data Bottlenecks

        **Authors**: *Guillaume Leclerc, Andrew Ilyas, Logan Engstrom, Sung Min Park, Hadi Salman, Aleksander Madry*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01156](https://doi.org/10.1109/CVPR52729.2023.01156)

        **Abstract**:

        We present FFCV, a library for easy and fast machine learning model training. FFCV speeds up model training by eliminating (often subtle) data bottlenecks from the training process. In particular, we combine techniques such as an efficient file storage format, caching, data pre-loading, asynchronous data transfer, and just-in-time compilation to (a) make data loading and transfer significantly more efficient, ensuring that GPUs can reach full utilization; and (b) offload as much data processing as possible to the CPU asynchronously, freeing GPU cycles for training. Using FFCV, we train ResNet-18 and ResNet-50 on the ImageNet dataset with a state-of-the-art tradeoff between accuracy and training time. For example, across the range of ResNet-50 models we test, we obtain the same accuracy as the best baselines in half the time. We demonstrate FFCV's performance, ease-of-use, extensibility, and ability to adapt to resource constraints through several case studies. Detailed installation instructions, documentation, and Slack support channel are available at https://ffcv.io/.

        ----

        ## [1145] Run, Don't Walk: Chasing Higher FLOPS for Faster Neural Networks

        **Authors**: *Jierun Chen, Shiu-hong Kao, Hao He, Weipeng Zhuo, Song Wen, Chul-Ho Lee, S.-H. Gary Chan*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01157](https://doi.org/10.1109/CVPR52729.2023.01157)

        **Abstract**:

        To design fast neural networks, many works have been focusing on reducing the number of floating-point operations (FLOPs). We observe that such reduction in FLOPs, however, does not necessarily lead to a similar level of re-duction in latency. This mainly stems from inefficiently low floating-point operations per second (FLOPS). To achieve faster networks, we revisit popular operators and demonstrate that such low FLOPS is mainly due to frequent memory access of the operators, especially the depthwise con-volution. We hence propose a novel partial convolution (PConv) that extracts spatial features more efficiently, by cutting down redundant computation and memory access simultaneously. Building upon our PConv, we further propose FasterNet, a new family of neural networks, which attains substantially higher running speed than others on a wide range of devices, without compromising on accuracy for various vision tasks. For example, on ImageNet-lk, our tiny FasterNet-TO is 2.8×, 3.3×, and 2.4× faster than MobileViT-XXS on GPU, CPU, and ARM processors, respectively, while being 2.9% more accurate. Our large FasterNet-L achieves impressive 83.5% top-1 accuracy, on par with the emerging Swin-B, while having 36% higher inference throughput on GPU, as well as saving 37% compute time on CPU. Code is available at https://github.com/JierunChen/FasterNet.

        ----

        ## [1146] Demystifying Causal Features on Adversarial Examples and Causal Inoculation for Robust Network by Adversarial Instrumental Variable Regression

        **Authors**: *Junho Kim, Byung-Kwan Lee, Yong Man Ro*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01184](https://doi.org/10.1109/CVPR52729.2023.01184)

        **Abstract**:

        The origin of adversarial examples is still inexplicable in research fields, and it arouses arguments from various view-points, albeit comprehensive investigations. In this paper, we propose a way of delving into the unexpected vulnerability in adversarially trained networks from a causal perspective, namely adversarial instrumental variable (IV) regression. By deploying it, we estimate the causal relation of adversarial prediction under an unbiased environment dissociated from unknown confounders. Our approach aims to demystify inherent causal features on adversarial examples by leveraging a zero-sum optimization game between a casual feature estimator (i.e., hypothesis model) and worst-case counterfactuals (i.e., test function) disturbing to find causal features. Through extensive analyses, we demonstrate that the estimated causal features are highly related to the correct prediction for adversarial robustness, and the counterfactuals exhibit extreme features significantly deviating from the correct prediction. In addition, we present how to effectively inoculate CAusal FEatures (CAFE) into defense networks for improving adversarial robustness.

        ----

        ## [1147] FIANCEE: Faster Inference of Adversarial Networks via Conditional Early Exits

        **Authors**: *Polina Karpikova, Ekaterina Radionova, Anastasia Yaschenko, Andrei Spiridonov, Leonid Kostyushko, Riccardo Fabbricatore, Aleksei Ivakhnenko*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01158](https://doi.org/10.1109/CVPR52729.2023.01158)

        **Abstract**:

        Generative DNNs are a powerful tool for image synthesis, but they are limited by their computational load. On the other hand, given a trained model and a task, e.g. faces generation within a range of characteristics, the output image quality will be unevenly distributed among images with different characteristics. It follows, that we might restrain the model's complexity on some instances, maintaining a high quality. We propose a method for diminishing computations by adding so-called early exit branches to the original architecture, and dynamically switching the computational path depending on how difficult it will be to render the output. We apply our method on two different SOTA models performing generative tasks: generation from a semantic map, and cross-reenactment of face expressions; showing it is able to output images with custom lower-quality thresholds. For a threshold of LPIPS ≤ 0.1, we diminish their computations by up to a half. This is especially relevant for real-time applications such as synthesis of faces, when quality loss needs to be contained, but most of the inputs need fewer computations than the complex instances.

        ----

        ## [1148] Gradient-based Uncertainty Attribution for Explainable Bayesian Deep Learning

        **Authors**: *Hanjing Wang, Dhiraj Joshi, Shiqiang Wang, Qiang Ji*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01159](https://doi.org/10.1109/CVPR52729.2023.01159)

        **Abstract**:

        Predictions made by deep learning models are prone to data perturbations, adversarial attacks, and out-of-distribution inputs. To build a trusted AI system, it is there-fore critical to accurately quantify the prediction uncertainties. While current efforts focus on improving uncertainty quantification accuracy and efficiency, there is a need to identify uncertainty sources and take actions to mitigate their effects on predictions. Therefore, we propose to develop explainable and actionable Bayesian deep learning methods to not only perform accurate uncertainty quantification but also explain the uncertainties, identify their sources, and propose strategies to mitigate the uncertainty impacts. Specifically, we introduce a gradient-based uncertainty attribution method to identify the most problematic regions of the input that contribute to the prediction uncertainty. Compared to existing methods, the proposed UA-Backprop has competitive accuracy, relaxed assumptions, and high efficiency. Moreover, we propose an uncertainty mitigation strategy that leverages the attribution results as attention to further improve the model performance. Both qualitative and quantitative evaluations are conducted to demonstrate the effectiveness of our proposed methods.

        ----

        ## [1149] How to Prevent the Continuous Damage of Noises to Model Training?

        **Authors**: *Xiaotian Yu, Yang Jiang, Tianqi Shi, Zunlei Feng, Yuexuan Wang, Mingli Song, Li Sun*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01160](https://doi.org/10.1109/CVPR52729.2023.01160)

        **Abstract**:

        Deep learning with noisy labels is challenging and inevitable in many circumstances. Existing methods reduce the impact of mislabeled samples by reducing loss weights or screening, which highly rely on the model's superior discriminative power for identifying mislabeled samples. However, in the training stage, the trainee model is imperfect and will wrongly predict some mislabeled samples, which cause continuous damage to the model training. Consequently, there is a large performance gap between existing anti-noise models trained with noisy samples and models trained with clean samples. In this paper, we put forward a Gradient Switching Strategy (GSS) to prevent the continuous damage of mislabeled samples to the classifier. Theoretical analysis shows that the damage comes from the misleading gradient direction computed from the mislabeled samples. The trainee model will deviate from the correct optimization direction under the influence of the accumulated misleading gradient of mislabeled samples. To address this problem, the proposed GSS alleviates the damage by switching the gradient direction of each sample based on the gradient direction pool, which contains all-class gradient directions with different probabilities. During training, each gradient direction pool is updated iteratively, which assigns higher probabilities to potential principal directions for high-confidence samples. Conversely, uncertain samples are forced to explore in different directions rather than mislead model in a fixed direction. Extensive experiments show that GSS can achieve comparable performance with a model trained with clean data. Moreover, the proposed GSS is pluggable for existing frameworks. This idea of switching gradient directions provides a new perspective for future noisy-label learning.

        ----

        ## [1150] Genie: Show Me the Data for Quantization

        **Authors**: *Yongkweon Jeon, Chungman Lee, Ho-Young Kim*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01161](https://doi.org/10.1109/CVPR52729.2023.01161)

        **Abstract**:

        Zero-shot quantization is a promising approach for developing lightweight deep neural networks when data is inaccessible owing to various reasons, including cost and issues related to privacy. By exploiting the learned parameters (
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$\mu$</tex>
 and 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$\sigma$</tex>
) of batch normalization layers in an FP32-pretrained model, zero-shot quantization schemes focus on generating synthetic data. Subsequently, they distill knowledge from the pre-trained model (teacher) to the quantized model (student) such that the quantized model can be optimized with the synthetic dataset. However, thus far, zeroshot quantization has primarily been discussed in the context of quantization-aware training methods, which require task-specific losses and long-term optimization as much as retraining. We thus introduce a post-training quantization scheme for zero-shot quantization that produces high-quality quantized networks within a few hours. Furthermore, we propose a framework called Genie that generates data suited for quantization. With the data synthesized by Genie, we can produce robust quantized models without real datasets, which is comparable to few-shot quantization. We also propose a post-training quantization algorithm to enhance the performance of quantized models. By combining them, we can bridge the gap between zero-shot and few-shot quantization while significantly improving the quantization performance compared to that of existing approaches. In other words, we can obtain a unique state-of-the-art zero-shot quantization approach. The code is available at https://github.com/SamsungLabs/Genie.

        ----

        ## [1151] OpenMix: Exploring Outlier Samples for Misclassification Detection

        **Authors**: *Fei Zhu, Zhen Cheng, Xu-Yao Zhang, Cheng-Lin Liu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01162](https://doi.org/10.1109/CVPR52729.2023.01162)

        **Abstract**:

        Reliable confidence estimation for deep neural classifiers is a challenging yet fundamental requirement in high-stakes applications. Unfortunately, modern deep neural networks are often overconfident for their erroneous predictions. In this work, we exploit the easily available outlier samples, i.e., unlabeled samples coming from non-target classes, for helping detect misclassification errors. Particularly, we find that the well-known Outlier Exposure, which is powerful in detecting out-of-distribution (OOD) samples from unknown classes, does not provide any gain in identifying misclassification errors. Based on these observations, we propose a novel method called OpenMix, which incorporates open-world knowledge by learning to reject uncertain pseudo-samples generated via outlier transformation. OpenMix significantly improves confidence reliability under various scenarios, establishing a strong and unified framework for detecting both misclassified samples from known classes and OOD samples from unknown classes. The code is publicly available at https://github.com/Impression2805/OpenMix.

        ----

        ## [1152] Data-Free Sketch-Based Image Retrieval

        **Authors**: *Abhra Chaudhuri, Ayan Kumar Bhunia, Yi-Zhe Song, Anjan Dutta*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01163](https://doi.org/10.1109/CVPR52729.2023.01163)

        **Abstract**:

        Rising concerns about privacy and anonymity preservation of deep learning models have facilitated research in data-free learning (DFL). For the first time, we identify that for data-scarce tasks like Sketch-Based Image Retrieval (SBIR), where the difficulty in acquiring paired photos and hand-drawn sketches limits data-dependent cross-modal learning algorithms, DFL can prove to be a much more practical paradigm. We thus propose Data-Free (DF)-SBIR, where, unlike existing DFL problems, pre-trained, single-modality classification models have to be leveraged to learn a cross-modal metric-space for retrieval without access to any training data. The widespread availability of pre-trained classification models, along with the difficulty in acquiring paired photo-sketch datasets for SBIR justify the practicality of this setting. We present a methodology for DF-SBIR, which can leverage knowledge from models independently trained to perform classification on photos and sketches. We evaluate our model on the Sketchy, TU-Berlin, and QuickDraw benchmarks, designing a variety of baselines based on state-of-the-art DFL literature, and observe that our method surpasses all of them by significant margins. Our method also achieves mAPs competitive with data-dependent approaches, all the while requiring no training data. Implementation is available at https://github.com/abhrac/data-free-sbir.

        ----

        ## [1153] GLeaD: Improving GANs with A Generator-Leading Task

        **Authors**: *Qingyan Bai, Ceyuan Yang, Yinghao Xu, Xihui Liu, Yujiu Yang, Yujun Shen*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01164](https://doi.org/10.1109/CVPR52729.2023.01164)

        **Abstract**:

        Generative adversarial network (GAN) is formulated as a two-player game between a generator (G) and a discriminator (D), where D is asked to differentiate whether an image comes from real data or is produced by G. Under such a formulation, D plays as the rule maker and hence tends to dominate the competition. Towards a fairer game in GANs, we propose a new paradigm for adversarial training, which makes G assign a task to D as well. Specifically, given an image, we expect D to extract representative features that can be adequately decoded by G to reconstruct the input. That way, instead of learning freely, D is urged to align with the view of G for domain classification. Experimental results on various datasets demonstrate the substantial superiority of our approach over the baselines. For instance, we improve the FID of StyleGAN2 from 4.30 to 2.55 on LSUN Bedroom and from 4.04 to 2.82 on LSUN Church. We believe that the pioneering attempt present in this work could inspire the community with better designed generator-leading tasks for GAN improvement. Project page is at https://ezioby.github.io/glead/.

        ----

        ## [1154] Learning on Gradients: Generalized Artifacts Representation for GAN-Generated Images Detection

        **Authors**: *Chuangchuang Tan, Yao Zhao, Shikui Wei, Guanghua Gu, Yunchao Wei*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01165](https://doi.org/10.1109/CVPR52729.2023.01165)

        **Abstract**:

        Recently, there has been a significant advancement in image generation technology, known as GAN. It can easily generate realistic fake images, leading to an increased risk of abuse. However, most image detectors suffer from sharp performance drops in unseen domains. The key of fake image detection is to develop a generalized representation to describe the artifacts produced by generation models. In this work, we introduce a novel detection framework, named Learning on Gradients (LGrad), designed for identifying GAN-generated images, with the aim of constructing a generalized detector with cross-model and cross-data. Specifically, a pretrained CNN model is employed as a transformation model to convert images into gradients. Subsequently, we leverage these gradients to present the generalized artifacts, which are fed into the classifier to ascertain the authenticity of the images. In our framework, we turn the data-dependent problem into a transformation-model-dependent problem. To the best of our knowledge, this is the first study to utilize gradients as the representation of artifacts in GAN-generated images. Extensive experiments demonstrate the effectiveness and robustness of gradients as generalized artifact representations. Our detector achieves a new state-of-the-art performance with a remarkable gain of 11.4%. The code is released at https://github.com/chuangchuangtan/LGrad.

        ----

        ## [1155] Adversarial Normalization: I Can visualize Everything (ICE)

        **Authors**: *Hoyoung Choi, Seungwan Jin, Kyungsik Han*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01166](https://doi.org/10.1109/CVPR52729.2023.01166)

        **Abstract**:

        Vision transformers use [CLS] tokens to predict image classes. Their explainability visualization has been studied using relevant information from [CLS] tokens or focusing on attention scores during self-attention. Such visualization, however, is challenging because of the dependence of the structure of a vision transformer on skip connections and attention operators, the instability of non-linearities in the learning process, and the limited reflection of self-attention scores on relevance. We argue that the output vectors for each input patch token in a vision transformer retain the image information of each patch location, which can facilitate the prediction of an image class. In this paper, we propose ICE (Adversarial Normalization: I Can visualize Everything), a novel method that enables a model to directly predict a class for each patch in an image; thus, advancing the effective visualization of the explainability of a vision transformer. Our method distinguishes background from foreground regions by predicting background classes for patches that do not determine image classes. We used the DeiT-S model, the most representative model employed in studies, on the explainability visualization of vision transformers. On the ImageNet-Segmentation dataset, ICE outperformed all explainability visualization methods for four cases depending on the model size. We also conducted quantitative and qualitative analyses on the tasks of weakly-supervised object localization and unsupervised object discovery. On the CUB-200-2011 and PASCALVOC07/12 datasets, ICE achieved comparable performance to the state-of-the-art methods. We incorporated ICE into the encoder of DeiT-S and improved efficiency by 44.01% on the ImageNet dataset over that achieved by the original DeiT-S model. We showed performance on the accuracy and efficiency comparable to EViT, the state-of-the-art pruning model, demonstrating the effectiveness of ICE. The code is available at https://github.com/Hanyang-HCC-Lab/ICE.

        ----

        ## [1156] Semi-Supervised Hand Appearance Recovery via Structure Disentanglement and Dual Adversarial Discrimination

        **Authors**: *Zimeng Zhao, Binghui Zuo, Zhiyu Long, Yangang Wang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01167](https://doi.org/10.1109/CVPR52729.2023.01167)

        **Abstract**:

        Enormous hand images with reliable annotations are collected through marker-based MoCap. Unfortunately, degradations caused by markers limit their application in hand appearance reconstruction. A clear appearance recovery insight is an image-to-image translation trained with unpaired data. However, most frameworks fail because there exists structure inconsistency from a degraded hand to a bare one. The core of our approach is to first disentangle the bare hand structure from those degraded images and then wrap the appearance to this structure with a dual adversarial discrimination (DAD) scheme. Both modules take full advantage of the semi-supervised learning paradigm: The structure disentanglement benefits from the modeling ability of ViT, and the translator is enhanced by the dual discrimination on both translation processes and translation results. Comprehensive evaluations have been conducted to prove that our framework can robustly recover photo-realistic hand appearance from diverse marker-contained and even object-occluded datasets. It provides a novel avenue to acquire bare hand appearance data for other down-stream learning problems.

        ----

        ## [1157] Look Around for Anomalies: Weakly-Supervised Anomaly Detection via Context-Motion Relational Learning

        **Authors**: *MyeongAh Cho, Minjung Kim, Sangwon Hwang, Chaewon Park, Kyungjae Lee, Sangyoun Lee*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01168](https://doi.org/10.1109/CVPR52729.2023.01168)

        **Abstract**:

        Weakly-supervised Video Anomaly Detection is the task of detecting frame-level anomalies using video-level labeled training data. It is difficult to explore class representative features using minimal supervision of weak labels with a single backbone branch. Furthermore, in real-world scenarios, the boundary between normal and abnormal is ambiguous and varies depending on the situation. For example, even for the same motion of running person, the abnormality varies depending on whether the surroundings are a playground or a roadway. Therefore, our aim is to extract discriminative features by widening the relative gap between classes' features from a single branch. In the proposed Class-Activate Feature Learning (CLAV), the features are extracted as per the weights that are implicitly activated depending on the class, and the gap is then enlarged through relative distance learning. Furthermore, as the relationship between context and motion is important in order to identify the anomalies in complex and diverse scenes, we propose a Context-Motion Interrelation Module (CoMo), which models the relationship between the appearance of the surroundings and motion, rather than utilizing only temporal dependencies or motion information. The proposed method shows SOTA performance on four benchmarks including large-scale real-world datasets, and we demonstrate the importance of relational information by analyzing the qualitative results and generalization ability.

        ----

        ## [1158] Diversity-Measurable Anomaly Detection

        **Authors**: *Wenrui Liu, Hong Chang, Bingpeng Ma, Shiguang Shan, Xilin Chen*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01169](https://doi.org/10.1109/CVPR52729.2023.01169)

        **Abstract**:

        Reconstruction-based anomaly detection models achieve their purpose by suppressing the generalization ability for anomaly. However, diverse normal patterns are consequently not well reconstructed as well. Although some efforts have been made to alleviate this problem by modeling sample diversity, they suffer from shortcut learning due to undesired transmission of abnormal information. In this paper, to better handle the tradeoff problem, we propose Diversity-Measurable Anomaly Detection (DMAD) framework to enhance reconstruction diversity while avoid the undesired generalization on anomalies. To this end, we design Pyramid Deformation Module (PDM), which models diverse normals and measures the severity of anomaly by estimating multi-scale deformation fields from reconstructed reference to original input. Integrated with an information compression module, PDM essentially decouples deformation from prototypical embedding and makes the final anomaly score more reliable. Experimental results on both surveillance videos and industrial images demonstrate the effectiveness of our method. In addition, DMAD works equally well in front of contaminated data and anomaly-like normal samples.

        ----

        ## [1159] Cloud-Device Collaborative Adaptation to Continual Changing Environments in the Real-World

        **Authors**: *Yulu Gan, Mingjie Pan, Rongyu Zhang, Zijian Ling, Lingran Zhao, Jiaming Liu, Shanghang Zhang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01170](https://doi.org/10.1109/CVPR52729.2023.01170)

        **Abstract**:

        When facing changing environments in the real world, the lightweight model on client devices suffers from severe performance drops under distribution shifts. The main limitations of the existing device model lie in (1) unable to update due to the computation limit of the device, (2) the limited generalization ability of the lightweight model. Meanwhile, recent large models have shown strong generalization capability on the cloud while they can not be deployed on client devices due to poor computation constraints. To enable the device model to deal with changing environments, we propose a new learning paradigm of Cloud-Device Collaborative Continual Adaptation, which encourages collaboration between cloud and device and improves the generalization of the device model. Based on this paradigm, we further propose an Uncertainty-based Visual Prompt Adapted (U-VPA) teacher-student model to transfer the generalization capability of the large model on the cloud to the device model. Specifically, we first design the Uncertainty Guided Sampling (UGS) to screen out challenging data continuously and transmit the most out-of-distribution samples from the device to the cloud. Then we propose a Visual Prompt Learning Strategy with Uncertainty guided updating (VPLU) to specifically deal with the selected samples with more distribution shifts. We transmit the visual prompts to the device and concatenate them with the incoming data to pull the device testing distribution closer to the cloud training distribution. We conduct extensive experiments on two object detection datasets with continually changing environments. Our proposed U-VPA teacher-student framework outperforms previous state-of-the-art test time adaptation and device-cloud collaboration methods. The code and datasets will be released.

        ----

        ## [1160] How to Prevent the Poor Performance Clients for Personalized Federated Learning?

        **Authors**: *Zhe Qu, Xingyu Li, Xiao Han, Rui Duan, Chengchao Shen, Lixing Chen*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01171](https://doi.org/10.1109/CVPR52729.2023.01171)

        **Abstract**:

        Personalized federated learning (pFL) collaboratively trains personalized models, which provides a customized model solution for individual clients in the presence of heterogeneous distributed local data. Although many recent studies have applied various algorithms to enhance personalization in pFL, they mainly focus on improving the performance from averaging or top perspective. However, part of the clients may fall into poor performance and are not clearly discussed. Therefore, how to prevent these poor clients should be considered critically. Intuitively, these poor clients may come from biased universal information shared with others. To address this issue, we propose a novel pFL strategy, called Personalize Locally, Generalize Universally (PLGU). PLGU generalizes the fine-grained universal information and moderates its biased performance by designing a Layer-Wised Sharpness Aware Minimization (LWSAM) algorithm while keeping the personalization local. Specifically, we embed our proposed PLGU strategy into two pFL schemes concluded in this paper: with/without a global model, and present the training procedures in detail. Through in-depth study, we show that the proposed PLGU strategy achieves competitive generalization bounds on both considered pFL schemes. Our extensive experimental results show that all the proposed PLGU based-algorithms achieve state-of-the-art performance.

        ----

        ## [1161] DYNAFED: Tackling Client Data Heterogeneity with Global Dynamics

        **Authors**: *Renjie Pi, Weizhong Zhang, Yueqi Xie, Jiahui Gao, Xiaoyu Wang, Sunghun Kim, Qifeng Chen*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01172](https://doi.org/10.1109/CVPR52729.2023.01172)

        **Abstract**:

        The Federated Learning (FL) paradigm is known to face challenges under heterogeneous client data. Local training on non-iid distributed data results in deflected local optimum, which causes the client models drift further away from each other and degrades the aggregated global model's performance. A natural solution is to gather all client data onto the server, such that the server has a global view of the entire data distribution. Unfortunately, this reduces to regular training, which compromises clients' privacy and conflicts with the purpose of FL. In this paper, we put forth an idea to collect and leverage global knowledge on the server without hindering data privacy. We unearth such knowledge from the dynamics of the global model's trajectory. Specifically, we first reserve a short trajectory of global model snapshots on the server. Then, we synthesize a small pseudo dataset such that the model trained on it mimics the dynamics of the reserved global model trajectory. Afterward, the synthesized data is used to help aggregate the deflected clients into the global model. We name our method Dynafed, which enjoys the following advantages: 1) we do not rely on any external on-server dataset, which requires no additional cost for data collection; 2) the pseudo data can be synthesized in early communication rounds, which enables Dyna fedto take effect early for boosting the convergence and stabilizing training; 3) the pseudo data only needs to be synthesized once and can be directly utilized on the server to help aggregation in subsequent rounds. Experiments across extensive benchmarks are conducted to showcase the effectiveness of Dyna fed. We also provide insights and understanding of the underlying mechanism of our method.

        ----

        ## [1162] Elastic Aggregation for Federated Optimization

        **Authors**: *Dengsheng Chen, Jie Hu, Vince Junkai Tan, Xiaoming Wei, Enhua Wu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01173](https://doi.org/10.1109/CVPR52729.2023.01173)

        **Abstract**:

        Federated learning enables the privacy-preserving training of neural network models using real-world data across distributed clients. FedAvg has become the preferred optimizer for federated learning because of its simplicity and effectiveness. FedAvg uses naïve aggregation to update the server model, interpolating client models based on the number of instances used in their training. However, naïve aggregation suffers from client drift when the data is heterogenous (non-IID), leading to unstable and slow convergence. In this work, we propose a novel aggregation approach, elastic aggregation, to overcome these issues. Elastic aggregation interpolates client models adaptively according to parameter sensitivity, which is measured by computing how much the overall prediction function output changes when each parameter is changed. This measurement is performed in an unsupervised and online manner. Elastic aggregation reduces the magnitudes of updates to the more sensitive parameters so as to prevent the server model from drifting to any one client distribution, and conversely boosts updates to the less sensitive parameters to better explore different client distributions. Empirical results on real and synthetic data as well as analytical results show that elastic aggregation leads to efficient training in both convex and nonconvex settings while being fully agnostic to client heterogeneity and robust to large numbers of clients, partial participation, and imbalanced data. Finally, elastic aggregation works well with other federated optimizers and achieves significant improvements across the board.

        ----

        ## [1163] Breaching FedMD: Image Recovery via Paired-Logits Inversion Attack

        **Authors**: *Hideaki Takahashi, Jingjing Liu, Yang Liu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01174](https://doi.org/10.1109/CVPR52729.2023.01174)

        **Abstract**:

        Federated Learning with Model Distillation (FedMD) is a nascent collaborative learning paradigm, where only output logits of public datasets are transmitted as distilled knowledge, instead of passing on private model parameters that are susceptible to gradient inversion attacks, a known privacy risk in federated learning. In this paper, we found that even though sharing output logits of public datasets is safer than directly sharing gradients, there still exists a sub-stantial risk of data exposure caused by carefully designed malicious attacks. Our study shows that a malicious server can inject a PLI (Paired-Logits Inversion) attack against FedMD and its variants by training an inversion neural network that exploits the confidence gap between the server and client models. Experiments on multiple facial recognition datasets validate that under FedMD-like schemes, by using paired server-client logits of public datasets only, the malicious server is able to reconstruct private images on all tested benchmarks with a high success rate.

        ----

        ## [1164] Learning to Measure the Point Cloud Reconstruction Loss in a Representation Space

        **Authors**: *Tianxin Huang, Zhonggan Ding, Jiangning Zhang, Ying Tai, Zhenyu Zhang, Mingang Chen, Chengjie Wang, Yong Liu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01175](https://doi.org/10.1109/CVPR52729.2023.01175)

        **Abstract**:

        For point cloud reconstruction-related tasks, the reconstruction losses to evaluate the shape differences between reconstructed results and the ground truths are typically used to train the task networks. Most existing works measure the training loss with point-to-point distance, which may introduce extra defects as predefined matching rules may deviate from the real shape differences. Although some learning-based works have been proposed to overcome the weaknesses of manually-defined rules, they still measure the shape differences in 3D Euclidean space, which may limit their ability to capture defects in reconstructed shapes. In this work, we propose a learning-based Contrastive Adver-sarial Loss (CALoss) to measure the point cloud reconstruction loss dynamically in a non-linear representation space by combining the contrastive constraint with the adversarial strategy. Specifically, we use the contrastive constraint to help CALoss learn a representation space with shape similarity, while we introduce the adversarial strategy to help CALoss mine differences between reconstructed results and ground truths. According to experiments on reconstruction-related tasks, CALoss can help task networks improve re-construction performances and learn more representative representations.

        ----

        ## [1165] Backdoor Cleansing with Unlabeled Data

        **Authors**: *Lu Pang, Tao Sun, Haibin Ling, Chao Chen*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01176](https://doi.org/10.1109/CVPR52729.2023.01176)

        **Abstract**:

        Due to the increasing computational demand of Deep Neural Networks (DNNs), companies and organizations have begun to outsource the training process. However, the externally trained DNNs can potentially be backdoor attacked. It is crucial to defend against such attacks, i.e., to postprocess a suspicious model so that its backdoor behavior is mitigated while its normal prediction power on clean inputs remain uncompromised. To remove the abnormal backdoor behavior, existing methods mostly rely on additional labeled clean samples. However, such requirement may be unrealistic as the training data are often unavailable to end users. In this paper, we investigate the possibility of circumventing such barrier. We propose a novel defense method that does not require training labels. Through a carefully designed layer-wise weight reinitialization and knowledge distillation, our method can effectively cleanse backdoor behaviors of a suspicious network with negligible compromise in its normal behavior. In experiments, we show that our method, trained without labels, is on-par with state-of-the-art defense methods trained using labels. We also observe promising defense results even on out-of-distribution data. This makes our method very practical. Code is available at: https://github.com/luluppang/BCU.

        ----

        ## [1166] Backdoor Defense via Deconfounded Representation Learning

        **Authors**: *Zaixi Zhang, Qi Liu, Zhicai Wang, Zepu Lu, Qingyong Hu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01177](https://doi.org/10.1109/CVPR52729.2023.01177)

        **Abstract**:

        Deep neural networks (DNNs) are recently shown to be vulnerable to backdoor attacks, where attackers embed hidden backdoors in the DNN model by injecting a few poisoned examples into the training dataset. While extensive efforts have been made to detect and remove backdoors from backdoored DNNs, it is still not clear whether a backdoor-free clean model can be directly obtained from poisoned datasets. In this paper, we first construct a causal graph to model the generation process of poisoned data and find that the backdoor attack acts as the confounder, which brings spurious associations between the input images and target labels, making the model predictions less reliable. Inspired by the causal understanding, we propose the Causality-inspired Backdoor Defense (CBD), to learn deconfounded representations for reliable classification. Specifically, a backdoored model is intentionally trained to capture the confounding effects. The other clean model dedicates to capturing the desired causal effects by minimizing the mutual information with the confounding representations from the backdoored model and employing a sample-wise re-weighting scheme. Extensive experiments on multiple benchmark datasets against 6 state-of-the-art attacks verify that our proposed defense method is effective in reducing backdoor threats while maintaining high accuracy in predicting benign samples. Further analysis shows that CBD can also resist potential adaptive attacks. The code is available at https://github.com/zaixizhang/CBD.

        ----

        ## [1167] Defending Against Patch-based Backdoor Attacks on Self-Supervised Learning

        **Authors**: *Ajinkya Tejankar, Maziar Sanjabi, Qifan Wang, Sinong Wang, Hamed Firooz, Hamed Pirsiavash, Liang Tan*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01178](https://doi.org/10.1109/CVPR52729.2023.01178)

        **Abstract**:

        Recently, self-supervised learning (SSL) was shown to be vulnerable to patch-based data poisoning backdoor attacks. It was shown that an adversary can poison a small part of the unlabeled data so that when a victim trains an SSL model on it, the final model will have a back-door that the adversary can exploit. This work aims to defend self-supervised learning against such attacks. We use a three-step defense pipeline, where we first train a model on the poisoned data. In the second step, our proposed defense algorithm (PatchSearch) uses the trained model to search the training data for poisoned samples and removes them from the training set. In the third step, a final model is trained on the cleaned-up training set. Our results show that PatchSearch is an effective defense. As an example, it improves a model's accuracy on images containing the trigger from 38.2% to 63.7% which is very close to the clean model's accuracy, 64.6%. More-over, we show that PatchSearch outperforms baselines and state-of-the-art defense approaches including those using additional clean, trusted data. Our code is available at https://github.com/UCDvision/PatchSearch

        ----

        ## [1168] Backdoor Attacks Against Deep Image Compression via Adaptive Frequency Trigger

        **Authors**: *Yi Yu, Yufei Wang, Wenhan Yang, Shijian Lu, Yap-Peng Tan, Alex C. Kot*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01179](https://doi.org/10.1109/CVPR52729.2023.01179)

        **Abstract**:

        Recent deep-learning-based compression methods have achieved superior performance compared with traditional approaches. However, deep learning models have proven to be vulnerable to backdoor attacks, where some specific trigger patterns added to the input can lead to malicious behavior of the models. In this paper, we present a novel backdoor attack with multiple triggers against learned image compression models. Motivated by the widely used discrete cosine transform (DCT) in existing compression systems and standards, we propose a frequency-based trigger injection model that adds triggers in the DCT domain. In particular, we design several attack objectives for various attacking scenarios, including: 1) attacking compression quality in terms of bit-rate and reconstruction quality; 2) attacking task-driven measures, such as downstream face recognition and semantic segmentation. Moreover, a novel simple dynamic loss is designed to balance the influence of different loss terms adaptively, which helps achieve more efficient training. Extensive experiments show that with our trained trigger injection models and simple modification of encoder parameters (of the compression model), the proposed attack can successfully inject several backdoors with corresponding triggers in a single image compression model.

        ----

        ## [1169] CAP: Robust Point Cloud Classification via Semantic and Structural Modeling

        **Authors**: *Daizong Ding, Erling Jiang, Yuanmin Huang, Mi Zhang, Wenxuan Li, Min Yang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01180](https://doi.org/10.1109/CVPR52729.2023.01180)

        **Abstract**:

        Recently, deep neural networks have shown great success on 3D point cloud classification tasks, which simultaneously raises the concern of adversarial attacks that cause severe damage to real-world applications. Moreover, defending against adversarial examples in point cloud data is extremely difficult due to the emergence of various attack strategies. In this work, with the insight of the fact that the adversarial examples in this task still preserve the same semantic and structural information as the original input, we design a novel defense framework for improving the robustness of existing classification models, which consists of two main modules: the attention-based pooling and the dynamic contrastive learning. In addition, we also develop an algorithm to theoretically certify the robustness of the proposed framework. Extensive empirical results on two datasets and three classification models show the robustness of our approach against various attacks, e.g., the averaged attack success rate of PointNet decreases from 70.2% to 2.7% on the ModelNet40 dataset under 9 common attacks.

        ----

        ## [1170] Evading DeepFake Detectors via Adversarial Statistical Consistency

        **Authors**: *Yang Hou, Qing Guo, Yihao Huang, Xiaofei Xie, Lei Ma, Jianjun Zhao*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01181](https://doi.org/10.1109/CVPR52729.2023.01181)

        **Abstract**:

        In recent years, as various realistic face forgery techniques known as DeepFake improves by leaps and bounds, more and more DeepFake detection techniques have been proposed. These methods typically rely on detecting statistical differences between natural (i.e., real) and DeepFake-generated images in both spatial and frequency domains. In this work, we propose to explicitly minimize the statistical differences to evade state-of-the-art DeepFake detectors. To this end, we propose a statistical consistency attack (StatAttack) against DeepFake detectors, which contains two main parts. First, we select several statistical-sensitive natural degradations (i.e., exposure, blur, and noise) and add them to the fake images in an adversarial way. Second, we find that the statistical differences between natural and DeepFake images are positively associated with the distribution shifting between the two kinds of images, and we propose to use a distribution-aware loss to guide the optimization of different degradations. As a result, the feature distributions of generated adversarial examples is close to the natural images. Furthermore, we extend the StatAttack to a more powerful version, MStatAttack, where we extend the single-layer degradation to multi-layer degradations sequentially and use the loss to tune the combination weights jointly. Comprehensive experimental results on four spatial-based detectors and two frequency-based detectors with four datasets demonstrate the effectiveness of our proposed attack method in both white-box and black-box settings.

        ----

        ## [1171] Enhancing the Self-Universality for Transferable Targeted Attacks

        **Authors**: *Zhipeng Wei, Jingjing Chen, Zuxuan Wu, Yu-Gang Jiang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01182](https://doi.org/10.1109/CVPR52729.2023.01182)

        **Abstract**:

        In this paper, we propose a novel transfer-based targeted attack method that optimizes the adversarial perturbations without any extra training efforts for auxiliary networks on training data. Our new attack method is proposed based on the observation that highly universal adversarial perturbations tend to be more transferable for targeted attacks. Therefore, we propose to make the perturbation to be agnostic to different local regions within one image, which we called as self-universality. Instead of optimizing the perturbations on different images, optimizing on different regions to achieve self-universality can get rid of using extra data. Specifically, we introduce a feature similarity loss that encourages the learned perturbations to be universal by maximizing the feature similarity between adversarial perturbed global images and randomly cropped local regions. With the feature similarity loss, our method makes the features from adversarial perturbations to be more dominant than that of benign images, hence improving targeted transferability. We name the proposed attack method as Self-Universality (SU) attack. Extensive experiments demonstrate that SU can achieve high success rates for transfer-based targeted attacks. On ImageNet-compatible dataset, SU yields an improvement of 12% compared with existing state-of-the-art methods. Code is available at https://github.com/zhipengwei/Self-Universality.

        ----

        ## [1172] Black-Box Sparse Adversarial Attack via Multi-Objective Optimisation CVPR Proceedings

        **Authors**: *Phoenix Neale Williams, Ke Li*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01183](https://doi.org/10.1109/CVPR52729.2023.01183)

        **Abstract**:

        Deep neural networks (DNNs) are susceptible to adversarial images, raising concerns about their reliability in safety-critical tasks. Sparse adversarial attacks, which limit the number of modified pixels, have shown to be highly effective in causing DNNs to misclassify. However, existing methods often struggle to simultaneously minimize the number of modified pixels and the size of the modifications, often requiring a large number of queries and assuming unrestricted access to the targeted DNN. In contrast, other methods that limit the number of modified pixels often permit unbounded modifications, making them easily detectable. To address these limitations, we propose a novel multi-objective sparse attack algorithm that efficiently minimizes the number of modified pixels and their size during the attack process. Our algorithm draws inspiration from evolutionary computation and incorporates a mechanism for prioritizing objectives that aligns with an attacker's goals. Our approach outperforms existing sparse attacks on CIFAR-10 and ImageNet trained DNN classifiers while requiring only a small query budget, attaining competitive attack success rates while perturbing fewer pixels. Overall, our proposed attack algorithm provides a solution to the limitations of current sparse attack methods by jointly minimizing the number of modified pixels and their size. Our results demonstrate the effectiveness of our approach in restricted scenarios, highlighting its potential to enhance DNN security.

        ----

        ## [1173] Seasoning Model Soups for Robustness to Adversarial and Natural Distribution Shifts

        **Authors**: *Francesco Croce, Sylvestre-Alvise Rebuffi, Evan Shelhamer, Sven Gowal*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01185](https://doi.org/10.1109/CVPR52729.2023.01185)

        **Abstract**:

        Adversarial training is widely used to make classifiers robust to a specific threat or adversary, such as lp- norm bounded perturbations of a given p-norm. However, existing methods for training classifiers robust to multiple threats require knowledge of all attacks during training and remain vulnerable to unseen distribution shifts. In this work, we describe how to obtain adversarially-robust model soups (i.e., linear combinations of parameters) that smoothly trade-off robustness to different lp-norm bounded adversaries. We demonstrate that such soups allow us to control the type and level of robustness, and can achieve robustness to all threats without jointly training on all of them. In some cases, the resulting model soups are more robust to a given lp-norm adversary than the constituent model specialized against that same adversary. Finally, we show that adversarially-robust model soups can be a viable tool to adapt to distribution shifts from a few examples.

        ----

        ## [1174] Towards Benchmarking and Assessing Visual Naturalness of Physical World Adversarial Attacks

        **Authors**: *Simin Li, Shuning Zhang, Gujun Chen, Dong Wang, Pu Feng, Jiakai Wang, Aishan Liu, Xin Yi, Xianglong Liu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01186](https://doi.org/10.1109/CVPR52729.2023.01186)

        **Abstract**:

        Physical world adversarial attack is a highly practical and threatening attack, which fools real world deep learning systems by generating conspicuous and maliciously crafted real world artifacts. In physical world attacks, evaluating naturalness is highly emphasized since human can easily detect and remove unnatural attacks. However, current studies evaluate naturalness in a case-by-case fashion, which suffers from errors, bias and inconsistencies. In this paper, we take the first step to benchmark and assess visual naturalness of physical world attacks, taking autonomous driving scenario as the first attempt. First, to benchmark attack naturalness, we contribute the first Physical Attack Naturalness (PAN) dataset with human rating and gaze. PAN verifies several insights for the first time: naturalness is (disparately) affected by contextual features (i.e., environmental and semantic variations) and correlates with behavioral feature (i.e., gaze signal). Second, to automatically assess attack naturalness that aligns with human ratings, we further introduce Dual Prior Alignment (DPA) network, which aims to embed human knowledge into model reasoning process. Specifically, DPA imitates human reasoning in naturalness assessment by rating prior alignment and mimics human gaze behavior by attentive prior alignment. We hope our work fosters researches to improve and automatically assess naturalness of physical world attacks. Our code and dataset can be found at https://github.com/zhangsn-19/PAN.

        ----

        ## [1175] Physically Adversarial Infrared Patches with Learnable Shapes and Locations

        **Authors**: *Xingxing Wei, Jie Yu, Yao Huang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01187](https://doi.org/10.1109/CVPR52729.2023.01187)

        **Abstract**:

        Owing to the extensive application of infrared object detectors in the safety-critical tasks, it is necessary to evaluate their robustness against adversarial examples in the real world. However, current few physical infrared attacks are complicated to implement in practical application because of their complex transformation from digital world to physical world. To address this issue, in this paper, we propose a physically feasible infrared attack method called “adversarial infrared patches”. Considering the imaging mechanism of infrared cameras by capturing objects' thermal radiation, adversarial infrared patches conduct attacks by attaching a patch of thermal insulation materials on the target object to manipulate its thermal distribution. To enhance adversarial attacks, we present a novel aggregation regularization to guide the simultaneous learning for the patch’ shape and location on the target object. Thus, a simple gradient-based optimization can be adapted to solve for them. We verify adversarial infrared patches in different object detection tasks with various object detectors. Experimental results show that our method achieves more than 90% Attack Success Rate (ASR) versus the pedestrian detector and vehicle detector in the physical environment, where the objects are captured in different angles, distances, postures, and scenes. More importantly, adversarial infrared patch is easy to implement, and it only needs 0.5 hours to be constructed in the physical world, which verifies its effectiveness and efficiency.

        ----

        ## [1176] MaLP: Manipulation Localization Using a Proactive Scheme

        **Authors**: *Vishal Asnani, Xi Yin, Tal Hassner, Xiaoming Liu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01188](https://doi.org/10.1109/CVPR52729.2023.01188)

        **Abstract**:

        Advancements in the generation quality of various Generative Models (GMs) has made it necessary to not only perform binary manipulation detection but also localize the modified pixels in an image. However, prior works termed as passive for manipulation localization exhibit poor generalization performance over unseen GMs and attribute modifications. To combat this issue, we propose a proactive scheme for manipulation localization, termed MaLP. We encrypt the real images by adding a learned template. If the image is manipulated by any GM, this added protection from the template not only aids binary detection but also helps in identifying the pixels modified by the GM. The template is learned by leveraging local and global-level features estimated by a two-branch architecture. We show that MaLP performs better than prior passive works. We also show the generalizability of MaLP by testing on 22 different GMs, providing a benchmark for future research on manipulation localization. Finally, we show that MaLP can be used as a discriminator for improving the generation quality of GMs. Our models/codes are available at www.github.com/vishal3477/pro_loc.

        ----

        ## [1177] Polarimetric iToF: Measuring High-Fidelity Depth Through Scattering Media

        **Authors**: *Daniel S. Jeon, Andreas Meuleman, Seung-Hwan Baek, Min H. Kim*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01189](https://doi.org/10.1109/CVPR52729.2023.01189)

        **Abstract**:

        Indirect time-of-flight (iToF) imaging allows us to capture dense depth information at a low cost. However, iToF imaging often suffers from multipath interference (MPI) artifacts in the presence of scattering media, resulting in severe depth-accuracy degradation. For instance, iToF cameras cannot measure depth accurately through fog because ToF active illumination scatters back to the sensor before reaching the farther target surface. In this work, we propose a polarimetric iToF imaging method that can capture depth information robustly through scattering media. Our observations on the principle of indirect ToF imaging and polarization of light allow us to formulate a novel computational model of scattering-aware polarimetric phase measurements that enables us to correct MPI errors. We first devise a scattering-aware polarimetric iToF model that can estimate the phase of unpolarized backscattered light. We then combine the optical filtering of polarization and our computational modeling of unpolarized backscattered light via scattering analysis of phase and amplitude. This allows us to tackle the MPI problem by estimating the scattering energy through the participating media. We validate our method on an experimental setup using a customized off-the-shelf iToF camera. Our method outperforms baseline methods by a significant margin by means of our scattering model and polarimetric phase measurements.

        ----

        ## [1178] NeRFLiX: High-Quality Neural View Synthesis by Learning a Degradation-Driven Inter-viewpoint MiXer

        **Authors**: *Kun Zhou, Wenbo Li, Yi Wang, Tao Hu, Nianjuan Jiang, Xiaoguang Han, Jiangbo Lu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01190](https://doi.org/10.1109/CVPR52729.2023.01190)

        **Abstract**:

        Neural radiance fields (NeRF) show great success in novel view synthesis. However, in real-world scenes, recovering high-quality details from the source images is still challenging for the existing NeRF-based approaches, due to the potential imperfect calibration information and scene representation inaccuracy. Even with high-quality training frames, the synthetic novel views produced by NeRF models still suffer from notable rendering artifacts, such as noise, blur, etc. Towards to improve the synthesis quality of NeRF-based approaches, we propose NeRFLiX, a general NeRF-agnostic restorer paradigm by learning a degradation-driven inter-viewpoint mixer. Specially, we design a NeRF-style degradation modeling approach and construct large-scale training data, enabling the possibility of effectively removing NeRF-native rendering artifacts for existing deep neural networks. Moreover, beyond the degradation removal, we propose an inter-viewpoint aggregation framework that is able to fuse highly related high-quality training images, pushing the performance of cutting-edge NeRF models to entirely new levels and producing highly photo-realistic synthetic views.

        ----

        ## [1179] SUDS: Scalable Urban Dynamic Scenes

        **Authors**: *Haithem Turki, Jason Y. Zhang, Francesco Ferroni, Deva Ramanan*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01191](https://doi.org/10.1109/CVPR52729.2023.01191)

        **Abstract**:

        We extend neural radiance fields (NeRFs) to dynamic large-scale urban scenes. Prior work tends to reconstruct single video clips of short durations (up to 10 seconds). Two reasons are that such methods (a) tend to scale linearly with the number of moving objects and input videos because a separate model is built for each and (b) tend to require supervision via 3D bounding boxes and panoptic labels, obtained manually or via category-specific models. As a step towards truly open-world reconstructions of dynamic cities, we introduce two key innovations: (a) we factorize the scene into three separate hash table data structures to efficiently encode static, dynamic, and far-field radiance fields, and (b) we make use of unlabeled target signals consisting of RGB images, sparse LiDAR, off-the-shelf self-supervised 2D descriptors, and most importantly, 2D optical flow. Operationalizing such inputs via photometric, geometric, and feature-metric reconstruction losses enables SUDS to decompose dynamic scenes into the static background, individual objects, and their motions. When combined with our multi-branch table representation, such reconstructions can be scaled to tens of thousands of objects across 1.2 million frames from 1700 videos spanning geospatial footprints of hundreds of kilometers, (to our knowledge) the largest dynamic NeRF built to date. We present qualitative initial results on a variety of tasks enabled by our representations, including novel-view synthesis of dynamic urban scenes, unsupervised 3D instance segmentation, and unsupervised 3D cuboid detection. To compare to prior work, we also evaluate on KITTI and Virtual KITTI 2, surpassing state-of-the-art methods that rely on ground truth 3D bounding box annotations while being 10x quicker to train.

        ----

        ## [1180] DP-NeRF: Deblurred Neural Radiance Field with Physical Scene Priors

        **Authors**: *Dogyoon Lee, Minhyeok Lee, Chajin Shin, Sangyoun Lee*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01192](https://doi.org/10.1109/CVPR52729.2023.01192)

        **Abstract**:

        Neural Radiance Field (NeRF) has exhibited outstanding three-dimensional (3D) reconstruction quality via the novel view synthesis from multi-view images and paired calibrated camera parameters. However, previous NeRF-based systems have been demonstrated under strictly controlled settings, with little attention paid to less ideal scenarios, including with the presence of noise such as exposure, illumination changes, and blur. In particular, though blur frequently occurs in real situations, NeRF that can handle blurred images has received little attention. The few studies that have investigated NeRF for blurred images have not considered geometric and appearance consistency in 3D space, which is one of the most important factors in 3D reconstruction. This leads to inconsistency and the degradation of the perceptual quality of the constructed scene. Hence, this paper proposes a DP-NeRF, a novel clean NeRF framework for blurred images, which is constrained with two physical priors. These priors are derived from the actual blurring process during image acquisition by the camera. DP-NeRF proposes rigid blurring kernel to impose 3D consistency utilizing the physical priors and adaptive weight proposal to refine the color composition error in consideration of the relationship between depth and blur. We present extensive experimental results for synthetic and real scenes with two types of blur: camera motion blur and defocus blur. The results demonstrate that DP-NeRF successfully improves the perceptual quality of the constructed NeRF ensuring 3D geometric and appearance consistency. We further demonstrate the effectiveness of our model with comprehensive ablation analysis. 
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
Code: https://github.com/dogyoonlee/DP-NeRF 
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">2</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">2</sup>
Project: https://dogyoonlee.github.io/dpNeRF/

        ----

        ## [1181] DyLiN: Making Light Field Networks Dynamic

        **Authors**: *Heng Yu, Joel Julin, Zoltan A. Milacski, Koichiro Niinuma, László A. Jeni*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01193](https://doi.org/10.1109/CVPR52729.2023.01193)

        **Abstract**:

        Light Field Networks, the re-formulations of radiance fields to oriented rays, are magnitudes faster than their coordinate network counterparts, and provide higher fidelity with respect to representing 3D structures from 2D observations. They would be well suited for generic scene representation and manipulation, but suffer from one problem: they are limited to holistic and static scenes. In this paper, we propose the Dynamic Light Field Network (DyLiN) method that can handle non-rigid deformations, including topological changes. We learn a deformation field from input rays to canonical rays, and lift them into a higher dimensional space to handle discontinuities. We further introduce CoDyLiN, which augments DyLiN with controllable attribute inputs. We train both models via knowledge distillation from pretrained dynamic radiance fields. We evaluated DyLiN using both synthetic and real world datasets that include various non-rigid deformations. DyLiN qualitatively outperformed and quantitatively matched state-of-the-art methods in terms of visual fidelity, while being 25 – 71× computationally faster. We also tested CoDyLiN on attribute annotated data and it surpassed its teacher model. Project page: https://dylin2023.github.io.

        ----

        ## [1182] Multi-Space Neural Radiance Fields

        **Authors**: *Ze-Xin Yin, Jiaxiong Qiu, Ming-Ming Cheng, Bo Ren*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01194](https://doi.org/10.1109/CVPR52729.2023.01194)

        **Abstract**:

        Existing Neural Radiance Fields (NeRF) methods suffer from the existence of reflective objects, often resulting in blurry or distorted rendering. Instead of calculating a single radiance field, we propose a multi-space neural radiance field (MS-NeRF) that represents the scene using a group of feature fields in parallel sub-spaces, which leads to a better understanding of the neural network toward the existence of reflective and refractive objects. Our multi-space scheme works as an enhancement to existing NeRF methods, with only small computational overheads needed for training and inferring the extra-space outputs. We demonstrate the superiority and compatibility of our approach using three representative NeRF-based models, i.e., NeRF, Mip-NeRF, and Mip-NeRF 360. Comparisons are performed on a novelly constructed dataset consisting of 25 synthetic scenes and 7 real captured scenes with complex reflection and refraction, all having 360-degree viewpoints. Extensive experiments show that our approach significantly outperforms the existing single-space NeRF methods for rendering high-quality scenes concerned with complex light paths through mirror-like objects. Our code and dataset will be publicly available at https://zx-yin.github.io/msnerf.

        ----

        ## [1183] NeRFLight: Fast and Light Neural Radiance Fields using a Shared Feature Grid

        **Authors**: *Fernando Rivas-Manzaneque, Jorge Sierra Acosta, Adrián Peñate Sánchez, Francesc Moreno-Noguer, Angela Ribeiro*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01195](https://doi.org/10.1109/CVPR52729.2023.01195)

        **Abstract**:

        While original Neural Radiance Fields (NeRF) have shown impressive results in modeling the appearance of a scene with compact MLP architectures, they are not able to achieve real-time rendering. This has been recently addressed by either baking the outputs of NeRF into a data structure or arranging trainable parameters in an explicit feature grid. These strategies, however, significantly increase the memory footprint of the model which prevents their deployment on bandwidth-constrained applications. In this paper, we extend the grid-based approach to achieve real-time view synthesis at more than 150 FPS using a lightweight model. Our main contribution is a novel architecture in which the density field of NeRF-based representations is split into N regions and the density is modeled using N different decoders which reuse the same feature grid. This results in a smaller grid where each feature is located in more than one spatial position, forcing them to learn a compact representation that is valid for different parts of the scene. We further reduce the size of the final model by disposing of the features symmetrically on each region, which favors feature pruning after training while also allowing smooth gradient transitions between neighboring voxels. An exhaustive evaluation demonstrates that our method achieves real-time performance and quality metrics on a pair with state-of-the-art with an improvement of more than 2× in the FPS/MB ratio.

        ----

        ## [1184] Cross-Guided Optimization of Radiance Fields with Multi-View Image Super-Resolution for High-Resolution Novel View Synthesis

        **Authors**: *Youngho Yoon, Kuk-Jin Yoon*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01196](https://doi.org/10.1109/CVPR52729.2023.01196)

        **Abstract**:

        Novel View Synthesis (NVS) aims at synthesizing an image from an arbitrary viewpoint using multi-view images and camera poses. Among the methods for NVS, Neural Radiance Fields (NeRF) is capable of NVS for an arbitrary resolution as it learns a continuous volumetric representation. However, radiance fields rely heavily on the spectral characteristics of coordinate-based networks. Thus, there is a limit to improving the performance of high-resolution novel view synthesis (HRNVS). To solve this problem, we propose a novel framework using cross-guided optimization of the single-image super-resolution (SISR) and radiance fields. We perform multi-view image super-resolution (MVSR) on train-view images during the radiance fields optimization process. It derives the updated SR result by fusing the feature map obtained from SISR and voxel-based uncertainty fields generated by integrated errors of train-view images. By repeating the updates during radiance fields optimization, train-view images for radiance fields optimization have multi-view consistency and high-frequency details simultaneously, ultimately improving the performance of HRNVS. Experiments of HRNVS and MVSR on various benchmark datasets show that the proposed method significantly surpasses existing methods.

        ----

        ## [1185] NeuralEditor: Editing Neural Radiance Fields via Manipulating Point Clouds

        **Authors**: *Jun-Kun Chen, Jipeng Lyu, Yu-Xiong Wang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01197](https://doi.org/10.1109/CVPR52729.2023.01197)

        **Abstract**:

        This paper proposes NeuralEditor that enables neural radiance fields (NeRFs) natively editable for general shape editing tasks. Despite their impressive results on novel-view synthesis, it remains a fundamental challenge for NeRFs to edit the shape of the scene. Our key insight is to exploit the explicit point cloud representation as the underlying structure to construct NeRFs, inspired by the intuitive interpretation of NeRF rendering as a process that projects or “plots” the associated 3D point cloud to a 2D image plane. To this end, NeuralEditor introduces a novel rendering scheme based on deterministic integration within K-D tree-guided density-adaptive voxels, which produces both high-quality rendering results and precise point clouds through optimization. NeuralEditor then performs shape editing via mapping associated points between point clouds. Extensive evaluation shows that NeuralEditor achieves state-of-the-art performance in both shape deformation and scene morphing tasks. Notably, NeuralEditor supports both zeroshot inference and further fine-tuning over the edited scene. Our code, benchmark, and demo video are available at immortalco.github.io/NeuralEditor.

        ----

        ## [1186] DINER: Depth-aware Image-based NEural Radiance fields

        **Authors**: *Malte Prinzler, Otmar Hilliges, Justus Thies*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01198](https://doi.org/10.1109/CVPR52729.2023.01198)

        **Abstract**:

        We present Depth-aware Image-based NEural Radiance fields (DINER). Given a sparse set of RGB input views, we predict depth and feature maps to guide the reconstruction of a volumetric scene representation that allows us to render 3D objects under novel views. Specifically, we propose novel techniques to incorporate depth information into feature fusion and efficient scene sampling. In comparison to the previous state of the art, DINER achieves higher synthesis quality and can process input views with greater disparity. This allows us to capture scenes more completely without changing capturing hardware requirements and ultimately enables larger viewpoint changes during novel view synthesis. We evaluate our method by synthesizing novel views, both for human heads and for general objects, and observe significantly improved qualitative results and increased perceptual metrics compared to the previous state of the art. The code is publicly available through the Project Webpage.

        ----

        ## [1187] Modernizing Old Photos Using Multiple References via Photorealistic Style Transfer

        **Authors**: *Agus Gunawan, Soo Ye Kim, Hyeonjun Sim, Jae-Ho Lee, Munchurl Kim*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01199](https://doi.org/10.1109/CVPR52729.2023.01199)

        **Abstract**:

        This paper firstly presents old photo modernization using multiple references by performing stylization and enhancement in a unified manner. In order to modernize old photos, we propose a novel multi-reference-based old photo modernization (MROPM) framework consisting of a network MROPM-Net and a novel synthetic data generation scheme. MROPM-Net stylizes old photos using multiple references via photorealistic style transfer (PST) and further enhances the results to produce modern-looking images. Meanwhile, the synthetic data generation scheme trains the network to effectively utilize multiple references to perform modernization. To evaluate the performance, we propose a new old photos benchmark dataset (CHD) consisting of diverse natural indoor and outdoor scenes. Extensive experiments show that the proposed method outperforms other baselines in performing modernization on real old photos, even though no old photos were used during training. Moreover, our method can appropriately select styles from multiple references for each semantic region in the old photo to further improve the modernization performance.

        ----

        ## [1188] Efficient Map Sparsification Based on 2D and 3D Discretized Grids

        **Authors**: *Xiaoyu Zhang, Yun-Hui Liu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01200](https://doi.org/10.1109/CVPR52729.2023.01200)

        **Abstract**:

        Localization in a pre-built map is a basic technique for robot autonomous navigation. Existing mapping and localization methods commonly work well in small-scale environments. As a map grows larger, however, more memory is required and localization becomes inefficient. To solve these problems, map sparsification becomes a practical necessity to acquire a subset of the original map for localization. Previous map sparsification methods add a quadratic term in mixed-integer programming to enforce a uniform distribution of selected landmarks, which requires high memory capacity and heavy computation. In this paper, we formulate map sparsification in an efficient linear form and select uniformly distributed landmarks based on 2D discretized grids. Furthermore, to reduce the influence of different spatial distributions between the mapping and query sequences, which is not considered in previous methods, we also introduce a space constraint term based on 3D discretized grids. The exhaustive experiments in different datasets demonstrate the superiority of the proposed methods in both efficiency and localization performance. The relevant codes will be released at https://github.com/fishmarch/SLAM_Map_Compression.

        ----

        ## [1189] K-Planes: Explicit Radiance Fields in Space, Time, and Appearance

        **Authors**: *Sara Fridovich-Keil, Giacomo Meanti, Frederik Rahbæk Warburg, Benjamin Recht, Angjoo Kanazawa*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01201](https://doi.org/10.1109/CVPR52729.2023.01201)

        **Abstract**:

        We introduce k-planes, a white-box model for radiance fields in arbitrary dimensions. Our model uses planes to represent a d-dimensional scene, providing a seamless way to go from static (d = 3) to dynamic (d= 4) scenes. This planar factorization makes adding dimension-specific priors easy, e.g. temporal smoothness and multi-resolution spatial structure, and induces a natural decomposition of static and dynamic components of a scene. We use a linear feature decoder with a learned color basis that yields similar performance as a nonlinear black-box MLP decoder. Across a range of synthetic and real, static and dynamic, fixed and varying appearance scenes, k-planes yields competitive and often state-of-the-art recon- struction fidelity with low memory usage, achieving 1000x compression over a full 4D grid, and fast optimization with a pure PyTorch implementation. For video results and code, please see sarafridov.github.io/K-Planes.

        ----

        ## [1190] I2-SDF: Intrinsic Indoor Scene Reconstruction and Editing via Raytracing in Neural SDFs

        **Authors**: *Jingsen Zhu, Yuchi Huo, Qi Ye, Fujun Luan, Jifan Li, Dianbing Xi, Lisha Wang, Rui Tang, Wei Hua, Hujun Bao, Rui Wang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01202](https://doi.org/10.1109/CVPR52729.2023.01202)

        **Abstract**:

        In this work, we present I
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">2</sup>
-SDF, a new method for intrinsic indoor scene reconstruction and editing using differentiable Monte Carlo raytracing on neural signed distance fields (SDFs). Our holistic neural SDF-based frame-work jointly recovers the underlying shapes, incident radiance and materials from multi-view images. We introduce a novel bubble loss for fine-grained small objects and error-guided adaptive sampling scheme to largely improve the reconstruction quality on large-scale indoor scenes. Further, we propose to decompose the neural radiance field into spatially-varying material of the scene as a neural field through surface-based, differentiable Monte Carlo raytracing and emitter semantic segmentations, which enables physically based and photorealistic scene relighting and editing applications. Through a number of qualitative and quantitative experiments, we demonstrate the superior quality of our method on indoor scene reconstruction, novel view synthesis, and scene editing compared to state-of-the-art baselines. Our project page is at https://jingsenzhu.github.io/i2-sdf.

        ----

        ## [1191] Multi-view Inverse Rendering for Large-scale Real-world Indoor Scenes

        **Authors**: *Zhen Li, Lingli Wang, Mofang Cheng, Cihui Pan, Jiaqi Yang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01203](https://doi.org/10.1109/CVPR52729.2023.01203)

        **Abstract**:

        We present a efficient multi-view inverse rendering method for large-scale real-world indoor scenes that reconstructs global illumination and physically-reasonable SVBRDFs. Unlike previous representations, where the global illumination of large scenes is simplified as multiple environment maps, we propose a compact representation called Texture-based Lighting (TBL). It consists of 3D mesh and HDR textures, and efficiently models direct and infinite-bounce indirect lighting of the entire large scene. Based on TBL, we further propose a hybrid lighting representation with precomputed irradiance, which significantly improves the efficiency and alleviates the rendering noise in the material optimization. To physically disentangle the ambiguity between materials, we propose a three-stage material optimization strategy based on the priors of semantic segmentation and room segmentation. Extensive experiments show that the proposed method outperforms the state-of-the-art quantitatively and qualitatively, and enables physically-reasonable mixed-reality applications such as material editing, editable novel view synthesis and relighting. The project page is at https://lzleejean.github.io/TexIR.

        ----

        ## [1192] Inverse Rendering of Translucent Objects using Physical and Neural Renderers

        **Authors**: *Chenhao Li, Trung Thanh Ngo, Hajime Nagahara*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01204](https://doi.org/10.1109/CVPR52729.2023.01204)

        **Abstract**:

        In this work, we propose an inverse rendering model that estimates 3D shape, spatially-varying reflectance, homogeneous subsurface scattering parameters, and an environment illumination jointly from only a pair of captured images of a translucent object. In order to solve the ambiguity problem of inverse rendering, we use a physically-based renderer and a neural renderer for scene reconstruction and material editing. Because two renderers are differentiable, we can compute a reconstruction loss to assist parameter estimation. To enhance the supervision of the proposed neural renderer, we also propose an augmented loss. In addition, we use a flash and no-flash image pair as the input. To supervise the training, we constructed a large-scale synthetic dataset of translucent objects, which consists of 117K scenes. Qualitative and quantitative results on both synthetic and real-world datasets demonstrated the effectiveness of the proposed model. Code and Data are available at https://github.com/ligoudaner377/homo_translucent

        ----

        ## [1193] Accidental Light Probes

        **Authors**: *Hong-Xing Yu, Samir Agarwala, Charles Herrmann, Richard Szeliski, Noah Snavely, Jiajun Wu, Deqing Sun*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01205](https://doi.org/10.1109/CVPR52729.2023.01205)

        **Abstract**:

        Recovering lighting in a scene from a single image is a fundamental problem in computer vision. While a mirror ball light probe can capture omnidirectional lighting, light probes are generally unavailable in everyday images. In this work, we study recovering lighting from accidental light probes (ALPs)-common, shiny objects like Coke cans, which often accidentally appear in daily scenes. We propose a physically-based approach to model ALPs and estimate lighting from their appearances in single images. The main idea is to model the appearance of ALPs by photogram-metrically principled shading and to invert this process via differentiable rendering to recover incidental illumination. We demonstrate that we can put an ALP into a scene to allow high-fidelity lighting estimation. Our model can also recover lighting for existing images that happen to contain an ALP**Project website: https://kovenyu.com/ALP. I'd rather be Shiny. - Tamatoa from Moana, 2016

        ----

        ## [1194] Humans as Light Bulbs: 3D Human Reconstruction from Thermal Reflection

        **Authors**: *Ruoshi Liu, Carl Vondrick*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01206](https://doi.org/10.1109/CVPR52729.2023.01206)

        **Abstract**:

        The relatively hot temperature of the human body causes people to turn into long-wave infrared light sources. Since this emitted light has a larger wavelength than visible light, many surfaces in typical scenes act as infrared mirrors with strong specular reflections. We exploit the thermal reflections of a person onto objects in order to locate their position and reconstruct their pose, even if they are not visible to a normal camera. We propose an analysis-by-synthesis framework that jointly models the objects, people, and their thermal reflections, which combines generative models with differentiable rendering of reflections. Quantitative and qualitative experiments show our approach works in highly challenging cases, such as with curved mirrors or when the person is completely unseen by a normal camera.

        ----

        ## [1195] HumanGen: Generating Human Radiance Fields with Explicit Priors

        **Authors**: *Suyi Jiang, Haoran Jiang, Ziyu Wang, Haimin Luo, Wenzheng Chen, Lan Xu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01207](https://doi.org/10.1109/CVPR52729.2023.01207)

        **Abstract**:

        Recent years have witnessed the tremendous progress of 3D GANs for generating view-consistent radiance fields with photo-realism. Yet, high-quality generation of human radiance fields remains challenging, partially due to the limited human-related priors adopted in existing methods. We present HumanGen, a novel 3D human generation scheme with detailed geometry and 360° realistic free-view rendering. It explicitly marries the 3D human generation with various priors from the 2D generator and 3D reconstructor of humans through the design of “anchor image”. We introduce a hybrid feature representation using the anchor image to bridge the latent space of HumanGen with the existing 2D generator. We then adopt a pronged design to disentangle the generation of geometry and appearance. With the aid of the anchor image, we adapt a 3D reconstructor for fine-grained details synthesis and propose a two-stage blending scheme to boost appearance generation. Extensive experiments demonstrate our effectiveness for state-of-the-art 3D human generation regarding geometry details, texture quality, and free-view performance. Notably, HumanGen can also incorporate various off-the-shelf 2D latent editing methods, seamlessly lifting them into 3D.

        ----

        ## [1196] Seeing Through the Glass: Neural 3D Reconstruction of Object Inside a Transparent Container

        **Authors**: *Jinguang Tong, Sundaram Muthu, Fahira Afzal Maken, Chuong Nguyen, Hongdong Li*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01208](https://doi.org/10.1109/CVPR52729.2023.01208)

        **Abstract**:

        In this paper, we define a new problem of recovering the 3D geometry of an object confined in a transparent enclosure. We also propose a novel method for solving this challenging problem. Transparent enclosures pose challenges of multiple light reflections and refractions at the interface between different propagation media e.g. air or glass. These multiple reflections and refractions cause serious image distortions which invalidate the single viewpoint assumption. Hence the 3D geometry of such objects cannot be reliably reconstructed using existing methods, such as traditional structure from motion or modern neural reconstruction methods. We solve this problem by explicitly modeling the scene as two distinct sub-spaces, inside and outside the transparent enclosure. We use an existing neural reconstruction method (NeuS) that implicitly represents the geometry and appearance of the inner subspace. In order to account for complex light interactions, we develop a hybrid rendering strategy that combines volume rendering with ray tracing. We then recover the underlying geometry and appearance of the model by minimizing the difference between the real and rendered images. We evaluate our method on both synthetic and real data. Experiment results show that our method outperforms the state-of-the-art (SOTA) methods. Codes and data will be available at https://github.com/hirotong/ReNeuS

        ----

        ## [1197] 3D Shape Reconstruction of Semi-Transparent Worms

        **Authors**: *Thomas P. Ilett, Omer Yuval, Thomas Ranner, Netta Cohen, David C. Hogg*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01209](https://doi.org/10.1109/CVPR52729.2023.01209)

        **Abstract**:

        3D shape reconstruction typically requires identifying object features or textures in multiple images of a subject. This approach is not viable when the subject is semi-transparent and moving in and out of focus. Here we overcome these challenges by rendering a candidate shape with adaptive blurring and transparency for comparison with the images. We use the microscopic nematode Caenorhabditis elegans as a case study as it freely explores a 3D complex fluid with constantly changing optical properties. We model the slender worm as a 3D curve using an intrinsic parametrisation that naturally admits biologically-informed constraints and regularisation. To account for the changing optics we develop a novel differentiable renderer to construct images from 2D projections and compare against raw images to generate a pixel-wise error to jointly update the curve, camera and renderer parameters using gradient descent. The method is robust to interference such as bubbles and dirt trapped in the fluid, stays consistent through complex sequences of postures, recovers reliable estimates from blurry images and provides a significant improvement on previous attempts to track C. elegans in 3D. Our results demonstrate the potential of direct approaches to shape estimation in complex physical environments in the absence of ground-truth data.

        ----

        ## [1198] Dionysus: Recovering Scene Structures by Dividing into Semantic Pieces

        **Authors**: *Likang Wang, Lei Chen*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01210](https://doi.org/10.1109/CVPR52729.2023.01210)

        **Abstract**:

        Most existing 3D reconstruction methods result in either detail loss or unsatisfying efficiency. However, effectiveness and efficiency are equally crucial in real-world applications, e.g., autonomous driving and augmented reality. We argue that this dilemma comes from wasted resources on valueless depth samples. This paper tackles the problem by proposing a novel learning-based 3D reconstruction framework named Dionysus. Our main contribution is to find out the most promising depth candidates from estimated semantic maps. This strategy simultaneously enables high effectiveness and efficiency by attending to the most reliable nominators. Specifically, we distinguish unreliable depth candidates by checking the cross-view semantic consistency and allow adaptive sampling by redistributing depth nominators among pixels. Experiments on the most popular datasets confirm our proposed framework's effectiveness.

        ----

        ## [1199] SparseFusion: Distilling View-Conditioned Diffusion for 3D Reconstruction

        **Authors**: *Zhizhuo Zhou, Shubham Tulsiani*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01211](https://doi.org/10.1109/CVPR52729.2023.01211)

        **Abstract**:

        We propose SparseFusion, a sparse view 3D reconstruction approach that unifies recent advances in neural rendering and probabilistic image generation. Existing approaches typically build on neural rendering with reprojected features but fail to generate unseen regions or handle uncertainty under large viewpoint changes. Alternate methods treat this as a (probabilistic) 2D synthesis task, and while they can generate plausible 2D images, they do not infer a consistent underlying 3D. However, we find that this trade-off between 3D consistency and probabilistic image generation does not need to exist. In fact, we show that geometric consistency and generative inference can be complementary in a mode-seeking behavior. By distilling a 3D consistent scene representation from a view-conditioned latent diffusion model, we are able to recover a plausible 3D representation whose renderings are both accurate and realistic. We evaluate our approach across 51 categories in the CO3D dataset and show that it outperforms existing methods, in both distortion and perception metrics, for sparse-view novel view synthesis.

        ----

        

[Go to the previous page](CVPR-2023-list05.md)

[Go to the next page](CVPR-2023-list07.md)

[Go to the catalog section](README.md)