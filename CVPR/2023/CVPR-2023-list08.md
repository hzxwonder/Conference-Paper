## [1400] A Large-Scale Robustness Analysis of Video Action Recognition Models

        **Authors**: *Madeline Chantry Schiappa, Naman Biyani, Prudvi Kamtam, Shruti Vyas, Hamid Palangi, Vibhav Vineet, Yogesh S. Rawat*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01412](https://doi.org/10.1109/CVPR52729.2023.01412)

        **Abstract**:

        We have seen a great progress in video action recognition in recent years. There are several models based on convolutional neural network (CNN) and some recent transformer based approaches which provide top performance on existing benchmarks. In this work, we perform a large-scale robustness analysis of these existing models for video action recognition. We focus on robustness against real-world distribution shift perturbations instead of adversarial perturbations. We propose four different benchmark datasets, HMDB51-P, UCF101-P, Kinetics400-P, and SSv2-P to perform this analysis. We study robustness of six state-of-the-art action recognition models against 90 different perturbations. The study reveals some interesting findings, 1) transformer based models are consistently more robust compared to CNN based models, 2) Pretraining improves robustness for Transformer based models more than CNN based models, and 3) All of the studied models are robust to temporal perturbations for all datasets but SSv2; suggesting the importance of temporal information for action recognition varies based on the dataset and activities. Next, we study the role of augmentations in model robustness and present a real-world dataset, UCF101-DS, which contains realistic distribution shifts, to further validate some of these findings. We believe this study will serve as a benchmark for future research in robust video action recognition 
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
More details available at bit.1y/3TJLMUF..

        ----

        ## [1401] The Wisdom of Crowds: Temporal Progressive Attention for Early Action Prediction

        **Authors**: *Alexandros Stergiou, Dima Damen*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01413](https://doi.org/10.1109/CVPR52729.2023.01413)

        **Abstract**:

        Early action prediction deals with inferring the ongoing action from partially-observed videos, typically at the outset of the video. We propose a bottleneck-based attention model that captures the evolution of the action, through progressive sampling over fine-to-coarse scales. Our proposed Temporal Progressive (TemPr) model is composed of multiple attention towers, one for each scale. The predicted action label is based on the collective agreement considering confidences of these towers. Extensive experiments over four video datasets showcase state-of-the-art performance on the task of Early Action Prediction across a range of encoder architectures. We demonstrate the effectiveness and consistency of TemPr through detailed ablations.
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">†</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">†</sup>
Code is available at: https://tinyurl.com/temprog

        ----

        ## [1402] STMixer: A One-Stage Sparse Action Detector

        **Authors**: *Tao Wu, Mengqi Cao, Ziteng Gao, Gangshan Wu, Limin Wang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01414](https://doi.org/10.1109/CVPR52729.2023.01414)

        **Abstract**:

        Traditional video action detectors typically adopt the two-stage pipeline, where a person detector is first employed to generate actor boxes and then 3D RoIAlign is used to extract actor-specific features for classification. This detection paradigm requires multi-stage training and inference, and cannot capture context information outside the bounding box. Recently, a few query-based action detectors are proposed to predict action instances in an end-to-end manner. However, they still lack adaptability in feature sampling and decoding, thus suffering from the issues of inferior performance or slower convergence. In this paper, we propose a new one-stage sparse action detector, termed STMixer. STMixer is based on two core designs. First, we present a query-based adaptive feature sampling module, which endows our STMixer with the flexibility of mining a set of discriminative features from the entire spatiotemporal domain. Second, we devise a dual-branch feature mixing module, which allows our STMixer to dynamically attend to and mix video features along the spatial and the temporal dimension respectively for better feature decoding. Coupling these two designs with a video backbone yields an efficient end-to-end action detector. Without bells and whistles, our STMixer obtains the state-of-the-art results on the datasets of AVA, UCF101-24, and JHMDB.

        ----

        ## [1403] Generating Human Motion from Textual Descriptions with Discrete Representations

        **Authors**: *Jianrong Zhang, Yangsong Zhang, Xiaodong Cun, Yong Zhang, Hongwei Zhao, Hongtao Lu, Xi Shen, Shan Ying*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01415](https://doi.org/10.1109/CVPR52729.2023.01415)

        **Abstract**:

        In this work, we investigate a simple and must-known conditional generative framework based on Vector Quantised-Variational AutoEncoder (VQ-VAE) and Generative Pre-trained Transformer (GPT) for human motion generation from textural descriptions. We show that a simple CNN-based VQ-VAE with commonly used training recipes (EMA and Code Reset) allows us to obtain high-quality discrete representations. For GPT, we incorporate a simple corruption strategy during the training to alleviate training-testing discrepancy. Despite its simplicity, our T2M-GPT shows better performance than competitive approaches, including recent diffusion-based approaches. For example, on HumanML3D, which is currently the largest dataset, we achieve comparable performance on the consistency between text and generated motion (R-Precision), but with FID 0.116 largely outperforming MotionDiffuse of 0.630. Additionally, we conduct analyses on HumanML3D and observe that the dataset size is a limitation of our approach. Our work suggests that VQ-VAE still remains a competitive approach for human motion generation. Our implementation is available on the project page: https://mael-zys.github.io/T2M-GPT/.

        ----

        ## [1404] Cascade Evidential Learning for Open-world Weakly-supervised Temporal Action Localization

        **Authors**: *Mengyuan Chen, Junyu Gao, Changsheng Xu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01416](https://doi.org/10.1109/CVPR52729.2023.01416)

        **Abstract**:

        Targeting at recognizing and localizing action instances with only video-level labels during training, Weakly-supervised Temporal Action Localization (WTAL) has achieved significant progress in recent years. However, living in the dynamically changing open world where unknown actions constantly spring up, the closed-set assumption of existing WTAL methods is invalid. Compared with traditional open-set recognition tasks, Open-world WTAL (OW-TAL) is challenging since not only are the annotations of unknown samples unavailable, but also the fine-grained annotations of known action instances can only be inferred ambiguously from the video category labels. To address this problem, we propose a Cascade Evidential Learning framework at an evidence level, which targets at OWTAL for the first time. Our method jointly leverages multi-scale temporal contexts and knowledge-guided prototype information to progressively collect cascade and enhanced evidence for known action, unknown action, and background separation. Extensive experiments conducted on THUMOS-14 and ActivityNet-v1.3 verify the effectiveness of our method. Besides the classification metrics adopted by previous open-set recognition methods, we also evaluate our method on localization metrics which are more reasonable for OWTAL.

        ----

        ## [1405] Distilling Vision-Language Pre-Training to Collaborate with Weakly-Supervised Temporal Action Localization

        **Authors**: *Chen Ju, Kunhao Zheng, Jinxiang Liu, Peisen Zhao, Ya Zhang, Jianlong Chang, Qi Tian, Yanfeng Wang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01417](https://doi.org/10.1109/CVPR52729.2023.01417)

        **Abstract**:

        Weakly-supervised temporal action localization (WTAL) learns to detect and classify action instances with only category labels. Most methods widely adopt the off-the-shelf Classification-Based Pre-training (CBP) to generate video features for action localization. However, the different optimization objectives between classification and localization, make temporally localized results suffer from the serious incomplete issue. To tackle this issue without additional annotations, this paper considers to distill free action knowledge from Vision-Language Pre-training (VLP), as we surprisingly observe that the localization results of vanilla VLP have an over-complete issue, which is just complementary to the CBP results. To fuse such complementarity, we propose a novel distillation-collaboration framework with two branches acting as CBP and VLP respectively. The framework is optimized through a dual-branch alternate training strategy. Specifically, during the B step, we distill the confident background pseudo-labels from the CBP branch; while during the F step, the confident foreground pseudo-labels are distilled from the VLP branch. As a result, the dualbranch complementarity is effectively fused to promote one strong alliance. Extensive experiments and ablation studies on THUMOS14 and ActivityNet1.2 reveal that our method significantly outperforms state-of-the-art methods.

        ----

        ## [1406] Simultaneously Short- and Long-Term Temporal Modeling for Semi-Supervised Video Semantic Segmentation

        **Authors**: *Jiangwei Lao, Weixiang Hong, Xin Guo, Yingying Zhang, Jian Wang, Jingdong Chen, Wei Chu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01418](https://doi.org/10.1109/CVPR52729.2023.01418)

        **Abstract**:

        In order to tackle video semantic segmentation task at a lower cost, e.g., only one frame annotated per video, lots of efforts have been devoted to investigate the utilization of those unlabeled frames by either assigning pseudo labels or performing feature enhancement. In this work, we propose a novel feature enhancement network to simultaneously model short- and long-term temporal correlation. Compared with existing work that only leverage short-term correspondence, the long-term temporal correlation obtained from distant frames can effectively expand the temporal perception field and provide richer contextual prior. More importantly, modeling adjacent and distant frames together can alleviate the risk of over-fitting, hence produce high-quality feature representation for the distant unlabeled frames in training set and unseen videos in testing set. To this end, we term our method SSLTM, short for Simultaneously Short- and Long-Term Temporal Modeling. In the setting of only one frame annotated per video, SSLTM significantly outperforms the state-of-the-art methods by 2% ∼ 3% mIoU on the challenging VSPW dataset. Furthermore, when working with a pseudo label based method such as MeanTeacher, our final model only exhibits 0.13% mIoU less than the ceiling performance (i.e., all frames are manually annotated).

        ----

        ## [1407] MIST : Multi-modal Iterative Spatial-Temporal Transformer for Long-form Video Question Answering

        **Authors**: *Difei Gao, Luowei Zhou, Lei Ji, Linchao Zhu, Yi Yang, Mike Zheng Shou*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01419](https://doi.org/10.1109/CVPR52729.2023.01419)

        **Abstract**:

        To build Video Question Answering (VideoQA) systems capable of assisting humans in daily activities, seeking answers from long-form videos with diverse and complex events is a must. Existing multi-modal VQA models achieve promising performance on images or short video clips, especially with the recent success of large-scale multi-modal pre-training. However, when extending these methods to long-form videos, new challenges arise. On the one hand, using a dense video sampling strategy is computationally prohibitive. On the other hand, methods relying on sparse sampling struggle in scenarios where multi-event and multi-granularity visual reasoning are required. In this work, we introduce a new model named 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$\mathcal{M}ulti{-}$</tex>
 · modal Iterative 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$\mathcal{S}$</tex>
 .patial-temporal Transformer 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$(\mathcal{MIST})$</tex>
) to better adapt pre-trained models for long-form VideoQA. Specifically, 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$\mathcal{MIST}$</tex>
 decomposes traditional dense spatial-temporal self-attention into cascaded segment and region selection modules that adaptively select frames and image regions that are closely relevant to the question itself. Visual concepts at different granularities are then processed efficiently through an attention module. In addition, 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$\mathcal{MIST}$</tex>
 iteratively conducts selection and attention over multiple layers to support reasoning over multiple events. The experimental results on four VideoQA datasets, including AGQA, NExT-QA, STAR, and Env-QA, show that 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$\mathcal{MIST}$</tex>
 achieves state-of-the-art performance and is superior at efficiency. The code is available at github.com/showlab/mist.

        ----

        ## [1408] Language-Guided Music Recommendation for Video via Prompt Analogies

        **Authors**: *Daniel McKee, Justin Salamon, Josef Sivic, Bryan C. Russell*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01420](https://doi.org/10.1109/CVPR52729.2023.01420)

        **Abstract**:

        We propose a method to recommend music for an input video while allowing a user to guide music selection with free-form natural language. A key challenge of this problem setting is that existing music video datasets provide the needed (video, music) training pairs, but lack text descriptions of the music. This work addresses this challenge with the following three contributions. First, we propose a text-synthesis approach that relies on an analogy-based prompting procedure to generate natural language music descriptions from a large-scale language model (BLOOM-176B) given pre-trained music tagger outputs and a small number of human text descriptions. Second, we use these synthesized music descriptions to train a new trimodal model, which fuses text and video input representations to query music samples. For training, we introduce a text dropout regularization mechanism which we show is critical to model performance. Our model design allows for the re-trieved music audio to agree with the two input modalities by matching visual style depicted in the video and musical genre, mood, or instrumentation described in the natural language query. Third, to evaluate our approach, we collect a testing dataset for our problem by annotating a subset of 4k clips from the YT8M-Music Video dataset with natural language music descriptions which we make publicly available. We show that our approach can match or exceed the performance of prior methods on video-to-music retrieval while significantly improving retrieval accuracy when using text guidance.

        ----

        ## [1409] Text-Visual Prompting for Efficient 2D Temporal Video Grounding

        **Authors**: *Yimeng Zhang, Xin Chen, Jinghan Jia, Sijia Liu, Ke Ding*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01421](https://doi.org/10.1109/CVPR52729.2023.01421)

        **Abstract**:

        In this paper, we study the problem of temporal video grounding (TVG), which aims to predict the starting/ending time points of moments described by a text sentence within a long untrimmed video. Benefiting from fine-grained 3D visual features, the TVG techniques have achieved remarkable progress in recent years. However, the high complexity of 3D convolutional neural networks (CNNs) makes extracting dense 3D visual features time-consuming, which calls for intensive memory and computing resources. Towards efficient TVG, we propose a novel text-visual prompting (TVP) framework, which incorporates optimized perturbation patterns (that we call 'prompts') into both visual inputs and textual features of a TVG model. In sharp contrast to 3D CNNs, we show that TVP allows us to effectively co-train vision encoder and language encoder in a 2D TVG model and improves the performance of crossmodal feature fusion using only low-complexity sparse 2D visual features. Further, we propose a Temporal-Distance IoU (TDIoU) loss for efficient learning of TVG. Experiments on two benchmark datasets, Charades-STA and Activityblet Captions datasets, empirically show that the proposed TVP significantly boosts the performance of 2D TVG (e.g., 9.79% improvement on Charades-STA and 30.77% improvement on ActivityNet Captions) and achieves 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$5\times$</tex>
 inference acceleration over TVG using 3D visual features. Codes are available at Open.Intel.

        ----

        ## [1410] CelebV-Text: A Large-Scale Facial Text-Video Dataset

        **Authors**: *Jianhui Yu, Hao Zhu, Liming Jiang, Chen Change Loy, Weidong Cai, Wayne Wu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01422](https://doi.org/10.1109/CVPR52729.2023.01422)

        **Abstract**:

        Text-driven generation models are flourishing in video generation and editing. However, face-centric text-to-video generation remains a challenge due to the lack of a suitable dataset containing high-quality videos and highly relevant texts. This paper presents Celeb V- Text, a large-scale, di-verse, and high-quality dataset of facial text-video pairs, to facilitate research on facial text-to- video generation tasks. CelebV-Text comprises 70,000 in-the-wild face video clips with diverse visual content, each paired with 20 texts gen-erated using the proposed semi-automatic text generation strategy. The provided texts are of high quality, describing both static and dynamic attributes precisely. The supe-riority of CelebV- Text over other datasets is demonstrated via comprehensive statistical analysis of the videos, texts, and text-video relevance. The effectiveness and potential of CelebV- Text are further shown through extensive self-evaluation. A benchmark is constructed with representative methods to standardize the evaluation of the facial text-to-video generation task. All data and models are publicly available
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
Project page: https://celebv-text.github.io.

        ----

        ## [1411] CNVid-35M: Build, Filter, and Pre-Train the Large-Scale Public Chinese Video-Text Dataset

        **Authors**: *Tian Gan, Qing Wang, Xingning Dong, Xiangyuan Ren, Liqiang Nie, Qingpei Guo*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01423](https://doi.org/10.1109/CVPR52729.2023.01423)

        **Abstract**:

        Owing to well-designed large-scale video-text datasets, recent years have witnessed tremendous progress in video-text Pre-training. However, existing large-scale video-text datasets are mostly English-only. Though there are certain methods studying the Chinese video-text Pre-training, they pre-train their models on private datasets whose videos and text are unavailable. This lack of large-scale public datasets and benchmarks in Chinese hampers the research and downstream applications of Chinese video-text Pre-training. Towards this end, we release and benchmark CNVid-3.5M, a large-scale public cross-modal dataset containing over 3.5M Chinese video-text pairs. We summarize our contributions by three verbs, i.e., “Build”, “Filter”, and “Pretrain”: 1) To build a public Chinese video-text dataset, we collect over 4.5M videos from the Chinese websites. 2) To improve the data quality, we propose a novel method to filter out 1M weakly-paired videos, resulting in the CNVid-3.5M dataset. And 3) we benchmark CNVid-3.5M with three mainstream pixel-level Pre-training architectures. At last, we propose the Hard Sample Curriculum Learning strategy to promote the Pre-training performance. To the best of our knowledge, CNVid-3.5M is the largest public video-text dataset in Chinese, and we provide the first pixel-level benchmarks for Chinese video-text Pre-training. The dataset, codebase, and pre-trained models are available at https://github.com/CNVid/CNVid-3.5M.

        ----

        ## [1412] Learning Procedure-aware Video Representation from Instructional Videos and Their Narrations

        **Authors**: *Yiwu Zhong, Licheng Yu, Yang Bai, Shangwen Li, Xueting Yan, Yin Li*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01424](https://doi.org/10.1109/CVPR52729.2023.01424)

        **Abstract**:

        The abundance of instructional videos and their narrations over the Internet offers an exciting avenue for understanding procedural activities. In this work, we propose to learn video representation that encodes both action steps and their temporal ordering, based on a large-scale dataset of web instructional videos and their narrations, without using human annotations. Our method jointly learns a video representation to encode individual step concepts, and a deep probabilistic model to capture both temporal dependencies and immense individual variations in the step ordering. We empirically demonstrate that learning temporal ordering not only enables new capabilities for procedure reasoning, but also reinforces the recognition of individual steps. Our model significantly advances the state-of-the-art results on step classification (+2.8%/+3.3% on COIN / EPIC-Kitchens) and step forecasting (+7.4% on COIN). Moreover, our model attains promising results in zero-shot inference for step classification and fore-casting, as well as in predicting diverse and plausible steps for incomplete procedures. Our code is available at https://github.com/facebookresearch/ProcedureVRL.

        ----

        ## [1413] PDPP: Projected Diffusion for Procedure Planning in Instructional Videos

        **Authors**: *Hanlin Wang, Yilu Wu, Sheng Guo, Limin Wang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01425](https://doi.org/10.1109/CVPR52729.2023.01425)

        **Abstract**:

        In this paper, we study the problem of procedure planning in instructional videos, which aims to make goal-directed plans given the current visual observations in unstructured real-life videos. Previous works cast this problem as a sequence planning problem and leverage either heavy intermediate visual observations or natural language instructions as supervision, resulting in complex learning schemes and expensive annotation costs. In contrast, we treat this problem as a distribution fitting problem. In this sense, we model the whole intermediate action sequence distribution with a diffusion model (PDPP), and thus transform the planning problem to a sampling process from this distribution. In addition, we remove the expensive intermediate supervision, and simply use task labels from instructional videos as supervision instead. Our model is a U-Net based diffusion model, which directly samples action sequences from the learned distribution with the given start and end observations. Furthermore, we apply an efficient projection method to provide accurate conditional guides for our model during the learning and sampling process. Experiments on three datasets with different scales show that our PDPP model can achieve the state-of-the-art performance on multiple metrics, even without the task supervision. Code and trained models are available at https://github.com/MCG-NJU/PDPP.

        ----

        ## [1414] Towards Fast Adaptation of Pretrained Contrastive Models for Multi-channel Video-Language Retrieval

        **Authors**: *Xudong Lin, Simran Tiwari, Shiyuan Huang, Manling Li, Mike Zheng Shou, Heng Ji, Shih-Fu Chang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01426](https://doi.org/10.1109/CVPR52729.2023.01426)

        **Abstract**:

        Multi-channel video-language retrieval require models to understand information from different channels (e.g. video+question, video+speech) to correctly link a video with a textual response or query. Fortunately, contrastive multimodal models are shown to be highly effective at aligning entities in images/videos and text, e.g., CLIP [20]; text contrastive models are extensively studied recently for their strong ability of producing discriminative sentence embeddings, e.g., SimCSE [5]. However, there is not a clear way to quickly adapt these two lines to multi-channel video-language retrieval with limited data and resources. In this paper, we identify a principled model design space with two axes: how to represent videos and how to fuse video and text information. Based on categorization of recent methods, we investigate the options of representing videos using continuous feature vectors or discrete text tokens; for the fusion method, we explore the use of a multimodal transformer or a pretrained contrastive text model. We extensively evaluate the four combinations on five video-language datasets. We surprisingly find that discrete text tokens coupled with a pretrained contrastive text model yields the best performance, which can even outperform state-of-the-art on the iVQA and How2QA datasets without additional training on millions of video-text data. Further analysis shows that this is because representing videos as text tokens captures the key visual information and text tokens are naturally aligned with text models that are strong retrievers after the contrastive pretraining process. All the empirical analysis establishes a solid foundation for future research on affordable and upgradable multimodal intelligence.

        ----

        ## [1415] Clover: Towards A Unified Video-Language Alignment and Fusion Model

        **Authors**: *Jingjia Huang, Yinan Li, Jiashi Feng, Xinglong Wu, Xiaoshuai Sun, Rongrong Ji*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01427](https://doi.org/10.1109/CVPR52729.2023.01427)

        **Abstract**:

        Building a universal Video-Language model for solving various video understanding tasks (e.g., text-video retrieval, video question answering) is an open challenge to the machine learning field. Towards this goal, most recent works build the model by stacking uni-modal and cross-modal feature encoders and train it with pair-wise contrastive pre-text tasks. Though offering attractive generality, the resulted models have to compromise between efficiency and performance. They mostly adopt different architectures to deal with different downstream tasks. We find this is because the pair-wise training cannot well align and fuse features from different modalities. We then introduce Clover-a Correlated Video-Language pre-training method-towards a universal Video-Language modelfor solving multiple video understanding tasks with neither performance nor efficiency compromise. It improves cross-modal feature alignment and fusion via a novel tri-modal alignment pre-training task. Additionally, we propose to enhance the tri-modal alignment via incorporating learning from semantic masked samples and a new pair-wise ranking loss. Clover establishes new state-of-the-arts on multiple downstream tasks, including three retrieval tasks for both zero-shot and fine-tuning settings, and eight video question answering tasks. Codes and pre-trained models will be released at https://github.com/LeeYN-43/Clover.

        ----

        ## [1416] Align and Attend: Multimodal Summarization with Dual Contrastive Losses

        **Authors**: *Bo He, Jun Wang, Jielin Qiu, Trung Bui, Abhinav Shrivastava, Zhaowen Wang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01428](https://doi.org/10.1109/CVPR52729.2023.01428)

        **Abstract**:

        The goal of multimodal summarization is to extract the most important information from different modalities to form summaries. Unlike unimodal summarization, the multimodal summarization task explicitly leverages cross-modal information to help generate more reliable and high-quality summaries. However, existing methods fail to lever-age the temporal correspondence between different modal-ities and ignore the intrinsic correlation between different samples. To address this issue, we introduce Align and Attend Multimodal Summarization (A2Summ), a unified multimodal transformer-based model which can effectively align and attend the multimodal input. In addition, we propose two novel contrastive losses to model both inter-sample and intra-sample correlations. Extensive experiments on two standard video summarization datasets (TVSum and SumMe) and two multimodal summarization datasets (Daily Mail and CNN) demonstrate the superiority of A2Summ, achieving state-of-the-art performances on all datasets. Moreover, we collected a large-scale multimodal summarization dataset BLiSS, which contains livestream videos and transcribed texts with annotated summaries. Our code and dataset are publicly available at https://boheumd.github.io/A2Summ/.

        ----

        ## [1417] Learning Situation Hyper-Graphs for Video Question Answering

        **Authors**: *Aisha Urooj Khan, Hilde Kuehne, Bo Wu, Kim Chheu, Walid Bousselham, Chuang Gan, Niels da Vitoria Lobo, Mubarak Shah*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01429](https://doi.org/10.1109/CVPR52729.2023.01429)

        **Abstract**:

        Answering questions about complex situations in videos requires not only capturing the presence of actors, objects, and their relations but also the evolution of these relationships over time. A situation hyper-graph is a representation that describes situations as scene sub-graphs for video frames and hyper-edges for connected sub-graphs and has been proposed to capture all such information in a compact structured form. In this work, we propose an architecture for Video Question Answering (VQA) that enables answering questions related to video content by predicting situation hyper-graphs, coined Situation Hyper-Graph based Video Question Answering (SHG- VQA). To this end, we train a situation hyper-graph decoder to implicitly identify graph representations with actions and object/human-object relationships from the input video clip. and to use cross-attention between the predicted situation hyper-graphs and the question embedding to predict the correct answer. The proposed method is trained in an end-to-end manner and optimized by a VQA loss with the cross-entropy function and a Hungarian matching loss for the situation graph prediction. The effectiveness of the proposed architecture is extensively evaluated on two challenging benchmarks: AGQA and STAR. Our results show that learning the underlying situation hyper-graphs helps the system to significantly improve its performance for novel challenges of video question-answering tasks
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
Code will be available at https://github.com/aurooj/SHG-VQA.

        ----

        ## [1418] Natural Language-Assisted Sign Language Recognition

        **Authors**: *Ronglai Zuo, Fangyun Wei, Brian Mak*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01430](https://doi.org/10.1109/CVPR52729.2023.01430)

        **Abstract**:

        Sign languages are visual languages which convey in-formation by signers' handshape, facial expression, body movement, and so forth. Due to the inherent restriction of combinations of these visual ingredients, there exist a significant number of visually indistinguishable signs (VISigns) in sign languages, which limits the recognition capacity of vision neural networks. To mitigate the problem, we propose the Natural Language-Assisted Sign Language Recognition (NLA-SLR) framework, which exploits semantic information contained in glosses (sign labels). First, for VISigns with similar semantic meanings, we propose language-aware label smoothing by generating soft labels for each training sign whose smoothing weights are computed from the normalized semantic similarities among the glosses to ease training. Second, for VISigns with distinct semantic meanings, we present an inter-modality mixup technique which blends vision and gloss features to further maximize the separability of different signs under the super-vision of blended labels. Besides, we also introduce a novel backbone, video-keypoint network, which not only models both RGB videos and human body keypoints but also derives knowledge from sign videos of different temporal receptive fields. Empirically, our method achieves state-of-the-art performance on three widely-adopted benchmarks: MSASL, WLASL, and NMFs-CSL. Codes are available at https://github.com/FangyunWeilSLRT.

        ----

        ## [1419] SkyEye: Self-Supervised Bird's-Eye-View Semantic Mapping Using Monocular Frontal View Images

        **Authors**: *Nikhil Gosala, Kürsat Petek, Paulo L. J. Drews-Jr, Wolfram Burgard, Abhinav Valada*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01431](https://doi.org/10.1109/CVPR52729.2023.01431)

        **Abstract**:

        Bird's-Eye-View (BEV) semantic maps have become an essential component of automated driving pipelines due to the rich representation they provide for decision-making tasks. However, existing approaches for generating these maps still follow a fully supervised training paradigm and hence rely on large amounts of annotated BEV data. In this work, we address this limitation by proposing the first self-supervised approach for generating a BEV semantic map using a single monocular image from the frontal view (FV). During training, we overcome the need for BEV ground truth annotations by leveraging the more easily available FV semantic annotations of video sequences. Thus, we propose the SkyEye architecture that learns based on two modes of self-supervision, namely, implicit supervision and explicit supervision. Implicit supervision trains the model by enforcing spatial consistency of the scene over time based on FV semantic sequences, while explicit supervision exploits BEV pseudolabels generated from FV semantic annotations and self-supervised depth estimates. Extensive evaluations on the KITTI-360 dataset demonstrate that our self-supervised approach performs on par with the state-of-the-art fully supervised methods and achieves competitive results using only 1 % of direct supervision in BEV compared to fully supervised approaches. Finally, we publicly release both our code and the BEV datasets generated from the KITTI-360 and Waymo datasets.

        ----

        ## [1420] Adaptive Zone-aware Hierarchical Planner for Vision-Language Navigation

        **Authors**: *Chen Gao, Xingyu Peng, Mi Yan, He Wang, Lirong Yang, Haibing Ren, Hongsheng Li, Si Liu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01432](https://doi.org/10.1109/CVPR52729.2023.01432)

        **Abstract**:

        The task of Vision-Language Navigation (VLN) is for an embodied agent to reach the global goal according to the instruction. Essentially, during navigation, a series of sub-goals need to be adaptively set and achieved, which is naturally a hierarchical navigation process. However, previous methods leverage a single-step planning scheme, i.e., directly performing navigation action at each step, which is unsuitable for such a hierarchical navigation process. In this paper, we propose an Adaptive Zone-aware Hierarchical Planner (AZHP) to explicitly divides the navigation process into two heterogeneous phases, i.e., sub-goal setting via zone partition/selection (high-level action) and sub-goal executing (low-level action), for hierarchical planning. Specifically, AZHP asynchronously performs two levels of action via the designed State-Switcher Module (SSM). For high-level action, we devise a Scene-aware adaptive Zone Partition (SZP) method to adaptively divide the whole navigation area into different zones on-the-fly. Then the Goal-oriented Zone Selection (GZS) method is proposed to select a proper zone for the current sub-goal. For low-level action, the agent conducts navigation-decision multi-steps in the selected zone. Moreover, we design a Hierarchical RL (HRL) strategy and auxiliary losses with curriculum learning to train the AZHP framework, which provides effective supervision signals for each stage. Extensive experiments demonstrate the superiority of our proposed method, which achieves state-of-the-art performance on three VLN benchmarks (REVERIE, SOON, R2R).

        ----

        ## [1421] Iterative Vision-and-Language Navigation

        **Authors**: *Jacob Krantz, Shurjo Banerjee, Wang Zhu, Jason J. Corso, Peter Anderson, Stefan Lee, Jesse Thomason*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01433](https://doi.org/10.1109/CVPR52729.2023.01433)

        **Abstract**:

        We present Iterative Vision-and-Language Navigation (IVLN), a paradigm for evaluating language-guided agents navigating in a persistent environment over time. Existing Vision-and-Language Navigation (VLN) benchmarks erase the agent's memory at the beginning of every episode, testing the ability to perform cold-start navigation with no prior information. However, deployed robots occupy the same environment for long periods of time. The IVLN paradigm addresses this disparity by training and evaluating VLN agents that maintain memory across tours of scenes that consist of up to 100 ordered instruction-following Room-to-Room (R2R) episodes, each defined by an individual language instruction and a target path. We present discrete and continuous Iterative Room-to-Room (IR2R) benchmarks comprising about 400 tours each in 80 indoor scenes. We find that extending the implicit memory of high-performing transformer VLN agents is not sufficient for IVLN, but agents that build maps can benefit from environment persistence, motivating a renewed focus on map-building agents in VLN.

        ----

        ## [1422] EXCALIBUR: Encouraging and Evaluating Embodied Exploration

        **Authors**: *Hao Zhu, Raghav Kapoor, So Yeon Min, Winson Han, Jiatai Li, Kaiwen Geng, Graham Neubig, Yonatan Bisk, Aniruddha Kembhavi, Luca Weihs*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01434](https://doi.org/10.1109/CVPR52729.2023.01434)

        **Abstract**:

        Experience precedes understanding. Humans constantly explore and learn about their environment out of curiosity, gather information, and update their models of the world. On the other hand, machines are either trained to learn passively from static and fixed datasets, or taught to complete specific goal-conditioned tasks. To encourage the development of exploratory interactive agents, we present the EXCALIBUR benchmark. EXCALIBUR allows agents to explore their environment for long durations and then query their understanding of the physical world via inquiries like: “is the small heavy red bowl made from glass?” or “is there a silver spoon heavier than the egg?”. This design encourages agents to perform free-form home exploration without myopia induced by goal conditioning. Once the agents have answered a series of questions, they can renter the scene to refine their knowledge, update their beliefs, and improve their performance on the questions. Our experiments demonstrate the challenges posed by this dataset for the present-day state-of-the-art embodied systems and the headroom afforded to develop new innovative methods. Finally, we present a virtual reality interface that enables humans to seamlessly interact within the simulated world and use it to gather human performance measures. EXCALIBUR affords unique challenges in comparison to presentday benchmarks and represents the next frontier for embodied AI research.

        ----

        ## [1423] Multimodal Prompting with Missing Modalities for Visual Recognition

        **Authors**: *Yi-Lun Lee, Yi-Hsuan Tsai, Wei-Chen Chiu, Chen-Yu Lee*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01435](https://doi.org/10.1109/CVPR52729.2023.01435)

        **Abstract**:

        In this paper, we tackle two challenges in multimodal learning for visual recognition: 1) when missing-modality occurs either during training or testing in real-world situations; and 2) when the computation resources are not available to finetune on heavy transformer models. To this end, we propose to utilize prompt learning and mitigate the above two challenges together. Specifically, our modality-missing-aware prompts can be plugged into multimodal transformers to handle general missing-modality cases, while only requiring less than 1% learnable parameters compared to training the entire model. We further explore the effect of different prompt configurations and analyze the robustness to missing modality. Extensive experiments are conducted to show the effectiveness of our prompt learning framework that improves the performance under various missing-modality cases, while alleviating the requirement of heavy model retraining. Code is available.
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
https://github.com/YiLunLee/missing_aware_prompts

        ----

        ## [1424] Visual Programming: Compositional visual reasoning without training

        **Authors**: *Tanmay Gupta, Aniruddha Kembhavi*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01436](https://doi.org/10.1109/CVPR52729.2023.01436)

        **Abstract**:

        We present Visprog, a neuro-symbolic approach to solving complex and compositional visual tasks given natural language instructions. Visprog avoids the need for any task-specific training. Instead, it uses the incontext learning ability of large language models to generate python-like modular programs, which are then executed to get both the solution and a comprehensive and interpretable rationale. Each line of the generated program may invoke one of several off-the-shelf computer vision models, image processing subroutines, or python functions to produce intermediate outputs that may be consumed by subsequent parts of the program. We demonstrate the flexibility of VIsPROG on 4 diverse tasks - compositional visual question answering, zero-shot reasoning on image pairs, factual knowledge object tagging, and language-guided image editing. We believe neuro-symbolic approaches like Visprog are an exciting avenue to easily and effectively expand the scope of AI systems to serve the long tail of complex tasks that people may wish to perform.

        ----

        ## [1425] Super-CLEVR: A Virtual Benchmark to Diagnose Domain Robustness in Visual Reasoning

        **Authors**: *Zhuowan Li, Xingrui Wang, Elias Stengel-Eskin, Adam Kortylewski, Wufei Ma, Benjamin Van Durme, Alan L. Yuille*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01437](https://doi.org/10.1109/CVPR52729.2023.01437)

        **Abstract**:

        Visual Question Answering (VQA) models often perform poorly on out-of-distribution data and struggle on domain generalization. Due to the multi-modal nature of this task, multiple factors of variation are intertwined, making generalization difficult to analyze. This motivates us to introduce a virtual benchmark, Super-CLEVR, where different factors in VQA domain shifts can be isolated in order that their effects can be studied independently. Four factors are considered: visual complexity, question redundancy, concept distribution and concept compositionality. With controllably generated data, Super-CLEVR enables us to test VQA methods in situations where the test data differs from the training data along each of these axes. We study four existing methods, including two neural symbolic methods NSCL [45] and NSVQA [59], and two non-symbolic methods FiLM [50] and mDETR [29]; and our proposed method, probabilistic NSVQA (P-NSVQA), which extends NSVQA with uncertainty reasoning. P-NSVQA outperforms other methods on three of the four domain shift factors. Our results suggest that disentangling reasoning and perception, combined with probabilistic uncertainty, form a strong VQA model that is more robust to domain shifts. The dataset and code are released at https://github.com/Lizw14/Super-CLEVR.

        ----

        ## [1426] Prompting Large Language Models with Answer Heuristics for Knowledge-Based Visual Question Answering

        **Authors**: *Zhenwei Shao, Zhou Yu, Meng Wang, Jun Yu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01438](https://doi.org/10.1109/CVPR52729.2023.01438)

        **Abstract**:

        Knowledge-based visual question answering (VQA) requires external knowledge beyond the image to answer the question. Early studies retrieve required knowledge from explicit knowledge bases (KBs), which often introduces irrelevant information to the question, hence restricting the performance of their models. Recent works have sought to use a large language model (i.e., GPT-3 [3]) as an implicit knowledge engine to acquire the necessary knowledge for answering. Despite the encouraging results achieved by these methods, we argue that they have not fully activated the capacity of GPT-3 as the provided input information is insufficient. In this paper, we present Prophet-a conceptually simple framework designed to 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$prompt$</tex>
 GPT-3 with answer heuristics for knowledge-based VQA. Specifically, we first train a vanilla VQA model on a specific knowledge-based VQA dataset without external knowledge. After that, we extract two types of complementary answer heuristics from the model: answer candidates and answer-aware examples. Finally, the two types of answer heuristics are encoded into the prompts to enable GPT-3 to better comprehend the task thus enhancing its capacity. Prophet significantly outperforms all existing state-of-the-art methods on two challenging knowledge-based VQA datasets, OK-VQA and A-OKVQA, delivering 61.1% and 55.7% accuracies on their testing sets, respectively.

        ----

        ## [1427] À-la-carte Prompt Tuning (APT): Combining Distinct Data Via Composable Prompting

        **Authors**: *Benjamin Bowman, Alessandro Achille, Luca Zancato, Matthew Trager, Pramuditha Perera, Giovanni Paolini, Stefano Soatto*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01439](https://doi.org/10.1109/CVPR52729.2023.01439)

        **Abstract**:

        We introduce À-la-carte Prompt Tuning (APT), a transformer-based scheme to tune prompts on distinct data so that they can be arbitrarily composed at inference time. The individual prompts can be trained in isolation, possibly on different devices, at different times, and on different distributions or domains. Furthermore each prompt only contains information about the subset of data it was exposed to during training. During inference, models can be assembled based on arbitrary selections of data sources, which we call à-la-carte learning. À-la-carte learning enables constructing bespoke models specific to each user's individual access rights and preferences. We can add or remove information from the model by simply adding or removing the corresponding prompts without retraining from scratch. We demonstrate that à-la-carte built models achieve accuracy within 5% of models trained on the union of the respective sources, with comparable cost in terms of training and inference time. For the continual learning benchmarks Split CIFAR- 100 and CORe50, we achieve state-of-the-art performance.

        ----

        ## [1428] ConStruct-VL: Data-Free Continual Structured VL Concepts Learning

        **Authors**: *James Seale Smith, Paola Cascante-Bonilla, Assaf Arbelle, Donghyun Kim, Rameswar Panda, David D. Cox, Diyi Yang, Zsolt Kira, Rogério Feris, Leonid Karlinsky*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01440](https://doi.org/10.1109/CVPR52729.2023.01440)

        **Abstract**:

        Recently, large-scale pre-trained Vision-and-Language (VL) foundation models have demonstrated remarkable capabilities in many zero-shot downstream tasks, achieving competitive results for recognizing objects defined by as little as short text prompts. However, it has also been shown that VL models are still brittle in Structured VL Concept (SVLC) reasoning, such as the ability to recognize object attributes, states, and inter-object relations. This leads to reasoning mistakes, which need to be corrected as they occur by teaching VL models the missing SVLC skills; often this must be done using private data where the issue was found, which naturally leads to a data-free continual (no task-id) VL learning setting. In this work, we introduce the first Continual Data-Free Structured VL Concepts Learning (ConStruct-VL) benchmark
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
Our code is publicly available at https://github.com/jamessealesmith/ConStruct-VL and show it is challenging for many existing data-free CL strategies. We, therefore, propose a data-free method comprised of a new approach of Adversarial Pseudo-Replay (APR) which generates adversarial reminders of past tasks from past task models. To use this method efficiently, we also propose a continual parameter-efficient Layered-LoRA (LaLo) neural architecture allowing no-memory-cost access to all past models at train time. We show this approach outperforms all data-free methods by as much as ~ 7% while even matching some levels of experience-replay (prohibitive for applications where data-privacy must be preserved).

        ----

        ## [1429] Q: How to Specialize Large Vision-Language Models to Data-Scarce VQA Tasks? A: Self-Train on Unlabeled Images!

        **Authors**: *Zaid Khan, B. G. Vijay Kumar, Samuel Schulter, Xiang Yu, Yun Fu, Manmohan Chandraker*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01441](https://doi.org/10.1109/CVPR52729.2023.01441)

        **Abstract**:

        Finetuning a large vision language model (VLM) on a target dataset after large scale pretraining is a dominant paradigm in visual question answering (VQA). Datasets for specialized tasks such as knowledge-based VQA or VQA in non natural-image domains are orders of magnitude smaller than those for general-purpose VQA. While collecting additional labels for specialized tasks or domains can be challenging, unlabeled images are often available. We introduce SelTDA (Self-Taught Data Augmentation), a strategy for finetuning large VLMs on small-scale VQA datasets. SelTDA uses the VLM and target dataset to build a teacher model that can generate question-answer pseudolabels directly conditioned on an image alone, allowing us to pseudolabel unlabeled images. SelTDA then finetunes the initial VLM on the original dataset augmented with freshly pseudolabeled images. We describe a series of experiments showing that our self-taught data augmentation increases robustness to adversarially searched questions, counterfactual examples and rephrasings, improves domain generalization, and results in greater retention of numerical reasoning skills. The proposed strategy requires no additional annotations or architectural modifications, and is compatible with any modern encoder-decoder multimodal transformer. Code available at https://github.com/codezakh/SelTDA.

        ----

        ## [1430] Learning to Exploit Temporal Structure for Biomedical Vision-Language Processing

        **Authors**: *Shruthi Bannur, Stephanie L. Hyland, Qianchu Liu, Fernando Pérez-García, Maximilian Ilse, Daniel C. Castro, Benedikt Boecking, Harshita Sharma, Kenza Bouzid, Anja Thieme, Anton Schwaighofer, Maria Wetscherek, Matthew P. Lungren, Aditya V. Nori, Javier Alvarez-Valle, Ozan Oktay*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01442](https://doi.org/10.1109/CVPR52729.2023.01442)

        **Abstract**:

        Self-supervised learning in vision-language processing (VLP) exploits semantic alignment between imaging and text modalities. Prior work in biomedical VLP has mostly relied on the alignment of single image and report pairs even though clinical notes commonly refer to prior images. This does not only introduce poor alignment between the modalities but also a missed opportunity to exploit rich self-supervision through existing temporal content in the data. In this work, we explicitly account for prior images and reports when available during both training and fine-tuning. Our approach, named BioViL-T, uses a CNN-Transformer hybrid multi-image encoder trained jointly with a text model. It is designed to be versatile to arising challenges such as pose variations and missing input images across time. The resulting model excels on downstream tasks both in single- and multi-image setups, achieving state-of-the-art (SOTA) performance on (I) progression classification, (II) phrase grounding, and (III) report generation, whilst offering consistent improvements on disease classification and sentence-similarity tasks. We release a novel multi-modal temporal benchmark dataset, MS-CXR-T, to quantify the quality of vision-language representations in terms of temporal semantics. Our experimental results show the advantages of incorporating prior images and reports to make most use of the data.

        ----

        ## [1431] FashionSAP: Symbols and Attributes Prompt for Fine-Grained Fashion Vision-Language Pre-Training

        **Authors**: *Yunpeng Han, Lisai Zhang, Qingcai Chen, Zhijian Chen, Zhonghua Li, Jianxin Yang, Zhao Cao*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01443](https://doi.org/10.1109/CVPR52729.2023.01443)

        **Abstract**:

        Fashion vision-language pre-training models have shown efficacy for a wide range of downstream tasks. However, general vision-language pre-training models pay less attention to fine-grained domain features, while these features are important in distinguishing the specific domain tasks from general tasks. We propose a method for fine-grained fashion vision-language pre-training based on fashion Symbols and Attributes Prompt (FashionSAP) to model fine-grained multi-modalities fashion attributes and characteristics. Firstly, we propose the fashion symbols, a novel abstract fashion concept layer, to represent different fashion items and to generalize various kinds of fine- grained fashion features, making modelling fine-grained attributes more effective. Secondly, the attributes prompt method is proposed to make the model learn specific attributes of fashion items explicitly. We design proper prompt templates according to the format of fashion data. Comprehensive experiments are conducted on two public fashion benchmarks, i.e., FashionGen and FashionIQ, and FashionSAP gets SOTA performances for four popular fashion tasks. The ablation study also shows the proposed abstract fashion symbols, and the attribute prompt method enables the model to acquire fine-grained semantics in the fashion domain effectively. The obvious performance gains from FashionSAP provide a new baseline for future fashion task research.
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
The source code is available at https://github.com/hssip/FashionSAP

        ----

        ## [1432] Advancing Visual Grounding with Scene Knowledge: Benchmark and Method

        **Authors**: *Zhihong Chen, Ruifei Zhang, Yibing Song, Xiang Wan, Guanbin Li*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01444](https://doi.org/10.1109/CVPR52729.2023.01444)

        **Abstract**:

        Visual grounding (VG) aims to establish fine-grained alignment between vision and language. Ideally, it can be a testbed for vision-and-language models to evaluate their understanding of the images and texts and their reasoning abilities over their joint space. However, most existing VG datasets are constructed using simple description texts, which do not require sufficient reasoning over the images and texts. This has been demonstrated in a recent study [27], where a simple LSTM-based text encoder without pretraining can achieve state-of-the-art performance on mainstream VG datasets. Therefore, in this paper, we propose a novel benchmark of Scene Knowledge-guided Visual Grounding (SK-VG), where the image content and referring expressions are not sufficient to ground the target objects, forcing the models to have a reasoning ability on the long-form scene knowledge. To perform this task, we propose two approaches to accept the triple-type input, where the former embeds knowledge into the image features before the image-query interaction; the latter leverages linguistic structure to assist in computing the image-text matching. We conduct extensive experiments to analyze the above methods and show that the proposed approaches achieve promising results but still leave room for improvement, including performance and interpretability. The dataset and code are available at https://github.com/zhjohnchan/SK-VG.

        ----

        ## [1433] Beyond Appearance: A Semantic Controllable Self-Supervised Learning Framework for Human-Centric Visual Tasks

        **Authors**: *Weihua Chen, Xianzhe Xu, Jian Jia, Hao Luo, Yaohua Wang, Fan Wang, Rong Jin, Xiuyu Sun*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01445](https://doi.org/10.1109/CVPR52729.2023.01445)

        **Abstract**:

        Human-centric visual tasks have attracted increasing research attention due to their widespread applications. In this paper, we aim to learn a general human representation from massive unlabeled human images which can benefit downstream human-centric tasks to the maximum extent. We call this method SOLIDER, a Semantic cOntrollable seLf-supervIseD lEaRning framework. Unlike the existing self-supervised learning methods, prior knowledge from human images is utilized in SOLIDER to build pseudo semantic labels and import more semantic information into the learned representation. Meanwhile, we note that different downstream tasks always require different ratios of semantic information and appearance information. For example, human parsing requires more semantic information, while person re-identification needs more appearance information for identification purpose. So a single learned representation cannot fit for all requirements. To solve this problem, SOLIDER introduces a conditional network with a semantic controller. After the model is trained, users can send values to the controller to produce representations with different ratios of semantic information, which can fit different needs of downstream tasks. Finally, SOLIDER is verified on six downstream human-centric visual tasks. It outperforms state of the arts and builds new baselines for these tasks. The code is released in https://github.com/tinyvision/SOLIDER.

        ----

        ## [1434] OCTET: Object-aware Counterfactual Explanations

        **Authors**: *Mehdi Zemni, Mickaël Chen, Éloi Zablocki, Hédi Ben-Younes, Patrick Pérez, Matthieu Cord*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01446](https://doi.org/10.1109/CVPR52729.2023.01446)

        **Abstract**:

        Nowadays, deep vision models are being widely deployed in safety-critical applications, e.g., autonomous driving, and explainability of such models is becoming a pressing concern. Among explanation methods, counter-factual explanations aim to find minimal and interpretable changes to the input image that would also change the output of the model to be explained. Such explanations point end-users at the main factors that impact the decision of the model. However, previous methods struggle to explain decision models trained on images with many objects, e.g., urban scenes, which are more difficult to work with but also arguably more critical to explain. In this work, we propose to tackle this issue with an object-centric framework for counterfactual explanation generation. Our method, inspired by recent generative modeling works, encodes the query image into a latent space that is structured in a way to ease object-level manipulations. Doing so, it provides the end-user with control over which search directions (e.g., spatial displacement of objects, style modification, etc.) are to be explored during the counterfactual generation. We conduct a set of experiments on counterfactual explanation benchmarks for driving scenes, and we show that our method can be adapted beyond classification, e.g., to explain semantic segmentation models. To complete our analysis, we design and run a user study that measures the usefulness of counterfactual explanations in understanding a decision model. Code is available at https://github.com/valeoai/OCTET.

        ----

        ## [1435] Local-Guided Global: Paired Similarity Representation for Visual Reinforcement Learning

        **Authors**: *Hyesong Choi, Hunsang Lee, Wonil Song, Sangryul Jeon, Kwanghoon Sohn, Dongbo Min*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01447](https://doi.org/10.1109/CVPR52729.2023.01447)

        **Abstract**:

        Recent vision-based reinforcement learning (RL) methods have found extracting high-level features from raw pixels with self-supervised learning to be effective in learning policies. However, these methods focus on learning global representations of images, and disregard local spatial structures present in the consecutively stacked frames. In this paper, we propose a novel approach, termed self-supervised Paired Similarity Representation Learning (PSRL) for effectively encoding spatial structures in an unsupervised manner. Given the input frames, the latent volumes are first generated individually using an encoder, and they are used to capture the variance in terms of local spatial structures, i.e., correspondence maps among multiple frames. This enables for providing plenty of fine-grained samples for training the encoder of deep RL. We further attempt to learn the global semantic representations in the action aware transform module that predicts future state representations using action vectors as a medium. The proposed method imposes similarity constraints on the three latent volumes; transformed query representations by estimated pixel-wise correspondence, predicted query representations from the action aware transform model, and target representations of future state, guiding action aware transform with locality-inherent volume. Experimental results on complex tasks in Atari Games and DeepMind Control Suite demonstrate that the RL methods are significantly boosted by the proposed self-supervised learning of paired similarity representations.

        ----

        ## [1436] What Can Human Sketches Do for Object Detection?

        **Authors**: *Pinaki Nath Chowdhury, Ayan Kumar Bhunia, Aneeshan Sain, Subhadeep Koley, Tao Xiang, Yi-Zhe Song*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01448](https://doi.org/10.1109/CVPR52729.2023.01448)

        **Abstract**:

        Sketches are highly expressive, inherently capturing subjective and fine-grained visual cues. The exploration of such innate properties of human sketches has, however, been limited to that of image retrieval. In this paper, for the first time, we cultivate the expressiveness of sketches but for the fundamental vision task of object detection. The end result is a sketch-enabled object detection framework that detects based on what you sketch – that “zebra” (e.g., one that is eating the grass) in a herd of zebras (instance-aware detection), and only the part (e.g., “head” of a “zebra”) that you desire (part-aware detection). We further dictate that our model works without (i) knowing which category to expect at testing (zero-shot) and (ii) not requiring additional bounding boxes (as per fully supervised) and class labels (as per weakly supervised). Instead of devising a model from the ground up, we show an intuitive synergy between foundation models (e.g., CLIP) and existing sketch models build for sketch-based image retrieval (SBIR), which can already elegantly solve the task – CLIP to provide model generalisation, and SBIR to bridge the (sketch→photo) gap. In particular, we first perform independent prompting on both sketch and photo branches of an SBIR model to build highly generalisable sketch and photo encoders on the back of the generalisation ability of CLIP. We then devise a training paradigm to adapt the learned encoders for object detection, such that the region embeddings of detected boxes are aligned with the sketch and photo embeddings from SBIR. Evaluating our framework on standard object detection datasets like PASCAL-VOC and MS-COCO outperforms both supervised (SOD) and weakly-supervised object detectors (WSOD) on zero-shot setups. Project Page: https://pinakinathc.github.io/sketch-detect

        ----

        ## [1437] Revisiting Multimodal Representation in Contrastive Learning: From Patch and Token Embeddings to Finite Discrete Tokens

        **Authors**: *Yuxiao Chen, Jianbo Yuan, Yu Tian, Shijie Geng, Xinyu Li, Ding Zhou, Dimitris N. Metaxas, Hongxia Yang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01449](https://doi.org/10.1109/CVPR52729.2023.01449)

        **Abstract**:

        Contrastive learning-based vision-language pretraining approaches, such as CLIP, have demonstrated great success in many vision-language tasks. These methods achieve cross-modal alignment by encoding a matched image-text pair with similar feature embeddings, which are generated by aggregating information from visual patches and language tokens. However, direct aligning cross-modal information using such representations is challenging, as visual patches and text tokens differ in semantic levels and granularities. To alleviate this issue, we propose a Finite Discrete Tokens (FDT) based multimodal representation. FDT is a set of learnable tokens representing certain visualsemantic concepts. Both images and texts are embedded using shared FDT by first grounding multimodal inputs to FDT space and then aggregating the activated FDT representations. The matched visual and semantic concepts are enforced to be represented by the same set of discrete tokens by a sparse activation constraint. As a result, the granularity gap between the two modalities is reduced. Through both quantitative and qualitative analyses, we demonstrate that using FDT representations in CLIP-style models improves cross-modal alignment and performance in visual recognition and vision-language downstream tasks. Furthermore, we show that our method can learn more comprehensive representations, and the learned FDT capture meaningful cross-modal correspondence, ranging from objects to actions and attributes.
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
The source code can be found at https://github.com/yuxiaochen1103/FDT.

        ----

        ## [1438] Correlational Image Modeling for Self-Supervised Visual Pre-Training

        **Authors**: *Wei Li, Jiahao Xie, Chen Change Loy*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01450](https://doi.org/10.1109/CVPR52729.2023.01450)

        **Abstract**:

        We introduce Correlational Image Modeling (CIM), a novel and surprisingly effective approach to self-supervised visual pre-training. Our CIM performs a simple pretext task: we randomly crop image regions (exemplars) from an input image (context) and predict correlation maps between the exemplars and the context. Three key designs enable correlational image modeling as a nontrivial and meaningful self-supervisory task. First, to generate useful exemplar-context pairs, we consider cropping image regions with various scales, shapes, rotations, and transformations. Second, we employ a bootstrap learning framework that involves online and target encoders. During pre-training, the former takes exemplars as inputs while the latter converts the context. Third, we model the output correlation maps via a simple cross-attention block, within which the context serves as queries and the exemplars offer values and keys. We show that CIM performs on par or better than the current state of the art on self-supervised and transfer benchmarks. Code is available at https://github.com/weivision/Correlational-Image-Modeling.git.

        ----

        ## [1439] Generalized Decoding for Pixel, Image, and Language

        **Authors**: *Xueyan Zou, Zi-Yi Dou, Jianwei Yang, Zhe Gan, Linjie Li, Chunyuan Li, Xiyang Dai, Harkirat Behl, Jianfeng Wang, Lu Yuan, Nanyun Peng, Lijuan Wang, Yong Jae Lee, Jianfeng Gao*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01451](https://doi.org/10.1109/CVPR52729.2023.01451)

        **Abstract**:

        We present X-Decoder, a generalized decoding model that can predict pixel-level segmentation and language tokens seamlessly. X-Decoder takes as input two types of queries: (i) generic non-semantic queries and (ii) semantic queries induced from text inputs, to decode different pixel-level and token-level outputs in the same semantic space. With such a novel design, X-Decoder is the first work that provides a unified way to support all types of image segmentation and a variety of vision-language (VL) tasks. Without any pseudo-labeling, our design enables seamless interactions across tasks at different granularities and brings mutual benefits by learning a common and rich pixel-level understanding. After pretraining on a mixed set of a limited amount of segmentation data and millions of image-text pairs, X-Decoder exhibits strong transferability to a wide range of downstream tasks in both zero-shot and finetuning settings. Notably, it achieves (1) state-of-the-art results on open-vocabulary segmentation and referring segmentation on seven datasets; (2) better or competitive finetuned performance to other generalist and specialist models on segmentation and VL tasks; and (3) flexibility for efficient fine-tuning and novel task composition (e.g., referring captioning and image editing shown in Fig. 1). Code, demo, video and visualization are available at: https://x-decoder-vl.github.io.

        ----

        ## [1440] Towards Modality-Agnostic Person Re-identification with Descriptive Query

        **Authors**: *Cuiqun Chen, Mang Ye, Ding Jiang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01452](https://doi.org/10.1109/CVPR52729.2023.01452)

        **Abstract**:

        Person re-identification (ReID) with descriptive query (text or sketch) provides an important supplement for general image-image paradigms, which is usually studied in a single cross-modality matching manner, e.g., text-to-image or sketch-to-photo. However, without a camera-captured photo query, it is uncertain whether the text or sketch is available or not in practical scenarios. This motivates us to study a new and challenging modality-agnostic person re-ideruification problem. Towards this goal, we propose a unified person re-identification (UNIReID) architecture that can effectively adapt to cross-modality and multi-modality tasks. Specifically, UNIReID incorporates a simple dual-encoder with task-specific modality learning to mine and fuse visual and textual modality information. To deal with the imbalanced training problem of different tasks in UNIReID, we propose a task-aware dynamic training strategy in terms of task difficulty, adaptively adjusting the training focus. Besides, we construct three multi-modal ReID datasets by collecting the corresponding sketches from photos to support this challenging study. The experimental results on three multi-modal ReID datasets show that our UNIReID greatly improves the retrieval accuracy and generalization ability on different tasks and unseen scenarios.

        ----

        ## [1441] M6Doc: A Large-Scale Multi-Format, Multi-Type, Multi-Layout, Multi-Language, Multi-Annotation Category Dataset for Modern Document Layout Analysis

        **Authors**: *Hiuyi Cheng, Peirong Zhang, Sihang Wu, Jiaxin Zhang, Qiyuan Zhu, Zecheng Xie, Jing Li, Kai Ding, Lianwen Jin*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01453](https://doi.org/10.1109/CVPR52729.2023.01453)

        **Abstract**:

        Document layout analysis is a crucial prerequisite for document understanding, including document retrieval and conversion. Most public datasets currently contain only PDF documents and lack realistic documents. Models trained on these datasets may not generalize well to real-world scenarios. Therefore, this paper introduces a large and diverse document layout analysis dataset called M
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">6</sup>
Doc. The M6 designation represents six properties: (1) Multi-Format (including scanned, photographed, and PDF documents); (2) Multi-Type (such as scientific articles, textbooks, books, test papers, magazines, newspapers, and notes); (3) Multi-Layout (rectangular, Manhattan, non-Manhattan, and multi-column Manhattan); (4) Multi-Language (Chinese and English); (5) Multi-Annotation Category (74 types of annotation labels with 237,116 annotation instances in 9,080 manually annotated pages); and (6) Modern documents. Additionally, we propose a transformer-based document layout analysis method called TransDLANet, which leverages an adaptive element matching mechanism that enables query embedding to better match ground truth to improve recall, and constructs a segmentation branch for more precise document image instance segmentation. We conduct a comprehensive evaluation of M
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">6</sup>
Doc with various layout analysis methods and demonstrate its effectiveness. TransDLANet achieves state-of-the-art performance on M6 Doc with 64.5% mAP. The M
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">6</sup>
Doc dataset will be available at https://github.com/HcIILAB/M6Doc.

        ----

        ## [1442] Learning Customized Visual Models with Retrieval-Augmented Knowledge

        **Authors**: *Haotian Liu, Kilho Son, Jianwei Yang, Ce Liu, Jianfeng Gao, Yong Jae Lee, Chunyuan Li*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01454](https://doi.org/10.1109/CVPR52729.2023.01454)

        **Abstract**:

        Image-text contrastive learning models such as CLIP have demonstrated strong task transfer ability. The high generality and usability of these visual models is achieved via a web-scale data collection process to ensure broad concept coverage, followed by expensive pre-training to feed all the knowledge into model weights. Alternatively, we propose React,REtrieval-Augmented CusTomization, a framework to acquire the relevant web knowledge to build customized visual models for target domains. We retrieve the most relevant image-text pairs 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$(\thicksim3\%$</tex>
 of CLIP pre-training data) from the web-scale database as external knowledge and propose to customize the model by only training new modularized blocks while freezing all the original weights. The effectiveness of Reactis demonstrated via extensive experiments on classification, retrieval, detection and segmentation tasks, including zero, few, and full-shot settings. Particularly, on the zero-shot classification task, compared with CLIP, it achieves up to 5.4% improvement on ImageNet and 3.7% on the Elevaterbenchmark (20 datasets).

        ----

        ## [1443] Learning Semantic Relationship among Instances for Image-Text Matching

        **Authors**: *Zheren Fu, Zhendong Mao, Yan Song, Yongdong Zhang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01455](https://doi.org/10.1109/CVPR52729.2023.01455)

        **Abstract**:

        Image-text matching, a bridge connecting image and language, is an important task, which generally learns a holistic cross-modal embedding to achieve a high-quality semantic alignment between the two modalities. However, previous studies only focus on capturing fragment-level relation within a sample from a particular modality, e.g., salient regions in an image or text words in a sentence, where they usually pay less attention to capturing instance-level interactions among samples and modalities, e.g., multiple images and texts. In this paper, we argue that sample relations could help learn subtle differences for hard negative instances, and thus transfer shared knowledge for infrequent samples should be promising in obtaining better holistic embeddings. Therefore, we propose a novel hierarchical relation modeling framework (HREM), which explicitly capture both fragment-and instance-level relations to learn discriminative and robust cross-modal embeddings. Extensive experiments on Flickr30K and MS-COCO show our proposed method out-performs the state-of-the-art ones by 4%-10% in terms of rSum. Our code is available at https://github.com/CrossmodalGroup/HREM.

        ----

        ## [1444] I2MVFormer: Large Language Model Generated Multi-View Document Supervision for Zero-Shot Image Classification

        **Authors**: *Muhammad Ferjad Naeem, Muhammad Gul Zain Ali Khan, Yongqin Xian, Muhammad Zeshan Afzal, Didier Stricker, Luc Van Gool, Federico Tombari*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01456](https://doi.org/10.1109/CVPR52729.2023.01456)

        **Abstract**:

        Recent works have shown that unstructured text (doc-uments) from online sources can serve as useful auxiliary information for zero-shot image classification. However, these methods require access to a high-quality source like Wikipedia and are limited to a single source of information. Large Language Models (LLM) trained on web-scale text show impressive abilities to repurpose their learned knowledge for a multitude of tasks. In this work, we provide a novel perspective on using an LLM to provide text supervision for a zero-shot image classification model. The LLM is provided with a few text descriptions from different annota-tors as examples. The LLM is conditioned on these exam-ples to generate multiple text descriptions for each class (re-ferred to as views). Our proposed model, I2MVFormer, learns multi-view semantic embeddings for zero-shot image classification with these class views. We show that each text view of a class provides complementary information allowing a model to learn a highly discriminative class embed-ding. Moreover, we show that I2MVFormer is better at consuming the multi-view text supervision from LLM compared to baseline models. I2MVFormer establishes a new state-of-the-art on three public benchmark datasets for zero-shot image classification with unsupervised semantic embeddings. Code available at https://github.com/ferjad/I2DFormer

        ----

        ## [1445] ImageBind One Embedding Space to Bind Them All

        **Authors**: *Rohit Girdhar, Alaaeldin El-Nouby, Zhuang Liu, Mannat Singh, Kalyan Vasudev Alwala, Armand Joulin, Ishan Misra*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01457](https://doi.org/10.1109/CVPR52729.2023.01457)

        **Abstract**:

        We present ImageBind, an approach to learn a joint embedding across six different modalities - images, text, audio, depth, thermal, and IMU data. We show that all combinations of paired data are not necessary to train such a joint embedding, and only image-paired data is sufficient to bind the modalities together. ImageBind can leverage recent large scale vision-language models, and extends their zero-shot capabilities to new modalities just by using their natural pairing with images. It enables novel emergent applications ‘out-of-the-box’ including cross-modal retrieval, composing modalities with arithmetic, cross-modal detection and generation. The emergent capabilities improve with the strength of the image encoder and we set a new state-of-the-art on emergent zero-shot recognition tasks across modalities, outperforming specialist supervised models. Finally, we show strong few-shot recognition results outperforming prior work, and that ImageBind serves as a new way to evaluate vision models for visual and non-visual tasks.

        ----

        ## [1446] Model-Agnostic Gender Debiased Image Captioning

        **Authors**: *Yusuke Hirota, Yuta Nakashima, Noa Garcia*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01458](https://doi.org/10.1109/CVPR52729.2023.01458)

        **Abstract**:

        Image captioning models are known to perpetuate and amplify harmful societal bias in the training set. In this work, we aim to mitigate such gender bias in image captioning models. While prior work has addressed this problem by forcing models to focus on people to reduce gender mis-classification, it conversely generates gender-stereotypical words at the expense of predicting the correct gender. From this observation, we hypothesize that there are two types of gender bias affecting image captioning models: 1) bias that exploits context to predict gender, and 2) bias in the probability of generating certain (often stereotypical) words because of gender. To mitigate both types of gender biases, we propose a framework, called LIBRA, that learns from synthetically biased samples to decrease both types of biases, correcting gender misclassification and changing gender-stereotypical words to more neutral ones.

        ----

        ## [1447] Boundary-aware Backward-Compatible Representation via Adversarial Learning in Image Retrieval

        **Authors**: *Tan Pan, Furong Xu, Xudong Yang, Sifeng He, Chen Jiang, Qingpei Guo, Feng Qian, Xiaobo Zhang, Yuan Cheng, Lei Yang, Wei Chu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01459](https://doi.org/10.1109/CVPR52729.2023.01459)

        **Abstract**:

        Image retrieval plays an important role in the Internet world. Usually, the core parts of mainstream visual retrieval systems include an online service of the embedding model and a large-scale vector database. For traditional model upgrades, the old model will not be replaced by the new one until the embeddings of all the images in the database are re-computed by the new model, which takes days or weeks for a large amount of data. Recently, backward-compatible training (BCT) enables the new model to be immediately deployed online by making the new embeddings directly comparable to the old ones. For BCT, improving the compatibility of two models with less negative impact on retrieval performance is the key challenge. In this paper, we introduce AdvBCT, an Adversarial Backward-Compatible Training method with an elastic boundary constraint that takes both compatibility and discrimination into consideration. We first employ adversarial learning to minimize the distribution disparity between embeddings of the new model and the old model. Meanwhile, we add an elastic boundary constraint during training to improve compatibility and discrimination efficiently. Extensive experiments on GLDv2, Revisited Oxford (ROxford), and Revisited Paris (RParis) demonstrate that our method outperforms other BCT methods on both compatibility and discrimination. The implementation of AdvBCT will be publicly available at https://github.com/Ashespt/AdvBCT.

        ----

        ## [1448] Prompt, Generate, Then Cache: Cascade of Foundation Models Makes Strong Few-Shot Learners

        **Authors**: *Renrui Zhang, Xiangfei Hu, Bohao Li, Siyuan Huang, Hanqiu Deng, Yu Qiao, Peng Gao, Hongsheng Li*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01460](https://doi.org/10.1109/CVPR52729.2023.01460)

        **Abstract**:

        Visual recognition in low-data regimes requires deep neural networks to learn generalized representations from limited training samples. Recently, CLIP-based methods have shown promising few-shot performance benefited from the contrastive language-image pre-training. We then question, if the more diverse pre-training knowledge can be cascaded to further assist few-shot representation learning. In this paper, we propose CaFo, a Cascade of Foundation models that incorporates diverse prior knowledge of various pretraining paradigms for better few-shot learning. Our CaFo incorporates CLIP's language-contrastive knowledge, DINO's vision-contrastive knowledge, DALL-E's vision-generative knowledge, and GPT-3's language-generative knowledge. Specifically, CaFo works by ‘Prompt, Generate, then Cache’. Firstly, we leverage GPT-3 to produce textual inputs for prompting CLIP with rich downstream linguistic semantics. Then, we generate synthetic images via DALL-E to expand the few-shot training data without any manpower. At last, we introduce a learnable cache model to adaptively blend the predictions from CLIP and DINO. By such collaboration, CaFo can fully unleash the potential of different pre-training methods and unify them to perform state-of-the-art for few-shot classification. Code is available at https://github.com/ZrrSkywalker/CaFo.

        ----

        ## [1449] Towards Unified Scene Text Spotting Based on Sequence Generation

        **Authors**: *Taeho Kil, Seonghyeon Kim, Sukmin Seo, Yoonsik Kim, Daehee Kim*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01461](https://doi.org/10.1109/CVPR52729.2023.01461)

        **Abstract**:

        Sequence generation models have recently made significant progress in unifying various vision tasks. Although some auto-regressive models have demonstrated promising results in end-to-end text spotting, they use specific detection formats while ignoring various text shapes and are limited in the maximum number of text instances that can be detected. To overcome these limitations, we propose a UNIfied scene Text Spotter, called UNITS. Our model unifies various detection formats, including quadrilaterals and polygons, allowing it to detect text in arbitrary shapes. Additionally, we apply starting-point prompting to enable the model to extract texts from an arbitrary starting point, thereby extracting more texts beyond the number of instances it was trained on. Experimental results demonstrate that our method achieves competitive performance compared to state-of-the-art methods. Further analysis shows that UNITS can extract a larger number of texts than it was trained on. We provide the code for our method at https://github.com/clovaai/units.

        ----

        ## [1450] CapDet: Unifying Dense Captioning and Open-World Detection Pretraining

        **Authors**: *Yanxin Long, Youpeng Wen, Jianhua Han, Hang Xu, Pengzhen Ren, Wei Zhang, Shen Zhao, Xiaodan Liang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01462](https://doi.org/10.1109/CVPR52729.2023.01462)

        **Abstract**:

        Benefiting from large-scale vision-language pre-training on image-text pairs, open-world detection methods have shown superior generalization ability under the zero-shot or few-shot detection settings. However, a pre-defined category space is still required during the inference stage of existing methods and only the objects belonging to that space will be predicted. To introduce a “real” open-world detector, in this paper, we propose a novel method named CapDet to either predict under a given category list or directly generate the category of predicted bounding boxes. Specifically, we unify the open-world detection and dense caption tasks into a single yet effective framework by introducing an additional dense captioning head to generate the region-grounded captions. Besides, adding the captioning task will in turn benefit the generalization of detection performance since the captioning dataset covers more concepts. Experiment results show that by unifying the dense caption task, our CapDet has obtained significant performance improvements (e.g., +2.1% mAP on LVIS rare classes) over the baseline method on LVIS (1203 classes). Besides, our CapDet also achieves state-of-the-art performance on dense captioning tasks, e.g., 15.44% mAP on VG V1.2 and 13.98% on the VG-COCO dataset.

        ----

        ## [1451] CLIP2: Contrastive Language-Image-Point Pretraining from Real-World Point Cloud Data

        **Authors**: *Yihan Zeng, Chenhan Jiang, Jiageng Mao, Jianhua Han, Chaoqiang Ye, Qingqiu Huang, Dit-Yan Yeung, Zhen Yang, Xiaodan Liang, Hang Xu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01463](https://doi.org/10.1109/CVPR52729.2023.01463)

        **Abstract**:

        Contrastive Language-Image Pre-training, benefiting from large-scale unlabeled text-image pairs, has demonstrated great performance in open-world vision understanding tasks. However, due to the limited Text-3D data pairs, adapting the success of 2D Vision-Language Models (VLM) to the 3D space remains an open problem. Existing works that leverage VLM for 3D understanding generally resort to constructing intermediate 2D representations for the 3D data, but at the cost of losing 3D geometry information. To take a step toward open-world 3D vision understanding, we propose Contrastive Language-Image-Point Cloud Pretraining (CLIP
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">2</sup>
) to directly learn the transferable 3D point cloud representation in realistic scenarios with a novel proxy alignment mechanism. Specifically, we exploit naturally-existed correspondences in 2D and 3D scenarios, and build well-aligned and instance-based text-image-point proxies from those complex scenarios. On top of that, we propose a cross-modal contrastive objective to learn semantic and instance-level aligned point cloud representation. Experimental results on both indoor and outdoor scenarios show that our learned 3D representation has great transfer ability in downstream tasks, including zero-shot and few-shot 3D recognition, which boosts the state-of-the-art methods by large margins. Furthermore, we provide analyses of the capability of different representations in real scenarios and present the optional ensemble scheme.

        ----

        ## [1452] Aligning Bag of Regions for Open-Vocabulary Object Detection

        **Authors**: *Size Wu, Wenwei Zhang, Sheng Jin, Wentao Liu, Chen Change Loy*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01464](https://doi.org/10.1109/CVPR52729.2023.01464)

        **Abstract**:

        Pre-trained vision-language models (VLMs) learn to align vision and language representations on large-scale datasets, where each image-text pair usually contains a bag of semantic concepts. However, existing open-vocabulary object detectors only align region embeddings individually with the corresponding features extracted from the VLMs. Such a design leaves the compositional structure of semantic concepts in a scene under-exploited, although the structure may be implicitly learned by the VLMs. In this work, we propose to align the embedding of bag of regions beyond individual regions. The proposed method groups contextually interrelated regions as a bag. The embeddings of regions in a bag are treated as embeddings of words in a sentence, and they are sent to the text encoder of a VLM to obtain the bag-of-regions embedding, which is learned to be aligned to the corresponding features extracted by a frozen VLM. Applied to the commonly used Faster R-CNN, our approach surpasses the previous best results by 4.6 box AP
<inf xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">50</inf>
 and 2.8 mask AP on novel categories of open-vocabulary COCO and LVIS benchmarks, respectively. Code and models are available at https://github.com/wusize/ovdet.

        ----

        ## [1453] Visual Recognition by Request

        **Authors**: *Chufeng Tang, Lingxi Xie, Xiaopeng Zhang, Xiaolin Hu, Qi Tian*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01465](https://doi.org/10.1109/CVPR52729.2023.01465)

        **Abstract**:

        Humans have the ability of recognizing visual semantics in an unlimited granularity, but existing visual recognition algorithms cannot achieve this goal. In this paper, we establish a new paradigm named visual recognition by request (ViRReq
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
We recommend the readers to pronounce ViRReqas/virik/.) to bridge the gap. The key lies in decomposing visual recognition into atomic tasks named requests and leveraging a knowledge base, a hierarchical and text-based dictionary, to assist task definition. ViRReq allows for (i) learning complicated whole-part hierarchies from highly incomplete annotations and (ii) inserting new concepts with minimal efforts. We also establish a solid baseline by integrating language-driven recognition into recent semantic and instance segmentation methods, and demonstrate its flexible recognition ability on CPP and ADE20K, two datasets with hierarchical whole-part annotations.

        ----

        ## [1454] Category Query Learning for Human-Object Interaction Classification

        **Authors**: *Chi Xie, Fangao Zeng, Yue Hu, Shuang Liang, Yichen Wei*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01466](https://doi.org/10.1109/CVPR52729.2023.01466)

        **Abstract**:

        Unlike most previous HOI methods that focus on learning better human-object features, we propose a novel and complementary approach called category query learning. Such queries are explicitly associated to interaction categories, converted to image specific category representation via a transformer decoder, and learnt via an auxiliary image-level classification task. This idea is motivated by an earlier multi-label image classification method, but is for the first time applied for the challenging human-object interaction classification task. Our method is simple, general and effective. It is validated on three representative HOI baselines and achieves new state-of-the-art results on two benchmarks. Code will be available at https://github.com/charles-xie/CQL.

        ----

        ## [1455] Self-Supervised Implicit Glyph Attention for Text Recognition

        **Authors**: *Tongkun Guan, Chaochen Gu, Jingzheng Tu, Xue Yang, Qi Feng, Yudi Zhao, Wei Shen*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01467](https://doi.org/10.1109/CVPR52729.2023.01467)

        **Abstract**:

        The attention mechanism has become the de facto module in scene text recognition (STR) methods, due to its capability of extracting character-level representations. These methods can be summarized into implicit attention based and supervised attention based, depended on how the attention is computed, i.e., implicit attention and supervised attention are learned from sequence-level text annotations and or character-level bounding box annotations, respectively. Implicit attention, as it may extract coarse or even incorrect spatial regions as character attention, is prone to suffering from an alignment-drifted issue. Supervised attention can alleviate the above issue, but it is character category-specific, which requires extra laborious character-level bounding box annotations and would be memory-intensive when handling languages with larger character categories. To address the aforementioned issues, we propose a novel attention mechanism for STR, self-supervised implicit glyph attention (SICA). SICA delineates the glyph structures of text images by jointly self-supervised text seg-mentation and implicit attention alignment, which serve as the supervision to improve attention correctness without extra character-level annotations. Experimental results demonstrate that SIGA performs consistently and significantly better than previous attention-based STR methods, in terms of both attention correctness and final recognition performance on publicly available context benchmarks and our contributed contextless benchmarks.

        ----

        ## [1456] Enlarging Instance-specific and Class-specific Information for Open-set Action Recognition

        **Authors**: *Jun Cen, Shiwei Zhang, Xiang Wang, Yixuan Pei, Zhiwu Qing, Yingya Zhang, Qifeng Chen*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01468](https://doi.org/10.1109/CVPR52729.2023.01468)

        **Abstract**:

        Open-set action recognition is to reject unknown human action cases which are out of the distribution of the training set. Existing methods mainly focus on learning better uncertainty scores but dismiss the importance of feature representations. We find that features with richer semantic diversity can significantly improve the open-set performance under the same uncertainty scores. In this paper, we begin with analyzing the feature representation behavior in the open-set action recognition (OSAR) problem based on the information bottleneck (IB) theory, and propose to enlarge the instance-specific (IS) and classspecific (CS) information contained in the feature for better performance. To this end, a novel Prototypical Similarity Learning (PSL) framework is proposed to keep the instance variance within the same class to retain more IS information. Besides, we notice that unknown samples sharing similar appearances to known samples are easily misclassified as known classes. To alleviate this issue, video shuffling is further introduced in our PSL to learn distinct temporal information between original and shuffled samples, which we find enlarges the CS information. Extensive experiments demonstrate that the proposed PSL can significantly boost both the open-set and closed-set performance and achieves state-of-the-art results on multiple benchmarks. Code is available at https://github.com/Jun-CEN/PSL.

        ----

        ## [1457] CLIP is Also an Efficient Segmenter: A Text-Driven Approach for Weakly Supervised Semantic Segmentation

        **Authors**: *Yuqi Lin, Minghao Chen, Wenxiao Wang, Boxi Wu, Ke Li, Binbin Lin, Haifeng Liu, Xiaofei He*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01469](https://doi.org/10.1109/CVPR52729.2023.01469)

        **Abstract**:

        Weakly supervised semantic segmentation (WSSS) with image-level labels is a challenging task. Mainstream approaches follow a multi-stage framework and suffer from high training costs. In this paper, we explore the potential of Contrastive Language-Image Pre-training models (CLIP) to localize different categories with only image-level labels and without further training. To efficiently generate high-quality segmentation masks from CLIP, we propose a novel WSSS framework called CLIP-ES. Our framework improves all three stages of WSSS with special designs for CLIP: 1) We introduce the softmax function into GradCAM and exploit the zero-shot ability of CLIP to suppress the confusion caused by non-target classes and backgrounds. Mean-while, to take full advantage of CLIP, we re-explore text inputs under the WSSS setting and customize two text-driven strategies: sharpness-based prompt selection and synonym fusion. 2) To simplify the stage of CAM refinement, we propose a real-time class-aware attention-based affinity (CAA) module based on the inherent multi-head self-attention (MHSA) in CLIP- ViTs. 3) When training the final segmentation model with the masks generated by CLIP, we introduced a confidence-guided loss (CGL) focus on confident regions. Our CLIP-ES achieves SOTA performance on Pascal VOC 2012 and MS COCO 2014 while only taking 10% time of previous methods for the pseudo mask generation. Code is available at https://github.com/linyq2117/CLIP-ES.

        ----

        ## [1458] Learning Attention as Disentangler for Compositional Zero-Shot Learning

        **Authors**: *Shaozhe Hao, Kai Han, Kwan-Yee K. Wong*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01470](https://doi.org/10.1109/CVPR52729.2023.01470)

        **Abstract**:

        Compositional zero-shot learning (CZSL) aims at learning visual concepts (i.e., attributes and objects) from seen compositions and combining concept knowledge into unseen compositions. The key to CZSL is learning the disentanglement of the attribute-object composition. To this end, we propose to exploit cross-attentions as compositional disentanglers to learn disentangled concept embeddings. For example, if we want to recognize an unseen composition “yellow flower”, we can learn the attribute concept “yellow” and object concept “flower” from different yellow objects and different flowers respectively. To further constrain the disentanglers to learn the concept of interest, we employ a regularization at the attention level. Specifically, we adapt the earth mover's distance (EMD) as a feature similarity metric in the cross-attention module. Moreover, benefiting from concept disentanglement, we improve the inference process and tune the prediction score by combining multiple concept probabilities. Comprehensive experiments on three CZSL benchmark datasets demonstrate that our method significantly outperforms previous works in both closed- and open-world settings, establishing a new state-of-the-art. Project page: https://haoosz.github.io/ade-czsl/

        ----

        ## [1459] Universal Instance Perception as Object Discovery and Retrieval

        **Authors**: *Bin Yan, Yi Jiang, Jiannan Wu, Dong Wang, Ping Luo, Zehuan Yuan, Huchuan Lu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01471](https://doi.org/10.1109/CVPR52729.2023.01471)

        **Abstract**:

        All instance perception tasks aim at finding certain objects specified by some queries such as category names, language expressions, and target annotations, but this complete field has been split into multiple independent sub-tasks. In this work, we present a universal instance perception model of the next generation, termed UNINEXT. UNINEXT reformulates diverse instance perception tasks into a unified object discovery and retrieval paradigm and can flexibly perceive different types of objects by simply changing the input prompts. This unified formulation brings the following benefits: (1) enormous data from different tasks and label vocabularies can be exploited for jointly training general instance-level representations, which is especially beneficial for tasks lacking in training data. (2) the unified model is parameter-efficient and can save redundant computation when handling multiple tasks simultaneously. UNINEXT shows superior performance on 20 challenging benchmarks from 10 instance-level tasks including classical image-level tasks (object detection and instance segmentation), vision-and-language tasks (referring expression comprehension and segmentation), and six video-level object tracking tasks. Code is available at https://github.com/MasterBin-IIAU/UNINEXT.

        ----

        ## [1460] Progressive Semantic-Visual Mutual Adaption for Generalized Zero-Shot Learning

        **Authors**: *Man Liu, Feng Li, Chunjie Zhang, Yunchao Wei, Huihui Bai, Yao Zhao*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01472](https://doi.org/10.1109/CVPR52729.2023.01472)

        **Abstract**:

        Generalized Zero-Shot Learning (GZSL) identifies unseen categories by knowledge transferred from the seen domain, relying on the intrinsic interactions between visual and semantic information. Prior works mainly localize regions corresponding to the sharing attributes. When various visual appearances correspond to the same attribute, the sharing attributes inevitably introduce semantic ambiguity, hampering the exploration of accurate semantic-visual interactions. In this paper, we deploy the dual semantic-visual transformer module (DSVTM) to progressively model the correspondences between attribute prototypes and visual features, constituting a progressive semantic-visual mutual adaption (PSVMA) network for semantic disambiguation and knowledge transferability improvement. Specifically, DSVTM devises an instance-motivated semantic encoder that learns instance-centric prototypes to adapt to different images, enabling the recast of the unmatched semantic-visual pair into the matched one. Then, a semantic-motivated instance decoder strengthens accurate cross-domain interactions between the matched pair for semantic-related instance adaption, en-couraging the generation of unambiguous visual representations. Moreover, to mitigate the bias towards seen classes in GZSL, a debiasing loss is proposed to pursue response consistency between seen and unseen predictions. The PSVMA consistently yields superior performances against other state-of-the-art methods. Code will be available at: https://github.com/ManLiuCoder/PSVMA.

        ----

        ## [1461] DPF: Learning Dense Prediction Fields with Weak Supervision

        **Authors**: *Xiaoxue Chen, Yuhang Zheng, Yupeng Zheng, Qiang Zhou, Hao Zhao, Guyue Zhou, Ya-Qin Zhang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01473](https://doi.org/10.1109/CVPR52729.2023.01473)

        **Abstract**:

        Nowadays, many visual scene understanding problems are addressed by dense prediction networks. But pixel-wise dense annotations are very expensive (e.g., for scene parsing) or impossible (e.g., for intrinsic image decomposition), motivating us to leverage cheap point-level weak supervision. However, existing pointly-supervised methods still use the same architecture designed for full supervision. In stark contrast to them, we propose a new paradigm that makes predictions for point coordinate queries, as inspired by the recent success of implicit representations, like distance or radiance fields. As such, the method is named as dense prediction fields (DPFs). DPFs generate expressive intermediate features for continuous sub-pixel locations, thus allowing outputs of an arbitrary resolution. DPFs are naturally compatible with point-level supervision. We showcase the effectiveness of DPFs using two substantially different tasks: high-level semantic parsing and low-level intrinsic image decomposition. In these two cases, supervision comes in the form of single-point semantic category and two-point relative reflectance, respectively. As benchmarked by three large-scale public datasets PASCALContext, ADE20K and IIW, DPFs set new state-of-the-art performance on all of them with significant margins. Code can be accessed at https://github.com/cxx226/DPF.

        ----

        ## [1462] Modeling Entities as Semantic Points for Visual Information Extraction in the Wild

        **Authors**: *Zhibo Yang, Rujiao Long, Pengfei Wang, Sibo Song, Humen Zhong, Wenqing Cheng, Xiang Bai, Cong Yao*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01474](https://doi.org/10.1109/CVPR52729.2023.01474)

        **Abstract**:

        Recently, Visual Information Extraction (VIE) has been becoming increasingly important in both the academia and industry, due to the wide range of real-world applications. Previously, numerous works have been proposed to tackle this problem. However, the benchmarks used to assess these methods are relatively plain, i.e., scenarios with real-world complexity are not fully represented in these benchmarks. As the first contribution of this work, we curate and release a new dataset for VIE, in which the document images are much more challenging in that they are taken from real applications, and difficulties such as blur, partial occlusion, and printing shift are quite common. All these factors may lead to failures in information extraction. Therefore, as the second contribution, we explore an alternative approach to precisely and robustly extract key information from document images under such tough conditions. Specifically, in contrast to previous methods, which usually either incorporate visual information into a multi-modal architecture or train text spotting and information extraction in an end-to-end fashion, we explicitly model entities as semantic points, i.e., center points of entities are enriched with semantic information describing the attributes and relationships of different entities, which could largely benefit entity labeling and linking. Extensive experiments on standard benchmarks in this field as well as the proposed dataset demonstrate that the proposed method can achieve significantly enhanced performance on entity labeling and linking, compared with previous state-of-the-art models. Dataset is available at https://www.modelscope.cn/datasets/damo/SIBR/summary.

        ----

        ## [1463] GeoNet: Benchmarking Unsupervised Adaptation across Geographies

        **Authors**: *Tarun Kalluri, Wangdong Xu, Manmohan Chandraker*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01475](https://doi.org/10.1109/CVPR52729.2023.01475)

        **Abstract**:

        In recent years, several efforts have been aimed at improving the robustness of vision models to domains and environments unseen during training. An important practical problem pertains to models deployed in a new geography that is under-represented in the training dataset, posing a direct challenge to fair and inclusive computer vision. In this paper, we study the problem of geographic robustness and make three main contributions. First, we introduce a large-scale dataset GeoNet for geographic adaptation containing benchmarks across diverse tasks like scene recognition (GeoPlaces), image classification (GeoImNet) and universal adaptation (GeoUniDA). Second, we investigate the nature of distribution shifts typical to the problem of geographic adaptation and hypothesize that the major source of domain shifts arise from significant variations in scene context (context shift), object design (design shift) and label distribution (prior shift) across geographies. Third, we conduct an extensive evaluation of several state-of-the-art unsupervised domain adaptation algorithms and architectures on GeoNet, showing that they do not suffice for geographical adaptation, and that large-scale pre-training using large vision models also does not lead to geographic robustness. Our dataset is publicly available at https://tarun005.github.io/GeoNet.

        ----

        ## [1464] SegLoc: Learning Segmentation-Based Representations for Privacy-Preserving Visual Localization

        **Authors**: *Maxime Pietrantoni, Martin Humenberger, Torsten Sattler, Gabriela Csurka*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01476](https://doi.org/10.1109/CVPR52729.2023.01476)

        **Abstract**:

        Inspired by properties of semantic segmentation, in this paper we investigate how to leverage robust image segmentation in the context of privacy-preserving visual localization. We propose a new localization framework, SegLoc, that leverages image segmentation to create robust, compact, and privacy-preserving scene representations, i.e., 3D maps. We build upon the correspondence-supervised, fine-grained segmentation approach from [42], making it more robust by learning a set of cluster labels with discriminative clustering, additional consistency regularization terms and we jointly learn a global image representation along with a dense local representation. In our localization pipeline, the former will be used for retrieving the most similar images, the latter to refine the retrieved poses by minimizing the label inconsistency between the 3D points of the map and their projection onto the query image. In various experiments, we show that our proposed representation allows to achieve (close-to) state-of-the-art pose estimation results while only using a compact 3D map that does not contain enough information about the original images for an attacker to reconstruct personal information.

        ----

        ## [1465] Towards Open-World Segmentation of Parts

        **Authors**: *Tai-Yu Pan, Qing Liu, Wei-Lun Chao, Brian Price*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01477](https://doi.org/10.1109/CVPR52729.2023.01477)

        **Abstract**:

        Segmenting object parts such as cup handles and animal bodies is important in many real-world applications but requires more annotation effort. The largest dataset nowadays contains merely two hundred object categories, implying the difficulty to scale up part segmentation to an unconstrained setting. To address this, we propose to explore a seemingly simplified but empirically useful and scalable task, class-agnostic part segmentation. In this problem, we disregard the part class labels in training and instead treat all of them as a single part class. We argue and demonstrate that models trained without part classes can better localize parts and segment them on objects unseen in training. We then present two further improvements. First, we propose to make the model object-aware, leveraging the fact that parts are “compositions”, whose extents are bounded by the corresponding objects and whose appearances are by nature not independent but bundled. Second, we introduce a novel approach to improve part segmentation on unseen objects, inspired by an interesting finding - for unseen objects, the pixel-wise features extracted by the model often reveal high-quality part segments. To this end, we propose a novel self-supervised procedure that iterates between pixel clustering and supervised contrastive learning that pulls pixels closer or pushes them away. Via extensive experiments on PartImageNet and Pascal-Part, we show notable and consistent gains by our approach, essentially a critical step towards open-world part segmentation.

        ----

        ## [1466] Pruning Parameterization with Bi-level Optimization for Efficient Semantic Segmentation on the Edge

        **Authors**: *Changdi Yang, Pu Zhao, Yanyu Li, Wei Niu, Jiexiong Guan, Hao Tang, Minghai Qin, Bin Ren, Xue Lin, Yanzhi Wang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01478](https://doi.org/10.1109/CVPR52729.2023.01478)

        **Abstract**:

        With the ever-increasing popularity of edge devices, it is necessary to implement real-time segmentation on the edge for autonomous driving and many other applications. Vision Transformers (ViTs) have shown considerably stronger results for many vision tasks. However, ViTs with the fullattention mechanism usually consume a large number of computational resources, leading to difficulties for real- time inference on edge devices. In this paper, we aim to derive ViTs with fewer computations and fast inference speed to facilitate the dense prediction of semantic segmentation on edge devices. To achieve this, we propose a pruning parameterization method to formulate the pruning problem of semantic segmentation. Then we adopt a bi-level optimization method to solve this problem with the help of implicit gradients. Our experimental results demonstrate that we can achieve 38.9 mIoU on ADE20K val with a speed of 56.5 FPS on Samsung S21, which is the highest mIoU under the same computation constraint with real-time inference.

        ----

        ## [1467] HGFormer: Hierarchical Grouping Transformer for Domain Generalized Semantic Segmentation

        **Authors**: *Jian Ding, Nan Xue, Gui-Song Xia, Bernt Schiele, Dengxin Dai*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01479](https://doi.org/10.1109/CVPR52729.2023.01479)

        **Abstract**:

        Current semantic segmentation models have achieved great success under the independent and identically distributed (i.i.d.) condition. However, in real-world applications, test data might come from a different domain than training data. Therefore, it is important to improve model robustness against domain differences. This work studies semantic segmentation under the domain generalization setting, where a model is trained only on the source domain and tested on the unseen target domain. Existing works show that Vision Transformers are more robust than CNNs and show that this is related to the visual grouping property of self-attention. In this work, we propose a novel hierarchical grouping transformer (HGFormer) to explicitly group pixels to form part-level masks and then whole-level masks. The masks at different scales aim to segment out both parts and a whole of classes. HGFormer combines mask classification results at both scales for class label prediction. We assemble multiple interesting cross-domain settings by using seven public semantic segmentation datasets. Experiments show that HGFormer yields more robust semantic segmentation results than per-pixel classification methods and flat-grouping transformers, and outperforms previous methods significantly. Code will be available at https://github.com/dingjianswl0l/HGFormer.

        ----

        ## [1468] Exemplar-FreeSOLO: Enhancing Unsupervised Instance Segmentation with Exemplars

        **Authors**: *Taoseef Ishtiak, Qing En, Yuhong Guo*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01480](https://doi.org/10.1109/CVPR52729.2023.01480)

        **Abstract**:

        Instance segmentation seeks to identify and segment each object from images, which often relies on a large number of dense annotations for model training. To alleviate this burden, unsupervised instance segmentation methods have been developed to train class-agnostic instance segmentation models without any annotation. In this paper, we propose a novel unsupervised instance segmentation approach, Exemplar-FreeSOLO, to enhance unsupervised instance segmentation by exploiting a limited number of unannotated and unsegmented exemplars. The proposed framework offers a new perspective on directly perceiving top-down information without annotations. Specifically, Exemplar-FreeSOLO introduces a novel exemplar-knowledge abstraction module to acquire beneficial top-down guidance knowledge for instances using unsupervised exemplar object extraction. Moreover, a new exemplar embedding contrastive module is designed to enhance the discriminative capability of the segmentation model by exploiting the contrastive exemplar-based guidance knowledge in the embedding space. To evaluate the proposed Exemplar-FreeSOLO, we conduct comprehensive experiments and perform in-depth analyses on three image instance segmentation datasets. The experimental results demonstrate that the proposed approach is effective and outperforms the state-of-the-art methods.

        ----

        ## [1469] Weakly-Supervised Domain Adaptive Semantic Segmentation with Prototypical Contrastive Learning

        **Authors**: *Anurag Das, Yongqin Xian, Dengxin Dai, Bernt Schiele*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01481](https://doi.org/10.1109/CVPR52729.2023.01481)

        **Abstract**:

        There has been a lot of effort in improving the performance of unsupervised domain adaptation for semantic segmentation task, however, there is still a huge gap in performance when compared with supervised learning. In this work, we propose a common framework to use different weak labels, e.g., image, point and coarse labels from the target domain to reduce this performance gap. Specifically, we propose to learn better prototypes that are representative class features by exploiting these weak labels. We use these improved prototypes for the contrastive alignment of class features. In particular, we perform two different feature alignments: first, we align pixel features with proto-types within each domain and second, we align pixel features from the source to prototype of target domain in an asymmetric way. This asymmetric alignment is beneficial as it preserves the target features during training, which is essential when weak labels are available from the target domain. Our experiments on various benchmarks show that our framework achieves significant improvement compared to existing works and can reduce the performance gap with supervised learning. Code will be available at https://github.com/anurag-198/WDASS.

        ----

        ## [1470] Spatial-temporal Concept based Explanation of 3D ConvNets

        **Authors**: *Ying Ji, Yu Wang, Jien Kato*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01482](https://doi.org/10.1109/CVPR52729.2023.01482)

        **Abstract**:

        Convolutional neural networks (CNNs) have shown remarkable performance on various tasks. Despite its widespread adoption, the decision procedure of the network still lacks transparency and interpretability, making it difficult to enhance the performance further. Hence, there has been considerable interest in providing explanation and interpretability for CNNs over the last few years. Explainable artificial intelligence (XAI) investigates the relationship between input images or videos and output predictions. Recent studies have achieved outstanding success in explaining 2D image classification ConvNets. On the other hand, due to the high computation cost and complexity of video data, the explanation of 3D video recognition ConvNets is relatively less studied. And none of them are able to produce a high-level explanation. In this paper, we propose a STCE (Spatial-temporal Concept-based Explanation) framework for interpreting 3D ConvNets. In our approach: (1) videos are represented with high-level supervoxels, similar supervoxels are clustered as a concept, which is straightforward for human to understand; and (2) the interpreting framework calculates a score for each concept, which reflects its significance in the ConvNet decision procedure. Experiments on diverse 3D ConvNets demonstrate that our method can identify global concepts with different importance levels, allowing us to investigate the impact of the concepts on a target task, such as action recognition, in-depth. The source codes are publicly available at https://github.com/yingji425/STCE.

        ----

        ## [1471] Sparsely Annotated Semantic Segmentation with Adaptive Gaussian Mixtures

        **Authors**: *Linshan Wu, Zhun Zhong, Leyuan Fang, Xingxin He, Qiang Liu, Jiayi Ma, Hao Chen*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01483](https://doi.org/10.1109/CVPR52729.2023.01483)

        **Abstract**:

        Sparsely annotated semantic segmentation (SASS) aims to learn a segmentation model by images with sparse labels (i.e., points or scribbles). Existing methods mainly focus on introducing low-level affinity or generating pseudo labels to strengthen supervision, while largely ignoring the inherent relation between labeled and unlabeled pixels. In this paper, we observe that pixels that are close to each other in the feature space are more likely to share the same class. Inspired by this, we propose a novel SASS framework, which is equipped with an Adaptive Gaussian Mixture Model (AGMM). Our AGMM can effectively endow reliable supervision for unlabeled pixels based on the distributions of labeled and unlabeled pixels. Specifically, we first build Gaussian mixtures using labeled pixels and their relatively similar unlabeled pixels, where the labeled pixels act as centroids, for modeling the feature distribution of each class. Then, we leverage the reliable information from labeled pixels and adaptively generated GMM predictions to supervise the training of unlabeled pixels, achieving online, dynamic, and robust selfsupervision. In addition, by capturing category-wise Gaussian mixtures, AGMM encourages the model to learn discriminative class decision boundaries in an end-to-end contrastive learning manner. Experimental results conducted on the PASCAL VOC 2012 and Cityscapes datasets demonstrate that our AGMM can establish new state-of-the-art SASS performance. Code is available at https://github.com/Luffy03/AGMM-SASS

        ----

        ## [1472] Fuzzy Positive Learning for Semi-Supervised Semantic Segmentation

        **Authors**: *Pengchong Qiao, Zhidan Wei, Yu Wang, Zhennan Wang, Guoli Song, Fan Xu, Xiangyang Ji, Chang Liu, Jie Chen*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01484](https://doi.org/10.1109/CVPR52729.2023.01484)

        **Abstract**:

        Semi-supervised learning (SSL) essentially pursues class boundary exploration with less dependence on human annotations. Although typical attempts focus on ameliorating the inevitable error-prone pseudo-labeling, we think differently and resort to exhausting informative semantics from multiple probably correct candidate labels. In this paper, we introduce Fuzzy Positive Learning (FPL) for accurate SSL semantic segmentation in a plug-and-play fashion, targeting adaptively encouraging fuzzy positive predictions and suppressing highly-probable negatives. Being conceptually simple yet practically effective, FPL can remarkably alleviate interference from wrong pseudo labels and progressively achieve clear pixel-level semantic discrimination. Concretely, our FPL approach consists of two main components, including fuzzy positive assignment (FPA) to provide an adaptive number of labels for each pixel and fuzzy positive regularization (FPR) to restrict the predictions of fuzzy positive categories to be larger than the rest under different perturbations. Theoretical analysis and extensive experiments on Cityscapes and VOC 2012 with consistent performance gain justify the superiority of our approach. Codes are provided in https://github.com/qpc1611094/FPL.

        ----

        ## [1473] STAR Loss: Reducing Semantic Ambiguity in Facial Landmark Detection

        **Authors**: *Zhenglin Zhou, Huaxia Li, Hong Liu, Nanyang Wang, Gang Yu, Rongrong Ji*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01485](https://doi.org/10.1109/CVPR52729.2023.01485)

        **Abstract**:

        Recently, deep learning-based facial landmark detection has achieved significant improvement. However, the semantic ambiguity problem degrades detection performance. Specifically, the semantic ambiguity causes inconsistent annotation and negatively affects the model's convergence, leading to worse accuracy and instability prediction. To solve this problem, we propose a Self-adapTive Ambiguity Reduction (STAR) loss by exploiting the properties of se-mantic ambiguity. We find that semantic ambiguity results in the anisotropic predicted distribution, which inspires us to use predicted distribution to represent semantic ambiguity. Based on this, we design the STAR loss that measures the anisotropism of the predicted distribution. Compared with the standard regression loss, STAR loss is encouraged to be small when the predicted distribution is anisotropic and thus adaptively mitigates the impact of semantic ambiguity. Moreover, we propose two kinds of eigen-value restriction methods that could avoid both distribution's abnormal change and the model's premature convergence. Finally, the comprehensive experiments demonstrate that STAR loss outperforms the state-of-the-art methods on three benchmarks, i.e., COFW, 300W, and WFLW, with negligible computation overhead. Code is at https://github.com/ZhenglinZhou/STAR

        ----

        ## [1474] Boosting Low-Data Instance Segmentation by Unsupervised Pre-training with Saliency Prompt

        **Authors**: *Hao Li, Dingwen Zhang, Nian Liu, Lechao Cheng, Yalun Dai, Chao Zhang, Xinggang Wang, Junwei Han*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01486](https://doi.org/10.1109/CVPR52729.2023.01486)

        **Abstract**:

        Inspired by DETR variants, query-based end-to-end instance segmentation (QEIS) methods have recently outperformed CNN-based models on large-scale datasets. Yet they would lose efficacy when only a small amount of training data is available since it's hard for the crucial queries/kernels to learn localization and shape priors. To this end, this work offers a novel unsupervised pre-training solution for low-data regimes. Inspired by the recent success of the Prompting technique, we introduce a new pre-training method that boosts QEIS models by giving Saliency Prompt for queries/kernels. Our method contains three parts: 1) Saliency Masks Proposal is responsible for generating pseudo masks from unlabeled images based on the saliency mechanism. 2) Prompt-Kernel Matching transfers pseudo masks into prompts and injects the corresponding localization and shape priors to the best-matched kernels. 3) Kernel Supervision is applied to supply supervision at the kernel level for robust learning. From a practical perspective, our pre-training method helps QEIS models achieve a similar convergence speed and comparable performance with CNN-based models in low-data regimes. Experimental results show that our method significantly boosts several QEIS models on three datasets.
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
Code: https://github.com/lifuguan/saliency.prompt

        ----

        ## [1475] Decoupled Semantic Prototypes enable learning from diverse annotation types for semi-weakly segmentation in expert-driven domains

        **Authors**: *Simon Reiß, Constantin Seibold, Alexander Freytag, Erik Rodner, Rainer Stiefelhagen*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01487](https://doi.org/10.1109/CVPR52729.2023.01487)

        **Abstract**:

        A vast amount of images and pixel-wise annotations allowed our community to build scalable segmentation solutions for natural domains. However, the transfer to expert-driven domains like microscopy applications or medical healthcare remains difficult as domain experts are a critical factor due to their limited availability for providing pixel-wise annotations. To enable affordable segmentation solutions for such domains, we need training strategies which can simultaneously handle diverse annotation types and are not bound to costly pixel-wise annotations. In this work, we analyze existing training algorithms towards their flexibility for different annotation types and scalability to small annotation regimes. We conduct an extensive evaluation in the challenging domain of organelle segmentation and find that existing semi- and semi-weakly supervised training algorithms are not able to fully exploit diverse annotation types. Driven by our findings, we introduce Decoupled Semantic Prototypes (DSP) as a training method for semantic segmentation which enables learning from annotation types as diverse as image-level-, point-, bounding box-, and pixel-wise annotations and which leads to remarkable accuracy gains over existing solutions for semi-weakly segmentation.

        ----

        ## [1476] The Treasure Beneath Multiple Annotations: An Uncertainty-Aware Edge Detector

        **Authors**: *Caixia Zhou, Yaping Huang, Mengyang Pu, Qingji Guan, Li Huang, Haibin Ling*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01488](https://doi.org/10.1109/CVPR52729.2023.01488)

        **Abstract**:

        Deep learning-based edge detectors heavily rely on pixel-wise labels which are often provided by multiple annotators. Existing methods fuse multiple annotations using a simple voting process, ignoring the inherent ambiguity of edges and labeling bias of annotators. In this paper, we propose a novel uncertainty-aware edge detector (UAED), which employs uncertainty to investigate the subjectivity and ambiguity of diverse annotations. Specifically, we first convert the deterministic label space into a learnable Gaussian distribution, whose variance measures the degree of ambiguity among different annotations. Then we regard the learned variance as the estimated uncertainty of the predicted edge maps, and pixels with higher uncertainty are likely to be hard samples for edge detection. Therefore we design an adaptive weighting loss to emphasize the learning from those pixels with high uncertainty, which helps the network to gradually concentrate on the important pixels. UAED can be combined with various encoder-decoder backbones, and the extensive experiments demonstrate that UAED achieves superior performance consistently across multiple edge detection benchmarks. The source code is available at https://github.com/ZhouCX117/UAED.

        ----

        ## [1477] Knowledge Combination to Learn Rotated Detection without Rotated Annotation

        **Authors**: *Tianyu Zhu, Bryce Ferenczi, Pulak Purkait, Tom Drummond, Hamid Rezatofighi, Anton van den Hengel*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01489](https://doi.org/10.1109/CVPR52729.2023.01489)

        **Abstract**:

        Rotated bounding boxes drastically reduce output ambiguity of elongated objects, making it superior to axis-aligned bounding boxes. Despite the effectiveness, rotated detectors are not widely employed. Annotating rotated bounding boxes is such a laborious process that they are not provided in many detection datasets where axis-aligned annotations are used instead. In this paper, we propose a framework that allows the model to predict precise rotated boxes only requiring cheaper axis-aligned annotation of the target dataset 
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
. To achieve this, we leverage the fact that neural networks are capable of learning richer representation of the target domain than what is utilized by the task. The under-utilized representation can be exploited to address a more detailed task. Our framework combines task knowledge of an out-of-domain source dataset with stronger annotation and domain knowledge of the target dataset with weaker annotation. A novel assignment process and projection loss are used to enable the cotraining on the source and target datasets. As a result, the model is able to solve the more detailed task in the target domain, without additional computation overhead during inference. We extensively evaluate the method on various target datasets including fresh-produce dataset, HRSC2016 and SSDD. Results show that the proposed method consistently performs on par with the fully supervised approach.

        ----

        ## [1478] Mapping Degeneration Meets Label Evolution: Learning Infrared Small Target Detection with Single Point Supervision

        **Authors**: *Xinyi Ying, Li Liu, Yingqian Wang, Ruojing Li, Nuo Chen, Zaiping Lin, Weidong Sheng, Shilin Zhou*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01490](https://doi.org/10.1109/CVPR52729.2023.01490)

        **Abstract**:

        Training a convolutional neural network (CNN) to detect infrared small targets in a fully supervised manner has gained remarkable research interests in recent years, but is highly labor expensive since a large number of per-pixel annotations are required. To handle this problem, in this paper, we make the first attempt to achieve infrared small target detection with point-level supervision. Interestingly, during the training phase supervised by point labels, we discover that CNNs first learn to segment a cluster of pixels near the targets, and then gradually converge to predict groundtruth point labels. Motivated by this “mapping degeneration” phenomenon, we propose a label evolution framework named label evolution with single point supervision (LESPS) to progressively expand the point label by leveraging the intermediate predictions of CNNs. In this way, the network predictions can finally approximate the updated pseudo labels, and a pixel-level target mask can be obtained to train CNNs in an end-to-end manner. We conduct extensive experiments with insightful visualizations to validate the effectiveness of our method. Experimental results show that CNNs equipped with LESPS can well recover the target masks from corresponding point labels, and can achieve over 70% and 95% of their fully supervised performance in terms of pixel-level intersection over union (IoU) and object-level probability of detection (P
<inf xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">d</inf>
), respectively. Code is available at https://github.com/XinyiYing/LESPS.

        ----

        ## [1479] SAP-DETR: Bridging the Gap Between Salient Points and Queries-Based Transformer Detector for Fast Model Convergency

        **Authors**: *Yang Liu, Yao Zhang, Yixin Wang, Yang Zhang, Jiang Tian, Zhongchao Shi, Jianping Fan, Zhiqiang He*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01491](https://doi.org/10.1109/CVPR52729.2023.01491)

        **Abstract**:

        Recently, the dominant DETR-based approaches apply central-concept spatial prior to accelerating Transformer detector convergency. These methods gradually refine the reference points to the center of target objects and imbue object queries with the updated central reference information for spatially conditional attention. However, centralizing reference points may severely deteriorate queries' saliency and confuse detectors due to the indiscriminative spatial prior. To bridge the gap between the reference points of salient queries and Transformer detectors, we propose SAlient Point-based DETR (SAP-DETR) by treating object detection as a transformation from salient points to instance objects. Concretely, we explicitly initialize a query-specific reference point for each object query, gradually aggregate them into an instance object, and then predict the distance from each side of the bounding box to these points. By rapidly attending to query-specific reference regions and the conditional box edges, SAP-DETR can effectively bridge the gap between the salient point and the query-based Transformer detector with a significant convergency speed. Experimentally, SAP-DETR achieves 1.4× convergency speed with competitive performance and stably promotes the SoTA approaches by ∼1.0 AP. Based on ResNet-DC-101, SAP-DETR achieves 46.9 AP. The code will be released at https://github.com/liuyang-ict/SAP-DETR.

        ----

        ## [1480] Zero-Shot Object Counting

        **Authors**: *Jingyi Xu, Hieu Le, Vu Nguyen, Viresh Ranjan, Dimitris Samaras*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01492](https://doi.org/10.1109/CVPR52729.2023.01492)

        **Abstract**:

        Class-agnostic object counting aims to count object instances of an arbitrary class at test time. Current methods for this challenging problem require human-annotated exemplars as inputs, which are often unavailable for novel categories, especially for autonomous systems. Thus, we propose zero-shot object counting (ZSC), a new setting where only the class name is available during test time. Such a counting system does not require human annotators in the loop and can operate automatically. Starting from a class name, we propose a method that can accurately identify the optimal patches which can then be used as counting exemplars. Specifically, we first construct a class prototype to select the patches that are likely to contain the objects of interest, namely class-relevant patches. Furthermore, we introduce a model that can quantitatively measure how suitable an arbitrary patch is as a counting exemplar. By applying this model to all the candidate patches, we can select the most suitable patches as exemplars for counting. Experimental results on a recent class-agnostic counting dataset, FSC-147, validate the effectiveness of our method. Code is available at https://github.com/cvlabstonybrook/zero-shot-counting.

        ----

        ## [1481] SOOD: Towards Semi-Supervised Oriented Object Detection

        **Authors**: *Wei Hua, Dingkang Liang, Jingyu Li, Xiaolong Liu, Zhikang Zou, Xiaoqing Ye, Xiang Bai*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01493](https://doi.org/10.1109/CVPR52729.2023.01493)

        **Abstract**:

        Semi-Supervised Object Detection (SSOD), aiming to explore unlabeled data for boosting object detectors, has become an active task in recent years. However, existing SSOD approaches mainly focus on horizontal objects, leaving multi-oriented objects that are common in aerial images unexplored. This paper proposes a novel Semi-supervised Oriented Object Detection model, termed SOOD, built upon the mainstream pseudo-labeling framework. Towards oriented objects in aerial scenes, we design two loss functions to provide better supervision. Focusing on the orientations of objects, the first loss regularizes the consistency between each pseudo-label-prediction pair (includes a prediction and its corresponding pseudo label) with adaptive weights based on their orientation gap. Focusing on the layout of an image, the second loss regularizes the similarity and explicitly builds the many-to-many relation between the sets of pseudo-labels and predictions. Such a global consistency constraint can further boost semi-supervised learning. Our experiments show that when trained with the two proposed losses, SOOD surpasses the state-of-the-art SSOD methods under various settings on the DOTAv1.5 benchmark. The code will be available at https://github.com/HamPerdredes/SOOD.

        ----

        ## [1482] Large-scale Training Data Search for Object Re-identification

        **Authors**: *Yue Yao, Tom Gedeon, Liang Zheng*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01494](https://doi.org/10.1109/CVPR52729.2023.01494)

        **Abstract**:

        We consider a scenario where we have access to the target domain, but cannot afford on-the-fly training data annotation, and instead would like to construct an alternative training set from a large-scale data pool such that a competitive model can be obtained. We propose a search and pruning (SnP) solution to this training data search problem, tailored to object re-identification (re-ID), an application aiming to match the same object captured by different cameras. Specifically, the search stage identifies and merges clusters of source identities which exhibit similar distributions with the target domain. The second stage, subject to a budget, then selects identities and their images from the Stage I output, to control the size of the resulting training set for efficient training. The two steps provide us with training sets 80% smaller than the source pool while achieving a similar or even higher re-ID accuracy. These training sets are also shown to be superior to a few existing search methods such as random sampling and greedy sampling under the same budget on training data size. If we release the budget, training sets resulting from the first stage alone allow even higher re-ID accuracy. We provide interesting discussions on the specificity of our method to the re-ID problem and particularly its role in bridging the re-ID domain gap. The code is available at https://github.com/yorkeyao/SnP

        ----

        ## [1483] Ambiguity-Resistant Semi-Supervised Learning for Dense Object Detection

        **Authors**: *Chang Liu, Weiming Zhang, Xiangru Lin, Wei Zhang, Xiao Tan, Junyu Han, Xiaomao Li, Errui Ding, Jingdong Wang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01495](https://doi.org/10.1109/CVPR52729.2023.01495)

        **Abstract**:

        With basic Semi-Supervised Object Detection (SSOD) techniques, one-stage detectors generally obtain limited promotions compared with two-stage clusters. We experimentally find that the root lies in two kinds of ambiguities: (1) Selection ambiguity that selected pseudo labels are less accurate, since classification scores cannot properly represent the localization quality. (2) Assignment ambiguity that samples are matched with improper labels in pseudo-label assignment, as the strategy is misguided by missed objects and inaccurate pseudo boxes. To tackle these problems, we propose a Ambiguity-Resistant Semi-supervised Learning (ARSL) for one-stage detectors. Specifically, to alleviate the selection ambiguity, Joint-Confidence Estimation (JCE) is proposed to jointly quantifies the classification and localization quality of pseudo labels. As for the assignment ambiguity, Task-Separation Assignment (TSA) is introduced to assign labels based on pixel-level predictions rather than unreliable pseudo boxes. It employs a ‘divide-and-conquer’ strategy and separately exploits positives for the classification and localization task, which is more robust to the assignment ambiguity. Comprehensive experiments demonstrate that ARSL effectively mitigates the ambiguities and achieves state-of-the-art SSOD performance on MS COCO and PASCAL VOC. Codes can be found at https://github.com/PaddlePaddle/PaddleDetection.

        ----

        ## [1484] Towards Effective Visual Representations for Partial-Label Learning

        **Authors**: *Shiyu Xia, Jiaqi Lv, Ning Xu, Gang Niu, Xin Geng*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01496](https://doi.org/10.1109/CVPR52729.2023.01496)

        **Abstract**:

        Under partial-label learning (PLL) where, for each training instance, only a set of ambiguous candidate labels containing the unknown true label is accessible, contrastive learning has recently boosted the performance of PLL on vision tasks, attributed to representations learned by contrasting the same/different classes of entities. Without access to true labels, positive points are predicted using pseudolabels that are inherently noisy, and negative points often require large batches or momentum encoders, resulting in unreliable similarity information and a high computational overhead. In this paper, we rethink a state-of-the-art contrastive PLL method PiCO [24], inspiring the design of a simple framework termed PaPi (Partial-label learning with a guided Prototypical classifier), which demonstrates significant scope for improvement in representation learning, thus contributing to label disambiguation. PaPi guides the optimization of a prototypical classifier by a linear classifier with which they share the same feature encoder, thus explicitly encouraging the representation to reflect visual similarity between categories. It is also technically appealing, as PaPi requires only a few components in PiCO with the opposite direction of guidance, and directly eliminates the contrastive learning module that would introduce noise and consume computational resources. We empirically demonstrate that PaPi significantly outperforms other PLL methods on various image classification tasks.

        ----

        ## [1485] Bi3D: Bi-Domain Active Learning for Cross-Domain 3D Object Detection

        **Authors**: *Jiakang Yuan, Bo Zhang, Xiangchao Yan, Tao Chen, Botian Shi, Yikang Li, Yu Qiao*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01497](https://doi.org/10.1109/CVPR52729.2023.01497)

        **Abstract**:

        Unsupervised Domain Adaptation (UDA) technique has been explored in 3D cross-domain tasks recently. Though preliminary progress has been made, the performance gap between the UDA-based 3D model and the supervised one trained with fully annotated target domain is still large. This motivates us to consider selecting partial-yet-important target data and labeling them at a minimum cost, to achieve a good trade-off between high performance and low annotation cost. To this end, we propose a Bi-domain active learning approach, namely Bi3D, to solve the cross-domain 3D object detection task. The Bi3D first develops a domainness-aware source sampling strategy, which identifies target-domain-like samples from the source domain to avoid the model being interfered by irrelevant source data. Then a diversity-based target sampling strategy is developed, which selects the most informative subset of target domain to improve the model adaptability to the target domain using as little annotation budget as possible. Experiments are conducted on typical cross-domain adaptation scenarios including cross-LiDAR-beam, cross-country, and cross-sensor, where Bi3D achieves a promising target-domain detection accuracy (89.63% on KITTI) compared with UDA-based work (84.29%), even surpassing the detector trained on the full set of the labeled target domain (88.98%). Our code is available at: https://github.com/PJLab-ADG/3DTrans.

        ----

        ## [1486] Boosting Detection in Crowd Analysis via Underutilized Output Features

        **Authors**: *Shaokai Wu, Fengyu Yang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01498](https://doi.org/10.1109/CVPR52729.2023.01498)

        **Abstract**:

        Detection-based methods have been viewed unfavorably in crowd analysis due to their poor performance in dense crowds. However, we argue that the potential of these methods has been underestimated, as they offer crucial information for crowd analysis that is often ignored. Specifically, the area size and confidence score of output proposals and bounding boxes provide insight into the scale and density of the crowd. To leverage these underutilized features, we propose Crowd Hat, a plug-and-play module that can be easily integrated with existing detection models. This module uses a mixed 2D-1D compression technique to refine the output features and obtain the spatial and numerical distribution of crowd-specific information. Based on these features, we further propose region-adaptive NMS thresholds and a decouple-then-align paradigm that address the major limitations of detection-based methods. Our extensive evaluations on various crowd analysis tasks, including crowd counting, localization, and detection, demonstrate the effectiveness of utilizing output features and the potential of detection-based methods in crowd analysis. Our code is available at https://github.com/wskingdom/Crowd-Hat.

        ----

        ## [1487] Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture

        **Authors**: *Mahmoud Assran, Quentin Duval, Ishan Misra, Piotr Bojanowski, Pascal Vincent, Michael G. Rabbat, Yann LeCun, Nicolas Ballas*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01499](https://doi.org/10.1109/CVPR52729.2023.01499)

        **Abstract**:

        This paper demonstrates an approach for learning highly semantic image representations without relying on hand-crafted data-augmentations. We introduce the Image-based Joint-Embedding Predictive Architecture (I-JEPA), a non-generative approach for self-supervised learning from images. The idea behind I-JEPA is simple: from a single context block, predict the representations of various target blocks in the same image. A core design choice to guide I-JEPA towards producing semantic representations is the masking strategy; specifically, it is crucial to (a) sample target blocks with sufficiently large scale (semantic), and to (b) use a sufficiently informative (spatially distributed) context block. Empirically, when combined with Vision Transformers, we find I-JEPA to be highly scalable. For instance, we train a ViT-Huge/14 on ImageNet using 16 A100 GPUs in under 72 hours to achieve strong downstream performance across a wide range of tasks, from linear classification to object counting and depth prediction.

        ----

        ## [1488] Weakly Supervised Segmentation with Point Annotations for Histopathology Images via Contrast-Based Variational Model

        **Authors**: *Hongrun Zhang, Liam Burrows, Yanda Meng, Declan Sculthorpe, Abhik Mukherjee, Sarah E. Coupland, Ke Chen, Yalin Zheng*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01500](https://doi.org/10.1109/CVPR52729.2023.01500)

        **Abstract**:

        Image segmentation is a fundamental task in the field of imaging and vision. Supervised deep learning for segmentation has achieved unparalleled success when sufficient training data with annotated labels are available. However, annotation is known to be expensive to obtain, especially for histopathology images where the target regions are usually with high morphology variations and irregular shapes. Thus, weakly supervised learning with sparse annotations of points is promising to reduce the annotation workload. In this work, we propose a contrast-based variational model to generate segmentation results, which serve as reliable complementary supervision to train a deep segmentation model for histopathology images. The proposed method considers the common characteristics of target regions in histopathology images and can be trained in an end-to-end manner. It can generate more regionally consistent and smoother boundary segmentation, and is more robust to unlabeled ‘novel’ regions. Experiments on two different histology datasets demonstrate its effectiveness and efficiency in comparison to previous models. Code is available at: https://github.com/hrzhang1123/CVM_WS_Segmentation.

        ----

        ## [1489] DoNet: Deep De-Overlapping Network for Cytology Instance Segmentation

        **Authors**: *Hao Jiang, Rushan Zhang, Yanning Zhou, Yumeng Wang, Hao Chen*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01501](https://doi.org/10.1109/CVPR52729.2023.01501)

        **Abstract**:

        Cell instance segmentation in cytology images has significant importance for biology analysis and cancer screening, while remains challenging due to 1) the extensive over-lapping translucent cell clusters that cause the ambigu-ous boundaries, and 2) the confusion of mimics and de-bris as nuclei. In this work, we proposed a De-overlapping Network (DoNet) in a decompose-and-recombined strategy. A Dual-path Region Segmentation Module (DRM) explicitly decomposes the cell clusters into intersection and complement regions, followed by a Semantic Consistency-guided Recombination Module (CRM) for integration. To further introduce the containment relationship of the nu-cleus in the cytoplasm, we design a Mask-guided Region Proposal Strategy (MRP) that integrates the cell attention maps for inner-cell instance prediction. We validate the proposed approach on ISBI2014 and CPS datasets. Ex-periments show that our proposed DoNet significantly outperforms other state-of-the-art (SOTA) cell instance segmentation methods. The code is available at https://github.com/DeepDoNet/DoNet.

        ----

        ## [1490] MCF: Mutual Correction Framework for Semi-Supervised Medical Image Segmentation

        **Authors**: *Yongchao Wang, Bin Xiao, Xiuli Bi, Weisheng Li, Xinbo Gao*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01502](https://doi.org/10.1109/CVPR52729.2023.01502)

        **Abstract**:

        Semi-supervised learning is a promising method for medical image segmentation under limited annotation. However, the model cognitive bias impairs the segmentation performance, especially for edge regions. Furthermore, current mainstream semi-supervised medical image segmentation (SSMIS) methods lack designs to handle model bias. The neural network has a strong learning ability, but the cognitive bias will gradually deepen during the training, and it is difficult to correct itself. We propose a novel mutual correction framework (MCF) to explore network bias correction and improve the performance of SSMIS. Inspired by the plain contrast idea, MCF introduces two different subnets to explore and utilize the discrepancies between subnets to correct cognitive bias of the model. More concretely, a contrastive difference review (CDR) module is proposed to find out inconsistent prediction regions and perform a review training. Additionally, a dynamic competitive pseudo-label generation (DCPLG) module is proposed to evaluate the performance of subnets in real-time, dynamically selecting more reliable pseudo-labels. Experimental results on two medical image databases with different modalities (CT and MRI) show that our method achieves superior performance compared to several state-of-the-art methods. The code will be available at https://github.com/WYC-321/MCF.

        ----

        ## [1491] Histopathology Whole Slide Image Analysis with Heterogeneous Graph Representation Learning

        **Authors**: *Tsai Hor Chan, Fernando Julio Cendra, Lan Ma, Guosheng Yin, Lequan Yu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01503](https://doi.org/10.1109/CVPR52729.2023.01503)

        **Abstract**:

        Graph-based methods have been extensively applied to whole slide histopathology image (WSI) analysis due to the advantage of modeling the spatial relationships among different entities. However, most of the existing methods focus on modeling WSIs with homogeneous graphs (e.g., with homogeneous node type). Despite their successes, these works are incapable of mining the complex structural relations between biological entities (e.g., the diverse interaction among different cell types) in the WSI. We propose a novel heterogeneous graph-based framework to leverage the inter-relationships among different types of nuclei for WSI analysis. Specifically, we formulate the WSI as a heterogeneous graph with “nucleus-type” attribute to each node and a semantic similarity attribute to each edge. We then present a new heterogeneous-graph edge attribute transformer (HEAT) to take advantage of the edge and node heterogeneity during massage aggregating. Further, we design a new pseudo-label-based semantic-consistent pooling mechanism to obtain graph-level features, which can mitigate the over-parameterization issue of conventional cluster-based pooling. Additionally, observing the limitations of existing association-based localization methods, we propose a causal-driven approach attributing the contribution of each node to improve the interpretability of our framework. Extensive experiments on three public TCGA benchmark datasets demonstrate that our frame-work outperforms the state-of-the-art methods with considerable margins on various tasks. Our codes are available at https://github.com/HKU-MedAI/WSI-HGNN.

        ----

        ## [1492] PEFAT: Boosting Semi-Supervised Medical Image Classification via Pseudo-Loss Estimation and Feature Adversarial Training

        **Authors**: *Qingjie Zeng, Yutong Xie, Zilin Lu, Yong Xia*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01504](https://doi.org/10.1109/CVPR52729.2023.01504)

        **Abstract**:

        Pseudo-labeling approaches have been proven beneficial for semi-supervised learning (SSL) schemes in computer vision and medical imaging. Most works are dedicated to finding samples with high-confidence pseudo-labels from the perspective of model predicted probability. Whereas this way may lead to the inclusion of incorrectly pseudo-labeled data if the threshold is not carefully adjusted. In addition, low-confidence probability samples are frequently disregarded and not employed to their full potential. In this paper, we propose a novel Pseudo-loss Estimation and Feature Adversarial Training semi-supervised framework, termed as PEFAT, to boost the performance of multi-class and multi-label medical image classification from the point of loss distribution modeling and adversarial training. Specifically, we develop a trustworthy data selection scheme to split a high-quality pseudo-labeled set, inspired by the dividable pseudo-loss assumption that clean data tend to show lower loss while noise data is the opposite. Instead of directly discarding these samples with low-quality pseudo-labels, we present a novel regularization approach to learn discriminate information from them via injecting adversarial noises at the feature-level to smooth the decision boundary. Experimental results on three medical and two natural image benchmarks validate that our PEFAT can achieve a promising performance and surpass other state-of-the-art methods. The code is available at https://github.com/maxwell0027/PEFAT.

        ----

        ## [1493] Causally-Aware Intraoperative Imputation for Overall Survival Time Prediction

        **Authors**: *Xiang Li, Xuelin Qian, Litian Liang, Lingjie Kong, Qiaole Dong, Jiejun Chen, Dingxia Liu, Xiuzhong Yao, Yanwei Fu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01505](https://doi.org/10.1109/CVPR52729.2023.01505)

        **Abstract**:

        Previous efforts in vision community are mostly made on learning good representations from visual patterns. Beyond this, this paper emphasizes the high-level ability of causal reasoning. We thus present a case study of solving the challenging task of Overall Survival (OS) time in primary liver cancers. Critically, the prediction of OS time at the early stage remains challenging, due to the unobvious image patterns of reflecting the OS. To this end, we propose a causal inference system by leveraging the intraoperative attributes and the correlation among them, as an intermediate supervision to bridge the gap between the images and the final OS. Particularly, we build a causal graph, and train the images to estimate the intraoperative attributes for final as prediction. We present a novel Causally-aware Intraoperative Imputation Model (CAWIM) that can sequentially predict each attribute using its parent nodes in the estimated causal graph. To determine the causal directions, we propose a splitting-voting mechanism, which votes for the direction for each pair of adjacent nodes among multiple predictions obtained via causal discovery from heterogeneity. The practicability and effectiveness of our method are demonstrated by the promising results on liver cancer dataset of 361 patients with long-term observations.

        ----

        ## [1494] Balanced Energy Regularization Loss for Out-of-distribution Detection

        **Authors**: *Hyunjun Choi, Hawook Jeong, Jin Young Choi*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01506](https://doi.org/10.1109/CVPR52729.2023.01506)

        **Abstract**:

        In the field of out-of-distribution (OOD) detection, a previous method that use auxiliary data as OOD data has shown promising performance. However, the method provides an equal loss to all auxiliary data to differentiate them from inliers. However, based on our observation, in various tasks, there is a general imbalance in the distribution of the auxiliary OOD data across classes. We propose a balanced energy regularization loss that is simple but generally effective for a variety of tasks. Our balanced energy regularization loss utilizes class-wise different prior probabilities for auxiliary data to address the class imbalance in OOD data. The main concept is to regularize auxiliary samples from majority classes, more heavily than those from minority classes. Our approach performs better for OOD detection in semantic segmentation, long-tailed image classification, and image classification than the prior energy regularization loss. Furthermore, our approach achieves state-of-the-art performance in two tasks: OOD detection in semantic segmentation and long-tailed image classification.

        ----

        ## [1495] Block Selection Method for Using Feature Norm in Out-of-Distribution Detection

        **Authors**: *Yeonguk Yu, Sungho Shin, Seongju Lee, Changhyun Jun, Kyoobin Lee*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01507](https://doi.org/10.1109/CVPR52729.2023.01507)

        **Abstract**:

        Detecting out-of-distribution (OOD) inputs during the inference stage is crucial for deploying neural networks in the real world. Previous methods typically relied on the highly activated feature map outputted by the network. In this study, we revealed that the norm of the feature map obtained from a block other than the last block can serve as a better indicator for OOD detection. To leverage this insight, we propose a simple framework that comprises two metrics: FeatureNorm, which computes the norm of the feature map, and NormRatio, which calculates the ratio of FeatureNorm for ID and OOD samples to evaluate the OOD detection performance of each block. To identify the block that provides the largest difference between FeatureNorm of ID and FeatureNorm of OOD, we create jigsaw puzzles as pseudo OOD from ID training samples and compute NormRatio, selecting the block with the highest value. After identifying the suitable block, OOD detection using FeatureNorm outperforms other methods by reducing FPR95 by up to 52.77% on CIFAR10 benchmark and up to 48.53% on ImageNet benchmark. We demonstrate that our framework can generalize to various architectures and highlight the significance of block selection, which can also improve previous OOD detection methods. Our code is available at https://github.com/gistailab/block-selection-for-OOD-detection.

        ----

        ## [1496] Highly Confident Local Structure Based Consensus Graph Learning for Incomplete Multi-view Clustering

        **Authors**: *Jie Wen, Chengliang Liu, Gehui Xu, Zhihao Wu, Chao Huang, Lunke Fei, Yong Xu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01508](https://doi.org/10.1109/CVPR52729.2023.01508)

        **Abstract**:

        Graph-based multi-view clustering has attracted extensive attention because of the powerful clustering-structure representation ability and noise robustness. Considering the reality of a large amount of incomplete data, in this paper, we propose a simple but effective method for incomplete multi-view clustering based on consensus graph learning, termed as HCLS_CGL. Unlike existing methods that utilize graph constructed from raw data to aid in the learning of consistent representation, our method directly learns a consensus graph across views for clustering. Specifically, we design a novel confidence graph and embed it to form a confidence structure driven consensus graph learning model. Our confidence graph is based on an intuitive similar-nearest-neighbor hypothesis, which does not require any additional information and can help the model to obtain a high-quality consensus graph for better clustering. Numerous experiments are performed to confirm the effectiveness of our method.

        ----

        ## [1497] Siamese DETR

        **Authors**: *Zeren Chen, Gengshi Huang, Wei Li, Jianing Teng, Kun Wang, Jing Shao, Chen Change Loy, Lu Sheng*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01509](https://doi.org/10.1109/CVPR52729.2023.01509)

        **Abstract**:

        Recent self-supervised methods are mainly designed for representation learning with the base model, e.g., ResNets or ViTs. They cannot be easily transferred to DETR, with task-specific Transformer modules. In this work, we present Siamese DETR, a Siamese self-supervised pretraining approach for the Transformer architecture in DETR. We consider learning view-invariant and detection-oriented representations simultaneously through two complementary tasks, i.e., localization and discrimination, in a novel multi-view learning framework. Two self-supervised pretext tasks are designed: (i) Multi-View Region Detection aims at learning to localize regions-of-interest between augmented views of the input, and (ii) Multi-View Semantic Discrimination attempts to improve object-level discrimination for each region. The proposed Siamese DETR achieves state-of-the-art transfer performance on COCO and PASCAL VOC detection using different DETR variants in all setups. Code is available at https://github.com/Zx55/SiameseDEtr.

        ----

        ## [1498] Towards Bridging the Performance Gaps of Joint Energy-Based Models

        **Authors**: *Xiulong Yang, Qing Su, Shihao Ji*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01510](https://doi.org/10.1109/CVPR52729.2023.01510)

        **Abstract**:

        Can we train a hybrid discriminative-generative model with a single network? This question has recently been answered in the affirmative, introducing the field of Joint Energy-based Model (JEM) [17], [48], which achieves high classification accuracy and image generation quality simultaneously. Despite recent advances, there remain two performance gaps: the accuracy gap to the standard softmax classifier, and the generation quality gap to state-of-the-art generative models. In this paper, we introduce a variety of training techniques to bridge the accuracy gap and the generation quality gap of JEM. 1) We incorporate a recently proposed sharpness-aware minimization (SAM) framework to train JEM, which promotes the energy landscape smoothness and the generalization of JEM. 2) We exclude data augmentation from the maximum likelihood estimate pipeline of JEM, and mitigate the negative impact of data augmentation to image generation quality. Extensive experiments on multiple datasets demonstrate our SADA-JEM achieves state-of-the-art performances and outperforms JEM in image classification, image generation, calibration, out-of-distribution detection and adversarial robustness by a notable margin. Our code is available at https://github.com/sndnyang/SADAJEM.

        ----

        ## [1499] Three Guidelines You Should Know for Universally Slimmable Self-Supervised Learning

        **Authors**: *Yun-Hao Cao, Peiqin Sun, Shuchang Zhou*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01511](https://doi.org/10.1109/CVPR52729.2023.01511)

        **Abstract**:

        We propose universally slimmable self-supervised learning (dubbed as US3L) to achieve better accuracy-efficiency trade-offs for deploying self-supervised models across different devices. We observe that direct adaptation of self-supervised learning (SSL) to universally slimmable networks misbehaves as the training process frequently collapses. We then discover that temporal consistent guidance is the key to the success of SSL for universally slimmable networks, and we propose three guidelines for the loss design to ensure this temporal consistency from a unified gradient perspective. Moreover, we propose dynamic sampling and group regularization strategies to simultaneously improve training efficiency and accuracy. Our US3L method has been empirically validated on both convolutional neural networks and vision transformers. With only once training and one copy of weights, our method outperforms various state-of-the-art methods (individually trained or not) on benchmarks including recognition, object detection and instance segmentation.

        ----

        ## [1500] Boosting Transductive Few-Shot Fine-tuning with Margin-based Uncertainty Weighting and Probability Regularization

        **Authors**: *Ran Tao, Hao Chen, Marios Savvides*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01512](https://doi.org/10.1109/CVPR52729.2023.01512)

        **Abstract**:

        Few-Shot Learning (FSL) has been rapidly developed in recent years, potentially eliminating the requirement for significant data acquisition. Few-shot fine-tuning has been demonstrated to be practically efficient and helpful, especially for out-of-distribution datum [7, 13, 17, 29]. In this work, we first observe that the few-shot fine-tuned methods are learned with the imbalanced class marginal distribution, leading to imbalanced per-class testing accuracy. This observation further motivates us to propose the Transductive Fine-tuning with Margin-based uncertainty weighting and Probability regularization (TF-MP), which learns a more balanced class marginal distribution as shown in Fig. 1. We first conduct sample weighting on unlabeled testing data with margin-based uncertainty scores and fur-ther regularize each test sample's categorical probability. TF-MP achieves state-of-the-art performance on in-/out-of-distribution evaluations of Meta- Dataset [31] and sur-passes previous transductive methods by a large margin.

        ----

        ## [1501] CHMATCH: Contrastive Hierarchical Matching and Robust Adaptive Threshold Boosted Semi-Supervised Learning

        **Authors**: *Jianlong Wu, Haozhe Yang, Tian Gan, Ning Ding, Feijun Jiang, Liqiang Nie*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01513](https://doi.org/10.1109/CVPR52729.2023.01513)

        **Abstract**:

        The recently proposed FixMatch and FlexMatch have achieved remarkable results in the field of semi-supervised learning. But these two methods go to two extremes as FixMatch and FlexMatch use a pre-defined constant threshold for all classes and an adaptive threshold for each category, respectively. By only investigating consistency regularization, they also suffer from unstable results and indiscriminative feature representation, especially under the situation of few labeled samples. In this paper, we propose a novel CHMatch method, which can learn robust adaptive thresholds for instance-level prediction matching as well as discriminative features by contrastive hierarchical matching. We first present a memory-bank based robust threshold learning strategy to select highly-confident samples. In the meantime, we make full use of the structured information in the hierarchical labels to learn an accurate affinity graph for contrastive learning. CHMatch achieves very stable and superior results on several commonly-used benchmarks. For example, CHMatch achieves 8.44% and 9.02% error rate reduction over FlexMatch on CIFAR-100 under WRN-28-2 with only 4 and 25 labeled samples per class, respectively
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
Project address: https://github.com/sailist/CHMatch.

        ----

        ## [1502] MarginMatch: Improving Semi-Supervised Learning with Pseudo-Margins

        **Authors**: *Tiberiu Sosea, Cornelia Caragea*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01514](https://doi.org/10.1109/CVPR52729.2023.01514)

        **Abstract**:

        We introduce MarginMatch, a new SSL approach combining consistency regularization and pseudo-labeling, with its main novelty arising from the use of unlabeled data training dynamics to measure pseudo-label quality. Instead of using only the model's confidence on an unlabeled example at an arbitrary iteration to decide if the example should be masked or not, MarginMatch also analyzes the behavior of the model on the pseudo-labeled examples as the training progresses, to ensure low quality predictions are masked out. MarginMatch brings substantial improvements on four vision benchmarks in low data regimes and on two large-scale datasets, emphasizing the importance of enforcing high-quality pseudo-labels. Notably, we obtain an improvement in error rate over the state-of-the-art of 3.25% on CIFAR-100 with only 25 labels per class and of 3.78% on STL-10 using as few as 4 labels per class. We make our code available at https://github.com/tsosea2/MarginMatch.

        ----

        ## [1503] Ranking Regularization for Critical Rare Classes: Minimizing False Positives at a High True Positive Rate

        **Authors**: *Kiarash Mohammadi, He Zhao, Mengyao Zhai, Frederick Tung*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01515](https://doi.org/10.1109/CVPR52729.2023.01515)

        **Abstract**:

        In many real-world settings, the critical class is rare and a missed detection carries a disproportionately high cost. For example, tumors are rare and a false negative diagnosis could have severe consequences on treatment outcomes; fraudulent banking transactions are rare and an undetected occurrence could result in significant losses or legal penalties. In such contexts, systems are often operated at a high true positive rate, which may require tolerating high false positives. In this paper, we present a novel approach to address the challenge of minimizing false positives for systems that need to operate at a high true positive rate. We propose a ranking-based regularization (RankReg) approach that is easy to implement, and show empirically that it not only effectively reduces false positives, but also complements conventional imbalanced learning losses. With this novel technique in hand, we conduct a series of experiments on three broadly explored datasets (CIFAR-10&100 and Melanoma) and show that our approach lifts the previous state-of-the-art performance by notable margins.

        ----

        ## [1504] Learning Imbalanced Data with Vision Transformers

        **Authors**: *Zhengzhuo Xu, Ruikang Liu, Shuo Yang, Zenghao Chai, Chun Yuan*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01516](https://doi.org/10.1109/CVPR52729.2023.01516)

        **Abstract**:

        The real-world data tends to be heavily imbalanced and severely skew the data-driven deep neural networks, which makes Long-Tailed Recognition (LTR) a massive challenging task. Existing LTR methods seldom train Vision Transformers (ViTs) with Long-Tailed (LT) data, while the off-the-shelf pretrain weight of ViTs always leads to unfair comparisons. In this paper, we systematically investigate the ViTs' performance in LTR and propose LiVT to train ViTs from scratch only with LT data. With the observation that ViTs suffer more severe LTR problems, we conduct Masked Generative Pretraining (MGP) to learn generalized features. With ample and solid evidence, we show that MGP is more robust than supervised manners. Although Binary Cross Entropy (BCE) loss performs well with ViTs, it struggles on the LTR tasks. We further propose the balanced BCE to ameliorate it with strong theoretical groundings. Specially, we derive the unbiased extension of Sigmoid and compensate extra logit margins for deploying it. Our Bal-BCE contributes to the quick convergence of ViTs in just a few epochs. Extensive experiments demonstrate that with MGP and Bal-BCE, LiVT successfully trains ViTs well without any additional data and outperforms comparable state-of-the-art methods significantly, e.g., our ViT-B achieves 81.0% Top-1 accuracy in iNaturalist 2018 without bells and whistles. Code is available at https://github.com/XuZhengzhuo/LiVT.

        ----

        ## [1505] No One Left Behind: Improving the Worst Categories in Long-Tailed Learning

        **Authors**: *Yingxiao Du, Jianxin Wu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01517](https://doi.org/10.1109/CVPR52729.2023.01517)

        **Abstract**:

        Unlike the case when using a balanced training dataset, the per-class recall (i.e., accuracy) of neural networks trained with an imbalanced dataset are known to vary a lot from category to category. The convention in long-tailed recognition is to manually split all categories into three subsets and report the average accuracy within each subset. We argue that under such an evaluation setting, some categories are inevitably sacrificed. On one hand, focusing on the average accuracy on a balanced test set incurs little penalty even if some worst performing categories have zero accuracy. On the other hand, classes in the “Few” subset do not necessarily perform worse than those in the “Many” or “Medium” subsets. We therefore advocate to focus more on improving the lowest recall among all categories and the harmonic mean of all recall values. Specifically, we propose a simple plug-in method that is applicable to a wide range of methods. By simply retraining the classifier of an existing pretrained model with our proposed loss function and using an optional ensemble trick that combines the predictions of the two classifiers, we achieve a more uniform distribution of recall values across categories, which leads to a higher harmonic mean accuracy while the (arithmetic) average accuracy is still high. The effectiveness of our method is justified on widely used benchmark datasets.

        ----

        ## [1506] Global and Local Mixture Consistency Cumulative Learning for Long-tailed Visual Recognitions

        **Authors**: *Fei Du, Peng Yang, Qi Jia, Fengtao Nan, Xiaoting Chen, Yun Yang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01518](https://doi.org/10.1109/CVPR52729.2023.01518)

        **Abstract**:

        In this paper, our goal is to design a simple learning paradigm for long-tail visual recognition, which not only improves the robustness of the feature extractor but also alleviates the bias of the classifier towards head classes while reducing the training skills and overhead. We propose an efficient one-stage training strategy for long-tailed visual recognition called Global and Local Mixture Consistency cumulative learning (GLMC). Our core ideas are twofold: (1) a global and local mixture consistency loss improves the robustness of the feature extractor. Specifically, we generate two augmented batches by the global MixUp and local CutMix from the same batch data, respectively, and then use cosine similarity to minimize the difference. (2) A cumulative head-tail soft label reweighted loss mitigates the head class bias problem. We use empirical class frequencies to reweight the mixed label of the head-tail class for long-tailed data and then balance the conventional loss and the rebalanced loss with a coefficient accumulated by epochs. Our approach achieves state-of-the-art accuracy on CIFAR10-LT, CIFAR100-LT, and ImageNet-LT datasets. Additional experiments on balanced ImageNet and CIFAR demonstrate that GLMC can significantly improve the gen-eralization of backbones. Code is made publicly available at https://github.com/ynu-yangpeng/GLMC.

        ----

        ## [1507] Curvature-Balanced Feature Manifold Learning for Long-Tailed Classification

        **Authors**: *Yanbiao Ma, Licheng Jiao, Fang Liu, Shuyuan Yang, Xu Liu, Lingling Li*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01519](https://doi.org/10.1109/CVPR52729.2023.01519)

        **Abstract**:

        To address the challenges of long-tailed classification, researchers have proposed several approaches to reduce model bias, most of which assume that classes with few samples are weak classes. However, recent studies have shown that tail classes are not always hard to learn, and model bias has been observed on sample-balanced datasets, suggesting the existence of other factors that affect model bias. In this work, we systematically propose a series of geometric measurements for perceptual manifolds in deep neural networks, and then explore the effect of the geometric characteristics of perceptual manifolds on classification difficulty and how learning shapes the geometric characteristics of perceptual manifolds. An unanticipated finding is that the correlation between the class accuracy and the separation degree of perceptual manifolds gradually decreases during training, while the negative correlation with the curvature gradually increases, implying that curvature imbalance leads to model bias. Therefore, we propose curvature regularization to facilitate the model to learn curvature-balanced and flatter perceptual manifolds. Evaluations on multiple long-tailed and non-long-tailed datasets show the excellent performance and exciting generality of our approach, especially in achieving significant performance improvements based on current state-of-the-art techniques. Our work opens up a geometric analysis perspective on model bias and reminds researchers to pay attention to model bias on non-long-tailed and even sample-balanced datasets. The code and model will be made public.

        ----

        ## [1508] DAA: A Delta Age AdaIN operation for age estimation via binary code transformer

        **Authors**: *Ping Chen, Xingpeng Zhang, Ye Li, Ju Tao, Bin Xiao, Bing Wang, Zongjie Jiang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01520](https://doi.org/10.1109/CVPR52729.2023.01520)

        **Abstract**:

        Naked eye recognition of age is usually based on comparison with the age of others. However, this idea is ignored by computer tasks because it is difficult to obtain representative contrast images of each age. Inspired by the transfer learning, we designed the Delta Age AdaIN (DAA) operation to obtain the feature difference with each age, which obtains the style map of each age through the learned values representing the mean and standard deviation. We let the input of transfer learning as the binary code of age natural number to obtain continuous age feature information. The learned two groups of values in Binary code mapping are corresponding to the mean and standard deviation of the comparison ages. In summary, our method consists of four parts: FaceEncoder, DAA operation, Binary code mapping, and AgeDecoder modules. After getting the delta age via AgeDecoder, we take the average value of all comparison ages and delta ages as the predicted age. Compared with state-of-the-art methods, our method achieves better performance with fewer parameters on multiple facial age datasets. Code is available at https://github.com/redcping/Delta_Age_AdaIN

        ----

        ## [1509] DLBD: A Self-Supervised Direct-Learned Binary Descriptor

        **Authors**: *Bin Xiao, Yang Hu, Bo Liu, Xiuli Bi, Weisheng Li, Xinbo Gao*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01521](https://doi.org/10.1109/CVPR52729.2023.01521)

        **Abstract**:

        For learning-based binary descriptors, the binarization process has not been well addressed. The reason is that the binarization blocks gradient back-propagation. Existing learning-based binary descriptors learn real-valued output, and then it is converted to binary descriptors by their proposed binarization processes. Since their binarizaiion processes are not a component of the network, the learning-based binary descriptor cannot fully utilize the advances of deep learning. To solve this issue, we propose a model-agnostic plugin binary transformation layer (BTL), making the network directly generate binary descriptors. Then, we present the first self-supervised, direct-learned binary descriptor, dubbed DLBD. Furthermore, we propose ultra-wide temperature-scaled crossentropy loss to adjust the distribution of learned descriptors in a larger range. Experiments demonstrate that the proposed BTL can substitute the previous binarization process. Our proposed DLBD outperforms SOTA on different tasks such as image retrieval and classification
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
Our code is available at: https://github.com/CQUPT-CV/DLBD.

        ----

        ## [1510] Progressive Open Space Expansion for Open-Set Model Attribution

        **Authors**: *Tianyun Yang, Danding Wang, Fan Tang, Xinying Zhao, Juan Cao, Sheng Tang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01522](https://doi.org/10.1109/CVPR52729.2023.01522)

        **Abstract**:

        Despite the remarkable progress in generative technology, the Janus-faced issues of intellectual property protection and malicious content supervision have arisen. Efforts have been paid to manage synthetic images by attributing them to a set of potential source models. However, the closed-set classification setting limits the application in real-world scenarios for handling contents generated by arbitrary models. In this study, we focus on a challenging task, namely Open-Set Model Attribution (OSMA), to simultaneously attribute images to known models and identify those from unknown ones. Compared to existing openset recognition (OSR) tasks focusing on semantic novelty, OSMA is more challenging as the distinction between images from known and unknown models may only lie in visually imperceptible traces. To this end, we propose a Progressive Open Space Expansion (POSE) solution, which simulates open-set samples that maintain the same semantics as closed-set samples but embedded with different imperceptible traces. Guided by a diversity constraint, the open space is simulated progressively by a set of lightweight augmentation models. We consider three real-world scenarios and construct an OSMA benchmark dataset, including unknown models trained with different random seeds, architectures, and datasets from known ones. Extensive experiments on the dataset demonstrate POSE is superior to both existing model attribution methods and off-the-shelf OSR methods. Github: https://github.com/ICTMCG/POSE

        ----

        ## [1511] DiGA: Distil to Generalize and then Adapt for Domain Adaptive Semantic Segmentation

        **Authors**: *Fengyi Shen, Akhil Gurram, Ziyuan Liu, He Wang, Alois Knoll*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01523](https://doi.org/10.1109/CVPR52729.2023.01523)

        **Abstract**:

        Domain adaptive semantic segmentation methods commonly utilize stage-wise training, consisting of a warm-up and a self-training stage. However, this popular approach still faces several challenges in each stage: for warm-up, the widely adopted adversarial training often results in limited performance gain, due to blind feature alignment; for self-training, finding proper categorical thresholds is very tricky. To alleviate these issues, we first propose to replace the adversarial training in the warm-up stage by a novel symmetric knowledge distillation module that only accesses the source domain data and makes the model domain generalizable. Surprisingly, this domain generalizable warm-up model brings substantial performance improvement, which can be further amplified via our proposed cross-domain mixture data augmentation technique. Then, for the self-training stage, we propose a threshold-free dynamic pseudo-label selection mechanism to ease the aforementioned threshold problem and make the model better adapted to the target domain. Extensive experiments demonstrate that our framework achieves remarkable and consistent improvements compared to the prior arts on popular benchmarks. Codes and models are available at https://github.com/fy-visionIDiGA

        ----

        ## [1512] Multi-Modal Learning with Missing Modality via Shared-Specific Feature Modelling

        **Authors**: *Hu Wang, Yuanhong Chen, Congbo Ma, Jodie Avery, Louise Hull, Gustavo Carneiro*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01524](https://doi.org/10.1109/CVPR52729.2023.01524)

        **Abstract**:

        The missing modality issue is critical but non-trivial to be solved by multi-modal models. Current methods aiming to handle the missing modality problem in multi-modal tasks, either deal with missing modalities only during evaluation or train separate models to handle specific missing modality settings. In addition, these models are designed for specific tasks, so for example, classification models are not easily adapted to segmentation tasks and vice versa. In this paper, we propose the Shared-Specific Feature Modelling (ShaSpec) method that is considerably simpler and more effective than competing approaches that address the issues above. ShaSpec is designed to take advantage of all available input modalities during training and evaluation by learning shared and specific features to better represent the input data. This is achieved from a strategy that relies on auxiliary tasks based on distribution alignment and domain classification, in addition to a residual feature fusion procedure. Also, the design simplicity of ShaSpec enables its easy adaptation to multiple tasks, such as classification and segmentation. Experiments are conducted on both medical image segmentation and computer vision classification, with results indicating that ShaSpec outperforms competing methods by a large margin. For instance, on BraTS2018, ShaSpec improves the SOTA by more than 3% for enhancing tumour, 5% for tumour core and 3% for whole tumour.
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
This work received funding from the Australian Government the through Medical Research Futures Fund: Primary Health Care Research Data Infrastructure Grant 2020 and from Endometriosis Australia. G.C. was supported by Australian Research Council through grant FT190100525.

        ----

        ## [1513] Towards All-in-One Pre-Training via Maximizing Multi-Modal Mutual Information

        **Authors**: *Weijie Su, Xizhou Zhu, Chenxin Tao, Lewei Lu, Bin Li, Gao Huang, Yu Qiao, Xiaogang Wang, Jie Zhou, Jifeng Dai*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01525](https://doi.org/10.1109/CVPR52729.2023.01525)

        **Abstract**:

        To effectively exploit the potential of large-scale models, various pre-training strategies supported by massive data from different sources are proposed, including supervised pre-training, weakly-supervised pre-training, and self-supervised pre-training. It has been proved that combining multiple pre-training strategies and data from various modalities/sources can greatly boost the training of large-scale models. However, current works adopt a multi-stage pre-training system, where the complex pipeline may increase the uncertainty and instability of the pre-training. It is thus desirable that these strategies can be integrated in a single-stage manner. In this paper, we first propose a general multimodal mutual information formula as a unified optimization target and demonstrate that all mainstream approaches are special cases of our framework. Under this unified perspective, we propose an all-in-one single-stage pre-training approach, named Maximizing Multi-modal Mutual Information Pre-Training (M3I Pre-training). Our approach achieves better performance than previous pre-training methods on various vision benchmarks, including ImageNet classification, COCO object detection, LVIS long-tailed object detection, and ADE20k semantic segmentation. Notably, we successfully pre-train a billion-level parameter image backbone and achieve state-of-the-art performance on various benchmarks under public data setting. Code shall be released at https://github.com/OpenGVLab/M3I-Pre-Training.

        ----

        ## [1514] Bi-Level Meta-Learning for Few-Shot Domain Generalization

        **Authors**: *Xiaorong Qin, Xinhang Song, Shuqiang Jiang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01526](https://doi.org/10.1109/CVPR52729.2023.01526)

        **Abstract**:

        The goal of few-shot learning is to learn the generalization from seen to unseen data with only a few samples. Most previous few-shot learning methods focus on learning the generalization within particular domains. However, the more practical scenarios may also require the generalization ability across domains. In this paper, we study the problem of few-shot domain generalization (FSDG), which is a more challenging variant of few-shot classification. FSDG requires additional generalization with larger gap from seen domains to unseen domains. We address FSDG problem by meta-learning two levels of meta-knowledge, where the lower-level meta-knowledge is domain-specific embedding spaces as subspaces of a base space for intra-domain generalization, and the upper-level meta-knowledge is the base space and a prior subspace over domain-specific spaces for inter-domain generalization. We formulate the two levels of meta-knowledge learning problem with bi-level optimization, and further develop an optimization algorithm without higher-order derivative information to solve it. We demonstrate our method is significantly superior to the previous works by evaluating it on the widely used benchmark Meta-Dataset.

        ----

        ## [1515] Train/Test-Time Adaptation with Retrieval

        **Authors**: *Luca Zancato, Alessandro Achille, Tian Yu Liu, Matthew Trager, Pramuditha Perera, Stefano Soatto*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01527](https://doi.org/10.1109/CVPR52729.2023.01527)

        **Abstract**:

        We introduce Train/Test-Time Adaptation with Retrieval (T
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">3</sup>
 AR), a method to adapt models both at train and test time by means of a retrieval module and a searchable pool of external samples. Before inference, T
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">3</sup>
AR adapts a given model to the downstream task using refined pseudo-labels and a self-supervised contrastive objective function whose noise distribution leverages retrieved real samples to improve feature adaptation on the target data manifold. The retrieval of real images is key to T
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">3</sup>
AR since it does not rely solely on synthetic data augmentations to compensate for the lack of adaptation data, as typically done by other adaptation algorithms. Furthermore, thanks to the retrieval module, our method gives the user or service provider the possibility to improve model adaptation on the downstream task by incorporating further relevant data or to fully remove samples that may no longer be available due to changes in user preference after deployment. First, we show that T
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">3</sup>
AR can be used at training time to improve downstream fine-grained classification over standard fine-tuning baselines, and the fewer the adaptation data the higher the relative improvement (up to 13%). Second, we apply T
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">3</sup>
ARfor test-time adaptation and show that exploiting a pool of external images at test-time leads to more robust representations over existing methods on DomainNet-126 and VISDA-C, especially when few adaptation data are available (up to 8%).

        ----

        ## [1516] Robust Test-Time Adaptation in Dynamic Scenarios

        **Authors**: *Longhui Yuan, Binhui Xie, Shuang Li*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01528](https://doi.org/10.1109/CVPR52729.2023.01528)

        **Abstract**:

        Test-time adaptation (TTA) intends to adapt the pretrained model to test distributions with only unlabeled test data streams. Most of the previous TTA methods have achieved great success on simple test data streams such as independently sampled data from single or multiple distributions. However, these attempts may fail in dynamic scenarios of real-world applications like autonomous driving, where the environments gradually change and the test data is sampled correlatively over time. In this work, we explore such practical test data streams to deploy the model on the fly, namely practical test-time adaptation (PTTA). To do so, we elaborate a Robust Test-Time Adaptation (RoTTA) method against the complex data stream in PTTA. More specifically, we present a robust batch normalization scheme to estimate the normalization statistics. Meanwhile, a memory bank is utilized to sample category-balanced data with consideration of timeliness and uncertainty. Further, to stabilize the training procedure, we develop a time-aware reweighting strategy with a teacher-student model. Extensive experiments prove that RoTTA enables continual test-time adaptation on the correlatively sampled data streams. Our method is easy to implement, making it a good choice for rapid deployment. The code is publicly available at https://github.com/BIT-DA/RoTTA

        ----

        ## [1517] Domain Expansion of Image Generators

        **Authors**: *Yotam Nitzan, Michaël Gharbi, Richard Zhang, Taesung Park, Jun-Yan Zhu, Daniel Cohen-Or, Eli Shechtman*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01529](https://doi.org/10.1109/CVPR52729.2023.01529)

        **Abstract**:

        Can one inject new concepts into an already trained generative model, while respecting its existing structure and knowledge? We propose a new task - domain expansion - to address this. Given a pretrained generator and novel (but related) domains, we expand the generator to jointly model all domains, old and new, harmoniously. First, we note the generator contains a meaningful, pretrained latent space. Is it possible to minimally perturb this hard-earned representation, while maximally representing the new domains? Interestingly, we find that the latent space offers unused, “dormant” directions, which do not affect the output. This provides an opportunity: By “repurposing” these directions, we can represent new domains without perturbing the original representation. In fact, we find that pretrained generators have the capacity to add several- even hundreds - of new domains! Using our expansion method, one “expanded” model can supersede numerous domain-specific models, without expanding the model size. Additionally, a single expanded generator natively supports smooth transitions between domains, as well as composition of domains. Code and project page available here.

        ----

        ## [1518] Switchable Representation Learning Framework with Self-Compatibility

        **Authors**: *Shengsen Wu, Yan Bai, Yihang Lou, Xiongkun Linghu, Jianzhong He, Ling-Yu Duan*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01530](https://doi.org/10.1109/CVPR52729.2023.01530)

        **Abstract**:

        Real-world visual search systems involve deployments on multiple platforms with different computing and storage resources. Deploying a unified model that suits the minimal-constrain platforms leads to limited accuracy. It is expected to deploy models with different capacities adapting to the resource constraints, which requires features extracted by these models to be aligned in the metric space. The method to achieve feature alignments is called “compatible learning”. Existing research mainly focuses on the one-to-one compatible paradigm, which is limited in learning compatibility among multiple models. We propose a Switchable representation learning Framework with Self-Compatibility (SFSC). SFSC generates a series of compatible sub-models with different capacities through one training process. The optimization of sub-models faces gradients conflict, and we mitigate this problem from the perspective of the magnitude and direction. We adjust the priorities of sub-models dynamically through uncertainty estimation to co-optimize sub-models properly. Besides, the gradients with conflicting directions are projected to avoid mutual interference. SFSC achieves state-of-the-art performance on the evaluated datasets.

        ----

        ## [1519] A New Benchmark: On the Utility of Synthetic Data with Blender for Bare Supervised Learning and Downstream Domain Adaptation

        **Authors**: *Hui Tang, Kui Jia*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01531](https://doi.org/10.1109/CVPR52729.2023.01531)

        **Abstract**:

        Deep learning in computer vision has achieved great success with the price of large-scale labeled training data. However, exhaustive data annotation is impracticable for each task of all domains of interest, due to high labor costs and unguaranteed labeling accuracy. Besides, the uncontrollable data collection process produces non-IID training and test data, where undesired duplication may exist. All these nuisances may hinder the verification of typical theories and exposure to new findings. To circumvent them, an alternative is to generate synthetic data via 3D rendering with domain randomization. We in this work push forward along this line by doing profound and extensive research on bare supervised learning and downstream domain adaptation. Specifically, under the well-controlled, IID data setting enabled by 3D rendering, we systematically verify the typical, important learning insights, e.g., shortcut learning, and discover the new laws of various data regimes and network architectures in generalization. We further investigate the effect of image formation factors on generalization, e.g., object scale, material texture, illumination, camera view-point, and background in a 3D scene. Moreover, we use the simulation-to-reality adaptation as a downstream task for comparing the transferability between synthetic and real data when used for pre-training, which demonstrates that synthetic data pre-training is also promising to improve real test results. Lastly, to promote future research, we develop a new large-scale synthetic-to-real benchmark for image classification, termed S2RDA, which provides more significant challenges for transfer from simulation to reality.

        ----

        ## [1520] Adapting Shortcut with Normalizing Flow: An Efficient Tuning Framework for Visual Recognition

        **Authors**: *Yaoming Wang, Bowen Shi, Xiaopeng Zhang, Jin Li, Yuchen Liu, Wenrui Dai, Chenglin Li, Hongkai Xiong, Qi Tian*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01532](https://doi.org/10.1109/CVPR52729.2023.01532)

        **Abstract**:

        Pretraining followed by fine-tuning has proven to be effective in visual recognition tasks. However, fine-tuning all parameters can be computationally expensive, particularly for large-scale models. To mitigate the computational and storage demands, recent research has explored Parameter-Efficient Fine-Tuning (PEFT), which focuses on tuning a minimal number of parameters for efficient adaptation. Existing methods, however, fail to analyze the impact of the additional parameters on the model, resulting in an unclear and suboptimal tuning process. In this paper, we introduce a novel and effective PEFT paradigm, named SNF (Shortcut adaptation via Normalization Flow), which utilizes normalizing flows to adjust the shortcut layers. We highlight that layers without Lipschitz constraints can lead to error propagation when adapting to downstream datasets. Since modifying the over-parameterized residual connections in these layers is expensive, we focus on adjusting the cheap yet crucial shortcuts. Moreover, learning new information with few parameters in PEFT can be challenging, and information loss can result in label information degradation. To address this issue, we propose an information-preserving normalizing flow. Experimental results demonstrate the effectiveness of SNF. Specifically, with only 0.036M parameters, SNF surpasses previous approaches on both the FGVC and VTAB-1k benchmarks using ViT/B-16 as the backbone. The code is available at https://github.com/Wang-Yaoming/SNF

        ----

        ## [1521] Manipulating Transfer Learning for Property Inference

        **Authors**: *Yulong Tian, Fnu Suya, Anshuman Suri, Fengyuan Xu, David Evans*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01533](https://doi.org/10.1109/CVPR52729.2023.01533)

        **Abstract**:

        Transfer learning is a popular method for tuning pretrained (upstream) models for different downstream tasks using limited data and computational resources. We study how an adversary with control over an upstream model used in transfer learning can conduct property inference attacks on a victim's tuned downstream model. For example, to infer the presence of images of a specific individual in the downstream training set. We demonstrate attacks in which an adversary can manipulate the upstream model to conduct highly effective and specific property inference attacks (AUC score > 0.9), without incurring significant performance loss on the main task. The main idea of the manipulation is to make the upstream model generate activations (intermediate features) with different distributions for samples with and without a target property, thus enabling the adversary to distinguish easily between downstream models trained with and without training examples that have the target property. Our code is available at https://github.com/yulongt23/Transfer-Inference.

        ----

        ## [1522] Heterogeneous Continual Learning

        **Authors**: *Divyam Madaan, Hongxu Yin, Wonmin Byeon, Jan Kautz, Pavlo Molchanov*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01534](https://doi.org/10.1109/CVPR52729.2023.01534)

        **Abstract**:

        We propose a novel framework and a solution to tackle the continual learning (CL) problem with changing network architectures. Most CL methods focus on adapting a single architecture to a new task/class by modifying its weights. However, with rapid progress in architecture design, the problem of adapting existing solutions to novel architectures becomes relevant. To address this limitation, we propose Heterogeneous Continual Learning (HCL), where a wide range of evolving network architectures emerge continually together with novel data/tasks. As a solution, we build on top of the distillation family of techniques and modify it to a new setting where a weaker model takes the role of a teacher; meanwhile, a new stronger architecture acts as a student. Furthermore, we consider a setup of limited access to previous data and propose Quick Deep Inversion (QDI) to recover prior task visual features to support knowledge transfer. QDI significantly reduces computational costs compared to previous solutions and improves overall performance. In summary, we propose a new setup for CL with a modified knowledge distillation paradigm and design a quick data inversion method to enhance distillation. Our evaluation of various benchmarks shows a significant improvement on accuracy in comparison to state-of-the-art methods over various networks architectures.

        ----

        ## [1523] Generic-to-Specific Distillation of Masked Autoencoders

        **Authors**: *Wei Huang, Zhiliang Peng, Li Dong, Furu Wei, Jianbin Jiao, Qixiang Ye*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01535](https://doi.org/10.1109/CVPR52729.2023.01535)

        **Abstract**:

        Large vision Transformers (ViTs) driven by self-supervised pre-training mechanisms achieved unprecedented progress. Lightweight ViT models limited by the model capacity, however, benefit little from those pre-training mechanisms. Knowledge distillation defines a paradigm to transfer representations from large (teacher) models to small (student) ones. However, the conventional single-stage distillation easily gets stuck on task-specific transfer, failing to retain the task-agnostic knowledge crucial for model generalization. In this study, we propose generic-to-specific distillation (G2SD), to tap the potential of small ViT models under the supervision of large models pretrained by masked autoencoders. In generic distillation, decoder of the small model is encouraged to align feature predictions with hidden representations of the large model, so that task-agnostic knowledge can be transferred. In specific distillation, predictions of the small model are constrained to be consistent with those of the large model, to transfer task-specific features which guarantee task performance. With G2SD, the vanilla ViT-Small model respectively achieves 98.7%, 98.1% and 99.3% the performance of its teacher (ViT-Base) for image classification, object detection, and semantic segmentation, setting a solid baseline for two-stage vision distillation. Code will be available at https://github.com/pengzhiliang/G2SD

        ----

        ## [1524] Towards a Smaller Student: Capacity Dynamic Distillation for Efficient Image Retrieval

        **Authors**: *Yi Xie, Huaidong Zhang, Xuemiao Xu, Jianqing Zhu, Shengfeng He*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01536](https://doi.org/10.1109/CVPR52729.2023.01536)

        **Abstract**:

        Previous Knowledge Distillation based efficient image retrieval methods employ a lightweight network as the stu-dent model for fast inference. However, the lightweight stu-dent model lacks adequate representation capacity for effective knowledge imitation during the most critical early training period, causing final performance degeneration. To tackle this issue, we propose a Capacity Dynamic Distillation framework, which constructs a student model with editable representation capacity. Specifically, the employed student model is initially a heavy model to fruitfully learn distilled knowledge in the early training epochs, and the stu-dent model is gradually compressed during the training. To dynamically adjust the model capacity, our dynamic frame-work inserts a learnable convolutional layer within each residual block in the student model as the channel importance indicator. The indicator is optimized simultaneously by the image retrieval loss and the compression loss, and a retrieval- guided gradient resetting mechanism is proposed to release the gradient conflict. Extensive experiments show that our method has superior inference speed and accu-racy, e.g., on the VeRi-776 dataset, given the ResNet101 as a teacher, our method saves 67.13% model parameters and 65.67% FLOPs without sacrificing accuracy. Code is avail-able at https://github.com/SCY-X/Capacity_Dynamic_Distillation.

        ----

        ## [1525] CafeBoost: Causal Feature Boost to Eliminate Task-Induced Bias for Class Incremental Learning

        **Authors**: *Benliu Qiu, Hongliang Li, Haitao Wen, Heqian Qiu, Lanxiao Wang, Fanman Meng, Qingbo Wu, Lili Pan*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01537](https://doi.org/10.1109/CVPR52729.2023.01537)

        **Abstract**:

        Continual learning requires a model to incrementally learn a sequence of tasks and aims to predict well on all the learned tasks so far, which notoriously suffers from the catastrophic forgetting problem. In this paper, we find a new type of bias appearing in continual learning, coined as task-induced bias. We place continual learning into a causal framework, based on which we find the task-induced bias is reduced naturally by two underlying mechanisms in task and domain incremental learning. However, these mechanisms do not exist in class incremental learning (CIL), in which each task contains a unique subset of classes. To eliminate the task-induced bias in CIL, we devise a causal intervention operation so as to cut off the causal path that causes the task-induced bias, and then implement it as a causal debias module that transforms biased features into unbiased ones. In addition, we propose a training pipeline to incorporate the novel module into existing methods and jointly optimize the entire architecture. Our overall approach does not rely on data replay, and is simple and convenient to plug into existing methods. Extensive empirical study on CIFAR-100 and ImageNet shows that our approach can improve accuracy and reduce forgetting of well-established methods by a large margin.

        ----

        ## [1526] Bilateral Memory Consolidation for Continual Learning

        **Authors**: *Xing Nie, Shixiong Xu, Xiyan Liu, Gaofeng Meng, Chunlei Huo, Shiming Xiang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01538](https://doi.org/10.1109/CVPR52729.2023.01538)

        **Abstract**:

        Humans are proficient at continuously acquiring and integrating new knowledge. By contrast, deep models forget catastrophically, especially when tackling highly long task sequences. Inspired by the way our brains constantly rewrite and consolidate past recollections, we propose a novel Bilateral Memory Consolidation (BiMeCo) framework that focuses on enhancing memory interaction capabilities. Specifically, BiMeCo explicitly decouples model parameters into short-term memory module and long-term memory module, responsible for representation ability of the model and generalization over all learned tasks, respectively. BiMeCo encourages dynamic interactions between two memory modules by knowledge distillation and momentum-based updating for forming generic knowledge to prevent forgetting. The proposed BiMeCo is parameter-efficient and can be integrated into existing methods seamlessly. Extensive experiments on challenging benchmarks show that BiMeCo significantly improves the performance of existing continual learning methods. For example, combined with the state-of-the-art method CwD [55], BiMeCo brings in significant gains of around 2% to 6% while using 2x fewer parameters on CIFAR-100 under ResNet-18.

        ----

        ## [1527] NICO++: Towards Better Benchmarking for Domain Generalization

        **Authors**: *Xingxuan Zhang, Yue He, Renzhe Xu, Han Yu, Zheyan Shen, Peng Cui*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01539](https://doi.org/10.1109/CVPR52729.2023.01539)

        **Abstract**:

        Despite the remarkable performance that modern deep neural networks have achieved on independent and identically distributed (I.I.D.) data, they can crash under distribution shifts. Most current evaluation methods for do-main generalization (DG) adopt the leave-one-out strategy as a compromise on the limited number of domains. We propose a large-scale benchmark with extensive labeled domains named 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$NICO^{++}$</tex>
 along with more rational evaluation methods for comprehensively evaluating DC algorithms. To evaluate DG datasets, we propose two metrics to quantify covariate shift and concept shift, respectively. Two novel generalization bounds from the perspective of data construction are proposed to prove that limited concept shift and significant covariate shift favor the evaluation capability for generalization. Through extensive experiments, 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$NlCO^{++}$</tex>
 shows its superior evaluation capability compared with current DG datasets and its contribution in alleviating unfairness caused by the leak of oracle knowledge in model selection. The data and code for the benchmark based on 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$NICO^{++}$</tex>
 are available at https://github.com/xxgege/NICO-plus.

        ----

        ## [1528] DART: Diversify-Aggregate-Repeat Training Improves Generalization of Neural Networks

        **Authors**: *Samyak Jain, Sravanti Addepalli, Pawan Kumar Sahu, Priyam Dey, R. Venkatesh Babu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01540](https://doi.org/10.1109/CVPR52729.2023.01540)

        **Abstract**:

        Generalization of Neural Networks is crucial for deploying them safely in the real world. Common training strategies to improve generalization involve the use of data augmentations, ensembling and model averaging. In this work, we first establish a surprisingly simple but strong benchmark for generalization which utilizes diverse augmentations within a training minibatch, and show that this can learn a more balanced distribution of features. Further, we propose Diversify-Aggregate-Repeat Training (DART) strategy that first trains diverse models using different augmentations (or domains) to explore the loss basin, and further Aggregates their weights to combine their expertise and obtain improved generalization. We find that Repeating the step of Aggregation throughout training improves the over-all optimization trajectory and also ensures that the individual models have sufficiently low loss barrier to obtain improved generalization on combining them. We theoretically justify the proposed approach and show that it indeed generalizes better. In addition to improvements in In-Domain generalization, we demonstrate SOTA performance on the Domain Generalization benchmarks in the popular DomainBed framework as well. Our method is generic and can easily be integrated with several base training algorithms to achieve performance gains. Our code is available here: https://github.com/val-iisc/DART.

        ----

        ## [1529] Differentiable Architecture Search with Random Features

        **Authors**: *Xuanyang Zhang, Yonggang Li, Xiangyu Zhang, Yongtao Wang, Jian Sun*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01541](https://doi.org/10.1109/CVPR52729.2023.01541)

        **Abstract**:

        Differentiable architecture search (DARTS) has signif-icantly promoted the development of NAS techniques because of its high search efficiency and effectiveness but suf-fers from performance collapse. In this paper, we make efforts to alleviate the performance collapse problem for DARTS from two aspects. First, we investigate the expres-sive power of the supernet in DARTS and then derive a new setup of DARTS paradigm with only training Batch-Norm. Second, we theoretically find that random features dilute the auxiliary connection role of skip-connection in supernet optimization and enable search algorithm focus on fairer operation selection, thereby solving the performance collapse problem. We instantiate DARTS and PC-DARTS with random features to build an improved version for each named RF-DARTS and RF-PCDARTS respectively. Experimental results show that RF-DARTS obtains 94.36% test accuracy on CIFAR-10 (which is the nearest optimal result in NAS-Bench-201), and achieves the newest state-of-the-art top-1 test error of 24.0% on ImageNet when transferring from CIFAR-10. Moreover, RF-DARTS performs robustly across three datasets (CIFAR-10, CIFAR-100, and SVHN) and four search spaces (S1-S4). Besides, RF-PCDARTS achieves even better results on ImageNet, that is, 23.9% top-1 and 7.1% top-5 test error, surpassing representative methods like single-path, training-free, and partial-channel paradigms directly searched on ImageNet.

        ----

        ## [1530] Class Adaptive Network Calibration

        **Authors**: *Bingyuan Liu, Jérôme Rony, Adrian Galdran, Jose Dolz, Ismail Ben Ayed*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01542](https://doi.org/10.1109/CVPR52729.2023.01542)

        **Abstract**:

        Recent studies have revealed that, beyond conventional accuracy, calibration should also be considered for training modern deep neural networks. To address miscalibration during learning, some methods have explored different penalty functions as part of the learning objective, along-side a standard classification loss, with a hyper-parameter controlling the relative contribution of each term. Nevertheless, these methods share two major drawbacks: 1) the scalar balancing weight is the same for all classes, hindering the ability to address different intrinsic difficulties or imbalance among classes; and 2) the balancing weight is usually fixed without an adaptive strategy, which may prevent from reaching the best compromise between accuracy and calibration, and requires hyper-parameter search for each application. We propose Class Adaptive Label Smoothing (CALS) for calibrating deep networks, which allows to learn class-wise multipliers during training, yielding a powerful alternative to common label smoothing penalties. Our method builds on a general Augmented Lagrangian approach, a well-established technique in constrained optimization, but we introduce several modifications to tailor it for large-scale, class-adaptive training. Comprehensive evaluation and multiple comparisons on a variety of benchmarks, including standard and long-tailed image classification, semantic segmentation, and text classification, demonstrate the superiority of the proposed method. The code is available at https://github.com/by-liu/CALS.

        ----

        ## [1531] Meta-Learning with a Geometry-Adaptive Preconditioner

        **Authors**: *Suhyun Kang, Duhun Hwang, Moonjung Eo, Taesup Kim, Wonjong Rhee*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01543](https://doi.org/10.1109/CVPR52729.2023.01543)

        **Abstract**:

        Model-agnostic meta-learning (MAML) is one of the most successful meta-learning algorithms. It has a bi-level optimization structure where the outer-loop process learns a shared initialization and the inner-loop process optimizes task-specific weights. Although MAML relies on the standard gradient descent in the inner-loop, recent studies have shown that controlling the inner-loop's gradient descent with a meta-learned preconditioner can be beneficial. Existing preconditioners, however, cannot simultaneously adapt in a task-specific and path-dependent way. Additionally, they do not satisfy the Riemannian metric condition, which can enable the steepest descent learning with preconditioned gradient. In this study, we propose Geometry-Adaptive Preconditioned gradient descent (GAP) that can overcome the limitations in MAML; GAP can efficiently meta-learn a preconditioner that is dependent on task-specific parameters, and its preconditioner can be shown to be a Riemannian metric. Thanks to the two properties, the geometry-adaptive preconditioner is effective for improving the inner-loop optimization. Experiment results show that GAP outperforms the state-of-the-art MAML family and preconditioned gradient descent-MAML (PGD-MAML) family in a variety of few-shot learning tasks. Code is available at: https://github.com/Suhyun777/CVPR23-GAP.

        ----

        ## [1532] DepGraph: Towards Any Structural Pruning

        **Authors**: *Gongfan Fang, Xinyin Ma, Mingli Song, Michael Bi Mi, Xinchao Wang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01544](https://doi.org/10.1109/CVPR52729.2023.01544)

        **Abstract**:

        Structural pruning enables model acceleration by removing structurally-grouped parameters from neural networks. However, the parameter-grouping patterns vary widely across different models, making architecture-specific pruners, which rely on manually-designed grouping schemes, non-generalizable to new architectures. In this work, we study a highly-challenging yet barely-explored task, any structural pruning, to tackle general structural pruning of arbitrary architecture like CNNs, RNNs, GNNs and Transformers. The most prominent obstacle towards this goal lies in the structural coupling, which not only forces different layers to be pruned simultaneously, but also expects all removed parameters to be consistently unimportant, thereby avoiding structural issues and significant performance degradation after pruning. To address this problem, we propose a general and fully automatic method, Dependency Graph (DepGraph), to explicitly model the dependency between layers and comprehensively group coupled parameters for pruning. In this work, we extensively evaluate our method on several architectures and tasks, including ResNe(X)t, DenseNet, MobileNet and Vision transformer for images, GAT for graph, DGCNN for 3D point cloud, alongside LSTM for language, and demonstrate that, even with a simple norm-based criterion, the proposed method consistently yields gratifying performances.

        ----

        ## [1533] Stitchable Neural Networks

        **Authors**: *Zizheng Pan, Jianfei Cai, Bohan Zhuang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01545](https://doi.org/10.1109/CVPR52729.2023.01545)

        **Abstract**:

        The public model zoo containing enormous powerful pretrained model families (e.g., ResNet/DeiT) has reached an unprecedented scope than ever, which significantly contributes to the success of deep learning. As each model family consists of pretrained models with diverse scales (e.g., DeiT-Ti/S/B), it naturally arises a fundamental question of how to efficiently assemble these readily available models in a family for dynamic accuracy-efficiency trade-offs at runtime. To this end, we present Stitchable Neural Networks (SN-Net), a novel scalable and efficient framework for model deployment. It cheaply produces numerous networks with different complexity and performance trade-offs given a family of pretrained neural networks, which we call anchors. Specifically, SN-Net splits the anchors across the blocks/layers and then stitches them together with simple stitching layers to map the activations from one anchor to another. With only a few epochs of training, SN-Net effectively interpolates between the performance of anchors with varying scales. At runtime, SN-Net can instantly adapt to dynamic resource constraints by switching the stitching positions. Extensive experiments on ImageNet classification demonstrate that SN-Net can obtain on-par or even better performance than many individually trained networks while supporting diverse deployment scenarios. For example, by stitching Swin Transformers, we challenge hundreds of models in Timm model zoo with a single network. We believe this new elastic model framework can serve as a strong baseline for further research in wider communities.

        ----

        ## [1534] Integral Neural Networks

        **Authors**: *Kirill Solodskikh, Azim Kurbanov, Ruslan Aydarkhanov, Irina Zhelavskaya, Yury Parfenov, Dehua Song, Stamatios Lefkimmiatis*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01546](https://doi.org/10.1109/CVPR52729.2023.01546)

        **Abstract**:

        We introduce a new family of deep neural networks, where instead of the conventional representation of network layers as N-dimensional weight tensors, we use a continuous layer representation along the filter and channel dimensions. We call such networks Integral Neural Networks (INNs). In particular, the weights of INNs are represented as continuous functions defined on N-dimensional hypercubes, and the discrete transformations of inputs to the layers are replaced by continuous integration operations, accordingly. During the inference stage, our continuous layers can be converted into the traditional tensor representation via numerical integral quadratures. Such kind of representation allows the discretization of a network to an arbitrary size with various discretization intervals for the integral kernels. This approach can be applied to prune the model directly on an edge device while suffering only a small performance loss at high rates of structural pruning without any fine-tuning. To evaluate the practical benefits of our proposed approach, we have conducted experiments using various neural network architectures on multiple tasks. Our reported results show that the proposed INNs achieve the same performance with their conventional discrete counterparts, while being able to preserve approximately the same performance (2% accuracy loss for ResNet18 on Imagenet) at a high rate (up to 30%) of structural pruning without fine-tuning, compared to 65% accuracy loss of the conventional pruning methods under the same conditions. Code is available at gitee.

        ----

        ## [1535] Regularization of polynomial networks for image recognition

        **Authors**: *Grigorios G. Chrysos, Bohan Wang, Jiankang Deng, Volkan Cevher*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01547](https://doi.org/10.1109/CVPR52729.2023.01547)

        **Abstract**:

        Deep Neural Networks (DNNs) have obtained impressive performance across tasks, however they still remain as black boxes, e.g., hard to theoretically analyze. At the same time, Polynomial Networks (PNs) have emerged as an alternative method with a promising performance and improved interpretability but have yet to reach the performance of the powerful DNN baselines. In this work, we aim to close this performance gap. We introduce a class of PNs, which are able to reach the performance of ResNet across a range of six benchmarks. We demonstrate that strong regularization is critical and conduct an extensive study of the exact regularization schemes required to match performance. To further motivate the regularization schemes, we introduce D-PolyNets that achieve a higher- degree of expansion than previously proposed polynomial networks. D-PolyNets are more parameter-efficient while achieving a similar performance as other polynomial networks. We expect that our new models can lead to an understanding of the role of elementwise activation functions (which are no longer required for training PNs). The source code is available at https://github.com/grigorisg9gr/regularized_polynomials.

        ----

        ## [1536] ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders

        **Authors**: *Sanghyun Woo, Shoubhik Debnath, Ronghang Hu, Xinlei Chen, Zhuang Liu, In So Kweon, Saining Xie*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01548](https://doi.org/10.1109/CVPR52729.2023.01548)

        **Abstract**:

        Driven by improved architectures and better representation learning frameworks, the field of visual recognition has enjoyed rapid modernization and performance boost in the early 2020s. For example, modern ConvNets, represented by ConvNeXt [33], have demonstrated strong performance in various scenarios. While these models were originally designed for supervised learning with ImageNet labels, they can also potentially benefit from self-supervised learning techniques such as masked autoencoders (MAE) [14]. However, we found that simply combining these two approaches leads to subpar performance. In this paper, we propose a fully convolutional masked autoencoder framework and a new Global Response Normalization (GRN) layer that can be added to the ConvNeXt architecture to enhance inter-channel feature competition. This co-design of self-supervised learning techniques and architectural improvement results in a new model family called ConvNeXt V2, which significantly improves the performance of pure ConvNets on various recognition benchmarks, including ImageNet classification, COCO detection, and ADE20K segmentation. We also provide pre-trained ConvNeXt V2 models of various sizes, ranging from an efficient 3.7M-parameter Atto model with 76.7% top-1 accuracy on ImageNet, to a 650M Huge model that achieves a state-of-the-art 88.9% accuracy using only public training data.

        ----

        ## [1537] Shortcomings of Top-Down Randomization-Based Sanity Checks for Evaluations of Deep Neural Network Explanations

        **Authors**: *Alexander Binder, Leander Weber, Sebastian Lapuschkin, Grégoire Montavon, Klaus-Robert Müller, Wojciech Samek*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01549](https://doi.org/10.1109/CVPR52729.2023.01549)

        **Abstract**:

        While the evaluation of explanations is an important step towards trustworthy models, it needs to be done carefully, and the employed metrics need to be well-understood. Specifically model randomization testing can be overinterpreted if regarded as a primary criterion for selecting or discarding explanation methods. To address shortcomings of this test, we start by observing an experimental gap in the ranking of explanation methods between randomization-based sanity checks [1] and model output faithfulness measures (e.g. [20]). We identify limitations of model-randomization-based sanity checks for the purpose of evaluating explanations. Firstly, we show that uninformative attribution maps created with zero pixel-wise covariance easily achieve high scores in this type of checks. Secondly, we show that top-down model randomization preserves scales of forward pass activations with high probability. That is, channels with large activations have a high probility to contribute strongly to the output, even after randomization of the network on top of them. Hence, explanations after randomization can only be expected to differ to a certain extent. This explains the observed experimental gap. In summary, these results demonstrate the inadequacy of model-randomization-based sanity checks as a criterion to rank attribution methods.

        ----

        ## [1538] Don't Lie to Me! Robust and Efficient Explainability with Verified Perturbation Analysis

        **Authors**: *Thomas Fel, Melanie Ducoffe, David Vigouroux, Rémi Cadène, Mikael Capelle, Claire Nicodème, Thomas Serre*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01550](https://doi.org/10.1109/CVPR52729.2023.01550)

        **Abstract**:

        A plethora of attribution methods have recently been developed to explain deep neural networks. These methods use different classes of perturbations (e.g, occlusion, blurring, masking, etc) to estimate the importance of individual image pixels to drive a model's decision. Nevertheless, the space of possible perturbations is vast and current attribution methods typically require significant computation time to accurately sample the space in order to achieve high-quality explanations. In this work, we introduce EVA (Explaining using Verified Perturbation Analysis) - the first explainability method which comes with guarantees that an entire set of possible perturbations has been exhaustively searched. We leverage recent progress in verified perturbation analysis methods to directly propagate bounds through a neural network to exhaustively probe a - potentially infinite-size - set of perturbations in a single forward pass. Our approach takes advantage of the beneficial properties of verified perturbation analysis, i.e., time efficiency and guaranteed complete - sampling agnostic - coverage of the perturbation space - to identify image pixels that drive a model's decision. We evaluate EVA systematically and demonstrate state-of-the-art results on multiple benchmarks. Our code isfreely available: github.com/deel-ai/formal-explainability

        ----

        ## [1539] OT-Filter: An Optimal Transport Filter for Learning with Noisy Labels

        **Authors**: *Chuanwen Feng, Yilong Ren, Xike Xie*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01551](https://doi.org/10.1109/CVPR52729.2023.01551)

        **Abstract**:

        The success of deep learning is largely attributed to the training over clean data. However, data is often coupled with noisy labels in practice. Learning with noisy labels is challenging because the performance of the deep neural networks (DNN) drastically degenerates, due to confirmation bias caused by the network memorization over noisy labels. To alleviate that, a recent prominent direction is on sample selection, which retrieves clean data samples from noisy samples, so as to enhance the model's robustness and tolerance to noisy labels. In this paper, we re-vamp the sample selection from the perspective of optimal transport theory and propose a novel method, called the OT-Filter. The OT-Filter provides geometrically meaningful distances and preserves distribution patterns to measure the data discrepancy, thus alleviating the confirmation bias. Extensive experiments on benchmarks, such as Clothing1M and ANIMAL-10N, show that the performance of the OT-Filter outperforms its counterparts. Meanwhile, results on benchmarks with synthetic labels, such as CIFAR-10/100, show the superiority of the OT-Filter in handling data labels of high noise.

        ----

        ## [1540] Robust Generalization Against Photon-Limited Corruptions via Worst-Case Sharpness Minimization

        **Authors**: *Zhuo Huang, Miaoxi Zhu, Xiaobo Xia, Li Shen, Jun Yu, Chen Gong, Bo Han, Bo Du, Tongliang Liu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01552](https://doi.org/10.1109/CVPR52729.2023.01552)

        **Abstract**:

        Robust generalization aims to tackle the most challenging data distributions which are rare in the training set and contain severe noises, i.e., photon-limited corruptions. Common solutions such as distributionally robust optimization (DRO) focus on the worst-case empirical risk to ensure low training error on the uncommon noisy distributions. However, due to the over-parameterized model being optimized on scarce worst-case data, DRO fails to produce a smooth loss landscape, thus struggling on generalizing well to the test set. Therefore, instead of focusing on the worst-case risk minimization, we propose SharpDRO by penalizing the sharpness of the worst-case distribution, which measures the loss changes around the neighbor of learning parameters. Through worst-case sharpness minimization, the proposed method successfully produces a flat loss curve on the corrupted distributions, thus achieving robust generalization. Moreover, by considering whether the distribution annotation is available, we apply SharpDRO to two problem settings and design a worst-case selection process for robust generalization. Theoretically, we show that SharpDRO has a great convergence guarantee. Experimentally, we simulate photon-limited corruptions using CIFAR10/100 and ImageNet30 datasets and show that SharpDRO exhibits a strong generalization ability against severe corruptions and exceeds well-known baseline methods with large performance gains.

        ----

        ## [1541] Learning with Noisy labels via Self-supervised Adversarial Noisy Masking

        **Authors**: *Yuanpeng Tu, Boshen Zhang, Yuxi Li, Liang Liu, Jian Li, Jiangning Zhang, Yabiao Wang, Chengjie Wang, Cairong Zhao*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01553](https://doi.org/10.1109/CVPR52729.2023.01553)

        **Abstract**:

        Collecting large-scale datasets is crucial for training deep models, annotating the data, however, inevitably yields noisy labels, which poses challenges to deep learning algorithms. Previous efforts tend to mitigate this problem via identifying and removing noisy samples or correcting their labels according to the statistical properties (e.g., loss values) among training samples. In this paper, we aim to tackle this problem from a new perspective, delving into the deep feature maps, we empirically find that models trained with clean and mislabeled samples manifest distinguishable activation feature distributions. From this observation, a novel robust training approach termed adversarial noisy masking is proposed. The idea is to regularize deep features with a label quality guided masking scheme, which adaptively modulates the input data and label simultaneously, preventing the model to overfit noisy samples. Further, an auxiliary task is designed to reconstruct input data, it naturally provides noise-free self-supervised signals to rein-force the generalization ability of models. The proposed method is simple yet effective, it is tested on synthetic and real-world noisy datasets, where significant improvements are obtained over previous methods. Code is available at https://github.com/yuanpengtu/SANM.

        ----

        ## [1542] Bit-shrinking: Limiting Instantaneous Sharpness for Improving Post-training Quantization

        **Authors**: *Chen Lin, Bo Peng, Zheyang Li, Wenming Tan, Ye Ren, Jun Xiao, Shiliang Pu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01554](https://doi.org/10.1109/CVPR52729.2023.01554)

        **Abstract**:

        Post-training quantization (PTQ) is an effective compression method to reduce the model size and computational cost. However, quantizing a model into a low-bit one, e.g., lower than 4, is difficult and often results in non-negligible performance degradation. To address this, we investigate the loss landscapes of quantized networks with various bit-widths. We show that the network with more ragged loss surface, is more easily trapped into bad local minima, which mostly appears in low-bit quantization. A deeper analysis indicates, the ragged surface is caused by the injection of excessive quantization noise. To this end, we detach a sharpness term from the loss which reflects the impact of quantization noise. To smooth the rugged loss surface, we propose to limit the sharpness term small and stable during optimization. Instead of directly optimizing the target bit network, we design a self-adapted shrinking scheduler for the bit-width in continuous domain from high bit-width to the target by limiting the increasing sharpness term within a proper range. It can be viewed as iteratively adding small “instant” quantization noise and adjusting the network to eliminate its impact. Widely experiments including classification and detection tasks demonstrate the effectiveness of the Bit-shrinking strategy in PTQ. On the Vision Transformer models, our INT8 and INT6 models drop within 0.5% and 1.5% Top-1 accuracy, respectively. On the traditional CNN networks, our INT4 quantized models drop within 1.3% and 3.5% Top-1 accuracy on ResNet18 and MobileNetV2 without fine-tuning, which achieves the state-of-the-art performance.

        ----

        ## [1543] Enhancing Multiple Reliability Measures via Nuisance-Extended Information Bottleneck

        **Authors**: *Jongheon Jeong, Sihyun Yu, Hankook Lee, Jinwoo Shin*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01555](https://doi.org/10.1109/CVPR52729.2023.01555)

        **Abstract**:

        In practical scenarios where training data is limited, many predictive signals in the data can be rather from some biases in data acquisition (i.e., less generalizable), so that one cannot prevent a model from co-adapting on such (so-called) “shortcut” signals: this makes the model fragile in various distribution shifts. To bypass such failure modes, we consider an adversarial threat model under a mutual information constraint to cover a wider class of perturbations in training. This motivates us to extend the standard information bottleneck to additionally model the nuisance information. We propose an autoencoder-based training to implement the objective, as well as practical encoder designs to facilitate the proposed hybrid discriminative-generative training concerning both convolutional- and Transformer-based architectures. Our experimental results show that the proposed scheme improves robustness of learned representations (remarkably without using any domain-specific knowledge), with respect to multiple challenging reliability measures. For example, our model could advance the state-of-the-art on a recent challenging OBJECTS benchmark in novelty detection by 78.4% → 87.2% in AUROC, while simultaneously enjoying improved corruption, background and (certified) adversarial robustness. Code is available at https://github.com/jh-jeong/nuisance_ib.

        ----

        ## [1544] AdaptiveMix: Improving GAN Training via Feature Space Shrinkage

        **Authors**: *Haozhe Liu, Wentian Zhang, Bing Li, Haoqian Wu, Nanjun He, Yawen Huang, Yuexiang Li, Bernard Ghanem, Yefeng Zheng*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01556](https://doi.org/10.1109/CVPR52729.2023.01556)

        **Abstract**:

        Due to the outstanding capability for data generation, Generative Adversarial Networks (GANs) have attracted considerable attention in unsupervised learning. However, training GANs is difficult, since the training distribution is dynamic for the discriminator, leading to unstable image representation. In this paper, we address the problem of training GANs from a novel perspective, i.e., robust image classification. Motivated by studies on robust image representation, we propose a simple yet effective module, namely AdaptiveMix, for GANs, which shrinks the regions of training data in the image representation space of the discriminator. Considering it is intractable to directly bound feature space, we propose to construct hard samples and narrow down the feature distance between hard and easy samples. The hard samples are constructed by mixing a pair of training images. We evaluate the effectiveness of our AdaptiveMix with widely-used and state-of-the-art GAN architectures. The evaluation results demonstrate that our AdaptiveMix can facilitate the training of GANs and effectively improve the image quality of generated samples. We also show that our AdaptiveMix can be further applied to image classification and Out-Of-Distribution (OOD) detection tasks, by equipping it with state-of-the-art methods. Extensive experiments on seven publicly available datasets show that our method effectively boosts the performance of baselines. The code is publicly available at https://github.com/WentianZhang-ML/AdaptiveMix.

        ----

        ## [1545] Re-GAN: Data-Efficient GANs Training via Architectural Reconfiguration

        **Authors**: *Divya Saxena, Jiannong Cao, Jiahao Xu, Tarun Kulshrestha*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01557](https://doi.org/10.1109/CVPR52729.2023.01557)

        **Abstract**:

        Training Generative Adversarial Networks (GANs) on high-fidelity images usually requires a vast number of training images. Recent research on GAN tickets reveals that dense GANs models contain sparse sub-networks or “lottery tickets” that, when trained separately, yield better results under limited data. However, finding GANs tickets requires an expensive process of train-prune-retrain. In this paper, we propose Re-GAN, a data-efficient GANs training that dynamically reconfigures GANs architecture during training to explore different sub-network structures in training time. Our method repeatedly prunes unimportant connections to regularize GANs network and regrows them to reduce the risk of prematurely pruning important connections. Re-GAN stabilizes the GANs models with less data and offers an alternative to the existing GANs tickets and progressive growing methods. We demonstrate that Re-GAN is a generic training methodology which achieves stability on datasets of varying sizes, domains, and resolutions (CIFAR-10, Tiny-ImageNet, and multiple few-shot generation datasets) as well as different GANs architectures (SNGAN, ProGAN, StyleGAN2 and AutoGAN). Re-GAN also improves performance when combined with the recent augmentation approaches. Moreover, Re-GAN requires fewer floating-point operations (FLOPs) and less training time by removing the unimportant connections during GANs training while maintaining comparable or even generating higher-quality samples. When compared to state-of-the-art StyleGAN2, our method outperforms without requiring any additional fine-tuning step. Code can be found at this link: https://github.com/IntellicentAI-Lab/Re-GAN

        ----

        ## [1546] Soft Augmentation for Image Classification

        **Authors**: *Yang Liu, Shen Yan, Laura Leal-Taixé, James Hays, Deva Ramanan*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01558](https://doi.org/10.1109/CVPR52729.2023.01558)

        **Abstract**:

        Modern neural networks are over-parameterized and thus rely on strong regularization such as data augmentation and weight decay to reduce overfitting and improve generalization. The dominant form of data augmentation applies invariant transforms, where the learning target of a sample is invariant to the transform applied to that sample. We draw inspiration from human visual classification studies and propose generalizing augmentation with invariant transforms to soft augmentation where the learning target softens non-linearly as a function of the degree of the transform applied to the sample: e.g., more aggressive image crop augmentations produce less confident learning targets. We demonstrate that soft targets allow for more aggressive data augmentation, offer more robust performance boosts, work with other augmentation policies, and interestingly, produce better calibrated models (since they are trained to be less confident on aggressively cropped/occluded examples). Combined with existing aggressive augmentation strategies, soft targets 1) double the top-1 accuracy boost across Cifar-10, Cifar-100, ImageNet-1K, and ImageNet-V2, 2) improve model occlusion performance by up to 4×, and 3) half the expected calibration error (ECE). Finally, we show that soft augmentation generalizes to self-supervised classification tasks. Code available at https://github.com/youngleox/soft_augmentation

        ----

        ## [1547] Boosting Verified Training for Robust Image Classifications via Abstraction

        **Authors**: *Zhaodi Zhang, Zhiyi Xue, Yang Chen, Si Liu, Yueling Zhang, Jing Liu, Min Zhang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01559](https://doi.org/10.1109/CVPR52729.2023.01559)

        **Abstract**:

        This paper proposes a novel, abstraction-based, certified training method for robust image classifiers. Via abstraction, all perturbed images are mapped into intervals before feeding into neural networks for training. By training on intervals, all the perturbed images that are mapped to the same interval are classified as the same label, rendering the variance of training sets to be small and the loss landscape of the models to be smooth. Consequently, our approach significantly improves the robustness of trained models. For the abstraction, our training method also enables a sound and complete black-box verification approach, which is orthogonal and scalable to arbitrary types of neural networks regardless of their sizes and architectures. We evaluate our method on a wide range of benchmarks in different scales. The experimental results show that our method outperforms state of the art by (i) reducing the verified errors of trained models up to 95.64%; (ii) totally achieving up to 602.50x speedup; and (iii) scaling up to larger models with up to 138 million trainable parameters. The demo is available at https://github.com/zhangzhaodi233/ABSCERT.git.

        ----

        ## [1548] A New Dataset Based on Images Taken by Blind People for Testing the Robustness of Image Classification Models Trained for ImageNet Categories

        **Authors**: *Reza Akbarian Bafghi, Danna Gurari*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01560](https://doi.org/10.1109/CVPR52729.2023.01560)

        **Abstract**:

        Our goal is to improve upon the status quo for designing image classification models trained in one domain that perform well on images from another domain. Complementing existing work in robustness testing, we introduce the first dataset for this purpose which comes from an authentic use case where photographers wanted to learn about the content in their images. We built a new test set using 8,900 images taken by people who are blind for which we collected metadata to indicate the presence versus absence of 200 ImageNet object categories. We call this dataset VizWiz-Classification. We characterize this dataset and how it compares to the mainstream datasets for evaluating how well ImageNet-trained classification models generalize. Finally, we analyze the performance of 100 ImageNet classification models on our new test dataset. Our fine-grained analysis demonstrates that these models struggle on images with quality issues. To enable future extensions to this work, we share our new dataset with evaluation server at: https://VizWiz.org/tasks-and-datasets/image-classification.

        ----

        ## [1549] Exploiting Completeness and Uncertainty of Pseudo Labels for Weakly Supervised Video Anomaly Detection

        **Authors**: *Chen Zhang, Guorong Li, Yuankai Qi, Shuhui Wang, Laiyun Qing, Qingming Huang, Ming-Hsuan Yang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01561](https://doi.org/10.1109/CVPR52729.2023.01561)

        **Abstract**:

        Weakly supervised video anomaly detection aims to identify abnormal events in videos using only video-level labels. Recently, two-stage self-training methods have achieved significant improvements by self-generating pseudo labels and self-refining anomaly scores with these labels. As the pseudo labels play a crucial role, we propose an enhancement framework by exploiting completeness and uncertainty properties for effective self-training. Specifically, we first design a multi-head classification module (each head serves as a classifier) with a diversity loss to maximize the distribution differences of predicted pseudo labels across heads. This encourages the generated pseudo labels to cover as many abnormal events as possible. We then devise an iterative uncertainty pseudo label refinement strategy, which improves not only the initial pseudo labels but also the updated ones obtained by the desired classifier in the second stage. Extensive experimental results demonstrate the proposed method performs favorably against state-of-the-art approaches on the UCF-Crime, TAD, and XD-Violence benchmark datasets.

        ----

        ## [1550] Prototypical Residual Networks for Anomaly Detection and Localization

        **Authors**: *Hui Zhang, Zuxuan Wu, Zheng Wang, Zhineng Chen, Yu-Gang Jiang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01562](https://doi.org/10.1109/CVPR52729.2023.01562)

        **Abstract**:

        Anomaly detection and localization are widely used in industrial manufacturing for its efficiency and effectiveness. Anomalies are rare and hard to collect and supervised models easily over-fit to these seen anomalies with a handful of abnormal samples, producing unsatisfactory performance. On the other hand, anomalies are typically subtle, hard to discern, and of various appearance, making it difficult to detect anomalies and let alone locate anomalous regions. To address these issues, we propose a framework called Prototypical Residual Network (PRN), which learns feature residuals of varying scales and sizes between anomalous and normal patterns to accurately reconstruct the segmentation maps of anomalous regions. PRN mainly consists of two parts: multi-scale prototypes that explicitly represent the residual features of anomalies to normal patterns; a multisize self-attention mechanism that enables variable-sized anomalous feature learning. Besides, we present a variety of anomaly generation strategies that consider both seen and unseen appearance variance to enlarge and diversify anomalies. Extensive experiments on the challenging and widely used MVTec AD benchmark show that PRN outperforms current state-of-the-art unsupervised and supervised methods. We further report SOTA results on three additional datasets to demonstrate the effectiveness and generalizability of PRN.

        ----

        ## [1551] Class Balanced Adaptive Pseudo Labeling for Federated Semi-Supervised Learning

        **Authors**: *Ming Li, Qingli Li, Yan Wang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01563](https://doi.org/10.1109/CVPR52729.2023.01563)

        **Abstract**:

        This paper focuses on federated semi-supervised learning (FSSL), assuming that few clients have fully labeled data (labeled clients) and the training datasets in other clients are fully unlabeled (unlabeled clients). Existing methods attempt to deal with the challenges caused by not independent and identically distributed data (Non-IID) setting. Though methods such as sub-consensus models have been proposed, they usually adopt standard pseudo labeling or consistency regularization on unlabeled clients which can be easily influenced by imbalanced class distribution. Thus, problems in FSSL are still yet to be solved. To seek for a fundamental solution to this problem, we present Class Balanced Adaptive Pseudo Labeling (CBAFed), to study FSSL from the perspective of pseudo labeling. In CBAFed, the first key element is a fixed pseudo labeling strategy to handle the catastrophic forgetting problem, where we keep a fixed set by letting pass information of unlabeled data at the beginning of the unlabeled client training in each communication round. The second key element is that we design class balanced adaptive thresholds via considering the empirical distribution of all training data in local clients, to encourage a balanced training process. To make the model reach a better optimum, we further propose a residual weight connection in local supervised training and global model aggregation. Extensive experiments on five datasets demonstrate the superiority of CBAFed. Code will be available at https://github.com/minglllli/CBAFed.

        ----

        ## [1552] Fair Federated Medical Image Segmentation via Client Contribution Estimation

        **Authors**: *Meirui Jiang, Holger R. Roth, Wenqi Li, Dong Yang, Can Zhao, Vishwesh Nath, Daguang Xu, Qi Dou, Ziyue Xu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01564](https://doi.org/10.1109/CVPR52729.2023.01564)

        **Abstract**:

        How to ensure fairness is an important topic in federated learning (FL). Recent studies have investigated how to reward clients based on their contribution (collaboration fairness), and how to achieve uniformity of performance across clients (performance fairness). Despite achieving progress on either one, we argue that it is critical to consider them together, in order to engage and motivate more diverse clients joining FL to derive a high-quality global model. In this work, we propose a novel method to optimize both types of fairness simultaneously. Specifically, we propose to estimate client contribution in gradient and data space. In gradient space, we monitor the gradient direction differences of each client with respect to others. And in data space, we measure the prediction error on client data using an auxiliary model. Based on this contribution estimation, we propose a FL method, federated training via contribution estimation (FedCE), i.e., using estimation as global model aggregation weights. We have theoretically analyzed our method and empirically evaluated it on two real-world medical datasets. The effectiveness of our approach has been validated with significant performance improvements, better collaboration fairness, better performance fairness, and comprehensive analytical studies. Code is available at https://nvidia.github.io/NVFlare/research/fed-ce

        ----

        ## [1553] Rethinking Federated Learning with Domain Shift: A Prototype View

        **Authors**: *Wenke Huang, Mang Ye, Zekun Shi, He Li, Bo Du*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01565](https://doi.org/10.1109/CVPR52729.2023.01565)

        **Abstract**:

        Federated learning shows a bright promise as a privacy-preserving collaborative learning technique. However, prevalent solutions mainly focus on all private data sampled from the same domain. An important challenge is that when distributed data are derived from diverse domains. The private model presents degenerative performance on other domains (with domain shift). Therefore, we expect that the global model optimized after the federated learning process stably provides generalizability performance on multiple domains. In this paper, we propose Federated Proto-types Learning (FPL) for federated learning under domain shift. The core idea is to construct cluster prototypes and unbiased prototypes, providing fruitful domain knowledge and a fair convergent target. On the one hand, we pull the sample embedding closer to cluster prototypes belonging to the same semantics than cluster prototypes from distinct classes. On the other hand, we introduce consistency regularization to align the local instance with the respective unbiased prototype. Empirical results on Digits and Office Caltech tasks demonstrate the effectiveness of the proposed solution and the efficiency of crucial modules.

        ----

        ## [1554] FedDM: Iterative Distribution Matching for Communication-Efficient Federated Learning

        **Authors**: *Yuanhao Xiong, Ruochen Wang, Minhao Cheng, Felix Yu, Cho-Jui Hsieh*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01566](https://doi.org/10.1109/CVPR52729.2023.01566)

        **Abstract**:

        Federated learning (FL) has recently attracted increasing attention from academia and industry, with the ultimate goal of achieving collaborative training under privacy and communication constraints. Existing iterative model averaging based FL algorithms require a large number of communication rounds to obtain a well-performed model due to extremely unbalanced and non-i.i.d data partitioning among different clients. Thus, we propose FedDM to build the global training objective from multiple local surrogate functions, which enables the server to gain a more global view of the loss landscape. In detail, we construct synthetic sets of data on each client to locally match the loss landscape from original data through distribution matching. FedDM reduces communication rounds and improves model quality by transmitting more informative and smaller synthesized data compared with unwieldy model weights. We conduct extensive experiments on three image classification datasets, and show that our method outperforms other FL counterparts in terms of efficiency and model performance given a limited number of communication rounds. Moreover, we demonstrate that FedDM can be adapted to preserve differential privacy with Gaussian mechanism and train a better model under the same privacy budget.

        ----

        ## [1555] Alias-Free Convnets: Fractional Shift Invariance via Polynomial Activations

        **Authors**: *Hagay Michaeli, Tomer Michaeli, Daniel Soudry*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01567](https://doi.org/10.1109/CVPR52729.2023.01567)

        **Abstract**:

        Although CNNs are believed to be invariant to translations, recent works have shown this is not the case due to aliasing effects that stem from down-sampling layers. The existing architectural solutions to prevent the aliasing effects are partial since they do not solve those effects that originate in non-linearities. We propose an extended antialiasing method that tackles both down-sampling and nonlinear layers, thus creating truly alias-free, shift-invariant CNNs
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
Our code is available at github.com/hmichaeli/alias_free_convnets/.. We show that the presented model is invariant to integer as well as fractional (i.e., sub-pixel) translations, thus outperforming other shift-invariant methods in terms of robustness to adversarial translations.

        ----

        ## [1556] STDLens: Model Hijacking-Resilient Federated Learning for Object Detection

        **Authors**: *Ka Ho Chow, Ling Liu, Wenqi Wei, Fatih Ilhan, Yanzhao Wu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01568](https://doi.org/10.1109/CVPR52729.2023.01568)

        **Abstract**:

        Federated Learning (FL) has been gaining popularity as a collaborative learning framework to train deep learning-based object detection models over a distributed population of clients. Despite its advantages, FL is vulnerable to model hijacking. The attacker can control how the object detection system should misbehave by implanting Trojaned gradients using only a small number of compromised clients in the collaborative learning process. This paper introduces STDLens, a principled approach to safeguarding FL against such attacks. We first investigate existing mitigation mechanisms and analyze their failures caused by the inherent errors in spatial clustering analysis on gradients. Based on the insights, we introduce a three-tier forensic framework to identify and expel Trojaned gradients and reclaim the performance over the course of FL. We consider three types of adaptive attacks and demonstrate the robustness of STDLens against advanced adversaries. Extensive experiments show that STDLens can protect FL against different model hijacking attacks and outperform existing methods in identifying and removing Trojaned gradients with significantly higher precision and much lower false-positive rates. The source code is available at https://github.com/git-disl/STDLens.

        ----

        ## [1557] Detecting Backdoors in Pre-trained Encoders

        **Authors**: *Shiwei Feng, Guanhong Tao, Siyuan Cheng, Guangyu Shen, Xiangzhe Xu, Yingqi Liu, Kaiyuan Zhang, Shiqing Ma, Xiangyu Zhang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01569](https://doi.org/10.1109/CVPR52729.2023.01569)

        **Abstract**:

        Self-supervised learning in computer vision trains on unlabeled data, such as images or (image, text) pairs, to obtain an image encoder that learns high-quality embeddings for input data. Emerging backdoor attacks towards encoders expose crucial vulnerabilities of self-supervised learning, since downstream classifiers (even further trained on clean data) may inherit backdoor behaviors from en-coders. Existing backdoor detection methods mainly focus on supervised learning settings and cannot handle pre-trained encoders especially when input labels are not available. In this paper, we propose DECREE, the first back-door detection approach for pre-trained encoders, requiring neither classifier headers nor input labels. We evaluate DECREE on over 400 encoders trojaned under 3 paradigms. We show the effectiveness of our method on image encoders pre-trained on ImageNet and OpenAI's CLIP 400 million image-text pairs. Our method consistently has a high detection accuracy even if we have only limited or no access to the pre-training dataset. Code is available at https://github.com/GiantSeaweed/DECREE.

        ----

        ## [1558] Detecting Backdoors During the Inference Stage Based on Corruption Robustness Consistency

        **Authors**: *Xiaogeng Liu, Minghui Li, Haoyu Wang, Shengshan Hu, Dengpan Ye, Hai Jin, Libing Wu, Chaowei Xiao*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01570](https://doi.org/10.1109/CVPR52729.2023.01570)

        **Abstract**:

        Deep neural networks are proven to be vulnerable to backdoor attacks. Detecting the trigger samples during the inference stage, i.e., the test-time trigger sample detection, can prevent the backdoor from being triggered. However, existing detection methods often require the defenders to have high accessibility to victim models, extra clean data, or knowledge about the appearance of backdoor triggers, limiting their practicality. In this paper, we propose the test-time corruption robustness consistency evaluation (TeCo)
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
https://github.com/CGCL-codes/TeCo, a novel test-time trigger sample detection method that only needs the hard-label outputs of the victim models without any extra information. Our journey begins with the intriguing observation that the backdoor-infected models have similar performance across different image corruptions for the clean images, but perform discrepantly for the trigger samples. Based on this phenomenon, we design TeCo to evaluate test-time robustness consistency by calculating the deviation of severity that leads to predictions' transition across different corruptions. Extensive experiments demonstrate that compared with state-of-the-art defenses, which even require either certain information about the trigger types or accessibility of clean data, TeCo outperforms them on different backdoor attacks, datasets, and model architectures, enjoying a higher AUROC by 10% and 5 times of stability.

        ----

        ## [1559] Can't Steal? Cont-Steal! Contrastive Stealing Attacks Against Image Encoders

        **Authors**: *Zeyang Sha, Xinlei He, Ning Yu, Michael Backes, Yang Zhang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01571](https://doi.org/10.1109/CVPR52729.2023.01571)

        **Abstract**:

        Self-supervised representation learning techniques have been developing rapidly to make full use of unlabeled images. They encode images into rich features that are oblivious to downstream tasks. Behind their revolutionary representation power, the requirements for dedicated model designs and a massive amount of computation resources expose image encoders to the risks of potential model stealing attacks - a cheap way to mimic the well-trained encoder performance while circumventing the demanding requirements. Yet conventional attacks only target supervised classifiers given their predicted labels and/or posteriors, which leaves the vulnerability of unsupervised encoders unexplored. In this paper, we first instantiate the conventional stealing attacks against encoders and demonstrate their severer vulnerability compared with downstream classifiers. To better leverage the rich representation of encoders, we further propose Cont-Steal, a contrastive-learning-based attack, and validate its improved stealing effectiveness in various experiment settings. As a takeaway, we appeal to our community's attention to the intellectual property protection of representation learning techniques, especially to the defenses against encoder stealing attacks like ours.
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
See our code in https://github.com/zeyangsha/Cont-Steal.

        ----

        ## [1560] Re-Thinking Model Inversion Attacks Against Deep Neural Networks

        **Authors**: *Ngoc-Bao Nguyen, Keshigeyan Chandrasegaran, Milad Abdollahzadeh, Ngai-Man Cheung*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01572](https://doi.org/10.1109/CVPR52729.2023.01572)

        **Abstract**:

        Model inversion (MI) attacks aim to infer and reconstruct private training data by abusing access to a model. MI attacks have raised concerns about the leaking of sen-sitive information (e.g. private face images used in training a face recognition system). Recently, several algorithms for MI have been proposed to improve the attack performance. In this work, we revisit MI, study two fundamental issues pertaining to all state-of-the-art (SOTA) MI algorithms, and propose solutions to these issues which lead to a significant boost in attack performance for all SOTA MI. In particular, our contributions are two-fold: 1) We ana-lyze the optimization objective of SOTA MI algorithms, ar-gue that the objective is sub-optimal for achieving MI, and propose an improved optimization objective that boosts attack performance significantly. 2) We analyze “MI overfitting”, show that it would prevent reconstructed images from learning semantics of training data, and propose a novel “model augmentation” idea to overcome this issue. Our proposed solutions are simple and improve all SOTA MI attack accuracy significantly. E.g., in the standard CelebA benchmark, our solutions improve accuracy by 11.8% and achieve for the first time over 90% attack accuracy. Our findings demonstrate that there is a clear risk of leaking sensitive information from deep learning models. We urge serious consideration to be given to the privacy im-plications. Our code, demo, and models are available at https://ngoc-nguyen-0.github.io/re-thinking_mode1_inversion_attacks/.

        ----

        ## [1561] Turning Strengths into Weaknesses: A Certified Robustness Inspired Attack Framework against Graph Neural Networks

        **Authors**: *Binghui Wang, Meng Pang, Yun Dong*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01573](https://doi.org/10.1109/CVPR52729.2023.01573)

        **Abstract**:

        Graph neural networks (GNNs) have achieved state-of-the-art performance in many graph learning tasks. However, recent studies show that GNNs are vulnerable to both test-time evasion and training-time poisoning attacks that perturb the graph structure. While existing attack methods have shown promising attack performance, we would like to design an attack framework to further enhance the performance. In particular, our attack framework is inspired by certified robustness, which was originally used by defenders to defend against adversarial attacks. We are the first, from the attacker perspective, to leverage its properties to better attack GNNs. Specifically, we first derive nodes' certified perturbation sizes against graph evasion and poisoning attacks based on randomized smoothing, respectively. A larger certified perturbation size of a node indicates this node is theoretically more robust to graph perturbations. Such a property motivates us to focus more on nodes with smaller certified perturbation sizes, as they are easier to be attacked after graph perturbations. Accordingly, we design a certified robustness inspired attack loss, when incorporated into (any) existing attacks, produces our certified robustness inspired attack counterpart. We apply our frame-work to the existing attacks and results show it can significantly enhance the existing base attacks' performance.

        ----

        ## [1562] Dynamic Generative Targeted Attacks with Pattern Injection

        **Authors**: *Weiwei Feng, Nanqing Xu, Tianzhu Zhang, Yongdong Zhang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01574](https://doi.org/10.1109/CVPR52729.2023.01574)

        **Abstract**:

        Adversarial attacks can evaluate model robustness and have been of great concern in recent years. Among various attacks, targeted attacks aim at misleading victim models to output adversary-desired predictions, which are more challenging and threatening than untargeted ones. Existing targeted attacks can be roughly divided into instance-specific and instance-agnostic attacks. Instance-specific attacks craft adversarial examples via iterative gradient updating on the specific instance. In contrast, instance-agnostic attacks learn a universal perturbation or a generative model on the global dataset to perform attacks. However, they rely too much on the classification boundary of substitute models, ignoring the realistic distribution of the target class, which may result in limited targeted attack performance. And there is no attempt to simultaneously combine the information of the specific instance and the global dataset. To deal with these limitations, we first conduct an analysis via a causal graph and propose to craft transferable targeted adversarial examples by injecting target patterns. Based on this analysis, we introduce a generative attack model composed of a cross-attention guided convolution module and a pattern injection module. Concretely, the former adopts a dynamic convolution kernel and a static convolution kernel for the specific instance and the global dataset, respectively, which can inherit the advantages of both instance-specific and instance-agnostic attacks. And the pattern injection module utilizes a pattern prototype to encode target patterns, which can guide the generation of targeted adversa rial examples. Besides, we also provide rigorous theoretical analysis to guarantee the effectiveness of our method. Extensive experiments demonstrate that our method shows superior performance than 10 existing adversarial attacks against 13 models.

        ----

        ## [1563] Transferable Adversarial Attacks on Vision Transformers with Token Gradient Regularization

        **Authors**: *Jianping Zhang, Yizhan Huang, Weibin Wu, Michael R. Lyu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01575](https://doi.org/10.1109/CVPR52729.2023.01575)

        **Abstract**:

        Vision transformers (ViTs) have been successfully deployed in a variety of computer vision tasks, but they are still vulnerable to adversarial samples. Transfer-based attacks use a local model to generate adversarial samples and directly transfer them to attack a target black-box model. The high efficiency of transfer-based attacks makes it a severe security threat to ViT-based applications. Therefore, it is vital to design effective transfer-based attacks to identify the deficiencies of ViTs beforehand in security-sensitive scenarios. Existing efforts generally focus on regularizing the input gradients to stabilize the updated direction of adversarial samples. However, the variance of the back-propagated gradients in intermediate blocks of ViTs may still be large, which may make the generated adversarial samples focus on some model-specific features and get stuck in poor local optima. To overcome the shortcomings of existing approaches, we propose the Token Gradient Regularization (TGR) method. According to the structural characteristics of ViTs, TGR reduces the variance of the back-propagated gradient in each internal block of ViTs in a token-wise manner and utilizes the regularized gradient to generate adversarial samples. Extensive experiments on attacking both ViTs and CNNs confirm the superiority of our approach. Notably, compared to the state-of-the-art transfer-based at-tacks, our TGR offers a performance improvement of 8.8% on average.

        ----

        ## [1564] Adversarial Counterfactual Visual Explanations

        **Authors**: *Guillaume Jeanneret, Loïc Simon, Frédéric Jurie*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01576](https://doi.org/10.1109/CVPR52729.2023.01576)

        **Abstract**:

        Counterfactual explanations and adversarial attacks have a related goal: flipping output labels with minimal perturbations regardless of their characteristics. Yet, adversarial attacks cannot be used directly in a counterfactual explanation perspective, as such perturbations are perceived as noise and not as actionable and understandable image modifications. Building on the robust learning literature, this paper proposes an elegant method to turn adversarial attacks into semantically meaningful perturbations, without modifying the classifiers to explain. The proposed approach hypothesizes that Denoising Diffusion Probabilistic Models are excellent regularizers for avoiding high-frequency and out-of-distribution perturbations when generating adversarial attacks. The paper's key idea is to build attacks through a diffusion model to polish them. This allows studying the target model regardless of its robustification level. Extensive experimentation shows the advantages of our counterfactual explanation approach over current State-of-the-Art in multiple testbeds.

        ----

        ## [1565] TWINS: A Fine-Tuning Framework for Improved Transferability of Adversarial Robustness and Generalization

        **Authors**: *Ziquan Liu, Yi Xu, Xiangyang Ji, Antoni B. Chan*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01577](https://doi.org/10.1109/CVPR52729.2023.01577)

        **Abstract**:

        Recent years have seen the ever-increasing importance of pre-trained models and their downstream training in deep learning research and applications. At the same time, the defense for adversarial examples has been mainly inves-tigated in the context of training from random initialization on simple classification tasks. To better exploit the potential of pre-trained models in adversarial robustness, this paper focuses on the fine-tuning of an adversarially pre-trained model in various classification tasks. Existing research has shown that since the robust pre-trained model has already learned a robust feature extractor, the crucial question is how to maintain the robustness in the pre-trained model when learning the downstream task. We study the model-based and data-based approaches for this goal and find that the two common approaches cannot achieve the objective of improving both generalization and adversarial robustness. Thus, we propose a novel statistics-based approach, Two-WIng NormliSation (TWINS)fine-tuning framework, which consists of two neural networks where one of them keeps the population means and variances of pre-training data in the batch normalization layers. Besides the robust information transfer, TWINS increases the effective learning rate without hurting the training stability since the relationship between a weight norm and its gradient norm in standard batch normalization layer is broken, resulting in a faster es-cape from the sub-optimal initialization and alleviating the robust overfitting. Finally, TWINS is shown to be effective on a wide range of image classification datasets in terms of both generalization and robustness.

        ----

        ## [1566] Randomized Adversarial Training via Taylor Expansion

        **Authors**: *Gaojie Jin, Xinping Yi, Dengyu Wu, Ronghui Mu, Xiaowei Huang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01578](https://doi.org/10.1109/CVPR52729.2023.01578)

        **Abstract**:

        In recent years, there has been an explosion of research into developing more robust deep neural networks against adversarial examples. Adversarial training appears as one of the most successful methods. To deal with both the robustness against adversarial examples and the accuracy over clean examples, many works develop enhanced adversarial training methods to achieve various trade-offs between them [[19], [38], [80]], Leveraging over the studies [8], [32] that smoothed update on weights during training may help find flat minima and improve generalization, we suggest reconciling the robustness-accuracy trade-off from another perspective, i.e., by adding random noise into deterministic weights. The randomized weights enable our design of a novel adversarial training method via Taylor expansion of a small Gaussian noise, and we show that the new adversarial training method can flatten loss landscape and find flat minima. With PGD, CW, and Auto Attacks, an extensive set of experiments demonstrate that our method enhances the state-of-the-art adversarial training methods, boosting both robustness and clean accuracy. The code is available at https://github.com/Alexkael/Randomized-Adversarial-Training.

        ----

        ## [1567] Improving Robust Generalization by Direct PAC-Bayesian Bound Minimization

        **Authors**: *Zifan Wang, Nan Ding, Tomer Levinboim, Xi Chen, Radu Soricut*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01579](https://doi.org/10.1109/CVPR52729.2023.01579)

        **Abstract**:

        Recent research in robust optimization has shown an overfitting-like phenomenon in which models trained against adversarial attacks exhibit higher robustness on the training set compared to the test set. Although previous work provided theoretical explanations for this phenomenon using a robust PAC-Bayesian bound over the adversarial test error, related algorithmic derivations are at best only loosely connected to this bound, which implies that there is still a gap between their empirical success and our understanding of adversarial robustness theory. To close this gap, in this paper we consider a different form of the robust PAC-Bayesian bound and directly minimize it with respect to the model posterior. The derivation of the optimal solution connects PAC-Bayesian learning to the geometry of the robust loss surface through a Trace of Hessian (TrH) regularizer that measures the surface flatness. In practice, we restrict the TrH regularizer to the top layer only, which results in an analytical solution to the bound whose computational cost does not depend on the network depth. Finally, we evaluate our TrH regularization approach over CIFAR-10/100 and ImageNet using Vision Transformers (ViT) and compare against baseline adversarial robustness algorithms. Experimental results show that TrH regularization leads to improved ViT robustness that either matches or surpasses previous state-of-the-art approaches while at the same time requires less memory and computational cost.

        ----

        ## [1568] Evading Forensic Classifiers with Attribute-Conditioned Adversarial Faces

        **Authors**: *Fahad Shamshad, Koushik Srivatsan, Karthik Nandakumar*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01580](https://doi.org/10.1109/CVPR52729.2023.01580)

        **Abstract**:

        The ability of generative models to produce highly realistic synthetic face images has raised security and ethical concerns. As a first line of defense against such fake faces, deep learning based forensic classifiers have been developed. While these forensic models can detect whether a face image is synthetic or real with high accuracy, they are also vulnerable to adversarial attacks. Although such attacks can be highly successful in evading detection by forensic classifiers, they introduce visible noise patterns that are detectable through careful human scrutiny. Additionally, these attacks assume access to the target model(s) which may not always be true. Attempts have been made to directly perturb the latent space of GANs to produce adversarial fake faces that can circumvent forensic classifiers. In this work, we go one step further and show that it is possible to successfully generate adversarial fake faces with a specified set of attributes (e.g., hair color, eye size, race, gender, etc.). To achieve this goal, we leverage the state-of-the-art generative model StyleGAN with disentangled representations, which enables a range of modifications without leaving the manifold of natural images. We propose a framework to search for adversarial latent codes within the feature space of StyleGAN, where the search can be guided either by a text prompt or a reference image. We also propose a meta-learning based optimization strategy to achieve transferable performance on unknown target models. Extensive experiments demonstrate that the proposed approach can produce semantically manipulated adversarial fake faces, which are true to the specified attribute set and can successfully fool forensic face classifiers, while remaining undetectable by humans. Code: https://github.com/koushiksrivats/face_attribute_attack.

        ----

        ## [1569] DartBlur: Privacy Preservation with Detection Artifact Suppression

        **Authors**: *Baowei Jiang, Bing Bai, Haozhe Lin, Yu Wang, Yuchen Guo, Lu Fang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01581](https://doi.org/10.1109/CVPR52729.2023.01581)

        **Abstract**:

        Nowadays, privacy issue has become a top priority when training AI algorithms. Machine learning algorithms are expected to benefit our daily life, while personal information must also be carefully protected from exposure. Facial information is particularly sensitive in this regard. Multiple datasets containing facial information have been taken offline, and the community is actively seeking solutions to remedy the privacy issues. Existing methods for privacy preservation can be divided into blur-based and face replacement-based methods. Owing to the advantages of review convenience and good accessibility, blur-based based methods have become a dominant choice in practice. However, blur-based methods would inevitably introduce training artifacts harmful to the performance of downstream tasks. In this paper, we propose a novel De-artifact Blurring (DartBlur) privacy-preserving method, which capitalizes on a DNN architecture to generate blurred faces. DartBlur can effectively hide facial privacy information while detection artifacts are simultaneously suppressed. We have designed four training objectives that particularly aim to improve review convenience and maximize detection artifact suppression. We associate the algorithm with an adversarial training strategy with a second-order optimization pipeline. Experimental results demonstrate that DartBlur outperforms the existing face-replacement method from both perspectives of review convenience and accessibility, and also shows an exclusive advantage in suppressing the training artifact compared to traditional blur-based methods. Our implementation is available at https://github.com/JaNg2333/DartBlur.

        ----

        ## [1570] Fresnel Microfacet BRDF: Unification of Polari-Radiometric Surface-Body Reflection

        **Authors**: *Tomoki Ichikawa, Yoshiki Fukao, Shohei Nobuhara, Ko Nishino*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01582](https://doi.org/10.1109/CVPR52729.2023.01582)

        **Abstract**:

        Computer vision applications have heavily relied on the linear combination of Lambertian diffuse and microfacet specular reflection models for representing reflected radiance, which turns out to be physically incompatible and limited in applicability. In this paper, we derive a novel analytical reflectance model, which we refer to as Fresnel Microfacet BRDF model, that is physically accurate and generalizes to various real-world surfaces. Our key idea is to model the Fresnel reflection and transmission of the surface microgeometry with a collection of oriented mirror facets, both for body and surface reflections. We carefully derive the Fresnel reflection and transmission for each microfacet as well as the light transport between them in the subsurface. This physically-grounded modeling also allows us to express the polarimetric behavior of reflected light in addition to its radiometric behavior. That is, FMBRDF unifies not only body and surface reflections but also light reflection in radiometry and polarization and represents them in a single model. Experimental results demonstrate its effectiveness in accuracy, expressive power, image-based estimation, and geometry recovery.

        ----

        ## [1571] JacobiNeRF: NeRF Shaping with Mutual Information Gradients

        **Authors**: *Xiaomeng Xu, Yanchao Yang, Kaichun Mo, Boxiao Pan, Li Yi, Leonidas J. Guibas*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01583](https://doi.org/10.1109/CVPR52729.2023.01583)

        **Abstract**:

        We propose a method that trains a neural radiance field (NeRF) to encode not only the appearance of the scene but also semantic correlations between scene points, regions, or entities - aiming to capture their mutual co-variation patterns. In contrast to the traditional first-order photometric reconstruction objective, our method explicitly regularizes the learning dynamics to align the Jacobians of highly-correlated entities, which proves to maximize the mutual information between them under random scene perturbations. By paying attention to this second-order information, we can shape a NeRF to express semantically meaningful synergies when the network weights are changed by a delta along the gradient of a single entity, region, or even a point. To demonstrate the merit of this mutual information modeling, we leverage the coordinated behavior of scene entities that emerges from our shaping to perform label propagation for semantic and instance segmentation. Our experiments show that a JacobiNeRF is more efficient in propagating annotations among 2D pixels and 3D points compared to NeRFs without mutual information shaping, especially in extremely sparse label regimes - thus reducing annotation burden. The same machinery can further be used for entity selection or scene modifications. Our code is available at https://github.com/xxm19/jacobinerf.

        ----

        ## [1572] ContraNeRF: Generalizable Neural Radiance Fields for Synthetic-to-real Novel View Synthesis via Contrastive Learning

        **Authors**: *Hao Yang, Lanqing Hong, Aoxue Li, Tianyang Hu, Zhenguo Li, Gim Hee Lee, Liwei Wang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01584](https://doi.org/10.1109/CVPR52729.2023.01584)

        **Abstract**:

        Although many recent works have investigated generalizable NeRF-based novel view synthesis for unseen scenes, they seldom consider the synthetic-to-real generalization, which is desired in many practical applications. In this work, we first investigate the effects of synthetic data in synthetic-to-real novel view synthesis and surprisingly observe that models trained with synthetic data tend to produce sharper but less accurate volume densities. For pixels where the volume densities are correct, fine-grained details will be obtained. Otherwise, severe artifacts will be produced. To maintain the advantages of using synthetic data while avoiding its negative effects, we propose to introduce geometry-aware contrastive learning to learn multi-view consistent features with geometric constraints. Meanwhile, we adopt cross-view attention to further enhance the geometry perception of features by querying features across input views. Experiments demonstrate that under the synthetic-to-real setting, our method can render images with higher quality and better fine-grained details, outperforming existing generalizable novel view synthesis methods in terms of PSNR, SSIM, and LPIPS. When trained on real data, our method also achieves state-of-the-art results. https://haoy945.github.io/contranerf/

        ----

        ## [1573] SCADE: NeRFs from Space Carving with Ambiguity-Aware Depth Estimates

        **Authors**: *Mikaela Angelina Uy, Ricardo Martin-Brualla, Leonidas J. Guibas, Ke Li*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01585](https://doi.org/10.1109/CVPR52729.2023.01585)

        **Abstract**:

        Neural radiance fields (NeRFs) have enabled high fidelity 3D reconstruction from multiple 2D input views. However, a well-known drawback of NeRFs is the less-than-ideal performance under a small number of views, due to insufficient constraints enforced by volumetric rendering. To address this issue, we introduce SCADE, a novel technique that improves NeRF reconstruction quality on sparse, unconstrained input views for in-the-wild indoor scenes. To constrain NeRF reconstruction, we leverage geometric priors in the form of per-view depth estimates produced with state-of-the-art monocular depth estimation models, which can generalize across scenes. A key challenge is that monocular depth estimation is an illposed problem, with inherent ambiguities. To handle this issue, we propose a new method that learns to predict, for each view, a continuous, multimodal distribution of depth estimates using conditional Implicit Maximum Likelihood Estimation (cIMLE). In order to disambiguate exploiting multiple views, we introduce an original space carving loss that guides the NeRF representation to fuse multiple hypothesized depth maps from each view and distill from them a common geometry that is consistent with all views. Experiments show that our approach enables higher fidelity novel view synthesis from sparse views. Our project page can be found at scade-spacecarving-nerfs.github.io.

        ----

        ## [1574] Removing Objects From Neural Radiance Fields

        **Authors**: *Silvan Weder, Guillermo Garcia-Hernando, Áron Monszpart, Marc Pollefeys, Gabriel J. Brostow, Michael Firman, Sara Vicente*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01586](https://doi.org/10.1109/CVPR52729.2023.01586)

        **Abstract**:

        Neural Radiance Fields (NeRFs) are emerging as a ubiquitous scene representation that allows for novel view synthesis. Increasingly, NeRFs will be shareable with other people. Before sharing a NeRF, though, it might be desirable to remove personal information or unsightly objects. Such removal is not easily achieved with the current NeRF editing frameworks. We propose a framework to remove objects from a NeRF representation created from an RGBD sequence. Our NeRF inpainting method leverages recent work in 2D image inpainting and is guided by a userprovided mask. Our algorithm is underpinned by a confidence based view selection procedure. It chooses which of the individual 2D inpainted images to use in the creation of the NeRF, so that the resulting inpainted NeRF is 3D consistent. We show that our method for NeRF editing is effective for synthesizing plausible inpaintings in a multi-view coherent manner, outperforming competing methods. We validate our approach by proposing a new and still-challenging dataset for the task of NeRF inpainting.

        ----

        ## [1575] Progressively Optimized Local Radiance Fields for Robust View Synthesis

        **Authors**: *Andreas Meuleman, Yu-Lun Liu, Chen Gao, Jia-Bin Huang, Changil Kim, Min H. Kim, Johannes Kopf*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01587](https://doi.org/10.1109/CVPR52729.2023.01587)

        **Abstract**:

        We present an algorithm for reconstructing the radiance field of a large-scale scene from a single casually captured video. The task poses two core challenges. First, most existing radiance field reconstruction approaches rely on accurate pre-estimated camera poses from Structure-from-Motion algorithms, which frequently fail on in-the-wild videos. Second, using a single, global radiance field with finite representational capacity does not scale to longer trajectories in an unbounded scene. For handling unknown poses, we jointly estimate the camera poses with radiance field in a progressive manner. We show that progressive optimization significantly improves the robustness of the reconstruction. For handling large unbounded scenes, we dynamically allocate new local radiance fields trained with frames within a temporal window. This further improves robustness (e.g., performs well even under moderate pose drifts) and allows us to scale to large scenes. Our extensive evaluation on the TANKS AND TEMPLES dataset and our collected outdoor dataset, STATIC HIKES, show that our approach compares favorably with the state-of-the-art.

        ----

        ## [1576] NeRFVS: Neural Radiance Fields for Free View Synthesis via Geometry Scaffolds

        **Authors**: *Chen Yang, Peihao Li, Zanwei Zhou, Shanxin Yuan, Bingbing Liu, Xiaokang Yang, Weichao Qiu, Wei Shen*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01588](https://doi.org/10.1109/CVPR52729.2023.01588)

        **Abstract**:

        We present NeRFVS, a novel neural radiance fields (NeRF) based method to enable free navigation in a room. NeRF achieves impressive performance in rendering images for novel views similar to the input views while suffering for novel views that are significantly different from the training views. To address this issue, we utilize the holistic priors, including pseudo depth maps and view coverage information, from neural reconstruction to guide the learning of implicit neural representations of 3D indoor scenes. Concretely, an off-the-shelf neural reconstruction method is leveraged to generate a geometry scaffold. Then, two loss functions based on the holistic priors are proposed to improve the learning of NeRF: 1) A robust depth loss that can tolerate the error of the pseudo depth map to guide the geometry learning of NeRF; 2) A variance loss to regularize the variance of implicit neural representations to reduce the geometry and color ambiguity in the learning procedure. These two loss functions are modulated during NeRF optimization according to the view coverage information to reduce the negative influence brought by the view coverage imbalance. Extensive results demonstrate that our NeRFVS outperforms state-of-the-art view synthesis methods quantitatively and qualitatively on indoor scenes, achieving high-fidelity free navigation results.

        ----

        ## [1577] ABLE-NeRF: Attention-Based Rendering with Learnable Embeddings for Neural Radiance Field

        **Authors**: *Zhe Jun Tang, Tat-Jen Cham, Haiyu Zhao*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01589](https://doi.org/10.1109/CVPR52729.2023.01589)

        **Abstract**:

        Neural Radiance Field (NeRF) is a popular method in representing 3D scenes by optimising a continuous volumetric scene function. Its large success which lies in applying volumetric rendering (VR) is also its Achilles' heel in producing view-dependent effects. As a consequence, glossy and transparent surfaces often appear murky. A remedy to reduce these artefacts is to constrain this VR equation by excluding volumes with back-facing normal. While this approach has some success in rendering glossy surfaces, translucent objects are still poorly represented. In this paper, we present an alternative to the physics-based VR approach by introducing a self-attention-based framework on volumes along a ray. In addition, inspired by modern game engines which utilise Light Probes to store local lighting passing through the scene, we incorporate Learnable Embeddings to capture view dependent effects within the scene. Our method, which we call ABLE-NeRF, significantly reduces ‘blurry’ glossy surfaces in rendering and produces realistic translucent surfaces which lack in prior art. In the Blender dataset, ABLE-NeRF achieves SOTA results and surpasses Ref-NeRF in all 3 image quality metrics PSNR, SSIM, LPIPS.

        ----

        ## [1578] MobileNeRF: Exploiting the Polygon Rasterization Pipeline for Efficient Neural Field Rendering on Mobile Architectures

        **Authors**: *Zhiqin Chen, Thomas A. Funkhouser, Peter Hedman, Andrea Tagliasacchi*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01590](https://doi.org/10.1109/CVPR52729.2023.01590)

        **Abstract**:

        Neural Radiance Fields (NeRFs) have demonstrated amazing ability to synthesize images of 3D scenes from novel views. However, they rely upon specialized volumetric rendering algorithms based on ray marching that are mismatched to the capabilities of widely deployed graphics hardware. This paper introduces a new NeRF representation based on textured polygons that can synthesize novel images efficiently with standard rendering pipelines. The NeRF is represented as a set of polygons with textures representing binary opacities and feature vectors. Traditional rendering of the polygons with a z-buffer yields an image with features at every pixel, which are interpreted by a small, view-dependent MLP running in a fragment shader to produce a final pixel color. This approach enables NeRFs to be rendered with the traditional polygon rasterization pipeline, which provides massive pixel-level parallelism, achieving interactive frame rates on a wide range of compute platforms, including mobile phones. Project page: https://mobile-nerf.github.io

        ----

        ## [1579] pCON: Polarimetric Coordinate Networks for Neural Scene Representations

        **Authors**: *Henry Peters, Yunhao Ba, Achuta Kadambi*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01591](https://doi.org/10.1109/CVPR52729.2023.01591)

        **Abstract**:

        Neural scene representations have achieved great success in parameterizing and reconstructing images, but current state of the art models are not optimized with the preservation of physical quantities in mind. While current architectures can reconstruct color images correctly, they create artifacts when trying to fit maps of polar quantities. We propose polarimetric coordinate networks (pCON), a new model architecture for neural scene representations aimed at preserving polarimetric information while accurately parameterizing the scene. Our model removes artifacts created by current coordinate network architectures when reconstructing three polarimetric quantities of interest. All code and data can be found at this link: http5://visual.ee.ucla.edu/pcon.htm.

        ----

        ## [1580] Complementary Intrinsics from Neural Radiance Fields and CNNs for Outdoor Scene Relighting

        **Authors**: *Siqi Yang, Xuanning Cui, Yongjie Zhu, Jiajun Tang, Si Li, Zhaofei Yu, Boxin Shi*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01593](https://doi.org/10.1109/CVPR52729.2023.01593)

        **Abstract**:

        Relighting an outdoor scene is challenging due to the diverse illuminations and salient cast shadows. Intrinsic image decomposition on outdoor photo collections could partly solve this problem by weakly supervised labels with albedo and normal consistency from multiview stereo. With neural radiance fields (NeRF), editing the appearance code could produce more realistic results without interpreting the outdoor scene image formation explicitly. This paper proposes to complement the intrinsic estimation from volume rendering using NeRF and from inversing the photometric image formation model using convolutional neural networks (CNNs). The former produces richer and more reliable pseudo labels (cast shadows and sky appearances in addition to albedo and normal) for training the latter to predict interpretable and editable lighting parameters via a single-image prediction pipeline. We demonstrate the advantages of our method for both intrinsic image decomposition and relighting for various real outdoor scenes.

        ----

        ## [1581] HyperReel: High-Fidelity 6-DoF Video with Ray-Conditioned Sampling

        **Authors**: *Benjamin Attal, Jia-Bin Huang, Christian Richardt, Michael Zollhöfer, Johannes Kopf, Matthew O'Toole, Changil Kim*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01594](https://doi.org/10.1109/CVPR52729.2023.01594)

        **Abstract**:

        Volumetric scene representations enable photorealistic view synthesis for static scenes and form the basis of several existing 6-DoF video techniques. However, the volume rendering procedures that drive these representations necessitate careful trade-offs in terms of quality, rendering speed, and memory efficiency. In particular, existing methods fail to simultaneously achieve real-time performance, small memory footprint, and high-quality rendering for challenging real-world scenes. To address these issues, we present HyperReel―a novel 6-DoF video representation. The two core components of HyperReel are: (1) a ray-conditioned sample prediction network that enables high-fidelity, high frame rate rendering at high resolutions and (2) a compact and memory-efficient dynamic volume representation. Our 6-DoF video pipeline achieves the best performance compared to prior and contemporary approaches in terms of visual quality with small memory requirements, while also rendering at up to 18 frames-per-second at megapixel resolution without any custom CUDA code.

        ----

        ## [1582] UV Volumes for Real-time Rendering of Editable Free-view Human Performance

        **Authors**: *Yue Chen, Xuan Wang, Xingyu Chen, Qi Zhang, Xiaoyu Li, Yu Guo, Jue Wang, Fei Wang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01595](https://doi.org/10.1109/CVPR52729.2023.01595)

        **Abstract**:

        Neural volume rendering enables photo-realistic renderings of a human performer in free-view, a critical task in immersive VR/AR applications. But the practice is severely limited by high computational costs in the rendering process. To solve this problem, we propose the UV Volumes, a new approach that can render an editable free-view video of a human performer in real-time. It separates the high-frequency (i.e., non-smooth) human appearance from the 3D volume, and encodes them into 2D neural texture stacks (NTS). The smooth UV volumes allow much smaller and shallower neural networks to obtain densities and texture coordinates in 3D while capturing detailed appearance in 2D NTS. For editability, the mapping between the parameterized human model and the smooth texture coordinates allows us a better generalization on novel poses and shapes. Furthermore, the use of NTS enables interesting applications, e.g., retexturing. Extensive experiments on CMU Panoptic, ZJU Mocap, and H36M datasets show that our model can render 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$960\times 540$</tex>
 images in 30FPS on average with comparable photo-realism to state-of-the-art methods. The project and supplementary materials are available at https://fanegg.github.io/UV-Volumes.

        ----

        ## [1583] Tensor4D: Efficient Neural 4D Decomposition for High-Fidelity Dynamic Reconstruction and Rendering

        **Authors**: *Ruizhi Shao, Zerong Zheng, Hanzhang Tu, Boning Liu, Hongwen Zhang, Yebin Liu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01596](https://doi.org/10.1109/CVPR52729.2023.01596)

        **Abstract**:

        We present Tensor4D, an efficient yet effective approach to dynamic scene modeling. The key of our solution is an efficient 4D tensor decomposition method so that the dynamic scene can be directly represented as a 4D spatio-temporal tensor. To tackle the accompanying memory issue, we decompose the 4D tensor hierarchically by projecting it first into three time-aware volumes and then nine compact feature planes. In this way, spatial information over time can be simultaneously captured in a compact and memory-efficient manner. When applying Tensor4D for dynamic scene reconstruction and rendering, we further factorize the 4D fields to different scales in the sense that structural motions and dynamic detailed changes can be learned from coarse to fine. The effectiveness of our method is validated on both synthetic and real-world scenes. Extensive experiments show that our method is able to achieve high-quality dynamic reconstruction and rendering from sparse-view camera rigs or even a monocular camera. The code and dataset will be released at https://github.com/DSaurus/Tensor4D.

        ----

        ## [1584] PixHt-Lab: Pixel Height Based Light Effect Generation for Image Compositing

        **Authors**: *Yichen Sheng, Jianming Zhang, Julien Philip, Yannick Hold-Geoffroy, Xin Sun, He Zhang, Lu Ling, Bedrich Benes*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01597](https://doi.org/10.1109/CVPR52729.2023.01597)

        **Abstract**:

        Lighting effects such as shadows or reflections are key in making synthetic images realistic and visually appealing. To generate such effects, traditional computer graphics uses a physically-based renderer along with 3D geometry. To compensate for the lack of geometry in 2D Image compositing, recent deep learning-based approaches introduced a pixel height representation to generate soft shadows and reflections. However, the lack of geometry limits the quality of the generated soft shadows and constrains reflections to pure specular ones. We introduce PixHt-Lab, a system leveraging an explicit mapping from pixel height representation to 3D space. Using this mapping, PixHt- Lab reconstructs both the cutout and background geometry and renders realistic, diverse lighting effects for image compositing. Given a surface with physically-based materials, we can render reflections with varying glossiness. To generate more realistic soft shadows, we further propose using 3D-aware buffer channels to guide a neural renderer. Both quantitative and qualitative evaluations demonstrate that PixHt-Lab significantly improves soft shadow generation. Project: https://shengcn.github.io/PixHtLab/

        ----

        ## [1585] Computational Flash Photography through Intrinsics

        **Authors**: *Sepideh Sarajian Maralan, Chris Careaga, Yagiz Aksoy*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01598](https://doi.org/10.1109/CVPR52729.2023.01598)

        **Abstract**:

        Flash is an essential tool as it often serves as the sole controllable light source in everyday photography. However, the use of flash is a binary decision at the time a photograph is captured with limited control over its characteristics such as strength or color. In this work, we study the computational control of the flash light in photographs taken with or without flash. We present a physically moti-vated intrinsic formulation for flash photograph formation and develop flash decomposition and generation methods for flash and no-flash photographs, respectively. We demonstrate that our intrinsic formulation outperforms alternatives in the literature and allows us to computationally control flash in in-the-wild images.

        ----

        ## [1586] RelightableHands: Efficient Neural Relighting of Articulated Hand Models

        **Authors**: *Shun Iwase, Shunsuke Saito, Tomas Simon, Stephen Lombardi, Timur M. Bagautdinov, Rohan Joshi, Fabian Prada, Takaaki Shiratori, Yaser Sheikh, Jason M. Saragih*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01599](https://doi.org/10.1109/CVPR52729.2023.01599)

        **Abstract**:

        We present the first neural relighting approach for rendering high-fidelity personalized hands that can be animated in real-time under novel illumination. Our approach adopts a teacher-student framework, where the teacher learns appearance under a single point light from images captured in a light-stage, allowing us to synthesize hands in arbitrary illuminations but with heavy compute. Using images rendered by the teacher model as training data, an efficient student model directly predicts appearance under natural illuminations in real-time. To achieve generalization, we condition the student model with physics-inspired illumination features such as visibility, diffuse shading, and specular reflections computed on a coarse proxy geometry, maintaining a small computational overhead. Our key insight is that these features have strong correlation with subsequent global light transport effects, which proves sufficient as conditioning data for the neural relighting network. Moreover, in contrast to bottleneck illumination conditioning, these features are spatially aligned based on underlying geometry, leading to better generalization to unseen illuminations and poses. In our experiments, we demonstrate the efficacy of our illumination feature representations, outperforming baseline approaches. We also show that our approach can photorealistically relight two interacting hands at real-time speeds. https://sh8.io/#/relightable_hands

        ----

        ## [1587] TMO: Textured Mesh Acquisition of Objects with a Mobile Device by using Differentiable Rendering

        **Authors**: *Jaehoon Choi, Dongki Jung, Taejae Lee, Sangwook Kim, Youngdong Jung, Dinesh Manocha, Donghwan Lee*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01600](https://doi.org/10.1109/CVPR52729.2023.01600)

        **Abstract**:

        We present a new pipeline for acquiring a textured mesh in the wild with a single smartphone which offers access to images, depth maps, and valid poses. Our method first introduces an RGBD-aided structure from motion, which can yield filtered depth maps and refines camera poses guided by corresponding depth. Then, we adopt the neural implicit surface reconstruction method, which allows for high-quality mesh and develops a new training process for applying a regularization provided by classical multi-view stereo methods. Moreover, we apply a differentiable rendering to fine-tune incomplete texture maps and generate textures which are perceptually closer to the original scene. Our pipeline can be applied to any common objects in the real world without the need for either in-the-lab environments or accurate mask images. We demonstrate results of captured objects with complex shapes and validate our method numerically against existing 3D reconstruction and texture mapping methods.

        ----

        ## [1588] VolRecon: Volume Rendering of Signed Ray Distance Functions for Generalizable Multi-View Reconstruction

        **Authors**: *Yufan Ren, Fangjinhua Wang, Tong Zhang, Marc Pollefeys, Sabine Süsstrunk*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01601](https://doi.org/10.1109/CVPR52729.2023.01601)

        **Abstract**:

        The success of the Neural Radiance Fields (NeRF) in novel view synthesis has inspired researchers to propose neural implicit scene reconstruction. However, most existing neural implicit reconstruction methods optimize perscene parameters and therefore lack generalizability to new scenes. We introduce VolRecon, a novel generalizable implicit reconstruction method with Signed Ray Distance Function (SRDF). To reconstruct the scene with fine details and little noise, VolRecon combines projection features aggregated from multi-view features, and volume features interpolated from a coarse global feature volume. Using a ray transformer, we compute SRDF values of sampled points on a ray and then render color and depth. On DTU dataset, VolRecon outperforms SparseNeuS by about 30% in sparse view reconstruction and achieves comparable accuracy as MVSNet in full view reconstruction. Furthermore, our approach exhibits good generalization performance on the large-scale ETH3D benchmark. Code is available at https://github.com/IVRL/VolRecon/.

        ----

        ## [1589] Multi-View Reconstruction Using Signed Ray Distance Functions (SRDF)

        **Authors**: *Pierre Zins, Yuanlu Xu, Edmond Boyer, Stefanie Wuhrer, Tony Tung*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01602](https://doi.org/10.1109/CVPR52729.2023.01602)

        **Abstract**:

        In this paper, we investigate a new optimization framework for multi-view 3D shape reconstructions. Recent differentiable rendering approaches have provided breakthrough performances with implicit shape representations though they can still lack precision in the estimated geometries. On the other hand multi-view stereo methods can yield pixel wise geometric accuracy with local depth predictions along viewing rays. Our approach bridges the gap between the two strategies with a novel volumetric shape representation that is implicit but parameterized with pixel depths to better materialize the shape surface with consistent signed distances along viewing rays. The approach retains pixel-accuracy while benefiting from volumetric integration in the optimization. To this aim, depths are optimized by evaluating, at each 3D location within the volumetric discretization, the agreement between the depth prediction consistency and the photometric consistency for the corresponding pixels. The optimization is agnostic to the associated photo-consistency term which can vary from a median-based baseline to more elaborate criteria, e.g. learned functions. Our experiments demonstrate the benefit of the volumetric integration with depth predictions. They also show that our approach outperforms existing approaches over standard 3D benchmarks with better geometry estimations.

        ----

        ## [1590] Structural Multiplane Image: Bridging Neural View Synthesis and 3D Reconstruction

        **Authors**: *Mingfang Zhang, Jinglu Wang, Xiao Li, Yifei Huang, Yoichi Sato, Yan Lu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01603](https://doi.org/10.1109/CVPR52729.2023.01603)

        **Abstract**:

        The Multiplane Image (MPI), containing a set of fronto-parallel 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$RGB_{\alpha}$</tex>
 layers, is an effective and efficient representation for view synthesis from sparse inputs. Yet, its fixed structure limits the performance, especially for surfaces imaged at oblique angles. We introduce the Structural MPI (S-MPI), where the plane structure approximates 3D scenes concisely. Conveying 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$RGB_{\alpha}$</tex>
 contexts with geometrically-faithful structures, the S-MPI directly bridges view synthe-sis and 3D reconstruction. It can not only overcome the critical limitations of MPI, i.e., discretization artifacts from sloped surfaces and abuse of redundant layers, and can also acquire planar 3D reconstruction. Despite the intu-ition and demand of applying S-MPI, great challenges are introduced, 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$e.g$</tex>
., high-fidelity approximation for both 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$RGB_{\alpha}$</tex>
 layers and plane poses, multi-view consistency, non-planar regions modeling, and efficient rendering with intersected planes. Accordingly, we propose a transformer-based network based on a segmentation model [4]. It predicts compact and expressive S-MPI layers with their corresponding masks, poses, and 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$RGB_{\alpha}$</tex>
 contexts. Non-planar regions are inclusively handled as a special case in our unified frame-work. Multi-view consistency is ensured by sharing global proxy embeddings, which encode plane-level features cov-ering the complete 3D scenes with aligned coordinates. In-tensive experiments show that our method outperforms both previous state-of-the-art MPI-based view synthesis methods and planar reconstruction methods.

        ----

        ## [1591] Octree Guided Unoriented Surface Reconstruction

        **Authors**: *Chamin Hewa Koneputugodage, Yizhak Ben-Shabat, Stephen Gould*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01604](https://doi.org/10.1109/CVPR52729.2023.01604)

        **Abstract**:

        We address the problem of surface reconstruction from unoriented point clouds. Implicit neural representations (INRs) have become popular for this task, but when information relating to the inside versus outside of a shape is not available (such as shape occupancy, signed distances or surface normal orientation) optimization relies on heuristics and regularizers to recover the surface. These methods can be slow to converge and easily get stuck in local minima. We propose a two-step approach, OG-INR, where we (1) construct a discrete octree and label what is inside and outside (2) optimize for a continuous and high-fidelity shape using an INR that is initially guided by the octree's labelling. To solve for our labelling, we propose an energy function over the discrete structure and provide an efficient move-making algorithm that explores many possible labellings. Furthermore we show that we can easily inject knowledge into the discrete octree, providing a simple way to influence the result from the continuous INR. We evaluate the effectiveness of our approach on two unoriented surface reconstruction datasets and show competitive performance compared to other unoriented, and some oriented, methods. Our results show that the exploration by the move-making algorithm avoids many of the bad local minima reached by purely gradient descent optimized methods (see Figure 1).

        ----

        ## [1592] Neural Vector Fields: Implicit Representation by Explicit Learning

        **Authors**: *Xianghui Yang, Guosheng Lin, Zhenghao Chen, Luping Zhou*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01605](https://doi.org/10.1109/CVPR52729.2023.01605)

        **Abstract**:

        Deep neural networks (DNNs) are widely applied for nowadays 3D surface reconstruction tasks and such methods can be further divided into two categories, which respectively warp templates explicitly by moving vertices or represent 3D surfaces implicitly as signed or unsigned distance functions. Taking advantage of both advanced explicit learning process and powerful representation ability of implicit functions, we propose a novel 3D representation method, Neural Vector Fields (NVF). It not only adopts the explicit learning process to manipulate meshes directly, but also leverages the implicit representation of unsigned distance functions (UDFs) to break the barriers in resolution and topology. Specifically, our method first predicts the displacements from queries towards the surface and models the shapes as Vector Fields. Rather than relying on network differentiation to obtain direction fields as most existing UDF-based methods, the produced vector fields encode the distance and direction fields both and mitigate the ambiguity at “ridge” points, such that the calculation of direction fields is straightforward and differentiation-free. The differentiation-free characteristic enables us to further learn a shape codebook via Vector Quantization, which encodes the cross-object priors, accelerates the training procedure, and boosts model generalization on cross-category reconstruction. The extensive experiments on surface reconstruction benchmarks indicate that our method outperforms those state-of-the-art methods in different evaluation scenarios including watertight vs non-watertight shapes, category-specific vs category-agnostic reconstruction, category-unseen reconstruction, and cross-domain reconstruction. Our code is released at https://github.com/Wi-sc/NVF.

        ----

        ## [1593] DA Wand: Distortion-Aware Selection Using Neural Mesh Parameterization

        **Authors**: *Richard Liu, Noam Aigerman, Vladimir G. Kim, Rana Hanocka*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01606](https://doi.org/10.1109/CVPR52729.2023.01606)

        **Abstract**:

        We present a neural technique for learning to select a local sub-region around a point which can be used for mesh parameterization. The motivation for our framework is driven by interactive workflows used for decaling, texturing, or painting on surfaces. Our key idea is to incorporate segmentation probabilities as weights of a classical parameterization method, implemented as a novel differentiable parameterization layer within a neural network framework. We train a segmentation network to select 3D regions that are parameterized into 2D and penalized by the resulting distortion, giving rise to segmentations which are distortion-aware. Following training, a user can use our system to interactively select a point on the mesh and obtain a large, meaningful region around the selection which induces a low-distortion parameterization. Our code
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
https://github.com/threedle/DA-Wand and project
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">2</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">2</sup>
https://threedle.github.io/DA-Wand/ are publicly available.

        ----

        ## [1594] Diffusion-based Generation, Optimization, and Planning in 3D Scenes

        **Authors**: *Siyuan Huang, Zan Wang, Puhao Li, Baoxiong Jia, Tengyu Liu, Yixin Zhu, Wei Liang, Song-Chun Zhu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01607](https://doi.org/10.1109/CVPR52729.2023.01607)

        **Abstract**:

        We introduce the SceneDiffuser, a conditional generative model for 3D scene understanding. SceneDiffuser provides a unified model for solving scene-conditioned generation, optimization, and planning. In contrast to prior work, SceneDiffuser is intrinsically scene-aware, physics-based, and goal-oriented. With an iterative sampling strategy, SceneDiffuser jointly formulates the scene-aware generation, physics-based optimization, and goal-oriented planning via a diffusion-based denoising process in a fully differentiable fashion. Such a design alleviates the discrepancies among different modules and the posterior collapse of previous scene-conditioned generative models. We evaluate the SceneDiffuser on various 3D scene understanding tasks, including human pose and motion generation, dexterous grasp generation, path planning for 3D navigation, and motion planning for robot arms. The results show significant improvements compared with previous models, demonstrating the tremendous potential of the SceneDiffuser for the broad community of 3D scene understanding.

        ----

        ## [1595] Patch-Based 3D Natural Scene Generation from a Single Example

        **Authors**: *Weiyu Li, Xuelin Chen, Jue Wang, Baoquan Chen*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01608](https://doi.org/10.1109/CVPR52729.2023.01608)

        **Abstract**:

        We target a 3D generative model for general natural scenes that are typically unique and intricate. Lacking the necessary volumes of training data, along with the difficulties of having ad hoc designs in presence of varying scene characteristics, renders existing setups intractable. Inspired by classical patch-based image models, we advocate for synthesizing 3D scenes at the patch level, given a single example. At the core of this work lies important algorithmic designs w.r.t the scene representation and generative patch nearest-neighbor module, that address unique challenges arising from lifting classical 2D patch-based framework to 3D generation. These design choices, on a collective level, contribute to a robust, effective, and efficient model that can generate high-quality general natural scenes with both realistic geometric structure and visual appearance, in large quantities and varieties, as demonstrated upon a variety of exemplar scenes. Data and code can be found at http://wyysf-98.github.io/Sin3DGen.

        ----

        ## [1596] Consistent View Synthesis with Pose-Guided Diffusion Models

        **Authors**: *Hung-Yu Tseng, Qinbo Li, Changil Kim, Suhib Alsisan, Jia-Bin Huang, Johannes Kopf*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01609](https://doi.org/10.1109/CVPR52729.2023.01609)

        **Abstract**:

        Novel view synthesis from a single image has been a cornerstone problem for many Virtual Reality applications that provide immersive experiences. However, most existing techniques can only synthesize novel views within a limited range of camera motion or fail to generate consistent and high-quality novel views under significant camera movement. In this work, we propose a pose-guided diffusion model to generate a consistent long-term video of novel views from a single image. We design an attention layer that uses epipolar lines as constraints to facilitate the association between different viewpoints. Experimental results on synthetic and real-world datasets demonstrate the effectiveness of the proposed diffusion model against state-of-the-art transformer-based and GAN-based approaches. More qualitative results are available at https://poseguided-diffusion.github.io/.

        ----

        ## [1597] Generalized Deep 3D Shape Prior via Part-Discretized Diffusion Process

        **Authors**: *Yuhan Li, Yishun Dou, Xuanhong Chen, Bingbing Ni, Yilin Sun, Yutian Liu, Fuzhen Wang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01610](https://doi.org/10.1109/CVPR52729.2023.01610)

        **Abstract**:

        We develop a generalized 3D shape generation prior model, tailored for multiple 3D tasks including unconditional shape generation, point cloud completion, and cross-modality shape generation, etc. On one hand, to precisely capture local fine detailed shape information, a vector quantized variational autoencoder (VQ-VAE) is utilized to index local geometry from a compactly learned code-book based on a broad set of task training data. On the other hand, a discrete diffusion generator is introduced to model the inherent structural dependencies among different tokens. In the meantime, a multi-frequency fusion module (MFM) is developed to suppress high-frequency shape feature fluctuations, guided by multi-frequency contextual information. The above designs jointly equip our proposed 3D shape prior model with high-fidelity, diverse features as well as the capability of cross-modality alignment, and extensive experiments have demonstrated superior performances on various 3D shape generation tasks.

        ----

        ## [1598] High Fidelity 3D Hand Shape Reconstruction via Scalable Graph Frequency Decomposition

        **Authors**: *Tianyu Luan, Yuanhao Zhai, Jingjing Meng, Zhong Li, Zhang Chen, Yi Xu, Junsong Yuan*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01611](https://doi.org/10.1109/CVPR52729.2023.01611)

        **Abstract**:

        Despite the impressive performance obtained by recent single-image hand modeling techniques, they lack the capability to capture sufficient details of the 3D hand mesh. This deficiency greatly limits their applications when high-fidelity hand modeling is required, e.g., personalized hand modeling. To address this problem, we design a frequency split network to generate 3D hand mesh using different frequency bands in a coarse-to-fine manner. To capture high-frequency personalized details, we transform the 3D mesh into the frequency domain, and propose a novel frequency decomposition loss to supervise each frequency component. By leveraging such a coarse-to-fine scheme, hand details that correspond to the higher frequency domain can be preserved. In addition, the proposed network is scalable, and can stop the inference at any resolution level to accommodate different hardware with varying computational powers. To quantitatively evaluate the performance of our method in terms of recovering personalized shape details, we introduce a new evaluation metric named Mean Signal-to-Noise Ratio (MSNR) to measure the signal-to-noise ratio of each mesh frequency component. Extensive experiments demonstrate that our approach generates fine-grained details for high-fidelity 3D hand reconstruction, and our evaluation metric is more effective for measuring mesh details compared with traditional metrics. The code is available at https://github.com/tyluann/FreqHand.

        ----

        ## [1599] TAPS3D: Text-Guided 3D Textured Shape Generation from Pseudo Supervision

        **Authors**: *Jiacheng Wei, Hao Wang, Jiashi Feng, Guosheng Lin, Kim-Hui Yap*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01612](https://doi.org/10.1109/CVPR52729.2023.01612)

        **Abstract**:

        In this paper, we investigate an open research task of generating controllable 3D textured shapes from the given textual descriptions. Previous works either require ground truth caption labeling or extensive optimization time. To resolve these issues, we present a novel framework, TAPS3D, to train a text-guided 3D shape generator with pseudo captions. Specifically, based on rendered 2D images, we retrieve relevant words from the CLIP vocabulary and construct pseudo captions using templates. Our constructed captions provide high-level semantic supervision for generated 3D shapes. Further, in order to produce fine-grained textures and increase geometry diversity, we propose to adopt low-level image regularization to enable fake-rendered images to align with the real ones. During the inference phase, our proposed model can generate 3D textured shapes from the given text without any additional optimization. We conduct extensive experiments to analyze each of our proposed components and show the efficacy of our framework in generating high-fidelity 3D textured and text-relevant shapes. Code is available at https://github.com/plusmultiply/TAPS3D

        ----

        

[Go to the previous page](CVPR-2023-list07.md)

[Go to the next page](CVPR-2023-list09.md)

[Go to the catalog section](README.md)