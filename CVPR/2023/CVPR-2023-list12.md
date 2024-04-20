## [2200] LAVENDER: Unifying Video-Language Understanding as Masked Language Modeling

        **Authors**: *Linjie Li, Zhe Gan, Kevin Lin, Chung-Ching Lin, Zicheng Liu, Ce Liu, Lijuan Wang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02214](https://doi.org/10.1109/CVPR52729.2023.02214)

        **Abstract**:

        Unified vision-language frameworks have greatly advanced in recent years, most of which adopt an encoder-decoder architecture to unify image-text tasks as sequence-to-sequence generation. However, existing video-language (VidL) models still require task-specific designs in model architecture and training objectives for each task. In this work, we explore a unified VidL framework LAVENDER, where Masked Language Modeling [13] (MLM) is used as the common interface for all pre-training and downstream tasks. Such unification leads to a simplified model architecture, where only a lightweight MLM head, instead of a decoder with much more parameters, is needed on top of the multimodal encoder. Surprisingly, experimental results show that this unified framework achieves competitive performance on 14 VidL benchmarks, covering video question answering, text-to-video retrieval and video captioning. Extensive analyses further demonstrate Lavender can (i) seamlessly support all downstream tasks with just a single set of parameter values when multi-task fine-tuned; (ii) generalize to various downstream tasks with limited training samples; and (iii) enable zero-shot evaluation on video question answering tasks. Code is available at https://github.com/microsoft/LAVENDER.

        ----

        ## [2201] DeCo: Decomposition and Reconstruction for Compositional Temporal Grounding via Coarse-to-Fine Contrastive Ranking

        **Authors**: *Lijin Yang, Quan Kong, Hsuan-Kung Yang, Wadim Kehl, Yoichi Sato, Norimasa Kobori*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02215](https://doi.org/10.1109/CVPR52729.2023.02215)

        **Abstract**:

        Understanding dense action in videos is a fundamental challenge towards the generalization of vision models. Several works show that compositionality is key to achieving generalization by combining known primitive elements, especially for handling novel composited structures. Compositional temporal grounding is the task of localizing dense action by using known words combined in novel ways in the form of novel query sentences for the actual grounding. In recent works, composition is assumed to be learned from pairs of whole videos and language embeddings through large scale self-supervised pre-training. Alternatively, one can process the video and language into word-level primitive elements, and then only learn fine-grained semantic correspondences. Both approaches do not consider the granularity of the compositions, where different query granularity corresponds to different video segments. Therefore, a good compositional representation should be sensitive to different video and query granularity. We propose a method to learn a coarse-to-fine compositional representation by decomposing the original query sentence into different granular levels, and then learning the correct correspondences between the video and recombined queries through a contrastive ranking constraint. Additionally, we run temporal boundary prediction in a coarse-to-fine manner for precise grounding boundary detection. Experiments are performed on two datasets, Charades-CG and ActivityNet-CG, showing the superior compositional generalizability of our approach.

        ----

        ## [2202] CVT-SLR: Contrastive Visual-Textual Transformation for Sign Language Recognition with Variational Alignment

        **Authors**: *Jiangbin Zheng, Yile Wang, Cheng Tan, Siyuan Li, Ge Wang, Jun Xia, Yidong Chen, Stan Z. Li*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02216](https://doi.org/10.1109/CVPR52729.2023.02216)

        **Abstract**:

        Sign language recognition (SLR) is a weakly supervised task that annotates sign videos as textual glosses. Recent studies show that insufficient training caused by the lack of large-scale available sign datasets becomes the main bottleneck for SLR. Most SLR works thereby adopt pretrained visual modules and develop two mainstream solutions. The multi-stream architectures extend multi-cue visual features, yielding the current SOTA performances but requiring complex designs and might introduce potential noise. Alternatively, the advanced single-cue SLR frameworks using explicit cross-modal alignment between visual and textual modalities are simple and effective, potentially competitive with the multi-cue framework. In this work, we propose a novel contrastive visual-textual transformation for SLR, CVT-SLR, to fully explore the pretrained knowledge of both the visual and language modalities. Based on the single-cue cross-modal alignment framework, we propose a variational autoencoder (VAE) for pretrained contextual knowledge while introducing the complete pretrained language module. The VAE implicitly aligns visual and textual modalities while benefiting from pretrained contextual knowledge as the traditional contextual module. Meanwhile, a contrastive cross-modal alignment algorithm is designed to explicitly enhance the consistency constraints. Extensive experiments on public datasets (PHOENIX-2014 and PHOENIX-2014T) demonstrate that our proposed CVT-SLR consistently outperforms existing single-cue methods and even outperforms SOTA multi-cue methods. The source codes and models are available at https://github.com/binbinjiang/CVT-SLR.

        ----

        ## [2203] Joint Visual Grounding and Tracking with Natural Language Specification

        **Authors**: *Li Zhou, Zikun Zhou, Kaige Mao, Zhenyu He*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02217](https://doi.org/10.1109/CVPR52729.2023.02217)

        **Abstract**:

        Tracking by natural language specification aims to locate the referred target in a sequence based on the natural language description. Existing algorithms solve this issue in two steps, visual grounding and tracking, and accordingly deploy the separated grounding model and tracking model to implement these two steps, respectively. Such a separated framework overlooks the link between visual grounding and tracking, which is that the natural language descriptions provide global semantic cues for localizing the target for both two steps. Besides, the separated framework can hardly be trained end-to-end. To handle these issues, we propose a joint visual grounding and tracking framework, which reformulates grounding and tracking as a unified task: localizing the referred target based on the given visual-language references. Specifically, we propose a multi-source relation modeling module to effectively build the relation between the visual-language references and the test image. In addition, we design a temporal modeling module to provide a temporal clue with the guidance of the global semantic information for our model, which effectively improves the adaptability to the appearance variations of the target. Extensive experimental results on TNL2K, LaSOT, OTB99, and RefCOCOg demonstrate that our method performs favorably against state-of-the-art algorithms for both tracking and grounding. Code is available at https://github.com/lizhou-cs/JointNLT.

        ----

        ## [2204] Accelerating Vision-Language Pretraining with Free Language Modeling

        **Authors**: *Teng Wang, Yixiao Ge, Feng Zheng, Ran Cheng, Ying Shan, Xiaohu Qie, Ping Luo*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02218](https://doi.org/10.1109/CVPR52729.2023.02218)

        **Abstract**:

        The state of the arts in vision-language pretraining (VLP) achieves exemplary performance but suffers from high training costs resulting from slow convergence and long training time, especially on large-scale web datasets. An essential obstacle to training efficiency lies in the entangled prediction rate (percentage of tokens for reconstruction) and corruption rate (percentage of corrupted tokens) in masked language modeling (MLM), that is, a proper corruption rate is achieved at the cost of a large portion of output tokens being excluded from prediction loss. To accelerate the convergence of VLP, we propose a new pretraining task, namely, free language modeling (FLM), that enables a 100% prediction rate with arbitrary corruption rates. FLM successfully frees the prediction rate from the tie-up with the corruption rate while allowing the corruption spans to be customized for each token to be predicted. FLM-trained models are encouraged to learn better and faster given the same GPU time by exploiting bidirectional contexts more flexibly. Extensive experiments show FLM could achieve an impressive 2.5 × pretraining time reduction in comparison to the MLM-based methods, while keeping competitive performance on both vision-language understanding and generation tasks. Code will be public at https://github.com/TencentARC/FLM.

        ----

        ## [2205] CoWs on Pasture: Baselines and Benchmarks for Language-Driven Zero-Shot Object Navigation

        **Authors**: *Samir Yitzhak Gadre, Mitchell Wortsman, Gabriel Ilharco, Ludwig Schmidt, Shuran Song*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02219](https://doi.org/10.1109/CVPR52729.2023.02219)

        **Abstract**:

        For robots to be generally useful, they must be able to find arbitrary objects described by people (i.e., be language-driven) even without expensive navigation training on in-domain data (i.e., perform zero-shot inference). We explore these capabilities in a unified setting: language- driven zero-shot object navigation (L-ZSON). Inspired by the recent success of open-vocabulary models for image classification, we investigate a straightforward framework, CLIP on Wheels (CoW), to adapt open-vocabulary models to this task without fine-tuning. To better evaluate L-ZSON, we introduce the Pasturebenchmark, which considers finding uncommon objects, objects described by spatial and appearance attributes, and hidden objects described relative to visible objects. We conduct an in-depth empirical study by directly deploying 22 CoW baselines across Habitat, Robothor,and Pasture. In total, we evaluate over 90k navigation episodes and find that (1) CoW baselines often struggle to leverage language descriptions but are proficient at finding uncommon objects. (2) A simple Co W, with CLIP-based object localization and classical exploration-and no additional training-matches the navigation efficiency of a state-of-the-art ZSON method trained for 500M steps on HabitatMp3d data. This same CoW provides a 15.6 percentage point improvement in success over a state-of-the-art ROBOTHOR ZSON model.
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
For code, data, and videos, see cow.cs.columbia.edu/

        ----

        ## [2206] Where We Are and What We're Looking At: Query Based Worldwide Image Geo-localization Using Hierarchies and Scenes

        **Authors**: *Brandon Clark, Alec Kerrigan, Parth Parag Kulkarni, Vicente Vivanco Cepeda, Mubarak Shah*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02220](https://doi.org/10.1109/CVPR52729.2023.02220)

        **Abstract**:

        Determining the exact latitude and longitude that a photo was taken is a useful and widely applicable task, yet it remains exceptionally difficult despite the accelerated progress of other computer vision tasks. Most previous approaches have opted to learn single representations of query images, which are then classified at different levels of geographic granularity. These approaches fail to exploit the different visual cues that give context to different hierarchies, such as the country, state, and city level. To this end, we introduce an end-to-end transformer-based architecture that exploits the relationship between different geographic levels (which we refer to as hierarchies) and the corresponding visual scene information in an image through hierarchical cross-attention. We achieve this by learning a query for each geographic hierarchy and scene type. Furthermore, we learn a separate representation for different environmental scenes, as different scenes in the same location are often defined by completely different visual features. We achieve state of the art accuracy on 4 standard geo-localization datasets : Im2GPS, Im2GPS3k, YFCC4k, and YFCC26k, as well as qualitatively demonstrate how our method learns different representations for different visual hierarchies and scenes, which has not been demonstrated in the previous methods. Above previous testing datasets mostly consist of iconic landmarks or images taken from social media, which makes the dataset a simple memory task, or makes it biased towards certain places. To address this issue we introduce a much harder testing dataset, Google-World-Streets-15k, comprised of images taken from Google Streetview covering the whole planet and present state of the art results. Our code can be found at https://github.com/AHKerrigan/GeoGuessNet.

        ----

        ## [2207] ANetQA: A Large-scale Benchmark for Fine-grained Compositional Reasoning over Untrimmed Videos

        **Authors**: *Zhou Yu, Lixiang Zheng, Zhou Zhao, Fei Wu, Jianping Fan, Kui Ren, Jun Yu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02221](https://doi.org/10.1109/CVPR52729.2023.02221)

        **Abstract**:

        Building benchmarks to systemically analyze different capabilities of video question answering (VideoQA) models is challenging yet crucial. Existing benchmarks often use non-compositional simple questions and suffer from language biases, making it difficult to diagnose model weaknesses incisively. A recent benchmark AGQA [8] poses a promising paradigm to generate QA pairs automatically from pre-annotated scene graphs, enabling it to measure diverse reasoning abilities with granular control. However, its questions have limitations in reasoning about the fine-grained semantics in videos as such information is absent in its scene graphs. To this end, we present ANetQA, a large-scale benchmark that supports fine-grained compositional reasoning over the challenging untrimmed videos from ActivityNet [4]. Similar to AGQA, the QA pairs in ANetQA are automatically generated from annotated video scene graphs. The fine-grained properties of ANetQA are reflected in the following: (i) untrimmed videos with fine-grained semantics; (ii) spatio-temporal scene graphs with fine-grained taxonomies; and (iii) diverse questions generated from fine-grained templates. ANetQA attains 1.4 billion unbalanced and 13.4 million balanced QA pairs, which is an order of magnitude larger than AGQA with a similar number of videos. Comprehensive experiments are performed for state-of-the-art methods. The best model achieves 44.5% accuracy while human performance tops out at 84.5%, leaving sufficient room for improvement.

        ----

        ## [2208] MetaCLUE: Towards Comprehensive Visual Metaphors Research

        **Authors**: *Arjun R. Akula, Brendan Driscoll, Pradyumna Narayana, Soravit Changpinyo, Zhiwei Jia, Suyash Damle, Garima Pruthi, Sugato Basu, Leonidas J. Guibas, William T. Freeman, Yuanzhen Li, Varun Jampani*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02222](https://doi.org/10.1109/CVPR52729.2023.02222)

        **Abstract**:

        Creativity is an indispensable part of human cognition and also an inherent part of how we make sense of the world. Metaphorical abstraction is fundamental in communicating creative ideas through nuanced relationships between abstract concepts such as feelings. While computer vision benchmarks and approaches predominantly focus on understanding and generating literal interpretations of images, metaphorical comprehension of images remains relatively unexplored. Towards this goal, we introduce Meta-CLUE, a set of vision tasks on visual metaphor. We also collect high-quality and rich metaphor annotations (abstract objects, concepts, relationships along with their corresponding object boxes) as there do not exist any datasets that facilitate the evaluation of these tasks. We perform a comprehensive analysis of state-of-the-art models in vision and language based on our annotations, highlighting strengths and weaknesses of current approaches in visual metaphor classification, localization, understanding (retrieval, question answering, captioning) and generation (text-to-image synthesis) tasks. We hope this work provides a concrete step towards developing AI systems with human-like creative capabilities. Project page: https://metaclue.github.io

        ----

        ## [2209] GeoVLN: Learning Geometry-Enhanced Visual Representation with Slot Attention for Vision-and-Language Navigation

        **Authors**: *Jingyang Huo, Qiang Sun, Boyan Jiang, Haitao Lin, Yanwei Fu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02223](https://doi.org/10.1109/CVPR52729.2023.02223)

        **Abstract**:

        Most existing works solving Room-to-Room VLN problem only utilize RGB images and do not consider local context around candidate views, which lack sufficient visual cues about surrounding environment. Moreover, natural language contains complex semantic information thus its correlations with visual inputs are hard to model merely with cross attention. In this paper, we propose GeoVLN, which learns Geometry-enhanced visual representation based on slot attention for robust Visual-and-Language Navigation. The RGB images are compensated with the corresponding depth maps and normal maps predicted by Omnidata as visual inputs. Technically, we introduce a two-stage module that combine local slot attention and CLIP model to produce geometry-enhanced representation from such input. We employ V&L BERT to learn a cross-modal representation that incorporate both language and vision informations. Additionally, a novel multiway attention module is designed, encouraging different phrases of input instruction to exploit the most related features from visual input. Extensive experiments demonstrate the effectiveness of our newly designed modules and show the compelling performance of the proposed method.

        ----

        ## [2210] Being Comes from Not-Being: Open-Vocabulary Text-to-Motion Generation with Wordless Training

        **Authors**: *Junfan Lin, Jianlong Chang, Lingbo Liu, Guanbin Li, Liang Lin, Qi Tian, Chang Wen Chen*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02224](https://doi.org/10.1109/CVPR52729.2023.02224)

        **Abstract**:

        Text-to-motion generation is an emerging and challenging problem, which aims to synthesize motion with the same semantics as the input text. However, due to the lack of diverse labeled training data, most approaches either limit to specific types of text annotations or require online optimizations to cater to the texts during inference at the cost of efficiency and stability. In this paper, we investigate offline open-vocabulary text-to-motion generation in a zero-shot learning manner that neither requires paired training data nor extra online optimization to adapt for unseen texts. Inspired by the prompt learning in NLP, we pretrain a motion generator that learns to reconstruct the full motion from the masked motion. During inference, instead of changing the motion generator, our method reformulates the input text into a masked motion as the prompt for the motion generator to “reconstruct” the motion. In constructing the prompt, the unmasked poses of the prompt are synthesized by a text-to-pose generator. To supervise the optimization of the text-to-pose generator, we propose the first text-pose alignment model for measuring the alignment between texts and 3D poses. And to prevent the pose generator from over-fitting to limited training texts, we further propose a novel wordless training mechanism that optimizes the text-to-pose generator without any training texts. The comprehensive experimental results show that our method obtains a significant improvement against the baseline methods. The code is available at https://github.com/junfanlin/oohmg.

        ----

        ## [2211] LASP: Text-to-Text Optimization for Language-Aware Soft Prompting of Vision & Language Models

        **Authors**: *Adrian Bulat, Georgios Tzimiropoulos*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02225](https://doi.org/10.1109/CVPR52729.2023.02225)

        **Abstract**:

        Soft prompt learning has recently emerged as one of the methods of choice for adapting V&L models to a downstream task using a few training examples. However, current methods significantly overfit the training data, suffering from large accuracy degradation when tested on unseen classes from the same domain. To this end, in this paper, we make the following 4 contributions: (1) To alleviate base class overfitting, we propose a novel Language- Aware Soft Prompting (LASP) learning method by means of a text-to-text cross-entropy loss that maximizes the probability of the learned prompts to be correctly classified with respect to pre-defined hand-crafted textual prompts. (2) To increase the representation capacity of the prompts, we propose grouped LASP where each group of prompts is optimized with respect to a separate subset of textual prompts. (3) We identify a visual-language misalignment introduced by prompt learning and LASP, and more importantly, propose a re-calibration mechanism to address it. (4) We show that LASP is inherently amenable to including, during training, virtual classes, i.e. class names for which no visual samples are available, further increasing the robustness of the learned prompts. Through evaluations on 11 datasets, we show that our approach (a) significantly outperforms all prior works on soft prompting, and (b) matches and surpasses, for the first time, the accuracy on novel classes obtained by hand-crafted prompts and CLIP for 8 out of 11 test datasets. Code will be made available here.

        ----

        ## [2212] Position-Guided Text Prompt for Vision-Language Pre-Training

        **Authors**: *Jinpeng Wang, Pan Zhou, Mike Zheng Shou, Shuicheng Yan*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02226](https://doi.org/10.1109/CVPR52729.2023.02226)

        **Abstract**:

        Vision-Language Pre-Training (VLP) has shown promising capabilities to align image and text pairs, facilitating a broad variety of cross-modal learning tasks. However, we observe that VLP models often lack the visual grounding/localization capability which is critical for many downstream tasks such as visual reasoning. In this work, we propose a novel Position-guided Text Prompt (PTP) paradigm to enhance the visual grounding ability of cross-modal models trained with VLP. Specifically, in the VLP phase, PTP divides the image into N x N blocks, and identifies the objects in each block through the widely used object detector in VLP. It then reformulates the visual grounding task into a fill-in-the-blank problem given a PTP by encouraging the model to predict the objects in the given blocks or regress the blocks of a given object, e.g. filling “[P]” or “[O]” in a PTP “The block [P] has a [O]”. This mechanism improves the visual grounding capability of VLP models and thus helps them better handle various downstream tasks. By introducing PTP into several state-of-the-art VLP frameworks, we observe consistently significant improvements across representative cross-modal learning model architectures and several benchmarks, e.g. zero-shot Flickr30K Retrieval (+4.8 in average recall@1) for ViLT [16] baseline, and COCO Captioning (+5.3 in CIDEr) for SOTA BLIP [19] baseline. Moreover, PTP achieves comparable results with object-detector based methods [8, 23, 45], and much faster inference speed since PTP discards its object detector for inference while the later cannot.

        ----

        ## [2213] Intrinsic Physical Concepts Discovery with Object-Centric Predictive Models

        **Authors**: *Qu Tang, Xiangyu Zhu, Zhen Lei, Zhaoxiang Zhang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02227](https://doi.org/10.1109/CVPR52729.2023.02227)

        **Abstract**:

        The ability to discover abstract physical concepts and understand how they work in the world through observing lies at the core of human intelligence. The acquisition of this ability is based on compositionally perceiving the environment in terms of objects and relations in an unsupervised manner. Recent approaches learn object-centric represen-tations and capture visually observable concepts of objects, e.g., shape, size, and location. In this paper, we take a step forward and try to discover and represent intrinsic physical concepts such as mass and charge. We introduce the PHYsi-cal Concepts Inference NEtwork (PHYCINE), a system that infers physical concepts in different abstract levels with-out supervision. The key insights underlining PHYCINE are two-fold, commonsense knowledge emerges with pre-diction, and physical concepts of different abstract levels should be reasoned in a bottom-up fashion. Empirical eval-uation demonstrates that variables inferred by our system work in accordance with the properties of the corresponding physical concepts. We also show that object representations containing the discovered physical concepts variables could help achieve better performance in causal reasoning tasks, i.e., ComPhy.

        ----

        ## [2214] MAP: Multimodal Uncertainty-Aware Vision-Language Pre-training Model

        **Authors**: *Yatai Ji, Junjie Wang, Yuan Gong, Lin Zhang, Yanru Zhu, Hongfa Wang, Jiaxing Zhang, Tetsuya Sakai, Yujiu Yang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02228](https://doi.org/10.1109/CVPR52729.2023.02228)

        **Abstract**:

        Multimodal semantic understanding often has to deal with uncertainty, which means the obtained messages tend to refer to multiple targets. Such uncertainty is problematic for our interpretation, including inter- and intra-modal uncertainty. Little effort has studied the modeling of this uncertainty, particularly in pretraining on unlabeled datasets and fine-tuning in task-specific downstream datasets. In this paper, we project the representations of all modalities as probabilistic distributions via a Probability Distribution Encoder (PDE) by utilizing sequence-level interactions. Compared to the existing deterministic methods, such uncertainty modeling can convey richer multimodal semantic information and more complex relationships. Furthermore, we integrate uncertainty modeling with popular pretraining frameworks and propose suitable pretraining tasks: Distribution-based Vision-Language Contrastive learning (D-VLC), Distribution-based Masked Language Modeling (D-MLM), and Distribution-based Image-Text Matching (D-ITM). The fine-tuned models are applied to challenging downstream tasks, including image-text retrieval, visual question answering, visual reasoning, and visual entailment, and achieve state-of-the-art results.

        ----

        ## [2215] CLAMP: Prompt-based Contrastive Learning for Connecting Language and Animal Pose

        **Authors**: *Xu Zhang, Wen Wang, Zhe Chen, Yufei Xu, Jing Zhang, Dacheng Tao*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02229](https://doi.org/10.1109/CVPR52729.2023.02229)

        **Abstract**:

        Animal pose estimation is challenging for existing image-based methods because of limited training data and large intra- and inter-species variances. Motivated by the progress of visual-language research, we propose that pre-trained language models (e.g., CLIP) can facilitate animal pose estimation by providing rich prior knowledge for describing animal keypoints in text. However, we found that building effective connections between pre-trained language models and visual animal keypoints is non-trivial since the gap between text-based descriptions and keypoint-based visual features about animal pose can be significant. To address this issue, we introduce a novel prompt-based Contrastive learning scheme for connecting Language and AniMal Pose (CLAMP) effectively. The CLAMP attempts to bridge the gap by adapting the text prompts to the animal keypoints during network training. The adaptation is decomposed into spatialaware and feature-aware processes, and two novel contrastive losses are devised correspondingly. In practice, the CLAMP enables the first cross-modal animal pose estimation paradigm. Experimental results show that our method achieves state-of-the-art performance under the supervised, few-shot, and zero-shot settings, outperforming image-based methods by a large margin. The code is available at https://github.com/xuzhang1199/CLAMP.

        ----

        ## [2216] Teacher-generated spatial-attention labels boost robustness and accuracy of contrastive models

        **Authors**: *Yushi Yao, Chang Ye, Junfeng He, Gamaleldin F. Elsayed*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02230](https://doi.org/10.1109/CVPR52729.2023.02230)

        **Abstract**:

        Human spatial attention conveys information about the regions of visual scenes that are important for performing visual tasks. Prior work has shown that the information about human attention can be leveraged to benefit various supervised vision tasks. Might providing this weak form of supervision be useful for self-supervised representation learning? Addressing this question requires collecting large datasets with human attention labels. Yet, collecting such large scale data is very expensive. To address this challenge, we construct an auxiliary teacher model to predict human attention, trained on a relatively small labeled dataset. This teacher model allows us to generate image (pseudo) attention labels for ImageNet. We then train a model with a primary contrastive objective; to this standard configuration, we add a simple output head trained to predict the attention map for each image, guided by the pseudo labels from teacher model. We measure the quality of learned representations by evaluating classification performance from the frozen learned embeddings as well as performance on image retrieval tasks (see supplementary material). We find that the spatial-attention maps predicted from the contrastive model trained with teacher guidance aligns better with human attention compared to vanilla contrastive models. Moreover, we find that our approach improves classification accuracy and robustness of the contrastive models on ImageNet and ImageNet-C. Further, we find that model representations become more useful for image retrieval task as measured by precision-recall performance on ImageNet, ImageNet-C, CIFAR10, and CIFAR10-C datasets.

        ----

        ## [2217] DegAE: A New Pretraining Paradigm for Low-Level Vision

        **Authors**: *Yihao Liu, Jingwen He, Jinjin Gu, Xiangtao Kong, Yu Qiao, Chao Dong*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02231](https://doi.org/10.1109/CVPR52729.2023.02231)

        **Abstract**:

        Self-supervised pretraining has achieved remarkable success in high-level vision, but its application in low-level vision remains ambiguous and not well-established. What is the primitive intention of pretraining? What is the core problem of pretraining in low-level vision? In this paper, we aim to answer these essential questions and establish a new pretraining scheme for low-level vision. Specifically, we examine previous pretraining methods in both high-level and low-level vision, and categorize current low-level vision tasks into two groups based on the difficulty of data acqui-sition: low-cost and high-cost tasks. Existing literature has mainly focused on pretraining for low-cost tasks, where the observed performance improvement is often limited. However, we argue that pretraining is more significant for high-cost tasks, where data acquisition is more challenging. To learn a general low-level vision representation that can improve the performance of various tasks, we propose a new pretraining paradigm called degradation autoencoder (De-gAE). DegAE follows the philosophy of designing pretext task for self-supervised pretraining and is elaborately tai-lored to low-level vision. With DegAE pretraining, SwinIR achieves a 6.88dB performance gain on image dehaze task, while Uformer obtains 3.22dB and 0.54dB improvement on dehaze and derain tasks, respectively.

        ----

        ## [2218] RILS: Masked Visual Reconstruction in Language Semantic Space

        **Authors**: *Shusheng Yang, Yixiao Ge, Kun Yi, Dian Li, Ying Shan, Xiaohu Qie, Xinggang Wang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02232](https://doi.org/10.1109/CVPR52729.2023.02232)

        **Abstract**:

        Both masked image modeling (MIM) and natural language supervision have facilitated the progress of transferable visual pre-training. In this work, we seek the synergy between two paradigms and study the emerging properties when MIM meets natural language supervision. To this end, we present a novel masked visual Reconstruction In Language semantic Space (RILS) pre-training framework, in which sentence representations, encoded by the text encoder, serve as prototypes to transform the vision-only signals into patch-sentence probabilities as semantically meaningful MIM reconstruction targets. The vision models can therefore capture useful components with structured information by predicting proper semantic of masked tokens. Better visual representations could, in turn, improve the text encoder via the image-text alignment objective, which is essential for the effective MIM target transformation. Extensive experimental results demonstrate that our method not only enjoys the best of previous MIM and CLIP but also achieves further improvements on various tasks due to their mutual benefits. RILS exhibits advanced transferability on downstream classification, detection, and segmentation, especially for low-shot regimes. Code is available at https://github.com/hustvl/RILS.

        ----

        ## [2219] Learning Geometry-aware Representations by Sketching

        **Authors**: *Hyundo Lee, Inwoo Hwang, Hyunsung Go, Won-Seok Choi, Kibeom Kim, Byoung-Tak Zhang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02233](https://doi.org/10.1109/CVPR52729.2023.02233)

        **Abstract**:

        Understanding geometric concepts, such as distance and shape, is essential for understanding the real world and also for many vision tasks. To incorporate such information into a visual representation of a scene, we propose learning to represent the scene by sketching, inspired by human behavior. Our method, coined Learning by Sketching (LBS), learns to convert an image into a set of colored strokes that explicitly incorporate the geometric information of the scene in a single inference step without requiring a sketch dataset. A sketch is then generated from the strokes where CLIP-based perceptual loss maintains a semantic similarity between the sketch and the image. We show theoretically that sketching is equivariant with respect to arbitrary affine transformations and thus provably preserves geometric information. Experimental results show that LBS substantially improves the performance of object attribute classification on the unlabeled CLEVR dataset, domain transfer between CLEVR and STL-10 datasets, and for diverse downstream tasks, confirming that LBS provides rich geometric information.

        ----

        ## [2220] SketchXAI: A First Look at Explainability for Human Sketches

        **Authors**: *Zhiyu Qu, Yulia Gryaditskaya, Ke Li, Kaiyue Pang, Tao Xiang, Yi-Zhe Song*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02234](https://doi.org/10.1109/CVPR52729.2023.02234)

        **Abstract**:

        This paper, for the very first time, introduces human sketches to the landscape of XAI (Explainable Artificial Intelligence). We argue that sketch as a “human-centred” data form, represents a natural interface to study explainability. We focus on cultivating sketch-specific explainability designs. This starts by identifying strokes as a unique building block that offers a degree of flexibility in object construction and manipulation impossible in photos. Following this, we design a simple explainability-friendly sketch encoder that accommodates the intrinsic properties of strokes: shape, location, and order. We then move on to define the first ever XAI task for sketch, that of stroke location inversion (SLI). Just as we have heat maps for photos, and correlation matrices for text, SLI offers an explainability angle to sketch in terms of asking a network how well it can recover stroke locations of an unseen sketch. We offer qualitative results for readers to interpret as snapshots of the SLI process in the paper, and as GIFs on the project page. A minor but interesting note is that thanks to its sketch-specific design, our sketch encoder also yields the best sketch recognition accuracy to date while having the smallest number of parameters. The code is available at https://sketchxai.github.io.

        ----

        ## [2221] MAGVLT: Masked Generative Vision-and-Language Transformer

        **Authors**: *Sungwoong Kim, Daejin Jo, Donghoon Lee, Jongmin Kim*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02235](https://doi.org/10.1109/CVPR52729.2023.02235)

        **Abstract**:

        While generative modeling on multimodal image-text data has been actively developed with large-scale paired datasets, there have been limited attempts to generate both image and text data by a single model rather than a generation of one fixed modality conditioned on the other modality. In this paper, we explore a unified generative vision-and-language (VL) model that can produce both images and text sequences. Especially, we propose a generative VL transformer based on the non-autoregressive mask prediction, named MAGVLT, and compare it with an autoregressive generative VL transformer (ARGVLT). In comparison to ARGVLT, the proposed MAGVLT enables bidirectional context encoding, fast decoding by parallel token predictions in an iterative refinement, and extended editing capabilities such as image and text infilling. For rigorous training of our MAGVLT with image-text pairs from scratch, we combine the image-to-text, text-to-image, and joint image-and-text mask prediction tasks. Moreover, we devise two additional tasks based on the step-unrolled mask prediction and the selective prediction on the mixture of two image-text pairs. Experimental results on various downstream generation tasks of VL benchmarks show that our MAGVLT outperforms ARGVLT by a large margin even with significant inference speedup. Particularly, MAGVLT achieves competitive results on both zero-shot image-to-text and text-to-image generation tasks from MS-COCO by one moderate-sized model (fewer than 500M parameters) even without the use of monomodal data and networks.

        ----

        ## [2222] Zero-Shot Everything Sketch-Based Image Retrieval, and in Explainable Style

        **Authors**: *Fengyin Lin, Mingkang Li, Da Li, Timothy M. Hospedales, Yi-Zhe Song, Yonggang Qi*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02236](https://doi.org/10.1109/CVPR52729.2023.02236)

        **Abstract**:

        This paper studies the problem of zero-short sketch-based image retrieval (ZS-SBIR), however with two significant differentiators to prior art (i) we tackle all variants (inter-category, intracategory, and cross datasets) of ZS-SBIR with just one network (“everything”), and (ii) we would really like to understand how this sketch-photo matching operates (“explainable”). Our key innovation lies with the realization that such a cross-modal matching problem could be reduced to comparisons of groups of key local patches - akin to the seasoned “bag-of-words” paradigm. Just with this change, we are able to achieve both of the aforementioned goals, with the added benefit of no longer requiring external semantic knowledge. Technically, ours is a transformer-based cross-modal network, with three novel components (i) a self-attention module with a learnable tokenizer to produce visual tokens that correspond to the most informative local regions, (ii) a cross-attention module to compute local correspondences between the visual tokens across two modalities, and finally (iii) a kernel-based relation network to assemble local putative matches and produce an overall similarity metric for a sketch-photo pair. Experiments show ours indeed delivers superior performances across all ZS-SBIR settings. The all important explainable goal is elegantly achieved by visualizing cross-modal token correspondences, and for the first time, via sketch to photo synthesis by universal replacement of all matched photo patches. Code and model are available at https://github.com/buptLinfy/ZSE-SBIR.

        ----

        ## [2223] Semantic-Conditional Diffusion Networks for Image Captioning

        **Authors**: *Jianjie Luo, Yehao Li, Yingwei Pan, Ting Yao, Jianlin Feng, Hongyang Chao, Tao Mei*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02237](https://doi.org/10.1109/CVPR52729.2023.02237)

        **Abstract**:

        Recent advances on text-to-image generation have witnessed the rise of diffusion models which act as powerful generative models. Nevertheless, it is not trivial to exploit such latent variable models to capture the dependency among discrete words and meanwhile pursue complex visual-language alignment in image captioning. In this paper, we break the deeply rooted conventions in learning Transformer-based encoder-decoder, and propose a new diffusion model based paradigm tailored for image captioning, namely Semantic-Conditional Diffusion Networks (SCD-Net). Technically, for each input image, we first search the semantically relevant sentences via cross-modal retrieval model to convey the comprehensive semantic information. The rich semantics are further regarded as semantic prior to trigger the learning of Diffusion Transformer, which produces the output sentence in a diffusion process. In SCD-Net, multiple Diffusion Transformer structures are stacked to progressively strengthen the output sentence with better visional-language alignment and linguistical coherence in a cascaded manner. Furthermore, to stabilize the diffusion process, a new self-critical sequence training strategy is designed to guide the learning of SCD-Net with the knowledge of a standard autoregressive Transformer model. Extensive experiments on COCO dataset demonstrate the promising potential of using diffusion models in the challenging image captioning task. Source code is available at

        ----

        ## [2224] Reveal: Retrieval-Augmented Visual-Language Pre-Training with Multi-Source Multimodal Knowledge Memory

        **Authors**: *Ziniu Hu, Ahmet Iscen, Chen Sun, Zirui Wang, Kai-Wei Chang, Yizhou Sun, Cordelia Schmid, David A. Ross, Alireza Fathi*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02238](https://doi.org/10.1109/CVPR52729.2023.02238)

        **Abstract**:

        In this paper, we propose an end-to-end Retrieval-Augmented Visual Language Model (REVEAL) that learns to encode world knowledge into a large-scale memory, and to retrieve from it to answer knowledge-intensive queries. Reveal consists of four key components: the memory, the encoder, the retriever and the generator. The large-scale memory encodes various sources of multimodal world knowledge (e.g. image-text pairs, question answering pairs, knowledge graph triplets, etc.) via a unified encoder. The retriever finds the most relevant knowledge entries in the memory, and the generator fuses the retrieved knowledge with the input query to produce the output. A key novelty in our approach is that the memory, encoder, retriever and generator are all pre-trained end-to-end on a massive amount of data. Furthermore, our approach can use a diverse set of multimodal knowledge sources, which is shown to result in significant gains. We show that Reveal achieves state-of-the-art results on visual question answering and image captioning. The project page of this work is reveal. github. io.

        ----

        ## [2225] Variational Distribution Learning for Unsupervised Text-to-Image Generation

        **Authors**: *Minsoo Kang, Doyup Lee, Jiseob Kim, Saehoon Kim, Bohyung Han*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02239](https://doi.org/10.1109/CVPR52729.2023.02239)

        **Abstract**:

        We propose a text-to-image generation algorithm based on deep neural networks when text captions for images are unavailable during training. In this work, instead of simply generating pseudo-ground-truth sentences of training images using existing image captioning methods, we employ a pretrained CLIP model, which is capable of properly aligning embeddings of images and corresponding texts in a joint space and, consequently, works well on zero-shot recognition tasks. We optimize a text-to-image generation model by maximizing the data log-likelihood conditioned on pairs of image-text CLIP embeddings. To better align data in the two domains, we employ a principled way based on a variational inference, which efficiently estimates an approximate posterior of the hidden text embedding given an image and its CLIP feature. Experimental results validate that the proposed framework outperforms existing approaches by large margins under unsupervised and semi-supervised text-to-image generation settings.

        ----

        ## [2226] Scaling Language-Image Pre-Training via Masking

        **Authors**: *Yanghao Li, Haoqi Fan, Ronghang Hu, Christoph Feichtenhofer, Kaiming He*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02240](https://doi.org/10.1109/CVPR52729.2023.02240)

        **Abstract**:

        We present Fast Language-Image Pre-training (FLIP), a simple and more efficient method for training CLIP [52]. Our method randomly masks out and removes a large portion of image patches during training. Masking allows us to learn from more image-text pairs given the same wall-clock time and contrast more samples per iteration with similar memory footprint. It leads to a favorable trade-off between accuracy and training time. In our experiments on 400 million image-text pairs, FLIP improves both accuracy and speed over the no-masking baseline. On a large diversity of downstream tasks, FLIP dominantly outperforms the CLIP counterparts trained on the same data. Facilitated by the speedup, we explore the scaling behavior of increasing the model size, data size, or training length, and report encouraging results and comparisons. We hope that our work will foster future research on scaling vision-language learning.

        ----

        ## [2227] LANIT: Language-Driven Image-to-Image Translation for Unlabeled Data

        **Authors**: *Jihye Park, Sunwoo Kim, Soohyun Kim, Seokju Cho, Jaejun Yoo, Youngjung Uh, Seungryong Kim*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02241](https://doi.org/10.1109/CVPR52729.2023.02241)

        **Abstract**:

        Existing techniques for image-to-image translation commonly have suffered from two critical problems: heavy reliance on per-sample domain annotation and/or inability to handle multiple attributes per image. Recent truly-unsupervised methods adopt clustering approaches to easily provide per-sample one-hot domain labels. However, they cannot account for the real-world setting: one sample may have multiple attributes. In addition, the semantics of the clusters are not easily coupled to human understanding. To overcome these, we present LANguage-driven Image-to-image Translation model, dubbed LANIT. We leverage easy-to-obtain candidate attributes given in texts for a dataset: the similarity between images and attributes indicates per-sample domain labels. This formulation naturally enables multi-hot labels so that users can specify the target domain with a set of attributes in language. To account for the case that the initial prompts are inaccurate, we also present prompt learning. We further present domain regularization loss that enforces translated images to be mapped to the corresponding domain. Experiments on several standard benchmarks demonstrate that LANIT achieves comparable or superior performance to existing models. The code is available at github.com/KU-CVLAB/LANIT.

        ----

        ## [2228] Revisiting Self-Similarity: Structural Embedding for Image Retrieval

        **Authors**: *Seongwon Lee, Suhyeon Lee, Hongje Seong, Euntai Kim*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02242](https://doi.org/10.1109/CVPR52729.2023.02242)

        **Abstract**:

        Despite advances in global image representation, existing image retrieval approaches rarely consider geometric structure during the global retrieval stage. In this work, we revisit the conventional self-similarity descriptor from a convolutional perspective, to encode both the visual and structural cues of the image to global image representation. Our proposed network, named Structural Embedding Network (SENet), captures the internal structure of the images and gradually compresses them into dense self-similarity descriptors while learning diverse structures from various images. These self-similarity descriptors and original image features are fused and then pooled into global embedding, so that global embedding can represent both geometric and visual cues of the image. Along with this novel structural embedding, our proposed network sets new state-of-the-art performances on several image retrieval benchmarks, convincing its robustness to look-alike distractors. The code and models are available: https://github.com/sungonce/SENet.

        ----

        ## [2229] Improving Cross-Modal Retrieval with Set of Diverse Embeddings

        **Authors**: *Dongwon Kim, Namyup Kim, Suha Kwak*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02243](https://doi.org/10.1109/CVPR52729.2023.02243)

        **Abstract**:

        Cross-modal retrieval across image and text modalities is a challenging task due to its inherent ambiguity: An image often exhibits various situations, and a caption can be coupled with diverse images. Set-based embedding has been studied as a solution to this problem. It seeks to encode a sample into a set of different embedding vectors that capture different semantics of the sample. In this paper, we present a novel set-based embedding method, which is distinct from previous work in two aspects. First, we present a new similarity function called smooth-Chamfer similarity, which is designed to alleviate the side effects of existing similarity functions for set-based embedding. Second, we propose a novel set prediction module to produce a set of embedding vectors that effectively captures diverse semantics of input by the slot attention mechanism. Our method is evaluated on the COCO and Flickr30K datasets across different visual backbones, where it outperforms existing methods including ones that demand substantially larger computation at inference.

        ----

        ## [2230] Masked Autoencoding Does Not Help Natural Language Supervision at Scale

        **Authors**: *Floris Weers, Vaishaal Shankar, Angelos Katharopoulos, Yinfei Yang, Tom Gunter*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02244](https://doi.org/10.1109/CVPR52729.2023.02244)

        **Abstract**:

        Self supervision and natural language supervision have emerged as two exciting ways to train general purpose image encoders which excel at a variety of downstream tasks. Recent works such as M3AE [31] and SLIP [63] have suggested that these approaches can be effectively combined, but most notably their results use small <20M examples) pre-training datasets and don't effectively reflect the large-scale regime (> 100M samples) that is commonly used for these approaches. Here we investigate whether a similar approach can be effective when trained with a much larger amount of data. We find that a combination of two state of the art approaches: masked autoencoders, MAE [37] and contrastive language image pretraining, CLIP [68] provides a benefit over CLIP when trained on a corpus of 11.3M image-text pairs, but little to no benefit (as evaluated on a suite of common vision tasks) over CLIP when trained on a large corpus of 1.4B images. Our work provides some much needed clarity into the effectiveness (or lack thereof) of self supervision for large-scale image-text training.

        ----

        ## [2231] Few-Shot Learning with Visual Distribution Calibration and Cross-Modal Distribution Alignment

        **Authors**: *Runqi Wang, Hao Zheng, Xiaoyue Duan, Jianzhuang Liu, Yuning Lu, Tian Wang, Songcen Xu, Baochang Zhang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02245](https://doi.org/10.1109/CVPR52729.2023.02245)

        **Abstract**:

        Pre-trained vision-language models have inspired much research on few-shot learning. However, with only a few training images, there exist two crucial problems: (1) the visual feature distributions are easily distracted by class-irrelevant information in images, and (2) the alignment between the visual and language feature distributions is difficult. To deal with the distraction problem, we propose a Selective Attack module, which consists of trainable adapters that generate spatial attention maps of images to guide the attacks on class-irrelevant image areas. By messing up these areas, the critical features are captured and the visual distributions of image features are calibrated. To better align the visual and language feature distributions that describe the same object class, we propose a cross-modal distribution alignment module, in which we introduce a vision-language prototype for each class to align the distributions, and adopt the Earth Mover's Distance (EMD) to optimize the prototypes. For efficient computation, the upper bound of EMD is derived. In addition, we propose an augmentation strategy to increase the diversity of the images and the text prompts, which can reduce overfitting to the few-shot training images. Extensive experiments on 11 datasets demonstrate that our method consistently outperforms prior arts in few-shot learning. The implementation code will be available at https://gitee.com/mindspore/models/tree/master/research/cv/SADA.

        ----

        ## [2232] Deep Hashing with Minimal-Distance-Separated Hash Centers

        **Authors**: *Liangdao Wang, Yan Pan, Cong Liu, Hanjiang Lai, Jian Yin, Ye Liu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02246](https://doi.org/10.1109/CVPR52729.2023.02246)

        **Abstract**:

        Deep hashing is an appealing approach for large-scale image retrieval. Most existing supervised deep hashing methods learn hash functions using pairwise or triple image similarities in randomly sampled mini-batches. They suffer from low training efficiency, insufficient coverage of data distribution, and pair imbalance problems. Recently, central similarity quantization (CSQ) attacks the above problems by using “hash centers” as a global similarity metric, which encourages the hash codes of similar images to approach their common hash center and distance themselves from other hash centers. Although achieving SOTA retrieval performance, CSQ falls short of a worst-case guarantee on the minimal distance between its constructed hash centers, i.e. the hash centers can be arbitrarily close. This paper presents an optimization method that finds hash centers with a constraint on the minimal distance between any pair of hash centers, which is non-trivial due to the non-convex nature of the problem. More importantly, we adopt the Gilbert-Varshamov bound from coding theory, which helps us to obtain a large minimal distance while ensuring the empirical feasibility of our optimization approach. With these clearly-separated hash centers, each is assigned to one image class, we propose several effective loss functions to train deep hashing networks. Extensive experiments on three datasets for image retrieval demonstrate that the proposed method achieves superior retrieval performance over the state-of-the-art deep hashing methods.

        ----

        ## [2233] ConZIC: Controllable Zero-shot Image Captioning by Sampling-Based Polishing

        **Authors**: *Zequn Zeng, Hao Zhang, Ruiying Lu, Dongsheng Wang, Bo Chen, Zhengjue Wang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02247](https://doi.org/10.1109/CVPR52729.2023.02247)

        **Abstract**:

        Zero-shot capability has been considered as a new revolution of deep learning, letting machines work on tasks without curated training data. As a good start and the only existing outcome of zero-shot image captioning (IC), ZeroCap abandons supervised training and sequentially searches every word in the caption using the knowledge of large-scale pre-trained models. Though effective, its autoregressive generation and gradient-directed searching mechanism limit the diversity of captions and inference speed, respectively. Moreover, ZeroCap does not consider the controllability issue of zero-shot IC. To move forward, we propose a framework for Controllable Zero-shot IC, named ConZIC. The core of ConZIC is a novel sampling-based non-autoregressive language model named Gibbs-BERT, which can generate and continuously polish every word. Extensive quantitative and qualitative results demonstrate the superior performance of our proposed ConZIC for both zero-shot IC and controllable zero-shot IC. Especially, ConZIC achieves about 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$5\times$</tex>
 generation speed than ZeroCap, and about 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$1.5\times$</tex>
 diversity scores, with accurate generation given different control signals. Our code is available at https://github.com/joeyz0z/ConZIC.

        ----

        ## [2234] Learning to Name Classes for Vision and Language Models

        **Authors**: *Sarah Parisot, Yongxin Yang, Steven McDonagh*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02248](https://doi.org/10.1109/CVPR52729.2023.02248)

        **Abstract**:

        Large scale vision and language models can achieve impressive zero-shot recognition performance by mapping class specific text queries to image content. Two distinct challenges that remain however, are high sensitivity to the choice of handcrafted class names that define queries, and the difficulty of adaptation to new, smaller datasets. Towards addressing these problems, we propose to leverage available data to learn, for each class, an optimal word embedding as a function of the visual content. By learning new word embeddings on an otherwise frozen model, we are able to retain zero-shot capabilities for new classes, easily adapt models to new datasets, and adjust potentially erroneous, non-descriptive or ambiguous class names. We show that our solution can easily be integrated in image classification and object detection pipelines, yields significant performance gains in multiple scenarios and provides insights into model biases and labelling errors.

        ----

        ## [2235] Data-Efficient Large Scale Place Recognition with Graded Similarity Supervision

        **Authors**: *Maria Leyva-Vallina, Nicola Strisciuglio, Nicolai Petkov*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02249](https://doi.org/10.1109/CVPR52729.2023.02249)

        **Abstract**:

        Visual place recognition (VPR) is a fundamental task of computer vision for visual localization. Existing methods are trained using image pairs that either depict the same place or not. Such a binary indication does not consider continuous relations of similarity between images of the same place taken from different positions, determined by the continuous nature of camera pose. The binary similarity induces a noisy supervision signal into the training of VPR methods, which stall in local minima and require expensive hard mining algorithms to guarantee convergence. Motivated by the fact that two images of the same place only partially share visual cues due to camera pose differences, we deploy an automatic re-annotation strategy to re-label VPR datasets. We compute graded similarity labels for image pairs based on available localization metadata. Furthermore, we propose a new Generalized Contrastive Loss (GCL) that uses graded similarity labels for training contrastive networks. We demonstrate that the use of the new labels and GCL allow to dispense from hard-pair mining, and to train image descriptors that perform better in VPR by nearest neighbor search, obtaining superior or comparable results than methods that require expensive hard-pair mining and re-ranking techniques.

        ----

        ## [2236] DetCLIPv2: Scalable Open-Vocabulary Object Detection Pre-training via Word-Region Alignment

        **Authors**: *Lewei Yao, Jianhua Han, Xiaodan Liang, Dan Xu, Wei Zhang, Zhenguo Li, Hang Xu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02250](https://doi.org/10.1109/CVPR52729.2023.02250)

        **Abstract**:

        This paper presents DetCLIPv2, an efficient and scalable training framework that incorporates large-scale imagetext pairs to achieve open-vocabulary object detection (OVD). Unlike previous OVD frameworks that typically rely on a pre-trained vision-language model (e.g., CLIP) or exploit image-text pairs via a pseudo labeling process, DetCLIPv2 directly learns the fine-grained word-region alignment from massive image-text pairs in an end-to-end manner. To accomplish this, we employ a maximum word-region similarity between region proposals and textual words to guide the contrastive objective. To enable the model to gain localization capability while learning broad concepts, DetCLIPv2 is trained with a hybrid supervision from detection, grounding and image-text pair data under a unified data formulation. By jointly training with an alternating scheme and adopting low-resolution input for image-text pairs, DetCLIPv2 exploits image-text pair data efficiently and effectively: DetCLIPv2 utilizes 13 × more image-text pairs than DetCLIP with a similar training time and improves performance. With 13M image-text pairs for pre-training, DetCLIPv2 demonstrates superior open-vocabulary detection performance, e.g., DetCLIPv2 with Swin-T backbone achieves 40.4% zero-shot AP on the LVIS benchmark, which outperforms previous works GLIP/GLIPv2/DetCLIP by 14.4/11.4/4.5% AP, respectively, and even beats its fully-supervised counterpart by a large margin.

        ----

        ## [2237] HOICLIP: Efficient Knowledge Transfer for HOI Detection with Vision-Language Models

        **Authors**: *Shan Ning, Longtian Qiu, Yongfei Liu, Xuming He*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02251](https://doi.org/10.1109/CVPR52729.2023.02251)

        **Abstract**:

        Human-Object Interaction (HOI) detection aims to localize human-object pairs and recognize their interactions. Recently, Contrastive Language-Image Pre-training (CLIP) has shown great potential in providing interaction prior for HOI detectors via knowledge distillation. However, such approaches often rely on large-scale training data and suffer from inferior performance under few/zero-shot scenarios. In this paper, we propose a novel HOI detection framework that efficiently extracts prior knowledge from CLIP and achieves better generalization. In detail, we first introduce a novel interaction decoder to extract informative regions in the visual feature map of CLIP via a cross-attention mechanism, which is then fused with the detection backbone by a knowledge integration block for more accurate human- object pair detection. In addition, prior knowledge in CLIP text encoder is leveraged to generate a classifier by embedding HOI descriptions. To distinguish fine-grained interactions, we build a verb classifier from training data via visual semantic arithmetic and a lightweight verb representation adapter. Furthermore, we propose a training-free enhancement to exploit global HOI predictions from CLIP. Extensive experiments demonstrate that our method outperforms the state of the art by a large margin on various settings, e.g. +4.04 mAP on HICO-Det. The source code is available in https://github.com/Artanic30/HOICLIP.

        ----

        ## [2238] OvarNet: Towards Open-Vocabulary Object Attribute Recognition

        **Authors**: *Keyan Chen, Xiaolong Jiang, Yao Hu, Xu Tang, Yan Gao, Jianqi Chen, Weidi Xie*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02252](https://doi.org/10.1109/CVPR52729.2023.02252)

        **Abstract**:

        In this paper, we consider the problem of simultaneously detecting objects and inferring their visual attributes in an image, even for those with no manual annotations provided at the training stage, resembling an open-vocabulary scenario. To achieve this goal, we make the following contributions: (i) we start with a naive two-stage approach for open-vocabulary object detection and attribute classification, termed CLIP-Attr. The candidate objects are first proposed with an offline RPN and later classified for semantic category and attributes; (ii) we combine all available datasets and train with a federated strategy to finetune the CLIP model, aligning the visual representation with attributes, additionally, we investigate the efficacy of leveraging freely available online image-caption pairs under weakly supervised learning; (iii) in pursuit of efficiency, we train a Faster-RCNN type model end-to-end with knowledge distillation, that performs class-agnostic object proposals and classification on semantic categories and attributes with classifiers generated from a text encoder; Finally, (iv) we conduct extensive experiments on VAW, MS-COCO, LSA, and OVAD datasets, and show that recognition of semantic category and attributes is complementary for visual scene understanding, i.e., jointly training object detection and attributes prediction largely outperform existing approaches that treat the two tasks independently, demonstrating strong generalization ability to novel attributes and categories.

        ----

        ## [2239] NeRF-RPN: A general framework for object detection in NeRFs

        **Authors**: *Benran Hu, Junkai Huang, Yichen Liu, Yu-Wing Tai, Chi-Keung Tang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02253](https://doi.org/10.1109/CVPR52729.2023.02253)

        **Abstract**:

        This paper presents the first significant object detection framework, NeRF-RPN, which directly operates on NeRF. Given a pre-trained NeRF model, NeRF-RPN aims to detect all bounding boxes of objects in a scene. By exploiting a novel voxel representation that incorporates multi-scale 3D neural volumetric features, we demonstrate it is possible to regress the 3D bounding boxes of objects in NeRF directly without rendering the NeRF at any viewpoint. NeRF-RPN is a general framework and can be applied to detect objects without class labels. We experimented NeRF-RPN with various backbone architectures, RPN head designs and loss functions. All of them can be trained in an end-to-end manner to estimate high quality 3D bounding boxes. To facilitate future research in object detection for NeRF, we built a new benchmark dataset which consists of both synthetic and real-world data with careful labeling and clean up. Code and dataset are available at htt ps: //github.com/lyclyc52/NeRF_RPN.

        ----

        ## [2240] Mask-Free OVIS: Open-Vocabulary Instance Segmentation without Manual Mask Annotations

        **Authors**: *Vibashan VS, Ning Yu, Chen Xing, Can Qin, Mingfei Gao, Juan Carlos Niebles, Vishal M. Patel, Ran Xu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02254](https://doi.org/10.1109/CVPR52729.2023.02254)

        **Abstract**:

        Existing instance segmentation models learn task-specific information using manual mask annotations from base (training) categories. These mask annotations require tremendous human effort, limiting the scalability to annotate novel (new) categories. To alleviate this problem, Open-Vocabulary (OV) methods leverage large-scale image-caption pairs and vision-language models to learn novel categories. In summary, an OV method learns task-specific information using strong supervision from base annotations and novel category information using weak supervision from image-captions pairs. This difference between strong and weak supervision leads to overfitting on base categories, resulting in poor generalization towards novel categories. In this work, we overcome this issue by learning both base and novel categories from pseudo-mask annotations generated by the vision-language model in a weakly supervised manner using our proposed Mask-free OVIS pipeline. Our method automatically generates pseudo-mask annotations by leveraging the localization ability of a pre-trained vision-language model for objects present in image-caption pairs. The generated pseudo-mask annotations are then used to supervise an instance segmentation model, freeing the entire pipeline from any labour-expensive instance-level annotations and overfitting. Our extensive experiments show that our method trained with just pseudo-masks significantly improves the mAP scores on the MS-COCO dataset and OpenImages dataset compared to the recent state-of-the-art methods trained with manual masks. Codes and models are provided in https://vibashan.github.io/ovis-web/.

        ----

        ## [2241] GP-VTON: Towards General Purpose Virtual Try-On via Collaborative Local-Flow Global-Parsing Learning

        **Authors**: *Zhenyu Xie, Zaiyu Huang, Xin Dong, Fuwei Zhao, Haoye Dong, Xijin Zhang, Feida Zhu, Xiaodan Liang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02255](https://doi.org/10.1109/CVPR52729.2023.02255)

        **Abstract**:

        Image-based Virtual Try-ON aims to transfer an in-shop garment onto a specific person. Existing methods employ a global warping module to model the anisotropic deformation for different garment parts, which fails to preserve the semantic information of different parts when receiving challenging inputs (e.g, intricate human poses, difficult garments). Moreover, most of them directly warp the input garment to align with the boundary of the preserved region, which usually requires texture squeezing to meet the boundary shape constraint and thus leads to texture distortion. The above inferior performance hinders existing methods from real-world applications. To address these problems and take a step towards real-world virtual try-on, we propose a General-Purpose Virtual Try-ON framework, named GP-VTON, by developing an innovative Local-Flow Global-Parsing (LFGP) warping module and a Dynamic Gradient Truncation (DGT) training strategy. Specifically, compared with the previous global warping mechanism, LFGP employs local flows to warp garments parts individually, and assembles the local warped results via the global garment parsing, resulting in reasonable warped parts and a semantic-correct intact garment even with challenging inputs. On the other hand, our DGT training strategy dynamically truncates the gradient in the overlap area and the warped garment is no more required to meet the boundary constraint, which effectively avoids the texture squeezing problem. Furthermore, our GP-VTON can be easily extended to multi-category scenario and jointly trained by using data from different garment categories. Extensive experiments on two high-resolution benchmarks demonstrate our superiority over the existing state-of-the-art methods.
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
Code is available at gp-vton.

        ----

        ## [2242] Decomposed Soft Prompt Guided Fusion Enhancing for Compositional Zero-Shot Learning

        **Authors**: *Xiaocheng Lu, Song Guo, Ziming Liu, Jingcai Guo*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02256](https://doi.org/10.1109/CVPR52729.2023.02256)

        **Abstract**:

        Compositional Zero-Shot Learning (CZSL) aims to recognize novel concepts formed by known states and objects during training. Existing methods either learn the combined state-object representation, challenging the generalization of unseen compositions, or design two classifiers to identify state and object separately from image features, ignoring the intrinsic relationship between them. To jointly eliminate the above issues and construct a more robust CZSL system, we propose a novel framework termed Decomposed Fusion with Soft Prompt (DFSP)
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
Code is available at: https://github.corn/Forest-art/DFSP.git, by involving vision-language models (VLMs)for unseen composition recognition. Specifically, DFSP constructs a vector combination of learnable soft prompts with state and object to establish the joint representation of them. In addition, a cross-modal decomposed fusion module is designed between the language and image branches, which decomposes state and object among language features instead of image features. Notably, being fused with the decomposed features, the image features can be more expressive for learning the relationship with states and objects, respectively, to improve the response of unseen compositions in the pair space, hence narrowing the domain gap between seen and unseen sets. Experimental results on three challenging benchmarks demonstrate that our approach significantly outperforms other state-of-the-art methods by large margins.

        ----

        ## [2243] Contrastive Grouping with Transformer for Referring Image Segmentation

        **Authors**: *Jiajin Tang, Ge Zheng, Cheng Shi, Sibei Yang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02257](https://doi.org/10.1109/CVPR52729.2023.02257)

        **Abstract**:

        Referring image segmentation aims to segment the target referent in an image conditioning on a natural language expression. Existing one-stage methods employ per-pixel classification frameworks, which attempt straightforwardly to align vision and language at the pixel level, thus failing to capture critical object-level information. In this paper, we propose a mask classification framework, Contrastive Grouping with Transformer network (CGFormer), which explicitly captures object-level information via token-based querying and grouping strategy. Specifically, CGFormer first introduces learnable query tokens to represent objects and then alternately queries linguistic features and groups visual features into the query tokens for object-aware cross-modal reasoning. In addition, CGFormer achieves cross-level interaction by jointly updating the query tokens and decoding masks in every two consecutive layers. Finally, CGFormer cooperates contrastive learning to the grouping strategy to identify the token and its mask corresponding to the referent. Experimental results demonstrate that CG-Former outperforms state-of-the-art methods in both segmentation and generalization settings consistently and significantly. Code is available at https://github.com/Toneyaya/CGFormer.

        ----

        ## [2244] GRES: Generalized Referring Expression Segmentation

        **Authors**: *Chang Liu, Henghui Ding, Xudong Jiang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02259](https://doi.org/10.1109/CVPR52729.2023.02259)

        **Abstract**:

        Referring Expression Segmentation (RES) aims to generate a segmentation mask for the object described by a given language expression. Existing classic RES datasets and methods commonly support single-target expressions only, i.e., one expression refers to one target object. Multitarget and no-target expressions are not considered. This limits the usage of RES in practice. In this paper, we introduce a new benchmark called Generalized Referring Expression Segmentation (GRES), which extends the classic RES to allow expressions to refer to an arbitrary number of target objects. Towards this, we construct the first largescale GRES dataset called gRefCOCO that contains multitarget, no-target, and single-target expressions. GRES and gRefCOCO are designed to be well-compatible with RES, facilitating extensive experiments to study the performance gap of the existing RES methods on the GRES task. In the experimental study, we find that one of the big challenges of GRES is complex relationship modeling. Based on this, we propose a region-based GRES baseline ReLA that adaptively divides the image into regions with subinstance clues, and explicitly models the region-region and region-language dependencies. The proposed approach ReLA achieves new state-of-the-art performance on the both newly proposed GRES and classic RES tasks. The proposed gRefCOCO dataset and method are available at https://henghuiding.github.io/GRES.

        ----

        ## [2245] Network-Free, Unsupervised Semantic Segmentation with Synthetic Images

        **Authors**: *Qianli Feng, Raghudeep Gadde, Wentong Liao, Eduard Ramon, Aleix Martinez*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02260](https://doi.org/10.1109/CVPR52729.2023.02260)

        **Abstract**:

        We derive a method that yields highly accurate semantic segmentation maps without the use of any additional neural network, layers, manually annotated training data, or supervised training. Our method is based on the observation that the correlation of a set of pixels belonging to the same semantic segment do not change when generating synthetic variants of an image using the style mixing approach in GANs. We show how we can use GAN inversion to accurately semantically segment synthetic and real photos as well as generate large training image-semantic segmentation mask pairs for downstream tasks.

        ----

        ## [2246] Few-shot Semantic Image Synthesis with Class Affinity Transfer

        **Authors**: *Marlène Careil, Jakob Verbeek, Stéphane Lathuilière*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02261](https://doi.org/10.1109/CVPR52729.2023.02261)

        **Abstract**:

        Semantic image synthesis aims to generate photo realistic images given a semantic segmentation map. Despite much recent progress, training them still requires large datasets of images annotated with per-pixel label maps that are extremely tedious to obtain. To alleviate the high annotation cost, we propose a transfer method that leverages a model trained on a large source dataset to improve the learning ability on small target datasets via estimated pairwise relations between source and target classes. The class affinity matrix is introduced as a first layer to the source model to make it compatible with the target label maps, and the source model is then further finetuned for the target domain. To estimate the class affinities we consider different approaches to leverage prior knowledge: semantic segmentation on the source domain, textual label embeddings, and self-supervised vision features. We apply our approach to GAN-based and diffusion-based architectures for semantic synthesis. Our experiments show that the different ways to estimate class affinity can be effectively combined, and that our approach significantly improves over existing state-of-the-art transfer approaches for generative image models.

        ----

        ## [2247] Ultra-High Resolution Segmentation with Ultra-Rich Context: A Novel Benchmark

        **Authors**: *Deyi Ji, Feng Zhao, Hongtao Lu, Mingyuan Tao, Jieping Ye*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02262](https://doi.org/10.1109/CVPR52729.2023.02262)

        **Abstract**:

        With the increasing interest and rapid development of methods for Ultra-High Resolution (UHR) segmentation, a large-scale benchmark covering a wide range of scenes with full fine-grained dense annotations is urgently needed to facilitate the field. To this end, the URUR dataset is introduced, in the meaning of Ultra-High Resolution dataset with Ultra-Rich Context. As the name suggests, URUR contains amounts of images with high enough resolution (3,008 images of size 5,120 × 5,120), a wide range of complex scenes (from 63 cities), rich-enough context (1 million instances with 8 categories) and fine-grained annotations (about 80 billion manually annotated pixels), which is far superior to all the existing UHR datasets including DeepGlobe, Inria Aerial, UDD, etc.. Moreover, we also propose WSDNet, a more efficient and effective framework for UHR segmentation especially with ultra-rich context. Specifically, multi-level Discrete Wavelet Transform (DWT) is naturally integrated to release computation burden while preserve more spatial details, along with a Wavelet Smooth Loss (WSL) to reconstruct original structured context and texture with a smooth constrain. Experiments on several UHR datasets demonstrate its state-of-the-art performance. The dataset is available at https://github.com/jankyee/URUR.

        ----

        ## [2248] Content-aware Token Sharing for Efficient Semantic Segmentation with Vision Transformers

        **Authors**: *Chenyang Lu, Daan de Geus, Gijs Dubbelman*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02263](https://doi.org/10.1109/CVPR52729.2023.02263)

        **Abstract**:

        This paper introduces Content-aware Token Sharing (CTS), a token reduction approach that improves the computational efficiency of semantic segmentation networks that use Vision Transformers (ViTs). Existing works have proposed token reduction approaches to improve the efficiency of ViT-based image classification networks, but these methods are not directly applicable to semantic segmentation, which we address in this work. We observe that, for semantic segmentation, multiple image patches can share a token if they contain the same semantic class, as they contain redundant information. Our approach leverages this by employing an efficient, class-agnostic policy network that predicts if image patches contain the same semantic class, and lets them share a token if they do. With experiments, we explore the critical design choices of CTS and show its effectiveness on the ADE20K, Pascal Context and Cityscapes datasets, various ViT backbones, and different segmentation decoders. With Content-aware Token Sharing, we are able to reduce the number of processed tokens by up to 44%, without diminishing the segmentation quality.

        ----

        ## [2249] Hierarchical Dense Correlation Distillation for Few-Shot Segmentation

        **Authors**: *Bohao Peng, Zhuotao Tian, Xiaoyang Wu, Chengyao Wang, Shu Liu, Jingyong Su, Jiaya Jia*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02264](https://doi.org/10.1109/CVPR52729.2023.02264)

        **Abstract**:

        Few-shot semantic segmentation (FSS) aims to form class-agnostic models segmenting unseen classes with only a handful of annotations. Previous methods limited to the semantic feature and prototype representation suffer from coarse segmentation granularity and train-set overfitting. In this work, we design Hierarchically Decoupled Matching Network (HDMNet) mining pixel-level support correlation based on the transformer architecture. The self-attention modules are used to assist in establishing hierarchical dense features, as a means to accomplish the cascade matching between query and support features. Moreover, we propose a matching module to reduce train-set overfitting and introduce correlation distillation leveraging semantic correspondence from coarse resolution to boost fine-grained segmentation. Our method performs decently in experiments. We achieve 50.0% mIoU on COCO-20
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">i</sup>
 dataset one-shot setting and 56.0% on five-shot segmentation, respectively. The code is available on the project website
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
https://github.com/Pbihao/HDMNet.

        ----

        ## [2250] On Calibrating Semantic Segmentation Models: Analyses and An Algorithm

        **Authors**: *Dongdong Wang, Boqing Gong, Liqiang Wang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02265](https://doi.org/10.1109/CVPR52729.2023.02265)

        **Abstract**:

        We study the problem of semantic segmentation calibration. Lots of solutions have been proposed to approach model miscalibration of confidence in image classification. However, to date, confidence calibration research on se-mantic segmentation is still limited. We provide a system-atic study on the calibration of semantic segmentation models and propose a simple yet effective approach. First, we find that model capacity, crop size, multi-scale testing, and prediction correctness have impact on calibration. Among them, prediction correctness, especially misprediction, is more important to miscalibration due to over-confidence. Next, we propose a simple, unifying, and effective approach, namely selective scaling, by separating correct/incorrect prediction for scaling and more focusing on misprediction logit smoothing. Then, we study popular existing cali-bration methods and compare them with selective scaling on semantic segmentation calibration. We conduct exten-sive experiments with a variety of benchmarks on both in-domain and domain-shift calibration and show that selective scaling consistently outperforms other methods.

        ----

        ## [2251] FastInst: A Simple Query-Based Model for Real-Time Instance Segmentation

        **Authors**: *Junjie He, Pengyu Li, Yifeng Geng, Xuansong Xie*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02266](https://doi.org/10.1109/CVPR52729.2023.02266)

        **Abstract**:

        Recent attention in instance segmentation has focused on query-based models. Despite being non-maximum suppression (NMS)-free and end-to-end, the superiority of these models on high-accuracy real-time benchmarks has not been well demonstrated. In this paper, we show the strong potential of query-based models on efficient instance segmentation algorithm designs. We present FastInst, a simple, effective query-based framework for real-time instance segmentation. FastInst can execute at a real-time speed (i.e., 32.5 FPS) while yielding an AP of more than 40 (i.e., 40.5 AP) on COCO test-dev without bells and whistles. Specifically, FastInst follows the meta-architecture of recently introduced Mask2Former. Its key designs include instance activation-guided queries, dual-path update strategy, and ground truth mask-guided learning, which enable us to use lighter pixel decoders, fewer Transformer decoder layers, while achieving better performance. The experiments show that FastInst outperforms most state-of-the-art real-time counterparts, including strong fully convolutional baselines, in both speed and accuracy. Code can be found at https://github.com/junjiehe96/FastInst.

        ----

        ## [2252] Out-of-Candidate Rectification for Weakly Supervised Semantic Segmentation

        **Authors**: *Zesen Cheng, Pengchong Qiao, Kehan Li, Siheng Li, Pengxu Wei, Xiangyang Ji, Li Yuan, Chang Liu, Jie Chen*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02267](https://doi.org/10.1109/CVPR52729.2023.02267)

        **Abstract**:

        Weakly supervised semantic segmentation is typically inspired by class activation maps, which serve as pseudo masks with class-discriminative regions highlighted. Although tremendous efforts have been made to recall precise and complete locations for each class, existing methods still commonly suffer from the unsolicited Out-of-Candidate (OC) error predictions that do not belong to the label candidates, which could be avoidable since the contradiction with image-level class tags is easy to be detected. In this paper, we develop a group ranking-based Out-of-f;Candidate Rectification (OCR) mechanism in a plug-and-play fashion. Firstly, we adaptively split the semantic categories into In-Candidate (IC) and OC groups for each OC pixel according to their prior annotation correlation and posterior prediction correlation. Then, we derive a differentiable rectification loss to force OC pixels to shift to the IC group. Incorporating OCR with seminal baselines (e.g., AffinityNet, SEAM, MCTformer), we can achieve remarkable performance gains on both Pascal VOC (+3.2%, +3.3%, +0.8% mIoU) and MS COCO (+1.0%, +1.3%, +0.5% mIoU) datasets with negligible extra training overhead, which jus-tifies the effectiveness and generality of OCR. 
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">†</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">†</sup>
Ŋ github.com/sennnnn/Out-of-Candidate-Rectification

        ----

        ## [2253] Foundation Model Drives Weakly Incremental Learning for Semantic Segmentation

        **Authors**: *Chaohui Yu, Qiang Zhou, Jingliang Li, Jianlong Yuan, Zhibin Wang, Fan Wang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02268](https://doi.org/10.1109/CVPR52729.2023.02268)

        **Abstract**:

        Modern incremental learning for semantic segmentation methods usually learn new categories based on dense annotations. Although achieve promising results, pixel-by-pixel labeling is costly and time-consuming. Weakly incremental learning for semantic segmentation (WILSS) is a novel and attractive task, which aims at learning to segment new classes from cheap and widely available image-level labels. Despite the comparable results, the image-level labels can not provide details to locate each segment, which limits the performance of WILSS. This inspires us to think how to improve and effectively utilize the supervision of new classes given image-level labels while avoiding forgetting old ones. In this work, we propose a novel and data-efficient frame-work for WILSS, named FMWISS. Specifically, we propose pre-training based co-segmentation to distill the knowledge of complementary foundation models for generating dense pseudo labels. We further optimize the noisy pseudo masks with a teacher-student architecture, where a plug-in teacher is optimized with a proposed dense contrastive loss. Moreover, we introduce memory-based copy-paste augmentation to improve the catastrophic forgetting problem of old classes. Extensive experiments on Pascal VOC and COCO datasets demonstrate the superior performance of our framework, e.g., FMWISS achieves 70.7% and 73.3% in the 15–5 VOC setting, outperforming the state-of-the-art method by 3.4% and 6.1%, respectively.

        ----

        ## [2254] Long-Tailed Visual Recognition via Self-Heterogeneous Integration with Knowledge Excavation

        **Authors**: *Yan Jin, Mengke Li, Yang Lu, Yiu-ming Cheung, Hanzi Wang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02269](https://doi.org/10.1109/CVPR52729.2023.02269)

        **Abstract**:

        Deep neural networks have made huge progress in the last few decades. However, as the real-world data often exhibits a long-tailed distribution, vanilla deep models tend to be heavily biased toward the majority classes. To address this problem, state-of-the-art methods usually adopt a mixture of experts (MoE) to focus on different parts of the long-tailed distribution. Experts in these methods are with the same model depth, which neglects the fact that different classes may have different preferences to be fit by models with different depths. To this end, we propose a novel MoE-based method called Self-Heterogeneous Integration with Knowledge Excavation (SHIKE). We first propose Depth-wise Knowledge Fusion (DKF) to fuse features between different shallow parts and the deep part in one network for each expert, which makes experts more diverse in terms of representation. Based on DKF, we further propose Dynamic Knowledge Transfer (DKT) to reduce the influence of the hardest negative class that has a non-negligible impact on the tail classes in our MoEframework. As a result, the classification accuracy of long-tailed data can be significantly improved, especially for the tail classes. SHIKE achieves the state-of-the-art performance of 56.3%, 60.3%, 75.4% and 41.9% on CIFAR100-LT (IF100), ImageNet-LT, iNaturalist 2018, and Places-LT, respectively. The source code is available at https://github.com/jinyan-06/SHIKE.

        ----

        ## [2255] Instance-Specific and Model-Adaptive Supervision for Semi-Supervised Semantic Segmentation

        **Authors**: *Zhen Zhao, Sifan Long, Jimin Pi, Jingdong Wang, Luping Zhou*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02270](https://doi.org/10.1109/CVPR52729.2023.02270)

        **Abstract**:

        Recently, semi-supervised semantic segmentation has achieved promising performance with a small fraction of labeled data. However, most existing studies treat all unlabeled data equally and barely consider the differences and training difficulties among unlabeled instances. Differentiating unlabeled instances can promote instance-specific supervision to adapt to the model's evolution dynamically. In this paper, we emphasize the cruciality of instance differences and propose an instance-specific and model-adaptive supervision for semi-supervised semantic segmentation, named iMAS. Relying on the model's performance, iMAS employs a class-weighted symmetric intersection-over-union to evaluate quantitative hardness of each unlabeled instance and supervises the training on unlabeled data in a model-adaptive manner. Specifically, iMAS learns from unlabeled instances progressively by weighing their corresponding consistency losses based on the evaluated hardness. Besides, iMAS dynamically adjusts the augmentation for each instance such that the distortion degree of augmented instances is adapted to the model's generalization capability across the training course. Not integrating additional losses and training procedures, iMAS can obtain remarkable performance gains against current state-of-the-art approaches on segmentation benchmarks under different semi-supervised partition protocols
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
Code and logs: https://github.com/zhenzhao/iMAS.

        ----

        ## [2256] Active Finetuning: Exploiting Annotation Budget in the Pretraining-Finetuning Paradigm

        **Authors**: *Yichen Xie, Han Lu, Junchi Yan, Xiaokang Yang, Masayoshi Tomizuka, Wei Zhan*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02271](https://doi.org/10.1109/CVPR52729.2023.02271)

        **Abstract**:

        Given the large-scale data and the high annotation cost, pretraining-finetuning becomes a popular paradigm in multiple computer vision tasks. Previous research has covered both the unsupervised pretraining and supervised finetuning in this paradigm, while little attention is paid to exploiting the annotation budget for finetuning. To fill in this gap, we formally define this new active finetuning task focusing on the selection of samples for annotation in the pretraining-finetuning paradigm. We propose a novel method called ActiveFT for active finetuning task to select a subset of data distributing similarly with the entire unlabeled pool and maintaining enough diversity by optimizing a parametric model in the continuous space. We prove that the Earth Mover's distance between the distributions of the selected subset and the entire data pool is also reduced in this process. Extensive experiments show the leading performance and high efficiency of ActiveFT superior to baselines on both image classification and semantic segmentation. Our code is released at https://github.com/yichen928/ActiveFT.

        ----

        ## [2257] IDGI: A Framework to Eliminate Explanation Noise from Integrated Gradients

        **Authors**: *Ruo Yang, Binghui Wang, Mustafa Bilgic*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02272](https://doi.org/10.1109/CVPR52729.2023.02272)

        **Abstract**:

        Integrated Gradients (IG) as well as its variants are well-known techniques for interpreting the decisions of deep neural networks. While IG-based approaches attain state-of-the-art performance, they often integrate noise into their explanation saliency maps, which reduce their interpretability. To minimize the noise, we examine the source of the noise analytically and propose a new approach to reduce the explanation noise based on our analytical findings. We propose the Important Direction Gradient Integration (IDGI) framework, which can be easily incorporated into any IG-based method that uses the Reimann Integration for integrated gradient computation. Extensive experiments with three IG-based methods show that IDGI improves them drastically on numerous interpretability metrics. The source code for IDGI is available at https://github.com/yangruo1226/IDGI.

        ----

        ## [2258] Weakly Supervised Posture Mining for Fine-Grained Classification

        **Authors**: *Zhenchao Tang, Hualin Yang, Calvin Yu-Chian Chen*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02273](https://doi.org/10.1109/CVPR52729.2023.02273)

        **Abstract**:

        Because the subtle differences between the different sub-categories of common visual categories such as bird species, fine-grained classification has been seen as a challenging task for many years. Most previous works focus towards the features in the single discriminative region isolatedly, while neglect the connection between the different discriminative regions in the whole image. However, the relationship between different discriminative regions contains rich posture information and by adding the posture information, model can learn the behavior of the object which attribute to improve the classification performance. In this paper, we propose a novel fine-grained framework named PMRC (posture mining and reverse cross-entropy), which is able to combine with different backbones to good effect. In PMRC, we use the Deep Navigator to generate the discriminative regions from the images, and then use them to construct the graph. We aggregate the graph by message passing and get the classification results. Specifically, in order to force PMRC to learn how to mine the posture information, we design a novel training paradigm, which makes the Deep Navigator and message passing communicate and train together. In addition, we propose the reverse cross-entropy (RCE) and demomenstate that compared to the cross-entropy (CE), RCE can not only promote the accurracy of our model but also generalize to promote the accuracy of other kinds of fine-grained classification models. Experimental results on benchmark datasets confirm that PMRC can achieve state-of-the-art.

        ----

        ## [2259] Vision Transformers are Good Mask Auto-Labelers

        **Authors**: *Shiyi Lan, Xitong Yang, Zhiding Yu, Zuxuan Wu, José M. Álvarez, Anima Anandkumar*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02274](https://doi.org/10.1109/CVPR52729.2023.02274)

        **Abstract**:

        We propose Mask Auto-Labeler (MAL), a high-quality Transformer-based mask auto-labeling framework for instance segmentation using only box annotations. MAL takes box-cropped images as inputs and conditionally generates their mask pseudo-labels. We show that Vision Transformers are good mask auto-labelers. Our method significantly reduces the gap between auto-labeling and human annotation regarding mask quality. Instance segmentation models trained using the MAL-generated masks can nearly match the performance of their fully-supervised counterparts, retaining up to 97.4% performance of fully supervised models. The best model achieves 44.1% mAP on COCO instance segmentation (test-dev 2017), outperforming state-of-the-art box-supervised methods by significant margins. Qualitative results indicate that masks produced by MAL are, in some cases, even better than human annotations.

        ----

        ## [2260] Enhanced Training of Query-Based Object Detection via Selective Query Recollection

        **Authors**: *Fangyi Chen, Han Zhang, Kai Hu, Yu-Kai Huang, Chenchen Zhu, Marios Savvides*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02275](https://doi.org/10.1109/CVPR52729.2023.02275)

        **Abstract**:

        This paper investigates a phenomenon where query-based object detectors mispredict at the last decoding stage while predicting correctly at an intermediate stage. We review the training process and attribute the overlooked phenomenon to two limitations: lack of training emphasis and cascading errors from decoding sequence. We design and present Selective Query Recollection (SQR), a simple and effective training strategy for query-based object detectors. It cumulatively collects intermediate queries as decoding stages go deeper and selectively forwards the queries to the downstream stages aside from the sequential structure. Such-wise, SQR places training emphasis on later stages and allows later stages to work with intermediate queries from earlier stages directly. SQR can be easily plugged into various query-based object detectors and significantly enhances their performance while leaving the inference pipeline unchanged. As a result, we apply SQR on Adamixer, DAB-DETR, and Deformable-DETR across various settings (backbone, number of queries, schedule) and consistently brings 1.4 ~2.8 AP improvement. Code is available at https://github.com/Fangyi-Chen/SQR

        ----

        ## [2261] Box-Level Active Detection

        **Authors**: *Mengyao Lyu, Jundong Zhou, Hui Chen, Yijie Huang, Dongdong Yu, Yaqian Li, Yandong Guo, Yuchen Guo, Liuyu Xiang, Guiguang Ding*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02276](https://doi.org/10.1109/CVPR52729.2023.02276)

        **Abstract**:

        Active learning selects informative samples for annotation within budget, which has proven efficient recently on object detection. However, the widely used active detection benchmarks conduct image-level evaluation, which is unrealistic in human workload estimation and biased towards crowded images. Furthermore, existing methods still perform image-level annotation, but equally scoring all targets within the same image incurs waste of budget and redundant labels. Having revealed above problems and limitations, we introduce a box-level active detection framework that controls a box-based budget per cycle, prioritizes informative targets and avoids redundancy for fair comparison and efficient application. Under the proposed box-level setting, we devise a novel pipeline, namely Complementary Pseudo Active Strategy (ComPAS). It exploits both human annotations and the model intelligence in a complementary fashion: an efficient input-end committee queries labels for informative objects only; meantime well-learned targets are identified by the model and compensated with pseudo-labels. ComPAS consistently outperforms 10 competitors under 4 settings in a unified codebase. With supervision from labeled data only, it achieves 100% supervised performance of VOC0712 with merely 19% box annotations. On the COCO dataset, it yields up to 4.3% mAP improvement over the second-best method. ComPAS also supports training with the unlabeled pool, where it surpasses 90% COCO supervised performance with 85% label reduction. Our source code is publicly available at https://github.com/lyumengyao/blad.

        ----

        ## [2262] CIGAR: Cross-Modality Graph Reasoning for Domain Adaptive Object Detection

        **Authors**: *Yabo Liu, Jinghua Wang, Chao Huang, Yaowei Wang, Yong Xu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02277](https://doi.org/10.1109/CVPR52729.2023.02277)

        **Abstract**:

        Unsupervised domain adaptive object detection (UDAOD) aims to learn a detector by generalizing knowledge from a labeled source domain to an unlabeled target domain. Though the existing graph-based methods for UDAOD perform well in some cases, they cannot learn a proper node set for the graph. In addition, these methods build the graph solely based on the visual features and do not consider the linguistic knowledge carried by the semantic prototypes, e.g., dataset labels. To overcome these problems, we propose a cross-modality graph reasoning adaptation (CIGAR) method to take advantage of both visual and linguistic knowledge. Specifically, our method performs cross-modality graph reasoning between the linguistic modality graph and visual modality graphs to enhance their representations. We also propose a discriminative feature selector to find the most discriminative features and take them as the nodes of the visual graph for both efficiency and effectiveness. In addition, we employ the linguistic graph matching loss to regulate the update of linguistic graphs and maintain their semantic representation during the training process. Comprehensive experiments validate the effectiveness of our proposed CIGAR.

        ----

        ## [2263] DA-DETR: Domain Adaptive Detection Transformer with Information Fusion

        **Authors**: *Jingyi Zhang, Jiaxing Huang, Zhipeng Luo, Gongjie Zhang, Xiaoqin Zhang, Shijian Lu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02278](https://doi.org/10.1109/CVPR52729.2023.02278)

        **Abstract**:

        The recent detection transformer (DETR) simplifies the object detection pipeline by removing hand-crafted designs and hyperparameters as employed in conventional two-stage object detectors. However, how to leverage the simple yet effective DETR architecture in domain adaptive object detection is largely neglected. Inspired by the unique DETR attention mechanisms, we design DA-DETR, a domain adaptive object detection transformer that introduces information fusion for effective transfer from a labeled source domain to an unlabeled target domain. DA-DETR introduces a novel CNN-Transformer Blender (CTBlender) that fuses the CNN features and Transformer features ingeniously for effective feature alignment and knowledge transfer across domains. Specifically, CTBlender employs the Transformer features to modulate the CNN features across multiple scales where the high-level semantic information and the low-level spatial information are fused for accurate object identification and localization. Extensive experiments show that DA-DETR achieves superior detection performance consistently across multiple widely adopted domain adaptation benchmarks.

        ----

        ## [2264] Continual Detection Transformer for Incremental Object Detection

        **Authors**: *Yaoyao Liu, Bernt Schiele, Andrea Vedaldi, Christian Rupprecht*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02279](https://doi.org/10.1109/CVPR52729.2023.02279)

        **Abstract**:

        Incremental object detection (IOD) aims to train an object detector in phases, each with annotations for new object categories. As other incremental settings, IOD is subject to catastrophic forgetting, which is often addressed by techniques such as knowledge distillation (KD) and exem-plar replay (ER). However, KD and ER do not work well if applied directly to state-of-the-art transformer-based object detectors such as Deformable DETR [59] and UP-DETR [9]. In this paper, we solve these issues by proposing a ContinuaL DEtection TRansformer (CL-DETR), a new method for transformer-based IOD which enables effective usage of KD and ER in this context. First, we introduce a Detector Knowledge Distillation (DKD) loss, focusing on the most informative and reliable predictions from old versions of the model, ignoring redundant background predictions, and ensuring compatibility with the available ground-truth labels. We also improve ER by proposing a calibration strategy to preserve the label distribution of the training set, therefore better matching training and testing statistics. We conduct extensive experiments on COCO 2017 and demonstrate that CL-DETR achieves state-of-the-art results in the IOD setting.
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
Code: https://1yy.mpi-inf.mpg.de/CL-DETR/

        ----

        ## [2265] Semi-DETR: Semi-Supervised Object Detection with Detection Transformers

        **Authors**: *Jiacheng Zhang, Xiangru Lin, Wei Zhang, Kuo Wang, Xiao Tan, Junyu Han, Errui Ding, Jingdong Wang, Guanbin Li*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02280](https://doi.org/10.1109/CVPR52729.2023.02280)

        **Abstract**:

        We analyze the DETR-based framework on semi-supervised object detection (SSOD) and observe that (1) the one-to-one assignment strategy generates incorrect matching when the pseudo ground-truth bounding box is inaccurate, leading to training inefficiency; (2) DETR-based detectors lack deterministic correspondence between the input query and its prediction output, which hinders the applicability of the consistency-based regularization widely used in current SSOD methods. We present Semi-DETR, the first transformer-based end-to-end semi-supervised object detector, to tackle these problems. Specifically, we propose a Stage-wise Hybrid Matching strategy that combines the one-to-many assignment and one-to-one assignment strategies to improve the training efficiency of the first stage and thus provide high-quality pseudo labels for the training of the second stage. Besides, we introduce a Cross-view Query Consistency method to learn the semantic feature invariance of object queries from different views while avoiding the need to find deterministic query correspondence. Furthermore, we propose a Cost-based Pseudo Label Mining module to dynamically mine more pseudo boxes based on the matching cost of pseudo ground truth bounding boxes for consistency training. Extensive experiments on all SSOD settings of both COCO and Pascal VOC benchmark datasets show that our Semi-DETR method outperforms all state-of-the-art methods by clear margins.

        ----

        ## [2266] Hierarchical Supervision and Shuffle Data Augmentation for 3D Semi-Supervised Object Detection

        **Authors**: *Chuandong Liu, Chenqiang Gao, Fangcen Liu, Pengcheng Li, Deyu Meng, Xinbo Gao*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02281](https://doi.org/10.1109/CVPR52729.2023.02281)

        **Abstract**:

        State-of-the-art 3D object detectors are usually trained on large-scale datasets with high-quality 3D annotations. However, such 3D annotations are often expensive and time-consuming, which may not be practical for real applications. A natural remedy is to adopt semi-supervised learning (SSL) by leveraging a limited amount of labeled samples and abundant unlabeled samples. Current pseudo-labeling-based SSL object detection methods mainly adopt a teacher-student framework, with a single fixed threshold strategy to generate supervision signals, which inevitably brings confused supervision when guiding the student network training. Besides, the data augmentation of the point cloud in the typical teacher-student framework is too weak, and only contains basic down sampling and flip-and-shift (i.e., rotate and scaling), which hinders the effective learning of feature information. Hence, we address these issues by introducing a novel approach of Hierarchical Supervision and Shuffle Data Augmentation (HSSDA), which is a simple yet effective teacher-student framework. The teacher network generates more reasonable supervision for the student network by designing a dynamic dual-threshold strategy. Besides, the shuffle data augmentation strategy is designed to strengthen the feature representation ability of the student network. Extensive experiments show that HSSDA consistently outperforms the recent state-of-the-art methods on different datasets. The code will be released at https://github.com/azhuantou/HSSDA.

        ----

        ## [2267] Harmonious Teacher for Cross-Domain Object Detection

        **Authors**: *Jinhong Deng, Dongli Xu, Wen Li, Lixin Duan*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02282](https://doi.org/10.1109/CVPR52729.2023.02282)

        **Abstract**:

        Self-training approaches recently achieved promising results in cross-domain object detection, where people iteratively generate pseudo labels for unlabeled target domain samples with a model, and select high-confidence samples to refine the model. In this work, we reveal that the consistency of classification and localization predictions are crucial to measure the quality of pseudo labels, and propose a new Harmonious Teacher approach to improve the self-training for cross-domain object detection. In particular, we first propose to enhance the quality of pseudo labels by regularizing the consistency of the classification and localization scores when training the detection model. The consistency losses are defined for both labeled source samples and the unlabeled target samples. Then, we further remold the traditional sample selection method by a sample reweighing strategy based on the consistency of classification and localization scores to improve the ranking of predictions. This allows us to fully exploit all instance predictions from the target domain without abandoning valuable hard examples. Without bells and whistles, our method shows superior performance in various cross-domain scenarios compared with the state-of-the-art baselines, which validates the effectiveness of our Harmonious Teacher. Our codes will be available at https://github.com/kinredon/Harmonious-Teacher.

        ----

        ## [2268] Contrastive Mean Teacher for Domain Adaptive Object Detectors

        **Authors**: *Shengcao Cao, Dhiraj Joshi, Liang-Yan Gui, Yu-Xiong Wang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02283](https://doi.org/10.1109/CVPR52729.2023.02283)

        **Abstract**:

        Object detectors often suffer from the domain gap between training (source domain) and real-world applications (target domain). Mean-teacher self-training is a powerful paradigm in unsupervised domain adaptation for object detection, but it struggles with low-quality pseudo-labels. In this work, we identify the intriguing alignment and synergy between mean-teacher self-training and contrastive learning. Motivated by this, we propose Contrastive Mean Teacher (CMT) - a unified, general-purpose framework with the two paradigms naturally integrated to maximize beneficial learning signals. Instead of using pseudo-labels solely for final predictions, our strategy extracts object-level features using pseudo-labels and optimizes them via contrastive learning, without requiring labels in the target domain. When combined with recent mean-teacher self-training methods, CMT leads to new state-of-the-art target-domain performance: 51.9% mAP on Foggy Cityscapes, outperforming the previously best by 2.1% mAP. Notably, CMT can stabilize performance and provide more significant gains as pseudo-label noise increases.

        ----

        ## [2269] Out-of-Distributed Semantic Pruning for Robust Semi-Supervised Learning

        **Authors**: *Yu Wang, Pengchong Qiao, Chang Liu, Guoli Song, Xiawu Zheng, Jie Chen*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02284](https://doi.org/10.1109/CVPR52729.2023.02284)

        **Abstract**:

        Recent advances in robust semi-supervised learning (SSL) typically filter out-of-distribution (OOD) information at the sample level. We argue that an overlooked problem of robust SSL is its corrupted information on semantic level, practically limiting the development of the field. In this paper, we take an initial step to explore and propose a unified framework termed OOD Semantic Pruning (OSP), which aims at pruning OOD semantics out from in-distribution (ID) features. Specifically, (i) we propose an aliasing OOD matching module to pair each ID sample with an OOD sample with semantic overlap. (ii) We design a soft orthogonality regularization, which first transforms each ID feature by suppressing its semantic component that is collinear with paired OOD sample. It then forces the predictions before and after soft orthogonality decomposition to be consistent. Being practically simple, our method shows a strong performance in OOD detection and ID classification on challenging benchmarks. In particular, OSP surpasses the previous state-of-the-art by 13.7% on accuracy for ID classification and 5.9% on AUROC for OOD detection on TinyImageNet dataset. The source codes are publicly available at https://github.com/rain305f/OSP.

        ----

        ## [2270] (ML)2P-Encoder: On Exploration of Channel-Class Correlation for Multi-Label Zero-Shot Learning

        **Authors**: *Ziming Liu, Song Guo, Xiaocheng Lu, Jingcai Guo, Jiewei Zhang, Yue Zeng, Fushuo Huo*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02285](https://doi.org/10.1109/CVPR52729.2023.02285)

        **Abstract**:

        Recent studies usually approach multi-label zeroshot learning (MLZSL) with visual-semantic mapping on spatial-class correlation, which can be computationally costly, and worse still, fails to capture fine-grained classspecific semantics. We observe that different channels may usually have different sensitivities on classes, which can correspond to specific semantics. Such an intrinsic channelclass correlation suggests a potential alternative for the more accurate and class-harmonious feature representations. In this paper, our interest is to fully explore the power of channel-class correlation as the unique base for MLZSL. Specifically, we propose a light yet efficient Multi-Label Multi-Layer Perceptron-based Encoder, dubbed (ML)
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">2</sup>
P-Encoder, to extract and preserve channel-wise semantics. We reorganize the generated feature maps into several groups, of which each of them can be trained independently with (ML)
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">2</sup>
P-Encoder. On top of that, a global groupwise attention module is further designed to build the multilabel specific class relationships among different classes, which eventually fulfills a novel Channel-Class Correlation MLZSL framework (C
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">3</sup>
-MLZSL)
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
Released code:github.com/simonzmliu/cvpr23_mlzsl. Extensive experiments on large-scale MLZSL benchmarks including NUS-WIDE and Open-Images-V4 demonstrate the superiority of our model against other representative state-of-the-art models.

        ----

        ## [2271] MagicNet: Semi-Supervised Multi-Organ Segmentation via Magic-Cube Partition and Recovery

        **Authors**: *Duowen Chen, Yunhao Bai, Wei Shen, Qingli Li, Lequan Yu, Yan Wang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02286](https://doi.org/10.1109/CVPR52729.2023.02286)

        **Abstract**:

        We propose a novel teacher-student model for semi-supervised multi-organ segmentation. In teacher-student model, data augmentation is usually adopted on unlabeled data to regularize the consistent training between teacher and student. We start from a key perspective that fixed relative locationsand variable sizes of different organs can provide distribution information where a multi-organ CT scan is drawn. Thus, we treat the prior anatomy as a strong tool to guide the data augmentation and reduce the mismatch between labeled and unlabeled images for semi-supervised learning. More specifically, we propose a data augmentation strategy based on partition-and-recovery N
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">3</sup>
 cubes cross-and within-labeled and unlabeled images. Our strategy encourages unlabeled images to learn organ semantics in relative locations from the labeled images (cross-branch) and enhances the learning ability for small organs (within-branch). For within-branch, we further propose to refine the quality of pseudo labels by blending the learned representations from small cubes to incorporate local attributes. Our method is termed as MagicNet, since it treats the CT volume as a magic-cube and N
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">3</sup>
-cube partition-and-recovery process matches with the rule of playing a magic-cube. Extensive experiments on two public CT multi-organ datasets demonstrate the effectiveness of MagicNet, and noticeably outperforms state-of-the-art semi-supervised medical image segmentation approaches, with + 7% DSC improvement on MACT dataset with 10% labeled images. Code is avaiable at https://github.com/DeepMed-Lab-ECNU/MagicNet.

        ----

        ## [2272] Devil is in the Queries: Advancing Mask Transformers for Real-world Medical Image Segmentation and Out-of-Distribution Localization

        **Authors**: *Mingze Yuan, Yingda Xia, Hexin Dong, Zifan Chen, Jiawen Yao, Mingyan Qiu, Ke Yan, Xiaoli Yin, Yu Shi, Xin Chen, Zaiyi Liu, Bin Dong, Jingren Zhou, Le Lu, Ling Zhang, Li Zhang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02287](https://doi.org/10.1109/CVPR52729.2023.02287)

        **Abstract**:

        Real-world medical image segmentation has tremendous long-tailed complexity of objects, among which tail conditions correlate with relatively rare diseases and are clinically significant. A trustworthy medical AI algorithm should demonstrate its effectiveness on tail conditions to avoid clinically dangerous damage in these out-of-distribution (OOD) cases. In this paper, we adopt the concept of object queries in Mask Transformers to formulate semantic segmentation as a soft cluster assignment. The queries fit the feature-level cluster centers of inliers during training. Therefore, when performing inference on a medical image in real-world scenarios, the similarity between pixels and the queries detects and localizes OOD regions. We term this OOD localization as MaxQuery. Furthermore, the foregrounds of real-world medical images, whether OOD objects or inliers, are lesions. The difference between them is less than that between the foreground and background, possibly misleading the object queries to focus redundantly on the background. Thus, we propose a query-distribution (QD) loss to enforce clear boundaries between segmentation targets and other regions at the query level, improving the inlier segmentation and OOD indication. Our proposed framework is tested on two real-world segmentation tasks, i.e., segmentation of pancreatic and liver tumors, outperforming previous state-of-the-art algorithms by an average of 7.39% on AUROC, 14.69% on AUPR, and 13.79% on FPR95 for OOD localization. On the other hand, our framework improves the performance of inlier segmentation by an average of 5.27% DSC when compared with the leading baseline nnUNet.

        ----

        ## [2273] SQUID: Deep Feature In-Painting for Unsupervised Anomaly Detection

        **Authors**: *Tiange Xiang, Yixiao Zhang, Yongyi Lu, Alan L. Yuille, Chaoyi Zhang, Weidong Cai, Zongwei Zhou*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02288](https://doi.org/10.1109/CVPR52729.2023.02288)

        **Abstract**:

        Radiography imaging protocols focus on particular body regions, therefore producing images of great similarity and yielding recurrent anatomical structures across patients. To exploit this structured information, we propose the use of Space-aware Memory Queues for In-painting and Detecting anomalies from radiography images (abbreviated as SQUID). We show that SQUID can taxonomize the ingrained anatomical structures into recurrent patterns; and in the inference, it can identify anomalies (unseen/modified patterns) in the image. SQUID surpasses 13 state-of-the-art methods in unsupervised anomaly detection by at least 5 points on two chest X-ray benchmark datasets measured by the Area Under the Curve (AUC). Additionally, we have created a new dataset (DigitAnatomy), which synthesizes the spatial correlation and consistent shape in chest anatomy. We hope DigitAnatomy can prompt the development, evaluation, and interpretability of anomaly detection methods.

        ----

        ## [2274] OCELOT: Overlapped Cell on Tissue Dataset for Histopathology

        **Authors**: *Jeongun Ryu, Aaron Valero Puche, Jaewoong Shin, Seonwook Park, Biagio Brattoli, Jinhee Lee, Wonkyung Jung, Soo Ick Cho, Kyunghyun Paeng, Chan-Young Ock, Donggeun Yoo, Sérgio Pereira*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02289](https://doi.org/10.1109/CVPR52729.2023.02289)

        **Abstract**:

        Cell detection is a fundamental task in computational pathology that can be used for extracting high-level medical information from whole-slide images. For accurate cell detection, pathologists often zoom out to understand the tissue-level structures and zoom in to classify cells based on their morphology and the surrounding context. However, there is a lack of efforts to reflect such behaviors by pathologists in the cell detection models, mainly due to the lack of datasets containing both cell and tissue annotations with overlapping regions. To overcome this limitation, we propose and publicly release OCELOT, a dataset purposely dedicated to the study of cell-tissue relationships for cell detection in histopathology. OCELOT provides overlapping cell and tissue annotations on images acquired from multiple organs. Within this setting, we also propose multi-task learning approaches that benefit from learning both cell and tissue tasks simultaneously. When compared against a model trained only for the cell detection task, our proposed approaches improve cell detection performance on 3 datasets: proposed OCELOT, public TIGER, and internal CARP datasets. On the OCELOT test set in particular, we show up to 6.79 improvement in F1-score. We believe the contributions of this paper, including the release of the OCELOT dataset at https://lunit-io.github.io/research/publications/OCELOT are a crucial starting point toward the important research direction of incorporating cell-tissue relationships in computation pathology.

        ----

        ## [2275] DeGPR: Deep Guided Posterior Regularization for Multi-Class Cell Detection and Counting

        **Authors**: *Aayush Kumar Tyagi, Chirag Mohapatra, Prasenjit Das, Govind Makharia, Lalita Mehra, Prathosh AP, Mausam*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02290](https://doi.org/10.1109/CVPR52729.2023.02290)

        **Abstract**:

        Multi-class cell detection and counting is an essential task for many pathological diagnoses. Manual counting is tedious and often leads to inter-observer variations among pathologists. While there exist multiple, general-purpose, deep learning-based object detection and counting methods, they may not readily transfer to detecting and counting cells in medical images, due to the limited data, presence of tiny overlapping objects, multiple cell types, severe class-imbalance, minute differences in size/shape of cells, etc. In response, we propose guided posterior regularization (DEGPR), which assists an object detector by guiding it to exploit discriminative features among cells. The features may be pathologist-provided or inferred directly from visual data. We validate our model on two publicly available datasets (CoNSeP and MoNuSAC), and on MuCeD, a novel dataset that we contribute. MuCeD consists of 55 biopsy images of the human duodenum for predicting celiac disease. We perform extensive experimentation with three object detection baselines on three datasets to show that DeGPR is model-agnostic, and consistently improves baselines obtaining up to 9% (absolute) mAP gains.

        ----

        ## [2276] Best of Both Worlds: Multimodal Contrastive Learning with Tabular and Imaging Data

        **Authors**: *Paul Hager, Martin J. Menten, Daniel Rueckert*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02291](https://doi.org/10.1109/CVPR52729.2023.02291)

        **Abstract**:

        Medical datasets and especially biobanks, often contain extensive tabular data with rich clinical information in addition to images. In practice, clinicians typically have less data, both in terms of diversity and scale, but still wish to deploy deep learning solutions. Combined with increasing medical dataset sizes and expensive annotation costs, the necessity for unsupervised methods that can pretrain multimodally and predict unimodally has risen. To address these needs, we propose the first self-supervised contrastive learning framework that takes advantage of images and tabular data to train unimodal encoders. Our solution combines SimCLR and SCARF, two leading contrastive learning strategies, and is simple and effective. In our experiments, we demonstrate the strength of our framework by predicting risks of myocardial infarction and coronary artery disease (CAD) using cardiac MR images and 120 clinical features from 40,000 UK Biobank subjects. Furthermore, we show the generalizability of our approach to natural images using the DVM car advertisement dataset. We take advantage of the high interpretability of tabular data and through attribution and ablation experiments find that morphometric tabular features, describing size and shape, have outsized importance during the contrastive learning process and improve the quality of the learned embeddings. Finally, we introduce a novel form of supervised contrastive learning, label as a feature (LaaF), by appending the ground truth label as a tabular feature during multimodal pretraining, outperforming all supervised contrastive baselines.
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
https://github.com/paulhager/MMCL-Tabular-Imaging

        ----

        ## [2277] RankMix: Data Augmentation for Weakly Supervised Learning of Classifying Whole Slide Images with Diverse Sizes and Imbalanced Categories

        **Authors**: *Yuan-Chih Chen, Chun-Shien Lu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02292](https://doi.org/10.1109/CVPR52729.2023.02292)

        **Abstract**:

        Whole Slide Images (WSIs) are usually gigapixel in size and lack pixel-level annotations. The WSI datasets are also imbalanced in categories. These unique characteristics, significantly different from the ones in natural images, pose the challenge of classifying WSI images as a kind of weakly supervise learning problems. In this study, we propose, RankMix, a data augmentation method of mixing ranked features in a pair of WSIs. RankMix introduces the concepts of pseudo labeling and ranking in order to extract key WSI regions in contributing to the WSI classification task. A two-stage training is further proposed to boost stable training and model performance. To our knowledge, the study of weakly supervised learning from the perspective of data augmentation to deal with the WSI classification problem that suffers from lack of training data and imbalance of categories is relatively un-explored.

        ----

        ## [2278] GEN: Pushing the Limits of Softmax-Based Out-of-Distribution Detection

        **Authors**: *Xixi Liu, Yaroslava Lochman, Christopher Zach*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02293](https://doi.org/10.1109/CVPR52729.2023.02293)

        **Abstract**:

        Out-of-distribution (OOD) detection has been exten-sively studied in order to successfully deploy neural networks, in particular, for safety-critical applications. More-over, performing OOD detection on large-scale datasets is closer to reality, but is also more challenging. Sev-eral approaches need to either access the training data for score design or expose models to outliers during training. Some post-hoc methods are able to avoid the afore-mentioned constraints, but are less competitive. In this work, we propose Generalized ENtropy score (GEN), a simple but effective entropy-based score function, which can be applied to any pre-trained softmax-based classifier. Its performance is demonstrated on the large-scale ImageNet-lk OOD detection benchmark. It consistently improves the average AUROC across six commonly-used CNN-based and visual transformer classifiers over a num-ber of state-of-the-art post-hoc methods. The average AU- ROC improvement is at least 3.5%. Furthermore, we used GEN on top of feature-based enhancing methods as well as methods using training statistics to further improve the OOD detection performance. The code is available at: https://github.com/XixiLiu95/GEN.

        ----

        ## [2279] Discriminating Known from Unknown Objects via Structure-Enhanced Recurrent Variational AutoEncoder

        **Authors**: *Aming Wu, Cheng Deng*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02294](https://doi.org/10.1109/CVPR52729.2023.02294)

        **Abstract**:

        Discriminating known from unknown objects is an important essential ability for human beings. To simulate this ability, a task of unsupervised out-of-distribution object detection (OOD-OD) is proposed to detect the objects that are never-seen-before during model training, which is beneficial for promoting the safe deployment of object detectors. Due to lacking unknown data for supervision, for this task, the main challenge lies in how to leverage the known in-distribution (ID) data to improve the detector's discrimination ability. In this paper, we first propose a method of Structure-Enhanced Recurrent Variational AutoEncoder (SR-VAE), which mainly consists of two dedicated recurrent VAE branches. Specifically, to boost the performance of object localization, we explore utilizing the classical Laplacian of Gaussian (LoG) operator to enhance the structure information in the extracted low-level features. Meanwhile, we design a VAE branch that recurrently generates the augmentation of the classification features to strengthen the discrimination ability of the object classifier. Finally, to alleviate the impact of lacking unknown data, another cycle-consistent conditional VAE branch is proposed to synthesize virtual OOD features that deviate from the distribution of ID features, which improves the capability of distinguishing OOD objects. In the experiments, our method is evaluated on OOD-OD, open-vocabulary detection, and incremental object detection. The significant performance gains over baselines show the superiorities of our method. The code will be released at https://github.com/AmingWu/SR-VAE.

        ----

        ## [2280] Sample-level Multi-view Graph Clustering

        **Authors**: *Yuze Tan, Yixi Liu, Shudong Huang, Wentao Feng, Jiancheng Lv*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02295](https://doi.org/10.1109/CVPR52729.2023.02295)

        **Abstract**:

        Multi-view clustering has hitherto been studied due to their effectiveness in dealing with heterogeneous data. Despite the empirical success made by recent works, there still exists several severe challenges. Particularly, previous multi-view clustering algorithms seldom consider the topological structure in data, which is essential for clustering data on manifold. Moreover, existing methods cannot fully explore the consistency of local structures between different views as they uncover the clustering structure in a intra-view way instead of a inter-view manner. In this paper, we propose to exploit the implied data manifold by learning the topological structure of data. Besides, considering that the consistency of multiple views is manifested in the generally similar local structure while the inconsistent structures are the minority, we further explore the intersections of multiple views in the sample level such that the cross-view consistency can be better maintained. We model the above concerns in a unified framework and design an efficient algorithm to solve the corresponding optimization problem. Experimental results on various multi-view datasets certificate the effectiveness of the proposed method and verify its superiority over other SOTA approaches.

        ----

        ## [2281] On the Effects of Self-supervision and Contrastive Alignment in Deep Multi-view Clustering

        **Authors**: *Daniel J. Trosten, Sigurd Løkse, Robert Jenssen, Michael C. Kampffmeyer*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02296](https://doi.org/10.1109/CVPR52729.2023.02296)

        **Abstract**:

        Self-supervised learning is a central component in recent approaches to deep multi-view clustering (MVC). However, we find large variations in the development of self-supervision-based methods for deep MVC, potentially slowing the progress of the field. To address this, we present Deep-MVC, a unified framework for deep MVC that includes many recent methods as instances. We leverage our framework to make key observations about the effect of self-supervision, and in particular, drawbacks of aligning representations with contrastive learning. Further, we prove that contrastive alignment can negatively influence cluster separability, and that this effect becomes worse when the number of views increases. Motivated by our findings, we develop several new DeepMVC instances with new forms of self-supervision. We conduct extensive experiments and find that (i) in line with our theoretical findings, contrastive alignments decreases performance on datasets with many views; (ii) all methods benefit from some form of self-supervision; and (iii) our new instances outperform previous methods on several datasets. Based on our results, we suggest several promising directions for future research. To enhance the openness of the field, we provide an open-source implementation of DeepMVC, including recent models and our new instances. Our implementation includes a consistent evaluation protocol, facilitating fair and accurate evaluation of methods and components
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
Code: https://github.com/DanielTrosten/DeepMVC.

        ----

        ## [2282] Deep Fair Clustering via Maximizing and Minimizing Mutual Information: Theory, Algorithm and Metric

        **Authors**: *Pengxin Zeng, Yunfan Li, Peng Hu, Dezhong Peng, Jiancheng Lv, Xi Peng*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02297](https://doi.org/10.1109/CVPR52729.2023.02297)

        **Abstract**:

        Fair clustering aims to divide data into distinct clusters while preventing sensitive attributes (e.g., gender, race, RNA sequencing technique) from dominating the clustering. Although a number of works have been conducted and achieved huge success recently, most of them are heuristical, and there lacks a unified theory for algorithm design. In this work, we fill this blank by developing a mutual information theory for deep fair clustering and accordingly designing a novel algorithm, dubbed FCMI. In brief, through maximizing and minimizing mutual information, FCMI is designed to achieve four characteristics highly expected by deep fair clustering, i.e., compact, balanced, and fair clusters, as well as informative features. Besides the contributions to theory and algorithm, another contribution of this work is proposing a novel fair clustering metric built upon information theory as well. Unlike existing evaluation metrics, our metric measures the clustering quality and fairness as a whole instead of separate manner. To verify the effectiveness of the proposed FCMI, we conduct experiments on six benchmarks including a single-cell RNA-seq atlas compared with 11 state-of-the-art methods in terms of five metrics. The code could be accessed from https://pengxi.me.

        ----

        ## [2283] Transductive Few-Shot Learning with Prototype-Based Label Propagation by Iterative Graph Refinement

        **Authors**: *Hao Zhu, Piotr Koniusz*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02298](https://doi.org/10.1109/CVPR52729.2023.02298)

        **Abstract**:

        Few-shot learning (FSL) is popular due to its ability to adapt to novel classes. Compared with inductive few-shot learning, transductive models typically perform better as they leverage all samples of the query set. The two existing classes of methods, prototype-based and graph-based, have the disadvantages of inaccurate prototype estimation and sub-optimal graph construction with kernel functions, respectively. In this paper, we propose a novel prototype-based label propagation to solve these issues. Specifically, our graph construction is based on the relation between prototypes and samples rather than between samples. As prototypes are being updated, the graph changes. We also estimate the label of each prototype instead of considering a prototype be the class centre. On mini-ImageNet, tiered-ImageNet, CIFAR-FS and CUB datasets, we show the proposed method outperforms other state-of-the-art methods in transductive FSL and semi-supervised FSL when some un-labeled data accompanies the novel few-shot task.

        ----

        ## [2284] Open-Set Likelihood Maximization for Few-Shot Learning

        **Authors**: *Malik Boudiaf, Etienne Bennequin, Myriam Tami, Antoine Toubhans, Pablo Piantanida, Céline Hudelot, Ismail Ben Ayed*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02299](https://doi.org/10.1109/CVPR52729.2023.02299)

        **Abstract**:

        We tackle the Few-Shot Open-Set Recognition (FSOSR) problem, i.e. classifying instances among a set of classes for which we only have a few labeled samples, while simultaneously detecting instances that do not belong to any known class. We explore the popular transductive setting, which leverages the unlabelled query instances at inference. Motivated by the observation that existing transductive methods perform poorly in open-set scenarios, we propose a generalization of the maximum likelihood principle, in which latent scores down-weighing the influence of potential outliers are introduced alongside the usual parametric model. Our formulation embeds supervision constraints from the support set and additional penalties discouraging overconfident predictions on the query set. We proceed with a block-coordinate descent, with the latent scores and parametric model co-optimized alternately, thereby benefiting from each other. We call our resulting formulation Open-Set Likelihood Optimization (OSLO). OSLO is interpretable and fully modular; it can be applied on top of any pre-trained model seamlessly. Through extensive experiments, we show that our method surpasses existing inductive and transductive methods on both aspects of open-set recognition, namely inlier classification and outlier detection. Code is available at https://github.com/ebennequin/few-shot-open-set.

        ----

        ## [2285] HyperMatch: Noise-Tolerant Semi-Supervised Learning via Relaxed Contrastive Constraint

        **Authors**: *Beitong Zhou, Jing Lu, Kerui Liu, Yunlu Xu, Zhanzhan Cheng, Yi Niu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02300](https://doi.org/10.1109/CVPR52729.2023.02300)

        **Abstract**:

        Recent developments of the application of Contrastive Learning in Semi-Supervised Learning (SSL) have demonstrated significant advancements, as a result of its exceptional ability to learn class-aware cluster representations and the full exploitation of massive unlabeled data. However, mismatched instance pairs caused by inaccurate pseudo labels would assign an unlabeled instance to the incorrect class in feature space, hence exacerbating SSL's renowned confirmation bias. To address this issue, we introduced a novel SSL approach, HyperMatch, which is a plugin to several SSL designs enabling noise-tolerant utilization of unlabeled data. In particular, confidence predictions are combined with semantic similarities to generate a more objective class distribution, followed by a Gaussian Mixture Model to divide pseudo labels into a ‘confident’ and a ‘less confident’ subset. Then, we introduce Relaxed Contrastive Loss by assigning the ‘less-confident’ samples to a hyper-class, i.e. the union of top-K nearest classes, which effectively regularizes the interference of incorrect pseudo labels and even increases the probability of pulling a ‘less confident’ sample close to its true class. Experiments and in-depth studies demonstrate that HyperMatch delivers remarkable state-of-the-art performance, outperforming Fix-Match on CIFAR100 with 400 and 2500 labeled samples by 11.86% and 4.88%, respectively.

        ----

        ## [2286] Token Boosting for Robust Self-Supervised Visual Transformer Pre-training

        **Authors**: *Tianjiao Li, Lin Geng Foo, Ping Hu, Xindi Shang, Hossein Rahmani, Zehuan Yuan, Jun Liu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02301](https://doi.org/10.1109/CVPR52729.2023.02301)

        **Abstract**:

        Learning with large-scale unlabeled data has become a powerful tool for pre-training Visual Transformers (VTs). However, prior works tend to overlook that, in real-world scenarios, the input data may be corrupted and unreliable. Pre-training VTs on such corrupted data can be challenging, especially when we pre-train via the masked autoencoding approach, where both the inputs and masked “ground truth” targets can potentially be unreliable in this case. To address this limitation, we introduce the Token Boosting Module (TBM) as a plug-and-play component for VTs that effectively allows the VT to learn to extract clean and robust features during masked autoencoding pre-training. We provide theoretical analysis to show how TBM improves model pre-training with more robust and generalizable representations, thus benefiting down stream tasks. We conduct extensive experiments to analyze TBM's effectiveness, and results on four corrupted datasets demonstrate that TBM consistently improves performance on downstream tasks.

        ----

        ## [2287] Difficulty-Based Sampling for Debiased Contrastive Representation Learning

        **Authors**: *Taeuk Jang, Xiaoqian Wang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02302](https://doi.org/10.1109/CVPR52729.2023.02302)

        **Abstract**:

        Contrastive learning is a self-supervised representation learning method that achieves milestone performance in various classification tasks. However, due to its unsupervised fashion, it suffers from the false negative sample problem: randomly drawn negative samples that are assumed to have a different label but actually have the same label as the anchor. This deteriorates the performance of contrastive learning as it contradicts the motivation of contrasting semantically similar and dissimilar pairs. This raised the attention and the importance of finding legitimate negative samples, which should be addressed by distinguishing between 1) true vs. false negatives; 2) easy vs. hard negatives. However, previous works were limited to the statistical approach to handle false negative and hard negative samples with hyperparameters tuning. In this paper, we go beyond the statistical approach and explore the connection between hard negative samples and data bias. We introduce a novel debiased contrastive learning method to explore hard negatives by relative difficulty referencing the bias amplifying counterpart. We propose triplet loss for training a biased encoder that focuses more on easy negative samples. We theoretically show that the triplet loss amplifies the bias in self-supervised representation learning. Finally, we empirically show the proposed method improves downstream classification performance.

        ----

        ## [2288] Improving Selective Visual Question Answering by Learning from Your Peers

        **Authors**: *Corentin Dancette, Spencer Whitehead, Rishabh Maheshwary, Ramakrishna Vedantam, Stefan Scherer, Xinlei Chen, Matthieu Cord, Marcus Rohrbach*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02303](https://doi.org/10.1109/CVPR52729.2023.02303)

        **Abstract**:

        Despite advances in Visual Question Answering (VQA), the ability of models to assess their own correctness remains under-explored. Recent work has shown that VQA models, out-of-the-box, can have difficulties abstaining from answering when they are wrong. The option to abstain, also called Selective Prediction, is highly relevant when deploying systems to users who must trust the system's output (e.g., VQA assistants for users with visual impairments). For such scenarios, abstention can be especially important as users may provide out-of-distribution (OOD) or adversarial inputs that make incorrect answers more likely. In this work, we explore Selective VQA in both in-distribution (ID) and OOD scenarios, where models are presented with mixtures of ID and OOD data. The goal is to maximize the number of questions answered while minimizing the risk of error on those questions. We propose a simple yet effective Learning from Your Peers (LYP) approach for training multimodal selection functions for making abstention decisions. Our approach uses predictions from models trained on distinct subsets of the training data as targets for optimizing a Selective VQA model. It does not require additional manual labels or held-out data and provides a signal for identifying examples that are easy/difficult to generalize to. In our extensive evaluations, we show this benefits a number of models across different architectures and scales. Overall, for ID, we reach 32.92% in the selective prediction metric coverage at 1 % risk of error 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$(\mathcal{C} {@} 1\%)$</tex>
 which doubles the previous best coverage of 15.79% on this task. For mixed ID/OOD, using models' softmax confidences for abstention decisions performs very poorly, answering <5% of questions at 1 % risk of error even when faced with only 10% OOD examples, but a learned selection function with LYP can increase that to 25.38% 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$\mathcal{C}$</tex>
@1%.

        ----

        ## [2289] Superclass Learning with Representation Enhancement

        **Authors**: *Zeyu Gan, Suyun Zhao, Jinlong Kang, Liyuan Shang, Hong Chen, Cuiping Li*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02304](https://doi.org/10.1109/CVPR52729.2023.02304)

        **Abstract**:

        In many real scenarios, data are often divided into a handful of artificial super categories in terms of expert knowledge rather than the representations of images. Concretely, a superclass may contain massive and various raw categories, such as refuse sorting. Due to the lack of common semantic features, the existing classification techniques are intractable to recognize superclass without raw class labels, thus they suffer severe performance damage or require huge annotation costs. To narrow this gap, this paper proposes a superclass learning framework, called SuperClass Learning with Representation Enhancement(SCLRE), to recognize super categories by leveraging enhanced representation. Specifically, by exploiting the self-attention technique across the batch, SCLRE collapses the boundaries of those raw categories and enhances the representation of each superclass. On the enhanced representation space, a superclassaware decision boundary is then reconstructed. Theoretically, we prove that by leveraging attention techniques the generalization error of SCLRE can be bounded under superclass scenarios. Experimentally, extensive results demonstrate that SCLRE outperforms the baseline and other contrastive-based methods on CIFAR-100 datasets and four high-resolution datasets.

        ----

        ## [2290] DISC: Learning from Noisy Labels via Dynamic Instance-Specific Selection and Correction

        **Authors**: *Yifan Li, Hu Han, Shiguang Shan, Xilin Chen*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02305](https://doi.org/10.1109/CVPR52729.2023.02305)

        **Abstract**:

        Existing studies indicate that deep neural networks (DNNs) can eventually memorize the label noise. We observe that the memorization strength of DNNs towards each instance is different and can be represented by the confidence value, which becomes larger and larger during the training process. Based on this, we propose a Dynamic Instance-specific Selection and Correction method (DISC) for learning from noisy labels (LNL). We first use a two- view-based backbone for image classification, obtaining confidence for each image from two views. Then we propose a dynamic threshold strategy for each instance, based on the momentum of each instance's memorization strength in previous epochs to select and correct noisy labeled data. Benefiting from the dynamic threshold strategy and two-view learning, we can effectively group each instance into one of the three subsets (i.e., clean, hard, and purified) based on the prediction consistency and discrepancy by two views at each epoch. Finally, we employ different regularization strategies to conquer subsets with different degrees of label noise, improving the whole network's robustness. Comprehensive evaluations on three controllable and four real-world LNL benchmarks show that our method outperforms the state-of-the-art (SOTA) methods to leverage useful information in noisy data while alleviating the pollution of label noise. Code is available at https://github.com/JackYFL/DISC.

        ----

        ## [2291] FCC: Feature Clusters Compression for Long-Tailed Visual Recognition

        **Authors**: *Jian Li, Ziyao Meng, Daqian Shi, Rui Song, Xiaolei Diao, Jingwen Wang, Hao Xu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02306](https://doi.org/10.1109/CVPR52729.2023.02306)

        **Abstract**:

        Deep Neural Networks (DNNs) are rather restrictive in long-tailed data, since they commonly exhibit an under-representation for minority classes. Various remedies have been proposed to tackle this problem from different perspectives, but they ignore the impact of the density of Backbone Features (BFs) on this issue. Through representation learning, DNNs can map BFs into dense clusters in feature space, while the features of minority classes often show sparse clusters. In practical applications, these features are discretely mapped or even cross the decision boundary resulting in misclassification. Inspired by this observation, we propose a simple and generic method, namely Feature Clusters Compression (FCC), to increase the density of BFs by compressing backbone feature clusters. The proposed FCC can be easily achieved by only multiplying original BFs by a scaling factor in training phase, which establishes a linear compression relationship between the original and multiplied features, and forces DNNs to map the former into denser clusters. In test phase, we directly feed original features without multiplying the factor to the classifier, such that BFs of test samples are mapped closer together and do not easily cross the decision boundary. Meanwhile, FCC can be friendly combined with existing long-tailed methods and further boost them. We apply FCC to numerous state-of-the-art methods and evaluate them on widely used long-tailed benchmark datasets. Extensive experiments fully verify the effectiveness and generality of our method. Code is available at https://github.com/lijian16/FCC.

        ----

        ## [2292] Dynamically Instance-Guided Adaptation: A Backward-free Approach for Test-Time Domain Adaptive Semantic Segmentation

        **Authors**: *Wei Wang, Zhun Zhong, Weijie Wang, Xi Chen, Charles Ling, Boyu Wang, Nicu Sebe*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02307](https://doi.org/10.1109/CVPR52729.2023.02307)

        **Abstract**:

        In this paper, we study the application of Test-time domain adaptation in semantic segmentation (TTDA-Seg) where both efficiency and effectiveness are crucial. Existing methods either have low efficiency (e.g., backward optimization) or ignore semantic adaptation (e.g., distribution alignment). Besides, they would suffer from the accumulated errors caused by unstable optimization and abnormal distributions. To solve these problems, we propose a novel backward-free approach for TTDA-Seg, called Dynamically Instance-Guided Adaptation (DIGA). Our principle is utilizing each instance to dynamically guide its own adaptation in a non-parametric way, which avoids the error accumulation issue and expensive optimizing cost. Specifically, DIGA is composed of a distribution adaptation module (DAM) and a semantic adaptation module (SAM), enabling us to jointly adapt the model in two indispensable aspects. DAM mixes the instance and source BN statistics to encourage the model to capture robust representation. SAM combines the historical prototypes with instance-level prototypes to adjust semantic predictions, which can be associated with the parametric classifier to mutually benefit the final results. Extensive experiments evaluated on five target domains demonstrate the effectiveness and efficiency of the proposed method. Our DIGA establishes new state-of-the-art performance in TTDA-Seg. Source code is available at: https://github.com/Waybaba/DIGA.

        ----

        ## [2293] Semi-Supervised Domain Adaptation with Source Label Adaptation

        **Authors**: *Yu-Chu Yu, Hsuan-Tien Lin*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02308](https://doi.org/10.1109/CVPR52729.2023.02308)

        **Abstract**:

        Semi-Supervised Domain Adaptation (SSDA) involves learning to classify unseen target data with a few labeled and lots of unlabeled target data, along with many labeled source data from a related domain. Current SSDA approaches usually aim at aligning the target data to the labeled source data with feature space mapping and pseudolabel assignments. Nevertheless, such a source-oriented model can sometimes align the target data to source data of the wrong classes, degrading the classification performance. This paper presents a novel source-adaptive paradigm that adapts the source data to match the target data. Our key idea is to view the source data as a noisily-labeled version of the ideal target data. Then, we propose an SSDA model that cleans up the label noise dynamically with the help of a robust cleaner component designed from the target perspective. Since the paradigm is very different from the core ideas behind existing SSDA approaches, our proposed model can be easily coupled with them to improve their performance. Empirical results on two state-of-the-art SSDA approaches demonstrate that the proposed model effectively cleans up the noise within the source labels and exhibits superior performance over those approaches across benchmark datasets. Our code is available at https://github.com/chu0802/SLA.

        ----

        ## [2294] Adjustment and Alignment for Unbiased Open Set Domain Adaptation

        **Authors**: *Wuyang Li, Jie Liu, Bo Han, Yixuan Yuan*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02309](https://doi.org/10.1109/CVPR52729.2023.02309)

        **Abstract**:

        Open Set Domain Adaptation (OSDA) transfers the model from a label-rich domain to a label-free one containing novel-class samples. Existing OSDA works overlook abundant novel-class semantics hidden in the source domain, leading to a biased model learning and transfer. Although the causality has been studied to remove the semantic-level bias, the non-available novel-class samples result in the failure of existing causal solutions in OSDA. To break through this barrier, we propose a novel causality-driven solution with the unexplored front-door adjustment theory, and then implement it with a theoretically grounded framework, coined Adjustment and Alignment (ANNA), to achieve an unbiased OSDA. In a nutshell, ANNA consists of Front-Door Adjustment (FDA) to correct the biased learning in the source domain and Decoupled Causal Alignment (DCA) to transfer the model unbiasedly. On the one hand, FDA delves into fine-grained visual blocks to discover novel-class regions hidden in the base-class image. Then, it corrects the biased model optimization by implementing causal debiasing. On the other hand, DCA disentangles the base-class and novel-class regions with orthogonal masks, and then adapts the decoupled distribution for an unbiased model transfer. Extensive experiments show that ANNA achieves state-of-the-art results. The code is available at https://github.com/CityU-AIM-Group/Anna.

        ----

        ## [2295] C-SFDA: A Curriculum Learning Aided Self-Training Framework for Efficient Source Free Domain Adaptation

        **Authors**: *Nazmul Karim, Niluthpol Chowdhury Mithun, Abhinav Rajvanshi, Han-Pang Chiu, Supun Samarasekera, Nazanin Rahnavard*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02310](https://doi.org/10.1109/CVPR52729.2023.02310)

        **Abstract**:

        Unsupervised domain adaptation (UDA) approaches focus on adapting models trained on a labeled source domain to an unlabeled target domain. In contrast to UDA, source-free domain adaptation (SFDA) is a more practical setup as access to source data is no longer required during adaptation. Recent state-of-the-art (SOTA) methods on SFDA mostly focus on pseudo-label refinement based self-training which generally suffers from two issues: i) inevitable occurrence of noisy pseudo-labels that could lead to early training time memorization, ii) refinement process requires maintaining a memory bank which creates a significant burden in resource constraint scenarios. To address these concerns, we propose C-SFDA, a curriculum learning aided self-training framework for SFDA that adapts efficiently and reliably to changes across domains based on selective pseudo-labeling. Specifically, we employ a curriculum learning scheme to promote learning from a restricted amount of pseudo labels selected based on their reliabilities. This simple yet effective step successfully prevents label noise propagation during different stages of adaptation and eliminates the need for costly memory-bank based label refinement. Our extensive experimental evaluations on both image recognition and semantic segmentation tasks confirm the effectiveness of our method. C-SFDA is also applicable to online test-time domain adaptation and outperforms previous SOTA methods in this task.

        ----

        ## [2296] ALOFT: A Lightweight MLP-Like Architecture with Dynamic Low-Frequency Transform for Domain Generalization

        **Authors**: *Jintao Guo, Na Wang, Lei Qi, Yinghuan Shi*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02311](https://doi.org/10.1109/CVPR52729.2023.02311)

        **Abstract**:

        Domain generalization (DG) aims to learn a model that generalizes well to unseen target domains utilizing multiple source domains without re-training. Most existing DG works are based on convolutional neural networks (CNNs). However, the local operation of the convolution kernel makes the model focus too much on local representations (e.g., texture), which inherently causes the model more prone to overfit to the source domains and hampers its generalization ability. Recently, several MLP-based methods have achieved promising results in supervised learning tasks by learning global interactions among different patches of the image. Inspired by this, in this paper, we first analyze the difference between CNN and MLP methods in DG and find that MLP methods exhibit a better generalization ability because they can better capture the global representations (e.g., structure) than CNN methods. Then, based on a recent lightweight MLP method, we obtain a strong baseline that outperforms most state-of-the-art CNN-based methods. The baseline can learn global structure representations with a filter to suppress structureirrelevant information in the frequency space. Moreover, we propose a dynAmic LOw-Frequency spectrum Transform (ALOFT) that can perturb local texture features while preserving global structure features, thus enabling the filter to remove structure-irrelevant information sufficiently. Extensive experiments on four benchmarks have demonstrated that our method can achieve great performance improvement with a small number of parameters compared to SOTA CNN-based DG methods. Our code is available at https://github.com/lingeringlight/ALOFT/.

        ----

        ## [2297] Modality-Agnostic Debiasing for Single Domain Generalization

        **Authors**: *Sanqing Qu, Yingwei Pan, Guang Chen, Ting Yao, Changjun Jiang, Tao Mei*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02312](https://doi.org/10.1109/CVPR52729.2023.02312)

        **Abstract**:

        Deep neural networks (DNNs) usually fail to generalize well to outside of distribution (OOD) data, especially in the extreme case of single domain generalization (single-DG) that transfers DNNs from single domain to multiple unseen domains. Existing single-DG techniques commonly devise various data-augmentation algorithms, and remould the multi-source domain generalization methodology to learn domain-generalized (semantic) features. Nevertheless, these methods are typically modality-specific, thereby being only applicable to one single modality (e.g., image). In contrast, we target a versatile Modality-Agnostic Debiasing (MAD) framework for single-DG, that enables generalization for different modalities. Technically, MAD introduces a novel two-branch classifier: a biased-branch encourages the classifier to identify the domain-specific (superficial) features, and a general-branch captures domain-generalized features based on the knowledge from biased-branch. Our MAD is appealing in view that it is pluggable to most single-DG models. We validate the superiority of our MAD in a variety of single-DG scenarios with different modalities, including recognition on 1D texts, 2D images, 3D point clouds, and semantic segmentation on 2D images. More remarkably, for recognition on 3D point clouds and semantic segmentation on 2D images, MAD improves DSU by 2.82% and 1.5% in accuracy and mIOU.

        ----

        ## [2298] ActMAD: Activation Matching to Align Distributions for Test-Time-Training

        **Authors**: *Muhammad Jehanzeb Mirza, Pol Jané-Soneira, Wei Lin, Mateusz Kozinski, Horst Possegger, Horst Bischof*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02313](https://doi.org/10.1109/CVPR52729.2023.02313)

        **Abstract**:

        Test-Time-Training (TTT) is an approach to cope with out-of-distribution (OOD) data by adapting a trained model to distribution shifts occurring at test-time. We propose to perform this adaptation via Activation Matching (ActMAD): We analyze activations of the model and align activation statistics of the OOD test data to those of the training data. In contrast to existing methods, which model the distribution of entire channels in the ultimate layer of the feature extractor, we model the distribution of each feature in multiple layers across the network. This results in a more fine-grained supervision and makes ActMAD attain state of the art performance on CIFAR-100C and Imagenet-C. ActMAD is also architecture-and task-agnostic, which lets us go beyond image classification, and score 15.4% improvement over previous approaches when evaluating a KITTI-trained object detector on KITTI-Fog. Our experiments highlight that ActMAD can be applied to online adaptation in realistic scenarios, requiring little data to attain its full performance.

        ----

        ## [2299] TIPI: Test Time Adaptation with Transformation Invariance

        **Authors**: *A. Tuan Nguyen, Thanh Nguyen-Tang, Ser-Nam Lim, Philip H. S. Torr*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02314](https://doi.org/10.1109/CVPR52729.2023.02314)

        **Abstract**:

        When deploying a machine learning model to a new environment, we often encounter the distribution shift problem - meaning the target data distribution is different from the model's training distribution. In this paper, we assume that labels are not provided for this new domain, and that we do not store the source data (e.g., for privacy reasons). It has been shown that even small shifts in the data distribution can affect the model's performance severely. Test Time Adaptation offers a means to combat this problem, as it allows the model to adapt during test time to the new data distribution, using only unlabeled test data batches. To achieve this, the predominant approach is to optimize a surrogate loss on the test-time unlabeled target data. In particular, minimizing the prediction's entropy on target samples [34] has received much interest as it is task-agnostic and does not require altering the model's training phase (e.g., does not require adding a self-supervised task during training on the source domain). However, as the target data's batch size is often small in real-world scenarios (e.g., autonomous driving models process each few frames in real-time), we argue that this surrogate loss is not optimal since it often collapses with small batch sizes. To tackle this problem, in this paper, we propose to use an invariance regularizer as the surrogate loss during test-time adaptation, motivated by our theoretical results regarding the model's performance under input transformations. The resulting method (TIPI-Test time adaPtation with transformation Invariance) is validated with extensive experiments in various benchmarks (Cifar10-C, Cifar100-C, ImageNet-C, DIGITS, and VisDA17). Remarkably, TIPI is robust against small batch sizes (as small as 2 in our experiments), and consistently outperforms TENT [34] in all settings. Our code is released at https://github.com/atuannguyen/TIPI.

        ----

        ## [2300] Improved Test-Time Adaptation for Domain Generalization

        **Authors**: *Liang Chen, Yong Zhang, Yibing Song, Ying Shan, Lingqiao Liu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02315](https://doi.org/10.1109/CVPR52729.2023.02315)

        **Abstract**:

        The main challenge in domain generalization (DG) is to handle the distribution shift problem that lies between the training and test data. Recent studies suggest that test-time training (TTT), which adapts the learned model with test data, might be a promising solution to the problem. Generally, a TTT strategy hinges its performance on two main factors: selecting an appropriate auxiliary TTT task for updating and identifying reliable parameters to update during the test phase. Both previous arts and our experiments indicate that TTT may not improve but be detrimental to the learned model if those two factors are not properly considered. This work addresses those two factors by proposing an Improved Test-Time Adaptation (ITTA) method. First, instead of heuristically defining an auxiliary objective, we propose a learnable consistency loss for the TTT task, which contains learnable parameters that can be adjusted toward better alignment between our TTT task and the main prediction task. Second, we introduce additional adaptive parameters for the trained model, and we suggest only updating the adaptive parameters during the test phase. Through extensive experiments, we show that the proposed two strategies are beneficial for the learned model (see Figure 1), and ITTA could achieve superior performance to the current state-of-the-art methods on several DG benchmarks. Code is available at https://github.com/liangchen527/ITTA.

        ----

        ## [2301] Learning with Fantasy: Semantic-Aware Virtual Contrastive Constraint for Few-Shot Class-Incremental Learning

        **Authors**: *Zeyin Song, Yifan Zhao, Yujun Shi, Peixi Peng, Li Yuan, Yonghong Tian*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02316](https://doi.org/10.1109/CVPR52729.2023.02316)

        **Abstract**:

        Few-shot class-incremental learning (FSCIL) aims at learning to classify new classes continually from limited samples without forgetting the old classes. The mainstream framework tackling FSCIL is first to adopt the cross-entropy (CE) loss for training at the base session, then freeze the feature extractor to adapt to new classes. However, in this work, we find that the CE loss is not ideal for the base session training as it suffers poor class separation in terms of representations, which further degrades generalization to novel classes. One tempting method to mitigate this problem is to apply an additional naïve supervised contrastive learning (SCL) in the base session. Unfortunately, we find that although SCL can create a slightly better representation separation among different base classes, it still struggles to separate base classes and new classes. Inspired by the observations made, we propose Semantic-Aware Virtual Contrastive model (SAVC), a novel method that facilitates separation between new classes and base classes by introducing virtual classes to SCL. These virtual classes, which are generated via pre-defined transformations, not only act as placeholders for unseen classes in the representation space, but also provide diverse semantic information. By learning to recognize and contrast in the fantasy space fostered by virtual classes, our SAVC significantly boosts base class separation and novel class generalization, achieving new state-of-the-art performance on the three widely-used FSCIL benchmark datasets. Code is available at: https://github.com/zysong0113/SAVC.

        ----

        ## [2302] NIFF: Alleviating Forgetting in Generalized Few-Shot Object Detection via Neural Instance Feature Forging

        **Authors**: *Karim Guirguis, Johannes Meier, George Eskandar, Matthias Kayser, Bin Yang, Jürgen Beyerer*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02317](https://doi.org/10.1109/CVPR52729.2023.02317)

        **Abstract**:

        Privacy and memory are two recurring themes in a broad conversation about the societal impact of AI. These con-cerns arise from the need for huge amounts of data to train deep neural networks. A promise of Generalized Few-shot Object Detection (G-FSOD), a learning paradigm in AI, is to alleviate the need for collecting abundant training samples of novel classes we wish to detect by leveraging prior knowledge from old classes (i.e., base classes). G-FSOD strives to learn these novel classes while alleviating catas-trophic forgetting of the base classes. However, existing approaches assume that the base images are accessible, an assumption that does not hold when sharing and storing data is problematic. In this work, we propose the first data-free knowledge distillation (DFKD) approach for G-FSOD that leverages the statistics of the region of interest (RoI) features from the base model to forge instance-level features without accessing the base images. Our contribution is three-fold: (1) we design a standalone lightweight generator with (2) class-wise heads (3) to generate and replay diverse instance-level base features to the RoI head while finetuning on the novel data. This stands in contrast to standard DFKD approaches in image classification, which invert the entire network to generate base images. Moreover, we make careful design choices in the novel finetuning pipeline to regularize the model. We show that our approach can dramatically reduce the base memory requirements, all while setting a new standard for G-FSOD on the challenging MS-COCO and PASCAL-VOC benchmarks.

        ----

        ## [2303] MixPHM: Redundancy-Aware Parameter-Efficient Tuning for Low-Resource Visual Question Answering

        **Authors**: *Jingjing Jiang, Nanning Zheng*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02318](https://doi.org/10.1109/CVPR52729.2023.02318)

        **Abstract**:

        Recently, finetuning pretrained vision-language models (VLMs) has been a prevailing paradigm for achieving state-of-the-art performance in VQA. However, as VLMs scale, it becomes computationally expensive, storage inefficient, and prone to overfitting when tuning full model parameters for a specific task in low-resource settings. Although current parameter-efficient tuning methods dramatically reduce the number of tunable parameters, there still exists a significant performance gap with full finetuning. In this paper, we propose MixPHM, a redundancy-aware parameter-efficient tuning method that outperforms full finetuning in low-resource VQA. Specifically, MixPHM is a lightweight module implemented by multiple PHM-experts in a mixture-of-experts manner. To reduce parameter redundancy, we reparameterize expert weights in a low-rank subspace and share part of the weights inside and across MixPHM. Moreover, based on our quantitative analysis of representation redundancy, we propose Redundancy Regularization, which facilitates MixPHM to reduce task-irrelevant redundancy while promoting task-relevant correlation. Experiments conducted on VQA v2, GQA, and OK-VQA with different low-resource settings show that our MixPHM outperforms state-of-the-art parameter-efficient methods and is the only one consistently surpassing full finetuning.

        ----

        ## [2304] PIVOT: Prompting for Video Continual Learning

        **Authors**: *Andrés Villa, Juan León Alcázar, Motasem Alfarra, Kumail Alhamoud, Julio Hurtado, Fabian Caba Heilbron, Alvaro Soto, Bernard Ghanem*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02319](https://doi.org/10.1109/CVPR52729.2023.02319)

        **Abstract**:

        Modern machine learning pipelines are limited due to data availability, storage quotas, privacy regulations, and expensive annotation processes. These constraints make it difficult or impossible to train and update large-scale models on such dynamic annotated sets. Continual learning directly approaches this problem, with the ultimate goal of devising methods where a deep neural network effectively learns relevant patterns for new (unseen) classes, without significantly altering its performance on previously learned ones. In this paper, we address the problem of continual learning for video data. We introduce PIVOT, a novel method that leverages extensive knowledge in pre-trained models from the image domain, thereby reducing the number of trainable parameters and the associated forgetting. Unlike previous methods, ours is the first approach that effectively uses prompting mechanisms for continual learning without any in-domain pre-training. Our experiments show that PIVOT improves state-of-the-art methods by a significant 27% on the 20-task ActivityNet setup.

        ----

        ## [2305] BlackVIP: Black-Box Visual Prompting for Robust Transfer Learning

        **Authors**: *Changdae Oh, Hyeji Hwang, Hee Young Lee, YongTaek Lim, Geunyoung Jung, Jiyoung Jung, Hosik Choi, Kyungwoo Song*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02320](https://doi.org/10.1109/CVPR52729.2023.02320)

        **Abstract**:

        With the surge of large-scale pre-trained models (PTMs), fine-tuning these models to numerous downstream tasks becomes a crucial problem. Consequently, parameter efficient transfer learning (PETL) of large models has grasped huge attention. While recent PETL methods showcase impressive performance, they rely on optimistic assumptions: 1) the entire parameter set of a PTM is available, and 2) a sufficiently large memory capacity for the fine-tuning is equipped. However, in most real-world applications, PTMs are served as a black-box API or proprietary software without explicit parameter accessibility. Besides, it is hard to meet a large memory requirement for modern PTMs. In this work, we propose black-box visual prompting (Black-VIP), which efficiently adapts the PTMs without knowledge about model architectures and parameters. Black-VIP has two components; 1) Coordinator and 2) simultaneous perturbation stochastic approximation with gradient correction (SPSA-GC). The Coordinator designs inputdependent image-shaped visual prompts, which improves few-shot adaptation and robustness on distribution/location shift. SPSA-GC efficiently estimates the gradient of a target model to update Coordinator. Extensive experiments on 16 datasets demonstrate that BlackVIP enables robust adaptation to diverse domains without accessing PTMs' parameters, with minimal memory requirements. Code: https://github.com/changdaeoh/BlackVIP

        ----

        ## [2306] DKT: Diverse Knowledge Transfer Transformer for Class Incremental Learning

        **Authors**: *Xinyuan Gao, Yuhang He, Songlin Dong, Jie Cheng, Xing Wei, Yihong Gong*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02321](https://doi.org/10.1109/CVPR52729.2023.02321)

        **Abstract**:

        In the context of incremental class learning, deep neural networks are prone to catastrophic forgetting, where the accuracy of old classes declines substantially as new knowledge is learned. While recent studies have sought to address this issue, most approaches suffer from either the stability-plasticity dilemma or excessive computational and parameter requirements. To tackle these challenges, we propose a novel framework, the Diverse Knowledge Transfer Transformer (DKT), which incorporates two knowledge transfer mechanisms that use attention mechanisms to transfer both task-specific and task-general knowledge to the current task, along with a duplex classifier to address the stability-plasticity dilemma. Additionally, we design a loss function that clusters similar categories and discriminates between old and new tasks in the feature space. The proposed method requires only a small number of extra parameters, which are negligible in comparison to the increasing number of tasks. We perform extensive experiments on CIFAR100, ImageNet100, and ImageNet1000 datasets, which demonstrate that our method outperforms other competitive methods and achieves state-of-the-art performance. Our source code is available at https://github.com/MIVXJTU/DKT.

        ----

        ## [2307] PCR: Proxy-Based Contrastive Replay for Online Class-Incremental Continual Learning

        **Authors**: *Huiwei Lin, Baoquan Zhang, Shanshan Feng, Xutao Li, Yunming Ye*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02322](https://doi.org/10.1109/CVPR52729.2023.02322)

        **Abstract**:

        Online class-incremental continual learning is a specific task of continual learning. It aims to continuously learn new classes from data stream and the samples of data stream are seen only once, which suffers from the catastrophic forgetting issue, i.e., forgetting historical knowledge of old classes. Existing replay-based methods effectively alleviate this issue by saving and replaying part of old data in a proxy-based or contrastive-based replay manner. Although these two replay manners are effective, the former would incline to new classes due to class imbalance issues, and the latter is unstable and hard to converge because of the limited number of samples. In this paper, we conduct a comprehensive analysis of these two replay manners and find that they can be complementary. Inspired by this finding, we propose a novel replay-based method called proxy-based contrastive replay (PCR). The key operation is to replace the contrastive samples of anchors with corresponding proxies in the contrastive-based way. It alleviates the phenomenon of catastrophic forgetting by effectively addressing the imbalance issue, as well as keeps a faster convergence of the model. We conduct extensive experiments on three real-world benchmark datasets, and empirical results consistently demonstrate the superiority of PCR over various state-of-the-art methods
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
https://github.com/FelixHuiweiLin/PCR.

        ----

        ## [2308] Masked Autoencoders Enable Efficient Knowledge Distillers

        **Authors**: *Yutong Bai, Zeyu Wang, Junfei Xiao, Chen Wei, Huiyu Wang, Alan L. Yuille, Yuyin Zhou, Cihang Xie*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02323](https://doi.org/10.1109/CVPR52729.2023.02323)

        **Abstract**:

        This paper studies the potential of distilling knowledge from pre-trained models, especially Masked Autoencoders. Our approach is simple: in addition to optimizing the pixel reconstruction loss on masked inputs, we minimize the distance between the intermediate feature map of the teacher model and that of the student model. This design leads to a computationally efficient knowledge distillation framework, given 1) only a small visible subset of patches is used, and 2) the (cumbersome) teacher model only needs to be partially executed, i.e., forward propagate inputs through the first few layers, for obtaining intermediate feature maps. Compared to directly distilling fine-tuned models, distilling pre-trained models substantially improves downstream performance. For example, by distilling the knowledge from an MAE pre-trained ViT-L into a ViT-B, our method achieves 84.0% ImageNet top-1 accuracy, outperforming the baseline of directly distilling afinetuned ViT-L by 1.2%. More intriguingly, our method can robustly distill knowledge from teacher models even with extremely high masking ratios: e.g., with 95% masking ratio where merely TEN patches are visible during distillation, our ViT-B competitively attains a top-1 ImageNet accuracy of 83.6%; surprisingly, it can still secure 82.4% top-1 ImageNet accuracy by aggressively training with just FOUR visible patches (98% masking ratio). The code and models are publicly available at https://github.com/UCSC-VLAA/DMAE.

        ----

        ## [2309] Data-Free Knowledge Distillation via Feature Exchange and Activation Region Constraint

        **Authors**: *Shikang Yu, Jiachen Chen, Hu Han, Shuqiang Jiang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02324](https://doi.org/10.1109/CVPR52729.2023.02324)

        **Abstract**:

        Despite the tremendous progress on data-free knowledge distillation (DFKD) based on synthetic data generation, there are still limitations in diverse and efficient data synthesis. It is naive to expect that a simple combination of generative network-based data synthesis and data augmentation will solve these issues. Therefore, this paper proposes a novel data-free knowledge distillation method (Spaceship-Net) based on channel-wise feature exchange (CFE) and multi-scale spatial activation region consistency (mSARC) constraint. Specifically, CFE allows our generative network to better sample from the feature space and efficiently synthesize diverse images for learning the student network. However, using CFE alone can severely amplify the unwanted noises in the synthesized images, which may result in failure to improve distillation learning and even have negative effects. Therefore, we propose mSARC to assure the student network can imitate not only the logit output but also the spatial activation region of the teacher network in order to alleviate the influence of unwanted noises in diverse synthetic images on distillation learning. Extensive experiments on CIFAR-10, CIFAR-100, Tiny-ImageNet, Imagenette, and ImageNet100 show that our method can work well with different backbone networks, and outperform the state-of-the-art DFKD methods. Code will be available at: https://github.com/skgyu/Spaceship-Net.

        ----

        ## [2310] Multi-Level Logit Distillation

        **Authors**: *Ying Jin, Jiaqi Wang, Dahua Lin*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02325](https://doi.org/10.1109/CVPR52729.2023.02325)

        **Abstract**:

        Knowledge Distillation (KD) aims at distilling the knowledge from the large teacher model to a lightweight student model. Mainstream KD methods can be divided into two categories, logit distillation, and feature distillation. The former is easy to implement, but inferior in performance, while the latter is not applicable to some practical circumstances due to concerns such as privacy and safety. Towards this dilemma, in this paper, we explore a stronger logit distillation method via making better utilization of logit outputs. Concretely, we propose a simple yet effective approach to logit distillation via multi-level prediction alignment. Through this framework, the prediction alignment is not only conducted at the instance level, but also at the batch and class level, through which the student model learns instance prediction, input correlation, and category correlation simultaneously. In addition, a prediction augmentation mechanism based on model calibration further boosts the performance. Extensive experiment results validate that our method enjoys consistently higher performance than previous logit distillation methods, and even reaches competitive performance with mainstream feature distillation methods. Code is available at https://github.com/Jin-Ying/Multi-Level-Logit-Distillation.

        ----

        ## [2311] Preserving Linear Separability in Continual Learning by Backward Feature Projection

        **Authors**: *Qiao Gu, Dongsub Shim, Florian Shkurti*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02326](https://doi.org/10.1109/CVPR52729.2023.02326)

        **Abstract**:

        Catastrophic forgetting has been a major challenge in continual learning, where the model needs to learn new tasks with limited or no access to data from previously seen tasks. To tackle this challenge, methods based on knowledge distillation in feature space have been proposed and shown to reduce forgetting [16, 17, 25 ]. However, most feature distillation methods directly constrain the new features to match the old ones, overlooking the need for plasticity. To achieve a better stability-plasticity trade-off, we propose Backward Feature Projection (BFP), a method for continual learning that allows the new features to change up to a learnable linear transformation of the old features. BFP preserves the linear separability of the old classes while allowing the emergence of new feature directions to accommodate new classes. BFP can be integrated with existing experience replay methods and boost performance by a significant margin. We also demonstrate that BFP helps learn a better representation space, in which linear separability is well preserved during continual learning and linear probing achieves high classification accuracy.

        ----

        ## [2312] Critical Learning Periods for Multisensory Integration in Deep Networks

        **Authors**: *Michael Kleinman, Alessandro Achille, Stefano Soatto*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02327](https://doi.org/10.1109/CVPR52729.2023.02327)

        **Abstract**:

        We show that the ability of a neural network to integrate information from diverse sources hinges critically on being exposed to properly correlated signals during the early phases of training. Interfering with the learning process during this initial stage can permanently impair the development of a skill, both in artificial and biological systems where the phenomenon is known as a critical learning period. We show that critical periods arise from the complex and unstable early transient dynamics, which are decisive of final performance of the trained system and their learned representations. This evidence challenges the view, engendered by analysis of wide and shallow networks, that early learning dynamics of neural networks are simple, akin to those of a linear model. Indeed, we show that even deep linear networks exhibit critical learning periods for multi-source integration, while shallow networks do not. To better understand how the internal representations change according to disturbances or sensory deficits, we introduce a new measure of source sensitivity, which allows us to track the inhibition and integration of sources during training. Our analysis of inhibition suggests cross-source reconstruction as a natural auxiliary training objective, and indeed we show that architectures trained with cross-sensor reconstruction objectives are remarkably more resilient to critical periods. Our findings suggest that the recent success in self-supervised multi-modal training compared to previous supervised efforts may be in part due to more robust learning dynamics and not solely due to better architectures and/or more data.

        ----

        ## [2313] SLACK: Stable Learning of Augmentations with Cold-Start and KL Regularization

        **Authors**: *Juliette Marrie, Michael Arbel, Diane Larlus, Julien Mairal*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02328](https://doi.org/10.1109/CVPR52729.2023.02328)

        **Abstract**:

        Data augmentation is known to improve the generalization capabilities of neural networks, provided that the set of transformations is chosen with care, a selection often performed manually. Automatic data augmentation aims at automating this process. However, most recent approaches still rely on some prior information; they start from a small pool of manually-selected default transformations that are either used to pretrain the network or forced to be part of the policy learned by the automatic data augmentation algorithm. In this paper, we propose to directly learn the augmentation policy without leveraging such prior knowledge. The resulting bilevel optimization problem becomes more challenging due to the larger search space and the inherent instability of bilevel optimization algorithms. To mitigate these issues (i) we follow a successive cold-start strategy with a Kullback-Leibler regularization, and (ii) we parameterize magnitudes as continuous distributions. Our approach leads to competitive results on standard benchmarks despite a more challenging setting, and generalizes beyond natural images.
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
Project page: https://europe.naverlabs.com/slack

        ----

        ## [2314] Improving Generalization with Domain Convex Game

        **Authors**: *Fangrui Lv, Jian Liang, Shuang Li, Jinming Zhang, Di Liu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02329](https://doi.org/10.1109/CVPR52729.2023.02329)

        **Abstract**:

        Domain generalization (DG) tends to alleviate the poor generalization capability of deep neural networks by learning model with multiple source domains. A classical solution to DG is domain augmentation, the common belief of which is that diversifying source domains will be conducive to the out-of-distribution generalization. However, these claims are understood intuitively, rather than mathematically. Our explorations empirically reveal that the correlation between model generalization and the diversity of domains may be not strictly positive, which limits the effectiveness of domain augmentation. This work therefore aim to guarantee and further enhance the validity of this strand. To this end, we propose a new perspective on DG that recasts it as a convex game between domains. We first encourage each diversified domain to enhance model generalization by elaborately designing a regularization term based on supermodularity. Meanwhile, a sample filter is constructed to eliminate low-quality samples, thereby avoiding the impact of potentially harmful information. Our framework presents a new avenue for the formal analysis of DG, heuristic analysis and extensive experiments demonstrate the rationality and effectiveness.
<inf xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</inf>
1Code is available at “https://github.com/BIT-DA/DCG”.

        ----

        ## [2315] Exploring Data Geometry for Continual Learning

        **Authors**: *Zhi Gao, Chen Xu, Feng Li, Yunde Jia, Mehrtash Harandi, Yuwei Wu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02330](https://doi.org/10.1109/CVPR52729.2023.02330)

        **Abstract**:

        Continual learning aims to efficiently learn from a non-stationary stream of data while avoiding forgetting the knowledge of old data. In many practical applications, data complies with non-Euclidean geometry. As such, the commonly used Euclidean space cannot gracefully capture non-Euclidean geometric structures of data, leading to in-ferior results. In this paper, we study continual learning from a novel perspective by exploring data geometry for the non-stationary stream of data. Our method dynamically expands the geometry of the underlying space to match growing geometric structures induced by new data, and pre-vents forgetting by keeping geometric structures of old data into account. In doing so, making use of the mixed cur-vature space, we propose an incremental search scheme, through which the growing geometric structures are en-coded. Then, we introduce an angular-regularization loss and a neighbor-robustness loss to train the model, capa-ble of penalizing the change of global geometric structures and local geometric structures. Experiments show that our method achieves better performance than baseline methods designed in Euclidean space.

        ----

        ## [2316] FlowGrad: Controlling the Output of Generative ODEs with Gradients

        **Authors**: *Xingchao Liu, Lemeng Wu, Shujian Zhang, Chengyue Gong, Wei Ping, Qiang Liu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02331](https://doi.org/10.1109/CVPR52729.2023.02331)

        **Abstract**:

        Generative modeling with ordinary differential equations (ODEs) has achieved fantastic results on a variety of applications. Yet, few works have focused on controlling the generated content of a pre-trained ODE-based generative model. In this paper, we propose to optimize the output of ODE models according to a guidance function to achieve controllable generation. We point out that, the gradients can be efficiently back-propagated from the output to any intermediate time steps on the ODE trajectory, by decomposing the back-propagation and computing vectorJacobian products. To further accelerate the computation of the back-propagation, we propose to use a non-uniform discretization to approximate the ODE trajectory, where we measure how straight the trajectory is and gather the straight parts into one discretization step. This allows us to save ∼ 90% of the back-propagation time with ignorable error. Our framework, named FlowGrad, outperforms the state-of-the-art baselines on text-guided image manipulation. Moreover, FlowGrad enables us to find global semantic directions in frozen ODE-based generative models that can be used to manipulate new images without extra optimization.

        ----

        ## [2317] Deep Graph Reprogramming

        **Authors**: *Yongcheng Jing, Chongbin Yuan, Li Ju, Yiding Yang, Xinchao Wang, Dacheng Tao*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02332](https://doi.org/10.1109/CVPR52729.2023.02332)

        **Abstract**:

        In this paper, we explore a novel model reusing task tailored for graph neural networks (GNNs), termed as “deep graph reprogramming”. We strive to reprogram a pretrained GNN, without amending raw node features nor model parameters, to handle a bunch of cross-level downstream tasks in various domains. To this end, we propose an innovative Data Reprogramming paradigm alongside a Model Reprogramming paradigm. The former one aims to address the challenge of diversified graph feature dimensions for various tasks on the input side, while the latter alleviates the dilemma of fixed per-task-per-model behavior on the model side. For data reprogramming, we specifically devise an elaborated Meta-FeatPadding method to deal with heterogeneous input dimensions, and also develop a transductive Edge-Slimming as well as an inductive Meta-GraPadding approach for diverse homogenous samples. Meanwhile, for model reprogramming, we propose a novel task-adaptive Reprogrammable-Aggregator, to endow the frozen model with larger expressive capacities in handling cross-domain tasks. Experiments on fourteen datasets across node/graph classification/regression, 3D object recognition, and distributed action recognition, demonstrate that the proposed methods yield gratifying results, on par with those by re-training from scratch.

        ----

        ## [2318] X-Pruner: eXplainable Pruning for Vision Transformers

        **Authors**: *Lu Yu, Wei Xiang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02333](https://doi.org/10.1109/CVPR52729.2023.02333)

        **Abstract**:

        Recently vision transformer models have become prominent models for a range of tasks. These models, however, usually suffer from intensive computational costs and heavy memory requirements, making them impractical for deployment on edge platforms. Recent studies have proposed to prune transformers in an unexplainable manner, which overlook the relationship between internal units of the model and the target class, thereby leading to inferior performance. To alleviate this problem, we propose a novel explainable pruning framework dubbed X-Pruner, which is designed by considering the explainability of the pruning criterion. Specifically, to measure each prunable unit's contribution to predicting each target class, a novel explainability-aware mask is proposed and learned in an end-to-end manner. Then, to preserve the most informative units and learn the layer-wise pruning rate, we adaptively search the layer-wise threshold that differentiates between unpruned and pruned units based on their explainability-aware mask values. To verify and evaluate our method, we apply the X-Pruner on representative transformer models including the DeiT and Swin Transformer. Comprehensive simulation results demonstrate that the proposed X-Pruner outperforms the state-of-the-art black-box methods with significantly reduced computational costs and slight performance degradation. Code is available at https://github.com/vickyyu90/XPruner.

        ----

        ## [2319] Bias in Pruned Vision Models: In-Depth Analysis and Countermeasures

        **Authors**: *Eugenia Iofinova, Alexandra Peste, Dan Alistarh*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02334](https://doi.org/10.1109/CVPR52729.2023.02334)

        **Abstract**:

        Pruning—that is, setting a significant subset of the parameters of a neural network to zero—is one of the most popular methods of model compression. Yet, several recent works have raised the issue that pruning may induce or exacerbate bias in the output of the compressed model. Despite existing evidence for this phenomenon, the relationship between neural network pruning and induced bias is not well-understood. In this work, we systematically investigate and characterize this phenomenon in Convolutional Neural Networks for computer vision. First, we show that it is in fact possible to obtain highly-sparse models, e.g. with less than 10% remaining weights, which do not decrease in accuracy nor substantially increase in bias when compared to dense models. At the same time, we also find that, at higher sparsities, pruned models exhibit higher uncertainty in their outputs, as well as increased correlations, which we directly link to increased bias. We propose easy-to-use criteria which, based only on the uncompressed model, establish whether bias will increase with pruning, and identify the samples most susceptible to biased predictions post-compression. Our code can be found at https://github.com/IST-DASLab/pruned-vision-model-bias.

        ----

        ## [2320] Compacting Binary Neural Networks by Sparse Kernel Selection

        **Authors**: *Yikai Wang, Wenbing Huang, Yinpeng Dong, Fuchun Sun, Anbang Yao*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02335](https://doi.org/10.1109/CVPR52729.2023.02335)

        **Abstract**:

        Binary Neural Network (BNN) represents convolution weights with 1-bit values, which enhances the efficiency of storage and computation. This paper is motivated by a previously revealed phenomenon that the binary kernels in successful BNNs are nearly power-law distributed: their values are mostly clustered into a small number of codewords. This phenomenon encourages us to compact typical BNNs and obtain further close performance through learning non-repetitive kernels within a binary kernel subspace. Specifically, we regard the binarization process as kernel grouping in terms of a binary codebook, and our task lies in learning to select a smaller subset of codewords from the full code-book. We then leverage the Gumbel-Sinkhorn technique to approximate the codeword selection process, and develop the Permutation Straight-Through Estimator (PSTE) that is able to not only optimize the selection process end-to-end but also maintain the non-repetitive occupancy of selected codewords. Experiments verify that our method reduces both the model size and bit-wise computational costs, and achieves accuracy improvements compared with state-of-the-art BNNs under comparable budgets.

        ----

        ## [2321] Deep Deterministic Uncertainty: A New Simple Baseline

        **Authors**: *Jishnu Mukhoti, Andreas Kirsch, Joost van Amersfoort, Philip H. S. Torr, Yarin Gal*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02336](https://doi.org/10.1109/CVPR52729.2023.02336)

        **Abstract**:

        Reliable uncertainty from deterministic single-forward pass models is sought after because conventional methods of uncertainty quantification are computationally expensive. We take two complex single-forward-pass uncertainty approaches, DUQ and SNGP, and examine whether they mainly rely on a well-regularized feature space. Crucially, without using their more complex methods for estimating uncertainty, we find that a single softmax neural net with such a regularized feature-space, achieved via residual connections and spectral normalization, outperforms DUQ and SNGP's epistemic uncertainty predictions using simple Gaussian Discriminant Analysis post-training as a separate feature-space density estimator-without fine-tuning on OoD data, feature ensembling, or input pre-procressing. Our conceptually simple Deep Deterministic Uncertainty (DDU) baseline can also be used to disentangle aleatoric and epistemic uncertainty and performs as well as Deep Ensembles, the state-of-the art for uncertainty prediction, on several OoD bench-marks (CIFAR-10/100 vs SVHN/Tiny-ImageNet, ImageNet vs ImageNet-O), active learning settings across different model architectures, as well as in large scale vision tasks like semantic segmentation, while being computationally cheaper.

        ----

        ## [2322] Understanding Deep Generative Models with Generalized Empirical Likelihoods

        **Authors**: *Suman V. Ravuri, Mélanie Rey, Shakir Mohamed, Marc Peter Deisenroth*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02337](https://doi.org/10.1109/CVPR52729.2023.02337)

        **Abstract**:

        Understanding how well a deep generative model captures a distribution of high-dimensional data remains an important open challenge. It is especially difficult for certain model classes, such as Generative Adversarial Networks and Diffusion Models, whose models do not admit exact likelihoods. In this work, we demonstrate that generalized empirical likelihood (GEL) methods offer a family of diagnostic tools that can identify many deficiencies of deep generative models (DGMs). We show, with appropriate specification of moment conditions, that the proposed method can identify which modes have been dropped, the degree to which DGMs are mode imbalanced, and whether DGMs sufficiently capture intra-class diversity. We show how to combine techniques from Maximum Mean Discrepancy and Generalized Empirical Likelihood to create not only distribution tests that retain per-sample interpretability, but also metrics that include label information. We find that such tests predict the degree of mode dropping and mode imbalance up to 60% better than metrics such as improved precision/recall.

        ----

        ## [2323] Fair Scratch Tickets: Finding Fair Sparse Networks without Weight Training

        **Authors**: *Pengwei Tang, Wei Yao, Zhicong Li, Yong Liu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02338](https://doi.org/10.1109/CVPR52729.2023.02338)

        **Abstract**:

        Recent studies suggest that computer vision models come at the risk of compromising fairness. There are exten-sive works to alleviate unfairness in computer vision using pre-processing, in-processing, and post-processing meth-ods. In this paper, we lead a novel fairness-aware learning paradigm for in-processing methods through the lens of the lottery ticket hypothesis (LTH) in the context of computer vision fairness. We randomly initialize a dense neural net-work and find appropriate binary masks for the weights to obtain fair sparse subnetworks without any weight training. Interestingly, to the best of our knowledge, we are the first to discover that such sparse subnetworks with inborn fair-ness exist in randomly initialized networks, achieving an accuracy-fairness trade-off comparable to that of dense neural networks trained with existing fairness-aware in-processing approaches. We term these fair subnetworks as Fair Scratch Tickets (FSTs). We also theoretically pro-vide fairness and accuracy guarantees for them. In our experiments, we investigate the existence of FSTs on var-ious datasets, target attributes, random initialization meth-ods, sparsity patterns, and fairness surrogates. We also find that FSTs can transfer across datasets and investigate other properties of FSTs.

        ----

        ## [2324] Hard Sample Matters a Lot in Zero-Shot Quantization

        **Authors**: *Huantong Li, Xiangmiao Wu, Fanbing Lv, Daihai Liao, Thomas H. Li, Yonggang Zhang, Bo Han, Mingkui Tan*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02339](https://doi.org/10.1109/CVPR52729.2023.02339)

        **Abstract**:

        Zero-shot quantization (ZSQ) is promising for compressing and accelerating deep neural networks when the data for training full-precision models are inaccessible. In ZSQ, network quantization is performed using synthetic samples, thus, the performance of quantized models depends heavily on the quality of synthetic samples. Nonetheless, we find that the synthetic samples constructed in existing ZSQ methods can be easily fitted by models. Accordingly, quantized models obtained by these methods suffer from significant performance degradation on hard samples. To address this issue, we propose HArd sample Synthesizing and Training (HAST). Specifically, HAST pays more attention to hard samples when synthesizing samples and makes synthetic samples hard to fit when training quantized models. HAST aligns features extracted by full-precision and quantized models to ensure the similarity between features extracted by these two models. Extensive experiments show that HAST significantly outperforms existing ZSQ methods, achieving performance comparable to models that are quantized with real data.

        ----

        ## [2325] PD-Quant: Post-Training Quantization Based on Prediction Difference Metric

        **Authors**: *Jiawei Liu, Lin Niu, Zhihang Yuan, Dawei Yang, Xinggang Wang, Wenyu Liu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02340](https://doi.org/10.1109/CVPR52729.2023.02340)

        **Abstract**:

        Post-training quantization (PTQ) is a neural network compression technique that converts a full-precision model into a quantized model using lower-precision data types. Although it can help reduce the size and computational cost of deep neural networks, it can also introduce quantization noise and reduce prediction accuracy, especially in extremely low-bit settings. How to determine the appropriate quantization parameters (e.g., scaling factors and rounding of weights) is the main problem facing now. Existing methods attempt to determine these parameters by minimize the distance between features before and after quantization, but such an approach only considers local information and may not result in the most optimal quantization parameters. We analyze this issue and propose PD-Quant, a method that addresses this limitation by considering global information. It determines the quantization parameters by using the information of differences between network prediction before and after quantization. In addition, PD-Quant can alleviate the overfitting problem in PTQ caused by the small number of calibration sets by adjusting the distribution of activations. Experiments show that PD-Quant leads to better quantization parameters and improves the prediction accuracy of quantized models, especially in low-bit settings. For example, PD-Quant pushes the accuracy of ResNet-18 up to 53.14% and RegNetX-600MF up to 40.67% in weight 2-bit activation 2-bit. The code is released at https://github.com/hustv1/PD-Quant.

        ----

        ## [2326] Vector Quantization with Self-Attention for Quality-Independent Representation Learning

        **Authors**: *Zhou Yang, Weisheng Dong, Xin Li, Mengluan Huang, Yulin Sun, Guangming Shi*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02341](https://doi.org/10.1109/CVPR52729.2023.02341)

        **Abstract**:

        Recently, the robustness of deep neural networks has drawn extensive attention due to the potential distribution shift between training and testing data (e.g., deep models trained on high-quality images are sensitive to corruption during testing). Many researchers attempt to make the model learn invariant representations from multiple corrupted data through data augmentation or image-pair-based feature distillation to improve the robustness. Inspired by sparse representation in image restoration, we opt to address this issue by learning image-quality-independent feature representation in a simple plug-and-play manner, that is, to introduce discrete vector quantization (VQ) to remove redundancy in recognition models. Specifically, we first add a codebook module to the network to quantize deep features. Then we concatenate them and design a self-attention module to enhance the representation. During training, we enforce the quantization of features from clean and corrupted images in the same discrete embedding space so that an invariant quality-independentfeature representation can be learned to improve the recognition robustness of low-quality images. Qualitative and quantitative experimental results show that our method achieved this goal effectively, leading to a new state-of-the-art result of 43.1 % mCE on ImageNet-C with ResNet50 as the backbone. On other robustness benchmark datasets, such as ImageNet-R, our method also has an accuracy improvement of almost 2%. The source code is available at https://see.xidian.edu.cn/faculty/wsdong/Projects/VQSA.htm

        ----

        ## [2327] Masked Auto-Encoders Meet Generative Adversarial Networks and Beyond

        **Authors**: *Zhengcong Fei, Mingyuan Fan, Li Zhu, Junshi Huang, Xiaoming Wei, Xiaolin Wei*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02342](https://doi.org/10.1109/CVPR52729.2023.02342)

        **Abstract**:

        Masked Auto-Encoder (MAE) pretraining methods randomly mask image patches and then train a vision Transformer to reconstruct the original pixels based on the unmasked patches. While they demonstrates impressive performance for downstream vision tasks, it generally requires a large amount of training resource. In this paper, we introduce a novel Generative Adversarial Networks alike framework, referred to as GAN-MAE, where a generator is used to generate the masked patches according to the remaining visible patches, and a discriminator is employed to predict whether the patch is synthesized by the generator. We believe this capacity of distinguishing whether the image patch is predicted or original is benefit to representation learning. Another key point lies in that the parameters of the vision Transformer backbone in the generator and discriminator are shared. Extensive experiments demonstrate that adversarial training of GAN-MAE framework is more efficient and accordingly outperforms the standard MAE given the same model size, training data, and computation resource. The gains are substantially robust for different model sizes and datasets, in particular, a ViT-B model trained with GAN-MAE for 200 epochs outperforms the MAE with 1600 epochs on fine-tuning top-1 accuracy of ImageNet-1k with much less FLOPs. Besides, our approach also works well at transferring downstream tasks.

        ----

        ## [2328] Sequential Training of GANs Against GAN-Classifiers Reveals Correlated "Knowledge Gaps" Present Among Independently Trained GAN Instances

        **Authors**: *Arkanath Pathak, Nicholas Dufour*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02343](https://doi.org/10.1109/CVPR52729.2023.02343)

        **Abstract**:

        Modern Generative Adversarial Networks (GANs) generate realistic images remarkably well. Previous work has demonstrated the feasibility of “GAN-classifiers” that are distinct from the co-trained discriminator, and operate on images generated from a frozen GAN. That such classifiers work at all affirms the existence of “knowledge gaps” (out-of-distribution artifacts across samples) present in GAN training. We iteratively train GAN-classifiers and train GANs that “fool” the classifiers (in an attempt to fill the knowledge gaps), and examine the effect on GAN training dynamics, output quality, and GAN-classifier generalization. We investigate two settings, a small DCGAN architecture trained on low dimensional images (MNIST), and StyleGAN2, a SOTA GAN architecture trained on high dimensional images (FFHQ). We find that the DCGAN is unable to effectively fool a held-out GAN-classifier without compromising the output quality. However, the StyleGAN2 can fool held-out classifiers with no change in output quality, and this effect persists over multiple rounds of GAN/classifier training which appears to reveal an ordering over optima in the generator parameter space. Finally, we study different classifier architectures and show that the architecture of the GAN-classifier has a strong influence on the set of its learned artifacts.

        ----

        ## [2329] Edges to Shapes to Concepts: Adversarial Augmentation for Robust Vision

        **Authors**: *Aditay Tripathi, Rishubh Singh, Anirban Chakraborty, Pradeep Shenoy*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02344](https://doi.org/10.1109/CVPR52729.2023.02344)

        **Abstract**:

        Recent work has shown that deep vision models tend to be overly dependent on low-level or “texture” features, leading to poor generalization. Various data augmentation strategies have been proposed to overcome this so-called texture bias in DNNs. We propose a simple, lightweight adversarial augmentation technique that explicitly incentivizes the network to learn holistic shapes for accurate prediction in an object classification setting. Our augmentations superpose edgemaps from one image onto another image with shuffled patches, using a randomly determined mixing proportion, with the image label of the edgemap image. To classify these augmented images, the model needs to not only detect and focus on edges but distinguish between relevant and spurious edges. We show that our augmentations significantly improve classification accuracy and robustness measures on a range of datasets and neural architectures. As an example, for ViT-S, We obtain absolute gains on classification accuracy gains up to 6%. We also obtain gains of up to 28% and 8.5% on natural adversarial and out-of-distribution datasets like ImageNet-A (for ViT-B) and ImageNet-R (for ViT-S), respectively. Analysis using a range of probe datasets shows substantially increased shape sensitivity in our trained models, explaining the observed improvement in robustness and classification accuracy.

        ----

        ## [2330] Towards Universal Fake Image Detectors that Generalize Across Generative Models

        **Authors**: *Utkarsh Ojha, Yuheng Li, Yong Jae Lee*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02345](https://doi.org/10.1109/CVPR52729.2023.02345)

        **Abstract**:

        With generative models proliferating at a rapid rate, there is a growing need for general purpose fake image detectors. In this work, we first show that the existing paradigm, which consists of training a deep network for real-vs-fake classification, fails to detect fake images from newer breeds of generative models when trained to detect GAN fake images. Upon analysis, we find that the resulting classifier is asymmetrically tuned to detect patterns that make an image fake. The real class becomes a ‘sink’ class holding anything that is not fake, including generated images from models not accessible during training. Building upon this discovery, we propose to perform real-vs-fake classification without learning; i.e., using a feature space not explicitly trained to distinguish real from fake images. We use nearest neighbor and linear probing as instantiations of this idea. When given access to the feature space of a large pretrained vision-language model, the very simple baseline of nearest neighbor classification has surprisingly good generalization ability in detecting fake images from a wide variety of generative models; e.g., it improves upon the SoTA [50] by +15.07 mAP and +25.90% acc when tested on unseen diffusion and autoregressive models. Our code, models, and data can be found at https://github.com/Yuheng-Li/UniversalFakeDetect

        ----

        ## [2331] Explicit Boundary Guided Semi-Push-Pull Contrastive Learning for Supervised Anomaly Detection

        **Authors**: *Xincheng Yao, Ruoqi Li, Jing Zhang, Jun Sun, Chongyang Zhang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02346](https://doi.org/10.1109/CVPR52729.2023.02346)

        **Abstract**:

        Most anomaly detection (AD) models are learned using only normal samples in an unsupervised way, which may result in ambiguous decision boundary and insufficient discriminability. In fact, a few anomaly samples are often available in real-world applications, the valuable knowledge of known anomalies should also be effectively exploited. However, utilizing a few known anomalies during training may cause another issue that the model may be biased by those known anomalies and fail to generalize to unseen anomalies. In this paper, we tackle supervised anomaly detection, i.e., we learn AD models using a few available anomalies with the objective to detect both the seen and unseen anomalies. We propose a novel explicit boundary guided semi-push-pull contrastive learning mechanism, which can enhance model's discriminability while mitigating the bias issue. Our approach is based on two core designs: First, we find an explicit and compact separating boundary as the guidance for further feature learning. As the boundary only relies on the normal feature distribution, the bias problem caused by a few known anomalies can be alleviated. Second, a boundary guided semi-push-pull loss is developed to only pull the normal features together while pushing the abnormal features apart from the separating boundary beyond a certain margin region. In this way, our model can form a more explicit and discriminative decision boundary to distinguish known and also unseen anomalies from normal samples more effectively. Code will be available at https://github.com/xcyao00/BGAD.

        ----

        ## [2332] Generating Anomalies for Video Anomaly Detection with Prompt-based Feature Mapping

        **Authors**: *Zuhao Liu, Xiao-Ming Wu, Dian Zheng, Kun-Yu Lin, Wei-Shi Zheng*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02347](https://doi.org/10.1109/CVPR52729.2023.02347)

        **Abstract**:

        Anomaly detection in surveillance videos is a challenging computer vision task where only normal videos are available during training. Recent work released the first virtual anomaly detection dataset to assist real-world detection. However, an anomaly gap exists because the anomalies are bounded in the virtual dataset but unbounded in the real world, so it reduces the generalization ability of the virtual dataset. There also exists a scene gap between virtual and real scenarios, including scene-specific anomalies (events that are abnormal in one scene but normal in another) and scene-specific attributes, such as the viewpoint of the surveillance camera. In this paper, we aim to solve the problem of the anomaly gap and scene gap by proposing a prompt-based feature mapping framework (PFMF). The PFMF contains a mapping network guided by an anomaly prompt to generate unseen anomalies with unbounded types in the real scenario, and a mapping adaptation branch to narrow the scene gap by applying domain classifier and anomaly classifier. The proposed framework outperforms the state-of-the-art on three benchmark datasets. Extensive ablation experiments also show the effectiveness of our framework design.

        ----

        ## [2333] Revisiting Reverse Distillation for Anomaly Detection

        **Authors**: *Tran Dinh Tien, Anh Tuan Nguyen, Nguyen Hoang Tran, Ta Duc Huy, Soan Thi Minh Duong, Chanh D. Tr. Nguyen, Steven Q. H. Truong*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02348](https://doi.org/10.1109/CVPR52729.2023.02348)

        **Abstract**:

        Anomaly detection is an important application in large-scale industrial manufacturing. Recent methods for this task have demonstrated excellent accuracy but come with a latency trade-off. Memory based approaches with dominant performances like PatchCore or Coupled-hypersphere-based Feature Adaptation (CFA) require an external memory bank, which significantly lengthens the execution time. Another approach that employs Reversed Distillation (RD) can perform well while maintaining low latency. In this paper, we revisit this idea to improve its performance, establishing a new state-of-the-art benchmark on the challenging MVTec dataset for both anomaly detection and localization. The proposed method, called RD++, runs six times faster than PatchCore, and two times faster than CFA but introduces a negligible latency compared to RD. We also experiment on the BTAD and Retinal OCT datasets to demonstrate our method's generalizability and conduct important ablation experiments to provide insights into its configurations. Source code will be available at https://github.com/tientrandinh/Revisiting-Reverse-Distillation.

        ----

        ## [2334] MetaMix: Towards Corruption-Robust Continual Learning with Temporally Self-Adaptive Data Transformation

        **Authors**: *Zhenyi Wang, Li Shen, Donglin Zhan, Qiuling Suo, Yanjun Zhu, Tiehang Duan, Mingchen Gao*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02349](https://doi.org/10.1109/CVPR52729.2023.02349)

        **Abstract**:

        Continual Learning (CL) has achieved rapid progress in recent years. However, it is still largely unknown how to determine whether a CL model is trustworthy and how to foster its trustworthiness. This work focuses on evaluating and improving the robustness to corruptions of existing CL models. Our empirical evaluation results show that existing state-of-the-art (SOTA) CL models are particularly vulnerable to various data corruptions during testing. To make them trustworthy and robust to corruptions deployed in safety-critical scenarios, we propose a meta-learning framework of self-adaptive data augmentation to tackle the corruption robustness in CL. The proposed framework, MetaMix, learns to augment and mix data, automatically transforming the new task data or memory data. It directly optimizes the generalization performance against data corruptions during training. To evaluate the corruption robustness of our proposed approach, we construct several CL corruption datasets with different levels of severity. We perform comprehensive experiments on both task- and class-continual learning. Extensive experiments demonstrate the effectiveness of our proposed method compared to SOTA baselines.

        ----

        ## [2335] ScaleFL: Resource-Adaptive Federated Learning with Heterogeneous Clients

        **Authors**: *Fatih Ilhan, Gong Su, Ling Liu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02350](https://doi.org/10.1109/CVPR52729.2023.02350)

        **Abstract**:

        Federated learning (FL) is an attractive distributed learning paradigm supporting real-time continuous learning and client privacy by default. In most FL approaches, all edge clients are assumed to have sufficient computation capabilities to participate in the learning of a deep neural network (DNN) model. However, in real-life applications, some clients may have severely limited resources and can only train a much smaller local model. This paper presents ScaleFL, a novel FL approach with two distinctive mechanisms to handle resource heterogeneity and provide an equitable FL framework for all clients. First, ScaleFL adaptively scales down the DNN model along width and depth dimensions by leveraging early exits to find the best-fit models for resource-aware local training on distributed clients. In this way, ScaleFL provides an efficient balance of preserving basic and complex features in local model splits with various sizes for joint training while enabling fast inference for model deployment. Second, ScaleFL utilizes self-distillation among exit predictions during training to improve aggregation through knowledge transfer among subnetworks. We conduct extensive experiments on benchmark CV (CIFAR-10/100, ImageNet) and NLP datasets (SST-2, AgNews). We demonstrate that ScaleFL outperforms existing representative heterogeneous FL approaches in terms of global/local model performance and provides inference efficiency, with up to 2x latency and 4x model size reduction with negligible performance drop below 2%.

        ----

        ## [2336] Confidence-Aware Personalized Federated Learning via Variational Expectation Maximization

        **Authors**: *Junyi Zhu, Xingchen Ma, Matthew B. Blaschko*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02351](https://doi.org/10.1109/CVPR52729.2023.02351)

        **Abstract**:

        Federated Learning (FL) is a distributed learning scheme to train a shared model across clients. One common and fundamental challenge in FL is that the sets of dnta across clients could be non-identically distributed and have different sizes. Personalized Federated Learning (P FL) attempts to solve this challenge via locally adapted models. In this work, we present a novel framework for PFL based on hierarchical Bayesian modeling and variational inference. A global model is introduced as a latent variable to augment the joint distribution of clients' parameters and capture the common trends of different clients, optimization is derived based on the principle of maximizing the marginal likelihood and conducted using variational expectation maximization. Our algorithm gives rise to a closed-form estimation of a confidence value which comprises the uncertainty of clients' parameters and local model deviations from the global model. The confidence value is used to weigh clients' parameters in the aggregation stage and adjust the regularization effect of the global model. We evaluate our method through extensive empirical studies on multiple datasets. Experimental results show that our approach obtains competitive results under mild heterogeneous circumstances while significantly outperforming state-of-the-art PFL frameworks in highly heterogeneous settings.

        ----

        ## [2337] Make Landscape Flatter in Differentially Private Federated Learning

        **Authors**: *Yifan Shi, Yingqi Liu, Kang Wei, Li Shen, Xueqian Wang, Dacheng Tao*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02352](https://doi.org/10.1109/CVPR52729.2023.02352)

        **Abstract**:

        To defend the inference attacks and mitigate the sensitive information leakages in Federated Learning (FL), clientlevel Differentially Private FL (DPFL) is the de-facto standard for privacy protection by clipping local updates and adding random noise. However, existing DPFL methods tend to make a sharper loss landscape and have poorer weight perturbation robustness, resulting in severe performance degradation. To alleviate these issues, we propose a novel DPFL algorithm named DP-FedSAM, which leverages gradient perturbation to mitigate the negative impact of DP. Specifically, DP-FedSAM integrates Sharpness Aware Minimization (SAM) optimizer to generate local flatness models with better stability and weight perturbation robustness, which results in the small norm of local updates and robustness to DP noise, thereby improving the performance. From the theoretical perspective, we analyze in detail how DP-FedSAM mitigates the performance degradation induced by DP. Meanwhile, we give rigorous privacy guarantees with Rényi DP and present the sensitivity analysis of local updates. At last, we empirically confirm that our algorithm achieves state-of-the-art (SOTA) performance compared with existing SOTA baselines in DPFL.

        ----

        ## [2338] Rethinking Domain Generalization for Face Anti-spoofing: Separability and Alignment

        **Authors**: *Yiyou Sun, Yaojie Liu, Xiaoming Liu, Yixuan Li, Wen-Sheng Chu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02353](https://doi.org/10.1109/CVPR52729.2023.02353)

        **Abstract**:

        This work studies the generalization issue of face anti-spoofing (FAS) models on domain gaps, such as image resolution, blurriness and sensor variations. Most prior works regard domain-specific signals as a negative impact, and apply metric learning or adversarial losses to remove them from feature representation. Though learning a domain-invariant feature space is viable for the training data, we show that the feature shift still exists in an unseen test domain, which backfires on the generalizability of the classifier. In this work, instead of constructing a domain-invariant feature space, we encourage domain separability while aligning the live-to-spoof transition (i.e., the trajectory from live to spoof) to be the same for all domains. We formulate this FAS strategy of separability and alignment (SA-FAS) as a problem of invariant risk minimization (IRM), and learn domain-variant feature representation but domain-invariant classifier. We demonstrate the effectiveness of SA-FAS on challenging cross-domain FAS datasets and establish state-of-the-art performance. Code is available at https://github.com/sunyiyou/SAFAS.

        ----

        ## [2339] StyleAdv: Meta Style Adversarial Training for Cross-Domain Few-Shot Learning

        **Authors**: *Yuqian Fu, Yu Xie, Yanwei Fu, Yu-Gang Jiang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02354](https://doi.org/10.1109/CVPR52729.2023.02354)

        **Abstract**:

        Cross-Domain Few-Shot Learning (CD-FSL) is a recently emerging task that tackles few-shot learning across different domains. It aims at transferring prior knowledge learned on the source dataset to novel target datasets. The CD-FSL task is especially challenged by the huge domain gap between different datasets. Critically, such a domain gap actually comes from the changes of visual styles, and wave-SAN [10] empirically shows that spanning the style distribution of the source data helps alleviate this issue. However, wave-SAN simply swaps styles of two images. Such a vanilla operation makes the generated styles “real” and “easy”, which still fall into the original set of the source styles. Thus, inspired by vanilla adversarial learning, a novel model-agnostic meta Style Adversarial training (StyleAdv) method together with a novel style adversarial attack method is proposed for CD-FSL. Particularly, our style attack method synthesizes both “virtual” and “hard” adversarial styles for model training. This is achieved by perturbing the original style with the signed style gradients. By continually attacking styles and forcing the model to recognize these challenging adversarial styles, our model is gradually robust to the visual styles, thus boosting the generalization ability for novel target datasets. Besides the typical CNN-based backbone, we also employ our StyleAdv method on large-scale pre-trained vision transformer. Extensive experiments conducted on eight various target datasets show the effectiveness of our method. Whether built upon ResNet or ViT, we achieve the new state of the art for CD-FSL. Code is available at https://github.com/lovelyqian/StyleAdv-CDFSL.

        ----

        ## [2340] The Dark Side of Dynamic Routing Neural Networks: Towards Efficiency Backdoor Injection

        **Authors**: *Simin Chen, Hanlin Chen, Mirazul Haque, Cong Liu, Wei Yang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02355](https://doi.org/10.1109/CVPR52729.2023.02355)

        **Abstract**:

        Recent advancements in deploying deep neural networks (DNNs) on resource-constrained devices have generated interest in input-adaptive dynamic neural networks (DyNNs). DyNNs offer more efficient inferences and enable the deployment of DNNs on devices with limited resources, such as mobile devices. However, we have discovered a new vulnerability in DyNNs that could potentially compromise their efficiency. Specifically, we investigate whether adversaries can manipulate DyNNs' computational costs to create a false sense of efficiency. To address this question, we propose EfficFrog, an adversarial attack that injects universal efficiency backdoors in DyNNs. To inject a backdoor trigger into DyNNs, EfficFrog poisons only a minimal percentage of the DyNNs' training data. During the inference phase, EfficFrog can slow down the backdoored DyNNs and abuse the computational resources of systems running DyNNs by adding the trigger to any input. To evaluate EfficFrog, we tested it on three DNN backbone architectures (based on VGG16, MobileNet, and ResNet56) using two popular datasets (CIFAR-10 and Tiny ImageNet). Our results demonstrate that EfficFrog reduces the efficiency of DyNNs on triggered input samples while keeping the efficiency of clean samples almost the same.

        ----

        ## [2341] Architectural Backdoors in Neural Networks

        **Authors**: *Mikel Bober-Irizar, Ilia Shumailov, Yiren Zhao, Robert D. Mullins, Nicolas Papernot*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02356](https://doi.org/10.1109/CVPR52729.2023.02356)

        **Abstract**:

        Machine learning is vulnerable to adversarial manipulation. Previous literature demonstrated that at the training stage attackers can manipulate data [14] and data sampling procedures [29] to control model behaviour. A common attack goal is to plant backdoors i.e. force the victim model to learn to recognise a trigger known only by the adversary. In this paper, we introduce a new class of backdoor attacks that hide inside model architectures i.e. in the inductive bias of the functions used to train. These backdoors are simple to implement, for instance by publishing open-source code for a backdoored model architecture that others will reuse unknowingly. We demonstrate that model architectural backdoors represent a real threat and, unlike other approaches, can survive a complete re-training from scratch. We formalise the main construction principles behind architectural backdoors, such as a connection between the input and the output, and describe some possible protections against them. We evaluate our attacks on computer vision benchmarks of different scales and demonstrate the underlying vulnerability is pervasive in a variety of common training settings.

        ----

        ## [2342] You Are Catching My Attention: Are Vision Transformers Bad Learners under Backdoor Attacks?

        **Authors**: *Zenghui Yuan, Pan Zhou, Kai Zou, Yu Cheng*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02357](https://doi.org/10.1109/CVPR52729.2023.02357)

        **Abstract**:

        Vision Transformers (ViTs), which made a splash in the field of computer vision (CV), have shaken the dominance of convolutional neural networks (CNNs). However, in the process of industrializing ViTs, backdoor attacks have brought severe challenges to security. The success of ViTs benefits from the self-attention mechanism. However, compared with CNNs, we find that this mechanism of capturing global information within patches makes ViTs more sensitive to patch-wise triggers. Under such observations, we delicately design a novel backdoor attack framework for ViTs, dubbed BadViT, which utilizes a universal patch-wise trigger to catch the model's attention from patches beneficial for classification to those with triggers, thereby manipulating the mechanism on which ViTs survive to confuse itself. Furthermore, we propose invisible variants of BadViT to increase the stealth of the attack by limiting the strength of the trigger perturbation. Through a large number of experiments, it is proved that BadViT is an efficient backdoor attack method against ViTs, which is less dependent on the number of poisons, with satisfactory convergence, and is transferable for downstream tasks. Furthermore, the risks inside of ViTs to backdoor attacks are also explored from the perspective of existing advanced defense schemes.

        ----

        ## [2343] A Practical Upper Bound for the Worst-Case Attribution Deviations

        **Authors**: *Fan Wang, Adams Wai-Kin Kong*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02358](https://doi.org/10.1109/CVPR52729.2023.02358)

        **Abstract**:

        Model attribution is a critical component of deep neural networks (DNNs) for its interpretability to complex models. Recent studies bring up attention to the security of attribution methods as they are vulnerable to attribution attacks that generate similar images with dramatically different attributions. Existing works have been investigating empirically improving the robustness of DNNs against those attacks; however, none of them explicitly quantifies the actual deviations of attributions. In this work, for the first time, a constrained optimization problem is formulated to derive an upper bound that measures the largest dissimilarity of attributions after the samples are perturbed by any noises within a certain region while the classification results remain the same. Based on the formulation, different practical approaches are introduced to bound the attributions above using Euclidean distance and cosine similarity under both 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$\ell_{2}$</tex>
 and 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$\ell_{\infty}$</tex>
-norm perturbations constraints. The bounds developed by our theoretical study are validated on various datasets and two different types of attacks (PGD attack and IFIA attribution attack). Over 10 million attacks in the experiments indicate that the proposed upper bounds effectively quantify the robustness of models based on the worst-case attribution dissimilarities.

        ----

        ## [2344] Sibling-Attack: Rethinking Transferable Adversarial Attacks against Face Recognition

        **Authors**: *Zexin Li, Bangjie Yin, Taiping Yao, Junfeng Guo, Shouhong Ding, Simin Chen, Cong Liu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02359](https://doi.org/10.1109/CVPR52729.2023.02359)

        **Abstract**:

        A hard challenge in developing practical face recognition (FR) attacks is due to the black-box nature of the target FR model, i.e., inaccessible gradient and parameter information to attackers. While recent research took an important step towards attacking black-box FR models through leveraging transferability, their performance is still limited, especially against online commercial FR systems that can be pessimistic (e.g., a less than 50% ASR-attack success rate on average). Motivated by this, we present Sibling-Attack, a new FR attack technique for the first time explores a novel multi-task perspective (i.e., leveraging extra information from multi-correlated tasks to boost attacking transferability). Intuitively, Sibling-Attack selects a set of tasks correlated with FR and picks the Attribute Recognition (AR) task as the task used in Sibling-Attack based on theoretical and quantitative analysis. Sibling-Attack then develops an optimization framework that fuses adversarial gradient information through (1) constraining the cross-task features to be under the same space, (2) a Jointtask meta optimization framework that enhances the gradient compatibility among tasks, and (3) a cross-task gradient stabilization method which mitigates the oscillation effect during attacking. Extensive experiments demonstrate that SiblingAttack outperforms state-of-the-art FR attack techniques by a non-trivial margin, boosting ASR by 12.61% and 55.77% on average on state-of-the-art pre-trained FR models and two well-known, widely used commercial FR systems.

        ----

        ## [2345] Angelic Patches for Improving Third-Party Object Detector Performance

        **Authors**: *Wenwen Si, Shuo Li, Sangdon Park, Insup Lee, Osbert Bastani*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02360](https://doi.org/10.1109/CVPR52729.2023.02360)

        **Abstract**:

        Deep learning models have shown extreme vulnerability to distribution shifts such as synthetic perturbations and spatial transformations. In this work, we explore whether we can adopt the characteristics of adversarial attack methods to help improve robustness of object detection to distribution shifts such as synthetic perturbations and spatial transformations. We study a class of realistic object detection settings wherein the target objects have control over their appearance. To this end, we propose a reversed Fast Gradient Sign Method (FGSM) to obtain these angelic patches that significantly increase the detection probability, even without pre-knowledge of the perturbations. In detail, we apply the patch to each object instance simultaneously, strengthening not only classification, but also bounding box accuracy. Experiments demonstrate the efficacy of the partial-covering patch in solving the complex bounding box problem. More importantly, the performance is also transferable to different detection models even under severe affine transformations and deformable shapes. To the best of our knowledge, we are the first object detection patch that achieves both cross-model efficacy and multiple patches. We observed average accuracy improvements of 30% in the real-world experiments. Our code is available at: https://github.com/averysi224/angelic_patches.

        ----

        ## [2346] Introducing Competition to Boost the Transferability of Targeted Adversarial Examples Through Clean Feature Mixup

        **Authors**: *Junyoung Byun, Myung-Joon Kwon, Seungju Cho, Yoonji Kim, Changick Kim*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02361](https://doi.org/10.1109/CVPR52729.2023.02361)

        **Abstract**:

        Deep neural networks are widely known to be susceptible to adversarial examples, which can cause incorrect predictions through subtle input modifications. These adversarial examples tend to be transferable between models, but targeted attacks still have lower attack success rates due to significant variations in decision boundaries. To enhance the transferability of targeted adversarial examples, we propose introducing competition into the optimization process. Our idea is to craft adversarial perturbations in the presence of two new types of competitor noises: adversarial perturbations towards different target classes and friendly perturbations towards the correct class. With these competitors, even if an adversarial example deceives a network to extract specific features leading to the target class, this disturbance can be suppressed by other competitors. Therefore, within this competition, adversarial examples should take different attack strategies by leveraging more diverse features to overwhelm their interference, leading to improving their transferability to different models. Considering the computational complexity, we efficiently simulate various interference from these two types of competitors in feature space by randomly mixing up stored clean features in the model inference and named this method Clean Feature Mixup (CFM). Our extensive experimental results on the ImageNet-Compatible and CIFAR-10 datasets show that the proposed method outperforms the existing baselines with a clear margin. Our code is available at https://github.com/dreamflake/CFM.

        ----

        ## [2347] Towards Compositional Adversarial Robustness: Generalizing Adversarial Training to Composite Semantic Perturbations

        **Authors**: *Lei Hsiung, Yun-Yun Tsai, Pin-Yu Chen, Tsung-Yi Ho*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02362](https://doi.org/10.1109/CVPR52729.2023.02362)

        **Abstract**:

        Model robustness against adversarial examples of single perturbation type such as the 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$\ell_{p}$</tex>
 -norm has been widely studied, yet its generalization to more realistic scenarios involving multiple semantic perturbations and their composition remains largely unexplored. In this paper, we first propose a novel method for generating composite adversarial examples. Our method can find the optimal attack composition by utilizing component-wise projected gradient descent and automatic attack-order scheduling. We then propose generalized adversarial training (GAT) to extend model robustness from 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$\ell_{p}$</tex>
 -ball to composite semantic perturbations, such as the combination of Hue, Saturation, Brightness, Contrast, and Rotation. Results obtained using ImageNet and CIFAR-10 datasets indicate that GAT can be robust not only to all the tested types of a single attack, but also to any combination of such attacks. GAT also outperforms baseline 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$\ell_{\infty}$</tex>
 -norm bounded adversarial training approaches by a significant margin.

        ----

        ## [2348] Boosting Accuracy and Robustness of Student Models via Adaptive Adversarial Distillation

        **Authors**: *Bo Huang, Mingyang Chen, Yi Wang, Junda Lu, Minhao Cheng, Wei Wang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02363](https://doi.org/10.1109/CVPR52729.2023.02363)

        **Abstract**:

        Distilled student models in teacher-student architectures are widely considered for computational-effective deployment in real-time applications and edge devices. However, there is a higher risk of student models to encounter adversarial attacks at the edge. Popular enhancing schemes such as adversarial training have limited performance on compressed networks. Thus, recent studies concern about adversarial distillation (AD) that aims to inherit not only prediction accuracy but also adversarial robustness of a robust teacher model under the paradigm of robust optimization. In the min-max framework of AD, existing AD methods generally use fixed supervision information from the teacher model to guide the inner optimization for knowledge distillation which often leads to an overcorrection towards model smoothness. In this paper, we propose an adaptive adversarial distillation (AdaAD) that involves the teacher model in the knowledge optimization process in a way interacting with the student model to adaptively search for the inner results. Comparing with state-of-the-art methods, the proposed AdaAD can significantly boost both the prediction accuracy and adversarial robustness of student models in most scenarios. In particular, the ResNet-18 model trained by AdaAD achieves top-rank performance (54.23% robust accuracy) on RobustBench under AutoAttack.

        ----

        ## [2349] The Enemy of My Enemy is My Friend: Exploring Inverse Adversaries for Improving Adversarial Training

        **Authors**: *Junhao Dong, Seyed-Mohsen Moosavi-Dezfooli, Jianhuang Lai, Xiaohua Xie*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02364](https://doi.org/10.1109/CVPR52729.2023.02364)

        **Abstract**:

        Although current deep learning techniques have yielded superior performance on various computer vision tasks, yet they are still vulnerable to adversarial examples. Adversarial training and its variants have been shown to be the most effective approaches to defend against adversarial examples. A particular class of these methods regularize the difference between output probabilities for an adversarial and its corresponding natural example. However, it may have a negative impact if a natural example is misclassified. To circumvent this issue, we propose a novel adversarial training scheme that encourages the model to produce similar output probabilities for an adversarial example and its “inverse adversarial” counterpart. Particularly, the counterpart is generated by maximizing the likelihood in the neighborhood of the natural example. Extensive experiments on various vision datasets and architectures demonstrate that our training method achieves state-of-the-art robustness as well as natural accuracy among robust models. Furthermore, using a universal version of inverse adversarial examples, we improve the performance of single-step adversarial training techniques at a low computational cost.

        ----

        ## [2350] Robust Single Image Reflection Removal Against Adversarial Attacks

        **Authors**: *Zhenbo Song, Zhenyuan Zhang, Kaihao Zhang, Wenhan Luo, Zhaoxin Fan, Wenqi Ren, Jianfeng Lu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02365](https://doi.org/10.1109/CVPR52729.2023.02365)

        **Abstract**:

        This paper addresses the problem of robust deep single-image reflection removal (SIRR) against adversarial attacks. Current deep learning based SIRR methods have shown significant performance degradation due to unnoticeable distortions and perturbations on input images. For a comprehensive robustness study, we first conduct diverse adversarial attacks specifically for the SIRR problem, i.e. towards different attacking targets and regions. Then we propose a robust SIRR model, which integrates the cross-scale attention module, the multi-scale fusion module, and the adversarial image discriminator. By exploiting the multi-scale mechanism, the model narrows the gap between features from clean and adversarial images. The image discriminator adaptively distinguishes clean or noisy inputs, and thus further gains reliable robustness. Extensive experiments on Nature, SIR
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">2</sup>
, and Real datasets demonstrate that our model remarkably improves the robustness of SIRR across disparate scenes.

        ----

        ## [2351] Physical-World Optical Adversarial Attacks on 3D Face Recognition

        **Authors**: *Yanjie Li, Yiquan Li, Xuelong Dai, Songtao Guo, Bin Xiao*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02366](https://doi.org/10.1109/CVPR52729.2023.02366)

        **Abstract**:

        The success rate of current adversarial attacks remains low on real-world 3D face recognition tasks because the 3D-printing attacks need to meet the requirement that the generated points should be adjacent to the surface, which limits the adversarial example’ searching space. Additionally, they have not considered unpredictable head movements or the non-homogeneous nature of skin reflectance in the real world. To address the real-world challenges, we propose a novel structured-light attack against structured-light-based 3D face recognition. We incorporate the 3D reconstruction process and skin's reflectance in the optimization process to get the end-to-end attack and present 3D transform invariant loss and sensitivity maps to improve robustness. Our attack enables adversarial points to be placed in any position and is resilient to random head movements while maintaining the perturbation unnoticeable. Experiments show that our new method can attack point-cloud-based and depth-image-based 3D face recognition systems with a high success rate, using fewer perturbations than previous physical 3D adversarial attacks.

        ----

        ## [2352] AUNet: Learning Relations Between Action Units for Face Forgery Detection

        **Authors**: *Weiming Bai, Yufan Liu, Zhipeng Zhang, Bing Li, Weiming Hu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02367](https://doi.org/10.1109/CVPR52729.2023.02367)

        **Abstract**:

        Face forgery detection becomes increasingly crucial due to the serious security issues caused by face manipulation techniques. Recent studies in deepfake detection have yielded promising results when the training and testing face forgeries are from the same domain. However, the problem remains challenging when one tries to generalize the detector to forgeries created by unseen methods during training. Observing that face manipulation may alter the relation between different facial action units (AU), we propose the Action-Units Relation Learning framework to improve the generality of forgery detection. In specific, it consists of the Action Units Relation Transformer (ART) and the Tampered AU Prediction (TAP). The ART constructs the relation between different AUs with AU-agnostic Branch and AU-specific Branch, which complement each other and work together to exploit forgery clues. In the Tampered AU Prediction, we tamper AU-related regions at the image level and develop challenging pseudo samples at the feature level. The model is then trained to predict the tampered AU regions with the generated location-specific supervision. Experimental results demonstrate that our method can achieve state-of-the-art performance in both the in-dataset and cross-dataset evaluations.

        ----

        

[Go to the previous page](CVPR-2023-list11.md)

[Go to the catalog section](README.md)