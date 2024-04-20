## [1800] Weakly Supervised Temporal Sentence Grounding with Uncertainty-Guided Self-training

        **Authors**: *Yifei Huang, Lijin Yang, Yoichi Sato*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01813](https://doi.org/10.1109/CVPR52729.2023.01813)

        **Abstract**:

        The task of weakly supervised temporal sentence grounding aims at finding the corresponding temporal moments of a language description in the video, given video-language correspondence only at video-level. Most existing works select mismatched video-language pairs as negative samples and train the model to generate better positive proposals that are distinct from the negative ones. However, due to the complex temporal structure of videos, proposals distinct from the negative ones may correspond to several video segments but not necessarily the correct ground truth. To alleviate this problem, we propose an uncertainty-guided self-training technique to provide extra self-supervision signal to guide the weakly-supervised learning. The self-training process is based on teacher-student mutual learning with weak-strong augmentation, which enables the teacher network to generate relatively more reliable outputs compared to the student network, so that the student network can learn from the teacher's output. Since directly applying existing self-training methods in this task easily causes error accumulation, we specifically design two techniques in our selftraining method: (1) we construct a Bayesian teacher network, leveraging its uncertainty as a weight to suppress the noisy teacher supervisory signals; (2) we leverage the cycle consistency brought by temporal data augmentation to perform mutual learning between the two networks. Experiments demonstrate our method's superiority on Charades-STA and ActivityNet Captions datasets. We also show in the experiment that our self-training method can be applied to improve the performance of multiple backbone methods.

        ----

        ## [1801] SViTT: Temporal Learning of Sparse Video-Text Transformers

        **Authors**: *Yi Li, Kyle Min, Subarna Tripathi, Nuno Vasconcelos*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01814](https://doi.org/10.1109/CVPR52729.2023.01814)

        **Abstract**:

        Do video-text transformers learn to model temporal relationships across frames? Despite their immense capacity and the abundance of multimodal training data, recent work has revealed the strong tendency of video-text models towards frame-based spatial representations, while temporal reasoning remains largely unsolved. In this work, we identify several key challenges in temporal learning of video-text transformers: the spatiotemporal trade-off from limited network size; the curse of dimensionality for multi-frame modeling; and the diminishing returns of semantic information by extending clip length. Guided by these findings, we propose SVi TT, a sparse video-text architecture that performs multi-frame reasoning with significantly lower cost than naïve transformers with dense attention. Analogous to graph-based networks, SViTT employs two forms of sparsity: edge sparsity that limits the query-key communications between tokens in self-attention, and node sparsity that discards uninformative visual tokens. Trained with a curriculum which increases model sparsity with the clip length, SVi TT outperforms dense transformer baselines on multiple video-text retrieval and question answering benchmarks, with a fraction of computational cost. Project page: http://svcl.ucsd.edu/projects/svitt.

        ----

        ## [1802] AutoAD: Movie Description in Context

        **Authors**: *Tengda Han, Max Bain, Arsha Nagrani, Gül Varol, Weidi Xie, Andrew Zisserman*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01815](https://doi.org/10.1109/CVPR52729.2023.01815)

        **Abstract**:

        The objective of this paper is an automatic Audio Description (AD) model that ingests movies and outputs AD in text form. Generating high-quality movie AD is challenging due to the dependency of the descriptions on context, and the limited amount of training data available. In this work, we leverage the power of pretrained foundation models, such as GPT and CLIP, and only train a mapping network that bridges the two models for visually-conditioned text generation. In order to obtain high-quality AD, we make the following four contributions: (i) we incorporate context from the movie clip, AD from previous clips, as well as the subtitles; (ii) we address the lack of training data by pretraining on large-scale datasets, where visual or contextual information is unavailable, e.g. text-only AD without movies or visual captioning datasets without context; (iii) we improve on the currently available AD datasets, by removing label noise in the MAD dataset, and adding character naming information; and (iv) we obtain strong results on the movie AD task compared with previous methods.

        ----

        ## [1803] Text with Knowledge Graph Augmented Transformer for Video Captioning

        **Authors**: *Xin Gu, Guang Chen, Yufei Wang, Libo Zhang, Tiejian Luo, Longyin Wen*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01816](https://doi.org/10.1109/CVPR52729.2023.01816)

        **Abstract**:

        Video captioning aims to describe the content of videos using natural language. Although significant progress has been made, there is still much room to improve the performance for real-world applications, mainly due to the long-tail words challenge. In this paper, we propose a text with knowledge graph augmented transformer (TextKG)for video captioning. Notably, TextKG is a two-stream transformer, formed by the external stream and internal stream. The external stream is designed to absorb additional knowledge, which models the interactions between the additional knowledge, e.g., pre-built knowledge graph, and the built-in information of videos, e.g., the salient object regions, speech transcripts, and video captions, to mitigate the long-tail words challenge. Meanwhile, the internal stream is designed to exploit the multi-modality information in videos (e.g., the appearance of video frames, speech transcripts, and video captions) to ensure the quality of caption results. In addition, the cross attention mechanism is also used in between the two streams for sharing information. In this way, the two streams can help each other for more accurate results. Extensive experiments conducted on four challenging video captioning datasets, i.e., YouCookII, ActivityNet Captions, MSR-VTT, and MSVD, demonstrate that the proposed method performs favorably against the state-of-the-art methods. Specifically, the proposed TextKG method out-performs the best published results by improving 18.7% absolute CIDEr scores on the YouCookII dataset.

        ----

        ## [1804] StepFormer: Self-Supervised Step Discovery and Localization in Instructional Videos

        **Authors**: *Nikita Dvornik, Isma Hadji, Ran Zhang, Konstantinos G. Derpanis, Richard P. Wildes, Allan D. Jepson*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01817](https://doi.org/10.1109/CVPR52729.2023.01817)

        **Abstract**:

        Instructional videos are an important resource to learn procedural tasks from human demonstrations. However, the instruction steps in such videos are typically short and sparse, with most of the video being irrelevant to the procedure. This motivates the need to temporally localize the instruction steps in such videos, i.e. the task called key-step localization. Traditional methods for key-step localization require video-level human annotations and thus do not scale to large datasets. In this work, we tackle the problem with no human supervision and introduce StepFormer, a self-supervised model that discovers and localizes instruction steps in a video. StepFormer is a transformer decoder that attends to the video with learnable queries, and produces a sequence of slots capturing the key-steps in the video. We train our system on a large dataset of instructional videos, using their automatically-generated subtitles as the only source of supervision. In particular, we supervise our system with a sequence of text narrations using an order-aware loss function that filters out irrelevant phrases. We show that our model outperforms all previous unsupervised and weakly-supervised approaches on step detection and localization by a large margin on three challenging benchmarks. Moreover, our model demonstrates an emergent property to solve zero-shot multi-step localization and outperforms all relevant baselines at this task.

        ----

        ## [1805] Dual Alignment Unsupervised Domain Adaptation for Video-Text Retrieval

        **Authors**: *Xiaoshuai Hao, Wanqian Zhang, Dayan Wu, Fei Zhu, Bo Li*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01818](https://doi.org/10.1109/CVPR52729.2023.01818)

        **Abstract**:

        Video-text retrieval is an emerging stream in both computer vision and natural language processing communities, which aims to find relevant videos given text queries. In this paper, we study the notoriously challenging task, i.e., Unsupervised Domain Adaptation Video-text Retrieval (UDAVR), wherein training and testing data come from different distributions. Previous works merely alleviate the domain shift, which however overlook the pairwise misalignment issue in target domain, i.e., there exist no semantic relationships between target videos and texts. To tackle this, we propose a novel method named Dual Alignment Domain Adaptation (DADA). Specifically, we first introduce the cross-modal semantic embedding to generate discriminative source features in a joint embedding space. Besides, we utilize the video and text domain adaptations to smoothly balance the minimization of the domain shifts. To tackle the pairwise misalignment in target domain, we propose the Dual Alignment Consistency (DAC) to fully exploit the semantic information of both modalities in target domain. The proposed DAC adaptively aligns the video-text pairs which are more likely to be relevant in target domain, enabling that positive pairs are increasing progressively and the noisy ones will potentially be aligned in the later stages. To that end, our method can generate more truly aligned target pairs and ensure the discriminability of target features. Compared with the state-of-the-art methods, DADA achieves 20.18% and 18.61% relative improvements on R@1 under the setting of TGIF→MSR-VTT and TGIF→MSVD respectively, demonstrating the superiority of our method.

        ----

        ## [1806] Hierarchical Semantic Correspondence Networks for Video Paragraph Grounding

        **Authors**: *Chaolei Tan, Zihang Lin, Jian-Fang Hu, Wei-Shi Zheng, Jianhuang Lai*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01819](https://doi.org/10.1109/CVPR52729.2023.01819)

        **Abstract**:

        Video Paragraph Grounding (VPG) is an essential yet challenging task in vision-language understanding, which aims to jointly localize multiple events from an untrimmed video with a paragraph query description. One of the critical challenges in addressing this problem is to comprehend the complex semantic relations between visual and textual modalities. Previous methods focus on modeling the contextual information between the video and text from a single-level perspective (i.e., the sentence level), ignoring rich visual-textual correspondence relations at different semantic levels, e.g., the video-word and video-paragraph correspondence. To this end, we propose a novel Hierarchical Semantic Correspondence Network (HSCNet), which explores multi-level visual-textual correspondence by learning hierarchical semantic alignment and utilizes dense supervision by grounding diverse levels of queries. Specifically, we develop a hierarchical encoder that encodes the multi-modal inputs into semantics-aligned representations at different levels. To exploit the hierarchical semantic correspondence learned in the encoder for multi-level supervision, we further design a hierarchical decoder that progressively performs finer grounding for lower-level queries conditioned on higher-level semantics. Extensive experiments demonstrate the effectiveness of HSCNet and our method significantly outstrips the state-of-the-arts on two challenging benchmarks, i.e., ActivityNet-Captions and TACoS.

        ----

        ## [1807] CLIPPING: Distilling CLIP-Based Models with a Student Base for Video-Language Retrieval

        **Authors**: *Renjing Pei, Jianzhuang Liu, Weimian Li, Bin Shao, Songcen Xu, Peng Dai, Juwei Lu, Youliang Yan*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01820](https://doi.org/10.1109/CVPR52729.2023.01820)

        **Abstract**:

        Pre-training a vision-language model and then fine-tuning it on downstream tasks have become a popular paradigm. However, pre-trained vision-language models with the Transformer architecture usually take long inference time. Knowledge distillation has been an efficient technique to transfer the capability of a large model to a small one while maintaining the accuracy, which has achieved remarkable success in natural language processing. However, it faces many problems when applying KD to the multi-modality applications. In this paper, we propose a novel knowledge distillation method, named CLIPPING
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
In this paper, CLIPPING means cutting something to make it smaller through distilling., where the plentiful knowledge of a large teacher model that has been fine-tuned for video-language tasks with the powerful pre-trained CLIP can be effectively transferred to a small student only at the fine-tuning stage. Especially, a new layer-wise alignment with the student as the base is proposed for knowledge distillation of the intermediate layers in CLIPPING, which enables the student's layers to be the bases of the teacher, and thus allows the student to fully absorb the knowledge of the teacher. CLIPPING with MobileViT-v2 as the vision encoder without any vision-language pre-training achieves 88.1%-95.3% of the performance of its teacher on three video-language retrieval benchmarks, with its vision encoder being 19.5x smaller. CLIPPING also significantly outperforms a state-of-the-art small baseline (ALL-in-one-B) on the MSR-VTT dataset, obtaining relatively 7.4% performance gain, with 29% fewer parameters and 86.9% fewer flops. Moreover, CLIPPING is comparable or even superior to many large pre-training models.

        ----

        ## [1808] Learning Emotion Representations from Verbal and Nonverbal Communication

        **Authors**: *Sitao Zhang, Yimu Pan, James Z. Wang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01821](https://doi.org/10.1109/CVPR52729.2023.01821)

        **Abstract**:

        Emotion understanding is an essential but highly challenging component of artificial general intelligence. The absence of extensive annotated datasets has significantly impeded advancements in this field. We present Emotion-CLIp, the first pre-training paradigm to extract visual emotion representations from verbal and nonverbal communication using only uncurated data. Compared to numerical labels or descriptions used in previous methods, communication naturally contains emotion information. Furthermore, acquiring emotion representations from communication is more congruent with the human learning process. We guide EmotionCLIP to attend to nonverbal emotion cues through subject-aware context encoding and verbal emotion cues using sentiment-guided contrastive learning. Extensive experiments validates the effectiveness and transferability of Emotion Clip. Using merely linear-probe evaluation protocol, EmotionCLIP outperforms the state-of-the-art supervised visual emotion recognition methods and rivals many multimodal approaches across various benchmarks. We anticipate that the advent of Emotion Clip will address the prevailing issue of data scarcity in emotion understanding, thereby fostering progress in related domains. The code and pretrained models are available at https://github.com/Xeaver/EmotionCLIP.

        ----

        ## [1809] Context De-Confounded Emotion Recognition

        **Authors**: *Dingkang Yang, Zhaoyu Chen, Yuzheng Wang, Shunli Wang, Mingcheng Li, Siao Liu, Xiao Zhao, Shuai Huang, Zhiyan Dong, Peng Zhai, Lihua Zhang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01822](https://doi.org/10.1109/CVPR52729.2023.01822)

        **Abstract**:

        Context-Aware Emotion Recognition (CAER) is a crucial and challenging task that aims to perceive the emotional states of the target person with contextual information. Recent approaches invariably focus on designing sophisticated architectures or mechanisms to extract seemingly meaningful representations from subjects and contexts. However, a long-overlooked issue is that a context bias in existing datasets leads to a significantly unbalanced distribution of emotional states among different context scenarios. Concretely, the harmful bias is a confounder that misleads existing models to learn spurious correlations based on conventional likelihood estimation, significantly limiting the models' performance. To tackle the issue, this paper provides a causality-based perspective to disentangle the models from the impact of such bias, and formulate the causalities among variables in the CAER task via a tailored causal graph. Then, we propose a Contextual Causal Intervention Module (CCIM) based on the backdoor adjustment to de-confound the confounder and exploit the true causal effect for model training. CCIM is plug-in and model-agnostic, which improves diverse state-of-the-art approaches by considerable margins. Extensive experiments on three benchmark datasets demonstrate the effectiveness of our CCIM and the significance of causal insight.

        ----

        ## [1810] CiCo: Domain-Aware Sign Language Retrieval via Cross-Lingual Contrastive Learning

        **Authors**: *Yiting Cheng, Fangyun Wei, Jianmin Bao, Dong Chen, Wenqiang Zhang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01823](https://doi.org/10.1109/CVPR52729.2023.01823)

        **Abstract**:

        This work focuses on sign language retrieval—a recently proposed task for sign language understanding. Sign language retrieval consists of two sub-tasks: text-to-sign-video (T2V) retrieval and sign-video-to-text (V2T) retrieval. Different from traditional video-text retrieval, sign language videos, not only contain visual signals but also carry abundant semantic meanings by themselves due to the fact that sign languages are also natural languages. Considering this character, we formulate sign language retrieval as a cross-lingual retrieval problem as well as a video-text retrieval task. Concretely, we take into account the linguistic properties of both sign languages and natural languages, and simultaneously identify the fine-grained cross-lingual (i.e., sign-to-word) mappings while contrasting the texts and the sign videos in a joint embedding space. This process is termed as cross-lingual contrastive learning. Another challenge is raised by the data scarcity issue—sign language datasets are orders of magnitude smaller in scale than that of speech recognition. We alleviate this issue by adopting a domain-agnostic sign encoder pre-trained on large-scale sign videos into the target domain via pseudo-labeling. Our framework, termed as domain-aware sign language retrieval via Cross-lingual Contrastive learning or CiCo for short, outperforms the pioneering method by large margins on various datasets, e.g., +22.4 T2V and +28.0 V2T R@1 improvements on How2Sign dataset, and +13.7 T2V and +17.1 V2T R@1 improvements on PHOENIX-2014T dataset. Code and models are available at: https://github.com/FangyunWei/SLRT.

        ----

        ## [1811] Discovering the Real Association: Multimodal Causal Reasoning in Video Question Answering

        **Authors**: *Chuanqi Zang, Hanqing Wang, Mingtao Pei, Wei Liang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01824](https://doi.org/10.1109/CVPR52729.2023.01824)

        **Abstract**:

        Video Question Answering (VideoQA) is challenging as it requires capturing accurate correlations between modalities from redundant information. Recent methods focus on the explicit challenges of the task, e.g. multimodal feature extraction, video-text alignment and fusion. Their frameworks reason the answer relying on statistical evidence causes, which ignores potential bias in the multimodal data. In our work, we investigate relational structure from a causal representation perspective on multimodal data and propose a novel inference framework. For visual data, question-irrelevant objects may establish simple matching associations with the answer. For textual data, the model prefers the local phrase semantics which may deviate from the global semantics in long sentences. Therefore, to enhance the generalization of the model, we discover the real association by explicitly capturing visual features that are causally related to the question semantics and weakening the impact of local language semantics on question answering. The experimental results on two large causal VideoQA datasets verify that our proposed framework 1) improves the accuracy of the existing VideoQA backbone, 2) demonstrates robustness on complex scenes and questions. The code will be released at https://github.com/Chuanqi-Zang/Discovering-the-Real-Association.

        ----

        ## [1812] LEGO-Net: Learning Regular Rearrangements of Objects in Rooms

        **Authors**: *Qiuhong Anna Wei, Sijie Ding, Jeong Joon Park, Rahul Sajnani, Adrien Poulenard, Srinath Sridhar, Leonidas J. Guibas*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01825](https://doi.org/10.1109/CVPR52729.2023.01825)

        **Abstract**:

        Humans universally dislike the task of cleaning up a messy room. If machines were to help us with this task, they must understand human criteria for regular arrangements, such as several types of symmetry, co-linearity or co-circularity, spacing uniformity in linear or circular patterns, and further inter-object relationships that relate to style and functionality. Previous approaches for this task relied on human input to explicitly specify goal state, or synthesized scenes from scratch - but such methods do not address the rearrangement of existing messy scenes without providing a goal state. In this paper, we present LEGO-Net, a data-driven transformer-based iterative method for LEarning reGular rearrangement of Objects in messy rooms. LEGO-Net is partly inspired by diffusion models - it starts with an initial messy state and iteratively “de-noises” the position and orientation of objects to a regular state while reducing distance traveled. Given randomly perturbed object positions and orientations in an existing dataset of professionally-arranged scenes, our method is trained to recover a regular rearrangement. Results demonstrate that our method is able to reliably rearrange room scenes and outperform other methods. We additionally propose a metric for evaluating regularity in room arrangements using number-theoretic machinery.

        ----

        ## [1813] LANA: A Language-Capable Navigator for Instruction Following and Generation

        **Authors**: *Xiaohan Wang, Wenguan Wang, Jiayi Shao, Yi Yang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01826](https://doi.org/10.1109/CVPR52729.2023.01826)

        **Abstract**:

        Recently, visual-language navigation (VLN) - entailing robot agents to follow navigation instructions - has shown great advance. However, existing literature put most emphasis on interpreting instructions into actions, only delivering “dumb” wayfinding agents. In this article, we devise LANA, a language-capable navigation agent which is able to not only execute human-written navigation commands, but also provide route descriptions to humans. This is achieved by simultaneously learning instruction following and generation with only one single model. More specifically, two encoders, respectively for route and language encoding, are built and shared by two decoders, respectively for action prediction and instruction generation, so as to exploit cross-task knowledge and capture task-specific characteristics. Throughout pretraining and fine-tuning, both instruction following and generation are set as optimization objectives. We empirically verify that, compared with recent advanced task-specific solutions, LANA attains better performances on both instruction following and route description, with nearly half complexity. In addition, endowed with language generation capability, Lanacan explain to human its behaviours and assist human's wayfinding. This work is expected to foster future efforts towards building more trustworthy and socially-intelligent navigation robots.

        ----

        ## [1814] Policy Adaptation from Foundation Model Feedback

        **Authors**: *Yuying Ge, Annabella Macaluso, Li Erran Li, Ping Luo, Xiaolong Wang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01827](https://doi.org/10.1109/CVPR52729.2023.01827)

        **Abstract**:

        Recent progress on vision-language foundation models have brought significant advancement to building generalpurpose robots. By using the pre-trained models to encode the scene and instructions as inputs for decision making, the instruction-conditioned policy can generalize across different objects and tasks. While this is encouraging, the policy still fails in most cases given an unseen task or environment. In this work, we propose Policy Adaptation from Foundation model Feedback (PAFF). When deploying the trained policy to a new task or a new environment, we first let the policy play with randomly generated instructions to record the demonstrations. While the execution could be wrong, we can use the pre-trained foundation models to provide feedback to relabel the demonstrations. This automatically provides new pairs of demonstration-instruction data for policy fine-tuning. We evaluate our method on a broad range of experiments with the focus on generalization on unseen objects, unseen tasks, unseen environments, and sim-to-real transfer. We show PAFF improves baselines by a large margin in all cases.

        ----

        ## [1815] Token Turing Machines

        **Authors**: *Michael S. Ryoo, Keerthana Gopalakrishnan, Kumara Kahatapitiya, Ted Xiao, Kanishka Rao, Austin Stone, Yao Lu, Julian Ibarz, Anurag Arnab*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01828](https://doi.org/10.1109/CVPR52729.2023.01828)

        **Abstract**:

        We propose Token Turing Machines (TTM), a sequential, autoregressive Transformer model with memory for real-world sequential visual understanding. Our model is inspired by the seminal Neural Turing Machine, and has an external memory consisting of a set of tokens which summarise the previous history (i.e., frames). This memory is efficiently addressed, read and written using a Transformer as the processing unit/controller at each step. The model's memory module ensures that a new observation will only be processed with the contents of the memory (and not the entire history), meaning that it can efficiently process long sequences with a bounded computational cost at each step. We show that TTM outperforms other alternatives, such as other Transformer models designed for long sequences and recurrent neural networks, on two real-world sequential visual understanding tasks: online temporal activity detection from videos and vision-based robot action policy learning. Code is publicly available at: https://github.com/google-research/scenic/tree/main/scenic/projects/token.turing.

        ----

        ## [1816] Unicode Analogies: An Anti-Objectivist Visual Reasoning Challenge

        **Authors**: *Steven Spratley, Krista A. Ehinger, Tim Miller*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01829](https://doi.org/10.1109/CVPR52729.2023.01829)

        **Abstract**:

        Analogical reasoning enables agents to extract relevant information from scenes, and efficiently navigate them in familiar ways. While progressive-matrix problems (PMPs) are becoming popular for the development and evaluation of analogical reasoning in computer vision, we argue that the dominant methodology in this area struggles to expose the lack of meaningful generalisation in solvers, and rein-forces an objectivist stance on perception - that objects can only be seen one way - which we believe to be counter-productive. In this paper, we introduce the Unicode Analogies challenge, consisting of polysemic, character-based PMPs to benchmark fluid conceptualisation ability in vision systems. Writing systems have evolved characters at multiple levels of abstraction, from iconic through to symbolic representations, producing both visually interrelated yet exceptionally diverse images when compared to those exhibited by existing PMP datasets. Our framework has been designed to challenge models by presenting tasks much harder to complete without robust feature extraction, while remaining largely solvable by human participants. We therefore argue that Unicode Analogies elegantly captures and tests for a facet of human visual reasoning that is severely lacking in current-generation AI.

        ----

        ## [1817] Exploring the Effect of Primitives for Compositional Generalization in Vision-and-Language

        **Authors**: *Chuanhao Li, Zhen Li, Chenchen Jing, Yunde Jia, Yuwei Wu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01830](https://doi.org/10.1109/CVPR52729.2023.01830)

        **Abstract**:

        Compositionality is one of the fundamental properties of human cognition (Fodor & Pylyshyn, 1988). Compositional generalization is critical to simulate the compositional capability of humans, and has received much attention in the vision-and-language (V&L) community. It is essential to understand the effect of the primitives, including words, image regions, and video frames, to improve the compositional generalization capability. In this paper, we explore the effect of primitives for compositional generalization in V&L. Specifically, we present a self-supervised learning based framework that equips existing V&L methods with two characteristics: semantic equivariance and semantic invariance. With the two characteristics, the methods understand primitives by perceiving the effect of primitive changes on sample semantics and ground-truth. Experimental results on two tasks: temporal video grounding and visual question answering, demonstrate the effectiveness of our framework.

        ----

        ## [1818] VQACL: A Novel Visual Question Answering Continual Learning Setting

        **Authors**: *Xi Zhang, Feifei Zhang, Changsheng Xu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01831](https://doi.org/10.1109/CVPR52729.2023.01831)

        **Abstract**:

        Research on continual learning has recently led to a variety of work in unimodal community, however little attention has been paid to multimodal tasks like visual question answering (VQA). In this paper, we establish a novel VQA Continual Learning setting named VQACL, which contains two key components: a dual-level task sequence where visual and linguistic data are nested, and a novel composition testing containing new skill-concept combinations. The former devotes to simulating the ever-changing multimodal datastream in real world and the latter aims at measuring models' generalizability for cognitive reasoning. Based on our VQACL, we perform in-depth evaluations of five well-established continual learning methods, and observe that they suffer from catastrophic forgetting and have weak generalizability. To address above issues, we propose a novel representation learning method, which leverages a sample-specific and a sample-invariant feature to learn representations that are both discriminative and generalizable for VQA. Furthermore, by respectively extracting such representation for visual and textual input, our method can explicitly disentangle the skill and concept. Extensive experimental results illustrate that our method significantly outperforms existing models, demonstrating the effectiveness and compositionality of the proposed approach. The code is available at https://github.com/zhangxi1997/VQACL.

        ----

        ## [1819] MaPLe: Multi-modal Prompt Learning

        **Authors**: *Muhammad Uzair Khattak, Hanoona Abdul Rasheed, Muhammad Maaz, Salman H. Khan, Fahad Shahbaz Khan*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01832](https://doi.org/10.1109/CVPR52729.2023.01832)

        **Abstract**:

        Pre-trained vision-language (V-L) models such as CLIP have shown excellent generalization ability to downstream tasks. However, they are sensitive to the choice of input text prompts and require careful selection of prompt templates to perform well. Inspired by the Natural Language Processing (NLP) literature, recent CLIP adaptation approaches learn prompts as the textual inputs to fine-tune CLIP for downstream tasks. We note that using prompting to adapt representations in a single branch of CLIP (language or vision) is sub-optimal since it does not allow the flexibility to dynamically adjust both representation spaces on a downstream task. In this work, we propose Multi-modal Prompt Learning (MaPLe) for both vision and language branches to improve alignment between the vision and language representations. Our design promotes strong coupling between the vision-language prompts to ensure mutual synergy and discourages learning independent uni-modal solutions. Further, we learn separate prompts across different early stages to progressively model the stage-wise feature relationships to allow rich context learning. We evaluate the effectiveness of our approach on three representative tasks of generalization to novel classes, new target datasets and unseen domain shifts. Compared with the state-of-the-art method Co-CoOp, MaPLe exhibits favorable performance and achieves an absolute gain of 3.45% on novel classes and 2.72% on overall harmonic-mean, averaged over 11 diverse image recognition datasets. Our code and pre-trained models are available at https://github.com/muzairkhattak/multimodal-prompt-learning.

        ----

        ## [1820] Meta-Personalizing Vision-Language Models to Find Named Instances in Video

        **Authors**: *Chun-Hsiao Yeh, Bryan C. Russell, Josef Sivic, Fabian Caba Heilbron, Simon Jenni*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01833](https://doi.org/10.1109/CVPR52729.2023.01833)

        **Abstract**:

        Large-scale vision-language models (VLM) have shown impressive results for language-guided search applications. While these models allow category-level queries, they currently struggle with personalized searches for moments in a video where a specific object instance such as “My dog Biscuit” appears. We present the following three contributions to address this problem. First, we describe a method to meta-personalize a pre-trained VLM, i.e., learning how to learn to personalize a VLM at test time to search in video. Our method extends the VLM's token vocabulary by learning novel word embeddings specific to each instance. To capture only instance-specific features, we represent each instance embedding as a combination of shared and learned global category features. Second, we propose to learn such personalization without explicit human supervision. Our approach automatically identifies moments of named visual instances in video using transcripts and vision-language similarity in the VLM's embedding space. Finally, we introduce This-Is-My, a personal video instance retrieval benchmark. We evaluate our approach on This-Is-My and Deep-Fashion2 and show that we obtain a 15% relative improvement over the state of the art on the latter dataset.

        ----

        ## [1821] Understanding and Improving Visual Prompting: A Label-Mapping Perspective

        **Authors**: *Aochuan Chen, Yuguang Yao, Pin-Yu Chen, Yihua Zhang, Sijia Liu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01834](https://doi.org/10.1109/CVPR52729.2023.01834)

        **Abstract**:

        We revisit and advance visual prompting (VP), an input prompting technique for vision tasks. VP can reprogram a fixed, pre-trained source model to accomplish downstream tasks in the target domain by simply incorporating universal prompts (in terms of input perturbation patterns) into downstream data points. Yet, it remains elusive why VP stays effective even given a ruleless label mapping (LM) between the source classes and the target classes. Inspired by the above, we ask: How is LM interrelated with VP? And how to exploit such a relationship to improve its accuracy on target tasks? We peer into the influence of LM on VP and provide an affirmative answer that a better ‘quality’ of LM (assessed by mapping precision and explanation) can consistently improve the effectiveness of VP. This is in contrast to the prior art where the factor of LM was missing. To optimize LM, we propose a new VP framework, termed ILM-VP (iterative label mapping-based visual prompting), which automatically re-maps the source labels to the target labels and progressively improves the target task accuracy of VP. Further, when using a contrastive language-image pretrained (CLIP) model for VP, we propose to integrate an LM process to assist the text prompt selection of CLIP and to improve the target task accuracy. Extensive experiments demonstrate that our proposal significantly outperforms state-of-the-art VP methods. As highlighted below, we show that when reprogramming an ImageNet-pretrained ResNet-18 to 13 target tasks, ILM-VP outperforms baselines by a substantial margin, e.g., 7.9% and 6.7% accuracy improvements in transfer learning to the target Flowers102 and CIFAR100 datasets. Besides, our proposal on CLIP-based VP provides 13.7% and 7.1% accuracy improvements on Flowers102 and DTD respectively. Code is available at https://github.com/OPTML-Group/ILM-VP.

        ----

        ## [1822] RefTeacher: A Strong Baseline for Semi-Supervised Referring Expression Comprehension

        **Authors**: *Jiamu Sun, Gen Luo, Yiyi Zhou, Xiaoshuai Sun, Guannan Jiang, Zhiyu Wang, Rongrong Ji*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01835](https://doi.org/10.1109/CVPR52729.2023.01835)

        **Abstract**:

        Referring expression comprehension (REC) often requires a large number of instance-level annotations for fully supervised learning, which are laborious and expensive. In this paper, we present the first attempt of semi-supervised learning for REC and propose a strong baseline method called RefTeacher. Inspired by the recent progress in computer vision, RefTeacher adopts a teacher-student learning paradigm, where the teacher REC network predicts pseudolabels for optimizing the student one. This paradigm allows REC models to exploit massive unlabeled data based on a small fraction of labeled. In particular, we also identify two key challenges in semi-supervised REC, namely, sparse supervision signals and worse pseudo-label noise. To address these issues, we equip RefTeacher with two novel designs called Attention-based Imitation Learning (AIL) and Adaptive Pseudo-label Weighting (APW). AIL can help the student network imitate the recognition behaviors of the teacher, thereby obtaining sufficient supervision signals. APW can help the model adaptively adjust the contributions of pseudo-labels with varying qualities, thus avoiding confirmation bias. To validate RefTeacher, we conduct extensive experiments on three REC benchmark datasets. Experimental results show that RefTeacher obtains obvious gains over the fully supervised methods. More importantly, using only 10% labeled data, our approach allows the model to achieve near 100% fully supervised performance, e.g., only −2.78% on RefCoco. Project: https://refteacher.github.io/.

        ----

        ## [1823] Leveraging per Image-Token Consistency for Vision-Language Pre-training

        **Authors**: *Yunhao Gou, Tom Ko, Hansi Yang, James T. Kwok, Yu Zhang, Mingxuan Wang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01836](https://doi.org/10.1109/CVPR52729.2023.01836)

        **Abstract**:

        Most existing vision-language pre-training (VLP) approaches adopt cross-modal masked language modeling (CMLM) to learn vision-language associations. However, we find that CMLM is insufficient for this purpose according to our observations: (1) Modality bias: a considerable amount of masked tokens in CMLM can be recovered with only the language information, ignoring the visual inputs. (2) Underutilization of the unmasked tokens: CMLM primarily focuses on the masked token but it cannot simultaneously leverage other tokens to learn vision-language associations. To handle those limitations, we propose EPIC (lEveraging Per Image-Token Consistency for vision-language pre-training). In EPIC, for each image-sentence pair, we mask tokens that are salient to the image (i.e., Saliency-based Masking Strategy) and replace them with alternatives sampled from a language model (i.e., Inconsistent Token Generation Procedure), and then the model is required to determine for each token in the sentence whether it is consistent with the image (i.e., Image-Token Consistency Task). The proposed EPIC method is easily combined with pre-training methods. Extensive experiments show that the combination of the EPIC method and state-of-the-art pre-training approaches, including ViLT, ALBEF, METER, and X-VLM, leads to significant improvements on downstream tasks. Our coude is released at https://github.com/gyhdog99/epic

        ----

        ## [1824] Improving Visual Grounding by Encouraging Consistent Gradient-Based Explanations

        **Authors**: *Ziyan Yang, Kushal Kafle, Franck Dernoncourt, Vicente Ordonez*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01837](https://doi.org/10.1109/CVPR52729.2023.01837)

        **Abstract**:

        We propose a margin-based loss for tuning joint vision-language models so that their gradient-based explanations are consistent with region-level annotations provided by humans for relatively smaller grounding datasets. We refer to this objective as Attention Mask Consistency (AMC) and demonstrate that it produces superior visual grounding results than previous methods that rely on using vision-language models to score the outputs of object detectors. Particularly, a model trained with AMC on top of standard vision-language modeling objectives obtains a state-of-the-art accuracy of 86.49% in the Flickr30k visual grounding benchmark, an absolute improvement of 5.38% when compared to the best previous model trained under the same level of supervision. Our approach also performs exceedingly well on established benchmarks for referring expression comprehension where it obtains 80.34% accuracy in the easy test of RefCOCO+, and 64.55% in the difficult split. AMC is effective, easy to implement, and is general as it can be adopted by any vision-language model, and can use any type of region annotations.

        ----

        ## [1825] Image as a Foreign Language: BEIT Pretraining for Vision and Vision-Language Tasks

        **Authors**: *Wenhui Wang, Hangbo Bao, Li Dong, Johan Bjorck, Zhiliang Peng, Qiang Liu, Kriti Aggarwal, Owais Khan Mohammed, Saksham Singhal, Subhojit Som, Furu Wei*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01838](https://doi.org/10.1109/CVPR52729.2023.01838)

        **Abstract**:

        A big convergence of language, vision, and multimodal pretraining is emerging. In this work, we introduce a general-purpose multimodal foundation model BEIT-3, which achieves excellent transfer performance on both vision and vision-language tasks. Specifically, we advance the big convergence from three aspects: backbone architecture, pretraining task, and model scaling up. We use Multiway Transformers for general-purpose modeling, where the modular architecture enables both deep fusion and modality-specific encoding. Based on the shared backbone, we perform masked “language” modeling on images (Imglish), texts (English), and image-text pairs (“parallel sentences”) in a unified manner. Experimental results show that BEIT-3 obtains remarkable performance on object detection (COCO), semantic segmentation (ADE20K), image classification (ImageNet), visual reasoning (NLVR2), visual question answering (VQAv2), image captioning (COCO), and cross-modal retrieval (Flickr30K, COCO).

        ----

        ## [1826] Language in a Bottle: Language Model Guided Concept Bottlenecks for Interpretable Image Classification

        **Authors**: *Yue Yang, Artemis Panagopoulou, Shenghao Zhou, Daniel Jin, Chris Callison-Burch, Mark Yatskar*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01839](https://doi.org/10.1109/CVPR52729.2023.01839)

        **Abstract**:

        Concept Bottleneck Models (CBM) are inherently interpretable models that factor model decisions into humanreadable concepts. They allow people to easily understand why a model is failing, a critical feature for high-stakes applications. CBMs require manually specified concepts and often under-perform their black box counterparts, preventing their broad adoption. We address these shortcomings and are first to show how to construct high-performance CBMs without manual specification of similar accuracy to black box models. Our approach, Language Guided Bottlenecks (LaBo), leverages a language model, GPT-3, to define a large space of possible bottlenecks. Given a problem domain, LaBo uses GPT-3 to produce factual sentences about categories to form candidate concepts. LaBo efficiently searches possible bottlenecks through a novel submodular utility that promotes the selection of discriminative and diverse information. Ultimately, GPT-3's sentential concepts can be aligned to images using CLIP, to form a bottleneck layer. Experiments demonstrate that LaBo is a highly effective prior for concepts important to visual recognition. In the evaluation with 11 diverse datasets, LaBo bottlenecks excel at few-shot classification: they are 11.7% more accurate than black box linear probes at 1 shot and comparable with more data. Overall, LaBo demonstrates that inherently interpretable models can be widely applied at similar, or better, performance than black box approaches.
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
Code and data are available at https://github.com/YueYANG1996/LaBo

        ----

        ## [1827] Shepherding Slots to Objects: Towards Stable and Robust Object-Centric Learning

        **Authors**: *Jinwoo Kim, Janghyuk Choi, Ho-Jin Choi, Seon Joo Kim*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01840](https://doi.org/10.1109/CVPR52729.2023.01840)

        **Abstract**:

        Object-centric learning (OCL) aspires general and compositional understanding of scenes by representing a scene as a collection of object-centric representations. OCL has also been extended to multi-view image and video datasets to apply various data-driven inductive biases by utilizing geometric or temporal information in the multi-image data. Single-view images carry less information about how to disentangle a given scene than videos or multi-view images do. Hence, owing to the difficulty of applying inductive biases, OCL for single-view images remains challenging, resulting in inconsistent learning of object-centric representation. To this end, we introduce a novel OCL framework for single-view images, SLot Attention via SHepherding (SLASH), which consists of two simple- yet-effective modules on top of Slot Attention. The new modules, Attention Refining Kernel (ARK) and Intermediate Point Predictor and Encoder (IPPE), respectively, prevent slots from being distracted by the background noise and indicate locations for slots to focus on to facilitate learning of object-centric representation. We also propose a weak semi-supervision approach for OCL, whilst our proposed framework can be used without any assistant annotation during the inference. Experiments show that our proposed method enables consistent learning of object-centric representation and achieves strong performance across four datasets. Code is available at https://github.com/object-understanding/SLASH.

        ----

        ## [1828] Learning Visual Representations via Language-Guided Sampling

        **Authors**: *Mohamed El Banani, Karan Desai, Justin Johnson*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01841](https://doi.org/10.1109/CVPR52729.2023.01841)

        **Abstract**:

        Although an object may appear in numerous contexts, we often describe it in a limited number of ways. Language allows us to abstract away visual variation to represent and communicate concepts. Building on this intuition, we propose an alternative approach to visual representation learning: using language similarity to sample semantically similar image pairs for contrastive learning. Our approach diverges from image-based contrastive learning by sampling view pairs using language similarity instead of handcrafted augmentations or learned clusters. Our approach also differs from image-text contrastive learning by relying on pre-trained language models to guide the learning rather than directly minimizing a cross-modal loss. Through a series of experiments, we show that language-guided learning yields better features than image-based and image-text representation learning approaches.

        ----

        ## [1829] L-CoIns: Language-based Colorization With Instance Awareness

        **Authors**: *Zheng Chang, Shuchen Weng, Peixuan Zhang, Yu Li, Si Li, Boxin Shi*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01842](https://doi.org/10.1109/CVPR52729.2023.01842)

        **Abstract**:

        Language-based colorization produces plausible colors consistent with the language description provided by the user. Recent studies introduce additional annotation to prevent color-object coupling and mismatch issues, but they still have difficulty in distinguishing instances corresponding to the same object words. In this paper, we propose a transformer-based framework to automatically aggregate similar image patches and achieve instance awareness without any additional knowledge. By applying our presented luminance augmentation and counter-color loss to break down the statistical correlation between luminance and color words, our model is driven to synthesize colors with better descriptive consistency. We further collect a dataset to provide distinctive visual characteristics and detailed language descriptions for multiple instances in the same image. Extensive experiments demonstrate our advantages of synthesizing visually pleasing and description-consistent results of instance-aware colorization.

        ----

        ## [1830] EDA: Explicit Text-Decoupling and Dense Alignment for 3D Visual Grounding

        **Authors**: *Yanmin Wu, Xinhua Cheng, Renrui Zhang, Zesen Cheng, Jian Zhang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01843](https://doi.org/10.1109/CVPR52729.2023.01843)

        **Abstract**:

        3D visual grounding aims to find the object within point clouds mentioned by free-form natural language descriptions with rich semantic cues. However, existing methods either extract the sentence-level features coupling all words or focus more on object names, which would lose the word-level information or neglect other attributes. To alleviate these issues, we present EDA that Explicitly Decouples the textual attributes in a sentence and conducts Dense Alignment between such fine-grained language and point cloud objects. Specifically, we first propose a text decoupling module to produce textual features for every semantic component. Then, we design two losses to supervise the dense matching between two modalities: position alignment loss and semantic alignment loss. On top of that, we further introduce a new visual grounding task, locating objects without object names, which can thoroughly evaluate the model's dense alignment capacity. Through experiments, we achieve state-of-the-art performance on two widely-adopted 3D visual grounding datasets, ScanRefer and SR3D/NR3D, and obtain absolute leadership on our newly-proposed task. The source code is available at https://github.com/yanmin-wu/EDA.

        ----

        ## [1831] MSINet: Twins Contrastive Search of Multi-Scale Interaction for Object ReID

        **Authors**: *Jianyang Gu, Kai Wang, Hao Luo, Chen Chen, Wei Jiang, Yuqiang Fang, Shanghang Zhang, Yang You, Jian Zhao*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01844](https://doi.org/10.1109/CVPR52729.2023.01844)

        **Abstract**:

        Neural Architecture Search (NAS) has been increasingly appealing to the society of object Re-Identification (ReID), for that task-specific architectures significantly improve the retrieval performance. Previous works explore new optimizing targets and search spaces for NAS ReID, yet they neglect the difference of training schemes between image classification and ReID. In this work, we propose a novel Twins Contrastive Mechanism (TCM) to provide more appropriate supervision for ReID architecture search. TCM reduces the category overlaps between the training and validation data, and assists NAS in simulating real-world ReID training schemes. We then design a Multi-Scale Interaction (MSI) search space to search for rational interaction operations between multi-scale features. In addition, we introduce a Spatial Alignment Module (SAM) to further enhance the attention consistency confronted with images from different sources. Under the proposed NAS scheme, a specific architecture is automatically searched, named as MSINet. Extensive experiments demonstrate that our method surpasses state-of-the-art ReID methods on both indomain and cross-domain scenarios. Source code available in https://github.com/vimar-gu/MSINet.

        ----

        ## [1832] Unifying Vision, Text, and Layout for Universal Document Processing

        **Authors**: *Zineng Tang, Ziyi Yang, Guoxin Wang, Yuwei Fang, Yang Liu, Chenguang Zhu, Michael Zeng, Cha Zhang, Mohit Bansal*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01845](https://doi.org/10.1109/CVPR52729.2023.01845)

        **Abstract**:

        We propose Universal Document Processing (UDOP), a foundation Document AI model which unifies text, image, and layout modalities together with varied task formats, including document understanding and generation. UDOP leverages the spatial correlation between textual content and document image to model image, text, and layout modalities with one uniform representation. With a novel Vision-Text-Layout Transformer, UDOP unifies pretraining and multi-domain downstream tasks into a prompt-based sequence generation scheme. UDOP is pretrained on both large-scale unlabeled document corpora using innovative self-supervised objectives and diverse labeled data. UDOP also learns to generate document images from text and layout modalities via masked image reconstruction. To the best of our knowledge, this is the first time in the field of document AI that one model simultaneously achieves high-quality neural document editing and content customization. Our method sets the state-of-the-art on 8 Document AI tasks, e.g., document understanding and QA, across diverse data domains like finance reports, academic papers, and web-sites. UDOP ranks first on the leaderboard of the Document Understanding Benchmark.
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
 Code and models: https://github.com/microsoft/i-Code/tree/main/i-Code-Doc

        ----

        ## [1833] RA-CLIP: Retrieval Augmented Contrastive Language-Image Pre-Training

        **Authors**: *Chen-Wei Xie, Siyang Sun, Xiong Xiong, Yun Zheng, Deli Zhao, Jingren Zhou*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01846](https://doi.org/10.1109/CVPR52729.2023.01846)

        **Abstract**:

        Contrastive Language-Image Pre-training (CLIP) is attracting increasing attention for its impressive zero-shot recognition performance on different down-stream tasks. However, training CLIP is data-hungry and requires lots of image-text pairs to memorize various semantic concepts. In this paper, we propose a novel and efficient framework: Retrieval Augmented Contrastive Language-Image Pre-training (RA-CLIP) to augment embeddings by online retrieval. Specifically, we sample part of image-text data as a hold-out reference set. Given an input image, relevant image-text pairs are retrieved from the reference set to enrich the representation of input image. This process can be considered as an open-book exam: with the reference set as a cheat sheet, the proposed method doesn't need to memorize all visual concepts in the training data. It explores how to recognize visual concepts by exploiting correspondence between images and texts in the cheat sheet. The proposed RA-CLIP implements this idea and comprehensive experiments are conducted to show how RA-CLIP works. Performances on 10 image classification datasets and 2 object detection datasets show that RA-CLIP outperforms vanilla CLIP baseline by a large margin on zero-shot image classification task (+12.7%), linear probe image classification task (+6.9%) and zero-shot ROI classification task (+2.8%).

        ----

        ## [1834] Fine-grained Image-text Matching by Cross-modal Hard Aligning Network

        **Authors**: *Zhengxin Pan, Fangyu Wu, Bailing Zhang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01847](https://doi.org/10.1109/CVPR52729.2023.01847)

        **Abstract**:

        Current state-of-the-art image-text matching methods implicitly align the visual-semantic fragments, like regions in images and words in sentences, and adopt cross-attention mechanism to discover fine-grained cross-modal semantic correspondence. However, the cross-attention mechanism may bring redundant or irrelevant region-word alignments, degenerating retrieval accuracy and limiting efficiency. Although many researchers have made progress in mining meaningful alignments and thus improving accuracy, the problem of poor efficiency remains unresolved. In this work, we propose to learn fine-grained image-text matching from the perspective of information coding. Specifically, we suggest a coding framework to explain the fragments aligning process, which provides a novel view to reexamine the cross-attention mechanism and analyze the problem of redundant alignments. Based on this framework, a Cross-modal Hard Aligning Network (CHAN) is designed, which comprehensively exploits the most relevant region-word pairs and eliminates all other alignments. Extensive experiments conducted on two public datasets, MS-COCO and Flickr30K, verify that the relevance of the most associated word-region pairs is discriminative enough as an indicator of the image-text similarity, with superior accuracy and efficiency over the state-of-the-art approaches on the bidirectional image and text retrieval tasks. Our code will be available at https://github.com/ppanzx/CHAN.

        ----

        ## [1835] Text-Guided Unsupervised Latent Transformation for Multi-Attribute Image Manipulation

        **Authors**: *Xiwen Wei, Zhen Xu, Cheng Liu, Si Wu, Zhiwen Yu, Hau-San Wong*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01848](https://doi.org/10.1109/CVPR52729.2023.01848)

        **Abstract**:

        Great progress has been made in StyleGAN-based image editing. To associate with preset attributes, most existing approaches focus on supervised learning for semantically meaningful latent space traversal directions, and each manipulation step is typically determined for an individual attribute. To address this limitation, we propose a Text-guided Unsupervised StyleGAN Latent Transformation (TUSLT) model, which adaptively infers a single transformation step in the latent space of StyleGAN to simultaneously manipulate multiple attributes on a given input image. Specifically, we adopt a two-stage architecture for a latent mapping network to break down the transformation process into two manageable steps. Our network first learns a diverse set of semantic directions tailored to an input image, and later nonlinearly fuses the ones associated with the target attributes to infer a residual vector. The resulting tightly interlinked two-stage architecture delivers the flexibility to handle diverse attribute combinations. By leveraging the cross-modal text-image representation of CLIP, we can perform pseudo annotations based on the semantic similarity between preset attribute text descriptions and training images, and further jointly train an auxiliary attribute classifier with the latent mapping network to provide semantic guidance. We perform extensive experiments to demonstrate that the adopted strategies contribute to the superior performance of TUSLT.

        ----

        ## [1836] Improving Image Recognition by Retrieving from Web-Scale Image-Text Data

        **Authors**: *Ahmet Iscen, Alireza Fathi, Cordelia Schmid*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01849](https://doi.org/10.1109/CVPR52729.2023.01849)

        **Abstract**:

        Retrieval augmented models are becoming increasingly popular for computer vision tasks after their recent success in NLP problems. The goal is to enhance the recognition capabilities of the model by retrieving similar examples for the visual input from an external memory set. In this work, we introduce an attention-based memory module, which learns the importance of each retrieved example from the memory. Compared to existing approaches, our method removes the influence of the irrelevant retrieved examples, and retains those that are beneficial to the input query. We also thoroughly study various ways of constructing the memory dataset. Our experiments show the benefit of using a massive-scale memory dataset of 1B image-text pairs, and demonstrate the performance of different memory representations. We evaluate our method in three different classification tasks, namely long-tailed recognition, learning with noisy labels, and fine-grained classification, and show that it achieves state-of-the-art accuracies in ImageNet-LT, Places-LT and Webvision datasets.

        ----

        ## [1837] Pic2Word: Mapping Pictures to Words for Zero-shot Composed Image Retrieval

        **Authors**: *Kuniaki Saito, Kihyuk Sohn, Xiang Zhang, Chun-Liang Li, Chen-Yu Lee, Kate Saenko, Tomas Pfister*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01850](https://doi.org/10.1109/CVPR52729.2023.01850)

        **Abstract**:

        In Composed Image Retrieval (CIR), a user combines a query image with text to describe their intended target. Existing methods rely on supervised learning of CIR models using labeled triplets consisting of the query image, text specification, and the target image. Labeling such triplets is expensive and hinders broad applicability of CIR. In this work, we propose to study an important task, Zero-Shot Composed Image Retrieval (ZS-CIR), whose goal is to build a CIR model without requiring labeled triplets for training. To this end, we propose a novel method, called Pic2Word, that requires only weakly labeled image-caption pairs and unlabeled image datasets to train. Unlike existing supervised CIR models, our model trained on weakly labeled or unlabeled datasets shows strong generalization across diverse ZS-CIR tasks, e.g., attribute editing, object composition, and domain conversion. Our approach outperforms several supervised CIR methods on the common CIR benchmark, CIRR and Fashion-IQ. Code will be made publicly available at https://github.com/google-research/composed_image_retrieval

        ----

        ## [1838] DATE: Domain Adaptive Product Seeker for E-Commerce

        **Authors**: *Haoyuan Li, Hao Jiang, Tao Jin, Mengyan Li, Yan Chen, Zhijie Lin, Yang Zhao, Zhou Zhao*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01851](https://doi.org/10.1109/CVPR52729.2023.01851)

        **Abstract**:

        Product Retrieval (PR) and Grounding (PG), aiming to seek image and object-level products respectively according to a textual query, have attracted great interest recently for better shopping experience. Owing to the lack of relevant datasets, we collect two large-scale benchmark datasets from Taobao Mall and Live domains with about 474k and 101k image-query pairs for PR, and manually annotate the object bounding boxes in each image for PG. As annotating boxes is expensive and time-consuming, we attempt to transfer knowledge from annotated domain to unannotated for PG to achieve unsupervised Domain Adaptation (PG-DA). We propose a Domain Adaptive Product Seeker (DATE) framework, regarding PR and PG as Product Seeking problem at different levels, to assist the query date the product. Concretely, we first design a semantics-aggregated feature extractor for each modality to obtain concentrated and comprehensive features for following efficient retrieval and fine-grained grounding tasks. Then, we present two cooperative seekers to simultaneously search the image for PR and localize the product for PG. Besides, we devise a domain aligner for PG-DA to alleviate unimodal marginal and multi-modal conditional distribution shift between source and target domains, and design a pseudo box generator to dynamically select reliable instances and generate bounding boxes for further knowledge transfer. Extensive experiments show that our DATE achieves satisfactory performance in fully-supervised PR, PG and unsupervised PG-DA. Our desensitized datasets will be publicly available here
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
https://github.com/Taobao-live/Product-Seeking.

        ----

        ## [1839] Multimodality Helps Unimodality: Cross-Modal Few-Shot Learning with Multimodal Models

        **Authors**: *Zhiqiu Lin, Samuel Yu, Zhiyi Kuang, Deepak Pathak, Deva Ramanan*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01852](https://doi.org/10.1109/CVPR52729.2023.01852)

        **Abstract**:

        The ability to quickly learn a new task with minimal instruction - known as few-shot learning - is a central aspect of intelligent agents. Classical few-shot benchmarks make use of few-shot samples from a single modality, but such samples may not be sufficient to characterize an entire concept class. In contrast, humans use cross-modal information to learn new concepts efficiently. In this work, we demonstrate that one can indeed build a better visual dog classifier by reading about dogs and listening to them bark. To do so, we exploit the fact that recent multimodal foundation models such as CLIP are inherently cross-modal, mapping different modalities to the same representation space. Specifically, we propose a simple cross-modal adaptation approach that learns from few-shot examples spanning different modalities. By repurposing class names as additional one-shot training samples, we achieve SOTA results with an embarrassingly simple linear classifier for vision-language adaptation. Furthermore, we show that our approach can benefit existing methods such as prefix tuning, adapters, and classifier ensembling. Finally, to explore other modalities beyond vision and language, we construct the first (to our knowledge) audiovisual few-shot benchmark and use cross-modal training to improve the performance of both image and audio classification. Project site at link.

        ----

        ## [1840] Finetune like you pretrain: Improved finetuning of zero-shot vision models

        **Authors**: *Sachin Goyal, Ananya Kumar, Sankalp Garg, Zico Kolter, Aditi Raghunathan*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01853](https://doi.org/10.1109/CVPR52729.2023.01853)

        **Abstract**:

        Finetuning image-text models such as CLIP achieves state-of-the-art accuracies on a variety of benchmarks. However, recent works (Kumar et al., 2022; Wortsman et al., 2021) have shown that even subtle differences in the finetuning process can lead to surprisingly large differences in the final performance, both for in-distribution (ID) and out-of-distribution (OOD) data. In this work, we show that a natural and simple approach of mimicking contrastive pretraining consistently outperforms alternative finetuning approaches. Specifically, we cast downstream class labels as text prompts and continue optimizing the contrastive loss between image embeddings and class-descriptive prompt embeddings (contrastive finetuning). Our method consistently outperforms baselines across 7 distribution shift, 6 transfer learning, and 3 few-shot learning benchmarks. On WILDS-iWILDCam, our proposed approach FLYP outperforms the top of the leaderboard by 2.3% ID and 2.7% OOD, giving the highest reported accuracy. Averaged across 7 OOD datasets (2 WILDS and 5 ImageNet associated shifts), FLYP gives gains of 4.2% OOD over standard finetuning and outperforms current state-of-the-art (LP-FT) by more than 1 % both ID and OOD. Similarly, on 3 few-shot learning benchmarks, FLYP gives gains up to 4.6% over standard finetuning and 4.4% over the state-of-the-art. Thus we establish our proposed method of contrastive finetuning as a simple and intuitive state-of-the-art for supervised finetuning of image-text models like CLIP. Code is available at https://github.com/locuslab/FLYP.

        ----

        ## [1841] DeepSolo: Let Transformer Decoder with Explicit Points Solo for Text Spotting

        **Authors**: *Maoyuan Ye, Jing Zhang, Shanshan Zhao, Juhua Liu, Tongliang Liu, Bo Du, Dacheng Tao*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01854](https://doi.org/10.1109/CVPR52729.2023.01854)

        **Abstract**:

        End-to-end text spotting aims to integrate scene text detection and recognition into a unified framework. Dealing with the relationship between the two sub-tasks plays a pivotal role in designing effective spotters. Although Transformer-based methods eliminate the heuristic postprocessing, they still suffer from the synergy issue between the sub-tasks and low training efficiency. In this paper, we present DeepSolo, a simple DETR-like baseline that lets a single Decoder with Explicit Points Solo for text detection and recognition simultaneously. Technically, for each text instance, we represent the character sequence as ordered points and model them with learnable explicit point queries. After passing a single decoder, the point queries have encoded requisite text semantics and locations, thus can be further decoded to the center line, boundary, script, and confidence of text via very simple prediction heads in parallel. Besides, we also introduce a text-matching criterion to deliver more accurate supervisory signals, thus enabling more efficient training. Quantitative experiments on public benchmarks demonstrate that DeepSolo outperforms previous state-of-the-art methods and achieves better training efficiency. In addition, DeepSolo is also compatible with line annotations, which require much less annotation cost than polygons. The code is available at https://github.com/ViTAE-Transformer/DeepSolo.

        ----

        ## [1842] EVA: Exploring the Limits of Masked Visual Representation Learning at Scale

        **Authors**: *Yuxin Fang, Wen Wang, Binhui Xie, Quan Sun, Ledell Wu, Xinggang Wang, Tiejun Huang, Xinlong Wang, Yue Cao*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01855](https://doi.org/10.1109/CVPR52729.2023.01855)

        **Abstract**:

        We launch EVA, a vision-centric foundation model to Explore the limits of Visual representation at scAle using only publicly accessible data. EVA is a vanilla ViT pre-trained to reconstruct the masked out image-text aligned vision features conditioned on visible image patches. Via this pretext task, we can efficiently scale up EVA to one billion parameters, and sets new records on a broad range of representative vision downstream tasks, such as image recognition, video action recognition, object detection, instance segmentation and semantic segmentation without heavy supervised training. Moreover, we observe quantitative changes in scaling EVA result in qualitative changes in transfer learning performance that are not present in other models. For instance, EVA takes a great leap in the challenging large vocabulary instance segmentation task: our model achieves almost the same state-of-the-art performance on LVIS dataset with over a thousand categories and COCO dataset with only eighty categories. Beyond a pure vision encoder, EVA can also serve as a vision-centric, multi-modal pivot to connect images and text. We find initializing the vision tower of a giant CLIP from EVA can greatly stabilize the training and outperform the training from scratch counterpart with much fewer samples and less compute, providing a new direction for scaling up and accelerating the costly training of multi-modal foundation models.

        ----

        ## [1843] $R^{2}$ Former: Unified Retrieval and Reranking Transformer for Place Recognition

        **Authors**: *Sijie Zhu, Linjie Yang, Chen Chen, Mubarak Shah, Xiaohui Shen, Heng Wang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01856](https://doi.org/10.1109/CVPR52729.2023.01856)

        **Abstract**:

        Visual Place Recognition (VPR) estimates the location of query images by matching them with images in a reference database. Conventional methods generally adopt aggregated CNN features for global retrieval and RANSAC-based geometric verification for reranking. However, RANSAC only employs geometric information but ignores other possible information that could be useful for reranking, e.g. local feature correlations, and attention values. In this paper, we propose a unified place recognition framework that handles both retrieval and reranking with a novel transformer model, named 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$R^{2}$</tex>
 Former. The proposed reranking module takes feature correlation, attention value, and 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$xy$</tex>
 coordinates into account, and learns to determine whether the image pair is from the same location. The whole pipeline is end-to-end trainable and the reranking module alone can also be adopted on other CNN or transformer backbones as a generic component. Remarkably, 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$R^{2}$</tex>
 Former significantly outperforms state-of-the-art methods on major VPR datasets with much less inference time and memory consumption. It also achieves the state-of-the-art on the hold-out MSLS challenge set and could serve as a simple yet strong solution for real-world large-scale applications. Experiments also show vision transformer tokens are comparable and sometimes better than CNN local features on local matching. The code is released at https://github.com/Jeff-Zilence/R2Former.

        ----

        ## [1844] Open-Set Fine-Grained Retrieval via Prompting Vision-Language Evaluator

        **Authors**: *Shijie Wang, Jianlong Chang, Haojie Li, Zhihui Wang, Wanli Ouyang, Qi Tian*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01857](https://doi.org/10.1109/CVPR52729.2023.01857)

        **Abstract**:

        Open-set fine-grained retrieval is an emerging challenge that requires an extra capability to retrieve unknown subcategories during evaluation. However, current works focus on close-set visual concepts, where all the subcategories are pre-defined, and make it hard to capture discriminative knowledge from unknown subcategories, consequently failing to handle unknown subcategories in open-world scenarios. In this work, we propose a novel Prompting vision-Language Evaluator (PLEor) framework based on the recently introduced contrastive language-image pretraining (CLIP) model, for open-set fine-grained retrieval. PLEor could leverage pre-trained CLIP model to infer the discrepancies encompassing both pre-defined and unknown subcategories, called category-specific discrepancies, and transfer them to the backbone network trained in the close-set scenarios. To make pre-trained CLIP model sensitive to category-specific discrepancies, we design a dual prompt scheme to learn a vision prompt specifying the categoryspecific discrepancies, and turn random vectors with category names in a text prompt into category-specific discrepancy descriptions. Moreover, a vision-language evaluator is proposed to semantically align the vision and text prompts based on CLIP model, and reinforce each other. In addition, we propose an open-set knowledge transfer to transfer the category-specific discrepancies into the backbone network using knowledge distillation mechanism. Quantitative and qualitative experiments show that our PLEor achieves promising performance on open-set fine-grained datasets.

        ----

        ## [1845] Open-Category Human-Object Interaction Pre-training via Language Modeling Framework

        **Authors**: *Sipeng Zheng, Boshen Xu, Qin Jin*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01858](https://doi.org/10.1109/CVPR52729.2023.01858)

        **Abstract**:

        Human-object interaction (HOI) has long been plagued by the conflict between limited supervised data and a vast number of possible interaction combinations in real life. Current methods trained from closed-set data predict HOIs as fixed-dimension log its, which restricts their scalability to open-set categories. To address this issue, we introduce Open Cat, a language modeling framework that reformulates HOI prediction as sequence generation. By converting HOI triplets into a token sequence through a serialization scheme, our model is able to exploit the open-set vocabulary of the language modeling framework to predict novel interaction classes with a high degree of freedom. In addition, inspired by the great success of vision-language pre-training, we collect a large amount of weakly-supervised data related to HOI from image-caption pairs, and devise several auxiliary proxy tasks, including soft relational matching and human-object relation prediction, to pre-train our model. Extensive experiments show that our OpenCat significantly boosts HOI performance, particularly on a broad range of rare and unseen categories.

        ----

        ## [1846] Neural Congealing: Aligning Images to a Joint Semantic Atlas

        **Authors**: *Dolev Ofri-Amar, Michal Geyer, Yoni Kasten, Tali Dekel*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01859](https://doi.org/10.1109/CVPR52729.2023.01859)

        **Abstract**:

        We present Neural Congealing – a zero-shot self-supervised framework for detecting and jointly aligning semantically-common content across a given set of images. Our approach harnesses the power of pre-trained DINO-ViT features to learn: (i) a joint semantic atlas – a 2D grid that captures the mode of DINO-ViT features in the input set, and (ii) dense mappings from the unified atlas to each of the input images. We derive a new robust self-supervised framework that optimizes the atlas representation and mappings per image set, requiring only a few real-world images as input without any additional input information (e.g., segmentation masks). Notably, we design our losses and training paradigm to account only for the shared content under severe variations in appearance, pose, background clutter or other distracting objects. We demonstrate results on a plethora of challenging image sets including sets of mixed domains (e.g., aligning images depicting sculpture and artwork of cats), sets depicting related yet different object categories (e.g., dogs and tigers), or domains for which large-scale training data is scarce (e.g., coffee mugs). We thoroughly evaluate our method and show that our test-time optimization approach performs favorably compared to a state-of-the-art method that requires extensive training on large-scale datasets. Project webpage: https://neural-congealing.github.io/

        ----

        ## [1847] Open Vocabulary Semantic Segmentation with Patch Aligned Contrastive Learning

        **Authors**: *Jishnu Mukhoti, Tsung-Yu Lin, Omid Poursaeed, Rui Wang, Ashish Shah, Philip H. S. Torr, Ser-Nam Lim*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01860](https://doi.org/10.1109/CVPR52729.2023.01860)

        **Abstract**:

        We introduce Patch Aligned Contrastive Learning (PACL), a modified compatibility function for CLIP's contrastive loss, intending to train an alignment between the patch tokens of the vision encoder and the CLS token of the text encoder. With such an alignment, a model can identify regions of an image corresponding to a given text input, and therefore transfer seamlessly to the task of open vocabulary semantic segmentation without requiring any segmentation annotations during training. Using pre-trained CLIP encoders with PACL, we are able to set the state-of-the-art on the task of open vocabulary zero-shot segmentation on 4 different segmentation benchmarks: Pascal VOC, Pascal Context, COCO Stuff and ADE20K. Furthermore, we show that PACL is also applicable to image-level predictions and when used with a CLIP backbone, provides a general improvement in zero-shot classification accuracy compared to CLIP, across a suite of 12 image classification datasets.

        ----

        ## [1848] Semantic Human Parsing via Scalable Semantic Transfer Over Multiple Label Domains

        **Authors**: *Jie Yang, Chaoqun Wang, Zhen Li, Junle Wang, Ruimao Zhang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01861](https://doi.org/10.1109/CVPR52729.2023.01861)

        **Abstract**:

        This paper presents Scalable Semantic Transfer (SST), a novel training paradigm, to explore how to leverage the mutual benefits of the data from different label domains (i.e. various levels of label granularity) to train a powerful human parsing network. In practice, two common application scenarios are addressed, termed universal parsing and dedicated parsing, where the former aims to learn homogeneous human representations from multiple label domains and switch predictions by only using different segmentation heads, and the latter aims to learn a specific domain prediction while distilling the semantic knowledge from other domains. The proposed SST has the following appealing benefits: (1) it can capably serve as an effective training scheme to embed semantic associations of human body parts from multiple label domains into the human representation learning process; (2) it is an extensible semantic transfer framework without predetermining the overall relations of multiple label domains, which allows continuously adding human parsing datasets to promote the training. (3) the relevant modules are only used for auxiliary training and can be removed during inference, eliminating the extra reasoning cost. Experimental results demonstrate SST can effectively achieve promising universal human parsing performance as well as impressive improvements compared to its counterparts on three human parsing benchmarks (i.e., PASCAL-Person-Part, ATR, and CIHP). Code is available at https://github.com/yangjie-cv/SST.

        ----

        ## [1849] Explicit Visual Prompting for Low-Level Structure Segmentations

        **Authors**: *Weihuang Liu, Xi Shen, Chi-Man Pun, Xiaodong Cun*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01862](https://doi.org/10.1109/CVPR52729.2023.01862)

        **Abstract**:

        We consider the generic problem of detecting low-level structures in images, which includes segmenting the manipulated parts, identifying out-of-focus pixels, separating shadow regions, and detecting concealed objects. Whereas each such topic has been typically addressed with a domain-specific solution, we show that a unified approach performs well across all of them. We take inspiration from the widely-used pre-training and then prompt tuning protocols in NLP and propose a new visual prompting model, named Explicit Visual Prompting (EVP). Different from the previous visual prompting which is typically a dataset-level implicit embedding, our key insight is to enforce the tunable parameters focusing on the explicit visual content from each individual image, i.e., the features from frozen patch embeddings and the input's high-frequency components. The proposed EVP significantly outperforms other parameter-efficient tuning protocols under the same amount of tunable parameters (5.7% extra trainable parameters of each task). EVP also achieves state-of-the-art performances on diverse low-level structure segmentation tasks compared to task-specific solutions. Our code is available at: https://github.com/NiFangBaAGe/Explicit-Visual-Prompt.

        ----

        ## [1850] FreeSeg: Unified, Universal and Open-Vocabulary Image Segmentation

        **Authors**: *Jie Qin, Jie Wu, Pengxiang Yan, Ming Li, Yuxi Ren, Xuefeng Xiao, Yitong Wang, Rui Wang, Shilei Wen, Xin Pan, Xingang Wang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01863](https://doi.org/10.1109/CVPR52729.2023.01863)

        **Abstract**:

        Recently, open-vocabulary learning has emerged to accomplish segmentation for arbitrary categories of text-based descriptions, which popularizes the segmentation system to more general-purpose application scenarios. However, existing methods devote to designing specialized architectures or parameters for specific segmentation tasks. These customized design paradigms lead to fragmentation between various segmentation tasks, thus hindering the uniformity of segmentation models. Hence in this paper, we propose FreeSeg, a generic framework to accomplish Unified, Universal and Open-Vocabulary Image Segmentation. FreeSeg optimizes an all-in-one network via one-shot training and employs the same architecture and parameters to handle diverse segmentation tasks seamlessly in the inference procedure. Additionally, adaptive prompt learning facilitates the unified model to capture task-aware and category-sensitive concepts, improving model robustness in multi-task and varied scenarios. Extensive experimental results demonstrate that FreeSeg establishes new state-of-the-art results in performance and generalization on three segmentation tasks, which outperforms the best task-specific architectures by a large margin: 5.5% mIoU on semantic segmentation, 17.6% mAP on instance segmentation, 20.1% PQ on panoptic segmentation for the unseen class on COCO. Project page: https://FreeSeg.github.io.

        ----

        ## [1851] Zero-shot Referring Image Segmentation with Global-Local Context Features

        **Authors**: *Seonghoon Yu, Paul Hongsuck Seo, Jeany Son*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01864](https://doi.org/10.1109/CVPR52729.2023.01864)

        **Abstract**:

        Referring image segmentation (RIS) aims to find a segmentation mask given a referring expression grounded to a region of the input image. Collecting labelled datasets for this task, however, is notoriously costly and labor-intensive. To overcome this issue, we propose a simple yet effective zero-shot referring image segmentation method by leveraging the pre-trained cross-modal knowledge from CLIP. In order to obtain segmentation masks grounded to the input text, we propose a mask-guided visual encoder that captures global and local contextual information of an input image. By utilizing instance masks obtained from off-the-shelf mask proposal techniques, our method is able to segment fine-detailed instance-level groundings. We also introduce a global-local text encoder where the global feature captures complex sentence-level semantics of the entire input expression while the local feature focuses on the target noun phrase extracted by a dependency parser. In our experiments, the proposed method outperforms several zero-shot baselines of the task and even the weakly supervised referring expression segmentation method with substantial margins. Our code is available at https://github.com/Seonghoon-Yu/Zero-shot-RIS.

        ----

        ## [1852] DejaVu: Conditional Regenerative Learning to Enhance Dense Prediction

        **Authors**: *Shubhankar Borse, Debasmit Das, Hyojin Park, Hong Cai, Risheek Garrepalli, Fatih Porikli*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01865](https://doi.org/10.1109/CVPR52729.2023.01865)

        **Abstract**:

        We present DejaVu, a novel framework which leverages conditional image regeneration as additional supervision during training to improve deep networks for dense prediction tasks such as segmentation, depth estimation, and surface normal prediction. First, we apply redaction to the input image, which removes certain structural information by sparse sampling or selective frequency removal. Next, we use a conditional regenerator, which takes the redacted image and the dense predictions as inputs, and reconstructs the original image by filling in the missing structural information. In the redacted image, structural attributes like boundaries are broken while semantic context is largely preserved. In order to make the regeneration feasible, the conditional generator will then require the structure information from the other input source, i.e., the dense predictions. As such, by including this conditional regeneration objective during training, DejaVu encourages the base network to learn to embed accurate scene structure in its dense prediction. This leads to more accurate predictions with clearer boundaries and better spatial consistency. When it is feasible to leverage additional computation, DejaVu can be extended to incorporate an attention-based regeneration module within the dense prediction network, which further improves accuracy. Through extensive experiments on multiple dense prediction benchmarks such as Cityscapes, COCO, ADE20K, NYUD-v2, and KITTI, we demonstrate the efficacy of employing DejaVu during training, as it out-performs SOTA methods at no added computation cost.

        ----

        ## [1853] Meta Compositional Referring Expression Segmentation

        **Authors**: *Li Xu, Mark He Huang, Xindi Shang, Zehuan Yuan, Ying Sun, Jun Liu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01866](https://doi.org/10.1109/CVPR52729.2023.01866)

        **Abstract**:

        Referring expression segmentation aims to segment an object described by a language expression from an image. Despite the recent progress on this task, existing models tackling this task may not be able to fully capture semantics and visual representations of individual concepts, which limits their generalization capability, especially when handling novel compositions of learned concepts. In this work, through the lens of meta learning, we propose a Meta Compositional Referring Expression Segmentation (MCRES) framework to enhance model compositional generalization performance. Specifically, to handle various levels of novel compositions, our framework first uses training data to construct a virtual training set and multiple virtual testing sets, where data samples in each virtual testing set contain a level of novel compositions w.r.t. the virtual training set. Then, following a novel meta optimization scheme to optimize the model to obtain good testing performance on the virtual testing sets after training on the virtual training set, our framework can effectively drive the model to better capture semantics and visual representations of individual concepts, and thus obtain robust generalization performance even when handling novel compositions. Extensive experiments on three benchmark datasets demonstrate the effectiveness of our framework.

        ----

        ## [1854] Interactive Segmentation as Gaussian Process Classification

        **Authors**: *Minghao Zhou, Hong Wang, Qian Zhao, Yuexiang Li, Yawen Huang, Deyu Meng, Yefeng Zheng*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01867](https://doi.org/10.1109/CVPR52729.2023.01867)

        **Abstract**:

        Click-based interactive segmentation (IS) aims to extract the target objects under user interaction. For this task, most of the current deep learning (DL)-based methods mainly follow the general pipelines of semantic segmentation. Albeit achieving promising performance, they do not fully and explicitly utilize and propagate the click information, inevitably leading to unsatisfactory segmentation results, even at clicked points. Against this issue, in this paper, we propose to formulate the IS task as a Gaussian process (GP)-based pixel-wise binary classification model on each image. To solve this model, we utilize amortized variational inference to approximate the intractable GP posterior in a data-driven manner and then decouple the approximated GP posterior into double space forms for efficient sampling with linear complexity. Then, we correspondingly construct a GP classification framework, named GPCIS, which is integrated with the deep kernel learning mechanism for more flexibility. The main specificities of the proposed GPCIS lie in: 1) Under the explicit guidance of the derived GP posterior, the information contained in clicks can be finely propagated to the entire image and then boost the segmentation; 2) The accuracy of predictions at clicks has good theoretical support. These merits of GPCIS as well as its good generality and high efficiency are substantiated by comprehensive experiments on several benchmarks, as compared with representative methods both quantitatively and qualitatively. Codes will be released at https://github.com/zmhhlnz/GPCIS_CVPR2023.

        ----

        ## [1855] Semantic-Promoted Debiasing and Background Disambiguation for Zero-Shot Instance Segmentation

        **Authors**: *Shuting He, Henghui Ding, Wei Jiang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01868](https://doi.org/10.1109/CVPR52729.2023.01868)

        **Abstract**:

        Zero-shot instance segmentation aims to detect and precisely segment objects of unseen categories without any training samples. Since the model is trained on seen categories, there is a strong bias that the model tends to classify all the objects into seen categories. Besides, there is a natural confusion between background and novel objects that have never shown up in training. These two challenges make novel objects hard to be raised in the final instance segmentation results. It is desired to rescue novel objects from background and dominated seen categories. To this end, we propose D
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">2</sup>
Zero with Semantic-Promoted Debiasing and Background Disambiguation to enhance the performance of Zero-shot instance segmentation. Semantic-promoted debiasing utilizes inter-class semantic relationships to involve unseen categories in visual feature training and learns an input-conditional classifier to conduct dynamical classification based on the input image. Background disambiguation produces image-adaptive background representation to avoid mistaking novel objects for background. Extensive experiments show that we significantly outperform previous state-of-the-art methods by a large margin, e.g., 16.86% improvement on COCO.

        ----

        ## [1856] Principles of Forgetting in Domain-Incremental Semantic Segmentation in Adverse Weather Conditions

        **Authors**: *Tobias Kalb, Jürgen Beyerer*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01869](https://doi.org/10.1109/CVPR52729.2023.01869)

        **Abstract**:

        Deep neural networks for scene perception in automated vehicles achieve excellent results for the domains they were trained on. However, in real-world conditions, the domain of operation and its underlying data distribution are subject to change. Adverse weather conditions, in particular, can significantly decrease model performance when such data are not available during training. Additionally, when a model is incrementally adapted to a new domain, it suffers from catastrophic forgetting, causing a significant drop in performance on previously observed domains. Despite recent progress in reducing catastrophic forgetting, its causes and effects remain obscure. Therefore, we study how the representations of semantic segmentation models are affected during domain-incremental learning in adverse weather conditions. Our experiments and representational analyses indicate that catastrophic forgetting is primarily caused by changes to low-level features in domain-incremental learning and that learning more general features on the source domain using pre-training and image augmentations leads to efficient feature reuse in subsequent tasks, which drastically reduces catastrophic forgetting. These findings highlight the importance of methods that facilitate generalized features for effective continual learning algorithms.

        ----

        ## [1857] AttentionShift: Iteratively Estimated Part-Based Attention Map for Pointly Supervised Instance Segmentation

        **Authors**: *Mingxiang Liao, Zonghao Guo, Yuze Wang, Peng Yuan, Bailan Feng, Fang Wan*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01870](https://doi.org/10.1109/CVPR52729.2023.01870)

        **Abstract**:

        Pointly supervised instance segmentation (PSIS) learns to segment objects using a single point within the object extent as supervision. Challenged by the non-negligible semantic variance between object parts, however, the single supervision point causes semantic bias and false segmentation. In this study, we propose an AttentionShift method, to solve the semantic bias issue by iteratively decomposing the instance attention map to parts and estimating fine-grained semantics of each part. AttentionShift consists of two modules plugged on the vision transformer backbone: (i) token querying for pointly supervised attention map generation, and (ii) key-point shift, which re-estimates part-based attention maps by key-point filtering in the feature space. These two steps are iteratively performed so that the part-based attention maps are optimized spatially as well as in the feature space to cover full object extent. Experiments on PASCAL VOC and MS COCO 2017 datasets show that AttentionShift respectively improves the state-of-the-art of by 7.7% and 4.8% under mAP@0.5, setting a solid PSIS baseline using vision transformer.

        ----

        ## [1858] PIDNet: A Real-time Semantic Segmentation Network Inspired by PID Controllers

        **Authors**: *Jiacong Xu, Zixiang Xiong, Shankar P. Bhattacharyya*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01871](https://doi.org/10.1109/CVPR52729.2023.01871)

        **Abstract**:

        Two-branch network architecture has shown its efficiency and effectiveness in real-time semantic segmentation tasks. However, direct fusion of high-resolution details and low-frequency context has the drawback of detailed features being easily overwhelmed by surrounding contextual information. This overshoot phenomenon limits the improvement of the segmentation accuracy of existing two-branch models. In this paper, we make a connection between Convolutional Neural Networks (CNN) and Proportional-Integral-Derivative (PID) controllers and reveal that a two-branch network is equivalent to a Proportional-Integral (PI) controller, which inherently suffers from similar overshoot issues. To alleviate this problem, we propose a novel three-branch network architecture: PIDNet, which contains three branches to parse detailed, context and boundary information, respectively, and employs boundary attention to guide the fusion of detailed and context branches. Our family of PIDNets achieve the best trade-off between inference speed and accuracy and their accuracy surpasses all the existing models with similar inference speed on the Cityscapes and CamVid datasets. Specifically, PIDNet-S achieves 78.6% mIOU with inference speed of 93.2 FPS on Cityscapes and 80.1% mIOU with speed of 153.7 FPS on CamVid.

        ----

        ## [1859] Leveraging Hidden Positives for Unsupervised Semantic Segmentation

        **Authors**: *Hyun Seok Seong, WonJun Moon, Su Been Lee, Jae-Pil Heo*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01872](https://doi.org/10.1109/CVPR52729.2023.01872)

        **Abstract**:

        Dramatic demand for manpower to label pixel-level annotations triggered the advent of unsupervised semantic segmentation. Although the recent work employing the vision transformer (ViT) backbone shows exceptional performance, there is still a lack of consideration for task-specific training guidance and local semantic consistency. To tackle these issues, we leverage contrastive learning by excavating hidden positives to learn rich semantic relationships and ensure semantic consistency in local regions. Specifically, we first discover two types of global hidden positives, task-agnostic and task-specific ones for each anchor based on the feature similarities defined by a fixed pre-trained back-bone and a segmentation head-in-training, respectively. A gradual increase in the contribution of the latter induces the model to capture task-specific semantic features. In addition, we introduce a gradient propagation strategy to learn semantic consistency between adjacent patches, under the inherent premise that nearby patches are highly likely to possess the same semantics. Specifically, we add the loss propagating to local hidden positives, semantically similar nearby patches, in proportion to the predefined similarity scores. With these training schemes, our proposed method achieves new state-of-the-art (SOTA) results in COCO-stuff, Cityscapes, and Potsdam-3 datasets. Our code is available at: https://github.com/hynnsk/HP.

        ----

        ## [1860] Understanding Imbalanced Semantic Segmentation Through Neural Collapse

        **Authors**: *Zhisheng Zhong, Jiequan Cui, Yibo Yang, Xiaoyang Wu, Xiaojuan Qi, Xiangyu Zhang, Jiaya Jia*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01873](https://doi.org/10.1109/CVPR52729.2023.01873)

        **Abstract**:

        A recent study has shown a phenomenon called neural collapse in that the within-class means of features and the classifier weight vectors converge to the vertices of a simplex equiangular tight frame at the terminal phase of training for classification. In this paper, we explore the cor-responding structures of the last-layer feature centers and classifiers in semantic segmentation. Based on our empirical and theoretical analysis, we point out that semantic segmentation naturally brings contextual correlation and imbalanced distribution among classes, which breaks the equiangular and maximally separated structure of neural collapse for both feature centers and classifiers. However, such a symmetric structure is beneficial to discrimination for the minor classes. To preserve these advantages, we in-troduce a regularizer on feature centers to encourage the network to learn features closer to the appealing structure in imbalanced semantic segmentation. Experimental results show that our method can bring significant improvements on both 2D and 3D semantic segmentation bench-marks. Moreover, our method ranks 1
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">st</sup>
 and sets a new record (+6.8% mIoU) on the ScanNet200 test leaderboard.

        ----

        ## [1861] Balancing Logit Variation for Long-Tailed Semantic Segmentation

        **Authors**: *Yuchao Wang, Jingjing Fei, Haochen Wang, Wei Li, Tianpeng Bao, Liwei Wu, Rui Zhao, Yujun Shen*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01874](https://doi.org/10.1109/CVPR52729.2023.01874)

        **Abstract**:

        Semantic segmentation usually suffers from a long-tail data distribution. Due to the imbalanced number of samples across categories, the features of those tail classes may get squeezed into a narrow area in the feature space. Towards a balanced feature distribution, we introduce category-wise variation into the network predictions in the training phase such that an instance is no longer projected to a feature point, but a small region instead. Such a perturbation is highly dependent on the category scale, which appears as assigning smaller variation to head classes and larger variation to tail classes. In this way, we manage to close the gap between the feature areas of different categories, resulting in a more balanced representation. It is note-worthy that the introduced variation is discarded at the inference stage to facilitate a confident prediction. Although with an embarrassingly simple implementation, our method manifests itself in strong generalizability to various datasets and task settings. Extensive experiments suggest that our plug-in design lends itself well to a range of state-of-the-art approaches and boosts the performance on top of them.
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
Code: https://github.com/grantword8/BLV.

        ----

        ## [1862] Boundary-enhanced Co-training for Weakly Supervised Semantic Segmentation

        **Authors**: *Shenghai Rong, Bohai Tu, Zilei Wang, Junjie Li*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01875](https://doi.org/10.1109/CVPR52729.2023.01875)

        **Abstract**:

        The existing weakly supervised semantic segmentation (WSSS) methods pay much attention to generating accurate and complete class activation maps (CAMs) as pseudo-labels, while ignoring the importance of training the segmentation networks. In this work, we observe that there is an inconsistency between the quality of the pseudo-labels in CAMs and the performance of the final segmentation model, and the mislabeled pixels mainly lie on the boundary areas. Inspired by these findings, we argue that the focus of WSSS should be shifted to robust learning given the noisy pseudo-labels, and further propose a boundary-enhanced co-training (BECO) method for training the segmentation model. To be specific, we first propose to use a co-training paradigm with two interactive networks to improve the learning of uncertain pixels. Then we propose a boundary-enhanced strategy to boost the prediction of difficult boundary areas, which utilizes reliable predictions to construct artificial boundaries. Benefiting from the design of co-training and boundary enhancement, our method can achieve promising segmentation performance for different CAMs. Extensive experiments on PASCAL VOC 2012 and MS COCO 2014 validate the superiority of our BECO over other state-of-the-art methods.
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
The code and models are available at https://github.com/ShenghaiRong/BECO.

        ----

        ## [1863] Conflict-Based Cross-View Consistency for Semi-Supervised Semantic Segmentation

        **Authors**: *Zicheng Wang, Zhen Zhao, Xiaoxia Xing, Dong Xu, Xiangyu Kong, Luping Zhou*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01876](https://doi.org/10.1109/CVPR52729.2023.01876)

        **Abstract**:

        Semi-supervised semantic segmentation (SSS) has recently gained increasing research interest as it can reduce the requirement for large-scale fully-annotated training data. The current methods often suffer from the confirmation bias from the pseudo-labelling process, which can be alleviated by the co-training framework. The current co-training-based SSS methods rely on hand-crafted perturbations to prevent the different sub-nets from collapsing into each other, but these artificial perturbations cannot lead to the optimal solution. In this work, we propose a new conflict-based cross-view consistency (CCVC) method based on a two-branch co-training framework which aims at enforcing the two sub-nets to learn informative features from irrelevant views. In particular, we first propose a new cross-view consistency (CVC) strategy that encourages the two sub-nets to learn distinct features from the same input by introducing a feature discrepancy loss, while these distinct features are expected to generate consistent prediction scores of the input. The CVC strategy helps to prevent the two sub-nets from stepping into the collapse. In addition, we further propose a conflict-based pseudo-labelling (CPL) method to guarantee the model will learn more useful information from conflicting predictions, which will lead to a stable training process. We validate our new CCVC approach on the SSS benchmark datasets where our method achieves new state-of-the-art performance. Our code is available at https://github.com/xiaoyao3302/CCVC.

        ----

        ## [1864] Learning Multi-Modal Class-Specific Tokens for Weakly Supervised Dense Object Localization

        **Authors**: *Lian Xu, Wanli Ouyang, Mohammed Bennamoun, Farid Boussaïd, Dan Xu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01877](https://doi.org/10.1109/CVPR52729.2023.01877)

        **Abstract**:

        Weakly supervised dense object localization (WSDOL) relies generally on Class Activation Mapping (CAM), which exploits the correlation between the class weights of the image classifier and the pixel-level features. Due to the limited ability to address intra-class variations, the image classifier cannot properly associate the pixel features, leading to inaccurate dense localization maps. In this paper, we propose to explicitly construct multi-modal class representations by leveraging the Contrastive Language-Image Pre-training (CLIP), to guide dense localization. More specifically, we propose a unified transformer framework to learn two-modalities of class-specific tokens, i.e., class-specific visual and textual tokens. The former captures semantics from the target visual data while the latter exploits the class-related language priors from CLIP, providing complementary information to better perceive the intra-class diversities. In addition, we propose to enrich the multi-modal class-specific tokens with sample-specific contexts comprising visual context and image-language context. This enables more adaptive class representation learning, which further facilitates dense localization. Extensive experiments show the superiority of the proposed method for WSDOL on two multi-label datasets, i.e., PASCAL VOC and MS COCO, and one single-label dataset, i.e., OpenImages. Our dense localization maps also lead to the state-of-the-art weakly supervised semantic segmentation (WSSS) results on PASCAL VOC and MS COCO.
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
https://github.com/xulianuwa/MMCST

        ----

        ## [1865] WinCLIP: Zero-/Few-Shot Anomaly Classification and Segmentation

        **Authors**: *Jongheon Jeong, Yang Zou, Taewan Kim, Dongqing Zhang, Avinash Ravichandran, Onkar Dabeer*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01878](https://doi.org/10.1109/CVPR52729.2023.01878)

        **Abstract**:

        Visual anomaly classification and segmentation are vital for automating industrial quality inspection. The focus of prior research in the field has been on training custom models for each quality inspection task, which requires task-specific images and annotation. In this paper we move away from this regime, addressing zero-shot and few-normal-shot anomaly classification and segmentation. Recently CLIP, a vision-language model, has shown revolutionary generality with competitive zero-/few-shot performance in comparison to full-supervision. But CLIP falls short on anomaly classification and segmentation tasks. Hence, we propose window-based CLIP (WinCLIP) with (1) a compositional ensemble on state words and prompt templates and (2) efficient extraction and aggregation of window/patch/image-level features aligned with text. We also propose its few-normal-shot extension Win-CLIP+, which uses complementary information from normal images. In MVTec-AD (and VisA), without further tuning, WinCLIP achieves 91.8%/85.1% (78.1%/79.6%) AU-ROC in zero-shot anomaly classification and segmentation while WinCLIP + does 93.1%/95.2% (83.8%/96.4%) in 1-normal-shot, surpassing state-of-the-art by large margins.

        ----

        ## [1866] DualRel: Semi-Supervised Mitochondria Segmentation from A Prototype Perspective

        **Authors**: *Huayu Mai, Rui Sun, Tianzhu Zhang, Zhiwei Xiong, Feng Wu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01879](https://doi.org/10.1109/CVPR52729.2023.01879)

        **Abstract**:

        Automatic mitochondria segmentation enjoys great popularity with the development of deep learning. However, existing methods rely heavily on the labor-intensive manual gathering by experienced domain experts. And naively applying semi-supervised segmentation methods in the natural image field to mitigate the labeling cost is undesirable. In this work, we analyze the gap between mitochondrial images and natural images and rethink how to achieve effective semi-supervised mitochondria segmentation, from the perspective of reliable prototype-level supervision. We propose a novel end-to-end dual-reliable (DualRel) network, including a reliable pixel aggregation module and a reliable prototype selection module. The proposed DualRel enjoys several merits. First, to learn the prototypes well without any explicit supervision, we carefully design the referential correlation to rectify the direct pairwise correlation. Second, the reliable prototype selection module is responsible for further evaluating the reliability of prototypes in constructing prototype-level consistency regularization. Extensive experimental results on three challenging benchmarks demonstrate that our method performs favorably against state-of-the-art semi-supervised segmentation methods. Importantly, with extremely few samples used for training, DualRel is also on par with current state-of-the-art fully supervised methods.

        ----

        ## [1867] Distilling Self-Supervised Vision Transformers for Weakly-Supervised Few-Shot Classification & Segmentation

        **Authors**: *Dahyun Kang, Piotr Koniusz, Minsu Cho, Naila Murray*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01880](https://doi.org/10.1109/CVPR52729.2023.01880)

        **Abstract**:

        We address the task of weakly-supervised few-shot image classification and segmentation, by leveraging a Vision Transformer (ViT) pretrained with self-supervision. Our proposed method takes token representations from the self-supervised ViT and leverages their correlations, via selfattention, to produce classification and segmentation predictions through separate task heads. Our model is able to effectively learn to perform classification and segmentation in the absence of pixel-level labels during training, using only image-level labels. To do this it uses attention maps, created from tokens generated by the self-supervised ViT backbone, as pixel-level pseudo-labels. We also explore a practical setup with “mixed” supervision, where a small number of training images contains ground-truth pixel-level labels and the remaining images have only image-level labels. For this mixed setup, we propose to improve the pseudo-labels using a pseudo-label enhancer that was trained using the available ground-truth pixel-level labels. Experiments on Pascal-5
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">i</sup>
 and COCO-20
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">i</sup>
 demonstrate significant performance gains in a variety of supervision settings, and in particular when little-to-no pixel-level labels are available.

        ----

        ## [1868] Co-Salient Object Detection with Uncertainty-Aware Group Exchange-Masking

        **Authors**: *Yang Wu, Huihui Song, Bo Liu, Kaihua Zhang, Dong Liu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01881](https://doi.org/10.1109/CVPR52729.2023.01881)

        **Abstract**:

        The traditional definition of co-salient object detection (CoSOD) task is to segment the common salient objects in a group of relevant images. Existing CoSOD models by-default adopt the group consensus assumption. This brings about model robustness defect under the condition of irrelevant images in the testing image group, which hinders the use of CoSOD models in real-world applications. To address this issue, this paper presents a group exchange-masking (GEM) strategy for robust CoSOD model learning. With two group of image containing different types of salient object as input, the GEM first selects a set of images from each group by the proposed learning based strategy, then these images are exchanged. The proposed feature extraction module considers both the uncertainty caused by the irrelevant images and group consensus in the remaining relevant images. We design a latent variable generator branch which is made of conditional variational autoencoder to generate uncertainly-based global stochastic features. A CoSOD transformer branch is devised to capture the correlation-based local features that contain the group consistency information. At last, the output of two branches are concatenated and fed into a transformer-based decoder, producing robust co-saliency prediction. Extensive evaluations on co-saliency detection with and without irrelevant images demonstrate the superiority of our method over a variety of state-of-the-art methods.

        ----

        ## [1869] Supervised Masked Knowledge Distillation for Few-Shot Transformers

        **Authors**: *Han Lin, Guangxing Han, Jiawei Ma, Shiyuan Huang, Xudong Lin, Shih-Fu Chang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01882](https://doi.org/10.1109/CVPR52729.2023.01882)

        **Abstract**:

        Vision Transformers (ViTs) emerge to achieve impressive performance on many data-abundant computer vision tasks by capturing long-range dependencies among local features. However, under few-shot learning (FSL) settings on small datasets with only a few labeled data, ViT tends to overfit and suffers from severe performance degradation due to its absence of CNN-alike inductive bias. Previous works in FSL avoid such problem either through the help of self-supervised auxiliary losses, or through the dextile uses of label information under supervised settings. But the gap between self-supervised and supervised few-shot Transformers is still unfilled. Inspired by recent advances in self-supervised knowledge distillation and masked image modeling (MIM), we propose a novel Supervised Masked Knowledge Distillation model (SMKD) for few-shot Transformers which incorporates label information into self-distillation frameworks. Compared with previous self-supervised methods, we allow intra-class knowledge distillation on both class and patch tokens, and introduce the challenging task of masked patch tokens reconstruction across intra-class images. Experimental results on four few-shot classification benchmark datasets show that our method with simple design outperforms previous methods by a large margin and achieves a new start-of-the-art. Detailed ablation studies confirm the effectiveness of each component of our model. Code for this paper is available here: https://github.com/HL-hanlin/SMKD.

        ----

        ## [1870] Modeling the Distributional Uncertainty for Salient Object Detection Models

        **Authors**: *Xinyu Tian, Jing Zhang, Mochu Xiang, Yuchao Dai*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01883](https://doi.org/10.1109/CVPR52729.2023.01883)

        **Abstract**:

        Most of the existing salient object detection (SOD) models focus on improving the overall model performance, without explicitly explaining the discrepancy between the training and testing distributions. In this paper, we investigate a particular type of epistemic uncertainty, namely distributional uncertainty, for salient object detection. Specifically, for the first time, we explore the existing class-aware distribution gap exploration techniques, i.e. long-tail learning, single-model uncertainty modeling and test-time strategies, and adapt them to model the distributional uncertainty for our class-agnostic task. We define test sample that is dissimilar to the training dataset as being “out-of-distribution” (OOD) samples. Different from the conventional OOD definition, where OOD samples are those not belonging to the closed-world training categories, OOD samples for SOD are those break the basic priors of saliency, i.e. center prior, color contrast prior, compactness prior and etc., indicating OOD as being “continuous” instead of being discrete for our task. We've carried out extensive experimental results to verify effectiveness of existing distribution gap modeling techniques for SOD, and conclude that both train-time single-model uncertainty estimation techniques and weight-regularization solutions that preventing model activation from drifting too much are promising directions for modeling distributional uncertainty for SOD.

        ----

        ## [1871] Weak-shot Object Detection through Mutual Knowledge Transfer

        **Authors**: *Xuanyi Du, Weitao Wan, Chong Sun, Chen Li*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01884](https://doi.org/10.1109/CVPR52729.2023.01884)

        **Abstract**:

        Weak-shot Object Detection methods exploit a fully-annotated source dataset to facilitate the detection performance on the target dataset which only contains image-level labels for novel categories. To bridge the gap between these two datasets, we aim to transfer the object knowledge between the source (S) and target (T) datasets in a bi-directional manner. We propose a novel Knowledge Transfer (KT) loss which simultaneously distills the knowledge of objectness and class entropy from a proposal generator trained on the S dataset to optimize a multiple instance learning module on the T dataset. By jointly optimizing the classification loss and the proposed KT loss, the multiple instance learning module effectively learns to classify object proposals into novel categories in the T dataset with the transferred knowledge from base categories in the S dataset. Noticing the predicted boxes on the T dataset can be regarded as an extension for the original annotations on the S dataset to refine the proposal generator in return, we further propose a novel Consistency Filtering (CF) method to reliably remove inaccurate pseudo labels by evaluating the stability of the multiple instance learning module upon noise injections. Via mutually transferring knowledge between the S and T datasets in an iterative manner, the detection performance on the target dataset is significantly improved. Extensive experiments on public benchmarks validate that the proposed method performs favourably against the state-of-the-art methods without increasing the model parameters or inference computational complexity.

        ----

        ## [1872] CAT: LoCalization and IdentificAtion Cascade Detection Transformer for Open-World Object Detection

        **Authors**: *Shuailei Ma, Yuefeng Wang, Ying Wei, Jiaqi Fan, Thomas H. Li, Hongli Liu, Fanbing Lv*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01885](https://doi.org/10.1109/CVPR52729.2023.01885)

        **Abstract**:

        Open-world object detection (OWOD), as a more general and challenging goal, requires the model trained from data on known objects to detect both known and unknown objects and incrementally learn to identify these unknown objects. The existing works which employ standard detection framework and fixed pseudo-labelling mechanism 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$(PLM)$</tex>
 have the following problems: (i) The inclusion of detecting unknown objects substantially reduces the model's ability to detect known ones. (ii) The 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$PLM$</tex>
 does not adequately utilize the priori knowledge of inputs. (iii) The fixed selection manner of 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$PLM$</tex>
 cannot guarantee that the model is trained in the right direction. We observe that humans subconsciously prefer to focus on all foreground objects and then identify each one in detail, rather than localize and identify a single object simultaneously, for alleviating the confusion. This motivates us to propose a novel solution called CAT: LoCalization and IdentificAtion Cascade Detection Transformer which decouples the detection process via the shared decoder in the cascade decoding way. In the meanwhile, we propose the self-adaptive pseudo-labelling mechanism which combines the model-driven with input-driven 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$PLM$</tex>
 and self-adaptively generates robust pseudo-labels for unknown objects, significantly improving the ability of CAT to retrieve unknown objects. Experiments on two benchmarks, i.e., MS-COCO and PASCAL VOC, show that our model outperforms the state-of-the-art methods. The code is publicly available at https://github.com/xiaomabufei/CAT.

        ----

        ## [1873] Adaptive Sparse Pairwise Loss for Object Re-Identification

        **Authors**: *Xiao Zhou, Yujie Zhong, Zhen Cheng, Fan Liang, Lin Ma*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01886](https://doi.org/10.1109/CVPR52729.2023.01886)

        **Abstract**:

        Object re-identification (ReID) aims to find instances with the same identity as the given probe from a large gallery. Pairwise losses play an important role in training a strong ReID network. Existing pairwise losses densely exploit each instance as an anchor and sample its triplets in a mini-batch. This dense sampling mechanism inevitably introduces positive pairs that share few visual similarities, which can be harmful to the training. To address this problem, we propose a novel loss paradigm termed Sparse Pair-wise (SP) loss that only leverages few appropriate pairs for each class in a mini-batch, and empirically demonstrate that it is sufficient for the ReID tasks. Based on the proposed loss framework, we propose an adaptive positive mining strategy that can dynamically adapt to diverse intra-class variations. Extensive experiments show that SP loss and its adaptive variant AdaSP loss outperform other pairwise losses, and achieve state-of-the-art performance across several ReID benchmarks. Code is available at https://github.com/Astaxanthin/AdaSP.

        ----

        ## [1874] DETRs with Hybrid Matching

        **Authors**: *Ding Jia, Yuhui Yuan, Haodi He, Xiaopei Wu, Haojun Yu, Weihong Lin, Lei Sun, Chao Zhang, Han Hu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01887](https://doi.org/10.1109/CVPR52729.2023.01887)

        **Abstract**:

        One-to-one set matching is a key design for DETR to establish its end-to-end capability, so that object detection does not require a hand-crafted NMS (non-maximum suppression) to remove duplicate detections. This end-to-end signature is important for the versatility of DETR, and it has been generalized to broader vision tasks. However, we note that there are few queries assigned as positive samples and the one-to-one set matching significantly reduces the training efficacy of positive samples. We propose a simple yet effective method based on a hybrid matching scheme that combines the original one-to-one matching branch with an auxiliary one-to-many matching branch during training. Our hybrid strategy has been shown to significantly improve accuracy. In inference, only the original one-to-one match branch is used, thus maintaining the end-to-end merit and the same inference efficiency of DETR. The method is namedℋ-DETR, and it shows that a wide range of representative DETR methods can be consistently improved across a wide range of visual tasks, including Deformable-DETR, PETRv2, PETR, and TransTrack, among others. Code is available at: https://github.com/HDETR.

        ----

        ## [1875] Generating Features with Increased Crop-Related Diversity for Few-Shot Object Detection

        **Authors**: *Jingyi Xu, Hieu Le, Dimitris Samaras*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01888](https://doi.org/10.1109/CVPR52729.2023.01888)

        **Abstract**:

        Two-stage object detectors generate object proposals and classify them to detect objects in images. These proposals often do not contain the objects perfectly but overlap with them in many possible ways, exhibiting great variability in the difficulty levels of the proposals. Training a robust classifier against this crop-related variability requires abundant training data, which is not available in few-shot settings. To mitigate this issue, we propose a novel variational autoencoder (VAE) based data generation model, which is capable of generating data with increased croprelated diversity. The main idea is to transform the latent space such latent codes with different norms represent different crop-related variations. This allows us to generate features with increased crop-related diversity in difficulty levels by simply varying the latent norm. In particular, each latent code is rescaled such that its norm linearly correlates with the IoU score of the input crop w.r.t. the ground-truth box. Here the IoU score is a proxy that represents the difficulty level of the crop. We train this VAE model on base classes conditioned on the semantic code of each class and then use the trained model to generate features for novel classes. In our experiments our generated features consistently improve state-of-the-art few-shot object detection methods on the PASCAL VOC and MS COCO datasets.

        ----

        ## [1876] ScaleKD: Distilling Scale-Aware Knowledge in Small Object Detector

        **Authors**: *Yichen Zhu, Qiqi Zhou, Ning Liu, Zhiyuan Xu, Zhicai Ou, Xiaofeng Mou, Jian Tang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01889](https://doi.org/10.1109/CVPR52729.2023.01889)

        **Abstract**:

        Despite the prominent success of general object detection, the performance and efficiency of Small Object Detection (SOD) are still unsatisfactory. Unlike existing works that struggle to balance the tradeoff between inference speed and SOD performance, in this paper, we propose a novel Scale-aware Knowledge Distillation (ScaleKD), which transfers knowledge of a complex teacher model to a compact student model. We design two novel modules to boost the quality of knowledge transfer in distillation for SOD: 1) a scale-decoupled feature distillation module that disentangled teacher's feature representation into multi-scale embedding that enables explicit feature mimicking of the student model on small objects. 2) a cross-scale assistant to refine the noisy and uninformative bounding boxes prediction student models, which can mislead the student model and impair the efficacy of knowledge distillation. A multi-scale cross-attention layer is established to capture the multi-scale semantic information to improve the student model. We conduct experiments on COCO and VisDrone datasets with diverse types of models, i.e., two-stage and one-stage detectors, to evaluate our proposed method. Our ScaleKD achieves superior performance on general detection performance and obtains spectacular improvement regarding the SOD performance.

        ----

        ## [1877] Multiclass Confidence and Localization Calibration for Object Detection

        **Authors**: *Bimsara Pathiraja, Malitha Gunawardhana, Muhammad Haris Khan*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01890](https://doi.org/10.1109/CVPR52729.2023.01890)

        **Abstract**:

        Albeit achieving high predictive accuracy across many challenging computer vision problems, recent studies suggest that deep neural networks (DNNs) tend to make over-confident predictions, rendering them poorly calibrated. Most of the existing attempts for improving DNN calibration are limited to classification tasks and restricted to calibrating in-domain predictions. Surprisingly, very little to no attempts have been made in studying the calibration of object detection methods, which occupy a pivotal space in vision-based security-sensitive, and safety-critical applications. In this paper, we propose a new train-time technique for calibrating modern object detection methods. It is capable of jointly calibrating multiclass confidence and box localization by leveraging their predictive uncertainties. We perform extensive experiments on several in-domain and out-of-domain detection benchmarks. Results demonstrate that our proposed train-time calibration method consistently outperforms several baselines in reducing calibration error for both in-domain and out-of-domain predictions. Our code and models are available at https://github.com/bimsarapathiraja/MCCL

        ----

        ## [1878] Open-Set Representation Learning through Combinatorial Embedding

        **Authors**: *Geeho Kim, Junoh Kang, Bohyung Han*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01891](https://doi.org/10.1109/CVPR52729.2023.01891)

        **Abstract**:

        Visual recognition tasks are often limited to dealing with a small subset of classes simply because the labels for the remaining classes are unavailable. We are interested in identifying novel concepts in a dataset through representation learning based on both labeled and unlabeled examples, and extending the horizon of recognition to both known and novel classes. To address this challenging task, we propose a combinatorial learning approach, which naturally clusters the examples in unseen classes using the compositional knowledge given by multiple supervised meta-classifiers on heterogeneous label spaces. The representations given by the combinatorial embedding are made more robust by unsupervised pairwise relation learning. The proposed algorithm discovers novel concepts via a joint optimization for enhancing the discrimitiveness of unseen classes as well as learning the representations of known classes generalizable to novel ones. Our extensive experiments demonstrate remarkable performance gains by the proposed approach on public datasets for image retrieval and image categorization with novel class discovery.

        ----

        ## [1879] ProD: Prompting-to-disentangle Domain Knowledge for Cross-domain Few-shot Image Classification

        **Authors**: *Tianyi Ma, Yifan Sun, Zongxin Yang, Yi Yang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01892](https://doi.org/10.1109/CVPR52729.2023.01892)

        **Abstract**:

        This paper considers few-shot image classification under the cross-domain scenario, where the train-to-test domain gap compromises classification accuracy. To mitigate the domain gap, we propose a prompting-to-disentangle (ProD) method through a novel exploration with the prompting mechanism. ProD adopts the popular multi-domain training scheme and extracts the backbone feature with a standard Convolutional Neural Network. Based on these two common practices, the key point of ProD is using the prompting mechanism in the transformer to disentangle the domain-general (DG) and domain-specific (DS) knowledge from the backbone feature. Specifically, ProD concatenates a DG and a DS prompt to the backbone feature and feeds them into a lightweight transformer. The DG prompt is learnable and shared by all the training domains, while the DS prompt is generated from the domain-of-interest on the fly. As a result, the transformer outputs DG and DS features in parallel with the two prompts, yielding the disentangling effect. We show that: 1) Simply sharing a single DG prompt for all the training domains already improves generalization towards the novel test domain. 2) The cross-domain generalization can be further reinforced by making the DG prompt neutral towards the training domains. 3) When inference, the DS prompt is generated from the support samples and can capture test domain knowledge through the prompting mechanism. Combining all three benefits, ProD significantly improves cross-domain few-shot classification. For instance, on CUB, ProD improves the 5-way 5-shot ac-curacy from 73.56% (baseline) to 79.19%, setting a new state of the art.

        ----

        ## [1880] Visual Language Pretrained Multiple Instance Zero-Shot Transfer for Histopathology Images

        **Authors**: *Ming Y. Lu, Bowen Chen, Andrew Zhang, Drew F. K. Williamson, Richard J. Chen, Tong Ding, Long Phi Le, Yung-Sung Chuang, Faisal Mahmood*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01893](https://doi.org/10.1109/CVPR52729.2023.01893)

        **Abstract**:

        Contrastive visual language pretraining has emerged as a powerful method for either training new language-aware image encoders or augmenting existing pretrained models with zero-shot visual recognition capabilities. However, existing works typically train on large datasets of imagetext pairs and have been designed to perform downstream tasks involving only small to medium sized-images, neither of which are applicable to the emerging field of computational pathology where there are limited publicly available paired image-text datasets and each image can span up to 100,000 × 100,000 pixels. In this paper we present MI-Zero, a simple and intuitive framework for unleashing the zero-shot transfer capabilities of contrastively aligned image and text models on gigapixel histopathology whole slide images, enabling multiple downstream diagnostic tasks to be carried out by pretrained encoders without requiring any additional labels. MI-Zero reformulates zero-shot transfer under the framework of multiple instance learning to overcome the computational challenge of inference on extremely large images. We used over 550k pathology reports and other available in-domain text corpora to pretrain our text encoder. By effectively leveraging strong pretrained encoders, our best model pretrained on over 33k histopathology image-caption pairs achieves an average median zero-shot accuracy of 70.2% across three different real-world cancer subtyping tasks. Our code is available at: https://github.com/mahmoodlab/MI-Zero.

        ----

        ## [1881] FFF: Fragment-Guided Flexible Fitting for Building Complete Protein Structures

        **Authors**: *Weijie Chen, Xinyan Wang, Yuhang Wang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01894](https://doi.org/10.1109/CVPR52729.2023.01894)

        **Abstract**:

        Cryo-electron microscopy (cryo-EM) is a technique for reconstructing the 3-dimensional (3D) structure of biomolecules (especially large protein complexes and molecular assemblies). As the resolution increases to the near-atomic scale, building protein structures de novo from cryo-EM maps becomes possible. Recently, recognition based de novo building methods have shown the potential to streamline this process. However, it cannot build a complete structure due to the low signal-to-noise ratio (SNR) problem. At the same time, AlphaFold has led to a great break through in predicting protein structures. This has inspired us to combine fragment recognition and structure prediction methods to build a complete structure. In this paper, we propose a new method named FFF that bridges protein structure prediction and protein structure recognition with flexible fitting. First, a multi-level recognition network is used to capture various structural features from the input 3D cryo-EM map. Next, protein structural fragments are generated using pseudo peptide vectors and a protein sequence alignment method based on these extracted features. Finally, a complete structural model is constructed using the predicted protein fragments via flexible fitting. Based on our benchmark tests, FFF outperforms the baseline methods for building complete protein structures.

        ----

        ## [1882] Pseudo-Label Guided Contrastive Learning for Semi-Supervised Medical Image Segmentation

        **Authors**: *Hritam Basak, Zhaozheng Yin*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01895](https://doi.org/10.1109/CVPR52729.2023.01895)

        **Abstract**:

        Although recent works in semi-supervised learning (SemiSL) have accomplished significant success in natural image segmentation, the task of learning discriminative representations from limited annotations has been an open problem in medical images. Contrastive Learning (CL) frameworks use the notion of similarity measure which is useful for classification problems, however, they fail to transfer these quality representations for accurate pixel-level segmentation. To this end, we propose a novel semi-supervised patch-based CL framework for medical image segmentation without using any explicit pretext task. We harness the power of both CL and SemiSL, where the pseudo-labels generated from SemiSL aid CL by providing additional guidance, whereas discriminative class information learned in CL leads to accurate multi-class segmentation. Additionally, we formulate a novel loss that synergistically encourages inter-class separability and intraclass compactness among the learned representations. A new inter-patch semantic disparity mapping using average patch entropy is employed for a guided sampling of positives and negatives in the proposed CL framework. Experimental analysis on three publicly available datasets of multiple modalities reveals the superiority of our proposed method as compared to the state-of-the-art methods. Code is available at: GitHub.

        ----

        ## [1883] Hierarchical Discriminative Learning Improves Visual Representations of Biomedical Microscopy

        **Authors**: *Cheng Jiang, Xinhai Hou, Akhil Kondepudi, Asadur Chowdury, Christian W. Freudiger, Daniel A. Orringer, Honglak Lee, Todd C. Hollon*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01896](https://doi.org/10.1109/CVPR52729.2023.01896)

        **Abstract**:

        Learning high-quality, self-supervised, visual representations is essential to advance the role of computer vision in biomedical microscopy and clinical medicine. Previous work has focused on self-supervised representation learning (SSL) methods developed for instance discrimination and applied them directly to image patches, or fields-of-view, sampled from gigapixel whole-slide images (WSIs) used for cancer diagnosis. However, this strategy is limited because it (1) assumes patches from the same patient are independent, (2) neglects the patient-slide-patch hierarchy of clinical biomedical microscopy, and (3) requires strong data augmentations that can degrade downstream performance. Importantly, sampled patches from WSIs of a patient's tumor are a diverse set of image examples that capture the same underlying cancer diagnosis. This motivated HiDisc, a data-driven method that leverages the inherent patient-slide-patch hierarchy of clinical biomedical microscopy to define a hierarchical discriminative learning task that implicitly learns features of the underlying diagnosis. HiDisc uses a self-supervised contrastive learning framework in which positive patch pairs are defined based on a common ancestry in the data hierarchy, and a unified patch, slide, and patient discriminative learning objective is used for visual SSL. We benchmark HiDisc visual representations on two vision tasks using two biomedical microscopy datasets, and demonstrate that (1) HiDisc Pre-training outperforms current state-of-the-art self-supervised Pre-training methods for cancer diagnosis and genetic mutation prediction, and (2) HiDisc learns high-quality visual representations using natural patch diversity without strong data augmentations.

        ----

        ## [1884] KiUT: Knowledge-injected U-Transformer for Radiology Report Generation

        **Authors**: *Zhongzhen Huang, Xiaofan Zhang, Shaoting Zhang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01897](https://doi.org/10.1109/CVPR52729.2023.01897)

        **Abstract**:

        Radiology report generation aims to automatically generate a clinically accurate and coherent paragraph from the X-ray image, which could relieve radiologists from the heavy burden of report writing. Although various image caption methods have shown remarkable performance in the natural image field, generating accurate reports for medical images requires knowledge of multiple modalities, including vision, language, and medical terminology. We propose a Knowledge-injected U-Transformer (KiUT) to learn multi-level visual representation and adaptively distill the information with contextual and clinical knowledge for word prediction. In detail, a U-connection schema between the encoder and decoder is designed to model interactions between different modalities. And a symptom graph and an injected knowledge distiller are developed to assist the report generation. Experimentally, we outperform state-of-the-art methods on two widely used benchmark datasets: IU-Xray and MIMIC-CXR. Further experimental results prove the advantages of our architecture and the complementary benefits of the injected knowledge.

        ----

        ## [1885] Image Quality-aware Diagnosis via Meta-knowledge Co-embedding

        **Authors**: *Haoxuan Che, Siyu Chen, Hao Chen*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01898](https://doi.org/10.1109/CVPR52729.2023.01898)

        **Abstract**:

        Medical images usually suffer from image degradation in clinical practice, leading to decreased performance of deep learning-based models. To resolve this problem, most previous works have focused on filtering out degradation-causing low-quality images while ignoring their potential value for models. Through effectively learning and leveraging the knowledge of degradations, models can better resist their adverse effects and avoid misdiagnosis. In this paper, we raise the problem of image quality-aware diagnosis, which aims to take advantage of low-quality images and image quality labels to achieve a more accurate and robust diagnosis. However, the diversity of degradations and superficially unrelated targets between image quality assessment and disease diagnosis makes it still quite challenging to effectively leverage quality labels to assist diagnosis. Thus, to tackle these issues, we propose a novel meta-knowledge co-embedding network, consisting of two subnets: Task Net and Meta Learner. Task Net constructs an explicit quality information utilization mechanism to enhance diagnosis via knowledge co-embedding features, while Meta Learner ensures the effectiveness and constrains the semantics of these features via meta-learning and joint-encoding masking. Superior performance on five datasets with four widely-used medical imaging modalities demonstrates the effectiveness and generalizability of our method.

        ----

        ## [1886] Interventional Bag Multi-Instance Learning On Whole-Slide Pathological Images

        **Authors**: *Tiancheng Lin, Zhimiao Yu, Hongyu Hu, Yi Xu, Chang Wen Chen*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01899](https://doi.org/10.1109/CVPR52729.2023.01899)

        **Abstract**:

        Multi-instance learning (MIL) is an effective paradigm for whole-slide pathological images (WSIs) classification to handle the gigapixel resolution and slide-level label. Prevailing MIL methods primarily focus on improving the feature extractor and aggregator. However, one deficiency of these methods is that the bag contextual prior may trick the model into capturing spurious correlations between bags and labels. This deficiency is a confounder that limits the performance of existing MIL methods. In this paper, we propose a novel scheme, Interventional Bag Multi-Instance Learning (IBMIL), to achieve deconfounded bag-level prediction. Unlike traditional likelihood-based strategies, the proposed scheme is based on the backdoor adjustment to achieve the interventional training, thus is capable of suppressing the bias caused by the bag contextual prior. Note that the principle of IBMIL is orthogonal to existing bag MIL methods. Therefore, IBMIL is able to bring consistent performance boosting to existing schemes, achieving new state-of-the-art performance. Code is available at https://github.com/HHHedo/IBMIL.

        ----

        ## [1887] Visual Prompt Tuning for Generative Transfer Learning

        **Authors**: *Kihyuk Sohn, Huiwen Chang, José Lezama, Luisa Polania, Han Zhang, Yuan Hao, Irfan Essa, Lu Jiang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01900](https://doi.org/10.1109/CVPR52729.2023.01900)

        **Abstract**:

        Learning generative image models from various domains efficiently needs transferring knowledge from an image synthesis model trained on a large dataset. We present a recipe for learning vision transformers by generative knowledge transfer. We base our framework on generative vision transformers representing an image as a sequence of visual tokens with the autoregressive or non-autoregressive transformers. To adapt to a new domain, we employ prompt tuning, which prepends learnable tokens called prompts to the image token sequence and introduces a new prompt design for our task. We study on a variety of visual domains with varying amounts of training images. We show the effectiveness of knowledge transfer and a significantly better image generation quality.
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
https://github.com/google-research/generative_transfer

        ----

        ## [1888] LINe: Out-of-Distribution Detection by Leveraging Important Neurons

        **Authors**: *Yong Hyun Ahn, Gyeong-Moon Park, Seong Tae Kim*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01901](https://doi.org/10.1109/CVPR52729.2023.01901)

        **Abstract**:

        It is important to quantify the uncertainty of input samples, especially in mission-critical domains such as autonomous driving and healthcare, where failure predictions on out-of-distribution (OOD) data are likely to cause big problems. OOD detection problem fundamentally begins in that the model cannot express what it is not aware of. Post-hoc OOD detection approaches are widely explored because they do not require an additional re-training process which might degrade the model's performance and increase the training cost. In this study, from the perspective of neurons in the deep layer of the model representing high-level features, we introduce a new aspect for analyzing the difference in model outputs between in-distribution data and OOD data. We propose a novel method, Leveraging Important Neurons (LINe), for post-hoc Out of distribution detection. Shapley value-based pruning reduces the effects of noisy outputs by selecting only high-contribution neurons for predicting specific classes of input data and masking the rest. Activation clipping fixes all values above a certain threshold into the same value, allowing LINe to treat all the class-specific features equally and just consider the difference between the number of activated feature differences between in-distribution and OOD data. Comprehensive experiments verify the effectiveness of the proposed method by outperforming state-of-the-art post-hoc OOD detection methods on CIFAR-10, CIFAR-100, and ImageNet datasets. Code is available on https://github.com/LINe-OOD

        ----

        ## [1889] GCFAgg: Global and Cross-View Feature Aggregation for Multi-View Clustering

        **Authors**: *Weiqing Yan, Yuanyang Zhang, Chenlei Lv, Chang Tang, Guanghui Yue, Liang Liao, Weisi Lin*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01902](https://doi.org/10.1109/CVPR52729.2023.01902)

        **Abstract**:

        Multi-view clustering can partition data samples into their categories by learning a consensus representation in unsupervised way and has received more and more attention in recent years. However, most existing deep clustering methods learn consensus representation or view-specific representations from multiple views via view-wise aggregation way, where they ignore structure relationship of all samples. In this paper, we propose a novel multi-view clustering network to address these problems, called Global and Cross-view Feature Aggregation for Multi-View Clustering (GCFAggMVC). Specifically, the consensus data presentation from multiple views is obtained via cross-sample and cross-view feature aggregation, which fully explores the complementary of similar samples. Moreover, we align the consensus representation and the view-specific representation by the structure-guided contrastive learning module, which makes the view-specific representations from different samples with high structure relationship similar. The proposed module is a flexible multi-view data representation module, which can be also embedded to the incomplete multi-view data clustering task via plugging our module into other frameworks. Extensive experiments show that the proposed method achieves excellent performance in both complete multi-view data clustering tasks and incomplete multi-view data clustering tasks.

        ----

        ## [1890] Exploring and Exploiting Uncertainty for Incomplete Multi-View Classification

        **Authors**: *Mengyao Xie, Zongbo Han, Changqing Zhang, Yichen Bai, Qinghua Hu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01903](https://doi.org/10.1109/CVPR52729.2023.01903)

        **Abstract**:

        Classifying incomplete multi-view data is inevitable since arbitrary view missing widely exists in real-world applications. Although great progress has been achieved, existing incomplete multi-view methods are still difficult to obtain a trustworthy prediction due to the relatively high uncertainty nature of missing views. First, the missing view is of high uncertainty, and thus it is not reasonable to provide a single deterministic imputation. Second, the quality of the imputed data itself is of high uncertainty. To explore and exploit the uncertainty, we propose an Uncertainty-induced Incomplete Multi-View Data Classification (UIMC) model to classify the incomplete multi-view data under a stable and reliable framework. We construct a distribution and sample multiple times to characterize the uncertainty of missing views, and adaptively utilize them according to the sampling quality. Accordingly, the proposed method realizes more perceivable imputation and controllable fusion. Specifically, we model each missing data with a distribution conditioning on the available views and thus introducing uncertainty. Then an evidence-based fusion strategy is employed to guarantee the trustworthy integration of the imputed views. Extensive experiments are conducted on multiple benchmark data sets and our method establishes a state-of-the-art performance in terms of both performance and trustworthiness.

        ----

        ## [1891] BiCro: Noisy Correspondence Rectification for Multi-modality Data via Bi-directional Cross-modal Similarity Consistency

        **Authors**: *Shuo Yang, Zhaopan Xu, Kai Wang, Yang You, Hongxun Yao, Tongliang Liu, Min Xu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01904](https://doi.org/10.1109/CVPR52729.2023.01904)

        **Abstract**:

        As one of the most fundamental techniques in multi-modal learning, cross-modal matching aims to project various sensory modalities into a shared feature space. To achieve this, massive and correctly aligned data pairs are required for model training. However, unlike unimodal datasets, multimodal datasets are extremely harder to collect and annotate precisely. As an alternative, the co-occurred data pairs (e.g., image-text pairs) collected from the Internet have been widely exploited in the area. Unfortunately, the cheaply collected dataset unavoidably contains many mismatched data pairs, which have been proven to be harmful to the model's performance. To address this, we propose a general framework called BiCro (Bidirectional Cross-modal similarity consistency), which can be easily integrated into existing cross-modal matching models and improve their robustness against noisy data. Specifically, BiCro aims to estimate soft labels for noisy data pairs to reflect their true correspondence degree. The basic idea of BiCro is motivated by that – taking image-text matching as an example – similar images should have similar textual descriptions and vice versa. Then the consistency of these two similarities can be recast as the estimated soft labels to train the matching model. The experiments on three popular cross-modal matching datasets demonstrate that our method significantly improves the noise-robustness of various matching models, and surpass the state-of-the-art by a clear margin. The code is available at https://github.com/xu5zhao/BiCro.

        ----

        ## [1892] Bi-Directional Distribution Alignment for Transductive Zero-Shot Learning

        **Authors**: *Zhicai Wang, Yanbin Hao, Tingting Mu, Ouxiang Li, Shuo Wang, Xiangnan He*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01905](https://doi.org/10.1109/CVPR52729.2023.01905)

        **Abstract**:

        Zero-shot learning (ZSL) suffers intensely from the domain shift issue, i.e., the mismatch (or misalignment) between the true and learned data distributions for classes without training data (unseen classes). By learning additionally from unlabelled data collected for the unseen classes, transductive ZSL (TZSL) could reduce the shift but only to a certain extent. To improve TZSL, we propose a novel approach Bi-VAEGAN which strengthens the distribution alignment between the visual space and an auxiliary space. As a result, it can reduce largely the domain shift. The proposed key designs include (1) a bi-directional distribution alignment, (2) a simple but effective L
<inf xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">2</inf>
-norm based feature normalization approach, and (3) a more sophisticated unseen class prior estimation. Evaluated by four benchmark datasets, Bi-VAEGAN
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
Code is available at https://github.com/Zhicaiwww/Bi-VAEGAN achieves the new state of the art under both the standard and generalized TZSL settings.

        ----

        ## [1893] HIER: Metric Learning Beyond Class Labels via Hierarchical Regularization

        **Authors**: *Sungyeon Kim, Boseung Jeong, Suha Kwak*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01906](https://doi.org/10.1109/CVPR52729.2023.01906)

        **Abstract**:

        Supervision for metric learning has long been given in the form of equivalence between human-labeled classes. Although this type of supervision has been a basis of metric learning for decades, we argue that it hinders further advances in the field. In this regard, we propose a new regularization method, dubbed HIER, to discover the latent semantic hierarchy of training data, and to deploy the hierarchy to provide richer and more fine-grained supervision than inter-class separability induced by common metric learning losses. HIER achieves this goal with no annotation for the semantic hierarchy but by learning hierarchical proxies in hyperbolic spaces. The hierarchical proxies are learnable parameters, and each of them is trained to serve as an ancestor of a group of data or other proxies to approximate the semantic hierarchy among them. HIER deals with the proxies along with data in hyperbolic space since the geometric properties of the space are well-suited to represent their hierarchical structure. The efficacy of HIER is evaluated on four standard benchmarks, where it consistently improved the performance of conventional methods when integrated with them, and consequently achieved the best records, surpassing even the existing hyperbolic metric learning technique, in almost all settings.

        ----

        ## [1894] MaskCon: Masked Contrastive Learning for Coarse-Labelled Dataset

        **Authors**: *Chen Feng, Ioannis Patras*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01907](https://doi.org/10.1109/CVPR52729.2023.01907)

        **Abstract**:

        Deep learning has achieved great success in recent years with the aid of advanced neural network structures and large-scale human-annotated datasets. However, it is often costly and difficult to accurately and efficiently annotate large-scale datasets, especially for some specialized domains where fine-grained labels are required. In this setting, coarse labels are much easier to acquire as they do not require expert knowledge. In this work, we propose a contrastive learning method, called masked contrastive learning (MaskCon) to address the under-explored problem setting, where we learn with a coarse-labelled dataset in order to address a finer labelling problem. More specifically, within the contrastive learning framework, for each sample our method generates soft-labels with the aid of coarse labels against other samples and another augmented view of the sample in question. By contrast to self-supervised contrastive learning where only the sample's augmentations are considered hard positives, and in supervised contrastive learning where only samples with the same coarse labels are considered hard positives, we propose soft labels based on sample distances, that are masked by the coarse labels. This allows us to utilize both inter-sample relations and coarse labels. We demonstrate that our method can obtain as special cases many existing state-of-the-art works and that it provides tighter bounds on the generalization error. Experimentally, our method achieves significant improvement over the current state-of-the-art in various datasets, including CIFAR10, CIFAR100, ImageNet-1K, Standford Online Products and Stanford Cars196 datasets. Code and annotations are available at https://github.com/MrChenFeng/MaskCon_CVPR2023.

        ----

        ## [1895] Class Prototypes based Contrastive Learning for Classifying Multi-Label and Fine-Grained Educational Videos

        **Authors**: *Rohit Gupta, Anirban Roy, Claire Christensen, Sujeong Kim, Sarah Gerard, Madeline Cincebeaux, Ajay Divakaran, Todd Grindal, Mubarak Shah*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01908](https://doi.org/10.1109/CVPR52729.2023.01908)

        **Abstract**:

        The recent growth in the consumption of online media by children during early childhood necessitates data-driven tools enabling educators to filter out appropriate educational content for young learners. This paper presents an approach for detecting educational content in online videos. We focus on two widely used educational content classes: literacy and math. For each class, we choose prominent codes (sub-classes) based on the Common Core Standards. For example, literacy codes include ‘letter names’, ‘letter sounds’, and math codes include ‘counting’, ‘sorting’. We pose this as a finegrained multilabel classification problem as videos can contain multiple types of educational content and the content classes can get visually similar (e.g., ‘letter names’vs ‘letter sounds’). We propose a novel class prototypes based supervised contrastive learning approach that can handle fine-grained samples associated with multiple labels. We learn a class prototype for each class and a loss function is employed to minimize the distances between a class prototype and the samples from the class. Similarly, distances between a class prototype and the samples from other classes are maximized. As the alignment between visual and audio cues are crucial for effective comprehension, we consider a multimodal transformer network to capture the interaction between visual and audio cues in videos while learning the embedding for videos. For evaluation, we present a dataset, APPROVE, employing educational videos from YouTube labeled with fine-grained education classes by education researchers. APPROVE consists of 193 hours of expert-annotated videos with 19 classes. The proposed approach outperforms strong baselines on APPROVE and other benchmarks such as Youtube-8M, and COIN. The dataset is available at https://nusci.csl.sri.com/project/APPROVE.

        ----

        ## [1896] Learning from Noisy Labels with Decoupled Meta Label Purifier

        **Authors**: *Yuanpeng Tu, Boshen Zhang, Yuxi Li, Liang Liu, Jian Li, Yabiao Wang, Chengjie Wang, Cairong Zhao*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01909](https://doi.org/10.1109/CVPR52729.2023.01909)

        **Abstract**:

        Training deep neural networks (DNN) with noisy labels is challenging since DNN can easily memorize inaccurate labels, leading to poor generalization ability. Recently, the meta-learning based label correction strategy is widely adopted to tackle this problem via identifying and correcting potential noisy labels with the help of a small set of clean validation data. Although training with purified labels can effectively improve performance, solving the meta-learning problem inevitably involves a nested loop of bi-level optimization between model weights and hyper-parameters (i.e., label distribution). As compromise, previous methods resort to a coupled learning process with alternating update. In this paper, we empirically find such simultaneous optimization over both model weights and label distribution can not achieve an optimal routine, consequently limiting the representation ability of backbone and accuracy of corrected labels. From this observation, a novel multi-stage label purifier named DMLP is proposed. DMLP decouples the label correction process into label-free representation learning and a simple meta label purifier, In this way, DMLP can focus on extracting discriminative feature and label correction in two distinctive stages. DMLP is a plug-and-play label purifier, the purified labels can be directly reused in naive end-to-end network retraining or other robust learning methods, where state-of-the-art results are obtained on several synthetic and real-world noisy datasets, especially under high noise levels. Code is available at https://github.com/yuanpengtu/DMLP.

        ----

        ## [1897] SuperDisco: Super-Class Discovery Improves Visual Recognition for the Long-Tail

        **Authors**: *Yingjun Du, Jiayi Shen, Xiantong Zhen, Cees G. M. Snoek*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01910](https://doi.org/10.1109/CVPR52729.2023.01910)

        **Abstract**:

        Modern image classifiers perform well on populated classes, while degrading considerably on tail classes with only a few instances. Humans, by contrast, effortlessly handle the long-tailed recognition challenge, since they can learn the tail representation based on different levels of semantic abstraction, making the learned tail features more discriminative. This phenomenon motivated us to propose SuperDisco, an algorithm that discovers super-class representations for long-tailed recognition using a graph model. We learn to construct the super-class graph to guide the representation learning to deal with long-tailed distributions. Through message passing on the super-class graph, image representations are rectified and refined by attending to the most relevant entities based on the semantic similarity among their super-classes. Moreover, we propose to meta-learn the super-class graph under the supervision of a prototype graph constructed from a small amount of imbalanced data. By doing so, we obtain a more robust super-class graph that further improves the long-tailed recognition performance. The consistent state-of-the-art experiments on the long-tailed CIFAR-100, ImageNet, Places and iNaturalist demonstrate the benefit of the discovered super-class graph for dealing with long-tailed distributions.

        ----

        ## [1898] Why is the Winner the Best?

        **Authors**: *Matthias Eisenmann, Annika Reinke, Vivienn Weru, Minu Dietlinde Tizabi, Fabian Isensee, Tim J. Adler, Sharib Ali, Vincent Andrearczyk, Marc Aubreville, Ujjwal Baid, Spyridon Bakas, Niranjan Balu, Sophia Bano, Jorge Bernal, Sebastian Bodenstedt, Alessandro Casella, Veronika Cheplygina, Marie Daum, Marleen de Bruijne, Adrien Depeursinge, Reuben Dorent, Jan Egger, David G. Ellis, Sandy Engelhardt, Melanie Ganz, Noha M. Ghatwary, Gabriel Girard, Patrick Godau, Anubha Gupta, Lasse Hansen, Kanako Harada, Mattias P. Heinrich, Nicholas Heller, Alessa Hering, Arnaud Huaulmé, Pierre Jannin, A. Emre Kavur, Oldrich Kodym, Michal Kozubek, Jianning Li, Hongwei Bran Li, Jun Ma, Carlos Martín-Isla, Bjoern H. Menze, J. Alison Noble, Valentin Oreiller, Nicolas Padoy, Sarthak Pati, Kelly Payette, Tim Rädsch, Jonathan Rafael-Patino, Vivek Singh Bawa, Stefanie Speidel, Carole H. Sudre, Kimberlin M. H. van Wijnen, Martin Wagner, D. Wei, Amine Yamlahi, Moi Hoon Yap, C. Yuan, Maximilian Zenk, A. Zia, David Zimmerer, Dogu Baran Aydogan, Binod Bhattarai, Louise Bloch, Raphael Brüngel, J. Cho, C. Choi, Q. Dou, Ivan Ezhov, Christoph M. Friedrich, C. Fuller, Rebati Raman Gaire, Adrian Galdran, Álvaro García-Faura, Maria Grammatikopoulou, S. Hong, Mostafa Jahanifar, I. Jang, Abdolrahim Kadkhodamohammadi, I. Kang, Florian Kofler, S. Kondo, Hugo Jaco Kuijf, M. Li, M. Luu, Tomaz Martincic, Pedro Morais, Mohamed A. Naser, Bruno Oliveira, David Owen, S. Pang, J. Park, S. Park, Szymon Plotka, Élodie Puybareau, Nasir M. Rajpoot, K. Ryu, Numan Saeed, Adam Shephard, Pengcheng Shi, Dejan Stepec, Ronast Subedi, Guillaume Tochon, Helena R. Torres, Hélène Urien, João L. Vilaça, Kareem A. Wahid, H. Wang, J. Wang, L. Wang, X. Wang, Benedikt Wiestler, Marek Wodzinski, F. Xia, J. Xie, Z. Xiong, S. Yang, Y. Yang, Z. Zhao, Klaus H. Maier-Hein, Paul F. Jäger, Annette Kopp-Schneider, Lena Maier-Hein*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01911](https://doi.org/10.1109/CVPR52729.2023.01911)

        **Abstract**:

        International benchmarking competitions have become fundamental for the comparative performance assessment of image analysis methods. However, little attention has been given to investigating what can be learnt from these competitions. Do they really generate scientific progress? What are common and successful participation strategies? What makes a solution superior to a competing method? To address this gap in the literature, we performed a multicenter study with all 80 competitions that were conducted in the scope of IEEE ISBI 2021 and MICCAI 2021. Statistical analyses performed based on comprehensive descriptions of the submitted algorithms linked to their rank as well as the underlying participation strategies revealed common characteristics of winning solutions. These typically include the use of multi-task learning (63%) and/or multi-stage pipelines (61%), and a focus on augmentation (100%), image preprocessing (97%), data curation (79%), and post-processing (66%). The “typical” lead of a winning team is a computer scientist with a doctoral degree, five years of experience in biomedical image analysis, and four years of experience in deep learning. Two core general development strategies stood out for highly-ranked teams: the reflection of the metrics in the method design and the focus on analyzing and handling failure cases. According to the organizers, 43% of the winning algorithms exceeded the state of the art but only 11% completely solved the respective domain problem. The insights of our study could help researchers (1) improve algorithm development strategies when approaching new problems, and (2) focus on open research questions revealed by this work.

        ----

        ## [1899] Balanced Product of Calibrated Experts for Long-Tailed Recognition

        **Authors**: *Emanuel Sanchez Aimar, Arvi Jonnarth, Michael Felsberg, Marco Kuhlmann*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01912](https://doi.org/10.1109/CVPR52729.2023.01912)

        **Abstract**:

        Many real-world recognition problems are characterized by long-tailed label distributions. These distributions make representation learning highly challenging due to limited generalization over the tail classes. If the test distribution differs from the training distribution, e.g. uniform versus long-tailed, the problem of the distribution shift needs to be addressed. A recent line of work proposes learning multiple diverse experts to tackle this issue. Ensemble diversity is encouraged by various techniques, e.g. by specializing different experts in the head and the tail classes. In this work, we take an analytical approach and extend the notion of logit adjustment to ensembles to form a Balanced Product of Experts (BalPoE). BalPoE combines a family of experts with different test-time target distributions, generalizing several previous approaches. We show how to properly define these distributions and combine the experts in order to achieve unbiased predictions, by proving that the ensemble is Fisher-consistent for minimizing the balanced error. Our theoretical analysis shows that our balanced ensemble requires calibrated experts, which we achieve in practice using mixup. We conduct extensive experiments and our method obtains new state-of-the-art results on three long-tailed datasets: CIFAR-100-LT, ImageNet-LT, and iNaturalist-2018. Our code is available at https://github.com/emasa/BalPoE-CalibratedLT.

        ----

        ## [1900] Transfer Knowledge from Head to Tail: Uncertainty Calibration under Long-tailed Distribution

        **Authors**: *Jiahao Chen, Bing Su*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01913](https://doi.org/10.1109/CVPR52729.2023.01913)

        **Abstract**:

        How to estimate the uncertainty of a given model is a crucial problem. Current calibration techniques treat different classes equally and thus implicitly assume that the distribution of training data is balanced, but ignore the fact that real-world data often follows a long-tailed distribution. In this paper, we explore the problem of calibrating the model trained from a long-tailed distribution. Due to the difference between the imbalanced training distribution and balanced test distribution, existing calibration methods such as temperature scaling can not generalize well to this problem. Specific calibration methods for domain adaptation are also not applicable because they rely on unlabeled target domain instances which are not available. Models trained from a long-tailed distribution tend to be more over-confident to head classes. To this end, we propose a novel knowledge-transferring-based calibration method by estimating the importance weights for samples of tail classes to realize long-tailed calibration. Our method models the distribution of each class as a Gaussian distribution and views the source statistics of head classes as a prior to calibrate the target distributions of tail classes. We adaptively transfer knowledge from head classes to get the target probability density of tail classes. The importance weight is estimated by the ratio of the target probability density over the source probability density. Extensive experiments on CIFAR-10-LT, MNIST-LT, CIFAR-100-LT, and ImageNet-LT datasets demonstrate the effectiveness of our method.

        ----

        ## [1901] FREDOM: Fairness Domain Adaptation Approach to Semantic Scene Understanding

        **Authors**: *Thanh-Dat Truong, Ngan Le, Bhiksha Raj, Jackson David Cothren, Khoa Luu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01914](https://doi.org/10.1109/CVPR52729.2023.01914)

        **Abstract**:

        Although Domain Adaptation in Semantic Scene Segmentation has shown impressive improvement in recent years, the fairness concerns in the domain adaptation have yet to be well defined and addressed. In addition, fairness is one of the most critical aspects when deploying the segmentation models into human-related real-world applications, e.g., autonomous driving, as any unfair predictions could influence human safety. In this paper, we propose a novel Fairness Domain Adaptation (FREDOM) approach to semantic scene segmentation. In particular, from the proposed formulated fairness objective, a new adaptation framework will be introduced based on the fair treatment of class distributions. Moreover, to generally model the context of structural dependency, a new conditional structural constraint is introduced to impose the consistency of predicted segmentation. Thanks to the proposed Conditional Structure Network, the self-attention mechanism has sufficiently modeled the structural information of segmentation. Through the ablation studies, the proposed method has shown the performance improvement of the segmentation models and promoted fairness in the model predictions. The experimental results on the two standard benchmarks, i.e., SYNTHIA 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$\rightarrow$</tex>
 Cityscapes and GTA5 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$\rightarrow$</tex>
 Cityscapes, have shown that our method achieved State-of-the-Art (SOTA) performance
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
The implementation of FREDOM is available at https://github.com/uark-cviu/FREDOM

        ----

        ## [1902] COT: Unsupervised Domain Adaptation with Clustering and Optimal Transport

        **Authors**: *Yang Liu, Zhipeng Zhou, Baigui Sun*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01915](https://doi.org/10.1109/CVPR52729.2023.01915)

        **Abstract**:

        Unsupervised domain adaptation (UDA) aims to transfer the knowledge from a labeled source domain to an unlabeled target domain. Typically, to guarantee desirable knowledge transfer, aligning the distribution between source and target domain from a global perspective is widely adopted in UDA. Recent researchers further point out the importance of local-level alignment and propose to construct instance-pair alignment by leveraging on Optimal Transport (OT) theory. However, existing OT-based UDA approaches are limited to handling class imbalance challenges and introduce a heavy computation overhead when considering a large-scale training situation. To cope with two aforementioned issues, we propose a Clustering-based Optimal Transport (COT) algorithm, which formulates the alignment procedure as an Optimal Transport problem and constructs a mapping between clustering centers in the source and target domain via an end-to-end manner. With this alignment on clustering centers, our COT eliminates the negative effect caused by class imbalance and reduces the computation cost simultaneously. Empirically, our COT achieves state-of-the-art performance on several authoritative benchmark datasets.

        ----

        ## [1903] MHPL: Minimum Happy Points Learning for Active Source Free Domain Adaptation

        **Authors**: *Fan Wang, Zhongyi Han, Zhiyan Zhang, Rundong He, Yilong Yin*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01916](https://doi.org/10.1109/CVPR52729.2023.01916)

        **Abstract**:

        Source free domain adaptation (SFDA) aims to transfer a trained source model to the unlabeled target domain without accessing the source data. However, the SFDA setting faces a performance bottleneck due to the absence of source data and target supervised information, as evidenced by the limited performance gains of the newest SFDA methods. Active source free domain adaptation (ASFDA) can break through the problem by exploring and exploiting a small set of informative samples via active learning. In this paper, we first find that those satisfying the proper-ties of neighbor-chaotic, individual-different, and source-dissimilar are the best points to select. We define them as the minimum happy (MH) points challenging to explore with existing methods. We propose minimum happy points learning (MHPL) to explore and exploit MH points actively. We design three unique strategies: neighbor environment uncertainty, neighbor diversity relaxation, and one-shot querying, to explore the MH points. Further, to fully exploit MH points in the learning process, we design a neighbor focal loss that assigns the weighted neighbor purity to the cross entropy loss of MH points to make the model focus more on them. Extensive experiments verify that MHPL remarkably exceeds the various types of baselines and achieves significant performance gains at a small cost of labeling.

        ----

        ## [1904] Upcycling Models Under Domain and Category Shift

        **Authors**: *Sanqing Qu, Tianpei Zou, Florian Röhrbein, Cewu Lu, Guang Chen, Dacheng Tao, Changjun Jiang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01917](https://doi.org/10.1109/CVPR52729.2023.01917)

        **Abstract**:

        Deep neural networks (DNNs) often perform poorly in the presence of domain shift and category shift. How to upcycle DNNs and adapt them to the target task remains an important open problem. Unsupervised Domain Adaptation (UDA), especially recently proposed Source-free Domain Adaptation (SFDA), has become a promising technology to address this issue. Nevertheless, existing SFDA methods require that the source domain and target domain share the same label space, consequently being only applicable to the vanilla closed-set setting. In this paper, we take one step further and explore the Source-free Universal Domain Adaptation (SF-UniDA). The goal is to identify “known” data samples under both domain and category shift, and reject those “unknown” data samples (not present in source classes), with only the knowledge from standard pre-trained source model. To this end, we introduce an innovative global and local clustering learning technique (GLC). Specifically, we design a novel, adaptive one-vs-all global clustering algorithm to achieve the distinction across different target classes and introduce a local k-NN clustering strategy to alleviate negative transfer. We examine the superiority of our GLC on multiple benchmarks with different category shift scenarios, including partial-set, open-set, and open-partial-set DA. Remarkably, in the most challenging open-partial-set DA scenario, GLC outperforms UMAD by 14.8% on the VisDA benchmark. The code is available at https://github.com/ispc-lab/GLC.

        ----

        ## [1905] PMR: Prototypical Modal Rebalance for Multimodal Learning

        **Authors**: *Yunfeng Fan, Wenchao Xu, Haozhao Wang, Junxiao Wang, Song Guo*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01918](https://doi.org/10.1109/CVPR52729.2023.01918)

        **Abstract**:

        Multimodal learning (MML) aims to jointly exploit the common priors of different modalities to compensate for their inherent limitations. However, existing MML methods often optimize a uniform objective for different modalities, leading to the notorious “modality imbalance” problem and counterproductive MML performance. To address the problem, some existing methods modulate the learning pace based on the fused modality, which is dominated by the better modality and eventually results in a limited improvement on the worse modal. To better exploit the features of multimodal, we propose Prototypical Modality Rebalance (PMR) to perform stimulation on the particular slow-learning modality without interference from other modalities. Specifically, we introduce the prototypes that represent general features for each class, to build the non-parametric classifiers for uni-modal performance evaluation. Then, we try to accelerate the slow-learning modality by enhancing its clustering toward prototypes. Furthermore, to alleviate the suppression from the dominant modality, we introduce a prototype-based entropy regularization term during the early training stage to prevent premature convergence. Besides, our method only relies on the representations of each modality and without restrictions from model structures and fusion methods, making it with great application potential for various scenarios. The source code is available here
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
https://github.com/fanyunfeng-bit/Modal-Imbalance-PMR.

        ----

        ## [1906] MMANet: Margin-Aware Distillation and Modality-Aware Regularization for Incomplete Multimodal Learning

        **Authors**: *Shicai Wei, Chunbo Luo, Yang Luo*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01919](https://doi.org/10.1109/CVPR52729.2023.01919)

        **Abstract**:

        Multimodal learning has shown great potentials in numerous scenes and attracts increasing interest recently. However, it often encounters the problem of missing modality data and thus suffers severe performance degradation in practice. To this end, we propose a general framework called MMANet to assist incomplete multimodal learning. It consists of three components: the deployment network used for inference, the teacher network transferring comprehensive multimodal information to the deployment network, and the regularization network guiding the deployment network to balance weak modality combinations. Specifically, we propose a novel margin-aware distillation (MAD) to assist the information transfer by weighing the sample contribution with the classification uncertainty. This encourages the deployment network to focus on the samples near decision boundaries and acquire the refined inter-class margin. Besides, we design a modality-aware regularization (MAR) algorithm to mine the weak modality combinations and guide the regularization network to calculate prediction loss for them. This forces the deployment network to improve its representation ability for the weak modality combinations adaptively. Finally, extensive experiments on multimodal classification and segmentation tasks demonstrate that our MMANet outperforms the state-of-the-art significantly. Code is available at: https://github.com/shicaiwei123/MMANet

        ----

        ## [1907] Feature Alignment and Uniformity for Test Time Adaptation

        **Authors**: *Shuai Wang, Daoan Zhang, Zipei Yan, Jianguo Zhang, Rui Li*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01920](https://doi.org/10.1109/CVPR52729.2023.01920)

        **Abstract**:

        Test time adaptation (TTA) aims to adapt deep neural networks when receiving out of distribution test domain samples. In this setting, the model can only access online unlabeled test samples and pretrained models on the training domains. We first address TTA as a feature revision problem due to the domain gap between source domains and target domains. After that, we follow the two measurements alignment and uniformity to discuss the test time feature revision. For test time feature uniformity, we propose a test time self-distillation strategy to guarantee the consistency of uniformity between representations of the current batch and all the previous batches. For test time feature alignment, we propose a memorized spatial local clustering strategy to align the representations among the neighborhood samples for the upcoming batch. To deal with the common noisy label problem, we propound the entropy and consistency filters to select and drop the possible noisy labels. To prove the scalability and efficacy of our method, we conduct experiments on four domain generalization bench marks and four medical image segmentation tasks with various backbones. Experiment results show that our method not only improves baseline stably but also outperforms existing state-of-the-art test time adaptation methods.

        ----

        ## [1908] Revisiting Prototypical Network for Cross Domain Few-Shot Learning

        **Authors**: *Fei Zhou, Peng Wang, Lei Zhang, Wei Wei, Yanning Zhang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01921](https://doi.org/10.1109/CVPR52729.2023.01921)

        **Abstract**:

        Prototypical Network is a popular few-shot solver that aims at establishing a feature metric generalizable to novel few-shot classification (FSC) tasks using deep neural networks. However, its performance drops dramatically when generalizing to the FSC tasks in new domains. In this study, we revisit this problem and argue that the devil lies in the simplicity bias pitfall in neural networks. In specific, the network tends to focus on some biased shortcut features (e.g., color, shape, etc.) that are exclusively sufficient to distinguish very few classes in the meta-training tasks within a pre-defined domain, but fail to generalize across domains as some desirable semantic features. To mitigate this problem, we propose a Local-global Distillation Prototypical Network (LDP-net). Different from the standard Prototypical Network, we establish a two-branch network to classify the query image and its random local crops, respectively. Then, knowledge distillation is conducted among these two branches to enforce their class affiliation consistency. The rationale behind is that since such global-local semantic relationship is expected to hold regardless of data domains, the local-global distillation is beneficial to exploit some cross-domain transferable semantic features for feature metric establishment. Moreover, such local-global semantic consistency is further enforced among different images of the same class to reduce the intra-class semantic variation of the resultant feature. In addition, we propose to update the local branch as Exponential Moving Average (EMA) over training episodes, which makes it possible to better distill cross-episode knowledge and further enhance the generalization performance. Experiments on eight cross-domain FSC benchmarks empirically clarify our argument and show the state-of-the-art results of LDP-net. Code is available in https://github.com/NWPUZhoufei/LDP-Net

        ----

        ## [1909] A Whac-A-Mole Dilemma: Shortcuts Come in Multiples Where Mitigating One Amplifies Others

        **Authors**: *Zhiheng Li, Ivan Evtimov, Albert Gordo, Caner Hazirbas, Tal Hassner, Cristian Canton-Ferrer, Chenliang Xu, Mark Ibrahim*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01922](https://doi.org/10.1109/CVPR52729.2023.01922)

        **Abstract**:

        Machine learning models have been found to learn shortcuts—unintended decision rules that are unable to generalize—undermining models' reliability. Previous works address this problem under the tenuous assumption that only a single shortcut exists in the training data. Real-world images are rife with multiple visual cues from background to texture. Key to advancing the reliability of vision systems is understanding whether existing methods can overcome multiple shortcuts or struggle in a Whac-A-Mole game, i.e., where mitigating one shortcut amplifies reliance on others. To address this shortcoming, we propose two benchmarks: 1) UrbanCars, a dataset with precisely controlled spurious cues, and 2) ImageNet-W, an evaluation set based on ImageNet for watermark, a shortcut we discovered affects nearly every modern vision model. Along with texture and background, ImageNet-W allows us to study multiple shortcuts emerging from training on natural images. We find computer vision models, including large foundation models—regardless of training set, architecture, and supervision—struggle when multiple shortcuts are present. Even methods explicitly designed to combat shortcuts struggle in a Whac-A-Mole dilemma. To tackle this challenge, we propose Last Layer Ensemble, a simple-yet-effective method to mitigate multiple shortcuts without Whac-A-Mole behavior. Our results surface multi-shortcut mitigation as an overlooked challenge critical to advancing the reliability of vision systems. The datasets and code are released: https://github.com/facebookresearch/Whac-A-Mole.

        ----

        ## [1910] Independent Component Alignment for Multi-Task Learning

        **Authors**: *Dmitry Senushkin, Nikolay Patakin, Arseny Kuznetsov, Anton Konushin*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01923](https://doi.org/10.1109/CVPR52729.2023.01923)

        **Abstract**:

        In a multi-task learning (MTL) setting, a single model is trained to tackle a diverse set of tasks Jointly. Despite rapid progress in the field, MTL remains challenging due to optimization issues such as conflicting and dominating gradients. In this work, we propose using a condition number of a linear system of gradients as a stability criterion of an MTL optimization. We theoretically demonstrate that a condition number reflects the afore-mentioned optimization issues. Accordingly, we present Aligned-MTL, a novel MTL optimization approach based on the proposed criterion, that eliminates instability in the training process by aligning the orthogonal components of the linear system of gradients. While many recent MTL approaches guaran-tee convergence to a minimum, task trade-offs cannot be specified in advance. In contrast, Aligned-MTL provably converges to an optimal point with pre-defined task-specific weights, which provides more control over the optimization result. Through experiments, we show that the proposed approach consistently improves performance on a diverse set of MTL benchmarks, including semantic and instance segmentation, depth estimation, surface normal estimation, and reinforcement learning. The source code is publicly available at https://github.com/SamsungLabs/MTL.

        ----

        ## [1911] MDL-NAS: A Joint Multi-domain Learning Framework for Vision Transformer

        **Authors**: *Shiguang Wang, Tao Xie, Jian Cheng, Xingcheng Zhang, Haijun Liu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01924](https://doi.org/10.1109/CVPR52729.2023.01924)

        **Abstract**:

        In this work, we introduce MDL-NAS, a unified frame-work that integrates multiple vision tasks into a manageable supernet and optimizes these tasks collectively under diverse dataset domains. MDL-NAS is storage-efficient since multiple models with a majority of shared parameters can be deposited into a single one. Technically, MDL-NAS constructs a coarse-to-fine search space, where the coarse search space offers various optimal architectures for different tasks while the fine search space provides fine-grained parameter sharing to tackle the inherent obstacles of multi-domain learning. In the fine search space, we suggest two parameter sharing policies, i.e., sequential sharing policy and mask sharing policy. Compared with previous works, such two sharing policies allow for the partial sharing and non-sharing of parameters at each layer of the network, hence attaining real fine-grained parameter sharing. Finally, we present a joint-subnet search algorithm that finds the optimal architecture and sharing parameters for each task within total resource constraints, challenging the traditional practice that downstream vision tasks are typically equipped with backbone networks designed for image classification. Experimentally, we demonstrate that MDL-NAS families fitted with non-hierarchical or hierarchical transformers deliver competitive performance for all tasks compared with state-of-the-art methods while maintaining efficient storage deployment and computation. We also demonstrate that MDL-NAS allows incremental learning and evades catastrophic forgetting when generalizing to a new task.

        ----

        ## [1912] MELTR: Meta Loss Transformer for Learning to Fine-tune Video Foundation Models

        **Authors**: *Dohwan Ko, Joonmyung Choi, Hyeong Kyu Choi, Kyoung-Woon On, Byungseok Roh, Hyunwoo J. Kim*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01925](https://doi.org/10.1109/CVPR52729.2023.01925)

        **Abstract**:

        Foundation models have shown outstanding performance and generalization capabilities across domains. Since most studies on foundation models mainly focus on the pretraining phase, a naive strategy to minimize a single task-specific loss is adopted for fine-tuning. However, such fine-tuning methods do not fully leverage other losses that are potentially beneficial for the target task. Therefore, we propose MEta Loss TRansformer (MELTR), a plug-in module that automatically and non-linearly combines various loss functions to aid learning the target task via auxiliary learning. We formulate the auxiliary learning as a bi-level optimization problem and present an efficient optimization algorithm based on Approximate Implicit Differentiation (AID). For evaluation, we apply our framework to various video foundation models (UniVL, Violet and All-in-one), and show significant performance gain on all four downstream tasks: text-to-video retrieval, video question answering, video captioning, and multimodal sentiment analysis. Our qualitative analyses demonstrate that MELTR adequately 'transforms' individual loss functions and 'melts' them into an effective unified loss. Code is available at https://github.com/mlvlab/MELTR.

        ----

        ## [1913] 1% VS 100%: Parameter-Efficient Low Rank Adapter for Dense Predictions

        **Authors**: *Dongshuo Yin, Yiran Yang, Zhechao Wang, Hongfeng Yu, Kaiwen Wei, Xian Sun*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01926](https://doi.org/10.1109/CVPR52729.2023.01926)

        **Abstract**:

        Fine-tuning large-scale pretrained vision models to downstream tasks is a standard technique for achieving state-of-the-art performance on computer vision benchmarks. However, fine-tuning the whole model with millions of parameters is inefficient as it requires storing a same-sized new model copy for each task. In this work, we propose LoRand, a method for fine-tuning large-scale vision models with a better tradeoff between task performance and the number of trainable parameters. LoRand generates tiny adapter structures with low-rank synthesis while keeping the original backbone parameters fixed, resulting in high parameter sharing. To demonstrate LoRand's effectiveness, we implement extensive experiments on object detection, semantic segmentation, and instance segmentation tasks. By only training a small percentage (1% to 3%) of the pretrained backbone parameters, LoRand achieves comparable performance to standard fine-tuning on COCO and ADE20K and outperforms fine-tuning in low-resource PASCAL VOC dataset.

        ----

        ## [1914] Rebalancing Batch Normalization for Exemplar-Based Class-Incremental Learning

        **Authors**: *Sungmin Cha, Sungjun Cho, Dasol Hwang, Sunwon Hong, Moontae Lee, Taesup Moon*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01927](https://doi.org/10.1109/CVPR52729.2023.01927)

        **Abstract**:

        Batch Normalization (BN) and its variants has been extensively studied for neural nets in various computer vision tasks, but relatively little work has been dedicated to studying the effect of BN in continual learning. To that end, we develop a new update patch for BN, particularly tailored for the exemplar-based class-incremental learning (CIL). The main issue of BN in CIL is the imbalance of training data between current and past tasks in a mini-batch, which makes the empirical mean and variance as well as the learnable affine transformation parameters of BN heavily biased toward the current task - contributing to the forgetting of past tasks. While one of the recent BN variants has been developed for” online” CIL; in which the training is done with a single epoch, we show that their method does not necessarily bring gains for “offline” CIL, in which a model is trained with multiple epochs on the imbalanced training data. The main reason for the ineffectiveness of their method lies in not fully addressing the data imbalance issue, especially in computing the gradients for learning the affine transformation parameters of BN. Accordingly, our new hyperparameter-free variant, dubbed as Task-Balanced BN (TBBN), is proposed to more correctly resolve the imbalance issue by making a horizontally-concatenated task-balanced batch using both reshape and repeat operations during training. Based on our experiments on class incremental learning of CIFAR-100, ImageNet-100, and five dissimilar task datasets, we demonstrate that our TBBN, which works exactly the same as the vanilla BN in the inference time, is easily applicable to most existing exemplar-based offline CIL algorithms and consistently outperforms other BN variants.

        ----

        ## [1915] Partial Network Cloning

        **Authors**: *Jingwen Ye, Songhua Liu, Xinchao Wang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01928](https://doi.org/10.1109/CVPR52729.2023.01928)

        **Abstract**:

        In this paper, we study a novel task that enables partial knowledge transfer from pre-trained models, which we term as Partial Network Cloning (PNC). Unlike prior methods that update all or at least part of the parameters in the target network throughout the knowledge transfer process, PNC conducts partial parametric “cloning” from a source network and then injects the cloned module to the target, without modifying its parameters. Thanks to the transferred module, the target network is expected to gain additional functionality, such as inference on new classes; whenever needed, the cloned module can be readily removed from the target, with its original parameters and competence kept intact. Specifically, we introduce an innovative learning scheme that allows us to identify simultaneously the component to be cloned from the source and the position to be inserted within the target network, so as to ensure the optimal performance. Experimental results on several datasets demonstrate that, our method yields a significant improvement of 5% in accuracy and 50% in locality when compared with parameter-tuning based methods. Our code is available at https://github.com/JngwenYe/PNCloning.

        ----

        ## [1916] ERM-KTP: Knowledge-Level Machine Unlearning via Knowledge Transfer

        **Authors**: *Shen Lin, Xiaoyu Zhang, Chenyang Chen, Xiaofeng Chen, Willy Susilo*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01929](https://doi.org/10.1109/CVPR52729.2023.01929)

        **Abstract**:

        Machine unlearning can fortify the privacy and security of machine learning applications. Unfortunately, the exact unlearning approaches are inefficient, and the approximate unlearning approaches are unsuitable for complicated CNNs. Moreover, the approximate approaches have serious security flaws because even unlearning completely different data points can produce the same contribution estimation as unlearning the target data points. To address the above problems, we try to define machine unlearning from the knowledge perspective, and we propose a knowledge-level machine unlearning method, namely ERM-KTP. Specifically, we propose an entanglement-reduced mask (ERM) structure to reduce the knowledge entanglement among classes during the training phase. When receiving the un-learning requests, we transfer the knowledge of the non-target data points from the original model to the unlearned model and meanwhile prohibit the knowledge of the target data points via our proposed knowledge transfer and prohibition (KTP) method. Finally, we will get the un-learned model as the result and delete the original model to accomplish the unlearning process. Especially, our proposed ERM-KTP is an interpretable unlearning method because the ERM structure and the crafted masks in KTP can explicitly explain the operation and the effect of un-learning data points. Extensive experiments demonstrate the effectiveness, efficiency, high fidelity, and scalability of the ERM-KTP unlearning method. Code is available at https://github.com/RUIYUN-ML/ERM-KTP

        ----

        ## [1917] Rethinking Feature-based Knowledge Distillation for Face Recognition

        **Authors**: *Jingzhi Li, Zidong Guo, Hui Li, Seungju Han, Ji-Won Baek, Min Yang, Ran Yang, Sungjoo Suh*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01930](https://doi.org/10.1109/CVPR52729.2023.01930)

        **Abstract**:

        With the continual expansion of face datasets, feature-based distillation prevails for large-scale face recognition. In this work, we attempt to remove identity supervision in student training, to spare the GPU memory from saving massive class centers. However, this naive removal leads to inferior distillation result. We carefully inspect the performance degradation from the perspective of intrinsic dimension, and argue that the gap in intrinsic dimension, namely the intrinsic gap, is intimately connected to the infamous capacity gap problem. By constraining the teacher's search space with reverse distillation, we narrow the intrinsic gap and unleash the potential of feature-only distillation. Remarkably, the proposed reverse distillation creates universally student-friendly teacher that demonstrates outstanding student improvement. We further enhance its effectiveness by designing a student proxy to better bridge the intrinsic gap. As a result, the proposed method surpasses state-of-the-art distillation techniques with identity supervision on various face recognition benchmarks, and the improvements are consistent across different teacher-student pairs.

        ----

        ## [1918] Regularizing Second-Order Influences for Continual Learning

        **Authors**: *Zhicheng Sun, Yadong Mu, Gang Hua*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01931](https://doi.org/10.1109/CVPR52729.2023.01931)

        **Abstract**:

        Continual learning aims to learn on non-stationary data streams without catastrophically forgetting previous knowledge. Prevalent replay-based methods address this challenge by rehearsing on a small buffer holding the seen data, for which a delicate sample selection strategy is required. However, existing selection schemes typically seek only to maximize the utility of the ongoing selection, overlooking the interference between successive rounds of selection. Motivated by this, we dissect the interaction of sequential selection steps within a framework built on influence functions. We manage to identify a new class of second-order influences that will gradually amplify incidental bias in the replay buffer and compromise the selection process. To regularize the second-order effects, a novel selection objective is proposed, which also has clear connections to two widely adopted criteria. Furthermore, we present an efficient implementation for optimizing the proposed criterion. Experiments on multiple continual learning benchmarks demonstrate the advantage of our approach over state-of-the-art methods. Code is available at https://github.com/feifeiobama/InfluenceCL.

        ----

        ## [1919] Generalization Matters: Loss Minima Flattening via Parameter Hybridization for Efficient Online Knowledge Distillation

        **Authors**: *Tianli Zhang, Mengqi Xue, Jiangtao Zhang, Haofei Zhang, Yu Wang, Lechao Cheng, Jie Song, Mingli Song*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01932](https://doi.org/10.1109/CVPR52729.2023.01932)

        **Abstract**:

        Most existing online knowledge distillation (OKD) techniques typically require sophisticated modules to produce diverse knowledge for improving students' generalization ability. In this paper, we strive to fully utilize multi-model settings instead of well-designed modules to achieve a distillation effect with excellent generalization performance. Generally, model generalization can be reflected in the flatness of the loss landscape. Since averaging parameters of multiple models can find flatter minima, we are inspired to extend the process to the sampled convex combinations of multi-student models in OKD. Specifically, by linearly weighting students' parameters in each training batch, we construct a Hybrid-Weight Model (HWM) to represent the parameters surrounding involved students. The supervision loss of HWM can estimate the landscape's curvature of the whole region around students to measure the generalization explicitly. Hence we integrate HWM's loss into students' training and propose a novel OKD framework via parameter hybridization (OKDPH) to promote flatter minima and obtain robust solutions. Considering the redundancy of parameters could lead to the collapse of HWM, we further introduce a fusion operation to keep the high similarity of students. Compared to the state-of-the-art (SOTA) OKD methods and SOTA methods of seeking flat minima, our OKDPH achieves higher performance with fewer parameters, benefiting OKD with lightweight and robust characteristics. Our code is publicly available at https://github.com/tianlizhang/OKDPH.

        ----

        ## [1920] Decoupling Learning and Remembering: a Bilevel Memory Framework with Knowledge Projection for Task-Incremental Learning

        **Authors**: *Wenju Sun, Qingyong Li, Jing Zhang, Wen Wang, Yangli-ao Geng*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01933](https://doi.org/10.1109/CVPR52729.2023.01933)

        **Abstract**:

        The dilemma between plasticity and stability arises as a common challenge for incremental learning. In contrast, the human memory system is able to remedy this dilemma owing to its multilevel memory structure, which motivates us to propose a Bilevel Memory system with Knowledge Projection (BMKP) for incremental learning. BMKP decouples the functions of learning and remembering via a bilevel-memory design: a working memory responsible for adaptively model learning, to ensure plasticity; a long-term memory in charge of enduringly storing the knowledge incorporated within the learned model, to guarantee stability. However, an emerging issue is how to extract the learned knowledge from the working memory and assimilate it into the long-term memory. To approach this issue, we reveal that the parameters learned by the working memory are actually residing in a redundant high-dimensional space, and the knowledge incorporated in the model can have a quite compact representation under a group of pattern basis shared by all incremental learning tasks. Therefore, we propose a knowledge projection process to adaptively maintain the shared basis, with which the loosely organized model knowledge of working memory is projected into the compact representation to be remembered in the long-term memory. We evaluate BMKP on CIFAR-10, CIFAR-100, and Tiny-ImageNet. The experimental results show that BMKP achieves state-of-the-art performance with lower memory usage
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
The code is available at https://github.com/SunWenJu123/BMKP.

        ----

        ## [1921] On the Stability-Plasticity Dilemma of Class-Incremental Learning

        **Authors**: *Dongwan Kim, Bohyung Han*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01934](https://doi.org/10.1109/CVPR52729.2023.01934)

        **Abstract**:

        A primary goal of class-incremental learning is to strike a balance between stability and plasticity, where models should be both stable enough to retain knowledge learned from previously seen classes, and plastic enough to learn concepts from new classes. While previous works demonstrate strong performance on class-incremental benchmarks, it is not clear whether their success comes from the models being stable, plastic, or a mixture of both. This paper aims to shed light on how effectively recent class-incremental learning algorithms address the stability-plasticity trade-off. We establish analytical tools that measure the stability and plasticity of feature representations, and employ such tools to investigate models trained with various algorithms on large-scale class-incremental benchmarks. Surprisingly, we find that the majority of class-incremental learning algorithms heavily favor stability over plasticity, to the extent that the feature extractor of a model trained on the initial set of classes is no less effective than that of the final incremental model. Our observations not only inspire two simple algorithms that highlight the importance of feature representation analysis, but also suggest that class-incremental learning approaches, in general, should strive for better feature representation learning.

        ----

        ## [1922] Simulated Annealing in Early Layers Leads to Better Generalization

        **Authors**: *AmirMohammad Sarfi, Zahra Karimpour, Muawiz Chaudhary, Nasir Mohammad Khalid, Mirco Ravanelli, Sudhir Mudur, Eugene Belilovsky*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01935](https://doi.org/10.1109/CVPR52729.2023.01935)

        **Abstract**:

        Recently, a number of iterative learning methods have been introduced to improve generalization. These typically rely on training for longer periods of time in exchange for improved generalization. LLF (later-layer-forgetting) is a state-of-the-art method in this category. It strengthens learning in early layers by periodically re-initializing the last few layers of the network. Our principal innovation in this work is to use Simulated annealing in EArly Layers (SEAL) of the network in place of re-initialization of later layers. Essentially, later layers go through the normal gradient descent process, while the early layers go through short stints of gradient ascent followed by gradient descent. Extensive experiments on the popular Tiny-ImageNet dataset benchmark and a series of transfer learning and few-shot learning tasks show that we outperform LLF by a significant margin. We further show that, compared to normal training, LLF features, although improving on the target task, degrade the transfer learning performance across all datasets we explored. In comparison, our method outperforms LLF across the same target datasets by a large margin. We also show that the prediction depth of our method is significantly lower than that of LLF and normal training, indicating on average better prediction performance. 
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
The code to reproduce our results is publicly available at: https://github.com/amiiir-sarfi/SEAL

        ----

        ## [1923] Frustratingly Easy Regularization on Representation Can Boost Deep Reinforcement Learning

        **Authors**: *Qiang He, Huangyuan Su, Jieyu Zhang, Xinwen Hou*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01936](https://doi.org/10.1109/CVPR52729.2023.01936)

        **Abstract**:

        Deep reinforcement learning (DRL) gives the promise that an agent learns good policy from high-dimensional information, whereas representation learning removes irrelevant and redundant information and retains pertinent information. In this work, we demonstrate that the learned representation of the Q-network and its target Q-network should, in theory, satisfy a favorable distinguishable representation property. Specifically, there exists an upper bound on the representation similarity of the value functions of two adjacent time steps in a typical DRL setting. However, through illustrative experiments, we show that the learned DRL agent may violate this property and lead to a sub-optimal policy. Therefore, we propose a simple yet effective regularizer called P_olicy E_valuation with E_asy Regularization on Representation (PEER), which aims to maintain the distinguishable representation property via explicit regularization on internal representations. And we provide the convergence rate guarantee of PEER. Implementing PEER requires only one line of code. Our experiments demonstrate that incorporating PEER into DRL can significantly improve performance and sample efficiency. Comprehensive experiments show that PEER achieves state-of-the-art performance on all 4 environments on PyBullet, 9 out of 12 tasks on DM-Control, and 19 out of 26 games on Atari. To the best of our knowledge, PEER is the first work to study the inherent representation property of Q-network and its target. Our code is available at https://sites.google.com/view/peer-cvpr2023/.

        ----

        ## [1924] Tunable Convolutions with Parametric Multi-Loss Optimization

        **Authors**: *Matteo Maggioni, Thomas Tanay, Francesca Babiloni, Steven McDonagh, Ales Leonardis*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01937](https://doi.org/10.1109/CVPR52729.2023.01937)

        **Abstract**:

        Behavior of neural networks is irremediably determined by the specific loss and data used during training. However it is often desirable to tune the model at inference time based on external factors such as preferences of the user or dynamic characteristics of the data. This is especially important to balance the perception-distortion trade-off of ill-posed image-to-image translation tasks. In this work, we propose to optimize a parametric tunable convolutional layer, which includes a number of different kernels, using a parametric multi-loss, which includes an equal number of objectives. Our key insight is to use a shared set of parameters to dynamically interpolate both the objectives and the kernels. During training, these parameters are sampled at random to explicitly optimize all possible combinations of objectives and consequently disentangle their effect into the corresponding kernels. During inference, these parameters become interactive inputs of the model hence enabling reliable and consistent control over the model behavior. Extensive experimental results demonstrate that our tunable convolutions effectively work as a drop-in replacement for traditional convolutions in existing neural networks at virtually no extra computational cost, outperforming state-of-the-art control strategies in a wide range of applications; including image denoising, deblurring, super-resolution, and style transfer.

        ----

        ## [1925] Re-basin via implicit Sinkhorn differentiation

        **Authors**: *Fidel A. Guerrero-Peña, Heitor Rapela Medeiros, Thomas Dubail, Masih Aminbeidokhti, Eric Granger, Marco Pedersoli*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01938](https://doi.org/10.1109/CVPR52729.2023.01938)

        **Abstract**:

        The recent emergence of new algorithms for permuting models into functionally equivalent regions of the solution space has shed some light on the complexity of error surfaces and some promising properties like mode connectivity. However, finding the permutation that minimizes some ob-jectives is challenging, and current optimization techniques are not differentiable, which makes it difficult to integrate into a gradient-based optimization, and often leads to sub-optimal solutions. In this paper, we propose a Sinkhorn re-basin network with the ability to obtain the transportation plan that better suits a given objective. Unlike the current state-of-art, our method is differentiable and, there-fore, easy to adapt to any task within the deep learning do-main. Furthermore, we show the advantage of our re-basin method by proposing a new cost function that allows per-forming incremental learning by exploiting the linear mode connectivity property. The benefit of our method is compared against similar approaches from the literature under several conditions for both optimal transport and linear mode connectivity. The effectiveness of our continual learning method based on re-basin is also shown for several common benchmark datasets, providing experimental results that are competitive with the state-of-art. The source code is provided at https://github.com/jagp/sinkhorn-rebasin.

        ----

        ## [1926] Gradient Norm Aware Minimization Seeks First-Order Flatness and Improves Generalization

        **Authors**: *Xingxuan Zhang, Renzhe Xu, Han Yu, Hao Zou, Peng Cui*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01939](https://doi.org/10.1109/CVPR52729.2023.01939)

        **Abstract**:

        Recently, flat minima are proven to be effective for improving generalization and sharpness-aware minimization (SAM) achieves state-of-the-art performance. Yet the current definition of flatness discussed in SAM and its follow-ups are limited to the zeroth-order flatness (i.e., the worst-case loss within a perturbation radius). We show that the zeroth-order flatness can be insufficient to discriminate minima with low generalization error from those with high generalization error both when there is a single minimum or multiple minima within the given perturbation radius. Thus we present first-order flatness, a stronger measure of flatness focusing on the maximal gradient norm within a perturbation radius which bounds both the maximal eigenvalue of Hessian at local minima and the regularization function of SAM. We also present a novel training procedure named Gradient norm Aware Minimization (GAM) to seek minima with uniformly small curvature across all directions. Experimental results show that GAM improves the generalization of models trained with current optimizers such as SGD and Adam W on various datasets and networks. Furthermore, we show that GAM can help SAM find flatter minima and achieve better generalization. The code is available at https://github.com/xxgege/GAM.

        ----

        ## [1927] AstroNet: When Astrocyte Meets Artificial Neural Network

        **Authors**: *Mengqiao Han, Liyuan Pan, Xiabi Liu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01940](https://doi.org/10.1109/CVPR52729.2023.01940)

        **Abstract**:

        Network structure learning aims to optimize network architectures and make them more efficient without compromising performance. In this paper, we first study the astrocytes, a new mechanism to regulate connections in the classic M-P neuron. Then, with the astrocytes, we propose an AstroNet that can adaptively optimize neuron connections and therefore achieves structure learning to achieve higher accuracy and efficiency. AstroNet is based on our built Astrocyte-Neuron model, with a temporal regulation mechanism and a global connection mechanism, which is inspired by the bidirectional communication property of astrocytes. With the model, the proposed AstroNet uses a neural network (NN) for performing tasks, and an astrocyte network (AN) to continuously optimize the connections of NN, i.e., assigning weight to the neuron units in the NN adaptively. Experiments on the classification task demonstrate that our AstroNet can efficiently optimize the network structure while achieving state-of-the-art (SOTA) accuracy.

        ----

        ## [1928] Network Expansion For Practical Training Acceleration

        **Authors**: *Ning Ding, Yehui Tang, Kai Han, Chao Xu, Yunhe Wang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01941](https://doi.org/10.1109/CVPR52729.2023.01941)

        **Abstract**:

        Recently, the sizes of deep neural networks and training datasets both increase drastically to pursue better performance in a practical sense. With the prevalence of transformer-based models in vision tasks, even more pressure is laid on the GPU platforms to train these heavy models, which consumes a large amount of time and computing resources as well. Therefore, it's crucial to accelerate the training process of deep neural networks. In this paper, we propose a general network expansion method to reduce the practical time cost of the model training process. Specifically, we utilize both width- and depth-level sparsity of dense models to accelerate the training of deep neural networks. Firstly, we pick a sparse sub-network from the original dense model by reducing the number of parameters as the starting point of training. Then the sparse architecture will gradually expand during the training procedure and finally grow into a dense one. We design different expanding strategies to grow CNNs and ViTs respectively, due to the great heterogeneity in between the two architectures. Our method can be easily integrated into popular deep learning frameworks, which saves considerable training time and hardware resources. Extensive experiments show that our acceleration method can significantly speed up the training process of modern vision models on general GPU devices with negligible performance drop (e.g. 1.42× faster for ResNet-101 and 1.34× faster for DeiT-base on ImageNet-1k). The code is available at https://github.com/huawei-noah/Efficient-Computing/tree/master/TrainingAcceleration/NetworkExpansion and https://gitee.com/mindspore/hub/blob/master/mshub_res/assets/noah-cvlab/gpu/1.8/networkexpansion_v1.0_imagenet2012.md

        ----

        ## [1929] Defining and Quantifying the Emergence of Sparse Concepts in DNNs

        **Authors**: *Jie Ren, Mingjie Li, Qirui Chen, Huiqi Deng, Quanshi Zhang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01942](https://doi.org/10.1109/CVPR52729.2023.01942)

        **Abstract**:

        This paper aims to illustrate the concept-emerging phenomenon in a trained DNN. Specifically, we find that the inference score of a DNN can be disentangled into the effects of a few interactive concepts. These concepts can be understood as causal patterns in a sparse, symbolic causal graph, which explains the DNN. The faithfulness of using such a causal graph to explain the DNN is theoretically guaranteed, because we prove that the causal graph can well mimic the DNN's outputs on an exponential number of different masked samples. Besides, such a causal graph can be further simplified and re-written as an And-Or graph (AOG), without losing much explanation accuracy. The code is released at https://github.com/sjtu-xai-lab/aog.

        ----

        ## [1930] Samples with Low Loss Curvature Improve Data Efficiency

        **Authors**: *Isha Garg, Kaushik Roy*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01943](https://doi.org/10.1109/CVPR52729.2023.01943)

        **Abstract**:

        In this paper, we study the second order properties of the loss of trained deep neural networks with respect to the training data points to understand the curvature of the loss surface in the vicinity of these points. We find that there is an unexpected concentration of samples with very low curvature. We note that these low curvature samples are largely consistent across completely different architectures, and identifiable in the early epochs of training. We show that the curvature relates to the ‘cleanliness’ of the data points, with low curvatures samples corresponding to clean, higher clarity samples, representative of their category. Alternatively, high curvature samples are often occluded, have conflicting features and visually atypical of their category. Armed with this insight, we introduce SLo-Curves, a novel coreset identification and training algorithm. SLo-curves identifies the samples with low curvatures as being more data-efficient and trains on them with an additional regularizer that penalizes high curvature of the loss surface in their vicinity. We demonstrate the efficacy of SLo-Curves on CIFAR-10 and CIFAR-100 datasets, where it outperforms state of the art coreset selection methods at small coreset sizes by up to 9%. The identified coresets generalize across architectures, and hence can be pre-computed to generate condensed versions of datasets for use in downstream tasks. Code is available at https://github.com/ishagarg/SLo-Curves.

        ----

        ## [1931] Masked Images Are Counterfactual Samples for Robust Fine-Tuning

        **Authors**: *Yao Xiao, Ziyi Tang, Pengxu Wei, Cong Liu, Liang Lin*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01944](https://doi.org/10.1109/CVPR52729.2023.01944)

        **Abstract**:

        Deep learning models are challenged by the distribution shift between the training data and test data. Recently, the large models pre-trained on diverse data have demonstrated unprecedented robustness to various distribution shifts. However, fine-tuning these models can lead to a trade-off between in-distribution (ID) performance and out-of-distribution (OOD) robustness. Existing methods for tackling this trade-off do not explicitly address the OOD robustness problem. In this paper, based on causal analysis of the aforementioned problems, we propose a novel fine-tuning method, which uses masked images as counterfactual samples that help improve the robustness of the fine-tuning model. Specifically, we mask either the semantics-related or semantics-unrelated patches of the images based on class activation map to break the spurious correlation, and refill the masked patches with patches from other images. The resulting counterfactual samples are used in feature-based distillation with the pre-trained model. Extensive experiments verify that regularizing the fine-tuning with the proposed masked images can achieve a better trade-off between ID and OOD performance, surpassing previous methods on the OOD performance. Our code is available at https://github.com/Coxy7/robust-finetuning.

        ----

        ## [1932] Bias Mimicking: A Simple Sampling Approach for Bias Mitigation

        **Authors**: *Maan Qraitem, Kate Saenko, Bryan A. Plummer*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01945](https://doi.org/10.1109/CVPR52729.2023.01945)

        **Abstract**:

        Prior work has shown that Visual Recognition datasets frequently underrepresent bias groups 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$B$</tex>
 (e.g. Female) within class labels 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$Y$</tex>
 (e.g. Programmers). This dataset bias can lead to models that learn spurious correlations between class labels and bias groups such as age, gender, or race. Most recent methods that address this problem require significant architectural changes or additional loss functions requiring more hyper-parameter tuning. Alternatively, data sampling baselines from the class imbalance literature (e.g. Undersampling, Upweighting), which can often be implemented in a single line of code and often have no hyperparameters, offer a cheaper and more efficient solution. However, these methods suffer from significant shortcomings. For example, Undersampling drops a significant part of the input distribution per epoch while Oversampling repeats samples, causing overfitting. To address these shortcomings, we introduce a new class-conditioned sampling method: Bias Mimicking. The method is based on the observation that if a class 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$c$</tex>
 bias distribution, i.e. 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$P_{D}(B\vert Y=c)$</tex>
 is mimicked across every 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$c^{\prime}\neq c$</tex>
, then 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$Y$</tex>
 and 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$B$</tex>
 are statistically independent. Using this notion, BM, through a novel training procedure, ensures that the model is exposed to the entire distribution per epoch without repeating samples. Consequently, Bias Mimicking improves underrepresented groups' accuracy of sampling methods by 3% over four benchmarks while maintaining and sometimes improving performance over nonsampling methods. Code: https://github.com/mqraitem/Bias-Mimicking

        ----

        ## [1933] NoisyQuant: Noisy Bias-Enhanced Post-Training Activation Quantization for Vision Transformers

        **Authors**: *Yijiang Liu, Huanrui Yang, Zhen Dong, Kurt Keutzer, Li Du, Shanghang Zhang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01946](https://doi.org/10.1109/CVPR52729.2023.01946)

        **Abstract**:

        The complicated architecture and high training cost of vision transformers urge the exploration of post-training quantization. However, the heavy-tailed distribution of vision transformer activations hinders the effectiveness of previous post-training quantization methods, even with advanced quantizer designs. Instead of tuning the quantizer to better fit the complicated activation distribution, this paper proposes NoisyQuant, a quantizer-agnostic enhancement for the post-training activation quantization performance of vision transformers. We make a surprising theoretical discovery that for a given quantizer, adding a fixed Uniform noisy bias to the values being quantized can significantly reduce the quantization error under provable conditions. Building on the theoretical insight, NoisyQuant achieves the first success on actively altering the heavy-tailed activation distribution with additive noisy bias to fit a given quantizer. Extensive experiments show NoisyQuant largely improves the post-training quantization performance of vision transformer with minimal computation overhead. For instance, on linear uniform 6-bit activation quantization, NoisyQuant improves SOTA top-1 accuracy on ImageNet by up to 1.7%, 1.1% and 0.5% for ViT, DeiT, and Swin Transformer respectively, achieving on-par or even higher performance than previous nonlinear, mixed-precision quantization.

        ----

        ## [1934] Practical Network Acceleration with Tiny Sets

        **Authors**: *Guo-Hua Wang, Jianxin Wu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01947](https://doi.org/10.1109/CVPR52729.2023.01947)

        **Abstract**:

        Due to data privacy issues, accelerating networks with tiny training sets has become a critical need in practice. Previous methods mainly adopt filter-level pruning to accelerate networks with scarce training samples. In this paper, we reveal that dropping blocks is a fundamentally superior approach in this scenario. It enjoys a higher acceleration ratio and results in a better latency-accuracy performance under the few-shot setting. To choose which blocks to drop, we propose a new concept namely recoverability to measure the difficulty of recovering the compressed network. Our recoverability is efficient and effective for choosing which blocks to drop. Finally, we propose an algorithm named Practise to accelerate networks using only tiny sets of training images. Practise outperforms previous methods by a significant margin. For 22% latency reduction, Practise surpasses previous methods by on average 7% on ImageNet-1k. It also enjoys high generalization ability, working well under data-free or out-of-domain data settings, too. Our code is at https://github.com/DoctorKey/Practise.

        ----

        ## [1935] TeSLA: Test-Time Self-Learning With Automatic Adversarial Augmentation

        **Authors**: *Devavrat Tomar, Guillaume Vray, Behzad Bozorgtabar, Jean-Philippe Thiran*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01948](https://doi.org/10.1109/CVPR52729.2023.01948)

        **Abstract**:

        Most recent test-time adaptation methods focus on only classification tasks, use specialized network architectures, destroy model calibration or rely on lightweight information from the source domain. To tackle these issues, this paper proposes a novel Test-time Self-Learning method with automatic Adversarial augmentation dubbed TeSLA for adapting a pre-trained source model to the unlabeled streaming test data. In contrast to conventional self-learning methods based on cross-entropy, we introduce a new test-time loss function through an implicitly tight connection with the mutual information and online knowledge distillation. Furthermore, we propose a learnable efficient adversarial augmentation module that further enhances online knowledge distillation by simulating high entropy augmented images. Our method achieves state-of-the-art classification and segmentation results on several benchmarks and types of domain shifts, particularly on challenging measurement shifts of medical images. TeSLA also benefits from several desirable properties compared to competing methods in terms of calibration, uncertainty metrics, insensitivity to model architectures, and source training strategies, all supported by extensive ablations. Our code and models are available at https://github.com/devavratTomar/TeSLA.

        ----

        ## [1936] Discriminator-Cooperated Feature Map Distillation for GAN Compression

        **Authors**: *Tie Hu, Mingbao Lin, Lizhou You, Fei Chao, Rongrong Ji*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01949](https://doi.org/10.1109/CVPR52729.2023.01949)

        **Abstract**:

        Despite excellent performance in image generation, Generative Adversarial Networks (GANs) are notorious for its requirements of enormous storage and intensive computation. As an awesome “performance maker”, knowledge distillation is demonstrated to be particularly efficacious in exploring low-priced GANs. In this paper, we investigate the irreplaceability of teacher discriminator and present an inventive discriminator-cooperated distillation, abbreviated as DCD, towards refining better feature maps from the generator. In contrast to conventional pixel-to-pixel match methods in feature map distillation, our DCD utilizes teacher discriminator as a transformation to drive intermediate results of the student generator to be perceptually close to corresponding outputs of the teacher generator. Furthermore, in order to mitigate mode collapse in GAN compression, we construct a collaborative adversarial training paradigm where the teacher discriminator is from scratch established to co-train with student generator in company with our DCD. Our DCD shows superior results compared with existing GAN compression methods. For instance, after reducing over 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$40\times MACs$</tex>
 and 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$80\times$</tex>
 parameters of CycleGAN, we well decrease FID metric from 61.53 to 48.24 while the current SoTA method merely has 51.92. This work's source code has been made accessible at https://github.com/poopit/DCD-official.

        ----

        ## [1937] Private Image Generation with Dual-Purpose Auxiliary Classifier

        **Authors**: *Chen Chen, Daochang Liu, Siqi Ma, Surya Nepal, Chang Xu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01950](https://doi.org/10.1109/CVPR52729.2023.01950)

        **Abstract**:

        Privacy-preserving image generation has been important for segments such as medical domains that have sensitive and limited data. The benefits of guaranteed privacy come at the costs of generated images' quality and utility due to the privacy budget constraints. The utility is currently measured by the gen2real accuracy (g2r%), i.e., the accuracy on real data of a downstream classifier trained using generated data. However, apart from this standard utility, we identify the “reversed utility” as another crucial aspect, which computes the accuracy on generated data of a classifier trained using real data, dubbed as real2gen accuracy (r2g%). Jointly considering these two views of utility, the standard and the reversed, could help the generation model better improve transferability between fake and real data. Therefore, we propose a novel private image generation method that incorporates a dual-purpose auxiliary classifier, which alternates between learning from real data and fake data, into the training of differentially private GANs. Additionally, our deliberate training strategies such as sequential training contributes to accelerating the generator's convergence and further boosting the performance upon exhausting the privacy budget. Our results achieve new state-of-the-arts over all metrics on three benchmarks: MNIST, Fashion-MNIST, and CelebA.

        ----

        ## [1938] ImageNet-E: Benchmarking Neural Network Robustness via Attribute Editing

        **Authors**: *Xiaodan Li, Yuefeng Chen, Yao Zhu, Shuhui Wang, Rong Zhang, Hui Xue*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01951](https://doi.org/10.1109/CVPR52729.2023.01951)

        **Abstract**:

        Recent studies have shown that higher accuracy on ImageNet usually leads to better robustness against different corruptions. Therefore, in this paper, instead of following the traditional research paradigm that investigates new out-of-distribution corruptions or perturbations deep models may encounter, we conduct model debugging in in-distribution data to explore which object attributes a model may be sensitive to. To achieve this goal, we create a toolkit for object editing with controls of backgrounds, sizes, positions, and directions, and create a rigorous benchmark named ImageNet-E(diting) for evaluating the image classifier robustness in terms of object attributes. With our ImageNet-E, we evaluate the performance of current deep learning models, including both convolutional neural networks and vision transformers. We find that most models are quite sensitive to attribute changes. A small change in the background can lead to an average of 9.23% drop on top-1 accuracy. We also evaluate some robust models including both adversarially trained models and other robust trained models and find that some models show worse robustness against attribute changes than vanilla models. Based on these findings, we discover ways to enhance attribute robustness with preprocessing, architecture designs, and training strategies. We hope this work can provide some insights to the community and open up a new avenue for research in robust computer vision. The code and dataset are available at https://github.com/alibaba/easyrobust.

        ----

        ## [1939] Masked Jigsaw Puzzle: A Versatile Position Embedding for Vision Transformers

        **Authors**: *Bin Ren, Yahui Liu, Yue Song, Wei Bi, Rita Cucchiara, Nicu Sebe, Wei Wang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01952](https://doi.org/10.1109/CVPR52729.2023.01952)

        **Abstract**:

        Position Embeddings (PEs), an arguably indispensable component in Vision Transformers (ViTs), have been shown to improve the performance of ViTs on many vision tasks. However, PEs have a potentially high risk of privacy leakage since the spatial information of the input patches is exposed. This caveat naturally raises a series of interesting questions about the impact of PEs on accuracy, privacy, prediction consistency, etc. To tackle these issues, we propose a Masked Jigsaw Puzzle (MJP) position embedding method. In particular, MJP first shuffles the selected patches via our block-wise random jigsaw puzzle shuffle algorithm, and their corresponding PEs are occluded. Meanwhile, for the non-occluded patches, the PEs remain the original ones but their spatial relation is strengthened via our dense absolute localization regressor. The experimental results reveal that 1) PEs explicitly encode the 2D spatial relationship and lead to severe privacy leakage problems under gradient inversion attack; 2) Training ViTs with the naively shuffled patches can alleviate the problem, but it harms the accuracy; 3) Under a certain shuffle ratio, the proposed MJP not only boosts the performance and robustness on large-scale datasets (i.e., ImageNet-1K and ImageNet-C, -A/O) but also improves the privacy preservation ability under typical gradient attacks by a large margin. The source code and trained models are available at https://github.com/yhlleo/MJP.

        ----

        ## [1940] A New Comprehensive Benchmark for Semi-supervised Video Anomaly Detection and Anticipation

        **Authors**: *Congqi Cao, Yue Lu, Peng Wang, Yanning Zhang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01953](https://doi.org/10.1109/CVPR52729.2023.01953)

        **Abstract**:

        Semi-supervised video anomaly detection (VAD) is a critical task in the intelligent surveillance system. However, an essential type of anomaly in VAD named scene-dependent anomaly has not received the attention of researchers. Moreover, there is no research investigating anomaly anticipation, a more significant task for preventing the occurrence of anomalous events. To this end, we propose a new comprehensive dataset, NWPU Campus, containing 43 scenes, 28 classes of abnormal events, and 16 hours of videos. At present, it is the largest semi-supervised VAD dataset with the largest number of scenes and classes of anomalies, the longest duration, and the only one considering the scene-dependent anomaly. Meanwhile, it is also the first dataset proposed for video anomaly anticipation. We further propose a novel model capable of detecting and anticipating anomalous events simultaneously. Compared with 7 outstanding VAD algorithms in recent years, our method can cope with scene-dependent anomaly detection and anomaly anticipation both well, achieving state-of-the-art performance on ShanghaiTech, CUHK Avenue, IITB Corridor and the newly proposed NWPU Campus datasets consistently. Our dataset and code is available at: https://campusvad.github.io.

        ----

        ## [1941] SimpleNet: A Simple Network for Image Anomaly Detection and Localization

        **Authors**: *Zhikang Liu, Yiming Zhou, Yuansheng Xu, Zilei Wang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01954](https://doi.org/10.1109/CVPR52729.2023.01954)

        **Abstract**:

        We propose a simple and application-friendly network (called SimpleNet) for detecting and localizing anoma-lies. SimpleNet consists of four components: (1) a pre-trained Feature Extractor that generates local features, (2) a shallow Feature Adapter that transfers local features to-wards target domain, (3) a simple Anomaly Feature Gener-ator that counterfeits anomaly features by adding Gaussian noise to normal features, and (4) a binary Anomaly Discriminator that distinguishes anomaly features from normal features. During inference, the Anomaly Feature Generator would be discarded. Our approach is based on three in-tuitions. First, transforming pre-trained features to target-oriented features helps avoid domain bias. Second, gen-erating synthetic anomalies in feature space is more effective, as defects may not have much commonality in the image space. Third, a simple discriminator is much efficient and practical. In spite of simplicity, SimpleNet outper-forms previous methods quantitatively and qualitatively. On The MVTec AD benchmark, SimpleNet achieves an anomaly detection AUROC of 99.6%, reducing the error by 55.5% compared to the next best performing model. Further-more, SimpleNet is faster than existing methods, with a high frame rate of 77 FPS on a 3080ti GPU. Additionally, SimpleNet demonstrates significant improvements in per-formance on the One-Class Novelty Detection task. Code: https://github.com/DonaldRR/SimpleNet.

        ----

        ## [1942] DaFKD: Domain-aware Federated Knowledge Distillation

        **Authors**: *Haozhao Wang, Yichen Li, Wenchao Xu, Ruixuan Li, Yufeng Zhan, Zhigang Zeng*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01955](https://doi.org/10.1109/CVPR52729.2023.01955)

        **Abstract**:

        Federated Distillation (FD) has recently attracted increasing attention for its efficiency in aggregating multiple diverse local models trained from statistically heterogeneous data of distributed clients. Existing FD methods generally treat these models equally by merely computing the average of their output soft predictions for some given input distillation sample, which does not take the diversity across all local models into account, thus leading to degraded performance of the aggregated model, especially when some local models learn little knowledge about the sample. In this paper, we propose a new perspective that treats the local data in each client as a specific domain and design a novel domain knowledge aware federated distillation method, dubbed DaFKD, that can discern the importance of each model to the distillation sample, and thus is able to optimize the ensemble of soft predictions from diverse models. Specifically, we employ a domain discriminator for each client, which is trained to identify the correlation factor between the sample and the corresponding domain. Then, to facilitate the training of the domain discriminator while saving communication costs, we propose sharing its partial parameters with the classification model. Extensive experiments on various datasets and settings show that the proposed method can improve the model accuracy by up to 6.02% compared to state-of-the-art baselines.

        ----

        ## [1943] Reliable and Interpretable Personalized Federated Learning

        **Authors**: *Zixuan Qin, Liu Yang, Qilong Wang, Yahong Han, Qinghua Hu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01956](https://doi.org/10.1109/CVPR52729.2023.01956)

        **Abstract**:

        Federated learning can coordinate multiple users to participate in data training while ensuring data privacy. The collaboration of multiple agents allows for a natural connection between federated learning and collective intelligence. When there are large differences in data distribution among clients, it is crucial for federated learning to design a reliable client selection strategy and an interpretable client communication framework to better utilize group knowledge. Herein, a reliable personalized federated learning approach, termed RIPFL, is proposed and fully interpreted from the perspective of social learning. RIPFL reliably selects and divides the clients involved in training such that each client can use different amounts of social information and more effectively communicate with other clients. Simultaneously, the method effectively integrates personal information with the social information generated by the global model from the perspective of Bayesian decision rules and evidence theory, enabling individuals to grow better with the help of collective wisdom. An interpretable federated learning mind is well scalable, and the experimental results indicate that the proposed method has superior robustness and accuracy than other state-of-the-art federated learning algorithms.

        ----

        ## [1944] Adaptive Channel Sparsity for Federated Learning under System Heterogeneity

        **Authors**: *Dongping Liao, Xitong Gao, Yiren Zhao, Chengzhong Xu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01957](https://doi.org/10.1109/CVPR52729.2023.01957)

        **Abstract**:

        Owing to the non-i.i.d. nature of client data, channel neurons in federated-learned models may specialize to distinct features for different clients. Yet, existing channel-sparse federated learning (FL) algorithms prescribe fixed sparsity strategies for client models, and may thus prevent clients from training channel neurons collaboratively. To minimize the impact of sparsity on FL convergence, we propose Flado to improve the alignment of client model update trajectories by tailoring the sparsities of individual neurons in each client. Empirical results show that while other sparse methods are surprisingly impactful to convergence, Flado can not only attain the highest task accuracies with unlimited budget across a range of datasets, but also significantly reduce the amount of floating-point operations (FLOPs) required for training more than by 10× under the same communications budget, and push the Pareto frontier of communication/computation trade-off notably further than competing FL algorithms.

        ----

        ## [1945] Bias-Eliminating Augmentation Learning for Debiased Federated Learning

        **Authors**: *Yuan-Yi Xu, Ci-Siang Lin, Yu-Chiang Frank Wang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01958](https://doi.org/10.1109/CVPR52729.2023.01958)

        **Abstract**:

        Learning models trained on biased datasets tend to observe correlations between categorical and undesirable features, which result in degraded performances. Most existing debiased learning models are designed for centralized machine learning, which cannot be directly applied to distributed settings like federated learning (FL), which collects data at distinct clients with privacy preserved. To tackle the challenging task of debiased federated learning, we present a novel FL framework of Bias-Eliminating Augmentation Learning (FedBEAL), which learns to deploy Bias-Eliminating Augmenters (BEA) for producing client-specific bias-conflicting samples at each client. Since the bias types or attributes are not known in advance, a unique learning strategy is presented to jointly train BEA with the proposed FL framework. Extensive image classification experiments on datasets with various bias types confirm the effectiveness and applicability of our FedBEAL, which performs favorably against state-of-the-art debiasing and FL methods for debiased FL.

        ----

        ## [1946] Instance-Aware Domain Generalization for Face Anti-Spoofing

        **Authors**: *Qianyu Zhou, Ke-Yue Zhang, Taiping Yao, Xuequan Lu, Ran Yi, Shouhong Ding, Lizhuang Ma*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01959](https://doi.org/10.1109/CVPR52729.2023.01959)

        **Abstract**:

        Face anti-spoofing (FAS) based on domain generalization (DG) has been recently studied to improve the generalization on unseen scenarios. Previous methods typically rely on domain labels to align the distribution of each domain for learning domain-invariant representations. However, artificial domain labels are coarse-grained and subjective, which cannot reflect real domain distributions accurately. Besides, such domain-aware methods focus on domain-level alignment, which is not fine-grained enough to ensure that learned representations are insensitive to domain styles. To address these issues, we propose a novel perspective for DG FAS that aligns features on the instance level without the need for domain labels. Specifically, Instance-Aware Domain Generalization framework is proposed to learn the generalizable feature by weakening the features' sensitivity to instance-specific styles. Concretely, we propose Asymmetric Instance Adaptive Whitening to adaptively eliminate the style-sensitive feature correlation, boosting the generalization. Moreover, Dynamic Kernel Generator and Categorical Style Assembly are proposed to first extract the instance-specific features and then generate the style-diversified features with large style shifts, respectively, further facilitating the learning of style-insensitive features. Extensive experiments and analysis demonstrate the superiority of our method over state-of-the-art competitors. Code will be publicly available at this link.

        ----

        ## [1947] Adversarially Masking Synthetic to Mimic Real: Adaptive Noise Injection for Point Cloud Segmentation Adaptation

        **Authors**: *Guangrui Li, Guoliang Kang, Xiaohan Wang, Yunchao Wei, Yi Yang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01960](https://doi.org/10.1109/CVPR52729.2023.01960)

        **Abstract**:

        This paper considers the synthetic-to-real adaptation of point cloud semantic segmentation, which aims to segment the real-world point clouds with only synthetic labels available. Contrary to synthetic data which is integral and clean, point clouds collected by real-world sensors typically contain unexpected and irregular noise because the sensors may be impacted by various environmental conditions. Consequently, the model trained on ideal synthetic data may fail to achieve satisfactory segmentation results on real data. Influenced by such noise, previous adversarial training methods, which are conventional for 2D adaptation tasks, become less effective. In this paper, we aim to mitigate the domain gap caused by target noise via learning to mask the source points during the adaptation procedure. To this end, we design a novel learnable masking module, which takes source features and 3D coordinates as inputs. We incorporate Gumbel-Softmax operation into the masking module so that it can generate binary masks and be trained end-to-end via gradient back-propagation. With the help of adversarial training, the masking module can learn to generate source masks to mimic the pattern of irregular target noise, thereby narrowing the domain gap. We name our method “Adversarial Masking” as adversarial training and learnable masking module depend on each other and cooperate with each other to mitigate the domain gap. Experiments on two synthetic-to-real adaptation benchmarks verify the effectiveness of the proposed method.

        ----

        ## [1948] Model Barrier: A Compact Un-Transferable Isolation Domain for Model Intellectual Property Protection

        **Authors**: *Lianyu Wang, Meng Wang, Daoqiang Zhang, Huazhu Fu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01961](https://doi.org/10.1109/CVPR52729.2023.01961)

        **Abstract**:

        As scientific and technological advancements result from human intellectual labor and computational costs, protecting model intellectual property (IP) has become increasingly important to encourage model creators and owners. Model IP protection involves preventing the use of well-trained models on unauthorized domains. To address this issue, we propose a novel approach called Compact Un-Transferable Isolation Domain (CUTI-domain), which acts as a barrier to block illegal transfers from authorized to unauthorized domains. Specifically, CUTI-domain blocks crossdomain transfers by highlighting the private style features of the authorized domain, leading to recognition failure on unauthorized domains with irrelevant private style features. Moreover, we provide two solutions for using CUTI-domain depending on whether the unauthorized domain is known or not: target-specified CUTI-domain and target-free CUTI-domain. Our comprehensive experimental results on four digit datasets, CIFAR10 & STL10, and VisDA-2017 dataset demonstrate that CUTI-domain can be easily implemented as a plug-and-play module with different backbones, providing an efficient solution for model IP protection.

        ----

        ## [1949] MEDIC: Remove Model Backdoors via Importance Driven Cloning

        **Authors**: *Qiuling Xu, Guanhong Tao, Jean Honorio, Yingqi Liu, Shengwei An, Guangyu Shen, Siyuan Cheng, Xiangyu Zhang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01962](https://doi.org/10.1109/CVPR52729.2023.01962)

        **Abstract**:

        We develop a novel method to remove injected backdoors in deep learning models. It works by cloning the benign behaviors of a trojaned model to a new model of the same structure. It trains the clone model from scratch on a very small subset of samples and aims to minimize a cloning loss that denotes the differences between the activations of important neurons across the two models. The set of important neurons varies for each input, depending on their magnitude of activations and their impact on the classification result. We theoretically show our method can better recover benign functions of the backdoor model. Meanwhile, we prove our method can be more effective in removing back-doors compared with fine-tuning. Our experiments show that our technique can effectively remove nine different types of backdoors with minor benign accuracy degradation, outper-forming the state-of-the-art backdoor removal techniques that are based on fine-tuning, knowledge distillation, and neuron pruning.
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>

        ----

        ## [1950] Progressive Backdoor Erasing via connecting Backdoor and Adversarial Attacks

        **Authors**: *Bingxu Mu, Zhenxing Niu, Le Wang, Xue Wang, Qiguang Mia, Rong Jin, Gang Hua*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01963](https://doi.org/10.1109/CVPR52729.2023.01963)

        **Abstract**:

        Deep neural networks (DNNs) are known to be vulnera-ble to both backdoor attacks as well as adversarial attacks. In the literature, these two types of attacks are commonly treated as distinct problems and solved separately, since they belong to training-time and inference-time attacks respectively. However, in this paper we find an intriguing connection between them: for a model planted with backdoors, we observe that its adversarial examples have similar behaviors as its triggered images, i.e., both activate the same subset of DNN neurons. It indicates that planting a back-door into a model will significantly affect the model's adversarial examples. Based on these observations, a novel Progressive Backdoor Erasing (PBE) algorithm is proposed to progressively purify the infected model by leveraging un-targeted adversarial attacks. Different from previous back-door defense methods, one significant advantage of our approach is that it can erase backdoor even when the clean extra dataset is unavailable. We empirically show that, against 5 state-of-the-art backdoor attacks, our PBE can effectively erase the backdoor without obvious performance degradation on clean samples and outperforms existing de-fense methods.

        ----

        ## [1951] Reinforcement Learning-Based Black-Box Model Inversion Attacks

        **Authors**: *Gyojin Han, Jaehyun Choi, Haeil Lee, Junmo Kim*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01964](https://doi.org/10.1109/CVPR52729.2023.01964)

        **Abstract**:

        Model inversion attacks are a type of privacy attack that reconstructs private data used to train a machine learning model, solely by accessing the model. Recently, white-box model inversion attacks leveraging Generative Adversarial Networks (GANs) to distill knowledge from public datasets have been receiving great attention because of their excellent attack performance. On the other hand, current black-box model inversion attacks that utilize GANs suffer from issues such as being unable to guarantee the completion of the attack process within a predetermined number of query accesses or achieve the same level of performance as white-box attacks. To overcome these limitations, we propose a reinforcement learning-based black-box model inversion attack. We formulate the latent space search as a Markov Decision Process (MDP) problem and solve it with reinforcement learning. Our method utilizes the confidence scores of the generated images to provide rewards to an agent. Finally, the private data can be reconstructed using the latent vectors found by the agent trained in the MDP. The experiment results on various datasets and models demonstrate that our attack successfully recovers the private information of the target model by achieving state-of-the-art attack performance. We emphasize the importance of studies on privacy-preserving machine learning by proposing a more advanced black-box model inversion attack.

        ----

        ## [1952] T-SEA: Transfer-Based Self-Ensemble Attack on Object Detection

        **Authors**: *Hao Huang, Ziyan Chen, Huanran Chen, Yongtao Wang, Kevin Zhang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01965](https://doi.org/10.1109/CVPR52729.2023.01965)

        **Abstract**:

        Compared to query-based black-box attacks, transferbased black-box attacks do not require any information of the attacked models, which ensures their secrecy. However, most existing transfer-based approaches rely on ensembling multiple models to boost the attack transferability, which is time- and resource-intensive, not to mention the difficulty of obtaining diverse models on the same task. To address this limitation, in this work, we focus on the single-model transfer-based black-box attack on object detection, utilizing only one model to achieve a high-transferability adversarial attack on multiple black-box detectors. Specifically, we first make observations on the patch optimization process of the existing method and propose an enhanced attack framework by slightly adjusting its training strategies. Then, we analogize patch optimization with regular model optimization, proposing a series of self-ensemble approaches on the input data, the attacked model, and the adversarial patch to efficiently make use of the limited information and prevent the patch from overfitting. The experimental results show that the proposed framework can be applied with multiple classical base attack methods (e.g., PGD and MIM) to greatly improve the black-box transferability of the well-optimized patch on multiple mainstream detectors, meanwhile boosting white-box performance. Our code is available at https://github.com/VDIGPKU/TSEA.

        ----

        ## [1953] Proximal Splitting Adversarial Attack for Semantic Segmentation

        **Authors**: *Jérôme Rony, Jean-Christophe Pesquet, Ismail Ben Ayed*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01966](https://doi.org/10.1109/CVPR52729.2023.01966)

        **Abstract**:

        Classification has been the focal point of research on adversarial attacks, but only a few works investigate methods suited to denser prediction tasks, such as semantic segmentation. The methods proposed in these works do not accurately solve the adversarial segmentation problem and, therefore, overestimate the size of the perturbations required to fool models. Here, we propose a white-box attack for these models based on a proximal splitting to produce adversarial perturbations with much smaller 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$\ell_{\infty}$</tex>
 norms. Our attack can handle large numbers of constraints within a nonconvex minimization framework via an Augmented Lagrangian approach, coupled with adaptive constraint scaling and masking strategies. We demonstrate that our attack significantly outperforms previously proposed ones, as well as classification attacks that we adapted for segmentation, providing a first comprehensive benchmark for this dense task.

        ----

        ## [1954] Towards Transferable Targeted Adversarial Examples

        **Authors**: *Zhibo Wang, Hongshan Yang, Yunhe Feng, Peng Sun, Hengchang Guo, Zhifei Zhang, Kui Ren*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01967](https://doi.org/10.1109/CVPR52729.2023.01967)

        **Abstract**:

        Transferability of adversarial examples is critical for black-box deep learning model attacks. While most existing studies focus on enhancing the transferability of untargeted adversarial attacks, few of them studied how to generate transferable targeted adversarial examples that can mislead models into predicting a specific class. Moreover, existing transferable targeted adversarial attacks usually fail to sufficiently characterize the target class distribution, thus suffering from limited transferability. In this paper, we propose the Transferable Targeted Adversarial Attack (TTAA), which can capture the distribution information of the target class from both label-wise and feature-wise perspectives, to generate highly transferable targeted adversarial examples. To this end, we design a generative adversarial training framework consisting of a generator to produce targeted adversarial examples, and feature-label dual discriminators to distinguish the generated adversarial examples from the target class images. Specifically, we design the label discriminator to guide the adversarial examples to learn label-related distribution information about the target class. Meanwhile, we design a feature discriminator, which extracts the feature-wise information with strong cross-model consistency, to enable the adversarial examples to learn the transferable distribution information. Furthermore, we introduce the random perturbation dropping to further enhance the transferability by augmenting the diversity of adversarial examples used in the training process. Experiments demonstrate that our method achieves excellent performance on the transferability of targeted adversarial examples. The targeted fooling rate reaches 95.13% when transferred from VGG-19 to DenseNet-121, which significantly outperforms the state-of-the-art methods.

        ----

        ## [1955] AGAIN: Adversarial Training with Attribution Span Enlargement and Hybrid Feature Fusion

        **Authors**: *Shenglin Yin, Kelu Yao, Sheng Shi, Yangzhou Du, Zhen Xiao*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01968](https://doi.org/10.1109/CVPR52729.2023.01968)

        **Abstract**:

        The deep neural networks (DNNs) trained by adversarial training (AT) usually suffered from significant robust generalization gap, i.e., DNNs achieve high training robustness but low test robustness. In this paper, we propose a generic method to boost the robust generalization of AT methods from the novel perspective of attribution span. To this end, compared with standard DNNs, we discover that the generalization gap of adversarially trained DNNs is caused by the smaller attribution span on the input image. In other words, adversarially trained DNNs tend to focus on specific visual concepts on training images, causing its limitation on test robustness. In this way, to enhance the robustness, we propose an effective method to enlarge the learned attribution span. Besides, we use hybrid feature statistics for feature fusion to enrich the diversity of features. Extensive experiments show that our method can effectively improves robustness of adversarially trained DNNs, outperforming previous SOTA methods. Furthermore, we provide a theoretical analysis of our method to prove its effectiveness.

        ----

        ## [1956] Generalist: Decoupling Natural and Robust Generalization

        **Authors**: *Hongjun Wang, Yisen Wang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01969](https://doi.org/10.1109/CVPR52729.2023.01969)

        **Abstract**:

        Deep neural networks obtained by standard training have been constantly plagued by adversarial examples. Although adversarial training demonstrates its capability to defend against adversarial examples, unfortunately, it leads to an inevitable drop in the natural generalization. To address the issue, we decouple the natural generalization and the robust generalization from joint training and formulate different training strategies for each one. Specifically, instead of minimizing a global loss on the expectation over these two generalization errors, we propose a hi-expert framework called Generalist where we simultaneously train base learners with task-aware strategies so that they can specialize in their own fields. The parameters of base learners are collected and combined to form a global learner at intervals during the training process. The global learner is then distributed to the base learners as initialized parameters for continued training. Theoretically, we prove that the risks of Generalist will get lower once the base learners are well trained. Extensive experiments verify the applicability of Generalist to achieve high accuracy on natural examples while maintaining considerable robustness to adversarial ones. Code is available at https://github.com/PKU-ML/Generalist.

        ----

        ## [1957] Cooperation or Competition: Avoiding Player Domination for Multi-Target Robustness via Adaptive Budgets

        **Authors**: *Yimu Wang, Dinghuai Zhang, Yihan Wu, Heng Huang, Hongyang Zhang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01970](https://doi.org/10.1109/CVPR52729.2023.01970)

        **Abstract**:

        Despite incredible advances, deep learning has been shown to be susceptible to adversarial attacks. Numerous approaches have been proposed to train robust networks both empirically and certifiably. However, most of them defend against only a single type of attack, while recent work takes steps forward in defending against multiple attacks. In this paper, to understand multi-target robustness, we view this problem as a bargaining game in which different players (adversaries) negotiate to reach an agreement on a joint direction of parameter updating. We identify a phenomenon named player domination in the bargaining game, namely that the existing max-based approaches, such as MAX and MSD, do not converge. Based on our theoretical analysis, we design a novel framework that adjusts the budgets of different adversaries to avoid any player dominance. Experiments on standard benchmarks show that employing the proposed framework to the existing approaches significantly advances multi-target robustness.

        ----

        ## [1958] Discrete Point-Wise Attack is Not Enough: Generalized Manifold Adversarial Attack for Face Recognition

        **Authors**: *Qian Li, Yuxiao Hu, Ye Liu, Dongxiao Zhang, Xin Jin, Yuntian Chen*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01971](https://doi.org/10.1109/CVPR52729.2023.01971)

        **Abstract**:

        Classical adversarial attacks for Face Recognition (FR) models typically generate discrete examples for target identity with a single state image. However, such paradigm of point-wise attack exhibits poor generalization against numerous unknown states of identity and can be easily defended. In this paper, by rethinking the inherent relationship between the face of target identity and its variants, we introduce a new pipeline of Generalized Manifold Adversarial Attack (GMAA)
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
https://github.com/tokaka22/GMAA to achieve a better attack performance by expanding the attack range. Specifically, this expansion lies on two aspects - GMAA not only expands the target to be attacked from one to many to encourage a good generalization ability for the generated adversarial examples, but it also expands the latter from discrete points to manifold by leveraging the domain knowledge that face expression change can be continuous, which enhances the attack effect as a data augmentation mechanism did. Moreover, we further design a dual supervision with local and global constraints as a minor contribution to improve the visual quality of the generated adversarial examples. We demonstrate the effectiveness of our method based on extensive experiments, and reveal that GMAA promises a semantic continuous adversarial space with a higher generalization ability and visual quality.

        ----

        ## [1959] RIATIG: Reliable and Imperceptible Adversarial Text-to-Image Generation with Natural Prompts

        **Authors**: *Han Liu, Yuhao Wu, Shixuan Zhai, Bo Yuan, Ning Zhang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01972](https://doi.org/10.1109/CVPR52729.2023.01972)

        **Abstract**:

        The field of text-to-image generation has made remarkable strides in creating high-fidelity and photorealistic images. As this technology gains popularity, there is a growing concern about its potential security risks. However, there has been limited exploration into the robustness of these models from an adversarial perspective. Existing research has primarily focused on untargeted settings, and lacks holistic consideration for reliability (attack success rate) and stealthiness (imperceptibility). In this paper, we propose RIATIG, a reliable and imperceptible adversarial attack against text-to-image models via inconspicuous examples. By formulating the example crafting as an optimization process and solving it using a genetic-based method, our proposed attack can generate imperceptible prompts for text-to-image generation models in a reliable way. Evaluation of six popular text-to-image generation models demonstrates the efficiency and stealthiness of our attack in both white-box and black-box settings. To allow the community to build on top of our findings, we've made the artifacts available
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
Code is available at: https://github.com/WUSTL-CSPL/RIATIG.

        ----

        ## [1960] CLIP2Protect: Protecting Facial Privacy Using Text-Guided Makeup via Adversarial Latent Search

        **Authors**: *Fahad Shamshad, Muzammal Naseer, Karthik Nandakumar*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01973](https://doi.org/10.1109/CVPR52729.2023.01973)

        **Abstract**:

        The success of deep learning based face recognition systems has given rise to serious privacy concerns due to their ability to enable unauthorized tracking of users in the digital world. Existing methods for enhancing privacy fail to generate “naturalistic” images that can protect facial privacy without compromising user experience. We propose a novel two-step approach for facial privacy protection that relies on finding adversarial latent codes in the low- dimensional manifold of a pretrained generative model. The first step inverts the given face image into the latent space and finetunes the generative model to achieve an accurate reconstruction of the given image from its latent code. This step produces a good initialization, aiding the generation of high-quality faces that resemble the given identity. Subsequently, user-defined makeup text prompts and identity- preserving regularization are used to guide the search for adversarial codes in the latent space. Extensive experiments demonstrate that faces generated by our approach have stronger black-box transferability with an absolute gain of 12.06% over the state-of-the-art facial privacy protection approach under the face verification task. Finally, we demonstrate the effectiveness of the proposed approach for commercial face recognition systems. Our code is available at https://github.com/fahadshamshad/Clip2Protect.

        ----

        ## [1961] TruFor: Leveraging All-Round Clues for Trustworthy Image Forgery Detection and Localization

        **Authors**: *Fabrizio Guillaro, Davide Cozzolino, Avneesh Sud, Nicholas Dufour, Luisa Verdoliva*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01974](https://doi.org/10.1109/CVPR52729.2023.01974)

        **Abstract**:

        In this paper we present TruFor, a forensic framework that can be applied to a large variety of image manipulation methods, from classic cheapfakes to more recent manipulations based on deep learning. We rely on the extraction of both high-level and low-level traces through a transformer-based fusion architecture that combines the RGB image and a learned noise-sensitive fingerprint. The latter learns to embed the artifacts related to the camera internal and external processing by training only on real data in a self-supervised manner. Forgeries are detected as deviations from the expected regular pattern that characterizes each pristine image. Looking for anomalies makes the approach able to robustly detect a variety of local manipulations, ensuring generalization. In addition to a pixel-level localization map and a whole-image integrity score, our approach outputs a reliability map that highlights areas where localization predictions may be error-prone. This is particularly important in forensic applications in order to reduce false alarms and allow for a large scale analysis. Extensive experiments on several datasets show that our method is able to reliably detect and localize both cheapfakes and deepfakes manipulations outperforming state-of-the-art works. Code is publicly available at https://grip-unina.github.io/TruFor/

        ----

        ## [1962] High-fidelity Event-Radiance Recovery via Transient Event Frequency

        **Authors**: *Jin Han, Yuta Asano, Boxin Shi, Yinqiang Zheng, Imari Sato*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01975](https://doi.org/10.1109/CVPR52729.2023.01975)

        **Abstract**:

        High-fidelity radiance recovery plays a crucial role in scene information reconstruction and understanding. Conventional cameras suffer from limited sensitivity in dynamic range, bit depth, and spectral response, etc. In this paper, we propose to use event cameras with bio-inspired silicon sensors, which are sensitive to radiance changes, to recover precise radiance values. We reveal that, under active lighting conditions, the transient frequency of event signals triggering linearly reflects the radiance value. We propose an innovative method to convert the high temporal resolution of event signals into precise radiance values. The precise radiance values yields several capabilities in image analysis. We demonstrate the feasibility of recovering radiance values solely from the transient event frequency (TEF) through multiple experiments.

        ----

        ## [1963] RobustNeRF: Ignoring Distractors with Robust Losses

        **Authors**: *Sara Sabour, Suhani Vora, Daniel Duckworth, Ivan Krasin, David J. Fleet, Andrea Tagliasacchi*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01976](https://doi.org/10.1109/CVPR52729.2023.01976)

        **Abstract**:

        Neural radiance fields (NeRF) excel at synthesizing new views given multi-view, calibrated images of a static scene. When scenes include distractors, which are not persistent during image capture (moving objects, lighting variations, shadows), artifacts appear as view-dependent effects or ‘floaters’. To cope with distractors, we advocate a form of robust estimation for NeRF training, modeling distractors in training data as outliers of an optimization problem. Our method successfully removes outliers from a scene and improves upon our baselines, on synthetic and real-world scenes. Our technique is simple to incorporate in modern NeRF frameworks, with few hyper-parameters. It does not assume a priori knowledge of the types of distractors, and is instead focused on the optimization problem rather than pre-processing or modeling transient objects. More results at https://robustnerf.github.io/public.

        ----

        ## [1964] NeRDi: Single-View NeRF Synthesis with Language-Guided Diffusion as General Image Priors

        **Authors**: *Congyue Deng, Chiyu Max Jiang, Charles R. Qi, Xinchen Yan, Yin Zhou, Leonidas J. Guibas, Dragomir Anguelov*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01977](https://doi.org/10.1109/CVPR52729.2023.01977)

        **Abstract**:

        2D-to-3D reconstruction is an ill-posed problem, yet humans are good at solving this problem due to their prior knowledge of the 3D world developed over years. Driven by this observation, we propose NeRDi, a single-view NeRF synthesis framework with general image priors from 2D diffusion models. Formulating single-view reconstruction as an image-conditioned 3D generation problem, we optimize the NeRF representations by minimizing a diffusion loss on its arbitrary view renderings with a pretrained image diffusion model under the input-view constraint. We leverage off-the-shelf vision-language models and introduce a two-section language guidance as conditioning inputs to the diffusion model. This is essentially helpful for improving multiview content coherence as it narrows down the general image prior conditioned on the semantic and visual features of the single-view input image. Additionally, we introduce a geometric loss based on estimated depth maps to regularize the underlying 3D geometry of the NeRF. Experimental results on the DTU MVS dataset show that our method can synthesize novel views with higher quality even compared to existing methods trained on this dataset. We also demonstrate our generalizability in zero-shot NeRF synthesis for in-the-wild images.

        ----

        ## [1965] GM-NeRF: Learning Generalizable Model-Based Neural Radiance Fields from Multi-View Images

        **Authors**: *Jianchuan Chen, Wentao Yi, Liqian Ma, Xu Jia, Huchuan Lu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01978](https://doi.org/10.1109/CVPR52729.2023.01978)

        **Abstract**:

        In this work, we focus on synthesizing high-fidelity novel view images for arbitrary human performers, given a set of sparse multi-view images. It is a challenging task due to the large variation among articulated body poses and heavy self-occlusions. To alleviate this, we introduce an effective generalizable framework Generalizable Model-based Neural Radiance Fields (GM-NeRF) to synthesize free-viewpoint images. Specifically, we propose a geometry-guided attention mechanism to register the appearance code from multi-view 2D images to a geometry proxy which can alleviate the misalignment between inaccurate geometry prior and pixel space. On top of that, we further conduct neural rendering and partial gradient back propagation for efficient perceptual supervision and improvement of the perceptual quality of synthesis. To evaluate our method, we conduct experiments on synthesized datasets THuman2.0 and Multi-garment, and real-world datasets Genebody and ZJUMocap. The results demonstrate that our approach outperforms state-of-the-art methods in terms of novel view synthesis and geometric reconstruction.

        ----

        ## [1966] MixNeRF: Modeling a Ray with Mixture Density for Novel View Synthesis from Sparse Inputs

        **Authors**: *Seunghyeon Seo, Donghoon Han, Yeonjin Chang, Nojun Kwak*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01979](https://doi.org/10.1109/CVPR52729.2023.01979)

        **Abstract**:

        Neural Radiance Field (NeRF) has broken new ground in the novel view synthesis due to its simple concept and state-of-the-art quality. However, it suffers from severe performance degradation unless trained with a dense set of images with different camera poses, which hinders its practical applications. Although previous methods addressing this problem achieved promising results, they relied heavily on the additional training resources, which goes against the philosophy of sparse-input novel-view synthesis pursuing the training efficiency. In this work, we propose MixNeRF, an effective training strategy for novel view synthesis from sparse inputs by modeling a ray with a mixture density model. Our MixNeRF estimates the joint distribution of RGB colors along the ray samples by modeling it with mixture of distributions. We also propose a new task of ray depth estimation as a useful training objective, which is highly correlated with 3D scene geometry. Moreover, we remodel the colors with regenerated blending weights based on the estimated ray depth and further improves the robustness for colors and viewpoints. Our MixNeRF outperforms other state-of-the-art methods in various standard benchmarks with superior efficiency of training and inference.

        ----

        ## [1967] SPIn-NeRF: Multiview Segmentation and Perceptual Inpainting with Neural Radiance Fields

        **Authors**: *Ashkan Mirzaei, Tristan Aumentado-Armstrong, Konstantinos G. Derpanis, Jonathan Kelly, Marcus A. Brubaker, Igor Gilitschenski, Alex Levinshtein*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01980](https://doi.org/10.1109/CVPR52729.2023.01980)

        **Abstract**:

        Neural Radiance Fields (NeRFs) have emerged as a popular approach for novel view synthesis. While NeRFs are quickly being adapted for a wider set of applications, intuitively editing NeRF scenes is still an open challenge. One important editing task is the removal of unwanted objects from a 3D scene, such that the replaced region is visually plausible and consistent with its context. We refer to this task as 3D inpainting. In 3D, solutions must be both consistent across multiple views and geometrically valid. In this paper, we propose a novel 3D inpainting method that addresses these challenges. Given a small set of posed images and sparse annotations in a single input image, our framework first rapidly obtains a 3D segmentation mask for a target object. Using the mask, a perceptual optimization-based approach is then introduced that leverages learned 2D image inpainters, distilling their information into 3D space, while ensuring view consistency. We also address the lack of a diverse benchmark for evaluating 3D scene inpainting methods by introducing a dataset comprised of challenging real-world scenes. In particular, our dataset contains views of the same scene with and without a target object, enabling more principled benchmarking of the 3D inpainting task. We first demonstrate the superiority of our approach on multiview segmentation, comparing to NeRF-based methods and 2D segmentation approaches. We then evaluate on the task of 3D inpainting, establishing state-of-the-art performance against other NeRF manipulation algorithms, as well as a strong 2D image inpainter baseline.

        ----

        ## [1968] Masked Wavelet Representation for Compact Neural Radiance Fields

        **Authors**: *Daniel Rho, Byeonghyeon Lee, Seungtae Nam, Joo Chan Lee, Jong Hwan Ko, Eunbyung Park*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01981](https://doi.org/10.1109/CVPR52729.2023.01981)

        **Abstract**:

        Neural radiance fields (NeRF) have demonstrated the potential of coordinate-based neural representation (neural fields or implicit neural representation) in neural rendering. However, using a multi-layer perceptron (MLP) to represent a 3D scene or object requires enormous computational resources and time. There have been recent studies on how to reduce these computational inefficiencies by using additional data structures, such as grids or trees. Despite the promising performance, the explicit data structure necessitates a substantial amount of memory. In this work, we present a method to reduce the size without compromising the advantages of having additional data structures. In detail, we propose using the wavelet transform on grid-based neural fields. Grid-based neural fields are for fast convergence, and the wavelet transform, whose efficiency has been demonstrated in high-performance standard codecs, is to improve the parameter efficiency of grids. Furthermore, in order to achieve a higher sparsity of grid coefficients while maintaining reconstruction quality, we present a novel trainable masking approach. Experimental results demonstrate that non-spatial grid coefficients, such as wavelet coefficients, are capable of attaining a higher level of sparsity than spatial grid coefficients, resulting in a more compact representation. With our proposed mask and compression pipeline, we achieved state-of-the-art performance within a memory budget of 2 MB. Our code is available at https://github.com/daniel03c1/masked_wavelet_nerf.

        ----

        ## [1969] PaletteNeRF: Palette-based Appearance Editing of Neural Radiance Fields

        **Authors**: *Zhengfei Kuang, Fujun Luan, Sai Bi, Zhixin Shu, Gordon Wetzstein, Kalyan Sunkavalli*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01982](https://doi.org/10.1109/CVPR52729.2023.01982)

        **Abstract**:

        Recent advances in neural radiance fields have enabled the high-fidelity 3D reconstruction of complex scenes for novel view synthesis. However, it remains underexplored how the appearance of such representations can be efficiently edited while maintaining photorealism. In this work, we present PaletteNeRF, a novel method for photorealistic appearance editing of neural radiance fields (NeRF) based on 3D color decomposition. Our method decom-poses the appearance of each 3D point into a linear combination of palette-based bases (i.e., 3D segmentations defined by a group of NeRF-type functions) that are shared across the scene. While our palette-based bases are view-independent, we also predict a view-dependent function to capture the color residual (e.g., specular shading). During training, we jointly optimize the basis functions and the color palettes, and we also introduce novel regulariz-ers to encourage the spatial coherence of the decomposition. Our method allows users to efficiently edit the appearance of the 3D scene by modifying the color palettes. We also extend our framework with compressed semantic features for semantic-aware appearance editing. We demonstrate that our technique is superior to baseline methods both quantitatively and qualitatively for appearance editing of complex real-world scenes. Our project page is https://palettenerf.github.io.

        ----

        ## [1970] SteerNeRF: Accelerating NeRF Rendering via Smooth Viewpoint Trajectory

        **Authors**: *Sicheng Li, Hao Li, Yue Wang, Yiyi Liao, Lu Yu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01983](https://doi.org/10.1109/CVPR52729.2023.01983)

        **Abstract**:

        Neural Radiance Fields (NeRF) have demonstrated superior novel view synthesis performance but are slow at rendering. To speed up the volume rendering process, many acceleration methods have been proposed at the cost of large memory consumption. To push the frontier of the efficiency-memory trade-off, we explore a new perspective to accelerate NeRF rendering, leveraging a key fact that the view-point change is usually smooth and continuous in interactive viewpoint control. This allows us to leverage the information of preceding viewpoints to reduce the number of rendered pixels as well as the number of sampled points along the ray of the remaining pixels. In our pipeline, a low-resolution feature map is rendered first by volume rendering, then a lightweight 2D neural renderer is applied to generate the output image at target resolution leveraging the features of preceding and current frames. We show that the proposed method can achieve competitive rendering quality while reducing the rendering time with little memory overhead, enabling 30FPS at 1080P image resolution with a low memory footprint.

        ----

        ## [1971] Transforming Radiance Field with Lipschitz Network for Photorealistic 3D Scene Stylization

        **Authors**: *Zicheng Zhang, Yinglu Liu, Congying Han, Yingwei Pan, Tiande Guo, Ting Yao*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01984](https://doi.org/10.1109/CVPR52729.2023.01984)

        **Abstract**:

        Recent advances in 3D scene representation and novel view synthesis have witnessed the rise of Neural Radiance Fields (NeRFs). Nevertheless, it is not trivial to exploit NeRF for the photorealistic 3D scene stylization task, which aims to generate visually consistent and photorealistic stylized scenes from novel views. Simply coupling NeRF with photorealistic style transfer (PST) will result in cross-view inconsistency and degradation of stylized view syntheses. Through a thorough analysis, we demonstrate that this non-trivial task can be simplified in a new light: When transforming the appearance representation of a pre-trained NeRF with Lipschitz mapping, the consistency and photorealism across source views will be seamlessly encoded into the syntheses. That motivates us to build a concise and flexible learning framework namely LipRF, which upgrades arbitrary 2D PST methods with Lipschitz mapping tailored for the 3D scene. Technically, LipRF first pre-trains a radiance field to reconstruct the 3D scene, and then emulates the style on each view by 2D PST as the prior to learn a Lipschitz network to stylize the pre-trained appearance. In view of that Lipschitz condition highly impacts the expressivity of the neural network, we devise an adaptive regularization to balance the reconstruction and stylization. A gradual gradient aggregation strategy is further introduced to optimize LipRF in a cost-efficient manner. We conduct extensive experiments to show the high quality and robust performance of LipRF on both photorealistic 3D stylization and object appearance editing.

        ----

        ## [1972] Occlusion-Free Scene Recovery via Neural Radiance Fields

        **Authors**: *Chengxuan Zhu, Renjie Wan, Yunkai Tang, Boxin Shi*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01985](https://doi.org/10.1109/CVPR52729.2023.01985)

        **Abstract**:

        Our everyday lives are filled with occlusions that we strive to see through. By aggregating desired background information from different viewpoints, we can easily eliminate such occlusions without any external occlusion-free supervision. Though several occlusion removal methods have been proposed to empower machine vision systems with such ability, their performances are still unsatisfactory due to reliance on external supervision. We propose a novel method for occlusion removal by directly building a mapping between position and viewing angles and the corresponding occlusion-free scene details leveraging Neural Radiance Fields (NeRF). We also develop an effective scheme to jointly optimize camera parameters and scene reconstruction when occlusions are present. An additional depth constraint is applied to supervise the entire optimizaion without labeled external data for training. The experimental results on existing and newly collected datasets validate the effectiveness of our method. Our project page: https://freebutuselesssoul.github.io/occnerf.

        ----

        ## [1973] TriVol: Point Cloud Rendering via Triple Volumes

        **Authors**: *Tao Hu, Xiaogang Xu, Ruihang Chu, Jiaya Jia*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01986](https://doi.org/10.1109/CVPR52729.2023.01986)

        **Abstract**:

        Existing learning-based methods for point cloud rendering adopt various 3D representations and feature querying mechanisms to alleviate the sparsity problem of point clouds. However, artifacts still appear in rendered images, due to the challenges in extracting continuous and discriminative 3D features from point clouds. In this paper, we present a dense while lightweight 3D representation, named TriVol, that can be combined with NeRF to render photo-realistic images from point clouds. Our TriVol consists of triple slim volumes, each of which is encoded from the point cloud. TriVol has two advantages. First, it fuses respective fields at different scales and thus extracts local and non-local features for discriminative representation. Second, since the volume size is greatly reduced, our 3D decoder can be efficiently inferred, allowing us to increase the resolution of the 3D space to render more point details. Extensive experiments on different benchmarks with varying kinds of scenes/objects demonstrate our framework's effectiveness compared with current approaches. Moreover, our framework has excellent generalization ability to render a category of scenes/objects without fine-tuning. The source code is available at https://github.com/dvlabresearch/TriVol.git.

        ----

        ## [1974] DyNCA: Real-Time Dynamic Texture Synthesis Using Neural Cellular Automata

        **Authors**: *Ehsan Pajouheshgar, Yitao Xu, Tong Zhang, Sabine Süsstrunk*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01987](https://doi.org/10.1109/CVPR52729.2023.01987)

        **Abstract**:

        Current Dynamic Texture Synthesis (DyTS) models can synthesize realistic videos. However, they require a slow iterative optimization process to synthesize a single fixed-size short video, and they do not offer any post-training control over the synthesis process. We propose Dynamic Neural Cellular Automata (DyNCA), a framework for real-time and controllable dynamic texture synthesis. Our method is built upon the recently introduced NCA models and can synthesize infinitely long and arbitrary-sized realistic video textures in real time. We quantitatively and qualitatively evaluate our model and show that our synthesized videos appear more realistic than the existing results. We improve the SOTA DyTS performance by 2 ~ 4 orders of magnitude. Moreover, our model offers several real-time video controls including motion speed, motion direction, and an editing brush tool. We exhibit our trained models in an online interactive demo that runs on local hardware and is accessible on personal computers and smartphones.

        ----

        ## [1975] Neural Scene Chronology

        **Authors**: *Haotong Lin, Qianqian Wang, Ruojin Cai, Sida Peng, Hadar Averbuch-Elor, Xiaowei Zhou, Noah Snavely*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01988](https://doi.org/10.1109/CVPR52729.2023.01988)

        **Abstract**:

        In this work, we aim to reconstruct a time-varying 3D model, capable of rendering photo-realistic renderings with independent control of viewpoint, illumination, and time, from Internet photos of large-scale landmarks. The core challenges are twofold. First, different types of temporal changes, such as illumination and changes to the underlying scene itself (such as replacing one graffiti artwork with another) are entangled together in the imagery. Second, scene-level temporal changes are often discrete and sporadic over time, rather than continuous. To tackle these problems, we propose a new scene representation equipped with a novel temporal step function encoding method that can model discrete scene-level content changes as piece-wise constant functions over time. Specifically, we represent the scene as a space-time radiance field with a per-image illumination embedding, where temporally-varying scene changes are encoded using a set of learned step functions. To facilitate our task of chronology reconstruction from Internet imagery, we also collect a new dataset of four scenes that exhibit various changes over time. We demonstrate that our method exhibits state-of-the-art view synthesis results on this dataset, while achieving independent control of view-point, time, and illumination. Code and data are available at https://zju3dv.github.io/NeuSC/

        ----

        ## [1976] ReLight My NeRF: A Dataset for Novel View Synthesis and Relighting of Real World Objects

        **Authors**: *Marco Toschi, Riccardo De Matteo, Riccardo Spezialetti, Daniele De Gregorio, Luigi Di Stefano, Samuele Salti*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01989](https://doi.org/10.1109/CVPR52729.2023.01989)

        **Abstract**:

        In this paper, we focus on the problem of rendering novel views from a Neural Radiance Field (NeRF) under unobserved light conditions. To this end, we introduce a novel dataset, dubbed ReNe (Relighting NeRF), framing real world objects under one-light-at-time (OLAT) conditions, annotated with accurate ground-truth camera and light poses. Our acquisition pipeline leverages two robotic arms holding, respectively, a camera and an omni-directional point-wise light source. We release a total of 20 scenes depicting a variety of objects with complex geometry and challenging materials. Each scene includes 2000 images, acquired from 50 different points of views under 40 different OLAT conditions. By leveraging the dataset, we perform an ablation study on the relighting capability of variants of the vanilla NeRF architecture and identify a lightweight architecture that can render novel views of an object under novel light conditions, which we use to establish a non-trivial baseline for the dataset. Dataset and benchmark are available at https://eyecan-ai.

        ----

        ## [1977] ORCa: Glossy Objects as Radiance-Field Cameras

        **Authors**: *Kushagra Tiwary, Akshat Dave, Nikhil Behari, Tzofi Klinghoffer, Ashok Veeraraghavan, Ramesh Raskar*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01990](https://doi.org/10.1109/CVPR52729.2023.01990)

        **Abstract**:

        Reflections on glossy objects contain valuable and hidden information about the surrounding environment. By converting these objects into cameras, we can unlock exciting applications, including imaging beyond the camera's field-of-view and from seemingly impossible vantage points, e.g. from reflections on the human eye. However, this task is challenging because reflections depend jointly on object geometry, material properties, the 3D environment, and the observer's viewing direction. Our approach converts glossy objects with unknown geometry into radiance-field cameras to image the world from the object's perspective. Our key insight is to convert the object surface into a virtual sensor that captures cast reflections as a 2D projection of the 5D environment radiance field visible to and surrounding the object. We show that recovering the environment radiance fields enables depth and radiance estimation from the object to its surroundings in addition to beyond field-of-view novel-view synthesis, i.e. rendering of novel views that are only directly visible to the glossy object present in the scene, but not the observer. Moreover, using the radiance field we can image around occluders caused by close-by objects in the scene. Our method is trained end-to-end on multi-view images of the object and jointly estimates object geometry, diffuse radiance, and the 5D environment radiance field. For more information, visit our website.

        ----

        ## [1978] Nighttime Smartphone Reflective Flare Removal Using Optical Center Symmetry Prior

        **Authors**: *Yuekun Dai, Yihang Luo, Shangchen Zhou, Chongyi Li, Chen Change Loy*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01991](https://doi.org/10.1109/CVPR52729.2023.01991)

        **Abstract**:

        Reflective flare is a phenomenon that occurs when light reflects inside lenses, causing bright spots or a “ghosting effect” in photos, which can impact their quality. Eliminating reflective flare is highly desirable but challenging. Many existing methods rely on manually designed features to detect these bright spots, but they often fail to identify reflective flares created by various types of light and may even mistakenly remove the light sources in scenarios with multiple light sources. To address these challenges, we propose an optical center symmetry prior, which suggests that the reflective flare and light source are always symmetrical around the lens's optical center. This prior helps to locate the reflective flare's proposal region more accurately and can be applied to most smartphone cameras. Building on this prior, we create the first reflective flare removal dataset called BracketFlare, which contains diverse and realistic reflective flare patterns. We use continuous bracketing to capture the reflective flare pattern in the underexposed image and combine it with a normally exposed image to synthesize a pair of flare-corrupted and flare-free images. With the dataset, neural networks can be trained to remove the reflective flares effectively. Extensive experiments demonstrate the effectiveness of our method on both synthetic and real-world datasets.

        ----

        ## [1979] SunStage: Portrait Reconstruction and Relighting Using the Sun as a Light Stage

        **Authors**: *Yifan Wang, Aleksander Holynski, Xiuming Zhang, Xuaner Zhang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01992](https://doi.org/10.1109/CVPR52729.2023.01992)

        **Abstract**:

        A light stage uses a series of calibrated cameras and lights to capture a subject's facial appearance under varying illumination and viewpoint. This captured information is crucial for facial reconstruction and relighting. Unfortunately, light stages are often inaccessible: they are expensive and require significant technical expertise for construction and operation. In this paper, we present SunStage: a lightweight alternative to a light stage that captures comparable data using only a smartphone camera and the sun. Our method only requires the user to capture a selfie video outdoors, rotating in place, and uses the varying angles between the sun and the face as guidance in joint reconstruction of facial geometry, reflectance, camera pose, and lighting parameters. Despite the in-the-wild un-calibrated setting, our approach is able to reconstruct detailed facial appearance and geometry, enabling compelling effects such as relighting, novel view synthesis, and reflectance editing.

        ----

        ## [1980] The Differentiable Lens: Compound Lens Search over Glass Surfaces and Materials for Object Detection

        **Authors**: *Geoffroi Côté, Fahim Mannan, Simon Thibault, Jean-François Lalonde, Felix Heide*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01993](https://doi.org/10.1109/CVPR52729.2023.01993)

        **Abstract**:

        Most camera lens systems are designed in isolation, separately from downstream computer vision methods. Recently, joint optimization approaches that design lenses alongside other components of the image acquisition and processing pipeline—notably, downstream neural networks—have achieved improved imaging quality or better performance on vision tasks. However, these existing methods optimize only a subset of lens parameters and cannot optimize glass materials given their categorical nature. In this work, we develop a differentiable spherical lens simulation model that accurately captures geometrical aberrations. We propose an optimization strategy to address the challenges of lens design—notorious for non-convex loss function landscapes and many manufacturing constraints—that are exacerbated in joint optimization tasks. Specifically, we introduce quantized continuous glass variables to facilitate the optimization and selection of glass materials in an end-to-end design context, and couple this with carefully designed constraints to support manufacturability. In automotive object detection, we report improved detection performance over existing designs even when simplifying designs to two- or three-element lenses, despite significantly degrading the image quality.

        ----

        ## [1981] Teleidoscopic Imaging System for Microscale 3D Shape Reconstruction

        **Authors**: *Ryo Kawahara, Meng-Yu Jennifer Kuo, Shohei Nobuhara*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01994](https://doi.org/10.1109/CVPR52729.2023.01994)

        **Abstract**:

        This paper proposes a practical method of microscale 3D shape capturing by a teleidoscopic imaging system. The main challenge in microscale 3D shape reconstruction is to capture the target from multiple viewpoints with a large enough depth-of-field. Our idea is to employ a teleidoscopic measurement system consisting of three planar mirrors and monocentric lens. The planar mirrors virtually define multiple viewpoints by multiple reflections, and the monocentric lens realizes a high magnification with less blurry and surround view even in closeup imaging. Our contributions include, a structured ray-pixel camera model which handles refractive and reflective projection rays efficiently, analytical evaluations of depth of field of our teleidoscopic imaging system, and a practical calibration algorithm of the teleidoscopic imaging system. Evaluations with real images prove the concept of our measurement system.

        ----

        ## [1982] Looking Through the Glass: Neural Surface Reconstruction Against High Specular Reflections

        **Authors**: *Jiaxiong Qiu, Peng-Tao Jiang, Yifan Zhu, Ze-Xin Yin, Ming-Ming Cheng, Bo Ren*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01995](https://doi.org/10.1109/CVPR52729.2023.01995)

        **Abstract**:

        Neural implicit methods have achieved high-quality 3D object surfaces under slight specular highlights. However, high specular reflections (HSR) often appear in front of target objects when we capture them through glasses. The complex ambiguity in these scenes violates the multi-view consistency, then makes it challenging for recent methods to reconstruct target objects correctly. To remedy this issue, we present a novel surface reconstruction framework, NeuS-HSR, based on implicit neural rendering. In NeuSHSR, the object surface is parameterized as an implicit signed distance function (SDF). To reduce the interference of HSR, we propose decomposing the rendered image into two appearances: the target object and the auxiliary plane. We design a novel auxiliary plane module by combining physical assumptions and neural networks to generate the auxiliary plane appearance. Extensive experiments on synthetic and real-world datasets demonstrate that NeuS-HSR outperforms state-of-the-art approaches for accurate and robust target surface reconstruction against HSR. Code is available at https://github.com/JiaxiongQ/NeuS-HSR.

        ----

        ## [1983] NeuralUDF: Learning Unsigned Distance Fields for Multi-View Reconstruction of Surfaces with Arbitrary Topologies

        **Authors**: *Xiaoxiao Long, Cheng Lin, Lingjie Liu, Yuan Liu, Peng Wang, Christian Theobalt, Taku Komura, Wenping Wang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01996](https://doi.org/10.1109/CVPR52729.2023.01996)

        **Abstract**:

        We present a novel method, called NeuralUDF, for reconstructing surfaces with arbitrary topologies from 2D images via volume rendering. Recent advances in neural rendering based reconstruction have achieved compelling results. However, these methods are limited to objects with closed surfaces since they adopt Signed Distance Function (SDF) as surface representation which requires the target shape to be divided into inside and outside. In this paper, we propose to represent surfaces as the Unsigned Distance Function (UDF) and develop a new volume rendering scheme to learn the neural UDF representation. Specifically, a new density function that correlates the property of UDF with the volume rendering scheme is introduced for robust optimization of the UDF fields. Experiments on the DTU and DeepFashion3D datasets show that our method not only enables high-quality reconstruction of non-closed shapes with complex typologies, but also achieves comparable performance to the SDF based methods on the reconstruction of closed surfaces. Visit our project page at https://www.xxlong.site/NeuralUDF.

        ----

        ## [1984] Sphere-Guided Training of Neural Implicit Surfaces

        **Authors**: *Andreea Dogaru, Andrei-Timotei Ardelean, Savva Ignatyev, Egor Zakharov, Evgeny Burnaev*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01997](https://doi.org/10.1109/CVPR52729.2023.01997)

        **Abstract**:

        In recent years, neural distance functions trained via volumetric ray marching have been widely adopted for multi-view 3D reconstruction. These methods, however, apply the ray marching procedure for the entire scene volume, leading to reduced sampling efficiency and, as a result, lower reconstruction quality in the areas of high-frequency details. In this work, we address this problem via joint training of the implicit function and our new coarse sphere-based surface reconstruction. We use the coarse representation to efficiently exclude the empty volume of the scene from the volumetric ray marching procedure without additional forward passes of the neural surface network, which leads to an increased fidelity of the reconstructions compared to the base systems. We evaluate our approach by incorporating it into the training procedures of several implicit surface modeling methods and observe uniform improvements across both synthetic and real-world datasets. Our codebase can be accessed via the project page
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">†</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">†</sup>
https://andreeadogaru.github.io/SphereGuided.

        ----

        ## [1985] OReX: Object Reconstruction from Planar Cross-sections Using Neural Fields

        **Authors**: *Haim Sawdayee, Amir Vaxman, Amit H. Bermano*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01998](https://doi.org/10.1109/CVPR52729.2023.01998)

        **Abstract**:

        Reconstructing 3D shapes from planar cross-sections is a challenge inspired by downstream applications like medical imaging and geographic informatics. The input is an in/out indicator function fully defined on a sparse collection of planes in space, and the output is an interpolation of the indicator function to the entire volume. Previous works addressing this sparse and ill-posed problem either produce low quality results, or rely on additional priors such as target topology, appearance information, or input normal directions. In this paper, we present OReX, a method for 3D shape reconstruction from slices alone, featuring a Neural Field as the interpolation prior. A modest neural network is trained on the input planes to return an inside/outside estimate for a given 3D coordinate, yielding a powerful prior that induces smoothness and self-similarities. The main challenge for this approach is high-frequency details, as the neural prior is overly smoothing. To alleviate this, we offer an iterative estimation architecture and a hierarchical input sampling scheme that encourage coarse-to-fine training, allowing the training process to focus on high frequencies at later stages. In addition, we identify and analyze a ripple-like effect stemming from the mesh extraction step. We mitigate it by regularizing the spatial gradients of the indicator function around input in/out boundaries during network training, tackling the problem at the root. Through extensive qualitative and quantitative experimentation, we demonstrate our method is robust, accurate, and scales well with the size of the input. We report state-of-the-art results compared to previous approaches and recent potential solutions, and demonstrate the benefit of our individual contributions through analysis and ablation studies. 
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
Code and data available at https://github.com/haimsaw/OReX

        ----

        ## [1986] Persistent Nature: A Generative Model of Unbounded 3D Worlds

        **Authors**: *Lucy Chai, Richard Tucker, Zhengqi Li, Phillip Isola, Noah Snavely*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01999](https://doi.org/10.1109/CVPR52729.2023.01999)

        **Abstract**:

        Despite increasingly realistic image quality, recent 3D image generative models often operate on 3D volumes of fixed extent with limited camera motions. We investigate the task of unconditionally synthesizing unbounded nature scenes, enabling arbitrarily large camera motion while maintaining a persistent 3D world model. Our scene representation consists of an extendable, planar scene layout grid, which can be rendered from arbitrary camera poses via a 3D decoder and volume rendering, and a panoramic sky-dome. Based on this representation, we learn a generative world model solely from single-view internet photos. Our method enables simulating long flights through 3D landscapes, while maintaining global scene consistency-for instance, returning to the starting point yields the same view of the scene. Our approach enables scene extrapolation beyond the fixed bounds of current 3D generative models, while also supporting a persistent, camera-independent world representation that stands in contrast to auto-regressive 3D prediction models. Our project page: https://chail.github.io/persistent-nature/.

        ----

        ## [1987] 3D Neural Field Generation Using Triplane Diffusion

        **Authors**: *J. Ryan Shue, Eric Ryan Chan, Ryan Po, Zachary Ankner, Jiajun Wu, Gordon Wetzstein*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02000](https://doi.org/10.1109/CVPR52729.2023.02000)

        **Abstract**:

        Diffusion models have emerged as the state-of-the-art for image generation, among other tasks. Here, we present an efficient diffusion-based model for 3D-aware generation of neural fields. Our approach pre-processes training data, such as ShapeNet meshes, by converting them to continuous occupancy fields and factoring them into a set of axis-aligned triplane feature representations. Thus, our 3D training scenes are all represented by 2D feature planes, and we can directly train existing 2D diffusion models on these representations to generate 3D neural fields with high quality and diversity, outperforming alternative approaches to 3D-aware generation. Our approach requires essential modifications to existing triplane factorization pipelines to make the resulting features easy to learn for the diffusion model. We demonstrate state-of-the-art results on 3D generation on several object classes from ShapeNet.

        ----

        ## [1988] Diffusion-Based Signed Distance Fields for 3D Shape Generation

        **Authors**: *Jaehyeok Shim, Changwoo Kang, Kyungdon Joo*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02001](https://doi.org/10.1109/CVPR52729.2023.02001)

        **Abstract**:

        We propose a 3D shape generation framework (SDF-Diffusion in short) that uses denoising diffusion models with continuous 3D representation via signed distance fields (SDF). Unlike most existing methods that depend on discontinuous forms, such as point clouds, SDF-Diffusion generates high-resolution 3D shapes while alleviating memory issues by separating the generative process into two-stage: generation and super-resolution. In the first stage, a diffusion-based generative model generates a low-resolution SDF of 3D shapes. Using the estimated low-resolution SDF as a condition, the second stage diffusion model performs super-resolution to generate high-resolution SDF. Our framework can generate a high-fidelity 3D shape despite the extreme spatial complexity. On the ShapeNet dataset, our model shows competitive performance to the state-of-the-art methods and shows applicability on the shape completion task without modification.

        ----

        ## [1989] Efficient View Synthesis and 3D-based Multi-Frame Denoising with Multiplane Feature Representations

        **Authors**: *Thomas Tanay, Ales Leonardis, Matteo Maggioni*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02002](https://doi.org/10.1109/CVPR52729.2023.02002)

        **Abstract**:

        While current multi-frame restoration methods combine information from multiple input images using 2D alignment techniques, recent advances in novel view synthesis are paving the way for a new paradigm relying on volu-metric scene representations. In this work, we introduce the first 3D-based multi-frame denoising method that significantly outperforms its 2D-based counterparts with lower computational requirements. Our method extends the mul-tiplane image (MPI) framework for novel view synthesis by introducing a learnable encoder-renderer pair manipulating multiplane representations in feature space. The encoder fuses information across views and operates in a depth-wise manner while the renderer fuses information across depths and operates in a view-wise manner. The two modules are trained end-to-end and learn to separate depths in an unsupervised way, giving rise to Multiplane Feature (MPF) representations. Experiments on the Spaces and Real Forward-Facing datasets as well as on raw burst data validate our approach for view synthesis, multi-frame denoising, and view synthesis under noisy conditions.

        ----

        ## [1990] Dream3D: Zero-Shot Text-to-3D Synthesis Using 3D Shape Prior and Text-to-Image Diffusion Models

        **Authors**: *Jiale Xu, Xintao Wang, Weihao Cheng, Yan-Pei Cao, Ying Shan, Xiaohu Qie, Shenghua Gao*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02003](https://doi.org/10.1109/CVPR52729.2023.02003)

        **Abstract**:

        Recent CLIP-guided 3D optimization methods, such as DreamFields [19] and PureCLIPNeRF [24], have achieved impressive results in zero-shot text-to-3D synthesis. However, due to scratch training and random initialization without prior knowledge, these methods often fail to generate accurate and faithful 3D structures that conform to the input text. In this paper, we make the first attempt to introduce explicit 3D shape priors into the CLIP-guided 3D optimization process. Specifically, we first generate a high-quality 3D shape from the input text in the text-to-shape stage as a 3D shape prior. We then use it as the initialization of a neural radiance field and optimize it with the full prompt. To address the challenging text-to-shape generation task, we present a simple yet effective approach that directly bridges the text and image modalities with a powerful text-to-image diffusion model. To narrow the style domain gap between the images synthesized by the text-to-image diffusion model and shape renderings used to train the image-to-shape generator, we further propose to jointly optimize a learnable text prompt and fine-tune the text-to-image diffusion model for rendering-style image generation. Our method, Dream3D, is capable of generating imaginative 3D content with superior visual quality and shape accuracy compared to state-of-the-art methods. Our project page is at https://bluestyle97.github.io/dream3d/.

        ----

        ## [1991] SINE: Semantic-driven Image-based NeRF Editing with Prior-guided Editing Field

        **Authors**: *Chong Bao, Yinda Zhang, Bangbang Yang, Tianxing Fan, Zesong Yang, Hujun Bao, Guofeng Zhang, Zhaopeng Cui*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02004](https://doi.org/10.1109/CVPR52729.2023.02004)

        **Abstract**:

        Despite the great success in 2D editing using user-friendly tools, such as Photoshop, semantic strokes, or even text prompts, similar capabilities in 3D areas are still limited, either relying on 3D modeling skills or allowing editing within only a few categories. In this paper, we present a novel semantic-driven NeRF editing approach, which enables users to edit a neural radiance field with a single image, and faithfully delivers edited novel views with high fidelity and multi-view consistency. To achieve this goal, we propose a prior-guided editing field to encode fine-grained geometric and texture editing in 3D space, and develop a series of techniques to aid the editing process, including cyclic constraints with a proxy mesh to facilitate geometric supervision, a color compositing mechanism to stabilize semantic-driven texture editing, and a feature-cluster-based regularization to preserve the irrelevant content unchanged. Extensive experiments and editing examples on both real-world and synthetic data demonstrate that our method achieves photo-realistic 3D editing using only a single edited image, pushing the bound of semantic-driven editing in 3D real-world scenes.

        ----

        ## [1992] 3D Highlighter: Localizing Regions on 3D Shapes via Text Descriptions

        **Authors**: *Dale Decatur, Itai Lang, Rana Hanocka*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02005](https://doi.org/10.1109/CVPR52729.2023.02005)

        **Abstract**:

        We present 3D Highlighter, a technique for localizing semantic regions on a mesh using text as input. A key feature of our system is the ability to interpret “out-of-domain” localizations. Our system demonstrates the ability to reason about where to place non-obviously related concepts on an input 3D shape, such as adding clothing to a bare 3D animal model. Our method contextualizes the text description using a neural field and colors the corresponding region of the shape using a probability-weighted blend. Our neural optimization is guided by a pre-trained CLIP encoder, which bypasses the need for any 3D datasets or 3D annotations. Thus, 3D Highlighter is highly flexible, general, and capable of producing localizations on a myriad of input shapes. Our code is publicly available at https://github.com/threedle/3DHighlighter.

        ----

        ## [1993] Self-Supervised Geometry-Aware Encoder for Style-Based 3D GAN Inversion

        **Authors**: *Yushi Lan, Xuyi Meng, Shuai Yang, Chen Change Loy, Bo Dai*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02006](https://doi.org/10.1109/CVPR52729.2023.02006)

        **Abstract**:

        StyleGAN has achieved great progress in 2D face reconstruction and semantic editing via image inversion and latent editing. While studies over extending 2D StyleGAN to 3D faces have emerged, a corresponding generic 3D GAN inversion framework is still missing, limiting the applications of 3D face reconstruction and semantic editing. In this paper, we study the challenging problem of 3D GAN inversion where a latent code is predicted given a single face image to faithfully recover its 3D shapes and detailed textures. The problem is ill-posed: innumerable compositions of shape and texture could be rendered to the current image. Furthermore, with the limited capacity of a global latent code, 2D inversion methods cannot preserve faithful shape and texture at the same time when applied to 3D models. To solve this problem, we devise an effective self-training scheme to constrain the learning of inversion. The learning is done efficiently without any real-world 2D-3D training pairs but proxy samples generated from a 3D GAN. In addition, apart from a global latent code that captures the coarse shape and texture information, we augment the generation network with a local branch, where pixel-aligned features are added to faithfully reconstruct face details. We further consider a new pipeline to perform 3D view-consistent editing. Extensive experiments show that our method outperforms state-of-the-art inversion methods in both shape and texture reconstruction quality.

        ----

        ## [1994] PanoHead: Geometry-Aware 3D Full-Head Synthesis in 360°

        **Authors**: *Sizhe An, Hongyi Xu, Yichun Shi, Guoxian Song, Ümit Y. Ogras, Linjie Luo*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02007](https://doi.org/10.1109/CVPR52729.2023.02007)

        **Abstract**:

        Synthesis and reconstruction of 3D human head has gained increasing interests in computer vision and computer graphics recently. Existing state-of-the-art 3D generative adversarial networks (GANs) for 3D human head synthesis are either limited to near-frontal views or hard to preserve 3D consistency in large view angles. We propose PanoHead, the first 3D-aware generative model that enables high-quality view-consistent image synthesis of full heads in 360° with diverse appearance and detailed geometry using only in-the-wild unstructured images for training. At its core, we lift up the representation power of recent 3D GANs and bridge the data alignment gap when training from in-the-wild images with widely distributed views. Specifically, we propose a novel two-stage self-adaptive image alignment for robust 3D GAN training. We further introduce a tri-grid neural volume representation that effectively addresses front-face and back-head feature entanglement rooted in the widely-adopted tri-plane formulation. Our method instills prior knowledge of 2D image segmentation in adversarial learning of 3D neural scene structures, enabling compositable head synthesis in diverse backgrounds. Benefiting from these designs, our method significantly outperforms previous 3D GANs, generating high-quality 3D heads with accurate geometry and diverse appearances, even with long wavy and afro hairstyles, renderable from arbitrary poses. Furthermore, we show that our system can reconstruct full 3D heads from single input images for personalized realistic 3D avatars.

        ----

        ## [1995] StyleGene: Crossover and Mutation of Region-level Facial Genes for Kinship Face Synthesis

        **Authors**: *Hao Li, Xianxu Hou, Zepeng Huang, Linlin Shen*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02008](https://doi.org/10.1109/CVPR52729.2023.02008)

        **Abstract**:

        High-fidelity kinship face synthesis has many potential applications, such as kinship verification, missing child identification, and social media analysis. However, it is challenging to synthesize high-quality descendant faces with genetic relations due to the lack of large-scale, high-quality annotated kinship data. This paper proposes RFG (Region-level Facial Gene) extraction framework to address this issue. We propose to use IGE (Image-based Gene Encoder), LGE (Latent-based Gene Encoder) and Gene Decoder to learn the RFGs of a given face image, and the relationships between RFGs and the latent space of Style-GAN2. As cycle-like losses are designed to measure the 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$\mathcal{L}_{2}$</tex>
 distances between the output of Gene Decoder and image encoder, and that between the output of LGE and IGE, only face images are required to train our framework, i.e. no paired kinship face data is required. Based upon the proposed RFGs, a crossover and mutation module is further designed to inherit the facial parts of parents. A Gene Pool has also been used to introduce the variations into the mutation of RFGs. The diversity of the faces of descendants can thus be significantly increased. Qualitative, quantitative, and subjective experiments on FIW, TSKinFace, and FF-Databases clearly show that the quality and diversity of kinship faces generated by our approach are much better than the existing state-of-the-art methods.

        ----

        ## [1996] Parameter Efficient Local Implicit Image Function Network for Face Segmentation

        **Authors**: *Mausoom Sarkar, Nikitha S. R., Mayur Hemani, Rishabh Jain, Balaji Krishnamurthy*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02009](https://doi.org/10.1109/CVPR52729.2023.02009)

        **Abstract**:

        Face parsing is defined as the per-pixel labeling of images containing human faces. The labels are defined to identify key facial regions like eyes, lips, nose, hair, etc. In this work, we make use of the structural consistency of the human face to propose a lightweight face-parsing method using a Local Implicit Function network, FP-LIIF. We propose a simple architecture having a convolutional encoder and a pixel MLP decoder that uses 1/26
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">th</sup>
 number of parameters compared to the state-of-the-art models and yet matches or outperforms state-of-the-art models on multiple datasets, like CelebAMask-HQ and LaPa. We do not use any pretraining, and compared to other works, our network can also generate segmentation at different resolutions without any changes in the input resolution. This work enables the use of facial segmentation on low-compute or low-bandwidth devices because of its higher FPS and smaller model size.

        ----

        ## [1997] Graphics Capsule: Learning Hierarchical 3D Face Representations from 2D Images

        **Authors**: *Chang Yu, Xiangyu Zhu, Xiaomei Zhang, Zhaoxiang Zhang, Zhen Lei*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02010](https://doi.org/10.1109/CVPR52729.2023.02010)

        **Abstract**:

        The function of constructing the hierarchy of objects is important to the visual process of the human brain. Previous studies have successfully adopted capsule networks to decompose the digits and faces into parts in an unsupervised manner to investigate the similar perception mechanism of neural networks. However, their descriptions are restricted to the 2D space, limiting their capacities to imitate the intrinsic 3D perception ability of humans. In this paper, we propose an Inverse Graphics Capsule Network (IGC-Net) to learn the hierarchical 3D face representations from large-scale unlabeled images. The core of IGC-Net is a new type of capsule, named graphics capsule, which represents 3D primitives with interpretable parameters in computer graphics (CG), including depth, albedo, and 3D pose. Specifically, IGC-Net first decomposes the objects into a set of semantic-consistent part-level descriptions and then assembles them into object-level descriptions to build the hierarchy. The learned graphics capsules reveal how the neural networks, oriented at visual perception, understand faces as a hierarchy of 3D models. Besides, the discovered parts can be deployed to the unsupervised face segmentation task to evaluate the semantic consistency of our method. Moreover, the part-level descriptions with explicit physical meanings provide insight into the face analysis that originally runs in a black box, such as the importance of shape and texture for face recognition. Experiments on CelebA, BP4D, and Multi-PIE demonstrate the characteristics of our IGC-Net.

        ----

        ## [1998] Next3D: Generative Neural Texture Rasterization for 3D-Aware Head Avatars

        **Authors**: *Jingxiang Sun, Xuan Wang, Lizhen Wang, Xiaoyu Li, Yong Zhang, Hongwen Zhang, Yebin Liu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02011](https://doi.org/10.1109/CVPR52729.2023.02011)

        **Abstract**:

        3D-aware generative adversarial networks (GANs) synthesize high-fidelity and multi-view-consistent facial images using only collections of single-view 2D imagery. Towards fine-grained control over facial attributes, recent efforts incorporate 3D Morphable Face Model (3DMM) to describe deformation in generative radiance fields either explicitly or implicitly. Explicit methods provide fine-grained expression control but cannot handle topological changes caused by hair and accessories, while implicit ones can model varied topologies but have limited generalization caused by the unconstrained deformation fields. We propose a novel 3D GAN framework for unsupervised learning of generative, high-quality and 3D-consistent facial avatars from unstructured 2D images. To achieve both deformation accuracy and topological flexibility, we propose a 3D representation called Generative Texture-Rasterized Tri-planes. The proposed representation learns Generative Neural Textures on top of parametric mesh templates and then projects them into three orthogonal-viewed feature planes through rasterization, forming a tri-plane feature representation for volume rendering. In this way, we combine both fine-grained expression control of mesh-guided explicit deformation and the flexibility of implicit volumetric representation. We further propose specific modules for modeling mouth interior which is not taken into account by 3DMM. Our method demonstrates state-of-the-art 3D-aware synthesis quality and animation ability through extensive experiments. Furthermore, serving as 3D prior, our animatable 3D representation boosts multiple applications including one-shot facial avatars and 3D-aware stylization. Project page: https://mrtornado24.github.io/Next3D/. Code: https://github.com/MrTornado24/Next3D.

        ----

        ## [1999] Learning Neural Parametric Head Models

        **Authors**: *Simon Giebenhain, Tobias Kirschstein, Markos Georgopoulos, Martin Rünz, Lourdes Agapito, Matthias Nießner*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.02012](https://doi.org/10.1109/CVPR52729.2023.02012)

        **Abstract**:

        We propose a novel 3D morphable model for complete human heads based on hybrid neural fields. At the core of our model lies a neural parametric representation that disentangles identity and expressions in disjoint latent spaces. To this end, we capture a person's identity in a canonical space as a signed distance field (SDF), and model facial expressions with a neural deformation field. In addition, our representation achieves high-fidelity local detail by introducing an ensemble of local fields centered around facial anchor points. To facilitate generalization, we train our model on a newly-captured dataset of over 3700 head scans from 203 different identities using a custom high-end 3D scanning setup. Our dataset significantly exceeds comparable existing datasets, both with respect to quality and completeness of geometry, averaging around 3.5M mesh faces per scan
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
We will publicly release our dataset along with a public benchmark for both neural head avatar construction as well as an evaluation on a hidden test-set for inference-time fitting.. Finally, we demonstrate that our approach outperforms state-of-the-art methods in terms of fitting error and reconstruction quality.

        ----

        

[Go to the previous page](CVPR-2023-list09.md)

[Go to the next page](CVPR-2023-list11.md)

[Go to the catalog section](README.md)