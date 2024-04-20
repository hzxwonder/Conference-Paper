## [1200] Block-Skim: Efficient Question Answering for Transformer

**Authors**: *Yue Guan, Zhengyi Li, Zhouhan Lin, Yuhao Zhu, Jingwen Leng, Minyi Guo*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i10.21316](https://doi.org/10.1609/aaai.v36i10.21316)

**Abstract**:

Transformer models have achieved promising results on natural language processing (NLP) tasks including extractive question answering (QA). Common Transformer encoders used in NLP tasks process the hidden states of all input tokens in the context paragraph throughout all layers. However, different from other tasks such as sequence classification, answering the raised question does not necessarily need all the tokens in the context paragraph. Following this motivation, we propose Block-skim, which learns to skim unnecessary context in higher hidden layers to improve and accelerate the Transformer performance. The key idea of Block-Skim is to identify the context that must be further processed and those that could be safely discarded early on during inference. Critically, we find that such information could be sufficiently derived from the self-attention weights inside the Transformer model. We further prune the hidden states corresponding to the unnecessary positions early in lower layers, achieving significant inference-time speedup. To our surprise, we observe that models pruned in this way outperform their full-size counterparts. Block-Skim improves QA models' accuracy on different datasets and achieves 3 times speedup on BERT-base model.

----

## [1201] Deep Clustering of Text Representations for Supervision-Free Probing of Syntax

**Authors**: *Vikram Gupta, Haoyue Shi, Kevin Gimpel, Mrinmaya Sachan*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i10.21317](https://doi.org/10.1609/aaai.v36i10.21317)

**Abstract**:

We explore deep clustering of multilingual text representations for unsupervised model interpretation and induction of syntax. As these representations are high-dimensional, out-of-the-box methods like K-means do not work well. Thus, our approach jointly transforms the representations into a lower-dimensional cluster-friendly space and clusters them. We consider two notions of syntax: Part of Speech Induction (POSI) and Constituency Labelling (CoLab) in this work. Interestingly, we find that Multilingual BERT (mBERT) contains surprising amount of syntactic knowledge of English; possibly even as much as English BERT (E-BERT). Our model can be used as a supervision-free probe which is arguably a less-biased way of probing. We find that unsupervised probes show benefits from higher layers as compared to supervised probes. We further note that our unsupervised probe utilizes E-BERT and mBERT representations differently, especially for POSI. We validate the efficacy of our probe by demonstrating its capabilities as a unsupervised syntax induction technique. Our probe works well for both syntactic formalisms by simply adapting the input representations. We report competitive performance of our probe on 45-tag English POSI, state-of-the-art performance on 12-tag POSI across 10 languages, and competitive results on CoLab. We also perform zero-shot syntax induction on resource impoverished languages and report strong results.

----

## [1202] Few-Shot Cross-Lingual Stance Detection with Sentiment-Based Pre-training

**Authors**: *Momchil Hardalov, Arnav Arora, Preslav Nakov, Isabelle Augenstein*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i10.21318](https://doi.org/10.1609/aaai.v36i10.21318)

**Abstract**:

The goal of stance detection is to determine the viewpoint expressed in a piece of text towards a target. These viewpoints or contexts are often expressed in many different languages depending on the user and the platform, which can be a local news outlet, a social media platform, a news forum, etc. Most research on stance detection, however, has been limited to working with a single language and on a few limited targets, with little work on cross-lingual stance detection. Moreover, non-English sources of labelled data are often scarce and present additional challenges. Recently, large multilingual language models have substantially improved the performance on many non-English tasks, especially such with a limited number of examples. This highlights the importance of model pre-training and its ability to learn from few examples. In this paper, we present the most comprehensive study of cross-lingual stance detection to date: we experiment with 15 diverse datasets in 12 languages from 6 language families, and with 6 low-resource evaluation settings each. For our experiments, we build on pattern-exploiting training (PET), proposing the addition of a novel label encoder to simplify the verbalisation procedure. We further propose sentiment-based generation of stance data for pre-training, which shows sizeable improvement of more than 6% F1 absolute in few-shot learning settings compared to several strong baselines.

----

## [1203] Attention Biasing and Context Augmentation for Zero-Shot Control of Encoder-Decoder Transformers for Natural Language Generation

**Authors**: *Devamanyu Hazarika, Mahdi Namazifar, Dilek Hakkani-Tür*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i10.21319](https://doi.org/10.1609/aaai.v36i10.21319)

**Abstract**:

Controlling neural network-based models for natural language generation (NLG) to realize desirable attributes in the generated outputs has broad applications in numerous areas such as machine translation, document summarization, and dialog systems. Approaches that enable such control in a zero-shot manner would be of great importance as, among other reasons, they remove the need for additional annotated data and training. In this work, we propose novel approaches for controlling encoder-decoder transformer-based NLG models in zero shot. While zero-shot control has previously been observed in massive models (e.g., GPT3), our method enables such control for smaller models. This is done by applying two control knobs, attention biasing and context augmentation, to these models directly during decoding and without additional training or auxiliary models. These knobs control the generation process by directly manipulating trained NLG models (e.g., biasing cross-attention layers). We show that not only are these NLG models robust to such manipulations but also their behavior could be controlled without an impact on their generation performance.

----

## [1204] GALAXY: A Generative Pre-trained Model for Task-Oriented Dialog with Semi-supervised Learning and Explicit Policy Injection

**Authors**: *Wanwei He, Yinpei Dai, Yinhe Zheng, Yuchuan Wu, Zheng Cao, Dermot Liu, Peng Jiang, Min Yang, Fei Huang, Luo Si, Jian Sun, Yongbin Li*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i10.21320](https://doi.org/10.1609/aaai.v36i10.21320)

**Abstract**:

Pre-trained models have proved to be powerful in enhancing task-oriented dialog systems. However, current pre-training methods mainly focus on enhancing dialog understanding and generation tasks while neglecting the exploitation of dialog policy. In this paper, we propose GALAXY, a novel pre-trained dialog model that explicitly learns dialog policy from limited labeled dialogs and large-scale unlabeled dialog corpora via semi-supervised learning. Specifically, we introduce a dialog act prediction task for policy optimization during pre-training and employ a consistency regularization term to refine the learned representation with the help of unlabeled dialogs. We also implement a gating mechanism to weigh suitable unlabeled dialog samples. Empirical results show that GALAXY substantially improves the performance of task-oriented dialog systems, and achieves new state-of-the-art results on benchmark datasets: In-Car, MultiWOZ2.0 and MultiWOZ2.1, improving their end-to-end combined scores by 2.5, 5.3 and 5.5 points, respectively. We also show that GALAXY has a stronger few-shot ability than existing models under various low-resource settings. For reproducibility, we release the code and data at https://github.com/siat-nlp/GALAXY.

----

## [1205] Protecting Intellectual Property of Language Generation APIs with Lexical Watermark

**Authors**: *Xuanli He, Qiongkai Xu, Lingjuan Lyu, Fangzhao Wu, Chenguang Wang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i10.21321](https://doi.org/10.1609/aaai.v36i10.21321)

**Abstract**:

Nowadays, due to the breakthrough in natural language generation (NLG), including machine translation, document summarization, image captioning, etc NLG models have been encapsulated in cloud APIs to serve over half a billion people worldwide and process over one hundred billion word generations per day. Thus, NLG APIs have already become essential profitable services in many commercial companies. Due to the substantial financial and intellectual investments, service providers adopt a pay-as-you-use policy to promote sustainable market growth. However, recent works have shown that cloud platforms suffer from financial losses imposed by model extraction attacks, which aim to imitate the functionality and utility of the victim services, thus violating the intellectual property (IP) of cloud APIs. This work targets at protecting IP of NLG APIs by identifying the attackers who have utilized watermarked responses from the victim NLG APIs. However, most existing watermarking techniques are not directly amenable for IP protection of NLG APIs. To bridge this gap, we first present a novel watermarking method for text generation APIs by conducting lexical modification to the original outputs. Compared with the competitive baselines, our watermark approach achieves better identifiable performance in terms of p-value, with fewer semantic losses. In addition, our watermarks are more understandable and intuitive to humans than the baselines. Finally, the empirical studies show our approach is also applicable to queries from different domains, and is effective on the attacker trained on a mixture of the corpus which includes less than 10% watermarked samples.

----

## [1206] BROS: A Pre-trained Language Model Focusing on Text and Layout for Better Key Information Extraction from Documents

**Authors**: *Teakgyu Hong, Donghyun Kim, Mingi Ji, Wonseok Hwang, Daehyun Nam, Sungrae Park*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i10.21322](https://doi.org/10.1609/aaai.v36i10.21322)

**Abstract**:

Key information extraction (KIE) from document images requires understanding the contextual and spatial semantics of texts in two-dimensional (2D) space.
Many recent studies try to solve the task by developing pre-trained language models focusing on combining visual features from document images with texts and their layout.
On the other hand, this paper tackles the problem by going back to the basic: effective combination of text and layout. 
Specifically, we propose a pre-trained language model, named BROS (BERT Relying On Spatiality), that encodes relative positions of texts in 2D space and learns from unlabeled documents with area-masking strategy.
With this optimized training scheme for understanding texts in 2D space, BROS shows comparable or better performance compared to previous methods on four KIE benchmarks (FUNSD, SROIE*, CORD, and SciTSR) without relying on visual features.
This paper also reveals two real-world challenges in KIE tasks--(1) minimizing the error from incorrect text ordering and (2) efficient learning from fewer downstream examples--and demonstrates the superiority of BROS over previous methods.

----

## [1207] Non-autoregressive Translation with Layer-Wise Prediction and Deep Supervision

**Authors**: *Chenyang Huang, Hao Zhou, Osmar R. Zaïane, Lili Mou, Lei Li*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i10.21323](https://doi.org/10.1609/aaai.v36i10.21323)

**Abstract**:

How do we perform efficient inference while retaining high translation quality? Existing neural machine translation models, such as Transformer, achieve high performance, but they decode words one by one, which is inefficient. Recent non-autoregressive translation models speed up the inference, but their quality is still inferior. In this work, we propose DSLP, a highly efficient and high-performance model for machine translation. The key insight is to train a non-autoregressive Transformer with Deep Supervision and feed additional Layer-wise Predictions. We conducted extensive experiments on four translation tasks (both directions of WMT'14 EN-DE and WMT'16 EN-RO). Results show that our approach consistently improves the BLEU scores compared with respective base models. Specifically, our best variant outperforms the autoregressive model on three translation tasks, while being 14.8 times more efficient in inference.

----

## [1208] Word Level Robustness Enhancement: Fight Perturbation with Perturbation

**Authors**: *Pei Huang, Yuting Yang, Fuqi Jia, Minghao Liu, Feifei Ma, Jian Zhang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i10.21324](https://doi.org/10.1609/aaai.v36i10.21324)

**Abstract**:

State-of-the-art deep NLP models have achieved impressive improvements on many tasks. However, they are found to be vulnerable to some perturbations. Before they are widely adopted, the fundamental issues of robustness need to be addressed. In this paper, we design a robustness enhancement method to defend against word substitution perturbation, whose basic idea is to fight perturbation with perturbation. We find that: although many well-trained deep models are not robust in the setting of the presence of adversarial samples, they satisfy weak robustness. That means they can handle most non-crafted perturbations well. Taking advantage of the weak robustness property of deep models, we utilize non-crafted perturbations to resist the adversarial perturbations crafted by attackers. Our method contains two main stages. The first stage is using randomized perturbation to conform the input to the data distribution. The second stage is using randomized perturbation to eliminate the instability of prediction results and enhance the robustness guarantee. Experimental results show that our method can significantly improve the ability of deep models to resist the state-of-the-art adversarial attacks while maintaining the prediction performance on the original clean data.

----

## [1209] Predicting Above-Sentence Discourse Structure Using Distant Supervision from Topic Segmentation

**Authors**: *Patrick Huber, Linzi Xing, Giuseppe Carenini*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i10.21325](https://doi.org/10.1609/aaai.v36i10.21325)

**Abstract**:

RST-style discourse parsing plays a vital role in many NLP tasks, revealing the underlying semantic/pragmatic structure of potentially complex and diverse documents. Despite its importance, one of the most prevailing limitations in modern day discourse parsing is the lack of large-scale datasets. To overcome the data sparsity issue, distantly supervised approaches from tasks like sentiment analysis and summarization have been recently proposed. Here, we extend this line of research by exploiting distant supervision from topic segmentation, which can arguably provide a strong and oftentimes complementary signal for high-level discourse structures. Experiments on two human-annotated discourse treebanks confirm that our proposal generates accurate tree structures on sentence and paragraph level, consistently outperforming previous distantly supervised models on the sentence-to-document task and occasionally reaching even higher scores on the sentence-to-paragraph level.

----

## [1210] Call for Customized Conversation: Customized Conversation Grounding Persona and Knowledge

**Authors**: *Yoonna Jang, Jungwoo Lim, Yuna Hur, Dongsuk Oh, Suhyune Son, Yeonsoo Lee, Dong-Hoon Shin, Seungryong Kim, Heuiseok Lim*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i10.21326](https://doi.org/10.1609/aaai.v36i10.21326)

**Abstract**:

Humans usually have conversations by making use of prior knowledge about a topic and background information of the people whom they are talking to. However, existing conversational agents and datasets do not consider such comprehensive information, and thus they have a limitation in generating the utterances where the knowledge and persona are fused properly. To address this issue, we introduce a call For Customized conversation (FoCus) dataset where the customized answers are built with the user's persona and Wikipedia knowledge. To evaluate the abilities to make informative and customized utterances of pre-trained language models, we utilize BART and GPT-2 as well as transformer-based models. We assess their generation abilities with automatic scores and conduct human evaluations for qualitative results. We examine whether the model reflects adequate persona and knowledge with our proposed two sub-tasks, persona grounding (PG) and knowledge grounding (KG). Moreover, we show that the utterances of our data are constructed with the proper knowledge and persona through grounding quality assessment.

----

## [1211] Towards Building ASR Systems for the Next Billion Users

**Authors**: *Tahir Javed, Sumanth Doddapaneni, Abhigyan Raman, Kaushal Santosh Bhogale, Gowtham Ramesh, Anoop Kunchukuttan, Pratyush Kumar, Mitesh M. Khapra*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i10.21327](https://doi.org/10.1609/aaai.v36i10.21327)

**Abstract**:

Recent methods in speech and language technology pretrain very large models which are fine-tuned for specific tasks. However, the benefits of such large models are often limited to a few resource rich languages of the world. In this work, we make multiple contributions towards building ASR systems for low resource languages from the Indian subcontinent. First, we curate 17,000 hours of raw speech data for 40 Indian languages from a wide variety of domains including education, news, technology, and finance. Second, using this raw speech data we pretrain several variants of wav2vec style models for 40 Indian languages. Third, we analyze the pretrained models to find key features: codebook vectors of similar sounding phonemes are shared across languages, representations across layers are discriminative of the language family, and attention heads often pay attention within small local windows. Fourth, we fine-tune this model for downstream ASR for 9 languages and obtain state-of-the-art results on 3 public datasets, including on very low-resource languages such as Sinhala and Nepali. Our work establishes that multilingual pretraining is an effective strategy for building ASR systems for the linguistically diverse speakers of the Indian subcontinent.

----

## [1212] Span-Based Semantic Role Labeling with Argument Pruning and Second-Order Inference

**Authors**: *Zixia Jia, Zhaohui Yan, Haoyi Wu, Kewei Tu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i10.21328](https://doi.org/10.1609/aaai.v36i10.21328)

**Abstract**:

We study graph-based approaches to span-based semantic role labeling. This task is difficult due to the need to enumerate all possible predicate-argument pairs and the high degree of imbalance between positive and negative samples. Based on these difficulties, high-order inference that considers interactions between multiple arguments and predicates is often deemed beneficial but has rarely been used in span-based semantic role labeling. Because even for second-order inference, there are already O(n^5) parts for a sentence of length n, and exact high-order inference is intractable. In this paper, we propose a framework consisting of two networks: a predicate-agnostic argument pruning network that reduces the number of candidate arguments to O(n), and a semantic role labeling network with an optional second-order decoder that is unfolded from an approximate inference algorithm. Our experiments show that our framework achieves significant and consistent improvement over previous approaches.

----

## [1213] Incorporating Constituent Syntax for Coreference Resolution

**Authors**: *Fan Jiang, Trevor Cohn*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i10.21329](https://doi.org/10.1609/aaai.v36i10.21329)

**Abstract**:

Syntax has been shown to benefit Coreference Resolution from incorporating long-range dependencies and structured information captured by syntax trees, either in traditional statistical machine learning based systems or recently proposed neural models. However, most leading systems use only dependency trees. We argue that constituent trees also encode important information, such as explicit span-boundary signals captured by nested multi-word phrases, extra linguistic labels and hierarchical structures useful for detecting anaphora. In this work, we propose a simple yet effective graph-based method to incorporate constituent syntactic structures. Moreover, we also explore to utilise higher-order neighbourhood information to encode rich structures in constituent trees. A novel message propagation mechanism is therefore proposed to enable information flow among elements in syntax trees. Experiments on the English and Chinese portions of OntoNotes 5.0 benchmark show that our proposed model either beats a strong baseline or achieves new state-of-the-art performance. Code is available at https://github.com/Fantabulous-J/Coref-Constituent-Graph.

----

## [1214] XLM-K: Improving Cross-Lingual Language Model Pre-training with Multilingual Knowledge

**Authors**: *Xiaoze Jiang, Yaobo Liang, Weizhu Chen, Nan Duan*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i10.21330](https://doi.org/10.1609/aaai.v36i10.21330)

**Abstract**:

Cross-lingual pre-training has achieved great successes using monolingual and bilingual plain text corpora. However, most pre-trained models neglect multilingual knowledge, which is language agnostic but comprises abundant cross-lingual structure alignment. In this paper, we propose XLM-K, a cross-lingual language model incorporating multilingual knowledge in pre-training. XLM-K augments existing multilingual pre-training with two knowledge tasks, namely Masked Entity Prediction Task and Object Entailment Task. We evaluate XLM-K on MLQA, NER and XNLI. Experimental results clearly demonstrate significant improvements over existing multilingual language models. The results on MLQA and NER  exhibit the superiority of XLM-K in knowledge related tasks. The success in XNLI shows a better cross-lingual transferability obtained in XLM-K. What is more, we provide a detailed probing analysis to confirm the desired knowledge captured in our pre-training regimen. The code is available at https://github.com/microsoft/Unicoder/tree/master/pretraining/xlmk.

----

## [1215] Hierarchical Context Tagging for Utterance Rewriting

**Authors**: *Lisa Jin, Linfeng Song, Lifeng Jin, Dong Yu, Daniel Gildea*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i10.21331](https://doi.org/10.1609/aaai.v36i10.21331)

**Abstract**:

Utterance rewriting aims to recover coreferences and omitted information from the latest turn of a multi-turn dialogue. Recently, methods that tag rather than linearly generate sequences have proven stronger in both in- and out-of-domain rewriting settings. This is due to a tagger's smaller search space as it can only copy tokens from the dialogue context. However, these methods may suffer from low coverage when phrases that must be added to a source utterance cannot be covered by a single context span. This can occur in languages like English that introduce tokens such as prepositions into the rewrite for grammaticality. We propose a hierarchical context tagger (HCT) that mitigates this issue by predicting slotted rules (e.g., "besides _") whose slots are later filled with context spans. HCT (i) tags the source string with token-level edit actions and slotted rules and (ii) fills in the resulting rule slots with spans from the dialogue context. This rule tagging allows HCT to add out-of-context tokens and multiple spans at once; we further cluster the rules to truncate the long tail of the rule distribution. Experiments on several benchmarks show that HCT can outperform state-of-the-art rewriting systems by ~2 BLEU points.

----

## [1216] Search and Learn: Improving Semantic Coverage for Data-to-Text Generation

**Authors**: *Shailza Jolly, Zi Xuan Zhang, Andreas Dengel, Lili Mou*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i10.21332](https://doi.org/10.1609/aaai.v36i10.21332)

**Abstract**:

Data-to-text generation systems aim to generate text descriptions based on input data (often represented in the tabular form). A typical system uses huge training samples for learning the correspondence between tables and texts. However, large training sets are expensive to obtain, limiting the applicability of these approaches in real-world scenarios. In this work, we focus on few-shot data-to-text generation. We observe that, while fine-tuned pretrained language models may generate plausible sentences, they suffer from the low semantic coverage problem in the few-shot setting. In other words, important input slots tend to be missing in the generated text. To this end, we propose a search-and-learning approach that leverages pretrained language models but inserts the missing slots to improve the semantic coverage. We further finetune our system based on the search results to smooth out the search noise, yielding better-quality text and improving inference efficiency to a large extent. Experiments show that our model achieves high performance on E2E and WikiBio datasets. Especially, we cover 98.35% of input slots on E2E, largely alleviating the low coverage problem.

----

## [1217] Braid: Weaving Symbolic and Neural Knowledge into Coherent Logical Explanations

**Authors**: *Aditya Kalyanpur, Tom Breloff, David A. Ferrucci*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i10.21333](https://doi.org/10.1609/aaai.v36i10.21333)

**Abstract**:

Traditional symbolic reasoning engines, while attractive for their precision and explicability, have a few major drawbacks: the use of brittle inference procedures that rely on exact matching (unification) of logical terms, an inability to deal with uncertainty, and the need for a precompiled rule-base of knowledge (the “knowledge acquisition” problem). To address these issues, we devise a novel logical reasoner called Braid, that supports probabilistic rules, and uses the notion of custom unification functions and dynamic rule generation to overcome the brittle matching and knowledge-gap problem prevalent in traditional reasoners. In this paper, we describe the reasoning algorithms used in Braid, and their implementation in a distributed task-based framework that builds proof/explanation graphs for an input query. We use a simple QA example from a children’s story to motivate Braid’s design and explain how the various components work together to produce a coherent logical explanation. Finally, we evaluate Braid on the ROC Story Cloze test and achieve close to state-of-the-art results while providing frame-based explanations.

----

## [1218] Self-Supervised Audio-and-Text Pre-training with Extremely Low-Resource Parallel Data

**Authors**: *Yu Kang, Tianqiao Liu, Hang Li, Yang Hao, Wenbiao Ding*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i10.21334](https://doi.org/10.1609/aaai.v36i10.21334)

**Abstract**:

Multimodal pre-training for audio-and-text has recently been proved to be effective and has significantly improved the performance of many downstream speech understanding tasks. However, these state-of-the-art pre-training audio-text models work well only when provided with large amount of parallel audio-and-text data, which brings challenges on many languages that are rich in unimodal corpora but scarce of parallel cross-modal corpus. In this paper, we investigate whether it is possible to pre-train an audio-text multimodal model with extremely low-resource parallel data and extra non-parallel unimodal data. Our pre-training framework consists of the following components: (1) Intra-modal Denoising Auto-Encoding (IDAE), which is able to reconstruct input text (audio) representations from a noisy version of itself. (2) Cross-modal Denoising Auto-Encoding (CDAE), which is pre-trained to reconstruct the input text (audio), given both a noisy version of the input text (audio) and the corresponding translated noisy audio features (text embeddings). (3) Iterative Denoising Process (IDP), which iteratively translates raw audio (text) and the corresponding text embeddings (audio features) translated from previous iteration into the new less-noisy text embeddings (audio features). We adapt a dual cross-modal Transformer as our backbone model which consists of two unimodal encoders for IDAE and two cross-modal encoders for CDAE and IDP. Our method achieves comparable performance on multiple downstream speech understanding tasks compared with the model pre-trained on fully parallel data, demonstrating the great potential of the proposed method.

----

## [1219] Bridging the Gap: Using Deep Acoustic Representations to Learn Grounded Language from Percepts and Raw Speech

**Authors**: *Gaoussou Youssouf Kebe, Luke E. Richards, Edward Raff, Francis Ferraro, Cynthia Matuszek*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i10.21335](https://doi.org/10.1609/aaai.v36i10.21335)

**Abstract**:

Learning to understand grounded language, which connects natural language to percepts, is a critical research area. Prior work in grounded language acquisition has focused primarily on textual inputs. In this work, we demonstrate the feasibility of performing grounded language acquisition on paired visual percepts and raw speech inputs. This will allow human-robot interactions in which language about novel tasks and environments is learned from end-users, reducing dependence on textual inputs and potentially mitigating the effects of demographic bias found in widely available speech recognition systems. We leverage recent work in self-supervised speech representation models and show that learned representations of speech can make language grounding systems more inclusive towards specific groups while maintaining or even increasing general performance.

----

## [1220] ALP: Data Augmentation Using Lexicalized PCFGs for Few-Shot Text Classification

**Authors**: *Hazel H. Kim, Daecheol Woo, Seong Joon Oh, Jeong-Won Cha, Yo-Sub Han*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i10.21336](https://doi.org/10.1609/aaai.v36i10.21336)

**Abstract**:

Data augmentation has been an important ingredient for boosting performances of learned models. Prior data augmentation methods for few-shot text classification have led to great performance boosts. However, they have not been designed to capture the intricate compositional structure of natural language. As a result, they fail to generate samples with plausible and diverse sentence structures. Motivated by this, we present the data Augmentation using Lexicalized Probabilistic context-free grammars (ALP) that generates augmented samples with diverse syntactic structures with plausible grammar. The lexicalized PCFG parse trees consider both the constituents and dependencies to produce a syntactic frame that maximizes a variety of word choices in a syntactically preservable manner without specific domain experts. Experiments on few-shot text classification tasks demonstrate that ALP enhances many state-of-the-art classification methods. As a second contribution, we delve into the train-val splitting methodologies when a data augmentation method comes into play. We argue empirically that the traditional splitting of training and validation sets is sub-optimal compared to our novel augmentation-based splitting strategies that further expand the training split with the same number of labeled data. Taken together, our contributions on the data augmentation strategies yield a strong training recipe for few-shot text classification tasks.

----

## [1221] CAISE: Conversational Agent for Image Search and Editing

**Authors**: *Hyounghun Kim, Doo Soon Kim, Seunghyun Yoon, Franck Dernoncourt, Trung Bui, Mohit Bansal*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i10.21337](https://doi.org/10.1609/aaai.v36i10.21337)

**Abstract**:

Demand for image editing has been increasing as users' desire for expression is also increasing. However, for most users, image editing tools are not easy to use since the tools require certain expertise in photo effects and have complex interfaces. Hence, users might need someone to help edit their images, but having a personal dedicated human assistant for every user is impossible to scale. For that reason, an automated assistant system for image editing is desirable. Additionally, users want more image sources for diverse image editing works, and integrating an image search functionality into the editing tool is a potential remedy for this demand. Thus, we propose a dataset of an automated Conversational Agent for Image Search and Editing (CAISE). To our knowledge, this is the first dataset that provides conversational image search and editing annotations, where the agent holds a grounded conversation with users and helps them to search and edit images according to their requests. To build such a system, we first collect image search and editing conversations between pairs of annotators. The assistant-annotators are equipped with a customized image search and editing tool to address the requests from the user-annotators. The functions that the assistant-annotators conduct with the tool are recorded as executable commands, allowing the trained system to be useful for real-world application execution. We also introduce a generator-extractor baseline model for this task, which can adaptively select the source of the next token (i.e., from the vocabulary or from textual/visual contexts) for the executable command. This serves as a strong starting point while still leaving a large human-machine performance gap for useful future work. Data and code are available: https://github.com/hyounghk/CAISE.

----

## [1222] Dual Task Framework for Improving Persona-Grounded Dialogue Dataset

**Authors**: *Minju Kim, Beong-woo Kwak, Youngwook Kim, Hong-in Lee, Seung-won Hwang, Jinyoung Yeo*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i10.21338](https://doi.org/10.1609/aaai.v36i10.21338)

**Abstract**:

This paper introduces a simple yet effective data-centric approach for the task of improving persona-conditioned dialogue agents. Prior model-centric approaches unquestioningly depend on the raw crowdsourced benchmark datasets such as Persona-Chat. In contrast, we aim to fix annotation artifacts in benchmarking, which is orthogonally applicable to any dialogue model. Specifically, we augment relevant personas to improve dialogue dataset/agent, by leveraging the primal-dual structure of the two tasks, predicting dialogue responses and personas based on each other. Experiments on Persona-Chat show that our approach outperforms pre-trained LMs by an 11.7 point gain in terms of accuracy.

----

## [1223] Minimally-Supervised Joint Learning of Event Volitionality and Subject Animacy Classification

**Authors**: *Hirokazu Kiyomaru, Sadao Kurohashi*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i10.21339](https://doi.org/10.1609/aaai.v36i10.21339)

**Abstract**:

Volitionality and subject animacy are fundamental and closely related properties of an event. Their classification is challenging because it requires contextual text understanding and a huge amount of labeled data. This paper proposes a novel method that jointly learns volitionality and subject animacy at a low cost, heuristically labeling events in a raw corpus. Volitionality labels are assigned using a small lexicon of volitional and non-volitional adverbs such as deliberately and accidentally; subject animacy labels are assigned using a list of animate and inanimate nouns obtained from ontological knowledge. We then consider the problem of learning a classifier from the labeled events so that it can perform well on unlabeled events without the words used for labeling. We view the problem as a bias reduction or unsupervised domain adaptation problem and apply the techniques. We conduct experiments with crowdsourced gold data in Japanese and English and show that our method effectively learns volitionality and subject animacy without manually labeled data.

----

## [1224] From Fully Trained to Fully Random Embeddings: Improving Neural Machine Translation with Compact Word Embedding Tables

**Authors**: *Krtin Kumar, Peyman Passban, Mehdi Rezagholizadeh, Yiu Sing Lau, Qun Liu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i10.21340](https://doi.org/10.1609/aaai.v36i10.21340)

**Abstract**:

Embedding matrices are key components in neural natural language processing (NLP) models that are responsible to provide numerical representations of input tokens (i.e. words or subwords). In this paper, we analyze the impact and utility of such matrices in the context of neural machine translation (NMT). We show that detracting syntactic and semantic information from word embeddings and running NMT systems with random embeddings is not as damaging as it initially sounds. We also show how incorporating only a limited amount of task-specific knowledge from fully-trained embeddings can boost the performance NMT systems. Our findings demonstrate that in exchange for negligible deterioration in performance, any NMT model can be run with partially random embeddings. Working with such structures means a minimal memory requirement as there is no longer need to store large embedding tables, which is a significant gain in industrial and on-device settings. We evaluated our embeddings in translating English into German and French and achieved a 5.3x compression rate. Despite having a considerably smaller architecture, our models in some cases are even able to outperform state-of-the-art baselines.

----

## [1225] SGD-X: A Benchmark for Robust Generalization in Schema-Guided Dialogue Systems

**Authors**: *Harrison Lee, Raghav Gupta, Abhinav Rastogi, Yuan Cao, Bin Zhang, Yonghui Wu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i10.21341](https://doi.org/10.1609/aaai.v36i10.21341)

**Abstract**:

Zero/few-shot transfer to unseen services is a critical challenge in task-oriented dialogue research. The Schema-Guided Dialogue (SGD) dataset introduced a paradigm for enabling models to support any service in zero-shot through schemas, which describe service APIs to models in natural language. We explore the robustness of dialogue systems to linguistic variations in schemas by designing SGD-X - a benchmark extending SGD with semantically similar yet stylistically diverse variants for every schema. We observe that two top state tracking models fail to generalize well across schema variants, measured by joint goal accuracy and a novel metric for measuring schema sensitivity. Additionally, we present a simple model-agnostic data augmentation method to improve schema robustness.

----

## [1226] Unifying Model Explainability and Robustness for Joint Text Classification and Rationale Extraction

**Authors**: *Dongfang Li, Baotian Hu, Qingcai Chen, Tujie Xu, Jingcong Tao, Yunan Zhang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i10.21342](https://doi.org/10.1609/aaai.v36i10.21342)

**Abstract**:

Recent works have shown explainability and robustness are two crucial ingredients of trustworthy and reliable text classification. However, previous works usually address one of two aspects: i) how to extract accurate rationales for explainability while being beneficial to prediction; ii) how to make the predictive model robust to different types of adversarial attacks. Intuitively, a model that produces helpful explanations should be more robust against adversarial attacks, because we cannot trust the model that outputs explanations but changes its prediction under small perturbations. To this end, we propose a joint classification and rationale extraction model named AT-BMC. It includes two key mechanisms: mixed Adversarial Training (AT) is designed to use various perturbations in discrete and embedding space to improve the model’s robustness, and Boundary Match Constraint (BMC) helps to locate rationales more precisely with the guidance of boundary information. Performances on benchmark datasets demonstrate that the proposed AT-BMC outperforms baselines on both classification and rationale extraction by a large margin. Robustness analysis shows that the proposed AT-BMC decreases the attack success rate effectively by up to 69%. The results indicate that there are connections between robust models and better explanations.

----

## [1227] Text Revision By On-the-Fly Representation Optimization

**Authors**: *Jingjing Li, Zichao Li, Tao Ge, Irwin King, Michael R. Lyu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i10.21343](https://doi.org/10.1609/aaai.v36i10.21343)

**Abstract**:

Text revision refers to a family of natural language generation tasks, where the source and target sequences share moderate resemblance in surface form but differentiate in attributes, such as text formality and simplicity. Current state-of-the-art methods formulate these tasks as sequence-to-sequence learning problems, which rely on large-scale parallel training corpus. In this paper, we present an iterative in-place editing approach for text revision, which requires no parallel data. In this approach, we simply fine-tune a pre-trained Transformer with masked language modeling and attribute classification. During inference, the editing at each iteration is realized by two-step span replacement. At the first step, the distributed representation of the text optimizes on the fly towards an attribute function. At the second step, a text span is masked and another new one is proposed conditioned on the optimized representation. The empirical experiments on two typical and important text revision tasks, text formalization and text simplification, show the effectiveness of our approach. It achieves competitive and even better performance than state-of-the-art supervised methods on text simplification, and gains better performance than strong unsupervised methods on text formalization. Our code and model are released at https://github.com/jingjingli01/OREO.

----

## [1228] Unified Named Entity Recognition as Word-Word Relation Classification

**Authors**: *Jingye Li, Hao Fei, Jiang Liu, Shengqiong Wu, Meishan Zhang, Chong Teng, Donghong Ji, Fei Li*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i10.21344](https://doi.org/10.1609/aaai.v36i10.21344)

**Abstract**:

So far, named entity recognition (NER) has been involved with three major types, including flat, overlapped (aka. nested), and discontinuous NER, which have mostly been studied individually. Recently, a growing interest has been built for unified NER, tackling the above three jobs concurrently with one single model. Current best-performing methods mainly include span-based and sequence-to-sequence models, where unfortunately the former merely focus on boundary identification and the latter may suffer from exposure bias. In this work, we present a novel alternative by modeling the unified NER as word-word relation classification, namely W^2NER. The architecture resolves the kernel bottleneck of unified NER by effectively modeling the neighboring relations between entity words with Next-Neighboring-Word (NNW) and Tail-Head-Word-* (THW-*) relations. Based on the W^2NER scheme we develop a neural framework, in which the unified NER is modeled as a 2D grid of word pairs. We then propose multi-granularity 2D convolutions for better refining the grid representations. Finally, a co-predictor is used to sufficiently reason the word-word relations. We perform extensive experiments on 14 widely-used benchmark datasets for flat, overlapped, and discontinuous NER (8 English and 6 Chinese datasets), where our model beats all the current top-performing baselines, pushing the state-of-the-art performances of unified NER.

----

## [1229] Sequence-to-Action: Grammatical Error Correction with Action Guided Sequence Generation

**Authors**: *Jiquan Li, Junliang Guo, Yongxin Zhu, Xin Sheng, Deqiang Jiang, Bo Ren, Linli Xu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i10.21345](https://doi.org/10.1609/aaai.v36i10.21345)

**Abstract**:

The task of Grammatical Error Correction (GEC) has received remarkable attention with wide applications in Natural Language Processing (NLP) in recent years. While one of the key principles of GEC is to keep the correct parts unchanged and avoid over-correction, previous sequence-to-sequence (seq2seq) models generate results from scratch, which are not guaranteed to follow the original sentence structure and may suffer from the over-correction problem. In the meantime, the recently proposed sequence tagging models can overcome the over-correction problem by only generating edit operations, but are conditioned on human designed language-specific tagging labels. In this paper, we combine the pros and alleviate the cons of both models by proposing a novel Sequence-to-Action (S2A) module. The S2A module jointly takes the source and target sentences as input, and is able to automatically generate a token-level action sequence before predicting each token, where each action is generated from three choices named SKIP, COPY and GENerate. Then the actions are fused with the basic seq2seq framework to provide final predictions. We conduct experiments on the benchmark datasets of both English and Chinese GEC tasks. Our model consistently outperforms the seq2seq baselines, while being able to significantly alleviate the over-correction problem as well as holding better generality and diversity in the generation results compared to the sequence tagging models.

----

## [1230] Dynamic Key-Value Memory Enhanced Multi-Step Graph Reasoning for Knowledge-Based Visual Question Answering

**Authors**: *Mingxiao Li, Marie-Francine Moens*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i10.21346](https://doi.org/10.1609/aaai.v36i10.21346)

**Abstract**:

Knowledge-based visual question answering (VQA) is a vision-language task that requires an agent to correctly answer image-related questions using knowledge that is not presented in the given image. It is not only a more challenging task than regular VQA but also a vital step towards building a general VQA system. Most existing knowledge-based VQA systems process knowledge and image information similarly and ignore the fact that the knowledge base (KB) contains complete information about a triplet, while the extracted image information might be incomplete as the relations between two objects are missing or wrongly detected. In this paper, we propose a novel model named dynamic knowledge memory enhanced multi-step graph reasoning (DMMGR), which performs explicit and implicit reasoning over a key-value knowledge memory module and a spatial-aware image graph, respectively. Specifically, the memory module learns a dynamic knowledge representation and generates a knowledge-aware question representation at each reasoning step. Then, this representation is used to guide a graph attention operator over the spatial-aware image graph. Our model achieves new state-of-the-art accuracy on the KRVQR and FVQA datasets. We also conduct ablation experiments to prove the effectiveness of each component of the proposed model.

----

## [1231] Knowledge Bridging for Empathetic Dialogue Generation

**Authors**: *Qintong Li, Piji Li, Zhaochun Ren, Pengjie Ren, Zhumin Chen*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i10.21347](https://doi.org/10.1609/aaai.v36i10.21347)

**Abstract**:

Lack of external knowledge makes empathetic dialogue systems difficult to perceive implicit emotions and learn emotional interactions from limited dialogue history. To address the above problems, we propose to leverage external knowledge, including commonsense knowledge and emotional lexical knowledge, to explicitly understand and express emotions in empathetic dialogue generation. We first enrich the dialogue history by jointly interacting with external knowledge and construct an emotional context graph. Then we learn emotional context representations from the knowledge-enriched emotional context graph and distill emotional signals, which are the prerequisites to predicate emotions expressed in responses. Finally, to generate the empathetic response, we propose an emotional cross-attention mechanism to learn the emotional dependencies from the emotional context graph. Extensive experiments conducted on a benchmark dataset verify the effectiveness of the proposed method. In addition, we find the performance of our method can be further improved by integrating with a pre-trained model that works orthogonally.

----

## [1232] Contrast and Generation Make BART a Good Dialogue Emotion Recognizer

**Authors**: *Shimin Li, Hang Yan, Xipeng Qiu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i10.21348](https://doi.org/10.1609/aaai.v36i10.21348)

**Abstract**:

In dialogue systems, utterances with similar semantics may have distinctive emotions under different contexts. Therefore, modeling long-range contextual emotional relationships with speaker dependency plays a crucial part in dialogue emotion recognition. Meanwhile, distinguishing the different emotion categories is non-trivial since they usually have semantically similar sentiments. To this end, we adopt supervised contrastive learning to make different emotions mutually exclusive to identify similar emotions better. Meanwhile, we utilize an auxiliary response generation task to enhance the model's ability of handling context information, thereby forcing the model to recognize emotions with similar semantics in diverse contexts. To achieve these objectives, we use the pre-trained encoder-decoder model BART as our backbone model since it is very suitable for both understanding and generation tasks. The experiments on four datasets demonstrate that our proposed model obtains significantly more favorable results than the state-of-the-art model in dialogue emotion recognition. The ablation study further demonstrates the effectiveness of supervised contrastive loss and generative loss.

----

## [1233] A Semi-supervised Learning Approach with Two Teachers to Improve Breakdown Identification in Dialogues

**Authors**: *Qian Lin, Hwee Tou Ng*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i10.21349](https://doi.org/10.1609/aaai.v36i10.21349)

**Abstract**:

Identifying breakdowns in ongoing dialogues helps to improve communication effectiveness. Most prior work on this topic relies on human annotated data and data augmentation to learn a classification model. While quality labeled dialogue data requires human annotation and is usually expensive to obtain, unlabeled data is easier to collect from various sources. In this paper, we propose a novel semi-supervised teacher-student learning framework to tackle this task. We introduce two teachers which are trained on labeled data and perturbed labeled data respectively. We leverage unlabeled data to improve classification in student training where we employ two teachers to refine the labeling of unlabeled data through teacher-student learning in a bootstrapping manner. Through our proposed training approach, the student can achieve improvements over single-teacher performance. Experimental results on the Dialogue Breakdown Detection Challenge dataset DBDC5 and Learning to Identify Follow-Up Questions dataset LIF show that our approach outperforms all previous published approaches as well as other supervised and semi-supervised baseline methods.

----

## [1234] DiffSinger: Singing Voice Synthesis via Shallow Diffusion Mechanism

**Authors**: *Jinglin Liu, Chengxi Li, Yi Ren, Feiyang Chen, Zhou Zhao*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i10.21350](https://doi.org/10.1609/aaai.v36i10.21350)

**Abstract**:

Singing voice synthesis (SVS) systems are built to synthesize high-quality and expressive singing voice, in which the acoustic model generates the acoustic features (e.g., mel-spectrogram) given a music score. Previous singing acoustic models adopt a simple loss (e.g., L1 and L2) or generative adversarial network (GAN) to reconstruct the acoustic features, while they suffer from over-smoothing and unstable training issues respectively, which hinder the naturalness of synthesized singing. 
In this work, we propose DiffSinger, an acoustic model for SVS based on the diffusion probabilistic model. DiffSinger is a parameterized Markov chain that iteratively converts the noise into mel-spectrogram conditioned on the music score. By implicitly optimizing variational bound, DiffSinger can be stably trained and generate realistic outputs. 
To further improve the voice quality and speed up inference, we introduce a shallow diffusion mechanism to make better use of the prior knowledge learned by the simple loss. Specifically, DiffSinger starts generation at a shallow step smaller than the total number of diffusion steps, according to the intersection of the diffusion trajectories of the ground-truth mel-spectrogram and the one predicted by a simple mel-spectrogram decoder. Besides, we propose boundary prediction methods to locate the intersection and determine the shallow step adaptively.
The evaluations conducted on a Chinese singing dataset demonstrate that DiffSinger outperforms state-of-the-art SVS work. Extensional experiments also prove the generalization of our methods on text-to-speech task (DiffSpeech). Audio samples: https://diffsinger.github.io. Codes: https://github.com/MoonInTheRiver/DiffSinger.

----

## [1235] KGR4: Retrieval, Retrospect, Refine and Rethink for Commonsense Generation

**Authors**: *Xin Liu, Dayiheng Liu, Baosong Yang, Haibo Zhang, Junwei Ding, Wenqing Yao, Weihua Luo, Haiying Zhang, Jinsong Su*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i10.21351](https://doi.org/10.1609/aaai.v36i10.21351)

**Abstract**:

Generative commonsense reasoning requires machines to generate sentences describing an everyday scenario given several concepts, which has attracted much attention recently. However, existing models cannot perform as well as humans, since sentences they produce are often implausible and grammatically incorrect. In this paper, inspired by the process of humans creating sentences, we propose a novel Knowledge-enhanced Commonsense Generation framework, termed KGR4, consisting of four stages: Retrieval, Retrospect, Refine, Rethink. Under this framework, we first perform retrieval to search for relevant sentences from external corpus as the prototypes. Then, we train the generator that either edits or copies these prototypes to generate candidate sentences, of which potential errors will be fixed by an autoencoder-based refiner. Finally, we select the output sentence from candidate sentences produced by generators with different hyper-parameters. Experimental results and in-depth analysis on the CommonGen benchmark strongly demonstrate the effectiveness of our framework. Particularly, KGR4 obtains 33.56 SPICE in the official leaderboard, outperforming the previously-reported best result by 2.49 SPICE and achieving state-of-the-art performance. We release the code at https://github.com/DeepLearnXMU/KGR-4.

----

## [1236] Improving Biomedical Information Retrieval with Neural Retrievers

**Authors**: *Man Luo, Arindam Mitra, Tejas Gokhale, Chitta Baral*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i10.21352](https://doi.org/10.1609/aaai.v36i10.21352)

**Abstract**:

Information retrieval (IR) is essential in search engines and dialogue systems as well as natural language processing tasks such as open-domain question answering. IR serve an important function in the biomedical domain, where content and sources of scientific knowledge may evolve rapidly. Although neural retrievers have surpassed traditional IR approaches such as TF-IDF and BM25 in standard open-domain question answering tasks, they are still found lacking in the biomedical domain. In this paper, we seek to improve information retrieval (IR) using neural retrievers (NR) in the biomedical domain, and achieve this goal using a three-pronged approach. First, to tackle the relative lack of data in the biomedical domain, we propose a template-based question generation method that can be leveraged to train neural retriever models. Second, we develop two novel pre-training tasks that are closely aligned to the downstream task of information retrieval. Third, we introduce the ``Poly-DPR'' model which encodes each context into multiple context vectors. Extensive experiments and analysis on the BioASQ challenge suggest that our proposed method leads to large gains over existing neural approaches and beats BM25 in the small-corpus setting. We show that BM25 and our method can complement each other, and a simple hybrid model leads to further gains in the large corpus setting.

----

## [1237] The King Is Naked: On the Notion of Robustness for Natural Language Processing

**Authors**: *Emanuele La Malfa, Marta Kwiatkowska*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i10.21353](https://doi.org/10.1609/aaai.v36i10.21353)

**Abstract**:

There is growing evidence that the classical notion of adversarial robustness originally introduced for images has been adopted as a de facto standard by a large part of the NLP research community.
 We show that this notion is problematic in the context of NLP as it considers a narrow spectrum of linguistic phenomena. In this paper, we argue for semantic robustness, which is better aligned with the human concept of linguistic fidelity. We characterize semantic robustness in terms of biases that it is expected to induce in a model. We study semantic robustness of a range of vanilla and robustly trained architectures using a template-based generative test bed. We complement the analysis with empirical evidence that, despite being harder to implement, semantic robustness can improve performance %gives guarantees for on complex linguistic phenomena where models robust in the classical sense fail.

----

## [1238] Selecting Optimal Context Sentences for Event-Event Relation Extraction

**Authors**: *Hieu Man, Nghia Trung Ngo, Linh Ngo Van, Thien Huu Nguyen*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i10.21354](https://doi.org/10.1609/aaai.v36i10.21354)

**Abstract**:

Understanding events entails recognizing the structural and temporal orders between event mentions to build event structures/graphs for input documents. To achieve this goal, our work addresses the problems of subevent relation extraction (SRE) and temporal event relation extraction (TRE) that aim to predict subevent and temporal relations between two given event mentions/triggers in texts. Recent state-of-the-art methods for such problems have employed transformer-based language models (e.g., BERT) to induce effective contextual representations for input event mention pairs. However, a major limitation of existing transformer-based models for SRE and TRE is that they can only encode input texts of limited length (i.e., up to 512 sub-tokens in BERT), thus unable to effectively capture important context sentences that are farther away in the documents. In this work, we introduce a novel method to better model document-level context with important context sentences for event-event relation extraction. Our method seeks to identify the most important context sentences for a given entity mention pair in a document and pack them into shorter documents to be consume entirely by transformer-based language models for representation learning. The REINFORCE algorithm is employed to train models where novel reward functions are presented to capture model performance, and context-based and knowledge-based similarity between sentences for our problem. Extensive experiments demonstrate the effectiveness of the proposed method with state-of-the-art performance on benchmark datasets.

----

## [1239] Semantic Parsing in Task-Oriented Dialog with Recursive Insertion-Based Encoder

**Authors**: *Elman Mansimov, Yi Zhang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i10.21355](https://doi.org/10.1609/aaai.v36i10.21355)

**Abstract**:

We introduce a Recursive INsertion-based Encoder (RINE), a novel approach for semantic parsing in task-oriented dialog. Our model consists of an encoder network that incrementally builds the semantic parse tree by predicting the non-terminal label and its positions in the linearized tree. At the generation time, the model constructs the semantic parse tree by recursively inserting the predicted non-terminal labels at the predicted positions until termination. RINE achieves state-of-the-art exact match accuracy on low- and high-resource versions of the conversational semantic parsing benchmark TOP, outperforming strong sequence-to-sequence models and transition-based parsers. We also show that our model design is applicable to nested named entity recognition task, where it performs on par with state-of-the-art approach designed for that task. Finally, we demonstrate that our approach is 2-3.5 times faster than the sequence-to-sequence model at inference time.

----

## [1240] CINS: Comprehensive Instruction for Few-Shot Learning in Task-Oriented Dialog Systems

**Authors**: *Fei Mi, Yasheng Wang, Yitong Li*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i10.21356](https://doi.org/10.1609/aaai.v36i10.21356)

**Abstract**:

As the labeling cost for different modules in task-oriented dialog (ToD) systems is high, a major challenge is to learn different tasks with the least amount of labeled data. Recently, pre-trained language models (PLMs) have shown promising results for few-shot learning in ToD. To better utilize the power of PLMs, this paper proposes Comprehensive Instruction (CINS) that exploits PLMs with extra task-specific instructions. We design a schema (definition, constraint, prompt) of instructions and their customized realizations for three important downstream tasks in ToD, ie. intent classification, dialog state tracking, and natural language generation. A sequence-to-sequence model (T5) is adopted to solve these three tasks in a unified framework. Extensive experiments are conducted on these ToD tasks in realistic few-shot learning scenarios with small validation data. Empirical results demonstrate that the proposed CINS approach consistently improves techniques that finetune PLMs with raw input or short prompt.

----

## [1241] Semantic Self-Segmentation for Abstractive Summarization of Long Documents in Low-Resource Regimes

**Authors**: *Gianluca Moro, Luca Ragazzi*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i10.21357](https://doi.org/10.1609/aaai.v36i10.21357)

**Abstract**:

The quadratic memory complexity of transformers prevents long document summarization in low computational resource scenarios. State-of-the-art models need to apply input truncation, thus discarding and ignoring potential summary-relevant contents, leading to a performance drop. Furthermore, this loss is generally destructive for semantic text analytics in high-impact domains such as the legal one. In this paper, we propose a novel semantic self-segmentation (Se3) approach for long document summarization to address the critical problems of low-resource regimes, namely to process inputs longer than the GPU memory capacity and produce accurate summaries despite the availability of only a few dozens of training instances. Se3 segments a long input into semantically coherent chunks, allowing transformers to summarize very long documents without truncation by summarizing each chunk and concatenating the results. Experimental outcomes show the approach significantly improves the performance of abstractive summarization transformers, even with just a dozen of labeled data, achieving new state-of-the-art results on two legal datasets of different domains and contents. Finally, we report ablation studies to evaluate each contribution of the components of our method to the performance gain.

----

## [1242] Eye of the Beholder: Improved Relation Generalization for Text-Based Reinforcement Learning Agents

**Authors**: *Keerthiram Murugesan, Subhajit Chaudhury, Kartik Talamadupula*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i10.21358](https://doi.org/10.1609/aaai.v36i10.21358)

**Abstract**:

Text-based games (TBGs) have become a popular proving ground for the demonstration of learning-based agents that make decisions in quasi real-world settings. The crux of the problem for a reinforcement learning agent in such TBGs is identifying the objects in the world, and those objects' relations with that world. While the recent use of text-based resources for increasing an agent's knowledge and improving its generalization have shown promise, we posit in this paper that there is much yet to be learned from visual representations of these same worlds. Specifically, we propose to retrieve images that represent specific instances of text observations from the world and train our agents on such images. This improves the agent's overall understanding of the game scene and objects' relationships to the world around them, and the variety of visual representations on offer allow the agent to generate a better generalization of a relationship. We show that incorporating such images improves the performance of agents in various TBG settings.

----

## [1243] Improving Neural Cross-Lingual Abstractive Summarization via Employing Optimal Transport Distance for Knowledge Distillation

**Authors**: *Thong Thanh Nguyen, Anh Tuan Luu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i10.21359](https://doi.org/10.1609/aaai.v36i10.21359)

**Abstract**:

Current state-of-the-art cross-lingual summarization models employ multi-task learning paradigm, which works on a shared vocabulary module and relies on the self-attention mechanism to attend among tokens in two languages. However, correlation learned by self-attention is often loose and implicit, inefficient in capturing crucial cross-lingual representations between languages. The matter worsens when performing on languages with separate morphological or structural features, making the cross-lingual alignment more challenging, resulting in the performance drop. To overcome this problem, we propose a novel Knowledge-Distillation-based framework for Cross-Lingual Summarization, seeking to explicitly construct cross-lingual correlation by distilling the knowledge of the monolingual summarization teacher into the cross-lingual summarization student. Since the representations of the teacher and the student lie on two different vector spaces, we further propose a Knowledge Distillation loss using Sinkhorn Divergence, an Optimal-Transport distance, to estimate the discrepancy between those teacher and student representations. Due to the intuitively geometric nature of Sinkhorn Divergence, the student model can productively learn to align its produced cross-lingual hidden states with monolingual hidden states, hence leading to a strong correlation between distant languages. Experiments on cross-lingual summarization datasets in pairs of distant languages demonstrate that our method outperforms state-of-the-art models under both high and low-resourced settings.

----

## [1244] HiTKG: Towards Goal-Oriented Conversations via Multi-Hierarchy Learning

**Authors**: *Jinjie Ni, Vlad Pandelea, Tom Young, Haicang Zhou, Erik Cambria*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i10.21360](https://doi.org/10.1609/aaai.v36i10.21360)

**Abstract**:

Human conversations are guided by short-term and long-term goals. We study how to plan short-term goal sequences as coherently as humans do and naturally direct them to an assigned long-term goal in open-domain conversations. Goal sequences are a series of knowledge graph (KG) entity-relation connections generated by KG walkers that traverse through the KG. The existing recurrent and graph attention based KG walkers either insufficiently utilize the conversation states or lack global guidance. In our work, a hierarchical model learns goal planning in a hierarchical learning framework. We present HiTKG, a hierarchical transformer-based graph walker that leverages multiscale inputs to make precise and flexible predictions on KG paths. Furthermore, we propose a two-hierarchy learning framework that employs two stages to learn both turn-level (short-term) and global-level (long-term) conversation goals. Specifically, at the first stage, HiTKG is trained in a supervised fashion to learn how to plan turn-level goal sequences; at the second stage, HiTKG tries to naturally approach the assigned global goal via reinforcement learning. In addition, we propose MetaPath as the backbone method for KG path representation to exploit the entity and relation information concurrently. We further propose Multi-source Decoding Inputs and Output-level Length Head to improve the decoding controllability. Our experiments show that HiTKG achieves a significant improvement in the performance of turn-level goal learning compared with state-of-the-art baselines. Additionally, both automatic and human evaluation prove the effectiveness of the two-hierarchy learning framework for both short-term and long-term goal planning.

----

## [1245] Is Discourse Role Important for Emotion Recognition in Conversation?

**Authors**: *Donovan Ong, Jian Su, Bin Chen, Anh Tuan Luu, Ashok Narendranath, Yue Li, Shuqi Sun, Yingzhan Lin, Haifeng Wang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i10.21361](https://doi.org/10.1609/aaai.v36i10.21361)

**Abstract**:

A conversation is a sequence of utterances, where each utterance plays a specific discourse role while expressing a particular emotion. This paper proposes a novel method to exploit latent discourse role information of an utterance to determine the emotion it conveys in a conversation. Specifically, we use a variant of the Variational-Autoencoder (VAE) to model the context-aware latent discourse roles of each utterance in an unsupervised way. The latent discourse role representation further equips the utterance representation with a salient clue for more accurate emotion recognition. Our experiments show that our proposed method beats the best-reported performances on three public Emotion Recognition in Conversation datasets. This proves that the discourse role information of an utterance plays an important role in the emotion recognition task, which no previous work has studied.

----

## [1246] Improved Text Classification via Contrastive Adversarial Training

**Authors**: *Lin Pan, Chung-Wei Hang, Avirup Sil, Saloni Potdar*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i10.21362](https://doi.org/10.1609/aaai.v36i10.21362)

**Abstract**:

We propose a simple and general method to regularize the fine-tuning of Transformer-based encoders for text classification tasks. Specifically, during fine-tuning we generate adversarial examples by perturbing the word embedding matrix of the model and perform contrastive learning on clean and adversarial examples in order to teach the model to learn noise-invariant representations. By training on both clean and adversarial examples along with the additional contrastive objective, we observe consistent improvement over standard fine-tuning on clean examples. On several GLUE benchmark tasks, our fine-tuned Bert_Large model outperforms Bert_Large baseline by 1.7% on average, and our fine-tuned Roberta_Large improves over Roberta_Large baseline by 1.3%. We additionally validate our method in different domains using three intent classification datasets, where our fine-tuned Roberta_Large outperforms Roberta_Large baseline by 1-2% on average. For the challenging low-resource scenario, we train our system using half of the training data (per intent) in each of the three intent classification datasets, and achieve similar performance compared to the baseline trained with full training data.

----

## [1247] LeSICiN: A Heterogeneous Graph-Based Approach for Automatic Legal Statute Identification from Indian Legal Documents

**Authors**: *Shounak Paul, Pawan Goyal, Saptarshi Ghosh*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i10.21363](https://doi.org/10.1609/aaai.v36i10.21363)

**Abstract**:

The task of Legal Statute Identification (LSI) aims to identify the legal statutes that are relevant to a given description of facts or evidence of a legal case. 
 Existing methods only utilize the textual content of facts and legal articles to guide such a task. However, the citation network among case documents and legal statutes is a rich source of additional information, which is not considered by existing models. 
 In this work, we take the first step towards utilising both the text and the legal citation network for the LSI task.
 We curate a large novel dataset for this task, including facts of cases from several major Indian Courts of Law, and statutes from the Indian Penal Code (IPC). 
 Modeling the statutes and training documents as a heterogeneous graph, our proposed model LeSICiN can learn rich textual and graphical features, and can also tune itself to correlate these features. 
 Thereafter, the model can be used to inductively predict links between test documents (new nodes whose graphical features are not available to the model) and statutes (existing nodes). 
 Extensive experiments on the dataset show that our model comfortably outperforms several state-of-the-art baselines, by exploiting the graphical structure along with textual features.

----

## [1248] Transformer Uncertainty Estimation with Hierarchical Stochastic Attention

**Authors**: *Jiahuan Pei, Cheng Wang, György Szarvas*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i10.21364](https://doi.org/10.1609/aaai.v36i10.21364)

**Abstract**:

Transformers are state-of-the-art in a wide range of NLP tasks and have also been applied to many real-world products. Understanding the reliability and certainty of transformer models is crucial for building trustable machine learning applications, e.g., medical diagnosis. Although many recent transformer extensions have been proposed, the study of the uncertainty estimation of transformer models is under-explored. In this work, we propose a novel way to enable transformers to have the capability of uncertainty estimation and, meanwhile, retain the original predictive performance. This is achieved by learning hierarchical stochastic self-attention that attends to values and a set of learnable centroids, respectively. Then new attention heads are formed with a mixture of sampled centroids using the Gumbel-Softmax trick. We theoretically show that the self-attention approximation by sampling from a Gumbel distribution is upper bounded. We empirically evaluate our model on two text classification tasks with both in-domain (ID) and out-of-domain (OOD) datasets.
 The experimental results demonstrate that our approach: (1) achieves the best predictive-uncertainty trade-off among compared methods; (2) exhibits very competitive (in most cases, better) predictive performance on ID datasets; (3) is on par with Monte Carlo dropout and ensemble methods in uncertainty estimation on OOD datasets.

----

## [1249] STEPS: Semantic Typing of Event Processes with a Sequence-to-Sequence Approach

**Authors**: *Sveva Pepe, Edoardo Barba, Rexhina Blloshmi, Roberto Navigli*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i10.21365](https://doi.org/10.1609/aaai.v36i10.21365)

**Abstract**:

Enabling computers to comprehend the intent of human actions by processing language is one of the fundamental goals of Natural Language Understanding. 
 An emerging task in this context is that of free-form event process typing, which aims at understanding the overall goal of a protagonist in terms of an action and an object, given a sequence of events.
 This task was initially treated as a learning-to-rank problem by exploiting the similarity between processes and action/object textual definitions. 
 However, this approach appears to be overly complex, binds the output types to a fixed inventory for possible word definitions and, moreover, leaves space for further enhancements as regards performance.
 In this paper, we advance the field by reformulating the free-form event process typing task as a sequence generation problem and put forward STEPS, an end-to-end approach for producing user intent in terms of actions and objects only, dispensing with the need for their definitions.
 In addition to this, we eliminate several dataset constraints set by previous works, while at the same time significantly outperforming them. 
 We release the data and software at https://github.com/SapienzaNLP/steps.

----

## [1250] Sparse Structure Learning via Graph Neural Networks for Inductive Document Classification

**Authors**: *Yinhua Piao, Sangseon Lee, Dohoon Lee, Sun Kim*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i10.21366](https://doi.org/10.1609/aaai.v36i10.21366)

**Abstract**:

Recently, graph neural networks (GNNs) have been widely used for document classification. However, most existing methods are based on static word co-occurrence graphs without sentence-level information, which poses three challenges:(1) word ambiguity, (2) word synonymity, and (3) dynamic contextual dependency. To address these challenges, we propose a novel GNN-based sparse structure learning model for inductive document classification. Specifically, a document-level graph is initially generated by a disjoint union of sentence-level word co-occurrence graphs. Our model collects a set of trainable edges connecting disjoint words between sentences, and employs structure learning to sparsely select edges with dynamic contextual dependencies. Graphs with sparse structure can jointly exploit local and global contextual information in documents through GNNs. For inductive learning, the refined document graph is further fed into a general readout function for graph-level classification and optimization in an end-to-end manner. Extensive experiments on several real-world datasets demonstrate that the proposed model outperforms most state-of-the-art results, and reveal the necessity to learn sparse structures for each document.

----

## [1251] STEM: Unsupervised STructural EMbedding for Stance Detection

**Authors**: *Ron Korenblum Pick, Vladyslav Kozhukhov, Dan Vilenchik, Oren Tsur*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i10.21367](https://doi.org/10.1609/aaai.v36i10.21367)

**Abstract**:

Stance detection is an important task, supporting many downstream tasks such as discourse parsing and modeling the propagation of fake news, rumors, and science denial. In this paper, we propose a novel framework for stance detection.  Our framework is unsupervised and domain-independent. Given a claim and a multi-participant discussion -- we construct the interaction network from which we derive topological embedding for each speaker. These speaker embedding enjoy the following property:  speakers with the same stance tend to be represented by similar vectors, while antipodal vectors represent speakers with opposing stances. These embedding are then used to divide the speakers into stance-partitions. We evaluate our method on three different datasets from different platforms. Our method outperforms or is comparable with supervised models while providing confidence levels for its output. Furthermore, we demonstrate how the structural embedding relate to the valence expressed by the speakers. Finally, we discuss some limitations inherent to the framework.

----

## [1252] ValueNet: A New Dataset for Human Value Driven Dialogue System

**Authors**: *Liang Qiu, Yizhou Zhao, Jinchao Li, Pan Lu, Baolin Peng, Jianfeng Gao, Song-Chun Zhu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i10.21368](https://doi.org/10.1609/aaai.v36i10.21368)

**Abstract**:

Building a socially intelligent agent involves many challenges, one of which is to teach the agent to speak guided by its value like a human. However, value-driven chatbots are still understudied in the area of dialogue systems. Most existing datasets focus on commonsense reasoning or social norm modeling. In this work, we present a new large-scale human value dataset called ValueNet, which contains human attitudes on 21,374 text scenarios. The dataset is organized in ten dimensions that conform to the basic human value theory in intercultural research. We further develop a Transformer-based value regression model on ValueNet to learn the utility distribution. Comprehensive empirical results show that the learned value model could benefit a wide range of dialogue tasks. For example, by teaching a generative agent with reinforcement learning and the rewards from the value model, our method attains state-of-the-art performance on the personalized dialog generation dataset: Persona-Chat. With values as additional features, existing emotion recognition models enable capturing rich human emotions in the context, which further improves the empathetic response generation performance in the EmpatheticDialogues dataset. To the best of our knowledge, ValueNet is the first large-scale text dataset for human value modeling, and we are the first one trying to incorporate a value model into emotionally intelligent dialogue systems. The dataset is available at https://liang-qiu.github.io/ValueNet/.

----

## [1253] Post-OCR Document Correction with Large Ensembles of Character Sequence-to-Sequence Models

**Authors**: *Juan Antonio Ramirez-Orta, Eduardo Xamena, Ana Gabriela Maguitman, Evangelos E. Milios, Axel J. Soto*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i10.21369](https://doi.org/10.1609/aaai.v36i10.21369)

**Abstract**:

In this paper, we propose a novel method to extend sequence-to-sequence models to accurately process sequences much longer than the ones used during training while being sample- and resource-efficient, supported by thorough experimentation. To investigate the effectiveness of our method, we apply it to the task of correcting documents already processed with Optical Character Recognition (OCR) systems using sequence-to-sequence models based on characters. We test our method on nine languages of the ICDAR 2019 competition on post-OCR text correction and achieve a new state-of-the-art performance in five of them. The strategy with the best performance involves splitting the input document in character n-grams and combining their individual corrections into the final output using a voting scheme that is equivalent to an ensemble of a large number of sequence models. We further investigate how to weigh the contributions from each one of the members of this ensemble. Our code for post-OCR correction is shared at https://github.com/jarobyte91/post_ocr_correction.

----

## [1254] MuMuQA: Multimedia Multi-Hop News Question Answering via Cross-Media Knowledge Extraction and Grounding

**Authors**: *Revanth Gangi Reddy, Xilin Rui, Manling Li, Xudong Lin, Haoyang Wen, Jaemin Cho, Lifu Huang, Mohit Bansal, Avirup Sil, Shih-Fu Chang, Alexander G. Schwing, Heng Ji*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i10.21370](https://doi.org/10.1609/aaai.v36i10.21370)

**Abstract**:

Recently, there has been an increasing interest in building question answering (QA) models that reason across multiple modalities, such as text and images. However, QA using images is often limited to just picking the answer from a pre-defined set of options. In addition, images in the real world, especially in news, have objects that are co-referential to the text, with complementary information from both modalities. In this paper, we present a new QA evaluation benchmark with 1,384 questions over news articles that require cross-media grounding of objects in images onto text. Specifically, the task involves multi-hop questions that require reasoning over image-caption pairs to identify the grounded visual object being referred to and then predicting a span from the news body text to answer the question. In addition, we introduce a novel multimedia data augmentation framework, based on cross-media knowledge extraction and synthetic question-answer generation, to automatically augment data that can provide weak supervision for this task. We evaluate both pipeline-based and end-to-end pretraining-based multimedia QA models on our benchmark, and show that they achieve promising performance, while considerably lagging behind human performance hence leaving large room for future work on this challenging new task.

----

## [1255] Pushing the Limits of Rule Reasoning in Transformers through Natural Language Satisfiability

**Authors**: *Kyle Richardson, Ashish Sabharwal*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i10.21371](https://doi.org/10.1609/aaai.v36i10.21371)

**Abstract**:

Investigating the reasoning abilities of transformer models, and discovering new challenging tasks for them, has been a topic of much interest. Recent studies have found these models to be surprisingly strong at performing deductive reasoning over formal logical theories expressed in natural language. A shortcoming of these studies, however, is that they do not take into account that logical theories, when sampled uniformly at random, do not necessarily lead to hard instances. We propose a new methodology for creating challenging algorithmic reasoning datasets that focus on natural language satisfiability (NLSat) problems. The key idea is to draw insights from empirical sampling of hard propositional SAT problems and from complexity-theoretic studies of language. This methodology allows us to distinguish easy from hard instances, and to systematically increase the complexity of existing reasoning benchmarks such as RuleTaker. We find that current transformers, given sufficient training data, are surprisingly robust at solving the resulting NLSat problems of substantially increased difficulty. They also exhibit some degree of scale-invariance—the ability to generalize to problems of larger size and scope. Our results, however, reveal important limitations too: careful sampling of training data is crucial for building models that generalize to larger problems, and transformer models’ limited scale-invariance suggests they are far from learning robust deductive reasoning algorithms.

----

## [1256] SFSRNet: Super-resolution for Single-Channel Audio Source Separation

**Authors**: *Joel Rixen, Matthias Renz*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i10.21372](https://doi.org/10.1609/aaai.v36i10.21372)

**Abstract**:

The problem of single-channel audio source separation is to recover (separate) multiple audio sources that are mixed in a single-channel audio signal (e.g. people talking over each other). Some of the best performing single-channel source separation methods utilize downsampling to either make the separation process faster or make the neural networks bigger and increase accuracy. The problem concerning downsampling is that it usually results in information loss. In this paper, we tackle this problem by introducing SFSRNet which contains a super-resolution (SR) network. The SR network is trained to reconstruct the missing information in the upper frequencies of the audio signal by operating on the spectrograms of the output audio source estimations and the input audio mixture. Any separation method where the length of the sequence is a bottleneck in speed and memory can be made faster or more accurate by using the SR network.
Based on the WSJ0-2mix benchmark where estimations of the audio signal of two speakers need to be extracted from the mixture, in our experiments our proposed SFSRNet reaches a scale-invariant signal-to-noise-ratio improvement (SI-SNRi) of 24.0 dB outperforming the state-of-the-art solution SepFormer which reaches an SI-SNRi of 22.3 dB.

----

## [1257] CEM: Commonsense-Aware Empathetic Response Generation

**Authors**: *Sahand Sabour, Chujie Zheng, Minlie Huang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i10.21373](https://doi.org/10.1609/aaai.v36i10.21373)

**Abstract**:

A key trait of daily conversations between individuals is the ability to express empathy towards others, and exploring ways to implement empathy is a crucial step towards human-like dialogue systems. Previous approaches on this topic mainly focus on detecting and utilizing the user’s emotion for generating empathetic responses. However, since empathy includes both aspects of affection and cognition, we argue that in addition to identifying the user’s emotion, cognitive understanding of the user’s situation should also be considered. To this end, we propose a novel approach for empathetic response generation, which leverages commonsense to draw more information about the user’s situation and uses this additional information to further enhance the empathy expression in generated responses. We evaluate our approach on EMPATHETICDIALOGUES, which is a widely-used benchmark dataset for empathetic response generation. Empirical results demonstrate that our approach outperforms the baseline models in both automatic and human evaluations and can generate more informative and empathetic responses. Our code is available at https://github.com/Sahandfer/CEM.

----

## [1258] Weakly Supervised Neuro-Symbolic Module Networks for Numerical Reasoning over Text

**Authors**: *Amrita Saha, Shafiq R. Joty, Steven C. H. Hoi*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i10.21374](https://doi.org/10.1609/aaai.v36i10.21374)

**Abstract**:

Neural Module Networks (NMNs) have been quite successful in incorporating explicit reasoning as learnable modules in various question answering tasks, including the most generic form of numerical reasoning over text in Machine Reading Comprehension (MRC). However to achieve this, contemporary Neural Module Networks models obtain strong supervision in form of specialized program annotation from the QA pairs through various heuristic parsing and exhaustive computation of all possible discrete operations on discrete arguments. Consequently they fail to generalize to more open-ended settings without such supervision. Hence, we propose Weakly Supervised Neuro-Symbolic Module Network (WNSMN) trained with answers as the sole supervision for numerical reasoning based MRC. WNSMN learns to execute a noisy heuristic program obtained from the dependency parse of the query, as discrete actions over both neural and symbolic reasoning modules and trains it end-to-end in a reinforcement learning framework with discrete reward from answer matching. On the subset of DROP having numerical answers, WNSMN outperforms NMN by 32% and the reasoning-free generative language model GenBERT by 8% in exact match accuracy under comparable weakly supervised settings. This showcases the effectiveness of modular networks that can handle explicit discrete reasoning over noisy programs in an end-to-end manner.

----

## [1259] Are Vision-Language Transformers Learning Multimodal Representations? A Probing Perspective

**Authors**: *Emmanuelle Salin, Badreddine Farah, Stéphane Ayache, Benoît Favre*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i10.21375](https://doi.org/10.1609/aaai.v36i10.21375)

**Abstract**:

In recent years, joint text-image embeddings have significantly improved thanks to the development of transformer-based Vision-Language models. Despite these advances, we still need to better understand the representations produced by those models. In this paper, we compare pre-trained and fine-tuned representations at a vision, language and multimodal level. To that end, we use a set of probing tasks to evaluate the performance of state-of-the-art Vision-Language models and introduce new datasets specifically for multimodal probing. These datasets are carefully designed to address a range of multimodal capabilities while minimizing the potential for models to rely on bias. Although the results confirm the ability of Vision-Language models to understand color at a multimodal level, the models seem to prefer relying on bias in text data for object position and size. On semantically adversarial examples, we find that those models are able to pinpoint fine-grained multimodal differences. Finally, we also notice that fine-tuning a Vision-Language model on multimodal tasks does not necessarily improve its multimodal ability. We make all datasets and code available to replicate experiments.

----

## [1260] Entailment Relation Aware Paraphrase Generation

**Authors**: *Abhilasha Sancheti, Balaji Vasan Srinivasan, Rachel Rudinger*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i10.21376](https://doi.org/10.1609/aaai.v36i10.21376)

**Abstract**:

We introduce a new task of entailment relation aware paraphrase generation which aims at generating a paraphrase conforming to a given entailment relation (e.g. equivalent, forward entailing, or reverse entailing) with respect to a given
input. We propose a reinforcement learning-based weakly-supervised paraphrasing system, ERAP, that can be trained using existing paraphrase and natural language inference (NLI) corpora without an explicit task-specific corpus. A combination of automated and human evaluations show that ERAP generates paraphrases conforming to the specified entailment relation and are of good quality as compared to the baselines and uncontrolled paraphrasing systems. Using ERAP for augmenting training data for downstream textual entailment task improves performance over an uncontrolled paraphrasing system, and introduces fewer training artifacts, indicating the benefit of explicit control during paraphrasing.

----

## [1261] Visual Definition Modeling: Challenging Vision & Language Models to Define Words and Objects

**Authors**: *Bianca Scarlini, Tommaso Pasini, Roberto Navigli*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i10.21377](https://doi.org/10.1609/aaai.v36i10.21377)

**Abstract**:

Architectures that model language and vision together havereceived much attention in recent years. Nonetheless, most tasks in this field focus on end-to-end applications without providing insights on whether it is the underlying semantics of visual objects or words that is captured. In this paper we draw on the established Definition Modeling paradigm and enhance it by grounding, for the first time, textual definitions to visual representations. We name this new task Visual Definition Modeling and put forward DEMETER and DIONYSUS, two benchmarks where, given an image as context, models have to generate a textual definition for a target being either i) a word that describes the image, or ii) an object patch therein. To measure the difficulty of our tasks we finetuned six different baselines and analyzed their performances, which show that a text-only encoder-decoder model is more effective than models pretrained for handling inputs of both modalities concurrently. This demonstrates the complexity of our benchmarks and encourages more research on text generation conditioned on multimodal inputs. The datasets for both benchmarks are available at https://github.com/SapienzaNLP/visual-definition-modeling as well as the code to reproduce our models.

----

## [1262] Active Learning on Pre-trained Language Model with Task-Independent Triplet Loss

**Authors**: *Seungmin Seo, Donghyun Kim, Youbin Ahn, Kyong-Ho Lee*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i10.21378](https://doi.org/10.1609/aaai.v36i10.21378)

**Abstract**:

Active learning attempts to maximize a task model’s performance gain by obtaining a set of informative samples from an unlabeled data pool. Previous active learning methods usually rely on specific network architectures or task-dependent sample acquisition algorithms. Moreover, when selecting a batch sample, previous works suffer from insufficient diversity of batch samples because they only consider the informativeness of each sample. This paper proposes a task-independent batch acquisition method using triplet loss to distinguish hard samples in an unlabeled data pool with similar features but difficult to identify labels. To assess the effectiveness of the proposed method, we compare the proposed method with state-of-the-art active learning methods on two tasks, relation extraction and sentence classification. Experimental results show that our method outperforms baselines on the benchmark datasets.

----

## [1263] OneRel: Joint Entity and Relation Extraction with One Module in One Step

**Authors**: *Yuming Shang, Heyan Huang, Xianling Mao*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i10.21379](https://doi.org/10.1609/aaai.v36i10.21379)

**Abstract**:

Joint entity and relation extraction is an essential task in natural language processing and knowledge graph construction. Existing approaches usually decompose the joint extraction task into several basic modules or processing steps to make it easy to conduct. However, such a paradigm ignores the fact that the three elements of a triple are interdependent and indivisible. Therefore, previous joint methods suffer from the problems of cascading errors and redundant information. To address these issues, in this paper, we propose a novel joint entity and relation extraction model, named OneRel, which casts joint extraction as a fine-grained triple classification problem. Specifically, our model consists of a scoring-based classifier and a relation-specific horns tagging strategy. The former evaluates whether a token pair and a relation belong to a factual triple. The latter ensures a simple but effective decoding process. Extensive experimental results on two widely used datasets demonstrate that the proposed method performs better than the state-of-the-art baselines, and delivers consistent performance gain on complex scenarios of various overlapping patterns and multiple triples.

----

## [1264] KATG: Keyword-Bias-Aware Adversarial Text Generation for Text Classification

**Authors**: *Lingfeng Shen, Shoushan Li, Ying Chen*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i10.21380](https://doi.org/10.1609/aaai.v36i10.21380)

**Abstract**:

Recent work has shown that current text classification models are vulnerable to small adversarial perturbation to inputs, and adversarial training that re-trains the models with the support of adversarial examples is the most popular way to alleviate the impact of the perturbation. However, current adversarial training methods have two principal problems: worse model generalization and ineffective defending against other text attacks. In this paper, we propose a Keyword-bias-aware Adversarial Text Generation model (KATG) that implicitly generates adversarial sentences using a generator-discriminator structure. Instead of using a benign sentence to generate an adversarial sentence, the KATG model utilizes extra multiple benign sentences (namely prior sentences) to guide adversarial sentence generation. Furthermore, to cover more perturbation used in existing attacks, a keyword-bias-aware sampling is proposed to select sentences containing biased words as prior sentences. Besides, to effectively utilize prior sentences, a generative flow mechanism is proposed to construct latent semantic space and learn a latent representation for the prior sentences. Experiments demonstrate that adversarial sentences generated by our KATG model can strengthen the victim model's robustness and generalization.

----

## [1265] Unsupervised Deep Keyphrase Generation

**Authors**: *Xianjie Shen, Yinghan Wang, Rui Meng, Jingbo Shang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i10.21381](https://doi.org/10.1609/aaai.v36i10.21381)

**Abstract**:

Keyphrase generation aims to summarize long documents with a collection of salient phrases. Deep neural models have demonstrated remarkable success in this task, with the capability of predicting keyphrases that are even absent from a document. However, such abstractiveness is acquired at the expense of a substantial amount of annotated data. In this paper, we present a novel method for keyphrase generation, AutoKeyGen, without the supervision of any annotated doc-keyphrase pairs. Motivated by the observation that an absent keyphrase in a document may appear in other places, in whole or in part, we construct a phrase bank by pooling all phrases extracted from a corpus.  With this phrase bank, we assign phrase candidates to new documents by a simple partial matching algorithm, and then we rank these candidates by their relevance to the document from both lexical and semantic perspectives. Moreover, we bootstrap a deep generative model using these top-ranked pseudo keyphrases to produce more absent candidates. Extensive experiments demonstrate that AutoKeyGen outperforms all unsupervised baselines and can even beat a strong supervised method in certain cases.

----

## [1266] Generation-Focused Table-Based Intermediate Pre-training for Free-Form Question Answering

**Authors**: *Peng Shi, Patrick Ng, Feng Nan, Henghui Zhu, Jun Wang, Jiarong Jiang, Alexander Hanbo Li, Rishav Chakravarti, Donald Weidner, Bing Xiang, Zhiguo Wang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i10.21382](https://doi.org/10.1609/aaai.v36i10.21382)

**Abstract**:

Question answering over semi-structured tables has attracted significant attention in the NLP community.
 However, most of the existing work focus on questions that can be answered with short-form answer, i.e. the answer is often a table cell or aggregation of multiple cells. 
 This can mismatch with the intents of users who want to ask more complex questions that require free-form answers such as explanations. 
 To bridge the gap, most recently, pre-trained sequence-to-sequence language models such as T5 are used for generating free-form answers based on the question and table inputs. 
 However, these pre-trained language models have weaker encoding abilities over table cells and schema. 
 To mitigate this issue, in this work, we present an intermediate pre-training framework, Generation-focused Table-based Intermediate Pre-training (GENTAP), that jointly learns representations of natural language questions and tables.
 GENTAP learns to generate via two training objectives to enhance the question understanding and table representation abilities for complex questions. 
 Based on experimental results, models that leverage GENTAP framework outperform the existing baselines on FETAQA benchmark. 
 The pre-trained models are not only useful for free-form question answering, but also for few-shot data-to-text generation task, thus showing good transfer ability by obtaining new state-of-the-art results.

----

## [1267] StepGame: A New Benchmark for Robust Multi-Hop Spatial Reasoning in Texts

**Authors**: *Zhengxiang Shi, Qiang Zhang, Aldo Lipani*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i10.21383](https://doi.org/10.1609/aaai.v36i10.21383)

**Abstract**:

Inferring spatial relations in natural language is a crucial ability an intelligent system should possess. The bAbI dataset tries to capture tasks relevant to this domain (task 17 and 19). However, these tasks have several limitations. Most importantly, they are limited to fixed expressions, they are limited in the number of reasoning steps required to solve them, and they fail to test the robustness of models to input that contains irrelevant or redundant information. In this paper, we present a new Question-Answering dataset called StepGame for robust multi-step spatial reasoning in texts. Our experiments demonstrate that state-of-the-art models on the bAbI dataset struggle on the StepGame dataset. Moreover, we propose a Tensor-Product based Memory-Augmented Neural Network (TP-MANN) specialized for spatial reasoning tasks. Experimental results on both datasets show that our model outperforms all the baselines with superior generalization and robustness performance.

----

## [1268] MINIMAL: Mining Models for Universal Adversarial Triggers

**Authors**: *Yaman Kumar Singla, Swapnil Parekh, Somesh Singh, Changyou Chen, Balaji Krishnamurthy, Rajiv Ratn Shah*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i10.21384](https://doi.org/10.1609/aaai.v36i10.21384)

**Abstract**:

It is well known that natural language models are vulnerable to adversarial attacks, which are mostly input-specific in nature. Recently, it has been shown that there also exist input-agnostic attacks in NLP models, called universal adversarial triggers. However, existing methods to craft universal triggers are data intensive. They require large amounts of data samples to generate adversarial triggers, which are typically inaccessible by attackers. For instance, previous works take 3000 data samples per class for the SNLI dataset to generate adversarial triggers. In this paper, we present a novel data-free approach, MINIMAL, to mine input-agnostic adversarial triggers from models. Using the triggers produced with our data-free algorithm, we reduce the accuracy of Stanford Sentiment Treebank’s positive class from 93.6% to 9.6%. Similarly, for the Stanford Natural LanguageInference (SNLI), our single-word trigger reduces the accuracy of the entailment class from 90.95% to less than 0.6%. Despite being completely data-free, we get equivalent accuracy drops as data-dependent methods

----

## [1269] Hierarchical Heterogeneous Graph Attention Network for Syntax-Aware Summarization

**Authors**: *Zixing Song, Irwin King*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i10.21385](https://doi.org/10.1609/aaai.v36i10.21385)

**Abstract**:

The task of summarization often requires a non-trivial understanding of the given text at the semantic level. In this work, we essentially incorporate the constituent structure into the single document summarization via the Graph Neural Networks to learn the semantic meaning of tokens. More specifically, we propose a novel hierarchical heterogeneous graph attention network over constituency-based parse trees for syntax-aware summarization. This approach reflects psychological findings that humans will pinpoint specific selection patterns to construct summaries hierarchically. Extensive experiments demonstrate that our model is effective for both the abstractive and extractive summarization tasks on five benchmark datasets from various domains. Moreover, further performance improvement can be obtained by virtue of state-of-the-art pre-trained models.

----

## [1270] Supervising Model Attention with Human Explanations for Robust Natural Language Inference

**Authors**: *Joe Stacey, Yonatan Belinkov, Marek Rei*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i10.21386](https://doi.org/10.1609/aaai.v36i10.21386)

**Abstract**:

Natural Language Inference (NLI) models are known to learn from biases and artefacts within their training data, impacting how well they generalise to other unseen datasets. Existing de-biasing approaches focus on preventing the models from learning these biases, which can result in restrictive models and lower performance. We instead investigate teaching the model how a human would approach the NLI task, in order to learn features that will generalise better to previously unseen examples. Using natural language explanations, we supervise the model’s attention weights to encourage more attention to be paid to the words present in the explanations, significantly improving model performance. Our experiments show that the in-distribution improvements of this method are also accompanied by out-of-distribution improvements, with the supervised models learning from features that generalise better to other NLI datasets. Analysis of the model indicates that human explanations encourage increased attention on the important words, with more attention paid to words in the premise and less attention paid to punctuation and stopwords.

----

## [1271] Hyperbolic Disentangled Representation for Fine-Grained Aspect Extraction

**Authors**: *Chang-Yu Tai, Ming-Yao Li, Lun-Wei Ku*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i10.21387](https://doi.org/10.1609/aaai.v36i10.21387)

**Abstract**:

Automatic identification of salient aspects from user reviews is especially useful for opinion analysis. There has been significant progress in utilizing weakly supervised approaches, which require only a small set of seed words for training aspect classifiers. However, there is always room for improvement. First, no weakly supervised approaches fully utilize latent hierarchies between words. Second, each seed word’s representation should have different latent semantics and be distinct when it represents a different aspect. In this paper we propose HDAE, a hyperbolic disentangled aspect extractor in which a hyperbolic aspect classifier captures words’ latent hierarchies, and an aspect-disentangled representation models the distinct latent semantics of each seed word. Compared to previous baselines, HDAE achieves average F1 performance gains of 18.2% and 24.1% on Amazon product review and restaurant review datasets, respectively. In addition, the embedding visualization experience demonstrates that HDAE is a more effective approach to leveraging seed words. An ablation study and a case study further attest the effectiveness of the proposed components.

----

## [1272] Procedural Text Understanding via Scene-Wise Evolution

**Authors**: *Jialong Tang, Hongyu Lin, Meng Liao, Yaojie Lu, Xianpei Han, Le Sun, Weijian Xie, Jin Xu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i10.21388](https://doi.org/10.1609/aaai.v36i10.21388)

**Abstract**:

Procedural text understanding requires machines to reason about entity states within the dynamical narratives. Current procedural text understanding approaches are commonly entity-wise, which separately track each entity and independently predict different states of each entity. Such an entity-wise paradigm does not consider the interaction between entities and their states. In this paper, we propose a new scene-wise paradigm for procedural text understanding, which jointly tracks states of all entities in a scene-by-scene manner. Based on this paradigm, we propose Scene Graph Reasoner (SGR), which introduces a series of dynamically evolving scene graphs to jointly formulate the evolution of entities, states and their associations throughout the narrative. In this way, the deep interactions between all entities and states can be jointly captured and simultaneously derived from scene graphs. Experiments show that SGR not only achieves the new state-of-the-art performance but also significantly accelerates the speed of reasoning.

----

## [1273] Debiasing NLU Models via Causal Intervention and Counterfactual Reasoning

**Authors**: *Bing Tian, Yixin Cao, Yong Zhang, Chunxiao Xing*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i10.21389](https://doi.org/10.1609/aaai.v36i10.21389)

**Abstract**:

Recent studies have shown that strong Natural Language Understanding (NLU) models are prone to relying on annotation biases of the datasets as a shortcut, which goes against the underlying mechanisms of the task of interest. To reduce such biases, several recent works introduce debiasing methods to regularize the training process of targeted NLU models. In this paper, we provide a new perspective with causal inference to find out the bias. On one hand, we show that there is an unobserved confounder for the natural language utterances and their respective classes, leading to spurious correlations from training data. To remove such confounder, the backdoor adjustment with causal intervention is utilized to find the true causal effect, which makes the training process fundamentally different from the traditional likelihood estimation. On the other hand, in inference process, we formulate the bias as the direct causal effect and remove it by pursuing the indirect causal effect with counterfactual reasoning. We conduct experiments on large-scale natural language inference and fact verification benchmarks, evaluating on bias sensitive datasets that are specifically designed to assess the robustness of models against known biases in the training data. Experimental results show that our proposed debiasing framework outperforms previous state-of-the-art debiasing methods while maintaining the original in-distribution performance.

----

## [1274] Chess as a Testbed for Language Model State Tracking

**Authors**: *Shubham Toshniwal, Sam Wiseman, Karen Livescu, Kevin Gimpel*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i10.21390](https://doi.org/10.1609/aaai.v36i10.21390)

**Abstract**:

Transformer language models have made tremendous strides in natural language understanding tasks. However, the complexity of natural language makes it challenging to ascertain how accurately these models are tracking the world state underlying the text. Motivated by this issue, we consider the task of language modeling for the game of chess. Unlike natural language, chess notations describe a simple, constrained, and deterministic domain. Moreover, we observe that the appropriate choice of chess notation allows for directly probing the world state, without requiring any additional probing-related machinery. We find that: (a) With enough training data, transformer language models can learn to track pieces and predict legal moves with high accuracy when trained solely on move sequences. (b) For small training sets providing access to board state information during training can yield significant improvements. (c) The success of transformer language models is dependent on access to the entire game history i.e. “full attention”. Approximating this full attention results in a significant performance drop. We propose this testbed as a benchmark for future work on the development and analysis of transformer language models.

----

## [1275] Contrast-Enhanced Semi-supervised Text Classification with Few Labels

**Authors**: *Austin Cheng-Yun Tsai, Sheng-Ya Lin, Li-Chen Fu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i10.21391](https://doi.org/10.1609/aaai.v36i10.21391)

**Abstract**:

Traditional text classification requires thousands of annotated data or an additional Neural Machine Translation (NMT) system, which are expensive to obtain in real applications. This paper presents a Contrast-Enhanced Semi-supervised Text Classification (CEST) framework under label-limited settings without incorporating any NMT systems. We propose a certainty-driven sample selection method and a contrast-enhanced similarity graph to utilize data more efficiently in self-training, alleviating the annotation-starving problem. The graph imposes a smoothness constraint on the unlabeled data to improve the coherence and the accuracy of pseudo-labels. Moreover, CEST formulates the training as a “learning from noisy labels” problem and performs the optimization accordingly. A salient feature of this formulation is the explicit suppression of the severe error propagation problem in conventional semi-supervised learning. With solely 30 labeled data per class for both training and validation dataset, CEST outperforms the previous state-of-the-art algorithms by 2.11% accuracy and only falls within the 3.04% accuracy range of fully-supervised pre-training language model fine-tuning on thousands of labeled data.

----

## [1276] Hybrid Autoregressive Inference for Scalable Multi-Hop Explanation Regeneration

**Authors**: *Marco Valentino, Mokanarangan Thayaparan, Deborah Ferreira, André Freitas*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i10.21392](https://doi.org/10.1609/aaai.v36i10.21392)

**Abstract**:

Regenerating natural language explanations in the scientific domain has been proposed as a benchmark to evaluate complex multi-hop and explainable inference. In this context, large language models can achieve state-of-the-art performance when employed as cross-encoder architectures and fine-tuned on human-annotated explanations. However, while much attention has been devoted to the quality of the explanations, the problem of performing inference efficiently is largely under studied. Cross-encoders, in fact, are intrinsically not scalable, possessing limited applicability to real-world scenarios that require inference on massive facts banks. To enable complex multi-hop reasoning at scale, this paper focuses on bi-encoder architectures, investigating the problem of scientific explanation regeneration at the intersection of dense and sparse models. Specifically, we present SCAR (for Scalable Autoregressive Inference), a hybrid framework that iteratively combines a Transformer-based bi-encoder with a sparse model of explanatory power, designed to leverage explicit inference patterns in the explanations. Our experiments demonstrate that the hybrid framework significantly outperforms previous sparse models, achieving performance comparable with that of state-of-the-art cross-encoders while being approx 50 times faster and scalable to corpora of millions of facts. Further analyses on semantic drift and multi-hop question answering reveal that the proposed hybridisation boosts the quality of the most challenging explanations, contributing to improved performance on downstream inference tasks.

----

## [1277] DetIE: Multilingual Open Information Extraction Inspired by Object Detection

**Authors**: *Michael Vasilkovsky, Anton Alekseev, Valentin Malykh, Ilya Shenbin, Elena Tutubalina, Dmitriy Salikhov, Mikhail Stepnov, Andrey Chertok, Sergey I. Nikolenko*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i10.21393](https://doi.org/10.1609/aaai.v36i10.21393)

**Abstract**:

State of the art neural methods for open information extraction (OpenIE) usually extract triplets (or tuples) iteratively in an autoregressive or predicate-based manner in order not to produce duplicates. In this work, we propose a different approach to the problem that can be equally or more successful. Namely, we present a novel single-pass method for OpenIE inspired by object detection algorithms from computer vision. We use an order-agnostic loss based on bipartite matching that forces unique predictions and a Transformer-based encoder-only architecture for sequence labeling. The proposed approach is faster and shows superior or similar performance in comparison with state of the art models on standard benchmarks in terms of both quality metrics and inference time. Our model sets the new state of the art performance of 67.7% F1 on CaRB evaluated as OIE2016 while being 3.35x faster at inference than previous state of the art. We also evaluate the multilingual version of our model in the zero-shot setting for two languages and introduce a strategy for generating synthetic multilingual data to fine-tune the model for each specific language. In this setting, we show performance improvement of 15% on multilingual Re-OIE2016, reaching 75% F1 for both Portuguese and Spanish languages. Code and models are available at https://github.com/sberbank-ai/DetIE.

----

## [1278] Hybrid Neural Networks for On-Device Directional Hearing

**Authors**: *Anran Wang, Maruchi Kim, Hao Zhang, Shyamnath Gollakota*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i10.21394](https://doi.org/10.1609/aaai.v36i10.21394)

**Abstract**:

On-device directional hearing requires audio source separation from a given direction while achieving stringent human-imperceptible latency requirements. While neural nets can achieve significantly better performance than traditional beamformers, all existing models fall short of supporting low-latency causal inference on computationally-constrained wearables. We present DeepBeam, a hybrid model that combines traditional beamformers with a custom lightweight neural net. The former reduces the computational burden of the latter and also improves its generalizability, while the latter is designed to further reduce the memory and computational overhead to enable real-time and low-latency operations. Our evaluation shows comparable performance to state-of-the-art causal inference models on synthetic data while achieving a 5x reduction of model size, 4x reduction of computation per second, 5x reduction in processing time and generalizing better to real hardware data. Further, our real-time hybrid model runs in 8 ms on mobile CPUs designed for low-power wearable devices and achieves an end-to-end latency of 17.5 ms.

----

## [1279] Non-parametric Online Learning from Human Feedback for Neural Machine Translation

**Authors**: *Dongqi Wang, Haoran Wei, Zhirui Zhang, Shujian Huang, Jun Xie, Jiajun Chen*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i10.21395](https://doi.org/10.1609/aaai.v36i10.21395)

**Abstract**:

We study the problem of online learning with human feedback in the human-in-the-loop machine translation, in which the human translators revise the machine-generated translations and then the corrected translations are used to improve the neural machine translation (NMT) system. However, previous methods require online model updating or additional translation memory networks to achieve high-quality performance, making them inflexible and inefficient in practice.
 In this paper, we propose a novel non-parametric online learning method without changing the model structure.
 This approach introduces two k-nearest-neighbor (KNN) modules: one module memorizes the human feedback, which is the correct sentences provided by human translators, 
 while the other balances the usage of the history human feedback and original NMT models adaptively. 
 Experiments conducted on EMEA and JRC-Acquis benchmarks demonstrate that our proposed method obtains substantial improvements on translation accuracy and achieves better adaptation performance with less repeating human correction operations.

----

## [1280] Parameter Differentiation Based Multilingual Neural Machine Translation

**Authors**: *Qian Wang, Jiajun Zhang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i10.21396](https://doi.org/10.1609/aaai.v36i10.21396)

**Abstract**:

Multilingual neural machine translation (MNMT) aims to translate multiple languages with a single model and has been proved successful thanks to effective knowledge transfer among different languages with shared parameters. However, it is still an open question which parameters should be shared and which ones need to be task-specific. Currently, the common practice is to heuristically design or search language-specific modules, which is difficult to find the optimal configuration. In this paper, we propose a novel parameter differentiation based method that allows the model to determine which parameters should be language-speciﬁc during training. Inspired by cellular differentiation, each shared parameter in our method can dynamically differentiate into more specialized types. We further deﬁne the differentiation criterion as inter-task gradient similarity. Therefore, parameters with conﬂicting inter-task gradients are more likely to be language-specific. Extensive experiments on multilingual datasets have demonstrated that our method signiﬁcantly outperforms various strong baselines with different parameter sharing conﬁgurations. Further analysis reveals that the parameter sharing configuration obtained by our method correlates well with the linguistic proximities.

----

## [1281] DisenCite: Graph-Based Disentangled Representation Learning for Context-Specific Citation Generation

**Authors**: *Yifan Wang, Yiping Song, Shuai Li, Chaoran Cheng, Wei Ju, Ming Zhang, Sheng Wang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i10.21397](https://doi.org/10.1609/aaai.v36i10.21397)

**Abstract**:

Citing and describing related literature are crucial to scientific writing. Many existing approaches show encouraging performance in citation recommendation, but are unable to accomplish the more challenging and onerous task of citation text generation. In this paper, we propose a novel disentangled representation based model DisenCite to automatically generate the citation text through integrating paper text and citation graph. A key novelty of our method compared with existing approaches is to generate context-specific citation text, empowering the generation of different types of citations for the same paper. In particular, we first build and make available a graph enhanced contextual citation dataset (GCite) with 25K edges in different types characterized by citation contained sections over 4.8K research papers. Based on this dataset, we encode each paper according to both textual contexts and structure information in the heterogeneous citation graph. The resulted paper representations are then disentangled by the mutual information regularization between this paper and its neighbors in graph. Extensive experiments demonstrate the superior performance of our method comparing to state-of-the-art approaches. We further conduct ablation and case studies to reassure that the improvement of our method comes from generating the context-specific citation through incorporating the citation graph.

----

## [1282] HEAL: A Knowledge Graph for Distress Management Conversations

**Authors**: *Anuradha Welivita, Pearl Pu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i10.21398](https://doi.org/10.1609/aaai.v36i10.21398)

**Abstract**:

The demands of the modern world are increasingly responsible for causing psychological burdens and bringing adverse impacts on our mental health. As a result, neural conversational agents with empathetic responding and distress management capabilities have recently gained popularity. However, existing end-to-end empathetic conversational agents often generate generic and repetitive empathetic statements such as "I am sorry to hear that", which fail to convey specificity to a given situation. Due to the lack of controllability in such models, they also impose the risk of generating toxic responses. Chatbots leveraging reasoning over knowledge graphs is seen as an efficient and fail-safe solution over end-to-end models. However, such resources are limited in the context of emotional distress. To address this, we introduce HEAL, a knowledge graph developed based on 1M distress narratives and their corresponding consoling responses curated from Reddit. It consists of 22K nodes identifying different types of stressors, speaker expectations, responses, and feedback types associated with distress dialogues and forms 104K connections between different types of nodes. Each node is associated with one of 41 affective states. Statistical and visual analysis conducted on HEAL reveals emotional dynamics between speakers and listeners in distress-oriented conversations and identifies useful response patterns leading to emotional relief. Automatic and human evaluation experiments show that HEAL's responses are more diverse, empathetic, and reliable compared to the baselines.

----

## [1283] Deep Fusing Pre-trained Models into Neural Machine Translation

**Authors**: *Rongxiang Weng, Heng Yu, Weihua Luo, Min Zhang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i10.21399](https://doi.org/10.1609/aaai.v36i10.21399)

**Abstract**:

Pre-training and fine-tuning have become the de facto paradigm in many natural language processing (NLP) tasks. However, compared to other NLP tasks, neural machine translation (NMT) aims to generate target language sentences through the contextual representation from the source language counterparts. This characteristic means the optimization objective of NMT is far from that of the universal pre-trained models (PTMs), leading to the standard procedure of pre-training and fine-tuning does not work well in NMT. In this paper, we propose a novel framework to deep fuse the pre-trained representation into NMT, fully exploring the potential of PTMs in NMT. Specifically, we directly replace the randomly initialized Transformer encoder with a pre-trained encoder and propose a layer-wise coordination structure to coordinate PTM and NMT decoder learning. Then, we introduce a partitioned multi-task learning method to fine-tune the pre-trained parameter, reducing the gap between PTM and NMT by progressively learning the task-specific representation. Experimental results show that our approach achieves considerable improvements on WMT14 En2De, WMT14 En2Fr, and WMT16 Ro2En translation benchmarks and outperforms previous work in both autoregressive and non-autoregressive NMT models.

----

## [1284] VAST: The Valence-Assessing Semantics Test for Contextualizing Language Models

**Authors**: *Robert Wolfe, Aylin Caliskan*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i10.21400](https://doi.org/10.1609/aaai.v36i10.21400)

**Abstract**:

We introduce VAST, the Valence-Assessing Semantics Test, a novel intrinsic evaluation task for contextualized word embeddings (CWEs). Despite the widespread use of contextualizing language models (LMs), researchers have no intrinsic evaluation task for understanding the semantic quality of CWEs and their unique properties as related to contextualization, the change in the vector representation of a word based on surrounding words; tokenization, the breaking of uncommon words into subcomponents; and LM-specific geometry learned during training. VAST uses valence, the association of a word with pleasantness, to measure the correspondence of word-level LM semantics with widely used human judgments, and examines the effects of contextualization, tokenization, and LM-specific geometry. Because prior research has found that CWEs from OpenAI's 2019 English-language causal LM GPT-2 perform poorly on other intrinsic evaluations, we select GPT-2 as our primary subject, and include results showing that VAST is useful for 7 other LMs, and can be used in 7 languages. GPT-2 results show that the semantics of a word are more similar to the semantics of context in layers closer to model output, such that VAST scores diverge between our contextual settings, ranging from Pearson’s rho of .55 to .77 in layer 11. We also show that multiply tokenized words are not semantically encoded until layer 8, where they achieve Pearson’s rho of .46, indicating the presence of an encoding process for multiply tokenized words which differs from that of singly tokenized words, for which rho is highest in layer 0. We find that a few neurons with values having greater magnitude than the rest mask word-level semantics in GPT-2’s top layer, but that word-level semantics can be recovered by nullifying non-semantic principal components: Pearson’s rho in the top layer improves from .32 to .76. Downstream POS tagging and sentence classification experiments indicate that the GPT-2 uses these principal components for non-semantic purposes, such as to represent sentence-level syntax relevant to next-word prediction. After isolating semantics, we show the utility of VAST for understanding LM semantics via improvements over related work on four word similarity tasks, with a score of .50 on SimLex-999, better than the previous best of .45 for GPT-2. Finally, we show that 8 of 10 WEAT bias tests, which compare differences in word embedding associations between groups of words, exhibit more stereotype-congruent biases after isolating semantics, indicating that non-semantic structures in LMs also mask social biases.

----

## [1285] A Label Dependence-Aware Sequence Generation Model for Multi-Level Implicit Discourse Relation Recognition

**Authors**: *Changxing Wu, Liuwen Cao, Yubin Ge, Yang Liu, Min Zhang, Jinsong Su*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i10.21401](https://doi.org/10.1609/aaai.v36i10.21401)

**Abstract**:

Implicit discourse relation recognition (IDRR) is a challenging but crucial task in discourse analysis. Most existing methods train multiple models to predict multi-level labels independently, while ignoring the dependence between hierarchically structured labels. In this paper, we consider multi-level IDRR as a conditional label sequence generation task and propose a Label Dependence-aware Sequence Generation Model (LDSGM) for it. Specifically, we first design a label attentive encoder to learn the global representation of an input instance and its level-specific contexts, where the label dependence is integrated to obtain better label embeddings. Then, we employ a label sequence decoder to output the predicted labels in a top-down manner, where the predicted higher-level labels are directly used to guide the label prediction at the current level. We further develop a mutual learning enhanced training method to exploit the label dependence in a bottom-up direction, which is captured by an auxiliary decoder introduced during training. Experimental results on the PDTB dataset show that our model achieves the state-of-the-art performance on multi-level IDRR. We release our code at https://github.com/nlpersECJTU/LDSGM.

----

## [1286] Fast and Constrained Absent Keyphrase Generation by Prompt-Based Learning

**Authors**: *Huanqin Wu, Baijiaxin Ma, Wei Liu, Tao Chen, Dan Nie*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i10.21402](https://doi.org/10.1609/aaai.v36i10.21402)

**Abstract**:

Generating absent keyphrases, which do not appear in the input document, is challenging in the keyphrase prediction task. Most previous works treat the problem as an autoregressive sequence-to-sequence generation task, which demonstrates promising results for generating grammatically correct and fluent absent keyphrases. However, such an end-to-end process with a complete data-driven manner is unconstrained, which is prone to generate keyphrases inconsistent with the input document. In addition, the existing autoregressive decoding method makes the generation of keyphrases must be done from left to right, leading to slow speed during inference. In this paper, we propose a constrained absent keyphrase generation method in a prompt-based learning fashion. Specifically, the prompt will be created firstly based on the keywords, which are defined as the overlapping words between absent keyphrase and document. Then, a mask-predict decoder is used to complete the absent keyphrase on the constraint of prompt. Experiments on keyphrase generation benchmarks have demonstrated the effectiveness of our approach. In addition, we evaluate the performance of constrained absent keyphrases generation from an information retrieval perspective. The result shows that our approach can generate more consistent keyphrases, which can improve document retrieval performance. What’s more, with a non-autoregressive decoding manner, our model can speed up the absent keyphrase generation by 8.67× compared with the autoregressive method.

----

## [1287] GraphMemDialog: Optimizing End-to-End Task-Oriented Dialog Systems Using Graph Memory Networks

**Authors**: *Jie Wu, Ian G. Harris, Hongzhi Zhao*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i10.21403](https://doi.org/10.1609/aaai.v36i10.21403)

**Abstract**:

Effectively integrating knowledge into end-to-end task-oriented dialog systems remains a challenge. It typically requires incorporation of an external knowledge base (KB) and capture of the intrinsic semantics of the dialog history. Recent research shows promising results by using Sequence-to-Sequence models, Memory Networks, and even Graph Convolutional Networks. However, current state-of-the-art models are less effective at integrating dialog history and KB into task-oriented dialog systems in the following ways: 1. The KB representation is not fully context-aware. The dynamic interaction between the dialog history and KB is seldom explored. 2. Both the sequential and structural information in the dialog history can contribute to capturing the dialog semantics, but they are not studied concurrently. In this paper, we propose a novel Graph Memory Network (GMN) based Seq2Seq model, GraphMemDialog, to effectively learn the inherent structural information hidden in dialog history, and to model the dynamic interaction between dialog history and KBs. We adopt a modified graph attention network to learn the rich structural representation of the dialog history, whereas the context-aware representation of KB entities are learnt by our novel GMN. To fully exploit this dynamic interaction, we design a learnable memory controller coupled with external KB entity memories to recurrently incorporate dialog history context into KB entities through a multi-hop reasoning mechanism. Experiments on three public datasets show that our GraphMemDialog model achieves state-of-the-art performance and outperforms strong baselines by a large margin, especially on datatests with more complicated KB information.

----

## [1288] Mastering the Explicit Opinion-Role Interaction: Syntax-Aided Neural Transition System for Unified Opinion Role Labeling

**Authors**: *Shengqiong Wu, Hao Fei, Fei Li, Meishan Zhang, Yijiang Liu, Chong Teng, Donghong Ji*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i10.21404](https://doi.org/10.1609/aaai.v36i10.21404)

**Abstract**:

Unified opinion role labeling (ORL) aims to detect all possible opinion structures of 'opinion-holder-target' in one shot, given a text. The existing transition-based unified method, unfortunately, is subject to longer opinion terms and fails to solve the term overlap issue. Current top performance has been achieved by employing the span-based graph model, which however still suffers from both high model complexity and insufficient interaction among opinions and roles. In this work, we investigate a novel solution by revisiting the transition architecture, and augmenting it with a pointer network (PointNet). The framework parses out all opinion structures in linear-time complexity, meanwhile breaks through the limitation of any length of terms with PointNet. To achieve the explicit opinion-role interactions, we further propose a unified dependency-opinion graph (UDOG), co-modeling the syntactic dependency structure and the partial opinion-role structure. We then devise a relation-centered graph aggregator (RCGA) to encode the multi-relational UDOG, where the resulting high-order representations are used to promote the predictions in the vanilla transition system. Our model achieves new state-of-the-art results on the MPQA benchmark. Analyses further demonstrate the superiority of our methods on both efficacy and efficiency.

----

## [1289] A Graph Convolutional Network with Adaptive Graph Generation and Channel Selection for Event Detection

**Authors**: *Zhipeng Xie, Yumin Tu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i10.21405](https://doi.org/10.1609/aaai.v36i10.21405)

**Abstract**:

Graph convolutional networks have been successfully applied to the task of event detection. However, existing works rely heavily on a fixed syntactic parse tree structure from an external parser. In addition, the information content extracted for aggregation is determined simply by the (syntactic) edge direction or type but irrespective of what semantics the vertices have, which is somewhat rigid. With this work, we propose a novel graph convolutional method that combines an adaptive graph generation technique and a multi-channel selection strategy. The adaptive graph generation technique enables the gradients to pass through the graph sampling layer by using the ST-Gumbel-Softmax trick. The multi-channel selection strategy allows two adjacent vertices to automatically determine which information channels to get through for information extraction and aggregation. The proposed method achieves the state-of-the-art performance on ACE2005 dataset.

----

## [1290] Leashing the Inner Demons: Self-Detoxification for Language Models

**Authors**: *Canwen Xu, Zexue He, Zhankui He, Julian J. McAuley*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i10.21406](https://doi.org/10.1609/aaai.v36i10.21406)

**Abstract**:

Language models (LMs) can reproduce (or amplify) toxic language seen during training, which poses a risk to their practical application. In this paper, we conduct extensive experiments to study this phenomenon. We analyze the impact of prompts, decoding strategies and training corpora on the output toxicity. Based on our findings, we propose a simple yet effective unsupervised method for language models to ``detoxify'' themselves without an additional large corpus or external discriminator. Compared to a supervised baseline, our proposed method shows better toxicity reduction with good generation quality in the generated content under multiple settings. Warning: some examples shown in the paper may contain uncensored offensive content.

----

## [1291] Zero-Shot Cross-Lingual Machine Reading Comprehension via Inter-sentence Dependency Graph

**Authors**: *Liyan Xu, Xuchao Zhang, Bo Zong, Yanchi Liu, Wei Cheng, Jingchao Ni, Haifeng Chen, Liang Zhao, Jinho D. Choi*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i10.21407](https://doi.org/10.1609/aaai.v36i10.21407)

**Abstract**:

We target the task of cross-lingual Machine Reading Comprehension (MRC) in the direct zero-shot setting, by incorporating syntactic features from Universal Dependencies (UD), and the key features we use are the syntactic relations within each sentence. While previous work has demonstrated effective syntax-guided MRC models, we propose to adopt the inter-sentence syntactic relations, in addition to the rudimentary intra-sentence relations, to further utilize the syntactic dependencies in the multi-sentence input of the MRC task. In our approach, we build the Inter-Sentence Dependency Graph (ISDG) connecting dependency trees to form global syntactic relations across sentences. We then propose the ISDG encoder that encodes the global dependency graph, addressing the inter-sentence relations via both one-hop and multi-hop dependency paths explicitly. Experiments on three multilingual MRC datasets (XQuAD, MLQA, TyDiQA-GoldP) show that our encoder that is only trained on English is able to improve the zero-shot performance on all 14 test sets covering 8 languages, with up to 3.8 F1 / 5.2 EM improvement on-average, and 5.2 F1 / 11.2 EM on certain languages. Further analysis shows the improvement can be attributed to the attention on the cross-linguistically consistent syntactic path. Our code is available at https://github.com/lxucs/multilingual-mrc-isdg.

----

## [1292] From Dense to Sparse: Contrastive Pruning for Better Pre-trained Language Model Compression

**Authors**: *Runxin Xu, Fuli Luo, Chengyu Wang, Baobao Chang, Jun Huang, Songfang Huang, Fei Huang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i10.21408](https://doi.org/10.1609/aaai.v36i10.21408)

**Abstract**:

Pre-trained Language Models (PLMs) have achieved great success in various Natural Language Processing (NLP) tasks under the pre-training and fine-tuning paradigm. 
 With large quantities of parameters, PLMs are computation-intensive and resource-hungry. Hence, model pruning has been introduced to compress large-scale PLMs. 
 However, most prior approaches only consider task-specific knowledge towards downstream tasks, but ignore the essential task-agnostic knowledge during pruning, which may cause catastrophic forgetting problem and lead to poor generalization ability. 
 To maintain both task-agnostic and task-specific knowledge in our pruned model, we propose ContrAstive Pruning (CAP) under the paradigm of pre-training and fine-tuning. 
 It is designed as a general framework, compatible with both structured and unstructured pruning. 
 Unified in contrastive learn- ing, CAP enables the pruned model to learn from the pre-trained model for task-agnostic knowledge, and fine-tuned model for task-specific knowledge. 
 Besides, to better retain the performance of the pruned model, the snapshots (i.e., the intermediate models at each pruning iteration) also serve as effective supervisions for pruning. 
 Our extensive experiments show that adopting CAP consistently yields significant improvements, especially in extremely high sparsity scenarios. 
 With only 3% model parameters reserved (i.e., 97% sparsity), CAP successfully achieves 99.2% and 96.3% of the original BERT performance in QQP and MNLI tasks. 
 In addition, our probing experiments demonstrate that the model pruned by CAP tends to achieve better generalization ability.

----

## [1293] Sequence Level Contrastive Learning for Text Summarization

**Authors**: *Shusheng Xu, Xingxing Zhang, Yi Wu, Furu Wei*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i10.21409](https://doi.org/10.1609/aaai.v36i10.21409)

**Abstract**:

Contrastive learning models have achieved great success in unsupervised visual representation learning, which maximize the similarities between feature representations of different views of the same image, while minimize the similarities between feature representations of views of different images. In text summarization, the output summary is a shorter form of the input document and they have similar meanings. In this paper, we propose a contrastive learning model for supervised abstractive text summarization, where we view a document, its gold summary and its model generated summaries as different views of the same mean representation and maximize the similarities between them during training. We improve over a strong sequence-to-sequence text generation model (i.e., BART) on three different summarization datasets. Human evaluation also shows that our model achieves better faithfulness ratings compared to its counterpart without contrastive objectives. We release our code at https://github.com/xssstory/SeqCo.

----

## [1294] Self-Supervised Knowledge Assimilation for Expert-Layman Text Style Transfer

**Authors**: *Wenda Xu, Michael Saxon, Misha Sra, William Yang Wang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i10.21410](https://doi.org/10.1609/aaai.v36i10.21410)

**Abstract**:

Expert-layman text style transfer technologies have the potential to improve communication between members of scientific communities and the general public. High-quality information produced by experts is often filled with difficult jargon laypeople struggle to understand. This is a particularly notable issue in the medical domain, where layman are often confused by medical text online. At present, two bottlenecks interfere with the goal of building high-quality medical expert-layman style transfer systems: a dearth of pretrained  medical-domain language models spanning both expert and layman terminologies and a lack of parallel corpora for training the transfer task itself. To mitigate the first issue, we propose a novel language model (LM) pretraining task, Knowledge Base Assimilation, to synthesize pretraining data from the edges of a graph of expert- and layman-style medical terminology terms into an LM during self-supervised learning. To mitigate the second issue, we build a large-scale parallel corpus in the medical expert-layman domain using a margin-based criterion. Our experiments show that transformer-based models pretrained on knowledge base assimilation and other well-established pretraining tasks fine-tuning on our new parallel corpus leads to considerable improvement against expert-layman transfer benchmarks, gaining an average relative improvement of our human evaluation, the Overall Success Rate (OSR), by 106%.

----

## [1295] Text Is No More Enough! A Benchmark for Profile-Based Spoken Language Understanding

**Authors**: *Xiao Xu, Libo Qin, Kaiji Chen, Guoxing Wu, Linlin Li, Wanxiang Che*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i10.21411](https://doi.org/10.1609/aaai.v36i10.21411)

**Abstract**:

Current researches on spoken language understanding (SLU) heavily are limited to a simple setting: the plain text-based SLU that takes the user utterance as input and generates its corresponding semantic frames (e.g., intent and slots). Unfortunately, such a simple setting may fail to work in complex real-world scenarios when an utterance is semantically ambiguous, which cannot be achieved by the text-based SLU models. In this paper, we first introduce a new and important task, Profile-based Spoken Language Understanding (ProSLU), which requires the model that not only relies on the plain text but also the supporting profile information to predict the correct intents and slots. To this end, we further introduce a large-scale human-annotated Chinese dataset with over 5K utterances and their corresponding supporting profile information (Knowledge Graph (KG), User Profile (UP), Context Awareness (CA)). In addition, we evaluate several state-of-the-art baseline models and explore a multi-level knowledge adapter to effectively incorporate profile information. Experimental results reveal that all existing text-based SLU models fail to work when the utterances are semantically ambiguous and our proposed framework can effectively fuse the supporting information for sentence-level intent detection and token-level slot filling. Finally, we summarize key challenges and provide new points for future directions, which hopes to facilitate the research.

----

## [1296] SAS: Self-Augmentation Strategy for Language Model Pre-training

**Authors**: *Yifei Xu, Jingqiao Zhang, Ru He, Liangzhu Ge, Chao Yang, Cheng Yang, Ying Nian Wu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i10.21412](https://doi.org/10.1609/aaai.v36i10.21412)

**Abstract**:

The core of self-supervised learning for pre-training language models includes pre-training task design as well as appropriate data augmentation. Most data augmentations in language model pre-training are context-independent. A seminal contextualized augmentation was recently proposed in ELECTRA and achieved state-of-the-art performance by introducing an auxiliary generation network (generator) to produce contextualized data augmentation for the training of a main discrimination network (discriminator). This design, however, introduces extra computation cost of the generator and a need to adjust the relative capability between the generator and the discriminator. In this paper, we propose a self-augmentation strategy (SAS) where a single network is utilized for both regular pre-training and contextualized data augmentation for the training in later epochs. Essentially, this strategy eliminates a separate generator and uses the single network to jointly conduct two pre-training tasks with MLM (Masked Language Modeling) and RTD (Replaced Token Detection) heads. It avoids the challenge to search for an appropriate size of the generator, which is critical to the performance as evidenced in ELECTRA and its subsequent variant models. In addition, SAS is a general strategy that can be seamlessly combined with many new techniques emerging recently or in the future, such as the disentangled attention mechanism from DeBERTa. Our experiments show that SAS is able to outperform ELECTRA and other state-of-the-art models in the GLUE tasks with similar or less computation cost.

----

## [1297] Hybrid Curriculum Learning for Emotion Recognition in Conversation

**Authors**: *Lin Yang, Yi Shen, Yue Mao, Longjun Cai*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i10.21413](https://doi.org/10.1609/aaai.v36i10.21413)

**Abstract**:

Emotion recognition in conversation (ERC) aims to detect the emotion label for each utterance. 
Motivated by recent studies which have proven that feeding training examples in a meaningful order rather than considering them randomly can boost the performance of models, we propose an ERC-oriented hybrid curriculum learning framework. Our framework consists of two curricula: (1) conversation-level curriculum (CC); and (2) utterance-level curriculum (UC). In CC, we construct a difficulty measurer based on ``emotion shift'' frequency within a conversation, then the conversations are scheduled in an ``easy to hard" schema according to the difficulty score returned by the difficulty measurer. For UC, it is implemented from an emotion-similarity perspective, which progressively strengthens the model’s ability in identifying the confusing emotions. With the proposed model-agnostic hybrid curriculum learning strategy, we observe significant performance boosts over a wide range of existing ERC models and we are able to achieve new state-of-the-art results on four public ERC datasets.

----

## [1298] NumHTML: Numeric-Oriented Hierarchical Transformer Model for Multi-Task Financial Forecasting

**Authors**: *Linyi Yang, Jiazheng Li, Ruihai Dong, Yue Zhang, Barry Smyth*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i10.21414](https://doi.org/10.1609/aaai.v36i10.21414)

**Abstract**:

Financial forecasting has been an important and active area of machine learning research because of the challenges it presents and the potential rewards that even minor improvements in prediction accuracy or forecasting may entail. Traditionally, financial forecasting has heavily relied on quantitative indicators and metrics derived from structured financial statements. Earnings conference call data, including text and audio, is an important source of unstructured data that has been used for various prediction tasks using deep earning and related approaches. However, current deep learning-based methods are limited in the way that they deal with numeric data; numbers are typically treated as plain-text tokens without taking advantage of their underlying numeric structure. This paper describes a numeric-oriented hierarchical transformer model (NumHTML) to predict stock returns, and financial risk using multi-modal aligned earnings calls data by taking advantage of the different categories of numbers (monetary, temporal, percentages etc.) and their magnitude. We present the results of a comprehensive evaluation of NumHTML against several state-of-the-art baselines using a real-world publicly available dataset. The results indicate that NumHTML significantly outperforms the current state-of-the-art across a variety of evaluation metrics and that it has the potential to offer significant financial gains in a practical trading context.

----

## [1299] Tracing Text Provenance via Context-Aware Lexical Substitution

**Authors**: *Xi Yang, Jie Zhang, Kejiang Chen, Weiming Zhang, Zehua Ma, Feng Wang, Nenghai Yu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i10.21415](https://doi.org/10.1609/aaai.v36i10.21415)

**Abstract**:

Text content created by humans or language models is often stolen or misused by adversaries. Tracing text provenance can help claim the ownership of text content or identify the malicious users who distribute misleading content like machine-generated fake news. There have been some attempts to achieve this, mainly based on watermarking techniques. Specifically, traditional text watermarking methods embed watermarks by slightly altering text format like line spacing and font, which, however, are fragile to cross-media transmissions like OCR. Considering this, natural language watermarking methods represent watermarks by replacing words in original sentences with synonyms from handcrafted lexical resources (e.g., WordNet), but they do not consider the substitution’s impact on the overall sentence's meaning. Recently, a transformer-based network was proposed to embed watermarks by modifying the unobtrusive words (e.g., function words), which also impair the sentence's logical and semantic coherence. Besides, one well-trained network fails on other different types of text content.
 To address the limitations mentioned above, we propose a natural language watermarking scheme based on context-aware lexical substitution (LS). Specifically, we employ BERT to suggest LS candidates by inferring the semantic relatedness between the candidates and the original sentence. Based on this, a selection strategy in terms of synchronicity and substitutability is further designed to test whether a word is exactly suitable for carrying the watermark signal. Extensive experiments demonstrate that, under both objective and subjective metrics, our watermarking scheme can well preserve the semantic integrity of original sentences and has a better transferability than existing methods. Besides, the proposed LS approach outperforms the state-of-the-art approach on the Stanford Word Substitution Benchmark.

----

## [1300] Fusing Task-Oriented and Open-Domain Dialogues in Conversational Agents

**Authors**: *Tom Young, Frank Xing, Vlad Pandelea, Jinjie Ni, Erik Cambria*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i10.21416](https://doi.org/10.1609/aaai.v36i10.21416)

**Abstract**:

The goal of building intelligent dialogue systems has largely been separately pursued under two paradigms: task-oriented dialogue (TOD) systems, which perform task-specific functions, and open-domain dialogue (ODD) systems, which focus on non-goal-oriented chitchat. The two dialogue modes can potentially be intertwined together seamlessly in the same conversation, as easily done by a friendly human assistant. Such ability is desirable in conversational agents, as the integration makes them more accessible and useful. Our paper addresses this problem of fusing TODs and ODDs in multi-turn dialogues. Based on the popular TOD dataset MultiWOZ, we build a new dataset FusedChat, by rewriting the existing TOD turns and adding new ODD turns. This procedure constructs conversation sessions containing exchanges from both dialogue modes. It features inter-mode contextual dependency, i.e., the dialogue turns from the two modes depend on each other. Rich dependency patterns such as co-reference and ellipsis are included. The new dataset, with 60k new human-written ODD turns and 5k re-written TOD turns, offers a benchmark to test a dialogue model's ability to perform inter-mode conversations. This is a more challenging task since the model has to determine the appropriate dialogue mode and generate the response based on the inter-mode context. However, such models would better mimic human-level conversation capabilities. We evaluate two baseline models on this task, including the classification-based two-stage models and the two-in-one fused models. We publicly release FusedChat and the baselines to propel future work on inter-mode dialogue systems.

----

## [1301] JAKET: Joint Pre-training of Knowledge Graph and Language Understanding

**Authors**: *Donghan Yu, Chenguang Zhu, Yiming Yang, Michael Zeng*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i10.21417](https://doi.org/10.1609/aaai.v36i10.21417)

**Abstract**:

Knowledge graphs (KGs) contain rich information about world knowledge, entities, and relations. Thus, they can be great supplements to existing pre-trained language models. However, it remains a challenge to efficiently integrate information from KG into language modeling. And the understanding of a knowledge graph requires related context. We propose a novel joint pre-training framework, JAKET, to model both the knowledge graph and language. The knowledge module and language module provide essential information to mutually assist each other: the knowledge module produces embeddings for entities in text while the language module generates context-aware initial embeddings for entities and relations in the graph. Our design enables the pre-trained model to easily adapt to unseen knowledge graphs in new domains. Experiment results on several knowledge-aware NLP tasks show that our proposed framework achieves superior performance by effectively leveraging knowledge in language understanding.

----

## [1302] KID-Review: Knowledge-Guided Scientific Review Generation with Oracle Pre-training

**Authors**: *Weizhe Yuan, Pengfei Liu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i10.21418](https://doi.org/10.1609/aaai.v36i10.21418)

**Abstract**:

The surge in the number of scientific submissions has brought challenges to the work of peer review. In this paper, as a first step, we explore the possibility of designing an automated system, which is not meant to replace humans, but rather providing a first-pass draft for a machine-assisted human review process. Specifically, we present an end-to-end knowledge-guided review generation framework for scientific papers grounded in cognitive psychology research that a better understanding of text requires different types of knowledge. In practice, we found that this seemingly intuitive idea suffered from training difficulties. In order to solve this problem, we put forward an oracle pre-training strategy, which can not only make the Kid-Review better educated but also make the generated review cover more aspects. Experimentally, we perform a comprehensive evaluation (human and automatic) from different perspectives. Empirical results have shown the effectiveness of different types of knowledge as well as oracle pre-training. We make all code, relevant dataset available: https://github.com/Anonymous4nlp233/KIDReview as well as the Kid-Review system: http://nlpeer.reviews.

----

## [1303] Reference-Based Speech Enhancement via Feature Alignment and Fusion Network

**Authors**: *Huanjing Yue, Wenxin Duo, Xiulian Peng, Jingyu Yang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i10.21419](https://doi.org/10.1609/aaai.v36i10.21419)

**Abstract**:

Speech enhancement aims at recovering a clean speech from a noisy input, which can be classified into single speech enhancement and personalized speech enhancement. Personalized speech enhancement usually utilizes the speaker identity extracted from the noisy speech itself (or a clean reference speech) as a global embedding to guide the enhancement process. Different from them, we observe that the speeches of the same speaker are correlated in terms of frame-level short-time Fourier Transform (STFT) spectrogram. Therefore, we propose reference-based speech enhancement via a feature alignment and fusion network (FAF-Net). Given a noisy speech and a clean reference speech spoken by the same speaker, we first propose a feature level alignment strategy to warp the clean reference with the noisy speech in frame level. Then, we fuse the reference feature with the noisy feature via a similarity-based fusion strategy. Finally, the fused features are skipped connected to the decoder, which generates the enhanced results. Experimental results demonstrate that the performance of the proposed FAF-Net is close to state-of-the-art speech enhancement methods on both DNS and Voice Bank+DEMAND datasets. Our code is available at https://github.com/HieDean/FAF-Net.

----

## [1304] MDD-Eval: Self-Training on Augmented Data for Multi-Domain Dialogue Evaluation

**Authors**: *Chen Zhang, Luis Fernando D'Haro, Thomas Friedrichs, Haizhou Li*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i10.21420](https://doi.org/10.1609/aaai.v36i10.21420)

**Abstract**:

Chatbots are designed to carry out human-like conversations across different domains, such as general chit-chat, knowledge exchange, and persona-grounded conversations. To measure the quality of such conversational agents, a dialogue evaluator is expected to conduct assessment across domains as well. However, most of the state-of-the-art automatic dialogue evaluation metrics (ADMs) are not designed for multi-domain evaluation. We are motivated to design a general and robust framework, MDD-Eval, to address the problem. Specifically, we first train a teacher evaluator with human-annotated data to acquire a rating skill to tell good dialogue responses from bad ones in a particular domain and then, adopt a self-training strategy to train a new evaluator with teacher-annotated multi-domain data, that helps the new evaluator to generalize across multiple domains. MDD-Eval is extensively assessed on six dialogue evaluation benchmarks. Empirical results show that the MDD-Eval framework achieves a strong performance with an absolute improvement of 7% over the state-of-the-art ADMs in terms of mean Spearman correlation scores across all the evaluation benchmarks.

----

## [1305] Efficient Dialog Policy Learning by Reasoning with Contextual Knowledge

**Authors**: *Haodi Zhang, Zhichao Zeng, Keting Lu, Kaishun Wu, Shiqi Zhang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i10.21421](https://doi.org/10.1609/aaai.v36i10.21421)

**Abstract**:

Goal-oriented dialog policy learning algorithms aim to learn a dialog policy for selecting language actions based on the current dialog state. Deep reinforcement learning methods have been used for dialog policy learning. This work is motivated by the observation that, although dialog is a domain with rich contextual knowledge, reinforcement learning methods are ill-equipped to incorporate such knowledge into the dialog policy learning process. In this paper, we develop a deep reinforcement learning framework for goal-oriented dialog policy learning that learns user preferences from user goal data, while leveraging commonsense knowledge from people. The developed framework has been evaluated using a realistic dialog simulation platform. Compared with baselines from the literature and the ablations of our approach, we see significant improvements in learning efficiency and the quality of the computed action policies.

----

## [1306] Hierarchical Cross-Modality Semantic Correlation Learning Model for Multimodal Summarization

**Authors**: *Litian Zhang, Xiaoming Zhang, Junshu Pan*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i10.21422](https://doi.org/10.1609/aaai.v36i10.21422)

**Abstract**:

Multimodal summarization with multimodal output (MSMO) generates a summary with both textual and visual content. Multimodal news report contains heterogeneous contents, which makes MSMO nontrivial. Moreover, it is observed that different modalities of data in the news report correlate hierarchically. Traditional MSMO methods indistinguishably handle different modalities of data by learning a representation for the whole data, which is not directly adaptable to the heterogeneous contents and hierarchical correlation. In this paper, we propose a hierarchical cross-modality semantic correlation learning model (HCSCL) to learn the intra- and inter-modal correlation existing in the multimodal data. HCSCL adopts a graph network to encode the intra-modal correlation. Then, a hierarchical fusion framework is proposed to learn the hierarchical correlation between text and images. Furthermore, we construct a new dataset with relevant image annotation and image object label information to provide the supervision information for the learning procedure. Extensive experiments on the dataset show that HCSCL significantly outperforms the baseline methods in automatic summarization metrics and fine-grained diversity tests.

----

## [1307] Adversarial Data Augmentation for Task-Specific Knowledge Distillation of Pre-trained Transformers

**Authors**: *Minjia Zhang, Uma-Naresh Niranjan, Yuxiong He*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i10.21423](https://doi.org/10.1609/aaai.v36i10.21423)

**Abstract**:

Deep and large pre-trained language models (e.g., BERT, GPT-3) are state-of-the-art for various natural language processing tasks. However, the huge size of these models brings challenges to fine-tuning and online deployment due to latency and cost constraints. Existing knowledge distillation methods reduce the model size, but they may encounter difficulties transferring knowledge from the teacher model to the student model due to the limited data from the downstream tasks. In this work, we propose AD^2, a novel and effective data augmentation approach to improving the task-specific knowledge transfer when compressing large pre-trained transformer models. Different from prior methods, AD^2 performs distillation by using an enhanced training set that contains both original inputs and adversarially perturbed samples that mimic the output distribution from the teacher. 
  
Experimental results show that this method allows better transfer of knowledge from the teacher to the student during distillation, producing student models that retain 99.6\% accuracy of the teacher model while outperforming existing task-specific knowledge distillation baselines by 1.2 points on average over a variety of natural language understanding tasks. Moreover, compared with alternative data augmentation methods, such as text-editing-based approaches, AD^2 is up to 28 times faster while achieving comparable or higher accuracy. In addition, when AD^2 is combined with more advanced task-agnostic distillation, we can advance the state-of-the-art performance even more. On top of the encouraging performance, this paper also provides thorough ablation studies and analysis. The discovered interplay between KD and adversarial data augmentation for compressing pre-trained Transformers may further inspire more advanced KD algorithms for compressing even larger scale models.

----

## [1308] Text-Based Interactive Recommendation via Offline Reinforcement Learning

**Authors**: *Ruiyi Zhang, Tong Yu, Yilin Shen, Hongxia Jin*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i10.21424](https://doi.org/10.1609/aaai.v36i10.21424)

**Abstract**:

Interactive recommendation with natural-language feedback can provide richer user feedback and has demonstrated advantages over traditional recommender systems. However, the classical online paradigm involves iteratively collecting experience via interaction with users, which is expensive and risky. We consider an offline interactive recommendation to exploit arbitrary experience collected by multiple unknown policies. A direct application of policy learning with such fixed experience suffers from the distribution shift. To tackle this issue, we develop a behavior-agnostic off-policy correction framework to make offline interactive recommendation possible. Specifically, we leverage the 
conservative Q-function to perform off-policy evaluation, which enables learning effective policies from fixed datasets without further interactions. Empirical results on the simulator derived from real-world datasets demonstrate the effectiveness of our proposed offline training framework.

----

## [1309] DKPLM: Decomposable Knowledge-Enhanced Pre-trained Language Model for Natural Language Understanding

**Authors**: *Taolin Zhang, Chengyu Wang, Nan Hu, Minghui Qiu, Chengguang Tang, Xiaofeng He, Jun Huang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i10.21425](https://doi.org/10.1609/aaai.v36i10.21425)

**Abstract**:

Knowledge-Enhanced Pre-trained Language Models (KEPLMs) are pre-trained models with relation triples injecting from knowledge graphs to improve language understanding abilities.Experiments show that our model outperforms other KEPLMs significantly over zero-shot knowledge probing tasks and multiple knowledge-aware language understanding tasks. To guarantee effective knowledge injection, previous studies integrate models with knowledge encoders for representing knowledge retrieved from knowledge graphs. The operations for knowledge retrieval and encoding bring significant computational burdens, restricting the usage of such models in real-world applications that require high inference speed. In this paper, we propose a novel KEPLM named DKPLM that
 decomposes knowledge injection process of the pre-trained language models in pre-training, fine-tuning and inference stages, which facilitates the applications of KEPLMs in real-world scenarios. Specifically, we first detect knowledge-aware long-tail entities as the target for knowledge injection, enhancing the KEPLMs' semantic understanding abilities and avoiding injecting redundant information.
 The embeddings of long-tail entities are replaced by ``pseudo token representations'' formed by relevant knowledge triples. We further design the relational knowledge decoding task for pre-training to force the models to truly understand the injected knowledge by relation triple reconstruction. Experiments show that our model outperforms other KEPLMs significantly over zero-shot knowledge probing tasks and multiple knowledge-aware language understanding tasks. We further show that DKPLM has a higher inference speed than other competing models due to the decomposing mechanism.

----

## [1310] Frequency-Aware Contrastive Learning for Neural Machine Translation

**Authors**: *Tong Zhang, Wei Ye, Baosong Yang, Long Zhang, Xingzhang Ren, Dayiheng Liu, Jinan Sun, Shikun Zhang, Haibo Zhang, Wen Zhao*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i10.21426](https://doi.org/10.1609/aaai.v36i10.21426)

**Abstract**:

Low-frequency word prediction remains a challenge in modern neural machine translation (NMT) systems. Recent adaptive training methods promote the output of infrequent words by emphasizing their weights in the overall training objectives. Despite the improved recall of low-frequency words, their prediction precision is unexpectedly hindered by the adaptive objectives. Inspired by the observation that low-frequency words form a more compact embedding space, we tackle this challenge from a representation learning perspective. Specifically, we propose a frequency-aware token-level contrastive learning method, in which the hidden state of each decoding step is pushed away from the counterparts of other target words, in a soft contrastive way based on the corresponding word frequencies. We conduct experiments on widely used NIST Chinese-English and WMT14 English-German translation tasks. Empirical results show that our proposed methods can not only significantly improve the translation quality but also enhance lexical diversity and optimize word representation space. Further investigation reveals that, comparing with related adaptive training strategies, the superiority of our method on low-frequency word prediction lies in the robustness of token-level recall across different frequencies without sacrificing precision.

----

## [1311] Probing Word Syntactic Representations in the Brain by a Feature Elimination Method

**Authors**: *Xiaohan Zhang, Shaonan Wang, Nan Lin, Jiajun Zhang, Chengqing Zong*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i10.21427](https://doi.org/10.1609/aaai.v36i10.21427)

**Abstract**:

Neuroimaging studies have identified multiple brain regions that are associated with semantic and syntactic processing when comprehending language. However, existing methods cannot explore the neural correlates of fine-grained word syntactic features, such as part-of-speech and dependency relations. This paper proposes an alternative framework to study how different word syntactic features are represented in the brain.
To separate each syntactic feature, we propose a feature elimination method, called Mean Vector Null space Projection (MVNP). This method can remove a specific feature from word representations, resulting in one-feature-removed representations. Then we respectively associate one-feature-removed and the original word vectors with brain imaging data to explore how the brain represents the removed feature.
This paper for the first time studies the cortical representations of multiple fine-grained syntactic features simultaneously and suggests some possible contributions of several brain regions to the complex division of syntactic processing. These findings indicate that the brain foundations of syntactic information processing might be broader than those suggested by classical studies.

----

## [1312] Unsupervised Sentence Representation via Contrastive Learning with Mixing Negatives

**Authors**: *Yanzhao Zhang, Richong Zhang, Samuel Mensah, Xudong Liu, Yongyi Mao*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i10.21428](https://doi.org/10.1609/aaai.v36i10.21428)

**Abstract**:

Unsupervised sentence representation learning is a fundamental problem in natural language processing. Recently, contrastive learning has made great success on this task. Existing constrastive learning based models usually apply random sampling to select negative examples for training. Previous work in computer vision has shown that hard negative examples help contrastive learning to achieve faster convergency and better optimization for representation learning. However, the importance of hard negatives in contrastive learning for sentence representation is yet to be explored. In this study, we prove that hard negatives are essential for maintaining strong gradient signals in the training process while random sampling negative examples is ineffective for sentence representation. Accordingly, we present a contrastive model, MixCSE, that extends the current state-of-the-art SimCSE by continually constructing hard negatives via mixing both positive and negative features. The superior performance of the proposed approach is demonstrated via empirical studies on Semantic Textual Similarity datasets and Transfer task datasets.

----

## [1313] RetGen: A Joint Framework for Retrieval and Grounded Text Generation Modeling

**Authors**: *Yizhe Zhang, Siqi Sun, Xiang Gao, Yuwei Fang, Chris Brockett, Michel Galley, Jianfeng Gao, Bill Dolan*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i10.21429](https://doi.org/10.1609/aaai.v36i10.21429)

**Abstract**:

Recent advances in large-scale pre-training such as GPT-3 allow seemingly high quality text to be generated from a given prompt. However, such generation systems often suffer from problems of hallucinated facts, and are not inherently designed to incorporate useful external information. Grounded generation models appear to offer remedies, but their training typically relies on rarely-available parallel data where information-relevant documents are provided for context. We propose a framework that alleviates this data constraint by jointly training a grounded generator and document retriever on the language model signal. The model learns to reward retrieval of the documents with the highest utility in generation, and attentively combines them using a Mixture-of-Experts (MoE) ensemble to generate follow-on text.  We demonstrate that both generator and retriever can take advantage of this joint training and work synergistically to produce more informative and relevant text in both prose and dialogue generation.

----

## [1314] BiRdQA: A Bilingual Dataset for Question Answering on Tricky Riddles

**Authors**: *Yunxiang Zhang, Xiaojun Wan*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i10.21430](https://doi.org/10.1609/aaai.v36i10.21430)

**Abstract**:

A riddle is a question or statement with double or veiled meanings, followed by an unexpected answer. Solving riddle is a challenging task for both machine and human, testing the capability of understanding figurative, creative natural language and reasoning with commonsense knowledge. We introduce BiRdQA, a bilingual multiple-choice question answering dataset with 6614 English riddles and 8751 Chinese riddles. For each riddle-answer pair, we provide four distractors with additional information from Wikipedia. The distractors are automatically generated at scale with minimal bias. Existing monolingual and multilingual QA models fail to perform well on our dataset, indicating that there is a long way to go before machine can beat human on solving tricky riddles. The dataset is publicly available at https://forms.gle/NvT7DfWhAPhvoFvH7.

----

## [1315] UniMS: A Unified Framework for Multimodal Summarization with Knowledge Distillation

**Authors**: *Zhengkun Zhang, Xiaojun Meng, Yasheng Wang, Xin Jiang, Qun Liu, Zhenglu Yang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i10.21431](https://doi.org/10.1609/aaai.v36i10.21431)

**Abstract**:

With the rapid increase of multimedia data, a large body of literature has emerged to work on multimodal summarization, the majority of which target at refining salient information from textual and image modalities to output a pictorial summary with the most relevant images. Existing methods mostly focus on either extractive or abstractive summarization and rely on the presence and quality of image captions to build image references. We are the first to propose a Unified framework for Multimodal Summarization grounding on BART, UniMS, that integrates extractive and abstractive objectives, as well as selecting the image output. Specially, we adopt knowledge distillation from a vision-language pretrained model to improve image selection, which avoids any requirement on the existence and quality of image captions. Besides, we introduce a visual guided decoder to better integrate textual and visual modalities in guiding abstractive text generation. Results show that our best model achieves a new state-of-the-art result on a large-scale benchmark dataset. The newly involved extractive objective as well as the knowledge distillation technique are proven to bring a noticeable improvement to the multimodal summarization task.

----

## [1316] DialogLM: Pre-trained Model for Long Dialogue Understanding and Summarization

**Authors**: *Ming Zhong, Yang Liu, Yichong Xu, Chenguang Zhu, Michael Zeng*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i10.21432](https://doi.org/10.1609/aaai.v36i10.21432)

**Abstract**:

Dialogue is an essential part of human communication and cooperation. Existing research mainly focuses on short dialogue scenarios in a one-on-one fashion. However, multi-person interactions in the real world, such as meetings or interviews, are frequently over a few thousand words. There is still a lack of corresponding research and powerful tools to understand and process such long dialogues. Therefore, in this work, we present a pre-training framework for long dialogue understanding and summarization. Considering the nature of long conversations, we propose a window-based denoising approach for generative pre-training. For a dialogue, it corrupts a window of text with dialogue-inspired noise, and guides the model to reconstruct this window based on the content of the remaining conversation. Furthermore, to process longer input, we augment the model with sparse attention which is combined with conventional attention in a hybrid manner. We conduct extensive experiments on five datasets of long dialogues, covering tasks of dialogue summarization, abstractive question answering and topic segmentation. Experimentally, we show that our pre-trained model DialogLM significantly surpasses the state-of-the-art models across datasets and tasks. Source code and all the pre-trained models are available on our GitHub repository (https://github.com/microsoft/DialogLM).

----

## [1317] Idiomatic Expression Paraphrasing without Strong Supervision

**Authors**: *Jianing Zhou, Ziheng Zeng, Hongyu Gong, Suma Bhat*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i10.21433](https://doi.org/10.1609/aaai.v36i10.21433)

**Abstract**:

Idiomatic expressions (IEs) play an essential role in natural language. In this paper, we study the task of idiomatic sentence paraphrasing (ISP), which aims to paraphrase a sentence with an IE by replacing the IE with its literal paraphrase. The lack of large-scale corpora with idiomatic-literal parallel sentences is a primary challenge for this task, for which we consider two separate solutions. First, we propose an unsupervised approach to ISP, which leverages an IE's contextual information and definition and does not require a parallel sentence training set. Second, we propose a weakly supervised approach using back-translation to jointly perform paraphrasing and generation of sentences with IEs to enlarge the small-scale parallel sentence training dataset. Other significant derivatives of the study include a model that replaces a literal phrase in a sentence with an IE to generate an idiomatic expression and a large scale parallel dataset with idiomatic/literal sentence pairs. The effectiveness of the proposed solutions compared to competitive baselines is seen in the relative gains of over 5.16 points in BLEU, over 8.75 points in METEOR, and over 19.57 points in SARI when the generated sentences are empirically validated on a parallel dataset using automatic and manual evaluations. We demonstrate the practical utility of ISP as a preprocessing step in En-De machine translation.

----

## [1318] Multilingual Code Snippets Training for Program Translation

**Authors**: *Ming Zhu, Karthik Suresh, Chandan K. Reddy*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i10.21434](https://doi.org/10.1609/aaai.v36i10.21434)

**Abstract**:

Program translation aims to translate source code from one programming language to another. It is particularly useful in applications such as multiple-platform adaptation and legacy code migration. Traditional rule-based program translation methods usually rely on meticulous manual rule-crafting, which is costly both in terms of time and effort. Recently, neural network based methods have been developed to address this problem. However, the absence of high-quality parallel code data is one of the main bottlenecks which impedes the development of program translation models. In this paper, we introduce CoST, a new multilingual Code Snippet Translation dataset that contains parallel data from 7 commonly used programming languages. The dataset is parallel at the level of code snippets, which provides much more fine-grained alignments between different languages than the existing translation datasets. We also propose a new program translation model that leverages multilingual snippet denoising auto-encoding and Multilingual Snippet Translation (MuST) pre-training. Extensive experiments show that the multilingual snippet training is effective in improving program translation performance, especially for low-resource languages. Moreover, our training method shows good generalizability and consistently improves the translation performance of a number of baseline models. The proposed model outperforms the baselines on both snippet-level and program-level translation, and achieves state-of-the-art performance on CodeXGLUE translation task. The code, data, and appendix for this paper can be found at https://github.com/reddy-lab-code-research/MuST-CoST.

----

## [1319] Conditional Synthetic Data Generation for Robust Machine Learning Applications with Limited Pandemic Data

**Authors**: *Hari Prasanna Das, Ryan Tran, Japjot Singh, Xiangyu Yue, Geoffrey H. Tison, Alberto L. Sangiovanni-Vincentelli, Costas J. Spanos*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21435](https://doi.org/10.1609/aaai.v36i11.21435)

**Abstract**:

Background: At the onset of a pandemic, such as COVID-19, data with proper labeling/attributes corresponding to the new disease might be unavailable or sparse. Machine Learning (ML) models trained with the available data, which is limited in quantity and poor in diversity, will often be biased and inaccurate. At the same time, ML algorithms designed to fight pandemics must have good performance and be developed in a time-sensitive manner. To tackle the challenges of limited data, and label scarcity in the available data, we propose generating conditional synthetic data, to be used alongside real data for developing robust ML models. Methods: We present a hybrid model consisting of a conditional generative flow and a classifier for conditional synthetic data generation. The classifier decouples the feature representation for the condition, which is fed to the flow to extract the local noise. We generate synthetic data by manipulating the local noise with fixed conditional feature representation. We also propose a semi-supervised approach to generate synthetic samples in the absence of labels for a majority of the available data. Results: We performed conditional synthetic generation for chest computed tomography (CT) scans corresponding to normal, COVID-19, and pneumonia afflicted patients. We show that our method significantly outperforms existing models both on qualitative and quantitative performance, and our semi-supervised approach can efficiently synthesize conditional samples under label scarcity. As an example of downstream use of synthetic data, we show improvement in COVID-19 detection from CT scans with conditional synthetic data augmentation.

----

## [1320] Socially Fair Mitigation of Misinformation on Social Networks via Constraint Stochastic Optimization

**Authors**: *Ahmed Abouzeid, Ole-Christoffer Granmo, Christian Webersik, Morten Goodwin*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21436](https://doi.org/10.1609/aaai.v36i11.21436)

**Abstract**:

Recent social networks' misinformation mitigation approaches tend to investigate how to reduce misinformation by considering a whole-network statistical scale. However, unbalanced misinformation exposures among individuals urge to study fair allocation of mitigation resources. Moreover, the network has random dynamics which change over time. Therefore, we introduce a stochastic and non-stationary knapsack problem, and we apply its resolution to mitigate misinformation in social network campaigns. We further propose a generic misinformation mitigation algorithm that is robust to different social networks' misinformation statistics, allowing a promising impact in real-world scenarios. A novel loss function ensures fair mitigation among users. We achieve fairness by intelligently allocating a mitigation incentivization budget to the knapsack, and optimizing the loss function. To this end, a team of Learning Automata (LA) drives the budget allocation. Each LA is associated with a user and learns to minimize its exposure to misinformation by performing a non-stationary and stochastic walk over its state space. Our results show how our LA-based method is robust and outperforms similar misinformation mitigation methods in how the mitigation is fairly influencing the network users.

----

## [1321] Personalized Public Policy Analysis in Social Sciences Using Causal-Graphical Normalizing Flows

**Authors**: *Sourabh Balgi, José M. Peña, Adel Daoud*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21437](https://doi.org/10.1609/aaai.v36i11.21437)

**Abstract**:

Structural Equation/Causal Models (SEMs/SCMs) are widely used in epidemiology and social sciences to identify and analyze the average causal effect (ACE) and conditional ACE (CACE). Traditional causal effect estimation methods such as Inverse Probability Weighting (IPW) and more recently Regression-With-Residuals (RWR) are widely used - as they avoid the challenging task of identifying the SCM parameters - to estimate ACE and CACE. However, much work remains before traditional estimation methods can be used for counterfactual inference, and for the benefit of Personalized Public Policy Analysis (P3A) in the social sciences. While doctors rely on personalized medicine to tailor treatments to patients in laboratory settings (relatively closed systems), P3A draws inspiration from such tailoring but adapts it for open social systems. In this article, we develop a method for counterfactual inference that we name causal-Graphical Normalizing Flow (c-GNF), facilitating P3A. A major advantage of c-GNF is that it suits the open system in which P3A is conducted. First, we show how c-GNF captures the underlying SCM without making any assumption about functional forms. This capturing capability is enabled by the deep neural networks that model the underlying SCM via observational data likelihood maximization using gradient descent. Second, we propose a novel dequantization trick to deal with discrete variables, which is a limitation of normalizing flows in general. Third, we demonstrate in experiments that c-GNF performs on-par with IPW and RWR in terms of bias and variance for estimating the ATE, when the true functional forms are known, and better when they are unknown. Fourth and most importantly, we conduct counterfactual inference with c-GNFs, demonstrating promising empirical performance. Because IPW and RWR, like other traditional methods, lack the capability of counterfactual inference, c-GNFs will likely play a major role in tailoring personalized treatment, facilitating P3A, optimizing social interventions - in contrast to the current `one-size-fits-all' approach of existing methods.

----

## [1322] Interpretable Low-Resource Legal Decision Making

**Authors**: *Rohan Bhambhoria, Hui Liu, Samuel Dahan, Xiaodan Zhu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21438](https://doi.org/10.1609/aaai.v36i11.21438)

**Abstract**:

Over the past several years, legal applications of deep learning have been on the rise. However, as with other high-stakes decision making areas, the requirement for interpretability is of crucial importance. Current models utilized by legal practitioners are more of the conventional machine learning type, wherein they are inherently interpretable, yet unable to harness the performance capabilities of data-driven deep learning models. In this work, we utilize deep learning models in the area of trademark law to shed light on the issue of likelihood of confusion between trademarks. Specifically, we introduce a model-agnostic interpretable intermediate layer, a technique which proves to be effective for legal documents. Furthermore, we utilize weakly supervised learning by means of a curriculum learning strategy, effectively demonstrating the improved performance of a deep learning model. This is in contrast to the conventional models which are only able to utilize the limited number of expensive manually-annotated samples by legal experts. Although the methods presented in this work tackles the task of risk of confusion for trademarks, it is straightforward to extend them to other fields of law, or more generally, to other similar high-stakes application scenarios.

----

## [1323] Noninvasive Lung Cancer Early Detection via Deep Methylation Representation Learning

**Authors**: *Xiangrui Cai, Jinsheng Tao, Shichao Wang, Zhiyu Wang, Jiaxian Wang, Mei Li, Hong Wang, Xixiang Tu, Hao Yang, Jian-Bing Fan, Hua Ji*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21439](https://doi.org/10.1609/aaai.v36i11.21439)

**Abstract**:

Early detection of lung cancer is crucial for five-year survival of patients. Compared with the pathological analysis and CT scans, the circulating tumor DNA (ctDNA) methylation based approach is noninvasive and cost-effective, and thus is one of the most promising methods for early detection of lung cancer. Existing studies on ctDNA methylation data measure the methylation level of each region with a predefined metric, ignoring the positions of methylated CpG sites and methylation patterns, thus are not able to capture the early cancer signals. In this paper, we propose a blood-based lung cancer detection method, and present the first ever study to represent methylation regions by continuous vectors. Specifically, we propose DeepMeth to regard each region as a one-channel image and develop an auto-encoder model to learn its representation. For each ctDNA methylation sample, DeepMeth achieves its representation via concatenating the region vectors. We evaluate DeepMeth on a multicenter clinical dataset collected from 14 hospitals. The experiments show that DeepMeth achieves about 5%-8% improvements compared with the baselines in terms of Area Under the Curve (AUC). Moreover, the experiments also demonstrate that DeepMeth can be combined with traditional scalar metrics to enhance the diagnostic power of ctDNA methylation classifiers. DeepMeth has been clinically deployed and applied to 450 patients from 94 hospitals nationally since April 2020.

----

## [1324] iGrow: A Smart Agriculture Solution to Autonomous Greenhouse Control

**Authors**: *Xiaoyan Cao, Yao Yao, Lanqing Li, Wanpeng Zhang, Zhicheng An, Zhong Zhang, Li Xiao, Shihui Guo, Xiaoyu Cao, Meihong Wu, Dijun Luo*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21440](https://doi.org/10.1609/aaai.v36i11.21440)

**Abstract**:

Agriculture is the foundation of human civilization. However, the rapid increase of the global population poses a challenge on this cornerstone by demanding more food. Modern autonomous greenhouses, equipped with sensors and actuators, provide a promising solution to the problem by empowering precise control for high-efficient food production. However, the optimal control of autonomous greenhouses is challenging, requiring decision-making based on high-dimensional sensory data, and the scaling of production is limited by the scarcity of labor capable of handling this task. With the advances of artificial intelligence (AI), the internet of things (IoT), and cloud computing technologies, we are hopeful to provide a solution to automate and smarten greenhouse control to address the above challenges. In this paper, we propose a smart agriculture solution named iGrow, for autonomous greenhouse control (AGC): (1) for the first time, we formulate the AGC problem as a Markov decision process (MDP) optimization problem; (2) we design a neural network-based simulator incorporated with the incremental mechanism to simulate the complete planting process of an autonomous greenhouse, which provides a testbed for the optimization of control strategies; (3) we propose a closed-loop bi-level optimization algorithm, which can dynamically re-optimize the greenhouse control strategy with newly observed data during real-world production. We not only conduct simulation experiments but also deploy iGrow in real scenarios, and experimental results demonstrate the effectiveness and superiority of iGrow in autonomous greenhouse simulation and optimal control. Particularly, compelling results from the tomato pilot project in real autonomous greenhouses show that our solution significantly increases crop yield (+10.15%) and net profit (+92.70%) with statistical significance compared to planting experts. Our solution opens up a new avenue for greenhouse production. The code is available at https://github.com/holmescao/iGrow.git.

----

## [1325] CODE: Contrastive Pre-training with Adversarial Fine-Tuning for Zero-Shot Expert Linking

**Authors**: *Bo Chen, Jing Zhang, Xiaokang Zhang, Xiaobin Tang, Lingfan Cai, Hong Chen, Cuiping Li, Peng Zhang, Jie Tang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21441](https://doi.org/10.1609/aaai.v36i11.21441)

**Abstract**:

Expert finding, a popular service provided by many online websites such as Expertise Finder, LinkedIn, and AMiner, is beneficial to seeking candidate qualifications, consultants, and collaborators. However, its quality is suffered from lack of ample sources of expert information. This paper employs AMiner as the basis with an aim at linking any external experts to the counterparts on AMiner. As it is infeasible to acquire sufficient linkages from arbitrary external sources, we explore the problem of zero-shot expert linking. In this paper, we propose CODE, which first pre-trains an expert linking model by contrastive learning on AMiner such that it can capture the representation and matching patterns of experts without supervised signals, then it is fine-tuned between AMinerand external sources to enhance the model’s transferability in an adversarial manner. For evaluation, we first design two intrinsic tasks, author identification and paper clustering, to validate the representation and matching capability endowed by contrastive learning. Then the final external expert linking performance on two genres of external sources also implies the superiority of adversarial fine-tuning method. Additionally, we show the online deployment of CODE, and continuously improve its online performance via active learning.

----

## [1326] Interpreting Gender Bias in Neural Machine Translation: Multilingual Architecture Matters

**Authors**: *Marta R. Costa-jussà, Carlos Escolano, Christine Basta, Javier Ferrando, Roser Batlle, Ksenia Kharitonova*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21442](https://doi.org/10.1609/aaai.v36i11.21442)

**Abstract**:

Multilingual neural machine translation architectures mainly differ in the number of sharing modules and parameters applied among languages. In this paper, and from an algorithmic perspective, we explore whether the chosen architecture, when trained with the same data, influences the level of gender bias. Experiments conducted in three language pairs show that language-specific encoder-decoders exhibit less bias than the shared architecture. We propose two methods for interpreting and studying gender bias in machine translation based on source embeddings and attention. Our analysis shows that, in the language-specific case, the embeddings encode more gender information, and their attention is more diverted. Both behaviors help in mitigating gender bias.

----

## [1327] Word Embeddings via Causal Inference: Gender Bias Reducing and Semantic Information Preserving

**Authors**: *Lei Ding, Dengdeng Yu, Jinhan Xie, Wenxing Guo, Shenggang Hu, Meichen Liu, Linglong Kong, Hongsheng Dai, Yanchun Bao, Bei Jiang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21443](https://doi.org/10.1609/aaai.v36i11.21443)

**Abstract**:

With widening deployments of natural language processing (NLP) in daily life, inherited social biases from NLP models have become more severe and problematic. Previous studies have shown that word embeddings trained on human-generated corpora have strong gender biases that can produce discriminative results in downstream tasks. Previous debiasing methods focus mainly on modeling bias and only implicitly consider semantic information while completely overlooking the complex underlying causal structure among bias and semantic components. To address these issues, we propose a novel methodology that leverages a causal inference framework to effectively remove gender bias. The proposed method allows us to construct and analyze the complex causal mechanisms facilitating gender information flow while retaining oracle semantic information within word embeddings. Our comprehensive experiments show that the proposed method achieves state-of-the-art results in gender-debiasing tasks. In addition, our methods yield better performance in word similarity evaluation and various extrinsic downstream NLP tasks.

----

## [1328] A GNN-RNN Approach for Harnessing Geospatial and Temporal Information: Application to Crop Yield Prediction

**Authors**: *Joshua Fan, Junwen Bai, Zhiyun Li, Ariel Ortiz-Bobea, Carla P. Gomes*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21444](https://doi.org/10.1609/aaai.v36i11.21444)

**Abstract**:

Climate change is posing new challenges to crop-related concerns, including food insecurity, supply stability, and economic planning. Accurately predicting crop yields is crucial for addressing these challenges. However, this prediction task is exceptionally complicated since crop yields depend on numerous factors such as weather, land surface, and soil quality, as well as their interactions. In recent years, machine learning models have been successfully applied in this domain. However, these models either restrict their tasks to a relatively small region, or only study over a single or few years, which makes them hard to generalize spatially and temporally. In this paper, we introduce a novel graph-based recurrent neural network for crop yield prediction, to incorporate both geographical and temporal knowledge in the model, and further boost predictive power. Our method is trained, validated, and tested on over 2000 counties from 41 states in the US mainland, covering years from 1981 to 2019. As far as we know, this is the first machine learning method that embeds geographical knowledge in crop yield prediction and predicts crop yields at the county level nationwide. We also laid a solid foundation by comparing our model on a nationwide scale with other well-known baseline methods, including linear models, tree-based models, and deep learning methods. Experiments show that our proposed method consistently outperforms the existing state-of-the-art methods on various metrics, validating the effectiveness of geospatial and temporal information.

----

## [1329] Has CEO Gender Bias Really Been Fixed? Adversarial Attacking and Improving Gender Fairness in Image Search

**Authors**: *Yunhe Feng, Chirag Shah*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21445](https://doi.org/10.1609/aaai.v36i11.21445)

**Abstract**:

Gender bias is one of the most common and well-studied demographic biases in information retrieval, and in general in AI systems. After discovering and reporting that gender bias for certain professions could change searchers' worldviews, mainstreaming image search engines, such as Google, quickly took action to correct and fix such a bias. However, given the nature of these systems, viz., being opaque, it is unclear if they addressed unequal gender representation and gender stereotypes in image search results systematically and in a sustainable way. In this paper, we propose adversarial attack queries composed of professions and countries (e.g., 'CEO United States') to investigate whether gender bias is thoroughly mitigated by image search engines. Our experiments on Google, Baidu, Naver, and Yandex Image Search show that the proposed attack can trigger high levels of gender bias in image search results very effectively. To defend against such attacks and mitigate gender bias, we design and implement three novel re-ranking algorithms -- epsilon-greedy algorithm, relevance-aware swapping algorithm, and fairness-greedy algorithm, to re-rank returned images for given image queries. Experiments on both simulated (three typical gender distributions) and real-world datasets demonstrate the proposed algorithms can mitigate gender bias effectively.

----

## [1330] Preserving Privacy in Federated Learning with Ensemble Cross-Domain Knowledge Distillation

**Authors**: *Xuan Gong, Abhishek Sharma, Srikrishna Karanam, Ziyan Wu, Terrence Chen, David S. Doermann, Arun Innanje*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21446](https://doi.org/10.1609/aaai.v36i11.21446)

**Abstract**:

Federated Learning (FL) is a machine learning paradigm where  local nodes collaboratively train a central model while the training data remains decentralized. Existing FL methods typically share model parameters or employ co-distillation to address the issue of unbalanced data distribution. However, they suffer from communication bottlenecks. More importantly, they risk privacy leakage risk. In this work, we develop a privacy preserving  and communication efficient method in a FL framework with one-shot offline knowledge distillation using unlabeled, cross-domain, non-sensitive public data. We propose a quantized and noisy ensemble of local predictions from completely trained local models for stronger privacy guarantees without sacrificing accuracy. Based on extensive experiments on image classification and text classification tasks, we show that our method outperforms baseline FL algorithms with superior performance in both accuracy and data privacy preservation.

----

## [1331] FairFoody: Bringing In Fairness in Food Delivery

**Authors**: *Anjali Gupta, Rahul Yadav, Ashish Nair, Abhijnan Chakraborty, Sayan Ranu, Amitabha Bagchi*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21447](https://doi.org/10.1609/aaai.v36i11.21447)

**Abstract**:

Along with the rapid growth and rise to prominence of food delivery platforms, concerns have also risen about the terms of employment of the ``gig workers'' underpinning this growth. Our analysis on data derived from a real-world food delivery platform across three large cities from India show that there is significant inequality in the money delivery agents earn. In this paper, we formulate the problem of fair income distribution among agents while also ensuring timely food delivery. We establish that the problem is not only NP-hard but also inapproximable in polynomial time. We overcome this computational bottleneck through a novel matching algorithm called FairFoody. Extensive experiments over real-world food delivery datasets show FairFoody imparts up to 10 times improvement in equitable income distribution when compared to baseline strategies, while also ensuring minimal impact on customer experience.

----

## [1332] Bayesian Optimisation for Active Monitoring of Air Pollution

**Authors**: *Sigrid Passano Hellan, Christopher G. Lucas, Nigel H. Goddard*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21448](https://doi.org/10.1609/aaai.v36i11.21448)

**Abstract**:

Air pollution is one of the leading causes of mortality globally, resulting in millions of deaths each year. Efficient monitoring is important to measure exposure and enforce legal limits. New low-cost sensors can be deployed in greater numbers and in more varied locations, motivating the problem of efficient automated placement. Previous work suggests Bayesian optimisation is an appropriate method, but only considered a satellite data set, with data aggregated over all altitudes. It is ground-level pollution, that humans breathe, which matters most. We improve on those results using hierarchical models and evaluate our models on urban pollution data in London to show that Bayesian optimisation can be successfully applied to the problem.

----

## [1333] ChildrEN SafEty and Rescue (CENSER) System for Trafficked Children from Brothels in India

**Authors**: *Raghu Vamshi Hemadri, Amarjot Singh, Ajeet Singh*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21449](https://doi.org/10.1609/aaai.v36i11.21449)

**Abstract**:

Human child trafficking has become a global epidemic with over 10 million children forced into labor or prostitution. In this paper, we propose the ChildrEN SafEty and Rescue (CENSER) system used by the Guria non-profit organization to retrieve trafficked children from brothels in India. The CENSER system is formed of the proposed Memory Augmented ScatterNet ResNet Hybrid (MSRHN) network trained on three databases containing images of trafficked children at different ages, their kins, and their sketches. The CENSER system encodes the input image of a child using the proposed Memory Augmented ScatterNet ResNet Hybrid (MSRHN) network and queries the encoding with the (i) Age, (ii) Kinship, and (iii) Sketch databases to establish the child's identity. The CENSER system can also predict if a child is a minor, which is used along with their identity to convince law enforcement to initiate the rescue operation. The MSRHN network is pre-trained on the KinFace database and then fine-tuned on the three databases. The performance of the proposed model is compared with several state-of-the-art methods.

----

## [1334] Gradual (In)Compatibility of Fairness Criteria

**Authors**: *Corinna Hertweck, Tim Räz*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21450](https://doi.org/10.1609/aaai.v36i11.21450)

**Abstract**:

Impossibility results show that important fairness measures (independence, separation, sufficiency) cannot be satisfied at the same time under reasonable assumptions. This paper explores whether we can satisfy and/or improve these fairness measures simultaneously to a certain degree. We introduce information-theoretic formulations of the fairness measures and define degrees of fairness based on these formulations. The information-theoretic formulations suggest unexplored theoretical relations between the three fairness measures. In the experimental part, we use the information-theoretic expressions as regularizers to obtain fairness-regularized predictors for three standard datasets. Our experiments show that a) fairness regularization directly increases fairness measures, in line with existing work, and b) some fairness regularizations indirectly increase other fairness measures, as suggested by our theoretical findings. This establishes that it is possible to increase the degree to which some fairness measures are satisfied at the same time -- some fairness measures are gradually compatible.

----

## [1335] Adaptive Energy Management for Self-Sustainable Wearables in Mobile Health

**Authors**: *Dina Hussein, Ganapati Bhat, Janardhan Rao Doppa*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21451](https://doi.org/10.1609/aaai.v36i11.21451)

**Abstract**:

Wearable devices that integrate multiple sensors, processors, and communication technologies have the potential to transform mobile health for remote monitoring of health parameters. However, the small form factor of the wearable devices limits the battery size and operating lifetime. As a result, the devices require frequent recharging, which has limited their widespread adoption. Energy harvesting has emerged as an effective method towards sustainable operation of wearable devices. Unfortunately, energy harvesting alone is not sufficient to fulfill the energy requirements of wearable devices. This paper studies the novel problem of adaptive energy management towards the goal of self-sustainable wearables by using harvested energy to supplement the battery energy and to reduce manual recharging by users. To solve this problem, we propose a principled algorithm referred as AdaEM. There are two key ideas behind AdaEM. First, it uses machine learning (ML) methods to learn predictive models of user activity and energy usage patterns. These models allow us to estimate the potential of energy harvesting in a day as a function of the user activities. Second, it reasons about the uncertainty in predictions and estimations from the ML models to optimize the energy management decisions using a dynamic robust optimization (DyRO) formulation. We propose a light-weight solution for DyRO to meet the practical needs of deployment. We validate the AdaEM approach on a wearable device prototype consisting of solar and motion energy harvesting using real-world data of user activities. Experiments show that AdaEM achieves solutions that are within 5% of the optimal with less than 0.005% execution time and energy overhead.

----

## [1336] Evaluating Explainable AI on a Multi-Modal Medical Imaging Task: Can Existing Algorithms Fulfill Clinical Requirements?

**Authors**: *Weina Jin, Xiaoxiao Li, Ghassan Hamarneh*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21452](https://doi.org/10.1609/aaai.v36i11.21452)

**Abstract**:

Being able to explain the prediction to clinical end-users is a necessity to leverage the power of artificial intelligence (AI) models for clinical decision support. For medical images, a feature attribution map, or heatmap, is the most common form of explanation that highlights important features for AI models' prediction. However, it is unknown how well heatmaps perform on explaining decisions on multi-modal medical images, where each image modality or channel visualizes distinct clinical information of the same underlying biomedical phenomenon. Understanding such modality-dependent features is essential for clinical users' interpretation of AI decisions. To tackle this clinically important but technically ignored problem, we propose the modality-specific feature importance (MSFI) metric. It encodes clinical image and explanation interpretation patterns of modality prioritization and modality-specific feature localization. We conduct a clinical requirement-grounded, systematic evaluation using computational methods and a clinician user study. Results show that the examined 16 heatmap algorithms failed to fulfill clinical requirements to correctly indicate AI model decision process or decision quality. The evaluation and MSFI metric can guide the design and selection of explainable AI algorithms to meet clinical requirements on multi-modal explanation.

----

## [1337] Unmasking the Mask - Evaluating Social Biases in Masked Language Models

**Authors**: *Masahiro Kaneko, Danushka Bollegala*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21453](https://doi.org/10.1609/aaai.v36i11.21453)

**Abstract**:

Masked Language Models (MLMs) have shown superior performances in numerous downstream Natural Language Processing (NLP) tasks.
 Unfortunately, MLMs also demonstrate significantly worrying levels of social biases.
 We show that the previously proposed evaluation metrics for quantifying the social biases in MLMs are problematic due to the following reasons:
 (1) prediction accuracy of the masked tokens itself tend to be low in some MLMs,
 which leads to unreliable evaluation metrics, and 
 (2) in most downstream NLP tasks, masks are not used; therefore prediction of the mask is not directly related to them, and 
 (3) high-frequency words in the training data are masked more often, introducing noise due to this selection bias in the test cases.
 Therefore, we propose All Unmasked Likelihood (AUL), a bias evaluation measure that predicts all tokens in a test case given the MLM embedding of the unmasked input and AUL with Attention weights (AULA) to evaluate tokens based on their importance in a sentence.
 Our experimental results show that the proposed bias evaluation measures accurately detect different types of biases in MLMs, and unlike AUL and AULA, previously proposed measures for MLMs systematically overestimate the measured biases and are heavily influenced by the unmasked tokens in the context.

----

## [1338] CrossWalk: Fairness-Enhanced Node Representation Learning

**Authors**: *Ahmad Khajehnejad, Moein Khajehnejad, Mahmoudreza Babaei, Krishna P. Gummadi, Adrian Weller, Baharan Mirzasoleiman*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21454](https://doi.org/10.1609/aaai.v36i11.21454)

**Abstract**:

The potential for machine learning systems to amplify social inequities and unfairness is receiving increasing popular and academic attention. Much recent work has focused on developing algorithmic tools to assess and mitigate such unfairness. However, there is little work on enhancing fairness in graph algorithms. Here, we develop a simple, effective and general method, CrossWalk, that enhances fairness of various graph algorithms, including influence maximization, link prediction and node classification, applied to node embeddings. CrossWalk is applicable to any random walk based node representation learning algorithm, such as DeepWalk and Node2Vec. The key idea is to bias random walks to cross group boundaries, by upweighting edges which (1) are closer to the groups’ peripheries or (2) connect different groups in the network. CrossWalk pulls nodes that are near groups’ peripheries towards their neighbors from other groups in the embedding space, while preserving the necessary structural properties of the graph. Extensive experiments show the effectiveness of our algorithm to enhance fairness in various graph algorithms, including influence maximization, link prediction and node classification in synthetic and real networks, with only a very small decrease in performance.

----

## [1339] COVID-EENet: Predicting Fine-Grained Impact of COVID-19 on Local Economies

**Authors**: *Doyoung Kim, Hyangsuk Min, Youngeun Nam, Hwanjun Song, Susik Yoon, Minseok Kim, Jae-Gil Lee*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21455](https://doi.org/10.1609/aaai.v36i11.21455)

**Abstract**:

Assessing the impact of the COVID-19 crisis on economies is fundamental to tailor the responses of the governments to recover from the crisis. In this paper, we present a novel approach to assessing the economic impact with a large-scale credit card transaction dataset at a fine granularity. For this purpose, we develop a fine-grained economic-epidemiological modeling framework COVID-EENet, which is featured with a two-level deep neural network. In support of the fine-grained EEM, COVID-EENet learns the impact of nearby mass infection cases on the changes of local economies in each district. Through the experiments using the nationwide dataset, given a set of active mass infection cases, COVID-EENet is shown to precisely predict the sales changes in two or four weeks for each district and business category. Therefore, policymakers can be informed of the predictive impact to put in the most effective mitigation measures. Overall, we believe that our work opens a new perspective of using financial data to recover from the economic crisis. For public use in this urgent problem, we release the source code at https://github.com/kaist-dmlab/COVID-EENet.

----

## [1340] A Search Engine for Discovery of Scientific Challenges and Directions

**Authors**: *Dan Lahav, Jon Saad-Falcon, Bailey Kuehl, Sophie Johnson, Sravanthi Parasa, Noam Shomron, Duen Horng Chau, Diyi Yang, Eric Horvitz, Daniel S. Weld, Tom Hope*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21456](https://doi.org/10.1609/aaai.v36i11.21456)

**Abstract**:

Keeping track of scientific challenges, advances and emerging directions is a fundamental part of research. However, researchers face a flood of papers that hinders discovery of important knowledge. In biomedicine, this directly impacts human lives. To address this problem, we present a novel task of extraction and search of scientific challenges and directions, to facilitate rapid knowledge discovery. We construct and release an expert-annotated corpus of texts sampled from full-length papers, labeled with novel semantic categories that generalize across many types of challenges and directions. We focus on a large corpus of interdisciplinary work relating to the COVID-19 pandemic, ranging from biomedicine to areas such as AI and economics. We apply a model trained on our data to identify challenges and directions across the corpus and build a dedicated search engine. In experiments with 19 researchers and clinicians using our system, we outperform a popular scientific search engine in assisting knowledge discovery. Finally, we show that models trained on our resource generalize to the wider biomedical domain and to AI papers, highlighting its broad utility. We make our data, model and search engine publicly available.

----

## [1341] Transcribing Natural Languages for the Deaf via Neural Editing Programs

**Authors**: *Dongxu Li, Chenchen Xu, Liu Liu, Yiran Zhong, Rong Wang, Lars Petersson, Hongdong Li*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21457](https://doi.org/10.1609/aaai.v36i11.21457)

**Abstract**:

This work studies the task of glossification, of which the aim is to em transcribe natural spoken language sentences for the Deaf (hard-of-hearing) community to ordered sign language glosses. Previous sequence-to-sequence language models trained with paired sentence-gloss data often fail to capture the rich connections between the two distinct languages, leading to unsatisfactory transcriptions. We observe that despite different grammars, glosses effectively simplify sentences for the ease of deaf communication, while sharing a large portion of vocabulary with sentences. This has motivated us to implement glossification by executing a collection of editing actions, e.g. word addition, deletion, and copying, called editing programs, on their natural spoken language counterparts. Specifically, we design a new neural agent that learns to synthesize and execute editing programs, conditioned on sentence contexts and partial editing results. The agent is trained to imitate minimal editing programs, while exploring more widely the program space via policy gradients to optimize sequence-wise transcription quality. Results show that our approach outperforms previous glossification models by a large margin,  improving the BLEU-4 score from 16.45 to 18.89 on RWTH-PHOENIX-WEATHER-2014T and from 18.38 to 21.30 on CSL-Daily.

----

## [1342] Optimal Local Explainer Aggregation for Interpretable Prediction

**Authors**: *Qiaomei Li, Rachel Cummings, Yonatan Mintz*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21458](https://doi.org/10.1609/aaai.v36i11.21458)

**Abstract**:

A key challenge for decision makers when incorporating black box machine learned models into practice is being able to understand the predictions provided by these models. One set of methods proposed to address this challenge is that of training surrogate explainer models which approximate how the more complex model is computing its predictions. Explainer methods are generally classified as either local or global explainers depending on what portion of the data space they are purported to explain. The improved coverage of global explainers usually comes at the expense of explainer fidelity (i.e., how well the explainer's predictions match that of the black box model). One way of trading off the advantages of both approaches is to aggregate several local explainers into a single explainer model with improved coverage. However, the problem of aggregating these local explainers is computationally challenging, and existing methods only use heuristics to form these aggregations.
  
 In this paper, we propose a local explainer aggregation method which selects local explainers using non-convex optimization. In contrast to other heuristic methods, we use an integer optimization framework to combine local explainers into a near-global aggregate explainer. Our framework allows a decision-maker to directly tradeoff coverage and fidelity of the resulting aggregation through the parameters of the optimization problem. We also propose a novel local explainer algorithm based on information filtering. We evaluate our algorithmic framework on two healthcare datasets: the Parkinson's Progression Marker Initiative (PPMI) data set and a geriatric mobility dataset from the UCI machine learning repository. Our choice of these healthcare-related datasets is motivated by the anticipated need for explainable precision medicine. We find that our method outperforms existing local explainer aggregation methods in terms of both fidelity and coverage of classification. It also improves on fidelity over existing global explainer methods, particularly in multi-class settings, where state-of-the-art methods achieve 70% and ours achieves 90%.

----

## [1343] Fair Conformal Predictors for Applications in Medical Imaging

**Authors**: *Charles Lu, Andréanne Lemay, Ken Chang, Katharina Höbel, Jayashree Kalpathy-Cramer*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21459](https://doi.org/10.1609/aaai.v36i11.21459)

**Abstract**:

Deep learning has the potential to automate many clinically useful tasks in medical imaging. However translation of deep learning into clinical practice has been hindered by issues such as lack of the transparency and interpretability in these ``black box'' algorithms compared to traditional statistical methods. Specifically, many clinical deep learning models lack rigorous and robust techniques for conveying certainty (or lack thereof) in their predictions -- ultimately limiting their appeal for extensive use in medical decision-making. Furthermore, numerous demonstrations of algorithmic bias have increased hesitancy towards deployment of deep learning for clinical applications. To this end, we explore how conformal predictions can complement existing deep learning approaches by providing an intuitive way of expressing uncertainty while facilitating greater transparency to clinical users. In this paper, we conduct field interviews with radiologists to assess possible use-cases for conformal predictors. Using insights gathered from these interviews, we devise two clinical use-cases and empirically evaluate several methods of conformal predictions on a dermatology photography dataset for skin lesion classification. We show how to modify conformal predictions to be more adaptive to subgroup differences in patient skin tones through equalized coverage. Finally, we compare conformal prediction against measures of epistemic uncertainty.

----

## [1344] Field Study in Deploying Restless Multi-Armed Bandits: Assisting Non-profits in Improving Maternal and Child Health

**Authors**: *Aditya Mate, Lovish Madaan, Aparna Taneja, Neha Madhiwalla, Shresth Verma, Gargi Singh, Aparna Hegde, Pradeep Varakantham, Milind Tambe*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21460](https://doi.org/10.1609/aaai.v36i11.21460)

**Abstract**:

The widespread availability of cell phones has enabled non-profits to deliver critical health information to their beneficiaries in a timely manner. This paper describes our work to assist non-profits that employ automated messaging programs to deliver timely preventive care information to beneficiaries (new and expecting mothers) during pregnancy and after delivery. Unfortunately, a key challenge in such information delivery programs is that a significant fraction of beneficiaries drop out of the program. Yet, non-profits often have limited health-worker resources (time) to place crucial service calls for live interaction with beneficiaries to prevent such engagement drops. To assist non-profits in optimizing this limited resource, we developed a Restless Multi-Armed Bandits (RMABs) system. One key technical contribution in this system is a novel clustering method of offline historical data to infer unknown RMAB parameters. Our second major contribution is evaluation of our RMAB system in collaboration with an NGO, via a real-world service quality improvement study. The study compared strategies for optimizing service calls to 23003 participants over a period of 7 weeks to reduce engagement drops. We show that the RMAB group provides statistically significant improvement over other comparison groups, reducing ~30% engagement drops. To the best of our knowledge, this is the first study demonstrating the utility of RMABs in real world public health settings. We are transitioning our RMAB system to the NGO for real-world use.

----

## [1345] Gender and Racial Stereotype Detection in Legal Opinion Word Embeddings

**Authors**: *Seán Matthews, John Hudzina, Dawn Sepehr*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21461](https://doi.org/10.1609/aaai.v36i11.21461)

**Abstract**:

Studies have shown that some Natural Language Processing (NLP) systems encode and replicate harmful biases with potential adverse ethical effects in our society. In this article, we propose an approach for identifying gender and racial stereotypes in word embeddings trained on judicial opinions from U.S. case law. Embeddings containing stereotype information may cause harm when used by downstream systems for classification, information extraction, question answering, or other machine learning systems used to build legal research tools. We first explain how previously proposed methods for identifying these biases are not well suited for use with word embeddings trained on legal opinion text. We then propose a domain adapted method for identifying gender and racial biases in the legal domain. Our analyses using these methods suggest that racial and gender biases are encoded into word embeddings trained on legal opinions. These biases are not mitigated by exclusion of historical data, and appear across multiple large topical areas of the law. Implications for downstream systems that use legal opinion word embeddings and suggestions for potential mitigation strategies based on our observations are also discussed.

----

## [1346] IS-Count: Large-Scale Object Counting from Satellite Images with Covariate-Based Importance Sampling

**Authors**: *Chenlin Meng, Enci Liu, Willie Neiswanger, Jiaming Song, Marshall Burke, David B. Lobell, Stefano Ermon*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21462](https://doi.org/10.1609/aaai.v36i11.21462)

**Abstract**:

Object detection in high-resolution satellite imagery is emerging as a scalable alternative to on-the-ground survey data collection in many environmental and socioeconomic monitoring applications. However, performing object detection over large geographies can still be prohibitively expensive due to the high cost of purchasing imagery and compute. Inspired by traditional survey data collection strategies, we propose an approach to estimate object count statistics over large geographies through sampling. Given a cost budget, our method selects a small number of representative areas by sampling from a learnable proposal distribution. Using importance sampling, we are able to accurately estimate object counts after processing only a small fraction of the images compared to an exhaustive approach. We show empirically that the proposed framework achieves strong performance on estimating the number of buildings in the United States and Africa, cars in Kenya, brick kilns in Bangladesh, and swimming pools in the U.S., while requiring as few as 0.01% of satellite images compared to an exhaustive approach.

----

## [1347] DevianceNet: Learning to Predict Deviance from a Large-Scale Geo-Tagged Dataset

**Authors**: *Jin-Hwi Park, Young-Jae Park, Junoh Lee, Hae-Gon Jeon*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21463](https://doi.org/10.1609/aaai.v36i11.21463)

**Abstract**:

Understanding how a city’s physical appearance and environmental surroundings impact society traits, such as safety, is an essential issue in social artificial intelligence. To demonstrate the relationship, most existing studies utilize subjective human perceptual attributes, categorization only for a few violent crimes, and images taken from still shot images. These lead to difficulty in identifying location-specific characteristics for urban safety. In this work, to address this problem, we propose a large-scale dataset and a novel method by adopting a concept of “Deviance" which explains behaviors violating social norms, both formally (e.g. crime) and informally (e.g. civil complaints). We first collect a geo-tagged dataset consisting of incident report data for seven metropolitan cities, with corresponding sequential images around incident sites obtained from Google street view. We also design a convolutional neural network that learns spatio-temporal visual attributes of deviant streets. Experimental results show that our framework can reliably recognize real-world deviance in various cities. Furthermore, we analyze which visual attribute is important for deviance identification and severity estimation. We have released our dataset and source codes at our project page: https://deviance-project.github.io/DevianceNet/

----

## [1348] Learning Economic Indicators by Aggregating Multi-Level Geospatial Information

**Authors**: *Sungwon Park, Sungwon Han, Donghyun Ahn, Jaeyeon Kim, Jeasurk Yang, Susang Lee, Seunghoon Hong, Jihee Kim, Sangyoon Park, Hyunjoo Yang, Meeyoung Cha*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21464](https://doi.org/10.1609/aaai.v36i11.21464)

**Abstract**:

High-resolution daytime satellite imagery has become a promising source to study economic activities. These images display detailed terrain over large areas and allow zooming into smaller neighborhoods. Existing methods, however, have utilized images only in a single-level geographical unit. This research presents a deep learning model to predict economic indicators via aggregating traits observed from multiple levels of geographical units. The model first measures hyperlocal economy over small communities via ordinal regression. The next step extracts district-level features by summarizing interconnection among hyperlocal economies. In the final step, the model estimates economic indicators of districts via aggregating the hyperlocal and district information. Our new multi-level learning model substantially outperforms strong baselines in predicting key indicators such as population, purchasing power, and energy consumption. The model is also robust against data shortage; the trained features from one country can generalize to other countries when evaluated with data gathered from Malaysia, the Philippines, Thailand, and Vietnam. We discuss the multi-level model's implications for measuring inequality, which is the essential first step in policy and social science research on inequality and poverty.

----

## [1349] Knowledge Sharing via Domain Adaptation in Customs Fraud Detection

**Authors**: *Sungwon Park, Sundong Kim, Meeyoung Cha*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21465](https://doi.org/10.1609/aaai.v36i11.21465)

**Abstract**:

Knowledge of the changing traffic is critical in risk management. Customs offices worldwide have traditionally relied on local resources to accumulate such knowledge and detect tax frauds. This naturally poses countries with weak infrastructure to become tax havens of potentially illicit trades. The current paper proposes DAS, a memory bank platform to facilitate knowledge sharing across multi-national customs administrations to support each other. We propose a domain adaptation method to share transferable knowledge of frauds as prototypes while safeguarding the local trade information. Data encompassing over 8 million import declarations have been used to test the feasibility of this new system, which shows that participating countries may benefit up to 2-11 times in fraud detection with the help of shared knowledge. We discuss implications for substantial tax revenue potential and strengthened policy against illicit trades.

----

## [1350] Learning the Physics of Particle Transport via Transformers

**Authors**: *Oscar Pastor-Serrano, Zoltán Perkó*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21466](https://doi.org/10.1609/aaai.v36i11.21466)

**Abstract**:

Particle physics simulations are the cornerstone of nuclear engineering applications. Among them radiotherapy (RT) is crucial for society, with 50% of cancer patients receiving radiation treatments. For the most precise targeting of tumors, next generation RT treatments aim for real-time correction during radiation delivery, necessitating particle transport algorithms that yield precise dose distributions in sub-second times even in highly heterogeneous patient geometries. This is infeasible with currently available, purely physics based simulations. In this study, we present a data-driven dose calculation algorithm predicting the dose deposited by mono-energetic proton beams for arbitrary energies and patient geometries. Our approach frames particle transport as sequence modeling, where convolutional layers extract important spatial features into tokens and the transformer self-attention mechanism routes information between such tokens in the sequence and a beam energy token. We train our network and evaluate prediction accuracy using computationally expensive but accurate Monte Carlo (MC) simulations, considered the gold standard in particle physics. Our proposed model is 33 times faster than current clinical analytic pencil beam algorithms, improving upon their accuracy in the most heterogeneous and challenging geometries. With a relative error of 0.34±0.2% and very high gamma pass rate of 99.59±0.7% (1%, 3 mm), it also greatly outperforms the only published similar data-driven proton dose algorithm, even at a finer grid resolution. Offering MC precision 4000 times faster, our model could overcome a major obstacle that has so far prohibited real-time adaptive proton treatments and significantly increase cancer treatment efficacy. Its potential to model physics interactions of other particles could also boost heavy ion treatment planning procedures limited by the speed of traditional methods.

----

## [1351] Accurate and Scalable Gaussian Processes for Fine-Grained Air Quality Inference

**Authors**: *Zeel B. Patel, Palak Purohit, Harsh M. Patel, Shivam Sahni, Nipun Batra*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21467](https://doi.org/10.1609/aaai.v36i11.21467)

**Abstract**:

Air pollution is a global problem and severely impacts human health. Fine-grained air quality (AQ) monitoring is important in mitigating air pollution. However, existing AQ station deployments are sparse. Conventional interpolation techniques fail to learn the complex AQ phenomena. Physics-based models require domain knowledge and pollution source data for AQ modeling. In this work, we propose a Gaussian processes based approach for estimating AQ. The important features of our approach are: a) a non-stationary (NS) kernel to allow input depended smoothness of fit; b) a Hamming distance-based kernel for categorical features; and c) a locally periodic kernel to capture temporal periodicity. We leverage batch-wise training to scale our approach to a large amount of data. Our approach outperforms the conventional baselines and a state-of-the-art neural attention-based approach.

----

## [1352] Investigations of Performance and Bias in Human-AI Teamwork in Hiring

**Authors**: *Andi Peng, Besmira Nushi, Emre Kiciman, Kori Inkpen, Ece Kamar*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21468](https://doi.org/10.1609/aaai.v36i11.21468)

**Abstract**:

In AI-assisted decision-making, effective hybrid (human-AI) teamwork is not solely dependent on AI performance alone, but also on its impact on human decision-making. While prior work studies the effects of model accuracy on humans, we endeavour here to investigate the complex dynamics of how both a model's predictive performance and bias may transfer to humans in a recommendation-aided decision task. We consider the domain of ML-assisted hiring, where humans---operating in a constrained selection setting---can choose whether they wish to utilize a trained model's inferences to help select candidates from written biographies. We conduct a large-scale user study leveraging a re-created dataset of real bios from prior work, where humans predict the ground truth occupation of given candidates with and without the help of three different NLP classifiers (random, bag-of-words, and deep neural network). Our results demonstrate that while high-performance models significantly improve human performance in a hybrid setting, some models mitigate hybrid bias while others accentuate it. We examine these findings through the lens of decision conformity and observe that our model architecture choices have an impact on human-AI conformity and bias, motivating the explicit need to assess these complex dynamics prior to deployment.

----

## [1353] LIMREF: Local Interpretable Model Agnostic Rule-Based Explanations for Forecasting, with an Application to Electricity Smart Meter Data

**Authors**: *Dilini Rajapaksha, Christoph Bergmeir*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21469](https://doi.org/10.1609/aaai.v36i11.21469)

**Abstract**:

Accurate electricity demand forecasts play a key role in sustainable power systems. To enable better decision-making especially for demand flexibility of the end-user, it is necessary to provide not only accurate but also understandable and actionable forecasts. To provide accurate forecasts Global Forecasting Models (GFM) that are trained across time series have shown superior results in many demand forecasting competitions and real-world applications recently, compared with univariate forecasting approaches. We aim to fill the gap between the accuracy and the interpretability in global forecasting approaches. 
 In order to explain the global model forecasts, we propose Local Interpretable Model-agnostic Rule-based Explanations for Forecasting (LIMREF), which is a local explainer framework that produces k-optimal impact rules for a particular forecast, considering the global forecasting model as a black-box model, in a model-agnostic way. It provides different types of rules which explain the forecast of the global model and the counterfactual rules, which provide actionable insights for potential changes to obtain different outputs for given instances. We conduct experiments using a large-scale electricity demand dataset with exogenous features such as temperature and calendar effects. Here, we evaluate the quality of the explanations produced by the LIMREF framework in terms of both qualitative and quantitative aspects such as accuracy, fidelity and comprehensibility, and benchmark those against other local explainers.

----

## [1354] 'Beach' to 'Bitch': Inadvertent Unsafe Transcription of Kids' Content on YouTube

**Authors**: *Krithika Ramesh, Ashiqur R. KhudaBukhsh, Sumeet Kumar*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21470](https://doi.org/10.1609/aaai.v36i11.21470)

**Abstract**:

Over the last few years, YouTube Kids has emerged as one of the highly competitive alternatives to television for children's entertainment. Consequently, YouTube Kids' content should receive an additional level of scrutiny to ensure children's safety. While research on detecting offensive or inappropriate content for kids is gaining momentum, little or no current work exists that investigates to what extent AI applications can (accidentally) introduce content that is inappropriate for kids. 

In this paper, we present a novel (and troubling) finding that well-known automatic speech recognition (ASR) systems may produce text content highly inappropriate for kids while transcribing YouTube Kids' videos. We dub this phenomenon as inappropriate content hallucination. Our analyses suggest that such hallucinations are far from occasional, and the ASR systems often produce them with high confidence. We release a first-of-its-kind data set of audios for which the existing state-of-the-art ASR systems hallucinate inappropriate content for kids. In addition, we demonstrate that some of these errors can be fixed using language models.

----

## [1355] ReforesTree: A Dataset for Estimating Tropical Forest Carbon Stock with Deep Learning and Aerial Imagery

**Authors**: *Gyri Reiersen, David Dao, Björn Lütjens, Konstantin Klemmer, Kenza Amara, Attila Steinegger, Ce Zhang, Xiaoxiang Zhu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21471](https://doi.org/10.1609/aaai.v36i11.21471)

**Abstract**:

Forest biomass is a key influence for future climate, and the world urgently needs highly scalable financing schemes, such as carbon offsetting certifications, to protect and restore forests. Current manual forest carbon stock inventory methods of measuring single trees by hand are time, labour, and cost intensive and have been shown to be subjective. They can lead to substantial overestimation of the carbon stock and ultimately distrust in forest financing. The potential for impact and scale of leveraging advancements in machine learning and remote sensing technologies is promising, but needs to be of high quality in order to replace the current forest stock protocols for certifications. 
 
 In this paper, we present ReforesTree, a benchmark dataset of forest carbon stock in six agro-forestry carbon offsetting sites in Ecuador. Furthermore, we show that a deep learning-based end-to-end model using individual tree detection from low cost RGB-only drone imagery is accurately estimating forest carbon stock within official carbon offsetting certification standards. Additionally, our baseline CNN model outperforms state-of-the-art satellite-based forest biomass and carbon stock estimates for this type of small-scale, tropical agro-forestry sites. We present this dataset to encourage machine learning research in this area to increase accountability and transparency of monitoring, verification and reporting (MVR) in carbon offsetting projects, as well as scaling global reforestation financing through accurate remote sensing.

----

## [1356] Deep Movement Primitives: Toward Breast Cancer Examination Robot

**Authors**: *Oluwatoyin Sanni, Giorgio Bonvicini, Muhammad Arshad Khan, Pablo C. López-Custodio, Kiyanoush Nazari, Amir M. Ghalamzan E.*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21472](https://doi.org/10.1609/aaai.v36i11.21472)

**Abstract**:

Breast cancer is the most common type of cancer worldwide. A robotic system performing autonomous breast palpation can make a significant impact on the related health sector worldwide. However, robot programming for breast palpating with different geometries is very complex and unsolved. Robot learning from demonstrations (LfD) reduces the programming time and cost. However, the available LfD are lacking the modelling of the manipulation path/trajectory as an explicit function of the visual sensory information. This paper presents a novel approach to manipulation path/trajectory planning called deep Movement Primitives that successfully generates the movements of a manipulator to reach a breast phantom and perform the palpation. We show the effectiveness of our approach by a series of real-robot experiments of reaching and palpating a breast phantom. The experimental results indicate our approach outperforms the state-of-the-art method.

----

## [1357] Multi-Agent Reinforcement Learning Controller to Maximize Energy Efficiency for Multi-Generator Industrial Wave Energy Converter

**Authors**: *Soumyendu Sarkar, Vineet Gundecha, Alexander Shmakov, Sahand Ghorbanpour, Ashwin Ramesh Babu, Paolo Faraboschi, Mathieu Cocho, Alexandre Pichard, Jonathan Fievez*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21473](https://doi.org/10.1609/aaai.v36i11.21473)

**Abstract**:

Waves in the oceans are one of the most significant renewable energy sources and are an excellent resource to tackle climate challenges through decarbonizing energy generation. Lowering the Levelized Cost of Energy (LCOE) for energy generation from ocean waves is critical for competitiveness with other forms of clean energy like wind and solar. It requires complex controllers to maximize efficiency for state-of-the-art multi-generator industrial Wave Energy Converters (WEC), which optimizes the reactive forces of the generators on multiple legs of WEC. This paper introduces Multi-Agent Reinforcement Learning controller (MARL) architectures that can handle these various objectives for LCOE. MARL can help increase energy capture efficiency to boost revenue, reduce structural stress to limit maintenance cost, and adaptively and proactively protect the wave energy converter from catastrophic weather events preserving investments and lowering effective capital cost. These architectures include 2-agent and 3-agent MARL implementing proximal policy optimization (PPO) with various optimizations to help sustain the training convergence in the complex hyperplane without falling off the cliff. Also, the design for trust assures the operation of WEC within a safe zone of mechanical compliance. As a part of this design, reward shaping for multiple objectives of energy capture and penalty for harmful motions minimizes stress and lowers the cost of maintenance. We achieved double-digit gains in energy capture efficiency across the waves of different principal frequencies over the baseline Spring Damper controller with the proposed MARL controllers.

----

## [1358] Reducing Energy Consumption of Pressure Sensor Calibration Using Polynomial HyperNetworks with Fourier Features

**Authors**: *Muhammad Sarmad, Mishal Fatima, Jawad Tayyub*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21474](https://doi.org/10.1609/aaai.v36i11.21474)

**Abstract**:

Our research aims to reduce the cost of pressure sensor calibration through machine learning. Pressure sensor calibration is a standard process whereby freshly manufactured pressure sensors are subjected to various controlled temperature and pressure setpoints to compute a mapping between the sensor's output and true pressure. Traditionally this mapping is calculated by fitting a polynomial with calibration data. Obtaining this data is costly since a large spectrum of temperature and pressure setpoints are required to model the sensor's behavior.
 
 We present a machine learning approach to predict a pre-defined calibration polynomial's parameters while requiring only one-third of the calibration data. Our method learns a pattern from past calibration sessions to predict the calibration polynomial's parameters from partial calibration setpoints for any newly manufactured sensor. We design a novel polynomial hypernetwork coupled with Fourier features and a weighted loss to solve this problem. We perform extensive evaluations and show that the current industry-standard method fails under similar conditions. In contrast, our approach saves two-thirds of the calibration time and cost. Furthermore, we conduct comprehensive ablations to study the effect of Fourier mapping and weighted loss. Code and a novel calibration dataset validated by calibration engineers are also made public.

----

## [1359] Bandit Data-Driven Optimization for Crowdsourcing Food Rescue Platforms

**Authors**: *Zheyuan Ryan Shi, Zhiwei Steven Wu, Rayid Ghani, Fei Fang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21475](https://doi.org/10.1609/aaai.v36i11.21475)

**Abstract**:

Food waste and insecurity are two societal challenges that coexist in many parts of the world. A prominent force to combat these issues, food rescue platforms match food donations to organizations that serve underprivileged communities, and then rely on external volunteers to transport the food. Previous work has developed machine learning models for food rescue volunteer engagement. However, having long worked with domain practitioners to deploy AI tools to help with food rescues, we understand that there are four main pain points that keep such a machine learning model from being actually useful in practice: small data, data collected only under the default intervention, unmodeled objectives due to communication gap, and unforeseen consequences of the intervention. In this paper, we introduce bandit data-driven optimization which not only helps address these pain points in food rescue, but also is applicable to other nonprofit domains that share similar challenges. Bandit data-driven optimization combines the advantages of online bandit learning and offline predictive analytics in an integrated framework. We propose PROOF, a novel algorithm for this framework and formally prove that it has no-regret. We show that PROOF performs better than existing baseline on food rescue volunteer recommendation.

----

## [1360] Sentiment and Emotion-Aware Multi-Modal Complaint Identification

**Authors**: *Apoorva Singh, Soumyodeep Dey, Anamitra Singha, Sriparna Saha*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21476](https://doi.org/10.1609/aaai.v36i11.21476)

**Abstract**:

The expression of displeasure on a consumer's behalf towards an organization, product, or event is denoted via the speech act known as complaint. Customers typically post reviews on retail websites and various social media platforms about the products or services they purchase, and the reviews may include complaints about the products or services. Automatic detection of consumers' complaints about items or services they buy can be critical for organizations and online merchants since they can use this insight to meet the customers' requirements, including handling and addressing the complaints. Previous studies on Complaint Identification (CI) are limited to text. Images posted with the reviews can provide cues to identify complaints better, thus emphasizing the importance of incorporating multi-modal inputs into the process. Furthermore, the customer's emotional state significantly impacts the complaint expression since emotions generally influence any speech act. As a result, the impact of emotion and sentiment on automatic complaint identification must also be investigated. One of the major contributions of this work is the creation of a new dataset- Complaint, Emotion, and Sentiment Annotated Multi-modal Amazon Reviews Dataset (CESAMARD), a collection of opinionated texts (reviews) and images of the products posted on the website of the retail giant Amazon. We present an attention-based multi-modal, adversarial multi-task deep neural network model for complaint detection to demonstrate the utility of the multi-modal dataset. Experimental results indicate that the multi-modality and multi-tasking complaint identification outperforms uni-modal and single-task variants.

----

## [1361] Sentence Simplification Capabilities of Transfer-Based Models

**Authors**: *Sanja Stajner, Kim Cheng Sheang, Horacio Saggion*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21477](https://doi.org/10.1609/aaai.v36i11.21477)

**Abstract**:

According to the official adult literacy report conducted in 24 highly-developed countries, more than 50% adults, on average, can only understand basic vocabulary, short sentences, and basic syntactic constructions. Everyday information found in news articles is thus inaccessible to many people, impeding their social inclusion and informed decision-making. Systems for automatic sentence simplification aim to provide scalable solution to this problem. In this paper, we propose new state-of-the-art sentence simplification systems for English and Spanish, and specifications for expert evaluation that are in accordance with well-established easy-to-read guidelines. We conduct expert evaluation of our new systems and the previous state-of-the-art systems for English and Spanish, and discuss strengths and weaknesses of each of them. Finally, we draw conclusions about the capabilities of the state-of-the-art sentence simplification systems and give some directions for future research.

----

## [1362] TransBoost: A Boosting-Tree Kernel Transfer Learning Algorithm for Improving Financial Inclusion

**Authors**: *Yiheng Sun, Tian Lu, Cong Wang, Yuan Li, Huaiyu Fu, Jingran Dong, Yunjie Xu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21478](https://doi.org/10.1609/aaai.v36i11.21478)

**Abstract**:

The prosperity of mobile and financial technologies has bred and expanded various kinds of financial products to a broader scope of people, which contributes to financial inclusion. It brings non-trivial social benefits of diminishing financial inequality. However, the technical challenges in individual financial risk evaluation exacerbated by the unforeseen user characteristic distribution and limited credit history of new users, as well as the inexperience of newly-entered companies in handling complex data and obtaining accurate labels, impede further promotion of financial inclusion. To tackle these challenges, this paper develops a novel transfer learning algorithm (i.e., TransBoost) that combines the merits of tree-based models and kernel methods. The TransBoost is designed with a parallel tree structure and
 efficient weights updating mechanism with theoretical guarantee, which enables it to excel in tackling real-world data with high dimensional features and sparsity in O(n) time complexity. We conduct extensive experiments on two public datasets and a unique largescale dataset from Tencent Mobile Payment. The results show that the TransBoost outperforms other state-of-the-
 art benchmark transfer learning algorithms in terms of prediction accuracy with superior efficiency, demonstrate stronger robustness to data sparsity, and provide meaningful model interpretation. Besides, given a financial risk level, the TransBoost enables financial service providers to serve the largest number of users including those who would otherwise be excluded by other algorithms. That is, the TransBoost improves financial inclusion.

----

## [1363] CausalGNN: Causal-Based Graph Neural Networks for Spatio-Temporal Epidemic Forecasting

**Authors**: *Lijing Wang, Aniruddha Adiga, Jiangzhuo Chen, Adam Sadilek, Srinivasan Venkatramanan, Madhav V. Marathe*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21479](https://doi.org/10.1609/aaai.v36i11.21479)

**Abstract**:

Infectious disease forecasting has been a key focus in the recent past owing to the COVID-19 pandemic and has proved to be an important tool in controlling the pandemic. With the advent of reliable spatiotemporal data, graph neural network models have been able to successfully model the inter-relation between the cross-region signals to produce quality forecasts, but like most deep-learning models they do not explicitly incorporate the underlying causal mechanisms. In this work, we employ a causal mechanistic model to guide the learning of the graph embeddings and propose a novel learning framework -- Causal-based Graph Neural Network (CausalGNN) that learns spatiotemporal embedding in a latent space where graph input features and epidemiological context are combined via a mutually learning mechanism using graph-based non-linear transformations. We design an attention-based dynamic GNN module to capture spatial and temporal disease dynamics. A causal module is added to the framework to provide epidemiological context for node embedding via ordinary differential equations. Extensive experiments on forecasting daily new cases of COVID-19 at global, US state, and US county levels show that the proposed method outperforms a broad range of baselines. The learned model which incorporates epidemiological context organizes the embedding in an efficient way by keeping the parameter size small leading to robust and accurate forecasting performance across various datasets.

----

## [1364] PrEF: Probabilistic Electricity Forecasting via Copula-Augmented State Space Model

**Authors**: *Zhiyuan Wang, Xovee Xu, Goce Trajcevski, Kunpeng Zhang, Ting Zhong, Fan Zhou*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21480](https://doi.org/10.1609/aaai.v36i11.21480)

**Abstract**:

Electricity forecasting has important implications for the key decisions in modern electricity systems, ranging from power generation, transmission, distribution and so on. In the literature, traditional statistic approaches, machine-learning methods and deep learning (e.g., recurrent neural network) based models are utilized to model the trends and patterns in electricity time-series data. However, they are restricted either by their deterministic forms or by independence in probabilistic assumptions -- thereby neglecting the uncertainty or significant correlations between distributions of electricity data. Ignoring these, in turn, may yield error accumulation, especially when relying on historical data and aiming at multi-step prediction. To overcome these, we propose a novel method named Probabilistic Electricity Forecasting (PrEF) by proposing a non-linear neural state space model (SSM) and incorporating copula-augmented mechanism into that, which can learn uncertainty-dependencies knowledge and understand interactive relationships between various factors from large-scale electricity time-series data. Our method distinguishes itself from existing models by its traceable inference procedure and its capability of providing high-quality probabilistic distribution predictions. Extensive experiments on two real-world electricity datasets demonstrate that our method consistently outperforms the alternatives.

----

## [1365] Fairness by "Where": A Statistically-Robust and Model-Agnostic Bi-level Learning Framework

**Authors**: *Yiqun Xie, Erhu He, Xiaowei Jia, Weiye Chen, Sergii Skakun, Han Bao, Zhe Jiang, Rahul Ghosh, Praveen Ravirathinam*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21481](https://doi.org/10.1609/aaai.v36i11.21481)

**Abstract**:

Fairness related to locations (i.e., "where") is critical for the use of machine learning in a variety of societal domains involving spatial datasets (e.g., agriculture, disaster response, urban planning). 
Spatial biases incurred by learning, if left unattended, may cause or exacerbate unfair distribution of resources, social division, spatial disparity, etc. The goal of this work is to develop statistically-robust formulations and model-agnostic learning strategies to understand and promote spatial fairness. The problem is challenging as locations are often from continuous spaces with no well-defined categories (e.g., gender), and statistical conclusions from spatial data are fragile to changes in spatial partitionings and scales. Existing studies in fairness-driven learning have generated valuable insights related to non-spatial factors including race, gender, education level, etc., but research to mitigate location-related biases still remains in its infancy, leaving the main challenges unaddressed. To bridge the gap, we first propose a robust space-as-distribution (SPAD) representation of spatial fairness to reduce statistical sensitivity related to partitioning and scales in continuous space. Furthermore, we propose a new SPAD-based stochastic strategy to efficiently optimize over an extensive distribution of fairness criteria, and a bi-level training framework to enforce fairness via adaptive adjustment of priorities among locations. Experiments on real-world crop monitoring show that SPAD can effectively reduce sensitivity in fairness evaluation and the stochastic bi-level training framework can greatly improve the fairness.

----

## [1366] DRAG: Dynamic Region-Aware GCN for Privacy-Leaking Image Detection

**Authors**: *Guang Yang, Juan Cao, Qiang Sheng, Peng Qi, Xirong Li, Jintao Li*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21482](https://doi.org/10.1609/aaai.v36i11.21482)

**Abstract**:

The daily practice of sharing images on social media raises a severe issue about privacy leakage. To address the issue, privacy-leaking image detection is studied recently, with the goal to automatically identify images that may leak privacy. Recent advance on this task benefits from focusing on crucial objects via pretrained object detectors and modeling their correlation. However, these methods have two limitations: 1) they neglect other important elements like scenes, textures, and objects beyond the capacity of pretrained object detectors. 2) the correlation among objects is fixed, but a fixed correlation is not appropriate for all the images. To overcome the limitations, we propose the Dynamic Region-Aware Graph Convolutional Network (DRAG) that dynamically finds out crucial regions including objects and other important elements, and model their correlation adaptively for each input image. To find out crucial regions, we cluster spatially-correlated feature channels into several region-aware feature maps. Furthermore, we dynamically model the correlation with the self-attention mechanism and explore the interaction among the regions with a graph convolutional network. The DRAG achieved an accuracy of 87% on the largest dataset for privacy-leaking image detection, which is 10 percentage points higher than the state of the art. The further case study demonstrates that it found out crucial regions containing not only objects but other important elements like textures. The code and more details are in https://github.com/guang-yanng/DRAG.

----

## [1367] D-vlog: Multimodal Vlog Dataset for Depression Detection

**Authors**: *Jeewoo Yoon, Chaewon Kang, Seungbae Kim, Jinyoung Han*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21483](https://doi.org/10.1609/aaai.v36i11.21483)

**Abstract**:

Detecting depression based on non-verbal behaviors has received great attention. However, most prior work on detecting depression mainly focused on detecting depressed individuals in laboratory settings, which are difficult to be generalized in practice. In addition, little attention has been paid to analyzing the non-verbal behaviors of depressed individuals in the wild. Therefore, in this paper, we present a multimodal depression dataset, D-Vlog, which consists of 961 vlogs (i.e., around 160 hours) collected from YouTube, which can be utilized in developing depression detection models based on the non-verbal behavior of individuals in real-world scenario. We develop a multimodal deep learning model that uses acoustic and visual features extracted from collected data to detect depression. Our proposed model employs the cross-attention mechanism to effectively capture the relationship across acoustic and visual features, and generates useful multimodal representations for depression detection. The extensive experimental results demonstrate that the proposed model significantly outperforms other baseline models. We believe our dataset and the proposed model are useful for analyzing and detecting depressed individuals based on non-verbal behavior.

----

## [1368] Longitudinal Fairness with Censorship

**Authors**: *Wenbin Zhang, Jeremy C. Weiss*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21484](https://doi.org/10.1609/aaai.v36i11.21484)

**Abstract**:

Recent works in artificial intelligence fairness attempt to mitigate discrimination by proposing constrained optimization programs that achieve parity for some fairness statistic. Most assume availability of the class label, which is impractical in many real-world applications such as precision medicine, actuarial analysis and recidivism prediction. Here we consider fairness in longitudinal right-censored environments, where the time to event might be unknown, resulting in censorship of the class label and inapplicability of existing fairness studies. We devise applicable fairness measures, propose a debiasing algorithm, and provide necessary theoretical constructs to bridge fairness with and without censorship for these important and socially-sensitive tasks. Our experiments on four censored datasets confirm the utility of our approach.

----

## [1369] Toward a New Science of Common Sense

**Authors**: *Ronald J. Brachman, Hector J. Levesque*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21485](https://doi.org/10.1609/aaai.v36i11.21485)

**Abstract**:

Common sense has always been of interest in AI, but has rarely taken center stage.  Despite its mention in one of John McCarthy's earliest papers and years of work by dedicated researchers, arguably no AI system with a serious amount of general common sense has ever emerged.  Why is that?  What's missing?  Examples of AI systems' failures of common sense abound, and they point to AI's frequent focus on expertise as the cause.  Those attempting to break the brittleness barrier, even in the context of modern deep learning, have tended to invest their energy in large numbers of small bits of commonsense knowledge.  But all the commonsense knowledge fragments in the world don't add up to a system that actually demonstrates common sense in a human-like way.  We advocate examining common sense from a broader perspective than in the past.  Common sense is more complex than it has been taken to be and is worthy of its own scientific exploration.

----

## [1370] Local Justice and the Algorithmic Allocation of Scarce Societal Resources

**Authors**: *Sanmay Das*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21486](https://doi.org/10.1609/aaai.v36i11.21486)

**Abstract**:

AI is increasingly used to aid decision-making about the allocation of scarce societal resources, for example housing for homeless people, organs for transplantation, and food donations. Recently, there have been several proposals for how to design objectives for these systems that attempt to achieve some combination of fairness, efficiency, incentive compatibility, and satisfactory aggregation of stakeholder preferences. This paper lays out possible roles and opportunities for AI in this domain, arguing for a closer engagement with the political philosophy literature on local justice, which provides a framework for thinking about how societies have over time framed objectives for such allocation problems. It also discusses how we may be able to integrate into this framework the opportunities and risks opened up by the ubiquity of data and the availability of algorithms that can use them to make accurate predictions about the future.

----

## [1371] Training on the Test Set: Mapping the System-Problem Space in AI

**Authors**: *José Hernández-Orallo, Wout Schellaert, Fernando Martínez-Plumed*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21487](https://doi.org/10.1609/aaai.v36i11.21487)

**Abstract**:

Many present and future problems associated with artificial intelligence are not due to its limitations, but to our poor assessment of its behaviour. Our evaluation procedures produce aggregated performance metrics that lack detail and quantified uncertainty about the following question: how will an AI system, with a particular profile \pi, behave for a new problem, characterised by a particular situation \mu? Instead of just aggregating test results, we can use machine learning methods to fully capitalise on this evaluation information. In this paper, we introduce the concept of an assessor model, \hat{R}(r|\pi,\mu), a conditional probability estimator trained on test data. We discuss how these assessors can be built by using information of the full system-problem space and illustrate a broad range of applications that derive from varied inferences and aggregations from \hat{R}. Building good assessor models will change the predictive and explanatory power of AI evaluation and will lead to new research directions for building and using them. We propose accompanying every deployed AI system with its own assessor.

----

## [1372] Symbols as a Lingua Franca for Bridging Human-AI Chasm for Explainable and Advisable AI Systems

**Authors**: *Subbarao Kambhampati, Sarath Sreedharan, Mudit Verma, Yantian Zha, Lin Guan*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21488](https://doi.org/10.1609/aaai.v36i11.21488)

**Abstract**:

Despite the surprising power of many modern AI systems that often learn their own representations, there is significant discontent about their inscrutability and the attendant problems in their ability to interact with humans. While alternatives such as  neuro-symbolic approaches have been proposed, there is a lack of consensus on what they are about. There are often two independent motivations (i) symbols as a lingua franca for human-AI interaction and (ii) symbols as (system-produced) abstractions use in its internal reasoning. The jury is still out on whether AI systems will need to use symbols in their internal reasoning to achieve general intelligence capabilities. Whatever the answer there is, the need for (human-understandable) symbols in human-AI interaction seems quite compelling. Symbols, like emotions, may well not be sine qua non for intelligence per se, but they will be crucial for AI systems to interact with us humans--as we can neither turn off our emotions not get by without our symbols. In particular, in many human-designed domains, humans would be interested in providing explicit (symbolic) knowledge and advice--and expect machine explanations in kind. This alone requires AI systems to at least do their I/O in symbolic terms. In this blue sky paper, we argue this point of view, and discuss research directions that need to be pursued to allow for this type of human-AI interaction.

----

## [1373] The Computational Gauntlet of Human-Like Learning

**Authors**: *Pat Langley*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21489](https://doi.org/10.1609/aaai.v36i11.21489)

**Abstract**:

In this paper, I pose a major challenge for AI researchers: to develop
systems that learn in a human-like manner. I briefly review the
history of machine learning, noting that early work made close contact
with results from cognitive psychology but that this is no longer the
case. I identify seven characteristics of human behavior that, if
reproduced, would offer better ways to acquire expertise than
statistical induction over massive training sets. I illustrate these
points with two domains - mathematics and driving - where people
are effective learners and review systems that address them. In
closing, I suggest ways to encourage more research on human-like
learning.

----

## [1374] BabelNet Meaning Representation: A Fully Semantic Formalism to Overcome Language Barriers

**Authors**: *Roberto Navigli, Rexhina Blloshmi, Abelardo Carlos Martinez Lorenzo*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21490](https://doi.org/10.1609/aaai.v36i11.21490)

**Abstract**:

Conceptual representations of meaning have long been the general focus of Artificial Intelligence (AI) towards the fundamental goal of machine understanding, with innumerable efforts made in Knowledge Representation, Speech and Natural Language Processing, Computer Vision, inter alia. Even today, at the core of Natural Language Understanding lies the task of Semantic Parsing, the objective of which is to convert natural sentences into machine-readable representations. Through this paper, we aim to revamp the historical dream of AI, by putting forward a novel, all-embracing, fully semantic meaning representation, that goes beyond the many existing formalisms. Indeed, we tackle their key limits by fully abstracting text into meaning and introducing language-independent concepts and semantic relations, in order to obtain an interlingual representation. Our proposal aims to overcome the language barrier, and connect not only texts across languages, but also images, videos, speech and sound, and logical formulas, across many fields of AI.

----

## [1375] Expert-Informed, User-Centric Explanations for Machine Learning

**Authors**: *Michael J. Pazzani, Severine Soltani, Robert Kaufman, Samson Qian, Albert Hsiao*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21491](https://doi.org/10.1609/aaai.v36i11.21491)

**Abstract**:

We argue that the dominant approach to explainable AI for explaining image classification, annotating images with heatmaps, provides little value for users unfamiliar with deep learning. We argue that explainable AI for images should produce output like experts produce when communicating with one another, with apprentices, and with novices. We provide an expanded set of goals of explainable AI systems and propose a Turing Test for explainable AI.

----

## [1376] Subjective Attributes in Conversational Recommendation Systems: Challenges and Opportunities

**Authors**: *Filip Radlinski, Craig Boutilier, Deepak Ramachandran, Ivan Vendrov*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21492](https://doi.org/10.1609/aaai.v36i11.21492)

**Abstract**:

The ubiquity of recommender systems has increased the need for higher-bandwidth, natural and efficient communication with users. This need is increasingly filled by recommenders that support natural language interaction, often conversationally. Given the inherent semantic subjectivity present in natural language, we argue that modeling subjective attributes in recommenders is a critical, yet understudied, avenue of AI research. We propose a novel framework for understanding different forms of subjectivity, examine various recommender tasks that will benefit from a systematic treatment of subjective attributes, and outline a number of research challenges.

----

## [1377] Market Design for Drone Traffic Management

**Authors**: *Sven Seuken, Paul Friedrich, Ludwig Dierks*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21493](https://doi.org/10.1609/aaai.v36i11.21493)

**Abstract**:

The rapid development of drone technology is leading to more and more use cases being proposed. In response, regulators are drawing up drone traffic management frameworks. However, to design solutions that are efficient, fair, simple, non-manipulable, and scalable, we need market design and AI expertise. To this end, we introduce the drone traffic management problem as a new research challenge to the market design and AI communities. We present five design desiderata that we have derived from our interviews with stakeholders from the regulatory side as well as from public and private enterprises. Finally, we provide an overview of the solution space to point out possible directions for future research.

----

## [1378] Consent as a Foundation for Responsible Autonomy

**Authors**: *Munindar P. Singh*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21494](https://doi.org/10.1609/aaai.v36i11.21494)

**Abstract**:

This paper focuses on a dynamic aspect of responsible autonomy, namely, to make intelligent agents be responsible at run time. That is, it considers settings where decision making by agents impinges upon the outcomes perceived by other agents. For an agent to act responsibly, it must accommodate the desires and other attitudes of its users and, through other agents, of their users.

The contribution of this paper is twofold. First, it provides a conceptual analysis of consent, its benefits and misuses, and how understanding consent can help achieve responsible autonomy. Second, it outlines challenges for AI (in particular, for agents and multiagent systems) that merit investigation to form as a basis for modeling consent in multiagent systems and applying consent to achieve responsible autonomy.

----

## [1379] Matching Market Design with Constraints

**Authors**: *Haris Aziz, Péter Biró, Makoto Yokoo*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21495](https://doi.org/10.1609/aaai.v36i11.21495)

**Abstract**:

Two-sided matching is an important research area that has had a major impact on the design of real-world matching markets. One consistent feature in many of the real-world applications is that they impose new feasibility constraints that lead to research challenges. We survey developments in the field of two-sided matching with various constraints, including those based on regions, diversity, multi-dimensional capacities, and matroids.

----

## [1380] Commonsense Knowledge Reasoning and Generation with Pre-trained Language Models: A Survey

**Authors**: *Prajjwal Bhargava, Vincent Ng*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21496](https://doi.org/10.1609/aaai.v36i11.21496)

**Abstract**:

While commonsense knowledge acquisition and reasoning has traditionally been a core research topic in the knowledge representation and reasoning community, recent years have seen a surge of interest in the natural language processing community in developing pre-trained models and testing their ability to address a variety of newly designed commonsense knowledge reasoning and generation tasks. This paper presents a survey of these tasks, discusses the strengths and weaknesses of state-of-the-art pre-trained models for commonsense reasoning and generation as revealed by these tasks, and reflects on future research directions.

----

## [1381] Target Languages (vs Inductive Biases) for Learning to Act and Plan

**Authors**: *Hector Geffner*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21497](https://doi.org/10.1609/aaai.v36i11.21497)

**Abstract**:

Recent  breakthroughs in AI have shown the remarkable power of deep learning and deep reinforcement learning. These developments, however, have been tied to specific tasks, and progress  in out-of-distribution generalization has been limited.  While it is assumed that these limitations can be overcome by incorporating suitable inductive biases, the notion of inductive biases itself is often left vague and does not provide meaningful guidance. In the paper, I articulate a different learning approach where representations do not emerge from  biases in a neural architecture  but are learned over a given  target language with  a known  semantics. The basic ideas are  implicit in mainstream AI where representations have been encoded in languages ranging from fragments of first-order logic to probabilistic structural causal models. The challenge is to learn from data the representations that have traditionally been crafted by hand. Generalization is then a  result of the semantics  of the language. The goals of this paper  are to make these ideas explicit, to place them in a broader context where the design of the target language is crucial, and to illustrate them in the context of learning to act and plan. For this, after a general discussion, I consider  learning representations of actions, general policies, and subgoals ("intrinsic rewards"). In  these cases, learning is formulated  as a combinatorial  problem  but nothing prevents the use of deep learning techniques instead. Indeed, learning representations over languages with a known semantics  provides an account of what is to be learned, while learning representations with neural nets   provides  a complementary account of how representations can be learned. The challenge and the opportunity is to bring the two  together.

----

## [1382] Model-Based Diagnosis of Multi-Agent Systems: A Survey

**Authors**: *Meir Kalech, Avraham Natan*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21498](https://doi.org/10.1609/aaai.v36i11.21498)

**Abstract**:

As systems involving multiple agents are increasingly deployed, there is a growing need to diagnose failures in such systems. Model-Based Diagnosis (MBD) is a well-known AI technique to diagnose faults in systems. In this approach, a model of the diagnosed system is given, and the real system is observed. A failure is announced when the real system's output contradicts the model's expected output. The model is then used to deduce the defective components that explain the unexpected observation. MBD has been increasingly being deployed in distributed and multi-agent systems. In this survey, we summarize twenty years of research in the field of model-based diagnosis algorithms for MAS diagnosis. We depict three attributes that should be considered when examining MAS diagnosis: (1) The objective of the diagnosis. Either diagnosing faults in the MAS plans or diagnosing coordination faults. (2) Centralized vs. distributed. The diagnosis method could be applied either by a centralized agent or by the agents in a distributed manner. (3) Temporal vs. non-temporal. Temporal diagnosis is used to diagnose the MAS's temporal behaviors, whereas non-temporal diagnosis is used to diagnose the conduct based on a single observation. We survey diverse studies in MBD of MAS based on these attributes, and provide novel research challenges in this field for the AI community.

----

## [1383] Delivering Trustworthy AI through Formal XAI

**Authors**: *João Marques-Silva, Alexey Ignatiev*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21499](https://doi.org/10.1609/aaai.v36i11.21499)

**Abstract**:

The deployment of systems of artificial intelligence (AI) in high-risk settings warrants the need for trustworthy AI. This crucial requirement is highlighted by recent EU guidelines and regulations, but also by recommendations from OECD and UNESCO, among several other examples. One critical premise of trustworthy AI involves the necessity of finding explanations that offer reliable guarantees of soundness. This paper argues that the best known eXplainable AI (XAI) approaches fail to provide sound explanations, or that alternatively find explanations which can exhibit significant redundancy. The solution to these drawbacks are explanation approaches that offer formal guarantees of rigor. These formal explanations are not only sound but guarantee irredundancy. This paper summarizes the recent developments in the emerging discipline of formal XAI. The paper also outlines existing challenges for formal XAI.

----

## [1384] Anatomizing Bias in Facial Analysis

**Authors**: *Richa Singh, Puspita Majumdar, Surbhi Mittal, Mayank Vatsa*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21500](https://doi.org/10.1609/aaai.v36i11.21500)

**Abstract**:

Existing facial analysis systems have been shown to yield biased results against certain demographic subgroups. Due to its impact on society, it has become imperative to ensure that these systems do not discriminate based on gender, identity, or skin tone of individuals. This has led to research in the identification and mitigation of bias in AI systems. In this paper, we encapsulate bias detection/estimation and mitigation algorithms for facial analysis. Our main contributions include a systematic review of algorithms proposed for understanding bias, along with a taxonomy and extensive overview of existing bias mitigation algorithms. We also discuss open challenges in the field of biased facial analysis.

----

## [1385] Intelligent Online Selling Point Extraction for E-commerce Recommendation

**Authors**: *Xiaojie Guo, Shugen Wang, Hanqing Zhao, Shiliang Diao, Jiajia Chen, Zhuoye Ding, Zhen He, Jianchao Lu, Yun Xiao, Bo Long, Han Yu, Lingfei Wu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21501](https://doi.org/10.1609/aaai.v36i11.21501)

**Abstract**:

In the past decade, automatic product description generation for e-commerce have witnessed significant advancement. As the services provided by e-commerce platforms become diverse, it is necessary to dynamically adapt the patterns of descriptions generated. The selling point of products is an important type of product description for which the length should be as short as possible while still conveying key information. In addition, this kind of product description should be eye-catching to the readers. Currently, product selling points are normally written by human experts. Thus, the creation and maintenance of these contents incur high costs. These costs can be significantly reduced if product selling points can be automatically generated by machines. In this paper, we report our experience developing and deploying the  Intelligent Online Selling Point Extraction (IOSPE) system to serve the recommendation system in the JD.com e-commerce platform. Since July 2020, IOSPE has become a core service for 62 key categories of products (covering more than 4 million products). So far, it has generated more than 1.1 billion selling points, thereby significantly scaling up the selling point creation operation and saving human labour. These IOSPE generated selling points have increased the click-through rate (CTR) by 1.89% and the average duration the customers spent on the products by more than 2.03% compared to the previous practice, which are significant improvements for such a large-scale e-commerce platform.

----

## [1386] Siamese BERT-Based Model for Web Search Relevance Ranking Evaluated on a New Czech Dataset

**Authors**: *Matej Kocián, Jakub Náplava, Daniel Stancl, Vladimír Kadlec*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21502](https://doi.org/10.1609/aaai.v36i11.21502)

**Abstract**:

Web search engines focus on serving highly relevant results within hundreds of milliseconds. Pre-trained language transformer models such as BERT are therefore hard to use in this scenario due to their high computational demands. We present our real-time approach to the document ranking problem leveraging a BERT-based siamese architecture. The model is already deployed in a commercial search engine and it improves production performance by more than 3%. For further research and evaluation, we release DaReCzech, a unique data set of 1.6 million Czech user query-document pairs with manually assigned relevance levels. We also release Small-E-Czech, an Electra-small language model pre-trained on a large Czech corpus. We believe this data will support endeavours both of search relevance and multilingual-focused research communities.

----

## [1387] Identifying Early Warning Signals from News Using Network Community Detection

**Authors**: *Nataliya Le Vine, Eric Boxer, Mustafa Dinani, Paolo Tortora, Subhradeep Das*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21503](https://doi.org/10.1609/aaai.v36i11.21503)

**Abstract**:

The paper addresses the challenge of accelerating identification of changes in risk drivers in the insurance industry. Specifically, the work presents a method to identify significant news events ("signals") from batches of news data to inform Life & Health insurance decisions. Signals are defined as events that are relevant to a tracked risk driver, widely discussed in multiple news outlets, contain novel information and affect stakeholders. The method converts unstructured data (news articles) into a sequence of keywords by employing a linguistic knowledge graph-based model. Then, for each time window, the method forms a graph with extracted keywords as nodes and draws weighted edges based on keyword co-occurrences in articles. Lastly, events are derived in an unsupervised way as graph communities and scored for the requirements of a signal: relevance, novelty and virality. The methodology is illustrated for a Life & Health topic using news articles from Dow Jones DNA proprietary data set, and assessed against baselines on a publicly available news data set. The method is implemented as an analytics engine in Early Warning System deployed at Swiss Re for the last 1.5 years to extract relevant events from live news data. We present the system's architectural design in production and discuss its use and impact.

----

## [1388] Prior-Guided Transfer Learning for Enhancing Item Representation in E-commerce

**Authors**: *Heng-Yi Li, Yabo Ni, Anxiang Zeng, Han Yu, Chunyan Miao*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21504](https://doi.org/10.1609/aaai.v36i11.21504)

**Abstract**:

Item representation learning is crucial for search and recommendation tasks in e-commerce. In e-commerce, the instances (e.g., items, users) in different domains are always related. Such instance relationship across domains contains useful local information for transfer learning. However, existing transfer learning based approaches did not leverage this knowledge. In this paper, we report on our experience designing and deploying Prior-Guided Transfer Learning (PGTL) to bridge this gap. It utilizes the instance relationship across domains to extract prior knowledge for the target domain and leverages it to guide the fine-grained transfer learning for e-commerce item representation learning tasks. Rather than directly transferring knowledge from the source domain to the target domain, the prior knowledge can serve as a bridge to link both domains and enhance knowledge transfer, especially when the domain distribution discrepancy is large. Since its deployment on the Taiwanese portal of Taobao in Aug 2020, PGTL has significantly improved the item exposure rate and item click-through rate compared to previous approaches

----

## [1389] Contribution-Aware Federated Learning for Smart Healthcare

**Authors**: *Zelei Liu, Yuanyuan Chen, Yansong Zhao, Han Yu, Yang Liu, Renyi Bao, Jinpeng Jiang, Zaiqing Nie, Qian Xu, Qiang Yang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21505](https://doi.org/10.1609/aaai.v36i11.21505)

**Abstract**:

Artificial intelligence (AI) is a promising technology to transform the healthcare industry. Due to the highly sensitive nature of patient data, federated learning (FL) is often leveraged to build models for smart healthcare applications. Existing deployed FL frameworks cannot address the key issues of varying data quality and heterogeneous data distributions across multiple institutions in this sector. In this paper, we report our experience developing and deploying the Contribution-Aware Federated Learning (CAFL) framework for smart healthcare. It provides an efficient and accurate approach to fairly evaluate FL participants' contribution to model performance without exposing their private data, and improves the FL model training protocol to allow the best performing intermediate models to be distributed to participants for FL training. Since its deployment in Yidu Cloud Technology Inc. in March 2021, CAFL has served 8 well-established medical institutions in China to build healthcare decision support models. It can perform contribution evaluations 2.84 times faster than the best existing approach, and has improved the average accuracy of the resulting models by 2.62% compared to the previous system (which is significant in industrial settings). To our knowledge, it is the first contribution-aware federated learning successfully deployed in the healthcare industry.

----

## [1390] AI Driven Accounts Payable Transformation

**Authors**: *Tarun Tater, Neelamadhav Gantayat, Sampath Dechu, Hussain Jagirdar, Harshit Rawat, Meena Guptha, Surbhi Gupta, Lukasz Strak, Shashi Kiran, Sivakumar Narayanan*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21506](https://doi.org/10.1609/aaai.v36i11.21506)

**Abstract**:

Accounts Payable (AP) is a resource-intensive business process in large enterprises for paying vendors within contractual payment deadlines for goods and services procured from them. There are multiple verifications before payment to the supplier/vendor. After the validations, the invoice flows through several steps such as vendor identification, line-item matching for Purchase order (PO) based invoices, Accounting Code identification for Non- Purchase order (Non-PO) based invoices, tax code identification, etc. Currently, each of these steps is mostly manual and cumbersome making it labor-intensive, error-prone, and requiring constant training of agents. Automatically processing these invoices for payment without any manual intervention is quite difficult. To tackle this challenge, we have developed an automated end-to-end invoice processing system using AI-based modules for multiple steps of the invoice processing pipeline. It can be configured to an individual client’s requirements with minimal effort. Currently, the system is deployed in production for two clients. It has successfully processed around ~80k invoices out of which 76% invoices were processed with low or no manual intervention.

----

## [1391] Harvest - a System for Creating Structured Rate Filing Data from Filing PDFs

**Authors**: *Ender Tekin, Qian You, Devin M. Conathan, Glenn Moo Fung, Thomas S. Kneubuehl*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21507](https://doi.org/10.1609/aaai.v36i11.21507)

**Abstract**:

We present a machine-learning-guided process that can efficiently extract factor tables from unstructured rate filing documents. Our approach combines multiple deep-learning-based models that work in tandem to create structured representations of tabular data present in unstructured documents such as pdf files. This process combines CNN's to detect tables, language-based models to extract table metadata and conventional computer vision techniques to improve the accuracy of tabular data on the machine-learning side. The extracted tabular data is validated through an intuitive user interface. This process, which we call Harvest, significantly reduces the time needed to extract tabular information from PDF files, enabling analysis of such data at a speed and scale that was previously unattainable.

----

## [1392] Automatic Product Copywriting for E-commerce

**Authors**: *Xueying Zhang, Yanyan Zou, Hainan Zhang, Jing Zhou, Shiliang Diao, Jiajia Chen, Zhuoye Ding, Zhen He, Xueqi He, Yun Xiao, Bo Long, Han Yu, Lingfei Wu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21508](https://doi.org/10.1609/aaai.v36i11.21508)

**Abstract**:

Product copywriting is a critical component of e-commerce recommendation platforms. It aims to attract users' interest and improve user experience by highlighting product characteristics with textual descriptions. In this paper, we report our experience deploying the proposed Automatic Product Copywriting Generation (APCG) system into the JD.com e-commerce product recommendation platform. It consists of two main components: 1) natural language generation, which is built from a transformer-pointer network and a pre-trained sequence-to-sequence model based on millions of training data from our in-house platform; and 2) copywriting quality control, which is based on both automatic evaluation and human screening. For selected domains, the models are trained and updated daily with the updated training data. In addition, the model is also used as a real-time writing assistant tool on our live broadcast platform. The APCG system has been deployed in JD.com since Feb 2021. By Sep 2021, it has generated 2.53 million product descriptions, and improved the overall averaged click-through rate (CTR) and the Conversion Rate (CVR) by 4.22% and 3.61%, compared to baselines, respectively on a year-on-year basis. The accumulated Gross Merchandise Volume (GMV) made by our system is improved by 213.42%, compared to the number in Feb 2021.

----

## [1393] Wasserstein Adversarial Transformer for Cloud Workload Prediction

**Authors**: *Shivani Arbat, Vinodh Kumaran Jayakumar, Jaewoo Lee, Wei Wang, In Kee Kim*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21509](https://doi.org/10.1609/aaai.v36i11.21509)

**Abstract**:

Predictive VM (Virtual Machine) auto-scaling is a promising technique to optimize cloud applications’ operating costs and performance. Understanding the job arrival rate is crucial for accurately predicting future changes in cloud workloads and proactively provisioning and de-provisioning VMs for hosting the applications. However, developing a model that accurately predicts cloud workload changes is extremely challenging due to the dynamic nature of cloud workloads. Long- Short-Term-Memory (LSTM) models have been developed for cloud workload prediction. Unfortunately, the state-of-the-art LSTM model leverages recurrences to predict, which naturally adds complexity and increases the inference overhead as input sequences grow longer. To develop a cloud workload prediction model with high accuracy and low inference overhead, this work presents a novel time-series forecasting model called WGAN-gp Transformer, inspired by the Transformer network and improved Wasserstein-GANs. The proposed method adopts a Transformer network as a generator and a multi-layer perceptron as a critic. The extensive evaluations with real-world workload traces show WGAN- gp Transformer achieves 5× faster inference time with up to 5.1% higher prediction accuracy against the state-of-the-art. We also apply WGAN-gp Transformer to auto-scaling mechanisms on Google cloud platforms, and the WGAN-gp Transformer-based auto-scaling mechanism outperforms the LSTM-based mechanism by significantly reducing VM over-provisioning and under-provisioning rates.

----

## [1394] Picking Pearl from Seabed: Extracting Artefacts from Noisy Issue Triaging Collaborative Conversations for Hybrid Cloud Services

**Authors**: *Amar Prakash Azad, Supriyo Ghosh, Ajay Gupta, Harshit Kumar, Prateeti Mohapatra, Lena Eckstein, Leonard Posner, Robert Kern*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21510](https://doi.org/10.1609/aaai.v36i11.21510)

**Abstract**:

Site Reliability Engineers (SREs) play a key role in identifying the cause of an issue and preforming remediation steps to resolve it. After an issue is reported, SREs come together in a virtual room (collaboration platform) to triage the issue. While doing so, they leave behind a wealth of information, in the form of conversations, which can be used later for triaging similar issues. However, usability of these conversations offer challenges due to them being and scarcity of conversation utterance label. This paper presents a novel approach for issue artefact extraction from noisy conversations with minimal labelled data. We propose a combination of unsupervised and supervised models with minimal human intervention that leverages domain knowledge to predict artefacts for a small amount of conversation data and use that for fine-tuning an already pre-trained language model for artefact prediction on a large amount of conversation data. Experimental results on our dataset show that the proposed ensemble of the unsupervised and supervised models is better than using either one of them individually. We also present a deployment case study of the proposed artefact prediction.

----

## [1395] Latent Space Simulation for Carbon Capture Design Optimization

**Authors**: *Brian R. Bartoldson, Rui Wang, Yucheng Fu, David Widemann, Sam Nguyen, Jie Bao, Zhijie Xu, Brenda Ng*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21511](https://doi.org/10.1609/aaai.v36i11.21511)

**Abstract**:

The CO2 capture efficiency in solvent-based carbon capture systems (CCSs) critically depends on the gas-solvent interfacial area (IA), making maximization of IA a foundational challenge in CCS design. While the IA associated with a particular CCS design can be estimated via a computational fluid dynamics (CFD) simulation, using CFD to derive the IAs associated with numerous CCS designs is prohibitively costly. Fortunately, previous works such as Deep Fluids (DF) (Kim et al., 2019) show that large simulation speedups are achievable by replacing CFD simulators with neural network (NN) surrogates that faithfully mimic the CFD simulation process. This raises the possibility of a fast, accurate replacement for a CFD simulator and therefore efficient approximation of the IAs required by CCS design optimization. Thus, here, we build on the DF approach to develop surrogates that can successfully be applied to our complex carbon-capture CFD simulations. Our optimized DF-style surrogates produce large speedups (4000x) while obtaining IA relative errors as low as 4% on unseen CCS configurations that lie within the range of training configurations. This hints at the promise of NN surrogates for our CCS design optimization problem. Nonetheless, DF has inherent limitations with respect to CCS design (e.g., limited transferability of trained models to new CCS packings). We conclude with ideas to address these challenges.

----

## [1396] Micronutrient Deficiency Prediction via Publicly Available Satellite Data

**Authors**: *Elizabeth Bondi, Haipeng Chen, Christopher D. Golden, Nikhil Behari, Milind Tambe*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21512](https://doi.org/10.1609/aaai.v36i11.21512)

**Abstract**:

Micronutrient deficiency (MND), which is a form of malnutrition that can have serious health consequences, is difficult to diagnose in early stages without blood draws, which are expensive and time-consuming to collect and process. It is even more difficult at a public health scale seeking to identify regions at higher risk of MND. To provide data more widely and frequently, we propose an accurate, scalable, low-cost, and interpretable regional-level MND prediction system. Specifically, our work is the first to use satellite data, such as forest cover, weather, and presence of water, to predict deficiency of micronutrients such as iron, Vitamin B12, and Vitamin A, directly from their biomarkers. We use real-world, ground truth biomarker data collected from four different regions across Madagascar for training, and demonstrate that satellite data are viable for predicting regional-level MND, surprisingly exceeding the performance of baseline predictions based only on survey responses. Our method could be broadly applied to other countries where satellite data are available, and potentially create high societal impact if these predictions are used by policy makers, public health officials, or healthcare providers.

----

## [1397] Using Public Data to Predict Demand for Mobile Health Clinics

**Authors**: *Haipeng Chen, Susobhan Ghosh, Gregory Fan, Nikhil Behari, Arpita Biswas, Mollie Williams, Nancy E. Oriol, Milind Tambe*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21513](https://doi.org/10.1609/aaai.v36i11.21513)

**Abstract**:

Improving health equity is an urgent task for our society. The advent of mobile clinics plays an important role in enhancing health equity, as they can provide easier access to preventive healthcare for patients from disadvantaged populations. For effective functioning of mobile clinics, accurate prediction of demand (expected number of individuals visiting mobile clinic) is the key to their daily operations and staff/resource allocation. Despite its importance, there have been very limited studies on predicting demand of mobile clinics. To the best of our knowledge, we are among the first to explore this area, using AI-based techniques. A crucial challenge in this task is that there are no known existing data sources from which we can extract useful information to account for the exogenous factors that may affect the demand, while considering protection of client privacy. We propose a novel methodology that completely uses public data sources to extract the features, with several new components that are designed to improve the prediction. Empirical evaluation on a real-world dataset from the mobile clinic The Family Van shows that, by leveraging publicly available data (which introduces no extra monetary cost to the mobile clinics), our AI-based method achieves 26.4% - 51.8% lower Root Mean Squared Error (RMSE) than the historical average-based estimation (which is presently employed by mobile clinics like The Family Van). Our algorithm makes it possible for mobile clinics to plan proactively, rather than reactively, as what has been doing.

----

## [1398] TCN: Pioneering Topological-Based Convolutional Networks for Planetary Terrain Learning

**Authors**: *Yuzhou Chen, Yuliya Marchetti, Elena Sizikova, Yulia R. Gel*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21514](https://doi.org/10.1609/aaai.v36i11.21514)

**Abstract**:

Implementations of artificial intelligence (AI) based on deep learning (DL) have proven to be highly successful in many domains, from biomedical imaging to natural language processing, but are still rarely applied in the space industry, particularly for onboard learning of planetary surfaces. In this project, we discuss the utility and limitations of DL, enhanced with topological footprints of the sensed objects, for multi-class classification of planetary surface patterns, in conjunction with tactile and embedded sensing in rover exploratory missions. We consider a Topological Convolutional Network (TCN) model with a persistence-based attention mechanism for supervised classification of various landforms. We study TCN's performance on the Barefoot surface pattern dataset, a novel surface pressure dataset from a prototype tactile rover wheel, known as the Barefoot Rover tactile wheel. Multi-class pattern recognition in the Barefoot data has neither been ever tackled before with DL nor assessed with topological methods. We provide insights into advantages and restrictions of topological DL as the early-stage concept for onboard learning and planetary exploration.

----

## [1399] CB+NN Ensemble to Improve Tracking Accuracy in Air Surveillance

**Authors**: *Anoop Karnik Dasika, Praveen Paruchuri*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21515](https://doi.org/10.1609/aaai.v36i11.21515)

**Abstract**:

Finding or tracking the location of an object accurately is a crucial problem in defense applications, robotics and computer vision. Radars fall into the spectrum of high-end defense sensors or systems upon which the security and surveillance of the entire world depends. There has been a lot of focus on the topic of Multi Sensor Tracking in recent years, with radars as the sensors. The Indian Air Force uses a Multi Sensor Tracking (MST) system to detect flights pan India, developed and supported by BEL(Bharat Electronics Limited), a defense agency we are working with. In this paper, we describe our Machine Learning approach, which is built on top of the existing system, the Air force uses. For purposes of this work, we trained our models on about 13 million anonymized real Multi Sensor tracking data points provided by radars performing tracking activity across the Indian air space. The approach has shown an increase in the accuracy of tracking by 5 percent from 91 to 96. The model and the corresponding code were transitioned to BEL, which has been tested in their simulation environment with a plan to take forward for ground testing. Our approach comprises of 3 steps: (a) We train a Neural Network model and a CatBoost model and ensemble them using a Logistic Regression model to predict one type of error, namely Splitting error, which can help to improve the accuracy of tracking. (b) We again train a Neural Network model and a CatBoost model and ensemble them using a different Logistic Regression model to predict the second type of error, namely Merging error, which can further improve the accuracy of tracking. (c) We use cosine similarity to find the nearest neighbour and correct the data points, predicted to have Splitting/Merging errors, by predicting the original global track of these data points.

----



[Go to the previous page](AAAI-2022-list06.md)

[Go to the next page](AAAI-2022-list08.md)

[Go to the catalog section](README.md)